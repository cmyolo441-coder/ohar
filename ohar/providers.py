"""OpenAI-compatible and Anthropic streaming providers. Stdlib only."""

from __future__ import annotations

import codecs
import contextlib
import http.client
import json
import socket
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from .types import ProviderConfig, ProviderError, StreamEvent

PROVIDER_TYPES = {"openai", "anthropic"}

_MAX_LINE = 2 * 1024 * 1024
_MAX_EVENT = 1024 * 1024
_MAX_ERROR_BODY = 64 * 1024
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal", "0.0.0.0"}
_CONTEXT_HINTS = (
    "context length",
    "context_length",
    "context window",
    "maximum context",
    "too many tokens",
    "token limit",
    "prompt is too long",
    "input is too long",
    "request too large",
    "reduce the length",
)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _normalize_usage(
    usage: dict[str, Any] | int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    *,
    input_tokens: int | None = None,
) -> dict[str, int]:
    if isinstance(usage, dict):
        input_tokens = _int_or_none(usage.get("input_tokens"))
        if input_tokens is None:
            input_tokens = _int_or_none(usage.get("prompt_tokens"))
        output_tokens = _int_or_none(usage.get("output_tokens"))
        if output_tokens is None:
            output_tokens = _int_or_none(usage.get("completion_tokens"))
        total_tokens = _int_or_none(usage.get("total_tokens"))
    elif isinstance(usage, int) and not isinstance(usage, bool):
        input_tokens = usage
    else:
        input_tokens = _int_or_none(input_tokens)

    result: dict[str, int] = {}
    if input_tokens is not None:
        result["input_tokens"] = input_tokens
    if output_tokens is not None:
        result["output_tokens"] = output_tokens
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    if total_tokens is not None:
        result["total_tokens"] = total_tokens
    return result


def _sse_lines(chunks: Iterable[bytes | str]) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buffer = ""
    for chunk in chunks:
        if isinstance(chunk, bytes):
            buffer += decoder.decode(chunk)
        else:
            buffer += str(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if len(line) > _MAX_LINE:
                raise ProviderError("SSE line too long", kind="protocol", retryable=True)
            yield line
        if len(buffer) > _MAX_LINE:
            raise ProviderError("SSE line too long", kind="protocol", retryable=True)
    buffer += decoder.decode(b"", final=True)
    if buffer:
        if len(buffer) > _MAX_LINE:
            raise ProviderError("SSE line too long", kind="protocol", retryable=True)
        yield buffer


def iter_sse(data_iterable: Iterable[bytes | str]) -> Iterator[tuple[str | None, str]]:
    event: str | None = None
    data: list[str] = []
    size = 0
    for line in _sse_lines(data_iterable):
        if line.endswith("\r"):
            line = line[:-1]
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data, size = None, [], 0
            continue
        if line.startswith(":"):
            continue
        field, sep, value = line.partition(":")
        if not sep:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            size += len(value) + 1
            if size > _MAX_EVENT:
                raise ProviderError("SSE event too large", kind="protocol", retryable=True)
            data.append(value)
    if data:
        yield event, "\n".join(data)


def parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, seconds)


def _redact(text: str, api_key: str | None) -> str:
    if api_key and api_key in text:
        text = text.replace(api_key, "***")
    return text


def _error_from_body(raw: bytes) -> tuple[str, str | None]:
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return "", None
    code: str | None = None
    message: Any = None
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text[:500], None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code") or error.get("type")
        elif isinstance(error, str):
            message = error
        else:
            message = payload.get("message")
            code = payload.get("code") or payload.get("type")
        if not isinstance(message, str) or not message.strip():
            detail = payload.get("detail")
            if isinstance(detail, dict):
                message = detail.get("message") or detail.get("msg")
            elif isinstance(detail, list) and detail:
                first = detail[0]
                if isinstance(first, dict):
                    message = first.get("msg") or first.get("message")
                elif isinstance(first, str):
                    message = first
        code = str(code) if isinstance(code, (str, int)) and not isinstance(code, bool) else None
        if isinstance(message, str) and message.strip():
            return message.strip()[:1000], code
    return text[:500], code


def _is_context_error(message: str, code: str | None) -> bool:
    haystack = f"{message} {code or ''}".lower()
    return any(hint in haystack for hint in _CONTEXT_HINTS)


def _http_error(exc: urllib.error.HTTPError, api_key: str | None) -> ProviderError:
    status = int(exc.code)
    try:
        raw = exc.read(_MAX_ERROR_BODY)
    except Exception:
        raw = b""
    finally:
        with contextlib.suppress(Exception):
            exc.close()
    message, code = _error_from_body(raw)
    reason = exc.reason if isinstance(exc.reason, str) else ""
    message = message or reason or f"HTTP {status} error"
    message = _redact(message, api_key)
    retry_after = parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)

    if 300 <= status < 400:
        kind, retryable = "redirect", False
    elif status in (401, 403):
        kind, retryable = "auth", False
    elif status == 400:
        kind = "context" if _is_context_error(message, code) else "bad_request"
        retryable = False
    elif status == 404:
        kind, retryable = "model", False
    elif status == 408:
        kind, retryable = "timeout", True
    elif status == 409:
        kind, retryable = "conflict", True
    elif status == 422:
        kind, retryable = "bad_request", False
    elif status == 429:
        kind, retryable = "rate_limit", True
    elif status >= 500:
        kind, retryable = "server", True
    elif 400 <= status < 500:
        kind, retryable = "bad_request", False
    else:
        kind, retryable = "unknown", False
    return ProviderError(
        message,
        status=status,
        retryable=retryable,
        retry_after=retry_after,
        code=code,
        kind=kind,
    )


def _network_error(exc: BaseException, api_key: str | None) -> ProviderError:
    reason = getattr(exc, "reason", None) or exc
    message = str(reason) or type(reason).__name__
    return ProviderError(_redact(f"Network error: {message}", api_key), retryable=True, kind="network")


def _payload_error(value: Any, api_key: str | None = None) -> ProviderError:
    message = ""
    code: str | None = None
    status: int | None = None
    if isinstance(value, dict):
        message = value.get("message") or value.get("detail") or ""
        code = value.get("code") or value.get("type")
        status = _int_or_none(value.get("status"))
    elif isinstance(value, str):
        message = value
    message = message if isinstance(message, str) and message else "Provider stream error"
    code = str(code) if isinstance(code, (str, int)) and not isinstance(code, bool) else None
    lowered = f"{code or ''} {message}".lower()
    if "context" in lowered and ("length" in lowered or "window" in lowered or "too long" in lowered):
        kind, retryable = "context", False
    elif "overloaded" in lowered:
        kind, retryable = "server", True
    elif "rate" in lowered and "limit" in lowered:
        kind, retryable = "rate_limit", True
    elif "auth" in lowered or "permission" in lowered or "api key" in lowered:
        kind, retryable = "auth", False
    elif "not_found" in lowered or "not found" in lowered or "model" in lowered:
        kind, retryable = "model", False
    elif "invalid" in lowered or "bad request" in lowered:
        kind, retryable = "bad_request", False
    elif "server" in lowered or "internal" in lowered or "timeout" in lowered:
        kind, retryable = "server", True
    else:
        kind, retryable = "server", False
    return ProviderError(
        _redact(message, api_key),
        status=status,
        retryable=retryable,
        code=code,
        kind=kind,
    )


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ProviderError("Request cancelled", kind="cancelled", retryable=False)


def _base_endpoint(base_url: str, default: str) -> str:
    return ((base_url or "").strip() or default).rstrip("/")


def _is_local_url(url: str) -> bool:
    try:
        host = urllib.parse.urlsplit(url).hostname
    except ValueError:
        return False
    return bool(host) and host.lower() in _LOCAL_HOSTS


def _request_headers(defaults: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    headers = dict(defaults)
    for key, value in (extra or {}).items():
        headers[str(key)] = str(value)
    return headers


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def _open_stream(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    config: ProviderConfig,
    api_key: str | None,
) -> Any:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    try:
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        opener = urllib.request.build_opener(_NoRedirect)
        return opener.open(request, timeout=config.timeout)
    except urllib.error.HTTPError as exc:
        raise _http_error(exc, api_key) from None
    except (urllib.error.URLError, ssl.SSLError, socket.timeout, OSError) as exc:
        raise _network_error(exc, api_key) from None
    except ValueError as exc:
        raise ProviderError(_redact(str(exc), api_key), retryable=False, kind="config") from None
    except http.client.HTTPException as exc:
        raise _network_error(exc, api_key) from None


def _read_lines(response: Any) -> Iterator[bytes]:
    while True:
        raw = response.readline(_MAX_LINE + 1)
        if not raw:
            return
        if len(raw) > _MAX_LINE:
            raise ProviderError("SSE line too long", kind="protocol", retryable=True)
        yield raw


def _tool_arguments(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "{}"
    try:
        json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return "{}"
    return raw


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(value)


def _as_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    text = _content_text(content)
    return [{"type": "text", "text": text}] if text else []


def _merge_content(left: Any, right: Any) -> Any:
    if isinstance(left, str) and isinstance(right, str):
        if not left:
            return right
        if not right:
            return left
        return f"{left}\n\n{right}"
    return [*_as_blocks(left), *_as_blocks(right)]


def _anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    tool_group: dict[str, Any] | None = None

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _content_text(content)
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": str(message.get("tool_call_id") or ""),
                "content": _content_text(content),
            }
            if tool_group is not None:
                tool_group["content"].append(block)
            else:
                tool_group = {"role": "user", "content": [block]}
                converted.append(tool_group)
            continue

        tool_group = None
        if role == "user":
            text = _content_text(content)
            if text:
                converted.append({"role": "user", "content": text})
            continue
        if role != "assistant":
            continue

        text = _content_text(content)
        blocks: list[dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        has_tools = False
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for position, call in enumerate(tool_calls, start=1):
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    function = {}
                name = function.get("name") or call.get("name")
                if not isinstance(name, str) or not name:
                    continue
                arguments = function.get("arguments", call.get("arguments", "{}"))
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments) if arguments.strip() else {}
                    except (json.JSONDecodeError, ValueError):
                        parsed = {}
                elif isinstance(arguments, dict):
                    parsed = arguments
                else:
                    parsed = {}
                if not isinstance(parsed, dict):
                    parsed = {}
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"toolu_{position}"
                blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": parsed})
                has_tools = True
        if not blocks:
            continue
        converted.append({"role": "assistant", "content": blocks if has_tools else text})

    merged: list[dict[str, Any]] = []
    for message in converted:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] = _merge_content(merged[-1]["content"], message["content"])
        else:
            merged.append(message)

    result: list[dict[str, Any]] = []
    for message in merged:
        content = message["content"]
        if not content:
            continue
        result.append({"role": message["role"], "content": content})
    return "\n\n".join(system_parts), result


def _anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            function = tool
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        schema = function.get("parameters")
        if not isinstance(schema, dict):
            schema = function.get("input_schema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        entry: dict[str, Any] = {"name": name, "input_schema": schema}
        description = function.get("description")
        if isinstance(description, str) and description:
            entry["description"] = description
        mapped.append(entry)
    return mapped


class Provider:
    kind = ""

    def __init__(self, config: ProviderConfig, api_key: str | None = None) -> None:
        self.config = config
        self.name = config.name
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.kind = type(self).kind or (config.type or "").strip().lower() or "unknown"

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        raise NotImplementedError


class OpenAIProvider(Provider):
    kind = "openai"
    _default_base = "https://api.openai.com/v1"

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        _check_cancel(cancel)
        key = self.api_key
        endpoint = f"{_base_endpoint(self.config.base_url, self._default_base)}/chat/completions"
        if not key and not _is_local_url(endpoint):
            raise ProviderError(
                f"Missing API key for provider '{self.name}'",
                retryable=False,
                kind="auth",
            )

        headers = _request_headers(
            {"Content-Type": "application/json", "Accept": "text/event-stream"},
            self.config.headers,
        )
        if key:
            for name in [name for name in headers if name.lower() == "authorization"]:
                del headers[name]
            headers["Authorization"] = f"Bearer {key}"

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        try:
            response = _open_stream(endpoint, headers, body, self.config, key)
        except ProviderError as exc:
            if exc.status == 400 and "stream_options" in f"{exc.message} {exc.code or ''}".lower():
                body.pop("stream_options", None)
                response = _open_stream(endpoint, headers, body, self.config, key)
            else:
                raise

        with contextlib.closing(response):
            yield from self._consume(response, key, cancel)

    def _consume(
        self,
        response: Any,
        api_key: str | None,
        cancel: threading.Event | None,
    ) -> Iterator[StreamEvent]:
        calls: dict[int, dict[str, str]] = {}
        usage: dict[str, int] | None = None
        finish_reason: str | None = None
        try:
            for _event, data in iter_sse(_read_lines(response)):
                _check_cancel(cancel)
                if not data:
                    continue
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except (json.JSONDecodeError, ValueError):
                    raise ProviderError("Malformed stream", kind="protocol", retryable=True) from None
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    raise _payload_error(chunk["error"], api_key)

                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    normalized = _normalize_usage(chunk_usage)
                    if normalized:
                        usage = normalized

                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield StreamEvent("text", text=content)
                    reasoning = delta.get("reasoning_content")
                    if not isinstance(reasoning, str) or not reasoning:
                        reasoning = delta.get("reasoning")
                    if isinstance(reasoning, str) and reasoning:
                        yield StreamEvent("reasoning", text=reasoning)
                    tool_deltas = delta.get("tool_calls")
                    if isinstance(tool_deltas, list):
                        for position, tool_delta in enumerate(tool_deltas):
                            if not isinstance(tool_delta, dict):
                                continue
                            index = tool_delta.get("index")
                            if not isinstance(index, int) or isinstance(index, bool):
                                index = position
                            state = calls.setdefault(
                                index, {"id": "", "name": "", "arguments": ""}
                            )
                            call_id = tool_delta.get("id")
                            if isinstance(call_id, str) and call_id:
                                state["id"] = call_id
                            function = tool_delta.get("function")
                            if not isinstance(function, dict):
                                continue
                            name = function.get("name")
                            if isinstance(name, str) and name:
                                state["name"] += name
                            arguments = function.get("arguments")
                            if isinstance(arguments, str) and arguments:
                                state["arguments"] += arguments
                reason = choice.get("finish_reason")
                if isinstance(reason, str) and reason:
                    finish_reason = reason
            _check_cancel(cancel)
            for index in sorted(calls):
                state = calls[index]
                yield StreamEvent(
                    "tool_call",
                    tool_call_id=state["id"] or None,
                    tool_name=state["name"] or None,
                    arguments=_tool_arguments(state["arguments"]),
                )
            if usage:
                yield StreamEvent("usage", usage=usage)
            yield StreamEvent("done", finish_reason=finish_reason)
        except ProviderError:
            raise
        except (
            urllib.error.URLError,
            ssl.SSLError,
            socket.timeout,
            OSError,
            http.client.HTTPException,
        ) as exc:
            raise _network_error(exc, api_key) from None


class AnthropicProvider(Provider):
    kind = "anthropic"
    _default_base = "https://api.anthropic.com/v1"

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        _check_cancel(cancel)
        key = self.api_key
        endpoint = f"{_base_endpoint(self.config.base_url, self._default_base)}/messages"
        if not key and not _is_local_url(endpoint):
            raise ProviderError(
                f"Missing API key for provider '{self.name}'",
                retryable=False,
                kind="auth",
            )

        headers = _request_headers(
            {
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "accept": "text/event-stream",
            },
            self.config.headers,
        )
        if key:
            for name in [name for name in headers if name.lower() == "x-api-key"]:
                del headers[name]
            headers["x-api-key"] = key

        system, converted = _anthropic_messages(messages)
        body: dict[str, Any] = {
            "model": model,
            "messages": converted,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "stream": True,
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            mapped = _anthropic_tools(tools)
            if mapped:
                body["tools"] = mapped

        response = _open_stream(endpoint, headers, body, self.config, key)
        with contextlib.closing(response):
            yield from self._consume(response, key, cancel)

    def _consume(
        self,
        response: Any,
        api_key: str | None,
        cancel: threading.Event | None,
    ) -> Iterator[StreamEvent]:
        blocks: dict[int, dict[str, Any]] = {}
        usage_data: dict[str, Any] = {}
        stop_reason: str | None = None
        done_sent = False
        try:
            for event_name, data in iter_sse(_read_lines(response)):
                _check_cancel(cancel)
                if not data:
                    continue
                if data.strip() == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except (json.JSONDecodeError, ValueError):
                    raise ProviderError("Malformed stream", kind="protocol", retryable=True) from None
                if not isinstance(payload, dict):
                    continue
                event = payload.get("type") if isinstance(payload.get("type"), str) else event_name

                if event == "error" or payload.get("error"):
                    raise _payload_error(payload.get("error") or payload, api_key)

                if event == "message_start":
                    message = payload.get("message")
                    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
                        for name in ("input_tokens", "output_tokens", "total_tokens"):
                            value = _int_or_none(message["usage"].get(name))
                            if value is not None:
                                usage_data[name] = value
                elif event == "content_block_start":
                    index = payload.get("index")
                    if not isinstance(index, int) or isinstance(index, bool):
                        index = 0
                    block = payload.get("content_block")
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "tool_use":
                        blocks[index] = {
                            "id": block.get("id") if isinstance(block.get("id"), str) else "",
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "json": "",
                            "input": block.get("input"),
                        }
                    elif block_type == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            yield StreamEvent("text", text=text)
                    elif block_type == "thinking":
                        thinking = block.get("thinking")
                        if isinstance(thinking, str) and thinking:
                            yield StreamEvent("reasoning", text=thinking)
                elif event == "content_block_delta":
                    delta = payload.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        text = delta.get("text")
                        if isinstance(text, str) and text:
                            yield StreamEvent("text", text=text)
                    elif delta_type == "thinking_delta":
                        thinking = delta.get("thinking")
                        if isinstance(thinking, str) and thinking:
                            yield StreamEvent("reasoning", text=thinking)
                    elif delta_type == "input_json_delta":
                        index = payload.get("index")
                        if not isinstance(index, int) or isinstance(index, bool):
                            index = 0
                        block = blocks.get(index)
                        partial = delta.get("partial_json")
                        if block is not None and isinstance(partial, str):
                            block["json"] += partial
                elif event == "content_block_stop":
                    index = payload.get("index")
                    if not isinstance(index, int) or isinstance(index, bool):
                        index = 0
                    block = blocks.pop(index, None)
                    if block is None or not block["name"]:
                        continue
                    arguments = _tool_arguments(block["json"])
                    if arguments == "{}" and isinstance(block.get("input"), dict) and block["input"]:
                        arguments = json.dumps(block["input"], ensure_ascii=False)
                    yield StreamEvent(
                        "tool_call",
                        tool_call_id=block["id"] or None,
                        tool_name=block["name"],
                        arguments=arguments,
                    )
                elif event == "message_delta":
                    delta = payload.get("delta")
                    if isinstance(delta, dict):
                        reason = delta.get("stop_reason")
                        if isinstance(reason, str) and reason:
                            stop_reason = reason
                    if isinstance(payload.get("usage"), dict):
                        for name in ("input_tokens", "output_tokens", "total_tokens"):
                            value = _int_or_none(payload["usage"].get(name))
                            if value is not None:
                                usage_data[name] = value
                elif event == "message_stop":
                    usage = _normalize_usage(usage_data)
                    if usage:
                        yield StreamEvent("usage", usage=usage)
                    yield StreamEvent("done", finish_reason=stop_reason)
                    done_sent = True
                    break
            if not done_sent:
                usage = _normalize_usage(usage_data)
                if usage:
                    yield StreamEvent("usage", usage=usage)
                yield StreamEvent("done", finish_reason=stop_reason)
        except ProviderError:
            raise
        except (
            urllib.error.URLError,
            ssl.SSLError,
            socket.timeout,
            OSError,
            http.client.HTTPException,
        ) as exc:
            raise _network_error(exc, api_key) from None


def create_provider(
    config: ProviderConfig, api_key: str | None
) -> OpenAIProvider | AnthropicProvider:
    kind = (config.type or "openai").strip().lower()
    if kind not in PROVIDER_TYPES:
        raise ProviderError(
            f"Unsupported provider type '{config.type}'",
            retryable=False,
            kind="config",
        )
    if kind == "anthropic":
        return AnthropicProvider(config, api_key)
    return OpenAIProvider(config, api_key)
