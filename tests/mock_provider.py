"""Offline OpenAI- and Anthropic-compatible mock provider for OHAR tests."""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

USAGE = {"prompt_tokens": 17, "completion_tokens": 9, "total_tokens": 26}
ANTHROPIC_USAGE = {"input_tokens": 17, "output_tokens": 9}
DONE = "[DONE]"


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _short(text: str, limit: int = 80) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return "done"
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            kind = part.get("type")
            if kind in (None, "text"):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif kind == "tool_result":
                nested = _content_text(part.get("content"))
                if nested:
                    parts.append(nested)
    return "\n".join(part for part in parts if part)


def _canned(text: str) -> str:
    return (
        "# Mock Response\n\n"
        "This is a canned markdown reply from the offline mock provider.\n\n"
        "- Bullet one\n"
        "- Inline code: `mock_provider`\n\n"
        "```python\n"
        "print(\"mock\")\n"
        "```\n\n"
        f"You said: {_short(text)}\n"
    )


def _final(text: str) -> str:
    return f"Mock final answer: {_short(text)}"


def _tool_marker(text: str) -> tuple[str, dict[str, Any]]:
    match = re.search(r"TOOL_NAME:\s*([A-Za-z0-9_.-]+)", text)
    name = match.group(1) if match else "read_file"
    if name == "shell":
        arguments = {"command": "printf mock"}
    elif name == "task":
        arguments = {
            "tasks": [
                {"title": f"mock research {index}", "prompt": f"inspect mock area {index}"}
                for index in range(8)
            ]
        }
    elif name == "update_plan":
        arguments = {
            "steps": [
                {"step": "inspect the request", "status": "completed"},
                {"step": "implement the change", "status": "in_progress"},
                {"step": "verify the result", "status": "pending"},
            ]
        }
    elif name == "remember":
        arguments = {
            "text": "Mock memory: the test suite uses the offline provider.",
            "tags": "mock, tests",
        }
    else:
        arguments = {"path": "README.md"}
    return name, arguments


def _plan(messages: list[Any]) -> tuple[str, str, str | None, dict[str, Any] | None]:
    last = messages[-1] if messages and isinstance(messages[-1], dict) else {}
    user_text = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            user_text = _content_text(message.get("content"))
            break
    result_text = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "tool":
            result_text = _content_text(message.get("content"))
            break
    text = "\n".join(part for part in (user_text, result_text) if part)
    if last.get("role") == "tool" or "FINAL" in text:
        return "final", _final(text), None, None
    if "USE_TOOL" in text:
        name, arguments = _tool_marker(text)
        return "tool", "", name, arguments
    return "text", _canned(text), None, None


def _has_tool_result(content: Any) -> bool:
    if isinstance(content, dict):
        return content.get("type") == "tool_result"
    if isinstance(content, list):
        return any(isinstance(part, dict) and part.get("type") == "tool_result" for part in content)
    return False


def _anthropic_plan(messages: list[Any]) -> tuple[str, str, str | None, dict[str, Any] | None]:
    text = ""
    has_result = False
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            text = _content_text(message.get("content"))
            has_result = _has_tool_result(message.get("content"))
            break
    if has_result or "FINAL" in text:
        return "final", _final(text), None, None
    if "USE_TOOL" in text:
        name, arguments = _tool_marker(text)
        return "tool", "", name, arguments
    return "text", _canned(text), None, None


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True) or [text]


class MockProvider:
    """Local mock of OpenAI chat completions and Anthropic messages."""

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.host = "127.0.0.1"
        self.port = 0
        self.delay_seconds = float(delay_seconds)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._requests: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._requests)

    @property
    def requests_count(self) -> int:
        with self._lock:
            return len(self._requests)

    @property
    def last_request(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._requests[-1]) if self._requests else None

    @property
    def headers_seen(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(request["headers"]) for request in self._requests]

    def url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        if path.startswith("/"):
            return f"http://{self.host}:{self.port}{path}"
        if path == "health" or path.startswith("v1/"):
            return f"http://{self.host}:{self.port}/{path}"
        return f"{self.base_url}/{path.lstrip('/')}"

    def start(self) -> "MockProvider":
        if self._server is not None:
            return self
        server = _MockServer((self.host, 0), self)
        self._server = server
        self.port = int(server.server_address[1])
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="mock-provider",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)

    def __enter__(self) -> "MockProvider":
        return self.start()

    def __exit__(self, *_exc: object) -> bool:
        self.stop()
        return False

    def _record(self, request: dict[str, Any]) -> None:
        with self._lock:
            self._requests.append(request)


class _MockServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], provider: MockProvider) -> None:
        self.provider = provider
        super().__init__(address, _MockHandler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        if not isinstance(sys.exc_info()[1], OSError):
            super().handle_error(request, client_address)


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _MockServer

    def log_message(self, _format: str, *args: object) -> None:
        return None

    def log_error(self, _format: str, *args: object) -> None:
        return None

    def do_GET(self) -> None:
        self.close_connection = True
        self._record_request(b"")
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        self.close_connection = True
        raw = self._read_body()
        request = self._record_request(raw)
        path = request["path"]
        body = request["body"]
        if not isinstance(body, dict):
            self._send_json(400, {"error": {"message": "invalid JSON body"}})
            return
        if path == "/v1/chat/completions":
            self._handle_openai(body)
        elif path == "/v1/messages":
            self._handle_anthropic(body)
        else:
            self._send_json(404, {"error": {"message": "not found"}})

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length > 0 else b""
        except (OSError, ValueError):
            return b""

    def _record_request(self, raw: bytes) -> dict[str, Any]:
        parsed = urlsplit(self.path)
        try:
            body: Any = json.loads(raw) if raw else {}
        except (UnicodeDecodeError, ValueError):
            body = None
        request = {
            "method": self.command,
            "path": parsed.path,
            "query": parse_qs(parsed.query),
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "raw_body": raw.decode("utf-8", "replace"),
            "body": body,
        }
        self.server.provider._record(request)
        return request

    def _send_json(self, status: int, payload: Any, extra_headers: dict[str, str] | None = None) -> None:
        body = _compact(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except OSError:
            pass
        finally:
            self.close_connection = True

    def _start_sse(self) -> bool:
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.flush()
            return True
        except OSError:
            return False

    def _send_event(self, data: str, event: str | None = None) -> bool:
        delay = self.server.provider.delay_seconds
        if delay > 0:
            time.sleep(delay)
        lines = [f"event: {event}"] if event else []
        lines.extend((f"data: {data}", "", ""))
        try:
            self.wfile.write("\n".join(lines).encode("utf-8"))
            self.wfile.flush()
            return True
        except OSError:
            return False

    def _send_chunk(self, model: str, delta: dict[str, Any], finish_reason: str | None = None) -> bool:
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return self._send_event(_compact(payload))

    def _handle_openai(self, body: dict[str, Any]) -> None:
        model = str(body.get("model") or "mock")
        lowered = model.lower()
        if "fail" in lowered:
            self._send_json(500, {"error": {"message": "mock failure"}})
            return
        if "ratelimit" in lowered:
            self._send_json(429, {"error": {"message": "mock rate limit"}}, {"Retry-After": "1"})
            return
        if "auth" in lowered:
            self._send_json(401, {"error": {"message": "mock unauthorized"}})
            return

        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        kind, text, tool_name, arguments = _plan(messages)
        if not body.get("stream"):
            message: dict[str, Any] = {"role": "assistant"}
            if kind == "tool":
                message["content"] = None
                message["tool_calls"] = [
                    {
                        "id": "call_mock_1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": _compact(arguments)},
                    }
                ]
                finish_reason = "tool_calls"
            else:
                message["content"] = text
                finish_reason = "stop"
            self._send_json(
                200,
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": USAGE,
                },
            )
            return

        if not self._start_sse():
            return
        if not self._send_chunk(model, {"role": "assistant"}):
            return
        if kind == "tool":
            tool_call = {
                "index": 0,
                "id": "call_mock_1",
                "type": "function",
                "function": {"name": tool_name, "arguments": ""},
            }
            if not self._send_chunk(model, {"tool_calls": [tool_call]}):
                return
            encoded = _compact(arguments)
            for start in range(0, len(encoded), 5):
                fragment = {"index": 0, "function": {"arguments": encoded[start : start + 5]}}
                if not self._send_chunk(model, {"tool_calls": [fragment]}):
                    return
            finish_reason = "tool_calls"
        else:
            for line in _lines(text):
                if not self._send_chunk(model, {"content": line}):
                    return
            finish_reason = "stop"
        if not self._send_chunk(model, {}, finish_reason):
            return
        usage_chunk = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": USAGE,
        }
        if not self._send_event(_compact(usage_chunk)):
            return
        self._send_event(DONE)

    def _handle_anthropic(self, body: dict[str, Any]) -> None:
        if not self.headers.get("x-api-key"):
            self._send_json(
                401,
                {"type": "error", "error": {"type": "authentication_error", "message": "missing x-api-key"}},
            )
            return
        model = str(body.get("model") or "claude-mock")
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        kind, text, tool_name, arguments = _anthropic_plan(messages)
        stop_reason = "tool_use" if kind == "tool" else "end_turn"
        if body.get("stream") is False:
            content: list[dict[str, Any]]
            if kind == "tool":
                content = [{"type": "tool_use", "id": "toolu_mock_1", "name": tool_name, "input": arguments}]
            else:
                content = [{"type": "text", "text": text}]
            self._send_json(
                200,
                {
                    "id": "msg_mock_1",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": content,
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": ANTHROPIC_USAGE,
                },
            )
            return

        if not self._start_sse():
            return
        start = {
            "type": "message_start",
            "message": {
                "id": "msg_mock_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 17, "output_tokens": 1},
            },
        }
        if not self._send_event(_compact(start), "message_start"):
            return
        if kind == "tool":
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_mock_1", "name": tool_name, "input": {}},
            }
            if not self._send_event(_compact(block_start), "content_block_start"):
                return
            encoded = _compact(arguments)
            for start_index in range(0, len(encoded), 5):
                delta = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": encoded[start_index : start_index + 5]},
                }
                if not self._send_event(_compact(delta), "content_block_delta"):
                    return
        else:
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            if not self._send_event(_compact(block_start), "content_block_start"):
                return
            for line in _lines(text):
                delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": line}}
                if not self._send_event(_compact(delta), "content_block_delta"):
                    return
        if not self._send_event(_compact({"type": "content_block_stop", "index": 0}), "content_block_stop"):
            return
        message_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 9},
        }
        if not self._send_event(_compact(message_delta), "message_delta"):
            return
        self._send_event(_compact({"type": "message_stop"}), "message_stop")
