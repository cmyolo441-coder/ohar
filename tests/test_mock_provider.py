"""Tests for the offline MockProvider used by OHAR integration tests."""

from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from typing import Any

from tests.mock_provider import MockProvider

USAGE = {"prompt_tokens": 17, "completion_tokens": 9, "total_tokens": 26}


def _request(
    provider: MockProvider,
    path: str,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> urllib.request.Request:
    return urllib.request.Request(
        provider.url(path),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )


def _post(
    provider: MockProvider,
    path: str,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    with urllib.request.urlopen(_request(provider, path, body, headers), timeout=5) as response:
        response_headers = {key.lower(): value for key, value in response.headers.items()}
        return response.status, response_headers, response.read().decode("utf-8")


def _events(raw: str) -> list[tuple[str | None, str]]:
    events: list[tuple[str | None, str]] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        event: str | None = None
        data: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data.append(line.split(":", 1)[1].lstrip())
        events.append((event, "\n".join(data)))
    return events


def _json_chunks(raw: str) -> list[dict[str, Any]]:
    return [json.loads(data) for _, data in _events(raw) if data and data != "[DONE]"]


def _content(chunks: list[dict[str, Any]]) -> str:
    return "".join(
        choice.get("delta", {}).get("content", "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )


class MockProviderTests(unittest.TestCase):
    def make_provider(self) -> MockProvider:
        provider = MockProvider()
        self.addCleanup(provider.stop)
        return provider.start()

    def test_health_endpoint_and_context_manager(self) -> None:
        provider = MockProvider()
        self.addCleanup(provider.stop)
        with provider as active:
            self.assertIs(active, provider)
            self.assertEqual(provider.host, "127.0.0.1")
            self.assertGreater(provider.port, 0)
            self.assertTrue(provider.base_url.endswith("/v1"))
            self.assertEqual(provider.url("/health"), f"http://127.0.0.1:{provider.port}/health")
            with urllib.request.urlopen(provider.url("/health"), timeout=5) as response:
                self.assertEqual(json.loads(response.read()), {"ok": True})
        provider.stop()
        self.assertEqual(provider.requests_count, 1)
        self.assertEqual(provider.last_request["path"], "/health")  # type: ignore[index]
        self.assertEqual(provider.requests[0]["method"], "GET")

    def test_openai_stream_content_usage_and_recording(self) -> None:
        provider = self.make_provider()
        provider.delay_seconds = 0.001
        body = {
            "model": "mock-openai",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello offline world"}],
        }
        status, headers, raw = _post(
            provider,
            "/v1/chat/completions",
            body,
            {"Authorization": "Bearer test-key"},
        )
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["content-type"])

        events = _events(raw)
        self.assertEqual(events[-1], (None, "[DONE]"))
        chunks = _json_chunks(raw)
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})

        content = _content(chunks)
        for expected in (
            "# Mock Response",
            "This is a canned markdown reply",
            "- Bullet one",
            "`mock_provider`",
            "```python",
            "Hello offline world",
        ):
            self.assertIn(expected, content)
        finish_reasons = [
            choice["finish_reason"]
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("finish_reason")
        ]
        self.assertEqual(finish_reasons, ["stop"])
        usages = [chunk["usage"] for chunk in chunks if "usage" in chunk]
        self.assertEqual(usages, [USAGE])

        self.assertEqual(provider.requests_count, 1)
        recorded = provider.last_request
        self.assertIsNotNone(recorded)
        self.assertEqual(recorded["path"], "/v1/chat/completions")  # type: ignore[index]
        self.assertEqual(recorded["body"]["model"], "mock-openai")
        self.assertEqual(provider.headers_seen[-1]["authorization"], "Bearer test-key")

    def test_openai_stream_tool_call_arguments(self) -> None:
        provider = self.make_provider()
        body = {
            "model": "mock",
            "stream": True,
            "messages": [{"role": "user", "content": "USE_TOOL TOOL_NAME:read_file"}],
        }
        _, _, raw = _post(provider, "/v1/chat/completions", body)
        chunks = _json_chunks(raw)
        self.assertEqual(_content(chunks), "")
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})

        call_id: str | None = None
        name: str | None = None
        arguments = ""
        fragments = 0
        for chunk in chunks:
            for choice in chunk.get("choices", []):
                for tool_call in choice.get("delta", {}).get("tool_calls", []):
                    self.assertEqual(tool_call["index"], 0)
                    call_id = tool_call.get("id") or call_id
                    function = tool_call.get("function", {})
                    name = function.get("name") or name
                    piece = function.get("arguments", "")
                    if piece:
                        arguments += piece
                        fragments += 1
        self.assertEqual(call_id, "call_mock_1")
        self.assertEqual(name, "read_file")
        self.assertGreater(fragments, 1)
        self.assertEqual(json.loads(arguments), {"path": "README.md"})

        finish_reasons = [
            choice["finish_reason"]
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("finish_reason")
        ]
        self.assertEqual(finish_reasons, ["tool_calls"])
        self.assertEqual([chunk["usage"] for chunk in chunks if "usage" in chunk], [USAGE])
        self.assertEqual(provider.requests_count, 1)

    def test_openai_final_text_after_tool_result(self) -> None:
        provider = self.make_provider()
        body = {
            "model": "mock",
            "stream": True,
            "messages": [
                {"role": "user", "content": "USE_TOOL TOOL_NAME:read_file"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_mock_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_mock_1", "content": "README contents"},
            ],
        }
        _, _, raw = _post(provider, "/v1/chat/completions", body)
        chunks = _json_chunks(raw)
        content = _content(chunks)
        self.assertTrue(content.startswith("Mock final answer:"))
        self.assertIn("README contents", content)
        self.assertFalse(
            any(
                choice.get("delta", {}).get("tool_calls")
                for chunk in chunks
                for choice in chunk.get("choices", [])
            )
        )
        self.assertEqual(
            [
                choice["finish_reason"]
                for chunk in chunks
                for choice in chunk.get("choices", [])
                if choice.get("finish_reason")
            ],
            ["stop"],
        )
        self.assertEqual([chunk["usage"] for chunk in chunks if "usage" in chunk], [USAGE])
        self.assertEqual(_events(raw)[-1][1], "[DONE]")

    def test_openai_non_stream_json(self) -> None:
        provider = self.make_provider()
        body = {
            "model": "mock",
            "stream": False,
            "messages": [{"role": "user", "content": "Hi there"}],
        }
        status, headers, raw = _post(provider, "/v1/chat/completions", body)
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["content-type"])
        payload = json.loads(raw)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], "mock")
        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIn("# Mock Response", choice["message"]["content"])
        self.assertIn("Hi there", choice["message"]["content"])
        self.assertEqual(payload["usage"], USAGE)

    def test_openai_failure_modes(self) -> None:
        provider = self.make_provider()
        cases = (
            ("mock-fail", 500, "mock failure"),
            ("mock-ratelimit", 429, "mock rate limit"),
            ("mock-auth", 401, "mock unauthorized"),
        )
        for model, expected_status, expected_message in cases:
            with self.subTest(model=model):
                body = {"model": model, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(_request(provider, "/v1/chat/completions", body), timeout=5)
                error = caught.exception
                self.assertEqual(error.code, expected_status)
                payload = json.loads(error.read().decode("utf-8"))
                error.close()
                self.assertEqual(payload["error"]["message"], expected_message)
                if expected_status == 429:
                    self.assertEqual(error.headers.get("Retry-After"), "1")
        self.assertEqual(provider.requests_count, 3)
        self.assertEqual(provider.last_request["body"]["model"], "mock-auth")  # type: ignore[index]

    def test_anthropic_stream_text_events(self) -> None:
        provider = self.make_provider()
        body = {
            "model": "claude-mock",
            "max_tokens": 128,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello Anthropic"}],
        }
        status, headers, raw = _post(
            provider,
            "/v1/messages",
            body,
            {"x-api-key": "anthropic-test-key", "anthropic-version": "2023-06-01"},
        )
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["content-type"])
        events = _events(raw)
        names = [name for name, _ in events]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[1], "content_block_start")
        self.assertIn("content_block_delta", names)
        self.assertIn("content_block_stop", names)
        self.assertIn("message_delta", names)
        self.assertEqual(names[-1], "message_stop")
        self.assertTrue(all(name is not None for name in names))

        payloads = [json.loads(data) for _, data in events]
        self.assertEqual(payloads[0]["type"], "message_start")
        self.assertEqual(payloads[0]["message"]["usage"]["input_tokens"], 17)
        text = "".join(
            payload.get("delta", {}).get("text", "")
            for payload in payloads
            if payload.get("type") == "content_block_delta"
        )
        self.assertIn("# Mock Response", text)
        self.assertIn("Hello Anthropic", text)
        message_delta = next(payload for payload in payloads if payload.get("type") == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "end_turn")
        self.assertEqual(message_delta["usage"]["output_tokens"], 9)
        self.assertEqual(provider.headers_seen[-1]["x-api-key"], "anthropic-test-key")

    def test_anthropic_tool_use_events(self) -> None:
        provider = self.make_provider()
        body = {
            "model": "claude-mock",
            "max_tokens": 128,
            "stream": True,
            "messages": [{"role": "user", "content": "USE_TOOL TOOL_NAME:shell"}],
        }
        _, _, raw = _post(provider, "/v1/messages", body, {"x-api-key": "anthropic-test-key"})
        events = _events(raw)
        payloads = [json.loads(data) for _, data in events]
        start = next(payload for payload in payloads if payload.get("type") == "content_block_start")
        self.assertEqual(start["content_block"]["type"], "tool_use")
        self.assertEqual(start["content_block"]["name"], "shell")
        arguments = "".join(
            payload["delta"]["partial_json"]
            for payload in payloads
            if payload.get("type") == "content_block_delta"
        )
        self.assertEqual(json.loads(arguments), {"command": "printf mock"})
        message_delta = next(payload for payload in payloads if payload.get("type") == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")
        self.assertEqual(events[-1][0], "message_stop")

    def test_anthropic_requires_api_key(self) -> None:
        provider = self.make_provider()
        body = {"model": "claude-mock", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(_request(provider, "/v1/messages", body), timeout=5)
        error = caught.exception
        self.assertEqual(error.code, 401)
        payload = json.loads(error.read().decode("utf-8"))
        error.close()
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "authentication_error")
        self.assertEqual(provider.requests_count, 1)


if __name__ == "__main__":
    unittest.main()
