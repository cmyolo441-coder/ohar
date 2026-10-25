"""Offline tests for provider model discovery."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ohar.models import ModelInfo, ModelListError, list_models
from ohar.types import Config, ProviderConfig


@dataclass(frozen=True)
class Request:
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]


class FakeAPI:
    """A tiny local HTTP server with a single configurable JSON response."""

    def __init__(self) -> None:
        self.status = 200
        self.body = b"{}"
        self.response_headers: dict[str, str] = {}
        self.requests: list[Request] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                owner.requests.append(
                    Request(
                        path=parsed.path,
                        query=parse_qs(parsed.query),
                        headers={key.lower(): value for key, value in self.headers.items()},
                    )
                )
                self.send_response(owner.status)
                self.send_header("Content-Type", "application/json")
                for key, value in owner.response_headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(owner.body)))
                self.end_headers()
                try:
                    self.wfile.write(owner.body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def json(self, payload: Any, status: int = 200) -> None:
        self.status = status
        self.body = json.dumps(payload).encode("utf-8")

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def _config(*providers: ProviderConfig) -> Config:
    return Config(providers={provider.name: provider for provider in providers})


class ModelListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.previous_home = os.environ.get("OHAR_HOME")
        os.environ["OHAR_HOME"] = self.temp.name
        self.addCleanup(self._restore_home)

    def _restore_home(self) -> None:
        if self.previous_home is None:
            os.environ.pop("OHAR_HOME", None)
        else:
            os.environ["OHAR_HOME"] = self.previous_home

    def api(self) -> FakeAPI:
        server = FakeAPI()
        self.addCleanup(server.close)
        return server

    def test_openai_fields_headers_and_dedup(self) -> None:
        api = self.api()
        api.json(
            {
                "data": [
                    {
                        "id": "beta",
                        "owned_by": "acme",
                        "context_length": 128000,
                        "max_output_tokens": 4096,
                    },
                    {
                        "id": "gamma",
                        "context_length": 0,
                        "max_output_tokens": -1,
                        "max_completion_tokens": 2048,
                    },
                    {
                        "id": "alpha",
                        "context_length": True,
                        "max_output_tokens": "8192",
                        "max_completion_tokens": 1024,
                    },
                    {"id": "beta", "owned_by": "duplicate", "context_length": 1},
                    {"id": 42},
                    {"id": "   "},
                    "not-an-object",
                ]
            }
        )
        provider = ProviderConfig(
            name="local",
            type="openai",
            base_url=api.url + "/v1/",
            api_key="sk-local-dummy",
        )

        models = list_models(_config(provider), "local", timeout=5)

        self.assertEqual(api.requests[0].path, "/v1/models")
        self.assertEqual(api.requests[0].headers.get("authorization"), "Bearer sk-local-dummy")
        self.assertEqual(api.requests[0].headers.get("accept"), "application/json")
        self.assertEqual([model.id for model in models], ["alpha", "beta", "gamma"])
        self.assertTrue(all(model.provider == "local" for model in models))
        self.assertEqual(models[0].context_length, None)
        self.assertEqual(models[0].max_output_tokens, 1024)
        self.assertEqual(models[1].owned_by, "acme")
        self.assertEqual(models[1].context_length, 128000)
        self.assertEqual(models[1].max_output_tokens, 4096)
        self.assertEqual(models[2].context_length, None)
        self.assertEqual(models[2].max_output_tokens, 2048)
        self.assertNotIn("sk-local-dummy", repr(models))
        self.assertIsInstance(models[0], ModelInfo)

    def test_openai_accepts_raw_list_and_models_key(self) -> None:
        api = self.api()
        provider = ProviderConfig(
            name="plain", type="openai", base_url=api.url + "/v1", api_key=None
        )
        cases = [
            [{"id": "raw-model", "context_length": 4096}],
            {"models": [{"id": "wrapped-model", "max_completion_tokens": 512}]},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                api.json(payload)
                models = list_models(_config(provider), "plain", timeout=5)
                self.assertEqual(len(models), 1)
                self.assertEqual(models[0].provider, "plain")
                self.assertNotIn("authorization", api.requests[-1].headers)
        self.assertEqual(api.requests[0].path, "/v1/models")

    def test_anthropic_headers_shape_and_filtering(self) -> None:
        api = self.api()
        api.json(
            {
                "data": [
                    {
                        "id": "claude-test",
                        "type": "model",
                        "display_name": "Claude Test",
                    },
                    {
                        "id": "claude-test",
                        "type": "model",
                        "display_name": "Duplicate",
                    },
                    {"id": "other-item", "type": "other", "display_name": "Skip"},
                    {"id": 7, "type": "model"},
                ],
                "has_more": False,
                "last_id": "claude-test",
            }
        )
        provider = ProviderConfig(
            name="claude",
            type="anthropic",
            base_url=api.url + "/v1",
            api_key="ant-dummy",
        )

        models = list_models(_config(provider), "claude", timeout=5)

        request = api.requests[0]
        self.assertEqual(request.path, "/v1/models")
        self.assertEqual(request.query, {"limit": ["1000"]})
        self.assertEqual(request.headers.get("x-api-key"), "ant-dummy")
        self.assertEqual(request.headers.get("anthropic-version"), "2023-06-01")
        self.assertNotIn("authorization", request.headers)
        self.assertEqual([model.id for model in models], ["claude-test"])
        self.assertEqual(models[0].provider, "claude")
        self.assertEqual(models[0].owned_by, "Claude Test")

    def test_provider_selection_aggregate_order_and_unknown(self) -> None:
        alpha = self.api()
        beta = self.api()
        alpha.json({"data": [{"id": "z"}, {"id": "a"}]})
        beta.json({"data": [{"id": "m"}]})
        providers = _config(
            ProviderConfig(name="beta", type="openai", base_url=beta.url + "/v1"),
            ProviderConfig(name="alpha", type="openai", base_url=alpha.url + "/v1"),
        )

        models = list_models(providers, timeout=5)
        self.assertEqual(
            [(model.provider, model.id) for model in models],
            [("alpha", "a"), ("alpha", "z"), ("beta", "m")],
        )
        self.assertEqual(len(alpha.requests), 1)
        self.assertEqual(len(beta.requests), 1)

        selected = list_models(providers, "beta", timeout=5)
        self.assertEqual([(model.provider, model.id) for model in selected], [("beta", "m")])
        self.assertEqual(len(alpha.requests), 1)
        self.assertEqual(len(beta.requests), 2)

        with self.assertRaises(ModelListError) as caught:
            list_models(providers, "missing")
        message = str(caught.exception)
        self.assertIn("Unknown provider", message)
        self.assertIn("alpha", message)
        self.assertIn("beta", message)

    def test_aggregate_keeps_successes_when_one_provider_fails(self) -> None:
        good = self.api()
        bad = self.api()
        good.json({"data": [{"id": "ok-model"}]})
        bad.json({"error": {"message": "backend exploded"}}, status=500)
        providers = _config(
            ProviderConfig(
                name="broken",
                type="openai",
                base_url=bad.url + "/v1",
                api_key="sk-bad-dummy",
            ),
            ProviderConfig(name="working", type="openai", base_url=good.url + "/v1"),
        )

        models = list_models(providers, timeout=5)

        self.assertEqual([(model.provider, model.id) for model in models], [("working", "ok-model")])
        self.assertEqual(len(bad.requests), 1)
        self.assertEqual(len(good.requests), 1)

    def test_all_failures_raise_with_concise_redacted_details(self) -> None:
        one = self.api()
        two = self.api()
        one.json({"error": {"message": "sk-secret-one was rejected"}}, status=500)
        two.json({"error": {"message": "denied for sk-secret-two"}}, status=503)
        providers = _config(
            ProviderConfig(
                name="one", type="openai", base_url=one.url + "/v1", api_key="sk-secret-one"
            ),
            ProviderConfig(
                name="two", type="openai", base_url=two.url + "/v1", api_key="sk-secret-two"
            ),
        )

        with self.assertRaises(ModelListError) as caught:
            list_models(providers, timeout=5)
        message = str(caught.exception)
        self.assertIn("All providers failed", message)
        self.assertIn("one", message)
        self.assertIn("two", message)
        self.assertIn("500", message)
        self.assertIn("503", message)
        self.assertIn("***", message)
        self.assertNotIn("sk-secret-one", message)
        self.assertNotIn("sk-secret-two", message)
        self.assertNotIn("sk-secret-one", repr(caught.exception))
        self.assertNotIn("sk-secret-two", repr(caught.exception))

        with self.assertRaises(ModelListError) as direct:
            list_models(providers, "one", timeout=5)
        self.assertIn("one:", str(direct.exception))
        self.assertNotIn("sk-secret-one", str(direct.exception))

    def test_malformed_json_and_unexpected_shape(self) -> None:
        api = self.api()
        provider = ProviderConfig(
            name="faulty", type="openai", base_url=api.url + "/v1", api_key="sk-faulty"
        )
        cases = [
            (b"not json at all", "JSON"),
            (b'{"unexpected": []}', "unexpected"),
            (b'"just a string"', "unexpected"),
        ]
        for body, expected in cases:
            with self.subTest(body=body):
                api.body = body
                with self.assertRaises(ModelListError) as caught:
                    list_models(_config(provider), "faulty", timeout=5)
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn("sk-faulty", str(caught.exception))

    def test_redirect_is_not_followed_and_auth_is_not_forwarded(self) -> None:
        target = self.api()
        target.json({"data": [{"id": "must-not-be-reached"}]})
        origin = self.api()
        origin.status = 302
        origin.body = b""
        origin.response_headers = {"Location": target.url + "/v1/models"}
        provider = ProviderConfig(
            name="redirector",
            type="openai",
            base_url=origin.url + "/v1",
            api_key="sk-redirect-dummy",
        )

        with self.assertRaises(ModelListError) as caught:
            list_models(_config(provider), "redirector", timeout=5)

        self.assertIn("302", str(caught.exception))
        self.assertNotIn("sk-redirect-dummy", str(caught.exception))
        self.assertEqual(target.requests, [])
        self.assertEqual(origin.requests[0].headers.get("authorization"), "Bearer sk-redirect-dummy")

    def test_response_body_is_bounded(self) -> None:
        api = self.api()
        api.body = b" " * (2 * 1024 * 1024 + 1)
        provider = ProviderConfig(
            name="huge", type="openai", base_url=api.url + "/v1", api_key="sk-huge"
        )

        with self.assertRaises(ModelListError) as caught:
            list_models(_config(provider), "huge", timeout=5)
        self.assertIn("exceeds", str(caught.exception))
        self.assertNotIn("sk-huge", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
