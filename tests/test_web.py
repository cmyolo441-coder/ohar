"""Tests for the stdlib-only web tools. No external network access."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from ohar.types import ApprovalRequest, ToolContext
from ohar.webtools import (
    MAX_LINKS,
    MAX_URL_CHARS,
    _search_blocked,
    extract_readable,
    make_web_tools,
    parse_ddg_html,
    parse_searx_json,
    validate_url,
)

PROJECT = Path(__file__).resolve().parent.parent
Route = Callable[[BaseHTTPRequestHandler], None]

RICH_HTML = """<!doctype html>
<html><head>
<title>  Test &amp; Page  </title>
<style>.secret { color: red; }</style>
<script>var secret = "script-text";</script>
</head><body>
<nav><a href="/nav-link">Navigation</a>Nav words</nav>
<header><h1>Site Header</h1></header>
<main>
<h1>Welcome</h1>
<p>Hello <b>world</b>.&nbsp;More</p>
<p>Second paragraph.</p>
<ul><li>One</li><li>Two</li></ul>
<a href="/relative">Relative Link</a>
<a href="https://example.org/abs">Absolute</a>
<a href="javascript:alert(1)">Bad JS</a>
<a href="mailto:test@example.org">Mail</a>
<a href="ftp://example.org/file">FTP</a>
<a href="#section">Fragment</a>
<a href="/relative">Relative Link</a>
<a href="https://user:pass@example.org/cred">Creds</a>
</main>
<footer>Footer words</footer>
<aside>Aside words</aside>
</body></html>"""

ANSI_HTML = (
    "<html><head><title>T\x1b[31mitle\x07</title>"
    "<style>.x{color:red}</style>"
    "<script>document.write('injected')</script></head><body>"
    "<p>red \x1b[31mtext\x1b[0m bell \x07 ctrl \x01 c1 \x9b end</p>"
    "<noscript>noscript words</noscript>"
    "</body></html>"
)

MANY_LINKS_HTML = (
    "<html><head><title>Many</title></head><body>"
    + "".join(f'<p><a href="/link/{index}">Link {index}</a></p>' for index in range(60))
    + "</body></html>"
)

DEEP_HTML = (
    "<!doctype html><html><head><title>Deep</title></head><body>"
    + "<div>" * 2000
    + "deep text"
    + "</div>" * 2000
    + '<a href="/' + "a" * 5000 + '" title="' + "t" * 10000 + '">Huge label '
    + "L" * 500
    + "</a></body></html>"
)

MALFORMED_HTML = (
    "<html><head><title>Broken</title></head><body>"
    '<div class=>unterminated <b>bold<br <p>next '
    '<a href="/unterminated">link<img src=x>'
)

DDG_HTML = """
<html><body>
<div class="result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone%3Fa%3D1%26b%3D2&amp;rut=abc">First <b>Result</b></a>
  <a class="result__snippet" href="#">Snippet <b>one</b> &amp; more</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.org/two">Second Result</a>
  <a class="result__snippet" href="#">Second snippet</a>
</div>
<div class="result">
  <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.org%2Fthree%3Fq%3Dx">Third Result</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/one?a=1&amp;b=2">Duplicate</a>
</div>
<div class="result">
  <a class="result__a" href="//example.net/proto">Protocol Relative</a>
</div>
<div class="result">
  <a class="result__a" href="javascript:alert(1)">Bad JS</a>
</div>
<div class="result">
  <a class="result__a" href="mailto:test@example.org">Mail</a>
</div>
<div class="result">
  <a class="result__a" href="https://user:pass@example.org/cred">Creds</a>
</div>
<div class="result">
  <a class="result__a" href="/relative">Relative</a>
</div>
</body></html>
"""


def _reply(
    handler: BaseHTTPRequestHandler,
    body: str | bytes,
    *,
    status: int = 200,
    content_type: str | None = "text/html; charset=utf-8",
    headers: dict[str, str] | None = None,
) -> None:
    payload = body.encode("utf-8") if isinstance(body, str) else body
    handler.send_response(status)
    if content_type is not None:
        handler.send_header("Content-Type", content_type)
    for key, value in (headers or {}).items():
        handler.send_header(key, value)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    try:
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _redirect(handler: BaseHTTPRequestHandler, location: str, status: int = 302) -> None:
    _reply(handler, b"", status=status, content_type=None, headers={"Location": location})


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        server = self.server
        server.requests.append(self.path)  # type: ignore[attr-defined]
        path = urllib.parse.urlsplit(self.path).path
        route = server.routes.get(path)  # type: ignore[attr-defined]
        if route is None:
            _reply(self, "missing", status=404, content_type="text/plain; charset=utf-8")
            return
        route(self)

    def log_message(self, format: str, *args: Any) -> None:
        return None


class _LocalServer:
    """Loopback-only HTTP server with a daemon worker thread."""

    def __init__(self, routes: dict[str, Route]) -> None:
        self.routes = routes
        self.requests: list[str] = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.routes = routes  # type: ignore[attr-defined]
        self.httpd.requests = self.requests  # type: ignore[attr-defined]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, name="ohar-web-test-server", daemon=True
        )
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("test HTTP server thread did not stop")


def _tool(tools: list[Any], name: str) -> Any:
    return next(tool for tool in tools if tool.name == name)


def _context(**kwargs: Any) -> ToolContext:
    return ToolContext(workspace=PROJECT, **kwargs)


class WebToolSchemaTests(unittest.TestCase):
    def test_names_read_only_and_strict_schemas(self) -> None:
        tools = make_web_tools()
        self.assertEqual([tool.name for tool in tools], ["web_fetch", "web_search"])
        fetch, search = tools
        for tool in tools:
            self.assertTrue(tool.read_only)
            self.assertFalse(tool.destructive)
            self.assertEqual(tool.parameters["type"], "object")
            self.assertIs(tool.parameters["additionalProperties"], False)
        self.assertEqual(fetch.parameters["required"], ["url"])
        self.assertEqual(set(fetch.parameters["properties"]), {"url", "max_chars"})
        self.assertEqual(fetch.parameters["properties"]["url"]["type"], "string")
        chars = fetch.parameters["properties"]["max_chars"]
        self.assertEqual(chars["type"], "integer")
        self.assertEqual(chars["minimum"], 1)
        self.assertEqual(chars["maximum"], 100_000)
        self.assertEqual(search.parameters["required"], ["query"])
        self.assertEqual(set(search.parameters["properties"]), {"query", "max_results"})
        self.assertEqual(search.parameters["properties"]["query"]["type"], "string")
        results = search.parameters["properties"]["max_results"]
        self.assertEqual(results["type"], "integer")
        self.assertEqual(results["minimum"], 1)
        self.assertEqual(results["maximum"], 10)
        spec = fetch.api_spec()
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "web_fetch")
        self.assertIs(spec["function"]["parameters"], fetch.parameters)

    def test_factory_rejects_invalid_options(self) -> None:
        cases = [
            {"timeout": 0},
            {"timeout": -1.0},
            {"timeout": True},
            {"timeout": "5"},
            {"max_bytes": 0},
            {"max_bytes": True},
            {"max_bytes": 1.5},
            {"max_chars": 0},
            {"max_chars": True},
            {"user_agent": ""},
            {"user_agent": "   "},
            {"user_agent": "bad\nagent"},
            {"user_agent": "bad\x7fagent"},
            {"search_backend": ""},
            {"search_backend": "   "},
            {"searx_url": 5},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_web_tools(**kwargs)

    def test_backend_is_normalized(self) -> None:
        tools = make_web_tools(search_backend="  SeArXNG  ")
        result = tools[1].handler({"query": ""}, _context())
        self.assertFalse(result.ok)
        self.assertEqual(result.meta["backend"], "searxng")


class ValidateUrlTests(unittest.TestCase):
    def assert_invalid(
        self, url: Any, *needles: str, allow_private: bool = False
    ) -> None:
        with self.assertRaises(ValueError) as caught:
            validate_url(url, allow_private=allow_private)
        message = str(caught.exception)
        for needle in needles:
            self.assertIn(needle, message)

    def test_allows_public_http_and_https(self) -> None:
        urls = [
            "http://8.8.8.8/",
            "https://8.8.8.8:8443/path?q=1#frag",
            "HTTP://8.8.8.8/",
            "http://[2606:4700:4700::1111]/",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(validate_url(url), url)
        self.assertEqual(validate_url("  http://8.8.8.8/  "), "http://8.8.8.8/")

    def test_rejects_non_http_schemes_and_missing_hosts(self) -> None:
        for url in (
            "ftp://8.8.8.8/file",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,hello",
            "ws://8.8.8.8/socket",
        ):
            with self.subTest(url=url):
                self.assert_invalid(url, "Only http and https")
        for url in ("http:///path", "https://:443/"):
            with self.subTest(url=url):
                self.assert_invalid(url, "hostname")
        self.assert_invalid("http://[::1", "cannot be parsed")

    def test_rejects_non_strings_and_empty_urls(self) -> None:
        for url in (None, 123, b"http://8.8.8.8/", ["http://8.8.8.8/"]):
            with self.subTest(url=url):
                self.assert_invalid(url, "must be a string")
        for url in ("", "   ", "\t"):
            with self.subTest(url=url):
                self.assert_invalid(url, "cannot be empty")

    def test_rejects_credentials(self) -> None:
        for url in (
            "http://user:pass@8.8.8.8/",
            "http://user@8.8.8.8/",
            "http://:pass@8.8.8.8/",
        ):
            with self.subTest(url=url):
                self.assert_invalid(url, "credentials")
                self.assert_invalid(url, "credentials", allow_private=True)

    def test_rejects_controls_whitespace_and_backslashes(self) -> None:
        urls = [
            "http://8.8.8.8/a b",
            "http://8.8.8.8/a\tb",
            "http://8.8.8.8/a\nb",
            "http://8.8.8.8/a\x00b",
            "http://8.8.8.8/a\x1bb",
            "http://8.8.8.8/a\x7fb",
            "http://8.8.8.8/a\x9fb",
            "http://8.8.8.8/a\\b",
            "http://8.8.8.8/a\xa0b",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assert_invalid(url, "controls, whitespace, or backslashes")

    def test_rejects_private_reserved_and_localhost_by_default(self) -> None:
        urls = [
            "http://127.0.0.1/",
            "http://127.0.0.1:8080/x",
            "http://10.0.0.1/",
            "http://172.16.5.4/",
            "http://192.168.1.2/",
            "http://169.254.169.254/latest",
            "http://0.0.0.0/",
            "http://100.64.0.1/",
            "http://198.18.0.1/",
            "http://[::1]/",
            "http://[fe80::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[64:ff9b::7f00:1]/",
            "http://localhost/",
        ]
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as caught:
                    validate_url(url)
                message = str(caught.exception)
                self.assertTrue(
                    "not publicly routable" in message
                    or "resolves to a private" in message,
                    message,
                )

    def test_allows_private_hosts_when_enabled(self) -> None:
        urls = [
            "http://127.0.0.1:5000/x",
            "http://10.0.0.1/",
            "http://localhost:9999/",
            "http://[::1]:8080/",
            "http://[::ffff:127.0.0.1]/",
            "http://[64:ff9b::7f00:1]/",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(validate_url(url, allow_private=True), url)
        self.assert_invalid("ftp://127.0.0.1/", "Only http and https", allow_private=True)
        self.assert_invalid("http://127.0.0.1/a\x01", "controls", allow_private=True)

    def test_port_validation(self) -> None:
        self.assert_invalid("http://8.8.8.8:0/", "out of range")
        self.assert_invalid("http://8.8.8.8:65536/", "invalid port")
        self.assert_invalid("http://8.8.8.8:-1/", "invalid port")
        self.assert_invalid("http://8.8.8.8:abc/", "invalid port")
        self.assertEqual(validate_url("http://8.8.8.8:65535/"), "http://8.8.8.8:65535/")

    def test_length_limit_and_boundary(self) -> None:
        prefix = "http://8.8.8.8/"
        exact = prefix + "a" * (MAX_URL_CHARS - len(prefix))
        self.assertEqual(len(exact), MAX_URL_CHARS)
        self.assertEqual(validate_url(exact), exact)
        long_url = prefix + "a" * (MAX_URL_CHARS - len(prefix) + 1)
        self.assert_invalid(long_url, "exceeds")
        self.assert_invalid(long_url, "exceeds", allow_private=True)


class ExtractReadableTests(unittest.TestCase):
    def test_title_text_and_links(self) -> None:
        page = extract_readable(RICH_HTML, "http://example.test/page")
        self.assertEqual(page.title, "Test & Page")
        for expected in (
            "Welcome",
            "Hello world. More",
            "Second paragraph.",
            "- One",
            "- Two",
        ):
            self.assertIn(expected, page.text)
        for hidden in (
            "script-text",
            "color: red",
            "Navigation",
            "Nav words",
            "Site Header",
            "Footer words",
            "Aside words",
        ):
            self.assertNotIn(hidden, page.text)
        self.assertEqual(
            page.links,
            [
                ("Relative Link", "http://example.test/relative"),
                ("Absolute", "https://example.org/abs"),
                ("Fragment", "http://example.test/page#section"),
            ],
        )

    def test_limits_links_and_keeps_order(self) -> None:
        page = extract_readable(MANY_LINKS_HTML, "http://example.test/")
        self.assertEqual(len(page.links), MAX_LINKS)
        self.assertEqual(page.links[0], ("Link 0", "http://example.test/link/0"))
        self.assertEqual(page.links[-1], ("Link 39", "http://example.test/link/39"))

    def test_ignores_unsafe_and_duplicate_links(self) -> None:
        html = (
            '<a href="javascript:alert(1)">js</a>'
            '<a href="data:text/html,x">data</a>'
            '<a href="mailto:x@example.test">mail</a>'
            '<a href="ftp://example.test/f">ftp</a>'
            '<a href="https://user:pass@example.test/">creds</a>'
            '<a href="//example.test/proto">proto</a>'
            '<a href="/has space">space</a>'
            '<a href="/has\x01control">ctrl</a>'
            '<a href="//example.test/proto">duplicate</a>'
        )
        page = extract_readable(html, "http://example.test/")
        self.assertEqual(page.links, [("proto", "http://example.test/proto")])

    def test_malformed_and_deep_markup_do_not_crash(self) -> None:
        page = extract_readable(MALFORMED_HTML, "http://example.test/")
        self.assertEqual(page.title, "Broken")
        self.assertIn("unterminated", page.text)
        deep = (
            "<html><body>"
            + "<div>" * 1500
            + "deep text"
            + "</div>" * 1500
            + "</body></html>"
        )
        nested = extract_readable(deep, "http://example.test/")
        self.assertIn("deep text", nested.text)
        self.assertEqual(nested.links, [])
        self.assertEqual(extract_readable("", "http://example.test/").title, "")
        self.assertEqual(
            extract_readable("<title>A &amp; B</title>", "http://example.test/").title,
            "A & B",
        )


class SearchParserTests(unittest.TestCase):
    def test_parse_ddg_html_titles_urls_snippets_and_dedup(self) -> None:
        hits = parse_ddg_html(DDG_HTML)
        self.assertEqual(
            [(hit.title, hit.url) for hit in hits],
            [
                ("First Result", "https://example.com/one?a=1&b=2"),
                ("Second Result", "https://example.org/two"),
                ("Third Result", "https://example.org/three?q=x"),
                ("Protocol Relative", "https://example.net/proto"),
            ],
        )
        self.assertEqual(hits[0].snippet, "Snippet one & more")
        self.assertEqual(hits[1].snippet, "Second snippet")
        self.assertEqual(hits[2].snippet, "")
        self.assertEqual(hits[3].snippet, "")

    def test_parse_ddg_lite_and_orphan_snippets(self) -> None:
        html = """
        <a class="result-link" href="javascript:alert(1)">Filtered result</a>
        <td class="result-snippet">Orphaned snippet</td>
        <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fgood">Good result</a>
        <td class="result-snippet">Right snippet</td>
        """
        hits = parse_ddg_html(html)
        self.assertEqual([(hit.title, hit.url) for hit in hits], [("Good result", "https://example.org/good")])
        self.assertEqual(hits[0].snippet, "Right snippet")

    def test_parse_ddg_html_blocked_detection(self) -> None:
        blocked = (
            '<html><body><div class="anomaly-modal">Unusual Traffic</div>'
            '<form class="challenge-form"></form></body></html>'
        )
        self.assertTrue(_search_blocked(blocked))
        self.assertTrue(_search_blocked(blocked.upper()))
        self.assertFalse(_search_blocked(DDG_HTML))
        self.assertEqual(parse_ddg_html(blocked), [])
        self.assertEqual(parse_ddg_html("<html><body>nothing</body></html>"), [])

    def test_parse_searx_dict_bytes_and_fallbacks(self) -> None:
        payload: dict[str, Any] = {
            "results": [
                {"url": "https://example.org/a", "title": "Alpha", "content": "First snippet"},
                {"url": "https://example.org/b", "title": "Beta", "snippet": "Second snippet"},
                {"url": "https://example.org/a", "title": "Duplicate", "content": "dup"},
                {"url": "ftp://example.org/file", "title": "FTP"},
                {"url": "javascript:alert(1)", "title": "JS"},
                {"url": "https://user:pass@example.org/cred", "title": "Creds"},
                {"url": "https://example.org/notitle", "content": "  spaced\n snippet "},
                {"title": "No URL"},
                "not a dict",
            ]
        }
        expected = [
            ("Alpha", "https://example.org/a", "First snippet"),
            ("Beta", "https://example.org/b", "Second snippet"),
            ("https://example.org/notitle", "https://example.org/notitle", "spaced snippet"),
        ]
        for payload_variant in (payload, json.dumps(payload), json.dumps(payload).encode()):
            with self.subTest(kind=type(payload_variant).__name__):
                hits = parse_searx_json(payload_variant)
                self.assertEqual(
                    [(hit.title, hit.url, hit.snippet) for hit in hits], expected
                )
        self.assertEqual(parse_searx_json(payload, limit=1)[0].title, "Alpha")
        self.assertEqual(parse_searx_json(payload, limit=0), [])
        self.assertEqual(parse_searx_json({"results": []}), [])

    def test_parse_searx_rejects_invalid_shapes(self) -> None:
        for payload in ([], {}, {"results": None}, {"results": {}}, "[]", "null", 12, None):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_searx_json(payload)  # type: ignore[arg-type]
        for payload in ('{"results": [', b"\xff", '{"results": ["x",'):
            with self.subTest(payload=payload), self.assertRaises(ValueError) as caught:
                parse_searx_json(payload)
            self.assertIn("Invalid SearXNG JSON", str(caught.exception))

    def test_parse_searx_truncates_fields_and_handles_large_input(self) -> None:
        long_url = "https://example.org/" + "u" * 1200
        hit = parse_searx_json(
            {"results": [{"url": long_url, "title": "T" * 500, "content": "C" * 1000}]}
        )[0]
        self.assertEqual(len(hit.url), 1000)
        self.assertEqual(len(hit.title), 300)
        self.assertEqual(len(hit.snippet), 800)
        large = {
            "results": [
                {
                    "url": f"https://example.org/{index}",
                    "title": f"Title {index}",
                    "content": "snippet " * 50,
                }
                for index in range(1000)
            ]
        }
        hits = parse_searx_json(large, limit=5)
        self.assertEqual(len(hits), 5)
        self.assertEqual(hits[-1].url, "https://example.org/4")


class WebFetchLocalServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        routes: dict[str, Route] = {
            "/page": lambda handler: _reply(handler, RICH_HTML),
            "/redirect": lambda handler: _redirect(handler, "/page"),
            "/redirect-chain": lambda handler: _redirect(handler, "/redirect"),
            "/redirect-ftp": lambda handler: _redirect(handler, "ftp://example.org/secret"),
            "/redirect-js": lambda handler: _redirect(handler, "javascript:alert(1)"),
            "/loop": lambda handler: _redirect(handler, "/loop"),
            "/no-location": lambda handler: _reply(handler, b"", status=302, content_type=None),
            "/json": lambda handler: _reply(
                handler, '{"ok": true, "message": "hello"}', content_type="application/json"
            ),
            "/problem+json": lambda handler: _reply(
                handler,
                '{"detail": "problem"}',
                content_type="application/problem+json; charset=utf-8",
            ),
            "/no-type": lambda handler: _reply(
                handler, b"no content type", content_type=None
            ),
            "/plain": lambda handler: _reply(
                handler, "plain text\nsecond line", content_type="text/plain; charset=utf-8"
            ),
            "/latin": lambda handler: _reply(
                handler,
                "<html><head><title>Café</title></head><body><p>naïve façade</p></body></html>".encode(
                    "latin-1"
                ),
                content_type="text/html; charset=iso-8859-1",
            ),
            "/bom": lambda handler: _reply(
                handler, b"\xef\xbb\xbfbom text", content_type="text/plain; charset=utf-8"
            ),
            "/bad-charset": lambda handler: _reply(
                handler,
                b"fallback text",
                content_type="text/plain; charset=no-such-charset",
            ),
            "/invalid-utf8": lambda handler: _reply(
                handler, b"before \xff after", content_type="text/plain; charset=utf-8"
            ),
            "/long": lambda handler: _reply(
                handler, "A" * 600, content_type="text/plain; charset=utf-8"
            ),
            "/big-body": lambda handler: _reply(
                handler, "0123456789" * 500, content_type="text/plain; charset=utf-8"
            ),
            "/binary": lambda handler: _reply(
                handler, b"<html><body>\x00binary</body></html>", content_type="text/html"
            ),
            "/image": lambda handler: _reply(
                handler, b"\x89PNG\r\n\x1a\nnot-really-a-png", content_type="image/png"
            ),
            "/octet-html": lambda handler: _reply(
                handler,
                "<!doctype html><html><head><title>Sniffed</title></head>"
                "<body><p>sniffed html</p></body></html>",
                content_type="application/octet-stream",
            ),
            "/octet-text": lambda handler: _reply(
                handler, b"octet stream text", content_type="application/octet-stream"
            ),
            "/xhtml": lambda handler: _reply(
                handler,
                "<html><head><title>XHTML</title></head><body><p>xhtml body</p></body></html>",
                content_type="application/xhtml+xml",
            ),
            "/ansi": lambda handler: _reply(handler, ANSI_HTML),
            "/deep": lambda handler: _reply(handler, DEEP_HTML),
            "/many-links": lambda handler: _reply(handler, MANY_LINKS_HTML),
            "/malformed": lambda handler: _reply(handler, MALFORMED_HTML),
            "/empty": lambda handler: _reply(handler, b"", content_type="text/html"),
            "/title-control": lambda handler: _reply(
                handler,
                "<html><head><title>\x1b[31m\x07</title></head><body>ok</body></html>",
            ),
            "/error500": lambda handler: _reply(
                handler,
                "server exploded " * 1000,
                status=500,
                content_type="text/plain; charset=utf-8",
            ),
        }
        cls.server = _LocalServer(routes)
        cls.base = cls.server.base_url
        cls.ctx = _context(approval_mode="auto")
        cls.tools = make_web_tools(
            allow_private=True, timeout=5, max_bytes=2_000_000, max_chars=12_000
        )
        cls.fetch = _tool(cls.tools, "web_fetch")
        cls.private_fetch = _tool(make_web_tools(timeout=5), "web_fetch")
        cls.approval_fetch = _tool(
            make_web_tools(allow_private=True, require_approval=True, timeout=5),
            "web_fetch",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.close()

    def fetch_url(self, path: str, **arguments: Any) -> Any:
        return self.fetch.handler(
            {"url": self.base + path, **arguments}, self.ctx
        )

    def test_html_extraction_title_text_links_and_metadata(self) -> None:
        result = self.fetch_url("/page")
        self.assertTrue(result.ok, result.content)
        self.assertEqual(result.meta["status"], 200)
        self.assertEqual(result.meta["final_url"], self.base + "/page")
        self.assertEqual(result.meta["content_type"], "text/html")
        self.assertEqual(result.meta["links"], 3)
        self.assertEqual(result.meta["redirects"], 0)
        self.assertFalse(result.meta["truncated"])
        self.assertGreater(result.meta["bytes"], 0)
        self.assertIn("Title: Test & Page", result.content)
        self.assertIn(f"URL: {self.base}/page", result.content)
        self.assertIn("HTTP 200 · text/html ·", result.content)
        self.assertIn("Welcome", result.content)
        self.assertIn("Hello world. More", result.content)
        self.assertIn("Second paragraph.", result.content)
        self.assertIn(f"- Relative Link: {self.base}/relative", result.content)
        self.assertIn("- Absolute: https://example.org/abs", result.content)
        self.assertIn(f"- Fragment: {self.base}/page#section", result.content)
        self.assertIn("Links (3):", result.content)
        for hidden in (
            "script-text",
            "color: red",
            "Navigation",
            "Site Header",
            "Footer words",
            "Aside words",
        ):
            self.assertNotIn(hidden, result.content)

    def test_redirect_final_url_and_chain(self) -> None:
        result = self.fetch_url("/redirect-chain")
        self.assertTrue(result.ok, result.content)
        self.assertEqual(result.meta["final_url"], self.base + "/page")
        self.assertEqual(result.meta["redirects"], 2)
        self.assertIn(f"URL: {self.base}/page", result.content)
        self.assertIn("Title: Test & Page", result.content)
        self.assertEqual(
            self.server.requests[-3:], ["/redirect-chain", "/redirect", "/page"]
        )

    def test_redirects_only_follow_http_and_https(self) -> None:
        for path in ("/redirect-ftp", "/redirect-js"):
            with self.subTest(path=path):
                result = self.fetch_url(path)
                self.assertFalse(result.ok)
                self.assertIn("Only http and https URLs are allowed.", result.content)

    def test_redirect_loop_is_bounded(self) -> None:
        result = self.fetch_url("/loop")
        self.assertFalse(result.ok)
        self.assertIn("Too many redirects", result.content)
        self.assertEqual(result.meta["redirects"], 5)
        self.assertEqual(result.meta["status"], 302)

    def test_redirect_without_location_is_an_http_error(self) -> None:
        result = self.fetch_url("/no-location")
        self.assertFalse(result.ok)
        self.assertIn("HTTP 302", result.content)
        self.assertEqual(result.meta["status"], 302)
        self.assertEqual(result.meta["redirects"], 0)

    def test_charset_bom_and_fallback(self) -> None:
        latin = self.fetch_url("/latin")
        self.assertTrue(latin.ok, latin.content)
        self.assertIn("Title: Café", latin.content)
        self.assertIn("naïve façade", latin.content)
        bom = self.fetch_url("/bom")
        self.assertTrue(bom.ok, bom.content)
        self.assertIn("bom text", bom.content)
        self.assertNotIn("\ufeff", bom.content)
        fallback = self.fetch_url("/bad-charset")
        self.assertTrue(fallback.ok, fallback.content)
        self.assertIn("fallback text", fallback.content)

    def test_invalid_utf8_uses_replacement_characters(self) -> None:
        result = self.fetch_url("/invalid-utf8")
        self.assertTrue(result.ok, result.content)
        self.assertIn("before \ufffd after", result.content)
        self.assertNotIn("\xff", result.content)

    def test_ansi_and_control_characters_are_stripped(self) -> None:
        result = self.fetch_url("/ansi")
        self.assertTrue(result.ok, result.content)
        self.assertIn("Title: Title", result.content)
        self.assertIn("red text", result.content)
        self.assertIn("bell", result.content)
        self.assertIn("ctrl", result.content)
        self.assertIn("end", result.content)
        self.assertNotIn("document.write", result.content)
        self.assertNotIn("noscript words", result.content)
        for char in result.content:
            code = ord(char)
            self.assertTrue(code == 10 or 32 <= code < 127 or code > 159, repr(char))

    def test_max_chars_argument_truncates_text(self) -> None:
        result = self.fetch_url("/long", max_chars=40)
        self.assertTrue(result.ok, result.content)
        self.assertIn("A" * 40, result.content)
        self.assertIn("[... truncated 560 chars ...]", result.content)
        self.assertTrue(result.meta["truncated"])
        configured = _tool(
            make_web_tools(allow_private=True, timeout=5, max_chars=25), "web_fetch"
        )
        default = configured.handler({"url": self.base + "/long"}, self.ctx)
        self.assertTrue(default.ok, default.content)
        self.assertIn("A" * 25, default.content)
        self.assertIn("[... truncated 575 chars ...]", default.content)
        self.assertTrue(default.meta["truncated"])

    def test_max_bytes_truncates_response_body(self) -> None:
        tool = _tool(
            make_web_tools(allow_private=True, timeout=5, max_bytes=64), "web_fetch"
        )
        result = tool.handler({"url": self.base + "/big-body"}, self.ctx)
        self.assertTrue(result.ok, result.content)
        self.assertEqual(result.meta["bytes"], 64)
        self.assertTrue(result.meta["truncated"])
        self.assertIn("truncated at 64 bytes", result.content)
        exact = _tool(
            make_web_tools(allow_private=True, timeout=5, max_bytes=600), "web_fetch"
        )
        boundary = exact.handler({"url": self.base + "/long"}, self.ctx)
        self.assertTrue(boundary.ok, boundary.content)
        self.assertEqual(boundary.meta["bytes"], 600)
        self.assertFalse(boundary.meta["truncated"])
        self.assertNotIn("truncated at", boundary.content)

    def test_json_text_is_bounded_by_max_chars(self) -> None:
        result = self.fetch_url("/json", max_chars=5)
        self.assertTrue(result.ok, result.content)
        self.assertTrue(result.meta["truncated"])
        self.assertIn("[... truncated", result.content)
        self.assertNotIn("Links (", result.content)

    def test_json_and_plain_text_content_types(self) -> None:
        for path, expected_type, needle in (
            ("/json", "application/json", '"message": "hello"'),
            ("/problem+json", "application/problem+json", '"detail": "problem"'),
            ("/plain", "text/plain", "plain text\nsecond line"),
            ("/no-type", "text/plain", "no content type"),
            ("/octet-text", "application/octet-stream", "octet stream text"),
        ):
            with self.subTest(path=path):
                result = self.fetch_url(path)
                self.assertTrue(result.ok, result.content)
                self.assertEqual(result.meta["content_type"], expected_type)
                self.assertIn("Title: (untitled)", result.content)
                self.assertIn(needle, result.content)
                self.assertNotIn("Links (", result.content)

    def test_octet_stream_html_is_sniffed(self) -> None:
        result = self.fetch_url("/octet-html")
        self.assertTrue(result.ok, result.content)
        self.assertIn("Title: Sniffed", result.content)
        self.assertIn("sniffed html", result.content)
        self.assertNotIn("Links (", result.content)

    def test_xhtml_is_treated_as_html(self) -> None:
        result = self.fetch_url("/xhtml")
        self.assertTrue(result.ok, result.content)
        self.assertIn("Title: XHTML", result.content)
        self.assertIn("xhtml body", result.content)

    def test_binary_and_unsupported_content_are_rejected(self) -> None:
        binary = self.fetch_url("/binary")
        self.assertFalse(binary.ok)
        self.assertIn("Response is binary", binary.content)
        self.assertEqual(binary.meta["error"], "content_type")
        self.assertEqual(binary.meta["status"], 200)
        image = self.fetch_url("/image")
        self.assertFalse(image.ok)
        self.assertIn("Unsupported content type 'image/png'", image.content)
        self.assertEqual(image.meta["error"], "content_type")
        self.assertEqual(image.meta["status"], 200)
        self.assertGreater(image.meta["bytes"], 0)

    def test_http_error_status_and_bounded_detail(self) -> None:
        missing = self.fetch_url("/missing")
        self.assertFalse(missing.ok)
        self.assertIn("HTTP 404 Not Found", missing.content)
        self.assertIn("missing", missing.content)
        self.assertEqual(missing.meta["status"], 404)
        self.assertEqual(missing.meta["final_url"], self.base + "/missing")
        exploded = self.fetch_url("/error500")
        self.assertFalse(exploded.ok)
        self.assertIn("HTTP 500 Internal Server Error", exploded.content)
        self.assertEqual(exploded.meta["status"], 500)
        self.assertTrue(exploded.meta["truncated"])
        self.assertLess(len(exploded.content), 1200)

    def test_default_tool_blocks_private_hosts_before_connecting(self) -> None:
        before = len(self.server.requests)
        result = self.private_fetch.handler(
            {"url": self.base + "/page"}, self.ctx
        )
        self.assertFalse(result.ok)
        self.assertIn("not publicly routable", result.content)
        self.assertEqual(result.meta["bytes"], 0)
        self.assertEqual(len(self.server.requests), before)

    def test_invalid_fetch_arguments_are_rejected_without_requests(self) -> None:
        before = len(self.server.requests)
        cases = [
            ({"url": 123}, "must be a string"),
            ({"url": self.base + "/page", "max_chars": 0}, "positive integer"),
            ({"url": self.base + "/page", "max_chars": True}, "positive integer"),
            ({"url": self.base + "/page", "max_chars": "10"}, "positive integer"),
        ]
        for arguments, needle in cases:
            with self.subTest(arguments=arguments):
                result = self.fetch.handler(arguments, self.ctx)
                self.assertFalse(result.ok)
                self.assertIn(needle, result.content)
                self.assertEqual(result.meta["error"], "invalid_args")
        self.assertEqual(len(self.server.requests), before)

    def test_empty_pages_and_control_only_titles(self) -> None:
        empty = self.fetch_url("/empty")
        self.assertTrue(empty.ok, empty.content)
        self.assertIn("Title: (untitled)", empty.content)
        self.assertIn("(no readable text)", empty.content)
        self.assertEqual(empty.meta["bytes"], 0)
        title = self.fetch_url("/title-control")
        self.assertTrue(title.ok, title.content)
        self.assertIn("Title: (untitled)", title.content)
        self.assertIn("ok", title.content)

    def test_deep_nesting_huge_attributes_and_malformed_markup(self) -> None:
        deep = self.fetch_url("/deep", max_chars=500)
        self.assertTrue(deep.ok, deep.content[:500])
        self.assertIn("deep text", deep.content)
        self.assertEqual(deep.meta["links"], 1)
        self.assertLess(len(deep.content), 2500)
        for line in deep.content.splitlines():
            self.assertLessEqual(len(line), 700)
        malformed = self.fetch_url("/malformed")
        self.assertTrue(malformed.ok, malformed.content)
        self.assertIn("unterminated", malformed.content)
        self.assertEqual(malformed.meta["links"], 0)

    def test_link_list_is_capped(self) -> None:
        result = self.fetch_url("/many-links")
        self.assertTrue(result.ok, result.content)
        self.assertEqual(result.meta["links"], MAX_LINKS)
        self.assertIn(f"Links ({MAX_LINKS}):", result.content)
        self.assertEqual(result.content.count("\n- "), MAX_LINKS)

    def test_approval_denied_in_read_only_context(self) -> None:
        before = len(self.server.requests)
        seen: list[ApprovalRequest] = []

        def confirm(request: ApprovalRequest) -> bool:
            seen.append(request)
            return True

        ctx = _context(approval_mode="ask", read_only=True, confirm=confirm)
        result = self.approval_fetch.handler(
            {"url": self.base + "/page"}, ctx
        )
        self.assertFalse(result.ok)
        self.assertIn("denied by approval policy", result.content)
        self.assertTrue(result.meta["denied"])
        self.assertEqual(result.meta["error"], "denied")
        self.assertEqual(seen, [])
        self.assertEqual(len(self.server.requests), before)

    def test_approval_denied_in_ask_mode_and_approved_with_callback(self) -> None:
        url = self.base + "/page"
        denied = self.approval_fetch.handler(
            {"url": url}, _context(approval_mode="ask")
        )
        self.assertFalse(denied.ok)
        self.assertIn("denied by approval policy", denied.content)
        seen: list[ApprovalRequest] = []

        def confirm(request: ApprovalRequest) -> bool:
            seen.append(request)
            return True

        approved = self.approval_fetch.handler(
            {"url": url}, _context(approval_mode="ask", confirm=confirm)
        )
        self.assertTrue(approved.ok, approved.content)
        self.assertEqual(len(seen), 1)
        request = seen[0]
        self.assertEqual(request.action, "web_fetch")
        self.assertEqual(request.summary, "Fetch a web page")
        self.assertEqual(request.detail, url)
        self.assertEqual(request.risk, "low")
        auto = self.approval_fetch.handler(
            {"url": url}, _context(approval_mode="auto")
        )
        self.assertTrue(auto.ok, auto.content)

    def test_approval_invalid_arguments_do_not_prompt(self) -> None:
        seen: list[ApprovalRequest] = []

        def confirm(request: ApprovalRequest) -> bool:
            seen.append(request)
            return True

        ctx = _context(approval_mode="ask", confirm=confirm)
        result = self.approval_fetch.handler(
            {"url": self.base + "/page", "max_chars": 0}, ctx
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.meta["error"], "invalid_args")
        self.assertEqual(seen, [])


class WebSearchLocalServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state: dict[str, Any] = {
            "body": json.dumps({"results": []}),
            "status": 200,
            "content_type": "application/json",
        }

        def search(handler: BaseHTTPRequestHandler) -> None:
            _reply(
                handler,
                cls.state["body"],
                status=cls.state["status"],
                content_type=cls.state["content_type"],
            )

        cls.server = _LocalServer({"/search": search})
        cls.base = cls.server.base_url
        cls.ctx = _context(approval_mode="auto")
        cls.search = _tool(
            make_web_tools(
                allow_private=True,
                timeout=5,
                search_backend="searxng",
                searx_url=cls.base + "/",
            ),
            "web_search",
        )
        cls.approval_search = _tool(
            make_web_tools(
                allow_private=True,
                require_approval=True,
                timeout=5,
                search_backend="searxng",
                searx_url=cls.base,
            ),
            "web_search",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.close()

    def setUp(self) -> None:
        self.state["body"] = json.dumps({"results": []})
        self.state["status"] = 200
        self.state["content_type"] = "application/json"

    @staticmethod
    def payload(count: int, *, prefix: str = "Result") -> str:
        results = [
            {
                "title": f"{prefix} {index}",
                "url": f"https://example.org/{index}",
                "content": f"Snippet {index}",
            }
            for index in range(count)
        ]
        return json.dumps({"results": results})

    def test_search_success_output_and_request(self) -> None:
        self.state["body"] = self.payload(2, prefix="Alpha")
        before = len(self.server.requests)
        result = self.search.handler(
            {"query": "python web", "max_results": 2}, self.ctx
        )
        self.assertTrue(result.ok, result.content)
        self.assertIn("Search results for 'python web' (2):", result.content)
        self.assertIn("1. Alpha 0", result.content)
        self.assertIn("   https://example.org/0", result.content)
        self.assertIn("   Snippet 0", result.content)
        self.assertIn("2. Alpha 1", result.content)
        self.assertEqual(result.meta["count"], 2)
        self.assertEqual(result.meta["backend"], "searxng")
        self.assertEqual(result.meta["status"], 200)
        self.assertEqual(
            result.meta["final_url"],
            self.base + "/search?q=python+web&format=json",
        )
        self.assertGreater(result.meta["bytes"], 0)
        self.assertFalse(result.meta["truncated"])
        self.assertEqual(len(self.server.requests), before + 1)
        self.assertEqual(
            self.server.requests[-1], "/search?q=python+web&format=json"
        )

    def test_search_no_results(self) -> None:
        result = self.search.handler({"query": "nothing"}, self.ctx)
        self.assertTrue(result.ok, result.content)
        self.assertIn("No results found for 'nothing'.", result.content)
        self.assertEqual(result.meta["count"], 0)
        self.assertEqual(result.meta["backend"], "searxng")

    def test_search_enforces_and_clamps_result_limits(self) -> None:
        self.state["body"] = self.payload(8)
        default = self.search.handler({"query": "default"}, self.ctx)
        self.assertTrue(default.ok, default.content)
        self.assertEqual(default.meta["count"], 5)
        self.assertIn("5. Result 4", default.content)
        self.assertNotIn("6. Result 5", default.content)
        two = self.search.handler(
            {"query": "two", "max_results": 2}, self.ctx
        )
        self.assertEqual(two.meta["count"], 2)
        self.assertNotIn("3. Result 2", two.content)
        capped = self.search.handler(
            {"query": "many", "max_results": 99}, self.ctx
        )
        self.assertEqual(capped.meta["count"], 8)

    def test_search_invalid_arguments_never_hit_the_server(self) -> None:
        before = len(self.server.requests)
        cases = [
            ({"query": ""}, "non-empty"),
            ({"query": "   "}, "non-empty"),
            ({"query": 123}, "non-empty"),
            ({"query": "x" * 501}, "exceeds"),
            ({"query": "ok", "max_results": 0}, "positive integer"),
            ({"query": "ok", "max_results": True}, "positive integer"),
            ({"query": "ok", "max_results": "2"}, "positive integer"),
        ]
        for arguments, needle in cases:
            with self.subTest(arguments=arguments):
                result = self.search.handler(arguments, self.ctx)
                self.assertFalse(result.ok)
                self.assertIn(needle, result.content)
                self.assertEqual(result.meta["error"], "invalid_args")
                self.assertEqual(result.meta["count"], 0)
        self.assertEqual(len(self.server.requests), before)

    def test_search_invalid_json_and_http_errors(self) -> None:
        self.state["body"] = "not json"
        invalid = self.search.handler({"query": "broken"}, self.ctx)
        self.assertFalse(invalid.ok)
        self.assertIn("invalid JSON", invalid.content)
        self.assertEqual(invalid.meta["error"], "search")
        self.assertEqual(invalid.meta["status"], 200)
        self.assertEqual(invalid.meta["backend"], "searxng")
        self.state["status"] = 500
        self.state["body"] = "backend down"
        failure = self.search.handler({"query": "broken"}, self.ctx)
        self.assertFalse(failure.ok)
        self.assertIn("HTTP 500", failure.content)
        self.assertIn("backend down", failure.content)
        self.assertEqual(failure.meta["status"], 500)
        self.assertEqual(failure.meta["count"], 0)

    def test_search_backend_configuration_failures(self) -> None:
        missing = _tool(
            make_web_tools(search_backend="searxng"), "web_search"
        )
        result = missing.handler({"query": "x"}, self.ctx)
        self.assertFalse(result.ok)
        self.assertIn("requires searx_url", result.content)
        self.assertEqual(result.meta["error"], "config")
        self.assertEqual(result.meta["count"], 0)
        unknown = _tool(
            make_web_tools(search_backend="bogus"), "web_search"
        )
        result = unknown.handler({"query": "x"}, self.ctx)
        self.assertFalse(result.ok)
        self.assertIn("Unknown search backend", result.content)
        self.assertEqual(result.meta["error"], "unknown_backend")

    def test_search_blocks_private_searx_by_default(self) -> None:
        tool = _tool(
            make_web_tools(
                timeout=5,
                search_backend="searxng",
                searx_url=self.base,
            ),
            "web_search",
        )
        before = len(self.server.requests)
        result = tool.handler({"query": "private"}, self.ctx)
        self.assertFalse(result.ok)
        self.assertIn("not publicly routable", result.content)
        self.assertEqual(result.meta["count"], 0)
        self.assertEqual(len(self.server.requests), before)

    def test_search_approval_denied_and_approved(self) -> None:
        self.state["body"] = self.payload(1)
        before = len(self.server.requests)
        seen: list[ApprovalRequest] = []

        def confirm(request: ApprovalRequest) -> bool:
            seen.append(request)
            return True

        denied = self.approval_search.handler(
            {"query": "private"},
            _context(approval_mode="ask", read_only=True, confirm=confirm),
        )
        self.assertFalse(denied.ok)
        self.assertIn("denied by approval policy", denied.content)
        self.assertTrue(denied.meta["denied"])
        self.assertEqual(seen, [])
        self.assertEqual(len(self.server.requests), before)
        ask_denied = self.approval_search.handler(
            {"query": "private"}, _context(approval_mode="ask")
        )
        self.assertFalse(ask_denied.ok)
        approved = self.approval_search.handler(
            {"query": "private"}, _context(approval_mode="ask", confirm=confirm)
        )
        self.assertTrue(approved.ok, approved.content)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].action, "web_search")
        self.assertIn("private", seen[0].summary)
        self.assertEqual(seen[0].risk, "low")
        auto = self.approval_search.handler(
            {"query": "private"}, _context(approval_mode="auto")
        )
        self.assertTrue(auto.ok, auto.content)

    def test_search_output_is_sanitized(self) -> None:
        self.state["body"] = json.dumps(
            {
                "results": [
                    {
                        "title": "Bad \x1b[31mTitle\x07",
                        "url": "https://example.org/clean",
                        "content": "Snip \x01 text \x9b",
                    }
                ]
            }
        )
        result = self.search.handler({"query": "q\x1b[31m"}, self.ctx)
        self.assertTrue(result.ok, result.content)
        self.assertIn("Bad Title", result.content)
        self.assertIn("Snip", result.content)
        for char in result.content:
            code = ord(char)
            self.assertTrue(code == 10 or 32 <= code < 127 or code > 159, repr(char))
        self.assertNotIn("\x1b", result.content)
        self.assertNotIn("\x01", result.content)


if __name__ == "__main__":
    unittest.main()
