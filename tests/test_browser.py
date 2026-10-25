from __future__ import annotations

import http.server
import os
import shutil
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ohar.browser import BrowserManager
from ohar.types import BrowserConfig, ToolContext

CHROMIUM = next(
    (path for path in (shutil.which(name) for name in (
        "chromium", "chromium-browser", "google-chrome", "chrome"
    )) if path),
    None,
)

READ_ONLY_TOOLS = {
    "browser_open",
    "browser_snapshot",
    "browser_scroll",
    "browser_back",
}
MUTATING_TOOLS = {
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_screenshot",
}
ALL_TOOLS = READ_ONLY_TOOLS | MUTATING_TOOLS


def _group_exists(pgid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        time.sleep(0.05)
    return True


class BrowserUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="ohar-browser-unit-")
        self.workspace = Path(self.temp.name) / "workspace"
        self.workspace.mkdir()
        self.managers: list[BrowserManager] = []

    def tearDown(self) -> None:
        for manager in self.managers:
            manager.close()
        self.temp.cleanup()

    def _manager(self, **overrides: object) -> BrowserManager:
        values: dict[str, object] = {
            "enabled": True,
            "allow_private": False,
            "require_approval": False,
            "timeout": 2.0,
            "chromium_path": "/definitely/missing/ohar-chromium",
        }
        values.update(overrides)
        manager = BrowserManager(BrowserConfig(**values))
        self.managers.append(manager)
        return manager

    def _tools(self, manager: BrowserManager) -> dict[str, object]:
        return {tool.name: tool for tool in manager.tools}

    def _ctx(self, *, read_only: bool = False, confirm=None) -> ToolContext:
        return ToolContext(
            workspace=self.workspace,
            approval_mode="ask" if confirm is not None else "never",
            read_only=read_only,
            confirm=confirm,
        )

    def test_disabled_manager_has_no_tools(self) -> None:
        manager = self._manager(enabled=False, chromium_path=None)
        self.assertEqual(manager.tools, [])
        summary = manager.summaries()
        self.assertEqual(len(summary), 1)
        self.assertEqual(set(summary[0]), {"name", "running", "pages", "error"})
        self.assertEqual(summary[0]["name"], "chromium")
        self.assertFalse(summary[0]["running"])
        self.assertEqual(summary[0]["pages"], 0)
        self.assertIsNone(manager._process)
        manager.close()
        manager.close()
        self.assertFalse(manager.summaries()[0]["running"])

    def test_tool_names_schemas_and_read_only_flags(self) -> None:
        manager = self._manager()
        tools = self._tools(manager)
        self.assertEqual(set(tools), ALL_TOOLS)
        expected_required = {
            "browser_open": ["url"],
            "browser_snapshot": [],
            "browser_click": ["selector"],
            "browser_type": ["selector", "text"],
            "browser_press": ["key"],
            "browser_scroll": ["direction"],
            "browser_back": [],
            "browser_screenshot": [],
        }
        for name, tool in tools.items():
            self.assertEqual(tool.parameters["type"], "object")
            self.assertIs(tool.parameters["additionalProperties"], False)
            self.assertEqual(tool.parameters["required"], expected_required[name])
            self.assertTrue(tool.description)
            self.assertTrue(callable(tool.handler))
        for name in READ_ONLY_TOOLS:
            self.assertTrue(tools[name].read_only, name)
        for name in MUTATING_TOOLS:
            self.assertFalse(tools[name].read_only, name)
        returned = manager.tools
        returned.clear()
        self.assertEqual(set(self._tools(manager)), ALL_TOOLS)

    def test_invalid_urls_fail_before_browser_launch(self) -> None:
        manager = self._manager()
        ctx = self._ctx()
        tool = self._tools(manager)["browser_open"]
        for url in (
            "",
            "file:///etc/passwd",
            "data:text/html,<h1>hi</h1>",
            "javascript:alert(1)",
            "http://127.0.0.1:9/",
            "https://",
        ):
            with self.subTest(url=url):
                result = tool.handler({"url": url}, ctx)
                self.assertFalse(result.ok)
                self.assertNotIn("denied", result.meta)
        self.assertIsNone(manager._process)
        self.assertFalse(manager.summaries()[0]["running"])

    def test_private_url_requires_explicit_allow_private(self) -> None:
        manager = self._manager()
        tool = self._tools(manager)["browser_open"]
        result = tool.handler({"url": "http://127.0.0.1:9/"}, self._ctx())
        self.assertFalse(result.ok)
        self.assertIn("publicly routable", result.content.lower())
        self.assertIsNone(manager._process)

    def test_approval_denial_returns_denied_without_launch(self) -> None:
        manager = self._manager(require_approval=True, allow_private=True)
        requests = []

        def deny(request) -> bool:
            requests.append(request)
            return False

        ctx = self._ctx(confirm=deny)
        result = self._tools(manager)["browser_open"].handler(
            {"url": "http://127.0.0.1:9/"}, ctx
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.meta.get("denied"))
        self.assertEqual(result.meta.get("error"), "denied")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].action, "browser_open")
        self.assertIsNone(manager._process)

    def test_read_only_context_denies_mutating_tools(self) -> None:
        manager = self._manager(allow_private=True)
        tools = self._tools(manager)
        ctx = self._ctx(read_only=True)
        cases = (
            ("browser_click", {"selector": "#go"}),
            ("browser_type", {"selector": "#q", "text": "hello"}),
            ("browser_press", {"key": "Enter"}),
            ("browser_screenshot", {}),
        )
        for name, args in cases:
            with self.subTest(tool=name):
                result = tools[name].handler(args, ctx)
                self.assertFalse(result.ok)
                self.assertTrue(result.meta.get("denied"))
                self.assertIn("Read-only mode", result.content)
        self.assertIsNone(manager._process)
        # Read-only handlers are not defensively rejected merely because the
        # context is read-only; they proceed to the (broken) browser instead.
        snapshot = tools["browser_snapshot"].handler({}, ctx)
        self.assertFalse(snapshot.ok)
        self.assertNotIn("Read-only mode", snapshot.content)
        self.assertFalse(snapshot.meta.get("denied"))

    def test_invalid_arguments_fail_before_launch(self) -> None:
        manager = self._manager()
        tools = self._tools(manager)
        ctx = self._ctx()
        cases = (
            ("browser_click", {"selector": "x" * 1_001}),
            ("browser_click", {"selector": "   "}),
            ("browser_type", {"selector": "#q", "text": "ok", "submit": "yes"}),
            ("browser_type", {"selector": "", "text": "ok"}),
            ("browser_press", {"key": "F13"}),
            ("browser_scroll", {"direction": "sideways"}),
            ("browser_screenshot", {"path": "shot.jpg"}),
            ("browser_screenshot", {"path": "../outside.png"}),
        )
        for name, args in cases:
            with self.subTest(tool=name, args=args):
                result = tools[name].handler(args, ctx)
                self.assertFalse(result.ok)
                self.assertNotIn("denied", result.meta)
        self.assertIsNone(manager._process)

    def test_broken_chromium_failure_is_actionable(self) -> None:
        manager = self._manager(allow_private=True)
        ctx = self._ctx()
        result = self._tools(manager)["browser_open"].handler(
            {"url": "http://127.0.0.1:9/"}, ctx
        )
        self.assertFalse(result.ok)
        self.assertIn("Chromium", result.content)
        self.assertIsNone(manager._process)
        summary = manager.summaries()[0]
        self.assertFalse(summary["running"])
        self.assertTrue(summary["error"])
        manager.close()
        manager.close()
        self.assertFalse(manager.summaries()[0]["running"])

    def test_startup_timeout_reaps_process_group(self) -> None:
        fake_dir = Path(self.temp.name) / "fake"
        fake_dir.mkdir()
        pidfile = fake_dir / "pid"
        script = fake_dir / "fake-chromium"
        script.write_text(
            "#!/bin/sh\n"
            f'echo $$ > "{pidfile}"\n'
            "sleep 30\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        manager = self._manager(
            chromium_path=str(script), allow_private=True, timeout=0.35
        )
        started = time.monotonic()
        result = self._tools(manager)["browser_snapshot"].handler({}, self._ctx())
        elapsed = time.monotonic() - started
        self.assertFalse(result.ok)
        self.assertIn("Chromium", result.content)
        self.assertLess(elapsed, 5.0)
        self.assertIsNone(manager._process)
        self.assertIsNone(manager._profile)
        self.assertIsNone(manager._pgid)
        deadline = time.monotonic() + 2.0
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(pidfile.exists())
        pid = int(pidfile.read_text(encoding="utf-8").strip())
        self.assertFalse(_group_exists(pid, timeout=1.0))

    def test_approval_callback_failure_is_treated_as_denial(self) -> None:
        manager = self._manager(require_approval=True, allow_private=True)

        def explode(_request) -> bool:
            raise RuntimeError("UI failed")

        result = self._tools(manager)["browser_open"].handler(
            {"url": "http://127.0.0.1:9/"}, self._ctx(confirm=explode)
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.meta.get("denied"))
        self.assertIsNone(manager._process)


_PAGE = b"""<!doctype html>
<html>
<head><title>OHAR Test Page</title></head>
<body>
  <h1>Browser integration</h1>
  <a id="next" href="/next">Next page</a>
  <a href="#fragment">Fragment link</a>
  <label for="q">Query</label>
  <input id="q" name="q" placeholder="Type here">
  <input name="query2" placeholder="Name only">
  <button id="go" onclick="document.getElementById('status').textContent =
    'Value: ' + document.getElementById('q').value">Go</button>
  <button id="hidden" style="display: none">Hidden</button>
  <select id="choice"><option>One</option><option>Two</option></select>
  <textarea id="notes" rows="2" cols="20"></textarea>
  <p id="status">idle</p>
  <p>Some readable text with \x1b[31mANSI\x1b[0m content.</p>
</body>
</html>
"""

_NEXT_PAGE = b"""<!doctype html>
<html><head><title>Next Page</title></head>
<body><h1>Second page</h1><p>Navigation worked.</p></body></html>
"""


@unittest.skipUnless(
    CHROMIUM is not None,
    "Chromium is not installed; skipping the real-browser integration test.",
)
class BrowserIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="ohar-browser-integration-")
        root = Path(self.temp.name)
        self.home = root / "home"
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.previous_home = os.environ.get("OHAR_HOME")
        os.environ["OHAR_HOME"] = str(self.home)

        page = _PAGE
        next_page = _NEXT_PAGE

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                if self.path == "/next":
                    body, status = next_page, 200
                elif self.path in {"/", "/index.html"}:
                    body, status = page, 200
                else:
                    body, status = b"not found", 404
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *args: object) -> None:
                return

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        self.manager = BrowserManager(
            BrowserConfig(
                allow_private=True,
                headless=True,
                timeout=10.0,
                chromium_path=CHROMIUM,
            )
        )
        self.ctx = ToolContext(workspace=self.workspace, approval_mode="never")
        self.tools = {tool.name: tool for tool in self.manager.tools}

    def tearDown(self) -> None:
        self.manager.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        if self.previous_home is None:
            os.environ.pop("OHAR_HOME", None)
        else:
            os.environ["OHAR_HOME"] = self.previous_home
        self.temp.cleanup()

    def test_private_final_page_is_blocked_and_cleared(self) -> None:
        opened = self.tools["browser_open"].handler({"url": self.base_url}, self.ctx)
        self.assertTrue(opened.ok, opened.content)
        self.manager._config.allow_private = False
        try:
            blocked = self.tools["browser_snapshot"].handler({}, self.ctx)
            self.assertFalse(blocked.ok)
            self.assertTrue(blocked.meta.get("blocked"))
            self.assertIn("Browser blocked the current page", blocked.content)
            self.assertNotIn("Browser integration", blocked.content)

            self.manager._config.allow_private = True
            reopened = self.tools["browser_open"].handler({"url": self.base_url}, self.ctx)
            self.assertTrue(reopened.ok, reopened.content)
            self.manager._config.allow_private = False
            shot_path = self.workspace / "private.png"
            shot = self.tools["browser_screenshot"].handler(
                {"path": "private.png"}, self.ctx
            )
            self.assertFalse(shot.ok)
            self.assertTrue(shot.meta.get("blocked"))
            self.assertFalse(shot_path.exists())
        finally:
            self.manager._config.allow_private = True
        time.sleep(0.2)
        cleared = self.tools["browser_snapshot"].handler({}, self.ctx)
        self.assertTrue(cleared.ok, cleared.content)
        self.assertEqual(cleared.meta["final_url"], "about:blank")

    def test_real_browser_flow(self) -> None:
        started = time.monotonic()
        try:
            opened = self.tools["browser_open"].handler(
                {"url": self.base_url}, self.ctx
            )
            self.assertTrue(opened.ok, opened.content)
            self.assertEqual(opened.meta["title"], "OHAR Test Page")
            self.assertTrue(opened.meta["final_url"].endswith("/"))
            self.assertGreaterEqual(opened.meta["elements"], 6)
            self.assertIn("selector: #next", opened.content)
            self.assertIn("selector: #q", opened.content)
            self.assertIn('selector: input[name="query2"]', opened.content)
            self.assertIn("selector: #choice", opened.content)
            self.assertIn("selector: #notes", opened.content)
            self.assertIn("data-ohar-ref", opened.content)
            self.assertNotIn("#hidden", opened.content)
            self.assertIn("ANSI content", opened.content)
            self.assertNotIn("\x1b", opened.content)

            snapshot = self.tools["browser_snapshot"].handler({}, self.ctx)
            self.assertTrue(snapshot.ok, snapshot.content)
            self.assertIn("Browser integration", snapshot.content)
            self.assertIn("Interactive elements", snapshot.content)

            clicked = self.tools["browser_click"].handler(
                {"selector": "#next"}, self.ctx
            )
            self.assertTrue(clicked.ok, clicked.content)
            self.assertEqual(clicked.meta["title"], "Next Page")
            self.assertTrue(clicked.meta["final_url"].endswith("/next"))

            back = self.tools["browser_back"].handler({}, self.ctx)
            self.assertTrue(back.ok, back.content)
            self.assertEqual(back.meta["title"], "OHAR Test Page")
            self.assertEqual(
                back.meta["final_url"].rstrip("/"), self.base_url.rstrip("/")
            )

            typed = self.tools["browser_type"].handler(
                {"selector": "#q", "text": "hello", "submit": False}, self.ctx
            )
            self.assertTrue(typed.ok, typed.content)
            pressed = self.tools["browser_press"].handler({"key": "Tab"}, self.ctx)
            self.assertTrue(pressed.ok, pressed.content)
            go = self.tools["browser_click"].handler({"selector": "#go"}, self.ctx)
            self.assertTrue(go.ok, go.content)
            self.assertIn("Value: hello", go.content)

            scrolled = self.tools["browser_scroll"].handler(
                {"direction": "bottom"}, self.ctx
            )
            self.assertTrue(scrolled.ok, scrolled.content)

            custom = self.tools["browser_screenshot"].handler(
                {"path": "artifacts/page.png"}, self.ctx
            )
            self.assertTrue(custom.ok, custom.content)
            self.assertEqual(custom.meta["format"], "png")
            custom_path = self.workspace / "artifacts" / "page.png"
            self.assertTrue(custom_path.is_file())
            self.assertEqual(custom_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(stat.S_IMODE(custom_path.stat().st_mode), 0o600)

            default = self.tools["browser_screenshot"].handler({}, self.ctx)
            self.assertTrue(default.ok, default.content)
            default_path = Path(default.meta["path"])
            self.assertEqual(default_path.parent, self.home / "screenshots")
            self.assertTrue(default_path.is_file())
            self.assertEqual(default_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(stat.S_IMODE(default_path.stat().st_mode), 0o600)
            self.assertEqual(
                stat.S_IMODE(default_path.parent.stat().st_mode), 0o700
            )

            process = self.manager._process
            profile = self.manager._profile
            self.assertIsNotNone(process)
            self.assertIsNotNone(profile)
            pgid = os.getpgid(process.pid)
            self.assertTrue(default.content.startswith("Screenshot saved:"))

            self.manager.close()
            self.manager.close()
            summary = self.manager.summaries()[0]
            self.assertFalse(summary["running"])
            self.assertEqual(summary["pages"], 0)
            self.assertFalse(_group_exists(pgid))
            self.assertFalse(profile.exists())
        finally:
            self.manager.close()
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 15.0, f"integration test took {elapsed:.1f}s")


if __name__ == "__main__":
    unittest.main()
