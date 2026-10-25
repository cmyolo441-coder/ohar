"""Tests for the stdlib-only MCP stdio client."""

from __future__ import annotations

import re
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ohar.mcp import McpManager
from ohar.types import McpServerConfig, ToolContext

FAKE_SERVER = r'''
import argparse
import json
import os
import sys
import time

LONG_TOOL = "t" * 80

WRITE_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string", "description": "Value to store."}},
    "required": ["value"],
    "additionalProperties": False,
}

NORMAL_TOOLS = [
    {
        "name": "echo",
        "description": "Echo \u001b[31mtext\u001b[0m back to the caller.",
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "write",
        "description": "Store a value; this tool has side effects.",
        "inputSchema": WRITE_SCHEMA,
        "annotations": {"readOnlyHint": False, "destructiveHint": True},
    },
]

COLLISION_TOOLS = [
    {
        "name": "a.b",
        "description": "Dot tool.",
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "a_b",
        "description": "Underscore tool.",
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": LONG_TOOL,
        "description": "Overlong name tool.",
        "annotations": {"readOnlyHint": True},
    },
]


def send(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def rpc_error(request_id, message):
    send({"jsonrpc": "2.0", "id": request_id,
          "error": {"code": -32601, "message": message}})


def tool_result(request_id, content, structured=None, is_error=False):
    result = {"content": content}
    if structured is not None:
        result["structuredContent"] = structured
    if is_error:
        result["isError"] = True
    send({"jsonrpc": "2.0", "id": request_id, "result": result})


def handle_call(request_id, params):
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if name == "echo":
        if arguments.get("fail"):
            tool_result(request_id, [{"type": "text", "text": "boom"}],
                        is_error=True)
            return
        if arguments.get("sleep"):
            time.sleep(30)
            return
        message = str(arguments.get("message", ""))
        tool_result(
            request_id,
            [
                {"type": "text", "text": "echo: " + message},
                {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///tmp/example.txt",
                        "mimeType": "text/plain",
                        "text": "resource-body",
                    },
                },
            ],
            structured={"echoed": arguments, "ok": True},
        )
        return
    if name == "write":
        value = str(arguments.get("value", ""))
        tool_result(request_id, [{"type": "text", "text": "wrote: " + value}],
                    structured={"value": value})
        return
    if name in ("a.b", "a_b", LONG_TOOL):
        tool_result(request_id, [{"type": "text", "text": "called " + name}])
        return
    rpc_error(request_id, "unknown tool: " + str(name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--marker", default=None)
    args = parser.parse_args()

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except Exception:
                continue

            method = message.get("method")
            request_id = message.get("id")
            params = message.get("params") or {}

            if method == "initialize":
                if args.mode == "timeout":
                    time.sleep(30)
                    continue
                if args.mode == "malformed":
                    sys.stdout.write("this is not json\n")
                    sys.stdout.flush()
                    continue
                if args.mode == "crash":
                    sys.stderr.write("fake server crashing\n")
                    sys.stderr.flush()
                    os._exit(9)
                send({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": params.get(
                            "protocolVersion", "2024-11-05"
                        ),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1.0"},
                    },
                })
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                if args.mode == "collisions":
                    send({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"tools": COLLISION_TOOLS},
                    })
                    continue
                cursor = params.get("cursor")
                if cursor in (None, ""):
                    send({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": NORMAL_TOOLS[:1],
                            "nextCursor": "page-2",
                        },
                    })
                elif cursor == "page-2":
                    send({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"tools": NORMAL_TOOLS[1:]},
                    })
                else:
                    rpc_error(request_id, "bad cursor: " + str(cursor))
            elif method == "tools/call":
                handle_call(request_id, params)
            elif request_id is not None:
                rpc_error(request_id, "unknown method: " + str(method))
    finally:
        if args.marker:
            try:
                with open(args.marker, "w", encoding="utf-8") as handle:
                    handle.write("closed")
            except OSError:
                pass


if __name__ == "__main__":
    main()
'''

DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "description": "Value to store."}
    },
    "required": ["value"],
    "additionalProperties": False,
}


class McpManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.script = self.dir / "fake_mcp.py"
        self.script.write_text(FAKE_SERVER, encoding="utf-8")
        self.managers: list[McpManager] = []
        self.counter = 0

    def tearDown(self) -> None:
        for manager in reversed(self.managers):
            try:
                manager.close()
            except Exception:
                pass
        self.temp.cleanup()

    def _config(
        self,
        name: str,
        *,
        mode: str = "normal",
        timeout: float = 3.0,
        enabled: bool = True,
        env: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
    ) -> tuple[McpServerConfig, Path]:
        self.counter += 1
        marker = self.dir / f"marker-{self.counter}.txt"
        if args is None:
            args = [
                str(self.script),
                "--mode",
                mode,
                "--marker",
                str(marker),
            ]
        config = McpServerConfig(
            name=name,
            command=command or sys.executable,
            args=list(args),
            env=dict(env or {}),
            timeout=timeout,
            enabled=enabled,
        )
        return config, marker

    def _start(
        self, configs: list[McpServerConfig]
    ) -> tuple[McpManager, list[str]]:
        manager = McpManager(configs)
        self.managers.append(manager)
        return manager, manager.start()

    def _ctx(
        self,
        *,
        read_only: bool = False,
        approval_mode: str = "auto",
        confirm=None,
    ) -> ToolContext:
        return ToolContext(
            workspace=self.dir,
            approval_mode=approval_mode,
            read_only=read_only,
            confirm=confirm,
        )

    @staticmethod
    def _tools(manager: McpManager) -> dict:
        return {tool.name: tool for tool in manager.tools}

    def test_discovery_naming_annotations_and_summaries(self) -> None:
        config, marker = self._config(
            "demo.server", env={"MCP_FAKE_TOKEN": "super-secret-value"}
        )
        manager, warnings = self._start([config])
        self.assertEqual(warnings, [])
        self.assertFalse(marker.exists())

        tools = self._tools(manager)
        self.assertEqual(
            set(tools),
            {"mcp__demo_server__echo", "mcp__demo_server__write"},
        )
        for name, tool in tools.items():
            self.assertRegex(name, re.compile(r"^[A-Za-z0-9_-]+$"))
            self.assertLessEqual(len(name), 64)

        echo = tools["mcp__demo_server__echo"]
        write = tools["mcp__demo_server__write"]
        self.assertTrue(echo.read_only)
        self.assertFalse(echo.destructive)
        self.assertFalse(write.read_only)
        self.assertTrue(write.destructive)
        self.assertEqual(echo.parameters, DEFAULT_SCHEMA)
        self.assertEqual(write.parameters, WRITE_SCHEMA)
        self.assertNotIn("\x1b", echo.description)
        self.assertIn("Echo", echo.description)

        summaries = manager.summaries()
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary["name"], "demo.server")
        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["tool_count"], 2)
        self.assertIsNone(summary["error"])
        self.assertEqual(summary["command"], sys.executable)
        self.assertNotIn("super-secret-value", repr(summaries))

        # start() is idempotent and returns the same warning list.
        before = [tool.name for tool in manager.tools]
        self.assertEqual(manager.start(), warnings)
        self.assertEqual([tool.name for tool in manager.tools], before)

    def test_echo_call_formats_text_placeholders_and_structured(self) -> None:
        config, _ = self._config("demo.server")
        manager, warnings = self._start([config])
        self.assertEqual(warnings, [])
        echo = self._tools(manager)["mcp__demo_server__echo"]

        result = echo.handler({"message": "hi"}, self._ctx())
        self.assertTrue(result.ok)
        self.assertIn("echo: hi", result.content)
        self.assertIn("[image", result.content)
        self.assertIn("mimeType=image/png", result.content)
        self.assertIn("[resource", result.content)
        self.assertIn("uri=file:///tmp/example.txt", result.content)
        self.assertIn("structuredContent:", result.content)
        self.assertIn('"echoed":{"message":"hi"}', result.content)
        self.assertIn('"ok":true', result.content)
        self.assertNotIn("aGVsbG8=", result.content)
        self.assertNotIn("resource-body", result.content)
        self.assertEqual(result.meta["server"], "demo.server")
        self.assertEqual(result.meta["tool"], "echo")
        self.assertIsInstance(result.meta["duration_ms"], float)
        self.assertEqual(
            result.meta["annotations"], {"readOnlyHint": True}
        )

    def test_is_error_and_non_object_arguments(self) -> None:
        config, _ = self._config("demo.server")
        manager, _ = self._start([config])
        echo = self._tools(manager)["mcp__demo_server__echo"]

        failed = echo.handler({"fail": True}, self._ctx())
        self.assertFalse(failed.ok)
        self.assertIn("boom", failed.content)
        self.assertTrue(failed.meta["is_error"])
        self.assertEqual(manager.summaries()[0]["status"], "ready")

        bad_args = echo.handler(["not", "an", "object"], self._ctx())
        self.assertFalse(bad_args.ok)
        self.assertIn("JSON object", bad_args.content)

    def test_write_approval_denial_and_auto_mode(self) -> None:
        config, _ = self._config("demo.server")
        manager, _ = self._start([config])
        write = self._tools(manager)["mcp__demo_server__write"]

        calls = []

        def deny(request):
            calls.append(request)
            return False

        read_only = self._ctx(read_only=True, approval_mode="ask", confirm=deny)
        denied = write.handler({"value": "x"}, read_only)
        self.assertFalse(denied.ok)
        self.assertTrue(denied.meta["denied"])
        self.assertEqual(calls, [])

        ask = self._ctx(approval_mode="ask", confirm=deny)
        denied = write.handler({"value": "x"}, ask)
        self.assertFalse(denied.ok)
        self.assertTrue(denied.meta["denied"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].action, "mcp_call")
        self.assertEqual(calls[0].risk, "medium")
        self.assertIn("write", calls[0].summary)

        def unexpected(_request):
            raise AssertionError("auto mode must not prompt")

        auto = self._ctx(approval_mode="auto", confirm=unexpected)
        allowed = write.handler({"value": "x"}, auto)
        self.assertTrue(allowed.ok)
        self.assertIn("wrote: x", allowed.content)

    def test_tool_name_collisions_and_original_metadata(self) -> None:
        config, _ = self._config("collide", mode="collisions")
        manager, warnings = self._start([config])
        self.assertEqual(warnings, [])
        tools = self._tools(manager)
        names = list(tools)
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertEqual(names[0], "mcp__collide__a_b")
        self.assertEqual(names[1], "mcp__collide__a_b_2")
        for name in names:
            self.assertRegex(name, re.compile(r"^[A-Za-z0-9_-]+$"))
            self.assertLessEqual(len(name), 64)

        first = tools[names[0]].handler({}, self._ctx())
        second = tools[names[1]].handler({}, self._ctx())
        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(first.meta["tool"], "a.b")
        self.assertEqual(second.meta["tool"], "a_b")

    def test_tool_call_timeout_returns_failure_not_exception(self) -> None:
        config, _ = self._config("demo.server", timeout=0.5)
        manager, warnings = self._start([config])
        self.assertEqual(warnings, [])
        echo = self._tools(manager)["mcp__demo_server__echo"]

        started = time.monotonic()
        result = echo.handler({"sleep": True}, self._ctx())
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 5.0)
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.content)
        self.assertEqual(result.meta["server"], "demo.server")
        self.assertEqual(result.meta["tool"], "echo")
        self.assertEqual(manager.summaries()[0]["status"], "failed")
        self.assertEqual(manager.tools, [])

    def test_malformed_server_does_not_block_good_server(self) -> None:
        bad, _ = self._config("bad.server", mode="malformed", timeout=2.0)
        good, _ = self._config("good.server")
        manager, warnings = self._start([bad, good])

        self.assertEqual(len(warnings), 1)
        self.assertIn("bad.server", warnings[0])
        self.assertIn("failed", warnings[0].lower())
        summaries = {item["name"]: item for item in manager.summaries()}
        self.assertEqual(summaries["bad.server"]["status"], "failed")
        self.assertTrue(summaries["bad.server"]["error"])
        self.assertEqual(summaries["bad.server"]["tool_count"], 0)
        self.assertEqual(summaries["good.server"]["status"], "ready")
        self.assertEqual(summaries["good.server"]["tool_count"], 2)
        self.assertTrue(manager.tools)
        visible = {tool.name for tool in manager.tools}
        self.assertIn("mcp__good_server__echo", visible)
        self.assertNotIn("mcp__bad_server__echo", visible)

    def test_timeout_fails_cleanly_and_quickly(self) -> None:
        config, _ = self._config("slow.server", mode="timeout", timeout=0.25)
        started = time.monotonic()
        manager, warnings = self._start([config])
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 5.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("slow.server", warnings[0])
        self.assertIn("timed out", warnings[0])
        summary = manager.summaries()[0]
        self.assertEqual(summary["status"], "failed")
        self.assertIn("timed out", summary["error"])
        self.assertEqual(summary["tool_count"], 0)
        self.assertEqual(manager.tools, [])

    def test_crash_reports_exit_code_and_stderr(self) -> None:
        config, _ = self._config("crash.server", mode="crash", timeout=2.0)
        manager, warnings = self._start([config])

        self.assertEqual(len(warnings), 1)
        self.assertIn("crash.server", warnings[0])
        summary = manager.summaries()[0]
        self.assertEqual(summary["status"], "failed")
        self.assertIn("exit code 9", summary["error"])
        self.assertIn("fake server crashing", summary["error"])
        self.assertEqual(manager.tools, [])

    def test_missing_command_returns_warning_without_raising(self) -> None:
        missing = str(self.dir / "definitely-not-a-command")
        config, _ = self._config(
            "missing.server", command=missing, args=[], timeout=1.0
        )
        manager, warnings = self._start([config])
        self.assertEqual(len(warnings), 1)
        self.assertIn("missing.server", warnings[0])
        self.assertIn("command not found", warnings[0])
        summary = manager.summaries()[0]
        self.assertEqual(summary["status"], "failed")
        self.assertIn("command not found", summary["error"])
        self.assertEqual(manager.tools, [])
        self.assertEqual(manager.start(), warnings)

    def test_disabled_server_is_not_started(self) -> None:
        config, marker = self._config(
            "off.server",
            enabled=False,
            command=str(self.dir / "never-launched"),
            args=[],
        )
        manager, warnings = self._start([config])
        self.assertEqual(warnings, [])
        summary = manager.summaries()[0]
        self.assertEqual(summary["status"], "disabled")
        self.assertEqual(summary["tool_count"], 0)
        self.assertIsNone(summary["error"])
        self.assertEqual(manager.tools, [])
        self.assertFalse(marker.exists())
        manager.close()
        self.assertEqual(manager.summaries()[0]["status"], "disabled")

    def test_close_terminates_server_drains_threads_and_is_idempotent(self) -> None:
        config, marker = self._config("demo.server")
        manager, warnings = self._start([config])
        self.assertEqual(warnings, [])
        self.assertTrue(manager.tools)
        self.assertFalse(marker.exists())

        manager.close()
        manager.close()

        self.assertTrue(marker.is_file())
        self.assertEqual(manager.tools, [])
        summary = manager.summaries()[0]
        self.assertEqual(summary["status"], "closed")
        self.assertEqual(summary["tool_count"], 0)
        self.assertIsNone(summary["error"])
        self.assertFalse(
            any(
                thread.name.startswith("ohar-mcp-") and thread.is_alive()
                for thread in threading.enumerate()
            )
        )


if __name__ == "__main__":
    unittest.main()
