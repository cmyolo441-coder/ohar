from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from ohar.config import load_config, new_config, write_config
from ohar.types import McpServerConfig
from tests.mock_provider import MockProvider
from tests.test_mcp import FAKE_SERVER

PROJECT = Path(__file__).resolve().parent.parent
MAIN = PROJECT / "main.py"


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name) / "home"
        self.home.mkdir()
        self.mock = MockProvider()
        self.mock.start()
        self.addCleanup(self.mock.stop)
        self.addCleanup(self.temp.cleanup)
        previous = os.environ.get("OHAR_HOME")
        os.environ["OHAR_HOME"] = str(self.home)
        self.addCleanup(
            lambda: os.environ.__setitem__("OHAR_HOME", previous)
            if previous is not None
            else os.environ.pop("OHAR_HOME", None)
        )
        config = new_config("openai", model="mock")
        config.providers["openai"].base_url = self.mock.base_url
        write_config(config)

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["OHAR_HOME"] = str(self.home)
        env["OPENAI_API_KEY"] = "test-key"
        env.pop("COLUMNS", None)
        return env

    def run_cli(self, *args: str, input_text: str | None = None, timeout: float = 30.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(MAIN), *args],
            cwd=PROJECT,
            env=self.env(),
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    def test_one_shot_plain_and_session(self) -> None:
        result = self.run_cli("--plain", "hello cli")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# Mock Response", result.stdout)
        self.assertIn("hello cli", result.stdout)
        self.assertNotIn("\x1b", result.stdout)
        transcripts = list((self.home / "sessions").glob("*/transcript.jsonl"))
        self.assertEqual(len(transcripts), 1)
        lines = [json.loads(line) for line in transcripts[0].read_text().splitlines() if line]
        self.assertEqual(lines[0]["role"], "user")
        self.assertEqual(lines[0]["content"], "hello cli")
        self.assertEqual(lines[-1]["role"], "assistant")

    def test_piped_repl_commands_and_resume(self) -> None:
        first = self.run_cli(
            "--plain",
            input_text="/status\nhello in repl\n/sessions\n/quit\n",
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("hello in repl", first.stdout)
        self.assertIn("Sessions", first.stdout)
        session_id = next(
            path.name for path in (self.home / "sessions").iterdir() if path.is_dir()
        )
        resumed = self.run_cli("--plain", "--resume", "last", "after resume")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("after resume", resumed.stdout)
        transcript = self.home / "sessions" / session_id / "transcript.jsonl"
        lines = [json.loads(line) for line in transcript.read_text().splitlines() if line]
        self.assertEqual(lines[0]["content"], "hello in repl")
        self.assertEqual(lines[-2]["content"], "after resume")
        listed = self.run_cli("--plain", "--list-sessions")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn(session_id, listed.stdout)

    def test_tool_call_with_yolo(self) -> None:
        result = self.run_cli(
            "--plain",
            "--yolo",
            "USE_TOOL TOOL_NAME:shell",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Mock final answer", result.stdout)
        self.assertGreaterEqual(self.mock.requests_count, 2)

    def test_new_session_command(self) -> None:
        result = self.run_cli(
            "--plain",
            input_text="first\n/new\nsecond\n/quit\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Started a new conversation.", result.stdout)
        sessions = [path for path in (self.home / "sessions").iterdir() if path.is_dir()]
        self.assertEqual(len(sessions), 2)

    def test_read_only_removes_write_and_shell_tools(self) -> None:
        result = self.run_cli(
            "--plain",
            "--read-only",
            "USE_TOOL TOOL_NAME:shell",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Unknown tool 'shell'", result.stdout)
        self.assertIn("Mock final answer", result.stdout)

    def test_workflow_commands_run_in_repl(self) -> None:
        result = self.run_cli(
            "--plain",
            "--yolo",
            input_text=(
                "USE_TOOL TOOL_NAME:task\n"
                "/goals new Verify the pipeline\n"
                "/board\n/brief\n/journal\n/workspace\n/handoff\n/dag\n/tools\n/quit\n"
            ),
            timeout=90.0,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stdout)
        self.assertNotIn("failed:", result.stdout.lower())
        self.assertIn("recorded", result.stdout)
        self.assertIn("Journal ·", result.stdout)
        self.assertIn("goal_create", result.stdout)
        self.assertIn("capabilities", result.stdout)
        self.assertIn("workspace_status", result.stdout)
        self.assertIn("unit_declare", result.stdout)

    def test_read_only_still_lists_read_workflow_tools(self) -> None:
        result = self.run_cli("--plain", "--read-only", input_text="/tools\n/quit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("capabilities", result.stdout)
        self.assertNotIn("goal_create", result.stdout)
        self.assertNotIn("unit_declare", result.stdout)
        self.assertNotIn("write_file", result.stdout)

    def test_headless_approval_is_denied(self) -> None:
        result = self.run_cli(
            "--plain",
            "USE_TOOL TOOL_NAME:shell",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Permission denied", result.stdout)
        self.assertNotIn("printf mock", result.stdout)

    def test_eight_subagents_through_cli(self) -> None:
        result = self.run_cli(
            "--plain",
            "USE_TOOL TOOL_NAME:task",
            timeout=45.0,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Mock final answer", result.stdout)
        self.assertGreaterEqual(self.mock.requests_count, 10)
        transcript = next((self.home / "sessions").glob("*/transcript.jsonl"))
        lines = [json.loads(line) for line in transcript.read_text().splitlines() if line]
        task_results = [
            message
            for message in lines
            if message.get("role") == "tool" and message.get("name") == "task"
        ]
        self.assertEqual(len(task_results), 1)
        self.assertIn("8/8 subagents completed", task_results[0]["content"])
        self.assertEqual(task_results[0]["content"].count("status: completed"), 8)

    def test_advanced_commands_and_plan_tool(self) -> None:
        result = self.run_cli(
            "--plain",
            input_text="/tools\n/plan\n/mcp\n/memory\n/theme ember\n/quit\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("web_fetch", result.stdout)
        self.assertIn("update_plan", result.stdout)
        self.assertIn("remember", result.stdout)
        self.assertIn("No active plan", result.stdout)
        self.assertIn("No MCP servers configured", result.stdout)
        self.assertIn("No saved memories", result.stdout)
        self.assertIn("Theme set to ember", result.stdout)
        planned = self.run_cli("--plain", "USE_TOOL TOOL_NAME:update_plan")
        self.assertEqual(planned.returncode, 0, planned.stderr)
        self.assertIn("Plan · 1/3 done", planned.stdout)
        self.assertIn("implement the change", planned.stdout)
        self.assertIn("Mock final answer", planned.stdout)

    def test_new_session_clears_plan(self) -> None:
        result = self.run_cli(
            "--plain",
            input_text="USE_TOOL TOOL_NAME:update_plan\n/plan\n/new\n/plan\n/quit\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Plan · 1/3 done", result.stdout)
        after_new = result.stdout.split("Started a new conversation.", 1)[1]
        self.assertIn("No active plan", after_new)

    def test_memory_persists_and_can_be_cleared(self) -> None:
        remembered = self.run_cli(
            "--plain",
            "--yolo",
            "USE_TOOL TOOL_NAME:remember",
        )
        self.assertEqual(remembered.returncode, 0, remembered.stderr)
        self.assertIn("Mock final answer", remembered.stdout)
        listed = self.run_cli("--plain", input_text="/memory mock\n/quit\n")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("Mock memory", listed.stdout)
        self.assertIn("mock, tests", listed.stdout)
        cleared = self.run_cli(
            "--plain",
            "--yolo",
            input_text="/memory clear\n/quit\n",
        )
        self.assertEqual(cleared.returncode, 0, cleared.stderr)
        self.assertIn("Cleared 1 memory record", cleared.stdout)
        empty = self.run_cli("--plain", input_text="/memory\n/quit\n")
        self.assertIn("No saved memories", empty.stdout)

    def test_mcp_startup_status_tools_and_cleanup(self) -> None:
        script = self.home / "fake_mcp.py"
        script.write_text(FAKE_SERVER, encoding="utf-8")
        marker = self.home / "mcp-closed.txt"
        config = load_config()
        config.mcp_servers = [
            McpServerConfig(
                name="fixture",
                command=sys.executable,
                args=[str(script), "--marker", str(marker)],
                timeout=3.0,
            )
        ]
        write_config(config)
        result = self.run_cli(
            "--plain",
            input_text="/mcp\n/tools\n/quit\n",
            timeout=30.0,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ready", result.stdout)
        self.assertIn("mcp__fixture__echo", result.stdout)
        self.assertIn("mcp__fixture__write", result.stdout)
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(marker.exists(), "MCP server was not terminated on exit")

    def test_quitting_without_messages_leaves_no_empty_session(self) -> None:
        result = self.run_cli("--plain", input_text="/quit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        sessions = [
            path
            for path in (self.home / "sessions").glob("*")
            if path.is_dir()
        ] if (self.home / "sessions").exists() else []
        self.assertEqual(sessions, [])

    def test_resume_last_inside_repl_skips_fresh_session(self) -> None:
        result = self.run_cli(
            "--plain",
            input_text="first conversation\n/new\n/resume last\n/quit\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Resumed", result.stdout)
        self.assertIn("· first conversation", result.stdout)
        sessions = [path for path in (self.home / "sessions").iterdir() if path.is_dir()]
        self.assertEqual(len(sessions), 1)

    def test_missing_config_is_actionable(self) -> None:
        (self.home / "config.toml").unlink()
        result = self.run_cli("hi")
        self.assertEqual(result.returncode, 2)
        self.assertIn("No config found", result.stderr)
        self.assertIn("config.example.toml", result.stderr)

    def test_bad_lane_and_model(self) -> None:
        lane = self.run_cli("--lane", "ghost", "hi")
        self.assertEqual(lane.returncode, 2)
        self.assertIn("Unknown lane", lane.stderr)
        model = self.run_cli("--model", "ghost/model", "hi")
        self.assertEqual(model.returncode, 2)
        self.assertIn("not a unique member", model.stderr)


if __name__ == "__main__":
    unittest.main()
