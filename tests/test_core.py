from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ohar.agent import Agent
from ohar.balancer import LaneManager
from ohar.config import (
    config_path,
    load_config,
    load_env_file,
    new_config,
    render_config,
    resolve_api_key,
    save_env_secret,
    write_config,
)
from ohar.sessions import SessionStore
from ohar.tools import Toolbox, default_tools
from ohar.subagents import make_task_tool
import json

from ohar.types import (
    AgentHooks,
    Config,
    ConfigError,
    LaneConfig,
    McpServerConfig,
    ProviderConfig,
    ProviderError,
    StreamEvent,
    ToolContext,
    UIConfig,
)

from tests.mock_provider import MockProvider

PROJECT = Path(__file__).resolve().parent.parent


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.previous = os.environ.get("OHAR_HOME")
        os.environ["OHAR_HOME"] = str(self.home)

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop("OHAR_HOME", None)
        else:
            os.environ["OHAR_HOME"] = self.previous
        self.temp.cleanup()

    def test_round_trip_and_example(self) -> None:
        config = new_config("deepseek", model="deepseek-chat")
        text = render_config(config)
        target = self.home / "config.toml"
        target.write_text(text, encoding="utf-8")
        loaded = load_config(target)
        self.assertEqual(loaded.default_lane, "main")
        self.assertEqual(loaded.lanes["main"].members, ["deepseek/deepseek-chat"])
        self.assertEqual(render_config(loaded), text)
        example = load_config(PROJECT / "config.example.toml")
        self.assertIn("main", example.lanes)
        self.assertGreaterEqual(len(example.providers), 2)

    def test_write_config_and_env_secret(self) -> None:
        config = new_config("openai", model="gpt-test")
        path = write_config(config)
        self.assertEqual(path, config_path())
        self.assertTrue(path.is_file())
        values = load_env_file(self.home / ".env")
        self.assertEqual(values, {})
        save_env_secret("OHAR_TEST_SECRET", "value-123")
        env_path = self.home / ".env"
        self.assertEqual(stat_mode(env_path), 0o600)
        load_env_file(env_path)
        self.assertEqual(os.environ.get("OHAR_TEST_SECRET"), "value-123")
        provider = next(iter(config.providers.values()))
        provider.api_key_env = "OHAR_TEST_SECRET"
        self.assertEqual(resolve_api_key(provider), "value-123")
        os.environ.pop("OHAR_TEST_SECRET", None)

    def test_literal_api_key_and_permissions(self) -> None:
        config = new_config(
            "openai", model="test-model", api_key="sk-local-secret", api_key_env=None
        )
        provider = next(iter(config.providers.values()))
        self.assertEqual(resolve_api_key(provider), "sk-local-secret")
        self.assertNotIn("sk-local-secret", repr(provider))
        target = self.home / "literal.toml"
        write_config(config, target)
        text = target.read_text()
        self.assertIn('api_key = "sk-local-secret"', text)
        loaded = load_config(target)
        self.assertEqual(resolve_api_key(next(iter(loaded.providers.values()))), "sk-local-secret")
        self.assertEqual(stat_mode(target), 0o600)

    def test_key_like_env_name_is_rejected(self) -> None:
        text = """
schema_version = 1
[providers.custom]
type = "openai"
base_url = "https://example.test/v1"
api_key_env = "SK-TEST-EXAMPLE-KEY"
[lanes.main]
members = ["custom/model"]
"""
        target = self.home / "badenv.toml"
        target.write_text(text)
        with self.assertRaises(ConfigError) as caught:
            load_config(target)
        self.assertIn("environment variable name", str(caught.exception))

    def test_rejects_unknown_provider_member(self) -> None:
        text = """
schema_version = 1
default_lane = "main"
[providers.openai]
type = "openai"
base_url = "https://api.openai.com/v1"
[lanes.main]
members = ["ghost/model"]
"""
        target = self.home / "bad.toml"
        target.write_text(text, encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(target)

    def test_advanced_sections_roundtrip(self) -> None:
        config = new_config("openai", model="test-model", api_key="sk-test")
        config.web.enabled = False
        config.web.require_approval = True
        config.web.allow_private = True
        config.web.search_backend = "searxng"
        config.web.searx_url = "http://localhost:8888"
        config.browser.headless = False
        config.browser.require_approval = True
        config.ui.syntax = False
        config.memory.max_records = 42
        config.memory.require_approval = False
        config.mcp_servers = [
            McpServerConfig(
                name="fake",
                command="python3",
                args=["server.py"],
                env={"MODE": "test"},
                cwd="/tmp",
                timeout=12.5,
                enabled=False,
            )
        ]
        target = self.home / "advanced.toml"
        write_config(config, target)
        loaded = load_config(target)
        self.assertFalse(loaded.web.enabled)
        self.assertTrue(loaded.web.require_approval)
        self.assertEqual(loaded.web.search_backend, "searxng")
        self.assertEqual(loaded.web.searx_url, "http://localhost:8888")
        self.assertFalse(loaded.browser.headless)
        self.assertTrue(loaded.browser.require_approval)
        self.assertFalse(loaded.ui.syntax)
        self.assertEqual(loaded.memory.max_records, 42)
        self.assertFalse(loaded.memory.require_approval)
        self.assertEqual(len(loaded.mcp_servers), 1)
        server = loaded.mcp_servers[0]
        self.assertEqual((server.name, server.command, server.args), ("fake", "python3", ["server.py"]))
        self.assertEqual(server.env, {"MODE": "test"})
        self.assertFalse(server.enabled)
        self.assertEqual(render_config(loaded), target.read_text())

    def test_advanced_section_validation(self) -> None:
        config = new_config("openai", model="m", api_key="sk-test")
        config.web.search_backend = "searxng"
        config.web.searx_url = None
        with self.assertRaises(ConfigError):
            write_config(config, self.home / "bad-web.toml")
        config = new_config("openai", model="m", api_key="sk-test")
        config.mcp_servers = [
            McpServerConfig(name="dup", command="one"),
            McpServerConfig(name="dup", command="two"),
        ]
        with self.assertRaises(ConfigError):
            write_config(config, self.home / "bad-mcp.toml")

    def test_rejects_newer_schema(self) -> None:
        target = self.home / "future.toml"
        target.write_text("schema_version = 99\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(target)


def stat_mode(path: Path) -> int:
    import stat

    return stat.S_IMODE(path.stat().st_mode)


class BalancerTests(unittest.TestCase):
    def config(self, strategy: str, cooldown: float = 0.0) -> Config:
        return Config(
            providers={
                "a": ProviderConfig(name="a", base_url="http://localhost:1/v1"),
                "b": ProviderConfig(name="b", base_url="http://localhost:1/v1"),
            },
            lanes={
                "main": LaneConfig(
                    name="main",
                    strategy=strategy,
                    members=["a/one", "b/two"],
                    cooldown_seconds=cooldown,
                )
            },
            ui=UIConfig(theme="mono"),
        )

    def manager(self, strategy: str) -> LaneManager:
        config = self.config(strategy, cooldown=0.08)
        return LaneManager(config, providers={"a": object(), "b": object()})

    def test_round_robin_and_weighted(self) -> None:
        lanes = self.manager("round_robin")
        first = [member.id for member in lanes.order("main")]
        second = [member.id for member in lanes.order("main")]
        self.assertEqual(first, ["a/one", "b/two"])
        self.assertEqual(second, ["b/two", "a/one"])
        weighted = self.manager("weighted")
        leads = {weighted.order("main")[0].id for _ in range(4)}
        self.assertEqual(len(leads), 2)

    def test_health_cooldown_reorders_and_recovers(self) -> None:
        lanes = self.manager("failover")
        member = lanes.members("main")[0]
        error = ProviderError("slow down", status=429, retryable=True, retry_after=0.08)
        lanes.report_failure(member, error)
        order = lanes.order("main")
        self.assertEqual(order[0].id, "b/two")
        status = {row["member"]: row for row in lanes.status()}
        self.assertEqual(status["a/one"]["health"], "cooling")
        time.sleep(0.1)
        status = {row["member"]: row for row in lanes.status()}
        self.assertEqual(status["a/one"]["health"], "ok")

    def test_acquire_capacity(self) -> None:
        config = self.config("failover")
        config.lanes["main"].max_inflight = 1
        lanes = LaneManager(config, providers={"a": object(), "b": object()})
        member = lanes.members("main")[0]
        self.assertTrue(lanes.acquire(member, timeout=0.1))
        self.assertFalse(lanes.acquire(member, timeout=0.05))
        lanes.release(member)
        self.assertTrue(lanes.acquire(member, timeout=0.05))
        lanes.release(member)


class SessionTests(unittest.TestCase):
    def test_session_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session = store.create(cwd="/tmp", model="openai/test", lane="main")
            session.add({"role": "user", "content": "Fix the parser"})
            session.add({"role": "assistant", "content": "On it."})
            store.save(session)
            loaded = store.load(session.meta.id)
            self.assertEqual(loaded.meta.title, "Fix the parser")
            self.assertEqual(len(loaded.messages), 2)
            self.assertEqual(store.latest().id, session.meta.id)
            self.assertEqual(store.resume("last").meta.id, session.meta.id)
            with self.assertRaises(ValueError):
                store.load("../escape")


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["OPENAI_API_KEY"] = "test-key"
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        (self.workspace / "sample.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_agent_streams_and_calls_tool(self) -> None:
        with MockProvider() as mock:
            config = new_config("openai", model="mock")
            config.providers["openai"].base_url = mock.base_url
            lanes = LaneManager(config)
            ctx = ToolContext(workspace=self.workspace, approval_mode="auto")
            toolbox = Toolbox(ctx)
            tool_ends: list[str] = []
            hooks = AgentHooks(
                tool_end=lambda name, result, seconds: tool_ends.append(name)
            )
            agent = Agent(config, lanes, toolbox, system_prompt="test", hooks=hooks)
            result = agent.run("USE_TOOL TOOL_NAME:read_file")
            self.assertIn("Mock final answer", result.text)
            self.assertEqual(tool_ends, ["read_file"])
            self.assertGreaterEqual(mock.requests_count, 2)

    def test_failover_tries_next_member(self) -> None:
        with MockProvider() as mock:
            config = Config(
                providers={
                    "bad": ProviderConfig(name="bad", base_url=mock.base_url, api_key_env="OPENAI_API_KEY"),
                    "good": ProviderConfig(name="good", base_url=mock.base_url, api_key_env="OPENAI_API_KEY"),
                },
                lanes={
                    "main": LaneConfig(
                        name="main", strategy="failover", members=["bad/mock-fail", "good/mock"]
                    )
                },
                ui=UIConfig(theme="mono"),
            )
            lanes = LaneManager(config)
            toolbox = Toolbox(ToolContext(workspace=self.workspace, approval_mode="auto"))
            agent = Agent(config, lanes, toolbox, system_prompt="test")
            result = agent.run("hello")
            self.assertEqual(result.model, "good/mock")
            self.assertTrue(result.text)

    def test_eight_subagents_run_in_parallel(self) -> None:
        barrier = threading.Barrier(8, timeout=3.0)

        class BarrierProvider:
            def stream_chat(self, model, messages, tools=None, **kwargs):
                barrier.wait()
                yield StreamEvent("text", text="independent report")
                yield StreamEvent("usage", usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
                yield StreamEvent("done", finish_reason="stop")

        with MockProvider() as mock:
            config = new_config("openai", model="mock")
            config.providers["openai"].base_url = mock.base_url
            providers = {"openai": BarrierProvider()}
            lanes = LaneManager(config, providers=providers)
            ctx = ToolContext(workspace=self.workspace, approval_mode="auto")
            events: list[dict] = []
            hooks = AgentHooks(subagent=events.append)
            task_tool = make_task_tool(
                config=config, lanes=lanes, workspace=self.workspace, hooks=hooks
            )
            toolbox = Toolbox(ctx, default_tools(ctx) + [task_tool])
            started = time.monotonic()
            result = toolbox.execute(
                "task",
                {"tasks": [{"title": f"job {i}", "prompt": f"inspect {i}"} for i in range(8)]},
            )
            elapsed = time.monotonic() - started
            self.assertTrue(result.ok, result.content)
            self.assertEqual(result.meta["completed"], 8)
            self.assertEqual(sum(1 for event in events if event["type"] == "start"), 8)
            self.assertEqual(sum(1 for event in events if event["type"] == "done"), 8)
            self.assertLess(elapsed, 2.8)

    def test_agent_triggers_parallel_task_tool(self) -> None:
        barrier = threading.Barrier(8, timeout=3.0)

        class TaskProvider:
            def stream_chat(self, model, messages, tools=None, **kwargs):
                system = "\n".join(
                    str(message.get("content", ""))
                    for message in messages
                    if message.get("role") == "system"
                )
                if "read-only research subagent" in system:
                    barrier.wait()
                    yield StreamEvent("text", text="subagent report")
                    yield StreamEvent("done", finish_reason="stop")
                    return
                if any(
                    message.get("role") == "tool" and message.get("name") == "task"
                    for message in messages
                ):
                    yield StreamEvent("text", text="Aggregated by parent.")
                    yield StreamEvent("done", finish_reason="stop")
                    return
                tasks = [
                    {"title": f"job {index}", "prompt": f"inspect area {index}"}
                    for index in range(8)
                ]
                yield StreamEvent(
                    "tool_call",
                    tool_call_id="call_task",
                    tool_name="task",
                    arguments=json.dumps({"tasks": tasks}),
                )
                yield StreamEvent("done", finish_reason="tool_calls")

        config = new_config("openai", model="test")
        lanes = LaneManager(config, providers={"openai": TaskProvider()})
        ctx = ToolContext(workspace=self.workspace, approval_mode="auto")
        events: list[dict] = []
        hooks = AgentHooks(subagent=events.append)
        task_tool = make_task_tool(
            config=config, lanes=lanes, workspace=self.workspace, hooks=hooks
        )
        toolbox = Toolbox(ctx, default_tools(ctx) + [task_tool])
        agent = Agent(config, lanes, toolbox, system_prompt="parent agent", hooks=hooks)
        result = agent.run("research eight independent areas")
        self.assertIn("Aggregated by parent.", result.text)
        self.assertEqual(sum(1 for event in events if event["type"] == "start"), 8)
        self.assertEqual(sum(1 for event in events if event["type"] == "done"), 8)
        tool_messages = [
            message
            for message in agent.messages
            if message.get("role") == "tool" and message.get("name") == "task"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("8/8 subagents completed", tool_messages[0]["content"])


class ToolSafetyTests(unittest.TestCase):
    def test_path_escape_and_denied_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            approvals: list[str] = []
            ctx = ToolContext(
                workspace=workspace,
                approval_mode="ask",
                confirm=lambda request: approvals.append(request.action) or False,
            )
            toolbox = Toolbox(ctx)
            escaped = toolbox.execute("read_file", {"path": "../secret"})
            self.assertFalse(escaped.ok)
            denied = toolbox.execute("write_file", {"path": "new.txt", "content": "x"})
            self.assertFalse(denied.ok)
            self.assertFalse((workspace / "new.txt").exists())
            self.assertEqual(approvals, ["write_file"])


if __name__ == "__main__":
    unittest.main()
