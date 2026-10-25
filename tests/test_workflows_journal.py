from __future__ import annotations

import json
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from ohar.types import AgentHooks, ToolResult
from ohar.workflows.journal import (
    AVAILABILITY,
    EVENT_NAMES,
    JOURNAL_NAME,
    UNKNOWN_AGE_SECONDS,
    ActivityJournal,
    JournalEvent,
    args_digest,
    wrap_hooks,
)
from ohar.workflows.store import WorkflowError

_BASE_EPOCH = 1_800_000_000.0


class JournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.ws = self.base / "ws"
        self.ws.mkdir()
        self.tick = [_BASE_EPOCH]
        self.journal = self.make_journal()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_journal(self, workspace: Path | None = None, **kwargs: object) -> ActivityJournal:
        return ActivityJournal(
            self.base / "workflows",
            workspace or self.ws,
            clock=lambda: self.tick[0],
            **kwargs,  # type: ignore[arg-type]
        )

    def advance(self, seconds: float) -> None:
        self.tick[0] += seconds

    def test_layout_is_private_and_workspace_scoped(self) -> None:
        self.journal.record("session_started", session_id="s1")
        self.assertEqual(self.journal.path.name, f"{JOURNAL_NAME}.jsonl")
        self.assertEqual(stat.S_IMODE(self.journal.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.journal.path.parent.stat().st_mode), 0o700)
        elsewhere = self.make_journal(workspace=self.base / "other")
        elsewhere.record("session_started", session_id="s1")
        self.assertNotEqual(self.journal.path.parent, elsewhere.path.parent)
        self.assertEqual(len(self.journal.recent()), 1)
        self.assertEqual(len(elsewhere.recent()), 1)

    def test_event_vocabulary_is_closed(self) -> None:
        self.assertEqual(
            set(EVENT_NAMES),
            {
                "tool_observed",
                "turn_completed",
                "approval_answered",
                "gate_opened",
                "gate_answered",
                "unit_declared",
                "unit_observed",
                "goal_updated",
                "session_started",
                "session_ended",
                "subagent_completed",
                "loop_cycle",
            },
        )
        self.assertEqual(AVAILABILITY, ("observed", "unavailable"))
        self.assertEqual(self.journal.age_seconds("nothing-ever-seen"), UNKNOWN_AGE_SECONDS)
        self.assertIsNone(self.journal.observed("nothing-ever-seen"))
        with self.assertRaises(WorkflowError):
            self.journal.record("model_claimed_success")
        with self.assertRaises(WorkflowError):
            self.journal.record("tool_observed", ok="yes")  # type: ignore[arg-type]

    def test_record_keeps_metadata_only(self) -> None:
        record = self.journal.record(
            "gate_opened",
            subject="write:src/app.py",
            session_id="s1",
            detail={"question": "allow write?", "token": "abc", "nested": {"x": 1}, "": "drop"},
        )
        self.assertEqual(record["availability"], AVAILABILITY[0])
        self.assertEqual(record["privacy"], "metadata_only")
        self.assertEqual(record["workspace"], self.journal.scope_key)
        self.assertEqual(set(record["detail"]), {"question", "token"})
        raw = self.journal.path.read_text(encoding="utf-8")
        self.assertIn("allow write?", raw)
        self.assertNotIn("nested", raw)

    def test_tool_observation_never_stores_arguments_or_output(self) -> None:
        result = ToolResult.failure("Traceback: SECRET-TOKEN leaked", exit_code=2, denied=True, noise="ignore")
        record = self.journal.record_tool(
            "shell", {"cmd": "echo SECRET-TOKEN"}, result, seconds=0.123456, session_id="s1"
        )
        raw = self.journal.path.read_text(encoding="utf-8")
        self.assertNotIn("SECRET-TOKEN", raw)
        detail = record["detail"]
        self.assertEqual(detail["exit_code"], 2)
        self.assertTrue(detail["denied"])
        self.assertNotIn("noise", detail)
        self.assertEqual(detail["args_digest"], args_digest("shell", {"cmd": "echo SECRET-TOKEN"}))
        self.assertEqual(record["subject"], "shell")
        self.assertFalse(record["ok"])

    def test_args_digest_is_order_independent_and_call_specific(self) -> None:
        self.assertEqual(args_digest("read", {"a": 1, "b": 2}), args_digest("read", {"b": 2, "a": 1}))
        self.assertNotEqual(args_digest("read", {"a": 1}), args_digest("read", {"a": 2}))
        self.assertNotEqual(args_digest("read", {"a": 1}), args_digest("write", {"a": 1}))
        self.assertEqual(len(args_digest("read", {})), 16)

    def test_turn_and_subagent_receipts(self) -> None:
        turn = self.journal.record_turn(
            model="ollama/qwen3:8b", lane="subagents", input_tokens=12, output_tokens=-3, seconds=1.5, session_id="s1"
        )
        self.assertEqual(turn["detail"]["input_tokens"], 12)
        self.assertEqual(turn["detail"]["output_tokens"], 0)
        self.assertEqual(turn["subject"], "ollama/qwen3:8b")
        self.assertEqual(self.journal.record_subagent({"type": "tool", "id": "1", "name": "read"}), {})
        self.assertEqual(len(self.journal.recent()), 1)
        done = self.journal.record_subagent(
            {"type": "done", "id": "2", "title": "Audit parser", "seconds": 4.0, "chars": 90}, session_id="s1"
        )
        self.assertTrue(done["ok"])
        failed = self.journal.record_subagent({"status": "failed", "id": "3", "error": "boom " * 200}, session_id="s1")
        self.assertFalse(failed["ok"])
        self.assertLessEqual(len(failed["detail"]["error"]), 160)
        self.assertIn("error_digest", failed["detail"])

    def test_recent_filters_and_limit(self) -> None:
        self.journal.record_tool("read", {"p": "a"}, ToolResult.success("ok"), seconds=0.1, session_id="s1")
        self.advance(5)
        self.journal.record_tool("write", {"p": "b"}, ToolResult.failure("no"), seconds=0.1, session_id="s2")
        self.advance(5)
        self.journal.record("unit_observed", subject="u1", ok=True, session_id="s1")
        self.assertEqual([item.event for item in self.journal.recent()], ["tool_observed", "tool_observed", "unit_observed"])
        self.assertEqual([item.event for item in self.journal.recent(limit=2)][-1], "unit_observed")
        self.assertEqual([item.subject for item in self.journal.recent(event="tool_observed")], ["read", "write"])
        self.assertEqual([item.subject for item in self.journal.recent(subject="read")], ["read"])
        self.assertEqual(len(self.journal.recent(session_id="s1")), 2)
        self.assertEqual([item.subject for item in self.journal.recent(since_seconds=6)], ["write", "u1"])
        self.assertEqual(self.journal.recent(limit=0), [])

    def test_observed_and_age_track_the_latest_matching_observation(self) -> None:
        self.journal.record_tool("tests", {}, ToolResult.failure("fail"), seconds=0.1, session_id="s1")
        self.advance(30)
        self.journal.record_tool("tests", {}, ToolResult.success("pass"), seconds=0.2, session_id="s1")
        latest = self.journal.observed("tests")
        self.assertIsNotNone(latest)
        self.assertTrue(latest.ok)
        self.assertEqual(latest.session_id, "s1")
        self.assertFalse(self.journal.observed("tests", ok=False).ok)
        self.assertIsNone(self.journal.observed("tests", event="turn_completed"))
        self.assertEqual(self.journal.age_seconds("tests", now=self.tick[0]), 0)
        self.assertEqual(self.journal.age_seconds("tests", now=self.tick[0] + 120), 120)
        self.assertEqual(self.journal.age_seconds("tests"), 0)
        self.assertIsNone(self.journal.observed(""))

    def test_events_from_the_future_are_not_trusted(self) -> None:
        self.tick[0] = _BASE_EPOCH + 100_000
        self.journal.record("loop_cycle", subject="goal-1", ok=True)
        self.tick[0] = _BASE_EPOCH
        self.assertIsNone(self.journal.observed("goal-1", ok=True))
        self.assertEqual(self.journal.age_seconds("goal-1"), UNKNOWN_AGE_SECONDS)
        self.assertEqual(self.journal.counts(), {})

    def test_counts_never_zero_fill(self) -> None:
        self.journal.record("session_started", session_id="s1")
        self.advance(60)
        self.journal.record_tool("read", {}, ToolResult.success("x"), seconds=0.1)
        self.assertEqual(self.journal.counts(), {"session_started": 1, "tool_observed": 1})
        self.assertEqual(self.journal.counts(within_seconds=30), {"tool_observed": 1})
        with self.assertRaises(WorkflowError):
            self.journal.record("model_claimed_success")
        self.assertEqual(self.journal.counts(), {"session_started": 1, "tool_observed": 1})

    def test_counts_include_failed_observations(self) -> None:
        self.journal.record_tool("read", {}, ToolResult.failure("nope"), seconds=0.1)
        self.assertEqual(self.journal.counts(), {"tool_observed": 1})

    def test_render_rows_match_headers_and_are_display_safe(self) -> None:
        self.journal.record_tool("read\x1b[31m", {"p": "a"}, ToolResult.success("ok"), seconds=0.1)
        self.journal.record("gate_answered", subject="\x00evil|subject", ok=True)
        headers = self.journal.headers()
        rows = self.journal.render_rows(limit=5)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(len(row), len(headers))
            for cell in row:
                self.assertNotIn("\n", cell)
                self.assertNotIn("\x1b", cell)
                self.assertNotIn("\x00", cell)
        self.assertEqual(rows[0][2], "evil|subject")
        self.assertEqual(rows[0][5], AVAILABILITY[0])
        self.assertEqual(self.journal.render_rows(limit=0), [])

    def test_max_events_bounds_the_tail(self) -> None:
        tight = self.make_journal(max_events=3)
        for index in range(10):
            tight.record("loop_cycle", subject=f"cycle-{index}", ok=True)
        self.assertEqual([item.subject for item in tight.recent(limit=50)], ["cycle-7", "cycle-8", "cycle-9"])
        self.assertEqual(tight.counts(), {"loop_cycle": 3})
        self.assertEqual(tight.max_events, 3)
        self.assertEqual(ActivityJournal(self.base / "w", self.ws, max_events=99_999).max_events, 2000)
        self.assertEqual(ActivityJournal(self.base / "w", self.ws, max_events=True).max_events, 2000)

    def test_foreign_workspace_records_are_ignored(self) -> None:
        self.journal.record("session_started", session_id="s1")
        forged = json.loads(self.journal.path.read_text(encoding="utf-8").splitlines()[0])
        forged["workspace"] = "0" * 16
        with open(self.journal.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(forged) + "\n")
        self.assertEqual(len(self.journal.recent()), 1)

    def test_event_round_trip_is_tolerant(self) -> None:
        record = self.journal.record("unit_declared", subject="u1", ok=False, session_id="s1", detail={"check": "pytest"})
        event = JournalEvent.from_dict(record)
        self.assertEqual(event.to_dict()["detail"], {"check": "pytest"})
        self.assertEqual(event.to_dict()["at"], record["at"])
        self.assertFalse(event.ok)
        broken = JournalEvent.from_dict({"event": "made_up", "ok": "yes", "detail": "nope", "subject": 3})
        self.assertEqual(broken.event, "")
        self.assertIsNone(broken.ok)
        self.assertEqual(broken.detail, {})
        self.assertEqual(broken.subject, "3")
        self.assertEqual(JournalEvent.from_dict({}).at, "")  # type: ignore[arg-type]

    def test_wrap_hooks_records_and_delegates(self) -> None:
        seen: list[tuple[str, object]] = []
        base = AgentHooks(
            tool_start=lambda name, args: seen.append(("start", name)),
            tool_end=lambda name, result, seconds: seen.append(("end", result.ok)),
            notice=lambda text, level: seen.append(("notice", text)),
            usage=lambda data: seen.append(("usage", data)),
            turn_end=lambda data: seen.append(("turn", data["model"])),
            subagent=lambda data: seen.append(("sub", data["id"])),
        )
        wrapped = wrap_hooks(base, self.journal, session_id="s9")
        wrapped.tool_start("read_file", {"path": "private.txt"})
        wrapped.tool_end("read_file", ToolResult.success("contents"), 0.25)
        wrapped.notice("hello", "info")
        wrapped.usage({"input_tokens": 3})
        wrapped.turn_end({"model": "p/m", "lane": "main", "elapsed_s": 2.0, "input_tokens": 5, "output_tokens": 6})
        wrapped.subagent({"type": "done", "id": "7", "title": "Worker", "seconds": 1.0, "chars": 12})
        self.assertEqual(
            seen,
            [("start", "read_file"), ("end", True), ("notice", "hello"), ("usage", {"input_tokens": 3}), ("turn", "p/m"), ("sub", "7")],
        )
        events = self.journal.recent(limit=10, session_id="s9")
        self.assertEqual([item.event for item in events], ["tool_observed", "turn_completed", "subagent_completed"])
        tool = events[0]
        self.assertEqual(tool.subject, "read_file")
        self.assertEqual(tool.detail["args_digest"], args_digest("read_file", {"path": "private.txt"}))
        self.assertNotIn("private.txt", self.journal.path.read_text(encoding="utf-8"))

    def test_wrap_hooks_survives_a_broken_journal(self) -> None:
        class Broken(ActivityJournal):
            def record_tool(self, *args: object, **kwargs: object) -> dict:
                raise WorkflowError("store is full")

            def record_turn(self, **kwargs: object) -> dict:
                raise OSError("disk gone")

        broken = Broken(self.base / "workflows", self.ws, clock=lambda: self.tick[0])
        fired: list[str] = []
        wrapped = wrap_hooks(
            AgentHooks(tool_end=lambda *a: fired.append("end"), turn_end=lambda data: fired.append("turn")),
            broken,
        )
        wrapped.tool_end("read", ToolResult.success("x"), 0.1)
        wrapped.turn_end({"model": "m", "lane": "main"})
        self.assertEqual(fired, ["end", "turn"])
        self.assertFalse(broken.path.exists())

    def test_concurrent_recording_keeps_every_line_parseable(self) -> None:
        barrier = threading.Barrier(4)

        def worker(index: int) -> None:
            barrier.wait()
            for step in range(10):
                self.journal.record("tool_observed", subject=f"tool-{index}", ok=True, detail={"step": step})

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        lines = self.journal.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 40)
        self.assertTrue(all(isinstance(json.loads(line), dict) for line in lines))
        self.assertEqual(self.journal.counts()["tool_observed"], 40)


if __name__ == "__main__":
    unittest.main()
