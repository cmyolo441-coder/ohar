from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ohar.balancer import LaneManager
from ohar.config import new_config
from ohar.subagents import make_task_tool
from ohar.terminal import make_theme, strip_ansi
from ohar.types import AgentHooks, ProviderError, StreamEvent, ToolContext

from ohar.tui.cells import cell_width
from ohar.tui.roster import (
    DONE,
    ERROR,
    ROSTER_CAP,
    RUNNING,
    TITLE_CHARS,
    TOOL_TRAIL,
    SubagentRoster,
    TaskState,
)

PLAIN = make_theme("aurora", color=False)
COLOURED = make_theme("ember", color=True, depth="truecolor")


def events() -> list[dict[str, object]]:
    return [
        {"type": "start", "id": "1", "title": "inspect repo", "lane": "subagents", "model": "gpt-4o"},
        {"type": "start", "id": "2", "title": "read the docs", "lane": "subagents", "model": "gpt-4o-mini"},
        {"type": "tool", "id": "1", "name": "read_file"},
        {"type": "tool", "id": "1", "name": "grep"},
        {"type": "done", "id": "1", "title": "inspect repo", "seconds": 4.2, "chars": 1234},
        {"type": "error", "id": "2", "title": "read the docs", "error": "timed out", "seconds": 30.5},
    ]


class FoldTests(unittest.TestCase):
    def test_a_whole_run_folds(self) -> None:
        roster = SubagentRoster()
        for event in events():
            self.assertTrue(roster.apply(event))
        self.assertEqual(roster.total, 2)
        self.assertEqual(roster.running, 0)
        first = roster.task("1")
        self.assertIsInstance(first, TaskState)
        assert first is not None
        self.assertEqual(
            (first.status, first.title, first.lane, first.model, first.seconds, first.chars),
            (DONE, "inspect repo", "subagents", "gpt-4o", 4.2, 1234),
        )
        self.assertEqual(first.tools, ("read_file", "grep"))
        self.assertEqual(first.error, "")
        second = roster.task("2")
        assert second is not None
        self.assertEqual((second.status, second.seconds, second.chars, second.error), (ERROR, 30.5, 0, "timed out"))
        self.assertEqual([task.id for task in roster.tasks()], ["1", "2"])

    def test_running_tasks_count_and_time_themselves(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": "work"})
        self.assertEqual(roster.running, 1)
        self.assertEqual(roster.total, 1)
        task = roster.task("1")
        assert task is not None
        self.assertEqual(task.status, RUNNING)
        self.assertGreaterEqual(task.seconds, 0.0)
        self.assertLess(task.seconds, 60.0)  # measured from the frame that showed life
        self.assertEqual(task.tools, ())

    def test_a_tool_frame_without_a_start_still_gets_a_row(self) -> None:
        roster = SubagentRoster()
        self.assertTrue(roster.apply({"type": "tool", "id": "7", "name": "read_file"}))
        task = roster.task("7")
        assert task is not None
        self.assertEqual((task.status, task.title), (RUNNING, ""))
        self.assertEqual(task.tools, ("read_file",))

    def test_the_tool_trail_is_a_ring(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": "work"})
        for index in range(TOOL_TRAIL + 4):
            self.assertTrue(roster.apply({"type": "tool", "id": "1", "name": f"call{index}"}))
        task = roster.task("1")
        assert task is not None
        self.assertEqual(len(task.tools), TOOL_TRAIL)
        self.assertEqual(task.tools[0], f"call{4}")
        self.assertEqual(task.tools[-1], f"call{TOOL_TRAIL + 3}")

    def test_an_unnamed_tool_is_still_a_sign_of_life(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1"})
        self.assertTrue(roster.apply({"type": "tool", "id": "1"}))
        self.assertTrue(roster.apply({"type": "tool", "id": "1", "name": 42}))
        task = roster.task("1")
        assert task is not None
        self.assertEqual(task.tools, ("tool", "tool"))

    def test_a_finished_task_is_frozen(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": "work"})
        roster.apply({"type": "tool", "id": "1", "name": "read_file"})
        roster.apply({"type": "done", "id": "1", "seconds": 2.0, "chars": 10})
        self.assertFalse(roster.apply({"type": "done", "id": "1", "seconds": 99.0, "chars": 99}))
        self.assertFalse(roster.apply({"type": "error", "id": "1", "error": "late"}))
        self.assertFalse(roster.apply({"type": "tool", "id": "1", "name": "late"}))
        task = roster.task("1")
        assert task is not None
        self.assertEqual((task.status, task.seconds, task.chars, task.error), (DONE, 2.0, 10, ""))
        self.assertEqual(task.tools, ("read_file",))

    def test_a_verdict_without_a_row_lands_nowhere(self) -> None:
        roster = SubagentRoster()
        self.assertFalse(roster.apply({"type": "done", "id": "9", "seconds": 1.0}))
        self.assertFalse(roster.apply({"type": "error", "id": "9", "error": "boom"}))
        self.assertEqual(roster.total, 0)

    def test_a_reused_id_starts_over(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": "first", "model": "a"})
        roster.apply({"type": "tool", "id": "1", "name": "read_file"})
        roster.apply({"type": "done", "id": "1", "seconds": 1.0, "chars": 40})
        self.assertTrue(roster.apply({"type": "start", "id": "1", "title": "second", "lane": "main"}))
        task = roster.task("1")
        assert task is not None
        self.assertEqual((task.status, task.title, task.lane, task.model), (RUNNING, "second", "main", ""))
        self.assertEqual((task.chars, task.tools, task.error), (0, (), ""))
        self.assertLess(task.seconds, 60.0)  # the finished run's figures are gone with the old row

    def test_missing_and_mistyped_fields_are_tolerated(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": None, "lane": 5, "model": ["gpt"]})
        self.assertTrue(roster.apply({"type": "done", "id": "1", "seconds": "soon", "chars": "many"}))
        self.assertFalse(roster.apply({"type": "done", "id": "2"}))  # a verdict on a task never shown
        task = roster.task("1")
        assert task is not None
        self.assertEqual(task.title, "")
        self.assertEqual((task.lane, task.model), ("", ""))
        self.assertEqual(task.status, DONE)
        self.assertEqual(task.chars, 0)
        self.assertGreaterEqual(task.seconds, 0.0)
        self.assertEqual(roster.total, 1)

    def test_numbers_are_bounded(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1"})
        roster.apply({"type": "done", "id": "1", "seconds": 10**12, "chars": 10**20})
        task = roster.task("1")
        assert task is not None
        self.assertLess(task.seconds, 60.0)  # the wild figure was refused, the measured one kept
        self.assertEqual(task.chars, 0)
        roster.apply({"type": "start", "id": "2"})
        roster.apply({"type": "done", "id": "2", "seconds": True, "chars": -3})
        task = roster.task("2")
        assert task is not None
        self.assertEqual(task.chars, 0)
        roster.apply({"type": "start", "id": "3"})
        roster.apply({"type": "error", "id": "3", "seconds": float("nan"), "error": "x"})
        task = roster.task("3")
        assert task is not None
        self.assertGreaterEqual(task.seconds, 0.0)

    def test_untrusted_text_is_cleaned_and_cut(self) -> None:
        roster = SubagentRoster()
        noisy = "\x1b[31mmulti\nline   title\x07" + "x" * 400
        self.assertTrue(roster.apply({"type": "start", "id": "1", "title": noisy}))
        self.assertTrue(roster.apply({"type": "error", "id": "1", "error": "\x1b]8;;http://e\x07bad\x1b[0m"}))
        task = roster.task("1")
        assert task is not None
        self.assertNotIn("\x1b", task.title)
        self.assertNotIn("\n", task.title)
        self.assertLessEqual(len(task.title), TITLE_CHARS)
        self.assertEqual(task.error, "bad")
        self.assertEqual(strip_ansi(task.title), task.title)

    def test_junk_events_change_nothing(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": "work"})
        junk = [
            {},
            {"type": "unknown", "id": "2"},
            {"type": 7, "id": "2"},
            {"type": "start"},
            {"type": "start", "id": ""},
            {"type": "start", "id": "   "},
            {"type": "start", "id": 3},
            {"type": "start", "id": "x" * 65},
            {"type": "done", "id": "\x1b"},
            None,
            "start",
            [],
        ]
        for event in junk:
            self.assertFalse(roster.apply(event))  # type: ignore[arg-type]
        self.assertEqual(roster.total, 1)
        self.assertEqual(roster.running, 1)

    def test_reset_forgets_everything(self) -> None:
        roster = SubagentRoster()
        for event in events():
            roster.apply(event)
        roster.reset()
        self.assertEqual((roster.total, roster.running), (0, 0))
        self.assertEqual((roster.tasks(), roster.task("1")), ((), None))
        self.assertEqual(roster.snapshot()["total"], 0)

    def test_the_roster_is_bounded(self) -> None:
        roster = SubagentRoster()
        for index in range(ROSTER_CAP + 5):
            roster.apply({"type": "start", "id": str(index)})
        self.assertEqual(roster.total, ROSTER_CAP)
        self.assertEqual(roster.evicted, 5)
        self.assertIsNone(roster.task("0"))
        self.assertIsNotNone(roster.task(str(ROSTER_CAP + 4)))
        roster.apply({"type": "done", "id": str(ROSTER_CAP + 4), "seconds": 1.0})
        roster.apply({"type": "start", "id": "newcomer"})
        self.assertIsNone(roster.task(str(ROSTER_CAP + 4)))  # a finished task leaves first


class RowTests(unittest.TestCase):
    def filled(self) -> SubagentRoster:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": "inspect repo 👨‍👩‍👧‍👦", "lane": "subagents", "model": "gpt-4o"})
        roster.apply({"type": "tool", "id": "1", "name": "read_file"})
        roster.apply({"type": "start", "id": "2", "title": "read the docs", "lane": "subagents"})
        roster.apply({"type": "done", "id": "2", "seconds": 8.0, "chars": 42})
        roster.apply({"type": "start", "id": "3", "title": "check the tests", "lane": "main"})
        roster.apply({"type": "error", "id": "3", "error": "lane unavailable", "seconds": 1.5})
        return roster

    @staticmethod
    def plain_row(row: str) -> str:
        return strip_ansi(row)

    def test_rows_lead_with_running_and_end_with_done(self) -> None:
        rows = self.filled().lines(80, PLAIN)
        self.assertEqual(len(rows), 3)
        self.assertIn("inspect repo", rows[0])  # still running: first
        self.assertIn("read_file", rows[0])
        self.assertIn("check the tests", rows[1])  # failed next, with its reason
        self.assertIn("lane unavailable", rows[1])
        self.assertIn("read the docs", rows[2])  # done last, with what it wrote
        self.assertIn("8.00s", rows[2])
        self.assertIn("42c", rows[2])

    def test_a_plain_theme_writes_no_ansi_and_a_coloured_one_only_adds_it(self) -> None:
        plain = self.filled().lines(80, PLAIN)
        for row in plain:
            self.assertEqual(self.plain_row(row), row)
        painted = self.filled().lines(80, COLOURED)
        self.assertEqual([self.plain_row(row) for row in painted], plain)  # colour never changes the text
        self.assertTrue(all("\x1b[" in row for row in painted))
        collapsed = self.filled().lines(80, COLOURED, cap=1)[1]
        self.assertIn("\x1b[2m", collapsed)  # the row that counts the rest is dimmed

    def test_ascii_glyphs_replace_the_marks_and_the_separators(self) -> None:
        rows = self.filled().lines(80, PLAIN, glyphs="ascii")
        self.assertTrue(rows[0].startswith("  o "))
        self.assertTrue(rows[1].startswith("  x "))
        self.assertTrue(rows[2].startswith("  + "))
        self.assertIn(" | ", rows[2])
        self.assertNotIn("·", "".join(rows))
        unicode_rows = self.filled().lines(80, PLAIN)
        self.assertTrue(unicode_rows[0].startswith("  ● "))
        self.assertIn("·", unicode_rows[2])

    def test_every_row_fits_every_width(self) -> None:
        roster = self.filled()
        for width in range(1, 90):
            for theme in (PLAIN, COLOURED):
                for glyphs in ("unicode", "ascii"):
                    for row in roster.lines(width, theme, cap=3, glyphs=glyphs):
                        self.assertLessEqual(cell_width(self.plain_row(row)), width, (width, glyphs, row))

    def test_a_narrow_row_keeps_the_mark_and_the_title_room(self) -> None:
        roster = self.filled()
        self.assertTrue(roster.lines(24, PLAIN)[0].startswith("  ● "))
        self.assertIn("…", roster.lines(12, PLAIN)[0])
        self.assertNotIn("…", roster.lines(80, PLAIN)[0])

    def test_the_cap_collapses_the_rest_into_one_row(self) -> None:
        roster = self.filled()
        rows = roster.lines(80, PLAIN, cap=2)
        self.assertEqual(len(rows), 3)
        self.assertIn("+1 more", rows[2])
        self.assertIn("1 done", rows[2])
        collapsed = roster.lines(80, PLAIN, cap=1)[1]
        self.assertIn("+2 more", collapsed)
        self.assertIn("1 failed", collapsed)
        self.assertIn("1 done", collapsed)
        self.assertEqual(roster.lines(80, PLAIN, cap=0), [])
        self.assertEqual(roster.lines(0, PLAIN), [])
        self.assertEqual(roster.lines(-5, PLAIN, cap=4), [])
        self.assertEqual(SubagentRoster().lines(80, PLAIN), [])

    def test_an_untitled_task_is_named_and_its_figures_shrink_gracefully(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "model": "a-very-long-model-name-for-some-one"})
        roster.apply({"type": "tool", "id": "1", "name": "read_file"})
        narrow = roster.lines(30, PLAIN)[0]
        self.assertIn("subagent", narrow)
        self.assertNotIn("a-very-long-model-name-for-some-one", narrow)  # the figures give way before the title
        self.assertIn("1 tool", narrow)
        self.assertIn("read_file", roster.lines(32, PLAIN)[0])
        wide = roster.lines(78, PLAIN)[0]
        self.assertIn("a-very-long-model-name-for-some-one", wide)
        for width in range(1, 90):
            self.assertLessEqual(cell_width(roster.lines(width, PLAIN)[0]), width)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_is_plain_and_json_safe(self) -> None:
        roster = SubagentRoster()
        for event in events():
            roster.apply(event)
        data = roster.snapshot()
        self.assertEqual(
            (data["total"], data["running"], data["done"], data["error"], data["evicted"]),
            (2, 0, 1, 1, 0),
        )
        self.assertEqual(data["tasks"][0]["tools"], ["read_file", "grep"])
        self.assertEqual(json.loads(json.dumps(data))["tasks"][1]["status"], ERROR)
        self.assertEqual(data["tasks"][0]["chars"], 1234)

    def test_task_state_defaults(self) -> None:
        state = TaskState(id="1", title="t", status=RUNNING)
        self.assertEqual((state.lane, state.model, state.seconds, state.chars, state.tools, state.error), ("", "", 0.0, 0, (), ""))
        self.assertFalse(hasattr(state, "__dict__"))


class RealTaskToolEventTests(unittest.TestCase):
    """The roster must fold what `subagents.make_task_tool` actually emits."""

    class Provider:
        def __init__(self) -> None:
            self.seen: dict[str, int] = {}

        def stream_chat(self, model: str, messages: list[dict], tools=None, **kwargs):
            prompt = str(messages[-1].get("content", ""))
            if "fail" in prompt:
                raise ProviderError("lane is dry")
            if self.seen.get(prompt, 0) == 0 and "tool" in prompt:
                self.seen[prompt] = 1
                yield StreamEvent(
                    "tool_call",
                    tool_name="read_file",
                    tool_call_id="call_1",
                    arguments=json.dumps({"path": "missing.txt"}),
                )
                yield StreamEvent("done", finish_reason="tool_calls")
                return
            yield StreamEvent("text", text="report body")
            yield StreamEvent("done", finish_reason="stop")

    def test_events_from_the_task_tool_fold_and_render(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            config = new_config("openai", model="mock")
            lanes = LaneManager(config, providers={"openai": self.Provider()})
            events: list[dict] = []
            tool = make_task_tool(
                config=config,
                lanes=lanes,
                workspace=workspace,
                hooks=AgentHooks(subagent=events.append),
            )
            ctx = ToolContext(workspace=workspace, approval_mode="read-only", read_only=True)
            result = tool.handler(
                {
                    "tasks": [
                        {"title": "tool job", "prompt": "use a tool please"},
                        {"title": "bad job", "prompt": "this must fail"},
                    ]
                },
                ctx,
            )
            self.assertTrue(result.ok)  # one of the two reported, so the call itself succeeded
            self.assertEqual((result.meta["completed"], result.meta["failed"]), (1, 1))
            roster = SubagentRoster()
            folded = sum(1 for event in events if roster.apply(event))
            self.assertEqual(folded, len(events))  # every frame the run emitted was usable
            self.assertEqual(roster.total, 2)
            by_title = {task.title: task for task in roster.tasks()}
            self.assertEqual(set(by_title), {"tool job", "bad job"})
            done = by_title["tool job"]
            self.assertEqual((done.status, done.lane, done.chars, done.tools), (DONE, "subagents", 11, ("read_file",)))
            self.assertGreaterEqual(done.seconds, 0.0)
            failed = by_title["bad job"]
            self.assertEqual(failed.status, ERROR)
            self.assertIn("lane is dry", failed.error)
            self.assertNotIn("\n", failed.error)  # a multi-line verdict becomes one clean row
            rows = roster.lines(78, PLAIN)
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0].startswith("  ✗ "))  # the failure leads the done row
            for row in rows:
                self.assertLessEqual(cell_width(row), 78)
            roster.reset()
            self.assertEqual(roster.total, 0)


if __name__ == "__main__":
    unittest.main()
