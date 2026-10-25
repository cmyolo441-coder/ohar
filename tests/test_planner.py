from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from ohar.planner import PlanState, PlanValidationError, make_plan_tool
from ohar.tools import Toolbox
from ohar.types import ToolContext

STEPS = [
    {"step": "Inspect repository", "status": "completed"},
    {"step": "Implement planner", "status": "in_progress"},
    {"step": "Run tests", "status": "pending"},
]
CTX = ToolContext(workspace=Path("."))


class PlanStateTests(unittest.TestCase):
    def test_update_counts_timestamp_and_dict(self) -> None:
        state = PlanState()
        rendered = state.update(STEPS)
        self.assertEqual(state.counts, {"pending": 1, "in_progress": 1, "completed": 1})
        stamp = datetime.fromisoformat(state.updated_at)
        self.assertEqual((stamp.utcoffset().total_seconds(), stamp.microsecond), (0, 0))
        self.assertIn("1/3 completed", rendered)
        self.assertEqual(state.to_dict(), {"steps": STEPS, "updated_at": state.updated_at, "counts": state.counts})

    def test_empty_update_clears(self) -> None:
        state = PlanState()
        state.update(STEPS)
        rendered = state.update([])
        self.assertEqual(state.steps, [])
        self.assertEqual(state.counts, {"pending": 0, "in_progress": 0, "completed": 0})
        self.assertEqual(rendered, "Plan · 0 steps")

    def test_invalid_updates_do_not_mutate(self) -> None:
        state = PlanState()
        state.update(STEPS[:1])
        before = (list(state.steps), state.updated_at)
        bad = [
            [{"step": "x", "status": "done"}],
            [{"step": "  ", "status": "pending"}],
            [{"step": 3, "status": "pending"}],
            [{"step": "x" * 241, "status": "pending"}],
            [{"step": "x"}],
            [{"step": "x", "status": "pending", "extra": True}],
            [{"step": str(i), "status": "pending"} for i in range(21)],
            "not-a-list",
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(PlanValidationError):
                state.update(raw)  # type: ignore[arg-type]
            self.assertEqual((state.steps, state.updated_at), before)

    def test_render_unicode_and_ascii(self) -> None:
        state = PlanState()
        state.update(STEPS)
        unicode = state.render(glyphs="unicode").splitlines()[1:]
        self.assertEqual(
            unicode,
            ["1. ✓ Inspect repository", "2. ▶ Implement planner", "3. ○ Run tests"],
        )
        plain = state.render(glyphs="ascii").splitlines()
        self.assertEqual(
            plain[1:],
            ["1. [x] Inspect repository", "2. [>] Implement planner", "3. [ ] Run tests"],
        )
        self.assertIn("Plan - 1/3 completed", plain[0])
        self.assertTrue(all(ord(char) < 128 for char in plain[0]))


class PlanToolTests(unittest.TestCase):
    def test_toolbox_updates_and_schema_is_strict(self) -> None:
        state = PlanState()
        tool = make_plan_tool(state)
        self.assertEqual((tool.name, tool.read_only), ("update_plan", True))
        self.assertFalse(tool.parameters["additionalProperties"])
        items = tool.parameters["properties"]["steps"]
        self.assertEqual((items["maxItems"], items["items"]["additionalProperties"]), (20, False))
        toolbox = Toolbox(CTX, [tool])
        result = toolbox.execute("update_plan", {"steps": STEPS})
        self.assertTrue(result.ok, result.content)
        self.assertEqual((result.meta["counts"]["completed"], result.meta["total"]), (1, 3))
        self.assertEqual(state.steps[1].step, "Implement planner")
        for arguments in (
            {"steps": [{"step": "x", "status": "bogus"}]},
            {"steps": STEPS, "extra": 1},
            {"steps": "bad"},
        ):
            self.assertFalse(toolbox.execute("update_plan", arguments).ok)
            self.assertEqual(len(state.steps), 3)

    def test_callback_exception_is_isolated(self) -> None:
        state = PlanState()
        calls: list[str] = []

        def on_change(updated: PlanState) -> None:
            calls.append(updated.updated_at)
            raise RuntimeError("UI exploded")

        toolbox = Toolbox(CTX, [make_plan_tool(state, on_change)])
        result = toolbox.execute("update_plan", {"steps": STEPS})
        self.assertTrue(result.ok, result.content)
        self.assertEqual((len(calls), state.counts["in_progress"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
