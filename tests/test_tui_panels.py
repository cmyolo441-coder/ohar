"""Tests for the HUD panels: chips, the plan block, the agents block, and the row budget."""

from __future__ import annotations

import unittest
from typing import Any

from ohar.planner import PlanState
from ohar.terminal import display_width, make_theme, strip_ansi
from ohar.tui.cells import strip_control
from ohar.tui.dag import DagNode, dag_nodes
from ohar.tui.panels import AGENTS_ROWS, DAG_ROWS, PLAN_ROWS, agents_lines, chips, hud_lines, plan_lines
from ohar.tui.roster import SubagentRoster

PLAIN = make_theme("mono", color=False)
COLOR = make_theme("aurora")

MIXED = ["done", "running", "done", "error", "running", "done"]


def plan_of(statuses: list[str], prefix: str = "step") -> dict[str, Any]:
    return {
        "steps": [{"step": f"{prefix} {index + 1}", "status": status} for index, status in enumerate(statuses)],
        "updated_at": "",
        "counts": {},
    }


def roster_of(kinds: list[str]) -> SubagentRoster:
    roster = SubagentRoster()
    for index, kind in enumerate(kinds, start=1):
        ident = str(index)
        roster.apply(
            {
                "type": "start",
                "id": ident,
                "title": f"task {index}",
                "lane": "sub",
                "model": "gpt-5-mini",
            }
        )
        if kind == "done":
            roster.apply({"type": "done", "id": ident, "seconds": 3.0, "chars": 120})
        elif kind == "error":
            roster.apply({"type": "error", "id": ident, "error": "boom"})
    return roster


def chain(count: int) -> list[DagNode]:
    return [
        DagNode(
            id=f"n{index}",
            title=f"node {index}",
            status="running" if index < 2 else "pending",
            parents=(f"n{index - 1}",) if index else (),
            lane="sub",
        )
        for index in range(count)
    ]


def said(rows: list[str], needle: str) -> int:
    return sum(1 for row in rows if needle in strip_ansi(row))


class ChipsTests(unittest.TestCase):
    def test_the_chips_say_the_mode_the_lane_and_the_counts(self) -> None:
        line = chips(approval="ask", model="gpt-5", lane="fast", theme=PLAIN, running=2)
        self.assertEqual(line, "[ask] │ [gpt-5 · fast] │ [2 running]")

    def test_a_chip_is_drawn_only_for_what_is_known(self) -> None:
        self.assertEqual(chips(approval="ask", model="", lane="", theme=PLAIN), "[ask]")
        self.assertEqual(chips(approval="", model="gpt-5", lane="", theme=PLAIN, running=0), "[gpt-5]")
        self.assertEqual(chips(approval="", model="", lane="", theme=PLAIN), "")

    def test_a_loose_mode_is_said_loudly_and_there_is_no_yolo_chip(self) -> None:
        strict = chips(approval="ask", model="", lane="", theme=COLOR)
        loose = chips(approval="yolo", model="", lane="", theme=COLOR)
        self.assertEqual(strip_ansi(strict), "[ask]")
        self.assertEqual(strip_ansi(loose), "[yolo]")
        self.assertNotEqual(strict, loose)
        self.assertNotIn("yolo mode", strip_ansi(loose))

    def test_chip_text_is_control_free_and_bounded(self) -> None:
        line = chips(approval="\x1b[2Jask\n", model="m" * 200, lane="fast\x07", theme=PLAIN)
        self.assertNotIn("\x1b", line)
        self.assertNotIn("\n", line)
        self.assertLessEqual(display_width(line), 16 + 32 + 16 + 16)
        self.assertIn("[m" * 1, line)

    def test_a_single_string_never_becomes_several_rows(self) -> None:
        self.assertNotIn("\n", chips(approval="ask", model="a\nb", lane="c\rd", theme=PLAIN))


class PlanLinesTests(unittest.TestCase):
    statuses = ["completed"] * 5 + ["in_progress"] + ["pending"] * 6

    def test_the_header_always_stays(self) -> None:
        for max_rows in (1, 2, 3):
            with self.subTest(max_rows=max_rows):
                rows = plan_lines(plan_of(self.statuses), 60, PLAIN, max_rows=max_rows)
                self.assertEqual(rows[0], "[Plan] · 5/12 done · 1 active")

    def test_the_window_closes_around_the_active_item(self) -> None:
        for max_rows in range(2, PLAN_ROWS + 1):
            with self.subTest(max_rows=max_rows):
                rows = plan_lines(plan_of(self.statuses), 60, PLAIN, max_rows=max_rows)
                self.assertLessEqual(len(rows), max_rows)
                self.assertEqual(said(rows, "▶ 6. step 6"), 1)

    def test_hidden_items_are_counted_not_lost(self) -> None:
        rows = plan_lines(plan_of(self.statuses), 60, PLAIN, max_rows=8)
        self.assertEqual(rows[1], "  … 4 earlier")
        self.assertEqual(rows[-1], "  … 3 later")
        self.assertEqual(len(rows), 8)

    def test_an_active_item_at_either_end_needs_no_fold_row(self) -> None:
        rows = plan_lines(plan_of(["in_progress"] + ["pending"] * 3), 60, PLAIN, max_rows=5)
        self.assertEqual(rows[1], "  ▶ 1. step 1")

    def test_never_more_rows_than_asked(self) -> None:
        for max_rows in range(0, PLAN_ROWS + 6):
            with self.subTest(max_rows=max_rows):
                self.assertLessEqual(len(plan_lines(plan_of(self.statuses), 60, PLAIN, max_rows=max_rows)), max_rows)

    def test_no_row_is_wider_than_the_screen(self) -> None:
        wide = plan_of(["in_progress"] + ["pending"] * 9, prefix="wiiiiiiiiiiiiide ")
        for width in (6, 12, 20, 40, 80):
            for max_rows in range(1, PLAN_ROWS + 2):
                for row in plan_lines(wide, width, COLOR, max_rows=max_rows):
                    with self.subTest(width=width, max_rows=max_rows):
                        self.assertLessEqual(display_width(row), width)

    def test_untrusted_step_text_is_stripped_to_one_line(self) -> None:
        rows = plan_lines({"steps": [{"step": "\x1b[31mred\nplain\x07", "status": "in_progress"}]}, 40, PLAIN)
        self.assertEqual(rows[-1], "  ▶ 1. red plain")

    def test_a_plan_state_is_read_from_its_own_dict(self) -> None:
        state = PlanState()
        state.update([{"step": "ship", "status": "in_progress"}])
        self.assertEqual(plan_lines(state, 40, PLAIN), plan_lines(state.to_dict(), 40, PLAIN))

    def test_malformed_input_draws_nothing_or_only_what_is_whole(self) -> None:
        self.assertEqual(plan_lines(None, 40, PLAIN), [])  # type: ignore[arg-type]
        self.assertEqual(plan_lines("steps", 40, PLAIN), [])  # type: ignore[arg-type]
        self.assertEqual(plan_lines({"steps": "nope"}, 40, PLAIN), [])
        self.assertEqual(plan_lines({"steps": []}, 40, PLAIN), [])
        self.assertEqual(plan_lines({"steps": [{"step": "  "}]}, 40, PLAIN), [])
        self.assertEqual(plan_lines({"updated_at": "now"}, 40, PLAIN), [])
        rows = plan_lines({"steps": [5, {"status": "pending"}, {"step": "ok", "status": "sideways"}]}, 40, PLAIN)
        self.assertEqual(rows[-1], "  ○ 1. ok")

    def test_a_finished_plan_says_it_is_done(self) -> None:
        rows = plan_lines(plan_of(["completed"] * 3), 40, PLAIN)
        self.assertEqual(rows[0], "[Plan] · 3/3 done")
        self.assertEqual(len(rows), 4)

    def test_ascii_glyphs_and_a_said_title(self) -> None:
        rows = plan_lines(plan_of(["completed", "in_progress", "pending"]), 40, PLAIN, title="Todo", glyphs="ascii")
        self.assertEqual(rows[0], "[Todo] | 1/3 done | 1 active")
        self.assertEqual([row[2:5] for row in rows[1:]], ["[x]", "[>]", "[ ]"])
        self.assertNotIn("…", "".join(rows))


class AgentsLinesTests(unittest.TestCase):
    def test_the_header_counts_running_and_failed(self) -> None:
        rows = agents_lines(roster_of(MIXED), 60, PLAIN)
        self.assertEqual(rows[0], "[Agents] · 2/6 running · 1 error")

    def test_an_empty_roster_draws_nothing(self) -> None:
        self.assertEqual(agents_lines(SubagentRoster(), 60, PLAIN), [])
        self.assertEqual(agents_lines(None, 60, PLAIN), [])  # type: ignore[arg-type]

    def test_pressure_drops_done_rows_first_and_running_rows_last(self) -> None:
        rows = agents_lines(roster_of(MIXED), 60, PLAIN, max_rows=4)
        self.assertEqual(len(rows), 4)
        self.assertIn("task 2", rows[1])
        self.assertIn("task 5", rows[2])
        self.assertEqual(rows[3], "  +4 more")

    def test_only_finished_rows_hidden_say_done(self) -> None:
        rows = agents_lines(roster_of(["done"] * 5), 60, PLAIN, max_rows=3)
        self.assertEqual(rows[-1], "  +4 done")

    def test_rows_keep_the_roster_order(self) -> None:
        rows = agents_lines(roster_of(MIXED), 60, PLAIN, max_rows=AGENTS_ROWS)
        self.assertEqual([row.split()[2] for row in rows[1:5]], ["1", "2", "4", "5"])

    def test_no_room_for_a_row_leaves_the_header_alone(self) -> None:
        roster = roster_of(MIXED)
        self.assertEqual(len(agents_lines(roster, 60, PLAIN, max_rows=1)), 1)
        self.assertEqual(len(agents_lines(roster, 60, PLAIN, max_rows=2)), 1)

    def test_an_error_row_carries_the_reason(self) -> None:
        rows = agents_lines(roster_of(["error"]), 60, PLAIN)
        self.assertIn("✗ task 1  boom", rows[1])

    def test_never_more_rows_than_asked(self) -> None:
        roster = roster_of(MIXED)
        for max_rows in range(0, AGENTS_ROWS + 4):
            with self.subTest(max_rows=max_rows):
                self.assertLessEqual(len(agents_lines(roster, 60, PLAIN, max_rows=max_rows)), max_rows)

    def test_no_row_is_wider_than_the_screen(self) -> None:
        roster = SubagentRoster()
        for index in range(3):
            roster.apply(
                {
                    "type": "start",
                    "id": str(index + 1),
                    "title": "a title that goes on "*4,
                    "lane": "a-lane",
                    "model": "a-model-name",
                }
            )
        for width in (6, 12, 20, 40, 80):
            for max_rows in range(1, AGENTS_ROWS + 1):
                for row in agents_lines(roster, width, COLOR, max_rows=max_rows):
                    with self.subTest(width=width):
                        self.assertLessEqual(display_width(row), width)

    def test_untrusted_fields_are_stripped_to_one_line(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": "hi\x1b[31m\nthere"})
        roster.apply({"type": "error", "id": "1", "error": "\x07boom\x1b[J"})
        rows = agents_lines(roster, 60, PLAIN)
        self.assertEqual(strip_control(rows[1]), rows[1])
        self.assertIn("there", rows[1])
        self.assertIn("boom", rows[1])


class HudLinesTests(unittest.TestCase):
    statuses = ["completed"] * 5 + ["in_progress"] + ["pending"] * 6

    def hud(self, budget: int, **kwargs: Any) -> list[str]:
        fields: dict[str, Any] = dict(
            plan=plan_of(self.statuses),
            roster=roster_of(MIXED),
            nodes=None,
            width=80,
            budget=budget,
            theme=PLAIN,
            approval="ask",
            model="gpt-5",
            lane="fast",
        )
        fields.update(kwargs)
        return hud_lines(**fields)

    def test_no_budget_draws_nothing(self) -> None:
        self.assertEqual(self.hud(0), [])
        self.assertEqual(self.hud(-3), [])
        self.assertEqual(hud_lines(plan=None, roster=None, nodes=None, width=80, budget=10, theme=PLAIN), ["[ask]"])
        self.assertEqual(
            hud_lines(plan=None, roster=None, nodes=None, width=80, budget=10, theme=PLAIN, approval=""),
            [],
        )

    def test_the_chips_line_leads(self) -> None:
        self.assertEqual(self.hud(20)[0], "[ask] │ [gpt-5 · fast] │ [2 running]")

    def test_only_the_chips_survive_a_one_row_budget(self) -> None:
        self.assertEqual(self.hud(1), ["[ask] │ [gpt-5 · fast] │ [2 running]"])
        self.assertEqual(self.hud(2), ["[ask] │ [gpt-5 · fast] │ [2 running]", "[Plan] · 5/12 done · 1 active"])

    def test_never_over_budget_and_never_over_width(self) -> None:
        nodes = dag_nodes(roster_of(MIXED))
        for width in (12, 24, 48, 80):
            for theme in (PLAIN, COLOR):
                for budget in range(0, PLAN_ROWS + AGENTS_ROWS + DAG_ROWS + 4):
                    rows = self.hud(
                        budget, nodes=nodes, width=width, theme=theme, plan=plan_of(self.statuses)
                    )
                    with self.subTest(width=width, budget=budget, theme=theme.name):
                        self.assertLessEqual(len(rows), budget)
                        for row in rows:
                            self.assertLessEqual(display_width(row), width)
                            self.assertNotIn("\n", row)

    def test_activity_rows_are_shed_before_plan_rows(self) -> None:
        natural = self.hud(30)
        tight = self.hud(len(natural) - 2)
        self.assertEqual(len(tight), len(natural) - 2)
        self.assertEqual(tight[:10], natural[:10])
        self.assertEqual(said(tight, "task "), said(natural, "task ") - 2)

    def test_the_tree_shrinks_before_the_plan_does(self) -> None:
        natural = self.hud(30, nodes=chain(6), roster=None)
        tight = self.hud(len(natural) - 2, nodes=chain(6), roster=None)
        self.assertEqual(len(tight), len(natural) - 2)
        self.assertEqual(tight[:9], natural[:9])
        self.assertLess(said(tight, "node "), said(natural, "node "))

    def test_a_graph_holding_every_agent_replaces_the_flat_rows(self) -> None:
        roster = roster_of(MIXED)
        rows = self.hud(30, roster=roster, nodes=dag_nodes(roster))
        self.assertNotIn("[Agents]", "\n".join(rows))
        self.assertGreaterEqual(said(rows, "task "), len(roster.tasks()))

    def test_a_graph_holding_some_agents_keeps_both_blocks(self) -> None:
        rows = self.hud(30, nodes=chain(2), roster=roster_of(MIXED))
        self.assertIn("[Agents]", "\n".join(rows))
        self.assertIn("node 0", "\n".join(rows))

    def test_a_finished_run_is_one_summary_row(self) -> None:
        ended = [DagNode(id="1", title="one", status="done"), DagNode(id="2", title="two", status="failed", parents=("1",))]
        rows = self.hud(30, plan=None, roster=None, nodes=ended)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1].startswith("DAG ·"))

    def test_a_plan_can_be_dropped_entirely(self) -> None:
        rows = self.hud(30, plan=None)
        self.assertEqual(rows[0], "[ask] │ [gpt-5 · fast] │ [2 running]")
        self.assertNotIn("[Plan]", "\n".join(rows))

    def test_an_empty_hud_draws_no_row_at_all(self) -> None:
        self.assertEqual(
            hud_lines(
                plan={"steps": []},
                roster=SubagentRoster(),
                nodes=[],
                width=80,
                budget=20,
                theme=PLAIN,
                approval="",
                model="",
                lane="",
            ),
            [],
        )

    def test_junk_inputs_draw_nothing_rather_than_raising(self) -> None:
        rows = self.hud(
            12,
            plan=object(),
            roster=None,
            nodes=[DagNode(id="1", title="x", status="sideways")],  # type: ignore[arg-type]
            width=40,
        )
        self.assertLessEqual(len(rows), 12)
        self.assertGreaterEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
