from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ohar.types import ToolContext
from ohar.workflows.blocks import BlockLog
from ohar.workflows.board import (
    BOARD_LANES,
    CLAIM_BOUNDARY,
    DEFAULT_LIMIT,
    ITEM_KINDS,
    MAX_ITEMS,
    MAX_REASON_CHARS,
    MAX_TITLE_CHARS,
    UNASSIGNED,
    Board,
    BoardItem,
    board_digest,
    build_board,
    headers,
    lane_for,
    make_board_tool,
    render,
    rows,
    validate_board,
)
from ohar.workflows.gates import DecisionGates
from ohar.workflows.goals import GoalLedger
from ohar.workflows.journal import ActivityJournal
from ohar.workflows.verification import VerificationLedger


def _item(**overrides: object) -> BoardItem:
    data: dict[str, object] = {
        "id": "goal:g1",
        "kind": "goal",
        "lane": "moving",
        "title": "ship the board",
        "status": "active",
        "owner": "s1",
        "goal_id": "g1",
        "age_seconds": 30,
        "missing_evidence": ["unit u1 unverified"],
        "next_action": "verify_unit",
    }
    data.update(overrides)
    return BoardItem(**data)  # type: ignore[arg-type]


def _board(items: list[BoardItem], *, truncated: bool = False) -> Board:
    counts = {lane: sum(1 for item in items if item.lane == lane) for lane in BOARD_LANES}
    total = sum(counts.values())
    return Board(
        taken_at="2026-10-25T12:00:00+00:00",
        items=items,
        counts=counts,
        digest=board_digest(items),
        truncated=truncated,
        claim_boundary=CLAIM_BOUNDARY,
    )


class LaneForTests(unittest.TestCase):
    def test_open_gate_outranks_every_other_state(self) -> None:
        self.assertEqual(
            lane_for("goal", "active", has_open_gate=True, evidence=["checkpoint c1"], age_seconds=5),
            "needs_decision",
        )
        self.assertEqual(lane_for("gate", "open", has_open_gate=False, evidence=[], age_seconds=0), "needs_decision")

    def test_blocked_and_terminal_words(self) -> None:
        self.assertEqual(lane_for("goal", "blocked", has_open_gate=False, evidence=[], age_seconds=1), "blocked")
        self.assertEqual(lane_for("unit", "observed_failed", has_open_gate=False, evidence=[], age_seconds=1), "blocked")
        self.assertEqual(lane_for("goal", "cancelled", has_open_gate=False, evidence=[], age_seconds=1), "closed")
        self.assertEqual(lane_for("unit", "abandoned", has_open_gate=False, evidence=[], age_seconds=1), "closed")

    def test_a_claim_without_evidence_is_only_a_record(self) -> None:
        self.assertEqual(lane_for("goal", "complete", has_open_gate=False, evidence=[], age_seconds=1), "recorded")
        self.assertEqual(
            lane_for("unit", "observed_passed", has_open_gate=False, evidence=["attempt via runner"], age_seconds=1),
            "closed",
        )

    def test_running_needs_an_observation_or_a_known_age(self) -> None:
        self.assertEqual(lane_for("goal", "active", has_open_gate=False, evidence=[], age_seconds=12), "moving")
        self.assertEqual(lane_for("unit", "claimed", has_open_gate=False, evidence=[], age_seconds=-1), "recorded")
        self.assertEqual(
            lane_for("session_activity", "running", has_open_gate=False, evidence=[], age_seconds=-1),
            "ready",
        )
        self.assertEqual(
            lane_for("session_activity", "running", has_open_gate=False, evidence=["turn_completed"], age_seconds=-1),
            "moving",
        )

    def test_startable_words_and_unknown_words(self) -> None:
        for status in ("pending", "not_observed", "configured", ""):
            self.assertEqual(lane_for("criterion", status, has_open_gate=False, evidence=[], age_seconds=3), "ready")
        self.assertEqual(lane_for("block", "perhaps", has_open_gate=False, evidence=[], age_seconds=3), "recorded")

    def test_observed_evidence_alone_reads_as_moving(self) -> None:
        self.assertEqual(
            lane_for("criterion", "pending", has_open_gate=False, evidence=["evidence ref:a1"], age_seconds=3),
            "moving",
        )


class DigestTests(unittest.TestCase):
    def test_digest_is_order_sensitive_and_state_only(self) -> None:
        first = _item(id="goal:a", lane="blocked")
        second = _item(id="goal:b", lane="ready")
        forward = board_digest([first, second])
        self.assertEqual(forward, board_digest([first, second]))
        self.assertNotEqual(forward, board_digest([second, first]))
        self.assertNotEqual(forward, board_digest([first]))

    def test_digest_does_not_place_a_clock_inside_the_payload(self) -> None:
        item = _item()
        board = _board([item])
        self.assertNotIn("2026-10-25", board.digest)
        self.assertEqual(board.digest, board_digest(board.items))


class RenderingTests(unittest.TestCase):
    def test_headers_and_rows_stay_aligned(self) -> None:
        board = _board([_item(), _item(id="unit:u1", kind="unit", lane="ready", missing_evidence=[])])
        table_rows = rows(board)
        self.assertEqual(len(headers()), 9)
        self.assertEqual([len(row) for row in table_rows], [len(headers())] * 2)
        self.assertEqual(table_rows[0][0], "moving")
        self.assertEqual(table_rows[0][6], "30.0s")

    def test_unknown_age_is_rendered_as_unknown(self) -> None:
        self.assertEqual(rows(_board([_item(age_seconds=-1)]))[0][6], "unknown")

    def test_rows_limit_pages_the_display(self) -> None:
        board = _board([_item(id=f"goal:g{index}") for index in range(5)])
        self.assertEqual(len(rows(board, limit=2)), 2)
        self.assertEqual(len(rows(board, limit=0)), 0)

    def test_lane_items_and_summary(self) -> None:
        board = _board(
            [
                _item(id="gate:x", kind="gate", lane="needs_decision"),
                _item(id="goal:a", lane="moving"),
                _item(id="goal:b", lane="moving"),
            ]
        )
        self.assertEqual([item.id for item in board.lane_items("moving")], ["goal:a", "goal:b"])
        self.assertEqual(board.lane_items("blocked"), [])
        self.assertEqual(board.lane_items("moving", limit=1)[0].id, "goal:a")
        summary = board.summary()
        self.assertIn("3 board items", summary)
        self.assertIn("2 moving", summary)
        self.assertIn("1 needs a decision", summary)
        self.assertEqual(_board([]).summary().startswith("Nothing recorded yet"), True)

    def test_render_groups_by_lane_and_carries_the_boundary(self) -> None:
        board = _board([_item(id="gate:x", kind="gate", lane="needs_decision"), _item()])
        lines = render(board)
        self.assertTrue(all("\x1b" not in line for line in lines))
        text = "\n".join(lines)
        self.assertIn("Needs a decision", text)
        self.assertIn("ship the board", text)
        self.assertIn("read-only projection", text)
        self.assertIn(board.digest[:12], text)

    def test_render_states_the_truncation_explicitly(self) -> None:
        board = _board([_item(id="goal:a")], truncated=True)
        board.counts = {lane: 0 for lane in BOARD_LANES}
        board.counts["moving"] = 1
        board.counts["ready"] = 6
        text = "\n".join(render(board))
        self.assertIn("Showing 1 of 7 items", text)
        self.assertIn("6 not shown because of the display limit", text)

    def test_render_of_an_empty_board_says_so(self) -> None:
        self.assertIn("No coordinated work recorded.", "\n".join(render(_board([]))))

    def test_ascii_glyphs_are_available(self) -> None:
        text = "\n".join(render(_board([_item()]), glyphs="ascii"))
        self.assertNotIn("—", text.split("digest")[0])
        self.assertIn("MOVING", text)


class ValidateBoardTests(unittest.TestCase):
    def test_a_built_board_payload_validates(self) -> None:
        board = _board([_item(), _item(id="gate:x", kind="gate", lane="needs_decision", missing_evidence=["answer"])])
        self.assertEqual(validate_board(board.to_dict()), [])

    def test_rejects_vocabulary_clock_and_boundary_faults(self) -> None:
        board = _board([_item()])
        payload = board.to_dict()
        payload["items"][0]["lane"] = "urgent"
        payload["digest"] = ""
        payload["claim_boundary"] = ""
        payload["counts"].pop("closed")
        errors = " ".join(validate_board(payload))
        self.assertIn("lane is unsupported", errors)
        self.assertIn("digest must be", errors)
        self.assertIn("claim_boundary is required", errors)
        self.assertIn("counts must carry exactly every board lane", errors)

    def test_a_recorded_lane_must_list_what_it_lacks(self) -> None:
        payload = _board([_item(lane="recorded", missing_evidence=[])]).to_dict()
        self.assertIn("recorded requires the evidence it lacks", " ".join(validate_board(payload)))

    def test_closed_cannot_carry_missing_evidence(self) -> None:
        payload = _board([_item(lane="closed", missing_evidence=["unit u1"])]).to_dict()
        self.assertIn("closed cannot carry missing evidence", " ".join(validate_board(payload)))

    def test_needs_decision_requires_the_decision(self) -> None:
        payload = _board([_item(lane="needs_decision", next_action="")]).to_dict()
        self.assertIn("needs_decision requires", " ".join(validate_board(payload)))

    def test_titles_are_bounded_and_clean(self) -> None:
        payload = _board([_item(title="x" * (MAX_TITLE_CHARS + 1))]).to_dict()
        self.assertIn("at most", " ".join(validate_board(payload)))
        payload = _board([_item(title="red\x1b[31mtxt")]).to_dict()
        self.assertIn("control or ANSI", " ".join(validate_board(payload)))

    def test_age_and_shape_requirements(self) -> None:
        payload = _board([_item(age_seconds=-2)]).to_dict()
        self.assertIn("age_seconds", " ".join(validate_board(payload)))
        payload = _board([_item()]).to_dict()
        payload["items"][0].pop("owner")
        self.assertIn("item is missing owner", " ".join(validate_board(payload)))
        self.assertEqual(validate_board("not a board"), ["board must be an object"])
        payload = _board([_item()]).to_dict()
        payload["items"] = "nope"
        self.assertIn("items must be a list", " ".join(validate_board(payload)))
        payload = _board([_item()]).to_dict()
        payload["truncated"] = "yes"
        self.assertIn("truncated must be boolean", " ".join(validate_board(payload)))

    def test_untruncated_board_must_show_every_counted_item(self) -> None:
        payload = _board([_item()]).to_dict()
        payload["counts"]["moving"] = 9
        self.assertIn("must display every counted item", " ".join(validate_board(payload)))


class _Ledgers(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.workspace = base / "workspace"
        self.workspace.mkdir()
        self.home = base / "home"
        self.moment = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        self.now = self.moment
        self.journal = ActivityJournal(self.home, self.workspace, clock=lambda: self.now)
        self.goals = GoalLedger(self.home, self.workspace, clock=self._iso, journal=self.journal)
        self.gates = DecisionGates(self.home, self.workspace, clock=lambda: self.now, journal=self.journal)
        self.blocks = BlockLog(self.home, self.workspace, clock=self._iso, journal=self.journal)
        self.units = VerificationLedger(self.home, self.workspace, journal=self.journal, clock=lambda: self.now)

    def _iso(self) -> str:
        return datetime.fromtimestamp(self.now, timezone.utc).isoformat(timespec="microseconds")

    def board(self, **kwargs: object) -> Board:
        options: dict[str, object] = {"now": self.now}
        options.update(kwargs)  # type: ignore[arg-type]
        return build_board(  # type: ignore[arg-type]
            goals=self.goals,
            gates=self.gates,
            blocks=self.blocks,
            units=self.units,
            journal=self.journal,
            **options,
        )

    def find(self, board: Board, kind: str) -> list[BoardItem]:
        return [item for item in board.items if item.kind == kind]


class BuildBoardTests(_Ledgers):
    def test_empty_ledgers_still_give_a_valid_board(self) -> None:
        board = self.board()
        self.assertEqual(board.items, [])
        self.assertEqual(set(board.counts), set(BOARD_LANES))
        self.assertEqual(sum(board.counts.values()), 0)
        self.assertEqual(validate_board(board.to_dict()), [])
        self.assertFalse(board.truncated)

    def test_every_ledger_is_projected(self) -> None:
        goal = self.goals.create("ship the board", ["tests pass"], session_id="s1")
        self.journal.record("turn_completed", subject="turn-1", ok=True, session_id="s1")
        self.blocks.record(
            source="policy",
            reason_domain="read_only_policy",
            reason_code="read_only",
            action="write_file",
            subject="a.txt",
            session_id="s1",
            goal_id=goal.id,
        )
        self.units.declare("run the suite", kind="command", check="python3 -m unittest", goal_id=goal.id)
        self.gates.open(
            subject=goal.id,
            question_code="allow_write",
            action="write_file",
            arg_digest="d1",
            session_id="s1",
        )
        board = self.board(session_id="s1")
        kinds = {item.kind for item in board.items}
        self.assertTrue({"goal", "criterion", "unit", "gate", "block", "session_activity"} <= kinds)
        self.assertEqual(validate_board(board.to_dict()), [])
        self.assertEqual(board.items[0].lane, "needs_decision")
        decisions = [item for item in board.items if item.lane == "needs_decision"]
        gate_item = next(item for item in decisions if item.kind == "gate")
        self.assertEqual(gate_item.owner, "operator")
        self.assertEqual(gate_item.next_action, "answer_gate")
        self.assertEqual(self.find(board, "goal")[0].lane, "needs_decision")
        blocks = self.find(board, "block")
        self.assertEqual(blocks[0].lane, "blocked")
        self.assertEqual(blocks[0].owner, "operator")
        self.assertEqual(blocks[0].next_action, "leave_read_only")
        self.assertEqual(self.find(board, "goal")[0].goal_id, goal.id)
        activity = self.find(board, "session_activity")
        self.assertEqual(activity[0].lane, "moving")
        self.assertEqual(activity[0].age_seconds, 0)

    def test_an_open_gate_moves_the_goal_it_gates_into_the_decision_lane(self) -> None:
        goal = self.goals.create("gate me", ["answer needed"])
        self.gates.open(subject=goal.id, question_code="resume_blocked_goal", action="goal", arg_digest="x")
        board = self.board()
        self.assertEqual(self.find(board, "goal")[0].lane, "needs_decision")
        self.assertIn("gate answer", " ".join(self.find(board, "goal")[0].missing_evidence))

    def test_a_status_word_leaves_the_missing_evidence_listed(self) -> None:
        goal = self.goals.create("ship it", ["tests pass"])
        criterion = goal.criteria[0]
        self.goals.checkpoint(
            goal.id,
            "ran the suite",
            status="done",
            criteria_refs=[criterion.id],
            evidence_refs=["unit-u1"],
        )
        self.goals.complete(goal.id, "shipped", evidence_refs=["unit-u1"])
        self.units.declare("run the suite", kind="command", check="python3 -m unittest", goal_id=goal.id)
        board = self.board()
        goal_item = self.find(board, "goal")[0]
        self.assertEqual(goal_item.status, "complete")
        self.assertEqual(goal_item.lane, "recorded")
        self.assertTrue(goal_item.missing_evidence)
        self.assertIn("unit", " ".join(goal_item.missing_evidence))

    def test_a_criterion_waits_for_verification_until_evidence_is_cited(self) -> None:
        goal = self.goals.create("prove it", ["suite passes"])
        pending = self.find(self.board(), "criterion")[0]
        self.assertEqual(pending.lane, "ready")
        self.assertEqual(pending.next_action, "verify_unit")
        self.goals.checkpoint(
            goal.id,
            "suite ran green",
            status="done",
            criteria_refs=[goal.criteria[0].id],
            evidence_refs=["unit-u1"],
        )
        closed = self.find(self.board(), "criterion")[0]
        self.assertEqual(closed.status, "satisfied")
        self.assertEqual(closed.lane, "closed")
        self.assertEqual(closed.missing_evidence, [])

    def test_a_blocked_goal_is_blocked_and_not_ready(self) -> None:
        goal = self.goals.create("blocked work", ["keep going"])
        self.goals.block(goal.id, "provider refuses")
        item = self.find(self.board(), "goal")[0]
        self.assertEqual(item.lane, "blocked")
        self.assertTrue(any("blocker" in gap for gap in item.missing_evidence))

    def test_units_move_up_the_ladder_and_never_into_ready_by_claim(self) -> None:
        unit = self.units.declare("compile", kind="command", check="python3 -c 1")
        item = self.find(self.board(), "unit")[0]
        self.assertIn(item.lane, ("ready", "recorded"))
        self.assertTrue(item.missing_evidence)
        self.units.record_attempt(unit.id, ok=True, exit_code=0, seconds=0.1, observed_via="operator")
        after = self.find(self.board(), "unit")[0]
        self.assertNotIn(after.lane, ("ready", "needs_decision"))
        self.assertGreaterEqual(after.age_seconds, 0)

    def test_journal_activity_is_not_double_counted_against_the_ledgers(self) -> None:
        goal = self.goals.create("one row per work", ["keep it singular"])
        self.journal.record("goal_updated", subject=goal.id, ok=True)
        self.journal.record("tool_observed", subject="shell", ok=True)
        board = self.board()
        subjects = [item.id for item in self.find(board, "session_activity")]
        self.assertNotIn(f"session:{goal.id}", subjects)
        self.assertIn("session:shell", subjects)

    def test_identical_state_hashes_identically_and_new_work_does_not(self) -> None:
        goal = self.goals.create("stable", ["one"])
        first = self.board()
        second = self.board()
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.items, second.items)
        self.goals.checkpoint(goal.id, "progress", status="done")
        self.assertNotEqual(first.digest, self.board().digest)

    def test_limit_changes_the_view_not_the_identity(self) -> None:
        for index in range(4):
            self.goals.create(f"goal {index}", ["criterion a", "criterion b"])
        wide = self.board(limit=MAX_ITEMS)
        narrow = self.board(limit=1)
        self.assertEqual(wide.digest, narrow.digest)
        self.assertEqual(len(narrow.items), 1)
        self.assertTrue(narrow.truncated)
        self.assertFalse(wide.truncated)
        self.assertGreaterEqual(len(wide.items), 8)
        self.assertEqual(sum(wide.counts.values()), len(wide.items))
        self.assertEqual(sum(narrow.counts.values()), sum(wide.counts.values()))

    def test_titles_are_bounded_and_stripped_of_control_characters(self) -> None:
        self.goals.create("  \x1b[31mred\x1b[0m objective\nwith a newline  " + "x" * 200, ["one"])
        item = self.find(self.board(), "goal")[0]
        self.assertLessEqual(len(item.title), MAX_TITLE_CHARS)
        self.assertNotIn("\x1b", item.title)
        self.assertNotIn("\n", item.title)
        self.assertEqual(validate_board(self.board().to_dict()), [])

    def test_unassigned_owner_when_nothing_names_one(self) -> None:
        self.blocks.record(
            source="runner",
            reason_domain="tool_failure",
            reason_code="tool_failure",
            action="shell",
            outcome="failed",
        )
        item = self.find(self.board(), "block")[0]
        self.assertIn(item.owner, (UNASSIGNED, "operator"))

    def test_gates_and_units_are_optional_sources(self) -> None:
        self.goals.create("goals only", ["one"])
        board = build_board(goals=self.goals, now=self.now)
        self.assertEqual({item.kind for item in board.items}, {"goal", "criterion"})
        self.assertEqual(validate_board(board.to_dict()), [])


class BoardToolTests(_Ledgers):
    def setUp(self) -> None:
        super().setUp()
        self.tool = make_board_tool(
            goals=self.goals,
            gates=self.gates,
            blocks=self.blocks,
            units=self.units,
            journal=self.journal,
        )
        self.ctx = ToolContext(workspace=self.workspace, approval_mode="ask")

    def test_tool_is_read_only_and_declares_its_vocabulary(self) -> None:
        self.assertTrue(self.tool.read_only)
        self.assertFalse(self.tool.destructive)
        self.assertEqual(self.tool.name, "board")
        schema = self.tool.api_spec()["function"]["parameters"]
        self.assertEqual(schema["properties"]["lane"]["enum"], list(BOARD_LANES))
        self.assertEqual(schema["properties"]["action"]["enum"], ["overview", "lane"])
        self.assertEqual(schema["required"], ["action"])
        self.assertIn("read-only", self.tool.description.lower())

    def test_overview_returns_the_board_text_and_metadata(self) -> None:
        self.goals.create("board tool", ["rendered"])
        result = self.tool.handler({"action": "overview"}, self.ctx)
        self.assertTrue(result.ok)
        self.assertNotIn("\x1b", result.content)
        self.assertIn("board tool", result.content)
        self.assertIn("Work board", result.content)
        self.assertEqual(len(result.meta["digest"]), 64)
        self.assertEqual(set(result.meta["counts"]), set(BOARD_LANES))
        self.assertFalse(result.meta["truncated"])

    def test_lane_returns_only_that_lane(self) -> None:
        self.goals.create("lane view", ["one"])
        self.gates.open(subject="src/a.py", question_code="allow_shell", action="shell", arg_digest="d")
        result = self.tool.handler({"action": "lane", "lane": "needs_decision"}, self.ctx)
        self.assertTrue(result.ok)
        self.assertEqual(result.meta["count"], 1)
        self.assertIn("gate:", result.content)
        self.assertNotIn("lane view", result.content)
        empty = self.tool.handler({"action": "lane", "lane": "closed"}, self.ctx)
        self.assertTrue(empty.ok)
        self.assertEqual(empty.meta["count"], 0)
        self.assertIn("Nothing is in the 'closed' lane", empty.content)

    def test_bad_arguments_are_refused_without_building_anything(self) -> None:
        missing_lane = self.tool.handler({"action": "lane"}, self.ctx)
        self.assertFalse(missing_lane.ok)
        self.assertEqual(missing_lane.meta["reason"], "lane_invalid")
        bad_action = self.tool.handler({"action": "dispatch"}, self.ctx)
        self.assertFalse(bad_action.ok)
        self.assertEqual(bad_action.meta["reason"], "action_invalid")
        bad_limit = self.tool.handler({"action": "overview", "limit": "many"}, self.ctx)
        self.assertFalse(bad_limit.ok)
        self.assertEqual(bad_limit.meta["reason"], "limit_invalid")

    def test_default_limit_constant_is_the_rendered_page(self) -> None:
        self.assertEqual(DEFAULT_LIMIT, 20)
        self.assertEqual(len(ITEM_KINDS), 6)
        self.assertEqual(MAX_REASON_CHARS, 160)


if __name__ == "__main__":
    unittest.main()
