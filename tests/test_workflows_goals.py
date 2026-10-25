from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ohar.types import ApprovalRequest, ToolContext, ToolResult
from ohar.workflows.goals import (
    BLOCKER_STATUSES,
    CHECKPOINT_STATUSES,
    CRITERION_STATUSES,
    FAILURE_REASON_CODES,
    GOAL_STATUSES,
    MAX_CHECKPOINTS,
    MAX_CRITERIA,
    MAX_SUMMARY_CHARS,
    NEXT_ACTIONS,
    TERMINAL_GOAL_STATUSES,
    Checkpoint,
    CompletionGate,
    Criterion,
    Goal,
    GoalJourney,
    GoalLedger,
    Blocker,
    criterion_id,
    make_goal_tools,
)
from ohar.workflows.journal import ActivityJournal
from ohar.workflows.store import StaleRevisionError, WorkflowError

DAY = 86_400.0


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="microseconds")


class GoalLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "workflows"
        self.ws = self.base / "ws"
        self.ws.mkdir()
        self.now = [1_800_000_000.0]
        self.journal = ActivityJournal(self.root, self.ws, clock=lambda: self.now[0])
        self.ledger = GoalLedger(
            self.root, self.ws, clock=lambda: iso(self.now[0]), journal=self.journal
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def goal(self, criteria: Any = ("tests pass", "docs written")) -> Goal:
        return self.ledger.create("Ship the goals ledger", criteria)

    def ctx(self, **kwargs: Any) -> ToolContext:
        return ToolContext(workspace=self.ws, **kwargs)

    def tools(self, **kwargs: Any) -> dict[str, Any]:
        return {tool.name: tool for tool in make_goal_tools(self.ledger, **kwargs)}

    # -- vocabularies --------------------------------------------------------

    def test_closed_vocabularies(self) -> None:
        self.assertEqual(GOAL_STATUSES, frozenset({"active", "blocked", "complete", "cancelled", "failed"}))
        self.assertEqual(TERMINAL_GOAL_STATUSES, ("complete", "cancelled", "failed"))
        self.assertNotIn("blocked", TERMINAL_GOAL_STATUSES)  # a blocker is recoverable
        self.assertEqual(CHECKPOINT_STATUSES, ("pending", "in_progress", "done", "blocked", "failed"))
        self.assertEqual(CRITERION_STATUSES, ("pending", "satisfied"))
        self.assertEqual(BLOCKER_STATUSES, ("active", "resolved"))
        self.assertEqual(
            FAILURE_REASON_CODES,
            ("target_absent", "refused_by_policy", "criteria_infeasible", "capability_missing"),
        )
        self.assertEqual(MAX_CRITERIA, 8)
        self.assertEqual(MAX_CHECKPOINTS, 50)
        self.assertEqual(MAX_SUMMARY_CHARS, 240)
        for action in NEXT_ACTIONS:
            self.assertIn(action, NEXT_ACTIONS)
        self.assertEqual(len(set(NEXT_ACTIONS)), len(NEXT_ACTIONS))

    def test_criterion_id_is_deterministic_and_content_bound(self) -> None:
        first = criterion_id("tests pass", 1)
        self.assertEqual(first, criterion_id("tests pass", 1))
        self.assertTrue(first.startswith("AC001-"))
        self.assertNotEqual(first, criterion_id("tests pass", 2))
        self.assertNotEqual(first, criterion_id("tests fail", 1))
        self.assertEqual(criterion_id("x", 0), criterion_id("x", 1))

    # -- create --------------------------------------------------------------

    def test_create_records_criteria_and_next_action(self) -> None:
        goal = self.goal()
        self.assertEqual(goal.status, "active")
        self.assertEqual(len(goal.criteria), 2)
        self.assertEqual([item.status for item in goal.criteria], ["pending", "pending"])
        self.assertTrue(all(item.required for item in goal.criteria))
        self.assertEqual(goal.next_action, "record_checkpoint")
        self.assertGreaterEqual(goal.revision, 1)
        self.assertEqual(goal.created_at, goal.updated_at)
        self.assertEqual(self.ledger.get(goal.id).to_dict(), goal.to_dict())
        self.assertEqual(self.journal.counts()["goal_updated"], 1)

    def test_create_accepts_mapping_criteria_and_resumes(self) -> None:
        goal = self.ledger.create(
            "Resume-safe goal",
            [{"id": "AC-ONE", "summary": "first", "required": False}, {"summary": "second"}],
            goal_id="goal-resume-1",
            session_id="s1",
        )
        self.assertEqual(goal.id, "goal-resume-1")
        self.assertEqual([item.id for item in goal.criteria], ["AC-ONE", goal.criteria[1].id])
        self.assertEqual([item.required for item in goal.criteria], [False, True])
        self.assertEqual(goal.session_id, "s1")
        again = self.ledger.create("different objective", ["other"], goal_id="goal-resume-1")
        self.assertEqual(again.objective, goal.objective)  # an existing id never rewrites

    def test_create_refusals(self) -> None:
        bad: list[Any] = [
            ("", ["x"]),
            ("   ", ["x"]),
            ("objective", []),
            ("objective", [""]),
            ("objective", ["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
            ("objective", [{"id": "AC-X", "summary": "one"}, {"id": "AC-X", "summary": "two"}]),
            ("objective", [{"summary": "ok", "status": "done"}]),
        ]
        for objective, criteria in bad:
            with self.subTest(criteria=criteria, objective=objective):
                with self.assertRaises(WorkflowError):
                    self.ledger.create(objective, criteria)
        with self.assertRaises(WorkflowError):
            self.ledger.create("no path goals", ["x"], goal_id="../escape")
        self.assertEqual(self.ledger.list(), [])

    def test_create_strips_control_characters(self) -> None:
        goal = self.ledger.create("\x1b[31mRed\x00 objective\nsecond line", ["\x1b]0;evil\x07 clean"])
        self.assertNotIn("\x1b", goal.objective)
        self.assertNotIn("\x00", goal.objective)
        self.assertIn("second line", goal.objective)
        self.assertEqual(goal.criteria[0].summary, "clean")
        self.assertNotIn("\x1b", self.ledger.render(goal))

    def test_objective_is_bounded(self) -> None:
        goal = self.ledger.create("w" * 5_000, ["x"])
        self.assertLessEqual(len(goal.objective), 2_000)

    # -- checkpoints ---------------------------------------------------------

    def test_done_checkpoint_naming_criteria_requires_evidence(self) -> None:
        goal = self.goal()
        with self.assertRaises(WorkflowError) as caught:
            self.ledger.checkpoint(goal.id, "I think it works", criteria_refs=[goal.criteria[0].id])
        self.assertIn("evidence", str(caught.exception))
        self.assertEqual(self.ledger.get(goal.id).checkpoints, [])
        self.assertEqual(self.ledger.get(goal.id).criteria[0].status, "pending")

    def test_checkpoint_satisfies_only_named_criteria(self) -> None:
        goal = self.goal()
        updated = self.ledger.checkpoint(
            goal.id,
            "ran the suite",
            criteria_refs=[goal.criteria[0].id],
            evidence_refs=["tests/test_goals.py", "run:42"],
        )
        self.assertEqual(updated.criteria[0].status, "satisfied")
        self.assertEqual(updated.criteria[0].evidence_refs, ["tests/test_goals.py", "run:42"])
        self.assertEqual(updated.criteria[1].status, "pending")
        self.assertEqual(updated.checkpoints[0].status, "done")
        self.assertEqual(updated.next_action, "satisfy_criterion")
        self.assertGreater(updated.revision, goal.revision)

    def test_in_progress_checkpoint_needs_no_evidence(self) -> None:
        goal = self.goal()
        updated = self.ledger.checkpoint(goal.id, "started the parser", status="in_progress")
        self.assertEqual(updated.checkpoints[0].status, "in_progress")
        self.assertTrue(all(item.status == "pending" for item in updated.criteria))

    def test_checkpoint_refusals(self) -> None:
        goal = self.goal()
        for args, kwargs in [
            ((goal.id, ""), {}),
            ((goal.id, "  "), {}),
            ((goal.id, "x"), {"status": "finished"}),
            ((goal.id, "x"), {"criteria_refs": ["AC999-nope"]}),
            (("goal-missing", "x"), {}),
        ]:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(WorkflowError):
                    self.ledger.checkpoint(*args, **kwargs)
        self.assertEqual(self.ledger.get(goal.id).checkpoints, [])

    def test_mutation_id_replays_instead_of_appending_twice(self) -> None:
        goal = self.goal()
        first = self.ledger.checkpoint(goal.id, "one attempt", mutation_id="retry-1")
        second = self.ledger.checkpoint(goal.id, "one attempt", mutation_id="retry-1")
        self.assertEqual(len(first.checkpoints), 1)
        self.assertEqual(len(second.checkpoints), 1)
        self.assertEqual(second.checkpoints[0].id, "retry-1")
        with self.assertRaises(WorkflowError):  # same retry token, different content
            self.ledger.checkpoint(goal.id, "a different attempt", mutation_id="retry-1")

    def test_mutation_id_derivation_survives_unsafe_ids(self) -> None:
        goal = self.goal()
        updated = self.ledger.checkpoint(goal.id, "loop cycle", mutation_id="session/12:34")
        self.assertEqual(len(updated.checkpoints), 1)
        self.assertTrue(updated.checkpoints[0].id.startswith("cp-"))

    def test_expected_revision_guard(self) -> None:
        goal = self.goal()
        self.ledger.checkpoint(goal.id, "moved", status="in_progress", expected_revision=goal.revision)
        with self.assertRaises(StaleRevisionError):
            self.ledger.checkpoint(goal.id, "stale", status="in_progress", expected_revision=goal.revision)
        self.assertEqual(len(self.ledger.get(goal.id).checkpoints), 1)

    def test_checkpoint_tail_is_bounded(self) -> None:
        goal = self.goal()
        for index in range(MAX_CHECKPOINTS + 6):
            goal = self.ledger.checkpoint(goal.id, f"cycle {index}", status="in_progress")
        self.assertEqual(len(goal.checkpoints), MAX_CHECKPOINTS)
        self.assertEqual(goal.checkpoints[0].summary, f"cycle {6}")

    def test_satisfied_criteria_accumulate_evidence(self) -> None:
        goal = self.goal()
        ref = goal.criteria[0].id
        self.ledger.checkpoint(goal.id, "first", criteria_refs=[ref], evidence_refs=["a"])
        second = self.ledger.checkpoint(goal.id, "second", criteria_refs=[ref], evidence_refs=["b"])
        self.assertEqual(second.criteria[0].evidence_refs, ["a", "b"])

    # -- blockers ------------------------------------------------------------

    def test_block_is_non_terminal_and_resumes_the_goal(self) -> None:
        goal = self.ledger.checkpoint(self.goal().id, "work", status="in_progress")
        blocked = self.ledger.block(goal.id, "provider is down", evidence_refs=["status:503"])
        self.assertEqual(blocked.status, "blocked")
        self.assertNotIn(blocked.status, TERMINAL_GOAL_STATUSES)
        self.assertEqual(blocked.next_action, "resolve_blocker")
        self.assertEqual([item.status for item in blocked.blockers], ["active"])
        gate = self.ledger.gate(blocked.id)
        self.assertFalse(gate.ready)
        self.assertEqual(gate.next_action, "resolve_blocker")
        self.assertEqual(gate.active_blockers[0]["summary"], "provider is down")
        # blocked is recoverable: further checkpoints are exactly how it clears
        moved = self.ledger.checkpoint(blocked.id, "tried a mirror", status="in_progress")
        self.assertEqual(moved.status, "blocked")

    def test_resolve_blocker_returns_the_goal_to_active(self) -> None:
        goal = self.goal()
        blocked = self.ledger.block(goal.id, "waiting on review")
        resolved = self.ledger.resolve_blocker(blocked.id, blocked.blockers[0].id, "review landed")
        self.assertEqual(resolved.status, "active")
        self.assertEqual([item.status for item in resolved.blockers], ["resolved"])
        self.assertEqual(resolved.blockers[0].resolution, "review landed")
        self.assertEqual(resolved.next_action, "record_checkpoint")

    def test_blocker_refusals(self) -> None:
        goal = self.goal()
        blocked = self.ledger.block(goal.id, "stuck")
        for args in ((goal.id, ""), (goal.id, "  "), ("goal-nope", "x")):
            with self.subTest(args=args):
                with self.assertRaises(WorkflowError):
                    self.ledger.block(*args)
        with self.assertRaises(WorkflowError):
            self.ledger.resolve_blocker(blocked.id, "blk-nope")
        with self.assertRaises(WorkflowError):
            self.ledger.resolve_blocker(blocked.id, "")
        first = blocked.blockers[0].id
        self.ledger.resolve_blocker(blocked.id, first)
        with self.assertRaises(WorkflowError):
            self.ledger.resolve_blocker(blocked.id, first)

    def test_dead_end_blocker_offers_failure_not_completion(self) -> None:
        goal = self.goal()
        self.ledger.checkpoint(goal.id, "target absent", status="failed")
        blocked = self.ledger.block(goal.id, "nothing to point at")
        gate = self.ledger.gate(blocked.id)
        self.assertEqual(gate.next_action, "mark_failed")
        self.assertFalse(gate.ready)

    # -- completion gate -----------------------------------------------------

    def test_gate_refuses_done_while_required_criterion_has_no_evidence(self) -> None:
        goal = self.goal()
        gate = self.ledger.gate(goal.id)
        self.assertFalse(gate.ready)
        self.assertEqual(len(gate.missing_criteria), 2)
        self.assertEqual(gate.next_action, "record_checkpoint")
        with self.assertRaises(WorkflowError) as caught:
            self.ledger.complete(goal.id, "trust me")
        self.assertIn("completion refused", str(caught.exception))
        self.assertEqual(self.ledger.get(goal.id).status, "active")

    def test_satisfied_without_evidence_still_blocks(self) -> None:
        # A hand-marked satisfied criterion that cites nothing is a claim, not proof.
        goal = self.ledger.create("Claimed done", [{"summary": "tests pass", "status": "satisfied"}])
        gate = self.ledger.gate(goal.id)
        self.assertEqual(gate.missing_criteria[0]["id"], goal.criteria[0].id)
        self.assertFalse(gate.ready)
        with self.assertRaises(WorkflowError):
            self.ledger.complete(goal.id, "already satisfied", evidence_refs=["nothing"])

    def test_complete_requires_final_evidence_and_records_a_checkpoint(self) -> None:
        goal = self.goal(["only one"])
        satisfied = self.ledger.checkpoint(
            goal.id, "proven", criteria_refs=[goal.criteria[0].id], evidence_refs=["test:1"]
        )
        self.assertTrue(self.ledger.gate(satisfied.id).ready)
        with self.assertRaises(WorkflowError):
            self.ledger.complete(satisfied.id, "done now")
        done = self.ledger.complete(satisfied.id, "shipped", evidence_refs=["release:9"])
        self.assertEqual(done.status, "complete")
        self.assertEqual(done.next_action, "show_status")
        self.assertEqual(done.resolution_summary, "shipped")
        self.assertEqual(done.checkpoints[-1].evidence_refs, ["release:9"])
        self.assertEqual(self.ledger.gate(done.id).summary, "Goal is ready for completion.")
        for verb in (
            lambda: self.ledger.checkpoint(done.id, "more"),
            lambda: self.ledger.block(done.id, "late"),
            lambda: self.ledger.complete(done.id, "again", evidence_refs=["x"]),
            lambda: self.ledger.cancel(done.id),
            lambda: self.ledger.fail(done.id, "target_absent", "late"),
        ):
            with self.assertRaises(WorkflowError):
                verb()

    def test_complete_bypasses_the_gate_only_when_told_to(self) -> None:
        goal = self.goal()
        forced = self.ledger.complete(goal.id, "operator override", require_gate=False)
        self.assertEqual(forced.status, "complete")

    def test_gate_reports_unverified_units(self) -> None:
        goal = self.goal(["tests pass"])
        satisfied = self.ledger.checkpoint(
            goal.id, "proven", criteria_refs=[goal.criteria[0].id], evidence_refs=["test:1"]
        )
        units = [
            {"id": "unit-1", "statement": "run the suite", "status": "observed_passed", "goal_id": goal.id},
            {"id": "unit-2", "statement": "type check", "status": "not_observed", "goal_id": goal.id},
            {"id": "unit-3", "statement": "foreign", "status": "configured", "goal_id": "goal-other"},
            {"id": "unit-4", "statement": "unowned", "status": "configured"},
        ]
        gate = self.ledger.gate(goal.id, units=units)
        self.assertFalse(gate.ready)
        self.assertEqual(gate.next_action, "verify_unit")
        self.assertEqual([item["id"] for item in gate.unverified_units], ["unit-2", "unit-4"])
        self.assertEqual(gate.unverified_units[0]["status"], "not_observed")
        ready = self.ledger.gate(goal.id, units=[units[0]])
        self.assertTrue(ready.ready)
        self.assertEqual(ready.next_action, "mark_complete")

    def test_gate_renders_gap_summary(self) -> None:
        goal = self.goal()
        self.ledger.block(goal.id, "blocked here")
        gate = self.ledger.gate(goal.id)
        self.assertIn("required criteria have no evidence", gate.summary)
        self.assertIn("blockers are still active", gate.summary)
        self.assertLessEqual(len(gate.summary), MAX_SUMMARY_CHARS)
        self.assertEqual(gate.goal_id, goal.id)
        self.assertEqual(gate.status, "blocked")

    # -- terminal verdicts ---------------------------------------------------

    def test_fail_needs_a_reason_code(self) -> None:
        goal = self.goal()
        with self.assertRaises(WorkflowError):
            self.ledger.fail(goal.id, "budget_exhausted", "out of attempts")
        with self.assertRaises(WorkflowError):
            self.ledger.fail(goal.id, "target_absent", "   ")
        failed = self.ledger.fail(goal.id, "target_absent", "the repository does not exist")
        self.assertEqual(failed.status, "failed")
        self.assertIn(failed.status, TERMINAL_GOAL_STATUSES)
        self.assertEqual(failed.failure_reason_code, "target_absent")
        self.assertEqual(failed.next_action, "show_status")
        self.assertIn("target_absent", self.ledger.render(failed))
        with self.assertRaises(WorkflowError):
            self.ledger.checkpoint(failed.id, "try again")

    def test_cancel_is_terminal_for_everyone(self) -> None:
        goal = self.goal()
        cancelled = self.ledger.cancel(goal.id, "superseded by other work")
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(cancelled.next_action, "show_status")
        self.assertEqual(cancelled.resolution_summary, "superseded by other work")
        gate = self.ledger.gate(cancelled.id)
        self.assertFalse(gate.ready)
        self.assertEqual(gate.next_action, "show_status")
        self.assertIn("cancelled", gate.summary)
        with self.assertRaises(WorkflowError):
            self.ledger.cancel(cancelled.id)

    # -- reads ---------------------------------------------------------------

    def test_list_active_and_filters(self) -> None:
        first = self.goal(["a"])
        self.now[0] += 1
        second = self.goal(["b"])
        self.now[0] += 1
        third = self.goal(["c"])
        self.ledger.block(third.id, "stuck")
        self.assertEqual([item.id for item in self.ledger.list()], [third.id, second.id, first.id])
        self.assertEqual([item.id for item in self.ledger.list(status="blocked")], [third.id])
        self.assertEqual(len(self.ledger.list(limit=2)), 2)
        self.assertEqual({item.id for item in self.ledger.active()}, {first.id, second.id, third.id})
        self.ledger.complete(second.id, "override", require_gate=False)
        self.assertNotIn(second.id, {item.id for item in self.ledger.active()})
        with self.assertRaises(WorkflowError):
            self.ledger.list(status="paused")
        with self.assertRaises(WorkflowError):
            self.ledger.get("goal-nope")

    def test_workspaces_are_isolated(self) -> None:
        other_ws = self.base / "other"
        other_ws.mkdir()
        other = GoalLedger(self.root, other_ws)
        goal = self.goal()
        self.assertEqual(other.list(), [])
        with self.assertRaises(WorkflowError):
            other.get(goal.id)

    # -- journey and brief ---------------------------------------------------

    def test_journey_stages(self) -> None:
        goal = self.goal()
        journey = self.ledger.journey(goal.id)
        self.assertEqual(journey.stage, "intent")
        self.assertEqual(journey.freshness, "unknown")  # no clock given, nothing inferred
        self.assertEqual(journey.next_action, "record_checkpoint")
        prepared = self.ledger.journey(goal.id, open_gate_ids=["gate-1"])
        self.assertEqual(prepared.stage, "preparation")
        self.assertEqual(prepared.next_action, "answer_gate")
        self.assertEqual(prepared.open_gates, ["gate-1"])
        moved = self.ledger.checkpoint(goal.id, "work", status="in_progress")
        active = self.ledger.journey(moved.id, now=self.now[0])
        self.assertEqual(active.stage, "activity")
        self.assertEqual(active.freshness, "fresh")
        self.assertEqual(active.next_action, "satisfy_criterion")
        self.assertEqual(active.criteria_satisfied, 0)
        self.assertEqual(active.criteria_total, 2)
        self.assertEqual(len(active.last_checkpoints), 1)
        blocked = self.ledger.journey(self.ledger.block(moved.id, "stuck").id)
        self.assertEqual(blocked.stage, "blocked")
        done = self.ledger.journey(self.ledger.cancel(moved.id).id)
        self.assertEqual(done.stage, "cancelled")

    def test_journey_refuses_verified_complete_without_evidence(self) -> None:
        goal = self.ledger.create("Overclaim", ["one"])
        self.ledger.checkpoint(goal.id, "claim", status="in_progress")
        stamped = self.ledger.complete(goal.id, "override", require_gate=False)
        journey = self.ledger.journey(stamped.id)
        self.assertEqual(journey.stage, "blocked")  # complete with no evidence is not verified
        self.assertEqual(journey.next_action, "show_status")

    def test_journey_freshness(self) -> None:
        goal = self.goal()
        stale = self.ledger.journey(goal.id, now=self.now[0] + 2 * DAY, stale_after_seconds=DAY)
        self.assertEqual(stale.freshness, "stale")
        self.assertEqual(self.ledger.journey(goal.id, now=self.now[0] + DAY / 2).freshness, "fresh")
        self.assertEqual(self.ledger.journey(goal.id, now=self.now[0] - 10).freshness, "unknown")
        broken = Goal.from_record({**goal.to_dict(), "goal_id": goal.id, "updated_at": "not-a-stamp"})
        self.assertEqual(self.ledger._freshness(broken, now=self.now[0], stale_after_seconds=DAY), "unknown")

    def test_continuation_brief_is_deterministic_and_bounded(self) -> None:
        goal = self.goal()
        self.ledger.checkpoint(goal.id, "half way", status="in_progress")
        brief = self.ledger.continuation_brief(goal.id, limit=2)
        self.assertEqual(brief, self.ledger.continuation_brief(goal.id, limit=2))
        self.assertIn(goal.id, brief)
        self.assertIn("Next action: satisfy_criterion", brief)
        self.assertIn("half way", brief)
        self.assertIn("Completion gate:", brief)
        self.assertNotIn("\x1b", brief)
        self.assertTrue(brief.endswith("test evidence."))
        with self.assertRaises(WorkflowError):
            self.ledger.continuation_brief("goal-nope")

    def test_render_rows_and_headers(self) -> None:
        goal = self.goal()
        blocked = self.ledger.block(goal.id, "stuck")
        text = self.ledger.render(blocked, gate=self.ledger.gate(blocked.id))
        self.assertIn("○", text)  # nothing satisfied yet
        self.assertIn("▲", text)  # the active blocker
        self.assertIn("Completion gate:", text)
        satisfied = self.ledger.checkpoint(
            blocked.id, "proven", criteria_refs=[goal.criteria[0].id], evidence_refs=["test:1"]
        )
        self.assertIn("●", self.ledger.render(satisfied))
        self.assertIn("evidence:1", self.ledger.render(satisfied))
        ascii_text = self.ledger.render(satisfied, glyphs="ascii")
        for glyph in ("●", "○", "▲"):
            self.assertNotIn(glyph, ascii_text)
        self.assertIn("*", ascii_text)
        self.assertEqual(
            self.ledger.headers(), ["goal", "status", "criteria", "blockers", "next", "updated"]
        )
        rows = self.ledger.rows([blocked])
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), len(self.ledger.headers()))
        self.assertEqual(rows[0][2], "0/2")
        self.assertEqual(rows[0][3], "1")
        self.assertEqual(self.ledger.rows([]), [])
        for row in rows[0]:
            self.assertNotIn("\x1b", row)

    def test_from_record_rejects_unusable_records(self) -> None:
        goal = self.goal()
        for tampered in (
            {**goal.to_dict(), "status": "paused"},
            {**goal.to_dict(), "next_action": "do_anything"},
            {**goal.to_dict(), "criteria": [{"id": "AC001", "summary": "x", "status": "done"}]},
            {**goal.to_dict(), "checkpoints": [{"id": "cp", "status": "finished", "summary": "x"}]},
            {**goal.to_dict(), "blockers": [{"id": "blk", "status": "gone", "summary": "x"}]},
        ):
            with self.subTest(tampered=tampered):
                with self.assertRaises(WorkflowError):
                    Goal.from_record(tampered)
        self.assertEqual(Goal.from_record({**goal.to_dict(), "goal_id": "goal-2"}).id, "goal-2")
        loose = Goal.from_record({"goal_id": goal.id, "status": "active", "next_action": "show_status"})
        self.assertEqual(loose.criteria, [])
        self.assertEqual(loose.revision, 0)
        with self.assertRaises(WorkflowError):
            Goal.from_record("not a mapping")  # type: ignore[arg-type]

    def test_dataclass_shapes_are_constructible(self) -> None:
        criterion = Criterion(id="AC001", summary="s")
        checkpoint = Checkpoint(id="cp", at="t", status="done", summary="s")
        blocker = Blocker(id="blk", at="t", status="active", summary="s")
        self.assertEqual(criterion.to_dict()["evidence_refs"], [])
        self.assertEqual(checkpoint.to_dict()["criteria_refs"], [])
        self.assertEqual(blocker.to_dict()["resolution"], "")
        gate = CompletionGate(goal_id="g", ready=False, status="active")
        self.assertEqual(gate.missing_criteria, [])
        journey = GoalJourney(goal_id="g", stage="intent", freshness="unknown", criteria_satisfied=0, criteria_total=0)
        self.assertEqual(journey.open_gates, [])
        self.assertEqual(journey.next_action, "record_checkpoint")

    # -- tools ---------------------------------------------------------------

    def test_tool_surface(self) -> None:
        tools = self.tools(require_approval=False)
        self.assertEqual(
            set(tools), {"goal_create", "goal_checkpoint", "goal_block", "goal_status", "goal_complete"}
        )
        self.assertTrue(tools["goal_status"].read_only)
        self.assertFalse(tools["goal_create"].read_only)
        self.assertTrue(tools["goal_checkpoint"].destructive)
        for tool in tools.values():
            spec = tool.api_spec()
            self.assertEqual(spec["function"]["name"], tool.name)
            self.assertEqual(spec["function"]["parameters"]["additionalProperties"], False)

    def test_tools_record_and_read(self) -> None:
        tools = self.tools(require_approval=False)
        created = tools["goal_create"].handler({"objective": "Ship it", "criteria": ["tests pass"]}, self.ctx())
        self.assertTrue(created.ok)
        goal_id = created.meta["goal_id"]
        self.assertIn("Goal ", created.content)
        stat = tools["goal_status"].handler({"goal": goal_id}, self.ctx())
        self.assertTrue(stat.ok)
        self.assertIn("no evidence", stat.content)
        listed = tools["goal_status"].handler({}, self.ctx())
        self.assertEqual(listed.meta["count"], 1)
        self.assertFalse(tools["goal_status"].handler({"goal": "goal-nope"}, self.ctx()).ok)
        ref = self.ledger.get(goal_id).criteria[0].id
        refused = tools["goal_checkpoint"].handler(
            {"goal": goal_id, "summary": "done", "criteria": ["AC001-nonsense"]}, self.ctx()
        )
        self.assertFalse(refused.ok)
        self.assertIn("unknown acceptance criteria", refused.content)
        no_evidence = tools["goal_checkpoint"].handler(
            {"goal": goal_id, "summary": "done", "criteria": [ref]}, self.ctx()
        )
        self.assertFalse(no_evidence.ok)
        self.assertIn("evidence", no_evidence.content)

    def test_goal_tools_satisfy_the_gate_end_to_end(self) -> None:
        tools = self.tools(require_approval=False)
        created = tools["goal_create"].handler({"objective": "Ship it", "criteria": ["tests pass"]}, self.ctx())
        goal = self.ledger.get(created.meta["goal_id"])
        ref = goal.criteria[0].id
        marked = tools["goal_checkpoint"].handler(
            {"goal": goal.id, "summary": "ran pytest", "criteria": [ref], "evidence": ["tests/test_goals.py"]},
            self.ctx(),
        )
        self.assertTrue(marked.ok)
        self.assertTrue(marked.meta["ready"])
        self.assertEqual(marked.meta["next_action"], "mark_complete")
        no_evidence = tools["goal_complete"].handler({"goal": goal.id, "summary": "done"}, self.ctx())
        self.assertFalse(no_evidence.ok)
        done = tools["goal_complete"].handler({"goal": goal.id, "summary": "done", "evidence": ["release:1"]}, self.ctx())
        self.assertTrue(done.ok)
        self.assertEqual(self.ledger.get(goal.id).status, "complete")
        blocked = tools["goal_block"].handler({"goal": goal.id, "summary": "late"}, self.ctx())
        self.assertFalse(blocked.ok)

    def test_mutating_tools_require_approval(self) -> None:
        asked: list[ApprovalRequest] = []
        tools = self.tools(require_approval=True, journal=self.journal)
        ctx = self.ctx(approval_mode="ask", confirm=lambda request: asked.append(request) or False)
        refused = tools["goal_create"].handler({"objective": "Nope", "criteria": ["x"]}, ctx)
        self.assertFalse(refused.ok)
        self.assertTrue(refused.meta["denied"])
        self.assertEqual(len(asked), 1)
        self.assertEqual(asked[0].action, "goal_create")
        self.assertEqual(self.ledger.list(), [])
        self.assertEqual(self.journal.counts()["goal_updated"], 1)  # the refusal was observed
        self.assertFalse(
            tools["goal_checkpoint"].handler({"goal": "g", "summary": "s"}, self.ctx(read_only=True)).ok
        )

    def test_auto_mode_records_without_asking(self) -> None:
        tools = self.tools(require_approval=True)
        result = tools["goal_create"].handler({"objective": "Auto", "criteria": ["x"]}, self.ctx(approval_mode="auto"))
        self.assertTrue(result.ok)

    def test_read_only_toolset_refuses_every_change(self) -> None:
        tools = self.tools(require_approval=False, read_only=True)
        self.assertTrue(tools["goal_create"].read_only)
        self.assertFalse(tools["goal_create"].destructive)
        for name, args in (
            ("goal_create", {"objective": "x", "criteria": ["y"]}),
            ("goal_checkpoint", {"goal": "g-1", "summary": "s"}),
            ("goal_block", {"goal": "g-1", "summary": "s"}),
            ("goal_complete", {"goal": "g-1", "summary": "s"}),
        ):
            result: ToolResult = tools[name].handler(args, self.ctx(approval_mode="auto"))
            with self.subTest(name=name):
                self.assertFalse(result.ok)
                self.assertIn("Read-only", result.content)
        self.assertEqual(self.ledger.list(), [])

    def test_tools_reject_malformed_arguments(self) -> None:
        tools = self.tools(require_approval=False)
        ctx = self.ctx()
        for name, args in (
            ("goal_create", {"objective": 5, "criteria": ["x"]}),
            ("goal_create", {"objective": "x", "criteria": [7]}),
            ("goal_checkpoint", {"goal": "", "summary": "s"}),
            ("goal_checkpoint", {"goal": "g", "summary": 3}),
            ("goal_block", {"goal": "g"}),
            ("goal_complete", {"goal": "g", "summary": 3}),
            ("goal_status", {"goal": "goal-nope"}),
        ):
            with self.subTest(name=name, args=args):
                self.assertFalse(tools[name].handler(args, ctx).ok)


if __name__ == "__main__":
    unittest.main()
