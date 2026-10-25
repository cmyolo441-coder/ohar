from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from ohar.types import ApprovalRequest, MAX_SUBAGENTS, ToolContext
from ohar.workflows.blocks import REASON_DOMAINS, BlockLog
from ohar.workflows.gates import DecisionGates
from ohar.workflows.goalloop import (
    LOOPABILITY_STATES,
    LOOP_PHASES,
    MAX_BRIEF_CHARS,
    STOP_REASONS,
    SURFACES,
    GoalLoop,
    LoopBudget,
    LoopCycle,
    LoopReport,
    LoopabilityAssessment,
    TurnFn,
    TurnOutcome,
    agent_turn,
    assess_loopability,
    make_loop_tool,
    validate_assessment,
)
from ohar.workflows.goals import NEXT_ACTIONS, GoalLedger
from ohar.workflows.journal import ActivityJournal

OBJECTIVE = "Improve the repository README until every test passes"
CRITERIA = ["install steps verified", "first example runs clean"]


class Clock:
    """A monotonic stand-in: each call advances by a fixed step."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


class FakeUnit:
    """The documented public shape of workflows.verification.Unit."""

    def __init__(
        self,
        unit_id: str,
        *,
        goal_id: str,
        criterion_ref: str = "",
        statement: str = "pytest -q passes",
        satisfied: bool = False,
    ) -> None:
        self.id = unit_id
        self.goal_id = goal_id
        self.criterion_ref = criterion_ref
        self.statement = statement
        self.satisfied = satisfied
        self.status = "observed_passed" if satisfied else "not_observed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "goal_id": self.goal_id,
            "criterion_ref": self.criterion_ref,
            "status": self.status,
        }


class FakeUnits:
    """A VerificationLedger stand-in, so these tests can control exactly which
    units observed passing without depending on how observations are ingested."""

    def __init__(self, rows: list[FakeUnit] | None = None) -> None:
        self.rows = list(rows or [])
        self.reconciled: list[str] = []

    def add(self, row: FakeUnit) -> None:
        self.rows.append(row)

    def units(self, *, goal_id: str | None = None, status: str | None = None, limit: int = 32) -> list[FakeUnit]:
        _ = status
        rows = [row for row in self.rows if not goal_id or row.goal_id == goal_id]
        return rows[: max(1, limit)]

    def reconcile_all(self, *, goal_id: str = "", now: float | None = None) -> list[FakeUnit]:
        self.reconciled.append(goal_id)
        return self.units(goal_id=goal_id)


class StubGoal:
    """A Goal stand-in for the ledger states `create()` deliberately refuses."""

    def __init__(self, criteria: list[Any] | None = None, *, status: str = "active") -> None:
        self.id = "goal-stub"
        self.status = status
        self.objective = "keep improving the repo install docs and verify the friction"
        self.criteria = list(criteria or [])
        self.checkpoints: list[Any] = []
        self.blockers: list[Any] = []
        self.next_action = "satisfy_criterion"
        self.session_id = ""


class StubLedger:
    def __init__(self, goal: StubGoal) -> None:
        self.goal = goal

    def get(self, goal_id: str) -> StubGoal:
        if goal_id != self.goal.id:
            raise ValueError(f"unknown goal: {goal_id}")
        return self.goal

    def gate(self, goal_id: str, *, units: Any = None) -> Any:
        raise ValueError("this stub has no completion gate")


class LedgerCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "workflows"
        self.ws = self.base / "ws"
        self.ws.mkdir(parents=True)
        self.journal = ActivityJournal(self.root, self.ws)
        self.gates = DecisionGates(self.root, self.ws, journal=self.journal)
        self.blocks = BlockLog(self.root, self.ws, journal=self.journal)
        self.units = FakeUnits()
        self.goals = GoalLedger(self.root, self.ws, journal=self.journal)
        self.goal = self.goals.create(OBJECTIVE, CRITERIA, session_id="s1")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def loop(self, **kwargs: Any) -> GoalLoop:
        kwargs.setdefault("gates", self.gates)
        kwargs.setdefault("blocks", self.blocks)
        kwargs.setdefault("units", self.units)
        kwargs.setdefault("journal", self.journal)
        kwargs.setdefault("budget", LoopBudget(max_cycles=4, no_progress_limit=2))
        return GoalLoop(self.goals, **kwargs)

    def ctx(self, **kwargs: Any) -> ToolContext:
        kwargs.setdefault("workspace", self.ws)
        kwargs.setdefault("approval_mode", "auto")
        return ToolContext(**kwargs)

    def work_rows(self, report: LoopReport) -> list[LoopCycle]:
        return [cycle for cycle in report.cycles if cycle.phase != "stop"]

    def open_gate(self, subject: str) -> Any:
        return self.gates.open(
            subject=subject, question_code="allow_write", action="loop_run", arg_digest="d1"
        )

    def block_during_turn(self) -> Any:
        def turn(prompt: str) -> TurnOutcome:
            self.goals.block(self.goal.id, "the target file vanished")
            return TurnOutcome("cannot continue")

        return turn

    def gate_in_turn(self) -> Any:
        def turn(prompt: str) -> TurnOutcome:
            self.open_gate(self.goal.id)
            return TurnOutcome("needs a decision")

        return turn


class VocabularyTests(unittest.TestCase):
    def test_closed_vocabularies_are_exact(self) -> None:
        self.assertEqual(LOOP_PHASES, ("orient", "act", "verify", "stop"))
        self.assertEqual(
            LOOPABILITY_STATES,
            (
                "direct_task",
                "loopable",
                "needs_clarification",
                "external_wait_only",
                "north_star_only",
                "recurring",
            ),
        )
        self.assertEqual(
            STOP_REASONS,
            (
                "gate_ready",
                "budget_cycles",
                "budget_tokens",
                "budget_seconds",
                "no_progress",
                "gate_open",
                "blocked",
                "cancelled",
                "provider_error",
                "invalid_goal",
            ),
        )
        self.assertEqual(SURFACES, ("single_turn", "goal_loop", "recurring", "external_wait"))
        self.assertEqual(MAX_BRIEF_CHARS, 6000)

    def test_loop_vocabulary_is_ledger_vocabulary(self) -> None:
        # The loop answers with goal-ledger verbs and block-log domains; a word
        # outside either would be one the rest of the package cannot read.
        from ohar.workflows.goalloop import _BLOCK_DOMAINS, _STOP_NEXT_ACTION

        for reason in STOP_REASONS:
            self.assertIn(_STOP_NEXT_ACTION[reason], NEXT_ACTIONS, reason)
        for reason, domain in _BLOCK_DOMAINS.items():
            self.assertIn(reason, STOP_REASONS)
            self.assertIn(domain, REASON_DOMAINS)


class LoopBudgetTests(unittest.TestCase):
    def test_defaults(self) -> None:
        budget = LoopBudget()
        self.assertEqual(
            (budget.max_cycles, budget.max_total_tokens, budget.max_seconds, budget.no_progress_limit),
            (4, 200_000, 900.0, 2),
        )

    def test_clamp_bounds_without_raising(self) -> None:
        clamped = LoopBudget(
            max_cycles=10_000, max_total_tokens=-5, max_seconds=99_999.0, no_progress_limit=0
        ).clamp()
        self.assertEqual(
            (clamped.max_cycles, clamped.max_total_tokens, clamped.max_seconds, clamped.no_progress_limit),
            (32, 1_000, 3600.0, 1),
        )

    def test_clamp_repairs_garbage_to_defaults(self) -> None:
        clamped = LoopBudget(
            max_cycles="many",  # type: ignore[arg-type]
            max_total_tokens=True,  # type: ignore[arg-type]
            max_seconds=float("nan"),
            no_progress_limit=None,  # type: ignore[arg-type]
        ).clamp()
        self.assertEqual(
            (clamped.max_cycles, clamped.max_total_tokens, clamped.no_progress_limit), (4, 200_000, 2)
        )
        self.assertEqual(clamped.max_seconds, 900.0)

    def test_clamp_returns_a_new_object(self) -> None:
        original = LoopBudget(max_cycles=2)
        self.assertIsNot(original.clamp(), original)
        self.assertEqual(original.max_cycles, 2)


class AssessLoopabilityTests(unittest.TestCase):
    def test_direct_task_is_not_a_loop(self) -> None:
        answer = assess_loopability("rename the config loader variable")
        self.assertEqual((answer.state, answer.surface), ("direct_task", "single_turn"))
        self.assertTrue(answer.stop_condition)
        self.assertEqual(validate_assessment(answer.to_dict()), [])

    def test_declared_criteria_decide_between_one_turn_and_a_loop(self) -> None:
        narrow = assess_loopability("rename the entrypoint variable", criteria=["no stale names"])
        self.assertEqual((narrow.state, narrow.surface), ("direct_task", "single_turn"))
        broad = assess_loopability(
            "reduce flaky failures in the test suite",
            criteria=["no reruns needed", "report published"],
        )
        self.assertEqual((broad.state, broad.surface), ("loopable", "goal_loop"))
        self.assertIn("criterion", broad.stop_condition.lower())
        backed = assess_loopability(
            "reduce flaky failures in the test suite",
            criteria=["no reruns needed"],
            verification_available=True,
        )
        self.assertEqual(backed.state, "loopable")

    def test_cadence_beats_loop_shape(self) -> None:
        answer = assess_loopability("check the repo build every morning and verify it")
        self.assertEqual((answer.state, answer.surface), ("recurring", "recurring"))
        self.assertIn("schedule", answer.stop_condition.lower())

    def test_heartbeat_has_no_completion(self) -> None:
        answer = assess_loopability("keep watching the health check endpoint")
        self.assertEqual((answer.state, answer.surface), ("recurring", "recurring"))
        self.assertIn("heartbeat", answer.stop_condition.lower())

    def test_stated_stop_criterion_is_a_goal_not_a_reframe(self) -> None:
        answer = assess_loopability("work on the docs until the tests pass")
        self.assertEqual((answer.state, answer.surface), ("loopable", "goal_loop"))
        self.assertIn("until the tests pass", answer.stop_condition)

    def test_external_wait_only(self) -> None:
        answer = assess_loopability("see how the market response lands after launch")
        self.assertEqual((answer.state, answer.surface), ("external_wait_only", "external_wait"))
        self.assertTrue(answer.reframe)
        bounded = assess_loopability("improve the install docs", externally_bounded=True)
        self.assertEqual(bounded.state, "external_wait_only")

    def test_north_star_only(self) -> None:
        answer = assess_loopability("grow this to 10k github stars and world-class revenue")
        self.assertEqual((answer.state, answer.surface), ("north_star_only", "single_turn"))
        self.assertIn("bounded arena", answer.reframe)

    def test_loop_shape_without_an_end_signal_asks_first(self) -> None:
        answer = assess_loopability("keep improving the repo install docs and verify the friction")
        self.assertEqual(answer.state, "needs_clarification")
        self.assertIn("criteria", answer.reframe.lower())
        self.assertEqual(
            assess_loopability(
                "keep improving the repo install docs and verify the friction",
                verification_available=True,
            ).state,
            "loopable",
        )

    def test_empty_text_still_answers(self) -> None:
        answer = assess_loopability("   ")
        self.assertEqual((answer.state, answer.surface), ("needs_clarification", "single_turn"))
        self.assertEqual(validate_assessment(answer.to_dict()), [])

    def test_untrusted_text_is_cleaned(self) -> None:
        answer = assess_loopability("\x1b[31mfix the failing test\x00 in the repo\x1b[0m" + "!" * 900)
        joined = answer.reason + answer.stop_condition + answer.reframe
        self.assertNotIn("\x1b", joined)
        self.assertNotIn("\x00", joined)
        self.assertTrue(all(len(value) <= 400 for value in answer.to_dict().values()))

    def test_no_assessment_can_fail_its_own_honesty_check(self) -> None:
        corpus = [
            "rename the config loader variable",
            "reduce flaky failures in the test suite",
            "check the repo build every morning and verify it",
            "keep watching the health check endpoint",
            "work on the docs until the tests pass",
            "see how the market response lands after launch",
            "grow this to 10k github stars and world-class revenue",
            "keep improving the repo install docs and verify the friction",
            "hi",
            "make it better",
            "",
            "polish the api package and measure adoption",
        ]
        for text in corpus:
            for criteria in ((), ("one",), ("one", "two")):
                for verification in (False, True):
                    for external in (False, True):
                        answer = assess_loopability(
                            text,
                            criteria=criteria,
                            verification_available=verification,
                            externally_bounded=external,
                        )
                        with self.subTest(text=text, criteria=len(criteria), verification=verification):
                            self.assertIn(answer.state, LOOPABILITY_STATES)
                            self.assertIn(answer.surface, SURFACES)
                            self.assertEqual(validate_assessment(answer.to_dict()), [])

    def test_to_dict_keys_are_the_contract(self) -> None:
        self.assertEqual(
            set(LoopabilityAssessment("loopable", "goal_loop").to_dict()),
            {"state", "surface", "stop_condition", "reason", "reframe"},
        )


class ValidateAssessmentTests(unittest.TestCase):
    @staticmethod
    def valid() -> dict[str, Any]:
        return {
            "state": "loopable",
            "surface": "goal_loop",
            "stop_condition": "Stops when every criterion is verified or a blocker is recorded.",
            "reason": "Criteria and a verification signal exist.",
            "reframe": "",
        }

    def test_valid_assessment_has_no_errors(self) -> None:
        self.assertEqual(validate_assessment(self.valid()), [])

    def test_non_mapping(self) -> None:
        self.assertEqual(validate_assessment("loopable"), ["assessment must be a mapping"])

    def test_missing_and_unsupported_keys(self) -> None:
        broken = {key: value for key, value in self.valid().items() if key != "reason"}
        broken["confidence"] = "high"
        errors = validate_assessment(broken)
        self.assertTrue(any("missing key(s): reason" in error for error in errors))
        self.assertTrue(any("unsupported key(s): confidence" in error for error in errors))

    def test_non_string_and_oversized_values(self) -> None:
        self.assertIn("assessment.state must be a string", validate_assessment({**self.valid(), "state": 7}))
        self.assertIn(
            "assessment.reason exceeds 400 characters",
            validate_assessment({**self.valid(), "reason": "why " * 300}),
        )

    def test_unknown_vocabulary(self) -> None:
        errors = validate_assessment({**self.valid(), "state": "vibes", "surface": "eternal"})
        self.assertIn("assessment.state is not in LOOPABILITY_STATES", errors)
        self.assertIn("assessment.surface is not in SURFACES", errors)

    def test_surface_must_match_state(self) -> None:
        errors = validate_assessment({**self.valid(), "surface": "recurring"})
        self.assertTrue(any("does not match state" in error for error in errors))

    def test_open_ended_surface_without_a_stop_condition_is_refused(self) -> None:
        cases = (
            ("loopable", "goal_loop", ""),
            ("recurring", "recurring", ""),
            ("external_wait_only", "external_wait", "   "),
        )
        for state, surface, stop in cases:
            errors = validate_assessment(
                {
                    "state": state,
                    "surface": surface,
                    "stop_condition": stop,
                    "reason": "a reason",
                    "reframe": "a reframe",
                }
            )
            with self.subTest(surface=surface):
                self.assertTrue(
                    any("without naming what ends it" in error for error in errors),
                    f"{surface} was accepted with a blank stop condition: {errors}",
                )

    def test_goal_loop_stop_condition_must_name_a_mechanism(self) -> None:
        errors = validate_assessment({**self.valid(), "stop_condition": "Keep working on this until it is done."})
        self.assertTrue(any("names no ending mechanism" in error for error in errors))

    def test_single_turn_may_have_no_stop_condition(self) -> None:
        self.assertEqual(
            validate_assessment(
                {
                    "state": "direct_task",
                    "surface": "single_turn",
                    "stop_condition": "",
                    "reason": "One bounded change.",
                    "reframe": "",
                }
            ),
            [],
        )

    def test_reframe_is_required_when_the_answer_is_not_go(self) -> None:
        errors = validate_assessment(
            {
                "state": "needs_clarification",
                "surface": "single_turn",
                "stop_condition": "",
                "reason": "Too vague.",
                "reframe": "",
            }
        )
        self.assertTrue(any("must carry a reframe" in error for error in errors))


class BriefTests(LedgerCase):
    def test_brief_is_deterministic_and_covers_the_ledger(self) -> None:
        loop = self.loop()
        first, second = loop.brief(self.goal.id), loop.brief(self.goal.id)
        self.assertEqual(first, second)
        self.assertIn("CONTINUATION BRIEF", first)
        self.assertIn(OBJECTIVE, first)
        for summary in CRITERIA:
            self.assertIn(summary, first)
        for criterion in self.goal.criteria:
            self.assertIn(criterion.id, first)
        self.assertIn("next action:", first)

    def test_brief_drops_satisfied_criteria(self) -> None:
        loop = self.loop()
        self.assertIn(CRITERIA[0], loop.brief(self.goal.id))
        self.goals.checkpoint(
            self.goal.id,
            "walked the install path",
            criteria_refs=[self.goal.criteria[0].id],
            evidence_refs=["unit:u1"],
        )
        refreshed = loop.brief(self.goal.id)
        self.assertNotIn(CRITERIA[0], refreshed)
        self.assertIn(CRITERIA[1], refreshed)

    def test_brief_names_blockers_gates_and_units(self) -> None:
        loop = self.loop()
        self.goals.block(self.goal.id, "the upstream branch was deleted")
        gate = self.open_gate(self.goal.id)
        self.units.add(FakeUnit("u-1", goal_id=self.goal.id, statement="pytest -q passes"))
        brief = loop.brief(self.goal.id)
        self.assertIn("active blockers:", brief)
        self.assertIn(gate.id, brief)
        self.assertIn("units without observed evidence:", brief)

    def test_brief_is_bounded(self) -> None:
        wide = self.goals.create(" ".join(f"arena {index}" for index in range(200)), [f"c{i}: {'x' * 200}" for i in range(8)])
        loop = self.loop()
        for index in range(16):
            self.units.add(FakeUnit(f"u-{index}", goal_id=wide.id, statement=f"{index} " + "x" * 200))
        for index in range(8):
            self.goals.block(wide.id, f"b{index} " + "y" * 200)
        self.open_gate(wide.id)
        brief = loop.brief(wide.id, limit=16)
        self.assertLessEqual(len(brief), MAX_BRIEF_CHARS)
        self.assertTrue(brief.endswith("…"), "the brief backstop never engaged")

    def test_brief_overflow_rows_are_counted_not_shown(self) -> None:
        loop = self.loop()
        for index in range(16):
            self.units.add(FakeUnit(f"u-{index}", goal_id=self.goal.id, statement=f"check {index}"))
        self.assertIn("(+1 more)", loop.brief(self.goal.id, limit=15))
        self.assertNotIn("(+1 more)", loop.brief(self.goal.id, limit=16))

    def test_unknown_goal_raises_to_the_caller(self) -> None:
        with self.assertRaises(ValueError):
            self.loop().brief("goal-nope")


class CanStartTests(LedgerCase):
    def test_open_goal_may_start(self) -> None:
        self.assertEqual(self.loop().can_start(self.goal.id), (True, ""))

    def test_unknown_goal_is_refused(self) -> None:
        allowed, why = self.loop().can_start("goal-nope")
        self.assertFalse(allowed)
        self.assertIn("not a goal", why)

    def test_goal_without_criteria_cannot_name_its_end(self) -> None:
        loop = GoalLoop(StubLedger(StubGoal()))
        allowed, why = loop.can_start("goal-stub")
        self.assertFalse(allowed)
        self.assertIn("no criteria", why)

    def test_terminal_goals_are_refused(self) -> None:
        complete = self.goals.create("ship it", ["docs updated"])
        self.goals.checkpoint(
            complete.id, "done", criteria_refs=[complete.criteria[0].id], evidence_refs=["unit:u"]
        )
        self.goals.complete(complete.id, "shipped", evidence_refs=["unit:u"])
        self.assertEqual(
            self.loop().can_start(complete.id),
            (False, "the goal is complete; no cycle can change that"),
        )
        cancelled = self.goals.create("drop it", ["docs updated"])
        self.goals.cancel(cancelled.id, "not now")
        self.assertIn("cancelled", self.loop().can_start(cancelled.id)[1])
        failed = self.goals.create("cannot do it", ["docs updated"])
        self.goals.fail(failed.id, "target_absent", "the file does not exist")
        self.assertIn("failed", self.loop().can_start(failed.id)[1])

    def test_open_gate_is_refused(self) -> None:
        gate = self.open_gate(self.goal.id)
        allowed, why = self.loop().can_start(self.goal.id)
        self.assertFalse(allowed)
        self.assertIn(gate.id, why)

    def test_prefixed_gate_subject_still_matches_the_goal(self) -> None:
        self.open_gate(f"goal:{self.goal.id}")
        allowed, why = self.loop().can_start(self.goal.id)
        self.assertFalse(allowed)
        self.assertIn("decision gate", why)

    def test_another_goals_gate_does_not_refuse_this_loop(self) -> None:
        other = self.goals.create("the neighbour goal", ["neighbour done"])
        self.open_gate(other.id)
        self.assertEqual(self.loop().can_start(self.goal.id), (True, ""))

    def test_active_blocker_is_refused(self) -> None:
        self.goals.block(self.goal.id, "waiting on the frozen check")
        allowed, why = self.loop().can_start(self.goal.id)
        self.assertFalse(allowed)
        self.assertIn("blocker", why)

    def test_resolved_blocker_allows_the_loop_again(self) -> None:
        blocked = self.goals.block(self.goal.id, "waiting on the frozen check")
        self.goals.resolve_blocker(blocked.id, blocked.blockers[0].id, "check restored")
        self.assertEqual(self.loop().can_start(self.goal.id), (True, ""))


class RunStopReasonTests(LedgerCase):
    def test_gate_ready_stops_the_loop_successfully(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=4, no_progress_limit=4))
        refs = [criterion.id for criterion in self.goal.criteria]

        def turn(prompt: str) -> TurnOutcome:
            self.assertIn("CONTINUATION BRIEF", prompt)
            self.goals.checkpoint(
                self.goal.id, "install path walked", criteria_refs=refs, evidence_refs=["unit:u"]
            )
            return TurnOutcome("verified the install steps", output_tokens=120)

        report = loop.run(self.goal.id, turn)
        self.assertEqual(report.stop_reason, "gate_ready")
        self.assertTrue(report.ready)
        self.assertEqual(report.next_action, "mark_complete")
        work = self.work_rows(report)
        self.assertEqual(len(work), 1)
        self.assertEqual(work[0].phase, "verify")
        self.assertEqual(work[0].criteria_satisfied, 2)
        self.assertTrue(work[0].checkpoint_id)
        self.assertEqual(report.cycles[-1].phase, "stop")
        self.assertTrue(report.cycles[-1].ok)

    def test_a_ready_gate_found_before_the_first_cycle_runs_no_cycle(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=4, no_progress_limit=4))
        refs = [criterion.id for criterion in self.goal.criteria]
        self.goals.checkpoint(self.goal.id, "both walked", criteria_refs=refs, evidence_refs=["unit:u"])
        report = loop.run(self.goal.id, lambda prompt: self.fail("no cycle should run"))
        self.assertEqual(report.stop_reason, "gate_ready")
        self.assertEqual(self.work_rows(report), [])

    def test_units_back_only_what_they_observed(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=2, no_progress_limit=4))
        unit = FakeUnit("u-9", goal_id=self.goal.id, criterion_ref=self.goal.criteria[0].id, satisfied=True)
        self.units.add(FakeUnit("u-10", goal_id=self.goal.id, criterion_ref=self.goal.criteria[1].id))
        self.units.add(unit)
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("reconciled", output_tokens=10))
        self.assertEqual(self.units.reconciled, [self.goal.id, self.goal.id])
        self.assertEqual(self.work_rows(report)[0].units_verified, 1)
        goal = self.goals.get(self.goal.id)
        self.assertEqual(goal.criteria[0].status, "satisfied")
        self.assertEqual(goal.criteria[0].evidence_refs, [f"unit:{unit.id}"])
        self.assertEqual(goal.criteria[1].status, "pending")

    def test_budget_cycles(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=2, no_progress_limit=5))
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("looked", output_tokens=5))
        self.assertEqual(report.stop_reason, "budget_cycles")
        self.assertEqual(len(self.work_rows(report)), 2)
        self.assertEqual(report.next_action, "show_status")

    def test_budget_tokens(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=6, max_total_tokens=1_000, no_progress_limit=6))
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("big", input_tokens=600, output_tokens=600))
        self.assertEqual(report.stop_reason, "budget_tokens")
        self.assertEqual(len(self.work_rows(report)), 1)

    def test_budget_seconds(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=8, no_progress_limit=8), clock=Clock(200.0))
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("slow"))
        self.assertEqual(report.stop_reason, "budget_seconds")
        self.assertIn("3m 20s", report.summary)

    def test_no_progress_on_repeated_brief_digest(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=8, no_progress_limit=2))
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("nothing changed"))
        digests = [cycle.brief_digest for cycle in self.work_rows(report)]
        self.assertEqual(report.stop_reason, "no_progress")
        self.assertEqual(report.next_action, "satisfy_criterion")
        self.assertIn("unchanged", report.summary)
        self.assertEqual(digests[-1], digests[-2])
        self.assertEqual(len(digests), 3)  # two observed repeats end the run
        self.assertLess(len(digests), 8, "the ladder did not fire before the cycle budget")

    def test_a_changing_brief_resets_the_no_progress_ladder(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=4, no_progress_limit=2))
        index = [0]

        def turn(prompt: str) -> TurnOutcome:
            index[0] += 1
            self.goals.block(self.goal.id, f"friction {index[0]}")
            return TurnOutcome("changed the ledger")

        # Each cycle moves the ledger (a new blocker), so the brief never repeats:
        # the run ends on its cycle budget rather than on no-progress.
        loop = GoalLoop(
            self.goals,
            gates=self.gates,
            blocks=self.blocks,
            units=self.units,
            journal=self.journal,
            budget=LoopBudget(max_cycles=4, no_progress_limit=2),
        )
        first = loop.run(self.goal.id, turn)
        self.assertEqual(first.stop_reason, "blocked")
        self.assertEqual(len(self.work_rows(first)), 1)

    def test_gate_open_mid_run_stops_the_loop(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=4, no_progress_limit=4))
        report = loop.run(self.goal.id, self.gate_in_turn())
        self.assertEqual(report.stop_reason, "gate_open")
        self.assertEqual(report.next_action, "answer_gate")
        self.assertEqual(len(self.work_rows(report)), 1)

    def test_blocker_mid_run_stops_the_loop(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=4, no_progress_limit=4))
        report = loop.run(self.goal.id, self.block_during_turn())
        self.assertEqual(report.stop_reason, "blocked")
        self.assertEqual(report.next_action, "resolve_blocker")

    def test_cancel_event_stops_the_loop(self) -> None:
        loop = self.loop()
        cancel = threading.Event()
        cancel.set()
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("never runs"), cancel=cancel)
        self.assertEqual(report.stop_reason, "cancelled")
        self.assertEqual(self.work_rows(report), [])

    def test_stop_request_stops_at_the_next_boundary(self) -> None:
        rows: list[LoopCycle] = []

        def on_cycle(cycle: LoopCycle) -> None:
            rows.append(cycle)
            if len([row for row in rows if row.phase != "stop"]) == 2:
                loop.stop(self.goal.id, "gate_open")

        loop = GoalLoop(
            self.goals,
            gates=self.gates,
            blocks=self.blocks,
            units=self.units,
            journal=self.journal,
            budget=LoopBudget(max_cycles=5, no_progress_limit=9),
            on_cycle=on_cycle,
        )
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("working"))
        self.assertEqual(report.stop_reason, "gate_open")
        self.assertEqual(len(self.work_rows(report)), 2)

    def test_provider_error_when_the_turn_raises(self) -> None:
        loop = self.loop()

        def turn(prompt: str) -> TurnOutcome:
            raise RuntimeError("all lane members failed")

        report = loop.run(self.goal.id, turn)
        self.assertEqual(report.stop_reason, "provider_error")
        self.assertFalse(report.ready)
        failing = [cycle for cycle in report.cycles if cycle.phase == "act"]
        self.assertEqual(len(failing), 1)
        self.assertFalse(failing[0].ok)
        self.assertIn("RuntimeError", failing[0].note)

    def test_repeated_failed_turns_stop_the_loop(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=6, no_progress_limit=2))
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("lane died", ok=False))
        self.assertEqual(report.stop_reason, "provider_error")
        self.assertEqual(len(self.work_rows(report)), 2)

    def test_invalid_goal_when_the_ledger_has_nothing_to_end_on(self) -> None:
        loop = GoalLoop(StubLedger(StubGoal()))
        report = loop.run("goal-stub", lambda prompt: TurnOutcome("x"))
        self.assertEqual(report.stop_reason, "invalid_goal")
        self.assertEqual(self.work_rows(report), [])
        self.assertEqual(report.cycles[-1].phase, "stop")
        self.assertFalse(report.cycles[-1].ok)

    def test_unknown_goal_id_is_reported_not_raised(self) -> None:
        report = self.loop().run("goal-nope", lambda prompt: TurnOutcome("x"))
        self.assertEqual(report.stop_reason, "invalid_goal")

    def test_run_refuses_a_second_loop_on_the_same_goal(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=2, no_progress_limit=9))
        inner: list[LoopReport] = []

        def turn(prompt: str) -> TurnOutcome:
            if not inner:
                inner.append(loop.run(self.goal.id, lambda text: TurnOutcome("nested")))
            return TurnOutcome("outer")

        outer = loop.run(self.goal.id, turn)
        self.assertEqual(len(inner), 1)
        self.assertEqual(inner[0].stop_reason, "blocked")
        self.assertIn("already running", inner[0].summary)
        self.assertEqual(outer.stop_reason, "budget_cycles")

    def test_the_concurrency_cap_is_the_subagent_cap(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=2, no_progress_limit=9))
        loop._active.update({f"other-{index}" for index in range(MAX_SUBAGENTS)})
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("never"))
        self.assertEqual(report.stop_reason, "blocked")
        self.assertIn(str(MAX_SUBAGENTS), report.summary)
        self.assertEqual(self.work_rows(report), [])

    def test_bad_arguments_are_programmer_errors(self) -> None:
        loop = self.loop()
        with self.assertRaises(TypeError):
            loop.run(self.goal.id, "not-callable")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            loop.run(self.goal.id, lambda prompt: TurnOutcome("x"), budget={"max_cycles": 2})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            GoalLoop(self.goals, clock=42)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            GoalLoop(None)  # type: ignore[arg-type]

    def test_budget_argument_overrides_the_instance_budget(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=8, no_progress_limit=9))
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("x"), budget=LoopBudget(max_cycles=1))
        self.assertEqual(report.stop_reason, "budget_cycles")
        self.assertEqual(len(self.work_rows(report)), 1)

    def test_turn_results_are_duck_typed(self) -> None:
        class Result:
            text = "plain object turn"
            input_tokens = "7"
            output_tokens = 3.0
            tool_failures = True

        report = self.loop(budget=LoopBudget(max_cycles=1, no_progress_limit=4)).run(
            self.goal.id, lambda prompt: Result()
        )
        self.assertEqual(report.cycles[0].tokens, 10)
        self.assertIn("plain object turn", report.cycles[0].note)
        self.assertIn("1 tool failure", report.cycles[0].note)

        bare = self.loop(budget=LoopBudget(max_cycles=1, no_progress_limit=4)).run(
            self.goal.id, lambda prompt: "a bare string turn"
        )
        self.assertIn("bare string", bare.cycles[0].note)
        self.assertTrue(bare.cycles[0].ok)

        mapping = self.loop(budget=LoopBudget(max_cycles=1, no_progress_limit=4)).run(
            self.goal.id, lambda prompt: {"text": "mapped", "ok": False, "input_tokens": 4}
        )
        self.assertFalse(mapping.cycles[0].ok)
        self.assertEqual(mapping.cycles[0].tokens, 4)

    def test_on_cycle_exception_cannot_break_the_run(self) -> None:
        loop = GoalLoop(
            self.goals,
            journal=self.journal,
            budget=LoopBudget(max_cycles=2, no_progress_limit=4),
            on_cycle=lambda cycle: 1 / 0,
        )
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("ok"))
        self.assertEqual(report.stop_reason, "budget_cycles")

    def test_control_characters_from_a_turn_never_reach_the_ledger(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=1, no_progress_limit=4))
        report = loop.run(
            self.goal.id, lambda prompt: TurnOutcome("\x1b[2J rm -f everything\x00 and hide it")
        )
        goal = self.goals.get(self.goal.id)
        self.assertNotIn("\x1b", goal.checkpoints[-1].summary)
        self.assertNotIn("\x1b", report.render())


class JournalTests(LedgerCase):
    def test_every_cycle_and_the_stop_reason_are_journaled(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=2, no_progress_limit=5))
        loop.run(self.goal.id, lambda prompt: TurnOutcome("recorded", output_tokens=40))
        events = self.journal.recent(event="loop_cycle", subject=self.goal.id, limit=50)
        self.assertEqual([event.detail.get("kind") for event in events], ["cycle", "cycle", "stop"])
        self.assertEqual(events[-1].detail.get("reason"), "budget_cycles")
        self.assertEqual(events[0].detail.get("phase"), "verify")
        self.assertTrue(all(event.subject == self.goal.id for event in events))
        self.assertTrue(all(event.session_id == "s1" for event in events))

    def test_the_brief_and_the_turn_text_are_not_journaled_verbatim(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=1, no_progress_limit=4))
        loop.run(self.goal.id, lambda prompt: TurnOutcome("a sentence with a private token in it"))
        for event in self.journal.recent(limit=50):
            if event.event != "loop_cycle":
                continue
            self.assertNotIn(OBJECTIVE, json.dumps(event.to_dict()))
            self.assertTrue(len(str(event.detail.get("note", ""))) <= 240)

    def test_a_stop_that_refused_work_writes_a_block_record(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=4, no_progress_limit=4))
        loop.run(self.goal.id, self.gate_in_turn())
        records = [
            record
            for record in self.blocks.recent(reason_domain="gate_open", limit=10)
            if record.goal_id == self.goal.id
        ]
        self.assertEqual([record.reason_code for record in records], ["gate_open"])
        self.assertEqual(records[0].source, "goal_loop")
        self.assertEqual(records[0].action, "loop_run")

    def test_a_spent_budget_writes_a_block_record(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=1, max_total_tokens=1_000, no_progress_limit=4))
        loop.run(self.goal.id, lambda prompt: TurnOutcome("x", input_tokens=900, output_tokens=900))
        domains = {record.reason_domain for record in self.blocks.recent(limit=20)}
        self.assertEqual(domains, {"budget_exhausted"})

    def test_no_progress_is_not_recorded_as_a_refusal(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=8, no_progress_limit=1))
        loop.run(self.goal.id, lambda prompt: TurnOutcome("nothing written"))
        self.assertEqual(self.blocks.recent(limit=20), [])


class StatusAndResumeTests(LedgerCase):
    def test_status_is_json_safe_and_describes_the_gate(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=2, no_progress_limit=5))
        loop.run(self.goal.id, lambda prompt: TurnOutcome("worked", output_tokens=30))
        status = loop.status(self.goal.id)
        self.assertIn(self.goal.id, json.dumps(status))
        self.assertEqual(status["cycles"], 2)
        self.assertEqual(status["phase"], "stop")
        self.assertEqual(status["last_stop_reason"], "budget_cycles")
        self.assertEqual(status["criteria_total"], 2)
        self.assertEqual(status["criteria_satisfied"], 0)
        self.assertEqual(status["missing_criteria"], 2)
        self.assertEqual(status["tokens"], 60)
        self.assertFalse(status["ready"])
        self.assertFalse(status["running"])
        self.assertTrue(status["last_brief_digest"])
        self.assertEqual(status["next_action"], "show_status")
        self.assertTrue(status["can_start"])

    def test_status_reports_an_open_gate_and_a_refusal(self) -> None:
        loop = self.loop()
        gate = self.open_gate(self.goal.id)
        status = loop.status(self.goal.id)
        self.assertEqual(status["open_gates"], [gate.id])
        self.assertFalse(status["can_start"])
        self.assertIn("decision gate", status["start_refusal"])
        self.assertEqual(status["active_blockers"], 0)

    def test_a_restart_continues_instead_of_re_deriving(self) -> None:
        first = self.loop(budget=LoopBudget(max_cycles=2, no_progress_limit=5))
        first.run(self.goal.id, lambda prompt: TurnOutcome("worked", input_tokens=100))

        second = self.loop(budget=LoopBudget(max_cycles=6, no_progress_limit=2))
        status = second.status(self.goal.id)
        self.assertEqual(status["cycles"], 2)
        self.assertEqual(status["last_stop_reason"], "budget_cycles")
        self.assertEqual(status["tokens"], 200)
        self.assertTrue(status["resumed"])
        report = second.run(self.goal.id, lambda prompt: TurnOutcome("worked again"))
        self.assertEqual(report.stop_reason, "no_progress")
        self.assertLess(len(self.work_rows(report)), 3, "a restart re-derived work the ledger already recorded")
        self.assertEqual(second.status(self.goal.id)["cycles"], 3)

    def test_stop_normalizes_an_unknown_reason(self) -> None:
        loop = self.loop()
        loop.stop(self.goal.id, "operator_walked_away")
        self.assertEqual(loop.status(self.goal.id)["stop_requested"], "cancelled")
        report = loop.run(self.goal.id, lambda prompt: TurnOutcome("never"))
        self.assertEqual(report.stop_reason, "cancelled")
        self.assertIn("outside STOP_REASONS", report.summary)

    def test_stop_clears_after_the_run_so_the_goal_can_restart(self) -> None:
        loop = self.loop(budget=LoopBudget(max_cycles=1, no_progress_limit=4))
        loop.stop(self.goal.id)
        self.assertEqual(loop.run(self.goal.id, lambda prompt: TurnOutcome("x")).stop_reason, "cancelled")
        self.assertEqual(loop.status(self.goal.id)["stop_requested"], "")
        self.assertEqual(loop.run(self.goal.id, lambda prompt: TurnOutcome("y")).stop_reason, "budget_cycles")

    def test_record_cycle_accepts_an_external_row(self) -> None:
        loop = self.loop()
        loop.record_cycle(
            self.goal.id,
            LoopCycle(index=1, phase="verify", brief_digest="d", tokens=25, seconds=0.5, note="imported"),
        )
        status = loop.status(self.goal.id)
        self.assertEqual(status["cycles"], 1)
        self.assertEqual(status["tokens"], 25)
        self.assertEqual(status["phase"], "verify")
        events = self.journal.recent(event="loop_cycle", subject=self.goal.id, limit=5)
        self.assertEqual(events[0].detail.get("kind"), "cycle")

    def test_record_cycle_of_an_unknown_goal_does_not_raise(self) -> None:
        loop = self.loop()
        loop.record_cycle("goal-nope", LoopCycle(index=1, phase="verify", tokens=5))
        self.assertEqual(loop.status("goal-nope")["cycles"], 1)


class AgentTurnTests(unittest.TestCase):
    def test_missing_run_is_refused(self) -> None:
        with self.assertRaises(TypeError):
            agent_turn(object())

    def test_turn_result_objects_are_adapted(self) -> None:
        class TurnResult:
            text = "the answer"
            input_tokens = 40
            output_tokens = 12

        class Agent:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def run(self, prompt: str) -> Any:
                self.prompts.append(prompt)
                return TurnResult()

        agent = Agent()
        turn = agent_turn(agent)
        self.assertIsInstance(turn, TurnFn)
        outcome = turn("do the thing")
        self.assertEqual(
            (outcome.text, outcome.ok, outcome.input_tokens, outcome.output_tokens),
            ("the answer", True, 40, 12),
        )
        self.assertEqual(agent.prompts, ["do the thing"])

    def test_strings_mappings_and_empty_prompts(self) -> None:
        class Echo:
            def run(self, prompt: str) -> Any:
                return {
                    "text": prompt.upper(),
                    "ok": False,
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "tool_failures": 1,
                }

        turn = agent_turn(Echo())
        outcome = turn("hi")
        self.assertEqual((outcome.text, outcome.ok, outcome.tool_failures), ("HI", False, 1))
        self.assertFalse(turn("   ").ok)
        self.assertEqual(turn("   ").text, "(empty brief)")

    def test_agent_errors_reach_the_loop_as_provider_error(self) -> None:
        class Exploding:
            def run(self, prompt: str) -> Any:
                raise RuntimeError("Cancelled.")

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        ws = base / "ws"
        ws.mkdir()
        goals = GoalLedger(base / "workflows", ws)
        goal = goals.create("improve the repo docs until the tests pass", ["docs verified"])
        loop = GoalLoop(goals, budget=LoopBudget(max_cycles=2, no_progress_limit=4))
        report = loop.run(goal.id, agent_turn(Exploding()))
        self.assertEqual(report.stop_reason, "provider_error")


class LoopToolTests(LedgerCase):
    def bundle(self, *, turn_factory: Any = None, budget: LoopBudget | None = None) -> dict[str, Any]:
        loop = self.loop(budget=budget or LoopBudget(max_cycles=2, no_progress_limit=4))
        factory = turn_factory or (lambda: (lambda prompt: TurnOutcome("did the work", output_tokens=20)))
        tools = {tool.name: tool for tool in make_loop_tool(loop, self.goals, turn_factory=factory)}
        return {"loop": loop, "tools": tools}

    def test_three_tools_with_the_documented_read_only_split(self) -> None:
        tools = self.bundle()["tools"]
        self.assertEqual(set(tools), {"loop_start", "loop_status", "loop_stop"})
        self.assertFalse(tools["loop_start"].read_only)
        self.assertTrue(tools["loop_status"].read_only)
        self.assertFalse(tools["loop_stop"].read_only)
        for tool in tools.values():
            self.assertEqual(tool.parameters["additionalProperties"], False)
            self.assertIn("goal_id", tool.parameters["required"])
            self.assertTrue(tool.description)

    def test_turn_factory_must_be_callable(self) -> None:
        with self.assertRaises(TypeError):
            make_loop_tool(self.loop(), self.goals, turn_factory="nope")  # type: ignore[arg-type]

    def test_loop_start_runs_and_renders(self) -> None:
        bundle = self.bundle()
        result = bundle["tools"]["loop_start"].handler({"goal_id": self.goal.id}, self.ctx())
        self.assertTrue(result.ok, result.content)
        self.assertEqual(result.meta["stop_reason"], "budget_cycles")
        self.assertEqual(result.meta["cycles"], 2)
        self.assertFalse(result.meta["ready"])
        self.assertIn("goal loop", result.content)
        self.assertEqual(bundle["loop"].status(self.goal.id)["cycles"], 2)

    def test_loop_start_is_approval_gated(self) -> None:
        asked: list[ApprovalRequest] = []

        def confirm(request: ApprovalRequest) -> bool:
            asked.append(request)
            return False

        bundle = self.bundle()
        result = bundle["tools"]["loop_start"].handler(
            {"goal_id": self.goal.id}, self.ctx(approval_mode="ask", confirm=confirm)
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.meta.get("denied"))
        self.assertEqual([request.action for request in asked], ["loop_start"])
        self.assertIn("cycles", asked[0].summary.lower())
        self.assertIn("criterion", asked[0].detail.lower())
        self.assertEqual(bundle["loop"].status(self.goal.id)["cycles"], 0)

    def test_loop_start_is_refused_in_read_only_mode(self) -> None:
        bundle = self.bundle()
        result = bundle["tools"]["loop_start"].handler({"goal_id": self.goal.id}, self.ctx(read_only=True))
        self.assertFalse(result.ok)
        self.assertTrue(result.meta.get("denied"))

    def test_loop_status_works_in_read_only_mode(self) -> None:
        bundle = self.bundle()
        result = bundle["tools"]["loop_status"].handler({"goal_id": self.goal.id}, self.ctx(read_only=True))
        self.assertTrue(result.ok)
        self.assertIn("cycles: 0", result.content)
        self.assertEqual(result.meta["goal_id"], self.goal.id)

    def test_loop_stop_needs_no_approval_even_in_read_only_mode(self) -> None:
        bundle = self.bundle()
        result = bundle["tools"]["loop_stop"].handler(
            {"goal_id": self.goal.id, "reason": "cancelled"}, self.ctx(read_only=True)
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.meta["reason"], "cancelled")
        self.assertIn("stop requested", result.content)

    def test_a_direct_task_is_refunded_to_one_turn(self) -> None:
        narrow = self.goals.create("rename the entrypoint variable", ["no stale names"])
        bundle = self.bundle()
        result = bundle["tools"]["loop_start"].handler({"goal_id": narrow.id}, self.ctx())
        self.assertFalse(result.ok)
        self.assertEqual(result.meta["state"], "direct_task")
        self.assertIn("one turn", result.content)
        forced = bundle["tools"]["loop_start"].handler(
            {"goal_id": narrow.id, "allow_unloopable": True}, self.ctx()
        )
        self.assertTrue(forced.ok, forced.content)

    def test_loop_start_refuses_an_open_gate_before_spending_a_cycle(self) -> None:
        gate = self.open_gate(self.goal.id)
        bundle = self.bundle()
        result = bundle["tools"]["loop_start"].handler({"goal_id": self.goal.id}, self.ctx())
        self.assertFalse(result.ok)
        self.assertIn(gate.id, result.content)
        self.assertEqual(bundle["loop"].status(self.goal.id)["cycles"], 0)

    def test_loop_start_refuses_a_goal_with_a_blocker(self) -> None:
        self.goals.block(self.goal.id, "the frozen check is gone")
        bundle = self.bundle()
        result = bundle["tools"]["loop_start"].handler({"goal_id": self.goal.id}, self.ctx())
        self.assertFalse(result.ok)
        self.assertIn("blocker", result.content)

    def test_max_cycles_argument_bounds_the_run(self) -> None:
        bundle = self.bundle()
        result = bundle["tools"]["loop_start"].handler(
            {"goal_id": self.goal.id, "max_cycles": 1}, self.ctx()
        )
        self.assertEqual(result.meta["cycles"], 1)
        garbage = bundle["tools"]["loop_start"].handler(
            {"goal_id": self.goal.id, "max_cycles": "lots"}, self.ctx()
        )
        self.assertTrue(garbage.ok, garbage.content)

    def test_loop_start_reports_a_stop_that_needs_the_operator(self) -> None:
        bundle = self.bundle(turn_factory=lambda: self.block_during_turn())
        result = bundle["tools"]["loop_start"].handler({"goal_id": self.goal.id}, self.ctx())
        self.assertFalse(result.ok)
        self.assertEqual(result.meta["stop_reason"], "blocked")
        self.assertEqual(result.meta["next_action"], "resolve_blocker")
        self.assertIn("goal loop", result.content)

    def test_argument_validation(self) -> None:
        bundle = self.bundle()
        start = bundle["tools"]["loop_start"]
        for args in (
            {},
            {"goal_id": "  "},
            {"goal_id": 7},
            {"goal_id": self.goal.id, "surprise": True},
        ):
            with self.subTest(args=args):
                self.assertFalse(start.handler(args, self.ctx()).ok)
        status = bundle["tools"]["loop_status"]
        self.assertFalse(status.handler({"goal_id": ""}, self.ctx()).ok)
        self.assertFalse(status.handler({"goal_id": self.goal.id, "extra": 1}, self.ctx()).ok)
        stop = bundle["tools"]["loop_stop"]
        self.assertFalse(stop.handler({}, self.ctx()).ok)
        self.assertFalse(stop.handler({"reason": "cancelled"}, self.ctx()).ok)

    def test_unknown_goal_is_reported_not_raised(self) -> None:
        bundle = self.bundle()
        result = bundle["tools"]["loop_start"].handler({"goal_id": "goal-nope"}, self.ctx())
        self.assertFalse(result.ok)
        self.assertIn("not a goal", result.content)

    def test_a_broken_turn_factory_fails_closed(self) -> None:
        def broken_factory() -> Any:
            raise RuntimeError("no lane configured")

        result = self.bundle(turn_factory=broken_factory)["tools"]["loop_start"].handler(
            {"goal_id": self.goal.id}, self.ctx()
        )
        self.assertFalse(result.ok)
        self.assertIn("turn_factory", result.content)

        def silent_factory() -> Any:
            return None

        result = self.bundle(turn_factory=silent_factory)["tools"]["loop_start"].handler(
            {"goal_id": self.goal.id}, self.ctx()
        )
        self.assertFalse(result.ok)
        self.assertIn("callable", result.content)


class ReportRenderTests(unittest.TestCase):
    @staticmethod
    def cycles() -> list[LoopCycle]:
        return [
            LoopCycle(
                index=1,
                phase="verify",
                brief_digest="abc123",
                at="2026-01-01T00:00:00+00:00",
                seconds=1.5,
                tokens=1500,
                ok=True,
                checkpoint_id="cp-1",
                criteria_satisfied=1,
                units_verified=1,
                note="walked the install path",
            ),
            LoopCycle(
                index=2,
                phase="stop",
                brief_digest="abc123",
                seconds=2.0,
                ok=False,
                note="no_progress: brief unchanged",
            ),
        ]

    def report(self) -> LoopReport:
        return LoopReport(
            goal_id="goal-1",
            cycles=self.cycles(),
            stop_reason="no_progress",
            ready=False,
            summary="1 cycle(s) this run - no_progress",
            next_action="satisfy_criterion",
        )

    def test_unicode_and_ascii_renderings(self) -> None:
        unicode = self.report().render()
        self.assertIn("goal loop · goal-1", unicode)
        self.assertIn("stopped: no_progress", unicode)
        self.assertIn("○", unicode)
        self.assertIn("ckpt cp-1", unicode)
        self.assertIn("1.50s", unicode)
        self.assertIn("1.5k tok", unicode)
        self.assertIn("no_progress: brief unchanged", unicode)
        ascii_text = self.report().render(glyphs="ascii")
        self.assertIn("goal loop - goal-1", ascii_text)
        self.assertNotIn("·", ascii_text)
        self.assertNotIn("✓", ascii_text)

    def test_unknown_glyph_mode_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.report().render(glyphs="braille")

    def test_empty_report_renders(self) -> None:
        text = LoopReport(goal_id="goal-2", stop_reason="invalid_goal").render()
        self.assertIn("0 cycle(s)", text)
        self.assertIn("next: show_status", text)

    def test_control_characters_cannot_escape_the_render(self) -> None:
        report = LoopReport(goal_id="goal-3", cycles=[LoopCycle(index=1, phase="stop", note="\x1b[2J pwn")])
        self.assertNotIn("\x1b", report.render())


if __name__ == "__main__":
    unittest.main()
