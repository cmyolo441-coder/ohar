from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from ohar.types import ApprovalRequest, Tool, ToolContext, ToolResult
from ohar.workflows.goals import GoalLedger
from ohar.workflows.journal import ActivityJournal, args_digest
from ohar.workflows.store import WorkflowError
from ohar.workflows.verification import (
    CHECK_KINDS,
    EVIDENCE_LADDER,
    MAX_ATTEMPTS_PER_UNIT,
    MAX_CHECK_CHARS,
    MAX_OUTPUT_TAIL_BYTES,
    MAX_UNITS,
    REPAIR_BLOCKERS,
    TERMINAL_STATUSES,
    Unit,
    UnitAttempt,
    VerificationLedger,
    check_digest,
    claims_completion,
    make_verify_tools,
    output_digest,
)


class TempBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "home"
        self.workspace = self.tmp / "repo"
        self.workspace.mkdir()
        self.journal = ActivityJournal(self.root, self.workspace)
        self.ledger = VerificationLedger(self.root, self.workspace, journal=self.journal)
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def auto(self) -> ToolContext:
        return ToolContext(workspace=self.workspace, approval_mode="auto")

    def asking(self, answer: bool = True) -> ToolContext:
        return ToolContext(
            workspace=self.workspace, approval_mode="ask", confirm=lambda _request: answer
        )

    def observed(self, name: str, args: dict, ok: bool, **meta: object) -> dict:
        result = ToolResult(ok, "output", meta)
        return self.journal.record_tool(name, args, result, seconds=0.5)


class VocabularyTests(unittest.TestCase):
    def test_ladder_and_bounds(self) -> None:
        self.assertEqual(tuple(CHECK_KINDS), ("command", "tool"))
        self.assertEqual(list(EVIDENCE_LADDER)[-1], "observed_passed")
        self.assertEqual(set(TERMINAL_STATUSES), {"observed_passed", "abandoned"})
        self.assertEqual(
            (MAX_UNITS, MAX_ATTEMPTS_PER_UNIT, MAX_CHECK_CHARS, MAX_OUTPUT_TAIL_BYTES),
            (32, 3, 512, 4096),
        )
        self.assertEqual(
            set(REPAIR_BLOCKERS),
            {"attempts_exhausted", "no_frozen_check", "read_only_policy", "approval_denied", "cancelled"},
        )

    def test_claims_completion_is_only_about_the_word(self) -> None:
        self.assertTrue(claims_completion("claimed"))
        self.assertTrue(claims_completion("observed_passed"))
        for status in ("configured", "not_observed", "observed_failed", "abandoned", "", "nonsense"):
            self.assertFalse(claims_completion(status), status)

    def test_check_digest_identity(self) -> None:
        self.assertTrue(check_digest("command", "pytest -q").startswith("chk-"))
        self.assertEqual(check_digest("command", "pytest -q"), check_digest("command", "pytest -q"))
        self.assertNotEqual(check_digest("command", "pytest -q"), check_digest("command", "pytest -x"))
        self.assertNotEqual(check_digest("command", "pytest -q"), check_digest("tool", 'pytest {"-q": true}'))
        self.assertEqual(check_digest("tool", "pytest -not-json"), "")

    def test_tool_check_digest_is_the_journal_digest(self) -> None:
        digest = check_digest("tool", 'read_file {"path": "a.txt", "line": 3}')
        self.assertEqual(digest, args_digest("read_file", {"path": "a.txt", "line": 3}))
        self.assertEqual(check_digest("tool", "remember"), args_digest("remember", {}))
        self.assertEqual(check_digest("command", ""), "")
        self.assertEqual(check_digest("tool", ""), "")

    def test_an_unparseable_tool_check_digests_to_nothing(self) -> None:
        # The digest is total so a bad shape cannot smuggle itself in; declare()
        # is where it gets refused.
        self.assertEqual(check_digest("tool", "read_file not-json"), "")


class UnitModelTests(unittest.TestCase):
    def test_round_trip_and_derived_claims(self) -> None:
        unit = Unit(
            id="unit-1", statement="suite passes", kind="command", check="pytest -q",
            digest=check_digest("command", "pytest -q"), status="observed_passed", goal_id="g1",
            criterion_ref="AC1",
            attempts=[UnitAttempt("2026-01-01T00:00:00+00:00", True, 0, "runner", 1.5, "tail-abc")],
        )
        self.assertTrue(unit.satisfied)
        restored = Unit.from_record(unit.to_dict())
        self.assertEqual(restored.status, "observed_passed")
        self.assertEqual(restored.attempts[0].observed_via, "runner")
        self.assertIn("pytest -q", restored.check)

    def test_a_planted_pass_without_an_attempt_is_repaired(self) -> None:
        planted = Unit.from_record(
            {"id": "unit-2", "statement": "x", "kind": "command", "check": "pytest -q",
             "status": "observed_passed", "attempts": [], "digest": "chk-abc"}
        )
        self.assertNotEqual(planted.status, "observed_passed")
        self.assertFalse(planted.satisfied)

    def test_an_operator_pass_is_a_claim(self) -> None:
        unit = Unit.from_record(
            {
                "id": "unit-3", "statement": "x", "kind": "command", "check": "pytest -q",
                "status": "configured", "digest": "chk-abc",
                "attempts": [{"at": "2026-01-01T00:00:00+00:00", "ok": True, "exit_code": 0,
                              "observed_via": "operator", "seconds": 1.0, "output_tail_digest": "tail-1"}],
            }
        )
        self.assertEqual(unit.status, "claimed")
        self.assertFalse(unit.satisfied)

    def test_attempt_output_tail_is_never_stored(self) -> None:
        attempt = UnitAttempt.from_record(
            {"at": "2026-01-01T00:00:00+00:00", "ok": False, "exit_code": 1, "observed_via": "journal",
             "seconds": 0.25, "output_tail_digest": "tail-abc", "output": "the whole failing suite"}
        )
        self.assertNotIn("output", attempt.to_dict())
        self.assertEqual(attempt.exit_code, 1)
        self.assertTrue(attempt.is_observed)

    def test_output_digest_bounds_the_tail(self) -> None:
        small = output_digest("a" * 10)
        self.assertTrue(small.startswith("tail-"))
        self.assertEqual(small, output_digest("a" * 10))
        self.assertNotEqual(small, output_digest("b" * 10))
        self.assertEqual(output_digest(""), "")
        # Only the tail is hashed, so two runs that end the same way share a handle.
        tail = ("FAILURE" + " trace" * 900)[: MAX_OUTPUT_TAIL_BYTES + 500]
        self.assertEqual(output_digest("a" * 9_000 + tail), output_digest("b" * 9_000 + tail))
        self.assertNotEqual(output_digest(tail), output_digest(tail + " and then a different end"))

    def test_control_characters_are_stripped_from_a_statement(self) -> None:
        unit = Unit.from_record({"id": "u", "statement": "done\x1b[31m\x00", "kind": "command",
                                 "check": "true", "status": "configured"})
        self.assertNotIn("\x1b", unit.statement)
        self.assertNotIn("\x1b", json.dumps(unit.to_dict()))


class DeclareTests(TempBase):
    def test_declare_freezes_one_check_and_reports_the_gap(self) -> None:
        unit = self.ledger.declare("the suite passes", kind="command", check="pytest -q", goal_id="g1")
        self.assertEqual(unit.status, "configured")
        self.assertEqual(unit.missing_evidence, ["observed_passed"])
        self.assertFalse(unit.satisfied)
        self.assertEqual(unit.goal_id, "g1")
        self.assertEqual(self.ledger.evidence_summary()["configured"], 1)

    def test_declare_is_idempotent_by_digest(self) -> None:
        first = self.ledger.declare("suite", kind="command", check="pytest -q")
        again = self.ledger.declare("suite reworded", kind="command", check="pytest -q")
        self.assertEqual(first.id, again.id)
        self.assertEqual(len(self.ledger.units()), 1)

    def test_declare_refuses_bad_input(self) -> None:
        with self.assertRaises(WorkflowError):
            self.ledger.declare("", kind="command", check="pytest -q")
        with self.assertRaises(WorkflowError):
            self.ledger.declare("x", kind="vibes", check="pytest -q")
        with self.assertRaises(WorkflowError):
            self.ledger.declare("x", kind="command", check="   ")
        with self.assertRaises(WorkflowError):
            self.ledger.declare("x", kind="tool", check="")
        with self.assertRaises(WorkflowError):
            self.ledger.declare("x", kind="tool", check="read_file [1,2]")

    def test_tool_checks_are_canonicalised(self) -> None:
        unit = self.ledger.declare("read it", kind="tool", check='read_file {"b": 2, "a": 1}')
        self.assertEqual(unit.check, 'read_file {"a": 1, "b": 2}')
        self.assertEqual(unit.digest, args_digest("read_file", {"a": 1, "b": 2}))

    def test_ledger_is_bounded(self) -> None:
        for index in range(MAX_UNITS):
            self.ledger.declare(f"unit {index}", kind="command", check=f"check {index}")
        self.assertEqual(len(self.ledger.units(limit=MAX_UNITS)), MAX_UNITS)
        with self.assertRaises(WorkflowError):
            self.ledger.declare("one too many", kind="command", check="check overflow")

    def test_get_and_units_filters(self) -> None:
        a = self.ledger.declare("a", kind="command", check="check a", goal_id="g1")
        b = self.ledger.declare("b", kind="tool", check="remember", goal_id="g2")
        self.assertEqual(self.ledger.get(a.id).id, a.id)
        with self.assertRaises(WorkflowError):
            self.ledger.get("unit-missing")
        self.assertEqual([u.id for u in self.ledger.units(goal_id="g2")], [b.id])
        self.assertEqual([u.id for u in self.ledger.units(status="configured")], [a.id, b.id])
        self.assertEqual(len(self.ledger.units(limit=1)), 1)
        self.assertEqual(len(self.ledger.headers()), len(self.ledger.rows([a, b])[0]))


class RunTests(TempBase):
    def test_run_observes_the_exact_frozen_line(self) -> None:
        unit = self.ledger.declare("pass", kind="command", check="true")
        ran = self.ledger.run(unit.id, self.auto())
        self.assertEqual(ran.status, "observed_passed")
        self.assertTrue(ran.satisfied)
        self.assertEqual(ran.attempts[-1].exit_code, 0)
        self.assertEqual(ran.attempts[-1].observed_via, "runner")
        self.assertEqual(ran.missing_evidence, [])

    def test_a_failure_is_observed_and_its_output_is_only_a_digest(self) -> None:
        emitter = self.workspace / "emit.sh"
        emitter.write_text("#!/bin/sh\necho secret-canary-lines >&2\nexit 3\n", encoding="utf-8")
        unit = self.ledger.declare("fail", kind="command", check="sh emit.sh")
        ran = self.ledger.run(unit.id, self.auto())
        self.assertEqual(ran.status, "observed_failed")
        self.assertFalse(ran.satisfied)
        self.assertEqual(ran.attempts[-1].exit_code, 3)
        self.assertTrue(ran.attempts[-1].output_tail_digest.startswith("tail-"))
        raw = self.ledger._store.path().read_text(encoding="utf-8")
        self.assertNotIn("secret-canary", raw)

    def test_run_happens_inside_the_workspace(self) -> None:
        unit = self.ledger.declare("mark the workspace", kind="command", check="touch made-by-check.txt")
        self.ledger.run(unit.id, self.auto())
        self.assertTrue((self.workspace / "made-by-check.txt").is_file())
        self.assertFalse((self.tmp / "made-by-check.txt").exists())

    def test_approval_gates_the_run(self) -> None:
        unit = self.ledger.declare("needs a person", kind="command", check="true")
        asked: list[ApprovalRequest] = []
        ctx = ToolContext(workspace=self.workspace, approval_mode="ask",
                          confirm=lambda request: asked.append(request) or True)
        self.assertEqual(self.ledger.run(unit.id, ctx).status, "observed_passed")
        self.assertEqual(asked[0].action, "unit_run")
        self.assertIn("true", asked[0].detail)
        denied = self.ledger.run(self.ledger.declare("declined", kind="command", check="echo x").id,
                                 self.asking(answer=False))
        self.assertIn("approval_denied", denied.missing_evidence)
        self.assertEqual(denied.attempts, [])

    def test_read_only_policy_blocks_without_burning_an_attempt(self) -> None:
        unit = self.ledger.declare("ro", kind="command", check="true")
        blocked = self.ledger.run(unit.id, ToolContext(workspace=self.workspace, read_only=True))
        self.assertIn("read_only_policy", blocked.missing_evidence)
        self.assertEqual(blocked.attempts, [])
        self.assertEqual(self.ledger.can_satisfy(unit.id), (False, "read_only_policy"))

    def test_a_zero_timeout_is_a_cancellation(self) -> None:
        unit = self.ledger.declare("no time granted", kind="command", check="true")
        cancelled = self.ledger.run(unit.id, self.auto(), timeout=0)
        self.assertIn("cancelled", cancelled.missing_evidence)
        self.assertEqual(cancelled.attempts, [])

    def test_attempts_are_bounded(self) -> None:
        unit = self.ledger.declare("keeps failing", kind="command", check="exit 1")
        for _ in range(MAX_ATTEMPTS_PER_UNIT + 2):
            ran = self.ledger.run(unit.id, self.auto())
        self.assertEqual(len(ran.attempts), MAX_ATTEMPTS_PER_UNIT)
        self.assertIn("attempts_exhausted", ran.missing_evidence)
        self.assertEqual(self.ledger.can_satisfy(ran.id), (False, "attempts_exhausted"))

    def test_a_timeout_is_an_observed_non_pass(self) -> None:
        unit = self.ledger.declare("slow", kind="command", check="sleep 5")
        ran = self.ledger.run(unit.id, self.auto(), timeout=0.2)
        self.assertEqual(ran.status, "observed_failed")
        self.assertIsNone(ran.attempts[-1].exit_code)

    def test_a_tool_check_is_observed_not_executed(self) -> None:
        unit = self.ledger.declare("tool shaped check", kind="tool", check="remember")
        refused = self.ledger.run(unit.id, self.auto())
        self.assertIn("no_frozen_check", refused.missing_evidence)

    def test_destructive_checks_are_refused_before_anything_runs(self) -> None:
        unit = self.ledger.declare("catastrophe", kind="command", check="sudo rm -rf / --no-preserve-root")
        with self.assertRaises(WorkflowError):
            self.ledger.run(unit.id, self.auto())
        self.assertEqual(self.ledger.get(unit.id).attempts, [])

    def test_run_of_a_terminal_unit_is_a_no_op(self) -> None:
        unit = self.ledger.declare("pass", kind="command", check="true")
        self.ledger.run(unit.id, self.auto())
        again = self.ledger.run(unit.id, self.auto())
        self.assertEqual(len(again.attempts), 1)


class EvidenceTests(TempBase):
    def test_record_attempt_labels_an_operator_claim_as_a_claim(self) -> None:
        unit = self.ledger.declare("model says done", kind="command", check="pytest -q")
        claimed = self.ledger.record_attempt(unit.id, ok=True, exit_code=0, seconds=1.0,
                                             observed_via="operator", output="all green")
        self.assertEqual(claimed.status, "claimed")
        self.assertFalse(claimed.satisfied)
        self.assertTrue(claims_completion(claimed.status))
        self.assertEqual(claimed.attempts[-1].output_tail_digest, output_digest("all green"))
        self.assertNotIn("all green", json.dumps(claimed.to_dict()))
        self.assertEqual(self.ledger.reconcile(unit.id).status, "claimed")

    def test_record_attempt_refuses_a_bogus_channel(self) -> None:
        unit = self.ledger.declare("x", kind="command", check="true")
        with self.assertRaises(WorkflowError):
            self.ledger.record_attempt(unit.id, ok=True, exit_code=0, seconds=1, observed_via="vibes")

    def test_ingest_matches_the_exact_tool_call_only(self) -> None:
        unit = self.ledger.declare("read the file", kind="tool", check='read_file {"path": "a.txt"}')
        other = self.ledger.ingest_tool_observation(
            self.observed("read_file", {"path": "b.txt"}, True)
        )
        self.assertEqual(other, [])
        self.assertEqual(self.ledger.get(unit.id).status, "configured")
        moved = self.ledger.ingest_tool_observation(self.observed("read_file", {"path": "a.txt"}, True))
        self.assertEqual(moved, [unit.id])
        self.assertTrue(self.ledger.get(unit.id).satisfied)

    def test_ingest_matches_a_command_observed_through_the_shell(self) -> None:
        unit = self.ledger.declare("suite", kind="command", check="pytest -q")
        self.assertEqual(self.ledger.ingest_tool_observation(self.observed("shell", {"command": "pytest -x"}, True)), [])
        moved = self.ledger.ingest_tool_observation(self.observed("shell", {"command": "pytest -q"}, True))
        self.assertEqual(moved, [unit.id])
        self.assertEqual(self.ledger.get(unit.id).status, "observed_passed")
        failed = self.ledger.declare("lint", kind="command", check="ruff .")
        self.assertEqual(
            self.ledger.ingest_tool_observation(self.observed("shell", {"command": "ruff ."}, False)),
            [failed.id],
        )
        self.assertEqual(self.ledger.get(failed.id).status, "observed_failed")

    def test_ingest_ignores_noise(self) -> None:
        self.ledger.declare("suite", kind="command", check="pytest -q")
        for event in (
            {},
            {"event": "turn_completed", "subject": "shell", "ok": True, "at": "2026-01-01T00:00:00+00:00"},
            {"event": "tool_observed", "subject": "shell", "ok": None, "at": "2026-01-01T00:00:00+00:00"},
            {"event": "tool_observed", "subject": "", "ok": True, "at": "2026-01-01T00:00:00+00:00"},
        ):
            self.assertEqual(self.ledger.ingest_tool_observation(event), [])

    def test_reconcile_follows_the_journal_for_the_frozen_digest(self) -> None:
        unit = self.ledger.declare("pass", kind="command", check="true")
        self.journal.record("unit_observed", subject=unit.id, ok=True,
                            detail={"digest": unit.digest, "via": "journal", "seconds": 0.1})
        self.assertEqual(self.ledger.reconcile(unit.id).status, "observed_passed")
        re_declared = self.ledger.declare("pass", kind="command", check="true --fast")
        self.journal.record("unit_observed", subject=re_declared.id, ok=True,
                            detail={"digest": "chk-someone-elses-check", "via": "journal"})
        self.assertEqual(self.ledger.reconcile(re_declared.id).status, "configured")

    def test_reconcile_ignores_refusals_and_future_rows(self) -> None:
        unit = self.ledger.declare("pass", kind="command", check="true")
        self.ledger.run(unit.id, self.asking(answer=False))
        self.assertEqual(self.ledger.reconcile(unit.id).status, "configured")
        future = time.time() + 3_600
        self.journal.record("unit_observed", subject=unit.id, ok=True,
                            detail={"digest": unit.digest, "via": "journal"})
        self.assertEqual(self.ledger.reconcile(unit.id, now=future).status, "observed_passed")

    def test_reconcile_all_reports_every_unit(self) -> None:
        self.ledger.declare("a", kind="command", check="true")
        self.ledger.declare("b", kind="command", check="false")
        moved = self.ledger.reconcile_all()
        self.assertEqual(len(moved), 2)
        self.assertEqual(len(self.ledger.reconcile_all(goal_id="nope")), 0)

    def test_abandon_is_terminal(self) -> None:
        unit = self.ledger.declare("give up", kind="command", check="true")
        abandoned = self.ledger.abandon(unit.id)
        self.assertEqual(abandoned.status, "abandoned")
        self.assertTrue(abandoned.terminal)
        self.assertEqual(self.ledger.can_satisfy(unit.id), (False, "attempts_exhausted"))
        self.assertEqual(self.ledger.run(unit.id, self.auto()).status, "abandoned")

    def test_missing_evidence_and_summary_are_per_goal(self) -> None:
        ok = self.ledger.declare("green", kind="command", check="true", goal_id="g1")
        bad = self.ledger.declare("red", kind="command", check="exit 1", goal_id="g1", criterion_ref="AC1")
        self.ledger.run(ok.id, self.auto())
        self.ledger.run(bad.id, self.auto())
        gaps = self.ledger.missing_evidence(goal_id="g1")
        self.assertEqual([gap["unit"] for gap in gaps], [bad.id])
        self.assertEqual(gaps[0]["missing"], "observed_passed")
        self.assertEqual(gaps[0]["next_action"], "unit_run")
        self.assertTrue(all(isinstance(value, str) for gap in gaps for value in gap.values()))
        summary = self.ledger.evidence_summary(goal_id="g1")
        self.assertEqual(summary["observed_passed"], 1)
        self.assertEqual(summary["observed_failed"], 1)
        self.assertEqual(summary["satisfied"], 1)
        self.assertEqual(self.ledger.evidence_summary()["units"], 2)

    def test_a_check_that_is_not_frozen_reports_no_frozen_check(self) -> None:
        # A record written by an older build (or hand-edited) can hold no check.
        # It reads back as unobserved and cannot be earned by running it.
        self.ledger._store.append(
            {"id": "unit-hollow", "statement": "no check", "kind": "command", "check": "",
             "digest": "", "status": "observed_passed", "goal_id": "", "criterion_ref": "",
             "attempts": [], "missing_evidence": []}
        )
        hollow = self.ledger.get("unit-hollow")
        self.assertEqual(hollow.status, "not_observed")
        self.assertFalse(hollow.satisfied)
        self.assertIn("no_frozen_check", hollow.missing_evidence)
        self.assertEqual(self.ledger.can_satisfy("unit-hollow"), (False, "no_frozen_check"))
        self.assertIn("no_frozen_check", self.ledger.run("unit-hollow", self.auto()).missing_evidence)

    def test_rows_render_the_ladder(self) -> None:
        unit = self.ledger.declare("row", kind="tool", check="remember", goal_id="g9")
        row = self.ledger.rows([unit])[0]
        self.assertEqual(row[2], "tool")
        self.assertEqual(row[6], "g9")
        self.assertIn("observed_passed", row[5])


class VerifyToolTests(TempBase):
    def setUp(self) -> None:
        super().setUp()
        self.tools: list[Tool] = make_verify_tools(self.ledger)
        self.by_name = {tool.name: tool for tool in self.tools}

    def test_tool_surface(self) -> None:
        self.assertEqual(sorted(self.by_name), ["unit_declare", "unit_run", "unit_status"])
        self.assertFalse(self.by_name["unit_declare"].read_only)
        self.assertFalse(self.by_name["unit_run"].read_only)
        self.assertTrue(self.by_name["unit_run"].destructive)
        self.assertTrue(self.by_name["unit_status"].read_only)
        self.assertIn("command", self.by_name["unit_declare"].parameters["properties"]["kind"]["enum"])

    def test_declare_run_status_through_the_tools(self) -> None:
        ctx = self.auto()
        declared = self.by_name["unit_declare"].handler(
            {"statement": "suite green", "kind": "command", "check": "true"}, ctx
        )
        self.assertTrue(declared.ok)
        unit_id = declared.meta["unit_id"]
        ran = self.by_name["unit_run"].handler({"unit": unit_id, "timeout": 30}, ctx)
        self.assertTrue(ran.ok)
        self.assertTrue(ran.meta["satisfied"])
        listing = self.by_name["unit_status"].handler({}, ctx)
        self.assertIn("suite green", listing.content)
        detail = self.by_name["unit_status"].handler({"unit": unit_id}, ctx)
        self.assertIn("can_satisfy: True", detail.content)

    def test_declare_and_run_refusals_are_reported_not_raised(self) -> None:
        denied = self.by_name["unit_declare"].handler(
            {"statement": "x", "kind": "command", "check": "true"}, self.asking(answer=False)
        )
        self.assertFalse(denied.ok)
        self.assertTrue(denied.meta.get("denied") is True or "denied" in denied.content)
        bad_args = self.by_name["unit_declare"].handler({"statement": 1}, self.auto())
        self.assertFalse(bad_args.ok)
        self.assertFalse(self.by_name["unit_run"].handler({}, self.auto()).ok)
        self.assertFalse(self.by_name["unit_run"].handler({"unit": "unit-ghost"}, self.auto()).ok)
        self.assertFalse(self.by_name["unit_status"].handler({"unit": "unit-ghost"}, self.auto()).ok)
        ro = ToolContext(workspace=self.workspace, read_only=True)
        self.assertFalse(self.by_name["unit_declare"].handler(
            {"statement": "x", "kind": "command", "check": "true"}, ro).ok)

    def test_a_failing_check_is_reported_as_a_failure(self) -> None:
        declared = self.by_name["unit_declare"].handler(
            {"statement": "red", "kind": "command", "check": "exit 2"}, self.auto()
        )
        ran = self.by_name["unit_run"].handler({"unit": declared.meta["unit_id"]}, self.auto())
        self.assertFalse(ran.ok)
        self.assertEqual(ran.meta["status"], "observed_failed")
        self.assertIn("still missing", ran.content)

    def test_an_observed_pass_cites_evidence_back_onto_the_goal(self) -> None:
        goals = GoalLedger(self.root, self.workspace, journal=self.journal)
        goal = goals.create("ship it", ["the suite is green"])
        criterion = goal.criteria[0]
        ledger = VerificationLedger(self.root, self.workspace, journal=self.journal)
        tools = {tool.name: tool for tool in make_verify_tools(ledger, goals=goals)}
        declared = tools["unit_declare"].handler(
            {"statement": "suite green", "kind": "command", "check": "true",
             "goal": goal.id, "criterion": criterion.id}, self.auto()
        )
        self.assertTrue(declared.ok)
        ran = tools["unit_run"].handler({"unit": declared.meta["unit_id"]}, self.auto())
        self.assertTrue(ran.ok)
        updated = goals.get(goal.id)
        self.assertEqual(updated.criteria[0].status, "satisfied")
        self.assertTrue(any("unit:" in ref for ref in updated.criteria[0].evidence_refs))
        gate = goals.gate(goal.id, units=[ledger.get(declared.meta["unit_id"]).to_dict()])
        self.assertTrue(gate.ready)

    def test_a_claim_never_cites_evidence_back(self) -> None:
        goals = GoalLedger(self.root, self.workspace, journal=self.journal)
        goal = goals.create("ship it", ["the suite is green"])
        ledger = VerificationLedger(self.root, self.workspace, journal=self.journal)
        unit = ledger.declare("suite green", kind="command", check="true",
                              goal_id=goal.id, criterion_ref=goal.criteria[0].id)
        tools = {tool.name: tool for tool in make_verify_tools(ledger, goals=goals)}
        self.assertFalse(tools["unit_run"].handler({"unit": unit.id}, self.asking(answer=False)).ok)
        ledger.record_attempt(unit.id, ok=True, exit_code=0, seconds=1.0, observed_via="operator")
        self.assertEqual(goals.get(goal.id).criteria[0].status, "pending")

    def test_require_approval_off_still_needs_a_writable_context(self) -> None:
        tools = {tool.name: tool for tool in make_verify_tools(self.ledger, require_approval=False)}
        declared = tools["unit_declare"].handler(
            {"statement": "no prompt", "kind": "command", "check": "true"},
            ToolContext(workspace=self.workspace, approval_mode="ask", confirm=lambda _r: False),
        )
        self.assertTrue(declared.ok)
        ran = tools["unit_run"].handler({"unit": declared.meta["unit_id"]},
                                        ToolContext(workspace=self.workspace, approval_mode="ask",
                                                    confirm=lambda _r: False))
        # unit_run is gated by the ledger itself, not by require_approval.
        self.assertIn("approval_denied", ran.content)


if __name__ == "__main__":
    unittest.main()
