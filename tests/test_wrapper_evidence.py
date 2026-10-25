from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ohar.types import ToolContext, ToolResult
from ohar.wrapper import evidence as ev
from ohar.wrapper import records, state

SESSION = "ev-1"


def item(kind: str = "command", subject: str = "pytest -q", result: str = "observed", exit_code: int | None = 0) -> ev.EvidenceItem:
    return ev.EvidenceItem(
        kind=kind,
        subject=subject,
        digest=records.digest_of(kind, subject, result, exit_code),
        result=result,
        exit_code=exit_code,
        at="",
    )


class EvidenceMixin(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("OHAR_HOME")
        os.environ["OHAR_HOME"] = str(Path(self.tmp.name) / "home")
        state.open_state(SESSION, goal="ship it", workspace="/work")

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("OHAR_HOME", None)
        else:
            os.environ["OHAR_HOME"] = self.previous_home
        self.tmp.cleanup()

    def claim_states(self, plan: dict | None = None) -> dict[str, str]:
        return {row.claim: row.state for row in ev.claims(SESSION, plan=plan)}

    def events(self) -> list[str]:
        rows, errors = records.read_events(SESSION, limit=records.MAX_EVENTS)
        self.assertEqual(errors, [])
        return [str(row["event"]) for row in rows]


class EvidenceItemTests(EvidenceMixin):
    def test_round_trip_and_defaults(self) -> None:
        row = item()
        payload = row.to_dict()
        self.assertEqual(
            set(payload),
            {"kind", "subject", "digest", "result", "exit_code", "at", "detail"},
        )
        self.assertTrue(payload["at"])
        self.assertEqual(ev.EvidenceItem.from_dict(payload), row)
        self.assertEqual(row.detail, "")

    def test_fields_are_cleaned_and_bounded(self) -> None:
        row = ev.EvidenceItem(
            kind="command",
            subject="\x1b[31m" + "x" * 500 + "\nsecond",
            digest="d" * 64,
            result="observed",
            exit_code=0,
            at="",
            detail="y" * 5000,
        )
        self.assertEqual(len(row.subject), ev.MAX_SUBJECT_CHARS)
        self.assertNotIn("\x1b", row.subject)
        self.assertNotIn("\n", row.subject)
        self.assertEqual(len(row.detail), ev.MAX_DETAIL_CHARS)
        self.assertEqual(len(row.digest), 64)

    def test_invalid_items_are_refused(self) -> None:
        base = {"digest": "d" * 64, "result": "observed", "exit_code": 0, "at": "", "subject": "s"}
        for label, override in (
            ("kind", {"kind": "guess"}),
            ("result", {"result": "maybe"}),
            ("subject", {"subject": "   "}),
            ("digest", {"digest": ""}),
            ("exit_code", {"exit_code": True}),
            ("exit_code_str", {"exit_code": "0"}),
        ):
            with self.subTest(label):
                with self.assertRaises(ev.EvidenceError):
                    ev.EvidenceItem(**{**base, "kind": "command", **override})  # type: ignore[arg-type]

    def test_from_dict_is_fail_closed(self) -> None:
        payload = item().to_dict()
        for label, broken in (
            ("not_a_mapping", None),
            ("missing_subject", {**payload, "subject": 3}),
            ("bad_exit", {**payload, "exit_code": "0"}),
            ("bad_detail", {**payload, "detail": 5}),
            ("bad_kind", {**payload, "kind": "magic"}),
            ("bad_at", {**payload, "at": "yesterday"}),
            ("naive_at", {**payload, "at": "2026-01-01T00:00:00"}),
            ("missing_at", {k: v for k, v in payload.items() if k != "at"}),
        ):
            with self.subTest(label):
                with self.assertRaises(ev.EvidenceError):
                    ev.EvidenceItem.from_dict(broken)  # type: ignore[arg-type]

    def test_claim_validation(self) -> None:
        self.assertEqual(ev.Claim("tests_pass", "pending").to_dict()["refs"], [])
        row = ev.Claim("tests_pass", "observed", "ran pytest", tuple("d" * 12 for _ in range(20)))
        self.assertEqual(len(row.refs), ev.MAX_REFS)
        with self.assertRaises(ev.EvidenceError):
            ev.Claim("nope", "observed")
        with self.assertRaises(ev.EvidenceError):
            ev.Claim("tests_pass", "probably")


class RecordTests(EvidenceMixin):
    def test_record_appends_item_and_event(self) -> None:
        row = ev.record(SESSION, item())
        self.assertEqual(row.subject, "pytest -q")
        stored = ev.items(SESSION)
        self.assertEqual([row.subject for row in stored], ["pytest -q"])
        self.assertEqual(self.events(), ["session_started", "evidence_observed"])
        record = records.read_record(ev.EVIDENCE_RECORD, SESSION)
        self.assertEqual(record["record_type"], ev.RECORD_TYPE)
        self.assertEqual(record["session_id"], SESSION)
        self.assertEqual(ev.validate_evidence_record(record), None)

    def test_items_are_bounded(self) -> None:
        with mock.patch.object(ev, "MAX_ITEMS", 5):
            for index in range(9):
                ev.record(SESSION, item(subject=f"cmd-{index}"))
            stored = ev.items(SESSION, limit=ev.MAX_ITEMS)
        self.assertEqual([row.subject for row in stored], [f"cmd-{index}" for index in range(4, 9)])

    def test_replay_does_not_duplicate(self) -> None:
        first = ev.record(SESSION, item(), mutation_id="retry-1")
        again = ev.record(SESSION, first, mutation_id="retry-1")
        self.assertEqual(len(ev.items(SESSION)), 1)
        self.assertEqual(self.events().count("evidence_observed"), 1)
        self.assertEqual(again.subject, first.subject)
        # Same mutation id, different content: refused, not silently replayed.
        with self.assertRaises(ev.EvidenceError):
            ev.record(SESSION, item(subject="make deploy"), mutation_id="retry-1")
        self.assertEqual(len(ev.items(SESSION)), 1)

    def test_record_refuses_garbage(self) -> None:
        with self.assertRaises(ev.EvidenceError):
            ev.record(SESSION, {"kind": "command"})  # type: ignore[arg-type]
        with self.assertRaises(ev.EvidenceError):
            ev.items("bad/../id")

    def test_malformed_stored_items_are_skipped(self) -> None:
        path = records.session_dir(SESSION) / ev.EVIDENCE_RECORD
        ev.record(SESSION, item(subject="keep me"))
        record = records.read_record(ev.EVIDENCE_RECORD, SESSION)
        record["items"].append({"kind": "nonsense", "subject": "x", "digest": "d", "result": "observed", "at": "t"})
        record["items"].append("not an object")
        records.write_record(ev.EVIDENCE_RECORD, SESSION, record)
        self.assertEqual([row.subject for row in ev.items(SESSION)], ["keep me"])
        with self.assertRaises(ev.EvidenceError):
            ev.validate_evidence_record(record)


class ObserveToolEndTests(EvidenceMixin):
    def test_shell_success_and_failure_become_command_evidence(self) -> None:
        ok = ev.observe_tool_end(
            "ev-1", "shell", ToolResult.success("8 passed in 0.2s\n[exit code 0; 0.201s]", exit_code=0, duration_s=0.2), 0.201
        )
        self.assertIsNotNone(ok)
        assert ok is not None
        self.assertEqual((ok.kind, ok.result, ok.exit_code), ("command", "observed", 0))
        self.assertEqual(ok.subject, "8 passed in 0.2s")
        self.assertIn("exit_code=0", ok.detail)
        failed = ev.observe_tool_end("ev-1", "shell", ToolResult.failure("[exit code 1]", exit_code=1), 0.5)
        assert failed is not None
        self.assertEqual((failed.result, failed.exit_code), ("failed", 1))
        self.assertEqual(len(ev.items(SESSION)), 2)

    def test_denied_calls_become_approval_evidence(self) -> None:
        denied = ev.observe_tool_end(
            SESSION, "edit_file", ToolResult.failure("Read-only mode: edit_file is disabled.", denied=True), 0.0
        )
        assert denied is not None
        self.assertEqual((denied.kind, denied.result), ("approval", "skipped"))
        self.assertEqual(self.claim_states()["approvals_resolved"], "pending")

    def test_file_task_and_fetch_tools(self) -> None:
        wrote = ev.observe_tool_end(
            SESSION, "write_file", ToolResult.success("Created a.txt (12 bytes).", path="a.txt", bytes=12), 0.01
        )
        edited = ev.observe_tool_end(
            SESSION, "edit_file", ToolResult.success("Edited b.txt (1 replacement).", path="b.txt", replacements=1), 0.01
        )
        task = ev.observe_tool_end(
            SESSION, "task", ToolResult.success("2/2 done", count=2, completed=2, failed=0, seconds=1.2), 1.2
        )
        fetch = ev.observe_tool_end(
            SESSION, "web_fetch", ToolResult.failure("blocked", status=403, final_url="https://x.example/a"), 0.3
        )
        self.assertEqual(wrote.kind, "file_write")  # type: ignore[union-attr]
        self.assertEqual(wrote.subject, "a.txt")  # type: ignore[union-attr]
        self.assertEqual(edited.subject, "b.txt")  # type: ignore[union-attr]
        self.assertEqual(task.subject, "task 2/2 subagents")  # type: ignore[union-attr]
        self.assertEqual(fetch.subject, "https://x.example/a")  # type: ignore[union-attr]
        self.assertEqual(fetch.result, "failed")  # type: ignore[union-attr]
        self.assertEqual(self.claim_states()["files_changed"], "observed")

    def test_uninteresting_and_invalid_ends(self) -> None:
        for name in ("read_file", "list_files", "grep", "glob", "web_search", "", "shell-echo"):
            with self.subTest(name=name):
                self.assertIsNone(ev.observe_tool_end(SESSION, name, ToolResult.success("x"), 0.1))
        self.assertEqual(ev.items(SESSION), [])
        with self.assertRaises(ev.EvidenceError):
            ev.observe_tool_end(SESSION, "shell", "not a result", 0.1)  # type: ignore[arg-type]
        weird = ev.observe_tool_end(SESSION, "shell", ToolResult.success("out", exit_code="0"), "not-a-float")  # type: ignore[arg-type]
        assert weird is not None
        self.assertIsNone(weird.exit_code)
        self.assertIn("elapsed=0.00s", weird.detail)

    def test_command_meta_is_preferred_as_the_subject(self) -> None:
        row = ev.observe_tool_end(
            SESSION, "shell", ToolResult.success("all good", exit_code=0, command="pytest -q tests"), 0.4
        )
        assert row is not None
        self.assertEqual(row.subject, "pytest -q tests")
        self.assertEqual(self.claim_states()["tests_pass"], "observed")


class ClaimsTests(EvidenceMixin):
    def test_everything_starts_pending(self) -> None:
        states = self.claim_states()
        self.assertEqual(set(states), set(ev.CLAIMS))
        self.assertEqual(set(states.values()), {"pending"})
        self.assertEqual(ev.claims(SESSION), ev.claims(SESSION))

    def test_failed_command_refutes_and_later_success_observes(self) -> None:
        ev.record(SESSION, item(subject="pytest -q", result="failed", exit_code=1))
        tests = [row for row in ev.claims(SESSION) if row.claim == "tests_pass"][0]
        self.assertEqual(tests.state, "refuted")
        self.assertIn("failed", tests.reason)
        self.assertEqual(tests.refs[0], ev.records.digest_of("command", "pytest -q", "failed", 1)[:12])
        ev.record(SESSION, item(subject="pytest -q", result="observed", exit_code=0))
        self.assertEqual(self.claim_states()["tests_pass"], "observed")

    def test_nonzero_exit_never_claims_success(self) -> None:
        ev.record(SESSION, item(subject="pytest -q", exit_code=2))
        self.assertEqual(self.claim_states()["tests_pass"], "refuted")

    def test_build_and_test_claims_are_narrowed_by_the_plan(self) -> None:
        ev.record(SESSION, item(subject="pytest -q"))
        plan = {"test_commands": ["mypy"], "build_commands": ["python -m build"]}
        states = self.claim_states(plan=plan)
        self.assertEqual(states["tests_pass"], "pending")
        ev.record(SESSION, item(subject="mypy ohar"))
        self.assertEqual(self.claim_states(plan=plan)["tests_pass"], "observed")
        self.assertEqual(self.claim_states()["build_ok"], "pending")

    def test_command_claims_ignore_other_kinds(self) -> None:
        ev.record(SESSION, item(kind="file_write", subject="ohar/wrapper/state.py"))
        states = self.claim_states()
        self.assertEqual(states["files_changed"], "observed")
        self.assertEqual(states["commands_ran"], "pending")
        self.assertEqual(states["tests_pass"], "pending")

    def test_work_verified_follows_granted_verification(self) -> None:
        self.assertEqual(self.claim_states()["work_verified"], "pending")
        ev.record(SESSION, item(subject="pytest -q"))
        self.assertEqual(self.claim_states()["work_verified"], "pending")
        granted = ev.grant_verification(SESSION, {"tests": True})
        self.assertEqual(granted["verification"]["work_verified"], "observed")
        self.assertEqual(self.claim_states()["work_verified"], "observed")
        again = ev.grant_verification(SESSION, {"tests": False})
        self.assertEqual(again["verification"]["work_verified"], "refuted")
        row = [row for row in ev.claims(SESSION) if row.claim == "work_verified"][0]
        self.assertEqual(row.state, "refuted")
        self.assertIn("tests", row.reason)

    def test_overclaim_guards_name_every_unmet_claim(self) -> None:
        rows = ev.claims(SESSION)
        guards = ev.overclaim_guards(rows)
        self.assertGreater(len(guards), 2)
        self.assertTrue(all(len(line) <= ev.MAX_DETAIL_CHARS for line in guards))
        for row in rows:
            if row.state != "observed":
                self.assertTrue(any(row.claim in line for line in guards))
        ev.record(SESSION, item(subject="pytest -q"))
        ev.grant_verification(SESSION, {"tests": True})
        shorter = ev.overclaim_guards(ev.claims(SESSION))
        self.assertLess(len(shorter), len(guards))


class PayloadTests(EvidenceMixin):
    def test_payload_shape_and_boundary(self) -> None:
        payload = ev.claims_payload(SESSION)
        self.assertEqual(payload["schema_version"], ev.CLAIMS_SCHEMA_VERSION)
        self.assertEqual(payload["session_id"], SESSION)
        self.assertEqual(payload["status"], "open")
        self.assertEqual(payload["allowed_claims"], [])
        self.assertEqual(len(payload["blocked_claims"]), len(ev.CLAIMS))
        self.assertEqual(payload["next_action"], "run_an_observed_command")
        self.assertIn("not evidence", payload["claim_boundary"])
        self.assertEqual(payload["items_recorded"], 0)
        self.assertEqual(len(payload["claims"]), len(ev.CLAIMS))

    def test_payload_tracks_the_session_state_and_progress(self) -> None:
        ev.record(SESSION, item(subject="pytest -q"))
        ev.record(SESSION, item(kind="file_write", subject="ohar/wrapper/evidence.py", exit_code=None))
        ev.record(SESSION, item(kind="command", subject="python -m build", exit_code=0))
        payload = ev.claims_payload(SESSION)
        self.assertEqual(payload["items_recorded"], 3)
        self.assertIn("commands_ran", payload["allowed_claims"])
        self.assertIn("tests_pass", payload["allowed_claims"])
        self.assertIn("build_ok", payload["allowed_claims"])
        self.assertEqual(payload["next_action"], "grant_verification")
        state.transition(SESSION, "plan_presented")
        self.assertEqual(ev.claims_payload(SESSION)["status"], "plan_presented")
        state.record_decision(SESSION, "cancel")
        closed = ev.claims_payload(SESSION)
        self.assertEqual(closed["status"], "cancelled")
        self.assertEqual(closed["next_action"], "session_closed")

    def test_require_claim(self) -> None:
        with self.assertRaises(ev.EvidenceError):
            ev.require_claim(SESSION, "files_changed")
        ev.record(SESSION, item(kind="file_write", subject="a.py", exit_code=None))
        ev.require_claim(SESSION, "files_changed")
        with self.assertRaises(ev.EvidenceError):
            ev.require_claim(SESSION, "not_a_claim")
        with self.assertRaises(ev.EvidenceError):
            ev.require_claim(SESSION, "work_verified")
        ev.record(SESSION, item(subject="pytest -q"))
        ev.grant_verification(SESSION, {"tests": True})
        ev.require_claim(SESSION, "work_verified")

    def test_require_claim_refuses_failed_evidence(self) -> None:
        ev.record(SESSION, item(subject="pytest -q", result="failed", exit_code=1))
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.require_claim(SESSION, "tests_pass")
        self.assertIn("Refusing to claim tests_pass", str(caught.exception))


class GrantVerificationTests(EvidenceMixin):
    def test_refuses_before_any_command_evidence(self) -> None:
        ev.record(SESSION, item(kind="file_write", subject="a.py", exit_code=None))
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.grant_verification(SESSION, {"tests": True})
        self.assertIn("before command evidence", str(caught.exception))

    def test_refuses_while_the_latest_command_failed(self) -> None:
        ev.record(SESSION, item(subject="pytest -q"))
        ev.record(SESSION, item(subject="pytest -q", result="failed", exit_code=1))
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.grant_verification(SESSION, {"tests": True})
        self.assertIn("latest command evidence is failed", str(caught.exception))

    def test_refuses_malformed_postconditions(self) -> None:
        ev.record(SESSION, item(subject="pytest -q"))
        for bad in ({}, {"tests": "yes"}, {"": True}, {None: True}, "tests"):
            with self.subTest(bad=bad):
                with self.assertRaises(ev.EvidenceError):
                    ev.grant_verification(SESSION, bad)  # type: ignore[arg-type]

    def test_refuses_a_terminal_session(self) -> None:
        ev.record(SESSION, item(subject="pytest -q"))
        state.transition(SESSION, "plan_presented")
        state.record_decision(SESSION, "accept")
        state.transition(SESSION, "working")
        state.transition(SESSION, "verified")
        state.close_state(SESSION)
        with self.assertRaises(ev.EvidenceError):
            ev.grant_verification(SESSION, {"tests": True})

    def test_grant_records_refs_boundary_and_event(self) -> None:
        first = ev.record(SESSION, item(subject="pytest -q"))
        payload = ev.grant_verification(SESSION, {"tests": True, "build": True})
        verification = payload["verification"]
        self.assertEqual(verification["postconditions"], {"build": True, "tests": True})
        self.assertEqual(verification["failed_postconditions"], [])
        self.assertEqual(verification["command_items"], 1)
        self.assertEqual(verification["refs"], [first.digest[:12]])  # type: ignore[index]
        self.assertIn("not evidence", payload["claim_boundary"])
        self.assertFalse(payload["replayed"])
        self.assertIn("verification_granted", self.events())
        self.assertEqual(payload["verification"]["claim_boundary"], ev.CLAIM_BOUNDARY)

    def test_grant_is_replay_safe_and_revision_gated(self) -> None:
        ev.record(SESSION, item(subject="pytest -q"))
        first = ev.grant_verification(SESSION, {"tests": True}, mutation_id="verify-1")
        again = ev.grant_verification(SESSION, {"tests": True}, mutation_id="verify-1")
        self.assertTrue(again["replayed"])
        self.assertEqual(self.events().count("verification_granted"), 1)
        self.assertEqual(first["verification"]["granted_at"], again["verification"]["granted_at"])
        with self.assertRaises(ev.EvidenceError):
            ev.grant_verification(SESSION, {"tests": False}, expected_revision=99)

    def test_verification_survives_later_items(self) -> None:
        ev.record(SESSION, item(subject="pytest -q"))
        ev.grant_verification(SESSION, {"tests": True})
        ev.record(SESSION, item(subject="git status", exit_code=0))
        self.assertEqual(self.claim_states()["work_verified"], "observed")
        self.assertEqual(len(ev.items(SESSION)), 2)


class ToolTests(EvidenceMixin):
    def test_tool_surface_is_read_only_and_bounded(self) -> None:
        tool = ev.make_evidence_tool(SESSION)
        self.assertEqual(tool.name, "claim_status")
        self.assertTrue(tool.read_only)
        self.assertFalse(tool.destructive)
        self.assertEqual(tool.parameters["properties"], {})
        spec = tool.api_spec()
        self.assertEqual(spec["function"]["name"], "claim_status")

        ev.record(SESSION, item(subject="pytest -q"))
        ev.grant_verification(SESSION, {"tests": True})
        result = tool.handler({}, ToolContext(workspace=Path(self.tmp.name)))
        self.assertTrue(result.ok)
        lines = result.content.splitlines()
        self.assertLessEqual(len(lines), ev.MAX_PAYLOAD_LINES + 4)
        self.assertIn("claims for ev-1", lines[0])
        self.assertTrue(all(len(line) <= 220 for line in lines))
        self.assertEqual(result.meta["allowed_claims"], ["commands_ran", "tests_pass", "work_verified"])
        self.assertEqual(result.meta["next_action"], "report_completion")
        self.assertEqual(result.meta["session_id"], SESSION)

    def test_tool_reports_blocked_claims(self) -> None:
        tool = ev.make_evidence_tool(SESSION)
        result = tool.handler({}, ToolContext(workspace=Path(self.tmp.name)))
        self.assertIn("blocked: commands_ran, tests_pass", result.content)
        self.assertIn("boundary:", result.content)

    def test_tool_factory_validates_the_session_id(self) -> None:
        for bad in ("../escape", "", "a/b"):
            with self.subTest(bad=bad):
                with self.assertRaises(ev.EvidenceError):
                    ev.make_evidence_tool(bad)

    def test_tool_failure_channel(self) -> None:
        tool = ev.make_evidence_tool(SESSION)
        with mock.patch.object(ev, "claims_payload", side_effect=ev.EvidenceError("store is unreadable")):
            result = tool.handler({}, ToolContext(workspace=Path(self.tmp.name)))
        self.assertFalse(result.ok)
        self.assertIn("store is unreadable", result.content)


class VocabTests(unittest.TestCase):
    def test_closed_vocabularies(self) -> None:
        self.assertEqual(
            ev.KINDS,
            ("command", "file_write", "approval", "subagent", "fetch", "skill", "manual"),
        )
        self.assertEqual(ev.RESULTS, ("observed", "failed", "skipped"))
        self.assertEqual(len(ev.CLAIMS), 6)
        self.assertEqual(ev.CLAIM_STATES, ("observed", "pending", "refuted"))
        self.assertEqual(ev.EVIDENCE_RECORD, "evidence.json")
        self.assertTrue(set(ev._TOOL_KINDS.values()) <= set(ev.KINDS))


if __name__ == "__main__":
    unittest.main()
