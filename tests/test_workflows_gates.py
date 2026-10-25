from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ohar.terminal import make_theme
from ohar.types import ApprovalRequest, ToolContext
from ohar.workflows.gates import (
    ACTION_CLASSES,
    CHOICE_RELEASES,
    GATE_CHOICES,
    GATE_STATES,
    GATE_TTL_SECONDS,
    QUESTION_CODES,
    RECEIPT_DECISIONS,
    REFUSAL_CODES,
    RISK_CLASSES,
    ApprovalLog,
    ApprovalReceipt,
    ApprovalVerdict,
    DecisionGate,
    DecisionGates,
    GateError,
    args_digest,
    classify_action,
    make_confirm_wrapper,
    make_gate_tool,
)
from ohar.workflows.journal import ActivityJournal
from ohar.workflows.store import WorkflowError

BASE = 1_800_000_000.0


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="microseconds")


class DecisionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "workflows"
        self.ws = self.base / "ws"
        self.ws.mkdir()
        self.now = [BASE]
        self.clock = lambda: self.now[0]
        self.journal = ActivityJournal(self.root, self.ws, clock=self.clock)
        self.gates = DecisionGates(self.root, self.ws, clock=self.clock, journal=self.journal)
        self.log = ApprovalLog(self.root, self.ws, clock=self.clock, journal=self.journal)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def open_one(self, **overrides: Any) -> DecisionGate:
        args: dict[str, Any] = {
            "subject": "notes.txt",
            "question_code": "allow_write",
            "action": "write_file",
            "arg_digest": args_digest("write_file", {"path": "notes.txt"}),
            "session_id": "s1",
        }
        args.update(overrides)
        return self.gates.open(**args)

    def request(self, **overrides: Any) -> ApprovalRequest:
        values: dict[str, Any] = {
            "action": "write_file",
            "summary": "Write notes.txt",
            "detail": "notes.txt",
            "risk": "low",
        }
        values.update(overrides)
        return ApprovalRequest(**values)

    def ctx(self, **overrides: Any) -> ToolContext:
        values: dict[str, Any] = {"workspace": self.ws, "approval_mode": "auto"}
        values.update(overrides)
        return ToolContext(**values)

    def history(self) -> list[dict[str, Any]]:
        return self.gates._store.records()  # noqa: SLF001 - asserting append-only shape

    # -- vocabularies --------------------------------------------------------

    def test_closed_vocabularies(self) -> None:
        self.assertEqual(GATE_CHOICES, ("approve", "decline", "defer"))
        self.assertEqual(CHOICE_RELEASES, {"approve": True, "decline": False, "defer": False})
        self.assertEqual([name for name, released in CHOICE_RELEASES.items() if released], ["approve"])
        self.assertEqual(GATE_STATES, ("open", "answered"))
        self.assertEqual(
            QUESTION_CODES,
            (
                "allow_write",
                "allow_shell",
                "allow_network",
                "allow_browser_effect",
                "accept_risky_change",
                "resume_blocked_goal",
                "discard_changes",
            ),
        )
        self.assertEqual(RISK_CLASSES, ("reversible", "costly_to_reverse", "irreversible"))
        self.assertEqual(GATE_TTL_SECONDS, 3600)
        self.assertEqual(RECEIPT_DECISIONS, ("granted", "denied", "revoked"))
        self.assertEqual(ACTION_CLASSES, ("read_only", "mutating", "sensitive", "disabled", "blocked"))
        self.assertEqual(len(set(REFUSAL_CODES)), len(REFUSAL_CODES))
        self.assertTrue(issubclass(GateError, WorkflowError))

    def test_classify_action_covers_every_class(self) -> None:
        seen = {
            classify_action("read_file", read_only=True, destructive=False, approval_mode="ask"),
            classify_action("write_file", read_only=False, destructive=False, approval_mode="ask"),
            classify_action("shell", read_only=False, destructive=False, approval_mode="ask"),
            classify_action("write_file", read_only=False, destructive=False, approval_mode="read-only"),
            classify_action("write_file", read_only=False, destructive=False, approval_mode="off"),
        }
        self.assertEqual(seen, set(ACTION_CLASSES))
        self.assertEqual(
            classify_action("write_file", read_only=False, destructive=True, approval_mode="ask"), "sensitive"
        )
        self.assertEqual(
            classify_action("browser_click", read_only=False, destructive=False, approval_mode="ask"), "sensitive"
        )
        self.assertEqual(
            classify_action("mcp_call", read_only=False, destructive=False, approval_mode="ask"), "sensitive"
        )
        self.assertEqual(
            classify_action("edit_file", read_only=True, destructive=True, approval_mode="off"), "read_only"
        )

    def test_args_digest_is_stable_and_order_independent(self) -> None:
        first = args_digest("write_file", {"path": "a.txt", "text": "x"})
        second = args_digest("write_file", {"text": "x", "path": "a.txt"})
        self.assertEqual(first, second)
        self.assertNotEqual(first, args_digest("write_file", {"path": "b.txt", "text": "x"}))
        self.assertNotEqual(first, args_digest("edit_file", {"path": "a.txt", "text": "x"}))

    # -- opening -------------------------------------------------------------

    def test_open_records_the_question(self) -> None:
        gate = self.open_one(risk="costly_to_reverse", detail="create one file")
        self.assertEqual(gate.state, "open")
        self.assertEqual(gate.subject, "notes.txt")
        self.assertEqual(gate.question_code, "allow_write")
        self.assertEqual(gate.choices, list(GATE_CHOICES))
        self.assertEqual(gate.risk, "costly_to_reverse")
        self.assertIn("Allow this change to files", gate.question)
        self.assertIn("create one file", gate.question)
        self.assertTrue(gate.resume_digest)
        self.assertEqual(gate.answered_choice, "")
        self.assertEqual(gate.supersedes, "")
        self.assertTrue(gate.id.startswith("gate-"))
        self.assertEqual(self.gates.head(gate.gate_id).gate_id, gate.gate_id)
        self.assertEqual(self.journal.counts()["gate_opened"], 1)
        self.assertEqual(gate.to_dict()["gate_id"], gate.gate_id)

    def test_re_asking_an_open_question_returns_the_record_on_file(self) -> None:
        first = self.open_one()
        again = self.open_one()
        self.assertEqual(first.id, again.id)
        self.assertEqual(len(self.history()), 1)
        self.assertEqual(self.journal.counts()["gate_opened"], 1)

    def test_a_subject_carries_at_most_one_open_question(self) -> None:
        self.open_one()
        with self.assertRaises(GateError) as caught:
            self.open_one(question_code="allow_shell", action="shell", arg_digest="d2")
        self.assertIn("already has an open question", str(caught.exception))
        self.assertEqual(len(self.gates.open_gates()), 1)
        other = self.open_one(subject="other.txt", question_code="allow_shell", action="shell", arg_digest="d2")
        self.assertEqual(other.state, "open")
        self.assertEqual(len(self.gates.open_gates()), 2)

    def test_open_refusals(self) -> None:
        bad: list[dict[str, Any]] = [
            {"subject": ""},
            {"subject": "   "},
            {"action": ""},
            {"question_code": "accept_anything"},
            {"risk": "catastrophic"},
            {"choices": ()},
            {"choices": ["maybe"]},
        ]
        for kwargs in bad:
            with self.subTest(**kwargs):
                with self.assertRaises(GateError):
                    self.open_one(**kwargs)
        self.assertEqual(self.history(), [])

    # -- answering -----------------------------------------------------------

    def test_answer_supersedes_rather_than_edits(self) -> None:
        gate = self.open_one()
        answered = self.gates.answer(gate.gate_id, choice="approve")
        self.assertEqual(answered.state, "answered")
        self.assertEqual(answered.answered_choice, "approve")
        self.assertEqual(answered.actor, "operator")
        self.assertTrue(answered.releases)
        self.assertNotEqual(answered.id, gate.id)
        self.assertEqual(answered.supersedes, gate.id)
        self.assertEqual(answered.decided_at, iso(BASE))
        self.assertEqual(answered.resume_digest, gate.resume_digest)
        self.assertEqual(self.gates.head(gate.gate_id).id, answered.id)
        self.assertEqual([item["state"] for item in self.history()], ["open", "answered"])
        self.assertEqual(self.journal.counts()["gate_answered"], 1)

    def test_decline_and_defer_release_nothing(self) -> None:
        for choice in ("decline", "defer"):
            gate = self.open_one(subject=f"subject-{choice}", arg_digest=f"digest-{choice}")
            answered = self.gates.answer(gate.gate_id, choice=choice)
            with self.subTest(choice=choice):
                self.assertFalse(answered.releases)
                self.assertEqual(answered.state, "answered")

    def test_answer_refusals(self) -> None:
        gate = self.open_one(choices=["approve", "decline"])
        with self.assertRaises(GateError):
            self.gates.answer(gate.gate_id, choice="defer")  # not offered by this gate
        with self.assertRaises(GateError):
            self.gates.answer(gate.gate_id, choice="yes")
        with self.assertRaises(GateError):
            self.gates.answer("gate-nope", choice="approve")
        with self.assertRaises(GateError):
            self.gates.answer("", choice="approve")
        self.gates.answer(gate.gate_id, choice="approve")
        with self.assertRaises(GateError):
            self.gates.answer(gate.gate_id, choice="decline")
        self.assertEqual(len(self.history()), 2)

    def test_answering_an_expired_gate_is_refused(self) -> None:
        gate = self.open_one()
        self.now[0] += GATE_TTL_SECONDS + 1
        with self.assertRaises(GateError) as caught:
            self.gates.answer(gate.gate_id, choice="approve")
        self.assertIn("expired", str(caught.exception))

    # -- freshness and reads -------------------------------------------------

    def test_expiry_is_derived_at_read_time(self) -> None:
        gate = self.open_one()
        self.assertFalse(self.gates.is_expired(gate))
        self.assertEqual(self.gates.pending_subjects(), ["notes.txt"])
        self.now[0] += GATE_TTL_SECONDS + 1
        self.assertTrue(self.gates.is_expired(gate))
        self.assertTrue(self.gates.is_expired(gate, now=self.now[0]))
        self.assertEqual(self.gates.pending_subjects(), [])
        self.assertEqual(len(self.gates.open_gates()), 1)  # on file, just no longer live
        # an expired question may be re-asked rather than wedging the subject
        fresh = self.open_one()
        self.assertNotEqual(fresh.id, gate.id)
        self.assertFalse(self.gates.is_expired(fresh))
        self.assertEqual(self.gates.pending_subjects(), ["notes.txt"])

    def test_ttl_is_injective_per_instance_and_bounded(self) -> None:
        short = DecisionGates(self.root, self.ws, ttl_seconds=0, clock=self.clock)
        self.assertEqual(short.ttl_seconds, 1.0)
        odd = DecisionGates(self.root, self.ws, ttl_seconds="many", clock=self.clock)  # type: ignore[arg-type]
        self.assertEqual(odd.ttl_seconds, float(GATE_TTL_SECONDS))
        gate = short.open(subject="quick.txt", question_code="allow_write", action="write_file", arg_digest="d")
        self.assertTrue(short.is_expired(gate, now=self.now[0] + 2))

    def test_unparseable_stamp_is_treated_as_expired(self) -> None:
        gate = self.open_one()
        fields = {key: value for key, value in gate.to_dict().items() if key not in {"gate_id", "schema"}}
        broken = DecisionGate(**{**fields, "opened_at": "yesterday"})
        self.assertTrue(self.gates.is_expired(broken))

    def test_open_gates_and_pending_subjects_filters(self) -> None:
        first = self.open_one()
        second = self.open_one(subject="other.txt", question_code="allow_shell", action="shell", arg_digest="d2")
        self.now[0] += 5
        third = self.open_one(
            subject="third.txt",
            session_id="s2",
            question_code="discard_changes",
            action="wipe",
            arg_digest="d3",
        )
        self.assertEqual([item.id for item in self.gates.open_gates()], [first.id, second.id, third.id])
        self.assertEqual(len(self.gates.open_gates(limit=2)), 2)
        self.assertEqual([item.subject for item in self.gates.open_gates(session_id="s2")], ["third.txt"])
        self.assertEqual(self.gates.open_gates(session_id="none"), [])
        self.assertEqual(self.gates.pending_subjects(), ["notes.txt", "other.txt", "third.txt"])
        self.assertEqual(self.gates.pending_subjects(session_id="s1"), ["notes.txt", "other.txt"])
        self.gates.answer(third.gate_id, choice="decline")
        self.assertNotIn("third.txt", self.gates.pending_subjects())
        self.assertIsNone(self.gates.head("gate-nope"))

    def test_render_rows_and_headers(self) -> None:
        gate = self.open_one(risk="irreversible")
        plain = self.gates.render(gate)
        self.assertIn(gate.gate_id, plain)
        self.assertIn("irreversible", plain)
        self.assertIn("cannot be undone", plain)
        self.assertIn("choices approve/decline/defer", plain)
        self.assertIn("one recorded question", plain)
        self.assertNotIn("\x1b", plain)
        card = self.gates.render(gate, theme=make_theme("aurora", color=False), glyphs="ascii")
        self.assertIn("approval", card)
        self.assertIn(gate.action, card)
        self.assertNotIn("\x1b", card)
        self.assertIn("\x1b[", self.gates.render(gate, theme=make_theme("aurora", color=True)))
        self.assertEqual(self.gates.headers(), ["gate", "subject", "question", "action", "state", "opened"])
        rows = self.gates.rows([gate])
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), len(self.gates.headers()))
        self.assertEqual(rows[0][0], gate.gate_id)
        self.assertEqual(rows[0][4], "open")
        self.assertEqual(self.gates.rows([]), [])
        self.now[0] += GATE_TTL_SECONDS + 1
        self.assertEqual(self.gates.rows([gate])[0][4], "expired")

    # -- receipts ------------------------------------------------------------

    def test_record_grants_and_denies(self) -> None:
        request = self.request()
        digest = args_digest("write_file", {"path": "notes.txt"})
        granted = self.log.record(request, approved=True, session_id="s1", arg_digest=digest, scope="notes.txt")
        self.assertEqual(granted.decision, "granted")
        self.assertEqual(granted.mode, "ask")
        self.assertEqual(granted.actor, "operator")
        self.assertEqual(granted.action, "write_file")
        self.assertTrue(granted.id.startswith("apr-"))
        self.assertEqual(granted.at, iso(BASE))
        self.assertEqual(granted.supersedes, "")
        denied = self.log.record(
            request, approved=False, session_id="s1", arg_digest=digest, scope="notes.txt", mode="auto"
        )
        self.assertEqual(denied.decision, "denied")
        self.assertEqual(denied.chain, granted.chain)  # one confirmation, two answers to it
        self.assertNotEqual(denied.mode, granted.mode)  # and the mode still binds
        self.assertEqual(len(self.log.receipts()), 2)
        self.assertEqual(self.journal.counts()["approval_answered"], 2)
        repeat = self.log.record(request, approved=False, session_id="s1", arg_digest=digest, scope="notes.txt", mode="auto")
        self.assertEqual(repeat.supersedes, denied.id)

    def test_record_refusals(self) -> None:
        with self.assertRaises(GateError):
            self.log.record("not a request", approved=True)  # type: ignore[arg-type]
        with self.assertRaises(GateError):
            self.log.record(self.request(action=""), approved=True)
        with self.assertRaises(GateError):
            self.log.record(self.request(risk="extreme"), approved=True)
        with self.assertRaises(GateError):
            self.log.record(self.request(), approved=True, mode="sometimes")

    def test_verdict_binds_five_dimensions_by_equality(self) -> None:
        digest = args_digest("write_file", {"path": "notes.txt"})
        receipt = self.log.record(self.request(), approved=True, session_id="s1", arg_digest=digest, scope="notes.txt")
        base = {"action": "write_file", "arg_digest": digest, "scope": "notes.txt", "session_id": "s1", "mode": "ask"}
        ok = self.log.verdict(**base)
        self.assertTrue(ok.satisfied)
        self.assertEqual(ok.reason_code, "")
        self.assertEqual(ok.age_seconds, 0)
        self.assertEqual(ok.expires_in_seconds, GATE_TTL_SECONDS)
        self.assertEqual(ok.receipt_id, receipt.id)
        widening = [
            ("action_not_approved", {**base, "action": "edit_file"}),
            ("subject_not_approved", {**base, "scope": "sub/notes.txt"}),
            ("session_not_approved", {**base, "session_id": "s2"}),
            ("mode_not_approved", {**base, "mode": "auto"}),
            ("arg_digest_not_approved", {**base, "arg_digest": args_digest("write_file", {"path": "other.txt"})}),
        ]
        for code, kwargs in widening:
            with self.subTest(code=code):
                verdict = self.log.verdict(**kwargs)
                self.assertFalse(verdict.satisfied)
                self.assertEqual(verdict.reason_code, code)
                self.assertEqual(verdict.expires_in_seconds, 0)
        self.assertIn("subject_not_approved", [code for code, _ in widening])

    def test_verdict_without_any_receipt_is_absent(self) -> None:
        verdict = self.log.verdict(action="write_file", arg_digest="d", scope="s", session_id="x", mode="ask")
        self.assertFalse(verdict.satisfied)
        self.assertEqual(verdict.reason_code, "approval_absent")
        self.assertEqual(verdict.receipt_id, "")
        self.assertEqual(verdict.age_seconds, -1)
        self.assertIn("No approval", verdict.reason)

    def test_verdict_expiry(self) -> None:
        digest = args_digest("write_file", {"path": "notes.txt"})
        self.log.record(self.request(), approved=True, session_id="s1", arg_digest=digest, scope="notes.txt")
        self.now[0] += GATE_TTL_SECONDS - 1
        self.assertTrue(
            self.log.verdict(
                action="write_file", arg_digest=digest, scope="notes.txt", session_id="s1", mode="ask"
            ).satisfied
        )
        self.now[0] += 2
        expired = self.log.verdict(
            action="write_file", arg_digest=digest, scope="notes.txt", session_id="s1", mode="ask"
        )
        self.assertEqual(expired.reason_code, "approval_expired")
        self.assertFalse(expired.satisfied)

    def test_verdict_superseded_by_a_later_answer(self) -> None:
        digest = args_digest("write_file", {"path": "notes.txt"})
        self.log.record(self.request(), approved=True, session_id="s1", arg_digest=digest, scope="notes.txt")
        later = self.log.record(self.request(), approved=True, session_id="s1", arg_digest="other", scope="notes.txt")
        first = self.log.verdict(
            action="write_file", arg_digest=digest, scope="notes.txt", session_id="s1", mode="ask"
        )
        self.assertEqual(first.reason_code, "approval_superseded")
        self.assertFalse(first.satisfied)
        current = self.log.verdict(
            action="write_file", arg_digest="other", scope="notes.txt", session_id="s1", mode="ask"
        )
        self.assertTrue(current.satisfied)
        self.assertEqual(current.receipt_id, later.id)

    def test_verdict_denied_and_revoked(self) -> None:
        digest = args_digest("write_file", {"path": "notes.txt"})
        denied = self.log.record(self.request(), approved=False, session_id="s1", arg_digest=digest, scope="notes.txt")
        verdict = self.log.verdict(
            action="write_file", arg_digest=digest, scope="notes.txt", session_id="s1", mode="ask"
        )
        self.assertEqual(verdict.reason_code, "approval_denied")
        self.assertEqual(verdict.receipt_id, denied.id)
        granted = self.log.record(self.request(), approved=True, session_id="s2", arg_digest=digest, scope="notes.txt")
        revoked = self.log.revoke(granted.id)
        after = self.log.verdict(
            action="write_file", arg_digest=digest, scope="notes.txt", session_id="s2", mode="ask"
        )
        self.assertEqual(after.reason_code, "approval_revoked")
        self.assertEqual(after.receipt_id, revoked.id)

    def test_revoke_guards(self) -> None:
        granted = self.log.record(self.request(), approved=True, session_id="s1", scope="notes.txt")
        with self.assertRaises(GateError):
            self.log.revoke("apr-nope")
        with self.assertRaises(GateError):
            self.log.revoke("")
        revoked = self.log.revoke(granted.id, actor="owner")
        self.assertEqual(revoked.decision, "revoked")
        self.assertEqual(revoked.actor, "owner")
        self.assertEqual(revoked.supersedes, granted.id)
        self.assertEqual(
            (revoked.action, revoked.scope, revoked.session_id, revoked.mode),
            (granted.action, granted.scope, granted.session_id, granted.mode),
        )
        with self.assertRaises(GateError):
            self.log.revoke(granted.id)  # no longer the current answer
        with self.assertRaises(GateError):
            self.log.revoke(revoked.id)  # already revoked

    def test_receipts_filters_and_rows(self) -> None:
        digest = args_digest("write_file", {"path": "notes.txt"})
        self.log.record(self.request(), approved=True, session_id="s1", arg_digest=digest, scope="notes.txt")
        self.log.record(self.request(), approved=False, session_id="s2", arg_digest=digest, scope="notes.txt")
        self.log.record(self.request(), approved=True, session_id="s2", arg_digest="other", scope="notes.txt")
        self.assertEqual(len(self.log.receipts()), 3)
        self.assertEqual(len(self.log.receipts(session_id="s1")), 1)
        self.assertEqual([item.decision for item in self.log.receipts(decision="denied")], ["denied"])
        self.assertEqual(len(self.log.receipts(limit=2)), 2)
        self.assertEqual(self.log.receipts(decision="granted")[0].session_id, "s2")
        self.assertEqual(len(self.log.receipts(limit=0)), 1)  # a limit below one clamps up
        with self.assertRaises(GateError):
            self.log.receipts(decision="maybe")
        self.assertEqual(self.log.headers(), ["receipt", "at", "action", "subject", "decision", "age"])
        rows = self.log.rows(self.log.receipts(limit=2), now=self.now[0])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][4], "granted")
        self.assertEqual(rows[0][5], "0s")
        self.assertEqual(self.log.rows([], now=self.now[0]), [])
        old = self.log.rows([self.log.receipts()[-1]], now=self.now[0] + 100)[0]
        self.assertEqual(old[5], "100s")

    def test_scope_canonicalisation_and_opaque_fallback(self) -> None:
        self.assertEqual(self.log.canonical_scope("notes.txt"), "notes.txt")
        self.assertEqual(self.log.canonical_scope("./notes.txt"), self.log.canonical_scope("notes.txt"))
        self.assertEqual(self.log.canonical_scope(""), "")
        self.assertEqual(self.log.canonical_scope("https://example.com/x"), "https://example.com/x")
        self.assertTrue(self.log.canonical_scope("/etc/passwd").startswith("ref:"))
        self.assertTrue(self.log.canonical_scope("../outside").startswith("ref:"))
        folded = self.log.canonical_scope("token=secret-value")
        self.assertTrue(folded.startswith("ref:"))
        self.assertNotIn("secret-value", folded)

    # -- the confirm wrapper -------------------------------------------------

    def test_wrapper_records_question_and_answer_and_never_auto_grants(self) -> None:
        asked: list[ApprovalRequest] = []

        def ui(request: ApprovalRequest) -> bool:
            asked.append(request)
            return True

        confirm = make_confirm_wrapper(ui, log=self.log, gates=self.gates, journal=self.journal, session_id="s1")
        request = self.request()
        self.assertTrue(confirm(request))
        self.assertTrue(confirm(request))
        self.assertEqual(len(asked), 2)  # a stored grant released nothing
        self.assertEqual(self.gates.open_gates(session_id="s1"), [])
        self.assertEqual([item["state"] for item in self.history()], ["open", "answered", "open", "answered"])
        self.assertEqual([item.decision for item in self.log.receipts()], ["granted", "granted"])
        self.assertEqual(self.journal.counts()["gate_opened"], 2)
        self.assertEqual(self.journal.counts()["gate_answered"], 2)

    def test_wrapper_decline_paths(self) -> None:
        confirm = make_confirm_wrapper(
            lambda request: False, log=self.log, gates=self.gates, journal=self.journal, session_id="s2"
        )
        self.assertFalse(confirm(self.request(action="shell", summary="rm -rf build", detail="build", risk="high")))
        self.assertEqual([item.decision for item in self.log.receipts(session_id="s2")], ["denied"])
        self.assertEqual(self.gates.pending_subjects(session_id="s2"), [])
        gate = self.gates.open_gates(session_id="s2")
        self.assertEqual(gate, [])
        digest = args_digest("shell", {"summary": "rm -rf build", "detail": "build"})
        verdict = self.log.verdict(action="shell", arg_digest=digest, scope="build", session_id="s2", mode="ask")
        self.assertEqual(verdict.reason_code, "approval_denied")

    def test_wrapper_without_a_human_surface_leaves_the_question_open(self) -> None:
        confirm = make_confirm_wrapper(None, log=self.log, gates=self.gates, journal=self.journal, session_id="s3")
        self.assertFalse(confirm(self.request(action="web_fetch", summary="fetch", detail="https://example.com")))
        open_gates = self.gates.open_gates(session_id="s3")
        self.assertEqual(len(open_gates), 1)
        self.assertEqual(open_gates[0].question_code, "allow_network")
        self.assertEqual(self.log.receipts(session_id="s3"), [])
        self.assertFalse(confirm(self.request(action="web_fetch", summary="fetch", detail="https://example.com")))
        self.assertEqual(len(self.gates.open_gates(session_id="s3")), 1)  # one question, not two
        self.assertEqual(self.journal.counts()["approval_answered"], 2)

    def test_wrapper_read_only_refuses_without_asking(self) -> None:
        confirm = make_confirm_wrapper(
            lambda request: True,
            log=self.log,
            gates=self.gates,
            journal=self.journal,
            session_id="s4",
            read_only=True,
        )
        self.assertFalse(confirm(self.request()))
        self.assertEqual(self.gates.open_gates(session_id="s4"), [])
        receipts = self.log.receipts(session_id="s4")
        self.assertEqual([(item.decision, item.mode) for item in receipts], [("denied", "read-only")])
        self.assertFalse(confirm("not a request"))  # type: ignore[arg-type]

    def test_wrapper_reopens_after_a_previous_answer(self) -> None:
        calls: list[ApprovalRequest] = []

        def ui(request: ApprovalRequest) -> bool:
            calls.append(request)
            return len(calls) == 1

        confirm = make_confirm_wrapper(ui, log=self.log, gates=self.gates, journal=self.journal, session_id="s5")
        request = self.request()
        self.assertTrue(confirm(request))
        self.assertFalse(confirm(request))
        self.assertEqual(len(calls), 2)
        self.assertEqual([item.decision for item in self.log.receipts(session_id="s5")], ["denied", "granted"])

    # -- the gates tool ------------------------------------------------------

    def test_gate_tool_reads(self) -> None:
        tool = make_gate_tool(self.gates, self.log)
        self.assertEqual(tool.name, "gates")
        self.assertFalse(tool.read_only)
        self.assertEqual(tool.api_spec()["function"]["parameters"]["required"], ["operation"])
        ctx = self.ctx()
        self.assertEqual(tool.handler({"operation": "list"}, ctx).meta["count"], 0)
        gate = self.open_one()
        listed = tool.handler({"operation": "list", "limit": 5, "session": "s1"}, ctx)
        self.assertIn(gate.gate_id, listed.content)
        self.assertEqual(listed.meta["count"], 1)
        pending = tool.handler({"operation": "pending", "session": "s1"}, ctx)
        self.assertIn("notes.txt", pending.content)
        self.assertFalse(tool.handler({"operation": "peek"}, ctx).ok)
        self.assertEqual(tool.handler({"operation": "list", "session": "none"}, ctx).meta["count"], 0)
        self.assertIn("No subjects", tool.handler({"operation": "pending", "session": "none"}, ctx).content)

    def test_gate_tool_answer_requires_approval_outside_ask_mode(self) -> None:
        tool = make_gate_tool(self.gates, self.log)
        gate = self.open_one()
        args = {"operation": "answer", "gate": gate.gate_id, "choice": "approve"}
        refused = tool.handler(args, self.ctx(approval_mode="read-only"))
        self.assertFalse(refused.ok)
        self.assertTrue(refused.meta["denied"])
        self.assertEqual(self.gates.head(gate.gate_id).state, "open")
        self.assertFalse(tool.handler({"operation": "answer", "gate": gate.gate_id}, self.ctx()).ok)
        self.assertFalse(tool.handler({"operation": "answer", "gate": gate.gate_id, "choice": "sure"}, self.ctx()).ok)
        asked: list[ApprovalRequest] = []
        prompt = self.ctx(approval_mode="ask", confirm=lambda request: asked.append(request) or False)
        # ask mode is already a prompt, so the tool does not ask a second time
        self.assertTrue(tool.handler(args, prompt).ok)
        self.assertEqual(asked, [])
        self.assertEqual(self.gates.head(gate.gate_id).state, "answered")
        self.assertFalse(tool.handler(args, self.ctx()).ok)  # one answer per question
        receipts = self.log.receipts()
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0].decision, "granted")
        verdict = self.log.verdict(
            action=gate.action,
            arg_digest=gate.arg_digest,
            scope=gate.subject,
            session_id=gate.session_id,
            mode="ask",
        )
        self.assertTrue(verdict.satisfied)

    def test_gate_tool_decline_writes_no_receipt(self) -> None:
        tool = make_gate_tool(self.gates, self.log)
        gate = self.open_one(subject="drop.txt", arg_digest="dd")
        result = tool.handler(
            {"operation": "answer", "gate": gate.gate_id, "choice": "decline"}, self.ctx()
        )
        self.assertTrue(result.ok)
        self.assertFalse(result.meta["released"])
        self.assertEqual(self.log.receipts(), [])

    # -- dataclass shapes ----------------------------------------------------

    def test_dataclass_defaults(self) -> None:
        gate = DecisionGate(
            id="gate-1",
            subject="s",
            session_id="",
            question_code="allow_write",
            question="q",
            action="a",
            arg_digest="d",
            risk="low",
            choices=["approve"],
        )
        self.assertEqual(gate.state, "open")
        self.assertEqual(gate.opened_at, "")
        self.assertFalse(gate.releases)
        self.assertTrue(gate.gate_id.startswith("gate-"))
        receipt = ApprovalReceipt(
            id="apr-1",
            at=iso(BASE),
            session_id="s",
            actor="operator",
            action="a",
            arg_digest="d",
            scope="s",
            mode="ask",
            decision="granted",
        )
        self.assertEqual(receipt.risk, "medium")
        self.assertEqual(receipt.gate_id, "")
        self.assertEqual(receipt.chain, receipt.chain)
        self.assertEqual(receipt.to_dict()["decision"], "granted")
        verdict = ApprovalVerdict(satisfied=False)
        self.assertEqual((verdict.reason_code, verdict.age_seconds, verdict.expires_in_seconds), ("", -1, 0))


if __name__ == "__main__":
    unittest.main()
