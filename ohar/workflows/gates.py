"""Durable consent: decision gates (the question) and approval receipts (the answer).

``ToolContext.request_approval`` is a synchronous in-process callback, so a pause
between turns simply vanishes: OHAR could not represent "a human was asked and has
not answered yet". A gate is that missing object — one append-only head record
bound to one subject, one blocked action digest and one resume digest. An answer
supersedes rather than edits, and a receipt binds five dimensions by equality with
no containment or prefix rule, so widening consent is structurally impossible.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..terminal import Theme, approval_card, clip, sanitize_text
from ..tools import ToolError, resolve_path
from ..types import ApprovalRequest, Tool, ToolContext, ToolResult
from .journal import UNKNOWN_AGE_SECONDS, ActivityJournal, args_digest as _journal_args_digest
from .store import (
    RecordStore,
    WorkflowError,
    canonical_digest,
    clean_text,
    new_record_id,
    now_iso,
    opaque_ref,
    short_digest,
    stamp_to_epoch,
    validate_vocabularies,
    workspace_key,
)

__all__ = [
    "GATE_CHOICES",
    "CHOICE_RELEASES",
    "GATE_STATES",
    "QUESTION_CODES",
    "RISK_CLASSES",
    "GATE_TTL_SECONDS",
    "RECEIPT_DECISIONS",
    "ACTION_CLASSES",
    "REFUSAL_CODES",
    "GateError",
    "classify_action",
    "args_digest",
    "DecisionGate",
    "ApprovalReceipt",
    "ApprovalVerdict",
    "DecisionGates",
    "ApprovalLog",
    "make_confirm_wrapper",
    "make_gate_tool",
]

GATE_CHOICES: tuple[str, ...] = ("approve", "decline", "defer")
# Exactly one choice releases the blocked action; the table is what makes that a
# lookup instead of a string comparison scattered across readers.
CHOICE_RELEASES: dict[str, bool] = {"approve": True, "decline": False, "defer": False}
# There is no `withdrawn` or `expired` member: expiry is a property a reader
# derives from the stamp, and a withdrawal is what `decline` already is.
GATE_STATES: tuple[str, ...] = ("open", "answered")
QUESTION_CODES: tuple[str, ...] = (
    "allow_write",
    "allow_shell",
    "allow_network",
    "allow_browser_effect",
    "accept_risky_change",
    "resume_blocked_goal",
    "discard_changes",
)
# Consequence classes a decision can carry. The approval card renders a level
# (low/medium/high); these name what undoing the effect costs, which is the part
# an operator actually acts on.
RISK_CLASSES: tuple[str, ...] = ("reversible", "costly_to_reverse", "irreversible")
GATE_TTL_SECONDS = 3600
RECEIPT_DECISIONS: tuple[str, ...] = ("granted", "denied", "revoked")
ACTION_CLASSES: tuple[str, ...] = ("read_only", "mutating", "sensitive", "disabled", "blocked")
REFUSAL_CODES: tuple[str, ...] = (
    "approval_absent",
    "approval_expired",
    "approval_revoked",
    "approval_superseded",
    "approval_denied",
    "action_not_approved",
    "subject_not_approved",
    "session_not_approved",
    "mode_not_approved",
    "arg_digest_not_approved",
)

GATE_SCHEMA = "decision_gate/v1"
RECEIPT_SCHEMA = "approval_receipt/v1"

APPROVAL_MODES: tuple[str, ...] = ("ask", "auto", "read-only")
_RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high")
_RISK_VALUES: tuple[str, ...] = _RISK_LEVELS + RISK_CLASSES
_CARD_RISK: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "reversible": "low",
    "costly_to_reverse": "medium",
    "irreversible": "high",
}
_RISK_TEXT: dict[str, str] = {
    "reversible": "Releasing this has an effect the same surface can undo.",
    "costly_to_reverse": "Releasing this can be undone only by work outside this session.",
    "irreversible": "Releasing this cannot be undone once it is applied.",
}
_QUESTION_TEXT: dict[str, str] = {
    "allow_write": "Allow this change to files in the workspace?",
    "allow_shell": "Allow this command to run?",
    "allow_network": "Allow this request to leave the machine?",
    "allow_browser_effect": "Allow this effect in the browser?",
    "accept_risky_change": "Accept this change, knowing it is not trivially reversible?",
    "resume_blocked_goal": "Resume the blocked goal under these terms?",
    "discard_changes": "Discard the changes made so far?",
}
_REFUSAL_REASONS: dict[str, str] = {
    "approval_absent": "No approval on record names this action, arguments, subject, session, and mode.",
    "approval_expired": "The approval on record is outside the consent window, measured now from its stamp.",
    "approval_revoked": "The approval on record was revoked.",
    "approval_superseded": "The approval on record was superseded by a later answer to this confirmation.",
    "approval_denied": "The confirmation on record was answered with a decline.",
    "action_not_approved": "The approval on record names another action; consent never extends to a sibling.",
    "subject_not_approved": (
        "The approval on record names another subject; consent covers exactly the subject it named."
    ),
    "session_not_approved": "The approval on record belongs to another session.",
    "mode_not_approved": "The approval on record was given under another approval mode.",
    "arg_digest_not_approved": "The approval on record was given for different arguments.",
}
# Effects that earn the strongest question whatever a tool declares about itself:
# an interpreter, a whole-notebook rewrite, a memory wipe, and anything that lands
# outside the machine.
_SENSITIVE_NAMES = frozenset({"shell", "bash", "apply_patch", "notebook_edit", "memory_clear"})
_SENSITIVE_PREFIXES = ("browser_", "web_", "mcp_")
_BROWSER_ACTIONS = frozenset({"browser_click", "browser_type", "browser_press", "browser_open", "browser_scroll"})
_NETWORK_ACTIONS = frozenset({"web_fetch", "web_search"})
_SHELL_ACTIONS = frozenset({"shell", "bash", "run"})
_MAX_QUESTION_CHARS = 240
_MAX_LIST = 50
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")

CLAIM_BOUNDARY = (
    "A decision gate is one recorded question about one subject with at most one recorded answer. "
    "An answered gate proves a human chose one of the offered choices; it is not execution, review, "
    "or verification evidence, and it releases exactly the one action its resume digest names."
)


class GateError(WorkflowError):
    """A gate or receipt was refused, or the record on file cannot be trusted."""


def classify_action(name: str, *, read_only: bool, destructive: bool, approval_mode: str) -> str:
    """What class of operation one tool call is, given the policy in force.

    The vocabulary is closed because it decides whether a gate is needed at all:
    a read never is, a destructive effect always is, and a policy that turns the
    capability off is reported as such rather than silently downgraded.
    """
    identity = clean_text(name, limit=64)
    if read_only:
        return "read_only"
    if approval_mode == "read-only":
        return "blocked"
    if approval_mode not in APPROVAL_MODES:
        return "disabled"
    if destructive or identity in _SENSITIVE_NAMES or identity.startswith(_SENSITIVE_PREFIXES):
        return "sensitive"
    return "mutating"


def args_digest(tool_name: str, args: Mapping[str, Any]) -> str:
    """Identify one call without keeping its arguments.

    Deliberately the same function the journal uses, so a gate's blocked-action
    digest and a journal observation of the same call agree by construction
    instead of by two rules that can drift apart.
    """
    return _journal_args_digest(tool_name, args)


@dataclass(slots=True)
class DecisionGate:
    id: str
    subject: str
    session_id: str
    question_code: str
    question: str
    action: str
    arg_digest: str
    risk: str
    choices: list[str]
    state: str = "open"
    opened_at: str = ""
    decided_at: str = ""
    actor: str = ""
    answered_choice: str = ""
    resume_digest: str = ""
    supersedes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gate_id": self.gate_id,
            "subject": self.subject,
            "session_id": self.session_id,
            "question_code": self.question_code,
            "question": self.question,
            "action": self.action,
            "arg_digest": self.arg_digest,
            "risk": self.risk,
            "choices": list(self.choices),
            "state": self.state,
            "opened_at": self.opened_at,
            "decided_at": self.decided_at,
            "actor": self.actor,
            "answered_choice": self.answered_choice,
            "resume_digest": self.resume_digest,
            "supersedes": self.supersedes,
            "schema": GATE_SCHEMA,
        }

    @property
    def gate_id(self) -> str:
        """The question's identity: stable across re-answers, unlike record ids."""
        return _gate_identity(
            subject=self.subject,
            question_code=self.question_code,
            action=self.action,
            arg_digest=self.arg_digest,
            session_id=self.session_id,
        )

    @property
    def releases(self) -> bool:
        return self.state == "answered" and CHOICE_RELEASES.get(self.answered_choice, False)


@dataclass(slots=True)
class ApprovalReceipt:
    id: str
    at: str
    session_id: str
    actor: str
    action: str
    arg_digest: str
    scope: str
    mode: str
    decision: str
    risk: str = "medium"
    gate_id: str = ""
    supersedes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "at": self.at,
            "session_id": self.session_id,
            "actor": self.actor,
            "action": self.action,
            "arg_digest": self.arg_digest,
            "scope": self.scope,
            "mode": self.mode,
            "decision": self.decision,
            "risk": self.risk,
            "gate_id": self.gate_id,
            "supersedes": self.supersedes,
            "chain": self.chain,
            "schema": RECEIPT_SCHEMA,
        }

    @property
    def chain(self) -> str:
        return _confirmation_chain(action=self.action, scope=self.scope, session_id=self.session_id)


@dataclass(slots=True)
class ApprovalVerdict:
    satisfied: bool
    reason_code: str = ""
    reason: str = ""
    receipt_id: str = ""
    age_seconds: int = UNKNOWN_AGE_SECONDS
    expires_in_seconds: int = 0


def _gate_identity(
    *,
    subject: str,
    question_code: str,
    action: str,
    arg_digest: str,
    session_id: str,
) -> str:
    """One question per subject, code, action, arguments, and session."""
    return "gate-" + short_digest(
        {
            "subject": subject,
            "question_code": question_code,
            "action": action,
            "arg_digest": arg_digest,
            "session_id": session_id,
        },
        16,
    )


def _confirmation_chain(*, action: str, scope: str, session_id: str) -> str:
    """Which question a receipt answers.

    Deliberately coarser than the five bound dimensions: the chain groups the
    confirmations one subject went through, while a verdict still requires exact
    equality on all five. That gap is what makes "this grant is no longer the
    current answer" expressible at all.
    """
    return short_digest({"action": action, "scope": scope, "session_id": session_id}, 32)


def _ref(value: object, length: int = 128) -> str:
    return opaque_ref(value, limit=length)


def _iso(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat(timespec="microseconds")
    except (OSError, OverflowError, TypeError, ValueError):
        return now_iso()


def _age(at: str, now: float) -> int:
    stamp = stamp_to_epoch(at)
    if stamp is None:
        return UNKNOWN_AGE_SECONDS
    return max(0, int(now - stamp))


def _normalized_choices(values: Sequence[str]) -> list[str]:
    wanted = [clean_text(item, limit=32) for item in values] if isinstance(values, (list, tuple)) else []
    out: list[str] = []
    for choice in wanted:
        if choice not in GATE_CHOICES:
            raise GateError(f"gate choice is unsupported: {clean_text(choice, limit=40)!r}")
        if choice not in out:
            out.append(choice)
    if not out:
        raise GateError("a gate must offer at least one choice")
    return out


def _gate_from(data: Mapping[str, Any]) -> DecisionGate:
    if not isinstance(data, Mapping):
        raise GateError("gate record must be a mapping")
    errors = validate_vocabularies(data, (("state", GATE_STATES), ("risk", _RISK_VALUES)))
    if errors:
        raise GateError(f"gate record is unusable: {errors[0]}")
    return DecisionGate(
        id=_ref(data.get("id", ""), 64),
        subject=_ref(data.get("subject", ""), 128),
        session_id=_ref(data.get("session_id", ""), 128),
        question_code=clean_text(data.get("question_code", ""), limit=64),
        question=clean_text(data.get("question", ""), limit=_MAX_QUESTION_CHARS),
        action=_ref(data.get("action", ""), 128),
        arg_digest=clean_text(data.get("arg_digest", ""), limit=64),
        risk=clean_text(data.get("risk", "medium"), limit=32),
        choices=[clean_text(item, limit=32) for item in data.get("choices", ()) if isinstance(item, str)][
            : len(GATE_CHOICES)
        ],
        state=str(data.get("state", "")),
        opened_at=clean_text(data.get("opened_at", ""), limit=40),
        decided_at=clean_text(data.get("decided_at", ""), limit=40),
        actor=_ref(data.get("actor", ""), 64),
        answered_choice=clean_text(data.get("answered_choice", ""), limit=32),
        resume_digest=clean_text(data.get("resume_digest", ""), limit=64),
        supersedes=_ref(data.get("supersedes", ""), 64),
    )


def _receipt_from(data: Mapping[str, Any]) -> ApprovalReceipt:
    if not isinstance(data, Mapping):
        raise GateError("receipt record must be a mapping")
    errors = validate_vocabularies(data, (("decision", RECEIPT_DECISIONS), ("risk", _RISK_VALUES)))
    if errors:
        raise GateError(f"receipt record is unusable: {errors[0]}")
    return ApprovalReceipt(
        id=_ref(data.get("id", ""), 64),
        at=clean_text(data.get("at", ""), limit=40),
        session_id=_ref(data.get("session_id", ""), 128),
        actor=_ref(data.get("actor", ""), 64),
        action=_ref(data.get("action", ""), 128),
        arg_digest=clean_text(data.get("arg_digest", ""), limit=64),
        scope=clean_text(data.get("scope", ""), limit=240),
        mode=clean_text(data.get("mode", ""), limit=32),
        decision=str(data.get("decision", "")),
        risk=clean_text(data.get("risk", "medium"), limit=32),
        gate_id=_ref(data.get("gate_id", ""), 64),
        supersedes=_ref(data.get("supersedes", ""), 64),
    )


def _question_for(question_code: str, detail: str) -> str:
    base = _QUESTION_TEXT.get(question_code, "Approve this action?")
    note = clean_text(detail, limit=120)
    return clean_text(f"{base} {note}".strip(), limit=_MAX_QUESTION_CHARS)


class DecisionGates:
    """Append-only question records; an answer never edits the question."""

    def __init__(
        self,
        root: Path,
        workspace: Path,
        *,
        ttl_seconds: int = GATE_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
        journal: ActivityJournal | None = None,
    ) -> None:
        self.root = Path(root)
        self.workspace = Path(workspace)
        seconds = (
            float(ttl_seconds)
            if isinstance(ttl_seconds, (int, float)) and not isinstance(ttl_seconds, bool)
            else float(GATE_TTL_SECONDS)
        )
        self.ttl_seconds = max(1.0, seconds)
        self._clock = clock
        self._journal = journal
        self._store = RecordStore(self.root, "gates", workspace=self.workspace)

    def open(
        self,
        *,
        subject: str,
        question_code: str,
        action: str,
        arg_digest: str,
        session_id: str = "",
        risk: str = "medium",
        detail: str = "",
        choices: Sequence[str] = GATE_CHOICES,
    ) -> DecisionGate:
        safe_subject = _ref(subject, 128)
        safe_action = _ref(action, 128)
        if not safe_subject:
            raise GateError("a gate needs a subject")
        if not safe_action:
            raise GateError("a gate needs an action")
        if question_code not in QUESTION_CODES:
            raise GateError(f"question_code is unsupported: {clean_text(question_code, limit=40)!r}")
        if risk not in _RISK_VALUES:
            raise GateError(f"risk is unsupported: {clean_text(risk, limit=40)!r}")
        offered = _normalized_choices(choices)
        digest = clean_text(arg_digest, limit=64) or args_digest(safe_action, {})
        identity = _gate_identity(
            subject=safe_subject,
            question_code=question_code,
            action=safe_action,
            arg_digest=digest,
            session_id=_ref(session_id, 128),
        )
        now = float(self._clock())
        heads = self._store.heads("gate_id")
        mine = heads.get(identity)
        # One subject carries at most one live question; an expired question is no
        # longer live, so it may be re-asked rather than wedging the subject.
        for record in heads.values():
            if str(record.get("state", "")) != "open" or self._expired_at(record.get("opened_at"), now):
                continue
            if str(record.get("gate_id", "")) == identity:
                return _gate_from(record)
            if str(record.get("subject", "")) == safe_subject:
                raise GateError(
                    f"subject {safe_subject} already has an open question; answer it before asking another"
                )
        prior_id = str(mine.get("id", "")) if mine else ""

        def mutate(current: dict[str, Any] | None) -> dict[str, Any] | None:
            if current is not None and str(current.get("state", "")) == "open":
                if not self._expired_at(current.get("opened_at"), now):
                    return None  # another surface opened the same question first
            record = {
                "id": new_record_id("gate"),
                "gate_id": identity,
                "subject": safe_subject,
                "session_id": _ref(session_id, 128),
                "question_code": question_code,
                "question": _question_for(question_code, detail),
                "action": safe_action,
                "arg_digest": digest,
                "risk": risk,
                "choices": offered,
                "state": "open",
                "opened_at": _iso(now),
                "decided_at": "",
                "actor": "",
                "answered_choice": "",
                "supersedes": str(current.get("id", "")) if current else "",
                "workspace": workspace_key(self.workspace),
                "claim_boundary": CLAIM_BOUNDARY,
            }
            record["resume_digest"] = self._resume_digest(record)
            return record

        record, replayed = self._store.transact("gate_id", identity, mutate)
        gate = _gate_from(record)
        if self._journal is not None and not replayed and str(record.get("id", "")) != prior_id:
            self._journal.record(
                "gate_opened",
                subject=identity,
                ok=True,
                session_id=gate.session_id,
                detail={
                    "action": gate.action,
                    "question_code": gate.question_code,
                    "subject": gate.subject,
                    "risk": gate.risk,
                    "arg_digest": gate.arg_digest,
                    "record": gate.id,
                },
            )
        return gate

    def answer(self, gate_id: str, *, choice: str, actor: str = "operator") -> DecisionGate:
        identity = clean_text(gate_id, limit=128, single_line=True)
        if not identity:
            raise GateError("a gate id is required")
        wanted = clean_text(choice, limit=32)
        if wanted not in GATE_CHOICES:
            raise GateError(f"gate choice is unsupported: {clean_text(choice, limit=40)!r}")
        safe_actor = _ref(actor, 64) or "operator"
        stamp = _iso(float(self._clock()))

        def mutate(current: dict[str, Any] | None) -> dict[str, Any] | None:
            if current is None:
                raise GateError(f"no gate on record: {_ref(identity, 60)}")
            if str(current.get("state", "")) != "open":
                raise GateError("this gate is already answered; asking again needs a new question")
            offered = [str(item) for item in current.get("choices", ()) if isinstance(item, str)]
            if wanted not in offered:
                raise GateError(f"this gate did not offer {wanted!r}")
            if self._expired_at(current.get("opened_at"), float(self._clock())):
                raise GateError("this gate has expired; re-ask it to put a live question on record")
            record = dict(current)
            record.update(
                {
                    "id": new_record_id("gate"),
                    "state": "answered",
                    "answered_choice": wanted,
                    "actor": safe_actor,
                    "decided_at": stamp,
                    "supersedes": str(current.get("id", "")),
                }
            )
            return record

        record, _replayed = self._store.transact("gate_id", identity, mutate)
        gate = _gate_from(record)
        if gate.state != "answered":
            raise GateError("the answer did not land on the record")
        if self._journal is not None:
            self._journal.record(
                "gate_answered",
                subject=identity,
                ok=gate.releases,
                session_id=gate.session_id,
                detail={
                    "choice": gate.answered_choice,
                    "actor": gate.actor,
                    "action": gate.action,
                    "record": gate.id,
                },
            )
        return gate

    def head(self, gate_id: str) -> DecisionGate | None:
        record = self._store.latest("gate_id", clean_text(gate_id, limit=128, single_line=True))
        return _gate_from(record) if record is not None else None

    def open_gates(self, *, session_id: str | None = None, limit: int = 10) -> list[DecisionGate]:
        gates = [_gate_from(item) for item in self._store.heads("gate_id").values()]
        live = [item for item in gates if item.state == "open"]
        if session_id is not None:
            wanted = _ref(session_id, 128)
            live = [item for item in live if item.session_id == wanted]
        # Oldest unanswered first: a queue, not a feed. Records that share a
        # stamp keep their append order, so the queue never depends on an id.
        live.sort(key=lambda item: stamp_to_epoch(item.opened_at) or 0.0)
        return live[: _clamp(limit, 1, _MAX_LIST)]

    def is_expired(self, gate: DecisionGate, *, now: float | None = None) -> bool:
        return self._expired_at(gate.opened_at, float(self._clock()) if now is None else float(now))

    def pending_subjects(self, *, session_id: str | None = None) -> list[str]:
        now = float(self._clock())
        subjects: set[str] = set()
        for record in self._store.heads("gate_id").values():
            if str(record.get("state", "")) != "open" or self._expired_at(record.get("opened_at"), now):
                continue
            if session_id is not None and str(record.get("session_id", "")) != _ref(session_id, 128):
                continue
            subject = _ref(record.get("subject", ""), 128)
            if subject:
                subjects.add(subject)
        return sorted(subjects)

    def render(self, gate: DecisionGate, *, theme: Theme | None = None, glyphs: str = "unicode") -> str:
        risk = _CARD_RISK.get(gate.risk, "medium")
        state = self._state_of(gate, float(self._clock()))
        foot = [
            f"gate {gate.gate_id} [{state}]  risk {gate.risk}  question {gate.question_code}",
            f"action {gate.action}  args {gate.arg_digest}  subject {gate.subject}",
            f"choices {'/'.join(gate.choices) or '(none)'}  answered {gate.answered_choice or '-'}"
            f"  actor {gate.actor or '-'}",
        ]
        if gate.risk in _RISK_TEXT:
            foot.append(_RISK_TEXT[gate.risk])
        foot.append(CLAIM_BOUNDARY)
        if theme is None:
            return "\n".join([clip(sanitize_text(gate.question), _MAX_QUESTION_CHARS), *foot])
        lines = approval_card(
            theme,
            action=gate.action,
            summary=gate.question,
            detail=f"subject {gate.subject} · args {gate.arg_digest} · session {gate.session_id or '-'}",
            risk=risk,
            glyphs=glyphs,
        )
        return "\n".join([*lines, *foot])

    def headers(self) -> list[str]:
        return ["gate", "subject", "question", "action", "state", "opened"]

    def rows(self, gates: Sequence[DecisionGate]) -> list[list[str]]:
        now = float(self._clock())
        out: list[list[str]] = []
        for gate in gates:
            out.append(
                [
                    clip(sanitize_text(gate.gate_id), 28),
                    clip(sanitize_text(gate.subject), 32),
                    clip(sanitize_text(gate.question_code), 24),
                    clip(sanitize_text(gate.action), 20),
                    self._state_of(gate, now),
                    clip(sanitize_text(gate.opened_at), 32),
                ]
            )
        return out

    def _state_of(self, gate: DecisionGate, now: float) -> str:
        """`expired` is never a stored state: readers derive it from the stamp."""
        if gate.state == "answered":
            return "answered"
        return "expired" if self._expired_at(gate.opened_at, now) else "open"

    def _expired_at(self, stamp: object, now: float) -> bool:
        epoch = stamp_to_epoch(clean_text(stamp, limit=40))
        if epoch is None:
            return True  # an unreadable stamp cannot prove freshness
        return (now - epoch) > self.ttl_seconds

    @staticmethod
    def _resume_digest(record: Mapping[str, Any]) -> str:
        return canonical_digest(
            {
                "subject": record.get("subject", ""),
                "action": record.get("action", ""),
                "arg_digest": record.get("arg_digest", ""),
                "session_id": record.get("session_id", ""),
                "question_code": record.get("question_code", ""),
            }
        )


class ApprovalLog:
    """Receipts of consent actually given, and verdicts derived from them."""

    def __init__(
        self,
        root: Path,
        workspace: Path,
        *,
        clock: Callable[[], float] = time.time,
        journal: ActivityJournal | None = None,
    ) -> None:
        self.root = Path(root)
        self.workspace = Path(workspace)
        # One consent window for the whole package: a gate and the receipt that
        # answers it must not disagree about how long an answer speaks for.
        self.ttl_seconds = float(GATE_TTL_SECONDS)
        self._clock = clock
        self._journal = journal
        self._store = RecordStore(self.root, "approvals", workspace=self.workspace)

    def record(
        self,
        request: ApprovalRequest,
        *,
        approved: bool,
        session_id: str = "",
        action: str = "",
        arg_digest: str = "",
        scope: str = "",
        mode: str = "ask",
        gate_id: str = "",
    ) -> ApprovalReceipt:
        if not isinstance(request, ApprovalRequest):
            raise GateError("ApprovalLog.record expects an ApprovalRequest")
        identity = _ref(action or request.action, 128)
        if not identity:
            raise GateError("a receipt needs an action")
        if request.risk not in _RISK_VALUES:
            raise GateError(f"risk is unsupported: {clean_text(request.risk, limit=40)!r}")
        if mode not in APPROVAL_MODES:
            raise GateError(f"approval mode is unsupported: {clean_text(mode, limit=40)!r}")
        digest = clean_text(arg_digest, limit=64) or args_digest(identity, {})
        safe_scope = self.canonical_scope(scope)
        safe_session = _ref(session_id, 128)
        now = float(self._clock())
        chain = _confirmation_chain(action=identity, scope=safe_scope, session_id=safe_session)
        prior = self._chain_head(chain)
        receipt = ApprovalReceipt(
            id=new_record_id("apr"),
            at=_iso(now),
            session_id=safe_session,
            actor="operator",
            action=identity,
            arg_digest=digest,
            scope=safe_scope,
            mode=mode,
            decision="granted" if approved else "denied",
            risk=request.risk,
            gate_id=_ref(gate_id, 64),
            supersedes=str(prior.get("id", "")) if prior else "",
        )
        record = receipt.to_dict()
        record["workspace"] = workspace_key(self.workspace)
        self._store.append(record)
        if self._journal is not None:
            self._journal.record(
                "approval_answered",
                subject=receipt.action,
                ok=approved,
                session_id=receipt.session_id,
                detail={
                    "decision": receipt.decision,
                    "arg_digest": receipt.arg_digest,
                    "mode": receipt.mode,
                    "gate_id": receipt.gate_id,
                    "receipt": receipt.id,
                },
            )
        return receipt

    def canonical_scope(self, value: object) -> str:
        """Confine a path-like scope to the workspace before it is digested.

        Anything that is not a workspace path stays an opaque reference: a scope
        is bound by equality, so it must never be guessed at or widened.
        """
        text = clean_text(value, limit=240)
        if not text:
            return ""
        folded = _ref(text, 240)
        if folded != text and not text.startswith("."):
            # The store's own ref policy already hides a secret-shaped scope; a
            # leading dot is just a relative path, which resolution normalises.
            return folded
        if _SCHEME_RE.match(text):
            return folded  # a URL or uri scope stays exactly what it was
        try:
            resolved = resolve_path(ToolContext(workspace=self.workspace), text)
        except (ToolError, OSError, TypeError, ValueError):
            return _ref(text, 240)
        try:
            relative = resolved.relative_to(self.workspace.expanduser().resolve(strict=False))
            return clean_text(str(relative) or ".", limit=240)
        except (OSError, RuntimeError, ValueError):
            return _ref(str(resolved), 240)

    def verdict(
        self,
        *,
        action: str,
        arg_digest: str,
        scope: str,
        session_id: str,
        mode: str,
        now: float | None = None,
    ) -> ApprovalVerdict:
        identity = _ref(action, 128)
        digest = clean_text(arg_digest, limit=64) or args_digest(identity, {})
        safe_scope = self.canonical_scope(scope)
        safe_session = _ref(session_id, 128)
        stamp = float(self._clock()) if now is None else float(now)
        wanted_mode = clean_text(mode, limit=32)
        records = self._records()
        exact = [
            item
            for item in records
            if str(item.get("action", "")) == identity
            and str(item.get("arg_digest", "")) == digest
            and str(item.get("scope", "")) == safe_scope
            and str(item.get("session_id", "")) == safe_session
            and str(item.get("mode", "")) == wanted_mode
        ]
        if not exact:
            return self._absent(
                {
                    "action": identity,
                    "scope": safe_scope,
                    "session_id": safe_session,
                    "mode": wanted_mode,
                    "arg_digest": digest,
                },
                records,
                stamp,
            )
        current = exact[-1]
        chain = _confirmation_chain(action=identity, scope=safe_scope, session_id=safe_session)
        # Superseded first: a later answer to this confirmation is the current
        # one whatever this record said.
        if str(self._chain_head(chain, records).get("id", "")) != str(current.get("id", "")):
            return self._refuse("approval_superseded", current, stamp)
        decision = str(current.get("decision", ""))
        if decision == "revoked":
            return self._refuse("approval_revoked", current, stamp)
        if decision != "granted":
            return self._refuse("approval_denied", current, stamp)
        age = _age(clean_text(current.get("at", ""), limit=40), stamp)
        if age == UNKNOWN_AGE_SECONDS or age > self.ttl_seconds:
            return self._refuse("approval_expired", current, stamp)
        return ApprovalVerdict(
            satisfied=True,
            reason_code="",
            reason="Consent on record covers exactly this action, arguments, subject, session, and mode.",
            receipt_id=_ref(current.get("id", ""), 64),
            age_seconds=age,
            expires_in_seconds=max(0, int(self.ttl_seconds - age)),
        )

    def receipts(
        self,
        *,
        session_id: str | None = None,
        decision: str | None = None,
        limit: int = 20,
    ) -> list[ApprovalReceipt]:
        if decision is not None and decision not in RECEIPT_DECISIONS:
            raise GateError(f"receipt decision is unsupported: {clean_text(decision, limit=40)!r}")
        wanted_session = _ref(session_id, 128) if session_id is not None else None
        out: list[ApprovalReceipt] = []
        for item in reversed(self._store.records()):
            if not isinstance(item, dict):
                continue
            if wanted_session is not None and str(item.get("session_id", "")) != wanted_session:
                continue
            if decision is not None and str(item.get("decision", "")) != decision:
                continue
            out.append(_receipt_from(item))
            if len(out) == _clamp(limit, 1, _MAX_LIST):
                break
        return out

    def revoke(self, receipt_id: str, *, actor: str = "operator") -> ApprovalReceipt:
        wanted = clean_text(receipt_id, limit=64, single_line=True)
        if not wanted:
            raise GateError("a receipt id is required")
        records = [item for item in self._store.records() if isinstance(item, dict)]
        target = next((item for item in reversed(records) if str(item.get("id", "")) == wanted), None)
        if target is None:
            raise GateError(f"no approval receipt on record: {_ref(wanted, 60)}")
        if str(target.get("decision", "")) == "revoked":
            raise GateError("this approval is already revoked")
        chain = str(target.get("chain", "")) or _confirmation_chain(
            action=str(target.get("action", "")),
            scope=str(target.get("scope", "")),
            session_id=str(target.get("session_id", "")),
        )
        head = self._chain_head(chain, records)
        if str(head.get("id", "")) != wanted:
            # A superseded grant speaks for nothing already, so there is no
            # second way to withdraw it: only the current answer can be revoked.
            raise GateError("this approval is no longer the current answer; revoke that one instead")
        receipt = _receipt_from(target)
        now = float(self._clock())
        revoked = ApprovalReceipt(
            id=new_record_id("apr"),
            at=_iso(now),
            session_id=receipt.session_id,
            actor=_ref(actor, 64) or "operator",
            action=receipt.action,
            arg_digest=receipt.arg_digest,
            scope=receipt.scope,
            mode=receipt.mode,
            decision="revoked",
            risk=receipt.risk,
            gate_id=receipt.gate_id,
            supersedes=receipt.id,
        )
        record = revoked.to_dict()
        record["workspace"] = workspace_key(self.workspace)
        self._store.append(record)
        if self._journal is not None:
            self._journal.record(
                "approval_answered",
                subject=revoked.action,
                ok=False,
                session_id=revoked.session_id,
                detail={"decision": "revoked", "receipt": revoked.id, "actor": revoked.actor},
            )
        return revoked

    def headers(self) -> list[str]:
        return ["receipt", "at", "action", "subject", "decision", "age"]

    def rows(self, receipts: Sequence[ApprovalReceipt], *, now: float | None = None) -> list[list[str]]:
        stamp = float(self._clock()) if now is None else float(now)
        out: list[list[str]] = []
        for receipt in receipts:
            age = _age(receipt.at, stamp)
            out.append(
                [
                    clip(sanitize_text(receipt.id), 28),
                    clip(sanitize_text(receipt.at), 32),
                    clip(sanitize_text(receipt.action), 20),
                    clip(sanitize_text(receipt.scope or "-"), 28),
                    clip(sanitize_text(receipt.decision), 12),
                    "--" if age == UNKNOWN_AGE_SECONDS else f"{age}s",
                ]
            )
        return out

    def _records(self) -> list[dict[str, Any]]:
        return [item for item in self._store.records() if isinstance(item, dict)]

    def _chain_head(self, chain: str, records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        found: dict[str, Any] = {}
        source = self._records() if records is None else records
        for item in source:
            if str(item.get("chain", "")) == chain:
                found = dict(item)
        return found

    def _absent(
        self,
        wanted: Mapping[str, str],
        records: Sequence[Mapping[str, Any]],
        now: float,
    ) -> ApprovalVerdict:
        """Name the dimension a near-miss grant fails on, in binding order.

        "There is an approval for something else" is a more useful refusal than
        "there is no approval", and either way the answer is still no.
        """
        order = (
            ("action", "action_not_approved"),
            ("scope", "subject_not_approved"),
            ("session_id", "session_not_approved"),
            ("mode", "mode_not_approved"),
            ("arg_digest", "arg_digest_not_approved"),
        )
        best: tuple[int, str, Mapping[str, Any]] | None = None
        for item in records:
            if str(item.get("decision", "")) != "granted":
                continue
            mismatches = [code for field, code in order if str(item.get(field, "")) != wanted[field]]
            if not mismatches:
                continue
            if best is None or len(mismatches) < best[0]:
                best = (len(mismatches), mismatches[0], item)
        if best is None:
            return self._refuse("approval_absent", {}, now)
        return self._refuse(best[1], best[2], now)

    def _refuse(self, code: str, record: Mapping[str, Any], now: float) -> ApprovalVerdict:
        if code not in REFUSAL_CODES:
            raise GateError(f"refusal code is unsupported: {clean_text(code, limit=40)!r}")
        age = _age(clean_text(record.get("at", ""), limit=40), now) if record else UNKNOWN_AGE_SECONDS
        return ApprovalVerdict(
            satisfied=False,
            reason_code=code,
            reason=_REFUSAL_REASONS[code],
            receipt_id=_ref(record.get("id", ""), 64) if record else "",
            age_seconds=age,
            expires_in_seconds=0,
        )


def _clamp(value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return high
    return max(low, min(high, value))


def _question_code_for(action: str, risk: str) -> str:
    identity = clean_text(action, limit=64).casefold()
    if identity in _SHELL_ACTIONS:
        return "allow_shell"
    if identity in _BROWSER_ACTIONS:
        return "allow_browser_effect"
    if identity in _NETWORK_ACTIONS or identity.startswith("mcp_"):
        return "allow_network"
    if risk in ("high", "irreversible", "costly_to_reverse"):
        return "accept_risky_change"
    return "allow_write"


def make_confirm_wrapper(
    ui_confirm: Callable[[ApprovalRequest], bool] | None,
    *,
    log: ApprovalLog,
    gates: DecisionGates,
    journal: ActivityJournal,
    session_id: str = "",
    read_only: bool = False,
) -> Callable[[ApprovalRequest], bool]:
    """Put every approval on durable record; never grant from a stored receipt.

    The returned callback is what ``ToolContext.confirm`` should be set to. It
    asks the human surface it was given and records the question and the answer.
    It deliberately does not consult ``ApprovalLog.verdict``: a prior grant must
    not release a later, differently-shaped action, which is how widened consent
    happens.
    """

    def confirm(request: ApprovalRequest) -> bool:
        if not isinstance(request, ApprovalRequest):
            journal.record(
                "approval_answered",
                ok=False,
                session_id=session_id,
                detail={"error": "malformed approval request"},
            )
            return False
        action = clean_text(request.action, limit=64) or "action"
        digest = args_digest(action, {"summary": request.summary, "detail": request.detail})
        if read_only:
            log.record(request, approved=False, session_id=session_id, arg_digest=digest, mode="read-only")
            return False
        subject = log.canonical_scope(request.detail) or f"{action}:{short_digest(digest, 12)}"
        gate = gates.open(
            subject=subject,
            question_code=_question_code_for(action, request.risk),
            action=action,
            arg_digest=digest,
            session_id=session_id,
            risk=request.risk,
            detail=request.summary,
        )
        if ui_confirm is None:
            # No human surface: the question stays open on record and nothing runs.
            journal.record(
                "approval_answered",
                subject=action,
                ok=None,
                session_id=session_id,
                detail={"error": "no approval surface", "gate": gate.gate_id},
            )
            return False
        approved = bool(ui_confirm(request))
        try:
            answered = gates.answer(gate.gate_id, choice="approve" if approved else "decline")
        except (GateError, OSError) as exc:
            journal.record(
                "approval_answered",
                subject=action,
                ok=False,
                session_id=session_id,
                detail={"error": clean_text(exc, limit=120), "gate": gate.gate_id},
            )
            return False
        log.record(
            request,
            approved=approved,
            session_id=session_id,
            action=action,
            arg_digest=digest,
            scope=subject,
            mode="ask",
            gate_id=answered.id,
        )
        return approved

    return confirm


def make_gate_tool(gates: DecisionGates, log: ApprovalLog) -> Tool:
    """The `gates` tool: list, pending, and answer one recorded question."""

    def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        operation = clean_text(args.get("operation", ""), limit=16).casefold()
        if operation not in ("list", "pending", "answer"):
            return ToolResult.failure("Argument 'operation' must be list, pending, or answer.")
        if operation in ("list", "pending"):
            if operation == "pending":
                subjects = gates.pending_subjects(
                    session_id=clean_text(args.get("session", ""), limit=128) or None
                )
                if not subjects:
                    return ToolResult.success("No subjects are waiting on a decision.", count=0)
                return ToolResult.success(
                    "Waiting on a decision:\n" + "\n".join(f"- {item}" for item in subjects),
                    count=len(subjects),
                    subjects=subjects,
                )
            limit = _clamp(args.get("limit"), 1, _MAX_LIST)
            shown = gates.open_gates(session_id=clean_text(args.get("session", ""), limit=128) or None, limit=limit)
            if not shown:
                return ToolResult.success("No open decision gates.", count=0)
            headers = gates.headers()
            body = [" | ".join(headers), " | ".join(["-"] * len(headers))]
            body.extend(" | ".join(row) for row in gates.rows(shown))
            return ToolResult.success("\n".join(body), count=len(shown))
        gate_id = clean_text(args.get("gate", ""), limit=128)
        choice = clean_text(args.get("choice", ""), limit=16)
        if not gate_id or choice not in GATE_CHOICES:
            return ToolResult.failure("Answering needs 'gate' and a 'choice' of approve, decline, or defer.")
        if ctx.approval_mode != "ask" and not ctx.request_approval(
            ApprovalRequest(
                action="gate_answer",
                summary=f"Answer gate {gate_id} with {choice}",
                detail="",
                risk="medium",
            )
        ):
            return ToolResult.failure("Gate answer denied; the question stays open.", denied=True)
        try:
            gate = gates.answer(gate_id, choice=choice)
        except (TypeError, ValueError, OSError) as exc:
            return ToolResult.failure(f"gates: {exc}")
        if gate.releases:
            log.record(
                ApprovalRequest(
                    action=gate.action,
                    summary=f"Durable approval for {gate.subject}",
                    detail=gate.question,
                    risk=gate.risk if gate.risk in _RISK_VALUES else "medium",
                ),
                approved=True,
                session_id=gate.session_id,
                action=gate.action,
                arg_digest=gate.arg_digest,
                scope=gate.subject,
                mode="ask",
                gate_id=gate.id,
            )
        return ToolResult.success(
            gates.render(gate),
            gate=gate.gate_id,
            state=gate.state,
            released=gate.releases,
        )

    return Tool(
        name="gates",
        description=(
            "Read and answer durable decision gates. 'list' shows open questions, "
            "'pending' shows subjects waiting on a decision, 'answer' records one "
            "choice against one gate; an approve answer also writes the receipt."
        ),
        parameters={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["list", "pending", "answer"]},
                "gate": {"type": "string", "description": "Gate id (answer only)."},
                "choice": {"type": "string", "enum": list(GATE_CHOICES), "description": "Answer (answer only)."},
                "session": {"type": "string", "description": "Restrict to one session id."},
                "limit": {"type": "integer", "description": "Max gates to list.", "minimum": 1, "maximum": _MAX_LIST},
            },
            "required": ["operation"],
            "additionalProperties": False,
        },
        handler=handler,
        read_only=False,
        destructive=False,
    )
