"""Durable answers to "why was this blocked?" and "how do I undo it?". Stdlib only.

A block record is metadata-only by construction: the refused request never
reaches the store as bytes. Only its class shape (field classes plus
closed-vocabulary values) and a non-reversible fingerprint of that shape are
kept, so "this exact shape of request was refused twice for the same reason"
survives the turn while no path, prompt, or payload ever does.

Nothing here runs a command, deletes anything, or rolls a workspace back. The
recovery anchor records what a baseline was and writes a recipe a person runs
themselves; `guidance` is text.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..terminal import sanitize_text
from ..types import Tool, ToolContext, ToolResult
from .journal import ActivityJournal
from .store import (
    MAX_TAIL_RECORDS,
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
)

REASON_DOMAINS: tuple[str, ...] = (
    "approval_denied",
    "read_only_policy",
    "workspace_escape",
    "gate_open",
    "provider_error",
    "budget_exhausted",
    "tool_failure",
    "cancelled",
    "verification_failed",
)

DECISION_OUTCOMES: tuple[str, ...] = ("allowed", "blocked", "cancelled", "completed", "failed")

WORK_CLAIMS: tuple[str, ...] = ("not_attempted", "attempted_not_completed", "completed")

ENFORCEMENT: tuple[str, ...] = ("enforced", "declared_not_enforced")

RECOVERY_ACTIONS: tuple[str, ...] = (
    "answer_gate",
    "allow_action",
    "leave_read_only",
    "retry_with_scope",
    "run_recovery_recipe",
    "no_recovery_available",
)

MAX_CLASS_SHAPE = 32
MAX_DIRTY_PATHS = 20

BLOCK_STORE_NAME = "blocks"
ANCHOR_STORE_NAME = "recovery_anchors"
MAX_SHAPE_FIELDS = 64
MAX_LABEL_CHARS = 48
MAX_REASON_CODE_CHARS = 64
MAX_DIRTY_SCAN = 2_000
MAX_RECIPE_STEPS = 8
MAX_NOTE_CHARS = 200
MAX_SCAN_RECORDS = MAX_TAIL_RECORDS
_CLAIM_CHARS = 400

# Sources OHAR itself stops work at, versus sources that only report that work
# stopped. The split decides `enforcement`, never a caller's wording: a model or
# an operator can describe a block they watched, and cannot make OHAR claim it
# was the one who enforced it.
ENFORCING_SOURCES: frozenset[str] = frozenset(
    {"tool", "tool_gate", "approval", "gate", "goal_loop", "policy", "runtime", "verification", "provider",
     "budget", "agent"}
)
# Reporting sources — model, assistant, subagent, operator, hook — are deliberately
# not listed as a constant: any name outside the enforcing set is declared, so a
# source someone invents later lands on the weaker side by construction.

# Outcome decides what a renderer may claim about the work itself, so an allow
# can never read as an attempt and a block can never read as a completion.
OUTCOME_WORK_CLAIMS: dict[str, str] = {
    "allowed": "not_attempted",
    "blocked": "not_attempted",
    "cancelled": "attempted_not_completed",
    "failed": "attempted_not_completed",
    "completed": "completed",
}

# The codes each domain accepts. Every list ends in `unspecified`: a code this
# build does not know becomes "a block we cannot explain", never caller text, and
# never a blank cell — `recovery_action_for` still answers with the domain action.
REASON_CODES: dict[str, tuple[str, ...]] = {
    "approval_denied": (
        "approval_absent", "approval_denied", "approval_expired", "approval_revoked",
        "approval_superseded", "unspecified",
    ),
    "read_only_policy": ("read_only_mode", "action_disabled", "tool_disabled", "policy_blocked", "unspecified"),
    "workspace_escape": (
        "outside_workspace", "path_outside_workspace", "symlink_escape", "write_refused", "unspecified",
    ),
    "gate_open": ("gate_open", "gate_pending", "unspecified"),
    "provider_error": ("provider_unavailable", "provider_auth", "provider_timeout", "rate_limited", "unspecified"),
    "budget_exhausted": ("turn_budget", "token_budget", "time_budget", "unspecified"),
    "tool_failure": ("tool_error", "timeout", "invalid_args", "unspecified"),
    "cancelled": ("operator_cancel", "turn_cancel", "unspecified"),
    "verification_failed": ("check_failed", "check_timeout", "no_frozen_check", "attempts_exhausted", "unspecified"),
}

# Recovery per reason code, then per domain.
RECOVERY_BY_CODE: dict[str, str] = {
    "approval_denied": "allow_action",
    "approval_absent": "answer_gate",
    "approval_expired": "answer_gate",
    "approval_revoked": "answer_gate",
    "gate_open": "answer_gate",
    "gate_pending": "answer_gate",
    "read_only": "leave_read_only",
    "read_only_mode": "leave_read_only",
    "outside_workspace": "retry_with_scope",
    "workspace_escape": "retry_with_scope",
    "symlink_escape": "retry_with_scope",
    "provider_error": "retry_with_scope",
    "rate_limited": "retry_with_scope",
    "budget_exhausted": "retry_with_scope",
    "tool_failure": "retry_with_scope",
    "cancelled": "no_recovery_available",
    "check_failed": "run_recovery_recipe",
    "verification_failed": "run_recovery_recipe",
    "no_frozen_check": "retry_with_scope",
}
RECOVERY_BY_DOMAIN: dict[str, str] = {
    "approval_denied": "allow_action",
    "read_only_policy": "leave_read_only",
    "workspace_escape": "retry_with_scope",
    "gate_open": "answer_gate",
    "provider_error": "retry_with_scope",
    "budget_exhausted": "retry_with_scope",
    "tool_failure": "retry_with_scope",
    "cancelled": "no_recovery_available",
    "verification_failed": "run_recovery_recipe",
}

BLOCK_HEADERS: tuple[str, ...] = (
    "at",
    "source",
    "domain",
    "reason",
    "action",
    "outcome",
    "enforcement",
    "recovery",
)

FIELD_CLASSES: tuple[str, ...] = (
    "secret",
    "path",
    "url",
    "command",
    "content",
    "vocabulary",
    "identifier",
    "scalar",
    "collection",
    "empty",
    "opaque",
)

BLOCK_CLAIM_BOUNDARY = (
    "A block record is one recorded decision about one request shape: what was refused, under which "
    "code, and what recovery it allows. It carries the class of the request and never its content, and "
    "from a declared source it is not evidence that OHAR enforced anything."
)
ANCHOR_CLAIM_BOUNDARY = (
    "A recovery anchor is bounded metadata plus written instructions. OHAR runs none of them, performs "
    "no rollback, and creates or deletes nothing. A prepared anchor is not a recoverable baseline."
)

_ANCHOR_STATUSES: tuple[str, ...] = ("prepared", "baseline_observed")
_RECOVERABLE_STATUS = "baseline_observed"
_DIRTY_STATES: tuple[str, ...] = ("not_observed", "candidates_found", "clean_by_mtime")

_SECRET_KEY_RE = re.compile(r"(?i)(token|secret|passw|api[_-]?key|key|credential|authorization|bearer|cookie)")
_PATH_KEY_RE = re.compile(r"(?i)(^|_)(path|file|dir|folder|cwd)(_|$)")
_URL_KEY_RE = re.compile(r"(?i)url|uri|endpoint")
_COMMAND_KEY_RE = re.compile(r"(?i)^(command|cmd|shell|script)$")
_CONTENT_KEY_RE = re.compile(r"(?i)(content|text|prompt|body|message|summary|detail|code|diff|patch|html)")
_ID_KEY_RE = re.compile(r"(?i)(^|_)(id|uuid|sha|digest|hash)(_|$)")
_PATH_SHAPE_RE = re.compile(r"^\.{0,2}[/\\]|[A-Za-z]:[\\/]")
_URL_SHAPE_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_ID_SHAPE_RE = re.compile(r"^[A-Za-z0-9._:-]{4,128}$")
_VOCABULARY_VALUES: frozenset[str] = frozenset(
    REASON_DOMAINS
    + DECISION_OUTCOMES
    + WORK_CLAIMS
    + ENFORCEMENT
    + RECOVERY_ACTIONS
    + ("ask", "auto", "read-only", "low", "medium", "high", "true", "false")
)
_SKIPPED_DIRS: frozenset[str] = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache"})


@dataclass(slots=True)
class BlockedWork:
    """One recorded refusal. Every field is a class label or a digest."""

    id: str
    at: str
    decision_id: str
    source: str
    reason_domain: str
    reason_code: str
    action: str
    subject_digest: str
    class_shape: list[str]
    fingerprint: str
    outcome: str
    work_claim: str
    enforcement: str
    recovery_action: str
    session_id: str
    goal_id: str
    gate_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "id": self.id,
            "at": self.at,
            "decision_id": self.decision_id,
            "source": self.source,
            "reason_domain": self.reason_domain,
            "reason_code": self.reason_code,
            "action": self.action,
            "subject_digest": self.subject_digest,
            "class_shape": list(self.class_shape),
            "fingerprint": self.fingerprint,
            "outcome": self.outcome,
            "work_claim": self.work_claim,
            "enforcement": self.enforcement,
            "recovery_action": self.recovery_action,
            "session_id": self.session_id,
            "goal_id": self.goal_id,
            "gate_id": self.gate_id,
            "privacy": "metadata_only",
            "claim_boundary": BLOCK_CLAIM_BOUNDARY,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> "BlockedWork":
        """Repair-on-read: derived claims are re-derived, never trusted from disk."""
        record = data if isinstance(data, Mapping) else {}
        source = _ref(record.get("source"), 32)
        domain = _ref(record.get("reason_domain"), 32)
        if domain not in REASON_DOMAINS:
            domain = "tool_failure"
        outcome = _ref(record.get("outcome"), 24)
        if outcome not in DECISION_OUTCOMES:
            outcome = "blocked"
        shape = _clean_shape(record.get("class_shape"))
        action = _ref(record.get("action"), 64)
        stored = _ref(record.get("fingerprint"), 80)
        code = _reason_code(domain, record.get("reason_code"))
        return cls(
            id=_ref(record.get("id"), 64) or new_record_id("block"),
            at=_stamp(record.get("at")),
            decision_id=_ref(record.get("decision_id"), 64),
            source=source,
            reason_domain=domain,
            reason_code=code,
            action=action,
            subject_digest=_ref(record.get("subject_digest"), 80),
            class_shape=shape,
            fingerprint=stored if stored.startswith("fp-") else request_fingerprint(action, shape),
            outcome=outcome,
            work_claim=OUTCOME_WORK_CLAIMS[outcome],
            enforcement=enforcement_for(source),
            recovery_action=recovery_action_for(domain, code),
            session_id=_ref(record.get("session_id"), 64),
            goal_id=_ref(record.get("goal_id"), 64),
            gate_id=_ref(record.get("gate_id"), 64),
        )

    def render(self) -> str:
        return (
            f"{self.at} · {self.reason_domain}/{self.reason_code} · {self.action or 'unnamed'} · "
            f"{self.outcome} ({self.work_claim}) · {self.enforcement} → {self.recovery_action}"
        )


def request_class_shape(request: Mapping[str, Any] | None) -> list[str]:
    """The classes present in a request — the only thing about it that is kept.

    Every label is `field:<class>:<tail>` where the class is one of
    `FIELD_CLASSES` and the tail is a presence word or a closed-vocabulary
    value. No key name and no value a caller authored survives, which is what
    makes the fingerprint built from this list non-reversible rather than
    merely hashed.
    """
    if not isinstance(request, Mapping):
        return []
    labels: list[str] = []
    ordered = sorted(request.items(), key=lambda item: str(item[0]))
    for index, (key, value) in enumerate(ordered):
        if index >= MAX_SHAPE_FIELDS:
            labels.append("field:opaque:set")
            break
        labels.append(_field_label(key, value))
    return _bounded_shape(labels)


def _field_label(key: object, value: object) -> str:
    if isinstance(value, Mapping) or (isinstance(value, (list, tuple, set, frozenset)) and not value):
        return "field:empty:empty" if not value else "field:collection:set"
    if value is None:
        return "field:empty:empty"
    if isinstance(value, bool):
        return f"field:vocabulary:{'true' if value else 'false'}"
    if isinstance(value, (int, float)):
        return "field:scalar:set"
    name = str(key).casefold().replace("-", "_")
    text = sanitize_text(str(value), keep_newlines=False).strip()
    if not text:
        return "field:empty:empty"
    if _SECRET_KEY_RE.search(name):
        return "field:secret:set"
    if _COMMAND_KEY_RE.search(name):
        return "field:command:set"
    if _PATH_KEY_RE.search(name) or _PATH_SHAPE_RE.match(text):
        return "field:path:set"
    if _URL_KEY_RE.search(name) or _URL_SHAPE_RE.match(text):
        return "field:url:set"
    if _CONTENT_KEY_RE.search(name) or len(text) > 80 or "\n" in str(value):
        return "field:content:set"
    if text.casefold() in _VOCABULARY_VALUES:
        return f"field:vocabulary:{text.casefold()}"
    if _ID_KEY_RE.search(name) or (_ID_SHAPE_RE.fullmatch(text) and not _looks_worded(text)):
        return "field:identifier:set"
    return "field:opaque:set"


def _looks_worded(text: str) -> bool:
    return " " in text


def _bounded_shape(labels: Sequence[str]) -> list[str]:
    """Every distinct label survives the budget; only repeat counts are trimmed."""
    counts: dict[str, int] = {}
    for label in labels:
        text = str(label)[:MAX_LABEL_CHARS]
        if _label_is_known(text):
            counts[text] = counts.get(text, 0) + 1
    distinct = sorted(counts)
    if len(distinct) >= MAX_CLASS_SHAPE:
        return distinct[:MAX_CLASS_SHAPE]
    kept = dict.fromkeys(distinct, 1)
    budget = MAX_CLASS_SHAPE - len(distinct)
    while budget > 0:
        spent = 0
        for label in distinct:
            if budget == 0:
                break
            if kept[label] < counts[label]:
                kept[label] += 1
                budget -= 1
                spent += 1
        if not spent:
            break
    return sorted(label for label in distinct for _ in range(kept[label]))


def _label_is_known(label: str) -> bool:
    parts = label.split(":")
    if len(parts) != 3 or parts[0] != "field" or parts[1] not in FIELD_CLASSES:
        return False
    return parts[2] in _TAIL_VALUES


_TAIL_VALUES: frozenset[str] = frozenset({"set", "empty", "true", "false"}) | _VOCABULARY_VALUES


def _clean_shape(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return _bounded_shape([str(item)[:MAX_LABEL_CHARS] for item in list(value)[:MAX_CLASS_SHAPE]])


def request_fingerprint(action: str, shape: Sequence[str]) -> str:
    """A stable handle for one request *shape*, which nothing can be read back out of.

    The digest's inputs are a tool name and closed class labels. There is no
    dictionary of caller values to hash against, so unlike a truncated sha of a
    path this cannot be inverted by brute force.
    """
    return "fp-" + canonical_digest({"action": _ref(action, 64), "shape": _clean_shape(list(shape))})[:32]


def subject_class_digest(subject: object, shape: Sequence[str]) -> str:
    """Which *class* of thing was refused, never which one.

    A subject is read for its shape and dropped. Two different paths of the same
    class share this handle, which is the point: the record groups repeat blocks
    and cannot name what it blocked.
    """
    kind = _field_label("path" if _PATH_SHAPE_RE.match(sanitize_text(str(subject or ""))) else "subject", subject)
    return "sub-" + canonical_digest({"kind": kind, "shape": _clean_shape(list(shape))})[:24]


def _reason_code(domain: str, value: object) -> str:
    """Fold a reason code into its domain's closed set."""
    code = _ref(value, MAX_REASON_CODE_CHARS).casefold()
    return code if code in REASON_CODES.get(domain, ()) else "unspecified"


def enforcement_for(source: str) -> str:
    """Only OHAR's own control points can claim it enforced a refusal."""
    return "enforced" if _ref(source, 32).casefold() in ENFORCING_SOURCES else "declared_not_enforced"


def recovery_action_for(domain: str, reason_code: str) -> str:
    """The one action a blocked decision allows, as a closed id."""
    code = _ref(reason_code, MAX_REASON_CODE_CHARS).casefold()
    if code in RECOVERY_BY_CODE:
        return RECOVERY_BY_CODE[code]
    if code.startswith(("secret_", "prohibited_", "data_class_")):
        return "allow_action"
    if code.endswith(("_count_exceeded", "_too_long", "_out_of_range")):
        return "retry_with_scope"
    return RECOVERY_BY_DOMAIN.get(_ref(domain, 32), "no_recovery_available")


def _ref(value: object, limit: int) -> str:
    return opaque_ref(value, limit=limit)


def _stamp(value: object) -> str:
    text = clean_text(value, limit=64)
    return text if stamp_to_epoch(text) is not None else ""


class BlockLog:
    """Append-only refusal records for one workspace."""

    def __init__(
        self,
        root: Path,
        workspace: Path,
        *,
        clock: Callable[[], str] = now_iso,
        journal: ActivityJournal | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.workspace = Path(workspace).expanduser().resolve(strict=False)
        self._clock = clock
        self.journal = journal
        self._store = RecordStore(self.root, BLOCK_STORE_NAME, workspace=self.workspace)

    @property
    def path(self) -> Path:
        return self._store.path()

    def record(
        self,
        *,
        source: str,
        reason_domain: str,
        reason_code: str,
        action: str,
        request: Mapping[str, Any] | None = None,
        subject: str = "",
        outcome: str = "blocked",
        session_id: str = "",
        goal_id: str = "",
        gate_id: str = "",
    ) -> BlockedWork:
        """Mint one refusal record. `request` is read for its shape and dropped."""
        domain = _ref(reason_domain, 32)
        if domain not in REASON_DOMAINS:
            raise WorkflowError(f"reason_domain is unsupported: {clean_text(reason_domain, limit=40)!r}")
        state = _ref(outcome, 24)
        if state not in DECISION_OUTCOMES:
            raise WorkflowError(f"outcome is unsupported: {clean_text(outcome, limit=40)!r}")
        name = _ref(action, 64)
        code = _reason_code(domain, reason_code)
        record_source = _ref(source, 32) or "report"
        shape = request_class_shape(request)
        fingerprint = request_fingerprint(name, shape)
        link = _ref(gate_id, 64) or self._linked_gate(name)
        blocked = BlockedWork(
            id=new_record_id("block"),
            at=_stamp(self._clock()) or now_iso(),
            decision_id="dec-" + short_digest(
                {"source": record_source, "action": name, "fingerprint": fingerprint}, 16
            ),
            source=record_source,
            reason_domain=domain,
            reason_code=code,
            action=name,
            subject_digest=subject_class_digest(subject, shape),
            class_shape=shape,
            fingerprint=fingerprint,
            outcome=state,
            work_claim=OUTCOME_WORK_CLAIMS[state],
            enforcement=enforcement_for(record_source),
            recovery_action=recovery_action_for(domain, code),
            session_id=_ref(session_id, 64),
            goal_id=_ref(goal_id, 64),
            gate_id=link,
        )
        payload = blocked.to_dict()
        errors = validate_vocabularies(
            payload,
            (
                ("reason_domain", REASON_DOMAINS),
                ("outcome", DECISION_OUTCOMES),
                ("work_claim", WORK_CLAIMS),
                ("enforcement", ENFORCEMENT),
                ("recovery_action", RECOVERY_ACTIONS),
            ),
        )
        if errors:
            raise WorkflowError("; ".join(errors))
        self._store.append(payload)
        return blocked

    def _linked_gate(self, action: str) -> str:
        """The gate the journal observed for this action, when the caller named none."""
        if self.journal is None or not action:
            return ""
        try:
            events = self.journal.recent(event="gate_opened", limit=20)
        except Exception:  # a correlation must never break the record it decorates
            return ""
        for item in reversed(events):
            detail = item.detail if isinstance(item.detail, Mapping) else {}
            if clean_text(detail.get("action"), limit=64) != action:
                continue
            return _ref(detail.get("record") or item.subject, 64)
        return ""

    def from_tool_result(
        self,
        name: str,
        args: Mapping[str, Any],
        result: ToolResult,
        *,
        session_id: str = "",
        goal_id: str = "",
    ) -> BlockedWork | None:
        """Record a refusal, and only a refusal: denied / read-only / escape / gate-open."""
        refusal = classify_refusal(result)
        if refusal is None:
            return None
        domain, code = refusal
        return self.record(
            source="tool_gate",
            reason_domain=domain,
            reason_code=code,
            action=name,
            request=args,
            subject=name,
            outcome="blocked",
            session_id=session_id,
            goal_id=goal_id,
        )

    def recent(
        self,
        *,
        session_id: str | None = None,
        reason_domain: str | None = None,
        limit: int = 20,
    ) -> list[BlockedWork]:
        scope = _ref(session_id, 64) if session_id is not None else ""
        domain = _ref(reason_domain, 32) if reason_domain is not None else ""
        records = [BlockedWork.from_record(item) for item in self._store.records(limit=MAX_SCAN_RECORDS)]
        matched = [
            item
            for item in records
            if (not scope or item.session_id == scope) and (not domain or item.reason_domain == domain)
        ]
        return list(reversed(matched))[: _clamp(limit, 1, 200)]

    def history(self, action: str, *, limit: int = 5) -> list[BlockedWork]:
        name = _ref(action, 64)
        if not name:
            return []
        matched = [item for item in self.recent(limit=MAX_SCAN_RECORDS) if item.action == name]
        return matched[: _clamp(limit, 1, 200)]

    def for_decision(self, decision_id: str) -> list[BlockedWork]:
        wanted = _ref(decision_id, 64)
        if not wanted:
            return []
        return [item for item in self.recent(limit=MAX_SCAN_RECORDS) if item.decision_id == wanted]

    def counts(self) -> dict[str, int]:
        totals: dict[str, int] = {domain: 0 for domain in REASON_DOMAINS}
        for item in self._store.records(limit=MAX_SCAN_RECORDS):
            record = BlockedWork.from_record(item)
            totals[record.reason_domain] = totals.get(record.reason_domain, 0) + 1
        totals["total"] = sum(value for key, value in totals.items() if key != "total")
        return totals

    def headers(self) -> list[str]:
        return list(BLOCK_HEADERS)

    def rows(self, records: Sequence[BlockedWork]) -> list[list[str]]:
        return [
            [
                clean_text(item.at, limit=32),
                clean_text(item.source, limit=24),
                clean_text(item.reason_domain, limit=24),
                clean_text(item.reason_code, limit=32),
                clean_text(item.action, limit=32),
                clean_text(item.outcome, limit=16),
                clean_text(item.enforcement, limit=24),
                clean_text(item.recovery_action, limit=24),
            ]
            for item in records
        ]


_REFUSAL_MARKERS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"(?i)read[- ]?only"), "read_only_policy", "read_only_mode"),
    (re.compile(r"(?i)outside (the )?workspace|escapes? (the )?workspace|symlink"),
     "workspace_escape", "outside_workspace"),
    (re.compile(r"(?i)gate|pending approval|awaiting (a )?decision|not approved"), "gate_open", "gate_pending"),
    (re.compile(r"(?i)denied|not authorized|approval required|refus"), "approval_denied", "approval_denied"),
)
_REFUSAL_META: tuple[tuple[str, str, str], ...] = (
    ("read_only", "read_only_policy", "read_only_mode"),
    ("denied", "approval_denied", "approval_denied"),
    ("gate_open", "gate_open", "gate_open"),
)


def classify_refusal(result: ToolResult) -> tuple[str, str] | None:
    """(reason_domain, reason_code) for a refusal, or None.

    Read-only prose is checked first, then the meta flags OHAR's own gates emit,
    then the remaining markers. Any other failure is a failure and is not
    recorded here: this answers "why was this refused", not "what broke".
    """
    if result is None or getattr(result, "ok", False):
        return None
    meta = result.meta if isinstance(result.meta, dict) else {}
    text = sanitize_text(str(result.content or ""), keep_newlines=False)[:512]
    # A read-only refusal is emitted with the same `denied` flag as an operator
    # refusal, so the mode is decided before the flag: "policy stopped this" and
    # "a person said no" are different answers to "why was this blocked?".
    if re.search(r"(?i)read[- ]?only", text):
        return "read_only_policy", "read_only_mode"
    for key, domain, code in _REFUSAL_META:
        if meta.get(key) is True:
            return domain, code
    for pattern, domain, code in _REFUSAL_MARKERS:
        if pattern.search(text):
            return domain, code
    if meta.get("blocked") is True:
        return "read_only_policy", "policy_blocked"
    return None


@dataclass(slots=True)
class RecoveryAnchor:
    """What a baseline was, and the bounded recipe a person runs to return to it."""

    id: str
    at: str
    workspace_digest: str
    status: str
    base_revision: str
    dirty_state: dict[str, Any] = field(default_factory=dict)
    recipe: list[str] = field(default_factory=list)
    recoverable: bool = False
    claim_boundary: str = ANCHOR_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "id": self.id,
            "at": self.at,
            "workspace_digest": self.workspace_digest,
            "status": self.status,
            "base_revision": self.base_revision,
            "dirty_state": dict(self.dirty_state),
            "recipe": list(self.recipe),
            "recoverable": bool(self.recoverable),
            "privacy": "metadata_only",
            "claim_boundary": self.claim_boundary,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> "RecoveryAnchor":
        """A record that claims recoverability it cannot support is repaired, not trusted."""
        record = data if isinstance(data, Mapping) else {}
        status = _ref(record.get("status"), 32)
        if status not in _ANCHOR_STATUSES:
            status = "prepared"
        # A prepared anchor is structurally incapable of describing a recovery,
        # so both fields are cleared rather than left to whoever wrote the file.
        revision = _ref(record.get("base_revision"), 64) if status == _RECOVERABLE_STATUS else ""
        recipe = _recipe(record.get("recipe")) if status == _RECOVERABLE_STATUS else []
        dirty = record.get("dirty_state")
        return cls(
            id=_ref(record.get("id"), 64) or new_record_id("anchor"),
            at=_stamp(record.get("at")),
            workspace_digest=_ref(record.get("workspace_digest"), 80),
            status=status,
            base_revision=revision,
            dirty_state=_clean_dirty_state(dirty),
            recipe=recipe,
            recoverable=status == _RECOVERABLE_STATUS and bool(revision) and bool(recipe),
            claim_boundary=clean_text(record.get("claim_boundary"), limit=_CLAIM_CHARS, single_line=False)
            or ANCHOR_CLAIM_BOUNDARY,
        )


def _clean_dirty_state(value: object) -> dict[str, Any]:
    """Keep a stored dirty state inside what this module could actually observe."""
    if not isinstance(value, Mapping):
        return not_observed_dirty_state()
    state = _ref(value.get("state"), 32)
    if state not in _DIRTY_STATES:
        return not_observed_dirty_state()
    paths = [clean_text(item, limit=200) for item in list(value.get("paths") or ()) if isinstance(item, str)]
    return {
        "state": state,
        "basis": _ref(value.get("basis"), 32) or "none",
        "observed": bool(value.get("observed")) and state != "not_observed",
        "paths": paths[:MAX_DIRTY_PATHS],
        "paths_truncated": bool(value.get("paths_truncated")) or len(paths) > MAX_DIRTY_PATHS,
        "scanned": _clamp(value.get("scanned"), 0, MAX_DIRTY_SCAN),
        "digest": _ref(value.get("digest"), 80),
    }


def _recipe(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    steps = [clean_text(step, limit=MAX_NOTE_CHARS) for step in list(value)[:MAX_RECIPE_STEPS]]
    return [step for step in steps if step]


def workspace_digest_of(workspace: Path) -> str:
    """A comparison handle for one workspace: normalized, filesystem-free.

    The path is never stored. `guidance` needs one thing from it — whether the
    anchor and the question are about the same checkout — and a digest answers
    that without naming the directory.
    """
    text = os.path.normcase(os.path.normpath(os.path.expanduser(str(Path(workspace)))))
    return "ws-" + short_digest(text, 16)


def not_observed_dirty_state() -> dict[str, Any]:
    """The dirty state of a workspace nobody looked at — never `clean`."""
    return {
        "state": "not_observed",
        "basis": "none",
        "observed": False,
        "paths": [],
        "paths_truncated": False,
        "scanned": 0,
        "digest": "",
    }


class RecoveryAnchors:
    """Baseline anchors taken from what the filesystem already says. No command runs."""

    def __init__(self, root: Path, clock: Callable[[], str] = now_iso) -> None:
        self.root = Path(root).expanduser()
        self._clock = clock
        self._store = RecordStore(self.root, ANCHOR_STORE_NAME)

    @property
    def path(self) -> Path:
        return self._store.path()

    def take(self, workspace: Path, *, action: str = "", note: str = "") -> RecoveryAnchor:
        """Record the baseline this workspace is actually at right now."""
        target = Path(workspace).expanduser()
        revision, status = _read_base_revision(target)
        dirty = _observe_dirty(target)
        recipe = _build_recipe(base_revision=revision, dirty_state=dirty, action=action, note=note)
        if status != _RECOVERABLE_STATUS:
            revision, recipe = "", []
        anchor = RecoveryAnchor(
            id=new_record_id("anchor"),
            at=_stamp(self._clock()) or now_iso(),
            workspace_digest=workspace_digest_of(target),
            status=status,
            base_revision=revision,
            dirty_state=dirty,
            recipe=recipe,
            recoverable=status == _RECOVERABLE_STATUS and bool(revision) and bool(recipe),
        )
        self._store.append(anchor.to_dict())
        return anchor

    def latest(self, workspace: Path) -> RecoveryAnchor | None:
        digest = workspace_digest_of(workspace)
        for record in reversed(self._store.records(limit=MAX_SCAN_RECORDS)):
            anchor = RecoveryAnchor.from_record(record)
            if anchor.workspace_digest == digest:
                return anchor
        return None

    def get(self, anchor_id: str) -> RecoveryAnchor | None:
        wanted = _ref(anchor_id, 64)
        if not wanted:
            return None
        record = self._store.latest("id", wanted)
        return RecoveryAnchor.from_record(record) if record else None

    def guidance(self, anchor_id: str, *, workspace: Path) -> list[str]:
        """The written recipe, or the reason there is none. Nothing is run.

        A baseline from another checkout cannot describe this one, and a prepared
        anchor never had a baseline observed, so both are refused outright rather
        than answered with something plausible.
        """
        anchor = self.get(anchor_id)
        if anchor is None:
            return ["No recovery anchor with that id is recorded, so there is nothing to return to."]
        if anchor.workspace_digest != workspace_digest_of(workspace):
            return [
                "This anchor was taken in a different workspace. Its baseline cannot describe this "
                "one, so no recovery steps are offered."
            ]
        if anchor.status != _RECOVERABLE_STATUS or not anchor.recoverable:
            return [
                "This anchor was prepared but no baseline revision was ever observed, so it cannot "
                "say what to return to. A prepared anchor is not a recoverable baseline."
            ]
        return list(anchor.recipe)


def _read_base_revision(workspace: Path) -> tuple[str, str]:
    """The commit this checkout is on, read from `.git` as it already stands."""
    head = workspace / ".git" / "HEAD"
    try:
        raw = head.read_text(encoding="utf-8", errors="replace")[:256]
    except (OSError, ValueError):
        return "", "prepared"
    text = raw.strip()
    revision = text
    if text.startswith("ref:"):
        name = text[4:].strip()
        revision = _read_ref(workspace, name) or _read_packed_ref(workspace, name)
    revision = clean_text(revision, limit=64).lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", revision):
        return "", "prepared"
    return revision, _RECOVERABLE_STATUS


def _read_ref(workspace: Path, name: str) -> str:
    if not name or name.startswith(("/", "..")) or "\x00" in name:
        return ""
    try:
        return (workspace / ".git" / name).read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return ""


def _read_packed_ref(workspace: Path, name: str) -> str:
    try:
        raw = (workspace / ".git" / "packed-refs").read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    for line in raw.splitlines()[:2_000]:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == name:
            return parts[0].strip()
    return ""


def _observe_dirty(workspace: Path) -> dict[str, Any]:
    """Files touched since the index was last written, bounded and never total.

    This is an mtime observation, not a git status: it can only say a file was
    written after the index. When there is no index to compare against the
    answer is `not_observed`, which is deliberately not `clean`.
    """
    try:
        marker = (workspace / ".git" / "index").stat().st_mtime
    except (OSError, ValueError):
        return not_observed_dirty_state()
    paths: list[str] = []
    scanned = 0
    truncated = False
    for current, dirnames, filenames in os.walk(workspace):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIPPED_DIRS and not name.startswith("."))
        for name in sorted(filenames):
            scanned += 1
            if scanned > MAX_DIRTY_SCAN:
                truncated = True
                break
            candidate = Path(current) / name
            try:
                if candidate.stat().st_mtime <= marker:
                    continue
            except OSError:
                continue
            if len(paths) >= MAX_DIRTY_PATHS:
                truncated = True
                break
            try:
                rel = candidate.relative_to(workspace).as_posix()
            except ValueError:
                continue
            paths.append(clean_text(rel, limit=200))
        if truncated:
            break
    state = "candidates_found" if paths else "clean_by_mtime"
    return {
        "state": state,
        "basis": "git_index_mtime",
        "observed": True,
        "paths": paths,
        "paths_truncated": truncated or scanned > MAX_DIRTY_SCAN,
        "scanned": min(scanned, MAX_DIRTY_SCAN),
        "digest": "dirty-" + short_digest(sorted(paths), 16),
    }


def _build_recipe(
    *, base_revision: str, dirty_state: Mapping[str, Any], action: str, note: str
) -> list[str]:
    """The written steps, derived from the anchor's own facts. Empty without a baseline."""
    if not base_revision:
        return []
    state = str(dirty_state.get("state", "not_observed"))
    steps = ["Confirm you are in the workspace this anchor names; guidance is refused for any other."]
    named_action = clean_text(action, limit=MAX_NOTE_CHARS)
    if named_action:
        steps.append(f"Work recorded against this anchor: {named_action}")
    named_note = clean_text(note, limit=MAX_NOTE_CHARS)
    if named_note:
        steps.append(f"Note taken with the anchor: {named_note}")
    if state == "candidates_found":
        steps.append(
            "Uncommitted work was present when this baseline was recorded, so save it before undoing "
            "anything: git stash push --include-untracked"
        )
    elif state == "clean_by_mtime":
        steps.append("Nothing was written after the index was last updated, so there is likely nothing to save.")
    else:
        steps.append(
            "No dirty-state observation was recorded, so assume uncommitted work may be present and save "
            "it first: git stash push --include-untracked"
        )
    steps.append(f"See what changed since the baseline: git diff {base_revision}")
    steps.append(f"Return tracked files to the baseline: git restore --source {base_revision} --staged --worktree .")
    steps.append("Untracked files are not part of the baseline, so the stash above is the only copy of them.")
    steps.append("OHAR runs none of these commands, performs no rollback, and deletes nothing.")
    return _bounded_recipe(steps, base_revision=base_revision)


def _bounded_recipe(steps: Sequence[str], *, base_revision: str) -> list[str]:
    """Bound the recipe without ever dropping the line that names the baseline."""
    bounded = [clean_text(step, limit=MAX_NOTE_CHARS) for step in steps if clean_text(step, limit=MAX_NOTE_CHARS)]
    if len(bounded) <= MAX_RECIPE_STEPS:
        return bounded
    keep = [
        step
        for step in bounded
        if base_revision in step or step.startswith(("Confirm", "OHAR runs none"))
    ]
    rest = [step for step in bounded if step not in keep]
    trimmed = []
    for step in keep + rest:
        if len(trimmed) >= MAX_RECIPE_STEPS:
            break
        trimmed.append(step)
    return trimmed


_VIEWS: tuple[str, ...] = ("why", "last", "history")


def make_block_tool(log: BlockLog) -> Tool:
    """`blocks` — read-only answers to why work stopped and what would undo it."""

    def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        _ = ctx
        view = clean_text(args.get("view", "why"), limit=16).casefold() or "why"
        if view not in _VIEWS:
            return ToolResult.failure(f"blocks: view must be one of {', '.join(_VIEWS)}.")
        limit = _clamp(args.get("limit"), 10 if view != "last" else 1, 50)
        try:
            if view == "why":
                domain = args.get("domain")
                records = log.recent(
                    reason_domain=clean_text(domain, limit=32) if isinstance(domain, str) else None,
                    limit=limit,
                )
                counts = log.counts()
                head = "Nothing has been blocked in this workspace."
                if not records:
                    return ToolResult.success(head, count=0)
                lines = [
                    f"Blocked work {_SEP} {counts.get('total', 0)} record(s) "
                    f"{_SEP} recovery per refusal is stated below",
                ]
                for item in records:
                    lines.append(item.render())
                    lines.append(f"   request shape: {', '.join(item.class_shape) or '(none)'}")
                    lines.append(f"   fingerprint: {item.fingerprint}")
                return ToolResult.success("\n".join(lines), count=len(records))
            if view == "last":
                action = args.get("action")
                records = (
                    log.history(clean_text(action, limit=64), limit=1)
                    if isinstance(action, str) and action
                    else log.recent(limit=1)
                )
                if not records:
                    return ToolResult.success("Nothing has been blocked in this workspace.", count=0)
                item = records[0]
                same = log.for_decision(item.decision_id)
                lines = [
                    item.render(),
                    f"decision_id: {item.decision_id} ({len(same)} record(s) on this decision question)",
                    f"gate: {item.gate_id or '(none recorded)'}",
                    f"session: {item.session_id or '(none)'} · goal: {item.goal_id or '(none)'}",
                    f"request shape: {', '.join(item.class_shape) or '(none)'}",
                    f"subject class: {item.subject_digest or '(none)'}",
                    f"claim boundary: {item.work_claim}",
                    RECOVERY_TEXT.get(item.recovery_action, "Recovery: " + item.recovery_action),
                ]
                return ToolResult.success("\n".join(lines), count=1, decision_id=item.decision_id)
            action = args.get("action")
            if not isinstance(action, str) or not action.strip():
                return ToolResult.failure("blocks: view 'history' needs argument 'action'.")
            records = log.history(clean_text(action, limit=64), limit=limit)
            if not records:
                return ToolResult.success(f"No block history for '{clean_text(action, limit=64)}'.", count=0)
            lines = [f"History for {records[0].action} ({len(records)} most recent):"]
            lines.extend(f"  {item.render()}" for item in records)
            return ToolResult.success("\n".join(lines), count=len(records))
        except WorkflowError as exc:
            return ToolResult.failure(f"blocks: {exc}")

    return Tool(
        name="blocks",
        description=(
            "Read why work was blocked and what recovery each block allows. Views: 'why' lists recent "
            "blocks, 'last' explains the most recent one, 'history' takes an action name. Records are "
            "metadata only: no request content is recoverable from them."
        ),
        parameters={
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": list(_VIEWS), "description": "Which view to render."},
                "action": {"type": "string", "description": "Action name for 'history' or 'last'."},
                "domain": {"type": "string", "enum": list(REASON_DOMAINS), "description": "Filter by reason domain."},
                "limit": {"type": "integer", "description": "Maximum records (1-50).", "minimum": 1, "maximum": 50},
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=handler,
        read_only=True,
        destructive=False,
    )


_SEP = "·"

RECOVERY_TEXT: dict[str, str] = {
    "answer_gate": "Recovery: answer the decision gate this refusal names; nothing else will release it.",
    "allow_action": "Recovery: allow the action once, then ask for it again.",
    "leave_read_only": "Recovery: the workspace is read-only by policy; the work needs it turned off first.",
    "retry_with_scope": "Recovery: restate the request inside the scope that is actually permitted.",
    "run_recovery_recipe": "Recovery: a recovery anchor holds the written steps back to the baseline.",
    "no_recovery_available": "Recovery: none is recorded for this decision.",
}

def _clamp(value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return high
    return max(low, min(high, value))


