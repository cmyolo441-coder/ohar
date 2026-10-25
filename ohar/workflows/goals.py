"""Durable, multi-session goal ledger: objectives, acceptance criteria, blockers, gates.

PlanState is per-turn and memory is unstructured prose, so nothing in OHAR could
hold "what this work is for and what would prove it is finished" across sessions.
A goal record does: closed acceptance criteria, append-only checkpoints, active
blockers, terminal statuses, revision-guarded mutation, and a completion gate that
refuses "done" while a required criterion has no evidence or a blocker is open.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..terminal import clip, format_duration, sanitize_text
from ..types import ApprovalRequest, Tool, ToolContext, ToolResult
from .journal import ActivityJournal
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
    "GOAL_STATUSES",
    "TERMINAL_GOAL_STATUSES",
    "CHECKPOINT_STATUSES",
    "CRITERION_STATUSES",
    "BLOCKER_STATUSES",
    "FAILURE_REASON_CODES",
    "NEXT_ACTIONS",
    "MAX_CRITERIA",
    "MAX_CHECKPOINTS",
    "MAX_SUMMARY_CHARS",
    "Criterion",
    "Checkpoint",
    "Blocker",
    "Goal",
    "CompletionGate",
    "GoalJourney",
    "GoalLedger",
    "criterion_id",
    "make_goal_tools",
]

GOAL_STATUSES: frozenset[str] = frozenset({"active", "blocked", "complete", "cancelled", "failed"})
# `blocked` is deliberately absent: a blocker is recoverable, and clearing one is
# exactly what later checkpoints are for. `failed` is a negative but conclusive
# verdict on the objective itself, so it is terminal alongside complete/cancelled.
TERMINAL_GOAL_STATUSES: tuple[str, ...] = ("complete", "cancelled", "failed")
CHECKPOINT_STATUSES: tuple[str, ...] = ("pending", "in_progress", "done", "blocked", "failed")
CRITERION_STATUSES: tuple[str, ...] = ("pending", "satisfied")
BLOCKER_STATUSES: tuple[str, ...] = ("active", "resolved")
FAILURE_REASON_CODES: tuple[str, ...] = (
    "target_absent",
    "refused_by_policy",
    "criteria_infeasible",
    "capability_missing",
)
NEXT_ACTIONS: tuple[str, ...] = (
    "record_checkpoint",
    "satisfy_criterion",
    "resolve_blocker",
    "answer_gate",
    "verify_unit",
    "show_status",
    "mark_complete",
    "mark_failed",
)
MAX_CRITERIA = 8
MAX_CHECKPOINTS = 50
MAX_SUMMARY_CHARS = 240

GOAL_SCHEMA = "goal/v1"

_MAX_OBJECTIVE_CHARS = 2_000
_MAX_REFS = 12
_JOURNEY_TAIL = 5
_RENDER_TAIL = 5
# The one status a verification unit may hold and still count as evidence. Read
# from plain mappings rather than imported so goals.py never depends on
# verification.py, which itself depends on this module.
_UNIT_VERIFIED_STATUS = "observed_passed"
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CLAIM_BOUNDARY = (
    "A goal card is recorded ledger state: satisfied criteria were marked, not verified, "
    "and nothing here is execution, review, or test evidence."
)


@dataclass(slots=True)
class Criterion:
    id: str
    summary: str
    required: bool = True
    status: str = "pending"
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "summary": self.summary,
            "required": self.required,
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(slots=True)
class Checkpoint:
    id: str
    at: str
    status: str
    summary: str
    criteria_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "at": self.at,
            "status": self.status,
            "summary": self.summary,
            "criteria_refs": list(self.criteria_refs),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(slots=True)
class Blocker:
    id: str
    at: str
    status: str
    summary: str
    evidence_refs: list[str] = field(default_factory=list)
    resolution: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "at": self.at,
            "status": self.status,
            "summary": self.summary,
            "evidence_refs": list(self.evidence_refs),
            "resolution": self.resolution,
        }


@dataclass(slots=True)
class Goal:
    id: str
    created_at: str
    updated_at: str
    revision: int
    status: str
    objective: str
    criteria: list[Criterion] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)
    next_action: str = "record_checkpoint"
    session_id: str = ""
    failure_reason_code: str = ""
    failure_summary: str = ""
    resolution_summary: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_GOAL_STATUSES

    def active_blockers(self) -> list[Blocker]:
        return [item for item in self.blockers if item.status == "active"]

    def missing_criteria(self) -> list[Criterion]:
        return [
            item
            for item in self.criteria
            if item.required and (item.status != "satisfied" or not item.evidence_refs)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision": self.revision,
            "status": self.status,
            "objective": self.objective,
            "criteria": [item.to_dict() for item in self.criteria],
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "blockers": [item.to_dict() for item in self.blockers],
            "next_action": self.next_action,
            "session_id": self.session_id,
            "failure_reason_code": self.failure_reason_code,
            "failure_summary": self.failure_summary,
            "resolution_summary": self.resolution_summary,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> "Goal":
        if not isinstance(data, Mapping):
            raise WorkflowError("goal record must be a mapping")
        errors = validate_vocabularies(
            data,
            (
                ("status", tuple(sorted(GOAL_STATUSES))),
                ("next_action", NEXT_ACTIONS),
            ),
        )
        if errors:
            raise WorkflowError(f"goal record is unusable: {errors[0]}")
        return cls(
            id=_ref(data.get("goal_id") or data.get("id") or "", 128),
            created_at=clean_text(data.get("created_at", ""), limit=40),
            updated_at=clean_text(data.get("updated_at", ""), limit=40),
            revision=_revision(data.get("revision")),
            status=str(data.get("status", "")),
            objective=clean_text(data.get("objective", ""), limit=_MAX_OBJECTIVE_CHARS, single_line=False),
            criteria=[_criterion(item) for item in _mappings(data, "criteria")][:MAX_CRITERIA],
            checkpoints=[_checkpoint(item) for item in _mappings(data, "checkpoints")][-MAX_CHECKPOINTS:],
            blockers=[_blocker(item) for item in _mappings(data, "blockers")][-MAX_CHECKPOINTS:],
            next_action=clean_text(data.get("next_action", ""), limit=64) or "record_checkpoint",
            session_id=_ref(data.get("session_id", ""), 128),
            failure_reason_code=clean_text(data.get("failure_reason_code", ""), limit=64),
            failure_summary=clean_text(data.get("failure_summary", ""), limit=MAX_SUMMARY_CHARS),
            resolution_summary=clean_text(data.get("resolution_summary", ""), limit=MAX_SUMMARY_CHARS),
        )


@dataclass(slots=True)
class CompletionGate:
    goal_id: str
    ready: bool
    status: str
    missing_criteria: list[dict[str, str]] = field(default_factory=list)
    active_blockers: list[dict[str, str]] = field(default_factory=list)
    unverified_units: list[dict[str, str]] = field(default_factory=list)
    next_action: str = "show_status"
    summary: str = ""


@dataclass(slots=True)
class GoalJourney:
    goal_id: str
    stage: str
    freshness: str
    criteria_satisfied: int
    criteria_total: int
    open_gates: list[str] = field(default_factory=list)
    last_checkpoints: list[Checkpoint] = field(default_factory=list)
    next_action: str = "record_checkpoint"


def criterion_id(summary: str, index: int) -> str:
    """Content-bound slot id, so a resumed session re-derives the same ref."""
    slot = max(1, int(index))
    return f"AC{slot:03d}-{short_digest(clean_text(summary, limit=MAX_SUMMARY_CHARS), 6)}"


def _revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _ref(value: object, length: int = 128) -> str:
    return opaque_ref(value, limit=length)


def _refs(values: Sequence[object] | None, *, limit: int = _MAX_REFS) -> list[str]:
    items: Sequence[object] = (
        values if isinstance(values, (list, tuple)) else [values] if isinstance(values, str) else ()
    )
    out: list[str] = []
    seen: set[str] = set()
    for value in items:
        ref = _ref(value, 160)
        if ref and ref not in seen:
            seen.add(ref)
            out.append(ref)
            if len(out) == limit:
                break
    return out


def _mappings(data: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    raw = data.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _criterion(data: Mapping[str, Any]) -> Criterion:
    summary = clean_text(data.get("summary", ""), limit=MAX_SUMMARY_CHARS)
    errors = validate_vocabularies(data, (("status", CRITERION_STATUSES),))
    if errors:
        raise WorkflowError(f"criterion record is unusable: {errors[0]}")
    return Criterion(
        id=_ref(data.get("id", ""), 64),
        summary=summary,
        required=bool(data.get("required", True)),
        status=str(data.get("status", "pending")),
        evidence_refs=_refs(data.get("evidence_refs", ())),
    )


def _checkpoint(data: Mapping[str, Any]) -> Checkpoint:
    errors = validate_vocabularies(data, (("status", CHECKPOINT_STATUSES),))
    if errors:
        raise WorkflowError(f"checkpoint record is unusable: {errors[0]}")
    return Checkpoint(
        id=_ref(data.get("id", ""), 64),
        at=clean_text(data.get("at", ""), limit=40),
        status=str(data.get("status", "")),
        summary=clean_text(data.get("summary", ""), limit=MAX_SUMMARY_CHARS),
        criteria_refs=_refs(data.get("criteria_refs"), limit=MAX_CRITERIA),
        evidence_refs=_refs(data.get("evidence_refs")),
    )


def _blocker(data: Mapping[str, Any]) -> Blocker:
    errors = validate_vocabularies(data, (("status", BLOCKER_STATUSES),))
    if errors:
        raise WorkflowError(f"blocker record is unusable: {errors[0]}")
    return Blocker(
        id=_ref(data.get("id", ""), 64),
        at=clean_text(data.get("at", ""), limit=40),
        status=str(data.get("status", "")),
        summary=clean_text(data.get("summary", ""), limit=MAX_SUMMARY_CHARS),
        evidence_refs=_refs(data.get("evidence_refs", ())),
        resolution=clean_text(data.get("resolution", ""), limit=MAX_SUMMARY_CHARS),
    )


def _mutation_item_id(mutation_id: str, prefix: str) -> str:
    """Derive the item id a retry owns, so a replayed mutation cannot append twice."""
    item = clean_text(mutation_id, limit=128, single_line=True)
    if not item:
        return new_record_id(prefix)
    if _ID_RE.fullmatch(item) and item not in {".", ".."}:
        return item
    return f"{prefix}-{short_digest(item, 24)}"


def _display(value: object, width: int = 72) -> str:
    """Records on disk are untrusted input: neutralise control characters at render."""
    return clip(sanitize_text(str(value if value is not None else "")).replace("\n", " "), width)


def _next_action(goal: Goal) -> str:
    if goal.terminal:
        return "show_status"
    if goal.status == "blocked" or goal.active_blockers():
        return "resolve_blocker"
    if goal.missing_criteria():
        return "record_checkpoint" if not goal.checkpoints else "satisfy_criterion"
    return "mark_complete"


def _journey_stage(goal: Goal, gate: CompletionGate, open_gates: Sequence[str]) -> str:
    if goal.status == "cancelled":
        return "cancelled"
    if goal.status == "complete":
        # A ledger that says complete while a required gate still lacks evidence
        # reads as blocked: reporting it as verified is the overclaim here.
        return "verified_complete" if gate.ready else "blocked"
    if goal.status == "failed" or goal.status == "blocked" or goal.active_blockers():
        return "blocked"
    if goal.checkpoints:
        return "activity"
    if open_gates:
        return "preparation"
    return "intent"


class GoalLedger:
    """Append-only goal records, guarded by revision and by a completion gate."""

    def __init__(
        self,
        root: Path,
        workspace: Path,
        *,
        clock: Callable[[], str] = now_iso,
        journal: ActivityJournal | None = None,
    ) -> None:
        self.root = Path(root)
        self.workspace = Path(workspace)
        self._clock = clock
        self._journal = journal
        self._store = RecordStore(self.root, "goals", workspace=self.workspace)
        self._workspace_key = workspace_key(self.workspace)

    # -- write ---------------------------------------------------------------

    def create(
        self,
        objective: str,
        criteria: Sequence[str | Mapping[str, Any]],
        *,
        goal_id: str | None = None,
        session_id: str = "",
    ) -> Goal:
        text = clean_text(objective, limit=_MAX_OBJECTIVE_CHARS, single_line=False)
        if not text:
            raise WorkflowError("a goal needs a non-empty objective")
        items = self._criteria(criteria)
        ident = self._goal_id(goal_id)
        existing = self._store.latest("goal_id", ident)
        if existing is not None:
            # Resume path: the same explicit id returns the ledger already on file.
            return Goal.from_record(existing)
        stamp = self._clock()
        goal = Goal(
            id=ident,
            created_at=stamp,
            updated_at=stamp,
            revision=0,
            status="active",
            objective=text,
            criteria=items,
            session_id=_ref(session_id, 128),
        )
        goal.next_action = _next_action(goal)
        record, _replayed = self._store.transact(
            "goal_id",
            ident,
            lambda current: self._write(current, goal),
        )
        self._journal_goal(record, ok=True, session_id=goal.session_id, detail="created")
        return Goal.from_record(record)

    def checkpoint(
        self,
        goal_id: str,
        summary: str,
        *,
        status: str = "done",
        criteria_refs: Sequence[str] = (),
        evidence_refs: Sequence[str] = (),
        mutation_id: str = "",
        expected_revision: int | None = None,
    ) -> Goal:
        if status not in CHECKPOINT_STATUSES:
            raise WorkflowError(f"unsupported checkpoint status: {clean_text(status, limit=40)!r}")
        text = clean_text(summary, limit=MAX_SUMMARY_CHARS)
        if not text:
            raise WorkflowError("checkpoint summary is required")
        refs = list(dict.fromkeys(_ref(item, 64) for item in criteria_refs if _ref(item, 64)))
        if len(refs) > MAX_CRITERIA:
            raise WorkflowError(f"a checkpoint can name at most {MAX_CRITERIA} criteria")
        evidence = _refs(evidence_refs)
        item_id = _mutation_item_id(mutation_id, "cp")
        stamp = self._clock()

        def mutate(current: dict[str, Any] | None) -> dict[str, Any] | None:
            goal = self._require(current, goal_id, "record a checkpoint on")
            if any(item.id == item_id for item in goal.checkpoints):
                return None  # already applied: the record itself is the replay proof
            self._require_live(goal, "record a checkpoint on")
            known = {item.id for item in goal.criteria}
            unknown = [item for item in refs if item not in known]
            if unknown:
                raise WorkflowError(f"unknown acceptance criteria: {', '.join(_display(x, 40) for x in unknown)}")
            if status == "done" and refs and not evidence:
                raise WorkflowError("a done checkpoint that names criteria must cite evidence_refs")
            goal.checkpoints.append(
                Checkpoint(
                    id=item_id,
                    at=stamp,
                    status=status,
                    summary=text,
                    criteria_refs=refs,
                    evidence_refs=evidence,
                )
            )
            goal.checkpoints = goal.checkpoints[-MAX_CHECKPOINTS:]
            if status == "done":
                for criterion in goal.criteria:
                    if criterion.id in refs:
                        criterion.status = "satisfied"
                        criterion.evidence_refs = _refs(criterion.evidence_refs + evidence)
            goal.next_action = _next_action(goal)
            return self._write(current, goal)

        return self._transact(
            goal_id,
            mutate,
            expected_revision=expected_revision,
            mutation_id=clean_text(mutation_id, limit=128, single_line=True),
            mutation_digest=canonical_digest(["checkpoint", item_id, text, status, refs, evidence]),
        )

    def block(
        self,
        goal_id: str,
        summary: str,
        *,
        evidence_refs: Sequence[str] = (),
        expected_revision: int | None = None,
    ) -> Goal:
        text = clean_text(summary, limit=MAX_SUMMARY_CHARS)
        if not text:
            raise WorkflowError("blocker summary is required")
        evidence = _refs(evidence_refs)
        stamp = self._clock()

        def mutate(current: dict[str, Any] | None) -> dict[str, Any] | None:
            goal = self._require(current, goal_id, "record a blocker on")
            self._require_live(goal, "record a blocker on")
            goal.blockers.append(
                Blocker(
                    id=new_record_id("blk"),
                    at=stamp,
                    status="active",
                    summary=text,
                    evidence_refs=evidence,
                )
            )
            goal.blockers = goal.blockers[-MAX_CHECKPOINTS:]
            goal.status = "blocked"
            goal.next_action = _next_action(goal)
            return self._write(current, goal)

        return self._transact(goal_id, mutate, expected_revision=expected_revision)

    def resolve_blocker(self, goal_id: str, blocker_id: str, summary: str = "") -> Goal:
        wanted = _ref(blocker_id, 64)
        if not wanted:
            raise WorkflowError("blocker id is required")
        note = clean_text(summary, limit=MAX_SUMMARY_CHARS)
        stamp = self._clock()

        def mutate(current: dict[str, Any] | None) -> dict[str, Any] | None:
            goal = self._require(current, goal_id, "resolve a blocker on")
            self._require_live(goal, "resolve a blocker on")
            target = next((item for item in goal.blockers if item.id == wanted), None)
            if target is None:
                raise WorkflowError(f"unknown blocker: {_display(wanted, 40)}")
            if target.status != "active":
                raise WorkflowError(f"blocker {_display(wanted, 40)} is already {target.status}")
            target.status = "resolved"
            target.resolution = note
            target.evidence_refs = _refs(target.evidence_refs + [f"resolved_at:{stamp}"])
            if not goal.active_blockers() and goal.status == "blocked":
                goal.status = "active"
            goal.updated_at = stamp
            goal.next_action = _next_action(goal)
            return self._write(current, goal)

        return self._transact(goal_id, mutate)

    def complete(
        self,
        goal_id: str,
        summary: str,
        *,
        evidence_refs: Sequence[str] = (),
        require_gate: bool = True,
    ) -> Goal:
        text = clean_text(summary, limit=MAX_SUMMARY_CHARS)
        if not text:
            raise WorkflowError("completion summary is required")
        evidence = _refs(evidence_refs)
        stamp = self._clock()

        def mutate(current: dict[str, Any] | None) -> dict[str, Any] | None:
            goal = self._require(current, goal_id, "complete")
            self._require_live(goal, "complete")
            if require_gate:
                gate = self._evaluate(goal, None)
                if not gate.ready:
                    raise WorkflowError(f"completion refused: {gate.summary}")
                if not evidence:
                    raise WorkflowError("completion requires evidence_refs naming what was observed")
            goal.checkpoints.append(
                Checkpoint(
                    id=new_record_id("cp"),
                    at=stamp,
                    status="done",
                    summary=text,
                    evidence_refs=evidence,
                )
            )
            goal.checkpoints = goal.checkpoints[-MAX_CHECKPOINTS:]
            goal.status = "complete"
            goal.resolution_summary = text
            goal.updated_at = stamp
            goal.next_action = _next_action(goal)
            return self._write(current, goal)

        return self._transact(goal_id, mutate)

    def fail(self, goal_id: str, reason_code: str, summary: str) -> Goal:
        code = clean_text(reason_code, limit=64)
        if code not in FAILURE_REASON_CODES:
            raise WorkflowError(f"unsupported failure reason_code: {_display(reason_code, 40)}")
        text = clean_text(summary, limit=MAX_SUMMARY_CHARS)
        if not text:
            raise WorkflowError("failure summary is required")
        stamp = self._clock()

        def mutate(current: dict[str, Any] | None) -> dict[str, Any] | None:
            goal = self._require(current, goal_id, "fail")
            self._require_live(goal, "fail")
            goal.status = "failed"
            goal.failure_reason_code = code
            goal.failure_summary = text
            goal.updated_at = stamp
            goal.next_action = _next_action(goal)
            return self._write(current, goal)

        return self._transact(goal_id, mutate)

    def cancel(self, goal_id: str, summary: str = "") -> Goal:
        note = clean_text(summary, limit=MAX_SUMMARY_CHARS)
        stamp = self._clock()

        def mutate(current: dict[str, Any] | None) -> dict[str, Any] | None:
            goal = self._require(current, goal_id, "cancel")
            self._require_live(goal, "cancel")
            goal.status = "cancelled"
            if note:
                goal.resolution_summary = note
            goal.updated_at = stamp
            goal.next_action = _next_action(goal)
            return self._write(current, goal)

        return self._transact(goal_id, mutate)

    # -- read ----------------------------------------------------------------

    def get(self, goal_id: str) -> Goal:
        record = self._store.latest("goal_id", _ref(goal_id, 128))
        if record is None:
            raise WorkflowError(f"unknown goal: {_display(goal_id, 60)}")
        return Goal.from_record(record)

    def list(self, *, status: str | None = None, limit: int = 20) -> list[Goal]:
        if status is not None and status not in GOAL_STATUSES:
            raise WorkflowError(f"unsupported goal status: {_display(status, 40)}")
        goals = [Goal.from_record(item) for item in self._store.heads("goal_id").values()]
        if status is not None:
            goals = [item for item in goals if item.status == status]
        goals.sort(key=lambda item: (stamp_to_epoch(item.updated_at) or 0.0, item.id), reverse=True)
        return goals[: _clamp(limit, 1, MAX_CHECKPOINTS)]

    def active(self) -> list[Goal]:
        return [item for item in self.list() if not item.terminal]

    def gate(self, goal_id: str, *, units: Sequence[Mapping[str, Any]] | None = None) -> CompletionGate:
        goal = self.get(goal_id)
        return self._evaluate(goal, units)

    def journey(
        self,
        goal_id: str,
        *,
        open_gate_ids: Sequence[str] = (),
        now: float | None = None,
        stale_after_seconds: float = 86400.0,
    ) -> GoalJourney:
        goal = self.get(goal_id)
        gate = self._evaluate(goal, None)
        open_ids = list(dict.fromkeys(_ref(item, 64) for item in open_gate_ids if _ref(item, 64)))
        stage = _journey_stage(goal, gate, open_ids)
        return GoalJourney(
            goal_id=goal.id,
            stage=stage,
            freshness=self._freshness(goal, now=now, stale_after_seconds=stale_after_seconds),
            criteria_satisfied=len([item for item in goal.criteria if item.status == "satisfied"]),
            criteria_total=len(goal.criteria),
            open_gates=open_ids,
            last_checkpoints=list(goal.checkpoints[-_JOURNEY_TAIL:]),
            next_action="answer_gate" if open_ids and stage != "cancelled" else gate.next_action,
        )

    def continuation_brief(self, goal_id: str, *, limit: int = 8) -> str:
        """The deterministic brief a loop driver feeds back to the agent."""
        goal = self.get(goal_id)
        gate = self._evaluate(goal, None)
        count = _clamp(limit, 1, MAX_CHECKPOINTS)
        lines = [
            f"Goal {goal.id} is {goal.status}.",
            f"Objective: {_display(goal.objective, 200)}",
            f"Criteria: {len([c for c in goal.criteria if c.status == 'satisfied'])} of "
            f"{len(goal.criteria)} satisfied.",
        ]
        for criterion in goal.criteria:
            mark = "satisfied" if criterion.status == "satisfied" else "pending"
            evidence = f", evidence {len(criterion.evidence_refs)}" if criterion.evidence_refs else ", no evidence"
            required = "" if criterion.required else ", optional"
            lines.append(f"- {criterion.id}: {_display(criterion.summary, 90)} [{mark}{evidence}{required}]")
        for blocker in goal.active_blockers():
            lines.append(f"- blocked by: {_display(blocker.summary, 90)} [{blocker.id}]")
        if goal.checkpoints:
            lines.append(f"Last checkpoints (of {len(goal.checkpoints)}):")
            for item in goal.checkpoints[-count:]:
                observed = "observed" if item.evidence_refs else "claimed"
                lines.append(f"- {_display(item.id, 32)}: {_display(item.summary, 90)} [{item.status}, {observed}]")
        else:
            lines.append("No checkpoints recorded yet.")
        age = _elapsed_seconds(goal.updated_at, self._clock())
        if age is not None:
            lines.append(f"Last updated {format_duration(age)} ago.")
        lines.append(f"Completion gate: {gate.summary}")
        lines.append(f"Next action: {gate.next_action}")
        lines.append(_CLAIM_BOUNDARY)
        return "\n".join(lines)

    # -- render --------------------------------------------------------------

    def render(self, goal: Goal, *, gate: CompletionGate | None = None, glyphs: str = "unicode") -> str:
        marks = ("●", "○", "▲") if glyphs != "ascii" else ("*", ".", "!")
        satisfied = len([item for item in goal.criteria if item.status == "satisfied"])
        lines = [
            f"Goal {goal.id} — {goal.status}",
            f"Objective: {_display(goal.objective, 120)}",
            f"Criteria: {satisfied} of {len(goal.criteria)} satisfied",
        ]
        for criterion in goal.criteria:
            mark = marks[0] if criterion.status == "satisfied" else marks[1]
            evidence = f" evidence:{len(criterion.evidence_refs)}" if criterion.evidence_refs else " no evidence"
            flag = "" if criterion.required else " optional"
            lines.append(f"  {mark} {criterion.id} {_display(criterion.summary, 60)}{evidence}{flag}")
        blockers = goal.active_blockers()
        if blockers:
            lines.append(f"Blockers ({len(blockers)}):")
            for item in blockers:
                lines.append(f"  {marks[2]} {_display(item.summary, 80)} [{item.id}]")
        if goal.checkpoints:
            lines.append(f"Checkpoints (last {min(_RENDER_TAIL, len(goal.checkpoints))}):")
            for item in goal.checkpoints[-_RENDER_TAIL:]:
                observed = "observed" if item.evidence_refs else "claimed"
                lines.append(
                    f"  - {_display(item.id, 28)}: {_display(item.summary, 70)} — {item.status}, {observed}"
                )
        if goal.status == "failed" and goal.failure_reason_code:
            lines.append(f"Failure: {goal.failure_reason_code} — {_display(goal.failure_summary, 80)}")
        lines.append(f"Next action: {goal.next_action}")
        if gate is not None:
            lines.append(f"Completion gate: {gate.summary}")
        lines.append(_CLAIM_BOUNDARY)
        return "\n".join(lines)

    def headers(self) -> list[str]:
        return ["goal", "status", "criteria", "blockers", "next", "updated"]

    def rows(self, goals: Sequence[Goal]) -> list[list[str]]:
        out: list[list[str]] = []
        for goal in goals:
            satisfied = len([item for item in goal.criteria if item.status == "satisfied"])
            out.append(
                [
                    _display(goal.id, 28),
                    _display(goal.status, 16),
                    f"{satisfied}/{len(goal.criteria)}",
                    str(len(goal.active_blockers())),
                    _display(goal.next_action, 20),
                    _display(goal.updated_at, 24),
                ]
            )
        return out

    # -- internals -----------------------------------------------------------

    def _criteria(self, raw: Sequence[str | Mapping[str, Any]]) -> list[Criterion]:
        if isinstance(raw, (str, Mapping)):
            raw = [raw]  # type: ignore[list-item]
        items: list[Criterion] = []
        seen: set[str] = set()
        for index, entry in enumerate(raw, start=1):
            if isinstance(entry, Mapping):
                summary = clean_text(entry.get("summary", ""), limit=MAX_SUMMARY_CHARS)
                ident = _ref(entry.get("id") or criterion_id(summary, index), 64)
                required = bool(entry.get("required", True))
                status = str(entry.get("status", "pending") or "pending")
                evidence = _refs(entry.get("evidence_refs", ()))
            else:
                summary = clean_text(entry, limit=MAX_SUMMARY_CHARS)
                ident = criterion_id(summary, index)
                required, status, evidence = True, "pending", []
            if not summary:
                raise WorkflowError(f"acceptance criterion {index} needs a summary")
            if status not in CRITERION_STATUSES:
                raise WorkflowError(f"unsupported criterion status: {_display(status, 32)}")
            if ident in seen:
                raise WorkflowError(f"duplicate acceptance criterion id: {_display(ident, 40)}")
            seen.add(ident)
            items.append(
                Criterion(id=ident, summary=summary, required=required, status=status, evidence_refs=evidence)
            )
        if not items:
            raise WorkflowError("a goal needs at least one acceptance criterion")
        if len(items) > MAX_CRITERIA:
            raise WorkflowError(f"a goal carries at most {MAX_CRITERIA} acceptance criteria")
        return items

    def _goal_id(self, goal_id: str | None) -> str:
        if goal_id is None:
            return new_record_id("goal")
        ident = clean_text(goal_id, limit=128, single_line=True)
        if not _ID_RE.fullmatch(ident) or ident in {".", ".."}:
            raise WorkflowError(f"goal id is not storage-safe: {_display(goal_id, 60)}")
        return ident

    def _require(self, current: dict[str, Any] | None, goal_id: str, verb: str) -> Goal:
        if current is None:
            raise WorkflowError(f"unknown goal: {_display(goal_id, 60)}")
        stored = clean_text(current.get("workspace", ""), limit=128)
        if stored and stored != self._workspace_key:
            raise WorkflowError(f"goal {_display(goal_id, 40)} belongs to another workspace")
        goal = Goal.from_record(current)
        if not goal.id:
            raise WorkflowError(f"goal record has no id: {_display(goal_id, 40)}")
        return goal

    def _require_live(self, goal: Goal, verb: str) -> None:
        if goal.terminal:
            raise WorkflowError(f"goal {goal.id} is {goal.status}; cannot {verb} it")

    def _write(self, current: dict[str, Any] | None, goal: Goal) -> dict[str, Any]:
        record = dict(current) if current else {}
        record.update(goal.to_dict())
        record["goal_id"] = goal.id
        record["schema"] = GOAL_SCHEMA
        record["workspace"] = self._workspace_key
        record["updated_at"] = goal.updated_at or self._clock()
        record["revision"] = _revision(record.get("revision")) + 1
        return record

    def _transact(
        self,
        goal_id: str,
        mutate: Callable[[dict[str, Any] | None], dict[str, Any] | None],
        *,
        expected_revision: int | None = None,
        mutation_id: str = "",
        mutation_digest: str = "",
    ) -> Goal:
        ident = _ref(goal_id, 128)
        record, _replayed = self._store.transact(
            "goal_id",
            ident,
            mutate,
            expected_revision=expected_revision,
            mutation_id=mutation_id,
            mutation_digest=mutation_digest,
        )
        goal = Goal.from_record(record)
        self._journal_goal(
            record,
            ok=True,
            session_id=goal.session_id,
            detail=f"{goal.status} → {goal.next_action}",
        )
        return goal

    def _journal_goal(self, record: Mapping[str, Any], *, ok: bool, session_id: str, detail: str) -> None:
        if self._journal is None:
            return
        self._journal.record(
            "goal_updated",
            subject=_ref(record.get("goal_id", ""), 128),
            ok=ok,
            session_id=session_id,
            detail={"status": clean_text(record.get("status", ""), limit=32), "note": clean_text(detail, limit=96)},
        )

    def _evaluate(self, goal: Goal, units: Sequence[Mapping[str, Any]] | None) -> CompletionGate:
        missing = [
            {"id": item.id, "summary": item.summary, "status": item.status} for item in goal.missing_criteria()
        ][:MAX_CRITERIA]
        blockers = [{"id": item.id, "summary": item.summary} for item in goal.active_blockers()][:MAX_CRITERIA]
        unverified = _unverified_units(goal, units)
        gaps: list[str] = []
        if goal.status == "blocked":
            gaps.append("goal status is blocked")
        if goal.status in ("cancelled", "failed"):
            gaps.append(f"goal status is {goal.status}")
        ready = not missing and not blockers and not unverified and not gaps
        parts: list[str] = []
        if missing:
            parts.append(f"{len(missing)} required criteria have no evidence")
        if blockers:
            parts.append(f"{len(blockers)} blockers are still active")
        if unverified:
            parts.append(f"{len(unverified)} declared units are unverified")
        if gaps:
            parts.append("; ".join(gaps))
        summary = clean_text(
            "Goal is ready for completion."
            if ready
            else "; ".join(parts) + "." if parts else f"Goal is {goal.status}.",
            limit=MAX_SUMMARY_CHARS,
        )
        return CompletionGate(
            goal_id=goal.id,
            ready=ready,
            status=goal.status,
            missing_criteria=missing,
            active_blockers=blockers,
            unverified_units=unverified,
            next_action=_gate_next_action(goal, missing, blockers, unverified, gaps),
            summary=summary,
        )

    def _freshness(self, goal: Goal, *, now: float | None, stale_after_seconds: float) -> str:
        if now is None:
            return "unknown"
        stamp = stamp_to_epoch(goal.updated_at)
        if stamp is None:
            return "unknown"
        age = now - stamp
        if age < 0:
            return "unknown"  # the two clocks disagree; freshness is not provable
        return "stale" if age > max(0.0, stale_after_seconds) else "fresh"


def _gate_next_action(
    goal: Goal,
    missing: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, Any]],
    unverified: Sequence[Mapping[str, Any]],
    gaps: Sequence[str],
) -> str:
    if goal.status in ("complete", "cancelled", "failed"):
        return "show_status"
    dead_end = bool(goal.checkpoints) and all(
        item.status in ("failed", "blocked") for item in goal.checkpoints
    )
    if blockers and dead_end:
        # Nothing recorded for this goal ever moved forward: failing it is the
        # conclusive option, and only the operator can clear the blocker.
        return "mark_failed"
    if blockers or "goal status is blocked" in gaps:
        return "resolve_blocker"
    if unverified:
        return "verify_unit"
    if missing:
        # Nothing recorded yet is a different gap from activity that stopped:
        # the first needs a checkpoint, the second needs named evidence.
        return "record_checkpoint" if not goal.checkpoints else "satisfy_criterion"
    if gaps:
        return "show_status"
    return "mark_complete"


def _unverified_units(goal: Goal, units: Sequence[Mapping[str, Any]] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for unit in units or ():
        if not isinstance(unit, Mapping):
            continue
        owner = _ref(unit.get("goal_id", ""), 128)
        if owner and owner != goal.id:
            continue
        status = clean_text(unit.get("status", ""), limit=48) or "not_observed"
        if status == _UNIT_VERIFIED_STATUS:
            continue
        out.append(
            {
                "id": _ref(unit.get("id", ""), 64) or "unit",
                "summary": clean_text(unit.get("statement", ""), limit=MAX_SUMMARY_CHARS) or "unnamed unit",
                "status": status,
            }
        )
        if len(out) == MAX_CHECKPOINTS:
            break
    return out


def _elapsed_seconds(stamp: str, now: str) -> float | None:
    then, current = stamp_to_epoch(stamp), stamp_to_epoch(now)
    if then is None or current is None:
        return None
    return max(0.0, current - then)


def _clamp(value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return high
    return max(low, min(high, value))


def _approve(ctx: ToolContext, enabled: bool, action: str, summary: str, detail: str, risk: str) -> str | None:
    """Return a refusal reason, or None when the caller may proceed."""
    if ctx.read_only:
        return "Read-only mode: goal changes are disabled."
    if not enabled:
        return None
    if not ctx.request_approval(ApprovalRequest(action=action, summary=summary, detail=detail, risk=risk)):
        return "Goal change denied; nothing was recorded."
    return None


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    handler: Any,
    *,
    read_only: bool = False,
) -> Tool:
    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        handler=handler,
        read_only=read_only,
        destructive=not read_only,
    )


def _strings(args: Mapping[str, Any], key: str) -> list[str]:
    raw = args.get(key) or ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise WorkflowError(f"argument {key!r} must be a list of strings")
    return [item for item in raw if isinstance(item, str)]


def make_goal_tools(
    ledger: GoalLedger,
    *,
    journal: ActivityJournal | None = None,
    require_approval: bool = True,
    read_only: bool = False,
) -> list[Tool]:
    """Build the goal tools; refusals are journaled so readiness stays honest."""

    def refuse(goal_id: str, session_id: str, error: str) -> None:
        if journal is None:
            return
        journal.record(
            "goal_updated",
            subject=_ref(goal_id, 128),
            ok=False,
            session_id=session_id,
            detail={"error": clean_text(error, limit=160)},
        )

    def goal_create(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = args.get("objective", "")
        if not isinstance(objective, str):
            return ToolResult.failure("Argument 'objective' must be a string.")
        session = clean_text(args.get("session", ""), limit=128, single_line=True)
        if read_only:
            refuse("", session, "read-only ledger")
            return ToolResult.failure("Read-only mode: goal changes are disabled.", denied=True)
        denied = _approve(
            ctx,
            require_approval,
            "goal_create",
            f"Create a goal: {_display(objective, 100)}",
            "",
            "low",
        )
        if denied is not None:
            refuse("", session, denied)
            return ToolResult.failure(denied, denied=True)
        try:
            goal = ledger.create(objective, _strings(args, "criteria"), session_id=session)
        except (TypeError, ValueError, OSError) as exc:
            refuse("", session, str(exc))
            return ToolResult.failure(f"goal_create: {exc}")
        return ToolResult.success(
            ledger.render(goal),
            goal_id=goal.id,
            status=goal.status,
            next_action=goal.next_action,
        )

    def goal_checkpoint(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        goal_id = str(args.get("goal", "") or "")
        summary = args.get("summary", "")
        if not goal_id or not isinstance(summary, str):
            return ToolResult.failure("Arguments 'goal' and 'summary' are required strings.")
        status = str(args.get("status", "done") or "done")
        session = clean_text(args.get("session", ""), limit=128, single_line=True)
        if read_only:
            refuse(goal_id, session, "read-only ledger")
            return ToolResult.failure("Read-only mode: goal changes are disabled.", denied=True)
        denied = _approve(
            ctx,
            require_approval,
            "goal_checkpoint",
            f"Checkpoint {goal_id}: {_display(summary, 100)}",
            "",
            "low",
        )
        if denied is not None:
            refuse(goal_id, session, denied)
            return ToolResult.failure(denied, denied=True)
        try:
            goal = ledger.checkpoint(
                goal_id,
                summary,
                status=status,
                criteria_refs=_strings(args, "criteria"),
                evidence_refs=_strings(args, "evidence"),
                mutation_id=str(args.get("mutation_id", "") or ""),
            )
        except (TypeError, ValueError, OSError) as exc:
            refuse(goal_id, session, str(exc))
            return ToolResult.failure(f"goal_checkpoint: {exc}")
        gate = ledger.gate(goal.id)
        return ToolResult.success(
            ledger.render(goal, gate=gate),
            goal_id=goal.id,
            status=goal.status,
            ready=gate.ready,
            next_action=gate.next_action,
        )

    def goal_block(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        goal_id = str(args.get("goal", "") or "")
        summary = args.get("summary", "")
        if not goal_id or not isinstance(summary, str):
            return ToolResult.failure("Arguments 'goal' and 'summary' are required strings.")
        session = clean_text(args.get("session", ""), limit=128, single_line=True)
        if read_only:
            refuse(goal_id, session, "read-only ledger")
            return ToolResult.failure("Read-only mode: goal changes are disabled.", denied=True)
        denied = _approve(
            ctx,
            require_approval,
            "goal_block",
            f"Block {goal_id}: {_display(summary, 100)}",
            "",
            "medium",
        )
        if denied is not None:
            refuse(goal_id, session, denied)
            return ToolResult.failure(denied, denied=True)
        try:
            goal = ledger.block(goal_id, summary, evidence_refs=_strings(args, "evidence"))
        except (TypeError, ValueError, OSError) as exc:
            refuse(goal_id, session, str(exc))
            return ToolResult.failure(f"goal_block: {exc}")
        return ToolResult.success(
            ledger.render(goal),
            goal_id=goal.id,
            status=goal.status,
            next_action=goal.next_action,
        )

    def goal_status(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        _ = ctx
        goal_id = str(args.get("goal", "") or "")
        if not goal_id:
            goals = ledger.list(limit=_clamp(args.get("limit"), 1, MAX_CHECKPOINTS))
            if not goals:
                return ToolResult.success("No goals recorded.", count=0)
            return ToolResult.success(
                "\n".join(ledger.render(item) for item in goals),
                count=len(goals),
                goals=[item.id for item in goals],
            )
        try:
            goal = ledger.get(goal_id)
            gate = ledger.gate(goal_id)
        except (TypeError, ValueError) as exc:
            return ToolResult.failure(f"goal_status: {exc}")
        return ToolResult.success(
            ledger.render(goal, gate=gate),
            goal_id=goal.id,
            status=goal.status,
            ready=gate.ready,
            next_action=gate.next_action,
        )

    def goal_complete(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        goal_id = str(args.get("goal", "") or "")
        summary = args.get("summary", "")
        if not goal_id or not isinstance(summary, str):
            return ToolResult.failure("Arguments 'goal' and 'summary' are required strings.")
        session = clean_text(args.get("session", ""), limit=128, single_line=True)
        if read_only:
            refuse(goal_id, session, "read-only ledger")
            return ToolResult.failure("Read-only mode: goal changes are disabled.", denied=True)
        denied = _approve(
            ctx,
            require_approval,
            "goal_complete",
            f"Complete {goal_id}: {_display(summary, 100)}",
            "",
            "medium",
        )
        if denied is not None:
            refuse(goal_id, session, denied)
            return ToolResult.failure(denied, denied=True)
        try:
            goal = ledger.complete(goal_id, summary, evidence_refs=_strings(args, "evidence"))
        except (TypeError, ValueError, OSError) as exc:
            refuse(goal_id, session, str(exc))
            return ToolResult.failure(f"goal_complete: {exc}")
        return ToolResult.success(ledger.render(goal), goal_id=goal.id, status=goal.status)

    return [
        _tool(
            "goal_create",
            "Record a durable goal with bounded acceptance criteria. The goal "
            "cannot be completed until every required criterion cites evidence.",
            {
                "objective": {"type": "string", "description": "What the work is for."},
                "criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Closed acceptance criteria, 1-{MAX_CRITERIA}.",
                    "maxItems": MAX_CRITERIA,
                },
                "session": {"type": "string", "description": "Opaque session id to attribute the goal to."},
            },
            ["objective", "criteria"],
            goal_create,
            read_only=read_only,
        ),
        _tool(
            "goal_checkpoint",
            "Record one checkpoint against a goal. A done checkpoint that names "
            "criteria satisfies them only when it cites evidence refs.",
            {
                "goal": {"type": "string", "description": "Goal id."},
                "summary": {"type": "string", "description": f"What changed (max {MAX_SUMMARY_CHARS} chars)."},
                "status": {
                    "type": "string",
                    "enum": list(CHECKPOINT_STATUSES),
                    "description": "Default done.",
                },
                "criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Criterion ids this checkpoint satisfies.",
                    "maxItems": MAX_CRITERIA,
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Refs that prove it: file paths, command results, unit ids.",
                    "maxItems": _MAX_REFS,
                },
                "mutation_id": {
                    "type": "string",
                    "description": "Retry token: re-sending the same id appends nothing.",
                },
                "session": {"type": "string", "description": "Opaque session id."},
            },
            ["goal", "summary"],
            goal_checkpoint,
            read_only=read_only,
        ),
        _tool(
            "goal_block",
            "Record an active blocker on a goal. Blocking is recoverable, not terminal.",
            {
                "goal": {"type": "string", "description": "Goal id."},
                "summary": {"type": "string", "description": "What is stopping progress."},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Refs proving the obstruction.",
                    "maxItems": _MAX_REFS,
                },
                "session": {"type": "string", "description": "Opaque session id."},
            },
            ["goal", "summary"],
            goal_block,
            read_only=read_only,
        ),
        _tool(
            "goal_status",
            "Show one goal's card, or the ledger: criteria, blockers, checkpoints, "
            "and the completion gate.",
            {
                "goal": {"type": "string", "description": "Goal id; omit to list goals."},
                "limit": {
                    "type": "integer",
                    "description": "Max goals when listing.",
                    "minimum": 1,
                    "maximum": MAX_CHECKPOINTS,
                },
            },
            [],
            goal_status,
            read_only=True,
        ),
        _tool(
            "goal_complete",
            "Mark a goal complete. Refused while a required criterion lacks "
            "evidence, a blocker is active, or no final evidence is cited.",
            {
                "goal": {"type": "string", "description": "Goal id."},
                "summary": {"type": "string", "description": "The completion claim."},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Final evidence refs; at least one is required.",
                    "maxItems": _MAX_REFS,
                },
                "session": {"type": "string", "description": "Opaque session id."},
            },
            ["goal", "summary"],
            goal_complete,
            read_only=read_only,
        ),
    ]
