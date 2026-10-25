"""One OHAR chat session's wrapper lifecycle as a closed, gated vocabulary.

The journey is ``open -> plan_presented -> plan_accepted/revision_requested ->
working -> blocked/needs_review -> verified -> done|cancelled``. Illegal jumps
are refused, as is any write to a terminal state, so /resume, the briefing,
mission control and route hints all read one authoritative "where is this
task" record instead of re-reading the transcript.

The plan decision gate (accept/revise/cancel) is the same gate the original
wrapper used: a decision is only legal from a status that is actually asking
for one, and every accepted jump appends exactly one event.

Kept session-scoped on purpose. A generic goal/decision-gate engine belongs in
``ohar.workflows``, which consumes this rather than redefining it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ohar.wrapper import records

STATE_RECORD = "state.json"
MAX_GOAL_CHARS = 240
MAX_LIST = 200

RECORD_TYPE = "wrapper_session_state"
DECISIONS: tuple[str, str, str] = ("accept", "revise", "cancel")
STATUSES: tuple[str, ...] = (
    "open",
    "plan_presented",
    "plan_accepted",
    "revision_requested",
    "working",
    "blocked",
    "needs_review",
    "verified",
    "done",
    "cancelled",
)
TERMINAL_STATUSES = frozenset({"done", "cancelled"})

__all__ = [
    "DECISIONS",
    "DECISION_GATES",
    "DECISION_TARGETS",
    "MAX_GOAL_CHARS",
    "NEXT_ACTIONS",
    "RECORD_TYPE",
    "STATE_RECORD",
    "STATUSES",
    "TERMINAL_STATUSES",
    "TRANSITIONS",
    "SessionState",
    "SessionStateError",
    "close_state",
    "list_states",
    "load_state",
    "next_action_for",
    "note",
    "open_state",
    "record_decision",
    "save_state",
    "transition",
]

TRANSITIONS: dict[str, frozenset[str]] = {
    "open": frozenset({"plan_presented", "cancelled"}),
    "plan_presented": frozenset({"plan_accepted", "revision_requested", "cancelled"}),
    "plan_accepted": frozenset({"working", "revision_requested", "cancelled"}),
    "revision_requested": frozenset({"plan_presented", "plan_accepted", "cancelled"}),
    "working": frozenset({"blocked", "needs_review", "verified", "cancelled"}),
    "blocked": frozenset({"working", "needs_review", "cancelled"}),
    "needs_review": frozenset({"working", "verified", "done", "cancelled"}),
    "verified": frozenset({"done", "cancelled"}),
    "done": frozenset(),
    "cancelled": frozenset(),
}

DECISION_TARGETS: dict[str, str] = {
    "accept": "plan_accepted",
    "revise": "revision_requested",
    "cancel": "cancelled",
}

DECISION_GATES: dict[str, frozenset[str]] = {
    "open": frozenset({"cancel"}),
    "plan_presented": frozenset(DECISIONS),
    "plan_accepted": frozenset({"revise", "cancel"}),
    "revision_requested": frozenset({"accept", "cancel"}),
    "working": frozenset({"cancel"}),
    "blocked": frozenset({"cancel"}),
    "needs_review": frozenset({"cancel"}),
    "verified": frozenset({"cancel"}),
    "done": frozenset(),
    "cancelled": frozenset(),
}

NEXT_ACTIONS: dict[str, str] = {
    "open": "draft_or_present_plan",
    "plan_presented": "accept_or_revise_plan",
    "plan_accepted": "begin_work",
    "revision_requested": "revise_plan",
    "working": "record_observed_evidence",
    "blocked": "surface_blocker",
    "needs_review": "review_and_verify",
    "verified": "close_session",
    "done": "session_complete",
    "cancelled": "session_cancelled",
}

_FIELD_LIMITS: dict[str, int] = {
    "decision": 32,
    "goal": MAX_GOAL_CHARS,
    "workspace": 240,
    "model": 96,
    "lane": 64,
    "skill": 96,
    "next_action": 120,
}
_MUTABLE_FIELDS = tuple(_FIELD_LIMITS)
_NOTES_FIELDS = ("model", "lane", "skill", "next_action")


class SessionStateError(ValueError):
    """The wrapper state vocabulary was violated or the record is unusable."""


@dataclass(slots=True)
class SessionState:
    """The persisted wrapper lifecycle record for one chat session."""

    session_id: str
    status: str = "open"
    decision: str = ""
    goal: str = ""
    workspace: str = ""
    model: str = ""
    lane: str = ""
    skill: str = ""
    next_action: str = ""
    updated_at: str = ""
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": records.SCHEMA_VERSION,
            "record_type": RECORD_TYPE,
            "session_id": self.session_id,
            "status": self.status,
            "decision": self.decision,
            "goal": self.goal,
            "workspace": self.workspace,
            "model": self.model,
            "lane": self.lane,
            "skill": self.skill,
            "next_action": self.next_action,
            "updated_at": self.updated_at,
            "record_revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionState":
        """Rebuild a state fail-closed: anything off-vocabulary is refused."""
        if not isinstance(data, Mapping):
            raise SessionStateError("Wrapper session state must be a JSON object.")
        problems = records.validate_record(data)
        if problems:
            raise SessionStateError(problems[0])
        if data.get("record_type") != RECORD_TYPE:
            raise SessionStateError(f"Wrapper state record_type must be {RECORD_TYPE!r}.")

        status = data.get("status")
        if not isinstance(status, str) or status not in STATUSES:
            raise SessionStateError(f"Wrapper state status is invalid: {status!r}.")
        decision = data.get("decision", "")
        if not isinstance(decision, str) or (decision and decision not in DECISIONS):
            raise SessionStateError(f"Wrapper state decision is invalid: {decision!r}.")

        values: dict[str, str] = {}
        for field_name, limit in _FIELD_LIMITS.items():
            value = data.get(field_name, "")
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise SessionStateError(f"Wrapper state {field_name!r} must be a string.")
            if len(value) > limit:
                raise SessionStateError(
                    f"Wrapper state {field_name!r} exceeds {limit} characters."
                )
            values[field_name] = value

        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise SessionStateError("Wrapper state session_id must be a non-empty string.")
        revision = records.revision_of(data)
        if data.get("record_revision") != revision:
            raise SessionStateError(
                f"Wrapper state record_revision is invalid: {data.get('record_revision')!r}."
            )

        return cls(
            session_id=session_id,
            status=status,
            revision=revision,
            updated_at=str(data.get("updated_at", "")),
            **values,
        )

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


def _field(field_name: str, value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, int, float, bool)):
        raise SessionStateError(f"Wrapper state {field_name!r} must be a string or None.")
    return records.clean_text(value, limit=_FIELD_LIMITS[field_name])


def _record_state(current: Mapping[str, Any], session_id: str) -> SessionState:
    if not current:
        raise SessionStateError(
            f"No wrapper state for session {session_id}; call open_state first."
        )
    return SessionState.from_dict(current)


def _check_writable(state: SessionState) -> None:
    if state.terminal:
        raise SessionStateError(
            f"Cannot write to a {state.status} wrapper state for session {state.session_id}."
        )


def _validate_state_record(record: Mapping[str, Any]) -> None:
    SessionState.from_dict(record)


def _append_event(session_id: str, event: str, data: Mapping[str, Any] | None) -> None:
    # The record is authoritative; a broken diagnostic trail must not fail a
    # write that already succeeded.
    try:
        records.append_event(session_id, event, data)
    except records.RecordError:
        pass


def _guarded(
    session_id: str,
    *,
    operation: str,
    mutate,
    digest: str,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
    event: str = "",
    event_data: Mapping[str, Any] | None = None,
) -> SessionState:
    try:
        result = records.guarded_update(
            STATE_RECORD,
            session_id,
            mutate,
            operation=operation,
            expected_revision=expected_revision,
            mutation_id=mutation_id,
            mutation_digest=digest,
            validate=_validate_state_record,
        )
    except records.RecordError as exc:
        raise SessionStateError(str(exc)) from exc
    if event and not result.replayed:
        data = {**(event_data or {}), "revision": result.revision}
        _append_event(session_id, event, data)
    return SessionState.from_dict(result.record)


def open_state(
    session_id: str,
    *,
    goal: str,
    workspace: str,
    model: str = "",
    lane: str = "",
) -> tuple[SessionState, bool]:
    """Return ``(state, resumed)`` for one session's wrapper lifecycle."""
    try:
        existing = records.read_record(STATE_RECORD, session_id)
    except records.RecordError as exc:
        raise SessionStateError(str(exc)) from exc
    if existing:
        state = _record_state(existing, session_id)
        _append_event(
            session_id,
            "session_resumed",
            {"status": state.status, "revision": state.revision},
        )
        return state, True

    draft = SessionState(
        session_id=session_id,
        status="open",
        goal=_field("goal", goal),
        workspace=_field("workspace", workspace),
        model=_field("model", model),
        lane=_field("lane", lane),
        next_action=NEXT_ACTIONS["open"],
    )
    raced: list[bool] = []

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        if current:
            raced.append(True)
            return dict(current)
        return draft.to_dict()

    saved = _guarded(
        session_id,
        operation="open_state",
        mutate=mutate,
        digest=records.digest_of("open_state", session_id, draft.to_dict()),
    )
    if raced:
        _append_event(session_id, "session_resumed", {"status": saved.status, "revision": saved.revision})
        return saved, True
    _append_event(
        session_id,
        "session_started",
        {"status": saved.status, "goal": saved.goal, "revision": saved.revision},
    )
    return saved, False


def load_state(session_id: str) -> SessionState | None:
    """Return the session's state, or ``None`` when absent or unusable."""
    try:
        record = records.read_record(STATE_RECORD, session_id)
    except records.RecordError:
        return None
    if not record:
        return None
    try:
        return SessionState.from_dict(record)
    except (SessionStateError, records.RecordError):
        return None


def save_state(
    state: SessionState,
    *,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
) -> SessionState:
    """Persist ``state`` wholesale, gated on the persisted record's status."""
    if not isinstance(state, SessionState):
        raise SessionStateError("save_state expects a SessionState instance.")
    payload = state.to_dict()

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        if not current:
            return dict(payload)
        _check_writable(_record_state(current, state.session_id))
        return dict(payload)

    return _guarded(
        state.session_id,
        operation="save_state",
        mutate=mutate,
        digest=records.digest_of("save_state", payload),
        expected_revision=expected_revision,
        mutation_id=mutation_id,
    )


def transition(
    session_id: str,
    status: str,
    *,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
    **fields: object,
) -> SessionState:
    """Move the session to ``status`` when the transition table allows it."""
    if status not in STATUSES:
        raise SessionStateError(
            f"Unknown wrapper status {status!r}; expected one of {', '.join(STATUSES)}."
        )
    unknown = sorted(set(fields) - set(_MUTABLE_FIELDS))
    if unknown:
        raise SessionStateError(f"Unknown wrapper state field(s): {', '.join(unknown)}.")
    updates = {name: _field(name, value) for name, value in fields.items()}
    jumped: list[str] = []

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        state = _record_state(current, session_id)
        _check_writable(state)
        allowed = TRANSITIONS.get(state.status, frozenset())
        if status not in allowed:
            raise SessionStateError(
                f"Illegal wrapper transition {state.status} -> {status} "
                f"for session {session_id}; allowed: "
                f"{', '.join(sorted(allowed)) or 'none'}."
            )
        jumped.append(state.status)
        payload = state.to_dict()
        payload.update(updates)
        payload["status"] = status
        if "next_action" not in updates:
            payload["next_action"] = NEXT_ACTIONS.get(status, "show_status")
        return payload

    return _guarded(
        session_id,
        operation="transition",
        mutate=mutate,
        digest=records.digest_of("transition", session_id, status, updates),
        expected_revision=expected_revision,
        mutation_id=mutation_id,
        event=f"status_{status}",
        event_data={"to": status, "from": jumped[-1] if jumped else "", "revision": None},
    )


def record_decision(
    session_id: str,
    decision: str,
    *,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
) -> SessionState:
    """Apply one accept/revise/cancel decision through the plan gate."""
    if decision not in DECISIONS:
        raise SessionStateError(
            f"Plan decision must be {', '.join(DECISIONS)}, got {decision!r}."
        )
    target = DECISION_TARGETS[decision]
    jumped: list[str] = []

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        state = _record_state(current, session_id)
        _check_writable(state)
        allowed = DECISION_GATES.get(state.status, frozenset())
        if decision not in allowed:
            raise SessionStateError(
                f"Cannot {decision} a wrapper state while status is {state.status}."
            )
        if target not in TRANSITIONS.get(state.status, frozenset()):
            raise SessionStateError(
                f"Decision {decision!r} maps to {target!r}, which {state.status} cannot reach."
            )
        jumped.append(state.status)
        payload = state.to_dict()
        payload["decision"] = decision
        payload["status"] = target
        payload["next_action"] = NEXT_ACTIONS.get(target, "show_status")
        return payload

    return _guarded(
        session_id,
        operation="record_plan_decision",
        mutate=mutate,
        digest=records.digest_of("record_plan_decision", session_id, decision),
        expected_revision=expected_revision,
        mutation_id=mutation_id,
        event=f"decision_{decision}",
        event_data={"decision": decision, "status": target, "from": jumped[-1] if jumped else ""},
    )


def note(
    session_id: str,
    *,
    model: str | None = None,
    lane: str | None = None,
    skill: str | None = None,
    next_action: str | None = None,
) -> SessionState:
    """Refresh the non-status hints on an open journey without a status jump."""
    candidates = (("model", model), ("lane", lane), ("skill", skill), ("next_action", next_action))
    updates = {name: _field(name, value) for name, value in candidates if value is not None}
    if not updates:
        raise SessionStateError("note requires at least one of model, lane, skill, next_action.")

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        state = _record_state(current, session_id)
        _check_writable(state)
        payload = state.to_dict()
        payload.update(updates)
        return payload

    return _guarded(
        session_id,
        operation="note",
        mutate=mutate,
        digest=records.digest_of("note", session_id, updates),
    )


def close_state(session_id: str, status: str = "done") -> SessionState:
    """Move the journey to a terminal status, or return it when already closed."""
    if status not in TERMINAL_STATUSES:
        raise SessionStateError(
            f"close_state requires {' or '.join(sorted(TERMINAL_STATUSES))}, got {status!r}."
        )
    state = load_state(session_id)
    if state is None:
        raise SessionStateError(f"No wrapper state for session {session_id}; call open_state first.")
    if state.terminal:
        return state
    return transition(session_id, status)


def list_states(*, limit: int = 50) -> list[SessionState]:
    """Return persisted wrapper states, most recently updated first."""
    bounded = limit if isinstance(limit, int) and not isinstance(limit, bool) else 50
    bounded = max(1, min(MAX_LIST, bounded))
    root = records.wrapper_root(create=False)
    found: list[SessionState] = []
    try:
        entries = sorted(root.iterdir()) if root.is_dir() else []
    except OSError:
        return []
    for entry in entries:
        try:
            session_id = records.session_dir(entry.name, create=False).name
        except records.RecordError:
            continue
        if not entry.is_dir():
            continue
        state = load_state(session_id)
        if state is not None:
            found.append(state)
    found.sort(key=lambda item: item.updated_at, reverse=True)
    return found[:bounded]


def next_action_for(state: SessionState) -> str:
    """The single next step: an explicit hint wins, then the status default."""
    hint = records.clean_text(state.next_action, limit=_FIELD_LIMITS["next_action"])
    return hint or NEXT_ACTIONS.get(state.status, "show_status")
