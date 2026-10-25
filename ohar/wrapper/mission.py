"""Read-only mission control for OHAR wrapper sessions. Stdlib only.

This is a portfolio view, not a dispatcher: one journey ladder derived from the
persisted state and the claims evidence.py can support, a recovery hint per
rung, an autonomy read from recorded accept decisions, and table rows for the
sessions the store can list but cannot describe.  Unknown or mismatched records
fall to ``broken_link`` rather than claiming progress, and nothing here runs a
command, resumes a session or merges anything.
"""

from __future__ import annotations

import textwrap
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from ohar.sessions import SessionMeta, SessionStore
from ohar.terminal import Theme, clip, display_width, status_dot, strip_ansi, table

from . import records
from .briefing import build_briefing, derive_signal, signal_glyph
from .continuity import Continuity, build_continuity
from .evidence import CLAIM_STATES, Claim, EvidenceItem
from .evidence import claims as recorded_claims
from .evidence import items as recorded_items
from .state import DECISIONS, TERMINAL_STATUSES, SessionState
from .state import STATUSES as SESSION_STATUSES
from .state import list_states, load_state

MISSION_SCHEMA_VERSION: str = "ohar_mission/v1"

JOURNEY_STATES: tuple[str, ...] = (
    "drafted",
    "approved",
    "working",
    "blocked",
    "review",
    "verified",
    "finished",
    "cancelled",
    "broken_link",
)
DEFAULT_STALE_MINUTES: int = 60

_DRAFTED: Final[str] = "drafted"
_BROKEN: Final[str] = "broken_link"
_ACCEPT: Final[str] = DECISIONS[0]

_LIST_LIMIT: Final[int] = 20
_SCAN_STATES: Final[int] = 200
_MAX_ROWS: Final[int] = 100
_EVENTS_SHOWN: Final[int] = 50
_EVIDENCE_SHOWN: Final[int] = 25
_MAX_ID_CHARS: Final[int] = 128

# Which signals can honestly describe a session in each status. A pairing that
# is not here means the state record and the claim records disagree, and the
# only safe answer to that is a broken link, not a chosen side.
_ALLOWED_SIGNALS: Final[dict[str, frozenset[str]]] = {
    "open": frozenset({"waiting", "stopped"}),
    "plan_presented": frozenset({"waiting", "stopped"}),
    "plan_accepted": frozenset({"waiting", "stopped"}),
    "revision_requested": frozenset({"waiting", "stopped"}),
    "working": frozenset({"running", "stopped"}),
    "blocked": frozenset({"stopped"}),
    "needs_review": frozenset({"waiting", "stopped"}),
    "verified": frozenset({"complete", "running", "stopped"}),
    "done": frozenset({"complete", "running", "stopped"}),
    "cancelled": frozenset({"stopped"}),
}

_RECOVERY: Final[dict[str, tuple[str, str, str]]] = {
    _DRAFTED: (
        "not_dispatched",
        "present_plan_for_approval",
        "Nothing has been accepted yet: record a plan and ask for a decision.",
    ),
    "approved": (
        "queued",
        "begin_work",
        "The plan is accepted; no work has been recorded as started.",
    ),
    "working": (
        "running_observed",
        "record_evidence",
        "Work is in progress; keep recording what the harness observes.",
    ),
    "blocked": (
        "recovery_blocked",
        "resolve_blocker",
        "The session is recorded as blocked; clearing it is a human decision.",
    ),
    "review": (
        "awaiting_review",
        "review_recorded_evidence",
        "Recorded evidence is waiting on a human review.",
    ),
    "verified": (
        "evidence_available",
        "confirm_completion",
        "Postconditions were verified; closing the session is still a human call.",
    ),
    "finished": (
        "closed",
        "start_next_goal",
        "The session is closed; its records stay readable for the next one.",
    ),
    "cancelled": (
        "cancelled_observed",
        "decide_redispatch",
        "The session was cancelled; nothing is still running to reattach to.",
    ),
    _BROKEN: (
        "recovery_blocked",
        "repair_wrapper_records",
        "The records for this session do not agree, so no journey is claimed.",
    ),
}

_CLAIM_BOUNDARY: Final[str] = (
    "Mission control is a read-only local projection of wrapper records; it does not dispatch, "
    "resume, approve or merge anything, and the only claims here are the ones observed evidence backs."
)

__all__ = [
    "DEFAULT_STALE_MINUTES",
    "JOURNEY_STATES",
    "MISSION_SCHEMA_VERSION",
    "autonomy",
    "build_mission",
    "journey_state",
    "list_missions",
    "mission_rows",
    "recovery",
    "render_mission",
]


def journey_state(state: SessionState, claims: Sequence[Claim], *, signal: str) -> str:
    """One ladder rung from the state record, cross-checked against the signal.

    Unreadable input, an unknown signal, or a state and claim pair that cannot
    both be true returns ``broken_link``: the ladder never claims progress it
    cannot support.
    """
    if not isinstance(state, SessionState) or state.status not in SESSION_STATUSES:
        return _BROKEN
    rows = _claims(claims)
    if rows is None:
        return _BROKEN
    if signal not in _ALLOWED_SIGNALS.get(state.status, frozenset()):
        return _BROKEN

    match state.status:
        case "open" | "plan_presented" | "revision_requested":
            journey = _DRAFTED
        case "plan_accepted":
            journey = "approved"
        case "working":
            journey = "working"
        case "blocked":
            journey = "blocked"
        case "needs_review":
            journey = "review"
        case "verified":
            journey = "verified"
        case "done":
            journey = "finished"
        case "cancelled":
            journey = "cancelled"
        case _:
            return _BROKEN

    # A refuted verification claim above a finished rung is the mismatch this
    # projection refuses to arbitrate: one record says done, the other says no.
    if journey in ("verified", "finished") and any(
        row.claim == "work_verified" and row.state == "refuted" for row in rows
    ):
        return _BROKEN
    return journey


def recovery(journey: str, *, resume_safe: bool) -> dict[str, str]:
    """The one hint for one ladder rung; an unknown rung recovers nothing."""
    name = journey if journey in JOURNEY_STATES else _BROKEN
    status, next_action, hint = _RECOVERY[name]
    return {
        "status": status,
        "resume_safe": "true" if resume_safe else "false",
        "next_action": next_action,
        "hint": hint,
    }


def autonomy(session_id: str, *, threshold: int = 2) -> dict[str, Any]:
    """Read an accepted-pattern count back from the recorded decisions.

    Autonomy is a statement about history, not a licence: the boundary that
    approvals still gate mutating work travels with the payload.
    """
    bounded = _clamp(threshold, 1, 50, default=2)
    current = _load(session_id)
    skill = _text(current.skill) if current is not None else ""
    known = _valid_id(session_id)
    counted = [
        row
        for row in _states()
        if row.decision == _ACCEPT
        and (
            (skill and row.skill == skill)
            or (not skill and row.session_id == session_id)
        )
    ]
    accepted = len(counted) if known else 0
    events, errors = _events(session_id)
    autonomous = accepted >= bounded and not errors
    return {
        "schema_version": MISSION_SCHEMA_VERSION,
        "session_id": session_id if known else "",
        "skill": skill,
        "accepted_count": accepted,
        "threshold": bounded,
        "mode": "autonomous_pattern" if autonomous else "guided_first_use",
        "next_action": "continue_recorded_pattern" if autonomous else "explain_before_next_step",
        "observed_events": len(events),
        "record_errors": len(errors),
        "confirmation": {
            "required": True,
            "policy": "pattern_based_autonomy" if autonomous else "optional_first_use",
            "boundary": "Approvals still gate mutating, destructive and external-write actions.",
        },
        "claim_boundary": (
            "Autonomy counts recorded accept decisions only; it does not prove an approval "
            "was waived or that earlier work was verified."
        ),
    }


def build_mission(
    session_id: str, *, store: SessionStore | None = None, now: float | None = None
) -> dict[str, Any]:
    """Aggregate one session's records into a read-only mission view."""
    moment = _now(now)
    state = _load(session_id)
    rows = _session_claims(session_id, state)
    evidence = _session_items(session_id)
    meta, transcript = _meta_for(session_id, store, state, moment)
    events, event_errors = _events(session_id)

    signal = derive_signal(state, rows) if state is not None else "unknown"
    journey = journey_state(state, rows, signal=signal)
    continuity = _continuity(meta=meta, state=state, transcript=transcript, evidence=evidence)
    resume_safe = continuity.resume_status in ("fresh", "conversation_safe")
    briefing = (
        build_briefing(
            state=state, claims=rows, evidence=evidence, events=events, continuity=continuity
        )
        if state is not None
        else None
    )
    hint = recovery(journey, resume_safe=resume_safe)
    updated = _text(state.updated_at) if state is not None else _text(meta.updated_at)
    age = _age(_epoch(updated), moment)
    stale_minutes = DEFAULT_STALE_MINUTES * 60

    return {
        "schema_version": MISSION_SCHEMA_VERSION,
        "session_id": records.clean_text(meta.id, limit=40) if _valid_id(session_id) else "",
        "journey": {"state": journey},
        "signal": signal,
        "status": _text(state.status) if state is not None else "unrecorded",
        "title": records.clean_text(meta.title, limit=80),
        "workspace": records.clean_text(
            state.workspace if state is not None else meta.cwd, limit=120
        ),
        "model": records.clean_text(state.model if state is not None else meta.model, limit=60),
        "lane": records.clean_text(state.lane if state is not None else meta.lane, limit=40),
        "updated_at": records.clean_text(updated, limit=40),
        "age_seconds": age,
        "stale": bool(age is not None and age > stale_minutes),
        "terminal": bool(state is not None and state.status in TERMINAL_STATUSES),
        "broken_link": journey == _BROKEN,
        "untracked": state is None,
        "record_errors": len(event_errors),
        "claims": {
            "observed": [row.claim for row in rows if row.state == "observed"],
            "pending": [row.claim for row in rows if row.state == "pending"],
            "refuted": [row.claim for row in rows if row.state == "refuted"],
            "allowed_claims": [row.claim for row in rows if row.state == "observed"],
            "blocked_claims": [row.claim for row in rows if row.state != "observed"],
        },
        "continuity": {
            "resume_status": continuity.resume_status,
            "workspace_drift": continuity.workspace_drift,
            "lane_drift": continuity.lane_drift,
            "memory": continuity.memory,
            "next_action": continuity.next_action,
        },
        "recovery": hint,
        "autonomy": autonomy(session_id),
        "headline": briefing["headline"]
        if briefing
        else "No wrapper state is recorded for this session.",
        "next_action": briefing["next_action"] if briefing else hint["next_action"],
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def list_missions(store: SessionStore, *, limit: int = 20) -> list[dict[str, Any]]:
    """Join the store's session list to the wrapper states it cannot describe."""
    bounded = _clamp(limit, 1, _MAX_ROWS, default=_LIST_LIMIT)
    states = {row.session_id: row for row in _states(_SCAN_STATES)}
    moment = _now(None)
    stale_minutes = DEFAULT_STALE_MINUTES * 60
    rows: list[dict[str, Any]] = []
    for meta in _metas(store, bounded):
        state = states.get(meta.id)
        # A portfolio row reads only the state record: claims are derived per
        # session on demand, so a row never borrows another session's evidence.
        signal = derive_signal(state, ()) if state is not None else "unknown"
        journey = journey_state(state, (), signal=signal)
        updated = _text(state.updated_at) if state is not None else _text(meta.updated_at)
        age = _age(_epoch(updated), moment)
        rows.append(
            {
                "schema_version": MISSION_SCHEMA_VERSION,
                "session_id": records.clean_text(meta.id, limit=40),
                "title": records.clean_text(meta.title, limit=80),
                "journey": journey,
                "signal": signal,
                "status": _text(state.status) if state is not None else "unrecorded",
                "workspace": records.clean_text(
                    state.workspace if state is not None else meta.cwd, limit=80
                ),
                "lane": records.clean_text(state.lane if state is not None else meta.lane, limit=40),
                "model": records.clean_text(state.model if state is not None else meta.model, limit=60),
                "updated_at": records.clean_text(updated, limit=40),
                "age_seconds": age,
                "stale": bool(age is not None and age > stale_minutes),
                "terminal": bool(state is not None and state.status in TERMINAL_STATUSES),
                "broken_link": journey == _BROKEN,
                "untracked": state is None,
                "next_action": recovery(journey, resume_safe=False)["next_action"],
                "claim_boundary": _CLAIM_BOUNDARY,
            }
        )
    return rows


def mission_rows(
    missions: Sequence[Mapping[str, Any]], *, now: float | None = None
) -> list[list[str]]:
    """Table rows: signal, journey, status, age, flags, then the session text."""
    moment = _now(now)
    stale_minutes = DEFAULT_STALE_MINUTES * 60
    rows: list[list[str]] = []
    for mission in missions:
        if not isinstance(mission, Mapping):
            continue
        journey = _nested_journey(mission) or _BROKEN
        age = _age(_epoch(_text(mission.get("updated_at"))), moment)
        if age is None:
            recorded = mission.get("age_seconds")
            if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
                age = None
            else:
                age = max(0.0, float(recorded))
        flags = [
            name
            for name, active in (
                ("stale", bool(mission.get("stale")) or (age is not None and age > stale_minutes)),
                ("terminal", bool(mission.get("terminal"))),
                ("broken_link", journey == _BROKEN),
                ("untracked", bool(mission.get("untracked"))),
            )
            if active
        ]
        rows.append(
            [
                records.clean_text(mission.get("session_id"), limit=24),
                signal_glyph(_text(mission.get("signal"))),
                records.clean_text(journey, limit=16),
                records.clean_text(_text(mission.get("status")) or "unrecorded", limit=16),
                _age_text(age),
                "|".join(flags) if flags else "-",
                records.clean_text(
                    _text(mission.get("title"))
                    or _text(mission.get("headline"))
                    or _text(mission.get("next_action")),
                    limit=48,
                ),
            ]
        )
    return rows


def render_mission(
    payload: Mapping[str, Any], theme: Theme, *, width: int, glyphs: str = "unicode"
) -> str:
    """Render one mission view as a bounded, themed block."""
    source = payload if isinstance(payload, Mapping) else {}
    budget = _bounded(width)
    ascii_glyphs = glyphs == "ascii"
    mark = "-" if ascii_glyphs else "·"

    journey = _nested_journey(source) or _BROKEN
    signal = _text(source.get("signal"))
    ok = journey in ("working", "verified", "finished")
    session_id = records.clean_text(source.get("session_id") or "unknown", limit=32)
    dot = status_dot(theme, ok, glyphs=glyphs)
    room = max(4, budget - display_width(strip_ansi(dot)) - 1)
    rows = [
        f"{dot} {theme.paint(clip(f'Mission {mark} {session_id}', room), fg='accent', bold=True)}"
    ]
    headline = _text(source.get("headline"))
    if headline:
        rows.append(
            theme.paint(clip(f"{signal_glyph(signal, glyphs=glyphs)} {headline}", budget), fg="text")
        )

    hint = _mapping(source.get("recovery"))
    auto = _mapping(source.get("autonomy"))
    continuity = _mapping(source.get("continuity"))
    claim_rows = _mapping(source.get("claims"))
    fields = (
        ("journey", journey),
        ("signal", f"{signal_glyph(signal, glyphs=glyphs)} {signal or 'unknown'}"),
        ("status", records.clean_text(source.get("status") or "unrecorded", limit=40)),
        ("recovery", records.clean_text(hint.get("status") or "unknown", limit=40)),
        ("resume_safe", records.clean_text(hint.get("resume_safe") or "false", limit=12)),
        ("autonomy", records.clean_text(auto.get("mode") or "guided_first_use", limit=40)),
        (
            "claims",
            f"{len(_as_list(claim_rows.get('allowed_claims')))} allowed / "
            f"{len(_as_list(claim_rows.get('blocked_claims')))} blocked",
        ),
        ("continuity", records.clean_text(continuity.get("resume_status") or "unknown", limit=40)),
        ("workspace", records.clean_text(source.get("workspace") or "unrecorded", limit=60)),
        (
            "next_action",
            records.clean_text(
                _text(source.get("next_action")) or _text(hint.get("next_action")), limit=48
            ),
        ),
    )
    rows.extend(
        table(
            theme,
            ("field", "value"),
            [list(field) for field in fields],
            widths=(12, max(8, budget - 16)),
            glyphs=glyphs,
        )
    )
    hint_text = records.clean_text(hint.get("hint"), limit=budget * 2)
    if hint_text:
        rows.append(theme.paint(clip(f"hint {mark} {hint_text}", budget), fg="warning"))
    boundary = _text(source.get("claim_boundary")) or _CLAIM_BOUNDARY
    for wrapped in textwrap.wrap(
        clip(boundary, budget * 3),
        width=max(16, budget - 2),
        break_long_words=False,
        break_on_hyphens=False,
    )[:3]:
        rows.append(theme.paint(f"{mark} {wrapped}", fg="dim", dim=True))
    return "\n".join(rows)


def _continuity(
    *,
    meta: SessionMeta,
    state: SessionState | None,
    transcript: tuple[Mapping[str, Any], ...],
    evidence: list[EvidenceItem],
) -> Continuity:
    # The live directory for a portfolio read is the one the session record
    # itself was opened in; the wrapper workspace is checked against it.
    return build_continuity(
        meta=meta,
        state=state,
        workspace=_live_workspace(meta),
        evidence=evidence,
        messages=transcript,
        memory_records=None,
    )


def _meta_for(
    session_id: str, store: SessionStore | None, state: SessionState | None, moment: float
) -> tuple[SessionMeta, tuple[Mapping[str, Any], ...]]:
    """Read the session's own metadata and transcript when the store can supply them."""
    if isinstance(store, SessionStore) and _valid_id(session_id):
        try:
            session = store.load(session_id)
            return session.meta, tuple(
                message for message in session.messages if isinstance(message, Mapping)
            )
        except (OSError, ValueError, TypeError):
            pass
        known = next((item for item in _metas(store, _MAX_ROWS) if item.id == session_id), None)
        if known is not None:
            return known, ()
    stamp = datetime.fromtimestamp(moment, timezone.utc).isoformat(timespec="seconds")
    return (
        SessionMeta(
            id=session_id if _valid_id(session_id) else "",
            title="Mission view",
            created_at=stamp,
            updated_at=_text(state.updated_at) if state is not None else "",
            cwd=_text(state.workspace) if state is not None else "",
            model=_text(state.model) if state is not None else "",
            lane=_text(state.lane) if state is not None else "",
        ),
        (),
    )


def _live_workspace(meta: SessionMeta) -> Path:
    recorded = _text(meta.cwd)
    if recorded:
        try:
            return Path(recorded).expanduser()
        except (OSError, ValueError):
            pass
    try:
        return Path.cwd()
    except OSError:
        return Path(".")


def _load(session_id: object) -> SessionState | None:
    if not isinstance(session_id, str):
        return None
    try:
        return load_state(session_id)
    except (records.RecordError, OSError, ValueError, TypeError):
        return None


def _session_claims(session_id: str, state: SessionState | None) -> list[Claim]:
    if state is None:
        return []
    try:
        return list(recorded_claims(session_id))
    except (records.RecordError, OSError, ValueError, TypeError):
        return []


def _session_items(session_id: str) -> list[EvidenceItem]:
    try:
        return list(recorded_items(session_id, limit=_EVIDENCE_SHOWN))
    except (records.RecordError, OSError, ValueError, TypeError):
        return []


def _states(limit: int = _SCAN_STATES) -> list[SessionState]:
    try:
        return list(list_states(limit=limit))
    except (records.RecordError, OSError, ValueError, TypeError):
        return []


def _events(session_id: object) -> tuple[list[dict[str, Any]], list[str]]:
    if not _valid_id(session_id):
        return [], ["invalid session id"]
    try:
        found, errors = records.read_events(session_id, limit=_EVENTS_SHOWN)
    except (records.RecordError, OSError, ValueError, TypeError):
        return [], ["unreadable event log"]
    return list(found), list(errors)


def _metas(store: object, limit: int) -> list[SessionMeta]:
    if not isinstance(store, SessionStore):
        return []
    try:
        listed = store.list(limit)
    except (OSError, ValueError, TypeError):
        return []
    return [meta for meta in listed if isinstance(meta, SessionMeta)]


def _claims(rows: object) -> list[Claim] | None:
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Iterable):
        return None
    found: list[Claim] = []
    for row in rows:
        if not isinstance(row, Claim) or row.state not in CLAIM_STATES:
            return None
        found.append(row)
    return found


def _valid_id(session_id: object) -> bool:
    if not isinstance(session_id, str) or not session_id or len(session_id) > _MAX_ID_CHARS:
        return False
    try:
        records.session_dir(session_id, create=False)
    except records.RecordError:
        return False
    return True


def _nested_journey(mission: Mapping[str, Any]) -> str:
    journey = mission.get("journey")
    if isinstance(journey, Mapping):
        return _text(journey.get("state"))
    return _text(journey)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _epoch(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.timestamp()
    except (OverflowError, OSError, ValueError):
        return None


def _now(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return time.time()
    return float(value)


def _age(updated: float | None, moment: float) -> float | None:
    if updated is None:
        return None
    return max(0.0, moment - updated)


def _age_text(age: float | None) -> str:
    if age is None:
        return "unknown"
    minutes = int(age // 60)
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"{minutes}m"
    if minutes < 60 * 24:
        return f"{minutes // 60}h"
    return f"{minutes // (60 * 24)}d"


def _clamp(value: object, low: int, high: int, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(low, min(high, value))


def _bounded(width: object) -> int:
    value = _clamp(width, 24, 400, default=48)
    return value


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
