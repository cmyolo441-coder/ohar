"""Deterministic, machine-keyed narrative of one wrapper session. Stdlib only.

The briefing is built only from wrapper records: the session state, the claims
evidence.py can support, the recorded evidence items, the plan shape and the
event tail.  The four-value ``signal`` answers one question -- does the human
have to move -- and ``complete`` is withheld unless every outstanding list is
empty.  Payload keys, step ids and ``next_action`` keep one spelling while only
the prose is rendered, so a consumer never has to know the reader's locale.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from ohar.terminal import Theme, clip, display_width, sanitize_text, table

from .continuity import Continuity
from .evidence import CLAIM_STATES, Claim, EvidenceItem
from .state import SessionState
from .state import STATUSES as SESSION_STATUSES

BRIEFING_SCHEMA_VERSION: str = "ohar_briefing/v1"

SIGNALS: tuple[str, ...] = ("waiting", "running", "stopped", "complete", "unknown")
BLOCKER_KINDS: tuple[str, ...] = ("failed", "blocked", "cancelled")
MAX_LINES: int = 8
MAX_GAP_IDS: int = 12

_BLOCKERS_SHOWN: Final[int] = 8
_REFS_SHOWN: Final[int] = 8
_LINE_CHARS: Final[int] = 200
_STEPS_SHOWN: Final[int] = 12
_MIN_WIDTH: Final[int] = 32

_STEP_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("goal", "Goal"),
    ("plan", "Plan"),
    ("approval", "Approval"),
    ("work", "Work"),
    ("evidence", "Evidence"),
    ("review", "Review"),
    ("verification", "Verification"),
)

_WAITING_STATUSES: Final[frozenset[str]] = frozenset(
    {"open", "plan_presented", "plan_accepted", "revision_requested", "needs_review"}
)
_STOPPED_STATUSES: Final[frozenset[str]] = frozenset({"blocked", "cancelled"})
_FINISHED_STATUSES: Final[frozenset[str]] = frozenset({"verified", "done"})
_STARTED_STATUSES: Final[frozenset[str]] = frozenset(
    {"working", "blocked", "needs_review", "verified", "done"}
)
# Handing work over for review is the end of the work step; the ladder only
# claims completion from ``verified`` or ``done``.
_WORK_DONE_STATUSES: Final[frozenset[str]] = frozenset({"needs_review", "verified", "done"})
_OUTSTANDING: Final[frozenset[str]] = frozenset({"pending", "in_progress", "blocked"})
_CLAIM_STATES: Final[frozenset[str]] = frozenset(CLAIM_STATES)

_SIGNAL_GLYPHS: Final[dict[str, tuple[str, str]]] = {
    "waiting": ("🟡", "[~]"),
    "running": ("🟢", "[>]"),
    "stopped": ("🔴", "[X]"),
    "complete": ("✅", "[+]"),
    "unknown": ("⚪", "[?]"),
}

_WAITING_HEADLINES: Final[dict[str, str]] = {
    "open": "Waiting on you: no goal or plan is recorded yet.",
    "plan_presented": "Waiting on you: accept or revise the plan.",
    "plan_accepted": "Waiting on you: the accepted plan has not been started.",
    "revision_requested": "Waiting on you: the plan needs revising.",
    "needs_review": "Review needed: recorded evidence is waiting on your decision.",
}

_STATUS_ACTIONS: Final[dict[str, str]] = {
    "open": "record_goal",
    "plan_presented": "await_plan_decision",
    "plan_accepted": "begin_work",
    "revision_requested": "revise_plan",
    "working": "record_evidence",
    "blocked": "resolve_blocker",
    "needs_review": "review_recorded_evidence",
    "verified": "confirm_completion",
    "done": "start_next_goal",
    "cancelled": "decide_redispatch",
}

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9_]{0,47}")

_CLAIM_BOUNDARY: Final[str] = (
    "This briefing is derived from persisted wrapper records only; recorded state is not "
    "execution, verification, review or approval of anything beyond what evidence observed."
)

__all__ = [
    "BLOCKER_KINDS",
    "BRIEFING_SCHEMA_VERSION",
    "MAX_GAP_IDS",
    "MAX_LINES",
    "SIGNALS",
    "build_briefing",
    "chat_briefing",
    "derive_signal",
    "render_briefing",
    "signal_glyph",
]


def derive_signal(
    state: SessionState, claims: Sequence[Claim], *, plan: Mapping[str, Any] | None = None
) -> str:
    """Return one of :data:`SIGNALS`: does the human have to move?

    A stopped step beats every other reading -- it is the only state where doing
    nothing leaves the work where it is -- and ``complete`` is withheld unless
    each gap list this briefing knows about is empty.
    """
    claim_list = _claims(claims)
    if claim_list is None:
        return "unknown"
    if not isinstance(state, SessionState) or state.status not in SESSION_STATUSES:
        return "unknown"
    if any(claim.state == "refuted" for claim in claim_list) or state.status in _STOPPED_STATUSES:
        return "stopped"
    if state.status in _WAITING_STATUSES:
        return "waiting"
    outstanding = _claim_gaps(claim_list) + _plan_gaps(plan)
    if state.status in _FINISHED_STATUSES:
        return "complete" if not outstanding else "running"
    if state.status == "working":
        return "running"
    return "unknown"


def signal_glyph(signal: str, *, glyphs: str = "unicode") -> str:
    """The mark for one signal; anything unrecognized falls to the unknown mark."""
    name = _text(signal)
    pair = _SIGNAL_GLYPHS.get(name) if name in SIGNALS else None
    chosen = pair or _SIGNAL_GLYPHS["unknown"]
    return chosen[1] if glyphs == "ascii" else chosen[0]


def build_briefing(
    *,
    state: SessionState,
    claims: Sequence[Claim],
    evidence: Sequence[EvidenceItem] = (),
    events: Sequence[Mapping[str, Any]] = (),
    plan: Mapping[str, Any] | None = None,
    continuity: Continuity | None = None,
    max_lines: int = MAX_LINES,
) -> dict[str, Any]:
    """Narrate one session's work from wrapper records, without claiming more."""
    claim_list = _claims(claims) or []
    items = _items(evidence)
    readable = isinstance(state, SessionState) and state.status in SESSION_STATUSES
    status = _text(state.status) if readable else ""
    goal = _text(state.goal) if readable else ""
    decision = _text(state.decision) if readable else ""

    signal = derive_signal(state, claim_list, plan=plan)
    progress = _progress(
        status=status,
        goal=goal,
        decision=decision,
        plan=plan,
        claims=claim_list,
        items=items,
        readable=readable,
    )
    blockers = _blockers(status=status, claims=claim_list, items=items)
    gaps, gap_count = _pending_gaps(progress, blockers, claims=claim_list, plan=plan)
    if signal == "complete" and _drifted(continuity):
        # The run looks finished, but the tree it finished in is not where the
        # record says it is: that is a question for a person, not a done claim.
        signal = "waiting"

    summary = _evidence_summary(progress=progress, claims=claim_list, items=items)
    next_action = _next_action(state if readable else None, signal=signal, status=status)
    headline = _safe(
        _headline(
            signal=signal,
            status=status,
            blockers=blockers,
            progress=progress,
            continuity=continuity,
        ),
        _LINE_CHARS,
    )
    lines = _bounded_lines(
        _user_facing_lines(
            headline=headline,
            signal=signal,
            status=status,
            decision=decision,
            progress=progress,
            blockers=blockers,
            gaps=gaps,
            gap_count=gap_count,
            summary=summary,
            continuity=continuity,
            last_event=_last_event(events),
            next_action=next_action,
        ),
        max_lines,
    )

    return {
        "schema_version": BRIEFING_SCHEMA_VERSION,
        "session_id": _text(state.session_id) if readable else "",
        "signal": signal,
        "headline": headline,
        "current_state": {
            "status": status or "unknown",
            "decision": decision or "none",
            "goal": _safe(goal, 120),
            "workspace": _safe(_text(state.workspace) if readable else "", 120),
            "model": _safe(_text(state.model) if readable else "", 80),
            "lane": _safe(_text(state.lane) if readable else "", 40),
            "skill": _safe(_text(state.skill) if readable else "", 60),
            "revision": state.revision if readable and isinstance(state.revision, int) else 0,
            "updated_at": _safe(_text(state.updated_at) if readable else "", 40),
            "terminal": bool(readable and state.terminal),
            "recorded": bool(readable),
            "last_event": _safe(_last_event(events), 60),
            "continuity": _text(continuity.resume_status) if continuity is not None else "",
        },
        "progress": progress,
        "blockers": blockers,
        "pending_gaps": gaps,
        "pending_gap_count": gap_count,
        "evidence_summary": summary,
        "next_action": next_action,
        "user_facing_lines": lines,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def chat_briefing(briefing: Mapping[str, Any]) -> dict[str, Any]:
    """Compact projection for notices and tool text: no tables, no internals."""
    source = briefing if isinstance(briefing, Mapping) else {}
    signal = _text(source.get("signal"))
    return {
        "schema_version": _text(source.get("schema_version")) or BRIEFING_SCHEMA_VERSION,
        "session_id": _text(source.get("session_id")),
        "headline": _safe(source.get("headline"), _LINE_CHARS),
        "signal": signal if signal in SIGNALS else "unknown",
        "lines": [
            _safe(line, _LINE_CHARS)
            for line in _as_list(source.get("user_facing_lines"))
            if _text(line)
        ][:MAX_LINES],
        "pending_gaps": [
            _safe(gap, 60) for gap in _as_list(source.get("pending_gaps")) if _text(gap)
        ][:MAX_GAP_IDS],
        "blockers": [
            {
                "id": _safe(item.get("id"), 60),
                "kind": _text(item.get("kind"))
                if _text(item.get("kind")) in BLOCKER_KINDS
                else "failed",
            }
            for item in _as_list(source.get("blockers"))
            if isinstance(item, Mapping)
        ][:_BLOCKERS_SHOWN],
        "next_action": _safe(source.get("next_action"), 48),
        "claim_boundary": _safe(source.get("claim_boundary"), 400),
    }


def render_briefing(
    briefing: Mapping[str, Any], theme: Theme, *, width: int, glyphs: str = "unicode"
) -> str:
    """Render the payload as one bounded, themed block."""
    source = briefing if isinstance(briefing, Mapping) else {}
    budget = _bounded(width)
    ascii_glyphs = glyphs == "ascii"
    mark = "-" if ascii_glyphs else "·"

    signal = _text(source.get("signal"))
    headline = _text(source.get("headline")) or "No briefing payload was recorded."
    ok = signal in ("complete", "running")
    glyph = signal_glyph(signal, glyphs=glyphs)
    room = max(4, budget - display_width(glyph) - 1)
    rows = [
        f"{glyph} {theme.paint(clip(headline, room), fg='accent' if ok else 'error', bold=True)}"
    ]

    current = source.get("current_state")
    current = current if isinstance(current, Mapping) else {}
    session_id = _text(source.get("session_id")) or "unknown"
    detail = " ".join(
        (
            f"session {mark} {session_id}",
            f"status {mark} {_text(current.get('status')) or 'unknown'}",
            f"decision {mark} {_text(current.get('decision')) or 'none'}",
        )
    )
    rows.append(theme.paint(clip(detail, budget), fg="muted"))

    steps = [step for step in _as_list(source.get("progress")) if isinstance(step, Mapping)]
    if steps:
        if budget >= 48:
            headers = ("step", "state", "summary")
            widths: tuple[int, ...] = (12, 12, max(8, budget - 30))
        else:
            headers = ("step", "state")
            widths = (12, 12)
        rows.extend(
            table(
                theme,
                headers,
                [_step_row(step, len(widths)) for step in steps[:_STEPS_SHOWN]],
                widths=widths,
                glyphs=glyphs,
            )
        )

    blockers = [item for item in _as_list(source.get("blockers")) if isinstance(item, Mapping)]
    if blockers:
        stopped = "; ".join(
            f"{_text(item.get('id'))} ({_text(item.get('kind')) or 'failed'})" for item in blockers[:4]
        )
        rows.append(theme.paint(clip(f"Stopped {mark} {stopped}", budget), fg="error"))
    gaps = [_text(gap) for gap in _as_list(source.get("pending_gaps")) if _text(gap)]
    if gaps:
        extra = max(0, len(gaps) - 5)
        remaining = ", ".join(gaps[:5]) + (f" (+{extra} more)" if extra else "")
        rows.append(theme.paint(clip(f"Remaining {mark} {remaining}", budget), fg="warning"))

    rows.append(
        theme.paint(
            clip(f"next action {mark} {_text(source.get('next_action')) or 'read_briefing'}", budget),
            fg="warning",
            bold=True,
        )
    )
    boundary = _text(source.get("claim_boundary")) or _CLAIM_BOUNDARY
    for wrapped in textwrap.wrap(
        clip(boundary, budget * 3),
        width=max(16, budget - 2),
        break_long_words=False,
        break_on_hyphens=False,
    )[:3]:
        rows.append(theme.paint(f"{mark} {wrapped}", fg="dim", dim=True))
    return "\n".join(rows)


def _headline(
    *,
    signal: str,
    status: str,
    blockers: list[dict[str, str]],
    progress: list[dict[str, str]],
    continuity: Continuity | None,
) -> str:
    if signal == "stopped":
        # A standing stop outranks every other headline: the ladder below
        # describes a run that is moving through it.
        first = blockers[0] if blockers else {"id": "work", "kind": "stopped"}
        return f"Work is stopped at {first['id']} ({first['kind']})."
    if signal == "unknown":
        return "The session state could not be read from the wrapper records."
    if signal == "waiting":
        if _drifted(continuity):
            return (
                f"The recorded workspace {continuity.workspace_drift}; "
                "confirm it before calling this done."
            )
        return _WAITING_HEADLINES.get(status, "Waiting on you: nothing is recorded as in flight.")
    if signal == "complete":
        return "Work is finished; nothing is outstanding."
    outstanding = sum(1 for step in progress if _text(step.get("state")) in _OUTSTANDING)
    if status in _FINISHED_STATUSES:
        return f"Recorded as {status}, with {outstanding} step(s) still outstanding."
    return "Work is in progress; nothing is waiting on you."


def _progress(
    *,
    status: str,
    goal: str,
    decision: str,
    plan: Mapping[str, Any] | None,
    claims: list[Claim],
    items: list[EvidenceItem],
    readable: bool,
) -> list[dict[str, str]]:
    if not readable:
        return [
            _step(step_id, label, "unknown", "No readable session state was recorded.")
            for step_id, label in _STEP_LABELS
        ]

    steps = _plan_steps(plan)
    approved = decision == "accept"
    started = status in _STARTED_STATUSES
    claim_state = {claim.claim: claim.state for claim in claims}
    observed_items = [item for item in items if item.result == "observed"]
    failed_items = [item for item in items if item.result == "failed"]

    if decision == "cancel":
        approval = _step("approval", "Approval", "blocked", "The plan was cancelled, not accepted.")
    elif decision == "revise":
        approval = _step(
            "approval", "Approval", "pending", "A revision was requested; the plan is not accepted."
        )
    else:
        approval = _step(
            "approval",
            "Approval",
            "complete" if approved else _state(False, bool(steps)),
            "The plan was accepted." if approved else "No acceptance of the plan is recorded.",
        )

    if status in _STOPPED_STATUSES:
        work = _step("work", "Work", "blocked", f"The session is recorded as {status}.")
    elif status in _WORK_DONE_STATUSES:
        work = _step("work", "Work", "complete", "Work reached a review or terminal status.")
    else:
        work = _step(
            "work",
            "Work",
            "in_progress" if status == "working" else _state(False, approved),
            "Work is in progress." if status == "working" else "No work has been recorded as started.",
        )

    if observed_items:
        evidence_step = _step(
            "evidence",
            "Evidence",
            "complete",
            f"{len(observed_items)} observed evidence item(s) recorded.",
        )
    elif failed_items:
        evidence_step = _step(
            "evidence",
            "Evidence",
            "blocked",
            f"{len(failed_items)} recorded evidence item(s) failed.",
        )
    else:
        evidence_step = _step(
            "evidence",
            "Evidence",
            "in_progress" if started else _state(False, started),
            "Work started with no evidence item recorded yet."
            if started
            else "No evidence has been recorded.",
        )

    if claim_state.get("work_verified") == "refuted":
        verification = _step(
            "verification",
            "Verification",
            "blocked",
            "Recorded evidence refutes the verification claim.",
        )
    elif claim_state.get("work_verified") == "observed" or status in ("verified", "done"):
        verification = _step(
            "verification", "Verification", "complete", "Postconditions were recorded as verified."
        )
    else:
        verification = _step(
            "verification",
            "Verification",
            _state(False, started),
            "Verification is pending on the recorded claims.",
        )

    review_state = (
        "complete"
        if status in ("verified", "done")
        else ("in_progress" if status == "needs_review" else _state(False, started))
    )
    return [
        _step(
            "goal",
            "Goal",
            "complete" if goal else "pending",
            "The goal is recorded." if goal else "No goal text is recorded.",
        ),
        _step(
            "plan",
            "Plan",
            _state(bool(steps), bool(goal)),
            f"{len(steps)} plan step(s) recorded." if steps else "No plan steps are recorded.",
        ),
        approval,
        work,
        evidence_step,
        _step(
            "review",
            "Review",
            review_state,
            "The work is waiting on a human review."
            if status == "needs_review"
            else "No review has been requested.",
        ),
        verification,
    ]


def _blockers(*, status: str, claims: list[Claim], items: list[EvidenceItem]) -> list[dict[str, str]]:
    """Steps something actually stopped, kept apart from steps that never happened."""
    found: list[dict[str, str]] = []
    if status == "blocked":
        found.append({"id": "work", "kind": "blocked", "reason": "the session is recorded as blocked"})
    if status == "cancelled":
        found.append({"id": "work", "kind": "cancelled", "reason": "the session was cancelled"})
    for claim in claims:
        if claim.state == "refuted":
            found.append(
                {
                    "id": _safe(claim.claim, 60) or "claim",
                    "kind": "failed",
                    "reason": _safe(claim.reason, 120) or "recorded evidence refutes this claim",
                }
            )
    failed = [item for item in items if item.result == "failed"]
    if failed:
        found.append(
            {
                "id": "evidence",
                "kind": "failed",
                "reason": f"{len(failed)} recorded evidence item(s) failed",
            }
        )
    return _dedupe_blockers(found)[:_BLOCKERS_SHOWN]


def _pending_gaps(
    progress: list[dict[str, str]],
    blockers: list[dict[str, str]],
    *,
    claims: list[Claim],
    plan: Mapping[str, Any] | None,
) -> tuple[list[str], int]:
    stopped_ids = {_text(blocker["id"]) for blocker in blockers}
    gaps = [
        _text(step["id"])
        for step in progress
        if _text(step.get("state")) in _OUTSTANDING and _text(step["id"]) not in stopped_ids
    ]
    gaps += _claim_gaps(claims)
    gaps += _plan_gaps(plan)
    all_gaps = _dedupe([gap for gap in gaps if gap])
    return all_gaps[:MAX_GAP_IDS], len(all_gaps)


def _claim_gaps(claims: Iterable[Claim]) -> list[str]:
    return [_safe(claim.claim, 60) for claim in claims if claim.state == "pending"]


def _plan_gaps(plan: object) -> list[str]:
    return [
        f"plan-{index}"
        for index, step in enumerate(_plan_steps(plan), 1)
        if _text(step.get("status")) != "completed"
    ]


def _plan_steps(plan: object) -> list[Mapping[str, Any]]:
    if not isinstance(plan, Mapping):
        return []
    steps = plan.get("steps")
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Iterable):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def _evidence_summary(
    *, progress: list[dict[str, str]], claims: list[Claim], items: list[EvidenceItem]
) -> dict[str, Any]:
    counts = {"observed": 0, "pending": 0, "refuted": 0}
    for claim in claims:
        if claim.state in counts:
            counts[claim.state] += 1
    return {
        "observed": [_text(step["id"]) for step in progress if _text(step.get("state")) == "complete"],
        "not_observed": [
            _text(step["id"]) for step in progress if _text(step.get("state")) in _OUTSTANDING
        ],
        "items": len(items),
        "observed_items": sum(1 for item in items if item.result == "observed"),
        "failed_items": sum(1 for item in items if item.result == "failed"),
        "skipped_items": sum(1 for item in items if item.result == "skipped"),
        "claims": counts,
        "refs": _dedupe([_safe(item.digest, 40) for item in items if _text(item.digest)])[:_REFS_SHOWN],
    }


def _user_facing_lines(
    *,
    headline: str,
    signal: str,
    status: str,
    decision: str,
    progress: list[dict[str, str]],
    blockers: list[dict[str, str]],
    gaps: list[str],
    gap_count: int,
    summary: dict[str, Any],
    continuity: Continuity | None,
    last_event: str,
    next_action: str,
) -> list[str]:
    # The glyph leads: this line is what a notice shows, so a reader who sees
    # only it still learns whether the session needs them.
    lines = [f"{signal_glyph(signal)} {headline}".strip()]
    if status:
        lines.append(f"Status: {status} · decision: {decision or 'none'}.")
    if progress and _text(progress[0].get("state")) != "unknown":
        done = sum(1 for step in progress if _text(step.get("state")) == "complete")
        lines.append(f"Progress: {done} of {len(progress)} recorded step(s) are complete.")
    if continuity is not None:
        lines.append(f"Continuity: {continuity.headline}")
    if blockers:
        stopped = "; ".join(f"{_text(item['id'])} ({_text(item['kind'])})" for item in blockers[:4])
        lines.append(f"Stopped: {stopped}.")
    if gaps:
        extra = max(0, gap_count - len(gaps))
        lines.append(f"Remaining: {', '.join(gaps[:5])}{f' (+{extra} more)' if extra else ''}.")
    if summary.get("items"):
        lines.append(
            f"Evidence: {summary['observed_items']} observed, {summary['failed_items']} failed, "
            f"{summary['skipped_items']} skipped."
        )
    if last_event:
        lines.append(f"Last recorded event: {last_event}.")
    # Said out loud, because the alternative is an absence: when nothing is left
    # every line above drops out and the briefing only got shorter.
    lines.append("Next: nothing is outstanding." if signal == "complete" else f"Next: {next_action}.")
    return _dedupe([_safe(line, _LINE_CHARS) for line in lines if _text(line)])


def _bounded_lines(lines: list[str], limit: object) -> list[str]:
    if not lines:
        return []
    try:
        budget = max(1, min(MAX_LINES, int(limit)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        budget = MAX_LINES
    if len(lines) <= budget:
        return lines
    if budget == 1:
        return [lines[0]]
    if budget == 2:
        return [lines[0], lines[-1]]
    kept = lines[: budget - 2]
    dropped = len(lines) - len(kept) - 1
    return [*kept, f"{dropped} further line(s) omitted; read the briefing payload.", lines[-1]]


def _next_action(state: SessionState | None, *, signal: str, status: str) -> str:
    if signal == "complete":
        # Deliberately not read from the record: next_action comes from an open
        # set of writers and none of them has to clear it on the way out, so the
        # last token written would ask for a step every gap list just agreed was
        # finished.
        return "nothing_outstanding"
    if state is not None:
        recorded = _text(state.next_action)
        if _TOKEN_RE.fullmatch(recorded):
            return recorded
    if signal == "stopped":
        return "resolve_blocker"
    if signal == "unknown":
        return "repair_wrapper_records"
    return _STATUS_ACTIONS.get(status, "read_briefing")


def _last_event(events: object) -> str:
    if isinstance(events, (str, bytes, Mapping)) or not isinstance(events, Iterable):
        return ""
    latest: Mapping[str, Any] | None = None
    for event in events:
        if isinstance(event, Mapping):
            latest = event
    if latest is None:
        return ""
    name = _text(latest.get("event") or latest.get("name") or latest.get("type"))
    when = _text(latest.get("at") or latest.get("time") or latest.get("timestamp"))
    if name and when:
        return f"{name} at {when}"
    return name or when


def _drifted(continuity: Continuity | None) -> bool:
    return bool(continuity is not None and continuity.workspace_drift in ("missing", "moved"))


def _state(done: bool, available: bool) -> str:
    if done:
        return "complete"
    return "pending" if available else "blocked"


def _step(step_id: str, label: str, state: str, summary: str) -> dict[str, str]:
    return {"id": step_id, "label": label, "state": state, "summary": _safe(summary, 120)}


def _step_row(step: Mapping[str, Any], columns: int) -> list[str]:
    row = [_text(step.get("id")), _text(step.get("state")) or "pending"]
    if columns > 2:
        row.append(_text(step.get("summary")))
    return row


def _claims(claims: object) -> list[Claim] | None:
    """Validate before projecting: a malformed claim list is not an empty one."""
    if isinstance(claims, (str, bytes, Mapping)) or not isinstance(claims, Iterable):
        return None
    found: list[Claim] = []
    for claim in claims:
        if not isinstance(claim, Claim) or claim.state not in _CLAIM_STATES:
            return None
        found.append(claim)
    return found


def _items(evidence: object) -> list[EvidenceItem]:
    if isinstance(evidence, (str, bytes, Mapping)) or not isinstance(evidence, Iterable):
        return []
    return [item for item in evidence if isinstance(item, EvidenceItem)]


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _dedupe_blockers(blockers: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for blocker in blockers:
        if blocker["kind"] not in BLOCKER_KINDS:
            continue
        key = (blocker["id"], blocker["kind"])
        if key in seen:
            continue
        seen.add(key)
        out.append(blocker)
    return out


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _safe(value: object, limit: int) -> str:
    collapsed = " ".join(sanitize_text("" if value is None else str(value)).split())
    return clip(collapsed, max(8, limit))


def _bounded(width: object) -> int:
    try:
        return max(_MIN_WIDTH, int(width))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _MIN_WIDTH
