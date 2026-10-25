"""Fail-closed continuity projection for a reopened OHAR session. Stdlib only.

Every claim here is read back from persisted wrapper records: the recorded
workspace path, the lane and model a session last ran on, the plan step counts,
the evidence tail and the transcript tail.  Unknown or malformed records degrade
to ``blocked``/``unknown`` instead of a reassuring default, and nothing in this
module resumes, deletes or rewrites a session.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, assert_never

from ohar.planner import STATUSES as PLAN_STATUSES
from ohar.sessions import SessionMeta
from ohar.terminal import Theme, clip, display_width, pad, sanitize_text, short_path, status_dot, strip_ansi

from .evidence import EvidenceItem
from .state import SessionState
from .state import STATUSES as SESSION_STATUSES

CONTINUITY_SCHEMA_VERSION: str = "ohar_continuity/v1"

ResumeStatus = Literal["fresh", "conversation_safe", "context_lost", "blocked"]
Drift = Literal["same", "moved", "missing", "unknown"]
MemoryState = Literal["available", "not_included", "unknown"]
LineDomain = Literal["workspace", "resume", "memory", "context"]

MAX_LINE_CHARS: Final[int] = 220
_CONTEXT_FACTS: Final[int] = 2
_PATH_CHARS: Final[int] = 48
_MIN_WIDTH: Final[int] = 24
_LABEL_CHARS: Final[int] = 11

_IN_PROGRESS: Final[str] = "in_progress"

_RESUME_OK: Final[frozenset[str]] = frozenset({"fresh", "conversation_safe"})

_CLAIM_BOUNDARY: Final[str] = (
    "This projection reads persisted wrapper records only; it does not resume a session, "
    "rewrite a workspace, or prove that recalled context was used."
)

__all__ = [
    "CONTINUITY_SCHEMA_VERSION",
    "Continuity",
    "ContinuityLine",
    "Drift",
    "LineDomain",
    "MemoryState",
    "ResumeStatus",
    "build_continuity",
    "render_continuity",
    "resume_status",
]


@dataclass(frozen=True, slots=True)
class ContinuityLine:
    """One sanitized sentence bound to the record domain it was read from."""

    domain: LineDomain
    text: str


@dataclass(frozen=True, slots=True)
class Continuity:
    resume_status: ResumeStatus
    workspace_drift: Drift
    lane_drift: Drift
    plan_counts: dict[str, int]
    memory: MemoryState
    headline: str
    lines: tuple[ContinuityLine, ...]
    next_action: str
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        counts = {status: _count(self.plan_counts.get(status)) for status in PLAN_STATUSES}
        return {
            "schema_version": CONTINUITY_SCHEMA_VERSION,
            "resume_status": self.resume_status,
            "workspace_drift": self.workspace_drift,
            "lane_drift": self.lane_drift,
            "plan_counts": counts,
            "memory": self.memory,
            "headline": self.headline,
            "lines": [{"domain": line.domain, "text": line.text} for line in self.lines],
            "next_action": self.next_action,
            "claim_boundary": self.claim_boundary,
        }


def resume_status(
    *,
    state: SessionState | None,
    messages: Sequence[Mapping[str, Any]],
    evidence: Sequence[EvidenceItem],
) -> ResumeStatus:
    """Say whether the persisted records support continuing this conversation.

    A malformed state, a transcript ending on an unanswered tool call, or a
    blocked or terminal journey all return ``blocked``: none of them may be read
    as "safe to carry on".  A transcript whose tail was trimmed returns
    ``context_lost`` rather than pretending the earlier turns are still there.
    """
    if state is not None:
        if not isinstance(state, SessionState) or state.status not in SESSION_STATUSES:
            return "blocked"
    if isinstance(evidence, (str, bytes, Mapping)) or not isinstance(evidence, Iterable):
        return "blocked"
    if not all(isinstance(item, EvidenceItem) for item in evidence):
        return "blocked"
    transcript = _transcript(messages)
    if transcript is None:
        return "blocked"
    if state is not None and (state.status == "blocked" or state.terminal):
        return "blocked"

    tail_dangling, gap = _tool_call_state(transcript)
    if tail_dangling:
        return "blocked"
    if gap or (state is None and transcript):
        return "context_lost"
    if transcript or evidence:
        return "conversation_safe"
    # A record past ``open`` means something already happened even when the
    # caller passed no tail: the session is not fresh, it is simply idle.
    if state is not None and (state.status != "open" or _text(state.goal) or _text(state.decision)):
        return "conversation_safe"
    return "fresh"


def build_continuity(
    *,
    meta: SessionMeta,
    state: SessionState | None,
    workspace: Path,
    plan: Mapping[str, Any] | None = None,
    evidence: Sequence[EvidenceItem] = (),
    messages: Sequence[Mapping[str, Any]] = (),
    memory_records: int | None = None,
) -> Continuity:
    """Project what actually breaks when *meta* is reopened in *workspace*."""
    if not isinstance(meta, SessionMeta):
        return _malformed_session_continuity()

    status = resume_status(state=state, messages=messages, evidence=evidence)
    counts, plan_malformed = _plan_counts(plan)
    recorded = _recorded_workspace(state, meta)
    live = _as_path(workspace)
    drift = _workspace_drift(recorded=recorded, live=live)
    lane_drift = _lane_drift(meta=meta, state=state)
    memory = _memory_state(memory_records)
    items = [item for item in evidence if isinstance(item, EvidenceItem)]
    observed = any(item.result == "observed" for item in items)

    context: list[str] = []
    if plan_malformed:
        context.append("The plan record was malformed, so step counts are reported as zero.")
    elif counts.get(_IN_PROGRESS, 0) and not observed:
        context.append(
            f"{counts[_IN_PROGRESS]} plan step(s) are in_progress with no observed evidence recorded."
        )
    if lane_drift == "moved":
        context.append("The recorded lane or model differs from the lane this session is on now.")
    trimmed = _trimmed_messages(meta=meta, messages=messages)
    if trimmed:
        context.append(
            f"Context was trimmed: {trimmed} of {meta.message_count} recorded message(s) "
            "are outside the transcript tail."
        )

    lines = [
        ContinuityLine("workspace", _workspace_line(drift, recorded=recorded, live=live)),
        ContinuityLine("resume", _resume_line(status)),
        ContinuityLine("memory", _memory_line(memory, memory_records=memory_records)),
    ]
    if context:
        lines.append(ContinuityLine("context", " · ".join(context[:_CONTEXT_FACTS])))
    lines = [ContinuityLine(line.domain, _safe(line.text, MAX_LINE_CHARS)) for line in lines]

    return Continuity(
        resume_status=status,
        workspace_drift=drift,
        lane_drift=lane_drift,
        plan_counts=counts,
        memory=memory,
        headline=_safe(_headline(status=status, drift=drift, lane_drift=lane_drift), MAX_LINE_CHARS),
        lines=tuple(lines),
        next_action=_next_action(
            status=status,
            drift=drift,
            lane_drift=lane_drift,
            counts=counts,
            observed=observed,
            memory=memory,
        ),
        claim_boundary=_CLAIM_BOUNDARY,
    )


def render_continuity(
    continuity: Continuity, theme: Theme, *, width: int, glyphs: str = "unicode"
) -> str:
    """Render the projection as one bounded, themed block."""
    budget = _bounded(width)
    ascii_glyphs = glyphs == "ascii"
    mark = "-" if ascii_glyphs else "·"

    ok = continuity.resume_status in _RESUME_OK
    dot = status_dot(theme, ok, glyphs=glyphs)
    room = max(4, budget - display_width(strip_ansi(dot)) - 1)
    title = theme.paint(
        clip(f"Continuity {mark} {continuity.headline}", room),
        fg="accent" if ok else "error",
        bold=True,
    )
    rows = [f"{dot} {title}"]
    for line in continuity.lines:
        label = theme.paint(pad(clip(f"{line.domain}:", _LABEL_CHARS - 1), _LABEL_CHARS), fg="muted")
        rows.append(f"{label}{theme.paint(clip(line.text, budget - _LABEL_CHARS), fg='text')}")
    rows.append(
        theme.paint(
            clip(f"next action {mark} {continuity.next_action}", budget),
            fg="warning",
            bold=True,
        )
    )
    boundary = textwrap.wrap(
        clip(continuity.claim_boundary, budget * 3),
        width=max(16, budget - 2),
        break_long_words=False,
        break_on_hyphens=False,
    )[:3]
    rows.extend(theme.paint(f"{mark} {line}", fg="dim", dim=True) for line in boundary)
    return "\n".join(rows)


def _malformed_session_continuity() -> Continuity:
    text = "The session record was missing or malformed, so no continuity claim is made."
    return Continuity(
        resume_status="blocked",
        workspace_drift="unknown",
        lane_drift="unknown",
        plan_counts={status: 0 for status in PLAN_STATUSES},
        memory="unknown",
        headline=_safe("Continuity could not be read from the session record.", MAX_LINE_CHARS),
        lines=tuple(
            ContinuityLine(domain, _safe(text, MAX_LINE_CHARS))
            for domain in ("workspace", "resume", "memory")
        ),
        next_action="repair_session_record",
        claim_boundary=_CLAIM_BOUNDARY,
    )


def _headline(*, status: ResumeStatus, drift: Drift, lane_drift: Drift) -> str:
    match status:
        case "blocked":
            return "Continuity is blocked; nothing will be resumed automatically."
        case "context_lost":
            return "Conversation context was lost; only the recorded tail survives."
        case "fresh":
            return "Fresh session; there is nothing to carry over yet."
        case "conversation_safe":
            pass
        case unreachable:
            assert_never(unreachable)
    if drift in ("missing", "moved"):
        return "The conversation tail is intact but the recorded workspace drifted."
    if lane_drift == "moved":
        return "The conversation tail is intact but the lane or model drifted."
    return "Conversation can continue from persisted records."


def _resume_line(status: ResumeStatus) -> str:
    match status:
        case "fresh":
            return "No conversation has been recorded yet, so nothing is being resumed."
        case "conversation_safe":
            return (
                "The transcript tail is complete, so the conversation continues from the "
                "messages and evidence already on disk."
            )
        case "context_lost":
            return (
                "Earlier turns are missing from the transcript tail; what was said before them "
                "cannot be quoted back."
            )
        case "blocked":
            return (
                "Persisted records do not support continuing this conversation: the journey is "
                "blocked or finished, or a recorded tool call was never answered."
            )
        case unreachable:
            assert_never(unreachable)


def _workspace_line(drift: Drift, *, recorded: str, live: Path | None) -> str:
    where = _shown(live) if live is not None else "unknown"
    if drift == "same":
        return f"The recorded workspace path resolves to this session's directory ({where})."
    if drift == "moved":
        return (
            f"The recorded workspace {_shown(recorded)} is a different directory "
            f"than this session's ({where})."
        )
    if drift == "missing":
        return f"The recorded workspace {_shown(recorded)} no longer exists on this filesystem."
    return "No usable workspace path was recorded, so nothing can be checked against the disk."


def _shown(value: object) -> str:
    """A path for display only: bounded, escaped, never trusted as an argument."""
    if isinstance(value, Path):
        candidate = str(value)
    else:
        candidate = _text(value)
    try:
        return _safe(short_path(candidate, None, _PATH_CHARS), _PATH_CHARS)
    except (OSError, RuntimeError, ValueError):
        return _safe(candidate, _PATH_CHARS) or "unknown"


def _memory_line(memory: MemoryState, *, memory_records: int | None) -> str:
    if memory == "available":
        return f"Workspace memory is readable: {int(memory_records or 0)} record(s) available."
    if memory == "not_included":
        return "Workspace memory is readable but holds no records for this workspace."
    return "Memory availability is unknown because the memory store was not readable or not consulted."


def _next_action(
    *,
    status: ResumeStatus,
    drift: Drift,
    lane_drift: Drift,
    counts: Mapping[str, int],
    observed: bool,
    memory: MemoryState,
) -> str:
    """One action token, chosen by severity, never by reassurance."""
    if status == "blocked":
        return "review_persisted_records"
    if drift == "missing":
        return "relocate_workspace"
    if drift == "moved":
        return "confirm_moved_workspace"
    if status == "context_lost":
        return "restate_goal_before_continuing"
    if lane_drift == "moved":
        return "confirm_lane_before_continuing"
    if counts.get(_IN_PROGRESS, 0) and not observed:
        return "review_unbacked_plan_steps"
    if memory == "unknown":
        return "check_workspace_memory"
    if status == "fresh":
        return "confirm_goal_before_work"
    return "continue_conversation"


def _transcript(messages: object) -> tuple[Mapping[str, Any], ...] | None:
    if isinstance(messages, (str, bytes, Mapping)) or not isinstance(messages, Iterable):
        return None
    transcript: list[Mapping[str, Any]] = []
    for item in messages:
        if not isinstance(item, Mapping) or not _role(item):
            return None
        transcript.append(item)
    return tuple(transcript)


def _call_ids(message: Mapping[str, Any]) -> tuple[frozenset[str], int]:
    calls = message.get("tool_calls")
    if calls is None:
        return frozenset(), 0
    if isinstance(calls, (str, bytes, Mapping)) or not isinstance(calls, Iterable):
        return frozenset(), 1
    ids: set[str] = set()
    anonymous = 0
    for call in calls:
        if not isinstance(call, Mapping):
            anonymous += 1
            continue
        identifier = call.get("id")
        if isinstance(identifier, str) and identifier:
            ids.add(identifier)
        else:
            anonymous += 1
    return frozenset(ids), anonymous


def _tool_call_state(transcript: tuple[Mapping[str, Any], ...]) -> tuple[bool, bool]:
    """Return (ends on an unanswered tool call, has an unresolvable gap)."""
    called: set[str] = set()
    anonymous = 0
    for message in transcript:
        if _role(message) == "assistant":
            ids, missing = _call_ids(message)
            called.update(ids)
            anonymous += missing
    resolved = {
        item.get("tool_call_id")
        for item in transcript
        if _role(item) == "tool" and isinstance(item.get("tool_call_id"), str)
    }
    orphan = any(_role(item) == "tool" and item.get("tool_call_id") not in called for item in transcript)
    unresolved = called - resolved

    tail_dangling = False
    if transcript and _role(transcript[-1]) == "assistant":
        ids, missing = _call_ids(transcript[-1])
        tail_dangling = missing > 0 or bool(ids & unresolved)
    gap = bool(unresolved) or orphan or anonymous > 0
    return tail_dangling, gap


def _role(message: Mapping[str, Any]) -> str:
    value = message.get("role")
    return value.strip().lower() if isinstance(value, str) else ""


def _trimmed_messages(*, meta: SessionMeta, messages: object) -> int:
    transcript = _transcript(messages)
    if not transcript:
        return 0
    return max(0, meta.message_count - len(transcript))


def _recorded_workspace(state: SessionState | None, meta: SessionMeta) -> str:
    # Control characters are dropped before any path reaches the filesystem:
    # this is a projection, and an unusable recorded string must not become an
    # exception or an escape sequence on screen.
    if state is not None:
        recorded = _one_line(state.workspace)
        if recorded:
            return recorded
    return _one_line(meta.cwd)


def _as_path(value: object) -> Path | None:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return Path(value).expanduser()
        except (OSError, ValueError):
            return None
    return None


def _workspace_drift(*, recorded: str, live: Path | None) -> Drift:
    if not recorded or live is None:
        return "unknown"
    try:
        origin = Path(recorded).expanduser()
        if not origin.exists():
            return "missing"
        return "same" if origin.resolve() == live.resolve() else "moved"
    except (OSError, RuntimeError, ValueError):
        return "unknown"


def _lane_drift(*, meta: SessionMeta, state: SessionState | None) -> Drift:
    if state is None:
        return "unknown"
    recorded = (_text(state.lane), _text(state.model))
    current = (_text(meta.lane), _text(meta.model))
    if not any(recorded) or not any(current):
        return "unknown"
    return "same" if recorded == current else "moved"


def _memory_state(value: object) -> MemoryState:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return "unknown"
    return "not_included" if value == 0 else "available"


def _plan_counts(plan: Mapping[str, Any] | None) -> tuple[dict[str, int], bool]:
    counts = {status: 0 for status in PLAN_STATUSES}
    if plan is None:
        return counts, False
    if not isinstance(plan, Mapping):
        return counts, True
    if "steps" not in plan and "counts" not in plan:
        return counts, False

    recorded = plan.get("counts")
    if isinstance(recorded, Mapping):
        checked: dict[str, int] = {}
        for status in PLAN_STATUSES:
            value = _count(recorded.get(status, 0), allow_default=False)
            if value is None:
                checked = {}
                break
            checked[status] = value
        if checked:
            return checked, False

    steps = plan.get("steps")
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Iterable):
        return counts, True
    malformed = False
    for step in steps:
        if not isinstance(step, Mapping) or step.get("status") not in counts:
            malformed = True
            continue
        counts[str(step["status"])] += 1
    return counts, malformed


def _count(value: object, *, allow_default: bool = True) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0 if allow_default else None
    return value


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _one_line(value: object) -> str:
    """Strip ANSI and control characters, then collapse to one whitespace-merged line."""
    return " ".join(sanitize_text("" if value is None else str(value)).split())


def _safe(value: object, limit: int) -> str:
    return clip(_one_line(value), max(8, limit))


def _bounded(width: object) -> int:
    try:
        return max(_MIN_WIDTH, int(width))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _MIN_WIDTH
