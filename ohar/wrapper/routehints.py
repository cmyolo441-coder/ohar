"""The wrapper's route-hint rail: which installed skill fits this turn, and what to do next.

Offline-only by construction. It ranks ``SkillLibrary.search`` results instead
of re-implementing token overlap, runs nothing, and persists a digest of the
request rather than the request. The one machine-learned input is the user's own
previously accepted picks, read back from wrapper event logs and applied as a
bounded weight, which is what separates ``guided_first_use`` from
``autonomous_pattern``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ohar.skills import Skill, SkillLibrary
from ohar.terminal import clip

from .records import RecordError, append_event, clean_text, read_events, wrapper_root
from .state import SessionStateError, load_state, note

__all__ = [
    "HINTS_SCHEMA_VERSION",
    "MAX_HINTS",
    "MIN_HINT_SCORE",
    "HISTORY_WEIGHT",
    "RouteHint",
    "preference_history",
    "record_choice",
    "hint_for",
    "answers_directly",
    "hint_line",
    "hints_payload",
]

HINTS_SCHEMA_VERSION = "ohar_route_hints/v1"
MAX_HINTS = 3
MIN_HINT_SCORE = 0.35
HISTORY_WEIGHT = 0.15

CHOICE_EVENT = "route_hint_choice"
PATTERN_THRESHOLD = 2
MIN_REQUEST_CHARS = 3
MAX_SKILL_CHARS = 64
MAX_REASON_CHARS = 120
MAX_ADJACENT = 3
MAX_HINT_LINE_CHARS = 200
MAX_HISTORY_SESSIONS = 200

_NEXT_LOADED = "load_skill"
_NEXT_CATALOG = "show_skills"
_SUBMIT_LOADED = "/skill"
_SUBMIT_CATALOG = "/skills"

_NAMED_STRENGTH = 0.85
_INDEX_STRENGTH = 0.50
_RANK_PENALTY = 0.12

_WORD_SPLIT_RE = re.compile(r"[-_\s]+")
_GREETING_RE = re.compile(
    r"^(hi|hey|hello|yo|hiya|howdy|good\s+(morning|afternoon|evening))[!.?]*$",
)
_THANKS_RE = re.compile(r"\b(thanks|thank\s+you|thank\s+u|ty|cheers|no\s+worries|awesome\s+thanks)\b")
_CATALOG_RE = re.compile(
    r"("
    r"(?:which|what|any|list|show|tell me about|do you have|how many)\s+"
    r"(?:the\s+)?(?:available\s+)?(?:installed\s+)?(?:skill|skills|capability|capabilities)\b"
    r"|\bskill\s+(?:list|index|catalog(?:ue)?)\b"
    r"|/skills\b"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RouteHint:
    """One bounded suggestion: a skill, why it fits, and the next action."""

    skill: str
    label: str
    reason: str
    next_action: str
    confidence: Literal["high", "medium", "low"]
    score: float
    submit_text: str
    adjacent: tuple[str, ...]


def preference_history(*, limit: int = 200) -> dict[str, int]:
    """Accepted skill picks counted back from this machine's wrapper event logs."""
    ceiling = _clamp_limit(limit, MAX_HISTORY_SESSIONS)
    counts: dict[str, int] = {}
    try:
        root = wrapper_root(create=False)
        entries = sorted(root.iterdir(), key=lambda path: path.name.casefold())
    except OSError:
        return counts
    for entry in entries[:ceiling]:
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        try:
            events, _errors = read_events(entry.name, limit=ceiling)
        except RecordError:
            continue
        for event in events:
            if not isinstance(event, Mapping):
                continue
            if _event_name(event) != CHOICE_EVENT:
                continue
            data = _event_data(event)
            if not _truthy(data.get("accepted")):
                continue
            skill = _skill_key(data.get("skill"))
            if skill:
                counts[skill] = counts.get(skill, 0) + 1
    return counts


def record_choice(session_id: str, skill: str, *, accepted: bool) -> None:
    """Record one picker outcome; an accepted pick also lands on the session state."""
    key = _skill_key(skill)
    append_event(session_id, CHOICE_EVENT, {"skill": key, "accepted": bool(accepted)})
    if not key or not accepted:
        return
    try:
        if load_state(session_id) is not None:
            note(session_id, skill=key)
    except SessionStateError:
        # A closed journey cannot be written; the event log is the history.
        pass


def hint_for(
    text: str,
    library: SkillLibrary,
    *,
    limit: int = 2,
    history: Mapping[str, int] | None = None,
) -> list[RouteHint]:
    """At most ``limit`` ranked hints, or none when nothing clears the floor."""
    wanted = _clamp_limit(limit, MAX_HINTS)
    request = " ".join(str(text or "").split())
    if not request:
        return []
    folded = request.casefold()
    if _is_catalog_question(folded):
        return _catalog_hints(library, wanted)

    picks = history if isinstance(history, Mapping) else {}
    strongest = max([int(value) for value in picks.values() if isinstance(value, int)] or [0])
    candidates = library.search(folded, limit=MAX_HINTS)
    names = tuple(_display_name(skill.name) for skill in candidates if _skill_key(skill.name))

    scored: list[RouteHint] = []
    for index, skill in enumerate(candidates):
        key = _skill_key(skill.name)
        if not key:
            continue
        named = _is_named(folded, skill)
        count = int(picks.get(key, 0)) if isinstance(picks.get(key), int) else 0
        score = (_NAMED_STRENGTH if named else _INDEX_STRENGTH) - _RANK_PENALTY * index
        if count > 0 and strongest > 0:
            score += HISTORY_WEIGHT * min(1.0, count / float(strongest))
        score = min(1.0, max(0.0, score))
        if score < MIN_HINT_SCORE:
            continue
        display = _display_name(skill.name)
        scored.append(
            RouteHint(
                skill=display,
                label=_label(display),
                reason=_reason(named, count, score),
                next_action=_NEXT_LOADED,
                confidence=_confidence(score),
                score=round(score, 3),
                submit_text=f"{_SUBMIT_LOADED} {display}",
                adjacent=_adjacent(names, key),
            )
        )
    # Search order is the index's own ranking; the history weight can move it.
    scored.sort(key=lambda hint: (-hint.score, _skill_key(hint.skill)))
    return scored[:wanted]


def answers_directly(text: str, hints: Sequence[RouteHint]) -> bool:
    """True when the surface should answer in chat instead of opening the rail.

    Greetings, thanks, a catalog question (answered straight from the skill
    index) and too-short requests all qualify — as does a turn the rail has
    nothing for, where rendering an empty hint line is noise.
    """
    request = " ".join(str(text or "").split())
    if not request or len(request) < MIN_REQUEST_CHARS:
        return True
    folded = request.casefold()
    if _GREETING_RE.match(folded) or _THANKS_RE.search(folded) or _is_catalog_question(folded):
        return True
    return not hints


def hint_line(hints: Sequence[RouteHint], *, glyphs: str = "unicode") -> str:
    """One terminal line naming the rail, or that there is no rail."""
    sep = "-" if glyphs == "ascii" else "·"
    if not hints:
        return clip(f"no strong skill hint {sep} {_SUBMIT_CATALOG} lists what exists", MAX_HINT_LINE_CHARS)
    parts: list[str] = []
    for hint in list(hints)[:MAX_HINTS]:
        tail = f" {sep} {len(hint.adjacent)} adjacent" if hint.adjacent else ""
        parts.append(
            f"{hint.label} {sep} next: {hint.next_action} {sep} {hint.confidence}{tail}"
        )
    return clip(f"skill hint {sep} " + " | ".join(parts), MAX_HINT_LINE_CHARS)


def hints_payload(
    text: str,
    library: SkillLibrary,
    *,
    limit: int = 2,
    history: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Metadata only: the request never appears in the returned payload."""
    request = str(text or "")
    folded = " ".join(request.split()).casefold()
    picks = history if isinstance(history, Mapping) else {}
    hints = hint_for(request, library, limit=limit, history=picks)
    top = hints[0] if hints else None
    stored = picks.get(_skill_key(top.skill)) if top else None
    accepted = int(stored) if isinstance(stored, int) and not isinstance(stored, bool) else 0
    digest = hashlib.sha256(request.encode("utf-8", errors="replace")).hexdigest()
    return {
        "schema_version": HINTS_SCHEMA_VERSION,
        "message_length": len(request),
        "prompt_digest": digest,
        "prompt_reference": f"sha256:{digest[:12]} · {len(request)} chars",
        "catalog_question": _is_catalog_question(folded),
        "answers_directly": answers_directly(request, hints),
        "mode": "autonomous_pattern" if accepted >= PATTERN_THRESHOLD else "guided_first_use",
        "accepted_pattern_count": accepted,
        "hints": [_hint_dict(hint) for hint in hints],
        "privacy": {
            "mode": "metadata_only",
            "raw_prompt_stored": False,
            "raw_prompt_echoed": False,
            "stored_fields": ["message_length", "prompt_digest", "skill names", "route hint"],
        },
        "claim_boundary": (
            "A route hint is a ranking of the installed skill index only. It is not skill "
            "selection, execution, generated output, or verification evidence."
        ),
    }


def _catalog_hints(library: SkillLibrary, limit: int) -> list[RouteHint]:
    skills = [skill for skill in library.list() if _skill_key(skill.name)]
    names = tuple(_display_name(skill.name) for skill in skills)
    hints: list[RouteHint] = []
    for skill in skills[:limit]:
        key = _skill_key(skill.name)
        display = _display_name(skill.name)
        hints.append(
            RouteHint(
                skill=display,
                label=_label(display),
                reason="listed from the installed skill index",
                next_action=_NEXT_CATALOG,
                confidence="high",
                score=1.0,
                submit_text=f"{_SUBMIT_LOADED} {display}",
                adjacent=_adjacent(names, key),
            )
        )
    return hints


def _hint_dict(hint: RouteHint) -> dict[str, Any]:
    return {
        "skill": hint.skill,
        "label": hint.label,
        "reason": hint.reason,
        "next_action": hint.next_action,
        "confidence": hint.confidence,
        "score": hint.score,
        "submit_text": hint.submit_text,
        "adjacent": list(hint.adjacent),
    }


def _adjacent(names: Sequence[str], key: str) -> tuple[str, ...]:
    """The other candidates, by their real names, so a picker can offer them."""
    return tuple(name for name in names if name and _skill_key(name) != key)[:MAX_ADJACENT]


def _reason(named: bool, count: int, score: float) -> str:
    parts = ["named in the request" if named else "matched the skill index"]
    if count >= PATTERN_THRESHOLD:
        parts.append(f"your usual pick ({count} accepted)")
    elif count > 0:
        parts.append(f"{count} accepted pick")
    elif score >= _NAMED_STRENGTH:
        parts.append("first use: confirm before loading")
    return clean_text(" · ".join(parts), limit=MAX_REASON_CHARS, single_line=True)


def _is_named(folded: str, skill: Skill) -> bool:
    names = [skill.name] + list(skill.aliases)
    return any(_skill_key(name) and _skill_key(name) in folded for name in names)


def _is_catalog_question(folded: str) -> bool:
    return bool(_CATALOG_RE.search(folded))


def _confidence(score: float) -> Literal["high", "medium", "low"]:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _label(name: str) -> str:
    words = [word for word in _WORD_SPLIT_RE.split(str(name or "")) if word]
    return clean_text(" ".join(word[:1].upper() + word[1:] for word in words), limit=MAX_SKILL_CHARS, single_line=True)


def _display_name(value: object) -> str:
    """A skill name that is safe to print and to put back on the command line."""
    return clean_text(value, limit=MAX_SKILL_CHARS, single_line=True)


def _skill_key(value: object) -> str:
    return _display_name(value).casefold()


def _event_name(event: Mapping[str, Any]) -> str:
    for key in ("event", "name", "type"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _event_data(event: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("data", "payload", "fields"):
        nested = event.get(key)
        if isinstance(nested, Mapping):
            return nested
    return event


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "accepted", "observed"}
    return bool(value)


def _clamp_limit(value: object, ceiling: int) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ceiling
    return max(1, min(number, ceiling))
