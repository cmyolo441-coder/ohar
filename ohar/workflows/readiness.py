"""One truthful answer to "can you do this now?" for OHAR's own capability surfaces.

Two axes, kept apart on purpose, because welding them is the defect:

* the **evidence tier** — what class of fact is the strongest thing anyone
  recorded: `installed` (a local configuration fact) < `host_observed` <
  `usable_observed` < `used`, plus the derived `stale` and the empty `none`;
* the **state** — the answer itself: `ready`, `blocked`, `not_observed`,
  `stale`, `failed`, with the smallest next action attached to every state that
  is not `ready`.

Configuration can mechanically never reach `ready`: only the four recordable
tiers may appear in an evidence record, and `installed` sits below the tier
level a positive answer requires. Freshness is derived at read time from the
horizon the caller passes, never stored in the record, so shortening the horizon
applies to evidence already on disk. Invalid evidence is rejected rather than
allowed to raise the answer, and a preserved prior is used only when nothing in
scope survived.

Everything here is pure with respect to the host: it takes already-resolved
facts (config sections, toolbox names, mcp status strings, browser availability)
as plain values, imports no cli/browser/mcp module, and probes nothing.
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from ..terminal import Theme, format_duration, make_theme, sanitize_text, strip_ansi, table
from ..types import Config, Tool, ToolContext, ToolResult
from .journal import UNKNOWN_AGE_SECONDS, ActivityJournal
from .store import WorkflowError, clean_text, now_iso, stamp_to_epoch, validate_vocabularies

EVIDENCE_TIERS: tuple[str, ...] = ("installed", "host_observed", "usable_observed", "used")
STALE_TIER = "stale"
NONE_TIER = "none"
# The tiers an answer may report: the four recordable ones plus the two derived.
REPORTED_TIERS: tuple[str, ...] = (*EVIDENCE_TIERS, STALE_TIER, NONE_TIER)
EVIDENCE_RESULTS: tuple[str, ...] = ("observed", "blocked", "failed")

READINESS_STATES: tuple[str, ...] = ("ready", "blocked", "not_observed", "stale", "failed")
STATE_SOURCES: tuple[str, ...] = ("derived", "preserved_prior", "no_valid_evidence")

STALE_AFTER_SECONDS = 21600
SURFACES: tuple[str, ...] = (
    "files",
    "shell",
    "web",
    "browser",
    "mcp",
    "subagents",
    "memory",
    "workflows",
)

NEXT_ACTIONS: dict[str, str] = {
    "ready": "perform_the_action",
    "blocked": "clear_the_recorded_blocker",
    "not_observed": "observe_the_surface_before_acting",
    "stale": "re_observe_the_surface_before_acting",
    "failed": "read_the_recorded_failure_before_retrying",
}

MAX_ACTION_CHARS = 160
MAX_REASONS = 6
MAX_DETAIL_CHARS = 200
MAX_SURFACE_CHARS = 80

CLAIM_BOUNDARY = (
    "This answer reports only what OHAR already recorded about one surface. Nothing here probed, launched, "
    "or connected to anything: a ready answer says the surface was observed working recently, not that this "
    "action will succeed, and no state here is execution, verification, review, or merge evidence."
)

_TIER_LEVELS: dict[str, int] = {tier: level for level, tier in enumerate(EVIDENCE_TIERS, start=1)}
# The lowest tier that can answer "yes, now". Below it a record describes setup
# or presence, not the surface doing the thing.
_READY_TIER_LEVEL = _TIER_LEVELS["usable_observed"]

_SERVER_SEPARATOR = ":"
_MAX_REASON_CHARS = 160
_MAX_SURVEY_ANSWERS = 64
_MAX_AT_CHARS = 64
_EVIDENCE_KEYS = ("surface", "action", "tier", "result", "at", "detail")
_RAW_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "env",
        "headers",
        "key",
        "password",
        "raw",
        "secret",
        "stderr",
        "token",
    }
)

# Which tools an observation of a surface can have come through: exact names,
# then prefixes for families OHAR names dynamically (`mcp__<server>__<tool>`).
_SURFACE_TOOLS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "files": (("edit_file", "glob", "grep", "list_files", "read_file", "write_file"), ()),
    "shell": (("shell",), ()),
    "web": (("web_fetch", "web_search"), ()),
    "browser": (
        (
            "browser_back",
            "browser_click",
            "browser_open",
            "browser_press",
            "browser_screenshot",
            "browser_scroll",
            "browser_snapshot",
            "browser_type",
        ),
        ("browser_",),
    ),
    "mcp": ((), ("mcp__",)),
    "subagents": (("task",), ()),
    "memory": (("forget", "recall", "remember"), ()),
    "workflows": (
        ("board", "capabilities", "goal_status", "unit_status", "why_blocked"),
        ("gate", "goal_", "loop_", "unit_", "verify_"),
    ),
}

_MUTATING_CAPABLE = frozenset(
    {"browser", "files", "mcp", "memory", "shell", "subagents", "web", "workflows"}
)
_MUTATING_TOKENS = frozenset(
    {
        "apply",
        "click",
        "commit",
        "create",
        "delete",
        "drop",
        "edit",
        "exec",
        "forget",
        "install",
        "move",
        "patch",
        "post",
        "push",
        "remove",
        "rename",
        "reset",
        "run",
        "send",
        "set",
        "shell",
        "start",
        "submit",
        "type",
        "update",
        "upload",
        "write",
    }
)


@dataclass(slots=True)
class Evidence:
    surface: str
    action: str
    tier: str
    result: str
    at: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _EVIDENCE_KEYS}


def build_evidence(
    *,
    surface: str,
    action: str,
    tier: str,
    result: str = "observed",
    at: str = "",
    detail: str = "",
) -> dict[str, Any]:
    """Mint one bounded evidence record, or refuse it."""
    record = Evidence(
        surface=clean_text(surface, limit=MAX_SURFACE_CHARS).lower(),
        action=clean_text(action, limit=MAX_ACTION_CHARS),
        tier=clean_text(tier, limit=32).lower(),
        result=clean_text(result, limit=16).lower() or "observed",
        at=clean_text(at, limit=_MAX_AT_CHARS) or now_iso(),
        detail=clean_text(detail, limit=MAX_DETAIL_CHARS),
    )
    errors = evidence_errors(record)
    if errors:
        raise WorkflowError(errors[0])
    return record.to_dict()


def evidence_errors(record: Any) -> list[str]:
    """Every reason one record is not usable evidence. Invalid means rejected, never fatal."""
    if isinstance(record, Evidence):
        record = record.to_dict()
    if not isinstance(record, Mapping):
        return ["evidence must be an object"]
    errors: list[str] = []
    forbidden = sorted(str(key) for key in record if str(key).lower() in _RAW_KEYS)
    if forbidden:
        errors.append(f"evidence must not carry raw or secret keys: {forbidden}")
    present = {str(key) for key in record}
    missing = sorted((set(_EVIDENCE_KEYS) - set(forbidden)) - present)
    if missing:
        errors.append(f"evidence is missing keys: {missing}")
    unexpected = sorted(present - set(_EVIDENCE_KEYS) - set(forbidden))
    if unexpected:
        errors.append(f"evidence has unsupported keys: {unexpected}")
    if errors:
        return errors
    if _surface_family(record.get("surface")) is None:
        errors.append(f"evidence surface is not an OHAR capability surface: {record.get('surface')!r}")
    errors.extend(validate_vocabularies(record, [("tier", EVIDENCE_TIERS), ("result", EVIDENCE_RESULTS)]))
    for key in ("surface", "action", "at", "detail"):
        value = record.get(key)
        if not isinstance(value, str):
            errors.append(f"{key} must be a string")
        elif value != sanitize_text(value, keep_newlines=False):
            errors.append(f"{key} must be one line without control or ANSI characters")
    if len(str(record.get("action", ""))) > MAX_ACTION_CHARS:
        errors.append(f"action must be at most {MAX_ACTION_CHARS} characters")
    if len(str(record.get("detail", ""))) > MAX_DETAIL_CHARS:
        errors.append(f"detail must be at most {MAX_DETAIL_CHARS} characters")
    if not str(record.get("at", "")).strip():
        errors.append("at is required; an undated observation cannot prove freshness")
    if record.get("tier") == "used" and not str(record.get("action", "")).strip():
        errors.append("used-tier evidence must name the action it observed")
    return errors


def tier_level(tier: str) -> int:
    """Strength of a tier: 0 for `stale`, `none`, and anything unrecognised."""
    return _TIER_LEVELS.get(str(tier or "").strip().lower(), 0)


@dataclass(slots=True)
class ReadinessAnswer:
    surface: str
    action: str
    state: str
    tier: str
    age_seconds: int
    next_action: str
    reasons: list[str] = field(default_factory=list)
    evidence_count: int = 0
    rejected_count: int = 0
    state_source: str = "derived"

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "action": self.action,
            "state": self.state,
            "tier": self.tier,
            "age_seconds": self.age_seconds,
            "next_action": self.next_action,
            "reasons": list(self.reasons),
            "evidence_count": self.evidence_count,
            "rejected_count": self.rejected_count,
            "state_source": self.state_source,
        }


def answer_readiness(
    *,
    surface: str,
    action: str,
    configured: bool,
    evidence: Sequence[Mapping[str, Any]] = (),
    prior: Mapping[str, Any] | None = None,
    now: float | None = None,
    stale_after_seconds: float = STALE_AFTER_SECONDS,
) -> ReadinessAnswer:
    """The one answer to "can you do this now?" for one surface and action.

    Supplied evidence is validated, then scoped, then aged — in that order, so
    bad evidence structurally cannot raise the answer. `now` is a parameter and
    the horizon is a parameter, so freshness is derived and testable.
    """
    moment = time.time() if now is None else float(now)
    horizon = max(1.0, float(stale_after_seconds))
    family, name = _surface_name(surface)
    asked = clean_text(action, limit=MAX_ACTION_CHARS)
    supplied = list(evidence)

    scoped, rejected = _scope(supplied, name=name, family=family, action=asked)
    prior_answer = _prior(prior, name=name, action=asked) if rejected and not scoped else None

    reasons: list[str] = [f"'{name}' is {'configured in OHAR' if configured else 'not configured in OHAR'}."]
    if rejected:
        reasons.append(
            f"{rejected} of {len(supplied)} supplied record(s) were invalid and were rejected; "
            "rejected evidence cannot raise this answer."
        )

    if prior_answer is not None:
        state = str(prior_answer["state"])
        tier = str(prior_answer["tier"])
        age = UNKNOWN_AGE_SECONDS
        source = "preserved_prior"
        reasons.append("Nothing supplied about this surface survived validation, so the last valid answer is kept.")
    elif not scoped:
        state = "not_observed"
        tier = "installed" if configured else NONE_TIER
        age = UNKNOWN_AGE_SECONDS
        source = "no_valid_evidence" if rejected else "derived"
    else:
        state, tier, age, source = _derive(scoped, moment=moment, horizon=horizon)

    # The mechanical form of "configuration alone is not capability".
    if state == "ready" and tier_level(tier) < _READY_TIER_LEVEL:
        state = "not_observed"
        reasons.append(f"{tier} evidence describes presence, not use, so it cannot answer ready.")

    reasons.extend(
        _state_reasons(state, tier=tier, age=age, horizon=horizon, name=name, scoped=scoped)
    )
    return ReadinessAnswer(
        surface=name,
        action=asked,
        state=state,
        tier=tier,
        age_seconds=age,
        next_action=NEXT_ACTIONS[state],
        reasons=_bounded(reasons),
        evidence_count=len(scoped),
        rejected_count=rejected,
        state_source=source,
    )


def _derive(scoped: Sequence[dict[str, Any]], *, moment: float, horizon: float) -> tuple[str, str, int, str]:
    """(state, tier, age, source) from in-scope valid evidence."""
    aged = [(_age_seconds(str(record.get("at", "")), moment, horizon), record) for record in scoped]
    fresh = [(age, record) for age, record in aged if age <= horizon]
    if not fresh:
        # Something was true here and none of it still is. Reporting the tier it
        # used to reach would be the exact confusion this module removes.
        return "stale", STALE_TIER, min(age for age, _ in aged), "derived"
    positives = [(age, record) for age, record in fresh if record.get("result") == "observed"]
    negatives = [(age, record) for age, record in fresh if record.get("result") in ("blocked", "failed")]
    strongest = max(positives, key=_positive_rank) if positives else None
    newest_negative = min(negatives, key=lambda entry: entry[0]) if negatives else None
    if newest_negative is not None and (strongest is None or newest_negative[0] <= strongest[0]):
        # A block or failure stands until something at least as new replaces it.
        tier = str(strongest[1].get("tier", NONE_TIER)) if strongest else NONE_TIER
        return str(newest_negative[1].get("result")), tier, newest_negative[0], "derived"
    if strongest is None:
        return "not_observed", NONE_TIER, UNKNOWN_AGE_SECONDS, "derived"
    tier = str(strongest[1].get("tier", NONE_TIER))
    state = "ready" if tier_level(tier) >= _READY_TIER_LEVEL else "not_observed"
    return state, tier, strongest[0], "derived"


def _positive_rank(entry: tuple[int, dict[str, Any]]) -> tuple[int, int]:
    """Strongest tier first, freshest as the tiebreak, so the pick is stable."""
    age, record = entry
    return (tier_level(str(record.get("tier", ""))), -age)


def _scope(
    evidence: Sequence[Mapping[str, Any]], *, name: str, family: str, action: str
) -> tuple[list[dict[str, Any]], int]:
    """Valid records that speak about this surface and action, plus the reject count."""
    scoped: list[dict[str, Any]] = []
    rejected = 0
    for record in evidence:
        if evidence_errors(record):
            rejected += 1
            continue
        data = dict(record)
        if not _in_scope(str(data.get("surface", "")), name=name, family=family):
            continue
        named = clean_text(data.get("action", ""), limit=MAX_ACTION_CHARS).lower()
        if named and action and named != action.lower():
            # Another action's truth is not weaker truth about this one.
            continue
        scoped.append(data)
    return scoped, rejected


def _surface_family(value: object) -> str | None:
    """The base surface a name refers to, or None when it names nothing OHAR has.

    Only `mcp` may carry a member suffix (`mcp:<server>`); `files:extra` is not a
    surface, so a typo cannot invent one.
    """
    text = str(value or "").strip().lower()
    if not text or len(text) > MAX_SURFACE_CHARS or text.count(_SERVER_SEPARATOR) > 1:
        return None
    base, _, suffix = text.partition(_SERVER_SEPARATOR)
    if base not in SURFACES:
        return None
    if not suffix:
        return base
    if base != "mcp" or not clean_text(suffix, limit=48) or suffix != clean_text(suffix, limit=48):
        return None
    return base


def _surface_name(value: object) -> tuple[str, str]:
    text = clean_text(value, limit=MAX_SURFACE_CHARS).lower()
    family = _surface_family(text)
    if family is None:
        raise WorkflowError(
            f"'{clean_text(value, limit=40)}' is not an OHAR capability surface. Known surfaces: "
            f"{', '.join(SURFACES)}, or mcp:<server>."
        )
    return family, text


def _in_scope(record_surface: str, *, name: str, family: str) -> bool:
    """Whether one record speaks about the asked surface.

    A member record (`mcp:fs`) answers for its member and for the bare family
    question, because one working member is evidence about the family. A family
    record never answers for one member, and sibling members never substitute
    for each other.
    """
    record_base, _, record_suffix = record_surface.strip().lower().partition(_SERVER_SEPARATOR)
    if record_base != family:
        return False
    asked_suffix = name.partition(_SERVER_SEPARATOR)[2]
    if record_suffix and not asked_suffix:
        return True
    return record_suffix == asked_suffix


def _is_mutating(action: str) -> bool:
    tokens = set(clean_text(action, limit=MAX_ACTION_CHARS).lower().replace("_", " ").split())
    return bool(tokens & _MUTATING_TOKENS)


def _stamp(moment: float) -> str:
    """The read moment as an ISO stamp, so host facts age from `now`, not the wall clock."""
    return datetime.fromtimestamp(moment, timezone.utc).isoformat(timespec="microseconds")


def _age_seconds(stamp: str, moment: float, horizon: float) -> int:
    """Age in whole seconds; unreadable and future stamps read as past the horizon."""
    epoch = stamp_to_epoch(stamp)
    if epoch is None:
        return int(horizon) + 1
    age = int(moment - epoch)
    return int(horizon) + 1 if age < 0 else age


def _prior(prior: Any, *, name: str, action: str) -> dict[str, str] | None:
    """The last valid answer, only when it is about this same surface and action."""
    if not isinstance(prior, Mapping):
        return None
    if str(prior.get("surface", "")).strip().lower() != name:
        return None
    if action and clean_text(prior.get("action", ""), limit=MAX_ACTION_CHARS).lower() != action.lower():
        return None
    state = str(prior.get("state", "") or "")
    tier = str(prior.get("tier", "") or "")
    if state not in READINESS_STATES or tier not in REPORTED_TIERS:
        # A prior that does not validate is not a weaker prior; it is not an answer.
        return None
    return {"state": state, "tier": tier}


def _state_reasons(
    state: str,
    *,
    tier: str,
    age: int,
    horizon: float,
    name: str,
    scoped: Sequence[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    for record in _deciding(state, tier=tier, scoped=scoped):
        label = clean_text(record.get("action", ""), limit=48) or "the surface itself"
        detail = clean_text(record.get("detail", ""), limit=MAX_DETAIL_CHARS) or "no detail recorded"
        reasons.append(f"{record.get('tier')} evidence about {label}: {detail}")
    if age >= 0:
        reasons.append(f"The deciding record is {format_duration(age)} old; the horizon is {format_duration(horizon)}.")
    if state == "ready":
        reasons.append(f"{name} was observed {'completing this action' if tier == 'used' else 'in use'} recently.")
    elif state == "blocked":
        reasons.append(f"A recording surface reported something in the way of {name}, so this cannot proceed yet.")
    elif state == "failed":
        reasons.append(f"An attempt touching {name} was observed not to happen; it stands until something newer replaces it.")
    elif state == "stale":
        reasons.append(f"Everything recorded about {name} is older than the horizon, so it says nothing about now.")
    elif tier in _NOT_OBSERVED_REASON:
        reasons.append(_NOT_OBSERVED_REASON[tier].format(name=name))
    return reasons


def _deciding(state: str, *, tier: str, scoped: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records a caller should read to understand the answer."""
    if state in ("blocked", "failed"):
        matching = [record for record in scoped if str(record.get("result", "")) == state]
    elif tier in (STALE_TIER, NONE_TIER):
        matching = sorted(scoped, key=lambda record: (str(record.get("at", "")), str(record.get("tier", ""))))
    else:
        matching = [record for record in scoped if str(record.get("tier", "")) == tier]
    return matching[-2:]


_NOT_OBSERVED_REASON: dict[str, str] = {
    NONE_TIER: "Nothing has been observed about {name}, so whether it works now is unobserved.",
    "installed": "{name} is configured locally and nothing has been observed using it, so this is unproven.",
    "host_observed": "A host reported having {name} but nothing was observed using it, so this is unproven.",
}


def _bounded(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = clean_text(value, limit=_MAX_REASON_CHARS)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) >= MAX_REASONS:
            break
    return out


@dataclass(slots=True)
class SurfaceState:
    name: str
    configured: bool
    enabled: bool
    host_ready: bool
    reason: str = ""
    detail: str = ""


class CapabilityReadiness:
    """Answers for OHAR's own surfaces from resolved surface facts plus the journal."""

    def __init__(
        self,
        *,
        approval_mode: str,
        surfaces: Sequence[SurfaceState],
        journal: ActivityJournal,
        tool_names: Sequence[str] = (),
        now: Callable[[], float] = time.time,
        stale_after_seconds: float = STALE_AFTER_SECONDS,
    ) -> None:
        self.approval_mode = clean_text(approval_mode, limit=16).lower() or "ask"
        self.journal = journal
        self._surfaces: dict[str, SurfaceState] = {}
        for state in surfaces:
            name = clean_text(getattr(state, "name", ""), limit=MAX_SURFACE_CHARS).lower()
            if _surface_family(name) is None or name in self._surfaces:
                continue
            self._surfaces[name] = state
        self._tool_names: tuple[str, ...] = tuple(
            dict.fromkeys(clean_text(item, limit=64) for item in tool_names if clean_text(item, limit=64))
        )
        self._now = now
        self._horizon = max(1.0, float(stale_after_seconds))

    @property
    def surfaces(self) -> tuple[str, ...]:
        return tuple(self._surfaces)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return self._tool_names

    def bound_tools(self, surface: str) -> list[str]:
        _, family = _surface_name(surface)
        exact, prefixes = _SURFACE_TOOLS.get(family, ((), ()))
        return [
            name
            for name in self._tool_names
            if name in exact or any(name.startswith(prefix) for prefix in prefixes)
        ]

    def answer(self, surface: str, action: str = "") -> ReadinessAnswer:
        family, name = _surface_name(surface)
        state = self._surfaces.get(name) or self._surfaces.get(family)
        configured = bool(state.configured) if state is not None else False
        moment = float(self._now())
        evidence = self._host_evidence(name, state, action=action, moment=moment)
        evidence.extend(self._journal_evidence(name, family, now=moment))
        return answer_readiness(
            surface=name,
            action=action,
            configured=configured,
            evidence=evidence,
            now=moment,
            stale_after_seconds=self._horizon,
        )

    def survey(self, *, actions: Mapping[str, Sequence[str]] | None = None) -> list[ReadinessAnswer]:
        names = self._survey_names()
        answers = [self.answer(name) for name in names]
        supplied = actions or {}
        for name in names:
            for action in list(supplied.get(name, ()) or ()):
                if len(answers) >= _MAX_SURVEY_ANSWERS:
                    return answers
                answers.append(self.answer(name, action))
        return answers[:_MAX_SURVEY_ANSWERS]

    def rows(self, answers: Sequence[ReadinessAnswer]) -> list[list[str]]:
        return [
            [
                answer.surface,
                answer.action or "any action",
                answer.state,
                answer.tier,
                format_duration(answer.age_seconds) if answer.age_seconds >= 0 else "unknown",
                clean_text(answer.next_action, limit=MAX_ACTION_CHARS),
                clean_text("; ".join(answer.reasons[:2]), limit=MAX_DETAIL_CHARS),
            ]
            for answer in answers
        ]

    def headers(self) -> list[str]:
        return ["surface", "action", "state", "evidence", "age", "next action", "why"]

    def _survey_names(self) -> list[str]:
        return list(SURFACES) + sorted(name for name in self._surfaces if name not in SURFACES)

    def _host_evidence(
        self, name: str, state: SurfaceState | None, *, action: str, moment: float
    ) -> list[dict[str, Any]]:
        """Configuration and host facts as `installed`-tier blocks: never an upgrade."""
        if state is None:
            return []
        blockers: list[str] = []
        if not state.enabled:
            blockers.append(state.reason or "the surface is disabled in configuration")
        if not state.host_ready:
            blockers.append(state.detail or state.reason or "the host reports this surface as not ready")
        if self.approval_mode == "read-only" and _is_mutating(action) and _surface_family(name) in _MUTATING_CAPABLE:
            blockers.append("approval mode is read-only, so a mutating action cannot run")
        if self._tool_names and not self.bound_tools(name):
            blockers.append("no bound tool can reach this surface")
        records: list[dict[str, Any]] = []
        stamp = _stamp(moment)
        for reason in blockers[:2]:
            try:
                records.append(
                    build_evidence(
                        surface=name,
                        action="",
                        tier="installed",
                        result="blocked",
                        at=stamp,
                        detail=reason,
                    )
                )
            except WorkflowError:
                continue
        return records

    def _journal_evidence(self, name: str, family: str, *, now: float) -> list[dict[str, Any]]:
        """What the journal observed through this surface's own tools.

        An observation with no recorded outcome is only `host_observed`: presence
        is not usability, so it can never carry the answer to `ready`.
        """
        subjects = [name, family, *self.bound_tools(name)]
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for subject in subjects:
            if not subject or subject in seen:
                continue
            seen.add(subject)
            try:
                event = self.journal.observed(subject, ok=None, now=now)
            except Exception:
                continue
            if event is None:
                continue
            outcome = getattr(event, "ok", None)
            detail = f"journal {clean_text(getattr(event, 'event', ''), limit=32) or 'activity'} for {subject}"
            session = clean_text(getattr(event, "session_id", ""), limit=48)
            if session:
                detail = f"{detail} in session {session}"
            try:
                records.append(
                    build_evidence(
                        surface=name,
                        action="",
                        tier="usable_observed" if outcome is True else "host_observed",
                        result="failed" if outcome is False else "observed",
                        at=str(getattr(event, "at", "") or ""),
                        detail=detail,
                    )
                )
            except WorkflowError:
                continue
        return records


def surfaces_from_config(
    config: Config,
    *,
    mcp_status: Mapping[str, str] | None = None,
    browser_available: bool = False,
    memory_enabled: bool = True,
) -> list[SurfaceState]:
    """Resolve the plain facts OHAR already knows into one surface list.

    `mcp_status` maps a server name to the status string the host reported
    (`ready`, `starting`, `failed`, `closed`). An absent entry is unobserved,
    never assumed ready.
    """
    web = getattr(config, "web", None)
    browser = getattr(config, "browser", None)
    memory = getattr(config, "memory", None)
    read_only = str(getattr(config, "approval", "ask") or "ask").lower() == "read-only"
    web_enabled = bool(getattr(web, "enabled", True))
    browser_enabled = bool(getattr(browser, "enabled", True))
    memory_enabled_state = bool(getattr(memory, "enabled", True)) and bool(memory_enabled)
    subagents_max = int(getattr(config, "subagents_max", 0) or 0)
    states = [
        SurfaceState(
            "files",
            True,
            True,
            True,
            "workspace files are reachable through the file tools",
            clean_text(getattr(config, "workspace", "") or "", limit=MAX_SURFACE_CHARS) or "the current workspace",
        ),
        SurfaceState(
            "shell",
            True,
            True,
            not read_only,
            "approval mode is read-only" if read_only else "",
            f"timeout {format_duration(float(getattr(config, 'shell_timeout', 120.0) or 120.0))}",
        ),
        SurfaceState(
            "web",
            web_enabled,
            web_enabled,
            web_enabled,
            "" if web_enabled else "web is disabled in configuration",
            f"search backend {clean_text(getattr(web, 'search_backend', '') or 'none', limit=32)}" if web else "",
        ),
        SurfaceState(
            "browser",
            browser_enabled,
            browser_enabled,
            bool(browser_available),
            "" if browser_available else "no chromium executable was found for browser.chromium_path",
            clean_text(getattr(browser, "chromium_path", "") or "", limit=MAX_SURFACE_CHARS),
        ),
        SurfaceState(
            "subagents",
            subagents_max > 0,
            subagents_max > 0,
            subagents_max > 0,
            "" if subagents_max > 0 else "subagents_max is 0",
            f"max {subagents_max} concurrent lanes",
        ),
        SurfaceState(
            "memory",
            bool(getattr(memory, "enabled", True)),
            memory_enabled_state,
            True,
            "" if memory_enabled_state else "memory is disabled",
            "",
        ),
        SurfaceState("workflows", True, True, True, "", "goals, gates, blocks, units, board"),
    ]
    statuses = {str(key).lower(): str(value or "") for key, value in (mcp_status or {}).items()}
    servers = list(getattr(config, "mcp_servers", []) or [])
    member_states: list[SurfaceState] = []
    for server in servers:
        server_name = clean_text(getattr(server, "name", ""), limit=48).lower()
        if not server_name:
            continue
        status = statuses.get(server_name, "")
        member_states.append(
            SurfaceState(
                f"mcp{_SERVER_SEPARATOR}{server_name}",
                True,
                bool(getattr(server, "enabled", True)),
                status == "ready",
                "" if status == "ready" else f"mcp server '{server_name}' status: {status or 'not_observed'}",
                clean_text(getattr(server, "command", "") or "", limit=MAX_SURFACE_CHARS),
            )
        )
    ready_hosts = sum(1 for state in member_states if state.host_ready)
    family_state = SurfaceState(
        "mcp",
        bool(member_states),
        any(state.enabled for state in member_states),
        ready_hosts > 0,
        ""
        if ready_hosts
        else "no mcp server reported ready by the host"
        if member_states
        else "no mcp servers are configured",
        f"{len(member_states)} server(s) declared, {ready_hosts} reported ready by the host",
    )
    states[SURFACES.index("mcp") : SURFACES.index("mcp")] = [family_state, *member_states]
    return states


def make_readiness_tool(reader: CapabilityReadiness) -> Tool:
    def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        surface = clean_text(args.get("surface", ""), limit=MAX_SURFACE_CHARS).lower()
        action = clean_text(args.get("action", ""), limit=MAX_ACTION_CHARS)
        if not surface:
            return ToolResult.failure(
                f"`surface` is required: {', '.join(SURFACES)}, or mcp:<server>.",
                reason="surface_required",
            )
        try:
            answers = reader.survey() if surface == "all" else [reader.answer(surface, action)]
        except WorkflowError as exc:
            return ToolResult.failure(str(exc), reason="surface_unknown")
        theme = make_theme("mono", color=False)
        lines = [_summary_line(answer, theme=theme) for answer in answers]
        lines.append("")
        lines.extend(strip_ansi(line) for line in table(theme, reader.headers(), reader.rows(answers)))
        for answer in answers:
            for reason in answer.reasons:
                lines.append(f"  {answer.surface}: {reason}")
        lines.append("")
        lines.extend(_wrapped(CLAIM_BOUNDARY))
        first = answers[0]
        return ToolResult.success(
            "\n".join(lines),
            surface=first.surface,
            state=first.state,
            tier=first.tier,
            age_seconds=first.age_seconds,
            next_action=first.next_action,
            count=len(answers),
        )

    return Tool(
        name="capabilities",
        description=(
            "Report whether one OHAR capability surface can be used right now: files, shell, web, browser, "
            "mcp, subagents, memory, workflows (or mcp:<server>; surface=all surveys every surface). Keeps "
            "configuration apart from observation, so a configured-only surface is never reported ready."
        ),
        parameters={
            "type": "object",
            "properties": {
                "surface": {"type": "string", "description": "Surface name, or 'all'."},
                "action": {"type": "string", "description": "Optional action to scope the answer to."},
            },
            "required": ["surface"],
            "additionalProperties": False,
        },
        handler=handler,
        read_only=True,
        destructive=False,
    )


def _summary_line(answer: ReadinessAnswer, *, theme: Theme) -> str:
    age = format_duration(answer.age_seconds) if answer.age_seconds >= 0 else "unknown age"
    body = (
        f"{answer.surface} · {answer.action or 'any action'}: {answer.state} "
        f"({answer.tier}, {age}, from {answer.state_source})"
    )
    return theme.paint(clean_text(body, limit=MAX_ACTION_CHARS), fg="accent", bold=True)


def _wrapped(text: str, *, width: int = 88) -> list[str]:
    return textwrap.wrap(clean_text(text, limit=1000), width=width)
