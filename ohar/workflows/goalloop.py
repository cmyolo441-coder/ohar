"""The loop engine over a durable goal, plus the honesty check that comes before it.

Two rules shape every line here.

* Nothing runs until the loop can name what ends it. `assess_loopability` decides
  which surface a request belongs on; `validate_assessment` refuses a
  recommendation that names a surface without naming its stop condition, because
  that condition is the only thing separating a goal loop from a schedule, a
  heartbeat, or one turn of work.
* A cycle is bounded, and its state is durable. The continuation brief is built
  from the ledger only — never the clock, never a cycle counter — so an unchanged
  ledger hashes to an unchanged brief. A loop that writes nothing down stops as
  `no_progress` instead of spending its budget re-reading the same objective, and
  a restart rehydrates the cycle count and the last digest from the journal
  rather than re-deriving them.

The turn callable is duck-typed (`TurnFn`), so this module never imports
`ohar.agent`: `agent_turn` adapts an existing agent object when a caller wants
one, and tests hand the loop a plain function.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..terminal import format_duration, format_tokens
from ..types import MAX_SUBAGENTS, ApprovalRequest, Tool, ToolContext, ToolResult
from .goals import GoalLedger, TERMINAL_GOAL_STATUSES
from .store import clean_text, now_iso, short_digest, stamp_to_epoch

if TYPE_CHECKING:
    from .blocks import BlockLog
    from .gates import DecisionGates
    from .journal import ActivityJournal
    from .verification import VerificationLedger

__all__ = [
    "LOOP_PHASES",
    "LOOPABILITY_STATES",
    "STOP_REASONS",
    "SURFACES",
    "MAX_BRIEF_CHARS",
    "LoopBudget",
    "LoopabilityAssessment",
    "TurnOutcome",
    "LoopCycle",
    "LoopReport",
    "assess_loopability",
    "validate_assessment",
    "TurnFn",
    "GoalLoop",
    "agent_turn",
    "make_loop_tool",
]

LOOP_PHASES: tuple[str, ...] = ("orient", "act", "verify", "stop")
LOOPABILITY_STATES: tuple[str, ...] = (
    "direct_task",
    "loopable",
    "needs_clarification",
    "external_wait_only",
    "north_star_only",
    "recurring",
)
STOP_REASONS: tuple[str, ...] = (
    "gate_ready",
    "budget_cycles",
    "budget_tokens",
    "budget_seconds",
    "no_progress",
    "gate_open",
    "blocked",
    "cancelled",
    "provider_error",
    "invalid_goal",
)
SURFACES: tuple[str, ...] = ("single_turn", "goal_loop", "recurring", "external_wait")

MAX_BRIEF_CHARS = 6000

MAX_CONCURRENT_LOOPS = MAX_SUBAGENTS
_MAX_CYCLES_CEILING = 32
_MAX_ASSESSMENT_CHARS = 400
_MAX_TURN_TEXT_CHARS = 4000
_MAX_NOTE_CHARS = 240
_BRIEF_ITEMS_CEILING = 16
_GATE_SCAN = 20
_UNIT_SCAN = 32
_REHYDRATE_EVENTS = 400

_JOURNAL_EVENT = "loop_cycle"
_DENIAL = "Permission denied by the operator; no cycles were run."

# Each loopability state owns exactly one surface. A mismatch is a routing bug,
# not a style choice, so `validate_assessment` refuses it.
_SURFACE_FOR_STATE: dict[str, str] = {
    "direct_task": "single_turn",
    "loopable": "goal_loop",
    "needs_clarification": "single_turn",
    "external_wait_only": "external_wait",
    "north_star_only": "single_turn",
    "recurring": "recurring",
}
# Surfaces that keep asking for work, and so owe an explicit ending mechanism.
_OPEN_ENDED_SURFACES = frozenset({"goal_loop", "recurring", "external_wait"})
# A goal_loop stop condition must name one of these; "until it is done" names none.
_END_MARKERS = (
    "criteri",
    "verif",
    "observ",
    "eviden",
    "blocker",
    "budget",
    "gate",
    "checked",
    "stops when",
    "stale",
)
# States whose answer is "not a loop yet": they must say what to do instead.
_REFRAME_REQUIRED = frozenset({"needs_clarification", "north_star_only", "external_wait_only"})
_ASSESSMENT_KEYS = frozenset({"state", "surface", "stop_condition", "reason", "reframe"})

# Stops that are an expected outcome rather than a fault: the loop did exactly
# what the ledger or the operator asked.
_CLEAN_STOPS = frozenset({"gate_ready", "gate_open", "blocked", "cancelled"})
# Stops that need somebody to act before another cycle can mean anything.
_OPERATOR_STOPS = frozenset({"gate_open", "blocked", "cancelled", "provider_error", "invalid_goal"})
_STOP_NEXT_ACTION: dict[str, str] = {
    "gate_ready": "mark_complete",
    "budget_cycles": "show_status",
    "budget_tokens": "show_status",
    "budget_seconds": "show_status",
    "no_progress": "satisfy_criterion",
    "gate_open": "answer_gate",
    "blocked": "resolve_blocker",
    "cancelled": "show_status",
    "provider_error": "show_status",
    "invalid_goal": "show_status",
}
# Only stops that are refusals of work get a blocked-work record; `no_progress`
# and `invalid_goal` are not refusals, and inventing a domain for them would
# make the block log lie.
_BLOCK_DOMAINS: dict[str, str] = {
    "budget_cycles": "budget_exhausted",
    "budget_tokens": "budget_exhausted",
    "budget_seconds": "budget_exhausted",
    "gate_open": "gate_open",
    "cancelled": "cancelled",
    "provider_error": "provider_error",
}
_TERMINAL_STOP_REASON: dict[str, str] = {
    "complete": "gate_ready",
    "cancelled": "cancelled",
    "failed": "blocked",
}
_GLYPHS: dict[str, dict[str, str]] = {
    "unicode": {"ready": "✓", "pending": "○", "ok": "✓", "fail": "✗", "sep": "·"},
    "ascii": {"ready": "[x]", "pending": "[ ]", "ok": "[ok]", "fail": "[!]", "sep": "-"},
}


@dataclass(slots=True)
class LoopBudget:
    max_cycles: int = 4
    max_total_tokens: int = 200_000
    max_seconds: float = 900.0
    no_progress_limit: int = 2

    def clamp(self) -> "LoopBudget":
        """Bound a caller-supplied budget into a runnable one. Never raises."""
        return LoopBudget(
            max_cycles=_bounded_int(self.max_cycles, 1, _MAX_CYCLES_CEILING, 4),
            max_total_tokens=_bounded_int(self.max_total_tokens, 1_000, 10_000_000, 200_000),
            max_seconds=_bounded_float(self.max_seconds, 1.0, 3600.0, 900.0),
            no_progress_limit=_bounded_int(self.no_progress_limit, 1, 8, 2),
        )


@dataclass(slots=True)
class LoopabilityAssessment:
    state: str
    surface: str
    stop_condition: str = ""
    reason: str = ""
    reframe: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "surface": self.surface,
            "stop_condition": self.stop_condition,
            "reason": self.reason,
            "reframe": self.reframe,
        }


@dataclass(slots=True)
class TurnOutcome:
    text: str
    ok: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    tool_failures: int = 0


@dataclass(slots=True)
class LoopCycle:
    index: int
    phase: str
    brief_digest: str = ""
    at: str = ""
    seconds: float = 0.0
    tokens: int = 0
    ok: bool = True
    checkpoint_id: str = ""
    criteria_satisfied: int = 0
    units_verified: int = 0
    note: str = ""


@dataclass(slots=True)
class LoopReport:
    goal_id: str = ""
    cycles: list[LoopCycle] = field(default_factory=list)
    stop_reason: str = ""
    ready: bool = False
    summary: str = ""
    next_action: str = "show_status"

    def render(self, *, glyphs: str = "unicode") -> str:
        marks = _GLYPHS.get(glyphs)
        if marks is None:
            raise ValueError("glyphs must be 'unicode' or 'ascii'.")
        sep = marks["sep"]
        work = [cycle for cycle in self.cycles if cycle.phase != "stop"]
        lines = [f"goal loop {sep} {clean_text(self.goal_id, limit=128)}"]
        flag = marks["ready"] if self.ready else marks["pending"]
        lines.append(f"{flag} stopped: {self.stop_reason or '(none)'} {sep} next: {self.next_action}")
        lines.append(
            f"{len(work)} cycle(s) {sep} {format_tokens(sum(cycle.tokens for cycle in work))} tokens "
            f"{sep} {format_duration(sum(cycle.seconds for cycle in work))}"
        )
        if self.summary:
            lines.append(f"  {clean_text(self.summary, limit=240)}")
        for cycle in self.cycles:
            mark = marks["ok"] if cycle.ok else marks["fail"]
            bits = [
                f"{cycle.index}. {cycle.phase} {mark}",
                f"digest {cycle.brief_digest or '(none)'}",
                format_duration(cycle.seconds),
                f"{format_tokens(cycle.tokens)} tok",
                f"criteria {cycle.criteria_satisfied}",
                f"units {cycle.units_verified}",
            ]
            if cycle.checkpoint_id:
                bits.append(f"ckpt {clean_text(cycle.checkpoint_id, limit=48)}")
            lines.append("  " + " ".join(bits))
            if cycle.note:
                lines.append(f"     {clean_text(cycle.note, limit=_MAX_NOTE_CHARS)}")
        return "\n".join(lines)


@runtime_checkable
class TurnFn(Protocol):
    def __call__(self, prompt: str) -> TurnOutcome: ...


def assess_loopability(
    text: str,
    *,
    criteria: Sequence[str] = (),
    verification_available: bool = False,
    externally_bounded: bool = False,
) -> LoopabilityAssessment:
    """Which surface does this request belong on, and what ends it there?

    The branches are ordered by the strength of the signal: a named cadence, a
    watched condition, or an external outcome answers "what ends this?" outright,
    and none of them is a loop. Only after those does the loop-shaped vocabulary
    get a turn.
    """
    summary = clean_text(text, limit=600)
    lowered = summary.lower()
    names = _criterion_names(criteria)
    cadence = _matched(lowered, _CADENCE_TERMS)
    watch = _matched(lowered, _WATCH_TERMS)
    external = _matched(lowered, _EXTERNAL_TERMS)
    criterion_said = _matched(lowered, _STOP_CRITERION_TERMS)
    arena = _matched(lowered, _ARENA_TERMS)
    signal = _matched(lowered, _LOOP_SIGNAL_TERMS)
    ambition = _matched(lowered, _AMBITION_TERMS) or re.search(r"\bstars?\b", lowered)
    simple = _matched(lowered, _DIRECT_TASK_TERMS)

    if not summary:
        return _assessment(
            "needs_clarification",
            reason="Nothing was named to work on, so there is no surface to choose.",
            reframe="Name the change you want and the one check that says it is done.",
        )
    if cadence:
        return _assessment(
            "recurring",
            reason=(
                f"The request names a cadence ('{cadence}'), so occurrences are triggered by "
                "a schedule rather than driven by what the last one found."
            ),
            stop_condition=_SCHEDULE_STOP,
        )
    if watch:
        return _assessment(
            "recurring",
            reason=(
                f"The request asks for '{watch}' with no end state, which is a liveness check "
                "rather than work that finishes."
            ),
            stop_condition=_HEARTBEAT_STOP,
        )
    if externally_bounded or external:
        return _assessment(
            "external_wait_only",
            reason=(
                "The outcome depends on something this workspace cannot observe"
                + (f" ('{external}')" if external else "")
                + "; cycles run against a signal that has not arrived yet would only report the wait."
            ),
            stop_condition=_EXTERNAL_STOP,
            reframe=(
                "Record the wait as a blocker on the goal, and name the local proxy signal — if any — "
                "that a loop could run against instead."
            ),
        )
    if criterion_said:
        return _assessment(
            "loopable",
            reason=(
                f"The request already states the condition that ends it ('{criterion_said}'), so it is "
                "one goal driven to a checked criterion rather than an open-ended loop."
            ),
            stop_condition=(
                f"Stops when the stated criterion is checked against observed evidence, or when a "
                f"blocker is recorded: {summary}"
            ),
        )
    if names:
        if verification_available or signal or arena or len(names) > 1:
            return _assessment(
                "loopable",
                reason=(
                    f"{len(names)} criterion(s) are declared"
                    + (" and a verification signal exists" if verification_available else "")
                    + ", so the loop has an end state to drive toward."
                ),
                stop_condition=_GOAL_LOOP_STOP,
            )
        return _assessment(
            "direct_task",
            reason="One criterion with no iterative shape: a single delivery turn is cheaper than loop state.",
            stop_condition="The requested bounded change is implemented and its targeted check passes.",
        )
    if simple and not ambition:
        return _assessment(
            "direct_task",
            reason=f"The request is a bounded change ('{simple}'), narrow enough for one turn.",
            stop_condition="The named change is done and the smallest targeted check that proves it passes.",
        )
    if ambition and not arena:
        return _assessment(
            "north_star_only",
            reason=f"'{ambition}' is an ambition, not an execution goal: no bounded arena names what a cycle would touch.",
            reframe=(
                "Convert the ambition into one bounded arena, one observable problem inside it, and one "
                "check that would show it improved."
            ),
        )
    if arena and signal:
        if verification_available:
            return _assessment(
                "loopable",
                reason=(
                    f"The request names a bounded arena ('{arena}') and an observable problem ('{signal}') "
                    "with a verification signal to check it against."
                ),
                stop_condition=_GOAL_LOOP_STOP,
            )
        return _assessment(
            "needs_clarification",
            reason=(
                f"'{arena}' and '{signal}' make the request loop-shaped, but no criterion or verification "
                "unit can tell a cycle when to stop."
            ),
            reframe=(
                "State 1-8 criteria the loop can check, or declare a verification unit, then start the loop; "
                "otherwise it ends on budget rather than on done."
            ),
        )
    if len(summary.split()) < 4 or _matched(lowered, _VAGUE_TERMS):
        return _assessment(
            "needs_clarification",
            reason="The request is too short or too vague to tell a task from a loop.",
            reframe="Name the slice to work on, the friction you can observe, and what would count as done.",
        )
    return _assessment(
        "needs_clarification",
        reason="The request has neither a bounded arena nor a verification signal, so no surface follows from it.",
        reframe=(
            "Give either a bounded slice to change in one turn, or an arena plus an observable problem and "
            "a check the loop can end on."
        ),
    )


def validate_assessment(assessment: Mapping[str, Any]) -> list[str]:
    """Errors for an assessment that names a surface without naming its end.

    Returns [] only when the recommendation could be shown to an operator
    without leaving the "what ends this?" question blank.
    """
    if not isinstance(assessment, Mapping):
        return ["assessment must be a mapping"]
    errors: list[str] = []
    keys = {key for key in assessment if isinstance(key, str)}
    missing = sorted(_ASSESSMENT_KEYS - keys)
    unsupported = sorted(keys - _ASSESSMENT_KEYS)
    if missing:
        errors.append(f"assessment is missing key(s): {', '.join(missing)}")
    if unsupported:
        errors.append(f"assessment has unsupported key(s): {', '.join(unsupported)}")

    fields: dict[str, str] = {}
    for key in _ASSESSMENT_KEYS & keys:
        value = assessment.get(key)
        if not isinstance(value, str):
            errors.append(f"assessment.{key} must be a string")
            continue
        if len(value) > _MAX_ASSESSMENT_CHARS:
            errors.append(f"assessment.{key} exceeds {_MAX_ASSESSMENT_CHARS} characters")
        fields[key] = value

    state = fields.get("state", "")
    surface = fields.get("surface", "")
    if state and state not in LOOPABILITY_STATES:
        errors.append("assessment.state is not in LOOPABILITY_STATES")
    if surface and surface not in SURFACES:
        errors.append("assessment.surface is not in SURFACES")
    if state in _SURFACE_FOR_STATE and surface in SURFACES and _SURFACE_FOR_STATE[state] != surface:
        errors.append(f"assessment.surface '{surface}' does not match state '{state}'")
    if not fields.get("reason", "").strip():
        errors.append("assessment.reason must say why this surface")

    stop = fields.get("stop_condition", "").strip()
    if surface in _OPEN_ENDED_SURFACES and not stop:
        errors.append(
            f"assessment names surface '{surface}' without naming what ends it"
        )
    if surface == "goal_loop" and stop and not any(marker in stop.lower() for marker in _END_MARKERS):
        errors.append(
            "assessment.stop_condition names no ending mechanism "
            f"({', '.join(_END_MARKERS)}); 'keep working on this' is not a stop condition"
        )
    if state in _REFRAME_REQUIRED and not fields.get("reframe", "").strip():
        errors.append(f"assessment.state '{state}' must carry a reframe: what to do instead of looping")
    return errors


def agent_turn(agent: object) -> TurnFn:
    """Adapt an agent-like object (duck-typed `run(prompt)`) to a `TurnFn`.

    Nothing from `ohar.agent` is imported: the loop and the agent package stay
    independent. Errors raised by the agent propagate, which is how the loop
    turns a failed lane into a `provider_error` stop.
    """
    run = getattr(agent, "run", None)
    if not callable(run):
        raise TypeError("agent_turn needs an object with a callable run(prompt).")

    def turn(prompt: str) -> TurnOutcome:
        if not str(prompt or "").strip():
            return TurnOutcome("(empty brief)", ok=False)
        return _turn_outcome(run(str(prompt)))

    return turn


class GoalLoop:
    """Bounded cycles over one durable goal, with every stop given a reason."""

    def __init__(
        self,
        goals: GoalLedger,
        *,
        gates: DecisionGates | None = None,
        blocks: BlockLog | None = None,
        units: VerificationLedger | None = None,
        journal: ActivityJournal | None = None,
        budget: LoopBudget | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_cycle: Callable[[LoopCycle], None] | None = None,
    ) -> None:
        if goals is None:
            raise TypeError("GoalLoop needs a GoalLedger.")
        if not callable(clock):
            raise TypeError("clock must be a callable returning seconds.")
        self.goals = goals
        self.gates = gates
        self.blocks = blocks
        self.units = units
        self.journal = journal
        self.budget = (budget or LoopBudget()).clamp()
        self.clock = clock
        self.on_cycle = on_cycle
        self._states: dict[str, _LoopState] = {}
        self._active: set[str] = set()
        self._lock = threading.RLock()

    def brief(self, goal_id: str, *, limit: int = 8) -> str:
        """The continuation prompt for one goal, deterministic in ledger state.

        No timestamps and no cycle counters appear here on purpose: the digest of
        this text is the no-progress signal, so it has to change only when the
        ledger changed.
        """
        key = _key(goal_id)
        goal = self.goals.get(key)
        items = _bounded_int(limit, 1, _BRIEF_ITEMS_CEILING, 8)
        gate = self._gate(key)
        criteria = _criteria(goal)
        pending = [criterion for criterion in criteria if _criterion_status(criterion) != "satisfied"]

        def bullets(header: str, entries: Sequence[str]) -> list[str]:
            if not entries:
                return []
            rows = [header]
            rows.extend(f"  - {clean_text(entry, limit=200)}" for entry in entries[:items])
            if len(entries) > items:
                rows.append(f"  - (+{len(entries) - items} more)")
            return rows

        lines = [
            f"CONTINUATION BRIEF {key} (status: {clean_text(getattr(goal, 'status', ''), limit=32) or 'unknown'})",
            f"objective: {clean_text(getattr(goal, 'objective', ''), limit=500)}",
            "criteria: "
            f"{len(criteria) - len(pending)}/{len(criteria)} satisfied; next action: "
            f"{_gate_value(gate, 'next_action') or clean_text(getattr(goal, 'next_action', ''), limit=48) or 'show_status'}",
        ]
        lines.extend(
            bullets(
                "missing criteria:",
                [f"{_item_id(criterion)}: {_item_summary(criterion)}" for criterion in pending],
            )
        )
        lines.extend(bullets("active blockers:", self._active_blockers(goal, gate)))
        lines.extend(
            bullets(
                "open decision gates (answer these before more cycles):",
                self._open_gate_ids(key),
            )
        )
        lines.extend(bullets("units without observed evidence:", _gate_entries(gate, "unverified_units")))
        lines.append(
            "Work one bounded step against the first missing item above, then record the checkpoint "
            "or record why you cannot. Do not restate the objective: a cycle that writes nothing down "
            "stops the loop."
        )
        return clean_text("\n".join(lines), limit=MAX_BRIEF_CHARS, single_line=False)

    def run(
        self,
        goal_id: str,
        turn: TurnFn,
        *,
        cancel: threading.Event | None = None,
        budget: LoopBudget | None = None,
    ) -> LoopReport:
        """Drive bounded cycles until the gate, a budget, or the ledger says stop."""
        if not callable(turn):
            raise TypeError("run() needs a callable turn(prompt).")
        if budget is not None and not isinstance(budget, LoopBudget):
            raise TypeError("budget must be a LoopBudget.")
        plan = self.budget if budget is None else budget.clamp()
        key = _key(goal_id)
        state = self._state_for(key)
        started = _seconds(self.clock)
        rows: list[LoopCycle] = []

        with self._lock:
            already = key in self._active
            crowded = (not already) and len(self._active) >= MAX_CONCURRENT_LOOPS
            if not (already or crowded):
                self._active.add(key)
                state.running = True
        if already:
            return self._finish(key, state, rows, "blocked", f"a loop is already running for '{key}'", started)
        if crowded:
            return self._finish(
                key, state, rows, "blocked", f"at most {MAX_CONCURRENT_LOOPS} loops run at once", started
            )

        try:
            reason, detail = self._ledger_stop(key)
            if not reason:
                for index in range(1, plan.max_cycles + 1):
                    reason, detail = self._pre_cycle_stop(key, state, plan, started, cancel)
                    if reason:
                        break
                    try:
                        prompt = self.brief(key, limit=8)
                    except (AttributeError, KeyError, TypeError, ValueError) as exc:
                        reason, detail = "invalid_goal", (
                            f"the continuation brief could not be built: {clean_text(exc, limit=160)}"
                        )
                        break
                    digest = short_digest(prompt)
                    if digest == state.last_digest:
                        state.no_progress += 1
                        if state.no_progress >= plan.no_progress_limit:
                            reason, detail = "no_progress", (
                                f"the continuation brief was unchanged for {state.no_progress} cycles"
                            )
                            break
                    else:
                        state.no_progress = 0
                        state.last_digest = digest
                    cycle, reason, detail = self._cycle(key, state, turn, plan, index, prompt, digest)
                    rows.append(cycle)
                    if reason:
                        break
                if not reason:
                    reason, detail = "budget_cycles", f"{plan.max_cycles} cycles used"
        finally:
            with self._lock:
                self._active.discard(key)
                state.running = False
        return self._finish(key, state, rows, reason, detail, started)

    def status(self, goal_id: str) -> dict[str, Any]:
        """The resumable view: what has run, where it stopped, what the gate says."""
        key = _key(goal_id)
        state = self._state_for(key)
        gate = self._gate(key)
        satisfied, total = self._criteria_counts(key)
        allowed, why = self.can_start(key)
        status = self._goal_status(key)
        with self._lock:
            payload: dict[str, Any] = {
                "goal_id": key,
                "goal_status": status,
                "running": state.running,
                "cycles": state.cycles,
                "phase": state.phase or LOOP_PHASES[0],
                "last_stop_reason": state.last_stop_reason,
                "stop_requested": state.stop_request,
                "no_progress": state.no_progress,
                "tokens": state.tokens,
                "seconds": round(state.seconds, 3),
                "last_brief_digest": state.last_digest,
                "resumed": state.resumed,
                "criteria_satisfied": satisfied,
                "criteria_total": total,
            }
        payload["ready"] = bool(getattr(gate, "ready", False))
        payload["missing_criteria"] = len(_gate_entries(gate, "missing_criteria"))
        payload["active_blockers"] = len(_gate_entries(gate, "active_blockers"))
        payload["unverified_units"] = len(_gate_entries(gate, "unverified_units"))
        payload["open_gates"] = self._open_gate_ids(key)
        payload["gate_next_action"] = _gate_value(gate, "next_action")
        payload["next_action"] = _STOP_NEXT_ACTION.get(payload["last_stop_reason"], payload["gate_next_action"] or "show_status")
        payload["can_start"] = allowed
        payload["start_refusal"] = "" if allowed else why
        return payload

    def stop(self, goal_id: str, reason: str = "cancelled") -> None:
        """Ask a running loop to end on its next cycle boundary."""
        key = _key(goal_id)
        state = self._state_for(key)
        text = clean_text(reason, limit=80)
        code = text if text in STOP_REASONS else "cancelled"
        note = (
            f"stop reason '{text}' is outside STOP_REASONS; recorded as {code}"
            if text and code != text
            else f"{code} requested"
        )
        with self._lock:
            state.stop_request = code
            state.stop_note = note

    def can_start(self, goal_id: str) -> tuple[bool, str]:
        """Refuse a loop the ledger cannot end: terminal goals, open gates, blockers, no criteria."""
        reason, detail = self._ledger_stop(_key(goal_id))
        return (not reason, detail)

    def record_cycle(self, goal_id: str, cycle: LoopCycle) -> None:
        """Journal one cycle (or one stop marker) and fold it into the resumable state."""
        key = _key(goal_id)
        state = self._state_for(key)
        stop_row = cycle.phase == "stop"
        with self._lock:
            if not stop_row:
                state.cycles += 1
                state.tokens += _bounded_int(cycle.tokens, 0, 10_000_000, 0)
                state.seconds += max(0.0, _as_float(cycle.seconds, 0.0))
            state.phase = cycle.phase if cycle.phase in LOOP_PHASES else state.phase
        self._journal_cycle(key, state, cycle)
        callback = self.on_cycle
        if callback is not None:
            try:
                callback(cycle)
            except Exception:  # a UI hook must never break the ledger
                pass

    def _cycle(
        self,
        key: str,
        state: _LoopState,
        turn: TurnFn,
        plan: LoopBudget,
        index: int,
        prompt: str,
        digest: str,
    ) -> tuple[LoopCycle, str, str]:
        started = _seconds(self.clock)
        satisfied, _total = self._criteria_counts(key)
        try:
            outcome = _turn_outcome(turn(prompt))
        except Exception as exc:  # a lane failure is a stop reason, not a crash
            note = f"turn raised {type(exc).__name__}: {clean_text(exc, limit=160)}"
            cycle = LoopCycle(
                index=index,
                phase="act",
                brief_digest=digest,
                at=now_iso(),
                seconds=round(max(0.0, _seconds(self.clock) - started), 3),
                tokens=0,
                ok=False,
                criteria_satisfied=satisfied,
                units_verified=self._units_verified(key),
                note=note,
            )
            self.record_cycle(key, cycle)
            return cycle, "provider_error", note
        seconds = round(max(0.0, _seconds(self.clock) - started), 3)
        tokens = max(0, outcome.input_tokens) + max(0, outcome.output_tokens)
        # Reconcile before checkpointing: a criterion may only be marked satisfied
        # by a unit that has just now been observed passing.
        verified = self._reconcile(key)
        checkpoint_id, note = self._record_observation(key, index, outcome)
        satisfied, _total = self._criteria_counts(key)
        cycle = LoopCycle(
            index=index,
            phase="verify",
            brief_digest=digest,
            at=now_iso(),
            seconds=seconds,
            tokens=tokens,
            ok=outcome.ok,
            checkpoint_id=checkpoint_id,
            criteria_satisfied=satisfied,
            units_verified=verified,
            note=note,
        )
        with self._lock:
            state.failures = 0 if outcome.ok else state.failures + 1
            failures = state.failures
        self.record_cycle(key, cycle)
        if failures and failures >= plan.no_progress_limit:
            return cycle, "provider_error", f"{failures} consecutive turns reported failure"
        # Re-evaluate the completion gate now, so a goal that became ready inside
        # this cycle stops as `gate_ready` rather than as a spent budget.
        reason, detail = self._ledger_stop(key)
        return cycle, reason, detail

    def _record_observation(self, key: str, index: int, outcome: TurnOutcome) -> tuple[str, str]:
        """Journal-grade note plus the checkpoint this cycle wrote."""
        said = clean_text(outcome.text or "(no text)", limit=170)
        bits = [f"cycle {index}: {said}"]
        if not outcome.ok:
            bits.append("(turn reported failure)")
        if outcome.tool_failures:
            bits.append(f"{outcome.tool_failures} tool failure(s)")
        note = clean_text(" ".join(bits), limit=_MAX_NOTE_CHARS)
        refs, evidence = self._backed_criterion_refs(key)
        if not outcome.ok:
            refs, evidence = [], []
        checkpoint_id = ""
        try:
            goal = self.goals.checkpoint(
                key,
                note,
                status="done" if outcome.ok else "failed",
                criteria_refs=refs,
                evidence_refs=evidence,
                mutation_id=f"loop:{key}:{index}",
            )
            checkpoint_id = _last_checkpoint_id(goal)
        except (AttributeError, KeyError, TypeError, ValueError):
            checkpoint_id = ""
        return checkpoint_id, note

    def _pre_cycle_stop(
        self,
        key: str,
        state: _LoopState,
        plan: LoopBudget,
        started: float,
        cancel: threading.Event | None,
    ) -> tuple[str, str]:
        if cancel is not None and cancel.is_set():
            return "cancelled", "cancellation was requested before the cycle"
        with self._lock:
            requested, note = state.stop_request, state.stop_note
        if requested:
            return requested, note or "a stop was requested"
        elapsed = max(0.0, _seconds(self.clock) - started)
        if elapsed >= plan.max_seconds:
            return "budget_seconds", f"{format_duration(elapsed)} used of {format_duration(plan.max_seconds)}"
        if state.tokens >= plan.max_total_tokens:
            return "budget_tokens", f"{format_tokens(state.tokens)} of {format_tokens(plan.max_total_tokens)} tokens used"
        return self._ledger_stop(key)

    def _ledger_stop(self, goal_id: str) -> tuple[str, str]:
        """What the ledger alone can decide, in ladder order. ('', '') means proceed."""
        key = _key(goal_id)
        try:
            goal = self.goals.get(key)
        except (AttributeError, KeyError, TypeError, ValueError):
            return "invalid_goal", f"'{key}' is not a goal in the ledger"
        status = clean_text(getattr(goal, "status", ""), limit=32)
        if status in TERMINAL_GOAL_STATUSES:
            reason = _TERMINAL_STOP_REASON.get(status, "blocked")
            return reason, f"the goal is {status}; no cycle can change that"
        criteria = _criteria(goal)
        if not criteria:
            return "invalid_goal", "the goal has no criteria, so the loop cannot name its own end"
        gate = self._gate(key)
        open_gates = self._open_gate_ids(key)
        if open_gates:
            listed = ", ".join(open_gates[:4])
            return "gate_open", f"decision gate(s) {listed} are open; answer them before another cycle"
        blockers = self._active_blockers(goal, gate)
        if blockers:
            return "blocked", f"active blocker: {clean_text(blockers[0], limit=160)}"
        if bool(getattr(gate, "ready", False)):
            return "gate_ready", "every required criterion is satisfied: the completion gate is ready"
        return "", ""

    def _finish(
        self,
        key: str,
        state: _LoopState,
        rows: list[LoopCycle],
        reason: str,
        detail: str,
        started: float,
    ) -> LoopReport:
        code = reason if reason in STOP_REASONS else "invalid_goal"
        seconds = round(max(0.0, _seconds(self.clock) - started), 3)
        gate = self._gate(key)
        satisfied, total = self._criteria_counts(key)
        ready = bool(getattr(gate, "ready", False)) or self._goal_status(key) == "complete"
        clean_detail = clean_text(detail or code, limit=_MAX_NOTE_CHARS)
        row = LoopCycle(
            index=len(rows) + 1,
            phase="stop",
            brief_digest=state.last_digest,
            at=now_iso(),
            seconds=seconds,
            tokens=0,
            ok=code in _CLEAN_STOPS,
            criteria_satisfied=satisfied,
            units_verified=self._units_verified(key),
            note=f"{code}: {clean_detail}",
        )
        with self._lock:
            state.last_stop_reason = code
            state.stop_request = ""
            state.stop_note = ""
            state.phase = "stop"
            spent = state.tokens
        # The summary describes this run; `status()` is where the durable,
        # across-restart totals live.
        worked = [item for item in rows if item.phase != "stop"]
        rows.append(row)
        self.record_cycle(key, row)
        self._record_block(key, code, clean_detail)
        summary = clean_text(
            f"{len(worked)} cycle(s) this run - {code}: {clean_detail} - criteria {satisfied}/{total} - "
            f"{format_tokens(sum(item.tokens for item in worked))} of {format_tokens(spent)} tokens - "
            f"{format_duration(seconds)}",
            limit=240,
        )
        return LoopReport(
            goal_id=key,
            cycles=rows,
            stop_reason=code,
            ready=ready,
            summary=summary,
            next_action=_STOP_NEXT_ACTION.get(code, "show_status"),
        )

    def _record_block(self, key: str, reason: str, detail: str) -> None:
        domain = _BLOCK_DOMAINS.get(reason)
        if self.blocks is None or domain is None:
            return
        try:
            self.blocks.record(
                source="goal_loop",
                reason_domain=domain,
                reason_code=reason,
                action="loop_run",
                subject=clean_text(detail, limit=160),
                outcome="blocked",
                session_id=self._session_id(key),
                goal_id=key,
            )
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            pass

    def _journal_cycle(self, key: str, state: _LoopState, cycle: LoopCycle) -> None:
        journal = self.journal
        if journal is None:
            return
        stop_row = cycle.phase == "stop"
        detail: dict[str, Any] = {
            "kind": "stop" if stop_row else "cycle",
            "index": cycle.index,
            "phase": cycle.phase,
            "digest": clean_text(cycle.brief_digest, limit=64),
            "seconds": round(max(0.0, cycle.seconds), 3),
            "tokens": max(0, cycle.tokens),
            "criteria_satisfied": max(0, cycle.criteria_satisfied),
            "units_verified": max(0, cycle.units_verified),
            "note": clean_text(cycle.note, limit=_MAX_NOTE_CHARS),
        }
        if stop_row:
            detail["reason"] = state.last_stop_reason
        try:
            journal.record(_JOURNAL_EVENT, subject=key, ok=cycle.ok, session_id=self._session_id(key), detail=detail)
        except (AttributeError, TypeError, ValueError, OSError):
            pass

    def _state_for(self, key: str) -> _LoopState:
        with self._lock:
            existing = self._states.get(key)
            if existing is not None:
                return existing
            state = self._rehydrate(key)
            self._states[key] = state
            return state

    def _rehydrate(self, key: str) -> _LoopState:
        """Rebuild the resumable view from journaled cycles instead of re-running them."""
        state = _LoopState(goal_id=key, phase="orient")
        journal = self.journal
        if journal is None:
            return state
        try:
            events = list(journal.recent(event=_JOURNAL_EVENT, subject=key, limit=_REHYDRATE_EVENTS))
        except (AttributeError, TypeError, ValueError, OSError):
            return state
        for event in sorted(events, key=lambda item: stamp_to_epoch(getattr(item, "at", "")) or 0.0):
            detail = getattr(event, "detail", None)
            if not isinstance(detail, Mapping):
                continue
            kind = str(detail.get("kind") or "")
            if kind == "cycle":
                state.cycles += 1
                state.tokens += max(0, _as_int(detail.get("tokens"), 0))
                state.seconds += max(0.0, _as_float(detail.get("seconds"), 0.0))
                digest = str(detail.get("digest") or "")
                if digest:
                    state.last_digest = digest
                phase = str(detail.get("phase") or "")
                if phase in LOOP_PHASES:
                    state.phase = phase
                state.failures = 0 if bool(detail.get("ok", True)) else state.failures + 1
            elif kind == "stop":
                reason = str(detail.get("reason") or "")
                if reason in STOP_REASONS:
                    state.last_stop_reason = reason
                state.phase = "stop"
        state.resumed = state.cycles > 0
        return state

    def _gate(self, goal_id: str) -> Any:
        key = _key(goal_id)
        payload: list[dict[str, Any]] | None = None
        if self.units is not None:
            try:
                payload = [
                    unit.to_dict()
                    for unit in self.units.units(goal_id=key, limit=_UNIT_SCAN)
                    if callable(getattr(unit, "to_dict", None))
                ]
            except (AttributeError, KeyError, TypeError, ValueError, OSError):
                payload = None
        try:
            return self.goals.gate(key, units=payload)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def _reconcile(self, goal_id: str) -> int:
        if self.units is None:
            return 0
        key = _key(goal_id)
        try:
            rows = list(self.units.reconcile_all(goal_id=key))
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            return 0
        return sum(1 for unit in rows if bool(getattr(unit, "satisfied", False)))

    def _units_verified(self, goal_id: str) -> int:
        if self.units is None:
            return 0
        key = _key(goal_id)
        try:
            rows = list(self.units.units(goal_id=key, limit=_UNIT_SCAN))
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            return 0
        return sum(1 for unit in rows if bool(getattr(unit, "satisfied", False)))

    def _backed_criterion_refs(self, goal_id: str) -> tuple[list[str], list[str]]:
        """Criteria a verification unit actually observed passing, and those units.

        A checkpoint may only satisfy a criterion when a frozen check observed it,
        so this is the sole path by which the loop claims progress on someone
        else's behalf — and the unit it cites is the evidence, never a restatement
        of the model's own claim.
        """
        if self.units is None:
            return [], []
        key = _key(goal_id)
        try:
            rows = list(self.units.units(goal_id=key, limit=_UNIT_SCAN))
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            return [], []
        refs: list[str] = []
        evidence: list[str] = []
        for unit in rows:
            if not bool(getattr(unit, "satisfied", False)):
                continue
            ref = clean_text(getattr(unit, "criterion_ref", ""), limit=64)
            if not ref:
                continue
            if ref not in refs:
                refs.append(ref)
            citation = f"unit:{_item_id(unit)}" if _item_id(unit) else ""
            if citation and citation not in evidence:
                evidence.append(citation)
        return refs[:8], evidence[:8]

    def _open_gate_ids(self, goal_id: str) -> list[str]:
        if self.gates is None:
            return []
        key = _key(goal_id)
        try:
            rows = list(self.gates.open_gates(session_id=None, limit=_GATE_SCAN))
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            return []
        return [
            clean_text(getattr(gate, "id", ""), limit=64)
            for gate in rows
            if _subject_matches(clean_text(getattr(gate, "subject", ""), limit=128), key)
        ]

    def _active_blockers(self, goal: Any, gate: Any) -> list[str]:
        entries = [_item_summary(entry) or _item_id(entry) for entry in _gate_entries(gate, "active_blockers")]
        found = [entry for entry in entries if entry]
        if found:
            return _unique(found)
        for blocker in list(getattr(goal, "blockers", None) or []):
            if clean_text(getattr(blocker, "status", ""), limit=32) != "active":
                continue
            text = _item_summary(blocker) or _item_id(blocker)
            if text:
                found.append(text)
        return _unique(found)

    def _criteria_counts(self, goal_id: str) -> tuple[int, int]:
        goal = self._goal(goal_id)
        criteria = _criteria(goal)
        if not criteria:
            return 0, 0
        return sum(1 for item in criteria if _criterion_status(item) == "satisfied"), len(criteria)

    def _goal_status(self, goal_id: str) -> str:
        return clean_text(getattr(self._goal(goal_id), "status", ""), limit=32)

    def _session_id(self, goal_id: str) -> str:
        return clean_text(getattr(self._goal(goal_id), "session_id", ""), limit=128)

    def _goal(self, goal_id: str) -> Any:
        try:
            return self.goals.get(_key(goal_id))
        except (AttributeError, KeyError, TypeError, ValueError):
            return None


@dataclass(slots=True)
class _LoopState:
    goal_id: str
    cycles: int = 0
    tokens: int = 0
    seconds: float = 0.0
    no_progress: int = 0
    failures: int = 0
    last_digest: str = ""
    phase: str = LOOP_PHASES[0]
    last_stop_reason: str = ""
    stop_request: str = ""
    stop_note: str = ""
    running: bool = False
    resumed: bool = False


def make_loop_tool(
    loop: GoalLoop,
    goals: GoalLedger,
    *,
    turn_factory: Callable[[], TurnFn],
) -> list[Tool]:
    """loop_start / loop_status / loop_stop over one GoalLoop.

    `loop_start` is the only mutating, approval-gated action; `loop_stop` is
    deliberately not gated, because refusing to stop a loop is never the safe
    direction.
    """
    if not callable(turn_factory):
        raise TypeError("turn_factory must be a callable returning a turn(prompt).")

    def start_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        unknown = sorted(set(args) - {"goal_id", "max_cycles", "allow_unloopable"})
        if unknown:
            return ToolResult.failure(f"loop_start: unknown argument(s): {', '.join(unknown)}.")
        goal_id = _arg_id(args.get("goal_id"))
        if not goal_id:
            return ToolResult.failure("loop_start: argument 'goal_id' must be a non-empty string.")
        if ctx.read_only:
            return ToolResult.failure("Read-only mode: loop cycles are disabled.", denied=True, goal_id=goal_id)
        short = clean_text(goal_id, limit=128)
        try:
            goal = goals.get(goal_id)
        except (AttributeError, KeyError, TypeError, ValueError):
            goal = None
        if goal is None:
            return ToolResult.failure(f"loop_start: '{short}' is not a goal in the ledger.")

        assessment = _assessment_for(goal, verification=_has_units(loop, goal_id))
        errors = validate_assessment(assessment.to_dict())
        if errors:
            return ToolResult.failure(
                "loop_start refused: " + "; ".join(errors), goal_id=goal_id, state=assessment.state
            )
        if assessment.state in ("direct_task", "needs_clarification") and not _arg_flag(args, "allow_unloopable"):
            return ToolResult.failure(
                f"loop_start refused ({assessment.state}): {assessment.reframe or assessment.reason} "
                "Do it in one turn, or pass allow_unloopable to loop anyway.",
                goal_id=goal_id,
                state=assessment.state,
                surface=assessment.surface,
            )
        allowed, why = loop.can_start(goal_id)
        if not allowed:
            return ToolResult.failure(f"loop_start refused: {why}", goal_id=goal_id)

        budget = loop.budget
        if "max_cycles" in args:
            cycles = _bounded_int(args.get("max_cycles"), 1, _MAX_CYCLES_CEILING, budget.max_cycles)
            budget = LoopBudget(
                max_cycles=cycles,
                max_total_tokens=budget.max_total_tokens,
                max_seconds=budget.max_seconds,
                no_progress_limit=budget.no_progress_limit,
            ).clamp()
        summary = (
            f"Run up to {budget.max_cycles} bounded cycles on goal {goal_id}: "
            f"{clean_text(getattr(goal, 'objective', ''), limit=160)}"
        )
        request = ApprovalRequest(
            action="loop_start",
            summary=clean_text(summary, limit=500),
            detail=clean_text(assessment.stop_condition, limit=2000, single_line=False),
            risk="medium",
        )
        if not ctx.request_approval(request):
            return ToolResult.failure(_DENIAL, denied=True, goal_id=goal_id)
        try:
            turn = turn_factory()
        except Exception as exc:
            return ToolResult.failure(f"loop_start: turn_factory failed: {clean_text(exc, limit=200)}")
        if not callable(turn):
            return ToolResult.failure("loop_start: turn_factory did not return a callable turn(prompt).")
        report = loop.run(goal_id, turn, budget=budget)
        meta = {
            "goal_id": report.goal_id,
            "stop_reason": report.stop_reason,
            "cycles": sum(1 for cycle in report.cycles if cycle.phase != "stop"),
            "ready": report.ready,
            "next_action": report.next_action,
        }
        if report.stop_reason in _OPERATOR_STOPS:
            return ToolResult.failure(report.render(), **meta)
        return ToolResult.success(report.render(), **meta)

    def status_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        _ = ctx
        unknown = sorted(set(args) - {"goal_id"})
        if unknown:
            return ToolResult.failure(f"loop_status: unknown argument(s): {', '.join(unknown)}.")
        goal_id = _arg_id(args.get("goal_id"))
        if not goal_id:
            return ToolResult.failure("loop_status: argument 'goal_id' must be a non-empty string.")
        payload = loop.status(goal_id)
        lines = [f"goal loop {goal_id}"]
        for key in _STATUS_ORDER:
            if key in payload:
                lines.append(f"{key.replace('_', ' ')}: {clean_text(_display(payload[key]), limit=160)}")
        return ToolResult.success("\n".join(lines), **payload)

    def stop_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        _ = ctx
        unknown = sorted(set(args) - {"goal_id", "reason"})
        if unknown:
            return ToolResult.failure(f"loop_stop: unknown argument(s): {', '.join(unknown)}.")
        goal_id = _arg_id(args.get("goal_id"))
        if not goal_id:
            return ToolResult.failure("loop_stop: argument 'goal_id' must be a non-empty string.")
        reason = clean_text(args.get("reason", "cancelled"), limit=48)
        loop.stop(goal_id, reason or "cancelled")
        payload = loop.status(goal_id)
        return ToolResult.success(
            f"stop requested for goal {goal_id} ({reason or 'cancelled'}); "
            f"running: {_display(payload['running'])}",
            goal_id=goal_id,
            reason=payload["stop_requested"],
            running=payload["running"],
        )

    goal_schema = {
        "goal_id": {
            "type": "string",
            "description": "Goal id from the goal ledger.",
            "maxLength": 128,
        }
    }
    return [
        Tool(
            name="loop_start",
            description=(
                "Run bounded cycles of work toward one goal in the goal ledger: each cycle gets a "
                "continuation brief built from the ledger, and the loop stops on a ready completion "
                "gate, an open decision gate, an active blocker, a spent budget, a brief that stops "
                "changing, or a stop request. Writes checkpoints to the goal. Requires approval."
            ),
            parameters={
                "type": "object",
                "properties": {
                    **goal_schema,
                    "max_cycles": {
                        "type": "integer",
                        "description": f"Cycles this run may use (1-{_MAX_CYCLES_CEILING}).",
                        "minimum": 1,
                        "maximum": _MAX_CYCLES_CEILING,
                    },
                    "allow_unloopable": {
                        "type": "boolean",
                        "description": "Start even when the goal reads as a direct task or needs clarification.",
                    },
                },
                "required": ["goal_id"],
                "additionalProperties": False,
            },
            handler=start_handler,
            read_only=False,
            destructive=False,
        ),
        Tool(
            name="loop_status",
            description=(
                "Read a goal's loop state: cycles run, phase, last stop reason, spent budget, and the "
                "completion gate. Nothing runs and nothing is written."
            ),
            parameters={
                "type": "object",
                "properties": goal_schema,
                "required": ["goal_id"],
                "additionalProperties": False,
            },
            handler=status_handler,
            read_only=True,
            destructive=False,
        ),
        Tool(
            name="loop_stop",
            description=(
                "Ask a goal's loop to stop at its next cycle boundary and record why. Never gated on "
                "approval, because stopping work is the safe direction."
            ),
            parameters={
                "type": "object",
                "properties": {
                    **goal_schema,
                    "reason": {
                        "type": "string",
                        "description": "A STOP_REASONS value; anything else is recorded as 'cancelled'.",
                        "maxLength": 48,
                    },
                },
                "required": ["goal_id"],
                "additionalProperties": False,
            },
            handler=stop_handler,
            read_only=False,
            destructive=False,
        ),
    ]


_STATUS_ORDER = (
    "goal_status",
    "phase",
    "cycles",
    "running",
    "last_stop_reason",
    "stop_requested",
    "no_progress",
    "last_brief_digest",
    "tokens",
    "seconds",
    "resumed",
    "criteria_satisfied",
    "criteria_total",
    "ready",
    "missing_criteria",
    "active_blockers",
    "unverified_units",
    "open_gates",
    "gate_next_action",
    "next_action",
    "can_start",
    "start_refusal",
)

_SCHEDULE_STOP = (
    "Stops when the operator removes the schedule, or its declared failure policy pauses it after the "
    "named repeat count; between occurrences nothing runs."
)
_HEARTBEAT_STOP = (
    "Stops when the watched condition changes and is delivered, or the operator withdraws the watch; "
    "a heartbeat has no completion of its own."
)
_EXTERNAL_STOP = (
    "Stops when the external evidence is observed and recorded, or when the wait goes stale and is "
    "converted into a blocker."
)
_GOAL_LOOP_STOP = (
    "Stops when every required criterion is satisfied with recorded evidence and its verification unit "
    "is observed passing, or when a blocker is recorded; the cycle, token, and second budgets also end it."
)

_CADENCE_TERMS = (
    "every morning", "every day", "every night", "every hour", "every week", "every monday",
    "every weekday", "each morning", "daily", "nightly", "hourly", "weekly", "monthly",
    "on a schedule", "cron", "periodically", "twice a day",
)
_WATCH_TERMS = (
    "keep watching", "keep monitoring", "watch continuously", "monitor continuously",
    "keep an eye on", "heartbeat", "liveness check", "health check", "notify me when",
    "let me know when it changes", "tell me when it changes",
)
_STOP_CRITERION_TERMS = (
    "until it passes", "until the tests pass", "until tests pass", "until the suite passes",
    "until all checks pass", "until ci is green", "until the build is green",
    "until every test passes", "when the suite passes",
)
_EXTERNAL_TERMS = (
    "market response", "user response", "community response", "adoption", "viral",
    "waiting for", "wait until", "until they reply", "when they respond", "vendor response",
    "legal review", "upstream review",
)
_AMBITION_TERMS = (
    "world-class", "best in class", "best-in-class", "market leader", "unicorn", "revenue",
    "100k", "10k", "1m users", "million users", "github star", "famous", "popular",
)
_ARENA_TERMS = (
    "repo", "repository", "readme", "docs", "documentation", "install", "installation",
    "onboarding", "first-run", "first run", "workflow", "module", "package", "prototype",
    "mvp", "cli", "api", "test suite", "this project", "the app",
)
_LOOP_SIGNAL_TERMS = (
    "verify", "verification", "test", "tests", "smoke", "evidence", "acceptance", "criteria",
    "check", "measure", "find", "discover", "investigate", "iterate", "improve", "reduce",
    "fix", "fail", "failing", "flaky", "friction", "slow", "confusing", "bug", "gap", "broken", "risk",
)
_DIRECT_TASK_TERMS = (
    "rename", "typo", "spelling", "one-line", "one line", "single file", "button color",
    "start the dev server", "restart the dev server", "run the tests", "bump the version",
    "add a comment", "fix the link",
)
_VAGUE_TERMS = ("make it better", "improve this", "clean it up", "refactor everything", "modernize")


def _assessment(
    state: str,
    *,
    reason: str,
    stop_condition: str = "",
    reframe: str = "",
) -> LoopabilityAssessment:
    return LoopabilityAssessment(
        state=state,
        surface=_SURFACE_FOR_STATE[state],
        stop_condition=clean_text(stop_condition, limit=_MAX_ASSESSMENT_CHARS),
        reason=clean_text(reason, limit=_MAX_ASSESSMENT_CHARS),
        reframe=clean_text(reframe, limit=_MAX_ASSESSMENT_CHARS),
    )


def _matched(text: str, terms: Iterable[str]) -> str:
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", text):
            return term
    return ""


def _criterion_names(criteria: Sequence[str]) -> list[str]:
    names: list[str] = []
    for item in criteria or ():
        text = _item_summary(item) if isinstance(item, Mapping) else clean_text(item, limit=240)
        if text and text not in names:
            names.append(text)
    return names[:8]


def _turn_outcome(value: object) -> TurnOutcome:
    """Normalize whatever a turn callable returns into a TurnOutcome."""
    if isinstance(value, TurnOutcome):
        return TurnOutcome(
            text=clean_text(value.text, limit=_MAX_TURN_TEXT_CHARS, single_line=False),
            ok=bool(value.ok),
            input_tokens=max(0, _as_int(value.input_tokens, 0)),
            output_tokens=max(0, _as_int(value.output_tokens, 0)),
            tool_failures=max(0, _as_int(value.tool_failures, 0)),
        )
    if value is None:
        return TurnOutcome("(empty turn)", ok=False)
    if isinstance(value, str):
        return TurnOutcome(clean_text(value, limit=_MAX_TURN_TEXT_CHARS, single_line=False))
    if isinstance(value, Mapping):
        return TurnOutcome(
            text=clean_text(value.get("text", ""), limit=_MAX_TURN_TEXT_CHARS, single_line=False),
            ok=bool(value.get("ok", True)),
            input_tokens=max(0, _as_int(value.get("input_tokens"), 0)),
            output_tokens=max(0, _as_int(value.get("output_tokens"), 0)),
            tool_failures=max(0, _as_int(value.get("tool_failures"), 0)),
        )
    text = getattr(value, "text", None)
    if text is None:
        return TurnOutcome(clean_text(value, limit=_MAX_TURN_TEXT_CHARS, single_line=False), ok=False)
    return TurnOutcome(
        text=clean_text(text, limit=_MAX_TURN_TEXT_CHARS, single_line=False),
        ok=bool(getattr(value, "ok", True)),
        input_tokens=max(0, _as_int(getattr(value, "input_tokens", 0), 0)),
        output_tokens=max(0, _as_int(getattr(value, "output_tokens", 0), 0)),
        tool_failures=max(0, _as_int(getattr(value, "tool_failures", 0), 0)),
    )


def _assessment_for(goal: Any, *, verification: bool) -> LoopabilityAssessment:
    criteria = [
        _item_summary(criterion)
        for criterion in _criteria(goal)
        if _item_summary(criterion)
    ]
    return assess_loopability(
        clean_text(getattr(goal, "objective", ""), limit=600),
        criteria=criteria,
        verification_available=verification,
    )


def _has_units(loop: GoalLoop, goal_id: str) -> bool:
    if loop.units is None:
        return False
    try:
        return bool(list(loop.units.units(goal_id=goal_id, limit=_UNIT_SCAN)))
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return False


def _subject_matches(subject: str, goal_id: str) -> bool:
    if not subject or not goal_id:
        return False
    if subject == goal_id:
        return True
    if subject.startswith("goal:"):
        return subject[len("goal:"):] == goal_id
    return subject.split(":")[-1] == goal_id


def _gate_entries(gate: Any, name: str) -> list[str]:
    entries: list[str] = []
    for entry in list(getattr(gate, name, None) or []):
        text = _item_summary(entry) if isinstance(entry, Mapping) else clean_text(entry, limit=200)
        text = text or _item_id(entry)
        if text:
            entries.append(text)
    return _unique(entries)


def _gate_value(gate: Any, name: str) -> str:
    return clean_text(getattr(gate, name, ""), limit=48)


def _criteria(goal: Any) -> list[Any]:
    if goal is None:
        return []
    return [criterion for criterion in list(getattr(goal, "criteria", None) or []) if criterion is not None]


def _criterion_status(criterion: Any) -> str:
    return clean_text(getattr(criterion, "status", ""), limit=32)


def _item_id(item: Any) -> str:
    if isinstance(item, Mapping):
        return clean_text(item.get("id", ""), limit=64)
    return clean_text(getattr(item, "id", ""), limit=64)


def _item_summary(item: Any) -> str:
    if isinstance(item, Mapping):
        return clean_text(item.get("summary", item.get("statement", "")), limit=200)
    return clean_text(getattr(item, "summary", getattr(item, "statement", "")), limit=200)


def _last_checkpoint_id(goal: Any) -> str:
    rows = list(getattr(goal, "checkpoints", None) or [])
    if not rows:
        return ""
    return _item_id(rows[-1])


def _key(goal_id: object) -> str:
    return clean_text(goal_id, limit=128)


def _arg_id(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _key(value)


def _arg_flag(args: Mapping[str, Any], name: str) -> bool:
    value = args.get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _display(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_display(item) for item in value) or "(none)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "(none)"
    return str(value)


def _unique(entries: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in entries:
        if entry and entry not in seen:
            seen.add(entry)
            ordered.append(entry)
    return ordered


def _seconds(clock: Callable[[], float]) -> float:
    try:
        return float(clock())
    except (TypeError, ValueError):
        return time.monotonic()


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool) or isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value) if value == value else default
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _as_float(value: object, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _bounded_int(value: object, minimum: int, maximum: int, default: int) -> int:
    if isinstance(value, bool):
        return default
    number = _as_int(value, default)
    return max(minimum, min(maximum, number))


def _bounded_float(value: object, minimum: float, maximum: float, default: float) -> float:
    if isinstance(value, bool):
        return default
    number = _as_float(value, default)
    return max(minimum, min(maximum, number))
