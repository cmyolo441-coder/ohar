"""A small, in-memory task plan with a visible update_plan tool."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .types import Tool, ToolContext, ToolResult

__all__ = ["PlanState", "PlanStep", "PlanValidationError", "make_plan_tool"]

PENDING, IN_PROGRESS, COMPLETED = "pending", "in_progress", "completed"
STATUSES = (PENDING, IN_PROGRESS, COMPLETED)
MAX_STEPS, MAX_STEP_CHARS = 20, 240

_ITEM_KEYS = frozenset({"step", "status"})
_GLYPHS = {
    "unicode": {COMPLETED: "✓", IN_PROGRESS: "▶", PENDING: "○"},
    "ascii": {COMPLETED: "[x]", IN_PROGRESS: "[>]", PENDING: "[ ]"},
}


class PlanValidationError(ValueError):
    """An invalid plan update; state is left untouched."""


@dataclass
class PlanStep:
    step: str
    status: str  # pending | in_progress | completed


def _validated(raw_steps: object) -> list[PlanStep]:
    if not isinstance(raw_steps, list):
        raise PlanValidationError("'steps' must be an array.")
    if len(raw_steps) > MAX_STEPS:
        raise PlanValidationError(f"At most {MAX_STEPS} steps are allowed.")

    steps: list[PlanStep] = []
    for number, entry in enumerate(raw_steps, 1):
        if not isinstance(entry, dict):
            raise PlanValidationError(f"Step {number} must be an object.")
        unknown = sorted(set(entry) - _ITEM_KEYS)
        if unknown:
            raise PlanValidationError(
                f"Step {number} has unknown field(s): {', '.join(unknown)}.")
        missing = [key for key in ("step", "status") if key not in entry]
        if missing:
            raise PlanValidationError(f"Step {number} is missing {', '.join(missing)}.")

        text = entry["step"]
        if not isinstance(text, str):
            raise PlanValidationError(f"Step {number} 'step' must be a string.")
        if len(text) > MAX_STEP_CHARS:
            raise PlanValidationError(
                f"Step {number} 'step' exceeds {MAX_STEP_CHARS} characters.")
        text = " ".join(text.split())
        if not text:
            raise PlanValidationError(f"Step {number} 'step' cannot be empty.")

        status = entry["status"]
        if status not in STATUSES:
            raise PlanValidationError(
                f"Step {number} has invalid status; expected one of: "
                f"{', '.join(STATUSES)}."
            )
        steps.append(PlanStep(text, status))
    return steps


@dataclass
class PlanState:
    steps: list[PlanStep] = field(default_factory=list)
    updated_at: str = ""

    @property
    def counts(self) -> dict[str, int]:
        counts = {status: 0 for status in STATUSES}
        for step in self.steps:
            if step.status in counts:
                counts[step.status] += 1
        return counts

    def update(self, raw_steps: list[dict]) -> str:
        """Validate and replace the whole plan, returning its rendering."""
        steps = _validated(raw_steps)
        self.steps = steps
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self.render()

    def render(self, *, glyphs: str = "unicode") -> str:
        marks = _GLYPHS.get(glyphs)
        if marks is None:
            raise ValueError("glyphs must be 'unicode' or 'ascii'.")
        total = len(self.steps)
        sep = "·" if glyphs == "unicode" else "-"
        if total == 0:
            return f"Plan {sep} 0 steps"
        counts = self.counts
        lines = [f"Plan {sep} {counts[COMPLETED]}/{total} completed {sep} "
                 f"{counts[IN_PROGRESS]} in progress {sep} {counts[PENDING]} pending"]
        for number, step in enumerate(self.steps, 1):
            lines.append(f"{number}. {marks[step.status]} {step.step}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        steps = [{"step": s.step, "status": s.status} for s in self.steps]
        return {"steps": steps, "updated_at": self.updated_at, "counts": self.counts}


def make_plan_tool(
    state: PlanState, on_change: Callable[[PlanState], None] | None = None
) -> Tool:
    """Build a read-only tool that updates an in-memory plan."""

    def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        extra = sorted(set(args) - {"steps"})
        if extra:
            return ToolResult.failure(
                f"Unknown argument(s): {', '.join(extra)}; state unchanged.",
                counts=dict(state.counts),
            )
        if "steps" not in args:
            return ToolResult.failure(
                "Missing required argument 'steps'; state unchanged.",
                counts=dict(state.counts),
            )
        try:
            rendered = state.update(args["steps"])
        except PlanValidationError as exc:
            return ToolResult.failure(
                f"Plan not updated: {exc}", counts=dict(state.counts)
            )

        if on_change is not None:
            try:
                on_change(state)
            except Exception:
                pass  # UI callbacks must never break the agent loop.

        return ToolResult.success(
            rendered,
            counts=dict(state.counts),
            updated_at=state.updated_at,
            total=len(state.steps),
        )

    return Tool(
        name="update_plan",
        description=(
            "Replace the visible task plan. Provide the complete list of steps "
            "with statuses pending, in_progress, or completed. Use an empty "
            "steps array to clear the plan."
        ),
        parameters={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "Complete ordered plan; empty clears it.",
                    "minItems": 0,
                    "maxItems": MAX_STEPS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_STEP_CHARS,
                                "description": "One concise action.",
                            },
                            "status": {
                                "type": "string",
                                "enum": list(STATUSES),
                                "description": "Current state of this step.",
                            },
                        },
                        "required": ["step", "status"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        },
        handler=handler,
        read_only=True,
    )
