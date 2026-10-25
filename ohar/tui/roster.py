"""SubagentRoster: OHAR's own `task` tool events folded into bounded per-task state.

Pure state and rendering. `apply` takes the event dicts `subagents.make_task_tool`
emits through `AgentHooks.subagent` — `{"type": "start" | "tool" | "done" | "error",
"id", "title", "lane", "model", "seconds", "chars", "error", "name"}` — and nothing
else: no file, socket or process is opened here, and nothing is ever sent to a
subagent. Unknown event types and missing or mistyped fields change nothing rather
than raising; every stored string has its control characters stripped, is collapsed
to one line and is cut to a fixed length.

The fold. A task is keyed by `id`; an event without a usable one changes nothing.
`start` (re)initialises the row — OHAR numbers tasks from 1 in every `task` call, so
a second call takes over the ids rather than piling up. `tool` trails call names in a
ring and makes a row when a `start` was missed. `done` and `error` only settle a row
that was already shown, and once a task has ended it is frozen: a repeated or late
event cannot change its verdict, its time or its trail. The roster holds
`ROSTER_CAP` tasks and pushes out a finished one, else the earliest, when full.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import time
from typing import Any

from ohar.terminal import Theme, format_duration
from ohar.types import MAX_SUBAGENTS

from .cells import cell_width, strip_control, truncate

RUNNING = "running"
DONE = "done"
ERROR = "error"

ROSTER_CAP = MAX_SUBAGENTS * 8
TOOL_TRAIL = 8
ID_CHARS = 64
TITLE_CHARS = 160
LABEL_CHARS = 48
DETAIL_CHARS = 200
MAX_SECONDS = 1_000_000_000
MAX_CHARS = 10**12

LEAD = "  "
GAP = "  "
FACT = " · "
MIN_TITLE_CELLS = 10
UNNAMED = "subagent"
UNNAMED_TOOL = "tool"

_UNICODE_MARKS = {RUNNING: "●", DONE: "✓", ERROR: "✗"}
_ASCII_MARKS = {RUNNING: "o", DONE: "+", ERROR: "x"}
_STATUS_COLORS = {RUNNING: "accent", DONE: "success", ERROR: "error"}
_RANK = {RUNNING: 0, ERROR: 1, DONE: 2}


@dataclass(slots=True)
class TaskState:
    id: str
    title: str
    status: str  # running | done | error
    lane: str = ""
    model: str = ""
    seconds: float = 0.0
    chars: int = 0
    tools: tuple[str, ...] = ()
    error: str = ""


def _one_line(value: object, limit: int) -> str:
    """A payload string as one bounded clean line; anything that is not a string is no text."""
    if not isinstance(value, str):
        return ""
    return truncate(" ".join(strip_control(value).split()), limit)


def _ident(value: object) -> str:
    """A task id as it arrived; blank, mistyped or over-long ids name no task."""
    if not isinstance(value, str):
        return ""
    text = " ".join(strip_control(value).split())
    return text if text and len(text) <= ID_CHARS else ""


def _seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= MAX_SECONDS:
        return None
    return number


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if not 0 <= value <= MAX_CHARS:
        return None
    return value


def _new_trail() -> deque[str]:
    return deque(maxlen=TOOL_TRAIL)


@dataclass(slots=True)
class _Task:
    id: str
    title: str = ""
    status: str = RUNNING
    lane: str = ""
    model: str = ""
    seconds: float = 0.0
    chars: int = 0
    tools: deque[str] = field(default_factory=_new_trail)
    error: str = ""
    started_at: float = field(default_factory=time.monotonic)

    @property
    def ended(self) -> bool:
        return self.status != RUNNING

    def view(self, now: float) -> TaskState:
        """A snapshot of this task; a running one reports the time it has been seen for."""
        return TaskState(
            id=self.id,
            title=self.title,
            status=self.status,
            lane=self.lane,
            model=self.model,
            seconds=self.seconds if self.ended else max(0.0, now - self.started_at),
            chars=self.chars,
            tools=tuple(self.tools),
            error=self.error,
        )


def _tools_text(count: int) -> str:
    return f"{count} tool{'' if count == 1 else 's'}"


def _variants(label: str, base: list[str], separator: str) -> list[str]:
    """Right-hand figures from richest to barest, deduped, always ending blank."""
    candidates = [f"{label}{separator}{base[0]}" if label else "", *base, ""]
    out: list[str] = []
    for figure in candidates:
        if figure and figure not in out:
            out.append(figure)
    out.append("")
    return out


class SubagentRoster:
    """The subagent tasks one run has reported, in the order they first reported."""

    def __init__(self) -> None:
        self._tasks: dict[str, _Task] = {}
        self.evicted = 0
        self._handlers = {
            "start": self._on_start,
            "tool": self._on_tool,
            "done": self._on_done,
            "error": self._on_error,
        }

    # ── input ──

    def apply(self, event: Mapping[str, Any]) -> bool:
        """Fold one `task` tool event; True when it changed the roster. Nothing raises."""
        if not isinstance(event, Mapping):
            return False
        kind = event.get("type")
        handler = self._handlers.get(kind) if isinstance(kind, str) else None
        if handler is None:
            return False
        task_id = _ident(event.get("id"))
        if not task_id:
            return False
        return handler(task_id, event)

    def _make_room(self) -> None:
        if len(self._tasks) < ROSTER_CAP:
            return
        evicted = next((key for key, task in self._tasks.items() if task.ended), None)
        if evicted is None:
            evicted = next(iter(self._tasks))
        del self._tasks[evicted]
        self.evicted += 1

    def _seen(self, task_id: str) -> _Task:
        task = self._tasks.get(task_id)
        if task is None:
            self._make_room()
            task = self._tasks[task_id] = _Task(id=task_id)
        return task

    def _on_start(self, task_id: str, event: Mapping[str, Any]) -> bool:
        task = self._seen(task_id)
        task.title = _one_line(event.get("title"), TITLE_CHARS)
        task.lane = _one_line(event.get("lane"), LABEL_CHARS)
        task.model = _one_line(event.get("model"), LABEL_CHARS)
        task.status = RUNNING
        task.error = ""
        task.chars = 0
        task.seconds = 0.0
        task.tools.clear()
        task.started_at = time.monotonic()
        return True

    def _on_tool(self, task_id: str, event: Mapping[str, Any]) -> bool:
        task = self._tasks.get(task_id)
        if task is None:  # a call without a `start` still gets its row
            task = self._seen(task_id)
        elif task.ended:
            return False
        task.tools.append(_one_line(event.get("name"), LABEL_CHARS) or UNNAMED_TOOL)
        return True

    def _on_done(self, task_id: str, event: Mapping[str, Any]) -> bool:
        return self._settle(task_id, event, DONE)

    def _on_error(self, task_id: str, event: Mapping[str, Any]) -> bool:
        return self._settle(task_id, event, ERROR)

    def _settle(self, task_id: str, event: Mapping[str, Any], status: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None or task.ended:  # a verdict on a task never shown has no row to land on
            return False
        task.status = status
        title = _one_line(event.get("title"), TITLE_CHARS)
        if title:
            task.title = title
        lane = _one_line(event.get("lane"), LABEL_CHARS)
        if lane:
            task.lane = lane
        model = _one_line(event.get("model"), LABEL_CHARS)
        if model:
            task.model = model
        reported = _seconds(event.get("seconds"))
        task.seconds = reported if reported is not None else max(0.0, time.monotonic() - task.started_at)
        chars = _count(event.get("chars"))
        if chars is not None:
            task.chars = chars
        task.error = (_one_line(event.get("error"), DETAIL_CHARS) or "failed") if status == ERROR else ""
        return True

    def reset(self) -> None:
        """Forget every task: a new run starts its own roster."""
        self._tasks.clear()
        self.evicted = 0

    # ── output ──

    def tasks(self) -> tuple[TaskState, ...]:
        return tuple(task.view(time.monotonic()) for task in self._tasks.values())

    def task(self, task_id: str) -> TaskState | None:
        task = self._tasks.get(task_id) if isinstance(task_id, str) else None
        return None if task is None else task.view(time.monotonic())

    @property
    def total(self) -> int:
        return len(self._tasks)

    @property
    def running(self) -> int:
        return sum(1 for task in self._tasks.values() if task.status == RUNNING)

    def snapshot(self) -> dict[str, Any]:
        views = self.tasks()
        return {
            "total": len(views),
            "running": sum(1 for view in views if view.status == RUNNING),
            "done": sum(1 for view in views if view.status == DONE),
            "error": sum(1 for view in views if view.status == ERROR),
            "evicted": self.evicted,
            "tasks": [
                {
                    "id": view.id,
                    "title": view.title,
                    "status": view.status,
                    "lane": view.lane,
                    "model": view.model,
                    "seconds": round(view.seconds, 3),
                    "chars": view.chars,
                    "tools": list(view.tools),
                    "error": view.error,
                }
                for view in views
            ],
        }

    def lines(self, width: int, theme: Theme, *, cap: int = 8, glyphs: str = "unicode") -> list[str]:
        """Compact activity rows, running first and done last, `+N more` for what `cap` has no room for.

        Each row is a status mark, the task title and right-hand figures (lane or
        model, elapsed, the latest tool call or the report size, an error's reason),
        shortened from the right until the title keeps `MIN_TITLE_CELLS`. Every row
        fits `width` as long as `width` fits the mark and one cluster.
        """
        if width <= 0 or cap <= 0 or not self._tasks:
            return []
        marks = _ASCII_MARKS if glyphs == "ascii" else _UNICODE_MARKS
        separator = " | " if glyphs == "ascii" else FACT
        ordered = sorted(self._tasks.values(), key=lambda task: _RANK.get(task.status, 3))
        shown = ordered[:cap]
        rows = [self._row(task, width, theme, marks, separator) for task in shown]
        hidden = ordered[cap:]
        if hidden:
            summary = truncate(self._collapse(hidden, separator), width)
            rows.append(theme.paint(summary, fg="muted", dim=True))
        return rows

    @staticmethod
    def _collapse(hidden: list[_Task], separator: str) -> str:
        counts = (
            (sum(1 for task in hidden if not task.ended), "running"),
            (sum(1 for task in hidden if task.status == ERROR), "failed"),
            (sum(1 for task in hidden if task.status == DONE), "done"),
        )
        said = separator.join(f"{count} {word}" for count, word in counts if count)
        return f"+{len(hidden)} more{separator}{said}" if said else f"+{len(hidden)} more"

    def _row(self, task: _Task, width: int, theme: Theme, marks: dict[str, str], separator: str) -> str:
        view = task.view(time.monotonic())
        mark = marks[view.status]
        fixed = cell_width(f"{LEAD}{mark} ")
        base: list[str] = []
        elapsed = format_duration(view.seconds)
        if view.status == ERROR:
            base = [view.error or "failed", elapsed]
        elif view.status == DONE:
            base = [f"{elapsed}{separator}{view.chars:,}c", elapsed] if view.chars else [elapsed]
        elif view.tools:
            base = [
                f"{elapsed}{separator}{view.tools[-1]}",
                f"{elapsed}{separator}{_tools_text(len(view.tools))}",
                elapsed,
            ]
        else:
            base = [elapsed]
        figures = _variants(view.model or view.lane, base, separator)
        title_room = cell_width(view.title or UNNAMED)
        needed = min(MIN_TITLE_CELLS, title_room)  # a short title does not have to hold out for a wider one
        figure = ""
        room = max(1, width - fixed)
        for candidate in figures:
            used = cell_width(f"{GAP}{candidate}") if candidate else 0
            room = width - fixed - used
            if room >= needed:
                figure = candidate
                break
        title = truncate(view.title or UNNAMED, max(1, room))
        plain = f"{LEAD}{mark} {title}"
        if cell_width(plain) > width:  # too narrow for figures at all: the mark and one cluster
            return theme.paint(truncate(plain, width), fg="muted")
        parts = [
            theme.paint(f"{LEAD}{mark}", fg=_STATUS_COLORS[view.status]),
            " ",
            theme.paint(title, fg="text" if not task.ended else "muted"),
        ]
        if figure:
            parts.append(theme.paint(f"{GAP}{figure}", fg="muted"))
        return "".join(parts)
