"""The HUD's panels: status chips, the plan block, the agents/dag block, under a row budget.

Every function here returns rendered rows (or one chip line) and never more
rows than it was given. That invariant is the point: a line based UI prints
these rows into the transcript, where one row too many scrolls something away,
so the caller owns the height and the panels fit to it.

When the budget is smaller than the blocks want, Hermes' shedding order
applies: activity rows go done-first and running-last behind a `+N more` row,
then the dag tree shrinks, then the plan's window closes in around its active
item. The plan header, which is the done/total count on its own, is the last
thing besides the chips to go.

Untrusted text (plan steps, agent titles, lanes, models, errors) is stripped of
control and escape sequences and bounded before it is painted, so no child's
output can move a cursor or widen a row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..planner import COMPLETED, IN_PROGRESS, PENDING
from ..terminal import Theme, display_width, format_duration, strip_ansi
from .cells import cell_width, strip_control, truncate
from .dag import DagNode, summary_line, tree_lines
from .roster import SubagentRoster, TaskState

# The rows a block asks for before pressure; `plan_lines`/`agents_lines` default to these.
PLAN_ROWS = 8
AGENTS_ROWS = 6
DAG_ROWS = 12
# The floor each block can be shed to and still say something: its own header, or the graph's summary.
PLAN_MIN_ROWS = 1
AGENTS_MIN_ROWS = 1
DAG_MIN_ROWS = 1

ENDED_STATES = frozenset({"done", "failed", "cancelled"})
_PLAN_STATUSES = frozenset({PENDING, IN_PROGRESS, COMPLETED})
# A mode that still asks a person; anything else is said in the warning tone, and no yolo chip is drawn.
_STRICT_APPROVALS = frozenset({"", "ask", "manual", "prompt", "default", "once"})

_MARKS = {
    "unicode": {COMPLETED: "✓", IN_PROGRESS: "▶", PENDING: "○"},
    "ascii": {COMPLETED: "[x]", IN_PROGRESS: "[>]", PENDING: "[ ]"},
}
_PLAN_ROLES: dict[str, tuple[str, bool]] = {
    COMPLETED: ("dim", False),
    IN_PROGRESS: ("success", True),
    PENDING: ("text", False),
}
_AGENT_MARKS = {
    "unicode": {"running": "●", "done": "✓", "error": "✗"},
    "ascii": {"running": "o", "done": "+", "error": "x"},
}
_AGENT_ROLES: dict[str, tuple[str, bool]] = {
    "running": ("accent", True),
    "done": ("success", False),
    "error": ("error", True),
}
# The order activity rows are kept in under pressure: running last out, done first.
_ROW_RANK = {"running": 0, "error": 1, "done": 2}
_UNNAMED = "subagent"


def chips(*, approval: str, model: str, lane: str, theme: Theme, running: int = 0) -> str:
    """The HUD's status chips on one line: approval mode, model/lane, how many agents run.

    The approval mode is stated as it was chosen for this session, never as a
    yolo flag: a saved reading that lags a switch says the opposite of the
    truth, which is why the widget's yolo chip is not drawn here.
    """
    pieces: list[str] = []
    mode = _text(approval, 16)
    if mode:
        loose = mode.lower() not in _STRICT_APPROVALS
        pieces.append(theme.paint(f"[{mode}]", fg="warning" if loose else "muted", bold=loose))
    identity = " · ".join(piece for piece in (_text(model, 32), _text(lane, 16)) if piece)
    if identity:
        pieces.append(theme.paint(f"[{identity}]", fg="dim"))
    count = max(0, int(running or 0))
    if count:
        pieces.append(theme.paint(f"[{count} running]", fg="accent2", bold=True))
    return theme.paint(" │ ", fg="border").join(pieces)


def plan_lines(
    plan: Mapping[str, Any],
    width: int,
    theme: Theme,
    *,
    max_rows: int = 8,
    title: str = "Plan",
    glyphs: str = "unicode",
) -> list[str]:
    """The plan block for `PlanState.to_dict()`: a header that always stays, then items around the active one.

    At most `max_rows` rows, none wider than `width`: the window of items
    closes in on the active step one row at a time, keeping a row of what is
    behind it when it can, and the items the window leaves out are counted in a
    fold row on each side.
    """
    if width <= 0 or max_rows <= 0:
        return []
    steps = _steps(plan)
    if not steps:
        return []
    rows = [_heading(_label(title), _plan_counts(steps, glyphs), _plan_role(steps), width, theme)]
    if max_rows < 2:
        return _fitted(rows, width)
    focus = _focus(steps)
    capacity = max_rows - 1
    start, end = _window(len(steps), focus, capacity)
    while capacity > 1 and _window_rows(len(steps), start, end) > max_rows:
        capacity -= 1
        start, end = _window(len(steps), focus, capacity)
    before, after = start > 0, end < len(steps)
    if _window_rows(len(steps), start, end) > max_rows:
        if max_rows - 1 < end - start:
            return _fitted(rows, width)
        # The items fit and a fold row would not: the header's counts already say some are out of sight.
        before = after = False
    if before:
        rows.append(_fold(start, "earlier", width, theme, glyphs))
    for index in range(start, end):
        rows.append(_plan_item(index + 1, steps[index], width, theme, glyphs))
    if after:
        rows.append(_fold(len(steps) - end, "later", width, theme, glyphs))
    return _fitted(rows[:max_rows], width)


def agents_lines(
    roster: SubagentRoster,
    width: int,
    theme: Theme,
    *,
    max_rows: int = 6,
    glyphs: str = "unicode",
) -> list[str]:
    """The agents block: a header, the activity rows, and `+N more` when pressure hid any.

    Done rows are the first out and running rows the last, kept in their own
    order; the header's counts stand alone when not even one row fits.
    """
    if width <= 0 or max_rows <= 0 or roster is None:
        return []
    tasks = list(roster.tasks())
    if not tasks:
        return []
    rows = [_heading("[Agents]", _agents_counts(tasks, glyphs), _agents_role(tasks), width, theme)]
    if max_rows < 2:
        return _fitted(rows, width)
    room = max_rows - 1
    if len(tasks) > room:
        room -= 1
    kept = _kept(tasks, room)
    for index in kept:
        rows.append(_task_row(tasks[index], width, theme, glyphs))
    hidden = len(tasks) - len(kept)
    if kept and hidden:
        shown = set(kept)
        dropped = [task for index, task in enumerate(tasks) if index not in shown]
        word = "done" if all(task.status == "done" for task in dropped) else "more"
        rows.append(_muted(f"  +{hidden} {word}", width, theme))
    return _fitted(rows[:max_rows], width)


def hud_lines(
    *,
    plan: Mapping[str, Any] | None,
    roster: SubagentRoster | None,
    nodes: Sequence[DagNode] | None,
    width: int,
    budget: int,
    theme: Theme,
    approval: str = "ask",
    model: str = "",
    lane: str = "",
    glyphs: str = "unicode",
) -> list[str]:
    """The HUD's rows: never more than `budget` of them, none wider than `width` cells.

    The chips line, the plan, the agents' activity rows and the dag tree make
    up one panel; a graph that already draws every agent stands in for the flat
    rows, so an agent is never counted twice on the screen.
    """
    if width <= 0 or budget <= 0:
        return []
    hud = _Hud(
        plan=plan,
        roster=roster,
        nodes=tuple(nodes or ()),
        width=width,
        theme=theme,
        glyphs=glyphs,
        chip=chips(
            approval=approval,
            model=model,
            lane=lane,
            theme=theme,
            running=int(roster.running) if roster is not None else 0,
        ),
    )
    layout = _natural(hud)
    blocks = _render(layout, hud)
    while sum(len(block) for block in blocks) > budget and _shed(layout):
        blocks = _render(layout, hud)
    return _fitted([row for block in blocks for row in block][:budget], width)


# ── composition ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Hud:
    """What the HUD draws, resolved once so every render pass agrees."""

    plan: Mapping[str, Any] | None
    roster: SubagentRoster | None
    nodes: tuple[DagNode, ...]
    width: int
    theme: Theme
    glyphs: str
    chip: str

    @property
    def tasks(self) -> list[TaskState]:
        return list(self.roster.tasks()) if self.roster is not None else []

    @property
    def show_plan(self) -> bool:
        return bool(_steps(self.plan))

    @property
    def show_agents(self) -> bool:
        """The flat rows, unless the graph above them already draws every agent."""
        tasks = self.tasks
        if not tasks or not self.nodes:
            return bool(tasks)
        ids = {node.id for node in self.nodes}
        return not all(task.id in ids for task in tasks)


@dataclass
class _Layout:
    """The rows each block may take; 0 turns the block off."""

    chips: bool
    plan: int
    agents: int
    dag: int


def _natural(hud: _Hud) -> _Layout:
    """What each block asks for on its own, measured rather than guessed, so shedding removes real rows."""
    return _Layout(
        chips=bool(hud.chip),
        plan=len(plan_lines(hud.plan, hud.width, hud.theme, max_rows=PLAN_ROWS, glyphs=hud.glyphs))
        if hud.show_plan
        else 0,
        agents=len(agents_lines(hud.roster, hud.width, hud.theme, max_rows=AGENTS_ROWS, glyphs=hud.glyphs))
        if hud.show_agents
        else 0,
        dag=len(_dag_lines(hud.nodes, hud.width, hud.theme, DAG_ROWS, hud.glyphs)),
    )


def _render(layout: _Layout, hud: _Hud) -> list[list[str]]:
    blocks: list[list[str]] = []
    if layout.chips and hud.chip:
        blocks.append([hud.chip])
    if layout.plan:
        blocks.append(
            plan_lines(hud.plan, hud.width, hud.theme, max_rows=layout.plan, glyphs=hud.glyphs)[: layout.plan]
        )
    if layout.agents and hud.roster is not None:
        blocks.append(
            agents_lines(hud.roster, hud.width, hud.theme, max_rows=layout.agents, glyphs=hud.glyphs)[: layout.agents]
        )
    if layout.dag:
        blocks.append(_dag_lines(hud.nodes, hud.width, hud.theme, layout.dag, hud.glyphs)[: layout.dag])
    return blocks


def _shed(layout: _Layout) -> bool:
    """Take away the least valuable row; False when nothing is left to take.

    Agent rows go first, then the tree shrinks to a few rows before the plan
    does, and only then does a block itself go. The plan header and the chips
    line, each of which is the whole summary of something, are last.
    """
    if layout.agents > AGENTS_MIN_ROWS:
        layout.agents -= 1
        return True
    if layout.dag > DAG_MIN_ROWS:
        layout.dag -= 1
        return True
    if layout.plan > PLAN_MIN_ROWS:
        layout.plan -= 1
        return True
    if layout.agents:
        layout.agents = 0
        return True
    if layout.dag:
        layout.dag = 0
        return True
    if layout.plan:
        layout.plan = 0
        return True
    if layout.chips:
        layout.chips = False
        return True
    return False


def _dag_lines(nodes: Sequence[DagNode], width: int, theme: Theme, cap: int, glyphs: str) -> list[str]:
    """The graph as a tree while it runs, and as its one summary line once every node ended."""
    if cap <= 0 or not nodes:
        return []
    if all(node.status in ENDED_STATES for node in nodes):
        return [summary_line(nodes, theme)]
    return tree_lines(nodes, width, cap, theme, glyphs=glyphs)


# ── plan block ───────────────────────────────────────────────────────────────


def _steps(plan: object) -> list[tuple[str, str]]:
    """The plan's items as (text, status); a malformed entry is skipped rather than drawn."""
    source: object = plan
    to_dict = getattr(source, "to_dict", None)
    if callable(to_dict):
        source = to_dict()
    if not isinstance(source, Mapping):
        return []
    raw = source.get("steps")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return []
    steps: list[tuple[str, str]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        text = _text(entry.get("step"))
        if not text:
            continue
        status = _text(entry.get("status")) or PENDING
        steps.append((text, status if status in _PLAN_STATUSES else PENDING))
    return steps


def _focus(steps: Sequence[tuple[str, str]]) -> int:
    """The item the panel exists to show: the one running, else the first left, else the last."""
    for index, (_, status) in enumerate(steps):
        if status == IN_PROGRESS:
            return index
    for index, (_, status) in enumerate(steps):
        if status != COMPLETED:
            return index
    return max(len(steps) - 1, 0)


def _window(total: int, focus: int, capacity: int) -> tuple[int, int]:
    """`[start, end)` of at most `capacity` items that holds `focus` and, when it can, one item behind it."""
    count = min(max(total, 0), max(capacity, 1))
    if count == 0:
        return 0, 0
    last = max(total - count, 0)
    start = min(max(focus - 1, 0), last) if count > 1 else min(max(focus, 0), last)
    return start, start + count


def _window_rows(total: int, start: int, end: int) -> int:
    """Rows a window needs: the header, its items, and a fold row per hidden side."""
    folds = (1 if start > 0 else 0) + (1 if end < total else 0)
    return 1 + (end - start) + folds


def _plan_counts(steps: Sequence[tuple[str, str]], glyphs: str) -> str:
    sep = "·" if _kind(glyphs) == "unicode" else "|"
    done = sum(1 for _, status in steps if status == COMPLETED)
    active = sum(1 for _, status in steps if status == IN_PROGRESS)
    counts = f"{sep} {done}/{len(steps)} done"
    if active:
        counts += f" {sep} {active} active"
    return counts


def _plan_role(steps: Sequence[tuple[str, str]]) -> str:
    if steps and all(status == COMPLETED for _, status in steps):
        return "success"
    if any(status == IN_PROGRESS for _, status in steps):
        return "warning"
    return "dim"


def _plan_item(number: int, step: tuple[str, str], width: int, theme: Theme, glyphs: str) -> str:
    text, status = step
    marks = _MARKS[_kind(glyphs)]
    role, bold = _PLAN_ROLES.get(status, _PLAN_ROLES[PENDING])
    mark = marks.get(status, marks[PENDING])
    index = f"{number}."
    spare = max(width - 2 - cell_width(mark) - cell_width(index) - 2, 8)
    return (
        "  "
        + theme.paint(mark, fg=role, bold=bold)
        + " "
        + theme.paint(index, fg="muted")
        + " "
        + theme.paint(truncate(text, spare), fg=role)
    )


def _fold(count: int, side: str, width: int, theme: Theme, glyphs: str) -> str:
    ellipsis = "…" if _kind(glyphs) == "unicode" else "..."
    return _muted(f"  {ellipsis} {count} {side}", width, theme)


# ── agents block ─────────────────────────────────────────────────────────────


def _agents_counts(tasks: Sequence[TaskState], glyphs: str) -> str:
    sep = "·" if _kind(glyphs) == "unicode" else "|"
    running = sum(1 for task in tasks if task.status == "running")
    failed = sum(1 for task in tasks if task.status == "error")
    counts = f"{sep} {running}/{len(tasks)} running"
    if failed:
        counts += f" {sep} {failed} error"
    return counts


def _agents_role(tasks: Sequence[TaskState]) -> str:
    if any(task.status == "error" for task in tasks):
        return "error"
    if any(task.status == "running" for task in tasks):
        return "warning"
    return "success"


def _kept(tasks: Sequence[TaskState], count: int) -> list[int]:
    """The indexes of the `count` rows to draw, in their own order: running last out, done first."""
    total = len(tasks)
    if count <= 0 or not total:
        return []
    if count >= total:
        return list(range(total))
    order = sorted(range(total), key=lambda index: (_ROW_RANK.get(tasks[index].status, 1), index))
    return sorted(order[:count])


def _task_row(task: TaskState, width: int, theme: Theme, glyphs: str) -> str:
    marks = _AGENT_MARKS[_kind(glyphs)]
    role, bold = _AGENT_ROLES.get(task.status, ("text", False))
    mark = marks.get(task.status, marks["done"])
    indent, gap = 2, 2
    tail = truncate(_task_tail(task, glyphs), max(width - indent - cell_width(mark) - 1 - gap - 8, 1))
    title_cells = max(width - indent - cell_width(mark) - 1 - gap - cell_width(tail), 8)
    title = theme.paint(
        truncate(_text(task.title) or _UNNAMED, title_cells), fg="error" if task.status == "error" else "text"
    )
    return (
        " " * indent
        + theme.paint(mark, fg=role, bold=bold)
        + " "
        + title
        + " " * gap
        + theme.paint(tail, fg="muted", dim=True)
    )


def _task_tail(task: TaskState, glyphs: str) -> str:
    sep = "·" if _kind(glyphs) == "unicode" else "|"
    pieces: list[str] = []
    if task.status == "error":
        pieces.append(_text(task.error, 40) or "failed")
    pieces.extend(value for value in (_text(task.lane, 16), _text(task.model, 24)) if value)
    seconds = float(task.seconds or 0.0)
    if seconds > 0:
        pieces.append(format_duration(seconds))
    tools = tuple(task.tools or ())
    if tools:
        pieces.append(_text(tools[-1], 18) + (f" ×{len(tools)}" if len(tools) > 1 else ""))
    chars = int(task.chars or 0)
    if chars:
        pieces.append(f"{chars}c")
    return f" {sep} ".join(pieces)


# ── shared rendering helpers ─────────────────────────────────────────────────


def _heading(label: str, counts: str, role: str, width: int, theme: Theme) -> str:
    """A block's header: its label and its counts, measured as plain text so the row always fits."""
    label = truncate(label, max(width - 4, 4))
    counts = truncate(counts, max(width - cell_width(label) - 1, 4))
    return theme.paint(label, fg="accent", bold=True) + " " + theme.paint(counts, fg=role)


def _muted(text: str, width: int, theme: Theme) -> str:
    return theme.paint(truncate(text, max(width, 1)), fg="muted", dim=True)


def _label(title: str) -> str:
    return f"[{_text(title, 24) or 'Plan'}]"


def _kind(glyphs: str) -> str:
    return "ascii" if glyphs == "ascii" else "unicode"


def _text(value: object, cells: int = 0) -> str:
    """Untrusted text as one control-free line, optionally inside `cells`."""
    text = " ".join(strip_control("" if value is None else str(value)).split())
    return truncate(text, cells) if cells > 0 else text


def _bounded(row: str, cells: int) -> str:
    """A row the caller must not outstep: one too wide for its own making is re-drawn plainly."""
    return row if display_width(row) <= cells else truncate(strip_ansi(row), max(cells, 1))


def _fitted(rows: Sequence[str], cells: int) -> list[str]:
    """Every row inside `cells`, so a screen narrower than a block's own floor still cannot be overflowed."""
    return [_bounded(row, cells) for row in rows]
