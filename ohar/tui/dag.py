"""Task-graph widgets: topological layering, a folding tree, and a transcript block.

Ports the Hermes TUI's dag_tree and dag_block to OHAR's own data: a roster's
subagent tasks (`dag_nodes`) or any list of `DagNode`. Rows are composed from
plain pieces and painted piece by piece, so no row is ever cut mid escape.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ohar.terminal import Theme, format_duration, format_tokens, strip_ansi

from .cells import cell_width, clip_line, strip_control, truncate
from .roster import SubagentRoster

__all__ = [
    "CONDENSED_BOXES",
    "CONDENSED_FAILURES",
    "SMALL_GRAPH_LANES",
    "DagNode",
    "FoldState",
    "block_lines",
    "dag_nodes",
    "fold_keys",
    "layers",
    "summary_line",
    "tree_lines",
]

OPEN = "▾ "
SHUT = "▸ "
LEAF = "  "
DETAIL_LEAD = "› "
STEP = 4  # cells a node's detail rows and the nodes nested under it sit right of its triangle
INDENT = 2  # the tree's left margin, in cells
INDENT_STR = "  "
LAYER_KEY = "layer:"
FACT = " · "
TITLE_CAP = 52  # a node title's cells at most, so the figure columns keep room on a wide terminal
MIN_TITLE_CELLS = 10
SMALL_GRAPH_LANES = 12  # up to this many nodes a block prints the graph whole; past it, condensed
CONDENSED_BOXES = 6  # the frontier nodes a condensed print draws boxes for
CONDENSED_FAILURES = 8  # the failed or stuck nodes a condensed end lists
CONDENSED_LAYERS = 8  # the layer rows a condensed print lists before it counts the rest
MORE_NOTE = "/dag shows the whole graph"
BOX_GAP = 6  # the smallest box interior that reads as a box; below it the block names its nodes
OPEN_STATES = frozenset({"running", "stuck", "failed"})  # open by default, and what a condensed end lists

STATES = ("pending", "running", "done", "failed", "stuck", "cancelled")
STATE_ORDER = ("running", "stuck", "failed", "pending", "cancelled", "done")
_TASK_STATES = {"running": "running", "done": "done", "error": "failed"}
_ALIASES = {
    "queued": "pending",
    "waiting": "pending",
    "start": "running",
    "started": "running",
    "in_progress": "running",
    "ok": "done",
    "success": "done",
    "complete": "done",
    "completed": "done",
    "error": "failed",
    "canceled": "cancelled",
    "timeout": "stuck",
    "stale": "stuck",
    "blocked": "stuck",
}
_GLYPHS: dict[str, dict[str, str]] = {
    "unicode": {"pending": "○", "running": "▶", "done": "✓", "failed": "✗", "stuck": "⊘", "cancelled": "×"},
    "ascii": {"pending": "-", "running": ">", "done": "+", "failed": "x", "stuck": "!", "cancelled": "c"},
}
_ROLES = {
    "pending": "muted",
    "running": "accent",
    "done": "success",
    "failed": "error",
    "stuck": "warning",
    "cancelled": "dim",
}
_BOX = {"top": "┌", "end": "┐", "foot": "└", "close": "┘", "rule": "─", "side": "│"}
_BOX_ASCII = {"top": "+", "end": "+", "foot": "+", "close": "+", "rule": "-", "side": "|"}
_TRIANGLES = {"unicode": (OPEN, SHUT, LEAF), "ascii": ("v ", "> ", "  ")}
_DETAILS = {"unicode": DETAIL_LEAD, "ascii": ": "}
_DROP_ORDER = ((0, 1, 2, 3), (0, 1, 2), (1, 2), (2,), ())


@dataclass(slots=True)
class DagNode:
    """One task of a task graph, with the parent edges that order it."""

    id: str
    title: str
    status: str  # pending | running | done | failed | stuck | cancelled
    parents: tuple[str, ...] = ()
    lane: str = ""
    model: str = ""
    seconds: float = 0.0
    tokens: int = 0
    detail: str = ""


class FoldState:
    """Which rows are open: one entry per row a person folded.

    `is_open` answers for a row nothing was set on with the caller's default, which is how the tree keeps its
    state defaults (running work open, settled work shut) without storing them. `locked` tells the tree which
    rows may not be shut to fit, including one deliberately left open.
    """

    def __init__(self) -> None:
        self._open: dict[str, bool] = {}

    def toggle(self, key: str) -> None:
        self._open[key] = not self.is_open(key, True)  # a row never folded reads as open

    def is_open(self, key: str, default: bool) -> bool:
        return self._open.get(key, default)

    def override(self, key: str, open_: bool) -> None:
        self._open[key] = bool(open_)

    def locked(self, key: str) -> bool:
        return key in self._open


@dataclass(frozen=True, slots=True)
class _Columns:
    """Where a node row's parts go: the title column's cells, then which figure columns show and how wide."""

    title: int
    shown: tuple[int, ...]
    widths: tuple[int, ...]


@dataclass(slots=True)
class _Row:
    """One tree row before it is drawn: layer, node, detail, more, or blank."""

    kind: str
    key: str = ""
    node: DagNode | None = None
    depth: int = 1
    is_open: bool = True
    foldable: bool = False
    text: str = ""
    hidden: int = 0  # the nodes this row stands for, or hides by being shut


@dataclass(slots=True)
class _Graph:
    """The drawn shape of a node list: its layers, each node nested under its first earlier-layer parent."""

    nodes: dict[str, DagNode] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    layers: list[list[str]] = field(default_factory=list)
    kids: dict[str, list[str]] = field(default_factory=dict)
    roots: dict[int, list[str]] = field(default_factory=dict)

    def subtree(self, node_id: str) -> list[str]:
        found: list[str] = []
        for kid in self.kids.get(node_id, ()):
            found.append(kid)
            found += self.subtree(kid)
        return found

    def drawn(self, at: int) -> list[str]:
        """The layer's own count as the tree shows it: its root nodes and everything nested under them."""
        found: list[str] = []
        for node_id in self.roots.get(at, ()):
            found.append(node_id)
            found += self.subtree(node_id)
        return found

    def active(self, node_id: str) -> bool:
        """Whether running, stuck, or failed work is this node or nested under it."""
        targets = (node_id, *self.subtree(node_id))
        return any(_state(self.nodes[each].status) in OPEN_STATES for each in targets)


def _state(status: str) -> str:
    key = str(status or "").strip().lower()
    if key in STATES:
        return key
    return _ALIASES.get(key, "pending")


def _flat(text: object) -> str:
    """Untrusted text as one clean row fragment: ANSI and control characters out, whitespace collapsed."""
    return " ".join(strip_control(strip_ansi(str(text or ""))).split())


def _marks(glyphs: str) -> dict[str, str]:
    return _GLYPHS.get(glyphs, _GLYPHS["unicode"])


def layers(nodes: Sequence[DagNode]) -> list[list[DagNode]]:
    """Topological layers, every node after its known parents; a cycle opens at its least-blocked node."""
    by_id: dict[str, DagNode] = {}
    ids: list[str] = []
    for node in nodes:
        if node.id in by_id:
            continue
        by_id[node.id] = node
        ids.append(node.id)
    waiting = {key: {parent for parent in by_id[key].parents if parent in by_id and parent != key} for key in ids}
    rank = {key: at for at, key in enumerate(ids)}
    remaining = list(ids)
    out: list[list[DagNode]] = []
    while remaining:
        ready = [key for key in remaining if not waiting[key]]
        if not ready:
            ready = [min(remaining, key=lambda key: (len(waiting[key]), rank[key]))]
        placed = set(ready)
        remaining = [key for key in remaining if key not in placed]
        for key in remaining:
            waiting[key] -= placed
        out.append([by_id[key] for key in ready])
    return out


def dag_nodes(roster: SubagentRoster) -> list[DagNode]:
    """The roster's tasks as graph nodes: one lane each, its last tool call or error as its detail line."""
    nodes: list[DagNode] = []
    for task in roster.tasks():
        ident = _flat(task.id) or "task"
        detail = task.error or (task.tools[-1] if task.tools else "")
        nodes.append(
            DagNode(
                id=ident,
                title=_flat(task.title) or ident,
                status=_TASK_STATES.get(str(task.status), str(task.status)),
                lane=_flat(task.lane),
                model=_flat(task.model),
                seconds=float(task.seconds or 0.0),
                tokens=int(task.chars or 0),
                detail=_flat(detail),
            )
        )
    return nodes


def _locked(fold: FoldState | None, key: str) -> bool:
    """Whether the person set this row, so fitting may not change it."""
    return fold is not None and fold.locked(key)


def _graph(nodes: Sequence[DagNode]) -> _Graph:
    graph = _Graph()
    depth_of: dict[str, int] = {}
    for at, layer in enumerate(layers(nodes)):
        fresh: list[str] = []
        for node in layer:
            depth_of[node.id] = at
            if node.id not in graph.nodes:
                graph.nodes[node.id] = node
                graph.order.append(node.id)
                fresh.append(node.id)
        graph.layers.append(fresh)
    for node_id in graph.order:
        depth = depth_of[node_id]
        edges = [str(each) for each in graph.nodes[node_id].parents if each and str(each) != node_id]
        parent = next((each for each in edges if depth_of.get(each, depth) < depth), "")
        if parent:
            graph.kids.setdefault(parent, []).append(node_id)
        else:
            graph.roots.setdefault(depth, []).append(node_id)
    return graph


def _figures(node: DagNode, glyphs: str) -> tuple[str, str, str, str]:
    """A node's figure columns: lane, model, state with its time, tokens."""
    state = _state(node.status)
    status = f"{state} {format_duration(node.seconds)}" if node.seconds else state
    arrow = "d" if glyphs == "ascii" else "↓"
    tokens = f"{arrow}{format_tokens(node.tokens)}" if node.tokens > 0 else ""
    return _flat(node.lane), _flat(node.model), status, tokens


def _open_defaults(graph: _Graph, fold: FoldState | None) -> dict[str, bool]:
    """The rows open before anyone folds: every layer, and a node holding moving work under it."""
    expanded: dict[str, bool] = {}
    for at in graph.roots:
        expanded[f"{LAYER_KEY}{at}"] = True if fold is None else fold.is_open(f"{LAYER_KEY}{at}", True)
    for node_id in graph.order:
        node = graph.nodes[node_id]
        foldable = bool(graph.kids.get(node_id)) or bool(node.detail)
        default = graph.active(node_id)
        expanded[node_id] = foldable and (default if fold is None else fold.is_open(node_id, default))
    return expanded


def _layer_text(graph: _Graph, at: int, ids: Sequence[str]) -> str:
    """`layer 2 · 4 nodes · 1 running · 3 done` for the nodes `ids` names."""
    counts: dict[str, int] = {}
    for node_id in ids:
        state = _state(graph.nodes[node_id].status)
        counts[state] = counts.get(state, 0) + 1
    total = len(ids)
    said = [f"layer {at + 1}", f"{total} node{'' if total == 1 else 's'}"]
    said += [f"{counts[state]} {state}" for state in STATE_ORDER if counts.get(state)]
    return FACT.join(said)


def _specs(graph: _Graph, expanded: dict[str, bool]) -> list[_Row]:
    """The rows the tree draws for these open keys, top to bottom."""
    rows: list[_Row] = []
    drawn = [at for at in sorted(graph.roots) if graph.roots[at]]

    def node_rows(node_id: str, depth: int) -> None:
        node = graph.nodes[node_id]
        kids = graph.kids.get(node_id, ())
        nested = graph.subtree(node_id)
        is_open = bool(expanded.get(node_id))
        rows.append(
            _Row(
                "node",
                node_id,
                node=node,
                depth=depth,
                is_open=is_open,
                foldable=bool(kids) or bool(node.detail),
                hidden=0 if is_open else len(nested),
            )
        )
        if not is_open:
            said = [f"{len(nested)} more nodes"] if kids else []
            if node.detail:
                said.append("1 detail line")
            if said:
                rows.append(_Row("more", node_id, node=node, depth=depth, text=FACT.join(said)))
            return
        if node.detail:
            rows.append(_Row("detail", node_id, node=node, depth=depth))
        for kid in kids:
            node_rows(kid, depth + 1)

    for place, at in enumerate(drawn):
        key = f"{LAYER_KEY}{at}"
        is_open = bool(expanded.get(key, True))
        if place and (is_open or bool(expanded.get(f"{LAYER_KEY}{drawn[place - 1]}", True))):
            rows.append(_Row("blank"))  # open layers are read as separate lists; shut ones stack
        rows.append(
            _Row(
                "layer",
                key,
                depth=0,
                is_open=is_open,
                foldable=True,
                text=_layer_text(graph, at, graph.drawn(at)),
                hidden=0 if is_open else len(graph.drawn(at)),
            )
        )
        if not is_open:
            continue
        for node_id in graph.roots[at]:
            node_rows(node_id, 1)
    return rows


def _fit(graph: _Graph, fold: FoldState | None, cap: int) -> dict[str, bool]:
    """The open keys, shut until the tree fits `cap` rows; rows the person set are never touched."""
    expanded = _open_defaults(graph, fold)
    if len(_specs(graph, expanded)) <= cap:
        return expanded
    quiet = [
        f"{LAYER_KEY}{at}"
        for at in sorted(graph.roots)
        if not _locked(fold, f"{LAYER_KEY}{at}")
        and expanded.get(f"{LAYER_KEY}{at}")
        and not any(graph.active(node_id) for node_id in graph.drawn(at))
    ]

    def settled(key: str) -> bool:
        at = int(key[len(LAYER_KEY):])
        return all(_state(graph.nodes[each].status) == "done" for each in graph.drawn(at))

    quiet.sort(key=lambda key: (not settled(key), -len(graph.drawn(int(key[len(LAYER_KEY):])))))
    for key in quiet:
        expanded[key] = False
        if len(_specs(graph, expanded)) <= cap:
            return expanded
    for node_id in reversed(graph.order):
        if _locked(fold, node_id) or not expanded.get(node_id):
            continue
        if any(_state(graph.nodes[kid].status) in OPEN_STATES for kid in graph.kids.get(node_id, ())):
            continue  # a node over moving work stays open: shutting it would hide that work
        expanded[node_id] = False
        if len(_specs(graph, expanded)) <= cap:
            return expanded
    return expanded


def _covers(row: _Row) -> int:
    """How many nodes a row stands for: an open node itself, a shut one everything beneath it."""
    if row.kind == "node":
        return 1 if row.is_open else row.hidden + 1
    if row.kind == "layer":
        return 0 if row.is_open else row.hidden
    return 0


def _lead(row: _Row) -> str:
    return " " * (INDENT + STEP * (row.depth - 1))


def _triangle(row: _Row, glyphs: str) -> str:
    open_, shut, leaf = _TRIANGLES.get(glyphs, _TRIANGLES["unicode"])
    return (open_ if row.is_open else shut) if row.foldable else leaf


def _node_row(row: _Row) -> DagNode:
    return row.node if row.node is not None else DagNode(id="", title="", status="pending")


def _prefix_cells(row: _Row, glyphs: str) -> int:
    return cell_width(_lead(row)) + 2 + cell_width(_marks(glyphs)[_state(_node_row(row).status)]) + 1


def _columns(rows: Sequence[_Row], width: int, glyphs: str) -> _Columns:
    """Node rows share a title column, then figure columns of equal width; whole columns drop from the right
    (tokens, lane, model) before the title gives way."""
    nodes = [row for row in rows if row.kind == "node"]
    if not nodes:
        return _Columns(0, (), ())
    prefixes = [_prefix_cells(row, glyphs) for row in nodes]
    titles = [cell_width(_flat(_node_row(row).title) or "untitled") for row in nodes]
    parts = [_figures(_node_row(row), glyphs) for row in nodes]
    widest = min(max(prefix + text for prefix, text in zip(prefixes, titles)), max(prefixes) + TITLE_CAP)
    room, shown, widths = widest, (), ()
    for candidate in _DROP_ORDER:
        shown = tuple(index for index in candidate if any(part[index] for part in parts))
        widths = tuple(max(cell_width(part[index]) for part in parts) for index in shown)
        block = sum(widths) + cell_width(FACT) * max(len(widths) - 1, 0)
        space = width - (2 + block if widths else 0)
        room = max(min(widest, space), MIN_TITLE_CELLS)
        if not widths or space >= max(prefixes) + MIN_TITLE_CELLS:
            break
    return _Columns(room, shown, widths)


def _pieces(row: _Row, width: int, columns: _Columns, glyphs: str) -> list[tuple[str, str]]:
    """The row's plain pieces with their palette roles, every piece already within `width`."""
    if row.kind == "blank":
        return []
    if row.kind == "layer":
        return [(INDENT_STR, ""), (_triangle(row, glyphs), "muted"), (row.text, "muted")]
    if row.kind == "more":
        shut = _TRIANGLES.get(glyphs, _TRIANGLES["unicode"])[1]
        return [(" " * (INDENT + STEP * row.depth), ""), (shut, "muted"), (row.text, "muted")]
    if row.kind == "detail":
        lead = " " * (INDENT + STEP * row.depth)
        mark = _DETAILS.get(glyphs, _DETAILS["unicode"])
        room = max(width - cell_width(lead) - cell_width(mark), 1)
        return [(lead, ""), (mark, "muted"), (truncate(_flat(_node_row(row).detail), room), "dim")]
    node = _node_row(row)
    state = _state(node.status)
    lead = _lead(row)
    glyph = _marks(glyphs)[state] + " "
    room = max(columns.title - cell_width(lead) - 2 - cell_width(glyph), 1)
    title = clip_line(truncate(_flat(node.title) or "untitled", room), room)
    pieces: list[tuple[str, str]] = [
        (lead, ""), (_triangle(row, glyphs), "muted"), (glyph, _ROLES[state]), (title, "text")
    ]
    parts = _figures(node, glyphs)
    shown = [parts[index] for index in columns.shown]
    while shown and not shown[-1]:
        shown.pop()  # no trailing separator for a column this node has nothing in
    figures = FACT.join(
        clip_line(text, columns.widths[at]) if at < len(shown) - 1 else text for at, text in enumerate(shown)
    )
    if figures:
        pieces.append((" " * max(room - cell_width(title), 0) + f"  {figures}", "muted"))
        if cell_width("".join(text for text, _role in pieces)) > width:
            pieces.pop()  # the figures give way before a title would be split
    return pieces


def _clip_pieces(pieces: Sequence[tuple[str, str]], width: int) -> list[tuple[str, str]]:
    """The row's pieces cut to `width` cells: whole pieces while they fit, then the last partial one."""
    out: list[tuple[str, str]] = []
    room = width
    for text, role in pieces:
        cells = cell_width(text)
        if cells <= room:
            out.append((text, role))
            room -= cells
            continue
        if room > 0:
            out.append((truncate(text, room, ellipsis=""), role))
        break
    return out


def _draw(pieces: Sequence[tuple[str, str]], theme: Theme, width: int, selected: bool) -> str:
    """One line from its pieces: a selected row reads as one painted bar across the full width."""
    clipped = _clip_pieces(pieces, width)
    plain = "".join(text for text, _role in clipped)
    if selected:
        return theme.paint(clip_line(plain, width), fg="accent2", bold=True)
    return "".join(theme.paint(text, fg=role or None) for text, role in clipped)


def tree_lines(
    nodes: Sequence[DagNode],
    width: int,
    cap: int,
    theme: Theme,
    *,
    fold: FoldState | None = None,
    selected: str | None = None,
    glyphs: str = "unicode",
) -> list[str]:
    """The graph as a folding tree in `cap` rows at most, none wider than `width`; `selected` marks one row."""
    if not nodes or width <= 0 or cap <= 0:
        return []
    graph = _graph(nodes)
    rows = _specs(graph, _fit(graph, fold, cap))
    columns = _columns(rows, width, glyphs)
    lines = [
        _draw(
            _pieces(row, width, columns, glyphs),
            theme,
            width,
            row.kind in ("layer", "node") and bool(row.key) and row.key == selected,
        )
        for row in rows
    ]
    if len(lines) <= cap:
        return lines
    cut = rows[cap - 1:]
    hidden = sum(_covers(row) for row in cut)
    note = f"… {hidden} more nodes" if hidden else "… more"
    return lines[: cap - 1] + [_draw([(INDENT_STR, ""), (note, "muted")], theme, width, False)]


def fold_keys(nodes: Sequence[DagNode]) -> list[str]:
    """Every row a fold can change, in draw order: each layer, then each node with children or a detail line."""
    if not nodes:
        return []
    graph = _graph(nodes)
    expanded = {key: True for key in (*graph.order, *(f"{LAYER_KEY}{at}" for at in graph.roots))}
    return [row.key for row in _specs(graph, expanded) if row.kind in ("layer", "node") and row.foldable]


def _tally(nodes: Sequence[DagNode]) -> str:
    counts: dict[str, int] = {}
    for node in nodes:
        state = _state(node.status)
        counts[state] = counts.get(state, 0) + 1
    total = len(nodes)
    said = [f"{total} node{'' if total == 1 else 's'}"]
    said += [f"{counts[state]} {state}" for state in STATE_ORDER if counts.get(state)]
    return FACT.join(said)


def _tally_pieces(nodes: Sequence[DagNode], width: int, indent: str = "") -> list[tuple[str, str]]:
    lead = f"{indent}DAG"
    room = max(width - cell_width(lead) - cell_width(FACT), 1)
    return [(lead, "accent2"), (f"{FACT}{truncate(_tally(nodes), room)}", "dim")]


def summary_line(nodes: Sequence[DagNode], theme: Theme) -> str:
    """`DAG · 6 nodes · 2 running · 3 done`: the graph's whole counts on one line."""
    if not nodes:
        return ""
    return _draw(_tally_pieces(nodes, 1000), theme, 1000, False)


def _node_line(node: DagNode, glyphs: str) -> list[tuple[str, str]]:
    """A node named on one line, for a condensed print with no room for a box."""
    state = _state(node.status)
    said = FACT.join([node.id, _flat(node.title) or "untitled", state])
    return [(INDENT_STR, ""), (_marks(glyphs)[state], _ROLES[state]), (f" {said}", "")]


def _box_rows(node: DagNode, width: int, theme: Theme, glyphs: str) -> list[str]:
    """One frontier node as a three-line box; nothing when the terminal is too narrow to read a box."""
    box = _BOX_ASCII if glyphs == "ascii" else _BOX
    inner = width - len(INDENT_STR) - 2
    if inner < BOX_GAP + 4:
        return []
    state = _state(node.status)
    parts = _figures(node, glyphs)
    figures = FACT.join(text for text in (parts[2], parts[3]) if text)
    head = f" {_marks(glyphs)[state]} {node.id}{FACT}{_flat(node.title) or 'untitled'} "
    pieces = [
        [(INDENT_STR, "border"), (box["top"], "border"), (clip_line(head, inner, fill=box["rule"]), _ROLES[state]),
         (box["end"], "border")],
    ]
    if figures:
        pieces.append(
            [(INDENT_STR, "border"), (box["side"], "border"), (clip_line(f" {figures}", inner), "dim"),
             (box["side"], "border")]
        )
    pieces.append(
        [(INDENT_STR, "border"), (box["foot"], "border"), (box["rule"] * inner, "border"), (box["close"], "border")]
    )
    return [_draw(row, theme, width, False) for row in pieces]


def _frontier(graph: _Graph) -> list[str]:
    """The nodes a condensed print shows: the work moving now, else the layer that starts the run."""
    live = [node_id for node_id in graph.order if _state(graph.nodes[node_id].status) in OPEN_STATES]
    if live:
        return live
    for layer in graph.layers:
        if layer:
            return list(layer)
    return []


def block_lines(
    nodes: Sequence[DagNode],
    title: str,
    width: int,
    theme: Theme,
    *,
    ended: bool = False,
    small_cap: int = SMALL_GRAPH_LANES,
    glyphs: str = "unicode",
) -> list[str]:
    """The graph as the transcript keeps it at a define or end moment; condensed past `small_cap` nodes."""
    if not nodes or width <= 0:
        return []
    cap = SMALL_GRAPH_LANES if small_cap <= 0 else int(small_cap)
    moment = [(INDENT_STR, ""), ("dag", "accent"), (f"{FACT}{truncate(_flat(title) or 'graph', width - 8)}", "dim")]
    lines = [_draw(moment, theme, width, False)]
    graph = _graph(nodes)
    if len(nodes) <= cap:
        fold = FoldState()
        for node_id in graph.order:
            fold.override(node_id, True)
        for at in graph.roots:
            fold.override(f"{LAYER_KEY}{at}", True)
        return lines + tree_lines(nodes, width, max(len(nodes) * 4 + 2, 4), theme, fold=fold, glyphs=glyphs)
    lines.append(_draw(_tally_pieces(nodes, width, INDENT_STR), theme, width, False))
    stopped = [node for node in nodes if _state(node.status) in OPEN_STATES]
    if ended:
        for node in stopped[:CONDENSED_FAILURES]:
            lines.append(_draw(_node_line(node, glyphs), theme, width, False))
        more = len(stopped) - CONDENSED_FAILURES
        if more > 0:
            lines.append(_draw([(INDENT_STR, ""), (f"… {more} more", "muted")], theme, width, False))
        return lines
    listed = [(at, members) for at, members in enumerate(graph.layers) if members]
    for at, members in listed[:CONDENSED_LAYERS]:
        lines.append(_draw([(INDENT_STR, ""), (_layer_text(graph, at, members), "dim")], theme, width, False))
    extra = len(listed) - CONDENSED_LAYERS
    if extra > 0:
        lines.append(_draw([(INDENT_STR, ""), (f"… {extra} more layers", "muted")], theme, width, False))
    shown = _frontier(graph)[:CONDENSED_BOXES]
    boxes = [line for node_id in shown for line in _box_rows(graph.nodes[node_id], width, theme, glyphs)]
    if boxes:
        lines.append("")
        lines += boxes
    else:
        lines += [_draw(_node_line(graph.nodes[node_id], glyphs), theme, width, False) for node_id in shown]
    left = len(nodes) - len(shown)
    if left > 0:
        note = f"… {left} more nodes{FACT}{MORE_NOTE}"
        lines.append(_draw([(INDENT_STR, ""), (note, "muted")], theme, width, False))
    return lines
