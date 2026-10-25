"""Inline unified-diff rendering for edit and write tool results.

Takes the `difflib.unified_diff` text the tools' diff preview already emits as
its payload: cleans it of ANSI and control debris, classifies every line, folds
the `---` / `+++` pair into one file header, numbers the lines in a dim gutter
from the `@@` starts, paints added and removed lines as full-width bands, and
caps the preview. Malformed input yields rows, never an exception.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ohar.terminal import Theme, strip_ansi

from .cells import clip_line, strip_control, truncate, truncate_left, wrap

__all__ = [
    "DIFF_ADDED",
    "DIFF_CONTEXT",
    "DIFF_FILE",
    "DIFF_HUNK",
    "DIFF_NOTE",
    "DIFF_REMOVED",
    "PREVIEW_LINES",
    "DiffLine",
    "compact_file_header",
    "has_diff",
    "parse_inline_diff",
    "render_diff",
]

DIFF_FILE = "file"
DIFF_HUNK = "hunk"
DIFF_ADDED = "added"
DIFF_REMOVED = "removed"
DIFF_CONTEXT = "context"
DIFF_NOTE = "note"  # the omitted-lines summary, "\ No newline at end of file", anything else

PREVIEW_LINES = 24  # diff lines shown in the live view; the pager shows them all
MAX_INPUT_LINES = 512  # a payload longer than this is read no further, so the cost of junk text stays bounded
MIN_BODY = 6  # cells a diff line needs beside the gutter, or the gutter gives way to its words
INDENT = 4  # the gutter's smallest width, in cells: where a result's words hang under their tool row

ARROW = " → "
ARROW_ASCII = " -> "
_ARROWS = (ARROW, ARROW_ASCII)
_GAPS = {"unicode": "⋮", "ascii": "!"}
_NO_FILE = "/dev/null"  # git's side of a created or deleted file
_SIDES = ("a/", "b/", "i/", "w/", "c/", "o/")


@dataclass(frozen=True)
class DiffLine:
    """One classified line of an inline diff, its `+`, `-`, or ` ` marker kept as its first cell."""

    kind: str  # file | hunk | added | removed | context | note
    text: str


def _clean(text: str) -> str:
    """Untrusted payload text with ANSI and control characters gone, spacing left alone."""
    if not isinstance(text, str):
        return ""
    return strip_control(strip_ansi(text))


def _side(name: str) -> str:
    """One side of a file header without git's `a/` or `b/` prefix."""
    for prefix in _SIDES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def compact_file_header(text: str) -> str:
    """`a/x → b/x` as `x` when one file is edited, created, or deleted; `x → y` only for a rename."""
    clean = " ".join(_clean(text).split())
    arrow = next((each for each in _ARROWS if each in clean), "")
    if not arrow:
        return _side(clean.strip())
    before, _, after = clean.partition(arrow)
    old, new = _side(before.strip()), _side(after.strip())
    if not old or old in (_NO_FILE, new):
        return new
    if not new or new == _NO_FILE:
        return old
    return f"{old}{ARROW}{new}"


def _strip_padding(line: DiffLine) -> DiffLine:
    """A code line without the padding a diff puts after its words; its `+`, `-`, or ` ` marker stays."""
    if line.kind in (DIFF_ADDED, DIFF_REMOVED, DIFF_CONTEXT):
        return DiffLine(line.kind, line.text[:1] + line.text[1:].rstrip())
    return line


def _classify(lines: Sequence[str]) -> list[DiffLine]:
    out: list[DiffLine] = []
    at, hunks = 0, 0
    while at < len(lines):
        line = lines[at]
        following = lines[at + 1] if at + 1 < len(lines) else ""
        if line.startswith("@@"):
            hunks += 1
            out.append(DiffLine(DIFF_HUNK, line))
        elif line.startswith("--- ") and following.startswith("+++ "):
            out.append(DiffLine(DIFF_FILE, compact_file_header(f"{line[4:]}{ARROW}{following[4:]}")))
            at += 1  # the pair is one header row
        elif line.startswith(("--- ", "+++ ")) and not hunks:
            out.append(DiffLine(DIFF_FILE, compact_file_header(line[4:])))
        elif line.startswith("+"):
            out.append(DiffLine(DIFF_ADDED, line))
        elif line.startswith("-"):
            out.append(DiffLine(DIFF_REMOVED, line))
        elif line.startswith(" "):
            out.append(DiffLine(DIFF_CONTEXT, line))
        elif any(each in line for each in _ARROWS):
            out.append(DiffLine(DIFF_FILE, compact_file_header(line)))
        else:
            out.append(DiffLine(DIFF_NOTE, line))
        at += 1
    return [_strip_padding(item) for item in out]


def parse_inline_diff(payload: str) -> list[DiffLine]:
    """The classified lines of a diff payload, its file header pair folded; nothing for an empty one."""
    text = _clean(payload).replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []
    lines = text.split("\n")[:MAX_INPUT_LINES]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    kept = [line for line in lines if line.strip() or line.startswith(" ")]
    return _classify(kept)


def has_diff(payload: str) -> bool:
    """Whether the payload carries a diff worth painting: a file or hunk header, and a changed line."""
    kinds = {line.kind for line in parse_inline_diff(payload)}
    return bool(kinds & {DIFF_ADDED, DIFF_REMOVED}) and bool(kinds & {DIFF_FILE, DIFF_HUNK})


def _hunk_starts(text: str) -> tuple[int | None, int | None]:
    """The `(old, new)` first line numbers of `@@ -a,b +c,d @@`; nothing when the header lacks either."""
    parts = text.split("@@")
    fields = parts[1].split() if len(parts) >= 3 else []
    if len(fields) < 2 or not fields[0].startswith("-") or not fields[1].startswith("+"):
        return None, None
    try:
        return int(fields[0][1:].partition(",")[0]), int(fields[1][1:].partition(",")[0])
    except ValueError:
        return None, None


def _numbered(lines: Sequence[DiffLine]) -> list[tuple[DiffLine, int | None]]:
    """Each line with its gutter number: the new file's for added and context lines, the old file's for removed."""
    old: int | None = None
    new: int | None = None
    numbered: list[tuple[DiffLine, int | None]] = []
    for line in lines:
        number: int | None = None
        if line.kind == DIFF_HUNK:
            old, new = _hunk_starts(line.text)
        elif line.kind in (DIFF_ADDED, DIFF_CONTEXT) and new is not None:
            number, new = new, new + 1
        elif line.kind == DIFF_REMOVED and old is not None:
            number, old = old, old + 1
        if line.kind == DIFF_CONTEXT and old is not None:
            old += 1
        numbered.append((line, number))
    return numbered


def _gutter(numbers: Sequence[int | None], width: int, indent: int) -> int:
    """`indent` cells, wider only for a number that needs it; 0 when the body would be starved of room."""
    if indent <= 0:
        return 0
    longest = max((len(str(number)) for number in numbers if number is not None), default=0)
    cells = max(indent, longest + 1)
    return cells if width - cells >= MIN_BODY else 0


def _role(kind: str) -> tuple[str, str, bool]:
    """The `(fg, bg, bold)` a kind paints with; added and removed read as full-width bands."""
    if kind == DIFF_ADDED:
        return "success", "border", False
    if kind == DIFF_REMOVED:
        return "error", "border", False
    if kind == DIFF_FILE:
        return "accent", "", True
    if kind == DIFF_CONTEXT:
        return "code", "", False
    return "muted", "", False


def _row(lead: str, body: str, fg: str, bg: str, theme: Theme, *, bold: bool = False) -> str:
    """One painted row: the dim gutter, then the body, whose band padding reaches the right edge."""
    left = theme.paint(lead, fg="dim") if lead.strip() else lead
    return left + theme.paint(body, fg=fg or None, bg=bg or None, bold=bold)


def _label(number: int | None, gutter: int) -> str:
    """`number` right-aligned in the gutter with one space of air; nothing where there is no gutter."""
    if not gutter:
        return ""
    if number is None:
        return " " * gutter
    return f"{number:>{gutter - 1}} "


def _code_body(text: str, cells: int) -> list[str]:
    """A diff line hard-wrapped like code; every continuation row repeats the `+`, `-`, or ` ` marker."""
    if cells < 2:
        return [truncate(text, max(cells, 1))]
    rows = wrap(text, cells, hard=True) or [text]
    if len(rows) < 2 or text[:1] not in ("+", "-", " "):
        return rows
    return [rows[0]] + [clip_line(f"{text[0]}{row}", cells) for row in rows[1:]]


def _prose_body(text: str, cells: int) -> list[str]:
    """Prose word-wrapped to the row's width: a summary, a note, a count."""
    if cells < 2:
        return [truncate(text, max(cells, 1))]
    return wrap(text, cells) or [text]


def _omit_note(hidden: int, glyphs: str) -> str:
    dots = "…" if glyphs != "ascii" else "..."
    noun = "line" if hidden == 1 else "lines"
    return f"{dots} omitted {hidden} {noun}"


def _line_rows(line: DiffLine, number: int | None, gutter: int, room: int, theme: Theme) -> list[str]:
    fg, bg, bold = _role(line.kind)
    label = _label(number, gutter)
    blank = " " * gutter
    if line.kind == DIFF_FILE:
        # One row, cut from the left: the end of a path names the file.
        return [_row(blank, truncate_left(line.text, room), fg, bg, theme, bold=bold)]
    if line.kind == DIFF_NOTE:
        return [_row(blank, row, fg, bg, theme) for row in _prose_body(line.text, room)]
    banded = line.kind in (DIFF_ADDED, DIFF_REMOVED)
    return [
        _row(label if at == 0 else blank, clip_line(row, room) if banded else row, fg, bg, theme)
        for at, row in enumerate(_code_body(line.text, room))
    ]


def render_diff(
    payload: str,
    width: int,
    theme: Theme,
    *,
    max_lines: int = PREVIEW_LINES,
    indent: int = INDENT,
    glyphs: str = "unicode",
) -> list[str]:
    """The payload's diff at `width`: the first `max_lines` lines (all of them when 0), numbered in a dim gutter,
    a `⋮` where the numbering jumps, and one note counting the lines the cap left out."""
    lines = parse_inline_diff(payload)
    if not lines or width <= 0:
        return []
    numbered = _numbered(lines)
    gutter = _gutter([number for _line, number in numbered], width, indent)
    room = max(width - gutter, 1)
    limit = len(numbered) if max_lines <= 0 else max(0, int(max_lines))
    shown, rest = numbered[:limit], numbered[limit:]
    rows: list[str] = []
    hunks = 0
    for line, number in shown:
        if line.kind == DIFF_HUNK:
            hunks += 1
            if hunks > 1 and gutter:
                mark = _GAPS.get(glyphs, _GAPS["unicode"])
                rows.append(_row(f"{mark:>{gutter - 1}} ", "", "muted", "", theme))
            continue
        rows.extend(_line_rows(line, number, gutter, room, theme))
    hidden = sum(1 for line, _number in rest if line.kind != DIFF_HUNK)
    if hidden:
        fg, bg, bold = _role(DIFF_NOTE)
        for row in _prose_body(_omit_note(hidden, glyphs), room):
            rows.append(_row(" " * gutter, row, fg, bg, theme, bold=bold))
    return rows
