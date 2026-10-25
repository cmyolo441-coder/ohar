"""The list picker shared by /sessions, /resume, /model, /theme and /rewind.

`FilterList` holds no IO: the caller feeds keys in, renders the rows, and
acts on the entry Enter picks. Typing filters case-insensitively over the
label and the value, Backspace takes a character back, arrows and page keys
move the highlight, Enter picks, Esc closes. A pick is always something one
command line carries as a single argument: `safe_token` refuses whitespace, a
leading dash of any kind (U+2012 to U+2015 among them) and control
characters, so a stored title or model name can never smuggle a flag such as
`--global` into a slash command. `pick_interactive` drives the same machine
over plain line input, so a CLI picker needs no tty raw mode.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..terminal import Theme
from .cells import cell_width, strip_control, truncate
from .pager import Key, decode_keys

__all__ = ["PickEntry", "FilterList", "pick_interactive", "safe_token"]

MOVE, PICK, CLOSE = "move", "pick", "close"

MAX_ENTRIES = 4096  # a picker lists a local store, never an unbounded stream
MAX_LABEL_CHARS = 240
MAX_DETAIL_CHARS = 120
MAX_FILTER_CHARS = 64
MAX_TITLE_CHARS = 80
MAX_WIDTH = 4096
MAX_ROWS = 512
MAX_PAGE = 512
DEFAULT_PAGE_ROWS = 10
MIN_LABEL_CELLS = 6

# The dashes a command line reads as the start of a flag: ASCII, the U+2012
# to U+2015 bar set, U+2011 and the minus sign, plus anything Pd by category.
_DASHES = frozenset("-\u2010\u2011\u2012\u2013\u2014\u2015\u2212")

_GLYPHS = {
    "unicode": {"pointer": "›", "current": "●", "rule": "─", "prompt": "❯"},
    "ascii": {"pointer": ">", "current": "*", "rule": "-", "prompt": ">"},
}

_CLOSE_KEYS = frozenset({"escape", "esc", "ctrlc", "ctrld", "quit"})
_ENTER_KEYS = frozenset({"enter", "return", "cr", "submit"})
_BACKSPACE_KEYS = frozenset({"backspace", "backwarddelete"})
_IGNORED_KEYS = frozenset({"left", "right", "tab", "insert", "wheelup", "wheeldown"})
_CTRL_BYTES = "\x03\x04\x1b"  # ctrl-c, ctrl-d, a bare esc: all mean leave


@dataclass(frozen=True)
class PickEntry:
    value: str
    label: str
    detail: str = ""
    current: bool = False


def safe_token(value: str) -> bool:
    """Whether `value` is one plain word a command line can carry as one argument."""
    if not isinstance(value, str) or not value:
        return False
    if value[0] in _DASHES or unicodedata.category(value[0]) == "Pd":
        return False
    return all(char.isprintable() and not char.isspace() for char in value)


def _mark(glyphs: str, name: str) -> str:
    return _GLYPHS["ascii" if glyphs == "ascii" else "unicode"][name]


def _norm(name: str) -> str:
    """`name` lower-cased with its separators gone, so any spelling of a key works."""
    if not isinstance(name, str):
        return ""
    out = name.strip().lower()
    for sep in ("_", "-", " ", "+"):
        out = out.replace(sep, "")
    return out


def _paint(theme: Theme, parts: Sequence[tuple[str, dict[str, Any]]], width: int, *, fill: bool = False) -> str:
    """`parts` painted left to right, clipped so their plain width never passes `width`."""
    out: list[str] = []
    used = 0
    last: dict[str, Any] = {}
    for text, style in parts:
        room = width - used
        if room <= 0:
            break
        piece = text if cell_width(text) <= room else truncate(text, room)
        used += cell_width(piece)
        last = style
        out.append(theme.paint(piece, **style))
    if fill and used < width:
        out.append(theme.paint(" " * (width - used), **last))
    return "".join(out)


def _one_line(text: str, cap: int) -> str:
    return " ".join(strip_control(text).split())[:cap]


class FilterList:
    """A scrollable, type-to-filter highlight over `PickEntry` rows."""

    def __init__(self, entries: Sequence[PickEntry], *, page_size: int | None = None) -> None:
        self._entries = _clean(entries)
        self.filter = ""
        self._index = next((n for n, entry in enumerate(self._entries) if entry.current), 0)
        self._fixed_page = None if page_size is None else min(max(int(page_size), 1), MAX_PAGE)
        self._body_rows = DEFAULT_PAGE_ROWS

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def matches(self) -> list[PickEntry]:
        """The entries the filter keeps, in list order."""
        needle = self.filter.casefold()
        if not needle:
            return list(self._entries)
        return [entry for entry in self._entries
                if needle in entry.label.casefold() or needle in entry.value.casefold()]

    @property
    def selected(self) -> PickEntry | None:
        """The highlighted entry, or None when the filter leaves nothing."""
        shown = self.matches
        if not shown:
            return None
        return shown[min(self._index, len(shown) - 1)]

    def feed_key(self, key: Key) -> str:
        """Apply one key: 'pick' when Enter chose an entry, 'close' when it leaves, else 'move'."""
        name = _norm(getattr(key, "name", ""))
        char = getattr(key, "char", "") or ""
        # `Key` carries no modifier flags; these only bite when the decoder that built it does.
        held = bool(getattr(key, "ctrl", False)) or bool(getattr(key, "alt", False))
        if name in _CLOSE_KEYS or (name == "char" and char != "" and (
                char in _CTRL_BYTES or (held and char.lower() in {"c", "d"}))):
            return CLOSE
        shown = self.matches
        if name in _ENTER_KEYS:
            if not shown:
                return MOVE
            entry = shown[min(self._index, len(shown) - 1)]
            return PICK if safe_token(entry.value) else MOVE
        if name in _BACKSPACE_KEYS:
            self.filter = self.filter[:-1]
            self._index = 0
            return MOVE
        if name not in _IGNORED_KEYS:
            step = _step_of(name, self._page())
            if step is not None and shown:
                self._index = min(max(self._index + step, 0), len(shown) - 1)
                return MOVE
            if name == "home" and shown:
                self._index = 0
                return MOVE
            if name == "end" and shown:
                self._index = len(shown) - 1
                return MOVE
        typed = " " if name == "space" else char
        if held:
            return MOVE
        if len(typed) == 1 and typed.isprintable():
            self.filter = (self.filter + typed)[:MAX_FILTER_CHARS]
            self._index = 0
        return MOVE

    def render(
        self,
        width: int,
        theme: Theme,
        *,
        title: str,
        max_rows: int = 12,
        glyphs: str = "unicode",
    ) -> list[str]:
        """A framed title with the highlight's place, the visible rows, the filter line last."""
        width = max(0, min(int(width), MAX_WIDTH))
        rows = max(0, min(int(max_rows), MAX_ROWS))
        if width <= 0 or rows <= 0:
            return []
        shown = self.matches
        self._index = min(self._index, max(len(shown) - 1, 0))
        body = rows - 2
        if body <= 0:  # no room for a frame: keep the highlighted row and the filter line
            if shown:
                only = [self._row(shown[self._index], True, width, theme, glyphs)]
            else:
                only = [self._no_match(width, theme)]
            only.append(self._filter_line(width, theme, glyphs))
            return only[:rows]
        self._body_rows = body
        lines = [self._header(shown, body, width, theme, title, glyphs)]
        if shown:
            start = min(max(self._index - body + 1, 0), max(len(shown) - body, 0))
            stop = min(start + body, len(shown))
            for number in range(start, stop):
                lines.append(self._row(shown[number], number == self._index, width, theme, glyphs))
        else:
            lines.append(self._no_match(width, theme))
        lines.append(self._filter_line(width, theme, glyphs))
        return lines

    def _page(self) -> int:
        return self._fixed_page if self._fixed_page is not None else max(1, self._body_rows)

    def _header(
        self,
        shown: Sequence[PickEntry],
        body: int,
        width: int,
        theme: Theme,
        title: str,
        glyphs: str,
    ) -> str:
        count, total = len(shown), len(self._entries)
        if count > body:
            position = f" ({min(self._index, count - 1) + 1}/{count})"
        elif self.filter:
            position = f" ({count}/{total})"
        else:
            position = f" ({count})"
        name = _one_line(title, MAX_TITLE_CHARS) or "Pick"
        rule = _mark(glyphs, "rule")
        head = f" {name}{position} "
        tail = max(0, width - 2 - cell_width(head))
        parts: list[tuple[str, dict[str, Any]]] = [
            (rule * 2, {"fg": "border"}),
            (head, {"fg": "accent", "bold": True}),
            (rule * tail, {"fg": "border"}),
        ]
        return _paint(theme, parts, width)

    def _row(self, entry: PickEntry, selected: bool, width: int, theme: Theme, glyphs: str) -> str:
        pointer = _mark(glyphs, "pointer") if selected else " "
        mark = _mark(glyphs, "current") if entry.current else " "
        lead = f"{pointer} {mark} "
        detail = f" {entry.detail}" if entry.detail else ""
        if selected:
            style: dict[str, Any] = {"fg": "text", "bg": "border", "bold": True}
        else:
            style = {"fg": "accent" if entry.current else "text"}
        room = width - cell_width(lead) - cell_width(detail)
        if room < MIN_LABEL_CELLS:
            label, gap = entry.label, ""
        else:
            label = truncate(entry.label, room)
            gap = " " * max(0, room - cell_width(label))
        parts: list[tuple[str, dict[str, Any]]] = [(lead + label + gap, style), (detail, {"fg": "dim"})]
        return _paint(theme, parts, width, fill=selected)

    def _no_match(self, width: int, theme: Theme) -> str:
        named = f' "{self.filter}"' if self.filter else ""
        return _paint(theme, [(f"  no entry matches{named}", {"fg": "muted"})], width)

    def _filter_line(self, width: int, theme: Theme, glyphs: str) -> str:
        prompt = f"{_mark(glyphs, 'prompt')} "
        if self.filter:
            parts: list[tuple[str, dict[str, Any]]] = [
                (prompt, {"fg": "accent", "bold": True}),
                (self.filter, {"fg": "text"}),
            ]
        else:
            parts = [(prompt, {"fg": "accent"}), ("type to filter", {"fg": "muted", "dim": True})]
        return _paint(theme, parts, width)


def pick_interactive(
    entries: Sequence[PickEntry],
    *,
    title: str,
    theme: Theme,
    width: int,
    read_line: Callable[[str], str],
) -> PickEntry | None:
    """Run a `FilterList` over a line reader: the frame goes out as the prompt, what is
    typed filters, the line's end picks. None when nothing was picked."""
    listing = FilterList(entries)
    while listing.matches:
        frame = "\n".join(listing.render(width, theme, title=title))
        try:
            raw = read_line(f"{frame}\n")
        except (EOFError, KeyboardInterrupt):
            return None
        if not isinstance(raw, str):
            return None
        before = (listing.filter, listing.selected)
        keys = list(decode_keys(raw.encode("utf-8", "replace")))
        if not raw.endswith(("\n", "\r")):
            keys.append(Key("enter"))  # the reader ate the Enter that ended the line
        for key in keys:
            action = listing.feed_key(key)
            if action == CLOSE:
                return None
            if action == PICK:
                picked = listing.selected
                return picked if picked is not None and safe_token(picked.value) else None
        if (listing.filter, listing.selected) == before:
            return None  # the line changed nothing, so asking again would ask forever
    return None


def _step_of(name: str, page: int) -> int | None:
    return {
        "up": -1,
        "down": 1,
        "pageup": -page,
        "pgup": -page,
        "prior": -page,
        "pagedown": page,
        "pgdown": page,
        "pgdn": page,
        "next": page,
    }.get(name)


def _clean(entries: Sequence[PickEntry]) -> list[PickEntry]:
    """`entries` kept to those a command can carry: an unsafe or blank value never shows."""
    kept: list[PickEntry] = []
    for entry in entries:
        if len(kept) >= MAX_ENTRIES:
            break
        value = strip_control(str(getattr(entry, "value", "") or "")).strip()
        if not safe_token(value):
            continue
        label = _one_line(str(getattr(entry, "label", "") or ""), MAX_LABEL_CHARS) or value
        detail = _one_line(str(getattr(entry, "detail", "") or ""), MAX_DETAIL_CHARS)
        kept.append(PickEntry(value, label, detail, bool(getattr(entry, "current", False))))
    return kept
