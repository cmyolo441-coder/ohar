"""An alternate-screen in-memory pager over rows rendered elsewhere.

The pager owns no content. Something else (``/history``, a long slash output, a
log) renders rows first and hands them over, either as a list or as a callable
that returns one, so opening a long session costs a single page. It switches to
the alternate screen, paints only the rows that changed since its last frame,
and switches back, which restores the main screen exactly as it was: nothing it
showed is left anywhere else, and the terminal's own scrollback is untouched.

Keys: j / Down one line down, k / Up one line up, Space / PageDown a page down,
b / PageUp a page up, g / Home the top, G / End the bottom, q / Esc leave.

Outside a terminal (a pipe, a file, a platform without termios) there is no
alternate screen to draw on, so :meth:`Pager.run` prints the rows plainly and
returns False. Importing this module never fails for that reason.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..terminal import Theme
from .cells import clip_line, strip_control

CHAR = "char"
ENTER = "enter"
TAB = "tab"
BACKSPACE = "backspace"
ESCAPE = "escape"
UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
HOME = "home"
END = "end"
INSERT = "insert"
DELETE = "delete"
PAGE_UP = "pageup"
PAGE_DOWN = "pagedown"
ALT = "alt"
CTRL_C = "ctrl+c"
CTRL_D = "ctrl+d"
UNKNOWN = "unknown"

QUIT_KEYS = frozenset({ESCAPE, CTRL_C, CTRL_D})

DEFAULT_HINT = "j/k line · space/b page · g/G ends · q quit"

CSI = "\x1b["
ALT_SCREEN_ON = f"{CSI}?1049h"
ALT_SCREEN_OFF = f"{CSI}?1049l"
CLEAR_SCREEN = f"{CSI}H{CSI}2J"
HIDE_CURSOR = f"{CSI}?25l"
SHOW_CURSOR = f"{CSI}?25h"
ERASE_LINE = f"{CSI}K"
# One synchronized update: a terminal that knows it shows a frame whole instead of row by row.
SYNC_BEGIN = f"{CSI}?2026h"
SYNC_END = f"{CSI}?2026l"

FOOTER_ROWS = 1
FALLBACK_COLS = 80
FALLBACK_ROWS = 24

_ESC = 0x1B
_BEL = 0x07
_CSI_INTRO = ord("[")
_SS3_INTRO = ord("O")
_OSC_INTRO = ord("]")
_MAX_CSI = 32
_MAX_OSC = 512

_CSI_LETTER_KEYS = {"A": UP, "B": DOWN, "C": RIGHT, "D": LEFT, "H": HOME, "F": END}
_SS3_KEYS = {**_CSI_LETTER_KEYS, "P": "f1", "Q": "f2", "R": "f3", "S": "f4"}
_TILDE_KEYS = {"1": HOME, "2": INSERT, "3": DELETE, "4": END, "5": PAGE_UP, "6": PAGE_DOWN, "7": HOME, "8": END}
_KEY_NAMES = {0x0D: ENTER, 0x0A: ENTER, 0x09: TAB, 0x08: BACKSPACE, 0x7F: BACKSPACE, 0x03: CTRL_C, 0x04: CTRL_D}


@dataclass(frozen=True)
class Key:
    name: str
    char: str = ""


def decode_keys(data: bytes) -> list[Key]:
    """Every key in `data`; an unknown sequence is one key and decoding goes on with the next byte.

    A sequence whose rest never arrived is dropped rather than guessed at: a
    pager reads in chunks large enough that this costs at most one key.
    """
    keys: list[Key] = []
    buffer = bytes(data or b"")
    index = 0
    total = len(buffer)
    while index < total:
        byte = buffer[index]
        if byte == _ESC:
            step, key = _decode_escape(buffer, index, total)
            if step <= 0:
                break
            if key is not None:
                keys.append(key)
            index += step
            continue
        if byte in _KEY_NAMES:
            keys.append(Key(_KEY_NAMES[byte]))
            index += 1
            continue
        if byte < 0x20:
            index += 1
            continue
        size = _utf8_len(byte)
        chunk = buffer[index : index + size]
        keys.append(Key(CHAR, chunk.decode("utf-8", errors="replace")))
        index += len(chunk)
    return keys


def _utf8_len(lead: int) -> int:
    if lead < 0x80:
        return 1
    if lead >= 0xF0:
        return 4
    if lead >= 0xE0:
        return 3
    if lead >= 0xC0:
        return 2
    return 1


def _decode_escape(data: bytes, index: int, total: int) -> tuple[int, Key | None]:
    """Bytes to consume from `index` and the key they are (``None`` for no key); 0 consumed ends decoding."""
    if index + 1 >= total:
        return 1, Key(ESCAPE)
    follow = data[index + 1]
    if follow == _CSI_INTRO:
        return _decode_csi(data, index, total)
    if follow == _SS3_INTRO:
        if index + 2 >= total:
            return 0, None
        return 3, Key(_SS3_KEYS.get(chr(data[index + 2]), UNKNOWN))
    if follow == _OSC_INTRO:
        end = _osc_end(data, index + 2)
        return (0, None) if end < 0 else (end - index, None)
    if follow == _ESC:
        return 1, Key(ESCAPE)
    size = _utf8_len(follow)
    text = data[index + 1 : index + 1 + size].decode("utf-8", errors="replace")
    return 1 + size, Key(ALT, text)


def _osc_end(data: bytes, start: int) -> int:
    """The index just past an OSC string's terminator, or -1 while it has none."""
    index = start
    limit = min(len(data), start + _MAX_OSC)
    while index < limit:
        byte = data[index]
        if byte == _BEL:
            return index + 1
        if byte == _ESC and index + 1 < limit and data[index + 1] == ord("\\"):
            return index + 2
        index += 1
    return -1


def _decode_csi(data: bytes, index: int, total: int) -> tuple[int, Key | None]:
    start = index + 2
    cursor = start
    while cursor < total and not (0x40 <= data[cursor] <= 0x7E):
        cursor += 1
    if cursor >= total:
        return 0, None
    if cursor - start > _MAX_CSI:
        return cursor - index + 1, Key(UNKNOWN)
    final = chr(data[cursor])
    name = _CSI_LETTER_KEYS.get(final)
    if name is not None:
        return cursor - index + 1, Key(name)
    if final == "~":
        first = data[start:cursor].decode("ascii", errors="ignore").split(";")[0]
        return cursor - index + 1, Key(_TILDE_KEYS.get(first, UNKNOWN))
    return cursor - index + 1, Key(UNKNOWN)


class Pager:
    """`lines` (or a callable returning them) on the alternate screen, one key at a time."""

    def __init__(
        self,
        lines: Sequence[str] | Callable[[], Sequence[str]],
        *,
        title: str = "",
        hint: str = DEFAULT_HINT,
        theme: Theme | None = None,
    ) -> None:
        self.title = title or ""
        self.hint = hint or ""
        self.theme = theme
        self._source = lines
        self._rows: list[str] | None = None
        self.top = 0
        self._body = FALLBACK_ROWS - FOOTER_ROWS
        self._painted: dict[int, str] = {}
        self._size: tuple[int, int] | None = None

    @property
    def lines(self) -> list[str]:
        """The rows paged over: a sequence is read afresh, a callable is asked once."""
        source = self._source
        if isinstance(source, str):
            return source.splitlines()
        if callable(source):
            if self._rows is None:
                self._rows = list(source() or ())
            return self._rows
        if isinstance(source, Sequence):
            return list(source)
        if self._rows is None:
            self._rows = list(source or ())
        return self._rows

    def invalidate(self) -> None:
        """Forget what is on the screen: the next frame paints every row."""
        self._painted = {}
        self._size = None

    def handle_key(self, key: Key) -> bool:
        """Move by one key; False when the reader left. Paging uses the height of the last page asked for."""
        if key.name in QUIT_KEYS:
            return False
        if key.name == CHAR and key.char in ("q", "Q"):
            return False
        body = self._body
        char = key.char if key.name == CHAR else ""
        if key.name == DOWN or char == "j":
            self._scroll(1, body)
        elif key.name == UP or char == "k":
            self._scroll(-1, body)
        elif key.name == PAGE_DOWN or char == " ":
            self._scroll(body, body)
        elif key.name == PAGE_UP or char == "b":
            self._scroll(-body, body)
        elif key.name == HOME or char == "g":
            self.top = 0
        elif key.name == END or char == "G":
            self.top = self._last_top(body)
        self._clamp(body)
        return True

    def page(self, cols: int, rows: int) -> list[str]:
        """The page for a `rows` x `cols` screen: the visible rows then the footer, never more than `rows`."""
        cols = max(int(cols), 1)
        rows = max(int(rows), 1)
        body = max(rows - FOOTER_ROWS, 1)
        self._body = body
        lines = self.lines
        self._clamp(body)
        shown = [_fitted(line, cols) for line in lines[self.top : self.top + body]]
        out = list(shown)
        if lines or self.title or self.hint:
            out.append(self._footer(cols, self.top + 1 if lines else 0, self.top + len(shown), len(lines)))
        return out[:rows]

    def _footer(self, cols: int, first: int, last: int, total: int) -> str:
        pieces = [piece for piece in (self.title, f"{first}-{last}/{total}", self.hint) if piece]
        text = clip_line(f" {' · '.join(pieces)} ", max(cols, 1))
        return self.theme.paint(text, fg="muted", dim=True) if self.theme is not None else text

    def _scroll(self, step: int, body: int) -> None:
        self.top = min(max(self.top + step, 0), self._last_top(body))

    def _last_top(self, body: int) -> int:
        return max(len(self.lines) - body, 0)

    def _clamp(self, body: int) -> None:
        self.top = min(max(int(self.top), 0), self._last_top(body))

    def _frame(self, cols: int, rows: int) -> str:
        """The bytes that make the screen show the current page: only the rows that changed, "" when none did."""
        if self._size != (cols, rows):
            self.invalidate()
            self._size = (cols, rows)
        shown = {row: line for row, line in enumerate(self.page(cols, rows))}
        parts: list[str] = []
        for row in sorted(set(shown) | set(self._painted)):
            text = shown.get(row, "")
            before = self._painted.get(row)
            if before == text or (before is None and not text):
                continue
            parts.append(f"{CSI}{row + 1};1H{ERASE_LINE}{text}")
        self._painted = {row: text for row, text in shown.items() if text}
        if not parts:
            return ""
        return SYNC_BEGIN + "".join(parts) + SYNC_END

    def run(self, *, stdin: Any | None = None, stdout: Any | None = None) -> bool:
        """Page through the rows on a terminal; False when there was none to page on."""
        source = sys.stdin if stdin is None else stdin
        sink = sys.stdout if stdout is None else stdout
        fd = _fileno(source)
        if fd is None or not _is_tty(source) or not _is_tty(sink):
            self._print_all(sink)
            return False
        restore = _enter_raw(fd)
        if restore is None:
            self._print_all(sink)
            return False
        try:
            _write(sink, ALT_SCREEN_ON + CLEAR_SCREEN + HIDE_CURSOR)
            self._drive(fd, sink)
        finally:
            _write(sink, SHOW_CURSOR + ALT_SCREEN_OFF)
            restore()
        return True

    def _drive(self, fd: int, sink: Any) -> None:
        while True:
            cols, rows = _size(fd, sink)
            frame = self._frame(cols, rows)
            if frame:
                _write(sink, frame)
            try:
                data = os.read(fd, 64)
            except OSError:
                return
            if not data:
                return
            for key in decode_keys(data):
                if not self.handle_key(key):
                    return

    def _print_all(self, sink: Any) -> None:
        cols, _rows = _size(None, sink)
        for line in self.lines:
            _write(sink, _fitted(line, cols) + "\n")


def _fitted(line: str, cols: int) -> str:
    """One screen row: control-free, one line, cut to `cols` without padding."""
    return clip_line(strip_control(line).replace("\n", " "), cols, fill="")


def show_lines(
    lines: Sequence[str],
    *,
    title: str = "",
    stdin: Any | None = None,
    stdout: Any | None = None,
) -> bool:
    """Page over rows already rendered; False when they were only printed (no terminal to page on)."""
    return Pager(lines, title=title).run(stdin=stdin, stdout=stdout)


def _is_tty(stream: Any) -> bool:
    probe = getattr(stream, "isatty", None)
    try:
        return bool(callable(probe) and probe())
    except (OSError, ValueError):
        return False


def _fileno(stream: Any) -> int | None:
    probe = getattr(stream, "fileno", None)
    if not callable(probe):
        return None
    try:
        fd = int(probe())
    except (OSError, TypeError, ValueError):
        return None
    return fd if fd >= 0 else None


def _write(stream: Any, text: str) -> None:
    try:
        stream.write(text)
        stream.flush()
    except (OSError, ValueError, TypeError):
        pass


def _size(fd: int | None, stream: Any) -> tuple[int, int]:
    """The screen to draw on: the terminal's own size, or a plausible one when it cannot say."""
    for candidate in (fd, _fileno(stream)):
        if candidate is None:
            continue
        try:
            size = os.get_terminal_size(candidate)
        except OSError:
            continue
        if size.columns > 0 and size.lines > 0:
            return int(size.columns), int(size.lines)
    fallback = shutil.get_terminal_size((FALLBACK_COLS, FALLBACK_ROWS))
    return max(20, int(fallback.columns)), max(3, int(fallback.lines))


def _enter_raw(fd: int) -> Callable[[], None] | None:
    """Put `fd` in raw mode and hand back the way to undo it; None where the platform cannot."""
    try:
        import termios
        import tty
    except ImportError:
        return None
    try:
        saved = termios.tcgetattr(fd)
        tty.setraw(fd)
    except (OSError, ValueError, termios.error):
        return None

    def restore() -> None:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except OSError:
            pass

    return restore
