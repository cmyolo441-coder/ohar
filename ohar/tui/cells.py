"""Terminal cell metrics and control stripping, cluster-exact: what every tui widget cuts on.

Text is measured by grapheme cluster, not by code point: a base character with its
combining marks, a skin tone, a VS16 emoji, a ZWJ family, a flag pair, a Hangul
syllable even when it arrives decomposed into jamo is one unit that `truncate`,
`truncate_left` and `wrap` never split. Widths follow what terminals paint: controls
and combining marks 0, East Asian Wide/Fullwidth 2, a symbol made emoji by VS16 2, a
ZWJ sequence one 2-cell glyph, East Asian Ambiguous 1. Every function NFC-normalizes
first: counted jamo by jamo, one Hangul syllable would claim three cells and the
screen and the terminal would disagree about where a line ends.

`strip_control` drops escape sequences whole, truncated ones included.
`terminal.strip_ansi` only matches complete sequences, so a cut-off CSI leaves its
parameter debris ("[38;2;1;1m") on the row and a cut-off OSC swallows the rest of it.
"""

from __future__ import annotations

from functools import lru_cache
import unicodedata

TAB_CELLS = 4  # a tab is drawn as this many spaces
ZWJ = "‍"
VS16 = "️"
NEWLINE = "\n"
SPACE = " "

_ESC = "\x1b"
_BEL = "\x07"
_CSI_8BIT = "\x9b"
_ST_8BIT = "\x9c"
_STRING_INTRODUCERS = frozenset("]PX^_")  # after ESC: OSC, DCS, SOS, PM, APC
_STRING_8BIT = frozenset("\x9d\x90\x98\x9e\x9f")  # the same five as single 8-bit characters
_INTRODUCERS = (_ESC, _CSI_8BIT, *_STRING_8BIT)

_ZERO_WIDTH_CATEGORIES = frozenset({"Mn", "Me", "Cf"})
_SPACING_MARK = "Mc"
_WIDE = frozenset({"W", "F"})
_HANGUL_TRAILING_JAMO = ((0x1160, 0x11FF), (0xD7B0, 0xD7FF))
_EMOJI_MODIFIERS = (0x1F3FB, 0x1F3FF)
_REGIONAL_INDICATORS = (0x1F1E6, 0x1F1FF)
# Bidirectional embeddings, overrides and isolates: a bidi-aware terminal reorders the
# row around them, so the cells it draws are not the cells measured. They are zero width
# already, and dropped so no painted row carries one.
BIDI_CONTROLS = frozenset(map(chr, (*range(0x202A, 0x202F), *range(0x2066, 0x206A))))

# Every frame measures the same short strings again (a status glyph, a title, a figure),
# so their clusters and widths are kept; longer text is measured afresh.
_KEPT_TEXT_CHARS = 256
_KEPT_ENTRIES = 2048


def is_control(ch: str) -> bool:
    """C0 and C1 controls plus DEL: never painted, never part of another glyph."""
    code_point = ord(ch)
    return code_point < 0x20 or 0x7F <= code_point < 0xA0


def _in_range(code_point: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= code_point <= bounds[1]


def _is_zero_width(ch: str) -> bool:
    code_point = ord(ch)
    if any(_in_range(code_point, bounds) for bounds in _HANGUL_TRAILING_JAMO):
        return True
    return unicodedata.category(ch) in _ZERO_WIDTH_CATEGORIES


def _char_width(ch: str) -> int:
    """Cells one code point occupies on its own."""
    if is_control(ch) or _is_zero_width(ch):
        return 0
    if unicodedata.east_asian_width(ch) in _WIDE:
        return 2
    return 1


def _extends(cluster: str, ch: str) -> bool:
    """True when `ch` belongs to the glyph `cluster` started rather than starting one."""
    if not cluster or is_control(cluster[0]) or is_control(ch):
        return False
    if cluster[-1] == ZWJ:
        return True
    if _is_zero_width(ch) or unicodedata.category(ch) == _SPACING_MARK:
        return True
    if _in_range(ord(ch), _EMOJI_MODIFIERS):
        return True
    return len(cluster) == 1 and _is_regional(cluster) and _is_regional(ch)


def _is_regional(ch: str) -> bool:
    return _in_range(ord(ch), _REGIONAL_INDICATORS)


def _segment(text: str) -> list[str]:
    result: list[str] = []
    for ch in text:
        if result and _extends(result[-1], ch):
            result[-1] += ch
        else:
            result.append(ch)
    return result


@lru_cache(maxsize=_KEPT_ENTRIES)
def _kept_clusters(text: str) -> tuple[str, ...]:
    return tuple(_segment(unicodedata.normalize("NFC", text)))


def clusters(text: str) -> list[str]:
    """NFC `text` as the units a cursor steps over and a cut may fall between."""
    if not text:
        return []
    if text.isascii():  # already NFC, and no ASCII character joins another
        return list(text)
    if len(text) <= _KEPT_TEXT_CHARS:
        return list(_kept_clusters(text))
    return _segment(unicodedata.normalize("NFC", text))


@lru_cache(maxsize=_KEPT_ENTRIES)
def _cluster_width(cluster: str) -> int:
    """Cells one cluster from `clusters()` occupies."""
    if cluster.isascii():
        return 1 if cluster.isprintable() else 0
    if VS16 in cluster:
        return 2
    if ZWJ in cluster and sum(1 for ch in cluster if not _is_zero_width(ch)) > 1:
        return 2
    width = 0
    for index, ch in enumerate(cluster):
        if index and _in_range(ord(ch), _EMOJI_MODIFIERS):
            continue
        width += _char_width(ch)
    return width


def cell_width(text: str) -> int:
    """Cells `text` occupies; a control paints nothing, so it counts as nothing."""
    if not text:
        return 0
    if text.isascii():
        return len(text) if text.isprintable() else sum(1 for ch in text if ch.isprintable())
    return sum(_cluster_width(cluster) for cluster in clusters(text))


def _items(text: str) -> list[tuple[str, int]]:
    return [(cluster, _cluster_width(cluster)) for cluster in clusters(text)]


def _items_width(items: list[tuple[str, int]]) -> int:
    return sum(width for _, width in items)


def _fit(items: list[tuple[str, int]], cells: int) -> tuple[list[tuple[str, int]], int]:
    """The longest prefix that fits `cells`, and the index of the first item left out."""
    kept: list[tuple[str, int]] = []
    used = 0
    for index, item in enumerate(items):
        if used + item[1] > cells:
            return kept, index
        kept.append(item)
        used += item[1]
    return kept, len(items)


def _text_of(items: list[tuple[str, int]]) -> str:
    return "".join(cluster for cluster, _ in items)


def _cut_right(items: list[tuple[str, int]], cells: int) -> str:
    kept, _ = _fit(items, cells)
    return _text_of(kept)


# ── control characters and escape sequences ───────────────────────────────


def _csi_end(text: str, index: int) -> int:
    """The index after a CSI whose parameters start at `index` (ECMA-48: params, intermediates, final).

    A sequence cut off or broken by a character outside that grammar ends where it
    broke: what was read is dropped and the rest of the text stays.
    """
    while index < len(text) and "0" <= text[index] <= "?":
        index += 1
    while index < len(text) and " " <= text[index] <= "/":
        index += 1
    if index < len(text) and "@" <= text[index] <= "~":
        return index + 1
    return index


def _string_end(text: str, index: int) -> int:
    """The index after a control string (OSC, DCS, ...) whose body starts at `index`.

    BEL or ST (ESC \\ or the 8-bit ST) ends it; an unterminated one runs to the end of
    the text, which is what a terminal would swallow too.
    """
    while index < len(text):
        ch = text[index]
        if ch in (_BEL, _ST_8BIT):
            return index + 1
        if ch == _ESC and text[index + 1 : index + 2] == "\\":
            return index + 2
        index += 1
    return index


def _escape_end(text: str, index: int) -> int:
    """The index after the escape sequence whose ESC is at `index`; a lone ESC is one character."""
    introducer = text[index + 1 : index + 2]
    if introducer == "[":
        return _csi_end(text, index + 2)
    if introducer and introducer in _STRING_INTRODUCERS:
        return _string_end(text, index + 2)
    after = index + 1
    while after < len(text) and " " <= text[after] <= "/":  # nF intermediates: ESC ( B
        after += 1
    if after < len(text) and "0" <= text[after] <= "~":
        return after + 1
    return index + 1


def _strip_escapes(text: str) -> str:
    if not any(mark in text for mark in _INTRODUCERS):
        return text
    kept: list[str] = []
    index = 0
    while index < len(text):
        ch = text[index]
        if ch == _ESC:
            index = _escape_end(text, index)
        elif ch == _CSI_8BIT:
            index = _csi_end(text, index + 1)
        elif ch in _STRING_8BIT:
            index = _string_end(text, index + 1)
        else:
            kept.append(ch)
            index += 1
    return "".join(kept)


def strip_control(text: str) -> str:
    """`text` free of terminal escapes (whole or cut off, 7-bit or 8-bit) and of controls.

    Line breaks survive as "\\n" ("\\r\\n" and "\\r" become one) and a tab becomes
    `TAB_CELLS` spaces; every other C0/C1 control, DEL and bidi control is dropped, so
    what is left can be measured and painted as it is.
    """
    if not text:
        return ""
    text = text.replace("\r\n", NEWLINE).replace("\r", NEWLINE)
    stripped = _strip_escapes(text)
    out: list[str] = []
    for ch in stripped:
        if ch == NEWLINE:
            out.append(ch)
        elif ch == "\t":
            out.append(" " * TAB_CELLS)
        elif is_control(ch) or ch in BIDI_CONTROLS:
            continue
        else:
            out.append(ch)
    return "".join(out)


# ── cutting ───────────────────────────────────────────────────────────────


def truncate(text: str, cells: int, *, ellipsis: str = "…") -> str:
    """`text` cut to at most `cells`, ending in `ellipsis` when anything was cut.

    A cluster is never split; with no room even for the ellipsis, as much of `text` as
    fits is kept.
    """
    if cells <= 0:
        return ""
    items = _items(text)
    if _items_width(items) <= cells:
        return _text_of(items)
    mark = _items(ellipsis)
    mark_width = _items_width(mark)
    if mark_width > cells:
        return _cut_right(items, cells)
    return _text_of(_fit(items, cells - mark_width)[0]) + _text_of(mark)


def truncate_left(text: str, cells: int, *, ellipsis: str = "…") -> str:
    """`text` cut from the LEFT to `cells`, keeping its end: for paths, whose end names them."""
    if cells <= 0:
        return ""
    items = _items(text)
    if _items_width(items) <= cells:
        return _text_of(items)
    mark = _items(ellipsis)
    mark_width = _items_width(mark)
    room = cells if mark_width > cells else cells - mark_width
    kept: list[tuple[str, int]] = []
    used = 0
    for item in reversed(items):
        if used + item[1] > room:
            break
        kept.insert(0, item)
        used += item[1]
    return ("" if mark_width > cells else _text_of(mark)) + _text_of(kept)


def clip_line(text: str, cells: int, *, fill: str = " ") -> str:
    """`text` cut and then padded to `cells` (a box row that never splits a cluster).

    Exactly `cells` is reached whenever the leftover room fits a `fill` cluster: the
    last cluster of a 2-cell glyph cut short of an odd width leaves one cell that only
    a 1-cell `fill` can take.
    """
    if cells <= 0:
        return ""
    items = _items(text)
    if _items_width(items) > cells:
        items = _fit(items, cells)[0]
    pads = _items(fill)
    used = _items_width(items)
    index = 0
    while pads and used + pads[index % len(pads)][1] <= cells:
        used += pads[index % len(pads)][1]
        items.append(pads[index % len(pads)])
        index += 1
    return _text_of(items)


# ── wrapping ──────────────────────────────────────────────────────────────


def _paragraphs(items: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
    paragraphs: list[list[tuple[str, int]]] = [[]]
    for item in items:
        if item[0] == NEWLINE:
            paragraphs.append([])
        else:
            paragraphs[-1].append(item)
    return paragraphs


def _tokens(items: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
    """Runs of spaces and runs of everything else, in order."""
    tokens: list[list[tuple[str, int]]] = []
    for item in items:
        if tokens and (tokens[-1][0][0] == SPACE) == (item[0] == SPACE):
            tokens[-1].append(item)
        else:
            tokens.append([item])
    return tokens


def _path_break(line: list[tuple[str, int]], following: tuple[str, int]) -> int:
    """Where a word too long for a line breaks: after the last "/" that has text before
    and after it (a path or a URL breaks between its parts), else at the line's end. A
    "/" of a "//" (a URL's scheme) is not one, so "https://" never stands alone."""
    for index in range(len(line) - 1, 0, -1):
        after = line[index + 1] if index + 1 < len(line) else following
        if line[index][0] == "/" and line[index - 1][0] != "/" and after[0] != "/":
            return index + 1
    return len(line)


def _fill_line(items: list[tuple[str, int]], cells: int) -> list[list[tuple[str, int]]]:
    """Hard wrap: fill each line to the last cluster that fits, breaking anywhere."""
    lines: list[list[tuple[str, int]]] = []
    line: list[tuple[str, int]] = []
    width = 0
    for item in items:
        if line and width + item[1] > cells:
            lines.append(line)
            line = []
            width = 0
        line.append(item)
        width += item[1]
    lines.append(line)
    return lines


def _wrap_paragraph(items: list[tuple[str, int]], cells: int, hard: bool) -> list[list[tuple[str, int]]]:
    if hard:
        return _fill_line(items, cells)
    lines: list[list[tuple[str, int]]] = []
    line: list[tuple[str, int]] = []
    width = 0
    pending: list[tuple[str, int]] = []  # spaces seen since the last word; dropped at a break
    for token in _tokens(items):
        if token[0][0] == SPACE:
            pending = token
            continue
        gap = _items_width(pending)
        if width + gap + _items_width(token) <= cells:
            line.extend(pending + token)
            width += gap + _items_width(token)
            pending = []
            continue
        if line:
            lines.append(line)
            line, width = [], 0
        pending = []
        for item in token:  # fits on a fresh line in one go, or is broken
            if line and width + item[1] > cells:
                cut = _path_break(line, item)
                lines.append(line[:cut])
                line = line[cut:]
                width = _items_width(line)
            line.append(item)
            width += item[1]
    trailing, _ = _fit(pending, cells - width)
    lines.append(line + trailing)
    return lines


def wrap(text: str, cells: int, *, hard: bool = False, indent: str = "") -> list[str]:
    """`text` as lines of at most `cells` each, `indent` in front of every non-blank one.

    Breaks on "\\n" and, unless `hard`, on spaces. A word longer than the width breaks
    after the last "/" that fits (a path or URL breaks between its parts) or else at a
    cluster boundary; `hard` breaks at cluster boundaries wherever the line fills up,
    which is what code and diff text need. A cluster wider than the width (a 2-cell
    glyph in 1 cell) gets a line of its own: there is no smaller unit.
    """
    if cells <= 0:
        return [""]
    source = text.replace("\r\n", NEWLINE).replace("\r", NEWLINE)
    indented = _items(indent)
    prefix = _text_of(indented)
    budget = cells - _items_width(indented)
    if budget <= 0:  # the indent alone leaves no room for text: paint without it
        prefix, budget = "", cells
    wrapped: list[list[tuple[str, int]]] = []
    for paragraph in _paragraphs(_items(source)):
        wrapped.extend(_wrap_paragraph(paragraph, budget, hard))
    return [prefix + _text_of(line) if line else _text_of(line) for line in wrapped]
