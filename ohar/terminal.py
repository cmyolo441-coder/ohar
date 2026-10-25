"""ANSI, themes, banners, Markdown rendering, spinners, and table helpers."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TextIO

from .syntax import highlight_line

_RESET = "\x1b[0m"
_ANSI_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
)
_HEX_RE = re.compile(r"#([0-9a-fA-F]{6})\Z")

_PALETTES: dict[str, dict[str, str]] = {
    "aurora": {
        "accent": "#22d3ee",
        "accent2": "#a78bfa",
        "accent3": "#f472b6",
        "success": "#34d399",
        "error": "#fb7185",
        "warning": "#fbbf24",
        "info": "#38bdf8",
        "text": "#e5e7eb",
        "dim": "#9ca3af",
        "muted": "#6b7280",
        "code": "#c4b5fd",
        "border": "#334155",
    },
    "ember": {
        "accent": "#fb923c",
        "accent2": "#f43f5e",
        "accent3": "#facc15",
        "success": "#4ade80",
        "error": "#f87171",
        "warning": "#fbbf24",
        "info": "#fb923c",
        "text": "#f5f5f4",
        "dim": "#a8a29e",
        "muted": "#78716c",
        "code": "#fdba74",
        "border": "#7c2d12",
    },
    "moss": {
        "accent": "#4ade80",
        "accent2": "#2dd4bf",
        "accent3": "#a3e635",
        "success": "#4ade80",
        "error": "#fb7185",
        "warning": "#facc15",
        "info": "#38bdf8",
        "text": "#e7e5e4",
        "dim": "#a3a3a3",
        "muted": "#737373",
        "code": "#86efac",
        "border": "#14532d",
    },
    "mono": {
        "accent": "#e5e5e5",
        "accent2": "#d4d4d4",
        "accent3": "#a3a3a3",
        "success": "#e5e5e5",
        "error": "#d4d4d4",
        "warning": "#d4d4d4",
        "info": "#d4d4d4",
        "text": "#e5e5e5",
        "dim": "#8a8a8a",
        "muted": "#666666",
        "code": "#d4d4d4",
        "border": "#444444",
    },
}

_ANSI16 = [
    (30, (0, 0, 0)),
    (31, (205, 49, 49)),
    (32, (13, 188, 121)),
    (33, (229, 229, 16)),
    (34, (36, 114, 200)),
    (35, (188, 63, 188)),
    (36, (17, 168, 205)),
    (37, (229, 229, 229)),
    (90, (102, 102, 102)),
    (91, (241, 76, 76)),
    (92, (35, 209, 139)),
    (93, (245, 245, 67)),
    (94, (59, 142, 234)),
    (95, (214, 112, 214)),
    (96, (41, 184, 219)),
    (97, (255, 255, 255)),
]
_CUBE = (0, 95, 135, 175, 215, 255)

_TRUE_RE = re.compile(r"(?i)(truecolor|24bit)")
_256_RE = re.compile(r"(?i)(256color|256)")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def sanitize_text(text: str, *, keep_newlines: bool = True) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = strip_ansi(text).replace("\x00", "")
    out: list[str] = []
    for char in text:
        if char == "\n":
            if keep_newlines:
                out.append(char)
            continue
        if char == "\t":
            out.append("    ")
            continue
        if char in "\n\r" or unicodedata.category(char) in {"Cc", "Cs"}:
            continue
        out.append(char)
    return "".join(out)


def display_width(text: str) -> int:
    plain = strip_ansi(text)
    width = 0
    for char in plain:
        if char == "\t":
            width += 4
            continue
        if unicodedata.combining(char):
            continue
        if unicodedata.east_asian_width(char) in {"W", "F"}:
            width += 2
        elif unicodedata.category(char) != "Cf":
            width += 1
    return width


def pad(text: str, width: int) -> str:
    shortfall = width - display_width(text)
    return text + " " * max(0, shortfall)


def clip(text: str, width: int, suffix: str = "…") -> str:
    plain = strip_ansi(text)
    if width <= 0:
        return ""
    if display_width(plain) <= width:
        return plain
    if width <= len(suffix):
        return suffix[:width]
    out: list[str] = []
    used = 0
    target = width - display_width(suffix)
    for char in plain:
        char_width = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        if used + char_width > target:
            break
        out.append(char)
        used += char_width
    return "".join(out) + suffix


def color_depth(stream: TextIO | None = None) -> str:
    stream = stream if stream is not None else sys.stdout
    try:
        interactive = bool(stream.isatty())
    except Exception:
        interactive = False
    if not interactive or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return "none"
    colorterm = os.environ.get("COLORTERM", "")
    term = os.environ.get("TERM", "")
    if _TRUE_RE.search(colorterm) or _TRUE_RE.search(term):
        return "truecolor"
    if _256_RE.search(term):
        return "256"
    return "16"


def use_color(stream: TextIO | None = None, force_plain: bool = False) -> bool:
    return not force_plain and color_depth(stream) != "none"


def _hex_rgb(value: str) -> tuple[int, int, int] | None:
    match = _HEX_RE.match(value)
    if not match:
        return None
    digits = match.group(1)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def _nearest_256(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    best_index, best_distance = 0, None
    for ri, rv in enumerate(_CUBE):
        for gi, gv in enumerate(_CUBE):
            for bi, bv in enumerate(_CUBE):
                distance = (r - rv) ** 2 + (g - gv) ** 2 + (b - bv) ** 2
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = 16 + 36 * ri + 6 * gi + bi
    average = (r + g + b) / 3.0
    gray_index = max(0, min(23, round((average - 8) / 10)))
    gray = 8 + gray_index * 10
    gray_distance = (r - gray) ** 2 + (g - gray) ** 2 + (b - gray) ** 2
    if gray_distance < (best_distance or 0):
        return 232 + gray_index
    return best_index


def _nearest_16(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    best_code, best_distance = 37, None
    for code, (pr, pg, pb) in _ANSI16:
        distance = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_code = code
    return best_code


@dataclass(slots=True)
class Theme:
    name: str
    color: bool
    depth: str
    palette: dict[str, str] = field(default_factory=dict)

    def rgb(self, role: str) -> tuple[int, int, int] | None:
        value = self.palette.get(role)
        if value is None and _HEX_RE.match(role):
            value = role
        return _hex_rgb(value) if value else None

    def _fg_code(self, role: str) -> str:
        rgb = self.rgb(role)
        if rgb is None:
            return ""
        if self.depth == "truecolor":
            return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
        if self.depth == "256":
            return f"\x1b[38;5;{_nearest_256(rgb)}m"
        return f"\x1b[{_nearest_16(rgb)}m"

    def _bg_code(self, role: str) -> str:
        rgb = self.rgb(role)
        if rgb is None:
            return ""
        if self.depth == "truecolor":
            return f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
        if self.depth == "256":
            return f"\x1b[48;5;{_nearest_256(rgb)}m"
        return f"\x1b[{_nearest_16(rgb) + 10}m"

    def paint(
        self,
        text: str,
        *,
        fg: str | None = None,
        bg: str | None = None,
        bold: bool = False,
        dim: bool = False,
        italic: bool = False,
        underline: bool = False,
    ) -> str:
        if not text:
            return ""
        if not self.color:
            return text
        codes: list[str] = []
        if bold:
            codes.append("\x1b[1m")
        if dim:
            codes.append("\x1b[2m")
        if italic:
            codes.append("\x1b[3m")
        if underline:
            codes.append("\x1b[4m")
        if bg:
            bg_code = self._bg_code(bg)
            if bg_code:
                codes.append(bg_code)
        if fg and self.name != "mono":
            fg_code = self._fg_code(fg)
            if fg_code:
                codes.append(fg_code)
        if not codes:
            return text
        return "".join(codes) + text + _RESET

    def gradient(self, text: str, roles: Sequence[str] = ("accent", "accent2", "accent3")) -> str:
        if not self.color or self.name == "mono" or not text:
            return self.paint(text, bold=True) if self.color else text
        stops = [self.rgb(role) for role in roles]
        stops = [stop for stop in stops if stop is not None]
        if not stops:
            return text
        if len(stops) == 1:
            return self.paint(text, fg=stops[0])
        pieces: list[str] = []
        visible = [char for char in text]
        count = max(1, len(visible) - 1)
        for index, char in enumerate(visible):
            if char.isspace():
                pieces.append(char)
                continue
            position = index / count * (len(stops) - 1)
            left = int(position)
            right = min(left + 1, len(stops) - 1)
            fraction = position - left
            first, second = stops[left], stops[right]
            rgb = tuple(round(first[i] * (1 - fraction) + second[i] * fraction) for i in range(3))
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
            pieces.append(self.paint(char, fg=hex_color, bold=True))
        return "".join(pieces)

    def rule(self, width: int, char: str = "─") -> str:
        return self.paint(char * max(0, width), fg="border")


def make_theme(name: str, *, color: bool = True, depth: str | None = None) -> Theme:
    key = (name or "aurora").strip().lower()
    if key not in _PALETTES:
        key = "aurora"
    resolved_depth = depth or ("truecolor" if color else "none")
    return Theme(name=key, color=color, depth=resolved_depth, palette=dict(_PALETTES[key]))


_UNICODE_LOGO = [
    " ██████╗  ██╗  ██╗   █████╗   ██████╗ ",
    "██╔═══██╗ ██║  ██║  ██╔══██╗  ██╔══██╗",
    "██║   ██║ ███████║  ███████║  ██████╔╝",
    "██║   ██║ ██╔══██║  ██╔══██║  ██╔══██╗",
    "╚██████╔╝ ██║  ██║  ██║  ██║  ██║  ██║",
    " ╚═════╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝",
]
_ASCII_LOGO = [
    "  ###  #   #   #   #### ",
    " #   # #   #  # #  #   #",
    " #   # ##### #   # #### ",
    " #   # #   # #   # #  # ",
    "  ###  #   # #   # #   #",
]


def short_path(path: str | Path, workspace: Path | None = None, max_len: int = 60) -> str:
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return clip(str(path), max_len)
    candidates: list[str] = []
    if workspace is not None:
        try:
            candidates.append(str(target.relative_to(workspace.resolve())))
        except (ValueError, OSError):
            pass
    try:
        home = Path.home().resolve()
        relative = target.relative_to(home)
        candidates.append("~" if str(relative) == "." else f"~/{relative}")
    except (ValueError, OSError):
        pass
    candidates.append(str(target))
    result = min(candidates, key=len)
    return clip(result, max_len)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}m {remainder:02d}s"


def format_tokens(count: int | float) -> str:
    number = int(count or 0)
    if number < 1000:
        return str(number)
    if number < 1_000_000:
        value = number / 1000
        return f"{value:.1f}k" if value < 100 else f"{value:.0f}k"
    value = number / 1_000_000
    return f"{value:.1f}m"


def human_bytes(count: int | float) -> str:
    number = float(count or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if number < 1024 or unit == "GB":
            return f"{number:.0f}{unit}" if unit == "B" else f"{number:.1f}{unit}"
        number /= 1024
    return f"{number:.1f}GB"


def format_tool_args(name: str, args: dict, max_len: int = 70) -> str:
    if not isinstance(args, dict):
        return ""
    priority = ("command", "path", "pattern", "query", "name", "title", "cwd", "url")
    text = ""
    for key in priority:
        if key in args and args[key] not in (None, "", [], {}):
            value = args[key]
            if isinstance(value, (list, tuple)):
                value = f"{len(value)} items"
            elif isinstance(value, dict):
                value = ", ".join(f"{k}={v}" for k, v in list(value.items())[:3])
            text = f"{key}: {value}" if name == "task" else str(value)
            break
    if not text and "tasks" in args and isinstance(args["tasks"], list):
        titles = [str(task.get("title") or task.get("prompt", ""))[:28] for task in args["tasks"][:2] if isinstance(task, dict)]
        text = f"{len(args['tasks'])} tasks" + (f" · {', '.join(titles)}" if titles else "")
    text = " ".join(sanitize_text(text).split())
    return clip(text, max_len)


def banner_lines(
    theme: Theme,
    *,
    version: str,
    workspace: str,
    model: str,
    lane: str,
    glyphs: str = "unicode",
    width: int = 100,
) -> list[str]:
    unicode_glyphs = glyphs != "ascii"
    logo = _UNICODE_LOGO if unicode_glyphs else _ASCII_LOGO
    logo_width = max(display_width(line) for line in logo)
    lines: list[str] = [""]
    if width >= logo_width + 2:
        lines.extend(theme.gradient(line) for line in logo)
    else:
        lines.append(theme.gradient("O H A R"))
    lines.append("")
    tagline = "think · build · ship" if unicode_glyphs else "think . build . ship"
    context = f"v{version}  ·  {short_path(workspace, max_len=40)}  ·  {model}  ·  {lane}"
    lines.append("  " + theme.paint(tagline, fg="accent3", italic=True))
    lines.append("  " + theme.paint(clip(context, width - 2), fg="dim"))
    lines.append("")
    return lines


def status_line(
    theme: Theme,
    *,
    model: str,
    lane: str,
    elapsed: float,
    input_tokens: int,
    output_tokens: int,
    width: int = 100,
    glyphs: str = "unicode",
) -> str:
    rule = "─" if glyphs != "ascii" else "-"
    dot = "·" if glyphs != "ascii" else "|"
    pieces = [
        model,
        lane,
        f"{format_tokens(input_tokens)} in / {format_tokens(output_tokens)} out",
        format_duration(elapsed),
    ]
    body = f" {dot} ".join(piece for piece in pieces if piece)
    prefix, suffix = f"{rule * 2} ", f" {rule * 2}"
    available = max(10, width - display_width(prefix) - display_width(suffix))
    return theme.paint(prefix + clip(body, available) + suffix, fg="border", dim=True)


def approval_card(
    theme: Theme,
    *,
    action: str,
    summary: str,
    detail: str,
    risk: str,
    width: int = 100,
    glyphs: str = "unicode",
) -> list[str]:
    chars = ("┌", "│", "└", "─") if glyphs != "ascii" else ("+", "|", "+", "-")
    top, side, bottom, rule = chars
    risk_role = {"low": "success", "medium": "warning", "high": "error"}.get(risk, "warning")
    head = (
        "  "
        + theme.paint(f"{top}{rule} approval", fg="border")
        + theme.paint(f"  {action}", fg="accent", bold=True)
        + theme.paint(f"  [{risk}]", fg=risk_role, bold=True)
    )
    lines = [head, f"  {theme.paint(side, fg='border')} " + theme.paint(clip(summary, width - 4), bold=True)]
    preview = sanitize_text(detail).splitlines()
    for row in preview[:20]:
        lines.append(f"  {theme.paint(side, fg='border')} " + theme.paint(clip(row, width - 4), fg="dim"))
    if len(preview) > 20:
        lines.append(f"  {theme.paint(side, fg='border')} " + theme.paint("…", fg="dim"))
    lines.append(f"  {theme.paint(bottom + rule * 12, fg='border')}")
    return lines


class MarkdownStream:
    def __init__(
        self,
        theme: Theme,
        *,
        enabled: bool = True,
        width: int = 100,
        base_indent: str = "  ",
        glyphs: str = "unicode",
        syntax: bool = True,
    ) -> None:
        self.theme = theme
        self.enabled = enabled
        self.width = max(20, width)
        self.base_indent = base_indent
        self.glyphs = glyphs
        self.syntax = syntax
        self._buffer = ""
        self._fence: str | None = None

    def feed(self, delta: str) -> list[str]:
        if not self.enabled:
            return []
        self._buffer += sanitize_text(delta)
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        output: list[str] = []
        for line in lines:
            output.extend(self._render_line(line))
        return output

    def flush(self) -> list[str]:
        if not self.enabled:
            return []
        output: list[str] = []
        if self._buffer:
            output.extend(self._render_line(self._buffer))
            self._buffer = ""
        self._fence = None
        return output

    def _render_line(self, line: str) -> list[str]:
        stripped = line.rstrip()
        if self._fence is not None:
            if stripped.startswith("```"):
                self._fence = None
                return []
            rail = "│ " if self.glyphs != "ascii" else "| "
            return [
                self.base_indent
                + "  "
                + self.theme.paint(rail, fg="border")
                + self._code_line(stripped, self._fence)
            ]
        fence = re.match(r"^\s*```([^\s`]*)\s*$", stripped)
        if fence:
            self._fence = fence.group(1) or "text"
            rail = "│ " if self.glyphs != "ascii" else "| "
            language = self._fence if self._fence != "text" else ""
            return [
                self.base_indent
                + "  "
                + self.theme.paint(rail, fg="border", dim=True)
                + self.theme.paint(language, fg="muted")
            ]
        if not stripped:
            return [""]
        if re.match(r"^\s*([-*_])\1{2,}\s*$", stripped):
            return [self.base_indent + self.theme.paint("─" * min(self.width - 4, 48), fg="border", dim=True)]
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            text = self._inline(heading.group(2).strip())
            mark = "▌ " if self.glyphs != "ascii" else "| "
            level = len(heading.group(1))
            return [self.base_indent + self.theme.paint(mark, fg="accent") + self.theme.paint(text, fg="accent", bold=True) if level <= 3 else self.base_indent + self.theme.paint(mark, fg="accent2") + self.theme.paint(text, bold=True)]
        quote = re.match(r"^\s*>\s?(.*)$", stripped)
        if quote:
            rail = "▏ " if self.glyphs != "ascii" else "| "
            return [self.base_indent + self.theme.paint(rail, fg="accent2") + self.theme.paint(self._inline(quote.group(1)), fg="dim", italic=True)]
        bullet = re.match(r"^(\s*)[-*+]\s+(.*)$", stripped)
        if bullet:
            mark = "• " if self.glyphs != "ascii" else "* "
            return [self.base_indent + self.theme.paint(mark, fg="accent") + self._inline(bullet.group(2))]
        numbered = re.match(r"^(\s*)(\d+[.)])\s+(.*)$", stripped)
        if numbered:
            return [self.base_indent + self.theme.paint(numbered.group(2) + " ", fg="accent") + self._inline(numbered.group(3))]
        if stripped.startswith("|") and stripped.endswith("|"):
            if re.match(r"^\|[\s:|-]+\|$", stripped):
                return [self.base_indent + self.theme.paint(stripped, fg="border", dim=True)]
            return [self.base_indent + self.theme.paint(stripped, fg="dim")]
        return [self.base_indent + self._inline(stripped)]

    def _code_line(self, text: str, language: str) -> str:
        if not self.syntax:
            return self.theme.paint(text, fg="code")
        try:
            tokens = highlight_line(text, language)
        except Exception:
            return self.theme.paint(text, fg="code")
        roles = {
            "plain": "code",
            "keyword": "accent2",
            "string": "success",
            "comment": "muted",
            "number": "warning",
            "function": "info",
            "builtin": "accent3",
            "operator": "dim",
            "punctuation": "muted",
            "property": "accent3",
            "tag": "accent",
            "attribute": "info",
            "type": "accent2",
            "variable": "accent3",
        }
        pieces: list[str] = []
        for token, kind in tokens:
            if not token:
                continue
            pieces.append(
                self.theme.paint(
                    token,
                    fg=roles.get(kind, "code"),
                    bold=kind in {"keyword", "tag", "function", "type"},
                    italic=kind == "comment",
                )
            )
        return "".join(pieces)

    def _inline(self, text: str) -> str:
        pieces = re.split(r"(`[^`]*`)", text)
        output: list[str] = []
        for piece in pieces:
            if len(piece) >= 2 and piece.startswith("`") and piece.endswith("`"):
                output.append(self.theme.paint(piece[1:-1], fg="code"))
                continue
            piece = re.sub(
                r"\[([^\]]+)\]\(([^)]+)\)",
                lambda match: self.theme.paint(match.group(1), fg="accent3", underline=True),
                piece,
            )
            piece = re.sub(
                r"\*\*(.+?)\*\*",
                lambda match: self.theme.paint(match.group(1), bold=True),
                piece,
            )
            piece = re.sub(
                r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
                lambda match: self.theme.paint(match.group(1), italic=True),
                piece,
            )
            output.append(piece)
        return "".join(output)


class Spinner:
    _UNICODE_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    _ASCII_FRAMES = ("|", "/", "-", "\\")

    def __init__(
        self,
        theme: Theme,
        stream: TextIO | None = None,
        *,
        glyphs: str = "unicode",
        enabled: bool = True,
        interval: float = 0.08,
    ) -> None:
        self.theme = theme
        self.stream = stream or sys.stdout
        self.frames = self._ASCII_FRAMES if glyphs == "ascii" else self._UNICODE_FRAMES
        self.enabled = enabled
        self.interval = interval
        self._label = ""
        self._active = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._silent = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self, label: str = "thinking") -> None:
        if not self.enabled or self._active:
            if self._active:
                self.update(label)
            return
        self._label = label
        self._active = True
        self._thread = threading.Thread(target=self._spin, name="ohar-spinner", daemon=True)
        self._thread.start()

    def update(self, label: str) -> None:
        self._label = label

    def _write(self, text: str) -> None:
        if self._silent:
            return
        try:
            self.stream.write(text)
            self.stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._silent = True

    def _spin(self) -> None:
        index = 0
        while self._active:
            frame = self.frames[index % len(self.frames)]
            index += 1
            label = self._label
            line = "\r\x1b[2K" + self.theme.paint(frame, fg="accent") + " " + self.theme.paint(label, fg="dim")
            self._write(line)
            time.sleep(self.interval)
        self._write("\r\x1b[2K")

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)


def table(
    theme: Theme,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    widths: Sequence[int] | None = None,
    glyphs: str = "unicode",
) -> list[str]:
    if not headers:
        return []
    count = len(headers)
    if widths is None:
        widths = []
        for index in range(count):
            values = [display_width(strip_ansi(row[index])) if index < len(row) else 0 for row in rows]
            widths.append(min(48, max([display_width(headers[index]), *values])))
    head = "  ".join(theme.paint(pad(headers[index], widths[index]), fg="accent", bold=True) for index in range(count))
    separator = theme.paint(("─" if glyphs != "ascii" else "-") * (sum(widths) + 2 * (count - 1)), fg="border", dim=True)
    lines = [head, separator]
    for row in rows:
        cells = []
        for index in range(count):
            value = row[index] if index < len(row) else ""
            cells.append(pad(clip(str(value), widths[index]), widths[index]))
        lines.append("  ".join(cells))
    return lines


def status_dot(theme: Theme, ok: bool, glyphs: str = "unicode") -> str:
    if glyphs == "ascii":
        return theme.paint("+" if ok else "x", fg="success" if ok else "error")
    return theme.paint("●" if ok else "×", fg="success" if ok else "error")
