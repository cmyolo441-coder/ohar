"""Interactive terminal UI for OHAR."""

from __future__ import annotations

import os
import shutil
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

from .terminal import (
    MarkdownStream,
    Spinner,
    Theme,
    approval_card,
    banner_lines,
    clip,
    color_depth,
    display_width,
    format_duration,
    format_tool_args,
    make_theme,
    sanitize_text,
    status_line,
    table,
)
from .types import ApprovalRequest, Config, ToolResult


class UI:
    def __init__(
        self,
        config: Config,
        workspace: Path,
        *,
        plain: bool | None = None,
        version: str = "0.2.0",
        stdout: TextIO | None = None,
        stdin: TextIO | None = None,
    ) -> None:
        self.config = config
        self.workspace = Path(workspace)
        self.version = version
        self.stdout = stdout or sys.stdout
        self.stdin = stdin or sys.stdin
        self.stderr = sys.stderr
        try:
            stdin_tty = bool(self.stdin.isatty())
            stdout_tty = bool(self.stdout.isatty())
        except Exception:
            stdin_tty = stdout_tty = False
        if plain is None:
            plain = not (stdin_tty and stdout_tty)
        self.plain = bool(plain)
        self.interactive = not self.plain
        depth = color_depth(self.stdout) if self.interactive else "none"
        self.theme: Theme = make_theme(
            config.ui.theme, color=depth != "none", depth=depth
        )
        self.glyphs = config.ui.glyphs if config.ui.glyphs in {"unicode", "ascii"} else "unicode"
        try:
            size = shutil.get_terminal_size((100, 30))
            self.width = max(40, min(180, size.columns))
        except OSError:
            self.width = 100
        self._lock = threading.RLock()
        self._spinner = Spinner(
            self.theme,
            self.stdout,
            glyphs=self.glyphs,
            enabled=self.interactive and os.environ.get("TERM") != "dumb",
        )
        self._markdown = MarkdownStream(
            self.theme,
            enabled=bool(config.ui.markdown and self.interactive),
            width=self.width,
            glyphs=self.glyphs,
            syntax=bool(config.ui.syntax),
        )
        self._streaming = False
        self._stream_text = False
        self._raw_stream = self.plain or not bool(config.ui.markdown)
        self._turn_started = False
        self._auto_approve = False
        self._broken = False
        self._completions: list[str] = []
        self._history_path = Path(os.environ.get("OHAR_HOME", "~/.ohar")).expanduser() / "history"
        self._readline = None
        self._plain_line_start = True
        self._setup_readline()

    # ---------------------------------------------------------------- properties

    @property
    def auto_approve(self) -> bool:
        return self._auto_approve

    @property
    def is_plain(self) -> bool:
        return self.plain

    # ------------------------------------------------------------------- output

    def _write(self, text: str, *, end: str = "\n", stream: TextIO | None = None) -> None:
        if self._broken or not text and not end:
            return
        target = stream or self.stdout
        try:
            target.write(text + end)
            target.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._broken = True

    def _write_lines(self, lines: list[str]) -> None:
        for line in lines:
            self._write(line)

    def _stop_live(self) -> None:
        self._spinner.stop()

    def startup(self, *, model: str, lane: str, resumed: str | None = None) -> None:
        if self.plain:
            return
        with self._lock:
            self._stop_live()
            if self.config.ui.banner:
                banner = banner_lines(
                    self.theme,
                    version=self.version,
                    workspace=str(self.workspace),
                    model=model,
                    lane=lane,
                    glyphs=self.glyphs,
                    width=self.width,
                )
                self._write_lines([line for line in banner if line is not None])
            else:
                self._write(
                    self.theme.paint("OHAR", fg="accent", bold=True)
                    + "  "
                    + self.theme.paint(f"{model} · {lane}", fg="dim")
                )
                self._write("")
            if resumed:
                point = "•" if self.glyphs != "ascii" else "*"
                self._write(
                    "  "
                    + self.theme.paint(point, fg="accent2")
                    + " "
                    + self.theme.paint(f"resumed · {clip(resumed, self.width - 16)}", fg="dim")
                )
                self._write("")
            hint = "  type your task  ·  /help commands  ·  ! shell  ·  Ctrl-C interrupt  ·  Ctrl-D exit"
            if self.glyphs == "ascii":
                hint = hint.replace("·", "|")
            self._write(self.theme.paint(clip(hint, self.width), fg="muted"))
            self._write("")

    def set_completions(self, words: list[str]) -> None:
        self._completions = [word for word in words if word]
        self._configure_completer()

    def info(self, text: str) -> None:
        if self.plain:
            text = sanitize_text(str(text), keep_newlines=True)
        else:
            text = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
            text = "".join(
                char for char in text if char in "\n\t\x1b" or ord(char) >= 0x20
            )
        with self._lock:
            self._stop_live()
            self._write_lines(text.splitlines() or [""])

    def notice(self, message: str, level: str = "info") -> None:
        message = sanitize_text(message).strip()
        if not message:
            return
        if self.plain:
            if level in {"warning", "error"}:
                self._write(f"ohar: {message}", stream=self.stderr)
            return
        glyphs = {
            "info": ("•", "accent2"),
            "warning": ("!", "warning"),
            "error": ("✗", "error"),
            "success": ("✓", "success"),
        }
        if self.glyphs == "ascii":
            glyphs["info"] = ("*", "accent2")
            glyphs["warning"] = ("!", "warning")
            glyphs["error"] = ("x", "error")
            glyphs["success"] = ("+", "success")
        glyph, role = glyphs.get(level, glyphs["info"])
        with self._lock:
            self._stop_live()
            prefix = "  " + self.theme.paint(glyph, fg=role, bold=True) + " "
            for index, line in enumerate(message.splitlines()):
                self._write((prefix if index == 0 else "    ") + clip(line, self.width - 4))

    def event(self, message: str, level: str = "info") -> None:
        """A command-level message: visible in plain mode when informational."""
        if self.plain and level == "info":
            self.info(message)
        else:
            self.notice(message, level)

    def error(self, message: str) -> None:
        message = sanitize_text(str(message)).strip() or "Unknown error."
        if self.plain:
            self._write(f"ohar: {message}", stream=self.stderr)
            return
        self.notice(message, "error")

    # ---------------------------------------------------------------- streaming

    def stream_start(self) -> None:
        with self._lock:
            self._streaming = True
            self._stream_text = False
            self._markdown = MarkdownStream(
                self.theme,
                enabled=bool(self.config.ui.markdown and self.interactive),
                width=self.width,
                glyphs=self.glyphs,
                syntax=bool(self.config.ui.syntax),
            )
            if not self.plain:
                self._spinner.start("thinking")

    def _begin_visible(self) -> None:
        if self._stream_text:
            return
        self._stop_live()
        self._stream_text = True
        if not self._turn_started:
            self._write("")
            self._turn_started = True

    def text(self, delta: str) -> None:
        if not delta:
            return
        with self._lock:
            if self._broken:
                return
            if not self._streaming:
                self._streaming = True
            if self._raw_stream:
                clean = sanitize_text(delta, keep_newlines=True)
                if clean:
                    if not self.plain and not self._stream_text:
                        self._begin_visible()
                    self._stream_text = True
                    self._write(clean, end="")
                    self._plain_line_start = clean.endswith("\n")
                return
            lines = self._markdown.feed(delta)
            if lines:
                self._begin_visible()
                self._write_lines(lines)

    def stream_end(self) -> None:
        with self._lock:
            if not self._streaming:
                self._stop_live()
                return
            if self._raw_stream:
                self._stop_live()
                if self._stream_text and not self._plain_line_start:
                    self._write("", end="\n")
                self._plain_line_start = True
            else:
                lines = self._markdown.flush()
                if lines:
                    self._begin_visible()
                    self._stop_live()
                    self._write_lines(lines)
                else:
                    self._stop_live()
            self._streaming = False
            self._stream_text = False

    def reasoning(self, delta: str) -> None:
        return None

    # -------------------------------------------------------------------- tools

    def tool_start(self, name: str, args: dict) -> None:
        if self.plain:
            return
        summary = format_tool_args(
            name, args, max_len=max(10, self.width - display_width(name) - 10)
        )
        with self._lock:
            self._stop_live()
            if not self._turn_started:
                self._write("")
                self._turn_started = True
            glyph = "⏵" if self.glyphs != "ascii" else ">"
            line = (
                "  "
                + self.theme.paint(glyph + " ", fg="accent")
                + self.theme.paint(name, fg="accent", bold=True)
            )
            if summary:
                line += "  " + self.theme.paint(summary, fg="dim")
            self._write(line)
            self._spinner.start(name)

    def tool_end(self, name: str, result: ToolResult, seconds: float) -> None:
        if self.plain:
            return
        with self._lock:
            self._stop_live()
            denied = bool(result.meta.get("denied"))
            if denied:
                glyph, role, word = ("⊘", "warning", "denied") if self.glyphs != "ascii" else ("!", "warning", "denied")
            elif result.ok:
                glyph, role, word = ("✓", "success", "done") if self.glyphs != "ascii" else ("+", "success", "done")
            else:
                glyph, role, word = ("✗", "error", "failed") if self.glyphs != "ascii" else ("x", "error", "failed")
            line = (
                "  "
                + self.theme.paint(glyph, fg=role, bold=True)
                + " "
                + self.theme.paint(f"{word} · {format_duration(seconds)}", fg=role)
            )
            self._write(line)
            if result.content:
                try:
                    limit = max(1, min(80, int(result.meta.get("preview_lines", 4))))
                except (TypeError, ValueError):
                    limit = 4
                preview = [
                    line
                    for line in sanitize_text(result.content).splitlines()
                    if line.strip()
                ][:limit]
                for row in preview:
                    self._write(
                        "    "
                        + self.theme.paint(clip(row, max(10, self.width - 6)), fg="dim")
                    )
            self._turn_started = True

    def subagent_event(self, event: dict) -> None:
        if self.plain or not isinstance(event, dict):
            return
        kind = str(event.get("type", ""))
        task_id = str(event.get("id", "?"))
        title = clip(str(event.get("title", "")), 44)
        with self._lock:
            was_live = self._spinner.active
            self._stop_live()
            if kind == "start":
                glyph = "⇢" if self.glyphs != "ascii" else ">"
                head = f"  {glyph} {task_id}  {title}"
                context = f"{event.get('lane', '')} · {event.get('model', '')}"
                remaining = max(6, self.width - display_width(head) - 3)
                self._write(
                    "  "
                    + self.theme.paint(glyph + " ", fg="accent2")
                    + self.theme.paint(f"{task_id}  {title}", fg="accent2")
                    + "  "
                    + self.theme.paint(clip(context, remaining), fg="muted")
                )
            elif kind == "tool":
                arrow = "↳" if self.glyphs != "ascii" else "->"
                self._write(
                    "      "
                    + self.theme.paint(arrow + " " + str(event.get("name", "")), fg="muted")
                )
            elif kind == "done":
                glyph = "✓" if self.glyphs != "ascii" else "+"
                self._write(
                    "  "
                    + self.theme.paint(glyph + " ", fg="success")
                    + self.theme.paint(f"{task_id}  {title}", fg="success")
                    + "  "
                    + self.theme.paint(
                        f"{format_duration(float(event.get('seconds', 0) or 0))} · "
                        f"{int(event.get('chars', 0) or 0):,} chars",
                        fg="dim",
                    )
                )
            elif kind == "error":
                glyph = "✗" if self.glyphs != "ascii" else "x"
                self._write(
                    "  "
                    + self.theme.paint(glyph + " ", fg="error")
                    + self.theme.paint(f"{task_id}  {title}", fg="error")
                    + "  "
                    + self.theme.paint(
                        clip(str(event.get("error", "")), max(10, self.width - 30)),
                        fg="dim",
                    )
                )
            if was_live:
                self._spinner.start("task · parallel agents")

    def turn_end(self, meta: dict) -> None:
        if self.plain:
            return
        with self._lock:
            self._stop_live()
            self._write("")
            line = status_line(
                self.theme,
                model=str(meta.get("model", "")),
                lane=str(meta.get("lane", "")),
                elapsed=float(meta.get("elapsed_s", 0) or 0),
                input_tokens=int(meta.get("input_tokens", 0) or 0),
                output_tokens=int(meta.get("output_tokens", 0) or 0),
                width=self.width,
                glyphs=self.glyphs,
            )
            self._write("  " + line)
            self._write("")

    # ----------------------------------------------------------------- approval

    def confirm(self, req: ApprovalRequest) -> bool:
        if self._auto_approve:
            return True
        if self.plain or not self.interactive:
            self.notice(
                f"{req.action} requires approval; denied in non-interactive mode. "
                "Use --yolo to auto-approve.",
                "warning",
            )
            return False
        with self._lock:
            self._stop_live()
            if not self._turn_started:
                self._write("")
                self._turn_started = True
            self._write_lines(
                approval_card(
                    self.theme,
                    action=req.action,
                    summary=req.summary,
                    detail=req.detail,
                    risk=req.risk,
                    width=self.width,
                    glyphs=self.glyphs,
                )
            )
        prompt_glyph = "❯" if self.glyphs != "ascii" else ">"
        try:
            answer = input(
                "  "
                + self.theme.paint("allow?", fg="accent", bold=True)
                + " "
                + self.theme.paint("[y]es / [n]o / [a]lways", fg="dim")
                + f"  {prompt_glyph} "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            self._write("")
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"a", "always"}:
            self._auto_approve = True
            self.notice("Auto-approving tools for the rest of this session.", "warning")
            return True
        return False

    def set_theme(self, name: str) -> bool:
        theme_name = (name or "").strip().lower()
        if theme_name not in {"aurora", "ember", "moss", "mono"}:
            return False
        self.config.ui.theme = theme_name
        depth = color_depth(self.stdout) if self.interactive else "none"
        with self._lock:
            self.theme = make_theme(theme_name, color=depth != "none", depth=depth)
            self._spinner.theme = self.theme
            self._markdown.theme = self.theme
        return True

    @staticmethod
    def theme_names() -> tuple[str, ...]:
        return ("aurora", "ember", "moss", "mono")

    @contextmanager
    def working(self, label: str = "working") -> Iterator[None]:
        with self._lock:
            self._stop_live()
            if not self.plain:
                self._spinner.start(label)
        try:
            yield
        finally:
            with self._lock:
                self._stop_live()

    def plan(self, plan: dict | list, *, title: str = "Plan") -> None:
        if isinstance(plan, dict):
            steps = plan.get("steps", [])
            counts = plan.get("counts", {})
        else:
            steps = plan
            counts = {}
        if not isinstance(steps, list):
            return
        if self.plain:
            total = len(steps)
            done = sum(
                1
                for step in steps
                if isinstance(step, dict) and step.get("status") == "completed"
            )
            self.info(f"Plan · {done}/{total} done")
            for index, step in enumerate(steps, start=1):
                if not isinstance(step, dict):
                    continue
                status = str(step.get("status", "pending"))
                mark = {"completed": "[x]", "in_progress": "[>]"}.get(status, "[ ]")
                self.info(f"  {index}. {mark} {step.get('step', '')}")
            return
        with self._lock:
            was_live = self._spinner.active
            self._stop_live()
            self._write("")
            total = len(steps)
            done = sum(1 for step in steps if isinstance(step, dict) and step.get("status") == "completed")
            heading = f"  {title}  ·  {done}/{total} done"
            self._write(self.theme.paint(heading, fg="accent", bold=True))
            if not steps:
                self._write("  " + self.theme.paint("No steps.", fg="dim"))
            for index, step in enumerate(steps, start=1):
                if not isinstance(step, dict):
                    continue
                status = str(step.get("status", "pending"))
                text = str(step.get("step", ""))
                if status == "completed":
                    glyph, role = ("✓", "success") if self.glyphs != "ascii" else ("+", "success")
                elif status == "in_progress":
                    glyph, role = ("▶", "accent") if self.glyphs != "ascii" else (">", "accent")
                else:
                    glyph, role = ("○", "muted") if self.glyphs != "ascii" else ("-", "muted")
                self._write(
                    "  "
                    + self.theme.paint(glyph, fg=role, bold=status == "in_progress")
                    + " "
                    + self.theme.paint(clip(f"{index}. {text}", self.width - 6), fg="dim" if status == "completed" else "text")
                )
            self._write("")
            if was_live:
                self._spinner.start("working")
            self._turn_started = True

    def data_table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[str]],
        *,
        widths: list[int] | None = None,
        empty: str = "Nothing to show.",
    ) -> None:
        if not rows:
            self.info(empty)
            return
        self.info("")
        self.info("  " + self.theme.paint(title, fg="accent", bold=True))
        lines = table(self.theme, headers, rows, widths=widths, glyphs=self.glyphs)
        for line in lines:
            self.info("  " + line)
        self.info("")

    # -------------------------------------------------------------------- input

    def _setup_readline(self) -> None:
        if self.plain:
            return
        try:
            import readline

            self._readline = readline
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._history_path.parent.chmod(0o700)
            except OSError:
                pass
            if self._history_path.is_file():
                try:
                    readline.read_history_file(str(self._history_path))
                except OSError:
                    pass
            readline.set_history_length(500)
            if hasattr(readline, "set_auto_history"):
                readline.set_auto_history(False)
            self._configure_completer()
        except (ImportError, OSError, AttributeError):
            self._readline = None

    def _configure_completer(self) -> None:
        readline = self._readline
        if readline is None:
            return
        words = list(self._completions)

        def complete(text: str, state: int):
            matches = [word for word in words if word.startswith(text)]
            return matches[state] if state < len(matches) else None

        readline.set_completer(complete)
        readline.set_completer_delims(" \t\n`\"'")
        try:
            readline.parse_and_bind("tab: complete")
        except Exception:
            pass

    def _remember(self, line: str) -> None:
        if not line.strip() or self._readline is None:
            return
        try:
            self._readline.add_history(line)
            self._readline.write_history_file(str(self._history_path))
            try:
                self._history_path.chmod(0o600)
            except OSError:
                pass
        except (OSError, ValueError):
            pass

    def read_prompt(self) -> str | None:
        if self._broken:
            return None
        if self.plain:
            line = self.stdin.readline()
            if line == "":
                return None
            return line.rstrip("\n")
        glyph = "❯" if self.glyphs != "ascii" else ">"
        prompt = self.theme.paint(glyph + " ", fg="accent", bold=True)
        try:
            line = input(prompt)
        except EOFError:
            self._write("")
            return None
        except KeyboardInterrupt:
            self._write("")
            raise
        parts = [line]
        while parts[-1].rstrip().endswith("\\"):
            parts[-1] = parts[-1].rstrip()[:-1]
            try:
                continuation = input(self.theme.paint("… ", fg="dim"))
            except EOFError:
                break
            parts.append(continuation)
        result = "\n".join(parts)
        self._remember(result)
        if result.strip():
            self._turn_started = False
        return result

    # ------------------------------------------------------------------- tables

    def lanes_table(self, rows: list[dict]) -> None:
        if not rows:
            self.info("  No lanes configured.")
            return
        headers = ["Lane", "Member", "Health", "Load", "Avg", "Cooldown", "Fails"]
        data = []
        for row in rows:
            cooldown = float(row.get("cooldown_remaining_s", 0) or 0)
            ewma = row.get("ewma_ms")
            data.append(
                [
                    str(row.get("lane", "")),
                    str(row.get("member", "")),
                    str(row.get("health", "")),
                    str(row.get("inflight", 0)),
                    f"{float(ewma):.0f}ms" if ewma is not None else "-",
                    f"{cooldown:.0f}s" if cooldown > 0 else "-",
                    str(row.get("failures", 0)),
                ]
            )
        self.info("")
        self.info("  " + self.theme.paint("Lanes", fg="accent", bold=True))
        lines = table(self.theme, headers, data, glyphs=self.glyphs)
        for line in lines:
            self.info("  " + line)
        self.info("")

    def sessions_table(self, rows: list[dict]) -> None:
        if not rows:
            self.info("No saved sessions.")
            return
        headers = ["Session", "Updated", "Msgs", "Title"]
        data = [
            [
                str(row.get("id", "")),
                str(row.get("updated_at", "")),
                str(row.get("message_count", "")),
                str(row.get("title", "")),
            ]
            for row in rows
        ]
        self.info("")
        self.info("  " + self.theme.paint("Sessions", fg="accent", bold=True))
        lines = table(
            self.theme,
            headers,
            data,
            widths=[30, 17, 5, max(20, min(60, self.width - 60))],
            glyphs=self.glyphs,
        )
        for line in lines:
            self.info("  " + line)
        self.info("")
