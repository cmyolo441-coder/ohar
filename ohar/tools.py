"""Confined local tools for OHAR agents. Standard library only."""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .types import ApprovalRequest, Tool, ToolContext, ToolResult

__all__ = ["Toolbox", "default_tools", "resolve_path"]

MAX_READ_CHARS = 24_000
MAX_OUTPUT_CHARS = 30_000
MAX_FILE_BYTES = 2 * 1024 * 1024
SKIP_DIRS = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", ".ohar", "dist", "build"}
)
_DENIED = "Permission denied by user; no changes were made."

_ANSI_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Root-destroying and host-destroying commands are refused before approval.
_BLOCKED_COMMANDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\brm\b(?:\s+(?:--[\w-]*|-[A-Za-z][\w-]*))*\s+/{1,2}(?:\.{1,2}|\*+)?(?=\s|$|[;&|])",
            re.IGNORECASE,
        ),
        "recursive deletion of /",
    ),
    (re.compile(r"\bmkfs(?:\.[A-Za-z0-9]+)?\b", re.IGNORECASE), "filesystem formatting"),
    (re.compile(r"\b(?:shutdown|reboot)\b", re.IGNORECASE), "system shutdown or reboot"),
    (
        re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
        "fork bomb",
    ),
)


class ToolError(ValueError):
    """An expected, user-correctable tool failure."""


def _sanitize(text: str) -> str:
    """Strip ANSI escapes and control characters, preserving newlines and tabs."""
    return _CONTROL_RE.sub("", _ANSI_RE.sub("", text))


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data


def _truncate(text: str, limit: int) -> str:
    """Keep 60% of the head and 40% of the tail with an explicit marker."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head_len = int(limit * 0.6)
    tail_len = limit - head_len
    omitted = len(text) - head_len - tail_len
    head, tail = text[:head_len], text[-tail_len:] if tail_len else ""
    return f"{head}\n… [truncated {omitted} chars] …\n{tail}"


def _fail(message: str, **meta: Any) -> ToolResult:
    return ToolResult.failure(_sanitize(message), **meta)


def _workspace(ctx: ToolContext) -> Path:
    try:
        return Path(ctx.workspace).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ToolError(f"Workspace cannot be resolved: {exc}") from None


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def resolve_path(
    ctx: ToolContext,
    raw: str,
    *,
    must_exist: bool = False,
    for_write: bool = False,
) -> Path:
    """Resolve a user path, confining it to the workspace.

    Symlinks are resolved (including parents of not-yet-created files), so a
    link that points outside the workspace is rejected rather than followed.
    Writes to the workspace root and to .git/.ohar are always refused.
    """
    if not isinstance(raw, str):
        raise ToolError("Path must be a string.")
    if not raw.strip():
        raise ToolError("Path cannot be empty.")
    if "\x00" in raw:
        raise ToolError("Path cannot contain NUL bytes.")
    try:
        expanded = Path(os.path.expanduser(raw))
    except (OSError, RuntimeError):
        raise ToolError("Path cannot be expanded.") from None
    workspace = _workspace(ctx)
    candidate = expanded if expanded.is_absolute() else workspace / expanded
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        raise ToolError(f"Cannot resolve path: {_sanitize(raw)}") from None
    if not _inside(resolved, workspace):
        raise ToolError(f"Path '{_sanitize(raw)}' is outside the workspace.")
    if must_exist and not resolved.exists():
        raise ToolError(f"Path does not exist: {_sanitize(raw)}")
    if for_write:
        if resolved == workspace:
            raise ToolError("Refusing to write to the workspace root.")
        try:
            rel = resolved.relative_to(workspace)
        except ValueError:
            raise ToolError(f"Path '{_sanitize(raw)}' is outside the workspace.") from None
        for part in rel.parts:
            if part in {".git", ".ohar"}:
                raise ToolError(f"Refusing to write inside '{part}'.")
    return resolved


def _rel(ctx: ToolContext, path: Path) -> str:
    try:
        rel = path.relative_to(_workspace(ctx))
    except ValueError:
        return path.as_posix()
    return "." if rel == Path(".") else rel.as_posix()


def _str(args: dict[str, Any], key: str, default: str | None = None) -> str:
    if key not in args or args[key] is None:
        if default is None:
            raise ToolError(f"Missing required argument '{key}'.")
        return default
    value = args[key]
    if not isinstance(value, str):
        raise ToolError(f"Argument '{key}' must be a string.")
    return value


def _optional_str(args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolError(f"Argument '{key}' must be a string.")
    return value


def _int(
    args: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = args.get(key, default)
    if isinstance(value, bool):
        raise ToolError(f"Argument '{key}' must be an integer.")
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            raise ToolError(f"Argument '{key}' must be an integer.") from None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ToolError(f"Argument '{key}' must be an integer.")
    if minimum is not None and value < minimum:
        raise ToolError(f"Argument '{key}' must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ToolError(f"Argument '{key}' must be <= {maximum}.")
    return value


def _bool(args: dict[str, Any], key: str, default: bool = False) -> bool:
    value = args.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ToolError(f"Argument '{key}' must be a boolean.")


def _parse_arguments(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise ToolError(
            "Arguments must be a JSON object or a JSON object string."
        )
    text = arguments.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolError(
            f"Invalid JSON arguments: {exc.msg} at line {exc.lineno}, column {exc.colno}."
        ) from None
    if not isinstance(value, dict):
        raise ToolError("Arguments must be a JSON object with named parameters.")
    return value


def _approve(
    ctx: ToolContext,
    action: str,
    summary: str,
    detail: str,
    risk: str,
) -> ToolResult | None:
    if ctx.read_only:
        return ToolResult.failure(
            "Read-only mode: this action is disabled.", denied=True
        )
    request = ApprovalRequest(
        action=action,
        summary=_sanitize(summary)[:500],
        detail=_truncate(_sanitize(detail), 8_000),
        risk=risk,
    )
    if not ctx.request_approval(request):
        return ToolResult.failure(_DENIED, denied=True)
    return None


class Toolbox:
    """An immutable, ordered set of tools bound to one ToolContext."""

    def __init__(
        self,
        ctx: ToolContext | Iterable[Tool],
        tools: Iterable[Tool] | ToolContext | None = None,
    ) -> None:
        if isinstance(ctx, ToolContext):
            context, source = ctx, tools
        elif isinstance(tools, ToolContext):
            context, source = tools, ctx
        else:
            raise TypeError(
                "Toolbox requires a ToolContext and an optional iterable of Tools."
            )
        if isinstance(source, ToolContext):
            raise TypeError("Toolbox accepts only one ToolContext.")
        self._ctx = context
        self._tools: dict[str, Tool] = {}
        for tool in default_tools(context) if source is None else source:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def specs(self) -> list[dict[str, Any]]:
        return [tool.api_spec() for tool in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, arguments: dict[str, Any] | str) -> ToolResult:
        if not isinstance(name, str):
            return ToolResult.failure("Tool name must be a string.")
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "(none)"
            return ToolResult.failure(
                f"Unknown tool '{_sanitize(name)}'. Available: {available}"
            )
        try:
            args = _parse_arguments(arguments)
            result = tool.handler(args, self._ctx)
            if not isinstance(result, ToolResult):
                return ToolResult.failure(f"{name} returned an invalid tool result.")
            return result
        except ToolError as exc:
            return _fail(f"{name}: {exc}")
        except Exception as exc:  # never let a handler crash the agent
            message = _sanitize(str(exc)) or type(exc).__name__
            return _fail(f"{name} failed unexpectedly: {message}")


def _read_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw = _str(args, "path")
    offset = _int(args, "offset", 1, minimum=1)
    limit = min(_int(args, "limit", 400, minimum=1), 2_000)
    path = resolve_path(ctx, raw, must_exist=True)
    display = _rel(ctx, path)
    if not path.is_file():
        raise ToolError(f"Not a file: {display}")
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            return _fail(
                f"File is too large to read safely ({size:,} bytes; limit "
                f"{MAX_FILE_BYTES:,}). Use grep or a shell command to inspect a slice.",
                too_large=True,
            )
        data = path.read_bytes()
    except OSError as exc:
        raise ToolError(f"Cannot read {display}: {exc.strerror or exc}") from None
    if _is_binary(data):
        return _fail(f"Binary file (NUL bytes detected): {display}", binary=True)
    text = _sanitize(data.decode("utf-8", errors="replace"))
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return ToolResult.success(
            "", total_lines=0, offset=offset, next_offset=None, truncated=False
        )
    if offset > total:
        return _fail(
            f"Offset {offset} is beyond the end of {display} ({total} lines)."
        )

    start = offset - 1
    stop = min(total, start + limit)
    width = len(str(stop))
    output: list[str] = []
    used = 0
    truncated = False
    next_offset: int | None = None
    for number in range(offset, stop + 1):
        rendered = f"{number:>{width}}\t{lines[number - 1]}"
        cost = len(rendered) + (1 if output else 0)
        if used + cost > MAX_READ_CHARS:
            if not output:
                output.append(_truncate(rendered, max(1, MAX_READ_CHARS - 64)))
                next_offset = number + 1 if number < total else None
            else:
                next_offset = number
            truncated = True
            break
        output.append(rendered)
        used += cost
    else:
        if stop < total:
            truncated = True
            next_offset = stop + 1

    return ToolResult.success(
        "\n".join(output),
        total_lines=total,
        offset=offset,
        next_offset=next_offset,
        truncated=truncated,
    )


def _list_files(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw = _str(args, "path", ".")
    depth = _int(args, "depth", 2, minimum=1, maximum=10)
    base = resolve_path(ctx, raw, must_exist=True)
    if not base.is_dir():
        raise ToolError(f"Not a directory: {_rel(ctx, base)}")

    entries: list[str] = []
    truncated = False

    def walk(directory: Path, level: int) -> None:
        nonlocal truncated
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            return
        for child in children:
            if len(entries) >= 500:
                truncated = True
                return
            is_dir = child.is_dir()
            if is_dir and child.name in SKIP_DIRS:
                continue
            rel = _sanitize(_rel(ctx, child))
            entries.append(f"{rel}/" if is_dir else rel)
            if is_dir and level < depth and not child.is_symlink():
                walk(child, level + 1)

    walk(base, 1)
    entries.sort()
    content = "\n".join(entries) if entries else "No files found."
    return ToolResult.success(
        content, path=_sanitize(_rel(ctx, base)), count=len(entries), truncated=truncated
    )


def _glob(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    pattern = _str(args, "pattern")
    raw = _str(args, "path", ".")
    max_results = _int(args, "max_results", 200, minimum=1, maximum=1_000)
    if not pattern.strip():
        raise ToolError("Pattern cannot be empty.")
    if "\x00" in pattern:
        raise ToolError("Pattern cannot contain NUL bytes.")
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ToolError("Pattern must be relative and cannot contain '..'.")
    base = resolve_path(ctx, raw, must_exist=True)
    if not base.is_dir():
        raise ToolError(f"Not a directory: {_rel(ctx, base)}")
    workspace = _workspace(ctx)
    matches: list[str] = []
    truncated = False
    try:
        for match in base.glob(pattern):
            try:
                rel = match.relative_to(workspace)
            except ValueError:
                continue
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if not match.is_file():
                continue
            try:
                resolved = match.resolve(strict=False)
            except OSError:
                continue
            if not _inside(resolved, workspace):
                continue
            matches.append(_sanitize(rel.as_posix()))
            if len(matches) > max_results:
                truncated = True
                matches.pop()
                break
    except (OSError, ValueError, NotImplementedError) as exc:
        raise ToolError(f"Invalid glob pattern: {exc}") from None
    matches.sort()
    content = "\n".join(matches) if matches else "No files matched."
    return ToolResult.success(content, count=len(matches), truncated=truncated)


def _iter_files(base: Path, ctx: ToolContext) -> Iterator[Path]:
    """Yield regular files under base, without following symlinks or heavy dirs."""
    workspace = _workspace(ctx)
    if base.is_file():
        yield base
        return
    for root, dirs, files in os.walk(base, topdown=True, followlinks=False):
        root_path = Path(root)
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in SKIP_DIRS and not (root_path / name).is_symlink()
        )
        for name in sorted(files):
            path = root_path / name
            try:
                resolved = path.resolve(strict=False)
            except OSError:
                continue
            if resolved.is_file() and _inside(resolved, workspace):
                yield path


def _grep(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    pattern = _str(args, "pattern")
    raw = _str(args, "path", ".")
    name_glob = _optional_str(args, "glob")
    case_insensitive = _bool(args, "case_insensitive", False)
    context = _int(args, "context", 0, minimum=0, maximum=3)
    max_results = _int(args, "max_results", 200, minimum=1, maximum=1_000)
    try:
        regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
    except re.error as exc:
        raise ToolError(f"Invalid regular expression: {exc}") from None
    base = resolve_path(ctx, raw, must_exist=True)

    records: dict[tuple[str, int], tuple[bool, str]] = {}
    found = 0
    scanned = 0
    truncated = False
    for file_path in _iter_files(base, ctx):
        if name_glob and not (
            fnmatch.fnmatch(file_path.name, name_glob)
            or fnmatch.fnmatch(file_path.as_posix(), name_glob)
        ):
            continue
        try:
            if file_path.stat().st_size > MAX_FILE_BYTES:
                continue
            data = file_path.read_bytes()
        except OSError:
            continue
        if _is_binary(data):
            continue
        scanned += 1
        rel = _sanitize(_rel(ctx, file_path))
        lines = data.decode("utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            if not regex.search(line):
                continue
            found += 1
            if found > max_results:
                truncated = True
                break
            number = index + 1
            first = max(1, number - context)
            last = min(len(lines), number + context)
            for current in range(first, last + 1):
                key = (rel, current)
                was_match, line_text = records.get(key, (False, lines[current - 1]))
                records[key] = (was_match or current == number, line_text)
        if truncated:
            break

    output: list[str] = []
    used = 0
    char_truncated = False
    for (rel, number), (is_match, line) in sorted(records.items()):
        separator = ":" if is_match else "-"
        rendered = f"{rel}{separator}{number}{separator}{_truncate(_sanitize(line), 2_000)}"
        output.append(rendered)
        used += len(rendered) + 1
        if used > MAX_OUTPUT_CHARS:
            char_truncated = True
            break

    if output:
        content = "\n".join(output)
        if len(content) > MAX_OUTPUT_CHARS:
            content = _truncate(content, max(1, MAX_OUTPUT_CHARS - 64))
            char_truncated = True
    else:
        content = "No matches."
    return ToolResult.success(
        content,
        count=min(found, max_results),
        files=scanned,
        truncated=truncated or char_truncated,
    )


def _preview(text: str, max_lines: int = 30) -> str:
    lines = text.splitlines()
    shown = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        shown += f"\n… ({len(lines) - max_lines} more lines)"
    return shown or "(empty)"


def _diff_preview(old: str, new: str, display: str, max_lines: int = 60) -> str:
    chunks: list[str] = []
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"a/{display}",
        tofile=f"b/{display}",
        lineterm="",
    )
    for index, line in enumerate(diff):
        if index >= max_lines:
            chunks.append("… (diff truncated)")
            break
        chunks.append(line)
    return "\n".join(chunks) or "(no changes)"


def _read_text_for_diff(path: Path, limit: int = 256_000) -> str:
    try:
        return path.read_bytes()[:limit].decode("utf-8", errors="replace")
    except OSError:
        return ""


def _atomic_write(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError(
            f"Cannot create parent directory: {exc.strerror or exc}"
        ) from None
    mode: int | None = None
    if path.exists():
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            mode = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
    except OSError as exc:
        raise ToolError(f"Cannot create temporary file: {exc.strerror or exc}") from None
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode if mode is not None else 0o644)
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise ToolError(f"Cannot write {path.name}: {exc.strerror or exc}") from None
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw = _str(args, "path")
    if "content" not in args or args.get("content") is None:
        raise ToolError("Missing required argument 'content'.")
    content = _str(args, "content")
    overwrite = _bool(args, "overwrite", False)
    if ctx.read_only:
        return ToolResult.failure(
            "Read-only mode: write_file is disabled.", denied=True
        )
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError:
        raise ToolError("Content is not valid UTF-8 text.") from None
    if len(data) > MAX_FILE_BYTES:
        return _fail(
            f"Content is too large ({len(data)} bytes); the limit is {MAX_FILE_BYTES} bytes."
        )
    path = resolve_path(ctx, raw, for_write=True)
    display = _rel(ctx, path)
    if path.exists() and path.is_dir():
        return _fail(f"Path is a directory: {display}")
    exists = path.exists()
    if exists and not overwrite:
        return _fail(
            f"File already exists: {display}. Set overwrite=true to replace it."
        )

    if exists:
        old_text = _read_text_for_diff(path)
        try:
            old_size = path.stat().st_size
        except OSError:
            old_size = len(old_text)
        detail = (
            f"Overwrite {display} ({old_size} -> {len(data)} bytes)\n\n"
            f"{_diff_preview(old_text, content, display)}"
        )
        summary = f"Overwrite {display} ({len(data)} bytes)"
    else:
        detail = f"Create {display} ({len(data)} bytes)\n\n{_preview(content)}"
        summary = f"Create {display} ({len(data)} bytes)"
    denied = _approve(ctx, "write_file", summary, detail, "medium")
    if denied is not None:
        return denied

    _atomic_write(path, data)
    verb = "Overwrote" if exists else "Created"
    return ToolResult.success(
        _sanitize(f"{verb} {display} ({len(data)} bytes)."),
        path=display,
        bytes=len(data),
        created=not exists,
    )


def _edit_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw = _str(args, "path")
    old = _str(args, "old_string")
    new = _str(args, "new_string")
    replace_all = _bool(args, "replace_all", False)
    if ctx.read_only:
        return ToolResult.failure(
            "Read-only mode: edit_file is disabled.", denied=True
        )
    if not old:
        raise ToolError("Argument 'old_string' cannot be empty.")
    if old == new:
        return _fail("old_string and new_string are identical; no change was made.")
    path = resolve_path(ctx, raw, must_exist=True, for_write=True)
    display = _rel(ctx, path)
    if not path.is_file():
        return _fail(f"Not a file: {display}")
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return _fail(f"File is too large to edit: {display}")
        data = path.read_bytes()
    except OSError as exc:
        raise ToolError(f"Cannot read {display}: {exc.strerror or exc}") from None
    if _is_binary(data):
        return _fail(f"Binary file (NUL bytes detected): {display}", binary=True)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return _fail(f"File is not valid UTF-8 text: {display}")

    count = text.count(old)
    if count == 0:
        return _fail(f"old_string was not found in {display}.")
    if count > 1 and not replace_all:
        return _fail(
            f"old_string occurs {count} times in {display}; add surrounding "
            "context to make it unique or set replace_all=true."
        )
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    try:
        updated_data = updated.encode("utf-8")
    except UnicodeEncodeError:
        return _fail("Replacement is not valid UTF-8 text.")
    if len(updated_data) > MAX_FILE_BYTES:
        return _fail(f"Edited file would exceed {MAX_FILE_BYTES} bytes.")

    diff = _diff_preview(text, updated, display)
    denied = _approve(
        ctx,
        "edit_file",
        f"Edit {display} ({count} replacement{'s' if count != 1 else ''})",
        f"File: {display}\nReplacements: {count}\n\n{diff}",
        "medium",
    )
    if denied is not None:
        return denied

    _atomic_write(path, updated_data)
    suffix = "s" if count != 1 else ""
    return ToolResult.success(
        _sanitize(f"Edited {display} ({count} replacement{suffix}).\n{diff}"),
        path=display,
        replacements=count,
    )


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def _shell(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = _str(args, "command")
    raw_cwd = _str(args, "cwd", ".")
    if not command.strip():
        raise ToolError("Command cannot be empty.")
    if "\x00" in command:
        raise ToolError("Command cannot contain NUL bytes.")
    if ctx.read_only:
        return ToolResult.failure("Read-only mode: shell is disabled.", denied=True)
    requested = _int(args, "timeout", 120)
    try:
        ctx_limit = int(ctx.shell_timeout)
    except (TypeError, ValueError):
        ctx_limit = 120
    ctx_limit = max(1, min(600, ctx_limit))
    timeout = max(1, min(600, requested, ctx_limit))
    cwd = resolve_path(ctx, raw_cwd, must_exist=True)
    if not cwd.is_dir():
        return _fail(f"Not a directory: {_rel(ctx, cwd)}")
    for pattern, reason in _BLOCKED_COMMANDS:
        if pattern.search(command):
            return _fail(f"Blocked unsafe command ({reason}).", blocked=True)

    cwd_display = _rel(ctx, cwd)
    detail = (
        f"Command:\n{_truncate(command, 4_000)}\n\n"
        f"Directory: {cwd_display}\nTimeout: {timeout}s"
    )
    denied = _approve(
        ctx,
        "shell",
        f"Run shell command in {cwd_display} (timeout {timeout}s)",
        detail,
        "high",
    )
    if denied is not None:
        return denied

    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            executable=shutil.which("bash") or None,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        raise ToolError(f"Cannot start command: {exc}") from None

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except KeyboardInterrupt:
        _kill_process_group(process)
        try:
            process.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError):
            pass
        raise
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        partial = ""
        try:
            stdout, stderr = process.communicate(timeout=2)
            partial = f"{stdout or ''}{stderr or ''}"
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        finally:
            try:
                process.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                pass
        message = f"Command timed out after {timeout}s and was killed."
        if partial.strip():
            message += "\n" + _truncate(_sanitize(partial), 4_000)
        return _fail(
            message,
            timed_out=True,
            timeout_s=timeout,
            exit_code=None,
            duration_s=round(time.monotonic() - started, 3),
        )

    duration = round(time.monotonic() - started, 3)
    code = process.returncode if process.returncode is not None else -1
    combined = _sanitize(f"{stdout or ''}{stderr or ''}").rstrip("\n")
    footer = f"[exit code {code}; {duration:.3f}s]"
    if not combined.strip():
        combined = "(no output)"
    body = _truncate(combined, max(1_000, MAX_OUTPUT_CHARS - len(footer) - 100))
    content = f"{body}\n{footer}"
    meta = {"exit_code": code, "duration_s": duration}
    if code == 0:
        return ToolResult.success(content, **meta)
    return ToolResult.failure(content, **meta)


def default_tools(ctx: ToolContext) -> list[Tool]:
    """Build the standard local toolset for one context."""
    _ = ctx
    return [
        Tool(
            name="read_file",
            description=(
                "Read a text file with line numbers. Use offset and limit to page "
                "through long files; binary files are rejected."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path inside the workspace.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start at.",
                        "minimum": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines (default 400).",
                        "minimum": 1,
                        "maximum": 2_000,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=_read_file,
            read_only=True,
        ),
        Tool(
            name="list_files",
            description=(
                "List files and directories under a path, recursively up to depth. "
                "Directories end with '/', and heavy build/cache directories are skipped."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory inside the workspace (default '.').",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Recursion depth (default 2).",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=_list_files,
            read_only=True,
        ),
        Tool(
            name="glob",
            description=(
                "Find files by glob pattern (supports '**'). Returns workspace-relative "
                "paths; directories and heavy build/cache folders are excluded."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Relative glob pattern, e.g. 'src/**/*.py'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory inside the workspace (default '.').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum files to return (default 200).",
                        "minimum": 1,
                        "maximum": 1_000,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=_glob,
            read_only=True,
        ),
        Tool(
            name="grep",
            description=(
                "Search text file contents with a regular expression. Returns "
                "path:line:text for matches and path-line-text for context lines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python regular expression to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search (default '.').",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional filename glob filter, e.g. '*.py'.",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Match without regard to case.",
                    },
                    "context": {
                        "type": "integer",
                        "description": "Context lines before and after each match (0-3).",
                        "minimum": 0,
                        "maximum": 3,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum matching lines (default 200).",
                        "minimum": 1,
                        "maximum": 1_000,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=_grep,
            read_only=True,
        ),
        Tool(
            name="write_file",
            description=(
                "Create or overwrite a file with UTF-8 text. Requires approval; "
                "set overwrite=true to replace an existing file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path inside the workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 file content.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Replace an existing file.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=_write_file,
            destructive=True,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace an exact string in an existing text file. Requires approval; "
                "the match must be unique unless replace_all=true."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path inside the workspace.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": False,
            },
            handler=_edit_file,
            destructive=True,
        ),
        Tool(
            name="shell",
            description=(
                "Run a shell command in the workspace. Requires approval and is "
                "disabled in read-only mode; output and timeout are bounded."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120, max 600).",
                        "minimum": 1,
                        "maximum": 600,
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory inside the workspace (default '.').",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            handler=_shell,
            destructive=True,
        ),
    ]


