"""Worktree binding without the executor: a read-only projection of the workspace.

OHAR records what the workspace *is* — root, branch, detached or not, revision
prefix, dirty count, known worktrees — so session continuity can later detect
that the work moved. Every git call is argv-only with a timeout and no shell,
and each one fails closed: a missing git, a non-repository path or a hung
command yields an empty fact, never an invented one.

The recipes are copyable instructions for isolating the work in a fresh
worktree. This module never creates a worktree, never runs a mutating command,
and never launches an external coding agent.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ohar.terminal import clip, sanitize_text
from ohar.tools import ToolError, resolve_path
from ohar.types import Tool, ToolContext, ToolResult

from .records import append_event, guarded_update, read_record
from .state import SessionState, load_state, open_state, save_state

__all__ = [
    "WORKSPACE_SCHEMA_VERSION",
    "GIT_TIMEOUT",
    "MAX_WORKTREES",
    "MAX_DIRTY",
    "WorkspaceError",
    "WorkspaceInfo",
    "LaunchRecipe",
    "probe",
    "bind",
    "binding",
    "launch_recipes",
    "workspace_line",
    "make_workspace_tool",
]

WORKSPACE_SCHEMA_VERSION = "ohar_workspace_binding/v1"
GIT_TIMEOUT = 5.0
MAX_WORKTREES = 16
MAX_DIRTY = 999

WORKSPACE_RECORD = "workspace.json"
WORKSPACE_RECORD_TYPE = "workspace_binding"
BOUND_EVENT = "workspace_bound"
REVISION_PREFIX = 12
MAX_RECIPES = 4
MAX_LINE_CHARS = 200
MAX_PATH_CHARS = 240
MAX_WORKTREE_PATH_CHARS = 160
_HEX_RE = re.compile(r"[0-9a-f]{7,40}")

CLAIM_BOUNDARY = (
    "This is a read-only projection of the workspace recorded for the session. It is not a "
    "worktree creation, an executor launch, a file change, or verification evidence."
)
_NOT_EVIDENCE_UNTIL_OBSERVED = (
    "revision_observed",
    "worktree_creation",
    "executor_dispatch",
    "verification",
)


class WorkspaceError(ValueError):
    """The workspace cannot be bound or read back."""


@dataclass(frozen=True, slots=True)
class WorkspaceInfo:
    """One bounded, observed snapshot of a directory as a repository."""

    root: str
    is_repo: bool
    branch: str
    detached: bool
    revision: str
    dirty: int
    worktrees: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "is_repo": self.is_repo,
            "branch": self.branch,
            "detached": self.detached,
            "revision": self.revision,
            "dirty": self.dirty,
            "worktrees": list(self.worktrees),
        }


@dataclass(frozen=True, slots=True)
class LaunchRecipe:
    """A copy-only command: shown to the user, never run here."""

    recipe_id: str
    label: str
    command: str
    when_to_use: str


def probe(path: Path | str, *, timeout: float = GIT_TIMEOUT) -> WorkspaceInfo:
    """Read the workspace as a repository, failing closed on any git absence."""
    target = _target_dir(path)
    if target is None:
        return WorkspaceInfo(_text(path), False, "", False, "", 0, ())

    toplevel = _run_git(["rev-parse", "--show-toplevel"], target, timeout)
    root = (toplevel or "").strip()
    if not root:
        return WorkspaceInfo(str(target), False, "", False, "", 0, ())

    branch, detached = _branch_state(root, timeout)
    return WorkspaceInfo(
        root=root,
        is_repo=True,
        branch=branch,
        detached=detached,
        revision=_revision(root, timeout),
        dirty=_dirty_count(root, timeout),
        worktrees=_worktrees(root, timeout),
    )


def bind(session_id: str, path: str | Path, ctx: ToolContext) -> SessionState:
    """Confine ``path`` to the workspace, observe it, and record it on the session."""
    raw = path if isinstance(path, (str, Path)) else ""
    if not str(raw).strip():
        raise WorkspaceError("A workspace path is required.")
    if not isinstance(ctx, ToolContext):
        raise WorkspaceError("A tool context is required to confine the workspace path.")
    try:
        resolved = resolve_path(ctx, str(raw), must_exist=True)
    except ToolError as exc:
        raise WorkspaceError(str(exc)) from None
    if not resolved.is_dir():
        raise WorkspaceError(f"Workspace is not a directory: {clip(resolved.name, 60)}")

    info = probe(resolved)
    state = load_state(session_id)
    if state is None:
        state, _resumed = open_state(session_id, goal="", workspace=str(resolved))
    else:
        state.workspace = str(resolved)
        state = save_state(state)

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        record = dict(current)
        record.update(
            {
                "record_type": WORKSPACE_RECORD_TYPE,
                "binding_schema_version": WORKSPACE_SCHEMA_VERSION,
                "session_id": session_id,
                "path": str(resolved),
                "bound_at": _now_iso(),
                "workspace": info.to_dict(),
            }
        )
        return record

    guarded_update(WORKSPACE_RECORD, session_id, mutate, operation="workspace_bind")
    append_event(
        session_id,
        BOUND_EVENT,
        {
            "path": str(resolved),
            "root": info.root,
            "is_repo": info.is_repo,
            "branch": info.branch,
            "revision": info.revision,
            "dirty": info.dirty,
            "worktrees": len(info.worktrees),
        },
    )
    return state


def binding(session_id: str, ctx: ToolContext | None = None) -> dict[str, Any]:
    """Read the recorded workspace back: observed record, filesystem only, or missing."""
    state = load_state(session_id)
    record = read_record(WORKSPACE_RECORD, session_id)
    ours = record.get("record_type") == WORKSPACE_RECORD_TYPE
    recorded = record.get("workspace")
    stored = _info_from(recorded) if ours and isinstance(recorded, Mapping) else None
    path = _clean_path(record.get("path")) or _clean_path(state.workspace if state else "")
    if path and ctx is not None:
        path = _confined(path, ctx)

    exists = bool(path) and Path(path).is_dir()
    if not path:
        status, info = "unknown", WorkspaceInfo("", False, "", False, "", 0, ())
    elif stored is not None and exists:
        status, info = "ready_observed", stored
    elif exists:
        status, info = "filesystem_only", probe(path)
    else:
        status, info = "missing", stored or WorkspaceInfo(path, False, "", False, "", 0, ())

    payload: dict[str, Any] = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "session_id": session_id,
        "status": status,
        "path": path,
        "workspace": info.to_dict(),
        "branch": info.branch,
        "detached": info.detached,
        "revision": info.revision,
        "dirty": info.dirty,
        "worktrees": list(info.worktrees),
        "launch_recipes": [_recipe_dict(recipe) for recipe in launch_recipes(info)][:MAX_RECIPES],
        "next_action": _next_action(status, info),
        "claim_boundary": CLAIM_BOUNDARY,
        "line": workspace_line(info),
    }
    bound_at = record.get("bound_at") if ours else None
    if isinstance(bound_at, str) and bound_at.strip():
        payload["bound_at"] = sanitize_text(bound_at, keep_newlines=False).strip()[:40]
    if status != "ready_observed":
        payload["not_evidence_until_observed"] = list(_NOT_EVIDENCE_UNTIL_OBSERVED)
    return payload


def launch_recipes(info: WorkspaceInfo) -> list[LaunchRecipe]:
    """Copyable, shell-quoted commands; OHAR runs none of them."""
    if not info.root.strip():
        return []
    root = shlex.quote(info.root)
    recipes = [
        LaunchRecipe(
            recipe_id="open_workspace",
            label="Open this workspace",
            command=f"cd {root}",
            when_to_use="Use when the terminal is not already in the recorded workspace.",
        )
    ]
    if not info.is_repo:
        return recipes

    recipes.append(
        LaunchRecipe(
            recipe_id="inspect_changes",
            label="Inspect uncommitted changes",
            command=f"git -C {root} --no-optional-locks status --short",
            when_to_use=f"Use to list the {info.dirty} uncommitted change(s) OHAR counted.",
        )
    )
    target = shlex.quote(_worktree_target(info))
    if info.detached or not info.branch:
        isolation = f"git -C {root} worktree add {target} --detach"
        use = "Use to isolate a detached revision; the new worktree starts detached at it."
    else:
        branch = shlex.quote(f"{info.branch}-ohar")
        isolation = f"git -C {root} worktree add {target} -b {branch}"
        use = "Use to isolate this work in a fresh worktree on its own branch."
    recipes.append(
        LaunchRecipe(
            recipe_id="isolate_worktree",
            label="Isolate the work in a fresh worktree",
            command=isolation,
            when_to_use=f"{use} OHAR never creates the worktree and never runs this.",
        )
    )
    revision = info.revision if _HEX_RE.fullmatch(info.revision or "") else ""
    recipes.append(
        LaunchRecipe(
            recipe_id="confirm_revision",
            label="Confirm the recorded revision",
            command=f"git -C {root} --no-pager log -1 --oneline{f' {revision}' if revision else ''}",
            when_to_use="Use to check the session is still bound to the commit you expect.",
        )
    )
    return recipes[:MAX_RECIPES]


def workspace_line(info: WorkspaceInfo, *, glyphs: str = "unicode") -> str:
    """One terminal line: where the session is, at which revision, how dirty."""
    sep = "-" if glyphs == "ascii" else "·"
    name = clip(_basename(info.root) or "workspace", 32)
    if not info.is_repo:
        return clip(f"workspace {name} {sep} not a repository", MAX_LINE_CHARS)
    where = info.branch or ("detached" if info.detached else "no branch")
    revision = info.revision or "no revision"
    worktrees = len(info.worktrees)
    parts = [
        f"workspace {name}",
        f"{where}@{revision}",
        f"{info.dirty} dirty",
        f"{worktrees} worktree{'' if worktrees == 1 else 's'}",
    ]
    return clip(f" {sep} ".join(parts), MAX_LINE_CHARS)


def make_workspace_tool(ctx: ToolContext, session_id: str) -> Tool:
    """Build the read-only ``workspace_status`` tool for one session."""

    def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        unexpected = sorted(set(args or {}) - set())
        if unexpected:
            return ToolResult.failure(
                f"Unknown argument(s): {', '.join(unexpected)}; nothing was run."
            )
        try:
            payload = binding(session_id, ctx)
        except (WorkspaceError, OSError, ValueError) as exc:
            return ToolResult.failure(f"workspace_status: {sanitize_text(str(exc))[:200]}")
        return ToolResult.success(
            _tool_text(payload),
            status=str(payload.get("status", "unknown")),
            next_action=str(payload.get("next_action", "")),
            recipes=len(payload.get("launch_recipes", []) or []),
        )

    return Tool(
        name="workspace_status",
        description=(
            "Report the workspace bound to this session: root, branch, revision, "
            "dirty count and known worktrees, plus copy-only recipes for isolating "
            "the work in a fresh worktree. Reads only; nothing is created or run."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=handler,
        read_only=True,
    )


def _run_git(argv: Sequence[str], cwd: Path | str, timeout: float) -> str | None:
    """One argv-only git call. No shell, and any failure reads as 'unknown'."""
    try:
        ceiling = max(0.1, float(timeout))
    except (TypeError, ValueError):
        ceiling = GIT_TIMEOUT
    command = ["git", "-C", str(cwd), "--no-optional-locks", *argv]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "LC_ALL": "C"}
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=ceiling,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout or ""


def _branch_state(root: str, timeout: float) -> tuple[str, bool]:
    short = _run_git(["symbolic-ref", "--quiet", "--short", "HEAD"], Path(root), timeout)
    if short and short.strip():
        return _clean_ref(short), False
    name = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], Path(root), timeout)
    if not name or not name.strip():
        return "", False
    branch = _clean_ref(name)
    if branch == "HEAD":
        return "", True
    return branch, False


def _revision(root: str, timeout: float) -> str:
    head = _run_git(["rev-parse", "--verify", "HEAD"], Path(root), timeout)
    value = (head or "").strip().casefold()
    if not _HEX_RE.fullmatch(value):
        return ""
    return value[:REVISION_PREFIX]


def _dirty_count(root: str, timeout: float) -> int:
    status = _run_git(["status", "--porcelain"], Path(root), timeout)
    if status is None:
        return 0
    count = sum(1 for line in status.splitlines() if line.strip())
    return max(0, min(count, MAX_DIRTY))


def _worktrees(root: str, timeout: float) -> tuple[str, ...]:
    listing = _run_git(["worktree", "list", "--porcelain"], Path(root), timeout)
    if not listing:
        return ()
    found: list[str] = []
    for line in listing.splitlines():
        if not line.startswith("worktree "):
            continue
        path = _clean_path(line[len("worktree "):], MAX_WORKTREE_PATH_CHARS)
        if path and path != root:
            found.append(path)
    return tuple(found[:MAX_WORKTREES])


def _target_dir(path: Path | str) -> Path | None:
    if isinstance(path, Path):
        candidate = path
    elif isinstance(path, str) and path.strip():
        candidate = Path(path)
    else:
        return None
    try:
        resolved = candidate.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        if resolved.is_dir():
            return resolved
        if resolved.is_file() and resolved.parent.is_dir():
            return resolved.parent
    except OSError:
        return None
    return None


def _info_from(data: Mapping[str, Any]) -> WorkspaceInfo | None:
    root = _clean_path(data.get("root"))
    if not root:
        return None
    worktrees = data.get("worktrees")
    items = worktrees if isinstance(worktrees, (list, tuple)) else []
    dirty = data.get("dirty")
    dirty_count = dirty if isinstance(dirty, int) and not isinstance(dirty, bool) else 0
    revision = _clean_path(data.get("revision")).casefold()
    return WorkspaceInfo(
        root=root,
        is_repo=bool(data.get("is_repo")),
        branch=_clean_ref(data.get("branch")),
        detached=bool(data.get("detached")),
        revision=revision[:REVISION_PREFIX] if _HEX_RE.fullmatch(revision) else "",
        dirty=max(0, min(dirty_count, MAX_DIRTY)),
        worktrees=tuple(
            _clean_path(item, MAX_WORKTREE_PATH_CHARS) for item in items if _clean_path(item, MAX_WORKTREE_PATH_CHARS)
        )[:MAX_WORKTREES],
    )


def _recipe_dict(recipe: LaunchRecipe) -> dict[str, Any]:
    return {
        "recipe_id": recipe.recipe_id,
        "label": recipe.label,
        "command": recipe.command,
        "when_to_use": recipe.when_to_use,
    }


def _tool_text(payload: Mapping[str, Any]) -> str:
    lines = [str(payload.get("line") or "workspace unknown"), f"status: {payload.get('status')}"]
    path = str(payload.get("path") or "")
    if path:
        lines.append(f"path: {clip(path, MAX_PATH_CHARS)}")
    for key in ("branch", "revision", "dirty", "worktrees"):
        lines.append(f"{key}: {_render_value(payload.get(key))}")
    recipes = payload.get("launch_recipes")
    if isinstance(recipes, list) and recipes:
        lines.append("copy-only recipes (OHAR does not run these):")
        for recipe in recipes:
            if isinstance(recipe, Mapping):
                lines.append(f"- {recipe.get('recipe_id')}: {recipe.get('command')}")
    else:
        lines.append("copy-only recipes: none")
    lines.append(f"next action: {payload.get('next_action')}")
    lines.append(str(payload.get("claim_boundary")))
    return "\n".join(lines)


def _render_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "(none)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value == 0:
        return "0"
    return sanitize_text(str(value if value is not None else ""), keep_newlines=False)[:MAX_PATH_CHARS] or "(unknown)"


def _next_action(status: str, info: WorkspaceInfo) -> str:
    if status == "unknown":
        return "bind_workspace"
    if status == "missing":
        return "rebind_workspace"
    if status == "filesystem_only":
        return "observe_workspace_revision"
    if info.detached:
        return "confirm_detached_revision"
    if info.dirty:
        return "isolate_work_in_worktree"
    return "continue_session"


def _worktree_target(info: WorkspaceInfo) -> str:
    root = Path(info.root or ".")
    name = root.name or "workspace"
    return str(root.parent / f"{name}-ohar")


def _confined(path: str, ctx: ToolContext) -> str:
    try:
        return str(resolve_path(ctx, path))
    except ToolError:
        # A recorded path outside this workspace is reported as unreadable, not followed.
        return ""


def _clean_path(value: object, limit: int = MAX_PATH_CHARS) -> str:
    text = " ".join(sanitize_text(str(value or ""), keep_newlines=False).split())
    return text[:limit]


def _clean_ref(value: object) -> str:
    text = " ".join(sanitize_text(str(value or ""), keep_newlines=False).split())
    return text[:80]


def _basename(path: str) -> str:
    text = _clean_path(path).rstrip(os.sep)
    if not text:
        return ""
    return Path(text).name or text


def _text(value: object) -> str:
    return _clean_path(str(value if isinstance(value, (str, Path)) else ""))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
