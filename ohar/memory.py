"""Explicit, workspace-scoped persistent memory for OHAR. Stdlib only."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import ApprovalRequest, Tool, ToolContext, ToolResult

__all__ = ["MemoryStore", "make_memory_tools"]
SCHEMA_VERSION = 1
MAX_TEXT_CHARS = 4_000
MAX_TAGS = 12
MAX_TAG_CHARS = 40
MAX_RECALL = 20
_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_ID_RE = re.compile(r"m-[0-9a-f]{8}")
_DENIED = "Memory change denied; nothing was changed."


def _clean(value: str, *, single_line: bool = False) -> str:
    text = _ANSI_RE.sub("", value)
    if single_line:
        text = _WS_RE.sub(" ", text.replace("\r", " ").replace("\n", " "))
    else:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_RE.sub("", text).strip()


def _clean_text(text: object) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    cleaned = _clean(text)[:MAX_TEXT_CHARS].strip()
    if not cleaned:
        raise ValueError("Memory text cannot be empty.")
    return cleaned


def _clean_tags(tags: list[str] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        items: list[Any] = tags.split(",")
    elif isinstance(tags, list):
        items = list(tags)
    else:
        raise TypeError("tags must be a string, a list of strings, or None.")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            raise TypeError("each tag must be a string.")
        tag = _clean(item, single_line=True)[:MAX_TAG_CHARS].strip()
        key = tag.casefold()
        if tag and key not in seen:
            seen.add(key)
            cleaned.append(tag)
            if len(cleaned) == MAX_TAGS:
                break
    return cleaned


def _clamp(value: object, default: int, maximum: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("limit must be an integer.")
    return max(1, min(maximum, value))


def _timestamp(record: dict[str, Any]) -> float:
    try:
        return datetime.fromisoformat(record["created_at"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return 0.0


def _copy(record: dict[str, Any]) -> dict[str, Any]:
    return {**record, "tags": list(record["tags"])}


def _score(record: dict[str, Any], tokens: list[str], query: str) -> int:
    text = record["text"].casefold()
    tag_text = " ".join(tag.casefold() for tag in record["tags"])
    text_tokens = set(_TOKEN_RE.findall(text))
    tag_tokens = set(_TOKEN_RE.findall(tag_text))
    if not tokens:
        return 1 if query in text or query in tag_text else 0
    score = 0
    for token in tokens:
        score += 7 if token in tag_tokens else 2 if token in tag_text else 0
        score += 3 if token in text_tokens else 1 if token in text else 0
    return score + (2 if query in text else 0) + (2 if query in tag_text else 0)


def _new_id(used: set[str]) -> str:
    for _ in range(32):
        memory_id = f"m-{secrets.token_hex(4)}"
        if memory_id not in used:
            return memory_id
    raise RuntimeError("Could not allocate a unique memory id.")


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class MemoryStore:
    """One private JSONL file per resolved workspace, protected by an RLock."""

    def __init__(self, root: Path, workspace: Path, *, max_records: int = 500) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int):
            raise TypeError("max_records must be an integer.")
        if max_records < 1:
            raise ValueError("max_records must be at least 1.")
        self.root = Path(root).expanduser()
        self.workspace = Path(workspace).expanduser().resolve(strict=False)
        self.max_records = max_records
        digest = hashlib.sha256(str(self.workspace).encode("utf-8")).hexdigest()
        self._path = self.root / f"{digest[:16]}.jsonl"
        self._workspace = str(self.workspace)
        self._lock = threading.RLock()
        self._ensure_root()

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    def _load_unlocked(self) -> list[dict[str, Any]]:
        try:
            lines = self._path.read_text(encoding="utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            return []
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                item = json.loads(line)
            except (TypeError, ValueError):
                continue
            if self._valid(item):
                records.append(item)
        return records

    def _valid(self, item: object) -> bool:
        if not isinstance(item, dict) or item.get("schema_version") != SCHEMA_VERSION:
            return False
        if not isinstance(item.get("id"), str) or not _ID_RE.fullmatch(item["id"]):
            return False
        if not isinstance(item.get("text"), str) or not item["text"]:
            return False
        tags = item.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            return False
        try:
            datetime.fromisoformat(item.get("created_at", ""))
        except (TypeError, ValueError):
            return False
        return item.get("workspace") == self._workspace

    def _write_unlocked(self, records: list[dict[str, Any]]) -> None:
        self._ensure_root()
        payload = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ).encode("utf-8")
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self._path.name}.", suffix=".tmp", dir=str(self.root)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self._path)
            _fsync_dir(self.root)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def remember(self, text: str, tags: list[str] | str | None = None) -> dict[str, Any]:
        clean_text = _clean_text(text)
        clean_tags = _clean_tags(tags)
        with self._lock:
            records = self._load_unlocked()
            record = {
                "schema_version": SCHEMA_VERSION,
                "id": _new_id({item["id"] for item in records}),
                "text": clean_text,
                "tags": clean_tags,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "workspace": self._workspace,
            }
            records.append(record)
            self._write_unlocked(records[-self.max_records :])
            return _copy(record)

    def recall(self, query: str = "", limit: int = 5) -> list[dict[str, Any]]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        limit = _clamp(limit, 5, MAX_RECALL)
        with self._lock:
            records = self._load_unlocked()
            if not query.strip():
                return self._recent_unlocked(records, limit)
            q = query.casefold()
            tokens = _TOKEN_RE.findall(q)
            scored = [
                (record, _score(record, tokens, q), index)
                for index, record in enumerate(records)
            ]
            scored = sorted(
                (item for item in scored if item[1] > 0),
                key=lambda item: (item[1], _timestamp(item[0]), item[2]),
                reverse=True,
            )
            return [_copy(item[0]) for item in scored[:limit]]

    def _recent_unlocked(
        self, records: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            enumerate(records),
            key=lambda item: (_timestamp(item[1]), item[0]),
            reverse=True,
        )
        return [_copy(record) for _, record in ordered[:limit]]

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        limit = _clamp(limit, 10, self.max_records)
        with self._lock:
            return self._recent_unlocked(self._load_unlocked(), limit)

    def forget(self, memory_id: str) -> int:
        if not isinstance(memory_id, str):
            raise TypeError("memory_id must be a string.")
        prefix = memory_id.strip()
        if not prefix:
            raise ValueError("Memory id cannot be empty.")
        with self._lock:
            records = self._load_unlocked()
            matches = [item for item in records if item["id"] == prefix]
            if not matches:
                matches = [item for item in records if item["id"].startswith(prefix)]
            if not matches:
                return 0
            if len(matches) > 1:
                raise ValueError(
                    f"Memory id prefix {prefix!r} is ambiguous; {len(matches)} records match."
                )
            target = matches[0]["id"]
            self._write_unlocked([item for item in records if item["id"] != target])
            return 1

    def clear(self) -> int:
        with self._lock:
            records = self._load_unlocked()
            self._write_unlocked([])
            return len(records)


def _approve(
    ctx: ToolContext, enabled: bool, action: str, summary: str, detail: str
) -> ToolResult | None:
    if ctx.read_only:
        return ToolResult.failure("Read-only mode: memory changes are disabled.", denied=True)
    if not enabled:
        return None
    if threading.current_thread() is not threading.main_thread():
        return ToolResult.failure(
            "Memory changes requiring approval must run on the main thread.", denied=True
        )
    request = ApprovalRequest(
        action=action,
        summary=_clean(summary, single_line=True)[:500],
        detail=_clean(detail, single_line=True)[:2_000],
        risk="low",
    )
    if not ctx.request_approval(request):
        return ToolResult.failure(_DENIED, denied=True)
    return None


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    handler: Any,
    *,
    read_only: bool = False,
) -> Tool:
    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        handler=handler,
        read_only=read_only,
        destructive=not read_only,
    )


def make_memory_tools(store: MemoryStore, *, require_approval: bool = True) -> list[Tool]:
    """Build remember/recall/forget tools; clear stays owner-only."""

    def remember_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        text, tags = args.get("text"), args.get("tags")
        if not isinstance(text, str):
            return ToolResult.failure("Argument 'text' must be a string.")
        if tags is not None and not isinstance(tags, (str, list)):
            return ToolResult.failure("Argument 'tags' must be a string.")
        denied = _approve(
            ctx, require_approval, "remember",
            _clean(text, single_line=True)[:160], _tag_detail(tags),
        )
        if denied is not None:
            return denied
        try:
            record = store.remember(text, tags)  # type: ignore[arg-type]
        except (TypeError, ValueError, OSError) as exc:
            return ToolResult.failure(f"remember: {exc}")
        short = record["text"].replace("\n", " ")[:160]
        return ToolResult.success(f"Remembered {record['id']}: {short}", id=record["id"])

    def recall_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        _ = ctx
        query = args.get("query", "")
        query = "" if query is None else query
        if not isinstance(query, str):
            return ToolResult.failure("Argument 'query' must be a string.")
        try:
            records = store.recall(query, _clamp(args.get("limit"), 5, MAX_RECALL))
        except (TypeError, ValueError, OSError) as exc:
            return ToolResult.failure(f"recall: {exc}")
        if not records:
            return ToolResult.success("No memories.")
        lines: list[str] = []
        for number, record in enumerate(records, 1):
            tags = ", ".join(record["tags"]) or "(none)"
            lines.append(
                f"{number}. {record['id']} | {record['created_at']} | tags: {tags}"
            )
            lines.append(f"   {record['text']}")
        return ToolResult.success("\n".join(lines), count=len(records))

    def forget_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        memory_id = args.get("id")
        if not isinstance(memory_id, str) or not memory_id.strip():
            return ToolResult.failure("Argument 'id' must be a non-empty string.")
        denied = _approve(
            ctx, require_approval, "forget",
            f"Forget memory {memory_id.strip()[:80]}", "",
        )
        if denied is not None:
            return denied
        try:
            count = store.forget(memory_id)
        except (ValueError, OSError) as exc:
            return ToolResult.failure(f"forget: {exc}")
        noun = "memory" if count == 1 else "memories"
        return ToolResult.success(f"Forgot {count} {noun}.", count=count)

    return [
        _tool(
            "remember",
            "Save an explicit persistent memory for this workspace. Requires "
            "approval; keep text concise and avoid secrets.",
            {
                "text": {"type": "string", "description": "Memory text (max 4000 characters)."},
                "tags": {"type": "string", "description": "Optional comma-separated tags (max 12)."},
            },
            ["text"],
            remember_tool,
        ),
        _tool(
            "recall",
            "Search persistent workspace memories. An empty query returns the "
            "most recent records.",
            {
                "query": {"type": "string", "description": "Words or tags to search for."},
                "limit": {
                    "type": "integer", "description": "Maximum records (1-20, default 5).",
                    "minimum": 1, "maximum": 20,
                },
            },
            [],
            recall_tool,
            read_only=True,
        ),
        _tool(
            "forget",
            "Delete one persistent memory by full id or unique id prefix. Requires approval.",
            {"id": {"type": "string", "description": "Memory id (m-xxxxxxxx) or unique prefix."}},
            ["id"],
            forget_tool,
        ),
    ]


def _tag_detail(tags: object) -> str:
    if isinstance(tags, str):
        return tags
    if isinstance(tags, list):
        return ", ".join(str(tag) for tag in tags)
    return ""
