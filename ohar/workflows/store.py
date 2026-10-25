"""Durable record stores shared by every workflows module. Stdlib only.

One private directory per workspace under ``ohar_home()/workflows`` holds
append-only JSONL stores and atomic JSON head records. The store is append-only:
a new record never edits an old one, so an answer supersedes a question by
arriving later rather than by rewriting what was asked.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import ensure_home
from ..terminal import sanitize_text

__all__ = [
    "WORKFLOWS_DIRNAME",
    "MAX_LINE_BYTES",
    "MAX_STORE_BYTES",
    "MAX_TAIL_RECORDS",
    "MAX_JSON_DEPTH",
    "MAX_JSON_NODES",
    "WorkflowError",
    "StaleRevisionError",
    "now_iso",
    "stamp_to_epoch",
    "canonical_digest",
    "short_digest",
    "clean_text",
    "opaque_ref",
    "new_record_id",
    "workspace_key",
    "workflows_root",
    "read_json_bounded",
    "write_json_atomic",
    "RecordStore",
    "validate_vocabularies",
]

WORKFLOWS_DIRNAME = "workflows"
MAX_LINE_BYTES = 65_536
MAX_STORE_BYTES = 8_388_608
MAX_TAIL_RECORDS = 2_000
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 20_000
MAX_STAMP_CHARS = 64

SCHEMA_VERSION = 1

_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_SAFE_CHAR_RE = re.compile(r"[^a-z0-9_-]+")
_WHITESPACE_RE = re.compile(r"\s+")
_REF_UNSAFE_RE = re.compile(r"[\"'`$\\<>|;&=\x00-\x1f\x7f]")
_SENSITIVE_RE = re.compile(
    r"(?i)(token|secret|passw|api[_-]?key|access[_-]?key|authorization|bearer|private[_-]?key|credential)\s*[=:]"
)
_TRUNCATED = "…"

_fcntl = None
_msvcrt = None
try:  # POSIX advisory locking: the real cross-process gate.
    import fcntl as _fcntl  # type: ignore[no-redef]
except ImportError:  # pragma: no cover - platform dependent
    try:
        import msvcrt as _msvcrt  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - exotic platforms
        pass

_thread_locks: dict[str, threading.RLock] = {}
_thread_locks_guard = threading.Lock()


class WorkflowError(ValueError):
    """A workflow store operation was refused, or its data cannot be trusted."""


class StaleRevisionError(WorkflowError):
    """The record moved since the caller read it; nothing was written."""


def workflows_root(workspace: Path | None = None) -> Path:
    """The private workflows directory, scoped to one workspace when given."""
    base = ensure_home() / WORKFLOWS_DIRNAME
    _prepare_directory(base)
    if workspace is None:
        return base
    scoped = base / workspace_key(workspace)
    _prepare_directory(scoped)
    return scoped


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def stamp_to_epoch(value: str) -> float | None:
    text = str(value or "").strip()
    if not text or len(text) > MAX_STAMP_CHARS:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        return str(value)
    except Exception:  # pragma: no cover - hostile __str__
        return ""


def clean_text(value: object, *, limit: int = 240, single_line: bool = True) -> str:
    """ANSI- and control-free text, collapsed and byte-aware bounded."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise WorkflowError("limit must be an integer.")
    limit = max(1, min(limit, MAX_LINE_BYTES))
    text = _as_text(value)
    text = sanitize_text(text, keep_newlines=not single_line)
    text = _WHITESPACE_RE.sub(" ", text).strip() if single_line else text.strip()
    if len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + _TRUNCATED
    return text.strip()


def opaque_ref(value: object, *, limit: int = 128) -> str:
    """A bounded, non-navigable reference: handles pass through, anything
    path-like or secret-shaped folds into an irreversible digest."""
    text = clean_text(value, limit=limit)
    if not text:
        return ""
    if _REF_UNSAFE_RE.search(text) or _SENSITIVE_RE.search(text) or ".." in text or text[0] in "/~.":
        return f"ref:{short_digest(text, 16)}"
    return text


def new_record_id(prefix: str) -> str:
    slug = _SAFE_CHAR_RE.sub("-", _as_text(prefix).casefold()).strip("-")[:32] or "rec"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{slug}-{stamp}-{secrets.token_hex(3)}"


def workspace_key(workspace: Path) -> str:
    resolved = Path(workspace).expanduser().resolve(strict=False)
    return hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]


def canonical_digest(value: object) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=_as_text, ensure_ascii=True
        )
    except (TypeError, ValueError):
        payload = _as_text(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def short_digest(value: object, length: int = 12) -> str:
    if isinstance(length, bool) or not isinstance(length, int):
        raise WorkflowError("length must be an integer.")
    return canonical_digest(value)[: max(4, min(length, 64))]


def _check_bounds(value: object, *, max_depth: int, max_nodes: int) -> bool:
    """True when the decoded document stays inside its depth and node budget."""
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            return False
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return True


def _decode_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except RecursionError as exc:
        raise WorkflowError("json_depth_exceeded") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkflowError("malformed_json") from exc
    if not isinstance(value, dict):
        raise WorkflowError("malformed_json: expected an object")
    if not _check_bounds(value, max_depth=MAX_JSON_DEPTH, max_nodes=MAX_JSON_NODES):
        raise WorkflowError("json_bounds_exceeded")
    return value


def _brief(path: Path) -> str:
    return clean_text(path.name, limit=80)


def read_json_bounded(path: Path, *, max_bytes: int = MAX_STORE_BYTES) -> dict[str, Any]:
    """Read one JSON object with a hard byte, depth and node budget."""
    target = Path(path)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise WorkflowError("max_bytes must be a positive integer.")
    budget = min(max_bytes, MAX_STORE_BYTES)
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise WorkflowError(f"store file is unreadable: {_brief(target)}") from exc
    if size > budget:
        raise WorkflowError(f"store file '{_brief(target)}' exceeds its {budget}-byte read bound")
    try:
        with open(target, "rb") as handle:
            raw = handle.read(budget + 1)
    except OSError as exc:
        raise WorkflowError(f"store file is unreadable: {_brief(target)}") from exc
    if len(raw) > budget:
        raise WorkflowError(f"store file '{_brief(target)}' exceeds its {budget}-byte read bound")
    return _decode_object(raw)


def _prepare_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass


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


def _thread_lock(key: str) -> threading.RLock:
    with _thread_locks_guard:
        lock = _thread_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _thread_locks[key] = lock
        return lock


def _acquire(fd: int) -> None:
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
        return
    if _msvcrt is not None:  # pragma: no cover - Windows only
        os.lseek(fd, 0, os.SEEK_SET)
        _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
        return
    raise WorkflowError("no OS file locking is available; refusing to write")  # pragma: no cover


def _release(fd: int) -> None:
    try:
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        elif _msvcrt is not None:  # pragma: no cover - Windows only
            os.lseek(fd, 0, os.SEEK_SET)
            _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """One OS lock around read+write so two surfaces cannot both believe they
    hold the only open gate. The data file itself may be replaced atomically,
    so the lock lives in a stable sidecar."""
    lock_path = path.with_name(path.name + ".lock")
    _prepare_directory(lock_path.parent)
    try:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    except OSError as exc:
        raise WorkflowError(f"cannot open the store lock for '{_brief(path)}'") from exc
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        with _thread_lock(str(lock_path)):
            try:
                _acquire(fd)
            except OSError as exc:
                raise WorkflowError(f"cannot lock the store '{_brief(path)}'") from exc
            try:
                yield
            finally:
                _release(fd)
    finally:
        os.close(fd)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one JSON head record atomically, mode 0600, under the store lock."""
    if not isinstance(payload, Mapping):
        raise WorkflowError("write_json_atomic expects a mapping.")
    target = Path(path)
    try:
        data = (json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkflowError("record is not JSON serializable") from exc
    if len(data) > MAX_STORE_BYTES:
        raise WorkflowError("record exceeds the store byte bound")
    _prepare_directory(target.parent)
    with _locked(target):
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
            _fsync_dir(target.parent)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise


def _safe_name(value: object, fallback: str = "records") -> str:
    name = _SAFE_CHAR_RE.sub("_", _as_text(value).casefold()).strip("_")[:64]
    if not name or not _NAME_RE.fullmatch(name):
        return fallback
    return name


def _tail_bytes(path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError:
        return b""
    try:
        with open(path, "rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise WorkflowError(f"store file is unreadable: {_brief(path)}") from exc
    if size > max_bytes:
        cut = raw.find(b"\n")
        raw = b"" if cut < 0 else raw[cut + 1 :]
    return raw


def _decode_lines(raw: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line or len(line) > MAX_LINE_BYTES:
            continue
        try:
            item = json.loads(line.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if not isinstance(item, dict):
            continue
        if not _check_bounds(item, max_depth=MAX_JSON_DEPTH, max_nodes=MAX_JSON_NODES):
            continue
        records.append(item)
    return records


def _revision_of(record: dict[str, Any] | None) -> int:
    """A hand-edited revision reads as 0 rather than crashing a transaction."""
    if record is None:
        return 0
    value = record.get("revision", 0)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class RecordStore:
    """Append-only JSONL store, one file per name and workspace."""

    root: Path
    name: str
    workspace: Path | None = None

    def path(self) -> Path:
        base = Path(self.root).expanduser()
        if self.workspace is not None:
            base = base / workspace_key(self.workspace)
        return base / f"{_safe_name(self.name)}.jsonl"

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        with _locked(self.path()):
            return self._append_locked(record)

    def records(self, *, limit: int = MAX_TAIL_RECORDS) -> list[dict[str, Any]]:
        """The most recent records in append order; unreadable lines are skipped."""
        count = MAX_TAIL_RECORDS if isinstance(limit, bool) or not isinstance(limit, int) else limit
        if count < 1:
            return []
        count = min(count, MAX_TAIL_RECORDS)
        tail = _decode_lines(_tail_bytes(self.path(), MAX_STORE_BYTES))
        return tail[-count:]

    def heads(self, key: str) -> dict[str, dict[str, Any]]:
        """The one record that speaks for each identity value: last write wins."""
        field = _safe_name(key, fallback="")
        heads: dict[str, dict[str, Any]] = {}
        for record in self.records():
            value = record.get(field)
            if isinstance(value, str) and value:
                heads[value] = record
        return heads

    def latest(self, key: str, value: str) -> dict[str, Any] | None:
        found = self.heads(key).get(str(value))
        return dict(found) if found is not None else None

    def transact(
        self,
        key: str,
        value: str,
        mutate: Callable[[dict[str, Any] | None], dict[str, Any] | None],
        *,
        expected_revision: int | None = None,
        mutation_id: str = "",
        mutation_digest: str = "",
    ) -> tuple[dict[str, Any], bool]:
        """Locked read-check-write on one identity. Returns (record, replayed).

        A replayed ``mutation_id`` returns the record already on file with no
        write and no revision bump; the same client id carrying different
        content is refused instead of silently handing back someone else's
        result.
        """
        field = _safe_name(key, fallback="")
        if not field:
            raise WorkflowError("transact requires an identity key.")
        identity = clean_text(value, limit=128, single_line=True)
        if not identity:
            raise WorkflowError("transact requires a non-empty identity value.")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int | type(None)):
            raise WorkflowError("expected_revision must be an integer or None.")
        if expected_revision is not None and expected_revision < 0:
            raise WorkflowError("expected_revision cannot be negative.")
        with _locked(self.path()):
            current = self.latest(field, identity)
            revision = _revision_of(current)
            mutation_id = clean_text(mutation_id, limit=128)
            if mutation_id and current is not None and str(current.get("mutation_id", "")) == mutation_id:
                stored = str(current.get("mutation_digest", ""))
                if mutation_digest and stored and stored != mutation_digest:
                    raise WorkflowError("mutation replay conflicts with the recorded mutation")
                return current, True
            if expected_revision is not None and expected_revision != revision:
                raise StaleRevisionError(f"expected revision {expected_revision}, found {revision}")
            updated = mutate(dict(current) if current is not None else None)
            if updated is None:
                return (current or {}), False
            if not isinstance(updated, Mapping):
                raise WorkflowError("a transaction must return a mapping or None")
            record = dict(updated)
            if field in record and str(record[field]) != identity:
                raise WorkflowError("a transaction cannot change its own identity")
            record[field] = identity
            record["revision"] = revision + 1
            if mutation_id:
                record["mutation_id"] = mutation_id
                record["mutation_digest"] = clean_text(mutation_digest, limit=128)
            return self._append_locked(record), False

    def _append_locked(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise WorkflowError("a store record must be a mapping")
        target = self.path()
        stored = dict(record)
        stored.setdefault("schema_version", SCHEMA_VERSION)
        stored.setdefault("record_id", new_record_id(_safe_name(self.name)))
        stored.setdefault("at", now_iso())
        try:
            line = json.dumps(stored, ensure_ascii=False, separators=(",", ":"), default=_as_text)
        except (TypeError, ValueError) as exc:
            raise WorkflowError("record is not JSON serializable") from exc
        if "\n" in line or "\r" in line:
            raise WorkflowError("record lines cannot contain newlines")
        data = (line + "\n").encode("utf-8")
        if len(data) > MAX_LINE_BYTES:
            raise WorkflowError(f"record exceeds the {MAX_LINE_BYTES}-byte line bound")
        try:
            size = target.stat().st_size if target.exists() else 0
        except OSError:
            size = 0
        if size + len(data) > MAX_STORE_BYTES:
            raise WorkflowError(f"store '{_brief(target)}' exceeded its {MAX_STORE_BYTES}-byte bound")
        _prepare_directory(target.parent)
        descriptor = os.open(str(target), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, data)
            os.fsync(descriptor)
            try:
                os.fchmod(descriptor, 0o600)
            except OSError:
                pass
        except OSError as exc:
            raise WorkflowError(f"cannot append to '{_brief(target)}'") from exc
        finally:
            os.close(descriptor)
        return stored


def validate_vocabularies(
    record: Mapping[str, Any], vocabularies: Sequence[tuple[str, Sequence[str]]]
) -> list[str]:
    """Closed-set validation: a field outside its vocabulary is an error."""
    errors: list[str] = []
    for field, allowed in vocabularies:
        options = [str(item) for item in allowed]
        if field not in record:
            errors.append(f"{field} is required; expected one of: {', '.join(options)}")
            continue
        value = record[field]
        if isinstance(value, (list, tuple)):
            for item in value:
                if not isinstance(item, str) or item not in options:
                    errors.append(f"{field} contains an unsupported value: {clean_text(item, limit=40)!r}")
            continue
        if not isinstance(value, str) or value not in options:
            shown = clean_text(value, limit=40)
            errors.append(f"{field} is unsupported: {shown!r}; expected one of: {', '.join(options)}")
    return errors
