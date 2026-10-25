"""Durable session-scoped sidecar records for the OHAR wrapper.

Wrapper state cannot live in the chat transcript: ``ohar.sessions`` validates
``schema_version`` strictly and ``Session.save`` only persists meta plus
transcript, so every wrapper record is its own file under
``ohar_home()/wrapper/<session-id>/``.

The module provides the primitives the rest of the wrapper builds on: atomic
0600 writes inside 0700 directories, optimistic concurrency on
``record_revision``, idempotent replay keyed on ``(operation, mutation_id)``
whose digest mismatch is refused rather than silently replayed, a bounded
append-only JSONL event log, and ANSI/control-stripped one-line text fields.

A corrupt record reads back as ``{}`` and the next guarded write replaces it,
so one torn file never wedges a session; the cost is that a corrupt file also
resets the revision floor.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ohar.config import ensure_home, ohar_home
from ohar.terminal import clip, sanitize_text

SCHEMA_VERSION = 1
MAX_EVENTS = 500
MAX_EVENT_CHARS = 400
MAX_APPLIED_MUTATIONS = 64

EVENTS_FILE = "events.jsonl"

__all__ = [
    "EVENTS_FILE",
    "MAX_APPLIED_MUTATIONS",
    "MAX_EVENT_CHARS",
    "MAX_EVENTS",
    "SCHEMA_VERSION",
    "RecordError",
    "UpdateResult",
    "append_event",
    "clean_text",
    "digest_of",
    "guarded_update",
    "read_events",
    "read_record",
    "revision_of",
    "session_dir",
    "validate_record",
    "wrapper_root",
    "write_record",
]

_REVISION_FIELD = "record_revision"

_ID_RE = re.compile(r"[A-Za-z0-9._-]+")
_MAX_EVENT_KEYS = 24
_MAX_EVENT_DEPTH = 3
_MAX_EVENT_LIST = 12
_MAX_EVENT_LINE = 4096
_MAX_ERROR_LINES = 8
_MAX_PARSE_ERROR_CHARS = 120


class RecordError(ValueError):
    """A wrapper record or event could not be read, validated, or written."""


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """Outcome of one :func:`guarded_update` transaction."""

    record: dict[str, Any]
    revision: int
    replayed: bool


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _valid_utc(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    offset = parsed.utcoffset()
    return parsed.tzinfo is not None and offset == timedelta(0)


def _check_id(value: object, label: str) -> str:
    """Validate a name that becomes a path component under the wrapper root."""
    if not isinstance(value, str) or not value:
        raise RecordError(f"{label} must be a non-empty string.")
    if value in {".", ".."} or ".." in value or _ID_RE.fullmatch(value) is None:
        raise RecordError(
            f"Invalid {label} {value!r}; use only letters, digits, '.', '_' and '-' "
            "and do not use '..'."
        )
    return value


def _record_file(name: str) -> str:
    checked = _check_id(name, "Record name")
    if checked.startswith("."):
        raise RecordError("Record name must not start with '.'.")
    return checked if checked.endswith(".json") else f"{checked}.json"


def _harden_dir(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except OSError:
        # mkdir(mode=0o700) already asked for private permissions; filesystems
        # without chmod must not make a readable record fail.
        pass


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _harden_dir(path)


def _fsync_dir(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    _ensure_private_dir(path.parent)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    fd: int | None = None
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        handle = os.fdopen(fd, "wb")
        fd = None
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def wrapper_root(*, create: bool = True) -> Path:
    """Return ``ohar_home()/wrapper``, created 0700 when ``create`` is set."""
    root = ohar_home() / "wrapper"
    if create:
        ensure_home()
        _ensure_private_dir(root)
    return root


def session_dir(session_id: str, *, create: bool = True) -> Path:
    """Return one session's sidecar directory, re-validating the session id."""
    checked = _check_id(session_id, "Session id")
    directory = wrapper_root(create=create) / checked
    if create:
        _ensure_private_dir(directory)
    return directory


def _record_path(name: str, session_id: str, *, create: bool) -> Path:
    return session_dir(session_id, create=create) / _record_file(name)


def _serialize(record: Mapping[str, Any]) -> bytes:
    payload: dict[str, Any] = {}
    for key, value in record.items():
        if not isinstance(key, str):
            raise RecordError(f"Record keys must be strings, got {key!r}.")
        payload[key] = value
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RecordError(f"Record is not JSON-serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def read_record(name: str, session_id: str) -> dict[str, Any]:
    """Read one sidecar record; missing or corrupt content returns ``{}``."""
    path = _record_path(name, session_id, create=False)
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_record(name: str, session_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically replace one record with ``record``, binding mode 0600."""
    path = _record_path(name, session_id, create=True)
    payload = dict(record)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("session_id", path.parent.name)
    data = _serialize(payload)
    _atomic_write(path, data)
    return payload


def revision_of(record: Mapping[str, Any]) -> int:
    """Read ``record_revision`` fail-closed: anything unusable is revision 0."""
    if not isinstance(record, Mapping):
        return 0
    value = record.get(_REVISION_FIELD, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def digest_of(*parts: object) -> str:
    """Digest one operation's own arguments, for replay-conflict detection."""
    try:
        payload = json.dumps(
            list(parts),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        payload = json.dumps([repr(part) for part in parts], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _applied_entries(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("applied_mutations") if isinstance(record, Mapping) else None
    if not isinstance(raw, list):
        return []
    return [dict(entry) for entry in raw if isinstance(entry, Mapping)]


def _find_applied(record: Mapping[str, Any], operation: str, mutation_id: str) -> dict[str, Any] | None:
    for entry in _applied_entries(record):
        if entry.get("operation") == operation and entry.get("mutation_id") == mutation_id:
            return entry
    return None


def validate_record(record: Mapping[str, Any]) -> list[str]:
    """Return structural problems with a wrapper record; empty means valid."""
    if not isinstance(record, Mapping):
        return ["Wrapper record must be a JSON object."]
    errors: list[str] = []

    version = record.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        errors.append(f"record schema_version must be {SCHEMA_VERSION}, got {version!r}.")

    if "session_id" in record:
        value = record["session_id"]
        if (
            not isinstance(value, str)
            or value in {".", ".."}
            or ".." in value
            or _ID_RE.fullmatch(value) is None
        ):
            errors.append(f"record session_id is invalid: {value!r}.")

    if _REVISION_FIELD in record and revision_of(record) != record[_REVISION_FIELD]:
        errors.append(f"record {_REVISION_FIELD} must be a non-negative integer.")

    if "updated_at" in record and not _valid_utc(record["updated_at"]):
        errors.append(f"record updated_at must be a UTC ISO 8601 timestamp, got {record['updated_at']!r}.")

    entries = record.get("applied_mutations")
    if entries is not None:
        if not isinstance(entries, list):
            errors.append("record applied_mutations must be a list.")
        elif len(entries) > MAX_APPLIED_MUTATIONS:
            errors.append(f"record applied_mutations exceeds {MAX_APPLIED_MUTATIONS} entries.")
        else:
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    errors.append(f"record applied_mutations[{index}] must be an object.")
                    continue
                for key in ("operation", "mutation_id", "digest"):
                    if not isinstance(entry.get(key), str):
                        errors.append(f"record applied_mutations[{index}].{key} must be a string.")

    for key in record:
        if not isinstance(key, str):
            errors.append(f"record keys must be strings, got {key!r}.")

    try:
        json.dumps(dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        errors.append(f"record is not JSON-serializable: {exc}")
    return errors


def guarded_update(
    name: str,
    session_id: str,
    mutate: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    operation: str,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
    mutation_digest: str | None = None,
    validate: Callable[[dict[str, Any]], None] | None = None,
) -> UpdateResult:
    """Run one read-check-write transaction on a record.

    Replay is keyed on ``(operation, mutation_id)``, so the same client id
    reused for a different logical operation still applies. A replayed id whose
    digest differs from the stored one is refused: silently returning another
    caller's result would launder a lost mutation.
    """
    if not callable(mutate):
        raise RecordError("guarded_update requires a callable mutate.")
    checked_operation = clean_text(operation, single_line=True)
    if not checked_operation:
        raise RecordError("guarded_update requires a non-empty operation.")
    if expected_revision is not None and (
        isinstance(expected_revision, bool) or not isinstance(expected_revision, int)
    ):
        raise RecordError("expected_revision must be an integer or None.")

    current = read_record(name, session_id)
    revision = revision_of(current)
    digest = (
        mutation_digest
        if mutation_digest is not None
        else digest_of(checked_operation, mutation_id)
    )

    if mutation_id is not None:
        entry = _find_applied(current, checked_operation, mutation_id)
        if entry is not None:
            stored = entry.get("digest")
            if isinstance(stored, str) and stored and stored != digest:
                raise RecordError(
                    f"Mutation {mutation_id!r} of {checked_operation!r} was already applied "
                    "with a different digest; refusing to replay."
                )
            return UpdateResult(record=current, revision=revision, replayed=True)

    if expected_revision is not None and expected_revision != revision:
        raise RecordError(f"Stale record revision: expected {expected_revision}, found {revision}.")

    candidate = mutate(copy.deepcopy(current))
    if not isinstance(candidate, dict):
        raise RecordError(
            f"{checked_operation} must return a dict, got {type(candidate).__name__}."
        )

    candidate[_REVISION_FIELD] = revision + 1
    candidate["schema_version"] = SCHEMA_VERSION
    candidate["updated_at"] = _now_iso()
    if mutation_id is not None:
        if "applied_mutations" not in candidate:
            candidate["applied_mutations"] = _applied_entries(current)
        entries = _applied_entries(candidate)
        entries.append(
            {
                "operation": checked_operation,
                "mutation_id": clean_text(mutation_id, limit=MAX_EVENT_CHARS),
                "digest": digest,
            }
        )
        candidate["applied_mutations"] = entries[-MAX_APPLIED_MUTATIONS:]

    errors = validate_record(candidate)
    if errors:
        raise RecordError(f"Invalid {name} record: {errors[0]}")
    if validate is not None:
        validate(candidate)

    written = write_record(name, session_id, candidate)
    return UpdateResult(record=written, revision=revision + 1, replayed=False)


def clean_text(value: object, *, limit: int = MAX_EVENT_CHARS, single_line: bool = True) -> str:
    """Strip ANSI and control characters and bound one untrusted text field."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    if single_line:
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    cleaned = sanitize_text(text, keep_newlines=not single_line)
    if single_line:
        cleaned = " ".join(cleaned.split())
    if limit <= 0:
        return ""
    return clip(cleaned, limit)


def _clean_value(value: object, depth: int) -> Any:
    if isinstance(value, Mapping):
        return _clean_data(value, depth + 1)
    if isinstance(value, (list, tuple)):
        return [_clean_value(item, depth + 1) for item in list(value)[:_MAX_EVENT_LIST]]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) <= 10**15 else clean_text(value, limit=64)
    if isinstance(value, float):
        return value if math.isfinite(value) else clean_text(value, limit=64)
    return clean_text(value)


def _clean_data(data: Mapping[str, Any] | None, depth: int = 0) -> dict[str, Any]:
    if not isinstance(data, Mapping) or depth >= _MAX_EVENT_DEPTH:
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in list(data.items())[:_MAX_EVENT_KEYS]:
        name = clean_text(key, limit=64)
        if not name:
            continue
        cleaned[name] = _clean_value(value, depth)
    return cleaned


def _read_lines(path: Path) -> list[str]:
    try:
        with path.open("rb") as handle:
            raw = handle.read()
    except OSError:
        return []
    lines: list[str] = []
    for chunk in raw.split(b"\n"):
        if not chunk:
            lines.append("")
            continue
        try:
            lines.append(chunk.decode("utf-8"))
        except UnicodeDecodeError:
            lines.append("\ufffd")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _append_line(path: Path, line: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _next_seq(lines: list[str]) -> int:
    """Highest recorded sequence plus one; rotation must not restart the ids."""
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            value = parsed.get("seq")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value + 1
    return len(lines) + 1


def append_event(session_id: str, event: str, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Append one sanitized JSONL event, rotating the log to ``MAX_EVENTS``."""
    name = clean_text(event)
    if not name:
        raise RecordError("Event name must not be empty.")
    directory = session_dir(session_id)
    path = directory / EVENTS_FILE
    existing = _read_lines(path)

    item: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seq": _next_seq(existing),
        "at": _now_iso(),
        "event": name,
        "data": _clean_data(data),
    }
    text = json.dumps(item, ensure_ascii=False, sort_keys=True)
    if len(text) > _MAX_EVENT_LINE:
        # Keep the event rather than lose it: the detail collapses to one
        # bounded line so an oversized payload cannot grow the log unbounded.
        item["data"] = {"truncated": True, "detail": clean_text(json.dumps(item["data"], default=str))}
        text = json.dumps(item, ensure_ascii=False, sort_keys=True)

    if len(existing) + 1 > MAX_EVENTS:
        kept = (existing + [text])[-MAX_EVENTS:]
        _atomic_write(path, ("\n".join(kept) + "\n").encode("utf-8"))
    else:
        _append_line(path, text + "\n")
    return item


def read_events(session_id: str, *, limit: int = 50) -> tuple[list[dict[str, Any]], list[str]]:
    """Return ``(recent events oldest-first, parse errors)`` for a session."""
    bounded = limit if isinstance(limit, int) and not isinstance(limit, bool) else 50
    bounded = max(1, min(MAX_EVENTS, bounded))
    path = session_dir(session_id, create=False) / EVENTS_FILE
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    for number, line in enumerate(_read_lines(path), start=1):
        if not line.strip():
            continue
        item: Any = None
        problem: str | None = None
        try:
            item = json.loads(line)
        except ValueError as exc:
            problem = f"{path}:{number}: {clean_text(str(exc), limit=_MAX_PARSE_ERROR_CHARS)}"
        if problem is None and not isinstance(item, dict):
            problem = f"{path}:{number}: event line is not a JSON object"
        if problem is not None:
            if len(errors) < _MAX_ERROR_LINES:
                errors.append(problem)
            continue
        items.append(item)
    return items[-bounded:], errors
