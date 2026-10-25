"""Persistent local chat sessions for OHAR. Stdlib only.

Sessions live under ``$OHAR_HOME/sessions`` (or ``~/.ohar/sessions``), one
directory per session.  Each directory contains ``meta.json`` and a
``transcript.jsonl`` file whose non-blank lines are canonical JSON message
objects.  An ``index.json`` file at the sessions root mirrors metadata for
cheap listing; transcripts are only read by :meth:`SessionStore.load`.

Nothing is created or scanned at import time, and the store creates its
directories lazily with private permissions.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_TITLE = "New session"
MAX_TITLE_LENGTH = 64

_MAX_LIST_LIMIT = 200
_MAX_CREATE_ATTEMPTS = 32
_INDEX_FILE = "index.json"
_META_FILE = "meta.json"
_TRANSCRIPT_FILE = "transcript.jsonl"
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9._-]+")

__all__ = [
    "DEFAULT_TITLE",
    "MAX_TITLE_LENGTH",
    "SCHEMA_VERSION",
    "Session",
    "SessionMeta",
    "SessionStore",
]


def _default_home() -> Path:
    configured = os.environ.get("OHAR_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".ohar"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_session_id(session_id: object) -> str:
    """Return a safe session id or raise ValueError.

    Session ids become directory names, so this rejects separators, parent
    traversal, and any other character outside a conservative allow list.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Session id must be a non-empty string.")
    if session_id in {".", ".."} or ".." in session_id or _SESSION_ID_RE.fullmatch(session_id) is None:
        raise ValueError(
            f"Invalid session id {session_id!r}; use only letters, digits, '.', '_' and '-' "
            "and do not use '..'."
        )
    return session_id


def _validate_timestamp(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty ISO 8601 timestamp string.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid ISO 8601 timestamp: {value!r}") from exc
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None or offset != timedelta(0):
        raise ValueError(f"{field_name} must be a timezone-aware UTC timestamp: {value!r}")
    return value


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _optional_meta_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Session metadata {key!r} must be a string.")
    return value


def _nonnegative_meta_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key, 0)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Session metadata {key!r} must be a non-negative integer.")
    return value


@dataclass(slots=True)
class SessionMeta:
    """Metadata shown in session lists and stored in ``meta.json``."""

    id: str
    title: str = DEFAULT_TITLE
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0
    cwd: str = ""
    model: str = ""
    lane: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "cwd": self.cwd,
            "model": self.model,
            "lane": self.lane,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionMeta":
        if not isinstance(data, Mapping):
            raise ValueError("Session metadata must be a JSON object.")
        version = data.get("schema_version", SCHEMA_VERSION)
        if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported session schema_version {version!r}; expected {SCHEMA_VERSION}.")

        try:
            session_id = _validate_session_id(data.get("id"))
        except ValueError as exc:
            raise ValueError(f"Invalid session metadata: {exc}") from exc

        title = data.get("title", DEFAULT_TITLE)
        if title is None:
            title = DEFAULT_TITLE
        if not isinstance(title, str):
            raise ValueError("Session metadata 'title' must be a string.")

        return cls(
            id=session_id,
            title=title,
            created_at=_validate_timestamp(data.get("created_at"), "created_at"),
            updated_at=_validate_timestamp(data.get("updated_at"), "updated_at"),
            message_count=_nonnegative_meta_int(data, "message_count"),
            cwd=_optional_meta_string(data, "cwd"),
            model=_optional_meta_string(data, "model"),
            lane=_optional_meta_string(data, "lane"),
            input_tokens=_nonnegative_meta_int(data, "input_tokens"),
            output_tokens=_nonnegative_meta_int(data, "output_tokens"),
        )


def _content_text(content: Any) -> str:
    """Extract readable text from a provider message's content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return " ".join(chunks)
    return ""


def _title_from_content(content: Any) -> str:
    text = " ".join(_content_text(content).split())
    if not text:
        return ""
    return text[:MAX_TITLE_LENGTH].rstrip()


@dataclass(slots=True)
class Session:
    """A chat session: metadata plus an ordered list of message dicts."""

    meta: SessionMeta
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add(self, msg: dict[str, Any]) -> None:
        if not isinstance(msg, dict):
            raise TypeError("Session messages must be dicts.")
        self.messages.append(msg)

    def title_from_first_user(self) -> str:
        """Derive a one-line title from the first user message.

        When there is no usable user text (or no user message), the current
        title is kept, or the default title is restored if it was blank.
        :meth:`SessionStore.save` only calls this while the title is still the
        default, so an explicit title is preserved by normal saves.
        """
        if not isinstance(self.meta.title, str) or not self.meta.title:
            self.meta.title = DEFAULT_TITLE
        for message in self.messages:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            title = _title_from_content(message.get("content"))
            if title:
                self.meta.title = title
            break
        return self.meta.title


def _serialize_meta(meta: SessionMeta) -> bytes:
    try:
        text = json.dumps(
            meta.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        return (text + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Session metadata is not JSON-serializable: {exc}. "
            "Metadata fields must be strings and non-negative integers."
        ) from exc


def _serialize_transcript(messages: Sequence[Any]) -> bytes:
    lines: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(
                f"Session message {index} must be a dict, got {type(message).__name__}."
            )
        try:
            encoded = json.dumps(
                dict(message),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            role = message.get("role", "unknown")
            raise ValueError(
                f"Session message {index} (role={role!r}) is not JSON-serializable: {exc}. "
                "Sessions store plain JSON messages; convert provider-specific objects "
                "to JSON-compatible values."
            ) from exc
        lines.append(encoded)
    if not lines:
        return b""
    try:
        return ("\n".join(lines) + "\n").encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"Session messages contain invalid Unicode text: {exc}") from exc


def _serialize_index(metas: Sequence[SessionMeta]) -> bytes:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sessions": [meta.to_dict() for meta in metas],
    }
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        return (text + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Session index is not JSON-serializable: {exc}") from exc


def _harden_dir(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except OSError:
        # mkdir(mode=0o700) already requested private permissions.  Some
        # filesystems (notably Windows-backed ones) do not implement chmod;
        # failing to read a session is worse than a best-effort mode fix.
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
    """Atomically replace *path* with *data* using a private temp file."""
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


def _validate_meta_for_write(meta: SessionMeta) -> None:
    _validate_session_id(meta.id)
    if not isinstance(meta.title, str) or not meta.title:
        raise ValueError("Session title must be a non-empty string.")
    _validate_timestamp(meta.created_at, "created_at")
    _validate_timestamp(meta.updated_at, "updated_at")
    for field_name in ("cwd", "model", "lane"):
        if not isinstance(getattr(meta, field_name), str):
            raise ValueError(f"Session {field_name} must be a string.")
    for field_name in ("message_count", "input_tokens", "output_tokens"):
        value = getattr(meta, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Session {field_name} must be a non-negative integer.")


class SessionStore:
    """Filesystem-backed store for local chat sessions.

    The store is deliberately lazy: constructing it performs no filesystem
    access.  ``create`` and ``save`` create the home, sessions root, and
    session directory with mode 0700; files are written with mode 0600.
    """

    def __init__(self, home: Path | None = None) -> None:
        self.home = Path(home).expanduser() if home is not None else _default_home()
        self.root = self.home / "sessions"

    def _ensure_root(self) -> None:
        _ensure_private_dir(self.home)
        _ensure_private_dir(self.root)

    def _session_dir(self, session_id: str) -> Path:
        return self.root / _validate_session_id(session_id)

    def _serialize_session(self, session: Session) -> tuple[bytes, bytes]:
        return _serialize_meta(session.meta), _serialize_transcript(session.messages)

    def _write_session_files(self, session_id: str, meta_bytes: bytes, transcript_bytes: bytes) -> None:
        directory = self._session_dir(session_id)
        _ensure_private_dir(directory)
        # Write the transcript before the metadata: a crash between the two
        # leaves the old metadata pointing at a valid session, and the next
        # save repairs the pair.
        _atomic_write(directory / _TRANSCRIPT_FILE, transcript_bytes)
        _atomic_write(directory / _META_FILE, meta_bytes)

    def _read_index(self) -> list[SessionMeta]:
        path = self.root / _INDEX_FILE
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError, TypeError):
            return []
        if not isinstance(data, Mapping):
            return []
        version = data.get("schema_version", SCHEMA_VERSION)
        if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
            return []
        entries = data.get("sessions")
        if not isinstance(entries, list):
            return []
        metas: list[SessionMeta] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            try:
                metas.append(SessionMeta.from_dict(entry))
            except (TypeError, ValueError):
                continue
        return metas

    def _update_index(self, meta: SessionMeta) -> None:
        existing = [item for item in self._read_index() if item.id != meta.id]
        # Keep the just-written session first among equal timestamps so
        # latest() follows the most recent write even within one second.
        ordered = [meta, *existing]
        ordered.sort(key=lambda item: _parse_timestamp(item.updated_at), reverse=True)
        _atomic_write(self.root / _INDEX_FILE, _serialize_index(ordered))

    def create(
        self, cwd: str, model: str, lane: str, *, persist: bool = True
    ) -> Session:
        """Create a session, optionally deferring the disk write until save."""
        if not isinstance(cwd, str) or not isinstance(model, str) or not isinstance(lane, str):
            raise TypeError("cwd, model, and lane must be strings.")
        created = datetime.now(timezone.utc)
        created_at = created.isoformat(timespec="seconds")

        def build() -> Session:
            session_id = f"{created.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
            meta = SessionMeta(
                id=session_id,
                title=DEFAULT_TITLE,
                created_at=created_at,
                updated_at=created_at,
                message_count=0,
                cwd=cwd,
                model=model,
                lane=lane,
                input_tokens=0,
                output_tokens=0,
            )
            return Session(meta=meta)

        if not persist:
            session = build()
            _validate_meta_for_write(session.meta)
            return session

        self._ensure_root()
        for _ in range(_MAX_CREATE_ATTEMPTS):
            session = build()
            meta = session.meta
            _validate_meta_for_write(meta)
            meta_bytes, transcript_bytes = self._serialize_session(session)

            directory = self.root / meta.id
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                continue
            _harden_dir(directory)
            self._write_session_files(meta.id, meta_bytes, transcript_bytes)
            self._update_index(meta)
            return session
        raise RuntimeError(f"Could not allocate a unique session id under {self.root}.")

    def save(self, session: Session) -> None:
        """Persist metadata and transcript atomically.

        All messages are serialized before any file is touched, so a bad
        message raises ValueError with the byte-for-byte previous session
        still intact.
        """
        if not isinstance(session, Session):
            raise TypeError("save() expects a Session instance.")
        if not isinstance(session.meta, SessionMeta):
            raise ValueError("Session.meta must be a SessionMeta instance.")
        if not isinstance(session.messages, list):
            raise TypeError("Session.messages must be a list.")

        session_id = _validate_session_id(session.meta.id)
        now = _now_iso()
        if not session.meta.created_at:
            session.meta.created_at = now
        session.meta.updated_at = now
        if not isinstance(session.meta.title, str) or not session.meta.title:
            session.meta.title = DEFAULT_TITLE
        if session.meta.title == DEFAULT_TITLE:
            session.title_from_first_user()
        session.meta.message_count = len(session.messages)
        _validate_meta_for_write(session.meta)

        meta_bytes, transcript_bytes = self._serialize_session(session)
        self._ensure_root()
        self._write_session_files(session_id, meta_bytes, transcript_bytes)
        self._update_index(session.meta)

    def load(self, session_id: str) -> Session:
        """Load a session by id.

        Blank and corrupt transcript lines are skipped so one torn write does
        not make an otherwise readable session unusable.  Corrupt metadata is
        a ValueError; a missing session is a FileNotFoundError.
        """
        session_id = _validate_session_id(session_id)
        directory = self.root / session_id
        meta_path = directory / _META_FILE
        transcript_path = directory / _TRANSCRIPT_FILE

        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Session {session_id!r} was not found under {self.root} (missing {meta_path})."
            )
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                raw_meta = json.load(handle)
        except OSError as exc:
            raise ValueError(f"Could not read metadata for session {session_id!r}: {exc}") from exc
        except ValueError as exc:
            raise ValueError(
                f"Session {session_id!r} has corrupt metadata at {meta_path}: {exc}"
            ) from exc

        try:
            meta = SessionMeta.from_dict(raw_meta)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Session {session_id!r} has invalid metadata: {exc}") from exc
        if meta.id != session_id:
            raise ValueError(
                f"Session {session_id!r} metadata id {meta.id!r} does not match its directory."
            )

        if not transcript_path.is_file():
            raise FileNotFoundError(
                f"Session {session_id!r} is missing its transcript at {transcript_path}."
            )
        messages: list[dict[str, Any]] = []
        try:
            with transcript_path.open("rb") as handle:
                for raw_line in handle:
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(item, dict):
                        messages.append(item)
        except OSError as exc:
            raise ValueError(f"Could not read transcript for session {session_id!r}: {exc}") from exc

        meta.message_count = len(messages)
        return Session(meta=meta, messages=messages)

    def list(self, limit: int = 20) -> list[SessionMeta]:
        """Return indexed session metadata, newest first.

        Only index.json is consulted.  A missing or unreadable index yields an
        empty list; this method never scans the sessions directory.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        bounded = max(1, min(_MAX_LIST_LIMIT, limit))
        metas = self._read_index()
        metas.sort(key=lambda item: _parse_timestamp(item.updated_at), reverse=True)
        return metas[:bounded]

    def latest(self) -> SessionMeta | None:
        sessions = self.list(1)
        return sessions[0] if sessions else None

    def resume(self, session_id: str | None = None) -> Session | None:
        """Resume an explicit session, or the latest when id is None/"last"."""
        if session_id is None or session_id == "last":
            meta = self.latest()
            if meta is None:
                return None
            return self.load(meta.id)
        return self.load(session_id)
