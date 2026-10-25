"""Observed-vs-claimed spine: one metadata-only activity journal per workspace.

Everything downstream (verification, readiness, board, goal loop) derives its
evidence from these records and never from model prose. A journal row is written
only when OHAR actually saw something, so the absence of a row means
``unavailable`` — never inferred, never zero-filled.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..types import AgentHooks, ToolResult
from .store import (
    MAX_TAIL_RECORDS,
    SCHEMA_VERSION,
    WorkflowError,
    clean_text,
    opaque_ref,
    short_digest,
    stamp_to_epoch,
    workspace_key,
    RecordStore,
)

__all__ = [
    "JOURNAL_NAME",
    "EVENT_NAMES",
    "AVAILABILITY",
    "UNKNOWN_AGE_SECONDS",
    "JournalEvent",
    "ActivityJournal",
    "args_digest",
    "wrap_hooks",
]

JOURNAL_NAME = "activity"
EVENT_NAMES = frozenset(
    {
        "tool_observed",
        "turn_completed",
        "approval_answered",
        "gate_opened",
        "gate_answered",
        "unit_declared",
        "unit_observed",
        "goal_updated",
        "session_started",
        "session_ended",
        "subagent_completed",
        "loop_cycle",
    }
)
AVAILABILITY = ("observed", "unavailable")
UNKNOWN_AGE_SECONDS = -1

_PRIVACY = "metadata_only"
_MAX_SUBJECT_CHARS = 120
_MAX_SESSION_CHARS = 128
_MAX_DETAIL_KEYS = 12
_MAX_DETAIL_ITEMS = 8
_MAX_DETAIL_CHARS = 160
_FUTURE_TOLERANCE_SECONDS = 120.0
_DETAIL_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
_TERMINAL_SUCCESS = {"done", "completed", "success"}
_TERMINAL_FAILURE = {"error", "failed", "failure", "cancelled"}
_META_FLAGS = ("denied", "truncated", "timed_out", "timeout")
_META_NUMBERS = ("exit_code",)


def _bounded_value(value: object) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(-(2**53), min(2**53, value))
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return None
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        for item in list(value)[:_MAX_DETAIL_ITEMS]:
            bounded = _bounded_value(item)
            if bounded is not None:
                items.append(bounded)
        return items
    return clean_text(value, limit=_MAX_DETAIL_CHARS)


def _bounded_detail(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    detail: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if len(detail) >= _MAX_DETAIL_KEYS:
            break
        key = clean_text(raw_key, limit=32).casefold().replace(" ", "_").replace("-", "_")
        if not _DETAIL_KEY_RE.fullmatch(key):
            continue
        bounded = _bounded_value(raw_value)
        if bounded is None:
            continue
        detail[key] = bounded
    return detail


def args_digest(name: str, args: Mapping[str, Any]) -> str:
    """Identify one call without keeping its arguments."""
    payload = {"name": clean_text(name, limit=64), "args": dict(args) if isinstance(args, Mapping) else {}}
    return short_digest(payload, 16)


@dataclass(slots=True)
class JournalEvent:
    at: str
    event: str
    subject: str
    ok: bool | None
    session_id: str
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "event": self.event,
            "subject": self.subject,
            "ok": self.ok,
            "session_id": self.session_id,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JournalEvent":
        record = data if isinstance(data, Mapping) else {}
        event = clean_text(record.get("event"), limit=64)
        ok = record.get("ok")
        return cls(
            at=clean_text(record.get("at"), limit=64),
            event=event if event in EVENT_NAMES else "",
            subject=clean_text(record.get("subject"), limit=_MAX_SUBJECT_CHARS),
            ok=ok if isinstance(ok, bool) else None,
            session_id=clean_text(record.get("session_id"), limit=_MAX_SESSION_CHARS),
            detail=_bounded_detail(record.get("detail")),
        )

    @property
    def epoch(self) -> float | None:
        return stamp_to_epoch(self.at)


class ActivityJournal:
    """Append-only journal scoped to one resolved workspace."""

    def __init__(
        self,
        root: Path,
        workspace: Path,
        *,
        clock: Callable[[], float] = time.time,
        max_events: int = 2000,
    ) -> None:
        self.root = Path(root).expanduser()
        self.workspace = Path(workspace).expanduser().resolve(strict=False)
        self._clock = clock
        bound = max_events if isinstance(max_events, int) and not isinstance(max_events, bool) else 2000
        self.max_events = max(1, min(MAX_TAIL_RECORDS, bound))
        self.store = RecordStore(self.root, JOURNAL_NAME, workspace=self.workspace)

    @property
    def path(self) -> Path:
        return self.store.path()

    @property
    def scope_key(self) -> str:
        return workspace_key(self.workspace)

    def record(
        self,
        event: str,
        *,
        subject: str = "",
        ok: bool | None = None,
        session_id: str = "",
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = clean_text(event, limit=64)
        if name not in EVENT_NAMES:
            raise WorkflowError(f"journal event is not canonical: {clean_text(event, limit=40)!r}")
        if ok is not None and not isinstance(ok, bool):
            raise WorkflowError("journal ok must be a bool or None.")
        return self.store.append(
            {
                "schema_version": SCHEMA_VERSION,
                "at": self._stamp(),
                "event": name,
                "subject": clean_text(subject, limit=_MAX_SUBJECT_CHARS),
                "ok": ok,
                "session_id": clean_text(session_id, limit=_MAX_SESSION_CHARS),
                "availability": AVAILABILITY[0],
                "privacy": _PRIVACY,
                "workspace": self.scope_key,
                "detail": _bounded_detail(detail),
            }
        )

    def record_tool(
        self,
        name: str,
        args: Mapping[str, Any],
        result: ToolResult,
        *,
        seconds: float,
        session_id: str = "",
    ) -> dict[str, Any]:
        meta = result.meta if isinstance(result.meta, dict) else {}
        content = result.content if isinstance(result.content, str) else str(result.content or "")
        detail: dict[str, Any] = {
            "tool": opaque_ref(name, limit=64),
            "args_digest": args_digest(name, args),
            "seconds": _bounded_value(seconds),
            "chars": min(len(content), 2**53),
            "output_digest": short_digest(content, 16),
        }
        for key in _META_NUMBERS:
            value = meta.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                detail[key] = value
        for key in _META_FLAGS:
            value = meta.get(key)
            if isinstance(value, bool):
                detail[key] = value
        return self.record(
            "tool_observed",
            subject=clean_text(name, limit=_MAX_SUBJECT_CHARS),
            ok=bool(result.ok),
            session_id=session_id,
            detail=detail,
        )

    def record_turn(
        self,
        *,
        model: str,
        lane: str,
        input_tokens: int,
        output_tokens: int,
        seconds: float,
        session_id: str = "",
    ) -> dict[str, Any]:
        def count(value: object) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                return 0
            return max(0, value)

        return self.record(
            "turn_completed",
            subject=clean_text(model or lane, limit=_MAX_SUBJECT_CHARS),
            session_id=session_id,
            detail={
                "model": opaque_ref(model, limit=64),
                "lane": opaque_ref(lane, limit=64),
                "input_tokens": count(input_tokens),
                "output_tokens": count(output_tokens),
                "seconds": _bounded_value(seconds),
            },
        )

    def record_subagent(self, event: Mapping[str, Any], *, session_id: str = "") -> dict[str, Any]:
        """Journal a subagent completion receipt; progress chatter is dropped."""
        payload = event if isinstance(event, Mapping) else {}
        kind = clean_text(payload.get("type") or payload.get("status"), limit=32).casefold()
        if kind not in _TERMINAL_SUCCESS and kind not in _TERMINAL_FAILURE:
            return {}
        error = payload.get("error")
        detail: dict[str, Any] = {
            "subagent_id": opaque_ref(payload.get("id"), limit=64),
            "kind": kind,
            "seconds": _bounded_value(payload.get("seconds")),
            "chars": _bounded_value(payload.get("chars")),
        }
        for key in ("model", "lane"):
            if payload.get(key):
                detail[key] = opaque_ref(payload.get(key), limit=64)
        if error:
            detail["error_digest"] = short_digest(error, 16)
            detail["error"] = clean_text(error, limit=_MAX_DETAIL_CHARS)
        return self.record(
            "subagent_completed",
            subject=clean_text(payload.get("title") or payload.get("id") or "subagent", limit=_MAX_SUBJECT_CHARS),
            ok=kind in _TERMINAL_SUCCESS,
            session_id=session_id,
            detail=detail,
        )

    def _tail(self) -> list[dict[str, Any]]:
        return self.store.records(limit=self.max_events)

    def _events(self) -> list[JournalEvent]:
        now = self._now()
        events: list[JournalEvent] = []
        for record in self._tail():
            if str(record.get("workspace") or "") not in ("", self.scope_key):
                continue
            event = JournalEvent.from_dict(record)
            epoch = event.epoch
            if epoch is None or epoch > now + _FUTURE_TOLERANCE_SECONDS:
                continue
            events.append(event)
        return events

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return time.time()
        return float(value)

    def _floor(self, seconds: float | None) -> float | None:
        """Oldest timestamp still inside the window; None means no window."""
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
            return None
        return self._now() - float(seconds)

    def _stamp(self) -> str:
        return datetime.fromtimestamp(self._now(), timezone.utc).isoformat(timespec="microseconds")

    def recent(
        self,
        *,
        limit: int = 50,
        event: str | None = None,
        subject: str | None = None,
        session_id: str | None = None,
        since_seconds: float | None = None,
    ) -> list[JournalEvent]:
        count = limit if isinstance(limit, int) and not isinstance(limit, bool) else 50
        if count < 1:
            return []
        wanted = clean_text(event, limit=64) if event is not None else ""
        target = clean_text(subject, limit=_MAX_SUBJECT_CHARS) if subject is not None else ""
        scope = clean_text(session_id, limit=_MAX_SESSION_CHARS) if session_id is not None else ""
        floor = self._floor(since_seconds)
        matched = [
            item
            for item in self._events()
            if (not wanted or item.event == wanted)
            and (not target or item.subject == target)
            and (not scope or item.session_id == scope)
            and (floor is None or (item.epoch is not None and item.epoch >= floor))
        ]
        return matched[-min(count, self.max_events) :]

    def observed(
        self,
        subject: str,
        *,
        ok: bool | None = True,
        event: str | None = None,
        now: float | None = None,
    ) -> JournalEvent | None:
        target = clean_text(subject, limit=_MAX_SUBJECT_CHARS)
        if not target:
            return None
        cutoff = self._now() if now is None else float(now)
        wanted = clean_text(event, limit=64) if event is not None else ""
        for item in reversed(self.recent(limit=self.max_events)):
            epoch = item.epoch
            if epoch is None or epoch > cutoff + _FUTURE_TOLERANCE_SECONDS:
                continue
            if item.subject != target:
                continue
            if wanted and item.event != wanted:
                continue
            if ok is not None and item.ok is not ok:
                continue
            return item
        return None

    def age_seconds(self, subject: str, *, now: float | None = None) -> int:
        target = clean_text(subject, limit=_MAX_SUBJECT_CHARS)
        cutoff = self._now() if now is None else float(now)
        for item in reversed(self.recent(limit=self.max_events)):
            if item.subject != target:
                continue
            epoch = item.epoch
            if epoch is None or epoch > cutoff + _FUTURE_TOLERANCE_SECONDS:
                continue
            return max(0, int(cutoff - epoch))
        return UNKNOWN_AGE_SECONDS

    def counts(self, *, within_seconds: float | None = None) -> dict[str, int]:
        """Event-name counts inside the window; names with no rows are absent."""
        floor = self._floor(within_seconds)
        totals: dict[str, int] = {}
        for item in self._events():
            epoch = item.epoch
            if not item.event:
                continue
            if floor is not None and (epoch is None or epoch < floor):
                continue
            totals[item.event] = totals.get(item.event, 0) + 1
        return totals

    def headers(self) -> list[str]:
        return ["at", "event", "subject", "status", "age", "availability"]

    def render_rows(self, *, limit: int = 15) -> list[list[str]]:
        count = limit if isinstance(limit, int) and not isinstance(limit, bool) else 15
        events = self.recent(limit=max(0, min(count, self.max_events)))
        now = self._now()
        rows: list[list[str]] = []
        for item in reversed(events):
            epoch = item.epoch
            age = "--" if epoch is None else f"{max(0, int(now - epoch))}s"
            status = "unknown" if item.ok is None else ("ok" if item.ok else "failed")
            rows.append(
                [
                    clean_text(_render_time(epoch), limit=32),
                    item.event,
                    clean_text(item.subject, limit=40),
                    status,
                    age,
                    AVAILABILITY[0],
                ]
            )
        return rows


def _render_time(epoch: float | None) -> str:
    if epoch is None:
        return "--:--:--"
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%H:%M:%S")


def _safe_record(write: Callable[[], Any]) -> None:
    """An observation that cannot be stored must not break the turn that made it."""
    try:
        write()
    except (OSError, ValueError):
        pass


def wrap_hooks(hooks: AgentHooks, journal: ActivityJournal, *, session_id: str = "") -> AgentHooks:
    """Wrap existing hooks so every observation lands in the journal."""
    base = hooks if isinstance(hooks, AgentHooks) else AgentHooks()
    pending: dict[str, dict[str, Any]] = {}

    def tool_start(name: str, args: dict[str, Any]) -> None:
        base.tool_start(name, args)
        if len(pending) > 64:
            pending.clear()
        pending[str(name)] = args if isinstance(args, dict) else {}

    def tool_end(name: str, result: ToolResult, seconds: float) -> None:
        base.tool_end(name, result, seconds)
        args = pending.pop(str(name), {})
        _safe_record(lambda: journal.record_tool(name, args, result, seconds=seconds, session_id=session_id))

    def turn_end(payload: dict[str, Any]) -> None:
        base.turn_end(payload)
        data = payload if isinstance(payload, dict) else {}

        def number(key: str) -> int:
            value = data.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) else 0

        elapsed = data.get("elapsed_s")
        _safe_record(
            lambda: journal.record_turn(
                model=str(data.get("model") or ""),
                lane=str(data.get("lane") or ""),
                input_tokens=number("input_tokens"),
                output_tokens=number("output_tokens"),
                seconds=float(elapsed) if isinstance(elapsed, (int, float)) else 0.0,
                session_id=session_id,
            ),
        )

    def subagent(payload: dict[str, Any]) -> None:
        base.subagent(payload)
        _safe_record(lambda: journal.record_subagent(payload, session_id=session_id))

    return AgentHooks(
        assistant_start=base.assistant_start,
        assistant_end=base.assistant_end,
        text=base.text,
        reasoning=base.reasoning,
        tool_start=tool_start,
        tool_end=tool_end,
        notice=base.notice,
        subagent=subagent,
        usage=base.usage,
        turn_end=turn_end,
    )
