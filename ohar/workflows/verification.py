"""The evidence ladder for acceptance: what was claimed, and what OHAR saw. Stdlib only.

One rule holds this module together: a unit is done only when OHAR itself ran or
observed that unit's frozen check and saw it succeed. A status word is a claim;
a model saying it finished, an operator agreeing, and a check OHAR ran are three
different things, and only the last one moves a unit to ``observed_passed``.

A unit declares ONE frozen check — a command line, or a tool plus its arguments
— identified by a digest. An observation counts only when it is of that exact
check, so a re-declared check has to be earned again. Output is digested, never
stored: the tail of a failing test run is the largest block of text a workspace
could put into a ledger, and the ledger only needs to know which run it was.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..terminal import format_duration, sanitize_text
from ..types import ApprovalRequest, Tool, ToolContext, ToolResult
from .goals import GoalLedger
from .journal import ActivityJournal, args_digest
from .store import (
    MAX_TAIL_RECORDS,
    RecordStore,
    WorkflowError,
    canonical_digest,
    clean_text,
    new_record_id,
    now_iso,
    opaque_ref,
    short_digest,
    stamp_to_epoch,
    validate_vocabularies,
)

CHECK_KINDS: tuple[str, ...] = ("command", "tool")

EVIDENCE_LADDER: tuple[str, ...] = (
    "claimed",
    "configured",
    "not_observed",
    "observed_failed",
    "observed_passed",
)

TERMINAL_STATUSES: tuple[str, ...] = ("observed_passed", "abandoned")

MAX_UNITS = 32
MAX_ATTEMPTS_PER_UNIT = 3
MAX_CHECK_CHARS = 512
MAX_OUTPUT_TAIL_BYTES = 4096

REPAIR_BLOCKERS: tuple[str, ...] = (
    "attempts_exhausted",
    "no_frozen_check",
    "read_only_policy",
    "approval_denied",
    "cancelled",
)

UNIT_STORE_NAME = "units"
MAX_STATEMENT_CHARS = 240
MAX_ATTEMPTS_HARD = 10
MAX_UNITS_SCAN = MAX_TAIL_RECORDS
OBSERVED_VIA: tuple[str, ...] = ("journal", "runner", "operator")
# Only these two rows are evidence. An `operator` row is what someone told us,
# and reconcile must not launder it into an observation.
_EVIDENCE_VIA: tuple[str, ...] = ("journal", "runner")
MISSING_OBSERVATION = "observed_passed"
MAX_MISSING_PER_UNIT = 4
MAX_OUTPUT_CHARS = 1_000_000

# A status word that says "done" without OHAR having seen it. `observed_passed`
# is in here too on purpose: it is still a word, and only `Unit.satisfied` —
# which demands the matching attempt — turns it into evidence.
_COMPLETION_CLAIMS: frozenset[str] = frozenset({"claimed", "completed", "done", "observed_passed", "satisfied"})

_UNIT_BLOCKERS = frozenset(REPAIR_BLOCKERS)
_DANGEROUS_CHECK = (
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    "> /dev/sd",
    "curl -s http",
    "wget -q http",
    ":(){",
    "sudo ",
    "chmod -R 777 /",
)

UNIT_CLAIM_BOUNDARY = (
    "A verification unit is one frozen acceptance check plus what OHAR observed about it. A status word "
    "is a claim: only an attempt OHAR recorded itself, running or watching that exact check, is evidence."
)

UNIT_HEADERS: tuple[str, ...] = ("id", "statement", "kind", "status", "attempts", "missing", "goal")


@dataclass(slots=True)
class UnitAttempt:
    """One run of the frozen check. Output never survives past its digest."""

    at: str
    ok: bool
    exit_code: int | None
    observed_via: str
    seconds: float
    output_tail_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "ok": bool(self.ok),
            "exit_code": self.exit_code,
            "observed_via": self.observed_via,
            "seconds": round(float(self.seconds or 0.0), 3),
            "output_tail_digest": self.output_tail_digest,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> "UnitAttempt":
        record = data if isinstance(data, Mapping) else {}
        via = clean_text(record.get("observed_via", ""), limit=16)
        code = record.get("exit_code")
        return cls(
            at=clean_text(record.get("at", ""), limit=64),
            ok=bool(record.get("ok")),
            exit_code=code if isinstance(code, int) and not isinstance(code, bool) else None,
            observed_via=via if via in OBSERVED_VIA else "operator",
            seconds=float(record.get("seconds") or 0.0) if isinstance(record.get("seconds"), (int, float)) else 0.0,
            output_tail_digest=opaque_ref(record.get("output_tail_digest", ""), limit=64),
        )

    @property
    def is_observed(self) -> bool:
        """OHAR watched this one, rather than being told about it."""
        return self.observed_via in _EVIDENCE_VIA


@dataclass(slots=True)
class Unit:
    """One frozen check and the evidence ladder position it has earned."""

    id: str
    statement: str
    kind: str
    check: str
    digest: str
    status: str
    goal_id: str
    criterion_ref: str
    attempts: list[UnitAttempt] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    claim_boundary: str = UNIT_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "id": self.id,
            "statement": self.statement,
            "kind": self.kind,
            "check": self.check,
            "digest": self.digest,
            "status": self.status,
            "goal_id": self.goal_id,
            "criterion_ref": self.criterion_ref,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "missing_evidence": list(self.missing_evidence),
            "satisfied": self.satisfied,
            "privacy": "metadata_only",
            "claim_boundary": self.claim_boundary,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> "Unit":
        """Repair-on-read: a stored status cannot outrank the attempts on file."""
        record = data if isinstance(data, Mapping) else {}
        kind = clean_text(record.get("kind", ""), limit=16)
        if kind not in CHECK_KINDS:
            kind = "command"
        check = clean_text(record.get("check", ""), limit=MAX_CHECK_CHARS)
        digest = opaque_ref(record.get("digest", ""), limit=64) or check_digest(kind, check)
        status = clean_text(record.get("status", ""), limit=24)
        if status not in EVIDENCE_LADDER + TERMINAL_STATUSES:
            status = "configured" if check else "not_observed"
        raw = record.get("attempts")
        rows = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ()
        attempts = [UnitAttempt.from_record(item) for item in rows][-_attempt_bound(record):]
        unit = cls(
            id=opaque_ref(record.get("id", ""), limit=64) or new_record_id("unit"),
            statement=clean_text(record.get("statement", ""), limit=MAX_STATEMENT_CHARS),
            kind=kind,
            check=check,
            digest=digest,
            status=status,
            goal_id=opaque_ref(record.get("goal_id", ""), limit=64),
            criterion_ref=opaque_ref(record.get("criterion_ref", ""), limit=64),
            attempts=attempts,
            missing_evidence=_blockers(record.get("missing_evidence")),
            # The boundary is what this ledger always means; a stored copy of it
            # is not a field a reader may restate.
            claim_boundary=UNIT_CLAIM_BOUNDARY,
        )
        unit.status = _derive_status(unit)
        unit.missing_evidence = _missing_for(unit, _attempt_bound(record))
        return unit

    @property
    def satisfied(self) -> bool:
        """`observed_passed` plus the attempt that earned it. A word alone never satisfies."""
        return self.status == "observed_passed" and any(
            attempt.ok and attempt.is_observed for attempt in self.attempts
        )

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


def _attempt_bound(record: Mapping[str, Any]) -> int:
    value = record.get("max_attempts")
    return value if isinstance(value, int) and not isinstance(value, bool) else MAX_ATTEMPTS_PER_UNIT


def check_digest(kind: str, check: str) -> str:
    """The frozen identity of one check.

    A tool check digests through `journal.args_digest`, the same function that
    journals an observed call, so "did we watch this exact call succeed" is one
    comparison and not a judgement. A command check is a line, digested directly.
    """
    name = clean_text(kind, limit=16)
    text = clean_text(check, limit=MAX_CHECK_CHARS)
    if name == "tool":
        try:
            tool, arguments = _parse_tool_check(text)
        except WorkflowError:
            return ""
        if not tool:
            return ""
        return args_digest(tool, arguments)
    if not text:
        return ""
    return "chk-" + canonical_digest({"kind": name, "check": text})[:32]


def claims_completion(status: str) -> bool:
    """True when this status word says the work is done. Says nothing about evidence."""
    return clean_text(status, limit=24) in _COMPLETION_CLAIMS


def _parse_tool_check(check: str) -> tuple[str, dict[str, Any]]:
    """`name` or `name <json-object>` — the only shape a tool check may take."""
    text = str(check or "").strip()
    if not text:
        return "", {}
    tool, _, raw = text.partition(" ")
    tool = clean_text(tool, limit=64)
    if not tool:
        return "", {}
    if not raw.strip():
        return tool, {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise WorkflowError("a tool check must be '<name> <json-object>'") from None
    if not isinstance(parsed, dict):
        raise WorkflowError("a tool check's arguments must be a JSON object")
    return tool, {str(key): value for key, value in list(parsed.items())[:32]}


def _blockers(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    out: list[str] = []
    for item in value:
        token = clean_text(item, limit=32)
        if token in _UNIT_BLOCKERS or token == MISSING_OBSERVATION:
            if token not in out:
                out.append(token)
        if len(out) >= MAX_MISSING_PER_UNIT:
            break
    return out


def _derive_status(unit: Unit) -> str:
    """The ladder position the attempts on file actually support."""
    if unit.status == "abandoned":
        return "abandoned"
    observed = [attempt for attempt in unit.attempts if attempt.is_observed]
    if observed:
        return "observed_passed" if observed[-1].ok else "observed_failed"
    if unit.status == "claimed" or any(attempt.ok for attempt in unit.attempts):
        # An operator or a model said it passed. That is a claim, never evidence.
        return "claimed"
    return "configured" if unit.check else "not_observed"


def _missing_for(unit: Unit, max_attempts: int = MAX_ATTEMPTS_PER_UNIT) -> list[str]:
    """What still has to happen before this unit can be accepted."""
    if unit.satisfied:
        return []
    if unit.status == "abandoned":
        return ["attempts_exhausted"]
    gaps: list[str] = []
    if not unit.check:
        gaps.append("no_frozen_check")
    if len(unit.attempts) >= max(1, max_attempts):
        gaps.append("attempts_exhausted")
    for blocker in unit.missing_evidence:
        if blocker in _UNIT_BLOCKERS and blocker not in gaps:
            gaps.append(blocker)
    if MISSING_OBSERVATION not in gaps:
        gaps.append(MISSING_OBSERVATION)
    return gaps[:MAX_MISSING_PER_UNIT]


class VerificationLedger:
    """Frozen checks, and only the evidence OHAR watched them produce."""

    def __init__(
        self,
        root: Path,
        workspace: Path,
        *,
        journal: ActivityJournal,
        clock: Callable[[], float] = time.time,
        max_attempts: int = MAX_ATTEMPTS_PER_UNIT,
    ) -> None:
        self.root = Path(root).expanduser()
        self.workspace = Path(workspace).expanduser().resolve(strict=False)
        self.journal = journal
        self.max_attempts = _clamp_attempts(max_attempts)
        self._clock = clock
        self._store = RecordStore(self.root, UNIT_STORE_NAME, workspace=self.workspace)

    # -- declare -------------------------------------------------------------

    def declare(
        self,
        statement: str,
        *,
        kind: str,
        check: str,
        goal_id: str = "",
        criterion_ref: str = "",
    ) -> Unit:
        """Freeze one check for one statement. Re-declaring the same digest returns the unit."""
        text = clean_text(statement, limit=MAX_STATEMENT_CHARS)
        if not text:
            raise WorkflowError("a unit needs a statement")
        name = clean_text(kind, limit=16)
        if name not in CHECK_KINDS:
            raise WorkflowError(f"check kind is unsupported: {clean_text(kind, limit=40)!r}")
        line = clean_text(check, limit=MAX_CHECK_CHARS)
        if name == "tool":
            tool, arguments = _parse_tool_check(line)
            if not tool:
                raise WorkflowError("a tool check names the tool")
            line = f"{tool} {json.dumps(arguments, sort_keys=True)}" if arguments else tool
        digest = check_digest(name, line)
        if not digest:
            raise WorkflowError("a check cannot be digested; it is empty or unreadable")
        existing = self._store.latest("digest", digest)
        if existing is not None:
            return Unit.from_record(existing)
        if len(self._store.heads("id")) >= MAX_UNITS:
            raise WorkflowError(f"the unit ledger is full ({MAX_UNITS} units); abandon one first")
        unit = Unit(
            id=new_record_id("unit"),
            statement=text,
            kind=name,
            check=line,
            digest=digest,
            status="configured",
            goal_id=opaque_ref(goal_id, limit=64),
            criterion_ref=opaque_ref(criterion_ref, limit=64),
            attempts=[],
        )
        unit.missing_evidence = _missing_for(unit, self.max_attempts)
        self._write(unit, event="unit_declared", ok=None)
        return unit

    # -- read ----------------------------------------------------------------

    def get(self, unit_id: str) -> Unit:
        record = self._store.latest("id", opaque_ref(unit_id, limit=64))
        if record is None:
            raise WorkflowError(f"no unit '{clean_text(unit_id, limit=48)}' is declared")
        return Unit.from_record(record)

    def units(self, *, goal_id: str | None = None, status: str | None = None, limit: int = 32) -> list[Unit]:
        scope = opaque_ref(goal_id, limit=64) if goal_id is not None else ""
        wanted = clean_text(status, limit=24) if status is not None else ""
        found: list[Unit] = []
        for record in self._store.heads("id").values():
            unit = Unit.from_record(record)
            if scope and unit.goal_id != scope:
                continue
            if wanted and unit.status != wanted:
                continue
            found.append(unit)
        # Declaration order: `heads` keeps the first write per id, so the listing
        # reads as the plan that was made rather than as whatever ran last.
        return found[-_clamp_limit(limit, 1, MAX_UNITS):]

    # -- evidence ------------------------------------------------------------

    def reconcile(self, unit_id: str, *, now: float | None = None) -> Unit:
        """Derive the status from what the journal observed of this exact check."""
        unit = self.get(unit_id)
        if unit.status == "abandoned":
            return unit
        observed = self._observation(unit, now=now)
        if observed is not None:
            ok = bool(observed.get("ok"))
            return self._observe(
                unit,
                ok=ok,
                exit_code=observed.get("exit_code"),
                seconds=observed.get("seconds") or 0.0,
                output="",
                observed_via="journal",
                at=_stamp(float(observed.get("epoch") or self._clock())),
            )
        return self._settle(unit)

    def reconcile_all(self, *, goal_id: str = "", now: float | None = None) -> list[Unit]:
        return [
            self.reconcile(unit.id, now=now)
            for unit in self.units(goal_id=goal_id or None, limit=MAX_UNITS)
        ]

    def ingest_tool_observation(self, event: Mapping[str, Any]) -> list[str]:
        """Fold one journaled tool observation into the units whose exact check it was.

        Matching is a digest comparison, never a similarity: a check counts only
        when the observed call is the one that was frozen.
        """
        record = event if isinstance(event, Mapping) else {}
        detail = record.get("detail") if isinstance(record.get("detail"), Mapping) else {}
        name = clean_text(record.get("subject") or detail.get("tool"), limit=64)
        if not name or record.get("event") not in (None, "", "tool_observed"):
            return []
        ok = record.get("ok")
        if not isinstance(ok, bool):
            return []
        epoch = stamp_to_epoch(str(record.get("at", ""))) or self._clock()
        stored_digest = clean_text(detail.get("args_digest") or detail.get("digest"), limit=64)
        arguments = detail.get("args") if isinstance(detail.get("args"), Mapping) else None
        moved: list[str] = []
        for unit in self.units(limit=MAX_UNITS):
            if unit.terminal or not unit.check:
                continue
            if not self._matches(unit, name, stored_digest, arguments):
                continue
            updated = self._observe(
                unit,
                ok=ok,
                exit_code=detail.get("exit_code") if isinstance(detail.get("exit_code"), int) else None,
                seconds=float(detail.get("seconds") or 0.0),
                output="",
                observed_via="journal",
                at=_stamp(epoch),
            )
            if updated.status != unit.status:
                moved.append(updated.id)
        return moved

    def record_attempt(
        self,
        unit_id: str,
        *,
        ok: bool,
        exit_code: int | None,
        seconds: float,
        output: str = "",
        observed_via: str = "operator",
    ) -> Unit:
        """Record what happened to one run of the frozen check."""
        unit = self.get(unit_id)
        via = clean_text(observed_via, limit=16)
        if via not in OBSERVED_VIA:
            raise WorkflowError(f"observed_via is unsupported: {clean_text(observed_via, limit=32)!r}")
        if unit.terminal:
            return unit
        if len(unit.attempts) >= self.max_attempts:
            return self._block(unit, "attempts_exhausted")
        return self._observe(
            unit,
            ok=bool(ok),
            exit_code=exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None,
            seconds=float(seconds or 0.0),
            output=output,
            observed_via=via,
            at=_stamp(self._clock()),
        )

    # -- run -----------------------------------------------------------------

    def run(self, unit_id: str, ctx: ToolContext, *, timeout: float = 120.0) -> Unit:
        """Run the frozen command line ourselves, in this workspace, after approval.

        An attempt is only burned when the check actually ran: a read-only mode,
        a declined approval, or a refused command shape does not consume one.
        """
        unit = self.get(unit_id)
        if unit.terminal:
            return unit
        if len(unit.attempts) >= self.max_attempts:
            return self._block(unit, "attempts_exhausted")
        if not unit.check:
            return self._block(unit, "no_frozen_check")
        if ctx.read_only:
            return self._block(unit, "read_only_policy")
        if unit.kind != "command":
            # A tool check is observed through the tool surface, not re-executed here.
            return self._block(unit, "no_frozen_check")
        try:
            granted = float(timeout)
        except (TypeError, ValueError):
            granted = 120.0
        if granted <= 0:
            # No window was granted at all: the run was cancelled before it started.
            return self._block(unit, "cancelled")
        line = unit.check
        for marker in _DANGEROUS_CHECK:
            if marker in line:
                raise WorkflowError(f"check refused by policy: the command line contains {marker.strip()!r}")
        approved = ctx.request_approval(
            ApprovalRequest(
                action="unit_run",
                summary=f"Run the frozen check for: {unit.statement[:120]}",
                detail=(
                    f"Command:\n{line}\nWorkspace: {ctx.workspace}"
                    f"\nTimeout: {format_duration(_clamp_timeout(timeout))}"
                ),
                risk="high",
            )
        )
        if not approved:
            return self._block(unit, "approval_denied")
        started = time.monotonic()
        try:
            process = subprocess.run(
                line,
                shell=True,
                executable=shutil.which("bash") or "/bin/sh",
                cwd=str(Path(ctx.workspace).expanduser().resolve(strict=False)),
                capture_output=True,
                timeout=_clamp_timeout(timeout),
                check=False,
            )
        except subprocess.TimeoutExpired:
            seconds = time.monotonic() - started
            return self._observe(
                unit, ok=False, exit_code=None, seconds=seconds, output="check timed out",
                observed_via="runner", at=_stamp(self._clock()),
            )
        except OSError as exc:
            # The check ran as far as failing to start: that is an observation of
            # a non-pass, not a policy refusal, and it burns an attempt.
            return self._observe(
                unit,
                ok=False,
                exit_code=None,
                seconds=time.monotonic() - started,
                output=f"cannot start check: {exc}",
                observed_via="runner",
                at=_stamp(self._clock()),
            )
        seconds = time.monotonic() - started
        combined = b"\n".join(part or b"" for part in (process.stdout, process.stderr))
        return self._observe(
            unit,
            ok=process.returncode == 0,
            exit_code=int(process.returncode),
            seconds=seconds,
            output=combined.decode("utf-8", errors="replace"),
            observed_via="runner",
            at=_stamp(self._clock()),
        )

    # -- gaps ----------------------------------------------------------------

    def can_satisfy(self, unit_id: str) -> tuple[bool, str]:
        """Can OHAR still earn this unit, and if not, what is in the way?"""
        unit = self.get(unit_id)
        if unit.satisfied:
            return True, ""
        if unit.status == "abandoned":
            return False, "attempts_exhausted"
        if not unit.check:
            return False, "no_frozen_check"
        for blocker in unit.missing_evidence:
            if blocker in _UNIT_BLOCKERS and blocker != "attempts_exhausted":
                return False, blocker
        if len(unit.attempts) >= self.max_attempts:
            return False, "attempts_exhausted"
        return True, ""

    def missing_evidence(self, *, goal_id: str = "") -> list[dict[str, str]]:
        gaps: list[dict[str, str]] = []
        for unit in self.units(goal_id=goal_id or None, limit=MAX_UNITS):
            if unit.satisfied:
                continue
            for item in unit.missing_evidence:
                gaps.append(
                    {
                        "unit": unit.id,
                        "statement": unit.statement,
                        "status": unit.status,
                        "missing": item,
                        "next_action": _next_action(item),
                    }
                )
        return gaps[:MAX_UNITS * MAX_MISSING_PER_UNIT]

    def evidence_summary(self, *, goal_id: str = "") -> dict[str, int]:
        totals = {status: 0 for status in EVIDENCE_LADDER}
        totals["abandoned"] = 0
        totals["units"] = 0
        totals["satisfied"] = 0
        for unit in self.units(goal_id=goal_id or None, limit=MAX_UNITS):
            totals["units"] += 1
            totals[unit.status] = totals.get(unit.status, 0) + 1
            if unit.satisfied:
                totals["satisfied"] += 1
        return totals

    def headers(self) -> list[str]:
        return list(UNIT_HEADERS)

    def rows(self, units: Sequence[Unit]) -> list[list[str]]:
        return [
            [
                clean_text(unit.id, limit=32),
                clean_text(unit.statement, limit=60),
                unit.kind,
                unit.status,
                f"{len(unit.attempts)}/{self.max_attempts}",
                ", ".join(unit.missing_evidence) or "none",
                unit.goal_id or "-",
            ]
            for unit in units
        ]

    # -- internals -----------------------------------------------------------

    def _observation(self, unit: Unit, *, now: float | None) -> dict[str, Any] | None:
        """The newest journal row that observed this exact check, or None."""
        if self.journal is None or not unit.digest:
            return None
        cutoff = self._clock() if now is None else float(now)
        try:
            events = list(self.journal.recent(event="unit_observed", subject=unit.id, limit=MAX_UNITS_SCAN))
            events += list(self.journal.recent(event="tool_observed", limit=MAX_UNITS_SCAN))
        except Exception:
            return None
        best: dict[str, Any] | None = None
        for item in events:
            detail = item.detail if isinstance(item.detail, Mapping) else {}
            epoch = item.epoch if item.epoch is not None else stamp_to_epoch(item.at)
            if epoch is None or epoch > cutoff:
                continue
            if item.event == "unit_observed":
                # Only rows that recorded an actual run count; a policy refusal or
                # a re-derivation is not an observation of the check.
                if clean_text(detail.get("digest"), limit=64) != unit.digest:
                    continue
                if clean_text(detail.get("via"), limit=16) not in _EVIDENCE_VIA:
                    continue
            elif not self._matches(unit, clean_text(item.subject, limit=64), "", None, detail=detail):
                continue
            exit_code = detail.get("exit_code")
            candidate = {
                "ok": bool(item.ok),
                "epoch": epoch,
                "exit_code": exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None,
                "seconds": float(detail.get("seconds") or 0.0)
                if isinstance(detail.get("seconds"), (int, float))
                else 0.0,
            }
            if best is None or candidate["epoch"] >= best["epoch"]:
                best = candidate
        return best

    def _matches(
        self,
        unit: Unit,
        name: str,
        stored_digest: str,
        arguments: Mapping[str, Any] | None,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> bool:
        payload = detail or {}
        digest = stored_digest or clean_text(payload.get("args_digest") or payload.get("digest"), limit=64)
        if unit.kind == "tool":
            if digest and digest == unit.digest:
                return True
            if arguments is None or name != unit.check.split(" ", 1)[0]:
                return False
            return args_digest(name, dict(arguments)) == unit.digest
        if not name or name not in ("shell", "bash", "run"):
            return False
        command = clean_text(arguments.get("command") if arguments else payload.get("command"), limit=MAX_CHECK_CHARS)
        if command and command == unit.check:
            return True
        # The journal keeps a digest, not the command line: recompute the exact call.
        return bool(digest) and digest == args_digest(name, {"command": unit.check})

    def _observe(
        self,
        unit: Unit,
        *,
        ok: bool,
        exit_code: int | None,
        seconds: float,
        output: str,
        observed_via: str,
        at: str,
    ) -> Unit:
        attempts = list(unit.attempts)
        attempts.append(
            UnitAttempt(
                at=at,
                ok=bool(ok),
                exit_code=exit_code,
                observed_via=observed_via,
                seconds=max(0.0, float(seconds or 0.0)),
                output_tail_digest=output_digest(output),
            )
        )
        attempts = attempts[-self.max_attempts :]
        updated = Unit(
            id=unit.id,
            statement=unit.statement,
            kind=unit.kind,
            check=unit.check,
            digest=unit.digest,
            status=unit.status,
            goal_id=unit.goal_id,
            criterion_ref=unit.criterion_ref,
            attempts=attempts,
            missing_evidence=[item for item in unit.missing_evidence if item not in _UNIT_BLOCKERS],
            claim_boundary=unit.claim_boundary,
        )
        updated.status = _derive_status(updated)
        updated.missing_evidence = _missing_for(updated, self.max_attempts)
        self._write(
            updated,
            event="unit_observed",
            ok=bool(ok),
            detail={
                "digest": updated.digest,
                "via": observed_via,
                "exit_code": exit_code,
                "seconds": round(float(seconds or 0.0), 3),
                "output_tail_digest": updated.attempts[-1].output_tail_digest,
            },
        )
        return updated

    def _settle(self, unit: Unit) -> Unit:
        """No new observation: re-derive from the attempts on file and rewrite only if it moved."""
        derived = _derive_status(unit)
        missing = _missing_for(unit, self.max_attempts)
        if derived == unit.status and missing == unit.missing_evidence:
            return unit
        unit.status = derived
        unit.missing_evidence = missing
        self._write(unit, event="unit_observed", ok=None, detail={"digest": unit.digest, "via": "reconcile"})
        return unit

    def _block(self, unit: Unit, blocker: str) -> Unit:
        """Record a blocker without burning an attempt: refused work was never run."""
        if blocker in unit.missing_evidence:
            return unit
        unit.missing_evidence = list(dict.fromkeys([*unit.missing_evidence, blocker]))
        unit.missing_evidence = _missing_for(unit, self.max_attempts)
        self._write(unit, event="unit_observed", ok=False, detail={"digest": unit.digest, "blocker": blocker})
        return unit

    def _write(
        self, unit: Unit, *, event: str, ok: bool | None, detail: Mapping[str, Any] | None = None
    ) -> None:
        payload = unit.to_dict()
        payload["max_attempts"] = self.max_attempts
        errors = validate_vocabularies(
            payload,
            (
                ("kind", CHECK_KINDS),
                ("status", EVIDENCE_LADDER + TERMINAL_STATUSES),
                ("missing_evidence", REPAIR_BLOCKERS + (MISSING_OBSERVATION,)),
            ),
        )
        if errors:
            raise WorkflowError("; ".join(errors))
        self._store.transact("id", unit.id, lambda current: payload)
        if self.journal is None:
            return
        try:
            self.journal.record(
                event,
                subject=unit.id,
                ok=ok,
                detail=dict(detail or {}) | {"statement": unit.statement[:80]},
            )
        except WorkflowError:
            pass

    def abandon(self, unit_id: str) -> Unit:
        """Stop earning a unit. Terminal, and the only route to `abandoned`."""
        unit = self.get(unit_id)
        unit.status = "abandoned"
        unit.missing_evidence = _missing_for(unit, self.max_attempts)
        self._write(unit, event="unit_observed", ok=False, detail={"digest": unit.digest, "via": "abandoned"})
        return unit


def output_digest(output: str) -> str:
    """A handle for the tail of one run. The tail itself is never stored."""
    text = sanitize_text(str(output or ""), keep_newlines=True)[:MAX_OUTPUT_CHARS]
    if not text:
        return ""
    tail = text.encode("utf-8", errors="replace")[-MAX_OUTPUT_TAIL_BYTES:]
    return "tail-" + short_digest(tail.decode("utf-8", errors="replace"), 24)


def _stamp(epoch: float | None = None) -> str:
    value = time.time() if epoch is None else float(epoch)
    try:
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return now_iso()


def _clamp_attempts(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return MAX_ATTEMPTS_PER_UNIT
    return max(1, min(MAX_ATTEMPTS_HARD, value))


def _clamp_limit(value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return high
    return max(low, min(high, value))


def _clamp_timeout(value: object) -> float:
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        seconds = 120.0
    return max(0.1, min(600.0, seconds))


def _next_action(blocker: str) -> str:
    return {
        "attempts_exhausted": "repair_the_check_or_abandon_the_unit",
        "no_frozen_check": "declare_a_frozen_check",
        "read_only_policy": "leave_read_only",
        "approval_denied": "allow_the_run",
        "cancelled": "retry_when_not_cancelled",
        MISSING_OBSERVATION: "unit_run",
    }.get(blocker, "unit_run")


def make_verify_tools(
    ledger: VerificationLedger,
    *,
    goals: GoalLedger | None = None,
    require_approval: bool = True,
) -> list[Tool]:
    """Declare / run / status tools. Running is approval-gated; reading never is."""

    def unit_declare(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        statement = args.get("statement")
        kind = args.get("kind")
        check = args.get("check")
        if not isinstance(statement, str) or not isinstance(kind, str) or not isinstance(check, str):
            return ToolResult.failure("unit_declare: 'statement', 'kind' and 'check' are required strings.")
        denied = _approve(
            ctx,
            require_approval,
            "unit_declare",
            f"Freeze a check for: {statement[:120]}",
            f"{kind}: {check[:MAX_CHECK_CHARS]}",
        )
        if denied is not None:
            return denied
        try:
            unit = ledger.declare(
                statement,
                kind=kind,
                check=check,
                goal_id=str(args.get("goal") or ""),
                criterion_ref=str(args.get("criterion") or ""),
            )
        except WorkflowError as exc:
            return ToolResult.failure(f"unit_declare: {exc}")
        return ToolResult.success(
            f"Unit {unit.id} frozen ({unit.kind}): {unit.statement}\nstatus: {unit.status} · digest: {unit.digest}",
            unit_id=unit.id,
            status=unit.status,
            digest=unit.digest,
        )

    def unit_run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        unit_id = args.get("unit")
        if not isinstance(unit_id, str) or not unit_id.strip():
            return ToolResult.failure("unit_run: argument 'unit' must be a non-empty string.")
        try:
            before = ledger.get(unit_id)
            unit = ledger.run(unit_id, ctx, timeout=_clamp_timeout(args.get("timeout")))
        except WorkflowError as exc:
            return ToolResult.failure(f"unit_run: {exc}", unit_id=unit_id)
        gained = unit.status == "observed_passed" and not before.satisfied
        if gained and goals is not None and unit.goal_id and unit.criterion_ref:
            # An observed pass is the only thing allowed to cite evidence back onto a criterion.
            try:
                goals.checkpoint(
                    unit.goal_id,
                    f"verified: {unit.statement}",
                    criteria_refs=[unit.criterion_ref],
                    evidence_refs=[f"unit:{unit.id}:{unit.digest}"],
                )
            except (TypeError, ValueError, OSError):
                pass
        lines = [
            f"Unit {unit.id} · {unit.status} · attempts {len(unit.attempts)}/{ledger.max_attempts}",
            f"check: {unit.check}",
        ]
        last = unit.attempts[-1] if unit.attempts else None
        if last is not None:
            lines.append(
                f"last attempt: exit={last.exit_code} via {last.observed_via} in {format_duration(last.seconds)}"
                + (f" tail {last.output_tail_digest}" if last.output_tail_digest else "")
            )
        if unit.missing_evidence:
            lines.append("still missing: " + ", ".join(unit.missing_evidence))
        if not unit.satisfied and claims_completion(unit.status):
            lines.append("note: this status is a claim, not evidence.")
        meta = {"unit_id": unit.id, "status": unit.status, "satisfied": unit.satisfied}
        text = "\n".join(lines)
        # Success means exactly one thing here: OHAR ran the frozen check and saw
        # it pass. A refusal to run and a failing run are both failures to report.
        return ToolResult.success(text, **meta) if unit.satisfied else ToolResult.failure(text, **meta)

    def unit_status(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        _ = ctx
        unit_id = args.get("unit")
        goal = args.get("goal")
        try:
            if isinstance(unit_id, str) and unit_id.strip():
                unit = ledger.get(unit_id)
                rows = ledger.rows([unit])
                headers = ledger.headers()
                body = "\n".join(f"{h}: {v}" for h, v in zip(headers[1:], rows[0]))
                can, blocker = ledger.can_satisfy(unit.id)
                return ToolResult.success(
                    f"{unit.id} · {unit.status} · satisfied={unit.satisfied}\n{body}\n"
                    f"can_satisfy: {can}{'' if not blocker else f' ({blocker})'}",
                    unit_id=unit.id,
                    status=unit.status,
                    satisfied=unit.satisfied,
                )
            found = ledger.units(goal_id=goal if isinstance(goal, str) else None, limit=MAX_UNITS)
            if not found:
                return ToolResult.success("No verification units are declared.", count=0)
            headers = ledger.headers()
            lines = [" | ".join(headers)]
            lines.extend(" | ".join(row) for row in ledger.rows(found))
            summary = ledger.evidence_summary(goal_id=goal if isinstance(goal, str) else "")
            lines.append("evidence: " + " ".join(f"{name}={count}" for name, count in summary.items()))
            gaps = ledger.missing_evidence(goal_id=goal if isinstance(goal, str) else "")
            if gaps:
                lines.append("missing evidence:")
                lines.extend(f"  {gap['unit']} · {gap['missing']} → {gap['next_action']}" for gap in gaps[:12])
            return ToolResult.success("\n".join(lines), count=len(found))
        except WorkflowError as exc:
            return ToolResult.failure(f"unit_status: {exc}")

    return [
        Tool(
            name="unit_declare",
            description=(
                "Freeze one acceptance check for one statement: kind 'command' takes a command line, "
                "kind 'tool' takes '<tool_name> <json-args>'. Completion needs OHAR to observe that "
                "exact check; declaring it records nothing about the work."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "description": "What must be true."},
                    "kind": {"type": "string", "enum": list(CHECK_KINDS)},
                    "check": {"type": "string", "description": f"The frozen check (max {MAX_CHECK_CHARS} chars)."},
                    "goal": {"type": "string", "description": "Optional goal id."},
                    "criterion": {"type": "string", "description": "Optional acceptance criterion id."},
                },
                "required": ["statement", "kind", "check"],
                "additionalProperties": False,
            },
            handler=unit_declare,
            read_only=False,
            destructive=False,
        ),
        Tool(
            name="unit_run",
            description=(
                "Run one unit's frozen command in this workspace after approval and record what OHAR "
                "saw. Bounded attempts; only an observed exit 0 satisfies the unit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "unit": {"type": "string", "description": "Unit id."},
                    "timeout": {"type": "number", "description": "Seconds, 0.1-600."},
                },
                "required": ["unit"],
                "additionalProperties": False,
            },
            handler=unit_run,
            read_only=False,
            destructive=True,
        ),
        Tool(
            name="unit_status",
            description=(
                "Read the evidence ladder: one unit, or every unit with what each still lacks. "
                "Reports claims and observations apart from each other."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "unit": {"type": "string", "description": "Unit id for one unit's detail."},
                    "goal": {"type": "string", "description": "Filter the listing by goal id."},
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=unit_status,
            read_only=True,
            destructive=False,
        ),
    ]


def _approve(ctx: ToolContext, enabled: bool, action: str, summary: str, detail: str) -> ToolResult | None:
    if ctx.read_only:
        return ToolResult.failure(f"Read-only mode: {action} is disabled.", denied=True)
    if not enabled:
        return None
    request = ApprovalRequest(
        action=action,
        summary=clean_text(summary, limit=200),
        detail=sanitize_text(clean_text(detail, limit=2_000), keep_newlines=True),
        risk="low",
    )
    if not ctx.request_approval(request):
        return ToolResult.failure(f"{action} denied; nothing was recorded.", denied=True)
    return None


