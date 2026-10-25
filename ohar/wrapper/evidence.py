"""Prepared-vs-observed evidence for one wrapper session.

Evidence is recorded by the harness from real tool results (exit code, digest,
timestamp, bounded one-line subject) and never self-reported by the model, so
completion claims are computed rather than asserted. Preconditions refuse to
mark a result verified before any command evidence exists, and declared
postcondition exit statuses are the only thing that can flip ``work_verified``.

This is the checkable form of the system-prompt rule "never claim a test passed
unless you actually ran it".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ohar.types import Tool, ToolContext, ToolResult
from ohar.wrapper import records, state

EVIDENCE_RECORD = "evidence.json"
MAX_ITEMS = 100
MAX_REFS = 8
MAX_PAYLOAD_LINES = 24
MAX_SUBJECT_CHARS = 160
MAX_DETAIL_CHARS = 320

RECORD_TYPE = "wrapper_evidence"
CLAIMS_SCHEMA_VERSION = "ohar_evidence_claims/v1"

KINDS: tuple[str, ...] = (
    "command",
    "file_write",
    "approval",
    "subagent",
    "fetch",
    "skill",
    "manual",
)
RESULTS: tuple[str, ...] = ("observed", "failed", "skipped")
CLAIMS: tuple[str, ...] = (
    "commands_ran",
    "tests_pass",
    "build_ok",
    "files_changed",
    "approvals_resolved",
    "work_verified",
)
CLAIM_STATES: tuple[str, ...] = ("observed", "pending", "refuted")

__all__ = [
    "CLAIMS",
    "CLAIMS_SCHEMA_VERSION",
    "CLAIM_BOUNDARY",
    "CLAIM_STATES",
    "EVIDENCE_RECORD",
    "KINDS",
    "MAX_DETAIL_CHARS",
    "MAX_ITEMS",
    "MAX_REFS",
    "MAX_SUBJECT_CHARS",
    "RECORD_TYPE",
    "RESULTS",
    "Claim",
    "EvidenceError",
    "EvidenceItem",
    "claims",
    "claims_payload",
    "grant_verification",
    "items",
    "make_evidence_tool",
    "observe_tool_end",
    "overclaim_guards",
    "record",
    "require_claim",
    "validate_evidence_record",
]

_TOOL_KINDS = {
    "shell": "command",
    "write_file": "file_write",
    "edit_file": "file_write",
    "task": "subagent",
    "web_fetch": "fetch",
}
_TEST_TOKENS = (
    "pytest",
    "unittest",
    "npm test",
    "yarn test",
    "pnpm test",
    "cargo test",
    "go test",
    "make test",
    "tests",
    "passed",
)
_BUILD_TOKENS = (
    "build",
    "compile",
    "compileall",
    "tsc",
    "webpack",
    "vite",
    "gradle",
    "mvn",
    "make",
)
_SUBJECT_META_KEYS = ("command", "path", "final_url", "url", "skill")
# A plan may declare the commands that count for a claim; the claim then stays
# pending instead of matching whatever string the harness happened to see.
_PLAN_KEYS = {"tests_pass": "test_commands", "build_ok": "build_commands"}

_CLAIM_LABELS = {
    "commands_ran": "command",
    "tests_pass": "test",
    "build_ok": "build",
    "files_changed": "file write",
    "approvals_resolved": "approval",
    "work_verified": "verification",
}

CLAIM_BOUNDARY = (
    "Only claims backed by recorded tool evidence are allowed here; a prepared plan, "
    "a wrapper status, and anything the model asserted are not evidence of execution, "
    "tests, build, or verification."
)


class EvidenceError(ValueError):
    """Evidence was malformed, or a claim is not supported by observed results."""


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One harness-observed fact: never a model self-report."""

    kind: str
    subject: str
    digest: str
    result: str
    exit_code: int | None
    at: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise EvidenceError(f"Evidence kind is invalid: {self.kind!r}.")
        if self.result not in RESULTS:
            raise EvidenceError(f"Evidence result is invalid: {self.result!r}.")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise EvidenceError(f"Evidence exit_code must be an integer or None: {self.exit_code!r}.")
        subject = records.clean_text(self.subject, limit=MAX_SUBJECT_CHARS)
        if not subject:
            raise EvidenceError("Evidence subject must not be empty.")
        detail = records.clean_text(self.detail, limit=MAX_DETAIL_CHARS)
        digest = records.clean_text(self.digest, limit=64)
        if not digest:
            raise EvidenceError("Evidence digest must not be empty.")
        at = records.clean_text(self.at, limit=40) or _now_iso()
        if not _valid_timestamp(at):
            raise EvidenceError(f"Evidence at must be a UTC ISO 8601 timestamp: {at!r}.")
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "detail", detail)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "at", at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "digest": self.digest,
            "result": self.result,
            "exit_code": self.exit_code,
            "at": self.at,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceItem":
        if not isinstance(data, Mapping):
            raise EvidenceError("Evidence item must be a JSON object.")
        exit_code = data.get("exit_code")
        if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
            raise EvidenceError(f"Evidence exit_code must be an integer or None: {exit_code!r}.")
        for key in ("kind", "subject", "digest", "result", "at"):
            if not isinstance(data.get(key), str):
                raise EvidenceError(f"Evidence {key!r} must be a string.")
        detail = data.get("detail", "")
        if detail is None:
            detail = ""
        if not isinstance(detail, str):
            raise EvidenceError("Evidence 'detail' must be a string.")
        return cls(
            kind=str(data["kind"]),
            subject=str(data["subject"]),
            digest=str(data["digest"]),
            result=str(data["result"]),
            exit_code=exit_code,
            at=str(data["at"]),
            detail=detail,
        )


@dataclass(frozen=True, slots=True)
class Claim:
    """A completion claim with the evidence state that supports it."""

    claim: str
    state: str
    reason: str = ""
    refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.claim not in CLAIMS:
            raise EvidenceError(f"Claim name is invalid: {self.claim!r}.")
        if self.state not in CLAIM_STATES:
            raise EvidenceError(f"Claim state is invalid: {self.state!r}.")
        object.__setattr__(self, "reason", records.clean_text(self.reason, limit=MAX_DETAIL_CHARS))
        object.__setattr__(
            self,
            "refs",
            tuple(records.clean_text(ref, limit=64) for ref in self.refs)[:MAX_REFS],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "state": self.state, "reason": self.reason, "refs": list(self.refs)}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _valid_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)


def _read_record(session_id: str) -> dict[str, Any]:
    try:
        return records.read_record(EVIDENCE_RECORD, session_id)
    except records.RecordError as exc:
        raise EvidenceError(str(exc)) from exc


def _base_record(session_id: str) -> dict[str, Any]:
    return {
        "schema_version": records.SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "session_id": session_id,
        "items": [],
        "verification": {},
    }


def validate_evidence_record(record: Mapping[str, Any]) -> None:
    """Raise when an evidence record drifts from this module's vocabulary."""
    problems = records.validate_record(record)
    if problems:
        raise EvidenceError(problems[0])
    if record.get("record_type") != RECORD_TYPE:
        raise EvidenceError(f"Evidence record_type must be {RECORD_TYPE!r}.")
    items = record.get("items")
    if not isinstance(items, list):
        raise EvidenceError("Evidence items must be a list.")
    if len(items) > MAX_ITEMS:
        raise EvidenceError(f"Evidence items exceed {MAX_ITEMS}.")
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise EvidenceError(f"Evidence items[{index}] must be an object.")
        try:
            EvidenceItem.from_dict(item)
        except EvidenceError as exc:
            raise EvidenceError(f"Evidence items[{index}]: {exc}") from exc
    verification = record.get("verification")
    if verification is not None and verification and not isinstance(verification, Mapping):
        raise EvidenceError("Evidence verification must be an object.")


def _parse_items(record: Mapping[str, Any]) -> list[EvidenceItem]:
    raw = record.get("items")
    rows = raw if isinstance(raw, list) else []
    parsed: list[EvidenceItem] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        try:
            parsed.append(EvidenceItem.from_dict(item))
        except EvidenceError:
            continue
    return parsed


def record(session_id: str, item: EvidenceItem, *, mutation_id: str | None = None) -> EvidenceItem:
    """Append one observed item to the session's evidence record."""
    if not isinstance(item, EvidenceItem):
        raise EvidenceError("record expects an EvidenceItem instance.")
    payload = item.to_dict()

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        base = dict(current) if current else _base_record(session_id)
        base.setdefault("verification", {})
        rows = [dict(row) for row in base.get("items", []) if isinstance(row, Mapping)]
        rows.append(payload)
        base["items"] = rows[-MAX_ITEMS:]
        return base

    try:
        result = records.guarded_update(
            EVIDENCE_RECORD,
            session_id,
            mutate,
            operation="record_evidence",
            mutation_id=mutation_id,
            mutation_digest=records.digest_of("record_evidence", payload),
            validate=validate_evidence_record,
        )
    except records.RecordError as exc:
        raise EvidenceError(str(exc)) from exc
    if not result.replayed:
        try:
            records.append_event(
                session_id,
                f"evidence_{item.result}",
                {
                    "kind": item.kind,
                    "subject": item.subject,
                    "exit_code": item.exit_code,
                    "digest": item.digest[:12],
                },
            )
        except records.RecordError:
            pass
    return item


def _first_line(content: str) -> str:
    for line in (content or "").splitlines():
        if line.strip():
            return line
    return ""


def _subject_for(name: str, meta: Mapping[str, Any], content: str) -> str:
    for key in _SUBJECT_META_KEYS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return records.clean_text(value, limit=MAX_SUBJECT_CHARS)
    count = meta.get("count")
    completed = meta.get("completed")
    if isinstance(count, int) and not isinstance(count, bool):
        shown = completed if isinstance(completed, int) and not isinstance(completed, bool) else "?"
        return records.clean_text(f"{name} {shown}/{count} subagents", limit=MAX_SUBJECT_CHARS)
    return records.clean_text(_first_line(content), limit=MAX_SUBJECT_CHARS) or name


def _detail_for(name: str, meta: Mapping[str, Any], elapsed: float, status: str) -> str:
    parts = [f"tool={records.clean_text(name, limit=48)}", f"result={status}", f"elapsed={elapsed:.2f}s"]
    for key in ("exit_code", "status", "bytes", "replacements", "failed", "truncated"):
        value = meta.get(key)
        if value is None or isinstance(value, (Mapping, list, tuple)):
            continue
        parts.append(f"{key}={records.clean_text(value, limit=32)}")
    return records.clean_text(" · ".join(parts), limit=MAX_DETAIL_CHARS)


def observe_tool_end(session_id: str, name: str, result: ToolResult, elapsed: float) -> EvidenceItem | None:
    """Turn one real tool end into evidence; ``None`` when it proves nothing."""
    if not isinstance(result, ToolResult):
        raise EvidenceError("observe_tool_end expects a ToolResult instance.")
    kind = _TOOL_KINDS.get(records.clean_text(name, limit=48))
    if kind is None:
        return None
    meta = result.meta if isinstance(result.meta, Mapping) else {}
    denied = meta.get("denied") is True
    if denied:
        kind, outcome = "approval", "skipped"
    else:
        outcome = "observed" if result.ok else "failed"
    raw_exit = meta.get("exit_code")
    exit_code = (
        raw_exit
        if isinstance(raw_exit, int) and not isinstance(raw_exit, bool)
        else None
    )
    try:
        seconds = max(0.0, float(elapsed))
    except (TypeError, ValueError):
        seconds = 0.0
    item = EvidenceItem(
        kind=kind,
        subject=_subject_for(name, meta, result.content),
        digest=records.digest_of(
            name, result.ok, exit_code, records.clean_text(result.content, limit=8000)
        ),
        result=outcome,
        exit_code=exit_code,
        at=_now_iso(),
        detail=_detail_for(name, meta, seconds, outcome),
    )
    return record(session_id, item)


def items(session_id: str, *, limit: int = 50) -> list[EvidenceItem]:
    """Return the most recent evidence for a session, oldest first."""
    bounded = limit if isinstance(limit, int) and not isinstance(limit, bool) else 50
    bounded = max(1, min(MAX_ITEMS, bounded))
    return _parse_items(_read_record(session_id))[-bounded:]


def _token_hit(item: EvidenceItem, declared: object, defaults: tuple[str, ...]) -> bool:
    tokens: tuple[str, ...] = defaults
    if isinstance(declared, (list, tuple)):
        chosen = tuple(str(token).lower() for token in declared if isinstance(token, str) and token.strip())
        if chosen:
            tokens = chosen
    haystack = f"{item.subject} {item.detail}".lower()
    return any(token in haystack for token in tokens)


def _matches(claim: str, item: EvidenceItem, plan: Mapping[str, Any] | None) -> bool:
    if claim == "commands_ran":
        return item.kind == "command"
    if claim == "files_changed":
        return item.kind == "file_write"
    if claim == "approvals_resolved":
        return item.kind == "approval"
    declared = plan.get(_PLAN_KEYS[claim]) if isinstance(plan, Mapping) else None
    if claim == "tests_pass":
        return item.kind == "command" and _token_hit(item, declared, _TEST_TOKENS)
    if claim == "build_ok":
        return item.kind == "command" and _token_hit(item, declared, _BUILD_TOKENS)
    return False


def _item_claim(claim: str, evidence: Sequence[EvidenceItem], plan: Mapping[str, Any] | None) -> Claim:
    label = _CLAIM_LABELS[claim]
    matched = [item for item in evidence if _matches(claim, item, plan)]
    refs = tuple(item.digest[:12] for item in matched[-MAX_REFS:])
    if not matched:
        return Claim(claim, "pending", f"no observed {label} evidence recorded", ())
    latest = matched[-1]
    if latest.result == "observed":
        if latest.exit_code not in (None, 0):
            return Claim(claim, "refuted", f"{label} evidence exited {latest.exit_code}", refs)
        return Claim(claim, "observed", f"observed {label} result: {latest.subject}", refs)
    if latest.result == "failed":
        code = f" (exit {latest.exit_code})" if latest.exit_code is not None else ""
        return Claim(claim, "refuted", f"the latest {label} evidence failed{code}: {latest.subject}", refs)
    return Claim(
        claim,
        "pending",
        f"the latest {label} evidence was skipped, not granted: {latest.subject}",
        refs,
    )


def _verification_claim(
    verification: Mapping[str, Any],
    evidence: Sequence[EvidenceItem],
    plan: Mapping[str, Any] | None,
) -> Claim:
    if isinstance(verification, Mapping) and verification.get("work_verified") in CLAIM_STATES:
        state_value = str(verification["work_verified"])
        reason = str(verification.get("reason") or "verification was granted from declared postconditions")
        raw_refs = verification.get("refs")
        refs = (
            tuple(str(ref) for ref in raw_refs if isinstance(ref, str))
            if isinstance(raw_refs, list)
            else ()
        )
        return Claim("work_verified", state_value, records.clean_text(reason), refs)
    if not any(item.kind == "command" for item in evidence):
        declared = _declared_postconditions(plan)
        suffix = f" declared postconditions {', '.join(declared)} remain unmet" if declared else ""
        return Claim("work_verified", "pending", f"no command evidence and no granted verification{suffix}")
    return Claim(
        "work_verified",
        "pending",
        "command evidence exists but verification has not been granted",
    )


def _declared_postconditions(plan: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(plan, Mapping):
        return ()
    raw = plan.get("postconditions")
    if not isinstance(raw, Mapping):
        return ()
    return tuple(str(name) for name in sorted(raw))


def claims(session_id: str, *, plan: Mapping[str, Any] | None = None) -> list[Claim]:
    """Compute the session's claims from recorded evidence only."""
    evidence = items(session_id, limit=MAX_ITEMS)
    record_row = _read_record(session_id)
    verification = record_row.get("verification")
    verification = verification if isinstance(verification, Mapping) else {}
    rows: list[Claim] = []
    for claim in CLAIMS:
        if claim == "work_verified":
            rows.append(_verification_claim(verification, evidence, plan))
        else:
            rows.append(_item_claim(claim, evidence, plan))
    return rows


def overclaim_guards(claims: Sequence[Claim]) -> list[str]:
    """Lines that say, plainly, what this session may not claim."""
    lines = [
        "A prepared plan or wrapper status is not execution evidence.",
        "Tests, builds, and verification require recorded tool results.",
    ]
    for claim in claims:
        if claim.state == "observed":
            continue
        lines.append(f"{claim.claim} is {claim.state}: {claim.reason or 'no observed evidence'}")
    return [records.clean_text(line, limit=MAX_DETAIL_CHARS) for line in lines[:MAX_PAYLOAD_LINES]]


def _next_action(rows: Sequence[Claim], closed: bool) -> str:
    if closed:
        return "session_closed"
    by = {row.claim: row for row in rows}
    if by["commands_ran"].state == "pending":
        return "run_an_observed_command"
    if by["commands_ran"].state == "refuted":
        return "surface_failed_command"
    if by["tests_pass"].state == "refuted":
        return "fix_failing_tests"
    if by["tests_pass"].state == "pending":
        return "run_tests"
    if by["build_ok"].state == "refuted":
        return "fix_broken_build"
    if by["approvals_resolved"].state == "refuted":
        return "surface_denied_approval"
    if by["work_verified"].state == "refuted":
        return "surface_failed_postconditions"
    if by["work_verified"].state == "pending":
        return "grant_verification"
    return "report_completion"


def claims_payload(session_id: str, *, plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The read-only claim surface: what is allowed, blocked, and next."""
    rows = claims(session_id, plan=plan)
    current = state.load_state(session_id)
    allowed = [row.claim for row in rows if row.state == "observed"]
    blocked = [
        {"claim": row.claim, "state": row.state, "reason": row.reason}
        for row in rows
        if row.state != "observed"
    ]
    return {
        "schema_version": CLAIMS_SCHEMA_VERSION,
        "session_id": session_id,
        "status": current.status if current is not None else "unknown",
        "items_recorded": len(items(session_id, limit=MAX_ITEMS)),
        "claims": [row.to_dict() for row in rows],
        "allowed_claims": allowed,
        "blocked_claims": blocked,
        "next_action": _next_action(rows, current.terminal if current else False),
        "overclaim_guards": overclaim_guards(rows),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def require_claim(session_id: str, claim: str, *, plan: Mapping[str, Any] | None = None) -> None:
    """Refuse a claim the recorded evidence does not support."""
    if claim not in CLAIMS:
        raise EvidenceError(f"Unknown claim {claim!r}; expected one of {', '.join(CLAIMS)}.")
    rows = claims(session_id, plan=plan)
    for row in rows:
        if row.claim != claim:
            continue
        if row.state == "observed":
            return
        raise EvidenceError(f"Refusing to claim {claim}: {row.reason or row.state}.")
    raise EvidenceError(f"Claim {claim!r} was not computed for session {session_id}.")


def grant_verification(
    session_id: str,
    postconditions: Mapping[str, bool],
    *,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
) -> dict[str, Any]:
    """Grant ``work_verified`` only from declared postcondition exit statuses."""
    if not isinstance(postconditions, Mapping) or not postconditions:
        raise EvidenceError("grant_verification requires at least one declared postcondition.")
    for name, value in postconditions.items():
        if not isinstance(name, str) or not name.strip():
            raise EvidenceError("Postcondition names must be non-empty strings.")
        if not isinstance(value, bool):
            raise EvidenceError(f"Postcondition {name!r} must be a boolean exit status.")

    current = state.load_state(session_id)
    if current is not None and current.terminal:
        raise EvidenceError(f"Cannot grant verification for a {current.status} session.")

    evidence = items(session_id, limit=MAX_ITEMS)
    commands = [item for item in evidence if item.kind == "command"]
    if not commands:
        raise EvidenceError("Cannot record verification before command evidence is observed.")
    if commands[-1].result != "observed":
        raise EvidenceError(
            f"Cannot record verification while the latest command evidence is {commands[-1].result}."
        )

    clean = {str(name): bool(value) for name, value in postconditions.items()}
    failed = sorted(name for name, value in clean.items() if not value)
    refs = tuple(item.digest[:12] for item in commands[-MAX_REFS:])
    work_verified = "observed" if not failed else "refuted"
    reason = (
        f"all {len(clean)} declared postconditions exited 0"
        if not failed
        else f"declared postconditions failed: {', '.join(failed)}"
    )
    granted_at = _now_iso()

    def mutate(row: dict[str, Any]) -> dict[str, Any]:
        base = dict(row) if row else _base_record(session_id)
        base["verification"] = {
            "granted": True,
            "granted_at": granted_at,
            "postconditions": dict(sorted(clean.items())),
            "failed_postconditions": failed,
            "work_verified": work_verified,
            "reason": records.clean_text(reason),
            "refs": list(refs),
            "command_items": len(commands),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        rows = [dict(item) for item in base.get("items", []) if isinstance(item, Mapping)]
        base["items"] = rows[-MAX_ITEMS:]
        return base

    try:
        result = records.guarded_update(
            EVIDENCE_RECORD,
            session_id,
            mutate,
            operation="grant_verification",
            expected_revision=expected_revision,
            mutation_id=mutation_id,
            mutation_digest=records.digest_of(
                "grant_verification", session_id, dict(sorted(clean.items()))
            ),
            validate=validate_evidence_record,
        )
    except records.RecordError as exc:
        raise EvidenceError(str(exc)) from exc

    if not result.replayed:
        try:
            records.append_event(
                session_id,
                "verification_granted",
                {
                    "work_verified": work_verified,
                    "failed": list(failed),
                    "revision": result.revision,
                },
            )
        except records.RecordError:
            pass

    payload = claims_payload(session_id)
    stored = result.record.get("verification")
    payload["verification"] = dict(stored) if isinstance(stored, Mapping) else {}
    payload["replayed"] = result.replayed
    return payload


def _claims_text(payload: Mapping[str, Any]) -> str:
    lines = [f"claims for {payload['session_id']} (status={payload['status']}):"]
    for row in payload["claims"]:
        name = records.clean_text(row.get("claim"), limit=32)
        row_state = records.clean_text(row.get("state"), limit=16)
        reason = records.clean_text(row.get("reason"), limit=96)
        lines.append(f"  {name:<22} {row_state:<9} {reason}")
    lines.append(f"next: {records.clean_text(payload.get('next_action'), limit=64)}")
    lines.append(f"allowed: {', '.join(payload['allowed_claims']) or 'none'}")
    lines.append(f"blocked: {', '.join(row['claim'] for row in payload['blocked_claims']) or 'none'}")
    lines.append(f"boundary: {records.clean_text(payload.get('claim_boundary'), limit=160)}")
    return "\n".join(lines[:MAX_PAYLOAD_LINES + 4])


def make_evidence_tool(session_id: str) -> Tool:
    """Build the read-only ``claim_status`` tool bound to one session."""
    try:
        checked = records.session_dir(session_id, create=False).name
    except records.RecordError as exc:
        raise EvidenceError(str(exc)) from exc

    def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        _ = args, ctx
        try:
            payload = claims_payload(checked)
        except (EvidenceError, records.RecordError) as exc:
            return ToolResult.failure(f"claim_status: {records.clean_text(exc)}")
        return ToolResult.success(
            _claims_text(payload),
            session_id=checked,
            allowed_claims=list(payload["allowed_claims"]),
            blocked_claims=len(payload["blocked_claims"]),
            next_action=payload["next_action"],
        )

    return Tool(
        name="claim_status",
        description=(
            "Report this session's completion claims and whether recorded tool "
            "evidence supports each one. Read-only: it never claims a result the "
            "harness did not observe."
        ),
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=handler,
        read_only=True,
    )
