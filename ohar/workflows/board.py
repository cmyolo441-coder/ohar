"""A read-only work board: what needs a decision, what is blocked, what is moving, what can start.

The board is a projection over the ledgers OHAR already keeps (goals, gates,
blocks, units, journal). It is never a second execution ledger: nothing here
writes, dispatches, approves, or verifies.

Three rules shape every lane decision:

* A status word is a claim, not evidence. `done`, `satisfied`, `observed_passed`
  move an item out of the ready lanes and the evidence it still lacks stays
  listed in `missing_evidence`; `closed` is unreachable from a claim alone.
* Lanes are ordered by who has to act: a human decision first, then blocked
  work, then work already moving, then work that can start.
* `now` is a parameter and the digest covers only the ordered items, so
  identical ledger state hashes identically whenever it is read.
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..terminal import Theme, format_duration, make_theme, sanitize_text, strip_ansi, table
from ..types import Tool, ToolContext, ToolResult
from .blocks import BlockLog
from .gates import DecisionGates
from .goals import GoalLedger
from .journal import ActivityJournal
from .store import clean_text, canonical_digest, now_iso, stamp_to_epoch, validate_vocabularies
from .verification import VerificationLedger

BOARD_LANES: tuple[str, ...] = (
    "needs_decision",
    "blocked",
    "moving",
    "ready",
    "recorded",
    "closed",
)

ITEM_KINDS: tuple[str, ...] = ("goal", "criterion", "unit", "gate", "block", "session_activity")

DEFAULT_LIMIT = 20
MAX_TITLE_CHARS = 120
MAX_REASON_CHARS = 160
UNASSIGNED = "unassigned"

# Bounded intake: the board reads a tail of each ledger, never a whole history.
SCAN_LIMIT = 40
MAX_ITEMS = 200
MAX_GAPS = 6
UNKNOWN_AGE_SECONDS = -1
ACTIVE_WINDOW_SECONDS = 900

LANE_LABELS: dict[str, str] = {
    "needs_decision": "needs a decision",
    "blocked": "blocked",
    "moving": "moving",
    "ready": "ready to start",
    "recorded": "recorded, unevidenced",
    "closed": "closed",
}

CLAIM_BOUNDARY = (
    "The board is a read-only projection of ledgers OHAR already keeps. A status word (done, complete, "
    "satisfied, observed_passed) is a claim, never evidence: it moves an item out of the ready lanes and "
    "leaves whatever it still lacks listed as missing evidence. A prepared action is not an invocation; "
    "nothing on this board ran, approved, verified, or wrote anything."
)

_BLOCKED_STATUSES = frozenset({"blocked", "denied", "declined", "failed", "observed_failed"})
_RUNNING_STATUSES = frozenset({"active", "attempted_not_completed", "in_progress", "running", "started"})
_CLAIM_STATUSES = frozenset(
    {
        "answered",
        "claimed",
        "complete",
        "completed",
        "done",
        "finished",
        "ok",
        "observed_passed",
        "passed",
        "satisfied",
        "success",
        "verified",
    }
)
_CLOSED_STATUSES = frozenset({"abandoned", "archived", "cancelled", "ended", "expired", "revoked"})
_STARTABLE_STATUSES = frozenset({"configured", "not_observed", "pending", "prepared", "todo"})
_DECISION_STATUSES = frozenset({"awaiting_answer", "open"})
_OPERATOR_RECOVERY = frozenset({"allow_action", "answer_gate", "leave_read_only", "run_recovery_recipe"})
_NO_EVIDENCE_GAP = "no observed evidence backs this status word"


@dataclass(slots=True)
class BoardItem:
    id: str
    kind: str
    lane: str
    title: str
    status: str
    owner: str
    goal_id: str
    age_seconds: int
    missing_evidence: list[str] = field(default_factory=list)
    next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "lane": self.lane,
            "title": self.title,
            "status": self.status,
            "owner": self.owner,
            "goal_id": self.goal_id,
            "age_seconds": self.age_seconds,
            "missing_evidence": list(self.missing_evidence),
            "next_action": self.next_action,
        }


@dataclass(slots=True)
class Board:
    taken_at: str
    items: list[BoardItem]
    counts: dict[str, int]
    digest: str
    truncated: bool
    claim_boundary: str

    @property
    def total(self) -> int:
        return sum(self.counts.get(lane, 0) for lane in BOARD_LANES)

    def summary(self) -> str:
        if self.total == 0:
            return "Nothing recorded yet: no goal, gate, unit, or block to project."
        parts = [f"{self.counts[lane]} {LANE_LABELS[lane]}" for lane in BOARD_LANES if self.counts.get(lane)]
        return f"{self.total} board items: " + ", ".join(parts) + "."

    def lane_items(self, lane: str, *, limit: int = DEFAULT_LIMIT) -> list[BoardItem]:
        wanted = [item for item in self.items if item.lane == lane]
        return wanted[: max(0, int(limit))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "taken_at": self.taken_at,
            "items": [item.to_dict() for item in self.items],
            "counts": dict(self.counts),
            "digest": self.digest,
            "truncated": self.truncated,
            "claim_boundary": self.claim_boundary,
        }


def lane_for(
    kind: str,
    status: str,
    *,
    has_open_gate: bool,
    evidence: Sequence[str],
    age_seconds: int,
) -> str:
    """The one lane an item belongs to. Pure: the same vocabulary in gives the same lane out."""
    observed = [str(entry) for entry in evidence if str(entry).strip()]
    word = str(status or "").strip().lower()
    if has_open_gate or word in _DECISION_STATUSES:
        return "needs_decision"
    if word in _BLOCKED_STATUSES:
        return "blocked"
    if word in _CLOSED_STATUSES:
        return "closed"
    if word in _CLAIM_STATUSES:
        # The claim boundary as a lane rule: a claim alone is only a record.
        return "closed" if observed else "recorded"
    if word in _RUNNING_STATUSES:
        if observed or age_seconds >= 0:
            return "moving"
        return "ready"
    if observed and kind != "session_activity":
        # Something was observed, so the work is in flight even if nobody has
        # touched the status word yet.
        return "moving"
    if word in _STARTABLE_STATUSES or not word:
        return "ready"
    # An unrecognised status word never gets to read as "ready to start".
    return "recorded"


def board_digest(items: Sequence[BoardItem]) -> str:
    """Digest of the ordered items only — never of `taken_at`, never of one page."""
    return canonical_digest([item.to_dict() for item in items])


def build_board(
    *,
    goals: GoalLedger,
    gates: DecisionGates | None = None,
    blocks: BlockLog | None = None,
    units: VerificationLedger | None = None,
    journal: ActivityJournal | None = None,
    session_id: str = "",
    limit: int = 60,
    now: float | None = None,
) -> Board:
    """Project the ledgers into one ordered board. Read-only, deterministic in `now`."""
    moment = time.time() if now is None else float(now)
    items: list[BoardItem] = []
    ledger_ids: set[str] = set()

    unit_payloads: list[dict[str, Any]] = []
    unit_rows: list[Any] = []
    if units is not None:
        try:
            unit_rows = list(units.units(limit=SCAN_LIMIT))
        except Exception:  # one unreadable ledger must not take the whole board down
            unit_rows = []
        for unit in unit_rows:
            unit_payloads.append(unit.to_dict())

    open_gates: list[Any] = []
    gated_subjects: set[str] = set()
    if gates is not None:
        try:
            open_gates = list(gates.open_gates(session_id=None, limit=SCAN_LIMIT))
        except Exception:
            open_gates = []
        gated_subjects = {str(gate.subject) for gate in open_gates if str(gate.subject or "")}
        for gate in open_gates:
            expired = gates.is_expired(gate, now=moment)
            items.append(_gate_item(gate, moment, expired=expired))

    goal_rows: list[Any] = []
    if goals is not None:
        try:
            goal_rows = list(goals.list(limit=SCAN_LIMIT))
        except Exception:
            goal_rows = []
        for goal in goal_rows:
            ledger_ids.add(str(goal.id))
            gate = _goal_gate(goals, goal, unit_payloads)
            items.append(_goal_item(goal, gate, gated_subjects, moment))
            items.extend(_criterion_items(goal, gate, gated_subjects, moment))

    for unit in unit_rows:
        ledger_ids.add(str(unit.id))
        items.append(_unit_item(unit, gated_subjects, moment))

    if blocks is not None:
        try:
            records = blocks.recent(session_id=session_id or None, limit=SCAN_LIMIT)
        except Exception:
            records = []
        for record in records:
            items.append(_block_item(record, moment))

    if journal is not None:
        try:
            items.extend(
                _activity_items(journal, session_id=session_id, moment=moment, ledger_ids=ledger_ids)
            )
        except Exception:
            pass

    items = [_finalize(item) for item in items]
    items.sort(key=_sort_key)
    counts = {lane: sum(1 for item in items if item.lane == lane) for lane in BOARD_LANES}
    digest = board_digest(items)
    page = items[: max(0, min(int(limit), MAX_ITEMS))]
    return Board(
        taken_at=now_iso(),
        items=page,
        counts=counts,
        digest=digest,
        truncated=len(items) > len(page),
        claim_boundary=CLAIM_BOUNDARY,
    )


def _goal_gate(ledger: GoalLedger, goal: Any, unit_payloads: Sequence[Mapping[str, Any]]) -> Any:
    try:
        return ledger.gate(str(goal.id), units=list(unit_payloads))
    except Exception:
        return None


def _age(at: object, moment: float) -> int:
    epoch = stamp_to_epoch(str(at or ""))
    if epoch is None:
        # An unreadable stamp cannot be shown to be recent.
        return UNKNOWN_AGE_SECONDS
    return max(0, int(moment - epoch))


def _title(value: object, fallback: str) -> str:
    return clean_text(value, limit=MAX_TITLE_CHARS) or fallback


def _ref(value: object, fallback: str = "") -> str:
    return clean_text(value, limit=64) or fallback


def _owner(value: object) -> str:
    return clean_text(value, limit=64) or UNASSIGNED


def _gap(value: object) -> str:
    return clean_text(value, limit=MAX_REASON_CHARS)


def _gaps(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) >= MAX_GAPS:
            break
    return out


def _finalize(item: BoardItem) -> BoardItem:
    """Keep the lane and the listed evidence honest with each other."""
    if item.lane == "closed" and item.missing_evidence:
        item.lane = "recorded"
    if item.lane == "recorded" and not item.missing_evidence:
        item.missing_evidence = [_NO_EVIDENCE_GAP]
    if item.lane == "needs_decision" and not item.next_action:
        item.next_action = "answer_gate"
    if item.lane == "blocked" and not item.next_action:
        item.next_action = "answer_gate"
    return item


def _gate_item(gate: Any, moment: float, *, expired: bool) -> BoardItem:
    status = clean_text(getattr(gate, "state", ""), limit=32) or "open"
    subject = _ref(getattr(gate, "subject", ""))
    question = _gap(getattr(gate, "question", "")) or _ref(getattr(gate, "question_code", ""))
    actor = clean_text(getattr(gate, "actor", ""), limit=64)
    return BoardItem(
        id=f"gate:{_ref(getattr(gate, 'id', ''), 'gate')}",
        kind="gate",
        lane=lane_for("gate", status, has_open_gate=True, evidence=[], age_seconds=_age(getattr(gate, "opened_at", ""), moment)),
        title=_title(f"{subject}: {question}" if subject else question, "decision gate"),
        status=status,
        owner=actor or "operator",
        goal_id=subject if subject.startswith("goal") else "",
        age_seconds=_age(getattr(gate, "opened_at", ""), moment),
        missing_evidence=[_gap("no answer recorded")],
        next_action="reopen_the_expired_gate" if expired else "answer_gate",
    )


def _goal_item(goal: Any, gate: Any, gated: set[str], moment: float) -> BoardItem:
    goal_id = _ref(getattr(goal, "id", ""))
    status = clean_text(getattr(goal, "status", ""), limit=32) or "unknown"
    age = _age(getattr(goal, "updated_at", ""), moment)
    checkpoints = [entry for entry in list(getattr(goal, "checkpoints", []) or []) if entry is not None]
    observed = [_gap(f"checkpoint {getattr(checkpoints[-1], 'id', '')}")] if checkpoints else []
    gaps: list[str] = []
    for blocker in list(getattr(goal, "blockers", []) or []):
        if clean_text(getattr(blocker, "status", ""), limit=32) == "active":
            gaps.append(_gap(f"blocker {getattr(blocker, 'id', '')}: {getattr(blocker, 'summary', '')}"))
    if gate is not None:
        gaps.extend(_gap(entry) for entry in _gate_strings(gate))
    if goal_id in gated:
        gaps.append(_gap("gate answer"))
    next_action = _gap(getattr(goal, "next_action", "")) or "show_status"
    if gate is not None:
        next_action = _gap(getattr(gate, "next_action", "")) or next_action
    return BoardItem(
        id=f"goal:{goal_id}",
        kind="goal",
        lane=lane_for("goal", status, has_open_gate=goal_id in gated, evidence=observed, age_seconds=age),
        title=_title(getattr(goal, "objective", ""), goal_id or "goal"),
        status=status,
        owner=_owner(getattr(goal, "session_id", "")),
        goal_id=goal_id,
        age_seconds=age,
        missing_evidence=_gaps(gaps),
        next_action=next_action,
    )


def _gate_strings(gate: Any) -> list[str]:
    """A completion gate's three gap lists, as board-readable lines."""
    out: list[str] = []
    for field_name, label in (
        ("missing_criteria", "criterion"),
        ("active_blockers", "blocker"),
        ("unverified_units", "unit"),
    ):
        for entry in list(getattr(gate, field_name, []) or []):
            if isinstance(entry, Mapping):
                ident = str(entry.get("id", "") or "")
                summary = str(entry.get("summary", "") or entry.get("statement", "") or "")
                out.append(f"{label} {ident or summary}".strip())
            else:
                out.append(f"{label} {entry}")
    return out


def _criterion_items(goal: Any, gate: Any, gated: set[str], moment: float) -> list[BoardItem]:
    if clean_text(getattr(goal, "status", ""), limit=32) in ("cancelled", "complete", "failed"):
        return []
    goal_id = _ref(getattr(goal, "id", ""))
    age = _age(getattr(goal, "updated_at", ""), moment)
    items: list[BoardItem] = []
    for criterion in list(getattr(goal, "criteria", []) or []):
        status = clean_text(getattr(criterion, "status", ""), limit=32) or "pending"
        refs = [str(entry) for entry in list(getattr(criterion, "evidence_refs", []) or []) if str(entry).strip()]
        criterion_id = _ref(getattr(criterion, "id", ""), "criterion")
        gaps = [] if refs else [_gap("no unit observed satisfying this criterion")]
        items.append(
            BoardItem(
                id=f"criterion:{goal_id}/{criterion_id}",
                kind="criterion",
                lane=lane_for(
                    "criterion",
                    status,
                    has_open_gate=goal_id in gated,
                    evidence=[_gap(f"evidence {ref}") for ref in refs],
                    age_seconds=age,
                ),
                title=_title(getattr(criterion, "summary", ""), criterion_id),
                status=status,
                owner=_owner(getattr(goal, "session_id", "")),
                goal_id=goal_id,
                age_seconds=age,
                missing_evidence=gaps,
                next_action=(
                    "answer_gate"
                    if goal_id in gated
                    else "verify_unit"
                    if gaps
                    else "record_checkpoint"
                ),
            )
        )
    return items[:MAX_GAPS]


def _unit_item(unit: Any, gated: set[str], moment: float) -> BoardItem:
    status = clean_text(getattr(unit, "status", ""), limit=32) or "not_observed"
    attempts = list(getattr(unit, "attempts", []) or [])
    goal_id = _ref(getattr(unit, "goal_id", ""))
    observed = [_gap(f"attempt via {getattr(attempts[-1], 'observed_via', '')}")] if attempts else []
    gaps = [_gap(entry) for entry in list(getattr(unit, "missing_evidence", []) or [])]
    if not attempts:
        gaps.append(_gap("no observed run of the frozen check"))
    age = _age(getattr(attempts[-1], "at", ""), moment) if attempts else UNKNOWN_AGE_SECONDS
    startable = status in _STARTABLE_STATUSES or status == "claimed"
    return BoardItem(
        id=f"unit:{_ref(getattr(unit, 'id', ''), 'unit')}",
        kind="unit",
        lane=lane_for("unit", status, has_open_gate=bool(goal_id) and goal_id in gated, evidence=observed, age_seconds=age),
        title=_title(getattr(unit, "statement", ""), _ref(getattr(unit, "id", ""), "unit")),
        status=status,
        owner=_owner(goal_id),
        goal_id=goal_id,
        age_seconds=age,
        missing_evidence=_gaps(gaps),
        next_action="unit_run" if startable else "unit_status",
    )


def _block_item(record: Any, moment: float) -> BoardItem:
    outcome = clean_text(getattr(record, "outcome", ""), limit=32) or "blocked"
    recovery = clean_text(getattr(record, "recovery_action", ""), limit=64)
    enforcement = clean_text(getattr(record, "enforcement", ""), limit=32)
    age = _age(getattr(record, "at", ""), moment)
    gaps: list[str] = []
    if outcome in ("blocked", "failed", "cancelled"):
        if enforcement != "enforced":
            gaps.append(_gap("enforcement was declared, not observed"))
        if not recovery or recovery == "no_recovery_available":
            gaps.append(_gap("no recovery recipe recorded"))
    return BoardItem(
        id=f"block:{_ref(getattr(record, 'id', ''), 'block')}",
        kind="block",
        lane=lane_for("block", outcome, has_open_gate=False, evidence=[_gap("block recorded")], age_seconds=age),
        title=_title(
            f"{getattr(record, 'action', '')} refused by {getattr(record, 'reason_domain', '')}",
            "blocked work",
        ),
        status=outcome,
        owner="operator" if recovery in _OPERATOR_RECOVERY else UNASSIGNED,
        goal_id=_ref(getattr(record, "goal_id", "")),
        age_seconds=age,
        missing_evidence=_gaps(gaps),
        next_action=recovery or "answer_gate",
    )


def _activity_items(
    journal: ActivityJournal,
    *,
    session_id: str,
    moment: float,
    ledger_ids: set[str],
) -> list[BoardItem]:
    newest: dict[str, Any] = {}
    for event in journal.recent(limit=SCAN_LIMIT, session_id=session_id or None):
        subject = clean_text(getattr(event, "subject", ""), limit=64)
        if not subject or subject in ledger_ids:
            # The ledger already answers for this subject; a second row would be
            # a second claim about the same work.
            continue
        current = newest.get(subject)
        if current is None or str(getattr(event, "at", "")) >= str(getattr(current, "at", "")):
            newest[subject] = event
    items: list[BoardItem] = []
    for subject, event in sorted(newest.items()):
        age = _age(getattr(event, "at", ""), moment)
        ok = getattr(event, "ok", None)
        if ok is False:
            status = "failed"
        elif 0 <= age <= ACTIVE_WINDOW_SECONDS:
            status = "active"
        else:
            status = "idle"
        gaps: list[str] = []
        if status in ("failed", "idle"):
            gaps.append(_gap("no ledger record carries this activity"))
        items.append(
            BoardItem(
                id=f"session:{subject}",
                kind="session_activity",
                lane=lane_for(
                    "session_activity",
                    status,
                    has_open_gate=False,
                    evidence=[_gap(getattr(event, "event", "activity"))],
                    age_seconds=age,
                ),
                title=_title(f"{getattr(event, 'event', 'activity')} {subject}", subject),
                status=status,
                owner=_owner(getattr(event, "session_id", "")),
                goal_id="",
                age_seconds=age,
                missing_evidence=gaps,
                next_action="retry_with_scope" if status == "failed" else "show_status",
            )
        )
    return items[:SCAN_LIMIT]


def _sort_key(item: BoardItem) -> tuple[int, int, int, str]:
    lane = BOARD_LANES.index(item.lane) if item.lane in BOARD_LANES else len(BOARD_LANES)
    kind = ITEM_KINDS.index(item.kind) if item.kind in ITEM_KINDS else len(ITEM_KINDS)
    # Freshest first inside a lane; an unknown age sorts with the oldest.
    age = -item.age_seconds if item.age_seconds >= 0 else 1
    return (lane, kind, age, item.id)


def validate_board(payload: Mapping[str, Any]) -> list[str]:
    """Every reason a payload is not a trustworthy board."""
    if not isinstance(payload, Mapping):
        return ["board must be an object"]
    errors: list[str] = []
    if len(sanitize_text(str(payload.get("taken_at", "") or ""), keep_newlines=False).split()) != 1 or not str(
        payload.get("taken_at", "") or ""
    ).strip():
        errors.append("taken_at must be one non-empty line")
    digest = str(payload.get("digest", "") or "").strip()
    if len(digest) < 8 or len(digest.split()) != 1:
        errors.append("digest must be one bounded token covering the ordered items")
    boundary = str(payload.get("claim_boundary", "") or "")
    if not sanitize_text(boundary, keep_newlines=False).strip():
        errors.append("claim_boundary is required")
    if not isinstance(payload.get("truncated"), bool):
        errors.append("truncated must be boolean")
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        errors.append("counts must be an object")
    elif sorted(str(key) for key in counts) != sorted(BOARD_LANES):
        errors.append("counts must carry exactly every board lane")
    elif any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()
    ):
        errors.append("every lane count must be a non-negative integer")
    entries = payload.get("items")
    if not isinstance(entries, list):
        errors.append("items must be a list")
        return errors
    if isinstance(counts, Mapping):
        listed = sum(1 for item in entries if isinstance(item, Mapping) and item.get("lane") in BOARD_LANES)
        if payload.get("truncated") is False and listed != sum(counts.get(lane, 0) for lane in BOARD_LANES):
            errors.append("an untruncated board must display every counted item")
    for index, item in enumerate(entries):
        for error in _item_errors(item):
            errors.append(f"items[{index}]: {error}")
    return errors


def _item_errors(item: Any) -> list[str]:
    if not isinstance(item, Mapping):
        return ["item must be an object"]
    errors = validate_vocabularies(item, [("kind", ITEM_KINDS), ("lane", BOARD_LANES)])
    for key in ("id", "kind", "lane", "title", "status", "owner", "age_seconds", "missing_evidence", "next_action"):
        if key not in item:
            errors.append(f"item is missing {key}")
    if not str(item.get("id", "") or "").strip():
        errors.append("item id is required")
    if not str(item.get("goal_id", "") or "").strip() and "goal_id" not in item:
        errors.append("item is missing goal_id")
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append("title must be a non-empty single line")
    elif len(title) > MAX_TITLE_CHARS:
        errors.append(f"title must be at most {MAX_TITLE_CHARS} characters")
    elif title != sanitize_text(title, keep_newlines=False):
        errors.append("title must not carry control or ANSI characters")
    missing = item.get("missing_evidence")
    if not isinstance(missing, list):
        errors.append("missing_evidence must be a list")
    else:
        for entry in missing:
            if not isinstance(entry, str) or not entry.strip() or len(entry) > MAX_REASON_CHARS:
                errors.append("every missing_evidence entry must be a bounded non-empty line")
    next_action = item.get("next_action")
    if not isinstance(next_action, str) or len(next_action) > MAX_REASON_CHARS:
        errors.append(f"next_action must be a line of at most {MAX_REASON_CHARS} characters")
    age = item.get("age_seconds")
    if not isinstance(age, int) or isinstance(age, bool) or age < UNKNOWN_AGE_SECONDS:
        errors.append(f"age_seconds must be an integer, or {UNKNOWN_AGE_SECONDS} when unknown")
    lane = str(item.get("lane", "") or "")
    if lane == "needs_decision" and not str(next_action or "").strip():
        errors.append("needs_decision requires the decision it is waiting for")
    if lane == "blocked" and not str(next_action or "").strip() and not str(item.get("owner", "") or "").strip():
        errors.append("blocked requires an owner or a recovery action")
    if lane == "recorded" and isinstance(missing, list) and not missing:
        errors.append("recorded requires the evidence it lacks to stay listed")
    if lane == "closed" and isinstance(missing, list) and missing:
        errors.append("closed cannot carry missing evidence")
    return errors


def headers() -> list[str]:
    return ["lane", "kind", "id", "title", "status", "owner", "age", "missing evidence", "next action"]


def rows(board: Board, *, limit: int = DEFAULT_LIMIT) -> list[list[str]]:
    return [_row(item) for item in board.items[: max(0, int(limit))]]


def _row(item: BoardItem) -> list[str]:
    age = format_duration(item.age_seconds) if item.age_seconds >= 0 else "unknown"
    missing = "; ".join(item.missing_evidence) or "none"
    return [
        LANE_LABELS.get(item.lane, item.lane),
        item.kind,
        item.id,
        item.title,
        item.status,
        item.owner,
        age,
        clean_text(missing, limit=MAX_REASON_CHARS),
        clean_text(item.next_action, limit=MAX_REASON_CHARS),
    ]


def render(board: Board, *, theme: Theme | None = None, glyphs: str = "unicode") -> list[str]:
    """Terminal lines: heading, one table per lane, explicit truncation, claim boundary."""
    paint = theme or make_theme("mono", color=False)
    dash = "—" if glyphs != "ascii" else "-"
    lines = [
        paint.paint("Work board", fg="accent", bold=True)
        + paint.paint(f" {dash} {board.total} items", fg="dim")
        + paint.paint(f" · digest {board.digest[:12]}", fg="muted"),
        board.summary(),
    ]
    for lane in BOARD_LANES:
        lane_items = [item for item in board.items if item.lane == lane]
        if not lane_items:
            continue
        label = LANE_LABELS.get(lane, lane)
        heading = label.upper() if glyphs == "ascii" else label.capitalize()
        lines.append("")
        lines.append(paint.paint(f"{heading} ({board.counts.get(lane, 0)})", fg="accent2", bold=True))
        lines.extend(table(paint, headers(), [_row(item) for item in lane_items], glyphs=glyphs))
    if not board.items:
        lines.append(paint.paint("No coordinated work recorded.", fg="dim"))
    dropped = board.total - len(board.items)
    if board.truncated or dropped > 0:
        lines.append("")
        lines.append(
            paint.paint(
                f"Showing {len(board.items)} of {board.total} items; {dropped} not shown because of the display limit.",
                fg="warning",
            )
        )
    lines.append("")
    lines.extend(paint.paint(chunk, fg="dim") for chunk in textwrap.wrap(board.claim_boundary, width=88))
    return lines


def make_board_tool(
    *,
    goals: GoalLedger,
    gates: DecisionGates,
    blocks: BlockLog,
    units: VerificationLedger,
    journal: ActivityJournal,
) -> Tool:
    def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        action = clean_text(args.get("action", "overview"), limit=16).lower() or "overview"
        lane = clean_text(args.get("lane", ""), limit=24).lower()
        try:
            limit = int(args["limit"]) if args.get("limit") is not None else DEFAULT_LIMIT
        except (TypeError, ValueError):
            return ToolResult.failure("`limit` must be an integer.", reason="limit_invalid")
        if action not in ("overview", "lane"):
            return ToolResult.failure("`action` must be one of: overview, lane.", reason="action_invalid")
        if action == "lane" and lane not in BOARD_LANES:
            return ToolResult.failure(f"`lane` must be one of: {', '.join(BOARD_LANES)}.", reason="lane_invalid")
        board = build_board(
            goals=goals,
            gates=gates,
            blocks=blocks,
            units=units,
            journal=journal,
            limit=max(1, min(limit, MAX_ITEMS)),
        )
        if action == "lane":
            items = board.lane_items(lane, limit=max(1, min(limit, MAX_ITEMS)))
            if not items:
                return ToolResult.success(
                    f"Nothing is in the '{lane}' lane. Board digest {board.digest[:12]}.",
                    lane=lane,
                    digest=board.digest,
                    count=0,
                )
            paint = make_theme("mono", color=False)
            body = [f"Lane {LANE_LABELS.get(lane, lane)} ({len(items)} of {board.counts.get(lane, 0)})"]
            body.extend(
                strip_ansi(line)
                for line in table(paint, headers(), [_row(item) for item in items], glyphs="unicode")
            )
            body.append(f"Digest {board.digest[:12]} · {board.claim_boundary}")
            return ToolResult.success(
                "\n".join(body),
                lane=lane,
                digest=board.digest,
                count=len(items),
            )
        return ToolResult.success(
            "\n".join(strip_ansi(line) for line in render(board)),
            digest=board.digest,
            counts=dict(board.counts),
            truncated=board.truncated,
        )

    return Tool(
        name="board",
        description=(
            "Read-only work board over OHAR's own ledgers: what needs a decision, what is blocked, "
            "what is moving, and what can start next. action=overview renders every lane; action=lane "
            "with lane=<name> lists one lane. Nothing here runs, approves, verifies, or writes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["overview", "lane"]},
                "lane": {"type": "string", "enum": list(BOARD_LANES)},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ITEMS},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        handler=handler,
        read_only=True,
        destructive=False,
    )
