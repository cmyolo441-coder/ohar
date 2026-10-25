"""Provenance gate plus the artifact manifest for one OHAR session handoff.

Two halves share this module because they answer the same two questions a
terminal user asks when a session outgrows its context window: *what produced
this* and *what can I copy out of it*.

The gate prints every field it knows and degrades the rest to the literal
``unknown`` rather than inventing a value or silently dropping a row: an absent
model is a fact worth printing. Warnings name the gap, they never refuse the
answer. Fences are derived from the fenced content, so an embedded block still
nests.

The manifest is an index. ``listing()`` hands back ids, labels and character
counts only; bodies come back one selected artifact at a time, so a picker
cannot spill a whole session into the terminal and copying never advances
session state.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ohar.sessions import Session
from ohar.terminal import display_width, pad, sanitize_text

from .evidence import Claim
from .records import clean_text
from .state import SessionState

__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "HANDOFF_SCHEMA_VERSION",
    "UNKNOWN",
    "MAX_FIELD_CHARS",
    "MAX_ARTIFACT_CHARS",
    "MAX_PROMPT_PREVIEW_CHARS",
    "ARTIFACT_IDS",
    "EVIDENCE_CLASSES",
    "HandoffError",
    "fence_marker",
    "digest_prompt",
    "prompt_reference",
    "build_provenance",
    "provenance_warnings",
    "render_provenance",
    "attach_provenance",
    "Artifact",
    "Handoff",
    "build_handoff",
    "select_artifact",
    "artifact_text",
    "resume_prompt",
]

PROVENANCE_SCHEMA_VERSION = "ohar_provenance/v1"
HANDOFF_SCHEMA_VERSION = "ohar_handoff/v1"
UNKNOWN = "unknown"
MAX_FIELD_CHARS = 96
MAX_ARTIFACT_CHARS = 1600
MAX_PROMPT_PREVIEW_CHARS = 320

ARTIFACT_IDS: tuple[str, ...] = (
    "goal",
    "current_state",
    "next_steps",
    "open_questions",
    "evidence",
    "resume_prompt",
)

EVIDENCE_CLASSES: tuple[str, ...] = ("observed", "model_reported", "planned", "unknown")

ARTIFACT_LABELS: dict[str, str] = {
    "goal": "Goal",
    "current_state": "Current state",
    "next_steps": "Next steps",
    "open_questions": "Open questions",
    "evidence": "Evidence",
    "resume_prompt": "Resume prompt",
}

# Identity, then model, then status: the same reading order as the status board.
# The lane rides inside the MODEL field as one visual unit, ``main (a/b)``.
_FIELD_ORDER: tuple[tuple[str, str], ...] = (
    ("SKILL", "skill"),
    ("MODEL", "model"),
    ("STATUS", "status"),
    ("TASK", "task"),
    ("PROMPT", "prompt"),
    ("EVIDENCE", "evidence_class"),
)

_RESUME_SECTIONS: tuple[str, ...] = (
    "goal",
    "current_state",
    "next_steps",
    "open_questions",
    "evidence",
)

_NO_ROUTE = "no model route"
_DIGEST_PREFIX = 12
_MAX_LIST_ITEMS = 12
_MAX_PREVIEW_CHARS = 200

WARNING_MISSING_MODEL_LABEL = "handoff_missing_model_label"
WARNING_EMPTY_PARENTHESES = "handoff_empty_parentheses"
WARNING_UNRESOLVED_MODEL_ROUTE = "handoff_unresolved_model_route"
WARNING_MISSING_STATUS = "handoff_missing_status"
WARNING_MISSING_PROMPT_REFERENCE = "handoff_missing_prompt_reference"
WARNING_UNKNOWN_EVIDENCE_CLASS = "handoff_unknown_evidence_class"


class HandoffError(ValueError):
    """An artifact request this handoff cannot satisfy."""


def fence_marker(text: str) -> str:
    """Backtick fence for embedding ``text``: longest run plus one, minimum three."""
    longest = 0
    run = 0
    for character in text or "":
        if character == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return "`" * max(3, longest + 1)


def digest_prompt(text: str) -> str:
    """sha256 of the prompt as given, so a digest can be compared across turns."""
    raw = text if isinstance(text, str) else str(text or "")
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def prompt_reference(digest: str, chars: int | None) -> str:
    """``sha256:533d406a4863 · 412 chars`` — a reference, never the prompt."""
    folded = "".join(character for character in str(digest or "").strip() if character.isalnum())
    if not folded:
        return UNKNOWN
    reference = f"sha256:{folded[:_DIGEST_PREFIX]}"
    if isinstance(chars, int) and not isinstance(chars, bool) and chars >= 0:
        return f"{reference} · {chars} chars"
    return reference


def build_provenance(
    *,
    model: str,
    lane: str,
    skill: str = "",
    status: str = "",
    task: str = "",
    prompt: str = "",
    prompt_digest: str = "",
    evidence_class: str = "observed",
) -> dict[str, Any]:
    """Project one response's execution provenance, degrading each gap to ``unknown``.

    ``prompt`` is the caller's decision: pass it and a bounded preview is kept
    alongside the digest reference; pass only ``prompt_digest`` and the raw text
    never reaches this payload at all.
    """
    text = prompt if isinstance(prompt, str) else str(prompt or "")
    digest = str(prompt_digest or "").strip()
    chars: int | None = None
    if text.strip():
        digest = digest_prompt(text)
        chars = len(text)

    route = _bounded(model)
    owner = _bounded(lane)
    fields: dict[str, str] = {
        "skill": _bounded(skill) or UNKNOWN,
        "model": _model_field(owner, route),
        "status": _bounded(status) or UNKNOWN,
        "task": _bounded(task) or UNKNOWN,
        "prompt": prompt_reference(digest, chars) if digest or text.strip() else UNKNOWN,
        "evidence_class": _evidence_class(evidence_class),
    }
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "fields": fields,
        "field_order": [key for _, key in _FIELD_ORDER],
        # Whether a model route was resolved at all, which the rendered MODEL
        # row cannot express: a known lane with no route and no lane at all
        # otherwise read as the same line.
        "model_route_resolved": bool(route),
    }
    if text.strip():
        payload["prompt_preview"] = _prompt_preview(text)
    payload["warnings"] = provenance_warnings(payload)
    return payload


def provenance_warnings(payload: Mapping[str, Any]) -> list[str]:
    """Name the shapes that under-disclose, without refusing to render them."""
    fields = _fields_of(payload)
    model = fields.get("model", "")
    warnings: list[str] = []
    if "model" in fields and (not model or model == UNKNOWN):
        warnings.append(WARNING_MISSING_MODEL_LABEL)
    if "()" in model:
        warnings.append(WARNING_EMPTY_PARENTHESES)
    if "model" in fields and payload.get("model_route_resolved") is False:
        warnings.append(WARNING_UNRESOLVED_MODEL_ROUTE)
    if fields.get("status", UNKNOWN) == UNKNOWN:
        warnings.append(WARNING_MISSING_STATUS)
    if fields.get("prompt", UNKNOWN) == UNKNOWN:
        warnings.append(WARNING_MISSING_PROMPT_REFERENCE)
    if fields.get("evidence_class", UNKNOWN) == UNKNOWN:
        warnings.append(WARNING_UNKNOWN_EVIDENCE_CLASS)
    return sorted(set(warnings))


def render_provenance(payload: Mapping[str, Any], *, plain: bool = False) -> str:
    """One line for a plain surface, or an aligned block inside a content fence."""
    fields = _fields_of(payload)
    if not fields:
        return ""
    labels = {key: label for label, key in _FIELD_ORDER}
    if plain:
        return " — ".join(
            f"{labels.get(key, key.replace('_', ' ')).lower()}: {fields[key]}"
            for key in _ordered_keys(payload, fields)
        )

    order = _ordered_keys(payload, fields)
    width = max(display_width(labels.get(key, key.replace("_", " "))) for key in order)
    rows = [
        f"{pad(labels.get(key, key.replace('_', ' ')), width)}  {fields[key]}" for key in order
    ]
    fence = fence_marker("\n".join(rows))
    lines = [fence, *rows, fence]

    preview = payload.get("prompt_preview")
    if isinstance(preview, str) and preview.strip():
        marker = fence_marker(preview)
        model = fields.get("model", UNKNOWN)
        title = "HANDOFF PROMPT" if model == UNKNOWN else f"HANDOFF PROMPT — {model}"
        lines.extend(["", title, marker, preview, marker])
    return "\n".join(lines)


def attach_provenance(body: str, payload: Mapping[str, Any], *, plain: bool = False) -> str:
    """Place the gate around ``body``: a plain line goes below the answer."""
    block = render_provenance(payload, plain=plain)
    text = body if isinstance(body, str) else str(body or "")
    if not block:
        return text
    if not text.strip():
        return block
    if plain:
        return f"{text.rstrip()}\n\n{block}"
    return f"{block}\n\n{text.lstrip()}"


@dataclass(frozen=True, slots=True)
class Artifact:
    """One copyable block of a handoff, already bounded and already sanitized."""

    artifact_id: str
    label: str
    text: str
    truncated: bool

    def entry(self) -> dict[str, Any]:
        """The index row for this artifact: no body, so a listing cannot spill it."""
        return {
            "artifact_id": self.artifact_id,
            "label": self.label,
            "chars": len(self.text),
            "truncated": self.truncated,
        }


@dataclass(slots=True)
class Handoff:
    """A session's copyable artifacts plus the provenance that produced them."""

    session_id: str
    artifacts: list[Artifact] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def listing(self) -> list[dict[str, Any]]:
        return [artifact.entry() for artifact in self.artifacts]


def build_handoff(
    *,
    state: SessionState,
    messages: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any] | None = None,
    claims: Sequence[Claim] = (),
    model: str,
    lane: str,
    skill: str = "",
) -> Handoff:
    """Project a session into artifacts without touching session state or records."""
    entries = _entries(messages)
    user_text = _last_text(entries, "user")
    assistant_text = _last_text(entries, "assistant")
    views = [_claim_view(claim) for claim in _sequence(claims)]
    evidence_class = _derive_evidence_class(views, assistant_text)

    texts = {
        "goal": _goal_text(state),
        "current_state": _current_state_text(state, user_text, assistant_text),
        "next_steps": _next_steps_text(state, plan),
        "open_questions": _open_questions_text(state, views),
        "evidence": _evidence_text(views),
    }
    resume = _compose_resume(state.session_id, texts, limit=MAX_ARTIFACT_CHARS)

    artifacts: list[Artifact] = []
    for artifact_id in ARTIFACT_IDS:
        if artifact_id == "resume_prompt":
            raw = resume
        else:
            raw = texts.get(artifact_id, UNKNOWN)
        text, truncated = _bound(raw)
        artifacts.append(
            Artifact(
                artifact_id=artifact_id,
                label=ARTIFACT_LABELS.get(artifact_id, artifact_id),
                text=text,
                truncated=truncated,
            )
        )

    provenance = build_provenance(
        model=model or state.model,
        lane=lane or state.lane,
        skill=skill or state.skill,
        status=state.status,
        task=state.goal,
        prompt=user_text,
        evidence_class=evidence_class,
    )
    provenance["handoff_schema_version"] = HANDOFF_SCHEMA_VERSION
    return Handoff(session_id=state.session_id, artifacts=artifacts, provenance=provenance)


def select_artifact(handoff: Handoff, artifact_id: str) -> Artifact:
    """Find one artifact by id or label; unknown ids name the available ones."""
    wanted = " ".join(str(artifact_id or "").split()).casefold()
    available = ", ".join(artifact.artifact_id for artifact in handoff.artifacts) or "(none)"
    if not wanted:
        raise HandoffError(f"No artifact selected. Available: {available}.")
    for artifact in handoff.artifacts:
        if wanted in (artifact.artifact_id.casefold(), artifact.label.casefold()):
            return artifact
    raise HandoffError(f"Unknown artifact '{wanted}'. Available: {available}.")


def artifact_text(handoff: Handoff, artifact_id: str, *, fence: bool = True) -> str:
    """The body of exactly one artifact, fenced by default so it copies cleanly."""
    artifact = select_artifact(handoff, artifact_id)
    if not fence:
        return artifact.text
    marker = fence_marker(artifact.text)
    return "\n".join([f"{artifact.label}:", marker, artifact.text, marker])


def resume_prompt(handoff: Handoff, *, limit: int = 4000) -> str:
    """The copyable prompt that continues this session in a fresh context window."""
    texts = {
        artifact.artifact_id: artifact.text
        for artifact in handoff.artifacts
        if artifact.artifact_id in _RESUME_SECTIONS
    }
    return _compose_resume(handoff.session_id, texts, limit=limit)


def _compose_resume(session_id: str, texts: Mapping[str, str], *, limit: int) -> str:
    try:
        ceiling = max(0, int(limit))
    except (TypeError, ValueError):
        ceiling = 0
    if ceiling == 0:
        return ""
    header = f"Continue OHAR session {session_id.strip() or UNKNOWN}."
    footer = (
        "Pick up at the first next step. Do not restate the plan or re-run "
        "completed work, and claim only what you observe."
    )
    blocks = [header]
    for section in _RESUME_SECTIONS:
        body = (texts.get(section) or UNKNOWN).strip() or UNKNOWN
        label = ARTIFACT_LABELS.get(section, section)
        blocks.append(f"{label}:\n{body}")
    blocks.append(footer)
    text = "\n\n".join(blocks)
    if len(text) <= ceiling:
        return text
    marker = f"\n… [truncated, {len(text)} chars total]"
    if ceiling <= len(marker):
        # A ceiling this tight cannot hold a marker as well as content.
        return text[:ceiling]
    return text[: ceiling - len(marker)].rstrip() + marker


def _prompt_preview(text: str) -> str:
    """The head of the prompt, bounded and marked, for a surface that opted in."""
    clean = sanitize_text(text).strip()
    if len(clean) <= MAX_PROMPT_PREVIEW_CHARS:
        return clean
    marker = f" … [truncated, {len(clean)} chars total]"
    return clean[: max(0, MAX_PROMPT_PREVIEW_CHARS - len(marker))].rstrip() + marker


def _derive_evidence_class(views: Sequence[tuple[str, str, str, tuple[str, ...]]], assistant_text: str) -> str:
    states = [view[1] for view in views]
    if any(state == "observed" for state in states):
        return "observed"
    if states:
        return "planned"
    return "model_reported" if assistant_text.strip() else "unknown"


def _goal_text(state: SessionState) -> str:
    return state.goal or UNKNOWN


def _current_state_text(state: SessionState, user_text: str, assistant_text: str) -> str:
    lines = [
        f"status: {state.status or UNKNOWN}",
        f"decision: {state.decision or 'none recorded'}",
        f"workspace: {state.workspace or UNKNOWN}",
        f"next action: {state.next_action or UNKNOWN}",
        f"record revision: {state.revision}",
        f"last request: {' '.join(user_text.split())[:_MAX_PREVIEW_CHARS] or UNKNOWN}",
        f"last response: {' '.join(assistant_text.split())[:_MAX_PREVIEW_CHARS] or UNKNOWN}",
    ]
    return "\n".join(lines)


def _next_steps_text(state: SessionState, plan: Mapping[str, Any] | None) -> str:
    open_steps = [
        (status, text)
        for status, text in _plan_steps(plan)
        if status in ("pending", "in_progress")
    ]
    lines = [f"{index}. [{status}] {text}" for index, (status, text) in enumerate(open_steps, 1)]
    if state.next_action:
        lines.append(f"session next action: {state.next_action}")
    if state.status == "blocked":
        lines.append("session is blocked; resolve the blocker before any next step")
    if not lines:
        lines.append("confirm the goal, then present a plan")
    return "\n".join(lines[:_MAX_LIST_ITEMS])


def _open_questions_text(state: SessionState, views: Sequence[tuple[str, str, str, tuple[str, ...]]]) -> str:
    lines = [
        f"- {name or UNKNOWN}: {status}{' — ' + reason if reason else ''}"
        for name, status, reason, _ in views
        if status in ("pending", "refuted")
    ]
    if state.status in ("needs_review", "blocked"):
        lines.append(f"- session status '{state.status}' still needs a decision")
    return "\n".join(lines[:_MAX_LIST_ITEMS]) or "(none recorded)"


def _evidence_text(views: Sequence[tuple[str, str, str, tuple[str, ...]]]) -> str:
    lines = []
    for name, status, reason, refs in views:
        detail = f" refs: {', '.join(refs)}" if refs else ""
        lines.append(f"- {name or UNKNOWN}: {status or UNKNOWN}{detail}")
    if not lines:
        return "(no recorded evidence)"
    counts: dict[str, int] = {}
    for _, status, _, _ in views:
        counts[status or UNKNOWN] = counts.get(status or UNKNOWN, 0) + 1
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    return "\n".join([f"claims: {summary}", *lines[:_MAX_LIST_ITEMS]])


def _plan_steps(plan: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    if not isinstance(plan, Mapping):
        return []
    steps = plan.get("steps")
    if isinstance(steps, (str, bytes, Mapping)) or not isinstance(steps, Sequence):
        return []
    out: list[tuple[str, str]] = []
    for entry in steps[:_MAX_LIST_ITEMS]:
        if isinstance(entry, Mapping):
            text = " ".join(str(entry.get("step", "") or "").split())
            status = " ".join(str(entry.get("status", "") or "").split())
            if text:
                out.append((status or "pending", text))
        elif isinstance(entry, str) and entry.strip():
            out.append(("pending", " ".join(entry.split())))
    return out


def _entries(messages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    source: Any = messages.messages if isinstance(messages, Session) else messages
    if isinstance(source, (str, bytes, Mapping)) or not isinstance(source, Sequence):
        return []
    return [message for message in source if isinstance(message, Mapping)]


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        return []
    return list(value)


def _last_text(entries: Sequence[Mapping[str, Any]], role: str) -> str:
    for message in reversed(list(entries)):
        if str(message.get("role", "")) != role:
            continue
        text = _content_text(message.get("content"))
        if text.strip():
            return text
    return ""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, (bytes, bytearray)):
        return ""
    if isinstance(content, Sequence):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return " ".join(chunks)
    return ""


def _claim_view(claim: Any) -> tuple[str, str, str, tuple[str, ...]]:
    if isinstance(claim, Mapping):
        name = claim.get("claim")
        status = claim.get("state")
        reason = claim.get("reason")
        refs = claim.get("refs")
    else:
        name = getattr(claim, "claim", "")
        status = getattr(claim, "state", "")
        reason = getattr(claim, "reason", "")
        refs = getattr(claim, "refs", ())
    ref_items = refs if isinstance(refs, (list, tuple)) else []
    return (
        " ".join(str(name or "").split()),
        " ".join(str(status or "").split()),
        " ".join(str(reason or "").split()),
        tuple(" ".join(str(item).split()) for item in ref_items if str(item).strip()),
    )


def _fields_of(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = payload.get("fields")
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items() if str(value or "")}


def _ordered_keys(payload: Mapping[str, Any], fields: Mapping[str, str]) -> list[str]:
    declared = payload.get("field_order")
    order = [str(key) for key in declared] if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)) else []
    keys = [key for key in order if key in fields]
    # A field this version does not know still renders, at the end.
    keys.extend(key for key in fields if key not in keys)
    return keys


def _model_field(owner: str, route: str) -> str:
    """``main (anthropic/sonnet)``; a lane with no route says so, never ``()``."""
    if not owner:
        return route or UNKNOWN
    return _bounded(f"{owner} ({route or _NO_ROUTE})") or UNKNOWN


def _evidence_class(value: object) -> str:
    text = " ".join(str(value or "").split()).casefold()
    return text if text in EVIDENCE_CLASSES else UNKNOWN


def _bounded(value: object) -> str:
    """Single line, length-capped, control characters gone: one broken row is enough."""
    text = clean_text(value, limit=MAX_FIELD_CHARS, single_line=True)
    return " ".join(sanitize_text(text, keep_newlines=False).split())


def _bound(text: str, limit: int = MAX_ARTIFACT_CHARS) -> tuple[str, bool]:
    clean = sanitize_text(text).strip("\n")
    if len(clean) <= limit:
        return clean, False
    notice = f"\n… [truncated, {len(clean)} characters total]"
    if len(notice) >= limit:
        return clean[:limit], True
    return clean[: limit - len(notice)].rstrip() + notice, True
