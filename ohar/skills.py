"""Skill discovery, indexing, and loading for OHAR. Stdlib only."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .types import Tool, ToolContext, ToolResult

MAX_SKILL_BYTES = 256 * 1024
DEFAULT_BODY_CHARS = 8000
DEFAULT_INDEX_CHARS = 1800
INDEX_LINE_CHARS = 80

_WORD_RE = re.compile(r"[a-z0-9]+")
_FRONTMATTER_KEYS = {"name", "description", "aliases", "license", "source"}


@dataclass(slots=True)
class Skill:
    """A single discoverable skill backed by a SKILL.md file."""

    name: str
    description: str
    path: Path
    body: str
    license: str = ""
    source: str = ""
    aliases: list[str] = field(default_factory=list)


def _clean_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def _parse_frontmatter(lines: list[str]) -> dict[str, str]:
    """Parse flat ``key: value`` frontmatter; ignore nested or unknown keys."""
    meta: dict[str, str] = {}
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or ":" not in text:
            continue
        key, value = text.split(":", 1)
        key = key.strip().lower()
        if key not in _FRONTMATTER_KEYS:
            continue
        meta[key] = _clean_scalar(value)
    return meta


def _split_aliases(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    aliases: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        alias = _clean_scalar(part)
        if not alias:
            continue
        key = alias.casefold()
        if key not in seen:
            seen.add(key)
            aliases.append(alias)
    return aliases


def _split_document(text: str) -> tuple[dict[str, str], str]:
    """Return (metadata, body), accepting optional ``---`` frontmatter."""
    text = text.lstrip("\ufeff")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return _parse_frontmatter(lines[1:index]), "".join(lines[index + 1 :])
    return {}, text


def _read_skill(directory: Path) -> Skill | None:
    path = directory / "SKILL.md"
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > MAX_SKILL_BYTES:
            return None
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    meta, body = _split_document(text)
    name = meta.get("name", "").strip() or directory.name
    aliases = [
        alias
        for alias in _split_aliases(meta.get("aliases", ""))
        if alias.casefold() != name.casefold()
    ]
    return Skill(
        name=name,
        description=meta.get("description", "").strip(),
        path=path,
        body=body.strip(),
        license=meta.get("license", "").strip(),
        source=meta.get("source", "").strip(),
        aliases=aliases,
    )


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _clip_with_notice(text: str, limit: int) -> str:
    """Truncate text to ``limit`` characters, reserving room for a notice."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    notice = f"\n\n[truncated: {len(text)} characters total]"
    if len(notice) >= limit:
        return text[:limit]
    return text[: limit - len(notice)].rstrip() + notice


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.casefold()))


def _overlap(query_tokens: set[str], target_tokens: set[str]) -> float:
    """Count matching tokens, giving partial credit for shared prefixes."""
    total = 0.0
    for query in query_tokens:
        best = 0.0
        for token in target_tokens:
            if query == token:
                best = 1.0
                break
            if len(query) >= 3 and len(token) >= 3:
                if token.startswith(query) or query.startswith(token):
                    best = max(best, 0.5)
        total += best
    return total


def _skill_header(skill: Skill) -> str:
    lines = [f"# Skill: {skill.name}"]
    if skill.description:
        lines.append(skill.description)
    lines.append(f"Source: {skill.path}")
    return "\n".join(lines) + "\n\n"


class SkillLibrary:
    """A name-indexed collection of skills discovered from one root."""

    def __init__(self, root: Path | str, skills: Iterable[Skill] = ()) -> None:
        self.root = Path(root)
        self._skills = sorted(skills, key=lambda skill: (skill.name.casefold(), str(skill.path)))
        self._lookup: dict[str, Skill] = {}
        # Names take precedence over aliases when the two collide.
        for skill in self._skills:
            self._lookup.setdefault(skill.name.casefold(), skill)
        for skill in self._skills:
            for alias in skill.aliases:
                self._lookup.setdefault(alias.casefold(), skill)

    @classmethod
    def discover(cls, root: Path | str) -> "SkillLibrary":
        """Scan immediate subdirectories of ``root`` for ``SKILL.md``."""
        root_path = Path(root)
        try:
            entries = sorted(root_path.iterdir(), key=lambda path: path.name.casefold())
        except (OSError, ValueError):
            entries = []

        discovered: list[Skill] = []
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                continue
            skill = _read_skill(entry)
            if skill is not None:
                discovered.append(skill)
        return cls(root_path, discovered)

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name or alias, case-insensitively."""
        if not isinstance(name, str) or not name.strip():
            return None
        return self._lookup.get(name.strip().casefold())

    @property
    def names(self) -> list[str]:
        return [skill.name for skill in self._skills]

    def list(self) -> list[Skill]:
        return list(self._skills)

    def search(self, query: str, limit: int = 3) -> list[Skill]:
        """Rank skills by weighted overlap across name, aliases, description."""
        query = (query or "").strip().casefold()
        try:
            limit = max(0, int(limit))
        except (TypeError, ValueError):
            limit = 3
        if not query or limit == 0:
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, str, Skill]] = []
        for skill in self._skills:
            name = skill.name.casefold()
            name_tokens = _tokens(name)
            alias_tokens: set[str] = set()
            alias_exact = False
            for alias in skill.aliases:
                folded = alias.casefold()
                alias_tokens.update(_tokens(folded))
                if query == folded:
                    alias_exact = True
            desc_tokens = _tokens(skill.description)

            score = 0.0
            if query == name:
                score += 12.0
            elif query in name:
                score += 5.0
            if alias_exact:
                score += 10.0
            score += 4.0 * _overlap(query_tokens, name_tokens)
            score += 3.0 * _overlap(query_tokens, alias_tokens)
            score += 1.0 * _overlap(query_tokens, desc_tokens)
            if score > 0:
                scored.append((score, name, skill))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [skill for _, _, skill in scored[:limit]]

    def index_text(self, max_chars: int = DEFAULT_INDEX_CHARS) -> str:
        """Compact one-line index: ``name — description`` per skill."""
        if max_chars <= 0:
            return ""
        entries: list[str] = []
        for skill in self._skills:
            description = " ".join(skill.description.split())
            line = f"{skill.name} — {description}" if description else f"{skill.name} —"
            entries.append(_truncate(line, INDEX_LINE_CHARS))
        return _truncate("\n".join(entries), max_chars)

    def body(self, name: str, max_chars: int = DEFAULT_BODY_CHARS) -> str | None:
        """Return a skill body, truncated with a notice when necessary."""
        skill = self.get(name)
        if skill is None:
            return None
        return _clip_with_notice(skill.body, max_chars)


def make_skill_tool(library: SkillLibrary, max_chars: int = DEFAULT_BODY_CHARS) -> Tool:
    """Build the read-only ``load_skill`` tool for a library."""

    def handler(args: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        raw = args.get("name", "")
        name = raw.strip() if isinstance(raw, str) else str(raw).strip()
        skill = library.get(name) if name else None
        if skill is None:
            available = ", ".join(library.names) if library.names else "(none installed)"
            requested = name or "(missing)"
            return ToolResult.failure(
                f"Unknown skill '{requested}'. Available skills: {available}",
                available=library.names,
            )

        content = _clip_with_notice(_skill_header(skill) + skill.body, max_chars)
        return ToolResult.success(content, skill=skill.name, path=str(skill.path))

    return Tool(
        name="load_skill",
        description=(
            "Load the full instructions for a named skill. Use it when a task "
            "matches a skill description or the user names one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name or alias to load.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=handler,
        read_only=True,
    )


def skill_prompt(library: SkillLibrary, name: str) -> str | None:
    """Return a truncated header plus body for slash-command injection."""
    skill = library.get(name)
    if skill is None:
        return None
    return _clip_with_notice(_skill_header(skill) + skill.body, DEFAULT_BODY_CHARS)


__all__ = ["Skill", "SkillLibrary", "make_skill_tool", "skill_prompt"]
