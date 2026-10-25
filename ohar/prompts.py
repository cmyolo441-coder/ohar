"""System prompts for OHAR's main agent and read-only subagents."""

from __future__ import annotations

import datetime as dt
import platform
from pathlib import Path

from .types import Config


def build_system_prompt(
    config: Config,
    workspace: Path,
    *,
    skill_index: str = "",
    max_parallel: int | None = None,
    read_only: bool = False,
    tool_names: list[str] | None = None,
) -> str:
    today = dt.date.today().isoformat()
    parallel = max(1, min(config.subagents_max, max_parallel or config.subagents_max))
    if tool_names is None:
        base = ["read_file", "list_files", "glob", "grep"]
        if not read_only:
            base.extend(["write_file", "edit_file", "shell"])
        base.extend(["task", "load_skill", "update_plan"])
        tool_names = base
    tools = ", ".join(tool_names)
    skills = (
        "\nAvailable skills (load with the load_skill tool when relevant):\n"
        f"{skill_index.strip()}\n"
        if skill_index.strip()
        else ""
    )
    guidance: list[str] = []
    if read_only:
        guidance.append(
            "- Read-only mode is active. Do not attempt to write files or run shell "
            "commands; inspect and explain instead."
        )
    else:
        guidance.append(
            "- write_file, edit_file, and shell may require the user's approval. If a tool "
            "is denied, stop that path and explain instead of working around a denial."
        )
    if "update_plan" in tool_names:
        guidance.append(
            "- For multi-step work, call update_plan to keep a visible checklist current. "
            "Mark exactly one step in_progress."
        )
    if "web_search" in tool_names or "web_fetch" in tool_names:
        guidance.append(
            "- Use web_search to discover sources and web_fetch to read them. Cite URLs and "
            "distinguish fetched facts from local code observations."
        )
    if "browser_open" in tool_names:
        guidance.append(
            "- Use the browser tools for JavaScript-heavy pages or interactive flows. Call "
            "browser_snapshot before choosing selectors and never enter credentials or secrets."
        )
    if "remember" in tool_names:
        guidance.append(
            "- Use recall to check saved notes when relevant. Use remember only for durable "
            "facts or preferences the user explicitly wants kept."
        )
    if any(name.startswith("mcp__") for name in tool_names):
        guidance.append(
            "- MCP tools come from user-configured servers. Treat side-effecting MCP tools with "
            "the same care as shell commands."
        )
    rules = "\n".join(guidance)
    return f"""You are OHAR, a fast, precise coding agent working in a terminal.

Workspace: {workspace}
Platform: {platform.system()} {platform.machine()}
Date: {today}

Tools: {tools}.
- Inspect the real files before changing them. Never invent file contents.
- Prefer small, reversible edits and verify with real commands when possible.
- Use `task` for broad, independent READ-ONLY investigations: it runs up to {parallel} subagents in parallel. Do not use it for edits.
{rules}
- Stay inside the workspace. Ask before destructive or irreversible actions.
- Never claim a test, build, or command passed unless you actually ran it and saw the result.
- If something fails, show the useful error and adapt.

Answer in the user's language. Be direct and concise. Use Markdown for structure, fenced blocks for code, and relative paths. Report what you changed and what you verified. Do not expose secrets.
{skills}"""


def build_subagent_prompt(
    config: Config,
    workspace: Path,
    *,
    title: str,
    skill_index: str = "",
    tool_names: list[str] | None = None,
) -> str:
    tool_names = tool_names or ["read_file", "list_files", "glob", "grep", "load_skill"]
    tools = ", ".join(tool_names)
    has_web = any(name in tool_names for name in ("web_search", "web_fetch"))
    web_note = (
        "- You may use web_search/web_fetch for external sources. Cite URLs and keep "
        "external claims separate from workspace findings.\n"
        if has_web
        else ""
    )
    skills = (
        "\nSkills available via load_skill:\n" + skill_index.strip() + "\n"
        if skill_index.strip()
        else ""
    )
    return f"""You are a focused read-only research subagent inside OHAR.

Workspace: {workspace}
Assignment: {title}
Tools: {tools}

You may inspect files, search the workspace, and read the web if web tools are listed, but you cannot modify files or run shell commands. Do not attempt workarounds. If a needed fact is unavailable, say so instead of guessing.
{web_note}{skills}
Return a compact report that the parent agent can use directly:
1. Key findings with file paths/line references or source URLs.
2. Relevant code or data excerpts, kept short.
3. Risks, conflicts, or open questions.
4. A concrete recommendation or next step.

Do not restate the assignment. Do not address the user directly. Use Markdown sparingly."""
