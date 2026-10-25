"""Parallel read-only subagents exposed to the main agent as the `task` tool."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .agent import Agent
from .balancer import LaneManager
from .prompts import build_subagent_prompt
from .skills import SkillLibrary, make_skill_tool
from .tools import Toolbox, default_tools
from .types import (
    AgentHooks,
    Config,
    Tool,
    ToolContext,
    ToolResult,
)

_TITLE_LIMIT = 80
_PROMPT_LIMIT = 20_000
_REPORT_LIMIT = 12_000
_TOTAL_LIMIT = 48_000


def _clean_title(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        text = " ".join(value.strip().splitlines()[0].split())
        return text[:_TITLE_LIMIT]
    return fallback


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]}\n… [truncated {len(text) - limit} chars] …\n{text[-tail:]}"


def make_task_tool(
    *,
    config: Config,
    lanes: LaneManager,
    workspace: Path,
    hooks: AgentHooks,
    skill_library: SkillLibrary | None = None,
    max_parallel: int | None = None,
    extra_tools: list[Tool] | None = None,
) -> Tool:
    limit = max(1, min(config.subagents_max, max_parallel or config.subagents_max))
    default_lane = "subagents" if "subagents" in config.lanes else config.default_lane
    skill_index = skill_library.index_text(1500) if skill_library else ""
    skill_tool = make_skill_tool(skill_library) if skill_library else None
    shared_extra = [tool for tool in (extra_tools or []) if tool.read_only]

    def dedupe(tools: list[Tool]) -> list[Tool]:
        seen: set[str] = set()
        ordered: list[Tool] = []
        for tool in tools:
            if tool.name in seen:
                continue
            seen.add(tool.name)
            ordered.append(tool)
        return ordered

    def emit(event: dict[str, Any]) -> None:
        try:
            hooks.subagent(event)
        except Exception:
            pass

    def run_one(
        index: int,
        task: dict[str, Any],
        lane: str,
        cancel: threading.Event,
        stopped: threading.Event,
        results: list[dict[str, Any]],
    ) -> None:
        title = task["title"]
        task_id = str(index + 1)
        started = time.monotonic()
        if stopped.is_set():
            return
        try:
            read_ctx = ToolContext(
                workspace=workspace,
                approval_mode="read-only",
                read_only=True,
                shell_timeout=config.shell_timeout,
            )
            tools = [tool for tool in default_tools(read_ctx) if tool.read_only]
            if skill_tool is not None:
                tools.append(skill_tool)
            tools.extend(shared_extra)
            tools = dedupe(tools)
            toolbox = Toolbox(read_ctx, tools)

            def on_tool_start(name: str, _args: dict[str, Any]) -> None:
                if not stopped.is_set():
                    emit({"type": "tool", "id": task_id, "name": name})

            worker_hooks = AgentHooks(tool_start=on_tool_start)
            agent = Agent(
                config,
                lanes,
                toolbox,
                system_prompt=build_subagent_prompt(
                    config,
                    workspace,
                    title=title,
                    skill_index=skill_index,
                    tool_names=toolbox.names,
                ),
                lane=lane,
                hooks=worker_hooks,
                max_turns=min(config.max_turns, 12),
                cancel=cancel,
            )
            result = agent.run(task["prompt"])
            seconds = time.monotonic() - started
            results[index] = {
                "status": "completed",
                "title": title,
                "lane": result.lane,
                "model": result.model,
                "seconds": seconds,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "text": result.text,
            }
            if not stopped.is_set():
                emit(
                    {
                        "type": "done",
                        "id": task_id,
                        "title": title,
                        "seconds": seconds,
                        "chars": len(result.text),
                    }
                )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                cancel.set()
            seconds = time.monotonic() - started
            results[index] = {
                "status": "failed",
                "title": title,
                "lane": lane,
                "model": "",
                "seconds": seconds,
                "input_tokens": 0,
                "output_tokens": 0,
                "text": "",
                "error": str(exc) or type(exc).__name__,
            }
            if not stopped.is_set():
                emit(
                    {
                        "type": "error",
                        "id": task_id,
                        "title": title,
                        "error": str(exc) or type(exc).__name__,
                    }
                )

    def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_tasks = args.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            return ToolResult.failure("task: 'tasks' must be a non-empty array.")
        if len(raw_tasks) > limit:
            return ToolResult.failure(
                f"task: at most {limit} subagents can run at once; got {len(raw_tasks)}."
            )
        tasks: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_tasks):
            if not isinstance(raw, dict):
                return ToolResult.failure(f"task: tasks[{index}] must be an object.")
            prompt = raw.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return ToolResult.failure(f"task: tasks[{index}].prompt must be a non-empty string.")
            if len(prompt) > _PROMPT_LIMIT:
                return ToolResult.failure(
                    f"task: tasks[{index}].prompt is too long (max {_PROMPT_LIMIT} chars)."
                )
            lane = raw.get("lane")
            if lane is None:
                lane = default_lane
            if not isinstance(lane, str) or lane not in config.lanes:
                return ToolResult.failure(
                    f"task: tasks[{index}] names unknown lane '{lane}'. "
                    f"Available: {', '.join(sorted(config.lanes))}."
                )
            title = _clean_title(raw.get("title"), f"subagent {index + 1}")
            tasks.append({"title": title, "prompt": prompt.strip(), "lane": lane})

        started = time.monotonic()
        deadline = started + max(1.0, config.subagent_timeout)
        results: list[dict[str, Any]] = [{} for _ in tasks]
        cancels = [threading.Event() for _ in tasks]
        stopped = [threading.Event() for _ in tasks]
        threads: list[threading.Thread] = []

        for index, task in enumerate(tasks):
            lane_members = lanes.members(task["lane"])
            model_hint = lane_members[0].id if lane_members else task["lane"]
            emit(
                {
                    "type": "start",
                    "id": str(index + 1),
                    "title": task["title"],
                    "lane": task["lane"],
                    "model": model_hint,
                }
            )
            thread = threading.Thread(
                target=run_one,
                args=(index, task, task["lane"], cancels[index], stopped[index], results),
                name=f"ohar-subagent-{index + 1}",
                daemon=True,
            )
            threads.append(thread)
            thread.start()

        try:
            for index, thread in enumerate(threads):
                remaining = max(0.0, deadline - time.monotonic())
                thread.join(timeout=remaining)
                if thread.is_alive():
                    stopped[index].set()
                    cancels[index].set()
                    results[index] = {
                        "status": "timed_out",
                        "title": tasks[index]["title"],
                        "lane": tasks[index]["lane"],
                        "model": "",
                        "seconds": time.monotonic() - started,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "text": "",
                        "error": f"timed out after {config.subagent_timeout:g}s",
                    }
                    emit(
                        {
                            "type": "error",
                            "id": str(index + 1),
                            "title": tasks[index]["title"],
                            "error": "timed out",
                        }
                    )
        except BaseException:
            for stop, cancel in zip(stopped, cancels):
                stop.set()
                cancel.set()
            raise

        elapsed = time.monotonic() - started
        sections: list[str] = []
        completed = 0
        for index, result in enumerate(results):
            if not result:
                result = {
                    "status": "failed",
                    "title": tasks[index]["title"],
                    "lane": tasks[index]["lane"],
                    "error": "subagent produced no result",
                }
            status = str(result.get("status", "failed"))
            if status == "completed":
                completed += 1
            header = f"## {index + 1}. {result.get('title', tasks[index]['title'])}"
            details = [f"status: {status}"]
            if result.get("model"):
                details.append(f"model: {result['model']}")
            if result.get("lane"):
                details.append(f"lane: {result['lane']}")
            if "seconds" in result:
                details.append(f"time: {float(result['seconds']):.1f}s")
            tokens_in = int(result.get("input_tokens", 0) or 0)
            tokens_out = int(result.get("output_tokens", 0) or 0)
            if tokens_in or tokens_out:
                details.append(f"tokens: {tokens_in} in / {tokens_out} out")
            body = str(result.get("text") or result.get("error") or "").strip()
            sections.append(
                header + "\n" + " · ".join(details) + "\n\n" + (_truncate(body, _REPORT_LIMIT) or "(empty)")
            )
        content = _truncate("\n\n".join(sections), _TOTAL_LIMIT)
        failed = len(tasks) - completed
        summary = f"{completed}/{len(tasks)} subagents completed in {elapsed:.1f}s"
        if failed:
            summary += f" · {failed} failed or timed out"
        content = summary + "\n\n" + content
        meta = {
            "count": len(tasks),
            "completed": completed,
            "failed": failed,
            "seconds": elapsed,
        }
        if completed:
            return ToolResult.success(content, **meta)
        return ToolResult.failure(content, **meta)

    return Tool(
        name="task",
        description=(
            f"Run up to {limit} independent READ-ONLY research subagents in parallel. "
            "Each task gets its own context and can inspect files, search code, load skills, "
            "and use configured web tools, but cannot modify files or run shell commands. "
            "Use this for broad independent investigation, not for edits. Provide a clear "
            "prompt and optional title/lane per task."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "Independent research tasks to run concurrently.",
                    "minItems": 1,
                    "maxItems": limit,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short label for this subagent.",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Self-contained assignment for the subagent.",
                            },
                            "lane": {
                                "type": "string",
                                "description": "Optional configured lane name.",
                            },
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
        handler=handler,
        read_only=True,
    )
