"""Streaming agent loop with lane failover, tools, and context trimming."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from .balancer import LaneManager
from .tools import Toolbox
from .types import (
    AgentError,
    AgentHooks,
    Config,
    ConfigError,
    Member,
    ProviderError,
    StreamEvent,
)


@dataclass(slots=True)
class TurnResult:
    text: str
    turns: int
    model: str
    lane: str
    input_tokens: int
    output_tokens: int
    elapsed: float


class Agent:
    """One conversation bound to a lane, toolset, and hook surface."""

    def __init__(
        self,
        config: Config,
        lanes: LaneManager,
        toolbox: Toolbox,
        *,
        system_prompt: str,
        lane: str | None = None,
        hooks: AgentHooks | None = None,
        max_turns: int | None = None,
        cancel: threading.Event | None = None,
        preferred_member: str | None = None,
    ) -> None:
        self.config = config
        self.lanes = lanes
        self.toolbox = toolbox
        self.system_prompt = system_prompt
        self.lane = lane or config.default_lane
        self.hooks = hooks or AgentHooks()
        self.max_turns = max(1, int(max_turns or config.max_turns))
        self.cancel = cancel or threading.Event()
        self.preferred_member = preferred_member
        self.messages: list[dict[str, Any]] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self._trimmed_notice = False
        self._assert_lane()

    def _assert_lane(self) -> None:
        try:
            members = self.lanes.members(self.lane)
        except ConfigError:
            raise
        if not members:
            raise ConfigError(f"Lane '{self.lane}' has no members.")

    def set_lane(self, lane: str) -> None:
        previous = self.lane
        self.lane = lane
        try:
            self._assert_lane()
        except ConfigError:
            self.lane = previous
            raise
        if self.preferred_member and not any(
            member.id == self.preferred_member for member in self.lanes.members(lane)
        ):
            self.preferred_member = None

    def set_preferred_member(self, member_id: str | None) -> None:
        if member_id is not None and not any(
            member.id == member_id for member in self.lanes.members(self.lane)
        ):
            raise ConfigError(f"'{member_id}' is not a member of lane '{self.lane}'.")
        self.preferred_member = member_id

    def set_messages(self, messages: list[dict[str, Any]] | None) -> None:
        self.messages = [
            dict(message)
            for message in (messages or [])
            if isinstance(message, dict) and message.get("role") != "system"
        ]
        while (
            self.messages
            and self.messages[-1].get("role") == "assistant"
            and self.messages[-1].get("tool_calls")
        ):
            self.messages.pop()

    def reset(self) -> None:
        self.messages = []
        self.input_tokens = 0
        self.output_tokens = 0

    def run(self, user_text: str) -> TurnResult:
        if not isinstance(user_text, str) or not user_text.strip():
            raise AgentError("The prompt is empty.")
        self.messages.append({"role": "user", "content": user_text.strip()})
        self._trimmed_notice = False
        started = time.monotonic()
        turn_input = 0
        turn_output = 0
        turns = 0
        last_text = ""
        used_model = self.preferred_member or ""
        used_lane = self.lane

        for _ in range(self.max_turns):
            self._raise_if_cancelled()
            turns += 1
            request = self._context_messages()
            member, text, calls, usage, finish = self._stream_turn(request)
            used_model = member.id
            used_lane = member.lane

            if usage:
                turn_input += usage.get("input_tokens", 0)
                turn_output += usage.get("output_tokens", 0)
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                self.hooks.usage(
                    {
                        "input_tokens": turn_input,
                        "output_tokens": turn_output,
                        "total_tokens": turn_input + turn_output,
                    }
                )
            last_text = text
            if finish == "length":
                self.hooks.notice(
                    "The model hit its output limit; the response may be incomplete.",
                    "warning",
                )

            assistant: dict[str, Any] = {"role": "assistant", "content": text or ""}
            if calls:
                assistant["tool_calls"] = calls
            self.messages.append(assistant)

            if not calls:
                return self._finish(started, turns, last_text, used_model, used_lane, turn_input, turn_output)

            for call in calls:
                self._raise_if_cancelled()
                name = str(call["function"]["name"])
                raw_arguments = call["function"]["arguments"]
                parsed = _parse_tool_arguments(raw_arguments)
                self.hooks.tool_start(name, parsed)
                tool_started = time.monotonic()
                result = self.toolbox.execute(name, raw_arguments)
                tool_seconds = time.monotonic() - tool_started
                self.hooks.tool_end(name, result, tool_seconds)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": result.content,
                    }
                )

        elapsed = time.monotonic() - started
        self.hooks.turn_end(
            {
                "model": used_model,
                "lane": used_lane,
                "elapsed_s": elapsed,
                "input_tokens": turn_input,
                "output_tokens": turn_output,
                "turns": turns,
                "max_turns": True,
            }
        )
        raise AgentError(
            f"Reached the {self.max_turns}-turn limit with work still pending. "
            "Ask me to continue if you want another pass."
        )

    def _finish(
        self,
        started: float,
        turns: int,
        text: str,
        model: str,
        lane: str,
        turn_input: int,
        turn_output: int,
    ) -> TurnResult:
        elapsed = time.monotonic() - started
        result = TurnResult(
            text=text,
            turns=turns,
            model=model,
            lane=lane,
            input_tokens=turn_input,
            output_tokens=turn_output,
            elapsed=elapsed,
        )
        self.hooks.turn_end(
            {
                "model": model,
                "lane": lane,
                "elapsed_s": elapsed,
                "input_tokens": turn_input,
                "output_tokens": turn_output,
                "turns": turns,
            }
        )
        return result

    def _candidate_members(self) -> list[Member]:
        members = self.lanes.order(self.lane)
        if self.preferred_member:
            members = sorted(
                members, key=lambda member: member.id != self.preferred_member
            )
        return members

    def _stream_turn(
        self, messages: list[dict[str, Any]]
    ) -> tuple[Member, str, list[dict[str, Any]], dict[str, int] | None, str | None]:
        attempted: list[str] = []
        last_error: ProviderError | None = None
        tools = self.toolbox.specs() or None
        members = self._candidate_members()
        if not members:
            raise AgentError(f"Lane '{self.lane}' has no available members.")

        for member in members:
            self._raise_if_cancelled()
            if not self.lanes.acquire(member, timeout=30.0):
                attempted.append(f"{member.id}: lane capacity wait timed out")
                continue
            provider = self.lanes.provider_for(member)
            stream_started = time.monotonic()
            text_parts: list[str] = []
            calls: list[dict[str, Any]] = []
            usage: dict[str, int] | None = None
            finish: str | None = None
            emitted_any = False
            self.hooks.assistant_start()
            try:
                events = provider.stream_chat(
                    member.model,
                    messages,
                    tools,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                    cancel=self.cancel,
                )
                for event in events:
                    self._raise_if_cancelled()
                    if not isinstance(event, StreamEvent):
                        continue
                    if event.type == "text" and event.text:
                        emitted_any = True
                        text_parts.append(event.text)
                        self.hooks.text(event.text)
                    elif event.type == "reasoning" and event.text:
                        self.hooks.reasoning(event.text)
                    elif event.type == "tool_call":
                        name = (event.tool_name or "").strip()
                        if not name:
                            continue
                        call_id = event.tool_call_id or f"call_{len(calls) + 1}"
                        arguments = event.arguments if isinstance(event.arguments, str) else "{}"
                        calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments or "{}"},
                            }
                        )
                        emitted_any = True
                    elif event.type == "usage" and event.usage:
                        usage = {
                            "input_tokens": int(event.usage.get("input_tokens", 0) or 0),
                            "output_tokens": int(event.usage.get("output_tokens", 0) or 0),
                            "total_tokens": int(event.usage.get("total_tokens", 0) or 0),
                        }
                    elif event.type == "done":
                        finish = event.finish_reason
                elapsed_ms = (time.monotonic() - stream_started) * 1000.0
                self.lanes.report_success(member, elapsed_ms)
                return member, "".join(text_parts), calls, usage, finish
            except ProviderError as exc:
                phase = "mid_stream" if emitted_any else "before_stream"
                self.lanes.report_failure(member, exc, phase=phase)
                last_error = exc
                if emitted_any:
                    raise AgentError(
                        f"Stream from {member.id} failed after partial output: {exc}"
                    ) from exc
                attempted.append(f"{member.id}: {exc}")
            except Exception as exc:
                self.lanes.report_failure(member, exc, phase="before_stream")
                raise AgentError(f"{member.id} failed unexpectedly: {exc}") from exc
            finally:
                self.hooks.assistant_end()
                self.lanes.release(member)

        details = "\n".join(f"- {line}" for line in attempted[-6:]) or "- no candidates"
        hint = ""
        if last_error is not None and last_error.kind == "auth":
            hint = "\nCheck your API key: run `python3 main.py --setup` or set the configured env var."
        raise AgentError(f"All lane members failed:\n{details}{hint}")

    def _raise_if_cancelled(self) -> None:
        if self.cancel.is_set():
            raise AgentError("Cancelled.")

    def _context_messages(self) -> list[dict[str, Any]]:
        budget = max(1024, self.config.context_window - self.config.max_output_tokens - 1500)
        tool_chars = len(json.dumps(self.toolbox.specs(), ensure_ascii=False)) if self.toolbox.names else 0
        system_chars = len(self.system_prompt)

        def estimate(messages: list[dict[str, Any]]) -> int:
            chars = system_chars + tool_chars
            chars += sum(len(json.dumps(message, ensure_ascii=False)) for message in messages)
            return chars // 4 + 4 * (len(messages) + 1)

        if estimate(self.messages) <= budget:
            return _with_system(self.system_prompt, self.messages)

        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for message in self.messages:
            if message.get("role") == "user" and current:
                groups.append(current)
                current = []
            current.append(message)
        if current:
            groups.append(current)

        kept = [list(group) for group in groups]
        while len(kept) > 1 and estimate([message for group in kept for message in group]) > budget:
            kept.pop(0)
        kept = [[dict(message) for message in group] for group in kept]
        flat = [message for group in kept for message in group]
        if estimate(flat) > budget:
            for message in flat:
                content = message.get("content")
                if isinstance(content, str) and len(content) > 12000:
                    message["content"] = content[:9000] + "\n… [trimmed to fit context] …\n" + content[-2000:]
        flat = [message for group in kept for message in group]
        if estimate(flat) > budget and kept:
            kept = [kept[-1]]
            flat = [message for group in kept for message in group]
        if not self._trimmed_notice:
            self._trimmed_notice = True
            self.hooks.notice("Earlier context was trimmed to fit the model window.", "info")
        return _with_system(self.system_prompt, flat)


def _with_system(system_prompt: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if system_prompt.strip():
        return [{"role": "system", "content": system_prompt}, *messages]
    return list(messages)


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
