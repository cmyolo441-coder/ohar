"""Shared core types for OHAR. Stdlib only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_LANE = "main"
SUBAGENT_LANE = "subagents"
MAX_SUBAGENTS = 8
SCHEMA_VERSION = 1


class ConfigError(Exception):
    """Invalid or unusable configuration."""


class AgentError(Exception):
    """The agent could not complete a turn."""


@dataclass(slots=True)
class ProviderConfig:
    name: str
    type: str = "openai"  # openai | anthropic
    base_url: str = ""
    api_key_env: str | None = None
    api_key: str | None = field(default=None, repr=False)
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0


@dataclass(slots=True)
class LaneConfig:
    name: str
    strategy: str = "failover"  # failover | round_robin | weighted | least_busy
    members: list[str] = field(default_factory=list)  # "provider/model"
    weights: list[float] = field(default_factory=list)
    cooldown_seconds: float = 60.0
    max_inflight: int = MAX_SUBAGENTS


@dataclass(slots=True)
class UIConfig:
    theme: str = "aurora"
    banner: bool = True
    glyphs: str = "unicode"  # unicode | ascii
    markdown: bool = True
    syntax: bool = True


@dataclass(slots=True)
class WebConfig:
    enabled: bool = True
    require_approval: bool = False
    allow_private: bool = False
    timeout: float = 30.0
    max_bytes: int = 2_000_000
    max_chars: int = 12_000
    search_backend: str = "duckduckgo"
    searx_url: str | None = None
    user_agent: str = "OHAR/0.2"


@dataclass(slots=True)
class BrowserConfig:
    enabled: bool = True
    require_approval: bool = False
    allow_private: bool = False
    headless: bool = True
    timeout: float = 30.0
    chromium_path: str | None = None


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = True
    max_records: int = 500
    require_approval: bool = True


@dataclass(slots=True)
class McpServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout: float = 20.0
    enabled: bool = True


@dataclass(slots=True)
class Config:
    schema_version: int = SCHEMA_VERSION
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    lanes: dict[str, LaneConfig] = field(default_factory=dict)
    default_lane: str = DEFAULT_LANE
    approval: str = "ask"  # ask | auto | read-only
    workspace: str | None = None
    max_turns: int = 24
    max_output_tokens: int = 4096
    context_window: int = 64000
    temperature: float | None = None
    subagents_max: int = MAX_SUBAGENTS
    subagent_timeout: float = 180.0
    shell_timeout: float = 120.0
    ui: UIConfig = field(default_factory=UIConfig)
    web: WebConfig = field(default_factory=WebConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mcp_servers: list[McpServerConfig] = field(default_factory=list)
    path: Path | None = None

    def lane(self, name: str | None = None) -> LaneConfig:
        key = name or self.default_lane
        try:
            return self.lanes[key]
        except KeyError as exc:
            raise ConfigError(f"Unknown lane '{key}'. Available: {', '.join(sorted(self.lanes))}") from exc

    def provider(self, name: str) -> ProviderConfig:
        try:
            return self.providers[name]
        except KeyError as exc:
            raise ConfigError(f"Unknown provider '{name}'.") from exc


@dataclass(frozen=True, slots=True)
class Member:
    lane: str
    provider: str
    model: str
    weight: float = 1.0
    index: int = 0

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(slots=True)
class StreamEvent:
    type: str  # text | reasoning | tool_call | usage | done
    text: str = ""
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: str = ""
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


class ProviderError(Exception):
    """A normalized provider/transport failure."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        code: str | None = None,
        kind: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after
        self.code = code
        self.kind = kind

    def __str__(self) -> str:
        bits = [self.message]
        if self.status is not None:
            bits.append(f"HTTP {self.status}")
        if self.kind and self.kind != "unknown":
            bits.append(self.kind)
        return " · ".join(bits)


@dataclass(slots=True)
class ToolResult:
    ok: bool
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, **meta: Any) -> "ToolResult":
        return cls(True, content, meta)

    @classmethod
    def failure(cls, content: str, **meta: Any) -> "ToolResult":
        return cls(False, content, meta)


@dataclass(slots=True)
class ApprovalRequest:
    action: str
    summary: str
    detail: str = ""
    risk: str = "medium"


@dataclass(slots=True)
class ToolContext:
    workspace: Path
    approval_mode: str = "ask"
    read_only: bool = False
    shell_timeout: float = 120.0
    confirm: Callable[[ApprovalRequest], bool] | None = None

    def request_approval(self, request: ApprovalRequest) -> bool:
        if self.read_only:
            return False
        if self.approval_mode == "auto":
            return True
        if self.approval_mode != "ask":
            return False
        if self.confirm is None:
            return False
        return bool(self.confirm(request))


ToolHandler = Callable[[dict[str, Any], ToolContext], ToolResult]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    read_only: bool = False
    destructive: bool = False

    def api_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


@dataclass(slots=True)
class AgentHooks:
    """UI callbacks. Worker agents must only publish immutable event data."""

    assistant_start: Callable[[], None] = _noop
    assistant_end: Callable[[], None] = _noop
    text: Callable[[str], None] = _noop
    reasoning: Callable[[str], None] = _noop
    tool_start: Callable[[str, dict[str, Any]], None] = _noop
    tool_end: Callable[[str, ToolResult, float], None] = _noop
    notice: Callable[[str, str], None] = _noop
    subagent: Callable[[dict[str, Any]], None] = _noop
    usage: Callable[[dict[str, int]], None] = _noop
    turn_end: Callable[[dict[str, Any]], None] = _noop
