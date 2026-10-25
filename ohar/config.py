"""Configuration loading, presets, and secure local storage for OHAR."""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from .types import (
    DEFAULT_LANE,
    MAX_SUBAGENTS,
    SCHEMA_VERSION,
    SUBAGENT_LANE,
    BrowserConfig,
    Config,
    ConfigError,
    LaneConfig,
    McpServerConfig,
    MemoryConfig,
    ProviderConfig,
    UIConfig,
    WebConfig,
)

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_STRATEGIES = {"failover", "round_robin", "weighted", "least_busy"}
_APPROVAL_MODES = {"ask", "auto", "read-only"}
_DEFAULT_BASE = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}

PRESETS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "type": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
    },
    "anthropic": {
        "label": "Anthropic",
        "type": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-5",
    },
    "openrouter": {
        "label": "OpenRouter",
        "type": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "openai/gpt-4.1-mini",
    },
    "deepseek": {
        "label": "DeepSeek",
        "type": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
    },
    "ollama": {
        "label": "Ollama / local",
        "type": "openai",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
        "model": "qwen3:8b",
    },
    "custom": {
        "label": "Custom OpenAI-compatible",
        "type": "openai",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "your-model",
    },
}


def ohar_home() -> Path:
    raw = os.environ.get("OHAR_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".ohar"


def config_path() -> Path:
    return ohar_home() / "config.toml"


def ensure_home() -> Path:
    home = ohar_home()
    home.mkdir(parents=True, exist_ok=True)
    try:
        home.chmod(0o700)
    except OSError:
        pass
    return home


def config_exists(path: Path | str | None = None) -> bool:
    return Path(path).expanduser().is_file() if path else config_path().is_file()


def _unquote_env(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        if value[0] == '"':
            try:
                return str(json.loads(value))
            except json.JSONDecodeError:
                return value[1:-1]
        return value[1:-1]
    return value


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if separator and _ENV_NAME.match(name):
            values[name] = _unquote_env(value)
    return values


def load_env_file(path: Path | str | None = None) -> dict[str, str]:
    env_path = Path(path).expanduser() if path else ohar_home() / ".env"
    values = _read_env_file(env_path)
    for name, value in values.items():
        os.environ.setdefault(name, value)
    return values


def resolve_api_key(provider: ProviderConfig | str) -> str | None:
    """Resolve a literal config key, then the process env, then ``.env``."""
    if isinstance(provider, ProviderConfig) and provider.api_key:
        value = provider.api_key.strip()
        if value:
            return value
    load_env_file()
    if isinstance(provider, str):
        env_name: str | None = provider or None
    else:
        env_name = provider.api_key_env
    if not env_name:
        return None
    value = os.environ.get(env_name)
    return value.strip() if value and value.strip() else None


def _expect_table(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a TOML table.")
    return value


def _table_of_tables(value: Any, where: str) -> dict[str, dict[str, Any]]:
    table = _expect_table(value, where)
    for name, entry in table.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{where} contains an invalid name.")
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}.{name} must be a TOML table.")
    return table


def _string(value: Any, where: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string.")
    return value


def _number(value: Any, where: str, *, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number.")
    return float(value)


def _integer(value: Any, where: str, *, default: int | None = None, minimum: int = 0, maximum: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ConfigError(f"{where} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{where} must be at most {maximum}.")
    return result


def _bool(value: Any, where: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be true or false.")
    return value


def _parse_providers(raw: Any) -> dict[str, ProviderConfig]:
    entries = _table_of_tables(raw, "providers") if raw is not None else {}
    providers: dict[str, ProviderConfig] = {}
    for name, data in entries.items():
        known = {"type", "base_url", "api_key_env", "api_key", "timeout", "headers"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                f"providers.{name} has unknown key(s): {', '.join(unknown)}."
            )
        kind = str(data.get("type", "openai")).strip().lower()
        if kind not in _DEFAULT_BASE:
            raise ConfigError(
                f"providers.{name}.type must be 'openai' or 'anthropic'."
            )
        base_url = _string(data.get("base_url"), f"providers.{name}.base_url", allow_none=True)
        if not base_url:
            base_url = _DEFAULT_BASE[kind]
        api_key_env = _string(
            data.get("api_key_env"), f"providers.{name}.api_key_env", allow_none=True
        )
        if api_key_env is not None:
            api_key_env = api_key_env.strip()
            if api_key_env and not _ENV_NAME.match(api_key_env):
                raise ConfigError(
                    f"providers.{name}.api_key_env must be an environment variable name. "
                    "Put the secret itself in api_key."
                )
            api_key_env = api_key_env or None
        api_key = _string(
            data.get("api_key"), f"providers.{name}.api_key", allow_none=True
        )
        if api_key is not None:
            api_key = api_key.strip() or None
        timeout = _number(data.get("timeout"), f"providers.{name}.timeout", default=120.0)
        if timeout <= 0:
            raise ConfigError(f"providers.{name}.timeout must be positive.")
        raw_headers = data.get("headers", {})
        headers = _expect_table(raw_headers, f"providers.{name}.headers")
        parsed_headers: dict[str, str] = {}
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ConfigError(
                    f"providers.{name}.headers must map strings to strings."
                )
            parsed_headers[key] = value
        providers[name] = ProviderConfig(
            name=name,
            type=kind,
            base_url=base_url.rstrip("/"),
            api_key_env=api_key_env,
            api_key=api_key,
            headers=parsed_headers,
            timeout=timeout,
        )
    return providers


def _parse_lanes(raw: Any, providers: dict[str, ProviderConfig]) -> dict[str, LaneConfig]:
    entries = _table_of_tables(raw, "lanes") if raw is not None else {}
    lanes: dict[str, LaneConfig] = {}
    for name, data in entries.items():
        known = {
            "strategy",
            "members",
            "weights",
            "cooldown_seconds",
            "max_inflight",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"lanes.{name} has unknown key(s): {', '.join(unknown)}.")
        strategy = str(data.get("strategy", "failover")).strip().lower()
        if strategy not in _STRATEGIES:
            raise ConfigError(
                f"lanes.{name}.strategy must be one of: {', '.join(sorted(_STRATEGIES))}."
            )
        raw_members = data.get("members", [])
        if not isinstance(raw_members, list) or not raw_members:
            raise ConfigError(
                f"lanes.{name}.members must be a non-empty array of 'provider/model' strings."
            )
        members: list[str] = []
        for index, member in enumerate(raw_members):
            if not isinstance(member, str) or "/" not in member:
                raise ConfigError(
                    f"lanes.{name}.members[{index}] must look like 'provider/model'."
                )
            provider, model = member.split("/", 1)
            provider, model = provider.strip(), model.strip()
            if not provider or not model:
                raise ConfigError(
                    f"lanes.{name}.members[{index}] must look like 'provider/model'."
                )
            if provider not in providers:
                raise ConfigError(
                    f"lanes.{name}.members[{index}] references unknown provider '{provider}'."
                )
            if member in members:
                raise ConfigError(
                    f"lanes.{name}.members contains duplicate member '{member}'."
                )
            members.append(member)
        raw_weights = data.get("weights", [])
        if not isinstance(raw_weights, list):
            raise ConfigError(f"lanes.{name}.weights must be an array of numbers.")
        if raw_weights and len(raw_weights) != len(members):
            raise ConfigError(
                f"lanes.{name}.weights must be empty or match the {len(members)} members."
            )
        weights: list[float] = []
        for index, weight in enumerate(raw_weights):
            parsed = _number(weight, f"lanes.{name}.weights[{index}]")
            if parsed <= 0:
                raise ConfigError(f"lanes.{name}.weights[{index}] must be positive.")
            weights.append(parsed)
        cooldown = _number(
            data.get("cooldown_seconds"),
            f"lanes.{name}.cooldown_seconds",
            default=60.0,
        )
        if cooldown < 0:
            raise ConfigError(f"lanes.{name}.cooldown_seconds must not be negative.")
        max_inflight = _integer(
            data.get("max_inflight"),
            f"lanes.{name}.max_inflight",
            default=MAX_SUBAGENTS,
            minimum=1,
            maximum=64,
        )
        lanes[name] = LaneConfig(
            name=name,
            strategy=strategy,
            members=members,
            weights=weights,
            cooldown_seconds=cooldown,
            max_inflight=max_inflight,
        )
    return lanes


def _parse_web(raw: Any) -> WebConfig:
    data = _expect_table(raw, "web") if raw is not None else {}
    known = {
        "enabled",
        "require_approval",
        "allow_private",
        "timeout",
        "max_bytes",
        "max_chars",
        "search_backend",
        "searx_url",
        "user_agent",
    }
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"web has unknown key(s): {', '.join(unknown)}.")
    backend = str(data.get("search_backend", "duckduckgo")).strip().lower()
    if backend not in {"duckduckgo", "searxng"}:
        raise ConfigError("web.search_backend must be 'duckduckgo' or 'searxng'.")
    searx_url = _string(data.get("searx_url"), "web.searx_url", allow_none=True)
    if searx_url is not None:
        searx_url = searx_url.strip() or None
    if backend == "searxng" and not searx_url:
        raise ConfigError("web.searx_url is required when search_backend = 'searxng'.")
    user_agent = _string(data.get("user_agent"), "web.user_agent", allow_none=True)
    user_agent = (user_agent or "OHAR/0.2").strip() or "OHAR/0.2"
    return WebConfig(
        enabled=_bool(data.get("enabled"), "web.enabled", default=True),
        require_approval=_bool(
            data.get("require_approval"), "web.require_approval", default=False
        ),
        allow_private=_bool(data.get("allow_private"), "web.allow_private", default=False),
        timeout=_number(data.get("timeout"), "web.timeout", default=30.0),
        max_bytes=_integer(data.get("max_bytes"), "web.max_bytes", default=2_000_000, minimum=1024, maximum=50_000_000),
        max_chars=_integer(data.get("max_chars"), "web.max_chars", default=12_000, minimum=500, maximum=200_000),
        search_backend=backend,
        searx_url=searx_url,
        user_agent=user_agent,
    )


def _parse_browser(raw: Any) -> BrowserConfig:
    data = _expect_table(raw, "browser") if raw is not None else {}
    known = {
        "enabled",
        "require_approval",
        "allow_private",
        "headless",
        "timeout",
        "chromium_path",
    }
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"browser has unknown key(s): {', '.join(unknown)}.")
    chromium_path = _string(
        data.get("chromium_path"), "browser.chromium_path", allow_none=True
    )
    if chromium_path is not None:
        chromium_path = str(Path(chromium_path).expanduser()) if chromium_path.strip() else None
    return BrowserConfig(
        enabled=_bool(data.get("enabled"), "browser.enabled", default=True),
        require_approval=_bool(
            data.get("require_approval"), "browser.require_approval", default=False
        ),
        allow_private=_bool(
            data.get("allow_private"), "browser.allow_private", default=False
        ),
        headless=_bool(data.get("headless"), "browser.headless", default=True),
        timeout=_number(data.get("timeout"), "browser.timeout", default=30.0),
        chromium_path=chromium_path,
    )


def _parse_memory(raw: Any) -> MemoryConfig:
    data = _expect_table(raw, "memory") if raw is not None else {}
    known = {"enabled", "max_records", "require_approval"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"memory has unknown key(s): {', '.join(unknown)}.")
    return MemoryConfig(
        enabled=_bool(data.get("enabled"), "memory.enabled", default=True),
        max_records=_integer(
            data.get("max_records"), "memory.max_records", default=500, minimum=1, maximum=10_000
        ),
        require_approval=_bool(
            data.get("require_approval"), "memory.require_approval", default=True
        ),
    )


def _parse_mcp(raw: Any) -> list[McpServerConfig]:
    data = _expect_table(raw, "mcp") if raw is not None else {}
    unknown = sorted(set(data) - {"servers"})
    if unknown:
        raise ConfigError(f"mcp has unknown key(s): {', '.join(unknown)}.")
    raw_servers = data.get("servers", [])
    if not isinstance(raw_servers, list):
        raise ConfigError("mcp.servers must be an array of tables.")
    specs: list[McpServerConfig] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_servers):
        where = f"mcp.servers[{index}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a table.")
        known = {"name", "command", "args", "env", "cwd", "timeout", "enabled"}
        unknown_entry = sorted(set(entry) - known)
        if unknown_entry:
            raise ConfigError(f"{where} has unknown key(s): {', '.join(unknown_entry)}.")
        name = _string(entry.get("name"), f"{where}.name")
        if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ConfigError(f"{where}.name must use letters, digits, '_' or '-'.")
        if name in seen:
            raise ConfigError(f"Duplicate MCP server name '{name}'.")
        seen.add(name)
        command = _string(entry.get("command"), f"{where}.command")
        if not command or not command.strip():
            raise ConfigError(f"{where}.command must not be empty.")
        raw_args = entry.get("args", [])
        if not isinstance(raw_args, list) or any(not isinstance(arg, str) for arg in raw_args):
            raise ConfigError(f"{where}.args must be an array of strings.")
        raw_env = entry.get("env", {})
        env_table = _expect_table(raw_env, f"{where}.env")
        env: dict[str, str] = {}
        for key, value in env_table.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ConfigError(f"{where}.env must map strings to strings.")
            env[key] = value
        cwd = _string(entry.get("cwd"), f"{where}.cwd", allow_none=True)
        if cwd is not None:
            cwd = str(Path(cwd).expanduser()) if cwd.strip() else None
        timeout = _number(entry.get("timeout"), f"{where}.timeout", default=20.0)
        if timeout <= 0:
            raise ConfigError(f"{where}.timeout must be positive.")
        specs.append(
            McpServerConfig(
                name=name,
                command=command.strip(),
                args=list(raw_args),
                env=env,
                cwd=cwd,
                timeout=timeout,
                enabled=_bool(entry.get("enabled"), f"{where}.enabled", default=True),
            )
        )
    return specs


def _validate(config: Config) -> Config:
    if not config.providers:
        raise ConfigError("No providers configured. Add at least one [providers.<name>] table.")
    if not config.lanes:
        raise ConfigError("No lanes configured. Add at least one [lanes.<name>] table.")
    if config.default_lane not in config.lanes:
        raise ConfigError(
            f"default_lane '{config.default_lane}' does not exist. "
            f"Available: {', '.join(sorted(config.lanes))}."
        )
    if config.approval not in _APPROVAL_MODES:
        raise ConfigError(
            f"approval must be one of: {', '.join(sorted(_APPROVAL_MODES))}."
        )
    for lane in config.lanes.values():
        for member in lane.members:
            provider = member.split("/", 1)[0]
            if provider not in config.providers:
                raise ConfigError(
                    f"Lane '{lane.name}' references unknown provider '{provider}'."
                )
    if config.max_turns < 1:
        raise ConfigError("max_turns must be at least 1.")
    if config.max_output_tokens < 1:
        raise ConfigError("max_output_tokens must be at least 1.")
    if config.context_window < 1:
        raise ConfigError("context_window must be at least 1.")
    if config.temperature is not None and not 0 <= config.temperature <= 2:
        raise ConfigError("temperature must be between 0 and 2.")
    config.subagents_max = max(1, min(MAX_SUBAGENTS, int(config.subagents_max)))
    if config.subagent_timeout <= 0:
        raise ConfigError("subagent_timeout must be positive.")
    if config.shell_timeout <= 0:
        raise ConfigError("shell_timeout must be positive.")
    config.ui.theme = (config.ui.theme or "aurora").strip().lower()
    config.ui.glyphs = (config.ui.glyphs or "unicode").strip().lower()
    if config.ui.glyphs not in {"unicode", "ascii"}:
        raise ConfigError("ui.glyphs must be 'unicode' or 'ascii'.")
    if config.web.timeout <= 0 or config.browser.timeout <= 0:
        raise ConfigError("web.timeout and browser.timeout must be positive.")
    if config.web.max_bytes < 1024 or config.web.max_chars < 500:
        raise ConfigError("web.max_bytes or web.max_chars is too small.")
    if config.web.search_backend not in {"duckduckgo", "searxng"}:
        raise ConfigError("web.search_backend must be 'duckduckgo' or 'searxng'.")
    if config.web.search_backend == "searxng" and not config.web.searx_url:
        raise ConfigError("web.searx_url is required when search_backend = 'searxng'.")
    if config.memory.max_records < 1:
        raise ConfigError("memory.max_records must be positive.")
    mcp_names: set[str] = set()
    for server in config.mcp_servers:
        if server.name in mcp_names:
            raise ConfigError(f"Duplicate MCP server name '{server.name}'.")
        mcp_names.add(server.name)
        if not server.command.strip():
            raise ConfigError(f"MCP server '{server.name}' has an empty command.")
        if server.timeout <= 0:
            raise ConfigError(f"MCP server '{server.name}' timeout must be positive.")
    return config


def load_config(path: Path | str | None = None) -> Config:
    source = Path(path).expanduser() if path else config_path()
    if not source.is_file():
        raise ConfigError(
            f"No config found at {source}. Run `python3 main.py --setup` to create one."
        )
    try:
        with source.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {source}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read {source}: {exc}") from exc

    version = data.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise ConfigError(
            f"{source} uses schema_version {version!r}; this OHAR supports up to {SCHEMA_VERSION}."
        )

    providers = _parse_providers(data.get("providers"))
    lanes = _parse_lanes(data.get("lanes"), providers)
    default_lane = data.get("default_lane")
    if default_lane is None:
        default_lane = DEFAULT_LANE if DEFAULT_LANE in lanes else next(iter(lanes), DEFAULT_LANE)
    if not isinstance(default_lane, str):
        raise ConfigError("default_lane must be a string.")

    approval = str(data.get("approval", "ask")).strip().lower()
    workspace = data.get("workspace")
    if workspace is not None:
        workspace = _string(workspace, "workspace")
        if workspace is not None:
            workspace = str(Path(workspace).expanduser())

    raw_ui = data.get("ui", {})
    ui_table = _expect_table(raw_ui, "ui")
    ui = UIConfig(
        theme=str(ui_table.get("theme", "aurora")),
        banner=_bool(ui_table.get("banner"), "ui.banner", default=True),
        glyphs=str(ui_table.get("glyphs", "unicode")),
        markdown=_bool(ui_table.get("markdown"), "ui.markdown", default=True),
        syntax=_bool(ui_table.get("syntax"), "ui.syntax", default=True),
    )
    web = _parse_web(data.get("web"))
    browser = _parse_browser(data.get("browser"))
    memory = _parse_memory(data.get("memory"))
    mcp_servers = _parse_mcp(data.get("mcp"))

    temperature = data.get("temperature")
    config = Config(
        schema_version=version,
        providers=providers,
        lanes=lanes,
        default_lane=default_lane,
        approval=approval,
        workspace=workspace,
        max_turns=_integer(data.get("max_turns"), "max_turns", default=24, minimum=1, maximum=500),
        max_output_tokens=_integer(
            data.get("max_output_tokens"), "max_output_tokens", default=4096, minimum=1, maximum=1_000_000
        ),
        context_window=_integer(
            data.get("context_window"), "context_window", default=64000, minimum=256, maximum=10_000_000
        ),
        temperature=None if temperature is None else _number(temperature, "temperature"),
        subagents_max=_integer(
            data.get("subagents_max"), "subagents_max", default=MAX_SUBAGENTS, minimum=1, maximum=MAX_SUBAGENTS
        ),
        subagent_timeout=_number(
            data.get("subagent_timeout"), "subagent_timeout", default=180.0
        ),
        shell_timeout=_number(data.get("shell_timeout"), "shell_timeout", default=120.0),
        ui=ui,
        web=web,
        browser=browser,
        memory=memory,
        mcp_servers=mcp_servers,
        path=source,
    )
    return _validate(config)


def new_config(
    preset: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    api_key: str | None = None,
    workspace: str | None = None,
) -> Config:
    key = (preset or "").strip().lower()
    if key not in PRESETS:
        raise ConfigError(
            f"Unknown provider preset '{preset}'. Choose: {', '.join(PRESETS)}."
        )
    info = PRESETS[key]
    chosen_model = (model or str(info["model"] or "")).strip()
    if not chosen_model:
        raise ConfigError("A model name is required.")
    chosen_base = (base_url or str(info["base_url"])).strip().rstrip("/")
    chosen_env = api_key_env if api_key_env is not None else info.get("api_key_env")
    if chosen_env is not None:
        chosen_env = str(chosen_env).strip() or None
        if chosen_env and not _ENV_NAME.match(chosen_env):
            raise ConfigError("api_key_env must be an environment variable name.")
    provider = ProviderConfig(
        name=key,
        type=str(info["type"]),
        base_url=chosen_base,
        api_key_env=chosen_env,
        api_key=(api_key.strip() or None) if isinstance(api_key, str) else None,
        timeout=120.0,
    )
    member = f"{key}/{chosen_model}"
    lanes = {
        DEFAULT_LANE: LaneConfig(
            name=DEFAULT_LANE,
            strategy="failover",
            members=[member],
            max_inflight=MAX_SUBAGENTS,
        ),
        SUBAGENT_LANE: LaneConfig(
            name=SUBAGENT_LANE,
            strategy="round_robin",
            members=[member],
            max_inflight=MAX_SUBAGENTS,
        ),
    }
    config = Config(
        schema_version=SCHEMA_VERSION,
        providers={key: provider},
        lanes=lanes,
        default_lane=DEFAULT_LANE,
        approval="ask",
        workspace=workspace,
        ui=UIConfig(),
    )
    return _validate(config)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_config(config: Config) -> str:
    lines: list[str] = [
        "# OHAR configuration — private (chmod 600). A literal api_key may be present.",
        f"schema_version = {int(config.schema_version)}",
        f"default_lane = {_toml_string(config.default_lane)}",
        f"approval = {_toml_string(config.approval)}",
    ]
    if config.workspace:
        lines.append(f"workspace = {_toml_string(config.workspace)}")
    lines.extend(
        [
            f"max_turns = {int(config.max_turns)}",
            f"max_output_tokens = {int(config.max_output_tokens)}",
            f"context_window = {int(config.context_window)}",
            f"subagents_max = {int(config.subagents_max)}",
            f"subagent_timeout = {float(config.subagent_timeout):g}",
            f"shell_timeout = {float(config.shell_timeout):g}",
        ]
    )
    if config.temperature is not None:
        lines.append(f"temperature = {float(config.temperature):g}")
    lines.extend(
        [
            "",
            "[ui]",
            f"theme = {_toml_string(config.ui.theme)}",
            f"banner = {str(config.ui.banner).lower()}",
            f"glyphs = {_toml_string(config.ui.glyphs)}",
            f"markdown = {str(config.ui.markdown).lower()}",
            f"syntax = {str(config.ui.syntax).lower()}",
            "",
            "[web]",
            f"enabled = {str(config.web.enabled).lower()}",
            f"require_approval = {str(config.web.require_approval).lower()}",
            f"allow_private = {str(config.web.allow_private).lower()}",
            f"timeout = {float(config.web.timeout):g}",
            f"max_bytes = {int(config.web.max_bytes)}",
            f"max_chars = {int(config.web.max_chars)}",
            f"search_backend = {_toml_string(config.web.search_backend)}",
        ]
    )
    if config.web.searx_url:
        lines.append(f"searx_url = {_toml_string(config.web.searx_url)}")
    lines.append(f"user_agent = {_toml_string(config.web.user_agent)}")
    lines.extend(
        [
            "",
            "[browser]",
            f"enabled = {str(config.browser.enabled).lower()}",
            f"require_approval = {str(config.browser.require_approval).lower()}",
            f"allow_private = {str(config.browser.allow_private).lower()}",
            f"headless = {str(config.browser.headless).lower()}",
            f"timeout = {float(config.browser.timeout):g}",
        ]
    )
    if config.browser.chromium_path:
        lines.append(f"chromium_path = {_toml_string(config.browser.chromium_path)}")
    lines.extend(
        [
            "",
            "[memory]",
            f"enabled = {str(config.memory.enabled).lower()}",
            f"max_records = {int(config.memory.max_records)}",
            f"require_approval = {str(config.memory.require_approval).lower()}",
        ]
    )
    for server in config.mcp_servers:
        lines.extend(
            [
                "",
                "[[mcp.servers]]",
                f"name = {_toml_string(server.name)}",
                f"command = {_toml_string(server.command)}",
                "args = [" + ", ".join(_toml_string(arg) for arg in server.args) + "]",
                f"enabled = {str(server.enabled).lower()}",
                f"timeout = {float(server.timeout):g}",
            ]
        )
        if server.cwd:
            lines.append(f"cwd = {_toml_string(server.cwd)}")
        if server.env:
            pairs = ", ".join(
                f"{_toml_string(key)} = {_toml_string(value)}"
                for key, value in server.env.items()
            )
            lines.append(f"env = {{ {pairs} }}")
    for name, provider in config.providers.items():
        lines.extend(
            [
                "",
                f"[providers.{name}]",
                f"type = {_toml_string(provider.type)}",
                f"base_url = {_toml_string(provider.base_url)}",
            ]
        )
        if provider.api_key:
            lines.append(f"api_key = {_toml_string(provider.api_key)}")
        if provider.api_key_env:
            lines.append(f"api_key_env = {_toml_string(provider.api_key_env)}")
        lines.append(f"timeout = {float(provider.timeout):g}")
        if provider.headers:
            lines.extend(["", f"[providers.{name}.headers]"])
            for header, value in provider.headers.items():
                lines.append(f"{_toml_string(header)} = {_toml_string(value)}")
    for name, lane in config.lanes.items():
        members = ", ".join(_toml_string(member) for member in lane.members)
        lines.extend(
            [
                "",
                f"[lanes.{name}]",
                f"strategy = {_toml_string(lane.strategy)}",
                f"members = [{members}]",
            ]
        )
        if lane.weights:
            lines.append(
                "weights = [" + ", ".join(f"{float(weight):g}" for weight in lane.weights) + "]"
            )
        lines.extend(
            [
                f"cooldown_seconds = {float(lane.cooldown_seconds):g}",
                f"max_inflight = {int(lane.max_inflight)}",
            ]
        )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_config(config: Config, path: Path | str | None = None) -> Path:
    target = Path(path).expanduser() if path else config_path()
    if target.is_dir():
        raise ConfigError(f"Config path {target} is a directory.")
    ensure_home()
    text = render_config(_validate(config))
    _atomic_write(target, text.encode("utf-8"), 0o600)
    config.path = target
    return target


def save_env_secret(name: str, value: str) -> Path:
    if not _ENV_NAME.match(name):
        raise ConfigError("Secret name must be a valid environment variable name.")
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("Secret value must not be empty.")
    home = ensure_home()
    env_path = home / ".env"
    lines: list[str] = []
    if env_path.is_file():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ConfigError(f"Cannot read {env_path}: {exc}") from exc
    kept = [
        line
        for line in lines
        if line.strip().partition("=")[0].strip().removeprefix("export ").strip() != name
    ]
    kept.append(f"{name}={json.dumps(value.strip(), ensure_ascii=False)}")
    _atomic_write(env_path, ("\n".join(kept) + "\n").encode("utf-8"), 0o600)
    os.environ[name] = value.strip()
    return env_path
