"""Command-line entry point and interactive shell for OHAR."""

from __future__ import annotations

import argparse
import dataclasses
import getpass
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

from . import __version__
from .agent import Agent
from .balancer import LaneManager
from .browser import BrowserManager
from .config import (
    PRESETS,
    config_path,
    load_config,
    load_env_file,
    new_config,
    ohar_home,
    resolve_api_key,
    write_config,
)
from .mcp import McpManager
from .memory import MemoryStore, make_memory_tools
from .models import ModelListError, list_models
from .planner import PlanState, make_plan_tool
from .prompts import build_system_prompt
from .sessions import Session, SessionStore
from .skills import SkillLibrary, make_skill_tool
from .subagents import make_task_tool
from .tools import Toolbox, default_tools
from .types import (
    AgentHooks,
    AgentError,
    ApprovalRequest,
    Config,
    ConfigError,
    Tool,
    ToolContext,
    ToolResult,
)
from .tui.dag import dag_nodes, summary_line as dag_summary_line, tree_lines as dag_tree_lines
from .tui.roster import SubagentRoster
from .ui import UI
from .webtools import make_web_tools
from .workflows.board import BlockLog, build_board, make_board_tool, render as render_board
from .workflows.gates import ApprovalLog, DecisionGates, make_confirm_wrapper
from .workflows.goals import GoalLedger, make_goal_tools
from .workflows.journal import ActivityJournal, wrap_hooks
from .workflows.readiness import CapabilityReadiness, make_readiness_tool, surfaces_from_config
from .workflows.store import workflows_root
from .workflows.verification import VerificationLedger, make_verify_tools
from .wrapper.briefing import build_briefing, render_briefing
from .wrapper.continuity import build_continuity, render_continuity
from .wrapper.handoff import build_handoff, render_provenance
from .wrapper.state import open_state, transition
from .wrapper.workspace import make_workspace_tool, probe as probe_workspace, workspace_line

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"
_HELP = """\
Commands
  /help                 show this help
  /model [member]       show or pick a configured lane member
  /lane [name]          show or switch lanes
  /lanes                lane health and load
  /agents               subagent limits and lane
  /skills               list available skills
  /skill name [task]    queue a skill, or run a task with it now
  /plan                 show the agent's current task plan
  /models [filter]      list models exposed by configured providers
  /mcp                  MCP server and tool status
  /memory [query]       inspect saved memory (use clear to erase)
  /tools                list the tools available this session
  /brief                mission briefing: continuity, goals, blockers
  /board                workflow board: goals, gates, units, blocks
  /goals [args]         list goals; new <objective> | done <id> <summary>
  /dag                  live subagent graph for this session
  /journal [n]          recent recorded activity (default 30)
  /workspace            probe the workspace (branch, dirty, recipes)
  /handoff              provenance bundle to pass work elsewhere
  /theme [name]         show or switch the terminal theme
  /new                  start a fresh conversation
  /sessions             list saved sessions
  /resume id|last       resume a saved session
  /status               session, model, and approval status
  /config               show configuration paths and limits
  /quit                 leave OHAR

Input
  ! command             run a shell command in the workspace
  Ctrl-C                interrupt a running turn
  Ctrl-D                exit
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ohar",
        description="OHAR — a fast, focused terminal AI agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 main.py\n"
            '  python3 main.py "find the bug in src/app.py and fix it"\n'
            "  python3 main.py --resume last\n"
            "  python3 main.py --model openai/gpt-4.1-mini --lane main\n"
            "  python3 main.py --setup\n"
        ),
    )
    parser.add_argument("prompt", nargs="*", help="One-shot prompt. Omit for interactive mode.")
    parser.add_argument("--setup", action="store_true", help="Create or replace the config.")
    parser.add_argument("--resume", nargs="?", const="last", metavar="SESSION", help="Resume a saved session.")
    parser.add_argument("--list-sessions", action="store_true", help="List saved sessions and exit.")
    parser.add_argument("--lane", metavar="NAME", help="Use a configured lane.")
    parser.add_argument(
        "--model",
        metavar="MEMBER",
        help="Prefer a configured lane member (provider/model or a unique model name).",
    )
    parser.add_argument("--workspace", metavar="DIR", help="Workspace root (default: current directory).")
    parser.add_argument("--config", metavar="FILE", help="Config file path.")
    parser.add_argument("--yolo", action="store_true", help="Auto-approve mutating tools.")
    parser.add_argument("--read-only", action="store_true", help="Disable write, shell, and other mutating tools.")
    parser.add_argument("--plain", action="store_true", help="Plain output, no colors or Markdown.")
    parser.add_argument("--no-banner", action="store_true", help="Hide the startup banner.")
    parser.add_argument(
        "--theme",
        choices=("aurora", "ember", "moss", "mono"),
        help="Override the UI theme.",
    )
    parser.add_argument("--version", action="version", version=f"OHAR {__version__}")
    return parser


def _target_config(args: argparse.Namespace) -> Path:
    return Path(args.config).expanduser() if args.config else config_path()


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return answer or default


def _secret(prompt: str) -> str:
    try:
        return getpass.getpass(prompt).strip()
    except EOFError:
        return ""


def _valid_env_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))


def _setup(target: Path, *, continue_after: bool = False) -> Config:
    if not sys.stdin.isatty():
        raise ConfigError(
            "No interactive terminal for setup. Copy config.example.toml to "
            f"{target}, set api_key (or api_key_env), then run again."
        )
    print("\nOHAR setup\n")
    keys = ["openai", "anthropic", "openrouter", "deepseek", "ollama", "custom"]
    for index, key in enumerate(keys, start=1):
        info = PRESETS[key]
        print(f"  {index}) {info['label']:<28} e.g. {info['model']}")
    choice = _ask("Provider", "1")
    try:
        preset = keys[int(choice) - 1]
    except (ValueError, IndexError):
        raise ConfigError("Setup cancelled: invalid provider choice.") from None
    info = PRESETS[preset]
    model = _ask("Model", str(info["model"] or ""))
    if preset == "custom":
        base_url = _ask(
            "Base URL (include /v1 when the API expects it)",
            str(info["base_url"]),
        )
    else:
        base_url = str(info["base_url"])
    print(
        "\nAPI key: paste it here to store it in config.toml (mode 0600), or leave "
        "blank to use an environment variable."
    )
    api_key = _secret("API key (hidden): ")
    api_key_env: str | None = None
    if api_key:
        print("The key will be written to config.toml; keep that file private.")
    else:
        default_env = str(info.get("api_key_env") or "")
        env_name = _ask("Env var name for the key (blank for none)", default_env)
        if env_name:
            if not _valid_env_name(env_name):
                raise ConfigError(
                    "That is not an environment variable name. If it is an API key, "
                    "run setup again and paste it at the hidden API key prompt."
                )
            api_key_env = env_name
        load_env_file()
        probe = new_config(
            preset,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
        )
        if resolve_api_key(next(iter(probe.providers.values()))):
            print(f"Using the key from {api_key_env or 'the environment'}.")
        else:
            print(
                f"Warning: no key is set for {api_key_env or 'this provider'}. "
                "Chat requests will fail until you add one."
            )
    config = new_config(
        preset,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=api_key or None,
    )
    if target.exists():
        backup = target.with_name(target.name + ".bak")
        shutil.copy2(target, backup)
        print(f"Existing config backed up to {backup}.")
    write_config(config, target)
    print(f"\nConfig written to {target}")
    if api_key:
        print("API key saved in config.toml (0600). Do not copy this file into a project.")
    if not continue_after:
        print("Start OHAR with: python3 main.py\n")
    return config


def _workspace(config: Config, override: str | None) -> Path:
    raw = override or config.workspace or os.getcwd()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists() or not path.is_dir():
        raise ConfigError(f"Workspace does not exist or is not a directory: {path}")
    return path


def _approval_mode(config: Config, args: argparse.Namespace) -> str:
    if args.yolo and args.read_only:
        raise ConfigError("--yolo and --read-only cannot be combined.")
    if args.read_only:
        return "read-only"
    if args.yolo:
        return "auto"
    return config.approval


def _is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


def _missing_keys(config: Config, lane: str) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    try:
        members = []
        for lane_config in config.lanes.values():
            if lane_config.name == lane:
                members = lane_config.members
                break
    except KeyError:
        return warnings
    for member in members:
        provider_name = member.split("/", 1)[0]
        if provider_name in seen:
            continue
        seen.add(provider_name)
        provider = config.providers.get(provider_name)
        if provider is None:
            warnings.append(f"Provider '{provider_name}' is not configured.")
            continue
        if _is_local_url(provider.base_url):
            continue
        if resolve_api_key(provider) is None:
            hint = (
                f"Set {provider.api_key_env}, add api_key to config.toml, "
                "or run python3 main.py --setup."
                if provider.api_key_env
                else "Add api_key or api_key_env to config.toml, or run python3 main.py --setup."
            )
            warnings.append(f"No API key for '{provider_name}'. {hint}")
    return warnings


def _row_sessions(sessions: list) -> list[dict]:
    return [
        {
            "id": meta.id,
            "updated_at": meta.updated_at.replace("T", " ")[:16],
            "message_count": str(meta.message_count),
            "title": meta.title,
        }
        for meta in sessions
    ]


class App:
    def __init__(
        self,
        config: Config,
        workspace: Path,
        ui: UI,
        *,
        lane: str,
        preferred_member: str | None,
        approval_mode: str,
        session: Session,
        store: SessionStore,
        resumed: bool = False,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.ui = ui
        self.store = store
        self.session = session
        self.pending_skill: tuple[str, str] | None = None
        self.skills = SkillLibrary.discover(SKILLS_DIR)
        self.lanes = LaneManager(config)
        self.read_only = approval_mode == "read-only"
        self.roster = SubagentRoster()
        self.state = None
        wf_root = workflows_root(workspace)
        self.journal = ActivityJournal(wf_root, workspace)
        self.goals = GoalLedger(wf_root, workspace, journal=self.journal)
        self.gates = DecisionGates(wf_root, workspace, journal=self.journal)
        self.approvals = ApprovalLog(wf_root, workspace, journal=self.journal)
        self.blocks = BlockLog(wf_root, workspace, journal=self.journal)
        self.units = VerificationLedger(wf_root, workspace, journal=self.journal)
        self.context = ToolContext(
            workspace=workspace,
            approval_mode=approval_mode,
            read_only=self.read_only,
            shell_timeout=config.shell_timeout,
            confirm=ui.confirm,
        )
        base_hooks = AgentHooks(
            assistant_start=ui.stream_start,
            assistant_end=ui.stream_end,
            text=ui.text,
            reasoning=ui.reasoning,
            tool_start=ui.tool_start,
            tool_end=ui.tool_end,
            notice=ui.notice,
            subagent=self._observe_subagent,
            turn_end=ui.turn_end,
        )
        self._base_hooks = base_hooks
        self.hooks = wrap_hooks(base_hooks, self.journal, session_id=session.meta.id)
        self.plan = PlanState()
        self.memory: MemoryStore | None = None
        self.browser: BrowserManager | None = None
        self.mcp: McpManager | None = None
        self.warnings: list[str] = []
        tools = default_tools(self.context)
        if self.read_only:
            tools = [tool for tool in tools if tool.read_only]
        if self.skills.list():
            tools.append(make_skill_tool(self.skills))
        tools.append(make_plan_tool(self.plan, self._plan_changed))

        shared_extras: list[Tool] = []
        if config.web.enabled:
            web_tools = make_web_tools(
                allow_private=config.web.allow_private,
                require_approval=config.web.require_approval,
                timeout=config.web.timeout,
                max_bytes=config.web.max_bytes,
                max_chars=config.web.max_chars,
                search_backend=config.web.search_backend,
                searx_url=config.web.searx_url,
                user_agent=config.web.user_agent,
            )
            tools.extend(web_tools)
            if not config.web.require_approval:
                shared_extras.extend(tool for tool in web_tools if tool.read_only)

        if config.memory.enabled:
            self.memory = MemoryStore(
                ohar_home() / "memory",
                workspace,
                max_records=config.memory.max_records,
            )
            memory_tools = make_memory_tools(
                self.memory, require_approval=config.memory.require_approval
            )
            tools.extend(memory_tools)
            shared_extras.extend(tool for tool in memory_tools if tool.read_only)

        if config.browser.enabled:
            self.browser = BrowserManager(config.browser)
            tools.extend(self.browser.tools)

        if config.mcp_servers:
            self.mcp = McpManager(config.mcp_servers)
            self.warnings.extend(self.mcp.start())
            mcp_tools = self.mcp.tools
            tools.extend(mcp_tools)
            shared_extras.extend(tool for tool in mcp_tools if tool.read_only)

        self.reader = CapabilityReadiness(
            approval_mode=approval_mode,
            surfaces=surfaces_from_config(
                config,
                browser_available=self.browser is not None,
                memory_enabled=config.memory.enabled,
            ),
            journal=self.journal,
            tool_names=[tool.name for tool in tools],
        )
        workflow_tools = [
            make_readiness_tool(self.reader),
            make_board_tool(
                goals=self.goals,
                gates=self.gates,
                blocks=self.blocks,
                units=self.units,
                journal=self.journal,
            ),
            make_workspace_tool(self.context, session.meta.id),
        ]
        if not self.read_only:
            workflow_tools.extend(
                make_goal_tools(self.goals, journal=self.journal, require_approval=False)
            )
            workflow_tools.extend(
                make_verify_tools(self.units, goals=self.goals, require_approval=True)
            )
        tools.extend(workflow_tools)
        shared_extras.extend(
            tool for tool in workflow_tools if tool.read_only and tool.name != "workspace_status"
        )

        tools.append(
            make_task_tool(
                config=config,
                lanes=self.lanes,
                workspace=workspace,
                hooks=self.hooks,
                skill_library=self.skills,
                extra_tools=shared_extras,
            )
        )
        if self.read_only:
            tools = [tool for tool in tools if tool.read_only]
        self.toolbox = Toolbox(self.context, tools)
        self.system_prompt = build_system_prompt(
            config,
            workspace,
            skill_index=self.skills.index_text(1600),
            max_parallel=config.subagents_max,
            read_only=self.read_only,
            tool_names=self.toolbox.names,
        )
        validate_lane = lane or config.default_lane
        self.agent = Agent(
            config,
            self.lanes,
            self.toolbox,
            system_prompt=self.system_prompt,
            lane=validate_lane,
            hooks=self.hooks,
            preferred_member=preferred_member,
        )
        self._load_session(session, restore_tokens=True)
        if resumed:
            ui.startup(
                model=self.current_member,
                lane=self.agent.lane,
                resumed=session.meta.title,
            )
        else:
            ui.startup(model=self.current_member, lane=self.agent.lane)
        for warning in self.warnings:
            ui.notice(warning, "warning")

    def _plan_changed(self, state: PlanState) -> None:
        self.ui.plan(state.to_dict())

    def close(self) -> None:
        for manager in (self.browser, self.mcp):
            if manager is None:
                continue
            try:
                manager.close()
            except Exception as exc:
                ui = getattr(self, "ui", None)
                if ui is not None and os.environ.get("OHAR_DEBUG"):
                    ui.error(f"cleanup: {type(exc).__name__}: {exc}")

    @property
    def current_member(self) -> str:
        if self.agent.preferred_member:
            return self.agent.preferred_member
        members = self.lanes.members(self.agent.lane)
        return members[0].id if members else "(none)"

    def _load_session(self, session: Session, *, restore_tokens: bool) -> None:
        self.session = session
        self.agent.set_messages(session.messages)
        if restore_tokens:
            self.agent.input_tokens = session.meta.input_tokens
            self.agent.output_tokens = session.meta.output_tokens

    def _save(self) -> None:
        if not self.agent.messages:
            return
        try:
            self.session.messages = self.agent.messages
            self.session.meta.lane = self.agent.lane
            self.session.meta.model = self.current_member
            self.session.meta.input_tokens = self.agent.input_tokens
            self.session.meta.output_tokens = self.agent.output_tokens
            self.store.save(self.session)
        except (OSError, ValueError) as exc:
            self.ui.error(f"Could not save session: {exc}")

    def _inject_skill(self, text: str, skill: tuple[str, str]) -> str:
        name, body = skill
        return (
            f"{text}\n\n"
            f"---\n\nUse the following skill as your working procedure for this request.\n\n"
            f"## Skill: {name}\n\n{body}"
        )

    def send(self, text: str, skill: tuple[str, str] | None = None) -> None:
        chosen = skill or self.pending_skill
        if chosen:
            text = self._inject_skill(text, chosen)
            self.ui.event(f"Using skill '{chosen[0]}'.", "info")
            self.pending_skill = None
        try:
            self.agent.run(text)
        except AgentError as exc:
            self.ui.error(str(exc))
        except KeyboardInterrupt:
            self.ui.notice("Interrupted. The conversation is saved.", "warning")
        except Exception as exc:
            if os.environ.get("OHAR_DEBUG"):
                traceback.print_exc()
            self.ui.error(f"{type(exc).__name__}: {exc}")
        finally:
            self._save()

    def new_session(self) -> None:
        self.session = self.store.create(
            cwd=str(self.workspace),
            model=self.current_member,
            lane=self.agent.lane,
            persist=False,
        )
        self.agent.reset()
        self.plan.update([])
        self.roster.reset()
        self.pending_skill = None
        self._bind_session_state()
        self.ui.event("Started a new conversation.", "info")

    def resume(self, session_id: str) -> None:
        session = self.store.resume(session_id)
        if session is None:
            self.ui.error(f"No session found for '{session_id}'.")
            return
        self._load_session(session, restore_tokens=True)
        self.plan.update([])
        if session.meta.lane in self.config.lanes:
            try:
                self.agent.set_lane(session.meta.lane)
            except ConfigError:
                pass
        self.agent.set_preferred_member(
            session.meta.model
            if any(
                member.id == session.meta.model
                for member in self.lanes.members(self.agent.lane)
            )
            else None
        )
        self.pending_skill = None
        self.ui.event(f"Resumed {session.meta.id} · {session.meta.title}", "info")

    def set_member(self, query: str) -> None:
        members = self.lanes.members(self.agent.lane)
        exact = [member.id for member in members if member.id == query]
        if exact:
            member_id = exact[0]
        else:
            matches = [member.id for member in members if member.model == query]
            if len(matches) == 1:
                member_id = matches[0]
            elif len(matches) > 1:
                self.ui.error(f"'{query}' is ambiguous: {', '.join(matches)}")
                return
            else:
                self.ui.error(
                    f"'{query}' is not configured in lane '{self.agent.lane}'. "
                    "Add it to your config, or run /lane to switch lanes."
                )
                return
        self.agent.set_preferred_member(member_id)
        self._save()
        self.ui.event(f"Model set to {member_id}.", "info")

    def set_lane(self, lane: str) -> None:
        if lane not in self.config.lanes:
            self.ui.error(
                f"Unknown lane '{lane}'. Available: {', '.join(sorted(self.config.lanes))}."
            )
            return
        try:
            self.agent.set_lane(lane)
        except ConfigError as exc:
            self.ui.error(str(exc))
            return
        self._save()
        self.ui.event(f"Lane set to {lane}.", "info")

    def show_models(self) -> None:
        self.ui.info(f"Lane {self.agent.lane} · strategy {self.config.lanes[self.agent.lane].strategy}")
        for member in self.lanes.members(self.agent.lane):
            marker = "●" if member.id == self.current_member else "○"
            self.ui.info(f"  {marker} {member.id}")
        self.ui.info("Use /model name to prefer a member. Edit config.toml to add members.")

    def show_lanes(self) -> None:
        self.ui.lanes_table(self.lanes.status())

    def show_agents(self) -> None:
        lane = "subagents" if "subagents" in self.config.lanes else self.config.default_lane
        self.ui.info(
            f"Subagents: up to {self.config.subagents_max} in parallel, read-only · "
            f"lane {lane} · timeout {self.config.subagent_timeout:g}s"
        )
        self.ui.info("Ask naturally for parallel research; the agent calls the task tool.")

    def show_skills(self) -> None:
        skills = self.skills.list()
        if not skills:
            self.ui.info("No skills found in ./skills.")
            return
        for skill in skills:
            self.ui.info(f"  {skill.name:<10} {skill.description}")
        self.ui.info("Use /skill name to queue one for your next message.")

    def queue_skill(self, name: str, task: str = "") -> None:
        body = self.skills.body(name, max_chars=8000)
        if body is None:
            self.ui.error(f"Unknown skill '{name}'. Use /skills to list them.")
            return
        if task:
            self.send(task, skill=(name, body))
        else:
            self.pending_skill = (name, body)
            self.ui.event(f"Skill '{name}' queued for the next message.", "info")

    def show_status(self) -> None:
        self.ui.info(f"Session   {self.session.meta.id}")
        self.ui.info(f"Title     {self.session.meta.title}")
        self.ui.info(f"Lane      {self.agent.lane}")
        self.ui.info(f"Model     {self.current_member}")
        self.ui.info(f"Approval  {self.context.approval_mode}")
        self.ui.info(f"Workspace {self.workspace}")
        self.ui.info(
            f"Tokens    {self.agent.input_tokens} in / {self.agent.output_tokens} out"
        )

    def show_config(self) -> None:
        path = self.config.path or config_path()
        self.ui.info(f"Config    {path}")
        self.ui.info(f"Home      {ohar_home()}")
        self.ui.info(f"Theme     {self.config.ui.theme}")
        self.ui.info(
            f"Limits    turns {self.config.max_turns} · output {self.config.max_output_tokens} · "
            f"context {self.config.context_window}"
        )
        self.ui.info(f"Subagents up to {self.config.subagents_max} · timeout {self.config.subagent_timeout:g}s")
        self.ui.info(
            f"Web       {'on' if self.config.web.enabled else 'off'} · "
            f"approval {self.config.web.require_approval} · {self.config.web.search_backend}"
        )
        self.ui.info(
            f"Browser   {'on' if self.config.browser.enabled else 'off'} · "
            f"approval {self.config.browser.require_approval}"
        )
        self.ui.info(
            f"Memory    {'on' if self.config.memory.enabled else 'off'} · "
            f"max {self.config.memory.max_records} records"
        )
        self.ui.info(f"MCP       {len(self.config.mcp_servers)} configured server(s)")

    def show_plan(self) -> None:
        if not self.plan.steps:
            self.ui.info("No active plan. The agent updates one with the update_plan tool.")
            return
        self.ui.plan(self.plan.to_dict())

    def show_tools(self) -> None:
        rows = [
            [
                tool.name,
                "read" if tool.read_only else "write",
                tool.description.replace("\n", " ")[:110],
            ]
            for tool in self.toolbox.tools
        ]
        self.ui.data_table(
            f"Tools · {len(rows)} available",
            ["tool", "access", "description"],
            rows,
            widths=[22, 7, 64],
            empty="No tools are available in this mode.",
        )

    # ------------------------------------------------------- workflow commands

    def _observe_subagent(self, event: dict) -> None:
        try:
            self.roster.apply(event)
        except Exception:
            pass
        self.ui.subagent_event(event)

    def _bind_session_state(self) -> None:
        session_id = self.session.meta.id
        if not self.read_only:
            self.context.confirm = make_confirm_wrapper(
                self.ui.confirm,
                log=self.approvals,
                gates=self.gates,
                journal=self.journal,
                session_id=session_id,
            )
        self.hooks = wrap_hooks(self._base_hooks, self.journal, session_id=session_id)
        self.agent.hooks = self.hooks
        try:
            self.state, _ = open_state(
                session_id,
                goal=self.session.meta.title or "",
                workspace=str(self.workspace),
                model=self.current_member,
                lane=self.agent.lane,
            )
        except (OSError, ValueError) as exc:
            self.state = None
            if os.environ.get("OHAR_DEBUG"):
                self.ui.error(f"session state unavailable: {exc}")

    def show_brief(self) -> None:
        if self.state is None:
            self.ui.error("Session state is unavailable, so there is nothing to brief.")
            return
        try:
            continuity = build_continuity(
                meta=self.session.meta,
                state=self.state,
                workspace=self.workspace,
                plan=self.plan.to_dict(),
            )
            briefing = build_briefing(
                state=self.state,
                claims=(),
                events=[dataclasses.asdict(event) for event in self.journal.recent(limit=8)],
                plan=self.plan.to_dict(),
                continuity=continuity,
            )
        except Exception as exc:
            self.ui.error(f"Briefing failed: {type(exc).__name__}: {exc}")
            return
        self.ui.info(render_continuity(continuity, self.ui.theme, width=self.ui.width))
        self.ui.info(render_briefing(briefing, self.ui.theme, width=self.ui.width))

    def show_board(self) -> None:
        try:
            board = build_board(
                goals=self.goals,
                gates=self.gates,
                blocks=self.blocks,
                units=self.units,
                journal=self.journal,
                session_id=self.session.meta.id,
            )
            lines = render_board(board, theme=self.ui.theme)
        except Exception as exc:
            self.ui.error(f"Board failed: {type(exc).__name__}: {exc}")
            return
        if lines:
            self.ui.info("\n".join(lines))

    def goals_command(self, argument: str) -> None:
        verb, _, rest = argument.partition(" ")
        verb = verb.lower()
        if verb == "new" and rest.strip():
            try:
                goal = self.goals.create(
                    rest.strip(),
                    ["Deliver what was asked and show evidence"],
                    session_id=self.session.meta.id,
                )
            except Exception as exc:
                self.ui.error(f"Could not record goal: {exc}")
                return
            try:
                self.state = transition(
                    self.session.meta.id,
                    "plan_presented",
                    goal=rest.strip(),
                    next_action=f"advance goal {goal.id}",
                )
            except (OSError, ValueError):
                pass
            self.ui.event(f"Goal {goal.id} recorded.", "info")
            self.ui.info(self.goals.render(goal))
            return
        if verb in {"done", "complete"} and rest.strip():
            goal_id, _, summary = rest.partition(" ")
            if not summary.strip():
                self.ui.error("Usage: /goals done <id> <summary>")
                return
            try:
                goal = self.goals.complete(goal_id, summary.strip(), require_gate=False)
            except Exception as exc:
                self.ui.error(f"Could not complete goal: {exc}")
                return
            self.ui.info(self.goals.render(goal))
            return
        try:
            listed = self.goals.list()
        except Exception as exc:
            self.ui.error(f"Could not read goals: {exc}")
            return
        if not listed:
            self.ui.notice(
                "No goals for this workspace. Use /goals new <objective> or ask the agent.",
                "info",
            )
            return
        for goal in listed:
            self.ui.info(self.goals.render(goal))

    def show_dag(self) -> None:
        nodes = dag_nodes(self.roster)
        if not nodes:
            self.ui.notice("No subagent runs recorded in this session yet.", "info")
            return
        lines = dag_tree_lines(nodes, self.ui.width, 40, self.ui.theme)
        self.ui.info(dag_summary_line(nodes, self.ui.theme))
        if lines:
            self.ui.info("\n".join(lines))

    def show_journal(self, argument: str) -> None:
        limit = 30
        if argument.isdigit():
            limit = max(1, min(200, int(argument)))
        try:
            events = self.journal.recent(limit=limit)
        except Exception as exc:
            self.ui.error(f"Could not read journal: {exc}")
            return
        rows = [
            [
                event.at[11:19],
                event.event,
                event.subject or "",
                "ok" if event.ok is True else ("fail" if event.ok is False else ""),
            ]
            for event in events
        ]
        self.ui.data_table(
            f"Journal · {len(rows)} events",
            ["time", "event", "subject", "result"],
            rows,
            widths=[9, 22, 28, 7],
            empty="Nothing recorded yet this session.",
        )

    def show_workspace_info(self) -> None:
        info = probe_workspace(self.workspace)
        self.ui.info(workspace_line(info))

    def show_handoff(self) -> None:
        if self.state is None:
            self.ui.error("Session state is unavailable, so a handoff cannot be built.")
            return
        try:
            handoff = build_handoff(
                state=self.state,
                messages=self.agent.messages,
                plan=self.plan.to_dict(),
                model=self.current_member,
                lane=self.agent.lane,
            )
        except Exception as exc:
            self.ui.error(f"Handoff failed: {type(exc).__name__}: {exc}")
            return
        self.ui.info(handoff.listing())
        self.ui.info(render_provenance(handoff.provenance, plain=self.ui.plain))

    def discover_models(self, query: str = "") -> None:
        try:
            with self.ui.working("fetching models"):
                models = list_models(self.config, timeout=20.0)
        except ModelListError as exc:
            self.ui.error(str(exc))
            return
        except Exception as exc:
            self.ui.error(f"Could not list models: {type(exc).__name__}: {exc}")
            return
        needle = query.strip().casefold()
        if needle:
            models = [
                model
                for model in models
                if needle
                in " ".join(
                    (model.id, model.provider, model.owned_by or "")
                ).casefold()
            ]
        if not models:
            suffix = f" matching '{query}'" if query.strip() else ""
            self.ui.info(f"No models found{suffix}.")
            return

        def compact(value: int | None) -> str:
            if value is None:
                return "?"
            if value >= 1_000_000:
                return f"{value / 1_000_000:.1f}M"
            if value >= 1_000:
                return f"{value / 1_000:.0f}k"
            return str(value)

        shown = models[:120]
        rows = [
            [
                model.id,
                model.provider,
                compact(model.context_length),
                compact(model.max_output_tokens),
            ]
            for model in shown
        ]
        suffix = f" matching '{query}'" if query.strip() else ""
        self.ui.data_table(
            f"Models · {len(shown)} of {len(models)}{suffix}",
            ["model", "provider", "context", "output"],
            rows,
            widths=[42, 14, 9, 9],
        )
        if len(models) > len(shown):
            self.ui.info(f"Showing the first {len(shown)}; narrow the filter to see more.")
        self.ui.info(
            "Add a desired provider/model to a lane's members in config.toml, "
            "then choose it with /model."
        )

    def show_mcp(self) -> None:
        if not self.config.mcp_servers:
            self.ui.info(
                "No MCP servers configured. Add [[mcp.servers]] entries to config.toml "
                "and restart OHAR."
            )
            return
        if self.mcp is None:
            self.ui.error("MCP manager is unavailable.")
            return
        rows = []
        for summary in self.mcp.summaries():
            rows.append(
                [
                    str(summary.get("name", "?")),
                    str(summary.get("status", "?")),
                    str(summary.get("tool_count", 0)),
                    str(summary.get("error", "") or "-").replace("\n", " ")[:80],
                ]
            )
        self.ui.data_table(
            f"MCP servers · {len(rows)}",
            ["server", "status", "tools", "error"],
            rows,
            widths=[18, 10, 6, 60],
        )
        if self.mcp.tools:
            self.ui.info("MCP tools: " + ", ".join(tool.name for tool in self.mcp.tools))

    def show_memory(self, argument: str = "") -> None:
        if self.memory is None:
            self.ui.info("Persistent memory is disabled in config.toml.")
            return
        if argument.strip().lower() == "clear":
            request = ApprovalRequest(
                action="memory_clear",
                summary="Erase all OHAR memories for this workspace",
                detail=str(self.workspace),
                risk="high",
            )
            approved = self.context.approval_mode == "auto" or self.ui.confirm(request)
            if not approved:
                self.ui.event("Memory clear cancelled.", "warning")
                return
            try:
                count = self.memory.clear()
            except (OSError, ValueError) as exc:
                self.ui.error(f"Could not clear memory: {exc}")
                return
            self.ui.event(f"Cleared {count} memory record(s).", "info")
            return
        try:
            records = (
                self.memory.recall(argument.strip(), 20)
                if argument.strip()
                else self.memory.recent(20)
            )
        except (OSError, TypeError, ValueError) as exc:
            self.ui.error(f"Could not read memory: {exc}")
            return
        if not records:
            suffix = f" for '{argument.strip()}'" if argument.strip() else ""
            self.ui.info(f"No saved memories{suffix}.")
            return
        rows = [
            [
                record["id"],
                str(record["created_at"]).replace("T", " ")[:16],
                ", ".join(record["tags"]) or "-",
                str(record["text"]).replace("\n", " ")[:100],
            ]
            for record in records
        ]
        title = f"Memory · {len(records)} record(s)"
        if argument.strip():
            title += f" matching '{argument.strip()[:40]}'"
        self.ui.data_table(
            title,
            ["id", "saved", "tags", "text"],
            rows,
            widths=[10, 16, 22, 48],
        )
        self.ui.info("Use /memory clear to erase workspace memory.")

    def choose_theme(self, name: str = "") -> None:
        if not name.strip():
            self.ui.info(
                f"Theme {self.config.ui.theme}. Available: "
                + ", ".join(self.ui.theme_names())
            )
            return
        if not self.ui.set_theme(name):
            self.ui.error(
                f"Unknown theme '{name}'. Choose from: "
                + ", ".join(self.ui.theme_names())
            )
            return
        self.ui.event(f"Theme set to {self.config.ui.theme} for this session.", "info")

    def handle_slash(self, line: str) -> bool:
        command, _, argument = line.partition(" ")
        command = command.lower()
        argument = argument.strip()
        if command in {"/quit", "/exit", "/q"}:
            return False
        if command == "/help":
            for row in _HELP.rstrip().splitlines():
                self.ui.info(row)
        elif command == "/model":
            if argument:
                self.set_member(argument)
            else:
                self.show_models()
        elif command == "/lane":
            if argument:
                self.set_lane(argument)
            else:
                self.show_lanes()
                self.ui.info(f"Current lane: {self.agent.lane}")
        elif command == "/lanes":
            self.show_lanes()
        elif command == "/agents":
            self.show_agents()
        elif command == "/skills":
            self.show_skills()
        elif command == "/skill":
            name, _, task = argument.partition(" ")
            if not name:
                self.show_skills()
            else:
                self.queue_skill(name, task.strip())
        elif command == "/plan":
            self.show_plan()
        elif command == "/models":
            self.discover_models(argument)
        elif command == "/mcp":
            self.show_mcp()
        elif command == "/memory":
            self.show_memory(argument)
        elif command == "/tools":
            self.show_tools()
        elif command == "/brief":
            self.show_brief()
        elif command == "/board":
            self.show_board()
        elif command == "/goals":
            self.goals_command(argument)
        elif command == "/dag":
            self.show_dag()
        elif command == "/journal":
            self.show_journal(argument)
        elif command == "/workspace":
            self.show_workspace_info()
        elif command == "/handoff":
            self.show_handoff()
        elif command == "/theme":
            self.choose_theme(argument)
        elif command in {"/new", "/clear"}:
            self.new_session()
        elif command == "/sessions":
            self.ui.sessions_table(_row_sessions(self.store.list(20)))
            self.ui.info("Resume with /resume id, or /resume last.")
        elif command == "/resume":
            if not argument:
                self.ui.error("Usage: /resume id|last")
            else:
                self.resume(argument)
        elif command == "/status":
            self.show_status()
        elif command == "/config":
            self.show_config()
        else:
            self.ui.error(f"Unknown command {command}. Type /help.")
        return True

    def shell(self, command: str) -> None:
        if not command.strip():
            self.ui.info("Usage: ! <command>")
            return
        self.ui.tool_start("shell", {"command": command})
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.config.shell_timeout,
                start_new_session=True,
            )
            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()
            parts = []
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"[stderr]\n{stderr}")
            if completed.returncode != 0:
                parts.append(f"[exit {completed.returncode}]")
            content = "\n".join(parts) or "(no output)"
            result = ToolResult(
                ok=completed.returncode == 0,
                content=content[:200_000],
                meta={"exit_code": completed.returncode, "preview_lines": 60},
            )
        except subprocess.TimeoutExpired:
            result = ToolResult.failure(
                f"Command timed out after {self.config.shell_timeout:g}s.",
                preview_lines=60,
            )
        except (OSError, ValueError) as exc:
            result = ToolResult.failure(f"Command failed: {exc}", preview_lines=60)
        if self.ui.is_plain:
            self.ui.info(result.content)
        else:
            self.ui.tool_end("shell", result, time.monotonic() - started)


def _print_sessions(store: SessionStore, config: Config, workspace: Path, args: argparse.Namespace) -> int:
    ui = UI(config, workspace, plain=args.plain or None, version=__version__)
    rows = _row_sessions(store.list(50))
    if not rows:
        ui.info("No saved sessions.")
    else:
        ui.sessions_table(rows)
    return 0


def _completions(skills: SkillLibrary) -> list[str]:
    words = [
        "/help",
        "/model",
        "/lane",
        "/lanes",
        "/agents",
        "/skills",
        "/skill",
        "/plan",
        "/models",
        "/mcp",
        "/memory",
        "/tools",
        "/brief",
        "/board",
        "/goals",
        "/dag",
        "/journal",
        "/workspace",
        "/handoff",
        "/theme",
        "/new",
        "/sessions",
        "/resume",
        "/status",
        "/config",
        "/quit",
    ]
    words.extend(f"/skill {name}" for name in skills.names)
    return words


def _run_repl(app: App) -> int:
    for warning in _missing_keys(app.config, app.agent.lane):
        app.ui.notice(warning, "warning")
    app.ui.set_completions(_completions(app.skills))
    while True:
        try:
            line = app.ui.read_prompt()
        except KeyboardInterrupt:
            app.ui.notice("Press Ctrl-D to exit.", "info")
            continue
        if line is None:
            app.ui.info("Bye.")
            return 0
        text = line.strip()
        if not text:
            continue
        if text.startswith("!"):
            try:
                app.shell(text[1:].strip())
            except KeyboardInterrupt:
                app.ui.notice("Interrupted.", "warning")
            continue
        if text.startswith("/"):
            try:
                if not app.handle_slash(text):
                    app.ui.info("Bye.")
                    return 0
            except KeyboardInterrupt:
                app.ui.notice("Interrupted.", "warning")
            continue
        app.send(text)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        load_env_file()
    except OSError:
        pass

    target = _target_config(args)
    try:
        if args.setup:
            _setup(target)
            return 0

        if not target.is_file():
            if not sys.stdin.isatty():
                raise ConfigError(
                    f"No config found at {target}. Copy config.example.toml there "
                    "and set your API key, or run python3 main.py --setup."
                )
            _setup(target, continue_after=True)

        config = load_config(target)
        if args.theme:
            config.ui.theme = args.theme
        if args.no_banner:
            config.ui.banner = False
        workspace = _workspace(config, args.workspace)
        store = SessionStore()

        if args.list_sessions:
            return _print_sessions(store, config, workspace, args)

        lane = args.lane or config.default_lane
        if lane not in config.lanes:
            raise ConfigError(
                f"Unknown lane '{lane}'. Available: {', '.join(sorted(config.lanes))}."
            )

        session: Session | None = None
        resumed = False
        if args.resume:
            session = store.resume(args.resume)
            if session is None:
                raise ConfigError(f"No saved session matches '{args.resume}'.")
            resumed = True
            if not args.lane and session.meta.lane in config.lanes:
                lane = session.meta.lane
        else:
            initial_members = config.lanes[lane].members
            initial_model = initial_members[0] if initial_members else ""
            session = store.create(
                cwd=str(workspace), model=initial_model, lane=lane, persist=False
            )
        preferred: str | None = None
        if args.model:
            candidates = config.lanes[lane].members
            if args.model in candidates:
                preferred = args.model
            else:
                matches = [member for member in candidates if member.split("/", 1)[-1] == args.model]
                if len(matches) != 1:
                    raise ConfigError(
                        f"Model '{args.model}' is not a unique member of lane '{lane}'. "
                        f"Choose from: {', '.join(candidates)}."
                    )
                preferred = matches[0]
        elif session and session.meta.model in config.lanes[lane].members:
            preferred = session.meta.model

        approval = _approval_mode(config, args)
        ui = UI(config, workspace, plain=args.plain or None, version=__version__)
        app = App(
            config,
            workspace,
            ui,
            lane=lane,
            preferred_member=preferred,
            approval_mode=approval,
            session=session,
            store=store,
            resumed=resumed,
        )

        try:
            if args.prompt:
                prompt = " ".join(args.prompt).strip()
                if not prompt:
                    raise ConfigError("The prompt is empty.")
                app.send(prompt)
                return 0
            return _run_repl(app)
        finally:
            app.close()
    except ConfigError as exc:
        print(f"ohar: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except OSError:
            pass
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
