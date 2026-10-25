# OHAR

**A fast, focused terminal AI agent.** Zero dependencies, one command, up to eight parallel subagents.

OHAR is a small Python coding agent that runs in your terminal. It streams answers, reads and edits
files, runs commands with approval, searches code, loads skills, researches the web, drives a real
headless Chromium, connects to MCP servers, remembers workspace facts, and delegates broad read-only
research to parallel subagents. It is built from scratch on the Python standard library — no pip
install required.

```text
  ██████╗  ██╗  ██╗   █████╗   ██████╗
 ██╔═══██╗ ██║  ██║  ██╔══██╗  ██╔══██╗
 ██║   ██║ ███████║  ███████║  ██████╔╝
 ██║   ██║ ██╔══██║  ██╔══██║  ██╔══██╗
 ╚██████╔╝ ██║  ██║  ██║  ██║  ██║  ██║
  ╚═════╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝
        think · build · ship
```

## Quick start

Requires **Python 3.11+**. No third-party packages.

```sh
cd ohar
python3 main.py --setup      # choose a provider, model, and API key
python3 main.py              # interactive terminal agent
python3 main.py "find and fix the failing test in tests/test_api.py"
```

One-shot mode prints the answer and exits. Interactive mode keeps a session, streams Markdown,
shows tool cards and token/time stats, and saves the conversation for `/resume`.

Works with any OpenAI-compatible endpoint (OpenAI, OpenRouter, DeepSeek, Groq, Together, Ollama,
LM Studio, vLLM, custom gateways) and with the native Anthropic Messages API. Web/browser features
need no browser for basic fetch; rich browser automation uses an installed Chromium and no extra
Python packages.

## What it does

- **Streaming agent loop.** Text appears as it is generated; tool calls are executed inside the loop,
  with bounded outputs, timeouts, and a hard turn limit.
- **Lanes and load balancing.** Models are grouped into lanes with `failover`, `round_robin`,
  `weighted`, or `least_busy` strategies. Failed/rate-limited members cool down and the next member
  is tried automatically. Bulkheads limit concurrent calls per lane.
- **Up to eight parallel subagents.** The agent can call a `task` tool to run eight independent
  **read-only** researchers at once. Each has its own context, lane, model, skills, web access, and
  timeout; results are aggregated for the main agent. Subagents cannot write files or run shell
  commands.
- **Web research and a real browser.** `web_search` and `web_fetch` return bounded, readable content
  (DuckDuckGo or SearXNG search). `browser_open`, `browser_snapshot`, `browser_click`, `browser_type`,
  `browser_press`, `browser_scroll`, `browser_back`, and `browser_screenshot` drive a headless
  Chromium via the DevTools protocol. Private/local URLs are blocked by default; optional approval
  gates are available for every web and browser action.
- **MCP tools.** Configure stdio MCP servers in `config.toml`; OHAR performs the JSON-RPC handshake,
  discovers their tools, namespaces them as `mcp__server__tool`, and exposes read-only tools to
  subagents. Side-effecting MCP tools require approval unless your approval mode is `auto`.
- **Visible plans and workspace memory.** `update_plan` renders a live checklist for multi-step work.
  `remember`, `recall`, and `forget` provide explicit, workspace-scoped persistence under `~/.ohar`
  with approval before writes; `/plan` and `/memory` let you inspect both.
- **Model discovery and syntax highlighting.** `/models` lists models exposed by configured providers
  so you can add them to lanes. Fenced code blocks are tokenized and highlighted in the terminal,
  with a `syntax = false` escape hatch.
- **Safe local tools.** `read_file`, `list_files`, `glob`, `grep`, `write_file`, `edit_file`, `shell`.
  Paths are confined to the workspace; writes require approval by default; dangerous shell patterns are
  blocked; output is truncated and ANSI/control characters are stripped.
- **Skills.** Sixteen curated coding skills (`plan`, `review`, `debug`, `verify`, `git`, `onboard`,
  `frontend`, `refactor`, `web`, `security`, `test`, `performance`, `docs`, `api`, `data`,
  `accessibility`) live in `skills/`. They are indexed in the prompt and loaded on demand.
- **Sessions.** Conversations are saved under `~/.ohar/sessions`; resume by id or `last`.
- **Mission control (ported from the Hermes wrapper/workflow layers).** Every session keeps a durable
  state machine plus private ledgers under `~/.ohar/workflows` and `~/.ohar/wrapper`: goals with
  acceptance criteria (`goal_create`, `goal_checkpoint`, `goal_complete`), frozen verification units
  (`unit_declare`, `unit_run`), approval receipts and decision gates, an activity journal, and a
  blocker log. `/brief` renders a mission briefing, `/board` a read-only work board, `/dag` a live
  subagent graph, `/journal` recent recorded activity, `/workspace` a git probe, and `/handoff` a
  provenance bundle for passing work elsewhere. Status words are claims, never evidence — the board
  always lists what is still missing.
- **Small and fast.** Pure-stdlib runtime, lazy browser/MCP startup, no background daemons, no
  telemetry, and no network calls except your configured provider, web tools, browser pages, and MCP
  servers.

## Configuration

The setup wizard writes `~/.ohar/config.toml` (or `$OHAR_HOME/config.toml`). See
[`config.example.toml`](config.example.toml) for every option.

```toml
schema_version = 1
default_lane = "main"
approval = "ask"              # ask | auto | read-only
max_turns = 24
max_output_tokens = 4096
context_window = 64000
subagents_max = 8

[ui]
theme = "aurora"              # aurora | ember | moss | mono
banner = true
glyphs = "unicode"            # unicode | ascii
markdown = true
syntax = true                 # highlight fenced code blocks

[web]
enabled = true
require_approval = false
allow_private = false
timeout = 30
max_bytes = 2000000
max_chars = 12000
search_backend = "duckduckgo" # duckduckgo | searxng
# searx_url = "http://localhost:8888"

[browser]
enabled = true
require_approval = false
allow_private = false
headless = true
timeout = 30
# chromium_path = "/usr/bin/chromium"

[memory]
enabled = true
max_records = 500
require_approval = true

[providers.openai]
type = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[providers.anthropic]
type = "anthropic"
base_url = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"

[lanes.main]
strategy = "failover"
members = ["openai/gpt-4.1-mini", "anthropic/claude-sonnet-4-5"]
cooldown_seconds = 60
max_inflight = 8

[lanes.subagents]
strategy = "least_busy"
members = ["openai/gpt-4.1-mini", "anthropic/claude-sonnet-4-5"]
max_inflight = 8

# Optional stdio MCP servers:
# [[mcp.servers]]
# name = "filesystem"
# command = "npx"
# args = ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/projects"]
# timeout = 20
```

A lane member is `provider/model`. The model may itself contain slashes
(`openrouter/openai/gpt-4.1-mini` parses as provider `openrouter`, model `openai/gpt-4.1-mini`).

### API keys

Set `api_key_env = "NAME"` and export the variable, or store it in `$OHAR_HOME/.env`
(mode `0600`). OHAR also supports a literal `api_key = "..."` in `config.toml` for
gateways that expect one; in that case keep `config.toml` mode `0600` and never
commit it. Keys are redacted from tool output and errors. Literal keys are never
written to sessions.

### Web, browser, and MCP

`web_fetch` uses only http/https, rejects credentials in URLs, blocks private,
loopback, and reserved address ranges by default, and limits redirects, bytes,
and output. Set `allow_private = true` only when you intentionally read local
services. The browser validates the requested URL and the final page after
navigation or clicks, and blanks a blocked page; subresources loaded by a public
page are still subject to Chromium's own network policy. Only browse untrusted
pages when the machine has no sensitive local services. Browser automation
starts lazily on the first browser tool and Chromium is terminated when OHAR
exits. It requires no Playwright or Selenium.

DuckDuckGo search first uses the HTML endpoint and automatically falls back to
the lighter endpoint when the main page is challenged. For a stable private
search path, point `web.searx_url` at your own SearXNG instance and set
`search_backend = "searxng"`.

MCP servers are launched over stdio with the command and arguments from your
config. Their tools are discovered at startup; a broken server produces a warning
without taking down OHAR. Treat MCP servers as trusted local code, because they
run with your user permissions. `/mcp` shows server status and tool counts.

### Load balancing

| Strategy | Behavior |
| --- | --- |
| `failover` | Configured order; cooling members go last |
| `round_robin` | Rotate the lead candidate on every call |
| `weighted` | Smooth weighted rotation using `weights = [...]` |
| `least_busy` | Lowest in-flight load, then observed latency |

Health is tracked per member: consecutive failures, cooldown, in-flight calls, and EWMA latency.
`429 Retry-After` and provider rejections cool the member down; `/lanes` shows live state.
A mid-stream failure never silently replays text — OHAR surfaces the partial response and error.

## Commands

| Command | Purpose |
| --- | --- |
| `/help` | List commands |
| `/model [member]` | Show or prefer a lane member |
| `/lane [name]` | Show lanes or switch lanes |
| `/lanes` | Health, load, and cooldown table |
| `/agents` | Subagent limits and lane |
| `/skills` | List skills |
| `/skill name [task]` | Queue a skill, or run a task with it now |
| `/plan` | Show the agent's current task plan |
| `/models [filter]` | Discover models from configured providers |
| `/mcp` | MCP server status and discovered tools |
| `/memory [query]` | Inspect workspace memory; `/memory clear` erases it |
| `/tools` | List tools available in this session |
| `/brief` | Mission briefing: continuity, goals, blockers, next action |
| `/board` | Read-only work board over goals, gates, units, blocks |
| `/goals [args]` | List goals; `new <objective>`, `done <id> <summary>` |
| `/dag` | Live subagent graph for this session |
| `/journal [n]` | Recent recorded activity |
| `/workspace` | Probe the workspace: branch, revision, dirty count |
| `/handoff` | Provenance bundle to pass work to another agent |
| `/theme [name]` | Show or switch the terminal theme |
| `/new`, `/clear` | Start a fresh conversation |
| `/sessions` | List saved sessions |
| `/resume id\|last` | Resume a session |
| `/status` | Session, model, approval, tokens |
| `/config` | Paths and limits |
| `/quit` | Exit |

Anything else goes to the agent. Lines beginning with `!` run directly in the workspace shell.
`Ctrl-C` interrupts a turn; `Ctrl-D` exits. Shell commands and file writes ask for approval by default:
`y` allow once, `n` deny, `a` allow for the rest of the session.

Useful flags:

```sh
python3 main.py --read-only                  # no writes, no shell
python3 main.py --yolo "fix this test"       # auto-approve tools
python3 main.py --lane subagents             # use another configured lane
python3 main.py --model openai/gpt-4.1-mini  # prefer a lane member
python3 main.py --resume last                # resume a session
python3 main.py --list-sessions              # show saved sessions
python3 main.py --plain                      # script-friendly output
```

## Project layout

```text
main.py                 entry point
ohar/
  cli.py                REPL, onboarding, slash commands, sessions, tool wiring
  agent.py              streaming tool loop, context trimming, failover
  providers.py          OpenAI-compatible + Anthropic SSE clients
  balancer.py           lanes, health, weighted/least-busy selection
  subagents.py          parallel read-only task tool (max 8)
  tools.py              confined local tools and approvals
  webtools.py           bounded web fetch/search with SSRF protection
  browser.py            headless Chromium CDP client and browser tools
  mcp.py                stdio MCP JSON-RPC client and tool bridge
  memory.py             workspace-scoped persistent memory
  planner.py            visible update_plan state and tool
  models.py             provider model discovery for /models
  syntax.py             dependency-free fenced-code tokenizer
  skills.py             stdlib skill loader
  sessions.py           atomic JSONL session storage
  config.py             TOML config, presets, env secrets
  prompts.py            system prompts
  types.py              shared contracts
  ui.py, terminal.py    ANSI terminal UI and renderer
  wrapper/              mission control: session state machine, continuity,
                        briefing, handoff provenance, workspace probe, records
  workflows/            goal ledger, decision gates, approval receipts,
                        activity journal, work board, verification units
  tui/                  widget layer: DAG view, diff view, pager, pickers,
                        panels, cell-width math, subagent roster
skills/                 sixteen curated skills + attribution
tests/                  stdlib unittest suite and offline mock provider
```

## Testing

```sh
python3 -m unittest discover -v
```

Tests use a local mock OpenAI/Anthropic server and local HTTP/MCP fixtures; no API key or external
network is needed. The suite covers providers, failover/load balancing, sessions, tool confinement,
the tool loop, eight-way subagent parallelism, web sanitization and SSRF rules, browser protocol
handling, MCP discovery/calls, planning, memory, models, syntax tokenization, and the goal, gate,
journal, board, verification, handoff, and DAG layers.

## Limitations

- Subagents are deliberately read-only and do not share a mutable browser page. Concurrent writers
  would need worktree isolation; OHAR keeps edits with the main agent to avoid silent conflicts.
- Browser automation requires a Chromium/Chrome binary and controls one page and the main frame; new
  tabs, iframe internals, and shadow-DOM internals are not adopted. Navigation settles on readiness
  instead of full network-idle, and screenshots capture the viewport. JS-heavy pages can still fail on
  bot challenges or navigation timeouts; the tool reports the failure instead of guessing.
- Web and browser tools block private and reserved destinations unless
  `allow_private = true`; the browser checks the starting and final page URL. A public page can still
  load its own subresources under Chromium's network rules, so avoid untrusted pages on machines with
  sensitive internal services. Search providers may rate-limit or challenge automated requests.
- MCP is stdio-only. MCP servers are user-configured local processes and are trusted to run with your
  permissions. Tool annotations are advisory; side-effecting tools ask for approval by default.
- Memory is explicit and workspace-scoped. It is not a vector index and is not injected automatically
  into every prompt; the agent calls `recall` when relevant.
- Context trimming drops the oldest complete turns before a request; the full transcript remains in
  the session file. There is no model-generated summary compaction.
- Approval prompts require an interactive terminal. In one-shot/headless mode, write, shell, browser
  actions that require approval, memory writes, and side-effecting MCP calls are denied unless
  `--yolo` or `approval = "auto"` is set.
- Provider compatibility follows the OpenAI Chat Completions API. Very strict servers may reject
  `stream_options`; OHAR retries once without it.

## Credits

OHAR is an independent project. It adapts sixteen skills, re-implements the mission-control ideas of
the upstream `wrapper`, `workflows`, and `tui` layers as stdlib-only OHAR-native modules, and draws
terminal presentation inspiration from [oh-my-hermes](https://github.com/rlaope/oh-my-hermes) (MIT).
See [NOTICE.md](NOTICE.md) and
[LICENSE](LICENSE). Upstream skill attribution is in [skills/NOTICE.md](skills/NOTICE.md).
