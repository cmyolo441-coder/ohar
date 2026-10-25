"""Model Context Protocol (MCP) stdio client.

Spawns configured MCP servers, performs the JSON-RPC 2.0 handshake, discovers
their tools, and exposes them as regular OHAR :class:`~ohar.types.Tool`
objects.  The implementation is deliberately self-contained and stdlib-only:
one reader thread per server dispatches responses to waiting requests through
queues, so a slow or noisy server can never deadlock the caller.
"""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .types import ApprovalRequest, McpServerConfig, Tool, ToolContext, ToolResult

__all__ = ["McpManager"]

_PROTOCOL_VERSION = "2024-11-05"
_MAX_LINE_CHARS = 8 * 1024 * 1024
_MAX_RESULT_CHARS = 200_000
_MAX_DESCRIPTION_CHARS = 2_000
_MAX_ERROR_CHARS = 1_500
_MAX_WARNING_CHARS = 600
_MAX_STDERR_LINES = 200
_MAX_STDERR_CHARS = 1_000
_MAX_PAGES = 20
_MAX_TOOL_NAME = 64

_ANSI_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SECRET_KEY_RE = re.compile(r"(?i)(?:token|secret|passw|key|auth|credential)")


def _sanitize(value: Any) -> str:
    """Strip ANSI escapes and control characters, keeping newlines/tabs."""
    text = _ANSI_RE.sub("", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_RE.sub("", text)


def _truncate(text: str, limit: int) -> str:
    """Bound *text* to *limit* characters, keeping head and tail."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    marker = f"\n… [truncated {len(text) - limit} chars] …\n"
    if len(marker) >= limit:
        return text[:limit]
    keep = limit - len(marker)
    head = (keep * 3) // 5
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), default=str
        )
    except (TypeError, ValueError):
        return str(value)


def _one_line(text: str) -> str:
    return " ".join(_sanitize(text).split())


def _name_component(value: Any, fallback: str) -> str:
    cleaned = _NAME_RE.sub("_", str(value)).strip("_")
    return cleaned or fallback


def _base_tool_name(server: str, tool: str) -> str:
    base = f"mcp__{_name_component(server, 'server')}__{_name_component(tool, 'tool')}"
    return base[:_MAX_TOOL_NAME]


def _unique_tool_name(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    index = 2
    while True:
        suffix = f"_{index}"
        candidate = base[: _MAX_TOOL_NAME - len(suffix)] + suffix
        if candidate not in used:
            return candidate
        index += 1


def _default_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


class _McpError(Exception):
    """Base class for MCP transport, protocol, and RPC failures."""


class _TransportError(_McpError):
    """The subprocess or its pipes failed."""


class _ProtocolError(_McpError):
    """The server violated the expected JSON-RPC/MCP shape."""


class _RpcError(_McpError):
    """The server returned a well-formed JSON-RPC error response."""

    def __init__(self, message: str, *, code: Any = None) -> None:
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        message = super().__str__()
        if self.code is None:
            return message
        return f"{message} (code {self.code})"


@dataclass(slots=True)
class _ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    read_only: bool
    annotations: dict[str, Any]


class _Connection:
    """One MCP server subprocess and its JSON-RPC session."""

    def __init__(
        self,
        config: McpServerConfig,
        *,
        client_name: str,
        client_version: str,
    ) -> None:
        self.config = config
        self._client_name = client_name
        self._client_version = client_version
        self._status = "failed"
        self.error: str | None = "not started"
        self.protocol_version: str | None = None
        self.specs: list[_ToolSpec] = []

        self._status_lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._pending: dict[Any, queue.Queue[tuple[str, Any]]] = {}
        self._next_id = 0

        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr: deque[str] = deque(maxlen=_MAX_STDERR_LINES)
        self._stdout_noise: deque[str] = deque(maxlen=20)

        self._stopped = False
        self._closing = False
        self._posix_session = False
        self._pgid: int | None = None

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def status_snapshot(self) -> str:
        with self._status_lock:
            return self._status

    def _set_status(self, status: str, error: str | None) -> None:
        with self._status_lock:
            self._status = status
            self.error = error

    def _timeout(self) -> float:
        try:
            value = float(self.config.timeout)
        except (TypeError, ValueError):
            value = 20.0
        return max(0.01, value)

    def _redact(self, text: str) -> str:
        secrets = [
            str(value)
            for key, value in self.config.env.items()
            if _SECRET_KEY_RE.search(str(key)) and len(str(value)) >= 4
        ]
        for secret in sorted(set(secrets), key=len, reverse=True):
            text = text.replace(secret, "[redacted]")
        return text

    def _decorate(self, reason: Any) -> str:
        text = _sanitize(reason).strip() or "unknown error"
        stderr_lines = [line for line in list(self._stderr)[-3:] if line.strip()]
        if stderr_lines:
            text += " (stderr: " + _truncate(" | ".join(stderr_lines), 600) + ")"
        text = self._redact(text)
        return _truncate(text, _MAX_ERROR_CHARS)

    def clean_error(self, reason: Any) -> str:
        return self._decorate(reason)

    def refresh_error(self) -> None:
        """Append drained stderr to a failure recorded before stderr finished."""
        with self._status_lock:
            if self._status != "failed" or not self.error:
                return
            current = self.error
        if "(stderr:" in current:
            return
        stderr_lines = [line for line in list(self._stderr)[-3:] if line.strip()]
        if not stderr_lines:
            return
        updated = current + " (stderr: " + _truncate(" | ".join(stderr_lines), 600) + ")"
        with self._status_lock:
            if self._status == "failed":
                self.error = _truncate(
                    self._redact(_sanitize(updated)), _MAX_ERROR_CHARS
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._set_status("starting", None)
        self._spawn()
        result = self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": self._client_name,
                    "version": self._client_version,
                },
            },
        )
        if not isinstance(result, dict):
            raise _ProtocolError("initialize returned a non-object result")
        version = result.get("protocolVersion")
        if isinstance(version, str) and version:
            self.protocol_version = version
        self._notify("notifications/initialized", {})
        self.specs = self._discover_tools()
        with self._status_lock:
            if self._status == "starting":
                self._status = "ready"
                self.error = None

    def _spawn(self) -> None:
        command = [str(self.config.command), *[str(arg) for arg in self.config.args]]
        env = dict(os.environ)
        for key, value in self.config.env.items():
            env[str(key)] = str(value)
        cwd: str | None = None
        if self.config.cwd:
            cwd = os.path.expanduser(self.config.cwd)

        kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "env": env,
        }
        if cwd is not None:
            kwargs["cwd"] = cwd
        if os.name == "posix":
            kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(command, **kwargs)
        except FileNotFoundError:
            if cwd is not None and not os.path.isdir(cwd):
                raise _TransportError(
                    f"working directory does not exist: {cwd}"
                ) from None
            raise _TransportError(
                f"command not found: {self.config.command!r}"
            ) from None
        except PermissionError:
            if cwd is not None and not os.path.isdir(cwd):
                raise _TransportError(
                    f"working directory does not exist: {cwd}"
                ) from None
            raise _TransportError(
                f"cannot execute command: {self.config.command!r}"
            ) from None
        except OSError as exc:
            detail = exc.strerror or str(exc)
            raise _TransportError(
                f"failed to start command {self.config.command!r}: {detail}"
            ) from None

        self._proc = proc
        self._posix_session = os.name == "posix"
        self._pgid = proc.pid if self._posix_session else None
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"ohar-mcp-{_name_component(self.config.name, 'server')}-reader",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            name=f"ohar-mcp-{_name_component(self.config.name, 'server')}-stderr",
            daemon=True,
        )
        self._reader.start()
        self._stderr_thread.start()

    def close(self) -> None:
        """Mark the connection closed and terminate its process group."""
        with self._close_lock:
            with self._status_lock:
                if self._status == "closed":
                    return
                self._closing = True
                self._status = "closed"
            self._fail_pending(f"MCP server '{self.config.name}' closed")
            self.stop()

    def stop(self) -> None:
        """Terminate the child, then drain and join its helper threads."""
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        proc = self._proc
        if proc is not None:
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass

            exited = False
            try:
                proc.wait(timeout=0.25)
                exited = True
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

            if not exited:
                self._signal_process(force=False)
                try:
                    proc.wait(timeout=0.5)
                    exited = True
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass

            if not exited:
                self._signal_process(force=True)
                try:
                    proc.wait(timeout=0.5)
                except Exception:
                    pass

            # The leader may have exited while descendants still hold the
            # inherited pipes.  Clean up the whole session/process group.
            if os.name == "posix" and self._posix_session and self._pgid is not None:
                self._signal_group(signal.SIGTERM)
                time.sleep(0.05)
                self._signal_group(signal.SIGKILL)

        self._join_threads()
        self.refresh_error()

        if proc is not None:
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream is not None and not stream.closed:
                        stream.close()
                except Exception:
                    pass

    def _join_threads(self) -> None:
        current = threading.current_thread()
        for thread in (self._reader, self._stderr_thread):
            if thread is None or thread is current:
                continue
            try:
                thread.join(timeout=1.0)
            except RuntimeError:
                pass

    def _signal_group(self, sig: int) -> None:
        if (
            os.name != "posix"
            or not self._posix_session
            or self._pgid is None
        ):
            return
        try:
            os.killpg(self._pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _signal_process(self, *, force: bool) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is not None:
                return
        except Exception:
            pass

        if os.name == "posix" and self._posix_session and self._pgid is not None:
            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                os.killpg(self._pgid, sig)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass

        try:
            if force:
                proc.kill()
            else:
                proc.terminate()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def _send(self, payload: dict[str, Any]) -> None:
        try:
            line = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ) + "\n"
        except (TypeError, ValueError) as exc:
            raise _McpError(f"request is not JSON-serializable: {exc}") from None

        with self._write_lock:
            if self._closing or self._stopped or self.status_snapshot() == "closed":
                raise _TransportError(f"MCP server '{self.config.name}' is closed")
            proc = self._proc
            stdin = proc.stdin if proc is not None else None
            if proc is None or stdin is None:
                raise _TransportError(
                    f"MCP server '{self.config.name}' has no input pipe"
                )
            try:
                stdin.write(line)
                stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise _TransportError(
                    f"cannot write to MCP server '{self.config.name}': {exc}"
                ) from None

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        timeout = self._timeout()
        with self._status_lock:
            status = self._status
            proc = self._proc
        if status not in ("starting", "ready") or proc is None:
            raise _TransportError(self.error or f"MCP server '{self.config.name}' is not running")
        try:
            if proc.poll() is not None:
                self._fail(f"MCP server '{self.config.name}' process is not running")
                raise _TransportError(self.error or "server process is not running")
        except _TransportError:
            raise
        except Exception:
            pass

        with self._id_lock:
            self._next_id += 1
            request_id = self._next_id
        waiter: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = waiter

        with self._status_lock:
            if self._status not in ("starting", "ready"):
                with self._pending_lock:
                    self._pending.pop(request_id, None)
                raise _TransportError(
                    self.error or f"MCP server '{self.config.name}' is unavailable"
                )

        try:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )
        except _TransportError as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            self._fail(str(exc))
            raise
        except _McpError:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise

        try:
            kind, payload = waiter.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            reason = f"{method} timed out after {timeout:g}s"
            self._fail(reason)
            raise _TransportError(reason) from None

        if kind == "error":
            raise _TransportError(str(payload))

        response = payload
        if not isinstance(response, dict):
            reason = f"{method} returned a malformed response"
            self._fail(reason)
            raise _TransportError(reason)

        if "error" in response:
            error = response.get("error")
            if not isinstance(error, dict):
                reason = f"{method} returned a malformed JSON-RPC error"
                self._fail(reason)
                raise _TransportError(reason)
            message = error.get("message")
            if not isinstance(message, str) or not message:
                message = "unknown server error"
            detail = error.get("data")
            if detail is not None:
                message += f" ({_truncate(_compact_json(detail), 400)})"
            raise _RpcError(
                f"{method} failed: {_sanitize(message)}",
                code=error.get("code"),
            )

        return response.get("result")

    def _read_loop(self) -> None:
        proc = self._proc
        stream = proc.stdout if proc is not None else None
        if stream is None:
            self._fail("stdout pipe is unavailable")
            return

        try:
            while True:
                try:
                    line = stream.readline(_MAX_LINE_CHARS + 1)
                except (OSError, ValueError) as exc:
                    if self._closing or self.status_snapshot() == "closed":
                        return
                    self._fail(f"failed reading stdout: {exc}")
                    return

                if not line:
                    if self._closing or self.status_snapshot() == "closed":
                        return
                    self._wait_for_stderr()
                    self._fail(self._eof_reason())
                    return

                if len(line) > _MAX_LINE_CHARS:
                    snippet = _sanitize(line[:300]).strip()
                    self._stdout_noise.append(snippet)
                    self._fail(
                        f"protocol line exceeded the {_MAX_LINE_CHARS}-character limit"
                    )
                    return

                text = line.strip()
                if not text:
                    continue

                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    snippet = _sanitize(line).strip()
                    self._stdout_noise.append(snippet[:500])
                    self._fail(
                        "invalid JSON on stdout: "
                        + repr(_truncate(snippet, 300))
                    )
                    return

                if not self._handle_payload(payload):
                    return
        except Exception as exc:  # never let the reader die silently
            if not self._closing and self.status_snapshot() != "closed":
                self._fail(f"stdout reader failed: {exc}")

    def _handle_payload(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            self._fail("protocol message is not a JSON object")
            return False

        version = payload.get("jsonrpc")
        if version is not None and version != "2.0":
            self._fail(f"unsupported JSON-RPC version: {version!r}")
            return False

        if "method" in payload:
            # Notifications are ignored; server-initiated requests are not
            # supported (we advertise no client capabilities).
            return True

        if "id" not in payload:
            self._fail("protocol response is missing an id")
            return False

        has_result = "result" in payload
        has_error = "error" in payload
        if has_result == has_error:
            self._fail(
                "protocol response must contain exactly one of result or error"
            )
            return False

        response_id = payload.get("id")
        with self._pending_lock:
            waiter = self._pending.pop(response_id, None)
            if waiter is None and isinstance(response_id, (int, str)):
                waiter = self._pending.pop(str(response_id), None)
            if waiter is None and isinstance(response_id, str):
                try:
                    waiter = self._pending.pop(int(response_id), None)
                except (TypeError, ValueError):
                    waiter = None

        # Unknown ids are late responses to timed-out requests; ignore them.
        if waiter is not None:
            waiter.put(("response", payload))
        return True

    def _eof_reason(self) -> str:
        proc = self._proc
        code: int | None = None
        if proc is not None:
            try:
                code = proc.poll()
            except Exception:
                code = None
        if code is None:
            return "server closed stdout unexpectedly"
        return f"server exited unexpectedly (exit code {code})"

    def _wait_for_stderr(self, timeout: float = 0.1) -> None:
        thread = self._stderr_thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            try:
                thread.join(timeout=timeout)
            except RuntimeError:
                pass

    def _stderr_loop(self) -> None:
        proc = self._proc
        stream = proc.stderr if proc is not None else None
        if stream is None:
            return
        try:
            while True:
                line = stream.readline(4096)
                if not line:
                    return
                text = _sanitize(line).strip()
                if text:
                    self._stderr.append(_truncate(text, _MAX_STDERR_CHARS))
        except Exception:
            return

    def _fail(self, reason: str) -> None:
        decorated = self._decorate(reason)
        with self._status_lock:
            if self._status in ("failed", "closed"):
                return
            self._status = "failed"
            self.error = decorated

        with self._pending_lock:
            waiters = list(self._pending.values())
            self._pending.clear()
        for waiter in waiters:
            waiter.put(("error", decorated))
        self._signal_process(force=False)

    def _fail_pending(self, reason: str) -> None:
        with self._pending_lock:
            waiters = list(self._pending.values())
            self._pending.clear()
        for waiter in waiters:
            waiter.put(("error", reason))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _discover_tools(self) -> list[_ToolSpec]:
        specs: list[_ToolSpec] = []
        cursor: str | None = None
        for _page in range(_MAX_PAGES):
            params: dict[str, Any] = {}
            if cursor is not None:
                params["cursor"] = cursor
            result = self._request("tools/list", params)
            if not isinstance(result, dict):
                raise _ProtocolError("tools/list returned a non-object result")
            raw_tools = result.get("tools")
            if not isinstance(raw_tools, list):
                raise _ProtocolError(
                    "tools/list result is missing a 'tools' array"
                )
            for raw in raw_tools:
                spec = self._parse_tool(raw)
                if spec is not None:
                    specs.append(spec)

            next_cursor = result.get("nextCursor")
            if next_cursor is None or next_cursor == "":
                break
            if not isinstance(next_cursor, str):
                raise _ProtocolError("tools/list returned an invalid cursor")
            if next_cursor == cursor:
                raise _ProtocolError("tools/list returned a repeated cursor")
            cursor = next_cursor
        return specs

    def _parse_tool(self, raw: Any) -> _ToolSpec | None:
        if not isinstance(raw, dict):
            return None
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            return None

        description = raw.get("description")
        if isinstance(description, str):
            description = _sanitize(description).strip()
        if not isinstance(description, str) or not description:
            description = (
                f"MCP tool '{_one_line(name)}' from server "
                f"'{_one_line(self.config.name)}'."
            )
        description = _truncate(
            self._redact(description), _MAX_DESCRIPTION_CHARS
        )

        input_schema = raw.get("inputSchema")
        if isinstance(input_schema, dict) and input_schema.get("type") == "object":
            parameters = input_schema
        else:
            parameters = _default_schema()

        annotations = raw.get("annotations")
        if not isinstance(annotations, dict):
            annotations = {}

        return _ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            read_only=annotations.get("readOnlyHint") is True,
            annotations=annotations,
        )

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------
    def call_tool(
        self, spec: _ToolSpec, arguments: Any, ctx: ToolContext
    ) -> ToolResult:
        started = time.monotonic()
        meta: dict[str, Any] = {
            "server": self.config.name,
            "tool": spec.name,
        }
        if spec.annotations:
            meta["annotations"] = spec.annotations

        def finish(ok: bool, content: str, **extra: Any) -> ToolResult:
            meta.update(extra)
            meta["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
            clean = _truncate(
                self._redact(_sanitize(content)), _MAX_RESULT_CHARS
            )
            return ToolResult(ok, clean, dict(meta))

        try:
            if ctx.read_only and not spec.read_only:
                return finish(
                    False,
                    f"Read-only mode: MCP tool '{spec.name}' requires write "
                    "access; call denied and nothing was sent.",
                    denied=True,
                )

            if not isinstance(arguments, dict):
                return finish(
                    False,
                    f"MCP tool '{spec.name}' arguments must be a JSON object.",
                )
            try:
                json.dumps(arguments, ensure_ascii=False)
            except (TypeError, ValueError):
                return finish(
                    False,
                    f"MCP tool '{spec.name}' arguments must be JSON-serializable.",
                )

            if self.status_snapshot() != "ready":
                return finish(
                    False,
                    f"MCP server '{self.config.name}' is unavailable: "
                    f"{self.error or 'not ready'}",
                    unavailable=True,
                )

            if not spec.read_only:
                detail = (
                    f"Server: {self.config.name}\n"
                    f"Tool: {spec.name}\n"
                    "Arguments: "
                    + _truncate(_compact_json(arguments), 8_000)
                )
                request = ApprovalRequest(
                    action="mcp_call",
                    summary=_truncate(
                        _one_line(
                            f"Call MCP tool '{spec.name}' on server "
                            f"'{self.config.name}'"
                        ),
                        500,
                    ),
                    detail=_truncate(
                        self._redact(_sanitize(detail)), 8_000
                    ),
                    risk="medium",
                )
                if not ctx.request_approval(request):
                    return finish(
                        False,
                        "Permission denied by user; MCP tool "
                        f"'{spec.name}' was not called.",
                        denied=True,
                    )

            if self.status_snapshot() != "ready":
                return finish(
                    False,
                    f"MCP server '{self.config.name}' became unavailable: "
                    f"{self.error or 'not ready'}",
                    unavailable=True,
                )

            result = self._request(
                "tools/call",
                {"name": spec.name, "arguments": arguments},
            )
            if not isinstance(result, dict):
                reason = "tools/call returned a non-object result"
                self._fail(reason)
                return finish(
                    False,
                    f"MCP tool '{spec.name}' failed: {reason}",
                    protocol_error=True,
                )

            content = result.get("content")
            if content is not None and not isinstance(content, list):
                reason = "tools/call returned a non-array content field"
                self._fail(reason)
                return finish(
                    False,
                    f"MCP tool '{spec.name}' failed: {reason}",
                    protocol_error=True,
                )

            text = self._format_call_content(content, result)
            if result.get("isError") is True:
                return finish(
                    False,
                    text or f"MCP tool '{spec.name}' returned an error.",
                    is_error=True,
                )
            return finish(True, text)
        except _RpcError as exc:
            return finish(
                False,
                f"MCP tool '{spec.name}' on server '{self.config.name}' "
                f"failed: {exc}",
            )
        except _TransportError as exc:
            return finish(
                False,
                f"MCP server '{self.config.name}' failed during "
                f"'{spec.name}': {exc}",
                unavailable=True,
            )
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            return finish(
                False,
                f"MCP tool '{spec.name}' failed unexpectedly: {message}",
            )

    def _format_call_content(
        self, content: Any, result: dict[str, Any]
    ) -> str:
        parts: list[str] = []
        if isinstance(content, list):
            for item in content:
                parts.append(self._format_content_item(item))

        if (
            "structuredContent" in result
            and result.get("structuredContent") is not None
        ):
            parts.append(
                "structuredContent:\n"
                + _compact_json(result.get("structuredContent"))
            )
        return "\n".join(part for part in parts if part != "")

    def _safe_label(self, value: Any, limit: int = 300) -> str:
        if value is None:
            return ""
        text = _one_line(_sanitize(value))
        return _truncate(self._redact(text), limit)

    def _format_content_item(self, item: Any) -> str:
        if not isinstance(item, dict):
            return "[invalid content item omitted]"
        kind = item.get("type")
        if kind == "text":
            text = item.get("text")
            if not isinstance(text, str):
                text = "" if text is None else _compact_json(text)
            return text

        if kind in ("image", "audio"):
            bits = [str(kind)]
            mime = self._safe_label(item.get("mimeType"), 200)
            uri = self._safe_label(item.get("uri"), 500)
            if mime:
                bits.append(f"mimeType={mime}")
            if uri:
                bits.append(f"uri={uri}")
            bits.append("data omitted")
            return "[" + "; ".join(bits) + "]"

        if kind == "resource":
            resource = item.get("resource")
            if not isinstance(resource, dict):
                resource = {}
            uri = self._safe_label(
                resource.get("uri") or item.get("uri"), 500
            )
            mime = self._safe_label(
                resource.get("mimeType") or item.get("mimeType"), 200
            )
            bits = ["resource"]
            if uri:
                bits.append(f"uri={uri}")
            if mime:
                bits.append(f"mimeType={mime}")
            bits.append("content omitted")
            return "[" + "; ".join(bits) + "]"

        if kind in ("resource_link", "resourceLink"):
            uri = self._safe_label(item.get("uri"), 500)
            mime = self._safe_label(item.get("mimeType"), 200)
            name = self._safe_label(item.get("name"), 200)
            bits = [str(kind)]
            if name:
                bits.append(f"name={name}")
            if uri:
                bits.append(f"uri={uri}")
            if mime:
                bits.append(f"mimeType={mime}")
            return "[" + "; ".join(bits) + "]"

        label = self._safe_label(kind, 80) or "unknown"
        return f"[{label} content omitted]"


class McpManager:
    """Own one subprocess connection per configured MCP server."""

    def __init__(
        self,
        servers: list[McpServerConfig],
        *,
        client_name: str = "ohar",
        client_version: str = "0.2.0",
    ) -> None:
        self._client_name = client_name
        self._client_version = client_version
        self._servers = list(servers)
        self._connections = [
            _Connection(
                server,
                client_name=client_name,
                client_version=client_version,
            )
            for server in self._servers
        ]
        self._lifecycle = threading.RLock()
        self._started = False
        self._closed = False
        self._warnings: list[str] = []
        self._tools: list[tuple[_Connection, Tool]] = []

    def start(self) -> list[str]:
        """Start every enabled server, returning concise failure warnings."""
        with self._lifecycle:
            if self._closed:
                return ["MCP manager is closed."]
            if self._started:
                return list(self._warnings)

            self._started = True
            warnings: list[str] = []
            for connection in self._connections:
                if not connection.config.enabled:
                    continue
                try:
                    connection.start()
                    if connection.status_snapshot() != "ready":
                        raise _TransportError(
                            connection.error or "server did not become ready"
                        )
                except Exception as exc:
                    if connection.status_snapshot() not in ("failed", "closed"):
                        message = str(exc) or type(exc).__name__
                        connection._fail(message)
                    try:
                        connection.stop()
                    except Exception:
                        pass
                    try:
                        connection.refresh_error()
                    except Exception:
                        pass
                    reason = connection.error or connection.clean_error(
                        str(exc) or type(exc).__name__
                    )
                    warnings.append(
                        _truncate(
                            _one_line(
                                f"MCP server '{connection.config.name}' "
                                f"failed: {reason}"
                            ),
                            _MAX_WARNING_CHARS,
                        )
                    )

            self._warnings = warnings
            self._build_tools()
            return list(warnings)

    @property
    def tools(self) -> list[Tool]:
        """Tools exposed by servers that are currently ready."""
        return [
            tool
            for connection, tool in list(self._tools)
            if connection.status_snapshot() == "ready"
        ]

    def summaries(self) -> list[dict[str, Any]]:
        """Return one status record per configured server."""
        summaries: list[dict[str, Any]] = []
        for connection in list(self._connections):
            config = connection.config
            if not config.enabled:
                status = "disabled"
                error: str | None = None
                tool_count = 0
            else:
                raw_status = connection.status_snapshot()
                if raw_status == "starting":
                    status = "failed"
                    error = connection.error or "startup in progress"
                elif raw_status in ("ready", "failed", "closed"):
                    status = raw_status
                    error = None if status == "ready" else connection.error
                else:
                    status = "failed"
                    error = connection.error or "server did not start"
                tool_count = (
                    len(connection.specs) if status == "ready" else 0
                )
            summaries.append(
                {
                    "name": config.name,
                    "status": status,
                    "tool_count": tool_count,
                    "error": (
                        _truncate(_sanitize(error), _MAX_ERROR_CHARS)
                        if error
                        else None
                    ),
                    "command": _truncate(_sanitize(config.command), 200),
                }
            )
        return summaries

    def close(self) -> None:
        """Stop all servers. Safe to call repeatedly and concurrently."""
        with self._lifecycle:
            if self._closed:
                return
            for connection in self._connections:
                if not connection.config.enabled:
                    continue
                try:
                    connection.close()
                except Exception:
                    pass
            self._closed = True

    def _build_tools(self) -> None:
        used: set[str] = set()
        entries: list[tuple[_Connection, Tool]] = []
        for connection in self._connections:
            if (
                not connection.config.enabled
                or connection.status_snapshot() != "ready"
            ):
                continue
            for spec in connection.specs:
                name = _unique_tool_name(
                    _base_tool_name(connection.config.name, spec.name),
                    used,
                )
                used.add(name)
                tool = Tool(
                    name=name,
                    description=spec.description,
                    parameters=spec.parameters,
                    handler=self._make_handler(connection, spec),
                    read_only=spec.read_only,
                    destructive=not spec.read_only,
                )
                entries.append((connection, tool))
        self._tools = entries

    @staticmethod
    def _make_handler(connection: _Connection, spec: _ToolSpec):
        def handler(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
            return connection.call_tool(spec, arguments, ctx)

        return handler
