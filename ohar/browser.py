"""Headless Chromium automation for OHAR using the DevTools Protocol.

The module intentionally depends only on the Python standard library. It
launches a private, temporary-profile Chromium instance, speaks the Chrome
DevTools Protocol over a small WebSocket client, and exposes a fixed set of
browser tools through :mod:`ohar.types`. Model supplied selectors are embedded
as JSON literals; arbitrary JavaScript supplied by the model is never run.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

from .tools import ToolError, resolve_path
from .types import ApprovalRequest, BrowserConfig, Tool, ToolContext, ToolResult
from .webtools import validate_url

__all__ = ["BrowserManager"]

MAX_OUTPUT_CHARS = 12_000
MAX_TEXT_EXCERPT = 4_000
MAX_SELECTOR_CHARS = 1_000
MAX_TYPE_CHARS = 20_000
MAX_SCREENSHOT_PATH_CHARS = 1_000
MAX_ELEMENTS = 40
MAX_WS_MESSAGE = 64 * 1024 * 1024
MAX_HTTP_JSON = 2 * 1024 * 1024
MAX_SCREENSHOT_BYTES = 32 * 1024 * 1024
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
CHROMIUM_NAMES = ("chromium", "chromium-browser", "google-chrome", "chrome")
LOCAL_DEVTOOLS_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
SENSITIVE_ENV_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_KEY|PRIVATE_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)$",
    re.IGNORECASE,
)

_KEY_TABLE: dict[str, tuple[str, str, int, str]] = {
    "enter": ("Enter", "Enter", 13, "\r"),
    "return": ("Enter", "Enter", 13, "\r"),
    "tab": ("Tab", "Tab", 9, ""),
    "escape": ("Escape", "Escape", 27, ""),
    "esc": ("Escape", "Escape", 27, ""),
    "backspace": ("Backspace", "Backspace", 8, ""),
    "delete": ("Delete", "Delete", 46, ""),
    "home": ("Home", "Home", 36, ""),
    "end": ("End", "End", 35, ""),
    "pageup": ("PageUp", "PageUp", 33, ""),
    "pagedown": ("PageDown", "PageDown", 34, ""),
    "arrowleft": ("ArrowLeft", "ArrowLeft", 37, ""),
    "arrowup": ("ArrowUp", "ArrowUp", 38, ""),
    "arrowright": ("ArrowRight", "ArrowRight", 39, ""),
    "arrowdown": ("ArrowDown", "ArrowDown", 40, ""),
    "space": (" ", "Space", 32, " "),
}


class _BrowserError(Exception):
    """An expected browser failure that is safe to show to the model."""

    kind = "browser"

    def __init__(self, message: str, *, kind: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if kind:
            self.kind = kind


class _StartupError(_BrowserError):
    kind = "startup"


class _TransportError(_BrowserError):
    kind = "transport"


class _CommandTimeout(_TransportError):
    kind = "timeout"


class _ArgsError(ValueError):
    """A tool argument was missing or malformed."""


def _sanitize(text: str) -> str:
    return CONTROL_RE.sub("", ANSI_RE.sub("", text))


def _one_line(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", _sanitize(text).replace("\xa0", " ")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _collapse(text: str) -> str:
    lines: list[str] = []
    blank = False
    for raw in text.replace("\xa0", " ").splitlines():
        line = re.sub(r"[ \t\f\v]+", " ", raw).strip()
        if line:
            lines.append(line)
            blank = False
        elif lines and not blank:
            lines.append("")
            blank = True
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _bounded(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = f"\n[... output truncated {len(text) - limit} chars ...]"
    keep = max(0, limit - len(marker))
    return text[:keep].rstrip() + marker, True


def _remaining(deadline: float, label: str = "Browser command timed out.") -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise _CommandTimeout(label)
    return left


def _is_local_devtools_url(url: str, expected_port: int | None = None) -> bool:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if parts.scheme.lower() != "ws":
        return False
    host = (parts.hostname or "").lower()
    if host not in LOCAL_DEVTOOLS_HOSTS:
        return False
    try:
        port = parts.port
    except ValueError:
        return False
    if port is None:
        port = 80
    return expected_port is None or port == expected_port


class _WebSocket:
    """Minimal RFC 6455 client sufficient for CDP traffic."""

    def __init__(self, url: str, timeout: float) -> None:
        self._sock: socket.socket | None = None
        self._buffer = b""
        self._closed = False
        try:
            parts = urllib.parse.urlsplit(url)
        except ValueError:
            raise _TransportError("Browser debugging URL is invalid.") from None
        if parts.scheme.lower() != "ws":
            raise _TransportError("Browser debugging URL is not a WebSocket URL.")
        host = (parts.hostname or "").lower()
        if host not in LOCAL_DEVTOOLS_HOSTS:
            raise _TransportError("Browser debugging endpoint is not local.")
        try:
            port = parts.port or 80
        except ValueError:
            raise _TransportError("Browser debugging URL has an invalid port.") from None
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        deadline = time.monotonic() + max(0.001, float(timeout))
        try:
            self._sock = socket.create_connection(
                (host, port), timeout=_remaining(deadline, "Timed out connecting to Chromium.")
            )
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._handshake(host, port, target, deadline)
        except _BrowserError:
            self._close_socket()
            raise
        except OSError as exc:
            self._close_socket()
            reason = getattr(exc, "strerror", None) or "connection failed"
            raise _TransportError(f"Could not connect to Chromium: {_one_line(reason, 160)}") from None

    @property
    def closed(self) -> bool:
        return self._closed or self._sock is None

    def _close_socket(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _handshake(self, host: str, port: int, target: str, deadline: float) -> None:
        assert self._sock is not None
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        host_header = f"[{host}]" if ":" in host else host
        if port != 80:
            host_header += f":{port}"
        request = (
            f"GET {target} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "User-Agent: OHAR-Browser/1\r\n"
            "\r\n"
        ).encode("ascii", errors="strict")
        try:
            self._sock.settimeout(_remaining(deadline, "Timed out connecting to Chromium."))
            self._sock.sendall(request)
        except socket.timeout:
            raise _CommandTimeout("Timed out connecting to Chromium.") from None
        except OSError as exc:
            reason = getattr(exc, "strerror", None) or "connection failed"
            raise _TransportError(f"Could not connect to Chromium: {_one_line(reason, 160)}") from None

        raw = b""
        while b"\r\n\r\n" not in raw:
            if len(raw) > 32_768:
                raise _TransportError("Chromium sent an invalid WebSocket handshake.")
            chunk = self._recv_socket(deadline, "Timed out connecting to Chromium.")
            raw += chunk
        head, _, leftover = raw.partition(b"\r\n\r\n")
        self._buffer = leftover
        try:
            lines = head.decode("iso-8859-1").split("\r\n")
        except UnicodeDecodeError:
            raise _TransportError("Chromium sent an invalid WebSocket handshake.") from None
        if not lines or not re.match(r"HTTP/1\.[01] 101(?:\s|$)", lines[0]):
            raise _TransportError("Chromium rejected the WebSocket handshake.")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
        if headers.get("upgrade", "").lower() != "websocket":
            raise _TransportError("Chromium sent an invalid WebSocket handshake.")
        if "upgrade" not in headers.get("connection", "").lower():
            raise _TransportError("Chromium sent an invalid WebSocket handshake.")
        expected = base64.b64encode(
            hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if headers.get("sec-websocket-accept", "") != expected:
            raise _TransportError("Chromium sent an invalid WebSocket handshake.")
        self._closed = False

    def _recv_socket(self, deadline: float, timeout_message: str) -> bytes:
        if self._sock is None or self._closed:
            raise _TransportError("Browser connection is closed.")
        try:
            self._sock.settimeout(_remaining(deadline, timeout_message))
            chunk = self._sock.recv(65_536)
        except socket.timeout:
            raise _CommandTimeout(timeout_message) from None
        except OSError as exc:
            reason = getattr(exc, "strerror", None) or "connection failed"
            raise _TransportError(f"Browser connection failed: {_one_line(reason, 160)}") from None
        if not chunk:
            self._closed = True
            raise _TransportError("Browser closed the connection.")
        return chunk

    def _recv_exact(self, size: int, deadline: float) -> bytes:
        while len(self._buffer) < size:
            self._buffer += self._recv_socket(deadline, "Browser command timed out.")
        data, self._buffer = self._buffer[:size], self._buffer[size:]
        return data

    def _send_frame(self, opcode: int, payload: bytes, deadline: float) -> None:
        if self._sock is None or self._closed:
            raise _TransportError("Browser connection is closed.")
        length = len(payload)
        header = bytearray()
        header.append(0x80 | (opcode & 0x0F))
        if length < 126:
            header.append(0x80 | length)
        elif length <= 0xFFFF:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = os.urandom(4)
        header.extend(mask)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        try:
            self._sock.settimeout(_remaining(deadline, "Browser command timed out."))
            self._sock.sendall(bytes(header) + masked)
        except socket.timeout:
            raise _CommandTimeout("Browser command timed out.") from None
        except OSError as exc:
            reason = getattr(exc, "strerror", None) or "connection failed"
            raise _TransportError(f"Browser connection failed: {_one_line(reason, 160)}") from None

    def send_text(self, text: str, deadline: float) -> None:
        self._send_frame(0x1, text.encode("utf-8"), deadline)

    def _read_frame(self, deadline: float) -> tuple[bool, int, bytes]:
        first = self._recv_exact(2, deadline)
        fin = bool(first[0] & 0x80)
        if first[0] & 0x70:
            raise _TransportError("Browser sent an unsupported WebSocket frame.")
        opcode = first[0] & 0x0F
        masked = bool(first[1] & 0x80)
        length = first[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2, deadline))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8, deadline))[0]
        if length > MAX_WS_MESSAGE:
            raise _TransportError("Browser sent an oversized protocol message.")
        mask = self._recv_exact(4, deadline) if masked else b""
        payload = self._recv_exact(length, deadline)
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return fin, opcode, payload

    def recv_text(self, deadline: float) -> str:
        fragments: list[bytes] = []
        message_opcode: int | None = None
        total = 0
        while True:
            fin, opcode, payload = self._read_frame(deadline)
            if opcode == 0x8:
                try:
                    self._send_frame(0x8, b"\x03\xe8", time.monotonic() + 0.2)
                except _BrowserError:
                    pass
                self._closed = True
                raise _TransportError("Browser closed the WebSocket connection.")
            if opcode == 0x9:
                self._send_frame(0xA, payload, deadline)
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x2):
                if message_opcode is not None:
                    raise _TransportError("Browser sent an invalid WebSocket frame sequence.")
                message_opcode = opcode
                fragments = []
                total = 0
            elif opcode == 0x0:
                if message_opcode is None:
                    raise _TransportError("Browser sent an invalid WebSocket frame sequence.")
            else:
                raise _TransportError("Browser sent an unsupported WebSocket frame.")
            total += len(payload)
            if total > MAX_WS_MESSAGE:
                raise _TransportError("Browser sent an oversized protocol message.")
            fragments.append(payload)
            if not fin:
                continue
            if message_opcode == 0x2:
                raise _TransportError("Browser sent an unsupported binary message.")
            try:
                return b"".join(fragments).decode("utf-8")
            except UnicodeDecodeError:
                raise _TransportError("Browser sent malformed UTF-8 data.") from None

    def close(self) -> None:
        if self._closed and self._sock is None:
            return
        sock = self._sock
        if sock is not None and not self._closed:
            try:
                self._send_frame(0x8, b"\x03\xe8", time.monotonic() + 0.2)
            except _BrowserError:
                pass
        self._closed = True
        self._close_socket()


class _CDP:
    """Synchronous DevTools Protocol request/response client."""

    def __init__(self, ws: _WebSocket) -> None:
        self.ws = ws
        self._next_id = 0

    def call(self, method: str, params: dict[str, Any] | None = None, *, timeout: float) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        payload: dict[str, Any] = {"id": request_id, "method": method}
        if params:
            payload["params"] = params
        try:
            encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            raise _BrowserError("Browser command could not be encoded.") from None
        deadline = time.monotonic() + max(0.05, float(timeout))
        self.ws.send_text(encoded, deadline)
        while True:
            raw = self.ws.recv_text(deadline)
            try:
                message = json.loads(raw)
            except ValueError:
                raise _TransportError("Browser sent an invalid protocol message.") from None
            if not isinstance(message, dict):
                continue
            if message.get("id") != request_id:
                # Events and late responses from abandoned commands are ignored.
                continue
            if "error" in message:
                error = message.get("error")
                detail = ""
                if isinstance(error, dict):
                    detail = str(error.get("message") or "")
                detail = _one_line(detail or "unknown error", 300)
                lowered = detail.lower()
                if "target closed" in lowered or "session closed" in lowered or "detached" in lowered:
                    raise _TransportError("Browser page connection was closed.")
                raise _BrowserError(f"{method} failed: {detail}")
            result = message.get("result")
            return result if isinstance(result, dict) else {}

    def evaluate(
        self,
        expression: str,
        *,
        timeout: float,
        user_gesture: bool = False,
    ) -> Any:
        params: dict[str, Any] = {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": False,
        }
        if user_gesture:
            params["userGesture"] = True
        result = self.call("Runtime.evaluate", params, timeout=timeout)
        details = result.get("exceptionDetails")
        if isinstance(details, dict):
            text = ""
            exception = details.get("exception")
            if isinstance(exception, dict):
                text = str(exception.get("description") or exception.get("value") or "")
            text = text or str(details.get("text") or "page script failed")
            raise _BrowserError(f"Page evaluation failed: {_one_line(text, 300)}")
        value = result.get("result")
        if not isinstance(value, dict) or "value" not in value:
            raise _BrowserError("Page did not return a usable value.")
        return value.get("value")


def _http_json(port: int, path: str, timeout: float) -> Any:
    deadline = time.monotonic() + max(0.05, float(timeout))
    connection = http.client.HTTPConnection(
        "127.0.0.1", port, timeout=_remaining(deadline, "Timed out talking to Chromium.")
    )
    try:
        connection.request(
            "GET",
            path,
            headers={"Accept": "application/json", "User-Agent": "OHAR-Browser/1"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise _TransportError("Chromium debugging endpoint returned an error.")
        body = response.read(MAX_HTTP_JSON + 1)
    except _BrowserError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        reason = getattr(exc, "strerror", None) or "request failed"
        raise _TransportError(
            f"Chromium debugging endpoint is unavailable: {_one_line(str(reason), 160)}"
        ) from None
    finally:
        connection.close()
    if len(body) > MAX_HTTP_JSON:
        raise _TransportError("Chromium debugging endpoint returned too much data.")
    try:
        return json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        raise _TransportError("Chromium debugging endpoint returned invalid JSON.") from None


def _list_targets(port: int, timeout: float) -> tuple[int, str | None]:
    data = _http_json(port, "/json/list", timeout)
    if not isinstance(data, list):
        raise _TransportError("Chromium debugging endpoint returned an unexpected target list.")
    pages: list[dict[str, Any]] = [
        item for item in data if isinstance(item, dict) and item.get("type") == "page"
    ]
    chosen: str | None = None
    for item in pages:
        url = item.get("webSocketDebuggerUrl")
        if not isinstance(url, str) or not _is_local_devtools_url(url, port):
            continue
        if chosen is None or str(item.get("url") or "") == "about:blank":
            chosen = url
    return len(pages), chosen


def _browser_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not SENSITIVE_ENV_RE.search(key)
    }
    return env


def _find_chromium(configured: str | None) -> str:
    if configured:
        raw = os.path.expanduser(configured.strip())
        if not raw:
            raise _StartupError("browser.chromium_path is empty.")
        if os.sep in raw or (os.altsep and os.altsep in raw):
            candidate = Path(raw)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
            raise _StartupError(
                f"Configured Chromium path is not executable: {_one_line(raw, 300)}"
            )
        found = shutil.which(raw)
        if found:
            return found
        raise _StartupError(f"Configured Chromium command was not found: {_one_line(raw, 120)}")
    for name in CHROMIUM_NAMES:
        found = shutil.which(name)
        if found:
            return found
    raise _StartupError(
        "Chromium was not found. Install Chromium or set browser.chromium_path."
    )


def _screenshots_dir() -> Path:
    raw = os.environ.get("OHAR_HOME")
    home = Path(raw).expanduser() if raw else Path.home() / ".ohar"
    return home / "screenshots"


def _new_screenshot_path() -> Path:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = _screenshots_dir()
    path = directory / f"browser-{stamp}.png"
    counter = 2
    while path.exists():
        path = directory / f"browser-{stamp}-{counter}.png"
        counter += 1
    return path


def _atomic_write_private(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise _BrowserError(f"Cannot create screenshot directory: {exc.strerror or 'error'}") from None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
    except OSError as exc:
        raise _BrowserError(f"Cannot create screenshot file: {exc.strerror or 'error'}") from None
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise _BrowserError(f"Cannot write screenshot: {exc.strerror or 'error'}") from None
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _display_path(ctx: ToolContext, path: Path) -> str:
    try:
        workspace = Path(ctx.workspace).expanduser().resolve(strict=False)
        return path.relative_to(workspace).as_posix()
    except (OSError, RuntimeError, ValueError):
        return str(path)


def _require_str(
    args: dict[str, Any],
    name: str,
    *,
    maximum: int,
    allow_empty: bool = False,
    strip: bool = False,
) -> str:
    value = args.get(name)
    if value is None:
        raise _ArgsError(f"Missing required argument '{name}'.")
    if not isinstance(value, str):
        raise _ArgsError(f"Argument '{name}' must be a string.")
    if strip:
        value = value.strip()
    if not allow_empty and not value.strip():
        raise _ArgsError(f"Argument '{name}' cannot be empty.")
    if len(value) > maximum:
        raise _ArgsError(f"Argument '{name}' exceeds {maximum} characters.")
    if "\x00" in value:
        raise _ArgsError(f"Argument '{name}' cannot contain NUL bytes.")
    return value


def _optional_str(args: dict[str, Any], name: str, *, maximum: int) -> str | None:
    value = args.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _ArgsError(f"Argument '{name}' must be a string.")
    if len(value) > maximum:
        raise _ArgsError(f"Argument '{name}' exceeds {maximum} characters.")
    return value


def _arg_bool(args: dict[str, Any], name: str, default: bool = False) -> bool:
    value = args.get(name, default)
    if not isinstance(value, bool):
        raise _ArgsError(f"Argument '{name}' must be a boolean.")
    return value


def _arg_error(exc: BaseException) -> ToolResult:
    return ToolResult.failure(_one_line(str(exc), 500), error="invalid_args")


_SNAPSHOT_JS = r"""(() => {
  const REF = 'data-ohar-ref';
  const MAX = 40;
  const clean = (value, limit) => String(value == null ? '' : value)
    .replace(/\s+/g, ' ').trim().slice(0, limit);
  try {
    document.querySelectorAll('[' + REF + ']').forEach((el) => el.removeAttribute(REF));
  } catch (err) {}
  const cssEscape = (value) => {
    if (window.CSS && typeof window.CSS.escape === 'function') {
      return window.CSS.escape(String(value));
    }
    return String(value).replace(/[^a-zA-Z0-9_-]/g, (ch) => '\\' + ch);
  };
  const quoteAttr = (value) => String(value).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    if (!style) return false;
    if (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse') {
      return false;
    }
    if (parseFloat(style.opacity || '1') === 0) return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 1 && rect.height > 1;
  };
  const labelFor = (el) => {
    const aria = el.getAttribute('aria-label') || el.getAttribute('title');
    if (aria) return clean(aria, 120);
    const tag = el.tagName.toLowerCase();
    const linked = el.labels && el.labels.length ? (el.labels[0].innerText || el.labels[0].textContent) : '';
    if (tag === 'input') {
      const type = (el.getAttribute('type') || 'text').toLowerCase();
      const own = type === 'submit' || type === 'button' || type === 'reset' ? el.value : '';
      return clean(own || el.getAttribute('placeholder') || el.getAttribute('alt') || linked || el.getAttribute('name') || type, 120);
    }
    if (tag === 'textarea') {
      return clean(el.getAttribute('placeholder') || linked || el.getAttribute('name') || 'textarea', 120);
    }
    if (tag === 'select') {
      const selected = el.selectedOptions && el.selectedOptions.length ? el.selectedOptions[0] : null;
      return clean(selected ? selected.textContent : (linked || el.getAttribute('name') || 'select'), 120);
    }
    return clean(el.innerText || el.textContent || el.getAttribute('alt') || '', 120);
  };
  const selectorFor = (el, index) => {
    const tag = el.tagName.toLowerCase();
    if (el.id) {
      const candidate = '#' + cssEscape(el.id);
      if (candidate.length <= 180) {
        try {
          if (document.querySelectorAll(candidate).length === 1) return candidate;
        } catch (err) {}
      }
    }
    const name = el.getAttribute('name');
    if (name && !/[\x00-\x1f\x7f]/.test(name)) {
      const candidate = tag + '[name="' + quoteAttr(name) + '"]';
      if (candidate.length <= 180) {
        try {
          if (document.querySelectorAll(candidate).length === 1) return candidate;
        } catch (err) {}
      }
    }
    const ref = 'ohar-' + (index + 1);
    el.setAttribute(REF, ref);
    return '[' + REF + '="' + ref + '"]';
  };
  const nodes = Array.from(document.querySelectorAll('a[href], button, input, select, textarea'));
  const elements = [];
  for (const el of nodes) {
    if (elements.length >= MAX) break;
    const tag = el.tagName.toLowerCase();
    if (tag === 'input' && (el.getAttribute('type') || '').toLowerCase() === 'hidden') continue;
    if (!visible(el)) continue;
    const index = elements.length;
    const item = {
      index: index + 1,
      tag: tag,
      text: labelFor(el),
      selector: selectorFor(el, index)
    };
    if (tag === 'input') {
      item.type = (el.getAttribute('type') || 'text').toLowerCase();
    }
    elements.push(item);
  }
  let text = '';
  try {
    text = document.body ? (document.body.innerText || document.body.textContent || '') : '';
  } catch (err) {}
  return {
    title: document.title || '',
    url: location.href,
    text: String(text).slice(0, 60000),
    elements: elements
  };
})()"""


def _click_expression(selector: str) -> str:
    encoded = json.dumps(selector, ensure_ascii=True)
    return r"""(() => {
  const selector = %s;
  let el;
  try {
    el = document.querySelector(selector);
  } catch (err) {
    return {ok: false, error: 'Invalid CSS selector.'};
  }
  if (!el) return {ok: false, error: 'No element matches that selector.'};
  if (el.disabled) return {ok: false, error: 'Element is disabled.'};
  const rect = el.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return {ok: false, error: 'Element is not visible.'};
  }
  try {
    if (typeof el.scrollIntoView === 'function') {
      el.scrollIntoView({block: 'center', inline: 'center'});
    }
    if (typeof el.focus === 'function') el.focus({preventScroll: true});
    if (typeof el.click === 'function') {
      el.click();
    } else {
      el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, view: window}));
    }
  } catch (err) {
    return {ok: false, error: 'Element could not be clicked.'};
  }
  const label = el.getAttribute('aria-label') || el.innerText || '';
  return {
    ok: true,
    tag: el.tagName.toLowerCase(),
    text: String(label).replace(/\s+/g, ' ').trim().slice(0, 120)
  };
})()""" % encoded


def _type_expression(selector: str, text: str) -> str:
    encoded_selector = json.dumps(selector, ensure_ascii=True)
    encoded_value = json.dumps(text, ensure_ascii=True)
    return r"""(() => {
  const selector = %s;
  const value = %s;
  let el;
  try {
    el = document.querySelector(selector);
  } catch (err) {
    return {ok: false, error: 'Invalid CSS selector.'};
  }
  if (!el) return {ok: false, error: 'No element matches that selector.'};
  if (el.disabled || el.readOnly) {
    return {ok: false, error: 'Element is disabled or read-only.'};
  }
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || 'text').toLowerCase();
  if (tag === 'input' && ['file', 'checkbox', 'radio', 'button', 'submit', 'reset', 'image', 'hidden'].includes(type)) {
    return {ok: false, error: 'Element does not accept typed text.'};
  }
  const editable = tag === 'input' || tag === 'textarea' || el.isContentEditable;
  if (!editable) {
    return {ok: false, error: 'Element is not a text input, textarea, or editable element.'};
  }
  const rect = el.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return {ok: false, error: 'Element is not visible.'};
  }
  try {
    if (typeof el.scrollIntoView === 'function') {
      el.scrollIntoView({block: 'center', inline: 'center'});
    }
    el.focus({preventScroll: true});
    if (tag === 'input' || tag === 'textarea') {
      const proto = tag === 'textarea'
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, 'value');
      if (setter && typeof setter.set === 'function') {
        setter.set.call(el, value);
      } else {
        el.value = value;
      }
    } else {
      el.textContent = value;
    }
    el.dispatchEvent(new Event('input', {bubbles: true, composed: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
  } catch (err) {
    return {ok: false, error: 'Element value could not be set.'};
  }
  return {ok: true, tag: tag};
})()""" % (encoded_selector, encoded_value)


def _scroll_expression(direction: str) -> str:
    encoded = json.dumps(direction, ensure_ascii=True)
    return r"""(() => {
  const direction = %s;
  const amount = Math.max((window.innerHeight || 600) * 0.8, 320);
  if (direction === 'top') {
    window.scrollTo(0, 0);
  } else if (direction === 'bottom') {
    const doc = document.scrollingElement || document.documentElement || document.body;
    const height = doc ? (doc.scrollHeight || 0) : 0;
    window.scrollTo(0, height);
  } else if (direction === 'up') {
    window.scrollBy(0, -amount);
  } else {
    window.scrollBy(0, amount);
  }
  return {
    ok: true,
    x: Math.round(window.scrollX || 0),
    y: Math.round(window.scrollY || 0)
  };
})()""" % encoded


class BrowserManager:
    """Own one private Chromium process and expose browser tools lazily."""

    def __init__(self, config: BrowserConfig) -> None:
        if not isinstance(config, BrowserConfig):
            raise TypeError("BrowserManager requires a BrowserConfig.")
        timeout = config.timeout
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ValueError("browser.timeout must be a positive number.")
        if config.chromium_path is not None and not isinstance(config.chromium_path, str):
            raise ValueError("browser.chromium_path must be a string or None.")
        self._config = config
        self._timeout = float(timeout)
        self._lock = threading.RLock()
        self._tools: list[Tool] = [] if not config.enabled else self._build_tools()
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None
        self._profile: Path | None = None
        self._pgid: int | None = None
        self._port: int | None = None
        self._ws: _WebSocket | None = None
        self._cdp: _CDP | None = None
        self._pages = 0
        self._last_error = ""

    # ------------------------------------------------------------------ API

    @property
    def tools(self) -> list[Tool]:
        if not self._config.enabled:
            return []
        return list(self._tools)

    def summaries(self) -> list[dict[str, Any]]:
        summary: dict[str, Any] = {
            "name": "chromium",
            "running": False,
            "pages": 0,
            "error": _one_line(self._last_error, 500),
        }
        acquired = self._lock.acquire(timeout=0.05)
        if not acquired:
            process = self._process
            if process is not None and process.poll() is None:
                summary["running"] = True
                summary["pages"] = self._pages
            return [summary]
        try:
            process = self._process
            if process is not None and process.poll() is None:
                summary["running"] = True
                if self._port is not None:
                    try:
                        pages, _ = _list_targets(
                            self._port, min(1.0, max(0.1, self._timeout))
                        )
                        self._pages = pages
                        summary["pages"] = pages
                    except Exception:
                        summary["pages"] = self._pages
            else:
                summary["pages"] = 0
            return [summary]
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._shutdown_locked()
            except Exception:
                pass

    # -------------------------------------------------------------- lifecycle

    def _shutdown_locked(self) -> None:
        ws, self._ws = self._ws, None
        self._cdp = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        process, self._process = self._process, None
        pgid, self._pgid = self._pgid, None
        profile, self._profile = self._profile, None
        self._port = None
        self._pages = 0
        if process is not None:
            self._terminate_process_group(process, pgid)
        if profile is not None:
            for _attempt in range(3):
                shutil.rmtree(profile, ignore_errors=True)
                if not profile.exists():
                    break
                time.sleep(0.05)

    @staticmethod
    def _terminate_process_group(
        process: subprocess.Popen[bytes], pgid: int | None = None
    ) -> None:
        if pgid is None:
            try:
                pgid = os.getpgid(process.pid)
            except (ProcessLookupError, OSError):
                pgid = None
        try:
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                if pgid is not None:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                else:
                    try:
                        process.kill()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
            if pgid is not None:
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    try:
                        os.killpg(pgid, 0)
                    except ProcessLookupError:
                        break
                    except PermissionError:
                        break
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        break
                    time.sleep(0.05)
        except Exception:
            pass
        finally:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1.0)
            except Exception:
                pass

    def _ensure_locked(self) -> _CDP:
        if self._closed:
            raise _BrowserError("Browser manager is closed.", kind="closed")
        if self._cdp is not None:
            alive = (
                self._process is not None
                and self._process.poll() is None
                and self._ws is not None
                and not self._ws.closed
            )
            if alive:
                return self._cdp
            self._shutdown_locked()
        elif self._process is not None:
            self._shutdown_locked()
        return self._start_locked()

    def _start_locked(self) -> _CDP:
        executable = _find_chromium(self._config.chromium_path)
        try:
            profile = Path(tempfile.mkdtemp(prefix="ohar-browser-"))
        except OSError as exc:
            raise _StartupError(
                f"Could not create a temporary Chromium profile: {exc.strerror or 'error'}"
            ) from None
        try:
            os.chmod(profile, 0o700)
        except OSError:
            pass
        headless_flags = ["--headless=new"] if self._config.headless else []
        command = [
            executable,
            *headless_flags,
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-sync",
            "--no-first-run",
            "--no-default-browser-check",
            "--password-store=basic",
            "--mute-audio",
            "--hide-scrollbars",
            "--window-size=1280,800",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "about:blank",
        ]
        deadline = time.monotonic() + self._timeout
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
                env=_browser_env(),
            )
        except OSError as exc:
            shutil.rmtree(profile, ignore_errors=True)
            reason = getattr(exc, "strerror", None) or "could not execute the binary"
            message = f"Chromium could not be started: {_one_line(reason, 160)}"
            self._last_error = message
            raise _StartupError(message) from None
        self._process = process
        self._profile = profile
        # start_new_session=True makes the child PID the process-group ID.
        self._pgid = process.pid
        try:
            port = self._wait_for_port(process, profile, deadline)
            self._port = port
            target = self._wait_for_page(process, port, deadline)
            cdp = self._connect_cdp(process, target, deadline)
            self._cdp = cdp
            self._ws = cdp.ws
            self._last_error = ""
            return cdp
        except _BrowserError as exc:
            message = _one_line(exc.message, 500)
            self._last_error = message
            self._shutdown_locked()
            raise _StartupError(message, kind=exc.kind) from None
        except Exception:
            message = "Chromium failed to start unexpectedly."
            self._last_error = message
            self._shutdown_locked()
            raise _StartupError(message) from None

    def _wait_for_port(
        self, process: subprocess.Popen[bytes], profile: Path, deadline: float
    ) -> int:
        marker = profile / "DevToolsActivePort"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise _StartupError(
                    "Chromium exited during startup"
                    + (f" (code {process.returncode})" if process.returncode is not None else "")
                    + ". Check browser.chromium_path and browser.headless."
                )
            try:
                raw = marker.read_text(encoding="ascii", errors="ignore")
            except (FileNotFoundError, OSError, UnicodeError):
                raw = ""
            lines = raw.splitlines()
            if lines:
                try:
                    port = int(lines[0].strip())
                except ValueError:
                    port = 0
                if 0 < port <= 65535:
                    return port
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if process.poll() is not None:
            raise _StartupError("Chromium exited during startup.")
        raise _StartupError(
            f"Chromium did not start within {self._timeout:g} seconds. "
            "Check browser.chromium_path and whether it can launch as this user."
        )

    def _wait_for_page(
        self, process: subprocess.Popen[bytes], port: int, deadline: float
    ) -> str:
        last_error = "Chromium started but did not expose a page."
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise _StartupError("Chromium exited before exposing a page.")
            try:
                pages, target = _list_targets(
                    port, min(2.0, max(0.1, deadline - time.monotonic()))
                )
                self._pages = pages
                if target:
                    return target
                last_error = "Chromium has no debuggable page yet."
            except _BrowserError as exc:
                last_error = exc.message
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        raise _StartupError(
            f"Chromium did not expose a page within {self._timeout:g} seconds"
            + (f": {_one_line(last_error, 200)}" if last_error else ".")
        )

    def _connect_cdp(
        self,
        process: subprocess.Popen[bytes],
        target: str,
        deadline: float,
    ) -> _CDP:
        last_error = "Could not connect to the Chromium page."
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise _StartupError("Chromium exited before the page could be controlled.")
            ws: _WebSocket | None = None
            try:
                remaining = deadline - time.monotonic()
                ws = _WebSocket(target, max(0.1, remaining))
                cdp = _CDP(ws)
                cdp.call(
                    "Page.enable",
                    timeout=max(0.1, min(self._timeout, deadline - time.monotonic())),
                )
                return cdp
            except _BrowserError as exc:
                last_error = exc.message
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
            except Exception:
                last_error = "Could not connect to the Chromium page."
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        raise _StartupError(
            f"Could not control Chromium within {self._timeout:g} seconds"
            + (f": {_one_line(last_error, 200)}" if last_error else ".")
        )

    # ----------------------------------------------------------- tool helpers

    def _run(
        self,
        ctx: ToolContext,
        name: str,
        *,
        read_only: bool,
        summary: str,
        detail: str,
        risk: str,
        action: Any,
    ) -> ToolResult:
        if self._closed:
            return ToolResult.failure("Browser manager is closed.", error="closed")
        if not self._config.enabled:
            return ToolResult.failure("Browser tools are disabled.", error="disabled")
        if ctx.read_only and not read_only:
            return ToolResult.failure(
                f"Read-only mode: {name} is disabled.", denied=True, error="denied"
            )
        if self._config.require_approval and not (ctx.read_only and read_only):
            request = ApprovalRequest(
                action=name,
                summary=_one_line(summary, 300),
                detail=_one_line(detail, 2_000),
                risk=risk,
            )
            try:
                approved = ctx.request_approval(request)
            except Exception:
                approved = False
            if not approved:
                return ToolResult.failure(
                    f"{name} denied by approval policy.", denied=True, error="denied"
                )
        with self._lock:
            try:
                cdp = self._ensure_locked()
                result = action(cdp)
                if result.ok:
                    self._last_error = ""
                return result
            except _TransportError as exc:
                message = _one_line(exc.message, 500)
                self._last_error = message
                self._shutdown_locked()
                return ToolResult.failure(
                    message + " The browser will restart on the next browser call.",
                    error=exc.kind,
                )
            except _BrowserError as exc:
                message = _one_line(exc.message, 1_000)
                self._last_error = message
                return ToolResult.failure(message, error=exc.kind)
            except (ValueError, ToolError) as exc:
                message = _one_line(str(exc), 500)
                self._last_error = message
                return ToolResult.failure(message, error="invalid_args")
            except Exception:
                message = "Unexpected browser failure; the browser was restarted."
                self._last_error = message
                self._shutdown_locked()
                return ToolResult.failure(message, error="unexpected")

    def _settle(self, cdp: _CDP, *, seconds: float = 1.2) -> None:
        time.sleep(0.12)
        deadline = time.monotonic() + min(seconds, max(0.2, self._timeout))
        while time.monotonic() < deadline:
            try:
                state = cdp.evaluate(
                    "document.readyState",
                    timeout=min(self._timeout, max(0.3, deadline - time.monotonic())),
                )
            except _TransportError:
                raise
            except _BrowserError:
                time.sleep(0.05)
                continue
            if state == "complete":
                return
            time.sleep(0.05)

    def _state(self, cdp: _CDP) -> dict[str, Any]:
        try:
            value = cdp.evaluate(_SNAPSHOT_JS, timeout=self._timeout)
        except _BrowserError as exc:
            if "context" in exc.message.lower():
                time.sleep(0.1)
                value = cdp.evaluate(_SNAPSHOT_JS, timeout=self._timeout)
            else:
                raise
        if not isinstance(value, dict):
            raise _BrowserError("Could not read the page state.")
        title = _one_line(str(value.get("title") or ""), 200)
        url = _one_line(str(value.get("url") or ""), 2_048)
        text = _collapse(_sanitize(str(value.get("text") or "")))
        elements: list[dict[str, str]] = []
        raw_elements = value.get("elements")
        if isinstance(raw_elements, list):
            for item in raw_elements[:MAX_ELEMENTS]:
                if not isinstance(item, dict):
                    continue
                tag = _one_line(str(item.get("tag") or ""), 20).lower()
                selector = _one_line(str(item.get("selector") or ""), 180)
                if not tag or not selector:
                    continue
                entry = {
                    "index": str(len(elements) + 1),
                    "tag": tag,
                    "text": _one_line(str(item.get("text") or ""), 80),
                    "selector": selector,
                }
                elements.append(entry)
        return {"title": title, "url": url, "text": text, "elements": elements}

    def _state_result(
        self, cdp: _CDP, *, prefix: str = "", **meta: Any
    ) -> ToolResult:
        state = self._state(cdp)
        blocked = self._unsafe_final_url(state["url"])
        if blocked is not None:
            return self._blocked_result(cdp, state["url"], blocked)
        content, truncated = self._format_state(state, prefix=prefix)
        result_meta: dict[str, Any] = {
            "title": state["title"],
            "final_url": state["url"],
            "elements": len(state["elements"]),
            "text_chars": len(state["text"]),
            "truncated": truncated,
        }
        result_meta.update(meta)
        return ToolResult.success(content, **result_meta)

    def _blocked_result(self, cdp: _CDP, url: str, reason: str) -> ToolResult:
        try:
            cdp.call(
                "Page.navigate",
                {"url": "about:blank"},
                timeout=min(self._timeout, 5.0),
            )
        except _BrowserError:
            pass
        return ToolResult.failure(
            f"Browser blocked the current page ({reason}). "
            "Private, reserved, and non-http(s) destinations are disabled. "
            "Set browser.allow_private = true only when you trust the page.",
            blocked=True,
            final_url=_one_line(url, 2_048),
        )

    def _unsafe_final_url(self, url: str) -> str | None:
        if self._config.allow_private:
            return None
        if url == "about:blank":
            return None
        if not url.startswith(("http://", "https://")):
            return "unsupported page URL"
        try:
            validate_url(url, allow_private=False)
        except ValueError as exc:
            return _one_line(str(exc), 300)
        return None

    def _format_state(
        self, state: dict[str, Any], *, prefix: str = ""
    ) -> tuple[str, bool]:
        title = state["title"] or "(untitled)"
        url = state["url"] or "(unknown)"
        elements: list[dict[str, str]] = state["elements"]
        text: str = state["text"]

        head: list[str] = []
        if prefix:
            head.extend([_one_line(prefix, 500), ""])
        head.extend([f"Title: {title}", f"URL: {url}", ""])
        head.append(f"Interactive elements ({len(elements)}):")
        if elements:
            for item in elements:
                label = item["text"] or "(no label)"
                head.append(
                    f"{item['index']}. <{item['tag']}> {label} | selector: {item['selector']}"
                )
        else:
            head.append("(none)")

        fixed = "\n".join(head)
        tail_label = "\n\nText excerpt:\n"
        remaining = MAX_OUTPUT_CHARS - len(fixed) - len(tail_label) - 80
        text_budget = max(0, min(MAX_TEXT_EXCERPT, remaining))
        excerpt = text[:text_budget].rstrip() if text_budget else ""
        text_truncated = len(text) > len(excerpt)
        if not excerpt:
            excerpt = "(no readable text)" if not text else "(omitted to keep output bounded)"
        if text_truncated and text_budget:
            excerpt += "\n[... text truncated ...]"
        content = fixed + tail_label + excerpt
        content, output_truncated = _bounded(content)
        return content, bool(text_truncated or output_truncated)

    def _do_open(self, cdp: _CDP, url: str) -> ToolResult:
        result = cdp.call("Page.navigate", {"url": url}, timeout=self._timeout)
        error_text = result.get("errorText")
        if error_text:
            raise _BrowserError(
                f"Navigation failed: {_one_line(str(error_text), 200)}"
            )
        self._settle(cdp)
        return self._state_result(cdp, action="open", requested_url=url)

    def _do_click(self, cdp: _CDP, selector: str) -> ToolResult:
        value = cdp.evaluate(
            _click_expression(selector), timeout=self._timeout, user_gesture=True
        )
        if not isinstance(value, dict) or not value.get("ok"):
            error = value.get("error") if isinstance(value, dict) else "click failed"
            raise _BrowserError(_one_line(str(error or "click failed"), 300))
        self._settle(cdp)
        tag = _one_line(str(value.get("tag") or "element"), 20)
        return self._state_result(
            cdp,
            prefix=f"Clicked <{tag}>.",
            action="click",
            selector=_one_line(selector, 500),
        )

    def _do_type(
        self, cdp: _CDP, selector: str, text: str, submit: bool
    ) -> ToolResult:
        value = cdp.evaluate(
            _type_expression(selector, text), timeout=self._timeout, user_gesture=True
        )
        if not isinstance(value, dict) or not value.get("ok"):
            error = value.get("error") if isinstance(value, dict) else "typing failed"
            raise _BrowserError(_one_line(str(error or "typing failed"), 300))
        if submit:
            self._press(cdp, "Enter")
        self._settle(cdp)
        tag = _one_line(str(value.get("tag") or "element"), 20)
        verb = "Typed and submitted in" if submit else "Typed into"
        return self._state_result(
            cdp,
            prefix=f"{verb} <{tag}> ({len(text)} chars).",
            action="type",
            selector=_one_line(selector, 500),
            submit=submit,
        )

    def _press(self, cdp: _CDP, key_name: str) -> None:
        key, code, vk, text = _KEY_TABLE[key_name.lower()]
        base = {
            "key": key,
            "code": code,
            "windowsVirtualKeyCode": vk,
            "nativeVirtualKeyCode": vk,
        }
        down = dict(base)
        down["type"] = "rawKeyDown"
        cdp.call("Input.dispatchKeyEvent", down, timeout=self._timeout)
        if text:
            cdp.call(
                "Input.dispatchKeyEvent",
                {**base, "type": "char", "text": text, "unmodifiedText": text},
                timeout=self._timeout,
            )
        cdp.call("Input.dispatchKeyEvent", {**base, "type": "keyUp"}, timeout=self._timeout)

    def _do_press(self, cdp: _CDP, key_name: str) -> ToolResult:
        self._press(cdp, key_name)
        self._settle(cdp)
        key, _, _, _ = _KEY_TABLE[key_name.lower()]
        shown = "Space" if key_name.lower() == "space" else key
        return self._state_result(
            cdp, prefix=f"Pressed {shown}.", action="press", key=shown
        )

    def _do_scroll(self, cdp: _CDP, direction: str) -> ToolResult:
        value = cdp.evaluate(
            _scroll_expression(direction), timeout=self._timeout, user_gesture=True
        )
        if not isinstance(value, dict) or not value.get("ok"):
            raise _BrowserError("The page could not be scrolled.")
        time.sleep(0.05)
        return self._state_result(
            cdp,
            prefix=f"Scrolled {direction}.",
            action="scroll",
            direction=direction,
        )

    def _do_back(self, cdp: _CDP) -> ToolResult:
        history = cdp.call("Page.getNavigationHistory", timeout=self._timeout)
        entries = history.get("entries")
        index = history.get("currentIndex")
        if (
            not isinstance(entries, list)
            or not isinstance(index, int)
            or index <= 0
            or index > len(entries)
        ):
            raise _BrowserError("No previous page in browser history.")
        entry = entries[index - 1]
        entry_id = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(entry_id, int):
            raise _BrowserError("No previous page in browser history.")
        cdp.call("Page.navigateToHistoryEntry", {"entryId": entry_id}, timeout=self._timeout)
        self._settle(cdp)
        return self._state_result(cdp, prefix="Went back.", action="back")

    def _do_screenshot(
        self, cdp: _CDP, path: Path | None, display_name: str | None = None
    ) -> ToolResult:
        state = self._state(cdp)
        blocked = self._unsafe_final_url(state["url"])
        if blocked is not None:
            return self._blocked_result(cdp, state["url"], blocked)
        result = cdp.call(
            "Page.captureScreenshot",
            {"format": "png", "fromSurface": True, "captureBeyondViewport": False},
            timeout=self._timeout,
        )
        encoded = result.get("data")
        if not isinstance(encoded, str) or not encoded:
            raise _BrowserError("Chromium did not return screenshot data.")
        try:
            data = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError):
            raise _BrowserError("Chromium returned invalid screenshot data.") from None
        if len(data) > MAX_SCREENSHOT_BYTES:
            raise _BrowserError("Screenshot is too large to save.")
        if not data.startswith(PNG_SIGNATURE) or len(data) < 24:
            raise _BrowserError("Chromium returned data that is not a PNG image.")
        target = path if path is not None else _new_screenshot_path()
        if target.exists() and target.is_dir():
            raise _BrowserError("Screenshot path is a directory.")
        _atomic_write_private(target, data)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        if path is None:
            try:
                os.chmod(target.parent, 0o700)
            except OSError:
                pass
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        display = _one_line(display_name or str(target), 500)
        return ToolResult.success(
            f"Screenshot saved: {display} (PNG, {len(data)} bytes, {width}x{height}).",
            path=display,
            bytes=len(data),
            width=width,
            height=height,
            format="png",
        )

    # ------------------------------------------------------------- tool setup

    def _build_tools(self) -> list[Tool]:
        return [
            Tool(
                name="browser_open",
                description=(
                    "Open an http(s) URL in a real headless Chromium browser. Returns the "
                    "page title, final URL, readable text excerpt, and numbered interactive "
                    "elements with CSS selectors usable by browser_click and browser_type. "
                    "Private hosts are blocked unless browser.allow_private is enabled."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Absolute http(s) URL to open.",
                            "minLength": 1,
                            "maxLength": 2_048,
                        }
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
                handler=self._handle_open,
                read_only=True,
            ),
            Tool(
                name="browser_snapshot",
                description=(
                    "Inspect the current Chromium page: title, URL, readable text excerpt, "
                    "and up to 40 numbered visible links, buttons, inputs, selects, and "
                    "textareas with CSS selectors. Use the shown selectors with the "
                    "browser_click and browser_type tools."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                handler=self._handle_snapshot,
                read_only=True,
            ),
            Tool(
                name="browser_click",
                description=(
                    "Click the visible element matching a CSS selector in the current page. "
                    "Use selectors reported by browser_snapshot or browser_open. The element "
                    "is scrolled into view first; navigation is allowed to settle briefly."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for a visible page element.",
                            "minLength": 1,
                            "maxLength": MAX_SELECTOR_CHARS,
                        }
                    },
                    "required": ["selector"],
                    "additionalProperties": False,
                },
                handler=self._handle_click,
                read_only=False,
            ),
            Tool(
                name="browser_type",
                description=(
                    "Focus a text input, textarea, or editable element selected by CSS "
                    "selector, set its value, and dispatch input/change events. Set submit "
                    "to true to press Enter afterwards."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for a text input or textarea.",
                            "minLength": 1,
                            "maxLength": MAX_SELECTOR_CHARS,
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to enter (use an empty string to clear).",
                            "maxLength": MAX_TYPE_CHARS,
                        },
                        "submit": {
                            "type": "boolean",
                            "description": "Press Enter after typing (default false).",
                        },
                    },
                    "required": ["selector", "text"],
                    "additionalProperties": False,
                },
                handler=self._handle_type,
                read_only=False,
            ),
            Tool(
                name="browser_press",
                description=(
                    "Press a key in the focused Chromium page element. Supports Enter, Tab, "
                    "Escape, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Backspace, Delete, "
                    "Home, End, PageUp, PageDown, and Space."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Key name, for example Enter, Tab, or ArrowDown.",
                            "minLength": 1,
                            "maxLength": 32,
                        }
                    },
                    "required": ["key"],
                    "additionalProperties": False,
                },
                handler=self._handle_press,
                read_only=False,
            ),
            Tool(
                name="browser_scroll",
                description=(
                    "Scroll the current Chromium page up, down, to the top, or to the "
                    "bottom, then return a fresh page snapshot."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "top", "bottom"],
                            "description": "Scroll direction.",
                        }
                    },
                    "required": ["direction"],
                    "additionalProperties": False,
                },
                handler=self._handle_scroll,
                read_only=True,
            ),
            Tool(
                name="browser_back",
                description=(
                    "Go back one entry in the current Chromium page history and return a "
                    "fresh page snapshot."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                handler=self._handle_back,
                read_only=True,
            ),
            Tool(
                name="browser_screenshot",
                description=(
                    "Capture a PNG screenshot of the current Chromium page. If path is "
                    "omitted, the image is saved under $OHAR_HOME/screenshots. A provided "
                    "path must end in .png and stay inside the workspace. Only the saved "
                    "path and image metadata are returned, never the binary."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Optional workspace-relative or absolute .png path. "
                                "Defaults to $OHAR_HOME/screenshots/browser-<timestamp>.png."
                            ),
                            "minLength": 1,
                            "maxLength": MAX_SCREENSHOT_PATH_CHARS,
                        }
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                handler=self._handle_screenshot,
                read_only=False,
            ),
        ]

    # ---------------------------------------------------------------- handlers

    def _handle_open(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            raw = _require_str(args, "url", maximum=2_048, strip=True)
            url = validate_url(raw, allow_private=self._config.allow_private)
        except (ValueError, _ArgsError) as exc:
            return _arg_error(exc)
        return self._run(
            ctx,
            "browser_open",
            read_only=True,
            summary="Open a page in Chromium",
            detail=url,
            risk="low",
            action=lambda cdp: self._do_open(cdp, url),
        )

    def _handle_snapshot(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self._run(
            ctx,
            "browser_snapshot",
            read_only=True,
            summary="Inspect the current Chromium page",
            detail="Capture page title, text, and interactive elements.",
            risk="low",
            action=lambda cdp: self._state_result(cdp, action="snapshot"),
        )

    def _handle_click(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            selector = _require_str(
                args, "selector", maximum=MAX_SELECTOR_CHARS, strip=False
            )
        except _ArgsError as exc:
            return _arg_error(exc)
        return self._run(
            ctx,
            "browser_click",
            read_only=False,
            summary="Click an element in Chromium",
            detail=f"selector: {selector}",
            risk="medium",
            action=lambda cdp: self._do_click(cdp, selector),
        )

    def _handle_type(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            selector = _require_str(
                args, "selector", maximum=MAX_SELECTOR_CHARS, strip=False
            )
            text = _require_str(
                args,
                "text",
                maximum=MAX_TYPE_CHARS,
                allow_empty=True,
                strip=False,
            )
            submit = _arg_bool(args, "submit", False)
        except _ArgsError as exc:
            return _arg_error(exc)
        return self._run(
            ctx,
            "browser_type",
            read_only=False,
            summary="Type into an element in Chromium",
            detail=(
                f"selector: {selector}; characters: {len(text)}; "
                f"submit: {'true' if submit else 'false'}"
            ),
            risk="medium",
            action=lambda cdp: self._do_type(cdp, selector, text, submit),
        )

    def _handle_press(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            key_name = _require_str(args, "key", maximum=32, strip=True)
            if key_name.lower() not in _KEY_TABLE:
                raise _ArgsError(
                    "Unsupported key. Use Enter, Tab, Escape, an Arrow key, or another "
                    "documented key."
                )
        except _ArgsError as exc:
            return _arg_error(exc)
        return self._run(
            ctx,
            "browser_press",
            read_only=False,
            summary="Press a key in Chromium",
            detail=f"key: {key_name}",
            risk="medium",
            action=lambda cdp: self._do_press(cdp, key_name),
        )

    def _handle_scroll(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            direction = _require_str(args, "direction", maximum=16, strip=True).lower()
            if direction not in {"up", "down", "top", "bottom"}:
                raise _ArgsError("Argument 'direction' must be up, down, top, or bottom.")
        except _ArgsError as exc:
            return _arg_error(exc)
        return self._run(
            ctx,
            "browser_scroll",
            read_only=True,
            summary="Scroll the Chromium page",
            detail=f"direction: {direction}",
            risk="low",
            action=lambda cdp: self._do_scroll(cdp, direction),
        )

    def _handle_back(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self._run(
            ctx,
            "browser_back",
            read_only=True,
            summary="Go back in Chromium history",
            detail="Go back one page in the session history.",
            risk="low",
            action=self._do_back,
        )

    def _handle_screenshot(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target: Path | None = None
        display = "default screenshots directory"
        try:
            raw = _optional_str(args, "path", maximum=MAX_SCREENSHOT_PATH_CHARS)
            if raw is not None:
                if not raw.strip():
                    raise _ArgsError("Argument 'path' cannot be empty.")
                target = resolve_path(ctx, raw, for_write=True)
                if target.suffix.lower() != ".png":
                    raise _ArgsError("Screenshot path must end in .png.")
                if target.exists() and target.is_dir():
                    raise _ArgsError("Screenshot path is a directory.")
                display = _display_path(ctx, target)
        except (ValueError, ToolError) as exc:
            return _arg_error(exc)
        return self._run(
            ctx,
            "browser_screenshot",
            read_only=False,
            summary="Capture a Chromium screenshot",
            detail=f"Save a PNG to {display}",
            risk="medium",
            action=lambda cdp: self._do_screenshot(
                cdp, target, display if target is not None else None
            ),
        )
