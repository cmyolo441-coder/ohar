"""Provider model discovery for OHAR. Standard library only."""

from __future__ import annotations

import http.client
import json
import re
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import resolve_api_key
from .providers import _NoRedirect, _http_error, _network_error, _redact
from .types import Config, ProviderConfig

_MAX_BODY = 2 * 1024 * 1024
_MAX_TEXT = 240
_ANTHROPIC_VERSION = "2023-06-01"
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ModelListError(Exception):
    """Model discovery failed for one provider or for every provider."""


@dataclass(frozen=True)
class ModelInfo:
    id: str
    provider: str
    owned_by: str = ""
    context_length: int | None = None
    max_output_tokens: int | None = None


def _sanitize(value: Any, limit: int = _MAX_TEXT) -> str:
    text = _CONTROL.sub("", str(value)).replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())[:limit]


def _safe(value: Any, key: str | None, limit: int = _MAX_TEXT) -> str:
    return _sanitize(_redact(str(value), key), limit)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _read_bounded(stream: Any, limit: int = _MAX_BODY) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(65536, limit - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ModelListError(f"response body exceeds {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise ModelListError("invalid JSON response") from None


def _parse_models(payload: Any, provider: str, anthropic: bool) -> list[ModelInfo]:
    if anthropic:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ModelListError("unexpected Anthropic models response")
        entries = payload["data"]
    elif isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = None
        for field in ("data", "models"):
            if isinstance(payload.get(field), list):
                entries = payload[field]
                break
        if entries is None:
            raise ModelListError("unexpected models response")
    else:
        raise ModelListError("unexpected models response")

    found: list[ModelInfo] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            continue
        model_id = _sanitize(raw_id, 200)
        if not model_id or model_id in seen:
            continue
        if anthropic:
            kind = entry.get("type")
            if isinstance(kind, str) and kind.strip() and kind.strip().lower() != "model":
                continue
            seen.add(model_id)
            display = entry.get("display_name")
            owned_by = (
                _sanitize(display, 120)
                if isinstance(display, str) and display.strip()
                else "anthropic"
            )
            found.append(ModelInfo(model_id, provider, owned_by))
            continue
        seen.add(model_id)
        owned = entry.get("owned_by")
        context = _positive_int(entry.get("context_length"))
        output = _positive_int(entry.get("max_output_tokens"))
        if output is None:
            output = _positive_int(entry.get("max_completion_tokens"))
        found.append(
            ModelInfo(
                id=model_id,
                provider=provider,
                owned_by=_sanitize(owned, 120) if isinstance(owned, str) else "",
                context_length=context,
                max_output_tokens=output,
            )
        )
    return sorted(found, key=lambda item: item.id)


def _fetch(provider: ProviderConfig, timeout: float) -> list[ModelInfo]:
    base_url = (provider.base_url or "").strip().rstrip("/")
    if not base_url:
        raise ModelListError("base_url is not configured")
    kind = (provider.type or "").strip().lower()
    if kind not in {"openai", "anthropic"}:
        raise ModelListError(f"unsupported provider type '{_sanitize(kind, 40)}'")

    key = resolve_api_key(provider)
    url = f"{base_url}/models"
    headers = {"Accept": "application/json", "User-Agent": "ohar/0.1"}
    headers.update({str(name): str(value) for name, value in provider.headers.items()})
    if kind == "openai":
        if key:
            headers["Authorization"] = f"Bearer {key}"
    else:
        if key:
            headers["x-api-key"] = key
        headers["anthropic-version"] = _ANTHROPIC_VERSION
        url += ("&" if "?" in url else "?") + "limit=1000"

    request = urllib.request.Request(url, headers=headers, method="GET")
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = _read_bounded(response)
    except urllib.error.HTTPError as exc:
        error = _http_error(exc, key)
        raise ModelListError(
            f"{error.kind} error: {_safe(error.message, key)} (HTTP {error.status})"
        ) from None
    except ValueError as exc:
        raise ModelListError(f"invalid request: {_safe(exc, key)}") from None
    except (
        urllib.error.URLError,
        ssl.SSLError,
        socket.timeout,
        OSError,
        http.client.HTTPException,
    ) as exc:
        error = _network_error(exc, key)
        raise ModelListError(f"{error.kind} error: {_safe(error.message, key)}") from None

    return _parse_models(_parse_json(raw), provider.name, anthropic=kind == "anthropic")


def list_models(
    config: Config, provider: str | None = None, *, timeout: float = 20.0
) -> list[ModelInfo]:
    """List models for one provider, or all configured providers when None."""
    if timeout <= 0:
        raise ModelListError("timeout must be positive")
    if provider is not None:
        provider_config = config.providers.get(provider)
        if provider_config is None:
            available = ", ".join(sorted(config.providers)) or "none"
            raise ModelListError(
                f"Unknown provider '{_sanitize(provider, 64)}'. Available: {available}."
            )
        try:
            return _fetch(provider_config, timeout)
        except ModelListError as exc:
            raise ModelListError(f"{_sanitize(provider, 64)}: {_sanitize(exc)}") from None

    if not config.providers:
        raise ModelListError("No providers configured.")
    found: list[ModelInfo] = []
    errors: list[str] = []
    for name in sorted(config.providers):
        try:
            found.extend(_fetch(config.providers[name], timeout))
        except ModelListError as exc:
            errors.append(f"{_sanitize(name, 64)}: {_sanitize(exc)}")
    if len(errors) == len(config.providers):
        details = "; ".join(errors[:8])
        if len(errors) > 8:
            details += f"; and {len(errors) - 8} more"
        raise ModelListError(f"All providers failed: {details}"[:1400])
    return sorted(found, key=lambda item: (item.provider, item.id))
