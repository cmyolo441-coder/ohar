"""Safe, bounded web fetch and search tools. Standard library only."""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from .types import ApprovalRequest, Tool, ToolContext, ToolResult

__all__ = ["make_web_tools"]

MAX_URL_CHARS = 2048
MAX_REDIRECTS = 5
MAX_LINKS = 40
MAX_QUERY_CHARS = 500
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
SKIP_TAGS = frozenset(
    {
        "script", "style", "noscript", "template", "svg", "canvas", "iframe",
        "object", "nav", "header", "footer", "aside", "form", "button",
        "select", "option", "label", "dialog", "menu",
    }
)
BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "hr", "section", "article", "main", "blockquote",
        "pre", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr", "ul", "ol",
        "dl", "dt", "dd", "figure", "figcaption",
    }
)
BLOCK_MARKERS = (
    "anomaly-modal", "unusual traffic", "bots use duckduckgo",
    "challenge-form", "captcha",
)
_NAT64_NET = ipaddress.ip_network("64:ff9b::/32")  # RFC 6052 well-known + local-use
_IPV4_COMPAT_NET = ipaddress.ip_network("::/96")
_SIXTOFOUR_NET = ipaddress.ip_network("2002::/16")
_TEREDO_NET = ipaddress.ip_network("2001::/32")
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class _Failure(Exception):
    def __init__(self, message: str, **meta: Any) -> None:
        super().__init__(message)
        self.message = message
        self.meta = meta


@dataclass(slots=True)
class _Config:
    allow_private: bool
    require_approval: bool
    timeout: float
    max_bytes: int
    max_chars: int
    search_backend: str
    searx_url: str | None
    user_agent: str


@dataclass(slots=True)
class _Fetched:
    url: str
    status: int
    content_type: str
    charset: str | None
    body: bytes
    truncated: bool
    redirects: int = 0


@dataclass(slots=True)
class ReadablePage:
    title: str
    text: str
    links: list[tuple[str, str]]


@dataclass(slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str = ""


def _sanitize(text: str) -> str:
    return CONTROL_RE.sub("", ANSI_RE.sub("", text))


def _inline(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


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


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    omitted = len(text) - limit
    return f"{text[:limit].rstrip()}\n[... truncated {omitted} chars ...]", True


def _public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in _NAT64_NET or ip in _IPV4_COMPAT_NET:
            # Translation/compatibility prefixes can encode a private IPv4 host.
            return False
        elif ip in _SIXTOFOUR_NET:
            embedded = ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)
            return bool(embedded.is_global)
        elif ip in _TEREDO_NET:
            return False
    return bool(ip.is_global)


def validate_url(url: Any, *, allow_private: bool = False) -> str:
    """Return a safe http(s) URL or raise ValueError."""
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")
    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty.")
    if len(url) > MAX_URL_CHARS:
        raise ValueError(f"URL exceeds {MAX_URL_CHARS} characters.")
    if any(ord(char) < 32 or 127 <= ord(char) <= 159 or char.isspace() or char == "\\" for char in url):
        raise ValueError("URL contains controls, whitespace, or backslashes.")
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        raise ValueError("URL cannot be parsed.") from None
    if parts.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed.")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URLs with embedded credentials are not allowed.")
    host = parts.hostname
    if not host:
        raise ValueError("URL has no hostname.")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("URL has an invalid port.") from None
    if port is not None and not 0 < port <= 65535:
        raise ValueError("URL port is out of range.")
    if allow_private:
        return url
    try:
        literal = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        literal = None
    if literal is not None:
        if not _public_ip(host):
            raise ValueError("URL host is not publicly routable.")
        return url
    try:
        infos = socket.getaddrinfo(
            host, port or (443 if parts.scheme.lower() == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except (OSError, UnicodeError):
        raise ValueError(f"Could not resolve host '{host}'.") from None
    if not infos:
        raise ValueError(f"Could not resolve host '{host}'.")
    for info in infos:
        if not _public_ip(str(info[4][0])):
            raise ValueError("URL host resolves to a private, reserved, or non-routable address.")
    return url


def _read_limited(stream: Any, limit: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    size = 0
    while size < limit:
        chunk = stream.read(min(65_536, limit - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    truncated = size >= limit and bool(stream.read(1))
    return b"".join(chunks), truncated


def _decode(body: bytes, charset: str | None) -> str:
    if body.startswith(b"\xef\xbb\xbf"):
        body = body[3:]
    for encoding in (charset, "utf-8"):
        if not encoding:
            continue
        try:
            return body.decode(encoding, errors="replace")
        except (LookupError, TypeError):
            continue
    return body.decode("utf-8", errors="replace")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def _open(url: str, cfg: _Config, accept: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": cfg.user_agent,
            "Accept": accept,
            "Accept-Encoding": "identity",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    return opener.open(request, timeout=cfg.timeout)


def _fetch_bytes(url: str, cfg: _Config, accept: str = "text/html,application/json,text/plain;q=0.9,*/*;q=0.5") -> _Fetched:
    current = validate_url(url, allow_private=cfg.allow_private)
    redirects = 0
    while True:
        try:
            response = _open(current, cfg, accept)
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location") if exc.headers else None
            if exc.code in REDIRECT_CODES and location:
                exc.close()
                if redirects >= MAX_REDIRECTS:
                    raise _Failure(
                        f"Too many redirects (maximum {MAX_REDIRECTS}).",
                        final_url=current, redirects=redirects, status=exc.code,
                    ) from None
                redirects += 1
                current = urllib.parse.urljoin(current, location)
                current = validate_url(current, allow_private=cfg.allow_private)
                continue
            try:
                body, truncated = _read_limited(exc, min(cfg.max_bytes, 4_000))
            except OSError:
                body, truncated = b"", False
            finally:
                exc.close()
            detail = _sanitize(_decode(body, None)).strip()[:800]
            message = f"HTTP {exc.code} {exc.reason}".strip()
            if detail:
                message += f"\n{detail}"
            raise _Failure(
                message, status=exc.code, final_url=current,
                bytes=len(body), truncated=truncated, redirects=redirects,
            ) from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            reason = getattr(exc, "reason", exc)
            raise _Failure(
                f"Network error: {_sanitize(str(reason))}",
                final_url=current, error="network", redirects=redirects,
            ) from None
        try:
            status = int(getattr(response, "status", 0) or response.getcode())
            content_type = response.headers.get_content_type().lower()
            charset = response.headers.get_content_charset()
            body, truncated = _read_limited(response, cfg.max_bytes)
        finally:
            response.close()
        return _Fetched(current, status, content_type, charset, body, truncated, redirects)


class _HTMLReader(HTMLParser):
    def __init__(self, base_url: str, max_links: int) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.max_links = max_links
        self.title_parts: list[str] = []
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._seen: set[str] = set()
        self._skip = 0
        self._title = 0
        self._anchor: list[Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        values = dict(attrs)
        if tag == "title":
            self._title += 1
        elif tag == "a":
            href = values.get("href")
            if href:
                self._anchor = [href, []]
        if tag in BLOCK_TAGS:
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in {"td", "th"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "title":
            self._title = max(0, self._title - 1)
        elif tag == "a" and self._anchor is not None:
            href, chunks = self._anchor
            self._add_link(str(href), " ".join(chunks))
            self._anchor = None
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._title:
            self.title_parts.append(data)
        else:
            self.parts.append(data)
            if self._anchor is not None:
                self._anchor[1].append(data)

    def _add_link(self, href: str, label: str) -> None:
        if len(self.links) >= self.max_links:
            return
        href = href.strip()
        if not href or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in href):
            return
        try:
            absolute = urllib.parse.urljoin(self.base_url, href)
            parts = urllib.parse.urlsplit(absolute)
        except ValueError:
            return
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return
        if parts.username is not None or parts.password is not None:
            return
        if absolute in self._seen:
            return
        self._seen.add(absolute)
        self.links.append((_inline(label) or absolute, absolute))


def extract_readable(html: str, base_url: str, *, max_links: int = MAX_LINKS) -> ReadablePage:
    parser = _HTMLReader(base_url, max_links)
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    return ReadablePage(
        _inline(" ".join(parser.title_parts)),
        _collapse("".join(parser.parts)),
        parser.links[:max_links],
    )


def _ddg_target(href: str) -> str | None:
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/l/"):
        href = "https://duckduckgo.com" + href
    try:
        parts = urllib.parse.urlsplit(href)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    target = href
    if (host == "duckduckgo.com" or host.endswith(".duckduckgo.com")) and parts.path.startswith("/l/"):
        values = urllib.parse.parse_qs(parts.query).get("uddg") or []
        if values:
            target = values[0]
    try:
        final = urllib.parse.urlsplit(target)
    except ValueError:
        return None
    if final.scheme.lower() not in {"http", "https"} or not final.hostname:
        return None
    if final.username is not None or final.password is not None:
        return None
    return target


class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[SearchHit] = []
        self._seen: set[str] = set()
        self._field: str | None = None
        self._href = ""
        self._chunks: list[str] = []
        self._current_hit: SearchHit | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        lowered = tag.lower()
        if lowered == "a" and ("result__a" in classes or "result-link" in classes):
            self._flush()
            self._current_hit = None
            self._field, self._href, self._chunks = "title", values.get("href") or "", []
        elif "result__snippet" in classes or "result-snippet" in classes:
            self._flush()
            self._field, self._href, self._chunks = "snippet", "", []

    def handle_data(self, data: str) -> None:
        if self._field:
            self._chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" or (self._field == "snippet" and tag.lower() in {"td", "div", "span"}):
            self._flush()

    def _flush(self) -> None:
        field, self._field = self._field, None
        text = _inline("".join(self._chunks))
        self._chunks = []
        if field == "title" and text:
            url = _ddg_target(self._href)
            if url and url not in self._seen:
                self._seen.add(url)
                hit = SearchHit(text, url)
                self.hits.append(hit)
                self._current_hit = hit
            else:
                self._current_hit = None
        elif field == "snippet" and text and self._current_hit is not None:
            if not self._current_hit.snippet:
                self._current_hit.snippet = text


def _search_blocked(html: str) -> bool:
    lowered = html.lower()
    return any(marker in lowered for marker in BLOCK_MARKERS)


def parse_ddg_html(html: str) -> list[SearchHit]:
    parser = _DDGParser()
    try:
        parser.feed(html)
        parser.close()
        parser._flush()
    except Exception:
        pass
    return parser.hits


def parse_searx_json(payload: str | bytes | dict[str, Any], limit: int = 10) -> list[SearchHit]:
    if isinstance(payload, (str, bytes)):
        try:
            data = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid SearXNG JSON: {exc}") from None
    else:
        data = payload
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise ValueError("SearXNG response has no results list.")
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for item in results:
        if len(hits) >= limit:
            break
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        try:
            parts = urllib.parse.urlsplit(url)
        except ValueError:
            continue
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            continue
        if parts.username is not None or parts.password is not None or url in seen:
            continue
        title = _inline(str(item.get("title") or "")) or url
        snippet = _inline(str(item.get("content") or item.get("snippet") or ""))
        seen.add(url)
        hits.append(SearchHit(title[:300], url[:1000], snippet[:800]))
    return hits


def _default_meta(url: Any) -> dict[str, Any]:
    return {
        "status": None, "final_url": url if isinstance(url, str) else "",
        "bytes": 0, "truncated": False, "redirects": 0,
    }


def _web_fetch(args: dict[str, Any], ctx: ToolContext, cfg: _Config) -> ToolResult:
    url: Any = args.get("url")
    try:
        if not isinstance(url, str):
            raise _Failure("Argument 'url' must be a string.", error="invalid_args")
        limit = cfg.max_chars
        if args.get("max_chars") is not None:
            raw_limit = args["max_chars"]
            if not isinstance(raw_limit, int) or isinstance(raw_limit, bool) or raw_limit < 1:
                raise _Failure("Argument 'max_chars' must be a positive integer.", error="invalid_args")
            limit = min(raw_limit, 100_000)
        if cfg.require_approval:
            approved = ctx.request_approval(
                ApprovalRequest(action="web_fetch", summary="Fetch a web page", detail=url, risk="low")
            )
            if not approved:
                raise _Failure("Web fetch denied by approval policy.", error="denied", denied=True)
        fetched = _fetch_bytes(url, cfg)
        if b"\x00" in fetched.body[:4_096]:
            raise _Failure(
                "Response is binary; only HTML, text, and JSON are supported.",
                status=fetched.status, final_url=fetched.url, bytes=len(fetched.body),
                truncated=fetched.truncated, redirects=fetched.redirects, error="content_type",
            )
        raw = _decode(fetched.body, fetched.charset)
        is_html = fetched.content_type in HTML_TYPES or (
            fetched.content_type in {"", "application/octet-stream"}
            and re.match(rb"\s*<(!doctype\s+html|html)", fetched.body[:512], re.IGNORECASE)
        )
        textual = (
            fetched.content_type.startswith("text/")
            or fetched.content_type in {
                "", "application/json", "application/ld+json", "application/xml",
                "application/javascript", "application/x-javascript", "application/octet-stream",
            }
            or fetched.content_type.endswith(("+json", "+xml"))
        )
        if is_html:
            page = extract_readable(raw, fetched.url, max_links=MAX_LINKS)
            title = _sanitize(page.title)[:200] or "(untitled)"
            body_text = _sanitize(page.text) or "(no readable text)"
            links = page.links
        elif textual:
            title = "(untitled)"
            body_text = _sanitize(raw).strip() or "(no content)"
            links = []
        else:
            raise _Failure(
                f"Unsupported content type '{fetched.content_type}'.",
                status=fetched.status, final_url=fetched.url, bytes=len(fetched.body),
                truncated=fetched.truncated, redirects=fetched.redirects, error="content_type",
            )
        body_text, char_truncated = _clip(body_text, limit)
        truncated = fetched.truncated or char_truncated
        header = (
            f"Title: {title}\nURL: {_sanitize(fetched.url)}\n"
            f"HTTP {fetched.status} · {fetched.content_type or 'unknown'} · {len(fetched.body)} bytes"
        )
        if fetched.truncated:
            header += f"\nNote: response body truncated at {cfg.max_bytes} bytes."
        pieces = [header, "", body_text]
        if links:
            pieces.extend(["", f"Links ({len(links)}):"])
            for label, link in links:
                clean_label = _sanitize(_inline(label))[:100]
                shown = _sanitize(link)
                shown = shown if len(shown) <= 300 else shown[:297] + "..."
                pieces.append(f"- {clean_label or shown}: {shown}")
        return ToolResult.success(
            "\n".join(pieces), status=fetched.status, final_url=fetched.url,
            bytes=len(fetched.body), truncated=truncated, redirects=fetched.redirects,
            content_type=fetched.content_type, links=len(links),
        )
    except _Failure as exc:
        return ToolResult.failure(_sanitize(exc.message), **{**_default_meta(url), **exc.meta})
    except Exception as exc:
        return ToolResult.failure(f"web_fetch failed: {_sanitize(str(exc))}", **_default_meta(url))


def _web_search(args: dict[str, Any], ctx: ToolContext, cfg: _Config) -> ToolResult:
    query: Any = args.get("query")
    base_meta: dict[str, Any] = {"count": 0, "backend": cfg.search_backend}
    try:
        if not isinstance(query, str) or not query.strip():
            raise _Failure("Argument 'query' must be a non-empty string.", error="invalid_args")
        query = query.strip()
        if len(query) > MAX_QUERY_CHARS:
            raise _Failure(f"Query exceeds {MAX_QUERY_CHARS} characters.", error="invalid_args")
        max_results = 5
        if args.get("max_results") is not None:
            raw_results = args["max_results"]
            if not isinstance(raw_results, int) or isinstance(raw_results, bool) or raw_results < 1:
                raise _Failure("Argument 'max_results' must be a positive integer.", error="invalid_args")
            max_results = min(raw_results, 10)
        if cfg.require_approval:
            approved = ctx.request_approval(
                ApprovalRequest(
                    action="web_search",
                    summary=f"Search the web for {_sanitize(query)[:80]!r}",
                    detail=query,
                    risk="low",
                )
            )
            if not approved:
                raise _Failure("Web search denied by approval policy.", error="denied", denied=True)
        if cfg.search_backend == "duckduckgo":
            query_params = urllib.parse.urlencode({"q": query})
            attempts = (
                "https://html.duckduckgo.com/html/?" + query_params,
                "https://lite.duckduckgo.com/lite/?" + query_params,
            )
            last_failure: _Failure | None = None
            for index, url in enumerate(attempts):
                try:
                    fetched = _fetch_bytes(
                        url,
                        cfg,
                        accept="text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                    )
                except _Failure as exc:
                    last_failure = exc
                    if index + 1 == len(attempts):
                        raise
                    continue
                body = _decode(fetched.body, fetched.charset)
                if _search_blocked(body):
                    last_failure = _Failure(
                        "Search is blocked or unavailable (DuckDuckGo challenge page).",
                        status=fetched.status,
                        final_url=fetched.url,
                        bytes=len(fetched.body),
                        truncated=fetched.truncated,
                        error="blocked",
                    )
                    if index + 1 == len(attempts):
                        raise last_failure
                    continue
                hits = parse_ddg_html(body)
                break
            else:
                raise last_failure or _Failure("DuckDuckGo search failed.", error="search")
        elif cfg.search_backend == "searxng":
            if not cfg.searx_url:
                raise _Failure("search_backend='searxng' requires searx_url.", error="config")
            url = cfg.searx_url.rstrip("/") + "/search?" + urllib.parse.urlencode(
                {"q": query, "format": "json"}
            )
            fetched = _fetch_bytes(url, cfg, accept="application/json,text/plain;q=0.5")
            try:
                hits = parse_searx_json(
                    _decode(fetched.body, fetched.charset), limit=max_results
                )
            except ValueError as exc:
                raise _Failure(
                    f"Search backend returned invalid JSON: {exc}",
                    status=fetched.status, final_url=fetched.url,
                    bytes=len(fetched.body), truncated=fetched.truncated, error="search",
                ) from None
        else:
            raise _Failure(
                f"Unknown search backend '{cfg.search_backend}'.",
                error="unknown_backend",
            )
        hits = hits[:max_results]
        meta = {
            "count": len(hits), "backend": cfg.search_backend,
            "status": fetched.status, "final_url": fetched.url,
            "bytes": len(fetched.body), "truncated": fetched.truncated,
        }
        if not hits:
            return ToolResult.success(
                f"No results found for '{_sanitize(query)}'.", **meta
            )
        lines = [f"Search results for '{_sanitize(query)}' ({len(hits)}):"]
        for index, hit in enumerate(hits, 1):
            lines.append(f"{index}. {_sanitize(hit.title)[:300]}")
            lines.append(f"   {_sanitize(hit.url)[:1000]}")
            if hit.snippet:
                lines.append(f"   {_sanitize(hit.snippet)[:800]}")
        return ToolResult.success("\n".join(lines), **meta)
    except _Failure as exc:
        return ToolResult.failure(
            _sanitize(exc.message), **{**base_meta, **exc.meta}
        )
    except Exception as exc:
        return ToolResult.failure(f"web_search failed: {_sanitize(str(exc))}", **base_meta)


def make_web_tools(
    *,
    allow_private: bool = False,
    require_approval: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 2_000_000,
    max_chars: int = 12_000,
    search_backend: str = "duckduckgo",
    searx_url: str | None = None,
    user_agent: str = "OHAR/0.2",
) -> list[Tool]:
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("timeout must be a positive number.")
    for name, value in (("max_bytes", max_bytes), ("max_chars", max_chars)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(user_agent, str) or not user_agent.strip() or any(
        ord(char) < 32 or ord(char) == 127 for char in user_agent
    ):
        raise ValueError("user_agent must be a non-empty single-line string.")
    if not isinstance(search_backend, str) or not search_backend.strip():
        raise ValueError("search_backend must be a non-empty string.")
    if searx_url is not None and not isinstance(searx_url, str):
        raise ValueError("searx_url must be a string or None.")
    cfg = _Config(
        allow_private=bool(allow_private),
        require_approval=bool(require_approval),
        timeout=float(timeout),
        max_bytes=max_bytes,
        max_chars=max_chars,
        search_backend=search_backend.strip().lower(),
        searx_url=searx_url.strip() if searx_url else None,
        user_agent=user_agent.strip(),
    )

    def fetch_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return _web_fetch(args, ctx, cfg)

    def search_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return _web_search(args, ctx, cfg)

    return [
        Tool(
            name="web_fetch",
            description=(
                "Fetch an http(s) page and return its title, final URL, cleaned readable "
                "text, and a short list of absolute links. Private and reserved hosts are "
                "blocked. Output is bounded and binary content is rejected."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL to fetch."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum readable text characters (default from config).",
                        "minimum": 1,
                        "maximum": 100_000,
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            handler=fetch_handler,
            read_only=True,
        ),
        Tool(
            name="web_search",
            description=(
                "Search the web and return numbered results with titles, URLs, and "
                "snippets. Uses DuckDuckGo HTML search or a configured SearXNG instance."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default 5, max 10).",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=search_handler,
            read_only=True,
        ),
    ]
