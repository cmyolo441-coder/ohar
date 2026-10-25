"""Health-aware lane balancing, capacity limits, and failover."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .types import Config, ConfigError, LaneConfig, Member, ProviderError

_STRATEGIES = frozenset({"failover", "round_robin", "weighted", "least_busy"})
_HARD_COOLDOWN_KINDS = frozenset({"auth", "model", "bad_request", "context"})
_TRANSIENT_COOLDOWN_KINDS = frozenset({"rate_limit", "server", "network", "protocol"})
_MAX_COOLDOWN = 600.0
_MID_STREAM_COOLDOWN = 30.0
_EWMA_ALPHA = 0.3


@dataclass(slots=True)
class _MemberState:
    failures: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = None
    ewma_ms: float | None = None
    last_used: float = 0.0
    inflight: int = 0


@dataclass(slots=True)
class _LaneState:
    lane: LaneConfig
    strategy: str
    members: list[Member]
    by_id: dict[str, Member]
    states: dict[str, _MemberState]
    semaphore: threading.BoundedSemaphore
    rr_counter: int = 0
    tie_counter: int = 0
    smooth: dict[str, float] = field(default_factory=dict)


def _invoke_factory(
    factory: Callable[..., object], provider_config: object, api_key: str | None
) -> object:
    """Call a provider factory with either ``(config, key)`` or ``(config,)``."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(provider_config, api_key)
    for args, kwargs in (
        ((provider_config, api_key), {}),
        ((provider_config,), {"api_key": api_key}),
        ((provider_config,), {}),
    ):
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return factory(*args, **kwargs)
    return factory(provider_config, api_key)


def _invoke_resolver(resolver: Callable[..., str | None], provider_config: object) -> str | None:
    """Call an API-key resolver with a ProviderConfig, or a provider name."""
    try:
        signature = inspect.signature(resolver)
    except (TypeError, ValueError):
        return resolver(provider_config)
    for argument in (provider_config, getattr(provider_config, "name", provider_config)):
        try:
            signature.bind(argument)
        except TypeError:
            continue
        return resolver(argument)
    return resolver(provider_config)


def _finite_seconds(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) else None


def _retry_after_seconds(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(seconds):
        return None
    if math.isinf(seconds):
        return _MAX_COOLDOWN if seconds > 0 else 0.0
    return seconds


def _is_cancelled(error: BaseException) -> bool:
    if isinstance(error, (asyncio.CancelledError, concurrent.futures.CancelledError)):
        return True
    if type(error).__name__ == "CancelledError":
        return True
    return _error_kind(error) == "cancelled"


def _error_kind(error: BaseException) -> str:
    kind = getattr(error, "kind", "")
    return kind.strip().lower() if isinstance(kind, str) else ""


class LaneManager:
    """Route work across lane members with health and capacity tracking.

    A typical agent loop calls :meth:`order`, then for each candidate:
    ``acquire``, use the provider, ``report_success`` or ``report_failure``,
    and finally ``release``.
    """

    def __init__(
        self,
        config: Config,
        *,
        providers: dict[str, object] | None = None,
        provider_factory: Callable[..., object] | None = None,
        api_key_resolver: Callable[..., str | None] | None = None,
    ) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._provider_factory = provider_factory
        self._api_key_resolver = api_key_resolver
        self._providers_override = providers is not None
        self.providers: dict[str, object] = providers if providers is not None else {}

        if not config.lanes:
            raise ConfigError("No lanes configured. Add at least one lane to the configuration.")

        self._lanes: dict[str, _LaneState] = {}
        provider_uses: dict[str, str] = {}
        for lane_name, lane_config in config.lanes.items():
            if not isinstance(lane_name, str) or not lane_name.strip():
                raise ConfigError("Lane names must be non-empty strings.")
            self._lanes[lane_name] = self._build_lane(lane_name, lane_config, provider_uses)

        known = set(self.providers) if self._providers_override else set(config.providers)
        missing = sorted(name for name in provider_uses if name not in known)
        if missing:
            name = missing[0]
            available = ", ".join(sorted(known)) or "(none)"
            raise ConfigError(
                f"{provider_uses[name]} references provider '{name}', but it is not "
                f"configured. Available providers: {available}."
            )

        if not self._providers_override:
            for name in sorted(provider_uses):
                self._create_provider(name)

    # ------------------------------------------------------------------ setup

    def _build_lane(
        self, lane_name: str, lane_config: LaneConfig, provider_uses: dict[str, str]
    ) -> _LaneState:
        if not lane_config.members:
            raise ConfigError(
                f"Lane '{lane_name}' has no members. "
                "Add at least one 'provider/model' entry."
            )

        strategy = str(lane_config.strategy or "failover").strip().lower()
        if strategy not in _STRATEGIES:
            available = ", ".join(sorted(_STRATEGIES))
            raise ConfigError(
                f"Lane '{lane_name}' uses unknown strategy '{lane_config.strategy}'. "
                f"Choose one of: {available}."
            )

        try:
            max_inflight = int(lane_config.max_inflight)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Lane '{lane_name}' has invalid max_inflight '{lane_config.max_inflight}'."
            ) from exc
        if max_inflight < 1:
            raise ConfigError(f"Lane '{lane_name}' max_inflight must be at least 1.")

        weights = lane_config.weights or []
        members: list[Member] = []
        states: dict[str, _MemberState] = {}
        by_id: dict[str, Member] = {}
        for index, raw in enumerate(lane_config.members):
            if not isinstance(raw, str):
                raise ConfigError(
                    f"Lane '{lane_name}' member #{index + 1} must be a 'provider/model' string."
                )
            text = raw.strip()
            provider, separator, model = text.partition("/")
            provider, model = provider.strip(), model.strip()
            if not separator or not provider or not model:
                raise ConfigError(
                    f"Invalid member '{raw}' in lane '{lane_name}': expected 'provider/model'."
                )

            weight = 1.0
            if index < len(weights):
                try:
                    weight = float(weights[index])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"Invalid weight for member '{text}' in lane '{lane_name}'."
                    ) from exc
                if not math.isfinite(weight) or weight <= 0:
                    raise ConfigError(
                        f"Weight for member '{text}' in lane '{lane_name}' "
                        "must be a positive number."
                    )

            member = Member(
                lane=lane_name,
                provider=provider,
                model=model,
                weight=weight,
                index=index,
            )
            members.append(member)
            by_id[member.id] = member
            states.setdefault(member.id, _MemberState())
            provider_uses.setdefault(provider, f"Lane '{lane_name}' member '{text}'")

        return _LaneState(
            lane=lane_config,
            strategy=strategy,
            members=members,
            by_id=by_id,
            states=states,
            semaphore=threading.BoundedSemaphore(max_inflight),
        )

    def _create_provider(self, name: str) -> object:
        if name in self.providers:
            return self.providers[name]
        try:
            provider_config = self.config.providers[name]
        except KeyError as exc:
            raise ConfigError(f"Unknown provider '{name}'.") from exc

        api_key = self._resolve_api_key(name, provider_config)
        factory = self._provider_factory
        try:
            if factory is None:
                try:
                    from .providers import create_provider
                except ImportError as exc:
                    raise ConfigError(
                        "Cannot create providers because 'ohar.providers' is unavailable. "
                        "Pass providers=... or provider_factory=... to LaneManager."
                    ) from exc
                provider = _invoke_factory(create_provider, provider_config, api_key)
            else:
                provider = _invoke_factory(factory, provider_config, api_key)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"Failed to create provider '{name}': {exc}") from exc

        self.providers[name] = provider
        return provider

    def _resolve_api_key(self, name: str, provider_config: object) -> str | None:
        resolver = self._api_key_resolver
        if resolver is None:
            try:
                from .config import resolve_api_key
            except ImportError as exc:
                raise ConfigError(
                    "Cannot resolve API keys because 'ohar.config' is unavailable. "
                    "Pass api_key_resolver=... to LaneManager."
                ) from exc
            resolver = resolve_api_key

        last_error: Exception | None = None
        arguments = [provider_config]
        provider_name = getattr(provider_config, "name", None)
        if provider_name is not None:
            arguments.append(provider_name)
        for argument in arguments:
            try:
                return _invoke_resolver(resolver, argument)
            except ConfigError:
                raise
            except Exception as exc:
                last_error = exc
        raise ConfigError(
            f"Failed to resolve the API key for provider '{name}': {last_error}"
        ) from last_error

    # ----------------------------------------------------------------- lookup

    def _lane(self, lane: str) -> _LaneState:
        state = self._lanes.get(lane)
        if state is None:
            available = ", ".join(sorted(self._lanes)) or "(none)"
            raise ConfigError(f"Unknown lane '{lane}'. Available lanes: {available}.")
        return state

    def _member_state(self, member: Member) -> _MemberState:
        lane = self._lanes.get(member.lane)
        if lane is None:
            available = ", ".join(sorted(self._lanes)) or "(none)"
            raise ConfigError(
                f"Member '{member.id}' belongs to unknown lane '{member.lane}'. "
                f"Available lanes: {available}."
            )
        state = lane.states.get(member.id)
        if state is None:
            raise ConfigError(
                f"Member '{member.id}' is not configured in lane '{member.lane}'."
            )
        return state

    def members(self, lane: str) -> list[Member]:
        """Return the configured members of ``lane`` in configuration order."""
        return list(self._lane(lane).members)

    def member_by_id(self, lane: str, id: str) -> Member | None:
        """Return a member by its ``provider/model`` id, or ``None``."""
        with self._lock:
            lane_state = self._lanes.get(lane)
            return None if lane_state is None else lane_state.by_id.get(id)

    def provider_for(self, member: Member) -> object:
        """Return the provider instance for ``member``."""
        with self._lock:
            try:
                return self.providers[member.provider]
            except KeyError:
                pass
            if self._providers_override:
                raise ConfigError(
                    f"Provider '{member.provider}' was not supplied to LaneManager."
                )
            if member.provider not in self.config.providers:
                raise ConfigError(f"Unknown provider '{member.provider}'.")
            return self._create_provider(member.provider)

    # ---------------------------------------------------------------- ordering

    def order(self, lane: str, exclude: set[str] | None = None) -> list[Member]:
        """Return a full candidate order for ``lane``.

        Cooling members are appended after healthy ones so an all-cooling lane
        still yields one probe candidate.
        """
        lane_state = self._lane(lane)
        excluded = set(exclude) if exclude else set()

        with self._lock:
            now = time.monotonic()
            candidates = [m for m in lane_state.members if m.id not in excluded]
            if not candidates:
                self._advance_empty(lane_state)
                return []

            if lane_state.strategy == "round_robin":
                offset = lane_state.rr_counter % len(candidates)
                lane_state.rr_counter += 1
                ordered = candidates[offset:] + candidates[:offset]
            elif lane_state.strategy == "weighted":
                ordered = self._weighted_order(lane_state, candidates)
            elif lane_state.strategy == "least_busy":
                ordered = self._least_busy_order(lane_state, candidates)
            else:
                ordered = candidates

            healthy: list[Member] = []
            cooling: list[Member] = []
            for member in ordered:
                if lane_state.states[member.id].cooldown_until > now:
                    cooling.append(member)
                else:
                    healthy.append(member)
            return healthy + cooling

    @staticmethod
    def _advance_empty(lane_state: _LaneState) -> None:
        if lane_state.strategy == "round_robin":
            lane_state.rr_counter += 1
        elif lane_state.strategy == "least_busy":
            lane_state.tie_counter += 1

    @staticmethod
    def _weighted_order(lane_state: _LaneState, candidates: list[Member]) -> list[Member]:
        """Smooth weighted ordering for the lane's full candidate list.

        One stateful smooth-WRR pick determines the lead candidate; the rest
        follow descending smooth weight, keeping the sequence deterministic
        and the weight accounting fair across calls.
        """
        total = sum(member.weight for member in candidates)
        for member in candidates:
            lane_state.smooth[member.id] = (
                lane_state.smooth.get(member.id, 0.0) + member.weight
            )

        best = candidates[0]
        best_current = lane_state.smooth[best.id]
        for member in candidates[1:]:
            current = lane_state.smooth[member.id]
            if current > best_current:
                best, best_current = member, current
        lane_state.smooth[best.id] = best_current - total

        tail = [member for member in candidates if member is not best]
        tail.sort(key=lambda member: (-lane_state.smooth[member.id], member.index))
        return [best, *tail]

    def _least_busy_order(self, lane_state: _LaneState, candidates: list[Member]) -> list[Member]:
        size = len(candidates)
        offset = lane_state.tie_counter % size
        lane_state.tie_counter += 1
        indexed = list(enumerate(candidates))

        def key(item: tuple[int, Member]) -> tuple[float, float, int, int]:
            position, member = item
            state = lane_state.states[member.id]
            load = state.inflight / member.weight
            latency = state.ewma_ms if state.ewma_ms is not None else math.inf
            return load, latency, state.failures, (position - offset) % size

        indexed.sort(key=key)
        return [member for _, member in indexed]

    # ---------------------------------------------------------------- capacity

    def acquire(self, member: Member, timeout: float | None = 30.0) -> bool:
        """Reserve one lane slot for ``member``.

        Returns ``False`` when ``timeout`` expires, ``None`` waits forever.
        """
        lane_state = self._lane(member.lane)
        with self._lock:
            state = self._member_state(member)
        semaphore = lane_state.semaphore

        if timeout is None:
            acquired = semaphore.acquire()
        else:
            try:
                wait = float(timeout)
            except (TypeError, ValueError):
                wait = 0.0
            if math.isnan(wait) or wait <= 0:
                acquired = semaphore.acquire(blocking=False)
            elif math.isinf(wait):
                acquired = semaphore.acquire()
            else:
                acquired = semaphore.acquire(timeout=wait)

        if not acquired:
            return False
        with self._lock:
            state.inflight += 1
        return True

    def release(self, member: Member) -> None:
        """Release a slot reserved by :meth:`acquire`.

        Extra releases are ignored and never decrement below zero.
        """
        lane_state = self._lanes.get(member.lane)
        if lane_state is None:
            return
        with self._lock:
            state = lane_state.states.get(member.id)
            if state is None or state.inflight <= 0:
                return
            state.inflight -= 1
        try:
            lane_state.semaphore.release()
        except ValueError:
            # Defensive: never turn a duplicate release into a caller failure.
            pass

    # ----------------------------------------------------------------- health

    def report_success(self, member: Member, latency_ms: float) -> None:
        """Record a successful call and update the member's EWMA latency."""
        latency = _finite_seconds(latency_ms)
        if latency is not None and latency < 0:
            latency = 0.0
        with self._lock:
            state = self._member_state(member)
            state.failures = 0
            state.cooldown_until = 0.0
            state.last_error = None
            state.last_used = time.monotonic()
            if latency is not None:
                if state.ewma_ms is None:
                    state.ewma_ms = latency
                else:
                    state.ewma_ms = (
                        _EWMA_ALPHA * latency + (1.0 - _EWMA_ALPHA) * state.ewma_ms
                    )

    def report_failure(
        self, member: Member, error: BaseException, *, phase: str = "before_stream"
    ) -> None:
        """Record a failed call and apply a bounded cooldown."""
        with self._lock:
            lane_state = self._lane(member.lane)
            state = self._member_state(member)
            state.failures += 1
            message = str(error).strip()
            state.last_error = message or type(error).__name__
            cooldown = self._cooldown_seconds(lane_state, error, phase)
            state.cooldown_until = (
                time.monotonic() + cooldown if cooldown > 0 else 0.0
            )

    @staticmethod
    def _cooldown_seconds(lane_state: _LaneState, error: BaseException, phase: str) -> float:
        if _is_cancelled(error):
            return 0.0

        lane_cooldown = _finite_seconds(lane_state.lane.cooldown_seconds)
        if lane_cooldown is None or lane_cooldown < 0:
            lane_cooldown = 0.0
        mid_stream = str(phase or "").strip().lower() == "mid_stream"

        if isinstance(error, ProviderError):
            kind = _error_kind(error)
            retry_after = _retry_after_seconds(error.retry_after)
            if retry_after is not None:
                cooldown = (
                    max(retry_after, 2.0) if kind == "rate_limit" else retry_after
                )
            elif kind in _HARD_COOLDOWN_KINDS:
                cooldown = 300.0
            elif mid_stream:
                cooldown = min(lane_cooldown, _MID_STREAM_COOLDOWN)
            elif kind in _TRANSIENT_COOLDOWN_KINDS:
                cooldown = lane_cooldown
            else:
                cooldown = lane_cooldown
        elif mid_stream:
            cooldown = min(lane_cooldown, _MID_STREAM_COOLDOWN)
        else:
            cooldown = lane_cooldown

        return min(max(cooldown, 0.0), _MAX_COOLDOWN)

    def reset_health(self) -> None:
        """Clear cooldowns and failure counters without touching live requests."""
        with self._lock:
            for lane_state in self._lanes.values():
                for state in lane_state.states.values():
                    state.failures = 0
                    state.cooldown_until = 0.0
                    state.last_error = None

    def status(self) -> list[dict[str, object]]:
        """Return a thread-safe health snapshot sorted by lane and index."""
        rows: list[dict[str, object]] = []
        with self._lock:
            now = time.monotonic()
            for lane_name in sorted(self._lanes):
                lane_state = self._lanes[lane_name]
                for member in sorted(lane_state.members, key=lambda item: item.index):
                    state = lane_state.states[member.id]
                    remaining = max(0.0, state.cooldown_until - now)
                    rows.append(
                        {
                            "lane": lane_name,
                            "member": member.id,
                            "strategy": lane_state.strategy,
                            "health": "cooling" if remaining > 0 else "ok",
                            "cooldown_remaining_s": round(remaining, 1),
                            "inflight": state.inflight,
                            "failures": state.failures,
                            "ewma_ms": state.ewma_ms,
                        }
                    )
        return rows


__all__ = ["LaneManager"]
