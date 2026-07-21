"""Process-local admission limits and provider usage budgets.

The primitives in this module deliberately have no FastAPI, WebSocket, room,
or provider dependency.  They are synchronous and guarded by ``RLock`` so one
instance may be shared by request handlers on multiple event loops or worker
threads.  The async methods are convenience wrappers over the same atomic
critical sections; they do not maintain a second set of counters.

These limits are process-local.  A deployment with multiple processes must
either keep traffic for one key/scope on one process or replace this module's
storage with an external atomic store.
"""

from __future__ import annotations

import math
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal


Clock = Callable[[], float]
AdmissionChannel = Literal["rest", "ws", "default"]


class LimitConfigurationError(ValueError):
    """A limiter or budget policy is invalid and was rejected fail-closed."""


class InvalidLimitKey(ValueError):
    """An admission or budget key is empty, too large, or otherwise invalid."""


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise LimitConfigurationError(f"{name} must be a positive finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as err:
        raise LimitConfigurationError(f"{name} must be a positive finite number") from err
    if parsed <= 0 or not math.isfinite(parsed):
        raise LimitConfigurationError(f"{name} must be a positive finite number")
    return parsed


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LimitConfigurationError(f"{name} must be a positive integer")
    return value


def _optional_non_negative_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LimitConfigurationError(f"{name} must be a non-negative integer or None")
    return value


def _usage_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _bounded_key(name: str, value: object, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise InvalidLimitKey(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise InvalidLimitKey(f"{name} must not be empty")
    if len(normalized) > max_length:
        raise InvalidLimitKey(f"{name} exceeds the {max_length}-character limit")
    return normalized


def _clock_value(clock: Clock) -> float:
    value = float(clock())
    if not math.isfinite(value):
        raise RuntimeError("monotonic clock returned a non-finite value")
    return value


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    """Configuration for one keyed token-bucket namespace.

    ``capacity`` is the maximum burst and ``refill_rate`` is measured in
    admission tokens per second.  Keys idle for ``key_ttl_seconds`` are
    removed.  If all ``max_keys`` entries are still active, a previously
    unseen key is rejected rather than evicting an active key and resetting
    its rate limit.
    """

    capacity: float
    refill_rate: float
    key_ttl_seconds: float = 900.0
    max_keys: int = 10_000
    max_key_length: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacity", _positive_finite("capacity", self.capacity))
        object.__setattr__(
            self,
            "refill_rate",
            _positive_finite("refill_rate", self.refill_rate),
        )
        object.__setattr__(
            self,
            "key_ttl_seconds",
            _positive_finite("key_ttl_seconds", self.key_ttl_seconds),
        )
        object.__setattr__(self, "max_keys", _positive_int("max_keys", self.max_keys))
        object.__setattr__(
            self,
            "max_key_length",
            _positive_int("max_key_length", self.max_key_length),
        )


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Result of one atomic admission attempt."""

    allowed: bool
    channel: AdmissionChannel
    key: str
    remaining_tokens: float
    retry_after_seconds: float
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float
    last_seen: float


class TokenBucketRateLimiter:
    """A bounded, keyed, process-local monotonic token bucket."""

    def __init__(self, config: RateLimitConfig, *, clock: Clock = time.monotonic) -> None:
        if not isinstance(config, RateLimitConfig):
            raise LimitConfigurationError("config must be a RateLimitConfig")
        if not callable(clock):
            raise LimitConfigurationError("clock must be callable")
        self.config = config
        self._clock = clock
        self._buckets: dict[tuple[AdmissionChannel, str], _Bucket] = {}
        self._lock = threading.RLock()
        self._last_now: float | None = None

    @property
    def key_count(self) -> int:
        with self._lock:
            return len(self._buckets)

    def _now_locked(self) -> float:
        current = _clock_value(self._clock)
        # A real monotonic clock cannot go backwards.  Clamping a faulty test
        # or platform clock prevents accidental refill or premature expiry.
        if self._last_now is not None and current < self._last_now:
            current = self._last_now
        self._last_now = current
        return current

    def _prune_locked(self, now: float) -> int:
        stale = [
            bucket_key
            for bucket_key, bucket in self._buckets.items()
            if now - bucket.last_seen >= self.config.key_ttl_seconds
        ]
        for bucket_key in stale:
            del self._buckets[bucket_key]
        return len(stale)

    def prune(self) -> int:
        """Remove idle keys and return the number removed."""
        with self._lock:
            return self._prune_locked(self._now_locked())

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()

    def forget(self, key: str, *, channel: AdmissionChannel = "default") -> bool:
        """Explicitly remove one bucket.

        This resets that identity's allowance, so callers should reserve it
        for lifecycle cleanup rather than invoking it after a denial.
        """
        normalized_key = _bounded_key("key", key, max_length=self.config.max_key_length)
        normalized_channel = self._channel(channel)
        with self._lock:
            return self._buckets.pop((normalized_channel, normalized_key), None) is not None

    @staticmethod
    def _channel(channel: object) -> AdmissionChannel:
        if channel not in {"rest", "ws", "default"}:
            raise InvalidLimitKey("channel must be 'rest', 'ws', or 'default'")
        return channel  # type: ignore[return-value]

    def admit(
        self,
        key: str,
        *,
        weight: float = 1.0,
        channel: AdmissionChannel = "default",
    ) -> RateLimitDecision:
        """Atomically consume ``weight`` tokens for ``(channel, key)``."""
        normalized_key = _bounded_key("key", key, max_length=self.config.max_key_length)
        normalized_channel = self._channel(channel)
        parsed_weight = _positive_finite("weight", weight)

        if parsed_weight > self.config.capacity:
            return RateLimitDecision(
                allowed=False,
                channel=normalized_channel,
                key=normalized_key,
                remaining_tokens=0.0,
                retry_after_seconds=math.inf,
                reason="weight_exceeds_capacity",
            )

        bucket_key = (normalized_channel, normalized_key)
        with self._lock:
            now = self._now_locked()
            self._prune_locked(now)
            bucket = self._buckets.get(bucket_key)
            if bucket is None:
                if len(self._buckets) >= self.config.max_keys:
                    return RateLimitDecision(
                        allowed=False,
                        channel=normalized_channel,
                        key=normalized_key,
                        remaining_tokens=0.0,
                        retry_after_seconds=self.config.key_ttl_seconds,
                        reason="key_capacity",
                    )
                bucket = _Bucket(
                    tokens=self.config.capacity,
                    updated_at=now,
                    last_seen=now,
                )
                self._buckets[bucket_key] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(
                    self.config.capacity,
                    bucket.tokens + elapsed * self.config.refill_rate,
                )
                bucket.updated_at = now
                bucket.last_seen = now

            if bucket.tokens + 1e-12 < parsed_weight:
                missing = parsed_weight - bucket.tokens
                return RateLimitDecision(
                    allowed=False,
                    channel=normalized_channel,
                    key=normalized_key,
                    remaining_tokens=max(0.0, bucket.tokens),
                    retry_after_seconds=missing / self.config.refill_rate,
                    reason="rate_limited",
                )

            bucket.tokens = max(0.0, bucket.tokens - parsed_weight)
            return RateLimitDecision(
                allowed=True,
                channel=normalized_channel,
                key=normalized_key,
                remaining_tokens=bucket.tokens,
                retry_after_seconds=0.0,
                reason="allowed",
            )

    def admit_rest(self, key: str, *, weight: float = 1.0) -> RateLimitDecision:
        return self.admit(key, weight=weight, channel="rest")

    def admit_ws(self, key: str, *, weight: float = 1.0) -> RateLimitDecision:
        return self.admit(key, weight=weight, channel="ws")

    async def admit_async(
        self,
        key: str,
        *,
        weight: float = 1.0,
        channel: AdmissionChannel = "default",
    ) -> RateLimitDecision:
        return self.admit(key, weight=weight, channel=channel)

    async def admit_rest_async(self, key: str, *, weight: float = 1.0) -> RateLimitDecision:
        return self.admit_rest(key, weight=weight)

    async def admit_ws_async(self, key: str, *, weight: float = 1.0) -> RateLimitDecision:
        return self.admit_ws(key, weight=weight)


class AdmissionLimiter:
    """REST and WebSocket admission with independently configured buckets."""

    def __init__(
        self,
        *,
        rest: RateLimitConfig,
        ws: RateLimitConfig,
        clock: Clock = time.monotonic,
    ) -> None:
        self.rest = TokenBucketRateLimiter(rest, clock=clock)
        self.ws = TokenBucketRateLimiter(ws, clock=clock)

    def admit_rest(self, key: str, *, weight: float = 1.0) -> RateLimitDecision:
        return self.rest.admit_rest(key, weight=weight)

    def admit_ws(self, key: str, *, weight: float = 1.0) -> RateLimitDecision:
        return self.ws.admit_ws(key, weight=weight)

    async def admit_rest_async(self, key: str, *, weight: float = 1.0) -> RateLimitDecision:
        return self.admit_rest(key, weight=weight)

    async def admit_ws_async(self, key: str, *, weight: float = 1.0) -> RateLimitDecision:
        return self.admit_ws(key, weight=weight)


@dataclass(frozen=True, slots=True)
class ProviderBudgetPolicy:
    """Hard per-scope provider limits.

    Token limits use provider-reported input plus output tokens.  No currency,
    price table, or inferred token amount is accepted by this ledger.
    ``None`` explicitly means that dimension is unlimited; zero denies it.
    """

    max_calls: int | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_calls",
            _optional_non_negative_int("max_calls", self.max_calls),
        )
        object.__setattr__(
            self,
            "max_tokens",
            _optional_non_negative_int("max_tokens", self.max_tokens),
        )


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    """One admitted provider call and its reserved token upper bound."""

    reservation_id: str
    scope_id: str
    token_budget: int | None


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Immutable committed and in-flight counters for one scope."""

    scope_id: str
    policy: ProviderBudgetPolicy
    calls: int
    input_tokens: int
    output_tokens: int
    reserved_calls: int
    reserved_tokens: int
    rejected_reservations: int
    unknown_usage_records: int
    closed: bool
    blocked_reason: str | None
    remaining_calls: int | None
    remaining_tokens: int | None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class BudgetReserveDecision:
    """Result of one atomic provider-call reservation attempt."""

    allowed: bool
    reason: str
    reservation: BudgetReservation | None
    snapshot: BudgetSnapshot | None

    def __bool__(self) -> bool:
        return self.allowed


@dataclass(frozen=True, slots=True)
class BudgetRecordResult:
    """Result of finalizing a reservation with provider-reported usage.

    Usage is always accounted once a provider call has happened.  Therefore a
    response that exceeds its reservation is charged to the scope and returns
    ``accepted=False``; it is never discarded merely to preserve a counter
    invariant.
    """

    accepted: bool
    reason: str
    snapshot: BudgetSnapshot

    def __bool__(self) -> bool:
        return self.accepted


class BudgetRejectedError(RuntimeError):
    """Raised by ``reserve`` when ``try_reserve`` denies admission."""

    def __init__(self, decision: BudgetReserveDecision) -> None:
        super().__init__(f"provider budget rejected reservation: {decision.reason}")
        self.decision = decision


class UnknownReservationError(KeyError):
    """A reservation is unknown, already recorded, or already cancelled."""


@dataclass(slots=True)
class _BudgetState:
    policy: ProviderBudgetPolicy
    calls: int
    input_tokens: int
    output_tokens: int
    reserved_calls: int
    reserved_tokens: int
    rejected_reservations: int
    unknown_usage_records: int
    closed: bool
    blocked_reason: str | None
    last_seen: float


@dataclass(slots=True)
class _ReservationState:
    reservation: BudgetReservation


class ProviderBudgetLedger:
    """Atomic per-run/per-room provider call and token accounting.

    Each provider attempt must reserve exactly one call before transport.  All
    in-flight reservations count against both hard limits, so concurrent
    callers cannot collectively pass a limit.  After transport, callers must
    either ``record`` provider-reported usage (failed attempts included) or
    ``cancel`` only when transport definitely did not start.

    Scope policies are immutable after registration.  Active scopes are never
    expired automatically because doing so would reset their budget.  Call
    ``close_scope`` at terminal run/room lifecycle; closed scopes with no
    reservations become eligible for TTL pruning.
    """

    def __init__(
        self,
        default_policy: ProviderBudgetPolicy | None = None,
        *,
        max_scopes: int = 10_000,
        max_inflight_reservations: int = 100_000,
        closed_scope_ttl_seconds: float = 3600.0,
        max_scope_id_length: int = 256,
        clock: Clock = time.monotonic,
        reservation_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if default_policy is not None and not isinstance(default_policy, ProviderBudgetPolicy):
            raise LimitConfigurationError(
                "default_policy must be a ProviderBudgetPolicy or None"
            )
        if not callable(clock):
            raise LimitConfigurationError("clock must be callable")
        if reservation_id_factory is not None and not callable(reservation_id_factory):
            raise LimitConfigurationError("reservation_id_factory must be callable")
        self.default_policy = default_policy or ProviderBudgetPolicy()
        self.max_scopes = _positive_int("max_scopes", max_scopes)
        self.max_inflight_reservations = _positive_int(
            "max_inflight_reservations",
            max_inflight_reservations,
        )
        self.closed_scope_ttl_seconds = _positive_finite(
            "closed_scope_ttl_seconds",
            closed_scope_ttl_seconds,
        )
        self.max_scope_id_length = _positive_int("max_scope_id_length", max_scope_id_length)
        self._clock = clock
        self._reservation_id_factory = reservation_id_factory or (
            lambda: secrets.token_urlsafe(18)
        )
        self._scopes: dict[str, _BudgetState] = {}
        self._reservations: dict[str, _ReservationState] = {}
        self._lock = threading.RLock()
        self._last_now: float | None = None
        self._rejected_reservations = 0

    @staticmethod
    def room_scope(room_id: str) -> str:
        return f"room:{room_id}"

    @staticmethod
    def run_scope(run_id: str) -> str:
        return f"run:{run_id}"

    @property
    def scope_count(self) -> int:
        with self._lock:
            return len(self._scopes)

    @property
    def inflight_reservation_count(self) -> int:
        with self._lock:
            return len(self._reservations)

    @property
    def rejected_reservations(self) -> int:
        with self._lock:
            return self._rejected_reservations

    def _scope_id(self, scope_id: str) -> str:
        return _bounded_key(
            "scope_id",
            scope_id,
            max_length=self.max_scope_id_length,
        )

    def _now_locked(self) -> float:
        current = _clock_value(self._clock)
        if self._last_now is not None and current < self._last_now:
            current = self._last_now
        self._last_now = current
        return current

    def _prune_locked(self, now: float) -> int:
        scopes_with_reservations = {
            item.reservation.scope_id for item in self._reservations.values()
        }
        stale = [
            scope_id
            for scope_id, state in self._scopes.items()
            if state.closed
            and scope_id not in scopes_with_reservations
            and now - state.last_seen >= self.closed_scope_ttl_seconds
        ]
        for scope_id in stale:
            del self._scopes[scope_id]
        return len(stale)

    def prune(self) -> int:
        with self._lock:
            return self._prune_locked(self._now_locked())

    def _snapshot_locked(self, scope_id: str, state: _BudgetState) -> BudgetSnapshot:
        used_tokens = state.input_tokens + state.output_tokens + state.reserved_tokens
        if state.policy.max_calls is None:
            remaining_calls = None
        else:
            remaining_calls = max(
                0,
                state.policy.max_calls - state.calls - state.reserved_calls,
            )
        if state.policy.max_tokens is None:
            remaining_tokens = None
        else:
            remaining_tokens = max(0, state.policy.max_tokens - used_tokens)
        return BudgetSnapshot(
            scope_id=scope_id,
            policy=state.policy,
            calls=state.calls,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            reserved_calls=state.reserved_calls,
            reserved_tokens=state.reserved_tokens,
            rejected_reservations=state.rejected_reservations,
            unknown_usage_records=state.unknown_usage_records,
            closed=state.closed,
            blocked_reason=state.blocked_reason,
            remaining_calls=remaining_calls,
            remaining_tokens=remaining_tokens,
        )

    def snapshot(self, scope_id: str) -> BudgetSnapshot | None:
        normalized = self._scope_id(scope_id)
        with self._lock:
            state = self._scopes.get(normalized)
            if state is None:
                return None
            return self._snapshot_locked(normalized, state)

    def snapshots(self) -> tuple[BudgetSnapshot, ...]:
        with self._lock:
            return tuple(
                self._snapshot_locked(scope_id, state)
                for scope_id, state in sorted(self._scopes.items())
            )

    def register_scope(
        self,
        scope_id: str,
        policy: ProviderBudgetPolicy | None = None,
    ) -> BudgetSnapshot:
        normalized = self._scope_id(scope_id)
        if policy is not None and not isinstance(policy, ProviderBudgetPolicy):
            raise LimitConfigurationError("policy must be a ProviderBudgetPolicy")
        with self._lock:
            now = self._now_locked()
            self._prune_locked(now)
            state = self._scopes.get(normalized)
            if state is not None:
                if policy is not None and state.policy != policy:
                    raise LimitConfigurationError(
                        "a scope's provider budget policy cannot change after registration"
                    )
                state.last_seen = now
                return self._snapshot_locked(normalized, state)
            if len(self._scopes) >= self.max_scopes:
                raise LimitConfigurationError("provider budget scope capacity reached")
            selected_policy = self.default_policy if policy is None else policy
            state = _BudgetState(
                policy=selected_policy,
                calls=0,
                input_tokens=0,
                output_tokens=0,
                reserved_calls=0,
                reserved_tokens=0,
                rejected_reservations=0,
                unknown_usage_records=0,
                closed=False,
                blocked_reason=None,
                last_seen=now,
            )
            self._scopes[normalized] = state
            return self._snapshot_locked(normalized, state)

    def _reject_locked(
        self,
        reason: str,
        scope_id: str,
        state: _BudgetState | None,
    ) -> BudgetReserveDecision:
        self._rejected_reservations += 1
        snapshot = None
        if state is not None:
            state.rejected_reservations += 1
            snapshot = self._snapshot_locked(scope_id, state)
        return BudgetReserveDecision(
            allowed=False,
            reason=reason,
            reservation=None,
            snapshot=snapshot,
        )

    def _new_reservation_id_locked(self) -> str:
        # A broken injected factory must fail closed instead of overwriting an
        # existing in-flight reservation.  Four attempts bound the operation.
        for _ in range(4):
            reservation_id = _bounded_key(
                "reservation_id",
                self._reservation_id_factory(),
                max_length=256,
            )
            if reservation_id not in self._reservations:
                return reservation_id
        raise LimitConfigurationError("reservation_id_factory produced repeated identifiers")

    def try_reserve(
        self,
        scope_id: str,
        *,
        token_budget: int | None = None,
        policy: ProviderBudgetPolicy | None = None,
    ) -> BudgetReserveDecision:
        """Atomically reserve one call and an output-inclusive token bound.

        When ``max_tokens`` is configured, ``token_budget`` is mandatory and
        positive.  This prevents an admitted call with an unbounded or
        unaccounted response.  The caller should use a provider-enforced upper
        bound, not a monetary estimate.
        """
        normalized = self._scope_id(scope_id)
        if token_budget is not None:
            parsed_token_budget = _usage_count("token_budget", token_budget)
        else:
            parsed_token_budget = 0
        if policy is not None and not isinstance(policy, ProviderBudgetPolicy):
            raise LimitConfigurationError("policy must be a ProviderBudgetPolicy")

        with self._lock:
            now = self._now_locked()
            self._prune_locked(now)
            state = self._scopes.get(normalized)
            new_scope = state is None
            if state is None:
                if len(self._scopes) >= self.max_scopes:
                    return self._reject_locked("scope_capacity", normalized, None)
                selected_policy = self.default_policy if policy is None else policy
                state = _BudgetState(
                    policy=selected_policy,
                    calls=0,
                    input_tokens=0,
                    output_tokens=0,
                    reserved_calls=0,
                    reserved_tokens=0,
                    rejected_reservations=0,
                    unknown_usage_records=0,
                    closed=False,
                    blocked_reason=None,
                    last_seen=now,
                )
            elif policy is not None and state.policy != policy:
                raise LimitConfigurationError(
                    "a scope's provider budget policy cannot change after registration"
                )
            state.last_seen = now

            if state.closed:
                return self._reject_locked(
                    "scope_closed",
                    normalized,
                    None if new_scope else state,
                )
            if state.blocked_reason is not None:
                return self._reject_locked(
                    state.blocked_reason,
                    normalized,
                    None if new_scope else state,
                )
            if len(self._reservations) >= self.max_inflight_reservations:
                return self._reject_locked(
                    "reservation_capacity",
                    normalized,
                    None if new_scope else state,
                )
            if state.policy.max_calls is not None and (
                state.calls + state.reserved_calls + 1 > state.policy.max_calls
            ):
                return self._reject_locked(
                    "call_limit",
                    normalized,
                    None if new_scope else state,
                )
            if state.policy.max_tokens is not None:
                if token_budget is None or parsed_token_budget <= 0:
                    return self._reject_locked(
                        "token_budget_required",
                        normalized,
                        None if new_scope else state,
                    )
                if (
                    state.input_tokens
                    + state.output_tokens
                    + state.reserved_tokens
                    + parsed_token_budget
                    > state.policy.max_tokens
                ):
                    return self._reject_locked(
                        "token_limit",
                        normalized,
                        None if new_scope else state,
                    )

            reservation = BudgetReservation(
                reservation_id=self._new_reservation_id_locked(),
                scope_id=normalized,
                token_budget=parsed_token_budget if token_budget is not None else None,
            )
            if new_scope:
                self._scopes[normalized] = state
            self._reservations[reservation.reservation_id] = _ReservationState(
                reservation=reservation
            )
            state.reserved_calls += 1
            state.reserved_tokens += parsed_token_budget
            return BudgetReserveDecision(
                allowed=True,
                reason="allowed",
                reservation=reservation,
                snapshot=self._snapshot_locked(normalized, state),
            )

    def reserve(
        self,
        scope_id: str,
        *,
        token_budget: int | None = None,
        policy: ProviderBudgetPolicy | None = None,
    ) -> BudgetReservation:
        decision = self.try_reserve(
            scope_id,
            token_budget=token_budget,
            policy=policy,
        )
        if not decision.allowed or decision.reservation is None:
            raise BudgetRejectedError(decision)
        return decision.reservation

    def _reservation_locked(
        self,
        reservation: BudgetReservation | str,
    ) -> tuple[_ReservationState, _BudgetState]:
        reservation_id = (
            reservation.reservation_id
            if isinstance(reservation, BudgetReservation)
            else reservation
        )
        normalized_id = _bounded_key("reservation_id", reservation_id, max_length=256)
        item = self._reservations.get(normalized_id)
        if item is None:
            raise UnknownReservationError(normalized_id)
        if isinstance(reservation, BudgetReservation) and item.reservation != reservation:
            raise UnknownReservationError(normalized_id)
        state = self._scopes.get(item.reservation.scope_id)
        if state is None:
            raise RuntimeError("provider budget ledger lost a live reservation scope")
        return item, state

    def record(
        self,
        reservation: BudgetReservation | str,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> BudgetRecordResult:
        """Finalize one provider attempt using reported token usage.

        Missing token counts never become guessed values.  With a hard token
        limit, incomplete usage blocks future calls for the scope.  Known
        components and the call itself are still recorded.
        """
        parsed_input = None if input_tokens is None else _usage_count("input_tokens", input_tokens)
        parsed_output = (
            None if output_tokens is None else _usage_count("output_tokens", output_tokens)
        )
        with self._lock:
            now = self._now_locked()
            item, state = self._reservation_locked(reservation)
            owned = item.reservation
            del self._reservations[owned.reservation_id]
            state.reserved_calls -= 1
            state.reserved_tokens -= owned.token_budget or 0
            state.calls += 1
            if parsed_input is not None:
                state.input_tokens += parsed_input
            if parsed_output is not None:
                state.output_tokens += parsed_output
            state.last_seen = now

            if parsed_input is None or parsed_output is None:
                state.unknown_usage_records += 1
                if state.policy.max_tokens is not None:
                    state.blocked_reason = "usage_unknown"
                    reason = "usage_unknown"
                    accepted = False
                else:
                    reason = "recorded_with_unknown_usage"
                    accepted = True
            else:
                actual_tokens = parsed_input + parsed_output
                accepted = owned.token_budget is None or actual_tokens <= owned.token_budget
                reason = "recorded" if accepted else "usage_exceeded_reservation"
                if (
                    state.policy.max_tokens is not None
                    and state.input_tokens + state.output_tokens > state.policy.max_tokens
                ):
                    state.blocked_reason = "token_limit_exceeded"
                    accepted = False
                    reason = "token_limit_exceeded"

            return BudgetRecordResult(
                accepted=accepted,
                reason=reason,
                snapshot=self._snapshot_locked(owned.scope_id, state),
            )

    def cancel(self, reservation: BudgetReservation | str) -> BudgetSnapshot:
        """Release a reservation only when provider transport did not start."""
        with self._lock:
            now = self._now_locked()
            item, state = self._reservation_locked(reservation)
            owned = item.reservation
            del self._reservations[owned.reservation_id]
            state.reserved_calls -= 1
            state.reserved_tokens -= owned.token_budget or 0
            state.last_seen = now
            return self._snapshot_locked(owned.scope_id, state)

    def close_scope(self, scope_id: str) -> BudgetSnapshot:
        """Reject future reservations while allowing in-flight finalization."""
        normalized = self._scope_id(scope_id)
        with self._lock:
            now = self._now_locked()
            state = self._scopes.get(normalized)
            if state is None:
                raise KeyError(normalized)
            state.closed = True
            state.last_seen = now
            return self._snapshot_locked(normalized, state)

    async def try_reserve_async(
        self,
        scope_id: str,
        *,
        token_budget: int | None = None,
        policy: ProviderBudgetPolicy | None = None,
    ) -> BudgetReserveDecision:
        return self.try_reserve(scope_id, token_budget=token_budget, policy=policy)

    async def reserve_async(
        self,
        scope_id: str,
        *,
        token_budget: int | None = None,
        policy: ProviderBudgetPolicy | None = None,
    ) -> BudgetReservation:
        return self.reserve(scope_id, token_budget=token_budget, policy=policy)

    async def record_async(
        self,
        reservation: BudgetReservation | str,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> BudgetRecordResult:
        return self.record(
            reservation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def cancel_async(self, reservation: BudgetReservation | str) -> BudgetSnapshot:
        return self.cancel(reservation)


__all__ = [
    "AdmissionLimiter",
    "BudgetRecordResult",
    "BudgetRejectedError",
    "BudgetReservation",
    "BudgetReserveDecision",
    "BudgetSnapshot",
    "InvalidLimitKey",
    "LimitConfigurationError",
    "ProviderBudgetLedger",
    "ProviderBudgetPolicy",
    "RateLimitConfig",
    "RateLimitDecision",
    "TokenBucketRateLimiter",
    "UnknownReservationError",
]
