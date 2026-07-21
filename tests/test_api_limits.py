"""Process-local admission and provider budget tests."""

from __future__ import annotations

import asyncio
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.api.limits import (
    AdmissionLimiter,
    BudgetRejectedError,
    InvalidLimitKey,
    LimitConfigurationError,
    ProviderBudgetLedger,
    ProviderBudgetPolicy,
    RateLimitConfig,
    TokenBucketRateLimiter,
    UnknownReservationError,
)


class _Clock:
    def __init__(self, initial: float = 100.0) -> None:
        self.value = initial
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.value += seconds

    def set(self, value: float) -> None:
        with self._lock:
            self.value = value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capacity", 0),
        ("capacity", math.inf),
        ("capacity", True),
        ("refill_rate", -1),
        ("refill_rate", math.nan),
        ("key_ttl_seconds", 0),
        ("max_keys", 0),
        ("max_keys", True),
        ("max_key_length", -1),
    ],
)
def test_rate_limit_config_rejects_invalid_values(field: str, value: object) -> None:
    kwargs: dict[str, object] = {"capacity": 2, "refill_rate": 1}
    kwargs[field] = value
    with pytest.raises(LimitConfigurationError):
        RateLimitConfig(**kwargs)  # type: ignore[arg-type]


def test_token_bucket_burst_refill_and_retry_after_are_monotonic() -> None:
    clock = _Clock()
    limiter = TokenBucketRateLimiter(
        RateLimitConfig(capacity=2, refill_rate=0.5),
        clock=clock,
    )

    first = limiter.admit_rest("client-a")
    second = limiter.admit_rest("client-a")
    denied = limiter.admit_rest("client-a")

    assert first.allowed and first.remaining_tokens == pytest.approx(1)
    assert second.allowed and second.remaining_tokens == pytest.approx(0)
    assert not denied.allowed
    assert denied.reason == "rate_limited"
    assert denied.retry_after_seconds == pytest.approx(2)

    clock.advance(1)
    partial = limiter.admit_rest("client-a")
    assert not partial.allowed
    assert partial.remaining_tokens == pytest.approx(0.5)
    assert partial.retry_after_seconds == pytest.approx(1)

    clock.advance(1)
    assert limiter.admit_rest("client-a").allowed

    # A faulty clock moving backwards cannot mint new admission tokens.
    clock.set(0)
    backwards = limiter.admit_rest("client-a")
    assert not backwards.allowed
    assert backwards.reason == "rate_limited"


def test_rest_ws_and_identity_buckets_are_isolated() -> None:
    limiter = TokenBucketRateLimiter(RateLimitConfig(capacity=1, refill_rate=1))

    assert limiter.admit_rest("same-identity").allowed
    assert not limiter.admit_rest("same-identity").allowed
    assert limiter.admit_ws("same-identity").allowed
    assert limiter.admit_rest("other-identity").allowed


def test_admission_limiter_uses_independent_rest_and_ws_policies() -> None:
    limiter = AdmissionLimiter(
        rest=RateLimitConfig(capacity=1, refill_rate=1),
        ws=RateLimitConfig(capacity=2, refill_rate=1),
    )

    assert [limiter.admit_rest("ip").allowed for _ in range(2)] == [True, False]
    assert [limiter.admit_ws("ip").allowed for _ in range(3)] == [True, True, False]


def test_rate_limiter_retention_is_bounded_and_stale_keys_expire() -> None:
    clock = _Clock()
    limiter = TokenBucketRateLimiter(
        RateLimitConfig(
            capacity=1,
            refill_rate=1,
            key_ttl_seconds=10,
            max_keys=2,
        ),
        clock=clock,
    )

    assert limiter.admit("a").allowed
    assert limiter.admit("b").allowed
    at_capacity = limiter.admit("attacker-key")
    assert not at_capacity.allowed
    assert at_capacity.reason == "key_capacity"
    assert limiter.key_count == 2

    clock.advance(10)
    assert limiter.admit("c").allowed
    assert limiter.key_count == 1
    # Expiry intentionally resets an idle identity's bucket.
    assert limiter.admit("a").allowed


def test_invalid_or_oversized_admissions_fail_closed_without_retention() -> None:
    limiter = TokenBucketRateLimiter(
        RateLimitConfig(capacity=2, refill_rate=1, max_key_length=4)
    )

    for key in ("", "   ", "abcde"):
        with pytest.raises(InvalidLimitKey):
            limiter.admit(key)
    with pytest.raises(LimitConfigurationError):
        limiter.admit("good", weight=0)

    oversized = limiter.admit("good", weight=3)
    assert not oversized.allowed
    assert oversized.reason == "weight_exceeds_capacity"
    assert math.isinf(oversized.retry_after_seconds)
    assert limiter.key_count == 0


def test_threaded_rate_admission_is_atomic() -> None:
    limiter = TokenBucketRateLimiter(RateLimitConfig(capacity=25, refill_rate=0.000001))
    gate = threading.Barrier(100)

    def attempt(_: int) -> bool:
        gate.wait()
        return limiter.admit_rest("shared").allowed

    with ThreadPoolExecutor(max_workers=100) as pool:
        admitted = list(pool.map(attempt, range(100)))

    assert sum(admitted) == 25


@pytest.mark.asyncio
async def test_async_admission_uses_the_same_atomic_counters() -> None:
    limiter = TokenBucketRateLimiter(RateLimitConfig(capacity=7, refill_rate=0.000001))

    decisions = await asyncio.gather(
        *(limiter.admit_ws_async("shared") for _ in range(40))
    )

    assert sum(item.allowed for item in decisions) == 7
    assert limiter.key_count == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_calls", -1),
        ("max_calls", 1.5),
        ("max_calls", True),
        ("max_tokens", -1),
        ("max_tokens", 3.5),
        ("max_tokens", False),
    ],
)
def test_provider_budget_policy_rejects_invalid_limits(field: str, value: object) -> None:
    with pytest.raises(LimitConfigurationError):
        ProviderBudgetPolicy(**{field: value})  # type: ignore[arg-type]


def test_budget_reserve_and_record_tracks_provider_reported_usage() -> None:
    policy = ProviderBudgetPolicy(max_calls=2, max_tokens=100)
    ledger = ProviderBudgetLedger(policy)
    scope = ledger.run_scope("run-1")

    reservation = ledger.reserve(scope, token_budget=60)
    pending = ledger.snapshot(scope)
    assert pending is not None
    assert pending.calls == 0
    assert pending.reserved_calls == 1
    assert pending.reserved_tokens == 60
    assert pending.remaining_calls == 1
    assert pending.remaining_tokens == 40

    recorded = ledger.record(reservation, input_tokens=20, output_tokens=15)
    assert recorded.accepted
    assert recorded.reason == "recorded"
    assert recorded.snapshot.calls == 1
    assert recorded.snapshot.input_tokens == 20
    assert recorded.snapshot.output_tokens == 15
    assert recorded.snapshot.total_tokens == 35
    assert recorded.snapshot.reserved_calls == 0
    assert recorded.snapshot.remaining_tokens == 65


def test_custom_registered_policy_is_reused_when_reserve_omits_policy() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=99))
    custom = ProviderBudgetPolicy(max_calls=1, max_tokens=10)
    ledger.register_scope("room:r1", custom)

    reservation = ledger.reserve("room:r1", token_budget=10)
    ledger.record(reservation, input_tokens=4, output_tokens=3)
    denied = ledger.try_reserve("room:r1", token_budget=1)

    assert not denied.allowed
    assert denied.reason == "call_limit"
    with pytest.raises(LimitConfigurationError):
        ledger.register_scope("room:r1", ProviderBudgetPolicy(max_calls=2))


def test_budget_call_and_token_limits_include_inflight_reservations() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=2, max_tokens=100))

    first = ledger.reserve("run:r", token_budget=60)
    token_denied = ledger.try_reserve("run:r", token_budget=50)
    assert not token_denied.allowed
    assert token_denied.reason == "token_limit"

    second = ledger.reserve("run:r", token_budget=40)
    call_denied = ledger.try_reserve("run:r", token_budget=1)
    assert not call_denied.allowed
    assert call_denied.reason == "call_limit"
    assert ledger.inflight_reservation_count == 2

    ledger.cancel(first)
    ledger.cancel(second)


def test_token_limited_scope_requires_a_positive_provider_bound() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_tokens=20))

    missing = ledger.try_reserve("run:r")
    zero = ledger.try_reserve("run:r", token_budget=0)
    over = ledger.try_reserve("run:r", token_budget=21)

    assert missing.reason == "token_budget_required"
    assert zero.reason == "token_budget_required"
    assert over.reason == "token_limit"
    assert ledger.inflight_reservation_count == 0
    assert ledger.rejected_reservations == 3
    assert ledger.scope_count == 0
    with pytest.raises(BudgetRejectedError) as raised:
        ledger.reserve("run:r", token_budget=21)
    assert raised.value.decision.reason == "token_limit"
    assert ledger.scope_count == 0


def test_cancel_releases_capacity_without_counting_a_provider_call() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=1, max_tokens=10))
    reservation = ledger.reserve("run:r", token_budget=10)

    released = ledger.cancel(reservation)
    assert released.calls == 0
    assert released.reserved_calls == 0
    assert released.remaining_calls == 1
    assert released.remaining_tokens == 10
    replacement = ledger.reserve("run:r", token_budget=10)
    with pytest.raises(UnknownReservationError):
        ledger.cancel(reservation)
    ledger.cancel(replacement)


def test_failed_provider_attempt_is_still_recorded_as_a_call() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=1))
    reservation = ledger.reserve("run:r")

    result = ledger.record(reservation, input_tokens=0, output_tokens=0)

    assert result.accepted
    assert result.snapshot.calls == 1
    assert ledger.try_reserve("run:r").reason == "call_limit"


def test_usage_above_reservation_is_accounted_and_reported() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=3, max_tokens=100))
    reservation = ledger.reserve("run:r", token_budget=20)

    result = ledger.record(reservation, input_tokens=18, output_tokens=12)

    assert not result.accepted
    assert result.reason == "usage_exceeded_reservation"
    assert result.snapshot.calls == 1
    assert result.snapshot.total_tokens == 30
    assert result.snapshot.remaining_tokens == 70
    # The ledger charged the real usage, but the hard scope maximum was not
    # crossed, so a correctly bounded later call may still be admitted.
    assert ledger.try_reserve("run:r", token_budget=70).allowed


def test_actual_usage_crossing_hard_limit_blocks_the_scope_without_hiding_usage() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=3, max_tokens=20))
    reservation = ledger.reserve("run:r", token_budget=20)

    result = ledger.record(reservation, input_tokens=14, output_tokens=9)

    assert not result.accepted
    assert result.reason == "token_limit_exceeded"
    assert result.snapshot.total_tokens == 23
    assert result.snapshot.blocked_reason == "token_limit_exceeded"
    assert ledger.try_reserve("run:r", token_budget=1).reason == "token_limit_exceeded"


def test_unknown_usage_never_invents_tokens_and_blocks_a_token_limited_scope() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=3, max_tokens=50))
    reservation = ledger.reserve("run:r", token_budget=50)

    result = ledger.record(reservation, input_tokens=10, output_tokens=None)

    assert not result.accepted
    assert result.reason == "usage_unknown"
    assert result.snapshot.calls == 1
    assert result.snapshot.input_tokens == 10
    assert result.snapshot.output_tokens == 0
    assert result.snapshot.unknown_usage_records == 1
    assert result.snapshot.blocked_reason == "usage_unknown"
    assert ledger.try_reserve("run:r", token_budget=1).reason == "usage_unknown"


def test_unknown_usage_is_visible_but_does_not_block_call_only_policy() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=2))
    reservation = ledger.reserve("run:r")

    result = ledger.record(reservation, input_tokens=None, output_tokens=None)

    assert result.accepted
    assert result.reason == "recorded_with_unknown_usage"
    assert result.snapshot.unknown_usage_records == 1
    assert ledger.try_reserve("run:r").allowed


def test_budget_scope_retention_is_bounded_and_only_closed_scopes_expire() -> None:
    clock = _Clock()
    ledger = ProviderBudgetLedger(
        ProviderBudgetPolicy(max_calls=1),
        max_scopes=2,
        closed_scope_ttl_seconds=10,
        clock=clock,
    )
    ledger.register_scope("run:active")
    ledger.register_scope("run:closed")
    ledger.close_scope("run:closed")

    full = ledger.try_reserve("run:new")
    assert not full.allowed
    assert full.reason == "scope_capacity"

    clock.advance(10)
    assert ledger.prune() == 1
    assert ledger.snapshot("run:active") is not None
    assert ledger.snapshot("run:closed") is None
    assert ledger.try_reserve("run:new").allowed


def test_closed_scope_rejects_new_calls_but_inflight_usage_can_finalize() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=2, max_tokens=20))
    reservation = ledger.reserve("room:r", token_budget=10)

    closed = ledger.close_scope("room:r")
    assert closed.closed
    assert ledger.try_reserve("room:r", token_budget=1).reason == "scope_closed"
    finalized = ledger.record(reservation, input_tokens=3, output_tokens=2)
    assert finalized.accepted
    assert finalized.snapshot.closed
    assert finalized.snapshot.calls == 1


def test_threaded_budget_reservations_cannot_oversubscribe_call_limit() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=20))
    gate = threading.Barrier(100)

    def attempt(_: int):
        gate.wait()
        return ledger.try_reserve("run:shared")

    with ThreadPoolExecutor(max_workers=100) as pool:
        decisions = list(pool.map(attempt, range(100)))

    allowed = [item for item in decisions if item.allowed]
    assert len(allowed) == 20
    assert ledger.inflight_reservation_count == 20
    snapshot = ledger.snapshot("run:shared")
    assert snapshot is not None
    assert snapshot.reserved_calls == 20
    assert snapshot.rejected_reservations == 80


@pytest.mark.asyncio
async def test_async_budget_methods_share_atomic_state() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=5, max_tokens=50))

    decisions = await asyncio.gather(
        *(ledger.try_reserve_async("run:shared", token_budget=10) for _ in range(20))
    )
    reservations = [item.reservation for item in decisions if item.allowed]
    assert len(reservations) == 5
    results = await asyncio.gather(
        *(
            ledger.record_async(item, input_tokens=3, output_tokens=2)
            for item in reservations
            if item is not None
        )
    )

    assert all(item.accepted for item in results)
    snapshot = ledger.snapshot("run:shared")
    assert snapshot is not None
    assert snapshot.calls == 5
    assert snapshot.total_tokens == 25
    assert snapshot.reserved_calls == 0


def test_reservation_ids_are_unique_and_duplicate_factory_fails_closed() -> None:
    ledger = ProviderBudgetLedger(
        ProviderBudgetPolicy(max_calls=2),
        reservation_id_factory=lambda: "same",
    )
    first = ledger.reserve("run:r")

    with pytest.raises(LimitConfigurationError):
        ledger.reserve("run:r")

    snapshot = ledger.snapshot("run:r")
    assert snapshot is not None
    assert snapshot.reserved_calls == 1
    assert ledger.inflight_reservation_count == 1
    ledger.cancel(first)
