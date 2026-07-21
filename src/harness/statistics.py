"""Reusable statistics helpers for harness experiment summaries."""
from __future__ import annotations

import math
import random
from typing import Any


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def numeric_values(values: list[Any] | tuple[Any, ...] | Any) -> list[float]:
    return [num for value in values if (num := as_float(value)) is not None]


def bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int = 2000,
    seed: int = 20260705,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1 or iterations <= 0:
        return (values[0], values[0])
    rng = random.Random(seed + len(values))
    means: list[float] = []
    n = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int(0.025 * (len(means) - 1)))
    hi_idx = min(len(means) - 1, int(0.975 * (len(means) - 1)))
    return means[lo_idx], means[hi_idx]


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def router_stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    b_calls = as_float(before.get("calls")) or 0.0
    a_calls = as_float(after.get("calls")) or 0.0
    calls_delta = a_calls - b_calls
    b_latency = as_float(before.get("total_latency"))
    a_latency = as_float(after.get("total_latency"))
    for key in sorted(set(before) | set(after)):
        if key == "avg_latency":
            continue
        b = as_float(before.get(key))
        a = as_float(after.get(key))
        if a is None:
            continue
        delta[key] = round(a - (b or 0.0), 6)
    if a_latency is not None and b_latency is not None and calls_delta > 0:
        delta["avg_latency"] = round((a_latency - b_latency) / calls_delta, 6)
    return delta
