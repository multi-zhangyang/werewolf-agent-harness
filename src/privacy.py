"""Dependency-free helpers for keeping model-private reasoning private.

Model reasoning is an admin trace concern, not a game-event field. The
runtime passes JSON-like dictionaries through several boundaries, so every
boundary uses this recursive copier instead of relying on callers to remove
only top-level keys.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MODEL_PRIVATE_REASONING_KEYS = frozenset({
    "reasoning",
    "thought",
    "private_reasoning",
})


def strip_model_private_reasoning(value: Any) -> Any:
    """Return a detached value with model-private reasoning keys removed.

    Event and observation payloads are JSON-like, but handling tuples as well
    makes the helper safe for internal callers that have not serialized yet.
    Mapping keys are compared only when they are strings, which covers normal
    JSON objects without changing unrelated structured keys.
    """
    if isinstance(value, Mapping):
        return {
            key: strip_model_private_reasoning(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key in MODEL_PRIVATE_REASONING_KEYS)
        }
    if isinstance(value, list):
        return [strip_model_private_reasoning(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_model_private_reasoning(item) for item in value)
    return value
