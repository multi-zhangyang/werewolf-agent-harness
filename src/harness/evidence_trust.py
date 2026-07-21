"""Process-local provenance markers for derived summary evidence.

JSON serialization can preserve a digest and a boolean, but it cannot preserve
the fact that the values were derived from a transcript that was validated in
this process.  This module keeps that fact outside the serialized model.  The
marker is deliberately private: callers obtain it only through the result
factory or a verifier-backed artifact loader.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


_ATTESTATION_ATTRIBUTE = "_evaluation_attestation_digest"


def _mark_verified(value: Any) -> Any:
    """Register one in-memory value as transcript-derived evidence."""

    setattr(value, _ATTESTATION_ATTRIBUTE, _content_digest(value))
    return value


def is_verified(value: Any) -> bool:
    """Return whether ``value`` is the unchanged object previously registered."""

    expected = getattr(value, _ATTESTATION_ATTRIBUTE, None)
    if not isinstance(expected, str):
        return False
    try:
        return expected == _content_digest(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _content_digest(value: Any) -> str:
    body = _canonical(value)
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _canonical(dump(mode="json", exclude_none=False))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonical(item) for item in value)
    return value
