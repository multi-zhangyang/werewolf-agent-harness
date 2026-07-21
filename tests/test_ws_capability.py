"""WebSocket capability transport contract tests."""
from __future__ import annotations

import pytest

from src.api.server import (
    WS_CAPABILITY_PROTOCOL_PREFIX,
    WS_PROTOCOL,
    _websocket_capability,
)


class _HeadersSocket:
    def __init__(self, protocols: str | None = None) -> None:
        self.headers = {}
        if protocols is not None:
            self.headers["sec-websocket-protocol"] = protocols


def test_subprotocol_capability_is_selected_without_url_token() -> None:
    token = "capability-for-unit-test"
    socket = _HeadersSocket(f"{WS_PROTOCOL}, {WS_CAPABILITY_PROTOCOL_PREFIX}{token}")

    resolved, selected, malformed = _websocket_capability(socket, None)

    assert resolved == token
    assert selected == WS_PROTOCOL
    assert malformed is False


def test_query_token_remains_compatible_when_no_capability_protocol_is_offered() -> None:
    socket = _HeadersSocket(WS_PROTOCOL)

    resolved, selected, malformed = _websocket_capability(socket, "legacy-query-capability")

    assert resolved == "legacy-query-capability"
    assert selected == WS_PROTOCOL
    assert malformed is False


def test_matching_query_and_subprotocol_capabilities_are_allowed() -> None:
    token = "same-capability"
    socket = _HeadersSocket(
        f"{WS_PROTOCOL}, {WS_CAPABILITY_PROTOCOL_PREFIX}{token}"
    )

    resolved, selected, malformed = _websocket_capability(socket, token)

    assert resolved == token
    assert selected == WS_PROTOCOL
    assert malformed is False


def test_conflicting_query_and_subprotocol_capabilities_fail_closed() -> None:
    socket = _HeadersSocket(
        f"{WS_PROTOCOL}, {WS_CAPABILITY_PROTOCOL_PREFIX}header-capability"
    )

    resolved, selected, malformed = _websocket_capability(socket, "query-capability")

    assert resolved == "header-capability"
    assert selected == WS_PROTOCOL
    assert malformed is True


@pytest.mark.parametrize(
    "protocols",
    [
        f"{WS_CAPABILITY_PROTOCOL_PREFIX}",
        f"{WS_CAPABILITY_PROTOCOL_PREFIX}one,{WS_CAPABILITY_PROTOCOL_PREFIX}two",
        f"{WS_CAPABILITY_PROTOCOL_PREFIX}{'x' * 257}",
    ],
)
def test_malformed_capability_protocols_fail_closed(protocols: str) -> None:
    socket = _HeadersSocket(protocols)

    _resolved, _selected, malformed = _websocket_capability(socket, None)

    assert malformed is True
