from __future__ import annotations

from pathlib import Path


def test_frontend_does_not_persist_plaintext_room_capabilities() -> None:
    """Capability credentials stay in the App lifetime, never localStorage."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "frontend/src/App.tsx").read_text(encoding="utf-8")

    assert "useState<Record<string, RoomAuth>>(() =>" in source
    assert "localStorage.setItem(\"werewolf.mas.roomAuth.v1\"" not in source
    assert "JSON.stringify(roomAuth)" not in source
    assert "window.localStorage.removeItem(\"werewolf.mas.roomAuth.v1\")" in source
    # A reload may retain the room identifier, but must not recreate a
    # privileged game screen without a fresh in-memory capability.
    assert 'return { name: "room", roomId: raw.roomId };' in source


def test_frontend_scopes_admin_capability_to_god_and_replay_surfaces() -> None:
    """Spectator/player components must not receive the admin capability."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "frontend/src/App.tsx").read_text(encoding="utf-8")

    assert 'const adminToken = mode === "god" || mode === "replay" ? auth?.admin_token : undefined;' in source
    assert 'setScreen({ name: "game", roomId, seat, mode, token, adminToken });' in source
    assert 'setScreen({ name: "game", roomId, seat, mode, token, adminToken: auth?.admin_token });' not in source
    assert 'if (!screen.adminToken) {' in source


def test_game_return_preserves_current_room_capability_for_terminal_replay() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "frontend/src/App.tsx").read_text(encoding="utf-8")

    assert "const backToRoom = useCallback" in source
    assert '? { name: "room", roomId: current.roomId }' in source
    assert source.count("onBack={backToRoom}") == 2
    assert 'aria-label="返回房间"' in (
        root / "frontend/src/views/GameView.tsx"
    ).read_text(encoding="utf-8")
