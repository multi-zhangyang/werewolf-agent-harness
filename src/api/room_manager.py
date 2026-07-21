"""房间管理器 —— 房间生命周期 + 游戏循环 + 信息隔离广播。

承 ARCHITECTURE.md §6:每个事件带 visibility+recipients,按 seat 过滤后下发。
spectate 只收公开事件;play 收该 seat 私有+公开;god/replay 收全知对局事件。
模型私有推理只通过 admin capability 保护的 decision trace 提供。
人类席位超时成为透明 SKIP/失败结算，不生成替代动作。
"""
from __future__ import annotations

import asyncio
from collections import deque
import inspect
import secrets
import json
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from ..config import (
    AGENT_DECISION_TIMEOUT,
    AGENT_DECISION_TIMEOUT_BY_PHASE,
    AGENT_PHASE_DEADLINE,
    AGENT_PHASE_DEADLINE_BY_PHASE,
    LLM_CONCURRENCY,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
    MAX_ROOMS,
    PROVIDER_BUDGET_CLOSED_SCOPE_TTL_SECONDS,
    PROVIDER_BUDGET_MAX_INFLIGHT_RESERVATIONS,
    PROVIDER_BUDGET_MAX_SCOPES,
    PROVIDER_BUDGET_POLICY,
    REST_RATE_LIMIT_CONFIG,
    TERMINAL_ROOM_TTL,
    WS_RATE_LIMIT_CONFIG,
)
from ..agent.actor import AgentActor
from ..environments.werewolf.plugin import WerewolfEnvironmentPlugin
from ..game.models import EventVisibility, GameState, Phase
from ..game.orchestrator import DEFAULT_TURN_POLICY, GameOrchestratorV2 as GameOrchestrator, build_actors
from ..game.roles import Role, default_role_deck
from ..game.rules import RulesEngine
from ..game.state import new_game
from ..harness.core_runner import (
    EnvironmentRunResult,
    PreparedEnvironmentRun,
    environment_cancellation_budget_seconds,
    run_prepared_environment_run,
)
from ..harness.core_spec import ActorSpec, CoreRunSpec
from ..harness.decision_runtime import DecisionRuntime
from ..harness.environment import (
    AgentRegistry,
    EnvironmentRunContext,
    EnvironmentRunEvidence,
)
from ..harness.spec_loader import legacy_werewolf_run_to_core
from ..harness.spec import ModelConfigManifest, RunSpec
from ..harness.transcript import (
    TRANSCRIPT_SCHEMA_VERSION,
    HarnessEvent,
    Transcript,
    payload_digest,
    redact_sensitive,
)
from ..harness.visibility import project_payload_for_audience
from ..llm.models import ModelConfig
from ..llm.router import LLMRouter
from ..privacy import strip_model_private_reasoning
from .persistence import (
    PERSISTENCE_SCHEMA_VERSION,
    PersistenceError,
    RoomPersistence,
    SQLiteRoomPersistence,
    hash_capability,
    verify_capability,
)
from .limits import AdmissionLimiter, ProviderBudgetLedger, ProviderBudgetPolicy

logger = logging.getLogger(__name__)

STANDARD_LLM_PROTOCOLS = {"openai", "openai_responses", "anthropic"}
TERMINAL_ROOM_STATUSES = {
    "ended",
    "incomplete",
    "failed",
    "timeout",
    "cancelled",
    "interrupted",
}
DEFAULT_WS_CLIENT_QUEUE_SIZE = 128
DEFAULT_WS_DELIVERY_HISTORY_SIZE = 4096
DEFAULT_MAX_WS_CLIENTS_PER_ROOM = 256
MAX_WS_CLIENTS_PER_ROOM_HARD_LIMIT = 100_000
DEFAULT_ROOM_CANCELLATION_GRACE_SECONDS = 1.0
DEFAULT_ROOM_CLEANUP_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_ROOM_EVIDENCE_ENTRIES = 20_000
MAX_ROOM_EVIDENCE_ENTRIES_HARD_LIMIT = 1_000_000


class RoomManagerUnavailableError(RuntimeError):
    """The manager is draining or closed and cannot accept new work."""


class RoomCapacityError(RuntimeError):
    """The configured retained-room capacity has been reached."""


class RoomClientCapacityError(RuntimeError):
    """A room cannot retain another live WebSocket client."""


class RoomInUseError(RuntimeError):
    """A room cannot be explicitly removed while it is active."""


class RoomEvidenceLimitError(RuntimeError):
    """A room reached its bounded event/decision evidence capacity."""

    def __init__(self, *, current: int, limit: int) -> None:
        self.current = int(current)
        self.limit = int(limit)
        super().__init__(
            f"room evidence limit reached (current={self.current}, limit={self.limit})"
        )


class CapabilityAuthorizationError(PermissionError):
    """A capability changed or was revoked before an operation committed."""


class RoomManagerCleanupError(RuntimeError):
    """Shutdown left explicit cleanup failures or in-process tasks behind."""

    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failures = [dict(item) for item in failures]
        self.pending_task_count = sum(
            int(item.get("pending_task_count") or 0) for item in failures
        )
        self.fatal_cleanup_failure = True
        super().__init__(
            "room manager cleanup failed "
            f"({len(self.failures)} failure(s), "
            f"{self.pending_task_count} pending task(s))"
        )


class _RoomRunDeadlineExceeded(TimeoutError):
    """The manager-owned wall-clock deadline expired."""


class _RoomRunCleanupTimeout(RuntimeError):
    """The orchestrator ignored bounded cancellation after its deadline."""

    def __init__(self, failure: dict[str, Any]) -> None:
        self.failure = dict(failure)
        super().__init__("orchestrator ignored bounded cancellation")


class InvalidDeliveryCursorError(ValueError):
    """A WebSocket resume cursor is malformed or outside the stream."""


class FutureDeliveryCursorError(InvalidDeliveryCursorError):
    """A WebSocket resume cursor points beyond the current stream head."""


class DeliveryHistoryGapError(RuntimeError):
    """The requested cursor predates the retained delivery window."""

    def __init__(self, *, requested: int, earliest: int, current: int) -> None:
        super().__init__(
            f"delivery history gap: requested={requested}, earliest={earliest}, current={current}"
        )
        self.requested = requested
        self.earliest = earliest
        self.current = current


@dataclass(frozen=True)
class DeliveryRecord:
    """One already-projected item in an authorization-specific stream."""

    seq: int
    delivery_id: str
    payload: dict[str, Any]
    initial_replay: bool


@dataclass
class DeliveryStream:
    """Monotonic stream for one audience projection.

    Streams are separate for spectators, each playable seat, and god/replay.
    Consequently a private event cannot reveal its existence as a gap in the
    public sequence.
    """

    key: str
    mode: str
    seat: int | None
    stream_id: str = field(default_factory=lambda: secrets.token_urlsafe(12))
    cursor: int = 0
    history: deque[DeliveryRecord] = field(default_factory=deque)
    history_gap: bool = False


@dataclass
class RoomClient:
    """A WebSocket plus its isolated outbound writer queue."""

    websocket: WebSocket
    seat: int | None
    mode: str
    stream_key: str
    queue: asyncio.Queue[str]
    loop: asyncio.AbstractEventLoop
    writer_task: asyncio.Task[None] | None = None
    handshake_task: asyncio.Task[Any] | None = None
    close_task: asyncio.Task[None] | None = None


@dataclass
class Room:
    """一个游戏房间。"""

    id: str
    state: GameState
    orchestrator: GameOrchestrator | None = None
    actors: dict[str, AgentActor] = field(default_factory=dict)
    status: str = "waiting"  # waiting / running / ended / incomplete / failed / timeout / cancelled / interrupted
    end_reason: str | None = None
    error: str | None = None
    default_config: ModelConfig | None = None
    seat_configs: dict[int, ModelConfig] = field(default_factory=dict)
    human_seats: set[int] = field(default_factory=set)
    base_seed: int | None = None
    role_seed: int | None = None
    actor_seed: int | None = None
    orchestrator_seed: int | None = None
    admin_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    seat_tokens: dict[int, str] = field(default_factory=dict)
    # Capability provenance.  Plaintext values above are process-local only;
    # persistence stores these salted hashes and versions instead.
    admin_token_hash: str | None = None
    admin_token_version: int = 1
    admin_token_revoked: bool = False
    revoked_admin_token_hashes: list[str] = field(default_factory=list)
    seat_token_hashes: dict[int, str] = field(default_factory=dict)
    seat_token_versions: dict[int, int] = field(default_factory=dict)
    revoked_seat_token_hashes: dict[int, list[str]] = field(default_factory=dict)
    # 连接的客户端:client_id -> 独立有界发送队列
    clients: dict[str, RoomClient] = field(default_factory=dict)
    # 完整事件流(回放用,按时间序)
    event_history: list[dict[str, Any]] = field(default_factory=list)
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    transcript: Transcript | None = None
    run_spec: RunSpec | None = None
    core_run_spec: CoreRunSpec | None = None
    prepared_run: PreparedEnvironmentRun | None = None
    core_result: EnvironmentRunResult | None = None
    core_terminal_committed: bool = False
    evidence_sink_error_type: str | None = None
    trace_seq: int = 0
    delivery_source_seq: int = 0
    delivery_source_history: deque[tuple[int, dict[str, Any], bool]] = field(default_factory=deque)
    delivery_streams: dict[str, DeliveryStream] = field(default_factory=dict)
    # Persisted absolute cursor identity for each authorization-specific stream.
    # Payload history is rebuilt from the bounded source window, so this stays
    # small while preserving reconnect semantics across process restart.
    delivery_stream_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    task: asyncio.Task | None = None
    evidence_limit_error: RoomEvidenceLimitError | None = None
    terminal_at: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    capability_lock: threading.RLock = field(default_factory=threading.RLock)
    # WebSocket delivery is touched by TestClient's portal loop and by game
    # callbacks. A dedicated synchronous lock gives one atomic cutover without
    # binding delivery state to either event loop.
    delivery_lock: threading.RLock = field(default_factory=threading.RLock)


class RoomManager:
    """全局房间管理。"""

    def __init__(
        self,
        router: LLMRouter | None = None,
        *,
        room_timeout: float | None = None,
        max_rooms: int | None = None,
        terminal_room_ttl: float | None = None,
        ws_client_queue_size: int | None = None,
        ws_delivery_history_size: int | None = None,
        max_ws_clients_per_room: int | None = None,
        persistence: RoomPersistence | None = None,
        persistence_path: str | os.PathLike[str] | None = None,
        restore_persisted_rooms: bool = True,
        admission_limiter: AdmissionLimiter | None = None,
        provider_budget_ledger: ProviderBudgetLedger | None = None,
        provider_budget_policy: ProviderBudgetPolicy | None = None,
        cancellation_grace_seconds: float = DEFAULT_ROOM_CANCELLATION_GRACE_SECONDS,
        cleanup_timeout_seconds: float = DEFAULT_ROOM_CLEANUP_TIMEOUT_SECONDS,
        max_evidence_entries: int | None = None,
    ) -> None:
        if persistence is not None and persistence_path is not None:
            raise ValueError("provide persistence or persistence_path, not both")
        if admission_limiter is not None and not isinstance(admission_limiter, AdmissionLimiter):
            raise TypeError("admission_limiter must be an AdmissionLimiter")
        if provider_budget_ledger is not None and not isinstance(provider_budget_ledger, ProviderBudgetLedger):
            raise TypeError("provider_budget_ledger must be a ProviderBudgetLedger")
        if provider_budget_policy is not None and not isinstance(provider_budget_policy, ProviderBudgetPolicy):
            raise TypeError("provider_budget_policy must be a ProviderBudgetPolicy")
        self.rooms: dict[str, Room] = {}
        existing_router_ledger = getattr(router, "budget_ledger", None) if router is not None else None
        if provider_budget_ledger is not None and existing_router_ledger is not None and provider_budget_ledger is not existing_router_ledger:
            raise ValueError("router and manager provider budget ledgers differ")
        self.provider_budget_ledger = (
            provider_budget_ledger
            or existing_router_ledger
            or ProviderBudgetLedger(
                default_policy=provider_budget_policy or PROVIDER_BUDGET_POLICY,
                max_scopes=PROVIDER_BUDGET_MAX_SCOPES,
                max_inflight_reservations=PROVIDER_BUDGET_MAX_INFLIGHT_RESERVATIONS,
                closed_scope_ttl_seconds=PROVIDER_BUDGET_CLOSED_SCOPE_TTL_SECONDS,
            )
        )
        self.provider_budget_policy = (
            provider_budget_policy
            or getattr(router, "budget_policy", None)
            or self.provider_budget_ledger.default_policy
        )
        self.admission_limiter = admission_limiter or AdmissionLimiter(
            rest=REST_RATE_LIMIT_CONFIG,
            ws=WS_RATE_LIMIT_CONFIG,
        )
        if router is None:
            self.router = LLMRouter(
                timeout=LLM_TIMEOUT,
                max_retries=LLM_MAX_RETRIES,
                concurrency=LLM_CONCURRENCY,
                budget_ledger=self.provider_budget_ledger,
                budget_policy=self.provider_budget_policy,
            )
        else:
            self.router = router
            # Real routers supplied by integrations inherit manager governance;
            # fake/test routers may intentionally omit the optional contract.
            if isinstance(router, LLMRouter):
                router.budget_ledger = self.provider_budget_ledger
                router.budget_policy = self.provider_budget_policy
        self.room_timeout = _room_timeout_from_env() if room_timeout is None else room_timeout
        self.max_rooms = MAX_ROOMS if max_rooms is None else int(max_rooms)
        self.terminal_room_ttl = (
            TERMINAL_ROOM_TTL if terminal_room_ttl is None else float(terminal_room_ttl)
        )
        self.ws_client_queue_size = (
            _positive_int_from_env("WEREWOLF_WS_CLIENT_QUEUE_SIZE", DEFAULT_WS_CLIENT_QUEUE_SIZE)
            if ws_client_queue_size is None
            else int(ws_client_queue_size)
        )
        self.ws_delivery_history_size = (
            _positive_int_from_env(
                "WEREWOLF_WS_DELIVERY_HISTORY_SIZE",
                DEFAULT_WS_DELIVERY_HISTORY_SIZE,
            )
            if ws_delivery_history_size is None
            else int(ws_delivery_history_size)
        )
        self.max_ws_clients_per_room = (
            _positive_int_from_env(
                "WEREWOLF_MAX_WS_CLIENTS_PER_ROOM",
                DEFAULT_MAX_WS_CLIENTS_PER_ROOM,
            )
            if max_ws_clients_per_room is None
            else int(max_ws_clients_per_room)
        )
        self.max_evidence_entries = (
            _positive_int_from_env(
                "WEREWOLF_MAX_ROOM_EVIDENCE_ENTRIES",
                DEFAULT_MAX_ROOM_EVIDENCE_ENTRIES,
            )
            if max_evidence_entries is None
            else int(max_evidence_entries)
        )
        self.cancellation_grace_seconds = _bounded_duration(
            cancellation_grace_seconds,
            name="cancellation_grace_seconds",
            minimum=0.0,
            maximum=60.0,
        )
        self.cleanup_timeout_seconds = _bounded_duration(
            cleanup_timeout_seconds,
            name="cleanup_timeout_seconds",
            minimum=0.0,
            maximum=300.0,
            minimum_inclusive=False,
        )
        if self.max_rooms <= 0:
            raise ValueError("max_rooms must be a positive integer")
        if self.terminal_room_ttl < 0 or not math.isfinite(self.terminal_room_ttl):
            raise ValueError("terminal_room_ttl must be a non-negative finite number")
        if self.ws_client_queue_size <= 0:
            raise ValueError("ws_client_queue_size must be a positive integer")
        if self.ws_delivery_history_size <= 0:
            raise ValueError("ws_delivery_history_size must be a positive integer")
        if not 0 < self.max_ws_clients_per_room <= MAX_WS_CLIENTS_PER_ROOM_HARD_LIMIT:
            raise ValueError(
                "max_ws_clients_per_room must be between 1 and "
                f"{MAX_WS_CLIENTS_PER_ROOM_HARD_LIMIT}"
            )
        if not 0 < self.max_evidence_entries <= MAX_ROOM_EVIDENCE_ENTRIES_HARD_LIMIT:
            raise ValueError(
                "max_evidence_entries must be between 1 and "
                f"{MAX_ROOM_EVIDENCE_ENTRIES_HARD_LIMIT}"
            )
        self._closing = False
        self._closed = False
        self._router_closed = False
        self._router_close_failed = False
        self._lifecycle_lock = asyncio.Lock()
        self._cleanup_failures: list[dict[str, Any]] = []
        self._quarantined_tasks: dict[asyncio.Future[Any], dict[str, Any]] = {}
        self._quarantined_tasks_lock = threading.RLock()
        self.persistence: RoomPersistence | None = (
            persistence
            if persistence is not None
            else SQLiteRoomPersistence(persistence_path)
            if persistence_path is not None
            else None
        )
        self._persistence_closed = False
        if self.persistence is not None and restore_persisted_rooms:
            self._restore_persisted_rooms()

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def cleanup_failures(self) -> tuple[dict[str, Any], ...]:
        """Credential-free lifecycle evidence accumulated by this manager."""
        return tuple(dict(item) for item in self._cleanup_failures)

    @property
    def unresolved_cleanup_task_count(self) -> int:
        with self._quarantined_tasks_lock:
            return sum(not task.done() for task in self._quarantined_tasks)

    def readiness(self) -> tuple[bool, dict[str, str]]:
        """Return bounded lifecycle checks without probing an external model."""
        if self._closed:
            manager_status = "closed"
        elif self._closing:
            manager_status = "closing"
        else:
            manager_status = "ready"

        router_has_contract = bool(
            self.router is not None
            and callable(getattr(self.router, "complete_json", None))
            and callable(getattr(self.router, "aclose", None))
        )
        if self._router_closed:
            router_status = "closed"
        elif self._router_close_failed:
            router_status = "unavailable"
        elif not router_has_contract:
            router_status = "unavailable"
        else:
            router_status = "ready"
        checks = {"room_manager": manager_status, "router": router_status}
        return all(status == "ready" for status in checks.values()), checks

    def _ensure_available(self) -> None:
        if self._closing or self._closed:
            raise RoomManagerUnavailableError("room manager is not accepting new work")

    def _provider_scope_id(self, room: Room) -> str:
        return self.provider_budget_ledger.room_scope(room.id)

    def _ensure_provider_scope(self, room: Room) -> None:
        scope_id = self._provider_scope_id(room)
        if self.provider_budget_ledger.snapshot(scope_id) is None:
            self.provider_budget_ledger.register_scope(
                scope_id,
                self.provider_budget_policy,
            )

    def _close_provider_scope(self, room: Room) -> None:
        scope_id = self._provider_scope_id(room)
        if self.provider_budget_ledger.snapshot(scope_id) is None:
            self.provider_budget_ledger.register_scope(
                scope_id,
                self.provider_budget_policy,
            )
        snapshot = self.provider_budget_ledger.snapshot(scope_id)
        if snapshot is not None and not snapshot.closed:
            self.provider_budget_ledger.close_scope(scope_id)

    # ------------------------------------------------------------------
    # Capability credentials
    # ------------------------------------------------------------------
    def _ensure_token_state(self, room: Room) -> None:
        """Materialize missing hashes without re-verifying on every event write.

        Capability verification is intentionally performed at the request
        boundary.  Re-running PBKDF2 for every transcript/event persistence
        write would block the event loop and make a busy room quadratic in the
        number of seats.  A missing hash is the only legacy state we repair;
        an existing hash is treated as immutable provenance.
        """
        if room.admin_token and not room.admin_token_hash:
            room.admin_token_hash = hash_capability(room.admin_token)
            room.admin_token_revoked = False
        for seat, token in list(room.seat_tokens.items()):
            if not token:
                continue
            current_hash = room.seat_token_hashes.get(seat)
            if not current_hash:
                room.seat_token_hashes[seat] = hash_capability(token)
            room.seat_token_versions.setdefault(seat, 1)

    @staticmethod
    def valid_admin_token(room: Room, token: str | None) -> bool:
        with room.capability_lock:
            if not token or room.admin_token_revoked:
                return False
            if room.admin_token_hash:
                return verify_capability(token, room.admin_token_hash)
            return secrets.compare_digest(str(token), room.admin_token)

    @staticmethod
    def valid_seat_token(room: Room, seat: int | None, token: str | None) -> bool:
        with room.capability_lock:
            if seat is None or not token:
                return False
            encoded = room.seat_token_hashes.get(seat)
            if encoded:
                return verify_capability(token, encoded)
            expected = room.seat_tokens.get(seat)
            if not expected:
                return False
            return secrets.compare_digest(str(token), expected)

    def _invalidate_capability_connections(
        self,
        room: Room,
        *,
        admin: bool = False,
        seat: int | None = None,
    ) -> None:
        """Drop live sockets whose capability was rotated or revoked."""
        client_ids: list[str] = []
        with room.delivery_lock:
            for cid, connection in room.clients.items():
                if admin and connection.mode in {"god", "replay"}:
                    client_ids.append(cid)
                elif seat is not None and connection.mode == "play" and connection.seat == seat:
                    client_ids.append(cid)
        # Remove authorization before scheduling socket close so no later
        # broadcast or human action can use the stale connection.
        for cid in client_ids:
            with room.delivery_lock:
                connection = room.clients.pop(cid, None)
            if isinstance(connection, RoomClient):
                self._terminate_client_on_owner_loop(
                    room,
                    connection,
                    code=4403,
                    reason="capability rotated or revoked",
                )

    def rotate_admin_token(self, room: Room) -> str:
        """Revoke the current admin capability and return a new one once."""
        with room.capability_lock:
            return self._rotate_admin_token_locked(room)

    def _rotate_admin_token_locked(self, room: Room) -> str:
        self._ensure_available()
        self._ensure_token_state(room)
        previous = (
            room.admin_token,
            room.admin_token_hash,
            room.admin_token_version,
            room.admin_token_revoked,
            list(room.revoked_admin_token_hashes),
        )
        revoked = list(room.revoked_admin_token_hashes)
        if room.admin_token_hash:
            revoked.append(room.admin_token_hash)
        token = secrets.token_urlsafe(24)
        room.admin_token = token
        room.admin_token_hash = hash_capability(token)
        room.admin_token_version = max(1, int(room.admin_token_version)) + 1
        room.admin_token_revoked = False
        room.revoked_admin_token_hashes = revoked[-32:]
        try:
            self._persist_room(room)
        except Exception:
            (
                room.admin_token,
                room.admin_token_hash,
                room.admin_token_version,
                room.admin_token_revoked,
                room.revoked_admin_token_hashes,
            ) = previous
            raise
        self._invalidate_capability_connections(room, admin=True)
        return token

    def revoke_admin_token(self, room: Room) -> None:
        """Revoke admin access.  Rotation is required to regain access."""
        with room.capability_lock:
            self._revoke_admin_token_locked(room)

    def _revoke_admin_token_locked(self, room: Room) -> None:
        self._ensure_available()
        self._ensure_token_state(room)
        previous = (
            room.admin_token,
            room.admin_token_hash,
            room.admin_token_version,
            room.admin_token_revoked,
            list(room.revoked_admin_token_hashes),
        )
        revoked = list(room.revoked_admin_token_hashes)
        if room.admin_token_hash:
            revoked.append(room.admin_token_hash)
        room.admin_token = ""
        room.admin_token_hash = None
        room.admin_token_version = max(1, int(room.admin_token_version)) + 1
        room.admin_token_revoked = True
        room.revoked_admin_token_hashes = revoked[-32:]
        try:
            self._persist_room(room)
        except Exception:
            (
                room.admin_token,
                room.admin_token_hash,
                room.admin_token_version,
                room.admin_token_revoked,
                room.revoked_admin_token_hashes,
            ) = previous
            raise
        self._invalidate_capability_connections(room, admin=True)

    def rotate_seat_token(self, room: Room, seat: int) -> str:
        """Revoke and replace one human-seat capability."""
        with room.capability_lock:
            return self._rotate_seat_token_locked(room, seat)

    def _rotate_seat_token_locked(self, room: Room, seat: int) -> str:
        self._ensure_available()
        if seat not in room.human_seats:
            raise ValueError(f"座位不存在或不是人类席位: {seat}")
        self._ensure_token_state(room)
        previous = (
            dict(room.seat_tokens),
            dict(room.seat_token_hashes),
            dict(room.seat_token_versions),
            {key: list(value) for key, value in room.revoked_seat_token_hashes.items()},
        )
        old_hash = room.seat_token_hashes.get(seat)
        if old_hash:
            revoked = room.revoked_seat_token_hashes.setdefault(seat, [])
            revoked.append(old_hash)
            room.revoked_seat_token_hashes[seat] = revoked[-32:]
        token = secrets.token_urlsafe(24)
        room.seat_tokens[seat] = token
        room.seat_token_hashes[seat] = hash_capability(token)
        room.seat_token_versions[seat] = max(1, int(room.seat_token_versions.get(seat, 1))) + 1
        try:
            self._persist_room(room)
        except Exception:
            (
                room.seat_tokens,
                room.seat_token_hashes,
                room.seat_token_versions,
                room.revoked_seat_token_hashes,
            ) = previous
            raise
        self._invalidate_capability_connections(room, seat=seat)
        return token

    def revoke_seat_token(self, room: Room, seat: int) -> None:
        """Revoke a seat capability without changing any game state."""
        with room.capability_lock:
            self._revoke_seat_token_locked(room, seat)

    def _revoke_seat_token_locked(self, room: Room, seat: int) -> None:
        self._ensure_available()
        if seat not in room.human_seats:
            raise ValueError(f"座位不存在或不是人类席位: {seat}")
        self._ensure_token_state(room)
        previous = (
            dict(room.seat_tokens),
            dict(room.seat_token_hashes),
            dict(room.seat_token_versions),
            {key: list(value) for key, value in room.revoked_seat_token_hashes.items()},
        )
        old_hash = room.seat_token_hashes.get(seat)
        if old_hash:
            revoked = room.revoked_seat_token_hashes.setdefault(seat, [])
            revoked.append(old_hash)
            room.revoked_seat_token_hashes[seat] = revoked[-32:]
        room.seat_tokens[seat] = ""
        room.seat_token_hashes.pop(seat, None)
        room.seat_token_versions[seat] = max(1, int(room.seat_token_versions.get(seat, 1))) + 1
        try:
            self._persist_room(room)
        except Exception:
            (
                room.seat_tokens,
                room.seat_token_hashes,
                room.seat_token_versions,
                room.revoked_seat_token_hashes,
            ) = previous
            raise
        self._invalidate_capability_connections(room, seat=seat)

    # ------------------------------------------------------------------
    # Optional durable room persistence
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_model_config(config: ModelConfig | None) -> dict[str, Any] | None:
        if config is None:
            return None
        # Keep model/runtime knobs, but deliberately omit api_key.  The
        # configured boolean is provenance only and cannot recreate a secret.
        manifest = ModelConfigManifest.from_config(config)
        data = config.model_dump(exclude={"api_key"})
        data["api_base"] = manifest.api_base
        data["configured"] = bool(config.model and config.api_key)
        # Generic key-based redaction would corrupt a valid JSON Schema whose
        # public property happens to be named ``api_key`` or ``token``. Redact
        # the ordinary config first, then restore the schema-aware safe copies
        # produced by ModelConfigManifest.
        data["response_format"] = None
        safe = redact_sensitive(data)
        if not isinstance(safe, dict):  # defensive: model_dump always returns a dict
            raise PersistenceError("model configuration could not be sanitized")
        safe["response_format"] = manifest.response_format
        safe["reasoning"] = manifest.reasoning
        safe["thinking"] = manifest.thinking
        return safe

    @staticmethod
    def _restore_model_config(data: Any) -> ModelConfig | None:
        if not isinstance(data, dict):
            return None
        allowed = set(ModelConfig.model_fields) - {"api_key"}
        values = {key: value for key, value in data.items() if key in allowed}
        values["api_key"] = ""
        try:
            return ModelConfig(**values)
        except Exception as err:  # noqa: BLE001
            raise PersistenceError("persisted model configuration is invalid") from err

    def _room_record(self, room: Room) -> dict[str, Any]:
        # Capability mutation and durable snapshot are one ordering domain.
        # Otherwise an older concurrent event snapshot could commit after a
        # successful rotation and reactivate the revoked hash on restart.
        with room.capability_lock:
            return self._room_record_locked(room)

    def _evidence_entry_count(self, room: Room) -> int:
        """Return the logical evidence count without double-counting storage."""
        history_count = len(room.event_history) + len(room.decision_trace)
        transcript_count = len(room.transcript.entries) if room.transcript is not None else 0
        state_event_count = len(room.state.events)
        # A normal room has one transcript row per event/decision source row.
        # ``max`` keeps a malformed/integration-created room fail-closed rather
        # than allowing any persisted representation to bypass the configured
        # cap or produce a snapshot that cannot be restored under that cap.
        return max(history_count, transcript_count, state_event_count)

    def _ensure_evidence_capacity(
        self,
        room: Room,
        *,
        additional: int = 1,
        state: GameState | None = None,
        transcript: Transcript | None = None,
    ) -> None:
        if type(additional) is not int or additional < 0:
            raise ValueError("additional evidence count must be a non-negative integer")
        # ``additional`` is about to be appended to both the manager-owned
        # source history and its transcript mirror.  A rules event may already
        # exist in ``state.events`` before its corresponding live event reaches
        # this boundary, so adding the same prospective count to the maximum of
        # all three representations would count that one event twice.  Bound
        # each representation independently, then compare their projected
        # maximum.
        history_count = len(room.event_history) + len(room.decision_trace)
        selected_state = state if state is not None else room.state
        selected_transcript = transcript if transcript is not None else room.transcript
        transcript_count = len(selected_transcript.entries) if selected_transcript is not None else 0
        projected = max(
            history_count + additional,
            transcript_count + additional,
            len(selected_state.events),
        )
        if projected > self.max_evidence_entries:
            raise RoomEvidenceLimitError(
                current=projected,
                limit=self.max_evidence_entries,
            )

    @staticmethod
    def _trip_evidence_limit(room: Room, error: RoomEvidenceLimitError) -> None:
        """Record a fatal evidence overflow even when sinks swallow exceptions."""
        if room.evidence_limit_error is None:
            room.evidence_limit_error = error
        orchestrator = room.orchestrator
        if orchestrator is not None:
            # Game/Agent trace callbacks deliberately treat observability as
            # non-fatal and may swallow this exception.  Aborting the owner at
            # the same boundary prevents more provider work while the wrapper
            # below re-raises the saved error for terminal room handling.
            setattr(orchestrator, "aborted", True)

    @staticmethod
    def _trip_evidence_sink(room: Room, error: BaseException) -> None:
        """Latch a durable-sink failure even if an environment swallows it."""
        if room.evidence_sink_error_type is None:
            room.evidence_sink_error_type = type(error).__name__
        orchestrator = room.orchestrator
        if orchestrator is not None:
            setattr(orchestrator, "aborted", True)

    def _room_record_locked(self, room: Room) -> dict[str, Any]:
        self._ensure_token_state(room)
        try:
            self._ensure_evidence_capacity(room, additional=0)
        except RoomEvidenceLimitError as err:
            self._trip_evidence_limit(room, err)
            raise
        terminal_epoch = None
        if room.status in TERMINAL_ROOM_STATUSES and room.terminal_at is not None:
            # ``terminal_at`` is monotonic and cannot survive a restart.  Store
            # an epoch companion solely for TTL accounting.
            terminal_epoch = time.time() - max(0.0, time.monotonic() - room.terminal_at)
        with room.delivery_lock:
            source_history = [
                {
                    "seq": int(seq),
                    "payload": redact_sensitive(payload),
                    "initial_replay": bool(initial_replay),
                }
                for seq, payload, initial_replay in room.delivery_source_history
            ]
            delivery_source_seq = int(room.delivery_source_seq)
            delivery_streams = {
                key: {
                    "stream_id": stream.stream_id,
                    "cursor": int(stream.cursor),
                    "history_gap": bool(stream.history_gap),
                }
                for key, stream in sorted(room.delivery_streams.items())
            }
        return {
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "room_id": room.id,
            "state": redact_sensitive(room.state.model_dump()),
            "status": room.status,
            "end_reason": redact_sensitive(room.end_reason),
            "error": redact_sensitive(room.error),
            "default_model_config": self._safe_model_config(room.default_config),
            "seat_configs": {
                str(seat): self._safe_model_config(config)
                for seat, config in sorted(room.seat_configs.items())
            },
            "human_seats": sorted(int(seat) for seat in room.human_seats),
            "base_seed": room.base_seed,
            "role_seed": room.role_seed,
            "actor_seed": room.actor_seed,
            "orchestrator_seed": room.orchestrator_seed,
            "capabilities": {
                "admin": {
                    "hash": room.admin_token_hash,
                    "version": int(room.admin_token_version),
                    "revoked": bool(room.admin_token_revoked),
                    "revoked_hashes": list(room.revoked_admin_token_hashes[-32:]),
                },
                "seats": {
                    str(seat): {
                        "hash": room.seat_token_hashes.get(seat),
                        "version": int(room.seat_token_versions.get(seat, 1)),
                        "revoked": not bool(room.seat_token_hashes.get(seat)),
                        "revoked_hashes": list(room.revoked_seat_token_hashes.get(seat, [])[-32:]),
                    }
                    for seat in sorted(room.human_seats)
                },
            },
            "event_history": redact_sensitive(room.event_history),
            "decision_trace": redact_sensitive(room.decision_trace),
            "run_spec": redact_sensitive(room.run_spec.model_dump()) if room.run_spec else None,
            "core_run_spec": (
                redact_sensitive(room.core_run_spec.model_dump(mode="json"))
                if room.core_run_spec
                else None
            ),
            "transcript": redact_sensitive(room.transcript.export()) if room.transcript else None,
            "trace_seq": int(room.trace_seq),
            "delivery_source_seq": delivery_source_seq,
            "delivery_source_history": source_history,
            "delivery_streams": delivery_streams,
            "terminal_at_epoch": terminal_epoch,
        }

    def _persist_room(self, room: Room) -> None:
        if self.persistence is None or self._persistence_closed:
            return
        with room.capability_lock:
            self.persistence.save_record(self._room_record_locked(room))

    @staticmethod
    def _restore_transcript(run_id: str, data: Any) -> Transcript | None:
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") != TRANSCRIPT_SCHEMA_VERSION:
            raise PersistenceError("persisted transcript schema is invalid")
        if data.get("run_id") != run_id:
            raise PersistenceError("persisted transcript run id does not match room")
        entries = data.get("entries")
        if not isinstance(entries, list):
            raise PersistenceError("persisted transcript entries are invalid")
        counts = data.get("counts_by_kind")
        expected_digest = data.get("stable_digest")
        if not isinstance(counts, dict) or any(
            not isinstance(kind, str) or type(count) is not int or count < 0
            for kind, count in counts.items()
        ):
            raise PersistenceError("persisted transcript counts are invalid")
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_digest)
        ):
            raise PersistenceError("persisted transcript stable digest is missing or invalid")
        try:
            validated_entries = [HarnessEvent.model_validate(row) for row in entries]
            actual_counts: dict[str, int] = {}
            for expected_seq, entry in enumerate(validated_entries, start=1):
                if entry.schema_version != TRANSCRIPT_SCHEMA_VERSION:
                    raise PersistenceError("persisted transcript entry schema is invalid")
                if entry.run_id != run_id:
                    raise PersistenceError("persisted transcript entry run id does not match")
                if entry.seq != expected_seq:
                    raise PersistenceError("persisted transcript sequence is not contiguous")
                if entry.payload_hash != payload_digest(entry.payload):
                    raise PersistenceError("persisted transcript payload hash does not match")
                actual_counts[entry.kind] = actual_counts.get(entry.kind, 0) + 1
            if counts != actual_counts:
                raise PersistenceError("persisted transcript counts do not match entries")
            transcript = Transcript.model_validate({
                "schema_version": data.get("schema_version"),
                "run_id": run_id,
                "metadata": data.get("metadata") or {},
                "entries": validated_entries,
            })
        except PersistenceError:
            raise
        except Exception as err:  # noqa: BLE001
            raise PersistenceError("persisted transcript is invalid") from err
        if transcript.stable_digest() != expected_digest:
            raise PersistenceError("persisted transcript digest does not match")
        return transcript

    @staticmethod
    def _validate_transcript_sources(
        room: Room,
        *,
        trace_seq_present: bool,
    ) -> None:
        """Cross-check source order and the unified persisted trace timeline."""
        if room.transcript is None:
            if room.event_history or room.decision_trace:
                raise PersistenceError("persisted histories are missing their transcript")
            if trace_seq_present and room.trace_seq != 0:
                raise PersistenceError("persisted trace cursor does not match histories")
            if not trace_seq_present and room.core_run_spec is not None:
                raise PersistenceError("persisted Core room trace cursor is missing")
            return
        expected_source_idx = {"event": 0, "decision": 0}
        sources = {
            "event": room.event_history,
            "decision": room.decision_trace,
        }
        previous_trace_seq = 0
        legacy_rows_without_trace_seq = False
        for entry in room.transcript.entries:
            if entry.kind not in sources:
                continue
            if entry.source_idx is None:
                raise PersistenceError("persisted transcript source index is missing")
            source_rows = sources[entry.kind]
            if entry.source_idx < 0 or entry.source_idx >= len(source_rows):
                raise PersistenceError("persisted transcript source index is out of range")
            if entry.source_idx != expected_source_idx[entry.kind]:
                raise PersistenceError(
                    f"persisted {entry.kind} source order does not match history"
                )
            source = redact_sensitive(source_rows[entry.source_idx])
            if not isinstance(source, dict) or payload_digest(source) != entry.payload_hash:
                raise PersistenceError("persisted transcript source payload does not match")
            raw_trace_seq = source.get("_trace_seq")
            if raw_trace_seq is None and not trace_seq_present and room.core_run_spec is None:
                legacy_rows_without_trace_seq = True
            elif (
                isinstance(raw_trace_seq, bool)
                or not isinstance(raw_trace_seq, int)
                or raw_trace_seq <= previous_trace_seq
            ):
                raise PersistenceError("persisted source trace sequence is invalid")
            else:
                if legacy_rows_without_trace_seq:
                    raise PersistenceError("persisted legacy trace sequence is inconsistent")
                previous_trace_seq = raw_trace_seq
            expected_source_idx[entry.kind] += 1
        for kind, source_rows in sources.items():
            if expected_source_idx[kind] != len(source_rows):
                raise PersistenceError(f"persisted {kind} history is not fully represented")
        if legacy_rows_without_trace_seq:
            if any(
                row.get("_trace_seq") is not None
                for rows in sources.values()
                for row in rows
            ):
                raise PersistenceError("persisted legacy trace sequence is inconsistent")
            room.trace_seq = 0
            return
        if not trace_seq_present:
            if room.core_run_spec is not None:
                raise PersistenceError("persisted Core room trace cursor is missing")
            # Explicit migration for a pre-cursor legacy record whose source
            # rows already carry a complete, ordered trace timeline.
            room.trace_seq = previous_trace_seq
            return
        if room.trace_seq != previous_trace_seq:
            raise PersistenceError("persisted trace cursor does not match histories")

    @staticmethod
    def _validate_legacy_core_binding(
        run_spec: RunSpec,
        core_run_spec: CoreRunSpec,
    ) -> None:
        """Verify that an interactive Core spec is the same migrated legacy run."""
        try:
            migrated = legacy_werewolf_run_to_core(run_spec)
        except Exception as err:  # noqa: BLE001
            raise PersistenceError("persisted RunSpec cannot migrate to CoreRunSpec") from err

        if core_run_spec.metadata.get("legacy_spec_hash") != run_spec.spec_hash:
            raise PersistenceError("persisted CoreRunSpec legacy_spec_hash does not match RunSpec")
        if core_run_spec.environment != migrated.environment:
            raise PersistenceError("persisted CoreRunSpec environment does not match RunSpec")
        for key, expected in migrated.environment_config.items():
            if key not in core_run_spec.environment_config or (
                core_run_spec.environment_config[key] != expected
            ):
                raise PersistenceError(
                    f"persisted CoreRunSpec environment_config.{key} does not match RunSpec"
                )
        allowed_core_environment_fields = {"decision_timeouts", "phase_deadlines"}
        unexpected_environment_fields = (
            set(core_run_spec.environment_config)
            - set(migrated.environment_config)
            - allowed_core_environment_fields
        )
        if unexpected_environment_fields:
            raise PersistenceError("persisted CoreRunSpec has unsupported environment fields")
        if core_run_spec.seeds != migrated.seeds:
            raise PersistenceError("persisted CoreRunSpec seeds do not match RunSpec")
        if core_run_spec.actors != migrated.actors:
            raise PersistenceError("persisted CoreRunSpec actors do not match RunSpec")
        if (
            core_run_spec.execution.run_timeout_seconds
            != migrated.execution.run_timeout_seconds
            or core_run_spec.execution.decision_timeout_seconds
            != migrated.execution.decision_timeout_seconds
        ):
            raise PersistenceError("persisted CoreRunSpec execution does not match RunSpec")
        if core_run_spec.metadata != migrated.metadata:
            raise PersistenceError("persisted CoreRunSpec metadata does not match RunSpec")

    @staticmethod
    def _valid_spec_hash(value: Any) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _validate_restored_run_identity(room: Room) -> None:
        """Reject replay provenance that names more than one executed run."""
        if room.state.id != room.id:
            raise PersistenceError("persisted game state id does not match room")
        if room.run_spec is not None and room.run_spec.run_id != room.id:
            raise PersistenceError("persisted RunSpec run id does not match room")
        if room.core_run_spec is not None and room.core_run_spec.run_id != room.id:
            raise PersistenceError("persisted CoreRunSpec run id does not match room")
        if room.transcript is not None and room.transcript.run_id != room.id:
            raise PersistenceError("persisted transcript run id does not match room")

        if room.status == "waiting":
            if (
                room.run_spec is not None
                or room.core_run_spec is not None
                or room.transcript is not None
                or room.event_history
                or room.decision_trace
            ):
                raise PersistenceError("persisted waiting room contains execution evidence")
            return

        if room.core_run_spec is not None:
            if room.run_spec is None:
                raise PersistenceError("persisted CoreRunSpec has no legacy RunSpec")
            RoomManager._validate_legacy_core_binding(room.run_spec, room.core_run_spec)
            if room.transcript is None:
                raise PersistenceError("persisted CoreRunSpec has no transcript")

        if room.transcript is None:
            return
        expected_spec_hash = room.transcript.metadata.get("run_spec_hash")
        canonical_spec = room.core_run_spec or room.run_spec
        if room.core_run_spec is not None and not RoomManager._valid_spec_hash(
            expected_spec_hash
        ):
            raise PersistenceError("persisted Core transcript run_spec_hash is missing or invalid")
        if expected_spec_hash is not None and not RoomManager._valid_spec_hash(
            expected_spec_hash
        ):
            raise PersistenceError("persisted transcript run_spec_hash is invalid")
        if expected_spec_hash is not None and canonical_spec is None:
            raise PersistenceError("persisted transcript run_spec_hash has no RunSpec")
        if expected_spec_hash is not None and expected_spec_hash != canonical_spec.spec_hash:
            raise PersistenceError(
                "persisted transcript run_spec_hash does not match RunSpec"
            )
        legacy_hash = room.transcript.metadata.get("legacy_run_spec_hash")
        if room.core_run_spec is not None and not RoomManager._valid_spec_hash(legacy_hash):
            raise PersistenceError(
                "persisted Core transcript legacy_run_spec_hash is missing or invalid"
            )
        if legacy_hash is not None:
            if not RoomManager._valid_spec_hash(legacy_hash):
                raise PersistenceError("persisted transcript legacy_run_spec_hash is invalid")
            if room.run_spec is None or legacy_hash != room.run_spec.spec_hash:
                raise PersistenceError(
                    "persisted transcript legacy_run_spec_hash does not match RunSpec"
                )

    def _restore_capabilities(self, room: Room, data: Any) -> None:
        caps = data if isinstance(data, dict) else {}
        admin = caps.get("admin") if isinstance(caps.get("admin"), dict) else {}
        old_hash = admin.get("hash")
        old_version = int(admin.get("version") or 1)
        old_revoked = bool(admin.get("revoked"))
        room.revoked_admin_token_hashes = [
            str(value) for value in (admin.get("revoked_hashes") or []) if isinstance(value, str)
        ][-32:]
        room.admin_token = ""
        room.admin_token_version = max(1, old_version)
        room.admin_token_hash = (
            str(old_hash)
            if isinstance(old_hash, str) and old_hash and not old_revoked
            else None
        )
        room.admin_token_revoked = bool(old_revoked or not room.admin_token_hash)

        seat_rows = caps.get("seats") if isinstance(caps.get("seats"), dict) else {}
        for seat in sorted(room.human_seats):
            row = seat_rows.get(str(seat)) if isinstance(seat_rows, dict) else None
            row = row if isinstance(row, dict) else {}
            old_hash = row.get("hash")
            old_version = int(row.get("version") or 1)
            revoked = [
                str(value) for value in (row.get("revoked_hashes") or []) if isinstance(value, str)
            ][-32:]
            room.revoked_seat_token_hashes[seat] = revoked[-32:]
            room.seat_token_versions[seat] = max(1, old_version)
            room.seat_tokens[seat] = ""
            if bool(row.get("revoked")) or not isinstance(old_hash, str) or not old_hash:
                room.seat_token_hashes.pop(seat, None)
            else:
                room.seat_token_hashes[seat] = old_hash

    def _room_from_record(self, record: dict[str, Any]) -> Room:
        raw_state = record.get("state")
        if not isinstance(raw_state, dict):
            raise PersistenceError("persisted game state is invalid")
        raw_state_events = raw_state.get("events") or []
        if not isinstance(raw_state_events, list):
            raise PersistenceError("persisted game events are invalid")
        if len(raw_state_events) > self.max_evidence_entries:
            raise PersistenceError("persisted game events exceed evidence capacity")
        try:
            state = GameState.model_validate(raw_state)
        except Exception as err:  # noqa: BLE001
            raise PersistenceError("persisted game state is invalid") from err
        room_id = str(record.get("room_id") or "")
        status = str(record.get("status") or "waiting")
        if status not in {"waiting", "running", *TERMINAL_ROOM_STATUSES}:
            raise PersistenceError("persisted room status is invalid")
        if status == "waiting" and state.id != room_id:
            # Rooms persisted before the interactive run-id boundary used an
            # unrelated generated GameState id. Waiting rooms have not begun
            # execution, so they can be migrated before their first request.
            state = state.model_copy(update={"id": room_id}, deep=True)
        room = Room(
            id=room_id,
            state=state,
            status=status,
            end_reason=record.get("end_reason"),
            error=record.get("error"),
            default_config=self._restore_model_config(record.get("default_model_config")),
            seat_configs={
                int(seat): config
                for seat, raw in (record.get("seat_configs") or {}).items()
                if (config := self._restore_model_config(raw)) is not None
            },
            human_seats={int(seat) for seat in (record.get("human_seats") or [])},
            base_seed=record.get("base_seed"),
            role_seed=record.get("role_seed"),
            actor_seed=record.get("actor_seed"),
            orchestrator_seed=record.get("orchestrator_seed"),
        )
        try:
            raw_event_history = record.get("event_history") or []
            raw_decision_trace = record.get("decision_trace") or []
            if (
                not isinstance(raw_event_history, list)
                or not isinstance(raw_decision_trace, list)
                or any(not isinstance(item, dict) for item in raw_event_history)
                or any(not isinstance(item, dict) for item in raw_decision_trace)
            ):
                raise PersistenceError("persisted room evidence histories are invalid")
            if len(raw_event_history) + len(raw_decision_trace) > self.max_evidence_entries:
                raise PersistenceError("persisted room evidence exceeds capacity")
            raw_transcript = record.get("transcript")
            if isinstance(raw_transcript, dict):
                raw_entries = raw_transcript.get("entries") or []
                if not isinstance(raw_entries, list):
                    raise PersistenceError("persisted transcript entries are invalid")
                if len(raw_entries) > self.max_evidence_entries:
                    raise PersistenceError("persisted transcript exceeds evidence capacity")
            room.event_history = list(raw_event_history)
            room.decision_trace = list(raw_decision_trace)
            trace_seq_present = "trace_seq" in record
            raw_trace_seq = record.get("trace_seq")
            if trace_seq_present and (
                isinstance(raw_trace_seq, bool)
                or not isinstance(raw_trace_seq, int)
                or raw_trace_seq < 0
            ):
                raise PersistenceError("persisted trace cursor is invalid")
            room.trace_seq = raw_trace_seq if trace_seq_present else 0
            room.delivery_source_seq = int(record.get("delivery_source_seq") or 0)
            if room.delivery_source_seq < 0:
                raise PersistenceError("persisted delivery source cursor is invalid")
            source_rows = record.get("delivery_source_history") or []
            if not isinstance(source_rows, list):
                raise PersistenceError("persisted delivery source history is invalid")
            restored_sources: list[tuple[int, dict[str, Any], bool]] = []
            previous_source_seq = 0
            for row in source_rows:
                if not isinstance(row, dict) or set(row) != {
                    "seq",
                    "payload",
                    "initial_replay",
                }:
                    raise PersistenceError("persisted delivery source row is invalid")
                source_seq = row.get("seq")
                payload = row.get("payload")
                initial_replay = row.get("initial_replay")
                if (
                    isinstance(source_seq, bool)
                    or not isinstance(source_seq, int)
                    or source_seq <= previous_source_seq
                    or not isinstance(payload, dict)
                    or not isinstance(initial_replay, bool)
                ):
                    raise PersistenceError("persisted delivery source row is invalid")
                restored_sources.append((source_seq, dict(payload), initial_replay))
                previous_source_seq = source_seq
            if restored_sources and restored_sources[-1][0] != room.delivery_source_seq:
                raise PersistenceError("persisted delivery source cursor does not match history")
            if not restored_sources and room.delivery_source_seq != 0:
                raise PersistenceError("persisted delivery source history is missing")
            room.delivery_source_history = deque(restored_sources)
            raw_delivery_streams = record.get("delivery_streams")
            if raw_delivery_streams is not None and not isinstance(raw_delivery_streams, dict):
                raise PersistenceError("persisted delivery stream metadata is invalid")
            room.delivery_stream_metadata = {
                str(key): dict(value)
                for key, value in (raw_delivery_streams or {}).items()
                if isinstance(value, dict)
            }
            if raw_delivery_streams and len(room.delivery_stream_metadata) != len(raw_delivery_streams):
                raise PersistenceError("persisted delivery stream metadata is invalid")
            room.run_spec = (
                RunSpec.model_validate(record["run_spec"])
                if isinstance(record.get("run_spec"), dict)
                else None
            )
            room.core_run_spec = (
                CoreRunSpec.model_validate(record["core_run_spec"])
                if isinstance(record.get("core_run_spec"), dict)
                else None
            )
            room.transcript = self._restore_transcript(room_id, record.get("transcript"))
            self._validate_restored_run_identity(room)
            self._validate_transcript_sources(
                room,
                trace_seq_present=trace_seq_present,
            )
        except PersistenceError:
            raise
        except Exception as err:  # noqa: BLE001
            raise PersistenceError("persisted room evidence is invalid") from err

        self._restore_capabilities(room, record.get("capabilities"))
        terminal_epoch = record.get("terminal_at_epoch")
        if status in TERMINAL_ROOM_STATUSES:
            try:
                elapsed = max(0.0, time.time() - float(terminal_epoch)) if terminal_epoch else 0.0
            except (TypeError, ValueError):
                elapsed = 0.0
            room.terminal_at = time.monotonic() - elapsed
        self._initialize_delivery_streams(room)
        if status == "running":
            self._ensure_evidence_capacity(room)
            room.status = "interrupted"
            room.end_reason = "process_restart"
            room.error = "room interrupted during process restart"
            room.terminal_at = time.monotonic()
            restart_payload = {
                "type": "room_status",
                "status": "interrupted",
                "reason": "process_restart",
                "error": room.error,
                "_trace_seq": self._next_trace_seq(room),
                "_ts": time.monotonic(),
            }
            room.event_history.append(restart_payload)
            self._append_transcript(
                room,
                "event",
                restart_payload,
                source_idx=len(room.event_history) - 1,
            )
        return room

    def _restore_persisted_rooms(self) -> None:
        assert self.persistence is not None
        records = self.persistence.load_records()
        for record in records:
            room = self._room_from_record(record)
            if room.id in self.rooms:
                raise PersistenceError("duplicate persisted room id")
            self._ensure_provider_scope(room)
            if room.status in TERMINAL_ROOM_STATUSES:
                self._close_provider_scope(room)
            self.rooms[room.id] = room
            # Persist generated post-restart capabilities and an interruption
            # marker immediately, before accepting any new requests.
            self._persist_room(room)

    def cleanup_expired_rooms(self, *, now: float | None = None) -> list[str]:
        """Remove expired terminal rooms that no task or client still owns."""
        if self.terminal_room_ttl <= 0:
            return []
        current = time.monotonic() if now is None else float(now)
        removed: list[str] = []
        for room_id, room in list(self.rooms.items()):
            if room.status not in TERMINAL_ROOM_STATUSES:
                continue
            if room.terminal_at is None:
                room.terminal_at = current
                continue
            if current - room.terminal_at < self.terminal_room_ttl:
                continue
            if (room.task is not None and not room.task.done()) or room.clients:
                continue
            if self.persistence is not None and not self._persistence_closed:
                self.persistence.delete_room(room_id)
            self._close_provider_scope(room)
            self.rooms.pop(room_id, None)
            removed.append(room_id)
        return removed

    def delete_room(self, room_id: str) -> Room | None:
        """Explicitly remove an idle room; running work is never evicted."""
        self._ensure_available()
        room = self.rooms.get(room_id)
        if room is None:
            return None
        if room.status == "running" or (room.task is not None and not room.task.done()):
            raise RoomInUseError("running room cannot be removed")
        if room.clients:
            raise RoomInUseError("room with active clients cannot be removed")
        if self.persistence is not None and not self._persistence_closed:
            self.persistence.delete_room(room_id)
        self._close_provider_scope(room)
        self.rooms.pop(room_id, None)
        return room

    # ------------------------------------------------------------------
    # 创建/查询
    # ------------------------------------------------------------------
    def create_room(
        self,
        *,
        player_names: list[str],
        default_model_config: ModelConfig | dict[str, Any] | None = None,
        roles: dict[str, str] | None = None,
        human_seats: set[int] | None = None,
        experiment_seed: int | None = None,
    ) -> Room:
        import uuid

        self._ensure_available()
        self.cleanup_expired_rooms()
        if len(self.rooms) >= self.max_rooms:
            raise RoomCapacityError(f"room capacity reached ({self.max_rooms})")
        room_id = uuid.uuid4().hex[:12]
        state = new_game(player_names, game_id=room_id)
        if isinstance(default_model_config, dict):
            default_model_config = ModelConfig(**default_model_config)
        elif isinstance(default_model_config, ModelConfig):
            default_model_config = default_model_config.model_copy(deep=True)
        base_seed = _room_base_seed(experiment_seed)
        room = Room(
            id=room_id,
            state=state,
            default_config=default_model_config,
            human_seats=human_seats or set(),
            base_seed=base_seed,
            role_seed=base_seed,
            actor_seed=base_seed + 100_000,
            orchestrator_seed=base_seed + 200_000,
        )
        room.seat_tokens = {
            seat: secrets.token_urlsafe(24)
            for seat in sorted(room.human_seats)
        }
        self._ensure_token_state(room)
        self._initialize_delivery_streams(room)
        self._ensure_provider_scope(room)
        self.rooms[room_id] = room
        try:
            self._persist_room(room)
        except Exception:
            # Do not expose a room that cannot be recovered durably when the
            # caller explicitly enabled persistence.
            self.rooms.pop(room_id, None)
            self._close_provider_scope(room)
            raise
        return room

    def get_room(self, room_id: str) -> Room | None:
        self.cleanup_expired_rooms()
        return self.rooms.get(room_id)

    def set_seat_model_config(self, room: Room, seat: int, config: ModelConfig | dict) -> None:
        if room.status != "waiting":
            raise ValueError("只能在 waiting 阶段修改座位配置")
        valid_seats = {player.seat for player in room.state.players}
        if seat not in valid_seats:
            raise ValueError(f"座位不存在: {seat}")
        if seat in room.human_seats:
            raise ValueError(f"人类席位不能设置模型 override: {seat}")
        raw_cfg = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        cfg = raw_cfg.model_copy(deep=True)
        had_previous = seat in room.seat_configs
        previous = room.seat_configs.get(seat)
        room.seat_configs[seat] = cfg
        try:
            self._persist_room(room)
        except Exception:
            if had_previous and previous is not None:
                room.seat_configs[seat] = previous
            else:
                room.seat_configs.pop(seat, None)
            raise

    # ------------------------------------------------------------------
    # 启动游戏
    # ------------------------------------------------------------------
    async def start_game(self, room: Room) -> None:
        """发牌 + 构建 actors + 启动编排器协程。"""
        async with self._lifecycle_lock:
            self._ensure_available()
            await self._start_game_locked(room)

    async def _start_game_locked(self, room: Room) -> None:
        """Start a room while holding the manager lifecycle lock."""
        async with room.lock:
            if room.status != "waiting":
                raise ValueError(f"房间状态 {room.status},无法开始")
            if room.state.id != room.id:
                raise ValueError("游戏 state.id 与 room.id 不一致,拒绝开始")
            waiting_state = room.state.model_copy(deep=True)
            waiting_default_config = (
                room.default_config.model_copy(deep=True)
                if room.default_config is not None
                else None
            )
            waiting_seat_configs = {
                seat: config.model_copy(deep=True)
                for seat, config in room.seat_configs.items()
            }
            waiting_run_spec = room.run_spec
            waiting_core_run_spec = room.core_run_spec
            waiting_prepared_run = room.prepared_run
            waiting_core_result = room.core_result
            waiting_core_terminal_committed = room.core_terminal_committed
            waiting_evidence_sink_error_type = room.evidence_sink_error_type
            waiting_transcript = room.transcript
            waiting_actors = room.actors
            waiting_orchestrator = room.orchestrator
            waiting_task = room.task
            waiting_end_reason = room.end_reason
            waiting_error = room.error
            waiting_terminal_at = room.terminal_at
            # Capture one detached configuration snapshot before provenance or
            # actors are built. ModelConfig is mutable, and caller-held room
            # configuration objects must not change execution after hashing.
            base_cfg = (room.default_config or ModelConfig()).model_copy(deep=True)
            seat_configs = {
                seat: config.model_copy(deep=True)
                for seat, config in room.seat_configs.items()
            }
            self._validate_ai_model_configs(room, base_cfg, seat_configs)
            deck = default_role_deck(len(room.state.players))
            # Build the complete live graph against a detached state. A binding,
            # manifest, constructor, or provenance failure must leave the
            # waiting room undealt and retryable instead of publishing a mixed
            # SETUP/status + dealt-role/runtime state.
            staged_state = room.state.model_copy(deep=True)
            RulesEngine.deal_roles(staged_state, deck=deck, seed=room.role_seed)
            run_spec = self._build_run_spec(room, base_cfg, seat_configs, deck)
            core_run_spec = self._build_core_run_spec(run_spec)
            transcript = Transcript(
                run_id=core_run_spec.run_id,
                metadata={
                    "room_id": room.id,
                    "status": room.status,
                    "core_run_spec_version": core_run_spec.schema_version,
                    "run_spec_hash": core_run_spec.spec_hash,
                    "legacy_run_spec_hash": run_spec.spec_hash,
                    "environment_id": core_run_spec.environment.id,
                    "environment_version": core_run_spec.environment.version,
                    "role_seed": room.role_seed,
                    "actor_seed": room.actor_seed,
                    "orchestrator_seed": room.orchestrator_seed,
                },
            )
            actors = build_actors(
                staged_state,
                model_config=base_cfg,
                router=self.router,
                seat_configs=seat_configs,
                human_seats=room.human_seats,
                rng=random.Random(room.actor_seed),
                budget_scope=self.provider_budget_ledger.room_scope(room.id),
            )
            actor_spec = core_run_spec.actors
            registry = self._resolve_live_actors(staged_state, actors, actor_spec)
            evidence = EnvironmentRunEvidence(
                transcript=transcript,
                event_sink=self._make_event_broadcaster(room),
                trace_sink=self._make_trace_recorder(room),
                harness_sink=self._make_harness_recorder(room),
            )
            plugin = WerewolfEnvironmentPlugin(room_state=staged_state)
            contract = plugin.decision_contract
            decision_runtime = DecisionRuntime(
                on_trace=evidence.emit_trace,
                envelope_type=contract.envelope_type,
                validate_envelope=contract.validate_envelope,
                default_timeout_seconds=core_run_spec.execution.decision_timeout_seconds,
                cancellation_grace_seconds=(
                    core_run_spec.execution.cancellation_grace_seconds
                ),
                expected_run_id=core_run_spec.run_id,
            )
            session = None
            try:
                context = EnvironmentRunContext(
                    run_id=core_run_spec.run_id,
                    config=plugin.resolve_config(
                        core_run_spec.environment_config,
                        core_run_spec.seeds,
                    ),
                    seeds=dict(core_run_spec.seeds),
                    actor_spec=actor_spec,
                    decision_runtime=decision_runtime,
                    emit_event=evidence.emit_event,
                    emit_trace=evidence.emit_trace,
                    resolve_agent=registry.resolve,
                    metadata=dict(core_run_spec.metadata),
                )
                session = await plugin.create_session(context)
                prepared_run = PreparedEnvironmentRun(
                    descriptor=plugin.descriptor,
                    session=session,
                    decision_runtime=decision_runtime,
                    evidence=evidence,
                    agent_registry=registry,
                    task_quarantine_sink=self._make_core_task_quarantine_sink(room),
                )
            except BaseException:
                await self._close_unclaimed_core_resources(
                    room_id=room.id,
                    session=session,
                    decision_runtime=decision_runtime,
                )
                raise
            orchestrator = session.orchestrator
            resolved_actor_ids = sorted(registry.snapshot())
            try:
                transcript.append("harness", {
                    "type": "run_started",
                    "environment_id": plugin.descriptor.id,
                    "environment_version": plugin.descriptor.version,
                })
                transcript.append("harness", {
                    "type": "agent_bindings_finalized",
                    "actor_count": len(resolved_actor_ids),
                    "actor_ids": resolved_actor_ids,
                })
                transcript.metadata.update({
                    "resolved_actor_count": len(resolved_actor_ids),
                    "resolved_actor_ids": resolved_actor_ids,
                })
                # The next lifecycle operation appends the running-status event
                # to manager history and this local transcript. Validate against
                # the dealt staged state rather than the waiting room state.
                self._ensure_evidence_capacity(
                    room,
                    additional=1,
                    state=staged_state,
                    transcript=transcript,
                )
            except BaseException:
                await self._close_unclaimed_core_resources(
                    room_id=room.id,
                    session=prepared_run.session,
                    decision_runtime=prepared_run.decision_runtime,
                )
                raise

            # All fallible construction and provenance checks above succeeded.
            # Publish the staged graph as one room lifecycle transition.
            room.state = staged_state
            room.default_config = (
                base_cfg.model_copy(deep=True)
                if room.default_config is not None
                else None
            )
            room.seat_configs = {
                seat: config.model_copy(deep=True)
                for seat, config in seat_configs.items()
            }
            room.run_spec = run_spec
            room.core_run_spec = core_run_spec
            room.prepared_run = prepared_run
            room.core_result = None
            room.core_terminal_committed = False
            room.evidence_sink_error_type = None
            room.transcript = transcript
            room.actors = actors
            room.orchestrator = orchestrator
            room.status = "running"
            room.end_reason = None
            room.error = None
            room.terminal_at = None
            try:
                # Persist the canonical graph before any running notification
                # can be delivered or any model task can start.
                self._persist_room(room)
            except BaseException:
                room.state = waiting_state
                room.default_config = waiting_default_config
                room.seat_configs = waiting_seat_configs
                room.run_spec = waiting_run_spec
                room.core_run_spec = waiting_core_run_spec
                room.prepared_run = waiting_prepared_run
                room.core_result = waiting_core_result
                room.core_terminal_committed = waiting_core_terminal_committed
                room.evidence_sink_error_type = waiting_evidence_sink_error_type
                room.transcript = waiting_transcript
                room.actors = waiting_actors
                room.orchestrator = waiting_orchestrator
                room.task = waiting_task
                room.status = "waiting"
                room.end_reason = waiting_end_reason
                room.error = waiting_error
                room.terminal_at = waiting_terminal_at
                await self._close_unclaimed_core_resources(
                    room_id=room.id,
                    session=prepared_run.session,
                    decision_runtime=prepared_run.decision_runtime,
                )
                raise
            try:
                await self._broadcast_room_status(room)
            except BaseException as err:
                # At this point a durable running snapshot exists and a delivery
                # projection may already have been queued. Never roll it back to
                # a contradictory waiting room; terminate it before any model
                # task is started and make cleanup best-effort.
                room.status = "failed"
                room.end_reason = "startup_failed"
                room.error = f"{type(err).__name__} during room startup"
                room.terminal_at = time.monotonic()
                room.task = None
                try:
                    # Do not recurse through the failed delivery path. Commit a
                    # local terminal row when capacity still permits so replay
                    # evidence cannot end at a contradictory running/binding
                    # record while the authoritative Room is already failed.
                    self._store_room_event(room, {
                        "type": "room_status",
                        "status": room.status,
                        "reason": room.end_reason,
                        "error": room.error,
                    })
                except Exception as evidence_err:  # noqa: BLE001 - retain original failure
                    logger.error(
                        "failed to append startup failure evidence "
                        "(room_id=%s error_type=%s)",
                        room.id,
                        type(evidence_err).__name__,
                    )
                try:
                    self._persist_room(room)
                except Exception:
                    logger.exception("failed to persist startup failure room_id=%s", room.id)
                try:
                    await self._close_unclaimed_core_resources(
                        room_id=room.id,
                        session=prepared_run.session,
                        decision_runtime=prepared_run.decision_runtime,
                    )
                    room.prepared_run = None
                    self._close_provider_scope(room)
                except Exception as cleanup_err:  # noqa: BLE001
                    self._record_cleanup_failure({
                        "stage": "provider_scope_close",
                        "error_type": type(cleanup_err).__name__,
                        "timeout": False,
                        "pending_task_count": 0,
                        "fatal": True,
                    }, room_id=room.id)
                raise
            # 启动游戏循环(后台任务)
            room.task = asyncio.create_task(self._run_room(room))

    async def _close_unclaimed_core_resources(
        self,
        *,
        room_id: str,
        session: Any | None,
        decision_runtime: DecisionRuntime | None,
    ) -> list[dict[str, Any]]:
        """Best-effort cleanup before a PreparedEnvironmentRun is claimed."""
        failures: list[dict[str, Any]] = []
        callbacks = (
            ("session_close", getattr(session, "aclose", None)),
            (
                "decision_runtime_close",
                getattr(decision_runtime, "aclose", None),
            ),
        )
        for stage, callback in callbacks:
            if not callable(callback):
                continue
            try:
                awaitable = callback()
            except Exception as err:  # noqa: BLE001 - normalized cleanup evidence
                failures.append(self._record_cleanup_failure({
                    "stage": stage,
                    "error_type": type(err).__name__,
                    "timeout": False,
                    "pending_task_count": 0,
                    "fatal": True,
                }, room_id=room_id))
                continue
            try:
                failure, _error = await self._bounded_cleanup_awaitable(
                    awaitable,
                    stage=stage,
                    room_id=room_id,
                )
            except asyncio.CancelledError as err:
                failures.append(self._record_cleanup_failure({
                    "stage": stage,
                    "error_type": "CleanupCancelled",
                    "timeout": False,
                    "pending_task_count": int(
                        getattr(err, "cleanup_pending_task_count", 0)
                    ),
                    "fatal": True,
                }, room_id=room_id))
                continue
            if failure is not None:
                failures.append(
                    self._record_cleanup_failure(failure, room_id=room_id)
                )
        return failures

    def _validate_ai_model_configs(
        self,
        room: Room,
        base_cfg: ModelConfig,
        seat_configs: dict[int, ModelConfig],
    ) -> None:
        """Fail fast before starting a real AI room without callable model config."""
        errors: list[str] = []
        for player in room.state.players:
            if player.seat in room.human_seats:
                continue
            cfg = base_cfg.merge(seat_configs.get(player.seat))
            missing: list[str] = []
            if cfg.provider not in STANDARD_LLM_PROTOCOLS:
                missing.append("provider")
            if not (cfg.model or "").strip():
                missing.append("model")
            if not (cfg.api_key or "").strip():
                missing.append("api_key")
            if missing:
                errors.append(f"{player.seat}号缺少 {','.join(missing)}")
        if errors:
            raise ValueError("AI 座位模型配置不完整,拒绝开始真实对局: " + "; ".join(errors))

    def _build_run_spec(
        self,
        room: Room,
        base_cfg: ModelConfig,
        seat_configs: dict[int, ModelConfig],
        deck: list[Role],
    ) -> RunSpec:
        seat_models: dict[int, ModelConfigManifest] = {}
        for seat, override in seat_configs.items():
            seat_models[seat] = ModelConfigManifest.from_config(base_cfg.merge(override))
        return RunSpec(
            run_id=room.id,
            player_names=[player.name for player in room.state.players],
            role_deck=[role.value for role in deck],
            turn_policy=DEFAULT_TURN_POLICY,
            role_seed=room.role_seed,
            actor_seed=room.actor_seed,
            orchestrator_seed=room.orchestrator_seed,
            human_seats=sorted(room.human_seats),
            max_speak_rounds=6,
            run_timeout_seconds=self.room_timeout if self.room_timeout and self.room_timeout > 0 else None,
            decision_timeout_seconds=AGENT_DECISION_TIMEOUT,
            phase_deadline_seconds=AGENT_PHASE_DEADLINE,
            seat_models=seat_models,
            default_model=ModelConfigManifest.from_config(base_cfg),
            metadata={
                "room_id": room.id,
                "source": "api_room",
            },
        )

    def _build_core_run_spec(self, run_spec: RunSpec) -> CoreRunSpec:
        """Freeze the exact generic execution contract for one live room."""
        migrated = legacy_werewolf_run_to_core(run_spec)
        raw = migrated.model_dump(mode="json")
        environment_config = dict(raw["environment_config"])
        environment_config.update({
            "decision_timeouts": dict(AGENT_DECISION_TIMEOUT_BY_PHASE),
            "phase_deadlines": dict(AGENT_PHASE_DEADLINE_BY_PHASE),
        })
        execution = dict(raw["execution"])
        execution.update({
            "cancellation_grace_seconds": self.cancellation_grace_seconds,
            "cleanup_timeout_seconds": self.cleanup_timeout_seconds,
        })
        raw["environment_config"] = environment_config
        raw["execution"] = execution
        return CoreRunSpec.model_validate(raw)

    @staticmethod
    def _build_live_actor_spec(run_spec: RunSpec) -> ActorSpec:
        default_model = (
            run_spec.default_model.model_dump(mode="json")
            if run_spec.default_model is not None
            else None
        )
        return ActorSpec(
            default_model=default_model,
            model_overrides={
                f"seat:{seat}": manifest.model_dump(mode="json")
                for seat, manifest in run_spec.seat_models.items()
            },
            human_actor_ids=[f"seat:{seat}" for seat in run_spec.human_seats],
        )

    @staticmethod
    def _resolve_live_actors(
        state: GameState,
        actors: dict[str, AgentActor],
        actor_spec: ActorSpec,
    ) -> AgentRegistry:
        """Resolve every live actor through the same canonical Core identities."""
        WerewolfEnvironmentPlugin._validate_actor_spec(state, actor_spec)
        players_by_actor_id = {
            f"seat:{player.seat}": player
            for player in state.players
        }

        def resolve(actor_id: str) -> AgentActor | None:
            player = players_by_actor_id.get(actor_id)
            if player is None:
                return None
            actor = actors.get(player.id)
            if actor is not None and getattr(actor, "seat", None) != player.seat:
                raise ValueError(
                    f"resolved actor identity does not match {actor_id}"
                )
            return actor

        registry = AgentRegistry(resolve)
        for actor_id in sorted(players_by_actor_id):
            actor = registry.resolve(actor_id)
            WerewolfEnvironmentPlugin._validate_actor_provenance(
                actor_id=actor_id,
                actor=actor,
                actor_spec=actor_spec,
            )
        return registry

    async def _run_core_room(self, room: Room) -> None:
        """Project one Core-owned lifecycle onto the interactive room surface."""
        prepared = room.prepared_run
        core_run_spec = room.core_run_spec
        if prepared is None or core_run_spec is None:
            cleanup_failures: list[dict[str, Any]] = []
            if prepared is not None:
                if not getattr(prepared, "_claimed", False):
                    cleanup_failures = await self._close_unclaimed_core_resources(
                        room_id=room.id,
                        session=prepared.session,
                        decision_runtime=prepared.decision_runtime,
                    )
                else:
                    cleanup_failures = [self._record_cleanup_failure({
                        "stage": "missing_core_runtime",
                        "error_type": "ClaimedCoreOwnershipAnomaly",
                        "timeout": False,
                        "pending_task_count": 0,
                        "fatal": True,
                    }, room_id=room.id)]
                room.prepared_run = None
            room.status = "failed"
            room.end_reason = (
                "cleanup_failure" if cleanup_failures else "missing_core_runtime"
            )
            room.error = (
                "Core runtime cleanup failed"
                if cleanup_failures
                else "Core runtime is not initialized"
            )
            for failure in cleanup_failures:
                await self._emit_room_event(room, {
                    "type": "room_cleanup_failed",
                    **failure,
                })
            await self._emit_room_event(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
            return

        try:
            result = await run_prepared_environment_run(
                core_run_spec,
                prepared=prepared,
            )
        except asyncio.CancelledError:
            room.prepared_run = None
            cleanup_row = self._latest_harness_payload(room, "run_cleanup_failed")
            if cleanup_row is not None:
                room.status = "failed"
                room.end_reason = "cleanup_failure"
                room.error = "Core cleanup failed while cancelling room"
                for failure in cleanup_row.get("failures") or []:
                    if isinstance(failure, dict):
                        self._record_cleanup_failure(failure, room_id=room.id)
            else:
                room.status = "cancelled"
                room.end_reason = "cancelled"
                room.error = "room task was cancelled"
            await self._broadcast_room_status(room)
            raise
        except Exception as err:  # validation failure before Core can normalize it
            if not getattr(prepared, "_claimed", False):
                await self._close_unclaimed_core_resources(
                    room_id=room.id,
                    session=prepared.session,
                    decision_runtime=prepared.decision_runtime,
                )
            room.prepared_run = None
            room.status = "failed"
            room.end_reason = "core_runtime_error"
            room.error = f"{type(err).__name__} during Core game loop"
            logger.error(
                "Core room lifecycle failed before result normalization "
                "room_id=%s error_type=%s",
                room.id,
                type(err).__name__,
            )
            await self._emit_room_event(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
            return

        room.prepared_run = None
        room.core_result = result
        if room.evidence_limit_error is not None:
            await self._finish_evidence_limit(room, room.evidence_limit_error)
            return
        if room.evidence_sink_error_type is not None:
            room.status = "failed"
            room.end_reason = "evidence_sink_failure"
            room.error = f"{room.evidence_sink_error_type} at durable evidence boundary"
            await self._emit_room_event(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
            return

        if result.status in {"completed", "incomplete"}:
            projected = self._core_terminal_projection(
                room,
                status=result.status,
                termination_reason=result.termination_reason,
            )
            if room.core_terminal_committed:
                current = (room.status, room.end_reason, room.error)
                if current != projected:
                    room.status = "failed"
                    room.end_reason = "core_terminal_projection_drift"
                    room.error = "Core result disagrees with committed terminal evidence"
                    await self._emit_room_event(room, {
                        "type": "game_error",
                        "reason": room.end_reason,
                        "message": room.error,
                    })
                    await self._broadcast_room_status(room)
                    return
                try:
                    self._close_provider_scope(room)
                except Exception as err:  # noqa: BLE001 - retain terminal cleanup evidence
                    failure = self._record_cleanup_failure({
                        "stage": "provider_scope_close",
                        "error_type": type(err).__name__,
                        "timeout": False,
                        "pending_task_count": 0,
                        "fatal": True,
                    }, room_id=room.id)
                    room.status = "failed"
                    room.end_reason = "cleanup_failure"
                    room.error = "provider budget scope cleanup failed"
                    await self._emit_room_event(room, {
                        "type": "room_cleanup_failed",
                        **failure,
                    })
                    await self._emit_room_event(room, {
                        "type": "game_error",
                        "reason": room.end_reason,
                        "message": room.error,
                    })
                    await self._broadcast_room_status(room)
                return
            room.status, room.end_reason, room.error = projected
        elif result.status == "timed_out":
            room.status = "timeout"
            room.end_reason = "timeout"
            room.error = "Core run deadline exceeded"
        else:
            cleanup_failures = int(
                result.harness_metrics.get("cleanup_failure_count") or 0
            )
            room.status = "failed"
            room.end_reason = (
                "cleanup_failure" if cleanup_failures else "core_run_failed"
            )
            room.error = f"{result.error_type or 'CoreRunError'} during game loop"

        if room.status in {"failed", "timeout"}:
            await self._emit_room_event(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
        await self._broadcast_room_status(room)

    @staticmethod
    def _core_terminal_projection(
        room: Room,
        *,
        status: str,
        termination_reason: str | None,
    ) -> tuple[str, str, str | None]:
        """Map a Core terminal outcome only when room state proves it."""
        if status == "completed":
            if room.state.phase == Phase.ENDED and room.state.winner is not None:
                return "ended", "completed", None
            return (
                "failed",
                "invalid_core_outcome",
                "Core completed without ended state and winner",
            )
        if status == "incomplete":
            reason = (termination_reason or "").strip()
            if (
                room.state.phase == Phase.ENDED
                and room.state.winner is None
                and reason
            ):
                return "incomplete", reason, None
            return (
                "failed",
                "invalid_core_outcome",
                "Core incomplete outcome has an invalid terminal state or reason",
            )
        raise ValueError(f"unsupported Core terminal status: {status}")

    @staticmethod
    def _latest_harness_payload(room: Room, event_type: str) -> dict[str, Any] | None:
        transcript = room.transcript
        if transcript is None:
            return None
        for entry in reversed(transcript.entries):
            if entry.kind == "harness" and entry.payload.get("type") == event_type:
                return dict(entry.payload)
        return None

    async def _finish_evidence_limit(
        self,
        room: Room,
        evidence_err: RoomEvidenceLimitError,
    ) -> None:
        """Terminate without appending to an evidence store that is already full."""
        room.status = "failed"
        room.end_reason = "evidence_limit"
        room.error = "room evidence capacity reached"
        room.terminal_at = time.monotonic()
        logger.critical(
            "room evidence capacity reached (room_id=%s current=%s limit=%s)",
            room.id,
            evidence_err.current,
            evidence_err.limit,
        )
        try:
            self._close_provider_scope(room)
        except Exception as err:  # noqa: BLE001 - retain terminal cleanup evidence
            self._record_cleanup_failure({
                "stage": "provider_scope_close",
                "error_type": type(err).__name__,
                "timeout": False,
                "pending_task_count": 0,
                "fatal": True,
            }, room_id=room.id)
        for payload in (
            {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            },
            {
                "type": "room_status",
                "status": room.status,
                "reason": room.end_reason,
                "error": room.error,
            },
        ):
            try:
                await self._broadcast(room, payload)
            except Exception as err:  # noqa: BLE001 - preserve terminal state
                logger.error(
                    "failed to broadcast evidence-limit terminal state "
                    "(room_id=%s error_type=%s)",
                    room.id,
                    type(err).__name__,
                )

    async def _run_orchestrator(self, room: Room) -> Any:
        """Run one orchestrator child without delegating cancellation to wait_for."""
        assert room.orchestrator is not None
        awaitable = room.orchestrator.run()
        if not inspect.isawaitable(awaitable):
            raise TypeError("orchestrator.run must return an awaitable")
        task = asyncio.ensure_future(awaitable)
        _set_task_name(task, f"room-orchestrator:{room.id}")
        try:
            if self.room_timeout is None or self.room_timeout <= 0:
                try:
                    result = await asyncio.shield(task)
                except Exception:
                    if room.evidence_limit_error is not None:
                        raise room.evidence_limit_error
                    raise
                if room.evidence_limit_error is not None:
                    raise room.evidence_limit_error
                return result
            done, _pending = await asyncio.wait(
                {task},
                timeout=self.room_timeout,
            )
        except asyncio.CancelledError as err:
            terminated, _interrupted = await _cancel_task_bounded(
                task,
                self.cancellation_grace_seconds,
            )
            if not terminated:
                failure = {
                    "stage": "orchestrator_run",
                    "error_type": "TaskIgnoredCancellation",
                    "timeout": False,
                    "pending_task_count": 1,
                    "fatal": True,
                }
                self._quarantine_task(task, failure, room_id=room.id)
                setattr(err, "room_cleanup_failure", failure)
            raise

        if task in done:
            # Calling result() here is deliberate: an orchestrator's own
            # TimeoutError is an ordinary run failure, not our wall deadline.
            try:
                result = task.result()
            except Exception:
                if room.evidence_limit_error is not None:
                    raise room.evidence_limit_error
                raise
            if room.evidence_limit_error is not None:
                raise room.evidence_limit_error
            return result

        terminated, caller_cancelled = await _cancel_task_bounded(
            task,
            self.cancellation_grace_seconds,
        )
        if not terminated:
            failure = {
                "stage": "orchestrator_run",
                "error_type": "TaskIgnoredCancellation",
                "timeout": True,
                "pending_task_count": 1,
                "fatal": True,
            }
            self._quarantine_task(task, failure, room_id=room.id)
        if caller_cancelled:
            cancelled = asyncio.CancelledError()
            if not terminated:
                setattr(cancelled, "room_cleanup_failure", failure)
            raise cancelled
        if not terminated:
            raise _RoomRunCleanupTimeout(failure)
        raise _RoomRunDeadlineExceeded

    async def _run_room(self, room: Room) -> None:
        if room.prepared_run is not None or room.core_run_spec is not None:
            await self._run_core_room(room)
            return
        if room.orchestrator is None:
            room.status = "failed"
            room.end_reason = "missing_orchestrator"
            room.error = "orchestrator is not initialized"
            await self._emit_room_event(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
            return
        try:
            await self._run_orchestrator(room)
        except _RoomRunCleanupTimeout as cleanup_err:
            failure = self._record_cleanup_failure(cleanup_err.failure, room_id=room.id)
            room.status = "failed"
            room.end_reason = "cleanup_failure"
            room.error = "room deadline cleanup failed after bounded cancellation"
            logger.critical(
                "room orchestrator ignored bounded deadline cancellation "
                "(room_id=%s pending_task_count=%s)",
                room.id,
                failure["pending_task_count"],
            )
            await self._emit_room_event(room, {
                "type": "room_cleanup_failed",
                **failure,
            })
            await self._emit_room_event(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
        except _RoomRunDeadlineExceeded:
            room.status = "timeout"
            room.end_reason = "timeout"
            room.error = f"room exceeded {self.room_timeout:.3f}s timeout"
            logger.error("房间 %s 游戏循环超时: %s", room.id, room.error)
            await self._emit_room_event(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
        except asyncio.CancelledError as cancelled:
            cleanup_failure = getattr(cancelled, "room_cleanup_failure", None)
            if isinstance(cleanup_failure, dict):
                failure = self._record_cleanup_failure(cleanup_failure, room_id=room.id)
                room.status = "failed"
                room.end_reason = "cleanup_failure"
                room.error = "room cancellation cleanup failed"
                await self._emit_room_event(room, {
                    "type": "room_cleanup_failed",
                    **failure,
                })
            else:
                room.status = "cancelled"
                room.end_reason = "cancelled"
                room.error = "room task was cancelled"
            await self._broadcast_room_status(room)
            raise
        except RoomEvidenceLimitError as evidence_err:
            # Evidence is append-only and source-indexed.  Once the cap is
            # reached, stop the game instead of silently evicting rows or
            # fabricating a partial transcript.  These terminal notifications
            # go through the bounded delivery stream but intentionally bypass
            # event_history/transcript, which are already at capacity.
            room.status = "failed"
            room.end_reason = "evidence_limit"
            room.error = "room evidence capacity reached"
            room.terminal_at = time.monotonic()
            logger.critical(
                "room evidence capacity reached (room_id=%s current=%s limit=%s)",
                room.id,
                evidence_err.current,
                evidence_err.limit,
            )
            try:
                self._close_provider_scope(room)
            except Exception as err:  # noqa: BLE001 - retain terminal cleanup evidence
                self._record_cleanup_failure(
                    {
                        "stage": "provider_scope_close",
                        "error_type": type(err).__name__,
                        "timeout": False,
                        "pending_task_count": 0,
                        "fatal": True,
                    },
                    room_id=room.id,
                )
            for payload in (
                {
                    "type": "game_error",
                    "reason": room.end_reason,
                    "message": room.error,
                },
                {
                    "type": "room_status",
                    "status": room.status,
                    "reason": room.end_reason,
                    "error": room.error,
                },
            ):
                try:
                    await self._broadcast(room, payload)
                except Exception as err:  # noqa: BLE001 - preserve terminal state
                    logger.error(
                        "failed to broadcast evidence-limit terminal state "
                        "(room_id=%s error_type=%s)",
                        room.id,
                        type(err).__name__,
                    )
        except Exception as err:  # noqa: BLE001
            room.status = "failed"
            room.end_reason = "error"
            room.error = _public_room_error(err)
            logger.error("房间 %s 游戏循环异常 error_type=%s", room.id, type(err).__name__)
            await self._emit_room_event(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
        else:
            if room.state.phase == Phase.ENDED and room.state.winner:
                room.status = "ended"
                room.end_reason = "completed"
                room.error = None
            elif (
                room.state.phase == Phase.ENDED
                and room.orchestrator is not None
                and room.orchestrator.termination_status == "incomplete"
            ):
                room.status = "incomplete"
                room.end_reason = room.orchestrator.termination_reason or "incomplete"
                room.error = None
            else:
                room.status = "failed"
                room.end_reason = "incomplete"
                room.error = "orchestrator returned before ended/winner"
                await self._emit_room_event(room, {
                    "type": "game_error",
                    "reason": room.end_reason,
                    "message": room.error,
                })
            await self._broadcast_room_status(room)

    async def _broadcast_room_status(self, room: Room) -> None:
        if room.status in TERMINAL_ROOM_STATUSES and room.terminal_at is None:
            room.terminal_at = time.monotonic()
        payload: dict[str, Any] = {
            "type": "room_status",
            "status": room.status,
            "reason": room.end_reason,
        }
        if room.error:
            payload["error"] = room.error
        await self._emit_room_event(room, payload)
        if room.status in TERMINAL_ROOM_STATUSES:
            try:
                self._close_provider_scope(room)
            except Exception as err:  # noqa: BLE001 - preserve terminal evidence
                failure = self._record_cleanup_failure({
                    "stage": "provider_scope_close",
                    "error_type": type(err).__name__,
                    "timeout": False,
                    "pending_task_count": 0,
                    "fatal": True,
                }, room_id=room.id)
                await self._emit_room_event(room, {
                    "type": "room_cleanup_failed",
                    **failure,
                })
                room.status = "failed"
                room.end_reason = "cleanup_failure"
                room.error = "provider budget scope cleanup failed"
                await self._emit_room_event(room, {
                    "type": "game_error",
                    "reason": room.end_reason,
                    "message": room.error,
                })
                await self._emit_room_event(room, {
                    "type": "room_status",
                    "status": room.status,
                    "reason": room.end_reason,
                    "error": room.error,
                })

    # ------------------------------------------------------------------
    # 信息隔离广播
    # ------------------------------------------------------------------
    def _make_event_broadcaster(self, room: Room):
        async def on_event(ev: dict[str, Any]) -> None:
            payload = dict(ev)
            with room.capability_lock:
                checkpoint = self._room_mutation_checkpoint(room)
                try:
                    self._store_room_event(room, payload)
                    await self._broadcast(room, payload)
                except BaseException as err:
                    self._rollback_room_mutation(room, checkpoint)
                    self._trip_evidence_sink(room, err)
                    raise
        return on_event

    @staticmethod
    def _room_mutation_checkpoint(room: Room) -> dict[str, Any]:
        transcript = room.transcript
        return {
            "event_len": len(room.event_history),
            "decision_len": len(room.decision_trace),
            "transcript": transcript,
            "transcript_len": len(transcript.entries) if transcript is not None else 0,
            "trace_seq": room.trace_seq,
            "status": room.status,
            "end_reason": room.end_reason,
            "error": room.error,
            "terminal_at": room.terminal_at,
            "core_terminal_committed": room.core_terminal_committed,
        }

    @staticmethod
    def _rollback_room_mutation(room: Room, checkpoint: dict[str, Any]) -> None:
        del room.event_history[int(checkpoint["event_len"]):]
        del room.decision_trace[int(checkpoint["decision_len"]):]
        original_transcript = checkpoint["transcript"]
        if original_transcript is None:
            room.transcript = None
        else:
            room.transcript = original_transcript
            del original_transcript.entries[int(checkpoint["transcript_len"]):]
        room.trace_seq = int(checkpoint["trace_seq"])
        room.status = str(checkpoint["status"])
        room.end_reason = checkpoint["end_reason"]
        room.error = checkpoint["error"]
        room.terminal_at = checkpoint["terminal_at"]
        room.core_terminal_committed = bool(checkpoint["core_terminal_committed"])

    def _store_room_event(self, room: Room, payload: dict[str, Any]) -> dict[str, Any]:
        # Replayable game surfaces never carry private model reasoning. That
        # evidence belongs only in ``decision_trace``.
        stored = strip_model_private_reasoning(payload)
        if not isinstance(stored, dict):  # defensive for non-standard Mapping callers
            stored = {}
        try:
            self._ensure_evidence_capacity(room)
        except RoomEvidenceLimitError as err:
            self._trip_evidence_limit(room, err)
            raise
        stored.setdefault("_ts", time.monotonic())
        generated_trace_seq = "_trace_seq" not in stored
        previous_trace_seq = room.trace_seq
        if generated_trace_seq:
            stored["_trace_seq"] = self._next_trace_seq(room)
        room.event_history.append(stored)
        try:
            self._append_transcript(
                room,
                "event",
                stored,
                source_idx=len(room.event_history) - 1,
            )
        except Exception:
            # Keep the two source representations transactional if transcript
            # validation rejects an integration-provided payload.
            room.event_history.pop()
            if generated_trace_seq and room.trace_seq == previous_trace_seq + 1:
                room.trace_seq = previous_trace_seq
            raise
        return stored

    async def _emit_room_event(self, room: Room, payload: dict[str, Any]) -> None:
        """Commit lifecycle evidence before making its live delivery visible."""
        with room.capability_lock:
            checkpoint = self._room_mutation_checkpoint(room)
            try:
                self._store_room_event(room, payload)
                await self._broadcast(room, payload)
            except BaseException:
                self._rollback_room_mutation(room, checkpoint)
                raise

    def _record_cleanup_failure(
        self,
        failure: dict[str, Any],
        *,
        room_id: str | None = None,
    ) -> dict[str, Any]:
        safe = redact_sensitive(dict(failure))
        if not isinstance(safe, dict):
            safe = {
                "stage": "unknown",
                "error_type": "CleanupError",
                "timeout": False,
                "pending_task_count": 0,
                "fatal": True,
            }
        if room_id is not None:
            safe["room_id"] = room_id
        safe.setdefault("fatal", True)
        safe.setdefault("pending_task_count", 0)
        self._cleanup_failures.append(safe)
        return safe

    def _quarantine_task(
        self,
        task: asyncio.Future[Any],
        failure: dict[str, Any],
        *,
        room_id: str | None = None,
    ) -> None:
        """Keep an unforceable task alive and consume any eventual result."""
        if task.done():
            _consume_task_result(task)
            return
        metadata = dict(failure)
        if room_id is not None:
            metadata["room_id"] = room_id
        with self._quarantined_tasks_lock:
            self._quarantined_tasks[task] = metadata

        def forget(done: asyncio.Future[Any]) -> None:
            with self._quarantined_tasks_lock:
                self._quarantined_tasks.pop(done, None)
            _consume_task_result(done)

        task.add_done_callback(forget)
        logger.critical(
            "task ignored bounded cancellation and remains in-process "
            "(stage=%s room_id=%s task_name=%s)",
            metadata.get("stage"),
            room_id,
            _task_name(task),
        )

    def _make_core_task_quarantine_sink(self, room: Room):
        """Bind Core child-task ownership to this room manager's shutdown."""
        def quarantine(task: asyncio.Future[Any], stage: str) -> None:
            self._quarantine_task(
                task,
                {
                    "stage": f"core:{stage}",
                    "error_type": "TaskIgnoredCancellation",
                    "timeout": True,
                    "pending_task_count": 1,
                    "fatal": True,
                },
                room_id=room.id,
            )

        return quarantine

    def _make_trace_recorder(self, room: Room):
        def on_trace(item: dict[str, Any]) -> None:
            with room.capability_lock:
                checkpoint = self._room_mutation_checkpoint(room)
                try:
                    stored = dict(item)
                    try:
                        self._ensure_evidence_capacity(room)
                    except RoomEvidenceLimitError as err:
                        self._trip_evidence_limit(room, err)
                        raise
                    stored.setdefault("_ts", time.monotonic())
                    if "_trace_seq" not in stored:
                        stored["_trace_seq"] = self._next_trace_seq(room)
                    room.decision_trace.append(stored)
                    self._append_transcript(
                        room,
                        "decision",
                        stored,
                        source_idx=len(room.decision_trace) - 1,
                    )
                    self._persist_room(room)
                except BaseException as err:
                    self._rollback_room_mutation(room, checkpoint)
                    self._trip_evidence_sink(room, err)
                    raise

        return on_trace

    def _make_harness_recorder(self, room: Room):
        """Return the full sink for Core lifecycle evidence in this room."""
        def on_harness(item: dict[str, Any]) -> None:
            event_type = str(item.get("type") or "")
            terminal_outcome = event_type in {"run_completed", "run_incomplete"}
            with room.capability_lock:
                checkpoint = self._room_mutation_checkpoint(room)
                try:
                    if terminal_outcome and room.evidence_sink_error_type is not None:
                        raise PersistenceError("prior durable evidence sink failure")
                    projection: tuple[str, str, str | None] | None = None
                    if terminal_outcome:
                        projection = self._core_terminal_projection(
                            room,
                            status=(
                                "completed" if event_type == "run_completed" else "incomplete"
                            ),
                            termination_reason=item.get("termination_reason"),
                        )
                    additional = 1
                    if projection is not None:
                        additional += 1 + int(projection[2] is not None)
                    try:
                        self._ensure_evidence_capacity(room, additional=additional)
                    except RoomEvidenceLimitError as err:
                        self._trip_evidence_limit(room, err)
                        raise
                    transcript = room.transcript
                    if transcript is None or transcript.run_id != room.id:
                        raise RuntimeError("room transcript is unavailable for Core evidence")
                    transcript.append("harness", dict(item))

                    delivery_payloads: list[tuple[dict[str, Any], bool]] = []
                    if projection is not None:
                        room.status, room.end_reason, room.error = projection
                        room.terminal_at = time.monotonic()
                        if room.error is not None:
                            error_payload = {
                                "type": "game_error",
                                "reason": room.end_reason,
                                "message": room.error,
                            }
                            self._store_room_event(room, error_payload)
                            delivery_payloads.append((error_payload, True))
                        status_payload: dict[str, Any] = {
                            "type": "room_status",
                            "status": room.status,
                            "reason": room.end_reason,
                        }
                        if room.error is not None:
                            status_payload["error"] = room.error
                        self._store_room_event(room, status_payload)
                        delivery_payloads.append((status_payload, False))

                    if delivery_payloads:
                        with room.delivery_lock:
                            delivery_checkpoint = self._delivery_checkpoint_locked(room)
                            try:
                                outbound: list[tuple[str, RoomClient, str]] = []
                                for payload, initial_replay in delivery_payloads:
                                    outbound.extend(self._publish_delivery_locked(
                                        room,
                                        payload,
                                        initial_replay=initial_replay,
                                    ))
                                self._persist_room(room)
                            except BaseException:
                                self._restore_delivery_checkpoint_locked(
                                    room,
                                    delivery_checkpoint,
                                )
                                raise
                            for cid, connection, message in outbound:
                                if room.clients.get(cid) is connection:
                                    self._enqueue_client_message(
                                        room,
                                        cid,
                                        connection,
                                        message,
                                    )
                    else:
                        self._persist_room(room)
                except BaseException as err:
                    self._rollback_room_mutation(room, checkpoint)
                    self._trip_evidence_sink(room, err)
                    raise
                if terminal_outcome:
                    room.core_terminal_committed = True

        return on_harness

    def _next_trace_seq(self, room: Room) -> int:
        room.trace_seq += 1
        return room.trace_seq

    def _append_transcript(
        self,
        room: Room,
        kind: str,
        payload: dict[str, Any],
        *,
        source_idx: int,
    ) -> None:
        if room.transcript is None:
            room.transcript = Transcript(
                run_id=room.id,
                metadata={
                    "room_id": room.id,
                    "role_seed": room.role_seed,
                    "actor_seed": room.actor_seed,
                    "orchestrator_seed": room.orchestrator_seed,
                },
            )
        room.transcript.append(
            kind,
            payload,
            ts_monotonic=float(payload.get("_ts") or time.monotonic()),
            source_idx=source_idx,
        )

    def _initialize_delivery_streams(self, room: Room) -> None:
        """Create every authorization stream before the first room event."""
        with room.delivery_lock:
            self._ensure_delivery_storage_locked(room)
            expected_keys = {"spectate", "god"} | {
                f"play:{seat}" for seat in sorted(room.human_seats)
            }
            unknown_keys = set(room.delivery_stream_metadata) - expected_keys
            if unknown_keys:
                raise PersistenceError("persisted delivery stream audience is invalid")
            self._ensure_delivery_stream_locked(room, mode="spectate", seat=None)
            self._ensure_delivery_stream_locked(room, mode="god", seat=None)
            for seat in sorted(room.human_seats):
                self._ensure_delivery_stream_locked(room, mode="play", seat=seat)

    def _ensure_delivery_storage_locked(self, room: Room) -> None:
        if room.delivery_source_history.maxlen != self.ws_delivery_history_size:
            room.delivery_source_history = deque(
                room.delivery_source_history,
                maxlen=self.ws_delivery_history_size,
            )

    @staticmethod
    def _delivery_stream_key(mode: str, seat: int | None) -> str:
        if mode in {"god", "replay"}:
            return "god"
        if mode == "play":
            if seat is None:
                raise ValueError("play delivery stream requires a seat")
            return f"play:{seat}"
        return "spectate"

    def _ensure_delivery_stream_locked(
        self,
        room: Room,
        *,
        mode: str,
        seat: int | None,
    ) -> DeliveryStream:
        self._ensure_delivery_storage_locked(room)
        key = self._delivery_stream_key(mode, seat)
        existing = room.delivery_streams.get(key)
        if existing is not None:
            return existing

        projection_mode = "god" if mode == "replay" else mode
        metadata = room.delivery_stream_metadata.get(key)
        persisted_cursor: int | None = None
        persisted_stream_id: str | None = None
        persisted_history_gap = False
        if metadata is not None:
            if set(metadata) != {"stream_id", "cursor", "history_gap"}:
                raise PersistenceError("persisted delivery stream metadata is invalid")
            raw_stream_id = metadata.get("stream_id")
            raw_cursor = metadata.get("cursor")
            raw_history_gap = metadata.get("history_gap")
            if (
                not isinstance(raw_stream_id, str)
                or not raw_stream_id
                or len(raw_stream_id) > 128
                or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for ch in raw_stream_id)
                or isinstance(raw_cursor, bool)
                or not isinstance(raw_cursor, int)
                or raw_cursor < 0
                or raw_cursor > room.delivery_source_seq
                or not isinstance(raw_history_gap, bool)
            ):
                raise PersistenceError("persisted delivery stream metadata is invalid")
            persisted_stream_id = raw_stream_id
            persisted_cursor = raw_cursor
            persisted_history_gap = raw_history_gap
        stream = DeliveryStream(
            key=key,
            mode=projection_mode,
            seat=seat,
            stream_id=persisted_stream_id or secrets.token_urlsafe(12),
            history=deque(maxlen=self.ws_delivery_history_size),
            history_gap=False,
        )
        visible_count = 0
        for _, payload, _initial_replay in room.delivery_source_history:
            if self._should_receive(room, payload, seat, projection_mode):
                projected = self._payload_for_client(room, payload, seat, projection_mode)
                if projected:
                    visible_count += 1
        if persisted_cursor is not None:
            if persisted_cursor < visible_count:
                raise PersistenceError("persisted delivery stream cursor is inconsistent")
            # The retained source window may begin after older visible events.
            # Start at the absolute cursor preceding that window so rebuilt
            # delivery IDs remain stable across process restart.
            stream.cursor = persisted_cursor - visible_count
            stream.history_gap = persisted_history_gap or stream.cursor > 0
        else:
            # Without metadata (legacy rows), preserve the old conservative
            # signal: a non-empty source cursor with a bounded window may have
            # dropped events. New rows use per-stream metadata above.
            stream.history_gap = room.delivery_source_seq > len(room.delivery_source_history)
        # This lazy path mainly supports directly-constructed Room instances in
        # tests. Real rooms initialize all authorized streams at creation.
        for _, payload, initial_replay in room.delivery_source_history:
            self._append_visible_delivery_locked(
                room,
                stream,
                payload,
                initial_replay=initial_replay,
            )
        room.delivery_streams[key] = stream
        return stream

    def _append_visible_delivery_locked(
        self,
        room: Room,
        stream: DeliveryStream,
        payload: dict[str, Any],
        *,
        initial_replay: bool,
    ) -> DeliveryRecord | None:
        if not self._should_receive(room, payload, stream.seat, stream.mode):
            return None
        projected = self._payload_for_client(room, payload, stream.seat, stream.mode)
        if not projected:
            return None
        stream.cursor += 1
        delivery_id = f"{stream.stream_id}.{stream.cursor}"
        delivered = dict(projected)
        # Never trust environment-provided delivery metadata. It is assigned
        # only after privacy projection for this exact stream.
        delivered["delivery_seq"] = stream.cursor
        delivered["delivery_id"] = delivery_id
        record = DeliveryRecord(
            seq=stream.cursor,
            delivery_id=delivery_id,
            payload=delivered,
            initial_replay=initial_replay,
        )
        if len(stream.history) == stream.history.maxlen:
            stream.history_gap = True
        stream.history.append(record)
        return record

    @staticmethod
    def _delivery_checkpoint_locked(room: Room) -> dict[str, Any]:
        return {
            "source_seq": room.delivery_source_seq,
            "source_history": deque(
                room.delivery_source_history,
                maxlen=room.delivery_source_history.maxlen,
            ),
            "streams": {
                key: DeliveryStream(
                    key=stream.key,
                    mode=stream.mode,
                    seat=stream.seat,
                    stream_id=stream.stream_id,
                    cursor=stream.cursor,
                    history=deque(stream.history, maxlen=stream.history.maxlen),
                    history_gap=stream.history_gap,
                )
                for key, stream in room.delivery_streams.items()
            },
        }

    @staticmethod
    def _restore_delivery_checkpoint_locked(
        room: Room,
        checkpoint: dict[str, Any],
    ) -> None:
        room.delivery_source_seq = int(checkpoint["source_seq"])
        room.delivery_source_history = checkpoint["source_history"]
        room.delivery_streams = checkpoint["streams"]

    def _publish_delivery_locked(
        self,
        room: Room,
        payload: dict[str, Any],
        *,
        initial_replay: bool,
    ) -> list[tuple[str, RoomClient, str]]:
        """Stage one source event and return messages to enqueue after commit."""
        # Rooms can be instantiated directly by integrations/tests instead of
        # through create_room(). Materialize their known audience streams before
        # recording the first source item so a later private event cannot turn
        # into a false initial history gap.
        self._ensure_delivery_stream_locked(room, mode="spectate", seat=None)
        self._ensure_delivery_stream_locked(room, mode="god", seat=None)
        for known_seat in sorted(room.human_seats):
            self._ensure_delivery_stream_locked(room, mode="play", seat=known_seat)
        self._ensure_delivery_storage_locked(room)
        room.delivery_source_seq += 1
        source = dict(payload)
        room.delivery_source_history.append((room.delivery_source_seq, source, initial_replay))

        appended: dict[str, DeliveryRecord] = {}
        for key, stream in room.delivery_streams.items():
            record = self._append_visible_delivery_locked(
                room,
                stream,
                source,
                initial_replay=initial_replay,
            )
            if record is not None:
                appended[key] = record

        serialized: dict[str, str] = {}
        outbound: list[tuple[str, RoomClient, str]] = []
        for cid, connection in list(room.clients.items()):
            # Preserve compatibility with rooms directly populated by older
            # lifecycle tests; only managed RoomClient instances are writable.
            if not isinstance(connection, RoomClient):
                continue
            record = appended.get(connection.stream_key)
            if record is None:
                continue
            message = serialized.get(connection.stream_key)
            if message is None:
                message = json.dumps(record.payload, ensure_ascii=False, default=str)
                serialized[connection.stream_key] = message
            outbound.append((cid, connection, message))
        return outbound

    def _enqueue_client_message(
        self,
        room: Room,
        cid: str,
        connection: RoomClient,
        message: str,
    ) -> None:
        """Queue on the socket's owner loop, including TestClient portal use."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is connection.loop:
            self._enqueue_client_message_on_loop(room, cid, connection, message)
            return
        try:
            connection.loop.call_soon_threadsafe(
                self._enqueue_client_message_on_loop,
                room,
                cid,
                connection,
                message,
            )
        except RuntimeError:
            # The owning loop has already stopped. Removing the client is safe
            # and avoids retaining a dead room indefinitely.
            room.clients.pop(cid, None)

    def _enqueue_client_message_on_loop(
        self,
        room: Room,
        cid: str,
        connection: RoomClient,
        message: str,
    ) -> None:
        overflowed = False
        with room.delivery_lock:
            if room.clients.get(cid) is not connection:
                return
            try:
                connection.queue.put_nowait(message)
            except asyncio.QueueFull:
                room.clients.pop(cid, None)
                overflowed = True
        if overflowed:
            logger.warning("WS client backpressure disconnect room=%s cid=%s", room.id, cid)
            self._terminate_client_on_owner_loop(
                room,
                connection,
                code=4410,
                reason="client too slow",
            )

    def _terminate_client_on_owner_loop(
        self,
        room: Room,
        connection: RoomClient,
        *,
        code: int,
        reason: str,
    ) -> None:
        async def close_socket() -> None:
            try:
                awaitable = connection.websocket.close(code=code, reason=reason)
            except Exception:  # noqa: BLE001
                return
            failure, _error = await self._bounded_cleanup_awaitable(
                awaitable,
                stage="ws_socket_close",
                room_id=room.id,
            )
            if failure is not None:
                self._record_cleanup_failure(failure, room_id=room.id)

        def terminate() -> None:
            current = asyncio.current_task()
            for task in (connection.handshake_task, connection.writer_task):
                if task is not None and task is not current and not task.done():
                    task.cancel()
            connection.close_task = connection.loop.create_task(
                close_socket(),
                name=f"ws-close:{room.id}",
            )

        try:
            if asyncio.get_running_loop() is connection.loop:
                terminate()
            else:
                connection.loop.call_soon_threadsafe(terminate)
        except RuntimeError:
            pass

    async def _client_writer(self, room: Room, cid: str, connection: RoomClient) -> None:
        send_failed = False
        try:
            while True:
                message = await connection.queue.get()
                await connection.websocket.send_text(message)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            send_failed = True
        finally:
            with room.delivery_lock:
                if room.clients.get(cid) is connection:
                    room.clients.pop(cid, None)
            if send_failed:
                try:
                    awaitable = connection.websocket.close(code=1011, reason="delivery failed")
                except Exception:  # noqa: BLE001
                    return
                failure, _error = await self._bounded_cleanup_awaitable(
                    awaitable,
                    stage="ws_delivery_failure_close",
                    room_id=room.id,
                )
                if failure is not None:
                    self._record_cleanup_failure(failure, room_id=room.id)

    async def _broadcast(
        self,
        room: Room,
        payload: dict[str, Any],
        *,
        initial_replay: bool | None = None,
    ) -> None:
        """Stage, persist, then enqueue without exposing an uncommitted row."""
        payload = strip_model_private_reasoning(payload)
        if not isinstance(payload, dict):
            payload = {}
        if initial_replay is None:
            # The snapshot itself is authoritative for lifecycle status; all
            # other room messages remain available for an initial replay.
            initial_replay = payload.get("type") != "room_status"
        with room.capability_lock:
            with room.delivery_lock:
                checkpoint = self._delivery_checkpoint_locked(room)
                try:
                    outbound = self._publish_delivery_locked(
                        room,
                        payload,
                        initial_replay=initial_replay,
                    )
                    self._persist_room(room)
                except BaseException:
                    self._restore_delivery_checkpoint_locked(room, checkpoint)
                    raise
                for cid, connection, message in outbound:
                    if room.clients.get(cid) is connection:
                        self._enqueue_client_message(
                            room,
                            cid,
                            connection,
                            message,
                        )

    def send_client_text(self, room: Room, cid: str, message: str) -> bool:
        """Serialize all control replies through the connection writer."""
        with room.delivery_lock:
            connection = room.clients.get(cid)
            if not isinstance(connection, RoomClient):
                return False
            self._enqueue_client_message(room, cid, connection, message)
            return True

    def send_client_payload(self, room: Room, cid: str, payload: dict[str, Any]) -> bool:
        return self.send_client_text(
            room,
            cid,
            json.dumps(payload, ensure_ascii=False, default=str),
        )

    def _payload_for_client(
        self,
        room: Room,
        payload: dict[str, Any],
        seat: int | None,
        mode: str,
    ) -> dict[str, Any]:
        """Return the mode-specific event payload after privacy filtering."""
        if mode in ("god", "replay"):
            projected = project_payload_for_audience(payload, kind="event", audience="god")
        elif mode == "play":
            player_id = None
            if seat is not None:
                player = next((p for p in room.state.players if p.seat == seat), None)
                player_id = player.id if player else None
            projected = project_payload_for_audience(
                payload,
                kind="event",
                audience="player",
                seat=seat,
                player_id=player_id,
            )
        else:
            projected = project_payload_for_audience(payload, kind="event", audience="public")
        return projected if projected is not None else {}

    def _should_receive(self, room: Room, payload: dict[str, Any], seat: int | None, mode: str) -> bool:
        """信息隔离裁决:该客户端能否收到此事件。"""
        etype = payload.get("type", "")

        # 全知模式:看一切
        if mode == "god":
            return True

        # 回放/复盘:看一切,但只有赛后连接才允许进入 replay 模式。
        if mode in ("replay",):
            return room.status in TERMINAL_ROOM_STATUSES

        recipients = payload.get("recipients") or []
        visibility = payload.get("visibility")
        if visibility == "private" or recipients:
            if mode != "play" or seat is None:
                return False
            player = next((p for p in room.state.players if p.seat == seat), None)
            recipient_set = {str(r) for r in recipients}
            return bool(
                player
                and (
                    player.id in recipient_set
                    or str(seat) in recipient_set
                    or payload.get("seat") == seat
                )
            )

        # 公开事件:所有人(spectate/play)都收
        public_types = {
            "phase_started", "night_resolved", "speech", "vote_cast", "vote_resolved",
            "last_words", "last_words_skipped", "hunter_shot", "game_ended", "room_status", "game_error",
            "vote_incomplete", "vote_rejected", "agent_decision_failed",
            "decision_envelope_rejected", "decision_validation_failed", "room_cleanup_failed",
        }
        if etype in public_types:
            return True

        # 人类操作请求:仅该 seat 的 play 模式收
        if etype == "human_action_request":
            return mode == "play" and seat == payload.get("seat")

        return False

    # ------------------------------------------------------------------
    # WebSocket 连接管理
    # ------------------------------------------------------------------
    async def connect(
        self,
        room: Room,
        ws: WebSocket,
        *,
        seat: int | None,
        mode: str,
        since: int | None = None,
        capability_token: str | None = None,
        websocket_subprotocol: str | None = None,
    ) -> str:
        import uuid

        cid = uuid.uuid4().hex[:8]
        loop = asyncio.get_running_loop()
        connection = RoomClient(
            websocket=ws,
            seat=seat,
            mode=mode,
            stream_key=self._delivery_stream_key(mode, seat),
            queue=asyncio.Queue(maxsize=self.ws_client_queue_size),
            loop=loop,
            handshake_task=asyncio.current_task(),
        )
        # Capability verification and registration share one critical section
        # with rotation/revocation. A socket can therefore only register before
        # a mutation (and be invalidated by it) or after it (and be rejected).
        with room.capability_lock:
            if mode in {"god", "replay"} and not self.valid_admin_token(
                room,
                capability_token,
            ):
                raise CapabilityAuthorizationError("admin capability changed or was revoked")
            if mode == "play" and (
                seat not in room.human_seats
                or not self.valid_seat_token(room, seat, capability_token)
            ):
                raise CapabilityAuthorizationError("seat capability changed or was revoked")
            with room.delivery_lock:
                if len(room.clients) >= self.max_ws_clients_per_room:
                    raise RoomClientCapacityError(
                        "room WebSocket client capacity reached "
                        f"({self.max_ws_clients_per_room})"
                    )
                stream = self._ensure_delivery_stream_locked(room, mode=mode, seat=seat)
                cutover = stream.cursor
                earliest = stream.history[0].seq if stream.history else cutover + 1
                if since is not None:
                    if isinstance(since, bool) or since < 0:
                        raise InvalidDeliveryCursorError("delivery cursor must be a non-negative integer")
                    if since > cutover:
                        raise FutureDeliveryCursorError(
                            f"future delivery cursor: requested={since}, current={cutover}"
                        )
                    if since < earliest - 1:
                        raise DeliveryHistoryGapError(
                            requested=since,
                            earliest=earliest,
                            current=cutover,
                        )
                replay = [
                    record
                    for record in stream.history
                    if (
                        (since is None and record.initial_replay)
                        or (since is not None and record.seq > since)
                    )
                ]
                snapshot = {
                    "type": "snapshot",
                    "status": room.status,
                    "view": self._view_for(room, seat, mode),
                    "stream_id": stream.stream_id,
                    "cursor": cutover,
                    "resumed_from": since,
                    "replay_from": replay[0].seq if replay else cutover + 1,
                    "history_gap": bool(since is None and (stream.history_gap or earliest > 1)),
                }
                # Registration and cutover capture are one critical section. Any
                # later delivery is queued and cannot interleave with handshake
                # snapshot/history writes.
                room.clients[cid] = connection

        try:
            await ws.accept(subprotocol=websocket_subprotocol)
            await ws.send_text(json.dumps(snapshot, ensure_ascii=False, default=str))
            for record in replay:
                await ws.send_text(json.dumps(record.payload, ensure_ascii=False, default=str))
        except asyncio.CancelledError:
            self.disconnect(room, cid)
            raise
        except Exception:
            self.disconnect(room, cid)
            raise

        with room.delivery_lock:
            if room.clients.get(cid) is connection:
                connection.handshake_task = None
                connection.writer_task = loop.create_task(
                    self._client_writer(room, cid, connection),
                    name=f"ws-writer:{room.id}:{cid}",
                )
            else:
                self._terminate_client_on_owner_loop(
                    room,
                    connection,
                    code=4410,
                    reason="client too slow during replay",
                )
        return cid

    def disconnect(self, room: Room, cid: str) -> None:
        with room.delivery_lock:
            connection = room.clients.pop(cid, None)
        if not isinstance(connection, RoomClient):
            return

        def cancel_writer() -> None:
            task = connection.writer_task
            if task is not None and not task.done():
                task.cancel()

        try:
            if asyncio.get_running_loop() is connection.loop:
                cancel_writer()
            else:
                connection.loop.call_soon_threadsafe(cancel_writer)
        except RuntimeError:
            pass

    def _view_for(self, room: Room, seat: int | None, mode: str) -> dict[str, Any]:
        """根据 mode 生成视图快照。"""
        state = room.state
        # Persona is a seat-owned strategic prior, not public game truth. It
        # is exposed only in the authorized God/Admin projection; publishing
        # it to public/player snapshots would reveal a private control
        # variable to other Agents.
        seat_to_persona: dict[str, str] = {}
        for pid, actor in room.actors.items():
            player = state.get_player(pid) if pid in [p.id for p in state.players] else None
            if player is None:
                continue
            seat_to_persona[str(player.seat)] = getattr(actor, "persona_name", "")
        if mode in {"god", "replay"}:
            view = state.public_view() | {"god": True, "players_full": [
                {"seat": p.seat, "name": p.name, "role": p.role, "team": p.team, "alive": p.alive,
                 "persona": seat_to_persona.get(str(p.seat), "")}
                for p in state.players
            ]}
            view["hidden_state"] = {
                "witch_antidote": state.witch_antidote,
                "witch_poison": state.witch_poison,
                "last_guarded_seat": state.last_guarded_seat,
                "pending_hunter": list(state.pending_hunter),
            }
            # LLM 统计
            view["llm_stats"] = self.router.stats.snapshot()
            return view
        view = state.public_view() if mode != "play" or seat is None else state.private_view_for(
            next((p.id for p in state.players if p.seat == seat), "")
        )
        return view

    # ------------------------------------------------------------------
    # 人类玩家操作(play 模式)
    # ------------------------------------------------------------------
    async def handle_human_action(self, room: Room, seat: int, action: dict[str, Any]) -> None:
        """处理人类玩家的操作(发言/投票/夜间行动)。

        将操作放入对应 AgentActor 的人类队列,由编排器消费。
        """
        player = next((p for p in room.state.players if p.seat == seat), None)
        if player is None:
            logger.warning("人类操作 seat=%s 不存在", seat)
            return
        actor = room.actors.get(player.id)
        if actor is None or not actor.is_human:
            logger.warning("seat=%s 不是人类玩家", seat)
            return
        accepted, reason = actor.enqueue_human_action(action)
        request_id = str(action.get("request_id") or "")
        if not accepted:
            logger.warning("人类操作被拒绝 seat=%s reason=%s", seat, reason)
            await self._broadcast(room, {
                "type": "human_action_rejected",
                "seat": seat,
                "request_id": request_id,
                "reason": reason,
                "visibility": "private",
                "recipients": [player.id],
            })
            return
        await self._broadcast(room, {
            "type": "human_action_accepted",
            "seat": seat,
            "request_id": request_id,
            "visibility": "private",
            "recipients": [player.id],
        })
        logger.info("人类操作已入队 seat=%s action=%s request_id=%s", seat, action.get("type"), request_id)

    async def _bounded_cleanup_awaitable(
        self,
        awaitable: Any,
        *,
        stage: str,
        room_id: str | None = None,
        cancel_on_timeout: bool = True,
    ) -> tuple[dict[str, Any] | None, BaseException | None]:
        if not inspect.isawaitable(awaitable):
            return ({
                "stage": stage,
                "error_type": "CleanupProtocolError",
                "timeout": False,
                "pending_task_count": 0,
                "fatal": True,
            }, None)
        task = asyncio.ensure_future(awaitable)
        _set_task_name(task, f"room-manager-cleanup:{stage}")
        try:
            done, _pending = await asyncio.wait(
                {task},
                timeout=self.cleanup_timeout_seconds,
            )
        except asyncio.CancelledError:
            terminated, _interrupted = await _cancel_task_bounded(
                task,
                self.cancellation_grace_seconds,
            )
            if not terminated:
                failure = {
                    "stage": stage,
                    "error_type": "TaskIgnoredCancellation",
                    "timeout": False,
                    "pending_task_count": 1,
                    "fatal": True,
                }
                self._quarantine_task(task, failure, room_id=room_id)
            raise
        if task in done:
            try:
                task.result()
            except asyncio.CancelledError as err:
                return ({
                    "stage": stage,
                    "error_type": "CleanupCancelled",
                    "timeout": False,
                    "pending_task_count": 0,
                    "fatal": True,
                }, err)
            except Exception as err:  # noqa: BLE001 - evidence excludes details
                return ({
                    "stage": stage,
                    "error_type": type(err).__name__,
                    "timeout": False,
                    "pending_task_count": 0,
                    "fatal": True,
                }, err)
            return None, None

        if not cancel_on_timeout:
            failure = {
                "stage": stage,
                "error_type": "CleanupWorkerTimeout",
                "timeout": True,
                "pending_task_count": 1,
                "fatal": True,
            }
            self._quarantine_task(task, failure, room_id=room_id)
            return failure, None

        terminated, caller_cancelled = await _cancel_task_bounded(
            task,
            self.cancellation_grace_seconds,
        )
        failure = {
            "stage": stage,
            "error_type": "CleanupTimeout" if terminated else "TaskIgnoredCancellation",
            "timeout": True,
            "pending_task_count": 0 if terminated else 1,
            "fatal": True,
        }
        if not terminated:
            self._quarantine_task(task, failure, room_id=room_id)
        if caller_cancelled:
            raise asyncio.CancelledError
        return failure, None

    async def _cancel_shutdown_tasks(
        self,
        task_metadata: dict[asyncio.Future[Any], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Cancel manager-owned tasks concurrently within one bounded window."""
        current_loop = asyncio.get_running_loop()
        local: set[asyncio.Future[Any]] = set()
        failures: list[dict[str, Any]] = []
        for task, metadata in task_metadata.items():
            if task.done():
                _consume_task_result(task)
                continue
            try:
                owner_loop = task.get_loop()
            except Exception:  # noqa: BLE001
                owner_loop = current_loop
            if owner_loop is not current_loop:
                try:
                    owner_loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    pass
                failures.append({
                    "stage": str(metadata.get("stage") or "background_task"),
                    "error_type": "ForeignLoopTaskPending",
                    "timeout": True,
                    "pending_task_count": 1,
                    "fatal": True,
                    **({"room_id": metadata["room_id"]} if metadata.get("room_id") else {}),
                })
                continue
            task.cancel()
            local.add(task)

        if not local:
            return failures
        try:
            done, pending = await asyncio.wait(
                local,
                timeout=self.cleanup_timeout_seconds,
            )
        except asyncio.CancelledError:
            pending, _interrupted = await _cancel_tasks_bounded(
                local,
                self.cancellation_grace_seconds,
            )
            for task in pending:
                metadata = task_metadata[task]
                failure = {
                    "stage": str(metadata.get("stage") or "background_task"),
                    "error_type": "TaskIgnoredCancellation",
                    "timeout": False,
                    "pending_task_count": 1,
                    "fatal": True,
                }
                self._quarantine_task(task, failure, room_id=metadata.get("room_id"))
            raise

        for task in done:
            if task.cancelled():
                continue
            try:
                task.result()
            except asyncio.CancelledError:
                continue
            except Exception as err:  # noqa: BLE001 - keep task details private
                metadata = task_metadata[task]
                failures.append({
                    "stage": str(metadata.get("stage") or "background_task"),
                    "error_type": type(err).__name__,
                    "timeout": False,
                    "pending_task_count": 0,
                    "fatal": True,
                    **({"room_id": metadata["room_id"]} if metadata.get("room_id") else {}),
                })

        if pending:
            initially_pending = set(pending)
            pending, caller_cancelled = await _cancel_tasks_bounded(
                pending,
                self.cancellation_grace_seconds,
            )
            for task in initially_pending:
                metadata = task_metadata[task]
                still_pending = task in pending
                failure = {
                    "stage": str(metadata.get("stage") or "background_task"),
                    "error_type": (
                        "TaskIgnoredCancellation" if still_pending else "CleanupTimeout"
                    ),
                    "timeout": True,
                    "pending_task_count": 1 if still_pending else 0,
                    "fatal": True,
                    **({"room_id": metadata["room_id"]} if metadata.get("room_id") else {}),
                }
                failures.append(failure)
                if still_pending:
                    self._quarantine_task(
                        task,
                        failure,
                        room_id=metadata.get("room_id"),
                    )
            if caller_cancelled:
                raise asyncio.CancelledError
        return failures

    async def _cancel_claimed_core_tasks(
        self,
        task_metadata: dict[asyncio.Future[Any], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Cancel each claimed Core run once, then supervise without recancelling."""
        current_loop = asyncio.get_running_loop()
        local: set[asyncio.Future[Any]] = set()
        failures: list[dict[str, Any]] = []
        for task, metadata in task_metadata.items():
            if task.done():
                _consume_task_result(task)
                continue
            try:
                owner_loop = task.get_loop()
            except Exception:  # noqa: BLE001
                owner_loop = current_loop
            if owner_loop is not current_loop:
                try:
                    owner_loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    pass
                failures.append({
                    "stage": "core_room_task",
                    "error_type": "ForeignLoopTaskPending",
                    "timeout": True,
                    "pending_task_count": 1,
                    "fatal": True,
                    "room_id": metadata.get("room_id"),
                })
                continue
            task.cancel()
            local.add(task)

        if not local:
            return failures
        timeout = max(
            float(task_metadata[task]["cleanup_budget_seconds"])
            for task in local
        )
        try:
            done, pending = await asyncio.wait(local, timeout=timeout)
        except asyncio.CancelledError:
            for task in local:
                if task.done():
                    _consume_task_result(task)
                    continue
                metadata = task_metadata[task]
                failure = {
                    "stage": "core_room_task",
                    "error_type": "ShutdownSupervisorCancelled",
                    "timeout": False,
                    "pending_task_count": 1,
                    "fatal": True,
                    "room_id": metadata.get("room_id"),
                }
                self._quarantine_task(
                    task,
                    failure,
                    room_id=metadata.get("room_id"),
                )
            raise

        for task in done:
            if task.cancelled():
                continue
            try:
                task.result()
            except asyncio.CancelledError:
                continue
            except Exception as err:  # noqa: BLE001 - keep task details private
                metadata = task_metadata[task]
                failures.append({
                    "stage": "core_room_task",
                    "error_type": type(err).__name__,
                    "timeout": False,
                    "pending_task_count": 0,
                    "fatal": True,
                    "room_id": metadata.get("room_id"),
                })
        for task in pending:
            metadata = task_metadata[task]
            failure = {
                "stage": "core_room_task",
                "error_type": "TaskIgnoredCancellation",
                "timeout": True,
                "pending_task_count": 1,
                "fatal": True,
                "room_id": metadata.get("room_id"),
            }
            failures.append(failure)
            self._quarantine_task(
                task,
                failure,
                room_id=metadata.get("room_id"),
            )
        return failures

    async def _await_shutdown_cleanup_tasks(
        self,
        task_metadata: dict[asyncio.Future[Any], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Await already-started cleanup tasks under one shared wall clock."""
        tasks = {task for task in task_metadata if not task.done()}
        failures: list[dict[str, Any]] = []
        for task in set(task_metadata) - tasks:
            _consume_task_result(task)
        if not tasks:
            return failures
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.cleanup_timeout_seconds,
            )
        except asyncio.CancelledError:
            pending, _interrupted = await _cancel_tasks_bounded(
                tasks,
                self.cancellation_grace_seconds,
            )
            for task in pending:
                metadata = task_metadata[task]
                failure = {
                    "stage": str(metadata.get("stage") or "cleanup_task"),
                    "error_type": "TaskIgnoredCancellation",
                    "timeout": False,
                    "pending_task_count": 1,
                    "fatal": True,
                }
                self._quarantine_task(task, failure, room_id=metadata.get("room_id"))
            raise
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                continue
            except Exception as err:  # noqa: BLE001
                metadata = task_metadata[task]
                failures.append({
                    "stage": str(metadata.get("stage") or "cleanup_task"),
                    "error_type": type(err).__name__,
                    "timeout": False,
                    "pending_task_count": 0,
                    "fatal": True,
                    **({"room_id": metadata["room_id"]} if metadata.get("room_id") else {}),
                })
        if pending:
            initially_pending = set(pending)
            pending, caller_cancelled = await _cancel_tasks_bounded(
                pending,
                self.cancellation_grace_seconds,
            )
            for task in initially_pending:
                metadata = task_metadata[task]
                still_pending = task in pending
                failure = {
                    "stage": str(metadata.get("stage") or "cleanup_task"),
                    "error_type": (
                        "TaskIgnoredCancellation" if still_pending else "CleanupTimeout"
                    ),
                    "timeout": True,
                    "pending_task_count": 1 if still_pending else 0,
                    "fatal": True,
                    **({"room_id": metadata["room_id"]} if metadata.get("room_id") else {}),
                }
                failures.append(failure)
                if still_pending:
                    self._quarantine_task(
                        task,
                        failure,
                        room_id=metadata.get("room_id"),
                    )
            if caller_cancelled:
                raise asyncio.CancelledError
        return failures

    async def _close_shutdown_clients(
        self,
        clients: list[tuple[Room, str, RoomClient]],
    ) -> list[dict[str, Any]]:
        current_loop = asyncio.get_running_loop()
        tasks: dict[asyncio.Future[Any], dict[str, Any]] = {}
        failures: list[dict[str, Any]] = []

        async def close_socket(connection: RoomClient) -> None:
            await connection.websocket.close(code=1001, reason="server shutdown")

        for room, cid, connection in clients:
            metadata = {
                "stage": "ws_socket_close",
                "room_id": room.id,
                "client_id": cid,
            }
            if connection.loop.is_closed():
                failures.append({
                    "stage": "ws_socket_close",
                    "error_type": "OwnerLoopClosed",
                    "timeout": False,
                    "pending_task_count": 0,
                    "fatal": True,
                    "room_id": room.id,
                })
                continue
            if connection.loop is current_loop:
                task = current_loop.create_task(
                    close_socket(connection),
                    name=f"ws-shutdown-close:{room.id}:{cid}",
                )
                connection.close_task = task
                tasks[task] = metadata
                continue
            try:
                concurrent_future = asyncio.run_coroutine_threadsafe(
                    close_socket(connection),
                    connection.loop,
                )
                tasks[asyncio.wrap_future(concurrent_future)] = metadata
            except RuntimeError:
                failures.append({
                    "stage": "ws_socket_close",
                    "error_type": "OwnerLoopClosed",
                    "timeout": False,
                    "pending_task_count": 0,
                    "fatal": True,
                    "room_id": room.id,
                })
        failures.extend(await self._await_shutdown_cleanup_tasks(tasks))
        return failures

    async def _record_shutdown_room_state(
        self,
        room: Room,
        failure: dict[str, Any] | None,
    ) -> None:
        if failure is not None:
            safe = self._record_cleanup_failure(failure, room_id=room.id)
            room.status = "failed"
            room.end_reason = "cleanup_failure"
            room.error = "room shutdown cleanup failed"
            await self._emit_room_event(room, {
                "type": "room_cleanup_failed",
                **safe,
            })
            await self._emit_room_event(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
        elif room.status == "running":
            # A test/integration may provide a room task that does not own the
            # standard _run_room wrapper. Shutdown still commits a terminal.
            room.status = "cancelled"
            room.end_reason = "shutdown"
            room.error = "room cancelled during manager shutdown"
            await self._broadcast_room_status(room)

    async def aclose(self) -> None:
        if self._closed:
            return
        # Flip readiness before waiting for an in-flight start operation to
        # finish registering its task. New rooms are rejected immediately.
        self._closing = True
        async with self._lifecycle_lock:
            if self._closed:
                self._closing = False
                return
            shutdown_failures: list[dict[str, Any]] = []
            first_error: BaseException | None = None
            try:
                task_metadata: dict[asyncio.Future[Any], dict[str, Any]] = {}
                claimed_core_tasks: dict[asyncio.Future[Any], dict[str, Any]] = {}
                for room in self.rooms.values():
                    if room.task is not None and not room.task.done():
                        prepared = room.prepared_run
                        if (
                            prepared is not None
                            and room.core_run_spec is not None
                        ):
                            claimed_core_tasks[room.task] = {
                                "stage": "core_room_task",
                                "room_id": room.id,
                                "cleanup_budget_seconds": (
                                    environment_cancellation_budget_seconds(
                                        room.core_run_spec
                                    )
                                ),
                            }
                        else:
                            task_metadata[room.task] = {
                                "stage": "room_task",
                                "room_id": room.id,
                            }
                with self._quarantined_tasks_lock:
                    for task, metadata in self._quarantined_tasks.items():
                        if not task.done():
                            task_metadata.setdefault(task, dict(metadata))

                room_task_failures = await self._cancel_claimed_core_tasks(
                    claimed_core_tasks
                )
                room_task_failures.extend(
                    await self._cancel_shutdown_tasks(task_metadata)
                )
                shutdown_failures.extend(room_task_failures)
                failure_by_room = {
                    str(item["room_id"]): item
                    for item in room_task_failures
                    if item.get("room_id")
                }
                for room in self.rooms.values():
                    prepared = room.prepared_run
                    task_finished = room.task is None or room.task.done()
                    if (
                        prepared is not None
                        and not getattr(prepared, "_claimed", False)
                        and task_finished
                    ):
                        prepared_failures = await self._close_unclaimed_core_resources(
                            room_id=room.id,
                            session=prepared.session,
                            decision_runtime=prepared.decision_runtime,
                        )
                        shutdown_failures.extend(prepared_failures)
                        if prepared_failures:
                            failure_by_room.setdefault(room.id, prepared_failures[0])
                        room.prepared_run = None
                for room in self.rooms.values():
                    await self._record_shutdown_room_state(
                        room,
                        failure_by_room.get(room.id),
                    )

                # Writer/handshake tasks are included even when disconnect was
                # initiated on another loop. Foreign-loop ownership is reported
                # rather than passed to asyncio.wait on the wrong loop.
                client_tasks: dict[asyncio.Future[Any], dict[str, Any]] = {}
                shutdown_clients: list[tuple[Room, str, RoomClient]] = []
                for room in self.rooms.values():
                    with room.delivery_lock:
                        clients = list(room.clients.items())
                        room.clients.clear()
                    for cid, connection in clients:
                        if not isinstance(connection, RoomClient):
                            continue
                        shutdown_clients.append((room, cid, connection))
                        for kind, task in (
                            ("ws_handshake", connection.handshake_task),
                            ("ws_writer", connection.writer_task),
                            ("ws_close", connection.close_task),
                        ):
                            if task is not None and not task.done():
                                client_tasks[task] = {
                                    "stage": kind,
                                    "room_id": room.id,
                                    "client_id": cid,
                                }
                writer_failures = await self._cancel_shutdown_tasks(client_tasks)
                writer_failures.extend(
                    await self._close_shutdown_clients(shutdown_clients)
                )
                shutdown_failures.extend(writer_failures)
                for failure in writer_failures:
                    safe = self._record_cleanup_failure(
                        failure,
                        room_id=str(failure.get("room_id") or "") or None,
                    )
                    room_id = str(safe.get("room_id") or "")
                    target_room = self.rooms.get(room_id)
                    if target_room is not None:
                        await self._emit_room_event(target_room, {
                            "type": "room_cleanup_failed",
                            **safe,
                        })

                for room in self.rooms.values():
                    try:
                        self._close_provider_scope(room)
                    except Exception as err:  # noqa: BLE001
                        failure = {
                            "stage": "provider_scope_close",
                            "error_type": type(err).__name__,
                            "timeout": False,
                            "pending_task_count": 0,
                            "fatal": True,
                            "room_id": room.id,
                        }
                        shutdown_failures.append(failure)
                        self._record_cleanup_failure(failure, room_id=room.id)

                try:
                    router_awaitable = self.router.aclose()
                except Exception as err:  # noqa: BLE001
                    router_failure = {
                        "stage": "router_close",
                        "error_type": type(err).__name__,
                        "timeout": False,
                        "pending_task_count": 0,
                        "fatal": True,
                    }
                    router_error: BaseException | None = err
                else:
                    router_failure, router_error = await self._bounded_cleanup_awaitable(
                        router_awaitable,
                        stage="router_close",
                    )
                if router_failure is None:
                    self._router_closed = True
                else:
                    shutdown_failures.append(router_failure)
                    self._record_cleanup_failure(router_failure)
                    self._router_close_failed = True
                    first_error = router_error

                if self.persistence is not None and not self._persistence_closed:
                    persistence_failure, persistence_error = (
                        await self._bounded_cleanup_awaitable(
                            asyncio.to_thread(self.persistence.close),
                            stage="persistence_close",
                            cancel_on_timeout=False,
                        )
                    )
                    self._persistence_closed = True
                    if persistence_failure is not None:
                        shutdown_failures.append(persistence_failure)
                        self._record_cleanup_failure(persistence_failure)
                        if first_error is None:
                            first_error = persistence_error

                # Room-task cancellation can discover a stubborn orchestrator
                # child after the initial ownership snapshot. Never let that
                # newly quarantined task disappear from the shutdown result.
                with self._quarantined_tasks_lock:
                    unresolved = [
                        dict(metadata)
                        for task, metadata in self._quarantined_tasks.items()
                        if not task.done()
                    ]
                known = {
                    (
                        str(item.get("stage") or ""),
                        str(item.get("room_id") or ""),
                        str(item.get("error_type") or ""),
                    )
                    for item in shutdown_failures
                    if int(item.get("pending_task_count") or 0) > 0
                }
                for metadata in unresolved:
                    key = (
                        str(metadata.get("stage") or ""),
                        str(metadata.get("room_id") or ""),
                        str(metadata.get("error_type") or ""),
                    )
                    if key not in known:
                        metadata.setdefault("fatal", True)
                        metadata["pending_task_count"] = max(
                            1,
                            int(metadata.get("pending_task_count") or 0),
                        )
                        shutdown_failures.append(metadata)
                        known.add(key)
            finally:
                if not self._router_closed:
                    self._router_close_failed = True
                self._closed = True
                self._closing = False
            if first_error is not None:
                raise first_error
            if shutdown_failures:
                raise RoomManagerCleanupError(shutdown_failures)


def _room_timeout_from_env() -> float | None:
    raw = os.getenv("WEREWOLF_ROOM_TIMEOUT", "900")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 900.0
    return value if value > 0 else None


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _bounded_duration(
    value: float,
    *,
    name: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a finite number") from err
    minimum_valid = parsed >= minimum if minimum_inclusive else parsed > minimum
    if not math.isfinite(parsed) or not minimum_valid or parsed > maximum:
        comparator = ">=" if minimum_inclusive else ">"
        raise ValueError(f"{name} must be finite, {comparator} {minimum}, and <= {maximum}")
    return parsed


async def _cancel_task_bounded(
    task: asyncio.Future[Any],
    grace_seconds: float,
) -> tuple[bool, bool]:
    pending, caller_cancelled = await _cancel_tasks_bounded({task}, grace_seconds)
    return not pending, caller_cancelled


async def _cancel_tasks_bounded(
    tasks: set[asyncio.Future[Any]] | list[asyncio.Future[Any]],
    grace_seconds: float,
) -> tuple[set[asyncio.Future[Any]], bool]:
    """Cancel twice within one shared grace while deferring repeat cancels."""
    pending = {task for task in tasks if not task.done()}
    for task in tasks:
        if task.done():
            _consume_task_result(task)
    if not pending:
        return set(), False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace_seconds
    caller_cancelled = False
    for task in pending:
        task.cancel()
    pending, interrupted = await _wait_tasks_until(
        pending,
        loop.time() + grace_seconds / 2,
    )
    caller_cancelled = caller_cancelled or interrupted
    if pending:
        for task in pending:
            task.cancel()
        pending, interrupted = await _wait_tasks_until(pending, deadline)
        caller_cancelled = caller_cancelled or interrupted
    return {task for task in pending if not task.done()}, caller_cancelled


async def _wait_tasks_until(
    tasks: set[asyncio.Future[Any]],
    deadline: float,
) -> tuple[set[asyncio.Future[Any]], bool]:
    pending = set(tasks)
    caller_cancelled = False
    while pending:
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        if remaining <= 0:
            break
        try:
            done, pending = await asyncio.wait(pending, timeout=remaining)
        except asyncio.CancelledError:
            caller_cancelled = True
            continue
        for task in done:
            _consume_task_result(task)
    return pending, caller_cancelled


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    if not task.done():
        return
    try:
        task.result()
    except BaseException:
        return


def _set_task_name(task: asyncio.Future[Any], name: str) -> None:
    setter = getattr(task, "set_name", None)
    if callable(setter):
        setter(name)


def _task_name(task: asyncio.Future[Any]) -> str:
    getter = getattr(task, "get_name", None)
    if callable(getter):
        return str(getter())
    return type(task).__name__


def _room_base_seed(seed: int | None) -> int:
    """Return a stable per-room base seed without using provider/model secrets."""
    if seed is not None:
        return int(seed)
    return secrets.randbits(63)


def _public_room_error(err: BaseException) -> str:
    """Public room-level error message; raw provider details stay in logs."""
    return f"{type(err).__name__} during game loop"
