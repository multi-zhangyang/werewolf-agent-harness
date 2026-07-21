"""Werewolf plugin configuration boundary tests."""
from __future__ import annotations

from typing import Any, cast

from pydantic import ValidationError
import pytest

from src.agent.actor import AgentActor
from src.environments.werewolf import WerewolfEnvironmentConfig, WerewolfEnvironmentPlugin
from src.environments.werewolf.plugin import _WerewolfSession
from src.game.roles import CLASSIC_RULESET_ID, Role
from src.game.rules import RulesEngine
from src.game.state import new_game
from src.harness.core_spec import ActorSpec
from src.harness.decision_runtime import DecisionRuntime
from src.harness.environment import AgentBindingError, AgentRegistry, EnvironmentRunContext
from src.harness.spec import ModelConfigManifest
from src.llm.models import ModelConfig


def _raw_config(**updates: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "role_deck": [
            "werewolf", "werewolf", "seer", "villager", "villager", "villager",
        ],
        "turn_policy": "fixed_round_robin",
        "max_speak_rounds": 1,
        "decision_timeout_seconds": 2,
        "phase_deadline_seconds": 0,
    }
    config.update(updates)
    return config


def _plugin() -> WerewolfEnvironmentPlugin:
    return WerewolfEnvironmentPlugin()


def _model_config(*, model: str = "declared-model") -> ModelConfig:
    return ModelConfig(
        provider="openai",
        model=model,
        api_base="https://models.example.invalid/v1",
        api_key="test-only-key",
    )


def _actor_spec(
    config: ModelConfig,
    *,
    human_actor_ids: list[str] | None = None,
) -> ActorSpec:
    return ActorSpec(
        default_model=ModelConfigManifest.from_config(config).model_dump(mode="json"),
        human_actor_ids=human_actor_ids or [],
    )


def _agents_for_state(
    state: Any,
    *,
    config: ModelConfig,
    human_seats: set[int] | None = None,
) -> dict[str, AgentActor]:
    human_seats = human_seats or set()
    return {
        f"seat:{player.seat}": AgentActor(
            seat=player.seat,
            name=player.name,
            role=Role(player.role),
            model_config=config.model_copy(),
            router=cast(Any, object()),
            is_human=player.seat in human_seats,
        )
        for player in state.players
    }


def _swap_first_distinct_roles(state: Any) -> None:
    for left, first in enumerate(state.players):
        for second in state.players[left + 1:]:
            if first.role != second.role:
                first.role, second.role = second.role, first.role
                return
    raise AssertionError("test deck must contain at least two distinct roles")


def _context(
    *,
    actor_spec: ActorSpec,
    resolve_agent: Any,
    run_id: str = "plugin-binding-run",
    config: WerewolfEnvironmentConfig | None = None,
) -> EnvironmentRunContext:
    async def emit_event(_payload: dict[str, Any]) -> None:
        return None

    return EnvironmentRunContext(
        run_id=run_id,
        config=config or WerewolfEnvironmentConfig.model_validate(_raw_config()),
        seeds={"role": 11, "actor": 22, "orchestrator": 33},
        actor_spec=actor_spec,
        decision_runtime=DecisionRuntime(on_trace=lambda _payload: None),
        emit_event=emit_event,
        emit_trace=lambda _payload: None,
        resolve_agent=resolve_agent,
        metadata={},
    )


def test_plugin_resolves_legacy_config_to_explicit_classic_v1():
    resolved = _plugin().resolve_config(_raw_config(), {})

    assert isinstance(resolved, WerewolfEnvironmentConfig)
    assert resolved.ruleset_id == CLASSIC_RULESET_ID
    assert resolved.model_dump()["ruleset_id"] == "classic.v1"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"ruleset_id": "classic.v2"}, "unsupported Werewolf ruleset"),
        (
            {"role_deck": ["werewolf", "villager", "villager", "villager", "villager"]},
            "size must match player count",
        ),
        ({"role_deck": ["villager"] * 6}, "at least one werewolf"),
        (
            {
                "role_deck": [
                    "werewolf", "seer", "seer", "villager", "villager", "villager",
                ],
            },
            "at most one of each power role",
        ),
        (
            {
                "role_deck": [
                    "werewolf", "villager", "villager", "villager", "villager", "cupid",
                ],
            },
            "unknown role",
        ),
    ],
)
def test_plugin_rejects_unknown_ruleset_and_invalid_decks_before_session(updates, message):
    with pytest.raises(ValidationError, match=message):
        _plugin().resolve_config(_raw_config(**updates), {})


@pytest.mark.asyncio
async def test_plugin_resolves_each_declared_seat_through_the_run_context():
    config = _model_config()
    agents: dict[str, AgentActor] = {}
    calls: list[str] = []

    def prepare_agents(state: Any) -> None:
        agents.update(_agents_for_state(state, config=config, human_seats={1}))

    def resolve_agent(actor_id: str) -> AgentActor:
        calls.append(actor_id)
        return agents[actor_id]

    plugin = WerewolfEnvironmentPlugin(on_state_ready=prepare_agents)
    session = await plugin.create_session(
        _context(
            actor_spec=_actor_spec(config, human_actor_ids=["seat:1"]),
            resolve_agent=resolve_agent,
        )
    )

    assert calls == [f"seat:{seat}" for seat in range(1, 7)]
    assert {
        f"seat:{actor.seat}": actor
        for actor in session.orchestrator.actors.values()
    } == agents
    assert agents["seat:1"].is_human is True
    assert all(not agents[f"seat:{seat}"].is_human for seat in range(2, 7))


@pytest.mark.asyncio
async def test_plugin_consumes_one_room_owned_state_without_redealing() -> None:
    run_id = "interactive-room-run"
    config = WerewolfEnvironmentConfig.model_validate(
        _raw_config(
            decision_timeouts={"night": 1.25},
            phase_deadlines={"night": 3.5},
        )
    )
    state = new_game(config.player_names, game_id=run_id)
    RulesEngine.deal_roles(
        state,
        deck=[Role(value) for value in config.role_deck],
        seed=11,
        ruleset_id=config.ruleset_id,
    )
    role_snapshot = [player.role for player in state.players]
    event_ids = [event.id for event in state.events]
    model_config = _model_config()
    agents = _agents_for_state(state, config=model_config, human_seats={1})
    registry = AgentRegistry(agents.__getitem__)
    ready: list[_WerewolfSession] = []
    plugin = WerewolfEnvironmentPlugin(
        room_state=state,
        on_session_ready=ready.append,
    )
    context = _context(
        run_id=run_id,
        config=config,
        actor_spec=_actor_spec(model_config, human_actor_ids=["seat:1"]),
        resolve_agent=registry.resolve,
    )

    session = await plugin.create_session(context)

    assert isinstance(session, _WerewolfSession)
    assert session.state is state
    assert session.orchestrator.state is state
    assert session.orchestrator._decision_runtime is context.decision_runtime
    assert session.orchestrator.decision_timeouts["night"] == 1.25
    assert session.orchestrator.phase_deadlines["night"] == 3.5
    assert [player.role for player in state.players] == role_snapshot
    assert [event.id for event in state.events] == event_ids
    assert ready == [session]
    with pytest.raises(RuntimeError, match="only create one session"):
        await plugin.create_session(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: setattr(state, "id", "different-run"), "id does not match"),
        (lambda state: setattr(state, "phase", "setup"), "not a fresh dealt game"),
        (lambda state: setattr(state.players[0], "name", "Different"), "names do not match"),
        (lambda state: setattr(state.players[0], "role", Role.GUARD), "role deck does not match"),
        (_swap_first_distinct_roles, "role assignment does not match seed"),
    ],
)
async def test_plugin_rejects_inconsistent_room_owned_state(mutate, message) -> None:
    run_id = "interactive-invalid-state"
    config = WerewolfEnvironmentConfig.model_validate(_raw_config())
    state = new_game(config.player_names, game_id=run_id)
    RulesEngine.deal_roles(
        state,
        deck=[Role(value) for value in config.role_deck],
        seed=11,
        ruleset_id=config.ruleset_id,
    )
    mutate(state)
    model_config = _model_config()
    agents = _agents_for_state(state, config=model_config)
    plugin = WerewolfEnvironmentPlugin(room_state=state)

    with pytest.raises(ValueError, match=message):
        await plugin.create_session(
            _context(
                run_id=run_id,
                config=config,
                actor_spec=_actor_spec(model_config),
                resolve_agent=AgentRegistry(agents.__getitem__).resolve,
            )
        )


@pytest.mark.asyncio
async def test_plugin_rejects_private_state_shared_between_distinct_seat_agents() -> None:
    run_id = "interactive-private-state-alias"
    config = WerewolfEnvironmentConfig.model_validate(_raw_config())
    state = new_game(config.player_names, game_id=run_id)
    RulesEngine.deal_roles(
        state,
        deck=[Role(value) for value in config.role_deck],
        seed=11,
        ruleset_id=config.ruleset_id,
    )
    model_config = _model_config()
    agents = _agents_for_state(state, config=model_config)
    agents["seat:2"].memory = agents["seat:1"].memory
    plugin = WerewolfEnvironmentPlugin(room_state=state)

    with pytest.raises(ValueError, match="share private execution state"):
        await plugin.create_session(
            _context(
                run_id=run_id,
                config=config,
                actor_spec=_actor_spec(model_config),
                resolve_agent=AgentRegistry(agents.__getitem__).resolve,
            )
        )


@pytest.mark.asyncio
async def test_plugin_rejects_actor_spec_without_execution_binding():
    calls: list[str] = []
    plugin = WerewolfEnvironmentPlugin(
        on_state_ready=lambda _state: pytest.fail("must reject before preparing agents")
    )

    with pytest.raises(ValueError, match="no execution binding"):
        await plugin.create_session(
            _context(
                actor_spec=ActorSpec(),
                resolve_agent=lambda actor_id: calls.append(actor_id),
            )
        )

    assert calls == []


@pytest.mark.asyncio
async def test_plugin_rejects_resolved_model_that_disagrees_with_actor_spec():
    declared = _model_config(model="declared-model")
    agents: dict[str, AgentActor] = {}

    def prepare_agents(state: Any) -> None:
        agents.update(
            _agents_for_state(
                state,
                config=_model_config(model="different-runtime-model"),
            )
        )

    plugin = WerewolfEnvironmentPlugin(on_state_ready=prepare_agents)
    with pytest.raises(ValueError, match="does not match ActorSpec for seat:1"):
        await plugin.create_session(
            _context(
                actor_spec=_actor_spec(declared),
                resolve_agent=agents.__getitem__,
            )
        )


@pytest.mark.asyncio
async def test_plugin_registry_rejects_one_agent_object_reused_across_seats():
    config = _model_config()
    shared: AgentActor | None = None

    def prepare_agent(state: Any) -> None:
        nonlocal shared
        shared = _agents_for_state(state, config=config)["seat:1"]

    def resolve_shared(_actor_id: str) -> AgentActor:
        assert shared is not None
        return shared

    plugin = WerewolfEnvironmentPlugin(on_state_ready=prepare_agent)
    registry = AgentRegistry(resolve_shared)
    with pytest.raises(AgentBindingError, match="multiple actors"):
        await plugin.create_session(
            _context(
                actor_spec=_actor_spec(config),
                resolve_agent=registry.resolve,
            )
        )


@pytest.mark.asyncio
async def test_werewolf_session_rejects_orchestrator_early_return_as_completed():
    state = new_game(["A", "B", "C", "D", "E", "F"])

    class EarlyReturnOrchestrator:
        termination_status = "running"
        termination_reason = None

        async def run(self):
            return state

    session = _WerewolfSession(
        state=state,
        orchestrator=cast(Any, EarlyReturnOrchestrator()),
        analysis={},
    )

    with pytest.raises(RuntimeError, match="non-terminal status 'running'"):
        await session.run()
