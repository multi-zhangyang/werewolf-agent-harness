"""Production AgentActor integration with the provider-neutral tool loop."""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
from typing import Any

import pytest

from src.agent.actor import AgentActor
from src.agent.information import build_observation
from src.agent.memory import AgentMemory
from src.agent.schemas import AgentAction, AgentObservation
from src.agent.session import ToolExecutionContext
from src.agent.werewolf_tools import build_werewolf_tool_registry
from src.game.roles import Role
from src.game.models import NightActionType, Phase
from src.game.orchestrator import GameOrchestratorV2, build_actors
from src.game.state import new_game
from src.game.rules import RulesEngine
from src.harness.agent_protocol import ActionRequest, LegalAction
from src.harness.agents import validate_decision_against_legal_actions
from src.llm.models import ModelConfig
from src.llm.router import LLMResponseError


class ScriptedToolRouter:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = list(calls)
        self.seen: list[list[dict[str, Any]]] = []
        self.message_batches: list[list[dict[str, Any]]] = []
        self.selected: list[str] = []

    async def complete_tools(self, messages, config, tools, **kwargs):
        self.seen.append(tools)
        self.message_batches.append([dict(item) for item in messages])
        item = self.calls.pop(0)
        self.selected.append(str(item["tool_calls"][0].get("name")))
        return item


@pytest.mark.asyncio
async def test_get_commitments_exposes_only_this_seats_accepted_vote_history() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.WEREWOLF,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    actor.memory = AgentMemory(seat=actor.seat, role=actor.role.value, max_observations=1)
    request = _request(actor)
    observation = AgentObservation.model_validate(request.observation)
    actor.observe_event(
        1,
        "voting",
        "vote",
        "1号投了3号",
        voter_seat=1,
        target_seat=3,
        pk=False,
    )
    actor.observe_event(
        1,
        "voting",
        "vote",
        "2号投了1号",
        voter_seat=2,
        target_seat=1,
        pk=False,
    )
    actor.observe_event(
        2,
        "voting",
        "vote",
        "1号投了4号",
        voter_seat=1,
        target_seat=4,
        pk=True,
    )
    actor.record_public_commitment(
        day=2,
        phase="day",
        kind="speech",
        text="我会继续追4号。",
    )

    registry = build_werewolf_tool_registry(actor, request, observation)
    context = ToolExecutionContext(
        request=request,
        seat=actor.seat,
        role=actor.role.value,
        step=1,
        state_version=0,
    )
    result = await registry.execute("commitments-1", "get_commitments", {}, context)
    beliefs = await registry.execute("beliefs-1", "get_beliefs", {}, context)

    assert result.ok
    assert beliefs.ok
    assert "commitments" not in beliefs.output
    assert result.output["public_commitments"][0]["text"] == "我会继续追4号。"
    assert result.output["public_vote_history"] == [
        {"day": 1, "phase": "voting", "target_seat": 3, "pk": False},
        {"day": 2, "phase": "voting", "target_seat": 4, "pk": True},
    ]


@pytest.mark.asyncio
async def test_read_public_votes_filters_the_bounded_environment_ledger() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    actor.memory = AgentMemory(
        seat=actor.seat,
        role=actor.role.value,
        max_observations=1,
        max_public_votes=3,
    )
    request = _request(actor)
    observation = AgentObservation.model_validate(request.observation)
    for day, voter, target, pk in (
        (1, 1, 3, False),
        (1, 2, 3, True),
        (2, 2, 4, False),
        (2, 3, 3, False),
    ):
        actor.observe_event(
            day,
            "voting",
            "vote",
            f"{voter}号投了{target}号",
            voter_seat=voter,
            target_seat=target,
            pk=pk,
        )

    registry = build_werewolf_tool_registry(actor, request, observation)
    definition = next(
        item["function"]
        for item in registry.definitions()
        if item["function"]["name"] == "read_public_votes"
    )
    assert set(definition["parameters"]["properties"]) == {
        "limit",
        "seat",
        "target",
        "pk",
    }
    assert definition["parameters"]["additionalProperties"] is False
    context = ToolExecutionContext(
        request=request,
        seat=actor.seat,
        role=actor.role.value,
        step=1,
        state_version=0,
    )

    latest = await registry.execute("votes-latest", "read_public_votes", {"limit": 2}, context)
    by_seat = await registry.execute("votes-seat", "read_public_votes", {"seat": 2}, context)
    by_target_pk = await registry.execute(
        "votes-target-pk",
        "read_public_votes",
        {"target": 3, "pk": False},
        context,
    )
    combined = await registry.execute(
        "votes-combined",
        "read_public_votes",
        {"seat": 2, "target": 3, "pk": True},
        context,
    )

    assert latest.ok and [item["voter_seat"] for item in latest.output["votes"]] == [2, 3]
    assert by_seat.ok and [item["target_seat"] for item in by_seat.output["votes"]] == [3, 4]
    assert by_target_pk.ok and by_target_pk.output["votes"] == [{
        "day": 2,
        "phase": "voting",
        "voter_seat": 3,
        "target_seat": 3,
        "pk": False,
    }]
    assert combined.ok and combined.output["votes"] == [{
        "day": 1,
        "phase": "voting",
        "voter_seat": 2,
        "target_seat": 3,
        "pk": True,
    }]
    assert latest.output["ledger"]["total_count"] == 4
    assert latest.output["ledger"]["retained_count"] == 3
    assert latest.output["ledger"]["archived_count"] == 1
    assert len(latest.output["ledger"]["archived_digest"]) == 64


@pytest.mark.asyncio
async def test_read_public_events_keeps_last_words_after_unrelated_memory_churn() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    actor.memory = AgentMemory(
        seat=actor.seat,
        role=actor.role.value,
        max_observations=3,
    )
    exact_text = "2号遗言:不要把我的公开说法当成环境真值。"
    actor.observe_event(
        1,
        "last_words",
        "last_words",
        exact_text,
        speaker_seat=2,
    )
    for index in range(8):
        actor.observe_event(
            1,
            "day",
            "agent_private_note",
            f"private-note-{index}",
        )
    assert any(item.kind == "last_words" for item in actor.memory.observations)

    request = _request(actor)
    observation = AgentObservation.model_validate(request.observation)
    observation.public_events = [
        {
            "id": f"noise-{index}",
            "phase": "day",
            "day": 1,
            "type": "noise",
            "message": f"newer-public-event-{index}",
            "visibility": "public",
            "payload": {},
        }
        for index in range(45)
    ]
    registry = build_werewolf_tool_registry(actor, request, observation)
    context = ToolExecutionContext(
        request=request,
        seat=actor.seat,
        role=actor.role.value,
        step=1,
        state_version=0,
    )

    result = await registry.execute(
        "public-events-last-words",
        "read_public_events",
        {"limit": 24},
        context,
    )
    default_result = await registry.execute(
        "public-events-default",
        "read_public_events",
        {},
        context,
    )
    too_large = await registry.execute(
        "public-events-too-large",
        "read_public_events",
        {"limit": 25},
        context,
    )

    assert result.ok
    assert default_result.ok and len(default_result.output["events"]) == 12
    assert not too_large.ok and too_large.error_code == "invalid_arguments"
    assert all(item["id"] != "noise-0" for item in result.output["events"])
    assert result.output["memory_window"] == [{
        "day": 1,
        "phase": "last_words",
        "kind": "last_words",
        "text": exact_text,
    }]
    definition = next(
        item["function"]
        for item in registry.definitions()
        if item["function"]["name"] == "read_public_events"
    )
    assert definition["parameters"]["properties"]["limit"]["maximum"] == 24


@pytest.mark.asyncio
async def test_read_public_events_deduplicates_current_speech_and_last_words() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    actor.observe_event(1, "day", "speech", "2号说:当前发言", speaker_seat=2)
    actor.observe_event(1, "last_words", "last_words", "3号遗言:当前遗言", speaker_seat=3)
    actor.observe_event(0, "last_words", "last_words", "4号遗言:历史遗言", speaker_seat=4)
    actor.observe_event(1, "voting", "vote", "2号投了3号", voter_seat=2, target_seat=3)
    request = _request(actor)
    observation = AgentObservation.model_validate(request.observation)
    observation.today_speeches = [{"seat": 2, "day": 1, "text": "当前发言"}]
    observation.public_events = [{
        "id": "last-words-current",
        "phase": "last_words",
        "day": 1,
        "type": "last_words",
        "message": "C(3号)的遗言:当前遗言",
        "visibility": "public",
        "payload": {"text": "当前遗言"},
    }]
    registry = build_werewolf_tool_registry(actor, request, observation)
    context = ToolExecutionContext(
        request=request,
        seat=actor.seat,
        role=actor.role.value,
        step=1,
        state_version=0,
    )

    result = await registry.execute("public-events-dedup", "read_public_events", {}, context)

    assert result.ok
    assert result.output["memory_window"] == [{
        "day": 0,
        "phase": "last_words",
        "kind": "last_words",
        "text": "4号遗言:历史遗言",
    }]


@pytest.mark.asyncio
async def test_read_turn_context_returns_one_complete_seat_private_snapshot() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    other_actor = AgentActor(
        seat=2,
        name="B",
        role=Role.WEREWOLF,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    other_actor.private_state.set_plan(
        selected_plan="OTHER_SEAT_PRIVATE_PLAN_MUST_NOT_LEAK",
        candidate_plans=["other plan one", "other plan two"],
    )
    base = _request(actor)
    observation = AgentObservation.model_validate(base.observation)
    observation.public_events = [{
        "id": "public-1",
        "phase": "day",
        "day": 1,
        "type": "speech",
        "message": "2号公开质疑3号。",
        "visibility": "public",
        "payload": {
            "speaker_seat": 2,
            "text": "我质疑3号。",
            "team": "NESTED_PRIVATE_TEAM_MUST_NOT_LEAK",
            "private_context": {"note": "NESTED_PRIVATE_CONTEXT_MUST_NOT_LEAK"},
            "claim": {"role": "seer", "checked_seat": 3, "result": "wolf"},
        },
    }]
    observation.today_speeches = [{"seat": 2, "day": 1, "text": "我质疑3号。"}]
    observation.private_events = [{
        "type": "private_notice",
        "payload": {"text": "OWNER_ONLY_PRIVATE_EVENT"},
    }]
    request = base.model_copy(
        update={
            "observation": observation.model_dump(mode="json"),
            "private_context": {"owner_note": "OWNER_ONLY_PRIVATE_CONTEXT"},
        },
        deep=True,
    )
    observation = AgentObservation.model_validate(request.observation)
    actor.observe_event(
        1,
        "voting",
        "vote",
        "1号投了3号",
        voter_seat=1,
        target_seat=3,
        pk=False,
    )
    actor.record_claim(2, 1, {"role": "seer", "checked_seat": 3, "result": "wolf"})
    actor.record_public_commitment(
        day=1,
        phase="day",
        kind="speech",
        text="我会继续审计3号。",
    )

    registry = build_werewolf_tool_registry(actor, request, observation)
    definition = next(
        item["function"]
        for item in registry.definitions()
        if item["function"]["name"] == "read_turn_context"
    )
    assert definition["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    result = await registry.execute(
        "turn-context-1",
        "read_turn_context",
        {},
        ToolExecutionContext(
            request=request,
            seat=actor.seat,
            role=actor.role.value,
            step=1,
            state_version=0,
        ),
    )

    assert result.ok
    assert result.output["legal"]["requested_action"] == "speak"
    assert result.output["legal"]["visible_seats"] == [1, 2, 3, 4, 5, 6]
    assert result.output["private_facts"]["private_context"]["owner_note"] == "OWNER_ONLY_PRIVATE_CONTEXT"
    assert result.output["private_facts"]["private_events"][0]["payload"]["text"] == "OWNER_ONLY_PRIVATE_EVENT"
    assert result.output["public_context"]["events"][0]["message"] == "2号公开质疑3号。"
    assert result.output["public_context"]["votes"][0]["target_seat"] == 3
    assert result.output["subjective_state"]["owner_seat"] == 1
    assert result.output["own_commitments"]["public_commitments"][0]["text"] == "我会继续审计3号。"
    assert result.output["public_claim_history"]["2"][0]["role"] == "seer"
    assert result.output["public_context"]["events"][0]["payload"]["claim"]["role"] == "seer"
    serialized = json.dumps(result.output, ensure_ascii=False, sort_keys=True)
    assert "OTHER_SEAT_PRIVATE_PLAN_MUST_NOT_LEAK" not in serialized
    assert "NESTED_PRIVATE_TEAM_MUST_NOT_LEAK" not in serialized
    assert "NESTED_PRIVATE_CONTEXT_MUST_NOT_LEAK" not in serialized


@pytest.mark.asyncio
async def test_read_turn_context_bounds_pathological_text_without_opaque_model_result() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    base = _request(actor)
    observation = AgentObservation.model_validate(base.observation)
    long_public_prefix = "PUBLIC_LONG_TEXT_REMAINS_IDENTIFIABLE_"
    observation.public_events = [
        {
            "id": f"long-{index}",
            "phase": "day",
            "day": 1,
            "type": "speech",
            "message": long_public_prefix + str(index) + ("x" * 6_000),
            "visibility": "public",
            "payload": {},
        }
        for index in range(20)
    ]
    request = base.model_copy(
        update={
            "observation": observation.model_dump(mode="json"),
            "private_context": {
                f"oversized_{index}": f"PRIVATE_VALUE_{index}_" + ("y" * 8_000)
                for index in range(40)
            },
        },
        deep=True,
    )
    observation = AgentObservation.model_validate(request.observation)
    registry = build_werewolf_tool_registry(actor, request, observation)
    result = await registry.execute(
        "turn-context-large",
        "read_turn_context",
        {},
        ToolExecutionContext(
            request=request,
            seat=actor.seat,
            role=actor.role.value,
            step=1,
            state_version=0,
        ),
    )

    assert result.ok
    serialized = json.dumps(
        result.output,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert len(serialized) <= 14_000
    assert result.output["snapshot_truncated"] is True
    assert result.output["source"] == "environment_and_seat_private_turn_snapshot"
    assert result.output["legal"]["legal_actions"][0]["action"] == "speak"
    assert long_public_prefix in serialized

    model_observation = json.loads(result.model_message()["content"])
    assert model_observation["result"]["source"] == "environment_and_seat_private_turn_snapshot"
    assert model_observation["result"].get("type") != "trace_value_truncated"


@pytest.mark.asyncio
async def test_read_turn_context_hard_bounds_untrusted_legal_metadata() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    base = _request(actor)
    oversized_metadata = {f"field_{index}": "z" * 1_000 for index in range(40)}
    legal_actions = [LegalAction(action="speak", can_skip=True, metadata=oversized_metadata)]
    legal_actions.extend(
        LegalAction(action=f"extra_{index}", metadata=oversized_metadata)
        for index in range(11)
    )
    request = base.model_copy(update={"legal_actions": legal_actions}, deep=True)
    observation = AgentObservation.model_validate(request.observation)
    registry = build_werewolf_tool_registry(actor, request, observation)
    result = await registry.execute(
        "turn-context-legal-large",
        "read_turn_context",
        {},
        ToolExecutionContext(
            request=request,
            seat=actor.seat,
            role=actor.role.value,
            step=1,
            state_version=0,
        ),
    )

    assert result.ok
    encoded = json.dumps(result.output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert len(encoded) <= 14_000
    assert result.output["legal"]["legal_actions"][0]["action"] == "speak"
    assert result.output["legal"]["legal_actions"][0]["target_seats"] == []
    assert result.output["legal"]["legal_actions"][0]["metadata"]["type"] == "section_truncated"
    assert result.model_message()["content"].find("trace_value_truncated") == -1


@pytest.mark.asyncio
async def test_read_turn_context_redacts_long_secret_before_preview_and_digest() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    base = _request(actor)
    secret = "sk-" + "S" * 40
    secret_text = (secret + " ") * 20
    sensitive_key_value = "short-sensitive-value-not-matched-by-pattern"
    request = base.model_copy(
        update={"private_context": {"note": secret_text, "api_key": sensitive_key_value}},
        deep=True,
    )
    observation = AgentObservation.model_validate(request.observation)
    registry = build_werewolf_tool_registry(actor, request, observation)
    result = await registry.execute(
        "turn-context-secret",
        "read_turn_context",
        {},
        ToolExecutionContext(
            request=request,
            seat=actor.seat,
            role=actor.role.value,
            step=1,
            state_version=0,
        ),
    )

    assert result.ok
    message = result.model_message()["content"]
    assert secret not in message
    assert hashlib.sha256(secret_text.encode("utf-8")).hexdigest() not in message
    assert sensitive_key_value not in message


def _request(actor: AgentActor) -> ActionRequest:
    state = new_game(["A", "B", "C", "D", "E", "F"], game_id="tool-actor")
    roles = [Role.VILLAGER, Role.WEREWOLF, Role.VILLAGER, Role.VILLAGER, Role.SEER, Role.DOCTOR]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    player_id = next(player.id for player in state.players if player.seat == actor.seat)
    observation = build_observation(
        state,
        player_id,
        rng=random.Random(3),
        available_actions=["speak"],
    )
    return ActionRequest(
        request_id="tool-actor-request",
        run_id=state.id,
        seat=actor.seat,
        phase="day",
        day=1,
        action_kind="speak",
        observation=observation.model_dump(),
        legal_actions=[LegalAction(action="speak", can_skip=True)],
    )


def _exact_text_request(actor: AgentActor, *, action_kind: str) -> ActionRequest:
    state = new_game(["A", "B", "C", "D", "E", "F"], game_id=f"tool-{action_kind}")
    roles = [Role.VILLAGER, Role.WEREWOLF, Role.VILLAGER, Role.VILLAGER, Role.SEER, Role.DOCTOR]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    state.phase = Phase.NIGHT if action_kind == "wolf_council" else Phase.DAY
    state.day = 1
    player_id = next(player.id for player in state.players if player.seat == actor.seat)
    targets = [1] if action_kind == "wolf_council" else []
    observation = build_observation(
        state,
        player_id,
        rng=random.Random(5),
        available_actions=[action_kind],
        candidate_targets=targets,
    )
    return ActionRequest(
        request_id=f"tool-{action_kind}-request",
        run_id=state.id,
        seat=actor.seat,
        phase=action_kind,
        day=1,
        action_kind=action_kind,
        observation=observation.model_dump(),
        legal_actions=[LegalAction(
            action=action_kind,
            target_seats=targets,
            target_required=bool(targets),
            can_skip=action_kind == "last_words",
        )],
        private_context={"reason": "exiled"} if action_kind == "last_words" else {},
    )


@pytest.mark.asyncio
async def test_tool_registry_binds_seat_arguments_to_the_visible_roster() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    request = _request(actor)
    observation = AgentObservation.model_validate(request.observation)
    observation.alive_seats = [1, 2, 4, 6]
    registry = build_werewolf_tool_registry(actor, request, observation)
    definitions = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in registry.definitions()
    }
    visible = [1, 2, 3, 4, 5, 6]
    opponents = [2, 3, 4, 5, 6]
    alive_opponents = [2, 4, 6]

    votes = definitions["read_public_votes"]["properties"]
    assert votes["seat"]["enum"] == visible
    assert votes["target"]["enum"] == visible
    assert definitions["analyze_claim_consistency"]["properties"]["seat"]["enum"] == visible
    assert definitions["update_belief"]["properties"]["seat"]["enum"] == opponents
    batch_belief = definitions["update_beliefs"]["properties"]["beliefs"]["items"]
    assert batch_belief["properties"]["seat"]["enum"] == opponents
    speak = definitions["speak"]["properties"]
    claim = speak["claim"]["anyOf"][0]["properties"]
    assert claim["checked_seat"]["enum"] == opponents
    assert speak["reply_to"]["anyOf"][0]["enum"] == opponents
    assert speak["accuses"]["items"]["enum"] == alive_opponents

    context = ToolExecutionContext(
        request=request,
        seat=actor.seat,
        role=actor.role.value,
        step=1,
        state_version=0,
    )
    legal = await registry.execute("legal-roster", "get_legal_actions", {}, context)
    assert legal.ok
    assert legal.output["visible_seats"] == visible
    assert legal.output["alive_seats"] == [1, 2, 4, 6]

    # Claims can report an earlier check on a dead player and replies can
    # address a dead player's prior speech or last words.  Current accusations
    # remain scoped to living opponents, including when a handler is invoked
    # behind the JSON-schema boundary.
    speak_spec = registry.get("speak")
    assert speak_spec is not None
    speech_decision = await speak_spec.handler(context, {
        "speech": "回应遗言，并把当前票型指向4号。",
        "bid": 1,
        "claim": {"role": "seer", "checked_seat": 3, "result": "village"},
        "reply_to": 5,
        "accuses": [3, 4, 5, 4],
    })
    assert speech_decision.claim == {
        "role": "seer",
        "checked_seat": 3,
        "result": "village",
    }
    assert speech_decision.reply_to == 5
    assert speech_decision.accuses == [4]

    solo_observation = observation.model_copy(deep=True)
    solo_observation.seats = [
        item for item in solo_observation.seats
        if item.get("seat") == actor.seat
    ]
    solo_observation.alive_seats = [actor.seat]
    solo_registry = build_werewolf_tool_registry(actor, request, solo_observation)
    solo_speak = next(
        item["function"]["parameters"]["properties"]
        for item in solo_registry.definitions()
        if item["function"]["name"] == "speak"
    )
    assert solo_speak["reply_to"]["anyOf"][0]["enum"] == []
    assert solo_speak["accuses"]["items"]["enum"] == []

    invalid_calls = [
        ("bad-vote-filter", "read_public_votes", {"seat": 9}),
        ("bad-claim-seat", "analyze_claim_consistency", {"seat": 9}),
        ("bad-self-belief", "update_belief", {
            "seat": 1,
            "wolf_probability": 0.5,
            "confidence": 0.5,
            "evidence": ["self"],
        }),
        ("bad-missing-belief", "update_beliefs", {"beliefs": [{
            "seat": 9,
            "wolf_probability": 0.5,
            "confidence": 0.5,
            "evidence": ["missing"],
        }]}),
        ("bad-speech-seat", "speak", {
            "speech": "测试非法座位元数据",
            "bid": 1,
            "reply_to": 9,
        }),
        ("dead-accusation", "speak", {
            "speech": "不能把已出局座位作为当前指控目标。",
            "bid": 1,
            "accuses": [3],
        }),
    ]
    for call_id, tool, arguments in invalid_calls:
        result = await registry.execute(call_id, tool, arguments, context)
        assert not result.ok and result.error_code == "invalid_arguments"


@pytest.mark.asyncio
async def test_terminal_target_schema_matches_the_exact_legal_action_targets() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    base = _request(actor)
    observation_data = dict(base.observation)
    observation_data["available_actions"] = ["vote"]
    observation_data["candidate_targets"] = [2, 4]
    vote_request = base.model_copy(update={
        "phase": "voting",
        "action_kind": "vote",
        "observation": observation_data,
        "legal_actions": [LegalAction(
            action="vote",
            target_seats=[2, 4],
            target_required=True,
        )],
    }, deep=True)
    vote_observation = AgentObservation.model_validate(vote_request.observation)
    vote_registry = build_werewolf_tool_registry(actor, vote_request, vote_observation)
    vote_schema = next(
        item["function"]["parameters"]
        for item in vote_registry.definitions()
        if item["function"]["name"] == "vote"
    )
    assert vote_schema["properties"]["target_seat"]["enum"] == [2, 4]
    context = ToolExecutionContext(
        request=vote_request,
        seat=actor.seat,
        role=actor.role.value,
        step=1,
        state_version=0,
    )
    invalid = await vote_registry.execute("vote-3", "vote", {"target_seat": 3}, context)
    assert not invalid.ok and invalid.error_code == "invalid_arguments"

    kill_request = base.model_copy(update={
        "phase": "night",
        "action_kind": "kill",
        "legal_actions": [LegalAction(
            action="night_kill",
            target_seats=[3, 5],
            target_required=True,
        )],
    }, deep=True)
    kill_registry = build_werewolf_tool_registry(
        actor,
        kill_request,
        AgentObservation.model_validate(kill_request.observation),
    )
    kill_schema = next(
        item["function"]["parameters"]
        for item in kill_registry.definitions()
        if item["function"]["name"] == "night_kill"
    )
    assert kill_schema["properties"]["target_seat"]["enum"] == [3, 5]

    empty_request = base.model_copy(update={
        "phase": "night",
        "action_kind": "save",
        "legal_actions": [LegalAction(
            action="save",
            target_seats=[],
            target_required=True,
            can_skip=True,
        )],
    }, deep=True)
    empty_registry = build_werewolf_tool_registry(
        actor,
        empty_request,
        AgentObservation.model_validate(empty_request.observation),
    )
    empty_definitions = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in empty_registry.definitions()
    }
    assert "save" not in empty_definitions
    assert "skip" in empty_definitions
    empty_context = ToolExecutionContext(
        request=empty_request,
        seat=actor.seat,
        role=actor.role.value,
        step=1,
        state_version=0,
    )
    invented_target = await empty_registry.execute(
        "invented-save-target",
        "save",
        {"target_seat": 999},
        empty_context,
    )
    assert not invented_target.ok and invented_target.error_code == "unknown_tool"


@pytest.mark.asyncio
async def test_hunter_shot_uses_the_canonical_night_kill_terminal_tool() -> None:
    router = ScriptedToolRouter([{
        "call_id": "hunter-model-call",
        "tool_calls": [{
            "id": "hunter-shot-call",
            "name": "night_kill",
            "arguments": {"target_seat": 3},
        }],
    }])
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.HUNTER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )
    state = new_game(["A", "B", "C", "D", "E", "F"], game_id="tool-hunter-shot")
    roles = [Role.HUNTER, Role.WEREWOLF, Role.VILLAGER, Role.VILLAGER, Role.SEER, Role.DOCTOR]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    player_id = next(player.id for player in state.players if player.seat == actor.seat)
    observation = build_observation(
        state,
        player_id,
        rng=random.Random(7),
        available_actions=["hunter_shot"],
        candidate_targets=[3, 5],
    )
    request = ActionRequest(
        request_id="tool-hunter-shot-request",
        run_id=state.id,
        seat=actor.seat,
        phase="hunter_shot",
        day=1,
        action_kind="hunter_shot",
        observation=observation.model_dump(),
        legal_actions=[LegalAction(
            action="night_kill",
            target_seats=[3, 5],
            target_required=True,
            can_skip=True,
        )],
    )

    envelope = await actor.decide(request)

    assert envelope.decision.action == AgentAction.NIGHT_KILL
    assert envelope.decision.target_seat == 3
    assert validate_decision_against_legal_actions(envelope, request).valid
    tool_names = {
        item["function"]["name"]
        for item in router.seen[0]
    }
    assert "night_kill" in tool_names
    assert "hunter_shot" not in tool_names
    night_kill_schema = next(
        item["function"]["parameters"]
        for item in router.seen[0]
        if item["function"]["name"] == "night_kill"
    )
    assert night_kill_schema["properties"]["target_seat"]["enum"] == [3, 5]
    registry = build_werewolf_tool_registry(actor, request, observation)
    invalid = await registry.execute(
        "illegal-hunter-shot",
        "night_kill",
        {"target_seat": 4},
        ToolExecutionContext(
            request=request,
            seat=actor.seat,
            role=actor.role.value,
            step=1,
            state_version=0,
        ),
    )
    assert not invalid.ok and invalid.error_code == "invalid_arguments"


def test_legacy_json_speech_sanitizer_keeps_history_but_accuses_only_alive_seats() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    observation = AgentObservation.model_validate(_request(actor).observation)
    observation.alive_seats = [1, 2, 4, 6]

    decision = actor._sanitize_speak({
        "speech": "保留历史查验和遗言回应，只指控仍存活的4号。",
        "bid": 1,
        "claim": {"role": "seer", "checked_seat": 3, "result": "wolf"},
        "reply_to": 5,
        "accuses": [3, 4, 5, 4],
    }, observation)
    dead_scalar = actor._sanitize_speak({
        "speech": "单个死亡席指控也必须被过滤。",
        "bid": 1,
        "accuses": 3,
    }, observation)

    assert decision.claim == {"role": "seer", "checked_seat": 3, "result": "wolf"}
    assert decision.reply_to == 5
    assert decision.accuses == [4]
    assert dead_scalar.accuses is None


@pytest.mark.asyncio
async def test_actor_initial_tool_turn_names_visible_and_alive_seats() -> None:
    router = ScriptedToolRouter([{
        "call_id": "m-roster",
        "tool_calls": [{
            "id": "speak-roster",
            "name": "speak",
            "arguments": {"speech": "只讨论本局存在的座位。", "bid": 1},
        }],
    }])
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )
    request = _request(actor)
    observation_data = dict(request.observation)
    observation_data["alive_seats"] = [1, 2, 4, 6]

    await actor.decide(request.model_copy(update={"observation": observation_data}, deep=True))

    initial_user = next(
        item
        for item in router.message_batches[0]
        if item.get("role") == "user"
    )
    payload = json.loads(str(initial_user["content"]))
    assert payload["visible_seats"] == [1, 2, 3, 4, 5, 6]
    assert payload["alive_seats"] == [1, 2, 4, 6]
    assert "禁止猜测、引用或向工具提交不存在的座位号" in payload["instruction"]
    assert "不必机械调用 get_legal_actions" in payload["instruction"]
    assert "update_private_state" in payload["instruction"]


@pytest.mark.asyncio
async def test_composite_private_state_tool_commits_once_and_projects_hard_facts() -> None:
    actor = AgentActor(
        seat=5,
        name="E",
        role=Role.SEER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    request = _request(actor)
    observation = AgentObservation.model_validate(request.observation)
    observation.private_events = [
        {
            "type": "seer_result",
            "payload": {"target_seat": 2, "team": "werewolves"},
        },
        {
            "type": "seer_result",
            "payload": {"target_seat": 3, "team": "village"},
        },
    ]
    registry = build_werewolf_tool_registry(actor, request, observation)
    context = ToolExecutionContext(
        request=request,
        seat=actor.seat,
        role=actor.role.value,
        step=1,
        state_version=0,
    )
    definition = next(
        item["function"]
        for item in registry.definitions()
        if item["function"]["name"] == "update_private_state"
    )
    assert set(definition["parameters"]["required"]) == {
        "beliefs",
        "candidate_plans",
        "selected_plan",
        "public_cover_role",
        "perceived_image",
        "deception_plan",
        "team_plan",
    }
    assert "Known wolf seats 2" in definition["parameters"]["description"]
    assert "Known village seats 3" in definition["parameters"]["description"]

    result = await registry.execute(
        "private-state-1",
        "update_private_state",
        {
            "beliefs": [
                {
                    "seat": 2,
                    "wolf_probability": 0.2,
                    "likely_role": None,
                    "confidence": 0.2,
                    "evidence": ["公开投票暂时没有结论"],
                },
                {
                    "seat": 3,
                    "wolf_probability": 0.8,
                    "likely_role": "werewolf",
                    "confidence": 0.4,
                    "evidence": ["怀疑其发言"],
                },
            ],
            "candidate_plans": ["保持低调观察", "先公开施压再归票"],
            "selected_plan": "保持低调观察并等待新证据",
            "public_cover_role": None,
            "perceived_image": "谨慎的普通村民",
            "deception_plan": "隐藏真实怀疑直到投票",
            "team_plan": None,
        },
        context,
    )

    assert result.ok
    assert result.output["updated"] is True
    assert result.output["revision"] == 1
    assert {item["seat"] for item in result.output["hard_fact_overrides"]} == {2, 3}
    snapshot = actor.private_state.snapshot()
    assert snapshot["beliefs"]["2"]["wolf_probability"] == 1.0
    assert snapshot["beliefs"]["3"]["wolf_probability"] == 0.0
    assert snapshot["selected_plan"] == "保持低调观察并等待新证据"


@pytest.mark.asyncio
async def test_actor_can_use_composite_private_state_before_exact_terminal_action() -> None:
    composite = {
        "beliefs": [{
            "seat": 2,
            "wolf_probability": 0.8,
            "likely_role": None,
            "confidence": 0.7,
            "evidence": ["vote mismatch"],
        }],
        "candidate_plans": ["accuse now", "wait for another claim"],
        "selected_plan": "wait for another claim",
        "public_cover_role": None,
        "perceived_image": "谨慎观察者",
        "deception_plan": "暂不公开真实怀疑",
        "team_plan": None,
    }
    router = ScriptedToolRouter([
        {"call_id": "m1", "tool_calls": [{
            "id": "state-call",
            "name": "update_private_state",
            "arguments": composite,
        }]},
        {"call_id": "m2", "tool_calls": [{
            "id": "speech-call",
            "name": "speak",
            "arguments": {"speech": "我先记录判断，再公开发言。", "bid": 1},
        }]},
    ])
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )

    envelope = await actor.decide(_request(actor))

    assert envelope.decision.action == AgentAction.SPEAK
    assert router.selected == ["update_private_state", "speak"]
    assert actor.private_state.snapshot()["selected_plan"] == "wait for another claim"


@pytest.mark.asyncio
async def test_actor_uses_private_tools_before_exact_terminal_speech() -> None:
    router = ScriptedToolRouter([
        {"call_id": "m1", "tool_calls": [{"id": "c1", "name": "get_legal_actions", "arguments": {}}]},
        {"call_id": "m2", "tool_calls": [{"id": "c2", "name": "update_belief", "arguments": {
            "seat": 2,
            "wolf_probability": 0.9,
            "confidence": 0.8,
            "evidence": ["vote mismatch"],
        }}]},
        {"call_id": "m3", "tool_calls": [{"id": "c3", "name": "set_plan", "arguments": {
            "selected_plan": "build a credible village case before accusing",
            "candidate_plans": ["accuse now", "build a credible village case before accusing"],
            "deception_plan": "keep the wolf suspicion private until the vote",
        }}]},
        {"call_id": "m4", "tool_calls": [{"id": "c4", "name": "speak", "arguments": {
            "speech": "我先记录2号的投票矛盾，暂不公开跳身份。",
            "bid": 2,
        }}]},
    ])
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )
    request = _request(actor)
    envelope = await actor.decide(request)

    assert envelope.decision.action == AgentAction.SPEAK
    assert envelope.decision.speech == "我先记录2号的投票矛盾，暂不公开跳身份。"
    assert validate_decision_against_legal_actions(envelope, request).valid
    snapshot = actor.private_state.snapshot()
    # The cognition layer projects subjective weights onto the configured
    # one-wolf mass, so the hard role-count constraint may rescale 0.9.
    assert snapshot["beliefs"]["2"]["wolf_probability"] > 0.25
    assert snapshot["selected_plan"].startswith("build a credible")
    assert snapshot["deception_plan"]
    assert router.selected == ["get_legal_actions", "update_belief", "set_plan", "speak"]
    trace = actor.agent_session.private_trace if actor.agent_session else []
    assert any(row["type"] == "tool_result" and row.get("terminal") for row in trace)
    assert all(row.get("visibility") == "admin" for row in trace)


@pytest.mark.asyncio
async def test_actor_batches_beliefs_before_one_exact_terminal_submission() -> None:
    router = ScriptedToolRouter([
        {"call_id": "m1", "tool_calls": [{"id": "c1", "name": "update_beliefs", "arguments": {
            "beliefs": [
                {
                    "seat": 2,
                    "wolf_probability": 0.9,
                    "confidence": 0.8,
                    "evidence": ["vote mismatch"],
                },
                {
                    "seat": 3,
                    "wolf_probability": 0.1,
                    "confidence": 0.7,
                    "evidence": ["consistent claim"],
                },
            ],
        }}]},
        {"call_id": "m2", "tool_calls": [{"id": "c2", "name": "speak", "arguments": {
            "speech": "我把几个座位的判断先统一记录，再公开发言。",
            "bid": 1,
        }}]},
    ])
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )

    envelope = await actor.decide(_request(actor))

    assert envelope.decision.action == AgentAction.SPEAK
    assert router.selected == ["update_beliefs", "speak"]
    snapshot = actor.private_state.snapshot()
    assert snapshot["revision"] == 1
    assert set(snapshot["beliefs"]) == {"2", "3", "4", "5", "6"}
    trace = actor.agent_session.private_trace if actor.agent_session else []
    terminal_rows = [
        row for row in trace
        if row.get("type") == "tool_result" and row.get("terminal") is True
    ]
    assert len(terminal_rows) == 1


@pytest.mark.asyncio
async def test_update_beliefs_tool_schema_and_failure_are_isolated() -> None:
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=object(),  # type: ignore[arg-type]
    )
    request = _request(actor)
    observation = AgentObservation.model_validate(request.observation)
    registry = build_werewolf_tool_registry(actor, request, observation)
    definition = next(
        item["function"]
        for item in registry.definitions()
        if item["function"]["name"] == "update_beliefs"
    )
    beliefs_schema = definition["parameters"]["properties"]["beliefs"]
    assert beliefs_schema["maxItems"] == 12
    assert beliefs_schema["uniqueItems"] is True
    assert set(beliefs_schema["items"]["required"]) == {
        "seat", "wolf_probability", "confidence", "evidence",
    }
    assert definition["parameters"]["additionalProperties"] is False
    assert "owner_seat" not in str(definition)

    context = ToolExecutionContext(
        request=request,
        seat=actor.seat,
        role=actor.role.value,
        step=1,
        state_version=0,
    )
    valid = await registry.execute(
        "batch-1",
        "update_beliefs",
        {"beliefs": [
            {"seat": 2, "wolf_probability": 0.8, "likely_role": "null", "confidence": 0.7, "evidence": ["vote"]},
            {"seat": 3, "wolf_probability": 0.2, "confidence": 0.6, "evidence": ["speech"]},
        ]},
        context,
    )
    assert valid.ok
    before = actor.private_state.snapshot()
    assert before["beliefs"]["2"]["likely_role"] is None
    invalid = await registry.execute(
        "batch-2",
        "update_beliefs",
        {"beliefs": [
            {"seat": 4, "wolf_probability": 0.9, "confidence": 0.8, "evidence": ["new"]},
            {"seat": 4, "wolf_probability": 0.1, "confidence": 0.8, "evidence": ["duplicate"]},
        ]},
        context,
    )
    assert not invalid.ok
    assert invalid.error_code == "invalid_cognition_update"
    assert invalid.error_details == {"constraint": "duplicate_seat_patch"}
    assert actor.private_state.snapshot() == before

    plan_invalid = await registry.execute(
        "plan-1",
        "set_plan",
        {
            "selected_plan": "hold position",
            "candidate_plans": ["push the claim", " push the claim "],
        },
        context,
    )
    assert not plan_invalid.ok
    assert plan_invalid.error_code == "invalid_cognition_update"
    assert plan_invalid.error_details == {"constraint": "private_state_constraint"}
    assert actor.private_state.snapshot() == before


@pytest.mark.asyncio
async def test_terminal_speech_preserves_model_whitespace_exactly() -> None:
    """Whitespace is part of accepted model text; only blankness is rejected."""
    public_text = "  先听完再投票。\n"
    router = ScriptedToolRouter([
        {"call_id": "m1", "tool_calls": [{"id": "c1", "name": "speak", "arguments": {
            "speech": public_text,
            "bid": 1,
        }}]},
    ])
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )

    envelope = await actor.decide(_request(actor))

    assert envelope.decision.action == AgentAction.SPEAK
    assert envelope.decision.speech == public_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action_kind", "seat", "role", "arguments", "field"),
    [
        (
            "wolf_council",
            2,
            Role.WEREWOLF,
            {"target_seat": 1, "team_message": "  团队原文。\n"},
            "team_message",
        ),
        (
            "last_words",
            1,
            Role.VILLAGER,
            {"speech": "  遗言原文。\n"},
            "speech",
        ),
    ],
)
async def test_other_terminal_text_tools_preserve_model_whitespace_exactly(
    action_kind: str,
    seat: int,
    role: Role,
    arguments: dict[str, Any],
    field: str,
) -> None:
    router = ScriptedToolRouter([
        {"call_id": "m1", "tool_calls": [{
            "id": "c1",
            "name": action_kind,
            "arguments": arguments,
        }]},
    ])
    actor = AgentActor(
        seat=seat,
        name="A",
        role=role,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )

    envelope = await actor.decide(_exact_text_request(actor, action_kind=action_kind))

    assert getattr(envelope.decision, field) == arguments[field]


@pytest.mark.asyncio
async def test_actor_tool_error_returns_to_same_loop_without_synthetic_skip() -> None:
    router = ScriptedToolRouter([
        {"call_id": "m1", "tool_calls": [{"id": "bad", "name": "speak", "arguments": {
            "speech": "bad",
            "bid": 9,
        }}]},
        {"call_id": "m2", "tool_calls": [{"id": "good", "name": "speak", "arguments": {
            "speech": "corrected",
            "bid": 1,
        }}]},
    ])
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )
    envelope = await actor.decide(_request(actor))
    assert envelope.decision.action == AgentAction.SPEAK
    assert envelope.decision.speech == "corrected"
    assert len(router.seen) == 2
    trace = actor.agent_session.private_trace if actor.agent_session else []
    assert any(
        row["type"] == "tool_result"
        and isinstance(row.get("error"), dict)
        and row["error"].get("code") == "invalid_arguments"
        for row in trace
    )


@pytest.mark.asyncio
async def test_one_actor_serializes_overlapping_decisions() -> None:
    class OverlapDetectingRouter:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.calls = 0

        async def complete_tools(self, _messages, _config, _tools, **_kwargs):
            self.calls += 1
            call_number = self.calls
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.02)
            finally:
                self.active -= 1
            return {
                "call_id": f"model-{call_number}",
                "tool_calls": [{
                    "id": f"terminal-{call_number}",
                    "name": "speak",
                    "arguments": {
                        "speech": f"serialized turn {call_number}",
                        "bid": 1,
                    },
                }],
            }

    router = OverlapDetectingRouter()
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )
    base = _request(actor)
    first, second = await asyncio.gather(
        actor.decide(base.model_copy(update={"request_id": "overlap-1"}, deep=True)),
        actor.decide(base.model_copy(update={"request_id": "overlap-2"}, deep=True)),
    )

    assert first.request_id == "overlap-1"
    assert second.request_id == "overlap-2"
    assert router.calls == 2
    assert router.max_active == 1


@pytest.mark.asyncio
async def test_actor_preserves_rejected_response_call_before_recovery() -> None:
    class RecoveringRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_tools(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                error = LLMResponseError("incomplete streamed tool response")
                error.llm_call_trace = {
                    "call_id": "rejected-real-call",
                    "request_hash": "a" * 64,
                    "response_hash": None,
                    "usage": {},
                    "transport_attempt_count": 1,
                    "transport_attempts": [],
                }
                raise error
            return {
                "call_id": "accepted-real-call",
                "trace": {
                    "call_id": "accepted-real-call",
                    "request_hash": "b" * 64,
                    "response_hash": "c" * 64,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                "tool_calls": [{
                    "id": "speak-terminal",
                    "name": "speak",
                    "arguments": {"speech": "recovered response", "bid": 1},
                }],
            }

    router = RecoveringRouter()
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )
    envelope = await actor.decide(_request(actor))
    trace = envelope.decision.llm_call_trace

    assert envelope.decision.speech == "recovered response"
    assert envelope.model_call_id == "accepted-real-call"
    assert trace["actor_response_attempt_count"] == 2
    assert [row["status"] for row in trace["actor_response_attempts"]] == [
        "response_rejected",
        "accepted",
    ]
    assert [
        row["llm_call"]["call_id"]
        for row in trace["actor_response_attempts"]
    ] == ["rejected-real-call", "accepted-real-call"]


@pytest.mark.asyncio
async def test_orchestrator_preserves_model_call_id_on_consumed_tool_decision() -> None:
    """The environment row must join the envelope to its accepted model turn."""
    router = ScriptedToolRouter([
        {
            "call_id": "model-seer-1",
            "trace": {
                "call_id": "model-seer-1",
                "context": {"request_id": "tool-orchestrator:request:000001"},
            },
            "tool_calls": [{
                "id": "tool-see-1",
                "name": "see",
                "arguments": {"target_seat": 2},
            }],
        },
    ])
    state = new_game(["A", "B", "C", "D", "E", "F"], game_id="tool-orchestrator")
    roles = [Role.SEER, Role.WEREWOLF, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    state.phase = Phase.NIGHT
    state.day = 1
    actors = build_actors(
        state,
        model_config=ModelConfig(provider="openai", model="tool-test", api_key="test"),
        router=router,  # type: ignore[arg-type]
        rng=random.Random(9),
    )
    trace: list[dict[str, Any]] = []
    orchestrator = GameOrchestratorV2(
        state=state,
        actors=actors,
        deck=roles,
        on_trace=trace.append,
        internal_events=True,
        decision_timeout=5,
    )

    await orchestrator._night_role_actions(Role.SEER, [NightActionType.SEE])

    response = next(row for row in trace if row.get("kind") == "agent_response")
    consumed = next(row for row in trace if row.get("type") == "decision_consumed")
    accepted_attempt = consumed["llm_call"]["actor_response_attempts"][-1]["llm_call"]
    assert response["envelope"]["model_call_id"] == "model-seer-1"
    assert consumed["model_call_id"] == "model-seer-1"
    assert consumed["call_id"] == consumed["model_call_id"]
    assert consumed["llm_call"]["call_id"] == "model-seer-1"
    assert accepted_attempt["call_id"] == consumed["model_call_id"]
    assert any(row.get("type") == "rules_result" and row["rules"]["status"] == "accepted" for row in trace)
