"""Rules engine for classic 6-12 player Werewolf MVP."""
from __future__ import annotations

import random
from collections import Counter
from uuid import uuid4

from .models import (
    DeathReason,
    Event,
    EventVisibility,
    GameState,
    NightAction,
    NightActionType,
    Phase,
    PlayerState,
    Vote,
)
from .roles import (
    CLASSIC_RULESET_ID,
    Role,
    Team,
    default_role_deck,
    validate_role_deck,
)


class RulesError(ValueError):
    """Raised when a requested game transition is illegal."""


class RulesEngine:
    """Pure-Python rules coordinator for setup, night, vote, and win checks."""

    @staticmethod
    def create_game(player_names: list[str], *, game_id: str | None = None) -> GameState:
        if not 6 <= len(player_names) <= 12:
            raise RulesError("classic MVP requires 6-12 players")
        if len(set(player_names)) != len(player_names):
            raise RulesError("player names must be unique")

        players = [
            PlayerState(id=str(uuid4()), name=name, seat=index + 1)
            for index, name in enumerate(player_names)
        ]
        return GameState(id=game_id or str(uuid4()), players=players)

    @staticmethod
    def deal_roles(
        state: GameState,
        *,
        deck: list[Role] | None = None,
        seed: int | str | None = None,
        include_hunter: bool | None = None,
        ruleset_id: str = CLASSIC_RULESET_ID,
    ) -> GameState:
        RulesEngine._require_phase(state, Phase.SETUP)
        raw_deck = (
            default_role_deck(len(state.players), include_hunter=include_hunter)
            if deck is None
            else list(deck)
        )
        try:
            deck = validate_role_deck(
                raw_deck,
                player_count=len(state.players),
                ruleset_id=ruleset_id,
            )
        except ValueError as err:
            raise RulesError(str(err)) from err

        rng = random.Random(seed)
        rng.shuffle(deck)

        for player, role in zip(state.players, deck, strict=True):
            player.role = role
            player.alive = True

        state.phase = Phase.NIGHT
        state.day = 1
        state.night_actions.clear()
        state.votes.clear()
        state.winner = None
        RulesEngine._append_event(state,
            Event(
                phase=Phase.SETUP,
                day=0,
                type="roles_dealt",
                message="Roles have been dealt. Night falls.",
            )
        )
        RulesEngine._emit_role_private_events(state)
        return state

    @staticmethod
    def submit_night_action(state: GameState, action: NightAction) -> GameState:
        """记录一个夜间行动(不立即结算)。编排器收集全部后调 resolve_night。

        约束校验:角色对应动作、药水存量、守卫连守、不杀队友。
        """
        RulesEngine._require_phase(state, Phase.NIGHT)
        actor = RulesEngine._require_living_player(state, action.actor_id)
        if not isinstance(action.target_id, str) or not action.target_id.strip():
            raise RulesError("night action target is required")
        try:
            target = state.get_player(action.target_id)
        except KeyError as err:
            raise RulesError("night action target must be a known player") from err
        if not target.alive:
            raise RulesError("night action target must be alive")
        if actor.role is None:
            raise RulesError("roles have not been dealt")

        expected = {
            Role.WEREWOLF: NightActionType.KILL,
            Role.SEER: NightActionType.SEE,
            Role.WITCH: {NightActionType.SAVE, NightActionType.POISON},
            Role.GUARD: NightActionType.GUARD,
            Role.DOCTOR: NightActionType.SAVE,
        }.get(Role(actor.role))
        if expected is None:
            raise RulesError(f"{actor.role} has no supported night action")
        if isinstance(expected, set):
            if action.action not in expected:
                raise RulesError(f"{actor.role} cannot perform {action.action}")
        elif action.action != expected:
            raise RulesError(f"{actor.role} cannot perform {action.action}")

        # 狼人不能杀队友
        if action.action == NightActionType.KILL and target and target.role == Role.WEREWOLF:
            raise RulesError("werewolves cannot target a teammate at night")
        if action.action in {NightActionType.SEE, NightActionType.POISON} and target.id == actor.id:
            raise RulesError(f"{action.action} cannot target the acting player")
        # 女巫药水存量
        if action.action == NightActionType.SAVE and not state.witch_antidote and Role(actor.role) == Role.WITCH:
            raise RulesError("witch antidote already used")
        if action.action == NightActionType.POISON and not state.witch_poison:
            raise RulesError("witch poison already used")
        if Role(actor.role) == Role.WITCH:
            witch_actions = [
                existing.action
                for existing in state.night_actions
                if existing.actor_id == actor.id
                and existing.action in {NightActionType.SAVE, NightActionType.POISON}
            ]
            if witch_actions and action.action not in witch_actions:
                raise RulesError("witch cannot use antidote and poison in the same night")
        # 守卫连守限制
        if action.action == NightActionType.GUARD and target:
            if state.last_guarded_seat == target.seat:
                raise RulesError("guard cannot protect the same player two nights in a row")

        # 同一 actor 对同一动作只保留最新提交。
        state.night_actions = [
            existing
            for existing in state.night_actions
            if not (existing.actor_id == action.actor_id and existing.action == action.action)
        ]
        state.night_actions.append(action)
        RulesEngine._append_event(state,
            Event(
                phase=state.phase,
                day=state.day,
                type="night_action_submitted",
                message="Your night action was recorded.",
                visibility=EventVisibility.PRIVATE,
                recipients=[actor.id],
                payload={"action": action.action, "target_id": action.target_id},
            )
        )
        return state

    @staticmethod
    def record_wolf_council_message(
        state: GameState,
        *,
        actor_id: str,
        target_id: str,
        message: str,
    ) -> Event:
        """Record one wolf's exact message for living wolf teammates only."""
        RulesEngine._require_phase(state, Phase.NIGHT)
        actor = RulesEngine._require_living_player(state, actor_id)
        target = RulesEngine._require_living_player(state, target_id)
        if actor.role != Role.WEREWOLF:
            raise RulesError("only a living werewolf may send a council message")
        if target.role == Role.WEREWOLF:
            raise RulesError("werewolf council target cannot be a teammate")
        text = str(message)
        if not text.strip():
            raise RulesError("werewolf council message must not be empty")
        recipients = [
            player.id
            for player in sorted(state.living_players(), key=lambda item: item.seat)
            if player.role == Role.WEREWOLF
        ]
        event = Event(
            phase=Phase.NIGHT,
            day=state.day,
            type="wolf_council_message",
            message=text,
            visibility=EventVisibility.PRIVATE,
            recipients=recipients,
            payload={
                "channel": "wolf_team",
                "speaker_id": actor.id,
                "speaker_seat": actor.seat,
                "target_id": target.id,
                "target_seat": target.seat,
            },
        )
        RulesEngine._append_event(state, event)
        return event

    @staticmethod
    def resolve_night(state: GameState) -> GameState:
        """结算夜晚:预言家查验 → 狼人击杀(受守卫/女巫救影响)→ 女巫毒 → 死亡公告。

        关键规则:
        - 守卫守护 + 女巫救同一人 = 同守同救,该人死亡(双救反死)。
        - 女巫不能在同一夜既救又毒(经典规则,编排器保证)。
        - 被毒死者 death_reason=POISONED,不能开枪(猎人)。
        - 死亡公告只说"X 死亡",不泄露角色与死因(除非配置公开)。
        """
        RulesEngine._require_phase(state, Phase.NIGHT)

        actions = state.night_actions
        kills = [a for a in actions if a.action == NightActionType.KILL]
        saves = [a for a in actions if a.action == NightActionType.SAVE]
        poisons = [a for a in actions if a.action == NightActionType.POISON]
        guards = [a for a in actions if a.action == NightActionType.GUARD]
        sees = [a for a in actions if a.action == NightActionType.SEE]

        # 1) 预言家查验(私有反馈)
        for action in sees:
            actor = state.get_player(action.actor_id)
            target = state.get_player(action.target_id)
            result = Team.WEREWOLVES if target.role == Role.WEREWOLF else Team.VILLAGE
            RulesEngine._append_event(state,
                Event(
                    phase=Phase.NIGHT,
                    day=state.day,
                    type="seer_result",
                    message=f"你查验了 {target.name}({target.seat}号),结果:{('狼人' if result == Team.WEREWOLVES else '好人')}",
                    visibility=EventVisibility.PRIVATE,
                    recipients=[actor.id],
                    payload={"target_id": target.id, "target_seat": target.seat, "team": result},
                )
            )

        # 2) 若调用方提交了多个狼人击杀动作，按 plurality 取目标。
        # 当前 orchestrator 会先完成 seeded tie-break，再只提交最终目标；
        # RulesEngine 的兜底必须仍与提交顺序无关，同票固定取最低座位。
        kill_target_id: str | None = None
        if kills:
            tally = Counter(a.target_id for a in kills)
            highest = max(tally.values())
            tied_targets = [target_id for target_id, count in tally.items() if count == highest]
            kill_target_id = min(
                tied_targets,
                key=lambda target_id: state.get_player(target_id).seat,
            )
        state.night_kill_target = kill_target_id

        # 3) 守卫守护集合
        guarded_ids = {a.target_id for a in guards}
        # 4) SAVE 保护集合（女巫解药与医生保护使用同一结算语义）
        saved_ids = {a.target_id for a in saves}

        killed: list[tuple[str, DeathReason]] = []
        # 狼刀结算:同守同救(既被守又被救)则死;只被守或只被救则活
        if kill_target_id:
            guarded = kill_target_id in guarded_ids
            saved = kill_target_id in saved_ids
            if guarded and saved:
                # 同守同救:双救反死
                killed.append((kill_target_id, DeathReason.WOLF_KILL))
            elif not guarded and not saved:
                killed.append((kill_target_id, DeathReason.WOLF_KILL))
            # 只守或只救 → 存活

        # 5) 女巫毒人(独立结算,被毒必死,且不能开枪)
        for action in poisons:
            if action.target_id:
                # 毒人不被守护/解药抵消(经典规则:毒不可解)
                victim_already = any(pid == action.target_id for pid, _ in killed)
                if not victim_already:
                    killed.append((action.target_id, DeathReason.POISONED))
                else:
                    # 已死者改记为毒杀(以禁开枪)
                    for i, (pid, _reason) in enumerate(killed):
                        if pid == action.target_id:
                            killed[i] = (pid, DeathReason.POISONED)

        # 6) 应用死亡 + 标记猎人待决
        state.night_deaths = []
        for pid, reason in killed:
            victim = state.get_player(pid)
            victim.alive = False
            victim.death_reason = reason
            victim.death_day = state.day
            state.night_deaths.append(
                {"id": pid, "seat": victim.seat, "name": victim.name, "reason": reason.value}
            )
            # 猎人:被毒不能开枪,其余死亡可开枪
            if victim.role == Role.HUNTER and reason != DeathReason.POISONED:
                state.pending_hunter.append(pid)

        # 7) 更新守卫连守记录
        if guards:
            # 取守卫(单个)的目标
            state.last_guarded_seat = state.get_player(guards[0].target_id).seat
        else:
            state.last_guarded_seat = None

        # 8) 死亡公告(不泄露角色/死因)
        if killed:
            names = [state.get_player(pid).name for pid, _ in killed]
            RulesEngine._append_event(state,
                Event(
                    phase=Phase.NIGHT,
                    day=state.day,
                    type="night_deaths",
                    message=f"天亮了。昨夜 {', '.join(names)} 死亡。",
                    payload={"dead_player_ids": [pid for pid, _ in killed]},
                )
            )
        else:
            RulesEngine._append_event(state,
                Event(
                    phase=Phase.NIGHT,
                    day=state.day,
                    type="night_deaths",
                    message="天亮了。昨夜平安夜,无人死亡。",
                    payload={"dead_player_ids": []},
                )
            )

        state.night_actions.clear()
        state.night_kill_target = None
        state.phase = Phase.DAY
        state.votes.clear()
        return state

    @staticmethod
    def apply_witch_save(state: GameState, *, used: bool) -> None:
        if used:
            state.witch_antidote = False

    @staticmethod
    def apply_witch_poison(state: GameState, *, used: bool) -> None:
        if used:
            state.witch_poison = False

    @staticmethod
    def queue_last_words(state: GameState, player_id: str, *, reason: str) -> None:
        """把待发表遗言入队。编排器收集后逐个让 agent 生成。"""
        try:
            player = state.get_player(player_id)
        except KeyError as err:
            raise RulesError("last-words player must be known") from err
        if any(item.get("id") == player_id for item in state.last_words_queue):
            raise RulesError("last-words opportunity is already queued")
        state.last_words_queue.append({"id": player_id, "seat": player.seat, "name": player.name, "reason": reason})

    @staticmethod
    def record_last_words(state: GameState, player_id: str, text: str) -> GameState:
        """记录某玩家的遗言(公开发布)。"""
        try:
            player = state.get_player(player_id)
        except KeyError as err:
            raise RulesError("last-words player must be known") from err
        if not isinstance(text, str) or not text.strip():
            raise RulesError("last-words text must not be empty")
        RulesEngine._append_event(state,
            Event(
                phase=state.phase,
                day=state.day,
                type="last_words",
                message=f"{player.name}({player.seat}号)的遗言:{text}",
                payload={"player_id": player_id, "text": text},
            )
        )
        state.last_words_queue = [q for q in state.last_words_queue if q["id"] != player_id]
        return state

    @staticmethod
    def hunter_shoot(state: GameState, hunter_id: str, target_id: str | None) -> GameState:
        """猎人开枪带走一人(或放弃)。开枪后猎人技能用尽。"""
        try:
            hunter = state.get_player(hunter_id)
        except KeyError as err:
            raise RulesError("hunter must be a known player") from err
        if hunter.role != Role.HUNTER:
            raise RulesError("only hunter can shoot")
        if hunter_id not in state.pending_hunter:
            raise RulesError("hunter has no pending shot (maybe poisoned)")
        if target_id:
            try:
                target = state.get_player(target_id)
            except KeyError as err:
                raise RulesError("hunter target must be a known player") from err
            if not target.alive:
                raise RulesError("hunter target must be alive")
            state.pending_hunter = [h for h in state.pending_hunter if h != hunter_id]
            target.alive = False
            target.death_reason = DeathReason.HUNTER_SHOT
            target.death_day = state.day
            RulesEngine._append_event(state,
                Event(
                    phase=state.phase,
                    day=state.day,
                    type="hunter_shot",
                    message=f"{hunter.name}({hunter.seat}号)开枪带走了 {target.name}({target.seat}号)。",
                    payload={"hunter_id": hunter_id, "target_id": target_id},
                )
            )
            # 被猎人带走的人若也是猎人则可连锁开枪(罕见)
            if target.role == Role.HUNTER:
                state.pending_hunter.append(target_id)
        else:
            state.pending_hunter = [h for h in state.pending_hunter if h != hunter_id]
            RulesEngine._append_event(state,
                Event(
                    phase=state.phase,
                    day=state.day,
                    type="hunter_shot",
                    message=f"{hunter.name}({hunter.seat}号)选择不开枪。",
                    payload={"hunter_id": hunter_id, "target_id": None},
                )
            )
        return state

    @staticmethod
    def start_vote(state: GameState) -> GameState:
        RulesEngine._require_phase(state, Phase.DAY)
        state.phase = Phase.VOTING
        state.votes.clear()
        RulesEngine._append_event(
            state,
            Event(phase=Phase.DAY, day=state.day, type="vote_started", message="Voting has started."),
        )
        return state

    @staticmethod
    def submit_vote(state: GameState, vote: Vote) -> GameState:
        RulesEngine._require_phase(state, Phase.VOTING)
        voter = RulesEngine._require_living_player(state, vote.voter_id)
        target = RulesEngine._require_living_player(state, vote.target_id)
        if voter.id == target.id:
            raise RulesError("players cannot vote for themselves")
        if state.pk_candidates and target.id not in set(state.pk_candidates):
            raise RulesError("PK vote target must be one of the tied candidates")
        state.votes[voter.id] = target.id
        RulesEngine._append_event(state,
            Event(
                phase=Phase.VOTING,
                day=state.day,
                type="vote_cast",
                message=f"{voter.name} voted for {target.name}.",
                payload={"voter_id": voter.id, "target_id": target.id,
                         "voter_seat": voter.seat, "target_seat": target.seat},
            )
        )
        return state

    @staticmethod
    def resolve_vote(state: GameState, *, allow_pk: bool = True, require_all: bool = True) -> GameState:
        """结算投票。平票时进入 PK(allow_pk=True);否则无人放逐。

        放逐死者:入队遗言、若为猎人则可开枪(被放逐可开枪)。
        """
        RulesEngine._require_phase(state, Phase.VOTING)
        living_ids = {player.id for player in state.living_players()}
        missing = living_ids - set(state.votes)
        if missing and require_all:
            raise RulesError("all living players must vote before resolving")
        tally = Counter(state.votes.values())
        if not tally:
            if require_all:
                raise RulesError("no votes to resolve")
            RulesEngine._append_event(state,
                Event(
                    phase=Phase.VOTING,
                    day=state.day,
                    type="vote_tied",
                    message="没有有效票型,无人被放逐。",
                    payload={
                        "tied_player_ids": [],
                        "votes": {},
                        "missing_player_ids": sorted(missing),
                    },
                )
            )
            state.pk_candidates = []
            state.votes.clear()
            return state
        top_count = tally.most_common(1)[0][1]
        tied_ids = sorted(pid for pid, count in tally.items() if count == top_count)

        if missing and top_count <= len(living_ids) / 2:
            RulesEngine._append_event(state,
                Event(
                    phase=Phase.VOTING,
                    day=state.day,
                    type="vote_tied",
                    message="投票不完整且有效票未过半,无人被放逐。",
                    payload={
                        "tied_player_ids": tied_ids,
                        "votes": dict(tally),
                        "missing_player_ids": sorted(missing),
                    },
                )
            )
            state.pk_candidates = []
            state.votes.clear()
            return state

        if len(tied_ids) == 1:
            exiled = state.get_player(tied_ids[0])
            exiled.alive = False
            exiled.death_reason = DeathReason.EXILED
            exiled.death_day = state.day
            RulesEngine._append_event(state,
                Event(
                    phase=Phase.VOTING,
                    day=state.day,
                    type="player_exiled",
                    message=f"{exiled.name}({exiled.seat}号)被投票放逐。",
                    payload={"exiled_player_id": exiled.id, "votes": dict(tally)},
                )
            )
            # 遗言入队(被放逐可发言)
            RulesEngine.queue_last_words(state, exiled.id, reason="exiled")
            # 猎人开枪(被放逐可开枪)
            if exiled.role == Role.HUNTER:
                state.pending_hunter.append(exiled.id)
            state.pk_candidates = []
        elif allow_pk and len(tied_ids) >= 2 and state.pk_candidates != tied_ids:
            # 首次平票:进入 PK
            state.pk_candidates = tied_ids
            RulesEngine._append_event(state,
                Event(
                    phase=Phase.VOTING,
                    day=state.day,
                    type="vote_tied_pk",
                    message=f"投票平票,{', '.join(state.get_player(p).name for p in tied_ids)} 进入 PK,请重新发言投票。",
                    payload={"tied_player_ids": tied_ids, "votes": dict(tally)},
                )
            )
            # PK 仍处于 VOTING 阶段,编排器让 PK 候选发言后重新收票
            state.votes.clear()
            return state
        else:
            # PK 后仍平票:无人放逐
            RulesEngine._append_event(state,
                Event(
                    phase=Phase.VOTING,
                    day=state.day,
                    type="vote_tied",
                    message="投票再次平票,无人被放逐。",
                    payload={"tied_player_ids": tied_ids, "votes": dict(tally)},
                )
            )
            state.pk_candidates = []

        state.votes.clear()
        return state

    @staticmethod
    def check_winner(state: GameState) -> Team | None:
        alive = state.living_players()
        wolves = [p for p in alive if p.role == Role.WEREWOLF]
        village = [p for p in alive if p.role != Role.WEREWOLF]
        if not wolves:
            return Team.VILLAGE
        if len(wolves) >= len(village):
            return Team.WEREWOLVES
        return None

    @staticmethod
    def _append_event(state: GameState, event: Event) -> None:
        """Append a domain event with a deterministic run-scoped identity."""
        event.id = f"{state.id}:event:{len(state.events) + 1:06d}"
        state.events.append(event)

    @staticmethod
    def _require_phase(state: GameState, phase: Phase) -> None:
        if state.phase != phase:
            raise RulesError(f"expected phase {phase}, got {state.phase}")

    @staticmethod
    def _require_living_player(state: GameState, player_id: str) -> PlayerState:
        try:
            player = state.get_player(player_id)
        except KeyError as err:
            raise RulesError("player must be known") from err
        if not player.alive:
            raise RulesError("player must be alive")
        return player

    @staticmethod
    def _emit_role_private_events(state: GameState) -> None:
        wolves = [p for p in state.players if p.role == Role.WEREWOLF]
        wolf_payload = [{"id": p.id, "name": p.name, "seat": p.seat} for p in wolves]
        for player in state.players:
            payload = {"role": player.role, "team": player.team}
            if player.role == Role.WEREWOLF:
                payload["teammates"] = wolf_payload
            RulesEngine._append_event(state,
                Event(
                    phase=Phase.SETUP,
                    day=0,
                    type="role_assigned",
                    message=f"Your role is {player.role}.",
                    visibility=EventVisibility.PRIVATE,
                    recipients=[player.id],
                    payload=payload,
                )
            )

    @staticmethod
    def _emit_win_event(state: GameState, winner: Team) -> None:
        RulesEngine._append_event(state,
            Event(
                phase=state.phase,
                day=state.day,
                type="game_ended",
                message=f"Game ended. Winner: {winner}.",
                payload={"winner": winner},
            )
        )
