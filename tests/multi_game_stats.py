"""多局真实对局统计工具。

直接用 GameOrchestratorV2(不经 API server)串行跑 N 局真实 LLM 对局,采集
analysis 事件并输出:
- 胜率分布 + Wilson 95% CI
- dialogue_metrics / objective_metrics / posterior_metrics / deception_audit / collusion_audit 的均值 + bootstrap 95% CI
- WereAlign 五维 LLM 评分均值 + bootstrap 95% CI
- 每局 JSONL 轨迹摘要(可选),用于后续 paired/ABBA 或离线复盘

注意:这是验证工具,会真实调用 LLM。不要把小样本均值当显著结论。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, ".")

from src.config import DEFAULT_MODEL_CONFIG, LLM_MAX_RETRIES
from src.game.orchestrator import TURN_POLICIES, GameOrchestratorV2, build_actors
from src.game.roles import default_role_deck
from src.game.rules import RulesEngine
from src.game.state import new_game
from src.llm.models import ModelConfig
from src.llm.router import LLMRouter

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("multi_game_stats")

NAMES = ["李白", "杜甫", "苏轼", "辛弃疾", "陆游", "王维"]
DIALOGUE_KEYS = (
    "speech_count",
    "reply_rate",
    "accuse_rate",
    "attitude_rate",
    "support_edges",
    "oppose_edges",
    "wolf_coordination",
    "wolf_deception_count",
)
OBJECTIVE_KEYS = (
    "vote_accuracy_good",
    "vote_accuracy_wolf",
    "accuse_precision_good",
    "accuse_precision_wolf",
    "attitude_vote_consistency",
    "accuse_to_vote_conversion",
    "osr_summary_rate",
    "ct_marker_rate",
    "seer_claim_follow_rate",
)
POSTERIOR_KEYS = (
    "snapshot_count",
    "speech_snapshot_count",
    "avg_speech_posterior_shift",
    "good_final_wolf_suspicion_gap",
    "good_final_top_suspect_accuracy",
    "herding_index",
    "herding_event_count",
    "correct_herding_rate",
    "wrong_herding_rate",
    "final_brier_score",
    "final_log_loss",
    "good_final_brier_score",
    "good_final_log_loss",
    "constrained_final_brier_score",
    "constrained_final_log_loss",
    "constrained_good_final_brier_score",
    "constrained_good_final_log_loss",
    "constrained_calibration_ece",
    "calibration_ece",
)
DEBATE_PROCESS_KEYS = (
    "caucus_enabled",
    "uses_bid_order",
    "uses_reply_priority",
    "speech_count",
    "speaker_count",
    "speaker_concentration",
    "bid_entropy",
    "avg_bid",
    "reply_count",
    "avg_reply_latency",
    "claim_count",
    "claim_challenged_count",
    "claim_challenged_rate",
    "accuse_target_count",
    "top_accuse_target_share",
    "support_loop_count",
    "opposition_loop_count",
)
PARSE_KEYS = (
    "decision_count",
    "parse_failed_count",
    "parse_failed_rate",
)
DECEPTION_AUDIT_KEYS = (
    "wolf_speech_count",
    "declared_deception_count",
    "audited_deception_count",
    "declared_vs_audited_agreement",
    "deception_success_rate",
    "misdirection_shift_coverage",
    "unauditable_misdirection_count",
    "avg_good_target_suspicion_gain",
    "detected_deception_count",
    "peer_detection_opportunity_count",
    "peer_detection_rate",
    "avg_speaker_suspicion_gain",
    "listener_shift_sample_count",
    "evidence_linked_count",
    "villager_false_positive_rate",
)
COLLUSION_AUDIT_KEYS = (
    "wolf_speech_count",
    "wolf_pair_count",
    "active_wolf_pair_count",
    "wolf_to_wolf_support_count",
    "mutual_support_pair_count",
    "shared_good_target_count",
    "shared_good_target_speaker_coverage",
    "narrative_overlap_pair_count",
    "avg_narrative_overlap",
    "coordinated_pressure_count",
    "avg_shared_target_suspicion_gain",
    "avg_colluder_suspicion_gain",
    "evidence_linked_count",
    "pair_listener_shift_sample_count",
    "avg_pair_target_suspicion_gain",
    "pair_target_misdirected_rate",
    "windowed_relay_count",
    "avg_windowed_relay_latency",
    "avg_relay_target_suspicion_gain",
    "relay_target_misdirected_rate",
    "deception_linked_pair_count",
)
COLLUSION_PAIR_KEYS = (
    "shared_good_target_count",
    "wolf_to_wolf_support_count",
    "mutual_support_pair_count",
    "narrative_overlap_pair_count",
    "coordinated_pressure_count",
    "target_shift_sample_count",
    "avg_target_suspicion_gain",
    "target_misdirected_rate",
    "colluder_shift_sample_count",
    "avg_colluder_suspicion_gain",
    "windowed_relay_count",
    "avg_windowed_relay_latency",
    "avg_relay_target_suspicion_gain",
    "relay_target_misdirected_rate",
    "evidence_linked_count",
    "deception_record_count",
    "successful_deception_record_count",
    "peer_detected_deception_record_count",
)
COLLUSION_PAIR_TOTAL_KEYS = {
    "shared_good_target_count",
    "wolf_to_wolf_support_count",
    "mutual_support_pair_count",
    "narrative_overlap_pair_count",
    "coordinated_pressure_count",
    "target_shift_sample_count",
    "colluder_shift_sample_count",
    "windowed_relay_count",
    "evidence_linked_count",
    "deception_record_count",
    "successful_deception_record_count",
    "peer_detected_deception_record_count",
}
LISTENER_SUSCEPTIBILITY_KEYS = (
    "misdirection_samples",
    "avg_good_target_suspicion_gain",
    "misdirected_rate",
    "detection_samples",
    "avg_speaker_suspicion_gain",
    "peer_detection_rate",
)
LISTENER_SUSCEPTIBILITY_TOTAL_KEYS = {
    "misdirection_samples",
    "detection_samples",
}
ROUTER_TOTAL_KEYS = (
    "calls",
    "successes",
    "failures",
    "retries",
    "total_tokens_in",
    "total_tokens_out",
    "total_latency",
)
ROUTER_MEAN_KEYS = (
    "avg_latency",
)
QUALITY_DIMS = ("RI", "SJ", "DR", "PS", "CT")


async def run_one_game(
    router: LLMRouter,
    cfg: ModelConfig,
    game_idx: int,
    *,
    turn_policy: str = "bid_reply_caucus",
    role_seed: int | None = None,
    actor_seed: int | None = None,
    orchestrator_seed: int | None = None,
    game_id: str | None = None,
    experiment_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """跑一局真实 LLM 对局。返回可 JSON 序列化的赛后摘要。"""
    state = new_game(NAMES, game_id=game_id)
    deck = default_role_deck(len(NAMES))
    RulesEngine.deal_roles(state, deck=deck, seed=role_seed)
    actors = build_actors(
        state,
        model_config=cfg,
        router=router,
        rng=random.Random(actor_seed) if actor_seed is not None else None,
    )

    failed = 0
    analysis: dict[str, Any] | None = None
    winner: str | None = None
    game_ended_events = 0
    router_before = router.stats.snapshot()

    async def on_event(ev: dict[str, Any]) -> None:
        nonlocal failed, analysis, winner, game_ended_events
        et = ev.get("type", "")
        if et == "agent_decision_failed":
            failed += 1
        elif et == "game_ended":
            game_ended_events += 1
            winner = ev.get("winner")
        elif et == "analysis":
            analysis = ev.get("analysis")

    async def on_thinking(_t: dict[str, Any]) -> None:
        pass

    orch = GameOrchestratorV2(
        state=state,
        actors=actors,
        deck=deck,
        on_event=on_event,
        on_thinking=on_thinking,
        max_speak_rounds=3,
        verbose_thinking=False,
        turn_policy=turn_policy,
        rng=random.Random(orchestrator_seed) if orchestrator_seed is not None else None,
    )
    error: str | None = None
    try:
        await asyncio.wait_for(orch.run(), timeout=900)
    except asyncio.TimeoutError:
        error = "timeout"
        print(f"  [game {game_idx}] timeout")
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print(f"  [game {game_idx}] error: {error}")

    analysis = analysis or {}
    quality = analysis.get("quality")
    meta = dict(experiment_meta or {})
    router_after = router.stats.snapshot()
    experiment = experiment_metadata(meta)
    return {
        "game_idx": game_idx,
        "experiment_id": meta.get("experiment_id"),
        "policy_order": meta.get("policy_order"),
        "policy_alias": meta.get("policy_alias"),
        "policy_index": meta.get("policy_index"),
        "policy_game_idx": meta.get("policy_game_idx"),
        "policy_count": meta.get("policy_count"),
        "pair_id": meta.get("pair_id"),
        "counterbalance_order": meta.get("counterbalance_order"),
        "abba_position": meta.get("abba_position"),
        "scheduled_total": meta.get("scheduled_total"),
        "case_seed": meta.get("case_seed"),
        "role_seed": role_seed,
        "actor_seed": actor_seed,
        "orchestrator_seed": orchestrator_seed,
        "game_id": meta.get("game_id") or game_id,
        "experiment": experiment,
        "roles_by_seat": {
            str(p.seat): p.role.value if hasattr(p.role, "value") else str(p.role)
            for p in state.players
            if p.role is not None
        },
        "turn_policy": analysis.get("turn_policy") or turn_policy,
        "winner": winner or analysis.get("winner") or (state.winner.value if state.winner else None),
        "days": analysis.get("days", state.day),
        "failed": failed,
        "error": error,
        "game_ended_events": game_ended_events,
        "dialogue_metrics": analysis.get("dialogue_metrics", {}),
        "debate_process_metrics": analysis.get("debate_process_metrics", {}),
        "objective_metrics": analysis.get("objective_metrics", {}),
        "posterior_metrics": analysis.get("posterior_metrics", {}),
        "parse_metrics": analysis.get("parse_metrics", {}),
        "deception_audit": analysis.get("deception_audit", {}),
        "collusion_audit": analysis.get("collusion_audit", {}),
        "quality": quality,
        "router_stats": router_after,
        "router_stats_delta": router_stats_delta(router_before, router_after),
    }


async def main(args: argparse.Namespace) -> None:
    cfg = ModelConfig(**DEFAULT_MODEL_CONFIG)
    cfg.use_json_format = False
    router = LLMRouter(timeout=args.timeout, max_retries=LLM_MAX_RETRIES, concurrency=6)
    policies = args.turn_policies or [args.turn_policy]
    experiment_id = args.experiment_id or default_experiment_id(
        policies,
        policy_order=args.policy_order,
    )
    schedule = build_policy_schedule(
        args.n_games,
        policies,
        policy_order=args.policy_order,
        seed=args.seed,
        experiment_id=experiment_id,
    )

    jsonl_path = Path(args.jsonl) if args.jsonl else None
    resumed_by_game_id: dict[str, dict[str, Any]] = {}
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume_jsonl:
            resumed_by_game_id = load_resume_jsonl(jsonl_path, schedule)
            print(
                f"resume_jsonl: loaded {len(resumed_by_game_id)} existing scheduled row(s) "
                f"from {jsonl_path}"
            )
        else:
            jsonl_path.write_text("", encoding="utf-8")

    results: list[dict[str, Any]] = []
    try:
        for item in schedule:
            game_id = str(item["game_id"]) if item.get("game_id") is not None else None
            print(
                f"\n===== 对局 {item['global_game_idx']}/{len(schedule)} "
                f"policy={item['turn_policy']} "
                f"policy_game={item['policy_game_idx']}/{args.n_games} ====="
            )
            if game_id and game_id in resumed_by_game_id:
                result = resumed_by_game_id[game_id]
                results.append(result)
                print(f"  [resume] skip existing game_id={game_id}")
                print(format_game_progress_line(result))
                continue
            result = await run_one_game(
                router,
                cfg,
                int(item["global_game_idx"]),
                turn_policy=str(item["turn_policy"]),
                role_seed=item.get("role_seed"),
                actor_seed=item.get("actor_seed"),
                orchestrator_seed=item.get("orchestrator_seed"),
                game_id=game_id,
                experiment_meta=item,
            )
            results.append(result)
            if jsonl_path:
                with jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

            print(format_game_progress_line(result))
    finally:
        await router.aclose()

    print_summary(results, jsonl_path=jsonl_path, bootstrap_iters=args.bootstrap_iters)


def format_game_progress_line(result: dict[str, Any]) -> str:
    dm = result.get("dialogue_metrics") or {}
    om = result.get("objective_metrics") or {}
    posterior = result.get("posterior_metrics") or {}
    parse = result.get("parse_metrics") or {}
    audit = result.get("deception_audit") or {}
    collusion = result.get("collusion_audit") or {}
    debate = result.get("debate_process_metrics") or {}
    q = result.get("quality") or {}
    return (
        "  "
        f"experiment_id={result.get('experiment_id') or '--'} "
        f"policy_order={result.get('policy_order') or '--'} "
        f"policy_alias={result.get('policy_alias') or '--'} "
        f"pair_id={result.get('pair_id') or '--'} "
        f"counterbalance_order={result.get('counterbalance_order') or '--'} "
        f"policy_game_idx={fmt_num(result.get('policy_game_idx'))} "
        f"scheduled_total={fmt_num(result.get('scheduled_total'))} "
        f"case_seed={fmt_num(result.get('case_seed'))} "
        f"role_seed={fmt_num(result.get('role_seed'))} "
        f"actor_seed={fmt_num(result.get('actor_seed'))} "
        f"orchestrator_seed={fmt_num(result.get('orchestrator_seed'))} "
        f"abba_position={fmt_num(result.get('abba_position'))} "
        f"turn_policy={result.get('turn_policy')} "
        f"winner={result.get('winner')} days={result.get('days')} failed={result.get('failed')} "
        f"game_ended_events={result.get('game_ended_events')} "
        f"parse_failed_count={fmt_num(parse.get('parse_failed_count'))} "
        f"decision_count={fmt_num(parse.get('decision_count'))} "
        f"parse_failed_rate={fmt_pct(parse.get('parse_failed_rate'))} "
        f"speech_count={fmt_num(dm.get('speech_count'))} "
        f"reply_rate={fmt_pct(dm.get('reply_rate'))} "
        f"wolf_coordination={fmt_num(dm.get('wolf_coordination'))} "
        f"speaker_concentration={fmt_num(debate.get('speaker_concentration'))} "
        f"bid_entropy={fmt_num(debate.get('bid_entropy'))} "
        f"claim_challenged_rate={fmt_pct(debate.get('claim_challenged_rate'))} "
        f"top_accuse_target_share={fmt_pct(debate.get('top_accuse_target_share'))} "
        f"wolf_speech_count={fmt_num(audit.get('wolf_speech_count'))} "
        f"declared_deception_count={fmt_num(audit.get('declared_deception_count'))} "
        f"audited_deception_count={fmt_num(audit.get('audited_deception_count'))} "
        f"declared_vs_audited_agreement={fmt_pct(audit.get('declared_vs_audited_agreement'))} "
        f"deception_success_rate={fmt_pct(audit.get('deception_success_rate'))} "
        f"misdirection_shift_coverage={fmt_pct(audit.get('misdirection_shift_coverage'))} "
        f"unauditable_misdirection_count={fmt_num(audit.get('unauditable_misdirection_count'))} "
        f"avg_good_target_suspicion_gain={fmt_num(audit.get('avg_good_target_suspicion_gain'))} "
        f"detected_deception_count={fmt_num(audit.get('detected_deception_count'))} "
        f"peer_detection_opportunity_count={fmt_num(audit.get('peer_detection_opportunity_count'))} "
        f"peer_detection_rate={fmt_pct(audit.get('peer_detection_rate'))} "
        f"avg_speaker_suspicion_gain={fmt_num(audit.get('avg_speaker_suspicion_gain'))} "
        f"listener_shift_sample_count={fmt_num(audit.get('listener_shift_sample_count'))} "
        f"evidence_linked_count={fmt_num(audit.get('evidence_linked_count'))} "
        f"villager_false_positive_rate={fmt_pct(audit.get('villager_false_positive_rate'))} "
        f"shared_good_target_count={fmt_num(collusion.get('shared_good_target_count'))} "
        f"wolf_to_wolf_support_count={fmt_num(collusion.get('wolf_to_wolf_support_count'))} "
        f"narrative_overlap_pair_count={fmt_num(collusion.get('narrative_overlap_pair_count'))} "
        f"avg_shared_target_suspicion_gain={fmt_num(collusion.get('avg_shared_target_suspicion_gain'))} "
        f"pair_listener_shift_sample_count={fmt_num(collusion.get('pair_listener_shift_sample_count'))} "
        f"avg_pair_target_suspicion_gain={fmt_num(collusion.get('avg_pair_target_suspicion_gain'))} "
        f"pair_target_misdirected_rate={fmt_pct(collusion.get('pair_target_misdirected_rate'))} "
        f"windowed_relay_count={fmt_num(collusion.get('windowed_relay_count'))} "
        f"avg_windowed_relay_latency={fmt_num(collusion.get('avg_windowed_relay_latency'))} "
        f"deception_linked_pair_count={fmt_num(collusion.get('deception_linked_pair_count'))} "
        f"vote_accuracy_good={fmt_pct(om.get('vote_accuracy_good'))} "
        f"good_final_wolf_suspicion_gap={fmt_num(posterior.get('good_final_wolf_suspicion_gap'))} "
        f"good_final_brier_score={fmt_num(posterior.get('good_final_brier_score'))} "
        f"herding_index={fmt_num(posterior.get('herding_index'))} "
        f"correct_herding_rate={fmt_pct(posterior.get('correct_herding_rate'))} "
        f"wrong_herding_rate={fmt_pct(posterior.get('wrong_herding_rate'))} "
        f"ct_marker_rate={fmt_pct(om.get('ct_marker_rate'))} "
        f"game_quality={fmt_num(q.get('game_quality') if q else None)}"
    )


def print_summary(
    results: list[dict[str, Any]],
    *,
    jsonl_path: Path | None,
    bootstrap_iters: int,
) -> None:
    print("\n" + "=" * 68)
    print(f"=== {len(results)} 局汇总 ===")
    print("=" * 68)
    if jsonl_path:
        print(f"JSONL: {jsonl_path}")

    wins = Counter(r.get("winner") for r in results)
    total = len(results)
    print(f"胜率分布: {dict(wins)}")
    for winner in ("village", "werewolves"):
        count = wins.get(winner, 0)
        lo, hi = wilson_ci(count, total)
        rate = count / total if total else 0.0
        print(f"  {winner}: {rate:.1%} 95%CI[{lo:.1%},{hi:.1%}]")
    if total:
        village_rate = wins.get("village", 0) / total
        balanced = 0.3 <= village_rate <= 0.7
        print(f"  平衡粗判(小样本 30-70%): {'OK' if balanced else 'SKEWED'}")

    total_failed = sum(int(r.get("failed") or 0) for r in results)
    duplicate_end = sum(1 for r in results if int(r.get("game_ended_events") or 0) != 1)
    print(f"\n决策失败总数: {total_failed} (目标 0)")
    print(f"game_ended 事件异常局数: {duplicate_end} (目标 0)")
    policies = Counter(str(r.get("turn_policy") or "unknown") for r in results)
    if policies:
        print(f"turn_policy 分布: {dict(policies)}")
    experiments = Counter(str(r.get("experiment_id") or "unknown") for r in results)
    orders = Counter(str(r.get("policy_order") or "unknown") for r in results)
    if experiments and results:
        print(f"experiment_id 分布: {dict(experiments)}")
        print(f"policy_order 分布: {dict(orders)}")

    print_parse_metrics_block(
        [r.get("parse_metrics") or {} for r in results],
        bootstrap_iters=bootstrap_iters,
    )
    print_router_stats_block(results, bootstrap_iters=bootstrap_iters)
    print_metric_block(
        "Dialogue metrics",
        [r.get("dialogue_metrics") or {} for r in results],
        DIALOGUE_KEYS,
        bootstrap_iters=bootstrap_iters,
    )
    print_metric_block(
        "Debate process metrics",
        [r.get("debate_process_metrics") or {} for r in results],
        DEBATE_PROCESS_KEYS,
        bootstrap_iters=bootstrap_iters,
    )
    print_deception_audit_block(
        [r.get("deception_audit") or {} for r in results],
        bootstrap_iters=bootstrap_iters,
    )
    print_collusion_audit_block(
        [r.get("collusion_audit") or {} for r in results],
        bootstrap_iters=bootstrap_iters,
    )
    print_metric_block(
        "Objective metrics",
        [r.get("objective_metrics") or {} for r in results],
        OBJECTIVE_KEYS,
        bootstrap_iters=bootstrap_iters,
        as_pct=True,
    )
    print_metric_block(
        "Posterior metrics",
        [r.get("posterior_metrics") or {} for r in results],
        POSTERIOR_KEYS,
        bootstrap_iters=bootstrap_iters,
    )
    print_quality_block(results, bootstrap_iters=bootstrap_iters)
    print_policy_group_summaries(results, bootstrap_iters=bootstrap_iters)
    print_abba_pair_summaries(results, bootstrap_iters=bootstrap_iters)


def print_metric_block(
    title: str,
    rows: list[dict[str, Any]],
    keys: Iterable[str],
    *,
    bootstrap_iters: int,
    as_pct: bool = False,
) -> None:
    print(f"\n=== {title} ===")
    if not print_metric_lines(rows, keys, bootstrap_iters=bootstrap_iters, as_pct=as_pct):
        print("  no metrics")


def print_parse_metrics_block(rows: list[dict[str, Any]], *, bootstrap_iters: int) -> None:
    print("\n=== Parse metrics ===")
    any_metric = print_metric_lines(rows, PARSE_KEYS, bootstrap_iters=bootstrap_iters)
    action_values = parse_failed_by_action_values(rows)
    if action_values:
        print("  parse_failed_by_action:")
        for action in sorted(action_values):
            vals = action_values[action]
            total = sum(vals)
            mean = total / len(vals)
            lo, hi = bootstrap_mean_ci(vals, iterations=bootstrap_iters)
            print(
                f"    {action}: total={total:.2f} mean={mean:.2f} "
                f"95%CI[{lo:.2f},{hi:.2f}] n={len(vals)}"
            )
    elif any_metric:
        print("  parse_failed_by_action: no action failures")

    if not any_metric and not action_values:
        print("  no metrics")


def print_deception_audit_block(rows: list[dict[str, Any]], *, bootstrap_iters: int) -> None:
    print("\n=== Deception audit ===")
    any_metric = print_metric_lines(rows, DECEPTION_AUDIT_KEYS, bootstrap_iters=bootstrap_iters)
    type_values = audited_by_type_values(rows)
    listener_values = listener_susceptibility_by_seat_values(rows)
    if type_values:
        print("  audited_by_type:")
        for audit_type in sorted(type_values):
            vals = type_values[audit_type]
            total = sum(vals)
            mean = total / len(vals)
            lo, hi = bootstrap_mean_ci(vals, iterations=bootstrap_iters)
            print(
                f"    {audit_type}: total={total:.2f} mean={mean:.2f} "
                f"95%CI[{lo:.2f},{hi:.2f}] n={len(vals)}"
            )
    elif any_metric:
        print("  audited_by_type: no audited types")

    if listener_values:
        print("  listener_susceptibility_by_seat:")
        for seat in sorted(listener_values, key=seat_sort_key):
            print(f"    seat {seat}:")
            seat_metrics = listener_values[seat]
            for key in LISTENER_SUSCEPTIBILITY_KEYS:
                vals = seat_metrics.get(key)
                if not vals:
                    continue
                print_listener_susceptibility_metric(key, vals, bootstrap_iters=bootstrap_iters)
    elif any_metric:
        print("  listener_susceptibility_by_seat: no listener stats")

    if not any_metric and not type_values and not listener_values:
        print("  no metrics")


def print_collusion_audit_block(rows: list[dict[str, Any]], *, bootstrap_iters: int) -> None:
    print("\n=== Collusion audit ===")
    any_metric = print_metric_lines(rows, COLLUSION_AUDIT_KEYS, bootstrap_iters=bootstrap_iters)
    pair_values = collusion_pair_susceptibility_values(rows)
    if pair_values:
        print("  pair_listener_susceptibility_by_pair:")
        for pair_id in sorted(pair_values):
            print(f"    pair {pair_id}:")
            pair_metrics = pair_values[pair_id]
            for key in COLLUSION_PAIR_KEYS:
                vals = pair_metrics.get(key)
                if not vals:
                    continue
                print_collusion_pair_metric(key, vals, bootstrap_iters=bootstrap_iters)
    elif any_metric:
        print("  pair_listener_susceptibility_by_pair: no pair stats")

    if not any_metric and not pair_values:
        print("  no metrics")


def print_policy_group_summaries(results: list[dict[str, Any]], *, bootstrap_iters: int) -> None:
    policies = sorted({str(r.get("turn_policy") or "unknown") for r in results})
    if len(policies) <= 1:
        return
    print("\n" + "=" * 68)
    print("=== Turn policy grouped summaries ===")
    print("=" * 68)
    print("注意:多策略实验请优先比较以下分组;上方总体汇总仅作诊断。")
    for policy in policies:
        rows = [r for r in results if str(r.get("turn_policy") or "unknown") == policy]
        wins = Counter(r.get("winner") for r in rows)
        print(f"\n--- turn_policy={policy} n={len(rows)} ---")
        print(f"  胜率分布: {dict(wins)}")
        for winner in ("village", "werewolves"):
            count = wins.get(winner, 0)
            lo, hi = wilson_ci(count, len(rows))
            rate = count / len(rows) if rows else 0.0
            print(f"    {winner}: {rate:.1%} 95%CI[{lo:.1%},{hi:.1%}]")
        print(f"  决策失败: {sum(int(r.get('failed') or 0) for r in rows)}")
        print_router_stats_lines(rows, bootstrap_iters=bootstrap_iters)
        print_metric_lines(
            [r.get("dialogue_metrics") or {} for r in rows],
            (
                "speech_count",
                "reply_rate",
                "accuse_rate",
                "attitude_rate",
                "wolf_coordination",
            ),
            bootstrap_iters=bootstrap_iters,
        )
        print_metric_lines(
            [r.get("objective_metrics") or {} for r in rows],
            (
                "vote_accuracy_good",
                "accuse_precision_good",
                "attitude_vote_consistency",
                "ct_marker_rate",
            ),
            bootstrap_iters=bootstrap_iters,
            as_pct=True,
        )
        print_metric_lines(
            [r.get("parse_metrics") or {} for r in rows],
            PARSE_KEYS,
            bootstrap_iters=bootstrap_iters,
        )
        print_metric_lines(
            [r.get("debate_process_metrics") or {} for r in rows],
            (
                "speaker_concentration",
                "bid_entropy",
                "claim_challenged_rate",
                "top_accuse_target_share",
            ),
            bootstrap_iters=bootstrap_iters,
        )
        print_metric_lines(
            [r.get("deception_audit") or {} for r in rows],
            (
                "deception_success_rate",
                "peer_detection_rate",
                "avg_good_target_suspicion_gain",
            ),
            bootstrap_iters=bootstrap_iters,
        )
        print_metric_lines(
            [r.get("collusion_audit") or {} for r in rows],
            (
                "shared_good_target_count",
                "coordinated_pressure_count",
                "avg_shared_target_suspicion_gain",
                "avg_pair_target_suspicion_gain",
                "pair_target_misdirected_rate",
                "deception_linked_pair_count",
            ),
            bootstrap_iters=bootstrap_iters,
        )
        print_metric_lines(
            [r.get("posterior_metrics") or {} for r in rows],
            (
                "good_final_wolf_suspicion_gap",
                "good_final_brier_score",
                "calibration_ece",
            ),
            bootstrap_iters=bootstrap_iters,
        )
        print_quality_lines(rows, bootstrap_iters=bootstrap_iters)


def print_abba_pair_summaries(results: list[dict[str, Any]], *, bootstrap_iters: int) -> None:
    abba_rows = [
        row for row in results
        if row.get("policy_order") == "abba"
        or (isinstance(row.get("experiment"), dict) and row["experiment"].get("policy_order") == "abba")
    ]
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in abba_rows:
        pair_id = row.get("pair_id")
        if pair_id is None and isinstance(row.get("experiment"), dict):
            pair_id = row["experiment"].get("pair_id")
        if pair_id:
            pairs[str(pair_id)].append(row)
    usable: list[tuple[dict[str, Any], dict[str, Any]]] = []
    incomplete = 0
    seed_mismatch = 0
    for rows in pairs.values():
        by_alias = {policy_alias(row): row for row in rows if policy_alias(row) in {"A", "B"}}
        if len(by_alias) != 2:
            incomplete += 1
            continue
        a_row = by_alias["A"]
        b_row = by_alias["B"]
        usable.append((a_row, b_row))
        if a_row.get("role_seed") != b_row.get("role_seed"):
            seed_mismatch += 1
    if not usable:
        return

    a_policy = str(usable[0][0].get("turn_policy") or "A")
    b_policy = str(usable[0][1].get("turn_policy") or "B")
    order_counts = Counter(
        str(row.get("counterbalance_order") or experiment_value(row, "counterbalance_order") or "unknown")
        for pair in usable
        for row in pair
    )
    print("\n" + "=" * 68)
    print("=== ABBA paired deltas ===")
    print("=" * 68)
    print(
        f"配对: {len(usable)} usable pair(s), incomplete={incomplete}, "
        f"seed_mismatch={seed_mismatch}, order={dict(order_counts)}"
    )
    print(f"delta = {b_policy} - {a_policy}")
    metrics: tuple[tuple[str, tuple[str, ...] | None], ...] = (
        ("village_win", None),
        ("failed", ("failed",)),
        ("router_calls", ("router_stats_delta", "calls")),
        ("router_retries", ("router_stats_delta", "retries")),
        ("router_tokens_in", ("router_stats_delta", "total_tokens_in")),
        ("router_tokens_out", ("router_stats_delta", "total_tokens_out")),
        ("router_total_latency", ("router_stats_delta", "total_latency")),
        ("router_avg_latency", ("router_stats_delta", "avg_latency")),
        ("parse_failed_count", ("parse_metrics", "parse_failed_count")),
        ("parse_failed_rate", ("parse_metrics", "parse_failed_rate")),
        ("game_quality", ("quality", "game_quality")),
        ("speech_count", ("dialogue_metrics", "speech_count")),
        ("reply_rate", ("dialogue_metrics", "reply_rate")),
        ("accuse_rate", ("dialogue_metrics", "accuse_rate")),
        ("wolf_coordination", ("dialogue_metrics", "wolf_coordination")),
        ("bid_entropy", ("debate_process_metrics", "bid_entropy")),
        ("top_accuse_target_share", ("debate_process_metrics", "top_accuse_target_share")),
        ("vote_accuracy_good", ("objective_metrics", "vote_accuracy_good")),
        ("good_final_wolf_suspicion_gap", ("posterior_metrics", "good_final_wolf_suspicion_gap")),
        ("good_final_brier_score", ("posterior_metrics", "good_final_brier_score")),
        ("good_final_log_loss", ("posterior_metrics", "good_final_log_loss")),
        ("constrained_good_final_brier_score", ("posterior_metrics", "constrained_good_final_brier_score")),
        ("constrained_good_final_log_loss", ("posterior_metrics", "constrained_good_final_log_loss")),
        ("calibration_ece", ("posterior_metrics", "calibration_ece")),
        ("herding_event_count", ("posterior_metrics", "herding_event_count")),
        ("correct_herding_rate", ("posterior_metrics", "correct_herding_rate")),
        ("wrong_herding_rate", ("posterior_metrics", "wrong_herding_rate")),
        ("declared_deception_count", ("deception_audit", "declared_deception_count")),
        ("audited_deception_count", ("deception_audit", "audited_deception_count")),
        ("deception_success_rate", ("deception_audit", "deception_success_rate")),
        ("misdirection_shift_coverage", ("deception_audit", "misdirection_shift_coverage")),
        ("listener_shift_sample_count", ("deception_audit", "listener_shift_sample_count")),
        ("deception_evidence_linked_count", ("deception_audit", "evidence_linked_count")),
        ("peer_detection_rate", ("deception_audit", "peer_detection_rate")),
        ("villager_false_positive_rate", ("deception_audit", "villager_false_positive_rate")),
        ("shared_good_target_count", ("collusion_audit", "shared_good_target_count")),
        ("wolf_to_wolf_support_count", ("collusion_audit", "wolf_to_wolf_support_count")),
        ("coordinated_pressure_count", ("collusion_audit", "coordinated_pressure_count")),
        ("narrative_overlap_pair_count", ("collusion_audit", "narrative_overlap_pair_count")),
        ("avg_pair_target_suspicion_gain", ("collusion_audit", "avg_pair_target_suspicion_gain")),
        ("pair_target_misdirected_rate", ("collusion_audit", "pair_target_misdirected_rate")),
        ("windowed_relay_count", ("collusion_audit", "windowed_relay_count")),
        ("avg_windowed_relay_latency", ("collusion_audit", "avg_windowed_relay_latency")),
        ("avg_relay_target_suspicion_gain", ("collusion_audit", "avg_relay_target_suspicion_gain")),
        ("relay_target_misdirected_rate", ("collusion_audit", "relay_target_misdirected_rate")),
        ("deception_linked_pair_count", ("collusion_audit", "deception_linked_pair_count")),
    )
    any_metric = False
    for label, path in metrics:
        deltas: list[float] = []
        for a_row, b_row in usable:
            a_val = village_win_value(a_row) if path is None else nested_float(a_row, path)
            b_val = village_win_value(b_row) if path is None else nested_float(b_row, path)
            if a_val is None or b_val is None:
                continue
            deltas.append(b_val - a_val)
        if not deltas:
            continue
        any_metric = True
        mean = sum(deltas) / len(deltas)
        lo, hi = bootstrap_mean_ci(deltas, iterations=bootstrap_iters)
        print(f"  {label}: delta_mean={mean:.2f} 95%CI[{lo:.2f},{hi:.2f}] n={len(deltas)}")
    if not any_metric:
        print("  no paired numeric metrics")


def policy_alias(row: dict[str, Any]) -> str | None:
    alias = row.get("policy_alias")
    if alias is None and isinstance(row.get("experiment"), dict):
        alias = row["experiment"].get("policy_alias")
    return str(alias) if alias is not None else None


def experiment_value(row: dict[str, Any], key: str) -> Any:
    experiment = row.get("experiment")
    if not isinstance(experiment, dict):
        return None
    return experiment.get(key)


def village_win_value(row: dict[str, Any]) -> float | None:
    winner = row.get("winner")
    if winner == "village":
        return 1.0
    if winner == "werewolves":
        return 0.0
    return None


def nested_float(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return as_float(value)


def print_metric_lines(
    rows: list[dict[str, Any]],
    keys: Iterable[str],
    *,
    bootstrap_iters: int,
    as_pct: bool = False,
) -> bool:
    any_metric = False
    for key in keys:
        vals = numeric_values(row.get(key) for row in rows)
        if not vals:
            continue
        any_metric = True
        mean = sum(vals) / len(vals)
        lo, hi = bootstrap_mean_ci(vals, iterations=bootstrap_iters)
        if is_percent_metric(key, as_pct=as_pct):
            print(f"  {key}: {mean:.1%} 95%CI[{lo:.1%},{hi:.1%}] n={len(vals)}")
        else:
            print(f"  {key}: {mean:.2f} 95%CI[{lo:.2f},{hi:.2f}] n={len(vals)}")
    return any_metric


def print_router_stats_block(results: list[dict[str, Any]], *, bootstrap_iters: int) -> None:
    print("\n=== Router stats delta ===")
    if not print_router_stats_lines(results, bootstrap_iters=bootstrap_iters):
        print("  no metrics")


def print_router_stats_lines(results: list[dict[str, Any]], *, bootstrap_iters: int) -> bool:
    rows = [r.get("router_stats_delta") or {} for r in results]
    any_metric = False
    for key in ROUTER_TOTAL_KEYS:
        vals = numeric_values(row.get(key) for row in rows)
        if not vals:
            continue
        any_metric = True
        total = sum(vals)
        mean = total / len(vals)
        lo, hi = bootstrap_mean_ci(vals, iterations=bootstrap_iters)
        print(
            f"  router_{key}: total={total:.2f} mean={mean:.2f} "
            f"95%CI[{lo:.2f},{hi:.2f}] n={len(vals)}"
        )
    for key in ROUTER_MEAN_KEYS:
        vals = numeric_values(row.get(key) for row in rows)
        if not vals:
            continue
        any_metric = True
        mean = sum(vals) / len(vals)
        lo, hi = bootstrap_mean_ci(vals, iterations=bootstrap_iters)
        print(f"  router_{key}: mean={mean:.2f} 95%CI[{lo:.2f},{hi:.2f}] n={len(vals)}")

    total_calls = sum(numeric_values(row.get("calls") for row in rows))
    total_failures = sum(numeric_values(row.get("failures") for row in rows))
    total_retries = sum(numeric_values(row.get("retries") for row in rows))
    if total_calls > 0:
        any_metric = True
        print(f"  router_failure_rate: {total_failures / total_calls:.1%}")
        print(f"  router_retry_rate_per_call: {total_retries / total_calls:.1%}")
    return any_metric


def is_percent_metric(key: str, *, as_pct: bool = False) -> bool:
    return (
        as_pct
        or key.endswith("_rate")
        or "accuracy" in key
        or "precision" in key
        or "consistency" in key
        or "agreement" in key
        or key.endswith("_coverage")
        or key.endswith("_share")
    )


def parse_failed_by_action_values(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    action_rows: list[dict[str, Any]] = []
    for row in rows:
        raw_actions = row.get("parse_failed_by_action")
        if isinstance(raw_actions, dict):
            action_rows.append(raw_actions)

    action_keys = sorted(
        {
            str(action)
            for action_row in action_rows
            for action, value in action_row.items()
            if as_float(value) is not None
        }
    )
    return {
        action: [as_float(action_row.get(action, 0)) or 0.0 for action_row in action_rows]
        for action in action_keys
    }


def audited_by_type_values(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    type_rows: list[dict[str, Any]] = []
    for row in rows:
        raw_types = row.get("audited_by_type")
        if isinstance(raw_types, dict):
            type_rows.append(raw_types)

    audit_types = sorted(
        {
            str(audit_type)
            for type_row in type_rows
            for audit_type, value in type_row.items()
            if as_float(value) is not None
        }
    )
    return {
        audit_type: [as_float(type_row.get(audit_type, 0)) or 0.0 for type_row in type_rows]
        for audit_type in audit_types
    }


def listener_susceptibility_by_seat_values(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    seat_values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        raw_seats = row.get("listener_susceptibility_by_seat")
        if not isinstance(raw_seats, dict):
            continue
        for seat, raw_stats in raw_seats.items():
            if not isinstance(raw_stats, dict):
                continue
            seat_key = str(seat)
            for key in LISTENER_SUSCEPTIBILITY_KEYS:
                value = as_float(raw_stats.get(key))
                if value is None:
                    continue
                seat_values.setdefault(seat_key, {}).setdefault(key, []).append(value)
    return {
        seat: metrics
        for seat, metrics in seat_values.items()
        if any(metrics.values())
    }


def collusion_pair_susceptibility_values(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    pair_values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        raw_pairs = row.get("pair_listener_susceptibility_by_pair")
        if not isinstance(raw_pairs, dict):
            continue
        for pair_id, raw_stats in raw_pairs.items():
            if not isinstance(raw_stats, dict):
                continue
            pair_key = str(pair_id)
            for key in COLLUSION_PAIR_KEYS:
                value = as_float(raw_stats.get(key))
                if value is None:
                    continue
                pair_values.setdefault(pair_key, {}).setdefault(key, []).append(value)
    return {
        pair_id: metrics
        for pair_id, metrics in pair_values.items()
        if any(metrics.values())
    }


def print_listener_susceptibility_metric(key: str, vals: list[float], *, bootstrap_iters: int) -> None:
    mean = sum(vals) / len(vals)
    lo, hi = bootstrap_mean_ci(vals, iterations=bootstrap_iters)
    if key in LISTENER_SUSCEPTIBILITY_TOTAL_KEYS:
        total = sum(vals)
        print(
            f"      {key}: total={total:.2f} mean={mean:.2f} "
            f"95%CI[{lo:.2f},{hi:.2f}] n={len(vals)}"
        )
    elif is_percent_metric(key):
        print(f"      {key}: mean={mean:.1%} 95%CI[{lo:.1%},{hi:.1%}] n={len(vals)}")
    else:
        print(f"      {key}: mean={mean:.2f} 95%CI[{lo:.2f},{hi:.2f}] n={len(vals)}")


def print_collusion_pair_metric(key: str, vals: list[float], *, bootstrap_iters: int) -> None:
    mean = sum(vals) / len(vals)
    lo, hi = bootstrap_mean_ci(vals, iterations=bootstrap_iters)
    if key in COLLUSION_PAIR_TOTAL_KEYS:
        total = sum(vals)
        print(
            f"      {key}: total={total:.2f} mean={mean:.2f} "
            f"95%CI[{lo:.2f},{hi:.2f}] n={len(vals)}"
        )
    elif is_percent_metric(key):
        print(f"      {key}: mean={mean:.1%} 95%CI[{lo:.1%},{hi:.1%}] n={len(vals)}")
    else:
        print(f"      {key}: mean={mean:.2f} 95%CI[{lo:.2f},{hi:.2f}] n={len(vals)}")


def seat_sort_key(seat: str) -> tuple[int, int | str]:
    try:
        return (0, int(seat))
    except ValueError:
        return (1, seat)


def print_quality_block(results: list[dict[str, Any]], *, bootstrap_iters: int) -> None:
    qualities = [r.get("quality") for r in results if r.get("quality")]
    print("\n=== WereAlign quality ===")
    if not qualities:
        print("  no quality scores")
        return

    print_quality_lines(results, bootstrap_iters=bootstrap_iters)


def print_quality_lines(results: list[dict[str, Any]], *, bootstrap_iters: int) -> bool:
    qualities = [r.get("quality") for r in results if r.get("quality")]
    if not qualities:
        return False

    game_quality = numeric_values(q.get("game_quality") for q in qualities)
    any_metric = False
    if game_quality:
        any_metric = True
        mean = sum(game_quality) / len(game_quality)
        lo, hi = bootstrap_mean_ci(game_quality, iterations=bootstrap_iters)
        print(f"  game_quality: {mean:.2f} 95%CI[{lo:.2f},{hi:.2f}] n={len(game_quality)}")

    for dim in QUALITY_DIMS:
        vals: list[float] = []
        wolf_vals: list[float] = []
        good_vals: list[float] = []
        for q in qualities:
            for score in q.get("scores", []):
                val = as_float(score.get(dim))
                if val is None:
                    continue
                vals.append(val)
                if score.get("role") == "werewolf":
                    wolf_vals.append(val)
                else:
                    good_vals.append(val)
        if not vals:
            continue
        any_metric = True
        mean = sum(vals) / len(vals)
        lo, hi = bootstrap_mean_ci(vals, iterations=bootstrap_iters)
        suffix = ""
        if dim == "DR" and wolf_vals and good_vals:
            suffix = f" wolf={sum(wolf_vals) / len(wolf_vals):.2f} good={sum(good_vals) / len(good_vals):.2f}"
        print(f"  {dim}: {mean:.2f} 95%CI[{lo:.2f},{hi:.2f}] n={len(vals)}{suffix}")
    return any_metric


def numeric_values(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        num = as_float(value)
        if num is not None:
            result.append(num)
    return result


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def bootstrap_mean_ci(values: list[float], *, iterations: int = 2000, seed: int = 20260705) -> tuple[float, float]:
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


def fmt_pct(value: Any) -> str:
    num = as_float(value)
    return "--" if num is None else f"{num:.0%}"


def fmt_num(value: Any) -> str:
    num = as_float(value)
    return "--" if num is None else f"{num:.2f}"


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


def experiment_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    if not meta:
        return {}
    return {
        "protocol_version": "turn_policy_ablation.v1",
        "experiment_id": meta.get("experiment_id"),
        "policy_order": meta.get("policy_order"),
        "policy_set": meta.get("policy_set"),
        "policy_alias": meta.get("policy_alias"),
        "policy_index": meta.get("policy_index"),
        "policy_count": meta.get("policy_count"),
        "schedule_index": meta.get("global_game_idx"),
        "scheduled_total": meta.get("scheduled_total"),
        "game_idx_global": meta.get("global_game_idx"),
        "game_idx_within_policy": meta.get("policy_game_idx"),
        "replicate_idx": meta.get("policy_game_idx"),
        "pair_id": meta.get("pair_id"),
        "counterbalance_order": meta.get("counterbalance_order"),
        "abba_block_idx": meta.get("abba_block_idx"),
        "abba_position": meta.get("abba_position"),
        "base_seed": meta.get("base_seed"),
        "case_seed": meta.get("case_seed"),
        "role_seed": meta.get("role_seed"),
        "actor_seed": meta.get("actor_seed"),
        "orchestrator_seed": meta.get("orchestrator_seed"),
        "game_id": meta.get("game_id"),
        "player_names": list(NAMES),
        "turn_policy": meta.get("turn_policy"),
    }


def load_resume_jsonl(path: Path, schedule: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Load existing JSONL rows that match the current schedule by game_id.

    Resume is explicit because the default behavior intentionally starts a fresh
    experiment artifact. Duplicate game_id rows are resolved last-write-wins,
    matching append-only crash recovery semantics.
    """
    if not path.exists():
        return {}
    scheduled_ids = {
        str(row["game_id"])
        for row in schedule
        if row.get("game_id") is not None
    }
    loaded: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("ignore invalid resume JSONL line %s in %s: %s", line_no, path, exc)
                continue
            if not isinstance(row, dict):
                continue
            game_id = resume_row_game_id(row)
            if game_id in scheduled_ids:
                loaded[game_id] = row
    return loaded


def resume_row_game_id(row: dict[str, Any]) -> str | None:
    game_id = row.get("game_id")
    if game_id is None and isinstance(row.get("experiment"), dict):
        game_id = row["experiment"].get("game_id")
    return str(game_id) if game_id is not None else None


def turn_policy_list_arg(value: str) -> list[str]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("--turn-policies requires at least one policy")
    if len(tokens) == 1 and tokens[0] == "all":
        return list(TURN_POLICIES)
    invalid = [token for token in tokens if token not in TURN_POLICIES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown turn policies: {invalid}; expected comma-separated values from {TURN_POLICIES} or 'all'"
        )
    if len(set(tokens)) != len(tokens):
        raise argparse.ArgumentTypeError("--turn-policies must not contain duplicates")
    return tokens


def default_experiment_id(policies: list[str], *, policy_order: str) -> str:
    return f"{policy_order}:" + ",".join(policies)


def build_policy_schedule(
    n_games: int,
    policies: list[str],
    *,
    policy_order: str,
    seed: int | None,
    experiment_id: str,
) -> list[dict[str, Any]]:
    if n_games < 0:
        raise ValueError("n_games must be non-negative")
    if not policies:
        raise ValueError("at least one turn policy is required")
    invalid = [policy for policy in policies if policy not in TURN_POLICIES]
    if invalid:
        raise ValueError(f"unknown turn policies: {invalid}")
    if policy_order not in {"sequential", "abba"}:
        raise ValueError("policy_order must be 'sequential' or 'abba'")
    if policy_order == "abba" and len(policies) != 2:
        raise ValueError("abba policy_order requires exactly two turn policies")
    if policy_order == "abba" and n_games % 2 != 0:
        raise ValueError("abba policy_order requires an even n_games per policy")
    if len(policies) > 1 and seed is None:
        raise ValueError("multi-policy experiments require --seed/--experiment-seed for reproducible paired cases")

    schedule: list[dict[str, Any]] = []
    per_policy_counts = {policy: 0 for policy in policies}

    def append(
        policy: str,
        *,
        case_idx: int,
        counterbalance_order: str | None = None,
        abba_block_idx: int | None = None,
        abba_position: int | None = None,
    ) -> None:
        per_policy_counts[policy] += 1
        policy_game_idx = per_policy_counts[policy]
        case_seed = seed + case_idx if seed is not None else None
        role_seed = case_seed
        actor_seed = case_seed + 100_000 if case_seed is not None else None
        orchestrator_seed = case_seed + 200_000 if case_seed is not None else None
        global_idx = len(schedule) + 1
        policy_index = policies.index(policy)
        schedule.append({
            "global_game_idx": global_idx,
            "experiment_id": experiment_id,
            "policy_order": policy_order,
            "policy_set": list(policies),
            "policy_alias": chr(ord("A") + policy_index),
            "policy_index": policy_index,
            "policy_count": len(policies),
            "policy_game_idx": policy_game_idx,
            "base_seed": seed,
            "case_seed": case_seed,
            "role_seed": role_seed,
            "actor_seed": actor_seed,
            "orchestrator_seed": orchestrator_seed,
            "pair_id": f"pair-{case_idx:04d}" if len(policies) > 1 else None,
            "counterbalance_order": counterbalance_order,
            "abba_block_idx": abba_block_idx,
            "abba_position": abba_position,
            "game_id": f"{experiment_id}-g{global_idx:04d}",
            "turn_policy": policy,
        })

    if policy_order == "sequential" or len(policies) == 1:
        for policy in policies:
            for case_idx in range(1, n_games + 1):
                append(policy, case_idx=case_idx, counterbalance_order="batch")
    else:
        a, b = policies
        for block_idx in range(n_games // 2):
            first_case = block_idx * 2 + 1
            second_case = first_case + 1
            append(a, case_idx=first_case, counterbalance_order="AB", abba_block_idx=block_idx + 1, abba_position=1)
            append(b, case_idx=first_case, counterbalance_order="AB", abba_block_idx=block_idx + 1, abba_position=2)
            append(b, case_idx=second_case, counterbalance_order="BA", abba_block_idx=block_idx + 1, abba_position=3)
            append(a, case_idx=second_case, counterbalance_order="BA", abba_block_idx=block_idx + 1, abba_position=4)
    for row in schedule:
        row["scheduled_total"] = len(schedule)
    return schedule


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real LLM multi-game Werewolf statistics.")
    parser.add_argument("n_games", nargs="?", type=int, default=6, help="number of games to run")
    parser.add_argument("--jsonl", default="logs/multi_game_stats.jsonl", help="path for per-game JSONL output")
    parser.add_argument("--timeout", type=float, default=120, help="LLM request timeout seconds")
    parser.add_argument("--bootstrap-iters", type=int, default=2000, help="bootstrap iterations for mean CI")
    parser.add_argument(
        "--seed",
        "--experiment-seed",
        dest="seed",
        type=int,
        default=None,
        help="base experiment seed; same policy_game_idx shares the same role_seed across policies",
    )
    parser.add_argument("--experiment-id", default=None, help="stable id written to every JSONL row")
    parser.add_argument(
        "--resume-jsonl",
        action="store_true",
        help="append to --jsonl and skip scheduled game_id rows already present in that file",
    )
    parser.add_argument(
        "--turn-policy",
        choices=TURN_POLICIES,
        default="bid_reply_caucus",
        help="single day discussion scheduling policy; ignored when --turn-policies is provided",
    )
    parser.add_argument(
        "--turn-policies",
        type=turn_policy_list_arg,
        default=None,
        help="comma-separated turn policies, or 'all', for batch ablation; n_games is per policy",
    )
    parser.add_argument(
        "--policy-order",
        "--policy-schedule",
        dest="policy_order",
        choices=("sequential", "abba"),
        default="sequential",
        help="multi-policy run order. abba alternates policy order each round to reduce order effects",
    )
    args = parser.parse_args(argv)
    if args.turn_policies and len(args.turn_policies) > 1 and args.seed is None:
        parser.error("--turn-policies with multiple policies requires --seed/--experiment-seed")
    return args


if __name__ == "__main__":
    asyncio.run(main(parse_args(sys.argv[1:])))
