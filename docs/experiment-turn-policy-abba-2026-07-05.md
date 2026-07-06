# Turn-policy ABBA experiment - 2026-07-05

本报告记录一次真实 LLM 小批量 ABBA 实验,用于验证 `tests/multi_game_stats.py`
的多策略协议已经能产出可解释的 JSONL 和 paired delta。它不是显著性结论。

## Command

```bash
timeout --foreground -s INT -k 30s 64m python -u tests/multi_game_stats.py 2 \
  --jsonl logs/turn_policy_abba_bid_reply_20260705.jsonl \
  --bootstrap-iters 500 \
  --turn-policies bid_only,bid_reply \
  --policy-schedule abba \
  --experiment-seed 20260705 \
  --experiment-id turn-policy-abba-bid-reply-20260705 \
  | tee logs/turn_policy_abba_bid_reply_20260705.out
```

`n_games=2` 在多策略模式下表示每个 policy 2 局,总计 4 局。调度顺序为:

| global | policy | pair | order | case_seed |
|---:|---|---|---|---:|
| 1 | bid_only | pair-0001 | AB | 20260706 |
| 2 | bid_reply | pair-0001 | AB | 20260706 |
| 3 | bid_reply | pair-0002 | BA | 20260707 |
| 4 | bid_only | pair-0002 | BA | 20260707 |

每个 pair 的 `role_seed/actor_seed/orchestrator_seed` 相同,因此适合做 paired
delta。真实 LLM 输出仍非确定性,所以这只是受控小样本,不是完全可复现的模拟。

## Artifacts

- JSONL: `logs/turn_policy_abba_bid_reply_20260705.jsonl`
- Console summary: `logs/turn_policy_abba_bid_reply_20260705.out`

## Overall result

| Metric | Value |
|---|---:|
| Games | 4 |
| Village wins | 4 / 4 |
| Werewolf wins | 0 / 4 |
| Days | all day2 |
| `agent_decision_failed` | 0 |
| `game_ended` abnormal games | 0 |
| `parse_failed_rate` | 0.0% |
| Router calls | 207 |
| Router successes | 207 |
| Router failures | 0 |
| Router retries | 1 |
| Tokens in | 908,853 |
| Tokens out | 73,988 |

The 4-game win distribution is village-skewed, but Wilson CI remains wide
(`village 100%, 95%CI[51.0%,100.0%]`). Do not treat this as a balance
conclusion.

Note: this run happened before `router_stats_delta.avg_latency` was fixed to
use per-game latency windows. Therefore this report uses calls/retries/tokens
from the JSONL, but does not interpret the old per-game `avg_latency` deltas.
Future runs include `total_latency` and correct windowed `avg_latency`.

## Policy summaries

| Metric | bid_only (n=2) | bid_reply (n=2) |
|---|---:|---:|
| Village wins | 2 / 2 | 2 / 2 |
| Decision failures | 0 | 0 |
| Router calls | 106 | 101 |
| Router retries | 1 | 0 |
| Tokens in | 463,431 | 445,422 |
| Tokens out | 36,952 | 37,036 |
| Speech count mean | 22.50 | 21.50 |
| Reply rate mean | 40.2% | 46.8% |
| Accuse rate mean | 86.5% | 88.2% |
| Good vote accuracy | 100.0% | 100.0% |
| Game quality mean | 0.75 | 0.80 |
| Good final wolf suspicion gap | 0.34 | 0.30 |
| Deception success rate | 54.1% | 56.2% |
| Shared good target count | 2.00 | 1.50 |

## ABBA paired deltas

Delta is `bid_reply - bid_only`.

| Metric | Delta mean | 95% CI | n |
|---|---:|---|---:|
| village_win | 0.00 | [0.00, 0.00] | 2 |
| failed | 0.00 | [0.00, 0.00] | 2 |
| game_quality | +0.05 | [0.00, 0.10] | 2 |
| reply_rate | +0.07 | [0.06, 0.07] | 2 |
| vote_accuracy_good | 0.00 | [0.00, 0.00] | 2 |
| good_final_wolf_suspicion_gap | -0.04 | [-0.05, -0.02] | 2 |
| deception_success_rate | +0.02 | [-0.12, 0.17] | 2 |
| shared_good_target_count | -0.50 | [-1.00, 0.00] | 2 |

Small-sample reading:

- `bid_reply` increased structured reply rate in both pairs.
- It did not change win outcome or good vote accuracy in this run.
- `bid_reply` had slightly higher game quality and lower shared-good-target
  collusion count, but `n=2` is far too small for a mechanism claim.
- `good_final_wolf_suspicion_gap` was lower under `bid_reply` in both pairs,
  so higher reply rate did not automatically mean better final posterior.
- Deception success remained comparable; peer detection stayed at 0% in the
  console summary, which suggests future audit work should focus on explicit
  detection/recognition rather than only posterior movement.

## Qualitative observations

Both pairs generated the same broad social pattern: true seer information
appeared early, wolves countered with "jumped too fast" framing, and villagers
used repetition/standing patterns to identify wolf alignment. The result shows
the ABBA protocol is able to compare process metrics under paired role/RNG
conditions.

The post-fix smoke run after this experiment produced a stronger qualitative
case: the seer died on night1, no hard check information survived, and villagers
still voted out both wolves by detecting the public `1+3 -> 4` pressure pattern.
That supports the project direction: social-process metrics and collusion audit
matter even when hard role evidence is absent.

## Caveats

- This is only 4 games. Treat all CIs and paired deltas as diagnostic.
- Both policies were run with caucus disabled (`bid_only` vs `bid_reply`);
  this experiment does not address `bid_reply_caucus`.
- The model/backend state may vary over time despite paired seeds.
- The previous summary lacked router aggregation; `tests/multi_game_stats.py`
  now prints `=== Router stats delta ===` and per-policy router lines for
  future runs.

## Next experiments

1. Run a larger ABBA batch, minimum 6-10 games per policy, for
   `bid_only` vs `bid_reply`.
2. Run `bid_reply` vs `bid_reply_caucus` to isolate day1 wolf caucus effects.
3. Add `fixed_round_robin` only as a lower-interaction control, not as a
   preferred production policy.
4. Interpret every run with win rate, parse/router stability, debate metrics,
   deception/collusion audit, posterior calibration, and paired deltas together.
