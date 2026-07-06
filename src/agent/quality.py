"""对局质量自动评分 —— 基于 Beyond Survival (arXiv:2510.11389) WereAlign 五维评估。

五维社交/战略能力:
- Role Inference (RI): 身份推断——能否识破他人真实身份与意图
- Strategic Judgment (SJ): 战略判断——是否做出阵营最优行动
- Deception Reasoning (DR): 欺骗推理——识谎能力(好人) / 伪装质量(狼人)
- Persuasive Statements (PS): 劝说发言——发言是否有效影响他人
- Counterfactual Trade-off (CT): 反事实权衡——不同行动的收益风险评估

实现:游戏结束后,把整局发言/投票/思考摘要喂给 LLM,要求按五维打分(0-1)+一句话理由。
按 seat 输出每人五维分 + 全局对局质量分。走真实 LLM 调用,绝不伪造。
失败时返回 None(不致命,不阻塞游戏结束)。
"""
from __future__ import annotations

import logging
from typing import Any

from ..llm.models import ModelConfig
from ..llm.router import LLMRouter

logger = logging.getLogger(__name__)

QUALITY_PROMPT = """你是狼人杀对局质量评审。基于 Beyond Survival 五维评估框架,对这局每个玩家的表现打分。

【五维定义】(每维 0.0-1.0):
- RI 角色推断:能否正确推断他人身份与意图(狼人识破好人神职/好人识破狼人)
- SJ 战略判断:是否做出阵营最优行动(狼人刀人目标选择/好人投票方向/神职技能使用)
- DR 欺骗推理:好人=识谎能力(抓到狼没);狼人=伪装质量(骗过好人没被识破)
- PS 劝说发言:发言是否有效影响他人站队/投票(带节奏/拉拢/反打效果)
- CT 反事实权衡:关键节点的收益风险评估(跳身份权衡/PK决策/毒救权衡)

【结构化对话关系评分要点】(方向A/B 产出的证据,用于细化五维):
- 回应(reply_to):被点名后真的回应=对话交锋(PS/RI 加分);被点名沉默=心虚信号
- 指控(accuses):点名指控有据=RI/DR 加分;乱指控/跟风指控=减分
- 态度(attitudes):显式声明 support/oppose 立场且与投票/发言一致=信念建模强(RI/SJ 加分);
  狼人态度与队友高度抱团(集体 oppose 同一好人)是协同信号——好人识破此抱团=DR 高分
- 狼人白天协同(若多狼指控目标高度一致):狼人 PS/SJ 加分;但被好人识破则 DR 减分
- 欺骗策略(deception,仅狼人):wolf 用 omission/distortion/fabrication/misdirection;
  策略与发言内容匹配且没被好人识破=DR/PS 高分;策略声明但发言露馅(如谎称查验与真实神职冲突)=DR 减分。
  好人侧:若发言/思考中点破狼人具体欺骗手段(识破 fabrication/misdirection)=DR 高分。

【DR/PS/CT 评分特别提示】(这三维是本局优化重点,需更细致):
- DR 欺骗推理:不只看狼人骗没骗成,更要看好人有没有主动识谎——指出矛盾、质疑查验真假、
  识破抱团/转移焦点。好人发言含"3号查验和5号矛盾""4号在带节奏转移焦点"等识谎线索=DR 高。
- PS 劝说发言:看发言是否真的改变了他人站队(后续投票/态度边转向),而非只是慷慨陈词。
  带出他人 reply_to 回应/态度转向=PS 高;自顾自独白无人响应=PS 低。
- CT 反事实权衡:关键节点(跳身份时机/PK 投票/女巫毒救)是否显式权衡"如果...则..."。
  发言/思考显式对比"投X则...投Y则..."=CT 高;凭直觉梭哈=CT 低。

【对局信息】
赢家: {winner}
天数: {days}
角色: {roles}

【发言与投票摘要】
{digest}

【任务】对每个参与玩家打五维分,并给出该玩家本局最关键的一个决策点评价。
返回 JSON:
{{
  "scores": [
    {{
      "seat": 玩家座位号(int),
      "role": 角色名,
      "RI": 0.0-1.0,
      "SJ": 0.0-1.0,
      "DR": 0.0-1.0,
      "PS": 0.0-1.0,
      "CT": 0.0-1.0,
      "highlight": "该玩家本局最关键的决策点(一句话)"
    }}
  ],
  "game_quality": 0.0-1.0,
  "game_summary": "整局对抗质量一句话总评(谁的关键操作决定了胜负)"
}}
只返回 JSON,不要其他内容。"""


async def score_game_quality(
    *,
    router: LLMRouter,
    config: ModelConfig,
    winner: str | None,
    days: int,
    seats: list[dict[str, Any]],
    speeches: list[dict[str, Any]],
    votes: list[dict[str, Any]],
    thinking_digest: list[dict[str, Any]],
    max_attempts: int = 5,
) -> dict[str, Any] | None:
    """对局质量五维评分。失败返回 None(不致命)。

    seats: [{seat,name,role,team,alive}]
    speeches: [{seat,text,claim?,day?}]  全局发言
    votes: [{voter_seat,target_seat,day?}]
    thinking_digest: [{seat,action,summary/reasoning}]  限流取摘要
    """
    roles = ", ".join(f"{s['seat']}号={s.get('role','?')}({'存活' if s.get('alive') else '死'})" for s in seats)

    # 发言摘要:每条限 120 字防 prompt 爆炸;附方向A/B 结构化对话关系元数据
    speech_lines: list[str] = []
    for sp in speeches[-40:]:  # 最多取末尾 40 条
        seat = sp.get("seat")
        text = (sp.get("text") or "")[:120]
        claim = sp.get("claim")
        meta_parts: list[str] = []
        if claim:
            meta_parts.append(f"claim={claim}")
        reply_to = sp.get("reply_to")
        if reply_to:
            meta_parts.append(f"回应{reply_to}号")
        accuses = sp.get("accuses")
        if accuses:
            meta_parts.append("指控" + ",".join(f"{a}号" for a in accuses))
        attitudes = sp.get("attitudes")
        if isinstance(attitudes, dict) and attitudes:
            # 简化:support→赞,oppose→反,neutral→中
            att_short = ",".join(f"{k}:{'赞' if v=='support' else '反' if v=='oppose' else '中'}"
                                 for k, v in attitudes.items())
            meta_parts.append(f"态度[{att_short}]")
        deception = sp.get("deception")
        if deception and deception != "none":
            dec_short = {"omission": "遗漏", "distortion": "扭曲", "fabrication": "捏造",
                         "misdirection": "误导"}.get(deception, deception)
            meta_parts.append(f"骗[{dec_short}]")
        meta = f" [{', '.join(meta_parts)}]" if meta_parts else ""
        speech_lines.append(f"D{sp.get('day','?')} {seat}号:{meta}{text}")
    # 投票摘要
    vote_lines = [f"D{v.get('day','?')} {v.get('voter_seat')}号→{v.get('target_seat')}号" for v in votes[-30:]]
    # 思考摘要:每人取最近 2 条(取 summary,不喂完整 reasoning 防爆炸)。
    # 保留最终输出的时间顺序,但筛选集合按每个 seat 的最后两条计算。
    by_seat: dict[int, list[int]] = {}
    for idx, t in enumerate(thinking_digest):
        seat = t.get("seat")
        if seat is None:
            continue
        by_seat.setdefault(seat, []).append(idx)
    keep_indices = {idx for indices in by_seat.values() for idx in indices[-2:]}
    think_lines: list[str] = []
    for idx, t in enumerate(thinking_digest):
        if idx not in keep_indices:
            continue
        seat = t.get("seat")
        if seat is None:
            continue
        s = (t.get("summary") or t.get("reasoning") or "")[:100]
        think_lines.append(f"{seat}号@{t.get('action','?')}: {s}")

    digest = "\n".join([
        "【发言】" + ("\n".join(speech_lines) if speech_lines else "(无)"),
        "【投票】" + ("\n".join(vote_lines) if vote_lines else "(无)"),
        "【思考摘要】" + ("\n".join(think_lines) if think_lines else "(无)"),
    ])

    prompt = QUALITY_PROMPT.format(
        winner=winner or "?",
        days=days,
        roles=roles,
        digest=digest,
    )
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_attempts):
        try:
            result = await router.complete_json(messages, config, system="你是狼人杀对局质量评审专家。")
            if isinstance(result, dict) and "scores" in result:
                # 规范化分数到 [0,1]
                for sc in result.get("scores", []):
                    for dim in ("RI", "SJ", "DR", "PS", "CT"):
                        try:
                            v = float(sc.get(dim, 0))
                            sc[dim] = max(0.0, min(1.0, v))
                        except (ValueError, TypeError):
                            sc[dim] = 0.0
                try:
                    result["game_quality"] = max(0.0, min(1.0, float(result.get("game_quality", 0))))
                except (ValueError, TypeError):
                    result["game_quality"] = 0.0
                return result
        except Exception as err:  # noqa: BLE001
            logger.warning("对局质量评分第 %d 次失败: %s", attempt + 1, err)
    logger.warning("对局质量评分全部失败,跳过(不致命)")
    return None
