"""agent memory + schemas 单元测试:三因子检索 + 方向A/B 字段校验。"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.agent.memory import AgentMemory
from src.agent.schemas import Decision, AgentAction


def test_three_factor_retrieval_prioritizes_important():
    """三因子检索:高 importance 的 claim 硬信号不被近期 speech 噪声淹没。"""
    m = AgentMemory(seat=1, role="villager")
    m.observe(1, "day", "phase_started", "第1天白天开始")  # importance 0.1
    m.observe(1, "day", "claim", "2号声称是预言家,查5号是狼")  # importance 0.9
    m.observe(1, "day", "speech", "3号说:我觉得5号可疑")  # importance 0.5
    # day2 噪声:20 条无关 speech(纯 recency 会把它们排前面,淹没 day1 claim)
    for i in range(20):
        m.observe(2, "day", "speech", f"{i}号说了一些无关紧要的话")
    m.update_trust({5: 0.9})  # 怀疑5号

    rec = m.recent_observations(limit=8)
    # claim 硬信号必须排第一(被检索到,不被噪声淹没)
    assert rec[0].kind == "claim", f"claim 应排第一,实际 {rec[0].kind}"
    assert "预言家" in rec[0].text


def test_relevance_boosts_suspect_mentions():
    """relevance:提及当前高怀疑座位的记忆加权。"""
    m = AgentMemory(seat=1, role="villager")
    m.observe(1, "day", "speech", "5号的行为很可疑")
    m.observe(1, "day", "speech", "2号说了些话")
    m.update_trust({5: 0.9})  # 怀疑5号,不怀疑2号
    rec = m.recent_observations(limit=2)
    # 提及5号的应排前面(relevance 加权)
    assert "5号" in rec[0].text


def test_attitudes_validator_normalizes():
    """方向B:attitudes validator 容忍 '3号'/'反对' → 归一化 support/oppose/neutral。"""
    d = Decision(action=AgentAction.SPEAK,
                 attitudes={"3号": "反对", "5": "支持", "7": "中立", "0": "x"})
    assert d.attitudes == {3: "oppose", 5: "support", 7: "neutral"}


def test_accuses_validator_filters_invalid():
    """方向A:accuses validator 过滤非正/重复/非法座位。"""
    d = Decision(action=AgentAction.SPEAK, accuses=["3号", 5, 5, "0", "abc", 3.0])
    assert d.accuses == [3, 5]


def test_reply_to_validator_tolerates():
    """方向A:reply_to validator 容忍 '3号'/3.0。"""
    d = Decision(action=AgentAction.SPEAK, reply_to="3号")
    assert d.reply_to == 3
    d2 = Decision(action=AgentAction.SPEAK, reply_to=0)
    assert d2.reply_to is None  # 非正座位→None


def test_deception_validator_normalizes():
    """task#31:deception validator 容忍中文/大小写 → 归一化 5 值。"""
    cases = {
        "OMISSION": "omission", "遗漏": "omission", "省略": "omission",
        "Distortion": "distortion", "扭曲": "distortion",
        "fabrication": "fabrication", "捏造": "fabrication",
        "misdirection": "misdirection", "误导": "misdirection", "转移": "misdirection",
        "none": "none", "无": "none", "诚实": "none", "真话": "none",
    }
    for raw, expected in cases.items():
        d = Decision(action=AgentAction.SPEAK, deception=raw)
        assert d.deception == expected, f"{raw} → 期望 {expected}, 实际 {d.deception}"
    # None / 未识别 → None
    assert Decision(action=AgentAction.SPEAK).deception is None
    assert Decision(action=AgentAction.SPEAK, deception="乱填的").deception is None


def test_claim_contradiction_detection():
    """claim 矛盾检测:对跳预言家 + 查验冲突。"""
    m = AgentMemory(seat=1, role="villager")
    m.record_claim(2, 1, {"role": "seer", "checked_seat": 5, "result": "wolf"})
    m.record_claim(3, 1, {"role": "seer", "checked_seat": 5, "result": "village"})  # 对跳 + 结果冲突
    ctrad = m.detect_claim_contradictions()
    assert any("对跳" in c for c in ctrad), "应检测到对跳预言家"
    assert any("查验冲突" in c or "冲突" in c for c in ctrad), "应检测到查验冲突"
