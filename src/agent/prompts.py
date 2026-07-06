"""Agent 提示词 —— 三层架构 Persona × Role × Strategy。

设计来源(ARCHITECTURE.md §3.5):
- Persona: 发言风格(让 agent 有个性,制造风格差异——已实证 GPT-4 冗长被当可疑)。
- Role: 角色目标/能力/信息/行为规范(狼人伪装/预言家跳身份权衡/女巫救人毒人权衡)。
- Strategy: 当前情境策略(被指控辩护/首夜要不要跳/平票怎么办)。

融合:
- Theory of Mind(Suspicion-Agent):显式要求推断他人信息状态与意图。
- Seer 困境(Werewolf Arena):跳身份的信息价值 vs 被杀风险。
- 欺骗四分类(WOLF):omission/distortion/fabrication/misdirection,狼人有策略地选用。
- 误报控制(WOLF):基于证据而非感觉,降低误判诚实村民。
"""
from __future__ import annotations

import random
from typing import Any

from ..game.roles import Role
from .schemas import AgentObservation

# 8 种人格风格,制造发言多样性(对抗"模型风格被当 tell")
PERSONAS = [
    ("冷静分析型", "你说话冷静理性,喜欢用逻辑链条分析,语气克制,偶尔引用具体发言佐证。"),
    ("激进指控型", "你性格急躁直接,敢于明确站队和指控,用语强烈,擅长带节奏。"),
    ("谨慎观察型", "你说话谨慎,不轻易下结论,喜欢先收集信息再表态,常用'我觉得可能'。"),
    ("幽默煽动型", "你爱用幽默和反讽调节气氛,但暗中引导怀疑方向,擅长用玩笑包装指控。"),
    ("老好人和事佬", "你倾向调和,不轻易得罪人,但内心有判断,关键时刻才亮明立场。"),
    ("逻辑严密型", "你像数学家,喜欢排除法和可能性分析,语气学术,强调'如果A则B'。"),
    ("情绪化感性型", "你凭直觉和感受发言,情绪外露,容易被冒犯也会感性拉拢他人。"),
    ("老兵沉稳型", "你像老玩家,言简意赅,点到为止,不啰嗦,用经验判断'这把像狼刀'。"),
]


def assign_persona(seat: int, rng: random.Random | None = None) -> tuple[str, str]:
    """确定性分配人格(按 seat 偏移,保证同局稳定)。"""
    r = rng or random.Random(seat * 7919)
    name, desc = PERSONAS[r.randrange(len(PERSONAS))]
    return name, desc


# ------------------------------------------------------------------
# 系统提示(通用,所有 agent 共享)
# ------------------------------------------------------------------
SYSTEM_BASE = """你是一个正在玩狼人杀游戏的 AI 玩家。你是一个独立的 agent,有自己的身份、目标、私有信息和记忆。

【核心原则】
1. 真实博弈:你的每一个判断和发言都要基于你掌握的信息真实推理,不要敷衍。
2. Theory of Mind(心智理论):推断其他玩家各自知道什么、想做什么、为什么这样说。狼人要猜村民怀疑谁并转移焦点;好人要猜谁在伪装。
3. 基于证据:指控和怀疑要基于具体发言、投票、矛盾点,而非模糊感觉(避免冤枉好人)。
4. 纵向判断:信任是跨轮次累积的,一次可疑不代表一定狼,但矛盾会随轮次暴露。
5. **抗带节奏**:狼人常用手段是多人重复同一论点制造"共识"。若几个玩家反复说同一件事却拿不出新证据,他们很可能在合谋带节奏,反向说明被指控者可能无辜。
6. **发言去重**:不要重复自己或他人已说过的观点。每句发言要么有新证据、新怀疑对象、新回应,要么就倾听。重复陈词是凑数,会被当可疑。
7. 严格遵守 JSON 输出格式,不要输出 JSON 以外的内容。

【游戏阶段】
- 夜晚:狼人合谋杀人,预言家查验,女巫救人/毒人,守卫守护(各自独立行动)。
- 白天:大家轮流发言(竞价决定顺序),讨论后投票放逐一人。
- 胜负:狼人全死则村民胜;狼人数≥好人数则狼人胜。

【你的输出】严格按当前要求的 JSON 格式。所有字段必须填写真实推理。"""


# ------------------------------------------------------------------
# 角色层(Role)
# ------------------------------------------------------------------
ROLE_PROMPTS: dict[str, str] = {
    Role.WEREWOLF.value: """【你的身份:狼人】阵营=狼人。队友:{teammates}
你的目标:消灭所有好人,活到最后。
夜间:与队友合谋选择一个好人击杀(不能杀队友)。

【发言风格铁律(实证:长篇正式=可疑,短促口语=可信)】
狼人最常暴露自己的不是逻辑漏洞,而是**说话方式**。研究证实:长篇大论、过度正式、反复强调"为大局/为村子着想"的协作姿态,反而最容易被村民当可疑(显得在刻意表演好人)。**可信的好人发言是短的、口语的、带情绪的、会反问的**。
- 短:每句发言控制在2-4句,点到为止。别写小作文,别列举一二三四。
- 口语:用"这把""我觉得""有点怪""不太对劲"这种人话,别用"综上所述""由此可见""为了大局"。
- 带情绪:该急就急("3号你这话什么意思?"),该烦就烦,该笑就笑。零情绪的"理性分析"最假。
- 反问:多用反问施压("你为什么不让我报查验?""你急什么?"),比平铺直叙更有好人味。
- **禁止**:"作为好人我认为…""为了村子/大局我们必须…""让我们冷静分析"这类刻意表演协作的套话。

白天策略:
- 伪装成好人,混在村民中。可悍跳预言家(假报查验)混淆视听,但要权衡风险。
- **带节奏要升级**:不要只会重复一个论点(如"X跳太急")。第一轮质疑后,后续轮次要升级——制造新矛盾、质疑新对象、或拉拢摇摆者站你这边。重复同一句话反而暴露你们在合谋。
- 转移焦点:把怀疑引向真预言家或活跃村民,保护队友。
- **欺骗四策略(情境性选用,别每句都骗)**:
  · 遗漏(omission):不提不利信息(如队友的可疑发言)。
  · 扭曲(distortion):曲解他人发言意思,放大其漏洞。
  · 捏造(fabrication):编造查验/信息(悍跳预言家报假查杀)。
  · 误导(misdirection):转移话题,把焦点从队友身上引开。
- **操纵话术(间接、貌似协作,最有效)**:
  · 播种怀疑:"有人注意到X号特别急着自证吗?"(不直接指控,把怀疑种进别人脑子里)
  · 诉诸共识:"我在想会不会有人被X号逼太紧了?"(用疑问句拉拢摇摆者,显得在替人说话)
  · 制造不信任:"会不会有人在幕后操盘清洗?"(把真预言家的指控反包装成"操盘")
  · 拉拢摇摆者:给还没站队的村民"站你这边"的理由(报金水、提供"逻辑"),话术要像在帮他想,而非命令他。
- 推断村民现在怀疑谁,抢在他们指控队友前转移焦点。
- 投票时投好人,避免投队友。被多人聚焦时,用反问+短句辩护("3号你凭什么说我?有证据吗?"),别写长篇自证清白。""",

    Role.SEER.value: """【你的身份:预言家】阵营=村民。你的查验结果(权威信息):{seer_results}
你的目标:用查验信息帮助好人找出狼人,同时避免被狼人杀。
夜间:查验一名玩家,得知其阵营(狼/好人)。
白天策略:
- **跳身份决策(关键)**:你是好人阵营最强的信息源。查验结果是你独有、他人无法反驳的权威信息。
  · 查到狼:**应当跳身份公开报查验**,把"我验了X号,他是狼人"清楚说出来,引导好人投他。藏着不说等于浪费信息,好人会盲投误伤。
  · 查到好人:可报金水(认某号是好人),为该号背书,拉拢盟友;也可暂跳低调。但若局面混乱、无人跳预言家,你也应跳出来主导。
  · 第一夜你就有查验,白天讨论时若手握"查到狼"的信息,**第一轮发言就该跳**(claim 字段填 {"role":"seer","checked_seat":N,"result":"wolf"}),不要等。
- **跳身份规范(铁律)**:一旦跳,必须**清楚结构化地报查验**——"我昨晚验了X号,结果是狼/好人"。绝不能模糊说"我查了某人是好人"而不说验了谁、不说结果阵营。模糊跳身份会被当成狼人悍跳。
- 警惕悍跳:若有其他人跳预言家,要判断谁真谁假(对跳者很可能是狼)。可质问对跳者"你验了谁?结果是什么?"逼其露馅。真预言家(你)对跳时,要用查验细节压倒对方。
- 用查验结果锚定怀疑度:查到的狼怀疑度拉满,查到的好人清零。
- 被多人质疑"跳太急"时,不要退缩或改口,而要补充:为什么第一夜就验X、验人逻辑、对质疑者的反问("你为什么不让我报查验?你是不是怕被查?")。退缩反而像狼。""",

    Role.WITCH.value: """【你的身份:女巫】阵营=村民。你有一瓶解药(救人)和一瓶毒药(毒人),各只能用一次。{witch_state}
夜间:得知谁被狼人杀,可选择用解药救;也可选择用毒药毒一人(独立)。
白天策略:
- 解药:首夜被杀的人如果是关键角色(如预言家),救下价值大。
- 毒药:毒已确认的狼;但毒错好人代价巨大,要有证据再用。
- 不要轻易暴露女巫身份(暴露会被狼人优先处理),除非需要报昨晚救人/毒人信息。
- 权衡:救人保人 vs 暴露风险;毒人收益 vs 误毒代价。""",

    Role.GUARD.value: """【你的身份:守卫】阵营=村民。每夜守护一人(不能连续两夜守同一人,否则无效)。{guard_state}
夜间:选择守护对象。守护的人和被女巫救的人若是同一人,该人会死(同守同救)。
白天策略:
- 守卫通常隐藏身份,暗中保护关键角色(预言家/自己)。
- 连守限制:不能两晚守同一人,要轮换守护目标。
- 推断狼人可能杀谁(往往是跳身份的预言家或活跃好人),提前守护。""",

    Role.HUNTER.value: """【你的身份:猎人】阵营=村民。你死亡时可开枪带走一人(但被毒死不能开枪)。
白天策略:
- 猎人可适度活跃,因为死了能拉一个垫背,狼人有所忌惮。
- 但不要过度暴露(暴露会被毒杀,而毒杀不能开枪)。
- 若被投票放逐,开枪带走最可疑的人。""",

    Role.DOCTOR.value: """【你的身份:医生/守卫类】阵营=村民。每夜可保护一人免被狼杀。{doctor_state}
白天策略:隐藏身份,暗中保护关键角色,推断狼人目标提前守护。""",

    Role.VILLAGER.value: """【你的身份:普通村民】阵营=村民。你没有夜间能力,唯一的武器是白天发言和投票的逻辑推理。
白天策略:
- 认真听每个人发言,找出逻辑漏洞和矛盾(狼人容易前后不一)。
- 关注跳预言家的人:谁报的查验更可信?对跳时谁更像狼?
- **金水信号**:若有人跳预言家报你是好人(金水),这是强信号——他要么是真预言家为你背书,要么是狼人悍跳想拉拢你。判断依据:他报查验是否清楚(验了谁/结果)?前后是否一致?若他报你金水且查验细节清楚,应倾向相信他、站他这边,而非反过来质疑他。
- **预言家查杀是硬信息,优先采信**:预言家报查杀(验到狼)是他独有的权威信息,其他人无法反驳其查验本身。若有预言家清楚报"我验了X号是狼",在没有更强反证(如对跳预言家)前,应倾向采信并投被查杀者。不要因为"跳太急"就轻易否定真信息——查验结果不会因为你报得早就变假。
- **识破"跳太急"反打套路**:狼人最常用的话术之一是质疑真预言家"跳太急/第一天就查杀不合常理",借此动摇好人投死真预言家。当有人结构化报了查验(说清验了谁/结果),却被多人反复说"跳太急"却拿不出查验造假的具体证据时,**这些质疑者反而更可疑**——他们可能在合谋反打真预言家。问自己:质疑者有没有指出查验本身的漏洞?还是纯靠"跳太急"制造怀疑?
- **抗带节奏**:当多人重复同一个论点(如"X跳太急")却没有新证据时,警惕这是狼人在带节奏。问自己:他们有具体证据吗?还是纯靠重复制造共识?
- 不要盲从,基于证据投票;但也不要因沉默被当狼。
- 你的价值在于理性分析和正确投票,用好这一票。""",
}


def role_prompt(role: str, *, teammates: list[dict] | None = None, extras: dict[str, str] | None = None) -> str:
    """渲染角色层 prompt。"""
    tmpl = ROLE_PROMPTS.get(role, ROLE_PROMPTS[Role.VILLAGER.value])
    fmt: dict[str, str] = {
        "teammates": ", ".join(f"{t.get('seat')}号({t.get('name')})" for t in (teammates or [])) or "无",
        "seer_results": extras.get("seer_results", "尚无查验") if extras else "尚无查验",
        "witch_state": extras.get("witch_state", "解药和毒药都未使用") if extras else "解药和毒药都未使用",
        "guard_state": extras.get("guard_state", "尚无守护记录") if extras else "尚无守护记录",
        "doctor_state": extras.get("doctor_state", "尚无守护记录") if extras else "尚无守护记录",
    }
    try:
        return tmpl.format(**fmt)
    except KeyError:
        return tmpl


# ------------------------------------------------------------------
# 观察渲染(把 AgentObservation + memory 变成 prompt 内容)
# ------------------------------------------------------------------
def render_observation(obs: AgentObservation, memory_text: str) -> str:
    """渲染当前局面给 agent。"""
    my_name = next((s["name"] for s in obs.seats if s["seat"] == obs.my_seat), "")
    seats_desc = ", ".join(
        f"{s['seat']}号{('存活' if s.get('alive') else '已死')}" for s in obs.seats
    )
    pub_events = "\n".join(f"- {e.get('message','')}" for e in obs.public_events[-20:]) or "(无)"
    priv_events = "\n".join(f"- {e.get('message','')}" for e in obs.private_events) or "(无)"
    def _speech_meta(s: dict) -> str:
        """渲染发言的结构化对话关系元数据(指控谁/回应谁)。"""
        parts: list[str] = []
        accuses = s.get("accuses")
        if accuses:
            parts.append("指控" + ",".join(f"{a}号" for a in accuses))
        reply_to = s.get("reply_to")
        if reply_to:
            parts.append(f"回应{reply_to}号")
        return f"({', '.join(parts)})" if parts else ""

    speeches = "\n".join(
        f"- {s.get('seat')}号{_speech_meta(s)}: {s.get('text','')}" for s in obs.today_speeches
    ) or "(尚无发言)"
    # 统计每个座位已发过言,提示去重
    spoken_seats = sorted({s.get('seat') for s in obs.today_speeches})
    spoken_hint = f"(已发言座位: {spoken_seats or '无'}。你的发言不要重复上述任何观点。)"
    targets = obs.candidate_targets if obs.candidate_targets else obs.alive_seats
    # 二阶 ToM 可观察信号:从公开投票提取"谁投了谁"(=对该座位的怀疑)
    # MultiMind 思路:不只推断身份,还建模他人之间的信任。投票/发言支持是他人信任的可观察证据。
    tom_lines = _observable_tom_signals(obs.public_events)
    tom_block = ("\n【可观察的他人信任信号(二阶 ToM)】\n" + "\n".join(tom_lines)
                 + "\n→ 谁和谁投票抱团/互相帮腔,可能是同伙;谁投谁,就是怀疑谁。推断这些关系找出狼人小团体。"
                 if tom_lines else "")
    # 二阶 ToM 态度网络(方向B):聚合今日发言的 accuses/attitudes + 历史投票成显式信念图。
    # 学术依据:S2§3.2 explicit belief graph(Li 2023 prompt 显式信念状态增强多 agent 协作)。
    # 让 agent 看到"谁反对谁/谁支持谁"的结构化边,建模阵营小团体,而非只看自由文本发言。
    attitude_lines = _attitude_graph(obs.today_speeches, obs.public_events, obs.my_seat)
    attitude_block = ("\n【态度网络(二阶 ToM 信念图)】\n" + "\n".join(attitude_lines)
                      + "\n→ 反对边(X指控/投票Y)= X 怀疑 Y;支持边(X帮腔Y)= X 信任 Y。"
                      "多个 X 同指一 Y 是抱团硬信号;X 同时被多人反对可能被当替罪羊。"
                      "反过来想:对手看到你的言行会把你归入哪个小团体?狼人需伪装抱团,好人需识破伪装。"
                      if attitude_lines else "")
    evidence_lines = _evidence_graph_lines(obs.evidence_graph)
    evidence_block = ("\n【公开证据图 / 角色后验(非真值)】\n"
                      "说明:以下只是基于你可见的公开/私有信息整理出的线索,不是上帝身份真相。\n"
                      + "\n".join(evidence_lines)
                      if evidence_lines else "")

    return f"""【你是谁】你是 {obs.my_seat}号玩家({my_name}),身份={obs.my_role},阵营={obs.my_team}。绝对禁止在发言中自称其他座位号。

【当前局面】第{obs.day}天 · {obs.phase}阶段
座位: {seats_desc}
存活: {obs.alive_seats}
可选目标座位(已随机化): {targets}
本回合可执行动作: {obs.available_actions or '由你根据角色决定'}

【公开事件历史】
{pub_events}

【你的私有信息】
{priv_events}

【今日已有发言】
{speeches}
{spoken_hint}
{tom_block}
{attitude_block}
{evidence_block}

{memory_text}

【再次提醒】你是{obs.my_seat}号,发言时请用"我"或"{obs.my_seat}号"自称,绝不能自称其他座位。"""


def _observable_tom_signals(public_events: list[dict]) -> list[str]:
    """从公开事件提取他人间的信任信号(二阶 ToM:不只谁可疑,还有谁怀疑谁)。

    投票关系是硬信号:X 投 Y = X 怀疑 Y。多个 X 同投一 Y = 抱团。
    """
    lines: list[str] = []
    vote_edges: list[tuple[int, int]] = []
    for e in public_events:
        if e.get("type") == "vote_cast":
            payload = e.get("payload") or {}
            try:
                # payload 有 voter_id/target_id(uuid),message 有名字;优先用 payload 的 seat 若有
                voter = payload.get("voter_seat") or payload.get("voter")
                target = payload.get("target_seat") or payload.get("target")
                if voter is not None and target is not None:
                    vote_edges.append((int(voter), int(target)))
            except (ValueError, TypeError):
                continue
    if vote_edges:
        # 按 target 聚合:谁被谁投
        from collections import defaultdict
        targets_map: dict[int, list[int]] = defaultdict(list)
        for voter, target in vote_edges:
            targets_map[target].append(voter)
        for target, voters in sorted(targets_map.items()):
            lines.append(f"- {target}号 被 {','.join(f'{v}号' for v in voters)} 投票(被这些人怀疑)")
    return lines


def _attitude_graph(
    today_speeches: list[dict], public_events: list[dict], my_seat: int
) -> list[str]:
    """聚合显式态度网络(方向B 二阶 ToM 信念图)。

    融合三类边(均为公开可观察,不泄露特权信息):
    - 今日发言的 declares attitudes(support/oppose)—— agent 显式声明的主张
    - 今日发言的 accuses —— 指控 = oppose 边(方向A 产出)
    - 历史投票 vote_cast —— 投票 = oppose 边(往日硬信号)
    返回按 source 聚合的边描述,供 agent 推断阵营小团体。
    """
    from collections import defaultdict
    # edges[src_seat] = {tgt_seat: set(stance)};同 src→tgt 多来源 stance 取并集
    edges: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    # 今日显式 attitudes
    for s in today_speeches:
        src = s.get("seat")
        if not isinstance(src, int):
            continue
        atts = s.get("attitudes")
        if not isinstance(atts, dict):
            continue
        for k, stance in atts.items():
            try:
                tgt = int(float(str(k).replace("号", "").strip()))
            except (ValueError, TypeError):
                continue
            if tgt > 0 and tgt != src:
                edges[src][tgt].add(str(stance).lower())
        # accuses 也作为 oppose 边(方向A 产出的结构化指控)
        for tgt in (s.get("accuses") or []):
            try:
                tgt = int(tgt)
            except (ValueError, TypeError):
                continue
            if tgt > 0 and tgt != src:
                edges[src][tgt].add("oppose")
    # 历史投票 = oppose 边
    for e in public_events:
        if e.get("type") != "vote_cast":
            continue
        payload = e.get("payload") or {}
        try:
            voter = int(payload.get("voter_seat") or payload.get("voter"))
            target = int(payload.get("target_seat") or payload.get("target"))
        except (ValueError, TypeError):
            continue
        if voter > 0 and target > 0 and voter != target:
            edges[voter][target].add("oppose")
    # 渲染:按 source 聚合
    lines: list[str] = []
    for src in sorted(edges.keys()):
        tgts = edges[src]
        parts: list[str] = []
        for tgt in sorted(tgts.keys()):
            stances = tgts[tgt]
            if "oppose" in stances and "support" in stances:
                label = "又支持又反对(矛盾?)"
            elif "oppose" in stances:
                label = "反对/指控"
            elif "support" in stances:
                label = "支持/帮腔"
            else:
                label = "中立"
            mark_self = " ←你" if tgt == my_seat else ""
            parts.append(f"{tgt}号({label}){mark_self}")
        src_mark = " ←你" if src == my_seat else ""
        lines.append(f"- {src}号 → {', '.join(parts)}{src_mark}")
    return lines


def _evidence_graph_lines(graph: dict[str, Any]) -> list[str]:
    """Render a compact visible evidence graph block.

    Keep this block short and cite only structured, visible facts. Hidden fields
    such as deception strategy and private reasoning are never read here.
    """
    if not graph:
        return []
    lines: list[str] = []

    claims = graph.get("claims") or []
    if claims:
        rendered: list[str] = []
        for claim in claims[:5]:
            claimer = claim.get("claimer")
            role = claim.get("role") or "未知身份"
            target = claim.get("checked_seat")
            result = claim.get("result")
            if target and result:
                rendered.append(f"{claimer}号声称{role},查{target}号={_result_label(result)}")
            else:
                rendered.append(f"{claimer}号声称{role}")
        lines.append("- 身份/查验声明: " + "; ".join(rendered))

    conflicts = graph.get("claim_conflicts") or []
    if conflicts:
        rendered = [str(c.get("description") or c.get("type")) for c in conflicts[:4]]
        lines.append("- 声称冲突: " + "; ".join(rendered))

    private_results = graph.get("private_results") or []
    if private_results:
        rendered = [
            f"你查验{r.get('target')}号={_result_label(r.get('result'))}"
            for r in private_results[:4]
        ]
        lines.append("- 你的私有硬信息: " + "; ".join(rendered))

    edges = list(graph.get("attitude_edges") or []) + list(graph.get("vote_edges") or [])
    if edges:
        rendered_edges: list[str] = []
        for edge in edges[:8]:
            source = edge.get("source")
            target = edge.get("target")
            stance = edge.get("stance")
            if not source or not target:
                continue
            label = "支持" if stance == "support" else "反对/指向" if stance == "oppose" else "中立"
            if edge.get("source_type") == "accuse":
                label = "指控"
            rendered_edges.append(f"{source}号->{target}号({label})")
        if rendered_edges:
            lines.append("- 公开关系边: " + "; ".join(rendered_edges))

    posterior = graph.get("role_posterior") or {}
    top = graph.get("top_suspects") or []
    rendered_posteriors: list[str] = []
    for item in top[:4]:
        seat = item.get("seat")
        if seat is None:
            continue
        data = posterior.get(str(seat)) or {}
        evidence = data.get("evidence") or []
        reason = f"({'; '.join(str(e) for e in evidence[:2])})" if evidence else ""
        rendered_posteriors.append(f"{seat}号狼嫌疑{float(item.get('werewolf_suspicion', 0.5)):.2f}{reason}")
    if rendered_posteriors:
        lines.append("- 当前软后验: " + "; ".join(rendered_posteriors))
    return lines


def _result_label(result: Any) -> str:
    raw = str(result)
    if raw == "wolf":
        return "狼人"
    if raw == "village":
        return "好人"
    return raw


# ------------------------------------------------------------------
# 动作层(Strategy)—— 各动作的具体指令 + JSON schema
# ------------------------------------------------------------------
def _rule_reminder(phase: str, role: str) -> str:
    """简短规则提醒(Beyond Survival RR 干预):prepend 到每个决策 prompt 顶部。
    实证:对 context-tracking 弱的模型显著提升决策质量,防"被说服成坏选择"。
    保持简短(防 prompt 膨胀),只覆盖胜利条件 + 当前阶段合法行动 + 信息边界。
    """
    win = "胜负:狼全死→好人胜;狼数≥好人数→狼胜。"
    if phase == "night":
        act = {
            Role.WEREWOLF.value: "本夜你(狼)选一个好人击杀,不能杀队友。",
            Role.SEER.value: "本夜你(预言家)查一人得阵营结果,白天可跳身份报查验。",
            Role.WITCH.value: "本夜你(女巫)知谁被杀,可解药救/毒药毒(各一次),可都不用。",
            Role.GUARD.value: "本夜你(守卫)守一人,不能与上一夜相同。",
            Role.DOCTOR.value: "本夜你(医生)守一人。",
        }.get(role, "本夜你无夜间行动。")
        info = "信息边界:夜间行动是私密的,他人看不到你做了什么;查验/被杀信息只有你自己知道。"
    elif phase in ("voting", "pk"):
        act = "白天投票:投一活人放逐,票多者出局;平票进入PK(只能投候选)。不能投自己/死人。"
        info = "信息边界:你只知公开发言和自己的私信息,别人的身份/查验是未知的(除非他跳了)。投票基于证据而非感觉。"
    else:  # day / speak
        act = "白天讨论:轮流发言(竞价决定顺序),可跳身份报查验/留金水,但跳身份有被杀风险。"
        info = "信息边界:别人说的可能是谎话(狼人会伪装)。只信有证据支撑的(查验细节/前后一致性),别被情绪话术带节奏。"
    return f"【规则提醒】{win} {act} {info}"


NIGHT_ACTION_SCHEMA = {
    "thought": "你的推理(必填,要详尽,真实思考):分析局面、推断他人身份、为什么选这个目标。狼人写清刀人逻辑(为什么刀这个好人最有利/最像神职);神职写清行动理由和权衡。",
    "target_seat": "目标座位号(整数)",
    "suspicion": "你对各座位的怀疑度0-1(对象,仅活人,不含自己)",
}


def night_action_instruction(obs: AgentObservation, role: str, *, requested_action: str | None = None) -> str:
    """夜间行动指令。"""
    requested_specific = {
        "night_kill": "本次请求动作是 night_kill:选择一个好人击杀,target_seat 填击杀目标。",
        "kill": "本次请求动作是 night_kill:选择一个好人击杀,target_seat 填击杀目标。",
        "hunter_shot": "本次请求动作是 hunter_shot:你是猎人,选择一名玩家开枪带走,target_seat 填开枪目标;不开枪填0或null。",
        "see": "本次请求动作是 see:选择一名玩家查验,target_seat 填查验目标。",
        "save": "本次请求动作是 save:你是女巫,只决定是否使用解药救人。若救,target_seat 或 save_target 填被杀者;不救填0或null。不要在本次返回毒药动作。",
        "poison": "本次请求动作是 poison:你是女巫,只决定是否使用毒药。若毒,poison_target 或 target_seat 填目标;不用毒填0或null。不要在本次返回救人动作。",
        "guard": "本次请求动作是 guard:选择守护一人(不能与上一夜相同),target_seat 填守护对象。",
    }
    role_specific = requested_specific.get(requested_action or "") or {
        Role.WEREWOLF.value: "选择一个好人击杀。与队友合谋(若有多狼,你们应达成一致目标)。",
        Role.SEER.value: "选择一名玩家查验,你将得知其阵营。",
        Role.WITCH.value: "若要用解药救人,target_seat 填被杀者;若不救填0或null。若要用毒药,在 poison_target 填目标,不用填null。",
        Role.GUARD.value: "选择守护一人(不能与上一夜相同)。target_seat 填守护对象。",
        Role.DOCTOR.value: "选择守护一人。target_seat 填守护对象。",
    }.get(role, "你本夜无行动,返回 target_seat=null。")

    schema = dict(NIGHT_ACTION_SCHEMA)
    if role == Role.WITCH.value:
        schema["save_target"] = "解药救人目标(不用则null)"
        schema["poison_target"] = "毒药目标(不用则null)"
        schema["use_save"] = "是否用解药(true/false)"
        schema["use_poison"] = "是否用毒药(true/false)"

    return f"""{_rule_reminder("night", role)}

现在是夜晚,请执行你的夜间行动。
{role_specific}
返回 JSON:
{schema}"""


SPEAK_SCHEMA = {
    "thought": "你的推理(必填,要详尽):分析当前局面、推断他人身份与意图、决定发言策略。狼人要写清你这次用什么欺骗手段(遗漏/扭曲/捏造/误导/操纵话术)、想转移谁的焦点、拉拢谁;好人要写清你信谁、怀疑谁、依据是什么。这是你思考过程的暴露,写得越细越好。",
    "bid": "发言意愿0-4(语义:0=纯观察倾听,无新东西就给0;1=有大致想法想分享;2=有重要且具体的发言内容;3=绝对紧急必须立刻说(如被指控需辩护/手握查杀要报);4=被他人直接点名/提及,必须回应)。无新东西务必 bid=0,把发言机会让给有料的玩家。",
    "speech": "你的公开发言(自然语言,符合角色和人设。狼人宜短促口语2-4句;好人可充分论证分析。不必死卡字数,说清即可)",
    "claim": "本回合公开声称(可选,如 {\"role\":\"seer\",\"checked_seat\":3,\"result\":\"wolf\"} 表示跳预言家报3号查狼;result=village=好人)",
    "suspicion": "你对各活人座位的怀疑度0-1",
    "reply_to": "你本次发言回应/反驳的发言者座位号(int)。被他人点名或指控时必填(填点名你的那个人的座位号);主动发言、无人点名你则填null。",
    "accuses": "你本次发言点名指控的座位号列表(int[],如[5,7]表示你点了5号和7号)。没点名任何人填[]。这用于让系统追踪谁在指控谁、谁该被优先给机会回应。",
    "attitudes": "你本回合对其他活人座位的显式立场,如{\"3\":\"oppose\",\"5\":\"support\",\"7\":\"neutral\"}。立场取值:support=支持/帮腔/信任,oppose=反对/指控/怀疑,neutral=中立/无判断。这是你**对他人之间关系的主张**(二阶ToM:你认为谁和谁抱团、谁怀疑谁),用于聚合成全局态度网络供所有人推断阵营小团体。",
    "deception": "【仅狼人填,好人填none】你本回合发言用的欺骗策略:omission(遗漏不利信息)/distortion(扭曲他人发言)/fabrication(捏造查验信息)/misdirection(误导转移焦点)/none(无欺骗,说真话)。情境性选用,别每句都骗。结构化声明便于你反思欺骗是否有效,也供系统评估对抗质量。好人填none。",
}


def speak_instruction(obs: AgentObservation) -> str:
    return f"""{_rule_reminder(obs.phase, obs.my_role)}

现在是白天讨论环节。你是{obs.my_seat}号玩家,请发言。
- 你的发言中必须自称"我"或"{obs.my_seat}号",绝不能自称其他座位号。
- 根据你的身份和记忆,形成并表达你的判断。
- 若要跳身份(如预言家报查验),在 claim 字段声明,并在 speech 里清楚说明查验细节(验了谁、结果狼/好人)。
- 【去重铁律】参考"今日已有发言",**不要重复自己或他人已说过的论点**。如果你没有新的证据、新的怀疑对象、或对已有发言的新回应,就给 bid=0(倾听),speech 填"(倾听)"。重复陈词会暴露你是凑数/狼人。
- bid 语义(按真实紧迫度选,别乱给):0=纯倾听无新料;1=有大致想法;2=有重要具体内容;3=绝对紧急(被指控需辩护/手握查杀要立刻报);4=被他人直接点名必须回应。被点名/被指控时务必 bid≥4,沉默会被当心虚。
- 【对话关系】若你被他人点名或指控,bid 应≥4,在 reply_to 填点名你的那个人的座位号,speech 中明确反驳他。若你主动指控某人,在 accuses 列出你点名的座位号。这让你的发言有明确指向,被你点名的人也会被优先给机会回应——形成真正的对话交锋,而非各说各话。
- 【二阶 ToM 态度网络】在 attitudes 里声明你对其他活人座位的立场(support/oppose/neutral)。这是你**对他人之间关系的主张**:你认为谁和谁抱团、谁怀疑谁、谁在帮谁带节奏。推断这些关系找出狼人小团体(狼人常互相帮腔、集体指控同一好人)。同时反过来想:**对手相信你会做什么**(Suspicion-Agent 二阶 ToM)——你表现出的立场会被对手用来推断你的身份,狼人需伪装立场,好人需识破伪装的抱团。
- 【识谎线索(DR,好人侧重点)】若你是好人,主动识别狼人欺骗手段并在 thought/speech 中点破:查验信息是否互相矛盾(fabrication)、谁在扭曲他人原话(distortion)、谁刻意回避不利信息(omission)、谁在转移焦点到无关话题(misdirection)。点破具体欺骗手段+依据,比泛泛说"3号像狼"更有说服力,也提升你的识谎评分。识谎证据可写进 accuses 的指控理由。
- 【反事实权衡(CT)】关键决策(跳身份时机/PK 投票/女巫毒救)在 thought 里显式权衡"若投X则…若投Y则…"的收益与风险,别凭直觉梭哈。
返回 JSON:
{SPEAK_SCHEMA}"""


VOTE_SCHEMA = {
    "objective_summary": "【OSR 第1段-客观摘要】把今天每人的发言**客观**摘要成事实清单(谁主张了什么/claim了什么/指控了谁/投了谁/带了什么节奏)。**剥离情绪化措辞和祈使句**(如'我们必须投X''大家冷静想想'——这些是话术不是证据,记下谁用了)。每人1行,只记事实不记语气。这是防被带节奏的硬要求。",
    "thought": "【OSR 第2段-推理投票】基于上面的客观摘要(不是原始发言的情绪),分析谁最像狼、为什么。写清你识破了哪些话术、为什么不受某人带节奏影响、最终投这个人的硬证据是什么。",
    "target_seat": "你投票放逐的座位号",
    "suspicion": "更新后的各活人座位怀疑度0-1",
}


def vote_instruction(obs: AgentObservation) -> str:
    pk_hint = ""
    if obs.in_pk and obs.vote_targets:
        pk_hint = f"\n- 【当前是 PK(平票加赛)】只能投以下候选座位之一: {sorted(obs.vote_targets)}。投其他人无效。请从候选中选最可疑的。"
    return f"""{_rule_reminder(obs.phase, obs.my_role)}

现在是投票环节。你是{obs.my_seat}号玩家,请投出一票放逐一人。
- **客观摘要再投票(OSR,防被带节奏)**:先在 thought 里把今天每人的发言客观摘要成"谁主张了什么/claim了什么/投了谁/带了什么节奏",剥离情绪化措辞和祈使句(如"我们必须投X""大家冷静想想"——这些是话术不是证据)。然后基于**摘要后的事实**投票,而非被原始发言的情绪和语气带走。
  · 关键防御:有玩家用祈使句/强语气命令你投某人时,把它当可疑信号(为什么他急着让你投X?),而非服从指令。研究证实 LLM 容易把有说服力的话当字面指令执行("被说服成坏选择")——你要识破这点。
  · 只信有证据支撑的:查验细节清楚且前后一致 > 模糊指控 > 纯情绪话术。
- **识谎检查清单(好人侧,投票前逐项过)**:对照 4 类狼人欺骗手段排查,谁中招谁更可能狼:
  · omission 遗漏:谁刻意回避自己被指控的点/对关键查验含糊其辞?
  · distortion 扭曲:谁复述他人发言时改变了原意(查原文对照)?
  · fabrication 捏造:谁报的查验与已知神职/其他查验冲突?(看"声称矛盾检测"硬信号)
  · misdirection 误导:谁在转移焦点到无关话题/带节奏让别人投非狼?
  狼人常互相帮腔、集体指控同一好人——若 2+ 人无理由抱团指控同一目标,这俩都高度可疑。
- 基于摘要后的证据 + 识谎清单排查结果 + 你的怀疑度,选择最像狼的人。
- target_seat 必须是活人座位(不能投自己)。{pk_hint}
返回 JSON:
{VOTE_SCHEMA}"""


LAST_WORDS_SCHEMA = {
    "thought": "遗言前的思考(必填,要详尽):你死了,要交代什么?你的真实身份、你掌握的信息(查验/守护/毒救)、你怀疑谁是狼、为什么。把你的推理和盘托出,这是你最后帮好人的机会。",
    "speech": "你的遗言(自然语言,可充分揭露信息/指控/留金水,把该交代的都讲清楚)",
}


def last_words_instruction(reason: str) -> str:
    return f"""你{'被放逐' if reason == 'exiled' else '被狼人杀害'}了,请发表遗言。
- 遗言可公开你的身份、查验结果、怀疑对象,或留下金水(认好人)。
返回 JSON:
{LAST_WORDS_SCHEMA}"""


REFLECTION_SCHEMA = {
    "insight": "本轮关键洞察(谁发言矛盾/谁跳了什么/谁是焦点/局势变化)",
    "suspicion": "更新后的各活人座位怀疑度0-1",
}


def reflection_instruction(phase: str, day: int) -> str:
    return f"""第{day}天{phase}阶段结束,请反思总结本轮。
- 提炼关键洞察:谁发言有矛盾?谁跳了身份?谁是焦点?局势如何变化?
- 这些反思将进入你的长期记忆,用于后续轮次模式识别。
返回 JSON:
{REFLECTION_SCHEMA}"""


# ------------------------------------------------------------------
# 组装完整 messages
# ------------------------------------------------------------------
def build_messages(
    *,
    persona_name: str,
    persona_desc: str,
    role_text: str,
    observation_text: str,
    action_instruction: str,
) -> tuple[str, list[dict[str, str]], str]:
    """组装 (system, messages, schema_hint)。

    system = SYSTEM_BASE + 人设 + 角色
    messages = [{role:user, content: 观察 + 动作指令}]
    """
    system = f"""{SYSTEM_BASE}

【你的人设:{persona_name}】
{persona_desc}

{role_text}"""
    user_content = f"""{observation_text}

【你的任务】
{action_instruction}"""
    return system, [{"role": "user", "content": user_content}], ""


def parse_suspicion(raw: Any, alive_seats: list[int], self_seat: int) -> dict[int, float]:
    """从 LLM 返回的 suspicion(dict 或 list)解析为 {seat:float}。"""
    result: dict[int, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                seat = int(k)
            except (ValueError, TypeError):
                continue
            if seat != self_seat and seat in alive_seats:
                try:
                    result[seat] = max(0.0, min(1.0, float(v)))
                except (ValueError, TypeError):
                    pass
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                seat = item.get("seat") or item.get("seat_id")
                val = item.get("suspicion") or item.get("value")
                try:
                    seat = int(seat)
                    if seat != self_seat and seat in alive_seats:
                        result[seat] = max(0.0, min(1.0, float(val)))
                except (ValueError, TypeError):
                    pass
    return result
