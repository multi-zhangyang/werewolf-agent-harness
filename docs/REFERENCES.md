# References and Claim Boundaries

These publications are real background references for agent evaluation, multi-agent coordination, memory-oriented interactive agents and social-deduction environments. They do not prove that this repository reproduces their systems, achieves their reported results or improves any metric.

## Agent evaluation

### AgentBench

- Title: *AgentBench: Evaluating LLMs as Agents*
- arXiv: [2308.03688](https://arxiv.org/abs/2308.03688)
- Initial submission: 2023-08-07; later version associated with ICLR 2024.

Relevant here: evaluating Agents through environment interaction motivates an explicit environment/action boundary and inspectable trajectories.

Not established here: Werewolf Agent Harness is not an AgentBench task implementation, and no AgentBench score is reported.

## Multi-agent application frameworks

### AutoGen

- Title: *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*
- arXiv: [2308.08155](https://arxiv.org/abs/2308.08155)

Relevant here: it is background for systems coordinating multiple model-backed agents.

Not established here: this repository does not use AutoGen as its runtime and is intentionally not organized as free-form Agent chat.

### CAMEL

- Title: *CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society*
- arXiv: [2303.17760](https://arxiv.org/abs/2303.17760)

Relevant here: it provides context for role-conditioned interaction among multiple language-model agents.

Not established here: role prompting alone is not evidence of emergent social behavior, coordination quality or faithful CAMEL reproduction.

## Interactive agents and memory

### Generative Agents

- Title: *Generative Agents: Interactive Simulacra of Human Behavior*
- arXiv: [2304.03442](https://arxiv.org/abs/2304.03442)

Relevant here: it is background for maintaining Agent experience across an interactive environment.

Not established here: `AgentMemory` is a delivered-fact timeline plus recorded
public claims. The separate `PrivateAgentState` persists subjective belief,
strategy and accepted public commitments, but it still does not reproduce the
paper's retrieval/reflection architecture.

## Social deduction evaluation

### Werewolf Arena

- Title: *Werewolf Arena: A Case Study in LLM Evaluation via Social Deduction*
- arXiv: [2407.13943](https://arxiv.org/abs/2407.13943)
- Initial submission: 2024-07-18.

Relevant here: it establishes social deduction as a real case-study setting for evaluating model behavior.

Not established here: this repository does not claim benchmark equivalence, reproduced results, independent deception detection or a validated Werewolf quality score.

### Communication-game Werewolf agents

- Title: *Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf*
- arXiv: [2309.04658](https://arxiv.org/abs/2309.04658)
- Official code: [xuyuzhuang11/Werewolf](https://github.com/xuyuzhuang11/Werewolf)

Relevant here: separate players, visibility-scoped messages, retrieval and
reflection support persistent per-player context rather than a central model
speaking for every seat.

Not established here: this repository does not reproduce its retrieval,
reflection or reported results.

### Strategic language agents

- Title: *Language Agents with Reinforcement Learning for Strategic Play in the Werewolf Game*
- arXiv: [2310.18940](https://arxiv.org/abs/2310.18940)
- Publication: ICML 2024.

Relevant here: it separates organized information/role deduction, diverse
candidate generation and final strategy selection, and documents fixed-action
bias in direct LLM decisions.

Not established here: one model response compares candidate strategies and
persists a selected plan. This is not the paper's RL selector, population
training or win-rate result.

### Recursive perspective-taking

- Title: *Avalon's Game of Thoughts: Battle Against Deception through Recursive Contemplation*
- arXiv: [2310.01320](https://arxiv.org/abs/2310.01320)

Relevant here: first-order opponent modeling and second-order reasoning about
how others perceive the Agent motivate the explicit private belief/public
speech split and `perceived_image` state.

Not established here: one structured decision response is not a reproduction
of the paper's multi-pass ReCon method.

### Structured probabilistic social deduction

- Title: *Bayesian Social Deduction with Graph-Informed Language Models*
- arXiv: [2506.17788](https://arxiv.org/abs/2506.17788)
- Version consulted: v2, 2026-04-10.

Relevant here: it provides evidence for externalizing constrained hidden-role
beliefs while retaining an LLM for language priors and interaction.

Not established here: the capped-simplex role-count projection is not GRAIL's
learned factor graph, max-product inference, calibration or human-study result.

### Longitudinal deception evaluation

- Title: *WOLF: Werewolf-based Observations for LLM Deception and Falsehoods*
- arXiv: [2512.09187](https://arxiv.org/abs/2512.09187)
- Official code: [MrinalA2009/WOLF-Werewolf-based-Observations-for-LLM-Deception-and-Falsehoods](https://github.com/MrinalA2009/WOLF-Werewolf-based-Observations-for-LLM-Deception-and-Falsehoods)

Relevant here: it motivates private scratchpads, longitudinal per-observer
belief traces and measuring deception production separately from detection.

Not established here: model self-reports are not treated as factual deception
labels, and no WOLF score is reported.

## Rules for future citations

When adding a reference:

1. Link to the primary paper or official project source.
2. Separate “why it is relevant” from “what this repository actually implements.”
3. Do not infer a performance claim from architectural resemblance.
4. Do not call a model self-report an independent evaluator.
5. Keep experimental claims attached to a versioned `RunSpec`, immutable transcript, raw factual summary and a documented analysis method.
