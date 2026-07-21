# Repository Guide

This repository implements an auditable agent harness. Werewolf is the current environment. It is not a chat product and must not grow a second, chat-shaped production decision path.

## Non-negotiable invariants

1. Every production action starts as an immutable `ActionRequest` and reaches exactly one terminal row: a `DecisionEnvelope` or a request-linked structured failure.
2. The environment owns observations, legal actions, legal targets, deadlines, rules, visibility and failure recording. It does not choose strategy for an agent.
3. LLM and human input are the only production decision sources. Scripted behavior belongs in tests only.
4. Public speech is the exact `decision.speech` from the same envelope. Do not add a rewrite call, censorship pass, completion pass or fake streaming layer.
5. `SKIP`, timeout, invalid target, parse failure and provider failure remain distinguishable. Never convert them into a plausible speech, vote or target.
6. A legal bluff, role claim, self-incrimination or deliberate silence is agent behavior. The harness validates protocol and game legality, not whether a strategy looks sensible.
7. Model-reported suspicion, confidence, deception, attitude or summary is not independent truth. Do not expose it as an audit, posterior, calibration score or objective metric.
8. Private reasoning, hidden roles, team-only observations and credentials must not enter public/player projections.
9. Replay is a read-only projection of an ended run's immutable transcript. It is not a ReplayAgent and does not re-execute the game.
10. Credentials never enter manifests, artifacts, logs, tests, screenshots or documentation. Use `WEREWOLF_*` configuration only.

## Production decision path

```text
GameOrchestratorV2
  → ActionRequest
  → DecisionRuntime
  → AgentActor.decide or human decision wait
  → DecisionEnvelope | linked response failure
  → validate_decision_against_legal_actions
  → RulesEngine
  → Transcript
```

The only production Agent entry point is:

```python
async def decide(self, request: ActionRequest) -> DecisionEnvelope
```

Do not restore legacy `decide_speak`, `decide_vote`, `decide_night_action`, replay factories or scripted fallbacks in production code. Test doubles may use helper methods internally when their `decide(request)` remains the exercised boundary.

## Ownership map

- `src/harness/agent_protocol.py`: versioned request/envelope contract.
- `src/harness/decision_runtime.py`: the single request deadline, validation and terminal-trace boundary.
- `src/harness/agents.py`: identity, action, skip and target validation.
- `src/game/orchestrator.py`: environment scheduling and request construction.
- `src/game/rules.py`: deterministic state transition and legality.
- `src/agent/actor.py`: LLM/Human protocol adapter; no `GameState` crosses this boundary.
- `src/agent/memory.py`: facts actually observed by one seat plus recorded public claims.
- `src/llm/router.py`: standard provider protocols, transport retry and factual call stats.
- `src/harness/transcript.py`: immutable ordered rows and redaction.
- `src/harness/runner.py`: unattended real-model runs; no human or scripted branch.
- `src/api/room_manager.py`: interactive room lifecycle and audience projection.
- `frontend/src/components/HarnessConsole.tsx`: protocol/environment console, not chat UI.

## Retry ownership

- Router retries retryable transport/provider failures (`408`, `409`, `425`, `429`, selected `5xx`, connection and timeout errors).
- Actor does not reissue a request after Router exhausts that budget.
- Actor may request a new model response for `LLMResponseError` or schema/required-field failure.
- All exhausted paths end in `AgentDecisionError`; they do not synthesize a Decision.

This separation prevents Router retries and Actor response attempts from multiplying into an uncontrolled number of external calls.

## Trace and metric semantics

`decision_count` means validated decisions consumed by the environment, including explicit `SKIP`. It is not the number of trace rows.

`decision_trace_metrics` separately counts:

- request rows;
- envelope response rows and no-envelope failure rows;
- terminal, unpaired, duplicate-terminal and orphan-terminal counts;
- consumed decision rows;
- rules resolution rows;
- total trace rows.

Batch output may aggregate state outcome, elapsed time, days, model calls, success/failure/retries, token usage, latency, accepted parse recoveries, rejected response categories and decision failures. Do not add inferred social truth or model-graded quality to production summaries.

## Change checklist

When changing the decision protocol:

1. Update Pydantic models and validators.
2. Update orchestrator request construction and rules consumption.
3. Update transcript/visibility tests.
4. Update TypeScript event and analysis types.
5. Update Harness Console rendering if the event is user-visible.
6. Update `docs/PROTOCOL.md`.

When adding an event, decide explicitly whether it is public, player-private, team/god or admin-only. Add it to fail-closed allowlists only after that decision. A missing allowlist entry should hide the event, not leak it.

When changing artifacts, preserve the three-file run output unless the change is explicitly justified:

```text
manifest.json
summary.json
transcript.jsonl
```

## Validation commands

```bash
source .venv/bin/activate
PYTHONPATH=. python -m pytest -q

cd frontend
npx tsc -b --pretty false
npm run build
```

Interactive UI work also requires browser validation. A real-model claim additionally requires Router calls greater than zero and matching `agent_request` / `agent_response` rows in the transcript.

## Documentation claims

Keep descriptions tied to code or generated artifacts. Papers in `docs/REFERENCES.md` are context only. They do not prove this implementation reproduces a paper, improves a metric or has an independent deception detector.
