# Contributing

Werewolf Agent Harness separates environment authority from Agent choice. Contributions must preserve that boundary and keep claims tied to executable evidence.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

Or use `make install`.

Run development services in separate terminals:

```bash
make dev-api
make dev-ui
```

## Required invariants

- Production decisions must pass through `DecisionRuntime` and
  `AgentProtocol.decide(ActionRequest)`, ending in one `DecisionEnvelope` or one
  request-linked structured failure.
- LLM and authenticated human input are the only production decision sources.
- Do not add scripted fallback speeches, votes, night actions, hunter shots, fake provider responses or production actor factories.
- Do not add a second model call to rewrite, complete, summarize or censor public output.
- Do not split an already completed string into fake streaming deltas.
- Legal strategy is not a protocol violation. Bluffing, false role claims, self-incrimination and silence may be intentional Agent behavior.
- Do not silently replace an illegal/unresolved target with a different legal target.
- Preserve the difference between skip, protocol rejection, rules rejection, response failure, provider failure and timeout.
- Do not present model self-reports as independent trust, suspicion, deception, posterior, calibration or quality truth.
- Keep replay read-only. It must not instantiate replay Agents or mutate the original run.
- Treat summary JSONL as a checkpoint/cache. Derived evaluations require an
  in-process validated result or `load_verified_run_summary()` over the
  committed artifact set; public digest/boolean fields never self-attest.

## Information isolation

Every new event must have an explicit audience decision. Projection allowlists are fail-closed.

Running public/player views must not expose:

- hidden role or team truth;
- another seat's private observation;
- private reasoning or raw prompt/response bodies;
- admin/seat tokens or model credentials;
- ended-run omniscient analysis before authorization permits it.

Public claims remain claims. Recording “seat 2 claimed seer” must not turn it into “seat 2 is seer.”

## Retry and failure changes

Router owns transport/provider retry. Actor owns fresh attempts for model response/schema errors. Do not reintroduce nested retry multiplication.

If a failure budget is exhausted, emit/record failure and let the environment resolve the missing action according to explicit rules. Never manufacture a plausible Agent choice.

## Protocol changes

When changing `ActionRequest`, `DecisionEnvelope`, `Decision`, legal-action validation or events:

1. update Pydantic schemas and validators;
2. update request construction and RulesEngine consumption;
3. update transcript/visibility tests;
4. update `frontend/src/lib/types.ts` and reducer handling;
5. update Harness Console rendering where relevant;
6. update `docs/PROTOCOL.md`;
7. run backend and frontend validation.

Do not add fields merely because a model can generate them. State who owns the field, whether it is observation, choice, environment truth or provenance, and who is allowed to see it.

## Tests

Full validation:

```bash
make test
```

Equivalent commands:

```bash
source .venv/bin/activate
PYTHONPATH=. python -m pytest -q

cd frontend
npx tsc -b --pretty false
npm run build
```

The opt-in real-browser matrix runs both replay and live spectator/player/God
journeys and closes its browser/server sessions on exit:

```bash
make test-browser
```

Unit and integration tests must not call an external model. Test-local protocol Agents may return deterministic decisions, but they must exercise the same `decide(request)` boundary.

For an intentional real-model check, configure `WEREWOLF_*` locally and run:

```bash
make harness-real HARNESS_SEED=100 HARNESS_RUNS=1
```

This spends real API quota. A result counts as real-model validation only when Router calls are greater than zero and transcript request/response IDs match. Do not commit generated artifacts or local logs by default.

## Frontend

The frontend is a Harness Console, not a chat client. UI changes should make environment state, request scope, exact Agent output, rules resolution and failure provenance easier to inspect.

Interactive changes require browser validation in addition to the build. Verify at least spectator and authorized god/admin projections; human-action changes also require a `play` seat check.

## Secrets

- Use only `WEREWOLF_*` variables or an ignored local `.env`.
- Never paste a real key into a source file, shell command, fixture, artifact, screenshot, issue or PR.
- `/api/config` may return only `api_key_configured`, never a key fragment.
- Structured manifest endpoints may retain a safe path; arbitrary provider error URLs must be reduced to origin.

## Documentation and research claims

Describe only behavior supported by current code/tests/artifacts. References in `docs/REFERENCES.md` are background, not evidence that this repository reproduces a paper or improves a metric. Do not revive deleted experimental claims without a new, inspectable experiment design and raw factual artifacts.
