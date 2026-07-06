# Contributing

werewolf-mas is a real LLM multi-agent social-deduction project. Contributions should preserve the core boundary: the harness schedules and validates, while agents make their own LLM-backed decisions.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

You can also use:

```bash
make install
```

## Development

Run the backend and frontend in separate terminals:

```bash
make dev-api
make dev-ui
```

Open `http://localhost:5173` for Vite development. For the single-process production-style path:

```bash
make build-ui
make dev-api
```

Then open `http://localhost:8000`.

## Tests

Before opening a PR, run:

```bash
make test
```

For backend-only changes:

```bash
PYTHONPATH=. pytest -q
```

For frontend-only changes:

```bash
cd frontend
npm run build
```

`tests/smoke_e2e.py` is a real LLM smoke test. It requires valid `WEREWOLF_*` model configuration and will make real API calls:

```bash
make smoke-real
```

Do not describe a run as smoke or LLM validation unless it actually called the configured model.

## No-Fallback Rule

- Do not add scripted backup speeches, votes, night actions, hunter shots, or fake model responses.
- If a model call fails after retries, surface a transparent failure event.
- Do not replace agent decisions with harness guesses.
- Do not weaken information isolation between public, private, god, and replay views.

## Frontend UI

The frontend uses real shadcn/ui components generated under `frontend/src/components/ui`. Use those components for controls, cards, badges, tabs, dialogs, accordions, scroll areas, progress bars, inputs, and tooltips. Do not create fake shadcn-looking controls in business components.

Every frontend change should pass:

```bash
cd frontend
npm run build
```

Interactive UI changes should also be checked in a real browser.

## Protocol Changes

When changing REST or WebSocket payloads:

- Update backend tests.
- Update `frontend/src/lib/types.ts`.
- Update reducer behavior in `frontend/src/lib/store.ts` if needed.
- Update `docs/PROTOCOL.md`.
- Verify that running-game views do not expose role truth, wolf caucus, private reasoning, or replay-only analysis before the game ends.

## Secrets

Never commit `.env`, API keys, credential fragments, local logs containing secrets, or generated archives. Use `.env.example` for placeholders only.
