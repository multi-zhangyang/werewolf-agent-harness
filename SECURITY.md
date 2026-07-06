# Security Policy

## Supported Use

werewolf-mas is designed for local development and controlled demonstrations. The default server should be run on `127.0.0.1`. Do not expose the backend directly to the public internet without adding authentication, rate limits, CORS restrictions, logging policy, and a reverse proxy that you control.

## Reporting Vulnerabilities

If you find a vulnerability, open a private report through the hosting platform if available. If private reporting is not available, open a minimal public issue that describes the affected area without posting secrets, exploit payloads, API keys, or private model gateway details.

Include:

- Affected version or commit.
- Reproduction steps.
- Expected and actual behavior.
- Whether credentials, hidden roles, private reasoning, or local files can be exposed.

## Secrets

Model credentials must be provided through `WEREWOLF_*` environment variables or a local `.env` file that is never committed.

The API config endpoint must never return API key prefixes, suffixes, masks, or fragments. It should only expose whether a key is configured.

Do not paste `.env` contents, provider tokens, local gateway keys, or credential fragments into issues, PRs, logs, screenshots, or documentation.

## Information Boundaries

Running games must not leak:

- Hidden roles to public spectators or players.
- Wolf caucus events to non-god live viewers.
- Private reasoning or raw model output.
- Replay-only truth analysis before the game has ended.

Post-game `analysis` and replay views intentionally reveal truth for review and research.

## Public Deployment Warning

The development app is not a hosted multi-tenant service. Before public deployment, add:

- Authentication.
- Per-room authorization.
- Strict CORS.
- Request and WebSocket rate limits.
- Server-side log redaction.
- HTTPS termination.
- Provider spend controls.
- Abuse monitoring.
