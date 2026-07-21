# Security Policy

## Supported Use

werewolf-mas is designed for local development and controlled demonstrations. The default server should be run on `127.0.0.1` with exactly one API worker. Do not expose the backend directly to the public internet without adding service-level authentication, rate limits, spend controls, a logging policy, and a reverse proxy that you control.

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
- Role/team-private observations to unauthorized viewers.
- Private reasoning or raw model output to any unauthorized audience. Authorized
  god/replay users receive reasoning only from the admin-capability trace endpoint.
- Admin-only protocol traces or ended-run truth analysis before authorization permits them.

Post-game `analysis`, trace, and replay require the room admin token. Replay is available only after the room ends.

## Public Deployment Warning

The API accepts only the exact browser origins listed in `WEREWOLF_CORS_ORIGINS`; the local defaults are the Vite origins on `127.0.0.1:5173` and `localhost:5173`. Entries must be comma-separated HTTP(S) origins without paths, credentials, or wildcards. `WEREWOLF_CORS_ALLOW_CREDENTIALS` defaults to `false`. CORS is a browser boundary, not authentication, and does not replace reverse-proxy policy for WebSocket upgrades or non-browser clients.

WebSocket upgrades with an `Origin` header must match that same exact allowlist. `WEREWOLF_WS_ALLOW_MISSING_ORIGIN` makes the native-client exception explicit and defaults to `true` for local/TestClient compatibility; set it to `false` for browser-only public deployments and enforce the policy again at the reverse proxy.

`WEREWOLF_MAX_ROOMS` bounds retained rooms. Expired terminal rooms can be reclaimed after `WEREWOLF_TERMINAL_ROOM_TTL`, while an authenticated `DELETE /api/rooms/{room_id}` explicitly removes an idle room. Running rooms and rooms with active connections are never silently evicted.

REST requests and WebSocket connections/messages use separately configured process-local token buckets (`WEREWOLF_REST_RATE_LIMIT_*` and `WEREWOLF_WS_RATE_LIMIT_*`). Provider transport attempts use per-run/per-room call and reported-token budgets (`WEREWOLF_PROVIDER_BUDGET_*`). A token-limited scope blocks when provider usage is missing; usage that exceeds a reservation is still charged and the response is rejected. OpenAI paths do not advertise a provider-enforced total-token cap, so a hard token budget rejects those calls before transport rather than treating an output setting as an input-plus-output bound. Anthropic's `max_tokens` bounds output only; reported input plus output is charged after the call and an overrun is explicit. Call budgets remain the portable pre-transport spend bound across all supported protocols.

The entire interactive API, not only these counters, currently requires one worker. `RoomManager` owns live room objects, game tasks and locks, capability state, WebSocket clients and authorization-specific delivery streams/cursors in process; it also coordinates SQLite snapshot writes. SQLite persistence supports restart recovery for one owner but does not provide distributed room ownership, locking, WebSocket fan-out or cross-process reconnect ordering. Multiple workers can disagree about whether a room exists, split REST and WebSocket traffic, lose broadcasts, or mutate the same persisted room concurrently. Multi-worker or multi-host deployments therefore need an external room owner/coordinator, shared state and locks, cross-process pub/sub and cursor storage, routing guarantees, an atomic admission limiter/provider ledger, and trusted proxy configuration. Forwarded client-address headers are deliberately not trusted by the application middleware.

`/healthz` reports process liveness. `/readyz` reports only bounded RoomManager and Router lifecycle states and returns `503` while the service is draining or unavailable. Neither endpoint probes a paid model API or returns credentials, endpoints, room identifiers, or usage data.

The development app is not a hosted multi-tenant service. Before public deployment, add:

- Authentication.
- Per-room authorization.
- Deployment-specific browser and WebSocket origin policy.
- Distributed request and WebSocket rate limits.
- Distributed room ownership, locking, persistence coordination, WebSocket
  fan-out and reconnect cursors before enabling more than one API worker.
- Server-side log redaction.
- HTTPS termination.
- Deployment-wide provider spend controls.
- Abuse monitoring.
