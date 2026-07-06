# Protocol

This document summarizes the current REST and WebSocket contract used by the React frontend. `frontend/src/lib/types.ts` is the practical frontend type mirror; backend behavior lives in `src/api/server.py`, `src/api/room_manager.py`, and `src/game/orchestrator.py`.

## REST

### `GET /api/config`

Returns public model configuration metadata. `api_key` must always be an empty string. Use `api_key_configured` to show whether the backend has a key.

### `GET /api/providers`

Returns supported provider metadata for the model config UI.

### `POST /api/rooms`

Creates a room.

Important fields:

- `player_names`: ordered seat names.
- `human_seats`: optional list of one-based human-controlled seats.
- model config fields: optional room defaults.

Unsupported deck configuration is rejected instead of silently ignored.

### `GET /api/rooms/{room_id}`

Returns room status and public player data. Running rooms must not expose hidden roles. Ended rooms may expose roles for review.

### `POST /api/rooms/{room_id}/start`

Starts a waiting room. The backend broadcasts `room_status: running` and then normal phase events.

### `GET /api/rooms/{room_id}/replay`

Allowed only after the room has ended. Returns event history, thinking history, and latest analysis for replay/research views.

## WebSocket

Path:

```text
/ws/rooms/{room_id}?mode=spectate|play|god|replay&seat=1
```

Modes:

- `spectate`: public live view.
- `play`: human player view for one seat.
- `god`: live full-observer/debug view.
- `replay`: ended-game replay only.

## Event Union

The frontend expects the event union in `frontend/src/lib/types.ts`.

Common public live events:

- `snapshot`
- `room_status`
- `phase_started`
- `night_resolved`
- `speech`
- `vote_cast`
- `vote_resolved`
- `vote_incomplete`
- `last_words`
- `hunter_shot`
- `agent_thinking` with sanitized summary
- `agent_decision_failed`
- `game_ended`
- `analysis` after game end
- `game_error`

God/replay-only or restricted events:

- `trust_update`
- `reflections_update`
- `wolf_caucus`
- `wolf_caucus_consensus`

Play-only restricted event:

- `human_action_request`

## Visibility Rules

Running public/player views must not receive hidden role truth, private reasoning, wolf caucus, raw provider errors, or replay-only analysis fields before the game ends.

Private events may use `visibility` and `recipients`; public events are additionally filtered by the room manager's public allowlist. Keep both paths in sync when adding events.

## Human Action Protocol

When the backend emits `human_action_request`, the `action_type` is concrete:

- `night_kill`
- `see`
- `save`
- `poison`
- `guard`
- `hunter_shot`
- `vote`

The frontend may send only the requested payload fields, such as `target_seat`. It should not have to duplicate the action type to make the backend understand the request.

## Post-Game Analysis

`analysis` is a post-game truth and research object. It may contain:

- winner, days, seats, roles, teams, and death reasons.
- `quality`
- `parse_metrics`
- `decision_failure_metrics`
- `dialogue_metrics`
- `debate_process_metrics`
- `objective_metrics`
- `posterior_metrics`
- `posterior_trace`
- `deception_audit`
- `collusion_audit`

It is suitable for ended-game replay and summaries. Do not feed truth-only analysis back into live agents.
