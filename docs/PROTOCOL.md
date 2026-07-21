# Protocol

This document describes the internal Agent contract, transcript rows, REST surface and WebSocket modes. The Pydantic/TypeScript schemas in the repository remain authoritative.

## 1. Agent protocol

Current version:

```text
werewolf.harness.agent_protocol.v2
```

The remainder of this first section describes the legacy Werewolf-compatible
seat protocol. The generic Core protocol is separate and exact-versioned as
`agent-harness.decision.v1`; it does not use seats, phase names or a
Werewolf `Decision` schema.

The environment calls exactly one production method:

```python
async def decide(request: ActionRequest) -> DecisionEnvelope
```

### `ActionRequest`

```json
{
  "protocol_version": "werewolf.harness.agent_protocol.v2",
  "request_id": "opaque-request-id",
  "run_id": "run-100-0001",
  "seat": 3,
  "phase": "voting",
  "day": 2,
  "action_kind": "vote",
  "observation": {
    "my_seat": 3,
    "phase": "voting",
    "alive_seats": [1, 2, 3, 5],
    "today_speeches": []
  },
  "legal_actions": [
    {
      "action": "vote",
      "target_seats": [1, 2, 5],
      "target_required": true,
      "can_skip": false,
      "metadata": {}
    }
  ],
  "deadline_monotonic": 12345.5,
  "private_context": {},
  "metadata": {
    "deadline_source": "decision",
    "effective_timeout_seconds": 30.0,
    "agent_context": {
      "context_version": "werewolf.agent-context.v1",
      "memory_digest": "sha256-hex",
      "private_state_digest": "sha256-hex"
    },
    "visible_event_ids": ["run:event:000001"],
    "team_event_ids": [],
    "deadline_owner": "decision_runtime"
  }
}
```

Properties:

- The model is frozen after construction.
- `observation` contains only information projected for that seat.
- `legal_actions` is the environment's declared action/target scope.
- `target_required` distinguishes a target-free action from a targeted action
  whose current legal target set is empty. In the latter case a non-skip
  response is invalid; `SKIP` is available only when `can_skip=true`.
- `deadline_monotonic` is the minimum of the per-decision deadline and wider phase deadline. It is a process-local monotonic deadline, not a portable wall-clock timestamp.
- `private_context` may contain environment facts required for this decision, such as a last-words reason. It is never public by default.
- `metadata.deadline_source` identifies whether the effective deadline came from the decision budget or the enclosing phase. `DecisionRuntime` adds `deadline_owner=decision_runtime` to the traced request so deadline-aware adapters do not race it with another wall-clock owner.

Current action kinds include `speak`, `vote`, `last_words`, `wolf_council`,
`night_kill`, `see`, `guard`, `save`, `poison` and `hunter_shot` as applicable
to role and phase. `wolf_council` carries an exact `team_message` and tentative
target to living wolves only; every living wolf later receives a separate
`night_kill` request.

### Seat-bound tool contract

The first tool-loop user message and `get_legal_actions` both include
`visible_seats` and `alive_seats`. They are environment-projected sets, not
model-supplied identity fields. The model must not invent a seat outside
`visible_seats`.

Tool JSON Schemas are request-scoped:

- public vote filters and claim-consistency reads accept only visible seats;
- belief updates, `checked_seat`, `reply_to` and `accuses` accept only other
  visible seats;
- terminal `target_seat` accepts exactly the target set on the matching
  `LegalAction` (`kill`/`hunter_shot` resolve to the `night_kill` action name);
- a seat outside those sets fails as `invalid_arguments` before a handler runs;
- an empty target set uses a valid non-enumerated integer schema rather than an
  illegal empty enum, and terminal preflight/environment validation still
  rejects a non-skip target. `skip` remains a separate terminal tool and exists
  only when `can_skip=true`.

The Werewolf registry exposes `read_turn_context` for the common path where a
decision needs more than the request envelope. It returns one bounded,
seat-private snapshot containing the exact legal actions/targets, private facts,
recent public events/votes/claims, subjective belief and plan state, and this
seat's accepted commitments. The snapshot is recursively bounded and redacted;
the granular read tools remain available for details outside that window.
`update_private_state` atomically replaces this seat's belief, candidate/selected
plan, cover and deception state. It is preferred over separate belief/plan
updates when those fields change together; it never writes another seat's state.

`read_public_events.limit` defaults to 12 and has a hard schema/handler maximum
of 24. `memory_window` contains only `speech` and `last_words`, excluding copies
already represented by current `events` or `today_speeches`; a historical last
statement absent from the current observation remains available. `get_beliefs`
does not return `commitments`; exact accepted public commitments are returned
only by `get_commitments`.

### `DecisionEnvelope`

```json
{
  "protocol_version": "werewolf.harness.agent_protocol.v2",
  "request_id": "opaque-request-id",
  "seat": 3,
  "decision": {
    "action": "vote",
    "target_seat": 2,
    "speech": null,
    "team_message": null,
    "reasoning": "private reasoning",
    "bid": null,
    "claim": null,
    "reply_to": null,
    "accuses": null,
    "skip_reason": null
  },
  "latency_seconds": 1.25,
  "model_call_id": "safe-call-id",
  "prompt_hash": "sha256-hex",
  "response_hash": "sha256-hex",
  "parse_status": "ok",
  "metadata": {
    "agent_kind": "llm",
    "provider": "openai",
    "model": "model-id"
  }
}
```

`reasoning` is private. It remains inside the admin-authorized envelope trace and is never copied into a public/player/god event stream.

`parse_status` describes the response that actually produced this envelope:

- `ok`: the provider response was a complete JSON object;
- `recovered`: the response required a lossless parser recovery, such as removing a Markdown fence, reading a Python-style object literal, extracting an embedded complete object, or adding only a missing closing brace;
- `not_applicable`: a human or other non-JSON agent supplied the decision.

An unparseable, incomplete, or lossy response does not produce a `DecisionEnvelope`. The Router records those response attempts separately as `response_parse_failures`, `incomplete_responses`, or `lossy_parse_rejections`; the Agent may make a bounded fresh response attempt, and final exhaustion becomes `agent_decision_failed`. Parse failures are therefore never represented as a boolean on an otherwise valid `Decision`.

The `Decision` schema rejects unknown fields. Model-authored beliefs and plans
are validated as Actor-private `private_state` response data and are not copied
into the environment action envelope. Removed audit-like fields such as
`suspicion`, `attitudes`, `deception` or `objective_summary` remain invalid on
`Decision` rather than being presented as independent measurements.

### Validation

Before a decision reaches game rules, the harness validates:

1. protocol version;
2. matching request ID;
3. matching seat;
4. action membership in `legal_actions`;
5. whether `SKIP` was advertised as legal;
6. whether an explicit `SKIP` has a reason and carries no executable/public payload;
7. whether the target is present exactly when an advertised target set requires it;
8. whether `SPEAK`/`LAST_WORDS` carry required exact text and whether `SPEAK` has `bid=1..4`;
9. whether non-speech actions improperly carry speech, bid, claim, reply or accusation fields;
10. whether only `wolf_council` carries non-empty `team_message` and never public `speech`.

The response trace records validation as accepted/rejected with concrete issue codes. Rejection is not repaired by picking a different action or target. If the validator itself raises or returns an invalid result, the Harness records `agent_response_validation_failed` with the received envelope and a structured validator failure. That is a Harness defect, not an Agent failure, and it does not create a second `agent_response_failed` row.

### Public output and `SKIP`

For accepted speech/last-words decisions, the public string is exactly the non-empty `decision.speech` from that envelope.

When skipping is legal, an Agent may return:

```json
{
  "action": "skip",
  "skip_reason": "agent_declined"
}
```

A skipped speech creates a rules trace but no public `speech` event. A skipped last-words opportunity produces a `last_words_skipped` environment event with `skip_reason` and no `text` field. This distinguishes an environment resolution from an Agent quote.

### Core Tool Protocol

The generic contract uses `CoreActionRequest` / `CoreDecisionEnvelope` aliases
from `src.harness.core_protocol`. A request has an opaque `actor_id`, a JSON
`observation`, exact `ActionOption(name, input_schema)` values, a `SkipPolicy`,
and optional deadline/labels/metadata. The environment validates identity,
legal action membership, JSON Schema arguments and skip policy with
`validate_decision_envelope` before consuming anything.

`CoreToolActor` converts those `ActionOption` schemas into a request-scoped set
of standard function tools. The model must use exactly one terminal function:
`submit_action_N` maps to the matching advertised action, and `submit_skip`
exists only when `SkipPolicy.allowed=true`. Calls use `tool_choice="required"`
and `parallel_tool_calls=false`. A malformed function name, call count or
argument object is a bounded response retry/failure, never a chat fallback or
an argument rewrite. No provider/model/endpoint-specific branch is involved.

Each Core tool Actor also owns a bounded private episodic memory containing
only its earlier authorized observations, request labels and submitted terminal
choices. The memory is passed back only to that same `actor_id`; it excludes
provider reasoning, raw output and trace records, and it never becomes a public,
player, god or other-Agent observation. A response-shape retry does not create
another memory entry because only the accepted terminal choice commits a turn.

For a successful Core tool decision, admin-only decision evidence is ordered as
`agent_turn_started`, `model_generation`, `tool_call_requested`, `tool_result`,
`agent_action_submitted`, the normal `agent_response`, then environment-owned
`decision_consumed` and `rules_result`. A failed/invalid response still has the
normal one request/one terminal pairing from `DecisionRuntime`, but has no
consumed choice. The nested `llm_call.actor_response_attempts` records each
provider-backed response attempt with request-bound call, usage and latency
provenance; it supports the same offline smoke verifier as the Werewolf tool
loop.

`council.cipher@1` uses this contract directly as its baseline. Its public event
names are custom and explicitly carry `visibility="public"`; a generic event is
not projected to clients merely because its name is unknown. Private role and
secret-commitment events carry `visibility="private"` and exact `recipients`.
The fixed missing-action rules are: no speech event, failed proposal attempt,
absent vote rather than fabricated rejection, and mission void/incomplete for
a missing secret commitment.

`council.cipher@2` adds an independent, simultaneous Cipher-only coordination
stage before each public proposal attempt. Each Cipher Actor receives a
`send_cipher_strategy_message` tool option; all eligible requests start
concurrently and each observation contains only prior delivered Cipher council
messages. The environment appends and emits nothing until every current-round
request has a terminal result. It then emits one private
`council_cipher_message` event for each accepted message, with `recipients`
equal to the exact current Cipher actor-ID set. These events are hard-classified
as private by projection policy: an accidental public label cannot expose them.
Council Actors and public clients receive neither the event nor its content.
The authorized god projection does receive omniscient environment events,
including this private event; decision rows remain absent from that projection.
A human God Console that presents the room admin capability may separately read
the bounded/redacted admin trace, which is never an Agent observation or a
WebSocket game event. A skip or failure is an absent message only: the
environment does not invent a message, route a public action-unavailable event,
or delegate the faction choice to a shared controller. `council.cipher@1`
intentionally has no such coordination stage.

## 2. Decision trace

Admin transcript rows of kind `decision` contain six Agent/Harness lifecycle payload shapes:

- `kind=agent_request`: the projected request and legal action scope;
- `kind=agent_response`: the envelope plus protocol validation result;
- `kind=agent_response_failed`: no envelope was produced; the row contains a
  structured failure linked to the request ID;
- `kind=agent_response_cancelled`: the owning room or run was cancelled while
  the request was active; it is terminal for pairing but is not an Agent or
  provider failure;
- `kind=agent_response_validation_failed`: an envelope was received, but the
  Harness validator raised or returned an invalid result; the row preserves
  the envelope, is terminal for pairing, and is not attributed to the Agent;
- `type=decision_consumed`: request-linked safe decision view and model-call provenance;

`type=rules_result` is a separate environment-adjudication row. It records a
request-linked accepted, rejected, skipped or not-selected rules resolution,
but it is not an Agent/Harness response terminal and is not used for request
pairing.

The Harness Console pairs request/response rows by `request_id`. Raw prompts and
raw provider envelopes are not returned; hashes, length, parse status, model
call ID and latency provide provenance. The admin-authorized AgentSession
subtrace may additionally expose bounded model content/reasoning and a bounded,
recursively credential-redacted copy of `tool_call_requested.arguments` beside
`arguments_hash`. Structured tool arguments are often the only observable plan
from a tool-only model. They are private admin evidence: another Agent, a player,
the public spectator projection and the ordinary god event stream never receive
them.

Before each provider call, `AgentSession` may create a rolling compacted copy of
its model history. Only older complete assistant tool-call + matching tool-result
groups are replaced, recent groups remain exact, and incomplete groups are kept
verbatim. The canonical session messages and admin audit rows remain complete.
The configured character value is a provider-input target rather than a promise
to truncate non-tool context; `limit_satisfied=false` is traced when safe atomic
compaction cannot reach it.

Attempt provenance is nested under the safe `llm_call` object. The parent
object carries safe request/response/reasoning hashes and
`api_base_fingerprint`; its `transport_attempts` rows record bounded Router
attempts using attempt number, status, latency, retry facts and optional HTTP
status or error type. `actor_response_attempts` records the Actor's bounded
fresh-response attempts as `accepted`,
`response_rejected`, or `provider_failed`, with an error type and nested safe
`llm_call` provenance when available. Exhausted no-envelope failures retain the
same sanitized Actor-attempt facts under `failure.llm_call_attempts`. These
structures contain no raw prompt, raw response, API key, or API-key
fingerprint.

Count definitions:

- `decision_count`: validated decisions consumed by the environment, including legal `SKIP`;
- `request_count`: emitted `ActionRequest` rows;
- `response_count`: `agent_response` rows, including envelopes whose completed validation result rejects them;
- `response_failure_count`: request failures that produced no envelope;
- `response_cancelled_count`: requests terminated by room/run cancellation;
- `response_validation_failure_count`: received envelopes whose Harness validator failed to produce a valid validation result;
- `terminal_response_count`: `response_count + response_failure_count + response_cancelled_count + response_validation_failure_count`;
- `unpaired_request_count`: requests with no terminal response row;
- `duplicate_terminal_count`: extra terminal rows sharing a request ID;
- `orphan_terminal_count`: terminal rows whose request ID has no request row;
- `consumed_decision_count`: validated envelope decisions consumed by the environment;
- `rules_resolution_count`: rules accepted/rejected/skipped/not-selected rows;
- `trace_row_count`: all decision trace rows.

## 3. Transcript

Transcript entries have a stable sequence, kind and redacted payload. Kinds are:

- `event`: environment/rules/public lifecycle events;
- `decision`: protocol and rules trace;
- `harness`: run-level failure/lifecycle rows.

Common public environment events include:

```text
phase_started
night_resolved
speech
vote_cast
vote_rejected
vote_incomplete
vote_resolved
last_words
last_words_skipped
hunter_shot
agent_decision_failed
decision_envelope_rejected
decision_validation_failed
game_ended
```

`agent_decision_failed` means no `DecisionEnvelope` was produced. `decision_envelope_rejected` means an envelope was produced and recorded as the request terminal, but protocol validation rejected it. `decision_validation_failed` means the Harness validator itself failed; its public reason is sanitized and explicitly does not attribute the defect to the Agent. A RulesEngine rejection is recorded as the independent `rules_result` environment-adjudication row; when the affected seat needs a live notification, it receives the private `action_rejected` event instead of an Agent-failure event.

`analysis` is admin-only and contains factual outcome/counts. It is not a quality judgment.
`analysis.decision_trace_metrics` includes request/terminal/rules pairing plus model-generation
and tool-call/result counts, safe tool failure code/tool histograms, requests with tool failures,
per-request maxima, `history_compaction_count`, `requests_with_history_compaction`,
`max_compacted_tool_groups`, before/after history character maxima and
`history_compaction_limit_unsatisfied_count`. These fields are recomputed from
trace rows and never contain tool arguments, provider error messages or private
reasoning text. The separate `AgentSessionResult.public_summary` reports
`history_compactions`, `max_compacted_tool_groups`, `peak_history_chars` and
`peak_model_history_chars`.

The same analysis includes a run-level `agent_turn_finished` rollup. Only one
finished telemetry row whose request ID and seat agree with a unique
`agent_request` contributes generation, retry, tool, latency and token totals.
Duplicate/orphan/missing, malformed and identity-mismatched rows are retained as
integrity counters. Numeric invariants reject inconsistent telemetry (for
example, generations plus generation failures must equal attempts, and tool
successes plus failures must equal tool calls); token usage is marked incomplete
when a paired row explicitly reports incomplete usage, while missing or invalid
paired rows are counted separately as usage unavailable.

Audience projections are fail-closed:

- public spectator receives only explicitly public event types;
- player receives public events and private rows addressed to that seat;
- god receives omniscient game events, without decision reasoning, tool arguments or raw credentials;
- admin receives the full local redacted machine trace. An authorized God/Admin
  console obtains bounded reasoning and structured tool arguments from this
  admin-protected trace, never from a game event stream.

### Artifact envelope

Offline persistence uses exactly `manifest.json`, `summary.json` and
`transcript.jsonl`. `write_run_artifacts` accepts either the legacy
`HarnessRunResult + RunSpec` pair or the generic
`EnvironmentRunResult + CoreRunSpec` pair. The legacy pair uses
`agent-harness.manifest.v2`; the generic pair uses the independent
`agent-harness.core-manifest.v1` schema. Readers select one of these exact
versions before model validation and reject a missing or unknown version.

Both formats commit the manifest last after atomically replacing the JSONL and
summary. The manifest binds those two files by SHA-256 and byte length. A
valid artifact set contains only the three canonical regular files; artifact
file symlinks are invalid, and the writer refuses a pre-existing run-directory
symlink or a run directory that resolves outside the artifact root.

Both manifests embed their exact spec. A legacy Werewolf manifest therefore
persists `RunSpec.ruleset_id` and `role_deck`; the generic manifest persists the
same values under its embedded `CoreRunSpec.environment_config`. These fields
are part of each spec's canonical hash. A Core manifest also embeds the typed,
credential-free `CoreRunSpec.actors` provenance.

Every persisted JSONL row must carry the declared transcript schema and run
ID, use a contiguous one-based `seq`, and match its own `payload_hash`.
`CoreRunManifest` also carries `transcript_metadata` and
`transcript_counts_by_kind`. Core verification confirms that metadata remains
redacted, recomputes the counts, reconstructs the Transcript from those
manifest fields plus the JSONL rows, and independently
recomputes `stable_digest`. Newly written legacy v2 artifacts carry a
versioned `transcript_integrity_version` plus metadata/counts and receive the
same independent reconstruction check. Pre-extension legacy v2 artifacts may
omit those fields and remain readable, but the verifier explicitly does not
claim independent digest reconstruction for them.

### Real-model smoke verification

`python -m src.harness.smoke <run-dir>` is an offline verifier for an artifact
already produced by a real-model run. It first applies the artifact checks
above, then requires a completed result, a non-zero model-call metric, exact
request/terminal pairing, and at least one valid `agent_response` with a
non-empty `model_call_id` that is referenced by `decision_consumed`. Duplicate
call IDs, call metrics smaller than transcript evidence, unredacted structured
credential fields, Bearer/`sk-` values, URL userinfo and credential query
parameters fail closed. The versioned report contains only IDs/digests and
counts; it does not copy prompts, responses or model-call IDs. Fixture evidence
does not satisfy the real-model gate without a provider-produced artifact.

## 4. REST API

### Health, readiness and browser origins

```text
GET /healthz
GET /readyz
```

`/healthz` is process liveness only. `/readyz` returns `503` while the
`RoomManager` is closing/closed or its Router contract is unavailable. Neither
endpoint probes a provider or exposes model configuration, room identity or
usage. Browser CORS origins come from `WEREWOLF_CORS_ORIGINS` and must be exact
HTTP(S) origins without wildcard, path, query, fragment or URL credentials.
CORS is not authentication and does not replace a reverse proxy's WebSocket
Origin policy. WebSocket upgrades with an `Origin` header must match the exact
`WEREWOLF_CORS_ORIGINS` entry; malformed, wildcard, path, query or credentialed
origins are rejected. Missing Origin is controlled by
`WEREWOLF_WS_ALLOW_MISSING_ORIGIN` (default `true` for native clients; set it to
`false` when only browser-originated upgrades are allowed).

### Configuration metadata

```text
GET /api/providers
GET /api/config
```

`/api/config` returns whether a key is configured but always returns an empty `api_key`. It may return a credential-free API base.

### Room lifecycle

```text
POST /api/rooms
GET  /api/rooms/{room_id}
POST /api/rooms/{room_id}/start
POST /api/rooms/{room_id}/seats/{seat}/model_config
DELETE /api/rooms/{room_id}
```

Create request:

```json
{
  "player_names": ["A", "B", "C", "D", "E", "F"],
  "human_seats": [1],
  "experiment_seed": 100,
  "model_config": {
    "provider": "openai",
    "api_base": "https://gateway.example/v1",
    "model": "model-id",
    "api_key": "provided-at-runtime"
  }
}
```

Room creation returns an `admin_token` and tokens for declared human seats. Tokens are capability credentials; do not log or persist them in artifacts. Capability-issuing responses are marked `Cache-Control: no-store`, `Pragma: no-cache`, and `Referrer-Policy: no-referrer`; clients should keep plaintext only for the current session.

Start and per-seat model configuration require:

```text
X-Room-Token: <admin_token>
```

Changing provider or API base does not silently inherit a key from a different trust boundary. Supply an explicit key for the new boundary.

Interactive Phase B uses one identity for `room_id`, `GameState.id`, legacy
`RunSpec.run_id`, canonical `CoreRunSpec.run_id`, `Transcript.run_id` and
subsequent `ActionRequest.run_id`.
Startup resolves every `seat:<n>` through a run-scoped `AgentRegistry`, validates
human/model provenance, and constructs the dealt state, both specs, the unique
transcript, Actors, shared `DecisionRuntime` and plugin session off to the side.
Only a successful durable commit
publishes the graph as `running`; a pre-commit failure restores the untouched
waiting room.

The published task delegates execution to `run_prepared_environment_run`.
Core owns the run timeout, cancellation, session/runtime close, bounded cleanup
and normalized environment result. Full `EnvironmentRunEvidence` sinks commit
event/decision/harness evidence to the room source rows and the same Transcript
exactly once. Source/Transcript/delivery mutations are staged, saved as one
snapshot and only then queued to WebSocket clients; a save failure restores the
source lengths, Transcript, trace cursor and delivery cursors. A completed or
incomplete Harness terminal is committed together with its state-validated room
status and replayable `room_status`, so restart cannot reinterpret that durable
terminal as an interrupted running room. `RoomManager` maps that result to the REST/WebSocket room status
and retains transport capabilities, delivery projection and persistence.

For a persisted Core room, restore requires a valid canonical `run_spec_hash`,
the legacy hash and `CoreRunSpec.metadata.legacy_spec_hash`, plus semantic
agreement between the legacy and Core environment, players, rules/deck/policy,
seeds, ActorSpec and shared timeout fields. Event and decision `source_idx`
values must each appear as `0..n-1` in Transcript order; their `_trace_seq`
values must be positive, unique and strictly increasing across both kinds, and
the room cursor must equal the maximum. A missing Core hash or cursor is not a
legacy fallback. Only a record with no Core spec may enter the explicit legacy
compatibility path.

Room retention is bounded by `WEREWOLF_MAX_ROOMS`; capacity exhaustion returns
`429` and never evicts a running room. `WEREWOLF_TERMINAL_ROOM_TTL` controls
lazy cleanup of completed/failed/timed-out/cancelled rooms with no live task or
client. Authenticated `DELETE` removes an idle room, returns `409` for running
or connected rooms, and returns `503` while the manager is draining. These are
single-process lifecycle limits, not distributed request-rate or provider-spend
controls. `WEREWOLF_MAX_ROOM_EVIDENCE_ENTRIES` bounds the combined event and
decision evidence retained by one room. Reaching it terminates the room with
`reason=evidence_limit`; existing transcript rows are never silently evicted.
`WEREWOLF_MAX_WS_CLIENTS_PER_ROOM` independently bounds simultaneously retained
WebSocket clients (default `256`); capacity rejection uses close code `4429`.

### Trace and replay

```text
GET /api/rooms/{room_id}/trace
GET /api/rooms/{room_id}/replay
```

Both require the admin token. Replay additionally requires the room status to be `ended`; otherwise it returns `409`.

`trace` reads the accumulated transcript and protocol rows. `replay` returns an ended-room history projection. Neither endpoint invokes a model or re-executes rules.

## 5. WebSocket

Endpoint:

```text
WS /ws/{room_id}?mode=<mode>&seat=<seat>
```

Browser clients send the room capability in the WebSocket handshake
`Sec-WebSocket-Protocol` list, not in the URL:

```text
Sec-WebSocket-Protocol: werewolf.v1, werewolf.cap.<capability>
```

The server selects only `werewolf.v1` in its response. The capability value is
an authorization credential, not an application subprotocol. Native/CLI
clients that cannot set a custom capability subprotocol may continue to use
the compatibility query form `?token=<capability>`; clients must never send
both forms with different values. Browsers cannot use an arbitrary
`X-Room-Token` header during the WebSocket upgrade.

Modes:

- `spectate`: public projection, no token, no seat;
- `play`: declared human seat, requires matching `seat` and seat token;
- `god`: live omniscient view, requires admin token;
- `replay`: ended-room view, requires admin token and ended status.

Supplying `seat` outside `play` is rejected. Unknown modes are rejected. `replay` before game end is rejected.

Human action message:

```json
{
  "type": "human_action",
  "request_id": "opaque-request-id",
  "action": "speak",
  "speech": "exact public text",
  "phase": "day",
  "day": 1
}
```

Targeted actions additionally include `target_seat`. The runtime binds the submission to the pending request ID, phase, day, action type and advertised targets. A timeout produces `human_action_expired`, an `agent_response_failed` terminal row and a linked `agent_decision_failed` event. It never becomes `SKIP`; only an explicit accepted human `action=skip` does.

Human request lifecycle events are:

- `human_action_request`: opens one request with the effective timeout shown by the client;
- `human_action_accepted`: confirms that the matching input entered the Agent queue and closes the client request;
- `human_action_rejected`: reports validation/staleness; terminal stale reasons close the client request;
- `human_action_expired`: closes the matching request when its effective deadline elapses.

A matching `agent_decision_failed`, `decision_envelope_rejected`, or `decision_validation_failed` also clears a pending client request. Public projection distinguishes `human` from `llm` failures in its copy without exposing raw provider or stack details; validator defects remain Harness-attributed.

Text `ping` receives `pong`.

## 6. Redaction

- Secret-like dictionary keys are replaced with `[redacted]`.
- Known room/admin/seat secrets are removed from arbitrary strings.
- Bearer tokens and key-like text are removed.
- Safe structured `api_base` fields remove user info, query and fragment but retain endpoint path for provenance.
- URLs embedded in arbitrary error/log strings are reduced to scheme + host + optional port; their paths are not retained.
- Admin tool arguments are recursively redacted and bounded by depth, key/item
  counts and text length before the console renders them; raw tool outputs and
  error detail objects are not part of that UI projection.

No endpoint should return an API-key prefix, suffix, mask or fingerprint.

## 7. Offline experiment scheduling

`ExperimentSpec.replicates` and the CLI `--runs` option both mean runs per turn
policy. Therefore the total schedule length is `replicates * len(turn_policies)`.

For a two-policy ABBA experiment, `replicates` must be even. With policies
`A,B` and `replicates=2`, the canonical schedule is `A,B,B,A`: the first `A/B`
pair shares one role/actor/orchestrator seed triplet, and the following `B/A`
pair shares the next triplet. Every `RunSpec.metadata` records the global and
within-policy indices, pair ID, counterbalance order, ABBA block/position,
`runs_per_policy`, scheduled total, and all three seeds used by that run.

An experiment may additionally request a cyclic seat permutation with the CLI
option `--seat-permutation cyclic`, or programmatically with
`ExperimentSpec.metadata.seat_permutation_mode="cyclic"`. The permutation is
case-based rather than policy-based: every policy evaluated for the same case
receives the same mapping, including both members of an ABBA pair. The next
case rotates the source player order by one seat. A concrete run records:

- `seat_permutation_mode`: currently `cyclic` when enabled;
- `seat_rotation`: zero-based cyclic offset for the case;
- `seat_permutation`: 1-based source seat assigned to each new seat;
- `permutation_id`: stable label such as `seat-rotation-01`;
- `source_player_names`: canonical player order before permutation.

The default `fixed` mode emits none of these additional fields, preserving the
legacy schedule and its resume identity. This control balances player identity
against seat and seeded role assignment; it does not claim to estimate a role
effect on its own. Runtime `seat_model_configs` are keyed by source seat and
move through the same mapping. Both the resolved `RunSpec.seat_models` manifest
and the actual model invocation therefore bind the override to its permuted
physical seat; paired policies receive identical model placement.

Role layout and persona are separate controlled dimensions rather than aliases
for the case seed. Explicit role-layout modes are `fixed` and
`counterbalanced`; both record the exact shuffled seat-role table and a stable
layout hash in every resolved `RunSpec`. A counterbalanced layout changes only
after a complete identity/persona control block. Explicit persona modes are
`fixed`, `randomized`, and `counterbalanced`. They use a versioned prompt-bearing
catalog and record the source assignment, physical-seat assignment, independent
seed, source/physical player mapping, and stable hashes. The runner recomputes
this provenance before Actor construction and rejects altered prompt text,
profile IDs, source players, positions, seeds, or hashes.

Independent cyclic controls are crossed as a Cartesian product. Seat rotation
is the fast axis and counterbalanced persona position is the slow axis. With six
seats and six persona profiles, a complete block therefore contains 36 cases
per policy and role layout. Advancing both controls together for only six cases
would sample a diagonal and leave them confounded, so a schedule claiming
counterbalancing is rejected unless it contains the complete product. Every
policy in one case receives the same role layout, source identities, personas,
and model placement. Explicit persona binding still delegates exactly one seat
to one unique `AgentActor`; it never creates a controller for multiple seats.

### Recomputable strategy evaluation

`RunSummaryRow` remains able to parse legacy
`werewolf.harness.run_summary.v3` JSONL. Every newly derived row is
`werewolf.harness.run_summary.v4`, including failed runs and runs without
strategy analysis, so new rows never masquerade as v3. When truth-bearing
`analysis.agent_strategy_metrics` exists, the row also carries the additive
`strategy_metrics` object. Its evidence comes only from environment truth,
accepted `decision_consumed` trace rows, and the complete decision-failure
counters. The capped failure record sample is not used when the full `by_seat`
counter exists. Model self-reports, generated summaries, and prose quality
grades are not inputs.

A v4 JSONL object is a resume checkpoint and cache, not an independent
evaluation attestation. Public fields such as `transcript_digest`,
`source_transcript_digest`, and `transcript_provenance_verified` can all be
rewritten together, so consumers never use them as a trust root. The live
factory attaches a process-local, non-serialized content attestation covering
the entire canonical row. Mutation, `model_copy(update=...)`, or JSON
round-trip removes that eligibility. `summarize_runs` still reports ordinary
cached outcome/resource totals, but operational, strategy, deception, and
comparative evaluations consume only attested rows. It reports
`evaluation_evidence_run_count` and `cache_only_run_count` so a missing
evaluation cannot be mistaken for a measured zero.

`load_verified_run_summary(run_dir)` is the offline recovery path. It reads
the committed manifest, summary, and transcript once, verifies their hashes,
reconstructs the transcript envelope, and re-runs the same summary factory. A
pre-extension legacy artifact without transcript metadata/counts remains
readable under its older integrity contract but cannot regain derived
evaluation trust. This establishes internal recomputability, not author
authentication: the manifest is unsigned and requires an external
signature/HMAC/append-only anchor when whole-directory replacement is in the
threat model.

The experiment summary exposes
`strategy_evaluation` (`werewolf.harness.strategy-evaluation.v1`) with
`overall`, `by_turn_policy`, `by_role`, `by_seat`, `by_persona`, and
`by_role_layout` aggregates. The experiment summary also records run counts by
persona mode and exact role-layout ID. Every derived value retains the counts
needed to recompute it:

```text
decision_failure_rate = decision_failure_count / decision_attempt_count
decision_attempt_count = decision_success_count + decision_failure_count
belief_brier = belief_brier_sum / belief_observation_count
false_role_claim_rate = false_role_claim_count / structured_claim_count
wolf_council_coverage = wolf_council_participant_count / wolf_council_eligible_seat_count
wolf_final_vote_target_diversity = wolf_final_vote_target_count / wolf_final_vote_count
wolf_vote_agreement_rate = wolf_vote_agreement_count / wolf_vote_agreement_opportunity_count
```

`belief_brier_sum` is the per-run Brier value multiplied by its actual belief
observation count before cross-run addition. This prevents a run with one
belief observation from receiving the same weight as a run with many. Council
coverage counts unique wolf seats that emitted an accepted council event.
Agreement opportunities count only runs (or wolf seat-runs in role/seat groups)
where the environment produced a non-null final-vote agreement fact. A zero
denominator yields `null`, not a fabricated zero. Mixed v3/v4 resume files are
valid; legacy and standalone rows remain in ordinary outcome/cost totals and
are excluded from every derived evaluation until artifact-backed
re-derivation.

False role claims and false seer results remain separate facts because the
same structured claim can contribute to both categories. Only false role
claims use `structured_claim_count` as a documented rate denominator.
`seer_result_contradiction_count` has no reliable opportunity denominator in
the runtime facts and therefore has no fabricated rate. Target diversity sums
each run's distinct final wolf targets and final-vote count before division.

### Objective deception/belief-shift evaluation

Every consumed LLM decision records an admin-only
`werewolf.agent-belief-trace.v1` checkpoint containing only the owning seat,
private-state revision, and bounded per-target probability/confidence facts.
Plans, free-form evidence, role truth, and commitments are not copied into this
checkpoint. Decision traces are artifacts/admin evidence; they are never game
events, Agent observations, or WebSocket deliveries.

`deception_metrics` (`werewolf.harness.run-deception-metrics.v1`) derives
signals only from accepted public structured claims contradicted by environment
truth. A false wolf cover claim measures movement in observers' belief about
the speaker; a false seer result measures movement about the checked target.
Village-role-to-village-role cover claims remain counted but are explicitly
unscoreable by wolf probability. Only opposing-team observers with both a
nearest pre-signal checkpoint and first post-signal checkpoint are paired.
Repeated equivalent signals inside one observer transition are counted once so
one probability change is not multiplied by repetition.

The experiment-level `deception_evaluation`
(`werewolf.harness.deception-evaluation.v1`) exposes overall, policy, true-role,
seat, persona, and exact role-layout views with these recomputable formulas:

```text
signal_pairing_rate = paired_signal_count / scoreable_signal_count
beneficial_shift_rate = beneficial_shift_count / belief_shift_observation_count
harmful_shift_rate = harmful_shift_count / belief_shift_observation_count
mean_deception_direction_shift = deception_direction_shift_sum / belief_shift_observation_count
```

A positive direction shift means the observer probability moved toward the
objectively false alignment proposition. The metric is a temporal association,
not a causal estimate: other public information between checkpoints may also
contribute. Unpaired signals and zero denominators remain explicit rather than
being converted to zero or silently discarded.

### Operational and projection audit evaluation

Every newly derived v4 row also records a deterministic transcript
`audit_metrics` cache object.
The row object is versioned as `werewolf.harness.run-audit-metrics.v1`.
`operational_evaluation` (`werewolf.harness.operational-evaluation.v1`)
aggregates attested new, failed, and artifact-rederived rows overall and by
turn policy. A JSONL-only resumed row remains in ordinary summary totals but
does not claim that its cached audit was performed. It
retains provider call/failure counts, structured responses, response parse
failure and lossy-rejection counts, incomplete responses, input/output tokens,
and total latency. Provider failure rate and average latency use provider calls;
parse, lossy-rejection, and incomplete rates use structured responses as their
explicit denominator.

The same evaluation records `public_vote_count` and
`prior_public_accusation_aligned_vote_count`. Alignment means the vote target
appeared in an earlier public `speech.accuses` event on the same game day; this
is a temporal association, not a causal influence claim.

Visibility auditing runs the deterministic transcript projection audit.
`private_information_leak_count` includes only public hidden-field markers,
public `private_context`, an admin analysis explicitly marked public, or a
private event missing its required visibility/recipient delivery boundary.
Authorized private reasoning and correctly addressed private events are not
leaks. The summary retains audited-run, issue, leaking-run, and leak counts, so
zero-error evidence is distinguishable from a legacy row that was never
audited.

### Core RunSpec schema and Actor bindings

The canonical schema is exactly `agent-harness.run-spec.v1`. Its Actor section
has this shape:

```json
{
  "actors": {
    "default_model": {
      "provider": "openai",
      "model": "model-default",
      "api_base": "https://gateway.example/v1",
      "configured": true
    },
    "model_overrides": {
      "seat:2": {
        "provider": "openai_responses",
        "model": "model-seat-2",
        "api_base": "https://gateway.example/v1",
        "configured": true
      }
    },
    "human_actor_ids": []
  }
}
```

`default_model` and every override are credential-free manifests, not runtime
credentials. Actor IDs are non-empty normalized strings; human IDs are unique
and sorted, and a human Actor cannot also have a model override. Recursive
CoreRunSpec credential checks cover `actors`, `environment_config` and
metadata. API keys and Authorization values therefore fail validation instead
of entering the Core hash or manifest.

`run_core_llm_environment()` uses those declarations as an execution contract,
not display-only metadata. Before it creates a model-backed generic Actor, the
resolved in-memory `ModelConfig` is converted to the same credential-free
manifest and must exactly equal the actor's default or explicit override. A
missing, malformed, human, or mismatched binding fails before any provider
request; sharing a Router never relaxes per-actor provenance.

`load_core_run_spec()` performs exact schema dispatch:

- `agent-harness.run-spec.v1` is validated as Core v1;
- `werewolf.harness.spec.v3` is migrated through
  `legacy_werewolf_run_to_core()`;
- a missing or unknown `schema_version` is rejected.

The legacy migration has one field mapping. It converts seat model overrides
and human seats to `seat:<n>` Actor IDs, copies environment fields into
`environment_config`, converts the three local seeds to namespaces and records
`source_schema_version` plus `legacy_spec_hash` in metadata. The migrated Core
spec computes a separate Core hash; `legacy_spec_hash` is provenance and must
not be interpreted as the Core identity. The offline Werewolf wrapper calls
this same function. Interactive Phase B persists the resulting `CoreRunSpec`,
resolves every `seat:<n>` through a run-scoped `AgentRegistry`, validates
human/model provenance, and executes the prepared plugin session through the
Core lifecycle. The Werewolf `RunSpec` remains only as an explicit compatibility
and source-provenance view; its hash is not substituted for the Core hash.

### Werewolf ruleset and role-deck fields

The Werewolf compatibility `RunSpec` records these environment identities:

```json
{
  "environment_id": "werewolf.classic",
  "environment_version": "1",
  "ruleset_id": "classic.v1",
  "role_deck": ["werewolf", "werewolf", "seer", "villager", "villager", "villager"]
}
```

The wrapper copies `ruleset_id` and `role_deck` into the Werewolf
`CoreRunSpec.environment_config`. `classic.v1` is currently the only supported
ruleset; the environment/plugin version and ruleset ID are independently exact
and neither accepts an unknown version. This protocol does not advertise a
second variant or a rules DSL.

Before a run starts, a custom deck must:

- contain exactly one card per player for 6–12 players;
- contain at least one `werewolf` and one non-werewolf;
- use only roles with implemented capabilities;
- contain at most one each of `seer`, `doctor`, `witch`, `guard` and `hunter`.

`werewolf` and `villager` may repeat. The built-in 6–12 player default decks
retain their existing composition and do not include Doctor. Invalid decks and
unknown ruleset IDs fail during RunSpec/plugin resolution, before session
creation or any model call; `RulesEngine.deal_roles` enforces the same contract
for direct domain callers.

An old input that omits `ruleset_id` is canonicalized to `classic.v1`. The
resolved hash now includes this explicit provenance, so an older summary row
whose hash was computed without the field does not match. `--resume` requires
both exact `run_id` and `run_spec_hash` and therefore rejects that row.
