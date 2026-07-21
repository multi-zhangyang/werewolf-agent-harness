# Architecture

## 1. Scope

The system is an agent harness with two built-in environment IDs: classic
Werewolf and Cipher Council. The exact production plugin references are
`werewolf.classic@1`, `council.cipher@1`, and `council.cipher@2`. Its purpose is
to make Agent decisions executable and auditable under explicit observations,
legal actions, deadlines and deterministic rules.

The system is not:

- a multi-character chat room;
- a chatbot orchestration demo;
- a source of scripted “AI-like” dialogue;
- an independent judge of deception, trust, psychology or conversation quality;
- a deterministic replay engine for nondeterministic model calls.

The architecture separates three kinds of authority:

| Authority | Owner | Examples |
| --- | --- | --- |
| Environment truth | `GameState` + `RulesEngine` | roles, alive/dead state, legal targets, winner |
| Agent choice | LLM or human through `AgentProtocol` | speech, vote, night target, skip |
| Observation/provenance | Harness + `Transcript` | visible events, request IDs, response hashes, failures |

No model-generated field can overwrite environment truth.

## 2. Components

The reusable core resolves environments by exact `(id, version)` through an
explicit `EnvironmentRegistry`. `run_environment_run` owns the run timeout,
Transcript, one shared `DecisionRuntime`, terminal status and session cleanup.
An environment plugin owns only its validated config, state, rules, scheduling
and outcome projection. The offline Werewolf path is registered as
`werewolf.classic@1`; its compatibility wrapper still exposes the v3 Werewolf
RunSpec and result shape while delegating lifecycle execution to the generic
runner.

```text
CoreRunSpec -> EnvironmentRegistry -> EnvironmentPlugin.create_session
                                      |              |
                                      |              v
                                      +-> shared DecisionRuntime
                                      +-> Transcript sinks <- EnvironmentSession.run
                                      +-> timeout/status/close -> EnvironmentOutcome
```

```text
                         ┌──────────────────────┐
                         │ RunSpec / Room config│
                         └──────────┬───────────┘
                                    │
┌───────────────┐  projected view   ▼               validated intent  ┌───────────────┐
│ GameState     │──────────────▶ Orchestrator ───────────────────────▶│ RulesEngine   │
│ hidden truth  │                  │      ▲                            │ transitions   │
└───────────────┘                  │      │                            └───────┬───────┘
                                   │      │                                    │
                           ActionRequest  terminal response                     │ events
                                   │      │                                    │
                                   ▼      │                                    ▼
                           DecisionRuntime                         Transcript / projections
                                   │      ▲
                                   ▼      │
                             AgentProtocol
                          ┌────────┴────────┐
                          │                 │
                       LLM Actor        Human seat
```

### `RunSpec`

Records one run's environment ID/version, exact Werewolf ruleset ID, player
names, role deck, turn policy, three independent seeds, timeout/deadline
settings, safe model manifests and caller metadata. API keys are never part of
the model.

`CoreRunSpec` is the frozen environment-neutral canonical form: exact
environment reference, opaque validated environment config, namespaced seeds,
execution limits, typed Actor bindings and credential-free metadata. Its
`ActorSpec` contains a safe default-model manifest, model overrides keyed by
environment `actor_id`, and human Actor IDs. Human IDs are normalized and
cannot overlap model overrides. Recursive credential checks include the entire
ActorSpec. The current `RunSpec` remains the Werewolf v3 compatibility input
used by the CLI and legacy manifest path. Generic results persist the exact
`CoreRunSpec` in a separate `CoreRunManifest`; the two models do not share a
field bag of optional environment-specific values.

The Core runner copies the frozen `ActorSpec` into `EnvironmentRunContext` and
wraps the caller's resolver in one run-scoped `AgentRegistry`. The
`werewolf.classic@1` plugin derives exactly one canonical `seat:<n>` actor ID
for each dealt physical seat and obtains the execution object only through
`context.resolve_agent`. Before play it rejects ActorSpec IDs outside the
table, seats without a default/override model or human binding, resolved
human/model-kind mismatches, and model Actors whose credential-free
`ModelConfigManifest` differs from the applicable default or per-actor
override. This makes executable Actor identity and model/human provenance part
of the same Core contract rather than trusting a plugin-private actor map.

For the Werewolf compatibility path, resolution canonicalizes an omitted
`ruleset_id` to `classic.v1`, fills the default role deck and copies both into
the resolved `RunSpec`. The legacy wrapper then places the same ruleset ID and
deck in `CoreRunSpec.environment_config`. Both values therefore participate in
the respective spec hash instead of living in untracked runtime defaults.

### Werewolf ruleset and role-deck boundary

The plugin identity `werewolf.classic@1` and the environment ruleset identity
`classic.v1` are distinct provenance fields. Only that exact ruleset is
implemented. Unknown or empty ruleset IDs fail closed; this does not claim that
multiple variants or a rules DSL exist.

`classic.v1` explicitly advertises only roles with implemented observation,
action and resolution paths. A deck must match the 6–12 player count and
contain at least one werewolf and one non-werewolf. `werewolf` and `villager`
are repeatable cards. The power roles `seer`, `doctor`, `witch`, `guard` and
`hunter` are single-card roles; in particular Witch consumables and Guard
history are per-game singleton state. No additional wolf-ratio or balance rule
is inferred.

Validation is deliberately layered. `RunSpec`/`ExperimentSpec` and
`WerewolfEnvironmentConfig` reject bad configuration before a session or Agent
is created. `RulesEngine.deal_roles` repeats the same validation as the final
domain boundary for API and direct engine callers. The existing default decks
for every player count from 6 through 12 retain their previous composition and
do not include Doctor.

### RunSpec loading and legacy migration

`load_core_run_spec` dispatches only on an exact `schema_version`. It validates
`agent-harness.run-spec.v1` directly and sends
`werewolf.harness.spec.v3` through the single
`legacy_werewolf_run_to_core` mapping. Missing and unknown versions fail closed
instead of being guessed from field names.

The migration is explicit:

| Werewolf v3 source | Core v1 destination |
| --- | --- |
| `environment_id` / `environment_version` | exact `environment` reference |
| ruleset, players, deck, policy and phase config | `environment_config` |
| role, Actor and orchestrator seeds | namespaced `seeds` |
| run/decision timeout | typed `execution` |
| `default_model`, per-seat models, human seats | `actors.default_model`, `actors.model_overrides["seat:<n>"]`, `actors.human_actor_ids` |
| caller metadata | metadata plus `source_schema_version` and `legacy_spec_hash` |

The legacy hash is retained only as source provenance. Core v1 is a different
schema and computes its own canonical hash; equality between the two hashes is
neither expected nor claimed. The offline `run_werewolf_run` wrapper invokes
this same migration function after resolving legacy defaults, so there is one
field mapping for execution and direct loading.

Interactive Phase B uses this same migration to persist a canonical
`CoreRunSpec` beside the legacy Werewolf `RunSpec` compatibility view. The
legacy hash remains explicit source provenance; `Transcript.metadata.run_spec_hash`
commits the Core hash. Room tokens, human queues, audience projection and
persistence remain adapter responsibilities, while `run_prepared_environment_run`
owns the live session timeout, cancellation, session/runtime cleanup and
environment result lifecycle.

### `EnvironmentRegistry` and plugin session

Registration is explicit and exact-versioned. Duplicate registrations,
unknown versions, incompatible plugin API versions and incomplete decision
contracts fail closed. Third-party entry points are loaded only through an
explicit opt-in call because loading a plugin executes Python code. A session
must return `EnvironmentOutcome` and is always closed by the generic runner.
Each `EnvironmentRunContext` owns one advancing `random.Random` stream per
declared seed namespace; repeated `context.rng(name)` calls never rewind the
stream, which keeps seeded plugin behavior reproducible.
An unresolved actor binding fails at `AgentRegistry.resolve` and is remembered
for the rest of the run, so a transient resolver result cannot later attach a
different private Agent object to the same actor ID.

If external cancellation arrives while `create_session` is unwinding and the
coroutine returns a session within the cancellation grace period, the runner
still adopts and closes that late session before propagating cancellation.
The low-level `run_environment_run()` contract intentionally propagates the
caller's `CancelledError` after this cleanup so task groups and batch owners
cannot mistake an interrupted run for success. Room/supervisor boundaries may
persist their own serializable `cancelled` status before re-raising; callers
that need a cancelled artifact must use that owning boundary rather than
silently converting the low-level control-flow signal into a normal result.

### `GameState`

Contains authoritative hidden state. It is kept inside the environment. An Agent receives an `AgentObservation`, not `GameState`.

### `RulesEngine`

Owns deterministic legality and transitions: role dealing, action application, vote resolution, deaths, hunter resolution and winner checks. It does not call a model.

For `classic.v1`, night actions are collected before resolution. Distinct actors'
submission order is not a rule input. Re-submitting the same action by the same
actor is an explicit replacement, so only that actor's latest choice remains.
Resolution then uses this fixed order:

| Step | Input | Deterministic resolution |
| --- | --- | --- |
| 1 | Seer checks | Emit one seat-private result per accepted check; no public role truth is added. |
| 2 | Werewolf council and final votes | Each AI wolf independently emits one exact team-private message/tentative target. All living wolves receive those private events, then independently submit final kill votes. The orchestrator uses final-vote plurality and a recorded seeded tie-break. |
| 3 | Guard and `SAVE` actions | Build protection sets. A wolf target with exactly one of guard or save survives; with neither it dies; with both it dies under the `classic.v1` double-protection rule. |
| 4 | Witch poison | Apply after wolf protection. Poison cannot be blocked and changes an already selected victim's reason to `poisoned`. |
| 5 | Death and hunter state | Apply deaths in wolf-target-then-poison order. A poisoned hunter cannot shoot; other eligible hunter deaths enter the pending queue. |
| 6 | Announcement and phase | Emit the public death list without role/death-cause disclosure, clear night actions, and enter day. |

The exhaustive six-action permutation test covers two tied final wolf votes plus
seer, doctor save, witch poison and guard actions. A separate orchestrator test
reverses Actor registration order and proves the seeded wolf tie-break is stable.

### `GameOrchestratorV2`

Schedules phases, projects observations, advertises legal action/target sets, asks `RulesEngine` to consume validated intent and records environment resolutions.

Construction is fail closed: Actor keys must exactly cover player IDs; player
IDs and seats, Actor objects and Memory objects must each be unique; and every
Actor/Memory seat, name and role must match its `PlayerState`. A request is
bound to that exact mapping and its `seat` is derived from `PlayerState`, not
trusted from the Actor. One player/one Agent is therefore a runtime invariant,
not merely a property of the default factory.

### Per-seat Agent runtime

Every production `AgentActor` owns distinct mutable objects for episodic
Memory, `PrivateAgentState`, RNG, persona/strategy prior, human queue and model
configuration. Actors may share one stateless `LLMRouter` transport pool; they
do not share prompts, dialogue sessions, memories, beliefs or strategies.

The seat-owned context separates epistemic categories:

| State | Authority | Meaning |
| --- | --- | --- |
| `AgentMemory` observations | Environment-delivered | Facts/events actually visible to this seat |
| public claim ledger | Environment-accepted speech | What players publicly claimed, never role truth |
| private role marginals | Agent-authored and constraint-projected | This seat's current subjective belief about other seats |
| strategy candidates/selection | Agent-authored | Alternatives considered and the plan selected for this action |
| cover/deception/team plan | Agent-authored | Private working strategy, never a factual audit label |
| public commitments | Environment-accepted own output | Exact statements this seat must remember for narrative continuity |

Each LLM decision must return a structured `private_state` alongside its exact
action. It includes belief updates, at least two distinct strategy candidates,
the selected plan, a second-order estimate of how others perceive the seat,
and optional cover/deception/team plans. The state is committed only to that
Actor and rendered into its next request. Public output may intentionally
contradict private belief; the environment does not correct a bluff.

Role marginals respect facts visible to that seat: the public number of wolves,
known wolf teammates and private Seer results. This is a capped-simplex
projection of model-authored marginals, not a claim to implement a calibrated
Bayesian posterior, joint role-assignment factor graph, RL policy or independent
deception evaluator.

All player/event text is serialized inside a delimited JSON observation and is
labelled as untrusted game data. A table utterance that resembles a prompt is
still opponent speech, not an instruction. Requests record visible event IDs,
wolf-team event IDs, Memory digest and private-state digest so ordering and
seat-context identity are auditable without exposing private text.

The tool loop also carries an explicit request-scoped seat contract. Its initial
turn and `get_legal_actions` expose sorted `visible_seats` and `alive_seats`.
Seat-bearing JSON schemas are generated from the current visible roster, from
the roster minus the owner for opponent beliefs/claims, or from the exact
matching `LegalAction.target_seats` for terminal actions. A nonexistent seat is
therefore rejected before a handler runs. An empty legal target set never
creates an invalid empty JSON-Schema enum; terminal preflight and the Rules
boundary remain authoritative.

For the common decide-after-one-read path, the Werewolf registry exposes
`read_turn_context` as a bounded, seat-private snapshot. It combines the exact
legal actions and target set, private facts, recent public events/votes and
claims, subjective belief/plan state, and this seat's accepted commitments.
The snapshot is recursively bounded and redacted; granular reads remain
available when a specific historical detail is needed. `update_private_state`
atomically commits this seat's beliefs, candidate/selected plans and deception
state, while the older single-field update tools remain compatibility paths.

`read_public_events` defaults to 12 rows and accepts at most 24. Its durable
memory window contains only speech and last words and excludes items already
represented by the current public events or today's speech view, while retaining
older last words absent from the current observation. Subjective state and
public commitments have distinct read paths: `get_beliefs` omits commitments,
and `get_commitments` is their only tool projection.

### `DecisionRuntime`

Owns the environment-to-Agent boundary for one request: it writes the request row, enforces the effective decision/phase deadline, invokes `AgentProtocol`, validates an envelope and writes exactly one `agent_response`, `agent_response_failed`, `agent_response_cancelled`, or `agent_response_validation_failed` terminal row. Generic core runs bind the runtime to their owning `run_id`; a request labelled for another run fails as `RunIdMismatch` before the Agent is invoked. The orchestrator does not wrap this path in a second timeout.

### `AgentProtocol`

Defines one production method:

```python
async def decide(request: ActionRequest) -> DecisionEnvelope
```

`AgentActor` adapts a standard LLM API to this protocol. The room runtime adapts human WebSocket input to the same protocol. No production scripted or replay implementation exists.

Each model turn inside an `AgentSession` uses protocol-level required tool
choice. The model still chooses among the seat-bound read, private-state and
terminal tools, but plain assistant chat cannot count as progress or end a
request. A provider that violates this requirement is handled fail-closed by
the bounded no-progress guard.

The session retains its complete canonical message/audit history. Before each
provider turn it creates a detached model view and, when that view exceeds the
configured character target, rolls older *complete* assistant-tool-call plus
tool-result groups into bounded structured summaries. Recent groups stay exact;
malformed/incomplete groups are never split or compacted, and the replacement
must actually reduce input size. This operation changes only provider input,
not `AgentSession.messages` or the admin trace. `AgentSessionResult` exposes
`history_compactions`, `max_compacted_tool_groups`, `peak_history_chars` and
`peak_model_history_chars`; transcript-derived analysis also records compaction
counts, affected requests, before/after maxima and unsatisfied-target counts.

### `LLMRouter`

Supports standard OpenAI Chat Completions, OpenAI Responses and Anthropic Messages protocols. It owns provider/network retries, timeout enforcement, response usage, latency and call statistics. Official SDK retries are disabled to keep retry ownership visible in one layer. Safe `llm_call.transport_attempts` record each bounded transport attempt; Actor-level `actor_response_attempts` separately record bounded fresh-response attempts after malformed or schema-inadequate responses. Generic API errors and structured SSE error events recover standard 429/5xx/529 semantics for bounded retry, while non-retryable 4xx responses remain terminal; this classification has no endpoint or model branch. When a provider budget ledger is configured, every transport attempt requires an explicit run/room scope and an atomic reservation. Failed attempts are charged with unknown usage, successful attempts record provider-reported input plus output usage, and missing usage blocks a token-limited scope. A portable hard call count is available for every protocol; token limits retain the provider-boundary caveats documented in `SECURITY.md`.

### `Transcript`

Appends ordered rows of kind `event`, `decision` or `harness`. It computes a stable digest and supports audience projections. Private reasoning remains inside the authorized `DecisionEnvelope` trace instead of being copied into a second synthetic “thinking” stream. Because tool-capable models often express their plan only in function arguments, `tool_call_requested` keeps a bounded, recursively credential-redacted argument view in the admin-only trace alongside its hash. The authorized console may render that structured private reasoning; it is never projected into public events or another Agent's observation. Transcript rows are the source for console trace and artifacts; the console does not rerun the environment.

### Adversarial scenario runner

`ScenarioSpec` binds an exact `CoreRunSpec`, fixed seeds and machine-checkable
invariants to a versioned scenario ID. `run_adversarial_scenario` executes that
spec through the same registry, generic runner and `DecisionRuntime` as other
core runs; it does not infer success from a static catalog. Every scenario must
at least prove request/terminal pairing, a nonzero request count, absence of a
fabricated choice on failure, and credential/marker absence. Reports contain
redacted run facts, invariant observations and transcript digests.

The repository's protocol-attack, provider-fault, deadline, hostile-leak,
contradictory-claim and collusion-capable agents/plugins remain test-local.
They exercise production harness boundaries without adding a scripted Agent to
the production runtime.

## 3. One decision lifecycle

For each required action:

1. The environment builds a seat-specific observation from public events and information legally visible to that seat.
2. It computes target seats from current state and rules.
3. It creates a frozen `ActionRequest` with `request_id`, phase/day, action kind, legal action declarations and the minimum of decision/phase deadlines.
4. `DecisionRuntime` appends an `agent_request` trace row and owns the effective wall-clock deadline.
5. The Agent either returns one `DecisionEnvelope` with the same request ID and seat or raises/fails to answer.
6. The runtime appends exactly one terminal row for that request: `agent_response` with validation and safe model-call provenance, `agent_response_failed` with structured failure facts and no envelope, `agent_response_cancelled` when the owning room/run is cancelled, or `agent_response_validation_failed` when an envelope exists but the Harness validator itself fails.
7. Protocol validation checks version, identity, advertised action, explicit skip semantics, target scope and action-specific payload fields.
8. A valid decision is recorded as `decision_consumed`. `RulesEngine` then writes a separate `rules_result` environment-adjudication row for accepted, rejected, skipped or not-selected resolution; it is not a request terminal.
9. The result is appended to the transcript and projected by audience.

Public speech is a special invariant: if an accepted `SPEAK` or `LAST_WORDS` decision has a non-empty `speech`, that exact string becomes the public text. There is no second generation pass. An explicit legal `SKIP` records a skipped resolution and emits no fabricated quote; timeout, invalid payload and provider failure remain failures rather than being converted to silence.

## 4. Observation and visibility

Visibility is fail-closed. A new event is not public until explicitly included in the projection allowlist.

Audience levels:

- `public` / spectator: public phase, speech, vote, death and failure structure.
- `player`: public events plus private events explicitly addressed to that seat.
- `god`: omniscient game events, but never decision reasoning inside the event/WebSocket stream.
- `admin`: capability-protected machine trace, including request/envelope provenance,
  safe hashes and model-private reasoning. The god/replay UI reads reasoning only
  through this admin-authorized trace endpoint.

Private reasoning, private state and wolf council messages never become public
speech. Werewolf teammates and role truth are only added to observations when
the rules authorize them. Public claims remain records of what someone said,
not role truth.

`AgentMemory` remains a per-seat factual timeline and claim ledger. Subjective
beliefs and strategy live in the separate per-seat `PrivateAgentState`, so
model inference cannot overwrite delivered facts. Neither object performs an
independent quality judgment or exposes mutable nested references to callers.

## 5. Failure model

The harness keeps failures distinct because collapsing them would falsify the run:

- `SKIP`: Agent deliberately takes no action where skipping is legal.
- protocol rejection: an envelope exists, is the terminal `agent_response` row, and fails identity/action/payload/target validation; the live event is `decision_envelope_rejected`.
- validator defect: an envelope exists, but the Harness validator raises or returns an invalid result; `agent_response_validation_failed` is the request terminal and the sanitized live event is `decision_validation_failed`. This is a Harness failure and is not attributed to the Agent.
- rules rejection: validated intent reached the environment but violates current game state; it remains a `rules_result`/private `action_rejected`, never an Agent response failure.
- response error: provider returned incomplete, malformed or schema-inadequate content.
- transport/provider failure: network, timeout, throttling or retryable provider status exhausted.
- decision deadline: `DecisionRuntime` wall-clock budget expired before an envelope existed.

Interactive human clients receive explicit request/accepted/rejected/expired lifecycle events. The UI clears a request when that request is accepted, expires, receives a terminal stale rejection or its matching Agent decision fails; it never keeps a longer cosmetic countdown than the deadline enforced by `DecisionRuntime`.

None of these paths creates a replacement speech, vote or target.

Retry ownership is bounded:

- Router retries retryable network/provider conditions, default maximum three attempts.
- Actor may request up to three fresh responses only for response/schema errors.
- Actor does not repeat a call after Router has exhausted a transport/provider budget.

This prevents nested retry amplification.

Attempt provenance preserves both retry layers without preserving model
content. A safe `llm_call` includes request/response/reasoning hashes,
`api_base_fingerprint`, status/error type, retry/latency facts and usage where
available. Its `transport_attempts` describe Router calls;
`actor_response_attempts` describe accepted, response-rejected or
provider-failed fresh-response calls and may nest their safe `llm_call` facts.
No raw prompt, raw response, API key, or API-key fingerprint is stored in these
attempt records.

## 6. Scheduling

The environment currently implements two public-discussion policies:

- `fixed_round_robin`: living seats are asked in seat order.
- `bid_reply`: later rounds request a structured bid and prioritize positive bids, including reply priority for mentioned seats.

Scheduling controls who receives the next `ActionRequest`; it does not write their speech. `bid=0`/`SKIP` means the seat is not scheduled for a public utterance.

Role dealing, Actor persona choice and orchestrator scheduling use separately recorded seeds. These make local random choices reproducible; they do not make external LLM output deterministic.
Werewolf protocol request IDs are a run-scoped monotonic sequence rather than a
fresh UUID, so local correlation identifiers do not introduce an unrecorded
random source. Provider output, model-call IDs, latency and usage remain external
observations and are not claimed to repeat.

## 7. Run modes

### Offline harness

`src.harness.runner.run_werewolf_run` requires real model configuration and explicit role/actor/orchestrator seeds. It rejects human seats because an unattended job cannot supply interactive input. It has no `actor_factory` or scripted mode. The legacy wrapper prepares one real Actor per dealt seat, but the plugin obtains every `seat:<n>` object through the Core `AgentRegistry` and verifies it against `CoreRunSpec.actors` before the game runs.

### Room runtime

`RoomManager` supports LLM seats and authenticated human seats. REST creates/starts rooms; WebSocket sends audience-projected events and receives human actions. Admin trace reads the same accumulated transcript and decision rows.

Interactive Phase B binds one identity across `Room.id`, `GameState.id`, live
legacy `RunSpec.run_id`, canonical `CoreRunSpec.run_id`, `Transcript.run_id` and
every later `ActionRequest.run_id`.
Every physical seat resolves as `seat:<n>` through one run-scoped
`AgentRegistry`, with distinct object/Memory ownership and ActorSpec kind/model
provenance checks. Startup is staged against a detached state: role dealing,
legacy/Core spec and transcript construction, Actor binding, the shared
`DecisionRuntime`, and a room-owned Werewolf plugin session must all succeed
before one durable cutover publishes `running`; pre-commit failures close
unclaimed Core resources and leave the waiting room undealt and retryable.

Restore is fail-closed for newly written Core rooms. The canonical Transcript
hash must exist and match `CoreRunSpec`; the legacy hash, Core
`metadata.legacy_spec_hash`, environment/player/deck/policy/seeds/ActorSpec and
shared timeout fields must all bind back to the persisted legacy `RunSpec`.
Within the Transcript, each event/decision `source_idx` must follow its source
history in exact order, and their positive `_trace_seq` values form one unique,
strictly increasing timeline whose maximum equals the persisted room cursor.
Only records without a Core spec use the explicit legacy missing-field path.
Waiting-room compatibility may normalize an old `GameState.id`, but it never
permits execution specs or evidence to be attached to a waiting room.

The published task invokes one `PreparedEnvironmentRun`. Core owns the run
deadline, cancellation, `EnvironmentSession.aclose`, `DecisionRuntime.aclose`,
cleanup failures and normalized environment outcome. `EnvironmentRunEvidence`
delegates event, decision and harness rows to full room sinks, which append
source history and the same Transcript exactly once. Each full-sink mutation
stages source rows, Transcript rows and delivery cursors, persists one snapshot,
and only then exposes queued WebSocket messages; persistence failure restores
the in-memory mutation. `run_completed`/`run_incomplete`, the validated room
terminal projection and its replayable `room_status` therefore share one
durable commit instead of leaving a crash window labelled `running`.
`RoomManager` remains the
interactive host: it maps the Core result to room status and owns capability
checks, WebSocket delivery, audience projection, provider scope and SQLite
snapshots. That host projection is not a second environment lifecycle.
Tasks that still ignore Core's bounded cancellation are handed to a run-scoped
RoomManager quarantine sink. Shutdown sends a claimed Core room one cancellation
and supervises its exported sequential session/runtime cleanup budget; it does
not repeatedly interrupt Core-owned cleanup. Any still-live child remains
strongly referenced and is reported as a fatal cleanup failure.

The interactive API is a single-worker runtime. One process owns each live
room object, game task and lock; its in-memory capability state, WebSocket
client queues, authorization-specific delivery streams/cursors and SQLite
snapshot writes are coordinated by that same `RoomManager`. SQLite persistence
supports restart recovery for one owner, but it is not distributed room
ownership, locking or pub/sub. Running multiple workers can route REST and
WebSocket traffic to different managers, split broadcasts and permit
conflicting mutations. A multi-worker or multi-host service therefore needs an
external room owner/coordinator, shared state and locks, cross-process
publication and reconnect cursors, routing guarantees, distributed admission
limits and a distributed provider budget ledger.

### Replay

Replay is available only after the room ends. It is a read-only projection of immutable history, analysis and transcript. It does not call an Agent or mutate `GameState`.

## 8. Artifacts and factual summaries

Each offline run writes exactly:

- `manifest.json`: safe configuration/provenance and artifact paths;
- `summary.json`: outcome, counts, failures, Router statistics and transcript digest;
- `transcript.jsonl`: ordered row payloads.

`write_run_artifacts` has two explicit type pairs. Legacy
`HarnessRunResult + RunSpec` writes `agent-harness.manifest.v2`; generic
`EnvironmentRunResult + CoreRunSpec` writes the independently versioned
`agent-harness.core-manifest.v1`. Both use the same exact three-file layout,
atomic replacement and manifest-last commit protocol. The manifest records
the SHA-256 and byte length of the summary and JSONL content files. An
interrupted replacement therefore leaves either a committed matching set or a
set that verification rejects.

Each manifest embeds its exact corresponding spec. For Werewolf runs that
means the resolved `ruleset_id=classic.v1` and validated role deck remain part
of persisted provenance. A Core manifest also embeds the credential-free
`ActorSpec`, including default/override model manifests and human Actor IDs.
Adding explicit provenance fields changes the resolved spec hash relative to
older rows that omitted them; strict resume intentionally rejects those old
rows rather than treating under-specified and versioned runs as identical.

Verification dispatches only on a recognized exact manifest schema; missing
or unknown versions fail closed. It checks canonical relative paths, the exact
file and integrity-record sets, regular non-symlink content files, summary
run/spec/digest links, and every JSONL row's transcript schema, run ID,
contiguous sequence and payload hash. The writer also rejects a pre-existing
run-directory symlink or a resolved run directory outside the artifact root.

`CoreRunManifest` and newly written legacy v2 manifests store the redacted
transcript metadata and counts by kind. A verifier can therefore check the
counts, reconstruct the `Transcript` from disk and independently recompute its
stable digest for either format. Legacy v2 manifests written before this
metadata/integrity extension may omit those fields; they remain readable under
their older file/row checks, but independent stable-digest reconstruction is
not claimed for that compatibility path.

Credentials are removed. `CoreRunSpec` rejects credential fields and obvious
credential-bearing values before hashing or persistence. A safe legacy
manifest API base may retain its endpoint path after removing user info,
query and fragment. Arbitrary URLs found in error/log text are reduced to
origin because paths may contain sensitive tenant or signed-route data.

Production summary fields are limited to recomputable facts such as status, winner, days, elapsed time, model calls/success/failure/retries, input/output tokens, model latency, decision count, parse failures and decision failures. A serialized `RunSummaryRow` is a cache/checkpoint rather than a trust root: its public transcript digest and provenance booleans are not sufficient to authorize derived evaluation. The live result factory keeps a non-serialized whole-row attestation, and `load_verified_run_summary()` rederives that attestation from a verified three-file artifact. Standalone JSONL rows remain available for ordinary checkpoint totals but are excluded from operational, strategy, deception, and comparative claims. This attestation proves internal recomputability only; the unsigned manifest does not authenticate the artifact's author against a whole-directory replacement.

When summary rows carry the canonical experiment schedule metadata,
`aggregate_comparative_metrics` groups them by experiment and `pair_id`, then
requires the same role/Actor/orchestrator seeds, role deck, player placement,
role-layout control and persona control before comparing policies. Missing,
duplicated and control-mismatched pairs are counted separately. For each
factual outcome/resource/strategy metric it reports the number of eligible
pairs, the exact `comparison - baseline` mean and a deterministically seeded
paired bootstrap interval. The report is explicitly descriptive with
`causal=false`; it does not turn a small interval, model self-report or
observational deception association into a causal claim. The batch result
continues to retain the raw `RunSummaryRow` values used to recompute it.

`src.harness.smoke` is a credential-free, offline release verifier. It accepts
only an already committed artifact set and cross-checks non-zero Router calls,
unique request/terminal pairing, unique model-call IDs and consumed valid
responses. Its report contains counts and digests rather than prompt/response
content. It does not invoke a provider, so a passing fixture report cannot
replace the required real-provider artifact.

`decision_count` means validated decisions consumed by the environment. Request,
response, no-envelope failure, cancellation, validator-failure,
consumed-decision, rules-resolution and total trace rows are reported separately
in `decision_trace_metrics`; validator-failure terminals are counted by
`response_validation_failure_count`. The same metrics object recomputes
model-generation and tool-call/result counts, safe failure-code/tool histograms,
requests with tool failures, maximum per-request amplification, provider-history
compaction counts and affected requests, the maximum compacted tool groups,
before/after character maxima and unsatisfied-target counts. It never copies
tool arguments, provider messages or private reasoning into aggregate analysis.
The run-level `agent_turn_finished` rollup only sums a unique, request/seat
identity-matched telemetry row. Duplicate, orphan, missing, malformed or
identity-conflicting rows remain explicit integrity counters and cannot inflate
generation, tool, latency or token totals; accepted rows also expose per-seat
fairness extrema.

The runtime does not claim posterior calibration, deception success, dialogue
quality or statistical significance from architecture alone. Those require
versioned multi-run evaluation. It does claim, and tests, exact seat isolation,
private/public separation, role-count-consistent subjective marginals and
two-stage wolf-team delivery before independent final votes.

### Generic Tool Actors And Cipher Council

`CoreToolActor` is the production real-model adapter for the environment-neutral
`agent-harness.decision.v1` contract. One instance is created for exactly one
Core `actor_id`; it owns an async decision lock, a copied `ModelConfig`, a
budget scope, a bounded actor-local episodic memory and an admin-only trace
sink. The memory retains only this Actor's previously authorized observations,
request labels and submitted terminal choices. It excludes provider reasoning,
raw model responses and traces, is capped by both entry count and serialized
size, and is supplied only to the same Actor on later turns. It compiles only the current
`ActionRequest.legal_actions` into `submit_action_N` functions (plus
`submit_skip` only when the Core skip policy permits it), calls
`LLMRouter.complete_tools(..., tool_choice="required", parallel_tool_calls=False)`,
and accepts exactly one terminal call. It never turns text into an action or
rewrites malformed arguments. Router-owned transport retries remain Router
owned; at most three malformed response shapes are retried by the Actor.

The generic system contract explicitly asks the independent Actor to optimize
its observation-defined objective over the run and to reason about incentives,
alliances, credible public signals, concealment and defection when those are
relevant to the advertised legal actions. This is strategic guidance, not an
environment action or an evaluation claim: whether a strategy succeeded must
still be recomputed from transcript and environment truth.

The generic tool trace uses the same evidence vocabulary as the release
verifier: `agent_turn_started`, `model_generation`, `tool_call_requested`,
`tool_result` and `agent_action_submitted`. The environment records a separate
`decision_consumed` and `rules_result` only after `DecisionRuntime` validates
and it consumes the choice. The full provider call trace, bounded private
reasoning and terminal arguments stay in admin-only decision rows;
public/player/god projections never receive those rows.

`council.cipher@1` is the Cipher Council baseline and imports no Werewolf code.
It deals a hidden Cipher minority with the `roles` seed and a proposer order
with the `order` seed. Every `council:<n>` participant receives only its own
faction, Cipher teammates when authorized, public deliberation history and the
exact current action schema. Public speech, nominations and votes create
explicit public events. Secret mission commitments are launched concurrently
for the proposed team, then resolved from environment-owned commitments.
Missing speech, nomination, vote and commitment each have a documented
no-synthetic-choice outcome; a missing secret commitment voids the mission and
produces an incomplete terminal result.

`council.cipher@2` preserves v1's configuration and rules, then adds a
simultaneous private Cipher strategy council before every public proposal
attempt. It requests `send_cipher_strategy_message` from each Cipher's own
Actor concurrently. The request observation contains only the bounded history
from earlier council rounds; no current-round message is released until every
current-round request reaches a terminal decision result. Accepted messages are
then emitted as private `council_cipher_message` events whose exact recipients
are all Cipher actor IDs. Council Actors and public projections do not receive
the message content. An authorized god projection can inspect omniscient
environment events, while the human God Console uses the separate
admin-capability `/trace` read to render a bounded, redacted decision trace.
Neither private event nor decision trace becomes an Agent observation. A
skipped or failed Cipher creates no message and no public unavailability event.
There is no team controller or leader: the environment only schedules and
delivers independent Actor decisions.

Generic Core runs use `run_core_llm_environment()` to construct one independent
`CoreToolActor` per resolved actor ID and attach a credential-free Router stats
delta to the result for artifact/smoke verification. Before each actor exists,
its actual `ModelConfig` must exactly match the credential-free default or
per-actor override manifest declared by `CoreRunSpec.actors`; generic artifact
provenance therefore cannot silently drift from the model that received the
request.

## 9. Frontend

The React application is a Harness Console. It renders:

- phase and environment state;
- exact public Agent output;
- human action requests;
- rule resolution and transparent failures;
- admin-only request/response pairings, legal actions/targets, hashes, parse status, latency and private reasoning.

There is no chat composer for AI seats, no fake typing stream and no second client-side interpretation of model output.

## 10. Known boundaries

- Offline and interactive Werewolf both use the generic registry/Core execution
  lifecycle. Interactive transport, capability handling, Werewolf-specific
  visibility projection and the v2 seat/action payload remain adapter-specific;
  the live run itself uses `CoreRunSpec.actors` and canonical `seat:<n>` IDs.
- Cipher Council v1/v2 are built-in production Core environment versions, but
  the interactive RoomManager/UI currently exposes Werewolf only. Cipher
  Council currently runs through the generic offline/Core runner and artifact
  path.
- Provider behavior can change independently of this repository.
- The API now has exact configurable CORS/WebSocket origins, stable
  liveness/readiness, bounded room retention, authenticated idle-room cleanup,
  process-local REST/WebSocket admission buckets and per-scope provider call/
  usage ledgers. The entire interactive API currently requires one worker,
  because room ownership, mutations, tasks/locks, WebSocket delivery state and
  persistence coordination are also process-local. Public multi-worker
  deployment needs an external room coordinator/shared state and pub/sub in
  addition to shared atomic counters, trusted proxy policy and service-level
  authentication.
- Research references provide design context only; they are not implementation verification or performance evidence.
