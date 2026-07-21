# Harness Quality Gates

This document is the completion contract for evolving the repository from a
single Werewolf runtime into a reusable, auditable adversarial-agent harness.
It separates framework truth, environment truth, Agent choice, evaluation
evidence, operations, and UI. A feature is complete only when its listed
evidence exists and passes; documentation or a plausible demo is insufficient.

## 1. Framework contract

| ID | Requirement | Required evidence | Current state |
| --- | --- | --- | --- |
| H-001 | Every accepted `ActionRequest` has one unique request ID and exactly one `agent_response`, `agent_response_failed`, `agent_response_cancelled`, or `agent_response_validation_failed` terminal row. | Protocol, runtime, transcript pairing, duplicate-ID, cancellation, and validator-defect tests. | Pass |
| H-002 | Legal actions express action membership, skip permission, payload shape, target requirement, and the exact target set without ambiguity; request-scoped tool schemas reject seats outside the visible roster or exact legal targets before handler execution. | Empty-target, missing-target, extra-target, illegal-target, dynamic-roster and ghost-seat tests. | Pass |
| H-003 | Environment truth cannot be overwritten by Agent output. | Environment contract and adversarial payload tests. | Pass for Werewolf and Cipher Council |
| H-004 | LLM and human decisions use the same versioned protocol boundary. | Shared contract and human lifecycle tests. | Pass |
| H-005 | Retry ownership and deadline ownership are singular and bounded. | Safe Router transport-attempt and Actor response-attempt traces, streaming timeout tests, and no retry multiplication tests. | Pass |
| H-006 | Public output is the exact accepted decision output; no fallback, rewrite, or fake stream is permitted. | Exact-output and failure-path tests. | Pass |
| H-007 | A new environment can run without importing Werewolf modules into the generic runtime. | Registered test environment plus production `council.cipher@1` executed through `run_core_llm_environment`; AST import-boundary, independent-Actor, per-actor model-manifest provenance, Core CLI exact-spec/trusted-plugin/artifact tests, tool-trace and artifact/smoke-evidence tests. | Pass: `src.harness.core_cli` runs one exact Core-tool-contract environment through an explicit trusted plugin reference and rejects legacy decision contracts before transport. |
| H-008 | Harness lifecycle is explicit: created, running, completed, timed out, failed, or cancelled. The low-level async runner records cancellation, completes bounded cleanup, and propagates the caller's `CancelledError`; a room/supervisor boundary records a serializable cancelled status. | State-transition and terminal-state tests, repeated-cancellation cleanup tests, room cancellation status tests, and cancellation trace pairing. | Pass: propagation is intentional control-flow semantics, not an environment failure |
| H-009 | A tool-using Agent has a bounded provider-input history without losing or corrupting canonical audit evidence. Compaction may replace only complete older tool-call/result groups and must expose whether its target was satisfied. | Atomic tool-group compaction, malformed/incomplete-group preservation, recent-group retention, full-history preservation, trace and metric tests in `tests/test_agent_session.py`. | Pass |

## 2. Provenance and reproducibility

| ID | Requirement | Required evidence | Current state |
| --- | --- | --- | --- |
| P-001 | Run specs are immutable, versioned, canonical, and include environment configuration plus all local seeds. | Core v1 schema/hash/ActorSpec, exact Werewolf v3 dispatch/migration, generic and Werewolf per-actor model/human provenance mismatch rejection, identical-seed deterministic request-ID/transcript evidence, and interactive shared room/state/run/transcript/request identity tests. | Pass: generic real-model Actors fail before transport when their safe binding is absent or mismatched; interactive persists canonical `CoreRunSpec` beside the legacy compatibility view and fail-closed cross-checks room/state/spec/transcript identities plus Legacy/Core shared semantics on restore. |
| P-002 | Policy schedules have one definition and preserve per-policy replicate semantics, paired seeds, counterbalancing, and stable run IDs. | Sequential and ABBA tests through CLI-to-batch. | Pass |
| P-003 | Resume accepts only rows matching the scheduled run ID and spec hash. | Stale/mismatched resume tests. | Pass |
| P-004 | Artifacts are written atomically and contain hashes that detect partial or altered files. | Interrupted-write, legacy/core round-trip, content/row tamper, schema-dispatch, and symlink tests. | Pass |
| P-005 | Credentials and capability tokens never enter specs, artifacts, logs, or public projections. | Recursive redaction, Core ActorSpec credential/overlap rejection, artifact, projection, hostile-string, capability lifecycle, no-store/no-referrer, and no-plaintext-log tests. | Partial: the one-process API/runtime boundary passes and browser capabilities remain memory-only; native/CLI query-token compatibility can still be observed by an outer access log, so public deployment requires header/subprotocol use plus an explicit proxy logging and authentication policy |
| P-006 | Transcript ordering and integrity can be verified independently of the live process. | Sequence, payload hash, run/schema identity, counts, stable-digest reconstruction, interactive source-index/global-trace reorder attacks, artifact verification, verified snapshot no-second-read tests, and verified summary re-derivation tests. | Pass: interactive restore additionally requires per-kind `source_idx=0..n-1`, one strictly increasing event/decision `_trace_seq` timeline and an exact room cursor. Newly written legacy v2/Core artifacts retain their versioned reconstruction checks; offline semantic verifiers consume one in-memory verified artifact snapshot rather than reopening files after hash verification. Standalone summary JSONL remains cache-only. |
| P-007 | Reproducibility claims explicitly exclude external-model nondeterminism. | Manifest fields and documentation. | Pass |

## 3. Adversarial environment contract

| ID | Requirement | Required evidence | Current state |
| --- | --- | --- | --- |
| G-001 | Werewolf phases form a finite executable state machine without busy loops or unbounded PK/death chains. | Transition and boundedness tests. | Pass |
| G-002 | Every advertised role has a complete observation, action, legality, resolution, memory, trace, and failure path. | Per-role scenario matrix. | Pass for every role advertised by `classic.v1` |
| G-003 | Simultaneous actions have documented ordering and deterministic conflict resolution. | `classic.v1` night-resolution table, exhaustive six-action permutation coverage in `tests/test_rules.py`, and seeded wolf tie-break registration-order coverage in `tests/test_orchestrator.py`. | Pass for `classic.v1` |
| G-004 | Team-private, seat-private, public, god, and admin information remain isolated under hostile payloads. Admin-only structured tool arguments are bounded/redacted, and ghost seats cannot enter seat-bearing tools. | Visibility audit, dynamic-roster, tool-trace redaction and audience projection tests. | Pass |
| G-005 | Strategic deception is allowed while protocol and rule violations remain rejectable and auditable. | Bluff tests separated from illegal-payload tests. | Pass |
| G-006 | Missing actions resolve only through explicit environment rules, never synthetic Agent choices. | Executable timeout/provider-failure matrix for `night_kill`, `see`, `guard`, doctor/witch `save`, witch `poison`, `hunter_shot`, `speak`, `vote`, and `last_words` in `tests/test_werewolf_missing_action_matrix.py`; each row verifies the shared `DecisionRuntime` failure terminal, no `decision_consumed`, and the explicit environment no-action resolution. | Pass |
| G-007 | Rule variants and role decks are validated before a run starts. | Exact validation in `src/game/roles.py`, `src/game/rules.py`, the Werewolf plugin and `src/harness/spec.py`; invalid-combination/default-deck/plugin/runner evidence in `tests/test_rules.py`, `tests/test_werewolf_plugin.py`, and `tests/test_harness_runner.py`. | Pass for `classic.v1`; no multi-variant support claimed |
| G-008 | Cipher Council preserves hidden factions, public deception space, secret simultaneous commitments and explicit missing-action semantics; v2 additionally preserves concurrent, faction-private coordination without a team controller. | `tests/test_cipher_council.py` covers independent actor observations, private/public/god/admin projection, role assignment, absent vote, missing nomination, mission void, deterministic transcript, v2 concurrent Cipher-only message delivery, `message + absent = scheduled request` accounting, generic artifact verification, snapshot no-second-read behavior, semantic tamper rejection, and raw artifact tamper rejection. `tests/test_core_llm_runner.py` verifies 15 tool decisions through the generic model-backed tool path plus the v2 `faction_size`/round/request/message/absence metrics and offline smoke contract. `verify_cipher_council_v2_artifacts()` consumes one verified artifact snapshot, then recomputes v2 faction size, round/request/message/absence counts, recipient routing, terminal-response delivery barrier, private tool schema, and observation isolation without returning strategy text or reasoning. The local real v2 artifact completed 15 provider/tool decisions with 0 provider failures/retries; both its offline smoke verifier and v2 semantic evidence verifier passed. | Pass for the `council.cipher@1` baseline and `council.cipher@2` private coordination. Real-provider runtime evidence exists for both; release evidence still requires attaching a credential-free artifact to the exact revision. |

## 4. Evaluation contract

| ID | Requirement | Required evidence | Current state |
| --- | --- | --- | --- |
| E-001 | Production summaries contain only recomputable outcome, trace, failure, latency, usage, and cost facts. | Summary schema, duplicate-row, transcript-attestation, forged-JSONL, artifact re-derivation, and private-payload exclusion tests for tool-loop and history-compaction metrics. | Pass for in-process rows and verified artifact re-derivation, including model/tool amplification, safe tool-failure histograms, compaction counts/affected requests, before/after character maxima and unsatisfied-target counts. Standalone summary JSONL remains a cache/checkpoint: ordinary totals may be displayed, but all derived evaluations are omitted until a transcript-backed row is rebuilt. |
| E-002 | Scenario suites cover protocol attacks, leaks, collusion-capable play, contradictory claims, timeouts, malformed responses, and provider faults. | Versioned `ScenarioSpec`/report schemas and eight executable generic-runner scenarios in `tests/test_harness_adversarial_scenarios.py`, including exact environment version, pairing, no-fabricated-choice and opaque-marker checks. | Pass |
| E-003 | Comparative experiments record paired design, sample size, uncertainty, and raw rows without unsupported causality. | Statistical report, attestation, duplicate/conflict, control-mismatch, and deterministic bootstrap tests. | Pass for attested rows: controlled `pair_id`/seed matching, exclusion diagnostics, paired differences, deterministic bootstrap intervals, raw rows, and `causal=false`. Hand-written or JSONL-only rows cannot create a comparison. |
| E-004 | Real-model validation proves nonzero provider calls and request/response pairing without exposing credentials. | Opt-in `src.harness.smoke` artifact verifier and a real provider artifact. | Runtime gate passed: r12 completed a six-seat game with 69 provider calls/successes, zero provider failures/retries, 31 request/terminal-response pairs, 69 tool call/result pairs, 31 consumed decisions (29 accepted rules resolutions plus 2 plurality `not_selected` wolf proposals), and six resolved Actor IDs (`seat:1` through `seat:6`). A generic `council.cipher@2` run also completed 15 provider calls/successes, 15 request/terminal-response pairs, 15 tool call/result pairs and 15 consumed decisions with zero provider failures/retries; its private-coordination metrics were faction size 2, one round, two requests, two messages and zero absences. Both local artifacts pass credential scanning and offline smoke verification. Release status remains pending until packaging attaches credential-free artifacts to the exact revision. |
| E-005 | Model self-reports are never treated as independent psychological or deception truth. | Schema and documentation checks. | Pass |

## 5. Operations and API

| ID | Requirement | Required evidence | Current state |
| --- | --- | --- | --- |
| O-001 | Room state and artifacts survive process restart when persistence is enabled. | Persistence adapter, source/Transcript/delivery rollback, terminal-before-Core-return restart, and credential-safe JSON Schema tests. | Pass; SQLite rewrites the full bounded snapshot per durable mutation, and clients see staged delivery only after that save succeeds. Long-run write amplification remains an operations cost rather than an unbounded-memory path. |
| O-002 | Room ownership, seat capabilities, token rotation/revocation, limits, cleanup and startup publication are explicit. | Authorization/lifecycle tests plus interactive shared-identity, per-seat AgentRegistry provenance, staged-construction/durable rollback, prepared-session execution, single-transcript sink, atomic Core terminal projection and cleanup tests. | Pass for single-worker RoomManager Phase B: Core owns environment timeout/cancellation/session-runtime cleanup/result lifecycle; RoomManager owns only the interactive task wrapper, result-to-room projection, capabilities, delivery and persistence. |
| O-003 | WebSocket replay and live delivery are ordered, deduplicated, bounded, and race-tested. | Concurrent connect/broadcast/reconnect tests. | Pass |
| O-004 | Public deployment has strict origins, rate/spend limits, redacted logs, and health/readiness endpoints. | Strict CORS parser, explicit WebSocket Origin/missing-Origin policy, room capacity/TTL/readiness/delete integration tests, plus request/provider rate and spend controls. | Pass for one process. The interactive API as a whole is single-worker: room ownership/mutations, game tasks/locks, WebSocket delivery streams/cursors, capability state, persistence coordination, admission counters and provider ledgers are process-local. Multi-worker deployment requires external coordination/shared state, locks and pub/sub, routing/reconnect guarantees, distributed counters/ledgers, and trusted-proxy policy. |
| O-005 | Provider clients and background tasks close on completion, timeout, cancellation, and shutdown. | Resource lifecycle, partial prepared-runtime, Core quarantine handoff and single-cancel shutdown tests. | Pass: stream, bounded Router/client cache, room shutdown and generic session/runtime cleanup are covered; unforceable Core/third-party tasks transfer to the run-scoped RoomManager quarantine, remain referenced, and are reported without repeatedly cancelling Core-owned sequential cleanup. |

## 6. Web UI (final phase)

The pass states above do not mean the repository is 100% complete. Interactive
Phase B is implemented and the current working tree produced the locally verified
r12 real-model artifact, but release evidence still requires packaging that
credential-free artifact with the exact revision. Multi-worker coordination,
interactive multi-environment room selection and independent psychological truth
or deception-quality calibration remain out of scope.

UI work starts only after H-001 through H-009 and the Werewolf G-series gates
are stable. The UI must be a protocol and environment console, not a chat-shaped
alternate decision path.

| ID | Requirement | Required evidence | Current state |
| --- | --- | --- | --- |
| U-001 | Spectator, player, god, admin trace, and replay expose only authorized data. | Desktop/mobile browser tests plus bounded/redacted structured tool-argument projection tests. | Pass: the live browser matrix completed 4 journeys at 1280x900 and 390x844 for spectator, player, god/admin trace and replay; public/private/god projections remain isolated, capabilities stay in memory rather than URL/query/storage, and only the admin-protected trace can render bounded tool arguments as private structured reasoning. |
| U-002 | Human controls render exactly the server-advertised schema and close on terminal events. | Component/browser tests for every action. | Pass: the same live matrix covers server-ordered target controls, speech text/bid, skip, stale-request rejection and terminal dialog cleanup on desktop/mobile. |
| U-003 | Replay has navigation, filtering, request/response pairing, and factual analysis without rerunning Agents. | Ended-run browser tests. | Pass: desktop/mobile ended-room journeys cover playhead navigation, filters, ActionRequest/DecisionEnvelope pairing and factual analysis; the live journey reaches terminal state, returns to the owning room with its in-memory capability and enters REST-only replay without opening a WebSocket or rerunning Agents. |
| U-004 | Reconnect, stale requests, duplicate events, long runs, and permanent WS errors have deterministic UX. | Network-fault browser tests. | Pass: reducer/component coverage verifies reconnect guards, duplicate delivery, long histories and action rejection; the live matrix verifies stale requests plus real 4410/4429 permanent closes remain disconnected until explicit manual retry. |
| U-005 | The console is readable and non-overlapping across supported viewports. | Screenshot and interaction verification. | Pass: lobby, waiting room, spectator, player, god/admin trace and replay are interaction/screenshot checked at 1280x900 and 390x844, including document/root overflow checks and mobile terminal-to-replay navigation. |

## 7. Release proof

A release candidate is acceptable only when all of the following are attached
to the same revision:

1. Backend unit, integration, adversarial, and property suites pass.
2. Frontend typecheck, build, component tests, and browser journeys pass.
3. A credential-free real-model smoke artifact passes the pairing verifier.
4. Artifact integrity verification passes after reload from disk.
5. Visibility audit reports no error-level issue for every audience.
6. Dependencies are locked and supported Python and Node versions documented.
7. The release working tree is committed and versioned.

Artifact verification proves that a three-file set is internally consistent
with its manifest and transcript digest. It is not source authentication: the
manifest is unsigned, so an actor able to replace the entire directory can
replace all hashes as well. A deployment that must resist that actor needs an
external signature, HMAC authority, or append-only digest anchor.
