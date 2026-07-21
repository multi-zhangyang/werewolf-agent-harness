"""Plugin and session contracts for adversarial harness environments."""
from __future__ import annotations

import inspect
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .core_spec import ActorSpec
from .decision_runtime import DecisionRuntime
from .transcript import Transcript


ENVIRONMENT_PLUGIN_API_VERSION = "agent-harness.environment.v1"
EventSink = Callable[[dict[str, Any]], Awaitable[None] | None]
TraceSink = Callable[[dict[str, Any]], None]
AgentResolver = Callable[[str], Any]
# Harness lifecycle notifications are emitted from an async runner, but an
# interactive owner may use either a synchronous persistence hook or an async
# broadcaster.  ``EnvironmentRunEvidence`` normalizes both forms.
HarnessSink = Callable[[dict[str, Any]], Awaitable[None] | None]


class AgentBindingError(RuntimeError):
    """Raised when one run tries to bind an agent object to two identities."""


class AgentRegistry:
    """Stable per-run mapping from actor IDs to one agent object each.

    Environment plugins may ask for the same actor more than once.  Resolve
    it through this registry so the actor's memory/private state remains
    attached to one object for the whole run.  A resolver returning the same
    object for two different IDs is rejected, which prevents accidental
    cross-seat state sharing in generic environments.
    """

    def __init__(self, resolver: AgentResolver) -> None:
        if not callable(resolver):
            raise TypeError("agent resolver must be callable")
        self._resolver = resolver
        self._by_actor: dict[str, Any] = {}
        self._object_bindings: dict[int, str] = {}
        self._binding_failures: dict[str, str] = {}

    def resolve(self, actor_id: str) -> Any:
        key = str(actor_id).strip()
        if not key:
            raise AgentBindingError("actor identity must not be empty")
        if key in self._by_actor:
            return self._by_actor[key]
        previous_failure = self._binding_failures.get(key)
        if previous_failure is not None:
            raise AgentBindingError(previous_failure)

        agent = self._resolver(key)
        if agent is None:
            # A missing binding is an identity/configuration failure, not an
            # empty Agent.  Cache the failure so a resolver whose backing
            # store changes mid-run cannot silently bind this actor later.
            reason = f"agent resolver returned no agent for actor {key!r}"
            self._binding_failures[key] = reason
            raise AgentBindingError(reason)

        declared_id = getattr(agent, "actor_id", None)
        if declared_id is not None and str(declared_id).strip() != key:
            raise AgentBindingError(
                f"resolved agent identity does not match actor {key!r}"
            )
        object_key = id(agent)
        previous = self._object_bindings.get(object_key)
        if previous is not None and previous != key:
            raise AgentBindingError(
                f"one agent object cannot serve multiple actors ({previous!r}, {key!r})"
            )
        self._by_actor[key] = agent
        self._object_bindings[object_key] = key
        return agent

    def __len__(self) -> int:
        return len(self._by_actor)

    def snapshot(self) -> dict[str, Any]:
        """Return a detached mapping for authorized run diagnostics."""
        return dict(self._by_actor)


@dataclass
class EnvironmentRunEvidence:
    """Route one run's evidence to a canonical transcript or full sinks.

    A sink is deliberately a *full* sink: when supplied, it owns the complete
    append operation for that evidence kind (including any source index,
    persistence, and live delivery required by an interactive owner).  The
    core must not append first and then notify the sink, because that creates a
    second transcript/source row.  With no sink, the generic harness appends
    directly to ``transcript``.

    The same object should be used while constructing ``EnvironmentRunContext``
    and ``DecisionRuntime``::

        evidence = EnvironmentRunEvidence(transcript, event_sink=..., ...)
        runtime = DecisionRuntime(on_trace=evidence.emit_trace, ...)
        context = EnvironmentRunContext(
            ..., emit_event=evidence.emit_event, emit_trace=evidence.emit_trace,
        )

    External sinks must append to this exact ``transcript`` instance.  The
    runner validates its run identity before taking ownership of a prepared
    session; it intentionally does not duplicate or second-guess a full sink's
    source-indexed mutation.
    """

    transcript: Transcript
    event_sink: EventSink | None = None
    trace_sink: TraceSink | None = None
    harness_sink: HarnessSink | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transcript, Transcript):
            raise TypeError("transcript must be a Transcript")
        for name, sink in (
            ("event_sink", self.event_sink),
            ("trace_sink", self.trace_sink),
            ("harness_sink", self.harness_sink),
        ):
            if sink is not None and not callable(sink):
                raise TypeError(f"{name} must be callable")

    async def emit_event(self, payload: dict[str, Any]) -> None:
        """Emit an environment event through one canonical full sink."""
        if not isinstance(payload, dict):
            raise TypeError("environment events must be dictionaries")
        if self.event_sink is None:
            self.transcript.append("event", payload)
            return
        result = self.event_sink(payload)
        if inspect.isawaitable(result):
            await result

    def emit_trace(self, payload: dict[str, Any]) -> None:
        """Emit a synchronous decision trace through one canonical sink."""
        if not isinstance(payload, dict):
            raise TypeError("decision traces must be dictionaries")
        if self.trace_sink is None:
            self.transcript.append("decision", payload)
            return
        result = self.trace_sink(payload)
        # DecisionRuntime's trace contract is synchronous.  Do not silently
        # create an un-awaited coroutine that would lose evidence.
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise TypeError("trace_sink must be synchronous")

    async def emit_harness(self, payload: dict[str, Any]) -> None:
        """Emit a lifecycle row through one canonical full sink."""
        if not isinstance(payload, dict):
            raise TypeError("harness events must be dictionaries")
        if self.harness_sink is None:
            self.transcript.append("harness", payload)
            return
        result = self.harness_sink(payload)
        if inspect.isawaitable(result):
            await result


@dataclass(frozen=True)
class DecisionContract:
    envelope_type: type[Any]
    validate_envelope: Callable[[Any, Any], Any]


class EnvironmentDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    plugin_api_version: str = ENVIRONMENT_PLUGIN_API_VERSION
    required_seeds: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    @field_validator("id", "version", "plugin_api_version")
    @classmethod
    def _nonempty_identity(cls, value: str) -> str:
        identity = str(value).strip()
        if not identity:
            raise ValueError("environment descriptor identity must not be empty")
        return identity

    @field_validator("required_seeds", "capabilities")
    @classmethod
    def _unique_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        labels = tuple(str(item).strip() for item in value)
        if any(not item for item in labels):
            raise ValueError("environment descriptor labels must not be empty")
        if len(labels) != len(set(labels)):
            raise ValueError("environment descriptor labels must be unique")
        return labels


class EnvironmentOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    terminal: bool
    status: Literal["completed", "incomplete"] = "completed"
    termination_reason: str | None = None
    outcome: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _incomplete_outcomes_need_a_reason(self) -> "EnvironmentOutcome":
        if not self.terminal:
            raise ValueError("EnvironmentOutcome must describe a terminal result")
        if self.status == "incomplete" and not (self.termination_reason or "").strip():
            raise ValueError("incomplete environment outcomes require termination_reason")
        return self


@dataclass(frozen=True)
class EnvironmentRunContext:
    run_id: str
    config: BaseModel
    seeds: Mapping[str, int]
    decision_runtime: DecisionRuntime
    emit_event: EventSink
    emit_trace: TraceSink
    resolve_agent: AgentResolver
    metadata: Mapping[str, Any]
    actor_spec: ActorSpec = field(default_factory=ActorSpec)
    # RNG streams are stateful resources owned by one run context.  Keeping a
    # cache here is important: constructing ``Random(seed)`` on every access
    # silently rewinds a namespace and makes a plugin's outcome depend on how
    # often it asks for the stream rather than on the declared seed.
    _rng_streams: dict[str, random.Random] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def rng(self, namespace: str) -> random.Random:
        name = str(namespace).strip()
        if not name:
            raise KeyError("environment seed namespace must not be empty")
        try:
            seed = self.seeds[name]
        except KeyError as err:
            raise KeyError(f"environment seed namespace is unavailable: {name}") from err
        stream = self._rng_streams.get(name)
        if stream is None:
            stream = random.Random(seed)
            self._rng_streams[name] = stream
        return stream


class EnvironmentSession(Protocol):
    async def run(self) -> EnvironmentOutcome:
        """Execute the environment until a terminal outcome or an error."""

    async def aclose(self) -> None:
        """Release all resources owned by this session."""


class EnvironmentPlugin(Protocol):
    descriptor: EnvironmentDescriptor
    decision_contract: DecisionContract

    def resolve_config(
        self,
        raw_config: Mapping[str, Any],
        seeds: Mapping[str, int],
    ) -> BaseModel:
        """Validate defaults and return the canonical environment config."""

    async def create_session(self, context: EnvironmentRunContext) -> EnvironmentSession:
        """Create one isolated run session."""
