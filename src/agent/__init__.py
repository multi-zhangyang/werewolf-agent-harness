"""Agent actors, cognition and per-seat tool-loop primitives.

Session exports are loaded lazily so ``harness.agent_protocol`` can import the
lightweight ``agent.schemas`` module without a package-level circular import.
"""

__all__ = [
    "AgentSession",
    "AgentSessionError",
    "AgentSessionLimits",
    "AgentSessionResult",
    "SessionStatus",
    "TerminalSubmission",
    "ToolExecutionContext",
    "ToolExecutionError",
    "ToolKind",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
]


def __getattr__(name: str):
    if name in __all__:
        from . import session

        return getattr(session, name)
    raise AttributeError(name)
