"""Typed failures at the environment-to-agent decision boundary."""
from __future__ import annotations


class AgentDecisionError(RuntimeError):
    """One ActionRequest failed to produce a consumable DecisionEnvelope."""
