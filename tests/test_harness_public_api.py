"""Public harness exports remain importable during the generic-core migration."""
from __future__ import annotations

import src.harness as harness
from src.agent.schemas import AgentAction, Decision


def test_public_api_has_no_dangling_exports() -> None:
    for name in harness.__all__:
        assert getattr(harness, name) is not None, name


def test_legacy_and_core_protocols_are_explicitly_distinguishable() -> None:
    assert harness.ActionRequest is not harness.CoreActionRequest
    assert harness.DecisionEnvelope is not harness.CoreDecisionEnvelope
    assert harness.RunManifest is not harness.CoreRunManifest
    decision = Decision(action=AgentAction.VOTE, target_seat=4)
    assert harness.decision_target_seat(decision) == 4
