"""smoke_e2e helper tests; no real LLM calls."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_smoke_module():
    path = Path(__file__).with_name("smoke_e2e.py")
    spec = importlib.util.spec_from_file_location("smoke_e2e_module", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_is_complete_requires_scores_and_bounded_dims():
    smoke = _load_smoke_module()

    assert smoke._quality_is_complete({
        "game_quality": 0.75,
        "scores": [{
            "seat": 1,
            "RI": 0.7,
            "SJ": 0.6,
            "DR": 0.5,
            "PS": 0.4,
            "CT": 0.3,
        }],
    })
    assert not smoke._quality_is_complete({"game_quality": 0.75, "scores": []})
    assert not smoke._quality_is_complete({
        "game_quality": 1.2,
        "scores": [{"RI": 0.7, "SJ": 0.6, "DR": 0.5, "PS": 0.4, "CT": 0.3}],
    })
    assert not smoke._quality_is_complete({
        "game_quality": 0.75,
        "scores": [{"RI": 0.7, "SJ": 0.6, "DR": 0.5, "PS": 0.4}],
    })

