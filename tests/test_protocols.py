import json
from pathlib import Path

import numpy as np
import pytest

from layerbirth.protocols import (
    resolve_adaptive_tau,
    resolve_control_settings,
    resolve_fixed_tau,
    resolve_matched_tau_group,
    resolve_run_settings,
)


P_SLOW = np.array(
    [
        [0.9, 0.1],
        [0.1, 0.9],
    ]
)

P_FAST = np.array(
    [
        [0.6, 0.4],
        [0.4, 0.6],
    ]
)


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_fixed_tau_resolution_and_invalid_tau():
    cfg = _load("configs/protocols/fixed_tau.json")
    resolved = resolve_fixed_tau(cfg)
    assert resolved["tau_protocol_name"] == "fixed"
    assert resolved["resolved_tau"] == 3
    assert resolved["closure_strength_lambda"] == 0.35

    bad = _load("configs/protocols/fixed_tau.json")
    bad["tau_protocol"]["tau"] = 0
    with pytest.raises(ValueError):
        resolve_fixed_tau(bad)


def test_adaptive_tau_exact_values_and_clipping():
    cfg = _load("configs/protocols/adaptive_tau.json")
    resolved = resolve_adaptive_tau(cfg, P_SLOW)
    d = resolved["tau_resolution_details"]
    assert resolved["tau_protocol_name"] == "adaptive"
    assert np.isclose(d["mu2_abs"], 0.8, atol=1e-12)
    assert np.isclose(d["spectral_gap"], 0.2, atol=1e-12)
    assert d["tau_raw"] == 5
    assert resolved["resolved_tau"] == 5

    clip_cfg = _load("configs/protocols/adaptive_tau.json")
    clip_cfg["tau_protocol"]["alpha"] = 100.0
    clip_cfg["tau_protocol"]["tau_max"] = 7
    clipped = resolve_adaptive_tau(clip_cfg, P_SLOW)
    assert clipped["resolved_tau"] == 7
    assert clipped["tau_resolution_details"]["clipped"] is True


def test_matched_tau_group_forces_shared_tau():
    matched_cfg = _load("configs/protocols/matched_tau_group.json")
    resolved = resolve_matched_tau_group(
        matched_cfg["runs"],
        {"match_ref": P_SLOW, "match_follow": P_FAST},
    )
    ref = resolved["match_ref"]
    follow = resolved["match_follow"]
    assert ref["resolved_tau"] == 5
    assert follow["resolved_tau"] == 5
    assert follow["matched_reference_run_id"] == "match_ref"
    assert follow["matched_role"] == "follower"

    # Follower-by-itself adaptive would be 2, but matched must inherit 5.
    adaptive_like = {
        "run_id": "match_follow",
        "tau_protocol": {
            "name": "adaptive",
            "method": "spectral_gap_scaled",
            "alpha": 1.0,
            "gap_floor": 1e-8,
            "tau_min": 1,
            "tau_max": 32,
        },
        "control": {"closure_strength_lambda": 0.75},
    }
    follower_own = resolve_adaptive_tau(adaptive_like, P_FAST)
    assert follower_own["resolved_tau"] == 2
    assert follow["resolved_tau"] != follower_own["resolved_tau"]


def test_control_parameter_validation():
    ok = resolve_control_settings({"control": {"closure_strength_lambda": 0.5}})
    assert ok["closure_strength_lambda"] == 0.5

    with pytest.raises(ValueError):
        resolve_control_settings({"control": {"closure_strength_lambda": -0.1}})
    with pytest.raises(ValueError):
        resolve_control_settings({"control": {"closure_strength_lambda": 1.2}})


def test_resolved_settings_contains_protocol_and_control_fields():
    cfg = _load("configs/protocols/adaptive_tau.json")
    resolved = resolve_run_settings(cfg, P=P_SLOW)
    assert "resolved_tau" in resolved
    assert "tau_protocol_name" in resolved
    assert "closure_strength_lambda" in resolved
