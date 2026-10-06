import json
from pathlib import Path

import numpy as np
import pytest

from layerbirth.metrics import (
    affinity_metric,
    closure_error,
    holonomy_metric,
    objecthood_order_parameter,
    staging_gap,
)
from layerbirth.numeric import validate_row_stochastic
from layerbirth.substrates import (
    build_substrate_family,
    driven_cycle_family,
    holonomy_control_family,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cfg(name: str) -> dict:
    path = _repo_root() / "configs" / "substrates" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _half_ring_lens() -> np.ndarray:
    return np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)


def test_driven_cycle_affinity_behaviors_and_reproducibility():
    low_cfg = _cfg("driven_cycle_low_bias.json")
    high_cfg = _cfg("driven_cycle_high_bias.json")

    low_a = build_substrate_family(low_cfg["name"], **low_cfg["params"])
    low_b = build_substrate_family(low_cfg["name"], **low_cfg["params"])
    high = build_substrate_family(high_cfg["name"], **high_cfg["params"])

    np.testing.assert_allclose(low_a["P"], low_b["P"], atol=1e-15)
    validate_row_stochastic(low_a["P"])
    validate_row_stochastic(high["P"])

    aff_low, _ = affinity_metric(low_a["P"], tau=1)
    aff_high, _ = affinity_metric(high["P"], tau=1)
    assert aff_low > 0.0
    assert aff_high > aff_low
    assert np.isfinite(aff_low)
    assert np.isfinite(aff_high)

    eq = driven_cycle_family(n=8, self_weight=0.1, forward_weight=0.45, backward_weight=0.45)
    aff_eq, _ = affinity_metric(eq["P"], tau=1)
    assert np.isclose(aff_eq, 0.0, atol=1e-10)


def test_holonomy_control_balanced_vs_unbalanced_clean_control():
    bal_cfg = _cfg("holonomy_control_balanced.json")
    unbal_cfg = _cfg("holonomy_control_unbalanced.json")

    bal = build_substrate_family(bal_cfg["name"], **bal_cfg["params"])
    unbal = build_substrate_family(unbal_cfg["name"], **unbal_cfg["params"])

    np.testing.assert_allclose(bal["P"], unbal["P"], atol=1e-15)
    np.testing.assert_array_equal(bal["coarse_lens"], unbal["coarse_lens"])
    assert not np.array_equal(bal["fine_lens"], unbal["fine_lens"])

    p = bal["P"]
    coarse = np.asarray(bal["coarse_lens"], dtype=int)
    k = int(np.max(coarse)) + 1

    ce_bal, _ = closure_error(p, 1, coarse, k)
    mobj_bal, _ = objecthood_order_parameter(p, 1, coarse, k)
    sg_bal, _ = staging_gap(p, 1, coarse, k)
    aff_bal, _ = affinity_metric(p, tau=1)

    ce_unbal, _ = closure_error(unbal["P"], 1, coarse, k)
    mobj_unbal, _ = objecthood_order_parameter(unbal["P"], 1, coarse, k)
    sg_unbal, _ = staging_gap(unbal["P"], 1, coarse, k)
    aff_unbal, _ = affinity_metric(unbal["P"], tau=1)

    hol_bal, _ = holonomy_metric(bal["fine_lens"], 4, bal["coarse_lens"], 2)
    hol_unbal, _ = holonomy_metric(unbal["fine_lens"], 4, unbal["coarse_lens"], 2)

    assert np.isclose(hol_bal, 0.0, atol=1e-12)
    assert np.isclose(hol_unbal, 0.25, atol=1e-12)
    assert np.isclose(aff_bal, 0.0, atol=1e-10)
    assert np.isclose(aff_unbal, 0.0, atol=1e-10)

    assert np.isclose(ce_bal, ce_unbal, atol=1e-12)
    assert np.isclose(mobj_bal, mobj_unbal, atol=1e-12)
    assert np.isclose(sg_bal, sg_unbal, atol=1e-12)


def test_driven_holonomy_error_paths():
    with pytest.raises(ValueError):
        driven_cycle_family(n=3, self_weight=0.1, forward_weight=0.6, backward_weight=0.3)
    with pytest.raises(ValueError):
        driven_cycle_family(n=8, self_weight=-0.1, forward_weight=0.6, backward_weight=0.3)
    with pytest.raises(ValueError):
        holonomy_control_family(
            n_blocks=2,
            block_size=4,
            fine_block_sizes_by_coarse=[[1, 1], [2, 2]],
        )
