import numpy as np
import pytest

from layerbirth.lifts import uniform_lift_family
from layerbirth.metrics import (
    affinity_metric,
    closure_error,
    objecthood_order_parameter,
    staging_gap,
)
from layerbirth.numeric import validate_row_stochastic
from layerbirth.substrates import (
    build_equilibrium_substrate_family,
    null_flat_mixing_family,
    reversible_block_family,
)


def _manual_two_block_lens_for_n8() -> np.ndarray:
    return np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)


def test_reversible_block_strong_vs_weak_separation_metrics():
    strong = reversible_block_family(
        n_blocks=2,
        block_size=4,
        intra_block_weight=1.0,
        inter_block_weight=0.05,
        self_weight=0.0,
    )
    weak = reversible_block_family(
        n_blocks=2,
        block_size=4,
        intra_block_weight=1.0,
        inter_block_weight=0.5,
        self_weight=0.0,
    )

    p_strong = strong["P"]
    p_weak = weak["P"]
    lens = np.asarray(strong["block_lens"], dtype=int)
    k = int(np.max(lens)) + 1

    validate_row_stochastic(p_strong)
    validate_row_stochastic(p_weak)
    np.testing.assert_allclose(p_strong, p_strong.T, atol=1e-12)
    np.testing.assert_allclose(p_weak, p_weak.T, atol=1e-12)

    _, lift_details = uniform_lift_family(lens, k)
    assert lift_details["UQ_identity_error"] <= 1e-12

    sg_strong, _ = staging_gap(p_strong, 1, lens, k)
    sg_weak, _ = staging_gap(p_weak, 1, lens, k)
    aff_strong, _ = affinity_metric(p_strong, tau=1)
    aff_weak, _ = affinity_metric(p_weak, tau=1)

    assert sg_strong > sg_weak
    assert abs(aff_strong) <= 1e-10
    assert abs(aff_weak) <= 1e-10


def test_metastable_seeded_family_reproducible_and_reversible_like():
    base = build_equilibrium_substrate_family(
        "metastable_block_family",
        n_blocks=2,
        block_size=4,
        intra_scale=1.0,
        inter_scale=0.1,
        seed=7,
        diagonal_bias=1.0,
    )
    same = build_equilibrium_substrate_family(
        "metastable_block_family",
        n_blocks=2,
        block_size=4,
        intra_scale=1.0,
        inter_scale=0.1,
        seed=7,
        diagonal_bias=1.0,
    )
    diff = build_equilibrium_substrate_family(
        "metastable_block_family",
        n_blocks=2,
        block_size=4,
        intra_scale=1.0,
        inter_scale=0.1,
        seed=13,
        diagonal_bias=1.0,
    )

    p_base = base["P"]
    p_same = same["P"]
    p_diff = diff["P"]
    lens = np.asarray(base["block_lens"], dtype=int)
    k = int(np.max(lens)) + 1

    validate_row_stochastic(p_base)
    np.testing.assert_allclose(p_base, p_same, atol=1e-15)
    assert not np.allclose(p_base, p_diff)

    aff, _ = affinity_metric(p_base, tau=1)
    assert abs(aff) <= 1e-10
    assert lens.shape[0] == p_base.shape[0]
    assert sorted(np.unique(lens).tolist()) == [0, 1]

    _, lift_details = uniform_lift_family(lens, k)
    assert lift_details["UQ_identity_error"] <= 1e-12


def test_null_flat_mixing_control_metrics():
    null = null_flat_mixing_family(n=8)
    p = null["P"]
    f = _manual_two_block_lens_for_n8()
    k = 2

    validate_row_stochastic(p)
    ce, _ = closure_error(p, 1, f, k)
    mobj, _ = objecthood_order_parameter(p, 1, f, k)
    sg, _ = staging_gap(p, 1, f, k)
    aff, _ = affinity_metric(p, tau=1)

    assert np.isclose(ce, 0.0, atol=1e-12)
    assert np.isclose(mobj, 0.0, atol=1e-12)
    assert np.isclose(sg, 0.0, atol=1e-12)
    assert np.isclose(aff, 0.0, atol=1e-12)


def test_substrate_validation_errors():
    with pytest.raises(ValueError):
        reversible_block_family(
            n_blocks=0,
            block_size=4,
            intra_block_weight=1.0,
            inter_block_weight=0.1,
        )
    with pytest.raises(ValueError):
        reversible_block_family(
            n_blocks=2,
            block_size=0,
            intra_block_weight=1.0,
            inter_block_weight=0.1,
        )
    with pytest.raises(ValueError):
        reversible_block_family(
            n_blocks=2,
            block_size=4,
            intra_block_weight=1.0,
            inter_block_weight=-0.1,
        )
    with pytest.raises(ValueError):
        null_flat_mixing_family(n=0)
