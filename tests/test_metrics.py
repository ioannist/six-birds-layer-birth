import numpy as np
import pytest

from layerbirth.metrics import (
    affinity_metric,
    closure_error,
    holonomy_metric,
    metric_contract_defaults,
    objecthood_order_parameter,
    staging_gap,
)
from layerbirth.numeric import packaging_projector, pushforward_matrix, uniform_lift_matrix


def _lens_4():
    f = np.array([0, 0, 1, 1], dtype=int)
    return f, 2


def test_metric_contract_helper_has_defaults_and_alternates():
    contract = metric_contract_defaults()
    assert contract["CE"]["default"] == "idempotence_defect_tv"
    assert "retention_error_max" in contract["CE"]["alternates"]
    assert contract["Hol"]["default"] == "lift_route_mismatch_tv"


def test_toy1_exact_closure_defaults():
    f, k = _lens_4()
    q = pushforward_matrix(f, k)
    u = uniform_lift_matrix(f, k)
    p = packaging_projector(q, u)

    ce, ce_details = closure_error(p, 1, f, k)
    mobj, _ = objecthood_order_parameter(p, 1, f, k)
    sg, _ = staging_gap(p, 1, f, k)
    aff, _ = affinity_metric(p, tau=1)
    hol, _ = holonomy_metric(f, 2, np.array([0, 0, 0, 0]), 1)

    assert ce == 0.0
    assert ce_details["retention_error_max"] == 0.0
    assert mobj == 1.0
    assert np.isclose(sg, 1.0, atol=1e-12)
    assert aff == 0.0
    assert hol == 0.0


def test_toy2_reversible_defaults():
    f, k = _lens_4()
    p_rev = np.array(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.2, 0.6, 0.2, 0.0],
            [0.0, 0.2, 0.6, 0.2],
            [0.0, 0.0, 0.2, 0.8],
        ]
    )
    ce, ce_details = closure_error(p_rev, 1, f, k)
    mobj, mobj_details = objecthood_order_parameter(p_rev, 1, f, k)
    sg, _ = staging_gap(p_rev, 1, f, k)
    aff, _ = affinity_metric(p_rev, tau=1)

    assert np.isclose(ce, 0.1, atol=1e-12)
    np.testing.assert_allclose(
        np.array(ce_details["retention_error_per_macro_state"]),
        np.array([0.1, 0.1]),
        atol=1e-12,
    )
    assert np.isclose(mobj, 0.8, atol=1e-12)
    np.testing.assert_allclose(
        np.array(mobj_details["retention_error_per_macro_state"]),
        np.array([0.1, 0.1]),
        atol=1e-12,
    )
    assert np.isclose(sg, 0.2828427125, atol=1e-10)
    assert np.isclose(aff, 0.0, atol=1e-12)


def test_toy3_driven_cycle_defaults():
    f, k = _lens_4()
    p_drive = np.array(
        [
            [0.1, 0.8, 0.0, 0.1],
            [0.1, 0.1, 0.8, 0.0],
            [0.0, 0.1, 0.1, 0.8],
            [0.8, 0.0, 0.1, 0.1],
        ]
    )
    ce, ce_details = closure_error(p_drive, 1, f, k)
    mobj, mobj_details = objecthood_order_parameter(p_drive, 1, f, k)
    sg, _ = staging_gap(p_drive, 1, f, k)
    aff, aff_details = affinity_metric(p_drive, tau=1)

    assert np.isclose(ce, 0.36, atol=1e-12)
    np.testing.assert_allclose(
        np.array(ce_details["retention_error_per_macro_state"]),
        np.array([0.45, 0.45]),
        atol=1e-12,
    )
    assert np.isclose(mobj, 0.1, atol=1e-12)
    np.testing.assert_allclose(
        np.array(mobj_details["retention_error_per_macro_state"]),
        np.array([0.45, 0.45]),
        atol=1e-12,
    )
    assert np.isclose(sg, 0.0928932188, atol=1e-10)
    assert aff > 0.0
    assert np.isclose(aff, 1.4556090792, atol=1e-10)
    assert np.isclose(aff_details["flux_l1_asymmetry"], 0.7, atol=1e-12)


def test_toy4_trivial_one_object_defaults():
    f = np.array([0, 0, 0], dtype=int)
    k = 1
    p_triv = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    ce, ce_details = closure_error(p_triv, 1, f, k)
    mobj, _ = objecthood_order_parameter(p_triv, 1, f, k)
    sg, _ = staging_gap(p_triv, 1, f, k)
    aff, _ = affinity_metric(p_triv, tau=1)

    assert ce == 0.0
    np.testing.assert_allclose(
        np.array(ce_details["retention_error_per_macro_state"]),
        np.array([0.0]),
        atol=1e-12,
    )
    assert mobj == 0.0
    assert np.isclose(sg, 1.0, atol=1e-12)
    assert np.isclose(aff, 0.0, atol=1e-12)


def test_auxiliary_holonomy_toy_nonzero():
    f_fine = np.array([0, 0, 1, 2, 2, 2], dtype=int)
    f_coarse = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    hol, details = holonomy_metric(f_fine, 3, f_coarse, 2)
    assert np.isclose(hol, 1.0 / 6.0, atol=1e-12)
    np.testing.assert_allclose(
        np.array(details["per_coarse_row_tv"]),
        np.array([1.0 / 6.0, 0.0]),
        atol=1e-12,
    )


def test_holonomy_hierarchy_validation_errors():
    # Fine label 0 spans two coarse labels -> invalid hierarchy.
    bad_fine = np.array([0, 0, 1, 1], dtype=int)
    bad_coarse = np.array([0, 1, 1, 1], dtype=int)
    with pytest.raises(ValueError):
        holonomy_metric(bad_fine, 2, bad_coarse, 2)

    f_fine = np.array([0, 0, 1, 1], dtype=int)
    f_coarse = np.array([0, 0, 1, 1], dtype=int)
    with pytest.raises(ValueError):
        holonomy_metric(
            f_fine, 2, f_coarse, 2, method="dynamic_macro_route_mismatch_tv"
        )
