"""Counterexamples to incorrect mathematical shortcuts in the observable layer."""

import json
import math

import numpy as np
import pytest

from layerbirth.boundaries import first_threshold_lambda
from layerbirth.controls import protocol_trap_control
from layerbirth.lenses import manual_partition_lens
from layerbirth.lifts import stationary_within_fiber_lift
from layerbirth.lifts import prototype_lift_family
from layerbirth.metrics import affinity_metric, default_metric_bundle
from layerbirth.numeric import kernel_power, row_normalize, stationary_distribution, validate_lift_matrix
from layerbirth.observable_map import compute_p4_dual_shift_min, compute_structural_boundary_lambda
from layerbirth.scaling import binder_like_cumulant
from layerbirth.taxonomy import (
    estimate_affinity_reference, estimate_structural_boundaries,
    evaluate_p5_activation, evaluate_p6_drive_activation,
)


def test_one_way_cycle_has_infinite_reversal_divergence():
    p = np.roll(np.eye(4), 1, axis=1)
    value, details = affinity_metric(p)
    assert value == math.inf
    assert details["support_asymmetric_directed_edges"] == 4
    assert affinity_metric(p, method="edge_log_ratio_rms")[0] == math.inf
    # Subsampling at its period erases the observed arrow; that is not micro EPR.
    assert affinity_metric(p, tau=4)[0] == 0.0


def test_periodic_reversible_chain_does_not_fake_drive():
    p = np.array([[0., 1., 0.], [.25, 0., .75], [0., 1., 0.]])
    value, details = affinity_metric(p, max_iter=1)
    np.testing.assert_allclose(details["stationary_distribution"], [.125, .5, .375], atol=1e-14)
    assert value < 1e-25
    assert details["converged"]
    u, _ = stationary_within_fiber_lift(p, [0, 1, 0], 2, max_iter=1)
    np.testing.assert_allclose(u[0], [.25, 0., .75], atol=1e-14)


def test_transient_edges_do_not_create_stationary_entropy():
    p = np.array([[0., .25, .75], [0., 1., 0.], [0., 0., 1.]])
    pi, _, _ = stationary_distribution(p)
    # Uniform initial mass is absorbed into two different recurrent classes.
    np.testing.assert_allclose(pi, [0., 5/12, 7/12], atol=1e-14)
    assert affinity_metric(p)[0] == 0.0


def test_rare_positive_reverse_flux_is_not_deleted_by_cutoff():
    p = np.array([[.1, .9 - 1e-14, 1e-14],
                  [1e-14, .1, .9 - 1e-14],
                  [.9 - 1e-14, 1e-14, .1]])
    expected = (.9 - 2e-14) * math.log((.9 - 1e-14) / 1e-14)
    low, _ = affinity_metric(p, flux_eps=0.0)
    high, details = affinity_metric(p, flux_eps=.4)
    assert high == pytest.approx(expected, rel=1e-13)
    assert low == high
    assert details["support_asymmetric_directed_edges"] == 0


def test_nonuniform_reversible_positive_kernel_has_no_drive_with_tiny_iteration_budget():
    p = np.array([[.9, .1], [.4, .6]])
    value, details = affinity_metric(p, max_iter=1)
    np.testing.assert_allclose(details["stationary_distribution"], [.8, .2], atol=1e-14)
    assert value < 1e-25


@pytest.mark.parametrize("f", [[0., .9, 1., 1.], [0., math.nan, 1., 1.]])
def test_lens_frontends_do_not_silently_round_invalid_partitions(f):
    with pytest.raises(ValueError):
        manual_partition_lens(f)
    with pytest.raises(ValueError):
        default_metric_bundle(np.eye(4), f)


def test_illegal_pushforward_is_not_a_packaging_certificate():
    with pytest.raises(ValueError):
        validate_lift_matrix([[2.], [1.], [0.]], [[0., 1., 0.]])
    with pytest.raises(ValueError):
        kernel_power([[.5, .5, 0.], [.5, .5, 0.]], 1)
    with pytest.raises(ValueError):
        kernel_power(np.eye(2), 1.5)
    np.testing.assert_allclose(row_normalize([[1e-30, 2e-30]]), [[1/3, 2/3]])


def test_threshold_interpolation_preserves_small_asymmetric_crossings():
    target = .025
    rows = [{"closure_strength_lambda": 0., "y": target + 1e-9},
            {"closure_strength_lambda": 1., "y": target - 3e-9}]
    assert first_threshold_lambda(rows, "y", target, "leq") == pytest.approx(.25)


def test_affinity_reference_uses_both_structural_boundaries():
    rows = [{"closure_strength_lambda": 0., "affinity": 0.},
            {"closure_strength_lambda": 1., "affinity": 1.}]
    ref = estimate_affinity_reference(rows, {"ce_boundary_lambda": .2, "mobj_boundary_lambda": .8})
    assert ref["affinity_ref"] == pytest.approx(.8)
    missing = estimate_affinity_reference(rows, {"ce_boundary_lambda": .2}, {"affinity_ref": 0.})
    assert not missing["affinity_ref_available"]
    assert math.isnan(missing["affinity_ref"])
    thresholds = {"p6_inactive_max": 1e-6, "p6_active_min": 1e-3}
    assert evaluate_p6_drive_activation(missing["affinity_ref"], thresholds)["state"] == "unknown"
    assert compute_structural_boundary_lambda(.2, None) is None


def test_separate_nonmonotone_proxy_hits_do_not_certify_joint_birth():
    rows = [{"closure_strength_lambda": 0., "closure_error": 0., "objecthood_order": 0.},
            {"closure_strength_lambda": 1., "closure_error": .1, "objecthood_order": 1.}]
    summary = estimate_structural_boundaries(rows)
    thresholds = {"ce_target": .025, "mobj_target": .9,
                  "ce_inactive_min": .05, "mobj_inactive_max": .85}
    assert evaluate_p5_activation(summary, thresholds)["state"] == "unknown"


def test_staging_map_uses_shift_magnitudes():
    assert compute_p4_dual_shift_min(-.2, .3) == pytest.approx(.2)


def test_binder_ratio_is_scale_invariant_for_nonzero_samples():
    sample = np.array([-1., 2., 3., -4.])
    expected, _ = binder_like_cumulant(sample)
    for scale in (1e-4, 1e-50):
        actual, details = binder_like_cumulant(sample * scale)
        assert actual == pytest.approx(expected)
        assert details["defined"]
    assert binder_like_cumulant([0., 0.])[1]["zero_sample_fallback"]


def test_protocol_schedule_rejects_duplicate_phases():
    with pytest.raises(ValueError):
        protocol_trap_control(schedule_order=["pair01", "pair12", "pair20", "pair01"])


def test_explicit_lift_is_used_by_the_measurement():
    p = np.array([[.8, 0., .2, 0.], [0., .2, 0., .8],
                  [.1, 0., .9, 0.], [0., .7, 0., .3]])
    lens = [0, 0, 1, 1]
    u, _ = prototype_lift_family(lens, 2, prototype_indices=[0, 2])
    chosen = default_metric_bundle(p, lens, U_f=u, lift_name="prototype_lift_family")
    uniform = default_metric_bundle(p, lens)
    # Macro prototype retention probabilities are .8 and .9; normalized trace
    # equals .7. Uniform retention probabilities are .5 and .6, giving .1.
    assert chosen["objecthood_order"] == pytest.approx(.7)
    assert uniform["objecthood_order"] == pytest.approx(.1)
    assert chosen["closure_error"] != pytest.approx(uniform["closure_error"])


def test_missing_controls_cannot_pass_a_no_fake_arrow_gate(tmp_path):
    from layerbirth.dashboard import summarize_no_fake_arrow
    (tmp_path / "summary.json").write_text(json.dumps({"rows": [
        {"case_name": "protocol_trap_hidden_schedule", "driven_candidate": True}
    ]}))
    assert not summarize_no_fake_arrow(tmp_path)["no_fake_arrow_checks_passed"]


def test_shadow_susceptibility_uses_system_size_and_is_replication_invariant():
    from layerbirth.phenomenology import group_shadow_panel_observations
    rows = [{"size": 16, "closure_strength_lambda": .5, "delta_mobj": v} for v in (-.1, .1)]
    a = group_shadow_panel_observations(rows)[0]
    b = group_shadow_panel_observations(rows * 3)[0]
    assert a["susceptibility_delta_mobj"] == pytest.approx(.16)
    assert b["susceptibility_delta_mobj"] == pytest.approx(a["susceptibility_delta_mobj"])


def test_exact_structural_reduction_against_known_packaging():
    from fractions import Fraction as F
    from layerbirth.exact_audit import controlled_kernel, structural_metrics
    p = [[F(1,4)]*4 for _ in range(4)]
    packaged = controlled_kernel(p, F(1))
    assert structural_metrics(packaged, 1) == (F(0), F(1))
    assert structural_metrics(packaged, 2) == (F(0), F(1))


def test_micro_arrow_does_not_descend_to_every_lens():
    from layerbirth.numeric import macro_kernel, pushforward_matrix, uniform_lift_matrix
    p = np.array([[.1, .8, 0., .1], [.1, .1, .8, 0.],
                  [0., .1, .1, .8], [.8, 0., .1, .1]])
    f = [0, 1, 0, 1]
    q, u = pushforward_matrix(f, 2), uniform_lift_matrix(f, 2)
    assert affinity_metric(p)[0] > 1.0
    assert affinity_metric(macro_kernel(p, 1, q, u))[0] < 1e-25
    assert default_metric_bundle(p, f)["affinity_carrier"] == "microstate_kernel"
