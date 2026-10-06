"""Counterexamples to unsupported persistence, coordinates and summary claims."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from layerbirth.capacity import compute_capacity_depth_proxies, extract_birth_lambda_reference, load_confirmed_class_references
from layerbirth.class3 import evaluate_strict_confirmation_variant
from layerbirth.observable_map import _extract_class_i_ii, _extract_class_iii_iv, _affinity_onset_from_rows, summarize_observable_separation
from layerbirth.freeze import _claim_coverage, format_ready_for_writing_status
from layerbirth.robustness import evaluate_lens_class_profile, summarize_lens_suite
from layerbirth.scaling import _resample_observations, grid_search_collapse_fit
from layerbirth.upstream_bridge import _kernel_from_upstream_substrate, classify_upstream_bridge_result

ROOT = Path(__file__).resolve().parents[1]


def test_one_size_or_missing_sizes_cannot_confirm_scale_persistence():
    row = {'size': 16, 'candidate_is_class_iii_strict': True, 'staging_shift_ce': .2, 'staging_shift_mobj': .4}
    single = evaluate_strict_confirmation_variant([row])
    assert not single['strict_hit_persistent']
    assert not single['scale_invariance_supported']
    partial = evaluate_strict_confirmation_variant([row, {**row, 'size': 32}], expected_sizes=[16, 32, 64])
    assert not partial['size_panel_complete']
    assert partial['final_variant_label'] != 'strict_confirmed'
    with pytest.raises(ValueError):
        evaluate_strict_confirmation_variant([row, row])
    with pytest.raises(ValueError):
        evaluate_strict_confirmation_variant([])


def test_cv_is_scale_invariant_and_near_miss_is_required_at_every_size():
    rows = [{'size': n, 'candidate_is_class_iii_strict': False, 'staging_shift_ce': .2, 'staging_shift_mobj': m}
            for n, m in [(16, .2), (32, .01)]]
    result = evaluate_strict_confirmation_variant(rows)
    tiny = evaluate_strict_confirmation_variant([{**r, 'staging_shift_ce': r['staging_shift_ce'] * 1e-100, 'staging_shift_mobj': r['staging_shift_mobj'] * 1e-100} for r in rows])
    assert result['shift_mobj_cv'] == pytest.approx(tiny['shift_mobj_cv'])
    assert result['final_variant_label'] != 'persistent_near_miss'


def test_capacity_depth_stops_at_an_earlier_failure_and_area_uses_time_spacing():
    rows = [{'lambda_name': channel, 'tau': t, 'closure_error': ce, 'objecthood_order': m, 'affinity': 0.}
            for channel in ('post', 'sat') for t, ce, m in [(1, .01, 1.), (2, .1, .5), (8, .01, 1.)]]
    p = compute_capacity_depth_proxies(rows, .7)
    assert p['depth_survival_post'] == p['depth_survival_sat'] == 1
    assert p['objecthood_tau_area_post'] == pytest.approx(.75)
    assert extract_birth_lambda_reference({'ce_boundary_lambda': .5, 'mobj_boundary_lambda': None})['birth_lambda_ref'] is None


def test_capacity_birth_is_remeasured_at_each_size():
    cfg = json.loads((ROOT / 'configs/campaigns/capacity_depth_extension.json').read_text())
    cfg['size_panel'] = [16, 64]
    refs = load_confirmed_class_references(cfg)
    assert refs['Class-II']['birth_by_size'][16]['birth_lambda_ref'] != refs['Class-II']['birth_by_size'][64]['birth_lambda_ref']
    assert all(ref['measurement_recomputed'] for ref in refs.values())


def test_map_uses_actual_size_and_same_tau_affinity_reference():
    cfg = json.loads((ROOT / 'configs/dashboards/four_class_observable_map.json').read_text())
    rows = _extract_class_i_ii(cfg, ROOT) + _extract_class_iii_iv(cfg, ROOT)
    assert all(r['representative_size_used'] == 64 and r['measurement_recomputed'] for r in rows)
    c2 = next(r for r in rows if r['class_name'] == 'Class-II')
    assert c2['ce_boundary_left_censored'] and c2['mobj_boundary_left_censored']
    assert all(r['canonical_class_label'] == r['class_name'] for r in rows)
    # Fresh tau1 reference at size 64; the separately recorded candidate maximum
    # happens to select tau1 for this particular canonical configuration.
    c4 = next(r for r in rows if r['class_name'] == 'Class-IV')
    assert c4['affinity_ref'] == pytest.approx(.0029871590763126815)
    assert c4['affinity_reference_tau'] == 1
    assert summarize_observable_separation(rows)['observable_map_separable']
    for change in ({'affinity_ref': -.1}, {'affinity_ref': float('nan')}, {'structural_boundary_lambda_ref': None}):
        bad = copy.deepcopy(rows)
        bad[1].update(change)
        assert not summarize_observable_separation(bad)['phase_map_figure_publishable']
    assert not summarize_observable_separation(rows + [rows[0]])['observable_map_separable']


def test_onset_is_an_interpolated_proxy():
    rows = [{'closure_strength_lambda': 0., 'affinity': 0.}, {'closure_strength_lambda': 1., 'affinity': .02}]
    assert _affinity_onset_from_rows(rows, [.01])['0.01'] == .5


def test_lens_suite_does_not_confirm_an_unverified_manual_reference():
    rows = [{'class_name': 'Class-III', 'representative_name': 'same', 'manual_reference_label': 'Class-III',
             'lens_name': lens, 'size': n, 'p5_state': 'active', 'p6_drive_state': 'inactive',
             'canonical_class_label': 'Class-III', 'p4_class_active': True}
            for lens in ('manual_family_coarse_lens', 'spectral_sign_pattern_lens', 'diffusion_quantile_lens') for n in (16, 32)]
    assert summarize_lens_suite(rows)['reviewer_facing_lens_robust_for_paper']
    rows[1]['canonical_class_label'] = 'Class-I'
    rows[1]['p4_class_active'] = False
    assert not summarize_lens_suite(rows)['manual_reference_verified']
    assert not summarize_lens_suite(rows)['reviewer_facing_lens_robust_for_paper']
    assert not summarize_lens_suite(rows[:1], expected_sizes=[16, 32])['manual_reference_verified']


def test_seed_bootstrap_preserves_complete_substrate_trajectories():
    rows = [{'size': 16, 'seed': seed, 'closure_strength_lambda': lam, 'order_value': value}
            for seed, values in [(1, [.1, .9]), (2, [.2, .8]), (3, [.4, .6])]
            for lam, value in zip([.25, .75], values)]
    for n in range(10):
        boot, unit = _resample_observations(rows, np.random.default_rng(n), 'size', 'closure_strength_lambda', 'order_value')
        assert sum(r['order_value'] for r in boot) == pytest.approx(3.)
        assert unit == 'seed_trajectory_within_size'
    with pytest.raises(ValueError):
        _resample_observations(rows[:-1], np.random.default_rng(0), 'size', 'closure_strength_lambda', 'order_value')


def test_no_finite_curve_comparison_is_not_a_fit():
    rows = [{'size': 16, 'closure_strength_lambda': l, 'order_mean': .5} for l in [.2, .8]]
    with pytest.raises(ValueError, match='no comparable'):
        grid_search_collapse_fit(rows, [.5], [.1], [1.])


def test_upstream_features_are_validated_and_sector_labels_are_surjective():
    sub = SimpleNamespace(rule_cp_applicability=np.ones((2, 3)), cell_cp_legality=np.ones((2, 3)),
                          cp_class_onehot=np.eye(3), cp_branch_preference_hint=np.array([[1., 0.], [.5, .5], [0., 1.]]),
                          cp_sector_index=np.array([3, 9, 9]))
    p, lens = _kernel_from_upstream_substrate(sub, .1)
    assert np.allclose(p.sum(axis=1), 1.)
    assert set(lens) == {0, 1}
    with pytest.raises(ValueError):
        _kernel_from_upstream_substrate(sub, 1.01)
    contradicted = classify_upstream_bridge_result({'canonical_class_label': 'Class-IV', 'p5_state': 'active', 'p6_drive_state': 'inactive', 'p4_class_active': False}, {})
    assert contradicted['canonical_class_label'] == 'unclear'


def test_missing_mapped_asset_cannot_pass_claim_or_readiness_gate():
    asset = {'asset_name': 'present', 'asset_tier': 'critical', 'bundle_root': '/example', 'reproducible': True,
             'required_files_present': True, 'rerun_success': True, 'stable_key_outputs': True}
    claims, _ = _claim_coverage([{'claim_id': 'claim', 'asset_names': ['present', 'missing']}], {'present': asset})
    assert not claims['claim']['all_supporting_assets_reproducible']
    assert not format_ready_for_writing_status([asset], claims)['ready_for_writing']
    assert not format_ready_for_writing_status([asset], {})['ready_for_writing']


def test_exact_undriven_portal_type_reduction_transports_packaging_metrics():
    from fractions import Fraction as F
    from layerbirth.exact_audit import canonical_kernel, controlled_kernel, structural_metrics
    cfg = {'fast_weight': F(1), 'fast_portal_weight': F(1, 50), 'portal_cross_weight': F(6, 5), 'portal_self_weight': F(4)}
    g, c = cfg['fast_portal_weight'], cfg['portal_cross_weight']
    degree = max(2 * cfg['fast_weight'] + g, 3 * g + c + cfg['portal_self_weight'])
    expected = [[1-g/degree, g/degree, F(0), F(0)],
                [3*g/degree, 1-(3*g+c)/degree, F(0), c/degree],
                [F(0), F(0), 1-g/degree, g/degree],
                [F(0), c/degree, 3*g/degree, 1-(3*g+c)/degree]]
    reference = {}
    for n in (16, 24, 40):
        p = canonical_kernel('replicated_portal_reversible_family', n, cfg)
        types = [2 * int(i >= n//2) + int(i % 4 == 3) for i in range(n)]
        for i, row in enumerate(p):
            assert [sum(row[j] for j in range(n) if types[j] == t) for t in range(4)] == expected[types[i]]
        for lam in (F(0), F(1, 5), F(4, 5), F(1)):
            for tau in (1, 2, 3):
                metrics = structural_metrics(controlled_kernel(p, lam), tau)
                if n == 16:
                    reference[lam, tau] = metrics
                else:
                    assert metrics == reference[lam, tau]


def test_alignment_unknown_measurements_do_not_certify_invariance_or_zero_affinity():
    from layerbirth.alignment import summarize_boundary_alignment, compare_tau_and_lens_dependence
    summary = summarize_boundary_alignment({}, {})
    assert not summary['baseline_structural_boundary_bias_invariant']
    verdict = compare_tau_and_lens_dependence({**summary, 'structural_rows': {}, 'affinity_rows': {}}, [])
    assert not verdict['baseline_claim_lens_robust']
    assert not verdict['affinity_secondary_activation_supported']
    structural = {f'16|{b}|manual_tau1': {'size':16, 'bias':b, 'analysis_mode':'manual_tau1',
                  'ce_crossings': {'0.025': None}, 'mobj_crossings': {'0.9': None}} for b in [0., .1]}
    summary = summarize_boundary_alignment(structural, {})
    assert not summary['baseline_structural_boundary_bias_invariant']
    assert np.isnan(summary['ce_boundary_ranges_manual_tau1'][16])


def test_linearity_cannot_be_certified_from_a_singleton_or_small_nonlinear_signal():
    from layerbirth.affinity_visibility import fit_lambda_linearity
    single = fit_lambda_linearity([{'closure_strength_lambda': .2, 'value': .3}], 'value')
    assert np.isnan(single['r_squared'])
    x = [0., .5, 1.]
    large = fit_lambda_linearity([{'closure_strength_lambda':a, 'value':v} for a,v in zip(x,[0.,1.,0.])], 'value')
    small = fit_lambda_linearity([{'closure_strength_lambda':a, 'value':v*1e-100} for a,v in zip(x,[0.,1.,0.])], 'value')
    assert large['r_squared'] == small['r_squared']
    assert small['r_squared'] < .999


def test_p4_rule_audit_requires_its_negative_controls():
    from layerbirth.class3 import evaluate_p4_criteria_on_profiles
    cfg = json.loads((ROOT / 'configs/pilots/class_iii_refinement_and_p4_audit.json').read_text())
    positives = {key: {'staging_shift_ce':.2, 'staging_shift_mobj':.4, 'presence_shift_any':False}
                 for key in ('replicated_portal_best_128','replicated_portal_best_256')}
    audit = evaluate_p4_criteria_on_profiles(positives, cfg['criteria'])
    assert audit['recommended_p4_criterion'] == 'none'
    assert not any(r['criterion_supported'] for r in audit['criteria'].values())


def test_scientific_json_preserves_unknown_and_extended_infinite_affinity():
    from layerbirth.serialization import observed_float, scientific_dumps
    from layerbirth.taxonomy import evaluate_p6_drive_activation, evaluate_p4_anomaly
    payload = json.loads(scientific_dumps({'missing':np.nan, 'one_way':np.inf, 'sample':np.array([1., np.nan])}),
                         parse_constant=lambda value: pytest.fail(f'nonstandard JSON token {value}'))
    assert payload == {'missing':None, 'one_way':'Infinity', 'sample':[1.,None]}
    assert np.isnan(observed_float(payload['missing']))
    thresholds = {'p6_active_min':.001, 'p6_inactive_max':.000001}
    assert evaluate_p6_drive_activation(payload['missing'], thresholds)['state'] == 'unknown'
    assert evaluate_p6_drive_activation(observed_float(payload['one_way']), thresholds)['state'] == 'active'
    assert evaluate_p4_anomaly({}, {'p4_like_shift_min':.05, 'p4_class_dual_shift_min':.15})['state'] == 'unknown'
    from layerbirth.p4 import compute_boundary_shift_summary, evaluate_p4_profile
    shift = compute_boundary_shift_summary({}, {'ce_boundary_lambda':.5, 'mobj_boundary_lambda':.7})
    assert evaluate_p4_profile(shift, {'p4_like_shift_min':.05, 'p4_class_dual_shift_min':.15})['p4_state'] == 'unknown'


def test_shadow_duplicate_points_and_missing_replicates_cannot_certify_coverage():
    from layerbirth.phenomenology import summarize_shadow_panel, filter_admissible_shadow_ensemble, build_shadow_ensemble_candidates
    admissibility = {'class_name':'Class-III', 'representative_name':'rep', 'admissible_count':6}
    rows = [{'size':n, 'closure_strength_lambda':.1, 'n_admissible_replicates':6,
             'susceptibility_delta_mobj':.2, 'binder_delta_mobj':.2, 'delta_mobj_mean':v}
            for n in (16,32) for v in (0.,.1,.2)]
    summary = summarize_shadow_panel(rows, admissibility, [.1,.2,.3], 6)
    assert not summary['usable_chi_u4_panel']
    assert not summary['observations_valid']
    with pytest.raises(ValueError, match='size panel'):
        filter_admissible_shadow_ensemble({}, [], [], [.1], '', {}, {}, 6)
    rep = {'representative_name':'rep', 'base_kwargs':{'x':1.}, 'drive_kwargs':{}}
    candidates = build_shadow_ensemble_candidates(rep, {'factors':[1.,1.001,1.002], 'structural_keys':['x']})
    assert len({r['candidate_id'] for r in candidates}) == 3


def test_capacity_table_availability_does_not_establish_scientific_support(tmp_path):
    from layerbirth.capacity_dashboard import summarize_capacity_postcritical_story
    (tmp_path/'analysis').mkdir()
    (tmp_path/'analysis'/'axis_effect_summary.json').write_text(json.dumps({'final_verdict':'capacity_orthogonal_or_mixed'}))
    (tmp_path/'analysis'/'final_verdict.json').write_text('{}')
    rows = [{'class_name':c} for c in ('Class-I','Class-II','Class-III','Class-IV')]
    assert not summarize_capacity_postcritical_story(tmp_path, rows)['capacity_postcritical_story_supported']


def test_reproducible_false_conclusion_cannot_pass_freeze_readiness(tmp_path):
    from layerbirth.freeze import check_scientific_evidence
    path = tmp_path/'summary.json'
    path.write_text('{"supported":false}')
    spec = {'evidence_checks':[{'path':'summary.json','field':'supported','equals':True}]}
    evidence = check_scientific_evidence(spec, tmp_path)
    assert not evidence['scientific_evidence_checks_passed']
    asset = {'asset_name':'a', 'asset_tier':'critical', 'reproducible':True,
             'required_files_present':True, 'rerun_success':True, 'stable_key_outputs':True, **evidence}
    assert not format_ready_for_writing_status([asset], {'c':{'supporting_asset_names':['a']}})['ready_for_writing']


def test_missing_equilibrium_affinity_cannot_certify_driven_equilibrium_separation(tmp_path, monkeypatch):
    from layerbirth import campaigns
    monkeypatch.setattr(campaigns, '_repo_root', lambda: tmp_path)
    best = {'lambda_c':.5, 'beta_over_nu':.1, 'inv_nu':1., 'objective':.1}
    driven = [{'closure_strength_lambda':.5, 'affinity_mean':.03}]
    summary = campaigns.compare_class_ii_to_class_i({'best_fit':best}, {'best_fit':best}, driven, [])
    assert np.isnan(summary['class_i_affinity_window_mean'])
    assert np.isnan(summary['class_ii_shadow_affinity_window_mean'])
    assert not summary['class_separation_plausible']


def test_incomplete_arrow_controls_block_the_consolidated_claim(tmp_path, monkeypatch):
    from layerbirth import dashboard
    cfg = json.loads((ROOT/'configs/dashboards/class_i_class_ii_consolidation.json').read_text())
    cfg['findings_note_path'] = str(tmp_path/'note.md')
    monkeypatch.setattr(dashboard, 'summarize_no_fake_arrow', lambda path: {
        'reversible_false_positive_count':0, 'protocol_trap_phase_aware_false_positive_count':0,
        'protocol_trap_hidden_schedule_driven_positive_count':1, 'no_fake_arrow_checks_passed':False})
    out = dashboard.build_class_consolidation_dashboard(cfg, output_root=tmp_path)
    summary = json.loads((Path(out['artifact_root'])/'analysis/consolidation_summary.json').read_text())
    assert not summary['publishable_non_exponent_claim_supported']
