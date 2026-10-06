"""Evidence must change when its generating computation changes."""
import copy
import json
from pathlib import Path

import pytest

from layerbirth import provenance
from layerbirth.sweep import execute_sweep_run, expand_sweep_grid
from layerbirth.campaigns import build_class_i_panel_rows


def test_computation_identity_includes_implementation_and_all_inputs(monkeypatch):
    payload = {"weights": [1.0, 0.25], "tau": 1}
    original = provenance.computation_hash(payload)
    assert original == provenance.computation_hash(dict(reversed(list(payload.items()))))
    assert original != provenance.computation_hash({**payload, "tau": 2})
    monkeypatch.setattr(provenance, "implementation_fingerprint", lambda: "changed implementation")
    assert original != provenance.computation_hash(payload)
    with pytest.raises(ValueError):
        provenance.computation_hash({"bad": float("nan")})


def test_sweep_recomputes_changed_inputs_even_with_reused_external_run_id(tmp_path):
    cfg = json.loads(Path("configs/sweeps/tiny_grid.json").read_text())
    spec = expand_sweep_grid(cfg)[0]
    before = execute_sweep_run(spec, tmp_path)
    assert execute_sweep_run(spec, tmp_path)["cache_status"] == "cached"
    changed = copy.deepcopy(spec)
    changed["closure_strength_lambda"] = 1.0
    after = execute_sweep_run(changed, tmp_path)
    assert after["cache_status"] == "executed"
    assert float(before["closure_error"]) > 0
    assert float(after["closure_error"]) < 1e-12
    assert float(after["objecthood_order"]) > float(before["objecthood_order"])


def test_sweep_recomputes_unverifiable_or_old_implementation_cache(tmp_path):
    cfg = json.loads(Path("configs/sweeps/tiny_grid.json").read_text())
    spec = expand_sweep_grid(cfg)[0]
    execute_sweep_run(spec, tmp_path)
    manifest = tmp_path / spec["sweep_id"] / "runs" / spec["run_id"] / "manifest.json"
    content = json.loads(manifest.read_text())
    content["code_version"]["implementation_sha256"] = "old"
    manifest.write_text(json.dumps(content))
    assert execute_sweep_run(spec, tmp_path)["cache_status"] == "executed"
    manifest.write_text('{}')
    assert execute_sweep_run(spec, tmp_path)["cache_status"] == "executed"


def test_campaign_kernel_parameters_invalidate_row_cache(tmp_path):
    cfg = json.loads(Path('configs/campaigns/class_i_equilibrium_scaling.json').read_text())
    cfg['primary_panel']['sizes'] = [8]
    cfg['primary_panel']['lambda_grid'] = [0.5]
    cfg['shadow_panel']['sizes'] = []
    first, _, executed, cached = build_class_i_panel_rows(cfg, tmp_path, True)
    assert (executed, cached) == (1, 0)
    _, _, executed, cached = build_class_i_panel_rows(cfg, tmp_path, True)
    assert (executed, cached) == (0, 1)
    cfg['primary_panel']['inter_block_weight'] = 0.8
    second, _, executed, cached = build_class_i_panel_rows(cfg, tmp_path, True)
    assert (executed, cached) == (1, 0)
    assert first[0]['run_id'] != second[0]['run_id']
    assert first[0]['order_value'] != second[0]['order_value']


def test_prerequisite_snapshot_must_match_the_requested_size_panel(tmp_path):
    from layerbirth.taxonomy import run_canonical_class_rubric
    cfg = json.loads(Path('configs/taxonomy/canonical_class_rubric.json').read_text())
    cfg['findings_note_path'] = str(tmp_path/'note.md')
    output = run_canonical_class_rubric(cfg, output_root=tmp_path)
    bundle = Path(output['artifact_root'])
    assert provenance.artifact_is_current(bundle, cfg)
    changed = copy.deepcopy(cfg)
    changed['reference_profiles'][0]['size'] = 64
    assert not provenance.artifact_is_current(bundle, changed)


def test_freeze_forces_fresh_sweep_computation(tmp_path):
    import csv
    import shlex
    from layerbirth.freeze import verify_asset_reproducibility
    cfg = json.loads(Path('configs/sweeps/tiny_grid.json').read_text())
    spec = expand_sweep_grid(cfg)[0]
    first = execute_sweep_run(spec, tmp_path)
    bundle = tmp_path / spec['sweep_id'] / 'runs' / spec['run_id']
    script = tmp_path / 'rerun.py'
    script.write_text('from pathlib import Path\nfrom layerbirth.sweep import execute_sweep_run\n'
                      f'execute_sweep_run({spec!r}, Path({str(tmp_path)!r}), use_cache=True)\n')
    asset = {'asset_name': 'fresh_sweep', 'asset_tier': 'critical', 'bundle_root': str(bundle),
             'required_files': ['manifest.json', 'metrics/metrics.csv'], 'key_output_paths': ['metrics/metrics.csv'],
             'rerun_entrypoint': shlex.join([str(Path('.venv/bin/python').absolute()), str(script)])}
    frozen = verify_asset_reproducibility(asset, Path.cwd())
    assert frozen['fresh_computation_verified']
    assert frozen['reproducible']
    with (bundle / 'metrics/metrics.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert row['cache_status'] == 'executed'
    assert float(row['closure_error']) == float(first['closure_error'])
