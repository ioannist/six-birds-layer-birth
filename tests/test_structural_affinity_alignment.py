import json
from pathlib import Path

from layerbirth.alignment import (
    estimate_affinity_onsets,
    estimate_structural_boundaries,
    run_structural_affinity_alignment,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "campaigns" / "structural_affinity_alignment.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "structural_affinity_alignment"
    assert "panel" in cfg


def test_crossing_and_onset_interpolation_plumbing():
    rows = [
        {"size": 8, "bias": 0.0, "analysis_mode": "manual_tau1", "closure_strength_lambda": 0.5, "closure_error": 0.08, "objecthood_order": 0.8, "affinity": 0.0},
        {"size": 8, "bias": 0.0, "analysis_mode": "manual_tau1", "closure_strength_lambda": 0.6, "closure_error": 0.04, "objecthood_order": 0.9, "affinity": 0.002},
        {"size": 8, "bias": 0.0, "analysis_mode": "manual_tau1", "closure_strength_lambda": 0.7, "closure_error": 0.02, "objecthood_order": 0.95, "affinity": 0.01},
    ]
    s = estimate_structural_boundaries(rows, [0.05, 0.025], [0.85, 0.9])
    a = estimate_affinity_onsets(rows, [1e-3, 1e-2])
    key = "8|0.0|manual_tau1"
    assert s[key]["ce_crossings"]["0.05"] is not None
    assert s[key]["mobj_crossings"]["0.9"] is not None
    assert a[key]["affinity_onsets"]["0.001"] is not None
    assert a[key]["affinity_onsets"]["0.01"] is not None


def test_reduced_execution_and_cache(tmp_path: Path):
    cfg = _load_cfg()
    cfg["panel"]["sizes"] = [8, 16]
    cfg["panel"]["bias_grid"] = [0.0, 0.2]
    cfg["panel"]["lambda_grid"] = [0.75, 1.0]
    cfg["plot_representative_size"] = 16
    cfg["findings_note_path"] = str(tmp_path / "lb17_align_note.md")
    out1 = run_structural_affinity_alignment(cfg, output_root=tmp_path, use_cache=True)
    out2 = run_structural_affinity_alignment(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out1["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "structural_boundaries.json").exists()
    assert (root / "analysis" / "affinity_onsets.json").exists()
    assert (root / "analysis" / "alignment_summary.json").exists()
    assert (root / "plots" / "manual_vs_diffusion_comparison.png").exists()
    payload = json.loads((root / "analysis" / "alignment_summary.json").read_text(encoding="utf-8"))
    assert "baseline_structural_boundary_bias_invariant" in payload["verdicts"]
    assert "affinity_secondary_activation_supported" in payload["verdicts"]
    assert "tau2_reveals_structural_bias_coupling" in payload["verdicts"]
    assert "baseline_claim_lens_robust" in payload["verdicts"]
    assert "final_framing_verdict" in payload["verdicts"]
    assert out2["cached_count"] >= out1["cached_count"]
