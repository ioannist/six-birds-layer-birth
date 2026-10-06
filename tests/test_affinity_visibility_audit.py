import json
from pathlib import Path

import numpy as np

from layerbirth.affinity_visibility import (
    compute_slope_scores,
    compute_window_scores,
    estimate_level_crossings,
    fit_lambda_linearity,
    run_affinity_visibility_audit,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "campaigns" / "affinity_visibility_audit.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "affinity_visibility_audit"
    assert "panel" in cfg


def test_slope_score_plumbing():
    coarse = [
        {"closure_strength_lambda": 0.55, "objecthood_order": 0.1, "closure_error": 0.2},
        {"closure_strength_lambda": 0.75, "objecthood_order": 0.3, "closure_error": 0.4},
        {"closure_strength_lambda": 1.0, "objecthood_order": 0.55, "closure_error": 0.65},
    ]
    dense = []
    for l in np.linspace(0.55, 1.0, 10):
        dense.append(
            {
                "closure_strength_lambda": float(l),
                "objecthood_order": float(0.1 + (l - 0.55)),
                "closure_error": float(0.2 + (l - 0.55)),
            }
        )
    cw = compute_window_scores(coarse, [0.55, 0.75, 1.0])["max_window_score"]
    dw = compute_window_scores(dense, [float(x) for x in np.linspace(0.55, 1.0, 10)])["max_window_score"]
    cs = compute_slope_scores(coarse, [0.55, 0.75, 1.0])["max_slope_score"]
    ds = compute_slope_scores(dense, [float(x) for x in np.linspace(0.55, 1.0, 10)])["max_slope_score"]
    assert not np.isclose(cw, dw)
    assert np.isclose(cs, ds, rtol=1e-10, atol=1e-10)


def test_linearity_summary_sanity():
    rows = []
    for l in [0.55, 0.65, 0.75, 0.85, 0.95]:
        rows.append({"closure_strength_lambda": l, "closure_error": 2.0 * l + 1.0})
    fit = fit_lambda_linearity(rows, "closure_error")
    assert np.isclose(fit["r_squared"], 1.0, atol=1e-12)
    slopes = np.asarray([fit["slope"], fit["slope"], fit["slope"]], dtype=np.float64)
    cv = float(np.std(slopes) / max(abs(float(np.mean(slopes))), 1e-15))
    assert np.isclose(cv, 0.0, atol=1e-12)


def test_level_crossing_interpolation():
    rows = [
        {"closure_strength_lambda": 0.5, "objecthood_order": 0.8},
        {"closure_strength_lambda": 0.6, "objecthood_order": 0.9},
        {"closure_strength_lambda": 0.7, "objecthood_order": 1.0},
    ]
    crossings = estimate_level_crossings(rows, "objecthood_order", [0.85, 0.95])
    assert np.isclose(crossings["0.85"], 0.55, atol=1e-12)
    assert np.isclose(crossings["0.95"], 0.65, atol=1e-12)


def test_reduced_execution(tmp_path: Path):
    cfg = _load_cfg()
    cfg["panel"]["sizes"] = [8]
    cfg["panel"]["bias_grid"] = [0.0, 0.2]
    cfg["panel"]["lambda_grids"]["coarse"] = [0.75, 1.0]
    cfg["panel"]["lambda_grids"]["dense"] = [0.75, 0.85, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "lb17_vis_note.md")
    out = run_affinity_visibility_audit(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "visibility_summary.json").exists()
    assert (root / "analysis" / "linearity_summary.json").exists()
    assert (root / "analysis" / "level_crossings.json").exists()
    assert (root / "plots" / "affinity_vs_bias.png").exists()
    payload = json.loads((root / "analysis" / "visibility_summary.json").read_text(encoding="utf-8"))
    assert "grid_resolution_binding" in payload["verdicts"]
    assert "projector_dominated_bias_invariance" in payload["verdicts"]
