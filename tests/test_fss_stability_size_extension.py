import json
from pathlib import Path

from layerbirth.fss_stability import (
    compare_grid_sensitivity,
    compare_size_extension,
    run_fss_stability_audit,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "campaigns" / "fss_stability_size_extension.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "fss_stability_size_extension"
    assert "fit_grid_suite" in cfg


def test_grid_sensitivity_rule():
    suite = {
        "original_grid": {
            "summary": {"best_fit": {"lambda_c": 0.8, "beta_over_nu": 0.1, "inv_nu": 1.0}, "best_is_boundary": False},
            "fit_grid": {"lambda_c_grid": [0.7, 0.8, 0.9], "beta_over_nu_grid": [0.0, 0.1, 0.2], "inv_nu_grid": [0.5, 1.0, 1.5]},
        },
        "ident_grid": {
            "summary": {"best_fit": {"lambda_c": 0.8, "beta_over_nu": 0.1, "inv_nu": 1.0}, "best_is_boundary": False},
            "fit_grid": {},
        },
        "union_grid": {
            "summary": {"best_fit": {"lambda_c": 0.8, "beta_over_nu": 0.1, "inv_nu": 1.0}, "best_is_boundary": False},
            "fit_grid": {},
        },
    }
    assert compare_grid_sensitivity(suite)["grid_spec_sensitive"] is False
    suite["ident_grid"]["summary"]["best_fit"]["lambda_c"] = 0.95
    assert compare_grid_sensitivity(suite)["grid_spec_sensitive"] is True


def test_size_extension_helpful_rule():
    old = {"summary": {"plateau_count_10pct": 20, "inv_nu_span_10pct": 1.0, "beta_span_10pct": 1.0, "best_is_boundary": True, "local_contrast_ratio": 1.5}}
    new = {"summary": {"plateau_count_10pct": 10, "inv_nu_span_10pct": 0.6, "beta_span_10pct": 0.6, "best_is_boundary": False, "local_contrast_ratio": 1.0}}
    assert compare_size_extension(old, new)["size128_helpful"] is True
    old2 = {"summary": {"plateau_count_10pct": 10, "inv_nu_span_10pct": 0.5, "beta_span_10pct": 0.5, "best_is_boundary": False, "local_contrast_ratio": 1.2}}
    new2 = {"summary": {"plateau_count_10pct": 11, "inv_nu_span_10pct": 0.55, "beta_span_10pct": 0.55, "best_is_boundary": False, "local_contrast_ratio": 1.25}}
    assert compare_size_extension(old2, new2)["size128_helpful"] is False


def test_reduced_execution(tmp_path: Path):
    cfg = _load_cfg()
    cfg["class_i_auxiliary_size128"]["lambda_grid"] = [0.75, 1.0]
    cfg["class_ii_auxiliary_size128"]["lambda_grid"] = [0.75, 1.0]
    cfg["fit_grid_suite"]["class_i"]["original_grid"]["lambda_c_grid"] = [0.75, 0.85]
    cfg["fit_grid_suite"]["class_i"]["ident_grid"]["lambda_c_grid"] = [0.75, 0.85]
    cfg["fit_grid_suite"]["class_i"]["union_grid"]["lambda_c_grid"] = [0.75, 0.85]
    cfg["fit_grid_suite"]["class_ii"]["original_grid"]["lambda_c_grid"] = [0.75, 0.85]
    cfg["fit_grid_suite"]["class_ii"]["ident_grid"]["lambda_c_grid"] = [0.75, 0.85]
    cfg["fit_grid_suite"]["class_ii"]["union_grid"]["lambda_c_grid"] = [0.75, 0.85]
    cfg["fit_grid_suite"]["class_i"]["original_grid"]["beta_over_nu_grid"] = [0.0, 0.1]
    cfg["fit_grid_suite"]["class_i"]["ident_grid"]["beta_over_nu_grid"] = [0.0, 0.1]
    cfg["fit_grid_suite"]["class_i"]["union_grid"]["beta_over_nu_grid"] = [0.0, 0.1]
    cfg["fit_grid_suite"]["class_ii"]["original_grid"]["beta_over_nu_grid"] = [0.0, 0.1]
    cfg["fit_grid_suite"]["class_ii"]["ident_grid"]["beta_over_nu_grid"] = [0.0, 0.1]
    cfg["fit_grid_suite"]["class_ii"]["union_grid"]["beta_over_nu_grid"] = [0.0, 0.1]
    cfg["fit_grid_suite"]["class_i"]["original_grid"]["inv_nu_grid"] = [0.5, 1.0]
    cfg["fit_grid_suite"]["class_i"]["ident_grid"]["inv_nu_grid"] = [0.5, 1.0]
    cfg["fit_grid_suite"]["class_i"]["union_grid"]["inv_nu_grid"] = [0.5, 1.0]
    cfg["fit_grid_suite"]["class_ii"]["original_grid"]["inv_nu_grid"] = [0.5, 1.0]
    cfg["fit_grid_suite"]["class_ii"]["ident_grid"]["inv_nu_grid"] = [0.5, 1.0]
    cfg["fit_grid_suite"]["class_ii"]["union_grid"]["inv_nu_grid"] = [0.5, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "lb17_fss_note.md")
    out1 = run_fss_stability_audit(cfg, output_root=tmp_path, use_cache=True)
    out2 = run_fss_stability_audit(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out1["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "fss_stability_summary.json").exists()
    assert (root / "analysis" / "class_i_oldsizes_original_grid.json").exists()
    assert (root / "analysis" / "class_ii_plus128_union_grid.json").exists()
    assert (root / "plots").exists()
    summary = json.loads((root / "analysis" / "fss_stability_summary.json").read_text(encoding="utf-8"))
    assert "class_i_grid_spec_sensitive" in summary["verdicts"]
    assert "class_i_size128_helpful" in summary["verdicts"]
    assert "fss_exponent_claim_supported" in summary["verdicts"]
    assert "final_recommendation" in summary["verdicts"]
    assert out2["cached_count"] >= out1["cached_count"]
