import json
from pathlib import Path

from layerbirth.identifiability import (
    compare_class_ii_observables,
    run_class_ii_identifiability_audit,
    summarize_landscape_sharpness,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "campaigns" / "class_ii_identifiability_audit.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "class_ii_identifiability_audit"
    assert "fit_grid" in cfg


def test_landscape_summary_plumbing():
    grid = {
        "lambda_c_grid": [0.8, 0.85],
        "beta_over_nu_grid": [0.0, 0.05],
        "inv_nu_grid": [0.5, 1.0],
    }
    landscape = {
        "ranked_points": [
            {"lambda_c": 0.8, "beta_over_nu": 0.0, "inv_nu": 0.5, "objective": 1.0},
            {"lambda_c": 0.85, "beta_over_nu": 0.0, "inv_nu": 0.5, "objective": 1.05},
            {"lambda_c": 0.8, "beta_over_nu": 0.05, "inv_nu": 0.5, "objective": 1.06},
        ]
    }
    sharp = summarize_landscape_sharpness(landscape, grid, tolerance_fraction=0.10)
    assert sharp["best_is_boundary"] is True
    assert sharp["plateau_count_10pct"] >= 2
    assert sharp["fit_underdetermined"] is True


def test_better_observable_rule_plumbing():
    cmp = compare_class_ii_observables(
        {
            "class_ii_mobj_tau1": {
                "collapse_objective": 1.0,
                "local_contrast_ratio": 1.2,
                "fit_underdetermined": True,
            },
            "class_ii_aff_tau1": {
                "collapse_objective": 0.5,
                "local_contrast_ratio": 1.0,
                "fit_underdetermined": False,
            },
            "class_ii_mobj_tau2": {
                "collapse_objective": 0.9,
                "local_contrast_ratio": 1.1,
                "fit_underdetermined": True,
            },
            "class_ii_aff_tau2": {
                "collapse_objective": 0.3,
                "local_contrast_ratio": 0.95,
                "fit_underdetermined": False,
            },
        }
    )
    assert cmp["affinity_better_than_mobj_for_class_ii"] is True
    assert cmp["tau2_improves_class_ii"] is True


def test_reduced_execution(tmp_path: Path):
    cfg = _load_cfg()
    cfg["auxiliary_tau2_panel"]["sizes"] = [8, 16]
    cfg["auxiliary_tau2_panel"]["lambda_grid"] = [0.75, 1.0]
    cfg["fit_grid"]["lambda_c_grid"] = [0.75, 0.8, 0.85]
    cfg["fit_grid"]["beta_over_nu_grid"] = [0.0, 0.1]
    cfg["fit_grid"]["inv_nu_grid"] = [0.5, 1.0]
    cfg["fit_grid"]["bootstrap_replicates"] = 5
    cfg["findings_note_path"] = str(tmp_path / "lb17_ident_note.md")
    out1 = run_class_ii_identifiability_audit(cfg, output_root=tmp_path, use_cache=True)
    out2 = run_class_ii_identifiability_audit(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out1["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "identifiability_summary.json").exists()
    assert (root / "analysis" / "class_ii_mobj_tau1_landscape.json").exists()
    assert (root / "analysis" / "class_ii_aff_tau1_landscape.json").exists()
    assert (root / "plots").exists()
    payload = json.loads((root / "analysis" / "identifiability_summary.json").read_text(encoding="utf-8"))
    assert "class_ii_mobj_tau1_underdetermined" in payload["verdicts"]
    assert "tau2_improves_class_ii" in payload["verdicts"]
    assert "provisional_framing_recommendation" in payload["verdicts"]
    assert out2["cached_count"] >= out1["cached_count"]
