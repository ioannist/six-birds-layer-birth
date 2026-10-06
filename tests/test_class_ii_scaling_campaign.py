import json
from pathlib import Path

from layerbirth.campaigns import run_class_ii_scaling_campaign


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "campaigns" / "class_ii_driven_scaling.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_class_ii_config_loads():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "class_ii_driven"
    assert "primary_panel" in cfg
    assert "shadow_panel" in cfg


def test_reduced_class_ii_execution(tmp_path: Path):
    cfg = _load_cfg()
    cfg["primary_panel"]["sizes"] = [8, 16]
    cfg["primary_panel"]["lambda_grid"] = [0.5, 0.75]
    cfg["shadow_panel"]["sizes"] = [8, 16]
    cfg["shadow_panel"]["lambda_grid"] = [0.5, 0.75]
    cfg["fit_grid"]["lambda_c_grid"] = [0.5, 0.75]
    cfg["fit_grid"]["beta_over_nu_grid"] = [0.0, 0.1]
    cfg["fit_grid"]["inv_nu_grid"] = [0.5, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "lb16_note.md")
    out = run_class_ii_scaling_campaign(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "metrics" / "metrics.csv").exists()
    assert (root / "analysis" / "primary_group_summary.csv").exists()
    assert (root / "analysis" / "shadow_group_summary.csv").exists()
    assert (root / "analysis" / "fit_summary.json").exists()
    assert (root / "analysis" / "comparison_to_class_i.json").exists()
    assert (root / "plots").exists()
    assert out["final_verdict"] in {"keep pursuing", "not yet strong enough"}


def test_comparison_plumbing(tmp_path: Path):
    cfg = _load_cfg()
    cfg["primary_panel"]["sizes"] = [8, 16]
    cfg["primary_panel"]["lambda_grid"] = [0.5, 0.75]
    cfg["shadow_panel"]["sizes"] = [8, 16]
    cfg["shadow_panel"]["lambda_grid"] = [0.5, 0.75]
    cfg["fit_grid"]["lambda_c_grid"] = [0.5, 0.75]
    cfg["fit_grid"]["beta_over_nu_grid"] = [0.0, 0.1]
    cfg["fit_grid"]["inv_nu_grid"] = [0.5, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "lb16_note2.md")
    out = run_class_ii_scaling_campaign(cfg, output_root=tmp_path, use_cache=True)
    cmp_path = Path(out["artifact_root"]) / "analysis" / "comparison_to_class_i.json"
    cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
    required = [
        "class_i_lambda_c_fit",
        "class_i_beta_over_nu_fit",
        "class_i_inv_nu_fit",
        "class_ii_lambda_c_fit",
        "class_ii_beta_over_nu_fit",
        "class_ii_inv_nu_fit",
        "class_i_affinity_window_mean",
        "class_ii_affinity_window_mean",
        "class_ii_shadow_affinity_window_mean",
        "class_separation_plausible",
    ]
    for key in required:
        assert key in cmp
    assert isinstance(cmp["class_separation_plausible"], bool)


def test_fit_or_failure_and_cache(tmp_path: Path):
    cfg = _load_cfg()
    cfg["primary_panel"]["sizes"] = [8, 16]
    cfg["primary_panel"]["lambda_grid"] = [0.5, 0.75]
    cfg["shadow_panel"]["sizes"] = [8, 16]
    cfg["shadow_panel"]["lambda_grid"] = [0.5, 0.75]
    cfg["fit_grid"]["lambda_c_grid"] = [0.5, 0.75]
    cfg["fit_grid"]["beta_over_nu_grid"] = [0.0, 0.1]
    cfg["fit_grid"]["inv_nu_grid"] = [0.5, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "lb16_note3.md")
    first = run_class_ii_scaling_campaign(cfg, output_root=tmp_path, use_cache=True)
    second = run_class_ii_scaling_campaign(cfg, output_root=tmp_path, use_cache=True)
    root = Path(first["artifact_root"])
    assert (root / "analysis" / "fit_summary.json").exists()
    if first["failure_report_needed"]:
        fr = root / "analysis" / "failure_report.json"
        assert fr.exists()
        payload = json.loads(fr.read_text(encoding="utf-8"))
        for key in ("best_fit_parameters", "collapse_objective", "affinity_levels", "diagnosis"):
            assert key in payload
    assert second["cached_count"] > 0
