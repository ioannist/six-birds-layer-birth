import csv
import json
from pathlib import Path

from layerbirth.class3 import run_four_class_synthesis_dashboard


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads(
        (_repo_root() / "configs" / "dashboards" / "four_class_synthesis.json").read_text(
            encoding="utf-8"
        )
    )


def test_config_points_to_canonical_references():
    cfg = _load_cfg()
    s = cfg["sources"]
    assert s["class_i_root"] == "results/campaigns/class_i_equilibrium"
    assert s["class_ii_root"] == "results/campaigns/class_ii_driven"
    assert s["class_i_ii_consolidation_root"] == "results/dashboards/class_i_class_ii_consolidation"
    assert s["class_iii_root"] == "results/campaigns/class_iii_full_campaign"
    assert s["class_iv_root"] == "results/campaigns/class_iv_full_campaign"


def test_driver_runs_and_bundle_exists(tmp_path: Path):
    cfg = _load_cfg()
    cfg["conventions"]["tau_depth_lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-08_four_class_synthesis_dashboard.md")
    out = run_four_class_synthesis_dashboard(cfg, output_root=tmp_path, use_cache=False)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "dashboard_summary.json").exists()
    assert (root / "analysis" / "class_activation_table.csv").exists()
    assert (root / "analysis" / "reference_map.json").exists()
    assert (root / "analysis" / "no_fake_arrow_summary.json").exists()
    assert (root / "analysis" / "boundary_proxy_table.csv").exists()
    assert (root / "analysis" / "tau_depth_sensitivity.json").exists()
    assert (root / "plots" / "p5_p6_p4_activation_matrix.png").exists()
    assert (root / "plots" / "no_fake_arrow_checks.png").exists()
    assert (root / "plots" / "boundary_proxy_comparison.png").exists()
    assert (root / "plots" / "tau_depth_sensitivity.png").exists()
    assert (root / "plots" / "four_class_overview.png").exists()


def test_activation_table_exact_patterns(tmp_path: Path):
    cfg = _load_cfg()
    cfg["conventions"]["tau_depth_lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-08_four_class_synthesis_dashboard.md")
    out = run_four_class_synthesis_dashboard(cfg, output_root=tmp_path, use_cache=False)
    rows = list(csv.DictReader((Path(out["artifact_root"]) / "analysis" / "class_activation_table.csv").open()))
    assert len(rows) == 4
    m = {r["class_label"]: r for r in rows}
    assert m["Class-I"]["candidate_p5_state"] == "active"
    assert m["Class-I"]["candidate_p6_drive_state"] == "inactive"
    assert m["Class-I"]["candidate_p4_state"] == "inactive"
    assert m["Class-II"]["candidate_p5_state"] == "active"
    assert m["Class-II"]["candidate_p6_drive_state"] == "active"
    assert m["Class-II"]["candidate_p4_state"] == "inactive"
    assert m["Class-III"]["candidate_p5_state"] == "active"
    assert m["Class-III"]["candidate_p6_drive_state"] == "inactive"
    assert m["Class-III"]["candidate_p4_state"] == "active"
    assert m["Class-IV"]["candidate_p5_state"] == "active"
    assert m["Class-IV"]["candidate_p6_drive_state"] == "active"
    assert m["Class-IV"]["candidate_p4_state"] == "active"


def test_programmatic_separability_explicit(tmp_path: Path):
    cfg = _load_cfg()
    cfg["conventions"]["tau_depth_lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-08_four_class_synthesis_dashboard.md")
    out = run_four_class_synthesis_dashboard(cfg, output_root=tmp_path, use_cache=False)
    d = json.loads((Path(out["artifact_root"]) / "analysis" / "dashboard_summary.json").read_text(encoding="utf-8"))
    assert d["unique_activation_signatures"] == 4
    assert d["all_four_classes_programmatically_separable"] is True


def test_no_fake_arrow_anchor(tmp_path: Path):
    cfg = _load_cfg()
    cfg["conventions"]["tau_depth_lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-08_four_class_synthesis_dashboard.md")
    out = run_four_class_synthesis_dashboard(cfg, output_root=tmp_path, use_cache=False)
    n = json.loads((Path(out["artifact_root"]) / "analysis" / "no_fake_arrow_summary.json").read_text(encoding="utf-8"))
    assert "class_i_class_ii_consolidation/analysis/no_fake_arrow_summary.json" in n["shared_control_source"]
    assert n["no_fake_arrow_checks_passed"] is True


def test_tau_depth_has_all_four_classes(tmp_path: Path):
    cfg = _load_cfg()
    cfg["conventions"]["tau_depth_lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-08_four_class_synthesis_dashboard.md")
    out = run_four_class_synthesis_dashboard(cfg, output_root=tmp_path, use_cache=False)
    t = json.loads((Path(out["artifact_root"]) / "analysis" / "tau_depth_sensitivity.json").read_text(encoding="utf-8"))
    assert "class_i_reference" in t
    assert "class_ii_reference" in t
    assert "class_iii_reference" in t
    assert "class_iv_reference" in t


def test_findings_note_and_registry():
    assert (_repo_root() / "notes" / "findings" / "S2-08_four_class_synthesis_dashboard.md").exists()
    text = (_repo_root() / "configs" / "experiment_registry.yaml").read_text(encoding="utf-8")
    assert "four_class_synthesis_dashboard" in text
