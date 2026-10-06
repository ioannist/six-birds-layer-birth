import csv
import json
from pathlib import Path

from layerbirth.class3 import run_affinity_result_consolidation


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads(
        (_repo_root() / "configs" / "dashboards" / "affinity_result_consolidation.json").read_text(
            encoding="utf-8"
        )
    )


def test_config_points_to_canonical_sources():
    cfg = _load_cfg()
    s = cfg["sources"]
    assert "affinity_visibility_audit" in s["affinity_visibility_root"]
    assert "structural_affinity_alignment" in s["structural_affinity_alignment_root"]
    assert "crossover_decision_affinity_boundary" in s["crossover_decision_root"]
    assert "class_i_class_ii_consolidation" in s["class_i_ii_consolidation_root"]
    assert "class_iii_full_campaign" in s["class_iii_campaign_root"]
    assert "class_iv_full_campaign" in s["class_iv_campaign_root"]


def test_driver_runs_and_bundle_exists(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-09_affinity_result_consolidation.md")
    out = run_affinity_result_consolidation(cfg, output_root=tmp_path, use_cache=False)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "result_summary.json").exists()
    assert (root / "analysis" / "monotonic_bias_affinity_summary.json").exists()
    assert (root / "analysis" / "monotonic_bias_affinity_table.csv").exists()
    assert (root / "analysis" / "class_affinity_boundary_table.csv").exists()
    assert (root / "analysis" / "orthogonality_summary.json").exists()
    assert (root / "plots" / "affinity_vs_bias_monotonic.png").exists()
    assert (root / "plots" / "affinity_boundary_by_class.png").exists()
    assert (root / "plots" / "p5_affinity_orthogonality.png").exists()
    assert (root / "plots" / "affinity_secondary_order_parameter_overview.png").exists()


def test_monotonic_bias_result_explicit(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-09_affinity_result_consolidation.md")
    out = run_affinity_result_consolidation(cfg, output_root=tmp_path, use_cache=False)
    m = json.loads((Path(out["artifact_root"]) / "analysis" / "monotonic_bias_affinity_summary.json").read_text(encoding="utf-8"))
    assert m["affinity_monotonic_with_bias"] is True
    assert m["bias_grid"]
    seq = [float(x) for x in m["mean_affinity_by_bias"]]
    assert all(seq[i] <= seq[i + 1] + 1e-15 for i in range(len(seq) - 1))


def test_class_affinity_boundary_pattern(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-09_affinity_result_consolidation.md")
    out = run_affinity_result_consolidation(cfg, output_root=tmp_path, use_cache=False)
    rows = list(csv.DictReader((Path(out["artifact_root"]) / "analysis" / "class_affinity_boundary_table.csv").open()))
    assert len(rows) == 4
    m = {r["class_label"]: r for r in rows}
    assert abs(float(m["Class-I"]["affinity_measure_value"])) < 1e-6
    assert float(m["Class-II"]["affinity_measure_value"]) > 0.0
    assert abs(float(m["Class-III"]["affinity_measure_value"])) < 1e-6
    assert float(m["Class-IV"]["affinity_measure_value"]) > 0.0
    assert m["Class-I"]["candidate_p5_state"] == "active"
    assert m["Class-II"]["candidate_p6_drive_state"] == "active"
    assert m["Class-III"]["candidate_p4_state"] == "active"
    assert m["Class-IV"]["candidate_p6_drive_state"] == "active"


def test_orthogonality_summary_explicit(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-09_affinity_result_consolidation.md")
    out = run_affinity_result_consolidation(cfg, output_root=tmp_path, use_cache=False)
    o = json.loads((Path(out["artifact_root"]) / "analysis" / "orthogonality_summary.json").read_text(encoding="utf-8"))
    assert o["orthogonality_supported_under_baseline_coarse_analysis"] is True
    assert o["nonzero_bias_has_candidate_window"] is False
    assert o["birth_location_shift_condition"] is False
    assert o["tau2_reveals_structural_bias_coupling"] is True


def test_findings_note_and_registry():
    assert (_repo_root() / "notes" / "findings" / "S2-09_affinity_result_consolidation.md").exists()
    text = (_repo_root() / "configs" / "experiment_registry.yaml").read_text(encoding="utf-8")
    assert "affinity_result_consolidation" in text


def test_final_verdict_explicit(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-09_affinity_result_consolidation.md")
    out = run_affinity_result_consolidation(cfg, output_root=tmp_path, use_cache=False)
    s = json.loads((Path(out["artifact_root"]) / "analysis" / "result_summary.json").read_text(encoding="utf-8"))
    assert bool(s["final_result_verdict"])
