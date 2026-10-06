import csv
import json
from pathlib import Path

from layerbirth.class3 import run_class_iv_full_campaign


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads(
        (_repo_root() / "configs" / "campaigns" / "class_iv_full_campaign.json").read_text(
            encoding="utf-8"
        )
    )


def test_config_exists_and_targets_selected_best_family():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "class_iv_full_campaign"
    assert cfg["candidate"]["candidate_name"] == "replicated_portal_sp4_drive_f"
    assert cfg["size_panel"] == [16, 32, 64, 128]


def test_driver_runs_and_bundle_exists(tmp_path: Path):
    cfg = _load_cfg()
    cfg["size_panel"] = [16, 32]
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-07_class_iv_full_campaign.md")
    out = run_class_iv_full_campaign(cfg, output_root=tmp_path, use_cache=False)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "campaign_summary.json").exists()
    assert (root / "analysis" / "per_size_diagnostics.csv").exists()
    assert (root / "analysis" / "comparison_to_class_ii.json").exists()
    assert (root / "analysis" / "comparison_to_class_iii.json").exists()
    assert (root / "analysis" / "class_comparison_table.csv").exists()
    assert (root / "plots" / "staging_shifts_by_size.png").exists()
    assert (root / "plots" / "affinity_ref_by_size.png").exists()
    assert (root / "plots" / "class_state_by_size.png").exists()
    assert (root / "plots" / "class_ii_iii_iv_comparison.png").exists()


def test_campaign_summary_explicit(tmp_path: Path):
    cfg = _load_cfg()
    cfg["size_panel"] = [16, 32]
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-07_class_iv_full_campaign.md")
    out = run_class_iv_full_campaign(cfg, output_root=tmp_path, use_cache=False)
    summary = json.loads((Path(out["artifact_root"]) / "analysis" / "campaign_summary.json").read_text(encoding="utf-8"))
    assert "structural_birth_present" in summary
    assert "affinity_present" in summary
    assert "staging_anomaly_present" in summary
    assert "final_class_verdict" in summary


def test_per_size_diagnostics_consistent(tmp_path: Path):
    cfg = _load_cfg()
    cfg["size_panel"] = [16, 32]
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-07_class_iv_full_campaign.md")
    out = run_class_iv_full_campaign(cfg, output_root=tmp_path, use_cache=False)
    rows = list(csv.DictReader((Path(out["artifact_root"]) / "analysis" / "per_size_diagnostics.csv").open()))
    assert rows
    for row in rows:
        assert row["candidate_p5_state"]
        assert row["candidate_p6_drive_state"]
        assert row["candidate_p4_state"]
        assert row["canonical_class_label"]
    summary = json.loads((Path(out["artifact_root"]) / "analysis" / "campaign_summary.json").read_text(encoding="utf-8"))
    if summary["final_campaign_verdict"] == "positive_class_iv_campaign":
        assert all(row["canonical_class_label"] == "Class-IV" for row in rows)
    else:
        assert summary["final_campaign_verdict"] == "explicit_failure_narrowing_report"


def test_comparison_artifacts_explicit(tmp_path: Path):
    cfg = _load_cfg()
    cfg["size_panel"] = [16, 32]
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-07_class_iv_full_campaign.md")
    out = run_class_iv_full_campaign(cfg, output_root=tmp_path, use_cache=False)
    root = Path(out["artifact_root"])
    c2 = json.loads((root / "analysis" / "comparison_to_class_ii.json").read_text(encoding="utf-8"))
    c3 = json.loads((root / "analysis" / "comparison_to_class_iii.json").read_text(encoding="utf-8"))
    assert "comparison_basis" in c2
    assert "comparison_basis" in c3
    table_rows = list(csv.DictReader((root / "analysis" / "class_comparison_table.csv").open()))
    assert any(r["class_name"] == "Class-II" for r in table_rows)
    assert any(r["class_name"] == "Class-III" for r in table_rows)
    assert any(r["class_name"] == "Class-IV" for r in table_rows)


def test_findings_note_exists():
    assert (_repo_root() / "notes" / "findings" / "S2-07_class_iv_full_campaign.md").exists()


def test_registry_update_exists():
    text = (_repo_root() / "configs" / "experiment_registry.yaml").read_text(encoding="utf-8")
    assert "class_iv_full_campaign" in text


def test_success_or_failure_explicit(tmp_path: Path):
    cfg = _load_cfg()
    cfg["size_panel"] = [16, 32]
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-07_class_iv_full_campaign.md")
    out = run_class_iv_full_campaign(cfg, output_root=tmp_path, use_cache=False)
    summary = json.loads((Path(out["artifact_root"]) / "analysis" / "campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["final_campaign_verdict"] in {"positive_class_iv_campaign", "explicit_failure_narrowing_report"}
