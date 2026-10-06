import csv
import json
from pathlib import Path

from layerbirth.class3 import run_class_iii_full_campaign


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads(
        (_repo_root() / "configs" / "campaigns" / "class_iii_full_campaign.json").read_text(
            encoding="utf-8"
        )
    )


def test_config_exists_and_targets_canonical_family():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "class_iii_full_campaign"
    assert cfg["candidate"]["candidate_name"] == "replicated_portal_sp4"
    assert cfg["candidate"]["family_name"] == "replicated_portal_reversible_family"
    assert cfg["size_panel"] == [16, 32, 64, 128]


def test_driver_runs_and_bundle_exists(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-05_class_iii_full_campaign.md")
    out = run_class_iii_full_campaign(cfg, output_root=tmp_path, use_cache=False)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "campaign_summary.json").exists()
    assert (root / "analysis" / "per_size_diagnostics.csv").exists()
    assert (root / "plots" / "staging_shifts_by_size.png").exists()
    assert (root / "plots" / "affinity_ref_by_size.png").exists()
    assert (root / "plots" / "class_state_by_size.png").exists()


def test_campaign_summary_matches_intended_verdict(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-05_class_iii_full_campaign.md")
    out = run_class_iii_full_campaign(cfg, output_root=tmp_path, use_cache=False)
    summary = json.loads((Path(out["artifact_root"]) / "analysis" / "campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["structural_birth_present"] is True
    assert summary["affinity_absent"] is True
    assert summary["staging_anomaly_present"] is True
    assert summary["final_class_verdict"] == "Class-III"


def test_per_size_diagnostics_consistent(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-05_class_iii_full_campaign.md")
    out = run_class_iii_full_campaign(cfg, output_root=tmp_path, use_cache=False)
    rows = list(csv.DictReader((Path(out["artifact_root"]) / "analysis" / "per_size_diagnostics.csv").open()))
    assert rows
    for row in rows:
        assert row["candidate_p5_state"] == "active"
        assert row["candidate_p6_drive_state"] == "inactive"
        assert row["candidate_p4_state"] == "active"
        assert row["canonical_class_label"] == "Class-III"


def test_findings_note_exists():
    assert (_repo_root() / "notes" / "findings" / "S2-05_class_iii_full_campaign.md").exists()


def test_registry_update_exists():
    text = (_repo_root() / "configs" / "experiment_registry.yaml").read_text(encoding="utf-8")
    assert "class_iii_full_campaign" in text
    assert "\"class_iii_strict_confirmation\": {\n      \"status\": \"superseded\"" in text
