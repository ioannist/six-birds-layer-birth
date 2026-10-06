import csv
import json
from pathlib import Path

from layerbirth.class3 import run_class_iv_candidates


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads(
        (_repo_root() / "configs" / "pilots" / "class_iv_candidates.json").read_text(
            encoding="utf-8"
        )
    )


def test_config_exists_and_rooted_in_canonical_class_iii():
    cfg = _load_cfg()
    assert cfg["pilot_id"] == "class_iv_candidates"
    base = cfg["base_candidate"]
    assert base["candidate_name"] == "replicated_portal_sp4"
    assert base["family_name"] == "replicated_portal_reversible_family"
    assert float(base["base_kwargs"]["portal_self_weight"]) == 4.0
    assert 4 <= len(cfg["candidates"]) <= 8
    assert cfg["stage1_sizes"] == [32]
    assert cfg["stage2_sizes"] == [32, 64]


def test_driver_runs_and_bundle_exists(tmp_path: Path):
    cfg = _load_cfg()
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["candidates"] = cfg["candidates"][:4]
    cfg["findings_note_path"] = str(tmp_path / "S2-06_class_iv_candidates.md")
    out = run_class_iv_candidates(cfg, output_root=tmp_path, use_cache=False)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "pilot_summary.json").exists()
    assert (root / "analysis" / "candidate_table.csv").exists()
    assert (root / "analysis" / "best_candidate_summary.json").exists()
    assert (root / "plots" / "p4_p5_p6_states_by_candidate.png").exists()
    assert (root / "plots" / "affinity_vs_staging_by_candidate.png").exists()


def test_candidate_table_required_columns(tmp_path: Path):
    cfg = _load_cfg()
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["candidates"] = cfg["candidates"][:4]
    cfg["findings_note_path"] = str(tmp_path / "S2-06_class_iv_candidates.md")
    out = run_class_iv_candidates(cfg, output_root=tmp_path, use_cache=False)
    rows = list(csv.DictReader((Path(out["artifact_root"]) / "analysis" / "candidate_table.csv").open()))
    assert rows
    required = {"candidate_p5_state", "candidate_p6_drive_state", "candidate_p4_state", "all_three_active"}
    assert required.issubset(set(rows[0].keys()))


def test_best_candidate_summary_explicit(tmp_path: Path):
    cfg = _load_cfg()
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["candidates"] = cfg["candidates"][:4]
    cfg["findings_note_path"] = str(tmp_path / "S2-06_class_iv_candidates.md")
    out = run_class_iv_candidates(cfg, output_root=tmp_path, use_cache=False)
    summary = json.loads((Path(out["artifact_root"]) / "analysis" / "best_candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["best_candidate_name"]
    assert "any_candidate_all_three_active" in summary
    assert summary["final_pilot_verdict"] in {"positive_class_iv_pilot_candidate", "explicit_failure_near_miss"}


def test_findings_note_exists():
    assert (_repo_root() / "notes" / "findings" / "S2-06_class_iv_candidates.md").exists()


def test_registry_update_exists():
    text = (_repo_root() / "configs" / "experiment_registry.yaml").read_text(encoding="utf-8")
    assert "class_iv_candidates" in text


def test_success_or_failure_explicit(tmp_path: Path):
    cfg = _load_cfg()
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["candidates"] = cfg["candidates"][:4]
    cfg["findings_note_path"] = str(tmp_path / "S2-06_class_iv_candidates.md")
    out = run_class_iv_candidates(cfg, output_root=tmp_path, use_cache=False)
    pilot = json.loads((Path(out["artifact_root"]) / "analysis" / "pilot_summary.json").read_text(encoding="utf-8"))
    assert pilot["final_pilot_verdict"] in {"positive_class_iv_pilot_candidate", "explicit_failure_near_miss"}
