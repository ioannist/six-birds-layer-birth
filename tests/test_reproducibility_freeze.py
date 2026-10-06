from __future__ import annotations

import json
from pathlib import Path

from layerbirth.freeze import build_evidence_ledger, format_ready_for_writing_status, verify_asset_reproducibility


def test_config_loading() -> None:
    cfg_path = Path("configs/freeze/reproducibility_freeze.json")
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["freeze_id"] == "reproducibility_freeze"
    assert len(cfg["assets"]) >= 14


def test_presence_reproducibility_logic() -> None:
    rows = [
        {"asset_tier": "critical", "asset_name": "a", "required_files_present": True, "rerun_success": True, "stable_key_outputs": True, "reproducible": True},
        {"asset_tier": "critical", "asset_name": "b", "required_files_present": True, "rerun_success": False, "stable_key_outputs": True, "reproducible": False},
    ]
    claims = {"CLAIM-X": {"supporting_asset_names": ["a"]}}
    s = format_ready_for_writing_status(rows, claims, [])
    assert s["ready_for_writing"] is False
    assert s["blocking_issues_present"] is True


def test_claim_coverage_logic() -> None:
    rows = [{"asset_tier": "critical", "asset_name": "a", "required_files_present": True, "rerun_success": True, "stable_key_outputs": True, "reproducible": True}]
    claims = {
        "CLAIM-OK": {"supporting_asset_names": ["a"]},
        "CLAIM-MISS": {"supporting_asset_names": []},
    }
    s = format_ready_for_writing_status(rows, claims, [])
    assert s["ready_for_writing"] is False


def test_ready_for_writing_logic() -> None:
    rows = [
        {"asset_tier": "critical", "asset_name": "a", "required_files_present": True, "rerun_success": True, "stable_key_outputs": True, "reproducible": True}
    ]
    claims = {"CLAIM-X": {"supporting_asset_names": ["a"]}}
    s = format_ready_for_writing_status(rows, claims, ["nonblocking"])
    assert s["ready_for_writing"] is True
    assert s["nonblocking_caveats_present"] is True


def test_cheap_execution(tmp_path: Path) -> None:
    cfg = json.loads(Path("configs/freeze/reproducibility_freeze.json").read_text(encoding="utf-8"))
    cfg["artifact_subdir"] = "reproducibility_freeze_test"
    cfg["write_repo_note"] = False
    cfg["assets"] = cfg["assets"][:2]
    cfg["claims"] = [
        {
            "claim_id": "CLAIM-01",
            "description": "smoke",
            "asset_names": [cfg["assets"][0]["asset_name"]],
        }
    ]
    out = build_evidence_ledger(cfg, output_root=tmp_path, use_cache=False)
    root = Path(out["artifact_root"])

    assert root.exists()
    assert (root / "analysis" / "evidence_ledger.json").exists()
    assert (root / "analysis" / "reproducibility_summary.json").exists()
    assert (root / "analysis" / "ready_for_writing_status.json").exists()


def test_stable_vs_volatile_hashing_rule(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "analysis").mkdir(parents=True, exist_ok=True)
    (bundle / "analysis" / "stable.json").write_text("{\"ok\": true}\n", encoding="utf-8")
    (bundle / "metrics").mkdir(parents=True, exist_ok=True)
    (bundle / "metrics" / "volatile.csv").write_text("count,1\n", encoding="utf-8")

    spec = {
        "asset_name": "synthetic",
        "asset_tier": "critical",
        "bundle_root": str(bundle),
        "required_files": ["analysis/stable.json", "metrics/volatile.csv"],
        "key_output_paths": ["analysis/stable.json"],
        "rerun_entrypoint": f"python3 -c \"from pathlib import Path; Path('{(bundle / 'metrics' / 'volatile.csv').as_posix()}').write_text('count,2\\n', encoding='utf-8')\"",
        "claim_ids": ["CLAIM-X"],
    }
    out = verify_asset_reproducibility(spec, Path.cwd())
    assert out["required_files_present"] is True
    assert out["rerun_success"] is True
    assert out["stable_key_outputs"] is True
    assert out["reproducible"] is False
    assert out["fresh_computation_verified"] is False


def test_note_status_consistency(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "analysis").mkdir(parents=True, exist_ok=True)
    (bundle / "analysis" / "stable.json").write_text("{\"ok\": true}\n", encoding="utf-8")
    note_path = tmp_path / "freeze_note.md"

    cfg = {
        "freeze_id": "reproducibility_freeze",
        "artifact_subdir": "reproducibility_freeze",
        "findings_note_path": str(note_path),
        "write_repo_note": True,
        "assets": [
            {
                "asset_name": "synthetic",
                "asset_tier": "critical",
                "bundle_root": str(bundle),
                "required_files": ["analysis/stable.json"],
                "key_output_paths": ["analysis/stable.json"],
                "rerun_entrypoint": "python3 -c \"print('ok')\"",
                "claim_ids": ["CLAIM-X"],
            }
        ],
        "claims": [{"claim_id": "CLAIM-X", "description": "x", "asset_names": ["synthetic"]}],
        "nonblocking_caveats": [],
    }
    out = build_evidence_ledger(cfg, output_root=tmp_path / "freeze", use_cache=False)
    status = json.loads(
        (Path(out["artifact_root"]) / "analysis" / "ready_for_writing_status.json").read_text(encoding="utf-8")
    )
    note_text = note_path.read_text(encoding="utf-8")
    if status["ready_for_writing"]:
        assert "**is the repo ready for writing the paper? yes**" in note_text
    else:
        assert "**is the repo ready for writing the paper? no**" in note_text
