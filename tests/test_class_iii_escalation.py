import json
from pathlib import Path

from layerbirth.class3 import (
    format_class_iii_escalation_summary,
    run_class_iii_escalation,
    select_stage2_variants,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads((_repo_root() / "configs" / "pilots" / "class_iii_escalation.json").read_text(encoding="utf-8"))


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["escalation_id"] == "class_iii_escalation_v1"
    assert len(cfg["stage1"]["families"]) == 2


def test_exploratory_label_and_verdict_logic():
    stage1 = [
        {
            "family_name": "hidden_sector",
            "variant_id": "a",
            "candidate_is_class_iii_strict": False,
            "mobj_dominant_p4_candidate": True,
            "staging_shift_ce": 0.08,
            "staging_shift_mobj": 0.16,
            "mobj_minus_ce_shift": 0.08,
            "affinity_ref_max": 0.0,
            "candidate_p5_state": "active",
            "candidate_p6_drive_state": "inactive",
            "staging_gap_anomaly_score": 0.16,
        },
        {
            "family_name": "hidden_sector",
            "variant_id": "b",
            "candidate_is_class_iii_strict": False,
            "mobj_dominant_p4_candidate": True,
            "staging_shift_ce": 0.07,
            "staging_shift_mobj": 0.17,
            "mobj_minus_ce_shift": 0.10,
            "affinity_ref_max": 0.0,
            "candidate_p5_state": "active",
            "candidate_p6_drive_state": "inactive",
            "staging_gap_anomaly_score": 0.17,
        },
    ]
    out = format_class_iii_escalation_summary(stage1, [], {"families": {"hidden_sector": {"mobj_dominant_p4_candidate_count": 2}}}, stage1[1])
    assert out["final_verdict"] == "systematic_mobj_dominant_near_miss"


def test_stage2_selection_ranking():
    rows = [
        {"family_name": "f", "variant_id": "a", "candidate_p5_state": "active", "candidate_p6_drive_state": "inactive", "staging_gap_anomaly_score": 0.2, "staging_shift_mobj": 0.15, "affinity_ref_max": 0.0},
        {"family_name": "f", "variant_id": "b", "candidate_p5_state": "active", "candidate_p6_drive_state": "inactive", "staging_gap_anomaly_score": 0.21, "staging_shift_mobj": 0.14, "affinity_ref_max": 0.0},
        {"family_name": "f", "variant_id": "c", "candidate_p5_state": "inactive", "candidate_p6_drive_state": "inactive", "staging_gap_anomaly_score": 0.9, "staging_shift_mobj": 0.9, "affinity_ref_max": 0.0},
    ]
    top = select_stage2_variants(rows, top_k=2)
    assert [r["variant_id"] for r in top] == ["b", "a"]


def test_cheap_execution_and_cache(tmp_path: Path):
    cfg = _load_cfg()
    cfg["stage1"]["sizes"] = [8]
    cfg["stage1"]["lambda_grid"] = [0.75, 1.0]
    for fam in cfg["stage1"]["families"]:
        fam["param_grid"] = {k: [v[0]] for k, v in fam["param_grid"].items()}
    cfg["stage2"]["top_k"] = 1
    cfg["findings_note_path"] = str(tmp_path / "S2-04_class_iii_escalation.md")

    out1 = run_class_iii_escalation(cfg, output_root=tmp_path, use_cache=True)
    out2 = run_class_iii_escalation(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out1["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "stage1_candidate_table.csv").exists()
    assert (root / "analysis" / "sensor_asymmetry_summary.json").exists()
    assert (root / "analysis" / "escalation_summary.json").exists()
    assert out2["final_verdict"] in {"strict_class_iii_candidate_found", "systematic_mobj_dominant_near_miss", "no_useful_signal"}
