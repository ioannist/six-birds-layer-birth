import json
from pathlib import Path

from layerbirth.class3 import (
    evaluate_p4_criteria_on_profiles,
    run_class_iii_refinement_and_p4_audit,
    select_refinement_stage2_variants,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads((_repo_root() / "configs" / "pilots" / "class_iii_refinement_and_p4_audit.json").read_text(encoding="utf-8"))


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["audit_id"] == "class_iii_refinement_and_p4_audit_v1"


def test_criterion_audit_logic_and_recommendation_order():
    cfg = _load_cfg()
    criteria = cfg["criteria"]
    profiles = {
        "class_i_reference": {"staging_shift_ce": 0.0, "staging_shift_mobj": 0.0, "presence_shift_any": False},
        "class_ii_reference": {"staging_shift_ce": 0.0, "staging_shift_mobj": 0.0, "presence_shift_any": False},
        "old_hidden_sector_128": {"staging_shift_ce": 0.0, "staging_shift_mobj": 0.0, "presence_shift_any": False},
        "old_two_timescale_128": {"staging_shift_ce": 0.0, "staging_shift_mobj": 0.0, "presence_shift_any": False},
        "replicated_portal_best_128": {"staging_shift_ce": 0.08, "staging_shift_mobj": 0.16, "presence_shift_any": False},
        "replicated_portal_best_256": {"staging_shift_ce": 0.09, "staging_shift_mobj": 0.17, "presence_shift_any": False},
    }
    out = evaluate_p4_criteria_on_profiles(profiles, criteria)
    assert out["criteria"]["hybrid_ce_gated"]["criterion_supported"] is True
    assert out["recommended_p4_criterion"] in {"strict_dual_shift", "hybrid_ce_gated", "mobj_only_shift", "none"}


def test_stage2_selection_ce_first_ranking():
    rows = [
        {"variant_id": "a", "candidate_p5_state": "active", "candidate_p6_drive_state": "inactive", "staging_shift_ce": 0.10, "staging_shift_mobj": 0.30, "affinity_ref_max": 0.0},
        {"variant_id": "b", "candidate_p5_state": "active", "candidate_p6_drive_state": "inactive", "staging_shift_ce": 0.11, "staging_shift_mobj": 0.20, "affinity_ref_max": 0.0},
        {"variant_id": "c", "candidate_p5_state": "active", "candidate_p6_drive_state": "inactive", "staging_shift_ce": 0.09, "staging_shift_mobj": 0.90, "affinity_ref_max": 0.0},
        {"variant_id": "d", "candidate_p5_state": "inactive", "candidate_p6_drive_state": "inactive", "staging_shift_ce": 0.5, "staging_shift_mobj": 0.5, "affinity_ref_max": 0.0},
    ]
    top = select_refinement_stage2_variants(rows, top_k=3)
    assert [r["variant_id"] for r in top] == ["b", "a", "c"]


def test_cheap_execution(tmp_path: Path):
    cfg = _load_cfg()
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["parameter_grid"] = {
        "fast_portal_weight": [0.02],
        "portal_cross_weight": [1.2],
        "portal_self_weight": [0.5]
    }
    cfg["stage2_top_k"] = 1
    cfg["findings_note_path"] = str(tmp_path / "S2-04_class_iii_refinement_and_p4_audit.md")
    out = run_class_iii_refinement_and_p4_audit(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "p4_criterion_audit.json").exists()
    assert (root / "analysis" / "refinement_summary.json").exists()
    assert out["final_verdict"] in {
        "strict_class_iii_found",
        "hybrid_criterion_supported_bulk_candidate_ready",
        "mobj_only_supported_but_too_broad",
        "no_stable_class_iii_signal_yet",
    }
