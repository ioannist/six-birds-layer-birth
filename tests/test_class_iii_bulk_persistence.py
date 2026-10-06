import json
from pathlib import Path

from layerbirth.class3 import (
    detect_boundary_floor_censoring,
    format_class_iii_bulk_persistence_summary,
    run_class_iii_bulk_persistence,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads((_repo_root() / "configs" / "pilots" / "class_iii_bulk_persistence.json").read_text(encoding="utf-8"))


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["pilot_id"] == "class_iii_bulk_persistence_v1"


def test_floor_censoring_logic():
    grid = [0.05, 0.1, 0.2]
    a = detect_boundary_floor_censoring({"ce_boundary_tau1": 0.05, "ce_boundary_tau2": 0.1, "mobj_boundary_tau1": 0.05, "mobj_boundary_tau2": 0.2}, grid)
    assert a["ce_floor_censored_tau1"] is True
    assert a["any_floor_censored"] is True
    b = detect_boundary_floor_censoring({"ce_boundary_tau1": 0.1, "ce_boundary_tau2": 0.2, "mobj_boundary_tau1": 0.1, "mobj_boundary_tau2": 0.2}, grid)
    assert b["any_floor_censored"] is False


def test_persistence_and_verdict_logic():
    summary = {
        "strict_class_iii_found": False,
        "floor_censoring_was_binding": True,
        "bulk_design_promising": True,
        "final_verdict": "bulk_persistence_promising_near_miss",
    }
    out = format_class_iii_bulk_persistence_summary([], [], [], summary, {"variant_id": "x"})
    assert out["summary"]["final_verdict"] == "bulk_persistence_promising_near_miss"


def test_cheap_execution(tmp_path: Path):
    cfg = _load_cfg()
    cfg["low_lambda_grid"] = [0.05, 0.55, 1.0]
    cfg["group2_new_bulk"][0]["param_grid"] = {"visible_hidden_weight": [0.02], "hidden_cross_weight": [1.2]}
    cfg["group2_new_bulk"][1]["param_grid"] = {"fast_portal_weight": [0.02], "portal_cross_weight": [1.2], "portal_self_weight": [0.5]}
    cfg["findings_note_path"] = str(tmp_path / "S2-04_class_iii_bulk_persistence.md")
    out = run_class_iii_bulk_persistence(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "persistence_summary.json").exists()
    assert out["final_verdict"] in {"strict_bulk_class_iii_found", "bulk_persistence_promising_near_miss", "surface_effect_or_censoring_only"}
