import json
from pathlib import Path

from layerbirth.taxonomy import (
    classify_activation_signature,
    evaluate_p4_anomaly,
    evaluate_p5_activation,
    evaluate_p6_drive_activation,
    run_canonical_class_rubric,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "taxonomy" / "canonical_class_rubric.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["rubric_id"] == "canonical_class_rubric_v1"
    assert len(cfg["reference_profiles"]) >= 2


def test_primitive_rules_and_unknown_handling():
    p5_thr = {"ce_target": 0.025, "mobj_target": 0.9, "ce_inactive_min": 0.05, "mobj_inactive_max": 0.85}
    p6_thr = {"p6_active_min": 1e-3, "p6_inactive_max": 1e-6}
    p4_thr = {"p4_boundary_shift_active_min": 0.05}

    assert evaluate_p5_activation({"ce_boundary_lambda": 0.8, "mobj_boundary_lambda": 0.9, "min_ce": 0.0, "max_mobj": 1.0}, p5_thr)["state"] == "active"
    assert evaluate_p5_activation({"ce_boundary_lambda": None, "mobj_boundary_lambda": None, "min_ce": 0.06, "max_mobj": 0.8}, p5_thr)["state"] == "inactive"
    assert evaluate_p5_activation({"ce_boundary_lambda": 0.8, "mobj_boundary_lambda": None, "min_ce": 0.01, "max_mobj": 0.95}, p5_thr)["state"] == "unknown"

    assert evaluate_p6_drive_activation(2e-3, p6_thr)["state"] == "active"
    assert evaluate_p6_drive_activation(1e-8, p6_thr)["state"] == "inactive"
    assert evaluate_p6_drive_activation(1e-5, p6_thr)["state"] == "unknown"

    p4_like_only = evaluate_p4_anomaly(
        {"tau_ce_shift": 0.06, "tau_mobj_shift": 0.0, "tau_ce_presence_change": False, "tau_mobj_presence_change": False},
        p4_thr,
    )
    assert p4_like_only["p4_like_signal"] is True
    assert p4_like_only["p4_class_active"] is False
    assert p4_like_only["state"] == "inactive"
    assert evaluate_p4_anomaly({"tau_ce_shift": 0.01, "tau_mobj_shift": 0.02, "tau_ce_presence_change": False, "tau_mobj_presence_change": False}, p4_thr)["state"] == "inactive"


def test_signature_mapping():
    cfg = _load_cfg()
    s = cfg["canonical_class_signatures"]
    assert classify_activation_signature("active", "inactive", "inactive", s) == "Class-I"
    assert classify_activation_signature("active", "active", "inactive", s) == "Class-II"
    assert classify_activation_signature("active", "inactive", "active", s) == "Class-III"
    assert classify_activation_signature("active", "active", "active", s) == "Class-IV"
    assert classify_activation_signature("unknown", "active", "inactive", s) == "unclassified"


def test_p4_like_without_class_activation() -> None:
    p4_thr = {"p4_like_shift_min": 0.05, "p4_class_dual_shift_min": 0.15}
    out = evaluate_p4_anomaly(
        {
            "tau_ce_shift": 0.061,
            "tau_mobj_shift": 0.115,
            "tau_ce_presence_change": False,
            "tau_mobj_presence_change": False,
        },
        p4_thr,
    )
    assert out["p4_like_signal"] is True
    assert out["p4_class_active"] is False
    assert out["state"] == "inactive"


def test_execution_and_cache_behavior(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-01_canonical_class_rubric.md")
    out1 = run_canonical_class_rubric(cfg, output_root=tmp_path, use_cache=True)
    out2 = run_canonical_class_rubric(cfg, output_root=tmp_path, use_cache=True)

    root = Path(out1["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "reference_classifications.json").exists()
    assert (root / "analysis" / "primitive_activation_table.csv").exists()

    assert out1["class_i_reference"]["canonical_class_label"] == "Class-I"
    assert out1["class_ii_reference"]["canonical_class_label"] == "Class-II"
    assert out2["class_i_reference"]["canonical_class_label"] == "Class-I"
    assert out2["class_ii_reference"]["canonical_class_label"] == "Class-II"
