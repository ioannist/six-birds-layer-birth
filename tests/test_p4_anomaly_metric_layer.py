import json
from pathlib import Path

from layerbirth.p4 import (
    compute_boundary_shift_summary,
    evaluate_p4_profile,
    run_p4_anomaly_metric_layer,
)
from layerbirth.taxonomy import classify_activation_signature


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "metrics" / "p4_anomaly_metric_layer.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["metric_layer_id"] == "p4_anomaly_metric_layer_v1"
    assert len(cfg["synthetic_cases"]) == 4


def test_synthetic_toy_logic():
    cfg = _load_cfg()
    thr = cfg["thresholds"]
    by_name = {c["name"]: c for c in cfg["synthetic_cases"]}
    for name, case in by_name.items():
        s = compute_boundary_shift_summary(case["tau1"], case["tau2"])
        ev = evaluate_p4_profile(s, thr)
        assert ev["p4_like_signal"] == case["expected"]["p4_like_signal"]
        assert ev["p4_class_active"] == case["expected"]["p4_class_active"]


def test_taxonomy_integration_logic():
    sig = {
        "Class-I": {"P5_active": True, "P6_drive_active": False, "P4_anomalous": False},
        "Class-II": {"P5_active": True, "P6_drive_active": True, "P4_anomalous": False},
        "Class-III": {"P5_active": True, "P6_drive_active": False, "P4_anomalous": True},
        "Class-IV": {"P5_active": True, "P6_drive_active": True, "P4_anomalous": True},
    }
    # p4_like true + p4_class_active false should keep canonical P4 inactive
    assert classify_activation_signature("active", "inactive", "inactive", sig) == "Class-I"
    assert classify_activation_signature("active", "active", "inactive", sig) == "Class-II"


def test_cheap_execution_and_cache(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-03_p4_anomaly_metric_layer.md")
    out1 = run_p4_anomaly_metric_layer(cfg, output_root=tmp_path, use_cache=True)
    out2 = run_p4_anomaly_metric_layer(cfg, output_root=tmp_path, use_cache=True)

    root = Path(out1["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "reference_p4_metrics.json").exists()
    assert (root / "analysis" / "taxonomy_integration_check.json").exists()
    assert (Path(out1["taxonomy_artifact_root"]) / "manifest.json").exists()
    assert (Path(out1["dashboard_artifact_root"]) / "manifest.json").exists()
    assert out1["class_i_reference"]["canonical_class_label"] == "Class-I"
    assert out1["class_ii_reference"]["canonical_class_label"] == "Class-II"
    assert out2["class_i_reference"]["canonical_class_label"] == "Class-I"
