import json
from pathlib import Path

from layerbirth.dashboard import build_class_consolidation_dashboard, format_consolidation_summary


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "dashboards" / "class_i_class_ii_consolidation.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _claim_supported(row: dict, rule: dict) -> bool:
    labels = row["class_i_label"] == "Class-I" and row["class_ii_label"] == "Class-II"
    primitives = row["class_i_p5"] == "active" and row["class_ii_p5"] == "active" and row["class_i_p6"] == "inactive" and row["class_ii_p6"] == "active"
    contrast = row["ratio"] >= float(rule["affinity_contrast_ratio_min"]) or row["diff"] >= float(rule["affinity_contrast_difference_min"])
    arrow_ok = row["reversible_false_positives"] <= int(rule["no_fake_arrow_false_positive_max"])
    return bool(labels and primitives and contrast and arrow_ok)


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["dashboard_id"] == "class_i_class_ii_consolidation_v1"
    assert "sources" in cfg


def test_summary_rule_plumbing():
    cfg = _load_cfg()
    rule = cfg["publishable_claim_rule"]
    good = {
        "class_i_label": "Class-I",
        "class_ii_label": "Class-II",
        "class_i_p5": "active",
        "class_ii_p5": "active",
        "class_i_p6": "inactive",
        "class_ii_p6": "active",
        "ratio": 2e3,
        "diff": 0.02,
        "reversible_false_positives": 0,
    }
    bad = dict(good)
    bad["reversible_false_positives"] = 1
    assert _claim_supported(good, rule)
    assert not _claim_supported(bad, rule)
    merged = format_consolidation_summary({"a": 1}, {"b": 2})
    assert merged["a"] == 1 and merged["b"] == 2


def test_cheap_execution_and_cache(tmp_path: Path):
    cfg = _load_cfg()
    cfg["findings_note_path"] = str(tmp_path / "S2-02_class_i_class_ii_consolidation.md")
    out1 = build_class_consolidation_dashboard(cfg, output_root=tmp_path, use_cache=True)
    out2 = build_class_consolidation_dashboard(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out1["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "consolidation_summary.json").exists()
    assert (root / "analysis" / "no_fake_arrow_summary.json").exists()
    assert (root / "plots" / "class_i_vs_class_ii_overview.png").exists()
    metrics = (root / "metrics" / "metrics.csv").read_text(encoding="utf-8")
    assert "class_i_reference" in metrics
    assert "class_ii_reference" in metrics
    assert out2["cached_count"] >= out1["cached_count"]
