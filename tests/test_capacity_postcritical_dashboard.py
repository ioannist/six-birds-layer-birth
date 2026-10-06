from __future__ import annotations

import json
from pathlib import Path

from layerbirth.capacity_dashboard import (
    _trend_label,
    build_capacity_postcritical_dashboard,
    summarize_capacity_postcritical_story,
)


def test_config_loading() -> None:
    cfg_path = Path("configs/dashboards/capacity_postcritical_dashboard.json")
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["source_capacity_bundle_root"] == "results/campaigns/capacity_depth_extension"


def test_trend_label_plumbing() -> None:
    assert _trend_label([1.0, 1.0, 1.0]) == "flat"
    assert _trend_label([1.0, 1.5, 2.0]) == "increasing"
    assert _trend_label([2.0, 1.5, 1.0]) == "decreasing"
    assert _trend_label([1.0, 2.0, 1.5]) == "mixed"


def test_dashboard_verdict_plumbing() -> None:
    source_root = Path("results/campaigns/capacity_depth_extension")
    rows = [
        {"class_name": "Class-I"},
        {"class_name": "Class-II"},
        {"class_name": "Class-III"},
        {"class_name": "Class-IV"},
    ]
    s = summarize_capacity_postcritical_story(source_root, rows)
    assert s["capacity_primary_classifier"] is False
    assert s["capacity_axis_alignment"] in {"p4", "p6", "mixed"}
    assert isinstance(s["capacity_postcritical_story_supported"], bool)


def test_cheap_execution(tmp_path: Path) -> None:
    cfg = json.loads(Path("configs/dashboards/capacity_postcritical_dashboard.json").read_text(encoding="utf-8"))
    cfg["artifact_subdir"] = "capacity_postcritical_dashboard_test"
    out = build_capacity_postcritical_dashboard(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out["artifact_root"])

    assert root.exists()
    assert (root / "analysis" / "representative_capacity_table.json").exists()
    assert (root / "analysis" / "capacity_dashboard_summary.json").exists()
    assert (root / "plots" / "capacity_postcritical_comparison.png").exists()
    assert (root / "plots" / "capacity_axis_summary.png").exists()

    rows = json.loads((root / "analysis" / "representative_capacity_table.json").read_text(encoding="utf-8"))["rows"]
    assert {r["class_name"] for r in rows} == {"Class-I", "Class-II", "Class-III", "Class-IV"}

    out2 = build_capacity_postcritical_dashboard(cfg, output_root=tmp_path, use_cache=True)
    assert Path(out2["artifact_root"]).exists()
