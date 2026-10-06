from __future__ import annotations

import json
from pathlib import Path

from layerbirth.observable_map import (
    build_four_class_observable_map,
    compute_p4_dual_shift_min,
    compute_structural_boundary_lambda,
    summarize_observable_separation,
)


def test_config_loading() -> None:
    cfg_path = Path("configs/dashboards/four_class_observable_map.json")
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["sources"]["class_i_ii_consolidation_root"].startswith("results/dashboards/class_i_class_ii_consolidation")
    assert cfg["sources"]["class_iii_campaign_root"].startswith("results/campaigns/class_iii_full_campaign")
    assert cfg["sources"]["class_iv_campaign_root"].startswith("results/campaigns/class_iv_full_campaign")


def test_coordinate_and_separation_plumbing() -> None:
    assert compute_structural_boundary_lambda(0.5, 0.7) == 0.7
    assert compute_p4_dual_shift_min(0.2, 0.35) == 0.2

    rows = [
        {"class_name": "Class-I", "affinity_ref": 0.0, "p4_dual_shift_min": 0.05, "structural_boundary_lambda_ref": 0.8},
        {"class_name": "Class-II", "affinity_ref": 0.01, "p4_dual_shift_min": 0.01, "structural_boundary_lambda_ref": 0.5},
        {"class_name": "Class-III", "affinity_ref": 1e-10, "p4_dual_shift_min": 0.2, "structural_boundary_lambda_ref": 0.75},
        {"class_name": "Class-IV", "affinity_ref": 0.02, "p4_dual_shift_min": 0.3, "structural_boundary_lambda_ref": 0.55},
    ]
    s = summarize_observable_separation(rows)
    assert s["p6_axis_clean"] is True
    assert s["p4_axis_clean"] is True
    assert s["structural_boundary_spread_present"] is True
    assert s["observable_map_separable"] is True
    assert s["phase_map_figure_publishable"] is True


def test_cheap_execution(tmp_path: Path) -> None:
    cfg = json.loads(Path("configs/dashboards/four_class_observable_map.json").read_text(encoding="utf-8"))
    cfg["artifact_subdir"] = "four_class_observable_map_test"
    out = build_four_class_observable_map(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out["artifact_root"])
    assert root.exists()
    assert (root / "analysis" / "class_coordinates.json").exists()
    assert (root / "analysis" / "separation_summary.json").exists()
    assert (root / "plots" / "four_class_observable_map.png").exists()
    assert (root / "plots" / "observable_axes_panels.png").exists()
    coords = json.loads((root / "analysis" / "class_coordinates.json").read_text(encoding="utf-8"))["class_coordinates"]
    assert {r["class_name"] for r in coords} == {"Class-I", "Class-II", "Class-III", "Class-IV"}

    out2 = build_four_class_observable_map(cfg, output_root=tmp_path, use_cache=True)
    assert Path(out2["artifact_root"]).exists()
