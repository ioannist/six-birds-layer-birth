from __future__ import annotations

import json
from pathlib import Path

from layerbirth.upstream_bridge import (
    assign_bridge_confidence,
    build_upstream_pica_bridge,
    classify_upstream_bridge_result,
)


def test_config_loading() -> None:
    cfg_path = Path("configs/bridges/upstream_pica_bridge.json")
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["upstream_source_root"].startswith("vendor/six-birds-pica-ma_snapshot_v71")
    assert len(cfg["candidates"]) >= 1


def test_classification_confidence_plumbing() -> None:
    cls = classify_upstream_bridge_result(
        {
            "canonical_class_label": "Class-II",
            "p5_state": "active",
            "p6_drive_state": "active",
            "p4_class_active": False,
        },
        {},
    )
    assert cls["canonical_class_label"] == "Class-II"
    assert assign_bridge_confidence(cls, support_instances=1) == "medium"
    assert assign_bridge_confidence(cls, support_instances=2) == "high"

    mixed = classify_upstream_bridge_result(
        {
            "canonical_class_label": "unclassified",
            "p5_state": "inactive",
            "p6_drive_state": "inactive",
            "p4_class_active": False,
        },
        {},
    )
    assert mixed["canonical_class_label"] in {"mixed", "unclear"}
    assert assign_bridge_confidence(mixed, support_instances=2) == "low"


def test_cheap_execution(tmp_path: Path) -> None:
    cfg = json.loads(Path("configs/bridges/upstream_pica_bridge.json").read_text(encoding="utf-8"))
    cfg["artifact_subdir"] = "upstream_pica_bridge_test"
    cfg["max_candidates"] = 1
    cfg["candidates"] = [cfg["candidates"][0]]
    cfg["lambda_grid"] = [0.20, 0.60, 1.00]

    out = build_upstream_pica_bridge(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out["artifact_root"])
    assert root.exists()
    assert (root / "analysis" / "upstream_candidate_table.json").exists()
    assert (root / "analysis" / "upstream_bridge_summary.json").exists()
    assert (root / "plots" / "upstream_on_four_class_map.png").exists()
    assert (root / "plots" / "upstream_observable_axes.png").exists()

    rows = json.loads((root / "analysis" / "upstream_candidate_table.json").read_text(encoding="utf-8"))["candidates"]
    assert len(rows) >= 1
    assert any(r["bridge_candidate_usable"] for r in rows)
    assert out["upstream_bridge_success"]

    out2 = build_upstream_pica_bridge(cfg, output_root=tmp_path, use_cache=True)
    assert Path(out2["artifact_root"]).exists()
