from __future__ import annotations

import csv
import json
from pathlib import Path

from layerbirth.robustness import (
    format_lens_suite_summary,
    run_class_iii_class_iv_lens_suite,
    summarize_lens_suite,
)


def test_config_loading() -> None:
    cfg_path = Path("configs/robustness/class_iii_class_iv_lens_suite.json")
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["class_iii"]["confirmation_summary_path"].startswith("results/pilots/class_iii_strict_confirmation_v2")
    assert cfg["class_iv"]["campaign_config_path"] == "configs/campaigns/class_iv_full_campaign.json"


def _base_rows() -> list[dict[str, object]]:
    return [
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "manual_family_coarse_lens", "size": 32, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-III", "p4_class_active": True},
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "manual_family_coarse_lens", "size": 128, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-III", "p4_class_active": True},
    ]


def test_verdict_rule_plumbing() -> None:
    rows = _base_rows()
    strong_rows = rows + [
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "spectral_sign_pattern_lens", "size": 32, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-III", "p4_class_active": True},
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "spectral_sign_pattern_lens", "size": 128, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-III", "p4_class_active": True},
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "diffusion_quantile_lens", "size": 32, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-III", "p4_class_active": True},
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "diffusion_quantile_lens", "size": 128, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-III", "p4_class_active": True},
    ]
    s = summarize_lens_suite(strong_rows)
    assert s["per_class_robustness_status"] == "strong"

    partial_rows = rows + [
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "spectral_sign_pattern_lens", "size": 32, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-III", "p4_class_active": True},
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "spectral_sign_pattern_lens", "size": 128, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-III", "p4_class_active": True},
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "diffusion_quantile_lens", "size": 32, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-I", "p4_class_active": False},
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "diffusion_quantile_lens", "size": 128, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-I", "p4_class_active": False},
    ]
    p = summarize_lens_suite(partial_rows)
    assert p["per_class_robustness_status"] == "partial"

    fragile_rows = rows + [
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "spectral_sign_pattern_lens", "size": 32, "p5_state": "inactive", "p6_drive_state": "inactive", "canonical_class_label": "unclassified", "p4_class_active": False},
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "spectral_sign_pattern_lens", "size": 128, "p5_state": "inactive", "p6_drive_state": "inactive", "canonical_class_label": "unclassified", "p4_class_active": False},
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "diffusion_quantile_lens", "size": 32, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-I", "p4_class_active": False},
        {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "manual_reference_label": "Class-III", "lens_name": "diffusion_quantile_lens", "size": 128, "p5_state": "active", "p6_drive_state": "inactive", "canonical_class_label": "Class-I", "p4_class_active": False},
    ]
    f = summarize_lens_suite(fragile_rows)
    assert f["per_class_robustness_status"] == "fragile"

    suite = format_lens_suite_summary(s, p)
    assert suite["class_iii_lens_robust_for_paper"] is True
    assert suite["class_iv_lens_robust_for_paper"] is True
    assert suite["lens_robustness_suite_supported"] is True


def test_cheap_execution(tmp_path: Path) -> None:
    cfg = json.loads(Path("configs/robustness/class_iii_class_iv_lens_suite.json").read_text(encoding="utf-8"))
    cfg["artifact_subdir"] = "class_iii_class_iv_lens_suite_test"
    cfg["size_panel"] = [32]
    cfg["lambda_grid"] = [0.30, 0.60, 1.00]
    out = run_class_iii_class_iv_lens_suite(cfg, output_root=tmp_path, use_cache=True)

    root = Path(out["artifact_root"])
    assert root.exists()
    assert (root / "analysis" / "class_iii_lens_summary.json").exists()
    assert (root / "analysis" / "class_iv_lens_summary.json").exists()
    assert (root / "analysis" / "lens_suite_summary.json").exists()

    rows = list(csv.DictReader((root / "metrics" / "metrics.csv").open("r", encoding="utf-8")))
    assert any(r["class_name"] == "Class-III" for r in rows)
    assert any(r["class_name"] == "Class-IV" for r in rows)

    summary = json.loads((root / "analysis" / "lens_suite_summary.json").read_text(encoding="utf-8"))
    assert "lens_robustness_suite_supported" in summary

    out2 = run_class_iii_class_iv_lens_suite(cfg, output_root=tmp_path, use_cache=True)
    assert Path(out2["artifact_root"]).exists()
