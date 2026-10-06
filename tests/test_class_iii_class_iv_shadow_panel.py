from __future__ import annotations

import csv
import json
from pathlib import Path

from layerbirth.phenomenology import (
    build_shadow_ensemble_candidates,
    run_class_iii_class_iv_shadow_panel,
    summarize_shadow_panel,
)


def test_config_loading() -> None:
    cfg_path = Path("configs/phenomenology/class_iii_class_iv_shadow_panel.json")
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["class_iii"]["target_label"] == "Class-III"
    assert cfg["class_iv"]["target_label"] == "Class-IV"


def test_admissibility_threshold_logic() -> None:
    rep = {
        "representative_name": "replicated_portal_sp4",
        "base_kwargs": {"fast_portal_weight": 0.02, "portal_cross_weight": 1.2, "portal_self_weight": 4.0},
        "drive_kwargs": {},
    }
    spec = {"structural_keys": ["fast_portal_weight", "portal_cross_weight", "portal_self_weight"], "factors": [0.95, 1.0, 1.05]}
    cands = build_shadow_ensemble_candidates(rep, spec)
    assert len(cands) >= 6

    group_rows_ok = [
        {"n_admissible_replicates": 6, "size": 32, "closure_strength_lambda": 0.10, "susceptibility_delta_mobj": 0.2, "binder_delta_mobj": 0.1, "delta_mobj_mean": 0.01},
        {"n_admissible_replicates": 6, "size": 32, "closure_strength_lambda": 0.20, "susceptibility_delta_mobj": 0.3, "binder_delta_mobj": 0.2, "delta_mobj_mean": 0.04},
        {"n_admissible_replicates": 6, "size": 32, "closure_strength_lambda": 0.30, "susceptibility_delta_mobj": 0.4, "binder_delta_mobj": 0.3, "delta_mobj_mean": 0.08},
        {"n_admissible_replicates": 6, "size": 64, "closure_strength_lambda": 0.10, "susceptibility_delta_mobj": 0.2, "binder_delta_mobj": 0.1, "delta_mobj_mean": 0.02},
        {"n_admissible_replicates": 6, "size": 64, "closure_strength_lambda": 0.20, "susceptibility_delta_mobj": 0.3, "binder_delta_mobj": 0.2, "delta_mobj_mean": 0.05},
        {"n_admissible_replicates": 6, "size": 64, "closure_strength_lambda": 0.30, "susceptibility_delta_mobj": 0.4, "binder_delta_mobj": 0.3, "delta_mobj_mean": 0.09},
    ]
    admissible_ok = {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "admissible_count": 6}
    summary_ok = summarize_shadow_panel(group_rows_ok, admissible_ok, [0.1, 0.2, 0.3], 6)
    assert summary_ok["ensemble_label_stable"] is True
    assert summary_ok["usable_chi_u4_panel"] is True
    assert summary_ok["final_per_class_verdict"] == "usable_panel"

    admissible_bad = {"class_name": "Class-III", "representative_name": "replicated_portal_sp4", "admissible_count": 2}
    summary_bad = summarize_shadow_panel(group_rows_ok, admissible_bad, [0.1, 0.2, 0.3], 6)
    assert summary_bad["ensemble_label_stable"] is False
    assert summary_bad["final_per_class_verdict"] == "not_stable_under_ensemble_perturbation"


def test_cheap_execution(tmp_path: Path) -> None:
    cfg = json.loads(Path("configs/phenomenology/class_iii_class_iv_shadow_panel.json").read_text(encoding="utf-8"))
    cfg["size_panel"] = [16, 32]
    cfg["lambda_grid"] = [0.10, 0.30, 0.60, 1.00]
    cfg["min_admissible_replicates"] = 2
    cfg["class_iii"]["perturbation_spec"]["factors"] = [1.0, 1.05]
    cfg["class_iv"]["perturbation_spec"]["factors"] = [1.0, 1.05]

    out = run_class_iii_class_iv_shadow_panel(cfg, output_root=tmp_path, use_cache=True)
    c3_root = Path(out["class_iii"]["artifact_root"])
    c4_root = Path(out["class_iv"]["artifact_root"])

    for root in [c3_root, c4_root]:
        assert root.exists()
        assert (root / "analysis" / "admissibility_summary.json").exists()
        assert (root / "analysis" / "group_summary.csv").exists()
        assert (root / "analysis" / "panel_summary.json").exists()
        rows = list(csv.DictReader((root / "metrics" / "metrics.csv").open("r", encoding="utf-8")))
        assert len(rows) >= 1

    out2 = run_class_iii_class_iv_shadow_panel(cfg, output_root=tmp_path, use_cache=True)
    assert "phenomenology_suite_supported" in out2
