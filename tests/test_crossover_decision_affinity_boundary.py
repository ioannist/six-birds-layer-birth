import json
from pathlib import Path

import numpy as np

from layerbirth.crossover_decision import (
    bias_to_cycle_weights,
    evaluate_crossover_necessity,
    run_crossover_decision_affinity_boundary,
)
from layerbirth.numeric import validate_row_stochastic
from layerbirth.substrates import build_substrate_family


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "campaigns" / "crossover_decision_affinity_boundary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_config_loads():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "crossover_decision_affinity_boundary"
    assert "decision_audit_inputs" in cfg
    assert "affinity_scan_panel" in cfg


def test_decision_rule_plumbing():
    strong = evaluate_crossover_necessity(
        {"best_fit": {"objective": 0.1, "lambda_c": 0.85}},
        {"best_fit": {"objective": 0.2, "lambda_c": 0.8}},
        {
            "class_separation_plausible": True,
            "affinity_contrast_ratio": 100.0,
            "affinity_contrast_difference": 0.02,
            "delta_lambda_c_fit": 0.05,
        },
        {"affinity_ratio_min": 10.0, "affinity_diff_min": 0.01, "lambda_delta_min": 0.05},
    )
    weak = evaluate_crossover_necessity(
        {"best_fit": {"objective": 0.1, "lambda_c": 0.85}},
        {"best_fit": {"objective": 0.2, "lambda_c": 0.8}},
        {
            "class_separation_plausible": False,
            "affinity_contrast_ratio": 2.0,
            "affinity_contrast_difference": 0.001,
            "delta_lambda_c_fit": 0.01,
        },
        {"affinity_ratio_min": 10.0, "affinity_diff_min": 0.01, "lambda_delta_min": 0.05},
    )
    assert strong["crossover_load_bearing"] is False
    assert weak["crossover_load_bearing"] is True


def test_bias_parameterization_and_kernel_stochasticity():
    sw, fw0, bw0 = bias_to_cycle_weights(0.1, 0.9, 0.0)
    _, fw1, bw1 = bias_to_cycle_weights(0.1, 0.9, 0.2)
    assert np.isclose(fw0, bw0)
    assert fw1 > fw0
    assert bw1 < bw0
    p = build_substrate_family(
        "driven_cycle_family",
        n=8,
        self_weight=sw,
        forward_weight=fw1,
        backward_weight=bw1,
    )["P"]
    validate_row_stochastic(np.asarray(p, dtype=np.float64))


def test_reduced_execution_and_cache(tmp_path: Path):
    cfg = _load_cfg()
    cfg["affinity_scan_panel"]["sizes"] = [8]
    cfg["affinity_scan_panel"]["bias_grid"] = [0.0, 0.2]
    cfg["affinity_scan_panel"]["lambda_grid"] = [0.75, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "lb17_decision_note.md")
    out1 = run_crossover_decision_affinity_boundary(cfg, output_root=tmp_path, use_cache=True)
    out2 = run_crossover_decision_affinity_boundary(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out1["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "decision_audit.json").exists()
    assert (root / "analysis" / "affinity_boundary_summary.json").exists()
    assert (root / "plots" / "affinity_vs_bias.png").exists()
    assert "crossover_load_bearing" in out1
    assert "affinity_crossover_path_plausible" in out1
    assert out2["cached_count"] > 0
