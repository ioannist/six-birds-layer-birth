import json
from pathlib import Path

from layerbirth.robustness import evaluate_variant_window, run_robustness_scan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cfg(name: str) -> dict:
    p = _repo_root() / "configs" / "robustness" / name
    return json.loads(p.read_text(encoding="utf-8"))


def test_robustness_configs_load():
    for name in ("equilibrium_like_robustness.json", "driven_low_bias_robustness.json"):
        cfg = _cfg(name)
        assert "variants" in cfg
        assert "lambda_grid" in cfg


def test_stability_scoring_rule():
    lambdas = [0.0, 0.5, 1.0]
    seq = [
        {"closure_strength_lambda": 0.0, "objecthood_order": 0.0, "closure_error": 0.0},
        {"closure_strength_lambda": 0.5, "objecthood_order": 0.2, "closure_error": 0.0},
        {"closure_strength_lambda": 1.0, "objecthood_order": 0.25, "closure_error": 0.0},
    ]
    out = evaluate_variant_window(seq, lambdas)
    assert out["window_present"] is True
    assert out["best_interval_index"] == 0

    flat = [
        {"closure_strength_lambda": 0.0, "objecthood_order": 0.1, "closure_error": 0.1},
        {"closure_strength_lambda": 0.5, "objecthood_order": 0.1, "closure_error": 0.1},
        {"closure_strength_lambda": 1.0, "objecthood_order": 0.1, "closure_error": 0.1},
    ]
    out2 = evaluate_variant_window(flat, lambdas)
    assert out2["window_present"] is False

    baseline_idx = 2
    shifted_idx = 1
    assert abs(shifted_idx - baseline_idx) <= 1
    unstable_idx = 0
    assert abs(unstable_idx - baseline_idx) > 1


def test_equilibrium_cheap_execution_and_cache(tmp_path: Path):
    cfg = _cfg("equilibrium_like_robustness.json")
    cfg["findings_note_path"] = str(tmp_path / "eq_note.md")
    first = run_robustness_scan(cfg, output_root=tmp_path, use_cache=True)
    second = run_robustness_scan(cfg, output_root=tmp_path, use_cache=True)
    root = Path(first["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "metrics" / "metrics.csv").exists()
    assert (root / "summary.json").exists()
    assert (root / "plots").exists()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert "variants" in summary
    assert "robust_enough_to_continue" in summary
    assert second["cached_count"] > 0


def test_driven_summary_contains_affinity_diagnosis(tmp_path: Path):
    cfg = _cfg("driven_low_bias_robustness.json")
    cfg["findings_note_path"] = str(tmp_path / "dr_note.md")
    out = run_robustness_scan(cfg, output_root=tmp_path, use_cache=True)
    summary = json.loads(Path(out["summary_path"]).read_text(encoding="utf-8"))
    assert "variants" in summary
    assert "robust_enough_to_continue" in summary
    assert "affinity_diagnosis" in summary
