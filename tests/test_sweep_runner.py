import csv
import json
from pathlib import Path

import pytest

from layerbirth.sweep import AGGREGATE_FIELDS, execute_sweep_run, expand_sweep_grid, run_sweep


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tiny_grid_config() -> dict:
    cfg_path = _repo_root() / "configs" / "sweeps" / "tiny_grid.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def test_expand_sweep_grid_deterministic_and_complete():
    cfg = _tiny_grid_config()
    runs_a = expand_sweep_grid(cfg)
    runs_b = expand_sweep_grid(cfg)
    assert len(runs_a) == 6
    assert [r["run_id"] for r in runs_a] == [r["run_id"] for r in runs_b]
    for run in runs_a:
        for key in (
            "family_name",
            "size",
            "seed",
            "closure_strength_lambda",
            "lens_spec",
            "lift_spec",
            "tau_protocol",
            "run_id",
        ):
            assert key in run


def test_one_run_executes_and_writes_artifacts(tmp_path: Path):
    cfg = _tiny_grid_config()
    run = expand_sweep_grid(cfg)[0]
    row = execute_sweep_run(run, output_root=tmp_path, use_cache=True)
    assert row["cache_status"] == "executed"
    run_root = tmp_path / run["sweep_id"] / "runs" / run["run_id"]
    assert (run_root / "manifest.json").exists()
    metrics_csv = run_root / "metrics" / "metrics.csv"
    assert metrics_csv.exists()
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 1
    for field in AGGREGATE_FIELDS:
        assert field in rows[0]


def test_full_tiny_grid_executes_and_cache_behavior(tmp_path: Path):
    cfg = _tiny_grid_config()
    first = run_sweep(cfg, output_root=tmp_path, use_cache=True)
    second = run_sweep(cfg, output_root=tmp_path, use_cache=True)

    assert first["total_runs"] == 6
    assert first["executed_count"] == 6
    assert first["cached_count"] == 0
    assert second["total_runs"] == 6
    assert second["cached_count"] == 6
    assert second["executed_count"] == 0

    sweep_root = tmp_path / cfg["sweep_id"]
    assert (sweep_root / "manifest.json").exists()
    aggregate = sweep_root / "metrics" / "metrics.csv"
    assert aggregate.exists()
    with aggregate.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6


def test_lambda_is_operational_in_tiny_grid(tmp_path: Path):
    cfg = _tiny_grid_config()
    run_sweep(cfg, output_root=tmp_path, use_cache=True)
    aggregate = tmp_path / cfg["sweep_id"] / "metrics" / "metrics.csv"
    with aggregate.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    target = [r for r in rows if r["family_name"] == "reversible_block_family" and r["seed"] == ""]
    assert len(target) == 2
    lambdas = sorted(float(r["closure_strength_lambda"]) for r in target)
    assert lambdas == [0.25, 0.75]
    ce_values = sorted(float(r["closure_error"]) for r in target)
    assert ce_values[0] != ce_values[1]


def test_validation_error_paths():
    cfg = _tiny_grid_config()
    bad_size = json.loads(json.dumps(cfg))
    bad_size["family_specs"][0]["size_grid"] = [7]
    run = [r for r in expand_sweep_grid(bad_size) if r["family_name"] == "reversible_block_family"][0]
    with pytest.raises(ValueError):
        execute_sweep_run(run, output_root=Path("tmp"), use_cache=False)

    bad_family = json.loads(json.dumps(cfg))
    bad_family["family_specs"][0]["family_name"] = "unknown_family"
    run = [r for r in expand_sweep_grid(bad_family) if r["family_name"] == "unknown_family"][0]
    with pytest.raises(ValueError):
        execute_sweep_run(run, output_root=Path("tmp"), use_cache=False)

    bad_lens = json.loads(json.dumps(cfg))
    bad_lens["lens_grid"] = [{"name": "unknown_lens"}]
    run = expand_sweep_grid(bad_lens)[0]
    with pytest.raises(ValueError):
        execute_sweep_run(run, output_root=Path("tmp"), use_cache=False)

    bad_lift = json.loads(json.dumps(cfg))
    bad_lift["lift_grid"] = [{"name": "unknown_lift"}]
    run = expand_sweep_grid(bad_lift)[0]
    with pytest.raises(ValueError):
        execute_sweep_run(run, output_root=Path("tmp"), use_cache=False)

    bad_control = json.loads(json.dumps(cfg))
    bad_control["control_application_name"] = "bad_mode"
    run = expand_sweep_grid(bad_control)[0]
    with pytest.raises(ValueError):
        execute_sweep_run(run, output_root=Path("tmp"), use_cache=False)
