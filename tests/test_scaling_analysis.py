import csv
from pathlib import Path
import runpy

import numpy as np

from layerbirth.scaling import (
    MockFssParams,
    binder_like_cumulant,
    bootstrap_collapse_fit,
    estimate_pairwise_crossings,
    generate_mock_observations,
    grid_search_collapse_fit,
    group_order_observations,
    susceptibility_from_samples,
)

def test_formula_sanity():
    chi, _ = susceptibility_from_samples([0, 1, 0, 1], system_size=8)
    assert chi == 2.0
    u4, _ = binder_like_cumulant([1, 1, 1, 1])
    assert np.isclose(u4, 2.0 / 3.0, atol=1e-12)


def test_pairwise_crossing_recovery():
    rows = []
    lam_grid = [0.4, 0.5, 0.6]
    for lam in lam_grid:
        rows.append({"size": 16, "closure_strength_lambda": lam, "binder_like_cumulant": lam - 0.5})
        rows.append({"size": 32, "closure_strength_lambda": lam, "binder_like_cumulant": 0.5 - lam})
        rows.append({"size": 64, "closure_strength_lambda": lam, "binder_like_cumulant": 0.5 - lam})
    out = estimate_pairwise_crossings(rows)
    assert out["crossing_count"] >= 1
    assert np.isclose(out["crossing_lambda_c_mean"], 0.5, atol=1e-12)


def test_collapse_fit_recovery_on_mock_data():
    params = MockFssParams(
        sizes=[16, 32, 64],
        lambda_grid=[0.4, 0.45, 0.5, 0.55, 0.6],
        lambda_c_true=0.5,
        beta_over_nu_true=0.125,
        inv_nu_true=1.0,
        replicate_count=64,
        noise_sigma=0.01,
        seed=2026,
    )
    raw = generate_mock_observations(params)
    grouped = group_order_observations(raw)
    fit = grid_search_collapse_fit(
        grouped,
        lambda_c_grid=[0.45, 0.475, 0.5, 0.525, 0.55],
        beta_over_nu_grid=[0.05, 0.125, 0.2],
        inv_nu_grid=[0.75, 1.0, 1.25],
    )["best_fit"]
    assert abs(fit["lambda_c"] - 0.5) <= 0.05
    assert abs(fit["beta_over_nu"] - 0.125) <= 0.05
    assert abs(fit["inv_nu"] - 1.0) <= 0.25


def test_bootstrap_output_sanity():
    params = MockFssParams(
        sizes=[16, 32, 64],
        lambda_grid=[0.4, 0.45, 0.5, 0.55, 0.6],
        lambda_c_true=0.5,
        beta_over_nu_true=0.125,
        inv_nu_true=1.0,
        replicate_count=32,
        noise_sigma=0.01,
        seed=2026,
    )
    raw = generate_mock_observations(params)
    boot = bootstrap_collapse_fit(
        raw,
        {
            "lambda_c_grid": [0.45, 0.475, 0.5, 0.525, 0.55],
            "beta_over_nu_grid": [0.05, 0.125, 0.2],
            "inv_nu_grid": [0.75, 1.0, 1.25],
            "ci_percentiles": [5, 95],
            "n_bins": 12,
        },
        n_boot=20,
        seed=2026,
    )
    assert "ci" in boot
    assert boot["successful_bootstrap_count"] > 0
    assert np.isfinite(boot["ci"]["lambda_c"][0])


def test_smoke_bundle_sanity(tmp_path: Path):
    smoke_mod = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "smoke_scaling_analysis.py"))
    summary = smoke_mod["run_scaling_smoke"](output_root=tmp_path)
    root = Path(summary["artifact_root"])
    required = [
        root / "manifest.json",
        root / "config" / "config_snapshot.json",
        root / "seeds" / "seeds.json",
        root / "metrics" / "metrics.csv",
        root / "notes" / "findings.md",
        root / "env" / "environment.json",
        root / "analysis" / "mock_observations.csv",
        root / "analysis" / "group_summary.csv",
        root / "analysis" / "fit_summary.json",
        root / "plots" / "collapse_best_fit.png",
        root / "plots" / "binder_crossings.png",
    ]
    for path in required:
        assert path.exists()
    with (root / "metrics" / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    for field in (
        "lambda_c_fit",
        "beta_over_nu_fit",
        "inv_nu_fit",
        "collapse_objective",
        "crossing_lambda_c_mean",
        "bootstrap_lambda_c_lo",
        "bootstrap_lambda_c_hi",
    ):
        assert field in row
