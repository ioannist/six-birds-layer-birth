"""Compact finite-size scaling analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def susceptibility_from_samples(values: list[float] | np.ndarray, system_size: int) -> tuple[float, dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("values must be a finite 1D sample")
    if int(system_size) != system_size or system_size <= 0:
        raise ValueError("system_size must be a positive integer")
    mean = float(np.mean(arr))
    var = float(np.var(arr, ddof=0))
    chi = float(system_size * var)
    return chi, {
        "count": int(arr.size),
        "mean": mean,
        "variance": var,
        "system_size": int(system_size),
    }


def binder_like_cumulant(values: list[float] | np.ndarray, eps: float = 1e-12) -> tuple[float, dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("values must be a finite 1D sample")
    m2 = float(np.mean(arr**2))
    m4 = float(np.mean(arr**4))
    denom = 3.0 * (m2**2)
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        u4 = 0.0
    else:
        # The moment ratio is invariant under rescaling. An absolute denominator
        # cutoff incorrectly made every sufficiently small observable zero.
        normalized = arr / scale
        m2_scaled = float(np.mean(normalized**2))
        m4_scaled = float(np.mean(normalized**4))
        u4 = float(1.0 - m4_scaled / (3.0 * m2_scaled**2))
    return u4, {"m2": m2, "m4": m4, "denominator": denom,
                "defined": scale > 0.0, "zero_sample_fallback": scale == 0.0}


def group_order_observations(
    rows: list[dict[str, Any]],
    size_key: str = "size",
    lambda_key: str = "closure_strength_lambda",
    value_key: str = "order_value",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], list[float]] = {}
    for row in rows:
        size = int(row[size_key])
        lam = float(row[lambda_key])
        val = float(row[value_key])
        if size != float(row[size_key]) or size <= 0 or not np.isfinite(lam) or not np.isfinite(val):
            raise ValueError("observations require positive integer sizes and finite lambda/value")
        grouped.setdefault((size, lam), []).append(val)

    out: list[dict[str, Any]] = []
    for (size, lam), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.asarray(vals, dtype=np.float64)
        chi, _ = susceptibility_from_samples(arr, size)
        binder, _ = binder_like_cumulant(arr)
        out.append(
            {
                "size": int(size),
                "closure_strength_lambda": float(lam),
                "n_replicates": int(arr.size),
                "order_mean": float(np.mean(arr)),
                "susceptibility": float(chi),
                "binder_like_cumulant": float(binder),
            }
        )
    return out


def estimate_pairwise_crossings(
    group_rows: list[dict[str, Any]],
    size_key: str = "size",
    lambda_key: str = "closure_strength_lambda",
    binder_key: str = "binder_like_cumulant",
) -> dict[str, Any]:
    sizes = sorted({int(r[size_key]) for r in group_rows})
    by_size: dict[int, dict[float, float]] = {}
    for size in sizes:
        by_size[size] = {
            float(r[lambda_key]): float(r[binder_key])
            for r in group_rows
            if int(r[size_key]) == size
        }
    pairwise: list[dict[str, Any]] = []
    tol = 1e-12
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            s1, s2 = sizes[i], sizes[j]
            shared = sorted(set(by_size[s1]).intersection(by_size[s2]))
            if len(shared) < 2:
                continue
            deltas = [by_size[s1][lam] - by_size[s2][lam] for lam in shared]
            if all(abs(d) <= tol for d in deltas):
                # Curves are effectively identical on sampled grid: no unique crossing.
                continue
            for k in range(len(shared) - 1):
                l0, l1 = shared[k], shared[k + 1]
                d0, d1 = deltas[k], deltas[k + 1]
                crossing = None
                if abs(d0) <= tol and abs(d1) <= tol:
                    continue
                if abs(d0) <= tol:
                    crossing = l0
                elif d0 * d1 < 0.0 and abs(d1 - d0) > tol:
                    crossing = l0 + (0.0 - d0) * (l1 - l0) / (d1 - d0)
                elif abs(d1) <= tol:
                    crossing = l1
                if crossing is not None:
                    pairwise.append(
                        {
                            "size_pair": [s1, s2],
                            "lambda_interval": [l0, l1],
                            "crossing_lambda": float(crossing),
                        }
                    )
                    break
    crossing_values = [x["crossing_lambda"] for x in pairwise]
    return {
        "pairwise_crossings": pairwise,
        "crossing_lambda_c_mean": None if not crossing_values else float(np.mean(crossing_values)),
        "crossing_count": len(crossing_values),
        "size_pairs": [x["size_pair"] for x in pairwise],
    }


def collapse_objective(
    group_rows: list[dict[str, Any]],
    lambda_c: float,
    beta_over_nu: float,
    inv_nu: float,
    size_key: str = "size",
    lambda_key: str = "closure_strength_lambda",
    order_key: str = "order_mean",
    n_bins: int = 12,
) -> float:
    """Declared heuristic score, not a pure finite-size curve-collapse loss.

    The score adds raw-lambda consistency and a weighted critical-slope anchor
    to pairwise interpolation error. Its minimizer is conditional on those
    penalties and the finite search grid; it is not an exponent certificate.
    """
    if not all(np.isfinite(v) for v in (lambda_c, beta_over_nu, inv_nu)):
        raise ValueError("collapse parameters must be finite")
    if int(n_bins) != n_bins or n_bins < 2:
        raise ValueError("collapse evaluation requires at least two points")
    transformed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    observed = set()
    for row in group_rows:
        L = int(row[size_key])
        lam = float(row[lambda_key])
        m = float(row[order_key])
        if L <= 0 or L != float(row[size_key]) or not np.isfinite(lam) or not np.isfinite(m):
            raise ValueError("collapse requires positive integer sizes and finite measurements")
        if (L, lam) in observed:
            raise ValueError("collapse requires one grouped observation per size and lambda")
        observed.add((L, lam))
        x = (lam - lambda_c) * (L ** inv_nu)
        y = m * (L ** beta_over_nu)
        if L not in transformed:
            transformed[L] = ([], [])
        transformed[L][0].append(x)  # type: ignore[index]
        transformed[L][1].append(y)  # type: ignore[index]
    if len(transformed) < 2:
        return float("inf")

    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L, (xs_list, ys_list) in transformed.items():
        xs = np.asarray(xs_list, dtype=np.float64)
        ys = np.asarray(ys_list, dtype=np.float64)
        order = np.argsort(xs)
        curves[L] = (xs[order], ys[order])

    sizes = sorted(curves)
    pair_errors: list[float] = []
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            s1, s2 = sizes[i], sizes[j]
            x1, y1 = curves[s1]
            x2, y2 = curves[s2]
            lo = max(float(np.min(x1)), float(np.min(x2)))
            hi = min(float(np.max(x1)), float(np.max(x2)))
            if not hi > lo:
                continue
            n_eval = max(8, int(n_bins))
            x_eval = np.linspace(lo, hi, n_eval)
            y1i = np.interp(x_eval, x1, y1)
            y2i = np.interp(x_eval, x2, y2)
            mse = float(np.mean((y1i - y2i) ** 2))
            pair_errors.append(mse)
    if not pair_errors:
        return float("inf")
    pair_term = float(np.mean(pair_errors))

    # Additional cross-size consistency term at shared raw lambda values.
    by_lambda: dict[float, list[float]] = {}
    for row in group_rows:
        L = float(row[size_key])
        lam = float(row[lambda_key])
        m = float(row[order_key])
        y = m * (L ** beta_over_nu)
        by_lambda.setdefault(lam, []).append(y)
    lambda_vars = [float(np.var(vals, ddof=0)) for vals in by_lambda.values() if len(vals) >= 2]
    consistency_term = 0.0 if not lambda_vars else float(np.mean(lambda_vars))

    # Anchor beta/nu using size scaling at lambda nearest candidate lambda_c.
    raw_lams = sorted(by_lambda.keys())
    lam_star = min(raw_lams, key=lambda z: abs(z - lambda_c))
    lam_rows = [
        (float(row[size_key]), float(row[order_key]))
        for row in group_rows
        if float(row[lambda_key]) == lam_star and float(row[order_key]) > 0.0
    ]
    slope_term = 0.0
    if len(lam_rows) >= 2:
        xs = np.log(np.asarray([r[0] for r in lam_rows], dtype=np.float64))
        ys = np.log(np.asarray([r[1] for r in lam_rows], dtype=np.float64))
        slope = float(np.polyfit(xs, ys, 1)[0])
        slope_term = float((slope + beta_over_nu) ** 2)
    return float(pair_term + consistency_term + 10.0 * slope_term)


def grid_search_collapse_fit(
    group_rows: list[dict[str, Any]],
    lambda_c_grid: list[float],
    beta_over_nu_grid: list[float],
    inv_nu_grid: list[float],
    n_bins: int = 12,
    top_k: int = 5,
) -> dict[str, Any]:
    if any(not grid or not np.isfinite(grid).all() for grid in (lambda_c_grid, beta_over_nu_grid, inv_nu_grid)):
        raise ValueError("fit grids must be nonempty and finite")
    ranked: list[dict[str, Any]] = []
    for lc in lambda_c_grid:
        for bon in beta_over_nu_grid:
            for inv in inv_nu_grid:
                obj = collapse_objective(
                    group_rows,
                    lambda_c=float(lc),
                    beta_over_nu=float(bon),
                    inv_nu=float(inv),
                    n_bins=n_bins,
                )
                ranked.append(
                    {
                        "lambda_c": float(lc),
                        "beta_over_nu": float(bon),
                        "inv_nu": float(inv),
                        "objective": float(obj),
                    }
                )
    ranked.sort(key=lambda r: r["objective"])
    best = ranked[0]
    if not np.isfinite(best["objective"]):
        raise ValueError("no comparable curves on the fit grid")
    return {
        "best_fit": best,
        "objective_kind": "pairwise_mse_plus_raw_lambda_variance_plus_10_times_slope_penalty",
        "fit_scope": "finite_grid_heuristic; no universal exponent certificate",
        "top_candidates": ranked[: max(1, int(top_k))],
        "num_combinations": len(ranked),
    }


def _resample_observations(raw_rows: list[dict[str, Any]], rng: np.random.Generator,
                           size_key: str, lambda_key: str, value_key: str) -> tuple[list[dict[str, Any]], str]:
    """Resample whole seed trajectories when seeds identify shared substrates."""
    has_seed = [r.get("seed") not in (None, "") for r in raw_rows]
    if any(has_seed) and not all(has_seed):
        raise ValueError("bootstrap cannot mix identified seed clusters and unidentified observations")
    groups: dict[Any, list[dict[str, Any]]] = {}
    boot = []
    if all(has_seed) and raw_rows:
        for r in raw_rows:
            groups.setdefault((int(r[size_key]), str(r["seed"])), []).append(r)
        sizes = sorted({size for size, _ in groups})
        for size in sizes:
            clusters = [rows for (n, _), rows in groups.items() if n == size]
            grids = [{float(r[lambda_key]) for r in rows} for rows in clusters]
            if any(len(rows) != len(grid) for rows, grid in zip(clusters, grids)) or any(grid != grids[0] for grid in grids):
                raise ValueError("seed trajectories require the same complete lambda grid at each size")
            for j in rng.integers(len(clusters), size=len(clusters)):
                boot.extend(dict(r) for r in clusters[j])
        return boot, "seed_trajectory_within_size"
    for r in raw_rows:
        groups.setdefault((int(r[size_key]), float(r[lambda_key])), []).append(r)
    for rows in groups.values():
        for j in rng.integers(len(rows), size=len(rows)):
            boot.append(dict(rows[j]))
    return boot, "independent_observations_within_size_lambda"


def bootstrap_collapse_fit(
    raw_rows: list[dict[str, Any]],
    fit_config: dict[str, Any],
    n_boot: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    if int(n_boot) != n_boot or n_boot <= 0 or not raw_rows:
        raise ValueError("bootstrap requires observations and a positive integer replicate count")
    rng = np.random.default_rng(seed)
    size_key = fit_config.get("size_key", "size")
    lambda_key = fit_config.get("lambda_key", "closure_strength_lambda")
    value_key = fit_config.get("value_key", "order_value")
    # Validate observations before resampling, including finite values.
    group_order_observations(raw_rows, size_key=size_key, lambda_key=lambda_key, value_key=value_key)
    successful = 0
    lambda_cs: list[float] = []
    beta_over_nus: list[float] = []
    inv_nus: list[float] = []
    for _ in range(int(n_boot)):
        boot_rows, resampling_unit = _resample_observations(raw_rows, rng, size_key, lambda_key, value_key)
        grouped = group_order_observations(boot_rows, size_key=size_key, lambda_key=lambda_key, value_key=value_key)
        fit = grid_search_collapse_fit(
            grouped,
            lambda_c_grid=list(fit_config["lambda_c_grid"]),
            beta_over_nu_grid=list(fit_config["beta_over_nu_grid"]),
            inv_nu_grid=list(fit_config["inv_nu_grid"]),
            n_bins=int(fit_config.get("n_bins", 12)),
            top_k=3,
        )["best_fit"]
        if not np.isfinite(fit["objective"]):
            continue
        successful += 1
        lambda_cs.append(float(fit["lambda_c"]))
        beta_over_nus.append(float(fit["beta_over_nu"]))
        inv_nus.append(float(fit["inv_nu"]))
    if successful == 0:
        raise ValueError("bootstrap had zero successful fits")
    pct_lo, pct_hi = fit_config.get("ci_percentiles", [5, 95])
    if not 0 <= pct_lo < pct_hi <= 100:
        raise ValueError("bootstrap percentiles require 0 <= low < high <= 100")

    def ci(arr: list[float]) -> list[float]:
        vals = np.asarray(arr, dtype=np.float64)
        return [float(np.percentile(vals, pct_lo)), float(np.percentile(vals, pct_hi))]

    return {
        "bootstrap_samples": {
            "lambda_c": lambda_cs,
            "beta_over_nu": beta_over_nus,
            "inv_nu": inv_nus,
        },
        "ci_percentiles": [pct_lo, pct_hi],
        "ci": {
            "lambda_c": ci(lambda_cs),
            "beta_over_nu": ci(beta_over_nus),
            "inv_nu": ci(inv_nus),
        },
        "successful_bootstrap_count": int(successful),
        "requested_bootstrap_count": int(n_boot),
        "seed": int(seed),
        "resampling_unit": resampling_unit,
        "uncertainty_scope": "conditional on declared sampling unit, objective and grid; singleton groups have no sampling variability",
    }


def format_scaling_report(
    best_fit: dict[str, Any],
    bootstrap_result: dict[str, Any],
    crossing_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "best_fit": best_fit,
        "objective_kind": "pairwise_mse_plus_raw_lambda_variance_plus_10_times_slope_penalty",
        "fit_scope": "finite_grid_heuristic; no universal exponent certificate",
        "bootstrap": {
            "ci_percentiles": bootstrap_result["ci_percentiles"],
            "ci": bootstrap_result["ci"],
            "successful_bootstrap_count": bootstrap_result["successful_bootstrap_count"],
            "requested_bootstrap_count": bootstrap_result["requested_bootstrap_count"],
            "seed": bootstrap_result["seed"],
            "resampling_unit": bootstrap_result.get("resampling_unit", "not_resampled"),
            "uncertainty_scope": bootstrap_result.get("uncertainty_scope", "no sampling interval estimated"),
        },
        "crossings": crossing_summary,
    }


@dataclass
class MockFssParams:
    sizes: list[int]
    lambda_grid: list[float]
    lambda_c_true: float
    beta_over_nu_true: float
    inv_nu_true: float
    replicate_count: int
    noise_sigma: float
    seed: int


def generate_mock_observations(params: MockFssParams) -> list[dict[str, Any]]:
    rng = np.random.default_rng(params.seed)
    rows: list[dict[str, Any]] = []
    for size in params.sizes:
        for lam in params.lambda_grid:
            x = (lam - params.lambda_c_true) * (size ** params.inv_nu_true)
            f_x = 1.0 / (1.0 + np.exp(-2.0 * x))
            mean = (size ** (-params.beta_over_nu_true)) * f_x
            for rep in range(params.replicate_count):
                noise = rng.normal(0.0, params.noise_sigma)
                val = float(np.clip(mean + noise, 0.0, 1.0))
                rows.append(
                    {
                        "size": int(size),
                        "closure_strength_lambda": float(lam),
                        "order_value": val,
                        "replicate_id": int(rep),
                    }
                )
    return rows
