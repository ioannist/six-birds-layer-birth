"""LB-15 class-I scaling campaign orchestration."""

from __future__ import annotations

import binascii
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import artifact_is_current, cache_reuse_allowed, computation_hash, implementation_fingerprint
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import pushforward_matrix
from .pilots import run_all_lb12_pilots
from .protocols import resolve_run_settings
from .robustness import run_robustness_pilots
from .scaling import (
    bootstrap_collapse_fit,
    estimate_pairwise_crossings,
    format_scaling_report,
    grid_search_collapse_fit,
    group_order_observations,
)
from .sweep import apply_closure_strength_control
from .substrates import build_substrate_family


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    return computation_hash(payload, length)


def _git_code_version(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"git_commit": commit or "unknown", "git_dirty": bool(dirty), "implementation_sha256": implementation_fingerprint(), "cache_reuse_disabled": not cache_reuse_allowed(True)}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": False, "implementation_sha256": implementation_fingerprint(), "cache_reuse_disabled": not cache_reuse_allowed(True)}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_png(path: Path, image: np.ndarray) -> None:
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))
    comp = zlib.compress(raw, level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", binascii.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1]:
            image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _line_plot(path: Path, xvals: list[float], series: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    plotted: list[float] = []
    for label, arr in series.items():
        n = min(len(xvals), len(arr))
        if n == 0:
            continue
        xs = np.asarray(xvals[:n], dtype=np.float64)
        ys = np.asarray(arr[:n], dtype=np.float64)
        plotted.extend(ys.tolist())
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=5.5, label=label)

    if not plotted:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    ymin, ymax = float(np.min(plotted)), float(np.max(plotted))
    if np.isclose(ymin, ymax):
        pad = 1.0 if np.isclose(ymin, round(ymin)) else max(1e-3, 0.15 * max(abs(ymin), 1.0))
        ax.set_ylim(ymin - pad, ymax + pad)
    elif ymin >= 0.0 and ymax <= 1.0 and all(v in {0.0, 1.0} for v in {round(v, 6) for v in plotted}):
        ax.set_ylim(-0.05, 1.05)

    stem = path.stem
    xlabel = r"Closure strength $\lambda$"
    ylabel = "Measured value"
    if "size" in stem:
        xlabel = "System size"
    if "affinity" in stem:
        ylabel = "Affinity"
    elif "strict_hits" in stem:
        ylabel = "Strict hit"
    elif "shift" in stem:
        ylabel = "Boundary shift"
    elif "state" in stem:
        ylabel = "State indicator"
    elif "susceptibility" in stem:
        ylabel = "Susceptibility"
    elif "binder" in stem:
        ylabel = "Binder-like cumulant"
    elif "delta_mobj" in stem:
        ylabel = r"$\Delta M_{\mathrm{obj}}$"

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    if len(series) <= 6:
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _resolve_primary_kernel(panel: dict[str, Any], size: int) -> dict[str, Any]:
    params = {
        "n_blocks": int(panel["n_blocks"]),
        "block_size": int(size // int(panel["n_blocks"])),
        "intra_block_weight": float(panel["intra_block_weight"]),
        "inter_block_weight": float(panel["inter_block_weight"]),
        "self_weight": float(panel["self_weight"]),
    }
    return build_substrate_family(panel["family_name"], **params)


def _resolve_shadow_kernel(panel: dict[str, Any], size: int, seed: int) -> dict[str, Any]:
    params = {
        "n_blocks": int(panel["n_blocks"]),
        "block_size": int(size // int(panel["n_blocks"])),
        "intra_scale": float(panel["intra_scale"]),
        "inter_scale": float(panel["inter_scale"]),
        "diagonal_bias": float(panel["diagonal_bias"]),
        "seed": int(seed),
    }
    return build_substrate_family(panel["family_name"], **params)


def build_class_i_panel_rows(config: dict[str, Any], runs_dir: Path, use_cache: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    primary = config["primary_panel"]
    shadow = config["shadow_panel"]
    primary_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    executed = 0
    cached = 0

    for size in primary["sizes"]:
        substrate = _resolve_primary_kernel(primary, int(size))
        p_base = np.asarray(substrate["P"], dtype=np.float64)
        lens = np.asarray(substrate["block_lens"], dtype=np.int64)
        q = pushforward_matrix(lens, int(np.max(lens)) + 1)
        lift, _ = build_lift_family(primary["lift_name"], f=lens, k=int(np.max(lens)) + 1)
        for lam in primary["lambda_grid"]:
            spec = {"panel": "primary", "panel_config": primary, "size": int(size), "lambda": float(lam)}
            run_id = f"cmp_{_stable_hash(spec)}"
            row_file = runs_dir / f"{run_id}.json"
            if cache_reuse_allowed(use_cache) and row_file.exists():
                row = json.loads(row_file.read_text(encoding="utf-8"))
                row["cache_status"] = "cached"
                cached += 1
            else:
                p = apply_closure_strength_control(
                    p_base,
                    closure_strength_lambda=float(lam),
                    Q_f=q,
                    U_f=np.asarray(lift),
                    mode=primary["control_application_name"],
                )
                tau_cfg = {
                    "run_id": run_id,
                    "tau_protocol": dict(primary["tau_protocol"]),
                    "control": {"closure_strength_lambda": float(lam)},
                }
                tau = int(resolve_run_settings(tau_cfg, P=p)["resolved_tau"])
                bundle = default_metric_bundle(p, lens, tau=tau, lift_name=primary["lift_name"], U_f=lift)
                row = {
                    "panel": "primary",
                    "size": int(size),
                    "closure_strength_lambda": float(lam),
                    "seed": "",
                    "run_id": run_id,
                    "analysis_k": int(bundle["analysis_k"]),
                    "order_value": observed_float(bundle["objecthood_order"]),
                    "closure_error": observed_float(bundle["closure_error"]),
                    "staging_gap": observed_float(bundle["staging_gap"]),
                    "affinity": observed_float(bundle["affinity"]),
                    "cache_status": "executed",
                }
                row_file.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
                executed += 1
            primary_rows.append(row)

    for size in shadow["sizes"]:
        for seed in shadow["seeds"]:
            substrate = _resolve_shadow_kernel(shadow, int(size), int(seed))
            p_base = np.asarray(substrate["P"], dtype=np.float64)
            lens = np.asarray(substrate["block_lens"], dtype=np.int64)
            q = pushforward_matrix(lens, int(np.max(lens)) + 1)
            lift, _ = build_lift_family(shadow["lift_name"], f=lens, k=int(np.max(lens)) + 1)
            for lam in shadow["lambda_grid"]:
                spec = {"panel": "shadow", "panel_config": shadow, "size": int(size), "lambda": float(lam), "seed": int(seed)}
                run_id = f"cmp_{_stable_hash(spec)}"
                row_file = runs_dir / f"{run_id}.json"
                if cache_reuse_allowed(use_cache) and row_file.exists():
                    row = json.loads(row_file.read_text(encoding="utf-8"))
                    row["cache_status"] = "cached"
                    cached += 1
                else:
                    p = apply_closure_strength_control(
                        p_base,
                        closure_strength_lambda=float(lam),
                        Q_f=q,
                        U_f=np.asarray(lift),
                        mode=shadow["control_application_name"],
                    )
                    tau_cfg = {
                        "run_id": run_id,
                        "tau_protocol": dict(shadow["tau_protocol"]),
                        "control": {"closure_strength_lambda": float(lam)},
                    }
                    tau = int(resolve_run_settings(tau_cfg, P=p)["resolved_tau"])
                    bundle = default_metric_bundle(p, lens, tau=tau, lift_name=shadow["lift_name"], U_f=lift)
                    row = {
                        "panel": "shadow",
                        "size": int(size),
                        "closure_strength_lambda": float(lam),
                        "seed": int(seed),
                        "run_id": run_id,
                        "analysis_k": int(bundle["analysis_k"]),
                        "order_value": observed_float(bundle["objecthood_order"]),
                        "closure_error": observed_float(bundle["closure_error"]),
                        "staging_gap": observed_float(bundle["staging_gap"]),
                        "affinity": observed_float(bundle["affinity"]),
                        "cache_status": "executed",
                    }
                    row_file.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
                    executed += 1
                shadow_rows.append(row)
    return primary_rows, shadow_rows, executed, cached


def _group_primary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["size"]), float(row["closure_strength_lambda"]))
        grouped.setdefault(key, []).append(row)
    out = []
    for (size, lam), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append(
            {
                "size": size,
                "closure_strength_lambda": lam,
                "analysis_k": int(vals[0]["analysis_k"]),
                "order_mean": float(np.mean([float(v["order_value"]) for v in vals])),
                "closure_error_mean": float(np.mean([observed_float(v["closure_error"]) for v in vals])),
                "staging_gap_mean": float(np.mean([observed_float(v["staging_gap"]) for v in vals])),
                "affinity_mean": float(np.mean([observed_float(v["affinity"]) for v in vals])),
                "n_replicates": len(vals),
            }
        )
    return out


def _group_shadow(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_order = group_order_observations(rows)
    grouped_other: dict[tuple[int, float], dict[str, float]] = {}
    for row in rows:
        key = (int(row["size"]), float(row["closure_strength_lambda"]))
        grouped_other.setdefault(key, {"closure_error": [], "staging_gap": [], "affinity": []})  # type: ignore[assignment]
        grouped_other[key]["closure_error"].append(observed_float(row["closure_error"]))  # type: ignore[index]
        grouped_other[key]["staging_gap"].append(observed_float(row["staging_gap"]))  # type: ignore[index]
        grouped_other[key]["affinity"].append(observed_float(row["affinity"]))  # type: ignore[index]
    out = []
    for row in grouped_order:
        key = (int(row["size"]), float(row["closure_strength_lambda"]))
        aux = grouped_other[key]
        out.append(
            {
                "size": int(row["size"]),
                "closure_strength_lambda": float(row["closure_strength_lambda"]),
                "n_replicates": int(row["n_replicates"]),
                "order_mean": float(row["order_mean"]),
                "susceptibility": float(row["susceptibility"]),
                "binder_like_cumulant": float(row["binder_like_cumulant"]),
                "closure_error_mean": float(np.mean(aux["closure_error"])),  # type: ignore[arg-type]
                "staging_gap_mean": float(np.mean(aux["staging_gap"])),  # type: ignore[arg-type]
                "affinity_mean": float(np.mean(aux["affinity"])),  # type: ignore[arg-type]
            }
        )
    return out


def fit_class_i_collapse(primary_group_rows: list[dict[str, Any]], fit_config: dict[str, Any]) -> dict[str, Any]:
    fit = grid_search_collapse_fit(
        primary_group_rows,
        lambda_c_grid=[float(x) for x in fit_config["lambda_c_grid"]],
        beta_over_nu_grid=[float(x) for x in fit_config["beta_over_nu_grid"]],
        inv_nu_grid=[float(x) for x in fit_config["inv_nu_grid"]],
        n_bins=int(fit_config.get("collapse_bins", 12)),
        top_k=6,
    )
    raw_rows = [
        {
            "size": int(r["size"]),
            "closure_strength_lambda": float(r["closure_strength_lambda"]),
            "order_value": float(r["order_mean"]),
        }
        for r in primary_group_rows
    ]
    boot = bootstrap_collapse_fit(
        raw_rows,
        {
            "lambda_c_grid": fit_config["lambda_c_grid"],
            "beta_over_nu_grid": fit_config["beta_over_nu_grid"],
            "inv_nu_grid": fit_config["inv_nu_grid"],
            "n_bins": int(fit_config.get("collapse_bins", 12)),
            "ci_percentiles": list(fit_config.get("ci_percentiles", [5, 95])),
        },
        n_boot=int(fit_config.get("bootstrap_replicates", 50)),
        seed=2026,
    )
    return format_scaling_report(fit["best_fit"], boot, {"pairwise_crossings": [], "crossing_count": 0})


def analyze_shadow_crossings(shadow_raw_rows: list[dict[str, Any]], fit_config: dict[str, Any]) -> dict[str, Any]:
    shadow_group = _group_shadow(shadow_raw_rows)
    return estimate_pairwise_crossings(shadow_group)


def write_class_i_plots(primary_group_rows: list[dict[str, Any]], shadow_group_rows: list[dict[str, Any]], fit_result: dict[str, Any], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_size_primary: dict[int, list[dict[str, Any]]] = {}
    for row in primary_group_rows:
        by_size_primary.setdefault(int(row["size"]), []).append(row)
    lambdas = sorted({float(r["closure_strength_lambda"]) for r in primary_group_rows})
    order_ce_series: dict[str, list[float]] = {}
    for size, rows in sorted(by_size_primary.items()):
        rmap = {float(r["closure_strength_lambda"]): r for r in rows}
        order_ce_series[f"L{size}:M"] = [float(rmap[lam]["order_mean"]) for lam in lambdas]
        order_ce_series[f"L{size}:CE"] = [observed_float(rmap[lam]["closure_error_mean"]) for lam in lambdas]
    p1 = output_dir / "order_ce_vs_lambda_by_size.png"
    _line_plot(p1, lambdas, order_ce_series)

    best = fit_result["best_fit"]
    collapse_series: dict[str, list[float]] = {}
    x_ref = sorted(
        {
            (float(r["closure_strength_lambda"]) - float(best["lambda_c"])) * (float(r["size"]) ** float(best["inv_nu"]))
            for r in primary_group_rows
        }
    )
    for size, rows in sorted(by_size_primary.items()):
        x_map = {
            (float(r["closure_strength_lambda"]) - float(best["lambda_c"])) * (float(r["size"]) ** float(best["inv_nu"])): float(r["order_mean"]) * (float(r["size"]) ** float(best["beta_over_nu"]))
            for r in rows
        }
        vals = []
        for x in x_ref:
            if x in x_map:
                vals.append(x_map[x])
            else:
                nearest = min(x_map.keys(), key=lambda z: abs(z - x))
                vals.append(x_map[nearest])
        collapse_series[f"L{size}"] = vals
    p2 = output_dir / "collapse_best_fit.png"
    _line_plot(p2, x_ref, collapse_series)

    by_size_shadow: dict[int, list[dict[str, Any]]] = {}
    for row in shadow_group_rows:
        by_size_shadow.setdefault(int(row["size"]), []).append(row)
    lambdas_shadow = sorted({float(r["closure_strength_lambda"]) for r in shadow_group_rows})
    binder_series: dict[str, list[float]] = {}
    susc_series: dict[str, list[float]] = {}
    for size, rows in sorted(by_size_shadow.items()):
        rmap = {float(r["closure_strength_lambda"]): r for r in rows}
        binder_series[f"L{size}"] = [float(rmap[lam]["binder_like_cumulant"]) for lam in lambdas_shadow]
        susc_series[f"L{size}"] = [float(rmap[lam]["susceptibility"]) for lam in lambdas_shadow]
    p3 = output_dir / "shadow_binder_crossings.png"
    p4 = output_dir / "shadow_susceptibility_vs_lambda.png"
    _line_plot(p3, lambdas_shadow, binder_series)
    _line_plot(p4, lambdas_shadow, susc_series)
    return [str(p1), str(p2), str(p3), str(p4)]


def format_class_i_campaign_summary(primary_fit: dict[str, Any], shadow_summary: dict[str, Any], campaign_diagnosis: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_fit": primary_fit,
        "shadow_crossings": shadow_summary,
        "campaign_diagnosis": campaign_diagnosis,
    }


def run_class_i_scaling_campaign(config_path_or_obj: str | Path | dict[str, Any], output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        config = config_path_or_obj
    else:
        config = json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))
    if output_root is None:
        output_root = _repo_root() / "results" / "campaigns"
    else:
        output_root = Path(output_root)
    artifact_root = output_root / config["artifact_subdir"]
    dirs = {
        "config": artifact_root / "config",
        "seeds": artifact_root / "seeds",
        "metrics": artifact_root / "metrics",
        "notes": artifact_root / "notes",
        "env": artifact_root / "env",
        "analysis": artifact_root / "analysis",
        "plots": artifact_root / "plots",
        "runs": artifact_root / "runs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    primary_raw, shadow_raw, executed_count, cached_count = build_class_i_panel_rows(config, dirs["runs"], use_cache=use_cache)

    primary_group = _group_primary(primary_raw)
    shadow_group = _group_shadow(shadow_raw)
    fit_report = fit_class_i_collapse(primary_group, config["fit_grid"])
    shadow_cross = analyze_shadow_crossings(shadow_raw, config["fit_grid"])

    best = fit_report["best_fit"]
    lam_min = min(float(x) for x in config["primary_panel"]["lambda_grid"])
    lam_max = max(float(x) for x in config["primary_panel"]["lambda_grid"])
    crossings_ok = shadow_cross["crossing_count"] > 0 and shadow_cross["crossing_lambda_c_mean"] is not None and (0.70 <= float(shadow_cross["crossing_lambda_c_mean"]) <= 1.00)
    candidate = bool(
        np.isfinite(float(best["objective"]))
        and lam_min <= float(best["lambda_c"]) <= lam_max
        and (crossings_ok or shadow_cross["crossing_count"] == 0)
    )
    go_no_go = "go" if candidate else "no-go"
    diagnosis = {
        "candidate_collapse_obtained": candidate,
        "go_no_go": go_no_go,
        "crossings_ok": crossings_ok,
        "crossing_count": int(shadow_cross["crossing_count"]),
    }
    if not candidate:
        failure_report = {
            "best_fit_parameters": best,
            "collapse_objective": best["objective"],
            "shadow_crossings": shadow_cross,
            "deterministic_primary_degeneracy_possible": True,
            "diagnosis": "credible candidate-collapse condition not met under current diagnostics",
        }
        (dirs["analysis"] / "failure_report.json").write_text(scientific_dumps(failure_report, indent=2) + "\n", encoding="utf-8")

    summary = format_class_i_campaign_summary(fit_report, shadow_cross, diagnosis)
    (dirs["analysis"] / "fit_summary.json").write_text(scientific_dumps(fit_report, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "shadow_crossings.json").write_text(scientific_dumps(shadow_cross, indent=2) + "\n", encoding="utf-8")

    _write_csv(
        dirs["analysis"] / "primary_raw_metrics.csv",
        primary_raw,
        ["panel", "size", "closure_strength_lambda", "seed", "run_id", "analysis_k", "order_value", "closure_error", "staging_gap", "affinity", "cache_status"],
    )
    _write_csv(
        dirs["analysis"] / "primary_group_summary.csv",
        primary_group,
        ["size", "closure_strength_lambda", "analysis_k", "order_mean", "closure_error_mean", "staging_gap_mean", "affinity_mean", "n_replicates"],
    )
    _write_csv(
        dirs["analysis"] / "shadow_raw_metrics.csv",
        shadow_raw,
        ["panel", "size", "closure_strength_lambda", "seed", "run_id", "analysis_k", "order_value", "closure_error", "staging_gap", "affinity", "cache_status"],
    )
    _write_csv(
        dirs["analysis"] / "shadow_group_summary.csv",
        shadow_group,
        ["size", "closure_strength_lambda", "n_replicates", "order_mean", "susceptibility", "binder_like_cumulant", "closure_error_mean", "staging_gap_mean", "affinity_mean"],
    )

    write_class_i_plots(primary_group, shadow_group, fit_report, dirs["plots"])

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    seeds = dirs["seeds"] / "seeds.json"
    seeds.write_text(scientific_dumps(config["shadow_panel"]["seeds"], indent=2) + "\n", encoding="utf-8")
    metrics_csv = dirs["metrics"] / "metrics.csv"
    _write_csv(
        metrics_csv,
        [
            {
                "campaign_name": config["campaign_id"],
                "primary_size_count": len(config["primary_panel"]["sizes"]),
                "shadow_replicate_count": len(config["shadow_panel"]["seeds"]),
                "lambda_c_fit": best["lambda_c"],
                "beta_over_nu_fit": best["beta_over_nu"],
                "inv_nu_fit": best["inv_nu"],
                "collapse_objective": best["objective"],
                "shadow_crossing_lambda_c_mean": shadow_cross["crossing_lambda_c_mean"],
                "shadow_crossing_count": shadow_cross["crossing_count"],
                "go_no_go": go_no_go,
                "executed_count": executed_count,
                "cached_count": cached_count,
            }
        ],
        [
            "campaign_name",
            "primary_size_count",
            "shadow_replicate_count",
            "lambda_c_fit",
            "beta_over_nu_fit",
            "inv_nu_fit",
            "collapse_objective",
            "shadow_crossing_lambda_c_mean",
            "shadow_crossing_count",
            "go_no_go",
            "executed_count",
            "cached_count",
        ],
    )
    notes = dirs["notes"] / "findings.md"
    notes.write_text(
        "\n".join(
            [
                "# Class-I campaign findings",
                "",
                f"- go_no_go: `{go_no_go}`",
                f"- candidate_collapse_obtained: `{candidate}`",
                f"- best_fit: `{best}`",
                f"- shadow_crossings: `{shadow_cross['crossing_count']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = dirs["env"] / "environment.json"
    env.write_text(
        scientific_dumps({"python": "unknown", "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    schema = load_schema(_repo_root() / "configs" / "result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_i_equilibrium_scaling_campaign",
        "bundle_id": config["campaign_id"],
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(_repo_root()),
        "config_snapshot_path": str(config_snapshot.relative_to(artifact_root)),
        "seed_list_path": str(seeds.relative_to(artifact_root)),
        "metrics_table_path": str(metrics_csv.relative_to(artifact_root)),
        "plots_dir_path": str(dirs["plots"].relative_to(artifact_root)),
        "notes_file_path": str(notes.relative_to(artifact_root)),
        "environment_snapshot_path": str(env.relative_to(artifact_root)),
    }
    validate_manifest(manifest, schema)
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    findings_note = _repo_root() / config["findings_note_path"]
    findings_note.parent.mkdir(parents=True, exist_ok=True)
    findings_note.write_text(
        "\n".join(
            [
                "# LB-15 class-I scaling campaign",
                "",
                f"- campaign config path: `{config['config_path_hint']}`",
                f"- artifact root: `{artifact_root}`",
                "- primary panel: reversible_block_family sizes [8,16,32,64]",
                "- shadow panel: metastable_block_family sizes [8,16,32,64], seeds [7,11,13,17,19,23]",
                f"- fitted parameters: `{best}`",
                f"- collapse diagnostics objective: `{best['objective']}`",
                f"- shadow crossing summary: `{shadow_cross}`",
                f"- candidate collapse obtained: `{'yes' if candidate else 'no'}`",
                f"- recommend proceed to class-II campaign: `{go_no_go}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "campaign_name": config["campaign_id"],
        "primary_size_count": len(config["primary_panel"]["sizes"]),
        "shadow_replicate_count": len(config["shadow_panel"]["seeds"]),
        "lambda_c_fit": best["lambda_c"],
        "beta_over_nu_fit": best["beta_over_nu"],
        "inv_nu_fit": best["inv_nu"],
        "collapse_objective": best["objective"],
        "go_no_go": go_no_go,
        "artifact_root": str(artifact_root),
        "executed_count": executed_count,
        "cached_count": cached_count,
        "failure_report_needed": not candidate,
    }


def run_class_i_campaign_driver(output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    root = _repo_root()
    # Restore prior roots if missing.
    subprocess.run([sys.executable, str(root / "scripts" / "smoke_scaling_analysis.py")], check=True)
    run_all_lb12_pilots(output_root=root / "results" / "pilots", use_cache=True)
    run_robustness_pilots(output_root=root / "results" / "robustness", use_cache=True)
    return run_class_i_scaling_campaign(
        root / "configs" / "campaigns" / "class_i_equilibrium_scaling.json",
        output_root=output_root,
        use_cache=use_cache,
    )


def _resolve_cycle_kernel(panel: dict[str, Any], size: int) -> dict[str, Any]:
    params = {
        "n": int(size),
        "self_weight": float(panel["self_weight"]),
        "forward_weight": float(panel["forward_weight"]),
        "backward_weight": float(panel["backward_weight"]),
    }
    return build_substrate_family(panel["family_name"], **params)


def _half_ring_lens(size: int) -> np.ndarray:
    half = size // 2
    return np.asarray([0 if i < half else 1 for i in range(size)], dtype=np.int64)


def _fit_maybe_bootstrap(group_rows: list[dict[str, Any]], fit_grid: dict[str, Any]) -> dict[str, Any]:
    fit = grid_search_collapse_fit(
        group_rows,
        lambda_c_grid=[float(x) for x in fit_grid["lambda_c_grid"]],
        beta_over_nu_grid=[float(x) for x in fit_grid["beta_over_nu_grid"]],
        inv_nu_grid=[float(x) for x in fit_grid["inv_nu_grid"]],
        n_bins=int(fit_grid.get("collapse_bins", 12)),
        top_k=6,
    )
    best = fit["best_fit"]
    if int(fit_grid.get("bootstrap_replicates", 0)) > 0:
        raw_rows = [
            {
                "size": int(r["size"]),
                "closure_strength_lambda": float(r["closure_strength_lambda"]),
                "order_value": float(r["order_mean"]),
            }
            for r in group_rows
        ]
        boot = bootstrap_collapse_fit(
            raw_rows,
            {
                "lambda_c_grid": fit_grid["lambda_c_grid"],
                "beta_over_nu_grid": fit_grid["beta_over_nu_grid"],
                "inv_nu_grid": fit_grid["inv_nu_grid"],
                "n_bins": int(fit_grid.get("collapse_bins", 12)),
                "ci_percentiles": list(fit_grid.get("ci_percentiles", [5, 95])),
            },
            n_boot=int(fit_grid["bootstrap_replicates"]),
            seed=2026,
        )
    else:
        boot = {
            "ci_percentiles": list(fit_grid.get("ci_percentiles", [5, 95])),
            "ci": {"lambda_c": [best["lambda_c"], best["lambda_c"]], "beta_over_nu": [best["beta_over_nu"], best["beta_over_nu"]], "inv_nu": [best["inv_nu"], best["inv_nu"]]},
            "successful_bootstrap_count": 0,
            "requested_bootstrap_count": int(fit_grid.get("bootstrap_replicates", 0)),
            "seed": 2026,
        }
    return format_scaling_report(best, boot, {"pairwise_crossings": [], "crossing_count": 0})


def build_class_ii_panel_rows(config: dict[str, Any], runs_dir: Path, use_cache: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    primary = config["primary_panel"]
    shadow = config["shadow_panel"]
    primary_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    executed = 0
    cached = 0

    for panel_name, panel, sink in [("primary", primary, primary_rows), ("shadow", shadow, shadow_rows)]:
        for size in panel["sizes"]:
            substrate = _resolve_cycle_kernel(panel, int(size))
            p_base = np.asarray(substrate["P"], dtype=np.float64)
            lens = _half_ring_lens(int(size))
            q = pushforward_matrix(lens, 2)
            lift, _ = build_lift_family(panel["lift_name"], f=lens, k=2)
            for lam in panel["lambda_grid"]:
                spec = {"panel": panel_name, "panel_config": panel, "size": int(size), "lambda": float(lam), "fw": panel["forward_weight"], "bw": panel["backward_weight"]}
                run_id = f"cmp_{_stable_hash(spec)}"
                row_file = runs_dir / f"{run_id}.json"
                if cache_reuse_allowed(use_cache) and row_file.exists():
                    row = json.loads(row_file.read_text(encoding="utf-8"))
                    row["cache_status"] = "cached"
                    cached += 1
                else:
                    p = apply_closure_strength_control(
                        p_base,
                        closure_strength_lambda=float(lam),
                        Q_f=q,
                        U_f=np.asarray(lift),
                        mode=panel["control_application_name"],
                    )
                    tau_cfg = {
                        "run_id": run_id,
                        "tau_protocol": dict(panel["tau_protocol"]),
                        "control": {"closure_strength_lambda": float(lam)},
                    }
                    tau = int(resolve_run_settings(tau_cfg, P=p)["resolved_tau"])
                    bundle = default_metric_bundle(p, lens, tau=tau, lift_name=panel["lift_name"], U_f=lift)
                    row = {
                        "panel": panel_name,
                        "size": int(size),
                        "closure_strength_lambda": float(lam),
                        "seed": "",
                        "run_id": run_id,
                        "analysis_k": int(bundle["analysis_k"]),
                        "order_value": observed_float(bundle["objecthood_order"]),
                        "closure_error": observed_float(bundle["closure_error"]),
                        "staging_gap": observed_float(bundle["staging_gap"]),
                        "affinity": observed_float(bundle["affinity"]),
                        "cache_status": "executed",
                    }
                    row_file.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
                    executed += 1
                sink.append(row)
    return primary_rows, shadow_rows, executed, cached


def fit_class_ii_collapse(primary_group_rows: list[dict[str, Any]], fit_config: dict[str, Any]) -> dict[str, Any]:
    return _fit_maybe_bootstrap(primary_group_rows, fit_config)


def _affinity_window_mean(group_rows: list[dict[str, Any]], lambda_center: float) -> float:
    if not group_rows or not np.isfinite(lambda_center):
        return float("nan")
    lambdas = sorted({float(r["closure_strength_lambda"]) for r in group_rows})
    idx = min(range(len(lambdas)), key=lambda i: abs(lambdas[i] - lambda_center))
    chosen = {lambdas[idx]}
    if idx > 0:
        chosen.add(lambdas[idx - 1])
    if idx < len(lambdas) - 1:
        chosen.add(lambdas[idx + 1])
    vals = [observed_float(r["affinity_mean"]) for r in group_rows if float(r["closure_strength_lambda"]) in chosen]
    return float(np.mean(vals)) if vals else float("nan")


def compare_class_ii_to_class_i(class_ii_fit: dict[str, Any], class_i_fit_summary: dict[str, Any], primary_group_rows: list[dict[str, Any]], shadow_group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_ii = class_ii_fit["best_fit"]
    class_i = class_i_fit_summary["best_fit"]
    class_i_lambda = float(class_i["lambda_c"])
    class_ii_lambda = float(class_ii["lambda_c"])

    class_i_group_path = _repo_root() / "results" / "campaigns" / "class_i_equilibrium" / "analysis" / "primary_group_summary.csv"
    class_i_group_rows: list[dict[str, Any]] = []
    if class_i_group_path.exists():
        with class_i_group_path.open("r", encoding="utf-8", newline="") as handle:
            class_i_group_rows = list(csv.DictReader(handle))
    class_i_aff = _affinity_window_mean(class_i_group_rows, class_i_lambda)
    class_ii_aff = _affinity_window_mean(primary_group_rows, class_ii_lambda)
    class_ii_shadow_aff = _affinity_window_mean(shadow_group_rows, class_ii_lambda)

    delta_lambda = class_ii_lambda - class_i_lambda
    delta_beta = float(class_ii["beta_over_nu"]) - float(class_i["beta_over_nu"])
    delta_inv = float(class_ii["inv_nu"]) - float(class_i["inv_nu"])
    aff_ratio = class_ii_aff / max(class_i_aff, 1e-8)
    aff_diff = class_ii_aff - class_i_aff

    candidate_class_ii = (
        np.isfinite(float(class_ii["objective"]))
        and class_ii_aff > 1e-2
    )
    separation = bool(
        candidate_class_ii
        and np.isfinite(class_i_aff) and class_i_aff >= 0
        and (
            class_ii_aff >= 10.0 * max(class_i_aff, 1e-8)
            or aff_diff >= 1e-2
        )
        and (
            abs(delta_lambda) >= 0.05
            or class_ii_aff >= 10.0 * max(class_i_aff, 1e-8)
            or aff_diff >= 1e-2
        )
    )
    diagnosis = (
        "driven/equilibrium separation appears plausible under affinity and fit diagnostics"
        if separation
        else "class separation not yet strong enough under current diagnostics"
    )
    return {
        "class_i_lambda_c_fit": class_i_lambda,
        "class_i_beta_over_nu_fit": float(class_i["beta_over_nu"]),
        "class_i_inv_nu_fit": float(class_i["inv_nu"]),
        "class_i_collapse_objective": float(class_i["objective"]),
        "class_ii_lambda_c_fit": class_ii_lambda,
        "class_ii_beta_over_nu_fit": float(class_ii["beta_over_nu"]),
        "class_ii_inv_nu_fit": float(class_ii["inv_nu"]),
        "class_ii_collapse_objective": float(class_ii["objective"]),
        "class_i_affinity_window_mean": float(class_i_aff),
        "class_ii_affinity_window_mean": float(class_ii_aff),
        "class_ii_shadow_affinity_window_mean": float(class_ii_shadow_aff),
        "delta_lambda_c_fit": float(delta_lambda),
        "delta_beta_over_nu_fit": float(delta_beta),
        "delta_inv_nu_fit": float(delta_inv),
        "affinity_contrast_ratio": float(aff_ratio),
        "affinity_contrast_difference": float(aff_diff),
        "class_separation_plausible": bool(separation),
        "diagnosis": diagnosis,
    }


def write_class_ii_plots(primary_group_rows: list[dict[str, Any]], shadow_group_rows: list[dict[str, Any]], fit_result: dict[str, Any], comparison_summary: dict[str, Any], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_size_primary: dict[int, list[dict[str, Any]]] = {}
    for row in primary_group_rows:
        by_size_primary.setdefault(int(row["size"]), []).append(row)
    lambdas = sorted({float(r["closure_strength_lambda"]) for r in primary_group_rows})
    order_aff_series: dict[str, list[float]] = {}
    for size, rows in sorted(by_size_primary.items()):
        rmap = {float(r["closure_strength_lambda"]): r for r in rows}
        order_aff_series[f"L{size}:M"] = [float(rmap[lam]["order_mean"]) for lam in lambdas]
        order_aff_series[f"L{size}:Aff"] = [observed_float(rmap[lam]["affinity_mean"]) for lam in lambdas]
    p1 = output_dir / "order_aff_vs_lambda_by_size.png"
    _line_plot(p1, lambdas, order_aff_series)

    best = fit_result["best_fit"]
    x_ref = sorted(
        {
            (float(r["closure_strength_lambda"]) - float(best["lambda_c"])) * (float(r["size"]) ** float(best["inv_nu"]))
            for r in primary_group_rows
        }
    )
    collapse_series: dict[str, list[float]] = {}
    for size, rows in sorted(by_size_primary.items()):
        x_map = {
            (float(r["closure_strength_lambda"]) - float(best["lambda_c"])) * (float(r["size"]) ** float(best["inv_nu"])): float(r["order_mean"]) * (float(r["size"]) ** float(best["beta_over_nu"]))
            for r in rows
        }
        collapse_series[f"L{size}"] = [x_map[min(x_map.keys(), key=lambda z: abs(z - x))] for x in x_ref]
    p2 = output_dir / "collapse_best_fit.png"
    _line_plot(p2, x_ref, collapse_series)

    by_size_shadow: dict[int, list[dict[str, Any]]] = {}
    for row in shadow_group_rows:
        by_size_shadow.setdefault(int(row["size"]), []).append(row)
    lambdas_shadow = sorted({float(r["closure_strength_lambda"]) for r in shadow_group_rows})
    shadow_aff_series: dict[str, list[float]] = {}
    for size, rows in sorted(by_size_shadow.items()):
        rmap = {float(r["closure_strength_lambda"]): r for r in rows}
        shadow_aff_series[f"L{size}:Aff"] = [observed_float(rmap[lam]["affinity_mean"]) for lam in lambdas_shadow]
    p3 = output_dir / "shadow_bias_affinity_vs_lambda.png"
    _line_plot(p3, lambdas_shadow, shadow_aff_series)

    cmp_series = {
        "class_i_aff": [comparison_summary["class_i_affinity_window_mean"], comparison_summary["class_i_affinity_window_mean"]],
        "class_ii_aff": [comparison_summary["class_ii_affinity_window_mean"], comparison_summary["class_ii_affinity_window_mean"]],
        "class_ii_shadow_aff": [comparison_summary["class_ii_shadow_affinity_window_mean"], comparison_summary["class_ii_shadow_affinity_window_mean"]],
    }
    p4 = output_dir / "class_i_vs_class_ii_comparison.png"
    _line_plot(p4, [0.0, 1.0], cmp_series)
    return [str(p1), str(p2), str(p3), str(p4)]


def format_class_ii_campaign_summary(fit_result: dict[str, Any], comparison_summary: dict[str, Any], campaign_diagnosis: dict[str, Any]) -> dict[str, Any]:
    return {
        "fit_result": fit_result,
        "comparison_to_class_i": comparison_summary,
        "campaign_diagnosis": campaign_diagnosis,
    }


def run_class_ii_scaling_campaign(config_path_or_obj: str | Path | dict[str, Any], output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        config = config_path_or_obj
    else:
        config = json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))
    if output_root is None:
        output_root = _repo_root() / "results" / "campaigns"
    else:
        output_root = Path(output_root)
    artifact_root = output_root / config["artifact_subdir"]
    dirs = {
        "config": artifact_root / "config",
        "seeds": artifact_root / "seeds",
        "metrics": artifact_root / "metrics",
        "notes": artifact_root / "notes",
        "env": artifact_root / "env",
        "analysis": artifact_root / "analysis",
        "plots": artifact_root / "plots",
        "runs": artifact_root / "runs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    primary_raw, shadow_raw, executed_count, cached_count = build_class_ii_panel_rows(config, dirs["runs"], use_cache=use_cache)
    primary_group = _group_primary(primary_raw)
    shadow_group = _group_primary(shadow_raw)
    fit_report = fit_class_ii_collapse(primary_group, config["fit_grid"])

    class_i_fit_path = _repo_root() / "results" / "campaigns" / "class_i_equilibrium" / "analysis" / "fit_summary.json"
    if not artifact_is_current(class_i_fit_path):
        run_class_i_campaign_driver(output_root=_repo_root() / "results" / "campaigns", use_cache=True)
    class_i_fit = json.loads(class_i_fit_path.read_text(encoding="utf-8"))
    comparison = compare_class_ii_to_class_i(fit_report, class_i_fit, primary_group, shadow_group)

    best = fit_report["best_fit"]
    lam_min = min(float(x) for x in config["primary_panel"]["lambda_grid"])
    lam_max = max(float(x) for x in config["primary_panel"]["lambda_grid"])
    candidate = bool(
        np.isfinite(float(best["objective"]))
        and lam_min <= float(best["lambda_c"]) <= lam_max
        and comparison["class_ii_affinity_window_mean"] > 1e-2
    )
    separation = bool(comparison["class_separation_plausible"])
    keep = "keep pursuing" if separation else "not yet strong enough"
    diagnosis = {
        "candidate_class_ii_collapse_obtained": candidate,
        "class_separation_plausible": separation,
        "final_verdict": keep,
    }

    if not candidate or not separation:
        failure = {
            "best_fit_parameters": best,
            "collapse_objective": best["objective"],
            "affinity_levels": {
                "class_i_affinity_window_mean": comparison["class_i_affinity_window_mean"],
                "class_ii_affinity_window_mean": comparison["class_ii_affinity_window_mean"],
                "class_ii_shadow_affinity_window_mean": comparison["class_ii_shadow_affinity_window_mean"],
            },
            "class_i_loaded": class_i_fit_path.exists(),
            "diagnosis": "class-II candidate and/or class separation criteria not fully met",
        }
        (dirs["analysis"] / "failure_report.json").write_text(scientific_dumps(failure, indent=2) + "\n", encoding="utf-8")

    (dirs["analysis"] / "fit_summary.json").write_text(scientific_dumps(fit_report, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "comparison_to_class_i.json").write_text(scientific_dumps(comparison, indent=2) + "\n", encoding="utf-8")
    _write_csv(
        dirs["analysis"] / "primary_raw_metrics.csv",
        primary_raw,
        ["panel", "size", "closure_strength_lambda", "seed", "run_id", "analysis_k", "order_value", "closure_error", "staging_gap", "affinity", "cache_status"],
    )
    _write_csv(
        dirs["analysis"] / "primary_group_summary.csv",
        primary_group,
        ["size", "closure_strength_lambda", "analysis_k", "order_mean", "closure_error_mean", "staging_gap_mean", "affinity_mean", "n_replicates"],
    )
    _write_csv(
        dirs["analysis"] / "shadow_raw_metrics.csv",
        shadow_raw,
        ["panel", "size", "closure_strength_lambda", "seed", "run_id", "analysis_k", "order_value", "closure_error", "staging_gap", "affinity", "cache_status"],
    )
    _write_csv(
        dirs["analysis"] / "shadow_group_summary.csv",
        shadow_group,
        ["size", "closure_strength_lambda", "analysis_k", "order_mean", "closure_error_mean", "staging_gap_mean", "affinity_mean", "n_replicates"],
    )
    write_class_ii_plots(primary_group, shadow_group, fit_report, comparison, dirs["plots"])

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    seeds = dirs["seeds"] / "seeds.json"
    seeds.write_text("[]\n", encoding="utf-8")
    metrics_csv = dirs["metrics"] / "metrics.csv"
    _write_csv(
        metrics_csv,
        [
            {
                "campaign_name": config["campaign_id"],
                "primary_size_count": len(config["primary_panel"]["sizes"]),
                "shadow_size_count": len(config["shadow_panel"]["sizes"]),
                "lambda_c_fit": best["lambda_c"],
                "beta_over_nu_fit": best["beta_over_nu"],
                "inv_nu_fit": best["inv_nu"],
                "collapse_objective": best["objective"],
                "class_separation_plausible": separation,
                "final_verdict": keep,
                "executed_count": executed_count,
                "cached_count": cached_count,
            }
        ],
        [
            "campaign_name",
            "primary_size_count",
            "shadow_size_count",
            "lambda_c_fit",
            "beta_over_nu_fit",
            "inv_nu_fit",
            "collapse_objective",
            "class_separation_plausible",
            "final_verdict",
            "executed_count",
            "cached_count",
        ],
    )
    notes = dirs["notes"] / "findings.md"
    notes.write_text(
        "\n".join(
            [
                "# Class-II campaign findings",
                "",
                f"- candidate_class_ii_collapse_obtained: `{candidate}`",
                f"- class_separation_plausible: `{separation}`",
                f"- final_verdict: `{keep}`",
                f"- best_fit: `{best}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = dirs["env"] / "environment.json"
    env.write_text(
        scientific_dumps({"python": "unknown", "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    schema = load_schema(_repo_root() / "configs" / "result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_ii_driven_scaling_campaign",
        "bundle_id": config["campaign_id"],
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(_repo_root()),
        "config_snapshot_path": str(config_snapshot.relative_to(artifact_root)),
        "seed_list_path": str(seeds.relative_to(artifact_root)),
        "metrics_table_path": str(metrics_csv.relative_to(artifact_root)),
        "plots_dir_path": str(dirs["plots"].relative_to(artifact_root)),
        "notes_file_path": str(notes.relative_to(artifact_root)),
        "environment_snapshot_path": str(env.relative_to(artifact_root)),
    }
    validate_manifest(manifest, schema)
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    findings_note = _repo_root() / config["findings_note_path"]
    findings_note.parent.mkdir(parents=True, exist_ok=True)
    findings_note.write_text(
        "\n".join(
            [
                "# LB-16 class-II scaling campaign",
                "",
                f"- campaign config path: `{config['config_path_hint']}`",
                f"- artifact root: `{artifact_root}`",
                "- primary panel: driven_cycle_family low-bias sizes [8,16,32,64]",
                "- shadow panel: driven_cycle_family high-bias sizes [8,16,32,64]",
                f"- fitted parameters: `{best}`",
                f"- class-I comparison: `{comparison}`",
                f"- candidate class-II collapse obtained: `{'yes' if candidate else 'no'}`",
                f"- class separation looks real enough to keep pursuing: `{'yes' if separation else 'no'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "campaign_name": config["campaign_id"],
        "primary_size_count": len(config["primary_panel"]["sizes"]),
        "shadow_size_count": len(config["shadow_panel"]["sizes"]),
        "class_ii_lambda_c_fit": best["lambda_c"],
        "class_ii_beta_over_nu_fit": best["beta_over_nu"],
        "class_ii_inv_nu_fit": best["inv_nu"],
        "class_ii_collapse_objective": best["objective"],
        "class_separation_plausible": separation,
        "final_verdict": keep,
        "artifact_root": str(artifact_root),
        "executed_count": executed_count,
        "cached_count": cached_count,
        "failure_report_needed": (not candidate) or (not separation),
    }


def run_class_ii_campaign_driver(output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    root = _repo_root()
    subprocess.run([sys.executable, str(root / "scripts" / "smoke_scaling_analysis.py")], check=True)
    run_all_lb12_pilots(output_root=root / "results" / "pilots", use_cache=True)
    run_robustness_pilots(output_root=root / "results" / "robustness", use_cache=True)
    run_class_i_campaign_driver(output_root=root / "results" / "campaigns", use_cache=True)
    return run_class_ii_scaling_campaign(
        root / "configs" / "campaigns" / "class_ii_driven_scaling.json",
        output_root=output_root,
        use_cache=use_cache,
    )
