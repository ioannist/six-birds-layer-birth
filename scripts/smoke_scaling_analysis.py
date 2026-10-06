#!/usr/bin/env python3
"""Run LB-14 scaling analysis smoke on mocked finite-size data."""

from __future__ import annotations

import binascii
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import subprocess
import zlib

import numpy as np

from layerbirth.contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from layerbirth.scaling import (
    MockFssParams,
    bootstrap_collapse_fit,
    estimate_pairwise_crossings,
    format_scaling_report,
    generate_mock_observations,
    grid_search_collapse_fit,
    group_order_observations,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
        return {"git_commit": commit or "unknown", "git_dirty": bool(dirty)}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": False}


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
    w, h = 860, 500
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 60, 20, 20, 50
    x0, x1 = left, w - right
    y0, y1 = top, h - bottom
    _draw_line(img, x0, y1, x1, y1, (0, 0, 0))
    _draw_line(img, x0, y0, x0, y1, (0, 0, 0))
    xmin, xmax = min(xvals), max(xvals)
    yvals = [v for arr in series.values() for v in arr]
    ymin, ymax = min(yvals), max(yvals)
    if np.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    colors = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180)]
    for i, arr in enumerate(series.values()):
        color = colors[i % len(colors)]
        pts = []
        for x, y in zip(xvals, arr):
            px = x0 + int((x - xmin) / max(xmax - xmin, 1e-12) * (x1 - x0))
            py = y1 - int((y - ymin) / max(ymax - ymin, 1e-12) * (y1 - y0))
            pts.append((px, py))
        for a, b in zip(pts[:-1], pts[1:]):
            _draw_line(img, a[0], a[1], b[0], b[1], color)
    _write_png(path, img)


def run_scaling_smoke(output_root: Path | None = None) -> dict:
    root = _repo_root()
    cfg_path = root / "configs" / "scaling" / "mock_fss.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if output_root is None:
        bundle_root = root / "results" / "scaling_smoke" / "mock_fss"
    else:
        bundle_root = Path(output_root) / "mock_fss"
    dirs = {
        "config": bundle_root / "config",
        "seeds": bundle_root / "seeds",
        "metrics": bundle_root / "metrics",
        "notes": bundle_root / "notes",
        "env": bundle_root / "env",
        "analysis": bundle_root / "analysis",
        "plots": bundle_root / "plots",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    params = MockFssParams(
        sizes=[int(x) for x in cfg["sizes"]],
        lambda_grid=[float(x) for x in cfg["lambda_grid"]],
        lambda_c_true=float(cfg["lambda_c_true"]),
        beta_over_nu_true=float(cfg["beta_over_nu_true"]),
        inv_nu_true=float(cfg["inv_nu_true"]),
        replicate_count=int(cfg["replicate_count"]),
        noise_sigma=float(cfg["noise_sigma"]),
        seed=int(cfg["seed"]),
    )
    raw_rows = generate_mock_observations(params)
    grouped = group_order_observations(raw_rows)
    crossings = estimate_pairwise_crossings(grouped)
    fit = grid_search_collapse_fit(
        grouped,
        lambda_c_grid=[float(x) for x in cfg["lambda_c_grid"]],
        beta_over_nu_grid=[float(x) for x in cfg["beta_over_nu_grid"]],
        inv_nu_grid=[float(x) for x in cfg["inv_nu_grid"]],
        n_bins=int(cfg.get("collapse_bins", 12)),
        top_k=5,
    )
    boot = bootstrap_collapse_fit(
        raw_rows,
        {
            "lambda_c_grid": [float(x) for x in cfg["lambda_c_grid"]],
            "beta_over_nu_grid": [float(x) for x in cfg["beta_over_nu_grid"]],
            "inv_nu_grid": [float(x) for x in cfg["inv_nu_grid"]],
            "n_bins": int(cfg.get("collapse_bins", 12)),
            "ci_percentiles": list(cfg["ci_percentiles"]),
        },
        n_boot=int(cfg["bootstrap_replicates"]),
        seed=int(cfg["seed"]),
    )
    report = format_scaling_report(fit["best_fit"], boot, crossings)

    obs_csv = dirs["analysis"] / "mock_observations.csv"
    with obs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["size", "closure_strength_lambda", "order_value", "replicate_id"]
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    group_csv = dirs["analysis"] / "group_summary.csv"
    with group_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "size",
                "closure_strength_lambda",
                "n_replicates",
                "order_mean",
                "susceptibility",
                "binder_like_cumulant",
            ],
        )
        writer.writeheader()
        writer.writerows(grouped)

    fit_json = dirs["analysis"] / "fit_summary.json"
    fit_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    metrics_csv = dirs["metrics"] / "metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_name",
                "lambda_c_true",
                "beta_over_nu_true",
                "inv_nu_true",
                "lambda_c_fit",
                "beta_over_nu_fit",
                "inv_nu_fit",
                "collapse_objective",
                "crossing_lambda_c_mean",
                "bootstrap_lambda_c_lo",
                "bootstrap_lambda_c_hi",
                "bootstrap_beta_over_nu_lo",
                "bootstrap_beta_over_nu_hi",
                "bootstrap_inv_nu_lo",
                "bootstrap_inv_nu_hi",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset_name": cfg["dataset_name"],
                "lambda_c_true": cfg["lambda_c_true"],
                "beta_over_nu_true": cfg["beta_over_nu_true"],
                "inv_nu_true": cfg["inv_nu_true"],
                "lambda_c_fit": fit["best_fit"]["lambda_c"],
                "beta_over_nu_fit": fit["best_fit"]["beta_over_nu"],
                "inv_nu_fit": fit["best_fit"]["inv_nu"],
                "collapse_objective": fit["best_fit"]["objective"],
                "crossing_lambda_c_mean": report["crossings"]["crossing_lambda_c_mean"],
                "bootstrap_lambda_c_lo": boot["ci"]["lambda_c"][0],
                "bootstrap_lambda_c_hi": boot["ci"]["lambda_c"][1],
                "bootstrap_beta_over_nu_lo": boot["ci"]["beta_over_nu"][0],
                "bootstrap_beta_over_nu_hi": boot["ci"]["beta_over_nu"][1],
                "bootstrap_inv_nu_lo": boot["ci"]["inv_nu"][0],
                "bootstrap_inv_nu_hi": boot["ci"]["inv_nu"][1],
            }
        )

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    seeds = dirs["seeds"] / "seeds.json"
    seeds.write_text(json.dumps([cfg["seed"]], indent=2) + "\n", encoding="utf-8")
    notes = dirs["notes"] / "findings.md"
    notes.write_text(
        "\n".join(
            [
                "# Scaling smoke findings",
                "",
                f"- lambda_c_true: `{cfg['lambda_c_true']}`",
                f"- lambda_c_fit: `{fit['best_fit']['lambda_c']}`",
                f"- beta_over_nu_fit: `{fit['best_fit']['beta_over_nu']}`",
                f"- inv_nu_fit: `{fit['best_fit']['inv_nu']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = dirs["env"] / "environment.json"
    env.write_text(
        json.dumps({"python": "unknown", "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    # Plots
    best = fit["best_fit"]
    by_size = {}
    for row in grouped:
        by_size.setdefault(int(row["size"]), []).append(row)
    collapse_series = {}
    for size, rows in sorted(by_size.items()):
        rows = sorted(rows, key=lambda r: float(r["closure_strength_lambda"]))
        xs = [
            (float(r["closure_strength_lambda"]) - float(best["lambda_c"])) * (size ** float(best["inv_nu"]))
            for r in rows
        ]
        ys = [float(r["order_mean"]) * (size ** float(best["beta_over_nu"])) for r in rows]
        collapse_series[f"L{size}"] = (xs, ys)
    # flatten x grid for plotting helper
    x_ref = sorted({x for xs, _ in collapse_series.values() for x in xs})
    collapse_plot_series = {}
    for label, (xs, ys) in collapse_series.items():
        y_map = {x: y for x, y in zip(xs, ys)}
        collapse_plot_series[label] = [float(y_map.get(x, np.nan)) for x in x_ref]
    # replace nans by nearest available for continuity in simple renderer
    for label, arr in collapse_plot_series.items():
        for i, v in enumerate(arr):
            if np.isnan(v):
                left = next((arr[j] for j in range(i - 1, -1, -1) if not np.isnan(arr[j])), None)
                right = next((arr[j] for j in range(i + 1, len(arr)) if not np.isnan(arr[j])), None)
                arr[i] = left if left is not None else (right if right is not None else 0.0)
    _line_plot(dirs["plots"] / "collapse_best_fit.png", x_ref, collapse_plot_series)

    binder_series = {}
    lam_ref = sorted({float(r["closure_strength_lambda"]) for r in grouped})
    for size, rows in sorted(by_size.items()):
        rmap = {float(r["closure_strength_lambda"]): float(r["binder_like_cumulant"]) for r in rows}
        binder_series[f"L{size}"] = [rmap[lam] for lam in lam_ref]
    _line_plot(dirs["plots"] / "binder_crossings.png", lam_ref, binder_series)

    susc_series = {}
    for size, rows in sorted(by_size.items()):
        rmap = {float(r["closure_strength_lambda"]): float(r["susceptibility"]) for r in rows}
        susc_series[f"L{size}"] = [rmap[lam] for lam in lam_ref]
    _line_plot(dirs["plots"] / "susceptibility_vs_lambda.png", lam_ref, susc_series)

    schema = load_schema(root / "configs" / "result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "scaling_mock_fss_smoke",
        "bundle_id": "mock_fss",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": str(config_snapshot.relative_to(bundle_root)),
        "seed_list_path": str(seeds.relative_to(bundle_root)),
        "metrics_table_path": str(metrics_csv.relative_to(bundle_root)),
        "plots_dir_path": str(dirs["plots"].relative_to(bundle_root)),
        "notes_file_path": str(notes.relative_to(bundle_root)),
        "environment_snapshot_path": str(env.relative_to(bundle_root)),
    }
    validate_manifest(manifest, schema)
    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "artifact_root": str(bundle_root),
        "manifest_path": str(manifest_path),
        "fit": fit["best_fit"],
        "crossings": crossings,
        "bootstrap_ci": boot["ci"],
    }


def main() -> int:
    summary = run_scaling_smoke()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
