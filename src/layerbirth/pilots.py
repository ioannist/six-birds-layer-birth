"""LB-12 pilot scan orchestration."""

from __future__ import annotations

import binascii
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import subprocess
import zlib
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import cache_reuse_allowed, computation_hash, implementation_fingerprint
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lenses import build_lens_family
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import pushforward_matrix
from .protocols import resolve_run_settings
from .sweep import apply_closure_strength_control
from .substrates import build_substrate_family


PILOT_FIELDS = [
    "pilot_family",
    "case_name",
    "run_id",
    "closure_strength_lambda",
    "tau_protocol_name",
    "resolved_tau",
    "analysis_k",
    "closure_error",
    "objecthood_order",
    "staging_gap",
    "affinity",
    "holonomy",
    "cache_status",
    "manifest_path",
]


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


def _load_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    path = Path(config_path_or_obj)
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_bundle_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "config": root / "config",
        "metrics": root / "metrics",
        "plots": root / "plots",
        "notes": root / "notes",
        "env": root / "env",
        "runs": root / "runs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_manifest(
    *,
    root: Path,
    experiment_id: str,
    bundle_id: str,
    config_snapshot_path: Path,
    metrics_path: Path,
    notes_path: Path,
    env_path: Path,
) -> Path:
    schema = load_schema(_repo_root() / "configs" / "result_bundle.schema.json")
    dummy_seed = root / "seeds.json"
    dummy_seed.write_text("[]\n", encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "bundle_id": bundle_id,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(_repo_root()),
        "config_snapshot_path": str(config_snapshot_path.relative_to(root)),
        "seed_list_path": str(dummy_seed.relative_to(root)),
        "metrics_table_path": str(metrics_path.relative_to(root)),
        "plots_dir_path": "plots",
        "notes_file_path": str(notes_path.relative_to(root)),
        "environment_snapshot_path": str(env_path.relative_to(root)),
    }
    validate_manifest(manifest, schema)
    out = root / "manifest.json"
    out.write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def _resolve_case_params(case: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    params = dict(case.get("family_params", {}))
    size = case.get("size")
    if size is not None and case.get("derive_block_size_from_size", False):
        n_blocks = int(params["n_blocks"])
        if int(size) % n_blocks != 0:
            raise ValueError("pilot size must be divisible by n_blocks")
        params["block_size"] = int(size) // n_blocks
    elif size is not None and "n" not in params:
        params["n"] = int(size)
    seed = case.get("seed")
    if seed is not None:
        params["seed"] = int(seed)
    return params, None if seed is None else int(seed)


def _resolve_lens(substrate: dict[str, Any], lens_spec: dict[str, Any]) -> np.ndarray:
    name = lens_spec["name"]
    if name == "manual_family_block_lens":
        return np.asarray(substrate["block_lens"], dtype=np.int64)
    if name == "family_coarse_lens":
        return np.asarray(substrate["coarse_lens"], dtype=np.int64)
    if name == "manual_partition_lens":
        lens, _ = build_lens_family("manual_partition_lens", labels=lens_spec["labels"])
        return np.asarray(lens, dtype=np.int64)
    kwargs = dict(lens_spec.get("kwargs", {}))
    lens, _ = build_lens_family(name, P=np.asarray(substrate["P"], dtype=np.float64), **kwargs)
    return np.asarray(lens, dtype=np.int64)


def _resolve_lift(lift_spec: dict[str, Any], P: np.ndarray, lens: np.ndarray) -> np.ndarray:
    name = lift_spec["name"]
    k = int(np.max(lens)) + 1
    kwargs = dict(lift_spec.get("kwargs", {}))
    if name in {"uniform_lift_family", "prototype_lift_family"}:
        u, _ = build_lift_family(name, f=lens, k=k, **kwargs)
        return np.asarray(u, dtype=np.float64)
    if name == "stationary_within_fiber_lift":
        u, _ = build_lift_family(name, P=P, f=lens, k=k, **kwargs)
        return np.asarray(u, dtype=np.float64)
    raise ValueError(f"Unknown lift family: {name}")


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


def _line_plot_png(
    path: Path,
    lambdas: list[float],
    series: dict[str, list[float]],
) -> None:
    w, h = 800, 480
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 60, 20, 20, 50
    x0, x1 = left, w - right
    y0, y1 = top, h - bottom
    _draw_line(img, x0, y1, x1, y1, (0, 0, 0))
    _draw_line(img, x0, y0, x0, y1, (0, 0, 0))

    x_min, x_max = min(lambdas), max(lambdas)
    all_vals = [v for values in series.values() for v in values]
    y_min, y_max = min(all_vals), max(all_vals)
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    colors = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180)]
    for i, values in enumerate(series.values()):
        color = colors[i % len(colors)]
        pts: list[tuple[int, int]] = []
        for lam, val in zip(lambdas, values):
            x = x0 + int((lam - x_min) / max(x_max - x_min, 1e-12) * (x1 - x0))
            y = y1 - int((val - y_min) / max(y_max - y_min, 1e-12) * (y1 - y0))
            pts.append((x, y))
        for a, b in zip(pts[:-1], pts[1:]):
            _draw_line(img, a[0], a[1], b[0], b[1], color)
    _write_png(path, img)


def summarize_pilot_window(metrics_rows: list[dict[str, Any]], family_kind: str) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in metrics_rows:
        by_case.setdefault(str(row["case_name"]), []).append(row)

    case_summaries: dict[str, Any] = {}
    for case_name, rows in by_case.items():
        ordered = sorted(rows, key=lambda r: float(r["closure_strength_lambda"]))
        best_score = -1.0
        best_interval: list[float] | None = None
        for a, b in zip(ordered[:-1], ordered[1:]):
            score = abs(observed_float(b["objecthood_order"]) - observed_float(a["objecthood_order"])) + abs(
                observed_float(b["closure_error"]) - observed_float(a["closure_error"])
            )
            if score > best_score:
                best_score = score
                best_interval = [float(a["closure_strength_lambda"]), float(b["closure_strength_lambda"])]
        if best_score >= 0.10 and best_interval is not None:
            diagnosis = f"candidate transition window on scanned grid: {best_interval}"
        else:
            diagnosis = "no clear transition window on scanned grid"
        case_summaries[case_name] = {
            "max_window_score": max(0.0, best_score),
            "candidate_interval": best_interval,
            "diagnosis": diagnosis,
        }

    out: dict[str, Any] = {"family_kind": family_kind, "cases": case_summaries}
    if family_kind == "holonomy_control":
        bal = [r for r in metrics_rows if r["case_name"] == "balanced"]
        unbal = [r for r in metrics_rows if r["case_name"] == "unbalanced"]
        paired = []
        for b in bal:
            lam = float(b["closure_strength_lambda"])
            u = [r for r in unbal if float(r["closure_strength_lambda"]) == lam][0]
            paired.append((b, u))
        hol_sep = max(abs(float(u["holonomy"]) - float(b["holonomy"])) for b, u in paired)
        aff_max = max(
            max(abs(observed_float(b["affinity"])), abs(observed_float(u["affinity"])))
            for b, u in paired
        )
        coarse_diff = max(
            max(
                abs(observed_float(u["closure_error"]) - observed_float(b["closure_error"])),
                abs(observed_float(u["objecthood_order"]) - observed_float(b["objecthood_order"])),
                abs(observed_float(u["staging_gap"]) - observed_float(b["staging_gap"])),
            )
            for b, u in paired
        )
        out["holonomy_separation"] = hol_sep
        out["max_affinity_abs"] = aff_max
        out["max_coarse_metric_diff"] = coarse_diff
        out["diagnosis"] = (
            "holonomy separates balanced vs unbalanced with near-zero affinity and aligned coarse metrics"
            if hol_sep > 0.05 and aff_max < 1e-6 and coarse_diff < 1e-9
            else "holonomy control not clean on scanned grid"
        )
    return out


def write_pilot_plots(metrics_rows: list[dict[str, Any]], output_dir: Path, family_kind: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if family_kind in {"equilibrium_like", "null_family"}:
        case = sorted({r["case_name"] for r in metrics_rows})[0]
        rows = sorted(
            [r for r in metrics_rows if r["case_name"] == case],
            key=lambda r: float(r["closure_strength_lambda"]),
        )
        lambdas = [float(r["closure_strength_lambda"]) for r in rows]
        series = {
            "CE": [observed_float(r["closure_error"]) for r in rows],
            "M_obj": [observed_float(r["objecthood_order"]) for r in rows],
            "SG": [observed_float(r["staging_gap"]) for r in rows],
            "Aff": [observed_float(r["affinity"]) for r in rows],
        }
        path = output_dir / "metrics_vs_lambda.png"
        _line_plot_png(path, lambdas, series)
        paths.append(str(path))
    elif family_kind == "driven_family":
        for case, fname in [
            ("low_bias", "low_bias_metrics_vs_lambda.png"),
            ("high_bias", "high_bias_metrics_vs_lambda.png"),
        ]:
            rows = sorted(
                [r for r in metrics_rows if r["case_name"] == case],
                key=lambda r: float(r["closure_strength_lambda"]),
            )
            lambdas = [float(r["closure_strength_lambda"]) for r in rows]
            series = {
                "CE": [observed_float(r["closure_error"]) for r in rows],
                "M_obj": [observed_float(r["objecthood_order"]) for r in rows],
                "SG": [observed_float(r["staging_gap"]) for r in rows],
                "Aff": [observed_float(r["affinity"]) for r in rows],
            }
            path = output_dir / fname
            _line_plot_png(path, lambdas, series)
            paths.append(str(path))
    elif family_kind == "holonomy_control":
        rows_bal = sorted(
            [r for r in metrics_rows if r["case_name"] == "balanced"],
            key=lambda r: float(r["closure_strength_lambda"]),
        )
        rows_unbal = sorted(
            [r for r in metrics_rows if r["case_name"] == "unbalanced"],
            key=lambda r: float(r["closure_strength_lambda"]),
        )
        lambdas = [float(r["closure_strength_lambda"]) for r in rows_bal]
        coarse_series = {
            "CE_bal": [observed_float(r["closure_error"]) for r in rows_bal],
            "CE_unbal": [observed_float(r["closure_error"]) for r in rows_unbal],
            "M_bal": [observed_float(r["objecthood_order"]) for r in rows_bal],
            "M_unbal": [observed_float(r["objecthood_order"]) for r in rows_unbal],
            "SG_bal": [observed_float(r["staging_gap"]) for r in rows_bal],
            "SG_unbal": [observed_float(r["staging_gap"]) for r in rows_unbal],
            "Aff_bal": [observed_float(r["affinity"]) for r in rows_bal],
            "Aff_unbal": [observed_float(r["affinity"]) for r in rows_unbal],
        }
        path1 = output_dir / "coarse_metrics_vs_lambda.png"
        _line_plot_png(path1, lambdas, coarse_series)
        paths.append(str(path1))

        hol_series = {
            "Hol_bal": [float(r["holonomy"]) for r in rows_bal],
            "Hol_unbal": [float(r["holonomy"]) for r in rows_unbal],
        }
        path2 = output_dir / "holonomy_vs_lambda.png"
        _line_plot_png(path2, lambdas, hol_series)
        paths.append(str(path2))
    return paths


def _evaluate_case_run(
    pilot_family: str,
    case: dict[str, Any],
    lam: float,
    tau_protocol: dict[str, Any],
    control_application_name: str,
    lens_spec: dict[str, Any],
    lift_spec: dict[str, Any],
) -> dict[str, Any]:
    family_name = str(case["family_name"])
    family_params, seed = _resolve_case_params(case)
    substrate = build_substrate_family(family_name, **family_params)
    p = np.asarray(substrate["P"], dtype=np.float64)
    lens = _resolve_lens(substrate, lens_spec)
    q = pushforward_matrix(lens, int(np.max(lens)) + 1)
    lift = _resolve_lift(lift_spec, p, lens)
    resolved = resolve_run_settings(
        {
            "run_id": "pilot_tmp",
            "tau_protocol": tau_protocol,
            "control": {"closure_strength_lambda": float(lam)},
        },
        P=p,
    )
    p_controlled = apply_closure_strength_control(
        p,
        closure_strength_lambda=float(lam),
        Q_f=q,
        U_f=lift,
        mode=control_application_name,
    )
    holonomy_inputs = None
    if pilot_family == "holonomy_control":
        holonomy_inputs = {
            "fine_lens": np.asarray(substrate["fine_lens"], dtype=np.int64),
            "coarse_lens": np.asarray(substrate["coarse_lens"], dtype=np.int64),
        }
    bundle = default_metric_bundle(
        p_controlled,
        lens,
        tau=int(resolved["resolved_tau"]),
        holonomy_inputs=holonomy_inputs,
        lift_name=str(lift_spec["name"]),
        U_f=lift,
    )
    run_spec = {
        "pilot_family": pilot_family,
        "case_name": case["case_name"],
        "family_name": family_name,
        "family_params": family_params,
        "closure_strength_lambda": float(lam),
        "tau_protocol": tau_protocol,
        "control_application_name": control_application_name,
        "lens_spec": lens_spec,
        "lift_spec": lift_spec,
        "seed": seed,
    }
    run_id = f"pilot_{_stable_hash(run_spec)}"
    return {
        "pilot_family": pilot_family,
        "case_name": case["case_name"],
        "run_id": run_id,
        "closure_strength_lambda": float(lam),
        "tau_protocol_name": resolved["tau_protocol_name"],
        "resolved_tau": int(resolved["resolved_tau"]),
        "analysis_k": int(bundle["analysis_k"]),
        "closure_error": observed_float(bundle["closure_error"]),
        "objecthood_order": observed_float(bundle["objecthood_order"]),
        "staging_gap": observed_float(bundle["staging_gap"]),
        "affinity": observed_float(bundle["affinity"]),
        "holonomy": bundle["holonomy"],
        "cache_status": "executed",
        "manifest_path": "",
        "run_spec": run_spec,
    }


def run_pilot_scan(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    config = _load_config(config_path_or_obj)
    pilot_family = str(config["pilot_family"])
    if output_root is None:
        output_root = _repo_root() / "results" / "pilots"
    else:
        output_root = Path(output_root)
    family_root = output_root / str(config.get("output_subdir", pilot_family))
    dirs = _ensure_bundle_dirs(family_root)

    rows: list[dict[str, Any]] = []
    for case in config["cases"]:
        for lam in config["closure_strength_lambda_grid"]:
            row = _evaluate_case_run(
                pilot_family=pilot_family,
                case=case,
                lam=float(lam),
                tau_protocol=dict(config["tau_protocol"]),
                control_application_name=str(config["control_application_name"]),
                lens_spec=dict(config["lens"]),
                lift_spec=dict(config["lift"]),
            )
            run_root = dirs["runs"] / row["run_id"]
            run_metrics = run_root / "metrics.csv"
            run_manifest = run_root / "manifest.json"
            if cache_reuse_allowed(use_cache) and run_metrics.exists() and run_manifest.exists():
                cached = json.loads((run_root / "row.json").read_text(encoding="utf-8"))
                cached["cache_status"] = "cached"
                rows.append(cached)
                continue
            run_root.mkdir(parents=True, exist_ok=True)
            run_metrics.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
            row["manifest_path"] = str(run_manifest)
            run_manifest.write_text(
                scientific_dumps(
                    {
                        "run_id": row["run_id"],
                        "pilot_family": pilot_family,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_root / "row.json").write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
            rows.append(row)

    rows.sort(key=lambda r: (r["case_name"], float(r["closure_strength_lambda"])))
    metrics_csv = dirs["metrics"] / "metrics.csv"
    _write_csv(metrics_csv, rows, PILOT_FIELDS)

    summary = summarize_pilot_window(rows, pilot_family)
    summary["pilot_family"] = pilot_family
    summary["run_count"] = len(rows)
    summary["artifact_root"] = str(family_root)
    summary_json = family_root / "summary.json"
    summary_json.write_text(scientific_dumps(summary, indent=2) + "\n", encoding="utf-8")

    plot_paths = write_pilot_plots(rows, dirs["plots"], pilot_family)
    notes_file = dirs["notes"] / "findings.md"
    notes_file.write_text(
        "\n".join(
            [
                "# Pilot findings",
                "",
                f"- pilot_family: `{pilot_family}`",
                f"- run_count: `{len(rows)}`",
                f"- diagnosis: `{summary.get('diagnosis', 'see case-level diagnoses')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file = dirs["env"] / "environment.json"
    env_file.write_text(
        scientific_dumps({"python": "unknown", "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    cfg_file = dirs["config"] / "config_snapshot.json"
    cfg_file.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest = _write_manifest(
        root=family_root,
        experiment_id=f"lb12_{pilot_family}",
        bundle_id=f"lb12_{pilot_family}",
        config_snapshot_path=cfg_file,
        metrics_path=metrics_csv,
        notes_path=notes_file,
        env_path=env_file,
    )

    findings_note_path = _repo_root() / str(config["findings_note_path"])
    findings_note_path.parent.mkdir(parents=True, exist_ok=True)
    case_lines = []
    for name, case_summary in summary["cases"].items():
        case_lines.append(f"- `{name}`: {case_summary['diagnosis']}")
    priority = "yes" if any(
        case_summary["diagnosis"].startswith("candidate transition window")
        for case_summary in summary["cases"].values()
    ) else "no"
    findings_note_path.write_text(
        "\n".join(
            [
                f"# LB-12 pilot: {pilot_family}",
                "",
                f"- pilot config: `{config.get('config_path_hint', 'in-memory')}`",
                f"- artifact root: `{family_root}`",
                "- case diagnoses:",
                *case_lines,
                f"- prioritize next: `{priority}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    executed = sum(1 for r in rows if r["cache_status"] == "executed")
    cached = sum(1 for r in rows if r["cache_status"] == "cached")
    return {
        "pilot_family": pilot_family,
        "run_count": len(rows),
        "executed_count": executed,
        "cached_count": cached,
        "artifact_root": str(family_root),
        "manifest_path": str(manifest),
        "summary_path": str(summary_json),
        "plot_paths": plot_paths,
        "diagnosis": summary.get("diagnosis", "see case-level diagnoses"),
        "cases": summary["cases"],
    }


def run_all_lb12_pilots(output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    root = _repo_root()
    cfgs = [
        root / "configs" / "pilots" / "equilibrium_like_lambda_scan.json",
        root / "configs" / "pilots" / "driven_family_lambda_scan.json",
        root / "configs" / "pilots" / "null_family_control_scan.json",
        root / "configs" / "pilots" / "holonomy_control_lambda_scan.json",
    ]
    results = [run_pilot_scan(cfg, output_root=output_root, use_cache=use_cache) for cfg in cfgs]
    return {"pilots": results}
