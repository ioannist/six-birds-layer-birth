"""LB-17 affinity visibility and degeneracy audit."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import cache_reuse_allowed, computation_hash
from .campaigns import _git_code_version, _line_plot, _repo_root, _write_csv
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lenses import build_lens_family
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import macro_kernel, pushforward_matrix
from .protocols import resolve_run_settings
from .sweep import apply_closure_strength_control
from .substrates import build_substrate_family


METRIC_FIELDS = [
    "size",
    "bias",
    "analysis_mode",
    "lambda_grid_name",
    "closure_strength_lambda",
    "resolved_tau",
    "analysis_k",
    "closure_error",
    "objecthood_order",
    "staging_gap",
    "affinity",
    "macro_offdiag_mean_tau",
    "max_window_score",
    "max_slope_score",
    "cache_status",
    "manifest_path",
]


def _stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    return computation_hash(payload, length)


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def _build_lens_for_mode(mode: str, p: np.ndarray) -> np.ndarray:
    n = p.shape[0]
    if mode.startswith("manual_"):
        return np.asarray([0 if i < (n // 2) else 1 for i in range(n)], dtype=np.int64)
    if mode.startswith("diffusion_"):
        lens, _ = build_lens_family("diffusion_quantile_lens", P=p, target_k=2, tau=1)
        return np.asarray(lens, dtype=np.int64)
    raise ValueError(f"unknown analysis mode: {mode}")


def _tau_for_mode(mode: str) -> int:
    if mode.endswith("tau1"):
        return 1
    if mode.endswith("tau2"):
        return 2
    raise ValueError(f"unknown tau suffix for mode {mode}")


def _linfit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r_squared": float("nan")}
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(set(x)) != len(x):
        raise ValueError("linearity requires finite observations at distinct coordinates")
    scale = max(float(np.max(np.abs(y))), 1e-300)
    normalized = y / scale
    a_scaled, b_scaled = np.polyfit(x, normalized, 1)
    ss_res = float(np.sum((normalized - (a_scaled * x + b_scaled)) ** 2))
    ss_tot = float(np.sum((normalized - np.mean(normalized)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else max(0.0, 1.0 - ss_res / ss_tot)
    a, b = a_scaled * scale, b_scaled * scale
    return {"slope": float(a), "intercept": float(b), "r_squared": float(r2)}


def compute_window_scores(rows: list[dict[str, Any]], lambda_grid: list[float]) -> dict[str, Any]:
    by_l = {float(r["closure_strength_lambda"]): r for r in rows}
    lambdas = [float(x) for x in lambda_grid if float(x) in by_l]
    if len(lambdas) < 2:
        return {"max_window_score": 0.0, "argmax_interval": None}
    best = -1.0
    argmax = None
    for i in range(len(lambdas) - 1):
        l0, l1 = lambdas[i], lambdas[i + 1]
        r0, r1 = by_l[l0], by_l[l1]
        score = abs(observed_float(r1["objecthood_order"]) - observed_float(r0["objecthood_order"])) + abs(
            observed_float(r1["closure_error"]) - observed_float(r0["closure_error"])
        )
        if score > best:
            best = float(score)
            argmax = (float(l0), float(l1))
    return {"max_window_score": float(max(best, 0.0)), "argmax_interval": argmax}


def compute_slope_scores(rows: list[dict[str, Any]], lambda_grid: list[float]) -> dict[str, Any]:
    by_l = {float(r["closure_strength_lambda"]): r for r in rows}
    lambdas = [float(x) for x in lambda_grid if float(x) in by_l]
    if len(lambdas) < 2:
        return {"max_slope_score": 0.0, "argmax_interval": None}
    best = -1.0
    argmax = None
    for i in range(len(lambdas) - 1):
        l0, l1 = lambdas[i], lambdas[i + 1]
        dl = max(abs(l1 - l0), 1e-12)
        r0, r1 = by_l[l0], by_l[l1]
        score = (
            abs(observed_float(r1["objecthood_order"]) - observed_float(r0["objecthood_order"]))
            + abs(observed_float(r1["closure_error"]) - observed_float(r0["closure_error"]))
        ) / dl
        if score > best:
            best = float(score)
            argmax = (float(l0), float(l1))
    return {"max_slope_score": float(max(best, 0.0)), "argmax_interval": argmax}


def fit_lambda_linearity(rows: list[dict[str, Any]], observable_key: str) -> dict[str, float]:
    x = np.asarray([float(r["closure_strength_lambda"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r[observable_key]) for r in rows], dtype=np.float64)
    order = np.argsort(x)
    return _linfit(x[order], y[order])


def estimate_level_crossings(
    rows: list[dict[str, Any]],
    observable_key: str,
    target_values: list[float],
) -> dict[str, float | None]:
    vals = sorted(rows, key=lambda r: float(r["closure_strength_lambda"]))
    out: dict[str, float | None] = {}
    for t in target_values:
        out[str(float(t))] = None
        for i in range(len(vals) - 1):
            l0, l1 = float(vals[i]["closure_strength_lambda"]), float(vals[i + 1]["closure_strength_lambda"])
            y0, y1 = float(vals[i][observable_key]), float(vals[i + 1][observable_key])
            if (y0 <= t <= y1) or (y1 <= t <= y0):
                if y1 == y0:
                    out[str(float(t))] = l0  # a flat target interval has no unique crossing
                else:
                    frac = (t - y0) / (y1 - y0)
                    out[str(float(t))] = float(l0 + frac * (l1 - l0))
                break
    return out


def _rows_key(row: dict[str, Any]) -> tuple[int, float, str, str]:
    return (
        int(row["size"]),
        float(row["bias"]),
        str(row["analysis_mode"]),
        str(row["lambda_grid_name"]),
    )


def summarize_bias_visibility(rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    coarse_windows = diagnostics["window_scores"]["coarse"]
    dense_windows = diagnostics["window_scores"]["dense"]
    coarse_slopes = diagnostics["slope_scores"]["coarse"]
    dense_slopes = diagnostics["slope_scores"]["dense"]
    coarse_cross = diagnostics["level_crossings"]["coarse"]
    dense_cross = diagnostics["level_crossings"]["dense"]

    grid_resolution_binding = False
    for key in dense_windows.keys():
        cw = coarse_windows.get(key)
        dw = dense_windows[key]
        if cw is None:
            continue
        if cw["argmax_interval"] != dw["argmax_interval"]:
            grid_resolution_binding = True
            break
        cs = float(coarse_slopes[key]["max_slope_score"])
        ds = float(dense_slopes[key]["max_slope_score"])
        rel = abs(ds - cs) / max(abs(cs), 1e-12)
        if rel >= 0.10:
            grid_resolution_binding = True
            break
        c_has = any(v is not None for side in coarse_cross.get(key, {}).values() for v in side.values())
        d_has = any(v is not None for side in dense_cross.get(key, {}).values() for v in side.values())
        if d_has and not c_has:
            grid_resolution_binding = True
            break

    linearity = diagnostics["linearity_summary"]
    projector_dominated_bias_invariance = bool(rows) and bool(linearity)
    for size in sorted({int(k.split("|")[0]) for k in linearity.keys()}):
        for obs in ("closure_error", "objecthood_order"):
            slopes = []
            intercepts = []
            for bias in sorted({float(k.split("|")[1]) for k in linearity.keys() if int(k.split("|")[0]) == size}):
                key = f"{size}|{bias}|manual_tau1|dense"
                entry = linearity.get(key)
                if entry is None or not np.isfinite(entry[obs]["r_squared"]) or entry[obs]["r_squared"] < 0.999:
                    projector_dominated_bias_invariance = False
                    break
                slopes.append(float(entry[obs]["slope"]))
                intercepts.append(float(entry[obs]["intercept"]))
            if not projector_dominated_bias_invariance:
                break
            arr = np.asarray(slopes, dtype=np.float64)
            if len(arr) < 2 or not np.isfinite(arr).all():
                projector_dominated_bias_invariance = False
                break
            scale = float(np.max(np.abs(arr)))
            cv = 0.0 if scale == 0 else float(np.std(arr / scale) / max(abs(float(np.mean(arr / scale))), 1e-15))
            if cv >= 1e-6 or not np.isfinite(intercepts).all() or max(intercepts) - min(intercepts) >= 1e-6:
                projector_dominated_bias_invariance = False
                break
        if not projector_dominated_bias_invariance:
            break

    by_mode_visibility: dict[str, bool] = {}
    for mode in sorted({str(r["analysis_mode"]) for r in rows}):
        visible = False
        mode_rows = [r for r in rows if str(r["analysis_mode"]) == mode and str(r["lambda_grid_name"]) == "dense"]
        for size in sorted({int(r["size"]) for r in mode_rows}):
            size_rows = [r for r in mode_rows if int(r["size"]) == size]
            for lam in sorted({float(r["closure_strength_lambda"]) for r in size_rows}):
                lam_rows = [r for r in size_rows if float(r["closure_strength_lambda"]) == lam]
                sg_range = max(observed_float(r["staging_gap"]) for r in lam_rows) - min(observed_float(r["staging_gap"]) for r in lam_rows)
                mo_range = max(float(r["macro_offdiag_mean_tau"]) for r in lam_rows) - min(float(r["macro_offdiag_mean_tau"]) for r in lam_rows)
                if sg_range >= 0.05 or mo_range >= 0.02:
                    visible = True
                    break
            if visible:
                break
            for obs in ("closure_error", "objecthood_order"):
                slopes = []
                for bias in sorted({float(r["bias"]) for r in size_rows}):
                    k = f"{size}|{bias}|{mode}|dense"
                    slopes.append(float(linearity[k][obs]["slope"]))
                arr = np.asarray(slopes, dtype=np.float64)
                rel = (float(np.max(arr)) - float(np.min(arr))) / max(abs(float(np.mean(arr))), 1e-15)
                if rel >= 0.05:
                    visible = True
                    break
            if visible:
                break
            crossings = diagnostics["level_crossings"]["dense"]
            for target_key in ("objecthood_order", "closure_error"):
                baseline = crossings.get(f"{size}|0.0|{mode}|dense", {}).get(target_key, {})
                for t in baseline:
                    base = baseline[t]
                    if base is None:
                        continue
                    vals = []
                    for bias in sorted({float(r["bias"]) for r in size_rows if float(r["bias"]) > 0.0}):
                        v = crossings.get(f"{size}|{bias}|{mode}|dense", {}).get(target_key, {}).get(t)
                        if v is not None:
                            vals.append(float(v))
                    if vals and max(abs(v - float(base)) for v in vals) >= 0.02:
                        visible = True
                        break
                if visible:
                    break
            if visible:
                break
        by_mode_visibility[mode] = visible

    bias_visibility_beyond_affinity = any(by_mode_visibility.values())
    visible_modes = [m for m, v in by_mode_visibility.items() if v]
    only_nonbaseline = bool(visible_modes) and all(m != "manual_tau1" for m in visible_modes)

    if projector_dominated_bias_invariance and (not grid_resolution_binding):
        final_diagnosis = "structurally invisible in CE/M_obj under baseline coarse analysis"
    elif grid_resolution_binding or (bias_visibility_beyond_affinity and only_nonbaseline):
        final_diagnosis = "not visible on this grid / under this analysis"
    else:
        final_diagnosis = "not visible on this grid / under this analysis"

    return {
        "grid_resolution_binding": bool(grid_resolution_binding),
        "projector_dominated_bias_invariance": bool(projector_dominated_bias_invariance),
        "bias_visibility_beyond_affinity": bool(bias_visibility_beyond_affinity),
        "visibility_modes": by_mode_visibility,
        "final_diagnosis": final_diagnosis,
    }


def format_affinity_visibility_summary(diagnostics: dict[str, Any], verdicts: dict[str, Any]) -> dict[str, Any]:
    return {"diagnostics": diagnostics, "verdicts": verdicts}


def _build_raw_rows(config: dict[str, Any], runs_dir: Path, use_cache: bool) -> tuple[list[dict[str, Any]], int, int]:
    panel = config["panel"]
    rows: list[dict[str, Any]] = []
    executed = 0
    cached = 0
    for size in [int(x) for x in panel["sizes"]]:
        for bias in [float(x) for x in panel["bias_grid"]]:
            fw = 0.5 * float(panel["flow_total"]) + bias
            bw = 0.5 * float(panel["flow_total"]) - bias
            substrate = build_substrate_family(
                "driven_cycle_family",
                n=size,
                self_weight=float(panel["self_weight"]),
                forward_weight=float(fw),
                backward_weight=float(bw),
            )
            p_base = np.asarray(substrate["P"], dtype=np.float64)
            for mode in panel["analysis_modes"]:
                mode_name = str(mode["name"])
                tau = _tau_for_mode(mode_name)
                lens = _build_lens_for_mode(mode_name, p_base)
                k = int(np.max(lens)) + 1
                q = pushforward_matrix(lens, k)
                lift, _ = build_lift_family("uniform_lift_family", f=lens, k=k)
                u = np.asarray(lift, dtype=np.float64)
                for grid_name, lam_grid in panel["lambda_grids"].items():
                    for lam in [float(x) for x in lam_grid]:
                        spec = {
                            "panel_config": panel,
                            "size": size,
                            "bias": bias,
                            "analysis_mode": mode_name,
                            "lambda_grid_name": grid_name,
                            "lambda": lam,
                        }
                        run_id = f"vis_{_stable_hash(spec)}"
                        row_path = runs_dir / f"{run_id}.json"
                        if cache_reuse_allowed(use_cache) and row_path.exists():
                            row = json.loads(row_path.read_text(encoding="utf-8"))
                            row["cache_status"] = "cached"
                            cached += 1
                        else:
                            p = apply_closure_strength_control(
                                p_base,
                                closure_strength_lambda=lam,
                                Q_f=q,
                                U_f=u,
                                mode=str(panel["control_application_name"]),
                            )
                            tau_cfg = {
                                "run_id": run_id,
                                "tau_protocol": {"name": "fixed", "tau": tau},
                                "control": {"closure_strength_lambda": lam},
                            }
                            resolved_tau = int(resolve_run_settings(tau_cfg, P=p)["resolved_tau"])
                            bundle = default_metric_bundle(p, lens, tau=resolved_tau)
                            phat = macro_kernel(p, resolved_tau, q, u)
                            if phat.shape[0] == 2:
                                macro_offdiag = float(0.5 * (phat[0, 1] + phat[1, 0]))
                            else:
                                macro_offdiag = 0.0
                            row = {
                                "size": int(size),
                                "bias": float(bias),
                                "analysis_mode": mode_name,
                                "lambda_grid_name": str(grid_name),
                                "closure_strength_lambda": float(lam),
                                "resolved_tau": int(resolved_tau),
                                "analysis_k": int(bundle["analysis_k"]),
                                "closure_error": observed_float(bundle["closure_error"]),
                                "objecthood_order": observed_float(bundle["objecthood_order"]),
                                "staging_gap": observed_float(bundle["staging_gap"]),
                                "affinity": observed_float(bundle["affinity"]),
                                "macro_offdiag_mean_tau": float(macro_offdiag),
                                "max_window_score": 0.0,
                                "max_slope_score": 0.0,
                                "cache_status": "executed",
                                "manifest_path": "",
                            }
                            row_path.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
                            executed += 1
                        rows.append(row)
    return rows, executed, cached


def run_affinity_visibility_audit(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    config = _coerce_config(config_path_or_obj)
    root = _repo_root()
    if output_root is None:
        output_root = root / "results" / "campaigns"
    else:
        output_root = Path(output_root)
    artifact_root = output_root / str(config["artifact_subdir"])
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

    rows, executed_count, cached_count = _build_raw_rows(config, dirs["runs"], use_cache)

    grouped: dict[tuple[int, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_rows_key(row)].append(row)

    win_coarse: dict[str, Any] = {}
    win_dense: dict[str, Any] = {}
    slope_coarse: dict[str, Any] = {}
    slope_dense: dict[str, Any] = {}
    linearity_summary: dict[str, Any] = {}
    crossings_coarse: dict[str, Any] = {}
    crossings_dense: dict[str, Any] = {}
    for key, vals in grouped.items():
        size, bias, mode, grid_name = key
        grid = [float(x) for x in config["panel"]["lambda_grids"][grid_name]]
        ws = compute_window_scores(vals, grid)
        ss = compute_slope_scores(vals, grid)
        for row in vals:
            row["max_window_score"] = float(ws["max_window_score"])
            row["max_slope_score"] = float(ss["max_slope_score"])
        line_ce = fit_lambda_linearity(vals, "closure_error")
        line_m = fit_lambda_linearity(vals, "objecthood_order")
        lin_key = f"{size}|{bias}|{mode}|{grid_name}"
        linearity_summary[lin_key] = {
            "closure_error": line_ce,
            "objecthood_order": line_m,
        }
        c_obj = estimate_level_crossings(vals, "objecthood_order", [0.85, 0.90, 0.95])
        c_ce = estimate_level_crossings(vals, "closure_error", [0.05, 0.025, 0.01])
        cross_entry = {"objecthood_order": c_obj, "closure_error": c_ce}
        if grid_name == "coarse":
            win_coarse[lin_key] = ws
            slope_coarse[lin_key] = ss
            crossings_coarse[lin_key] = cross_entry
        else:
            win_dense[lin_key] = ws
            slope_dense[lin_key] = ss
            crossings_dense[lin_key] = cross_entry

    diagnostics = {
        "window_scores": {"coarse": win_coarse, "dense": win_dense},
        "slope_scores": {"coarse": slope_coarse, "dense": slope_dense},
        "linearity_summary": linearity_summary,
        "level_crossings": {"coarse": crossings_coarse, "dense": crossings_dense},
    }
    verdicts = summarize_bias_visibility(rows, diagnostics)
    final_summary = format_affinity_visibility_summary(diagnostics, verdicts)

    for row in rows:
        row["manifest_path"] = str((artifact_root / "manifest.json").relative_to(artifact_root))
    _write_csv(dirs["metrics"] / "metrics.csv", rows, METRIC_FIELDS)
    _write_csv(dirs["analysis"] / "raw_metrics.csv", rows, METRIC_FIELDS)
    (dirs["analysis"] / "linearity_summary.json").write_text(
        scientific_dumps(linearity_summary, indent=2) + "\n", encoding="utf-8"
    )
    (dirs["analysis"] / "level_crossings.json").write_text(
        scientific_dumps(diagnostics["level_crossings"], indent=2) + "\n", encoding="utf-8"
    )
    (dirs["analysis"] / "visibility_summary.json").write_text(
        scientific_dumps(final_summary, indent=2) + "\n", encoding="utf-8"
    )

    rep_size = int(config["panel"]["sizes"][-1])
    rep_rows = [r for r in rows if int(r["size"]) == rep_size and str(r["lambda_grid_name"]) == "dense"]
    biases = sorted({float(r["bias"]) for r in rep_rows})
    mean_aff_series = {}
    for mode in sorted({str(r["analysis_mode"]) for r in rep_rows}):
        mode_rows = [r for r in rep_rows if str(r["analysis_mode"]) == mode]
        mean_aff_series[mode] = [
            float(np.mean([observed_float(v["affinity"]) for v in mode_rows if float(v["bias"]) == b]))
            for b in biases
        ]
    _line_plot(dirs["plots"] / "affinity_vs_bias.png", biases, mean_aff_series)

    for mode in ("manual_tau1", "manual_tau2", "diffusion_tau1"):
        mode_rows = [r for r in rep_rows if str(r["analysis_mode"]) == mode]
        lambdas = sorted({float(r["closure_strength_lambda"]) for r in mode_rows})
        series: dict[str, list[float]] = {}
        for b in biases:
            b_rows = {float(r["closure_strength_lambda"]): r for r in mode_rows if float(r["bias"]) == b}
            series[f"b={b}:CE"] = [observed_float(b_rows[l]["closure_error"]) for l in lambdas]
            series[f"b={b}:M"] = [observed_float(b_rows[l]["objecthood_order"]) for l in lambdas]
        _line_plot(dirs["plots"] / f"ce_mobj_vs_lambda_{mode}.png", lambdas, series)

    slope_series: dict[str, list[float]] = {}
    for mode in ("manual_tau1", "manual_tau2", "diffusion_tau1"):
        vals = []
        for b in biases:
            key = f"{rep_size}|{b}|{mode}|dense"
            vals.append(float(slope_dense[key]["max_slope_score"]))
        slope_series[mode] = vals
    _line_plot(dirs["plots"] / "slope_scores_vs_bias.png", biases, slope_series)

    cross_series: dict[str, list[float]] = {}
    for mode in ("manual_tau1", "manual_tau2", "diffusion_tau1"):
        vals = []
        for b in biases:
            key = f"{rep_size}|{b}|{mode}|dense"
            v = diagnostics["level_crossings"]["dense"][key]["objecthood_order"]["0.9"]
            vals.append(0.0 if v is None else float(v))
        cross_series[mode] = vals
    _line_plot(dirs["plots"] / "level_crossings_vs_bias.png", biases, cross_series)

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    notes = dirs["notes"] / "findings.md"
    notes.write_text(
        "\n".join(
            [
                "# Affinity visibility audit findings",
                "",
                f"- grid_resolution_binding: `{verdicts['grid_resolution_binding']}`",
                f"- projector_dominated_bias_invariance: `{verdicts['projector_dominated_bias_invariance']}`",
                f"- bias_visibility_beyond_affinity: `{verdicts['bias_visibility_beyond_affinity']}`",
                f"- final_diagnosis: `{verdicts['final_diagnosis']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = dirs["env"] / "environment.json"
    env.write_text(
        scientific_dumps({"python": "unknown", "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )

    schema = load_schema(root / "configs" / "result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "affinity_visibility_audit",
        "bundle_id": str(config["campaign_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": str(config_snapshot.relative_to(artifact_root)),
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    findings_note = root / str(config["findings_note_path"])
    findings_note.parent.mkdir(parents=True, exist_ok=True)
    findings_note.write_text(
        "\n".join(
            [
                "# LB-17 affinity visibility audit",
                "",
                f"- config path: `{config['config_path_hint']}`",
                f"- artifact root: `{artifact_root}`",
                "- coarse vs dense grid comparison: completed across manual_tau1/manual_tau2/diffusion_tau1",
                f"- linearity results: `{linearity_summary}`",
                f"- level-crossing results: `{diagnostics['level_crossings']}`",
                f"- is the negative result mainly a grid-resolution problem? `{'yes' if verdicts['grid_resolution_binding'] else 'no'}`",
                f"- are CE/M_obj structurally bias-invariant under the baseline coarse analysis? `{'yes' if verdicts['projector_dominated_bias_invariance'] else 'no'}`",
                f"- interpretation update: `{verdicts['final_diagnosis']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "grid_resolution_binding": bool(verdicts["grid_resolution_binding"]),
        "projector_dominated_bias_invariance": bool(verdicts["projector_dominated_bias_invariance"]),
        "bias_visibility_beyond_affinity": bool(verdicts["bias_visibility_beyond_affinity"]),
        "final_diagnosis": str(verdicts["final_diagnosis"]),
        "artifact_root": str(artifact_root),
        "executed_count": int(executed_count),
        "cached_count": int(cached_count),
    }
