"""LB-17 crossover decision gate and affinity-boundary reconnaissance."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import artifact_is_current, cache_reuse_allowed, computation_hash
from .campaigns import (
    _git_code_version,
    _line_plot,
    _repo_root,
    _write_csv,
    run_class_i_campaign_driver,
    run_class_ii_campaign_driver,
)
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import pushforward_matrix
from .protocols import resolve_run_settings
from .sweep import apply_closure_strength_control
from .substrates import build_substrate_family


METRIC_FIELDS = [
    "size",
    "bias",
    "closure_strength_lambda",
    "analysis_k",
    "resolved_tau",
    "closure_error",
    "objecthood_order",
    "staging_gap",
    "affinity",
    "candidate_birth_window_start",
    "candidate_birth_window_end",
    "birth_location_estimate",
    "cache_status",
    "manifest_path",
]


def _stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    return computation_hash(payload, length)


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def bias_to_cycle_weights(self_weight: float, flow_total: float, bias: float) -> tuple[float, float, float]:
    fw = 0.5 * float(flow_total) + float(bias)
    bw = 0.5 * float(flow_total) - float(bias)
    if fw < 0.0 or bw < 0.0:
        raise ValueError("bias too large: resulting forward/backward weight became negative")
    if self_weight < 0.0:
        raise ValueError("self_weight must be nonnegative")
    return float(self_weight), float(fw), float(bw)


def load_existing_class_summaries(
    class_i_fit_path: str | Path,
    class_ii_fit_path: str | Path,
    class_comparison_path: str | Path,
    *,
    restore_if_missing: bool = True,
) -> dict[str, Any]:
    p_i = Path(class_i_fit_path)
    p_ii = Path(class_ii_fit_path)
    p_cmp = Path(class_comparison_path)
    if restore_if_missing and (not artifact_is_current(p_i) or not artifact_is_current(p_ii) or not artifact_is_current(p_cmp)):
        root = _repo_root()
        run_class_i_campaign_driver(output_root=root / "results" / "campaigns", use_cache=True)
        run_class_ii_campaign_driver(output_root=root / "results" / "campaigns", use_cache=True)
    return {
        "class_i_fit": json.loads(Path(class_i_fit_path).read_text(encoding="utf-8")),
        "class_ii_fit": json.loads(Path(class_ii_fit_path).read_text(encoding="utf-8")),
        "class_comparison": json.loads(Path(class_comparison_path).read_text(encoding="utf-8")),
    }


def evaluate_crossover_necessity(
    class_i_fit: dict[str, Any],
    class_ii_fit: dict[str, Any],
    class_comparison: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    ratio_thr = observed_float(thresholds.get("affinity_ratio_min", 10.0))
    diff_thr = observed_float(thresholds.get("affinity_diff_min", 1e-2))
    lambda_sep_thr = float(thresholds.get("lambda_delta_min", 0.05))

    i_best = class_i_fit.get("best_fit", {})
    ii_best = class_ii_fit.get("best_fit", {})

    class_i_ok = bool(np.isfinite(float(i_best.get("objective", np.inf))))
    class_ii_ok = bool(np.isfinite(float(ii_best.get("objective", np.inf))))
    class_separation_plausible = bool(class_comparison.get("class_separation_plausible", False))

    affinity_ratio = observed_float(class_comparison.get("affinity_contrast_ratio", 0.0))
    affinity_diff = observed_float(class_comparison.get("affinity_contrast_difference", 0.0))
    affinity_strong = bool(affinity_ratio >= ratio_thr or affinity_diff >= diff_thr)

    delta_lambda = abs(float(class_comparison.get("delta_lambda_c_fit", 0.0)))
    lambda_or_aff_ok = bool(delta_lambda >= lambda_sep_thr or affinity_strong)

    crossover_not_required = bool(
        class_i_ok
        and class_ii_ok
        and class_separation_plausible
        and affinity_strong
        and lambda_or_aff_ok
    )
    crossover_load_bearing = not crossover_not_required
    diagnosis = (
        "current paper can stand on two-class separation without a continuous crossover family"
        if crossover_not_required
        else "crossover story remains load-bearing under current evidence"
    )

    return {
        "class_i_ok": class_i_ok,
        "class_ii_ok": class_ii_ok,
        "class_separation_plausible": class_separation_plausible,
        "affinity_contrast_ratio": affinity_ratio,
        "affinity_contrast_difference": affinity_diff,
        "delta_lambda_c_fit_abs": delta_lambda,
        "crossover_load_bearing": bool(crossover_load_bearing),
        "diagnosis": diagnosis,
        "thresholds_used": {
            "affinity_ratio_min": ratio_thr,
            "affinity_diff_min": diff_thr,
            "lambda_delta_min": lambda_sep_thr,
        },
    }


def estimate_boundary_windows(rows: list[dict[str, Any]], threshold: float = 0.10) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["size"]), float(row["bias"])), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for (size, bias), vals in grouped.items():
        vals = sorted(vals, key=lambda r: float(r["closure_strength_lambda"]))
        if len(vals) < 2:
            out[f"{size}|{bias}"] = {
                "size": int(size),
                "bias": float(bias),
                "candidate_birth_window_start": None,
                "candidate_birth_window_end": None,
                "birth_location_estimate": None,
                "window_present": False,
                "max_window_score": 0.0,
            }
            continue
        best_score = -1.0
        best_window: tuple[float, float] | None = None
        for i in range(len(vals) - 1):
            v0 = vals[i]
            v1 = vals[i + 1]
            score = abs(observed_float(v1["objecthood_order"]) - observed_float(v0["objecthood_order"])) + abs(
                observed_float(v1["closure_error"]) - observed_float(v0["closure_error"])
            )
            if score > best_score:
                best_score = score
                best_window = (float(v0["closure_strength_lambda"]), float(v1["closure_strength_lambda"]))
        if best_window is None or best_score < float(threshold):
            out[f"{size}|{bias}"] = {
                "size": int(size),
                "bias": float(bias),
                "candidate_birth_window_start": None,
                "candidate_birth_window_end": None,
                "birth_location_estimate": None,
                "window_present": False,
                "max_window_score": float(max(best_score, 0.0)),
            }
        else:
            out[f"{size}|{bias}"] = {
                "size": int(size),
                "bias": float(bias),
                "candidate_birth_window_start": float(best_window[0]),
                "candidate_birth_window_end": float(best_window[1]),
                "birth_location_estimate": float(0.5 * (best_window[0] + best_window[1])),
                "window_present": True,
                "max_window_score": float(best_score),
            }
    return out


def summarize_affinity_boundary(rows: list[dict[str, Any]], window_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_bias: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_bias.setdefault(float(row["bias"]), []).append(row)
    biases = sorted(by_bias.keys())
    by_bias_summary: list[dict[str, Any]] = []
    for b in biases:
        vals = by_bias[b]
        mean_aff = float(np.mean([observed_float(v["affinity"]) for v in vals]))
        ws = [w for w in window_summary.values() if float(w["bias"]) == b]
        present_any = any(bool(w["window_present"]) for w in ws)
        locs = [float(w["birth_location_estimate"]) for w in ws if w["birth_location_estimate"] is not None]
        by_bias_summary.append(
            {
                "bias": float(b),
                "mean_affinity": mean_aff,
                "window_present_any_size": present_any,
                "mean_birth_location_estimate": None if not locs else float(np.mean(locs)),
            }
        )

    aff_means = [observed_float(x["mean_affinity"]) for x in by_bias_summary]
    monotonic_tol = 1e-9
    domains = [{(int(r["size"]), float(r["closure_strength_lambda"])) for r in by_bias[b]} for b in biases]
    comparable = len(biases) >= 2 and 0.0 in biases and all(domains[i] == domains[0] and len(domains[i]) == len(by_bias[b]) for i,b in enumerate(biases))
    monotonic = comparable and all(np.isfinite(a) and a >= 0 for a in aff_means) and all((aff_means[i + 1] + monotonic_tol) >= aff_means[i] for i in range(len(aff_means) - 1))

    nonzero_has_window = any(
        (float(s["bias"]) > 0.0 and bool(s["window_present_any_size"])) for s in by_bias_summary
    )

    shift = False
    if by_bias_summary:
        base_locs = [float(w["birth_location_estimate"]) for w in window_summary.values() if float(w["bias"]) == 0.0 and w["birth_location_estimate"] is not None]
        for b in biases:
            if b <= 0.0:
                continue
            locs = [float(w["birth_location_estimate"]) for w in window_summary.values() if float(w["bias"]) == b and w["birth_location_estimate"] is not None]
            if base_locs and locs and abs(float(np.mean(locs)) - float(np.mean(base_locs))) >= 0.05:
                shift = True
                break

    presence_diff = False
    for size in sorted({int(r["size"]) for r in rows}):
        base = window_summary.get(f"{size}|0.0")
        for b in biases:
            if b <= 0.0:
                continue
            cur = window_summary.get(f"{size}|{b}")
            if base is None or cur is None:
                continue
            if bool(base["window_present"]) != bool(cur["window_present"]):
                presence_diff = True
                break
        if presence_diff:
            break

    curve_change = False
    for size in sorted({int(r["size"]) for r in rows}):
        base_rows = sorted(
            [r for r in rows if int(r["size"]) == size and float(r["bias"]) == 0.0],
            key=lambda x: float(x["closure_strength_lambda"]),
        )
        base_m = [observed_float(r["objecthood_order"]) for r in base_rows]
        base_c = [observed_float(r["closure_error"]) for r in base_rows]
        for b in biases:
            if b <= 0.0:
                continue
            cur_rows = sorted(
                [r for r in rows if int(r["size"]) == size and float(r["bias"]) == b],
                key=lambda x: float(x["closure_strength_lambda"]),
            )
            if not base_rows or [r["closure_strength_lambda"] for r in cur_rows] != [r["closure_strength_lambda"] for r in base_rows]:
                continue
            dm = max(abs(observed_float(c["objecthood_order"]) - m) for c, m in zip(cur_rows, base_m))
            dc = max(abs(observed_float(c["closure_error"]) - ce) for c, ce in zip(cur_rows, base_c))
            if max(dm, dc) >= 0.03:
                curve_change = True
                break
        if curve_change:
            break

    affinity_crossover_path_plausible = bool(monotonic and nonzero_has_window and (shift or presence_diff or curve_change))
    diagnosis = (
        "affinity appears to be a plausible replacement crossover field in this reconnaissance scan"
        if affinity_crossover_path_plausible
        else "affinity reconnaissance did not yet meet crossover-path plausibility criteria"
    )
    return {
        "by_bias_summary": by_bias_summary,
        "window_summary": window_summary,
        "affinity_monotonic_with_bias": bool(monotonic),
        "bias_panel_comparable": bool(comparable),
        "nonzero_bias_has_candidate_window": bool(nonzero_has_window),
        "birth_location_shift_condition": bool(shift),
        "window_presence_difference_condition": bool(presence_diff),
        "curve_shape_change_condition": bool(curve_change),
        "affinity_crossover_path_plausible": bool(affinity_crossover_path_plausible),
        "diagnosis": diagnosis,
    }


def run_affinity_boundary_scan(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    config = _coerce_config(config_path_or_obj)
    panel = config["affinity_scan_panel"]
    if output_root is None:
        output_root = _repo_root() / "results" / "campaigns"
    else:
        output_root = Path(output_root)
    artifact_root = output_root / str(config["artifact_subdir"])
    runs_dir = artifact_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    executed = 0
    cached = 0
    for size in [int(x) for x in panel["sizes"]]:
        lens = np.asarray([0 if i < (size // 2) else 1 for i in range(size)], dtype=np.int64)
        q = pushforward_matrix(lens, 2)
        lift, _ = build_lift_family(panel["lift_name"], f=lens, k=2)
        u = np.asarray(lift, dtype=np.float64)
        for bias in [float(x) for x in panel["bias_grid"]]:
            self_w, fw, bw = bias_to_cycle_weights(
                float(panel["self_weight"]),
                float(panel["flow_total"]),
                bias,
            )
            substrate = build_substrate_family(
                "driven_cycle_family",
                n=size,
                self_weight=self_w,
                forward_weight=fw,
                backward_weight=bw,
            )
            p_base = np.asarray(substrate["P"], dtype=np.float64)
            for lam in [float(x) for x in panel["lambda_grid"]]:
                spec = {"panel_config": panel, "size": size, "bias": bias, "lambda": lam}
                run_id = f"aff_{_stable_hash(spec)}"
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
                        mode=panel["control_application_name"],
                    )
                    tau_cfg = {
                        "run_id": run_id,
                        "tau_protocol": dict(panel["tau_protocol"]),
                        "control": {"closure_strength_lambda": lam},
                    }
                    run_settings = resolve_run_settings(tau_cfg, P=p)
                    tau = int(run_settings["resolved_tau"])
                    bundle = default_metric_bundle(p, lens, tau=tau)
                    row = {
                        "size": size,
                        "bias": bias,
                        "closure_strength_lambda": lam,
                        "analysis_k": int(bundle["analysis_k"]),
                        "resolved_tau": tau,
                        "closure_error": observed_float(bundle["closure_error"]),
                        "objecthood_order": observed_float(bundle["objecthood_order"]),
                        "staging_gap": observed_float(bundle["staging_gap"]),
                        "affinity": observed_float(bundle["affinity"]),
                        "candidate_birth_window_start": None,
                        "candidate_birth_window_end": None,
                        "birth_location_estimate": None,
                        "cache_status": "executed",
                        "manifest_path": "",
                    }
                    row_path.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
                    executed += 1
                rows.append(row)

    windows = estimate_boundary_windows(rows, threshold=float(config["decision_thresholds"]["window_score_threshold"]))
    for row in rows:
        key = f"{int(row['size'])}|{float(row['bias'])}"
        w = windows[key]
        row["candidate_birth_window_start"] = w["candidate_birth_window_start"]
        row["candidate_birth_window_end"] = w["candidate_birth_window_end"]
        row["birth_location_estimate"] = w["birth_location_estimate"]

    summary = summarize_affinity_boundary(rows, windows)
    return {
        "rows": rows,
        "summary": summary,
        "executed_count": int(executed),
        "cached_count": int(cached),
        "artifact_root": str(artifact_root),
    }


def format_crossover_decision_summary(decision_audit: dict[str, Any], boundary_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_audit": decision_audit,
        "affinity_boundary_summary": boundary_summary,
    }


def run_crossover_decision_affinity_boundary(
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

    inp = config["decision_audit_inputs"]
    summaries = load_existing_class_summaries(
        root / inp["class_i_fit_summary_path"],
        root / inp["class_ii_fit_summary_path"],
        root / inp["class_comparison_path"],
        restore_if_missing=True,
    )
    decision_audit = evaluate_crossover_necessity(
        summaries["class_i_fit"],
        summaries["class_ii_fit"],
        summaries["class_comparison"],
        config["decision_thresholds"],
    )

    scan = run_affinity_boundary_scan(config, output_root=output_root, use_cache=use_cache)
    rows = scan["rows"]
    for row in rows:
        row["manifest_path"] = str((artifact_root / "manifest.json").relative_to(artifact_root))
    boundary_summary = scan["summary"]
    combined = format_crossover_decision_summary(decision_audit, boundary_summary)

    _write_csv(dirs["metrics"] / "metrics.csv", rows, METRIC_FIELDS)
    _write_csv(dirs["analysis"] / "affinity_boundary_raw_metrics.csv", rows, METRIC_FIELDS)
    (dirs["analysis"] / "decision_audit.json").write_text(scientific_dumps(decision_audit, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "affinity_boundary_summary.json").write_text(scientific_dumps(boundary_summary, indent=2) + "\n", encoding="utf-8")

    by_bias = {float(x["bias"]): observed_float(x["mean_affinity"]) for x in boundary_summary["by_bias_summary"]}
    biases = sorted(by_bias.keys())
    _line_plot(dirs["plots"] / "affinity_vs_bias.png", biases, {"mean_affinity": [by_bias[b] for b in biases]})

    panel = config["affinity_scan_panel"]
    rep_size = int(panel["sizes"][-1])
    rep_rows = [r for r in rows if int(r["size"]) == rep_size]
    lambda_grid = sorted({float(r["closure_strength_lambda"]) for r in rep_rows})
    ce_m_series: dict[str, list[float]] = {}
    for b in sorted({float(r["bias"]) for r in rep_rows}):
        b_rows = {float(r["closure_strength_lambda"]): r for r in rep_rows if float(r["bias"]) == b}
        ce_m_series[f"bias={b}:CE"] = [observed_float(b_rows[l]["closure_error"]) for l in lambda_grid]
        ce_m_series[f"bias={b}:M"] = [observed_float(b_rows[l]["objecthood_order"]) for l in lambda_grid]
    _line_plot(dirs["plots"] / "ce_mobj_vs_lambda_by_bias.png", lambda_grid, ce_m_series)

    b_loc_series = []
    for b in biases:
        locs = [
            float(w["birth_location_estimate"])
            for w in boundary_summary["window_summary"].values()
            if float(w["bias"]) == b and w["birth_location_estimate"] is not None
        ]
        b_loc_series.append(0.0 if not locs else float(np.mean(locs)))
    _line_plot(dirs["plots"] / "birth_location_vs_bias.png", biases, {"birth_location_estimate": b_loc_series})

    cmp = summaries["class_comparison"]
    _line_plot(
        dirs["plots"] / "class_i_vs_class_ii_decision_panel.png",
        [0.0, 1.0],
        {
            "class_i_aff": [observed_float(cmp.get("class_i_affinity_window_mean", 0.0))] * 2,
            "class_ii_aff": [observed_float(cmp.get("class_ii_affinity_window_mean", 0.0))] * 2,
            "class_ii_shadow_aff": [observed_float(cmp.get("class_ii_shadow_affinity_window_mean", 0.0))] * 2,
        },
    )

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    notes = dirs["notes"] / "findings.md"
    notes.write_text(
        "\n".join(
            [
                "# Crossover decision and affinity boundary findings",
                "",
                f"- crossover_load_bearing: `{decision_audit['crossover_load_bearing']}`",
                f"- class_separation_plausible: `{decision_audit['class_separation_plausible']}`",
                f"- affinity_crossover_path_plausible: `{boundary_summary['affinity_crossover_path_plausible']}`",
                f"- decision diagnosis: `{decision_audit['diagnosis']}`",
                f"- boundary diagnosis: `{boundary_summary['diagnosis']}`",
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
        "experiment_id": "crossover_decision_affinity_boundary",
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

    note_path = root / str(config["findings_note_path"])
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "\n".join(
            [
                "# LB-17 crossover decision and affinity-boundary reconnaissance",
                "",
                f"- config path: `{config['config_path_hint']}`",
                f"- artifact root: `{artifact_root}`",
                f"- class-I / class-II audit outcome: `{decision_audit}`",
                f"- affinity-boundary scan outcome: `{boundary_summary}`",
                f"- is a crossover family load-bearing for the current paper? `{'yes' if decision_audit['crossover_load_bearing'] else 'no'}`",
                f"- is direct affinity the right replacement crossover field to pursue next? `{'yes' if boundary_summary['affinity_crossover_path_plausible'] else 'no'}`",
                "- affinity route is cleaner mechanistically than holonomy route because bias directly controls directional imbalance and avoids routed-projector structural ambiguity.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "crossover_load_bearing": bool(decision_audit["crossover_load_bearing"]),
        "class_separation_plausible": bool(decision_audit["class_separation_plausible"]),
        "affinity_crossover_path_plausible": bool(boundary_summary["affinity_crossover_path_plausible"]),
        "artifact_root": str(artifact_root),
        "executed_count": int(scan["executed_count"]),
        "cached_count": int(scan["cached_count"]),
    }
