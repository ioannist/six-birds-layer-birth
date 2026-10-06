"""LB-17 holonomy crossover campaign orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import cache_reuse_allowed, computation_hash
from .campaigns import _git_code_version, _line_plot, _repo_root, _write_csv
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import macro_kernel, pushforward_matrix, uniform_lift_matrix, validate_lens
from .protocols import resolve_run_settings
from .sweep import apply_closure_strength_control
from .substrates import build_substrate_family


METRIC_FIELDS = [
    "case_name",
    "control_route",
    "closure_strength_lambda",
    "analysis_k",
    "resolved_tau",
    "closure_error",
    "objecthood_order",
    "staging_gap",
    "affinity",
    "holonomy",
    "candidate_birth_window_start",
    "candidate_birth_window_end",
    "birth_location_estimate",
    "cache_status",
    "manifest_path",
]


def _stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    return computation_hash(payload, length)


def _induced_fine_to_coarse_map(fine_lens: np.ndarray, coarse_lens: np.ndarray, coarse_k: int) -> np.ndarray:
    fine_k = int(np.max(fine_lens)) + 1
    mapping = np.full(fine_k, -1, dtype=np.int64)
    for fine_label in range(fine_k):
        idx = np.flatnonzero(fine_lens == fine_label)
        coarse_vals = np.unique(coarse_lens[idx])
        if coarse_vals.size != 1:
            raise ValueError("fine fibers must each map to a unique coarse label")
        mapping[fine_label] = int(coarse_vals[0])
    validate_lens(mapping, coarse_k)
    return mapping


def build_direct_and_routed_projectors(
    coarse_lens: np.ndarray | list[int],
    fine_lens: np.ndarray | list[int],
) -> dict[str, np.ndarray]:
    coarse = np.asarray(coarse_lens, dtype=np.int64)
    fine = np.asarray(fine_lens, dtype=np.int64)
    k_c = int(np.max(coarse)) + 1
    k_f = int(np.max(fine)) + 1
    q_c = pushforward_matrix(coarse, k_c)
    u_c, _ = build_lift_family("uniform_lift_family", f=coarse, k=k_c)
    u_f, _ = build_lift_family("uniform_lift_family", f=fine, k=k_f)
    g = _induced_fine_to_coarse_map(fine, coarse, k_c)
    u_g = uniform_lift_matrix(g, k_c)
    u_route = u_g @ u_f
    return {
        "Q_c": q_c,
        "U_c": np.asarray(u_c, dtype=np.float64),
        "U_route": np.asarray(u_route, dtype=np.float64),
        "Pi_direct": np.asarray(q_c @ u_c, dtype=np.float64),
        "Pi_route": np.asarray(q_c @ u_route, dtype=np.float64),
        "coarse_k": np.int64(k_c),
        "fine_k": np.int64(k_f),
    }


def estimate_birth_window(rows: list[dict[str, Any]], lambda_grid: list[float], threshold: float = 0.10) -> dict[str, Any]:
    vals = {float(r["closure_strength_lambda"]): r for r in rows}
    lambdas = [float(x) for x in lambda_grid if float(x) in vals]
    if len(lambdas) < 2:
        return {
            "candidate_birth_window_start": None,
            "candidate_birth_window_end": None,
            "birth_location_estimate": None,
            "max_window_score": 0.0,
            "window_present": False,
        }
    best_score = -1.0
    best_pair: tuple[float, float] | None = None
    for i in range(len(lambdas) - 1):
        l0, l1 = lambdas[i], lambdas[i + 1]
        r0, r1 = vals[l0], vals[l1]
        score = abs(observed_float(r1["objecthood_order"]) - observed_float(r0["objecthood_order"])) + abs(
            observed_float(r1["closure_error"]) - observed_float(r0["closure_error"])
        )
        if score > best_score:
            best_score = score
            best_pair = (l0, l1)
    if best_pair is None or best_score < float(threshold):
        return {
            "candidate_birth_window_start": None,
            "candidate_birth_window_end": None,
            "birth_location_estimate": None,
            "max_window_score": float(max(best_score, 0.0)),
            "window_present": False,
        }
    return {
        "candidate_birth_window_start": float(best_pair[0]),
        "candidate_birth_window_end": float(best_pair[1]),
        "birth_location_estimate": float(0.5 * (best_pair[0] + best_pair[1])),
        "max_window_score": float(best_score),
        "window_present": True,
    }


def build_holonomy_crossover_rows(config: dict[str, Any], runs_dir: Path, use_cache: bool = True) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    executed = 0
    cached = 0
    lambda_grid = [float(x) for x in config["lambda_grid"]]
    tau_protocol = dict(config["tau_protocol"])
    for case in config["cases"]:
        case_name = str(case["case_name"])
        substrate = build_substrate_family(
            "holonomy_control_family",
            fine_block_sizes_by_coarse=case["fine_block_sizes_by_coarse"],
            n_blocks=int(config["base_kernel"]["n_blocks"]),
            block_size=int(config["base_kernel"]["block_size"]),
            intra_block_weight=float(config["base_kernel"]["intra_block_weight"]),
            inter_block_weight=float(config["base_kernel"]["inter_block_weight"]),
            self_weight=float(config["base_kernel"]["self_weight"]),
        )
        p_base = np.asarray(substrate["P"], dtype=np.float64)
        coarse_lens = np.asarray(substrate["coarse_lens"], dtype=np.int64)
        fine_lens = np.asarray(substrate["fine_lens"], dtype=np.int64)
        proj = build_direct_and_routed_projectors(coarse_lens, fine_lens)
        q_c = np.asarray(proj["Q_c"], dtype=np.float64)
        u_c = np.asarray(proj["U_c"], dtype=np.float64)
        u_route = np.asarray(proj["U_route"], dtype=np.float64)
        k_c = int(proj["coarse_k"])
        for route in config["control_routes"]:
            route_name = str(route["route_name"])
            control_mode = str(route["control_application_name"])
            for lam in lambda_grid:
                spec = {
                    "config": config,
                    "campaign_id": str(config["campaign_id"]),
                    "case_name": case_name,
                    "control_route": route_name,
                    "lambda": lam,
                    "control_mode": control_mode,
                }
                run_id = f"xov_{_stable_hash(spec)}"
                row_path = runs_dir / f"{run_id}.json"
                if cache_reuse_allowed(use_cache) and row_path.exists():
                    row = json.loads(row_path.read_text(encoding="utf-8"))
                    row["cache_status"] = "cached"
                    cached += 1
                else:
                    p_ctrl = apply_closure_strength_control(
                        p_base,
                        closure_strength_lambda=lam,
                        Q_f=q_c,
                        U_f=u_c,
                        U_route=u_route,
                        mode=control_mode,
                    )
                    tau_cfg = {
                        "run_id": run_id,
                        "tau_protocol": tau_protocol,
                        "control": {"closure_strength_lambda": lam},
                    }
                    run_settings = resolve_run_settings(tau_cfg, P=p_ctrl)
                    tau = int(run_settings["resolved_tau"])
                    bundle = default_metric_bundle(
                        p_ctrl,
                        coarse_lens,
                        tau=tau,
                        holonomy_inputs={
                            "fine_lens": fine_lens,
                            "coarse_lens": coarse_lens,
                            "fine_k": int(np.max(fine_lens)) + 1,
                            "coarse_k": k_c,
                        },
                    )
                    row = {
                        "case_name": case_name,
                        "control_route": route_name,
                        "closure_strength_lambda": lam,
                        "analysis_k": int(bundle["analysis_k"]),
                        "resolved_tau": tau,
                        "closure_error": observed_float(bundle["closure_error"]),
                        "objecthood_order": observed_float(bundle["objecthood_order"]),
                        "staging_gap": observed_float(bundle["staging_gap"]),
                        "affinity": observed_float(bundle["affinity"]),
                        "holonomy": float(bundle["holonomy"]) if bundle["holonomy"] is not None else None,
                        "candidate_birth_window_start": None,
                        "candidate_birth_window_end": None,
                        "birth_location_estimate": None,
                        "cache_status": "executed",
                        "manifest_path": "",
                    }
                    row_path.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
                    executed += 1
                rows.append(row)
    return rows, executed, cached


def compare_direct_vs_routed(rows: list[dict[str, Any]], lambda_grid: list[float], threshold: float = 0.10) -> dict[str, Any]:
    by_curve: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["case_name"]), str(row["control_route"]))
        by_curve.setdefault(key, []).append(row)
    curve_summaries: dict[str, dict[str, Any]] = {}
    for (case, route), curve_rows in by_curve.items():
        curve_rows = sorted(curve_rows, key=lambda r: float(r["closure_strength_lambda"]))
        bw = estimate_birth_window(curve_rows, lambda_grid, threshold=threshold)
        for row in curve_rows:
            row["candidate_birth_window_start"] = bw["candidate_birth_window_start"]
            row["candidate_birth_window_end"] = bw["candidate_birth_window_end"]
            row["birth_location_estimate"] = bw["birth_location_estimate"]
        curve_summaries[f"{case}_{route}"] = {
            "case_name": case,
            "control_route": route,
            "mean_holonomy": float(np.mean([float(r["holonomy"]) for r in curve_rows if r["holonomy"] is not None])),
            "mean_affinity": float(np.mean([observed_float(r["affinity"]) for r in curve_rows])),
            "max_closure_error": float(np.max([observed_float(r["closure_error"]) for r in curve_rows])),
            "birth_window": [
                bw["candidate_birth_window_start"],
                bw["candidate_birth_window_end"],
            ],
            "birth_location_estimate": bw["birth_location_estimate"],
            "window_present": bw["window_present"],
            "max_window_score": bw["max_window_score"],
        }

    all_means_aff = [observed_float(v["mean_affinity"]) for v in curve_summaries.values()]
    no_fake_arrow = all(v < 1e-3 for v in all_means_aff)
    bal_r = curve_summaries["balanced_routed"]
    unbal_r = curve_summaries["unbalanced_routed"]
    holonomy_separation = (float(unbal_r["mean_holonomy"]) - float(bal_r["mean_holonomy"])) >= 0.10

    shift = False
    if bal_r["birth_location_estimate"] is not None and unbal_r["birth_location_estimate"] is not None:
        shift = abs(float(unbal_r["birth_location_estimate"]) - float(bal_r["birth_location_estimate"])) >= 0.05
    presence_flip = bool(bal_r["window_present"]) != bool(unbal_r["window_present"])
    b_rows = {(float(r["closure_strength_lambda"])): r for r in by_curve[("balanced", "routed")]}
    u_rows = {(float(r["closure_strength_lambda"])): r for r in by_curve[("unbalanced", "routed")]}
    shape_delta_max = 0.0
    for lam in sorted(set(b_rows.keys()) & set(u_rows.keys())):
        rb = b_rows[lam]
        ru = u_rows[lam]
        shape_delta = abs(observed_float(ru["objecthood_order"]) - observed_float(rb["objecthood_order"])) + abs(
            observed_float(ru["closure_error"]) - observed_float(rb["closure_error"])
        )
        shape_delta_max = max(shape_delta_max, shape_delta)
    shape_change = shape_delta_max >= 0.10

    crossover_success = bool(no_fake_arrow and holonomy_separation and (shift or presence_flip or shape_change))
    family_adequate = bool(crossover_success)
    if crossover_success:
        diagnosis = "holonomy acts like a crossover field: routed unbalanced case changes boundary/shape while Aff stays near zero"
    elif no_fake_arrow:
        diagnosis = "useful as no-fake-arrow control, but inadequate as crossover-field family under current construction"
    else:
        diagnosis = "fails no-fake-arrow requirement: affinity contamination prevents clean crossover interpretation"

    return {
        "curve_summaries": curve_summaries,
        "no_fake_arrow_condition": bool(no_fake_arrow),
        "holonomy_separation_condition": bool(holonomy_separation),
        "boundary_shift_condition": bool(shift),
        "window_presence_flip_condition": bool(presence_flip),
        "curve_shape_change_condition": bool(shape_change),
        "shape_delta_max": float(shape_delta_max),
        "crossover_field_success": bool(crossover_success),
        "family_adequate_for_crossover": bool(family_adequate),
        "diagnosis": diagnosis,
    }


def format_holonomy_crossover_summary(comparison_rows: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    return {
        "comparison": comparison_rows,
        "campaign_diagnosis": diagnosis,
    }


def run_holonomy_crossover_campaign(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        config = config_path_or_obj
    else:
        config = json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))

    if output_root is None:
        output_root = _repo_root() / "results" / "campaigns"
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

    rows, executed_count, cached_count = build_holonomy_crossover_rows(config, dirs["runs"], use_cache=use_cache)
    comparison = compare_direct_vs_routed(rows, [float(x) for x in config["lambda_grid"]], threshold=float(config.get("birth_window_threshold", 0.10)))
    diagnosis = {
        "did_holonomy_behave_like_crossover_field": bool(comparison["crossover_field_success"]),
        "family_adequate_for_crossover": bool(comparison["family_adequate_for_crossover"]),
        "diagnosis": str(comparison["diagnosis"]),
    }
    summary = format_holonomy_crossover_summary(comparison, diagnosis)

    for row in rows:
        row["manifest_path"] = str((artifact_root / "manifest.json").relative_to(artifact_root))
    _write_csv(dirs["analysis"] / "raw_metrics.csv", rows, METRIC_FIELDS)
    _write_csv(dirs["metrics"] / "metrics.csv", rows, METRIC_FIELDS)
    (dirs["analysis"] / "comparison_summary.json").write_text(scientific_dumps(summary, indent=2) + "\n", encoding="utf-8")

    by_case_route: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case_route.setdefault(f"{row['case_name']}_{row['control_route']}", []).append(row)
    lambda_x = sorted({float(r["closure_strength_lambda"]) for r in rows})
    ce_mobj_series: dict[str, list[float]] = {}
    hol_aff_series: dict[str, list[float]] = {}
    for key, vals in sorted(by_case_route.items()):
        vmap = {float(v["closure_strength_lambda"]): v for v in vals}
        ce_mobj_series[f"{key}:CE"] = [observed_float(vmap[l]["closure_error"]) for l in lambda_x]
        ce_mobj_series[f"{key}:M"] = [observed_float(vmap[l]["objecthood_order"]) for l in lambda_x]
        hol_aff_series[f"{key}:Hol"] = [float(vmap[l]["holonomy"]) for l in lambda_x]
        hol_aff_series[f"{key}:Aff"] = [observed_float(vmap[l]["affinity"]) for l in lambda_x]
    _line_plot(dirs["plots"] / "ce_mobj_vs_lambda_by_case.png", lambda_x, ce_mobj_series)
    _line_plot(dirs["plots"] / "hol_aff_vs_lambda_by_case.png", lambda_x, hol_aff_series)
    bw_series: dict[str, list[float]] = {}
    x_bw = [0.0, 1.0]
    for key in ["balanced_direct", "balanced_routed", "unbalanced_direct", "unbalanced_routed"]:
        loc = comparison["curve_summaries"][key]["birth_location_estimate"]
        val = observed_float(loc)
        bw_series[key] = [val, val]
    _line_plot(dirs["plots"] / "birth_window_comparison.png", x_bw, bw_series)

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    seeds = dirs["seeds"] / "seeds.json"
    seeds.write_text("[]\n", encoding="utf-8")
    notes = dirs["notes"] / "findings.md"
    notes.write_text(
        "\n".join(
            [
                "# Holonomy crossover campaign findings",
                "",
                f"- no_fake_arrow_condition: `{comparison['no_fake_arrow_condition']}`",
                f"- holonomy_separation_condition: `{comparison['holonomy_separation_condition']}`",
                f"- crossover_field_success: `{comparison['crossover_field_success']}`",
                f"- family_adequate_for_crossover: `{comparison['family_adequate_for_crossover']}`",
                f"- diagnosis: `{comparison['diagnosis']}`",
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
        "experiment_id": "holonomy_crossover_campaign",
        "bundle_id": str(config["campaign_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(_repo_root()),
        "config_snapshot_path": str(config_snapshot.relative_to(artifact_root)),
        "seed_list_path": str(seeds.relative_to(artifact_root)),
        "metrics_table_path": str((dirs["metrics"] / "metrics.csv").relative_to(artifact_root)),
        "plots_dir_path": str(dirs["plots"].relative_to(artifact_root)),
        "notes_file_path": str(notes.relative_to(artifact_root)),
        "environment_snapshot_path": str(env.relative_to(artifact_root)),
    }
    validate_manifest(manifest, schema)
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    findings_note = _repo_root() / str(config["findings_note_path"])
    findings_note.parent.mkdir(parents=True, exist_ok=True)
    findings_note.write_text(
        "\n".join(
            [
                "# LB-17 holonomy crossover campaign",
                "",
                f"- config path: `{config['config_path_hint']}`",
                f"- artifact root: `{artifact_root}`",
                "- cases: balanced_direct, balanced_routed, unbalanced_direct, unbalanced_routed",
                f"- summary metrics: `{comparison['curve_summaries']}`",
                f"- did holonomy behave like a crossover field rather than a fake arrow source? `{'yes' if comparison['crossover_field_success'] else 'no'}`",
                f"- family adequate for crossover campaign use: `{'yes' if comparison['family_adequate_for_crossover'] else 'no'}`",
                f"- diagnosis: `{comparison['diagnosis']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "campaign_name": str(config["campaign_id"]),
        "cases_run": ["balanced_direct", "balanced_routed", "unbalanced_direct", "unbalanced_routed"],
        "no_fake_arrow_condition": bool(comparison["no_fake_arrow_condition"]),
        "holonomy_separation_condition": bool(comparison["holonomy_separation_condition"]),
        "crossover_field_success": bool(comparison["crossover_field_success"]),
        "family_adequate_for_crossover": bool(comparison["family_adequate_for_crossover"]),
        "artifact_root": str(artifact_root),
        "executed_count": int(executed_count),
        "cached_count": int(cached_count),
    }


def build_holonomy_crossover_rows_v2(
    config: dict[str, Any],
    runs_dir: Path,
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    executed = 0
    cached = 0
    lambda_grid = [float(x) for x in config["lambda_grid"]]
    tau_protocol = dict(config["tau_protocol"])

    for case in config["cases"]:
        case_name = str(case["case_name"])
        substrate = build_substrate_family(
            "holonomy_interface_skew_family",
            fine_block_sizes_by_coarse=case["fine_block_sizes_by_coarse"],
            n_blocks=int(config["base_kernel"]["n_blocks"]),
            block_size=int(config["base_kernel"]["block_size"]),
            intra_block_weight=float(config["base_kernel"]["intra_block_weight"]),
            interface_intra_weight=float(config["base_kernel"]["interface_intra_weight"]),
            interface_cross_weight=float(config["base_kernel"]["interface_cross_weight"]),
            noninterface_cross_weight=float(config["base_kernel"]["noninterface_cross_weight"]),
            self_weight=float(config["base_kernel"]["self_weight"]),
            interface_position_in_block=int(config["base_kernel"]["interface_position_in_block"]),
        )
        p_base = np.asarray(substrate["P"], dtype=np.float64)
        coarse_lens = np.asarray(substrate["coarse_lens"], dtype=np.int64)
        fine_lens = np.asarray(substrate["fine_lens"], dtype=np.int64)
        proj = build_direct_and_routed_projectors(coarse_lens, fine_lens)
        q_c = np.asarray(proj["Q_c"], dtype=np.float64)
        u_c = np.asarray(proj["U_c"], dtype=np.float64)
        u_route = np.asarray(proj["U_route"], dtype=np.float64)
        k_c = int(proj["coarse_k"])
        for route in config["control_routes"]:
            route_name = str(route["route_name"])
            control_mode = str(route["control_application_name"])
            case_label = f"{case_name}_{route_name}"
            for lam in lambda_grid:
                spec = {
                    "config": config,
                    "campaign_id": str(config["campaign_id"]),
                    "case_name": case_label,
                    "lambda": lam,
                    "control_mode": control_mode,
                }
                run_id = f"xov2_{_stable_hash(spec)}"
                row_path = runs_dir / f"{run_id}.json"
                if cache_reuse_allowed(use_cache) and row_path.exists():
                    row = json.loads(row_path.read_text(encoding="utf-8"))
                    row["cache_status"] = "cached"
                    cached += 1
                else:
                    p_ctrl = apply_closure_strength_control(
                        p_base,
                        closure_strength_lambda=lam,
                        Q_f=q_c,
                        U_f=u_c,
                        U_route=u_route,
                        mode=control_mode,
                    )
                    tau_cfg = {
                        "run_id": run_id,
                        "tau_protocol": tau_protocol,
                        "control": {"closure_strength_lambda": lam},
                    }
                    run_settings = resolve_run_settings(tau_cfg, P=p_ctrl)
                    tau = int(run_settings["resolved_tau"])
                    bundle = default_metric_bundle(
                        p_ctrl,
                        coarse_lens,
                        tau=tau,
                        holonomy_inputs={
                            "fine_lens": fine_lens,
                            "coarse_lens": coarse_lens,
                            "fine_k": int(np.max(fine_lens)) + 1,
                            "coarse_k": k_c,
                        },
                    )
                    phat = macro_kernel(p_ctrl, tau, q_c, u_c)
                    macro_offdiag = float(0.5 * (phat[0, 1] + phat[1, 0]))
                    row = {
                        "case_name": case_label,
                        "control_route": route_name,
                        "closure_strength_lambda": lam,
                        "analysis_k": int(bundle["analysis_k"]),
                        "resolved_tau": tau,
                        "closure_error": observed_float(bundle["closure_error"]),
                        "objecthood_order": observed_float(bundle["objecthood_order"]),
                        "staging_gap": observed_float(bundle["staging_gap"]),
                        "affinity": observed_float(bundle["affinity"]),
                        "holonomy": float(bundle["holonomy"]) if bundle["holonomy"] is not None else None,
                        "macro_offdiag_mean_tau": macro_offdiag,
                        "candidate_birth_window_start": None,
                        "candidate_birth_window_end": None,
                        "birth_location_estimate": None,
                        "cache_status": "executed",
                        "manifest_path": "",
                    }
                    row_path.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
                    executed += 1
                rows.append(row)
    return rows, executed, cached


def compare_direct_vs_routed_v2(rows: list[dict[str, Any]], lambda_grid: list[float], threshold: float = 0.05) -> dict[str, Any]:
    by_curve: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_curve.setdefault(str(row["case_name"]), []).append(row)

    curve_summaries: dict[str, dict[str, Any]] = {}
    for case_name, curve_rows in by_curve.items():
        curve_rows = sorted(curve_rows, key=lambda r: float(r["closure_strength_lambda"]))
        bw = estimate_birth_window(curve_rows, lambda_grid, threshold=threshold)
        for row in curve_rows:
            row["candidate_birth_window_start"] = bw["candidate_birth_window_start"]
            row["candidate_birth_window_end"] = bw["candidate_birth_window_end"]
            row["birth_location_estimate"] = bw["birth_location_estimate"]
        curve_summaries[case_name] = {
            "case_name": case_name,
            "mean_holonomy": float(np.mean([float(r["holonomy"]) for r in curve_rows if r["holonomy"] is not None])),
            "mean_affinity": float(np.mean([observed_float(r["affinity"]) for r in curve_rows])),
            "max_closure_error": float(np.max([observed_float(r["closure_error"]) for r in curve_rows])),
            "mean_macro_offdiag_mean_tau": float(np.mean([float(r["macro_offdiag_mean_tau"]) for r in curve_rows])),
            "birth_window": [bw["candidate_birth_window_start"], bw["candidate_birth_window_end"]],
            "birth_location_estimate": bw["birth_location_estimate"],
            "window_present": bw["window_present"],
            "max_window_score": bw["max_window_score"],
            "curve_objecthood": [observed_float(r["objecthood_order"]) for r in curve_rows],
            "curve_closure_error": [observed_float(r["closure_error"]) for r in curve_rows],
        }

    no_fake_arrow = all(observed_float(v["mean_affinity"]) < 1e-6 for v in curve_summaries.values())
    hol_sep = (
        float(curve_summaries["unbalanced_routed_rev"]["mean_holonomy"])
        - float(curve_summaries["balanced_routed_rev"]["mean_holonomy"])
    ) >= 0.10
    bal = curve_summaries["balanced_routed_rev"]
    unb = curve_summaries["unbalanced_routed_rev"]
    shift = False
    if bal["birth_location_estimate"] is not None and unb["birth_location_estimate"] is not None:
        shift = abs(float(unb["birth_location_estimate"]) - float(bal["birth_location_estimate"])) >= 0.05
    presence_flip = bool(bal["window_present"]) != bool(unb["window_present"])
    ce_diff_max = max(abs(a - b) for a, b in zip(unb["curve_closure_error"], bal["curve_closure_error"]))
    m_diff_max = max(abs(a - b) for a, b in zip(unb["curve_objecthood"], bal["curve_objecthood"]))
    curve_delta = max(ce_diff_max, m_diff_max) >= 0.03
    macro_delta = abs(
        float(unb["mean_macro_offdiag_mean_tau"]) - float(bal["mean_macro_offdiag_mean_tau"])
    ) >= 0.02
    crossover_success = bool(no_fake_arrow and hol_sep and (shift or presence_flip or curve_delta or macro_delta))
    family_adequate = bool(crossover_success)
    if crossover_success:
        diagnosis = "v2 supports crossover-field interpretation: structural holonomy modifies coarse behavior while remaining no-fake-arrow."
    elif no_fake_arrow and hol_sep:
        diagnosis = "v2 is clean and structurally separated, but crossover-effect thresholds are not yet met."
    elif no_fake_arrow:
        diagnosis = "v2 remains no-fake-arrow but structural holonomy separation is insufficient."
    else:
        diagnosis = "v2 fails strict no-fake-arrow condition despite reversibleized routing."
    return {
        "curve_summaries": curve_summaries,
        "no_fake_arrow_condition": bool(no_fake_arrow),
        "holonomy_separation_condition": bool(hol_sep),
        "birth_location_shift_condition": bool(shift),
        "window_presence_flip_condition": bool(presence_flip),
        "curve_shape_delta_condition": bool(curve_delta),
        "macro_offdiag_delta_condition": bool(macro_delta),
        "max_curve_delta": float(max(ce_diff_max, m_diff_max)),
        "macro_offdiag_delta": float(
            abs(float(unb["mean_macro_offdiag_mean_tau"]) - float(bal["mean_macro_offdiag_mean_tau"]))
        ),
        "crossover_field_success": bool(crossover_success),
        "family_adequate_for_crossover": bool(family_adequate),
        "diagnosis": diagnosis,
    }


def format_holonomy_crossover_summary_v2(comparison_rows: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    return {
        "comparison": comparison_rows,
        "campaign_diagnosis": diagnosis,
    }


def run_holonomy_crossover_campaign_v2(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        config = config_path_or_obj
    else:
        config = json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))

    if output_root is None:
        output_root = _repo_root() / "results" / "campaigns"
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

    rows, executed_count, cached_count = build_holonomy_crossover_rows_v2(config, dirs["runs"], use_cache=use_cache)
    comparison = compare_direct_vs_routed_v2(
        rows,
        [float(x) for x in config["lambda_grid"]],
        threshold=float(config.get("birth_window_threshold", 0.05)),
    )
    diagnosis = {
        "did_holonomy_behave_like_crossover_field": bool(comparison["crossover_field_success"]),
        "family_adequate_for_crossover": bool(comparison["family_adequate_for_crossover"]),
        "diagnosis": str(comparison["diagnosis"]),
    }
    summary = format_holonomy_crossover_summary_v2(comparison, diagnosis)

    metric_fields = [
        "case_name",
        "control_route",
        "closure_strength_lambda",
        "analysis_k",
        "resolved_tau",
        "closure_error",
        "objecthood_order",
        "staging_gap",
        "affinity",
        "holonomy",
        "macro_offdiag_mean_tau",
        "candidate_birth_window_start",
        "candidate_birth_window_end",
        "birth_location_estimate",
        "cache_status",
        "manifest_path",
    ]
    for row in rows:
        row["manifest_path"] = str((artifact_root / "manifest.json").relative_to(artifact_root))
    _write_csv(dirs["analysis"] / "raw_metrics.csv", rows, metric_fields)
    _write_csv(dirs["metrics"] / "metrics.csv", rows, metric_fields)
    (dirs["analysis"] / "comparison_summary.json").write_text(scientific_dumps(summary, indent=2) + "\n", encoding="utf-8")

    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_name"]), []).append(row)
    lambda_x = sorted({float(r["closure_strength_lambda"]) for r in rows})
    ce_mobj_series: dict[str, list[float]] = {}
    hol_aff_series: dict[str, list[float]] = {}
    macro_series: dict[str, list[float]] = {}
    for key, vals in sorted(by_case.items()):
        vmap = {float(v["closure_strength_lambda"]): v for v in vals}
        ce_mobj_series[f"{key}:CE"] = [observed_float(vmap[l]["closure_error"]) for l in lambda_x]
        ce_mobj_series[f"{key}:M"] = [observed_float(vmap[l]["objecthood_order"]) for l in lambda_x]
        hol_aff_series[f"{key}:Hol"] = [float(vmap[l]["holonomy"]) for l in lambda_x]
        hol_aff_series[f"{key}:Aff"] = [observed_float(vmap[l]["affinity"]) for l in lambda_x]
        macro_series[key] = [float(vmap[l]["macro_offdiag_mean_tau"]) for l in lambda_x]
    _line_plot(dirs["plots"] / "ce_mobj_vs_lambda_by_case.png", lambda_x, ce_mobj_series)
    _line_plot(dirs["plots"] / "hol_aff_vs_lambda_by_case.png", lambda_x, hol_aff_series)
    _line_plot(dirs["plots"] / "macro_offdiag_vs_lambda_by_case.png", lambda_x, macro_series)
    bw_series: dict[str, list[float]] = {}
    for key in ["balanced_direct_rev", "balanced_routed_rev", "unbalanced_direct_rev", "unbalanced_routed_rev"]:
        loc = comparison["curve_summaries"][key]["birth_location_estimate"]
        val = observed_float(loc)
        bw_series[key] = [val, val]
    _line_plot(dirs["plots"] / "birth_window_comparison.png", [0.0, 1.0], bw_series)

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    seeds = dirs["seeds"] / "seeds.json"
    seeds.write_text("[]\n", encoding="utf-8")
    notes = dirs["notes"] / "findings.md"
    notes.write_text(
        "\n".join(
            [
                "# Holonomy crossover campaign v2 findings",
                "",
                f"- no_fake_arrow_condition: `{comparison['no_fake_arrow_condition']}`",
                f"- holonomy_separation_condition: `{comparison['holonomy_separation_condition']}`",
                f"- crossover_field_success: `{comparison['crossover_field_success']}`",
                f"- family_adequate_for_crossover: `{comparison['family_adequate_for_crossover']}`",
                f"- diagnosis: `{comparison['diagnosis']}`",
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
        "experiment_id": "holonomy_crossover_campaign_v2",
        "bundle_id": str(config["campaign_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(_repo_root()),
        "config_snapshot_path": str(config_snapshot.relative_to(artifact_root)),
        "seed_list_path": str(seeds.relative_to(artifact_root)),
        "metrics_table_path": str((dirs["metrics"] / "metrics.csv").relative_to(artifact_root)),
        "plots_dir_path": str(dirs["plots"].relative_to(artifact_root)),
        "notes_file_path": str(notes.relative_to(artifact_root)),
        "environment_snapshot_path": str(env.relative_to(artifact_root)),
    }
    validate_manifest(manifest, schema)
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    findings_note = _repo_root() / str(config["findings_note_path"])
    findings_note.parent.mkdir(parents=True, exist_ok=True)
    findings_note.write_text(
        "\n".join(
            [
                "# LB-17 holonomy crossover campaign v2",
                "",
                f"- config path: `{config['config_path_hint']}`",
                f"- artifact root: `{artifact_root}`",
                "- cases: balanced_direct_rev, balanced_routed_rev, unbalanced_direct_rev, unbalanced_routed_rev",
                f"- summary metrics: `{comparison['curve_summaries']}`",
                f"- did holonomy behave like a crossover field rather than a fake arrow source in v2? `{'yes' if comparison['crossover_field_success'] else 'no'}`",
                f"- family adequate for crossover campaign use in v2: `{'yes' if comparison['family_adequate_for_crossover'] else 'no'}`",
                "- v2 vs v1: reversibleized routed controls reduce arrow contamination and add interface-skew tau=2 sensitivity; crossover effect remains threshold-limited if verdict is negative.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "campaign_name": str(config["campaign_id"]),
        "cases_run": [
            "balanced_direct_rev",
            "balanced_routed_rev",
            "unbalanced_direct_rev",
            "unbalanced_routed_rev",
        ],
        "no_fake_arrow_condition": bool(comparison["no_fake_arrow_condition"]),
        "holonomy_separation_condition": bool(comparison["holonomy_separation_condition"]),
        "crossover_field_success": bool(comparison["crossover_field_success"]),
        "family_adequate_for_crossover": bool(comparison["family_adequate_for_crossover"]),
        "artifact_root": str(artifact_root),
        "executed_count": int(executed_count),
        "cached_count": int(cached_count),
    }
