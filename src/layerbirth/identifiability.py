"""LB-17 class-II identifiability and observable audit."""

from __future__ import annotations

from collections import defaultdict
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
from .scaling import bootstrap_collapse_fit, collapse_objective
from .sweep import apply_closure_strength_control
from .substrates import build_substrate_family


FIT_FIELDS = [
    "fit_name",
    "observable_name",
    "tau",
    "lambda_c_fit",
    "beta_over_nu_fit",
    "inv_nu_fit",
    "collapse_objective",
    "fit_underdetermined",
    "best_is_boundary",
    "plateau_count_10pct",
    "beta_span_10pct",
    "inv_nu_span_10pct",
    "lambda_span_10pct",
    "local_contrast_ratio",
]


def _stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    return computation_hash(payload, length)


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def _group_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(int(row["size"]), float(row["closure_strength_lambda"]))].append(row)
    out: list[dict[str, Any]] = []
    for (size, lam), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append(
            {
                "size": int(size),
                "closure_strength_lambda": float(lam),
                "analysis_k": int(vals[0].get("analysis_k", 2)),
                "order_mean": float(np.mean([observed_float(v["objecthood_order"]) for v in vals])),
                "closure_error_mean": float(np.mean([observed_float(v["closure_error"]) for v in vals])),
                "staging_gap_mean": float(np.mean([observed_float(v["staging_gap"]) for v in vals])),
                "affinity_mean": float(np.mean([observed_float(v["affinity"]) for v in vals])),
                "n_replicates": int(len(vals)),
            }
        )
    return out


def compute_objective_landscape(
    group_rows: list[dict[str, Any]],
    observable_key: str,
    fit_grid: dict[str, Any],
) -> dict[str, Any]:
    order_key = {
        "objecthood_order": "order_mean",
        "affinity": "affinity_mean",
    }[observable_key]
    points: list[dict[str, Any]] = []
    for lc in fit_grid["lambda_c_grid"]:
        for bon in fit_grid["beta_over_nu_grid"]:
            for inv in fit_grid["inv_nu_grid"]:
                obj = collapse_objective(
                    group_rows,
                    lambda_c=float(lc),
                    beta_over_nu=float(bon),
                    inv_nu=float(inv),
                    order_key=order_key,
                    n_bins=int(fit_grid.get("collapse_bins", 12)),
                )
                points.append(
                    {
                        "lambda_c": float(lc),
                        "beta_over_nu": float(bon),
                        "inv_nu": float(inv),
                        "objective": float(obj),
                    }
                )
    points.sort(key=lambda x: x["objective"])
    if not points or not np.isfinite(points[0]["objective"]):
        raise ValueError("no finite comparable objective on identifiability grid")
    return {"best_fit": points[0], "ranked_points": points, "num_points": len(points)}


def summarize_landscape_sharpness(
    landscape: dict[str, Any],
    fit_grid: dict[str, Any],
    tolerance_fraction: float = 0.10,
) -> dict[str, Any]:
    ranked = landscape["ranked_points"]
    best = ranked[0]
    best_obj = float(best["objective"])
    cutoff = best_obj * (1.0 + float(tolerance_fraction))
    plateau = [p for p in ranked if float(p["objective"]) <= cutoff + 1e-15]
    lambda_vals = sorted({float(p["lambda_c"]) for p in plateau})
    beta_vals = sorted({float(p["beta_over_nu"]) for p in plateau})
    inv_vals = sorted({float(p["inv_nu"]) for p in plateau})

    lambda_grid = [float(x) for x in fit_grid["lambda_c_grid"]]
    beta_grid = [float(x) for x in fit_grid["beta_over_nu_grid"]]
    inv_grid = [float(x) for x in fit_grid["inv_nu_grid"]]
    best_is_boundary = bool(
        np.isclose(float(best["beta_over_nu"]), min(beta_grid))
        or np.isclose(float(best["beta_over_nu"]), max(beta_grid))
        or np.isclose(float(best["inv_nu"]), min(inv_grid))
        or np.isclose(float(best["inv_nu"]), max(inv_grid))
    )
    second = ranked[1] if len(ranked) > 1 else None
    contrast = float(second["objective"] / max(best_obj, 1e-15)) if second is not None else float("inf")
    fit_underdetermined = bool(best_is_boundary or len(beta_vals) > 1 or len(inv_vals) > 1)
    return {
        "best_fit": best,
        "best_is_boundary": best_is_boundary,
        "plateau_count_10pct": int(len(plateau)),
        "lambda_span_10pct": float(max(lambda_vals) - min(lambda_vals)) if lambda_vals else 0.0,
        "beta_span_10pct": float(max(beta_vals) - min(beta_vals)) if beta_vals else 0.0,
        "inv_nu_span_10pct": float(max(inv_vals) - min(inv_vals)) if inv_vals else 0.0,
        "local_contrast_ratio": float(contrast),
        "fit_underdetermined": fit_underdetermined,
        "top_candidates": ranked[:10],
        "grid_axes": {
            "lambda_c_grid": lambda_grid,
            "beta_over_nu_grid": beta_grid,
            "inv_nu_grid": inv_grid,
        },
    }


def profile_landscape_axes(landscape: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    ranked = landscape["ranked_points"]
    by_lambda: dict[float, float] = {}
    by_beta: dict[float, float] = {}
    by_inv: dict[float, float] = {}
    for p in ranked:
        lc = float(p["lambda_c"])
        bo = float(p["beta_over_nu"])
        inv = float(p["inv_nu"])
        obj = float(p["objective"])
        by_lambda[lc] = min(obj, by_lambda.get(lc, float("inf")))
        by_beta[bo] = min(obj, by_beta.get(bo, float("inf")))
        by_inv[inv] = min(obj, by_inv.get(inv, float("inf")))
    return {
        "lambda_profile": [{"x": k, "objective_min": v} for k, v in sorted(by_lambda.items())],
        "beta_profile": [{"x": k, "objective_min": v} for k, v in sorted(by_beta.items())],
        "inv_nu_profile": [{"x": k, "objective_min": v} for k, v in sorted(by_inv.items())],
    }


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _restore_if_missing(root: Path) -> None:
    req = [
        root / "results/campaigns/class_i_equilibrium/analysis/primary_group_summary.csv",
        root / "results/campaigns/class_i_equilibrium/analysis/fit_summary.json",
        root / "results/campaigns/class_ii_driven/analysis/primary_group_summary.csv",
        root / "results/campaigns/class_ii_driven/analysis/primary_raw_metrics.csv",
        root / "results/campaigns/class_ii_driven/analysis/fit_summary.json",
        root / "results/campaigns/class_ii_driven/analysis/comparison_to_class_i.json",
    ]
    if not all(artifact_is_current(p) for p in req):
        run_class_i_campaign_driver(output_root=root / "results" / "campaigns", use_cache=True)
        run_class_ii_campaign_driver(output_root=root / "results" / "campaigns", use_cache=True)


def run_auxiliary_class_ii_tau2_panel(
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
    runs_dir = artifact_root / "runs_aux_tau2"
    runs_dir.mkdir(parents=True, exist_ok=True)

    panel = config["auxiliary_tau2_panel"]
    rows: list[dict[str, Any]] = []
    executed = 0
    cached = 0
    for size in [int(x) for x in panel["sizes"]]:
        substrate = build_substrate_family(
            "driven_cycle_family",
            n=size,
            self_weight=float(panel["self_weight"]),
            forward_weight=float(panel["forward_weight"]),
            backward_weight=float(panel["backward_weight"]),
        )
        p_base = np.asarray(substrate["P"], dtype=np.float64)
        lens = np.asarray([0 if i < (size // 2) else 1 for i in range(size)], dtype=np.int64)
        q = pushforward_matrix(lens, 2)
        u = np.asarray(build_lift_family("uniform_lift_family", f=lens, k=2)[0], dtype=np.float64)
        for lam in [float(x) for x in panel["lambda_grid"]]:
            run_id = f"idf_{_stable_hash({'panel_config': panel, 'size': size, 'lambda': lam, 'tau': 2})}"
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
                bundle = default_metric_bundle(p, lens, tau=2)
                row = {
                    "size": int(size),
                    "closure_strength_lambda": float(lam),
                    "analysis_k": int(bundle["analysis_k"]),
                    "objecthood_order": observed_float(bundle["objecthood_order"]),
                    "closure_error": observed_float(bundle["closure_error"]),
                    "staging_gap": observed_float(bundle["staging_gap"]),
                    "affinity": observed_float(bundle["affinity"]),
                    "cache_status": "executed",
                }
                row_path.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
                executed += 1
            rows.append(row)
    return {"raw_rows": rows, "group_rows": _group_rows(rows), "executed_count": executed, "cached_count": cached}


def _better_observable(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    c_obj = float(candidate["collapse_objective"])
    b_obj = float(baseline["collapse_objective"])
    c_ctr = float(candidate["local_contrast_ratio"])
    b_ctr = float(baseline["local_contrast_ratio"])
    cond1 = (b_obj / max(c_obj, 1e-15) >= 1.5) or (b_ctr / max(c_ctr, 1e-15) >= 1.25)
    cond2 = (not bool(candidate["fit_underdetermined"])) or bool(baseline["fit_underdetermined"])
    return bool(cond1 and cond2)


def compare_class_ii_observables(class_ii_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = class_ii_summaries["class_ii_mobj_tau1"]
    aff_better = _better_observable(class_ii_summaries["class_ii_aff_tau1"], base) or _better_observable(
        class_ii_summaries["class_ii_aff_tau2"], base
    )
    tau2_improves = _better_observable(class_ii_summaries["class_ii_mobj_tau2"], class_ii_summaries["class_ii_mobj_tau1"]) or _better_observable(
        class_ii_summaries["class_ii_aff_tau2"], class_ii_summaries["class_ii_aff_tau1"]
    )
    return {
        "class_ii_mobj_tau1_underdetermined": bool(base["fit_underdetermined"]),
        "affinity_better_than_mobj_for_class_ii": bool(aff_better),
        "tau2_improves_class_ii": bool(tau2_improves),
    }


def _framing(class_i: dict[str, Any], class_ii: dict[str, dict[str, Any]], cmp: dict[str, Any]) -> str:
    best_name = min(
        ["class_ii_mobj_tau1", "class_ii_aff_tau1", "class_ii_mobj_tau2", "class_ii_aff_tau2"],
        key=lambda n: float(class_ii[n]["collapse_objective"]),
    )
    best = class_ii[best_name]
    class_i_obj = float(class_i["collapse_objective"])
    class_ii_obj = float(best["collapse_objective"])
    distinct = (
        class_ii_obj <= class_i_obj / 1.5
        or abs(float(best["lambda_c_fit"]) - float(class_i["lambda_c_fit"])) >= 0.05
    )
    if distinct:
        return "distinct_classes_strengthened"
    if (
        (best_name.startswith("class_ii_aff"))
        and bool(cmp.get("class_separation_plausible", False))
    ):
        return "same_scaling_secondary_affinity_plausible"
    return "class_ii_still_underdetermined"


def format_identifiability_summary(
    class_i_summary: dict[str, Any],
    class_ii_summaries: dict[str, dict[str, Any]],
    verdicts: dict[str, Any],
) -> dict[str, Any]:
    return {"class_i": class_i_summary, "class_ii": class_ii_summaries, "verdicts": verdicts}


def _plot_landscape(path: Path, profile: list[dict[str, float]], label: str) -> None:
    xs = [float(x["x"]) for x in profile]
    ys = [float(x["objective_min"]) for x in profile]
    _line_plot(path, xs, {label: ys})


def run_class_ii_identifiability_audit(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    config = _coerce_config(config_path_or_obj)
    root = _repo_root()
    _restore_if_missing(root)
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
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    class_i_group = _load_csv(root / config["inputs"]["class_i_primary_group_summary"])
    class_i_fit = json.loads((root / config["inputs"]["class_i_fit_summary"]).read_text(encoding="utf-8"))
    class_ii_group = _load_csv(root / config["inputs"]["class_ii_primary_group_summary"])
    class_ii_raw = _load_csv(root / config["inputs"]["class_ii_primary_raw_metrics"])
    class_ii_fit = json.loads((root / config["inputs"]["class_ii_fit_summary"]).read_text(encoding="utf-8"))
    class_cmp = json.loads((root / config["inputs"]["class_ii_comparison"]).read_text(encoding="utf-8"))

    aux = run_auxiliary_class_ii_tau2_panel(config, output_root=output_root, use_cache=use_cache)
    aux_raw = aux["raw_rows"]
    aux_group = aux["group_rows"]

    fit_grid = config["fit_grid"]
    landscapes: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    profiles: dict[str, dict[str, Any]] = {}

    def _run(name: str, group_rows: list[dict[str, Any]], observable: str, tau: int) -> None:
        land = compute_objective_landscape(group_rows, observable, fit_grid)
        sharp = summarize_landscape_sharpness(land, fit_grid, tolerance_fraction=float(config["plateau_tolerance_fraction"]))
        prof = profile_landscape_axes(land)
        boot_input = []
        if observable == "objecthood_order":
            for r in group_rows:
                nrep = max(1, int(r.get("n_replicates", 1)))
                for _ in range(nrep):
                    boot_input.append(
                        {
                            "size": int(r["size"]),
                            "closure_strength_lambda": float(r["closure_strength_lambda"]),
                            "order_value": float(r["order_mean"]),
                        }
                    )
        else:
            for r in group_rows:
                nrep = max(1, int(r.get("n_replicates", 1)))
                for _ in range(nrep):
                    boot_input.append(
                        {
                            "size": int(r["size"]),
                            "closure_strength_lambda": float(r["closure_strength_lambda"]),
                            "order_value": observed_float(r["affinity_mean"]),
                        }
                    )
        try:
            boot = bootstrap_collapse_fit(
                boot_input,
                {
                    "lambda_c_grid": fit_grid["lambda_c_grid"],
                    "beta_over_nu_grid": fit_grid["beta_over_nu_grid"],
                    "inv_nu_grid": fit_grid["inv_nu_grid"],
                    "n_bins": int(fit_grid.get("collapse_bins", 12)),
                    "ci_percentiles": list(fit_grid.get("ci_percentiles", [5, 95])),
                },
                n_boot=int(fit_grid.get("bootstrap_replicates", 25)),
                seed=2026,
            )
            sharp["bootstrap"] = {
                "ci": boot["ci"],
                "successful_bootstrap_count": boot["successful_bootstrap_count"],
            }
        except Exception:
            sharp["bootstrap"] = {
                "ci": None,
                "successful_bootstrap_count": 0,
            }
        landscapes[name] = land
        profiles[name] = prof
        summaries[name] = {
            "fit_name": name,
            "observable_name": observable,
            "tau": int(tau),
            "lambda_c_fit": float(sharp["best_fit"]["lambda_c"]),
            "beta_over_nu_fit": float(sharp["best_fit"]["beta_over_nu"]),
            "inv_nu_fit": float(sharp["best_fit"]["inv_nu"]),
            "collapse_objective": float(sharp["best_fit"]["objective"]),
            "fit_underdetermined": bool(sharp["fit_underdetermined"]),
            "best_is_boundary": bool(sharp["best_is_boundary"]),
            "plateau_count_10pct": int(sharp["plateau_count_10pct"]),
            "beta_span_10pct": float(sharp["beta_span_10pct"]),
            "inv_nu_span_10pct": float(sharp["inv_nu_span_10pct"]),
            "lambda_span_10pct": float(sharp["lambda_span_10pct"]),
            "local_contrast_ratio": float(sharp["local_contrast_ratio"]),
            "top_candidates": sharp["top_candidates"],
            "bootstrap": sharp["bootstrap"],
        }

    _run("class_i_mobj_tau1", class_i_group, "objecthood_order", 1)
    _run("class_ii_mobj_tau1", class_ii_group, "objecthood_order", 1)
    _run("class_ii_aff_tau1", class_ii_group, "affinity", 1)
    _run("class_ii_mobj_tau2", aux_group, "objecthood_order", 2)
    _run("class_ii_aff_tau2", aux_group, "affinity", 2)

    cmp = compare_class_ii_observables(
        {
            "class_ii_mobj_tau1": summaries["class_ii_mobj_tau1"],
            "class_ii_aff_tau1": summaries["class_ii_aff_tau1"],
            "class_ii_mobj_tau2": summaries["class_ii_mobj_tau2"],
            "class_ii_aff_tau2": summaries["class_ii_aff_tau2"],
        }
    )
    framing = _framing(
        summaries["class_i_mobj_tau1"],
        {
            "class_ii_mobj_tau1": summaries["class_ii_mobj_tau1"],
            "class_ii_aff_tau1": summaries["class_ii_aff_tau1"],
            "class_ii_mobj_tau2": summaries["class_ii_mobj_tau2"],
            "class_ii_aff_tau2": summaries["class_ii_aff_tau2"],
        },
        class_cmp,
    )
    verdicts = {**cmp, "provisional_framing_recommendation": framing}
    ident = format_identifiability_summary(
        summaries["class_i_mobj_tau1"],
        {
            "class_ii_mobj_tau1": summaries["class_ii_mobj_tau1"],
            "class_ii_aff_tau1": summaries["class_ii_aff_tau1"],
            "class_ii_mobj_tau2": summaries["class_ii_mobj_tau2"],
            "class_ii_aff_tau2": summaries["class_ii_aff_tau2"],
        },
        verdicts,
    )

    for name in landscapes.keys():
        (dirs["analysis"] / f"{name}_landscape.json").write_text(
            scientific_dumps({"landscape": landscapes[name], "summary": summaries[name], "profiles": profiles[name]}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    _write_csv(
        dirs["analysis"] / "auxiliary_tau2_raw_metrics.csv",
        aux_raw,
        ["size", "closure_strength_lambda", "analysis_k", "objecthood_order", "closure_error", "staging_gap", "affinity", "cache_status"],
    )
    _write_csv(
        dirs["analysis"] / "auxiliary_tau2_group_summary.csv",
        aux_group,
        ["size", "closure_strength_lambda", "analysis_k", "order_mean", "closure_error_mean", "staging_gap_mean", "affinity_mean", "n_replicates"],
    )
    (dirs["analysis"] / "identifiability_summary.json").write_text(scientific_dumps(ident, indent=2) + "\n", encoding="utf-8")

    fit_rows = [summaries[n] for n in ["class_i_mobj_tau1", "class_ii_mobj_tau1", "class_ii_aff_tau1", "class_ii_mobj_tau2", "class_ii_aff_tau2"]]
    _write_csv(dirs["metrics"] / "metrics.csv", fit_rows, FIT_FIELDS)

    for name in ["class_i_mobj_tau1", "class_ii_mobj_tau1", "class_ii_aff_tau1", "class_ii_mobj_tau2", "class_ii_aff_tau2"]:
        _plot_landscape(
            dirs["plots"] / f"{name}_landscape.png",
            profiles[name]["lambda_profile"],
            f"{name}:lambda_profile",
        )
    _line_plot(
        dirs["plots"] / "observable_comparison_summary.png",
        [0, 1, 2, 3],
        {
            "mobj_tau1": [summaries["class_ii_mobj_tau1"]["collapse_objective"]] * 4,
            "aff_tau1": [summaries["class_ii_aff_tau1"]["collapse_objective"]] * 4,
            "mobj_tau2": [summaries["class_ii_mobj_tau2"]["collapse_objective"]] * 4,
            "aff_tau2": [summaries["class_ii_aff_tau2"]["collapse_objective"]] * 4,
        },
    )

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    notes = dirs["notes"] / "findings.md"
    notes.write_text(
        "\n".join(
            [
                "# Class-II identifiability audit findings",
                "",
                f"- class_ii_mobj_tau1_underdetermined: `{verdicts['class_ii_mobj_tau1_underdetermined']}`",
                f"- affinity_better_than_mobj_for_class_ii: `{verdicts['affinity_better_than_mobj_for_class_ii']}`",
                f"- tau2_improves_class_ii: `{verdicts['tau2_improves_class_ii']}`",
                f"- provisional_framing_recommendation: `{verdicts['provisional_framing_recommendation']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = dirs["env"] / "environment.json"
    env.write_text(scientific_dumps({"python": "unknown", "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_ii_identifiability_audit",
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
                "# LB-17 class-II identifiability audit",
                "",
                f"- config path: `{config['config_path_hint']}`",
                f"- artifact root: `{artifact_root}`",
                f"- best-fit summaries: `{fit_rows}`",
                f"- is class-II M_obj, tau=1 underdetermined? `{'yes' if verdicts['class_ii_mobj_tau1_underdetermined'] else 'no'}`",
                f"- is affinity a better scaling observable for class-II than M_obj under current evidence? `{'yes' if verdicts['affinity_better_than_mobj_for_class_ii'] else 'no'}`",
                f"- does tau=2 improve class-II identifiability? `{'yes' if verdicts['tau2_improves_class_ii'] else 'no'}`",
                f"- provisional framing recommendation: `{verdicts['provisional_framing_recommendation']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    best_class_ii_name = min(
        ["class_ii_mobj_tau1", "class_ii_aff_tau1", "class_ii_mobj_tau2", "class_ii_aff_tau2"],
        key=lambda n: float(summaries[n]["collapse_objective"]),
    )
    return {
        "best_class_ii_fit_name": best_class_ii_name,
        "class_ii_mobj_tau1_underdetermined": bool(verdicts["class_ii_mobj_tau1_underdetermined"]),
        "affinity_better_than_mobj_for_class_ii": bool(verdicts["affinity_better_than_mobj_for_class_ii"]),
        "tau2_improves_class_ii": bool(verdicts["tau2_improves_class_ii"]),
        "provisional_framing_recommendation": str(verdicts["provisional_framing_recommendation"]),
        "artifact_root": str(artifact_root),
        "executed_count": int(aux["executed_count"]),
        "cached_count": int(aux["cached_count"]),
    }
