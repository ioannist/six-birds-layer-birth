"""LB-17 FSS stability and size-extension audit."""

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
from .scaling import collapse_objective
from .sweep import apply_closure_strength_control
from .substrates import build_substrate_family


METRIC_FIELDS = [
    "fit_name",
    "class_name",
    "size_panel",
    "fit_grid_name",
    "lambda_c_fit",
    "beta_over_nu_fit",
    "inv_nu_fit",
    "collapse_objective",
    "best_is_boundary",
    "plateau_count_10pct",
    "lambda_span_10pct",
    "beta_span_10pct",
    "inv_nu_span_10pct",
    "local_contrast_ratio",
    "grid_spec_sensitive",
    "size128_helpful",
]


def _stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    return computation_hash(payload, length)


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _compute_landscape(group_rows: list[dict[str, Any]], fit_grid: dict[str, Any]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for lc in fit_grid["lambda_c_grid"]:
        for bon in fit_grid["beta_over_nu_grid"]:
            for inv in fit_grid["inv_nu_grid"]:
                obj = collapse_objective(
                    group_rows,
                    lambda_c=float(lc),
                    beta_over_nu=float(bon),
                    inv_nu=float(inv),
                    order_key="order_mean",
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
        raise ValueError("no finite comparable curves in the landscape grid")
    return {"best_fit": points[0], "ranked_points": points, "num_points": len(points)}


def _summarize_landscape(
    landscape: dict[str, Any],
    fit_grid: dict[str, Any],
    tolerance_fraction: float = 0.10,
) -> dict[str, Any]:
    ranked = landscape["ranked_points"]
    best = ranked[0]
    best_obj = float(best["objective"])
    cutoff = best_obj * (1.0 + float(tolerance_fraction))
    plateau = [p for p in ranked if float(p["objective"]) <= cutoff + 1e-15]
    lvals = sorted({float(p["lambda_c"]) for p in plateau})
    bvals = sorted({float(p["beta_over_nu"]) for p in plateau})
    ivals = sorted({float(p["inv_nu"]) for p in plateau})
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
    return {
        "best_fit": best,
        "best_is_boundary": best_is_boundary,
        "plateau_count_10pct": int(len(plateau)),
        "lambda_span_10pct": float(max(lvals) - min(lvals)) if lvals else 0.0,
        "beta_span_10pct": float(max(bvals) - min(bvals)) if bvals else 0.0,
        "inv_nu_span_10pct": float(max(ivals) - min(ivals)) if ivals else 0.0,
        "local_contrast_ratio": float(contrast),
        "top_candidates": ranked[:10],
    }


def compute_landscape_suite(
    group_rows: list[dict[str, Any]],
    fit_grid_suite: dict[str, dict[str, Any]],
    observable_key: str = "order_mean",
) -> dict[str, dict[str, Any]]:
    if observable_key != "order_mean":
        raise ValueError("this audit supports observable_key='order_mean' only")
    out: dict[str, dict[str, Any]] = {}
    for grid_name, grid in fit_grid_suite.items():
        land = _compute_landscape(group_rows, grid)
        summ = _summarize_landscape(land, grid, tolerance_fraction=0.10)
        out[grid_name] = {"landscape": land, "summary": summ, "fit_grid": grid}
    return out


def _grid_step(values: list[float]) -> float:
    vals = sorted(float(x) for x in values)
    if len(vals) < 2:
        return 0.0
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    return float(min(diffs))


def compare_grid_sensitivity(landscape_suite: dict[str, dict[str, Any]]) -> dict[str, Any]:
    orig = landscape_suite["original_grid"]["summary"]
    ident = landscape_suite["ident_grid"]["summary"]
    union = landscape_suite["union_grid"]["summary"]
    og = landscape_suite["original_grid"]["fit_grid"]

    step_l = _grid_step(list(og["lambda_c_grid"]))
    step_b = _grid_step(list(og["beta_over_nu_grid"]))
    step_i = _grid_step(list(og["inv_nu_grid"]))

    def _shift(a: dict[str, Any], b: dict[str, Any], key: str) -> float:
        return abs(float(a["best_fit"][key]) - float(b["best_fit"][key]))

    shifts = {
        "orig_to_ident": {
            "lambda_c": _shift(orig, ident, "lambda_c"),
            "beta_over_nu": _shift(orig, ident, "beta_over_nu"),
            "inv_nu": _shift(orig, ident, "inv_nu"),
        },
        "orig_to_union": {
            "lambda_c": _shift(orig, union, "lambda_c"),
            "beta_over_nu": _shift(orig, union, "beta_over_nu"),
            "inv_nu": _shift(orig, union, "inv_nu"),
        },
    }
    boundary_flip = bool(
        (orig["best_is_boundary"] != ident["best_is_boundary"])
        or (orig["best_is_boundary"] != union["best_is_boundary"])
    )
    spec_sensitive = bool(
        shifts["orig_to_ident"]["lambda_c"] > step_l
        or shifts["orig_to_union"]["lambda_c"] > step_l
        or shifts["orig_to_ident"]["beta_over_nu"] > step_b
        or shifts["orig_to_union"]["beta_over_nu"] > step_b
        or shifts["orig_to_ident"]["inv_nu"] > step_i
        or shifts["orig_to_union"]["inv_nu"] > step_i
        or boundary_flip
    )
    return {
        "grid_spec_sensitive": spec_sensitive,
        "shifts": shifts,
        "original_steps": {"lambda_c": step_l, "beta_over_nu": step_b, "inv_nu": step_i},
        "boundary_flip": boundary_flip,
    }


def compare_size_extension(old_suite: dict[str, Any], extended_suite: dict[str, Any]) -> dict[str, Any]:
    old = old_suite["summary"]
    new = extended_suite["summary"]
    plateau_reduction = float(old["plateau_count_10pct"] - new["plateau_count_10pct"]) / max(
        float(old["plateau_count_10pct"]), 1.0
    )
    inv_span_reduction = float(old["inv_nu_span_10pct"] - new["inv_nu_span_10pct"]) / max(
        float(old["inv_nu_span_10pct"]), 1e-15
    )
    beta_span_reduction = float(old["beta_span_10pct"] - new["beta_span_10pct"]) / max(
        float(old["beta_span_10pct"]), 1e-15
    )
    boundary_release = bool(old["best_is_boundary"] and (not new["best_is_boundary"]))
    contrast_improve = float(old["local_contrast_ratio"] - new["local_contrast_ratio"]) / max(
        float(old["local_contrast_ratio"]), 1e-15
    )
    helpful = bool(
        plateau_reduction >= 0.25
        or inv_span_reduction >= 0.25
        or beta_span_reduction >= 0.25
        or boundary_release
        or contrast_improve >= 0.25
    )
    return {
        "size128_helpful": helpful,
        "plateau_reduction_frac": plateau_reduction,
        "inv_span_reduction_frac": inv_span_reduction,
        "beta_span_reduction_frac": beta_span_reduction,
        "boundary_release": boundary_release,
        "contrast_improve_frac": contrast_improve,
    }


def run_auxiliary_size_extension_panels(
    config: dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    root = _repo_root()
    if output_root is None:
        output_root = root / "results" / "campaigns"
    else:
        output_root = Path(output_root)
    artifact_root = output_root / str(config["artifact_subdir"])
    runs_dir = artifact_root / "runs_size128"
    runs_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {"class_i_raw": [], "class_ii_raw": [], "executed_count": 0, "cached_count": 0}

    # Class I size-128 panel
    c1 = config["class_i_auxiliary_size128"]
    substrate = build_substrate_family(
        "reversible_block_family",
        n_blocks=int(c1["n_blocks"]),
        block_size=int(c1["block_size"]),
        intra_block_weight=float(c1["intra_block_weight"]),
        inter_block_weight=float(c1["inter_block_weight"]),
        self_weight=float(c1["self_weight"]),
    )
    p_base = np.asarray(substrate["P"], dtype=np.float64)
    lens = np.asarray(substrate["block_lens"], dtype=np.int64)
    q = pushforward_matrix(lens, 2)
    u = np.asarray(build_lift_family("uniform_lift_family", f=lens, k=2)[0], dtype=np.float64)
    for lam in [float(x) for x in c1["lambda_grid"]]:
        run_id = f"sz1_{_stable_hash({'class': 'i', 'panel_config': c1, 'lambda': lam})}"
        row_path = runs_dir / f"{run_id}.json"
        if cache_reuse_allowed(use_cache) and row_path.exists():
            row = json.loads(row_path.read_text(encoding="utf-8"))
            row["cache_status"] = "cached"
            out["cached_count"] += 1
        else:
            p = apply_closure_strength_control(
                p_base,
                closure_strength_lambda=lam,
                Q_f=q,
                U_f=u,
                mode=str(c1["control_application_name"]),
            )
            bundle = default_metric_bundle(p, lens, tau=1)
            row = {
                "size": 128,
                "closure_strength_lambda": lam,
                "analysis_k": int(bundle["analysis_k"]),
                "objecthood_order": observed_float(bundle["objecthood_order"]),
                "closure_error": observed_float(bundle["closure_error"]),
                "staging_gap": observed_float(bundle["staging_gap"]),
                "affinity": observed_float(bundle["affinity"]),
                "cache_status": "executed",
            }
            row_path.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
            out["executed_count"] += 1
        out["class_i_raw"].append(row)

    # Class II size-128 panel
    c2 = config["class_ii_auxiliary_size128"]
    substrate = build_substrate_family(
        "driven_cycle_family",
        n=128,
        self_weight=float(c2["self_weight"]),
        forward_weight=float(c2["forward_weight"]),
        backward_weight=float(c2["backward_weight"]),
    )
    p_base = np.asarray(substrate["P"], dtype=np.float64)
    lens = np.asarray([0 if i < 64 else 1 for i in range(128)], dtype=np.int64)
    q = pushforward_matrix(lens, 2)
    u = np.asarray(build_lift_family("uniform_lift_family", f=lens, k=2)[0], dtype=np.float64)
    for lam in [float(x) for x in c2["lambda_grid"]]:
        run_id = f"sz2_{_stable_hash({'class': 'ii', 'panel_config': c2, 'lambda': lam})}"
        row_path = runs_dir / f"{run_id}.json"
        if cache_reuse_allowed(use_cache) and row_path.exists():
            row = json.loads(row_path.read_text(encoding="utf-8"))
            row["cache_status"] = "cached"
            out["cached_count"] += 1
        else:
            p = apply_closure_strength_control(
                p_base,
                closure_strength_lambda=lam,
                Q_f=q,
                U_f=u,
                mode=str(c2["control_application_name"]),
            )
            bundle = default_metric_bundle(p, lens, tau=1)
            row = {
                "size": 128,
                "closure_strength_lambda": lam,
                "analysis_k": int(bundle["analysis_k"]),
                "objecthood_order": observed_float(bundle["objecthood_order"]),
                "closure_error": observed_float(bundle["closure_error"]),
                "staging_gap": observed_float(bundle["staging_gap"]),
                "affinity": observed_float(bundle["affinity"]),
                "cache_status": "executed",
            }
            row_path.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
            out["executed_count"] += 1
        out["class_ii_raw"].append(row)

    out["class_i_group"] = _group_rows(out["class_i_raw"])
    out["class_ii_group"] = _group_rows(out["class_ii_raw"])
    return out


def summarize_fss_support(
    class_i_summary: dict[str, Any],
    class_ii_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
) -> dict[str, Any]:
    def _step(grid: list[float]) -> float:
        return _grid_step([float(x) for x in grid])

    ci_plus = class_i_summary["plus128_union_grid"]["summary"]
    cii_plus = class_ii_summary["plus128_union_grid"]["summary"]
    ci_grid = class_i_summary["plus128_union_grid"]["fit_grid"]
    cii_grid = class_ii_summary["plus128_union_grid"]["fit_grid"]

    ci_ok = bool(
        (not ci_plus["best_is_boundary"])
        and ci_plus["beta_span_10pct"] <= _step(ci_grid["beta_over_nu_grid"]) + 1e-15
        and ci_plus["inv_nu_span_10pct"] <= _step(ci_grid["inv_nu_grid"]) + 1e-15
        and ci_plus["local_contrast_ratio"] >= 1.05
        and (not class_i_summary["grid_sensitivity"]["grid_spec_sensitive"])
    )
    cii_ok = bool(
        (not cii_plus["best_is_boundary"])
        and cii_plus["beta_span_10pct"] <= _step(cii_grid["beta_over_nu_grid"]) + 1e-15
        and cii_plus["inv_nu_span_10pct"] <= _step(cii_grid["inv_nu_grid"]) + 1e-15
        and cii_plus["local_contrast_ratio"] >= 1.05
        and (not class_ii_summary["grid_sensitivity"]["grid_spec_sensitive"])
    )
    fss_supported = bool(ci_ok and cii_ok)
    affinity_primary = bool((not fss_supported) and comparison_summary.get("class_separation_plausible", False))

    any_helpful = bool(
        class_i_summary["size_extension"]["size128_helpful"]
        or class_ii_summary["size_extension"]["size128_helpful"]
    )
    if fss_supported:
        rec = "fss_claims_supported"
    elif any_helpful:
        rec = "stabilize_with_size_extension"
    else:
        rec = "demote_exponent_claims_keep_affinity_separation"
    return {
        "class_i_grid_spec_sensitive": bool(class_i_summary["grid_sensitivity"]["grid_spec_sensitive"]),
        "class_ii_grid_spec_sensitive": bool(class_ii_summary["grid_sensitivity"]["grid_spec_sensitive"]),
        "class_i_size128_helpful": bool(class_i_summary["size_extension"]["size128_helpful"]),
        "class_ii_size128_helpful": bool(class_ii_summary["size_extension"]["size128_helpful"]),
        "fss_exponent_claim_supported": fss_supported,
        "affinity_separation_remains_primary": affinity_primary,
        "final_recommendation": rec,
    }


def format_fss_stability_summary(
    class_i_summary: dict[str, Any],
    class_ii_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
) -> dict[str, Any]:
    verdicts = summarize_fss_support(class_i_summary, class_ii_summary, comparison_summary)
    return {
        "class_i": class_i_summary,
        "class_ii": class_ii_summary,
        "comparison_summary": comparison_summary,
        "verdicts": verdicts,
    }


def run_fss_stability_audit(
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
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    req = [
        root / config["inputs"]["class_i_group_summary"],
        root / config["inputs"]["class_i_fit_summary"],
        root / config["inputs"]["class_ii_group_summary"],
        root / config["inputs"]["class_ii_fit_summary"],
        root / config["inputs"]["class_ii_comparison_summary"],
    ]
    if not all(artifact_is_current(p) for p in req):
        run_class_i_campaign_driver(output_root=root / "results" / "campaigns", use_cache=True)
        run_class_ii_campaign_driver(output_root=root / "results" / "campaigns", use_cache=True)

    class_i_old = _load_csv(root / config["inputs"]["class_i_group_summary"])
    class_ii_old = _load_csv(root / config["inputs"]["class_ii_group_summary"])
    comparison_summary = json.loads((root / config["inputs"]["class_ii_comparison_summary"]).read_text(encoding="utf-8"))

    aux = run_auxiliary_size_extension_panels(config, output_root=output_root, use_cache=use_cache)
    _write_csv(
        dirs["analysis"] / "class_i_auxiliary_size128_raw.csv",
        aux["class_i_raw"],
        ["size", "closure_strength_lambda", "analysis_k", "objecthood_order", "closure_error", "staging_gap", "affinity", "cache_status"],
    )
    _write_csv(
        dirs["analysis"] / "class_ii_auxiliary_size128_raw.csv",
        aux["class_ii_raw"],
        ["size", "closure_strength_lambda", "analysis_k", "objecthood_order", "closure_error", "staging_gap", "affinity", "cache_status"],
    )

    class_i_plus = sorted(class_i_old + aux["class_i_group"], key=lambda r: (int(r["size"]), float(r["closure_strength_lambda"])))
    class_ii_plus = sorted(class_ii_old + aux["class_ii_group"], key=lambda r: (int(r["size"]), float(r["closure_strength_lambda"])))

    class_i_suite = compute_landscape_suite(class_i_old, config["fit_grid_suite"]["class_i"])
    class_ii_suite = compute_landscape_suite(class_ii_old, config["fit_grid_suite"]["class_ii"])
    class_i_plus_land = _compute_landscape(class_i_plus, config["fit_grid_suite"]["class_i"]["union_grid"])
    class_ii_plus_land = _compute_landscape(class_ii_plus, config["fit_grid_suite"]["class_ii"]["union_grid"])
    class_i_plus_sum = _summarize_landscape(class_i_plus_land, config["fit_grid_suite"]["class_i"]["union_grid"], 0.10)
    class_ii_plus_sum = _summarize_landscape(class_ii_plus_land, config["fit_grid_suite"]["class_ii"]["union_grid"], 0.10)

    class_i_summary = {
        **class_i_suite,
        "plus128_union_grid": {
            "landscape": class_i_plus_land,
            "summary": class_i_plus_sum,
            "fit_grid": config["fit_grid_suite"]["class_i"]["union_grid"],
        },
    }
    class_ii_summary = {
        **class_ii_suite,
        "plus128_union_grid": {
            "landscape": class_ii_plus_land,
            "summary": class_ii_plus_sum,
            "fit_grid": config["fit_grid_suite"]["class_ii"]["union_grid"],
        },
    }
    class_i_summary["grid_sensitivity"] = compare_grid_sensitivity(class_i_summary)
    class_ii_summary["grid_sensitivity"] = compare_grid_sensitivity(class_ii_summary)
    class_i_summary["size_extension"] = compare_size_extension(class_i_summary["union_grid"], class_i_summary["plus128_union_grid"])
    class_ii_summary["size_extension"] = compare_size_extension(class_ii_summary["union_grid"], class_ii_summary["plus128_union_grid"])

    full_summary = format_fss_stability_summary(class_i_summary, class_ii_summary, comparison_summary)

    # write landscape summaries
    mapping = {
        "class_i_oldsizes_original_grid": class_i_summary["original_grid"],
        "class_i_oldsizes_ident_grid": class_i_summary["ident_grid"],
        "class_i_oldsizes_union_grid": class_i_summary["union_grid"],
        "class_i_plus128_union_grid": class_i_summary["plus128_union_grid"],
        "class_ii_oldsizes_original_grid": class_ii_summary["original_grid"],
        "class_ii_oldsizes_ident_grid": class_ii_summary["ident_grid"],
        "class_ii_oldsizes_union_grid": class_ii_summary["union_grid"],
        "class_ii_plus128_union_grid": class_ii_summary["plus128_union_grid"],
    }
    for name, payload in mapping.items():
        (dirs["analysis"] / f"{name}.json").write_text(
            scientific_dumps({"summary": payload["summary"], "best_fit": payload["landscape"]["best_fit"], "top_candidates": payload["summary"]["top_candidates"]}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    (dirs["analysis"] / "fss_stability_summary.json").write_text(scientific_dumps(full_summary, indent=2) + "\n", encoding="utf-8")

    # metrics csv
    metrics_rows = []
    for name, payload in mapping.items():
        cls = "class_i" if name.startswith("class_i") else "class_ii"
        size_panel = "plus128" if "plus128" in name else "oldsizes"
        grid_name = "union_grid" if "union_grid" in name else ("ident_grid" if "ident_grid" in name else "original_grid")
        s = payload["summary"]
        metrics_rows.append(
            {
                "fit_name": name,
                "class_name": cls,
                "size_panel": size_panel,
                "fit_grid_name": grid_name,
                "lambda_c_fit": float(s["best_fit"]["lambda_c"]),
                "beta_over_nu_fit": float(s["best_fit"]["beta_over_nu"]),
                "inv_nu_fit": float(s["best_fit"]["inv_nu"]),
                "collapse_objective": float(s["best_fit"]["objective"]),
                "best_is_boundary": bool(s["best_is_boundary"]),
                "plateau_count_10pct": int(s["plateau_count_10pct"]),
                "lambda_span_10pct": float(s["lambda_span_10pct"]),
                "beta_span_10pct": float(s["beta_span_10pct"]),
                "inv_nu_span_10pct": float(s["inv_nu_span_10pct"]),
                "local_contrast_ratio": float(s["local_contrast_ratio"]),
                "grid_spec_sensitive": class_i_summary["grid_sensitivity"]["grid_spec_sensitive"] if cls == "class_i" else class_ii_summary["grid_sensitivity"]["grid_spec_sensitive"],
                "size128_helpful": class_i_summary["size_extension"]["size128_helpful"] if cls == "class_i" else class_ii_summary["size_extension"]["size128_helpful"],
            }
        )
    _write_csv(dirs["metrics"] / "metrics.csv", metrics_rows, METRIC_FIELDS)

    # plots
    x = [0, 1, 2]
    _line_plot(
        dirs["plots"] / "class_i_grid_sensitivity.png",
        x,
        {
            "objective": [
                class_i_summary["original_grid"]["summary"]["best_fit"]["objective"],
                class_i_summary["ident_grid"]["summary"]["best_fit"]["objective"],
                class_i_summary["union_grid"]["summary"]["best_fit"]["objective"],
            ]
        },
    )
    _line_plot(
        dirs["plots"] / "class_ii_grid_sensitivity.png",
        x,
        {
            "objective": [
                class_ii_summary["original_grid"]["summary"]["best_fit"]["objective"],
                class_ii_summary["ident_grid"]["summary"]["best_fit"]["objective"],
                class_ii_summary["union_grid"]["summary"]["best_fit"]["objective"],
            ]
        },
    )
    _line_plot(
        dirs["plots"] / "plateau_count_comparison.png",
        [0, 1],
        {
            "class_i": [
                class_i_summary["union_grid"]["summary"]["plateau_count_10pct"],
                class_i_summary["plus128_union_grid"]["summary"]["plateau_count_10pct"],
            ],
            "class_ii": [
                class_ii_summary["union_grid"]["summary"]["plateau_count_10pct"],
                class_ii_summary["plus128_union_grid"]["summary"]["plateau_count_10pct"],
            ],
        },
    )
    _line_plot(
        dirs["plots"] / "best_fit_drift.png",
        [0, 1],
        {
            "class_i_lambda_c": [
                class_i_summary["union_grid"]["summary"]["best_fit"]["lambda_c"],
                class_i_summary["plus128_union_grid"]["summary"]["best_fit"]["lambda_c"],
            ],
            "class_ii_lambda_c": [
                class_ii_summary["union_grid"]["summary"]["best_fit"]["lambda_c"],
                class_ii_summary["plus128_union_grid"]["summary"]["best_fit"]["lambda_c"],
            ],
        },
    )

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    notes = dirs["notes"] / "findings.md"
    verdicts = full_summary["verdicts"]
    notes.write_text(
        "\n".join(
            [
                "# FSS stability and size-extension findings",
                "",
                f"- class_i_grid_spec_sensitive: `{verdicts['class_i_grid_spec_sensitive']}`",
                f"- class_ii_grid_spec_sensitive: `{verdicts['class_ii_grid_spec_sensitive']}`",
                f"- class_i_size128_helpful: `{verdicts['class_i_size128_helpful']}`",
                f"- class_ii_size128_helpful: `{verdicts['class_ii_size128_helpful']}`",
                f"- fss_exponent_claim_supported: `{verdicts['fss_exponent_claim_supported']}`",
                f"- affinity_separation_remains_primary: `{verdicts['affinity_separation_remains_primary']}`",
                f"- final_recommendation: `{verdicts['final_recommendation']}`",
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
        "experiment_id": "fss_stability_size_extension",
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
                "# LB-17 FSS stability and size-extension audit",
                "",
                f"- config path: `{config['config_path_hint']}`",
                f"- artifact root: `{artifact_root}`",
                f"- class-I grid sensitivity: `{class_i_summary['grid_sensitivity']}`",
                f"- class-II grid sensitivity: `{class_ii_summary['grid_sensitivity']}`",
                f"- class-I size128 extension: `{class_i_summary['size_extension']}`",
                f"- class-II size128 extension: `{class_ii_summary['size_extension']}`",
                f"- are the current exponent fits grid-specification sensitive? `{'yes' if (verdicts['class_i_grid_spec_sensitive'] or verdicts['class_ii_grid_spec_sensitive']) else 'no'}`",
                f"- does adding size 128 materially stabilize the FSS story? `{'yes' if (verdicts['class_i_size128_helpful'] or verdicts['class_ii_size128_helpful']) else 'no'}`",
                f"- should the paper currently make exponent claims? `{'yes' if verdicts['fss_exponent_claim_supported'] else 'no'}`",
                f"- affinity-separation primary distinction remains supported: `{verdicts['affinity_separation_remains_primary']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "class_i_grid_spec_sensitive": bool(verdicts["class_i_grid_spec_sensitive"]),
        "class_ii_grid_spec_sensitive": bool(verdicts["class_ii_grid_spec_sensitive"]),
        "class_i_size128_helpful": bool(verdicts["class_i_size128_helpful"]),
        "class_ii_size128_helpful": bool(verdicts["class_ii_size128_helpful"]),
        "fss_exponent_claim_supported": bool(verdicts["fss_exponent_claim_supported"]),
        "affinity_separation_remains_primary": bool(verdicts["affinity_separation_remains_primary"]),
        "final_recommendation": str(verdicts["final_recommendation"]),
        "artifact_root": str(artifact_root),
        "executed_count": int(aux["executed_count"]),
        "cached_count": int(aux["cached_count"]),
    }
