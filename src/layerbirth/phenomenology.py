"""F-02 Class-III / Class-IV susceptibility and Binder shadow panel."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import artifact_is_current
from .class3 import (
    _build_class_iv_candidate_substrate,
    _evaluate_tau_panel,
    _repo_root,
    _write_csv,
    evaluate_class_iii_candidate,
)
from .campaigns import _git_code_version
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .robustness import _line_plot
from .scaling import binder_like_cumulant, susceptibility_from_samples


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(scientific_dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ensure_prereq_roots(root: Path, use_cache: bool) -> None:
    from .class3 import run_class_iv_full_campaign, run_class_iii_strict_confirmation_v2
    from .p4 import run_p4_anomaly_metric_layer
    from .robustness import run_class_iii_class_iv_lens_suite
    from .taxonomy import run_canonical_class_rubric

    req = {
        "class3": root / "results" / "pilots" / "class_iii_strict_confirmation_v2",
        "class4": root / "results" / "campaigns" / "class_iv_full_campaign",
        "taxonomy": root / "results" / "taxonomy" / "canonical_class_rubric",
        "p4": root / "results" / "metrics" / "p4_anomaly_metric_layer",
        "lens": root / "results" / "robustness" / "class_iii_class_iv_lens_suite",
    }
    if not artifact_is_current(req["class3"], root / "configs/pilots/class_iii_strict_confirmation_v2.json"):
        run_class_iii_strict_confirmation_v2(
            root / "configs" / "pilots" / "class_iii_strict_confirmation_v2.json",
            output_root=root / "results" / "pilots",
            use_cache=use_cache,
        )
    if not artifact_is_current(req["class4"], root / "configs/campaigns/class_iv_full_campaign.json"):
        run_class_iv_full_campaign(
            root / "configs" / "campaigns" / "class_iv_full_campaign.json",
            output_root=root / "results" / "campaigns",
            use_cache=use_cache,
        )
    if not artifact_is_current(req["taxonomy"], root / "configs/taxonomy/canonical_class_rubric.json"):
        run_canonical_class_rubric(
            root / "configs" / "taxonomy" / "canonical_class_rubric.json",
            output_root=root / "results" / "taxonomy",
            use_cache=use_cache,
        )
    if not artifact_is_current(req["p4"], root / "configs/metrics/p4_anomaly_metric_layer.json"):
        run_p4_anomaly_metric_layer(
            root / "configs" / "metrics" / "p4_anomaly_metric_layer.json",
            output_root=root / "results" / "metrics",
            use_cache=use_cache,
        )
    if not artifact_is_current(req["lens"], root / "configs/robustness/class_iii_class_iv_lens_suite.json"):
        run_class_iii_class_iv_lens_suite(
            root / "configs" / "robustness" / "class_iii_class_iv_lens_suite.json",
            output_root=root / "results" / "robustness",
            use_cache=use_cache,
        )


def _resolve_class_iii_representative(section: dict[str, Any], root: Path) -> dict[str, Any]:
    summary = json.loads((root / section["representative_source"]["confirmation_summary_path"]).read_text(encoding="utf-8"))
    cfg = json.loads((root / section["representative_source"]["strict_config_path"]).read_text(encoding="utf-8"))
    best = str(summary["overall"]["best_variant"])
    candidates = {str(c["candidate_name"]): c for c in cfg["candidates"]}
    chosen = candidates[best]
    return {
        "class_name": "Class-III",
        "target_label": section["target_label"],
        "representative_name": str(chosen["candidate_name"]),
        "family_name": str(chosen["family_name"]),
        "base_kwargs": dict(chosen.get("base_kwargs", {})),
        "drive_kwargs": {},
        "artifact_subdir": str(section["artifact_subdir"]),
    }


def _resolve_class_iv_representative(section: dict[str, Any], root: Path) -> dict[str, Any]:
    campaign_cfg = json.loads((root / section["representative_source"]["campaign_config_path"]).read_text(encoding="utf-8"))
    cand = dict(campaign_cfg["candidate"])
    return {
        "class_name": "Class-IV",
        "target_label": section["target_label"],
        "representative_name": str(cand["candidate_name"]),
        "family_name": str(cand["family_name"]),
        "base_kwargs": dict(cand.get("base_kwargs", {})),
        "drive_kwargs": dict(cand.get("drive_kwargs", {})),
        "artifact_subdir": str(section["artifact_subdir"]),
    }


def build_shadow_ensemble_candidates(representative_config: dict[str, Any], perturbation_spec: dict[str, Any]) -> list[dict[str, Any]]:
    factors = [float(x) for x in perturbation_spec.get("factors", [1.0])]
    if any(not np.isfinite(factor) or factor < 0 for factor in factors):
        raise ValueError("perturbation factors must be finite and nonnegative")
    structural_keys = [str(k) for k in perturbation_spec.get("structural_keys", [])]
    drive_keys = [str(k) for k in perturbation_spec.get("drive_keys", [])]
    base_struct = dict(representative_config.get("base_kwargs", {}))
    base_drive = dict(representative_config.get("drive_kwargs", {}))

    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": f"{representative_config['representative_name']}_base",
            "base_kwargs": dict(base_struct),
            "drive_kwargs": dict(base_drive),
        }
    ]

    for key in structural_keys:
        if key not in base_struct:
            continue
        for fac in factors:
            if fac == 1.0:
                continue
            st = dict(base_struct)
            st[key] = float(st[key]) * float(fac)
            candidates.append(
                {
                    "candidate_id": f"{representative_config['representative_name']}_struct_{key}_x{fac!r}",
                    "base_kwargs": st,
                    "drive_kwargs": dict(base_drive),
                }
            )

    for key in drive_keys:
        if key not in base_drive:
            continue
        for fac in factors:
            if fac == 1.0:
                continue
            dr = dict(base_drive)
            dr[key] = float(dr[key]) * float(fac)
            candidates.append(
                {
                    "candidate_id": f"{representative_config['representative_name']}_drive_{key}_x{fac!r}",
                    "base_kwargs": dict(base_struct),
                    "drive_kwargs": dr,
                }
            )

    dedup: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        key = scientific_dumps({"b": cand["base_kwargs"], "d": cand["drive_kwargs"]}, sort_keys=True)
        dedup[key] = cand
    return list(dedup.values())


def _build_substrate_for_candidate(representative_config: dict[str, Any], candidate: dict[str, Any], size: int) -> dict[str, Any]:
    from .substrates import build_class_iii_candidate_family

    if representative_config["class_name"] == "Class-III":
        return build_class_iii_candidate_family(
            representative_config["family_name"],
            **{**dict(candidate["base_kwargs"]), "n": int(size)},
        )
    return _build_class_iv_candidate_substrate(
        representative_config["family_name"],
        dict(candidate["base_kwargs"]),
        dict(candidate.get("drive_kwargs", {})),
        int(size),
    )


def filter_admissible_shadow_ensemble(
    representative_config: dict[str, Any],
    candidates: list[dict[str, Any]],
    sizes: list[int],
    lambda_grid: list[float],
    control_mode: str,
    rubric_config: dict[str, Any],
    p4_config: dict[str, Any],
    min_admissible_replicates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not sizes or any(not np.isfinite(size) or int(size) != size or size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("shadow size panel must contain distinct positive integers")
    if not lambda_grid or any(not np.isfinite(x) or not 0 <= x <= 1 for x in lambda_grid) or any(b <= a for a, b in zip(lambda_grid, lambda_grid[1:])):
        raise ValueError("shadow lambda grid must be finite, ordered and in [0,1]")
    if min_admissible_replicates < 1 or int(min_admissible_replicates) != min_admissible_replicates:
        raise ValueError("minimum replicate count must be a positive integer")
    if len({candidate['candidate_id'] for candidate in candidates}) != len(candidates):
        raise ValueError("candidate identifiers must be distinct")
    evaluated: list[dict[str, Any]] = []
    admissible: list[dict[str, Any]] = []

    for candidate in candidates:
        labels_by_size: dict[str, str] = {}
        stable = True
        for size in sizes:
            substrate = _build_substrate_for_candidate(representative_config, candidate, int(size))
            tau1_rows, tau1_boundary, tau1_aff = _evaluate_tau_panel(substrate, lambda_grid, 1, control_mode)
            tau2_rows, tau2_boundary, tau2_aff = _evaluate_tau_panel(substrate, lambda_grid, 2, control_mode)
            ev = evaluate_class_iii_candidate(
                {
                    "tau1_rows": tau1_rows,
                    "tau2_rows": tau2_rows,
                    "tau1_boundary": tau1_boundary,
                    "tau2_boundary": tau2_boundary,
                    "tau1_affinity_ref": tau1_aff,
                    "tau2_affinity_ref": tau2_aff,
                },
                rubric_config,
                p4_config,
            )
            label = str(ev["canonical_class_label"])
            labels_by_size[str(size)] = label
            if label != representative_config["target_label"]:
                stable = False

        evaluated.append({"candidate_id": candidate["candidate_id"], "labels_by_size": labels_by_size, "admissible": stable})
        if stable:
            admissible.append(candidate)

    candidate_count = len(candidates)
    admissible_count = len(admissible)
    summary = {
        "class_name": representative_config["class_name"],
        "representative_name": representative_config["representative_name"],
        "candidate_count": candidate_count,
        "admissible_count": admissible_count,
        "rejected_count": candidate_count - admissible_count,
        "min_admissible_replicates": int(min_admissible_replicates),
        "ensemble_label_stable": bool(admissible_count >= int(min_admissible_replicates)),
        "admissibility_rows": evaluated,
        "diagnosis": (
            "label preserved under perturbation ensemble"
            if admissible_count >= int(min_admissible_replicates)
            else "insufficient class-preserving perturbation replicates"
        ),
    }
    return admissible, summary


def compute_shadow_panel_rows(
    representative_config: dict[str, Any],
    admissible_candidates: list[dict[str, Any]],
    lambda_grid: list[float],
    sizes: list[int],
    tau_values: list[int],
    control_mode: str,
) -> list[dict[str, Any]]:
    tau_values = [int(t) for t in tau_values]
    if sorted(tau_values) != [1, 2]:
        raise ValueError("F-02 requires tau values [1,2].")

    raw_rows: list[dict[str, Any]] = []
    for cand in admissible_candidates:
        for size in sizes:
            substrate = _build_substrate_for_candidate(representative_config, cand, int(size))
            tau1_rows, _, _ = _evaluate_tau_panel(substrate, lambda_grid, 1, control_mode)
            tau2_rows, _, _ = _evaluate_tau_panel(substrate, lambda_grid, 2, control_mode)
            by_lambda_1 = {float(r["closure_strength_lambda"]): r for r in tau1_rows}
            by_lambda_2 = {float(r["closure_strength_lambda"]): r for r in tau2_rows}
            for lam in lambda_grid:
                r1 = by_lambda_1[float(lam)]
                r2 = by_lambda_2[float(lam)]
                m1 = observed_float(r1["objecthood_order"])
                m2 = observed_float(r2["objecthood_order"])
                raw_rows.append(
                    {
                        "class_name": representative_config["class_name"],
                        "representative_name": representative_config["representative_name"],
                        "candidate_id": cand["candidate_id"],
                        "size": int(size),
                        "closure_strength_lambda": float(lam),
                        "M_obj_tau1": m1,
                        "M_obj_tau2": m2,
                        "CE_tau1": observed_float(r1["closure_error"]),
                        "CE_tau2": observed_float(r2["closure_error"]),
                        "Aff_tau1": observed_float(r1["affinity"]),
                        "Aff_tau2": observed_float(r2["affinity"]),
                        "delta_mobj": float(m2 - m1),
                    }
                )
    return raw_rows


def _susceptibility(values: np.ndarray, system_size: int) -> float:
    if values.size == 0:
        return float("nan")
    return susceptibility_from_samples(values, system_size)[0]


def _binder(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return binder_like_cumulant(values)[0]


def group_shadow_panel_observations(raw_rows: list[dict[str, Any]], observable_key: str = "delta_mobj") -> list[dict[str, Any]]:
    groups: dict[tuple[int, float], list[float]] = {}
    for row in raw_rows:
        key = (int(row["size"]), float(row["closure_strength_lambda"]))
        groups.setdefault(key, []).append(float(row[observable_key]))

    grouped: list[dict[str, Any]] = []
    for (size, lam), vals in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.asarray(vals, dtype=np.float64)
        grouped.append(
            {
                "size": int(size),
                "closure_strength_lambda": float(lam),
                "n_admissible_replicates": int(arr.size),
                "delta_mobj_mean": float(np.mean(arr)) if arr.size else float("nan"),
                "susceptibility_delta_mobj": _susceptibility(arr, size),
                "binder_delta_mobj": _binder(arr),
            }
        )
    return grouped


def summarize_shadow_panel(
    group_rows: list[dict[str, Any]],
    admissibility_summary: dict[str, Any],
    lambda_grid: list[float],
    min_admissible_replicates: int,
    expected_sizes: list[int] | None = None,
) -> dict[str, Any]:
    ensemble_label_stable = bool(admissibility_summary["admissible_count"] >= int(min_admissible_replicates))
    coverage_required = math.ceil(0.75 * len(lambda_grid))
    size_points: dict[int, set[float]] = {}
    domain = set(lambda_grid)
    unique_points: set[tuple[int, float]] = set()
    observations_valid = bool(group_rows and domain)
    for row in group_rows:
        point = (int(row["size"]), float(row["closure_strength_lambda"]))
        observations_valid &= point not in unique_points and point[1] in domain
        observations_valid &= int(row.get("n_admissible_replicates", 0)) >= min_admissible_replicates
        unique_points.add(point)
        size_points.setdefault(point[0], set()).add(point[1])
    covered_sizes = [s for s, points in size_points.items() if len(points & domain) >= coverage_required]
    panel_complete = expected_sizes is None or set(expected_sizes) <= set(covered_sizes)

    chi_vals = np.asarray([float(r["susceptibility_delta_mobj"]) for r in group_rows], dtype=np.float64) if group_rows else np.asarray([], dtype=np.float64)
    binder_vals = np.asarray([float(r["binder_delta_mobj"]) for r in group_rows], dtype=np.float64) if group_rows else np.asarray([], dtype=np.float64)
    delta_vals = np.asarray([float(r["delta_mobj_mean"]) for r in group_rows], dtype=np.float64) if group_rows else np.asarray([], dtype=np.float64)

    usable = bool(
        ensemble_label_stable
        and observations_valid
        and panel_complete
        and len(covered_sizes) >= 2
        and np.all(np.isfinite(chi_vals))
        and np.all(np.isfinite(binder_vals))
        and np.all(np.isfinite(delta_vals))
        and (float(np.max(delta_vals) - np.min(delta_vals)) if delta_vals.size else 0.0) >= 0.05
    )

    if usable:
        verdict = "usable_panel"
    elif ensemble_label_stable:
        verdict = "label_stable_but_phenomenology_flat"
    else:
        verdict = "not_stable_under_ensemble_perturbation"

    return {
        "class_name": admissibility_summary["class_name"],
        "representative_name": admissibility_summary["representative_name"],
        "ensemble_label_stable": ensemble_label_stable,
        "usable_chi_u4_panel": usable,
        "coverage_required": coverage_required,
        "observations_valid": bool(observations_valid),
        "declared_size_coverage_complete": bool(panel_complete),
        "sizes_with_sufficient_lambda_coverage": sorted(covered_sizes),
        "delta_mobj_mean_range": float(np.max(delta_vals) - np.min(delta_vals)) if delta_vals.size else 0.0,
        "final_per_class_verdict": verdict,
        "diagnosis": (
            "usable susceptibility/Binder panel under class-preserving perturbations"
            if verdict == "usable_panel"
            else "label-preserving ensemble exists but panel is too flat/degenerate"
            if verdict == "label_stable_but_phenomenology_flat"
            else "insufficient class-preserving perturbation replicates"
        ),
    }


def _write_panel_bundle(
    root: Path,
    artifact_root: Path,
    config_snapshot: dict[str, Any],
    class_name: str,
    admissibility_summary: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    panel_summary: dict[str, Any],
) -> None:
    def _collapse_identical_series(
        series: dict[str, list[float]],
        *,
        atol: float = 1e-12,
        rtol: float = 1e-9,
    ) -> dict[str, list[float]]:
        if len(series) <= 1:
            return series
        labels = list(series.keys())
        first = np.asarray(series[labels[0]], dtype=np.float64)
        if first.size == 0:
            return series
        for label in labels[1:]:
            other = np.asarray(series[label], dtype=np.float64)
            if other.shape != first.shape or not np.allclose(first, other, atol=atol, rtol=rtol):
                return series
        size_labels = [label.removeprefix("size_") for label in labels]
        merged_label = f"all sizes ({', '.join(size_labels)})"
        return {merged_label: series[labels[0]]}

    dirs = {
        "config": artifact_root / "config",
        "seeds": artifact_root / "seeds",
        "metrics": artifact_root / "metrics",
        "notes": artifact_root / "notes",
        "env": artifact_root / "env",
        "analysis": artifact_root / "analysis",
        "plots": artifact_root / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    _write_json(dirs["config"] / "config_snapshot.json", config_snapshot)
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    _write_json(dirs["env"] / "environment.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "python": "python3"})
    _write_json(dirs["analysis"] / "admissibility_summary.json", admissibility_summary)
    _write_json(dirs["analysis"] / "panel_summary.json", panel_summary)

    raw_fields = [
        "class_name",
        "representative_name",
        "candidate_id",
        "size",
        "closure_strength_lambda",
        "M_obj_tau1",
        "M_obj_tau2",
        "CE_tau1",
        "CE_tau2",
        "Aff_tau1",
        "Aff_tau2",
        "delta_mobj",
    ]
    _write_csv(dirs["analysis"] / "raw_panel_rows.csv", raw_rows, raw_fields)

    group_fields = [
        "size",
        "closure_strength_lambda",
        "n_admissible_replicates",
        "delta_mobj_mean",
        "susceptibility_delta_mobj",
        "binder_delta_mobj",
    ]
    _write_csv(dirs["analysis"] / "group_summary.csv", group_rows, group_fields)

    metrics_rows = [
        {
            "class_name": class_name,
            "size": int(r["size"]),
            "closure_strength_lambda": float(r["closure_strength_lambda"]),
            "n_admissible_replicates": int(r["n_admissible_replicates"]),
            "delta_mobj_mean": float(r["delta_mobj_mean"]),
            "susceptibility_delta_mobj": float(r["susceptibility_delta_mobj"]),
            "binder_delta_mobj": float(r["binder_delta_mobj"]),
            "cache_status": "executed",
            "manifest_path": "manifest.json",
        }
        for r in group_rows
    ]
    metrics_fields = [
        "class_name",
        "size",
        "closure_strength_lambda",
        "n_admissible_replicates",
        "delta_mobj_mean",
        "susceptibility_delta_mobj",
        "binder_delta_mobj",
        "cache_status",
        "manifest_path",
    ]
    _write_csv(dirs["metrics"] / "metrics.csv", metrics_rows, metrics_fields)

    by_size = sorted({int(r["size"]) for r in group_rows})
    for metric_key, fname in [
        ("delta_mobj_mean", "delta_mobj_mean_vs_lambda.png"),
        ("susceptibility_delta_mobj", "susceptibility_vs_lambda.png"),
        ("binder_delta_mobj", "binder_vs_lambda.png"),
    ]:
        series: dict[str, list[float]] = {}
        for s in by_size:
            rows = [r for r in group_rows if int(r["size"]) == s]
            rows = sorted(rows, key=lambda x: float(x["closure_strength_lambda"]))
            series[f"size_{s}"] = [float(r[metric_key]) for r in rows]
            x = [float(r["closure_strength_lambda"]) for r in rows]
        if group_rows:
            _line_plot(dirs["plots"] / fname, x, _collapse_identical_series(series))

    (dirs["notes"] / "findings.md").write_text(
        "\n".join(
            [
                f"# {class_name} shadow panel",
                "",
                f"- admissible_replicates: `{admissibility_summary['admissible_count']}`",
                f"- ensemble_label_stable: `{panel_summary['ensemble_label_stable']}`",
                f"- usable_chi_u4_panel: `{panel_summary['usable_chi_u4_panel']}`",
                f"- verdict: `{panel_summary['final_per_class_verdict']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": f"{class_name.lower().replace('-', '_').replace(' ', '_')}_shadow_panel",
        "bundle_id": f"{class_name.lower().replace('-', '_').replace(' ', '_')}_shadow_panel",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    schema = load_schema(root / "configs" / "result_bundle.schema.json")
    validate_manifest(manifest, schema)
    _write_json(artifact_root / "manifest.json", manifest)


def run_class_iii_class_iv_shadow_panel(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    _ensure_prereq_roots(root, use_cache=use_cache)
    out_root = (root / "results" / "phenomenology") if output_root is None else Path(output_root)

    rubric = json.loads((root / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))

    class3_rep = _resolve_class_iii_representative(cfg["class_iii"], root)
    class4_rep = _resolve_class_iv_representative(cfg["class_iv"], root)

    sizes = [int(s) for s in cfg["size_panel"]]
    lambdas = [float(x) for x in cfg["lambda_grid"]]
    taus = [int(t) for t in cfg["tau_values"]]
    min_adm = int(cfg["min_admissible_replicates"])
    control_mode = str(cfg["control_application_name"])

    outputs: dict[str, Any] = {}
    for rep, section in [(class3_rep, cfg["class_iii"]), (class4_rep, cfg["class_iv"])]:
        candidates = build_shadow_ensemble_candidates(rep, section["perturbation_spec"])
        admissible, adm_summary = filter_admissible_shadow_ensemble(
            rep,
            candidates,
            sizes,
            lambdas,
            control_mode,
            rubric,
            p4_cfg,
            min_adm,
        )
        raw_rows = compute_shadow_panel_rows(rep, admissible, lambdas, sizes, taus, control_mode) if admissible else []
        group_rows = group_shadow_panel_observations(raw_rows, "delta_mobj") if raw_rows else []
        panel_summary = summarize_shadow_panel(group_rows, adm_summary, lambdas, min_adm, sizes)

        artifact_root = out_root / str(rep["artifact_subdir"])
        _write_panel_bundle(root, artifact_root, cfg, rep["class_name"], adm_summary, raw_rows, group_rows, panel_summary)

        outputs[rep["class_name"]] = {
            "artifact_root": str(artifact_root),
            "representative_name": rep["representative_name"],
            "admissible_replicates": int(adm_summary["admissible_count"]),
            "ensemble_label_stable": bool(panel_summary["ensemble_label_stable"]),
            "usable_chi_u4_panel": bool(panel_summary["usable_chi_u4_panel"]),
            "final_per_class_verdict": str(panel_summary["final_per_class_verdict"]),
        }

    suite = {
        "class_iii_usable_panel": bool(outputs["Class-III"]["usable_chi_u4_panel"]),
        "class_iv_usable_panel": bool(outputs["Class-IV"]["usable_chi_u4_panel"]),
        "phenomenology_suite_supported": bool(outputs["Class-III"]["usable_chi_u4_panel"] or outputs["Class-IV"]["usable_chi_u4_panel"]),
        "both_usable": bool(outputs["Class-III"]["usable_chi_u4_panel"] and outputs["Class-IV"]["usable_chi_u4_panel"]),
        "class_iii": outputs["Class-III"],
        "class_iv": outputs["Class-IV"],
    }

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/F-02_class_iii_class_iv_shadow_panel.md"))
    note_path.write_text(
        "\n".join(
            [
                "# F-02 Class-III / Class-IV susceptibility and Binder panel",
                "",
                "- Config: `configs/phenomenology/class_iii_class_iv_shadow_panel.json`",
                f"- Class-III artifact: `{outputs['Class-III']['artifact_root']}`",
                f"- Class-IV artifact: `{outputs['Class-IV']['artifact_root']}`",
                f"- Class-III representative: `{outputs['Class-III']['representative_name']}`",
                f"- Class-IV representative: `{outputs['Class-IV']['representative_name']}`",
                f"- Class-III admissible replicates: `{outputs['Class-III']['admissible_replicates']}`",
                f"- Class-IV admissible replicates: `{outputs['Class-IV']['admissible_replicates']}`",
                f"- Class-III verdict: `{outputs['Class-III']['final_per_class_verdict']}`",
                f"- Class-IV verdict: `{outputs['Class-IV']['final_per_class_verdict']}`",
                f"- Overall suite verdict: `{suite['phenomenology_suite_supported']}`",
                "",
                f"**does Class-III have a usable susceptibility/Binder panel under class-preserving perturbations? {'yes' if suite['class_iii_usable_panel'] else 'no'}**",
                f"**does Class-IV have a usable susceptibility/Binder panel under class-preserving perturbations? {'yes' if suite['class_iv_usable_panel'] else 'no'}**",
                "- Negative-result cause: report whether it is label instability vs flat/degenerate phenomenology; this run records that explicitly per class.",
                "- Proxy caveat: this panel uses operational shadow-ensemble observables, not universality/exponent claims.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return suite
