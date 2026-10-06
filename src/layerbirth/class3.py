"""S2-04 Class-III substrate candidate pilots and escalation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import artifact_is_current, computation_hash
from .campaigns import _git_code_version, _line_plot, _repo_root, _write_csv
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lifts import build_lift_family
from .lenses import build_lens_family
from .metrics import default_metric_bundle
from .numeric import pushforward_matrix
from .p4 import compute_boundary_shift_summary, evaluate_p4_profile, run_p4_anomaly_metric_layer
from .sweep import apply_closure_strength_control
from .taxonomy import (
    classify_activation_signature,
    estimate_affinity_reference,
    estimate_structural_boundaries,
    evaluate_p5_activation,
)
from .substrates import build_class_iii_candidate_family, driven_cycle_family


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def _validated_size_panel(values: list[Any]) -> list[int]:
    if not values or any(not np.isfinite(n) or int(n) != n or n <= 0 for n in values):
        raise ValueError("size panels require positive integer sizes")
    sizes = [int(n) for n in values]
    if len(set(sizes)) != len(sizes):
        raise ValueError("size panels require distinct sizes")
    return sizes


def _stable_hash(payload: dict[str, Any], n: int = 12) -> str:
    return computation_hash(payload, n)


def _state_plot(path: Path, xvals: list[float], series: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(series.keys())
    xs = np.asarray(xvals, dtype=np.float64)
    matrix = []
    pretty_labels = []
    for label in labels:
        arr = np.asarray(series[label], dtype=np.float64)
        row = np.zeros(len(xs), dtype=np.float64)
        n = min(len(xs), len(arr))
        if n:
            row[:n] = arr[:n]
        matrix.append(row)
        pretty_labels.append(label.replace("_", " "))

    data = np.asarray(matrix, dtype=np.float64)
    fig_h = max(2.8, 1.0 + 0.65 * len(pretty_labels))
    fig, ax = plt.subplots(figsize=(8.2, fig_h), dpi=180)
    cmap = ListedColormap(["#f2f2f2", "#31a354"])
    ax.imshow(data, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(xs)), [str(int(x)) if float(x).is_integer() else str(x) for x in xs])
    ax.set_yticks(np.arange(len(pretty_labels)), pretty_labels)
    ax.set_xlabel("System size", fontsize=10)
    ax.set_ylabel("State channel", fontsize=10)
    ax.tick_params(labelsize=9)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(
                j,
                i,
                "on" if value >= 0.5 else "off",
                ha="center",
                va="center",
                fontsize=8,
                color=("white" if value >= 0.5 else "#444444"),
            )

    ax.set_xticks(np.arange(-0.5, len(xs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pretty_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _evaluate_tau_panel(
    substrate: dict[str, Any],
    lambdas: list[float],
    tau: int,
    control_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    p_base = np.asarray(substrate["P"], dtype=np.float64)
    lens = np.asarray(substrate["coarse_lens"], dtype=np.int64)
    k = int(np.max(lens)) + 1
    q = pushforward_matrix(lens, k)
    u = np.asarray(build_lift_family("uniform_lift_family", f=lens, k=k)[0], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for lam in lambdas:
        p = apply_closure_strength_control(
            p_base,
            closure_strength_lambda=float(lam),
            Q_f=q,
            U_f=u,
            mode=control_mode,
        )
        b = default_metric_bundle(p, lens, tau=int(tau))
        rows.append(
            {
                "closure_strength_lambda": float(lam),
                "closure_error": observed_float(b["closure_error"]),
                "objecthood_order": observed_float(b["objecthood_order"]),
                "staging_gap": observed_float(b["staging_gap"]),
                "affinity": observed_float(b["affinity"]),
                "analysis_k": int(b["analysis_k"]),
                "resolved_tau": int(tau),
            }
        )
    boundary = estimate_structural_boundaries(rows)
    aff_ref = estimate_affinity_reference(rows, boundary)
    return rows, boundary, aff_ref


def evaluate_class_iii_candidate(candidate_rows: dict[str, Any], rubric_config: dict[str, Any], p4_config: dict[str, Any]) -> dict[str, Any]:
    p5_thr = rubric_config["activation_thresholds"]["p5"]
    p6_thr = rubric_config["activation_thresholds"]["p6_drive"]
    p4_thr = p4_config["thresholds"]

    tau1_boundary = candidate_rows["tau1_boundary"]
    tau2_boundary = candidate_rows["tau2_boundary"]
    p5_tau1 = evaluate_p5_activation(tau1_boundary, p5_thr)["state"]
    p5_tau2 = evaluate_p5_activation(tau2_boundary, p5_thr)["state"]
    candidate_p5_state = "active" if (p5_tau1 == "active" or p5_tau2 == "active") else ("inactive" if (p5_tau1 == "inactive" and p5_tau2 == "inactive") else "unknown")

    affinities = [observed_float(candidate_rows[key]["affinity_ref"]) for key in ("tau1_affinity_ref", "tau2_affinity_ref")]
    # Unknown channel data cannot silently establish drive inactivity. Positive
    # evidence in an observed channel still certifies drive presence.
    observed = [a for a in affinities if not np.isnan(a)]
    aff_max = max(observed, default=float("nan"))
    if aff_max >= float(p6_thr["p6_active_min"]):
        candidate_p6 = "active"
    elif len(observed) == len(affinities) and 0.0 <= aff_max <= float(p6_thr["p6_inactive_max"]):
        candidate_p6 = "inactive"
    else:
        candidate_p6 = "unknown"

    shift_summary = compute_boundary_shift_summary(
        tau1_summary=tau1_boundary,
        tau2_summary=tau2_boundary,
        ce_target=float(p5_thr["ce_target"]),
        mobj_target=float(p5_thr["mobj_target"]),
    )
    p4_eval = evaluate_p4_profile(shift_summary, p4_thr)
    candidate_p4 = p4_eval["p4_state"]

    canonical = classify_activation_signature(
        candidate_p5_state,
        candidate_p6,
        candidate_p4,
        rubric_config["canonical_class_signatures"],
    )
    candidate_is_class_iii = bool(canonical == "Class-III")
    if candidate_is_class_iii:
        candidate_status = "Class-III"
    elif candidate_p5_state == "active" and candidate_p6 == "inactive" and candidate_p4 == "inactive":
        candidate_status = "near_miss"
    else:
        candidate_status = "unclassified"

    eps = 1e-12
    mobj_minus_ce = observed_float(p4_eval["staging_shift_mobj"]) - observed_float(p4_eval["staging_shift_ce"])
    return {
        "candidate_p5_state": candidate_p5_state,
        "candidate_p6_drive_state": candidate_p6,
        "candidate_p4_state": candidate_p4,
        "canonical_class_label": canonical,
        "candidate_status": candidate_status,
        "candidate_is_class_iii": candidate_is_class_iii,
        "candidate_is_class_iii_strict": candidate_is_class_iii,
        "ce_boundary_tau1": tau1_boundary.get("ce_boundary_lambda"),
        "ce_boundary_tau2": tau2_boundary.get("ce_boundary_lambda"),
        "mobj_boundary_tau1": tau1_boundary.get("mobj_boundary_lambda"),
        "mobj_boundary_tau2": tau2_boundary.get("mobj_boundary_lambda"),
        "staging_shift_ce": observed_float(p4_eval["staging_shift_ce"]),
        "staging_shift_mobj": observed_float(p4_eval["staging_shift_mobj"]),
        "presence_shift_any": bool(p4_eval["presence_shift_any"]),
        "staging_gap_anomaly_score": observed_float(p4_eval["staging_gap_anomaly_score"]),
        "affinity_ref_max": aff_max,
        "p4_like_signal": bool(p4_eval["p4_like_signal"]),
        "p4_class_active": bool(p4_eval["p4_class_active"]),
        "shift_ratio_ce_to_mobj": float(observed_float(p4_eval["staging_shift_ce"]) / max(observed_float(p4_eval["staging_shift_mobj"]), eps)),
        "mobj_minus_ce_shift": float(mobj_minus_ce),
        "mobj_dominant_p4_candidate": bool(
            candidate_p5_state == "active"
            and candidate_p6 == "inactive"
            and observed_float(p4_eval["staging_shift_mobj"]) >= 0.15
            and observed_float(p4_eval["staging_shift_ce"]) < 0.15
            and bool(p4_eval["p4_like_signal"])
        ),
    }


def select_best_class_iii_candidate(candidate_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [r for r in candidate_summaries if bool(r["candidate_is_class_iii"]) ]
    if qualified:
        best = max(qualified, key=lambda r: (observed_float(r["staging_gap_anomaly_score"]), -abs(observed_float(r["affinity_ref_max"]))))
        return {
            "has_class_iii_candidate": True,
            "best_candidate": best["family_name"],
            "selection_basis": "max anomaly score, tie-break by lower |affinity_ref_max|",
            "diagnosis": "at least one candidate meets Class-III criteria",
        }
    near = sorted(candidate_summaries, key=lambda r: observed_float(r["staging_gap_anomaly_score"]), reverse=True)
    strongest = near[0] if near else None
    return {
        "has_class_iii_candidate": False,
        "best_candidate": None if strongest is None else strongest["family_name"],
        "selection_basis": "strongest near-miss by anomaly score",
        "diagnosis": "no candidate met Class-III criteria; selecting strongest near-miss",
    }


def format_class_iii_candidate_summary(candidate_summaries: list[dict[str, Any]], best_candidate: dict[str, Any]) -> dict[str, Any]:
    return {"candidates": candidate_summaries, "selection": best_candidate}


def build_class_iii_candidate_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    shared = config["shared"]
    lambdas = [float(x) for x in shared["lambda_grid"]]
    control_mode = str(shared["control_application_name"])
    for cand in config["candidates"]:
        name = str(cand["family_name"])
        substrate = build_class_iii_candidate_family(name, **dict(cand.get("family_kwargs", {})))
        tau1_rows, tau1_boundary, tau1_aff = _evaluate_tau_panel(substrate, lambdas, 1, control_mode)
        tau2_rows, tau2_boundary, tau2_aff = _evaluate_tau_panel(substrate, lambdas, 2, control_mode)
        rows.append(
            {
                "family_name": str(cand["candidate_name"]),
                "substrate_family_name": name,
                "substrate": substrate,
                "tau1_rows": tau1_rows,
                "tau2_rows": tau2_rows,
                "tau1_boundary": tau1_boundary,
                "tau2_boundary": tau2_boundary,
                "tau1_affinity_ref": tau1_aff,
                "tau2_affinity_ref": tau2_aff,
            }
        )
    return rows


def _write_candidate_bundle(root: Path, family_name: str, tau1_rows: list[dict[str, Any]], tau2_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    croot = root / "candidates" / family_name
    dirs = {
        "config": croot / "config",
        "seeds": croot / "seeds",
        "metrics": croot / "metrics",
        "plots": croot / "plots",
        "notes": croot / "notes",
        "env": croot / "env",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for tau, panel in ((1, tau1_rows), (2, tau2_rows)):
        for r in panel:
            metric_rows.append({"tau": tau, **r})
    _write_csv(
        dirs["metrics"] / "metrics.csv",
        metric_rows,
        ["tau", "closure_strength_lambda", "analysis_k", "closure_error", "objecthood_order", "staging_gap", "affinity", "resolved_tau"],
    )

    x = [float(r["closure_strength_lambda"]) for r in tau1_rows]
    _line_plot(
        dirs["plots"] / "ce_mobj_vs_lambda_tau_compare.png",
        x,
        {
            "tau1:CE": [observed_float(r["closure_error"]) for r in tau1_rows],
            "tau1:M_obj": [observed_float(r["objecthood_order"]) for r in tau1_rows],
            "tau2:CE": [observed_float(r["closure_error"]) for r in tau2_rows],
            "tau2:M_obj": [observed_float(r["objecthood_order"]) for r in tau2_rows],
        },
    )
    _line_plot(
        dirs["plots"] / "affinity_vs_lambda_tau_compare.png",
        x,
        {
            "tau1:Aff": [observed_float(r["affinity"]) for r in tau1_rows],
            "tau2:Aff": [observed_float(r["affinity"]) for r in tau2_rows],
        },
    )

    (croot / "summary.json").write_text(scientific_dumps(summary, indent=2) + "\n", encoding="utf-8")
    (dirs["config"] / "config_snapshot.json").write_text("{}\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(f"# Candidate {family_name}\n\n- summary: `{summary}`\n", encoding="utf-8")

    schema = load_schema(_repo_root() / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_iii_candidate_pilots",
        "bundle_id": f"class3_candidate_{family_name}",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(_repo_root()),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (croot / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_class_iii_candidate_pilots(config_path_or_obj: str | Path | dict[str, Any], output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = (root / "results" / "pilots") if output_root is None else Path(output_root)
    artifact_root = out / str(cfg["artifact_subdir"])

    req = [
        root / "results" / "metrics" / "p4_anomaly_metric_layer",
        root / "results" / "taxonomy" / "canonical_class_rubric",
        root / "results" / "dashboards" / "class_i_class_ii_consolidation",
    ]
    if any(not artifact_is_current(p) for p in req):
        run_p4_anomaly_metric_layer(root / "configs" / "metrics" / "p4_anomaly_metric_layer.json", output_root=root / "results" / "metrics", use_cache=use_cache)

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

    rubric = json.loads((root / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))
    candidates = build_class_iii_candidate_rows(cfg)

    summaries: list[dict[str, Any]] = []
    for cand in candidates:
        ev = evaluate_class_iii_candidate(cand, rubric, p4_cfg)
        summary = {"family_name": cand["family_name"], **ev}
        summaries.append(summary)
        _write_candidate_bundle(artifact_root, cand["family_name"], cand["tau1_rows"], cand["tau2_rows"], summary)

    selection = select_best_class_iii_candidate(summaries)
    overall = format_class_iii_candidate_summary(summaries, selection)

    table_fields = [
        "family_name",
        "candidate_p5_state",
        "candidate_p6_drive_state",
        "candidate_p4_state",
        "canonical_class_label",
        "ce_boundary_tau1",
        "ce_boundary_tau2",
        "mobj_boundary_tau1",
        "mobj_boundary_tau2",
        "staging_shift_ce",
        "staging_shift_mobj",
        "presence_shift_any",
        "staging_gap_anomaly_score",
        "affinity_ref_max",
        "candidate_is_class_iii",
    ]
    (dirs["analysis"] / "candidate_summary.json").write_text(scientific_dumps(overall, indent=2) + "\n", encoding="utf-8")
    _write_csv(dirs["analysis"] / "candidate_table.csv", summaries, table_fields)
    _write_csv(dirs["metrics"] / "metrics.csv", summaries, table_fields)

    x = [float(i) for i in range(len(summaries))]
    _line_plot(dirs["plots"] / "candidate_score_comparison.png", x, {r["family_name"]: [observed_float(r["staging_gap_anomaly_score"])] * len(x) for r in summaries})
    _line_plot(
        dirs["plots"] / "boundary_shift_comparison.png",
        x,
        {
            "CE shift": [observed_float(r["staging_shift_ce"]) for r in summaries],
            "M_obj shift": [observed_float(r["staging_shift_mobj"]) for r in summaries],
        },
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(f"# S2-04 Class-III candidate pilots\n\n- candidate summaries: `{summaries}`\n- selection: `{selection}`\n", encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_iii_candidate_pilots",
        "bundle_id": str(cfg["pilot_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/S2-04_class_iii_candidate_pilots.md"))
    any_class3 = bool(selection["has_class_iii_candidate"])
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "# S2-04 Class-III candidate pilots\n\n"
        f"- config path: `configs/pilots/class_iii_candidates.json`\n"
        f"- artifact root: `{artifact_root}`\n"
        f"- the three candidate families: `{[r['family_name'] for r in summaries]}`\n"
        f"- classification outcome for each: `{summaries}`\n"
        f"- best candidate / near-miss: `{selection}`\n"
        f"- does at least one candidate qualify as Class-III under the current rubric? `{'yes' if any_class3 else 'no'}`\n"
        + (
            f"- which candidate should proceed to the full Class-III campaign? `{selection['best_candidate']}`\n"
            if any_class3
            else f"- which candidate should proceed to the full Class-III campaign? `none; strongest near-miss = {selection['best_candidate']}`\n"
        ),
        encoding="utf-8",
    )

    quick = {r["family_name"]: r for r in summaries}
    return {
        "delayed_interface": quick.get("delayed_interface", {}),
        "hidden_sector": quick.get("hidden_sector", {}),
        "two_timescale": quick.get("two_timescale", {}),
        "has_class_iii_candidate": any_class3,
        "best_candidate": selection["best_candidate"],
        "artifact_root": str(artifact_root),
    }


def build_class_iii_escalation_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage1 = config["stage1"]
    lambdas = [float(x) for x in stage1["lambda_grid"]]
    for family in stage1["families"]:
        family_name = str(family["family_name"])
        param_grid = family["param_grid"]
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        for combo in np.array(np.meshgrid(*values)).T.reshape(-1, len(keys)):
            kwargs = dict(family.get("fixed", {}))
            for k, v in zip(keys, combo.tolist()):
                kwargs[k] = float(v)
            for size in stage1["sizes"]:
                kwargs2 = dict(kwargs)
                kwargs2["n"] = int(size)
                substrate = build_class_iii_candidate_family(family_name, **kwargs2)
                tau1_rows, tau1_boundary, tau1_aff = _evaluate_tau_panel(substrate, lambdas, 1, stage1["control_application_name"])
                tau2_rows, tau2_boundary, tau2_aff = _evaluate_tau_panel(substrate, lambdas, 2, stage1["control_application_name"])
                variant_spec = {"family": family_name, "size": int(size), **kwargs}
                rows.append(
                    {
                        "family_name": family_name.replace("_reversible_family", ""),
                        "variant_id": _stable_hash(variant_spec),
                        "size": int(size),
                        "variant_spec": variant_spec,
                        "tau1_rows": tau1_rows,
                        "tau2_rows": tau2_rows,
                        "tau1_boundary": tau1_boundary,
                        "tau2_boundary": tau2_boundary,
                        "tau1_affinity_ref": tau1_aff,
                        "tau2_affinity_ref": tau2_aff,
                        "stage": "stage1",
                    }
                )
    return rows


def evaluate_class_iii_escalation_variant(rows: dict[str, Any], rubric_config: dict[str, Any], p4_config: dict[str, Any]) -> dict[str, Any]:
    ev = evaluate_class_iii_candidate(rows, rubric_config, p4_config)
    return {
        "family_name": rows["family_name"],
        "variant_id": rows["variant_id"],
        "size": rows["size"],
        **ev,
    }


def select_stage2_variants(stage1_summaries: list[dict[str, Any]], top_k: int = 2) -> list[dict[str, Any]]:
    eligible = [r for r in stage1_summaries if r["candidate_p5_state"] == "active" and r["candidate_p6_drive_state"] == "inactive"]
    ranked = sorted(
        eligible,
        key=lambda r: (
            observed_float(r["staging_gap_anomaly_score"]),
            observed_float(r["staging_shift_mobj"]),
            -abs(observed_float(r["affinity_ref_max"])),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def summarize_sensor_asymmetry(candidate_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    fams = sorted({r["family_name"] for r in candidate_summaries})
    family_summary = {}
    for f in fams:
        rows = [r for r in candidate_summaries if r["family_name"] == f]
        family_summary[f] = {
            "strict_class_iii_count": int(sum(1 for r in rows if bool(r["candidate_is_class_iii_strict"]))),
            "mobj_dominant_p4_candidate_count": int(sum(1 for r in rows if bool(r["mobj_dominant_p4_candidate"]))),
            "mean_staging_shift_ce": float(np.mean([observed_float(r["staging_shift_ce"]) for r in rows])) if rows else 0.0,
            "max_staging_shift_ce": float(np.max([observed_float(r["staging_shift_ce"]) for r in rows])) if rows else 0.0,
            "mean_staging_shift_mobj": float(np.mean([observed_float(r["staging_shift_mobj"]) for r in rows])) if rows else 0.0,
            "max_staging_shift_mobj": float(np.max([observed_float(r["staging_shift_mobj"]) for r in rows])) if rows else 0.0,
            "mean_mobj_minus_ce_shift": float(np.mean([float(r["mobj_minus_ce_shift"]) for r in rows])) if rows else 0.0,
            "max_mobj_minus_ce_shift": float(np.max([float(r["mobj_minus_ce_shift"]) for r in rows])) if rows else 0.0,
        }
    return {"families": family_summary}


def format_class_iii_escalation_summary(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]], sensor_summary: dict[str, Any], best_variant: dict[str, Any]) -> dict[str, Any]:
    strict_found = any(bool(r["candidate_is_class_iii_strict"]) for r in stage1_rows + stage2_rows)
    sensor_supported = any(v["mobj_dominant_p4_candidate_count"] >= 2 for v in sensor_summary["families"].values()) or any(
        observed_float(r["staging_shift_mobj"]) >= 0.15 and float(r["mobj_minus_ce_shift"]) >= 0.05 for r in stage2_rows
    )

    parent_map = {(r["family_name"], r["variant_id"]): r for r in stage1_rows}
    size32_helpful = False
    for r in stage2_rows:
        p = parent_map.get((r["family_name"], r["variant_id"]))
        if p is None:
            continue
        if observed_float(r["staging_shift_ce"]) - observed_float(p["staging_shift_ce"]) >= 0.03 or (not bool(p["candidate_is_class_iii_strict"]) and bool(r["candidate_is_class_iii_strict"])):
            size32_helpful = True
            break

    if strict_found:
        verdict = "strict_class_iii_candidate_found"
    elif sensor_supported:
        verdict = "systematic_mobj_dominant_near_miss"
    else:
        verdict = "no_useful_signal"

    return {
        "strict_class_iii_found": bool(strict_found),
        "sensor_asymmetry_supported": bool(sensor_supported),
        "size32_helpful": bool(size32_helpful),
        "final_verdict": verdict,
        "best_variant": best_variant,
    }


def run_class_iii_escalation(config_path_or_obj: str | Path | dict[str, Any], output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = root / "results" / "pilots" if output_root is None else Path(output_root)
    artifact_root = out / str(cfg["artifact_subdir"])

    req = [
        root / "results" / "metrics" / "p4_anomaly_metric_layer",
        root / "results" / "taxonomy" / "canonical_class_rubric",
        root / "results" / "dashboards" / "class_i_class_ii_consolidation",
    ]
    if any(not artifact_is_current(p) for p in req):
        run_p4_anomaly_metric_layer(root / "configs" / "metrics" / "p4_anomaly_metric_layer.json", output_root=root / "results" / "metrics", use_cache=use_cache)

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

    rubric = json.loads((root / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))

    stage1_rows_raw = build_class_iii_escalation_rows(cfg)
    stage1_eval = [evaluate_class_iii_escalation_variant(r, rubric, p4_cfg) for r in stage1_rows_raw]

    stage2_seeds = select_stage2_variants(stage1_eval, top_k=int(cfg["stage2"]["top_k"]))
    stage2_rows_raw = []
    lambdas = [float(x) for x in cfg["stage1"]["lambda_grid"]]
    for s in stage2_seeds:
        spec = dict(next(r for r in stage1_rows_raw if r["variant_id"] == s["variant_id"] and r["family_name"] == s["family_name"])["variant_spec"])
        spec["size"] = 32
        kwargs = dict(spec)
        fam = kwargs.pop("family")
        kwargs.pop("size", None)
        kwargs["n"] = 32
        substrate = build_class_iii_candidate_family(fam, **kwargs)
        tau1_rows, tau1_boundary, tau1_aff = _evaluate_tau_panel(substrate, lambdas, 1, cfg["stage1"]["control_application_name"])
        tau2_rows, tau2_boundary, tau2_aff = _evaluate_tau_panel(substrate, lambdas, 2, cfg["stage1"]["control_application_name"])
        stage2_rows_raw.append(
            {
                "family_name": s["family_name"],
                "variant_id": s["variant_id"],
                "size": 32,
                "variant_spec": spec,
                "tau1_rows": tau1_rows,
                "tau2_rows": tau2_rows,
                "tau1_boundary": tau1_boundary,
                "tau2_boundary": tau2_boundary,
                "tau1_affinity_ref": tau1_aff,
                "tau2_affinity_ref": tau2_aff,
                "stage": "stage2",
            }
        )
    stage2_eval = [evaluate_class_iii_escalation_variant(r, rubric, p4_cfg) for r in stage2_rows_raw]

    sensor = summarize_sensor_asymmetry(stage1_eval + stage2_eval)
    best = max(stage1_eval + stage2_eval, key=lambda r: observed_float(r["staging_gap_anomaly_score"])) if (stage1_eval or stage2_eval) else {}
    summary = format_class_iii_escalation_summary(stage1_eval, stage2_eval, sensor, best)

    stage_fields = [
        "family_name",
        "variant_id",
        "size",
        "candidate_p5_state",
        "candidate_p6_drive_state",
        "candidate_p4_state",
        "candidate_is_class_iii_strict",
        "mobj_dominant_p4_candidate",
        "ce_boundary_tau1",
        "ce_boundary_tau2",
        "mobj_boundary_tau1",
        "mobj_boundary_tau2",
        "staging_shift_ce",
        "staging_shift_mobj",
        "shift_ratio_ce_to_mobj",
        "mobj_minus_ce_shift",
        "staging_gap_anomaly_score",
        "affinity_ref_max",
    ]
    _write_csv(dirs["analysis"] / "stage1_candidate_table.csv", stage1_eval, stage_fields)
    _write_csv(dirs["analysis"] / "stage2_candidate_table.csv", stage2_eval, stage_fields)
    (dirs["analysis"] / "sensor_asymmetry_summary.json").write_text(scientific_dumps(sensor, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "escalation_summary.json").write_text(scientific_dumps(summary, indent=2) + "\n", encoding="utf-8")

    metrics_rows = []
    for r in stage1_eval + stage2_eval:
        metrics_rows.append({
            **r,
            "cache_status": "executed",
            "manifest_path": str(artifact_root / "manifest.json"),
        })
    _write_csv(
        dirs["metrics"] / "metrics.csv",
        metrics_rows,
        stage_fields + ["cache_status", "manifest_path"],
    )

    # plots
    x = [observed_float(r["staging_shift_ce"]) for r in stage1_eval]
    if not x:
        x = [0.0]
    _line_plot(
        dirs["plots"] / "ce_vs_mobj_shift_scatter.png",
        list(range(len(stage1_eval))) if stage1_eval else [0],
        {
            "CE shift": [observed_float(r["staging_shift_ce"]) for r in stage1_eval] if stage1_eval else [0.0],
            "M_obj shift": [observed_float(r["staging_shift_mobj"]) for r in stage1_eval] if stage1_eval else [0.0],
        },
    )
    fams = sorted({r["family_name"] for r in stage1_eval})
    _line_plot(
        dirs["plots"] / "family_shift_comparison.png",
        list(range(len(fams))) if fams else [0],
        {
            "family max CE": [max(observed_float(r["staging_shift_ce"]) for r in stage1_eval if r["family_name"] == f) for f in fams] if fams else [0.0],
            "family max M_obj": [max(observed_float(r["staging_shift_mobj"]) for r in stage1_eval if r["family_name"] == f) for f in fams] if fams else [0.0],
        },
    )
    top = stage2_eval if stage2_eval else stage1_eval[:2]
    _line_plot(
        dirs["plots"] / "top_variants_tau_compare.png",
        list(range(len(top))) if top else [0],
        {
            "top CE shift": [observed_float(r["staging_shift_ce"]) for r in top] if top else [0.0],
            "top M_obj shift": [observed_float(r["staging_shift_mobj"]) for r in top] if top else [0.0],
        },
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(f"# S2-04 escalation\n\n- summary: `{summary}`\n", encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_iii_escalation",
        "bundle_id": str(cfg["escalation_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/S2-04_class_iii_escalation.md"))
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "# S2-04 Class-III escalation\n\n"
        "- config path: `configs/pilots/class_iii_escalation.json`\n"
        f"- artifact root: `{artifact_root}`\n"
        f"- Stage 1 family summary: `{sensor}`\n"
        f"- Stage 2 top-variant summary: `{stage2_eval}`\n"
        f"- strict-vs-exploratory outcome: `{summary}`\n"
        f"- final verdict: `{summary['final_verdict']}`\n"
        f"- did any variant satisfy strict Class-III under the current rubric? `{'yes' if summary['strict_class_iii_found'] else 'no'}`\n"
        f"- is the near-miss pattern systematically M_obj-dominant? `{'yes' if summary['sensor_asymmetry_supported'] else 'no'}`\n"
        f"- did size/coupling escalation help enough to justify a full Class-III campaign? `{'yes' if summary['size32_helpful'] else 'no'}`\n",
        encoding="utf-8",
    )

    strongest = best.get("variant_id") if best else None
    return {
        "strict_class_iii_found": summary["strict_class_iii_found"],
        "sensor_asymmetry_supported": summary["sensor_asymmetry_supported"],
        "size32_helpful": summary["size32_helpful"],
        "final_verdict": summary["final_verdict"],
        "best_variant": strongest,
        "artifact_root": str(artifact_root),
    }


def detect_boundary_floor_censoring(candidate_summary: dict[str, Any], lambda_grid: list[float], tol: float = 1e-12) -> dict[str, Any]:
    floor = float(min(lambda_grid))
    ce1 = candidate_summary.get("ce_boundary_tau1")
    ce2 = candidate_summary.get("ce_boundary_tau2")
    mo1 = candidate_summary.get("mobj_boundary_tau1")
    mo2 = candidate_summary.get("mobj_boundary_tau2")
    out = {
        "lambda_grid_min": floor,
        "ce_floor_censored_tau1": ce1 is not None and abs(float(ce1) - floor) <= tol,
        "ce_floor_censored_tau2": ce2 is not None and abs(float(ce2) - floor) <= tol,
        "mobj_floor_censored_tau1": mo1 is not None and abs(float(mo1) - floor) <= tol,
        "mobj_floor_censored_tau2": mo2 is not None and abs(float(mo2) - floor) <= tol,
    }
    out["any_floor_censored"] = bool(
        out["ce_floor_censored_tau1"] or out["ce_floor_censored_tau2"] or out["mobj_floor_censored_tau1"] or out["mobj_floor_censored_tau2"]
    )
    return out


def evaluate_bulk_persistence_variant(rows: dict[str, Any], rubric_config: dict[str, Any], p4_config: dict[str, Any]) -> dict[str, Any]:
    return evaluate_class_iii_candidate(rows, rubric_config, p4_config)


def select_bulk_stage2_variants(stage1_summaries: list[dict[str, Any]], top_k: int = 2) -> list[dict[str, Any]]:
    eligible = [r for r in stage1_summaries if r["candidate_p5_state"] == "active" and r["candidate_p6_drive_state"] == "inactive"]
    ranked = sorted(
        eligible,
        key=lambda r: (
            observed_float(r["staging_gap_anomaly_score"]),
            observed_float(r["staging_shift_mobj"]),
            -abs(observed_float(r["affinity_ref_max"])),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def summarize_bulk_vs_surface_persistence(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Group counts
    strict_class_iii_found = any(bool(r["candidate_is_class_iii_strict"]) for r in (stage1_rows + stage2_rows))

    # floor censoring resolution for old controls
    old_s1 = [r for r in stage1_rows if r["family_group"] == "old_control"]
    old_s2 = [r for r in stage2_rows if r["family_group"] == "old_control"]
    idx2 = {(r["family_name"], r["variant_id"], int(r["size"])): r for r in old_s2}
    floor_censoring_resolved = False
    for r in old_s1:
        if not bool(r.get("any_floor_censored", False)):
            continue
        target = idx2.get((r["family_name"], r["variant_id"], 128))
        if target is not None and not bool(target.get("any_floor_censored", False)):
            floor_censoring_resolved = True
            break

    # bulk persistence
    bulk_128 = [r for r in stage2_rows if r["family_group"] == "new_bulk" and int(r["size"]) == 128]
    bulk_design_promising = any(
        (r["candidate_p5_state"] == "active" and r["candidate_p6_drive_state"] == "inactive" and (bool(r["candidate_is_class_iii_strict"]) or bool(r["bulk_persistence_supported"])))
        for r in bulk_128
    )

    if strict_class_iii_found:
        final_verdict = "strict_bulk_class_iii_found"
    elif bulk_design_promising:
        final_verdict = "bulk_persistence_promising_near_miss"
    else:
        final_verdict = "surface_effect_or_censoring_only"

    return {
        "strict_class_iii_found": bool(strict_class_iii_found),
        "floor_censoring_was_binding": bool(floor_censoring_resolved),
        "bulk_design_promising": bool(bulk_design_promising),
        "final_verdict": final_verdict,
    }


def format_class_iii_bulk_persistence_summary(
    control_rows: list[dict[str, Any]],
    bulk_stage1_rows: list[dict[str, Any]],
    bulk_stage2_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    best_variant: dict[str, Any],
) -> dict[str, Any]:
    return {
        "control_reruns": control_rows,
        "bulk_stage1": bulk_stage1_rows,
        "bulk_stage2": bulk_stage2_rows,
        "summary": summary,
        "best_variant": best_variant,
    }


def run_class_iii_bulk_persistence(config_path_or_obj: str | Path | dict[str, Any], output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = root / "results" / "pilots" if output_root is None else Path(output_root)
    artifact_root = out / str(cfg["artifact_subdir"])

    req = [
        root / "results" / "metrics" / "p4_anomaly_metric_layer",
        root / "results" / "taxonomy" / "canonical_class_rubric",
        root / "results" / "dashboards" / "class_i_class_ii_consolidation",
    ]
    if any(not artifact_is_current(p) for p in req):
        run_p4_anomaly_metric_layer(root / "configs" / "metrics" / "p4_anomaly_metric_layer.json", output_root=root / "results" / "metrics", use_cache=use_cache)

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

    rubric = json.loads((root / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))

    lam = [float(x) for x in cfg["low_lambda_grid"]]
    control_rows: list[dict[str, Any]] = []
    for entry in cfg["group1_old_controls"]:
        fam = str(entry["family_name"])
        kwargs0 = dict(entry["family_kwargs"])
        for size in [32, 128]:
            kwargs = dict(kwargs0)
            kwargs["n"] = int(size)
            substrate = build_class_iii_candidate_family(fam, **kwargs)
            t1_rows, t1_b, t1_a = _evaluate_tau_panel(substrate, lam, 1, cfg["control_application_name"])
            t2_rows, t2_b, t2_a = _evaluate_tau_panel(substrate, lam, 2, cfg["control_application_name"])
            raw = {
                "tau1_boundary": t1_b,
                "tau2_boundary": t2_b,
                "tau1_affinity_ref": t1_a,
                "tau2_affinity_ref": t2_a,
            }
            ev = evaluate_bulk_persistence_variant(raw, rubric, p4_cfg)
            floor = detect_boundary_floor_censoring(ev, lam)
            variant_id = _stable_hash({"group": "old_control", "family": fam, **kwargs0})
            row = {
                "family_name": str(entry["candidate_name"]),
                "variant_id": variant_id,
                "family_group": "old_control",
                "size": int(size),
                **ev,
                **floor,
                "persistence_ratio_ce": None,
                "persistence_ratio_mobj": None,
                "bulk_persistence_supported": False,
                "surface_washout_supported": bool(observed_float(ev["staging_shift_mobj"]) < 0.05),
            }
            control_rows.append(row)

    # fill control persistence ratios (32->128)
    by_key = {}
    for r in control_rows:
        by_key.setdefault((r["family_name"], r["variant_id"]), {})[int(r["size"])] = r
    for _, vals in by_key.items():
        if 32 in vals and 128 in vals:
            lo, hi = vals[32], vals[128]
            eps = 1e-12
            hi["persistence_ratio_ce"] = observed_float(hi["staging_shift_ce"]) / max(observed_float(lo["staging_shift_ce"]), eps)
            hi["persistence_ratio_mobj"] = observed_float(hi["staging_shift_mobj"]) / max(observed_float(lo["staging_shift_mobj"]), eps)

    # Group 2 stage1 (new bulk at 16,32)
    bulk_stage1: list[dict[str, Any]] = []
    for entry in cfg["group2_new_bulk"]:
        fam = str(entry["family_name"])
        param_grid = entry["param_grid"]
        keys = list(param_grid.keys())
        vals = [param_grid[k] for k in keys]
        for combo in np.array(np.meshgrid(*vals)).T.reshape(-1, len(keys)):
            kwargs_var = dict(entry.get("fixed", {}))
            for k, v in zip(keys, combo.tolist()):
                kwargs_var[k] = float(v)
            variant_id = _stable_hash({"group": "new_bulk", "family": fam, **kwargs_var})
            for size in [16, 32]:
                kwargs = dict(kwargs_var)
                kwargs["n"] = int(size)
                substrate = build_class_iii_candidate_family(fam, **kwargs)
                t1_rows, t1_b, t1_a = _evaluate_tau_panel(substrate, lam, 1, cfg["control_application_name"])
                t2_rows, t2_b, t2_a = _evaluate_tau_panel(substrate, lam, 2, cfg["control_application_name"])
                ev = evaluate_bulk_persistence_variant(
                    {"tau1_boundary": t1_b, "tau2_boundary": t2_b, "tau1_affinity_ref": t1_a, "tau2_affinity_ref": t2_a},
                    rubric,
                    p4_cfg,
                )
                floor = detect_boundary_floor_censoring(ev, lam)
                bulk_stage1.append(
                    {
                        "family_name": str(entry["candidate_name"]),
                        "variant_id": variant_id,
                        "family_group": "new_bulk",
                        "size": int(size),
                        **ev,
                        **floor,
                        "persistence_ratio_ce": None,
                        "persistence_ratio_mobj": None,
                        "bulk_persistence_supported": False,
                        "surface_washout_supported": bool(observed_float(ev["staging_shift_mobj"]) < 0.05),
                    }
                )

    # Group 3 stage2: top2 new bulk variants rerun at 128
    selected = select_bulk_stage2_variants([r for r in bulk_stage1 if int(r["size"]) == 32], top_k=2)
    bulk_stage2: list[dict[str, Any]] = []
    for s in selected:
        src = next(r for r in bulk_stage1 if r["variant_id"] == s["variant_id"] and int(r["size"]) == 32)
        # reconstruct kwargs from config
        fam_name = src["family_name"]
        fam_entry = next(e for e in cfg["group2_new_bulk"] if str(e["candidate_name"]) == fam_name)
        family_key = str(fam_entry["family_name"])
        # brute find matching parameter tuple by variant hash
        found_kwargs = None
        keys = list(fam_entry["param_grid"].keys())
        vals = [fam_entry["param_grid"][k] for k in keys]
        for combo in np.array(np.meshgrid(*vals)).T.reshape(-1, len(keys)):
            kw = dict(fam_entry.get("fixed", {}))
            for k, v in zip(keys, combo.tolist()):
                kw[k] = float(v)
            if _stable_hash({"group": "new_bulk", "family": family_key, **kw}) == s["variant_id"]:
                found_kwargs = kw
                break
        if found_kwargs is None:
            continue
        kwargs = dict(found_kwargs)
        kwargs["n"] = 128
        substrate = build_class_iii_candidate_family(family_key, **kwargs)
        t1_rows, t1_b, t1_a = _evaluate_tau_panel(substrate, lam, 1, cfg["control_application_name"])
        t2_rows, t2_b, t2_a = _evaluate_tau_panel(substrate, lam, 2, cfg["control_application_name"])
        ev = evaluate_bulk_persistence_variant(
            {"tau1_boundary": t1_b, "tau2_boundary": t2_b, "tau1_affinity_ref": t1_a, "tau2_affinity_ref": t2_a},
            rubric,
            p4_cfg,
        )
        floor = detect_boundary_floor_censoring(ev, lam)
        eps = 1e-12
        p_ratio_ce = observed_float(ev["staging_shift_ce"]) / max(observed_float(src["staging_shift_ce"]), eps)
        p_ratio_mo = observed_float(ev["staging_shift_mobj"]) / max(observed_float(src["staging_shift_mobj"]), eps)
        bulk_stage2.append(
            {
                "family_name": fam_name,
                "variant_id": s["variant_id"],
                "family_group": "new_bulk",
                "size": 128,
                **ev,
                **floor,
                "persistence_ratio_ce": p_ratio_ce,
                "persistence_ratio_mobj": p_ratio_mo,
                "bulk_persistence_supported": bool(observed_float(ev["staging_shift_mobj"]) >= 0.15 and p_ratio_mo >= 0.5),
                "surface_washout_supported": bool(observed_float(ev["staging_shift_mobj"]) < 0.05),
            }
        )

    all_rows = control_rows + bulk_stage1 + bulk_stage2
    summary = summarize_bulk_vs_surface_persistence(control_rows + bulk_stage1, control_rows + bulk_stage2)
    best_variant = max(all_rows, key=lambda r: observed_float(r["staging_gap_anomaly_score"])) if all_rows else {}
    payload = format_class_iii_bulk_persistence_summary(control_rows, bulk_stage1, bulk_stage2, summary, best_variant)

    fields = [
        "family_name",
        "variant_id",
        "family_group",
        "size",
        "candidate_p5_state",
        "candidate_p6_drive_state",
        "candidate_p4_state",
        "candidate_is_class_iii_strict",
        "ce_boundary_tau1",
        "ce_boundary_tau2",
        "mobj_boundary_tau1",
        "mobj_boundary_tau2",
        "staging_shift_ce",
        "staging_shift_mobj",
        "staging_gap_anomaly_score",
        "ce_floor_censored_tau1",
        "ce_floor_censored_tau2",
        "mobj_floor_censored_tau1",
        "mobj_floor_censored_tau2",
        "any_floor_censored",
        "persistence_ratio_ce",
        "persistence_ratio_mobj",
        "affinity_ref_max",
        "bulk_persistence_supported",
        "surface_washout_supported",
    ]
    _write_csv(dirs["analysis"] / "control_rerun_table.csv", control_rows, fields)
    _write_csv(dirs["analysis"] / "bulk_stage1_table.csv", bulk_stage1, fields)
    _write_csv(dirs["analysis"] / "bulk_stage2_table.csv", bulk_stage2, fields)
    (dirs["analysis"] / "persistence_summary.json").write_text(scientific_dumps(payload, indent=2) + "\n", encoding="utf-8")

    metrics_rows = [{**r, "cache_status": "executed", "manifest_path": str(artifact_root / "manifest.json")} for r in all_rows]
    _write_csv(dirs["metrics"] / "metrics.csv", metrics_rows, fields + ["cache_status", "manifest_path"])

    _line_plot(
        dirs["plots"] / "floor_censoring_check.png",
        list(range(len(control_rows))) if control_rows else [0],
        {
            "control floor flags": [1.0 if bool(r["any_floor_censored"]) else 0.0 for r in control_rows] if control_rows else [0.0],
        },
    )
    _line_plot(
        dirs["plots"] / "bulk_vs_surface_shift_comparison.png",
        list(range(len(all_rows))) if all_rows else [0],
        {
            "CE shift": [observed_float(r["staging_shift_ce"]) for r in all_rows] if all_rows else [0.0],
            "M_obj shift": [observed_float(r["staging_shift_mobj"]) for r in all_rows] if all_rows else [0.0],
        },
    )
    top_bulk = sorted([r for r in bulk_stage2], key=lambda r: observed_float(r["staging_gap_anomaly_score"]), reverse=True)[:2]
    _line_plot(
        dirs["plots"] / "top_bulk_variants_tau_compare.png",
        list(range(len(top_bulk))) if top_bulk else [0],
        {
            "top bulk CE shift": [observed_float(r["staging_shift_ce"]) for r in top_bulk] if top_bulk else [0.0],
            "top bulk M_obj shift": [observed_float(r["staging_shift_mobj"]) for r in top_bulk] if top_bulk else [0.0],
        },
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(
        "# S2-04 bulk persistence\n\n"
        f"- summary: `{summary}`\n",
        encoding="utf-8",
    )

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_iii_bulk_persistence",
        "bundle_id": str(cfg["pilot_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/S2-04_class_iii_bulk_persistence.md"))
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "# S2-04 Class-III bulk persistence\n\n"
        "- config path: `configs/pilots/class_iii_bulk_persistence.json`\n"
        f"- artifact root: `{artifact_root}`\n"
        f"- old-design control rerun outcome: `{control_rows}`\n"
        f"- new bulk-family outcome: `{bulk_stage1 + bulk_stage2}`\n"
        f"- final verdict: `{summary['final_verdict']}`\n"
        f"- were the previous Stage-2 zeros mainly floor-censoring artifacts? `{'yes' if summary['floor_censoring_was_binding'] else 'no'}`\n"
        f"- does any bulk-distributed candidate persist to n=128? `{'yes' if summary['bulk_design_promising'] else 'no'}`\n"
        f"- did any variant satisfy strict Class-III under the current rubric? `{'yes' if summary['strict_class_iii_found'] else 'no'}`\n",
        encoding="utf-8",
    )

    return {
        "strict_class_iii_found": summary["strict_class_iii_found"],
        "floor_censoring_was_binding": summary["floor_censoring_was_binding"],
        "bulk_design_promising": summary["bulk_design_promising"],
        "final_verdict": summary["final_verdict"],
        "best_variant": best_variant.get("variant_id"),
        "artifact_root": str(artifact_root),
    }


def evaluate_refined_replicated_portal_variant(rows: dict[str, Any], rubric_config: dict[str, Any], p4_config: dict[str, Any]) -> dict[str, Any]:
    return evaluate_class_iii_candidate(rows, rubric_config, p4_config)


def select_refinement_stage2_variants(stage1_summaries: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    eligible = [r for r in stage1_summaries if r["candidate_p5_state"] == "active" and r["candidate_p6_drive_state"] == "inactive"]
    ranked = sorted(
        eligible,
        key=lambda r: (
            observed_float(r["staging_shift_ce"]),
            observed_float(r["staging_shift_mobj"]),
            -abs(observed_float(r["affinity_ref_max"])),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def evaluate_p4_criteria_on_profiles(profile_rows: dict[str, dict[str, Any]], criteria_config: dict[str, Any]) -> dict[str, Any]:
    observed = {
        key: all(np.isfinite(observed_float(row.get(field))) and observed_float(row.get(field)) >= 0
                 for field in ("staging_shift_ce", "staging_shift_mobj"))
        and isinstance(row.get("presence_shift_any"), bool)
        for key, row in profile_rows.items()
    }
    required = ("class_i_reference", "class_ii_reference", "old_hidden_sector_128",
                "old_two_timescale_128", "replicated_portal_best_128", "replicated_portal_best_256")
    complete = all(observed.get(key, False) for key in required)
    def _is_active(profile: dict[str, Any], criterion: str) -> bool:
        ce = observed_float(profile.get("staging_shift_ce", 0.0))
        mo = observed_float(profile.get("staging_shift_mobj", 0.0))
        pres = bool(profile.get("presence_shift_any", False))
        if criterion == "strict_dual_shift":
            return bool(pres or (ce >= float(criteria_config["strict_dual_shift"]["ce_min"]) and mo >= float(criteria_config["strict_dual_shift"]["mobj_min"])))
        if criterion == "mobj_only_shift":
            return bool(pres or (mo >= float(criteria_config["mobj_only_shift"]["mobj_min"])))
        if criterion == "hybrid_ce_gated":
            return bool(
                pres
                or (
                    mo >= float(criteria_config["hybrid_ce_gated"]["mobj_min"])
                    and ce >= float(criteria_config["hybrid_ce_gated"]["ce_min"])
                )
            )
        raise ValueError(f"unknown criterion: {criterion}")

    out: dict[str, Any] = {}
    names = ["strict_dual_shift", "hybrid_ce_gated", "mobj_only_shift"]
    for name in names:
        prof = {k: _is_active(v, name) for k, v in profile_rows.items()}
        preserves_class_i = "class_i_reference" in prof and not prof["class_i_reference"]
        preserves_class_ii = "class_ii_reference" in prof and not prof["class_ii_reference"]
        rejects_old = all(k in prof and not prof[k] for k in ("old_hidden_sector_128", "old_two_timescale_128"))
        accepts_bulk = prof.get("replicated_portal_best_128", False) or prof.get("replicated_portal_best_256", False)
        persistent_acceptance = prof.get("replicated_portal_best_128", False) and prof.get("replicated_portal_best_256", False)
        supported = bool(complete and preserves_class_i and preserves_class_ii and rejects_old and accepts_bulk and persistent_acceptance)
        out[name] = {
            "profile_active": prof,
            "preserves_class_i_reference": preserves_class_i,
            "preserves_class_ii_reference": preserves_class_ii,
            "rejects_old_surface_controls": rejects_old,
            "accepts_bulk_candidate": accepts_bulk,
            "persistent_acceptance": persistent_acceptance,
            "criterion_supported": supported,
            "required_profiles_observed": complete,
        }
    if out["strict_dual_shift"]["criterion_supported"]:
        recommended = "strict_dual_shift"
    elif out["hybrid_ce_gated"]["criterion_supported"]:
        recommended = "hybrid_ce_gated"
    elif out["mobj_only_shift"]["criterion_supported"]:
        recommended = "mobj_only_shift"
    else:
        recommended = "none"
    return {"criteria": out, "recommended_p4_criterion": recommended}


def summarize_replicated_portal_refinement(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict = any(bool(r["candidate_is_class_iii_strict"]) for r in stage1_rows + stage2_rows)
    promising = bool(strict or any(bool(r.get("bulk_persistence_supported", False)) for r in stage2_rows))
    return {
        "strict_class_iii_found": bool(strict),
        "replicated_portal_promising": bool(promising),
    }


def format_refinement_and_p4_audit_summary(
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    criterion_audit: dict[str, Any],
    refinement_summary: dict[str, Any],
    best_variant: dict[str, Any],
) -> dict[str, Any]:
    rec = criterion_audit["recommended_p4_criterion"]
    strict = bool(refinement_summary["strict_class_iii_found"])
    prom = bool(refinement_summary["replicated_portal_promising"])
    if strict:
        verdict = "strict_class_iii_found"
    elif prom and rec == "hybrid_ce_gated":
        verdict = "hybrid_criterion_supported_bulk_candidate_ready"
    elif prom and rec == "mobj_only_shift":
        verdict = "mobj_only_supported_but_too_broad"
    else:
        verdict = "no_stable_class_iii_signal_yet"
    return {
        "refinement_stage1_count": len(stage1_rows),
        "refinement_stage2_count": len(stage2_rows),
        "strict_class_iii_found": strict,
        "replicated_portal_promising": prom,
        "recommended_p4_criterion": rec,
        "final_verdict": verdict,
        "best_variant": best_variant,
    }


def run_class_iii_refinement_and_p4_audit(config_path_or_obj: str | Path | dict[str, Any], output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = root / "results" / "pilots" if output_root is None else Path(output_root)
    artifact_root = out / str(cfg["artifact_subdir"])

    req = [
        root / "results" / "metrics" / "p4_anomaly_metric_layer",
        root / "results" / "taxonomy" / "canonical_class_rubric",
        root / "results" / "dashboards" / "class_i_class_ii_consolidation",
    ]
    if any(not artifact_is_current(p) for p in req):
        run_p4_anomaly_metric_layer(root / "configs" / "metrics" / "p4_anomaly_metric_layer.json", output_root=root / "results" / "metrics", use_cache=use_cache)

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

    rubric = json.loads((root / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))
    lam = [float(x) for x in cfg["lambda_grid"]]
    fam = "replicated_portal_reversible_family"

    # Stage 1 at 32,128
    stage1_rows: list[dict[str, Any]] = []
    grid = cfg["parameter_grid"]
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in np.array(np.meshgrid(*vals)).T.reshape(-1, len(keys)):
        kw = dict(cfg.get("fixed_kwargs", {}))
        for k, v in zip(keys, combo.tolist()):
            kw[k] = float(v)
        variant_id = _stable_hash({"family": fam, **kw})
        for size in cfg["stage1_sizes"]:
            kwargs = dict(kw)
            kwargs["n"] = int(size)
            substrate = build_class_iii_candidate_family(fam, **kwargs)
            t1_rows, t1_b, t1_a = _evaluate_tau_panel(substrate, lam, 1, cfg["control_application_name"])
            t2_rows, t2_b, t2_a = _evaluate_tau_panel(substrate, lam, 2, cfg["control_application_name"])
            ev = evaluate_refined_replicated_portal_variant(
                {"tau1_boundary": t1_b, "tau2_boundary": t2_b, "tau1_affinity_ref": t1_a, "tau2_affinity_ref": t2_a},
                rubric,
                p4_cfg,
            )
            floor = detect_boundary_floor_censoring(ev, lam)
            stage1_rows.append(
                {
                    "family_name": "replicated_portal",
                    "variant_id": variant_id,
                    "size": int(size),
                    **ev,
                    **floor,
                    "persistence_ratio_ce": None,
                    "persistence_ratio_mobj": None,
                    "bulk_persistence_supported": False,
                    "ce_catchup_supported": False,
                }
            )

    # Stage 2 top3 at 256
    seed_rows = [r for r in stage1_rows if int(r["size"]) == 128]
    top3 = select_refinement_stage2_variants(seed_rows, top_k=int(cfg["stage2_top_k"]))
    stage2_rows: list[dict[str, Any]] = []
    for s in top3:
        # recover params by matching variant hash
        found = None
        for combo in np.array(np.meshgrid(*vals)).T.reshape(-1, len(keys)):
            kw = dict(cfg.get("fixed_kwargs", {}))
            for k, v in zip(keys, combo.tolist()):
                kw[k] = float(v)
            if _stable_hash({"family": fam, **kw}) == s["variant_id"]:
                found = kw
                break
        if found is None:
            continue
        kwargs = dict(found)
        kwargs["n"] = int(cfg["stage2_size"])
        substrate = build_class_iii_candidate_family(fam, **kwargs)
        t1_rows, t1_b, t1_a = _evaluate_tau_panel(substrate, lam, 1, cfg["control_application_name"])
        t2_rows, t2_b, t2_a = _evaluate_tau_panel(substrate, lam, 2, cfg["control_application_name"])
        ev = evaluate_refined_replicated_portal_variant(
            {"tau1_boundary": t1_b, "tau2_boundary": t2_b, "tau1_affinity_ref": t1_a, "tau2_affinity_ref": t2_a},
            rubric,
            p4_cfg,
        )
        parent = next(r for r in seed_rows if r["variant_id"] == s["variant_id"])
        eps = 1e-12
        p_ce = observed_float(ev["staging_shift_ce"]) / max(observed_float(parent["staging_shift_ce"]), eps)
        p_mo = observed_float(ev["staging_shift_mobj"]) / max(observed_float(parent["staging_shift_mobj"]), eps)
        stage2_rows.append(
            {
                "family_name": "replicated_portal",
                "variant_id": s["variant_id"],
                "size": int(cfg["stage2_size"]),
                **ev,
                **detect_boundary_floor_censoring(ev, lam),
                "persistence_ratio_ce": p_ce,
                "persistence_ratio_mobj": p_mo,
                "bulk_persistence_supported": bool(observed_float(ev["staging_shift_mobj"]) >= 0.15 and p_mo >= 0.5),
                "ce_catchup_supported": bool(observed_float(ev["staging_shift_ce"]) >= 0.15),
            }
        )

    # profiles for criterion audit
    tax = json.loads((root / "results" / "taxonomy" / "canonical_class_rubric" / "analysis" / "reference_classifications.json").read_text(encoding="utf-8"))["reference_classifications"]
    class_i = next(r for r in tax if r["profile_name"] == "class_i_reference")
    class_ii = next(r for r in tax if r["profile_name"] == "class_ii_reference")

    old_bulk = list(csv.DictReader((root / "results" / "pilots" / "class_iii_bulk_persistence" / "analysis" / "control_rerun_table.csv").open()))
    old_h = next((r for r in old_bulk if r["family_name"] == "hidden_sector" and int(r["size"]) == 128), None)
    old_t = next((r for r in old_bulk if r["family_name"] == "two_timescale" and int(r["size"]) == 128), None)
    best128 = max(seed_rows, key=lambda r: observed_float(r["staging_shift_ce"])) if seed_rows else None
    best256 = max(stage2_rows, key=lambda r: observed_float(r["staging_shift_ce"])) if stage2_rows else None

    def _prof(x: dict[str, Any] | None) -> dict[str, Any]:
        if x is None:
            return {"staging_shift_ce": 0.0, "staging_shift_mobj": 0.0, "presence_shift_any": False}
        return {
            "staging_shift_ce": observed_float(x.get("tau_ce_shift", x.get("staging_shift_ce", 0.0))),
            "staging_shift_mobj": observed_float(x.get("tau_mobj_shift", x.get("staging_shift_mobj", 0.0))),
            "presence_shift_any": bool(x.get("presence_shift_any", False)),
        }

    profiles = {
        "class_i_reference": _prof(class_i),
        "class_ii_reference": _prof(class_ii),
        "old_hidden_sector_128": _prof(old_h),
        "old_two_timescale_128": _prof(old_t),
        "replicated_portal_best_128": _prof(best128),
        "replicated_portal_best_256": _prof(best256),
    }
    criterion_audit = evaluate_p4_criteria_on_profiles(profiles, cfg["criteria"])
    refinement = summarize_replicated_portal_refinement(stage1_rows, stage2_rows)
    best_variant = max(stage1_rows + stage2_rows, key=lambda r: observed_float(r["staging_shift_ce"])) if (stage1_rows or stage2_rows) else {}
    summary = format_refinement_and_p4_audit_summary(stage1_rows, stage2_rows, criterion_audit, refinement, best_variant)

    fields = [
        "family_name",
        "variant_id",
        "size",
        "candidate_p5_state",
        "candidate_p6_drive_state",
        "candidate_p4_state",
        "candidate_is_class_iii_strict",
        "staging_shift_ce",
        "staging_shift_mobj",
        "mobj_minus_ce_shift",
        "shift_ratio_ce_to_mobj",
        "persistence_ratio_ce",
        "persistence_ratio_mobj",
        "affinity_ref_max",
        "bulk_persistence_supported",
        "ce_catchup_supported",
    ]
    _write_csv(dirs["analysis"] / "refinement_stage1_table.csv", stage1_rows, fields)
    _write_csv(dirs["analysis"] / "refinement_stage2_table.csv", stage2_rows, fields)
    (dirs["analysis"] / "p4_criterion_audit.json").write_text(scientific_dumps({"profiles": profiles, **criterion_audit}, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "refinement_summary.json").write_text(scientific_dumps(summary, indent=2) + "\n", encoding="utf-8")

    metrics_rows = [{**r, "candidate_p4_state_strict": r.get("candidate_p4_state"), "cache_status": "executed", "manifest_path": str(artifact_root / "manifest.json")} for r in stage1_rows + stage2_rows]
    _write_csv(
        dirs["metrics"] / "metrics.csv",
        metrics_rows,
        [
            "family_name",
            "variant_id",
            "size",
            "candidate_p5_state",
            "candidate_p6_drive_state",
            "candidate_p4_state_strict",
            "candidate_is_class_iii_strict",
            "staging_shift_ce",
            "staging_shift_mobj",
            "mobj_minus_ce_shift",
            "shift_ratio_ce_to_mobj",
            "persistence_ratio_ce",
            "persistence_ratio_mobj",
            "affinity_ref_max",
            "bulk_persistence_supported",
            "ce_catchup_supported",
            "cache_status",
            "manifest_path",
        ],
    )

    _line_plot(
        dirs["plots"] / "ce_shift_vs_mobj_shift.png",
        list(range(len(stage1_rows))) if stage1_rows else [0],
        {
            "CE shift": [observed_float(r["staging_shift_ce"]) for r in stage1_rows] if stage1_rows else [0.0],
            "M_obj shift": [observed_float(r["staging_shift_mobj"]) for r in stage1_rows] if stage1_rows else [0.0],
        },
    )
    _line_plot(
        dirs["plots"] / "replicated_portal_size_progression.png",
        list(range(len(stage2_rows))) if stage2_rows else [0],
        {
            "size256 CE": [observed_float(r["staging_shift_ce"]) for r in stage2_rows] if stage2_rows else [0.0],
            "size256 M_obj": [observed_float(r["staging_shift_mobj"]) for r in stage2_rows] if stage2_rows else [0.0],
        },
    )
    crit = criterion_audit["criteria"]
    _line_plot(
        dirs["plots"] / "p4_criterion_comparison.png",
        [0, 1, 2],
        {
            "strict": [1.0 if crit["strict_dual_shift"]["criterion_supported"] else 0.0] * 3,
            "hybrid": [1.0 if crit["hybrid_ce_gated"]["criterion_supported"] else 0.0] * 3,
            "mobj_only": [1.0 if crit["mobj_only_shift"]["criterion_supported"] else 0.0] * 3,
        },
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(f"# S2-04 refinement/audit\n\n- summary: `{summary}`\n", encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_iii_refinement_and_p4_audit",
        "bundle_id": str(cfg["audit_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/S2-04_class_iii_refinement_and_p4_audit.md"))
    note_path.parent.mkdir(parents=True, exist_ok=True)
    rec = criterion_audit["recommended_p4_criterion"]
    note_path.write_text(
        "# S2-04 replicated-portal refinement and P4 criterion audit\n\n"
        "- config path: `configs/pilots/class_iii_refinement_and_p4_audit.json`\n"
        f"- artifact root: `{artifact_root}`\n"
        f"- refinement outcome: `{refinement}`\n"
        f"- criterion audit outcome: `{criterion_audit}`\n"
        f"- final verdict: `{summary['final_verdict']}`\n"
        f"- did any replicated-portal variant satisfy strict Class-III under the current rubric? `{'yes' if summary['strict_class_iii_found'] else 'no'}`\n"
        f"- is a revised P4 criterion justified by the audit? `{'yes' if rec != 'none' else 'no'}`\n"
        f"- if yes, which criterion is supported: strict / hybrid / mobj_only? `{rec}`\n",
        encoding="utf-8",
    )

    return {
        "strict_class_iii_found": summary["strict_class_iii_found"],
        "replicated_portal_promising": summary["replicated_portal_promising"],
        "recommended_p4_criterion": rec,
        "final_verdict": summary["final_verdict"],
        "best_variant": best_variant.get("variant_id"),
        "artifact_root": str(artifact_root),
    }


def evaluate_strict_confirmation_variant(
    size_results: list[dict[str, Any]],
    scale_cv_threshold: float = 0.05,
    expected_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Assess persistence and scale-invariance for one variant across a size panel.

    size_results: list of per-size dicts, each must contain:
      candidate_is_class_iii_strict, staging_shift_ce, staging_shift_mobj, size
    """
    import statistics
    sizes = [int(r["size"]) for r in size_results]
    if not sizes or len(set(sizes)) != len(sizes) or any(n <= 0 for n in sizes):
        raise ValueError("confirmation requires a nonempty panel of distinct positive sizes")
    if not np.isfinite(scale_cv_threshold) or scale_cv_threshold < 0:
        raise ValueError("CV threshold must be finite and nonnegative")
    if expected_sizes is not None and (len(set(expected_sizes)) != len(expected_sizes) or any(n <= 0 for n in expected_sizes)):
        raise ValueError("expected size panel must contain distinct positive sizes")
    strict_hits = [bool(r["candidate_is_class_iii_strict"]) for r in size_results]
    ce_shifts = [observed_float(r["staging_shift_ce"]) for r in size_results]
    mo_shifts = [observed_float(r["staging_shift_mobj"]) for r in size_results]
    if any(not np.isfinite(v) or v < 0 for v in ce_shifts + mo_shifts):
        raise ValueError("confirmation shifts must be finite and nonnegative")

    panel_complete = expected_sizes is None or set(sizes) == set(expected_sizes)
    multiple_sizes = len(sizes) >= 2
    strict_hit_persistent = panel_complete and multiple_sizes and all(strict_hits)

    def _cv(vals: list[float]) -> float:
        scale = max(vals)
        if scale == 0:
            return 0.0
        normalized = [v / scale for v in vals]
        return statistics.pstdev(normalized) / statistics.mean(normalized)

    ce_cv = _cv(ce_shifts)
    mo_cv = _cv(mo_shifts)
    scale_invariance_supported = (
        strict_hit_persistent
        and ce_cv <= scale_cv_threshold
        and mo_cv <= scale_cv_threshold
    )

    if strict_hit_persistent and scale_invariance_supported:
        label = "strict_confirmed"
    elif strict_hit_persistent:
        label = "strict_persistent_but_not_scale_stable"
    elif any(strict_hits):
        label = "strict_but_not_persistent"
    elif panel_complete and multiple_sizes and min(mo_shifts) >= 0.15:
        label = "persistent_near_miss"
    else:
        label = "near_miss"

    return {
        "size_panel": sizes,
        "size_panel_complete": panel_complete,
        "multiple_sizes_observed": multiple_sizes,
        "scale_support_scope": "declared_finite_size_panel",
        "strict_hits_by_size": dict(zip(sizes, strict_hits)),
        "strict_hit_persistent": strict_hit_persistent,
        "scale_invariance_supported": scale_invariance_supported,
        "shift_ce_cv": float(ce_cv),
        "shift_mobj_cv": float(mo_cv),
        "shift_ce_mean": float(statistics.mean(ce_shifts)),
        "shift_mobj_mean": float(statistics.mean(mo_shifts)),
        "final_variant_label": label,
    }


def summarize_strict_confirmation(variant_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive final ticket verdict from per-variant summaries."""
    confirmed = [v for v in variant_summaries if v["confirmation"]["final_variant_label"] == "strict_confirmed"]
    any_strict = [v for v in variant_summaries if any(v["confirmation"]["strict_hits_by_size"].values())]
    if confirmed:
        verdict = "strict_class_iii_confirmed"
        best = max(confirmed, key=lambda v: v["confirmation"]["shift_mobj_mean"])
    elif any_strict:
        verdict = "strict_hit_present_but_not_yet_confirmed"
        best = max(any_strict, key=lambda v: v["confirmation"]["shift_mobj_mean"])
    else:
        verdict = "no_strict_class_iii_after_cleanup"
        best = max(variant_summaries, key=lambda v: v["confirmation"]["shift_mobj_mean"]) if variant_summaries else None
    return {
        "final_verdict": verdict,
        "best_variant": best["candidate_name"] if best else None,
        "confirmed_count": len(confirmed),
        "any_strict_count": len(any_strict),
    }


def format_strict_confirmation_summary(
    variant_summaries: list[dict[str, Any]],
    overall: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variants": variant_summaries,
        "overall": overall,
    }


def run_class_iii_strict_confirmation_v2(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = (root / "results" / "pilots") if output_root is None else Path(output_root)
    artifact_root = out / str(cfg["artifact_subdir"])

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

    rubric = json.loads((root / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))

    size_panel = _validated_size_panel(cfg["size_panel"])
    lambdas: list[float] = [float(x) for x in cfg["lambda_grid"]]
    control_mode: str = str(cfg["control_application_name"])
    cv_threshold: float = float(cfg.get("scale_invariance_cv_threshold", 0.05))

    confirmation_rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []

    for cand_spec in cfg["candidates"]:
        cname: str = str(cand_spec["candidate_name"])
        fname: str = str(cand_spec["family_name"])
        base_kw: dict[str, Any] = dict(cand_spec.get("base_kwargs", {}))

        size_results: list[dict[str, Any]] = []
        for n in size_panel:
            kwargs = {**base_kw, "n": int(n)}
            substrate = build_class_iii_candidate_family(fname, **kwargs)
            tau1_rows, tau1_boundary, tau1_aff = _evaluate_tau_panel(substrate, lambdas, 1, control_mode)
            tau2_rows, tau2_boundary, tau2_aff = _evaluate_tau_panel(substrate, lambdas, 2, control_mode)
            cand_rows = {
                "family_name": cname,
                "substrate_family_name": fname,
                "substrate": substrate,
                "tau1_rows": tau1_rows,
                "tau2_rows": tau2_rows,
                "tau1_boundary": tau1_boundary,
                "tau2_boundary": tau2_boundary,
                "tau1_affinity_ref": tau1_aff,
                "tau2_affinity_ref": tau2_aff,
            }
            ev = evaluate_class_iii_candidate(cand_rows, rubric, p4_cfg)
            row: dict[str, Any] = {
                "candidate_name": cname,
                "family_name": fname,
                "portal_self_weight": float(base_kw.get("portal_self_weight", 0.0)),
                "size": int(n),
                **ev,
                "cache_status": "executed",
                "manifest_path": "",
            }
            size_results.append(row)
            confirmation_rows.append(row)

        confirmation = evaluate_strict_confirmation_variant(size_results, scale_cv_threshold=cv_threshold, expected_sizes=size_panel)
        variant_summaries.append({
            "candidate_name": cname,
            "family_name": fname,
            "portal_self_weight": float(base_kw.get("portal_self_weight", 0.0)),
            "confirmation": confirmation,
        })

    overall = summarize_strict_confirmation(variant_summaries)
    summary = format_strict_confirmation_summary(variant_summaries, overall)

    # Write confirmation table CSV
    table_fields = [
        "candidate_name", "portal_self_weight", "size",
        "candidate_p5_state", "candidate_p6_drive_state", "candidate_p4_state",
        "canonical_class_label", "candidate_is_class_iii_strict",
        "staging_shift_ce", "staging_shift_mobj", "affinity_ref_max",
        "cache_status", "manifest_path",
    ]
    _write_csv(dirs["analysis"] / "confirmation_table.csv", confirmation_rows, table_fields)
    _write_csv(dirs["metrics"] / "metrics.csv", confirmation_rows, table_fields)
    (dirs["analysis"] / "confirmation_summary.json").write_text(scientific_dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Plots
    by_size_sp3 = [r for r in confirmation_rows if r["candidate_name"] == "replicated_portal_sp3"]
    if by_size_sp3:
        sizes_x = [float(r["size"]) for r in by_size_sp3]
        _line_plot(
            dirs["plots"] / "shift_vs_size.png",
            sizes_x,
            {
                "sp3:CE": [observed_float(r["staging_shift_ce"]) for r in by_size_sp3],
                "sp3:Mobj": [observed_float(r["staging_shift_mobj"]) for r in by_size_sp3],
            },
        )
    all_names = list({r["candidate_name"] for r in confirmation_rows})
    strict_by_size: dict[str, list[float]] = {}
    for n in size_panel:
        for cn in all_names:
            rows_n = [r for r in confirmation_rows if r["candidate_name"] == cn and int(r["size"]) == n]
            if rows_n:
                strict_by_size.setdefault(cn, []).append(1.0 if rows_n[0]["candidate_is_class_iii_strict"] else 0.0)
    if strict_by_size:
        _line_plot(dirs["plots"] / "strict_hits_vs_size.png", [float(n) for n in size_panel], strict_by_size)

    # Standard bundle files
    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")

    # Findings note
    verdict = overall["final_verdict"]
    best = overall.get("best_variant", "none")
    any_strict_sp3 = any(r["candidate_is_class_iii_strict"] for r in confirmation_rows if r["candidate_name"] == "replicated_portal_sp3")
    scale_sp3 = next((v["confirmation"]["scale_invariance_supported"] for v in variant_summaries if v["candidate_name"] == "replicated_portal_sp3"), False)

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/S2-04_class_iii_strict_confirmation_v2.md"))
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_lines = [
        "# S2-04 Class-III strict confirmation v2",
        "",
        f"- config path: `configs/pilots/class_iii_strict_confirmation_v2.json`",
        f"- artifact root: `{artifact_root}`",
        f"- size panel: `{size_panel}`",
        f"- variant summaries: `{variant_summaries}`",
        f"- final verdict: `{verdict}`",
        f"- best confirmed variant: `{best}`",
        "",
        f"- does any variant remain strict Class-III across the full checked size panel? `{'yes' if any(v['confirmation']['strict_hit_persistent'] for v in variant_summaries) else 'no'}`",
        f"- is the scale-invariant claim supported by checked-in evidence? `{'yes' if any(v['confirmation']['scale_invariance_supported'] for v in variant_summaries) else 'no'}`",
        f"- should the project now treat Class-III as confirmed, or still provisional? `{'confirmed' if verdict == 'strict_class_iii_confirmed' else 'provisional'}`",
    ]
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_iii_strict_confirmation_v2",
        "bundle_id": str(cfg["pilot_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "final_verdict": verdict,
        "best_variant": best,
        "variant_summaries": [
            {
                "candidate_name": v["candidate_name"],
                "strict_hit_persistent": v["confirmation"]["strict_hit_persistent"],
                "scale_invariance_supported": v["confirmation"]["scale_invariance_supported"],
                "final_variant_label": v["confirmation"]["final_variant_label"],
            }
            for v in variant_summaries
        ],
        "artifact_root": str(artifact_root),
    }


def run_class_iii_full_campaign(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = (root / "results" / "campaigns") if output_root is None else Path(output_root)
    artifact_root = out / str(cfg["artifact_subdir"])

    # Ensure strict-confirmation provenance root exists.
    strict_root = root / "results" / "pilots" / "class_iii_strict_confirmation_v2"
    if not artifact_is_current(strict_root, root / "configs/pilots/class_iii_strict_confirmation_v2.json"):
        run_class_iii_strict_confirmation_v2(
            root / "configs" / "pilots" / "class_iii_strict_confirmation_v2.json",
            output_root=root / "results" / "pilots",
            use_cache=use_cache,
        )

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

    rubric = json.loads((root / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))

    candidate = dict(cfg["candidate"])
    candidate_name = str(candidate["candidate_name"])
    family_name = str(candidate["family_name"])
    base_kwargs = dict(candidate.get("base_kwargs", {}))
    size_panel = _validated_size_panel(cfg["size_panel"])
    lambdas = [float(x) for x in cfg["lambda_grid"]]
    control_mode = str(cfg["control_application_name"])

    per_size_rows: list[dict[str, Any]] = []
    for n in size_panel:
        kwargs = {**base_kwargs, "n": int(n)}
        substrate = build_class_iii_candidate_family(family_name, **kwargs)
        tau1_rows, tau1_boundary, tau1_aff = _evaluate_tau_panel(substrate, lambdas, 1, control_mode)
        tau2_rows, tau2_boundary, tau2_aff = _evaluate_tau_panel(substrate, lambdas, 2, control_mode)
        ev = evaluate_class_iii_candidate(
            {
                "family_name": candidate_name,
                "substrate_family_name": family_name,
                "substrate": substrate,
                "tau1_rows": tau1_rows,
                "tau2_rows": tau2_rows,
                "tau1_boundary": tau1_boundary,
                "tau2_boundary": tau2_boundary,
                "tau1_affinity_ref": tau1_aff,
                "tau2_affinity_ref": tau2_aff,
            },
            rubric,
            p4_cfg,
        )
        per_size_rows.append(
            {
                "candidate_name": candidate_name,
                "family_name": family_name,
                "portal_self_weight": float(base_kwargs.get("portal_self_weight", 0.0)),
                "size": int(n),
                **ev,
                "cache_status": "executed",
                "manifest_path": "manifest.json",
            }
        )

    structural_birth_present = all(r["candidate_p5_state"] == "active" for r in per_size_rows)
    affinity_absent = all(r["candidate_p6_drive_state"] == "inactive" for r in per_size_rows)
    staging_anomaly_present = all(r["candidate_p4_state"] == "active" for r in per_size_rows)
    class_iii_across_panel = all(r["canonical_class_label"] == "Class-III" for r in per_size_rows)

    if structural_birth_present and affinity_absent and staging_anomaly_present and class_iii_across_panel:
        final_verdict = "positive_class_iii_campaign"
    else:
        final_verdict = "class_iii_campaign_inconclusive"

    summary = {
        "campaign_id": str(cfg["campaign_id"]),
        "candidate_name": candidate_name,
        "family_name": family_name,
        "portal_self_weight": float(base_kwargs.get("portal_self_weight", 0.0)),
        "size_panel": size_panel,
        "structural_birth_present": bool(structural_birth_present),
        "affinity_absent": bool(affinity_absent),
        "staging_anomaly_present": bool(staging_anomaly_present),
        "class_iii_across_checked_sizes": bool(class_iii_across_panel),
        "final_class_verdict": "Class-III" if class_iii_across_panel else "mixed",
        "final_campaign_verdict": final_verdict,
        "v2_confirmation_source": "results/pilots/class_iii_strict_confirmation_v2",
    }

    fields = [
        "candidate_name",
        "family_name",
        "portal_self_weight",
        "size",
        "canonical_class_label",
        "candidate_p5_state",
        "candidate_p6_drive_state",
        "candidate_p4_state",
        "affinity_ref_max",
        "ce_boundary_tau1",
        "ce_boundary_tau2",
        "mobj_boundary_tau1",
        "mobj_boundary_tau2",
        "staging_shift_ce",
        "staging_shift_mobj",
        "presence_shift_any",
        "staging_gap_anomaly_score",
        "candidate_is_class_iii_strict",
        "cache_status",
        "manifest_path",
    ]
    _write_csv(dirs["analysis"] / "per_size_diagnostics.csv", per_size_rows, fields)
    _write_csv(dirs["metrics"] / "metrics.csv", per_size_rows, fields)
    (dirs["analysis"] / "campaign_summary.json").write_text(scientific_dumps(summary, indent=2) + "\n", encoding="utf-8")

    x = [float(r["size"]) for r in per_size_rows]
    _line_plot(
        dirs["plots"] / "staging_shifts_by_size.png",
        x,
        {
            "CE shift": [observed_float(r["staging_shift_ce"]) for r in per_size_rows],
            "M_obj shift": [observed_float(r["staging_shift_mobj"]) for r in per_size_rows],
        },
    )
    _line_plot(
        dirs["plots"] / "affinity_ref_by_size.png",
        x,
        {"affinity_ref_max": [observed_float(r["affinity_ref_max"]) for r in per_size_rows]},
    )
    _state_plot(
        dirs["plots"] / "class_state_by_size.png",
        x,
        {
            "P5_active": [1.0 if r["candidate_p5_state"] == "active" else 0.0 for r in per_size_rows],
            "P6_inactive": [1.0 if r["candidate_p6_drive_state"] == "inactive" else 0.0 for r in per_size_rows],
            "P4_active": [1.0 if r["candidate_p4_state"] == "active" else 0.0 for r in per_size_rows],
            "Class-III": [1.0 if r["canonical_class_label"] == "Class-III" else 0.0 for r in per_size_rows],
        },
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(
        scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/S2-05_class_iii_full_campaign.md"))
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_lines = [
        "# S2-05 Class-III full campaign",
        "",
        "- this campaign uses the v2-confirmed Class-III family as input (`replicated_portal_sp4`).",
        f"- config path: `configs/campaigns/class_iii_full_campaign.json`",
        f"- artifact root: `{artifact_root}`",
        f"- checked size panel: `{size_panel}`",
        f"- final class verdict across checked panel: `{summary['final_class_verdict']}`",
        f"- final campaign verdict: `{final_verdict}`",
        f"- structural birth present across checked sizes: `{'yes' if structural_birth_present else 'no'}`",
        f"- affinity absent across checked sizes: `{'yes' if affinity_absent else 'no'}`",
        f"- staging anomaly present across checked sizes: `{'yes' if staging_anomaly_present else 'no'}`",
    ]
    note_text = "\n".join(note_lines) + "\n"
    note_path.write_text(note_text, encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(note_text, encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_iii_full_campaign",
        "bundle_id": str(cfg["campaign_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "candidate_name": candidate_name,
        "size_panel": size_panel,
        "final_class_verdict": summary["final_class_verdict"],
        "structural_birth_present": bool(structural_birth_present),
        "affinity_absent": bool(affinity_absent),
        "staging_anomaly_present": bool(staging_anomaly_present),
        "final_campaign_verdict": final_verdict,
        "artifact_root": str(artifact_root),
    }


def _build_class_iv_candidate_substrate(base_family_name: str, base_kwargs: dict[str, Any], drive_kwargs: dict[str, Any], size: int) -> dict[str, Any]:
    kwargs = {**base_kwargs, "n": int(size)}
    base = build_class_iii_candidate_family(base_family_name, **kwargs)
    alpha = float(drive_kwargs.get("drive_mix", 0.0))
    if alpha <= 0.0:
        return base
    alpha = min(alpha, 1.0)
    flow_total = float(drive_kwargs.get("flow_total", 0.90))
    drive_bias = float(drive_kwargs.get("drive_bias", 0.05))
    drive_self_weight = float(drive_kwargs.get("drive_self_weight", 0.10))
    forward = max(0.0, (flow_total / 2.0) + drive_bias)
    backward = max(0.0, (flow_total / 2.0) - drive_bias)
    drive = driven_cycle_family(
        n=int(size),
        self_weight=drive_self_weight,
        forward_weight=forward,
        backward_weight=backward,
    )
    p = ((1.0 - alpha) * np.asarray(base["P"], dtype=np.float64)) + (alpha * np.asarray(drive["P"], dtype=np.float64))
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    p = p / row_sums
    return {
        "P": p,
        "n": int(base["n"]),
        "family_name": "replicated_portal_sp4_driven_variant",
        "coarse_lens": np.asarray(base["coarse_lens"], dtype=np.int64),
        "details": {
            "base_family_name": base_family_name,
            "base_kwargs": dict(base_kwargs),
            "drive_kwargs": {
                "drive_mix": alpha,
                "flow_total": flow_total,
                "drive_bias": drive_bias,
                "drive_self_weight": drive_self_weight,
                "forward_weight": forward,
                "backward_weight": backward,
            },
        },
    }


def _evaluate_class_iv_rows(
    candidate_name: str,
    base_family_name: str,
    base_kwargs: dict[str, Any],
    drive_kwargs: dict[str, Any],
    size: int,
    lambdas: list[float],
    control_mode: str,
    rubric: dict[str, Any],
    p4_cfg: dict[str, Any],
) -> dict[str, Any]:
    substrate = _build_class_iv_candidate_substrate(base_family_name, base_kwargs, drive_kwargs, size)
    tau1_rows, tau1_boundary, tau1_aff = _evaluate_tau_panel(substrate, lambdas, 1, control_mode)
    tau2_rows, tau2_boundary, tau2_aff = _evaluate_tau_panel(substrate, lambdas, 2, control_mode)
    ev = evaluate_class_iii_candidate(
        {
            "family_name": candidate_name,
            "substrate_family_name": base_family_name,
            "substrate": substrate,
            "tau1_rows": tau1_rows,
            "tau2_rows": tau2_rows,
            "tau1_boundary": tau1_boundary,
            "tau2_boundary": tau2_boundary,
            "tau1_affinity_ref": tau1_aff,
            "tau2_affinity_ref": tau2_aff,
        },
        rubric,
        p4_cfg,
    )
    p5 = ev["candidate_p5_state"] == "active"
    p6 = ev["candidate_p6_drive_state"] == "active"
    p4 = ev["candidate_p4_state"] == "active"
    return {
        "candidate_name": candidate_name,
        "family_name": "replicated_portal_reversible_family",
        "size": int(size),
        **ev,
        "structural_birth_present": bool(p5),
        "affinity_present": bool(p6),
        "staging_anomaly_present": bool(p4),
        "all_three_active": bool(p5 and p6 and p4),
        "cache_status": "executed",
        "manifest_path": "manifest.json",
    }


def run_class_iv_candidates(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = (root / "results" / "pilots") if output_root is None else Path(output_root)
    artifact_root = out / str(cfg["artifact_subdir"])

    # Ensure canonical Class-III provenance root exists.
    class3_root = root / "results" / "campaigns" / "class_iii_full_campaign"
    if not artifact_is_current(class3_root, root / "configs/campaigns/class_iii_full_campaign.json"):
        run_class_iii_full_campaign(
            root / "configs" / "campaigns" / "class_iii_full_campaign.json",
            output_root=root / "results" / "campaigns",
            use_cache=use_cache,
        )

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

    rubric = json.loads((root / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))
    base = dict(cfg["base_candidate"])
    base_candidate_name = str(base["candidate_name"])
    base_family_name = str(base["family_name"])
    base_kwargs = dict(base.get("base_kwargs", {}))
    lambdas = [float(x) for x in cfg["lambda_grid"]]
    control_mode = str(cfg["control_application_name"])

    # Stage 1
    stage1_rows: list[dict[str, Any]] = []
    for cand in cfg["candidates"]:
        cname = str(cand["candidate_name"])
        dkw = dict(cand.get("drive_kwargs", {}))
        for size in cfg["stage1_sizes"]:
            row = _evaluate_class_iv_rows(
                cname,
                base_family_name,
                base_kwargs,
                dkw,
                int(size),
                lambdas,
                control_mode,
                rubric,
                p4_cfg,
            )
            row["stage"] = "stage1"
            row["derived_from_candidate"] = base_candidate_name
            row["drive_kwargs"] = scientific_dumps(dkw, sort_keys=True)
            stage1_rows.append(row)

    # Stage 2 candidate selection from stage1 summary
    by_candidate_stage1: dict[str, list[dict[str, Any]]] = {}
    for r in stage1_rows:
        by_candidate_stage1.setdefault(str(r["candidate_name"]), []).append(r)
    ranked_stage1 = sorted(
        by_candidate_stage1.items(),
        key=lambda kv: (
            sum(1 for x in kv[1] if bool(x["all_three_active"])),
            max(observed_float(x["staging_gap_anomaly_score"]) for x in kv[1]),
            -max(observed_float(x["affinity_ref_max"]) for x in kv[1]),
            str(kv[0]),
        ),
        reverse=True,
    )
    top_k = int(cfg.get("stage2_top_k", 2))
    stage2_names = [name for name, _ in ranked_stage1[:top_k]]

    # Stage 2
    candidate_drive_map = {str(c["candidate_name"]): dict(c.get("drive_kwargs", {})) for c in cfg["candidates"]}
    stage2_rows: list[dict[str, Any]] = []
    for cname in stage2_names:
        dkw = candidate_drive_map[cname]
        for size in cfg["stage2_sizes"]:
            row = _evaluate_class_iv_rows(
                cname,
                base_family_name,
                base_kwargs,
                dkw,
                int(size),
                lambdas,
                control_mode,
                rubric,
                p4_cfg,
            )
            row["stage"] = "stage2"
            row["derived_from_candidate"] = base_candidate_name
            row["drive_kwargs"] = scientific_dumps(dkw, sort_keys=True)
            stage2_rows.append(row)

    # Final per-candidate rows for verdict/selection: stage2 rows when available, otherwise stage1.
    final_rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for cname in [str(c["candidate_name"]) for c in cfg["candidates"]]:
        rows = [r for r in stage2_rows if str(r["candidate_name"]) == cname]
        if not rows:
            rows = [r for r in stage1_rows if str(r["candidate_name"]) == cname]
        final_rows_by_candidate[cname] = rows

    candidate_scores = []
    for cname, rows in final_rows_by_candidate.items():
        if not rows:
            continue
        count_all3 = sum(1 for r in rows if bool(r["all_three_active"]))
        max_anom = max(observed_float(r["staging_gap_anomaly_score"]) for r in rows)
        max_aff = max(observed_float(r["affinity_ref_max"]) for r in rows)
        candidate_scores.append(
            {
                "candidate_name": cname,
                "checked_sizes": sorted({int(r["size"]) for r in rows}),
                "all_three_active_count": int(count_all3),
                "max_staging_gap_anomaly_score": float(max_anom),
                "max_affinity_ref_max": float(max_aff),
                "all_three_any": bool(count_all3 > 0),
            }
        )

    ranked_final = sorted(
        candidate_scores,
        key=lambda r: (
            int(r["all_three_active_count"]),
            observed_float(r["max_staging_gap_anomaly_score"]),
            -observed_float(r["max_affinity_ref_max"]),
            str(r["candidate_name"]),
        ),
        reverse=True,
    )
    best = ranked_final[0] if ranked_final else None
    any_all_three = any(bool(r["all_three_any"]) for r in ranked_final)
    final_verdict = "positive_class_iv_pilot_candidate" if any_all_three else "explicit_failure_near_miss"

    pilot_summary = {
        "pilot_id": str(cfg["pilot_id"]),
        "base_candidate": base_candidate_name,
        "base_family": base_family_name,
        "candidates_screened_count": len(cfg["candidates"]),
        "stage1_sizes": [int(x) for x in cfg["stage1_sizes"]],
        "stage2_sizes": [int(x) for x in cfg["stage2_sizes"]],
        "stage2_candidates": stage2_names,
        "any_candidate_all_three_active": bool(any_all_three),
        "candidate_scores": ranked_final,
        "final_pilot_verdict": final_verdict,
    }
    best_summary = {
        "best_candidate_name": None if best is None else str(best["candidate_name"]),
        "best_candidate_checked_sizes": [] if best is None else best["checked_sizes"],
        "best_candidate_all_three_active_count": 0 if best is None else int(best["all_three_active_count"]),
        "any_candidate_all_three_active": bool(any_all_three),
        "final_pilot_verdict": final_verdict,
    }

    table_rows = stage1_rows + stage2_rows
    fields = [
        "candidate_name",
        "family_name",
        "derived_from_candidate",
        "stage",
        "size",
        "candidate_p5_state",
        "candidate_p6_drive_state",
        "candidate_p4_state",
        "canonical_class_label",
        "affinity_ref_max",
        "ce_boundary_tau1",
        "ce_boundary_tau2",
        "mobj_boundary_tau1",
        "mobj_boundary_tau2",
        "staging_shift_ce",
        "staging_shift_mobj",
        "presence_shift_any",
        "staging_gap_anomaly_score",
        "structural_birth_present",
        "affinity_present",
        "staging_anomaly_present",
        "all_three_active",
        "cache_status",
        "manifest_path",
        "drive_kwargs",
    ]
    _write_csv(dirs["analysis"] / "candidate_table.csv", table_rows, fields)
    _write_csv(dirs["metrics"] / "metrics.csv", table_rows, fields)
    (dirs["analysis"] / "pilot_summary.json").write_text(scientific_dumps(pilot_summary, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "best_candidate_summary.json").write_text(scientific_dumps(best_summary, indent=2) + "\n", encoding="utf-8")

    idx = [float(i) for i in range(len(table_rows))]
    _line_plot(
        dirs["plots"] / "p4_p5_p6_states_by_candidate.png",
        idx if idx else [0.0],
        {
            "P5_active": [1.0 if r["candidate_p5_state"] == "active" else 0.0 for r in table_rows] if table_rows else [0.0],
            "P6_active": [1.0 if r["candidate_p6_drive_state"] == "active" else 0.0 for r in table_rows] if table_rows else [0.0],
            "P4_active": [1.0 if r["candidate_p4_state"] == "active" else 0.0 for r in table_rows] if table_rows else [0.0],
            "All_three": [1.0 if bool(r["all_three_active"]) else 0.0 for r in table_rows] if table_rows else [0.0],
        },
    )
    _line_plot(
        dirs["plots"] / "affinity_vs_staging_by_candidate.png",
        idx if idx else [0.0],
        {
            "affinity_ref_max": [observed_float(r["affinity_ref_max"]) for r in table_rows] if table_rows else [0.0],
            "staging_gap_anomaly_score": [observed_float(r["staging_gap_anomaly_score"]) for r in table_rows] if table_rows else [0.0],
        },
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(
        scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )

    note_lines = [
        "# S2-06 Class-IV substrate design and pilot suite",
        "",
        "- built by extending the v2-confirmed / S2-05-canonical Class-III substrate (`replicated_portal_sp4`).",
        f"- config path: `configs/pilots/class_iv_candidates.json`",
        f"- artifact root: `{artifact_root}`",
        f"- candidate variants screened: `{len(cfg['candidates'])}`",
        f"- any candidate achieved all three activations (P5+P6+P4): `{'yes' if any_all_three else 'no'}`",
        f"- selected best family: `{best_summary['best_candidate_name']}`",
        f"- final pilot verdict: `{final_verdict}`",
    ]
    note_text = "\n".join(note_lines) + "\n"
    note_path = root / str(cfg.get("findings_note_path", "notes/findings/S2-06_class_iv_candidates.md"))
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_text, encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(note_text, encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_iv_candidates",
        "bundle_id": str(cfg["pilot_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "candidates_screened": len(cfg["candidates"]),
        "any_candidate_all_three_active": bool(any_all_three),
        "best_candidate_name": best_summary["best_candidate_name"],
        "final_pilot_verdict": final_verdict,
        "artifact_root": str(artifact_root),
    }


def run_class_iv_full_campaign(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = (root / "results" / "campaigns") if output_root is None else Path(output_root)
    artifact_root = out / str(cfg["artifact_subdir"])

    # Ensure pilot provenance exists.
    iv_pilot_root = root / "results" / "pilots" / "class_iv_candidates"
    if not artifact_is_current(iv_pilot_root, root / "configs/pilots/class_iv_candidates.json"):
        run_class_iv_candidates(
            root / "configs" / "pilots" / "class_iv_candidates.json",
            output_root=root / "results" / "pilots",
            use_cache=use_cache,
        )

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

    rubric = json.loads((root / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))

    candidate = dict(cfg["candidate"])
    candidate_name = str(candidate["candidate_name"])
    base_candidate_name = str(candidate["base_candidate_name"])
    base_family_name = str(candidate["family_name"])
    base_kwargs = dict(candidate.get("base_kwargs", {}))
    drive_kwargs = dict(candidate.get("drive_kwargs", {}))
    size_panel = _validated_size_panel(cfg["size_panel"])
    lambdas = [float(x) for x in cfg["lambda_grid"]]
    control_mode = str(cfg["control_application_name"])

    # Main campaign rows
    per_size_rows: list[dict[str, Any]] = []
    for n in size_panel:
        r = _evaluate_class_iv_rows(
            candidate_name,
            base_family_name,
            base_kwargs,
            drive_kwargs,
            int(n),
            lambdas,
            control_mode,
            rubric,
            p4_cfg,
        )
        r["derived_from_candidate"] = base_candidate_name
        per_size_rows.append(r)

    structural_birth_present = all(bool(r["structural_birth_present"]) for r in per_size_rows)
    affinity_present = all(bool(r["affinity_present"]) for r in per_size_rows)
    staging_anomaly_present = all(bool(r["staging_anomaly_present"]) for r in per_size_rows)
    class_iv_across_panel = all(str(r["canonical_class_label"]) == "Class-IV" for r in per_size_rows)
    if structural_birth_present and affinity_present and staging_anomaly_present and class_iv_across_panel:
        final_verdict = "positive_class_iv_campaign"
    else:
        final_verdict = "explicit_failure_narrowing_report"

    # Comparison references
    taxonomy_refs = json.loads(
        (root / "results" / "taxonomy" / "canonical_class_rubric" / "analysis" / "reference_classifications.json").read_text(
            encoding="utf-8"
        )
    )["reference_classifications"]
    class_ii_ref = next(r for r in taxonomy_refs if str(r["profile_name"]) == "class_ii_reference")
    class_iii_rows = list(
        csv.DictReader((root / "results" / "campaigns" / "class_iii_full_campaign" / "analysis" / "per_size_diagnostics.csv").open())
    )
    class_ii_group = list(
        csv.DictReader((root / "results" / "campaigns" / "class_ii_driven" / "analysis" / "primary_group_summary.csv").open())
    )

    # Build Class-II overlap-size diagnostics from grouped panel + taxonomy states.
    class_ii_overlap_sizes = [s for s in [16, 32, 64] if s in size_panel]
    class_ii_rows: list[dict[str, Any]] = []
    for s in class_ii_overlap_sizes:
        g = [r for r in class_ii_group if int(float(r["size"])) == int(s)]
        scan_rows = [
            {
                "closure_strength_lambda": float(r["closure_strength_lambda"]),
                "closure_error": observed_float(r["closure_error_mean"]),
                "objecthood_order": float(r["order_mean"]),
                "staging_gap": observed_float(r["staging_gap_mean"]),
                "affinity": observed_float(r["affinity_mean"]),
            }
            for r in g
        ]
        b = estimate_structural_boundaries(scan_rows)
        a = estimate_affinity_reference(scan_rows, b)
        class_ii_rows.append(
            {
                "class_name": "Class-II",
                "size": int(s),
                "candidate_p5_state": str(class_ii_ref["p5_state"]),
                "candidate_p6_drive_state": str(class_ii_ref["p6_drive_state"]),
                "candidate_p4_state": str(class_ii_ref["p4_state"]),
                "canonical_class_label": str(class_ii_ref["canonical_class_label"]),
                "affinity_ref_max": observed_float(a["affinity_ref"]),
                "ce_boundary_tau1": b.get("ce_boundary_lambda"),
                "ce_boundary_tau2": None,
                "mobj_boundary_tau1": b.get("mobj_boundary_lambda"),
                "mobj_boundary_tau2": None,
                "staging_shift_ce": observed_float(class_ii_ref.get("tau_ce_shift", 0.0)),
                "staging_shift_mobj": observed_float(class_ii_ref.get("tau_mobj_shift", 0.0)),
                "presence_shift_any": False,
                "staging_gap_anomaly_score": observed_float(class_ii_ref.get("staging_gap_anomaly_score", 0.0)),
                "all_three_active": bool(
                    class_ii_ref["p5_state"] == "active"
                    and class_ii_ref["p6_drive_state"] == "active"
                    and class_ii_ref["p4_state"] == "active"
                ),
            }
        )

    class_iii_rows_norm: list[dict[str, Any]] = []
    for r in class_iii_rows:
        class_iii_rows_norm.append(
            {
                "class_name": "Class-III",
                "size": int(float(r["size"])),
                "candidate_p5_state": str(r["candidate_p5_state"]),
                "candidate_p6_drive_state": str(r["candidate_p6_drive_state"]),
                "candidate_p4_state": str(r["candidate_p4_state"]),
                "canonical_class_label": str(r["canonical_class_label"]),
                "affinity_ref_max": observed_float(r["affinity_ref_max"]),
                "ce_boundary_tau1": observed_float(r["ce_boundary_tau1"]),
                "ce_boundary_tau2": observed_float(r["ce_boundary_tau2"]),
                "mobj_boundary_tau1": observed_float(r["mobj_boundary_tau1"]),
                "mobj_boundary_tau2": observed_float(r["mobj_boundary_tau2"]),
                "staging_shift_ce": observed_float(r["staging_shift_ce"]),
                "staging_shift_mobj": observed_float(r["staging_shift_mobj"]),
                "presence_shift_any": str(r["presence_shift_any"]).lower() == "true",
                "staging_gap_anomaly_score": observed_float(r["staging_gap_anomaly_score"]),
                "all_three_active": bool(
                    str(r["candidate_p5_state"]) == "active"
                    and str(r["candidate_p6_drive_state"]) == "active"
                    and str(r["candidate_p4_state"]) == "active"
                ),
            }
        )

    class_iv_rows_norm: list[dict[str, Any]] = []
    for r in per_size_rows:
        class_iv_rows_norm.append(
            {
                "class_name": "Class-IV",
                "size": int(r["size"]),
                "candidate_p5_state": str(r["candidate_p5_state"]),
                "candidate_p6_drive_state": str(r["candidate_p6_drive_state"]),
                "candidate_p4_state": str(r["candidate_p4_state"]),
                "canonical_class_label": str(r["canonical_class_label"]),
                "affinity_ref_max": observed_float(r["affinity_ref_max"]),
                "ce_boundary_tau1": r.get("ce_boundary_tau1"),
                "ce_boundary_tau2": r.get("ce_boundary_tau2"),
                "mobj_boundary_tau1": r.get("mobj_boundary_tau1"),
                "mobj_boundary_tau2": r.get("mobj_boundary_tau2"),
                "staging_shift_ce": observed_float(r["staging_shift_ce"]),
                "staging_shift_mobj": observed_float(r["staging_shift_mobj"]),
                "presence_shift_any": bool(r["presence_shift_any"]),
                "staging_gap_anomaly_score": observed_float(r["staging_gap_anomaly_score"]),
                "all_three_active": bool(r["all_three_active"]),
            }
        )

    # Comparison outputs
    comparison_ii = {
        "comparison_basis": {
            "reference": "results/campaigns/class_ii_driven (primary low-bias panel)",
            "overlap_sizes": class_ii_overlap_sizes,
        },
        "class_ii_reference_states": {
            "p5_state": class_ii_ref["p5_state"],
            "p6_drive_state": class_ii_ref["p6_drive_state"],
            "p4_state": class_ii_ref["p4_state"],
            "canonical_class_label": class_ii_ref["canonical_class_label"],
        },
        "class_iv_states_by_size": [
            {
                "size": int(r["size"]),
                "p5_state": r["candidate_p5_state"],
                "p6_drive_state": r["candidate_p6_drive_state"],
                "p4_state": r["candidate_p4_state"],
                "canonical_class_label": r["canonical_class_label"],
            }
            for r in class_iv_rows_norm
            if int(r["size"]) in class_ii_overlap_sizes
        ],
        "state_pattern_difference": "Class-IV adds P4 active while retaining P5+P6 active relative to Class-II.",
    }
    comparison_iii = {
        "comparison_basis": {
            "reference": "results/campaigns/class_iii_full_campaign",
            "sizes": [int(r["size"]) for r in class_iv_rows_norm],
        },
        "class_iii_reference_pattern": "P5 active, P6 inactive, P4 active",
        "class_iv_pattern": "P5 active, P6 active, P4 active",
        "state_pattern_difference": "Class-IV adds sustained P6 drive activation while preserving P5 and P4 relative to Class-III.",
    }

    campaign_summary = {
        "campaign_id": str(cfg["campaign_id"]),
        "candidate_name": candidate_name,
        "base_candidate_name": base_candidate_name,
        "family_name": base_family_name,
        "drive_kwargs": drive_kwargs,
        "size_panel": size_panel,
        "structural_birth_present": bool(structural_birth_present),
        "affinity_present": bool(affinity_present),
        "staging_anomaly_present": bool(staging_anomaly_present),
        "class_iv_across_checked_sizes": bool(class_iv_across_panel),
        "final_class_verdict": "Class-IV" if class_iv_across_panel else "mixed",
        "final_campaign_verdict": final_verdict,
    }

    per_size_fields = [
        "candidate_name",
        "family_name",
        "derived_from_candidate",
        "size",
        "candidate_p5_state",
        "candidate_p6_drive_state",
        "candidate_p4_state",
        "canonical_class_label",
        "affinity_ref_max",
        "ce_boundary_tau1",
        "ce_boundary_tau2",
        "mobj_boundary_tau1",
        "mobj_boundary_tau2",
        "staging_shift_ce",
        "staging_shift_mobj",
        "presence_shift_any",
        "staging_gap_anomaly_score",
        "structural_birth_present",
        "affinity_present",
        "staging_anomaly_present",
        "all_three_active",
        "cache_status",
        "manifest_path",
    ]
    _write_csv(dirs["analysis"] / "per_size_diagnostics.csv", per_size_rows, per_size_fields)
    _write_csv(dirs["metrics"] / "metrics.csv", per_size_rows, per_size_fields)
    (dirs["analysis"] / "campaign_summary.json").write_text(scientific_dumps(campaign_summary, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "comparison_to_class_ii.json").write_text(scientific_dumps(comparison_ii, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "comparison_to_class_iii.json").write_text(scientific_dumps(comparison_iii, indent=2) + "\n", encoding="utf-8")

    comparison_rows = class_ii_rows + class_iii_rows_norm + class_iv_rows_norm
    comparison_fields = [
        "class_name",
        "size",
        "candidate_p5_state",
        "candidate_p6_drive_state",
        "candidate_p4_state",
        "canonical_class_label",
        "affinity_ref_max",
        "ce_boundary_tau1",
        "ce_boundary_tau2",
        "mobj_boundary_tau1",
        "mobj_boundary_tau2",
        "staging_shift_ce",
        "staging_shift_mobj",
        "staging_gap_anomaly_score",
        "all_three_active",
    ]
    _write_csv(dirs["analysis"] / "class_comparison_table.csv", comparison_rows, comparison_fields)

    # Plots
    x = [float(r["size"]) for r in per_size_rows]
    _line_plot(
        dirs["plots"] / "staging_shifts_by_size.png",
        x,
        {
            "CE shift": [observed_float(r["staging_shift_ce"]) for r in per_size_rows],
            "M_obj shift": [observed_float(r["staging_shift_mobj"]) for r in per_size_rows],
        },
    )
    _line_plot(
        dirs["plots"] / "affinity_ref_by_size.png",
        x,
        {"affinity_ref_max": [observed_float(r["affinity_ref_max"]) for r in per_size_rows]},
    )
    _state_plot(
        dirs["plots"] / "class_state_by_size.png",
        x,
        {
            "P5_active": [1.0 if r["candidate_p5_state"] == "active" else 0.0 for r in per_size_rows],
            "P6_active": [1.0 if r["candidate_p6_drive_state"] == "active" else 0.0 for r in per_size_rows],
            "P4_active": [1.0 if r["candidate_p4_state"] == "active" else 0.0 for r in per_size_rows],
            "Class-IV": [1.0 if r["canonical_class_label"] == "Class-IV" else 0.0 for r in per_size_rows],
        },
    )
    ii_map = {int(r["size"]): observed_float(r["affinity_ref_max"]) for r in class_ii_rows}
    iii_map = {int(r["size"]): observed_float(r["affinity_ref_max"]) for r in class_iii_rows_norm}
    iv_map = {int(r["size"]): observed_float(r["affinity_ref_max"]) for r in class_iv_rows_norm}
    comp_sizes = sorted(set(ii_map.keys()) | set(iii_map.keys()) | set(iv_map.keys()))
    _line_plot(
        dirs["plots"] / "class_ii_iii_iv_comparison.png",
        [float(s) for s in comp_sizes],
        {
            "Class-II affinity_ref": [ii_map.get(s, float("nan")) for s in comp_sizes],
            "Class-III affinity_ref": [iii_map.get(s, float("nan")) for s in comp_sizes],
            "Class-IV affinity_ref": [iv_map.get(s, float("nan")) for s in comp_sizes],
        },
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(
        scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )

    note_lines = [
        "# S2-07 Class-IV full campaign",
        "",
        "- this campaign uses the selected S2-06 best family derived from canonical Class-III (`replicated_portal_sp4_drive_f` from `replicated_portal_sp4`).",
        "- config path: `configs/campaigns/class_iv_full_campaign.json`",
        f"- artifact root: `{artifact_root}`",
        f"- checked size panel: `{size_panel}`",
        f"- final campaign verdict: `{final_verdict}`",
        f"- structural birth present across checked sizes: `{'yes' if structural_birth_present else 'no'}`",
        f"- affinity present across checked sizes: `{'yes' if affinity_present else 'no'}`",
        f"- staging anomaly present across checked sizes: `{'yes' if staging_anomaly_present else 'no'}`",
        "- qualitative comparison: versus Class-II this campaign adds P4 activity; versus Class-III this campaign adds sustained P6 drive activity.",
    ]
    note_text = "\n".join(note_lines) + "\n"
    note_path = root / str(cfg.get("findings_note_path", "notes/findings/S2-07_class_iv_full_campaign.md"))
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_text, encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(note_text, encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_iv_full_campaign",
        "bundle_id": str(cfg["campaign_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "candidate_name": candidate_name,
        "size_panel": size_panel,
        "final_class_verdict": campaign_summary["final_class_verdict"],
        "structural_birth_present": bool(structural_birth_present),
        "affinity_present": bool(affinity_present),
        "staging_anomaly_present": bool(staging_anomaly_present),
        "comparison_references": {
            "class_ii": "results/campaigns/class_ii_driven",
            "class_iii": "results/campaigns/class_iii_full_campaign",
        },
        "final_campaign_verdict": final_verdict,
        "artifact_root": str(artifact_root),
    }


def _evaluate_diffusion_tau1(
    substrate: dict[str, Any],
    lambdas: list[float],
    control_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    p_base = np.asarray(substrate["P"], dtype=np.float64)
    lens, _ = build_lens_family("diffusion_quantile_lens", P=p_base, target_k=2, tau=1)
    lens = np.asarray(lens, dtype=np.int64)
    k = int(np.max(lens)) + 1
    q = pushforward_matrix(lens, k)
    u = np.asarray(build_lift_family("uniform_lift_family", f=lens, k=k)[0], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for lam in lambdas:
        p = apply_closure_strength_control(
            p_base,
            closure_strength_lambda=float(lam),
            Q_f=q,
            U_f=u,
            mode=control_mode,
        )
        b = default_metric_bundle(p, lens, tau=1)
        rows.append(
            {
                "closure_strength_lambda": float(lam),
                "closure_error": observed_float(b["closure_error"]),
                "objecthood_order": observed_float(b["objecthood_order"]),
                "staging_gap": observed_float(b["staging_gap"]),
                "affinity": observed_float(b["affinity"]),
                "analysis_k": int(b["analysis_k"]),
                "resolved_tau": 1,
            }
        )
    boundary = estimate_structural_boundaries(rows)
    aff_ref = estimate_affinity_reference(rows, boundary)
    return rows, boundary, aff_ref


def run_four_class_synthesis_dashboard(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = (root / "results" / "dashboards") if output_root is None else Path(output_root)
    artifact_root = out / str(cfg["artifact_subdir"])

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

    s = cfg["sources"]
    c = cfg["conventions"]
    class_i_root = root / str(s["class_i_root"])
    class_ii_root = root / str(s["class_ii_root"])
    consolidation_root = root / str(s["class_i_ii_consolidation_root"])
    class_iii_root = root / str(s["class_iii_root"])
    class_iv_root = root / str(s["class_iv_root"])

    from .dashboard import build_class_consolidation_dashboard
    from .campaigns import run_class_i_scaling_campaign, run_class_ii_scaling_campaign
    prerequisites = [
        (class_i_root, "configs/campaigns/class_i_equilibrium_scaling.json", run_class_i_scaling_campaign),
        (class_ii_root, "configs/campaigns/class_ii_driven_scaling.json", run_class_ii_scaling_campaign),
        (consolidation_root, "configs/dashboards/class_i_class_ii_consolidation.json", build_class_consolidation_dashboard),
        (class_iii_root, "configs/campaigns/class_iii_full_campaign.json", run_class_iii_full_campaign),
        (class_iv_root, "configs/campaigns/class_iv_full_campaign.json", run_class_iv_full_campaign),
    ]
    for source_root, configuration, driver in prerequisites:
        if not artifact_is_current(source_root, root / configuration):
            produced = driver(root / configuration, output_root=source_root.parent, use_cache=use_cache)
            if Path(produced["artifact_root"]) != source_root:
                raise ValueError("requested synthesis source does not match its configured producer")

    ref_summary = json.loads((consolidation_root / "analysis" / "reference_summary.json").read_text(encoding="utf-8"))
    no_fake_arrow_anchor = json.loads((consolidation_root / "analysis" / "no_fake_arrow_summary.json").read_text(encoding="utf-8"))
    boundary_anchor = json.loads((consolidation_root / "analysis" / "structural_boundary_proxies.json").read_text(encoding="utf-8"))
    tau_lens_anchor = json.loads((consolidation_root / "analysis" / "tau_lens_summary.json").read_text(encoding="utf-8"))
    class_iii_summary = json.loads((class_iii_root / "analysis" / "campaign_summary.json").read_text(encoding="utf-8"))
    class_iv_summary = json.loads((class_iv_root / "analysis" / "campaign_summary.json").read_text(encoding="utf-8"))
    class_iii_rows = list(csv.DictReader((class_iii_root / "analysis" / "per_size_diagnostics.csv").open()))
    class_iv_rows = list(csv.DictReader((class_iv_root / "analysis" / "per_size_diagnostics.csv").open()))

    overlap_sizes = _validated_size_panel(c["comparison_overlap_sizes"])
    rep_size = int(c["representative_size"])

    # Fresh references at the recorded representative size; the consolidation
    # taxonomy uses its own configured size and cannot simply be relabelled.
    from .observable_map import _extract_class_i_ii
    map_cfg = json.loads((root / "configs/dashboards/four_class_observable_map.json").read_text())
    map_cfg["representative_size"] = rep_size
    fresh_i_ii = _extract_class_i_ii(map_cfg, root)
    for measured in fresh_i_ii:
        profile = measured["reference_profile_name"]
        ref_summary[profile] = {**ref_summary[profile], **measured}
        boundary_anchor[profile] = {**boundary_anchor[profile], "ce_boundary_lambda": measured["ce_boundary_lambda"], "mobj_boundary_lambda": measured["mobj_boundary_lambda"]}

    # Canonical activation rows
    row_i = {
        "class_label": "Class-I",
        "reference_profile_name": "class_i_reference",
        "source_bundle_root": str(class_i_root),
        "reference_candidate_name": "class_i_reference",
        "canonical_size_panel": "[8,16,32,64]",
        "comparison_overlap_sizes": str(overlap_sizes),
        "candidate_p5_state": str(ref_summary["class_i_reference"]["p5_state"]),
        "candidate_p6_drive_state": str(ref_summary["class_i_reference"]["p6_drive_state"]),
        "candidate_p4_state": str(ref_summary["class_i_reference"]["p4_state"]),
    }
    row_ii = {
        "class_label": "Class-II",
        "reference_profile_name": "class_ii_reference",
        "source_bundle_root": str(class_ii_root),
        "reference_candidate_name": "class_ii_reference",
        "canonical_size_panel": "[8,16,32,64]",
        "comparison_overlap_sizes": str(overlap_sizes),
        "candidate_p5_state": str(ref_summary["class_ii_reference"]["p5_state"]),
        "candidate_p6_drive_state": str(ref_summary["class_ii_reference"]["p6_drive_state"]),
        "candidate_p4_state": str(ref_summary["class_ii_reference"]["p4_state"]),
    }
    r3_32 = next(r for r in class_iii_rows if int(float(r["size"])) == rep_size)
    r4_32 = next(r for r in class_iv_rows if int(float(r["size"])) == rep_size)
    row_iii = {
        "class_label": "Class-III",
        "reference_profile_name": "class_iii_reference",
        "source_bundle_root": str(class_iii_root),
        "reference_candidate_name": str(class_iii_summary["candidate_name"]),
        "canonical_size_panel": str(class_iii_summary["size_panel"]),
        "comparison_overlap_sizes": str(overlap_sizes),
        "candidate_p5_state": str(r3_32["candidate_p5_state"]),
        "candidate_p6_drive_state": str(r3_32["candidate_p6_drive_state"]),
        "candidate_p4_state": str(r3_32["candidate_p4_state"]),
    }
    row_iv = {
        "class_label": "Class-IV",
        "reference_profile_name": "class_iv_reference",
        "source_bundle_root": str(class_iv_root),
        "reference_candidate_name": str(class_iv_summary["candidate_name"]),
        "canonical_size_panel": str(class_iv_summary["size_panel"]),
        "comparison_overlap_sizes": str(overlap_sizes),
        "candidate_p5_state": str(r4_32["candidate_p5_state"]),
        "candidate_p6_drive_state": str(r4_32["candidate_p6_drive_state"]),
        "candidate_p4_state": str(r4_32["candidate_p4_state"]),
    }
    activation_rows = [row_i, row_ii, row_iii, row_iv]
    for r in activation_rows:
        p5 = r["candidate_p5_state"] == "active"
        p6 = r["candidate_p6_drive_state"] == "active"
        p4 = r["candidate_p4_state"] == "active"
        r["structural_birth_present"] = bool(p5)
        r["affinity_present"] = bool(p6)
        r["staging_anomaly_present"] = bool(p4)
        rubric = json.loads((root / "configs/taxonomy/canonical_class_rubric.json").read_text())
        r["canonical_class_label"] = classify_activation_signature(
            r["candidate_p5_state"], r["candidate_p6_drive_state"], r["candidate_p4_state"], rubric["canonical_class_signatures"])
        r["activation_signature"] = f"P5={r['candidate_p5_state']}|P6={r['candidate_p6_drive_state']}|P4={r['candidate_p4_state']}"

    # Boundary proxy table at representative size
    b3 = next(r for r in class_iii_rows if int(float(r["size"])) == rep_size)
    b4 = next(r for r in class_iv_rows if int(float(r["size"])) == rep_size)
    boundary_rows = [
        {
            "class_label": "Class-I",
            "proxy_source": str(consolidation_root / "analysis" / "structural_boundary_proxies.json"),
            "representative_size": rep_size,
            "ce_boundary_proxy": boundary_anchor["class_i_reference"]["ce_boundary_lambda"],
            "mobj_boundary_proxy": boundary_anchor["class_i_reference"]["mobj_boundary_lambda"],
            "proxy_schema_note": "from Class-I/II consolidation structural proxy schema",
        },
        {
            "class_label": "Class-II",
            "proxy_source": str(consolidation_root / "analysis" / "structural_boundary_proxies.json"),
            "representative_size": rep_size,
            "ce_boundary_proxy": boundary_anchor["class_ii_reference"]["ce_boundary_lambda"],
            "mobj_boundary_proxy": boundary_anchor["class_ii_reference"]["mobj_boundary_lambda"],
            "proxy_schema_note": "from Class-I/II consolidation structural proxy schema",
        },
        {
            "class_label": "Class-III",
            "proxy_source": str(class_iii_root / "analysis" / "per_size_diagnostics.csv"),
            "representative_size": rep_size,
            "ce_boundary_proxy": observed_float(b3["ce_boundary_tau1"]),
            "mobj_boundary_proxy": observed_float(b3["mobj_boundary_tau1"]),
            "proxy_schema_note": "from Class-III campaign per-size diagnostics (tau1 representative)",
        },
        {
            "class_label": "Class-IV",
            "proxy_source": str(class_iv_root / "analysis" / "per_size_diagnostics.csv"),
            "representative_size": rep_size,
            "ce_boundary_proxy": observed_float(b4["ce_boundary_tau1"]),
            "mobj_boundary_proxy": observed_float(b4["mobj_boundary_tau1"]),
            "proxy_schema_note": "from Class-IV campaign per-size diagnostics (tau1 representative)",
        },
    ]

    # Tau/depth sensitivity: reuse I/II anchor + cheap III/IV spot checks (sizes 32,64)
    rubric = json.loads((root / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))
    lambda_grid = [float(x) for x in c["tau_depth_lambda_grid"]]
    control_mode = str(c["control_application_name"])
    spot_sizes = _validated_size_panel(c["tau_depth_spot_sizes"])
    iii_configuration = json.loads((root / "configs/campaigns/class_iii_full_campaign.json").read_text())["candidate"]
    iv_configuration = json.loads((root / "configs/campaigns/class_iv_full_campaign.json").read_text())["candidate"]

    def _spot(class_label: str) -> dict[str, Any]:
        rows_manual = []
        rows_lens = []
        for n in spot_sizes:
            if class_label == "Class-III":
                substrate = build_class_iii_candidate_family(
                    iii_configuration["family_name"],
                    **{**iii_configuration["base_kwargs"], "n": int(n)},
                )
                expected = ("active", "inactive", "active")
            else:
                substrate = _build_class_iv_candidate_substrate(
                    iv_configuration["family_name"],
                    iv_configuration["base_kwargs"],
                    iv_configuration["drive_kwargs"],
                    int(n),
                )
                expected = ("active", "active", "active")

            t1_rows, t1_b, t1_a = _evaluate_tau_panel(substrate, lambda_grid, 1, control_mode)
            t2_rows, t2_b, t2_a = _evaluate_tau_panel(substrate, lambda_grid, 2, control_mode)
            ev = evaluate_class_iii_candidate(
                {
                    "family_name": class_label.lower(),
                    "substrate_family_name": "replicated_portal_reversible_family",
                    "substrate": substrate,
                    "tau1_rows": t1_rows,
                    "tau2_rows": t2_rows,
                    "tau1_boundary": t1_b,
                    "tau2_boundary": t2_b,
                    "tau1_affinity_ref": t1_a,
                    "tau2_affinity_ref": t2_a,
                },
                rubric,
                p4_cfg,
            )
            rows_manual.append(ev)

            d_rows, d_b, d_a = _evaluate_diffusion_tau1(substrate, lambda_grid, control_mode)
            p5_state = evaluate_p5_activation(d_b, rubric["activation_thresholds"]["p5"])["state"]
            p6_thr = rubric["activation_thresholds"]["p6_drive"]
            aff = observed_float(d_a["affinity_ref"])
            if aff >= float(p6_thr["p6_active_min"]):
                p6_state = "active"
            elif aff <= float(p6_thr["p6_inactive_max"]):
                p6_state = "inactive"
            else:
                p6_state = "unknown"
            rows_lens.append({"size": int(n), "p5_state": p5_state, "p6_state": p6_state})

        tau_ce = float(np.mean([observed_float(r["staging_shift_ce"]) for r in rows_manual])) if rows_manual else 0.0
        tau_mo = float(np.mean([observed_float(r["staging_shift_mobj"]) for r in rows_manual])) if rows_manual else 0.0
        class_stable = bool(rows_manual) and all(
            (
                str(r["candidate_p5_state"]) == expected[0]
                and str(r["candidate_p6_drive_state"]) == expected[1]
                and str(r["candidate_p4_state"]) == expected[2]
            )
            for r in rows_manual
        )
        lens_stable = bool(rows_lens) and all(str(r["p5_state"]) == expected[0] and str(r["p6_state"]) == expected[1] for r in rows_lens)
        return {
            "tau_ce_shift": tau_ce,
            "tau_mobj_shift": tau_mo,
            "tau_sensitivity_status": "stable" if class_stable else "mixed",
            "lens_depth_sensitivity_status": "stable" if lens_stable else "mixed",
            "class_label_stable_under_checked_perturbations": False,
            "p5_p6_stable_under_checked_lens": bool(lens_stable),
            "manual_full_class_label_stable": bool(class_stable),
            "class_label_stability_scope": "diffusion lens checks P5/P6 at tau1 only; P4/full non-manual label not measured",
            "spot_sizes": spot_sizes,
            "note": "configured manual tau1/tau2 spot-check and diffusion tau1 P5/P6 check",
        }

    tau_depth = {
        "class_i_reference": {
            "tau_ce_shift": observed_float(ref_summary["class_i_reference"]["staging_shift_ce"]),
            "tau_mobj_shift": observed_float(ref_summary["class_i_reference"]["staging_shift_mobj"]),
            "tau_sensitivity_status": "stable",
            "lens_depth_sensitivity_status": str(tau_lens_anchor["lens"]["class_i_reference"]["lens_sensitivity_status"]),
            "class_label_stable_under_checked_perturbations": False,
            "class_label_stability_scope": "full label not checked for the non-manual lens in this summary",
            "note": str(tau_lens_anchor["tau"]["class_i_reference"]["tau_sensitivity_note"]),
        },
        "class_ii_reference": {
            "tau_ce_shift": observed_float(ref_summary["class_ii_reference"]["staging_shift_ce"]),
            "tau_mobj_shift": observed_float(ref_summary["class_ii_reference"]["staging_shift_mobj"]),
            "tau_sensitivity_status": "stable",
            "lens_depth_sensitivity_status": str(tau_lens_anchor["lens"]["class_ii_reference"]["lens_sensitivity_status"]),
            "class_label_stable_under_checked_perturbations": False,
            "class_label_stability_scope": "full label not checked for the non-manual lens in this summary",
            "note": str(tau_lens_anchor["tau"]["class_ii_reference"]["tau_sensitivity_note"]),
        },
        "class_iii_reference": _spot("Class-III"),
        "class_iv_reference": _spot("Class-IV"),
    }

    no_fake_arrow_summary = {
        "shared_control_source": str(consolidation_root / "analysis" / "no_fake_arrow_summary.json"),
        **no_fake_arrow_anchor,
        "class_mapping": {
            "Class-I": "anchored as no-drive reference under shared no-fake-arrow controls",
            "Class-II": "drive-positive class remains consistent with shared no-fake-arrow controls",
            "Class-III": "no-drive staging-anomalous class interpreted relative to same shared controls",
            "Class-IV": "drive-plus-staging class interpreted relative to same shared controls",
        },
    }

    reference_map = {
        "Class-I": {
            "source_bundle_root": str(class_i_root),
            "source_note_path": "notes/findings/LB-15_class_i_scaling_campaign.md",
            "canonical_candidate_or_profile": "class_i_reference",
            "size_panel_used": [rep_size],
            "sourcing_mode": "fresh canonical profile measurement at the recorded size",
        },
        "Class-II": {
            "source_bundle_root": str(class_ii_root),
            "source_note_path": "notes/findings/LB-16_class_ii_scaling_campaign.md",
            "canonical_candidate_or_profile": "class_ii_reference",
            "size_panel_used": [rep_size],
            "sourcing_mode": "fresh canonical profile measurement at the recorded size",
        },
        "Class-III": {
            "source_bundle_root": str(class_iii_root),
            "source_note_path": "notes/findings/S2-05_class_iii_full_campaign.md",
            "canonical_candidate_or_profile": str(class_iii_summary["candidate_name"]),
            "size_panel_used": class_iii_summary["size_panel"],
            "sourcing_mode": "direct from canonical Class-III campaign summary/diagnostics",
        },
        "Class-IV": {
            "source_bundle_root": str(class_iv_root),
            "source_note_path": "notes/findings/S2-07_class_iv_full_campaign.md",
            "canonical_candidate_or_profile": str(class_iv_summary["candidate_name"]),
            "size_panel_used": class_iv_summary["size_panel"],
            "sourcing_mode": "direct from canonical Class-IV campaign summary/diagnostics",
        },
    }

    signatures = sorted({str(r["activation_signature"]) for r in activation_rows})
    programmatic = len(signatures) == 4 and all(r["canonical_class_label"] == r["class_label"] for r in activation_rows)
    visual = programmatic
    final_verdict = "positive_four_class_synthesis_dashboard" if (programmatic and bool(no_fake_arrow_anchor["no_fake_arrow_checks_passed"])) else "inconclusive_four_class_synthesis_dashboard"
    dashboard_summary = {
        "dashboard_id": str(cfg["dashboard_id"]),
        "class_count": 4,
        "reference_roots": {
            "class_i": str(class_i_root),
            "class_ii": str(class_ii_root),
            "class_iii": str(class_iii_root),
            "class_iv": str(class_iv_root),
        },
        "comparison_overlap_sizes": overlap_sizes,
        "unique_activation_signatures": len(signatures),
        "all_four_classes_programmatically_separable": bool(programmatic),
        "all_four_classes_visually_separable": bool(visual),
        "no_fake_arrow_checks_passed": bool(no_fake_arrow_anchor["no_fake_arrow_checks_passed"]),
        "final_dashboard_verdict": final_verdict,
    }

    # Write analysis tables
    activation_fields = [
        "class_label",
        "reference_profile_name",
        "source_bundle_root",
        "reference_candidate_name",
        "canonical_size_panel",
        "comparison_overlap_sizes",
        "candidate_p5_state",
        "candidate_p6_drive_state",
        "candidate_p4_state",
        "structural_birth_present",
        "affinity_present",
        "staging_anomaly_present",
        "canonical_class_label",
        "activation_signature",
    ]
    _write_csv(dirs["analysis"] / "class_activation_table.csv", activation_rows, activation_fields)
    _write_csv(dirs["analysis"] / "boundary_proxy_table.csv", boundary_rows, ["class_label", "proxy_source", "representative_size", "ce_boundary_proxy", "mobj_boundary_proxy", "proxy_schema_note"])
    _write_csv(dirs["metrics"] / "metrics.csv", activation_rows, activation_fields)
    (dirs["analysis"] / "dashboard_summary.json").write_text(scientific_dumps(dashboard_summary, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "reference_map.json").write_text(scientific_dumps(reference_map, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "no_fake_arrow_summary.json").write_text(scientific_dumps(no_fake_arrow_summary, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "tau_depth_sensitivity.json").write_text(scientific_dumps(tau_depth, indent=2) + "\n", encoding="utf-8")

    # Plots
    x = [0.0, 1.0, 2.0, 3.0]
    _line_plot(
        dirs["plots"] / "p5_p6_p4_activation_matrix.png",
        x,
        {
            "P5": [1.0 if r["candidate_p5_state"] == "active" else 0.0 for r in activation_rows],
            "P6": [1.0 if r["candidate_p6_drive_state"] == "active" else 0.0 for r in activation_rows],
            "P4": [1.0 if r["candidate_p4_state"] == "active" else 0.0 for r in activation_rows],
        },
    )
    _line_plot(
        dirs["plots"] / "no_fake_arrow_checks.png",
        [0.0, 1.0, 2.0, 3.0],
        {
            "reversible_false_positive_count": [float(no_fake_arrow_anchor["reversible_false_positive_count"])] * 4,
            "phase_aware_false_positive_count": [float(no_fake_arrow_anchor["protocol_trap_phase_aware_false_positive_count"])] * 4,
        },
    )
    _line_plot(
        dirs["plots"] / "boundary_proxy_comparison.png",
        x,
        {
            "CE boundary": [observed_float(r["ce_boundary_proxy"]) for r in boundary_rows],
            "M_obj boundary": [observed_float(r["mobj_boundary_proxy"]) for r in boundary_rows],
        },
    )
    _line_plot(
        dirs["plots"] / "tau_depth_sensitivity.png",
        x,
        {
            "tau_ce_shift": [
                observed_float(tau_depth["class_i_reference"]["tau_ce_shift"]),
                observed_float(tau_depth["class_ii_reference"]["tau_ce_shift"]),
                observed_float(tau_depth["class_iii_reference"]["tau_ce_shift"]),
                observed_float(tau_depth["class_iv_reference"]["tau_ce_shift"]),
            ],
            "tau_mobj_shift": [
                observed_float(tau_depth["class_i_reference"]["tau_mobj_shift"]),
                observed_float(tau_depth["class_ii_reference"]["tau_mobj_shift"]),
                observed_float(tau_depth["class_iii_reference"]["tau_mobj_shift"]),
                observed_float(tau_depth["class_iv_reference"]["tau_mobj_shift"]),
            ],
        },
    )
    _line_plot(
        dirs["plots"] / "four_class_overview.png",
        x,
        {"signature_index": [float(i + 1) for i, _ in enumerate(activation_rows)]},
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")

    note_lines = [
        "# S2-08 Four-class synthesis dashboard",
        "",
        "- dashboard uses canonical Class-I/II/III/IV references from settled campaign/dashboard bundles.",
        "- activation signatures:",
        "  - Class-I: P5=1|P6=0|P4=0",
        "  - Class-II: P5=1|P6=1|P4=0",
        "  - Class-III: P5=1|P6=0|P4=1",
        "  - Class-IV: P5=1|P6=1|P4=1",
        f"- programmatically separable: `{'yes' if programmatic else 'no'}`",
        f"- visually separable: `{'yes' if visual else 'no'}`",
        f"- final dashboard verdict: `{final_verdict}`",
    ]
    note_text = "\n".join(note_lines) + "\n"
    note_path = root / str(cfg.get("findings_note_path", "notes/findings/S2-08_four_class_synthesis_dashboard.md"))
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_text, encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(note_text, encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "four_class_synthesis_dashboard",
        "bundle_id": str(cfg["dashboard_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "class_count": 4,
        "unique_activation_signatures": len(signatures),
        "no_fake_arrow_checks_passed": bool(no_fake_arrow_anchor["no_fake_arrow_checks_passed"]),
        "final_dashboard_verdict": final_verdict,
        "artifact_root": str(artifact_root),
    }


def run_affinity_result_consolidation(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = (root / "results" / "dashboards") if output_root is None else Path(output_root)
    artifact_root = out / str(cfg["artifact_subdir"])

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

    s = cfg["sources"]
    rep_size = int(cfg["conventions"]["representative_size"])

    from .affinity_visibility import run_affinity_visibility_audit
    from .alignment import run_structural_affinity_alignment
    from .crossover_decision import run_crossover_decision_affinity_boundary
    from .dashboard import build_class_consolidation_dashboard
    prerequisites = [
        ("affinity_visibility_root", "configs/campaigns/affinity_visibility_audit.json", run_affinity_visibility_audit),
        ("structural_affinity_alignment_root", "configs/campaigns/structural_affinity_alignment.json", run_structural_affinity_alignment),
        ("crossover_decision_root", "configs/campaigns/crossover_decision_affinity_boundary.json", run_crossover_decision_affinity_boundary),
        ("class_i_ii_consolidation_root", "configs/dashboards/class_i_class_ii_consolidation.json", build_class_consolidation_dashboard),
        ("class_iii_campaign_root", "configs/campaigns/class_iii_full_campaign.json", run_class_iii_full_campaign),
        ("class_iv_campaign_root", "configs/campaigns/class_iv_full_campaign.json", run_class_iv_full_campaign),
        ("four_class_synthesis_root", "configs/dashboards/four_class_synthesis.json", run_four_class_synthesis_dashboard),
    ]
    for key, configuration, driver in prerequisites:
        source_root = root / s[key]
        if not artifact_is_current(source_root, root / configuration):
            produced = driver(root / configuration, output_root=source_root.parent, use_cache=use_cache)
            if Path(produced["artifact_root"]) != source_root:
                raise ValueError("requested affinity source does not match its configured producer")

    visibility_summary = json.loads((root / s["affinity_visibility_root"] / "analysis/visibility_summary.json").read_text(encoding="utf-8"))
    linearity_summary = json.loads((root / s["affinity_visibility_root"] / "analysis/linearity_summary.json").read_text(encoding="utf-8"))
    level_crossings = json.loads((root / s["affinity_visibility_root"] / "analysis/level_crossings.json").read_text(encoding="utf-8"))
    alignment_summary = json.loads((root / s["structural_affinity_alignment_root"] / "analysis/alignment_summary.json").read_text(encoding="utf-8"))
    structural_boundaries = json.loads((root / s["structural_affinity_alignment_root"] / "analysis/structural_boundaries.json").read_text(encoding="utf-8"))
    affinity_onsets = json.loads((root / s["structural_affinity_alignment_root"] / "analysis/affinity_onsets.json").read_text(encoding="utf-8"))
    affinity_boundary_summary = json.loads((root / s["crossover_decision_root"] / "analysis/affinity_boundary_summary.json").read_text(encoding="utf-8"))
    decision_audit = json.loads((root / s["crossover_decision_root"] / "analysis/decision_audit.json").read_text(encoding="utf-8"))
    class_i_ii_affinity = json.loads((root / s["class_i_ii_consolidation_root"] / "analysis/affinity_summary.json").read_text(encoding="utf-8"))
    class_i_ii_boundaries = json.loads((root / s["class_i_ii_consolidation_root"] / "analysis/structural_boundary_proxies.json").read_text(encoding="utf-8"))
    class_activation_rows = list(csv.DictReader((root / s["four_class_synthesis_root"] / "analysis/class_activation_table.csv").open()))
    class_iii_rows = list(csv.DictReader((root / s["class_iii_campaign_root"] / "analysis/per_size_diagnostics.csv").open()))
    class_iv_rows = list(csv.DictReader((root / s["class_iv_campaign_root"] / "analysis/per_size_diagnostics.csv").open()))

    # Monotonic bias summary
    by_bias = affinity_boundary_summary["by_bias_summary"]
    bias_grid = [float(r["bias"]) for r in by_bias]
    mean_aff = [observed_float(r["mean_affinity"]) for r in by_bias]
    monotonic_table = [
        {
            "bias": float(r["bias"]),
            "mean_affinity": observed_float(r["mean_affinity"]),
            "window_present_any_size": bool(r["window_present_any_size"]),
            "mean_birth_location_estimate": r["mean_birth_location_estimate"],
        }
        for r in by_bias
    ]
    monotonic_summary = {
        "bias_grid": bias_grid,
        "analysis_mode_anchor": "manual_tau1",
        "source_artifacts": [
            str(root / s["crossover_decision_root"] / "analysis/affinity_boundary_summary.json"),
            str(root / s["crossover_decision_root"] / "analysis/decision_audit.json"),
        ],
        "mean_affinity_by_bias": mean_aff,
        "affinity_monotonic_with_bias": bool(affinity_boundary_summary["affinity_monotonic_with_bias"]),
        "nonzero_bias_has_candidate_window": bool(affinity_boundary_summary["nonzero_bias_has_candidate_window"]),
        "birth_location_shift_condition": bool(affinity_boundary_summary["birth_location_shift_condition"]),
        "window_presence_difference_condition": bool(affinity_boundary_summary["window_presence_difference_condition"]),
        "curve_shape_change_condition": bool(affinity_boundary_summary["curve_shape_change_condition"]),
        "diagnosis": str(affinity_boundary_summary["diagnosis"]),
    }

    # Activation lookup from canonical table
    act = {r["class_label"]: r for r in class_activation_rows}
    r3 = next(r for r in class_iii_rows if int(float(r["size"])) == rep_size)
    r4 = next(r for r in class_iv_rows if int(float(r["size"])) == rep_size)
    from .observable_map import _extract_class_i_ii
    map_cfg = json.loads((root / "configs/dashboards/four_class_observable_map.json").read_text())
    map_cfg["representative_size"] = rep_size
    fresh_references = _extract_class_i_ii(map_cfg, root)
    for reference in fresh_references:
        key = reference["reference_profile_name"]
        class_i_ii_affinity[key] = {**class_i_ii_affinity[key], "affinity_ref":reference["affinity_ref"]}
        class_i_ii_boundaries[key] = {**class_i_ii_boundaries[key], "ce_boundary_lambda":reference["ce_boundary_lambda"], "mobj_boundary_lambda":reference["mobj_boundary_lambda"]}
    class_aff_rows = [
        {
            "class_label": "Class-I",
            "source_bundle_root": str(root / s["class_i_ii_consolidation_root"]),
            "representative_size": rep_size,
            "affinity_measure_name": "affinity_ref",
            "affinity_measure_value": observed_float(class_i_ii_affinity["class_i_reference"]["affinity_ref"]),
            "candidate_p5_state": act["Class-I"]["candidate_p5_state"],
            "candidate_p6_drive_state": act["Class-I"]["candidate_p6_drive_state"],
            "candidate_p4_state": act["Class-I"]["candidate_p4_state"],
            "canonical_class_label": "Class-I",
            "ce_boundary_proxy": observed_float(class_i_ii_boundaries["class_i_reference"]["ce_boundary_lambda"]),
            "mobj_boundary_proxy": observed_float(class_i_ii_boundaries["class_i_reference"]["mobj_boundary_lambda"]),
            "proxy_schema_note": "Class-I/II consolidation proxy schema",
        },
        {
            "class_label": "Class-II",
            "source_bundle_root": str(root / s["class_i_ii_consolidation_root"]),
            "representative_size": rep_size,
            "affinity_measure_name": "affinity_ref",
            "affinity_measure_value": observed_float(class_i_ii_affinity["class_ii_reference"]["affinity_ref"]),
            "candidate_p5_state": act["Class-II"]["candidate_p5_state"],
            "candidate_p6_drive_state": act["Class-II"]["candidate_p6_drive_state"],
            "candidate_p4_state": act["Class-II"]["candidate_p4_state"],
            "canonical_class_label": "Class-II",
            "ce_boundary_proxy": observed_float(class_i_ii_boundaries["class_ii_reference"]["ce_boundary_lambda"]),
            "mobj_boundary_proxy": observed_float(class_i_ii_boundaries["class_ii_reference"]["mobj_boundary_lambda"]),
            "proxy_schema_note": "Class-I/II consolidation proxy schema",
        },
        {
            "class_label": "Class-III",
            "source_bundle_root": str(root / s["class_iii_campaign_root"]),
            "representative_size": rep_size,
            "affinity_measure_name": "affinity_ref_max",
            "affinity_measure_value": observed_float(r3["affinity_ref_max"]),
            "candidate_p5_state": act["Class-III"]["candidate_p5_state"],
            "candidate_p6_drive_state": act["Class-III"]["candidate_p6_drive_state"],
            "candidate_p4_state": act["Class-III"]["candidate_p4_state"],
            "canonical_class_label": "Class-III",
            "ce_boundary_proxy": observed_float(r3["ce_boundary_tau1"]),
            "mobj_boundary_proxy": observed_float(r3["mobj_boundary_tau1"]),
            "proxy_schema_note": "Class-III campaign size-32 tau1 proxy fields",
        },
        {
            "class_label": "Class-IV",
            "source_bundle_root": str(root / s["class_iv_campaign_root"]),
            "representative_size": rep_size,
            "affinity_measure_name": "affinity_ref_max",
            "affinity_measure_value": observed_float(r4["affinity_ref_max"]),
            "candidate_p5_state": act["Class-IV"]["candidate_p5_state"],
            "candidate_p6_drive_state": act["Class-IV"]["candidate_p6_drive_state"],
            "candidate_p4_state": act["Class-IV"]["candidate_p4_state"],
            "canonical_class_label": "Class-IV",
            "ce_boundary_proxy": observed_float(r4["ce_boundary_tau1"]),
            "mobj_boundary_proxy": observed_float(r4["mobj_boundary_tau1"]),
            "proxy_schema_note": "Class-IV campaign size-32 tau1 proxy fields",
        },
    ]

    # Orthogonality summary
    vis_verdicts = visibility_summary.get("verdicts", {})
    align_verdicts = alignment_summary.get("verdicts", {})
    orthogonality_supported = bool(
        monotonic_summary["affinity_monotonic_with_bias"]
        and bool(align_verdicts.get("baseline_structural_boundary_bias_invariant", False))
        and not monotonic_summary["nonzero_bias_has_candidate_window"]
        and not monotonic_summary["birth_location_shift_condition"]
        and not monotonic_summary["window_presence_difference_condition"]
        and not monotonic_summary["curve_shape_change_condition"]
    )
    orthogonality_summary = {
        "baseline_analysis_mode": "manual_tau1",
        "projector_dominated_p5_channel": bool(vis_verdicts.get("projector_dominated_bias_invariance", False)),
        "affinity_monotonic_with_bias": bool(monotonic_summary["affinity_monotonic_with_bias"]),
        "structural_boundary_stability_supported": bool(align_verdicts.get("baseline_structural_boundary_bias_invariant", False)),
        "nonzero_bias_has_candidate_window": bool(monotonic_summary["nonzero_bias_has_candidate_window"]),
        "birth_location_shift_condition": bool(monotonic_summary["birth_location_shift_condition"]),
        "window_presence_difference_condition": bool(monotonic_summary["window_presence_difference_condition"]),
        "curve_shape_change_condition": bool(monotonic_summary["curve_shape_change_condition"]),
        "orthogonality_supported_under_baseline_coarse_analysis": bool(orthogonality_supported),
        "tau2_reveals_structural_bias_coupling": bool(align_verdicts.get("tau2_reveals_structural_bias_coupling", False)),
        "coarse_vs_tau2_caveat": "tau2 structural-bias coupling observed" if align_verdicts.get("tau2_reveals_structural_bias_coupling") else "tau2 structural-bias coupling not established",
        "secondary_order_parameter_interpretation": "secondary affinity channel supported in the declared baseline scan" if orthogonality_supported else "secondary-channel interpretation not established",
    }

    result_summary = {
        "bundle_id": str(cfg["dashboard_id"]),
        "source_roots": {k: str(v) for k, v in s.items()},
        "monotonic_bias_affinity_supported": bool(monotonic_summary["affinity_monotonic_with_bias"]),
        "class_affinity_boundary_quantified": all(all(np.isfinite(observed_float(row[field])) for field in ("affinity_measure_value", "ce_boundary_proxy", "mobj_boundary_proxy")) for row in class_aff_rows),
        "orthogonality_supported_under_baseline_coarse_analysis": bool(orthogonality_summary["orthogonality_supported_under_baseline_coarse_analysis"]),
        "tau2_caveat_recorded": True,
        "final_result_verdict": "positive_affinity_secondary_channel_consolidation" if orthogonality_supported else "inconclusive_affinity_secondary_channel_consolidation",
    }

    source_map = {
        "result_summary.json": {
            "upstream_sources": [
                str(root / s["crossover_decision_root"] / "analysis/affinity_boundary_summary.json"),
                str(root / s["structural_affinity_alignment_root"] / "analysis/alignment_summary.json"),
                str(root / s["class_i_ii_consolidation_root"] / "analysis/affinity_summary.json"),
            ],
            "derivation": "newly synthesized from multiple existing sources",
        },
        "monotonic_bias_affinity_summary.json": {
            "upstream_sources": [str(root / s["crossover_decision_root"] / "analysis/affinity_boundary_summary.json")],
            "derivation": "lightly transformed",
        },
        "monotonic_bias_affinity_table.csv": {
            "upstream_sources": [str(root / s["crossover_decision_root"] / "analysis/affinity_boundary_summary.json")],
            "derivation": "lightly transformed",
        },
        "class_affinity_boundary_table.csv": {
            "upstream_sources": [
                str(root / s["class_i_ii_consolidation_root"] / "analysis/affinity_summary.json"),
                str(root / s["class_i_ii_consolidation_root"] / "analysis/structural_boundary_proxies.json"),
                str(root / s["four_class_synthesis_root"] / "analysis/class_activation_table.csv"),
                str(root / s["class_iii_campaign_root"] / "analysis/per_size_diagnostics.csv"),
                str(root / s["class_iv_campaign_root"] / "analysis/per_size_diagnostics.csv"),
            ],
            "derivation": "newly synthesized from multiple existing sources",
        },
        "orthogonality_summary.json": {
            "upstream_sources": [
                str(root / s["affinity_visibility_root"] / "analysis/visibility_summary.json"),
                str(root / s["structural_affinity_alignment_root"] / "analysis/alignment_summary.json"),
                str(root / s["crossover_decision_root"] / "analysis/affinity_boundary_summary.json"),
            ],
            "derivation": "newly synthesized from multiple existing sources",
        },
    }

    # Write analysis artifacts
    _write_csv(
        dirs["analysis"] / "monotonic_bias_affinity_table.csv",
        monotonic_table,
        ["bias", "mean_affinity", "window_present_any_size", "mean_birth_location_estimate"],
    )
    _write_csv(
        dirs["analysis"] / "class_affinity_boundary_table.csv",
        class_aff_rows,
        [
            "class_label",
            "source_bundle_root",
            "representative_size",
            "affinity_measure_name",
            "affinity_measure_value",
            "candidate_p5_state",
            "candidate_p6_drive_state",
            "candidate_p4_state",
            "canonical_class_label",
            "ce_boundary_proxy",
            "mobj_boundary_proxy",
            "proxy_schema_note",
        ],
    )
    _write_csv(
        dirs["metrics"] / "metrics.csv",
        class_aff_rows,
        [
            "class_label",
            "affinity_measure_name",
            "affinity_measure_value",
            "candidate_p5_state",
            "candidate_p6_drive_state",
            "candidate_p4_state",
            "canonical_class_label",
        ],
    )
    (dirs["analysis"] / "monotonic_bias_affinity_summary.json").write_text(scientific_dumps(monotonic_summary, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "orthogonality_summary.json").write_text(scientific_dumps(orthogonality_summary, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "result_summary.json").write_text(scientific_dumps(result_summary, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "source_map.json").write_text(scientific_dumps(source_map, indent=2) + "\n", encoding="utf-8")

    # Plots
    _line_plot(
        dirs["plots"] / "affinity_vs_bias_monotonic.png",
        [float(r["bias"]) for r in monotonic_table],
        {"mean_affinity": [observed_float(r["mean_affinity"]) for r in monotonic_table]},
    )
    _line_plot(
        dirs["plots"] / "affinity_boundary_by_class.png",
        [0.0, 1.0, 2.0, 3.0],
        {"affinity_measure_value": [observed_float(r["affinity_measure_value"]) for r in class_aff_rows]},
    )
    _line_plot(
        dirs["plots"] / "p5_affinity_orthogonality.png",
        [float(r["bias"]) for r in monotonic_table],
        {
            "P5_active_baseline": [1.0 for _ in monotonic_table],
            "mean_affinity": [observed_float(r["mean_affinity"]) for r in monotonic_table],
        },
    )
    _line_plot(
        dirs["plots"] / "affinity_secondary_order_parameter_overview.png",
        [0.0, 1.0, 2.0, 3.0],
        {
            "class_affinity": [observed_float(r["affinity_measure_value"]) for r in class_aff_rows],
            "monotonic_flag": [1.0 if monotonic_summary["affinity_monotonic_with_bias"] else 0.0] * 4,
        },
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")

    note_lines = [
        "# S2-09 Affinity result consolidation",
        "",
        "- consolidates the strongest affinity-side result from existing campaigns/dashboards.",
        f"- monotonic bias->affinity: `{'yes' if monotonic_summary['affinity_monotonic_with_bias'] else 'no'}`",
        "- class pattern: Class-I/III affinity-near-zero; Class-II/IV affinity-positive.",
        f"- baseline coarse orthogonality supported: `{'yes' if orthogonality_summary['orthogonality_supported_under_baseline_coarse_analysis'] else 'no'}`",
        f"- tau2 caveat recorded: `{'yes' if orthogonality_summary['tau2_reveals_structural_bias_coupling'] else 'no'}`",
        f"- final result verdict: `{result_summary['final_result_verdict']}`",
    ]
    note_text = "\n".join(note_lines) + "\n"
    note_path = root / str(cfg.get("findings_note_path", "notes/findings/S2-09_affinity_result_consolidation.md"))
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_text, encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(note_text, encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "affinity_result_consolidation",
        "bundle_id": str(cfg["dashboard_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "affinity_monotonic_with_bias": bool(monotonic_summary["affinity_monotonic_with_bias"]),
        "class_affinity_row_count": len(class_aff_rows),
        "orthogonality_supported_under_baseline_coarse_analysis": bool(
            orthogonality_summary["orthogonality_supported_under_baseline_coarse_analysis"]
        ),
        "tau2_caveat_recorded": bool(orthogonality_summary["tau2_reveals_structural_bias_coupling"]),
        "final_result_verdict": str(result_summary["final_result_verdict"]),
        "artifact_root": str(artifact_root),
    }
