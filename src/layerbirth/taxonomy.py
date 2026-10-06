"""S2-01 canonical class rubric and machine-readable taxonomy."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .campaigns import _git_code_version, _line_plot, _repo_root, _write_csv
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import pushforward_matrix
from .p4 import evaluate_p4_profile
from .sweep import apply_closure_strength_control
from .substrates import build_substrate_family
from .boundaries import first_threshold_lambda, interpolate_value, structural_reference_lambda


def _primitive_activation_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [str(r["canonical_class_label"]) for r in rows]
    cols = ["P5", "P6", "P4"]
    data = np.asarray(
        [
            [
                1.0 if str(r["p5_state"]) == "active" else 0.0,
                1.0 if str(r["p6_drive_state"]) == "active" else 0.0,
                1.0 if str(r["p4_state"]) == "active" else 0.0,
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(3.15, 3.0), dpi=180)
    ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(labelsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "on" if data[i, j] >= 0.5 else "off", ha="center", va="center", fontsize=7, color="black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _canonical_signature_rows(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    order = ["Class-I", "Class-II", "Class-III", "Class-IV"]
    rows: list[dict[str, Any]] = []
    for label in order:
        sig = canonical[label]
        rows.append(
            {
                "canonical_class_label": label,
                "p5_state": "active" if bool(sig["P5_active"]) else "inactive",
                "p6_drive_state": "active" if bool(sig["P6_drive_active"]) else "inactive",
                "p4_state": "active" if bool(sig["P4_anomalous"]) else "inactive",
            }
        )
    return rows


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def _first_threshold_lambda(
    curve_rows: list[dict[str, Any]],
    observable_key: str,
    target: float,
    mode: str,
) -> float | None:
    return first_threshold_lambda(curve_rows, observable_key, target, mode)


def _interp_value_at_lambda(curve_rows: list[dict[str, Any]], observable_key: str, lam: float | None) -> float | None:
    return interpolate_value(curve_rows, observable_key, lam)


def estimate_structural_boundaries(rows: list[dict[str, Any]], ce_target: float = 0.025, mobj_target: float = 0.90) -> dict[str, Any]:
    ce = _first_threshold_lambda(rows, "closure_error", float(ce_target), mode="leq")
    mo = _first_threshold_lambda(rows, "objecthood_order", float(mobj_target), mode="geq")
    min_ce = float(min((observed_float(r["closure_error"]) for r in rows), default=float("nan")))
    max_mo = float(max((observed_float(r["objecthood_order"]) for r in rows), default=float("nan")))
    reference = structural_reference_lambda(ce, mo)
    ce_at_ref = interpolate_value(rows, "closure_error", reference)
    mo_at_ref = interpolate_value(rows, "objecthood_order", reference)
    return {
        "ce_boundary_lambda": ce,
        "mobj_boundary_lambda": mo,
        "min_ce": min_ce,
        "max_mobj": max_mo,
        "structural_boundary_lambda_ref": reference,
        "structural_targets_met_at_reference": bool(
            ce_at_ref is not None and mo_at_ref is not None
            and ce_at_ref <= ce_target + 1e-12 and mo_at_ref >= mobj_target - 1e-12
        ),
    }


def estimate_affinity_reference(rows: list[dict[str, Any]], boundary_summary: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    # An unrelated supplied/global affinity cannot establish affinity at birth.
    _ = fallback
    lam = structural_reference_lambda(
        boundary_summary.get("ce_boundary_lambda"), boundary_summary.get("mobj_boundary_lambda")
    )
    if boundary_summary.get("structural_targets_met_at_reference") is False:
        lam = None
    ref = interpolate_value(rows, "affinity", lam)
    return {
        "affinity_ref": float("nan") if ref is None else ref,
        "source": "unavailable_structural_reference" if ref is None else "affinity_at_structural_reference",
        "structural_boundary_lambda_ref": lam,
        "affinity_ref_available": ref is not None,
    }


def evaluate_p5_activation(boundary_summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    ce_target = float(thresholds["ce_target"])
    mobj_target = float(thresholds["mobj_target"])
    ce_inactive = float(thresholds["ce_inactive_min"])
    mobj_inactive = float(thresholds["mobj_inactive_max"])
    ce_ok = boundary_summary.get("ce_boundary_lambda") is not None
    mo_ok = boundary_summary.get("mobj_boundary_lambda") is not None
    if ce_ok and mo_ok and boundary_summary.get("structural_targets_met_at_reference", True):
        return {"state": "active"}
    if float(boundary_summary["min_ce"]) > ce_inactive and float(boundary_summary["max_mobj"]) < mobj_inactive:
        return {"state": "inactive"}
    return {"state": "unknown", "ce_target": ce_target, "mobj_target": mobj_target}


def evaluate_p6_drive_activation(affinity_ref: float, thresholds: dict[str, Any]) -> dict[str, Any]:
    affinity_ref = observed_float(affinity_ref)
    a_min = float(thresholds["p6_active_min"])
    i_max = float(thresholds["p6_inactive_max"])
    if not np.isfinite(a_min) or not np.isfinite(i_max) or not 0 <= i_max < a_min:
        raise ValueError("drive thresholds must satisfy 0 <= inactive < active")
    if np.isnan(affinity_ref) or float(affinity_ref) < 0.0:
        return {"state": "unknown"}
    if float(affinity_ref) >= a_min:
        return {"state": "active"}
    if float(affinity_ref) <= i_max:
        return {"state": "inactive"}
    return {"state": "unknown"}


def evaluate_p4_anomaly(staging_check_summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    if "p4_like_shift_min" in thresholds:
        p4_thresholds = {
            "p4_like_shift_min": float(thresholds["p4_like_shift_min"]),
            "p4_class_dual_shift_min": float(thresholds.get("p4_class_dual_shift_min", 0.15)),
        }
    else:
        p4_thresholds = {
            "p4_like_shift_min": 0.05,
            "p4_class_dual_shift_min": float(thresholds.get("p4_boundary_shift_active_min", 0.15)),
        }

    staging_shift_ce = observed_float(
        staging_check_summary.get(
            "staging_shift_ce",
            abs(observed_float(staging_check_summary.get("tau_ce_shift"))),
        )
    )
    staging_shift_mobj = observed_float(
        staging_check_summary.get(
            "staging_shift_mobj",
            abs(observed_float(staging_check_summary.get("tau_mobj_shift"))),
        )
    )
    presence_shift_ce = bool(
        staging_check_summary.get(
            "presence_shift_ce",
            bool(staging_check_summary.get("tau_ce_presence_change", False)),
        )
    )
    presence_shift_mobj = bool(
        staging_check_summary.get(
            "presence_shift_mobj",
            bool(staging_check_summary.get("tau_mobj_presence_change", False)),
        )
    )
    normalized = {
        "staging_shift_ce": staging_shift_ce,
        "staging_shift_mobj": staging_shift_mobj,
        "presence_shift_ce": presence_shift_ce,
        "presence_shift_mobj": presence_shift_mobj,
        "presence_shift_any": bool(
            staging_check_summary.get(
                "presence_shift_any",
                presence_shift_ce or presence_shift_mobj,
            )
        ),
    }
    p4_eval = evaluate_p4_profile(normalized, p4_thresholds)
    return {
        "state": p4_eval["p4_state"],
        "p4_like_signal": bool(p4_eval["p4_like_signal"]),
        "p4_class_active": bool(p4_eval["p4_class_active"]),
        "staging_gap_anomaly_score": observed_float(p4_eval["staging_gap_anomaly_score"]),
    }


def classify_activation_signature(
    p5_state: str,
    p6_state: str,
    p4_state: str,
    canonical_signatures: dict[str, dict[str, bool]],
) -> str:
    if p5_state not in {"active", "inactive"} or p6_state not in {"active", "inactive"} or p4_state not in {"active", "inactive"}:
        return "unclassified"
    sig = {
        "P5_active": p5_state == "active",
        "P6_drive_active": p6_state == "active",
        "P4_anomalous": p4_state == "active",
    }
    for cls, s in canonical_signatures.items():
        if bool(s["P5_active"]) == sig["P5_active"] and bool(s["P6_drive_active"]) == sig["P6_drive_active"] and bool(s["P4_anomalous"]) == sig["P4_anomalous"]:
            return cls
    return "unclassified"


def _build_curve_rows_for_profile(profile: dict[str, Any], tau: int | None = None) -> list[dict[str, Any]]:
    family = str(profile["family_name"])
    lam_grid = [float(x) for x in profile["lambda_grid"]]
    size = int(profile["size"])
    if tau is None:
        tau = int(profile["tau"])
    if family == "reversible_block_family":
        substrate = build_substrate_family(
            "reversible_block_family",
            n_blocks=int(profile["n_blocks"]),
            block_size=int(profile["block_size"]),
            intra_block_weight=float(profile["intra_block_weight"]),
            inter_block_weight=float(profile["inter_block_weight"]),
            self_weight=float(profile["self_weight"]),
        )
        lens = np.asarray(substrate["block_lens"], dtype=np.int64)
    elif family == "driven_cycle_family":
        substrate = build_substrate_family(
            "driven_cycle_family",
            n=size,
            self_weight=float(profile["self_weight"]),
            forward_weight=float(profile["forward_weight"]),
            backward_weight=float(profile["backward_weight"]),
        )
        lens = np.asarray([0 if i < (size // 2) else 1 for i in range(size)], dtype=np.int64)
    else:
        raise ValueError(f"unsupported family for taxonomy profile: {family}")
    p_base = np.asarray(substrate["P"], dtype=np.float64)
    q = pushforward_matrix(lens, 2)
    u = np.asarray(build_lift_family("uniform_lift_family", f=lens, k=2)[0], dtype=np.float64)
    out: list[dict[str, Any]] = []
    for lam in lam_grid:
        p = apply_closure_strength_control(
            p_base,
            closure_strength_lambda=lam,
            Q_f=q,
            U_f=u,
            mode=str(profile["control_application_name"]),
        )
        bundle = default_metric_bundle(p, lens, tau=int(tau))
        out.append(
            {
                "closure_strength_lambda": float(lam),
                "closure_error": observed_float(bundle["closure_error"]),
                "objecthood_order": observed_float(bundle["objecthood_order"]),
                "affinity": observed_float(bundle["affinity"]),
            }
        )
    return out


def _staging_summary_from_profile(staging_cfg: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    tau1_rows = _build_curve_rows_for_profile(staging_cfg, tau=1)
    tau2_rows = _build_curve_rows_for_profile(staging_cfg, tau=2)
    b1 = estimate_structural_boundaries(
        tau1_rows,
        ce_target=float(thresholds["ce_target"]),
        mobj_target=float(thresholds["mobj_target"]),
    )
    b2 = estimate_structural_boundaries(
        tau2_rows,
        ce_target=float(thresholds["ce_target"]),
        mobj_target=float(thresholds["mobj_target"]),
    )
    return {
        "tau1": b1,
        "tau2": b2,
        "tau_ce_shift": None if (b1["ce_boundary_lambda"] is None or b2["ce_boundary_lambda"] is None) else float(b2["ce_boundary_lambda"] - b1["ce_boundary_lambda"]),
        "tau_mobj_shift": None if (b1["mobj_boundary_lambda"] is None or b2["mobj_boundary_lambda"] is None) else float(b2["mobj_boundary_lambda"] - b1["mobj_boundary_lambda"]),
        "tau_ce_presence_change": bool((b1["ce_boundary_lambda"] is None) != (b2["ce_boundary_lambda"] is None)),
        "tau_mobj_presence_change": bool((b1["mobj_boundary_lambda"] is None) != (b2["mobj_boundary_lambda"] is None)),
    }


def evaluate_reference_profile(
    profile_config: dict[str, Any],
    rubric_config: dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    _ = output_root, use_cache
    thresholds = rubric_config["activation_thresholds"]
    curve_rows = _build_curve_rows_for_profile(profile_config, tau=int(profile_config["tau"]))
    boundary = estimate_structural_boundaries(
        curve_rows,
        ce_target=float(thresholds["p5"]["ce_target"]),
        mobj_target=float(thresholds["p5"]["mobj_target"]),
    )
    p5 = evaluate_p5_activation(boundary, thresholds["p5"])
    aff_ref = estimate_affinity_reference(curve_rows, boundary, fallback=None)
    p6 = evaluate_p6_drive_activation(aff_ref["affinity_ref"], thresholds["p6_drive"])

    staging_summary = _staging_summary_from_profile(profile_config["staging_check"], thresholds["p5"])
    p4_eval = evaluate_p4_anomaly(staging_summary, thresholds["p4_anomalous"])
    p4_state = p4_eval["state"]

    cls = classify_activation_signature(p5["state"], p6["state"], p4_state, rubric_config["canonical_class_signatures"])
    return {
        "profile_name": str(profile_config["profile_name"]),
        "p5_state": p5["state"],
        "p6_drive_state": p6["state"],
        "p4_state": p4_state,
        "p4_like_signal": bool(p4_eval["p4_like_signal"]),
        "p4_class_active": bool(p4_eval["p4_class_active"]),
        "staging_gap_anomaly_score": observed_float(p4_eval["staging_gap_anomaly_score"]),
        "canonical_class_label": cls,
        "ce_boundary_lambda": boundary["ce_boundary_lambda"],
        "mobj_boundary_lambda": boundary["mobj_boundary_lambda"],
        "affinity_ref": aff_ref["affinity_ref"],
        "tau_ce_shift": staging_summary["tau_ce_shift"],
        "tau_mobj_shift": staging_summary["tau_mobj_shift"],
        "diagnosis": f"classified via canonical rubric with p5={p5['state']}, p6={p6['state']}, p4={p4_state}, p4_like={p4_eval['p4_like_signal']}",
        "staging_check": {
            **staging_summary,
            "p4_like_signal": bool(p4_eval["p4_like_signal"]),
            "p4_class_active": bool(p4_eval["p4_class_active"]),
            "staging_gap_anomaly_score": observed_float(p4_eval["staging_gap_anomaly_score"]),
        },
    }


def run_canonical_class_rubric(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    config = _coerce_config(config_path_or_obj)
    root = _repo_root()
    if output_root is None:
        output_root = root / "results" / "taxonomy"
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

    canonical = config["canonical_class_signatures"]
    references = config["reference_profiles"]

    ref_rows: list[dict[str, Any]] = []
    staging_rows: dict[str, Any] = {}
    for ref in references:
        row = evaluate_reference_profile(ref, config, output_root=output_root, use_cache=use_cache)
        ref_rows.append(row)
        staging_rows[row["profile_name"]] = row["staging_check"]

    (dirs["analysis"] / "canonical_taxonomy.json").write_text(
        scientific_dumps({"canonical_class_signatures": canonical}, indent=2) + "\n", encoding="utf-8"
    )
    (dirs["analysis"] / "reference_classifications.json").write_text(
        scientific_dumps({"reference_classifications": ref_rows}, indent=2) + "\n", encoding="utf-8"
    )
    (dirs["analysis"] / "reference_staging_checks.json").write_text(
        scientific_dumps({"reference_staging_checks": staging_rows}, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        dirs["analysis"] / "primitive_activation_table.csv",
        ref_rows,
        [
            "profile_name",
            "p5_state",
            "p6_drive_state",
            "p4_state",
            "p4_like_signal",
            "p4_class_active",
            "canonical_class_label",
            "ce_boundary_lambda",
            "mobj_boundary_lambda",
            "affinity_ref",
            "tau_ce_shift",
            "tau_mobj_shift",
            "staging_gap_anomaly_score",
        ],
    )
    _write_csv(
        dirs["metrics"] / "metrics.csv",
        ref_rows,
        [
            "profile_name",
            "p5_state",
            "p6_drive_state",
            "p4_state",
            "p4_like_signal",
            "p4_class_active",
            "canonical_class_label",
            "ce_boundary_lambda",
            "mobj_boundary_lambda",
            "affinity_ref",
            "tau_ce_shift",
            "tau_mobj_shift",
            "staging_gap_anomaly_score",
        ],
    )

    x = [0, 1]
    _primitive_activation_heatmap(
        dirs["plots"] / "primitive_activation_heatmap.png",
        _canonical_signature_rows(canonical),
    )
    _line_plot(
        dirs["plots"] / "reference_staging_shifts.png",
        x,
        {r["profile_name"]: [observed_float(r["tau_ce_shift"]), observed_float(r["tau_mobj_shift"])] for r in ref_rows},
    )

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    notes = dirs["notes"] / "findings.md"
    notes.write_text(
        "\n".join(
            [
                "# Canonical class rubric findings",
                "",
                f"- class_i_reference: `{next(r for r in ref_rows if r['profile_name']=='class_i_reference')}`",
                f"- class_ii_reference: `{next(r for r in ref_rows if r['profile_name']=='class_ii_reference')}`",
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

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "canonical_class_rubric",
        "bundle_id": str(config["rubric_id"]),
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

    findings_note_rel = str(config.get("findings_note_path", "notes/findings/S2-01_canonical_class_rubric.md"))
    findings_note = root / findings_note_rel
    findings_note.parent.mkdir(parents=True, exist_ok=True)
    c1 = next(r for r in ref_rows if r["profile_name"] == "class_i_reference")
    c2 = next(r for r in ref_rows if r["profile_name"] == "class_ii_reference")
    clean = c1["canonical_class_label"] == "Class-I" and c2["canonical_class_label"] == "Class-II"
    findings_note.write_text(
        "\n".join(
            [
                "# S2-01 canonical class rubric",
                "",
                "- config path: `configs/taxonomy/canonical_class_rubric.json`",
                f"- artifact root: `{artifact_root}`",
                f"- rubric thresholds: `{config['activation_thresholds']}`",
                f"- class_i_reference classification: `{c1}`",
                f"- class_ii_reference classification: `{c2}`",
                "- Class-III / Class-IV are canonical slots not yet instantiated experimentally.",
                "- P4 rubric is provisional and will be refined later.",
                f"- does the current reference evidence classify cleanly into Class-I and Class-II? `{'yes' if clean else 'no'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "class_i_reference": {
            "p5_state": c1["p5_state"],
            "p6_drive_state": c1["p6_drive_state"],
            "p4_state": c1["p4_state"],
            "p4_like_signal": c1["p4_like_signal"],
            "p4_class_active": c1["p4_class_active"],
            "canonical_class_label": c1["canonical_class_label"],
        },
        "class_ii_reference": {
            "p5_state": c2["p5_state"],
            "p6_drive_state": c2["p6_drive_state"],
            "p4_state": c2["p4_state"],
            "p4_like_signal": c2["p4_like_signal"],
            "p4_class_active": c2["p4_class_active"],
            "canonical_class_label": c2["canonical_class_label"],
        },
        "artifact_root": str(artifact_root),
    }
