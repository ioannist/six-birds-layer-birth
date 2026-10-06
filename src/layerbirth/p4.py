"""S2-03 P4-anomaly metric layer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .boundaries import first_threshold_lambda, interpolate_value, structural_reference_lambda

from .campaigns import _git_code_version, _line_plot, _repo_root, _write_csv
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import pushforward_matrix
from .sweep import apply_closure_strength_control
from .substrates import build_substrate_family


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def _first_threshold_lambda(rows: list[dict[str, Any]], key: str, target: float, mode: str) -> float | None:
    return first_threshold_lambda(rows, key, target, mode)


def _build_curve_rows_for_profile(profile: dict[str, Any], tau: int) -> list[dict[str, Any]]:
    family = str(profile["family_name"])
    lam_grid = [float(x) for x in profile["lambda_grid"]]
    size = int(profile["size"])
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
        raise ValueError(f"unsupported family for p4 profile: {family}")

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


def compute_staging_shift(tau1_lambda: float | None, tau2_lambda: float | None) -> float:
    if tau1_lambda is None or tau2_lambda is None:
        return 0.0
    return float(abs(float(tau2_lambda) - float(tau1_lambda)))


def compute_presence_shift(tau1_present: bool, tau2_present: bool) -> bool:
    return bool(bool(tau1_present) != bool(tau2_present))


def compute_boundary_shift_summary(
    tau1_summary: dict[str, Any],
    tau2_summary: dict[str, Any],
    ce_target: float = 0.025,
    mobj_target: float = 0.90,
) -> dict[str, Any]:
    _ = ce_target, mobj_target
    ce1 = tau1_summary.get("ce_boundary_lambda")
    ce2 = tau2_summary.get("ce_boundary_lambda")
    mo1 = tau1_summary.get("mobj_boundary_lambda")
    mo2 = tau2_summary.get("mobj_boundary_lambda")
    ce_observed = all("ce_boundary_lambda" in summary for summary in (tau1_summary, tau2_summary))
    mo_observed = all("mobj_boundary_lambda" in summary for summary in (tau1_summary, tau2_summary))
    return {
        "tau1": tau1_summary,
        "tau2": tau2_summary,
        "staging_shift_ce": compute_staging_shift(ce1, ce2) if ce_observed else float("nan"),
        "staging_shift_mobj": compute_staging_shift(mo1, mo2) if mo_observed else float("nan"),
        "presence_shift_ce": ce_observed and compute_presence_shift(ce1 is not None, ce2 is not None),
        "presence_shift_mobj": mo_observed and compute_presence_shift(mo1 is not None, mo2 is not None),
        "presence_shift_any": (ce_observed and compute_presence_shift(ce1 is not None, ce2 is not None)) or (mo_observed and compute_presence_shift(mo1 is not None, mo2 is not None)),
        "boundary_measurements_available": bool(ce_observed and mo_observed),
    }


def compute_staging_gap_anomaly_score(boundary_shift_summary: dict[str, Any]) -> float:
    return float(
        max(
            observed_float(boundary_shift_summary.get("staging_shift_ce", 0.0)),
            observed_float(boundary_shift_summary.get("staging_shift_mobj", 0.0)),
        )
        + (1.0 if bool(boundary_shift_summary.get("presence_shift_any", False)) else 0.0)
    )


def evaluate_p4_like_signal(boundary_shift_summary: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    shift_min = float(thresholds["p4_like_shift_min"])
    return bool(
        bool(boundary_shift_summary.get("presence_shift_any", False))
        or max(
            observed_float(boundary_shift_summary.get("staging_shift_ce", 0.0)),
            observed_float(boundary_shift_summary.get("staging_shift_mobj", 0.0)),
        )
        >= shift_min
    )


def evaluate_p4_class_activation(boundary_shift_summary: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    dual_min = float(thresholds["p4_class_dual_shift_min"])
    if bool(boundary_shift_summary.get("presence_shift_any", False)):
        return True
    return bool(
        observed_float(boundary_shift_summary.get("staging_shift_ce", 0.0)) >= dual_min
        and observed_float(boundary_shift_summary.get("staging_shift_mobj", 0.0)) >= dual_min
    )


def evaluate_p4_profile(profile_rows_or_summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    summary = dict(profile_rows_or_summary)
    score = compute_staging_gap_anomaly_score(summary)
    like = evaluate_p4_like_signal(summary, thresholds)
    active = evaluate_p4_class_activation(summary, thresholds)
    complete = all(np.isfinite(observed_float(summary.get(key))) and observed_float(summary.get(key)) >= 0
                   for key in ("staging_shift_ce", "staging_shift_mobj"))
    state = "active" if active else "inactive" if complete else "unknown"
    return {
        "staging_shift_ce": observed_float(summary.get("staging_shift_ce")),
        "staging_shift_mobj": observed_float(summary.get("staging_shift_mobj")),
        "presence_shift_ce": bool(summary.get("presence_shift_ce", False)),
        "presence_shift_mobj": bool(summary.get("presence_shift_mobj", False)),
        "presence_shift_any": bool(summary.get("presence_shift_any", False)),
        "staging_gap_anomaly_score": float(score),
        "p4_like_signal": bool(like),
        "p4_class_active": bool(active),
        "p4_state": state,
    }


def _boundary_summary_from_curve(rows: list[dict[str, Any]], ce_target: float, mobj_target: float) -> dict[str, Any]:
    return {
        "ce_boundary_lambda": _first_threshold_lambda(rows, "closure_error", ce_target, mode="leq"),
        "mobj_boundary_lambda": _first_threshold_lambda(rows, "objecthood_order", mobj_target, mode="geq"),
    }


def _evaluate_reference_profile(profile: dict[str, Any], ce_target: float, mobj_target: float, thresholds: dict[str, Any]) -> dict[str, Any]:
    tau1_rows = _build_curve_rows_for_profile(profile, tau=1)
    tau2_rows = _build_curve_rows_for_profile(profile, tau=2)
    tau1 = _boundary_summary_from_curve(tau1_rows, ce_target, mobj_target)
    tau2 = _boundary_summary_from_curve(tau2_rows, ce_target, mobj_target)
    shift = compute_boundary_shift_summary(tau1, tau2, ce_target=ce_target, mobj_target=mobj_target)
    evald = evaluate_p4_profile(shift, thresholds)
    return {
        "profile_name": str(profile["profile_name"]),
        **evald,
        "tau1": tau1,
        "tau2": tau2,
        "canonical_p4_state": "active" if bool(evald["p4_class_active"]) else "inactive",
    }


def run_p4_anomaly_metric_layer(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    if output_root is None:
        output_root = root / "results" / "metrics"
    else:
        output_root = Path(output_root)
    artifact_root = output_root / str(cfg["artifact_subdir"])

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

    ce_target = float(cfg["boundary_targets"]["ce_target"])
    mobj_target = float(cfg["boundary_targets"]["mobj_target"])
    thresholds = cfg["thresholds"]

    synth_rows = []
    for case in cfg["synthetic_cases"]:
        shift = compute_boundary_shift_summary(case["tau1"], case["tau2"], ce_target=ce_target, mobj_target=mobj_target)
        evald = evaluate_p4_profile(shift, thresholds)
        synth_rows.append(
            {
                "profile_name": str(case["name"]),
                **evald,
                "canonical_p4_state": "active" if bool(evald["p4_class_active"]) else "inactive",
                "expected_p4_like_signal": bool(case["expected"]["p4_like_signal"]),
                "expected_p4_class_active": bool(case["expected"]["p4_class_active"]),
            }
        )

    ref_rows = [
        _evaluate_reference_profile(profile, ce_target, mobj_target, thresholds)
        for profile in cfg["reference_profiles"]
    ]

    # capture labels before refresh if taxonomy exists
    taxonomy_root = root / "results" / "taxonomy" / "canonical_class_rubric"
    before = {}
    if (taxonomy_root / "analysis" / "reference_classifications.json").exists():
        old = json.loads((taxonomy_root / "analysis" / "reference_classifications.json").read_text(encoding="utf-8"))["reference_classifications"]
        before = {r["profile_name"]: r.get("canonical_class_label") for r in old}

    from .taxonomy import run_canonical_class_rubric
    from .dashboard import build_class_consolidation_dashboard

    tax_summary = run_canonical_class_rubric(root / "configs" / "taxonomy" / "canonical_class_rubric.json", output_root=root / "results" / "taxonomy", use_cache=use_cache)
    dash_summary = build_class_consolidation_dashboard(root / "configs" / "dashboards" / "class_i_class_ii_consolidation.json", output_root=root / "results" / "dashboards", use_cache=use_cache)

    after = {
        "class_i_reference": tax_summary["class_i_reference"]["canonical_class_label"],
        "class_ii_reference": tax_summary["class_ii_reference"]["canonical_class_label"],
    }

    integration = {
        "class_i_reference": {
            "before": before.get("class_i_reference"),
            "after": after["class_i_reference"],
        },
        "class_ii_reference": {
            "before": before.get("class_ii_reference"),
            "after": after["class_ii_reference"],
        },
        "adhoc_override_in_use": False,
        "diagnosis": "taxonomy derives canonical P4 state from conservative p4_class_active; p4_like_signal is tracked separately",
    }

    (dirs["analysis"] / "synthetic_case_table.json").write_text(scientific_dumps({"synthetic_cases": synth_rows}, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "reference_p4_metrics.json").write_text(scientific_dumps({"references": ref_rows}, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "reference_p4_summary.json").write_text(
        scientific_dumps(
            {
                "class_i_reference": next(r for r in ref_rows if r["profile_name"] == "class_i_reference"),
                "class_ii_reference": next(r for r in ref_rows if r["profile_name"] == "class_ii_reference"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (dirs["analysis"] / "taxonomy_integration_check.json").write_text(scientific_dumps(integration, indent=2) + "\n", encoding="utf-8")

    metric_rows = [
        {
            "profile_name": r["profile_name"],
            "staging_shift_ce": r["staging_shift_ce"],
            "staging_shift_mobj": r["staging_shift_mobj"],
            "presence_shift_any": r["presence_shift_any"],
            "staging_gap_anomaly_score": r["staging_gap_anomaly_score"],
            "p4_like_signal": r["p4_like_signal"],
            "p4_class_active": r["p4_class_active"],
            "canonical_p4_state": r["canonical_p4_state"],
        }
        for r in (ref_rows + synth_rows)
    ]
    _write_csv(
        dirs["metrics"] / "metrics.csv",
        metric_rows,
        [
            "profile_name",
            "staging_shift_ce",
            "staging_shift_mobj",
            "presence_shift_any",
            "staging_gap_anomaly_score",
            "p4_like_signal",
            "p4_class_active",
            "canonical_p4_state",
        ],
    )

    x = [0, 1]
    _line_plot(
        dirs["plots"] / "p4_synthetic_cases.png",
        x,
        {r["profile_name"]: [observed_float(r["staging_gap_anomaly_score"])] * 2 for r in synth_rows},
    )
    _line_plot(
        dirs["plots"] / "reference_p4_shifts.png",
        x,
        {
            "class_i_reference:CE": [observed_float(next(r for r in ref_rows if r["profile_name"] == "class_i_reference")["staging_shift_ce"])] * 2,
            "class_i_reference:Mobj": [observed_float(next(r for r in ref_rows if r["profile_name"] == "class_i_reference")["staging_shift_mobj"])] * 2,
            "class_ii_reference:CE": [observed_float(next(r for r in ref_rows if r["profile_name"] == "class_ii_reference")["staging_shift_ce"])] * 2,
            "class_ii_reference:Mobj": [observed_float(next(r for r in ref_rows if r["profile_name"] == "class_ii_reference")["staging_shift_mobj"])] * 2,
        },
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(
        "\n".join(
            [
                "# S2-03 P4 anomaly metric layer",
                "",
                f"- class_i_reference: `{next(r for r in ref_rows if r['profile_name']=='class_i_reference')}`",
                f"- class_ii_reference: `{next(r for r in ref_rows if r['profile_name']=='class_ii_reference')}`",
                "- P4_like_signal is not equivalent to canonical P4 active.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "p4_anomaly_metric_layer",
        "bundle_id": str(cfg["metric_layer_id"]),
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

    note = root / str(cfg.get("findings_note_path", "notes/findings/S2-03_p4_anomaly_metric_layer.md"))
    note.parent.mkdir(parents=True, exist_ok=True)
    c1 = next(r for r in ref_rows if r["profile_name"] == "class_i_reference")
    c2 = next(r for r in ref_rows if r["profile_name"] == "class_ii_reference")
    note.write_text(
        "\n".join(
            [
                "# S2-03 P4 anomaly metric layer",
                "",
                "- config path: `configs/metrics/p4_anomaly_metric_layer.json`",
                f"- artifact root: `{artifact_root}`",
                f"- synthetic-case outcomes: `{synth_rows}`",
                f"- class_i_reference P4 metrics: `{c1}`",
                f"- class_ii_reference P4 metrics: `{c2}`",
                f"- taxonomy integration result: `{integration}`",
                f"- does the metric layer flag the already-observed manual_tau2 effect as a real P4-like signal? `{'yes' if c1['p4_like_signal'] or c2['p4_like_signal'] else 'no'}`",
                f"- does the taxonomy still classify the current references as Class-I and Class-II without ad hoc P4 override? `{'yes' if after['class_i_reference']=='Class-I' and after['class_ii_reference']=='Class-II' else 'no'}`",
                "- P4_like_signal is not the same as canonical P4_state=active.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "class_i_reference": {
            "p4_like_signal": c1["p4_like_signal"],
            "p4_class_active": c1["p4_class_active"],
            "canonical_class_label": after["class_i_reference"],
        },
        "class_ii_reference": {
            "p4_like_signal": c2["p4_like_signal"],
            "p4_class_active": c2["p4_class_active"],
            "canonical_class_label": after["class_ii_reference"],
        },
        "artifact_root": str(artifact_root),
        "taxonomy_artifact_root": str(root / "results" / "taxonomy" / "canonical_class_rubric"),
        "dashboard_artifact_root": str(root / "results" / "dashboards" / "class_i_class_ii_consolidation"),
        "dashboard_publishable_non_exponent_claim_supported": bool(dash_summary["publishable_non_exponent_claim_supported"]),
    }
