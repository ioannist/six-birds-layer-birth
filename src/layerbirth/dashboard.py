"""S2-02 Class-I / Class-II consolidation dashboard."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import artifact_is_current
from .boundaries import first_threshold_lambda, interpolate_value, structural_reference_lambda

from .campaigns import _git_code_version, _line_plot, _repo_root, _write_csv
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest


DASHBOARD_FIELDS = [
    "profile_name",
    "canonical_class_label",
    "p5_state",
    "p6_drive_state",
    "p4_state",
    "ce_boundary_lambda",
    "mobj_boundary_lambda",
    "affinity_ref",
    "affinity_onset_1e3",
    "affinity_onset_1e2",
    "affinity_onset_5e2",
    "tau_ce_shift",
    "tau_mobj_shift",
    "lens_sensitivity_status",
    "no_fake_arrow_checks_passed",
]


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _overview_plot(path: Path, metrics_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = ["Class-I", "Class-II"]
    ce = [observed_float(metrics_rows[0]["ce_boundary_lambda"]), observed_float(metrics_rows[1]["ce_boundary_lambda"])]
    mo = [observed_float(metrics_rows[0]["mobj_boundary_lambda"]), observed_float(metrics_rows[1]["mobj_boundary_lambda"])]
    aff = [abs(observed_float(metrics_rows[0]["affinity_ref"])), abs(observed_float(metrics_rows[1]["affinity_ref"]))]
    x = np.arange(len(labels))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=180)
    ax0, ax1 = axes
    ax0.bar(x - width / 2, ce, width=width, label="CE boundary", color="#2c7fb8")
    ax0.bar(x + width / 2, mo, width=width, label=r"$M_{\mathrm{obj}}$ boundary", color="#41ab5d")
    ax0.set_xticks(x, labels)
    ax0.set_ylim(0.0, 1.0)
    ax0.set_title("Structural boundary proxies", fontsize=10)
    ax0.set_ylabel(r"Boundary location $\lambda$", fontsize=10)
    ax0.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax0.legend(frameon=False, fontsize=8)
    for bars in ax0.containers:
        ax0.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax1.bar(x, aff, width=0.5, color=["#7bccc4", "#d95f0e"])
    ax1.set_xticks(x, labels)
    ax1.set_yscale("log")
    ax1.set_title("Affinity at structural birth", fontsize=10)
    ax1.set_ylabel(r"$\mathrm{Aff}_{\mathrm{ref}}$", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    for i, v in enumerate(aff):
        ax1.text(i, v * 1.25, f"{v:.2e}", ha="center", va="bottom", fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _no_fake_arrow_plot(path: Path, no_fake_arrow: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    labels = ["Hidden schedule\npositives", "Phase-aware\nfalse positives", "Reversible\nfalse positives"]
    values = [
        float(no_fake_arrow["protocol_trap_hidden_schedule_driven_positive_count"]),
        float(no_fake_arrow["protocol_trap_phase_aware_false_positive_count"]),
        float(no_fake_arrow["reversible_false_positive_count"]),
    ]
    colors = ["#d95f0e", "#2c7fb8", "#636363"]

    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=180)
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.55)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Count")
    ax.set_title("No-fake-arrow control outcomes", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    for i, v in enumerate(values):
        ax.text(i, v + 0.03, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _interp_lambda_for_threshold(rows: list[dict[str, Any]], ykey: str, target: float, mode: str = "geq") -> float | None:
    return first_threshold_lambda(rows, ykey, target, mode)


def _interp_value_at_lambda(rows: list[dict[str, Any]], ykey: str, lam: float | None) -> float | None:
    return interpolate_value(rows, ykey, lam)


def _group_rows_for_size(group_csv_rows: list[dict[str, str]], size: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in group_csv_rows:
        if int(float(r["size"])) != int(size):
            continue
        out.append(
            {
                "size": int(float(r["size"])),
                "closure_strength_lambda": float(r["closure_strength_lambda"]),
                "closure_error": observed_float(r.get("closure_error_mean", r.get("closure_error"))),
                "objecthood_order": observed_float(r.get("order_mean", r.get("objecthood_order"))),
                "affinity": observed_float(r.get("affinity_mean", r.get("affinity"))),
            }
        )
    return sorted(out, key=lambda x: x["closure_strength_lambda"])


def load_consolidation_sources(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    src = cfg["sources"]
    return {
        "config": cfg,
        "taxonomy_bundle_root": root / src["taxonomy_bundle_root"],
        "class_i_bundle_root": root / src["class_i_bundle_root"],
        "class_ii_bundle_root": root / src["class_ii_bundle_root"],
        "alignment_bundle_root": root / src["alignment_bundle_root"],
        "robustness_class_i_root": root / src["robustness_class_i_root"],
        "robustness_class_ii_root": root / src["robustness_class_ii_root"],
        "arrow_controls_root": root / src["arrow_controls_root"],
    }


def extract_reference_classification(taxonomy_bundle: Path, profile_name: str) -> dict[str, Any]:
    rows = _read_json(taxonomy_bundle / "analysis" / "reference_classifications.json")["reference_classifications"]
    for r in rows:
        if r["profile_name"] == profile_name:
            return r
    raise KeyError(f"missing profile in taxonomy bundle: {profile_name}")


def estimate_affinity_onset_from_group_rows(group_rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for t in thresholds:
        out[str(t)] = _interp_lambda_for_threshold(group_rows, "affinity", float(t), mode="geq")
    return out


def summarize_structural_boundary_proxies(
    class_i_source: list[dict[str, Any]],
    class_ii_source: list[dict[str, Any]],
    taxonomy_bundle: Path,
    ce_target: float,
    mobj_target: float,
    class_i_profile_name: str,
    class_ii_profile_name: str,
) -> dict[str, Any]:
    c1 = extract_reference_classification(taxonomy_bundle, class_i_profile_name)
    c2 = extract_reference_classification(taxonomy_bundle, class_ii_profile_name)

    def _derive_if_missing(profile: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, float | None]:
        ce = profile.get("ce_boundary_lambda")
        mo = profile.get("mobj_boundary_lambda")
        if ce is None:
            ce = _interp_lambda_for_threshold(rows, "closure_error", ce_target, mode="leq")
        if mo is None:
            mo = _interp_lambda_for_threshold(rows, "objecthood_order", mobj_target, mode="geq")
        return {"ce_boundary_lambda": ce, "mobj_boundary_lambda": mo}

    b1 = _derive_if_missing(c1, class_i_source)
    b2 = _derive_if_missing(c2, class_ii_source)
    return {
        "class_i_reference": b1,
        "class_ii_reference": b2,
    }


def summarize_affinity_boundary_and_onset(
    class_i_source: list[dict[str, Any]],
    class_ii_source: list[dict[str, Any]],
    structural_summary: dict[str, Any],
    taxonomy_bundle: Path,
    class_i_profile_name: str,
    class_ii_profile_name: str,
    onset_thresholds: list[float],
    alignment_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    c1 = extract_reference_classification(taxonomy_bundle, class_i_profile_name)
    c2 = extract_reference_classification(taxonomy_bundle, class_ii_profile_name)
    c1_ce = structural_summary["class_i_reference"]["ce_boundary_lambda"]
    c2_ce = structural_summary["class_ii_reference"]["ce_boundary_lambda"]

    c1_aff = observed_float(c1.get("affinity_ref", float("nan")))
    c2_aff = observed_float(c2.get("affinity_ref", float("nan")))

    c1_on = estimate_affinity_onset_from_group_rows(class_i_source, onset_thresholds)
    c2_on = estimate_affinity_onset_from_group_rows(class_ii_source, onset_thresholds)
    out = {
        "class_i_reference": {
            "affinity_ref": c1_aff,
            "affinity_onsets": c1_on,
        },
        "class_ii_reference": {
            "affinity_ref": c2_aff,
            "affinity_onsets": c2_on,
        },
    }
    if alignment_source is not None:
        out["alignment_support"] = alignment_source
    return out


def summarize_tau_sensitivity(
    class_i_source: dict[str, Any],
    class_ii_source: dict[str, Any],
    taxonomy_bundle: Path,
    alignment_source: dict[str, Any],
    class_i_profile_name: str,
    class_ii_profile_name: str,
) -> dict[str, Any]:
    c1 = extract_reference_classification(taxonomy_bundle, class_i_profile_name)
    c2 = extract_reference_classification(taxonomy_bundle, class_ii_profile_name)
    v = alignment_source.get("verdicts", {})
    c1_note = "baseline class label unchanged; higher-order coupling not central"
    c2_note = "higher-order dynamics reveal extra coupling but do not overturn class label" if v.get("tau2_reveals_structural_bias_coupling", False) else "baseline class label unchanged; higher-order coupling not central"
    return {
        "class_i_reference": {
            "tau_ce_shift": c1.get("tau_ce_shift"),
            "tau_mobj_shift": c1.get("tau_mobj_shift"),
            "tau_sensitivity_note": c1_note,
        },
        "class_ii_reference": {
            "tau_ce_shift": c2.get("tau_ce_shift"),
            "tau_mobj_shift": c2.get("tau_mobj_shift"),
            "tau_sensitivity_note": c2_note,
        },
    }


def summarize_lens_sensitivity(robustness_class_i_root: Path, robustness_class_ii_root: Path) -> dict[str, Any]:
    c1 = _read_json(robustness_class_i_root / "summary.json")
    c2 = _read_json(robustness_class_ii_root / "summary.json")

    def _status(payload: dict[str, Any]) -> tuple[str, str]:
        variants = payload.get("variants", [])
        spectral = [v for v in variants if "spectral" in str(v.get("variant_name", ""))]
        diffusion = [v for v in variants if "diffusion" in str(v.get("variant_name", ""))]
        spectral_ok = bool(spectral) and all(bool(v.get("window_stable", False)) for v in spectral)
        diffusion_ok = bool(diffusion) and all(bool(v.get("window_stable", False)) for v in diffusion)
        if spectral_ok and diffusion_ok:
            return "stable", "stable under spectral/diffusion lens perturbations"
        if (spectral or diffusion):
            return "mixed", "mixed / fragile under spectral/diffusion lens perturbations"
        return "insufficient_data", "insufficient data"

    s1, n1 = _status(c1)
    s2, n2 = _status(c2)
    return {
        "class_i_reference": {"lens_sensitivity_status": s1, "lens_sensitivity_note": n1},
        "class_ii_reference": {"lens_sensitivity_status": s2, "lens_sensitivity_note": n2},
    }


def summarize_no_fake_arrow(arrow_controls_root: Path) -> dict[str, Any]:
    payload = _read_json(arrow_controls_root / "summary.json")
    rows = payload.get("rows", [])
    reversible_names = {"reversible_block_manual", "reversible_block_spectral", "null_flat_mixing_manual"}
    reversible_rows = [r for r in rows if str(r.get("case_name")) in reversible_names]
    reversible_false_positive_count = int(sum(1 for r in reversible_rows if bool(r.get("driven_candidate", False))))
    hidden_positive_count = int(
        sum(
            1
            for r in rows
            if str(r.get("case_name")) == "protocol_trap_hidden_schedule" and bool(r.get("driven_candidate", False))
        )
    )
    phase_rows = [r for r in rows if str(r.get("case_name", "")).startswith("protocol_trap_phase_aware")]
    phase_aware_false_positive_count = int(sum(1 for r in phase_rows if bool(r.get("driven_candidate", False))))
    return {
        "reversible_control_case_count": len(reversible_rows),
        "reversible_false_positive_count": reversible_false_positive_count,
        "protocol_trap_hidden_schedule_driven_positive_count": hidden_positive_count,
        "protocol_trap_phase_aware_false_positive_count": phase_aware_false_positive_count,
        "no_fake_arrow_checks_passed": bool(
            {str(r.get("case_name")) for r in reversible_rows} == reversible_names
            and {str(r.get("case_name")) for r in phase_rows} == {
                "protocol_trap_phase_aware_pair01", "protocol_trap_phase_aware_pair12", "protocol_trap_phase_aware_pair20"
            }
            and all(isinstance(r.get("driven_candidate"), bool) for r in reversible_rows + phase_rows)
            and reversible_false_positive_count == 0
            and phase_aware_false_positive_count == 0
            and hidden_positive_count >= 1
        ),
    }


def format_consolidation_summary(summary_rows: dict[str, Any], verdicts: dict[str, Any]) -> dict[str, Any]:
    out = dict(summary_rows)
    out.update(verdicts)
    return out


def build_class_consolidation_dashboard(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    loaded = load_consolidation_sources(config_path_or_obj)
    cfg = loaded["config"]
    root = _repo_root()

    if output_root is None:
        output_root = root / "results" / "dashboards"
    else:
        output_root = Path(output_root)
    artifact_root = output_root / str(cfg["artifact_subdir"])

    # restore taxonomy bundle if missing
    taxonomy_root = loaded["taxonomy_bundle_root"]
    if not artifact_is_current(taxonomy_root):
        from .taxonomy import run_canonical_class_rubric

        run_canonical_class_rubric(root / "configs" / "taxonomy" / "canonical_class_rubric.json", output_root=root / "results" / "taxonomy", use_cache=use_cache)

    existed = (artifact_root / "manifest.json").exists()
    cache_status = "cached" if existed and use_cache else "executed"

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

    rep_size = int(cfg["conventions"]["representative_size"])
    c1_group = _group_rows_for_size(
        _read_csv(loaded["class_i_bundle_root"] / "analysis" / "primary_group_summary.csv"),
        rep_size,
    )
    c2_group = _group_rows_for_size(
        _read_csv(loaded["class_ii_bundle_root"] / "analysis" / "primary_group_summary.csv"),
        rep_size,
    )
    alignment_summary = _read_json(loaded["alignment_bundle_root"] / "analysis" / "alignment_summary.json")

    class_i_profile_name = str(cfg["conventions"]["class_i_profile_name"])
    class_ii_profile_name = str(cfg["conventions"]["class_ii_profile_name"])
    ce_target = float(cfg["conventions"]["ce_target"])
    mobj_target = float(cfg["conventions"]["mobj_target"])
    aff_thresholds = [float(x) for x in cfg["conventions"]["affinity_onset_thresholds"]]

    structural = summarize_structural_boundary_proxies(
        c1_group,
        c2_group,
        taxonomy_root,
        ce_target,
        mobj_target,
        class_i_profile_name,
        class_ii_profile_name,
    )
    affinity = summarize_affinity_boundary_and_onset(
        c1_group,
        c2_group,
        structural,
        taxonomy_root,
        class_i_profile_name,
        class_ii_profile_name,
        aff_thresholds,
        alignment_source=alignment_summary,
    )
    tau = summarize_tau_sensitivity(
        {},
        {},
        taxonomy_root,
        alignment_summary,
        class_i_profile_name,
        class_ii_profile_name,
    )
    lens = summarize_lens_sensitivity(loaded["robustness_class_i_root"], loaded["robustness_class_ii_root"])
    no_fake_arrow = summarize_no_fake_arrow(loaded["arrow_controls_root"])

    c1_ref = extract_reference_classification(taxonomy_root, class_i_profile_name)
    c2_ref = extract_reference_classification(taxonomy_root, class_ii_profile_name)

    c1_aff = observed_float(affinity["class_i_reference"]["affinity_ref"])
    c2_aff = observed_float(affinity["class_ii_reference"]["affinity_ref"])
    ratio = float(abs(c2_aff) / max(abs(c1_aff), 1e-12))
    diff = float(c2_aff - c1_aff)

    rule = cfg["publishable_claim_rule"]
    good_labels = c1_ref["canonical_class_label"] == "Class-I" and c2_ref["canonical_class_label"] == "Class-II"
    good_p = (
        c1_ref["p5_state"] == "active"
        and c2_ref["p5_state"] == "active"
        and c1_ref["p6_drive_state"] == "inactive"
        and c2_ref["p6_drive_state"] == "active"
    )
    good_aff = ratio >= observed_float(rule["affinity_contrast_ratio_min"]) or diff >= observed_float(rule["affinity_contrast_difference_min"])
    good_arrow = bool(no_fake_arrow["no_fake_arrow_checks_passed"]) and int(no_fake_arrow["reversible_false_positive_count"]) <= int(rule["no_fake_arrow_false_positive_max"])

    publishable = bool(good_labels and good_p and good_aff and good_arrow)

    ref_summary = {
        "class_i_reference": {
            "profile_name": class_i_profile_name,
            "canonical_class_label": c1_ref["canonical_class_label"],
            "p5_state": c1_ref["p5_state"],
            "p6_drive_state": c1_ref["p6_drive_state"],
            "p4_state": c1_ref["p4_state"],
            "source_bundle_roots": {
                "taxonomy": str(loaded["taxonomy_bundle_root"]),
                "class_bundle": str(loaded["class_i_bundle_root"]),
            },
        },
        "class_ii_reference": {
            "profile_name": class_ii_profile_name,
            "canonical_class_label": c2_ref["canonical_class_label"],
            "p5_state": c2_ref["p5_state"],
            "p6_drive_state": c2_ref["p6_drive_state"],
            "p4_state": c2_ref["p4_state"],
            "source_bundle_roots": {
                "taxonomy": str(loaded["taxonomy_bundle_root"]),
                "class_bundle": str(loaded["class_ii_bundle_root"]),
            },
        },
    }

    tau_lens = {
        "tau": tau,
        "lens": lens,
    }

    consolidation_rows = {
        "class_i_reference": ref_summary["class_i_reference"],
        "class_ii_reference": ref_summary["class_ii_reference"],
        "affinity_contrast_ratio": ratio,
        "affinity_contrast_difference": diff,
        "no_fake_arrow_checks_passed": bool(no_fake_arrow["no_fake_arrow_checks_passed"]),
    }
    verdicts = {
        "publishable_non_exponent_claim_supported": publishable,
        "diagnosis": "Class-I vs Class-II non-exponent claim is supported" if publishable else "Consolidation thresholds not fully satisfied",
    }
    consolidation_summary = format_consolidation_summary(consolidation_rows, verdicts)

    (dirs["analysis"] / "reference_summary.json").write_text(scientific_dumps(ref_summary, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "structural_boundary_proxies.json").write_text(scientific_dumps(structural, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "affinity_summary.json").write_text(scientific_dumps(affinity, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "tau_lens_summary.json").write_text(scientific_dumps(tau_lens, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "no_fake_arrow_summary.json").write_text(scientific_dumps(no_fake_arrow, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "consolidation_summary.json").write_text(scientific_dumps(consolidation_summary, indent=2) + "\n", encoding="utf-8")

    metrics_rows = [
        {
            "profile_name": class_i_profile_name,
            "canonical_class_label": c1_ref["canonical_class_label"],
            "p5_state": c1_ref["p5_state"],
            "p6_drive_state": c1_ref["p6_drive_state"],
            "p4_state": c1_ref["p4_state"],
            "ce_boundary_lambda": structural["class_i_reference"]["ce_boundary_lambda"],
            "mobj_boundary_lambda": structural["class_i_reference"]["mobj_boundary_lambda"],
            "affinity_ref": affinity["class_i_reference"]["affinity_ref"],
            "affinity_onset_1e3": affinity["class_i_reference"]["affinity_onsets"][str(aff_thresholds[0])],
            "affinity_onset_1e2": affinity["class_i_reference"]["affinity_onsets"][str(aff_thresholds[1])],
            "affinity_onset_5e2": affinity["class_i_reference"]["affinity_onsets"][str(aff_thresholds[2])],
            "tau_ce_shift": tau["class_i_reference"]["tau_ce_shift"],
            "tau_mobj_shift": tau["class_i_reference"]["tau_mobj_shift"],
            "lens_sensitivity_status": lens["class_i_reference"]["lens_sensitivity_status"],
            "no_fake_arrow_checks_passed": bool(no_fake_arrow["no_fake_arrow_checks_passed"]),
            "cache_status": cache_status,
            "manifest_path": str(artifact_root / "manifest.json"),
        },
        {
            "profile_name": class_ii_profile_name,
            "canonical_class_label": c2_ref["canonical_class_label"],
            "p5_state": c2_ref["p5_state"],
            "p6_drive_state": c2_ref["p6_drive_state"],
            "p4_state": c2_ref["p4_state"],
            "ce_boundary_lambda": structural["class_ii_reference"]["ce_boundary_lambda"],
            "mobj_boundary_lambda": structural["class_ii_reference"]["mobj_boundary_lambda"],
            "affinity_ref": affinity["class_ii_reference"]["affinity_ref"],
            "affinity_onset_1e3": affinity["class_ii_reference"]["affinity_onsets"][str(aff_thresholds[0])],
            "affinity_onset_1e2": affinity["class_ii_reference"]["affinity_onsets"][str(aff_thresholds[1])],
            "affinity_onset_5e2": affinity["class_ii_reference"]["affinity_onsets"][str(aff_thresholds[2])],
            "tau_ce_shift": tau["class_ii_reference"]["tau_ce_shift"],
            "tau_mobj_shift": tau["class_ii_reference"]["tau_mobj_shift"],
            "lens_sensitivity_status": lens["class_ii_reference"]["lens_sensitivity_status"],
            "no_fake_arrow_checks_passed": bool(no_fake_arrow["no_fake_arrow_checks_passed"]),
            "cache_status": cache_status,
            "manifest_path": str(artifact_root / "manifest.json"),
        },
    ]
    _write_csv(dirs["metrics"] / "metrics.csv", metrics_rows, DASHBOARD_FIELDS)

    x = [0, 1]
    _overview_plot(dirs["plots"] / "class_i_vs_class_ii_overview.png", metrics_rows)
    _line_plot(
        dirs["plots"] / "structural_boundary_proxies.png",
        x,
        {
            "Class-I CE boundary": [observed_float(metrics_rows[0]["ce_boundary_lambda"])] * 2,
            "Class-I M_obj boundary": [observed_float(metrics_rows[0]["mobj_boundary_lambda"])] * 2,
            "Class-II CE boundary": [observed_float(metrics_rows[1]["ce_boundary_lambda"])] * 2,
            "Class-II M_obj boundary": [observed_float(metrics_rows[1]["mobj_boundary_lambda"])] * 2,
        },
    )
    _line_plot(
        dirs["plots"] / "affinity_boundary_onset.png",
        x,
        {
            "Class-I onset 1e-3": [observed_float(metrics_rows[0]["affinity_onset_1e3"])] * 2,
            "Class-II onset 1e-3": [observed_float(metrics_rows[1]["affinity_onset_1e3"])] * 2,
            "Class-II onset 1e-2": [observed_float(metrics_rows[1]["affinity_onset_1e2"])] * 2,
            "Class-II onset 5e-2": [observed_float(metrics_rows[1]["affinity_onset_5e2"])] * 2,
        },
    )
    _line_plot(
        dirs["plots"] / "tau_sensitivity.png",
        x,
        {
            "Class-I tau CE shift": [observed_float(metrics_rows[0]["tau_ce_shift"])] * 2,
            "Class-I tau M_obj shift": [observed_float(metrics_rows[0]["tau_mobj_shift"])] * 2,
            "Class-II tau CE shift": [observed_float(metrics_rows[1]["tau_ce_shift"])] * 2,
            "Class-II tau M_obj shift": [observed_float(metrics_rows[1]["tau_mobj_shift"])] * 2,
        },
    )
    _line_plot(
        dirs["plots"] / "lens_sensitivity.png",
        x,
        {
            "Class-I lens stable": [1.0 if lens["class_i_reference"]["lens_sensitivity_status"] == "stable" else 0.0] * 2,
            "Class-II lens stable": [1.0 if lens["class_ii_reference"]["lens_sensitivity_status"] == "stable" else 0.0] * 2,
        },
    )
    _no_fake_arrow_plot(dirs["plots"] / "no_fake_arrow_checks.png", no_fake_arrow)

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(
        scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )
    (dirs["notes"] / "findings.md").write_text(
        "\n".join(
            [
                "# Class-I / Class-II consolidation findings",
                "",
                f"- class_i_reference label: `{c1_ref['canonical_class_label']}`",
                f"- class_ii_reference label: `{c2_ref['canonical_class_label']}`",
                f"- affinity_contrast_ratio: `{ratio}`",
                f"- no_fake_arrow_checks_passed: `{no_fake_arrow['no_fake_arrow_checks_passed']}`",
                f"- publishable_non_exponent_claim_supported: `{publishable}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "class_i_class_ii_consolidation_dashboard",
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

    findings_note = root / str(cfg.get("findings_note_path", "notes/findings/S2-02_class_i_class_ii_consolidation.md"))
    findings_note.parent.mkdir(parents=True, exist_ok=True)
    findings_note.write_text(
        "\n".join(
            [
                "# S2-02 Class-I / Class-II consolidation",
                "",
                "- config path: `configs/dashboards/class_i_class_ii_consolidation.json`",
                f"- artifact root: `{artifact_root}`",
                f"- class-I canonical label: `{c1_ref['canonical_class_label']}`",
                f"- class-II canonical label: `{c2_ref['canonical_class_label']}`",
                f"- structural boundary comparison: CE `({metrics_rows[0]['ce_boundary_lambda']} vs {metrics_rows[1]['ce_boundary_lambda']})`, M_obj `({metrics_rows[0]['mobj_boundary_lambda']} vs {metrics_rows[1]['mobj_boundary_lambda']})`",
                f"- affinity-at-boundary / onset comparison: affinity_ref `({metrics_rows[0]['affinity_ref']} vs {metrics_rows[1]['affinity_ref']})`, onset 1e-3 `({metrics_rows[0]['affinity_onset_1e3']} vs {metrics_rows[1]['affinity_onset_1e3']})`",
                f"- tau / lens / no-fake-arrow support: tau shifts `({metrics_rows[0]['tau_ce_shift']},{metrics_rows[0]['tau_mobj_shift']})` and `({metrics_rows[1]['tau_ce_shift']},{metrics_rows[1]['tau_mobj_shift']})`; lens `({metrics_rows[0]['lens_sensitivity_status']},{metrics_rows[1]['lens_sensitivity_status']})`; no-fake-arrow `{no_fake_arrow['no_fake_arrow_checks_passed']}`",
                f"- does the current evidence support a publishable Class-I vs Class-II non-exponent claim? `{'yes' if publishable else 'no'}`",
                f"- is affinity the primary robust discriminant between the two confirmed classes? `{'yes' if good_aff else 'no'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "class_i_reference": {
            "canonical_class_label": c1_ref["canonical_class_label"],
            "p5_state": c1_ref["p5_state"],
            "p6_drive_state": c1_ref["p6_drive_state"],
            "p4_state": c1_ref["p4_state"],
        },
        "class_ii_reference": {
            "canonical_class_label": c2_ref["canonical_class_label"],
            "p5_state": c2_ref["p5_state"],
            "p6_drive_state": c2_ref["p6_drive_state"],
            "p4_state": c2_ref["p4_state"],
        },
        "affinity_contrast_ratio": ratio,
        "affinity_contrast_difference": diff,
        "no_fake_arrow_checks_passed": bool(no_fake_arrow["no_fake_arrow_checks_passed"]),
        "publishable_non_exponent_claim_supported": publishable,
        "artifact_root": str(artifact_root),
        "executed_count": 0 if cache_status == "cached" else 1,
        "cached_count": 1 if cache_status == "cached" else 0,
    }
