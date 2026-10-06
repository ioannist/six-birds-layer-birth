"""LB-17 structural-boundary vs affinity-onset alignment campaign."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import cache_reuse_allowed, computation_hash
from .boundaries import first_threshold_lambda, interpolate_value
from .campaigns import _git_code_version, _line_plot, _repo_root, _write_csv
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lenses import build_lens_family
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import pushforward_matrix
from .sweep import apply_closure_strength_control
from .substrates import build_substrate_family


METRIC_FIELDS = [
    "size",
    "bias",
    "analysis_mode",
    "closure_strength_lambda",
    "resolved_tau",
    "analysis_k",
    "closure_error",
    "objecthood_order",
    "staging_gap",
    "affinity",
    "cache_status",
    "manifest_path",
]


def _stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    return computation_hash(payload, length)


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def _tau_for_mode(mode: str) -> int:
    if mode == "manual_tau1":
        return 1
    if mode == "manual_tau2":
        return 2
    if mode == "diffusion_tau1":
        return 1
    raise ValueError(f"unsupported analysis mode: {mode}")


def _lens_for_mode(mode: str, p: np.ndarray) -> np.ndarray:
    n = p.shape[0]
    if mode in {"manual_tau1", "manual_tau2"}:
        return np.asarray([0 if i < (n // 2) else 1 for i in range(n)], dtype=np.int64)
    if mode == "diffusion_tau1":
        lens, _ = build_lens_family("diffusion_quantile_lens", P=p, target_k=2, tau=1)
        return np.asarray(lens, dtype=np.int64)
    raise ValueError(f"unsupported analysis mode: {mode}")


def _interpolate_crossings(
    curve_rows: list[dict[str, Any]],
    observable_key: str,
    targets: list[float],
) -> dict[str, float | None]:
    mode = "leq" if observable_key == "closure_error" else "geq"
    return {str(float(t)): first_threshold_lambda(curve_rows, observable_key, float(t), mode) for t in targets}


def _interp_value_at_lambda(curve_rows: list[dict[str, Any]], observable_key: str, lam: float) -> float | None:
    return interpolate_value(curve_rows, observable_key, lam)


def estimate_structural_boundaries(
    rows: list[dict[str, Any]],
    ce_targets: list[float],
    mobj_targets: list[float],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    grouped: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(int(r["size"]), float(r["bias"]), str(r["analysis_mode"]))].append(r)
    for (size, bias, mode), vals in grouped.items():
        key = f"{size}|{bias}|{mode}"
        ce_cross = _interpolate_crossings(vals, "closure_error", ce_targets)
        m_cross = _interpolate_crossings(vals, "objecthood_order", mobj_targets)
        out[key] = {
            "size": int(size),
            "bias": float(bias),
            "analysis_mode": mode,
            "ce_crossings": ce_cross,
            "mobj_crossings": m_cross,
        }
    return out


def estimate_affinity_onsets(rows: list[dict[str, Any]], affinity_targets: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    grouped: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(int(r["size"]), float(r["bias"]), str(r["analysis_mode"]))].append(r)
    for (size, bias, mode), vals in grouped.items():
        key = f"{size}|{bias}|{mode}"
        aff_cross = _interpolate_crossings(vals, "affinity", affinity_targets)
        out[key] = {
            "size": int(size),
            "bias": float(bias),
            "analysis_mode": mode,
            "affinity_onsets": aff_cross,
        }
    return out


def summarize_boundary_alignment(
    structural_rows: dict[str, Any],
    affinity_rows: dict[str, Any],
) -> dict[str, Any]:
    sizes = sorted({int(v["size"]) for v in structural_rows.values()})
    modes = sorted({str(v["analysis_mode"]) for v in structural_rows.values()})
    biases = sorted({float(v["bias"]) for v in structural_rows.values()})
    ce_ranges, m_ranges = {}, {}
    for size in sizes:
        ce_ranges[size] = _boundary_range(structural_rows, size, biases, "manual_tau1", "ce_crossings", "0.025")
        m_ranges[size] = _boundary_range(structural_rows, size, biases, "manual_tau1", "mobj_crossings", "0.9")
    complete = bool(sizes) and len(biases) >= 2 and all(np.isfinite(v) for v in list(ce_ranges.values()) + list(m_ranges.values()))
    invariant = complete and all(v <= .02 for v in list(ce_ranges.values()) + list(m_ranges.values()))
    onset_count = sum(any(affinity_rows.get(f"{size}|{b}|manual_tau1", {}).get("affinity_onsets", {}).get("0.001") is not None
                          for size in sizes) for b in biases if b != 0)
    return {
        "sizes": sizes, "biases": biases, "modes": modes,
        "ce_boundary_ranges_manual_tau1": ce_ranges,
        "mobj_boundary_ranges_manual_tau1": m_ranges,
        "fallbacks": [],
        "boundary_targets": {"closure_error": .025, "objecthood_order": .9},
        "baseline_panel_comparable": complete,
        "baseline_structural_boundary_bias_invariant": bool(invariant),
        "nonzero_bias_affinity_onset_count_manual_tau1": onset_count,
        "affinity_at_boundary_by_bias_manual_tau1": {b: float("nan") for b in biases},
        "mean_affinity_by_bias_manual_tau1": {b: float("nan") for b in biases},
        "boundary_interpretation": "first threshold proxies; first sampled hits are left censored",
    }


def _boundary_range(structural: dict[str, Any], size: int, biases: list[float], mode: str, field: str, target: str) -> float:
    values = [structural.get(f"{size}|{b}|{mode}", {}).get(field, {}).get(target) for b in biases]
    if len(values) < 2 or any(v is None or not np.isfinite(v) for v in values):
        return float("nan")
    return float(max(values) - min(values))


def compare_tau_and_lens_dependence(summary_rows: dict[str, Any], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from .taxonomy import estimate_structural_boundaries as joint_boundaries, estimate_affinity_reference
    sizes, biases = summary_rows["sizes"], summary_rows["biases"]
    structural = summary_rows["structural_rows"]

    def means(mode: str) -> dict[float, float]:
        values = {}
        baseline_coordinates = None
        for b in biases:
            rows = [r for r in raw_rows if r["analysis_mode"] == mode and float(r["bias"]) == b]
            coords = {(int(r["size"]), float(r["closure_strength_lambda"])) for r in rows}
            if baseline_coordinates is None:
                baseline_coordinates = coords
            comparable = bool(rows) and len(coords) == len(rows) and coords == baseline_coordinates
            affinity = [observed_float(r["affinity"]) for r in rows]
            values[b] = float(np.mean(affinity)) if comparable and all(np.isfinite(v) and v >= 0 for v in affinity) else float("nan")
        return values

    def monotonic(values: dict[float, float]) -> bool:
        return len(biases) >= 2 and all(np.isfinite(values[b]) and values[b] >= 0 for b in biases) and all(
            values[b1] + 1e-12 >= values[b0] for b0, b1 in zip(biases, biases[1:]))

    mean_manual = means("manual_tau1")
    at_boundary = {}
    for b in biases:
        values = []
        for size in sizes:
            curve = [r for r in raw_rows if int(r["size"]) == size and float(r["bias"]) == b and r["analysis_mode"] == "manual_tau1"]
            if not curve:
                values.append(float("nan"))
                continue
            boundary = joint_boundaries(curve, ce_target=.025, mobj_target=.9)
            affinity = estimate_affinity_reference(curve, boundary)["affinity_ref"] if boundary["structural_targets_met_at_reference"] else float("nan")
            values.append(affinity)
        at_boundary[b] = float(np.mean(values)) if values and all(np.isfinite(v) and v >= 0 for v in values) else float("nan")
    activation = bool(summary_rows["nonzero_bias_affinity_onset_count_manual_tau1"] >= 2
                      and 0.0 in biases and monotonic(at_boundary))

    tau2_coupling = False
    diffusion_ranges = []
    for size in sizes:
        for field, target in (("ce_crossings", "0.025"), ("mobj_crossings", "0.9")):
            r1 = _boundary_range(structural, size, biases, "manual_tau1", field, target)
            r2 = _boundary_range(structural, size, biases, "manual_tau2", field, target)
            if np.isfinite(r1) and np.isfinite(r2) and r2 - r1 >= .02:
                tau2_coupling = True
            diffusion_ranges.append(_boundary_range(structural, size, biases, "diffusion_tau1", field, target))
    diffusion_invariant = bool(diffusion_ranges) and all(np.isfinite(v) and v <= .02 for v in diffusion_ranges)
    lens_robust = bool(summary_rows["baseline_structural_boundary_bias_invariant"] and activation
                       and diffusion_invariant and monotonic(means("diffusion_tau1")))
    if summary_rows["baseline_structural_boundary_bias_invariant"] and activation:
        verdict = "higher_order_dynamics_only" if tau2_coupling else "same_structural_boundary_secondary_affinity_supported"
    else:
        verdict = "alignment_not_yet_supported"
    return {
        "baseline_structural_boundary_bias_invariant": bool(summary_rows["baseline_structural_boundary_bias_invariant"]),
        "affinity_secondary_activation_supported": activation,
        "tau2_reveals_structural_bias_coupling": tau2_coupling,
        "baseline_claim_lens_robust": lens_robust,
        "final_framing_verdict": verdict,
        "mean_affinity_by_bias_manual_tau1": mean_manual,
        "affinity_at_structural_boundary_by_bias_manual_tau1": at_boundary,
        "affinity_reference_rule": "joint CE0.025/Mobj0.9 reference in tau1; no target fallback",
    }


def format_alignment_summary(alignment_summary: dict[str, Any], verdicts: dict[str, Any]) -> dict[str, Any]:
    return {
        "alignment_summary": alignment_summary,
        "verdicts": verdicts,
    }


def run_structural_affinity_alignment(
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
                for lam in [float(x) for x in panel["lambda_grid"]]:
                    run_id = f"ali_{_stable_hash({'panel_config': panel, 'size': size, 'bias': bias, 'mode': mode_name, 'lambda': lam})}"
                    row_path = dirs["runs"] / f"{run_id}.json"
                    if cache_reuse_allowed(use_cache) and row_path.exists():
                        row = json.loads(row_path.read_text(encoding="utf-8"))
                        row["cache_status"] = "cached"
                        cached += 1
                    else:
                        lens = _lens_for_mode(mode_name, p_base)
                        k = int(np.max(lens)) + 1
                        q = pushforward_matrix(lens, k)
                        u = np.asarray(build_lift_family("uniform_lift_family", f=lens, k=k)[0], dtype=np.float64)
                        p = apply_closure_strength_control(
                            p_base,
                            closure_strength_lambda=lam,
                            Q_f=q,
                            U_f=u,
                            mode=str(panel["control_application_name"]),
                        )
                        bundle = default_metric_bundle(p, lens, tau=tau)
                        row = {
                            "size": int(size),
                            "bias": float(bias),
                            "analysis_mode": mode_name,
                            "closure_strength_lambda": float(lam),
                            "resolved_tau": int(tau),
                            "analysis_k": int(bundle["analysis_k"]),
                            "closure_error": observed_float(bundle["closure_error"]),
                            "objecthood_order": observed_float(bundle["objecthood_order"]),
                            "staging_gap": observed_float(bundle["staging_gap"]),
                            "affinity": observed_float(bundle["affinity"]),
                            "cache_status": "executed",
                            "manifest_path": "",
                        }
                        row_path.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
                        executed += 1
                    rows.append(row)

    structural = estimate_structural_boundaries(
        rows,
        ce_targets=[0.05, 0.025, 0.01],
        mobj_targets=[0.85, 0.90, 0.95],
    )
    onsets = estimate_affinity_onsets(rows, affinity_targets=[1e-3, 1e-2, 5e-2])
    base_summary = summarize_boundary_alignment(structural, onsets)
    base_summary["structural_rows"] = structural
    base_summary["affinity_rows"] = onsets
    verdicts = compare_tau_and_lens_dependence(base_summary, rows)
    final_summary = format_alignment_summary(
        {
            "structural_boundaries": structural,
            "affinity_onsets": onsets,
            **{k: v for k, v in base_summary.items() if k not in {"structural_rows", "affinity_rows"}},
        },
        verdicts,
    )

    for row in rows:
        row["manifest_path"] = str((artifact_root / "manifest.json").relative_to(artifact_root))
    _write_csv(dirs["metrics"] / "metrics.csv", rows, METRIC_FIELDS)
    _write_csv(dirs["analysis"] / "raw_metrics.csv", rows, METRIC_FIELDS)
    (dirs["analysis"] / "structural_boundaries.json").write_text(
        scientific_dumps(structural, indent=2) + "\n", encoding="utf-8"
    )
    (dirs["analysis"] / "affinity_onsets.json").write_text(
        scientific_dumps(onsets, indent=2) + "\n", encoding="utf-8"
    )
    (dirs["analysis"] / "alignment_summary.json").write_text(
        scientific_dumps(final_summary, indent=2) + "\n", encoding="utf-8"
    )

    rep_size = int(config["plot_representative_size"])
    biases = sorted({float(r["bias"]) for r in rows if int(r["size"]) == rep_size})
    for mode in ("manual_tau1", "manual_tau2"):
        ce_series = {}
        mo_series = {}
        for size in sorted({int(r["size"]) for r in rows}):
            ce_vals = []
            mo_vals = []
            for b in biases:
                key = f"{size}|{b}|{mode}"
                ce = structural[key]["ce_crossings"]["0.025"]
                mo = structural[key]["mobj_crossings"]["0.9"]
                ce_vals.append(float("nan") if ce is None else float(ce))
                mo_vals.append(float("nan") if mo is None else float(mo))
            ce_series[f"L{size}:CE0.025"] = ce_vals
            mo_series[f"L{size}:M0.90"] = mo_vals
        merged = {**ce_series, **mo_series}
        _line_plot(dirs["plots"] / f"structural_boundaries_vs_bias_{mode}.png", biases, merged)

    onset_series = {}
    for size in sorted({int(r["size"]) for r in rows}):
        vals = []
        for b in biases:
            key = f"{size}|{b}|manual_tau1"
            v = onsets[key]["affinity_onsets"]["0.001"]
            vals.append(float("nan") if v is None else float(v))
        onset_series[f"L{size}:Aff1e-3"] = vals
    _line_plot(dirs["plots"] / "affinity_onsets_vs_bias.png", biases, onset_series)

    _line_plot(
        dirs["plots"] / "affinity_at_structural_boundary_vs_bias.png",
        biases,
        {"manual_tau1": [float(verdicts["affinity_at_structural_boundary_by_bias_manual_tau1"][b]) for b in biases]},
    )

    diff_series = {
        "manual_tau1_aff": [float(verdicts["mean_affinity_by_bias_manual_tau1"][b]) for b in biases],
    }
    # add diffusion means
    diff_mean = {}
    for b in biases:
        vals = [
            observed_float(r["affinity"])
            for r in rows
            if int(r["size"]) == rep_size
            and str(r["analysis_mode"]) == "diffusion_tau1"
            and float(r["bias"]) == b
        ]
        diff_mean[b] = float(np.mean(vals)) if vals else float("nan")
    diff_series["diffusion_tau1_aff"] = [diff_mean[b] for b in biases]
    _line_plot(dirs["plots"] / "manual_vs_diffusion_comparison.png", biases, diff_series)

    config_snapshot = dirs["config"] / "config_snapshot.json"
    config_snapshot.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    notes = dirs["notes"] / "findings.md"
    notes.write_text(
        "\n".join(
            [
                "# Structural-affinity alignment findings",
                "",
                f"- baseline_structural_boundary_bias_invariant: `{verdicts['baseline_structural_boundary_bias_invariant']}`",
                f"- affinity_secondary_activation_supported: `{verdicts['affinity_secondary_activation_supported']}`",
                f"- tau2_reveals_structural_bias_coupling: `{verdicts['tau2_reveals_structural_bias_coupling']}`",
                f"- baseline_claim_lens_robust: `{verdicts['baseline_claim_lens_robust']}`",
                f"- final_framing_verdict: `{verdicts['final_framing_verdict']}`",
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
        "experiment_id": "structural_affinity_alignment",
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
                "# LB-17 structural-affinity alignment",
                "",
                f"- config path: `{config['config_path_hint']}`",
                f"- artifact root: `{artifact_root}`",
                f"- baseline structural-boundary results: `{base_summary['ce_boundary_ranges_manual_tau1']}`, `{base_summary['mobj_boundary_ranges_manual_tau1']}`",
                f"- affinity-onset results: `{onsets}`",
                f"- tau2 comparison: `tau2_reveals_structural_bias_coupling={verdicts['tau2_reveals_structural_bias_coupling']}`",
                f"- lens comparison: `baseline_claim_lens_robust={verdicts['baseline_claim_lens_robust']}`",
                f"- does current evidence support same structural boundary + secondary affinity activation? `{'yes' if verdicts['final_framing_verdict'] in {'same_structural_boundary_secondary_affinity_supported','higher_order_dynamics_only'} else 'no'}`",
                f"- does tau=2 materially change that interpretation? `{'yes' if verdicts['tau2_reveals_structural_bias_coupling'] else 'no'}`",
                f"- paper framing: foreground non-exponent story strengthened = `{verdicts['final_framing_verdict'] != 'alignment_not_yet_supported'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "baseline_structural_boundary_bias_invariant": bool(verdicts["baseline_structural_boundary_bias_invariant"]),
        "affinity_secondary_activation_supported": bool(verdicts["affinity_secondary_activation_supported"]),
        "tau2_reveals_structural_bias_coupling": bool(verdicts["tau2_reveals_structural_bias_coupling"]),
        "baseline_claim_lens_robust": bool(verdicts["baseline_claim_lens_robust"]),
        "final_framing_verdict": str(verdicts["final_framing_verdict"]),
        "artifact_root": str(artifact_root),
        "executed_count": int(executed),
        "cached_count": int(cached),
    }
