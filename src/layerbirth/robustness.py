"""LB-13 robustness pilot orchestration."""

from __future__ import annotations

import binascii
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import subprocess
import zlib
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import artifact_is_current, cache_reuse_allowed, computation_hash, implementation_fingerprint
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lenses import build_lens_family
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import pushforward_matrix
from .pilots import run_all_lb12_pilots
from .protocols import resolve_adaptive_tau, resolve_run_settings
from .sweep import apply_closure_strength_control
from .substrates import build_class_iii_candidate_family, build_substrate_family


ROBUST_FIELDS = [
    "family_name",
    "variant_name",
    "run_id",
    "closure_strength_lambda",
    "tau_protocol_name",
    "resolved_tau",
    "lens_name",
    "lift_name",
    "analysis_k",
    "closure_error",
    "objecthood_order",
    "staging_gap",
    "affinity",
    "holonomy",
    "window_score_to_next",
    "cache_status",
    "manifest_path",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    return computation_hash(payload, length)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


def _write_png(path: Path, image: np.ndarray) -> None:
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))
    comp = zlib.compress(raw, level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", binascii.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1]:
            image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _line_plot(path: Path, lambdas: list[float], series: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    plotted: list[float] = []
    for label, arr in series.items():
        n = min(len(lambdas), len(arr))
        if n == 0:
            continue
        xs = np.asarray(lambdas[:n], dtype=np.float64)
        ys = np.asarray(arr[:n], dtype=np.float64)
        plotted.extend(ys.tolist())
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=5.0, label=label)

    if not plotted:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    ymin, ymax = float(np.min(plotted)), float(np.max(plotted))
    if np.isclose(ymin, ymax):
        pad = max(1e-3, 0.15 * max(abs(ymin), 1.0))
        ax.set_ylim(ymin - pad, ymax + pad)

    stem = path.stem
    xlabel = r"Closure strength $\lambda$"
    ylabel = "Measured value"
    if "susceptibility" in stem:
        ylabel = "Susceptibility"
    elif "binder" in stem:
        ylabel = "Binder-like cumulant"
    elif "delta_mobj" in stem:
        ylabel = r"$\Delta M_{\mathrm{obj}}$"
    elif "affinity" in stem:
        ylabel = "Affinity"
    elif "shift" in stem:
        ylabel = "Boundary shift"

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    if len(series) <= 6:
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_variant_window(
    metrics_rows: list[dict[str, Any]],
    lambda_grid: list[float],
    threshold: float = 0.10,
) -> dict[str, Any]:
    rows = sorted(metrics_rows, key=lambda r: float(r["closure_strength_lambda"]))
    max_score = -1.0
    best_idx = None
    scores_to_next: dict[float, float | None] = {}
    for i in range(len(rows)):
        scores_to_next[float(rows[i]["closure_strength_lambda"])] = None
    for i in range(len(rows) - 1):
        a = rows[i]
        b = rows[i + 1]
        score = abs(observed_float(b["objecthood_order"]) - observed_float(a["objecthood_order"])) + abs(
            observed_float(b["closure_error"]) - observed_float(a["closure_error"])
        )
        scores_to_next[float(a["closure_strength_lambda"])] = score
        if score > max_score:
            max_score = score
            best_idx = i
    present = bool(max_score >= threshold and best_idx is not None)
    candidate = None if best_idx is None else [float(rows[best_idx]["closure_strength_lambda"]), float(rows[best_idx + 1]["closure_strength_lambda"])]
    return {
        "candidate_window": candidate if present else None,
        "max_window_score": max(0.0, float(max_score)),
        "window_present": present,
        "best_interval_index": best_idx if present else None,
        "scores_to_next": scores_to_next,
    }


def summarize_family_robustness(summary_rows: list[dict[str, Any]], baseline_variant_name: str) -> dict[str, Any]:
    by_name = {row["variant_name"]: row for row in summary_rows}
    baseline = by_name[baseline_variant_name]
    baseline_ok = baseline["window_present"]
    tau_ok = any(r["group"] == "tau" and r["window_stable"] for r in summary_rows)
    lens_ok = any(r["group"] == "lens" and r["window_stable"] for r in summary_rows)
    lift_ok = any(r["group"] == "lift" and r["window_stable"] for r in summary_rows)
    robust = bool(baseline_ok and tau_ok and lens_ok and lift_ok)
    diagnosis = (
        "robust enough to continue"
        if robust
        else "rule out for now / fragile under obvious perturbations"
    )
    return {
        "baseline_variant_name": baseline_variant_name,
        "baseline_window": baseline["candidate_window"],
        "tau_perturbation_stable": tau_ok,
        "lens_perturbation_stable": lens_ok,
        "lift_perturbation_stable": lift_ok,
        "robust_enough_to_continue": robust,
        "diagnosis": diagnosis,
    }


def write_robustness_plots(metrics_rows: list[dict[str, Any]], output_dir: Path, family_name: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in metrics_rows:
        by_variant.setdefault(row["variant_name"], []).append(row)
    lambdas = sorted({float(r["closure_strength_lambda"]) for r in metrics_rows})

    ce_mobj: dict[str, list[float]] = {}
    sg_aff: dict[str, list[float]] = {}
    tau_series: dict[str, list[float]] = {}
    for variant, rows in by_variant.items():
        ordered = sorted(rows, key=lambda r: float(r["closure_strength_lambda"]))
        ce_mobj[f"{variant}:CE"] = [observed_float(r["closure_error"]) for r in ordered]
        ce_mobj[f"{variant}:M_obj"] = [observed_float(r["objecthood_order"]) for r in ordered]
        sg_aff[f"{variant}:SG"] = [observed_float(r["staging_gap"]) for r in ordered]
        sg_aff[f"{variant}:Aff"] = [observed_float(r["affinity"]) for r in ordered]
        tau_series[variant] = [float(r["resolved_tau"]) for r in ordered]

    p1 = output_dir / "ce_mobj_vs_lambda_by_variant.png"
    p2 = output_dir / "sg_aff_vs_lambda_by_variant.png"
    p3 = output_dir / "resolved_tau_by_variant.png"
    _line_plot(p1, lambdas, ce_mobj)
    _line_plot(p2, lambdas, sg_aff)
    _line_plot(p3, lambdas, tau_series)
    return [str(p1), str(p2), str(p3)]


def _resolve_lens(spec: dict[str, Any], substrate: dict[str, Any], P: np.ndarray) -> tuple[np.ndarray, str]:
    name = spec["name"]
    if name == "manual_family_block_lens":
        return np.asarray(substrate["block_lens"], dtype=np.int64), name
    if name == "manual_partition_lens":
        lens, _ = build_lens_family("manual_partition_lens", labels=spec["labels"])
        return np.asarray(lens, dtype=np.int64), name
    kwargs = dict(spec.get("kwargs", {}))
    lens, _ = build_lens_family(name, P=P, **kwargs)
    return np.asarray(lens, dtype=np.int64), name


def _resolve_lift(spec: dict[str, Any], P: np.ndarray, lens: np.ndarray) -> tuple[np.ndarray, str]:
    name = spec["name"]
    k = int(np.max(lens)) + 1
    kwargs = dict(spec.get("kwargs", {}))
    if name in {"uniform_lift_family", "prototype_lift_family"}:
        u, _ = build_lift_family(name, f=lens, k=k, **kwargs)
        return np.asarray(u, dtype=np.float64), name
    if name == "stationary_within_fiber_lift":
        u, _ = build_lift_family(name, P=P, f=lens, k=k, **kwargs)
        return np.asarray(u, dtype=np.float64), name
    raise ValueError(f"Unknown lift: {name}")


def _git_code_version() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"git_commit": commit or "unknown", "git_dirty": bool(dirty), "implementation_sha256": implementation_fingerprint(), "cache_reuse_disabled": not cache_reuse_allowed(True)}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": False, "implementation_sha256": implementation_fingerprint(), "cache_reuse_disabled": not cache_reuse_allowed(True)}


def _write_manifest(root: Path, config_path: Path, metrics_path: Path, notes_path: Path, env_path: Path, experiment_id: str) -> Path:
    schema = load_schema(_repo_root() / "configs" / "result_bundle.schema.json")
    seeds = root / "seeds.json"
    seeds.write_text("[]\n", encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "bundle_id": experiment_id,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(),
        "config_snapshot_path": str(config_path.relative_to(root)),
        "seed_list_path": str(seeds.relative_to(root)),
        "metrics_table_path": str(metrics_path.relative_to(root)),
        "plots_dir_path": "plots",
        "notes_file_path": str(notes_path.relative_to(root)),
        "environment_snapshot_path": str(env_path.relative_to(root)),
    }
    validate_manifest(manifest, schema)
    out = root / "manifest.json"
    out.write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def run_robustness_scan(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        config = config_path_or_obj
    else:
        config = json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))
    if output_root is None:
        output_root = _repo_root() / "results" / "robustness"
    else:
        output_root = Path(output_root)
    family_name = config["family_name"]
    artifact_root = output_root / config["artifact_subdir"]
    (artifact_root / "runs").mkdir(parents=True, exist_ok=True)
    (artifact_root / "metrics").mkdir(parents=True, exist_ok=True)
    (artifact_root / "plots").mkdir(parents=True, exist_ok=True)
    (artifact_root / "notes").mkdir(parents=True, exist_ok=True)
    (artifact_root / "config").mkdir(parents=True, exist_ok=True)
    (artifact_root / "env").mkdir(parents=True, exist_ok=True)

    lambdas = [float(x) for x in config["lambda_grid"]]
    substrate = build_substrate_family(config["family"]["name"], **config["family"]["params"])
    p_base = np.asarray(substrate["P"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    for variant in config["variants"]:
        vname = variant["name"]
        vgroup = variant["group"]
        matched_ref_tau = None
        matched_ref_lambda = None
        matched_ref_run_id = None
        if variant["tau"]["mode"] == "matched":
            matched_ref_lambda = float(variant["tau"]["reference_lambda"])
            lens_ref, _ = _resolve_lens(variant["lens"], substrate, p_base)
            lift_ref, _ = _resolve_lift(variant["lift"], p_base, lens_ref)
            q_ref = pushforward_matrix(lens_ref, int(np.max(lens_ref)) + 1)
            p_ref = apply_closure_strength_control(
                p_base,
                closure_strength_lambda=matched_ref_lambda,
                Q_f=q_ref,
                U_f=lift_ref,
                mode=config["control_application_name"],
            )
            adaptive_cfg = {
                "run_id": f"{family_name}_{vname}_ref",
                "tau_protocol": {
                    "name": "adaptive",
                    **variant["tau"]["base_adaptive"],
                },
                "control": {"closure_strength_lambda": matched_ref_lambda},
            }
            resolved = resolve_adaptive_tau(adaptive_cfg, p_ref)
            matched_ref_tau = int(resolved["resolved_tau"])
            matched_ref_run_id = f"{vname}_lambda_{matched_ref_lambda}"

        for lam in lambdas:
            lens, lens_name = _resolve_lens(variant["lens"], substrate, p_base)
            lift, lift_name = _resolve_lift(variant["lift"], p_base, lens)
            q = pushforward_matrix(lens, int(np.max(lens)) + 1)
            p_controlled = apply_closure_strength_control(
                p_base,
                closure_strength_lambda=lam,
                Q_f=q,
                U_f=lift,
                mode=config["control_application_name"],
            )
            tau_mode = variant["tau"]["mode"]
            if tau_mode == "fixed":
                run_settings = resolve_run_settings(
                    {
                        "run_id": "rob_tmp",
                        "tau_protocol": {"name": "fixed", "tau": int(variant["tau"]["tau"])},
                        "control": {"closure_strength_lambda": lam},
                    },
                    P=p_controlled,
                )
            elif tau_mode == "adaptive":
                run_settings = resolve_run_settings(
                    {
                        "run_id": "rob_tmp",
                        "tau_protocol": {
                            "name": "adaptive",
                            "method": variant["tau"]["method"],
                            "alpha": variant["tau"]["alpha"],
                            "gap_floor": variant["tau"]["gap_floor"],
                            "tau_min": variant["tau"]["tau_min"],
                            "tau_max": variant["tau"]["tau_max"],
                        },
                        "control": {"closure_strength_lambda": lam},
                    },
                    P=p_controlled,
                )
            elif tau_mode == "matched":
                run_settings = {
                    "tau_protocol_name": "matched",
                    "resolved_tau": int(matched_ref_tau),
                    "tau_resolution_details": {
                        "matched_reference_lambda": matched_ref_lambda,
                        "matched_reference_run_id": matched_ref_run_id,
                        "matched_reference_tau": matched_ref_tau,
                    },
                }
            else:
                raise ValueError(f"Unknown tau mode: {tau_mode}")

            bundle = default_metric_bundle(p_controlled, lens, tau=int(run_settings["resolved_tau"]), lift_name=lift_name, U_f=lift)
            run_spec = {
                "family_config": config["family"],
                "variant_config": variant,
                "control_application_name": config["control_application_name"],
                "resolved_tau": int(run_settings["resolved_tau"]),
                "family_name": family_name,
                "variant_name": vname,
                "closure_strength_lambda": lam,
                "tau_mode": tau_mode,
                "lens_name": lens_name,
                "lift_name": lift_name,
            }
            run_id = f"rob_{_stable_hash(run_spec)}"
            run_root = artifact_root / "runs" / run_id
            row_path = run_root / "row.json"
            if cache_reuse_allowed(use_cache) and row_path.exists():
                cached = json.loads(row_path.read_text(encoding="utf-8"))
                cached["cache_status"] = "cached"
                rows.append(cached)
                continue
            run_root.mkdir(parents=True, exist_ok=True)
            run_manifest = run_root / "manifest.json"
            run_manifest.write_text(
                scientific_dumps(
                    {
                        "run_id": run_id,
                        "variant_name": vname,
                        "closure_strength_lambda": lam,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            row = {
                "family_name": family_name,
                "variant_name": vname,
                "run_id": run_id,
                "closure_strength_lambda": lam,
                "tau_protocol_name": run_settings["tau_protocol_name"],
                "resolved_tau": int(run_settings["resolved_tau"]),
                "lens_name": lens_name,
                "lift_name": lift_name,
                "analysis_k": int(bundle["analysis_k"]),
                "closure_error": observed_float(bundle["closure_error"]),
                "objecthood_order": observed_float(bundle["objecthood_order"]),
                "staging_gap": observed_float(bundle["staging_gap"]),
                "affinity": observed_float(bundle["affinity"]),
                "holonomy": bundle["holonomy"],
                "window_score_to_next": "",
                "cache_status": "executed",
                "manifest_path": str(run_manifest),
            }
            if tau_mode == "matched":
                row["matched_reference_lambda"] = matched_ref_lambda
                row["matched_reference_run_id"] = matched_ref_run_id
                row["matched_reference_tau"] = matched_ref_tau
            row_path.write_text(scientific_dumps(row, indent=2) + "\n", encoding="utf-8")
            rows.append(row)

        vrows = [r for r in rows if r["variant_name"] == vname]
        window = evaluate_variant_window(vrows, lambdas, threshold=0.10)
        for r in vrows:
            score = window["scores_to_next"].get(float(r["closure_strength_lambda"]))
            r["window_score_to_next"] = "" if score is None else score
        base_idx = None
        if config.get("baseline_window") is not None:
            b0, b1 = [float(x) for x in config["baseline_window"]]
            if b0 in lambdas and b1 in lambdas:
                base_idx = lambdas.index(b0)
        stable = bool(
            window["window_present"]
            and base_idx is not None
            and abs(int(window["best_interval_index"]) - int(base_idx)) <= 1
        )
        variant_summary = {
            "variant_name": vname,
            "group": vgroup,
            "candidate_window": window["candidate_window"],
            "max_window_score": window["max_window_score"],
            "window_present": window["window_present"],
            "window_stable": stable,
        }
        if tau_mode == "matched":
            variant_summary["matched_reference_lambda"] = matched_ref_lambda
            variant_summary["matched_reference_run_id"] = matched_ref_run_id
            variant_summary["matched_reference_tau"] = matched_ref_tau
        variant_summaries.append(variant_summary)

    rows.sort(key=lambda r: (r["variant_name"], float(r["closure_strength_lambda"])))
    metrics_path = artifact_root / "metrics" / "metrics.csv"
    _write_csv(metrics_path, rows, ROBUST_FIELDS)

    family_summary = summarize_family_robustness(variant_summaries, config["baseline_variant_name"])
    aff_min = min(observed_float(r["affinity"]) for r in rows)
    aff_max = max(observed_float(r["affinity"]) for r in rows)
    if family_name == "driven_low_bias":
        aff_diag = (
            "Aff stays clearly positive across variants"
            if aff_min > 1e-6
            else "Aff is sensitive; some variants approach non-driven levels"
        )
    else:
        aff_diag = "Aff remains near equilibrium scale for this family"
    summary = {
        "family_name": family_name,
        "baseline_variant_name": config["baseline_variant_name"],
        "baseline_window": config["baseline_window"],
        "variants": variant_summaries,
        "robust_enough_to_continue": family_summary["robust_enough_to_continue"],
        "diagnosis": family_summary["diagnosis"],
        "affinity_diagnosis": aff_diag,
    }
    summary_path = artifact_root / "summary.json"
    summary_path.write_text(scientific_dumps(summary, indent=2) + "\n", encoding="utf-8")

    plot_paths = write_robustness_plots(rows, artifact_root / "plots", family_name)

    notes_path = artifact_root / "notes" / "findings.md"
    notes_path.write_text(
        "\n".join(
            [
                "# Robustness findings",
                "",
                f"- family_name: `{family_name}`",
                f"- diagnosis: `{summary['diagnosis']}`",
                f"- affinity_diagnosis: `{summary['affinity_diagnosis']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg_path = artifact_root / "config" / "config_snapshot.json"
    cfg_path.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    env_path = artifact_root / "env" / "environment.json"
    env_path.write_text(
        scientific_dumps({"python": "unknown", "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        artifact_root,
        cfg_path,
        metrics_path,
        notes_path,
        env_path,
        f"robustness_{family_name}",
    )

    findings_note = _repo_root() / config["findings_note_path"]
    findings_note.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for v in variant_summaries:
        if v["window_present"]:
            lines.append(f"- `{v['variant_name']}`: candidate_window={v['candidate_window']} stable={v['window_stable']}")
        else:
            lines.append(f"- `{v['variant_name']}`: no_clear_transition_window stable={v['window_stable']}")
    findings_note.write_text(
        "\n".join(
            [
                f"# LB-13 robustness: {family_name}",
                "",
                f"- config path: `{config['config_path_hint']}`",
                f"- artifact root: `{artifact_root}`",
                f"- baseline window: `{config['baseline_window']}`",
                "- per-variant windows:",
                *lines,
                f"- family-level verdict: `{summary['diagnosis']}`",
                f"- affinity note: `{summary['affinity_diagnosis']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    executed_count = sum(1 for r in rows if r["cache_status"] == "executed")
    cached_count = sum(1 for r in rows if r["cache_status"] == "cached")
    return {
        "family_name": family_name,
        "variant_count": len(config["variants"]),
        "run_count": len(rows),
        "executed_count": executed_count,
        "cached_count": cached_count,
        "robust_enough_to_continue": summary["robust_enough_to_continue"],
        "artifact_root": str(artifact_root),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "plot_paths": plot_paths,
    }


def run_robustness_pilots(output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    root = _repo_root()
    run_all_lb12_pilots(output_root=root / "results" / "pilots", use_cache=True)
    if output_root is None:
        output_root = root / "results" / "robustness"
    else:
        output_root = Path(output_root)
    eq = run_robustness_scan(
        root / "configs" / "robustness" / "equilibrium_like_robustness.json",
        output_root=output_root,
        use_cache=use_cache,
    )
    driven = run_robustness_scan(
        root / "configs" / "robustness" / "driven_low_bias_robustness.json",
        output_root=output_root,
        use_cache=use_cache,
    )
    return {"families": [eq, driven]}


LENS_SUITE_FIELDS = [
    "class_name",
    "representative_name",
    "lens_name",
    "size",
    "p5_state",
    "p6_drive_state",
    "p4_like_signal",
    "p4_class_active",
    "canonical_class_label",
    "ce_boundary_tau1",
    "ce_boundary_tau2",
    "mobj_boundary_tau1",
    "mobj_boundary_tau2",
    "staging_shift_ce",
    "staging_shift_mobj",
    "affinity_ref",
    "affinity_ref_max",
    "cache_status",
    "manifest_path",
]


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(scientific_dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ensure_lens_suite_prereqs(root: Path, use_cache: bool) -> None:
    from .class3 import run_class_iii_strict_confirmation_v2, run_class_iv_full_campaign
    from .p4 import run_p4_anomaly_metric_layer
    from .taxonomy import run_canonical_class_rubric

    class3_root = root / "results" / "pilots" / "class_iii_strict_confirmation_v2"
    class4_root = root / "results" / "campaigns" / "class_iv_full_campaign"
    taxonomy_root = root / "results" / "taxonomy" / "canonical_class_rubric"
    p4_root = root / "results" / "metrics" / "p4_anomaly_metric_layer"
    if not artifact_is_current(class3_root, root / "configs/pilots/class_iii_strict_confirmation_v2.json"):
        run_class_iii_strict_confirmation_v2(
            root / "configs" / "pilots" / "class_iii_strict_confirmation_v2.json",
            output_root=root / "results" / "pilots",
            use_cache=use_cache,
        )
    if not artifact_is_current(class4_root, root / "configs/campaigns/class_iv_full_campaign.json"):
        run_class_iv_full_campaign(
            root / "configs" / "campaigns" / "class_iv_full_campaign.json",
            output_root=root / "results" / "campaigns",
            use_cache=use_cache,
        )
    if not artifact_is_current(taxonomy_root, root / "configs/taxonomy/canonical_class_rubric.json"):
        run_canonical_class_rubric(
            root / "configs" / "taxonomy" / "canonical_class_rubric.json",
            output_root=root / "results" / "taxonomy",
            use_cache=use_cache,
        )
    if not artifact_is_current(p4_root, root / "configs/metrics/p4_anomaly_metric_layer.json"):
        run_p4_anomaly_metric_layer(
            root / "configs" / "metrics" / "p4_anomaly_metric_layer.json",
            output_root=root / "results" / "metrics",
            use_cache=use_cache,
        )


def _resolve_class_iii_representative(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / str(cfg["class_iii"]["confirmation_summary_path"])
    pilot_cfg_path = root / str(cfg["class_iii"]["pilot_config_path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pilot_cfg = json.loads(pilot_cfg_path.read_text(encoding="utf-8"))
    best_variant = str(summary.get("overall", {}).get("best_variant", "")).strip()
    if not best_variant:
        best_variant = str(cfg["class_iii"].get("fallback_candidate_name", "replicated_portal_sp4"))
    pilot_candidates = {str(c["candidate_name"]): c for c in pilot_cfg.get("candidates", [])}
    selected = pilot_candidates.get(best_variant, pilot_candidates.get(str(cfg["class_iii"].get("fallback_candidate_name", "replicated_portal_sp4"))))
    if selected is None:
        raise ValueError("Unable to resolve Class-III representative from strict confirmation sources.")
    return {
        "class_name": "Class-III",
        "reference_profile_name": "class_iii_reference",
        "manual_reference_label": "Class-III",
        "representative_name": str(selected["candidate_name"]),
        "family_name": str(selected["family_name"]),
        "base_kwargs": dict(selected.get("base_kwargs", {})),
    }


def _resolve_class_iv_representative(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    campaign_cfg_path = root / str(cfg["class_iv"]["campaign_config_path"])
    campaign_cfg = json.loads(campaign_cfg_path.read_text(encoding="utf-8"))
    cand = dict(campaign_cfg["candidate"])
    return {
        "class_name": "Class-IV",
        "reference_profile_name": "class_iv_reference",
        "manual_reference_label": "Class-IV",
        "representative_name": str(cand["candidate_name"]),
        "family_name": str(cand["family_name"]),
        "base_kwargs": dict(cand.get("base_kwargs", {})),
        "drive_kwargs": dict(cand.get("drive_kwargs", {})),
    }


def _build_lens_for_substrate(substrate: dict[str, Any], lens_name: str) -> np.ndarray:
    if lens_name == "manual_family_coarse_lens":
        return np.asarray(substrate["coarse_lens"], dtype=np.int64)
    p = np.asarray(substrate["P"], dtype=np.float64)
    target_k = int(np.max(np.asarray(substrate["coarse_lens"], dtype=np.int64))) + 1
    lens, _ = build_lens_family(lens_name, P=p, target_k=target_k, tau=1)
    return np.asarray(lens, dtype=np.int64)


def _evaluate_tau_panel_with_lens(
    substrate: dict[str, Any],
    lens_name: str,
    lambdas: list[float],
    tau: int,
    control_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from .taxonomy import estimate_affinity_reference, estimate_structural_boundaries

    p_base = np.asarray(substrate["P"], dtype=np.float64)
    lens = _build_lens_for_substrate(substrate, lens_name)
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
        bundle = default_metric_bundle(p, lens, tau=int(tau))
        rows.append(
            {
                "closure_strength_lambda": float(lam),
                "closure_error": observed_float(bundle["closure_error"]),
                "objecthood_order": observed_float(bundle["objecthood_order"]),
                "staging_gap": observed_float(bundle["staging_gap"]),
                "affinity": observed_float(bundle["affinity"]),
                "analysis_k": int(bundle["analysis_k"]),
                "resolved_tau": int(tau),
            }
        )
    boundary = estimate_structural_boundaries(rows)
    affinity_ref = estimate_affinity_reference(rows, boundary)
    return rows, boundary, affinity_ref


def evaluate_lens_class_profile(rows: dict[str, Any], rubric_config: dict[str, Any], p4_config: dict[str, Any]) -> dict[str, Any]:
    from .class3 import evaluate_class_iii_candidate

    ev = evaluate_class_iii_candidate(rows, rubric_config, p4_config)
    return {
        **{k: ev[k] for k in ("p4_like_signal", "p4_class_active", "canonical_class_label", "ce_boundary_tau1", "ce_boundary_tau2", "mobj_boundary_tau1", "mobj_boundary_tau2", "staging_shift_ce", "staging_shift_mobj", "affinity_ref_max")},
        "p5_state": ev["candidate_p5_state"],
        "p6_drive_state": ev["candidate_p6_drive_state"],
        "affinity_ref": ev["affinity_ref_max"],
    }


def summarize_lens_suite(class_rows: list[dict[str, Any]], expected_sizes: list[int] | None = None) -> dict[str, Any]:
    manual_rows = [r for r in class_rows if r["lens_name"] == "manual_family_coarse_lens"]
    if not manual_rows:
        raise ValueError("Manual lens rows are required.")
    class_name = str(manual_rows[0]["class_name"])
    representative_name = str(manual_rows[0]["representative_name"])
    manual_label = str(manual_rows[0]["manual_reference_label"])
    keys = [(r["lens_name"], int(r["size"])) for r in class_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("lens suite requires one row per lens and size")
    if any(r["class_name"] != class_name or r["representative_name"] != representative_name for r in class_rows):
        raise ValueError("lens comparison requires one fixed representative")
    sizes = sorted(set(expected_sizes) if expected_sizes is not None else {int(r["size"]) for r in class_rows})
    manual_by_size = {int(r["size"]): r for r in manual_rows}
    reference_valid = bool(sizes) and all(s in manual_by_size and manual_by_size[s]["canonical_class_label"] == manual_label
                                       and manual_by_size[s]["p5_state"] in {"active", "inactive"}
                                       and manual_by_size[s]["p6_drive_state"] in {"active", "inactive"} for s in sizes)

    def _lens_status(lens_name: str) -> dict[str, Any]:
        rows = [r for r in class_rows if r["lens_name"] == lens_name]
        by_size = {int(r["size"]): r for r in rows}
        full_label = reference_valid and all(s in by_size and str(by_size[s]["canonical_class_label"]) == str(manual_by_size[s]["canonical_class_label"])
                                                and by_size[s].get("p4_class_active") == manual_by_size[s].get("p4_class_active") for s in sizes)
        structural_drive = reference_valid and all(
            int(s) in by_size
            and str(by_size[int(s)]["p5_state"]) == str(manual_by_size[s]["p5_state"])
            and str(by_size[int(s)]["p6_drive_state"]) == str(manual_by_size[s]["p6_drive_state"])
            for s in sizes
        )
        return {
            "lens_name": lens_name,
            "full_label_reproduced": bool(full_label),
            "structural_drive_signature_reproduced": bool(structural_drive),
            "labels_by_size": {str(s): str(by_size[s]["canonical_class_label"]) for s in sizes if s in by_size},
        }

    spectral = _lens_status("spectral_sign_pattern_lens")
    diffusion = _lens_status("diffusion_quantile_lens")
    if spectral["full_label_reproduced"] and diffusion["full_label_reproduced"]:
        status = "strong"
    elif (
        spectral["full_label_reproduced"] and diffusion["structural_drive_signature_reproduced"]
    ) or (
        diffusion["full_label_reproduced"] and spectral["structural_drive_signature_reproduced"]
    ):
        status = "partial"
    else:
        status = "fragile"
    robust = bool(status in {"strong", "partial"})
    return {
        "class_name": class_name,
        "representative_name": representative_name,
        "manual_class_label": manual_label,
        "manual_reference_verified": reference_valid,
        "declared_size_panel": sizes,
        "spectral_label_status": spectral,
        "diffusion_label_status": diffusion,
        "per_class_robustness_status": status,
        "reviewer_facing_lens_robust_for_paper": robust,
        "diagnosis": (
            "both non-manual lenses reproduce full class label"
            if status == "strong"
            else "one non-manual lens reproduces full label; the other preserves P5/P6 signature"
            if status == "partial"
            else "at least one non-manual lens fails full and structural/drive reproduction"
        ),
    }


def format_lens_suite_summary(class3_summary: dict[str, Any], class4_summary: dict[str, Any]) -> dict[str, Any]:
    class3_ok = bool(class3_summary["reviewer_facing_lens_robust_for_paper"])
    class4_ok = bool(class4_summary["reviewer_facing_lens_robust_for_paper"])
    supported = bool(class3_ok and class4_ok)
    return {
        "class_iii_lens_robust_for_paper": class3_ok,
        "class_iv_lens_robust_for_paper": class4_ok,
        "class_iii_status": str(class3_summary["per_class_robustness_status"]),
        "class_iv_status": str(class4_summary["per_class_robustness_status"]),
        "lens_robustness_suite_supported": supported,
        "diagnosis": (
            "both classes satisfy strong/partial robustness criterion"
            if supported
            else "one or both classes are fragile under non-manual lens swaps"
        ),
    }


def run_class_iii_class_iv_lens_suite(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg = config_path_or_obj if isinstance(config_path_or_obj, dict) else json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))
    root = _repo_root()
    _ensure_lens_suite_prereqs(root, use_cache=use_cache)
    out_root = (root / "results" / "robustness") if output_root is None else Path(output_root)
    artifact_root = out_root / str(cfg["artifact_subdir"])

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
    class3_ref = _resolve_class_iii_representative(cfg, root)
    class4_ref = _resolve_class_iv_representative(cfg, root)
    refs = [class3_ref, class4_ref]

    size_panel = [int(s) for s in cfg["size_panel"]]
    lambdas = [float(v) for v in cfg["lambda_grid"]]
    lens_names = [str(v["name"]) for v in cfg["lens_family_grid"]]
    control_mode = str(cfg["control_application_name"])

    metric_rows: list[dict[str, Any]] = []
    for ref in refs:
        for lens_name in lens_names:
            for size in size_panel:
                if ref["class_name"] == "Class-III":
                    substrate = build_class_iii_candidate_family(ref["family_name"], **{**dict(ref["base_kwargs"]), "n": int(size)})
                else:
                    from .class3 import _build_class_iv_candidate_substrate

                    substrate = _build_class_iv_candidate_substrate(
                        ref["family_name"],
                        dict(ref["base_kwargs"]),
                        dict(ref["drive_kwargs"]),
                        int(size),
                    )
                tau1_rows, tau1_boundary, tau1_aff = _evaluate_tau_panel_with_lens(substrate, lens_name, lambdas, tau=1, control_mode=control_mode)
                tau2_rows, tau2_boundary, tau2_aff = _evaluate_tau_panel_with_lens(substrate, lens_name, lambdas, tau=2, control_mode=control_mode)
                ev = evaluate_lens_class_profile(
                    {
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
                metric_rows.append(
                    {
                        "class_name": ref["class_name"],
                        "representative_name": ref["representative_name"],
                        "manual_reference_label": ref["manual_reference_label"],
                        "lens_name": lens_name,
                        "size": int(size),
                        **ev,
                        "cache_status": "executed",
                        "manifest_path": "manifest.json",
                    }
                )

    class3_rows = [r for r in metric_rows if r["class_name"] == "Class-III"]
    class4_rows = [r for r in metric_rows if r["class_name"] == "Class-IV"]
    class3_summary = summarize_lens_suite(class3_rows, expected_sizes=size_panel)
    class4_summary = summarize_lens_suite(class4_rows, expected_sizes=size_panel)
    suite_summary = format_lens_suite_summary(class3_summary, class4_summary)

    _write_csv(dirs["metrics"] / "metrics.csv", metric_rows, LENS_SUITE_FIELDS)
    _write_json(dirs["analysis"] / "class_iii_lens_summary.json", class3_summary)
    _write_json(dirs["analysis"] / "class_iv_lens_summary.json", class4_summary)
    _write_json(dirs["analysis"] / "lens_suite_summary.json", suite_summary)

    class_plot_specs = [
        ("Class-III", class3_rows, "class_iii_lens_comparison.png"),
        ("Class-IV", class4_rows, "class_iv_lens_comparison.png"),
    ]
    for _, rows, fname in class_plot_specs:
        sizes = sorted({int(r["size"]) for r in rows})
        by_lens = {ln: {int(r["size"]): r for r in rows if r["lens_name"] == ln} for ln in lens_names}
        series = {
            ln: [observed_float(by_lens[ln][s]["staging_shift_mobj"]) for s in sizes]
            for ln in lens_names
        }
        _line_plot(dirs["plots"] / fname, [float(s) for s in sizes], series)

    lens_x = [0.0, 1.0, 2.0]
    ordered = ["manual_family_coarse_lens", "spectral_sign_pattern_lens", "diffusion_quantile_lens"]
    p4_series = {
        "Class-III": [float(np.mean([observed_float(r["staging_shift_mobj"]) for r in class3_rows if r["lens_name"] == ln])) for ln in ordered],
        "Class-IV": [float(np.mean([observed_float(r["staging_shift_mobj"]) for r in class4_rows if r["lens_name"] == ln])) for ln in ordered],
    }
    _line_plot(dirs["plots"] / "p4_shift_by_lens.png", lens_x, p4_series)
    affinity_series = {
        "Class-III": [float(np.mean([observed_float(r["affinity_ref"]) for r in class3_rows if r["lens_name"] == ln])) for ln in ordered],
        "Class-IV": [float(np.mean([observed_float(r["affinity_ref"]) for r in class4_rows if r["lens_name"] == ln])) for ln in ordered],
    }
    _line_plot(dirs["plots"] / "affinity_by_lens.png", lens_x, affinity_series)

    _write_json(dirs["config"] / "config_snapshot.json", cfg)
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    _write_json(dirs["env"] / "environment.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "python": "python3"})
    (dirs["notes"] / "findings.md").write_text(
        "\n".join(
            [
                "# F-01 Class-III / Class-IV lens robustness suite",
                "",
                f"- Class-III status: `{class3_summary['per_class_robustness_status']}`",
                f"- Class-IV status: `{class4_summary['per_class_robustness_status']}`",
                f"- Overall verdict: `{suite_summary['lens_robustness_suite_supported']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/F-01_class_iii_class_iv_lens_suite.md"))
    note_path.write_text(
        "\n".join(
            [
                "# F-01 Class-III / Class-IV lens robustness suite",
                "",
                "- Config: `configs/robustness/class_iii_class_iv_lens_suite.json`",
                f"- Artifact root: `{artifact_root}`",
                f"- Class-III representative: `{class3_ref['representative_name']}`",
                f"- Class-IV representative: `{class4_ref['representative_name']}`",
                f"- Class-III robustness status: `{class3_summary['per_class_robustness_status']}`",
                f"- Class-IV robustness status: `{class4_summary['per_class_robustness_status']}`",
                f"- Overall suite verdict: `{suite_summary['lens_robustness_suite_supported']}`",
                "",
                f"**is Class-III lens-robust enough for the paper? {'yes' if class3_summary['reviewer_facing_lens_robust_for_paper'] else 'no'}**",
                f"**is Class-IV lens-robust enough for the paper? {'yes' if class4_summary['reviewer_facing_lens_robust_for_paper'] else 'no'}**",
                "- Caveat note: non-manual failures are interpreted as P4-channel fragility unless P5/P6 signature also collapses.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": str(cfg["suite_id"]),
        "bundle_id": str(cfg["suite_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    schema = load_schema(root / "configs" / "result_bundle.schema.json")
    validate_manifest(manifest, schema)
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "artifact_root": str(artifact_root),
        "class_iii_lens_robust_for_paper": bool(class3_summary["reviewer_facing_lens_robust_for_paper"]),
        "class_iv_lens_robust_for_paper": bool(class4_summary["reviewer_facing_lens_robust_for_paper"]),
        "class_iii_status": str(class3_summary["per_class_robustness_status"]),
        "class_iv_status": str(class4_summary["per_class_robustness_status"]),
        "lens_robustness_suite_supported": bool(suite_summary["lens_robustness_suite_supported"]),
        "class_iii_representative": str(class3_ref["representative_name"]),
        "class_iv_representative": str(class4_ref["representative_name"]),
    }
