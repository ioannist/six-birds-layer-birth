"""F-05 upstream/PICA bridge campaign."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import bundle_is_current, cache_reuse_allowed
from .campaigns import _git_code_version
from .class3 import _evaluate_tau_panel, _repo_root, _write_csv, evaluate_class_iii_candidate
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .observable_map import compute_p4_dual_shift_min, compute_structural_boundary_lambda
from .robustness import _write_png
from .numeric import row_normalize, validate_row_stochastic
from .boundaries import first_threshold_lambda
from .taxonomy import classify_activation_signature


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(scientific_dumps(payload, indent=2) + "\n", encoding="utf-8")


def discover_upstream_bridge_candidates(config_path_or_obj: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    cfg = config_path_or_obj if isinstance(config_path_or_obj, dict) else json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))
    root = _repo_root()
    candidates = [dict(c) for c in cfg.get("candidates", [])][: int(cfg.get("max_candidates", 3))]
    return cfg, root, candidates


def _ensure_prereq_roots(root: Path, use_cache: bool) -> None:
    from .observable_map import build_four_class_observable_map
    from .p4 import run_p4_anomaly_metric_layer
    from .taxonomy import run_canonical_class_rubric

    req = [
        root / "results/dashboards/four_class_observable_map",
        root / "results/taxonomy/canonical_class_rubric",
        root / "results/metrics/p4_anomaly_metric_layer",
    ]
    if not cache_reuse_allowed(use_cache) or not bundle_is_current(req[0], json.loads((root / "configs/dashboards/four_class_observable_map.json").read_text())):
        build_four_class_observable_map(
            root / "configs/dashboards/four_class_observable_map.json",
            output_root=root / "results/dashboards",
            use_cache=use_cache,
        )
    if not cache_reuse_allowed(use_cache) or not bundle_is_current(req[1], json.loads((root / "configs/taxonomy/canonical_class_rubric.json").read_text())):
        run_canonical_class_rubric(
            root / "configs/taxonomy/canonical_class_rubric.json",
            output_root=root / "results/taxonomy",
            use_cache=use_cache,
        )
    if not cache_reuse_allowed(use_cache) or not bundle_is_current(req[2], json.loads((root / "configs/metrics/p4_anomaly_metric_layer.json").read_text())):
        run_p4_anomaly_metric_layer(
            root / "configs/metrics/p4_anomaly_metric_layer.json",
            output_root=root / "results/metrics",
            use_cache=use_cache,
        )


def _upstream_import_path(root: Path, cfg: dict[str, Any]) -> Path:
    return root / str(cfg["upstream_source_root"]) / "src"


def _kernel_from_upstream_substrate(substrate: Any, drive_mix: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    # Construct a feature-derived kernel; this is not native upstream dynamics.
    A = np.asarray(substrate.rule_cp_applicability, dtype=np.float64)  # rule x cp
    C = np.asarray(substrate.cell_cp_legality, dtype=np.float64)  # cell x cp
    H = np.asarray(substrate.cp_class_onehot, dtype=np.float64)  # cp x class
    B = np.asarray(substrate.cp_branch_preference_hint, dtype=np.float64)  # cp x 2
    sectors = np.asarray(substrate.cp_sector_index)
    if not np.isfinite(sectors).all() or not np.equal(sectors, np.floor(sectors)).all():
        raise ValueError("sector labels must be finite integers")
    cp_sector = np.asarray(sectors, dtype=np.int64)

    if A.ndim != 2 or C.ndim != 2 or H.ndim != 2 or B.ndim != 2 or cp_sector.ndim != 1:
        raise ValueError("upstream feature matrices must have their declared dimensions")
    n = int(A.shape[1])
    if C.shape[1] != n or H.shape[0] != n or B.shape != (n, 2) or cp_sector.shape != (n,):
        raise ValueError("upstream feature dimensions must agree on CP states")
    if any(not np.isfinite(m).all() for m in (A, C, H, B)) or any((m < 0).any() for m in (A, C, H)):
        raise ValueError("upstream Gram features must be finite and nonnegative")
    if not np.isfinite(drive_mix) or not 0 <= drive_mix <= 1:
        raise ValueError("drive_mix must be in [0, 1]")
    if n <= 1:
        raise ValueError("Upstream substrate has insufficient CP states for kernel bridge")

    sym = (A.T @ A) + 0.15 * (C.T @ C) + 0.1 * (H @ H.T)
    # Sector-coincidence boost
    same_sector = (cp_sector[:, None] == cp_sector[None, :]).astype(np.float64)
    sym = sym + 0.2 * same_sector + 1e-6 * np.eye(n)
    sym = 0.5 * (sym + sym.T)
    sym_row = row_normalize(sym)

    pref = B[:, 0] - B[:, 1]
    directed = np.maximum(pref[:, None] - pref[None, :], 0.0)
    directed = directed + 1e-9 * np.eye(n)
    directed_row = row_normalize(directed)

    alpha = float(drive_mix)
    P = (1.0 - alpha) * sym_row + alpha * directed_row
    validate_row_stochastic(P)
    _, lens = np.unique(cp_sector, return_inverse=True)
    return P, np.asarray(lens, dtype=np.int64)


def run_upstream_candidate(candidate_config: dict[str, Any], output_root: str | Path | None = None, use_cache: bool = True, bridge_config: dict[str, Any] | None = None) -> dict[str, Any]:
    _ = output_root
    _ = use_cache
    root = _repo_root()
    cfg = bridge_config or json.loads((root / "configs/bridges/upstream_pica_bridge.json").read_text(encoding="utf-8"))
    import_path = _upstream_import_path(root, cfg)
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

    entrypoint = "closurelab.matrix_compile.compile_symbolic_instance_to_substrate"
    if candidate_config["entrypoint"] != entrypoint:
        raise ValueError("unsupported upstream compiler entrypoint")
    from closurelab.matrix_compile import compile_symbolic_instance_to_substrate
    import closurelab.matrix_compile as compiler
    if not Path(compiler.__file__).resolve().is_relative_to(import_path.resolve()):
        raise ValueError("loaded upstream compiler does not belong to the requested source root")

    upstream_root = root / str(cfg["upstream_source_root"])
    diagram_path = upstream_root / str(candidate_config["diagram_path"])
    substrate = compile_symbolic_instance_to_substrate(
        diagram_path=diagram_path,
        config=str(candidate_config["config"]),
        rule_profile=str(candidate_config["rule_profile"]),
        max_steps=int(candidate_config.get("max_steps", 4)),
    )

    P, lens = _kernel_from_upstream_substrate(substrate, drive_mix=float(candidate_config.get("drive_mix", 0.05)))
    return {
        "candidate_name": str(candidate_config["name"]),
        "upstream_source_root": str(upstream_root),
        "upstream_entrypoint": str(candidate_config["entrypoint"]),
        "diagram_path": str(diagram_path),
        "config": str(candidate_config["config"]),
        "rule_profile": str(candidate_config["rule_profile"]),
        "n": int(P.shape[0]),
        "P": P,
        "coarse_lens": lens,
        "bridge_candidate_usable": True,
        "kernel_origin": "constructed_from_compiled_upstream_features",
        "kernel_construction": "row-normalized feature Gram plus sector boost, mixed with an artificial preference-gradient kernel",
        "diagram_sha256": hashlib.sha256(diagram_path.read_bytes()).hexdigest(),
        "compiler_sha256": hashlib.sha256(Path(compiler.__file__).read_bytes()).hexdigest(),
    }


def extract_upstream_kernel_family(candidate_output: dict[str, Any]) -> dict[str, Any]:
    return {
        "P": np.asarray(candidate_output["P"], dtype=np.float64),
        "coarse_lens": np.asarray(candidate_output["coarse_lens"], dtype=np.int64),
        "n": int(candidate_output["n"]),
    }


def _evaluate_observables_for_kernel(
    P: np.ndarray,
    lens: np.ndarray,
    lambdas: list[float],
    control_mode: str,
    rubric_cfg: dict[str, Any],
    p4_cfg: dict[str, Any],
    onset_thresholds: list[float],
) -> dict[str, Any]:
    substrate = {
        "P": np.asarray(P, dtype=np.float64),
        "coarse_lens": np.asarray(lens, dtype=np.int64),
        "n": int(P.shape[0]),
    }
    tau1_rows, tau1_boundary, tau1_aff = _evaluate_tau_panel(substrate, lambdas, 1, control_mode)
    tau2_rows, tau2_boundary, tau2_aff = _evaluate_tau_panel(substrate, lambdas, 2, control_mode)
    ev = evaluate_class_iii_candidate(
        {
            "tau1_rows": tau1_rows,
            "tau2_rows": tau2_rows,
            "tau1_boundary": tau1_boundary,
            "tau2_boundary": tau2_boundary,
            "tau1_affinity_ref": tau1_aff,
            "tau2_affinity_ref": tau2_aff,
        },
        rubric_cfg,
        p4_cfg,
    )
    onset = {str(t): first_threshold_lambda(tau1_rows, "affinity", t, "geq") for t in onset_thresholds}

    ce = observed_float(ev["ce_boundary_tau1"]) if ev["ce_boundary_tau1"] is not None else None
    mo = observed_float(ev["mobj_boundary_tau1"]) if ev["mobj_boundary_tau1"] is not None else None
    stce = observed_float(ev["staging_shift_ce"])
    stmo = observed_float(ev["staging_shift_mobj"])
    p4_dual = compute_p4_dual_shift_min(stce, stmo)

    return {
        "p5_state": str(ev["candidate_p5_state"]),
        "p6_drive_state": str(ev["candidate_p6_drive_state"]),
        "p4_like_signal": bool(ev["p4_like_signal"]),
        "p4_class_active": bool(ev["p4_class_active"]),
        "canonical_class_label": str(ev["canonical_class_label"]),
        "ce_boundary_lambda": ce,
        "mobj_boundary_lambda": mo,
        "structural_boundary_lambda_ref": compute_structural_boundary_lambda(ce, mo),
        "affinity_ref": observed_float(tau1_aff["affinity_ref"]),
        "affinity_ref_max_over_tau": observed_float(ev["affinity_ref_max"]),
        "affinity_reference_tau": 1,
        "affinity_carrier": "microstate_kernel",
        "affinity_onset_1e-3": onset.get("0.001"),
        "affinity_onset_1e-2": onset.get("0.01"),
        "affinity_onset_5e-2": onset.get("0.05"),
        "staging_shift_ce": stce,
        "staging_shift_mobj": stmo,
        "p4_dual_shift_min": p4_dual,
        "tau1_rows": tau1_rows,
        "tau2_rows": tau2_rows,
    }


def evaluate_upstream_on_canonical_observables(kernel_rows: dict[str, Any], rubric_config: dict[str, Any], p4_config: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    P = np.asarray(kernel_rows["P"], dtype=np.float64)
    lens = np.asarray(kernel_rows["coarse_lens"], dtype=np.int64)
    return _evaluate_observables_for_kernel(
        P,
        lens,
        lambdas=[float(x) for x in cfg["lambda_grid"]],
        control_mode=str(cfg["control_application_name"]),
        rubric_cfg=rubric_config,
        p4_cfg=p4_config,
        onset_thresholds=[float(x) for x in cfg["affinity_onset_thresholds"]],
    )


def classify_upstream_bridge_result(observable_summary: dict[str, Any], taxonomy_config: dict[str, Any]) -> dict[str, Any]:
    label = str(observable_summary.get("canonical_class_label", "unclassified"))
    p_states = [str(observable_summary.get("p5_state", "unknown")), str(observable_summary.get("p6_drive_state", "unknown")),
                "unknown" if observable_summary.get("p4_class_active") is None else "active" if observable_summary["p4_class_active"] else "inactive"]
    signatures = taxonomy_config.get("canonical_class_signatures")
    if signatures is None:
        signatures = json.loads((_repo_root() / "configs/taxonomy/canonical_class_rubric.json").read_text())["canonical_class_signatures"]
    derived = classify_activation_signature(*p_states, signatures)
    if any(state not in {"active", "inactive"} for state in p_states):
        final_label = "unclear"
    elif label in signatures and label != derived:
        final_label = "unclear"  # supplied label contradicts its primitive evidence
    elif derived in signatures:
        final_label = derived
    else:
        final_label = "mixed"
    return {
        "canonical_class_label": final_label,
        "raw_canonical_label": label,
        "primitive_states": {
            "p5_state": p_states[0],
            "p6_drive_state": p_states[1],
            "p4_class_active": p_states[2] == "active",
        },
    }


def assign_bridge_confidence(classification_summary: dict[str, Any], support_instances: int = 1) -> str:
    label = str(classification_summary.get("canonical_class_label", "unclear"))
    p5 = str(classification_summary.get("primitive_states", {}).get("p5_state", "unknown"))
    p6 = str(classification_summary.get("primitive_states", {}).get("p6_drive_state", "unknown"))
    if label in {"mixed", "unclear"}:
        return "low"
    if p5 != "unknown" and p6 != "unknown" and support_instances >= 2:
        return "high"
    return "medium"


def _overlay_plot(path: Path, upstream_rows: list[dict[str, Any]], class_map_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=180)
    class_colors = {
        "Class-I": "#2873b4",
        "Class-II": "#23aa5a",
        "Class-III": "#d28219",
        "Class-IV": "#b42d2d",
    }
    for r in class_map_rows:
        x = float(r["structural_boundary_lambda_ref"])
        y = math.log10(max(abs(observed_float(r["affinity_ref"])), 1e-12))
        ax.scatter([x], [y], s=80, color=class_colors.get(str(r["class_name"]), "black"), edgecolors="black", linewidths=0.5)
        ax.annotate(str(r["class_name"]), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    for r in upstream_rows:
        x = float(r["structural_boundary_lambda_ref"])
        y = math.log10(max(abs(observed_float(r["affinity_ref"])), 1e-12))
        ax.scatter([x], [y], s=100, color="#7a2ca6", marker="D", edgecolors="black", linewidths=0.6)
        ax.annotate(str(r["upstream_candidate_name"]), (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel(r"Structural boundary $\lambda_{\mathrm{struct,ref}}$", fontsize=10)
    ax.set_ylabel(r"$\log_{10} |\mathrm{Aff}_{\mathrm{ref}}|$", fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _axes_plot(path: Path, best_row: dict[str, Any], class_map_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    rows = class_map_rows + [best_row]
    labels = [r["class_name"] for r in class_map_rows] + [f"upstream:{best_row['upstream_candidate_name']}"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.0), dpi=180)
    x = np.arange(len(labels))
    metrics = [
        ("structural_boundary_lambda_ref", r"Structural boundary $\lambda_{\mathrm{struct,ref}}$", "#2873b4"),
        ("affinity_ref", r"$|\mathrm{Aff}_{\mathrm{ref}}|$", "#23aa5a"),
        ("p4_dual_shift_min", r"$P4_{\mathrm{dual}}$", "#d28219"),
    ]
    for ax, (key, ylabel, color) in zip(axes, metrics):
        vals = [abs(float(r[key])) if key == "affinity_ref" else float(r[key]) for r in rows]
        bars = ax.bar(x, vals, color=color, width=0.62)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        if key == "affinity_ref":
            ax.set_yscale("log")
        else:
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_upstream_pica_bridge(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg, root, candidates = discover_upstream_bridge_candidates(config_path_or_obj)
    _ensure_prereq_roots(root, use_cache=use_cache)

    rubric = json.loads((root / "configs/taxonomy/canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((root / "configs/metrics/p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))

    successful: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for cand in candidates:
        try:
            ran = run_upstream_candidate(cand, output_root=None, use_cache=use_cache, bridge_config=cfg)
            kfam = extract_upstream_kernel_family(ran)
            obs = evaluate_upstream_on_canonical_observables(kfam, rubric, p4_cfg, cfg)
            cls = classify_upstream_bridge_result(obs, rubric)
            row = {
                "upstream_candidate_name": ran["candidate_name"],
                "upstream_source_root": ran["upstream_source_root"],
                "upstream_entrypoint": ran["upstream_entrypoint"],
                "n": int(ran["n"]),
                "kernel_origin": ran["kernel_origin"],
                "kernel_construction": ran["kernel_construction"],
                "diagram_sha256": ran["diagram_sha256"],
                "compiler_sha256": ran["compiler_sha256"],
                **obs,
                "canonical_class_label": cls["canonical_class_label"],
                "raw_canonical_label": cls["raw_canonical_label"],
                "bridge_candidate_usable": True,
                "bridge_candidate_overlay_ready": all(obs.get(k) is not None and np.isfinite(float(obs[k])) and float(obs[k]) >= 0
                                                      for k in ("structural_boundary_lambda_ref", "affinity_ref", "p4_dual_shift_min")),
            }
            successful.append(row)
        except Exception as exc:
            failures.append(
                {
                    "upstream_candidate_name": str(cand.get("name", "unknown")),
                    "upstream_source_root": str(root / cfg["upstream_source_root"]),
                    "upstream_entrypoint": str(cand.get("entrypoint", "unknown")),
                    "canonical_class_label": "unclear",
                    "classification_confidence": "low",
                    "bridge_candidate_usable": False,
                    "bridge_candidate_classifiable": False,
                    "bridge_candidate_overlay_ready": False,
                    "reason": str(exc),
                }
            )

    label_instances: dict[str, set[str]] = {}
    for r in successful:
        label_instances.setdefault(r["canonical_class_label"], set()).add(r["diagram_sha256"])

    for r in successful:
        conf = assign_bridge_confidence(
            {
                "canonical_class_label": r["canonical_class_label"],
                "primitive_states": {"p5_state": r["p5_state"], "p6_drive_state": r["p6_drive_state"]},
            },
            support_instances=len(label_instances.get(r["canonical_class_label"], set())),
        )
        r["classification_confidence"] = conf
        r["confidence_interpretation"] = "heuristic count of distinct compiled examples with matching operational labels; not statistical confidence or native dynamics validation"
        r["bridge_candidate_classifiable"] = bool(
            r["canonical_class_label"] in {"Class-I", "Class-II", "Class-III", "Class-IV"} and conf in {"medium", "high"}
        )

    all_rows = successful + failures
    overlay_candidates = [r for r in successful if r["bridge_candidate_overlay_ready"]]
    if overlay_candidates:
        def _score(r: dict[str, Any]) -> tuple[int, int, float]:
            label_rank = 0 if r["canonical_class_label"] in {"Class-I", "Class-II", "Class-III", "Class-IV"} else 1
            conf_rank = {"high": 0, "medium": 1, "low": 2}.get(str(r.get("classification_confidence", "low")), 2)
            return (label_rank, conf_rank, -float(r.get("p4_dual_shift_min", 0.0) or 0.0))

        best = sorted(overlay_candidates, key=_score)[0]
        bridge_success = True
    else:
        best = None
        bridge_success = False

    summary = {
        "upstream_bridge_success": bool(bridge_success),
        "best_bridge_candidate_name": best["upstream_candidate_name"] if best else None,
        "best_bridge_candidate_label": best["canonical_class_label"] if best else None,
        "best_bridge_candidate_confidence": best.get("classification_confidence") if best else None,
        "diagnosis": (
            "at least one upstream-feature-derived kernel executed and finite overlay coordinates extracted"
            if bridge_success
            else "no usable upstream candidate; see failure reasons"
        ),
    }

    out_root = (root / "results" / "bridges") if output_root is None else Path(output_root)
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

    _write_json(dirs["config"] / "config_snapshot.json", cfg)
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    _write_json(dirs["env"] / "environment.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "python": "python3"})

    _write_json(dirs["analysis"] / "upstream_candidate_table.json", {"candidates": all_rows})
    _write_csv(
        dirs["analysis"] / "upstream_candidate_table.csv",
        all_rows,
        [
            "upstream_candidate_name",
            "upstream_entrypoint",
            "n",
            "canonical_class_label",
            "classification_confidence",
            "p5_state",
            "p6_drive_state",
            "p4_like_signal",
            "p4_class_active",
            "ce_boundary_lambda",
            "mobj_boundary_lambda",
            "structural_boundary_lambda_ref",
            "affinity_ref",
            "affinity_onset_1e-3",
            "affinity_onset_1e-2",
            "affinity_onset_5e-2",
            "staging_shift_ce",
            "staging_shift_mobj",
            "p4_dual_shift_min",
            "bridge_candidate_usable",
            "bridge_candidate_classifiable",
            "bridge_candidate_overlay_ready",
        ],
    )
    _write_json(dirs["analysis"] / "upstream_bridge_summary.json", summary)

    _write_csv(
        dirs["metrics"] / "metrics.csv",
        all_rows,
        [
            "upstream_candidate_name",
            "upstream_entrypoint",
            "n",
            "canonical_class_label",
            "classification_confidence",
            "p5_state",
            "p6_drive_state",
            "p4_class_active",
            "ce_boundary_lambda",
            "mobj_boundary_lambda",
            "structural_boundary_lambda_ref",
            "affinity_ref",
            "p4_dual_shift_min",
            "bridge_candidate_usable",
            "bridge_candidate_classifiable",
            "bridge_candidate_overlay_ready",
        ],
    )

    # Figures: overlay using existing four-class coordinates.
    class_map = json.loads((root / "results/dashboards/four_class_observable_map/analysis/class_coordinates.json").read_text(encoding="utf-8"))["class_coordinates"]
    if best is not None:
        _overlay_plot(dirs["plots"] / "upstream_on_four_class_map.png", [best], class_map)
        _axes_plot(dirs["plots"] / "upstream_observable_axes.png", best, class_map)
    else:
        # write blank placeholders
        _write_png(dirs["plots"] / "upstream_on_four_class_map.png", np.full((300, 500, 3), 255, dtype=np.uint8))
        _write_png(dirs["plots"] / "upstream_observable_axes.png", np.full((300, 500, 3), 255, dtype=np.uint8))

    (dirs["notes"] / "findings.md").write_text(
        "\n".join(
            [
                "# F-05 upstream bridge",
                "",
                f"- upstream_bridge_success: `{summary['upstream_bridge_success']}`",
                f"- best_bridge_candidate_name: `{summary['best_bridge_candidate_name']}`",
                f"- best_bridge_candidate_label: `{summary['best_bridge_candidate_label']}`",
                f"- best_bridge_candidate_confidence: `{summary['best_bridge_candidate_confidence']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": str(cfg["bridge_id"]),
        "bundle_id": str(cfg["bridge_id"]),
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

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/F-05_upstream_pica_bridge.md"))
    best_label = summary["best_bridge_candidate_label"] if summary["best_bridge_candidate_label"] is not None else "unclear"
    best_conf = summary["best_bridge_candidate_confidence"] if summary["best_bridge_candidate_confidence"] is not None else "low"
    note_path.write_text(
        "\n".join(
            [
                "# F-05 Upstream/PICA bridge campaign",
                "",
                "- Config: `configs/bridges/upstream_pica_bridge.json`",
                f"- Artifact root: `{artifact_root}`",
                f"- Candidate sources tried: `{[c.get('name') for c in candidates]}`",
                f"- Best candidate: `{summary['best_bridge_candidate_name']}`",
                f"- Best label: `{best_label}`",
                f"- Confidence: `{best_conf}`",
                f"- Bridge success: `{summary['upstream_bridge_success']}`",
                "",
                f"**does at least one real upstream-style system appear on the same comparison assets as the synthetic classes? {'yes' if summary['upstream_bridge_success'] else 'no'}**",
                f"**where does the best upstream candidate land in the canonical taxonomy, and with what confidence? {best_label}, {best_conf}**",
                "- Note: a mixed/unclear landing is still a useful external-validity bridge when observables are extracted on the shared map.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "artifact_root": str(artifact_root),
        **summary,
        "evaluated_candidates": len(successful),
    }


__all__ = [
    "discover_upstream_bridge_candidates",
    "run_upstream_candidate",
    "extract_upstream_kernel_family",
    "evaluate_upstream_on_canonical_observables",
    "classify_upstream_bridge_result",
    "assign_bridge_confidence",
    "build_upstream_pica_bridge",
]
