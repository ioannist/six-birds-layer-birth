"""F-03 four-class observable-space map."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import implementation_fingerprint
from .campaigns import _git_code_version
from .class3 import _build_class_iv_candidate_substrate, _evaluate_tau_panel, _repo_root, _write_csv, evaluate_class_iii_candidate
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .robustness import _write_png
from .substrates import build_class_iii_candidate_family
from .boundaries import structural_reference_lambda, first_threshold_lambda
from .taxonomy import evaluate_reference_profile, _build_curve_rows_for_profile


def load_observable_map_sources(config_path_or_obj: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path]:
    cfg = config_path_or_obj if isinstance(config_path_or_obj, dict) else json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))
    root = _repo_root()
    return cfg, root


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(scientific_dumps(payload, indent=2) + "\n", encoding="utf-8")


def compute_structural_boundary_lambda(ce_boundary_lambda: float | None, mobj_boundary_lambda: float | None) -> float | None:
    return structural_reference_lambda(ce_boundary_lambda, mobj_boundary_lambda)


def compute_p4_dual_shift_min(staging_shift_ce: float | None, staging_shift_mobj: float | None) -> float | None:
    vals = [v for v in [staging_shift_ce, staging_shift_mobj] if v is not None]
    if len(vals) < 2 or any(not math.isfinite(v) for v in vals):
        return None
    return float(min(abs(v) for v in vals))


def _rubric_for_map(config: dict[str, Any], root: Path) -> dict[str, Any]:
    rubric = json.loads((root / "configs/taxonomy/canonical_class_rubric.json").read_text())
    # A different structural threshold defines a different experiment. Require
    # the canonical thresholds rather than silently keeping canonical labels.
    for key in ("ce_target", "mobj_target"):
        if float(config[key]) != float(rubric["activation_thresholds"]["p5"][key]):
            raise ValueError("observable map targets must match the canonical rubric")
    return rubric


def _coordinate_metadata(config: dict[str, Any], size: int, source: Path) -> dict[str, Any]:
    return {
        "representative_size_used": size,
        "representative_size_fallback": size != int(config["representative_size"]),
        "measurement_recomputed": True,
        "affinity_carrier": "microstate_kernel",
        "affinity_reference_tau": 1,
        "boundary_interpretation": "sampled_piecewise_linear_proxy",
        "ce_target": float(config["ce_target"]),
        "mobj_target": float(config["mobj_target"]),
        "source_provenance": {
            "measurement_configuration": str(source.resolve()),
            "implementation_sha256": implementation_fingerprint(),
        },
    }


def _censored(boundary: float | None, rows: list[dict[str, Any]]) -> bool:
    return boundary is not None and boundary == min(float(r["closure_strength_lambda"]) for r in rows)


def _extract_class_i_ii(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rubric = _rubric_for_map(config, root)
    size = int(config["representative_size"])
    if size <= 0:
        raise ValueError("representative size must be positive")
    out = []
    for original, cname in zip(rubric["reference_profiles"], ("Class-I", "Class-II")):
        profile = copy.deepcopy(original)
        for channel in (profile, profile["staging_check"]):
            channel["size"] = size
            if "block_size" in channel:
                blocks = int(channel["n_blocks"])
                if size % blocks:
                    raise ValueError("representative size must be divisible by the block count")
                channel["block_size"] = size // blocks
        measured = evaluate_reference_profile(profile, rubric, use_cache=False)
        rows = _build_curve_rows_for_profile(profile, tau=1)
        onset = _affinity_onset_from_rows(rows, config["affinity_onset_thresholds"])
        ce, mo = measured["ce_boundary_lambda"], measured["mobj_boundary_lambda"]
        stce, stmo = abs(measured["tau_ce_shift"]), abs(measured["tau_mobj_shift"])
        out.append({
            **_coordinate_metadata(config, size, root / "configs/taxonomy/canonical_class_rubric.json"),
            "class_name": cname,
            "canonical_class_label": measured["canonical_class_label"],
            "p5_state": measured["p5_state"],
            "p6_drive_state": measured["p6_drive_state"],
            "p4_state": measured["p4_state"],
            "reference_profile_name": profile["profile_name"],
            "representative_name": profile["profile_name"],
            "ce_boundary_lambda": ce,
            "mobj_boundary_lambda": mo,
            "ce_boundary_left_censored": _censored(ce, rows),
            "mobj_boundary_left_censored": _censored(mo, rows),
            "structural_boundary_lambda_ref": compute_structural_boundary_lambda(ce, mo),
            "affinity_ref": measured["affinity_ref"],
            "affinity_onset_1e-3": onset.get("0.001"),
            "affinity_onset_1e-2": onset.get("0.01"),
            "affinity_onset_5e-2": onset.get("0.05"),
            "staging_shift_ce": stce,
            "staging_shift_mobj": stmo,
            "p4_dual_shift_min": compute_p4_dual_shift_min(stce, stmo),
            "p4_like_signal": measured["p4_like_signal"],
            "p4_class_active": measured["p4_class_active"],
            "source_bundle_root": str((root / config["sources"]["taxonomy_root"]).resolve()),
        })
        out[-1]["source_provenance"]["resolved_profile"] = profile
    return out


def _affinity_onset_from_rows(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, float | None]:
    if any(observed_float(r["affinity"]) < 0 for r in rows):
        raise ValueError("stationary reversal affinity must be nonnegative")
    return {str(float(t)): first_threshold_lambda(rows, "affinity", float(t), "geq") for t in thresholds}


def _extract_class_iii_iv(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rubric = _rubric_for_map(config, root)
    p4 = json.loads((root / "configs/metrics/p4_anomaly_metric_layer.json").read_text())
    out = []
    for cname, config_name, source_key in [
        ("Class-III", "class_iii_full_campaign", "class_iii_campaign_root"),
        ("Class-IV", "class_iv_full_campaign", "class_iv_campaign_root"),
    ]:
        source = root / f"configs/campaigns/{config_name}.json"
        cfg = json.loads(source.read_text())
        sizes = [int(n) for n in cfg["size_panel"]]
        size = min(sizes, key=lambda n: (abs(n - int(config["representative_size"])), n))
        candidate = cfg["candidate"]
        if cname == "Class-III":
            sub = build_class_iii_candidate_family(candidate["family_name"], **{**candidate["base_kwargs"], "n": size})
        else:
            sub = _build_class_iv_candidate_substrate(candidate["family_name"], candidate["base_kwargs"], candidate.get("drive_kwargs", {}), size)
        channels = [_evaluate_tau_panel(sub, cfg["lambda_grid"], tau=t, control_mode=cfg["control_application_name"]) for t in (1, 2)]
        rows, boundary, affinity = channels[0]
        measured = evaluate_class_iii_candidate({
            "tau1_boundary": channels[0][1], "tau2_boundary": channels[1][1],
            "tau1_affinity_ref": channels[0][2], "tau2_affinity_ref": channels[1][2],
        }, rubric, p4)
        onset = _affinity_onset_from_rows(rows, config["affinity_onset_thresholds"])
        ce, mo = boundary["ce_boundary_lambda"], boundary["mobj_boundary_lambda"]
        stce, stmo = measured["staging_shift_ce"], measured["staging_shift_mobj"]
        out.append({
            **_coordinate_metadata(config, size, source),
            "class_name": cname,
            "canonical_class_label": measured["canonical_class_label"],
            "p5_state": measured["candidate_p5_state"],
            "p6_drive_state": measured["candidate_p6_drive_state"],
            "p4_state": measured["candidate_p4_state"],
            "reference_profile_name": cname.lower().replace("-", "_") + "_reference",
            "representative_name": candidate["candidate_name"],
            "ce_boundary_lambda": ce,
            "mobj_boundary_lambda": mo,
            "ce_boundary_left_censored": _censored(ce, rows),
            "mobj_boundary_left_censored": _censored(mo, rows),
            "structural_boundary_lambda_ref": compute_structural_boundary_lambda(ce, mo),
            "affinity_ref": affinity["affinity_ref"],
            "affinity_ref_max_over_tau": measured["affinity_ref_max"],
            "affinity_onset_1e-3": onset.get("0.001"),
            "affinity_onset_1e-2": onset.get("0.01"),
            "affinity_onset_5e-2": onset.get("0.05"),
            "staging_shift_ce": stce,
            "staging_shift_mobj": stmo,
            "p4_dual_shift_min": compute_p4_dual_shift_min(stce, stmo),
            "p4_like_signal": measured["p4_like_signal"],
            "p4_class_active": measured["p4_class_active"],
            "source_bundle_root": str((root / config["sources"][source_key]).resolve()),
        })
        out[-1]["source_provenance"]["resolved_campaign"] = cfg
    return out


def extract_class_coordinate(profile_or_bundle: dict[str, Any], representative_size: int, targets: dict[str, float]) -> dict[str, Any]:
    if profile_or_bundle.get("representative_size_used") != representative_size:
        raise ValueError("coordinate does not describe the requested representative size")
    for key in ("ce_target", "mobj_target"):
        if key not in profile_or_bundle or float(profile_or_bundle[key]) != float(targets[key]):
            raise ValueError("coordinate does not describe the requested structural thresholds")
    return dict(profile_or_bundle)


def summarize_observable_separation(class_rows: list[dict[str, Any]], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    thr = thresholds or {}
    p6_inactive_max = float(thr.get("p6_inactive_max", 1e-6))
    p6_active_min = float(thr.get("p6_active_min", 1e-3))
    p4_active_min = float(thr.get("p4_active_min", 0.15))
    spread_min = float(thr.get("structural_spread_min", 0.05))

    by = {r["class_name"]: r for r in class_rows}
    p6_inactive = ["Class-I", "Class-III"]
    p6_active = ["Class-II", "Class-IV"]
    p4_inactive = ["Class-I", "Class-II"]
    p4_active = ["Class-III", "Class-IV"]

    complete = len(class_rows) == 4 and set(by) == set(p6_inactive + p6_active)
    def valid(row: dict[str, Any], key: str) -> bool:
        value = row.get(key)
        return value is not None and math.isfinite(float(value)) and float(value) >= 0

    p6_axis_clean = complete and all(valid(r, "affinity_ref") for r in class_rows) and all(
        observed_float(by[c]["affinity_ref"]) <= p6_inactive_max for c in p6_inactive
    ) and all(observed_float(by[c]["affinity_ref"]) >= p6_active_min for c in p6_active)
    p4_axis_clean = complete and all(valid(r, "p4_dual_shift_min") for r in class_rows) and all(
        float(by[c]["p4_dual_shift_min"]) < p4_active_min for c in p4_inactive
    ) and all(float(by[c]["p4_dual_shift_min"]) >= p4_active_min for c in p4_active)
    structural_valid = complete and all(valid(r, "structural_boundary_lambda_ref") for r in class_rows)
    structural_vals = [float(r["structural_boundary_lambda_ref"]) for r in class_rows] if structural_valid else []
    spread = max(structural_vals) - min(structural_vals) if structural_vals else None
    structural_boundary_spread_present = spread is not None and spread >= spread_min

    observable_map_separable = bool(p6_axis_clean and p4_axis_clean)
    labels_agree = all(r.get("canonical_class_label", r["class_name"]) == r["class_name"] for r in class_rows)
    phase_map_figure_publishable = bool(observable_map_separable and structural_valid and labels_agree)

    return {
        "p6_axis_clean": bool(p6_axis_clean),
        "complete_four_class_panel": complete,
        "measured_class_labels_agree": labels_agree,
        "p4_axis_clean": bool(p4_axis_clean),
        "structural_boundary_spread_present": structural_boundary_spread_present,
        "structural_boundary_spread": spread,
        "observable_map_separable": observable_map_separable,
        "phase_map_figure_publishable": phase_map_figure_publishable,
        "diagnosis": (
            "four classes are quantitatively separable on P6 and P4 axes"
            if observable_map_separable
            else "at least one axis separation condition failed"
        ),
    }


def _draw_circle(img: np.ndarray, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
    h, w, _ = img.shape
    y0, y1 = max(0, cy - r), min(h - 1, cy + r)
    x0, x1 = max(0, cx - r), min(w - 1, cx + r)
    rr = r * r
    for y in range(y0, y1 + 1):
        dy = y - cy
        for x in range(x0, x1 + 1):
            dx = x - cx
            if dx * dx + dy * dy <= rr:
                img[y, x] = color


def _render_map(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    xs = np.asarray([float(r["structural_boundary_lambda_ref"]) for r in rows], dtype=np.float64)
    ys = np.asarray([math.log10(max(abs(observed_float(r["affinity_ref"])), 1e-12)) for r in rows], dtype=np.float64)
    cs = np.asarray([float(r["p4_dual_shift_min"]) for r in rows], dtype=np.float64)
    labels = [str(r["class_name"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=180)
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=90, edgecolors="black", linewidths=0.6)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel(r"Structural boundary $\lambda_{\mathrm{struct,ref}}$", fontsize=10)
    ax.set_ylabel(r"$\log_{10} |\mathrm{Aff}_{\mathrm{ref}}|$", fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$P4_{\mathrm{dual}}$", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_axes_panels(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    classes = ["Class-I", "Class-II", "Class-III", "Class-IV"]
    mapping = {r["class_name"]: r for r in rows}

    metrics = [
        ("structural_boundary_lambda_ref", r"Structural boundary $\lambda_{\mathrm{struct,ref}}$", (20 / 255, 120 / 255, 200 / 255)),
        ("affinity_ref", r"$|\mathrm{Aff}_{\mathrm{ref}}|$", (40 / 255, 180 / 255, 80 / 255)),
        ("p4_dual_shift_min", r"$P4_{\mathrm{dual}}$", (200 / 255, 100 / 255, 30 / 255)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.8), dpi=180)
    x = np.arange(len(classes))
    for ax, (key, ylabel, color) in zip(axes, metrics):
        vals = [abs(float(mapping[c][key])) if key == "affinity_ref" else float(mapping[c][key]) for c in classes]
        bars = ax.bar(x, vals, color=color, width=0.6)
        ax.set_xticks(x, classes, rotation=20)
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


def format_observable_map_summary(class_rows: list[dict[str, Any]], separation_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "dashboard_id": "four_class_observable_map",
        "class_count": len(class_rows),
        "classes": [r["class_name"] for r in class_rows],
        **separation_summary,
    }


def build_four_class_observable_map(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    _ = use_cache
    cfg, root = load_observable_map_sources(config_path_or_obj)

    class_rows = _extract_class_i_ii(cfg, root) + _extract_class_iii_iv(cfg, root)
    class_rows = sorted(class_rows, key=lambda r: ["Class-I", "Class-II", "Class-III", "Class-IV"].index(r["class_name"]))
    for r in class_rows:
        r["log10_affinity_ref_clipped"] = float(math.log10(max(abs(observed_float(r["affinity_ref"])), 1e-12)))

    sep = summarize_observable_separation(class_rows)
    summary = format_observable_map_summary(class_rows, sep)

    out_root = (root / "results" / "dashboards") if output_root is None else Path(output_root)
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

    _write_json(dirs["analysis"] / "class_coordinates.json", {"class_coordinates": class_rows})
    _write_json(dirs["analysis"] / "separation_summary.json", sep)
    _write_csv(dirs["analysis"] / "class_coordinates.csv", class_rows, [
        "class_name",
        "representative_name",
        "representative_size_used",
        "canonical_class_label",
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
        "p4_class_active",
        "source_bundle_root",
    ])

    _write_csv(dirs["metrics"] / "metrics.csv", class_rows, [
        "class_name",
        "representative_name",
        "representative_size_used",
        "canonical_class_label",
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
        "p4_class_active",
        "source_bundle_root",
    ])

    _render_map(dirs["plots"] / "four_class_observable_map.png", class_rows)
    _render_axes_panels(dirs["plots"] / "observable_axes_panels.png", class_rows)

    (dirs["notes"] / "findings.md").write_text(
        "\n".join(
            [
                "# F-03 four-class observable map",
                "",
                f"- observable_map_separable: `{sep['observable_map_separable']}`",
                f"- phase_map_figure_publishable: `{sep['phase_map_figure_publishable']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": str(cfg["dashboard_id"]),
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
    schema = load_schema(root / "configs" / "result_bundle.schema.json")
    validate_manifest(manifest, schema)
    _write_json(artifact_root / "manifest.json", manifest)

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/F-03_four_class_observable_map.md"))
    note_path.write_text(
        "\n".join(
            [
                "# F-03 Four-class observable phase map",
                "",
                "- Config: `configs/dashboards/four_class_observable_map.json`",
                f"- Artifact root: `{artifact_root}`",
                "- Channel: canonical manual-lens observables.",
                f"- Separation summary: p6_axis_clean=`{sep['p6_axis_clean']}`, p4_axis_clean=`{sep['p4_axis_clean']}`",
                f"- Final map verdict: observable_map_separable=`{sep['observable_map_separable']}`, phase_map_figure_publishable=`{sep['phase_map_figure_publishable']}`",
                "",
                f"**are the four confirmed classes visibly and quantitatively separable in measured observable space? {'yes' if sep['observable_map_separable'] else 'no'}**",
                f"**is this phase-map style figure strong enough to support the paper’s non-exponent class story? {'yes' if sep['phase_map_figure_publishable'] else 'no'}**",
                "- Scope note: this is an observable-space map, not a universality-exponent phase diagram.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "artifact_root": str(artifact_root),
        "p6_axis_clean": bool(sep["p6_axis_clean"]),
        "p4_axis_clean": bool(sep["p4_axis_clean"]),
        "observable_map_separable": bool(sep["observable_map_separable"]),
        "phase_map_figure_publishable": bool(sep["phase_map_figure_publishable"]),
        "class_rows": class_rows,
    }
