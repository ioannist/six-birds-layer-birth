"""F-04 capacity postcritical dashboard."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import bundle_is_current, cache_reuse_allowed
from .campaigns import _git_code_version
from .class3 import _repo_root, _write_csv
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .robustness import _write_png

def load_capacity_dashboard_sources(config_path_or_obj: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    cfg = config_path_or_obj if isinstance(config_path_or_obj, dict) else json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))
    root = _repo_root()
    source_root = root / str(cfg["source_capacity_bundle_root"])
    return cfg, root, source_root


def _trend_label(values: list[float], tol: float = 1e-9) -> str:
    if not values or not np.all(np.isfinite(values)):
        return "unavailable"
    if len(values) <= 1:
        return "flat"
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if all(abs(d) <= tol for d in diffs):
        return "flat"
    if all(d >= -tol for d in diffs) and any(d > tol for d in diffs):
        return "increasing"
    if all(d <= tol for d in diffs) and any(d < -tol for d in diffs):
        return "decreasing"
    return "mixed"


def extract_representative_capacity_rows(source_bundle: Path, representative_size: int = 64) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    metrics_rows = list(csv.DictReader((source_bundle / "metrics" / "metrics.csv").open("r", encoding="utf-8")))
    by_class: dict[str, list[dict[str, Any]]] = {}
    for r in metrics_rows:
        by_class.setdefault(str(r["class_name"]), []).append(r)

    rep_rows: list[dict[str, Any]] = []
    trends: dict[str, dict[str, str]] = {}
    for cname in ["Class-I", "Class-II", "Class-III", "Class-IV"]:
        rows = sorted(by_class.get(cname, []), key=lambda r: int(r["size"]))
        if not rows:
            continue
        sizes = [int(r["size"]) for r in rows]
        chosen_size = min(sizes, key=lambda s: abs(s - int(representative_size)))
        chosen = next(r for r in rows if int(r["size"]) == chosen_size)

        lvals = [float(r["lambda_headroom"]) for r in rows]
        dvals = [float(r["depth_survival_sat"]) for r in rows]
        cvals = [float(r["capacity_depth_index"]) for r in rows]
        trends[cname] = {
            "lambda_headroom_trend": _trend_label(lvals),
            "depth_survival_sat_trend": _trend_label(dvals),
            "capacity_depth_index_trend": _trend_label(cvals),
        }

        rep_rows.append(
            {
                "class_name": cname,
                "representative_name": str(chosen["representative_name"]),
                "representative_size_used": int(chosen_size),
                "representative_size_fallback": bool(int(chosen_size) != int(representative_size)),
                "birth_lambda_ref": float(chosen["birth_lambda_ref"]),
                "lambda_headroom": float(chosen["lambda_headroom"]),
                "depth_survival_sat": float(chosen["depth_survival_sat"]),
                "objecthood_tau_area_sat": observed_float(chosen["objecthood_tau_area_sat"]),
                "capacity_depth_index": float(chosen["capacity_depth_index"]),
                "p5_p6_retention_fraction": float(chosen["p5_p6_retention_fraction"]),
                **trends[cname],
            }
        )
    return rep_rows, trends


def summarize_capacity_postcritical_story(source_bundle: Path, representative_rows: list[dict[str, Any]]) -> dict[str, Any]:
    axis_summary = json.loads((source_bundle / "analysis" / "axis_effect_summary.json").read_text(encoding="utf-8"))
    final_summary = json.loads((source_bundle / "analysis" / "final_verdict.json").read_text(encoding="utf-8"))
    source_verdict = str(axis_summary.get("final_verdict", final_summary.get("final_verdict", "capacity_orthogonal_or_mixed")))

    if source_verdict == "capacity_tracks_p4_axis":
        axis = "p4"
    elif source_verdict == "capacity_tracks_p6_axis":
        axis = "p6"
    else:
        axis = "mixed"

    capacity_primary_classifier = False
    fields = ("birth_lambda_ref", "lambda_headroom", "depth_survival_sat", "objecthood_tau_area_sat", "capacity_depth_index", "p5_p6_retention_fraction")
    complete = {row.get("class_name") for row in representative_rows} == {"Class-I", "Class-II", "Class-III", "Class-IV"} and len(representative_rows) == 4
    measured = complete and all(all(np.isfinite(observed_float(row.get(field))) for field in fields) for row in representative_rows)
    capacity_postcritical_story_supported = bool(
        measured
        and axis_summary.get("comparison_available")
        and source_verdict in {"capacity_orthogonal_or_mixed", "capacity_tracks_p4_axis", "capacity_tracks_p6_axis"}
    )
    dashboard_publishable = bool(capacity_postcritical_story_supported and len(representative_rows) == 4)

    return {
        "source_final_verdict": source_verdict,
        "capacity_primary_classifier": capacity_primary_classifier,
        "capacity_postcritical_story_supported": capacity_postcritical_story_supported,
        "capacity_axis_alignment": axis,
        "dashboard_publishable": dashboard_publishable,
        "axis_effect_summary": axis_summary,
        "support_scope": "descriptive dependent post-birth proxies; not a capacity theorem or independent axis evidence",
        "diagnosis": (
            "measured dependent post-birth proxies available for all four representatives"
            if capacity_postcritical_story_supported
            else "source bundle or representative table incomplete for postcritical story"
        ),
    }


def format_capacity_dashboard_summary(summary_rows: list[dict[str, Any]], source_verdict: dict[str, Any]) -> dict[str, Any]:
    return {
        "class_count": len(summary_rows),
        "classes": [r["class_name"] for r in summary_rows],
        "capacity_primary_classifier": bool(source_verdict["capacity_primary_classifier"]),
        "capacity_postcritical_story_supported": bool(source_verdict["capacity_postcritical_story_supported"]),
        "capacity_axis_alignment": str(source_verdict["capacity_axis_alignment"]),
        "dashboard_publishable": bool(source_verdict["dashboard_publishable"]),
        "diagnosis": str(source_verdict["diagnosis"]),
    }


def _draw_bar(img: np.ndarray, x0: int, x1: int, y0: int, y1: int, color: tuple[int, int, int]) -> None:
    h, w, _ = img.shape
    xa, xb = max(0, x0), min(w, x1)
    ya, yb = max(0, y0), min(h, y1)
    if xa < xb and ya < yb:
        img[ya:yb, xa:xb] = color


def _plot_capacity_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    classes = ["Class-I", "Class-II", "Class-III", "Class-IV"]
    by = {r["class_name"]: r for r in rows}
    metrics = [
        ("lambda_headroom", (40, 120, 200)),
        ("depth_survival_sat", (50, 170, 90)),
        ("capacity_depth_index", (200, 110, 40)),
    ]
    x = np.arange(len(classes))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.8, 4.2), dpi=180)
    for i, (key, color) in enumerate(metrics):
        vals = [float(by[c][key]) for c in classes]
        bars = ax.bar(x + (i - 1) * width, vals, width=width, color=np.asarray(color) / 255.0, label=key)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, classes)
    ax.set_ylabel("Normalized value", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_axis_summary(path: Path, axis_summary: dict[str, Any]) -> None:
    per_proxy = axis_summary.get("per_proxy", {})
    proxies = ["lambda_headroom", "depth_survival_sat", "objecthood_tau_area_sat", "capacity_depth_index"]
    w, h = 900, 380
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 50, 40, 30, 45
    x0, x1 = left, w - right
    y_base = h - bottom
    img[y_base - 1 : y_base + 1, x0:x1] = 0

    max_v = 1.0
    for p in proxies:
        pr = per_proxy.get(p, {})
        max_v = max(max_v, float(pr.get("p4_axis_effect", 0.0)), float(pr.get("p6_axis_effect", 0.0)))

    grp_w = (x1 - x0) // len(proxies)
    for i, p in enumerate(proxies):
        pr = per_proxy.get(p, {})
        p4 = float(pr.get("p4_axis_effect", 0.0))
        p6 = float(pr.get("p6_axis_effect", 0.0))
        base_x = x0 + i * grp_w + 18
        bw = max(8, (grp_w - 30) // 2)
        h4 = int((p4 / max_v) * (y_base - top - 10))
        h6 = int((p6 / max_v) * (y_base - top - 10))
        _draw_bar(img, base_x, base_x + bw, y_base - h4, y_base, (200, 100, 40))
        _draw_bar(img, base_x + bw + 8, base_x + 2 * bw + 8, y_base - h6, y_base, (40, 140, 200))

    _write_png(path, img)


def build_capacity_postcritical_dashboard(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    _ = use_cache
    cfg, root, source_root = load_capacity_dashboard_sources(config_path_or_obj)

    capacity_config = json.loads((root / "configs/campaigns/capacity_depth_extension.json").read_text())
    if not cache_reuse_allowed(use_cache) or not bundle_is_current(source_root, capacity_config):
        from .capacity import run_capacity_depth_extension

        run_capacity_depth_extension(
            root / "configs" / "campaigns" / "capacity_depth_extension.json",
            output_root=root / "results" / "campaigns",
            use_cache=True,
        )

    rep_rows, _ = extract_representative_capacity_rows(source_root, int(cfg.get("representative_size", 64)))
    source_verdict = summarize_capacity_postcritical_story(source_root, rep_rows)
    summary = format_capacity_dashboard_summary(rep_rows, source_verdict)

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

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(
        scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "python": "python3"}, indent=2) + "\n",
        encoding="utf-8",
    )

    (dirs["analysis"] / "representative_capacity_table.json").write_text(scientific_dumps({"rows": rep_rows}, indent=2) + "\n", encoding="utf-8")
    _write_csv(
        dirs["analysis"] / "representative_capacity_table.csv",
        rep_rows,
        [
            "class_name",
            "representative_name",
            "representative_size_used",
            "birth_lambda_ref",
            "lambda_headroom",
            "depth_survival_sat",
            "objecthood_tau_area_sat",
            "capacity_depth_index",
            "p5_p6_retention_fraction",
            "lambda_headroom_trend",
            "depth_survival_sat_trend",
            "capacity_depth_index_trend",
        ],
    )
    (dirs["analysis"] / "capacity_dashboard_summary.json").write_text(scientific_dumps(summary, indent=2) + "\n", encoding="utf-8")

    _write_csv(
        dirs["metrics"] / "metrics.csv",
        rep_rows,
        [
            "class_name",
            "representative_name",
            "representative_size_used",
            "birth_lambda_ref",
            "lambda_headroom",
            "depth_survival_sat",
            "objecthood_tau_area_sat",
            "capacity_depth_index",
            "p5_p6_retention_fraction",
            "lambda_headroom_trend",
            "depth_survival_sat_trend",
            "capacity_depth_index_trend",
        ],
    )

    _plot_capacity_comparison(dirs["plots"] / "capacity_postcritical_comparison.png", rep_rows)
    _plot_axis_summary(dirs["plots"] / "capacity_axis_summary.png", source_verdict["axis_effect_summary"])

    (dirs["notes"] / "findings.md").write_text(
        "\n".join(
            [
                "# F-04 capacity postcritical dashboard",
                "",
                f"- capacity_primary_classifier: `{summary['capacity_primary_classifier']}`",
                f"- capacity_axis_alignment: `{summary['capacity_axis_alignment']}`",
                f"- dashboard_publishable: `{summary['dashboard_publishable']}`",
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
    (artifact_root / "manifest.json").write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    note_path = root / str(cfg.get("findings_note_path", "notes/findings/F-04_capacity_postcritical_dashboard.md"))
    note_path.write_text(
        "\n".join(
            [
                "# F-04 Capacity postcritical dashboard",
                "",
                "- Config: `configs/dashboards/capacity_postcritical_dashboard.json`",
                f"- Artifact root: `{artifact_root}`",
                f"- Source bundle: `{source_root}`",
                f"- Axis summary verdict: `{source_verdict['source_final_verdict']}`",
                f"- capacity_axis_alignment: `{summary['capacity_axis_alignment']}`",
                f"- dashboard_publishable: `{summary['dashboard_publishable']}`",
                "",
                f"**does capacity/depth-growth appear to be a post-birth field rather than a primary class classifier? {'yes' if (summary['capacity_primary_classifier'] is False) else 'no'}**",
                f"**does the signal align more with the P4 axis, the P6 axis, or neither? {summary['capacity_axis_alignment']}**",
                "- Proxy caveat: these are operational post-birth proxies, not theorem-level capacity quantities.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "artifact_root": str(artifact_root),
        "capacity_primary_classifier": bool(summary["capacity_primary_classifier"]),
        "capacity_postcritical_story_supported": bool(summary["capacity_postcritical_story_supported"]),
        "capacity_axis_alignment": str(summary["capacity_axis_alignment"]),
        "dashboard_publishable": bool(summary["dashboard_publishable"]),
        "rows": rep_rows,
    }


__all__ = [
    "load_capacity_dashboard_sources",
    "extract_representative_capacity_rows",
    "summarize_capacity_postcritical_story",
    "build_capacity_postcritical_dashboard",
    "format_capacity_dashboard_summary",
    "_trend_label",
]
