"""LB-19 capacity/depth-growth extension on confirmed class representatives."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import observed_float, scientific_dumps
from .provenance import implementation_fingerprint
from .campaigns import _git_code_version, _line_plot, _repo_root, _write_csv
from .class3 import _build_class_iv_candidate_substrate, _evaluate_tau_panel, evaluate_class_iii_candidate
from .boundaries import structural_reference_lambda
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import pushforward_matrix
from .sweep import apply_closure_strength_control
from .substrates import build_class_iii_candidate_family, driven_cycle_family, reversible_block_family


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def _manual_half_lens(n: int) -> np.ndarray:
    return np.array([0 if i < (n // 2) else 1 for i in range(n)], dtype=np.int64)


def _build_substrate(class_name: str, size: int, cfg: dict[str, Any]) -> dict[str, Any]:
    if size <= 0 or size % 2:
        raise ValueError("canonical capacity representatives require a positive even size")
    if class_name == "Class-I":
        return reversible_block_family(
            n_blocks=2,
            block_size=int(size // 2),
            intra_block_weight=float(cfg["intra_block_weight"]),
            inter_block_weight=float(cfg["inter_block_weight"]),
            self_weight=float(cfg["self_weight"]),
            topology="ring",
        )
    if class_name == "Class-II":
        return driven_cycle_family(
            n=int(size),
            self_weight=float(cfg["self_weight"]),
            forward_weight=float(cfg["forward_weight"]),
            backward_weight=float(cfg["backward_weight"]),
        )
    if class_name == "Class-III":
        kw = dict(cfg["base_kwargs"])
        kw["n"] = int(size)
        return build_class_iii_candidate_family("replicated_portal_reversible_family", **kw)
    if class_name == "Class-IV":
        return _build_class_iv_candidate_substrate(
            "replicated_portal_reversible_family",
            dict(cfg["base_kwargs"]),
            dict(cfg["drive_kwargs"]),
            int(size),
        )
    raise ValueError(f"unsupported class {class_name}")


def _select_lens(class_name: str, substrate: dict[str, Any]) -> np.ndarray:
    if class_name in {"Class-I", "Class-II"}:
        return _manual_half_lens(int(substrate["n"]))
    return np.asarray(substrate["coarse_lens"], dtype=np.int64)


def _metric_at(substrate: dict[str, Any], lens: np.ndarray, lam: float, tau: int, control_mode: str) -> dict[str, Any]:
    p_base = np.asarray(substrate["P"], dtype=np.float64)
    k = int(np.max(lens)) + 1
    q = pushforward_matrix(lens, k)
    u = np.asarray(build_lift_family("uniform_lift_family", f=lens, k=k)[0], dtype=np.float64)
    p = apply_closure_strength_control(p_base, closure_strength_lambda=float(lam), Q_f=q, U_f=u, mode=control_mode)
    b = default_metric_bundle(p, lens, tau=int(tau))
    return {
        "closure_error": observed_float(b["closure_error"]),
        "objecthood_order": observed_float(b["objecthood_order"]),
        "staging_gap": observed_float(b["staging_gap"]),
        "affinity": observed_float(b["affinity"]),
        "analysis_k": int(b["analysis_k"]),
    }


def extract_birth_lambda_reference(source_bundle_or_rows: dict[str, Any], ce_target: float = 0.025, mobj_target: float = 0.90) -> dict[str, Any]:
    ce = source_bundle_or_rows.get("ce_boundary_lambda")
    mo = source_bundle_or_rows.get("mobj_boundary_lambda")
    birth = structural_reference_lambda(ce, mo)
    if source_bundle_or_rows.get("structural_targets_met_at_reference") is False:
        birth = None
    fallback = None if birth is not None else "joint_birth_unavailable"
    return {
        "ce_target": float(ce_target),
        "mobj_target": float(mobj_target),
        "ce_boundary_lambda": ce,
        "mobj_boundary_lambda": mo,
        "birth_lambda_ref": birth,
        "fallback": fallback,
    }


def load_confirmed_class_references(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    rubric = json.loads((root / "configs/taxonomy/canonical_class_rubric.json").read_text())
    p4 = json.loads((root / "configs/metrics/p4_anomaly_metric_layer.json").read_text())
    refs = {}
    for class_name, family in cfg["representatives"].items():
        campaign_name = {
            "Class-I": "class_i_equilibrium_scaling", "Class-II": "class_ii_driven_scaling",
            "Class-III": "class_iii_full_campaign", "Class-IV": "class_iv_full_campaign",
        }[class_name]
        campaign = json.loads((root / f"configs/campaigns/{campaign_name}.json").read_text())
        if class_name in {"Class-I", "Class-II"}:
            panel = campaign["primary_panel"]
            available = panel["sizes"]
            lambdas = panel["lambda_grid"]
        else:
            available, lambdas = campaign["size_panel"], campaign["lambda_grid"]
        sizes = [int(n) for n in cfg["size_panel"] if int(n) in available]
        if not sizes or len(set(sizes)) != len(sizes):
            raise ValueError("capacity panel requires distinct supported sizes for every class")
        ref = {
            "representative_name": class_name + "_configured_representative",
            "canonical_size_panel": sizes,
            "family_config": family,
            "birth_by_size": {}, "affinity_anchor_by_size": {}, "states_by_size": {},
            "measurement_recomputed": True,
            "implementation_sha256": implementation_fingerprint(),
            "birth_lambda_grid": lambdas,
        }
        for size in sizes:
            sub = _build_substrate(class_name, size, family)
            sub["coarse_lens"] = _select_lens(class_name, sub)
            channels = [_evaluate_tau_panel(sub, lambdas, t, cfg["control_application_name"]) for t in (1, 2)]
            ref["birth_by_size"][size] = extract_birth_lambda_reference(channels[0][1])
            if ref["birth_by_size"][size]["birth_lambda_ref"] is None:
                raise ValueError(f"joint birth unavailable for {class_name} at size {size}")
            ev = evaluate_class_iii_candidate({
                "tau1_boundary": channels[0][1], "tau2_boundary": channels[1][1],
                "tau1_affinity_ref": channels[0][2], "tau2_affinity_ref": channels[1][2],
            }, rubric, p4)
            if ev["canonical_class_label"] != class_name:
                raise ValueError(f"configured capacity representative does not reproduce {class_name} at size {size}")
            ref["affinity_anchor_by_size"][size] = channels[0][2]["affinity_ref"]
            ref["states_by_size"][size] = {
                "p5": ev["candidate_p5_state"], "p6": ev["candidate_p6_drive_state"], "p4": ev["candidate_p4_state"],
            }
            ref["birth_by_size"][size]["left_censored"] = ref["birth_by_size"][size]["birth_lambda_ref"] == min(lambdas)
        refs[class_name] = ref

    return refs


def run_post_birth_tau_panel(reference_config: dict[str, Any], output_root: str | Path | None = None, use_cache: bool = True) -> list[dict[str, Any]]:
    tau_ladder = [int(t) for t in reference_config["tau_ladder"]]
    if not tau_ladder or any(int(t) != t or t <= 0 for t in reference_config["tau_ladder"]) or len(set(tau_ladder)) != len(tau_ladder):
        raise ValueError("tau ladder requires distinct positive integer times")
    control_mode = str(reference_config["control_application_name"])
    lambda_offset = float(reference_config["lambda_post_offset"])
    if not np.isfinite(lambda_offset) or lambda_offset < 0:
        raise ValueError("post-birth lambda offset must be finite and nonnegative")
    rows: list[dict[str, Any]] = []
    for class_name, ref in reference_config["references"].items():
        sizes = [int(s) for s in ref["canonical_size_panel"]]
        for size in sizes:
            substrate = _build_substrate(class_name, size, ref["family_config"])
            lens = _select_lens(class_name, substrate)
            birth_info = ref["birth_by_size"][int(size)]
            birth = float(birth_info["birth_lambda_ref"])
            lambda_post = min(1.0, birth + lambda_offset)
            lambda_sat = 1.0
            for lambda_name, lam in (("post", lambda_post), ("sat", lambda_sat)):
                for tau in tau_ladder:
                    m = _metric_at(substrate, lens, lam, tau, control_mode)
                    rows.append(
                        {
                            "class_name": class_name,
                            "representative_name": ref["representative_name"],
                            "size": int(size),
                            "lambda_name": lambda_name,
                            "lambda_value": float(lam),
                            "tau": int(tau),
                            **m,
                        }
                    )
    return rows


def compute_capacity_depth_proxies(panel_rows: list[dict[str, Any]], birth_lambda_ref: float) -> dict[str, Any]:
    if not np.isfinite(birth_lambda_ref) or not 0 <= birth_lambda_ref <= 1:
        raise ValueError("birth reference must be finite and in [0, 1]")
    if not panel_rows or any(int(r["tau"]) != r["tau"] or r["tau"] <= 0 for r in panel_rows):
        raise ValueError("capacity requires positive integer sampled times")
    tau_vals = sorted({int(r["tau"]) for r in panel_rows})
    post_rows = [r for r in panel_rows if str(r["lambda_name"]) == "post"]
    sat_rows = [r for r in panel_rows if str(r["lambda_name"]) == "sat"]
    p5 = lambda r: observed_float(r["closure_error"]) <= 0.025 and observed_float(r["objecthood_order"]) >= 0.90
    p6_state = lambda r: "active" if observed_float(r["affinity"]) >= 1e-3 else ("inactive" if observed_float(r["affinity"]) <= 1e-6 else "unknown")

    def summaries(rows: list[dict[str, Any]]) -> tuple[int, float]:
        ordered = sorted(rows, key=lambda r: r["tau"])
        if not ordered or len({r["tau"] for r in ordered}) != len(ordered):
            raise ValueError("both capacity channels require distinct observed times")
        if any(not np.isfinite(float(r[key])) for r in ordered for key in ("closure_error", "objecthood_order")):
            raise ValueError("capacity structure measurements must be finite")
        depth = 0
        by_tau = {r["tau"]: r for r in ordered}
        for tau in tau_vals:
            if tau not in by_tau or not p5(by_tau[tau]):
                break
            depth = tau
        # Normalized trapezoidal area over the actual sampled tau interval.
        # A singleton represents only its own value, not an interval estimate.
        if len(ordered) == 1:
            area = observed_float(ordered[0]["objecthood_order"])
        else:
            area = sum((b["tau"] - a["tau"]) * (observed_float(a["objecthood_order"]) + observed_float(b["objecthood_order"])) / 2
                       for a, b in zip(ordered, ordered[1:])) / (ordered[-1]["tau"] - ordered[0]["tau"])
        return int(depth), float(area)

    depth_post, area_post = summaries(post_rows)
    depth_sat, area_sat = summaries(sat_rows)
    headroom = float(1.0 - float(birth_lambda_ref))
    cdi = float(headroom * area_sat * depth_sat)

    return {
        "lambda_headroom": headroom,
        "depth_survival_post": int(depth_post),
        "depth_survival_sat": int(depth_sat),
        "objecthood_tau_area_post": area_post,
        "objecthood_tau_area_sat": area_sat,
        "capacity_depth_index": cdi,
        "p5_states_sat": {int(r["tau"]): p5(r) for r in sat_rows},
        "p6_states_sat": {int(r["tau"]): p6_state(r) for r in sat_rows},
        "tau_ladder": tau_vals,
        "depth_scope": "unbroken prefix of the sampled tau ladder; no interpolation in tau",
        "objecthood_area_definition": "trapezoidal integral divided by sampled tau span",
    }


def summarize_capacity_by_class(proxy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = {}
    for row in proxy_rows:
        by_class.setdefault(str(row["class_name"]), {"rows": []})["rows"].append(row)
    out: dict[str, Any] = {}
    for cls, blk in by_class.items():
        rows = blk["rows"]
        out[cls] = {
            "count": len(rows),
            "lambda_headroom_mean": float(np.mean([float(r["lambda_headroom"]) for r in rows])),
            "depth_survival_sat_mean": float(np.mean([float(r["depth_survival_sat"]) for r in rows])),
            "objecthood_tau_area_sat_mean": float(np.mean([observed_float(r["objecthood_tau_area_sat"]) for r in rows])),
            "capacity_depth_index_mean": float(np.mean([float(r["capacity_depth_index"]) for r in rows])),
        }
    return out


def _axis_effect(values: list[dict[str, Any]], group_a: set[str], group_b: set[str], key: str) -> float:
    rows_a = [r for r in values if str(r["class_name"]) in group_a]
    rows_b = [r for r in values if str(r["class_name"]) in group_b]
    if not rows_a or not rows_b:
        return 0.0
    mu_a = float(np.mean([float(r[key]) for r in rows_a]))
    mu_b = float(np.mean([float(r[key]) for r in rows_b]))
    # pooled within-class variability over sizes
    by_class: dict[str, list[float]] = {}
    for r in rows_a + rows_b:
        by_class.setdefault(str(r["class_name"]), []).append(float(r[key]))
    sse = 0.0
    dof = 0
    for vals in by_class.values():
        if len(vals) > 1:
            m = float(np.mean(vals))
            sse += float(np.sum([(v - m) ** 2 for v in vals]))
            dof += len(vals) - 1
    pooled_std = float(np.sqrt(sse / dof)) if dof > 0 else 0.0
    return float(abs(mu_a - mu_b) / max(pooled_std, 1e-12))


def compute_axis_effects(proxy_rows: list[dict[str, Any]], class_metadata: dict[str, Any]) -> dict[str, Any]:
    # Use the same sizes in all four classes so class effects cannot be an
    # artifact of giving only some classes an additional large system.
    classes = {"Class-I", "Class-II", "Class-III", "Class-IV"}
    matched_sizes = set.intersection(*[{int(r["size"]) for r in proxy_rows if r["class_name"] == c} for c in classes])
    proxy_rows = [r for r in proxy_rows if int(r["size"]) in matched_sizes]
    if class_metadata:
        for cls in classes:
            if cls not in class_metadata:
                raise ValueError("axis comparison requires metadata for all four classes")
    proxies = ["lambda_headroom", "depth_survival_sat", "objecthood_tau_area_sat", "capacity_depth_index"]
    p4_on = {"Class-III", "Class-IV"}
    p4_off = {"Class-I", "Class-II"}
    p6_on = {"Class-II", "Class-IV"}
    p6_off = {"Class-I", "Class-III"}
    per_proxy: dict[str, Any] = {}
    p4_hits = 0
    p6_hits = 0
    for key in proxies:
        e4 = _axis_effect(proxy_rows, p4_on, p4_off, key)
        e6 = _axis_effect(proxy_rows, p6_on, p6_off, key)
        per_proxy[key] = {
            "p4_axis_effect": float(e4),
            "p6_axis_effect": float(e6),
            "p4_dominant_hit": bool(e4 >= 1.0 and e4 >= 1.25 * e6),
            "p6_dominant_hit": bool(e6 >= 1.0 and e6 >= 1.25 * e4),
        }
        if per_proxy[key]["p4_dominant_hit"]:
            p4_hits += 1
        if per_proxy[key]["p6_dominant_hit"]:
            p6_hits += 1
    tracks_p4 = p4_hits >= 2
    tracks_p6 = p6_hits >= 2
    mixed = not tracks_p4 and not tracks_p6
    if tracks_p4:
        verdict = "capacity_tracks_p4_axis"
    elif tracks_p6:
        verdict = "capacity_tracks_p6_axis"
    else:
        verdict = "capacity_orthogonal_or_mixed"
    return {
        "per_proxy": per_proxy,
        "matched_size_panel": sorted(matched_sizes),
        "comparison_available": bool(matched_sizes),
        "axis_evidence_scope": "descriptive dependent proxy contrasts on matched sizes",
        "capacity_tracks_p4_axis": bool(tracks_p4),
        "capacity_tracks_p6_axis": bool(tracks_p6),
        "capacity_orthogonal_or_mixed": bool(mixed),
        "final_verdict": verdict,
        "capacity_primary_classifier": False,
    }


def format_capacity_depth_summary(proxy_rows: list[dict[str, Any]], axis_summary: dict[str, Any], verdicts: dict[str, Any]) -> dict[str, Any]:
    return {
        "proxy_row_count": len(proxy_rows),
        "saturation_interpretation": "lambda=1 installs QU; idempotency implies CE=0, Mobj=1 and affinity=0 for every positive tau",
        "capacity_interpretation": "operational proxy; saturation index reduces to (1-birth_lambda_ref)*max(tau_ladder) under projector mixing",
        "retention_scope": "P5 and P6 only; no P4 retention claim",
        "axis_evidence_scope": "descriptive contrasts of dependent proxies, not independent tests",
        "axis_summary": axis_summary,
        "verdicts": verdicts,
    }


def run_capacity_depth_extension(config_path_or_obj: str | Path | dict[str, Any], output_root: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    cfg = _coerce_config(config_path_or_obj)
    root = _repo_root()
    out = (root / "results" / "campaigns") if output_root is None else Path(output_root)
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

    refs = load_confirmed_class_references(cfg)
    run_cfg = {
        "references": refs,
        "tau_ladder": cfg["tau_ladder"],
        "control_application_name": cfg["control_application_name"],
        "lambda_post_offset": cfg["lambda_post_offset"],
    }
    panel_rows = run_post_birth_tau_panel(run_cfg, output_root=artifact_root, use_cache=use_cache)

    # Per (class,size) proxy rows
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in panel_rows:
        grouped.setdefault((str(r["class_name"]), int(r["size"])), []).append(r)
    proxy_rows: list[dict[str, Any]] = []
    birth_ref_map: dict[str, Any] = {}
    for (cls, size), rows in sorted(grouped.items()):
        birth_info = refs[cls]["birth_by_size"][int(size)]
        birth = float(birth_info["birth_lambda_ref"])
        birth_ref_map.setdefault(cls, {})[str(size)] = birth_info
        p = compute_capacity_depth_proxies(rows, birth)
        expected = refs[cls]["states_by_size"][size]
        sat_rows = [r for r in rows if str(r["lambda_name"]) == "sat"]
        retain = []
        for sr in sat_rows:
            p5 = bool(observed_float(sr["closure_error"]) <= 0.025 and observed_float(sr["objecthood_order"]) >= 0.90)
            p6 = "active" if observed_float(sr["affinity"]) >= 1e-3 else ("inactive" if observed_float(sr["affinity"]) <= 1e-6 else "unknown")
            retain.append(bool(("active" if p5 else "inactive") == expected["p5"] and p6 == expected["p6"]))
        retention = float(np.mean([1.0 if x else 0.0 for x in retain])) if retain else 0.0
        proxy_rows.append(
            {
                "class_name": cls,
                "representative_name": refs[cls]["representative_name"],
                "size": int(size),
                "birth_lambda_ref": float(birth),
                "lambda_headroom": float(p["lambda_headroom"]),
                "depth_survival_post": int(p["depth_survival_post"]),
                "depth_survival_sat": int(p["depth_survival_sat"]),
                "objecthood_tau_area_post": observed_float(p["objecthood_tau_area_post"]),
                "objecthood_tau_area_sat": observed_float(p["objecthood_tau_area_sat"]),
                "capacity_depth_index": float(p["capacity_depth_index"]),
                "p5_p6_retention_fraction": float(retention),
                "cache_status": "executed",
                "manifest_path": "manifest.json",
            }
        )

    capacity_summary = summarize_capacity_by_class(proxy_rows)
    axis_summary = compute_axis_effects(proxy_rows, refs)
    final_verdict = {
        "capacity_tracks_p4_axis": bool(axis_summary["capacity_tracks_p4_axis"]),
        "capacity_tracks_p6_axis": bool(axis_summary["capacity_tracks_p6_axis"]),
        "capacity_orthogonal_or_mixed": bool(axis_summary["capacity_orthogonal_or_mixed"]),
        "capacity_primary_classifier": False,
        "final_verdict": str(axis_summary["final_verdict"]),
    }
    merged = format_capacity_depth_summary(proxy_rows, axis_summary, final_verdict)

    # Write artifacts
    (dirs["analysis"] / "capacity_depth_summary.json").write_text(scientific_dumps(merged, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "reference_birth_lambdas.json").write_text(scientific_dumps(birth_ref_map, indent=2) + "\n", encoding="utf-8")
    _write_csv(
        dirs["analysis"] / "post_birth_tau_panel.csv",
        panel_rows,
        [
            "class_name",
            "representative_name",
            "size",
            "lambda_name",
            "lambda_value",
            "tau",
            "closure_error",
            "objecthood_order",
            "staging_gap",
            "affinity",
            "analysis_k",
        ],
    )
    (dirs["analysis"] / "capacity_proxy_summary.json").write_text(scientific_dumps(capacity_summary, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "axis_effect_summary.json").write_text(scientific_dumps(axis_summary, indent=2) + "\n", encoding="utf-8")
    (dirs["analysis"] / "final_verdict.json").write_text(scientific_dumps(final_verdict, indent=2) + "\n", encoding="utf-8")
    _write_csv(
        dirs["metrics"] / "metrics.csv",
        proxy_rows,
        [
            "class_name",
            "representative_name",
            "size",
            "birth_lambda_ref",
            "lambda_headroom",
            "depth_survival_post",
            "depth_survival_sat",
            "objecthood_tau_area_post",
            "objecthood_tau_area_sat",
            "capacity_depth_index",
            "p5_p6_retention_fraction",
            "cache_status",
            "manifest_path",
        ],
    )

    # Plots
    class_order = ["Class-I", "Class-II", "Class-III", "Class-IV"]
    for cls in class_order:
        rows = sorted([r for r in proxy_rows if str(r["class_name"]) == cls], key=lambda x: int(x["size"]))
        if not rows:
            continue
        x = [float(r["size"]) for r in rows]
        _line_plot(
            dirs["plots"] / "capacity_depth_index_vs_size.png",
            x,
            {cls: [float(r["capacity_depth_index"]) for r in rows]},
        )
    # Multi-series depth survival and class proxy comparisons
    x_sizes = sorted({int(r["size"]) for r in proxy_rows})
    depth_series: dict[str, list[float]] = {}
    cdi_series: dict[str, list[float]] = {}
    for cls in class_order:
        rows = {int(r["size"]): r for r in proxy_rows if str(r["class_name"]) == cls}
        depth_series[f"{cls} depth_sat"] = [float(rows[s]["depth_survival_sat"]) if s in rows else float("nan") for s in x_sizes]
        cdi_series[f"{cls} cdi"] = [float(rows[s]["capacity_depth_index"]) if s in rows else float("nan") for s in x_sizes]
    _line_plot(dirs["plots"] / "depth_survival_vs_size.png", [float(s) for s in x_sizes], depth_series)
    _line_plot(dirs["plots"] / "class_proxy_comparison.png", [float(s) for s in x_sizes], cdi_series)
    pkeys = ["lambda_headroom", "depth_survival_sat", "objecthood_tau_area_sat", "capacity_depth_index"]
    _line_plot(
        dirs["plots"] / "axis_effect_comparison.png",
        [0.0, 1.0, 2.0, 3.0],
        {
            "p4_axis_effect": [float(axis_summary["per_proxy"][k]["p4_axis_effect"]) for k in pkeys],
            "p6_axis_effect": [float(axis_summary["per_proxy"][k]["p6_axis_effect"]) for k in pkeys],
        },
    )

    (dirs["config"] / "config_snapshot.json").write_text(scientific_dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dirs["seeds"] / "seeds.json").write_text("[]\n", encoding="utf-8")
    (dirs["env"] / "environment.json").write_text(scientific_dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")

    note_lines = [
        "# LB-19 Capacity / depth-growth extension",
        "",
        f"- config path: `configs/campaigns/capacity_depth_extension.json`",
        f"- artifact root: `{artifact_root}`",
        f"- representatives used: Class-I `{refs['Class-I']['representative_name']}`, Class-II `{refs['Class-II']['representative_name']}`, Class-III `{refs['Class-III']['representative_name']}`, Class-IV `{refs['Class-IV']['representative_name']}`",
        f"- proxy summary: `{capacity_summary}`",
        f"- axis-effect summary: `{axis_summary}`",
        f"- final verdict: `{final_verdict['final_verdict']}`",
        "- does capacity/depth-growth appear to track class structure, or is it largely orthogonal/mixed? "
        f"`{'tracks class structure' if final_verdict['final_verdict']!='capacity_orthogonal_or_mixed' else 'largely orthogonal/mixed'}`",
        "- does the signal align more with the P4 axis, the P6 axis, or neither? "
        + (
            "`P4 axis`" if final_verdict["capacity_tracks_p4_axis"] else ("`P6 axis`" if final_verdict["capacity_tracks_p6_axis"] else "`neither`")
        ),
        "- these are operational post-birth proxies, not theorem-level capacity quantities.",
    ]
    note_text = "\n".join(note_lines) + "\n"
    note_path = root / str(cfg.get("findings_note_path", "notes/findings/LB-19_capacity_depth_extension.md"))
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_text, encoding="utf-8")
    (dirs["notes"] / "findings.md").write_text(note_text, encoding="utf-8")

    schema = load_schema(root / "configs/result_bundle.schema.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "capacity_depth_extension",
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
        **final_verdict,
        "artifact_root": str(artifact_root),
    }
