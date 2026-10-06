"""Replay exact canonical classifications and compare with production metrics."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from layerbirth.class3 import _build_class_iv_candidate_substrate, _evaluate_tau_panel
from layerbirth.exact_audit import audit_panel, canonical_kernel
from layerbirth.substrates import build_class_iii_candidate_family, build_substrate_family
from layerbirth.taxonomy import _build_curve_rows_for_profile


ROOT = Path(__file__).resolve().parents[1]


def read_config(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(), parse_float=Fraction)


def floating(config):
    if isinstance(config, Fraction):
        return float(config)
    if isinstance(config, dict):
        return {k: floating(v) for k, v in config.items()}
    if isinstance(config, list):
        return [floating(v) for v in config]
    return config


def check_metrics(exact, production_curves):
    discrepancy = 0.0
    for tau in (1, 2):
        rows = production_curves[tau]
        if len(rows) != len(exact["curves"][tau]):
            raise AssertionError("grid coverage mismatch")
        for e, r in zip(exact["curves"][tau], rows):
            if float(e["lambda"]) != r["closure_strength_lambda"]:
                raise AssertionError("grid coordinate mismatch")
            for exact_key, measured_key in (("ce", "closure_error"), ("mobj", "objecthood_order")):
                discrepancy = max(discrepancy, abs(float(e[exact_key])-r[measured_key]))
    if discrepancy > 1e-12:
        raise AssertionError(f"production metrics disagree with rational derivation: {discrepancy}")
    return discrepancy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "notes/math_review/canonical_exact_audit.json")
    args = parser.parse_args()
    paths = ["configs/taxonomy/canonical_class_rubric.json", "configs/campaigns/class_iii_full_campaign.json",
             "configs/campaigns/class_iv_full_campaign.json"]
    rubric, c3, c4 = [read_config(p) for p in paths]
    expected_signatures = {
        "Class-I": {"P5_active": True, "P6_drive_active": False, "P4_anomalous": False},
        "Class-II": {"P5_active": True, "P6_drive_active": True, "P4_anomalous": False},
        "Class-III": {"P5_active": True, "P6_drive_active": False, "P4_anomalous": True},
        "Class-IV": {"P5_active": True, "P6_drive_active": True, "P4_anomalous": True},
    }
    if rubric["canonical_class_signatures"] != expected_signatures:
        raise ValueError("declared canonical signatures changed")
    if [p["profile_name"] for p in rubric["reference_profiles"]] != ["class_i_reference", "class_ii_reference"]:
        raise ValueError("declared canonical reference profiles changed")
    thresholds = rubric["activation_thresholds"]
    if (thresholds["p5"]["ce_target"] != Fraction(1,40)
            or thresholds["p5"]["mobj_target"] != Fraction(9,10)
            or thresholds["p6_drive"]["p6_active_min"] != Fraction(1,1000)
            or thresholds["p6_drive"]["p6_inactive_max"] != Fraction(1,1000000)
            or thresholds["p4_anomalous"]["p4_class_dual_shift_min"] != Fraction(3,20)):
        raise ValueError("certificate thresholds differ from the declared rubric")
    records = []
    for profile, label in zip(rubric["reference_profiles"], ("Class-I", "Class-II")):
        if profile["control_application_name"] != "mix_with_packaging_projector":
            raise ValueError("unsupported canonical control")
        p = canonical_kernel(profile["family_name"], profile["size"], profile)
        keys = ("n_blocks", "block_size", "intra_block_weight", "inter_block_weight", "self_weight") if label == "Class-I" else (
            "self_weight", "forward_weight", "backward_weight")
        params = {key: floating(profile[key]) for key in keys}
        if label == "Class-II":
            params["n"] = profile["size"]
        substrate = build_substrate_family(profile["family_name"], **params)
        discrepancy_kernel = float(np.max(np.abs(np.array(p, dtype=float) - substrate["P"])))
        if discrepancy_kernel > 1e-12:
            raise AssertionError("independent reference kernel disagrees")
        exact = audit_panel(p, profile["lambda_grid"])
        production = {tau: _build_curve_rows_for_profile(floating(profile), tau) for tau in (1,2)}
        discrepancy = check_metrics(exact, production)
        if exact["canonical_class_label"] != label:
            raise AssertionError(f"canonical claim failed: {label}")
        records.append({"representative": profile["profile_name"], "size": profile["size"],
                        "production_kernel_max_abs_error": discrepancy_kernel,
                        "production_metric_max_abs_error": discrepancy, **exact})
    for cfg, label in ((c3, "Class-III"), (c4, "Class-IV")):
        if cfg["control_application_name"] != "mix_with_packaging_projector":
            raise ValueError("unsupported canonical control")
        candidate = cfg["candidate"]
        for size in cfg["size_panel"]:
            p = canonical_kernel(candidate["family_name"], size, candidate["base_kwargs"], candidate.get("drive_kwargs"))
            kw = floating(candidate["base_kwargs"])
            if label == "Class-III":
                substrate = build_class_iii_candidate_family(candidate["family_name"], n=size, **kw)
            else:
                substrate = _build_class_iv_candidate_substrate(candidate["family_name"], kw, floating(candidate["drive_kwargs"]), size)
            discrepancy_kernel = float(np.max(np.abs(np.array(p, dtype=float) - substrate["P"])))
            if discrepancy_kernel > 1e-12:
                raise AssertionError("independent kernel construction disagrees")
            exact = audit_panel(p, cfg["lambda_grid"])
            production = {tau: _evaluate_tau_panel(substrate, floating(cfg["lambda_grid"]), tau, cfg["control_application_name"])[0]
                          for tau in (1, 2)}
            discrepancy = check_metrics(exact, production)
            if exact["canonical_class_label"] != label:
                raise AssertionError(f"canonical claim failed: {label} at size {size}")
            records.append({"representative": candidate["candidate_name"], "size": size,
                            "production_kernel_max_abs_error": discrepancy_kernel,
                            "production_metric_max_abs_error": discrepancy, **exact})
            print(f"Certified {label} at size {size}", flush=True)
    # Rational values remain exact in the persisted record as numerator/denominator.
    report = {"arithmetic": "fractions.Fraction; configuration decimals interpreted as rationals",
              "scope": "declared finite grids and size panels, manual equal-block uniform lift; interpolated boundary proxies",
              "config_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
              "all_declared_canonical_labels_certified": True, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=lambda v: str(v) if isinstance(v, Fraction) else None) + "\n")
    print(f"Exact audit written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
