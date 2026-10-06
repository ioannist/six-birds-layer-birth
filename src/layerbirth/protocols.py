"""Tau protocols and continuous control resolution for layer-birth runs."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .numeric import validate_row_stochastic


def resolve_control_settings(config: dict[str, Any]) -> dict[str, Any]:
    control = config.get("control", {})
    if not isinstance(control, dict):
        raise ValueError("control must be an object")
    if "closure_strength_lambda" not in control:
        raise ValueError("control.closure_strength_lambda is required")
    lam = control["closure_strength_lambda"]
    if not isinstance(lam, (int, float)):
        raise ValueError("closure_strength_lambda must be numeric")
    lam = float(lam)
    if not math.isfinite(lam):
        raise ValueError("closure_strength_lambda must be finite")
    if lam < 0.0 or lam > 1.0:
        raise ValueError("closure_strength_lambda must be in [0,1]")
    return {
        "closure_strength_lambda": lam,
        "control_details": {
            "closure_strength_lambda": lam,
            "range_validation": "[0,1]",
        },
    }


def resolve_fixed_tau(config: dict[str, Any]) -> dict[str, Any]:
    protocol = config.get("tau_protocol", {})
    if not isinstance(protocol, dict) or protocol.get("name") != "fixed":
        raise ValueError("tau_protocol.name must be 'fixed'")
    tau = protocol.get("tau")
    if not isinstance(tau, int) or tau <= 0:
        raise ValueError("fixed tau must be a positive integer")
    ctrl = resolve_control_settings(config)
    return {
        "run_id": config.get("run_id", "unknown"),
        "tau_protocol_name": "fixed",
        "resolved_tau": int(tau),
        "tau_resolution_details": {
            "fixed_tau": int(tau),
        },
        "closure_strength_lambda": ctrl["closure_strength_lambda"],
        "control_details": ctrl["control_details"],
    }


def _adaptive_details(protocol: dict[str, Any], P: np.ndarray) -> dict[str, Any]:
    validate_row_stochastic(P)
    if protocol.get("method", "spectral_gap_scaled") != "spectral_gap_scaled":
        raise ValueError("unsupported adaptive method")
    alpha = float(protocol.get("alpha", 1.0))
    gap_floor = float(protocol.get("gap_floor", 1e-8))
    tau_min = int(protocol.get("tau_min", 1))
    tau_max = int(protocol.get("tau_max", 32))
    if tau_min <= 0 or tau_max < tau_min:
        raise ValueError("invalid tau bounds for adaptive protocol")
    vals = np.linalg.eigvals(np.asarray(P, dtype=np.float64))
    abs_vals = np.sort(np.abs(vals))[::-1]
    mu2 = 0.0 if abs_vals.size < 2 else float(abs_vals[1])
    spectral_gap = float(max(0.0, 1.0 - mu2))
    # Numerical guard prevents 5.000000000000001 from ceiling to 6.
    tau_raw = int(math.ceil((alpha / max(spectral_gap, gap_floor)) - 1e-12))
    resolved = int(min(max(tau_raw, tau_min), tau_max))
    details = {
        "method": "spectral_gap_scaled",
        "alpha": alpha,
        "gap_floor": gap_floor,
        "mu2_abs": mu2,
        "spectral_gap": spectral_gap,
        "tau_raw": tau_raw,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "clipped": resolved != tau_raw,
    }
    return {"resolved_tau": resolved, "tau_resolution_details": details}


def resolve_adaptive_tau(config: dict[str, Any], P: np.ndarray) -> dict[str, Any]:
    protocol = config.get("tau_protocol", {})
    if not isinstance(protocol, dict) or protocol.get("name") != "adaptive":
        raise ValueError("tau_protocol.name must be 'adaptive'")
    if P is None:
        raise ValueError("adaptive protocol requires kernel P")
    adaptive = _adaptive_details(protocol, np.asarray(P, dtype=np.float64))
    ctrl = resolve_control_settings(config)
    return {
        "run_id": config.get("run_id", "unknown"),
        "tau_protocol_name": "adaptive",
        "resolved_tau": adaptive["resolved_tau"],
        "tau_resolution_details": adaptive["tau_resolution_details"],
        "closure_strength_lambda": ctrl["closure_strength_lambda"],
        "control_details": ctrl["control_details"],
    }


def resolve_matched_tau_group(
    run_configs: list[dict[str, Any]],
    kernels_by_run_id: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    if not run_configs:
        raise ValueError("run_configs cannot be empty")
    group_ids = {
        cfg.get("tau_protocol", {}).get("match_group_id")
        for cfg in run_configs
        if isinstance(cfg.get("tau_protocol"), dict)
    }
    if len(group_ids) != 1:
        raise ValueError("matched group must have exactly one match_group_id")
    match_group_id = next(iter(group_ids))
    refs = [
        cfg
        for cfg in run_configs
        if cfg.get("tau_protocol", {}).get("name") == "matched"
        and cfg.get("tau_protocol", {}).get("role") == "reference"
    ]
    if len(refs) != 1:
        raise ValueError("matched group must contain exactly one reference run")
    ref_cfg = refs[0]
    ref_run_id = str(ref_cfg.get("run_id"))
    ref_protocol = ref_cfg["tau_protocol"]
    base_protocol = ref_protocol.get("base_protocol")
    if not isinstance(base_protocol, dict):
        raise ValueError("matched reference must include base_protocol")

    base_cfg = copy.deepcopy(ref_cfg)
    base_cfg["tau_protocol"] = base_protocol
    if base_protocol.get("name") == "fixed":
        ref_resolved = resolve_fixed_tau(base_cfg)
    elif base_protocol.get("name") == "adaptive":
        if ref_run_id not in kernels_by_run_id:
            raise ValueError(f"missing kernel for reference run {ref_run_id}")
        ref_resolved = resolve_adaptive_tau(base_cfg, kernels_by_run_id[ref_run_id])
    else:
        raise ValueError("matched reference base_protocol must be fixed or adaptive")

    out: dict[str, dict[str, Any]] = {}
    for cfg in run_configs:
        protocol = cfg.get("tau_protocol", {})
        if protocol.get("name") != "matched":
            raise ValueError("all runs in matched resolver must use matched protocol")
        run_id = str(cfg.get("run_id"))
        role = str(protocol.get("role"))
        ctrl = resolve_control_settings(cfg)
        if role == "reference":
            resolved_tau = int(ref_resolved["resolved_tau"])
            matched_reference_run_id = ref_run_id
        elif role == "follower":
            expected_ref = str(protocol.get("reference_run_id"))
            if expected_ref != ref_run_id:
                raise ValueError("follower reference_run_id does not match group reference")
            resolved_tau = int(ref_resolved["resolved_tau"])
            matched_reference_run_id = expected_ref
        else:
            raise ValueError("matched role must be 'reference' or 'follower'")
        out[run_id] = {
            "run_id": run_id,
            "tau_protocol_name": "matched",
            "resolved_tau": resolved_tau,
            "tau_resolution_details": {
                "match_group_id": match_group_id,
                "matched_reference_run_id": matched_reference_run_id,
                "matched_role": role,
                "reference_tau_resolution_details": ref_resolved["tau_resolution_details"],
            },
            "closure_strength_lambda": ctrl["closure_strength_lambda"],
            "control_details": ctrl["control_details"],
            "match_group_id": match_group_id,
            "matched_reference_run_id": matched_reference_run_id,
            "matched_role": role,
        }
    return out


def resolve_run_settings(
    config: dict[str, Any],
    P: np.ndarray | None = None,
    group_context: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    protocol = config.get("tau_protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("tau_protocol must be an object")
    name = protocol.get("name")
    if name == "fixed":
        return resolve_fixed_tau(config)
    if name == "adaptive":
        if P is None:
            raise ValueError("adaptive protocol requires P")
        return resolve_adaptive_tau(config, P)
    if name == "matched":
        if group_context is None:
            raise ValueError("matched protocol requires group_context")
        run_id = str(config.get("run_id"))
        if run_id not in group_context:
            raise ValueError(f"run_id {run_id} not found in matched group_context")
        return group_context[run_id]
    raise ValueError(f"Unknown tau protocol: {name}")
