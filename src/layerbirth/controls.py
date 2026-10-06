"""Control cases for arrow-of-time sanity checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .serialization import observed_float
from .lenses import spectral_sign_pattern_lens
from .metrics import default_metric_bundle, driven_candidate
from .numeric import validate_row_stochastic
from .substrates import build_substrate_family


def _as_float_matrix(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def protocol_trap_control(
    *,
    pair_self_weight: float = 0.2,
    schedule_order: list[str] | None = None,
    handling: str | None = None,
) -> dict[str, Any]:
    if not (0.0 <= pair_self_weight <= 1.0):
        raise ValueError("pair_self_weight must be in [0, 1]")
    if schedule_order is None:
        schedule_order = ["pair01", "pair12", "pair20"]
    if len(schedule_order) == 0:
        raise ValueError("schedule_order must be non-empty")
    expected = {"pair01", "pair12", "pair20"}
    if len(schedule_order) != len(expected) or set(schedule_order) != expected:
        raise ValueError("schedule_order must contain pair01, pair12, pair20 exactly once")
    if handling is not None and handling not in {"hidden_schedule", "phase_aware"}:
        raise ValueError("handling must be one of hidden_schedule, phase_aware")

    a = float(pair_self_weight)
    p01 = _as_float_matrix([[a, 1.0 - a, 0.0], [1.0 - a, a, 0.0], [0.0, 0.0, 1.0]])
    p12 = _as_float_matrix([[1.0, 0.0, 0.0], [0.0, a, 1.0 - a], [0.0, 1.0 - a, a]])
    p20 = _as_float_matrix([[a, 0.0, 1.0 - a], [0.0, 1.0, 0.0], [1.0 - a, 0.0, a]])
    phase_kernels = {"pair01": p01, "pair12": p12, "pair20": p20}
    for kernel in phase_kernels.values():
        validate_row_stochastic(kernel)
    hidden = phase_kernels[schedule_order[0]]
    for phase in schedule_order[1:]:
        hidden = hidden @ phase_kernels[phase]
    validate_row_stochastic(hidden)
    analysis_lens = np.array([0, 1, 2], dtype=np.int64)
    return {
        "control_name": "protocol_trap_control",
        "phase_kernels": phase_kernels,
        "hidden_effective_kernel": hidden,
        "analysis_lens": analysis_lens,
        "details": {
            "pair_self_weight": a,
            "schedule_order": schedule_order,
            "supported_handling_modes": ["hidden_schedule", "phase_aware"],
            "requested_handling": handling,
        },
    }


def no_fake_arrow_controls(
    *, cases_config: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if cases_config is None:
        cases_config = [
            {
                "case_name": "reversible_block_manual",
                "family": "reversible_block_family",
                "family_params": {
                    "n_blocks": 2,
                    "block_size": 4,
                    "intra_block_weight": 1.0,
                    "inter_block_weight": 0.05,
                    "self_weight": 0.0,
                },
                "lens_mode": "manual",
                "manual_lens": [0, 0, 0, 0, 1, 1, 1, 1],
            },
            {
                "case_name": "reversible_block_spectral",
                "family": "reversible_block_family",
                "family_params": {
                    "n_blocks": 2,
                    "block_size": 4,
                    "intra_block_weight": 1.0,
                    "inter_block_weight": 0.05,
                    "self_weight": 0.0,
                },
                "lens_mode": "spectral_sign_pattern",
                "target_k": 2,
                "tau": 1,
            },
            {
                "case_name": "null_flat_mixing_manual",
                "family": "null_flat_mixing_family",
                "family_params": {"n": 8},
                "lens_mode": "manual",
                "manual_lens": [0, 0, 0, 0, 1, 1, 1, 1],
            },
        ]

    built: list[dict[str, Any]] = []
    for case in cases_config:
        mode = case.get("lens_mode")
        substrate = build_substrate_family(case["family"], **case.get("family_params", {}))
        p = np.asarray(substrate["P"], dtype=np.float64)
        validate_row_stochastic(p)
        if mode == "manual":
            if "manual_lens" not in case:
                raise ValueError("manual lens_mode requires manual_lens")
            lens = np.asarray(case["manual_lens"], dtype=np.int64)
        elif mode == "spectral_sign_pattern":
            lens, _ = spectral_sign_pattern_lens(
                p,
                tau=int(case.get("tau", 1)),
                target_k=int(case.get("target_k", 2)),
            )
        else:
            raise ValueError("unsupported lens_mode in no_fake_arrow_controls")
        built.append(
            {
                "case_name": case["case_name"],
                "P": p,
                "analysis_lens": lens,
                "details": {
                    "family": case["family"],
                    "family_params": case.get("family_params", {}),
                    "lens_mode": mode,
                },
            }
        )
    return built


def evaluate_control_case(
    *,
    case_name: str,
    control_family: str,
    P: np.ndarray,
    analysis_lens: np.ndarray | list[int],
    tau: int = 1,
    holonomy_inputs: dict[str, Any] | None = None,
    affinity_threshold: float = 1e-6,
) -> dict[str, Any]:
    bundle = default_metric_bundle(
        P,
        analysis_lens,
        tau=tau,
        lift_name="uniform",
        lift_kwargs=None,
        holonomy_inputs=holonomy_inputs,
    )
    return {
        "case_name": case_name,
        "control_family": control_family,
        "n": int(np.asarray(P).shape[0]),
        "analysis_k": int(bundle["analysis_k"]),
        "closure_error": observed_float(bundle["closure_error"]),
        "objecthood_order": observed_float(bundle["objecthood_order"]),
        "staging_gap": observed_float(bundle["staging_gap"]),
        "affinity": observed_float(bundle["affinity"]),
        "holonomy": bundle["holonomy"],
        "driven_candidate": bool(driven_candidate(bundle, affinity_threshold=affinity_threshold)),
    }
