#!/usr/bin/env python3
"""Smoke the LB-05 metric contract on required toy systems."""

from __future__ import annotations

import json

import numpy as np

from layerbirth.metrics import (
    affinity_metric,
    closure_error,
    holonomy_metric,
    objecthood_order_parameter,
    staging_gap,
)
from layerbirth.numeric import packaging_projector, pushforward_matrix, uniform_lift_matrix


def _run_common(p: np.ndarray, f: np.ndarray, k: int) -> dict:
    ce, ce_details = closure_error(p, 1, f, k)
    mobj, mobj_details = objecthood_order_parameter(p, 1, f, k)
    sg, sg_details = staging_gap(p, 1, f, k)
    aff, aff_details = affinity_metric(p, tau=1)
    return {
        "CE": ce,
        "CE_details": ce_details,
        "M_obj": mobj,
        "M_obj_details": mobj_details,
        "SG": sg,
        "SG_details": sg_details,
        "Aff": aff,
        "Aff_details": aff_details,
    }


def main() -> int:
    f4 = np.array([0, 0, 1, 1], dtype=int)
    q4 = pushforward_matrix(f4, 2)
    u4 = uniform_lift_matrix(f4, 2)
    p_exact = packaging_projector(q4, u4)
    p_rev = np.array(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.2, 0.6, 0.2, 0.0],
            [0.0, 0.2, 0.6, 0.2],
            [0.0, 0.0, 0.2, 0.8],
        ]
    )
    p_drive = np.array(
        [
            [0.1, 0.8, 0.0, 0.1],
            [0.1, 0.1, 0.8, 0.0],
            [0.0, 0.1, 0.1, 0.8],
            [0.8, 0.0, 0.1, 0.1],
        ]
    )

    f_triv = np.array([0, 0, 0], dtype=int)
    p_triv = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    hol_exact, hol_exact_details = holonomy_metric(f4, 2, np.array([0, 0, 0, 0]), 1)
    hol_aux, hol_aux_details = holonomy_metric(
        np.array([0, 0, 1, 2, 2, 2], dtype=int),
        3,
        np.array([0, 0, 0, 1, 1, 1], dtype=int),
        2,
    )

    payload = {
        "exact_closure": {
            **_run_common(p_exact, f4, 2),
            "Hol": hol_exact,
            "Hol_details": hol_exact_details,
        },
        "reversible": _run_common(p_rev, f4, 2),
        "driven_cycle": _run_common(p_drive, f4, 2),
        "trivial_one_object": _run_common(p_triv, f_triv, 1),
        "aux_holonomy": {
            "Hol": hol_aux,
            "Hol_details": hol_aux_details,
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
