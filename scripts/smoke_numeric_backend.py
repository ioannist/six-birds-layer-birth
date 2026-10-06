#!/usr/bin/env python3
"""Numeric backend smoke for Toy A and Toy B."""

from __future__ import annotations

import json

import numpy as np

from layerbirth.numeric import (
    empirical_endomap,
    fiber_level_mismatch,
    idempotence_defect_tv,
    macro_kernel,
    packaging_projector,
    pushforward_matrix,
    retention_error,
    tv_distance,
    uniform_lift_matrix,
)


def _run_toy_a() -> dict:
    f = np.array([0, 0, 1, 1], dtype=int)
    q = pushforward_matrix(f, 2)
    u = uniform_lift_matrix(f, 2)
    p = packaging_projector(q, u)
    phat = macro_kernel(p, 1, q, u)
    e = empirical_endomap(p, 1, q, u)
    ret_max, ret_vec = retention_error(p, 1, q, u)
    mismatch_vec, mismatch_max = fiber_level_mismatch(p, 1, f, q, u)
    return {
        "Q_f": q.tolist(),
        "U_f": u.tolist(),
        "packaging_projector": p.tolist(),
        "macro_kernel": phat.tolist(),
        "empirical_endomap": e.tolist(),
        "idempotence_defect_tv": idempotence_defect_tv(e),
        "retention_error_max": ret_max,
        "retention_error_vector": ret_vec.tolist(),
        "fiber_level_mismatch_vector": mismatch_vec.tolist(),
        "fiber_level_mismatch_max": mismatch_max,
    }


def _run_toy_b() -> dict:
    f = np.array([0, 0, 1, 1], dtype=int)
    q = pushforward_matrix(f, 2)
    u = uniform_lift_matrix(f, 2)
    p = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    phat = macro_kernel(p, 1, q, u)
    e = empirical_endomap(p, 1, q, u)
    ret_max, ret_vec = retention_error(p, 1, q, u)
    mismatch_vec, mismatch_max = fiber_level_mismatch(p, 1, f, q, u)
    return {
        "Q_f": q.tolist(),
        "U_f": u.tolist(),
        "packaging_projector": packaging_projector(q, u).tolist(),
        "macro_kernel": phat.tolist(),
        "empirical_endomap": e.tolist(),
        "tv_check": tv_distance(np.array([1.0, 0.0]), np.array([0.5, 0.5])),
        "idempotence_defect_tv": idempotence_defect_tv(e),
        "retention_error_max": ret_max,
        "retention_error_vector": ret_vec.tolist(),
        "fiber_level_mismatch_vector": mismatch_vec.tolist(),
        "fiber_level_mismatch_max": mismatch_max,
    }


def main() -> int:
    payload = {
        "toy_a": _run_toy_a(),
        "toy_b": _run_toy_b(),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
