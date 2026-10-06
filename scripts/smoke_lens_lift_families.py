#!/usr/bin/env python3
"""Smoke lens/lift families on deterministic toy inputs."""

from __future__ import annotations

import json

import numpy as np

from layerbirth.lenses import (
    diffusion_quantile_lens,
    manual_partition_lens,
    random_surjective_lens,
    spectral_sign_pattern_lens,
)
from layerbirth.lifts import (
    prototype_lift_family,
    stationary_within_fiber_lift,
    uniform_lift_family,
)
from layerbirth.numeric import pushforward_matrix


def _uq_identity_error(u: np.ndarray, q: np.ndarray) -> float:
    k = u.shape[0]
    return float(np.max(np.abs((u @ q) - np.eye(k))))


def main() -> int:
    p_rev = np.array(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.2, 0.6, 0.2, 0.0],
            [0.0, 0.2, 0.6, 0.2],
            [0.0, 0.0, 0.2, 0.8],
        ]
    )
    p_stat = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.4, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.8],
            [0.0, 0.0, 0.1, 0.9],
        ]
    )

    lens_manual, details_manual = manual_partition_lens([0, 0, 1, 1])
    lens_sign, _ = spectral_sign_pattern_lens(p_rev, tau=1, target_k=2)
    lens_diff, _ = diffusion_quantile_lens(p_rev, tau=1, target_k=2)
    lens_rand_a, _ = random_surjective_lens(n=8, k=3, seed=13)
    lens_rand_b, _ = random_surjective_lens(n=8, k=3, seed=13)
    lens_rand_c, _ = random_surjective_lens(n=8, k=3, seed=17)

    q = pushforward_matrix(lens_manual, 2)
    u_uniform, _ = uniform_lift_family(lens_manual, 2)
    u_prototype, _ = prototype_lift_family(lens_manual, 2, prototype_indices=[1, 3])
    u_stationary, stat_details = stationary_within_fiber_lift(p_stat, lens_manual, 2)

    payload = {
        "manual_lens": {"labels": lens_manual.tolist(), "k": details_manual["k"]},
        "spectral_sign_pattern_lens": {"labels": lens_sign.tolist(), "k": 2},
        "diffusion_quantile_lens": {"labels": lens_diff.tolist(), "k": 2},
        "random_surjective_lens": {
            "seed_13": lens_rand_a.tolist(),
            "same_seed_equal": bool(np.array_equal(lens_rand_a, lens_rand_b)),
            "different_seed_differs": bool(not np.array_equal(lens_rand_a, lens_rand_c)),
        },
        "uniform_lift": {
            "rows": u_uniform.tolist(),
            "UQ_identity_error": _uq_identity_error(u_uniform, q),
        },
        "prototype_lift": {
            "rows": u_prototype.tolist(),
            "UQ_identity_error": _uq_identity_error(u_prototype, q),
        },
        "stationary_within_fiber_lift": {
            "rows": u_stationary.tolist(),
            "UQ_identity_error": _uq_identity_error(u_stationary, q),
            "zero_mass_fibers": stat_details["zero_mass_fibers"],
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
