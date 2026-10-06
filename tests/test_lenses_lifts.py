import numpy as np
import numpy.testing as npt
import pytest

from layerbirth.lenses import (
    build_lens_family,
    diffusion_quantile_lens,
    manual_partition_lens,
    random_surjective_lens,
    spectral_sign_pattern_lens,
)
from layerbirth.lifts import (
    build_lift_family,
    prototype_lift_family,
    stationary_within_fiber_lift,
    uniform_lift_family,
)
from layerbirth.numeric import pushforward_matrix, validate_lens


P_REV = np.array(
    [
        [0.8, 0.2, 0.0, 0.0],
        [0.2, 0.6, 0.2, 0.0],
        [0.0, 0.2, 0.6, 0.2],
        [0.0, 0.0, 0.2, 0.8],
    ]
)

P_STAT = np.array(
    [
        [0.9, 0.1, 0.0, 0.0],
        [0.4, 0.6, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.8],
        [0.0, 0.0, 0.1, 0.9],
    ]
)


def _uq_identity_error(u: np.ndarray, q: np.ndarray) -> float:
    k = u.shape[0]
    return float(np.max(np.abs((u @ q) - np.eye(k))))


def test_manual_partition_lens_canonicalization_and_validation():
    lens, details = manual_partition_lens([7, 7, 3, 3])
    npt.assert_array_equal(lens, np.array([0, 0, 1, 1]))
    assert details["k"] == 2
    validate_lens(lens, 2)

    with pytest.raises(ValueError):
        manual_partition_lens([0, 0, 0], k=2)


def test_spectral_and_diffusion_lenses_on_reversible_toy():
    lens_sign, _ = spectral_sign_pattern_lens(P_REV, tau=1, target_k=2)
    lens_diff, _ = diffusion_quantile_lens(P_REV, tau=1, target_k=2)
    npt.assert_array_equal(lens_sign, np.array([0, 0, 1, 1]))
    npt.assert_array_equal(lens_diff, np.array([0, 0, 1, 1]))
    validate_lens(lens_sign, 2)
    validate_lens(lens_diff, 2)


def test_spectral_and_diffusion_determinism():
    lens1, _ = spectral_sign_pattern_lens(P_REV, tau=1, target_k=2)
    lens2, _ = spectral_sign_pattern_lens(P_REV, tau=1, target_k=2)
    npt.assert_array_equal(lens1, lens2)

    d1, _ = diffusion_quantile_lens(P_REV, tau=1, target_k=2)
    d2, _ = diffusion_quantile_lens(P_REV, tau=1, target_k=2)
    npt.assert_array_equal(d1, d2)


def test_random_surjective_lens_reproducibility_and_variation():
    lens1, _ = random_surjective_lens(n=8, k=3, seed=13)
    lens2, _ = random_surjective_lens(n=8, k=3, seed=13)
    lens3, _ = random_surjective_lens(n=8, k=3, seed=17)
    npt.assert_array_equal(lens1, lens2)
    assert not np.array_equal(lens1, lens3)
    validate_lens(lens1, 3)

    with pytest.raises(ValueError):
        random_surjective_lens(n=3, k=4, seed=0)


def test_uniform_and_prototype_lifts_with_identity_check():
    lens = np.array([0, 0, 1, 1], dtype=int)
    q = pushforward_matrix(lens, 2)
    u_uniform, details_u = uniform_lift_family(lens, 2)
    assert details_u["UQ_identity_error"] <= 1e-12
    assert _uq_identity_error(u_uniform, q) <= 1e-12

    u_proto, details_p = prototype_lift_family(lens, 2, prototype_indices=[1, 3])
    expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    npt.assert_allclose(u_proto, expected, atol=1e-12)
    assert details_p["UQ_identity_error"] <= 1e-12
    assert _uq_identity_error(u_proto, q) <= 1e-12

    with pytest.raises(ValueError):
        prototype_lift_family(lens, 2, prototype_indices=[2, 3])


def test_stationary_within_fiber_lift_nonuniform_rows():
    lens = np.array([0, 0, 1, 1], dtype=int)
    q = pushforward_matrix(lens, 2)
    u_stat, details = stationary_within_fiber_lift(P_STAT, lens, 2, tau=1)
    expected = np.array(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 9.0, 8.0 / 9.0],
        ]
    )
    npt.assert_allclose(u_stat, expected, atol=1e-10)
    assert details["UQ_identity_error"] <= 1e-10
    assert _uq_identity_error(u_stat, q) <= 1e-10


def test_dispatchers_cover_required_families():
    lens_manual, _ = build_lens_family("manual_partition_lens", labels=[0, 0, 1, 1])
    lens_sign, _ = build_lens_family(
        "spectral_sign_pattern_lens", P=P_REV, tau=1, target_k=2
    )
    lens_diff, _ = build_lens_family("diffusion_quantile_lens", P=P_REV, tau=1, target_k=2)
    lens_rand, _ = build_lens_family("random_surjective_lens", n=6, k=2, seed=5)
    validate_lens(lens_manual, 2)
    validate_lens(lens_sign, 2)
    validate_lens(lens_diff, 2)
    validate_lens(lens_rand, 2)

    u_u, _ = build_lift_family("uniform_lift_family", f=lens_manual, k=2)
    u_p, _ = build_lift_family(
        "prototype_lift_family", f=lens_manual, k=2, prototype_indices=[0, 2]
    )
    u_s, _ = build_lift_family("stationary_within_fiber_lift", P=P_STAT, f=lens_manual, k=2)
    q = pushforward_matrix(lens_manual, 2)
    assert _uq_identity_error(u_u, q) <= 1e-12
    assert _uq_identity_error(u_p, q) <= 1e-12
    assert _uq_identity_error(u_s, q) <= 1e-10
