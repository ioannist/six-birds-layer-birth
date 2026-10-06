import numpy as np
import pytest

from layerbirth.numeric import (
    empirical_endomap,
    fiber_level_mismatch,
    idempotence_defect_tv,
    kernel_power,
    macro_kernel,
    packaging_projector,
    pushforward_matrix,
    retention_error,
    tv_distance,
    uniform_lift_matrix,
    validate_lens,
    validate_lift_matrix,
    validate_row_stochastic,
)


def _toy_lens():
    f = np.array([0, 0, 1, 1], dtype=int)
    q = pushforward_matrix(f, 2)
    u = uniform_lift_matrix(f, 2)
    return f, q, u


def test_toy_a_exact_packaging_projector_case():
    f, q, u = _toy_lens()
    p = packaging_projector(q, u)

    expected_q = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    expected_u = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
        ]
    )
    expected_projector = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
        ]
    )
    np.testing.assert_allclose(q, expected_q, atol=1e-12)
    np.testing.assert_allclose(u, expected_u, atol=1e-12)
    np.testing.assert_allclose(p, expected_projector, atol=1e-12)

    phat = macro_kernel(p, 1, q, u)
    e = empirical_endomap(p, 1, q, u)
    np.testing.assert_allclose(phat, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(e, expected_projector, atol=1e-12)

    assert idempotence_defect_tv(e) == 0.0
    max_ret, vec_ret = retention_error(p, 1, q, u)
    assert max_ret == 0.0
    np.testing.assert_allclose(vec_ret, np.array([0.0, 0.0]), atol=1e-12)
    vec_mismatch, max_mismatch = fiber_level_mismatch(p, 1, f, q, u)
    np.testing.assert_allclose(vec_mismatch, np.array([0.0, 0.0]), atol=1e-12)
    assert max_mismatch == 0.0


def test_toy_b_leaky_case():
    f, q, u = _toy_lens()
    p = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    expected_phat = np.array(
        [
            [0.5, 0.5],
            [0.0, 1.0],
        ]
    )
    expected_e = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
        ]
    )

    phat = macro_kernel(p, 1, q, u)
    e = empirical_endomap(p, 1, q, u)
    np.testing.assert_allclose(phat, expected_phat, atol=1e-12)
    np.testing.assert_allclose(e, expected_e, atol=1e-12)

    assert tv_distance(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == 0.5
    assert idempotence_defect_tv(e) == 0.5

    max_ret, vec_ret = retention_error(p, 1, q, u)
    assert max_ret == 0.5
    np.testing.assert_allclose(vec_ret, np.array([0.5, 0.0]), atol=1e-12)

    vec_mismatch, max_mismatch = fiber_level_mismatch(p, 1, f, q, u)
    np.testing.assert_allclose(vec_mismatch, np.array([1.0, 0.0]), atol=1e-12)
    assert max_mismatch == 1.0


def test_validation_error_paths():
    with pytest.raises(ValueError):
        validate_lens(np.array([0, 0, 0]), 2)

    with pytest.raises(ValueError):
        validate_row_stochastic(np.array([[0.8, 0.3], [0.2, 0.8]]))

    f, q, _ = _toy_lens()
    bad_u = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.2],
        ]
    )
    with pytest.raises(ValueError):
        validate_lift_matrix(q, bad_u)

    p = np.eye(4)
    np.testing.assert_allclose(kernel_power(p, 2), p, atol=1e-12)
