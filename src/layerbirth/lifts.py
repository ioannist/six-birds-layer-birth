"""Deterministic lift families for layer-birth experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from .numeric import (
    _coerce_lens,
    kernel_power,
    pushforward_matrix,
    uniform_lift_matrix,
    validate_lens,
    validate_lift_matrix,
    validate_row_stochastic,
    stationary_distribution,
)


def _stationary_distribution_power(
    P: np.ndarray,
    *,
    max_iter: int = 10000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, bool, int]:
    return stationary_distribution(P, max_iter=max_iter, tol=tol)


def _identity_error(U_f: np.ndarray, Q_f: np.ndarray) -> float:
    k = U_f.shape[0]
    return float(np.max(np.abs((U_f @ Q_f) - np.eye(k, dtype=np.float64))))


def uniform_lift_family(f: np.ndarray | list[int], k: int) -> tuple[np.ndarray, dict[str, Any]]:
    lens = _coerce_lens(f)
    validate_lens(lens, k)
    q = pushforward_matrix(lens, k)
    u = uniform_lift_matrix(lens, k)
    validate_lift_matrix(q, u)
    details = {
        "family": "uniform_lift_family",
        "k": k,
        "UQ_identity_error": _identity_error(u, q),
    }
    return u, details


def prototype_lift_family(
    f: np.ndarray | list[int],
    k: int,
    *,
    prototype_indices: list[int] | np.ndarray | None = None,
    strategy: str = "first_in_fiber",
) -> tuple[np.ndarray, dict[str, Any]]:
    lens = _coerce_lens(f)
    validate_lens(lens, k)
    q = pushforward_matrix(lens, k)
    n = lens.shape[0]
    fibers = [np.flatnonzero(lens == x) for x in range(k)]
    if prototype_indices is None:
        if strategy != "first_in_fiber":
            raise ValueError("unsupported prototype strategy")
        prototypes = np.array([int(idx[0]) for idx in fibers], dtype=np.int64)
    else:
        prototypes = _coerce_lens(prototype_indices)
        if prototypes.shape != (k,):
            raise ValueError("prototype_indices must have length k")
    u = np.zeros((k, n), dtype=np.float64)
    for x in range(k):
        p_idx = int(prototypes[x])
        if p_idx < 0 or p_idx >= n:
            raise ValueError("prototype index out of bounds")
        if lens[p_idx] != x:
            raise ValueError("prototype index must lie inside its fiber")
        u[x, p_idx] = 1.0
    validate_lift_matrix(q, u)
    details = {
        "family": "prototype_lift_family",
        "k": k,
        "strategy": strategy if prototype_indices is None else "explicit",
        "prototype_indices": prototypes.tolist(),
        "UQ_identity_error": _identity_error(u, q),
    }
    return u, details


def stationary_within_fiber_lift(
    P: np.ndarray,
    f: np.ndarray | list[int],
    k: int,
    *,
    tau: int = 1,
    zero_mass_eps: float = 1e-15,
    max_iter: int = 10000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    lens = _coerce_lens(f)
    validate_lens(lens, k)
    validate_row_stochastic(P)
    p_tau = kernel_power(P, tau)
    q = pushforward_matrix(lens, k)
    pi, converged, iterations = _stationary_distribution_power(
        p_tau, max_iter=max_iter, tol=tol
    )
    n = lens.shape[0]
    u = np.zeros((k, n), dtype=np.float64)
    zero_mass_fibers: list[int] = []
    for x in range(k):
        idx = np.flatnonzero(lens == x)
        weights = pi[idx]
        mass = float(np.sum(weights))
        if mass <= zero_mass_eps:
            u[x, idx] = 1.0 / float(idx.size)
            zero_mass_fibers.append(x)
        else:
            u[x, idx] = weights / mass
    validate_lift_matrix(q, u)
    details = {
        "family": "stationary_within_fiber_lift",
        "k": k,
        "tau": tau,
        "stationary_distribution": pi.tolist(),
        "converged": converged,
        "iterations": int(iterations),
        "zero_mass_fibers": zero_mass_fibers,
        "UQ_identity_error": _identity_error(u, q),
    }
    return u, details


def build_lift_family(name: str, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "uniform_lift_family":
        return uniform_lift_family(**kwargs)
    if name == "prototype_lift_family":
        return prototype_lift_family(**kwargs)
    if name == "stationary_within_fiber_lift":
        return stationary_within_fiber_lift(**kwargs)
    raise ValueError(f"Unknown lift family: {name}")
