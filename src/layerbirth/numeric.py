"""Canonical numeric backend for layer-birth row-vector dynamics.

Conventions:
- Micro distributions are row vectors on Z, shape ``(n,)``.
- Macro distributions are row vectors on X, shape ``(k,)``.
- ``Q_f`` is pushforward with shape ``(n, k)``.
- ``U_f`` is lift with shape ``(k, n)``.
- Kernels are row-stochastic.
- Empirical endomap: ``E_{tau,f} = P^tau Q_f U_f``.
- Macro kernel: ``P_hat_{tau,f} = U_f P^tau Q_f``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _coerce_lens(f: npt.ArrayLike) -> IntArray:
    arr = np.asarray(f)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("lens assignment must be a non-empty 1D array")
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    if not np.issubdtype(arr.dtype, np.floating) or not np.isfinite(arr).all():
        raise ValueError("lens assignment must be finite and real")
    rounded = np.rint(arr)
    if not np.array_equal(arr, rounded):
        raise ValueError("lens assignment must be integer-valued")
    return rounded.astype(np.int64, copy=False)


def validate_lens(f: npt.ArrayLike, k: int) -> None:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be positive")
    lens = _coerce_lens(f)
    if np.any(lens < 0) or np.any(lens >= k):
        raise ValueError("lens values must be in [0, k-1]")
    present = set(np.unique(lens).tolist())
    expected = set(range(k))
    if present != expected:
        raise ValueError("lens must be surjective")


def fiber_indices(f: npt.ArrayLike, k: int) -> list[np.ndarray]:
    lens = _coerce_lens(f)
    validate_lens(lens, k)
    return [np.flatnonzero(lens == x) for x in range(k)]


def pushforward_matrix(f: npt.ArrayLike, k: int) -> FloatArray:
    lens = _coerce_lens(f)
    validate_lens(lens, k)
    n = lens.shape[0]
    q = np.zeros((n, k), dtype=np.float64)
    q[np.arange(n), lens] = 1.0
    return q


def uniform_lift_matrix(f: npt.ArrayLike, k: int) -> FloatArray:
    lens = _coerce_lens(f)
    validate_lens(lens, k)
    n = lens.shape[0]
    u = np.zeros((k, n), dtype=np.float64)
    for x, idx in enumerate(fiber_indices(lens, k)):
        u[x, idx] = 1.0 / float(idx.size)
    return u


def prototype_lift_matrix(f: npt.ArrayLike, k: int, prototypes: npt.ArrayLike) -> FloatArray:
    lens = _coerce_lens(f)
    validate_lens(lens, k)
    proto = np.asarray(prototypes, dtype=np.float64)
    n = lens.shape[0]
    if proto.shape != (k, n):
        raise ValueError(f"prototypes must have shape {(k, n)}")
    if np.any(proto < 0.0) or not np.isfinite(proto).all():
        raise ValueError("prototypes must be finite and nonnegative")
    if not np.allclose(proto.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("prototype rows must sum to 1")
    for x in range(k):
        if np.any(proto[x, lens != x] > 1e-12):
            raise ValueError("prototype row has mass outside its fiber")
    q = pushforward_matrix(lens, k)
    validate_lift_matrix(q, proto)
    return proto


def validate_lift_matrix(Q_f: npt.ArrayLike, U_f: npt.ArrayLike, atol: float = 1e-12) -> None:
    q = np.asarray(Q_f, dtype=np.float64)
    u = np.asarray(U_f, dtype=np.float64)
    if q.ndim != 2 or u.ndim != 2:
        raise ValueError("Q_f and U_f must be matrices")
    n, k = q.shape
    validate_row_stochastic(q, atol=atol)
    if np.any(q < 0.0):
        raise ValueError("Q_f must be nonnegative")
    if u.shape != (k, n):
        raise ValueError(f"U_f must have shape {(k, n)}")
    if np.any(u < 0.0) or not np.isfinite(u).all():
        raise ValueError("U_f must be finite and nonnegative")
    if not np.allclose(u.sum(axis=1), 1.0, atol=atol, rtol=0.0):
        raise ValueError("U_f must be row-stochastic")
    if not np.allclose(u @ q, np.eye(k, dtype=np.float64), atol=atol, rtol=0.0):
        raise ValueError("U_f @ Q_f must equal identity")


def validate_row_stochastic(P: npt.ArrayLike, atol: float = 1e-12) -> None:
    p = np.asarray(P, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("P must be a non-empty matrix")
    if np.any(p < -atol) or not np.isfinite(p).all():
        raise ValueError("P must be finite and nonnegative")
    if not np.allclose(p.sum(axis=1), 1.0, atol=atol, rtol=0.0):
        raise ValueError("P rows must sum to 1")


def row_normalize(M: npt.ArrayLike, eps: float = 1e-15) -> FloatArray:
    matrix = np.asarray(M, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("M must be a matrix")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("M must be finite and nonnegative")
    out = np.zeros_like(matrix, dtype=np.float64)
    sums = matrix.sum(axis=1)
    # Positive weights remain legal even when their absolute scale is tiny.
    good = sums > 0.0
    out[good] = matrix[good] / sums[good, None]
    return out


def stationary_distribution(
    P: npt.ArrayLike, *, max_iter: int = 10000, tol: float = 1e-12
) -> tuple[FloatArray, bool, int]:
    """Stationary law selected by a uniform initial law (Cesaro limit).

    A power-iteration shortcut is used for a strictly positive kernel. Otherwise
    closed communicating classes and absorption probabilities are solved directly.
    This handles periodic/reducible chains and gives transient states exactly zero
    mass, which matters for entropy-production support. Failure raises instead of
    returning a nonstationary iterate as a scientific observable.
    """
    p = np.asarray(P, dtype=np.float64)
    validate_row_stochastic(p)
    if p.shape[0] != p.shape[1]:
        raise ValueError("stationary distribution requires a square kernel")
    if np.any(p < 0.0):
        raise ValueError("stationary distribution requires nonnegative entries")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    n = p.shape[0]
    pi = np.full(n, 1.0 / n, dtype=np.float64)
    if np.all(p > 0.0):
        for step in range(1, max_iter + 1):
            nxt = pi @ p
            if np.linalg.norm(nxt - pi, ord=1) < tol:
                nxt /= nxt.sum()
                if np.linalg.norm(nxt @ p - nxt, ord=1) < tol:
                    return nxt, True, step
            pi = nxt

    # Kosaraju's algorithm on exact positive support, using iterative DFS.
    adjacency = [np.flatnonzero(row > 0.0).tolist() for row in p]
    reverse = [np.flatnonzero(row > 0.0).tolist() for row in p.T]
    seen: set[int] = set()
    finish: list[int] = []
    for start in range(n):
        if start in seen:
            continue
        seen.add(start)
        stack = [(start, iter(adjacency[start]))]
        while stack:
            node, neighbors = stack[-1]
            neighbor = next(neighbors, None)
            if neighbor is None:
                finish.append(node)
                stack.pop()
            elif neighbor not in seen:
                seen.add(neighbor)
                stack.append((neighbor, iter(adjacency[neighbor])))
    seen.clear()
    closed: list[list[int]] = []
    for start in reversed(finish):
        if start in seen:
            continue
        component: list[int] = []
        pending = [start]
        seen.add(start)
        while pending:
            node = pending.pop()
            component.append(node)
            for neighbor in reverse[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        members = set(component)
        if all(j in members for i in component for j in adjacency[i]):
            closed.append(sorted(component))

    recurrent = {i for component in closed for i in component}
    transient = sorted(set(range(n)) - recurrent)
    weights = np.array([len(c) / n for c in closed], dtype=np.float64)
    if transient:
        exit_probs = np.column_stack(
            [p[np.ix_(transient, c)].sum(axis=1) for c in closed]
        )
        absorption = np.linalg.solve(
            np.eye(len(transient)) - p[np.ix_(transient, transient)], exit_probs
        )
        if (not np.isfinite(absorption).all() or np.any(absorption < -tol)
                or not np.allclose(absorption.sum(axis=1), 1.0, atol=tol, rtol=0.0)):
            raise ValueError("could not resolve closed-class absorption probabilities")
        weights += np.maximum(absorption, 0.0).sum(axis=0) / n
    pi = np.zeros(n, dtype=np.float64)
    for component, weight in zip(closed, weights):
        pc = p[np.ix_(component, component)]
        uniform = np.full(len(component), 1.0 / len(component))
        if np.linalg.norm(uniform @ pc - uniform, ord=1) <= tol:
            pi[component] = weight * uniform
            continue
        balance = pc.T - np.eye(len(component))
        balance[-1] = 1.0
        rhs = np.zeros(len(component))
        rhs[-1] = 1.0
        law = np.linalg.solve(balance, rhs)
        if not np.isfinite(law).all() or np.any(law < -tol):
            raise ValueError("could not resolve a nonnegative stationary law")
        law = np.maximum(law, 0.0)
        law /= law.sum()
        pi[component] = weight * law
    pi /= pi.sum()
    if np.linalg.norm(pi @ p - pi, ord=1) > tol:
        raise ValueError("stationary law failed the balance residual check")
    return pi, True, 0  # zero iterations denotes the communicating-class solve


def kernel_power(P: npt.ArrayLike, tau: int) -> FloatArray:
    if isinstance(tau, (bool, np.bool_)) or not isinstance(tau, (int, np.integer)) or tau < 1:
        raise ValueError("tau must be >= 1")
    p = np.asarray(P, dtype=np.float64)
    validate_row_stochastic(p)
    if p.shape[0] != p.shape[1]:
        raise ValueError("P must be square")
    if np.any(p < 0.0):
        raise ValueError("P must be nonnegative")
    return np.linalg.matrix_power(p, tau).astype(np.float64, copy=False)


def packaging_projector(Q_f: npt.ArrayLike, U_f: npt.ArrayLike) -> FloatArray:
    validate_lift_matrix(Q_f, U_f)
    q = np.asarray(Q_f, dtype=np.float64)
    u = np.asarray(U_f, dtype=np.float64)
    return (q @ u).astype(np.float64, copy=False)


def macro_kernel(P: npt.ArrayLike, tau: int, Q_f: npt.ArrayLike, U_f: npt.ArrayLike) -> FloatArray:
    validate_lift_matrix(Q_f, U_f)
    p_tau = kernel_power(P, tau)
    q = np.asarray(Q_f, dtype=np.float64)
    u = np.asarray(U_f, dtype=np.float64)
    n = q.shape[0]
    if p_tau.shape != (n, n):
        raise ValueError("P shape does not match Q_f/U_f")
    return (u @ p_tau @ q).astype(np.float64, copy=False)


def empirical_endomap(P: npt.ArrayLike, tau: int, Q_f: npt.ArrayLike, U_f: npt.ArrayLike) -> FloatArray:
    validate_lift_matrix(Q_f, U_f)
    p_tau = kernel_power(P, tau)
    q = np.asarray(Q_f, dtype=np.float64)
    u = np.asarray(U_f, dtype=np.float64)
    n = q.shape[0]
    if p_tau.shape != (n, n):
        raise ValueError("P shape does not match Q_f/U_f")
    return (p_tau @ q @ u).astype(np.float64, copy=False)


def tv_distance(p: npt.ArrayLike, q: npt.ArrayLike) -> float:
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.shape != q_arr.shape:
        raise ValueError("vectors must have same shape")
    return float(0.5 * np.abs(p_arr - q_arr).sum())


def idempotence_defect_tv(E: npt.ArrayLike) -> float:
    e = np.asarray(E, dtype=np.float64)
    if e.ndim != 2 or e.shape[0] != e.shape[1]:
        raise ValueError("E must be square")
    e2 = e @ e
    return float(max(tv_distance(e[i], e2[i]) for i in range(e.shape[0])))


def retention_error(
    P: npt.ArrayLike,
    tau: int,
    Q_f: npt.ArrayLike,
    U_f: npt.ArrayLike,
) -> tuple[float, FloatArray]:
    phat = macro_kernel(P, tau, Q_f, U_f)
    k = phat.shape[0]
    per_state = np.zeros(k, dtype=np.float64)
    for x in range(k):
        target = np.zeros(k, dtype=np.float64)
        target[x] = 1.0
        per_state[x] = tv_distance(phat[x], target)
    return float(per_state.max()), per_state


def fiber_level_mismatch(
    P: npt.ArrayLike,
    tau: int,
    f: npt.ArrayLike,
    Q_f: npt.ArrayLike,
    U_f: npt.ArrayLike,
) -> tuple[FloatArray, float]:
    endomap = empirical_endomap(P, tau, Q_f, U_f)
    q = np.asarray(Q_f, dtype=np.float64)
    k = q.shape[1]
    fibers = fiber_indices(f, k)
    per_fiber = np.zeros(k, dtype=np.float64)
    for x, idx in enumerate(fibers):
        if idx.size <= 1:
            per_fiber[x] = 0.0
            continue
        max_tv = 0.0
        for i in range(idx.size):
            for j in range(i + 1, idx.size):
                max_tv = max(max_tv, tv_distance(endomap[idx[i]], endomap[idx[j]]))
        per_fiber[x] = max_tv
    return per_fiber, float(per_fiber.max())
