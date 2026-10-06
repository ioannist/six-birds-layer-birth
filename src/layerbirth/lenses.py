"""Deterministic lens families for layer-birth experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .numeric import _coerce_lens, kernel_power, validate_lens, validate_row_stochastic


def canonicalize_lens_labels(labels: np.ndarray | list[int]) -> tuple[np.ndarray, dict[str, Any]]:
    arr = _coerce_lens(labels)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("labels must be a non-empty 1D array")
    mapping: dict[int, int] = {}
    out = np.zeros_like(arr)
    next_id = 0
    for i, value in enumerate(arr.tolist()):
        if value not in mapping:
            mapping[value] = next_id
            next_id += 1
        out[i] = mapping[value]
    return out, {"label_mapping": {str(k): v for k, v in mapping.items()}, "k": next_id}


def manual_partition_lens(
    labels: np.ndarray | list[int],
    *,
    k: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    lens, details = canonicalize_lens_labels(labels)
    inferred_k = int(details["k"])
    if k is not None and inferred_k != k:
        raise ValueError(f"manual partition has {inferred_k} groups but k={k}")
    validate_lens(lens, inferred_k)
    details.update({"family": "manual_partition_lens", "k": inferred_k})
    return lens, details


def _deterministic_eigenvectors(P_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eig(P_tau.T)
    order = sorted(
        range(len(vals)),
        key=lambda i: (
            -float(abs(vals[i])),
            -float(np.real(vals[i])),
            float(np.imag(vals[i])),
            i,
        ),
    )
    vals_ord = np.array([vals[i] for i in order])
    vecs_ord = np.array([np.real(vecs[:, i]) for i in order]).T
    # Deterministic sign convention: largest-magnitude component is nonnegative.
    for j in range(vecs_ord.shape[1]):
        col = vecs_ord[:, j]
        idx = int(np.argmax(np.abs(col)))
        if col[idx] < 0.0 or (col[idx] == 0.0 and np.sum(col) < 0.0):
            vecs_ord[:, j] = -col
    return vals_ord, vecs_ord


def _quantile_partition(coord: np.ndarray, k: int) -> np.ndarray:
    n = coord.shape[0]
    if k > n:
        raise ValueError("k must be <= n for quantile partition")
    order = np.argsort(coord, kind="mergesort")
    labels = np.zeros(n, dtype=np.int64)
    base = n // k
    rem = n % k
    start = 0
    for group in range(k):
        size = base + (1 if group < rem else 0)
        idx = order[start : start + size]
        labels[idx] = group
        start += size
    return labels


def spectral_sign_pattern_lens(
    P: np.ndarray,
    *,
    tau: int = 1,
    target_k: int = 2,
    num_vectors: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    validate_row_stochastic(P)
    p_tau = kernel_power(P, tau)
    n = p_tau.shape[0]
    if target_k < 1 or target_k > n:
        raise ValueError("target_k must be in [1, n]")
    vals, vecs = _deterministic_eigenvectors(p_tau)
    if target_k == 1:
        lens = np.zeros(n, dtype=np.int64)
        return lens, {
            "family": "spectral_sign_pattern_lens",
            "tau": tau,
            "target_k": target_k,
            "num_vectors_used": 0,
            "fallback_used": False,
            "eigenvalues_abs_sorted": np.abs(vals).tolist(),
        }

    m = num_vectors if num_vectors is not None else int(max(1, math.ceil(math.log2(target_k))))
    m = min(m, max(1, n - 1))
    coords = vecs[:, 1 : 1 + m]
    bits = (coords >= 0.0).astype(np.int64)
    codes = np.zeros(n, dtype=np.int64)
    for j in range(bits.shape[1]):
        codes = (codes << 1) + bits[:, j]
    lens, canon_details = canonicalize_lens_labels(codes)
    fallback_used = False
    if len(np.unique(lens)) != target_k:
        fallback_used = True
        lens = _quantile_partition(coords[:, 0], target_k)
    lens, canon2 = canonicalize_lens_labels(lens)
    validate_lens(lens, target_k)
    details = {
        "family": "spectral_sign_pattern_lens",
        "tau": tau,
        "target_k": target_k,
        "num_vectors_used": int(m),
        "fallback_used": fallback_used,
        "eigenvalues_abs_sorted": np.abs(vals).tolist(),
        "initial_code_canonicalization": canon_details,
        "final_canonicalization": canon2,
    }
    return lens, details


def diffusion_quantile_lens(
    P: np.ndarray,
    *,
    tau: int = 1,
    target_k: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    validate_row_stochastic(P)
    p_tau = kernel_power(P, tau)
    n = p_tau.shape[0]
    if target_k < 1 or target_k > n:
        raise ValueError("target_k must be in [1, n]")
    vals, vecs = _deterministic_eigenvectors(p_tau)
    if target_k == 1:
        lens = np.zeros(n, dtype=np.int64)
        return lens, {
            "family": "diffusion_quantile_lens",
            "tau": tau,
            "target_k": target_k,
            "eigenvalues_abs_sorted": np.abs(vals).tolist(),
        }
    coord = vecs[:, 1]
    lens = _quantile_partition(coord, target_k)
    lens, canon = canonicalize_lens_labels(lens)
    validate_lens(lens, target_k)
    details = {
        "family": "diffusion_quantile_lens",
        "tau": tau,
        "target_k": target_k,
        "eigenvalues_abs_sorted": np.abs(vals).tolist(),
        "coordinate": coord.tolist(),
        "canonicalization": canon,
    }
    return lens, details


def random_surjective_lens(n: int, k: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if k <= 0 or k > n:
        raise ValueError("k must be in [1, n]")
    rng = np.random.default_rng(seed)
    base = np.arange(k, dtype=np.int64)
    rest = rng.integers(0, k, size=n - k, endpoint=False, dtype=np.int64)
    labels = np.concatenate([base, rest])
    perm = rng.permutation(n)
    lens = labels[perm]
    lens, canon = canonicalize_lens_labels(lens)
    validate_lens(lens, k)
    details = {
        "family": "random_surjective_lens",
        "n": n,
        "k": k,
        "seed": seed,
        "canonicalization": canon,
    }
    return lens, details


def build_lens_family(name: str, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "manual_partition_lens":
        return manual_partition_lens(**kwargs)
    if name == "spectral_sign_pattern_lens":
        return spectral_sign_pattern_lens(**kwargs)
    if name == "diffusion_quantile_lens":
        return diffusion_quantile_lens(**kwargs)
    if name == "random_surjective_lens":
        return random_surjective_lens(**kwargs)
    raise ValueError(f"Unknown lens family: {name}")
