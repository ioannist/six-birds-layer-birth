"""First-pass metric contract for layer-birth observables."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

from .serialization import observed_float
from .numeric import (
    _coerce_lens,
    empirical_endomap,
    fiber_indices,
    fiber_level_mismatch,
    idempotence_defect_tv,
    kernel_power,
    macro_kernel,
    pushforward_matrix,
    retention_error,
    tv_distance,
    uniform_lift_matrix,
    validate_lens,
    validate_lift_matrix,
    validate_row_stochastic,
    stationary_distribution,
)


DEFAULT_METRIC_CONTRACT = {
    "CE": {
        "default": "idempotence_defect_tv",
        "alternates": ["retention_error_max", "fiber_level_mismatch_max"],
    },
    "M_obj": {
        "default": "soft_stable_count_normalized",
        "alternates": [
            "soft_stable_fraction",
            "thresholded_object_count",
            "stable_count_soft",
        ],
    },
    "SG": {
        "default": "spectral_separation_gap",
        "alternates": ["spectral_separation_ratio", "macro_relaxation_gap"],
    },
    "Aff": {
        "default": "entropy_production_rate",
        "alternates": ["flux_l1_asymmetry", "edge_log_ratio_rms"],
    },
    "Hol": {
        "default": "lift_route_mismatch_tv",
        "alternates": [
            "lift_route_mismatch_frobenius",
            "dynamic_macro_route_mismatch_tv",
        ],
    },
}


def metric_contract_defaults() -> dict[str, dict[str, Any]]:
    return deepcopy(DEFAULT_METRIC_CONTRACT)


def _resolve_q_u(f: np.ndarray | list[int], k: int, U_f: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lens = _coerce_lens(f)
    validate_lens(lens, k)
    q = pushforward_matrix(lens, k)
    u = uniform_lift_matrix(lens, k) if U_f is None else np.asarray(U_f, dtype=np.float64)
    validate_lift_matrix(q, u)
    return lens, q, u


def closure_error(
    P: np.ndarray,
    tau: int,
    f: np.ndarray | list[int],
    k: int,
    *,
    method: str = "idempotence_defect_tv",
    U_f: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    lens, q, u = _resolve_q_u(f, k, U_f)
    e = empirical_endomap(P, tau, q, u)
    ce_default = idempotence_defect_tv(e)
    ret_max, ret_vec = retention_error(P, tau, q, u)
    mismatch_vec, mismatch_max = fiber_level_mismatch(P, tau, lens, q, u)
    details = {
        "idempotence_defect_tv": float(ce_default),
        "retention_error_max": float(ret_max),
        "retention_error_per_macro_state": ret_vec.tolist(),
        "fiber_level_mismatch_max": float(mismatch_max),
        "fiber_level_mismatch_per_fiber": mismatch_vec.tolist(),
    }
    if method == "idempotence_defect_tv":
        value = ce_default
    elif method == "retention_error_max":
        value = ret_max
    elif method == "fiber_level_mismatch_max":
        value = mismatch_max
    else:
        raise ValueError(f"Unknown closure_error method: {method}")
    return float(value), details


def objecthood_order_parameter(
    P: np.ndarray,
    tau: int,
    f: np.ndarray | list[int],
    k: int,
    *,
    method: str = "soft_stable_count_normalized",
    stability_threshold: float = 0.8,
    U_f: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    _, q, u = _resolve_q_u(f, k, U_f)
    _, ret_vec = retention_error(P, tau, q, u)
    scores = np.clip(1.0 - ret_vec, 0.0, 1.0)
    n_eff = float(np.sum(scores))
    soft_fraction = float(n_eff / float(k))
    thresholded_count = int(np.sum(scores >= stability_threshold))
    if k == 1:
        normalized = 0.0
    else:
        normalized = float(np.clip((n_eff - 1.0) / float(k - 1), 0.0, 1.0))
    details = {
        "retention_error_per_macro_state": ret_vec.tolist(),
        "stability_scores": scores.tolist(),
        "stable_count_soft": n_eff,
        "soft_stable_fraction": soft_fraction,
        "thresholded_object_count": thresholded_count,
        "threshold": stability_threshold,
        "soft_stable_count_normalized": normalized,
    }
    if method == "soft_stable_count_normalized":
        value = normalized
    elif method == "soft_stable_fraction":
        value = soft_fraction
    elif method == "thresholded_object_count":
        value = float(thresholded_count)
    elif method == "stable_count_soft":
        value = n_eff
    else:
        raise ValueError(f"Unknown objecthood_order_parameter method: {method}")
    return float(value), details


def staging_gap(
    P: np.ndarray,
    tau: int,
    f: np.ndarray | list[int],
    k: int,
    *,
    method: str = "spectral_separation_gap",
    eps: float = 1e-12,
    U_f: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    _, q, u = _resolve_q_u(f, k, U_f)
    p_tau = kernel_power(P, tau)
    eig = np.linalg.eigvals(p_tau)
    mu = np.sort(np.abs(eig))[::-1]
    n = p_tau.shape[0]
    if k > n:
        raise ValueError("k cannot exceed microstate dimension")
    if k == n:
        gap = 1.0
        ratio = 1.0
        k_fallback = True
    else:
        gap = float(max(0.0, mu[k - 1] - mu[k]))
        ratio = float(mu[k - 1] / max(mu[k], eps))
        k_fallback = False

    phat = macro_kernel(P, tau, q, u)
    phat_eig = np.sort(np.abs(np.linalg.eigvals(phat)))[::-1]
    macro_relax = 1.0 if phat.shape[0] <= 1 else float(1.0 - phat_eig[1])
    details = {
        "abs_eigenvalues_sorted": mu.tolist(),
        "k": int(k),
        "k_equals_n_fallback": k_fallback,
        "spectral_separation_gap": gap,
        "spectral_separation_ratio": ratio,
        "macro_relaxation_gap": macro_relax,
    }
    if method == "spectral_separation_gap":
        value = gap
    elif method == "spectral_separation_ratio":
        value = ratio
    elif method == "macro_relaxation_gap":
        value = macro_relax
    else:
        raise ValueError(f"Unknown staging_gap method: {method}")
    return float(value), details


def _stationary_distribution_power(
    P: np.ndarray,
    *,
    max_iter: int = 10000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, bool, int]:
    return stationary_distribution(P, max_iter=max_iter, tol=tol)


def affinity_metric(
    P: np.ndarray,
    *,
    tau: int = 1,
    method: str = "entropy_production_rate",
    flux_eps: float = 1e-12,
    max_iter: int = 10000,
    tol: float = 1e-12,
) -> tuple[float, dict[str, Any]]:
    """Stationary time-reversal divergence for the observed kernel ``P**tau``.

    This is per tau-step observation, not per microstep. One-way stationary flux
    has infinite entropy production. ``flux_eps`` affects diagnostics only; it
    must not erase positive flux from the defining relative entropy.
    """
    if not np.isfinite(flux_eps) or flux_eps < 0.0:
        raise ValueError("flux_eps must be finite and nonnegative")
    p_tau = kernel_power(P, tau)
    validate_row_stochastic(p_tau)
    pi, converged, iterations = _stationary_distribution_power(
        p_tau, max_iter=max_iter, tol=tol
    )
    flux = pi[:, None] * p_tau
    flux_t = flux.T
    bi_mask = (flux > 0.0) & (flux_t > 0.0)
    log_ratio = np.zeros_like(flux)
    log_ratio[bi_mask] = np.log(flux[bi_mask]) - np.log(flux_t[bi_mask])
    directed_only_mask = (flux > 0.0) & (flux_t == 0.0)
    # Sum unordered pairs to avoid signed cancellation near detailed balance.
    upper = np.triu(bi_mask, k=1)
    entropy_rate = float("inf") if np.any(directed_only_mask) else float(
        np.sum((flux[upper] - flux_t[upper]) * log_ratio[upper])
    )
    flux_l1 = float(0.5 * np.sum(np.abs(flux - flux_t)))

    support_asym_edges = int(np.sum(directed_only_mask))
    if support_asym_edges:
        edge_log_ratio_rms = float("inf")
    elif np.any(bi_mask):
        edge_log_ratio_rms = float(np.sqrt(np.mean(np.square(log_ratio[bi_mask]))))
    else:
        edge_log_ratio_rms = 0.0

    details = {
        "stationary_distribution": pi.tolist(),
        "converged": converged,
        "iterations": int(iterations),
        "entropy_production_rate": entropy_rate,
        "flux_l1_asymmetry": flux_l1,
        "edge_log_ratio_rms": edge_log_ratio_rms,
        "support_asymmetric_directed_edges": support_asym_edges,
        "stationary_balance_residual_l1": float(np.linalg.norm(pi @ p_tau - pi, ord=1)),
        "flux_eps": float(flux_eps),
        "flux_edges_above_eps": int(np.sum(flux > flux_eps)),
        "observation_tau": int(tau),
    }
    if method == "entropy_production_rate":
        value = entropy_rate
    elif method == "flux_l1_asymmetry":
        value = flux_l1
    elif method == "edge_log_ratio_rms":
        value = edge_log_ratio_rms
    else:
        raise ValueError(f"Unknown affinity_metric method: {method}")
    return float(value), details


def _induced_coarse_over_fine_map(
    f_fine: np.ndarray,
    k_f: int,
    f_coarse: np.ndarray,
    k_c: int,
) -> np.ndarray:
    validate_lens(f_fine, k_f)
    validate_lens(f_coarse, k_c)
    if f_fine.shape[0] != f_coarse.shape[0]:
        raise ValueError("fine and coarse lenses must share microstate cardinality")
    mapping = np.full(k_f, -1, dtype=np.int64)
    for fine_label in range(k_f):
        idx = np.flatnonzero(f_fine == fine_label)
        coarse_labels = np.unique(f_coarse[idx])
        if coarse_labels.size != 1:
            raise ValueError("fine fibers must each map to a unique coarse label")
        mapping[fine_label] = int(coarse_labels[0])
    validate_lens(mapping, k_c)
    return mapping


def holonomy_metric(
    f_fine: np.ndarray | list[int],
    k_f: int,
    f_coarse: np.ndarray | list[int],
    k_c: int,
    *,
    method: str = "lift_route_mismatch_tv",
    P: np.ndarray | None = None,
    tau: int = 1,
) -> tuple[float, dict[str, Any]]:
    fine = _coerce_lens(f_fine)
    coarse = _coerce_lens(f_coarse)
    g = _induced_coarse_over_fine_map(fine, k_f, coarse, k_c)

    q_f = pushforward_matrix(fine, k_f)
    u_f = uniform_lift_matrix(fine, k_f)
    q_c = pushforward_matrix(coarse, k_c)
    u_c = uniform_lift_matrix(coarse, k_c)
    q_g = pushforward_matrix(g, k_c)
    u_g = uniform_lift_matrix(g, k_c)

    direct = u_c
    composed = u_g @ u_f
    per_row_tv = np.array([tv_distance(direct[i], composed[i]) for i in range(k_c)], dtype=float)
    tv_max = float(np.max(per_row_tv))
    fro_norm = float(np.linalg.norm(direct - composed, ord="fro"))

    dynamic_tv_max = None
    if P is not None:
        p_tau = kernel_power(P, tau)
        direct_dyn = u_c @ p_tau @ q_c
        composed_dyn = u_g @ (u_f @ p_tau @ q_f) @ q_g
        dynamic_tv = np.array(
            [tv_distance(direct_dyn[i], composed_dyn[i]) for i in range(k_c)], dtype=float
        )
        dynamic_tv_max = float(np.max(dynamic_tv))

    details = {
        "induced_coarse_over_fine_map": g.tolist(),
        "per_coarse_row_tv": per_row_tv.tolist(),
        "direct_lift_shape": list(direct.shape),
        "composed_lift_shape": list(composed.shape),
        "lift_route_mismatch_tv": tv_max,
        "lift_route_mismatch_frobenius": fro_norm,
        "dynamic_macro_route_mismatch_tv": dynamic_tv_max,
    }
    if method == "lift_route_mismatch_tv":
        value = tv_max
    elif method == "lift_route_mismatch_frobenius":
        value = fro_norm
    elif method == "dynamic_macro_route_mismatch_tv":
        if dynamic_tv_max is None:
            raise ValueError("P must be provided for dynamic_macro_route_mismatch_tv")
        value = dynamic_tv_max
    else:
        raise ValueError(f"Unknown holonomy_metric method: {method}")
    return float(value), details



def default_metric_bundle(
    P: np.ndarray,
    f: np.ndarray | list[int],
    tau: int = 1,
    lift_name: str = "uniform",
    lift_kwargs: dict[str, Any] | None = None,
    holonomy_inputs: dict[str, Any] | None = None,
    *,
    U_f: np.ndarray | None = None,
) -> dict[str, Any]:
    if U_f is None and lift_name not in {"uniform", "uniform_lift_family"}:
        raise ValueError("a nonuniform lift requires its explicit U_f matrix")
    if lift_kwargs:
        raise ValueError("lift_kwargs is currently unsupported for default_metric_bundle")

    lens = _coerce_lens(f)
    k = int(np.max(lens)) + 1
    ce, ce_details = closure_error(P, tau, lens, k, method="idempotence_defect_tv", U_f=U_f)
    mobj, mobj_details = objecthood_order_parameter(
        P, tau, lens, k, method="soft_stable_count_normalized", U_f=U_f
    )
    sg, sg_details = staging_gap(P, tau, lens, k, method="spectral_separation_gap", U_f=U_f)
    aff, aff_details = affinity_metric(P, tau=tau, method="entropy_production_rate")

    hol_value: float | None = None
    hol_details: dict[str, Any] | None = None
    if holonomy_inputs is not None:
        fine_lens = _coerce_lens(holonomy_inputs["fine_lens"])
        coarse_lens = _coerce_lens(holonomy_inputs["coarse_lens"])
        fine_k = int(holonomy_inputs.get("fine_k", int(np.max(fine_lens)) + 1))
        coarse_k = int(holonomy_inputs.get("coarse_k", int(np.max(coarse_lens)) + 1))
        hol_value, hol_details = holonomy_metric(
            fine_lens,
            fine_k,
            coarse_lens,
            coarse_k,
            method="lift_route_mismatch_tv",
        )

    return {
        "analysis_k": int(k),
        "measurement_lift_name": lift_name,
        "affinity_carrier": "microstate_kernel",
        "closure_error": float(ce),
        "objecthood_order": float(mobj),
        "staging_gap": float(sg),
        "affinity": float(aff),
        "holonomy": None if hol_value is None else float(hol_value),
        "details": {
            "closure_error": ce_details,
            "objecthood_order": mobj_details,
            "staging_gap": sg_details,
            "affinity": aff_details,
            "holonomy": hol_details,
        },
    }


def driven_candidate(bundle: dict[str, Any], affinity_threshold: float = 1e-6) -> bool:
    affinity = observed_float(bundle.get("affinity", 0.0))
    return affinity > float(affinity_threshold)
