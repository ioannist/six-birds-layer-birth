"""Equilibrium-like synthetic substrate families for layer-birth."""

from __future__ import annotations

from typing import Any

import numpy as np

from .numeric import row_normalize, validate_lens, validate_row_stochastic


def block_partition_labels(n_blocks: int, block_size: int) -> np.ndarray:
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    labels = np.repeat(np.arange(n_blocks, dtype=np.int64), block_size)
    validate_lens(labels, n_blocks)
    return labels


def _ring_neighbor_block(block_i: int, block_j: int, n_blocks: int) -> bool:
    if n_blocks <= 1:
        return False
    return (block_i - block_j) % n_blocks in (1, n_blocks - 1)


def reversible_block_family(
    *,
    n_blocks: int,
    block_size: int,
    intra_block_weight: float,
    inter_block_weight: float,
    self_weight: float = 0.0,
    topology: str = "ring",
) -> dict[str, Any]:
    if topology != "ring":
        raise ValueError("only topology='ring' is supported")
    if intra_block_weight < 0.0 or inter_block_weight < 0.0 or self_weight < 0.0:
        raise ValueError("weights must be nonnegative")
    labels = block_partition_labels(n_blocks, block_size)
    n = labels.size
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        bi = int(labels[i])
        for j in range(i, n):
            bj = int(labels[j])
            if i == j:
                weight = self_weight
            elif bi == bj:
                weight = intra_block_weight
            elif _ring_neighbor_block(bi, bj, n_blocks):
                weight = inter_block_weight
            else:
                weight = 0.0
            w[i, j] = weight
            w[j, i] = weight
    p = row_normalize(w)
    validate_row_stochastic(p)
    row_sums = w.sum(axis=1)
    return {
        "P": p,
        "n": int(n),
        "family_name": "reversible_block_family",
        "block_lens": labels,
        "details": {
            "n_blocks": n_blocks,
            "block_size": block_size,
            "intra_block_weight": float(intra_block_weight),
            "inter_block_weight": float(inter_block_weight),
            "self_weight": float(self_weight),
            "topology": topology,
            "weight_row_sums": row_sums.tolist(),
        },
    }


def metastable_block_family(
    *,
    n_blocks: int,
    block_size: int,
    intra_scale: float,
    inter_scale: float,
    seed: int,
    diagonal_bias: float = 1.0,
) -> dict[str, Any]:
    if intra_scale < 0.0 or inter_scale < 0.0 or diagonal_bias <= 0.0:
        raise ValueError("intra/inter scales must be nonnegative and diagonal_bias positive")
    labels = block_partition_labels(n_blocks, block_size)
    n = labels.size
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            same_block = int(labels[i]) == int(labels[j])
            scale = intra_scale if same_block else inter_scale
            weight = scale * float(rng.uniform(0.1, 1.0))
            w[i, j] = weight
            w[j, i] = weight
    for i in range(n):
        w[i, i] = diagonal_bias + float(rng.uniform(0.0, 0.25))
    p = row_normalize(w)
    validate_row_stochastic(p)
    row_sums = w.sum(axis=1)
    stationary = row_sums / float(np.sum(row_sums))
    return {
        "P": p,
        "n": int(n),
        "family_name": "metastable_block_family",
        "block_lens": labels,
        "details": {
            "n_blocks": n_blocks,
            "block_size": block_size,
            "intra_scale": float(intra_scale),
            "inter_scale": float(inter_scale),
            "seed": int(seed),
            "diagonal_bias": float(diagonal_bias),
            "stationary_weights": stationary.tolist(),
            "weight_row_sums": row_sums.tolist(),
        },
    }


def null_flat_mixing_family(*, n: int) -> dict[str, Any]:
    if n <= 0:
        raise ValueError("n must be positive")
    p = np.full((n, n), 1.0 / float(n), dtype=np.float64)
    validate_row_stochastic(p)
    return {
        "P": p,
        "n": int(n),
        "family_name": "null_flat_mixing_family",
        "details": {"n": int(n)},
        "block_lens": None,
    }


def driven_cycle_family(
    *,
    n: int,
    self_weight: float,
    forward_weight: float,
    backward_weight: float,
) -> dict[str, Any]:
    if n < 4:
        raise ValueError("n must be >= 4 for driven_cycle_family")
    if self_weight < 0.0 or forward_weight < 0.0 or backward_weight < 0.0:
        raise ValueError("driven_cycle_family weights must be nonnegative")
    if self_weight + forward_weight + backward_weight <= 0.0:
        raise ValueError("at least one driven_cycle_family weight must be positive")

    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        w[i, i] = self_weight
        w[i, (i + 1) % n] = forward_weight
        w[i, (i - 1) % n] = backward_weight
    p = row_normalize(w)
    validate_row_stochastic(p)

    lens = np.array([0 if i < (n // 2) else 1 for i in range(n)], dtype=np.int64)
    if n % 2 == 0:
        validate_lens(lens, 2)
    else:
        lens = None

    return {
        "P": p,
        "n": int(n),
        "family_name": "driven_cycle_family",
        "details": {
            "n": int(n),
            "self_weight": float(self_weight),
            "forward_weight": float(forward_weight),
            "backward_weight": float(backward_weight),
        },
        "recommended_analysis_lens": lens,
    }


def _fine_lens_from_splits(
    coarse_lens: np.ndarray,
    coarse_k: int,
    fine_block_sizes_by_coarse: list[list[int]],
) -> np.ndarray:
    if len(fine_block_sizes_by_coarse) != coarse_k:
        raise ValueError("fine_block_sizes_by_coarse length must match coarse_k")
    fine_lens = np.zeros_like(coarse_lens, dtype=np.int64)
    next_fine = 0
    for coarse_idx in range(coarse_k):
        micro_idx = np.flatnonzero(coarse_lens == coarse_idx)
        splits = fine_block_sizes_by_coarse[coarse_idx]
        if not splits or any(s <= 0 for s in splits):
            raise ValueError("all fine split sizes must be positive")
        if int(np.sum(splits)) != int(micro_idx.size):
            raise ValueError("fine split sizes must sum to each coarse-block size")
        offset = 0
        for size in splits:
            block = micro_idx[offset : offset + size]
            fine_lens[block] = next_fine
            next_fine += 1
            offset += size
    validate_lens(fine_lens, int(next_fine))
    return fine_lens


def holonomy_control_family(
    *,
    fine_block_sizes_by_coarse: list[list[int]],
    n_blocks: int = 2,
    block_size: int = 4,
    intra_block_weight: float = 1.0,
    inter_block_weight: float = 0.05,
    self_weight: float = 0.0,
) -> dict[str, Any]:
    if n_blocks <= 0 or block_size <= 0:
        raise ValueError("n_blocks and block_size must be positive")
    base = reversible_block_family(
        n_blocks=n_blocks,
        block_size=block_size,
        intra_block_weight=intra_block_weight,
        inter_block_weight=inter_block_weight,
        self_weight=self_weight,
        topology="ring",
    )
    coarse_lens = np.asarray(base["block_lens"], dtype=np.int64)
    coarse_k = n_blocks
    fine_lens = _fine_lens_from_splits(coarse_lens, coarse_k, fine_block_sizes_by_coarse)

    details = {
        "n_blocks": int(n_blocks),
        "block_size": int(block_size),
        "fine_block_sizes_by_coarse": fine_block_sizes_by_coarse,
        "pattern": "balanced"
        if all(len(s) == 2 and s[0] == s[1] for s in fine_block_sizes_by_coarse)
        else "unbalanced",
        "base_kernel_details": base["details"],
    }
    return {
        "P": base["P"],
        "n": int(base["n"]),
        "family_name": "holonomy_control_family",
        "coarse_lens": coarse_lens,
        "fine_lens": fine_lens,
        "details": details,
    }


def holonomy_interface_skew_family(
    *,
    fine_block_sizes_by_coarse: list[list[int]],
    n_blocks: int = 2,
    block_size: int = 4,
    intra_block_weight: float = 1.0,
    interface_intra_weight: float | None = None,
    interface_cross_weight: float = 1.5,
    noninterface_cross_weight: float = 0.0,
    self_weight: float = 0.0,
    interface_position_in_block: int = 0,
) -> dict[str, Any]:
    if n_blocks != 2:
        raise ValueError("holonomy_interface_skew_family currently supports n_blocks=2 only")
    if block_size <= 1:
        raise ValueError("block_size must be > 1")
    if (
        intra_block_weight < 0.0
        or (interface_intra_weight is not None and interface_intra_weight < 0.0)
        or interface_cross_weight < 0.0
        or noninterface_cross_weight < 0.0
        or self_weight < 0.0
    ):
        raise ValueError("weights must be nonnegative")
    if interface_position_in_block < 0 or interface_position_in_block >= block_size:
        raise ValueError("interface_position_in_block out of range")

    coarse_lens = block_partition_labels(n_blocks, block_size)
    n = coarse_lens.size
    block0 = np.arange(0, block_size, dtype=np.int64)
    block1 = np.arange(block_size, 2 * block_size, dtype=np.int64)
    iface0 = int(block0[interface_position_in_block])
    iface1 = int(block1[interface_position_in_block])
    interface_set = {iface0, iface1}

    if interface_intra_weight is None:
        interface_intra_weight = intra_block_weight

    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            ci = int(coarse_lens[i])
            cj = int(coarse_lens[j])
            if i == j:
                weight = self_weight
            elif ci == cj:
                if (i in interface_set) ^ (j in interface_set):
                    weight = float(interface_intra_weight)
                else:
                    weight = intra_block_weight
            else:
                if i in interface_set and j in interface_set:
                    weight = interface_cross_weight
                else:
                    weight = noninterface_cross_weight
            w[i, j] = weight
            w[j, i] = weight
    p = row_normalize(w)
    validate_row_stochastic(p)

    fine_lens = _fine_lens_from_splits(coarse_lens, n_blocks, fine_block_sizes_by_coarse)
    details = {
        "n_blocks": int(n_blocks),
        "block_size": int(block_size),
        "intra_block_weight": float(intra_block_weight),
        "interface_intra_weight": float(interface_intra_weight),
        "interface_cross_weight": float(interface_cross_weight),
        "noninterface_cross_weight": float(noninterface_cross_weight),
        "self_weight": float(self_weight),
        "interface_position_in_block": int(interface_position_in_block),
        "interface_indices": [iface0, iface1],
        "fine_block_sizes_by_coarse": fine_block_sizes_by_coarse,
    }
    return {
        "P": p,
        "n": int(n),
        "family_name": "holonomy_interface_skew_family",
        "coarse_lens": coarse_lens,
        "fine_lens": fine_lens,
        "details": details,
    }


def _balanced_symmetric_kernel(w: np.ndarray, self_weight: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    if self_weight < 0.0:
        raise ValueError("self_weight must be nonnegative")
    n = w.shape[0]
    w2 = np.asarray(w, dtype=np.float64).copy()
    w2 = 0.5 * (w2 + w2.T)
    if self_weight > 0.0:
        for i in range(n):
            w2[i, i] += float(self_weight)
    row_sums = w2.sum(axis=1)
    target = float(np.max(row_sums))
    for i in range(n):
        add = target - float(row_sums[i])
        if add > 0.0:
            w2[i, i] += add
    p = row_normalize(w2)
    validate_row_stochastic(p)
    return p, {
        "row_sum_target": target,
        "row_sum_min": float(np.min(w2.sum(axis=1))),
        "row_sum_max": float(np.max(w2.sum(axis=1))),
    }


def delayed_interface_reversible_family(
    *,
    n: int = 8,
    core_core_weight: float = 1.0,
    core_interface_weight: float = 0.08,
    interface_cross_weight: float = 1.20,
    self_weight: float = 0.0,
) -> dict[str, Any]:
    if n != 8:
        raise ValueError("delayed_interface_reversible_family currently supports n=8")
    if min(core_core_weight, core_interface_weight, interface_cross_weight, self_weight) < 0.0:
        raise ValueError("weights must be nonnegative")
    coarse_lens = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    validate_lens(coarse_lens, 2)
    w = np.zeros((n, n), dtype=np.float64)
    block0 = [0, 1, 2, 3]
    block1 = [4, 5, 6, 7]
    iface0, iface1 = 3, 7
    for block in (block0, block1):
        cores = [i for i in block if i not in {iface0, iface1}]
        for i in cores:
            for j in cores:
                if i < j:
                    w[i, j] = core_core_weight
                    w[j, i] = core_core_weight
        for i in cores:
            iface = iface0 if i in block0 else iface1
            w[i, iface] = core_interface_weight
            w[iface, i] = core_interface_weight
    w[iface0, iface1] = interface_cross_weight
    w[iface1, iface0] = interface_cross_weight
    p, bal = _balanced_symmetric_kernel(w, self_weight=self_weight)
    return {
        "P": p,
        "n": int(n),
        "family_name": "delayed_interface_reversible_family",
        "coarse_lens": coarse_lens,
        "details": {
            "core_core_weight": float(core_core_weight),
            "core_interface_weight": float(core_interface_weight),
            "interface_cross_weight": float(interface_cross_weight),
            "self_weight": float(self_weight),
            "interface_indices": [iface0, iface1],
            **bal,
        },
    }


def hidden_sector_reversible_family(
    *,
    n: int = 8,
    visible_weight: float = 1.0,
    hidden_weight: float = 1.0,
    visible_hidden_weight: float = 0.05,
    hidden_cross_weight: float = 1.20,
    self_weight: float = 0.0,
) -> dict[str, Any]:
    if n < 8 or n % 2 != 0:
        raise ValueError("hidden_sector_reversible_family requires even n >= 8")
    if min(visible_weight, hidden_weight, visible_hidden_weight, hidden_cross_weight, self_weight) < 0.0:
        raise ValueError("weights must be nonnegative")
    block_size = n // 2
    coarse_lens = np.asarray([0] * block_size + [1] * block_size, dtype=np.int64)
    validate_lens(coarse_lens, 2)
    w = np.zeros((n, n), dtype=np.float64)
    block0 = list(range(0, block_size))
    block1 = list(range(block_size, n))
    visible0 = block0[: max(2, block_size // 2)]
    hidden0 = block0[len(visible0) :]
    visible1 = block1[: max(2, block_size // 2)]
    hidden1 = block1[len(visible1) :]
    portal0 = hidden0[-1]
    portal1 = hidden1[-1]
    for grp, weight in ((visible0, visible_weight), (visible1, visible_weight), (hidden0, hidden_weight), (hidden1, hidden_weight)):
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b = grp[i], grp[j]
                w[a, b] = weight
                w[b, a] = weight
    for i in visible0:
        for j in hidden0:
            w[i, j] = visible_hidden_weight
            w[j, i] = visible_hidden_weight
    for i in visible1:
        for j in hidden1:
            w[i, j] = visible_hidden_weight
            w[j, i] = visible_hidden_weight
    w[portal0, portal1] = hidden_cross_weight
    w[portal1, portal0] = hidden_cross_weight
    p, bal = _balanced_symmetric_kernel(w, self_weight=self_weight)
    return {
        "P": p,
        "n": int(n),
        "family_name": "hidden_sector_reversible_family",
        "coarse_lens": coarse_lens,
        "details": {
            "n": int(n),
            "visible_weight": float(visible_weight),
            "hidden_weight": float(hidden_weight),
            "visible_hidden_weight": float(visible_hidden_weight),
            "hidden_cross_weight": float(hidden_cross_weight),
            "self_weight": float(self_weight),
            "hidden_portal_indices": [portal0, portal1],
            **bal,
        },
    }


def two_timescale_reversible_family(
    *,
    n: int = 8,
    fast_weight: float = 1.0,
    fast_portal_weight: float = 0.03,
    portal_cross_weight: float = 1.20,
    portal_self_weight: float = 0.50,
) -> dict[str, Any]:
    if n < 8 or n % 2 != 0:
        raise ValueError("two_timescale_reversible_family requires even n >= 8")
    if min(fast_weight, fast_portal_weight, portal_cross_weight, portal_self_weight) < 0.0:
        raise ValueError("weights must be nonnegative")
    block_size = n // 2
    coarse_lens = np.asarray([0] * block_size + [1] * block_size, dtype=np.int64)
    validate_lens(coarse_lens, 2)
    w = np.zeros((n, n), dtype=np.float64)
    block0 = list(range(0, block_size))
    block1 = list(range(block_size, n))
    portal0, portal1 = block0[-1], block1[-1]
    fast0, fast1 = block0[:-1], block1[:-1]
    for triad in (fast0, fast1):
        for i in range(len(triad)):
            for j in range(i + 1, len(triad)):
                a, b = triad[i], triad[j]
                w[a, b] = fast_weight
                w[b, a] = fast_weight
    for i in fast0:
        w[i, portal0] = fast_portal_weight
        w[portal0, i] = fast_portal_weight
    for i in fast1:
        w[i, portal1] = fast_portal_weight
        w[portal1, i] = fast_portal_weight
    w[portal0, portal1] = portal_cross_weight
    w[portal1, portal0] = portal_cross_weight
    w[portal0, portal0] += portal_self_weight
    w[portal1, portal1] += portal_self_weight
    p, bal = _balanced_symmetric_kernel(w, self_weight=0.0)
    return {
        "P": p,
        "n": int(n),
        "family_name": "two_timescale_reversible_family",
        "coarse_lens": coarse_lens,
        "details": {
            "n": int(n),
            "fast_weight": float(fast_weight),
            "fast_portal_weight": float(fast_portal_weight),
            "portal_cross_weight": float(portal_cross_weight),
            "portal_self_weight": float(portal_self_weight),
            "portal_indices": [portal0, portal1],
            **bal,
        },
    }


def build_class_iii_candidate_family(name: str, **kwargs: Any) -> dict[str, Any]:
    if name == "delayed_interface_reversible_family":
        return delayed_interface_reversible_family(**kwargs)
    if name == "hidden_sector_reversible_family":
        return hidden_sector_reversible_family(**kwargs)
    if name == "two_timescale_reversible_family":
        return two_timescale_reversible_family(**kwargs)
    if name in {"distributed_hidden_sector_reversible_family", "replicated_portal_reversible_family"}:
        return build_class_iii_bulk_family(name, **kwargs)
    raise ValueError(f"Unknown class-III candidate family: {name}")


def distributed_hidden_sector_reversible_family(
    *,
    n: int,
    visible_visible_weight: float = 1.0,
    hidden_hidden_weight: float = 1.0,
    visible_hidden_weight: float = 0.02,
    hidden_cross_weight: float = 1.20,
    self_weight: float = 0.0,
) -> dict[str, Any]:
    if n < 16 or n % 2 != 0:
        raise ValueError("distributed_hidden_sector_reversible_family requires even n >= 16")
    if min(visible_visible_weight, hidden_hidden_weight, visible_hidden_weight, hidden_cross_weight, self_weight) < 0.0:
        raise ValueError("weights must be nonnegative")
    block_size = n // 2
    if block_size % 2 != 0:
        raise ValueError("block_size must be even")
    coarse_lens = np.asarray([0] * block_size + [1] * block_size, dtype=np.int64)
    validate_lens(coarse_lens, 2)
    b0 = list(range(0, block_size))
    b1 = list(range(block_size, n))
    vis0, hid0 = b0[: block_size // 2], b0[block_size // 2 :]
    vis1, hid1 = b1[: block_size // 2], b1[block_size // 2 :]

    w = np.zeros((n, n), dtype=np.float64)
    for grp, weight in ((vis0, visible_visible_weight), (vis1, visible_visible_weight), (hid0, hidden_hidden_weight), (hid1, hidden_hidden_weight)):
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b = grp[i], grp[j]
                w[a, b] = weight
                w[b, a] = weight
    for vis, hid in ((vis0, hid0), (vis1, hid1)):
        for i in vis:
            for j in hid:
                w[i, j] = visible_hidden_weight
                w[j, i] = visible_hidden_weight
    for i in hid0:
        for j in hid1:
            w[i, j] = hidden_cross_weight
            w[j, i] = hidden_cross_weight

    p, bal = _balanced_symmetric_kernel(w, self_weight=self_weight)
    return {
        "P": p,
        "n": int(n),
        "family_name": "distributed_hidden_sector_reversible_family",
        "coarse_lens": coarse_lens,
        "details": {
            "n": int(n),
            "visible_visible_weight": float(visible_visible_weight),
            "hidden_hidden_weight": float(hidden_hidden_weight),
            "visible_hidden_weight": float(visible_hidden_weight),
            "hidden_cross_weight": float(hidden_cross_weight),
            "self_weight": float(self_weight),
            "hidden_fraction": 0.5,
            **bal,
        },
    }


def replicated_portal_reversible_family(
    *,
    n: int,
    fast_weight: float = 1.0,
    fast_portal_weight: float = 0.02,
    portal_cross_weight: float = 1.20,
    portal_self_weight: float = 0.50,
) -> dict[str, Any]:
    if n < 16 or n % 8 != 0:
        raise ValueError("replicated_portal_reversible_family requires n divisible by 8 and >= 16")
    if min(fast_weight, fast_portal_weight, portal_cross_weight, portal_self_weight) < 0.0:
        raise ValueError("weights must be nonnegative")
    block_size = n // 2
    modules_per_block = block_size // 4
    coarse_lens = np.asarray([0] * block_size + [1] * block_size, dtype=np.int64)
    validate_lens(coarse_lens, 2)

    w = np.zeros((n, n), dtype=np.float64)
    portals0: list[int] = []
    portals1: list[int] = []
    for block in (0, 1):
        start = block * block_size
        for m in range(modules_per_block):
            base = start + 4 * m
            fast = [base, base + 1, base + 2]
            portal = base + 3
            if block == 0:
                portals0.append(portal)
            else:
                portals1.append(portal)
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = fast[i], fast[j]
                    w[a, b] = fast_weight
                    w[b, a] = fast_weight
            for f in fast:
                w[f, portal] = fast_portal_weight
                w[portal, f] = fast_portal_weight
            w[portal, portal] += portal_self_weight

    for p0, p1 in zip(portals0, portals1):
        w[p0, p1] = portal_cross_weight
        w[p1, p0] = portal_cross_weight

    p, bal = _balanced_symmetric_kernel(w, self_weight=0.0)
    return {
        "P": p,
        "n": int(n),
        "family_name": "replicated_portal_reversible_family",
        "coarse_lens": coarse_lens,
        "details": {
            "n": int(n),
            "fast_weight": float(fast_weight),
            "fast_portal_weight": float(fast_portal_weight),
            "portal_cross_weight": float(portal_cross_weight),
            "portal_self_weight": float(portal_self_weight),
            "portal_fraction": 0.25,
            "modules_per_block": int(modules_per_block),
            **bal,
        },
    }


def build_class_iii_bulk_family(name: str, **kwargs: Any) -> dict[str, Any]:
    if name == "distributed_hidden_sector_reversible_family":
        return distributed_hidden_sector_reversible_family(**kwargs)
    if name == "replicated_portal_reversible_family":
        return replicated_portal_reversible_family(**kwargs)
    raise ValueError(f"Unknown class-III bulk family: {name}")


def build_substrate_family(name: str, **kwargs: Any) -> dict[str, Any]:
    if name == "reversible_block_family":
        return reversible_block_family(**kwargs)
    if name == "metastable_block_family":
        return metastable_block_family(**kwargs)
    if name == "null_flat_mixing_family":
        return null_flat_mixing_family(**kwargs)
    if name == "driven_cycle_family":
        return driven_cycle_family(**kwargs)
    if name == "holonomy_control_family":
        return holonomy_control_family(**kwargs)
    if name == "holonomy_interface_skew_family":
        return holonomy_interface_skew_family(**kwargs)
    if name in {
        "delayed_interface_reversible_family",
        "hidden_sector_reversible_family",
        "two_timescale_reversible_family",
        "distributed_hidden_sector_reversible_family",
        "replicated_portal_reversible_family",
    }:
        return build_class_iii_candidate_family(name, **kwargs)
    raise ValueError(f"Unknown substrate family: {name}")


def build_equilibrium_substrate_family(name: str, **kwargs: Any) -> dict[str, Any]:
    return build_substrate_family(name, **kwargs)
