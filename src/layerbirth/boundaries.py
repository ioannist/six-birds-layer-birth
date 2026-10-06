"""Sampled, piecewise-linear boundary proxies; no continuum certification."""

from __future__ import annotations

import math
from typing import Any
from .serialization import observed_float


def _ordered(rows: list[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    values = sorted((float(r["closure_strength_lambda"]), observed_float(r[key])) for r in rows)
    if any(not math.isfinite(x) or math.isnan(y) for x, y in values):
        raise ValueError("boundary samples must have finite coordinates and non-NaN values")
    if any(x1 == x0 for (x0, _), (x1, _) in zip(values, values[1:])):
        raise ValueError("boundary samples must have distinct lambda coordinates")
    return values


def first_threshold_lambda(
    rows: list[dict[str, Any]], key: str, target: float, mode: str
) -> float | None:
    if mode not in {"leq", "geq"} or not math.isfinite(target):
        raise ValueError("threshold requires a finite target and mode leq or geq")
    values = _ordered(rows, key)
    for i, (x, y) in enumerate(values):
        if not (y <= target if mode == "leq" else y >= target):
            continue
        if i == 0:
            return x  # a left-censored proxy, not a resolved crossing
        x0, y0 = values[i - 1]
        if not math.isfinite(y0) or not math.isfinite(y):
            return x  # support singularity: only the sampled hit is known
        # A hit preceded by a miss has unequal values, however close they are.
        return x0 + (target - y0) * (x - x0) / (y - y0)
    return None


def interpolate_value(
    rows: list[dict[str, Any]], key: str, lam: float | None
) -> float | None:
    if lam is None or not math.isfinite(lam):
        return None
    values = _ordered(rows, key)
    if not values or not values[0][0] <= lam <= values[-1][0]:
        return None
    for i, (x, y) in enumerate(values):
        if x == lam:
            return y
        if x > lam:
            x0, y0 = values[i - 1]
            fraction = (lam - x0) / (x - x0)
            return (1.0 - fraction) * y0 + fraction * y
    return values[-1][1]


def structural_reference_lambda(ce: float | None, mo: float | None) -> float | None:
    if ce is None or mo is None or not math.isfinite(ce) or not math.isfinite(mo):
        return None
    return max(ce, mo)
