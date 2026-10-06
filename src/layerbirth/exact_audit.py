"""Independent rational audit of the four canonical manual-channel witnesses.

Only the declared equal two-block families are represented here; the canonical
certificate uses tau=1,2 and the structural helper supports positive integer tau.
This module does not call the floating metric/classification implementation.
Decimal configuration weights are interpreted as exact rationals.
"""

from __future__ import annotations

from fractions import Fraction as F
from typing import Any


def rational(value: Any) -> F:
    return F(str(value))


def _zeros(n: int) -> list[list[F]]:
    return [[F(0) for _ in range(n)] for _ in range(n)]


def cycle_kernel(n: int, config: dict[str, Any]) -> list[list[F]]:
    weights = [rational(config[key]) for key in ("self_weight", "forward_weight", "backward_weight")]
    if n < 4 or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("invalid cycle")
    p = _zeros(n)
    for i in range(n):
        for j, w in zip((i, (i+1) % n, (i-1) % n), weights):
            p[i][j] = w / sum(weights)
    return p


def canonical_kernel(family: str, n: int, config: dict[str, Any], drive: dict[str, Any] | None = None) -> list[list[F]]:
    if n % 2 or n < 4:
        raise ValueError("equal-block audit requires even n >= 4")
    h = n // 2
    if family == "driven_cycle_family":
        p = cycle_kernel(n, config)
    elif family == "reversible_block_family":
        if int(config["n_blocks"]) != 2 or int(config["block_size"]) != h:
            raise ValueError("audit represents exactly two equal blocks")
        within, cross, self_w = [rational(config[k]) for k in ("intra_block_weight", "inter_block_weight", "self_weight")]
        denom = (h-1)*within + h*cross + self_w
        p = [[(self_w if i == j else within if (i<h) == (j<h) else cross) / denom
              for j in range(n)] for i in range(n)]
    elif family == "replicated_portal_reversible_family":
        if n < 16 or n % 8:
            raise ValueError("invalid replicated portal size")
        fast, fp, pc, ps = [rational(config[k]) for k in ("fast_weight", "fast_portal_weight", "portal_cross_weight", "portal_self_weight")]
        w = _zeros(n)
        for start in range(0, n, 4):
            portal = start + 3
            for i in range(start, portal):
                for j in range(i+1, portal):
                    w[i][j] = w[j][i] = fast
                w[i][portal] = w[portal][i] = fp
            w[portal][portal] = ps
        for i in range(3, h, 4):
            w[i][i+h] = w[i+h][i] = pc
        target = max(map(sum, w))
        for i in range(n):
            w[i][i] += target - sum(w[i])
        p = [[v / target for v in row] for row in w]
    else:
        raise ValueError(f"unrepresented family {family}")
    if drive:
        alpha = rational(drive["drive_mix"])
        total, bias = rational(drive["flow_total"]), rational(drive["drive_bias"])
        if not 0 <= alpha <= 1 or abs(bias) > total/2:
            raise ValueError("invalid drive parameters")
        cycle = cycle_kernel(n, {"self_weight": drive["drive_self_weight"],
                                 "forward_weight": total/2+bias, "backward_weight": total/2-bias})
        p = [[(1-alpha)*p[i][j] + alpha*cycle[i][j] for j in range(n)] for i in range(n)]
    if any(v < 0 for row in p for v in row) or any(sum(row) != 1 for row in p):
        raise ValueError("not an exact stochastic kernel")
    if any(sum(p[i][j] for i in range(n)) != 1 for j in range(n)):
        raise ValueError("this certificate requires exact uniform stationarity")
    return p


def controlled_kernel(p: list[list[F]], lam: F) -> list[list[F]]:
    n, h = len(p), len(p)//2
    if not 0 <= lam <= 1:
        raise ValueError("lambda outside [0,1]")
    return [[(1-lam)*p[i][j] + (lam/h if (i<h) == (j<h) else 0)
             for j in range(n)] for i in range(n)]


def structural_metrics(p: list[list[F]], tau: int) -> tuple[F, F]:
    """Exact CE and M_obj from A=P**tau Q, using disjoint lift support."""
    if isinstance(tau, bool) or int(tau) != tau or tau < 1:
        raise ValueError("structural audit requires positive integer tau")
    n, h = len(p), len(p)//2
    # Two columns sum to one, so the first column determines A entirely.
    a = [sum(row[:h]) for row in p]
    for _ in range(int(tau) - 1):
        a = [sum(v * a[j] for j, v in enumerate(row) if v) for row in p]
    b0, b1 = sum(a[:h])/h, sum(a[h:])/h
    ce = max(abs(v - (v*b0 + (1-v)*b1)) for v in a)
    mobj = max(F(0), min(F(1), b0-b1))  # trace(U A)-1 for k=2
    return ce, mobj


def entropy_lower_bound(p: list[list[F]]) -> F | None:
    """Exact lower bound using (a-b)log(a/b) >= 2(a-b)^2/(a+b).

    Uniform stationarity is checked by the constructor. None denotes infinity.
    """
    n = len(p)
    if any(sum(row) != 1 for row in p) or any(sum(p[i][j] for i in range(n)) != 1 for j in range(n)):
        raise ValueError("entropy bound requires a doubly stochastic kernel")
    bound = F(0)
    for i in range(n):
        for j in range(i+1, n):
            a, b = p[i][j], p[j][i]
            if (a == 0) != (b == 0):
                return None
            if a+b:
                bound += 2*(a-b)**2/(a+b)/n
    return bound


def _crossing(rows: list[dict[str, F]], key: str, target: F, below: bool) -> F | None:
    for i, row in enumerate(rows):
        y, x = row[key], row["lambda"]
        if not (y <= target if below else y >= target):
            continue
        if i == 0:
            return x
        prev = rows[i-1]
        return prev["lambda"] + (target-prev[key])*(x-prev["lambda"])/(y-prev[key])
    return None


def _interpolate(rows: list[dict[str, F]], key: str, lam: F) -> F:
    for i, row in enumerate(rows):
        if row["lambda"] == lam:
            return row[key]
        if row["lambda"] > lam:
            prev = rows[i-1]
            t = (lam-prev["lambda"])/(row["lambda"]-prev["lambda"])
            return (1-t)*prev[key] + t*row[key]
    raise ValueError("reference outside sampled curve")


def audit_panel(p: list[list[F]], lambda_grid: list[Any]) -> dict[str, Any]:
    lambdas = sorted(rational(v) for v in lambda_grid)
    if len(set(lambdas)) != len(lambdas):
        raise ValueError("duplicate grid points")
    curves: dict[int, list[dict[str, F]]] = {1: [], 2: []}
    for lam in lambdas:
        controlled = controlled_kernel(p, lam)
        for tau in (1, 2):
            ce, mo = structural_metrics(controlled, tau)
            curves[tau].append({"lambda": lam, "ce": ce, "mobj": mo})
    boundaries = {tau: {"ce": _crossing(rows, "ce", F(1,40), True),
                         "mobj": _crossing(rows, "mobj", F(9,10), False)}
                  for tau, rows in curves.items()}
    if any(v is None for b in boundaries.values() for v in b.values()):
        raise ValueError("canonical certificate requires both proxies in both channels")
    for tau, boundary in boundaries.items():
        ref = max(boundary.values())
        if _interpolate(curves[tau], "ce", ref) > F(1,40) or _interpolate(curves[tau], "mobj", ref) < F(9,10):
            raise ValueError("separate proxy crossings do not certify joint structural birth")
    shift_ce = abs(boundaries[2]["ce"] - boundaries[1]["ce"])
    shift_mo = abs(boundaries[2]["mobj"] - boundaries[1]["mobj"])
    reference = max(boundaries[1].values())
    # Ref is an interpolated curve coordinate. Bound the matching interpolated
    # entropy readout by interpolating exact lower bounds at the enclosing samples.
    entropy_samples = [entropy_lower_bound(controlled_kernel(p, lam)) for lam in lambdas]
    if any(v is None for v in entropy_samples):
        raise ValueError("canonical finite-affinity certificate has one-way support")
    entropy_bound = entropy_samples[-1]
    for i, lam in enumerate(lambdas):
        if lam == reference:
            entropy_bound = entropy_samples[i]
            break
        if lam > reference:
            t = (reference-lambdas[i-1])/(lam-lambdas[i-1])
            entropy_bound = (1-t)*entropy_samples[i-1] + t*entropy_samples[i]
            break
    reversible = all(p[i][j] == p[j][i] for i in range(len(p)) for j in range(len(p)))
    p6 = "inactive" if reversible else "active" if entropy_bound >= F(1,1000) else "unknown"
    p4 = min(shift_ce, shift_mo) >= F(3,20)
    label = {("inactive", False): "Class-I", ("active", False): "Class-II",
             ("inactive", True): "Class-III", ("active", True): "Class-IV"}.get((p6,p4), "unclassified")
    return {"boundaries": boundaries, "shift_ce": shift_ce, "shift_mobj": shift_mo,
            "affinity_ref_lower_bound": entropy_bound, "exact_reversibility": reversible,
            "p6_state": p6, "p4_active": p4, "canonical_class_label": label,
            "curves": curves}
