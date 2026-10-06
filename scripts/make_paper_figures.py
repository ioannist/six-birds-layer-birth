#!/usr/bin/env python3
"""Render the manuscript figures into paper/figures/.

Inputs are the exact rational audit (notes/math_review/canonical_exact_audit.json),
the frozen result bundles under results/, and exact kernels rebuilt with
layerbirth.exact_audit. Affinity curves are evaluated in floating point from the
exact rational kernels (uniform stationary law, checked by the constructor).
"""

from __future__ import annotations

import csv
import json
import math
from fractions import Fraction as F
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from layerbirth.exact_audit import canonical_kernel, controlled_kernel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

# Validated categorical slots (blue, orange, aqua, violet); identity is also carried
# by marker shape and direct labels.
CLASS_COLOR = {"Class-I": "#2a78d6", "Class-II": "#eb6834", "Class-III": "#1baf7a", "Class-IV": "#4a3aa7"}
CLASS_MARKER = {"Class-I": "o", "Class-II": "s", "Class-III": "D", "Class-IV": "^"}
TAU_COLOR = {"1": "#2a78d6", "2": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SIZE_RAMP = ["#a9c8ef", "#6aa1e3", "#2a78d6", "#174a8a"]  # sequential, light -> dark

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.5,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def load_json(rel: str, **kw):
    return json.loads((ROOT / rel).read_text(), **kw)


def representatives():
    """(label, family, size, base config, drive config, lambda grid) for the four exact witnesses."""
    rubric = load_json("configs/taxonomy/canonical_class_rubric.json", parse_float=F)
    c3 = load_json("configs/campaigns/class_iii_full_campaign.json", parse_float=F)
    c4 = load_json("configs/campaigns/class_iv_full_campaign.json", parse_float=F)
    reps = []
    for prof, label in zip(rubric["reference_profiles"], ("Class-I", "Class-II")):
        reps.append((label, prof["profile_name"], prof["family_name"], int(prof["size"]), prof, None, prof["lambda_grid"]))
    for cfg, label in ((c3, "Class-III"), (c4, "Class-IV")):
        cand = cfg["candidate"]
        reps.append((label, cand["candidate_name"], cand["family_name"], 64, cand["base_kwargs"],
                     cand.get("drive_kwargs"), cfg["lambda_grid"]))
    return reps


def reversal_divergence(p, lam) -> float:
    q = controlled_kernel(p, F(lam))
    n = len(q)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = float(q[i][j]) / n, float(q[j][i]) / n
            if a > 0 and b > 0:
                total += (a - b) * math.log(a / b)
            elif a != b:
                return math.inf
    return max(total, 0.0)


def exact_record(name: str, size: int):
    audit = load_json("notes/math_review/canonical_exact_audit.json")
    for rec in audit["records"]:
        if rec["representative"] == name and rec["size"] == size:
            return rec
    raise KeyError((name, size))


def fig_structural_curves(reps):
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharey="row")
    for col, (label, name, _fam, size, *_rest) in enumerate(reps):
        rec = exact_record(name, size)
        ax_m, ax_c = axes[0, col], axes[1, col]
        for tau in ("1", "2"):
            rows = rec["curves"][tau]
            lam = [float(F(r["lambda"])) for r in rows]
            ax_m.plot(lam, [float(F(r["mobj"])) for r in rows], color=TAU_COLOR[tau], marker="o", ms=2.2,
                      label=fr"$\tau={tau}$")
            ax_c.plot(lam, [float(F(r["ce"])) for r in rows], color=TAU_COLOR[tau], marker="o", ms=2.2)
            b = rec["boundaries"][tau]
            for ax, key in ((ax_m, "mobj"), (ax_c, "ce")):
                if b[key] is not None:
                    ax.axvline(float(F(b[key])), color=TAU_COLOR[tau], lw=0.8, ls=":")
        ax_m.axhline(0.9, color=MUTED, lw=0.8, ls="--")
        ax_c.axhline(0.025, color=MUTED, lw=0.8, ls="--")
        ax_m.set_title(f"{label}  (N={size})", color=INK)
        shift_ce, shift_m = float(F(rec["shift_ce"])), float(F(rec["shift_mobj"]))
        ax_c.text(0.04, 0.97, rf"$\Delta\lambda_{{\mathrm{{CE}}}}={shift_ce:.3f}$" "\n"
                  rf"$\Delta\lambda_{{M}}={shift_m:.3f}$", transform=ax_c.transAxes, ha="left", va="top",
                  fontsize=7, color=INK)
        ax_c.set_xlabel(r"closure strength $\lambda$")
        for ax in (ax_m, ax_c):
            ax.set_xlim(0, 1.02)
    axes[0, 0].set_ylabel(r"objecthood $M_{\mathrm{obj}}$")
    axes[1, 0].set_ylabel("closure error CE")
    axes[0, 0].set_ylim(0.6, 1.02)
    axes[1, 0].set_ylim(-0.003, 0.125)
    axes[0, 3].legend(loc="lower right", frameon=False)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(OUT / "structural_curves.pdf")
    plt.close(fig)


def fig_drive_curves(reps):
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    floor = 1e-6
    for label, _name, fam, size, base, drive, grid in reps:
        p = canonical_kernel(fam, size, base, drive)
        lam = [float(x) for x in grid]
        aff = [reversal_divergence(p, x) for x in grid]
        y = [max(a, floor) for a in aff]
        exact_zero = all(a == 0.0 for a in aff)
        ax.plot(lam, y, color=CLASS_COLOR[label], marker=CLASS_MARKER[label], ms=3.2,
                ls="--" if exact_zero else "-", label=f"{label} (N={size})", zorder=3 if label == "Class-I" else 2)
        if not exact_zero:
            ax.annotate(f"{label} (N={size})", (lam[0], y[0]), xytext=(0, 6), textcoords="offset points",
                        fontsize=7, color=INK, ha="left" if label == "Class-IV" else "center")
    ax.text(0.05, 1.6e-6, "Class-I (N=32, from $\\lambda=0.55$) and Class-III (N=64): exactly 0",
            fontsize=7, color=INK)
    ax.axhline(1e-3, color=MUTED, lw=0.8, ls="--")
    ax.text(0.02, 1.15e-3, r"active threshold $10^{-3}$", fontsize=7, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 0.2)
    ax.set_xlim(0, 1.03)
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["0 (exact)", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    ax.set_xlabel(r"closure strength $\lambda$")
    ax.set_ylabel(r"affinity $\mathrm{Aff}(\lambda)$")
    fig.tight_layout()
    fig.savefig(OUT / "drive_curves.pdf")
    plt.close(fig)


def fig_observable_map():
    coords = load_json("results/dashboards/four_class_observable_map/analysis/class_coordinates.json")["class_coordinates"]
    with (ROOT / "results/bridges/upstream_pica_bridge/analysis/upstream_candidate_table.csv").open() as fh:
        upstream = list(csv.DictReader(fh))
    floor = -12.0
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.axhspan(floor - 0.9, floor + 0.9, color="#f1f0ec", zorder=0, lw=0)
    ax.text(0.60, floor - 0.6, r"floor: $\mathrm{Aff}_{\mathrm{ref}}<10^{-12}$ (exactly 0 for Class-I, Class-III)",
            fontsize=7, color=MUTED)
    ax.axhline(-3, color=MUTED, lw=0.8, ls="--")
    ax.text(0.985, -2.75, r"drive-active threshold $10^{-3}$", transform=ax.get_yaxis_transform(),
            ha="right", fontsize=7, color=MUTED)
    for row in upstream:
        x = float(row["structural_boundary_lambda_ref"])
        ax.scatter([x], [floor], marker="x", s=26, color=MUTED, lw=1.0, zorder=3)
    ax.annotate("upstream feature kernels\n(3 candidates, all Class-I)", (0.969, floor), xytext=(1.0, -8.6),
                textcoords="data", ha="right", fontsize=7, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    for c in coords:
        label = c["class_name"]
        x, y = c["structural_boundary_lambda_ref"], c["log10_affinity_ref_clipped"]
        active = c["p4_class_active"]
        ax.scatter([x], [y], s=70, marker=CLASS_MARKER[label], zorder=4, linewidths=1.6,
                   facecolors=CLASS_COLOR[label] if active else "white", edgecolors=CLASS_COLOR[label])
        if c["ce_boundary_left_censored"] or c["mobj_boundary_left_censored"]:
            ax.annotate("", (x - 0.03, y), xytext=(x - 0.008, y),
                        arrowprops=dict(arrowstyle="->", color=CLASS_COLOR[label], lw=1.2))
        text = f"{label}\n$P4_{{\\mathrm{{dual}}}}={c['p4_dual_shift_min']:.3f}$"
        right = label != "Class-I"
        ax.annotate(text, (x, y), xytext=(9 if right else -9, 5), textcoords="offset points", fontsize=7.5,
                    color=INK, va="bottom", ha="left" if right else "right")
    ax.set_xlim(0.44, 1.02)
    ax.set_ylim(floor - 0.9, 0)
    ax.set_xlabel(r"structural boundary $\lambda_{\mathrm{struct,ref}}$  (N = 64)")
    ax.set_ylabel(r"$\log_{10}\mathrm{Aff}_{\mathrm{ref}}$ (clipped at $-12$)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", mfc="#777", mec="#777", label="$P4$ class-active (filled)"),
               Line2D([], [], marker="o", ls="", mfc="white", mec="#777", label="$P4$ inactive (hollow)"),
               Line2D([], [], marker=r"$\leftarrow$", ls="", color="#777", ms=9, label="left-censored boundary")]
    ax.legend(handles=handles, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55))
    fig.tight_layout()
    fig.savefig(OUT / "observable_map.pdf")
    plt.close(fig)


def fig_shadow_panels():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (tag, label) in zip(axes, (("iii", "Class-III"), ("iv", "Class-IV"))):
        with (ROOT / f"results/phenomenology/class_{tag}_shadow_panel/analysis/group_summary.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        for color, size in zip(SIZE_RAMP, ("16", "32", "64", "128")):
            sel = sorted((r for r in rows if r["size"] == size), key=lambda r: float(r["closure_strength_lambda"]))
            ax.plot([float(r["closure_strength_lambda"]) for r in sel],
                    [float(r["susceptibility_delta_mobj"]) * 1e5 for r in sel],
                    color=color, marker="o", ms=2.2, label=f"N={size}")
        ax.set_title(f"{label} shadow ensemble", color=INK)
        ax.set_xlabel(r"closure strength $\lambda$")
    axes[0].set_ylabel(r"$\chi = N\,\mathrm{Var}(\Delta M_{\mathrm{obj}})$  [$\times 10^{-5}$]")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "shadow_susceptibility.pdf")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reps = representatives()
    fig_structural_curves(reps)
    fig_drive_curves(reps)
    fig_observable_map()
    fig_shadow_panels()
    print(f"figures written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
