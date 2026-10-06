#!/usr/bin/env python3
"""Smoke driven and holonomy-control substrate families."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from layerbirth.metrics import (
    affinity_metric,
    closure_error,
    holonomy_metric,
    objecthood_order_parameter,
    staging_gap,
)
from layerbirth.substrates import build_substrate_family


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _half_ring_lens() -> np.ndarray:
    return np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)


def _coarse_metrics(P: np.ndarray, lens: np.ndarray, k: int) -> tuple[float, float, float, float]:
    ce, _ = closure_error(P, 1, lens, k)
    mobj, _ = objecthood_order_parameter(P, 1, lens, k)
    sg, _ = staging_gap(P, 1, lens, k)
    aff, _ = affinity_metric(P, tau=1)
    return float(ce), float(mobj), float(sg), float(aff)


def _run_driven(case_name: str, cfg: dict) -> dict:
    built = build_substrate_family(cfg["name"], **cfg["params"])
    p = np.asarray(built["P"], dtype=np.float64)
    lens = _half_ring_lens()
    k = 2
    ce, mobj, sg, aff = _coarse_metrics(p, lens, k)
    return {
        "case_name": case_name,
        "family_name": built["family_name"],
        "n": int(built["n"]),
        "analysis_k": int(k),
        "closure_error": ce,
        "objecthood_order": mobj,
        "staging_gap": sg,
        "affinity": aff,
        "holonomy": None,
        "seed": cfg.get("params", {}).get("seed"),
    }


def _run_holonomy(case_name: str, cfg: dict) -> dict:
    built = build_substrate_family(cfg["name"], **cfg["params"])
    p = np.asarray(built["P"], dtype=np.float64)
    coarse = np.asarray(built["coarse_lens"], dtype=int)
    fine = np.asarray(built["fine_lens"], dtype=int)
    ce, mobj, sg, aff = _coarse_metrics(p, coarse, 2)
    hol, _ = holonomy_metric(fine, 4, coarse, 2)
    return {
        "case_name": case_name,
        "family_name": built["family_name"],
        "n": int(built["n"]),
        "analysis_k": 2,
        "closure_error": ce,
        "objecthood_order": mobj,
        "staging_gap": sg,
        "affinity": aff,
        "holonomy": float(hol),
        "seed": cfg.get("params", {}).get("seed"),
    }


def main() -> int:
    root = _repo_root()
    cfg_root = root / "configs" / "substrates"
    out_root = root / "results" / "substrate_smoke" / "driven_holonomy"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = [
        _run_driven("driven_cycle_low_bias", _load_json(cfg_root / "driven_cycle_low_bias.json")),
        _run_driven("driven_cycle_high_bias", _load_json(cfg_root / "driven_cycle_high_bias.json")),
        _run_holonomy(
            "holonomy_control_balanced", _load_json(cfg_root / "holonomy_control_balanced.json")
        ),
        _run_holonomy(
            "holonomy_control_unbalanced",
            _load_json(cfg_root / "holonomy_control_unbalanced.json"),
        ),
    ]

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family_name",
                "case_name",
                "n",
                "analysis_k",
                "closure_error",
                "objecthood_order",
                "staging_gap",
                "affinity",
                "holonomy",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_root / "summary.json"
    payload = {"rows": rows, "summary_csv": str(csv_path), "summary_json": str(json_path)}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
