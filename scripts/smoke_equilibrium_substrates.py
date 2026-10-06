#!/usr/bin/env python3
"""Smoke equilibrium-like substrate families and write compact summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from layerbirth.metrics import affinity_metric, closure_error, objecthood_order_parameter, staging_gap
from layerbirth.substrates import build_equilibrium_substrate_family


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _run_case(case_name: str, cfg: dict) -> dict:
    built = build_equilibrium_substrate_family(cfg["name"], **cfg["params"])
    p = np.asarray(built["P"], dtype=np.float64)
    n = int(built["n"])

    if built.get("block_lens") is None:
        lens = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    else:
        lens = np.asarray(built["block_lens"], dtype=int)
    k = int(np.max(lens)) + 1

    ce, _ = closure_error(p, 1, lens, k)
    mobj, _ = objecthood_order_parameter(p, 1, lens, k)
    sg, _ = staging_gap(p, 1, lens, k)
    aff, _ = affinity_metric(p, tau=1)

    return {
        "case_name": case_name,
        "family_name": built["family_name"],
        "n": n,
        "analysis_k": k,
        "closure_error": float(ce),
        "objecthood_order": float(mobj),
        "staging_gap": float(sg),
        "affinity": float(aff),
        "seed": cfg.get("params", {}).get("seed"),
    }


def main() -> int:
    root = _repo_root()
    cfg_root = root / "configs" / "substrates"
    out_root = root / "results" / "substrate_smoke" / "equilibrium_like"
    out_root.mkdir(parents=True, exist_ok=True)

    cases = [
        ("reversible_strong", _load_json(cfg_root / "reversible_block_strong_separation.json")),
        ("reversible_weak", _load_json(cfg_root / "reversible_block_weak_separation.json")),
        ("metastable_seeded", _load_json(cfg_root / "metastable_block_seeded.json")),
        ("null_flat_mixing", _load_json(cfg_root / "null_flat_mixing.json")),
    ]
    rows = [_run_case(name, cfg) for name, cfg in cases]

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_name",
                "family_name",
                "n",
                "analysis_k",
                "closure_error",
                "objecthood_order",
                "staging_gap",
                "affinity",
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
