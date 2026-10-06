#!/usr/bin/env python3
"""Run LB-17 crossover decision gate and affinity-boundary reconnaissance."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.crossover_decision import run_crossover_decision_affinity_boundary


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "campaigns" / "crossover_decision_affinity_boundary.json"
    out = root / "results" / "campaigns"
    summary = run_crossover_decision_affinity_boundary(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
