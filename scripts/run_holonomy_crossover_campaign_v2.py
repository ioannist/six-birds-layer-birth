#!/usr/bin/env python3
"""Run LB-17 holonomy crossover campaign v2."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.crossover import run_holonomy_crossover_campaign_v2


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "campaigns" / "holonomy_crossover_v2.json"
    out = root / "results" / "campaigns"
    summary = run_holonomy_crossover_campaign_v2(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
