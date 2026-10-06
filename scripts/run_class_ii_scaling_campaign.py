#!/usr/bin/env python3
"""Run LB-16 class-II scaling campaign."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.campaigns import run_class_ii_campaign_driver


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "campaigns"
    summary = run_class_ii_campaign_driver(output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
