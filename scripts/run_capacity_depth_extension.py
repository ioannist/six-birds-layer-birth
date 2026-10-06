#!/usr/bin/env python3
"""Run adapted LB-19 capacity/depth extension."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.capacity import run_capacity_depth_extension


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "campaigns" / "capacity_depth_extension.json"
    out = root / "results" / "campaigns"
    summary = run_capacity_depth_extension(cfg, output_root=out, use_cache=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
