#!/usr/bin/env python3
"""Run the LB-11 tiny-grid sweep."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.sweep import run_sweep


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "sweeps" / "tiny_grid.json"
    out = root / "results" / "sweeps"
    summary = run_sweep(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
