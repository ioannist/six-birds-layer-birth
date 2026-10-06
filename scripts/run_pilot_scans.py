#!/usr/bin/env python3
"""Run all LB-12 pilot scans."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.pilots import run_all_lb12_pilots


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "pilots"
    summary = run_all_lb12_pilots(output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
