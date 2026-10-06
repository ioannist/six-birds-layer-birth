#!/usr/bin/env python3
"""Run S2-08 four-class synthesis dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.class3 import run_four_class_synthesis_dashboard


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "dashboards" / "four_class_synthesis.json"
    out = root / "results" / "dashboards"
    summary = run_four_class_synthesis_dashboard(cfg, output_root=out, use_cache=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
