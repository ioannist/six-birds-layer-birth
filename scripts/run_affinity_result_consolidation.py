#!/usr/bin/env python3
"""Run S2-09 affinity result consolidation."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.class3 import run_affinity_result_consolidation


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "dashboards" / "affinity_result_consolidation.json"
    out = root / "results" / "dashboards"
    summary = run_affinity_result_consolidation(cfg, output_root=out, use_cache=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
