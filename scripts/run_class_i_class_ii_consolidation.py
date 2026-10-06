#!/usr/bin/env python3
"""Run S2-02 Class-I / Class-II consolidation dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.dashboard import build_class_consolidation_dashboard


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "dashboards" / "class_i_class_ii_consolidation.json"
    out = root / "results" / "dashboards"
    summary = build_class_consolidation_dashboard(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
