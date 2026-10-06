#!/usr/bin/env python3
"""Run S2-01 canonical class rubric classification."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.taxonomy import run_canonical_class_rubric


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "taxonomy" / "canonical_class_rubric.json"
    out = root / "results" / "taxonomy"
    summary = run_canonical_class_rubric(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
