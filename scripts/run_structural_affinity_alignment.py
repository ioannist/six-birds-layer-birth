#!/usr/bin/env python3
"""Run LB-17 structural-boundary vs affinity-onset alignment campaign."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.alignment import run_structural_affinity_alignment


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "campaigns" / "structural_affinity_alignment.json"
    out = root / "results" / "campaigns"
    summary = run_structural_affinity_alignment(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
