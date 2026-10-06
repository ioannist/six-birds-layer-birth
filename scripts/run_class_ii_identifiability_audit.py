#!/usr/bin/env python3
"""Run LB-17 class-II identifiability and observable audit."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.identifiability import run_class_ii_identifiability_audit


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "campaigns" / "class_ii_identifiability_audit.json"
    out = root / "results" / "campaigns"
    summary = run_class_ii_identifiability_audit(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
