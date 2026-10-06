#!/usr/bin/env python3
"""Run LB-17 affinity visibility and degeneracy audit."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.affinity_visibility import run_affinity_visibility_audit


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "campaigns" / "affinity_visibility_audit.json"
    out = root / "results" / "campaigns"
    summary = run_affinity_visibility_audit(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
