#!/usr/bin/env python3
"""Run S2-04 replicated-portal refinement and P4 criterion audit."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.class3 import run_class_iii_refinement_and_p4_audit


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "pilots" / "class_iii_refinement_and_p4_audit.json"
    out = root / "results" / "pilots"
    summary = run_class_iii_refinement_and_p4_audit(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
