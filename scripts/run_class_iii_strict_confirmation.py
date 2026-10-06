#!/usr/bin/env python3
"""Run S2-05 Class-III strict confirmation pilot."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.class3 import run_class_iii_candidate_pilots


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "pilots" / "class_iii_strict_confirmation.json"
    out = root / "results" / "pilots"
    summary = run_class_iii_candidate_pilots(cfg, output_root=out, use_cache=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
