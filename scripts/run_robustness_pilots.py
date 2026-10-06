#!/usr/bin/env python3
"""Run LB-13 robustness pilots."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.robustness import run_robustness_pilots


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "robustness"
    summary = run_robustness_pilots(output_root=out, use_cache=True)
    compact = {
        "families": [
            {
                "family_name": row["family_name"],
                "variant_count": row["variant_count"],
                "robust_enough_to_continue": row["robust_enough_to_continue"],
                "artifact_root": row["artifact_root"],
            }
            for row in summary["families"]
        ]
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
