#!/usr/bin/env python3
"""Run S2-05 full Class-III campaign."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.class3 import run_class_iii_full_campaign


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "campaigns" / "class_iii_full_campaign.json"
    out = root / "results" / "campaigns"
    summary = run_class_iii_full_campaign(cfg, output_root=out, use_cache=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
