#!/usr/bin/env python3
"""Run LB-17 FSS stability and size-extension audit."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.fss_stability import run_fss_stability_audit


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "campaigns" / "fss_stability_size_extension.json"
    out = root / "results" / "campaigns"
    summary = run_fss_stability_audit(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
