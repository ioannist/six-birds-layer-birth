#!/usr/bin/env python3
"""Run S2-03 P4 anomaly metric layer."""

from __future__ import annotations

import json
from pathlib import Path

from layerbirth.p4 import run_p4_anomaly_metric_layer


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "metrics" / "p4_anomaly_metric_layer.json"
    out = root / "results" / "metrics"
    summary = run_p4_anomaly_metric_layer(cfg, output_root=out, use_cache=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
