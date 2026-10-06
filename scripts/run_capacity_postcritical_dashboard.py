#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from layerbirth.capacity_dashboard import build_capacity_postcritical_dashboard


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = build_capacity_postcritical_dashboard(
        root / "configs" / "dashboards" / "capacity_postcritical_dashboard.json",
        output_root=root / "results" / "dashboards",
        use_cache=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
