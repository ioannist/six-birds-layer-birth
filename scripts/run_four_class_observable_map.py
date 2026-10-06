#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from layerbirth.observable_map import build_four_class_observable_map


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = build_four_class_observable_map(
        root / "configs" / "dashboards" / "four_class_observable_map.json",
        output_root=root / "results" / "dashboards",
        use_cache=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
