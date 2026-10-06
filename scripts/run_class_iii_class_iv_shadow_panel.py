#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from layerbirth.phenomenology import run_class_iii_class_iv_shadow_panel


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = run_class_iii_class_iv_shadow_panel(
        root / "configs" / "phenomenology" / "class_iii_class_iv_shadow_panel.json",
        output_root=root / "results" / "phenomenology",
        use_cache=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
