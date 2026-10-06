#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from layerbirth.robustness import run_class_iii_class_iv_lens_suite


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = run_class_iii_class_iv_lens_suite(
        root / "configs" / "robustness" / "class_iii_class_iv_lens_suite.json",
        output_root=root / "results" / "robustness",
        use_cache=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
