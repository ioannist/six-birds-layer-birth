#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from layerbirth.freeze import build_evidence_ledger


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = build_evidence_ledger(
        root / "configs" / "freeze" / "reproducibility_freeze.json",
        output_root=root / "results" / "freeze",
        use_cache=False,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
