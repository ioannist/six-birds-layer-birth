#!/usr/bin/env python3
"""Smoke protocol-trap and no-fake-arrow controls."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from layerbirth.controls import evaluate_control_case, no_fake_arrow_controls, protocol_trap_control


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def main() -> int:
    root = _repo_root()
    cfg_root = root / "configs" / "controls"
    out_root = root / "results" / "control_smoke" / "arrow_controls"
    out_root.mkdir(parents=True, exist_ok=True)

    hidden_cfg = _load_json(cfg_root / "protocol_trap_hidden_schedule.json")
    phase_cfg = _load_json(cfg_root / "protocol_trap_phase_aware.json")
    nofake_cfg = _load_json(cfg_root / "no_fake_arrow_controls.json")

    rows: list[dict] = []

    hidden = protocol_trap_control(
        pair_self_weight=float(hidden_cfg["pair_self_weight"]),
        schedule_order=list(hidden_cfg["schedule_order"]),
        handling=str(hidden_cfg["handling"]),
    )
    rows.append(
        evaluate_control_case(
            case_name="protocol_trap_hidden_schedule",
            control_family="protocol_trap",
            P=np.asarray(hidden["hidden_effective_kernel"], dtype=np.float64),
            analysis_lens=np.asarray(hidden["analysis_lens"], dtype=np.int64),
        )
    )

    phase = protocol_trap_control(
        pair_self_weight=float(phase_cfg["pair_self_weight"]),
        schedule_order=list(phase_cfg["schedule_order"]),
        handling=str(phase_cfg["handling"]),
    )
    for phase_name in phase_cfg["schedule_order"]:
        rows.append(
            evaluate_control_case(
                case_name=f"protocol_trap_phase_aware_{phase_name}",
                control_family="protocol_trap",
                P=np.asarray(phase["phase_kernels"][phase_name], dtype=np.float64),
                analysis_lens=np.asarray(phase["analysis_lens"], dtype=np.int64),
            )
        )

    for case in no_fake_arrow_controls(cases_config=nofake_cfg["cases"]):
        rows.append(
            evaluate_control_case(
                case_name=case["case_name"],
                control_family="no_fake_arrow",
                P=np.asarray(case["P"], dtype=np.float64),
                analysis_lens=np.asarray(case["analysis_lens"], dtype=np.int64),
            )
        )

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_name",
                "control_family",
                "n",
                "analysis_k",
                "closure_error",
                "objecthood_order",
                "staging_gap",
                "affinity",
                "holonomy",
                "driven_candidate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_root / "summary.json"
    payload = {"rows": rows, "summary_csv": str(csv_path), "summary_json": str(json_path)}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
