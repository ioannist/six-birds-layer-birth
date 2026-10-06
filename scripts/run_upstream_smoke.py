#!/usr/bin/env python3
"""Run a minimal vendored upstream numeric smoke and emit a result bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

from layerbirth.manifest import create_dry_run_bundle


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _run_vendored_numeric(vendored_src: Path, tau: int, lens_assignment: list[int]) -> dict[str, float]:
    if str(vendored_src) not in sys.path:
        sys.path.insert(0, str(vendored_src))

    from closurelab.numeric import (  # type: ignore[import-not-found]
        build_Q,
        build_U_uniform,
        empirical_endomap,
        idempotence_defect_tv,
        macro_kernel,
    )

    p = np.array(
        [
            [0.70, 0.20, 0.05, 0.05],
            [0.15, 0.70, 0.10, 0.05],
            [0.10, 0.10, 0.65, 0.15],
            [0.05, 0.10, 0.20, 0.65],
        ],
        dtype=float,
    )
    f = np.asarray(lens_assignment, dtype=int)
    q = build_Q(f, 2)
    u = build_U_uniform(f, 2)
    endomap = empirical_endomap(p, tau, q, u)
    macro = macro_kernel(p, tau, q, u)
    return {
        "idempotence_defect_tv": float(idempotence_defect_tv(endomap)),
        "macro_trace": float(np.trace(macro)),
        "endomap_l1_sum": float(np.abs(endomap).sum()),
    }


def run(config_path: Path, output_root_override: Path | None) -> Path:
    repo_root = _repo_root()
    config = _load_config(config_path)
    experiment_id = str(config["experiment_id"])
    output_root = (
        output_root_override
        if output_root_override is not None
        else repo_root / str(config["output_root"])
    )
    vendored_path = repo_root / str(config["vendored_path"])
    vendored_src = vendored_path / "src"
    vendored_entrypoint = str(config["vendored_entrypoint"])
    tau = int(config["parameters"]["tau"])
    lens_assignment = list(config["parameters"]["lens_assignment"])

    start = time.perf_counter()
    numeric = _run_vendored_numeric(vendored_src, tau=tau, lens_assignment=lens_assignment)
    elapsed_seconds = time.perf_counter() - start

    manifest_path = create_dry_run_bundle(experiment_id, output_root)
    bundle_root = manifest_path.parent

    config_snapshot_path = bundle_root / "config" / "config_snapshot.json"
    config_snapshot_path.write_text(
        json.dumps(
            {
                "upstream_smoke_config": config,
                "resolved_vendored_src": str(vendored_src),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics_path = bundle_root / "metrics" / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["status", "success"])
        writer.writerow(["elapsed_seconds", f"{elapsed_seconds:.6f}"])
        writer.writerow(["vendored_entrypoint", vendored_entrypoint])
        writer.writerow(["idempotence_defect_tv", f"{numeric['idempotence_defect_tv']:.12f}"])
        writer.writerow(["macro_trace", f"{numeric['macro_trace']:.12f}"])
        writer.writerow(["endomap_l1_sum", f"{numeric['endomap_l1_sum']:.12f}"])

    notes_path = bundle_root / "notes" / "findings.md"
    notes_path.write_text(
        "\n".join(
            [
                "# Upstream smoke findings",
                "",
                f"- vendored path: `{config['vendored_path']}`",
                f"- vendored entrypoint: `{vendored_entrypoint}`",
                f"- idempotence_defect_tv: `{numeric['idempotence_defect_tv']:.12f}`",
                f"- macro_trace: `{numeric['macro_trace']:.12f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env_path = bundle_root / "env" / "environment.json"
    env_payload = json.loads(env_path.read_text(encoding="utf-8"))
    env_payload["numpy_version"] = np.__version__
    env_payload["vendored_path"] = str(vendored_path)
    env_path.write_text(json.dumps(env_payload, indent=2) + "\n", encoding="utf-8")

    log_path = bundle_root / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "upstream smoke wrapper completed",
                f"elapsed_seconds={elapsed_seconds:.6f}",
                f"vendored_entrypoint={vendored_entrypoint}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/run_upstream_smoke.py")
    parser.add_argument(
        "--config",
        default="configs/upstream_smoke.yaml",
        help="Path to wrapper config (JSON-compatible YAML).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root override (defaults to config value).",
    )
    args = parser.parse_args(argv)

    manifest_path = run(
        config_path=Path(args.config),
        output_root_override=Path(args.output_root) if args.output_root else None,
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
