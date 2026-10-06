#!/usr/bin/env python3
"""Smoke fixed/adaptive/matched tau protocols and emit result bundles."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from layerbirth.contracts import load_schema, validate_manifest
from layerbirth.protocols import resolve_adaptive_tau, resolve_fixed_tau, resolve_matched_tau_group


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_code_version(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"git_commit": commit or "unknown", "git_dirty": bool(dirty)}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": False}


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_bundle(run_id: str, resolved: dict, output_root: Path, schema: dict) -> Path:
    bundle_root = output_root / run_id
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    config_dir = bundle_root / "config"
    seeds_dir = bundle_root / "seeds"
    metrics_dir = bundle_root / "metrics"
    plots_dir = bundle_root / "plots"
    notes_dir = bundle_root / "notes"
    env_dir = bundle_root / "env"
    for p in (config_dir, seeds_dir, metrics_dir, plots_dir, notes_dir, env_dir):
        p.mkdir(parents=True, exist_ok=True)

    config_snapshot = config_dir / "config_snapshot.json"
    seed_list = seeds_dir / "seeds.json"
    metrics_csv = metrics_dir / "metrics.csv"
    notes_file = notes_dir / "findings.md"
    env_file = env_dir / "environment.json"

    config_snapshot.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    seed_list.write_text("[]\n", encoding="utf-8")

    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["tau_protocol_name", resolved["tau_protocol_name"]])
        writer.writerow(["resolved_tau", resolved["resolved_tau"]])
        writer.writerow(["closure_strength_lambda", resolved["closure_strength_lambda"]])
        if resolved["tau_protocol_name"] == "adaptive":
            d = resolved["tau_resolution_details"]
            writer.writerow(["mu2_abs", d["mu2_abs"]])
            writer.writerow(["spectral_gap", d["spectral_gap"]])
            writer.writerow(["tau_raw", d["tau_raw"]])
        if resolved["tau_protocol_name"] == "matched":
            writer.writerow(["match_group_id", resolved["match_group_id"]])
            writer.writerow(["matched_reference_run_id", resolved["matched_reference_run_id"]])
            writer.writerow(["matched_role", resolved["matched_role"]])

    notes_file.write_text(
        "\n".join(
            [
                "# Protocol smoke findings",
                "",
                f"- run_id: `{run_id}`",
                f"- tau_protocol_name: `{resolved['tau_protocol_name']}`",
                f"- resolved_tau: `{resolved['resolved_tau']}`",
                f"- closure_strength_lambda: `{resolved['closure_strength_lambda']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file.write_text(
        json.dumps(
            {
                "python": (
                    subprocess.run(
                        ["python", "--version"], capture_output=True, text=True, check=False
                    ).stdout.strip()
                    or subprocess.run(
                        ["python", "--version"], capture_output=True, text=True, check=False
                    ).stderr.strip()
                ),
                "platform": "unknown",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "1",
        "experiment_id": "protocol_smoke",
        "bundle_id": run_id,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(_repo_root()),
        "config_snapshot_path": str(config_snapshot.relative_to(bundle_root)),
        "seed_list_path": str(seed_list.relative_to(bundle_root)),
        "metrics_table_path": str(metrics_csv.relative_to(bundle_root)),
        "plots_dir_path": str(plots_dir.relative_to(bundle_root)),
        "notes_file_path": str(notes_file.relative_to(bundle_root)),
        "environment_snapshot_path": str(env_file.relative_to(bundle_root)),
    }
    validate_manifest(manifest, schema)
    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    root = _repo_root()
    cfg_dir = root / "configs" / "protocols"
    fixed_cfg = _read_json(cfg_dir / "fixed_tau.json")
    adaptive_cfg = _read_json(cfg_dir / "adaptive_tau.json")
    matched_cfg = _read_json(cfg_dir / "matched_tau_group.json")

    p_slow = [
        [0.9, 0.1],
        [0.1, 0.9],
    ]
    p_fast = [
        [0.6, 0.4],
        [0.4, 0.6],
    ]

    fixed_resolved = resolve_fixed_tau(fixed_cfg)
    adaptive_resolved = resolve_adaptive_tau(adaptive_cfg, p_slow)
    matched_resolved = resolve_matched_tau_group(
        matched_cfg["runs"],
        {"match_ref": p_slow, "match_follow": p_fast},
    )

    out_root = root / "results" / "protocol_smoke"
    schema = load_schema(root / "configs" / "result_bundle.schema.json")
    manifests = {
        "fixed_demo": str(_write_bundle("fixed_demo", fixed_resolved, out_root, schema)),
        "adaptive_demo": str(_write_bundle("adaptive_demo", adaptive_resolved, out_root, schema)),
        "match_ref": str(_write_bundle("match_ref", matched_resolved["match_ref"], out_root, schema)),
        "match_follow": str(
            _write_bundle("match_follow", matched_resolved["match_follow"], out_root, schema)
        ),
    }

    payload = {
        "fixed_demo": fixed_resolved,
        "adaptive_demo": adaptive_resolved,
        "matched": matched_resolved,
        "manifests": manifests,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
