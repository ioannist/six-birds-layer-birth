"""Dry-run bundle manifest utility."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

from .serialization import scientific_dumps
from .provenance import cache_reuse_allowed, implementation_fingerprint
from .contracts import (
    MANIFEST_SCHEMA_VERSION,
    load_registry,
    load_schema,
    validate_manifest,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
        return {"git_commit": commit or "unknown", "git_dirty": bool(dirty), "implementation_sha256": implementation_fingerprint(), "cache_reuse_disabled": not cache_reuse_allowed(True)}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": False, "implementation_sha256": implementation_fingerprint(), "cache_reuse_disabled": not cache_reuse_allowed(True)}


def create_dry_run_bundle(experiment_id: str, output_root: Path) -> Path:
    root = _repo_root()
    registry = load_registry(root / "configs" / "experiment_registry.yaml")
    schema = load_schema(root / "configs" / "result_bundle.schema.json")
    experiments = registry["experiments"]
    if experiment_id not in experiments:
        raise ValueError(f"Unknown experiment_id: {experiment_id}")

    bundle_id = f"{experiment_id}_dryrun"
    bundle_root = output_root / bundle_id
    if bundle_root.exists():
        shutil.rmtree(bundle_root)

    config_dir = bundle_root / "config"
    seeds_dir = bundle_root / "seeds"
    metrics_dir = bundle_root / "metrics"
    plots_dir = bundle_root / "plots"
    notes_dir = bundle_root / "notes"
    env_dir = bundle_root / "env"
    for path in (config_dir, seeds_dir, metrics_dir, plots_dir, notes_dir, env_dir):
        path.mkdir(parents=True, exist_ok=True)

    config_snapshot = config_dir / "config_snapshot.json"
    seed_list = seeds_dir / "seeds.json"
    metrics_table = metrics_dir / "metrics.csv"
    notes_file = notes_dir / "findings.md"
    environment_snapshot = env_dir / "environment.json"

    experiment_entry = experiments[experiment_id]
    config_snapshot.write_text(
        scientific_dumps(
            {
                "experiment_id": experiment_id,
                "registry_entry": experiment_entry,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    seed_source = root / experiment_entry["seed_policy"]["seed_file"]
    seed_values = json.loads(seed_source.read_text(encoding="utf-8"))
    seed_list.write_text(scientific_dumps(seed_values, indent=2) + "\n", encoding="utf-8")
    with metrics_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
    notes_file.write_text("# Dry-run findings\n", encoding="utf-8")
    environment_snapshot.write_text(
        scientific_dumps(
            {
                "python": subprocess.run(
                    [sys.executable, "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                or subprocess.run(
                    [sys.executable, "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stderr.strip(),
                "platform": "unknown",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "bundle_id": bundle_id,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": str(config_snapshot.relative_to(bundle_root)),
        "seed_list_path": str(seed_list.relative_to(bundle_root)),
        "metrics_table_path": str(metrics_table.relative_to(bundle_root)),
        "plots_dir_path": str(plots_dir.relative_to(bundle_root)),
        "notes_file_path": str(notes_file.relative_to(bundle_root)),
        "environment_snapshot_path": str(environment_snapshot.relative_to(bundle_root)),
    }
    validate_manifest(manifest, schema)

    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m layerbirth.manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--experiment-id", required=True)
    dry_run.add_argument("--output-root", required=True)

    args = parser.parse_args(argv)
    if args.command == "dry-run":
        output_root = Path(args.output_root)
        manifest_path = create_dry_run_bundle(args.experiment_id, output_root)
        print(manifest_path)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
