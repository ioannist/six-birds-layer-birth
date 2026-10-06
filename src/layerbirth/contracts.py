"""Contracts for experiment registry and dry-run bundle manifests."""

from __future__ import annotations

import json
from pathlib import Path


REGISTRY_VERSION = "1"
MANIFEST_SCHEMA_VERSION = "1"
REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "experiment_id",
    "bundle_id",
    "created_at",
    "code_version",
    "config_snapshot_path",
    "seed_list_path",
    "metrics_table_path",
    "plots_dir_path",
    "notes_file_path",
    "environment_snapshot_path",
)


def load_json_document(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def load_registry(path: Path) -> dict:
    data = load_json_document(path)
    if data.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("Unsupported registry version")
    experiments = data.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError("Registry must define experiments")
    for experiment_id, entry in experiments.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid entry for {experiment_id}")
        for field in (
            "status",
            "description",
            "default_bundle_root",
            "default_notes_path",
            "default_metric_schema",
            "seed_policy",
        ):
            if field not in entry:
                raise ValueError(f"Missing {field} for {experiment_id}")
    return data


def load_schema(path: Path) -> dict:
    data = load_json_document(path)
    if data.get("type") != "object":
        raise ValueError("Schema type must be object")
    required = data.get("required")
    if not isinstance(required, list):
        raise ValueError("Schema missing required list")
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in required]
    if missing:
        raise ValueError(f"Schema missing required fields: {missing}")
    return data


def validate_manifest(manifest: dict, schema: dict) -> None:
    required = schema.get("required", [])
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema version")
    code_version = manifest.get("code_version")
    if not isinstance(code_version, dict):
        raise ValueError("code_version must be an object")
    if not isinstance(code_version.get("git_commit"), str):
        raise ValueError("code_version.git_commit must be a string")
    if not isinstance(code_version.get("git_dirty"), bool):
        raise ValueError("code_version.git_dirty must be a boolean")
    for field in REQUIRED_MANIFEST_FIELDS:
        if field == "code_version":
            continue
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"{field} must be a non-empty string")
