import json
from pathlib import Path

from layerbirth.contracts import (
    load_registry,
    load_schema,
    validate_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registry_loads_and_has_contract_smoke():
    registry = load_registry(REPO_ROOT / "configs" / "experiment_registry.yaml")
    entry = registry["experiments"]["contract_smoke"]
    assert entry["status"] == "active"
    assert entry["default_bundle_root"] == "artifacts/dryrun"


def test_registry_loads_upstream_smoke_entry():
    registry = load_registry(REPO_ROOT / "configs" / "experiment_registry.yaml")
    entry = registry["experiments"]["upstream_smoke_baseline"]
    assert entry["status"] == "active"
    assert entry["default_bundle_root"] == "results/upstream_smoke"


def test_schema_loads_and_declares_required_fields():
    schema = load_schema(REPO_ROOT / "configs" / "result_bundle.schema.json")
    assert "experiment_id" in schema["required"]
    assert "metrics_table_path" in schema["required"]


def test_validate_manifest_rejects_missing_required_fields():
    schema = load_schema(REPO_ROOT / "configs" / "result_bundle.schema.json")
    manifest = {
        "schema_version": "1",
        "experiment_id": "contract_smoke",
    }
    try:
        validate_manifest(manifest, schema)
    except ValueError as exc:
        assert "missing required fields" in str(exc).lower()
    else:
        raise AssertionError("validate_manifest should reject incomplete manifests")
