import json
import subprocess
import sys
from pathlib import Path

from layerbirth.contracts import load_schema, validate_manifest


def test_dry_run_command_creates_bundle(tmp_path):
    output_root = tmp_path / "dryrun"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "layerbirth.manifest",
            "dry-run",
            "--experiment-id",
            "contract_smoke",
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    manifest_path = Path(result.stdout.strip())
    assert manifest_path.exists()

    bundle_root = manifest_path.parent
    expected_paths = [
        bundle_root / "config" / "config_snapshot.json",
        bundle_root / "seeds" / "seeds.json",
        bundle_root / "metrics" / "metrics.csv",
        bundle_root / "plots",
        bundle_root / "notes" / "findings.md",
        bundle_root / "env" / "environment.json",
    ]
    for path in expected_paths:
        assert path.exists()

    schema = load_schema(Path("configs/result_bundle.schema.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, schema)
    assert manifest["bundle_id"] == "contract_smoke_dryrun"
