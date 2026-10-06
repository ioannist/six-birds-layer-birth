"""Computation identities for numerical evidence and cache reuse."""

from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


def cache_reuse_allowed(requested: bool) -> bool:
    """Freeze reruns must compute observations rather than replay row caches."""
    return bool(requested) and os.environ.get("LAYERBIRTH_FORCE_RECOMPUTE") != "1"


@lru_cache(maxsize=1)
def implementation_fingerprint() -> str:
    """Fingerprint the installed package sources and numerical runtime.

    This is process-scoped: start a new interpreter after changing source code.
    Hashing package files also covers uncommitted mathematical repairs, unlike
    a commit identifier plus a dirty boolean.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    # Prerequisite bundles use repository configurations as implicit inputs.
    # Include them so changing a canonical panel also invalidates its consumers.
    config_root = Path(__file__).resolve().parents[2] / "configs"
    if config_root.is_dir():
        for path in sorted(p for p in config_root.rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(config_root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    repo_root = config_root.parent
    for directory in (repo_root / "scripts", repo_root / "vendor"):
        for path in sorted(directory.rglob("*.py")):
            if any(part in {".venv", "__pycache__", "site-packages", "build", ".git"} for part in path.parts):
                continue
            digest.update(str(path.relative_to(repo_root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    for path in (repo_root / "pyproject.toml",):
        digest.update(path.read_bytes())
    for package in ("numpy", "matplotlib", "PyYAML"):
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "not-installed"
        digest.update(f"{package}={package_version}".encode())
    digest.update(sys.version.encode())
    digest.update(np.__version__.encode())
    return digest.hexdigest()


def computation_hash(payload: dict[str, Any], length: int = 12) -> str:
    """Hash a complete computation specification under this implementation."""
    if length < 12:
        raise ValueError("computation identities require at least 12 hex digits")
    record = {"implementation_sha256": implementation_fingerprint(), "inputs": payload}
    blob = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def bundle_is_current(root: Path, config: dict[str, Any] | None = None) -> bool:
    """Require the current implementation, complete manifest files and inputs.

    This checks provenance and completeness, not the mathematical conclusions
    in the bundle. Old evidence remains inspectable but cannot satisfy a fresh
    prerequisite merely because its directory exists.
    """
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("code_version", {}).get("implementation_sha256") != implementation_fingerprint():
            return False
        for key in ("config_snapshot_path", "seed_list_path", "metrics_table_path", "notes_file_path", "environment_snapshot_path"):
            if not (root / manifest[key]).is_file():
                return False
        snapshot = json.loads((root / manifest["config_snapshot_path"]).read_text())
        return config is None or snapshot == config
    except (OSError, ValueError, KeyError, TypeError):
        return False


def artifact_is_current(path: Path, config: dict[str, Any] | Path | None = None) -> bool:
    """Check a prerequisite file or root through its enclosing result manifest."""
    if not cache_reuse_allowed(True) or not path.exists():
        return False
    if isinstance(config, Path):
        config = json.loads(config.read_text())
    candidate = path if path.is_dir() else path.parent
    for root in (candidate, *candidate.parents):
        if (root / "manifest.json").is_file():
            return bundle_is_current(root, config)
    return False
