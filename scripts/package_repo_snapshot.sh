#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$repo_root" python3 - <<'PY'
from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import json
import os
import pathlib
import zipfile

root = pathlib.Path(os.environ["REPO_ROOT"]).resolve()
repo_name = root.name
version_file = root / ".package-repo-snapshot-version"
config_path = root / ".package-repo-snapshot.json"
if not config_path.exists():
    vendor_config_path = root / "vendor" / "six-birds-pica" / ".package-repo-snapshot.json"
    if vendor_config_path.exists():
        config_path = vendor_config_path

raw_config: dict[str, object] = {}
config_base = root
if config_path.exists():
    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise SystemExit("Invalid .package-repo-snapshot.json: expected a JSON object.")
    raw_config = parsed
    config_base = config_path.parent

excluded_dirs = set(
    raw_config.get(
        "exclude_dirs",
        [
            ".git",
            ".lake",
            ".venv",
            "venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "target",
            "dist",
            "build",
            "artifacts",
            "results",
            ".egg-info",
        ],
    )
)
excluded_file_patterns = set(
    raw_config.get(
        "exclude_file_patterns",
        [
            "*.pyc",
            "*.pyo",
            "*.so",
            "*.d",
            "*.rlib",
            "*.rmeta",
            "*.a",
            "*.o",
            "*.log",
            "*.json",
            "*.jsonl",
            "*.tmp",
            "*.out",
            "*.dll",
            "*.exe",
            "*.class",
            "*.whl",
            "*.zip",
            "*.tar",
            "*.tar.gz",
            "*.tgz",
            "*.7z",
            "*.bz2",
            "*.xz",
            "*.pdf",
            "*.gz",
            "*.csv",
            "*.tex",
            "*.fls",
            "*.aux",
            "*.fdb_latexmk",
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
        ],
    )
)

allowed_roots: dict[str, dict[str, list[str]]] = {}
allowed_root_files: set[str] = set()
allowlist_scope: pathlib.Path | None = None
if isinstance(raw_config.get("allowed_roots"), dict):
    allowlist_scope = config_base
    for name, spec in raw_config["allowed_roots"].items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            continue
        exts = spec.get("extensions")
        names = spec.get("filenames")
        if not isinstance(exts, list) or not all(isinstance(item, str) for item in exts):
            exts = []
        if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
            names = []
        try:
            rel_root = (config_base / name).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        allowed_roots[rel_root] = {
            "extensions": exts,
            "filenames": names,
        }
if isinstance(raw_config.get("allowed_root_files"), list):
    for item in raw_config["allowed_root_files"]:
        if not isinstance(item, str):
            continue
        try:
            rel_file = (config_base / item).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        allowed_root_files.add(rel_file)

current_version = -1
if version_file.exists():
    raw = version_file.read_text(encoding="utf-8").strip()
    if raw.isdigit():
        current_version = int(raw)

next_version = current_version + 1
zip_path = root / f"{repo_name}_snapshot_v{next_version}.zip"


@dataclass(frozen=True)
class IgnoreRule:
    base: pathlib.Path
    pattern: str
    negated: bool
    anchored: bool
    dir_only: bool


def _parse_gitignore(path: pathlib.Path) -> list[IgnoreRule]:
    rules: list[IgnoreRule] = []
    if not path.exists():
        return rules
    base = path.parent
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(r"\#") or line.startswith(r"\!"):
            line = line[1:]
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        anchored = line.startswith("/")
        if anchored:
            line = line[1:]
        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1]
        if not line:
            continue
        rules.append(
            IgnoreRule(
                base=base,
                pattern=line,
                negated=negated,
                anchored=anchored,
                dir_only=dir_only,
            )
        )
    return rules


def _match_rule(rule: IgnoreRule, rel_posix: str, is_dir: bool) -> bool:
    if rule.dir_only and not is_dir:
        parts = rel_posix.split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            if _match_rule(
                IgnoreRule(
                    base=rule.base,
                    pattern=rule.pattern,
                    negated=rule.negated,
                    anchored=rule.anchored,
                    dir_only=False,
                ),
                prefix,
                is_dir=True,
            ):
                return True
        return False
    if rule.anchored:
        return fnmatch.fnmatchcase(rel_posix, rule.pattern)
    if "/" in rule.pattern:
        return fnmatch.fnmatchcase(rel_posix, rule.pattern)
    return fnmatch.fnmatchcase(pathlib.PurePosixPath(rel_posix).name, rule.pattern)


def _collect_rules(root_path: pathlib.Path) -> list[IgnoreRule]:
    rules: list[IgnoreRule] = []
    rules.append(
        IgnoreRule(
            base=root_path,
            pattern=".git",
            negated=False,
            anchored=False,
            dir_only=True,
        )
    )
    rules.append(
        IgnoreRule(
            base=root_path,
            pattern="*_snapshot.zip",
            negated=False,
            anchored=False,
            dir_only=False,
        )
    )
    rules.append(
        IgnoreRule(
            base=root_path,
            pattern="*_snapshot_v*.zip",
            negated=False,
            anchored=False,
            dir_only=False,
        )
    )
    for dirpath, dirnames, filenames in os.walk(root_path):
        if ".git" in dirnames:
            dirnames.remove(".git")
        ignore_path = pathlib.Path(dirpath) / ".gitignore"
        if ignore_path.exists():
            rules.extend(_parse_gitignore(ignore_path))
    return rules


def _is_ignored(path: pathlib.Path, rules: list[IgnoreRule]) -> bool:
    ignored = False
    for rule in rules:
        try:
            rel_to_rule = path.relative_to(rule.base).as_posix()
        except ValueError:
            continue
        if _match_rule(rule, rel_to_rule, path.is_dir()):
            ignored = not rule.negated
    return ignored


def _matches_allowed_root_rule(path: pathlib.Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return True
    if allowlist_scope is None:
        return True
    try:
        path.relative_to(allowlist_scope)
    except ValueError:
        return True
    if not allowed_roots and not allowed_root_files:
        return True

    matched_root = None
    for key in allowed_roots:
        if rel == key or rel.startswith(f"{key}/"):
            matched_root = key
            break
    if not matched_root:
        return path.parent == root and (rel in allowed_root_files)

    rule = allowed_roots[matched_root]
    filename = path.name
    if filename in rule.get("filenames", []):
        return True

    extensions = rule.get("extensions", [])
    if "*" in extensions:
        return True
    suffix = path.suffix
    return suffix in extensions


def _is_allowed_root_file(path: pathlib.Path) -> bool:
    name = path.name
    if name.startswith(".package-repo-snapshot"):
        return True
    if not allowed_root_files:
        return True
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return False
    return rel in allowed_root_files


def _within_allowlist_scope(path: pathlib.Path) -> bool:
    if allowlist_scope is None:
        return False
    try:
        path.relative_to(allowlist_scope)
        return True
    except ValueError:
        return False


def _is_ignored_file(path: pathlib.Path, patterns: set[str]) -> bool:
    if allowlist_scope is not None and not _within_allowlist_scope(path):
        return False
    name = path.name
    rel = path.relative_to(root).as_posix()
    for pattern in patterns:
        if fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(rel, pattern):
            return True
    return False


rules = _collect_rules(root)
files: list[pathlib.Path] = []
for dirpath, dirnames, filenames in os.walk(root):
    parent = pathlib.Path(dirpath)
    if allowlist_scope is not None and not _within_allowlist_scope(parent):
        dirnames[:] = [d for d in dirnames if d != ".git"]
    else:
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
    for filename in filenames:
        path = pathlib.Path(dirpath) / filename
        if not path.exists():
            continue
        allowlisted = _matches_allowed_root_rule(path)
        if not allowlisted:
            if path.parent == root:
                if not _is_allowed_root_file(path):
                    continue
            else:
                continue
        if not allowlisted and _is_ignored_file(path, excluded_file_patterns):
            continue
        if _is_ignored(path, rules):
            # When an explicit allowlist is configured, treat it as authoritative
            # for files under the allowlist scope (e.g., results bundles).
            if allowlist_scope is None or not _within_allowlist_scope(path):
                continue
        files.append(path)

if not files:
    raise SystemExit("No files to package (all files ignored).")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in files:
        rel = path.relative_to(root).as_posix()
        zf.write(path, rel)

version_file.write_text(str(next_version), encoding="utf-8")
previous_path = None
if current_version >= 0:
    previous_path = root / f"{repo_name}_snapshot_v{current_version}.zip"
if previous_path and previous_path.exists():
    previous_path.unlink()

print(f"Wrote {zip_path}")
PY
