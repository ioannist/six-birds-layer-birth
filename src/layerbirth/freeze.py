"""F-06 reproducibility freeze bundle builder."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .serialization import scientific_dumps
from .provenance import implementation_fingerprint
from .campaigns import _git_code_version
from .class3 import _repo_root, _write_csv
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest

def _write_json(path: Path, payload: Any) -> None:
    path.write_text(scientific_dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_freeze_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return dict(config_path_or_obj)
    return json.loads(Path(config_path_or_obj).read_text(encoding="utf-8"))


def check_asset_presence(asset_spec: dict[str, Any], root: Path | None = None) -> tuple[bool, list[str], Path]:
    repo_root = _repo_root() if root is None else Path(root)
    bundle_root = repo_root / str(asset_spec["bundle_root"])
    missing: list[str] = []
    for rel in asset_spec.get("required_files", []):
        if not (bundle_root / str(rel)).exists():
            missing.append(str(rel))
    return bundle_root.is_dir() and bool(asset_spec.get("required_files")) and len(missing) == 0, missing, bundle_root


def hash_key_outputs(asset_spec: dict[str, Any], root: Path | None = None) -> dict[str, str | None]:
    repo_root = _repo_root() if root is None else Path(root)
    bundle_root = repo_root / str(asset_spec["bundle_root"])
    hashes: dict[str, str | None] = {}
    for rel in asset_spec.get("key_output_paths", []):
        p = bundle_root / str(rel)
        hashes[str(rel)] = _sha256_file(p) if p.exists() else None
    return hashes


def rerun_asset_entrypoint(asset_spec: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    repo_root = _repo_root() if root is None else Path(root)
    cmd = str(asset_spec.get("rerun_entrypoint", "")).strip()
    if not cmd:
        return {
            "rerun_success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "missing rerun_entrypoint",
        }
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        shell=True,
        capture_output=True,
        text=True,
        env={**os.environ, "LAYERBIRTH_FORCE_RECOMPUTE": "1"},
    )
    return {
        "rerun_success": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def check_scientific_evidence(asset_spec: dict[str, Any], bundle_root: Path) -> dict[str, Any]:
    """Evaluate declared conclusion checks separately from reproducibility.

    Repeating a false conclusion exactly does not support a claim. These finite
    checks index the producing analyses; they do not turn their numerical results
    into formal proofs or certify unspecified claims.
    """
    checks = []
    for check in asset_spec.get("evidence_checks", []):
        try:
            value = json.loads((bundle_root / check["path"]).read_text())
            for part in check["field"].split("."):
                value = value[int(part)] if isinstance(value, list) else value[part]
            passed = type(value) is type(check["equals"]) and value == check["equals"]
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            passed = False
        checks.append({**check, "passed": bool(passed)})
    return {"scientific_evidence_checks_passed": all(check["passed"] for check in checks),
            "scientific_evidence_check_count": len(checks), "scientific_evidence_checks": checks}


def verify_asset_reproducibility(asset_spec: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    repo_root = _repo_root() if root is None else Path(root)
    required_files_present, missing_files, bundle_root = check_asset_presence(asset_spec, repo_root)
    hashes_before = hash_key_outputs(asset_spec, repo_root)

    rerun_result = rerun_asset_entrypoint(asset_spec, repo_root)
    rerun_success = bool(rerun_result["rerun_success"])

    required_files_present_after, missing_after, _ = check_asset_presence(asset_spec, repo_root)
    hashes_after = hash_key_outputs(asset_spec, repo_root)

    stable = bool(
        required_files_present
        and required_files_present_after
        and rerun_success
        and hashes_before == hashes_after
        and all(v is not None for v in hashes_before.values())
    )
    try:
        code_version = json.loads((bundle_root / "manifest.json").read_text())["code_version"]
        fresh_computation = code_version.get("implementation_sha256") == implementation_fingerprint() and code_version.get("cache_reuse_disabled") is True
    except (OSError, ValueError, KeyError, TypeError):
        fresh_computation = False
    reproducible = bool(required_files_present and rerun_success and stable and fresh_computation and hashes_before)

    notes = []
    if not required_files_present:
        notes.append(f"missing_required_files_before={missing_files}")
    if not rerun_success:
        notes.append(f"rerun_failed_rc={rerun_result['returncode']}")
    if not stable and rerun_success:
        notes.append("key_outputs_changed_or_missing")
    if not fresh_computation:
        notes.append("fresh_computation_not_verified; hash stability alone is not recomputation")
    if reproducible:
        notes.append("asset_reproducible")

    return {
        "asset_name": str(asset_spec["asset_name"]),
        "asset_tier": str(asset_spec["asset_tier"]),
        "bundle_root": str(bundle_root),
        "claim_ids": [str(c) for c in asset_spec.get("claim_ids", [])],
        "required_files_present": bool(required_files_present and required_files_present_after),
        "missing_required_files": sorted(set(missing_files + missing_after)),
        "rerun_entrypoint": str(asset_spec.get("rerun_entrypoint", "")),
        "rerun_success": rerun_success,
        "stable_key_outputs": stable,
        "reproducible": reproducible,
        "fresh_computation_verified": fresh_computation,
        "key_output_paths": [str(p) for p in asset_spec.get("key_output_paths", [])],
        "key_output_hashes_before": hashes_before,
        "key_output_hashes_after": hashes_after,
        "notes": "; ".join(notes),
        **check_scientific_evidence(asset_spec, bundle_root),
    }


def _claim_coverage(claims: list[dict[str, Any]], by_asset: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    blocking: list[str] = []
    for c in claims:
        cid = str(c["claim_id"])
        names = [str(x) for x in c.get("asset_names", [])]
        supported = [by_asset[n] for n in names if n in by_asset]
        missing_names = [n for n in names if n not in by_asset]
        all_repro = bool(supported) and not missing_names and all(bool(r.get("reproducible", False)) for r in supported)
        claim_ok = bool(supported) and not missing_names
        if not claim_ok:
            blocking.append(f"{cid}: no valid supporting assets mapped")
        diag = "all supporting assets reproducible" if all_repro else "one or more supporting assets not reproducible"
        if missing_names:
            diag = f"missing assets in freeze config/results: {missing_names}"
        out[cid] = {
            "claim_id": cid,
            "description": str(c.get("description", "")),
            "supporting_asset_names": names,
            "supporting_bundle_roots": [str(r["bundle_root"]) for r in supported],
            "all_supporting_assets_reproducible": bool(all_repro),
            "missing_supporting_asset_names": missing_names,
            "diagnosis": diag,
        }
    return out, blocking


def format_ready_for_writing_status(
    asset_results: list[dict[str, Any]],
    claim_results: dict[str, Any],
    nonblocking_caveats: list[str] | None = None,
) -> dict[str, Any]:
    by_tier = {"critical": [], "supporting": []}
    for r in asset_results:
        by_tier.setdefault(str(r["asset_tier"]), []).append(r)

    blocking_issues: list[str] = []
    for r in by_tier.get("critical", []):
        name = str(r["asset_name"])
        if not bool(r["required_files_present"]):
            blocking_issues.append(f"critical asset missing files: {name}")
        if not bool(r["rerun_success"]):
            blocking_issues.append(f"critical asset rerun failed: {name}")
        if not bool(r["stable_key_outputs"]):
            blocking_issues.append(f"critical asset unstable key outputs: {name}")

    for cid, rec in claim_results.items():
        names = rec.get("supporting_asset_names", [])
        by_name = {r["asset_name"]: r for r in asset_results}
        if not names:
            blocking_issues.append(f"unassigned claim: {cid}")
        elif any(n not in by_name for n in names):
            blocking_issues.append(f"claim refers to missing assets: {cid}")
        elif not all(bool(by_name[n].get("reproducible")) for n in names):
            blocking_issues.append(f"claim supporting evidence not reproducible: {cid}")
        elif not all(by_name[n].get("scientific_evidence_checks_passed", True) for n in names):
            blocking_issues.append(f"claim supporting scientific checks failed: {cid}")

    all_critical_repro = bool(by_tier.get("critical")) and all(bool(r.get("reproducible", False)) for r in by_tier["critical"])
    all_claims_have_assets = bool(claim_results) and all(bool(v.get("supporting_asset_names")) for v in claim_results.values())

    ready = bool(all_critical_repro and all_claims_have_assets and not blocking_issues)
    caveats = list(nonblocking_caveats or [])

    return {
        "ready_for_writing": ready,
        "blocking_issues_present": bool(blocking_issues),
        "nonblocking_caveats_present": bool(caveats),
        "blocking_issues": blocking_issues,
        "nonblocking_caveats": caveats,
        "diagnosis": (
            "all critical assets reproducible and indexed to claims"
            if ready
            else "freeze incomplete: resolve blocking issues before writing"
        ),
    }


def build_evidence_ledger(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = False,
) -> dict[str, Any]:
    _ = use_cache
    cfg = load_freeze_config(config_path_or_obj)
    for entries, field in ((cfg["assets"], "asset_name"), (cfg["claims"], "claim_id")):
        if len({entry[field] for entry in entries}) != len(entries):
            raise ValueError(f"freeze {field} values must be distinct")
    root = _repo_root()

    out_root = (root / "results" / "freeze") if output_root is None else Path(output_root)
    artifact_root = out_root / str(cfg["artifact_subdir"])

    dirs = {
        "config": artifact_root / "config",
        "seeds": artifact_root / "seeds",
        "metrics": artifact_root / "metrics",
        "notes": artifact_root / "notes",
        "env": artifact_root / "env",
        "analysis": artifact_root / "analysis",
        "plots": artifact_root / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    _write_json(dirs["config"] / "config_snapshot.json", cfg)
    _write_json(dirs["seeds"] / "seeds.json", [])
    _write_json(
        dirs["env"] / "environment.json",
        {"generated_at": datetime.now(timezone.utc).isoformat(), "python": "python3"},
    )

    asset_results = [verify_asset_reproducibility(dict(a), root) for a in cfg.get("assets", [])]
    by_asset = {str(r["asset_name"]): r for r in asset_results}

    claim_results, claim_blockers = _claim_coverage([dict(c) for c in cfg.get("claims", [])], by_asset)

    reproducibility_summary = {
        "total_critical_asset_count": sum(1 for r in asset_results if r["asset_tier"] == "critical"),
        "reproducible_critical_asset_count": sum(
            1 for r in asset_results if r["asset_tier"] == "critical" and bool(r["reproducible"])
        ),
        "total_supporting_asset_count": sum(1 for r in asset_results if r["asset_tier"] == "supporting"),
        "reproducible_supporting_asset_count": sum(
            1 for r in asset_results if r["asset_tier"] == "supporting" and bool(r["reproducible"])
        ),
    }
    reproducibility_summary["all_paper_critical_assets_reproducible"] = bool(
        reproducibility_summary["total_critical_asset_count"] > 0
        and reproducibility_summary["reproducible_critical_asset_count"] == reproducibility_summary["total_critical_asset_count"]
    )
    reproducibility_summary["diagnosis"] = (
        "all critical assets are reproducible"
        if reproducibility_summary["all_paper_critical_assets_reproducible"]
        else "one or more critical assets are not reproducible"
    )

    ready = format_ready_for_writing_status(
        asset_results,
        claim_results,
        nonblocking_caveats=[str(x) for x in cfg.get("nonblocking_caveats", [])],
    )
    if claim_blockers:
        ready["blocking_issues_present"] = True
        ready["blocking_issues"] = sorted(set(list(ready["blocking_issues"]) + claim_blockers))
        ready["ready_for_writing"] = False
        ready["diagnosis"] = "freeze incomplete: resolve blocking claim coverage issues"

    _write_json(dirs["analysis"] / "evidence_ledger.json", {"rows": asset_results})
    _write_csv(
        dirs["analysis"] / "evidence_ledger.csv",
        [
            {
                **r,
                "claim_ids": ";".join(r["claim_ids"]),
                "key_output_paths": ";".join(r["key_output_paths"]),
                "key_output_hashes_before": scientific_dumps(r["key_output_hashes_before"], sort_keys=True),
                "key_output_hashes_after": scientific_dumps(r["key_output_hashes_after"], sort_keys=True),
            }
            for r in asset_results
        ],
        [
            "asset_name",
            "asset_tier",
            "bundle_root",
            "claim_ids",
            "required_files_present",
            "rerun_entrypoint",
            "rerun_success",
            "stable_key_outputs",
            "reproducible",
            "key_output_paths",
            "key_output_hashes_before",
            "key_output_hashes_after",
            "notes",
        ],
    )

    _write_json(dirs["analysis"] / "claim_coverage.json", {"claims": claim_results})
    _write_json(dirs["analysis"] / "reproducibility_summary.json", reproducibility_summary)
    _write_json(dirs["analysis"] / "ready_for_writing_status.json", ready)

    _write_csv(
        dirs["metrics"] / "metrics.csv",
        [
            {
                "asset_name": r["asset_name"],
                "asset_tier": r["asset_tier"],
                "bundle_root": r["bundle_root"],
                "required_files_present": r["required_files_present"],
                "rerun_success": r["rerun_success"],
                "stable_key_outputs": r["stable_key_outputs"],
                "reproducible": r["reproducible"],
                "claim_ids": ";".join(r["claim_ids"]),
            }
            for r in asset_results
        ],
        [
            "asset_name",
            "asset_tier",
            "bundle_root",
            "required_files_present",
            "rerun_success",
            "stable_key_outputs",
            "reproducible",
            "claim_ids",
        ],
    )

    (dirs["notes"] / "findings.md").write_text(
        "\n".join(
            [
                "# F-06 reproducibility freeze",
                "",
                f"- all_paper_critical_assets_reproducible: `{reproducibility_summary['all_paper_critical_assets_reproducible']}`",
                f"- ready_for_writing: `{ready['ready_for_writing']}`",
                f"- blocking_issues: `{len(ready['blocking_issues'])}`",
                f"- nonblocking_caveats: `{len(ready['nonblocking_caveats'])}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": str(cfg["freeze_id"]),
        "bundle_id": str(cfg["freeze_id"]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(root),
        "config_snapshot_path": "config/config_snapshot.json",
        "seed_list_path": "seeds/seeds.json",
        "metrics_table_path": "metrics/metrics.csv",
        "plots_dir_path": "plots",
        "notes_file_path": "notes/findings.md",
        "environment_snapshot_path": "env/environment.json",
    }
    schema = load_schema(root / "configs" / "result_bundle.schema.json")
    validate_manifest(manifest, schema)
    _write_json(artifact_root / "manifest.json", manifest)

    should_write_repo_note = bool(cfg.get("write_repo_note", True))
    if should_write_repo_note:
        findings_path = root / str(cfg.get("findings_note_path", "notes/findings/F-06_reproducibility_freeze.md"))
        findings_path.write_text(
            "\n".join(
                [
                    "# F-06 Reproducibility freeze",
                    "",
                    "- Config: `configs/freeze/reproducibility_freeze.json`",
                    f"- Freeze bundle root: `{artifact_root}`",
                    f"- Critical assets checked: `{reproducibility_summary['total_critical_asset_count']}`",
                    f"- Claim count: `{len(claim_results)}`",
                    f"- all_paper_critical_assets_reproducible: `{reproducibility_summary['all_paper_critical_assets_reproducible']}`",
                    f"- ready_for_writing: `{ready['ready_for_writing']}`",
                    "",
                    f"**are all paper-critical assets present, rerunnable, and indexed? {'yes' if reproducibility_summary['all_paper_critical_assets_reproducible'] else 'no'}**",
                    f"**is the repo ready for writing the paper? {'yes' if ready['ready_for_writing'] else 'no'}**",
                    f"- Blocking issues: `{len(ready['blocking_issues'])}`; Nonblocking caveats: `{len(ready['nonblocking_caveats'])}`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    return {
        "artifact_root": str(artifact_root),
        "all_paper_critical_assets_reproducible": bool(reproducibility_summary["all_paper_critical_assets_reproducible"]),
        "ready_for_writing": bool(ready["ready_for_writing"]),
        "blocking_issue_count": len(ready["blocking_issues"]),
        "nonblocking_caveat_count": len(ready["nonblocking_caveats"]),
    }


__all__ = [
    "load_freeze_config",
    "check_asset_presence",
    "hash_key_outputs",
    "rerun_asset_entrypoint",
    "verify_asset_reproducibility",
    "build_evidence_ledger",
    "format_ready_for_writing_status",
]
