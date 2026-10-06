"""Deterministic sweep runner for layer-birth tiny-grid experiments."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .serialization import scientific_dumps
from .provenance import cache_reuse_allowed, computation_hash, implementation_fingerprint
from .contracts import MANIFEST_SCHEMA_VERSION, load_schema, validate_manifest
from .lenses import build_lens_family
from .lifts import build_lift_family
from .metrics import default_metric_bundle
from .numeric import packaging_projector, pushforward_matrix, row_normalize, stationary_distribution, validate_row_stochastic
from .protocols import resolve_run_settings
from .substrates import build_substrate_family


AGGREGATE_FIELDS = [
    "sweep_id",
    "run_id",
    "family_name",
    "size",
    "seed",
    "closure_strength_lambda",
    "control_application_name",
    "lens_name",
    "lift_name",
    "tau_protocol_name",
    "resolved_tau",
    "analysis_k",
    "closure_error",
    "objecthood_order",
    "staging_gap",
    "affinity",
    "holonomy",
    "cache_status",
    "manifest_path",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    return computation_hash(payload, length)


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


def _coerce_config(config_path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path_or_obj, dict):
        return config_path_or_obj
    path = Path(config_path_or_obj)
    return json.loads(path.read_text(encoding="utf-8"))


def _build_family_params(family_spec: dict[str, Any], size: int, seed: int | None) -> dict[str, Any]:
    params = dict(family_spec.get("family_params", {}))
    derive_block = bool(family_spec.get("derive_block_size_from_size", False))
    if derive_block:
        n_blocks = int(params.get("n_blocks", 0))
        if n_blocks <= 0 or size % n_blocks != 0:
            raise ValueError("size must be divisible by n_blocks when derive_block_size_from_size=true")
        params["block_size"] = int(size // n_blocks)
    if "size_param_name" in family_spec:
        params[str(family_spec["size_param_name"])] = int(size)
    elif not derive_block:
        params.setdefault("n", int(size))
    if seed is not None:
        params["seed"] = int(seed)
    return params


def expand_sweep_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    sweep_id = str(config["sweep_id"])
    families = list(config.get("family_specs", []))
    lambda_grid = list(config.get("closure_strength_lambda_grid", []))
    lens_grid = list(config.get("lens_grid", []))
    lift_grid = list(config.get("lift_grid", []))
    tau_grid = list(config.get("tau_protocol_grid", []))
    control_application_name = str(config.get("control_application_name", "passthrough"))
    if not families:
        raise ValueError("family_specs cannot be empty")
    if not lambda_grid or not lens_grid or not lift_grid or not tau_grid:
        raise ValueError("lambda/lens/lift/tau grids cannot be empty")

    runs: list[dict[str, Any]] = []
    for family_spec in families:
        family_name = str(family_spec["family_name"])
        sizes = [int(v) for v in family_spec.get("size_grid", [])]
        if not sizes:
            raise ValueError("each family spec requires a non-empty size_grid")
        seed_grid = family_spec.get("seed_grid", [None])
        for size in sizes:
            for seed in seed_grid:
                for lam in lambda_grid:
                    for lens_spec in lens_grid:
                        for lift_spec in lift_grid:
                            for tau_spec in tau_grid:
                                spec_core = {
                                    "sweep_id": sweep_id,
                                    "family_name": family_name,
                                    "size": int(size),
                                    "seed": None if seed is None else int(seed),
                                    "closure_strength_lambda": float(lam),
                                    "control_application_name": control_application_name,
                                    "lens_spec": lens_spec,
                                    "lift_spec": lift_spec,
                                    "tau_protocol": tau_spec,
                                    "family_spec": family_spec,
                                }
                                run_id = f"run_{_stable_hash(spec_core)}"
                                runs.append({**spec_core, "run_id": run_id})
    runs.sort(key=lambda r: r["run_id"])
    return runs


def apply_closure_strength_control(
    P: np.ndarray,
    closure_strength_lambda: float,
    Q_f: np.ndarray | None = None,
    U_f: np.ndarray | None = None,
    U_route: np.ndarray | None = None,
    mode: str = "passthrough",
) -> np.ndarray:
    def _stationary_distribution_power(M: np.ndarray, max_iter: int = 20000, tol: float = 1e-14) -> np.ndarray:
        return stationary_distribution(M, max_iter=max_iter, tol=tol)[0]

    def _mh_reversibleize(proposal: np.ndarray, pi: np.ndarray) -> np.ndarray:
        n = proposal.shape[0]
        out = np.zeros_like(proposal)
        for i in range(n):
            row_offdiag = 0.0
            for j in range(n):
                if i == j:
                    continue
                qij = float(proposal[i, j])
                if qij <= 0.0:
                    continue
                qji = float(proposal[j, i])
                den = float(pi[i]) * qij
                if den <= 0.0:
                    a = 0.0
                else:
                    num = float(pi[j]) * qji
                    a = min(1.0, num / den)
                val = qij * a
                out[i, j] = val
                row_offdiag += val
            out[i, i] = max(0.0, 1.0 - row_offdiag)
        out = row_normalize(out)
        validate_row_stochastic(out)
        return out

    lam = float(closure_strength_lambda)
    if not np.isfinite(lam) or lam < 0.0 or lam > 1.0:
        raise ValueError("closure_strength_lambda must be in [0,1]")
    p = np.asarray(P, dtype=np.float64)
    validate_row_stochastic(p)
    if mode == "passthrough":
        return p.copy()
    if mode == "mix_with_packaging_projector":
        if Q_f is None or U_f is None:
            raise ValueError("Q_f and U_f are required for mix_with_packaging_projector")
        proj = packaging_projector(Q_f, U_f)
        mixed = (1.0 - lam) * p + lam * proj
        mixed = row_normalize(mixed)
        validate_row_stochastic(mixed)
        return mixed
    if mode == "mix_with_routed_packaging_projector":
        if Q_f is None or U_route is None:
            raise ValueError("Q_f and U_route are required for mix_with_routed_packaging_projector")
        proj = packaging_projector(Q_f, U_route)
        mixed = (1.0 - lam) * p + lam * proj
        mixed = row_normalize(mixed)
        validate_row_stochastic(mixed)
        return mixed
    if mode == "mix_with_pi_reversible_direct_projector":
        if Q_f is None or U_f is None:
            raise ValueError("Q_f and U_f are required for mix_with_pi_reversible_direct_projector")
        pi = _stationary_distribution_power(p)
        proposal = row_normalize(np.asarray(Q_f, dtype=np.float64) @ np.asarray(U_f, dtype=np.float64))
        kernel_rev = _mh_reversibleize(proposal, pi)
        mixed = (1.0 - lam) * p + lam * kernel_rev
        mixed = row_normalize(mixed)
        validate_row_stochastic(mixed)
        return mixed
    if mode == "mix_with_pi_reversible_routed_projector":
        if Q_f is None or U_route is None:
            raise ValueError("Q_f and U_route are required for mix_with_pi_reversible_routed_projector")
        pi = _stationary_distribution_power(p)
        proposal = row_normalize(np.asarray(Q_f, dtype=np.float64) @ np.asarray(U_route, dtype=np.float64))
        kernel_rev = _mh_reversibleize(proposal, pi)
        mixed = (1.0 - lam) * p + lam * kernel_rev
        mixed = row_normalize(mixed)
        validate_row_stochastic(mixed)
        return mixed
    raise ValueError(f"unknown control mode: {mode}")


def resolve_run_artifacts(run_spec: dict[str, Any], output_root: Path) -> dict[str, Path]:
    sweep_root = output_root / str(run_spec["sweep_id"])
    run_root = sweep_root / "runs" / str(run_spec["run_id"])
    return {
        "sweep_root": sweep_root,
        "run_root": run_root,
        "run_manifest": run_root / "manifest.json",
        "run_metrics_csv": run_root / "metrics" / "metrics.csv",
    }


def _ensure_bundle_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "config": root / "config",
        "seeds": root / "seeds",
        "metrics": root / "metrics",
        "plots": root / "plots",
        "notes": root / "notes",
        "env": root / "env",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_manifest(
    *,
    root: Path,
    experiment_id: str,
    bundle_id: str,
    config_snapshot_path: Path,
    seed_list_path: Path,
    metrics_path: Path,
    notes_path: Path,
    env_path: Path,
    schema: dict[str, Any],
) -> Path:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "bundle_id": bundle_id,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version": _git_code_version(_repo_root()),
        "config_snapshot_path": str(config_snapshot_path.relative_to(root)),
        "seed_list_path": str(seed_list_path.relative_to(root)),
        "metrics_table_path": str(metrics_path.relative_to(root)),
        "plots_dir_path": "plots",
        "notes_file_path": str(notes_path.relative_to(root)),
        "environment_snapshot_path": str(env_path.relative_to(root)),
    }
    validate_manifest(manifest, schema)
    out = root / "manifest.json"
    out.write_text(scientific_dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def _load_cached_row(run_metrics_csv: Path) -> dict[str, Any]:
    with run_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError("cached run metrics table must contain exactly one row")
    return dict(rows[0])


def _resolve_lens(substrate: dict[str, Any], lens_spec: dict[str, Any]) -> tuple[np.ndarray, str]:
    name = str(lens_spec["name"])
    if name == "manual_family_block_lens":
        lens = substrate.get("block_lens")
        if lens is None:
            raise ValueError("manual_family_block_lens requires substrate block_lens")
        return np.asarray(lens, dtype=np.int64), name
    if name == "manual_partition_lens":
        labels = lens_spec.get("labels")
        if labels is None:
            raise ValueError("manual_partition_lens requires labels")
        lens, _ = build_lens_family("manual_partition_lens", labels=labels)
        return np.asarray(lens, dtype=np.int64), name
    kwargs = dict(lens_spec.get("kwargs", {}))
    lens, _ = build_lens_family(name, P=np.asarray(substrate["P"]), **kwargs)
    return np.asarray(lens, dtype=np.int64), name


def _resolve_lift(
    lift_spec: dict[str, Any],
    P: np.ndarray,
    lens: np.ndarray,
) -> tuple[np.ndarray, str]:
    name = str(lift_spec["name"])
    k = int(np.max(lens)) + 1
    kwargs = dict(lift_spec.get("kwargs", {}))
    if name == "uniform_lift_family" or name == "prototype_lift_family":
        u, _ = build_lift_family(name, f=lens, k=k, **kwargs)
        return u, name
    if name == "stationary_within_fiber_lift":
        u, _ = build_lift_family(name, P=P, f=lens, k=k, **kwargs)
        return u, name
    raise ValueError(f"unknown lift family: {name}")


def execute_sweep_run(
    run_spec: dict[str, Any],
    output_root: Path,
    use_cache: bool = True,
) -> dict[str, Any]:
    artifacts = resolve_run_artifacts(run_spec, output_root)
    run_root = artifacts["run_root"]
    run_manifest = artifacts["run_manifest"]
    run_metrics_csv = artifacts["run_metrics_csv"]

    if cache_reuse_allowed(use_cache) and run_manifest.exists() and run_metrics_csv.exists():
        try:
            manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
            snapshot = run_root / manifest["config_snapshot_path"]
            current_implementation = manifest.get("code_version", {}).get("implementation_sha256") == implementation_fingerprint()
            current_inputs = snapshot.exists() and json.loads(snapshot.read_text(encoding="utf-8")).get("run_spec") == run_spec
        except (OSError, ValueError, KeyError, TypeError):
            current_implementation = current_inputs = False
        if current_implementation and current_inputs:
            row = _load_cached_row(run_metrics_csv)
            row["cache_status"] = "cached"
            return row

    family_spec = dict(run_spec["family_spec"])
    size = int(run_spec["size"])
    seed = run_spec["seed"]
    family_name = str(run_spec["family_name"])
    family_params = _build_family_params(family_spec, size, seed)
    substrate = build_substrate_family(family_name, **family_params)
    p_base = np.asarray(substrate["P"], dtype=np.float64)

    lens, lens_name = _resolve_lens(substrate, dict(run_spec["lens_spec"]))
    q = pushforward_matrix(lens, int(np.max(lens)) + 1)
    lift, lift_name = _resolve_lift(dict(run_spec["lift_spec"]), p_base, lens)

    tau_config = {
        "run_id": run_spec["run_id"],
        "tau_protocol": dict(run_spec["tau_protocol"]),
        "control": {"closure_strength_lambda": float(run_spec["closure_strength_lambda"])},
    }
    resolved = resolve_run_settings(tau_config, P=p_base)
    tau = int(resolved["resolved_tau"])

    p_controlled = apply_closure_strength_control(
        p_base,
        closure_strength_lambda=float(run_spec["closure_strength_lambda"]),
        Q_f=q,
        U_f=lift,
        mode=str(run_spec["control_application_name"]),
    )
    bundle = default_metric_bundle(p_controlled, lens, tau=tau, lift_name=lift_name, U_f=lift)

    dirs = _ensure_bundle_dirs(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    cfg_path = dirs["config"] / "config_snapshot.json"
    seeds_path = dirs["seeds"] / "seeds.json"
    metrics_path = dirs["metrics"] / "metrics.csv"
    notes_path = dirs["notes"] / "findings.md"
    env_path = dirs["env"] / "environment.json"

    cfg_path.write_text(
        scientific_dumps(
            {
                "run_spec": run_spec,
                "resolved_tau": resolved,
                "family_params": family_params,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    seeds_payload = [] if seed is None else [int(seed)]
    seeds_path.write_text(scientific_dumps(seeds_payload, indent=2) + "\n", encoding="utf-8")

    row = {
        "sweep_id": str(run_spec["sweep_id"]),
        "run_id": str(run_spec["run_id"]),
        "family_name": family_name,
        "size": str(size),
        "seed": "" if seed is None else str(seed),
        "closure_strength_lambda": str(float(run_spec["closure_strength_lambda"])),
        "control_application_name": str(run_spec["control_application_name"]),
        "lens_name": lens_name,
        "lift_name": lift_name,
        "tau_protocol_name": str(resolved["tau_protocol_name"]),
        "resolved_tau": str(tau),
        "analysis_k": str(bundle["analysis_k"]),
        "closure_error": str(bundle["closure_error"]),
        "objecthood_order": str(bundle["objecthood_order"]),
        "staging_gap": str(bundle["staging_gap"]),
        "affinity": str(bundle["affinity"]),
        "holonomy": "" if bundle["holonomy"] is None else str(bundle["holonomy"]),
        "cache_status": "executed",
        "manifest_path": "",
    }
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    notes_path.write_text(
        "\n".join(
            [
                "# Sweep run",
                "",
                f"- run_id: `{run_spec['run_id']}`",
                f"- family_name: `{family_name}`",
                f"- closure_strength_lambda: `{run_spec['closure_strength_lambda']}`",
                f"- control_application_name: `{run_spec['control_application_name']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_path.write_text(
        scientific_dumps(
            {"python": "unknown", "generated_at": datetime.now(timezone.utc).isoformat()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    schema = load_schema(_repo_root() / "configs" / "result_bundle.schema.json")
    manifest_path = _write_manifest(
        root=run_root,
        experiment_id="tiny_grid_sweep_run",
        bundle_id=str(run_spec["run_id"]),
        config_snapshot_path=cfg_path,
        seed_list_path=seeds_path,
        metrics_path=metrics_path,
        notes_path=notes_path,
        env_path=env_path,
        schema=schema,
    )
    row["manifest_path"] = str(manifest_path)
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return row


def run_sweep(
    config_path_or_obj: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    config = _coerce_config(config_path_or_obj)
    sweep_id = str(config["sweep_id"])
    if output_root is None:
        output_root = _repo_root() / "results" / "sweeps"
    else:
        output_root = Path(output_root)
    run_specs = expand_sweep_grid(config)
    sweep_root = output_root / sweep_id
    runs_root = sweep_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    rows = [execute_sweep_run(spec, output_root, use_cache=use_cache) for spec in run_specs]
    rows.sort(key=lambda r: r["run_id"])
    executed_count = sum(1 for r in rows if r["cache_status"] == "executed")
    cached_count = sum(1 for r in rows if r["cache_status"] == "cached")

    dirs = _ensure_bundle_dirs(sweep_root)
    cfg_path = dirs["config"] / "config_snapshot.json"
    seeds_path = dirs["seeds"] / "seeds.json"
    metrics_path = dirs["metrics"] / "metrics.csv"
    notes_path = dirs["notes"] / "findings.md"
    env_path = dirs["env"] / "environment.json"

    cfg_path.write_text(scientific_dumps(config, indent=2) + "\n", encoding="utf-8")
    seed_values = sorted(
        {
            int(spec["seed"])
            for spec in run_specs
            if spec.get("seed") is not None and spec.get("seed") != ""
        }
    )
    seeds_path.write_text(scientific_dumps(seed_values, indent=2) + "\n", encoding="utf-8")
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in AGGREGATE_FIELDS})
    notes_path.write_text(
        "\n".join(
            [
                "# Sweep summary",
                "",
                f"- sweep_id: `{sweep_id}`",
                f"- total_runs: `{len(rows)}`",
                f"- executed: `{executed_count}`",
                f"- cached: `{cached_count}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_path.write_text(
        scientific_dumps(
            {"python": "unknown", "generated_at": datetime.now(timezone.utc).isoformat()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    schema = load_schema(_repo_root() / "configs" / "result_bundle.schema.json")
    manifest_path = _write_manifest(
        root=sweep_root,
        experiment_id="tiny_grid_sweep",
        bundle_id=sweep_id,
        config_snapshot_path=cfg_path,
        seed_list_path=seeds_path,
        metrics_path=metrics_path,
        notes_path=notes_path,
        env_path=env_path,
        schema=schema,
    )
    return {
        "sweep_id": sweep_id,
        "total_runs": len(rows),
        "executed_count": executed_count,
        "cached_count": cached_count,
        "metrics_csv_path": str(metrics_path),
        "manifest_path": str(manifest_path),
    }
