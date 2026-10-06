import json
from pathlib import Path

import numpy as np

from layerbirth.crossover import build_direct_and_routed_projectors, run_holonomy_crossover_campaign_v2
from layerbirth.numeric import macro_kernel, pushforward_matrix, validate_row_stochastic
from layerbirth.sweep import apply_closure_strength_control
from layerbirth.substrates import build_substrate_family


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "campaigns" / "holonomy_crossover_v2.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _stationary(P: np.ndarray, steps: int = 50000, tol: float = 1e-14) -> np.ndarray:
    n = P.shape[0]
    pi = np.full(n, 1.0 / float(n), dtype=np.float64)
    for _ in range(steps):
        nxt = pi @ P
        if np.linalg.norm(nxt - pi, ord=1) < tol:
            pi = nxt
            break
        pi = nxt
    return pi / max(float(np.sum(pi)), 1e-15)


def _db_residual(P: np.ndarray, pi: np.ndarray) -> float:
    lhs = pi[:, None] * P
    rhs = lhs.T
    return float(np.max(np.abs(lhs - rhs)))


def test_holonomy_crossover_v2_config_loads():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "holonomy_crossover_v2"
    assert cfg["tau_protocol"]["tau"] == 2
    assert len(cfg["cases"]) == 2
    assert len(cfg["control_routes"]) == 2


def test_pi_reversible_projector_plumbing():
    cfg = _load_cfg()
    bk = cfg["base_kernel"]
    bal = build_substrate_family(
        "holonomy_interface_skew_family",
        fine_block_sizes_by_coarse=[[2, 2], [2, 2]],
        n_blocks=bk["n_blocks"],
        block_size=bk["block_size"],
        intra_block_weight=bk["intra_block_weight"],
        interface_intra_weight=bk["interface_intra_weight"],
        interface_cross_weight=bk["interface_cross_weight"],
        noninterface_cross_weight=bk["noninterface_cross_weight"],
        self_weight=bk["self_weight"],
        interface_position_in_block=bk["interface_position_in_block"],
    )
    unb = build_substrate_family(
        "holonomy_interface_skew_family",
        fine_block_sizes_by_coarse=[[1, 3], [1, 3]],
        n_blocks=bk["n_blocks"],
        block_size=bk["block_size"],
        intra_block_weight=bk["intra_block_weight"],
        interface_intra_weight=bk["interface_intra_weight"],
        interface_cross_weight=bk["interface_cross_weight"],
        noninterface_cross_weight=bk["noninterface_cross_weight"],
        self_weight=bk["self_weight"],
        interface_position_in_block=bk["interface_position_in_block"],
    )
    pb = build_direct_and_routed_projectors(bal["coarse_lens"], bal["fine_lens"])
    pu = build_direct_and_routed_projectors(unb["coarse_lens"], unb["fine_lens"])

    assert np.allclose(pb["Pi_direct"], pb["Pi_route"], atol=1e-12, rtol=0.0)
    assert not np.allclose(pu["Pi_direct"], pu["Pi_route"], atol=1e-12, rtol=0.0)

    p_bd = apply_closure_strength_control(
        bal["P"],
        closure_strength_lambda=1.0,
        Q_f=pb["Q_c"],
        U_f=pb["U_c"],
        mode="mix_with_pi_reversible_direct_projector",
    )
    p_br = apply_closure_strength_control(
        bal["P"],
        closure_strength_lambda=1.0,
        Q_f=pb["Q_c"],
        U_f=pb["U_c"],
        U_route=pb["U_route"],
        mode="mix_with_pi_reversible_routed_projector",
    )
    p_ud = apply_closure_strength_control(
        unb["P"],
        closure_strength_lambda=1.0,
        Q_f=pu["Q_c"],
        U_f=pu["U_c"],
        mode="mix_with_pi_reversible_direct_projector",
    )
    p_ur = apply_closure_strength_control(
        unb["P"],
        closure_strength_lambda=1.0,
        Q_f=pu["Q_c"],
        U_f=pu["U_c"],
        U_route=pu["U_route"],
        mode="mix_with_pi_reversible_routed_projector",
    )

    for p in [p_bd, p_br, p_ud, p_ur]:
        validate_row_stochastic(p)
    pi_b = _stationary(np.asarray(bal["P"], dtype=np.float64))
    pi_u = _stationary(np.asarray(unb["P"], dtype=np.float64))
    assert _db_residual(p_bd, pi_b) < 1e-8
    assert _db_residual(p_br, pi_b) < 1e-8
    assert _db_residual(p_ud, pi_u) < 1e-8
    assert _db_residual(p_ur, pi_u) < 1e-8


def test_interface_skew_mechanism_macro_offdiag_tau2():
    cfg = _load_cfg()
    bk = cfg["base_kernel"]
    bal = build_substrate_family(
        "holonomy_interface_skew_family",
        fine_block_sizes_by_coarse=[[2, 2], [2, 2]],
        n_blocks=bk["n_blocks"],
        block_size=bk["block_size"],
        intra_block_weight=bk["intra_block_weight"],
        interface_intra_weight=bk["interface_intra_weight"],
        interface_cross_weight=bk["interface_cross_weight"],
        noninterface_cross_weight=bk["noninterface_cross_weight"],
        self_weight=bk["self_weight"],
        interface_position_in_block=bk["interface_position_in_block"],
    )
    unb = build_substrate_family(
        "holonomy_interface_skew_family",
        fine_block_sizes_by_coarse=[[1, 3], [1, 3]],
        n_blocks=bk["n_blocks"],
        block_size=bk["block_size"],
        intra_block_weight=bk["intra_block_weight"],
        interface_intra_weight=bk["interface_intra_weight"],
        interface_cross_weight=bk["interface_cross_weight"],
        noninterface_cross_weight=bk["noninterface_cross_weight"],
        self_weight=bk["self_weight"],
        interface_position_in_block=bk["interface_position_in_block"],
    )
    pb = build_direct_and_routed_projectors(bal["coarse_lens"], bal["fine_lens"])
    pu = build_direct_and_routed_projectors(unb["coarse_lens"], unb["fine_lens"])
    lam = 0.85
    p_br = apply_closure_strength_control(
        bal["P"],
        closure_strength_lambda=lam,
        Q_f=pb["Q_c"],
        U_f=pb["U_c"],
        U_route=pb["U_route"],
        mode="mix_with_pi_reversible_routed_projector",
    )
    p_ur = apply_closure_strength_control(
        unb["P"],
        closure_strength_lambda=lam,
        Q_f=pu["Q_c"],
        U_f=pu["U_c"],
        U_route=pu["U_route"],
        mode="mix_with_pi_reversible_routed_projector",
    )
    q = pushforward_matrix(np.asarray(bal["coarse_lens"], dtype=np.int64), 2)
    u = np.asarray(pb["U_c"], dtype=np.float64)
    ph_b = macro_kernel(p_br, 2, q, u)
    ph_u = macro_kernel(p_ur, 2, q, u)
    off_b = float(0.5 * (ph_b[0, 1] + ph_b[1, 0]))
    off_u = float(0.5 * (ph_u[0, 1] + ph_u[1, 0]))
    assert abs(off_u - off_b) > 1e-6


def test_reduced_v2_execution_no_fake_arrow_and_cache(tmp_path: Path):
    cfg = _load_cfg()
    cfg["lambda_grid"] = [0.75, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "lb17_v2_note.md")
    out1 = run_holonomy_crossover_campaign_v2(cfg, output_root=tmp_path, use_cache=True)
    out2 = run_holonomy_crossover_campaign_v2(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out1["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "metrics" / "metrics.csv").exists()
    assert (root / "analysis" / "comparison_summary.json").exists()
    assert (root / "plots").exists()
    summary = json.loads((root / "analysis" / "comparison_summary.json").read_text(encoding="utf-8"))
    assert "no_fake_arrow_condition" in summary["comparison"]
    assert "holonomy_separation_condition" in summary["comparison"]
    assert "family_adequate_for_crossover" in summary["comparison"]

    rows = list((root / "metrics" / "metrics.csv").read_text(encoding="utf-8").splitlines())
    assert len(rows) >= 2
    data = np.genfromtxt(root / "metrics" / "metrics.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    aff = np.atleast_1d(data["affinity"]).astype(float)
    assert np.all(aff < 1e-6 + 1e-9)
    assert out2["cached_count"] > 0
