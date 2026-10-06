import json
from pathlib import Path

import numpy as np

from layerbirth.crossover import build_direct_and_routed_projectors, run_holonomy_crossover_campaign
from layerbirth.numeric import validate_row_stochastic
from layerbirth.substrates import build_substrate_family
from layerbirth.sweep import apply_closure_strength_control


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "campaigns" / "holonomy_crossover.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_holonomy_crossover_config_loads():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "holonomy_crossover"
    assert len(cfg["cases"]) == 2
    assert len(cfg["control_routes"]) == 2


def test_routed_projector_plumbing_and_row_stochastic():
    common = {
        "n_blocks": 2,
        "block_size": 4,
        "intra_block_weight": 1.0,
        "inter_block_weight": 0.05,
        "self_weight": 0.0,
    }
    bal = build_substrate_family(
        "holonomy_control_family",
        fine_block_sizes_by_coarse=[[2, 2], [2, 2]],
        **common,
    )
    unbal = build_substrate_family(
        "holonomy_control_family",
        fine_block_sizes_by_coarse=[[1, 3], [1, 3]],
        **common,
    )

    pb = build_direct_and_routed_projectors(bal["coarse_lens"], bal["fine_lens"])
    pu = build_direct_and_routed_projectors(unbal["coarse_lens"], unbal["fine_lens"])
    assert np.allclose(pb["Pi_direct"], pb["Pi_route"], atol=1e-12, rtol=0.0)
    assert not np.allclose(pu["Pi_direct"], pu["Pi_route"], atol=1e-12, rtol=0.0)

    lam = 0.75
    p_bal_direct = apply_closure_strength_control(
        bal["P"],
        closure_strength_lambda=lam,
        Q_f=pb["Q_c"],
        U_f=pb["U_c"],
        mode="mix_with_packaging_projector",
    )
    p_bal_routed = apply_closure_strength_control(
        bal["P"],
        closure_strength_lambda=lam,
        Q_f=pb["Q_c"],
        U_f=pb["U_c"],
        U_route=pb["U_route"],
        mode="mix_with_routed_packaging_projector",
    )
    p_unbal_direct = apply_closure_strength_control(
        unbal["P"],
        closure_strength_lambda=lam,
        Q_f=pu["Q_c"],
        U_f=pu["U_c"],
        mode="mix_with_packaging_projector",
    )
    p_unbal_routed = apply_closure_strength_control(
        unbal["P"],
        closure_strength_lambda=lam,
        Q_f=pu["Q_c"],
        U_f=pu["U_c"],
        U_route=pu["U_route"],
        mode="mix_with_routed_packaging_projector",
    )
    for p in [p_bal_direct, p_bal_routed, p_unbal_direct, p_unbal_routed]:
        validate_row_stochastic(p)


def test_reduced_holonomy_crossover_execution(tmp_path: Path):
    cfg = _load_cfg()
    cfg["lambda_grid"] = [0.75, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "lb17_note.md")
    out = run_holonomy_crossover_campaign(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "metrics" / "metrics.csv").exists()
    assert (root / "analysis" / "comparison_summary.json").exists()
    assert (root / "plots").exists()
    summary = json.loads((root / "analysis" / "comparison_summary.json").read_text(encoding="utf-8"))
    assert "no_fake_arrow_condition" in summary["comparison"]
    assert "holonomy_separation_condition" in summary["comparison"]
    assert "family_adequate_for_crossover" in summary["comparison"]


def test_no_fake_arrow_logic_and_cache(tmp_path: Path):
    cfg = _load_cfg()
    cfg["lambda_grid"] = [0.75, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "lb17_note2.md")
    first = run_holonomy_crossover_campaign(cfg, output_root=tmp_path, use_cache=True)
    second = run_holonomy_crossover_campaign(cfg, output_root=tmp_path, use_cache=True)
    root = Path(first["artifact_root"])
    summary = json.loads((root / "analysis" / "comparison_summary.json").read_text(encoding="utf-8"))
    assert isinstance(summary["comparison"]["no_fake_arrow_condition"], bool)
    assert "curve_summaries" in summary["comparison"]
    for key in ("balanced_direct", "balanced_routed", "unbalanced_direct", "unbalanced_routed"):
        assert key in summary["comparison"]["curve_summaries"]
    assert second["cached_count"] > 0
