import json
from pathlib import Path

import numpy as np
import pytest

from layerbirth.controls import evaluate_control_case, no_fake_arrow_controls, protocol_trap_control
from layerbirth.metrics import affinity_metric
from layerbirth.numeric import validate_row_stochastic


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cfg(rel: str) -> dict:
    path = _repo_root() / "configs" / "controls" / rel
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_trap_hidden_vs_phase_aware():
    hidden_cfg = _cfg("protocol_trap_hidden_schedule.json")
    phase_cfg = _cfg("protocol_trap_phase_aware.json")
    ctrl_hidden = protocol_trap_control(
        pair_self_weight=hidden_cfg["pair_self_weight"],
        schedule_order=hidden_cfg["schedule_order"],
        handling=hidden_cfg["handling"],
    )
    ctrl_phase = protocol_trap_control(
        pair_self_weight=phase_cfg["pair_self_weight"],
        schedule_order=phase_cfg["schedule_order"],
        handling=phase_cfg["handling"],
    )

    expected_hidden = np.array(
        [
            [0.552, 0.16, 0.288],
            [0.288, 0.04, 0.672],
            [0.16, 0.8, 0.04],
        ],
        dtype=np.float64,
    )

    for kernel in ctrl_phase["phase_kernels"].values():
        validate_row_stochastic(kernel)
        aff_phase, _ = affinity_metric(kernel, tau=1)
        assert np.isclose(aff_phase, 0.0, atol=1e-12)

    np.testing.assert_allclose(ctrl_hidden["hidden_effective_kernel"], expected_hidden, atol=1e-12)
    aff_hidden, _ = affinity_metric(ctrl_hidden["hidden_effective_kernel"], tau=1)
    assert aff_hidden > 0.0
    assert np.isclose(aff_hidden, 0.0575968733, atol=1e-10)

    hidden_eval = evaluate_control_case(
        case_name="protocol_trap_hidden_schedule",
        control_family="protocol_trap",
        P=ctrl_hidden["hidden_effective_kernel"],
        analysis_lens=ctrl_hidden["analysis_lens"],
    )
    assert hidden_eval["driven_candidate"] is True

    for phase_name in hidden_cfg["schedule_order"]:
        phase_eval = evaluate_control_case(
            case_name=f"protocol_trap_phase_aware_{phase_name}",
            control_family="protocol_trap",
            P=ctrl_phase["phase_kernels"][phase_name],
            analysis_lens=ctrl_phase["analysis_lens"],
        )
        assert np.isclose(phase_eval["affinity"], 0.0, atol=1e-12)
        assert phase_eval["driven_candidate"] is False


def test_no_fake_arrow_controls_no_spurious_driven():
    cfg = _cfg("no_fake_arrow_controls.json")
    controls = no_fake_arrow_controls(cases_config=cfg["cases"])
    assert len(controls) == 3

    results = []
    for case in controls:
        validate_row_stochastic(case["P"])
        eval_row = evaluate_control_case(
            case_name=case["case_name"],
            control_family="no_fake_arrow",
            P=case["P"],
            analysis_lens=case["analysis_lens"],
        )
        assert np.isclose(eval_row["affinity"], 0.0, atol=1e-10)
        assert eval_row["driven_candidate"] is False
        results.append((case["case_name"], case["P"]))

    p_manual = [p for name, p in results if name == "reversible_block_manual"][0]
    p_spectral = [p for name, p in results if name == "reversible_block_spectral"][0]
    np.testing.assert_allclose(p_manual, p_spectral, atol=1e-15)


def test_control_error_paths():
    with pytest.raises(ValueError):
        protocol_trap_control(pair_self_weight=-0.1)
    with pytest.raises(ValueError):
        protocol_trap_control(pair_self_weight=1.1)
    with pytest.raises(ValueError):
        protocol_trap_control(pair_self_weight=0.2, schedule_order=[])
    with pytest.raises(ValueError):
        no_fake_arrow_controls(
            cases_config=[
                {
                    "case_name": "bad",
                    "family": "reversible_block_family",
                    "family_params": {
                        "n_blocks": 2,
                        "block_size": 4,
                        "intra_block_weight": 1.0,
                        "inter_block_weight": 0.05,
                        "self_weight": 0.0,
                    },
                    "lens_mode": "manual",
                }
            ]
        )
