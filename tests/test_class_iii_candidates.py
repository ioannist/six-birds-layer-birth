import json
from pathlib import Path

import numpy as np

from layerbirth.class3 import evaluate_class_iii_candidate, run_class_iii_candidate_pilots
from layerbirth.numeric import validate_row_stochastic
from layerbirth.substrates import (
    delayed_interface_reversible_family,
    hidden_sector_reversible_family,
    two_timescale_reversible_family,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads((_repo_root() / "configs" / "pilots" / "class_iii_candidates.json").read_text(encoding="utf-8"))


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["pilot_id"] == "class_iii_candidates_v1"
    assert len(cfg["candidates"]) == 3


def test_family_generator_sanity():
    fams = [
        delayed_interface_reversible_family(),
        hidden_sector_reversible_family(),
        two_timescale_reversible_family(),
    ]
    for f in fams:
        p = np.asarray(f["P"], dtype=np.float64)
        validate_row_stochastic(p)
        assert np.allclose(p, p.T, atol=1e-10)
        lens = np.asarray(f["coarse_lens"], dtype=np.int64)
        assert lens.shape[0] == p.shape[0]
        assert int(np.min(lens)) == 0
        assert int(np.max(lens)) == 1


def test_candidate_evaluation_rule():
    rubric = json.loads((_repo_root() / "configs" / "taxonomy" / "canonical_class_rubric.json").read_text(encoding="utf-8"))
    p4_cfg = json.loads((_repo_root() / "configs" / "metrics" / "p4_anomaly_metric_layer.json").read_text(encoding="utf-8"))

    candidate = {
        "tau1_boundary": {"ce_boundary_lambda": 0.6, "mobj_boundary_lambda": 0.6, "min_ce": 0.0, "max_mobj": 1.0},
        "tau2_boundary": {"ce_boundary_lambda": 0.8, "mobj_boundary_lambda": 0.8, "min_ce": 0.0, "max_mobj": 1.0},
        "tau1_affinity_ref": {"affinity_ref": 0.0},
        "tau2_affinity_ref": {"affinity_ref": 0.0},
    }
    out = evaluate_class_iii_candidate(candidate, rubric, p4_cfg)
    assert out["candidate_p5_state"] == "active"
    assert out["candidate_p6_drive_state"] == "inactive"
    assert out["candidate_p4_state"] == "active"
    assert out["candidate_is_class_iii"] is True


def test_cheap_execution_and_cache(tmp_path: Path):
    cfg = _load_cfg()
    cfg["shared"]["lambda_grid"] = [0.75, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-04_class_iii_candidate_pilots.md")
    out1 = run_class_iii_candidate_pilots(cfg, output_root=tmp_path, use_cache=True)
    out2 = run_class_iii_candidate_pilots(cfg, output_root=tmp_path, use_cache=True)

    root = Path(out1["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "candidate_summary.json").exists()
    assert (root / "analysis" / "candidate_table.csv").exists()
    assert (root / "candidates" / "delayed_interface" / "summary.json").exists()
    assert out1["delayed_interface"].get("candidate_status") in {"Class-III", "near_miss", "unclassified"}
    assert out2["best_candidate"] == out1["best_candidate"]
