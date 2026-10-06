import json
from pathlib import Path

from layerbirth.capacity import (
    compute_axis_effects,
    compute_capacity_depth_proxies,
    run_capacity_depth_extension,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads(
        (_repo_root() / "configs" / "campaigns" / "capacity_depth_extension.json").read_text(
            encoding="utf-8"
        )
    )


def test_config_loading():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "capacity_depth_extension"
    assert cfg["tau_ladder"] == [1, 2, 4, 8]
    assert set(cfg["representatives"].keys()) == {"Class-I", "Class-II", "Class-III", "Class-IV"}


def test_proxy_plumbing():
    rows = [
        {"lambda_name": "post", "tau": 1, "closure_error": 0.02, "objecthood_order": 0.91, "affinity": 0.0},
        {"lambda_name": "post", "tau": 2, "closure_error": 0.03, "objecthood_order": 0.89, "affinity": 0.0},
        {"lambda_name": "sat", "tau": 1, "closure_error": 0.01, "objecthood_order": 0.95, "affinity": 0.0},
        {"lambda_name": "sat", "tau": 2, "closure_error": 0.02, "objecthood_order": 0.96, "affinity": 0.0},
        {"lambda_name": "sat", "tau": 4, "closure_error": 0.03, "objecthood_order": 0.80, "affinity": 0.0},
    ]
    p = compute_capacity_depth_proxies(rows, 0.7)
    assert abs(p["lambda_headroom"] - 0.3) < 1e-12
    assert p["depth_survival_post"] == 1
    assert p["depth_survival_sat"] == 2
    assert p["objecthood_tau_area_post"] > 0
    assert p["objecthood_tau_area_sat"] > 0
    assert p["capacity_depth_index"] > 0


def test_axis_effect_logic():
    # p4-dominant synthetic
    rows = []
    for cls, base in [("Class-I", 1.0), ("Class-II", 1.1), ("Class-III", 3.0), ("Class-IV", 3.1)]:
        for s in [16, 32]:
            rows.append(
                {
                    "class_name": cls,
                    "size": s,
                    "lambda_headroom": base,
                    "depth_survival_sat": base,
                    "objecthood_tau_area_sat": base,
                    "capacity_depth_index": base,
                }
            )
    a = compute_axis_effects(rows, {})
    assert a["capacity_tracks_p4_axis"] is True
    assert a["capacity_primary_classifier"] is False


def test_cheap_execution(tmp_path: Path):
    cfg = _load_cfg()
    cfg["tau_ladder"] = [1, 2]
    cfg["size_panel"] = [16, 32]
    cfg["findings_note_path"] = str(tmp_path / "LB-19_capacity_depth_extension.md")
    out = run_capacity_depth_extension(cfg, output_root=tmp_path, use_cache=False)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "capacity_proxy_summary.json").exists()
    assert (root / "analysis" / "axis_effect_summary.json").exists()
    assert (root / "analysis" / "final_verdict.json").exists()
    assert out["final_verdict"] in {
        "capacity_tracks_p4_axis",
        "capacity_tracks_p6_axis",
        "capacity_orthogonal_or_mixed",
    }


def test_registry_entry_exists():
    text = (_repo_root() / "configs" / "experiment_registry.yaml").read_text(encoding="utf-8")
    assert "capacity_depth_extension" in text
