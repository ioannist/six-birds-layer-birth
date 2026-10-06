import json
from pathlib import Path

from layerbirth.class3 import (
    evaluate_strict_confirmation_variant,
    summarize_strict_confirmation,
    run_class_iii_strict_confirmation_v2,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    return json.loads(
        (_repo_root() / "configs" / "pilots" / "class_iii_strict_confirmation_v2.json").read_text(
            encoding="utf-8"
        )
    )


# A. Config loading

def test_config_loading():
    cfg = _load_cfg()
    assert cfg["pilot_id"] == "class_iii_strict_confirmation_v2"
    assert cfg["artifact_subdir"] == "class_iii_strict_confirmation_v2"
    assert len(cfg["candidates"]) == 3
    names = [c["candidate_name"] for c in cfg["candidates"]]
    assert "replicated_portal_sp2" in names
    assert "replicated_portal_sp3" in names
    assert "replicated_portal_sp4" in names
    assert cfg["scale_invariance_cv_threshold"] == 0.05
    assert cfg["size_panel"] == [16, 32, 64, 128]


# B. Persistence / invariance logic with synthetic data

_size_results_confirmed = [
    {"size": 16, "candidate_is_class_iii_strict": True, "staging_shift_ce": 0.163, "staging_shift_mobj": 0.310},
    {"size": 32, "candidate_is_class_iii_strict": True, "staging_shift_ce": 0.163, "staging_shift_mobj": 0.310},
    {"size": 64, "candidate_is_class_iii_strict": True, "staging_shift_ce": 0.163, "staging_shift_mobj": 0.310},
    {"size": 128, "candidate_is_class_iii_strict": True, "staging_shift_ce": 0.163, "staging_shift_mobj": 0.310},
]

_size_results_near_miss = [
    {"size": 16, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.10, "staging_shift_mobj": 0.20},
    {"size": 32, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.10, "staging_shift_mobj": 0.20},
    {"size": 64, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.10, "staging_shift_mobj": 0.20},
    {"size": 128, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.10, "staging_shift_mobj": 0.20},
]

_size_results_partial = [
    {"size": 16, "candidate_is_class_iii_strict": True, "staging_shift_ce": 0.163, "staging_shift_mobj": 0.310},
    {"size": 32, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.10, "staging_shift_mobj": 0.10},
    {"size": 64, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.10, "staging_shift_mobj": 0.10},
    {"size": 128, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.10, "staging_shift_mobj": 0.10},
]

_size_results_low_shift_near_miss = [
    {"size": 16, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.05, "staging_shift_mobj": 0.05},
    {"size": 32, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.05, "staging_shift_mobj": 0.05},
    {"size": 64, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.05, "staging_shift_mobj": 0.05},
    {"size": 128, "candidate_is_class_iii_strict": False, "staging_shift_ce": 0.05, "staging_shift_mobj": 0.05},
]


def test_persistence_confirmed_label():
    result = evaluate_strict_confirmation_variant(_size_results_confirmed, scale_cv_threshold=0.05)
    assert result["strict_hit_persistent"] is True
    assert result["scale_invariance_supported"] is True
    assert result["final_variant_label"] == "strict_confirmed"


def test_persistence_near_miss_label():
    result = evaluate_strict_confirmation_variant(_size_results_near_miss, scale_cv_threshold=0.05)
    assert result["strict_hit_persistent"] is False
    assert result["scale_invariance_supported"] is False
    # mobj_max is 0.20 >= 0.15 → persistent_near_miss
    assert result["final_variant_label"] == "persistent_near_miss"


def test_persistence_low_shift_near_miss():
    result = evaluate_strict_confirmation_variant(_size_results_low_shift_near_miss, scale_cv_threshold=0.05)
    assert result["strict_hit_persistent"] is False
    # max mo_shift is 0.05 < 0.15 → near_miss
    assert result["final_variant_label"] == "near_miss"


def test_persistence_partial_strict():
    result = evaluate_strict_confirmation_variant(_size_results_partial, scale_cv_threshold=0.05)
    assert result["strict_hit_persistent"] is False
    assert any(result["strict_hits_by_size"].values()) is True
    assert result["final_variant_label"] == "strict_but_not_persistent"


def test_size_panel_and_strict_hits_by_size():
    result = evaluate_strict_confirmation_variant(_size_results_confirmed, scale_cv_threshold=0.05)
    assert result["size_panel"] == [16, 32, 64, 128]
    assert all(result["strict_hits_by_size"].values())


def test_cv_threshold_enforcement():
    # Introduce variance in CE to exceed threshold
    varied = [
        {"size": 16, "candidate_is_class_iii_strict": True, "staging_shift_ce": 0.10, "staging_shift_mobj": 0.310},
        {"size": 32, "candidate_is_class_iii_strict": True, "staging_shift_ce": 0.50, "staging_shift_mobj": 0.310},
        {"size": 64, "candidate_is_class_iii_strict": True, "staging_shift_ce": 0.10, "staging_shift_mobj": 0.310},
        {"size": 128, "candidate_is_class_iii_strict": True, "staging_shift_ce": 0.50, "staging_shift_mobj": 0.310},
    ]
    result = evaluate_strict_confirmation_variant(varied, scale_cv_threshold=0.05)
    # persistent (all strict) but CE CV is large → scale_invariance_supported False
    assert result["strict_hit_persistent"] is True
    assert result["scale_invariance_supported"] is False
    assert result["final_variant_label"] == "strict_persistent_but_not_scale_stable"


def test_summarize_strict_confirmation_confirmed():
    vs = [
        {
            "candidate_name": "sp3",
            "confirmation": {
                "final_variant_label": "strict_confirmed",
                "strict_hits_by_size": {16: True, 32: True},
                "shift_mobj_mean": 0.31,
            },
        },
        {
            "candidate_name": "sp2",
            "confirmation": {
                "final_variant_label": "near_miss",
                "strict_hits_by_size": {16: False, 32: False},
                "shift_mobj_mean": 0.20,
            },
        },
    ]
    overall = summarize_strict_confirmation(vs)
    assert overall["final_verdict"] == "strict_class_iii_confirmed"
    assert overall["best_variant"] == "sp3"
    assert overall["confirmed_count"] == 1


def test_summarize_strict_confirmation_no_confirmed():
    vs = [
        {
            "candidate_name": "sp3",
            "confirmation": {
                "final_variant_label": "strict_but_not_persistent",
                "strict_hits_by_size": {16: True, 32: False},
                "shift_mobj_mean": 0.25,
            },
        },
        {
            "candidate_name": "sp2",
            "confirmation": {
                "final_variant_label": "near_miss",
                "strict_hits_by_size": {16: False, 32: False},
                "shift_mobj_mean": 0.10,
            },
        },
    ]
    overall = summarize_strict_confirmation(vs)
    assert overall["final_verdict"] == "strict_hit_present_but_not_yet_confirmed"
    assert overall["best_variant"] == "sp3"


def test_summarize_strict_confirmation_all_miss():
    vs = [
        {
            "candidate_name": "sp2",
            "confirmation": {
                "final_variant_label": "near_miss",
                "strict_hits_by_size": {16: False, 32: False},
                "shift_mobj_mean": 0.10,
            },
        },
    ]
    overall = summarize_strict_confirmation(vs)
    assert overall["final_verdict"] == "no_strict_class_iii_after_cleanup"


# C. Cheap execution

def test_cheap_execution(tmp_path: Path):
    cfg = _load_cfg()
    cfg["size_panel"] = [16]
    cfg["lambda_grid"] = [0.55, 1.0]
    cfg["findings_note_path"] = str(tmp_path / "S2-04_class_iii_strict_confirmation_v2.md")
    out = run_class_iii_strict_confirmation_v2(cfg, output_root=tmp_path, use_cache=False)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "analysis" / "confirmation_table.csv").exists()
    assert (root / "analysis" / "confirmation_summary.json").exists()
    assert (root / "metrics" / "metrics.csv").exists()
    assert out["final_verdict"] in {
        "strict_class_iii_confirmed",
        "strict_hit_present_but_not_yet_confirmed",
        "no_strict_class_iii_after_cleanup",
    }
    assert "variant_summaries" in out
    assert len(out["variant_summaries"]) == 3


# D. Provenance cleanup

def test_registry_contains_v2_entry():
    registry_path = _repo_root() / "configs" / "experiment_registry.yaml"
    text = registry_path.read_text(encoding="utf-8")
    assert "class_iii_strict_confirmation_v2" in text


def test_registry_legacy_entry_not_confirmed():
    registry_path = _repo_root() / "configs" / "experiment_registry.yaml"
    text = registry_path.read_text(encoding="utf-8")
    # The legacy entry must no longer claim "First confirmed strict Class-III substrate."
    assert "First confirmed strict Class-III substrate." not in text


def test_registry_legacy_entry_is_superseded():
    registry_path = _repo_root() / "configs" / "experiment_registry.yaml"
    text = registry_path.read_text(encoding="utf-8")
    # Legacy entry should now have status superseded
    assert "superseded" in text
