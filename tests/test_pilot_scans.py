import json
from pathlib import Path

from layerbirth.pilots import run_pilot_scan, summarize_pilot_window


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(name: str) -> dict:
    path = _repo_root() / "configs" / "pilots" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_pilot_configs_load():
    for name in (
        "equilibrium_like_lambda_scan.json",
        "driven_family_lambda_scan.json",
        "null_family_control_scan.json",
        "holonomy_control_lambda_scan.json",
    ):
        cfg = _load_config(name)
        assert isinstance(cfg, dict)
        assert "pilot_family" in cfg
        assert "cases" in cfg


def test_cheap_pilot_execution(tmp_path: Path):
    eq = _load_config("equilibrium_like_lambda_scan.json")
    null = _load_config("null_family_control_scan.json")
    eq["findings_note_path"] = str(tmp_path / "eq_note.md")
    null["findings_note_path"] = str(tmp_path / "null_note.md")

    out_eq = run_pilot_scan(eq, output_root=tmp_path, use_cache=True)
    out_null = run_pilot_scan(null, output_root=tmp_path, use_cache=True)
    for out in (out_eq, out_null):
        root = Path(out["artifact_root"])
        assert (root / "manifest.json").exists()
        assert (root / "metrics" / "metrics.csv").exists()
        summary_path = Path(out["summary_path"])
        assert summary_path.exists()
        assert (root / "plots").exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "cases" in summary


def test_window_rule_logic():
    synth = [
        {
            "case_name": "a",
            "closure_strength_lambda": 0.0,
            "objecthood_order": 0.0,
            "closure_error": 0.0,
            "affinity": 0.0,
            "staging_gap": 0.0,
            "holonomy": None,
        },
        {
            "case_name": "a",
            "closure_strength_lambda": 0.5,
            "objecthood_order": 0.2,
            "closure_error": 0.1,
            "affinity": 0.0,
            "staging_gap": 0.0,
            "holonomy": None,
        },
    ]
    out = summarize_pilot_window(synth, "equilibrium_like")
    assert out["cases"]["a"]["diagnosis"].startswith("candidate transition window")

    flat = [
        {
            "case_name": "b",
            "closure_strength_lambda": 0.0,
            "objecthood_order": 0.1,
            "closure_error": 0.1,
            "affinity": 0.0,
            "staging_gap": 0.0,
            "holonomy": None,
        },
        {
            "case_name": "b",
            "closure_strength_lambda": 1.0,
            "objecthood_order": 0.1,
            "closure_error": 0.1,
            "affinity": 0.0,
            "staging_gap": 0.0,
            "holonomy": None,
        },
    ]
    out2 = summarize_pilot_window(flat, "null_family")
    assert out2["cases"]["b"]["diagnosis"] == "no clear transition window on scanned grid"


def test_holonomy_control_diagnosis(tmp_path: Path):
    cfg = _load_config("holonomy_control_lambda_scan.json")
    cfg["findings_note_path"] = str(tmp_path / "hol_note.md")
    out = run_pilot_scan(cfg, output_root=tmp_path, use_cache=True)
    summary = json.loads(Path(out["summary_path"]).read_text(encoding="utf-8"))
    assert summary["holonomy_separation"] > 0.05
    assert summary["max_affinity_abs"] < 1e-6
    assert "driven-family success" not in summary.get("diagnosis", "")


def test_simple_cache_rerun(tmp_path: Path):
    cfg = _load_config("equilibrium_like_lambda_scan.json")
    cfg["findings_note_path"] = str(tmp_path / "eq_note_cache.md")
    first = run_pilot_scan(cfg, output_root=tmp_path, use_cache=True)
    second = run_pilot_scan(cfg, output_root=tmp_path, use_cache=True)
    assert first["executed_count"] > 0
    assert second["cached_count"] == first["run_count"]
