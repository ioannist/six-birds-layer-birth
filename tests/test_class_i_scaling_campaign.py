import json
from pathlib import Path

from layerbirth.campaigns import run_class_i_scaling_campaign


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    p = _repo_root() / "configs" / "campaigns" / "class_i_equilibrium_scaling.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_campaign_config_loads():
    cfg = _load_cfg()
    assert cfg["campaign_id"] == "class_i_equilibrium"
    assert "primary_panel" in cfg
    assert "shadow_panel" in cfg


def test_reduced_campaign_execution(tmp_path: Path):
    cfg = _load_cfg()
    cfg["primary_panel"]["sizes"] = [8, 16]
    cfg["primary_panel"]["lambda_grid"] = [0.75, 1.0]
    cfg["shadow_panel"]["sizes"] = [8, 16]
    cfg["shadow_panel"]["lambda_grid"] = [0.75, 1.0]
    cfg["shadow_panel"]["seeds"] = [7, 11]
    cfg["findings_note_path"] = str(tmp_path / "note.md")

    out = run_class_i_scaling_campaign(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out["artifact_root"])
    assert (root / "manifest.json").exists()
    assert (root / "metrics" / "metrics.csv").exists()
    assert (root / "analysis" / "primary_group_summary.csv").exists()
    assert (root / "analysis" / "shadow_group_summary.csv").exists()
    assert (root / "analysis" / "fit_summary.json").exists()
    assert (root / "plots").exists()
    assert out["go_no_go"] in {"go", "no-go"}


def test_fit_or_failure_plumbing(tmp_path: Path):
    cfg = _load_cfg()
    cfg["primary_panel"]["sizes"] = [8, 16]
    cfg["primary_panel"]["lambda_grid"] = [0.75, 1.0]
    cfg["shadow_panel"]["sizes"] = [8, 16]
    cfg["shadow_panel"]["lambda_grid"] = [0.75, 1.0]
    cfg["shadow_panel"]["seeds"] = [7, 11]
    cfg["findings_note_path"] = str(tmp_path / "note2.md")
    out = run_class_i_scaling_campaign(cfg, output_root=tmp_path, use_cache=True)
    root = Path(out["artifact_root"])
    fit_path = root / "analysis" / "fit_summary.json"
    assert fit_path.exists()
    if out["failure_report_needed"]:
        fr = root / "analysis" / "failure_report.json"
        assert fr.exists()
        payload = json.loads(fr.read_text(encoding="utf-8"))
        for key in ("best_fit_parameters", "collapse_objective", "shadow_crossings", "diagnosis"):
            assert key in payload


def test_cache_behavior(tmp_path: Path):
    cfg = _load_cfg()
    cfg["primary_panel"]["sizes"] = [8, 16]
    cfg["primary_panel"]["lambda_grid"] = [0.75, 1.0]
    cfg["shadow_panel"]["sizes"] = [8, 16]
    cfg["shadow_panel"]["lambda_grid"] = [0.75, 1.0]
    cfg["shadow_panel"]["seeds"] = [7, 11]
    cfg["findings_note_path"] = str(tmp_path / "note3.md")
    first = run_class_i_scaling_campaign(cfg, output_root=tmp_path, use_cache=True)
    second = run_class_i_scaling_campaign(cfg, output_root=tmp_path, use_cache=True)
    assert first["executed_count"] > 0
    assert second["cached_count"] > 0
