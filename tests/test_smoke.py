import sys
import subprocess


def test_smoke_module_runs():
    result = subprocess.run(
        [sys.executable, "-m", "layerbirth.smoke"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "layerbirth smoke check passed" in result.stdout
