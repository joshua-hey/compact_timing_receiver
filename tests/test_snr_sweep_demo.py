import subprocess
import sys
from pathlib import Path


def test_snr_sweep_demo_script_runs_from_repository_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "examples/snr_sweep_demo.py"],
        check=True,
        capture_output=True,
        cwd=repo_root,
        text=True,
    )

    output = completed.stdout
    assert "snr_db" in output
    assert "mean_rms_error" in output
    assert "mean_rms_error_samples" in output
    assert "mean_bias_error" in output
    assert "mean_missed_count" in output
