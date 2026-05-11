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
    assert "detection_rate" in output
    assert "false_detections_per_100_pulses" in output
    assert "mean_rms_error_samples" in output
    assert "p95_abs_error" in output
