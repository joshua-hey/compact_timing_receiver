import subprocess
import sys
from pathlib import Path


def test_phase_1_demo_script_runs_from_repository_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "examples/phase_1_demo.py"],
        check=True,
        capture_output=True,
        cwd=repo_root,
        text=True,
    )

    output = completed.stdout
    assert "Phase 1 timing-recovery demo" in output
    assert "True pulses:" in output
    assert "Estimated pulses:" in output
    assert "Missed detections:" in output
    assert "Extra detections:" in output
    assert "Mean timing error:" in output
    assert "RMS timing error:" in output
    assert "Max absolute timing error:" in output
