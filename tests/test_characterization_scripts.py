import subprocess
import sys
from pathlib import Path


def test_crlb_characterization_script_help_includes_output_dir() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "examples/crlb_characterization.py", "--help"],
        check=True,
        capture_output=True,
        cwd=repo_root,
        text=True,
    )

    assert "--output-dir" in completed.stdout


def test_interpolation_characterization_script_help_includes_output_dir() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "examples/interpolation_characterization.py", "--help"],
        check=True,
        capture_output=True,
        cwd=repo_root,
        text=True,
    )

    assert "--output-dir" in completed.stdout
