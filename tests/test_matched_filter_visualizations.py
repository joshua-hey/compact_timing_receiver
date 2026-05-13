import subprocess
import sys
import uuid
from pathlib import Path


def test_matched_filter_visualizations_script_help_includes_output_options() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "examples/matched_filter_visualizations.py", "--help"],
        check=True,
        capture_output=True,
        cwd=repo_root,
        text=True,
    )

    assert "--output-dir" in completed.stdout
    assert "--plot" in completed.stdout
    assert "--heatmap-offset-count" in completed.stdout
    assert "--efficiency-snr-db" in completed.stdout
    assert "--cdf-snr-db" in completed.stdout
    assert "--tradeoff-snr-db" in completed.stdout


def test_matched_filter_visualizations_script_writes_plots() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "artifacts" / f"pytest_matched_filter_viz_{uuid.uuid4().hex}"
    peak_path = output_dir / "matched_filter_peak_anatomy.png"
    heatmap_path = output_dir / "matched_filter_fractional_offset_error_heatmap.png"
    efficiency_path = output_dir / "matched_filter_crlb_efficiency.png"
    cdf_path = output_dir / "matched_filter_absolute_error_cdf.png"
    tradeoff_path = output_dir / "matched_filter_detection_tradeoff.png"

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "examples/matched_filter_visualizations.py",
                "--output-dir",
                str(output_dir),
                "--heatmap-offset-count",
                "5",
                "--heatmap-trial-count",
                "1",
                "--heatmap-snr-db",
                "20",
                "40",
                "--efficiency-snr-db",
                "20",
                "40",
                "--efficiency-trial-count",
                "1",
                "--efficiency-pulse-count",
                "3",
                "--cdf-snr-db",
                "30",
                "20",
                "--cdf-trial-count",
                "1",
                "--cdf-pulse-count",
                "3",
                "--tradeoff-snr-db",
                "0",
                "3",
                "--tradeoff-trial-count",
                "1",
                "--tradeoff-pulse-count",
                "3",
                "--tradeoff-threshold-count",
                "5",
            ],
            check=True,
            capture_output=True,
            cwd=repo_root,
            text=True,
        )

        assert "Matched-filter visualizations" in completed.stdout
        assert "peak_anatomy_sample_grid_error_samples:" in completed.stdout
        assert "peak_anatomy_parabolic_error_samples:" in completed.stdout
        assert "heatmap_offset_count: 5" in completed.stdout
        assert "efficiency_trial_count: 1" in completed.stdout
        assert "cdf_trial_count: 1" in completed.stdout
        assert "tradeoff_threshold_count: 5" in completed.stdout
        assert peak_path.exists()
        assert heatmap_path.exists()
        assert efficiency_path.exists()
        assert cdf_path.exists()
        assert tradeoff_path.exists()
        assert peak_path.stat().st_size > 0
        assert heatmap_path.stat().st_size > 0
        assert efficiency_path.stat().st_size > 0
        assert cdf_path.stat().st_size > 0
        assert tradeoff_path.stat().st_size > 0
    finally:
        peak_path.unlink(missing_ok=True)
        heatmap_path.unlink(missing_ok=True)
        efficiency_path.unlink(missing_ok=True)
        cdf_path.unlink(missing_ok=True)
        tradeoff_path.unlink(missing_ok=True)
        if output_dir.exists():
            output_dir.rmdir()
