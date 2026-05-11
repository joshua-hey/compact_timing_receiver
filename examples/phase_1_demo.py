"""Run a small Phase 1 timing-recovery demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_timing_receiver.experiments import run_timing_recovery_trial


def _format_seconds(value: float) -> str:
    return f"{value:.3e} s"


def main() -> None:
    result = run_timing_recovery_trial(
        seed=7,
        white_noise_std=0.02,
        baseline_drift_amplitude=0.03,
        baseline_drift_frequency=2.0,
        adc_bits=10,
    )
    summary = result["summary"]

    print("Phase 1 timing-recovery demo")
    print(f"True pulses: {result['true_arrival_times'].size}")
    print(f"Estimated pulses: {result['estimated_arrival_times'].size}")
    print(f"Missed detections: {summary['missed_count']}")
    print(f"Extra detections: {summary['extra_count']}")
    print(f"Mean timing error: {_format_seconds(summary['mean_error'])}")
    print(f"RMS timing error: {_format_seconds(summary['rms_error'])}")
    print(f"Max absolute timing error: {_format_seconds(summary['max_abs_error'])}")


if __name__ == "__main__":
    main()
