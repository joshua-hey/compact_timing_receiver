"""Run a compact white-noise SNR sweep demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_timing_receiver.sweeps import run_white_noise_snr_sweep


def main() -> None:
    results = run_white_noise_snr_sweep(
        [30, 25, 20, 15, 10, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4, 3, 2, 1, 0],
        trial_count=20,
        pulse_count=100,
        sample_rate=10_000,
        off_grid=True,
        base_seed=100,
    )

    print("Off-grid AWGN SNR sweep")
    print(
        "snr_db  trial_count  requested_pulse_count  total_true_pulses  detection_rate  "
        "detection_rate_ci_low  detection_rate_ci_high  "
        "false_detections_per_100_pulses  mean_rms_error_samples  "
        "mean_bias_error_samples  p95_abs_error_samples"
    )
    for row in results:
        print(
            f"{row['snr_db']:6.1f}  "
            f"{row['trial_count']:11d}  "
            f"{row['requested_pulse_count']:21d}  "
            f"{row['total_true_pulses']:17d}  "
            f"{row['detection_rate']:14.3f}  "
            f"{row['detection_rate_ci_low']:21.3f}  "
            f"{row['detection_rate_ci_high']:22.3f}  "
            f"{row['false_detections_per_100_pulses']:32.3f}  "
            f"{row['mean_rms_error_samples']:22.3f}  "
            f"{row['mean_bias_error_samples']:23.3f}  "
            f"{row['p95_abs_error_samples']:21.3f}"
        )


if __name__ == "__main__":
    main()
