"""Run a compact white-noise SNR sweep demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_timing_receiver.sweeps import run_white_noise_snr_sweep


def main() -> None:
    results = run_white_noise_snr_sweep(
        [30, 25, 20, 15, 10, 5, 0],
        trial_count=20,
        pulse_count=20,
        base_seed=100,
    )

    print(
        "snr_db  trial_count  pulse_count  total_true_pulses  detection_rate  "
        "missed_detection_rate  false_detections_per_100_pulses  "
        "mean_rms_error_samples  mean_bias_error  p95_abs_error"
    )
    for row in results:
        print(
            f"{row['snr_db']:6.1f}  "
            f"{row['trial_count']:11d}  "
            f"{row['pulse_count']:11d}  "
            f"{row['total_true_pulses']:17d}  "
            f"{row['detection_rate']:14.3f}  "
            f"{row['missed_detection_rate']:21.3f}  "
            f"{row['false_detections_per_100_pulses']:32.3f}  "
            f"{row['mean_rms_error_samples']:22.3f}  "
            f"{row['mean_bias_error']:15.3e}  "
            f"{row['p95_abs_error']:13.3e}"
        )


if __name__ == "__main__":
    main()
