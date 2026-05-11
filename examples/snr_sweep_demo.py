"""Run a compact white-noise SNR sweep demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_timing_receiver.sweeps import run_white_noise_snr_sweep


def main() -> None:
    results = run_white_noise_snr_sweep(
        [30, 25, 20, 15, 10, 5],
        trial_count=5,
        base_seed=100,
    )

    print(
        "snr_db  trial_count  mean_rms_error  mean_rms_error_samples  "
        "mean_bias_error  mean_missed_count  mean_extra_count  max_rms_error"
    )
    for row in results:
        print(
            f"{row['snr_db']:6.1f}  "
            f"{row['trial_count']:11d}  "
            f"{row['mean_rms_error']:14.3e}  "
            f"{row['mean_rms_error_samples']:22.3f}  "
            f"{row['mean_bias_error']:15.3e}  "
            f"{row['mean_missed_count']:17.3f}  "
            f"{row['mean_extra_count']:16.3f}  "
            f"{row['max_rms_error']:13.3e}"
        )


if __name__ == "__main__":
    main()
