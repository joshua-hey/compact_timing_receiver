# Compact Optical/RF Timing Receiver

Compact Optical/RF Timing Receiver is a Phase 1 prototype: a small Python toolkit for generating synthetic pulse trains, adding simple signal impairments, estimating pulse arrival times with threshold and matched-filter methods, and characterizing timing performance under white-noise SNR sweeps.

The goal of this first phase is to provide a readable, testable foundation for timing-recovery experiments. It is intended for simulation, algorithm exploration, and early validation only. It does not yet include hardware integration, calibrated RF/optical front-end models, clock-recovery loops, receiver-state modeling, or production-ready performance guarantees.

## Project Status

This repository is public as an early prototype. The current code is intentionally compact and focused on core timing-estimation behavior:

- Synthetic pulse-train generation
- Basic additive noise modeling
- Threshold-based time-of-arrival estimation
- Matched-filter time-of-arrival estimation
- Detection and timing-error metrics for estimator comparisons
- White-noise SNR sweeps with empirical detection, false-detection, and error summaries
- CRLB efficiency, detection tradeoff, and high-SNR error-floor diagnostic artifacts
- Unit tests for the simulator, estimators, metrics, sweeps, and characterization helpers

Future phases may expand the receiver model, add more realistic impairments, and introduce hardware-facing workflows.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Example

```python
import matplotlib.pyplot as plt

from compact_timing_receiver.estimators import (
    estimate_toa_matched_filter,
    estimate_toa_threshold,
)
from compact_timing_receiver.noise import add_white_noise
from compact_timing_receiver.pulse_train import generate_pulse_train

t, clean, true_arrivals = generate_pulse_train(
    sample_rate=50_000,
    duration=0.25,
    pulse_rate=80,
    pulse_width=0.001,
    amplitude=1.0,
    seed=1,
)

noisy = add_white_noise(clean, std=0.08, seed=2)

threshold_times = estimate_toa_threshold(
    t,
    noisy,
    threshold=0.5,
    refractory=0.006,
)
matched_times = estimate_toa_matched_filter(
    t,
    noisy,
    pulse_width=0.001,
)

plt.plot(t, noisy, label="noisy signal")
plt.vlines(true_arrivals, ymin=-0.2, ymax=1.2, color="black", alpha=0.25, label="true")
plt.vlines(threshold_times, ymin=-0.2, ymax=1.2, color="tab:orange", alpha=0.7, label="threshold")
plt.vlines(matched_times, ymin=-0.2, ymax=1.2, color="tab:green", alpha=0.7, label="matched filter")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
```

## Characterization Workflows

Run a compact off-grid AWGN SNR sweep:

```bash
python examples/snr_sweep_demo.py
```

Generate the current characterization artifacts:

```bash
python examples/crlb_characterization.py
```

This writes generated artifacts to `artifacts/` by default:

- `snr_sweep_characterization.csv`: SNR sweep metrics, confidence intervals, CRLB samples, and estimator efficiency
- `floor_diagnosis.md`: notes on SNR convention, detector thresholding, and high-SNR error-floor diagnostics

Compare sample-grid and parabolic matched-filter timing:

```bash
python examples/interpolation_characterization.py
```

This writes interpolation comparison artifacts to `artifacts/` by default.

Generate explanatory matched-filter visualizations:

```bash
python examples/matched_filter_visualizations.py
```

This writes:

- `matched_filter_peak_anatomy.png`: one-pulse correlation peak, sample-grid peak, parabolic vertex, and true center
- `matched_filter_fractional_offset_error_heatmap.png`: signed timing error vs fractional sample offset and SNR
- `matched_filter_crlb_efficiency.png`: RMSE/CRLB efficiency vs SNR with 1x and 2x reference lines
- `matched_filter_absolute_error_cdf.png`: absolute matched-error CDFs at selected SNRs
- `matched_filter_detection_tradeoff.png`: ROC-style detection/false-alarm tradeoff with the fixed threshold marked

## Tests

```bash
pytest
```
