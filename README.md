# Compact Optical/RF Timing Receiver

Compact Optical/RF Timing Receiver is a Phase 1 prototype: a small Python toolkit for generating synthetic pulse trains, adding simple signal impairments, and estimating pulse arrival times with threshold and matched-filter methods.

The goal of this first phase is to provide a readable, testable foundation for timing-recovery experiments. It is intended for simulation, algorithm exploration, and early validation only. It does not yet include hardware integration, calibrated RF/optical front-end models, clock-recovery loops, receiver-state modeling, or production-ready performance guarantees.

## Project Status

This repository is public as an early prototype. The current code is intentionally compact and focused on core timing-estimation behavior:

- Synthetic pulse-train generation
- Basic additive noise modeling
- Threshold-based time-of-arrival estimation
- Matched-filter time-of-arrival estimation
- Unit tests for the Phase 1 estimator behavior

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

## Tests

```bash
pytest
```
