"""Synthetic Gaussian pulse-train generation."""

from __future__ import annotations

import numpy as np


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_generate_args(
    sample_rate: float,
    duration: float,
    pulse_rate: float,
    pulse_width: float,
    amplitude: float,
    baseline: float,
    clock_offset: float,
    clock_drift: float,
    jitter: float,
    dropout: float,
) -> None:
    for name, value in (
        ("sample_rate", sample_rate),
        ("duration", duration),
        ("pulse_rate", pulse_rate),
        ("pulse_width", pulse_width),
        ("amplitude", amplitude),
        ("baseline", baseline),
        ("clock_offset", clock_offset),
        ("clock_drift", clock_drift),
        ("jitter", jitter),
        ("dropout", dropout),
    ):
        _require_finite(name, value)

    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if pulse_rate <= 0.0:
        raise ValueError("pulse_rate must be positive")
    if pulse_width <= 0.0:
        raise ValueError("pulse_width must be positive")
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("dropout must be between 0 and 1")


def generate_pulse_train(
    sample_rate: float,
    duration: float,
    pulse_rate: float,
    pulse_width: float,
    amplitude: float = 1.0,
    baseline: float = 0.0,
    clock_offset: float = 0.0,
    clock_drift: float = 0.0,
    jitter: float = 0.0,
    dropout: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic Gaussian pulse train.

    ``pulse_width`` is treated as the approximate full temporal width of each
    pulse, spanning about six Gaussian standard deviations.
    """

    _validate_generate_args(
        sample_rate,
        duration,
        pulse_rate,
        pulse_width,
        amplitude,
        baseline,
        clock_offset,
        clock_drift,
        jitter,
        dropout,
    )

    sample_count = int(np.floor(sample_rate * duration))
    if sample_count < 2:
        raise ValueError("sample_rate and duration must produce at least two samples")

    t = np.arange(sample_count, dtype=float) / sample_rate
    signal = np.full(sample_count, baseline, dtype=float)
    rng = np.random.default_rng(seed)

    period = 1.0 / pulse_rate
    nominal_arrivals = np.arange(period, duration, period, dtype=float)
    drifted_arrivals = nominal_arrivals * (1.0 + clock_drift)
    arrival_times = drifted_arrivals + clock_offset

    if jitter > 0.0:
        arrival_times = arrival_times + rng.normal(0.0, jitter, size=arrival_times.shape)
    if dropout > 0.0:
        keep = rng.random(arrival_times.shape) >= dropout
        arrival_times = arrival_times[keep]

    arrival_times = arrival_times[(arrival_times >= 0.0) & (arrival_times < duration)]
    arrival_times.sort()

    sigma = pulse_width / 6.0
    for arrival_time in arrival_times:
        signal += amplitude * np.exp(-0.5 * ((t - arrival_time) / sigma) ** 2)

    return t, signal, arrival_times.astype(float, copy=True)
