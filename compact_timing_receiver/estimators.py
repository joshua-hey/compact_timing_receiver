"""Pulse time-of-arrival estimators."""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, find_peaks


def _as_1d_float_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_time_signal(t: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time = _as_1d_float_array("t", t)
    samples = _as_1d_float_array("signal", signal)
    if time.size != samples.size:
        raise ValueError("t and signal must have the same length")
    if time.size < 2:
        raise ValueError("t and signal must contain at least two samples")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("t must be strictly increasing")
    return time, samples


def _sample_interval(t: np.ndarray) -> float:
    return float(np.median(np.diff(t)))


def estimate_toa_threshold(
    t: np.ndarray,
    signal: np.ndarray,
    threshold: float,
    refractory: float,
) -> np.ndarray:
    """Estimate arrival times from rising threshold crossings."""

    time, samples = _validate_time_signal(t, signal)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if refractory < 0.0 or not np.isfinite(refractory):
        raise ValueError("refractory must be finite and non-negative")

    crossings: list[float] = []
    last_detection = -np.inf

    below = samples[:-1] < threshold
    at_or_above = samples[1:] >= threshold
    indices = np.flatnonzero(below & at_or_above)

    for index in indices:
        y0 = samples[index]
        y1 = samples[index + 1]
        if y1 == y0:
            crossing_time = time[index]
        else:
            fraction = (threshold - y0) / (y1 - y0)
            crossing_time = time[index] + fraction * (time[index + 1] - time[index])

        if crossing_time - last_detection >= refractory:
            crossings.append(float(crossing_time))
            last_detection = crossing_time

    return np.asarray(crossings, dtype=float)


def estimate_toa_matched_filter(
    t: np.ndarray,
    signal: np.ndarray,
    pulse_width: float,
    threshold: float | None = None,
    refractory: float | None = None,
) -> np.ndarray:
    """Estimate arrival times using a Gaussian matched filter."""

    time, samples = _validate_time_signal(t, signal)
    if pulse_width <= 0.0 or not np.isfinite(pulse_width):
        raise ValueError("pulse_width must be finite and positive")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite when provided")
    if refractory is not None and (refractory < 0.0 or not np.isfinite(refractory)):
        raise ValueError("refractory must be finite and non-negative when provided")

    dt = _sample_interval(time)
    sigma = pulse_width / 6.0
    half_samples = max(1, int(np.ceil(3.0 * sigma / dt)))
    offsets = np.arange(-half_samples, half_samples + 1, dtype=float) * dt
    template = np.exp(-0.5 * (offsets / sigma) ** 2)
    template -= np.mean(template)
    norm = np.linalg.norm(template)
    if norm == 0.0:
        raise ValueError("pulse_width is too small for the sampling interval")
    template /= norm

    centered = samples - np.median(samples)
    filtered = correlate(centered, template, mode="same")

    if threshold is None:
        baseline = float(np.median(filtered))
        mad = float(np.median(np.abs(filtered - baseline)))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 0.0:
            peak_threshold = baseline + 5.0 * robust_sigma
        else:
            peak_threshold = baseline + 0.5 * (float(np.max(filtered)) - baseline)
    else:
        peak_threshold = threshold

    if refractory is None:
        refractory = 2.0 * pulse_width

    distance = max(1, int(round(refractory / dt)))
    peaks, _ = find_peaks(filtered, height=peak_threshold, distance=distance)
    return time[peaks].astype(float, copy=True)
