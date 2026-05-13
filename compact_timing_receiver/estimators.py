"""Pulse time-of-arrival estimators.

This module estimates pulse arrival times from sampled time-domain signals using
rising threshold crossings and Gaussian matched filtering.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from compact_timing_receiver._matched_filter import (
    find_matched_filter_peaks,
    matched_filter_response,
    parabolic_peak_offset_samples,
)

# Convert values to a finite one-dimensional float array.
def _as_1d_float_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array

# Validate matching time and signal arrays for sampled waveform analysis.
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

# Estimate the sample spacing from the time array.
def _sample_interval(t: np.ndarray) -> float:
    return float(np.median(np.diff(t)))


def _validate_uniform_sampling(t: np.ndarray) -> float:
    intervals = np.diff(t)
    dt = _sample_interval(t)
    absolute_tolerance = 16.0 * np.finfo(float).eps * max(
        1.0,
        abs(float(t[0])),
        abs(float(t[-1])),
    )
    if not np.allclose(intervals, dt, rtol=1e-6, atol=absolute_tolerance):
        raise ValueError("t must be uniformly sampled")
    return dt


# Detects when a signal crosses upward through a fixed threshold.
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

    below = samples[:-1] < threshold # array of booleans where signal is below threshold
    at_or_above = samples[1:] >= threshold # array of booleans where signal is at or above threshold
    indices = np.flatnonzero(below & at_or_above)  # left-side indices of upward threshold crossings

    for index in indices:
        y0 = samples[index]
        y1 = samples[index + 1]  # sample immediately after y0
        if y1 == y0:
            crossing_time = time[index]
        else:
            # Assuming the signal changes, we can linearly interpolate to find the crossing time more accurately.
            # fraction is the proportion of the way from y0 to y1 that the threshold is crossed.
            # crossing_time is the time at the crossing index plus the fraction of the time interval
            # to the next sample.
            fraction = (threshold - y0) / (y1 - y0)
            crossing_time = time[index] + fraction * (time[index + 1] - time[index])

        if crossing_time - last_detection >= refractory:
            crossings.append(float(crossing_time))
            last_detection = crossing_time

    return np.asarray(crossings, dtype=float)


# Detects pulses by correlating the signal with an expected pulse shape.
def estimate_toa_matched_filter(
    t: np.ndarray,
    signal: np.ndarray,
    pulse_width: float,
    threshold: float | None = None,
    refractory: float | None = None,
    interpolation: Literal["none", "parabolic"] = "none",
) -> np.ndarray:
    """Estimate arrival times using a Gaussian matched filter."""

    time, samples = _validate_time_signal(t, signal)
    if pulse_width <= 0.0 or not np.isfinite(pulse_width):
        raise ValueError("pulse_width must be finite and positive")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite when provided")
    if refractory is not None and (refractory < 0.0 or not np.isfinite(refractory)):
        raise ValueError("refractory must be finite and non-negative when provided")
    if interpolation not in {"none", "parabolic"}:
        raise ValueError('interpolation must be "none" or "parabolic"')

    dt = _validate_uniform_sampling(time)
    # Correlate the signal with the Gaussian matched-filter template.
    # Peaks in the filtered output occur where the signal best matches the expected pulse shape.
    filtered = matched_filter_response(samples, 1.0 / dt, pulse_width)

    # If threshold is not specified, calculate a robust threshold based on the median
    # and median absolute deviation of the filtered signal.
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

    # Refractory is the minimum allowed time between detected pulses to avoid multiple
    # detections of the same pulse. If unspecified, it defaults to twice the pulse width.
    if refractory is None:
        refractory = 2.0 * pulse_width

    # Find likely pulse centers by detecting peaks in the filtered signal that exceed 
    # the calculated threshold, ensuring that detected peaks are separated by at 
    # least the refractory period to avoid multiple detections of the same pulse.
    peaks = find_matched_filter_peaks(filtered, peak_threshold, refractory, 1.0 / dt)

    # Return array of times corresponding to the detected peaks.
    if interpolation == "none":
        return time[peaks].astype(float, copy=True)

    refined_times: list[float] = []
    for peak in peaks:
        delta = parabolic_peak_offset_samples(
            filtered,
            peak,
            out_of_bounds="zero",
            use_flat_tolerance=True,
        )
        refined_times.append(float(time[peak] + delta * dt))

    return np.asarray(refined_times, dtype=float)
