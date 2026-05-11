"""Timing-recovery evaluation metrics."""

from __future__ import annotations

import numpy as np


def _as_1d_float_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)
    if not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    return tolerance


def match_arrival_times(
    true_times: np.ndarray,
    estimated_times: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match true arrivals to estimated arrivals within ``tolerance``.

    Returns two integer arrays: matched true indices and matched estimate
    indices. Each true arrival and each estimate can appear at most once.
    """

    true_times = _as_1d_float_array("true_times", true_times)
    estimated_times = _as_1d_float_array("estimated_times", estimated_times)
    tolerance = _validate_tolerance(tolerance)

    candidates: list[tuple[float, int, int]] = []
    for true_index, true_time in enumerate(true_times):
        for estimate_index, estimated_time in enumerate(estimated_times):
            error = abs(estimated_time - true_time)
            if error <= tolerance:
                candidates.append((error, true_index, estimate_index))

    matched_true: list[int] = []
    matched_estimated: list[int] = []
    used_true: set[int] = set()
    used_estimated: set[int] = set()

    for _, true_index, estimate_index in sorted(candidates):
        if true_index in used_true or estimate_index in used_estimated:
            continue
        matched_true.append(true_index)
        matched_estimated.append(estimate_index)
        used_true.add(true_index)
        used_estimated.add(estimate_index)

    order = np.argsort(matched_true)
    return (
        np.asarray(matched_true, dtype=int)[order],
        np.asarray(matched_estimated, dtype=int)[order],
    )


def compute_timing_errors(
    true_times: np.ndarray,
    estimated_times: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, int, int]:
    """Return signed timing errors plus missed and extra detection counts.

    Timing errors follow the convention ``estimated_time - true_time``.
    """

    true_times = _as_1d_float_array("true_times", true_times)
    estimated_times = _as_1d_float_array("estimated_times", estimated_times)
    true_indices, estimated_indices = match_arrival_times(
        true_times,
        estimated_times,
        tolerance,
    )

    errors = estimated_times[estimated_indices] - true_times[true_indices]
    missed_count = true_times.size - true_indices.size
    extra_count = estimated_times.size - estimated_indices.size
    return errors.astype(float, copy=True), int(missed_count), int(extra_count)


def summarize_timing_errors(
    errors: np.ndarray,
    missed_count: int,
    extra_count: int,
) -> dict[str, float | int]:
    """Summarize matched timing errors and detection counts."""

    errors = _as_1d_float_array("errors", errors)

    if errors.size == 0:
        mean_error = float("nan")
        mean_absolute_error = float("nan")
        rms_error = float("nan")
        max_absolute_error = float("nan")
    else:
        mean_error = float(np.mean(errors))
        mean_absolute_error = float(np.mean(np.abs(errors)))
        rms_error = float(np.sqrt(np.mean(errors**2)))
        max_absolute_error = float(np.max(np.abs(errors)))

    return {
        "matched_count": int(errors.size),
        "missed_count": int(missed_count),
        "extra_count": int(extra_count),
        "mean_error": mean_error,
        "mean_absolute_error": mean_absolute_error,
        "rms_error": rms_error,
        "max_absolute_error": max_absolute_error,
    }


__all__ = [
    "compute_timing_errors",
    "match_arrival_times",
    "summarize_timing_errors",
]
