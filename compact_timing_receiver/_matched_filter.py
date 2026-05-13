"""Shared Gaussian matched-filter helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import correlate, find_peaks


def pulse_template_samples(
    sample_rate: float,
    pulse_width: float,
    *,
    oversample: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    effective_rate = sample_rate * oversample
    sigma = pulse_width / 6.0
    half_samples = max(1, int(np.ceil(3.0 * sigma * effective_rate)))
    offsets = np.arange(-half_samples, half_samples + 1, dtype=float) / effective_rate
    template = np.exp(-0.5 * (offsets / sigma) ** 2)
    return offsets, template


def matched_filter_template(
    sample_rate: float,
    pulse_width: float,
) -> np.ndarray:
    _, template = pulse_template_samples(sample_rate, pulse_width)
    template = template.astype(float, copy=True)
    template -= np.mean(template)
    norm = np.linalg.norm(template)
    if norm == 0.0:
        raise ValueError("pulse_width is too small for the sampling interval")
    return template / norm


def matched_filter_response(
    signal: np.ndarray,
    sample_rate: float,
    pulse_width: float,
) -> np.ndarray:
    template = matched_filter_template(sample_rate, pulse_width)
    centered = np.asarray(signal, dtype=float) - np.median(signal)
    return correlate(centered, template, mode="same")


def find_matched_filter_peaks(
    response: np.ndarray,
    threshold: float,
    refractory: float,
    sample_rate: float,
) -> np.ndarray:
    distance = max(1, int(round(refractory * sample_rate)))
    peaks, _ = find_peaks(response, height=threshold, distance=distance)
    return peaks


def parabolic_peak_offset_samples(
    response: np.ndarray,
    peak: int,
    *,
    out_of_bounds: Literal["zero", "clip"] = "zero",
    use_flat_tolerance: bool = True,
) -> float:
    if peak == 0 or peak == response.size - 1:
        return 0.0

    y0 = float(response[peak - 1])
    y1 = float(response[peak])
    y2 = float(response[peak + 1])
    denominator = y0 - 2.0 * y1 + y2
    scale = max(1.0, abs(y0), abs(y1), abs(y2))
    if not np.isfinite(denominator) or denominator == 0.0:
        return 0.0
    if use_flat_tolerance and abs(denominator) <= np.finfo(float).eps * scale:
        return 0.0

    offset = 0.5 * (y0 - y2) / denominator
    if not np.isfinite(offset):
        return 0.0
    if out_of_bounds == "clip":
        return float(np.clip(offset, -1.0, 1.0))
    if abs(offset) > 1.0:
        return 0.0
    return float(offset)
