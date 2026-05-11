"""CRLB and matched-filter diagnostics for synthetic timing sweeps."""

from __future__ import annotations

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


def compute_rms_bandwidth_hz(
    sample_rate: float,
    pulse_width: float,
    *,
    oversample: int = 1,
) -> float:
    _, template = pulse_template_samples(
        sample_rate,
        pulse_width,
        oversample=oversample,
    )
    spectrum = np.fft.fft(template)
    frequencies = np.fft.fftfreq(template.size, d=1.0 / (sample_rate * oversample))
    power = np.abs(spectrum) ** 2
    return float(np.sqrt(np.sum((frequencies**2) * power) / np.sum(power)))


def sigma_crlb_seconds(beta_rms_hz: float, snr_linear: float) -> float:
    if beta_rms_hz <= 0.0 or snr_linear <= 0.0:
        return float("nan")
    variance = 1.0 / (8.0 * np.pi**2 * beta_rms_hz**2 * snr_linear)
    return float(np.sqrt(variance))


def matched_filter_template(
    sample_rate: float,
    pulse_width: float,
    *,
    template_oversample: int = 1,
) -> np.ndarray:
    _, oversampled = pulse_template_samples(
        sample_rate,
        pulse_width,
        oversample=template_oversample,
    )
    if template_oversample > 1:
        center = oversampled.size // 2
        half_samples = max(1, int(np.ceil(0.5 * (oversampled.size - 1) / template_oversample)))
        indices = center + np.arange(-half_samples, half_samples + 1) * template_oversample
        indices = indices[(indices >= 0) & (indices < oversampled.size)]
        template = oversampled[indices]
    else:
        template = oversampled

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
    *,
    template_oversample: int = 1,
) -> np.ndarray:
    template = matched_filter_template(
        sample_rate,
        pulse_width,
        template_oversample=template_oversample,
    )
    centered = np.asarray(signal, dtype=float) - np.median(signal)
    return correlate(centered, template, mode="same")


def estimate_post_correlation_snr_linear(
    clean_signal: np.ndarray,
    true_arrival_times: np.ndarray,
    sample_rate: float,
    pulse_width: float,
    noise_std: float,
) -> float:
    if noise_std <= 0.0:
        return float("inf")

    response = matched_filter_response(clean_signal, sample_rate, pulse_width)
    peak_indices = np.rint(true_arrival_times * sample_rate).astype(int)
    peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < response.size)]
    if peak_indices.size == 0:
        return float("nan")

    peak_power = float(np.mean(response[peak_indices] ** 2))
    return peak_power / (noise_std**2)


def estimate_matched_filter_times_diagnostic(
    t: np.ndarray,
    signal: np.ndarray,
    pulse_width: float,
    *,
    threshold: float,
    refractory: float,
    interpolation_factor: int = 1,
    template_oversample: int = 1,
) -> np.ndarray:
    sample_rate = 1.0 / float(np.median(np.diff(t)))
    response = matched_filter_response(
        signal,
        sample_rate,
        pulse_width,
        template_oversample=template_oversample,
    )
    distance = max(1, int(round(refractory * sample_rate)))
    peaks, _ = find_peaks(response, height=threshold, distance=distance)

    if interpolation_factor <= 1:
        return t[peaks].astype(float, copy=True)

    dt = 1.0 / sample_rate
    refined: list[float] = []
    for peak in peaks:
        if peak == 0 or peak == response.size - 1:
            refined.append(float(t[peak]))
            continue

        y0 = response[peak - 1]
        y1 = response[peak]
        y2 = response[peak + 1]
        denominator = y0 - 2.0 * y1 + y2
        if denominator == 0.0:
            offset_samples = 0.0
        else:
            offset_samples = 0.5 * (y0 - y2) / denominator
            offset_samples = float(np.clip(offset_samples, -1.0, 1.0))
        offset_samples = round(offset_samples * interpolation_factor) / interpolation_factor
        refined.append(float(t[peak] + offset_samples * dt))

    return np.asarray(refined, dtype=float)


def resolution_cell_count(
    sample_count: int,
    sample_rate: float,
    refractory: float,
) -> int:
    distance = max(1, int(round(refractory * sample_rate)))
    return max(1, sample_count // distance)


__all__ = [
    "compute_rms_bandwidth_hz",
    "estimate_matched_filter_times_diagnostic",
    "estimate_post_correlation_snr_linear",
    "matched_filter_response",
    "matched_filter_template",
    "pulse_template_samples",
    "resolution_cell_count",
    "sigma_crlb_seconds",
]
