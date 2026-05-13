"""CRLB and matched-filter diagnostics for synthetic timing sweeps."""

from __future__ import annotations

import numpy as np

from compact_timing_receiver._matched_filter import (
    find_matched_filter_peaks,
    matched_filter_response as _matched_filter_response,
    matched_filter_template as _matched_filter_template,
    parabolic_peak_offset_samples,
    pulse_template_samples as _pulse_template_samples,
)


def pulse_template_samples(
    sample_rate: float,
    pulse_width: float,
    *,
    oversample: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    return _pulse_template_samples(sample_rate, pulse_width, oversample=oversample)


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
) -> np.ndarray:
    return _matched_filter_template(sample_rate, pulse_width)


def matched_filter_response(
    signal: np.ndarray,
    sample_rate: float,
    pulse_width: float,
) -> np.ndarray:
    return _matched_filter_response(signal, sample_rate, pulse_width)


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
) -> np.ndarray:
    sample_rate = 1.0 / float(np.median(np.diff(t)))
    response = matched_filter_response(signal, sample_rate, pulse_width)
    peaks = find_matched_filter_peaks(response, threshold, refractory, sample_rate)

    if interpolation_factor <= 1:
        return t[peaks].astype(float, copy=True)

    dt = 1.0 / sample_rate
    refined: list[float] = []
    for peak in peaks:
        offset_samples = parabolic_peak_offset_samples(
            response,
            peak,
            out_of_bounds="clip",
            use_flat_tolerance=False,
        )
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
