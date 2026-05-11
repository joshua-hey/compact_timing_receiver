import math

import numpy as np

from compact_timing_receiver.crlb import (
    compute_rms_bandwidth_hz,
    estimate_matched_filter_times_diagnostic,
    estimate_post_correlation_snr_linear,
    sigma_crlb_seconds,
)
from compact_timing_receiver.pulse_train import generate_pulse_train


def test_compute_rms_bandwidth_is_positive() -> None:
    beta_rms_hz = compute_rms_bandwidth_hz(10_000, 0.0012)

    assert math.isfinite(beta_rms_hz)
    assert beta_rms_hz > 0.0


def test_crlb_sigma_decreases_with_snr() -> None:
    beta_rms_hz = compute_rms_bandwidth_hz(10_000, 0.0012)

    low_snr_sigma = sigma_crlb_seconds(beta_rms_hz, 10.0)
    high_snr_sigma = sigma_crlb_seconds(beta_rms_hz, 1000.0)

    assert high_snr_sigma < low_snr_sigma


def test_post_correlation_snr_is_finite_for_clean_pulses() -> None:
    _, clean_signal, true_arrival_times = generate_pulse_train(
        sample_rate=10_000,
        duration=0.3,
        pulse_rate=20,
        pulse_width=0.0012,
        seed=1,
    )

    snr_linear = estimate_post_correlation_snr_linear(
        clean_signal,
        true_arrival_times,
        sample_rate=10_000,
        pulse_width=0.0012,
        noise_std=0.1,
    )

    assert math.isfinite(snr_linear)
    assert snr_linear > 0.0


def test_diagnostic_matched_filter_estimator_returns_numpy_array() -> None:
    t, clean_signal, _ = generate_pulse_train(
        sample_rate=10_000,
        duration=0.3,
        pulse_rate=20,
        pulse_width=0.0012,
        seed=1,
    )

    estimates = estimate_matched_filter_times_diagnostic(
        t,
        clean_signal,
        pulse_width=0.0012,
        threshold=0.2,
        refractory=0.01,
        interpolation_factor=10,
    )

    assert isinstance(estimates, np.ndarray)
    assert estimates.size > 0
