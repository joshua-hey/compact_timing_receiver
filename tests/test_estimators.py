import numpy as np

from compact_timing_receiver.estimators import (
    estimate_toa_matched_filter,
    estimate_toa_threshold,
)
from compact_timing_receiver.pulse_train import generate_pulse_train


def _nearest_errors(estimates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.array([np.min(np.abs(estimates - value)) for value in truth])


def test_threshold_estimator_recovers_clean_pulse_times() -> None:
    t, signal, true_arrivals = generate_pulse_train(
        sample_rate=100_000,
        duration=0.12,
        pulse_rate=50,
        pulse_width=0.0012,
        amplitude=1.0,
    )

    estimates = estimate_toa_threshold(
        t,
        signal,
        threshold=0.5,
        refractory=0.01,
    )

    assert estimates.size == true_arrivals.size
    assert np.max(_nearest_errors(estimates, true_arrivals)) < 0.001


def test_matched_filter_estimator_recovers_clean_pulse_times() -> None:
    t, signal, true_arrivals = generate_pulse_train(
        sample_rate=100_000,
        duration=0.12,
        pulse_rate=50,
        pulse_width=0.0012,
        amplitude=1.0,
    )

    estimates = estimate_toa_matched_filter(
        t,
        signal,
        pulse_width=0.0012,
        threshold=0.2,
        refractory=0.01,
    )

    assert estimates.size == true_arrivals.size
    assert np.max(_nearest_errors(estimates, true_arrivals)) <= 1.0 / 100_000
