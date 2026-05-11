import numpy as np
import pytest

from compact_timing_receiver.estimators import (
    estimate_toa_matched_filter,
    estimate_toa_threshold,
)
from compact_timing_receiver.pulse_train import generate_pulse_train


def _nearest_errors(estimates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.array([np.min(np.abs(estimates - value)) for value in truth])


def test_threshold_estimator_recovers_clean_pulse_times() -> None:
    sample_rate = 100_000
    pulse_width = 0.0012
    threshold = 0.5
    t, signal, true_arrivals = generate_pulse_train(
        sample_rate=sample_rate,
        duration=0.12,
        pulse_rate=50,
        pulse_width=pulse_width,
        amplitude=1.0,
    )

    estimates = estimate_toa_threshold(
        t,
        signal,
        threshold=threshold,
        refractory=0.01,
    )

    sigma = pulse_width / 6.0
    expected_crossings = true_arrivals - sigma * np.sqrt(-2.0 * np.log(threshold))

    assert estimates.size == true_arrivals.size
    assert np.max(np.abs(estimates - expected_crossings)) <= 1.0 / sample_rate


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


def test_matched_filter_estimator_recovers_clean_pulse_times_with_auto_threshold() -> None:
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
        threshold=None,
        refractory=0.01,
    )

    assert estimates.size == true_arrivals.size
    assert np.max(_nearest_errors(estimates, true_arrivals)) <= 1.0 / 100_000


def test_threshold_estimator_validates_parameters() -> None:
    t = np.array([0.0, 0.001])
    signal = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        estimate_toa_threshold(t, signal, threshold=float("nan"), refractory=0.001)

    with pytest.raises(ValueError):
        estimate_toa_threshold(t, signal, threshold=0.5, refractory=-0.001)


def test_matched_filter_estimator_validates_parameters() -> None:
    t = np.array([0.0, 0.001, 0.002])
    signal = np.array([0.0, 1.0, 0.0])

    with pytest.raises(ValueError):
        estimate_toa_matched_filter(t, signal, pulse_width=0.0)

    with pytest.raises(ValueError):
        estimate_toa_matched_filter(t, signal, pulse_width=0.001, threshold=float("nan"))

    with pytest.raises(ValueError):
        estimate_toa_matched_filter(t, signal, pulse_width=0.001, refractory=-0.001)
