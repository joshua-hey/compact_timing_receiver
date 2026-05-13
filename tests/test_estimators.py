import numpy as np
import pytest

from compact_timing_receiver.estimators import (
    estimate_toa_matched_filter,
    estimate_toa_threshold,
)
from compact_timing_receiver.noise import add_white_noise
from compact_timing_receiver.pulse_train import generate_pulse_train


def _nearest_errors(estimates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.array([np.min(np.abs(estimates - value)) for value in truth])


def _rms_error_samples(estimates: np.ndarray, truth: np.ndarray, sample_rate: float) -> float:
    return float(np.sqrt(np.mean(_nearest_errors(estimates, truth) ** 2)) * sample_rate)


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


def test_threshold_estimator_allows_nonuniform_time_axis() -> None:
    t = np.array([0.0, 0.001, 0.003])
    signal = np.array([0.0, 1.0, 0.0])

    estimates = estimate_toa_threshold(
        t,
        signal,
        threshold=0.5,
        refractory=0.0,
    )

    np.testing.assert_allclose(estimates, np.array([0.0005]))


def test_matched_filter_estimator_rejects_nonuniform_time_axis() -> None:
    t, signal, _ = generate_pulse_train(
        sample_rate=10_000,
        duration=0.12,
        pulse_rate=50,
        pulse_width=0.0012,
        amplitude=1.0,
    )
    nonuniform_t = t.copy()
    nonuniform_t[1::2] += 1e-7

    with pytest.raises(ValueError, match="uniformly sampled"):
        estimate_toa_matched_filter(
            nonuniform_t,
            signal,
            pulse_width=0.0012,
            threshold=0.2,
            refractory=0.01,
        )


def test_matched_filter_default_interpolation_stays_on_sample_grid() -> None:
    sample_rate = 10_000
    t, signal, _ = generate_pulse_train(
        sample_rate=sample_rate,
        duration=0.22,
        pulse_rate=50,
        pulse_width=0.0012,
        amplitude=1.0,
        clock_offset=0.37 / sample_rate,
        seed=21,
    )

    estimates = estimate_toa_matched_filter(
        t,
        signal,
        pulse_width=0.0012,
        threshold=0.2,
        refractory=0.01,
    )

    np.testing.assert_allclose(estimates * sample_rate, np.rint(estimates * sample_rate))


def test_matched_filter_interpolation_none_matches_default_behavior() -> None:
    t, signal, _ = generate_pulse_train(
        sample_rate=10_000,
        duration=0.22,
        pulse_rate=50,
        pulse_width=0.0012,
        amplitude=1.0,
        clock_offset=0.37 / 10_000,
        seed=21,
    )

    default_estimates = estimate_toa_matched_filter(
        t,
        signal,
        pulse_width=0.0012,
        threshold=0.2,
        refractory=0.01,
    )
    explicit_estimates = estimate_toa_matched_filter(
        t,
        signal,
        pulse_width=0.0012,
        threshold=0.2,
        refractory=0.01,
        interpolation="none",
    )

    np.testing.assert_array_equal(explicit_estimates, default_estimates)


def test_matched_filter_parabolic_interpolation_returns_finite_toas() -> None:
    t, signal, _ = generate_pulse_train(
        sample_rate=10_000,
        duration=0.22,
        pulse_rate=50,
        pulse_width=0.0012,
        amplitude=1.0,
        clock_offset=0.37 / 10_000,
        seed=21,
    )

    estimates = estimate_toa_matched_filter(
        t,
        signal,
        pulse_width=0.0012,
        threshold=0.2,
        refractory=0.01,
        interpolation="parabolic",
    )

    assert estimates.size > 0
    assert np.all(np.isfinite(estimates))


def test_matched_filter_parabolic_interpolation_preserves_time_origin() -> None:
    sample_rate = 10_000
    t, signal, _ = generate_pulse_train(
        sample_rate=sample_rate,
        duration=0.22,
        pulse_rate=50,
        pulse_width=0.0012,
        amplitude=1.0,
        clock_offset=0.37 / sample_rate,
        seed=21,
    )

    unshifted_estimates = estimate_toa_matched_filter(
        t,
        signal,
        pulse_width=0.0012,
        threshold=0.2,
        refractory=0.01,
        interpolation="parabolic",
    )
    time_origin = 1.25
    shifted_estimates = estimate_toa_matched_filter(
        t + time_origin,
        signal,
        pulse_width=0.0012,
        threshold=0.2,
        refractory=0.01,
        interpolation="parabolic",
    )

    np.testing.assert_allclose(
        shifted_estimates,
        unshifted_estimates + time_origin,
        rtol=0.0,
        atol=1e-7,
    )


def test_matched_filter_parabolic_interpolation_reduces_high_snr_off_grid_rmse() -> None:
    sample_rate = 10_000
    t, clean_signal, true_arrivals = generate_pulse_train(
        sample_rate=sample_rate,
        duration=0.22,
        pulse_rate=50,
        pulse_width=0.0012,
        amplitude=1.0,
        clock_offset=0.37 / sample_rate,
        seed=21,
    )
    signal_power = float(np.mean(clean_signal**2))
    noise_std = float(np.sqrt(signal_power / (10.0 ** (80.0 / 10.0))))
    noisy_signal = add_white_noise(clean_signal, std=noise_std, seed=22)

    sample_grid_estimates = estimate_toa_matched_filter(
        t,
        noisy_signal,
        pulse_width=0.0012,
        threshold=0.2,
        refractory=0.01,
        interpolation="none",
    )
    parabolic_estimates = estimate_toa_matched_filter(
        t,
        noisy_signal,
        pulse_width=0.0012,
        threshold=0.2,
        refractory=0.01,
        interpolation="parabolic",
    )

    assert sample_grid_estimates.size == true_arrivals.size
    assert parabolic_estimates.size == true_arrivals.size
    assert _rms_error_samples(parabolic_estimates, true_arrivals, sample_rate) < (
        _rms_error_samples(sample_grid_estimates, true_arrivals, sample_rate)
    )


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

    with pytest.raises(ValueError, match="interpolation"):
        estimate_toa_matched_filter(t, signal, pulse_width=0.001, interpolation="linear")
