from collections.abc import Callable

import numpy as np

from compact_timing_receiver.estimators import estimate_toa_matched_filter
from compact_timing_receiver.metrics import (
    compute_timing_errors,
    summarize_timing_errors,
)
from compact_timing_receiver.noise import (
    add_baseline_drift,
    add_white_noise,
    quantize_adc,
)
from compact_timing_receiver.pulse_train import generate_pulse_train


def _assert_matched_filter_recovers_mildly_impaired_pulses(
    impair: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    sample_rate = 100_000
    sample_period = 1.0 / sample_rate
    pulse_width = 0.0012
    t, clean_signal, true_arrivals = generate_pulse_train(
        sample_rate=sample_rate,
        duration=0.12,
        pulse_rate=50,
        pulse_width=pulse_width,
        amplitude=1.0,
    )
    impaired_signal = impair(t, clean_signal)

    estimated_arrivals = estimate_toa_matched_filter(
        t,
        impaired_signal,
        pulse_width=pulse_width,
        threshold=0.2,
        refractory=0.01,
    )
    errors, missed_count, extra_count = compute_timing_errors(
        true_arrivals,
        estimated_arrivals,
        tolerance=3.0 * sample_period,
    )
    summary = summarize_timing_errors(errors, missed_count, extra_count)

    assert missed_count == 0
    assert extra_count == 0
    assert summary["rms_error"] <= 3.0 * sample_period


def test_matched_filter_recovers_pulse_times_with_mild_white_noise() -> None:
    _assert_matched_filter_recovers_mildly_impaired_pulses(
        lambda _t, signal: add_white_noise(signal, std=0.02, seed=12),
    )


def test_matched_filter_recovers_pulse_times_with_mild_baseline_drift() -> None:
    _assert_matched_filter_recovers_mildly_impaired_pulses(
        lambda t, signal: add_baseline_drift(
            t,
            signal,
            amplitude=0.05,
            frequency=3.0,
        ),
    )


def test_matched_filter_recovers_pulse_times_with_mild_adc_quantization() -> None:
    _assert_matched_filter_recovers_mildly_impaired_pulses(
        lambda _t, signal: quantize_adc(signal, bits=10, v_min=-0.1, v_max=1.1),
    )
