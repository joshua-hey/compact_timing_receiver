from compact_timing_receiver.estimators import estimate_toa_matched_filter
from compact_timing_receiver.metrics import (
    compute_timing_errors,
    summarize_timing_errors,
)
from compact_timing_receiver.pulse_train import generate_pulse_train


def test_metrics_summarize_matched_filter_timing_errors() -> None:
    sample_rate = 100_000
    sample_period = 1.0 / sample_rate
    pulse_width = 0.0012

    t, signal, true_arrivals = generate_pulse_train(
        sample_rate=sample_rate,
        duration=0.12,
        pulse_rate=50,
        pulse_width=pulse_width,
        amplitude=1.0,
    )

    estimated_arrivals = estimate_toa_matched_filter(
        t,
        signal,
        pulse_width=pulse_width,
        threshold=0.2,
        refractory=0.01,
    )
    errors, missed_count, extra_count = compute_timing_errors(
        true_arrivals,
        estimated_arrivals,
        tolerance=sample_period,
    )
    summary = summarize_timing_errors(errors, missed_count, extra_count)

    assert missed_count == 0
    assert extra_count == 0
    assert summary["rms_error"] <= sample_period
