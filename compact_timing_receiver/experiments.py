"""Small Phase 1 timing-recovery experiment runner."""

from __future__ import annotations

from typing import Any

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


def run_timing_recovery_trial(
    *,
    sample_rate: float = 100_000,
    duration: float = 0.12,
    pulse_rate: float = 50,
    pulse_width: float = 0.0012,
    amplitude: float = 1.0,
    baseline: float = 0.0,
    seed: int | None = 0,
    white_noise_std: float = 0.0,
    baseline_drift_amplitude: float = 0.0,
    baseline_drift_frequency: float = 1.0,
    adc_bits: int | None = None,
    adc_v_min: float = -0.1,
    adc_v_max: float = 1.1,
    estimator_threshold: float | None = 0.2,
    estimator_refractory: float | None = 0.01,
    match_tolerance: float | None = None,
) -> dict[str, Any]:
    """Run one synthetic matched-filter timing-recovery trial."""

    t, signal, true_arrival_times = generate_pulse_train(
        sample_rate=sample_rate,
        duration=duration,
        pulse_rate=pulse_rate,
        pulse_width=pulse_width,
        amplitude=amplitude,
        baseline=baseline,
        seed=seed,
    )

    if white_noise_std > 0.0:
        signal = add_white_noise(signal, std=white_noise_std, seed=seed)
    if baseline_drift_amplitude != 0.0:
        signal = add_baseline_drift(
            t,
            signal,
            amplitude=baseline_drift_amplitude,
            frequency=baseline_drift_frequency,
        )
    if adc_bits is not None:
        signal = quantize_adc(signal, bits=adc_bits, v_min=adc_v_min, v_max=adc_v_max)

    estimated_arrival_times = estimate_toa_matched_filter(
        t,
        signal,
        pulse_width=pulse_width,
        threshold=estimator_threshold,
        refractory=estimator_refractory,
    )

    if match_tolerance is None:
        match_tolerance = 3.0 / sample_rate

    timing_errors, missed_count, extra_count = compute_timing_errors(
        true_arrival_times,
        estimated_arrival_times,
        match_tolerance,
    )
    summary = summarize_timing_errors(timing_errors, missed_count, extra_count)

    return {
        "t": t,
        "signal": signal,
        "true_arrival_times": true_arrival_times,
        "estimated_arrival_times": estimated_arrival_times,
        "timing_errors": timing_errors,
        "summary": summary,
    }


__all__ = ["run_timing_recovery_trial"]
