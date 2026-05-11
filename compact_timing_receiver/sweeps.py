"""Deterministic SNR sweeps for Phase 1 timing-recovery experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from compact_timing_receiver.estimators import estimate_toa_matched_filter
from compact_timing_receiver.metrics import (
    compute_timing_errors,
    summarize_timing_errors,
)
from compact_timing_receiver.noise import add_white_noise
from compact_timing_receiver.pulse_train import generate_pulse_train


def _validate_snr_values(snr_db_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(list(snr_db_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("snr_db_values must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("snr_db_values must contain only finite values")
    return values


def run_white_noise_snr_sweep(
    snr_db_values: Sequence[float],
    *,
    trial_count: int = 1,
    base_seed: int = 0,
    sample_rate: float = 100_000,
    duration: float = 0.12,
    pulse_rate: float = 50,
    pulse_width: float = 0.0012,
    amplitude: float = 1.0,
    baseline: float = 0.0,
    estimator_threshold: float | None = 0.2,
    estimator_refractory: float | None = 0.01,
    match_tolerance: float | None = None,
) -> list[dict[str, Any]]:
    """Run matched-filter timing recovery across white-noise SNR levels."""

    snr_db_array = _validate_snr_values(snr_db_values)
    if trial_count < 1:
        raise ValueError("trial_count must be at least 1")
    if match_tolerance is None:
        match_tolerance = 3.0 / sample_rate

    results: list[dict[str, Any]] = []

    for snr_index, snr_db in enumerate(snr_db_array):
        rms_errors: list[float] = []
        bias_errors: list[float] = []
        missed_counts: list[int] = []
        extra_counts: list[int] = []
        true_pulse_counts: list[int] = []

        for trial_index in range(trial_count):
            trial_seed = base_seed + snr_index * trial_count + trial_index
            t, clean_signal, true_arrival_times = generate_pulse_train(
                sample_rate=sample_rate,
                duration=duration,
                pulse_rate=pulse_rate,
                pulse_width=pulse_width,
                amplitude=amplitude,
                baseline=baseline,
                seed=trial_seed,
            )
            signal_power = float(np.mean(clean_signal**2))
            noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
            noise_std = float(np.sqrt(noise_power))
            noisy_signal = add_white_noise(clean_signal, std=noise_std, seed=trial_seed)

            estimated_arrival_times = estimate_toa_matched_filter(
                t,
                noisy_signal,
                pulse_width=pulse_width,
                threshold=estimator_threshold,
                refractory=estimator_refractory,
            )
            timing_errors, missed_count, extra_count = compute_timing_errors(
                true_arrival_times,
                estimated_arrival_times,
                match_tolerance,
            )
            summary = summarize_timing_errors(timing_errors, missed_count, extra_count)

            rms_errors.append(float(summary["rms_error"]))
            bias_errors.append(float(summary["mean_error"]))
            missed_counts.append(int(summary["missed_count"]))
            extra_counts.append(int(summary["extra_count"]))
            true_pulse_counts.append(int(true_arrival_times.size))

        mean_rms_error = float(np.mean(rms_errors))
        sample_period = 1.0 / sample_rate

        results.append(
            {
                "estimator": "matched_filter",
                "snr_db": float(snr_db),
                "trial_count": int(trial_count),
                "total_trials": int(trial_count),
                "total_true_pulses": int(np.sum(true_pulse_counts)),
                "mean_bias_error": float(np.mean(bias_errors)),
                "mean_rms_error": mean_rms_error,
                "mean_rms_error_samples": float(mean_rms_error / sample_period),
                "mean_missed_count": float(np.mean(missed_counts)),
                "mean_extra_count": float(np.mean(extra_counts)),
                "max_rms_error": float(np.max(rms_errors)),
                "max_missed_count": int(np.max(missed_counts)),
                "max_extra_count": int(np.max(extra_counts)),
            }
        )

    return results


__all__ = ["run_white_noise_snr_sweep"]
