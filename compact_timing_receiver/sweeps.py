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
    if values.size == 0:
        raise ValueError("snr_db_values must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("snr_db_values must contain only finite values")
    return values


def run_white_noise_snr_sweep(
    snr_db_values: Sequence[float],
    *,
    trial_count: int = 1,
    base_seed: int = 0,
    pulse_count: int = 20,
    sample_rate: float = 100_000,
    duration: float | None = None,
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
    if pulse_count < 1:
        raise ValueError("pulse_count must be at least 1")
    if duration is None:
        duration = (pulse_count + 1) / pulse_rate
    if match_tolerance is None:
        match_tolerance = 3.0 / sample_rate

    results: list[dict[str, Any]] = []
    sample_period = 1.0 / sample_rate

    for snr_index, snr_db in enumerate(snr_db_array):
        rms_errors: list[float] = []
        missed_counts: list[int] = []
        extra_counts: list[int] = []
        true_pulse_counts: list[int] = []
        estimated_pulse_counts: list[int] = []
        matched_errors: list[float] = []

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
            # SNR uses full-waveform average signal power, including any baseline.
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
            missed_counts.append(int(summary["missed_count"]))
            extra_counts.append(int(summary["extra_count"]))
            true_pulse_counts.append(int(true_arrival_times.size))
            estimated_pulse_counts.append(int(estimated_arrival_times.size))
            matched_errors.extend(float(error) for error in timing_errors)

        total_true_pulses = int(np.sum(true_pulse_counts))
        total_estimated_pulses = int(np.sum(estimated_pulse_counts))
        total_missed_count = int(np.sum(missed_counts))
        total_extra_count = int(np.sum(extra_counts))
        matched_error_array = np.asarray(matched_errors, dtype=float)
        if matched_error_array.size == 0:
            mean_rms_error = float("nan")
            mean_bias_error = float("nan")
            p95_abs_error = float("nan")
        else:
            mean_rms_error = float(np.sqrt(np.mean(matched_error_array**2)))
            mean_bias_error = float(np.mean(matched_error_array))
            p95_abs_error = float(np.percentile(np.abs(matched_error_array), 95.0))
        finite_rms_errors = np.asarray(rms_errors, dtype=float)
        finite_rms_errors = finite_rms_errors[np.isfinite(finite_rms_errors)]
        if finite_rms_errors.size == 0:
            max_rms_error = float("nan")
        else:
            max_rms_error = float(np.max(finite_rms_errors))

        results.append(
            {
                "estimator": "matched_filter",
                "snr_db": float(snr_db),
                "trial_count": int(trial_count),
                "pulse_count": int(pulse_count),
                "requested_pulse_count": int(pulse_count),
                "total_trials": int(trial_count),
                "total_true_pulses": total_true_pulses,
                "total_estimated_pulses": total_estimated_pulses,
                "total_missed_count": total_missed_count,
                "total_extra_count": total_extra_count,
                "detection_rate": float(
                    (total_true_pulses - total_missed_count) / total_true_pulses
                ),
                "missed_detection_rate": float(total_missed_count / total_true_pulses),
                "false_detections_per_trial": float(total_extra_count / trial_count),
                "false_detections_per_100_pulses": float(
                    100.0 * total_extra_count / total_true_pulses
                ),
                "mean_rms_error": mean_rms_error,
                "mean_rms_error_samples": float(mean_rms_error / sample_period),
                "max_rms_error": max_rms_error,
                "mean_bias_error": mean_bias_error,
                "p95_abs_error": p95_abs_error,
                "mean_missed_count": float(np.mean(missed_counts)),
                "mean_extra_count": float(np.mean(extra_counts)),
                "max_missed_count": int(np.max(missed_counts)),
                "max_extra_count": int(np.max(extra_counts)),
            }
        )

    return results


__all__ = ["run_white_noise_snr_sweep"]
