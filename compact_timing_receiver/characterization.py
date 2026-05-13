"""Characterization helpers for Phase 1 timing-recovery reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from compact_timing_receiver.crlb import (
    estimate_matched_filter_times_diagnostic,
    estimate_post_correlation_snr_linear,
)
from compact_timing_receiver.metrics import compute_timing_errors
from compact_timing_receiver.noise import add_white_noise
from compact_timing_receiver.pulse_train import generate_pulse_train


def write_sweep_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty")

    fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def one_trial_snr_diagnostics(
    snr_db_values: list[float],
    *,
    base_seed: int,
    sample_rate: float,
    pulse_count: int,
    pulse_rate: float,
    pulse_width: float,
    amplitude: float,
    threshold: float,
    refractory: float,
    off_grid: bool,
) -> list[dict[str, float]]:
    duration = (pulse_count + 1) / pulse_rate
    sample_period = 1.0 / sample_rate
    rows: list[dict[str, float]] = []

    for snr_index, snr_db in enumerate(snr_db_values):
        trial_seed = base_seed + snr_index * 1000
        if off_grid:
            clock_offset = float(np.random.default_rng(trial_seed).uniform(0.0, sample_period))
        else:
            clock_offset = 0.0
        t, clean_signal, true_arrival_times = generate_pulse_train(
            sample_rate=sample_rate,
            duration=duration,
            pulse_rate=pulse_rate,
            pulse_width=pulse_width,
            amplitude=amplitude,
            clock_offset=clock_offset,
            seed=trial_seed,
        )
        signal_power = float(np.mean(clean_signal**2))
        input_snr_linear = 10.0 ** (float(snr_db) / 10.0)
        noise_power = signal_power / input_snr_linear
        noise_std = float(np.sqrt(noise_power))
        output_snr_linear = estimate_post_correlation_snr_linear(
            clean_signal,
            true_arrival_times,
            sample_rate,
            pulse_width,
            noise_std,
        )
        rows.append(
            {
                "input_snr_db": float(snr_db),
                "post_correlation_snr_db": float(10.0 * np.log10(output_snr_linear)),
                "processing_gain_db": float(10.0 * np.log10(output_snr_linear) - snr_db),
            }
        )

    return rows


def diagnostic_rmse_samples(
    *,
    snr_db: float,
    trial_count: int,
    pulse_count: int,
    base_seed: int,
    sample_rate: float,
    pulse_rate: float,
    pulse_width: float,
    amplitude: float,
    threshold: float,
    refractory: float,
    off_grid: bool,
    interpolation_factor: int = 1,
) -> float:
    duration = (pulse_count + 1) / pulse_rate
    sample_period = 1.0 / sample_rate
    errors: list[float] = []

    for trial_index in range(trial_count):
        trial_seed = base_seed + trial_index
        if off_grid:
            clock_offset = float(np.random.default_rng(trial_seed).uniform(0.0, sample_period))
        else:
            clock_offset = 0.0
        t, clean_signal, true_arrival_times = generate_pulse_train(
            sample_rate=sample_rate,
            duration=duration,
            pulse_rate=pulse_rate,
            pulse_width=pulse_width,
            amplitude=amplitude,
            clock_offset=clock_offset,
            seed=trial_seed,
        )
        signal_power = float(np.mean(clean_signal**2))
        noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
        noise_std = float(np.sqrt(noise_power))
        noisy_signal = add_white_noise(clean_signal, std=noise_std, seed=trial_seed)
        estimates = estimate_matched_filter_times_diagnostic(
            t,
            noisy_signal,
            pulse_width,
            threshold=threshold,
            refractory=refractory,
            interpolation_factor=interpolation_factor,
        )
        timing_errors, _, _ = compute_timing_errors(
            true_arrival_times,
            estimates,
            tolerance=3.0 * sample_period,
        )
        errors.extend(float(error) for error in timing_errors)

    if not errors:
        return float("nan")
    error_array = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(error_array**2)) / sample_period)


__all__ = [
    "diagnostic_rmse_samples",
    "one_trial_snr_diagnostics",
    "write_sweep_csv",
]
