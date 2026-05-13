"""Characterization helpers for Phase 1 timing-recovery reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compact_timing_receiver._matched_filter import find_matched_filter_peaks
from compact_timing_receiver.crlb import (
    compute_rms_bandwidth_hz,
    estimate_matched_filter_times_diagnostic,
    estimate_post_correlation_snr_linear,
    matched_filter_response,
    resolution_cell_count,
)
from compact_timing_receiver.metrics import compute_timing_errors, summarize_timing_errors
from compact_timing_receiver.noise import add_white_noise
from compact_timing_receiver.pulse_train import generate_pulse_train
from compact_timing_receiver.sweeps import run_white_noise_snr_sweep


def write_sweep_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty")

    fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_crlb_overlay(rows: list[dict[str, Any]], path: str | Path) -> None:
    snr_db = np.asarray([row["snr_db"] for row in rows], dtype=float)
    snr_linear = 10.0 ** (snr_db / 10.0)
    rmse_samples = np.asarray([row["mean_rms_error_samples"] for row in rows], dtype=float)
    crlb_samples = np.asarray([row["sigma_crlb_samples"] for row in rows], dtype=float)
    order = np.argsort(snr_linear)
    snr_db = snr_db[order]
    snr_linear = snr_linear[order]
    rmse_samples = rmse_samples[order]
    crlb_samples = crlb_samples[order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(snr_linear, rmse_samples, marker="o", label="RMSE")
    ax.plot(snr_linear, crlb_samples, marker="s", label="CRLB")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(snr_linear)
    ax.set_xticklabels([f"{value:g}" for value in snr_db])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Timing error (samples)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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


def roc_at_snr(
    *,
    snr_db: float,
    trial_count: int,
    pulse_count: int,
    base_seed: int,
    sample_rate: float,
    pulse_rate: float,
    pulse_width: float,
    amplitude: float,
    refractory: float,
    off_grid: bool,
    threshold_count: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    duration = (pulse_count + 1) / pulse_rate
    sample_period = 1.0 / sample_rate
    trial_data = []
    max_response = 0.0
    cells_per_trial = 0

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
        response = matched_filter_response(noisy_signal, sample_rate, pulse_width)
        max_response = max(max_response, float(np.max(response)))
        cells_per_trial = resolution_cell_count(response.size, sample_rate, refractory)
        trial_data.append((t, response, true_arrival_times))

    thresholds = np.linspace(0.0, max_response, threshold_count)
    detection_rates: list[float] = []
    false_rates: list[float] = []

    for threshold in thresholds:
        detected = 0
        total_true = 0
        extra = 0
        for t, response, true_arrival_times in trial_data:
            peaks = find_matched_filter_peaks(response, threshold, refractory, sample_rate)
            estimates = t[peaks].astype(float, copy=True)
            timing_errors, missed_count, extra_count = compute_timing_errors(
                true_arrival_times,
                estimates,
                tolerance=3.0 * sample_period,
            )
            detected += timing_errors.size
            total_true += true_arrival_times.size
            extra += extra_count
        detection_rates.append(detected / total_true)
        false_rates.append(extra / (trial_count * cells_per_trial))

    return thresholds, np.asarray(detection_rates), np.asarray(false_rates), cells_per_trial


def plot_roc(
    false_rates: np.ndarray,
    detection_rates: np.ndarray,
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(false_rates, detection_rates, marker="o")
    ax.set_xlabel("False detections per resolution cell")
    ax.set_ylabel("Detection probability")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


__all__ = [
    "diagnostic_rmse_samples",
    "one_trial_snr_diagnostics",
    "plot_crlb_overlay",
    "plot_roc",
    "roc_at_snr",
    "write_sweep_csv",
]
