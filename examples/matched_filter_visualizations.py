"""Generate explanatory matched-filter timing visualizations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_timing_receiver._matched_filter import (
    find_matched_filter_peaks,
    matched_filter_response,
    parabolic_peak_offset_samples,
)
from compact_timing_receiver.crlb import resolution_cell_count
from compact_timing_receiver.estimators import estimate_toa_matched_filter
from compact_timing_receiver.metrics import compute_timing_errors
from compact_timing_receiver.noise import add_white_noise
from compact_timing_receiver.pulse_train import generate_pulse_train
from compact_timing_receiver.sweeps import run_white_noise_snr_sweep


SAMPLE_RATE = 10_000.0
PULSE_RATE = 50.0
PULSE_WIDTH = 0.0012
AMPLITUDE = 1.0
PEAK_ANATOMY_SNR_DB = 60.0
THRESHOLD = 0.2
REFRACTORY = 0.01
BASE_SEED = 100
PEAK_ANATOMY_CLOCK_OFFSET_FRACTION = 0.37
PEAK_WINDOW_HALF_WIDTH_SAMPLES = 14
DEFAULT_HEATMAP_SNR_DB_VALUES = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
DEFAULT_HEATMAP_OFFSET_COUNT = 51
DEFAULT_HEATMAP_TRIAL_COUNT = 12
DEFAULT_EFFICIENCY_SNR_DB_VALUES = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
DEFAULT_EFFICIENCY_TRIAL_COUNT = 20
DEFAULT_EFFICIENCY_PULSE_COUNT = 100
DEFAULT_CDF_SNR_DB_VALUES = [30.0, 10.0, 0.0]
DEFAULT_CDF_TRIAL_COUNT = 30
DEFAULT_CDF_PULSE_COUNT = 100
DEFAULT_TRADEOFF_SNR_DB_VALUES = [0.0, 3.0, 6.0]
DEFAULT_TRADEOFF_TRIAL_COUNT = 20
DEFAULT_TRADEOFF_PULSE_COUNT = 100
DEFAULT_TRADEOFF_THRESHOLD_COUNT = 80


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--plot",
        choices=(
            "all",
            "peak-anatomy",
            "offset-heatmap",
            "crlb-efficiency",
            "error-cdf",
            "detection-tradeoff",
        ),
        default="all",
        help="Visualization to generate.",
    )
    parser.add_argument(
        "--heatmap-offset-count",
        type=int,
        default=DEFAULT_HEATMAP_OFFSET_COUNT,
        help="Number of fractional sample offsets in the heatmap.",
    )
    parser.add_argument(
        "--heatmap-trial-count",
        type=int,
        default=DEFAULT_HEATMAP_TRIAL_COUNT,
        help="Noise trials per SNR/offset heatmap cell.",
    )
    parser.add_argument(
        "--heatmap-snr-db",
        type=float,
        nargs="+",
        default=DEFAULT_HEATMAP_SNR_DB_VALUES,
        help="SNR values in dB for the heatmap y-axis.",
    )
    parser.add_argument(
        "--efficiency-snr-db",
        type=float,
        nargs="+",
        default=DEFAULT_EFFICIENCY_SNR_DB_VALUES,
        help="SNR values in dB for the CRLB efficiency plot.",
    )
    parser.add_argument(
        "--efficiency-trial-count",
        type=int,
        default=DEFAULT_EFFICIENCY_TRIAL_COUNT,
        help="Trials per SNR for the CRLB efficiency plot.",
    )
    parser.add_argument(
        "--efficiency-pulse-count",
        type=int,
        default=DEFAULT_EFFICIENCY_PULSE_COUNT,
        help="Pulses per trial for the CRLB efficiency plot.",
    )
    parser.add_argument(
        "--cdf-snr-db",
        type=float,
        nargs="+",
        default=DEFAULT_CDF_SNR_DB_VALUES,
        help="SNR values in dB for the absolute-error CDF plot.",
    )
    parser.add_argument(
        "--cdf-trial-count",
        type=int,
        default=DEFAULT_CDF_TRIAL_COUNT,
        help="Trials per SNR for the absolute-error CDF plot.",
    )
    parser.add_argument(
        "--cdf-pulse-count",
        type=int,
        default=DEFAULT_CDF_PULSE_COUNT,
        help="Pulses per trial for the absolute-error CDF plot.",
    )
    parser.add_argument(
        "--tradeoff-snr-db",
        type=float,
        nargs="+",
        default=DEFAULT_TRADEOFF_SNR_DB_VALUES,
        help="SNR values in dB for the detection tradeoff plot.",
    )
    parser.add_argument(
        "--tradeoff-trial-count",
        type=int,
        default=DEFAULT_TRADEOFF_TRIAL_COUNT,
        help="Trials per SNR for the detection tradeoff plot.",
    )
    parser.add_argument(
        "--tradeoff-pulse-count",
        type=int,
        default=DEFAULT_TRADEOFF_PULSE_COUNT,
        help="Pulses per trial for the detection tradeoff plot.",
    )
    parser.add_argument(
        "--tradeoff-threshold-count",
        type=int,
        default=DEFAULT_TRADEOFF_THRESHOLD_COUNT,
        help="Correlation-height thresholds sampled for the detection tradeoff plot.",
    )
    return parser.parse_args()


def _resolve_output_dir(repo_root: Path, output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    return repo_root / output_dir


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _parabola_values(response: np.ndarray, peak: int, offsets: np.ndarray) -> np.ndarray:
    y0 = float(response[peak - 1])
    y1 = float(response[peak])
    y2 = float(response[peak + 1])
    a = 0.5 * (y0 - 2.0 * y1 + y2)
    b = 0.5 * (y2 - y0)
    return a * offsets**2 + b * offsets + y1


def _one_pulse_response(
    *,
    offset_fraction: float,
    snr_db: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    sample_period = 1.0 / SAMPLE_RATE
    duration = 1.8 / PULSE_RATE

    t, clean_signal, true_arrival_times = generate_pulse_train(
        sample_rate=SAMPLE_RATE,
        duration=duration,
        pulse_rate=PULSE_RATE,
        pulse_width=PULSE_WIDTH,
        amplitude=AMPLITUDE,
        clock_offset=offset_fraction * sample_period,
        seed=seed,
    )
    if true_arrival_times.size != 1:
        raise RuntimeError("expected exactly one true pulse")

    signal_power = float(np.mean(clean_signal**2))
    noise_std = float(np.sqrt(signal_power / (10.0 ** (float(snr_db) / 10.0))))
    noisy_signal = add_white_noise(clean_signal, std=noise_std, seed=seed + 1)
    response = matched_filter_response(noisy_signal, SAMPLE_RATE, PULSE_WIDTH)
    peaks = find_matched_filter_peaks(response, THRESHOLD, REFRACTORY, SAMPLE_RATE)
    if peaks.size == 0:
        raise RuntimeError("matched-filter detector found no peaks")

    true_center = float(true_arrival_times[0])
    peak = int(peaks[np.argmin(np.abs(t[peaks] - true_center))])
    return t, response, true_center, peak


def _sample_grid_and_parabolic_errors(
    *,
    offset_fraction: float,
    snr_db: float,
    seed: int,
) -> tuple[float, float]:
    t, response, true_center, peak = _one_pulse_response(
        offset_fraction=offset_fraction,
        snr_db=snr_db,
        seed=seed,
    )
    sample_period = 1.0 / SAMPLE_RATE
    sample_grid_time = float(t[peak])
    parabolic_offset = parabolic_peak_offset_samples(
        response,
        peak,
        out_of_bounds="zero",
        use_flat_tolerance=True,
    )
    parabolic_time = sample_grid_time + parabolic_offset * sample_period
    return (
        float((sample_grid_time - true_center) * SAMPLE_RATE),
        float((parabolic_time - true_center) * SAMPLE_RATE),
    )


def _build_peak_anatomy() -> dict[str, object]:
    t, response, true_center, peak = _one_pulse_response(
        offset_fraction=PEAK_ANATOMY_CLOCK_OFFSET_FRACTION,
        snr_db=PEAK_ANATOMY_SNR_DB,
        seed=BASE_SEED,
    )
    sample_period = 1.0 / SAMPLE_RATE
    parabolic_offset = parabolic_peak_offset_samples(
        response,
        peak,
        out_of_bounds="zero",
        use_flat_tolerance=True,
    )
    sample_grid_peak_time = float(t[peak])
    parabolic_vertex_time = float(sample_grid_peak_time + parabolic_offset * sample_period)

    return {
        "t": t,
        "response": response,
        "true_center": true_center,
        "peak": peak,
        "sample_grid_peak_time": sample_grid_peak_time,
        "parabolic_vertex_time": parabolic_vertex_time,
        "parabolic_offset": float(parabolic_offset),
    }


def _plot_peak_anatomy(data: dict[str, object], path: Path) -> None:
    t = data["t"]
    response = data["response"]
    if not isinstance(t, np.ndarray) or not isinstance(response, np.ndarray):
        raise TypeError("peak anatomy data must contain NumPy arrays")

    peak = int(data["peak"])
    true_center = float(data["true_center"])
    sample_grid_peak_time = float(data["sample_grid_peak_time"])
    parabolic_vertex_time = float(data["parabolic_vertex_time"])

    start = max(0, peak - PEAK_WINDOW_HALF_WIDTH_SAMPLES)
    stop = min(response.size, peak + PEAK_WINDOW_HALF_WIDTH_SAMPLES + 1)
    time_window = t[start:stop]
    response_window = response[start:stop]
    x_samples = (time_window - true_center) * SAMPLE_RATE

    fit_offsets = np.linspace(-1.25, 1.25, 200)
    fit_times = sample_grid_peak_time + fit_offsets / SAMPLE_RATE
    fit_x_samples = (fit_times - true_center) * SAMPLE_RATE
    fit_response = _parabola_values(response, peak, fit_offsets)

    sample_grid_error_samples = (sample_grid_peak_time - true_center) * SAMPLE_RATE
    parabolic_error_samples = (parabolic_vertex_time - true_center) * SAMPLE_RATE

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        x_samples,
        response_window,
        color="tab:blue",
        linewidth=1.5,
        label="correlation response",
    )
    ax.scatter(
        x_samples,
        response_window,
        color="tab:blue",
        s=22,
        zorder=3,
        label="sampled response",
    )
    ax.plot(
        fit_x_samples,
        fit_response,
        color="tab:green",
        linestyle="--",
        linewidth=1.7,
        label="three-point parabola",
    )
    ax.axvline(
        0.0,
        color="black",
        linestyle="-",
        linewidth=1.2,
        label="true pulse center",
    )
    ax.axvline(
        sample_grid_error_samples,
        color="tab:orange",
        linestyle=":",
        linewidth=2.0,
        label="sample-grid peak",
    )
    ax.axvline(
        parabolic_error_samples,
        color="tab:green",
        linestyle="-.",
        linewidth=2.0,
        label="parabolic vertex",
    )
    ax.scatter(
        [sample_grid_error_samples],
        [response[peak]],
        color="tab:orange",
        edgecolor="white",
        linewidth=0.8,
        s=70,
        zorder=5,
    )
    ax.scatter(
        [parabolic_error_samples],
        [_parabola_values(response, peak, np.asarray([data["parabolic_offset"]]))[0]],
        color="tab:green",
        edgecolor="white",
        linewidth=0.8,
        s=70,
        zorder=5,
    )

    annotation = (
        f"sample-grid error: {sample_grid_error_samples:+.3f} samples\n"
        f"parabolic error: {parabolic_error_samples:+.3f} samples"
    )
    ax.text(
        0.03,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    ax.set_title(f"One-pulse matched-filter peak anatomy ({PEAK_ANATOMY_SNR_DB:g} dB input SNR)")
    ax.set_xlabel("Offset from true pulse center (samples)")
    ax.set_ylabel("Matched-filter correlation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _validate_heatmap_args(
    offset_count: int,
    trial_count: int,
    snr_db_values: list[float],
) -> np.ndarray:
    if offset_count < 2:
        raise ValueError("heatmap offset count must be at least 2")
    if trial_count < 1:
        raise ValueError("heatmap trial count must be at least 1")
    snr_db = np.asarray(snr_db_values, dtype=float)
    if snr_db.ndim != 1 or snr_db.size == 0:
        raise ValueError("heatmap SNR values must be a non-empty one-dimensional list")
    if not np.all(np.isfinite(snr_db)):
        raise ValueError("heatmap SNR values must be finite")
    return snr_db


def _build_fractional_offset_error_grid(
    *,
    offset_count: int,
    trial_count: int,
    snr_db_values: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snr_db = _validate_heatmap_args(offset_count, trial_count, snr_db_values)
    offsets = np.linspace(0.0, 1.0, offset_count, endpoint=False)
    sample_grid_errors = np.full((snr_db.size, offsets.size), np.nan, dtype=float)
    parabolic_errors = np.full_like(sample_grid_errors, np.nan)

    for snr_index, snr_value in enumerate(snr_db):
        for offset_index, offset_fraction in enumerate(offsets):
            sample_grid_trials: list[float] = []
            parabolic_trials: list[float] = []
            for trial_index in range(trial_count):
                seed = BASE_SEED + snr_index * 100_000 + offset_index * 1_000 + trial_index
                try:
                    sample_error, parabolic_error = _sample_grid_and_parabolic_errors(
                        offset_fraction=float(offset_fraction),
                        snr_db=float(snr_value),
                        seed=seed,
                    )
                except RuntimeError:
                    continue
                sample_grid_trials.append(sample_error)
                parabolic_trials.append(parabolic_error)

            if sample_grid_trials:
                sample_grid_errors[snr_index, offset_index] = float(
                    np.mean(sample_grid_trials)
                )
                parabolic_errors[snr_index, offset_index] = float(np.mean(parabolic_trials))

    return offsets, snr_db, sample_grid_errors, parabolic_errors


def _axis_edges(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        step = 1.0
        return np.asarray([values[0] - 0.5 * step, values[0] + 0.5 * step])
    midpoints = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def _plot_fractional_offset_error_heatmap(
    *,
    offsets: np.ndarray,
    snr_db: np.ndarray,
    sample_grid_errors: np.ndarray,
    parabolic_errors: np.ndarray,
    path: Path,
) -> None:
    offset_edges = np.linspace(0.0, 1.0, offsets.size + 1)
    snr_edges = _axis_edges(snr_db)
    combined = np.concatenate(
        [
            sample_grid_errors[np.isfinite(sample_grid_errors)],
            parabolic_errors[np.isfinite(parabolic_errors)],
        ]
    )
    max_abs_error = float(np.max(np.abs(combined))) if combined.size else 0.5
    max_abs_error = max(max_abs_error, 0.05)
    norm = TwoSlopeNorm(vmin=-max_abs_error, vcenter=0.0, vmax=max_abs_error)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="0.85")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    panels: list[tuple[str, np.ndarray]] = [
        ("Sample-grid peak", sample_grid_errors),
        ("Same peak + parabolic vertex", parabolic_errors),
    ]
    mesh = None
    for ax, (title, values) in zip(axes, panels):
        mesh = ax.pcolormesh(
            offset_edges,
            snr_edges,
            np.ma.masked_invalid(values),
            cmap=cmap,
            norm=norm,
            shading="flat",
        )
        ax.set_title(title)
        ax.set_xlabel("True fractional sample offset")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(False)

    axes[0].set_ylabel("Input SNR (dB)")
    axes[0].set_yticks(snr_db)
    fig.suptitle("Mean signed TOA error vs true fractional sample offset", y=0.98)
    if mesh is None:
        raise RuntimeError("heatmap mesh was not created")
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.02)
    colorbar.set_label("Mean signed TOA error (samples)")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.14, top=0.84, wspace=0.08)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _validate_snr_values(name: str, snr_db_values: list[float]) -> np.ndarray:
    snr_db = np.asarray(snr_db_values, dtype=float)
    if snr_db.ndim != 1 or snr_db.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional list")
    if not np.all(np.isfinite(snr_db)):
        raise ValueError(f"{name} must contain only finite values")
    return snr_db


def _validate_positive_count(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _sweep_kwargs(pulse_count: int) -> dict[str, float | int | bool]:
    return {
        "pulse_count": pulse_count,
        "sample_rate": SAMPLE_RATE,
        "pulse_rate": PULSE_RATE,
        "pulse_width": PULSE_WIDTH,
        "amplitude": AMPLITUDE,
        "off_grid": True,
        "estimator_threshold": THRESHOLD,
        "estimator_refractory": REFRACTORY,
    }


def _build_crlb_efficiency_curves(
    *,
    snr_db_values: list[float],
    trial_count: int,
    pulse_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    snr_db = _validate_snr_values("efficiency SNR values", snr_db_values)
    _validate_positive_count("efficiency trial count", trial_count)
    _validate_positive_count("efficiency pulse count", pulse_count)

    none_rows = run_white_noise_snr_sweep(
        snr_db.tolist(),
        trial_count=trial_count,
        base_seed=BASE_SEED,
        estimator_interpolation="none",
        **_sweep_kwargs(pulse_count),
    )
    parabolic_rows = run_white_noise_snr_sweep(
        snr_db.tolist(),
        trial_count=trial_count,
        base_seed=BASE_SEED,
        estimator_interpolation="parabolic",
        **_sweep_kwargs(pulse_count),
    )

    none_efficiency = np.asarray([row["efficiency"] for row in none_rows], dtype=float)
    parabolic_efficiency = np.asarray(
        [row["efficiency"] for row in parabolic_rows],
        dtype=float,
    )
    return snr_db, none_efficiency, parabolic_efficiency


def _plot_crlb_efficiency(
    *,
    snr_db: np.ndarray,
    none_efficiency: np.ndarray,
    parabolic_efficiency: np.ndarray,
    path: Path,
) -> None:
    order = np.argsort(snr_db)
    snr_db = snr_db[order]
    none_efficiency = none_efficiency[order]
    parabolic_efficiency = parabolic_efficiency[order]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for values, label, color, marker in (
        (none_efficiency, "sample-grid peak", "tab:orange", "o"),
        (parabolic_efficiency, "parabolic vertex", "tab:green", "s"),
    ):
        finite = np.isfinite(values) & (values > 0.0)
        if np.any(finite):
            ax.semilogy(
                snr_db[finite],
                values[finite],
                color=color,
                marker=marker,
                linewidth=1.8,
                label=label,
            )

    ax.axhline(1.0, color="black", linestyle="-", linewidth=1.0, label="1x CRLB")
    ax.axhline(2.0, color="0.35", linestyle="--", linewidth=1.0, label="2x CRLB")
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("Efficiency = RMSE / CRLB")
    ax.set_title("CRLB efficiency vs SNR")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _noisy_off_grid_trial(
    *,
    snr_db: float,
    pulse_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_period = 1.0 / SAMPLE_RATE
    duration = (pulse_count + 1) / PULSE_RATE
    clock_offset = float(np.random.default_rng(seed).uniform(0.0, sample_period))
    t, clean_signal, true_arrival_times = generate_pulse_train(
        sample_rate=SAMPLE_RATE,
        duration=duration,
        pulse_rate=PULSE_RATE,
        pulse_width=PULSE_WIDTH,
        amplitude=AMPLITUDE,
        clock_offset=clock_offset,
        seed=seed,
    )
    signal_power = float(np.mean(clean_signal**2))
    noise_std = float(np.sqrt(signal_power / (10.0 ** (float(snr_db) / 10.0))))
    noisy_signal = add_white_noise(clean_signal, std=noise_std, seed=seed + 1)
    return t, noisy_signal, true_arrival_times


def _build_absolute_error_cdf_data(
    *,
    snr_db_values: list[float],
    trial_count: int,
    pulse_count: int,
) -> dict[float, dict[str, np.ndarray]]:
    snr_db = _validate_snr_values("CDF SNR values", snr_db_values)
    _validate_positive_count("CDF trial count", trial_count)
    _validate_positive_count("CDF pulse count", pulse_count)
    tolerance = 3.0 / SAMPLE_RATE
    errors_by_snr: dict[float, dict[str, list[float]]] = {
        float(snr_value): {"none": [], "parabolic": []} for snr_value in snr_db
    }

    for snr_index, snr_value in enumerate(snr_db):
        for trial_index in range(trial_count):
            seed = BASE_SEED + 700_000 + snr_index * 100_000 + trial_index
            t, noisy_signal, true_arrival_times = _noisy_off_grid_trial(
                snr_db=float(snr_value),
                pulse_count=pulse_count,
                seed=seed,
            )
            for interpolation in ("none", "parabolic"):
                estimated_arrival_times = estimate_toa_matched_filter(
                    t,
                    noisy_signal,
                    pulse_width=PULSE_WIDTH,
                    threshold=THRESHOLD,
                    refractory=REFRACTORY,
                    interpolation=interpolation,
                )
                timing_errors, _, _ = compute_timing_errors(
                    true_arrival_times,
                    estimated_arrival_times,
                    tolerance,
                )
                errors_by_snr[float(snr_value)][interpolation].extend(
                    float(abs(error) * SAMPLE_RATE) for error in timing_errors
                )

    return {
        snr_value: {
            interpolation: np.asarray(values, dtype=float)
            for interpolation, values in values_by_mode.items()
        }
        for snr_value, values_by_mode in errors_by_snr.items()
    }


def _plot_absolute_error_cdf(
    *,
    errors_by_snr: dict[float, dict[str, np.ndarray]],
    path: Path,
) -> None:
    snr_values = list(errors_by_snr)
    fig, axes_array = plt.subplots(
        1,
        len(snr_values),
        figsize=(4.2 * len(snr_values), 4.4),
        sharey=True,
    )
    axes = np.atleast_1d(axes_array)
    all_values = [
        values
        for values_by_mode in errors_by_snr.values()
        for values in values_by_mode.values()
        if values.size > 0
    ]
    x_max = 1.0
    if all_values:
        x_max = max(1.0, float(np.max(np.concatenate(all_values))))

    for ax, snr_value in zip(axes, snr_values):
        for interpolation, label, color in (
            ("none", "sample-grid peak", "tab:orange"),
            ("parabolic", "parabolic vertex", "tab:green"),
        ):
            values = np.sort(errors_by_snr[snr_value][interpolation])
            if values.size == 0:
                continue
            probabilities = np.arange(1, values.size + 1, dtype=float) / values.size
            p95 = float(np.percentile(values, 95.0))
            ax.step(
                values,
                probabilities,
                where="post",
                color=color,
                linewidth=1.8,
                label=f"{label} (p95 {p95:.3g})",
            )

        ax.set_title(f"{snr_value:g} dB")
        ax.set_xlabel("|Timing error| (samples)")
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 1.01)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("Cumulative probability")
    fig.suptitle("Absolute matched timing-error CDF", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_detection_tradeoff_curves(
    *,
    snr_db_values: list[float],
    trial_count: int,
    pulse_count: int,
    threshold_count: int,
) -> list[dict[str, np.ndarray | float]]:
    snr_db = _validate_snr_values("tradeoff SNR values", snr_db_values)
    _validate_positive_count("tradeoff trial count", trial_count)
    _validate_positive_count("tradeoff pulse count", pulse_count)
    if threshold_count < 2:
        raise ValueError("tradeoff threshold count must be at least 2")

    curves: list[dict[str, np.ndarray | float]] = []
    tolerance = 3.0 / SAMPLE_RATE
    for snr_index, snr_value in enumerate(snr_db):
        trial_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        max_response = 0.0
        cells_per_trial = 1
        for trial_index in range(trial_count):
            seed = BASE_SEED + 900_000 + snr_index * 100_000 + trial_index
            t, noisy_signal, true_arrival_times = _noisy_off_grid_trial(
                snr_db=float(snr_value),
                pulse_count=pulse_count,
                seed=seed,
            )
            response = matched_filter_response(noisy_signal, SAMPLE_RATE, PULSE_WIDTH)
            max_response = max(max_response, float(np.max(response)))
            cells_per_trial = resolution_cell_count(response.size, SAMPLE_RATE, REFRACTORY)
            trial_data.append((t, response, true_arrival_times))

        thresholds = np.linspace(0.0, max_response, threshold_count)
        thresholds = np.unique(np.concatenate((thresholds, np.asarray([THRESHOLD]))))
        detection_rates: list[float] = []
        false_rates: list[float] = []
        total_resolution_cells = trial_count * cells_per_trial

        for threshold in thresholds:
            detected_count = 0
            total_true_count = 0
            extra_count = 0
            for t, response, true_arrival_times in trial_data:
                peaks = find_matched_filter_peaks(
                    response,
                    float(threshold),
                    REFRACTORY,
                    SAMPLE_RATE,
                )
                estimated_arrival_times = t[peaks].astype(float, copy=True)
                timing_errors, _, trial_extra_count = compute_timing_errors(
                    true_arrival_times,
                    estimated_arrival_times,
                    tolerance,
                )
                detected_count += int(timing_errors.size)
                total_true_count += int(true_arrival_times.size)
                extra_count += int(trial_extra_count)

            detection_rates.append(float(detected_count / total_true_count))
            false_rates.append(float(extra_count / total_resolution_cells))

        operating_index = int(np.flatnonzero(thresholds == THRESHOLD)[0])
        curves.append(
            {
                "snr_db": float(snr_value),
                "thresholds": thresholds,
                "detection_rates": np.asarray(detection_rates, dtype=float),
                "false_rates": np.asarray(false_rates, dtype=float),
                "operating_index": operating_index,
                "false_rate_floor": float(0.5 / total_resolution_cells),
            }
        )

    return curves


def _plot_detection_tradeoff(
    *,
    curves: list[dict[str, np.ndarray | float]],
    path: Path,
) -> None:
    positive_false_rates = [
        rate
        for curve in curves
        for rate in np.asarray(curve["false_rates"], dtype=float)
        if rate > 0.0
    ]
    floors = [float(curve["false_rate_floor"]) for curve in curves]
    false_rate_floor = min(positive_false_rates) / 2.0 if positive_false_rates else min(floors)
    false_rate_floor = max(false_rate_floor, min(floors))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(curves)))
    for curve, color in zip(curves, colors):
        snr_db = float(curve["snr_db"])
        false_rates = np.asarray(curve["false_rates"], dtype=float)
        detection_rates = np.asarray(curve["detection_rates"], dtype=float)
        plot_false_rates = np.maximum(false_rates, false_rate_floor)
        operating_index = int(curve["operating_index"])

        ax.plot(
            plot_false_rates,
            detection_rates,
            color=color,
            linewidth=1.8,
            label=f"{snr_db:g} dB",
        )
        ax.scatter(
            [plot_false_rates[operating_index]],
            [detection_rates[operating_index]],
            color=[color],
            edgecolor="black",
            linewidth=0.8,
            s=70,
            zorder=5,
        )

    ax.scatter(
        [],
        [],
        color="white",
        edgecolor="black",
        linewidth=0.8,
        s=70,
        label=f"threshold = {THRESHOLD:g}",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False detections per resolution cell")
    ax.set_ylabel("Detection probability")
    ax.set_title("Detection tradeoff with fixed operating point")
    ax.grid(True, which="both", alpha=0.3)
    if any(
        np.any(np.asarray(curve["false_rates"], dtype=float) == 0.0)
        for curve in curves
    ):
        ax.text(
            0.02,
            0.04,
            "zero false rates are clipped to the plotting floor",
            transform=ax.transAxes,
            fontsize=9,
            color="0.35",
        )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_peak_anatomy(output_dir: Path) -> tuple[Path, dict[str, object]]:
    path = output_dir / "matched_filter_peak_anatomy.png"
    data = _build_peak_anatomy()
    _plot_peak_anatomy(data, path)
    return path, data


def _write_fractional_offset_heatmap(
    output_dir: Path,
    *,
    offset_count: int,
    trial_count: int,
    snr_db_values: list[float],
) -> Path:
    path = output_dir / "matched_filter_fractional_offset_error_heatmap.png"
    offsets, snr_db, sample_grid_errors, parabolic_errors = (
        _build_fractional_offset_error_grid(
            offset_count=offset_count,
            trial_count=trial_count,
            snr_db_values=snr_db_values,
        )
    )
    _plot_fractional_offset_error_heatmap(
        offsets=offsets,
        snr_db=snr_db,
        sample_grid_errors=sample_grid_errors,
        parabolic_errors=parabolic_errors,
        path=path,
    )
    return path


def _write_crlb_efficiency(
    output_dir: Path,
    *,
    snr_db_values: list[float],
    trial_count: int,
    pulse_count: int,
) -> Path:
    path = output_dir / "matched_filter_crlb_efficiency.png"
    snr_db, none_efficiency, parabolic_efficiency = _build_crlb_efficiency_curves(
        snr_db_values=snr_db_values,
        trial_count=trial_count,
        pulse_count=pulse_count,
    )
    _plot_crlb_efficiency(
        snr_db=snr_db,
        none_efficiency=none_efficiency,
        parabolic_efficiency=parabolic_efficiency,
        path=path,
    )
    return path


def _write_absolute_error_cdf(
    output_dir: Path,
    *,
    snr_db_values: list[float],
    trial_count: int,
    pulse_count: int,
) -> Path:
    path = output_dir / "matched_filter_absolute_error_cdf.png"
    errors_by_snr = _build_absolute_error_cdf_data(
        snr_db_values=snr_db_values,
        trial_count=trial_count,
        pulse_count=pulse_count,
    )
    _plot_absolute_error_cdf(errors_by_snr=errors_by_snr, path=path)
    return path


def _write_detection_tradeoff(
    output_dir: Path,
    *,
    snr_db_values: list[float],
    trial_count: int,
    pulse_count: int,
    threshold_count: int,
) -> Path:
    path = output_dir / "matched_filter_detection_tradeoff.png"
    curves = _build_detection_tradeoff_curves(
        snr_db_values=snr_db_values,
        trial_count=trial_count,
        pulse_count=pulse_count,
        threshold_count=threshold_count,
    )
    _plot_detection_tradeoff(curves=curves, path=path)
    return path


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = _resolve_output_dir(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_choice: Literal[
        "all",
        "peak-anatomy",
        "offset-heatmap",
        "crlb-efficiency",
        "error-cdf",
        "detection-tradeoff",
    ] = args.plot

    print("Matched-filter visualizations")
    print(f"seed: {BASE_SEED}")

    if plot_choice in {"all", "peak-anatomy"}:
        peak_path, data = _write_peak_anatomy(output_dir)
        true_center = float(data["true_center"])
        sample_grid_peak_time = float(data["sample_grid_peak_time"])
        parabolic_vertex_time = float(data["parabolic_vertex_time"])
        print(f"peak_anatomy_snr_db: {PEAK_ANATOMY_SNR_DB:g}")
        print(
            "peak_anatomy_sample_grid_error_samples: "
            f"{(sample_grid_peak_time - true_center) * SAMPLE_RATE:+.6f}"
        )
        print(
            "peak_anatomy_parabolic_error_samples: "
            f"{(parabolic_vertex_time - true_center) * SAMPLE_RATE:+.6f}"
        )
        print(f"wrote: {_display_path(peak_path, repo_root)}")

    if plot_choice in {"all", "offset-heatmap"}:
        heatmap_path = _write_fractional_offset_heatmap(
            output_dir,
            offset_count=args.heatmap_offset_count,
            trial_count=args.heatmap_trial_count,
            snr_db_values=args.heatmap_snr_db,
        )
        print(f"heatmap_offset_count: {args.heatmap_offset_count}")
        print(f"heatmap_trial_count: {args.heatmap_trial_count}")
        print(f"heatmap_snr_db: {', '.join(f'{value:g}' for value in args.heatmap_snr_db)}")
        print(f"wrote: {_display_path(heatmap_path, repo_root)}")

    if plot_choice in {"all", "crlb-efficiency"}:
        efficiency_path = _write_crlb_efficiency(
            output_dir,
            snr_db_values=args.efficiency_snr_db,
            trial_count=args.efficiency_trial_count,
            pulse_count=args.efficiency_pulse_count,
        )
        print(f"efficiency_trial_count: {args.efficiency_trial_count}")
        print(f"efficiency_pulse_count: {args.efficiency_pulse_count}")
        print(
            "efficiency_snr_db: "
            f"{', '.join(f'{value:g}' for value in args.efficiency_snr_db)}"
        )
        print(f"wrote: {_display_path(efficiency_path, repo_root)}")

    if plot_choice in {"all", "error-cdf"}:
        cdf_path = _write_absolute_error_cdf(
            output_dir,
            snr_db_values=args.cdf_snr_db,
            trial_count=args.cdf_trial_count,
            pulse_count=args.cdf_pulse_count,
        )
        print(f"cdf_trial_count: {args.cdf_trial_count}")
        print(f"cdf_pulse_count: {args.cdf_pulse_count}")
        print(f"cdf_snr_db: {', '.join(f'{value:g}' for value in args.cdf_snr_db)}")
        print(f"wrote: {_display_path(cdf_path, repo_root)}")

    if plot_choice in {"all", "detection-tradeoff"}:
        tradeoff_path = _write_detection_tradeoff(
            output_dir,
            snr_db_values=args.tradeoff_snr_db,
            trial_count=args.tradeoff_trial_count,
            pulse_count=args.tradeoff_pulse_count,
            threshold_count=args.tradeoff_threshold_count,
        )
        print(f"tradeoff_trial_count: {args.tradeoff_trial_count}")
        print(f"tradeoff_pulse_count: {args.tradeoff_pulse_count}")
        print(f"tradeoff_threshold_count: {args.tradeoff_threshold_count}")
        print(
            "tradeoff_snr_db: "
            f"{', '.join(f'{value:g}' for value in args.tradeoff_snr_db)}"
        )
        print(f"wrote: {_display_path(tradeoff_path, repo_root)}")


if __name__ == "__main__":
    main()
