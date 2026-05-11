"""Generate CRLB overlay artifacts and floor diagnostics."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_timing_receiver.characterization import (
    diagnostic_rmse_samples,
    one_trial_snr_diagnostics,
    plot_crlb_overlay,
    plot_roc,
    roc_at_snr,
    write_sweep_csv,
)
from compact_timing_receiver.crlb import compute_rms_bandwidth_hz
from compact_timing_receiver.sweeps import run_white_noise_snr_sweep


SNR_DB_VALUES = [30, 25, 20, 15, 10, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4, 3, 2, 1, 0]
BASE_SEED = 100
PILOT_TRIAL_COUNT = 20
HIGH_RIGOR_TRIAL_COUNT = 200
PULSE_COUNT = 100
SAMPLE_RATE = 10_000.0
PULSE_RATE = 50.0
PULSE_WIDTH = 0.0012
AMPLITUDE = 1.0
THRESHOLD = 0.2
REFRACTORY = 0.01
OFF_GRID = True


def _sweep_kwargs() -> dict[str, float | int | bool]:
    return {
        "pulse_count": PULSE_COUNT,
        "sample_rate": SAMPLE_RATE,
        "pulse_rate": PULSE_RATE,
        "pulse_width": PULSE_WIDTH,
        "amplitude": AMPLITUDE,
        "off_grid": OFF_GRID,
        "estimator_threshold": THRESHOLD,
        "estimator_refractory": REFRACTORY,
    }


def _characterization_rows() -> list[dict[str, object]]:
    pilot_rows = run_white_noise_snr_sweep(
        SNR_DB_VALUES,
        trial_count=PILOT_TRIAL_COUNT,
        base_seed=BASE_SEED,
        **_sweep_kwargs(),
    )
    final_rows: list[dict[str, object]] = []

    for row in pilot_rows:
        snr_db = float(row["snr_db"])
        trial_count = (
            HIGH_RIGOR_TRIAL_COUNT
            if float(row["false_detections_per_100_pulses"]) < 1.0
            else PILOT_TRIAL_COUNT
        )
        if trial_count == PILOT_TRIAL_COUNT:
            final_rows.append(row)
            continue

        final_rows.extend(
            run_white_noise_snr_sweep(
                [snr_db],
                trial_count=trial_count,
                base_seed=BASE_SEED + int(round((snr_db + 100.0) * 10.0)),
                **_sweep_kwargs(),
            )
        )

    return final_rows


def _conclusion(
    baseline: float,
    interpolation_100x: float,
    oversampled_template: float,
) -> str:
    if math.isfinite(interpolation_100x) and interpolation_100x < 0.5 * baseline:
        return "The high-SNR floor is dominated by sample-grid peak picking."
    if math.isfinite(oversampled_template) and oversampled_template < 0.5 * baseline:
        return "The high-SNR floor is dominated by the discrete template representation."
    return "The high-SNR floor is not explained by the tested single-parameter diagnostics."


def _write_report(
    path: Path,
    *,
    beta_rms_hz: float,
    h1_baseline: float,
    h1_10x: float,
    h1_100x: float,
    h3_10x: float,
    snr_diagnostics: list[dict[str, float]],
    cells_per_trial: int,
) -> str:
    conclusion = _conclusion(h1_baseline, h1_100x, h3_10x)
    lines = [
        "# Floor Diagnosis",
        "",
        f"Seed: `{BASE_SEED}`",
        "",
        f"Beta RMS: `{beta_rms_hz:.6g}` Hz, computed numerically from the sampled Gaussian pulse template used by the simulator.",
        "",
        "SNR convention: the sweep input SNR is full-waveform average signal power divided by AWGN sample noise power. The CRLB overlay uses the estimated post-correlation peak SNR from the matched-filter response.",
        "",
        f"Detection threshold rule: fixed matched-filter correlation-height threshold `{THRESHOLD}` with `find_peaks` distance `{REFRACTORY}` seconds. This is not CFAR and is not a fixed-Pfa detector.",
        "",
        f"Search window length: `{int(round(((PULSE_COUNT + 1) / PULSE_RATE) * SAMPLE_RATE))}` samples. Resolution cells per trial: `{cells_per_trial}`.",
        "",
        "False detections are reported as an empirical extra-detection rate per resolution cell or per 100 true pulses, not as formal Pfa.",
        "",
        "## SNR Convention Check",
        "",
        "| input_snr_db | post_correlation_snr_db | processing_gain_db |",
        "| ---: | ---: | ---: |",
    ]
    for row in snr_diagnostics:
        lines.append(
            f"| {row['input_snr_db']:.1f} | {row['post_correlation_snr_db']:.2f} | {row['processing_gain_db']:.2f} |"
        )

    h1_conclusion = (
        "RMSE dropped with finer diagnostic interpolation."
        if math.isfinite(h1_100x) and h1_100x < 0.5 * h1_baseline
        else "RMSE did not materially drop with finer diagnostic interpolation."
    )
    h3_conclusion = (
        "RMSE dropped with a 10x oversampled diagnostic template."
        if math.isfinite(h3_10x) and h3_10x < 0.5 * h1_baseline
        else "RMSE did not materially drop with a 10x oversampled diagnostic template."
    )

    lines.extend(
        [
            "",
            "## Floor Tests",
            "",
            "| hypothesis | parameter changed | RMSE before samples | RMSE after samples | conclusion |",
            "| --- | --- | ---: | ---: | --- |",
            f"| H1 | diagnostic peak interpolation grid 10x | {h1_baseline:.4g} | {h1_10x:.4g} | {h1_conclusion} |",
            f"| H1 | diagnostic peak interpolation grid 100x | {h1_baseline:.4g} | {h1_100x:.4g} | {h1_conclusion} |",
            f"| H2 | SNR convention only | {h1_baseline:.4g} | {h1_baseline:.4g} | This changes CRLB scaling, not estimator RMSE. |",
            f"| H3 | diagnostic template oversample 10x | {h1_baseline:.4g} | {h3_10x:.4g} | {h3_conclusion} |",
            "",
            f"Conclusion: {conclusion}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return conclusion


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "snr_sweep_characterization.csv"
    crlb_plot_path = repo_root / "crlb_overlay.png"
    roc_plot_path = repo_root / "roc_0db.png"
    report_path = repo_root / "floor_diagnosis.md"

    print("CRLB characterization")
    print(f"seed: {BASE_SEED}")
    print(f"detector threshold rule: fixed matched-filter height threshold {THRESHOLD}")

    rows = _characterization_rows()
    beta_rms_hz = compute_rms_bandwidth_hz(SAMPLE_RATE, PULSE_WIDTH)
    write_sweep_csv(rows, csv_path)
    plot_crlb_overlay(rows, crlb_plot_path)

    snr_diagnostics = one_trial_snr_diagnostics(
        SNR_DB_VALUES,
        base_seed=BASE_SEED,
        sample_rate=SAMPLE_RATE,
        pulse_count=PULSE_COUNT,
        pulse_rate=PULSE_RATE,
        pulse_width=PULSE_WIDTH,
        amplitude=AMPLITUDE,
        threshold=THRESHOLD,
        refractory=REFRACTORY,
        off_grid=OFF_GRID,
    )

    print("input_snr_db  post_correlation_snr_db  processing_gain_db")
    for row in snr_diagnostics:
        print(
            f"{row['input_snr_db']:12.1f}  "
            f"{row['post_correlation_snr_db']:24.2f}  "
            f"{row['processing_gain_db']:18.2f}"
        )

    h1_baseline = diagnostic_rmse_samples(
        snr_db=30.0,
        trial_count=PILOT_TRIAL_COUNT,
        pulse_count=PULSE_COUNT,
        base_seed=BASE_SEED,
        sample_rate=SAMPLE_RATE,
        pulse_rate=PULSE_RATE,
        pulse_width=PULSE_WIDTH,
        amplitude=AMPLITUDE,
        threshold=THRESHOLD,
        refractory=REFRACTORY,
        off_grid=OFF_GRID,
    )
    h1_10x = diagnostic_rmse_samples(
        snr_db=30.0,
        trial_count=PILOT_TRIAL_COUNT,
        pulse_count=PULSE_COUNT,
        base_seed=BASE_SEED,
        sample_rate=SAMPLE_RATE,
        pulse_rate=PULSE_RATE,
        pulse_width=PULSE_WIDTH,
        amplitude=AMPLITUDE,
        threshold=THRESHOLD,
        refractory=REFRACTORY,
        off_grid=OFF_GRID,
        interpolation_factor=10,
    )
    h1_100x = diagnostic_rmse_samples(
        snr_db=30.0,
        trial_count=PILOT_TRIAL_COUNT,
        pulse_count=PULSE_COUNT,
        base_seed=BASE_SEED,
        sample_rate=SAMPLE_RATE,
        pulse_rate=PULSE_RATE,
        pulse_width=PULSE_WIDTH,
        amplitude=AMPLITUDE,
        threshold=THRESHOLD,
        refractory=REFRACTORY,
        off_grid=OFF_GRID,
        interpolation_factor=100,
    )
    h3_10x = diagnostic_rmse_samples(
        snr_db=30.0,
        trial_count=PILOT_TRIAL_COUNT,
        pulse_count=PULSE_COUNT,
        base_seed=BASE_SEED,
        sample_rate=SAMPLE_RATE,
        pulse_rate=PULSE_RATE,
        pulse_width=PULSE_WIDTH,
        amplitude=AMPLITUDE,
        threshold=THRESHOLD,
        refractory=REFRACTORY,
        off_grid=OFF_GRID,
        template_oversample=10,
    )

    _, detection_rates, false_rates, cells_per_trial = roc_at_snr(
        snr_db=0.0,
        trial_count=PILOT_TRIAL_COUNT,
        pulse_count=PULSE_COUNT,
        base_seed=BASE_SEED,
        sample_rate=SAMPLE_RATE,
        pulse_rate=PULSE_RATE,
        pulse_width=PULSE_WIDTH,
        amplitude=AMPLITUDE,
        refractory=REFRACTORY,
        off_grid=OFF_GRID,
    )
    plot_roc(false_rates, detection_rates, roc_plot_path)

    conclusion = _write_report(
        report_path,
        beta_rms_hz=beta_rms_hz,
        h1_baseline=h1_baseline,
        h1_10x=h1_10x,
        h1_100x=h1_100x,
        h3_10x=h3_10x,
        snr_diagnostics=snr_diagnostics,
        cells_per_trial=cells_per_trial,
    )

    print(f"beta_rms_hz: {beta_rms_hz:.6g}")
    print(f"wrote: {csv_path.name}")
    print(f"wrote: {crlb_plot_path.name}")
    print(f"wrote: {roc_plot_path.name}")
    print(f"wrote: {report_path.name}")
    print(f"conclusion: {conclusion}")


if __name__ == "__main__":
    main()
