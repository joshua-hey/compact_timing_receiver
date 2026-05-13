"""Compare sample-grid and parabolic matched-filter timing estimates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_timing_receiver.characterization import write_sweep_csv
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


def _seed_for_snr(snr_index: int, snr_db: float, trial_count: int) -> int:
    if trial_count == PILOT_TRIAL_COUNT:
        return BASE_SEED + snr_index * PILOT_TRIAL_COUNT
    return BASE_SEED + int(round((snr_db + 100.0) * 10.0))


def _run_one_snr(
    snr_db: float,
    *,
    trial_count: int,
    base_seed: int,
    interpolation: str,
) -> dict[str, object]:
    return run_white_noise_snr_sweep(
        [snr_db],
        trial_count=trial_count,
        base_seed=base_seed,
        estimator_interpolation=interpolation,
        **_sweep_kwargs(),
    )[0]


def _comparison_rows() -> list[dict[str, float]]:
    pilot_rows = run_white_noise_snr_sweep(
        SNR_DB_VALUES,
        trial_count=PILOT_TRIAL_COUNT,
        base_seed=BASE_SEED,
        estimator_interpolation="none",
        **_sweep_kwargs(),
    )

    rows: list[dict[str, float]] = []
    for snr_index, pilot_row in enumerate(pilot_rows):
        snr_db = float(pilot_row["snr_db"])
        trial_count = (
            HIGH_RIGOR_TRIAL_COUNT
            if float(pilot_row["false_detections_per_100_pulses"]) < 1.0
            else PILOT_TRIAL_COUNT
        )
        base_seed = _seed_for_snr(snr_index, snr_db, trial_count)
        if trial_count == PILOT_TRIAL_COUNT:
            none_row = pilot_row
        else:
            none_row = _run_one_snr(
                snr_db,
                trial_count=trial_count,
                base_seed=base_seed,
                interpolation="none",
            )
        parabolic_row = _run_one_snr(
            snr_db,
            trial_count=trial_count,
            base_seed=base_seed,
            interpolation="parabolic",
        )

        rmse_none = float(none_row["mean_rms_error_samples"])
        rmse_parabolic = float(parabolic_row["mean_rms_error_samples"])
        improvement_factor = (
            float("inf") if rmse_parabolic == 0.0 else float(rmse_none / rmse_parabolic)
        )
        rows.append(
            {
                "snr_db": snr_db,
                "rmse_samples_none": rmse_none,
                "rmse_samples_parabolic": rmse_parabolic,
                "sigma_crlb_samples": float(none_row["sigma_crlb_samples"]),
                "efficiency_none": float(none_row["efficiency"]),
                "efficiency_parabolic": float(parabolic_row["efficiency"]),
                "improvement_factor": improvement_factor,
            }
        )

    return rows


def _plot(rows: list[dict[str, float]], path: Path) -> None:
    snr_db = np.asarray([row["snr_db"] for row in rows], dtype=float)
    rmse_none = np.asarray([row["rmse_samples_none"] for row in rows], dtype=float)
    rmse_parabolic = np.asarray(
        [row["rmse_samples_parabolic"] for row in rows],
        dtype=float,
    )
    crlb = np.asarray([row["sigma_crlb_samples"] for row in rows], dtype=float)
    order = np.argsort(snr_db)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(snr_db[order], rmse_none[order], marker="o", label="sample-grid RMSE")
    ax.plot(snr_db[order], rmse_parabolic[order], marker="o", label="parabolic RMSE")
    ax.plot(snr_db[order], crlb[order], marker="s", label="CRLB")
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Timing error (samples)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_report(rows: list[dict[str, float]], path: Path) -> str:
    high_snr = max(rows, key=lambda row: row["snr_db"])
    conclusion = (
        "Parabolic interpolation removes most of the observed high-SNR sample-grid floor "
        f"and measures {high_snr['efficiency_parabolic']:.2f}x the CRLB at "
        f"{high_snr['snr_db']:.1f} dB."
    )
    if high_snr["efficiency_parabolic"] > 3.0:
        conclusion = (
            "Parabolic interpolation reduces the observed high-SNR sample-grid floor, "
            f"but remains {high_snr['efficiency_parabolic']:.2f}x the CRLB at "
            f"{high_snr['snr_db']:.1f} dB."
        )

    lines = [
        "# Interpolation Diagnosis",
        "",
        f"Seed: `{BASE_SEED}`",
        "",
        "Estimator modes compared: `none` and `parabolic` matched-filter interpolation.",
        "",
        f"High-SNR point: `{high_snr['snr_db']:.1f}` dB input SNR",
        "",
        f"High-SNR RMSE before interpolation: `{high_snr['rmse_samples_none']:.6g}` samples",
        f"High-SNR RMSE after interpolation: `{high_snr['rmse_samples_parabolic']:.6g}` samples",
        f"High-SNR CRLB sigma: `{high_snr['sigma_crlb_samples']:.6g}` samples",
        f"High-SNR efficiency before interpolation: `{high_snr['efficiency_none']:.6g}`",
        f"High-SNR efficiency after interpolation: `{high_snr['efficiency_parabolic']:.6g}`",
        f"Improvement factor: `{high_snr['improvement_factor']:.6g}`",
        "",
        f"Conclusion: {conclusion}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return conclusion


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory relative to the repository root unless absolute.",
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


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = _resolve_output_dir(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "interpolation_characterization.csv"
    plot_path = output_dir / "interpolation_crlb_overlay.png"
    report_path = output_dir / "interpolation_diagnosis.md"

    rows = _comparison_rows()
    write_sweep_csv(rows, csv_path)
    _plot(rows, plot_path)
    conclusion = _write_report(rows, report_path)

    print("Interpolation characterization")
    print(f"seed: {BASE_SEED}")
    print("modes: none, parabolic")
    print(f"wrote: {_display_path(csv_path, repo_root)}")
    print(f"wrote: {_display_path(plot_path, repo_root)}")
    print(f"wrote: {_display_path(report_path, repo_root)}")
    print(f"conclusion: {conclusion}")


if __name__ == "__main__":
    main()
