import math

from compact_timing_receiver.characterization import (
    diagnostic_rmse_samples,
    one_trial_snr_diagnostics,
)


def test_diagnostic_rmse_samples_is_finite_for_high_snr() -> None:
    rmse_samples = diagnostic_rmse_samples(
        snr_db=40.0,
        trial_count=1,
        pulse_count=3,
        base_seed=4,
        sample_rate=10_000,
        pulse_rate=20,
        pulse_width=0.0012,
        amplitude=1.0,
        threshold=0.2,
        refractory=0.01,
        off_grid=True,
        interpolation_factor=10,
    )

    assert math.isfinite(rmse_samples)
    assert rmse_samples >= 0.0


def test_one_trial_snr_diagnostics_reports_processing_gain() -> None:
    rows = one_trial_snr_diagnostics(
        [30.0, 20.0],
        base_seed=4,
        sample_rate=10_000,
        pulse_count=3,
        pulse_rate=20,
        pulse_width=0.0012,
        amplitude=1.0,
        threshold=0.2,
        refractory=0.01,
        off_grid=True,
    )

    assert len(rows) == 2
    for row in rows:
        assert math.isfinite(row["input_snr_db"])
        assert math.isfinite(row["post_correlation_snr_db"])
        assert math.isfinite(row["processing_gain_db"])
