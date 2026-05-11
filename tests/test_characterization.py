import math

from compact_timing_receiver.characterization import (
    diagnostic_rmse_samples,
    roc_at_snr,
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


def test_roc_at_snr_returns_bounded_rates() -> None:
    thresholds, detection_rates, false_rates, cells_per_trial = roc_at_snr(
        snr_db=0.0,
        trial_count=1,
        pulse_count=3,
        base_seed=4,
        sample_rate=10_000,
        pulse_rate=20,
        pulse_width=0.0012,
        amplitude=1.0,
        refractory=0.01,
        off_grid=True,
        threshold_count=5,
    )

    assert thresholds.size == 5
    assert detection_rates.size == 5
    assert false_rates.size == 5
    assert cells_per_trial > 0
    assert (detection_rates >= 0.0).all()
    assert (detection_rates <= 1.0).all()
    assert (false_rates >= 0.0).all()
