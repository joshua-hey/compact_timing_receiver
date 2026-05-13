import math

import numpy as np
import pytest

import compact_timing_receiver.sweeps as sweeps
from compact_timing_receiver.sweeps import run_white_noise_snr_sweep


def test_white_noise_snr_sweep_returns_one_matched_filter_row_per_snr() -> None:
    results = run_white_noise_snr_sweep(
        [30.0, 40.0],
        trial_count=2,
        pulse_count=5,
        base_seed=5,
    )

    assert len(results) == 2
    assert [row["estimator"] for row in results] == ["matched_filter", "matched_filter"]
    assert [row["snr_db"] for row in results] == [30.0, 40.0]


def test_white_noise_snr_sweep_result_fields_are_present() -> None:
    result = run_white_noise_snr_sweep(
        [40.0],
        trial_count=1,
        pulse_count=5,
        base_seed=5,
    )[0]

    assert {
        "estimator",
        "snr_db",
        "trial_count",
        "pulse_count",
        "requested_pulse_count",
        "off_grid",
        "total_trials",
        "total_true_pulses",
        "total_estimated_pulses",
        "total_missed_count",
        "total_extra_count",
        "detection_rate",
        "detection_rate_ci_low",
        "detection_rate_ci_high",
        "missed_detection_rate",
        "missed_detection_rate_ci_low",
        "missed_detection_rate_ci_high",
        "false_detections_per_trial",
        "false_detections_per_100_pulses",
        "search_window_samples",
        "resolution_cells_per_trial",
        "false_detection_rate_per_resolution_cell",
        "false_detection_rate_ci_low",
        "false_detection_rate_ci_high",
        "mean_rms_error",
        "mean_rms_error_samples",
        "sigma_crlb_samples",
        "efficiency",
        "max_rms_error",
        "mean_bias_error",
        "mean_bias_error_s",
        "mean_bias_error_samples",
        "p95_abs_error",
        "p95_abs_error_s",
        "p95_abs_error_samples",
    } <= set(result)


def test_white_noise_snr_sweep_is_deterministic_for_same_base_seed() -> None:
    first = run_white_noise_snr_sweep(
        [30.0, 40.0],
        trial_count=2,
        pulse_count=5,
        base_seed=9,
    )
    second = run_white_noise_snr_sweep(
        [30.0, 40.0],
        trial_count=2,
        pulse_count=5,
        base_seed=9,
    )

    assert first == second


def test_white_noise_snr_sweep_off_grid_false_matches_default_behavior() -> None:
    implicit = run_white_noise_snr_sweep(
        [40.0],
        trial_count=2,
        pulse_count=5,
        base_seed=13,
    )
    explicit = run_white_noise_snr_sweep(
        [40.0],
        trial_count=2,
        pulse_count=5,
        base_seed=13,
        off_grid=False,
    )

    assert implicit == explicit


def test_white_noise_snr_sweep_off_grid_true_is_deterministic() -> None:
    first = run_white_noise_snr_sweep(
        [40.0],
        trial_count=2,
        pulse_count=5,
        base_seed=13,
        off_grid=True,
    )
    second = run_white_noise_snr_sweep(
        [40.0],
        trial_count=2,
        pulse_count=5,
        base_seed=13,
        off_grid=True,
    )

    assert first == second


def test_matched_filter_snr_sweep_has_finite_rms_timing_error() -> None:
    results = run_white_noise_snr_sweep(
        [30.0, 40.0],
        trial_count=2,
        pulse_count=5,
        base_seed=7,
    )

    for result in results:
        assert math.isfinite(result["mean_bias_error"])
        assert math.isfinite(result["mean_bias_error_s"])
        assert math.isfinite(result["mean_bias_error_samples"])
        assert math.isfinite(result["mean_rms_error"])
        assert math.isfinite(result["mean_rms_error_samples"])
        assert math.isfinite(result["sigma_crlb_samples"])
        assert math.isfinite(result["efficiency"])
        assert math.isfinite(result["max_rms_error"])
        assert math.isfinite(result["p95_abs_error"])
        assert math.isfinite(result["p95_abs_error_s"])
        assert math.isfinite(result["p95_abs_error_samples"])


def test_high_snr_matched_filter_sweep_has_no_missed_or_extra_detections() -> None:
    result = run_white_noise_snr_sweep(
        [40.0],
        trial_count=3,
        pulse_count=5,
        base_seed=11,
    )[0]

    assert result["total_missed_count"] == 0
    assert result["total_extra_count"] == 0


def test_white_noise_snr_sweep_reflects_trial_count_in_every_row() -> None:
    results = run_white_noise_snr_sweep(
        [30.0, 40.0],
        trial_count=4,
        pulse_count=5,
        base_seed=3,
    )

    assert [result["trial_count"] for result in results] == [4, 4]
    assert [result["total_trials"] for result in results] == [4, 4]


def test_white_noise_snr_sweep_reports_total_true_pulses_from_trials() -> None:
    result = run_white_noise_snr_sweep(
        [40.0],
        trial_count=4,
        pulse_count=5,
        base_seed=3,
    )[0]

    assert result["total_true_pulses"] == 20


def test_white_noise_snr_sweep_reports_requested_pulse_count() -> None:
    result = run_white_noise_snr_sweep(
        [40.0],
        trial_count=4,
        pulse_count=5,
        base_seed=3,
    )[0]

    assert result["pulse_count"] == 5
    assert result["requested_pulse_count"] == 5


def test_white_noise_snr_sweep_reports_off_grid_setting() -> None:
    result = run_white_noise_snr_sweep(
        [40.0],
        trial_count=2,
        pulse_count=5,
        base_seed=3,
        off_grid=True,
    )[0]

    assert result["off_grid"] is True


def test_white_noise_snr_sweep_detection_rates_are_bounded() -> None:
    result = run_white_noise_snr_sweep(
        [20.0],
        trial_count=2,
        pulse_count=5,
        base_seed=3,
    )[0]

    assert 0.0 <= result["detection_rate"] <= 1.0
    assert 0.0 <= result["missed_detection_rate"] <= 1.0


def test_white_noise_snr_sweep_rates_are_inside_confidence_intervals() -> None:
    result = run_white_noise_snr_sweep(
        [20.0],
        trial_count=2,
        pulse_count=5,
        base_seed=3,
    )[0]

    assert result["detection_rate_ci_low"] <= result["detection_rate"]
    assert result["detection_rate"] <= result["detection_rate_ci_high"]
    assert result["missed_detection_rate_ci_low"] <= result["missed_detection_rate"]
    assert result["missed_detection_rate"] <= result["missed_detection_rate_ci_high"]
    assert result["false_detection_rate_ci_low"] <= (
        result["false_detection_rate_per_resolution_cell"]
    )
    assert result["false_detection_rate_per_resolution_cell"] <= (
        result["false_detection_rate_ci_high"]
    )


def test_white_noise_snr_sweep_false_detection_rate_is_nonnegative() -> None:
    result = run_white_noise_snr_sweep(
        [20.0],
        trial_count=2,
        pulse_count=5,
        base_seed=3,
    )[0]

    assert result["false_detections_per_100_pulses"] >= 0.0
    assert result["false_detection_rate_per_resolution_cell"] >= 0.0


def test_white_noise_snr_sweep_unit_specific_timing_fields_are_consistent() -> None:
    sample_rate = 100_000
    sample_period = 1.0 / sample_rate
    result = run_white_noise_snr_sweep(
        [40.0],
        trial_count=2,
        pulse_count=5,
        base_seed=3,
        sample_rate=sample_rate,
        off_grid=True,
    )[0]

    if math.isfinite(result["mean_bias_error_s"]):
        assert result["mean_bias_error"] == result["mean_bias_error_s"]
        assert result["mean_bias_error_samples"] == pytest.approx(
            result["mean_bias_error_s"] / sample_period,
        )
    if math.isfinite(result["p95_abs_error_s"]):
        assert result["p95_abs_error"] == result["p95_abs_error_s"]
        assert result["p95_abs_error_samples"] == pytest.approx(
            result["p95_abs_error_s"] / sample_period,
        )


def test_high_snr_off_grid_sweep_has_finite_nonnegative_rms_error() -> None:
    result = run_white_noise_snr_sweep(
        [80.0],
        trial_count=3,
        pulse_count=5,
        base_seed=17,
        off_grid=True,
    )[0]

    assert math.isfinite(result["mean_rms_error"])
    assert math.isfinite(result["mean_rms_error_samples"])
    assert result["mean_rms_error"] >= 0.0
    assert result["mean_rms_error_samples"] >= 0.0


def test_high_snr_off_grid_sweep_has_nonzero_sample_error() -> None:
    result = run_white_noise_snr_sweep(
        [80.0],
        trial_count=3,
        pulse_count=5,
        base_seed=17,
        off_grid=True,
    )[0]

    assert result["mean_rms_error_samples"] > 0.0


def test_white_noise_snr_sweep_detection_count_invariant() -> None:
    result = run_white_noise_snr_sweep(
        [10.0],
        trial_count=2,
        pulse_count=5,
        base_seed=4,
    )[0]

    assert result["total_estimated_pulses"] == (
        result["total_true_pulses"]
        - result["total_missed_count"]
        + result["total_extra_count"]
    )


def test_white_noise_snr_sweep_timing_aggregates_ignore_no_match_trials(
    monkeypatch,
) -> None:
    calls = {"count": 0}

    def fake_estimator(*_args, **_kwargs) -> np.ndarray:
        calls["count"] += 1
        if calls["count"] == 1:
            return np.array([], dtype=float)
        return np.array([0.1, 0.2, 0.3], dtype=float)

    monkeypatch.setattr(sweeps, "estimate_toa_matched_filter", fake_estimator)

    result = sweeps.run_white_noise_snr_sweep(
        [40.0],
        trial_count=2,
        pulse_count=3,
        pulse_rate=10,
        base_seed=5,
    )[0]

    assert math.isfinite(result["mean_rms_error"])
    assert math.isfinite(result["mean_bias_error"])
    assert math.isfinite(result["p95_abs_error"])
    assert math.isfinite(result["max_rms_error"])


def test_white_noise_snr_sweep_timing_aggregates_are_nan_without_matches(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sweeps,
        "estimate_toa_matched_filter",
        lambda *_args, **_kwargs: np.array([], dtype=float),
    )

    result = sweeps.run_white_noise_snr_sweep(
        [40.0],
        trial_count=2,
        pulse_count=3,
        pulse_rate=10,
        base_seed=5,
    )[0]

    assert math.isnan(result["mean_rms_error"])
    assert math.isnan(result["mean_bias_error"])
    assert math.isnan(result["p95_abs_error"])
    assert math.isnan(result["max_rms_error"])


def test_white_noise_snr_sweep_rejects_empty_snr_values() -> None:
    with pytest.raises(ValueError):
        run_white_noise_snr_sweep([])


@pytest.mark.parametrize("trial_count", [0, -1])
def test_white_noise_snr_sweep_rejects_nonpositive_trial_count(
    trial_count: int,
) -> None:
    with pytest.raises(ValueError):
        run_white_noise_snr_sweep([40.0], trial_count=trial_count)


@pytest.mark.parametrize("pulse_count", [0, -1])
def test_white_noise_snr_sweep_rejects_nonpositive_pulse_count(
    pulse_count: int,
) -> None:
    with pytest.raises(ValueError):
        run_white_noise_snr_sweep([40.0], pulse_count=pulse_count)


def test_white_noise_snr_sweep_rejects_zero_amplitude() -> None:
    with pytest.raises(ValueError, match="amplitude"):
        run_white_noise_snr_sweep([40.0], amplitude=0.0)


def test_white_noise_snr_sweep_rejects_duration_with_no_true_pulses() -> None:
    with pytest.raises(ValueError, match="no true pulse arrivals"):
        run_white_noise_snr_sweep(
            [40.0],
            sample_rate=10_000,
            duration=0.001,
            pulse_rate=50,
        )


@pytest.mark.parametrize("snr_db", [float("nan"), float("inf")])
def test_white_noise_snr_sweep_rejects_nonfinite_snr(snr_db: float) -> None:
    with pytest.raises(ValueError):
        run_white_noise_snr_sweep([snr_db])
