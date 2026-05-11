import math

from compact_timing_receiver.sweeps import run_white_noise_snr_sweep


def test_white_noise_snr_sweep_returns_one_matched_filter_row_per_snr() -> None:
    results = run_white_noise_snr_sweep([30.0, 40.0], trial_count=2, base_seed=5)

    assert len(results) == 2
    assert [row["estimator"] for row in results] == ["matched_filter", "matched_filter"]
    assert [row["snr_db"] for row in results] == [30.0, 40.0]


def test_white_noise_snr_sweep_result_fields_are_present() -> None:
    result = run_white_noise_snr_sweep([40.0], trial_count=1, base_seed=5)[0]

    assert {
        "estimator",
        "snr_db",
        "trial_count",
        "total_trials",
        "total_true_pulses",
        "mean_bias_error",
        "mean_rms_error",
        "mean_rms_error_samples",
        "mean_missed_count",
        "mean_extra_count",
        "max_rms_error",
        "max_missed_count",
        "max_extra_count",
    } <= set(result)


def test_white_noise_snr_sweep_is_deterministic_for_same_base_seed() -> None:
    first = run_white_noise_snr_sweep([30.0, 40.0], trial_count=2, base_seed=9)
    second = run_white_noise_snr_sweep([30.0, 40.0], trial_count=2, base_seed=9)

    assert first == second


def test_matched_filter_snr_sweep_has_finite_rms_timing_error() -> None:
    results = run_white_noise_snr_sweep([30.0, 40.0], trial_count=2, base_seed=7)

    for result in results:
        assert math.isfinite(result["mean_bias_error"])
        assert math.isfinite(result["mean_rms_error"])
        assert math.isfinite(result["mean_rms_error_samples"])
        assert math.isfinite(result["max_rms_error"])


def test_high_snr_matched_filter_sweep_has_no_missed_or_extra_detections() -> None:
    result = run_white_noise_snr_sweep([40.0], trial_count=3, base_seed=11)[0]

    assert result["mean_missed_count"] == 0.0
    assert result["mean_extra_count"] == 0.0
    assert result["max_missed_count"] == 0
    assert result["max_extra_count"] == 0


def test_white_noise_snr_sweep_reflects_trial_count_in_every_row() -> None:
    results = run_white_noise_snr_sweep([30.0, 40.0], trial_count=4, base_seed=3)

    assert [result["trial_count"] for result in results] == [4, 4]
    assert [result["total_trials"] for result in results] == [4, 4]


def test_white_noise_snr_sweep_reports_total_true_pulses() -> None:
    result = run_white_noise_snr_sweep([40.0], trial_count=4, base_seed=3)[0]

    assert result["total_true_pulses"] > 0
