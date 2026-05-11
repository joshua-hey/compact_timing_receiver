import numpy as np

from compact_timing_receiver.experiments import run_timing_recovery_trial


def test_timing_recovery_trial_returns_expected_keys() -> None:
    result = run_timing_recovery_trial()

    assert set(result) == {
        "t",
        "signal",
        "true_arrival_times",
        "estimated_arrival_times",
        "timing_errors",
        "summary",
    }
    assert {
        "missed_count",
        "extra_count",
        "mean_error",
        "rms_error",
        "std_error",
        "max_abs_error",
    } <= set(result["summary"])


def test_clean_timing_recovery_trial_has_no_missed_or_extra_detections() -> None:
    result = run_timing_recovery_trial()
    summary = result["summary"]

    assert summary["missed_count"] == 0
    assert summary["extra_count"] == 0


def test_clean_timing_recovery_trial_has_sample_level_rms_error() -> None:
    sample_rate = 100_000
    result = run_timing_recovery_trial(sample_rate=sample_rate)

    assert result["summary"]["rms_error"] <= 3.0 / sample_rate


def test_seeded_timing_recovery_trials_are_deterministic() -> None:
    first = run_timing_recovery_trial(
        seed=12,
        white_noise_std=0.02,
        baseline_drift_amplitude=0.03,
        baseline_drift_frequency=2.0,
        adc_bits=10,
    )
    second = run_timing_recovery_trial(
        seed=12,
        white_noise_std=0.02,
        baseline_drift_amplitude=0.03,
        baseline_drift_frequency=2.0,
        adc_bits=10,
    )

    np.testing.assert_array_equal(
        first["true_arrival_times"],
        second["true_arrival_times"],
    )
    np.testing.assert_array_equal(
        first["estimated_arrival_times"],
        second["estimated_arrival_times"],
    )
    np.testing.assert_array_equal(first["timing_errors"], second["timing_errors"])
    assert first["summary"] == second["summary"]
