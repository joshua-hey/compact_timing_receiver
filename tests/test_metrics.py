import numpy as np

from compact_timing_receiver.metrics import (
    compute_timing_errors,
    match_arrival_times,
    summarize_timing_errors,
)


def test_match_arrival_times_pairs_perfect_matches() -> None:
    true_times = np.array([0.1, 0.2, 0.3])
    estimated_times = np.array([0.1, 0.2, 0.3])

    true_indices, estimated_indices = match_arrival_times(
        true_times,
        estimated_times,
        tolerance=1e-9,
    )

    np.testing.assert_array_equal(true_indices, np.array([0, 1, 2]))
    np.testing.assert_array_equal(estimated_indices, np.array([0, 1, 2]))


def test_compute_timing_errors_reports_early_and_late_estimates() -> None:
    true_times = np.array([0.1, 0.2, 0.3])
    estimated_times = np.array([0.099, 0.202, 0.3])

    errors, missed_count, extra_count = compute_timing_errors(
        true_times,
        estimated_times,
        tolerance=0.003,
    )

    np.testing.assert_allclose(errors, np.array([-0.001, 0.002, 0.0]), atol=1e-12)
    assert missed_count == 0
    assert extra_count == 0


def test_compute_timing_errors_counts_missed_detections() -> None:
    true_times = np.array([0.1, 0.2, 0.3])
    estimated_times = np.array([0.1, 0.3])

    errors, missed_count, extra_count = compute_timing_errors(
        true_times,
        estimated_times,
        tolerance=1e-9,
    )

    np.testing.assert_allclose(errors, np.array([0.0, 0.0]), atol=1e-12)
    assert missed_count == 1
    assert extra_count == 0


def test_compute_timing_errors_counts_extra_detections() -> None:
    true_times = np.array([0.1, 0.2])
    estimated_times = np.array([0.1, 0.15, 0.2])

    errors, missed_count, extra_count = compute_timing_errors(
        true_times,
        estimated_times,
        tolerance=1e-9,
    )

    np.testing.assert_allclose(errors, np.array([0.0, 0.0]), atol=1e-12)
    assert missed_count == 0
    assert extra_count == 1


def test_compute_timing_errors_counts_estimates_outside_tolerance() -> None:
    true_times = np.array([0.1, 0.2])
    estimated_times = np.array([0.1, 0.206])

    errors, missed_count, extra_count = compute_timing_errors(
        true_times,
        estimated_times,
        tolerance=0.005,
    )

    np.testing.assert_allclose(errors, np.array([0.0]), atol=1e-12)
    assert missed_count == 1
    assert extra_count == 1


def test_summarize_timing_errors_reports_error_statistics_and_counts() -> None:
    summary = summarize_timing_errors(
        errors=np.array([-0.001, 0.002, 0.0]),
        missed_count=1,
        extra_count=2,
    )

    assert summary["matched_count"] == 3
    assert summary["missed_count"] == 1
    assert summary["extra_count"] == 2
    np.testing.assert_allclose(
        summary["mean_error"],
        np.mean(np.array([-0.001, 0.002, 0.0])),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        summary["mean_absolute_error"],
        np.mean(np.array([0.001, 0.002, 0.0])),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        summary["rms_error"],
        np.sqrt(np.mean(np.array([-0.001, 0.002, 0.0]) ** 2)),
        atol=1e-12,
    )
    np.testing.assert_allclose(summary["max_absolute_error"], 0.002, atol=1e-12)
