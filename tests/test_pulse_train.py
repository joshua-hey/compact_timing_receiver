import numpy as np
import pytest

from compact_timing_receiver.pulse_train import generate_pulse_train


def test_pulse_generator_returns_expected_shapes_and_arrivals() -> None:
    t, signal, arrivals = generate_pulse_train(
        sample_rate=10_000,
        duration=0.1,
        pulse_rate=50,
        pulse_width=0.001,
        seed=123,
    )

    assert t.shape == signal.shape
    assert t.ndim == 1
    assert signal.ndim == 1
    assert arrivals.ndim == 1
    assert arrivals.size > 0
    assert np.all(arrivals >= 0.0)
    assert np.all(arrivals < 0.1)


def test_pulse_generator_is_deterministic_with_seed() -> None:
    first = generate_pulse_train(10_000, 0.1, 50, 0.001, jitter=0.0001, seed=5)
    second = generate_pulse_train(10_000, 0.1, 50, 0.001, jitter=0.0001, seed=5)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_pulse_generator_validates_arguments() -> None:
    with pytest.raises(ValueError):
        generate_pulse_train(0.0, 0.1, 50, 0.001)

    with pytest.raises(ValueError):
        generate_pulse_train(10_000, 0.1, 50, 0.001, dropout=1.5)

    with pytest.raises(ValueError, match="clock_drift"):
        generate_pulse_train(10_000, 0.1, 50, 0.001, clock_drift=-1.0)


def test_generate_pulse_train_clock_offset_shifts_all_arrivals() -> None:
    _, _, nominal_arrivals = generate_pulse_train(
        sample_rate=100_000,
        duration=0.2,
        pulse_rate=20,
        pulse_width=0.001,
    )
    _, _, offset_arrivals = generate_pulse_train(
        sample_rate=100_000,
        duration=0.2,
        pulse_rate=20,
        pulse_width=0.001,
        clock_offset=0.007,
    )

    assert offset_arrivals.size == nominal_arrivals.size
    np.testing.assert_allclose(
        offset_arrivals - nominal_arrivals,
        0.007,
        rtol=0.0,
        atol=1e-12,
    )


def test_generate_pulse_train_clock_drift_progressively_changes_arrivals() -> None:
    _, _, nominal_arrivals = generate_pulse_train(
        sample_rate=100_000,
        duration=0.5,
        pulse_rate=10,
        pulse_width=0.001,
    )
    _, _, drifted_arrivals = generate_pulse_train(
        sample_rate=100_000,
        duration=0.5,
        pulse_rate=10,
        pulse_width=0.001,
        clock_drift=0.1,
    )

    arrival_delta = drifted_arrivals - nominal_arrivals

    assert drifted_arrivals.size == nominal_arrivals.size
    assert np.all(np.diff(arrival_delta) > 0.0)
    np.testing.assert_allclose(
        arrival_delta,
        nominal_arrivals * 0.1,
        rtol=0.0,
        atol=1e-12,
    )
