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
