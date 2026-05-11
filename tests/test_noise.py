import numpy as np
import pytest

from compact_timing_receiver.noise import (
    add_amplitude_fluctuation,
    add_baseline_drift,
    add_white_noise,
    apply_saturation,
    quantize_adc,
)


def test_noise_functions_do_not_mutate_input_arrays() -> None:
    t = np.linspace(0.0, 1.0, 8)
    signal = np.linspace(-1.0, 1.0, 8)
    original_t = t.copy()
    original_signal = signal.copy()

    _ = add_white_noise(signal, std=0.1, seed=1)
    np.testing.assert_array_equal(signal, original_signal)

    _ = add_baseline_drift(t, signal, amplitude=0.2, frequency=1.0)
    np.testing.assert_array_equal(t, original_t)
    np.testing.assert_array_equal(signal, original_signal)

    _ = add_amplitude_fluctuation(signal, std=0.1, seed=2)
    np.testing.assert_array_equal(signal, original_signal)

    _ = apply_saturation(signal, min_value=-0.5, max_value=0.5)
    np.testing.assert_array_equal(signal, original_signal)

    _ = quantize_adc(signal, bits=8, v_min=-1.0, v_max=1.0)
    np.testing.assert_array_equal(signal, original_signal)


def test_white_noise_is_deterministic_with_seed() -> None:
    signal = np.ones(5)

    first = add_white_noise(signal, std=0.1, seed=10)
    second = add_white_noise(signal, std=0.1, seed=10)

    np.testing.assert_array_equal(first, second)


def test_quantize_adc_uses_expected_levels() -> None:
    signal = np.array([-1.0, -0.2, 0.2, 1.0])

    quantized = quantize_adc(signal, bits=2, v_min=-1.0, v_max=1.0)

    expected = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
    np.testing.assert_allclose(quantized, expected)


def test_noise_functions_validate_parameters() -> None:
    signal = np.ones(4)
    t = np.linspace(0.0, 1.0, 4)

    with pytest.raises(ValueError):
        add_white_noise(signal, std=-0.1)

    with pytest.raises(ValueError):
        add_baseline_drift(t[:-1], signal, amplitude=0.1, frequency=1.0)

    with pytest.raises(ValueError):
        add_amplitude_fluctuation(signal, std=-0.1)

    with pytest.raises(ValueError):
        apply_saturation(signal, min_value=1.0, max_value=0.0)

    with pytest.raises(ValueError):
        quantize_adc(signal, bits=0, v_min=0.0, v_max=1.0)

    with pytest.raises(ValueError):
        quantize_adc(signal, bits=8, v_min=1.0, v_max=0.0)
