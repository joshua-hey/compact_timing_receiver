"""Simple synthetic signal impairments."""

from __future__ import annotations

import numpy as np


def _as_float_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def add_white_noise(signal: np.ndarray, std: float, seed: int | None = None) -> np.ndarray:
    """Return signal with additive Gaussian white noise."""

    samples = _as_float_array("signal", signal)
    if std < 0.0 or not np.isfinite(std):
        raise ValueError("std must be finite and non-negative")

    rng = np.random.default_rng(seed)
    return samples + rng.normal(0.0, std, size=samples.shape)


def add_baseline_drift(
    t: np.ndarray,
    signal: np.ndarray,
    amplitude: float,
    frequency: float,
) -> np.ndarray:
    """Return signal with sinusoidal baseline drift added."""

    time = _as_float_array("t", t)
    samples = _as_float_array("signal", signal)
    if time.shape != samples.shape:
        raise ValueError("t and signal must have the same shape")
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    if frequency < 0.0 or not np.isfinite(frequency):
        raise ValueError("frequency must be finite and non-negative")

    return samples + amplitude * np.sin(2.0 * np.pi * frequency * time)


def add_amplitude_fluctuation(
    signal: np.ndarray,
    std: float,
    seed: int | None = None,
) -> np.ndarray:
    """Return signal multiplied by Gaussian amplitude fluctuations."""

    samples = _as_float_array("signal", signal)
    if std < 0.0 or not np.isfinite(std):
        raise ValueError("std must be finite and non-negative")

    rng = np.random.default_rng(seed)
    gain = 1.0 + rng.normal(0.0, std, size=samples.shape)
    return samples * gain


def apply_saturation(signal: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """Return signal clipped to the provided limits."""

    samples = _as_float_array("signal", signal)
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        raise ValueError("saturation limits must be finite")
    if min_value >= max_value:
        raise ValueError("min_value must be less than max_value")

    return np.clip(samples, min_value, max_value)


def quantize_adc(signal: np.ndarray, bits: int, v_min: float, v_max: float) -> np.ndarray:
    """Return signal quantized to evenly spaced ADC levels."""

    samples = _as_float_array("signal", signal)
    if bits < 1:
        raise ValueError("bits must be at least 1")
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        raise ValueError("ADC limits must be finite")
    if v_min >= v_max:
        raise ValueError("v_min must be less than v_max")

    levels = 2**bits
    clipped = np.clip(samples, v_min, v_max)
    scaled = (clipped - v_min) / (v_max - v_min)
    codes = np.rint(scaled * (levels - 1))
    return v_min + codes * (v_max - v_min) / (levels - 1)
