"""Synthetic timing-receiver utilities."""

from compact_timing_receiver.estimators import (
    estimate_toa_matched_filter,
    estimate_toa_threshold,
)
from compact_timing_receiver.pulse_train import generate_pulse_train

__all__ = [
    "estimate_toa_matched_filter",
    "estimate_toa_threshold",
    "generate_pulse_train",
]

__version__ = "0.1.0"
