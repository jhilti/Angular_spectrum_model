"""Small analysis helpers for focal metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def fwhm(coordinate: ArrayLike, amplitude: ArrayLike) -> float:
    """Return the full width at half maximum around the global peak.

    The half-maximum is based on pressure amplitude, equivalent to -6.02 dB.
    Linear interpolation is used at the two crossings.
    """

    x = np.asarray(coordinate, dtype=float)
    y = np.abs(np.asarray(amplitude))
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 3:
        raise ValueError("coordinate and amplitude must be equal 1D arrays")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("coordinate and amplitude must be finite")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("coordinate must be strictly increasing")
    peak_index = int(np.argmax(y))
    peak = y[peak_index]
    if peak <= 0.0:
        return float("nan")
    threshold = 0.5 * peak

    left_below = np.flatnonzero(y[:peak_index] < threshold)
    right_below = np.flatnonzero(y[peak_index + 1 :] < threshold)
    if left_below.size == 0 or right_below.size == 0:
        return float("nan")

    left_low = int(left_below[-1])
    left_high = left_low + 1
    right_high = peak_index + 1 + int(right_below[0])
    right_low = right_high - 1

    def crossing(i0: int, i1: int) -> float:
        fraction = (threshold - y[i0]) / (y[i1] - y[i0])
        return float(x[i0] + fraction * (x[i1] - x[i0]))

    x_left = crossing(left_low, left_high)
    x_right = crossing(right_low, right_high)
    return x_right - x_left


def amplitude_db(amplitude: ArrayLike, floor_db: float = -60.0) -> np.ndarray:
    """Normalize a pressure magnitude and express it in dB."""

    value = np.abs(np.asarray(amplitude))
    peak = float(np.max(value))
    if peak == 0.0:
        return np.full(value.shape, floor_db, dtype=float)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(value / peak)
    return np.maximum(db, floor_db)
