"""Small plotting helpers shared by the interactive application."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
from numpy.typing import ArrayLike


INTERFACE_REFLECTIONS = (
    "Water–plate",
    "Plate–DMSO",
    "DMSO–air",
)


def interface_reflection_window_us(
    simulated_arrivals_us: Mapping[str, float],
    measured_arrivals_us: Mapping[str, float] | None = None,
    *,
    minimum_padding_us: float = 0.8,
    fractional_padding: float = 0.12,
    minimum_right_padding_us: float = 0.0,
) -> tuple[float, float]:
    """Return a time window that contains all three interface reflections.

    The coordinate system is unchanged: values remain referenced to the
    excitation start.  When a survey overlay is present, its three picked
    interface times are included as well so neither trace is silently cropped.
    """

    if not math.isfinite(minimum_padding_us) or minimum_padding_us <= 0.0:
        raise ValueError("minimum_padding_us must be finite and > 0")
    if not math.isfinite(fractional_padding) or fractional_padding < 0.0:
        raise ValueError("fractional_padding must be finite and >= 0")
    if (
        not math.isfinite(minimum_right_padding_us)
        or minimum_right_padding_us < 0.0
    ):
        raise ValueError("minimum_right_padding_us must be finite and >= 0")

    values: list[float] = []
    for label, arrivals in (
        ("simulated", simulated_arrivals_us),
        ("measured", measured_arrivals_us),
    ):
        if arrivals is None:
            continue
        missing = [name for name in INTERFACE_REFLECTIONS if name not in arrivals]
        if missing:
            raise ValueError(
                f"{label} arrivals are missing: {', '.join(missing)}"
            )
        for name in INTERFACE_REFLECTIONS:
            arrival = float(arrivals[name])
            if not math.isfinite(arrival):
                raise ValueError(f"{label} {name} arrival must be finite")
            values.append(arrival)

    first_arrival = min(values)
    last_arrival = max(values)
    echo_span = last_arrival - first_arrival
    padding = max(minimum_padding_us, fractional_padding * echo_span)
    return (
        max(0.0, first_arrival - padding),
        last_arrival + max(padding, minimum_right_padding_us),
    )


def modeled_echo_right_padding_us(
    time_us: ArrayLike,
    response_envelope: ArrayLike,
    *,
    last_arrival_us: float,
    drive_duration_us: float,
    minimum_ringdown_margin_us: float = 0.8,
    relative_threshold: float = 0.01,
    quiet_interval_us: float = 0.25,
) -> float:
    """Return right padding that retains a finite pulse and modeled ring-down.

    The result is always at least the electrical drive duration plus a
    conservative ring-down margin. When the displayed component envelope is
    available, its contiguous post-peak support is also retained down to
    ``relative_threshold``. A sustained quiet interval prevents isolated
    numerical FFT tails from expanding an otherwise useful echo view.
    """

    time = np.asarray(time_us, dtype=float)
    envelope = np.asarray(response_envelope, dtype=float)
    if (
        time.ndim != 1
        or envelope.ndim != 1
        or time.size != envelope.size
        or time.size < 2
        or np.any(~np.isfinite(time))
        or np.any(~np.isfinite(envelope))
        or np.any(np.diff(time) <= 0.0)
        or np.any(envelope < 0.0)
    ):
        raise ValueError(
            "time_us and response_envelope must be equal finite 1D arrays "
            "with increasing time and a nonnegative envelope"
        )
    for name, value, allow_zero in (
        ("last_arrival_us", last_arrival_us, True),
        ("drive_duration_us", drive_duration_us, True),
        ("minimum_ringdown_margin_us", minimum_ringdown_margin_us, True),
        ("relative_threshold", relative_threshold, False),
        ("quiet_interval_us", quiet_interval_us, False),
    ):
        if (
            not math.isfinite(value)
            or value < 0.0
            or (not allow_zero and value == 0.0)
        ):
            comparison = ">= 0" if allow_zero else "> 0"
            raise ValueError(f"{name} must be finite and {comparison}")
    if relative_threshold >= 1.0:
        raise ValueError("relative_threshold must be < 1")

    required_padding = drive_duration_us + minimum_ringdown_margin_us
    after_arrival = np.flatnonzero(time >= last_arrival_us)
    if after_arrival.size == 0:
        return float(required_padding)
    local_envelope = envelope[after_arrival]
    peak_local_index = int(np.argmax(local_envelope))
    peak = float(local_envelope[peak_local_index])
    if peak <= 0.0:
        return float(required_padding)

    peak_index = int(after_arrival[peak_local_index])
    quiet_samples = max(
        1,
        int(math.ceil(quiet_interval_us / float(np.min(np.diff(time))))),
    )
    above_threshold = envelope >= relative_threshold * peak
    support_end_us = float(time[-1])
    for index in range(peak_index, time.size - quiet_samples + 1):
        if not np.any(above_threshold[index : index + quiet_samples]):
            support_end_us = float(time[index])
            break
    modeled_padding = max(0.0, support_end_us - last_arrival_us)
    return float(max(required_padding, modeled_padding))
