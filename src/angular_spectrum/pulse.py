"""Optional wideband pulse reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import AngularSpectrumModel


@dataclass(frozen=True)
class PulseResult:
    """Time signal and spectra returned by :func:`propagate_pulse_on_axis`."""

    time_s: NDArray[np.float64]
    input_signal: NDArray[np.float64]
    output_signal: NDArray[np.float64]
    frequency_hz: NDArray[np.float64]
    input_spectrum: NDArray[np.complex128]
    output_spectrum: NDArray[np.complex128]
    simulated_bin_mask: NDArray[np.bool_]


def square_burst(
    *,
    center_frequency_hz: float,
    cycles: float,
    sample_rate_hz: float,
    record_length_s: float,
    start_time_s: float = 0.5e-6,
    amplitude: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a finite bipolar square-wave burst."""

    for name, value in {
        "center_frequency_hz": center_frequency_hz,
        "cycles": cycles,
        "sample_rate_hz": sample_rate_hz,
        "record_length_s": record_length_s,
        "amplitude": amplitude,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    if not np.isfinite(start_time_s) or start_time_s < 0.0:
        raise ValueError("start_time_s must be finite and >= 0")
    if sample_rate_hz <= 2.0 * center_frequency_hz:
        raise ValueError("sample_rate_hz must exceed twice the centre frequency")

    sample_count = int(np.ceil(record_length_s * sample_rate_hz))
    time = np.arange(sample_count, dtype=float) / sample_rate_hz
    local_time = time - start_time_s
    duration = cycles / center_frequency_hz
    active = (local_time >= 0.0) & (local_time < duration)
    carrier = np.sin(2.0 * np.pi * center_frequency_hz * local_time)
    signal = np.zeros_like(time)
    signal[active] = amplitude * np.where(carrier[active] >= 0.0, 1.0, -1.0)
    return time, signal


def gaussian_transducer_response(
    frequency_hz: ArrayLike,
    *,
    center_frequency_hz: float = 10.0e6,
    fractional_bandwidth_6db: float = 0.5,
) -> NDArray[np.float64]:
    """Simple zero-phase bandpass placeholder with -6 dB pressure bandwidth."""

    frequency = np.asarray(frequency_hz, dtype=float)
    if center_frequency_hz <= 0.0:
        raise ValueError("center_frequency_hz must be > 0")
    if not 0.0 < fractional_bandwidth_6db <= 2.0:
        raise ValueError("fractional_bandwidth_6db must lie in (0, 2]")
    normalized_offset = (
        2.0
        * (frequency - center_frequency_hz)
        / (fractional_bandwidth_6db * center_frequency_hz)
    )
    return np.exp(-np.log(2.0) * normalized_offset**2)


def asymmetric_gaussian_response(
    frequency_hz: ArrayLike,
    *,
    peak_frequency_hz: float,
    lower_frequency_6db_hz: float,
    upper_frequency_6db_hz: float,
) -> NDArray[np.float64]:
    """Return a zero-phase response with asymmetric -6 dB frequencies.

    This is useful when a pulse-echo certificate reports a peak frequency and
    the lower/upper -6 dB crossings rather than a one-way transducer response.
    The result is 1 at ``peak_frequency_hz`` and 0.5 at both crossings.
    """

    frequency = np.asarray(frequency_hz, dtype=float)
    parameters = {
        "peak_frequency_hz": peak_frequency_hz,
        "lower_frequency_6db_hz": lower_frequency_6db_hz,
        "upper_frequency_6db_hz": upper_frequency_6db_hz,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in parameters.values()):
        raise ValueError("response frequencies must be finite and > 0")
    if not lower_frequency_6db_hz < peak_frequency_hz < upper_frequency_6db_hz:
        raise ValueError(
            "response frequencies must satisfy lower < peak < upper"
        )
    if np.any(~np.isfinite(frequency)):
        raise ValueError("frequency_hz must be finite")

    lower_width = peak_frequency_hz - lower_frequency_6db_hz
    upper_width = upper_frequency_6db_hz - peak_frequency_hz
    normalized_offset = np.where(
        frequency <= peak_frequency_hz,
        (frequency - peak_frequency_hz) / lower_width,
        (frequency - peak_frequency_hz) / upper_width,
    )
    return np.exp(-np.log(2.0) * normalized_offset**2)


def propagate_pulse_on_axis(
    model: AngularSpectrumModel,
    time_s: ArrayLike,
    drive_signal: ArrayLike,
    *,
    z_after_plate_m: float,
    transducer_response: (
        Callable[[NDArray[np.float64]], ArrayLike] | ArrayLike | None
    ) = None,
    relative_spectrum_threshold: float = 1.0e-3,
    minimum_frequency_hz: float = 0.5e6,
    maximum_frequency_hz: float | None = None,
) -> PulseResult:
    """Propagate a drive waveform and reconstruct the on-axis time signal.

    The output remains in relative pressure units unless the aperture pressure
    and electro-acoustic transfer function are calibrated.  The model phasor
    convention is conjugated before multiplication with NumPy's positive-
    frequency inverse-FFT convention.
    """

    time = np.asarray(time_s, dtype=float)
    drive = np.asarray(drive_signal, dtype=float)
    if time.ndim != 1 or drive.ndim != 1 or time.size != drive.size:
        raise ValueError("time_s and drive_signal must be equal 1D arrays")
    if time.size < 8 or np.any(~np.isfinite(time)) or np.any(~np.isfinite(drive)):
        raise ValueError("time_s and drive_signal must contain finite samples")
    delta_t = np.diff(time)
    if np.any(delta_t <= 0.0) or not np.allclose(
        delta_t, delta_t[0], rtol=1e-8, atol=0.0
    ):
        raise ValueError("time_s must be strictly increasing and uniformly sampled")
    if not 0.0 <= relative_spectrum_threshold < 1.0:
        raise ValueError("relative_spectrum_threshold must lie in [0, 1)")
    if minimum_frequency_hz < 0.0:
        raise ValueError("minimum_frequency_hz must be >= 0")
    if maximum_frequency_hz is not None and maximum_frequency_hz <= 0.0:
        raise ValueError("maximum_frequency_hz must be > 0 or None")

    frequency = np.fft.rfftfreq(time.size, d=delta_t[0])
    input_spectrum = np.fft.rfft(drive).astype(np.complex128)
    if transducer_response is None:
        response = np.ones_like(frequency)
    elif callable(transducer_response):
        response = np.asarray(transducer_response(frequency), dtype=np.complex128)
    else:
        response = np.asarray(transducer_response, dtype=np.complex128)
    if response.shape != frequency.shape:
        raise ValueError("transducer_response must match the rFFT frequency grid")
    if np.any(~np.isfinite(response)):
        raise ValueError("transducer_response must be finite")

    driven_spectrum = input_spectrum * response
    peak = float(np.max(np.abs(driven_spectrum)))
    active = (frequency > 0.0) & (frequency >= minimum_frequency_hz)
    if maximum_frequency_hz is not None:
        active &= frequency <= maximum_frequency_hz
    if peak > 0.0:
        active &= np.abs(driven_spectrum) >= relative_spectrum_threshold * peak
    else:
        active[:] = False

    output_spectrum = np.zeros_like(input_spectrum)
    for index in np.flatnonzero(active):
        transfer = model.on_axis_value_after_plate(
            float(frequency[index]), z_after_plate_m
        )
        output_spectrum[index] = driven_spectrum[index] * np.conj(transfer)

    output = np.fft.irfft(output_spectrum, n=time.size)
    return PulseResult(
        time_s=time,
        input_signal=drive,
        output_signal=output,
        frequency_hz=frequency,
        input_spectrum=input_spectrum,
        output_spectrum=output_spectrum,
        simulated_bin_mask=active,
    )
