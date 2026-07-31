"""Monostatic pulse-echo response of a fluid-loaded elastic plate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .materials import Fluid
from .model import AngularSpectrumModel
from .plate import (
    elastic_plate_scattering_map,
    fluid_interface_scattering,
    vertical_wavenumber,
)


@dataclass(frozen=True)
class PulseEchoResult:
    """Signals and spectra returned by :func:`simulate_monostatic_pulse_echo`."""

    time_s: NDArray[np.float64]
    drive_signal: NDArray[np.float64]
    received_signal: NDArray[np.float64]
    plate_front_signal: NDArray[np.float64]
    backing_signal: NDArray[np.float64]
    frequency_hz: NDArray[np.float64]
    drive_spectrum: NDArray[np.complex128]
    received_spectrum: NDArray[np.complex128]
    electroacoustic_response: NDArray[np.complex128]
    round_trip_transfer: NDArray[np.complex128]
    simulated_bin_mask: NDArray[np.bool_]


def sine_burst(
    *,
    center_frequency_hz: float,
    cycles: float,
    sample_rate_hz: float,
    record_length_s: float,
    start_time_s: float = 0.0,
    amplitude: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a finite sine burst starting at a positive-going zero crossing."""

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
    signal = np.zeros_like(time)
    signal[active] = amplitude * np.sin(
        2.0 * np.pi * center_frequency_hz * local_time[active]
    )
    return time, signal


def _validate_time_signal(
    time_s: ArrayLike,
    signal: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    time = np.asarray(time_s, dtype=float)
    values = np.asarray(signal, dtype=float)
    if time.ndim != 1 or values.ndim != 1 or time.size != values.size:
        raise ValueError("time_s and drive_signal must be equal 1D arrays")
    if time.size < 8 or np.any(~np.isfinite(time)) or np.any(~np.isfinite(values)):
        raise ValueError("time_s and drive_signal must contain finite samples")
    delta_t = np.diff(time)
    if np.any(delta_t <= 0.0) or not np.allclose(
        delta_t, delta_t[0], rtol=1e-8, atol=0.0
    ):
        raise ValueError("time_s must be strictly increasing and uniformly sampled")
    return time, values, float(delta_t[0])


def _frequency_response(
    frequency_hz: NDArray[np.float64],
    response: Callable[[NDArray[np.float64]], ArrayLike] | ArrayLike | None,
    *,
    name: str,
) -> NDArray[np.complex128]:
    if response is None:
        values = np.ones_like(frequency_hz, dtype=np.complex128)
    elif callable(response):
        values = np.asarray(response(frequency_hz), dtype=np.complex128)
    else:
        values = np.asarray(response, dtype=np.complex128)
    if values.shape != frequency_hz.shape:
        raise ValueError(f"{name} must match the rFFT frequency grid")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def _monostatic_transfer_components(
    model: AngularSpectrumModel,
    frequency_hz: float,
    *,
    fluid_layer_thickness_m: float,
    backing_fluid: Fluid,
) -> tuple[complex, complex]:
    _, _, q = model.grid.spectral_mesh()
    radial_samples = model.plate_radial_samples
    reflection_forward, transmission_forward = elastic_plate_scattering_map(
        q,
        frequency_hz,
        model.incident_fluid,
        model.plate,
        model.transmitted_fluid,
        radial_samples=radial_samples,
    )
    reflection_reverse, transmission_reverse = elastic_plate_scattering_map(
        q,
        frequency_hz,
        model.transmitted_fluid,
        model.plate,
        model.incident_fluid,
        radial_samples=radial_samples,
    )
    backing_reflection = fluid_interface_scattering(
        q,
        frequency_hz,
        model.transmitted_fluid,
        backing_fluid,
    )[0]
    layer_kz = vertical_wavenumber(
        model.transmitted_fluid.wavenumber(frequency_hz), q
    )
    cavity_load = backing_reflection * np.exp(
        2j * layer_kz * fluid_layer_thickness_m
    )
    backing_term = (
        transmission_forward
        * transmission_reverse
        * cavity_load
        / (1.0 - reflection_reverse * cavity_load)
    )

    source_pressure = model.source_pressure(frequency_hz)
    source_spectrum = np.fft.fft2(source_pressure)
    propagation = model._propagator(
        model.incident_fluid,
        frequency_hz,
        model.water_path_m,
    )
    receiver_weight = (
        source_pressure / model.aperture.pressure_amplitude_pa
    )
    receiver_normalization = float(np.sum(np.abs(receiver_weight) ** 2))

    def project(reflection: NDArray[np.complex128]) -> complex:
        returned_spectrum = (
            source_spectrum * propagation**2 * reflection
        )
        returned_field = np.fft.ifft2(returned_spectrum)
        return complex(
            np.sum(receiver_weight * returned_field)
            / receiver_normalization
        )

    return project(reflection_forward), project(backing_term)


def simulate_monostatic_pulse_echo(
    model: AngularSpectrumModel,
    time_s: ArrayLike,
    drive_signal: ArrayLike,
    *,
    fluid_layer_thickness_m: float,
    backing_fluid: Fluid,
    transducer_response: (
        Callable[[NDArray[np.float64]], ArrayLike] | ArrayLike | None
    ) = None,
    round_trip_response: (
        Callable[[NDArray[np.float64]], ArrayLike] | ArrayLike | None
    ) = None,
    relative_spectrum_threshold: float = 1.0e-3,
    minimum_frequency_hz: float = 0.5e6,
    maximum_frequency_hz: float | None = None,
) -> PulseEchoResult:
    """Reconstruct the voltage-proportional echo at the transmitting aperture.

    The same focused aperture is used for transmit and receive.  The returned
    pressure is projected onto the reciprocal receive sensitivity.  The fluid
    layer is terminated by ``backing_fluid`` and includes all reverberations
    between the elastic plate and that final fluid interface.

    ``transducer_response`` is a one-way electro-acoustic response and is
    therefore applied once on transmit and once on receive.  Alternatively,
    ``round_trip_response`` accepts a response measured directly in pulse-echo
    mode and applies it once.  The two arguments are mutually exclusive.  The
    result remains in relative units unless the complete system is calibrated.
    """

    time, drive, delta_t = _validate_time_signal(time_s, drive_signal)
    if (
        not np.isfinite(fluid_layer_thickness_m)
        or fluid_layer_thickness_m <= 0.0
    ):
        raise ValueError("fluid_layer_thickness_m must be finite and > 0")
    if not 0.0 <= relative_spectrum_threshold < 1.0:
        raise ValueError("relative_spectrum_threshold must lie in [0, 1)")
    if minimum_frequency_hz < 0.0:
        raise ValueError("minimum_frequency_hz must be >= 0")
    if maximum_frequency_hz is not None and maximum_frequency_hz <= 0.0:
        raise ValueError("maximum_frequency_hz must be > 0 or None")
    if transducer_response is not None and round_trip_response is not None:
        raise ValueError(
            "use either transducer_response or round_trip_response, not both"
        )

    frequency = np.fft.rfftfreq(time.size, d=delta_t)
    drive_spectrum = np.fft.rfft(drive).astype(np.complex128)
    if round_trip_response is not None:
        electroacoustic_response = _frequency_response(
            frequency,
            round_trip_response,
            name="round_trip_response",
        )
    else:
        one_way_response = _frequency_response(
            frequency,
            transducer_response,
            name="transducer_response",
        )
        electroacoustic_response = one_way_response**2
    driven_spectrum = drive_spectrum * electroacoustic_response
    peak = float(np.max(np.abs(driven_spectrum)))
    active = (frequency > 0.0) & (frequency >= minimum_frequency_hz)
    if maximum_frequency_hz is not None:
        active &= frequency <= maximum_frequency_hz
    if peak > 0.0:
        active &= np.abs(driven_spectrum) >= relative_spectrum_threshold * peak
    else:
        active[:] = False

    front_transfer = np.zeros_like(drive_spectrum)
    backing_transfer = np.zeros_like(drive_spectrum)
    for index in np.flatnonzero(active):
        front, backing = _monostatic_transfer_components(
            model,
            float(frequency[index]),
            fluid_layer_thickness_m=fluid_layer_thickness_m,
            backing_fluid=backing_fluid,
        )
        front_transfer[index] = front
        backing_transfer[index] = backing

    # Package phasors use exp(-i omega t), whereas NumPy's positive-frequency
    # inverse FFT needs the complex-conjugated propagation transfer.
    front_spectrum = driven_spectrum * np.conj(front_transfer)
    backing_spectrum = driven_spectrum * np.conj(backing_transfer)
    received_spectrum = front_spectrum + backing_spectrum
    front_signal = np.fft.irfft(front_spectrum, n=time.size)
    backing_signal = np.fft.irfft(backing_spectrum, n=time.size)
    received_signal = front_signal + backing_signal

    return PulseEchoResult(
        time_s=time,
        drive_signal=drive,
        received_signal=received_signal,
        plate_front_signal=front_signal,
        backing_signal=backing_signal,
        frequency_hz=frequency,
        drive_spectrum=drive_spectrum,
        received_spectrum=received_spectrum,
        electroacoustic_response=electroacoustic_response,
        round_trip_transfer=front_transfer + backing_transfer,
        simulated_bin_mask=active,
    )
