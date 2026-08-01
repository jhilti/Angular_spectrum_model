"""Monostatic pulse-echo response of a fluid-loaded elastic plate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .materials import Fluid
from .model import AngularSpectrumModel, validate_focused_grid_support
from .plate import (
    elastic_plate_scattering_map,
    fluid_interface_scattering,
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
    physical_round_trip_phasor: NDArray[np.complex128]
    applied_round_trip_transfer: NDArray[np.complex128]
    round_trip_transfer: NDArray[np.complex128]
    simulated_bin_mask: NDArray[np.bool_]
    fluid_cavity_echo_count: int


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
    duration = cycles / center_frequency_hz
    available_end_s = sample_count / sample_rate_hz
    tolerance_s = np.finfo(float).eps * max(available_end_s, 1.0)
    if start_time_s + duration > available_end_s + tolerance_s:
        raise ValueError(
            "record_length_s is too short to contain the complete sine burst"
        )
    time = np.arange(sample_count, dtype=float) / sample_rate_hz
    local_time = time - start_time_s
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
    if values.ndim == 0:
        values = np.full(frequency_hz.shape, values, dtype=np.complex128)
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
    fluid_cavity_echo_count: int | None = None,
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
    layer_round_trip = model._propagator(
        model.transmitted_fluid,
        frequency_hz,
        2.0 * fluid_layer_thickness_m,
    )
    cavity_load = backing_reflection * layer_round_trip
    cavity_loop = reflection_reverse * cavity_load
    if fluid_cavity_echo_count is None:
        cavity_response = cavity_load / (1.0 - cavity_loop)
    else:
        if fluid_cavity_echo_count < 0:
            raise ValueError("fluid_cavity_echo_count must be >= 0")
        if fluid_cavity_echo_count == 0:
            cavity_response = np.zeros_like(cavity_load)
        else:
            # Each later echo has a longer physical liquid path and therefore
            # needs its own band-limited propagator. Reusing the 2h mask via a
            # geometric power would retain transverse modes that have left the
            # finite FFT window on the 4h, 6h, ... paths.
            interface_loop = reflection_reverse * backing_reflection
            interface_power = np.ones_like(interface_loop)
            cavity_response = np.zeros_like(cavity_load)
            for order in range(1, fluid_cavity_echo_count + 1):
                order_propagation = model._propagator(
                    model.transmitted_fluid,
                    frequency_hz,
                    2.0 * order * fluid_layer_thickness_m,
                )
                cavity_response += (
                    backing_reflection
                    * interface_power
                    * order_propagation
                    * model._combined_bandlimit_mask(
                        frequency_hz,
                        (
                            (
                                model.incident_fluid,
                                2.0 * model.water_path_m,
                            ),
                            (
                                model.transmitted_fluid,
                                2.0 * order * fluid_layer_thickness_m,
                            ),
                        ),
                    )
                )
                interface_power *= interface_loop
    backing_term = (
        transmission_forward
        * transmission_reverse
        * cavity_response
    )

    source_pressure = model.source_pressure(frequency_hz)
    source_spectrum = np.fft.fft2(source_pressure)
    round_trip_propagation = model._propagator(
        model.incident_fluid,
        frequency_hz,
        2.0 * model.water_path_m,
    )
    round_trip_propagation *= model._combined_bandlimit_mask(
        frequency_hz,
        ((model.incident_fluid, 2.0 * model.water_path_m),),
    )
    receiver_weight = (
        source_pressure / model.aperture.pressure_amplitude_pa
    )
    receiver_normalization = float(np.sum(np.abs(receiver_weight) ** 2))

    def project(reflection: NDArray[np.complex128]) -> complex:
        returned_spectrum = (
            source_spectrum * round_trip_propagation * reflection
        )
        returned_field = np.fft.ifft2(returned_spectrum)
        return complex(
            np.sum(receiver_weight * returned_field)
            / receiver_normalization
        )

    return project(reflection_forward), project(backing_term)


def _causal_fluid_cavity_echo_count(
    model: AngularSpectrumModel,
    time_s: NDArray[np.float64],
    drive_signal: NDArray[np.float64],
    *,
    fluid_layer_thickness_m: float,
    drive_time_support_s: tuple[float, float] | None = None,
    response_guard_s: float | None = None,
) -> int:
    """Return the number of surface echoes that fit in the FFT record.

    A frequency-domain infinite cavity series is periodic under an inverse
    discrete Fourier transform: echoes later than the record wrap to its
    beginning. A finite series containing only causally observable echoes
    avoids that artifact. The nominal burst duration is retained as a guard
    for the finite pulse and its band-limited ringing.
    """

    delta_t = float(time_s[1] - time_s[0])
    record_period_s = time_s.size * delta_t
    if drive_time_support_s is not None:
        if len(drive_time_support_s) != 2:
            raise ValueError("drive_time_support_s must contain (start, end)")
        support_start_s, support_end_s = map(float, drive_time_support_s)
        record_end_s = float(time_s[0] + record_period_s)
        if (
            not np.isfinite(support_start_s)
            or not np.isfinite(support_end_s)
            or support_start_s < time_s[0]
            or support_end_s <= support_start_s
            or support_end_s > record_end_s + 8.0 * np.finfo(float).eps
        ):
            raise ValueError(
                "drive_time_support_s must be a finite (start, end) interval "
                "inside the supplied time record"
            )
        first_drive_s = support_start_s - float(time_s[0])
        last_drive_s = support_end_s - float(time_s[0])
        drive_duration_s = support_end_s - support_start_s
    else:
        magnitude = np.abs(drive_signal)
        peak = float(np.max(magnitude))
        if peak > 0.0:
            driven = np.flatnonzero(magnitude > peak * 1.0e-12)
            first_drive_s = float(time_s[driven[0]] - time_s[0])
            last_drive_s = float(time_s[driven[-1]] - time_s[0] + delta_t)
            drive_duration_s = last_drive_s - first_drive_s
        else:
            return 0

    first_front_delay_s = (
        2.0 * model.water_path_m / model.incident_fluid.sound_speed_m_s
    )
    if response_guard_s is not None and (
        not np.isfinite(response_guard_s) or response_guard_s < 0.0
    ):
        raise ValueError("response_guard_s must be finite and >= 0 or None")
    guard_s = max(
        8.0 * delta_t,
        drive_duration_s if response_guard_s is None else response_guard_s,
    )
    if record_period_s - last_drive_s - first_front_delay_s - guard_s < 0.0:
        raise ValueError(
            "the time record is too short to contain the first plate-front "
            "echo and its pulse guard; increase record_length_s"
        )

    first_surface_delay_s = 2.0 * (
        model.water_path_m / model.incident_fluid.sound_speed_m_s
        + model.plate.thickness_m
        / model.plate.solid.longitudinal_speed_m_s
        + fluid_layer_thickness_m
        / model.transmitted_fluid.sound_speed_m_s
    )
    cavity_round_trip_s = (
        2.0
        * fluid_layer_thickness_m
        / model.transmitted_fluid.sound_speed_m_s
    )
    available_after_first_s = (
        record_period_s
        - last_drive_s
        - first_surface_delay_s
        - guard_s
    )
    if available_after_first_s < 0.0:
        return 0
    # An echo whose nominal guarded end lands exactly on the periodic record
    # boundary is not retained: any oblique-path or group-delay excess would
    # wrap it to the beginning of the inverse FFT record.
    strict_available_s = np.nextafter(available_after_first_s, -np.inf)
    return int(np.floor(strict_available_s / cavity_round_trip_s)) + 1


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
    drive_time_support_s: tuple[float, float] | None = None,
    response_guard_s: float | None = None,
    fluid_cavity_echo_count: int | None = None,
) -> PulseEchoResult:
    """Reconstruct the voltage-proportional echo at the transmitting aperture.

    The same focused aperture is used for transmit and receive.  The returned
    pressure is projected onto the reciprocal receive sensitivity. The fluid
    layer is terminated by ``backing_fluid``. Its time-domain cavity response
    contains every surface echo that fits causally in the supplied record;
    later echoes are omitted instead of wrapping to the record start.

    ``transducer_response`` is a one-way electro-acoustic response and is
    therefore applied once on transmit and once on receive.  Alternatively,
    ``round_trip_response`` accepts a response measured directly in pulse-echo
    mode and applies it once.  The two arguments are mutually exclusive.  The
    result remains in relative units unless the complete system is calibrated.
    ``time_s`` should extend beyond the latest echo of interest by the burst
    duration and a ringing margin. The high-level app and bundled examples
    retain the first robust liquid-surface order; callers can request more
    orders after increasing the grid and record and checking convergence.
    ``drive_time_support_s`` may provide the causal start/end of an original
    finite source burst when ``drive_signal`` has already been filtered by a
    zero-phase frequency response. This prevents circular filter tails from
    being mistaken for an infinitely long source.
    ``response_guard_s`` reserves time for measured electro-acoustic ring-down;
    the default is one source duration (and at least eight samples). An
    explicit nonnegative ``fluid_cavity_echo_count`` may retain fewer echoes,
    but values that would extend past the record are rejected.
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

    maximum_cavity_echo_count = _causal_fluid_cavity_echo_count(
        model,
        time,
        drive,
        fluid_layer_thickness_m=fluid_layer_thickness_m,
        drive_time_support_s=drive_time_support_s,
        response_guard_s=response_guard_s,
    )
    if fluid_cavity_echo_count is None:
        retained_cavity_echo_count = maximum_cavity_echo_count
    else:
        if (
            isinstance(fluid_cavity_echo_count, bool)
            or not isinstance(fluid_cavity_echo_count, (int, np.integer))
            or fluid_cavity_echo_count < 0
        ):
            raise ValueError("fluid_cavity_echo_count must be an integer >= 0")
        if fluid_cavity_echo_count > maximum_cavity_echo_count:
            raise ValueError(
                "fluid_cavity_echo_count would extend beyond the supplied "
                "time record and wrap under the inverse FFT"
            )
        retained_cavity_echo_count = int(fluid_cavity_echo_count)

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

    active_indices = np.flatnonzero(active)
    if active_indices.size:
        validation_segments = [
            (
                "water pulse-echo round trip",
                model.incident_fluid,
                2.0 * model.water_path_m,
            )
        ]
        if retained_cavity_echo_count > 0:
            validation_segments.append(
                (
                    "first fluid-layer cavity round trip",
                    model.transmitted_fluid,
                    2.0 * fluid_layer_thickness_m,
                )
            )
        validate_focused_grid_support(
            model,
            maximum_frequency_hz=float(frequency[active_indices[-1]]),
            propagation_segments=validation_segments,
        )

    front_transfer = np.zeros_like(drive_spectrum)
    backing_transfer = np.zeros_like(drive_spectrum)
    for index in active_indices:
        front, backing = _monostatic_transfer_components(
            model,
            float(frequency[index]),
            fluid_layer_thickness_m=fluid_layer_thickness_m,
            backing_fluid=backing_fluid,
            fluid_cavity_echo_count=retained_cavity_echo_count,
        )
        front_transfer[index] = front
        backing_transfer[index] = backing

    # Package phasors use exp(-i omega t), whereas NumPy's positive-frequency
    # inverse FFT needs the complex-conjugated propagation transfer.
    physical_round_trip_phasor = front_transfer + backing_transfer
    applied_round_trip_transfer = np.conj(physical_round_trip_phasor)
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
        physical_round_trip_phasor=physical_round_trip_phasor,
        applied_round_trip_transfer=applied_round_trip_transfer,
        # Backward-compatible alias for the package-convention phasor. New
        # code should choose one of the two explicitly named fields above.
        round_trip_transfer=physical_round_trip_phasor,
        simulated_bin_mask=active,
        fluid_cavity_echo_count=retained_cavity_echo_count,
    )
