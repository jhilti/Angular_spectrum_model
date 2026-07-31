"""Optional electrical and calibrated electro-acoustic pulse-echo models.

The angular-spectrum solver prescribes pressure at an equivalent aperture.  This
module adds a deliberately separate layer that converts an open-circuit source
voltage into terminal voltage/current and, when calibration data are supplied,
into aperture pressure and received voltage.  No probe-specific absolute
calibration is assumed.

Frequency responses use NumPy's rFFT convention: multiplying the positive-
frequency spectrum by a complex response and applying ``irfft`` reconstructs
the corresponding real time-domain waveform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .materials import Fluid
from .model import AngularSpectrumModel
from .pulse_echo import PulseEchoResult, simulate_monostatic_pulse_echo


FrequencyResponse = (
    complex
    | float
    | ArrayLike
    | Callable[[NDArray[np.float64]], ArrayLike]
)


def _validate_time_waveform(
    time_s: ArrayLike,
    waveform: ArrayLike,
    *,
    waveform_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    time = np.asarray(time_s, dtype=float)
    values = np.asarray(waveform, dtype=float)
    if time.ndim != 1 or values.ndim != 1 or time.size != values.size:
        raise ValueError(
            f"time_s and {waveform_name} must be equal 1D arrays"
        )
    if time.size < 8 or np.any(~np.isfinite(time)) or np.any(~np.isfinite(values)):
        raise ValueError(
            f"time_s and {waveform_name} must contain finite samples"
        )
    delta_t = np.diff(time)
    if np.any(delta_t <= 0.0) or not np.allclose(
        delta_t, delta_t[0], rtol=1e-8, atol=0.0
    ):
        raise ValueError("time_s must be strictly increasing and uniformly sampled")
    return time, values, float(delta_t[0])


def _resolve_response(
    frequency_hz: NDArray[np.float64],
    response: FrequencyResponse,
    *,
    name: str,
) -> NDArray[np.complex128]:
    if callable(response):
        values = np.asarray(response(frequency_hz), dtype=np.complex128)
    else:
        values = np.asarray(response, dtype=np.complex128)
    if values.ndim == 0:
        values = np.full(frequency_hz.shape, values.item(), dtype=np.complex128)
    if values.shape != frequency_hz.shape:
        raise ValueError(f"{name} must be scalar or match the rFFT frequency grid")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def _resolve_impedance(
    frequency_hz: NDArray[np.float64],
    impedance_ohm: FrequencyResponse,
    *,
    name: str,
    allow_open_circuit: bool,
) -> NDArray[np.complex128]:
    if callable(impedance_ohm):
        values = np.asarray(impedance_ohm(frequency_hz), dtype=np.complex128)
    else:
        values = np.asarray(impedance_ohm, dtype=np.complex128)
    if values.ndim == 0:
        values = np.full(frequency_hz.shape, values.item(), dtype=np.complex128)
    if values.shape != frequency_hz.shape:
        raise ValueError(f"{name} must be scalar or match the rFFT frequency grid")
    if np.any(np.isnan(values.real) | np.isnan(values.imag)):
        raise ValueError(f"{name} must not contain NaN")
    if not allow_open_circuit and np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


@dataclass(frozen=True)
class ButterworthVanDyke:
    """Four-element Butterworth–Van Dyke terminal-impedance model.

    The static capacitance is in parallel with a series motional RLC branch.
    ``series_resistance_ohm`` represents leads, electrodes, and other series
    loss.  This circuit reproduces electrical resonance and antiresonance, but
    it does not by itself determine acoustic pressure in Pa/V.
    """

    static_capacitance_f: float
    motional_resistance_ohm: float
    motional_inductance_h: float
    motional_capacitance_f: float
    series_resistance_ohm: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "static_capacitance_f": self.static_capacitance_f,
            "motional_resistance_ohm": self.motional_resistance_ohm,
            "motional_inductance_h": self.motional_inductance_h,
            "motional_capacitance_f": self.motional_capacitance_f,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if (
            not np.isfinite(self.series_resistance_ohm)
            or self.series_resistance_ohm < 0.0
        ):
            raise ValueError("series_resistance_ohm must be finite and >= 0")

    @property
    def series_resonance_hz(self) -> float:
        """Return the undamped motional series-resonance frequency."""

        return float(
            1.0
            / (
                2.0
                * np.pi
                * np.sqrt(
                    self.motional_inductance_h * self.motional_capacitance_f
                )
            )
        )

    def impedance(
        self,
        frequency_hz: ArrayLike,
    ) -> NDArray[np.complex128]:
        """Return complex terminal impedance for nonnegative frequencies."""

        frequency = np.asarray(frequency_hz, dtype=float)
        if np.any(~np.isfinite(frequency)) or np.any(frequency < 0.0):
            raise ValueError("frequency_hz must be finite and >= 0")

        result = np.full(frequency.shape, complex(np.inf, 0.0), dtype=np.complex128)
        positive = frequency > 0.0
        omega = 2.0 * np.pi * frequency[positive]
        motional_impedance = (
            self.motional_resistance_ohm
            + 1j * omega * self.motional_inductance_h
            + 1.0 / (1j * omega * self.motional_capacitance_f)
        )
        parallel_admittance = (
            1j * omega * self.static_capacitance_f
            + 1.0 / motional_impedance
        )
        result[positive] = (
            self.series_resistance_ohm + 1.0 / parallel_admittance
        )
        return result


@dataclass(frozen=True)
class ElectricalDriveResult:
    """Voltage, current, impedance, and energy at the probe connector."""

    time_s: NDArray[np.float64]
    frequency_hz: NDArray[np.float64]
    source_voltage_v: NDArray[np.float64]
    terminal_voltage_v: NDArray[np.float64]
    terminal_current_a: NDArray[np.float64]
    source_voltage_spectrum_v: NDArray[np.complex128]
    terminal_voltage_spectrum_v: NDArray[np.complex128]
    terminal_current_spectrum_a: NDArray[np.complex128]
    source_impedance_ohm: NDArray[np.complex128]
    transducer_impedance_ohm: NDArray[np.complex128]
    delivered_energy_j: float

    @property
    def peak_terminal_voltage_v(self) -> float:
        return float(np.max(np.abs(self.terminal_voltage_v)))

    @property
    def peak_terminal_current_a(self) -> float:
        return float(np.max(np.abs(self.terminal_current_a)))


def solve_thevenin_drive(
    time_s: ArrayLike,
    source_voltage_v: ArrayLike,
    *,
    source_impedance_ohm: FrequencyResponse = 50.0,
    transducer_impedance_ohm: FrequencyResponse,
) -> ElectricalDriveResult:
    """Apply a Thevenin source to a frequency-dependent probe impedance.

    ``source_voltage_v`` is the open-circuit generator voltage at the probe
    connector reference plane.  Cable effects should therefore either be
    included in this measured waveform or in the supplied impedances.
    """

    time, source_voltage, delta_t = _validate_time_waveform(
        time_s,
        source_voltage_v,
        waveform_name="source_voltage_v",
    )
    frequency = np.fft.rfftfreq(time.size, d=delta_t)
    source_spectrum = np.fft.rfft(source_voltage).astype(np.complex128)
    source_impedance = _resolve_impedance(
        frequency,
        source_impedance_ohm,
        name="source_impedance_ohm",
        allow_open_circuit=False,
    )
    load_impedance = _resolve_impedance(
        frequency,
        transducer_impedance_ohm,
        name="transducer_impedance_ohm",
        allow_open_circuit=True,
    )

    open_load = np.isinf(np.abs(load_impedance))
    denominator = source_impedance + load_impedance
    loaded = ~open_load
    if np.any(np.abs(denominator[loaded]) <= np.finfo(float).tiny):
        raise ValueError("source and transducer impedances produce a zero denominator")

    voltage_ratio = np.ones_like(load_impedance)
    voltage_ratio[loaded] = load_impedance[loaded] / denominator[loaded]
    current_spectrum = np.zeros_like(source_spectrum)
    current_spectrum[loaded] = source_spectrum[loaded] / denominator[loaded]
    terminal_voltage_spectrum = source_spectrum * voltage_ratio
    terminal_voltage = np.fft.irfft(terminal_voltage_spectrum, n=time.size)
    terminal_current = np.fft.irfft(current_spectrum, n=time.size)
    delivered_energy = float(
        delta_t * np.dot(terminal_voltage, terminal_current)
    )

    return ElectricalDriveResult(
        time_s=time,
        frequency_hz=frequency,
        source_voltage_v=source_voltage,
        terminal_voltage_v=terminal_voltage,
        terminal_current_a=terminal_current,
        source_voltage_spectrum_v=source_spectrum,
        terminal_voltage_spectrum_v=terminal_voltage_spectrum,
        terminal_current_spectrum_a=current_spectrum,
        source_impedance_ohm=source_impedance,
        transducer_impedance_ohm=load_impedance,
        delivered_energy_j=delivered_energy,
    )


@dataclass(frozen=True)
class ElectroAcousticCalibration:
    """Separate transmit, receive, and receiver-chain calibration responses.

    ``transmit_pressure_pa_per_v`` maps terminal voltage to equivalent aperture
    pressure.  ``receive_voltage_v_per_pa`` maps the reciprocal aperture-
    projected return pressure to open receiver voltage.  ``receiver_response``
    includes loaded gain and filtering after the probe connector.  Set
    ``absolute=True`` only when the sensitivities have an absolute calibration.
    """

    transmit_pressure_pa_per_v: FrequencyResponse
    receive_voltage_v_per_pa: FrequencyResponse
    receiver_response: FrequencyResponse = 1.0
    adc_counts_per_v: float | None = None
    absolute: bool = False

    def __post_init__(self) -> None:
        if self.adc_counts_per_v is not None and (
            not np.isfinite(self.adc_counts_per_v)
            or self.adc_counts_per_v <= 0.0
        ):
            raise ValueError("adc_counts_per_v must be finite and > 0 or None")


@dataclass(frozen=True)
class ElectroAcousticPulseEchoResult:
    """Electrical, acoustic-pressure, receiver-voltage, and ADC results."""

    electrical: ElectricalDriveResult
    acoustic: PulseEchoResult
    aperture_pressure_pa: NDArray[np.float64]
    aperture_pressure_spectrum_pa: NDArray[np.complex128]
    returned_pressure_pa: NDArray[np.float64]
    received_voltage_v: NDArray[np.float64]
    plate_front_voltage_v: NDArray[np.float64]
    backing_voltage_v: NDArray[np.float64]
    receive_chain_response_v_per_pa: NDArray[np.complex128]
    received_adc_counts: NDArray[np.float64] | None
    absolute_calibration: bool

    @property
    def peak_aperture_pressure_pa(self) -> float:
        return float(np.max(np.abs(self.aperture_pressure_pa)))

    @property
    def peak_received_voltage_v(self) -> float:
        return float(np.max(np.abs(self.received_voltage_v)))


def simulate_electroacoustic_pulse_echo(
    model: AngularSpectrumModel,
    time_s: ArrayLike,
    source_voltage_v: ArrayLike,
    *,
    source_impedance_ohm: FrequencyResponse = 50.0,
    transducer_impedance_ohm: FrequencyResponse,
    calibration: ElectroAcousticCalibration,
    fluid_layer_thickness_m: float,
    backing_fluid: Fluid,
    relative_spectrum_threshold: float = 1.0e-3,
    minimum_frequency_hz: float = 0.5e6,
    maximum_frequency_hz: float | None = None,
) -> ElectroAcousticPulseEchoResult:
    """Simulate a calibrated voltage-driven monostatic pulse echo.

    Absolute Pa, V, and ADC values are meaningful only if the impedance,
    transmit sensitivity, receive sensitivity, receiver loading/gain, and ADC
    scale have all been measured consistently.  With ``absolute=False`` the
    same calculation remains useful for relative voltage and tone-length
    sweeps, but the numerical amplitude scale must be treated as provisional.
    """

    electrical = solve_thevenin_drive(
        time_s,
        source_voltage_v,
        source_impedance_ohm=source_impedance_ohm,
        transducer_impedance_ohm=transducer_impedance_ohm,
    )
    frequency = electrical.frequency_hz
    transmit_response = _resolve_response(
        frequency,
        calibration.transmit_pressure_pa_per_v,
        name="transmit_pressure_pa_per_v",
    )
    receive_response = _resolve_response(
        frequency,
        calibration.receive_voltage_v_per_pa,
        name="receive_voltage_v_per_pa",
    )
    receiver_response = _resolve_response(
        frequency,
        calibration.receiver_response,
        name="receiver_response",
    )

    aperture_pressure_spectrum = (
        electrical.terminal_voltage_spectrum_v * transmit_response
    )
    aperture_pressure = np.fft.irfft(
        aperture_pressure_spectrum,
        n=electrical.time_s.size,
    )
    normalized_aperture_drive = (
        aperture_pressure / model.aperture.pressure_amplitude_pa
    )
    acoustic = simulate_monostatic_pulse_echo(
        model,
        electrical.time_s,
        normalized_aperture_drive,
        fluid_layer_thickness_m=fluid_layer_thickness_m,
        backing_fluid=backing_fluid,
        relative_spectrum_threshold=relative_spectrum_threshold,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
    )

    receive_chain = receive_response * receiver_response
    front_pressure_spectrum = np.fft.rfft(acoustic.plate_front_signal)
    backing_pressure_spectrum = np.fft.rfft(acoustic.backing_signal)
    front_voltage_spectrum = front_pressure_spectrum * receive_chain
    backing_voltage_spectrum = backing_pressure_spectrum * receive_chain
    received_voltage_spectrum = front_voltage_spectrum + backing_voltage_spectrum
    plate_front_voltage = np.fft.irfft(
        front_voltage_spectrum,
        n=electrical.time_s.size,
    )
    backing_voltage = np.fft.irfft(
        backing_voltage_spectrum,
        n=electrical.time_s.size,
    )
    received_voltage = plate_front_voltage + backing_voltage
    received_adc = None
    if calibration.adc_counts_per_v is not None:
        received_adc = received_voltage * calibration.adc_counts_per_v

    return ElectroAcousticPulseEchoResult(
        electrical=electrical,
        acoustic=acoustic,
        aperture_pressure_pa=aperture_pressure,
        aperture_pressure_spectrum_pa=aperture_pressure_spectrum,
        returned_pressure_pa=acoustic.received_signal,
        received_voltage_v=received_voltage,
        plate_front_voltage_v=plate_front_voltage,
        backing_voltage_v=backing_voltage,
        receive_chain_response_v_per_pa=receive_chain,
        received_adc_counts=received_adc,
        absolute_calibration=calibration.absolute,
    )
