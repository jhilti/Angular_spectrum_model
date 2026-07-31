"""Conservative calibration helpers for qualitative survey comparisons.

The functions in this module do not turn ADC counts into pressure.  They use a
measured water--plate echo only as an in-situ reference for the *shape* of the
combined pulser, transducer, receiver and ADC response.  Geometry and material
properties remain independent model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ReferenceTransferCalibration:
    """Regularized complex correction derived from a reference echo."""

    frequency_hz: NDArray[np.float64]
    response: NDArray[np.complex128]
    gate_start_s: float
    gate_end_s: float
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    regularization: float
    maximum_correction_db: float

    @property
    def magnitude_correction_db(self) -> NDArray[np.float64]:
        return 20.0 * np.log10(np.maximum(np.abs(self.response), 1e-12))


def _uniform_time_signal(
    name: str,
    time_s: ArrayLike,
    signal: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    time = np.asarray(time_s, dtype=float)
    values = np.asarray(signal, dtype=float)
    if time.ndim != 1 or values.ndim != 1 or time.size != values.size:
        raise ValueError(f"{name} time and signal must be equal 1D arrays")
    if time.size < 16 or np.any(~np.isfinite(time)) or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} time and signal must contain finite samples")
    intervals = np.diff(time)
    if np.any(intervals <= 0.0) or not np.allclose(
        intervals,
        intervals[0],
        rtol=1e-7,
        atol=0.0,
    ):
        raise ValueError(f"{name} time must be strictly increasing and uniform")
    return time, values, float(intervals[0])


def _smooth(values: NDArray[np.float64], sample_count: int) -> NDArray[np.float64]:
    count = max(1, int(sample_count))
    if count % 2 == 0:
        count += 1
    if count <= 1 or values.size < count:
        return values.copy()
    window = np.hanning(count)
    window /= float(np.sum(window))
    padding = count // 2
    padded = np.pad(values, padding, mode="edge")
    return np.convolve(padded, window, mode="valid")


def estimate_reference_transfer(
    measured_time_s: ArrayLike,
    measured_signal: ArrayLike,
    simulated_time_s: ArrayLike,
    simulated_signal: ArrayLike,
    *,
    measured_arrival_s: float,
    simulated_arrival_s: float,
    target_time_s: ArrayLike,
    gate_start_s: float = -0.22e-6,
    gate_end_s: float = 0.38e-6,
    minimum_frequency_hz: float = 2.0e6,
    maximum_frequency_hz: float = 18.0e6,
    regularization: float = 0.02,
    maximum_correction_db: float = 18.0,
    smoothing_bins: int = 9,
) -> ReferenceTransferCalibration:
    """Estimate a bounded common-system correction from one reference echo.

    Both echoes are placed on the same local time axis before a Hann gate is
    applied.  A regularized spectral ratio is then smoothed and limited in
    magnitude.  The result captures common waveform and group-delay effects,
    but deliberately excludes geometry, absolute gain, and echo-specific loss.
    """

    measured_time, measured, measured_dt = _uniform_time_signal(
        "measured", measured_time_s, measured_signal
    )
    simulated_time, simulated, simulated_dt = _uniform_time_signal(
        "simulated", simulated_time_s, simulated_signal
    )
    target_time = np.asarray(target_time_s, dtype=float)
    _, _, target_dt = _uniform_time_signal(
        "target", target_time, np.zeros_like(target_time)
    )
    for name, value in {
        "measured_arrival_s": measured_arrival_s,
        "simulated_arrival_s": simulated_arrival_s,
        "gate_start_s": gate_start_s,
        "gate_end_s": gate_end_s,
        "minimum_frequency_hz": minimum_frequency_hz,
        "maximum_frequency_hz": maximum_frequency_hz,
        "regularization": regularization,
        "maximum_correction_db": maximum_correction_db,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if gate_end_s <= gate_start_s:
        raise ValueError("gate_end_s must be greater than gate_start_s")
    if minimum_frequency_hz < 0.0 or maximum_frequency_hz <= minimum_frequency_hz:
        raise ValueError("calibration frequency limits are invalid")
    if not 0.0 < regularization <= 1.0:
        raise ValueError("regularization must lie in (0, 1]")
    if maximum_correction_db <= 0.0:
        raise ValueError("maximum_correction_db must be > 0")

    common_dt = max(measured_dt, simulated_dt)
    local_time = np.arange(
        gate_start_s,
        gate_end_s + 0.5 * common_dt,
        common_dt,
    )
    if local_time.size < 16:
        raise ValueError("reference gate contains too few samples")
    measured_gate = np.interp(
        measured_arrival_s + local_time,
        measured_time,
        measured,
        left=0.0,
        right=0.0,
    )
    simulated_gate = np.interp(
        simulated_arrival_s + local_time,
        simulated_time,
        simulated,
        left=0.0,
        right=0.0,
    )
    window = np.hanning(local_time.size)
    measured_gate = (measured_gate - float(np.mean(measured_gate))) * window
    simulated_gate = (simulated_gate - float(np.mean(simulated_gate))) * window
    measured_norm = float(np.linalg.norm(measured_gate))
    simulated_norm = float(np.linalg.norm(simulated_gate))
    if measured_norm <= 1e-30 or simulated_norm <= 1e-30:
        raise ValueError("reference gate has zero usable signal")
    measured_gate /= measured_norm
    simulated_gate /= simulated_norm

    minimum_fft_size = local_time.size * 32
    fft_size = max(2048, 1 << int(np.ceil(np.log2(minimum_fft_size))))
    calibration_frequency = np.fft.rfftfreq(fft_size, common_dt)
    measured_spectrum = np.fft.rfft(measured_gate, fft_size)
    simulated_spectrum = np.fft.rfft(simulated_gate, fft_size)
    simulated_power = np.abs(simulated_spectrum) ** 2
    power_scale = max(float(np.max(simulated_power)), 1e-30)
    ratio = (
        measured_spectrum
        * np.conj(simulated_spectrum)
        / (simulated_power + regularization * power_scale)
    )
    support = (
        (calibration_frequency >= minimum_frequency_hz)
        & (calibration_frequency <= maximum_frequency_hz)
        & (np.abs(simulated_spectrum) >= 0.015 * np.sqrt(power_scale))
        & (
            np.abs(measured_spectrum)
            >= 0.005 * max(float(np.max(np.abs(measured_spectrum))), 1e-30)
        )
    )
    indices = np.flatnonzero(support)
    if indices.size < 8:
        raise ValueError("reference echo has insufficient calibration bandwidth")

    supported_frequency = calibration_frequency[indices]
    log_magnitude = np.log(np.maximum(np.abs(ratio[indices]), 1e-12))
    phase = np.unwrap(np.angle(ratio[indices]))
    log_magnitude = _smooth(log_magnitude, smoothing_bins)
    phase = _smooth(phase, smoothing_bins)
    reference_index = int(np.argmax(np.abs(simulated_spectrum[indices])))
    log_magnitude -= log_magnitude[reference_index]
    maximum_log = maximum_correction_db * np.log(10.0) / 20.0
    log_magnitude = np.clip(log_magnitude, -maximum_log, maximum_log)

    target_frequency = np.fft.rfftfreq(target_time.size, target_dt)
    target_log_magnitude = np.zeros_like(target_frequency)
    target_phase = np.zeros_like(target_frequency)
    inside = (
        (target_frequency >= supported_frequency[0])
        & (target_frequency <= supported_frequency[-1])
    )
    target_log_magnitude[inside] = np.interp(
        target_frequency[inside],
        supported_frequency,
        log_magnitude,
    )
    target_phase[inside] = np.interp(
        target_frequency[inside],
        supported_frequency,
        phase,
    )
    transition_hz = min(
        0.75e6,
        0.1 * (supported_frequency[-1] - supported_frequency[0]),
    )
    if transition_hz > 0.0:
        blend = np.zeros_like(target_frequency)
        blend[inside] = 1.0
        lower_transition = inside & (
            target_frequency < supported_frequency[0] + transition_hz
        )
        upper_transition = inside & (
            target_frequency > supported_frequency[-1] - transition_hz
        )
        lower_fraction = (
            target_frequency[lower_transition] - supported_frequency[0]
        ) / transition_hz
        upper_fraction = (
            supported_frequency[-1] - target_frequency[upper_transition]
        ) / transition_hz
        blend[lower_transition] = 0.5 - 0.5 * np.cos(
            np.pi * lower_fraction
        )
        blend[upper_transition] = 0.5 - 0.5 * np.cos(
            np.pi * upper_fraction
        )
        target_log_magnitude *= blend
        target_phase *= blend
    response = np.exp(target_log_magnitude + 1j * target_phase)
    response[~inside] = 1.0 + 0.0j
    return ReferenceTransferCalibration(
        frequency_hz=target_frequency,
        response=response.astype(np.complex128),
        gate_start_s=gate_start_s,
        gate_end_s=gate_end_s,
        minimum_frequency_hz=float(supported_frequency[0]),
        maximum_frequency_hz=float(supported_frequency[-1]),
        regularization=regularization,
        maximum_correction_db=maximum_correction_db,
    )


def apply_reference_transfer(
    time_s: ArrayLike,
    signal: ArrayLike,
    calibration: ReferenceTransferCalibration,
) -> NDArray[np.float64]:
    """Apply a reference calibration to another signal on the same time grid."""

    time, values, interval_s = _uniform_time_signal("signal", time_s, signal)
    frequency = np.fft.rfftfreq(time.size, interval_s)
    if calibration.response.shape != frequency.shape or not np.allclose(
        calibration.frequency_hz,
        frequency,
        rtol=1e-8,
        atol=1e-6,
    ):
        raise ValueError("calibration does not match the signal time grid")
    return np.fft.irfft(
        np.fft.rfft(values) * calibration.response,
        n=time.size,
    ).astype(float)
