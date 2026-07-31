"""Focused intensity at a fluid layer terminated by another fluid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .materials import Fluid
from .model import AngularSpectrumModel
from .plate import (
    elastic_plate_scattering_map,
    fluid_interface_scattering,
    vertical_wavenumber,
)


@dataclass(frozen=True)
class MeniscusSweepResult:
    """Optimally focused forward intensity versus fluid-layer height."""

    height_m: NDArray[np.float64]
    intensity_with_reverberation_w_m2: NDArray[np.float64]
    single_pass_intensity_w_m2: NDArray[np.float64]
    interference_gain: NDArray[np.float64]


def _on_axis_pressure_and_velocity(
    pressure_spectrum: NDArray[np.complex128],
    centre_phase: NDArray[np.complex128],
    vertical_wavenumber: NDArray[np.complex128],
    *,
    angular_frequency_rad_s: float,
    density_kg_m3: float,
) -> tuple[complex, complex]:
    normalization = pressure_spectrum.size
    pressure = np.sum(pressure_spectrum * centre_phase) / normalization
    velocity_spectrum = (
        vertical_wavenumber
        / (angular_frequency_rad_s * density_kg_m3)
        * pressure_spectrum
    )
    velocity = np.sum(velocity_spectrum * centre_phase) / normalization
    return complex(pressure), complex(velocity)


def optimal_meniscus_intensity_sweep(
    model: AngularSpectrumModel,
    frequency_hz: float,
    heights_m: Iterable[float],
    *,
    backing_fluid: Fluid,
) -> MeniscusSweepResult:
    """Sweep the forward intensity immediately below a planar meniscus.

    For every height, the aperture phase is optimized for the single-pass
    pressure at the meniscus.  The same phase profile is then evaluated with
    all reverberations between the plate and the final fluid interface.  Their
    ratio isolates the constructive/destructive interference from focusing.

    The result is the forward-wave axial intensity just inside the layer.  It
    is not the near-zero net power transmitted into a pressure-release air
    boundary.
    """

    height = np.asarray(tuple(heights_m), dtype=float)
    if height.ndim != 1 or height.size == 0:
        raise ValueError("heights_m must be a non-empty 1D sequence")
    if np.any(~np.isfinite(height)) or np.any(height <= 0.0):
        raise ValueError("heights_m must be finite and > 0")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and > 0")

    _, _, q = model.grid.spectral_mesh()
    radial_samples = model.plate_radial_samples
    _, plate_transmission = elastic_plate_scattering_map(
        q,
        frequency_hz,
        model.incident_fluid,
        model.plate,
        model.transmitted_fluid,
        radial_samples=radial_samples,
    )
    reverse_reflection, _ = elastic_plate_scattering_map(
        q,
        frequency_hz,
        model.transmitted_fluid,
        model.plate,
        model.incident_fluid,
        radial_samples=radial_samples,
    )
    meniscus_reflection = fluid_interface_scattering(
        q,
        frequency_hz,
        model.transmitted_fluid,
        backing_fluid,
    )[0]
    water_propagation = model._propagator(
        model.incident_fluid,
        frequency_hz,
        model.water_path_m,
    )
    layer_kz = vertical_wavenumber(
        model.transmitted_fluid.wavenumber(frequency_hz), q
    )
    centre_phase = model.grid.centre_ifft_phase()
    aperture_amplitude = (
        np.abs(model.source_pressure(frequency_hz))
        / model.aperture.pressure_amplitude_pa
    )
    sample_count = model.grid.nx * model.grid.ny
    omega = 2.0 * np.pi * frequency_hz

    single_pass_intensity = np.empty(height.size, dtype=float)
    reverberant_intensity = np.empty(height.size, dtype=float)
    for index, layer_height in enumerate(height):
        layer_propagation = model._propagator(
            model.transmitted_fluid,
            frequency_hz,
            float(layer_height),
        )
        single_pass_transfer = (
            water_propagation * plate_transmission * layer_propagation
        )

        # The on-axis pressure is a linear functional of the aperture field.
        # Back-propagating that functional yields the phase-only optimum.
        focusing_kernel = np.fft.fft2(
            single_pass_transfer * centre_phase
        ) / sample_count
        aperture_pressure = aperture_amplitude * np.exp(
            -1j * np.angle(focusing_kernel)
        )
        aperture_spectrum = np.fft.fft2(aperture_pressure)

        single_pass_spectrum = aperture_spectrum * single_pass_transfer
        pressure, velocity = _on_axis_pressure_and_velocity(
            single_pass_spectrum,
            centre_phase,
            layer_kz,
            angular_frequency_rad_s=omega,
            density_kg_m3=model.transmitted_fluid.density_kg_m3,
        )
        single_pass_intensity[index] = max(
            0.5 * float(np.real(pressure * np.conj(velocity))),
            0.0,
        )

        cavity_loop = (
            reverse_reflection
            * meniscus_reflection
            * layer_propagation**2
        )
        reverberant_spectrum = (
            single_pass_spectrum / (1.0 - cavity_loop)
        )
        pressure, velocity = _on_axis_pressure_and_velocity(
            reverberant_spectrum,
            centre_phase,
            layer_kz,
            angular_frequency_rad_s=omega,
            density_kg_m3=model.transmitted_fluid.density_kg_m3,
        )
        reverberant_intensity[index] = max(
            0.5 * float(np.real(pressure * np.conj(velocity))),
            0.0,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        gain = reverberant_intensity / single_pass_intensity
    return MeniscusSweepResult(
        height_m=height,
        intensity_with_reverberation_w_m2=reverberant_intensity,
        single_pass_intensity_w_m2=single_pass_intensity,
        interference_gain=gain,
    )
