"""Focused intensity at a fluid layer terminated by another fluid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .materials import Fluid
from .model import AngularSpectrumModel, validate_focused_grid_support
from .plate import (
    elastic_plate_scattering_map,
    fluid_interface_scattering,
    vertical_wavenumber,
)


@dataclass(frozen=True)
class MeniscusSweepResult:
    """Optimally focused fields versus fluid-layer height.

    Each height uses its own phase-conjugate aperture phase that maximizes
    single-pass on-axis pressure amplitude. The reported intensity is then
    evaluated for that phase; it is not a mathematically strict intensity
    maximum because modal admittance varies across the angular spectrum.

    ``total_interface_*`` contains the on-axis total field immediately on the
    layer side of the final interface.  It includes both the forward cavity
    field and the wave reflected by the meniscus.  The normal direction points
    from the layer into ``backing_fluid``.
    """

    height_m: NDArray[np.float64]
    intensity_with_reverberation_w_m2: NDArray[np.float64]
    single_pass_intensity_w_m2: NDArray[np.float64]
    interference_gain: NDArray[np.float64]
    total_interface_pressure_pa: NDArray[np.complex128]
    total_interface_normal_velocity_m_s: NDArray[np.complex128]
    total_interface_normal_displacement_m: NDArray[np.complex128]
    cavity_orders_retained: NDArray[np.int64]
    cavity_series_converged: NDArray[np.bool_]
    cavity_relative_tolerance: float
    pressure_phase_optimized_independently_per_height: bool = True
    is_phase_only_pressure_upper_bound: bool = True

    @property
    def phase_optimized_independently_per_height(self) -> bool:
        """Backward-compatible alias for pressure-phase optimization."""

        return self.pressure_phase_optimized_independently_per_height

    @property
    def is_ideal_phase_conjugate_upper_bound(self) -> bool:
        """Backward-compatible alias; the bound applies to pressure only."""

        return self.is_phase_only_pressure_upper_bound


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
    cavity_relative_tolerance: float = 1.0e-8,
    maximum_cavity_orders: int = 256,
) -> MeniscusSweepResult:
    """Sweep the forward intensity immediately below a planar meniscus.

    For every height, the aperture phase is independently optimized for the
    single-pass pressure at the meniscus. This is a phase-only upper bound for
    on-axis pressure amplitude, not a strict intensity upper bound, and it is
    not a height scan using one fixed transducer phase.
    The same optimized phase profile is then evaluated with all reverberations
    between the plate and the final fluid interface.  Their ratio isolates the
    constructive/destructive interference from focusing.

    The result is the forward-wave axial intensity just inside the layer.  It
    is not the near-zero net power transmitted into a pressure-release air
    boundary. The local cycle-averaged axial flux is retained with its sign;
    structured-field interference can produce local backflow.
    """

    height = np.asarray(tuple(heights_m), dtype=float)
    if height.ndim != 1 or height.size == 0:
        raise ValueError("heights_m must be a non-empty 1D sequence")
    if np.any(~np.isfinite(height)) or np.any(height <= 0.0):
        raise ValueError("heights_m must be finite and > 0")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and > 0")
    if (
        not np.isfinite(cavity_relative_tolerance)
        or not 0.0 < cavity_relative_tolerance < 1.0
    ):
        raise ValueError("cavity_relative_tolerance must lie in (0, 1)")
    if (
        isinstance(maximum_cavity_orders, bool)
        or not isinstance(maximum_cavity_orders, (int, np.integer))
        or maximum_cavity_orders < 1
    ):
        raise ValueError("maximum_cavity_orders must be an integer >= 1")

    validate_focused_grid_support(
        model,
        maximum_frequency_hz=frequency_hz,
        propagation_segments=(
            (
                "one-way water path",
                model.incident_fluid,
                model.water_path_m,
            ),
            (
                "liquid-layer cavity round trip",
                model.transmitted_fluid,
                3.0 * float(np.max(height)),
            ),
        ),
    )

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
    aperture_amplitude = np.abs(model.source_pressure(frequency_hz))
    sample_count = model.grid.nx * model.grid.ny
    omega = 2.0 * np.pi * frequency_hz

    single_pass_intensity = np.empty(height.size, dtype=float)
    reverberant_intensity = np.empty(height.size, dtype=float)
    total_interface_pressure = np.empty(height.size, dtype=np.complex128)
    total_interface_velocity = np.empty(height.size, dtype=np.complex128)
    total_interface_displacement = np.empty(
        height.size, dtype=np.complex128
    )
    cavity_orders_retained = np.empty(height.size, dtype=np.int64)
    cavity_series_converged = np.zeros(height.size, dtype=bool)
    for index, layer_height in enumerate(height):
        layer_propagation = model._propagator(
            model.transmitted_fluid,
            frequency_hz,
            float(layer_height),
        )
        single_pass_transfer = (
            water_propagation * plate_transmission * layer_propagation
        )
        single_pass_transfer *= model._combined_bandlimit_mask(
            frequency_hz,
            (
                (model.incident_fluid, model.water_path_m),
                (model.transmitted_fluid, float(layer_height)),
            ),
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
        single_pass_intensity[index] = (
            0.5 * float(np.real(pressure * np.conj(velocity)))
        )

        # Sum cavity orders with their actual total path. A geometric
        # denominator would reuse the one-round-trip BLAS mask for every later
        # order and thereby wrap high-angle energy on the finite FFT window.
        launch_spectrum = (
            aperture_spectrum * water_propagation * plate_transmission
        )
        interface_loop = reverse_reflection * meniscus_reflection
        interface_power = np.ones_like(interface_loop)
        reverberant_spectrum = np.zeros_like(single_pass_spectrum)
        base_norm = max(float(np.linalg.norm(single_pass_spectrum)), 1e-300)
        for order in range(maximum_cavity_orders):
            total_layer_distance_m = (
                (2.0 * order + 1.0) * float(layer_height)
            )
            order_spectrum = (
                launch_spectrum
                * interface_power
                * model._propagator(
                    model.transmitted_fluid,
                    frequency_hz,
                    total_layer_distance_m,
                )
                * model._combined_bandlimit_mask(
                    frequency_hz,
                    (
                        (model.incident_fluid, model.water_path_m),
                        (
                            model.transmitted_fluid,
                            total_layer_distance_m,
                        ),
                    ),
                )
            )
            reverberant_spectrum += order_spectrum
            cavity_orders_retained[index] = order + 1
            if (
                order >= 1
                and float(np.linalg.norm(order_spectrum))
                <= cavity_relative_tolerance * base_norm
            ):
                cavity_series_converged[index] = True
                break
            interface_power *= interface_loop
        pressure, velocity = _on_axis_pressure_and_velocity(
            reverberant_spectrum,
            centre_phase,
            layer_kz,
            angular_frequency_rad_s=omega,
            density_kg_m3=model.transmitted_fluid.density_kg_m3,
        )
        reverberant_intensity[index] = (
            0.5 * float(np.real(pressure * np.conj(velocity)))
        )

        # At the meniscus, pressure adds while the normal velocity of the
        # reflected (-z) wave has the opposite sign.  These are the total
        # fluid-side fields, including the complete plate/meniscus cavity sum.
        total_pressure_spectrum = reverberant_spectrum * (
            1.0 + meniscus_reflection
        )
        total_velocity_spectrum = (
            layer_kz
            / (omega * model.transmitted_fluid.density_kg_m3)
            * reverberant_spectrum
            * (1.0 - meniscus_reflection)
        )
        total_interface_pressure[index] = (
            np.sum(total_pressure_spectrum * centre_phase) / sample_count
        )
        total_interface_velocity[index] = (
            np.sum(total_velocity_spectrum * centre_phase) / sample_count
        )
        total_interface_displacement[index] = (
            total_interface_velocity[index] / (-1j * omega)
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        gain = reverberant_intensity / single_pass_intensity
    return MeniscusSweepResult(
        height_m=height,
        intensity_with_reverberation_w_m2=reverberant_intensity,
        single_pass_intensity_w_m2=single_pass_intensity,
        interference_gain=gain,
        total_interface_pressure_pa=total_interface_pressure,
        total_interface_normal_velocity_m_s=total_interface_velocity,
        total_interface_normal_displacement_m=total_interface_displacement,
        cavity_orders_retained=cavity_orders_retained,
        cavity_series_converged=cavity_series_converged,
        cavity_relative_tolerance=float(cavity_relative_tolerance),
    )
