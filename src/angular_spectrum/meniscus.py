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
class MeniscusCavityMetric:
    """Reflection-induced exposure changes at one configured meniscus.

    The reference is the forward acoustic power crossing the meniscus plane on
    the *first* pass, after water, plate and liquid propagation, but before any
    wave reflected by the meniscus has returned from the plate.  The power is
    integrated over the transverse FFT plane using the modal normal
    admittance.  Consequently the ratios do not depend on the arbitrary source
    pressure amplitude.

    ``coherent_power_gain`` is the monochromatic steady-state forward-power
    ratio after coherently summing all retained cavity passes.  It can be above
    or below one because the returns interfere with the first pass.

    ``narrowband_separated_pass_exposure_gain`` is the sum of the power in the
    individual forward passes divided by the first-pass power, all evaluated
    at ``frequency_hz``.  It becomes a fluence ratio only for a sufficiently
    narrow-band pulse whose returned passes are separated in time (or have
    randomized relative phase), so cross terms vanish.  It is therefore a
    single-frequency exposure proxy, not the spectrally integrated fluence of
    the configured finite burst.  A value above one represents repeated visits
    of recirculating energy to the meniscus plane; it does not mean that the
    cavity creates energy or that this energy is transmitted into the backing
    fluid.

    ``electrical_overlap_regime`` compares the ideal electrical tone length
    with the nominal cavity round trip.  It cannot establish whether acoustic
    returns overlap because the probe's causal transmit ring-down is not
    calibrated by the magnitude-only certificate response.

    The height-sensitivity interval is a deterministic sweep over
    ``height_m +/- height_uncertainty_m``.  It is not a statistical confidence
    interval.
    """

    frequency_hz: float
    height_m: float
    height_uncertainty_m: float
    electrical_burst_duration_s: float
    cavity_round_trip_s: float
    electrical_overlap_regime: str
    first_pass_forward_power_w: float
    coherent_forward_power_w: float
    incoherent_forward_power_sum_w: float
    coherent_power_gain: float
    narrowband_separated_pass_exposure_gain: float
    coherent_gain_height_min: float
    coherent_gain_height_max: float
    narrowband_gain_height_min: float
    narrowband_gain_height_max: float
    cavity_orders_retained: int
    cavity_series_converged: bool
    sensitivity_series_converged: bool
    cavity_relative_tolerance: float
    limitations: tuple[str, ...]

    @property
    def coherent_percent_change(self) -> float:
        """CW forward-power change relative to the first pass."""

        return 100.0 * (self.coherent_power_gain - 1.0)

    @property
    def narrowband_separated_pass_percent_change(self) -> float:
        """Single-frequency separated-pass exposure change."""

        return 100.0 * (self.narrowband_separated_pass_exposure_gain - 1.0)

    @property
    def coherent_gain_db(self) -> float:
        """CW forward-power gain in decibels."""

        return float(10.0 * np.log10(self.coherent_power_gain))

    @property
    def narrowband_separated_pass_gain_db(self) -> float:
        """Single-frequency separated-pass exposure gain in decibels."""

        return float(
            10.0 * np.log10(self.narrowband_separated_pass_exposure_gain)
        )

    @property
    def electrical_burst_to_round_trip_ratio(self) -> float:
        return self.electrical_burst_duration_s / self.cavity_round_trip_s


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


@dataclass(frozen=True)
class _CavitySpectrumResult:
    coherent_spectrum: NDArray[np.complex128]
    incoherent_forward_power_w: float
    first_pass_forward_power_w: float
    orders_retained: int
    converged: bool


def _forward_power_w(
    pressure_spectrum: NDArray[np.complex128],
    vertical_wavenumber: NDArray[np.complex128],
    *,
    angular_frequency_rad_s: float,
    density_kg_m3: float,
    sample_area_m2: float,
) -> float:
    """Return plane-integrated forward normal power from an FFT spectrum."""

    sample_count = pressure_spectrum.size
    modal_admittance = np.real(
        vertical_wavenumber / (angular_frequency_rad_s * density_kg_m3)
    )
    # Parseval for NumPy's unnormalised FFT and normalised inverse FFT:
    # sum_x p v* = sum_k P V* / N.  Evanescent modes have zero real normal
    # admittance in a lossless fluid and therefore carry no forward power.
    power = (
        0.5
        * sample_area_m2
        / sample_count
        * np.sum(modal_admittance * np.abs(pressure_spectrum) ** 2)
    )
    return float(np.real(power))


def _sum_forward_cavity_orders(
    model: AngularSpectrumModel,
    frequency_hz: float,
    layer_height_m: float,
    *,
    launch_spectrum: NDArray[np.complex128],
    interface_loop: NDArray[np.complex128],
    layer_vertical_wavenumber: NDArray[np.complex128],
    cavity_relative_tolerance: float,
    maximum_cavity_orders: int,
    accumulate_incoherent_power: bool = False,
) -> _CavitySpectrumResult:
    """Sum forward meniscus passes with order-specific propagation support."""

    coherent_spectrum = np.zeros_like(launch_spectrum)
    interface_power = np.ones_like(interface_loop)
    incoherent_power_w = 0.0
    first_pass_power_w = np.nan
    base_norm = np.nan
    converged = False
    orders_retained = 0
    omega = 2.0 * np.pi * frequency_hz
    for order in range(maximum_cavity_orders):
        total_layer_distance_m = (2.0 * order + 1.0) * layer_height_m
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
                    (model.transmitted_fluid, total_layer_distance_m),
                ),
            )
        )
        if order == 0:
            base_norm = max(float(np.linalg.norm(order_spectrum)), 1e-300)
        if accumulate_incoherent_power:
            order_power_w = _forward_power_w(
                order_spectrum,
                layer_vertical_wavenumber,
                angular_frequency_rad_s=omega,
                density_kg_m3=model.transmitted_fluid.density_kg_m3,
                sample_area_m2=model.grid.dx_m * model.grid.dy_m,
            )
            if order == 0:
                first_pass_power_w = order_power_w
            incoherent_power_w += order_power_w
        coherent_spectrum += order_spectrum
        orders_retained = order + 1
        if (
            order >= 1
            and float(np.linalg.norm(order_spectrum))
            <= cavity_relative_tolerance * base_norm
        ):
            converged = True
            break
        interface_power *= interface_loop

    return _CavitySpectrumResult(
        coherent_spectrum=coherent_spectrum,
        incoherent_forward_power_w=float(incoherent_power_w),
        first_pass_forward_power_w=float(first_pass_power_w),
        orders_retained=orders_retained,
        converged=converged,
    )


def configured_meniscus_cavity_metric(
    model: AngularSpectrumModel,
    frequency_hz: float,
    height_m: float,
    *,
    backing_fluid: Fluid,
    excitation_cycles: float,
    height_uncertainty_m: float = 0.0,
    height_sensitivity_samples: int = 9,
    cavity_relative_tolerance: float = 1.0e-8,
    maximum_cavity_orders: int = 256,
) -> MeniscusCavityMetric:
    """Compare configured meniscus exposure with the first forward pass.

    This calculation uses the configured focused-aperture phase, rather than
    re-optimising the aperture for the meniscus.  It returns two intentionally
    different limits:

    * coherent monochromatic forward power, appropriate to a settled CW field;
    * a single-frequency separated-pass exposure proxy.  It can approximate
      a fluence ratio only for a sufficiently narrow-band waveform with
      non-overlapping acoustic returns.

    Neither quantity is the near-zero net power transmitted through a
    liquid-air pressure-release boundary.  See :class:`MeniscusCavityMetric`
    for the precise baseline and interpretation.
    """

    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and > 0")
    if not np.isfinite(height_m) or height_m <= 0.0:
        raise ValueError("height_m must be finite and > 0")
    if not np.isfinite(excitation_cycles) or excitation_cycles <= 0.0:
        raise ValueError("excitation_cycles must be finite and > 0")
    if (
        not np.isfinite(height_uncertainty_m)
        or height_uncertainty_m < 0.0
        or height_uncertainty_m >= height_m
    ):
        raise ValueError(
            "height_uncertainty_m must be finite, >= 0 and smaller than height_m"
        )
    if (
        isinstance(height_sensitivity_samples, bool)
        or not isinstance(height_sensitivity_samples, (int, np.integer))
        or height_sensitivity_samples < 3
        or height_sensitivity_samples % 2 == 0
    ):
        raise ValueError(
            "height_sensitivity_samples must be an odd integer >= 3"
        )
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

    maximum_height_m = height_m + height_uncertainty_m
    validate_focused_grid_support(
        model,
        maximum_frequency_hz=frequency_hz,
        propagation_segments=(
            ("one-way water path", model.incident_fluid, model.water_path_m),
            (
                "liquid-layer cavity round trip",
                model.transmitted_fluid,
                3.0 * maximum_height_m,
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
    launch_spectrum = (
        model.source_spectrum(frequency_hz)
        * model._propagator(
            model.incident_fluid,
            frequency_hz,
            model.water_path_m,
        )
        * plate_transmission
    )
    interface_loop = reverse_reflection * meniscus_reflection
    layer_kz = vertical_wavenumber(
        model.transmitted_fluid.wavenumber(frequency_hz), q
    )

    if height_uncertainty_m == 0.0:
        heights = np.array([height_m], dtype=float)
    else:
        heights = np.linspace(
            height_m - height_uncertainty_m,
            height_m + height_uncertainty_m,
            height_sensitivity_samples,
        )

    evaluations: list[_CavitySpectrumResult] = []
    coherent_powers: list[float] = []
    coherent_gains: list[float] = []
    separated_gains: list[float] = []
    omega = 2.0 * np.pi * frequency_hz
    for evaluated_height_m in heights:
        evaluation = _sum_forward_cavity_orders(
            model,
            frequency_hz,
            float(evaluated_height_m),
            launch_spectrum=launch_spectrum,
            interface_loop=interface_loop,
            layer_vertical_wavenumber=layer_kz,
            cavity_relative_tolerance=cavity_relative_tolerance,
            maximum_cavity_orders=maximum_cavity_orders,
            accumulate_incoherent_power=True,
        )
        coherent_power_w = _forward_power_w(
            evaluation.coherent_spectrum,
            layer_kz,
            angular_frequency_rad_s=omega,
            density_kg_m3=model.transmitted_fluid.density_kg_m3,
            sample_area_m2=model.grid.dx_m * model.grid.dy_m,
        )
        baseline_w = evaluation.first_pass_forward_power_w
        if not np.isfinite(baseline_w) or baseline_w <= 0.0:
            raise RuntimeError(
                "the first-pass forward meniscus power is not positive; "
                "check grid support and material parameters"
            )
        evaluations.append(evaluation)
        coherent_powers.append(coherent_power_w)
        coherent_gains.append(coherent_power_w / baseline_w)
        separated_gains.append(
            evaluation.incoherent_forward_power_w / baseline_w
        )

    nominal_index = heights.size // 2
    nominal = evaluations[nominal_index]
    coherent_gain = float(coherent_gains[nominal_index])
    separated_gain = float(separated_gains[nominal_index])
    if (
        not np.isfinite(coherent_gain)
        or coherent_gain <= 0.0
        or not np.isfinite(separated_gain)
        or separated_gain < 1.0 - 1.0e-12
    ):
        raise RuntimeError("the cavity gain calculation produced invalid power")

    electrical_burst_duration_s = excitation_cycles / frequency_hz
    cavity_round_trip_s = (
        2.0 * height_m / model.transmitted_fluid.sound_speed_m_s
    )
    duration_ratio = electrical_burst_duration_s / cavity_round_trip_s
    if duration_ratio < 1.0:
        electrical_overlap_regime = "electrical-burst-shorter"
    elif duration_ratio < 5.0:
        electrical_overlap_regime = "electrical-burst-intermediate"
    else:
        electrical_overlap_regime = "electrical-burst-long"

    limitations = (
        "Forward exposure on the liquid side of the meniscus; not net energy "
        "transmitted into air and not an ejection-efficiency prediction.",
        "The separated-pass value is evaluated only at the stated frequency; "
        "it is not the spectrally integrated fluence of the configured pulse.",
        "Electrical burst duration alone cannot determine acoustic overlap; "
        "uncalibrated electro-acoustic ring-down can overlap later passes.",
        "The coherent value is a single-frequency steady-state limit and is "
        "not the energy gain of a one-cycle pulse.",
        "Linear acoustics, a planar parallel meniscus and the configured "
        "material losses are assumed; curvature, tilt, cavitation, nonlinear "
        "propagation and unmeasured losses are excluded.",
        "The fill-height range is a deterministic sensitivity interval, not "
        "a statistical confidence interval.",
    )
    return MeniscusCavityMetric(
        frequency_hz=float(frequency_hz),
        height_m=float(height_m),
        height_uncertainty_m=float(height_uncertainty_m),
        electrical_burst_duration_s=float(electrical_burst_duration_s),
        cavity_round_trip_s=float(cavity_round_trip_s),
        electrical_overlap_regime=electrical_overlap_regime,
        first_pass_forward_power_w=float(
            nominal.first_pass_forward_power_w
        ),
        coherent_forward_power_w=float(coherent_powers[nominal_index]),
        incoherent_forward_power_sum_w=float(
            nominal.incoherent_forward_power_w
        ),
        coherent_power_gain=coherent_gain,
        narrowband_separated_pass_exposure_gain=separated_gain,
        coherent_gain_height_min=float(np.min(coherent_gains)),
        coherent_gain_height_max=float(np.max(coherent_gains)),
        narrowband_gain_height_min=float(np.min(separated_gains)),
        narrowband_gain_height_max=float(np.max(separated_gains)),
        cavity_orders_retained=nominal.orders_retained,
        cavity_series_converged=nominal.converged,
        sensitivity_series_converged=all(
            evaluation.converged for evaluation in evaluations
        ),
        cavity_relative_tolerance=float(cavity_relative_tolerance),
        limitations=limitations,
    )


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

        launch_spectrum = (
            aperture_spectrum * water_propagation * plate_transmission
        )
        interface_loop = reverse_reflection * meniscus_reflection
        cavity = _sum_forward_cavity_orders(
            model,
            frequency_hz,
            float(layer_height),
            launch_spectrum=launch_spectrum,
            interface_loop=interface_loop,
            layer_vertical_wavenumber=layer_kz,
            cavity_relative_tolerance=cavity_relative_tolerance,
            maximum_cavity_orders=maximum_cavity_orders,
        )
        reverberant_spectrum = cavity.coherent_spectrum
        cavity_orders_retained[index] = cavity.orders_retained
        cavity_series_converged[index] = cavity.converged
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
