"""Optional axisymmetric transient free-surface response model.

This module is deliberately separate from the angular-spectrum and Streamlit
paths.  It models the *sub-threshold* deformation of a single-valued liquid
surface in a circular well.  The acoustic input is a slowly varying normal
radiation stress, not the raw MHz carrier and not the nearly-zero total
first-order pressure at a liquid--air pressure-release boundary.

The dynamic surface perturbation is expanded in volume-conserving radial
finite-element modes.  Each mode follows the finite-depth capillary--gravity
dispersion relation with the weak-viscosity decay rate ``2 nu k**2``::

    q_ddot + 4 nu k**2 q_dot + omega**2 q
        = k tanh(k H) p_mode / rho

    omega**2 = (g k + sigma k**3 / rho) tanh(k H)

The equilibrium meniscus is obtained from the nonlinear axisymmetric
Young--Laplace energy with a prescribed static contact angle and a volume
constraint. Dynamic perturbations currently retain that angle. A pinned
contact-line option is reserved but rejected until its edge constraint is
coupled consistently across the rigid-wall no-flux fluid modes.

This is a one-way, linearized jet-precursor model.  It cannot represent an
overturning interface, pinch-off, a detached drop, satellites, cavitation,
streaming, or feedback of the moving surface into the acoustic cavity.  Its
``positive_mound_volume_m3`` is therefore not a predicted droplet volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


ContactLineMode = Literal["fixed_contact_angle", "pinned_contact_line"]


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


@dataclass(frozen=True)
class FreeSurfaceLiquid:
    """Hydrodynamic properties measured at the operating composition.

    These values are intentionally not inferred from the acoustic-only DMSO
    table.  Viscosity and surface tension should be measured or sourced for
    the actual concentration and temperature.
    """

    density_kg_m3: float
    dynamic_viscosity_pa_s: float
    surface_tension_n_m: float

    def __post_init__(self) -> None:
        _positive("density_kg_m3", self.density_kg_m3)
        _nonnegative("dynamic_viscosity_pa_s", self.dynamic_viscosity_pa_s)
        _positive("surface_tension_n_m", self.surface_tension_n_m)

    @property
    def kinematic_viscosity_m2_s(self) -> float:
        return self.dynamic_viscosity_pa_s / self.density_kg_m3


@dataclass(frozen=True)
class ContactLineModel:
    """Static wetting angle and linearized dynamic contact-line condition.

    ``equilibrium_contact_angle_deg`` is measured through the liquid at the
    vertical well wall.  ``fixed_contact_angle`` lets the edge move while the
    perturbation slope remains zero.  ``pinned_contact_line`` is reserved but
    intentionally rejected by the transient solver: a physically consistent
    pinned model must retain the rigid-wall no-flux fluid modes and impose the
    edge constraint as a coupled condition.  Advancing/receding-angle
    hysteresis and a dynamic contact-line mobility law are not included.
    """

    mode: ContactLineMode = "fixed_contact_angle"
    equilibrium_contact_angle_deg: float = 90.0

    def __post_init__(self) -> None:
        if self.mode not in {"fixed_contact_angle", "pinned_contact_line"}:
            raise ValueError(
                "mode must be 'fixed_contact_angle' or 'pinned_contact_line'"
            )
        angle = float(self.equilibrium_contact_angle_deg)
        if not np.isfinite(angle) or not 5.0 <= angle <= 175.0:
            raise ValueError(
                "equilibrium_contact_angle_deg must lie between 5 and 175"
            )


@dataclass(frozen=True)
class AxisymmetricSurfaceModes:
    """Volume-conserving modes used by the transient surface solver."""

    radius_m: NDArray[np.float64]
    mode_shapes_m_inv: NDArray[np.float64]
    radial_wavenumber_m_inv: NDArray[np.float64]
    natural_angular_frequency_rad_s: NDArray[np.float64]
    viscous_amplitude_decay_rate_s: NDArray[np.float64]
    damping_ratio: NDArray[np.float64]
    contact_line: ContactLineModel
    liquid_depth_m: float
    gravity_m_s2: float

    @property
    def natural_frequency_hz(self) -> NDArray[np.float64]:
        return self.natural_angular_frequency_rad_s / (2.0 * np.pi)


@dataclass(frozen=True)
class AxisymmetricFreeSurfaceResult:
    """Transient pre-ejection surface response and validity diagnostics."""

    time_s: NDArray[np.float64]
    radius_m: NDArray[np.float64]
    applied_normal_stress_pa: NDArray[np.float64]
    equilibrium_elevation_m: NDArray[np.float64]
    dynamic_elevation_m: NDArray[np.float64]
    surface_elevation_m: NDArray[np.float64]
    vertical_velocity_m_s: NDArray[np.float64]
    curvature_1_m: NDArray[np.float64]
    modal_displacement_m2: NDArray[np.float64]
    modal_velocity_m2_s: NDArray[np.float64]
    radial_wavenumber_m_inv: NDArray[np.float64]
    natural_frequency_hz: NDArray[np.float64]
    viscous_amplitude_decay_rate_s: NDArray[np.float64]
    damping_ratio: NDArray[np.float64]
    active_mode_mask: NDArray[np.bool_]
    positive_mound_volume_m3: NDArray[np.float64]
    volume_residual_m3: NDArray[np.float64]
    mechanical_energy_j: NDArray[np.float64]
    cumulative_acoustic_work_j: NDArray[np.float64]
    maximum_dynamic_abs_slope: float
    maximum_equilibrium_abs_slope: float
    maximum_abs_dynamic_elevation_m: float
    maximum_abs_equilibrium_elevation_m: float
    minimum_local_liquid_depth_m: float
    linear_slope_limit: float
    equilibrium_slope_limit: float
    weak_viscosity_limit: float
    linear_slope_limit_exceeded: bool
    equilibrium_slope_limit_exceeded: bool
    nonpositive_depth_reached: bool
    weak_viscosity_limit_exceeded: bool
    frozen_acoustic_feedback_likely: bool
    forcing_is_absolute: bool
    contact_line: ContactLineModel
    limitations: tuple[str, ...]

    @property
    def apex_dynamic_elevation_m(self) -> NDArray[np.float64]:
        return self.dynamic_elevation_m[:, 0]

    @property
    def apex_surface_elevation_m(self) -> NDArray[np.float64]:
        return self.surface_elevation_m[:, 0]

    @property
    def apex_vertical_velocity_m_s(self) -> NDArray[np.float64]:
        return self.vertical_velocity_m_s[:, 0]

    @property
    def peak_positive_apex_displacement_m(self) -> float:
        return float(np.max(self.apex_dynamic_elevation_m))

    @property
    def can_predict_detached_drop_volume(self) -> bool:
        """Always false: this model has no interface topology change."""

        return False

    @property
    def within_reduced_model_validity(self) -> bool:
        return not (
            self.linear_slope_limit_exceeded
            or self.equilibrium_slope_limit_exceeded
            or self.nonpositive_depth_reached
            or self.weak_viscosity_limit_exceeded
            or self.frozen_acoustic_feedback_likely
        )


def _validate_radius(radius_m: ArrayLike) -> NDArray[np.float64]:
    radius = np.asarray(radius_m, dtype=float)
    if radius.ndim != 1 or radius.size < 16 or np.any(~np.isfinite(radius)):
        raise ValueError("radius_m must contain at least 16 finite 1D nodes")
    if abs(float(radius[0])) > 1.0e-12 * float(radius[-1]):
        raise ValueError("radius_m must start at zero")
    if np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius_m must be strictly increasing")
    return radius


def _validate_time(time_s: ArrayLike) -> tuple[NDArray[np.float64], float]:
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or time.size < 3 or np.any(~np.isfinite(time)):
        raise ValueError("time_s must contain at least three finite 1D samples")
    delta = np.diff(time)
    if np.any(delta <= 0.0) or not np.allclose(
        delta, delta[0], rtol=1.0e-8, atol=0.0
    ):
        raise ValueError("time_s must be strictly increasing and uniformly sampled")
    return time, float(delta[0])


def _axisymmetric_fem_matrices(
    radius_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Linear-element mass/stiffness matrices with the radial ``r`` weight."""

    node_count = radius_m.size
    mass = np.zeros((node_count, node_count), dtype=float)
    stiffness = np.zeros_like(mass)
    gauss_x = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    for left in range(node_count - 1):
        right = left + 1
        r0 = float(radius_m[left])
        r1 = float(radius_m[right])
        length = r1 - r0
        local_mass = np.zeros((2, 2), dtype=float)
        derivative = np.array([-1.0 / length, 1.0 / length])
        radial_measure = 0.5 * (r1**2 - r0**2)
        local_stiffness = radial_measure * np.outer(derivative, derivative)
        for xi in gauss_x:
            shape = np.array([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)])
            radial_position = 0.5 * ((1.0 - xi) * r0 + (1.0 + xi) * r1)
            local_mass += (
                0.5 * length * radial_position * np.outer(shape, shape)
            )
        indices = np.ix_([left, right], [left, right])
        mass[indices] += local_mass
        stiffness[indices] += local_stiffness
    return mass, stiffness


def _equilibrium_residual_and_tangent(
    elevation_m: NDArray[np.float64],
    lagrange_force_n: float,
    *,
    radius_m: NDArray[np.float64],
    mass: NDArray[np.float64],
    liquid: FreeSurfaceLiquid,
    gravity_m_s2: float,
    contact_angle_deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    node_count = radius_m.size
    capillary_force = np.zeros(node_count, dtype=float)
    capillary_tangent = np.zeros((node_count, node_count), dtype=float)
    sigma = liquid.surface_tension_n_m
    for left in range(node_count - 1):
        right = left + 1
        length = float(radius_m[right] - radius_m[left])
        derivative = np.array([-1.0 / length, 1.0 / length])
        slope = float(elevation_m[right] - elevation_m[left]) / length
        radial_measure = 0.5 * (
            float(radius_m[right]) ** 2 - float(radius_m[left]) ** 2
        )
        normalized_slope = slope / np.sqrt(1.0 + slope**2)
        local_force = sigma * radial_measure * normalized_slope * derivative
        local_tangent = (
            sigma
            * radial_measure
            / (1.0 + slope**2) ** 1.5
            * np.outer(derivative, derivative)
        )
        capillary_force[[left, right]] += local_force
        capillary_tangent[np.ix_([left, right], [left, right])] += local_tangent

    radius = float(radius_m[-1])
    volume_constraint = (mass @ np.ones(node_count)) / radius**2
    angle_rad = np.deg2rad(contact_angle_deg)
    boundary_force = np.zeros(node_count, dtype=float)
    boundary_force[-1] = sigma * radius * np.cos(angle_rad)
    gravity_tangent = liquid.density_kg_m3 * gravity_m_s2 * mass
    residual = (
        capillary_force
        + gravity_tangent @ elevation_m
        - lagrange_force_n * volume_constraint
        - boundary_force
    )
    tangent = capillary_tangent + gravity_tangent
    constraint_residual = float(volume_constraint @ elevation_m)
    return residual, tangent, constraint_residual


def equilibrium_meniscus_profile(
    radius_m: ArrayLike,
    *,
    liquid: FreeSurfaceLiquid,
    equilibrium_contact_angle_deg: float,
    gravity_m_s2: float = 9.80665,
    relative_tolerance: float = 1.0e-10,
    maximum_iterations: int = 40,
) -> NDArray[np.float64]:
    """Solve the nonlinear static Young--Laplace meniscus with fixed volume.

    Elevation is reported relative to the area-weighted mean plane.  The
    contact angle is measured through the liquid at the vertical wall.
    """

    radius = _validate_radius(radius_m)
    gravity = _nonnegative("gravity_m_s2", gravity_m_s2)
    angle = float(equilibrium_contact_angle_deg)
    if not np.isfinite(angle) or not 5.0 <= angle <= 175.0:
        raise ValueError("equilibrium_contact_angle_deg must lie in [5, 175]")
    tolerance = _positive("relative_tolerance", relative_tolerance)
    if tolerance >= 1.0:
        raise ValueError("relative_tolerance must be smaller than one")
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, (int, np.integer))
        or maximum_iterations < 1
    ):
        raise ValueError("maximum_iterations must be an integer >= 1")

    mass, stiffness = _axisymmetric_fem_matrices(radius)
    node_count = radius.size
    radius_max = float(radius[-1])
    constraint = (mass @ np.ones(node_count)) / radius_max**2
    boundary = np.zeros(node_count, dtype=float)
    boundary[-1] = (
        liquid.surface_tension_n_m
        * radius_max
        * np.cos(np.deg2rad(angle))
    )
    linear_tangent = (
        liquid.surface_tension_n_m * stiffness
        + liquid.density_kg_m3 * gravity * mass
    )
    linear_system = np.block(
        [
            [linear_tangent, -constraint[:, None]],
            [constraint[None, :], np.zeros((1, 1))],
        ]
    )
    initial = np.linalg.solve(
        linear_system,
        np.concatenate([boundary, np.zeros(1)]),
    )
    elevation = initial[:-1]
    lagrange_force = float(initial[-1])
    force_scale = max(
        liquid.surface_tension_n_m * radius_max,
        float(np.max(np.abs(boundary))),
        1.0e-18,
    )

    def merit(candidate: NDArray[np.float64], multiplier: float) -> float:
        residual, _, volume_error = _equilibrium_residual_and_tangent(
            candidate,
            multiplier,
            radius_m=radius,
            mass=mass,
            liquid=liquid,
            gravity_m_s2=gravity,
            contact_angle_deg=angle,
        )
        return max(
            float(np.max(np.abs(residual))) / force_scale,
            abs(volume_error) / radius_max,
        )

    for _ in range(maximum_iterations):
        residual, tangent, volume_error = _equilibrium_residual_and_tangent(
            elevation,
            lagrange_force,
            radius_m=radius,
            mass=mass,
            liquid=liquid,
            gravity_m_s2=gravity,
            contact_angle_deg=angle,
        )
        current_merit = max(
            float(np.max(np.abs(residual))) / force_scale,
            abs(volume_error) / radius_max,
        )
        if current_merit <= tolerance:
            return elevation
        jacobian = np.block(
            [
                [tangent, -constraint[:, None]],
                [constraint[None, :], np.zeros((1, 1))],
            ]
        )
        correction = np.linalg.solve(
            jacobian,
            -np.concatenate([residual, np.array([volume_error])]),
        )
        step = 1.0
        accepted = False
        while step >= 1.0 / 128.0:
            candidate = elevation + step * correction[:-1]
            candidate_multiplier = lagrange_force + step * float(correction[-1])
            if merit(candidate, candidate_multiplier) < current_merit:
                elevation = candidate
                lagrange_force = candidate_multiplier
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    raise RuntimeError(
        "the nonlinear Young-Laplace equilibrium did not converge; refine the "
        "radial grid or use a contact angle closer to 90 degrees"
    )


def _constraint_null_space(constraint: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(constraint))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("the radial volume constraint is singular")
    q, _ = np.linalg.qr((constraint / norm)[:, None], mode="complete")
    return q[:, 1:]


def axisymmetric_surface_modes(
    radius_m: ArrayLike,
    *,
    liquid: FreeSurfaceLiquid,
    liquid_depth_m: float,
    contact_line: ContactLineModel = ContactLineModel(),
    mode_count: int = 24,
    gravity_m_s2: float = 9.80665,
) -> AxisymmetricSurfaceModes:
    """Build volume-conserving radial modes for a circular well."""

    radius = _validate_radius(radius_m)
    depth = _positive("liquid_depth_m", liquid_depth_m)
    gravity = _nonnegative("gravity_m_s2", gravity_m_s2)
    if contact_line.mode == "pinned_contact_line":
        raise NotImplementedError(
            "pinned_contact_line dynamics are not implemented: pinning must "
            "be coupled across rigid-wall no-flux fluid modes; use "
            "fixed_contact_angle for the validated reduced model"
        )
    if (
        isinstance(mode_count, bool)
        or not isinstance(mode_count, (int, np.integer))
        or mode_count < 1
    ):
        raise ValueError("mode_count must be an integer >= 1")

    mass, stiffness = _axisymmetric_fem_matrices(radius)
    full_volume_vector = mass @ np.ones(radius.size)
    active_count = radius.size
    active_mass = mass
    active_stiffness = stiffness
    constraint = full_volume_vector

    null_space = _constraint_null_space(constraint)
    if mode_count > null_space.shape[1]:
        raise ValueError(
            f"mode_count must be <= {null_space.shape[1]} for this radial grid"
        )
    projected_mass = null_space.T @ active_mass @ null_space
    projected_stiffness = null_space.T @ active_stiffness @ null_space
    cholesky = np.linalg.cholesky(projected_mass)
    left_solved = np.linalg.solve(cholesky, projected_stiffness)
    standard_stiffness = np.linalg.solve(cholesky, left_solved.T).T
    standard_stiffness = 0.5 * (
        standard_stiffness + standard_stiffness.T
    )
    eigenvalues, standard_modes = np.linalg.eigh(standard_stiffness)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    standard_modes = standard_modes[:, order]
    positive = eigenvalues > max(float(eigenvalues[-1]), 1.0) * 1.0e-12
    eigenvalues = eigenvalues[positive][:mode_count]
    standard_modes = standard_modes[:, positive][:, :mode_count]
    if eigenvalues.size != mode_count:
        raise RuntimeError("insufficient positive radial surface modes")
    active_modes = null_space @ np.linalg.solve(
        cholesky.T, standard_modes
    )
    mode_shapes = np.zeros((radius.size, mode_count), dtype=float)
    mode_shapes[:active_count] = active_modes
    for index in range(mode_count):
        if mode_shapes[0, index] < 0.0:
            mode_shapes[:, index] *= -1.0

    wavenumber = np.sqrt(eigenvalues)
    depth_factor = np.tanh(wavenumber * depth)
    omega_squared = (
        gravity * wavenumber
        + liquid.surface_tension_n_m
        / liquid.density_kg_m3
        * wavenumber**3
    ) * depth_factor
    omega = np.sqrt(omega_squared)
    decay_rate = (
        2.0 * liquid.kinematic_viscosity_m2_s * wavenumber**2
    )
    damping_ratio = decay_rate / omega
    return AxisymmetricSurfaceModes(
        radius_m=radius,
        mode_shapes_m_inv=mode_shapes,
        radial_wavenumber_m_inv=wavenumber,
        natural_angular_frequency_rad_s=omega,
        viscous_amplitude_decay_rate_s=decay_rate,
        damping_ratio=damping_ratio,
        contact_line=contact_line,
        liquid_depth_m=depth,
        gravity_m_s2=gravity,
    )


def radiation_stress_from_incident_pressure_envelope(
    incident_peak_pressure_envelope_pa: ArrayLike,
    *,
    density_kg_m3: float,
    sound_speed_m_s: float,
    reflected_intensity_fraction: float = 1.0,
    transmitted_intensity_fraction: float = 0.0,
    backing_sound_speed_m_s: float | None = None,
) -> NDArray[np.float64]:
    """Convert a peak-pressure envelope to cycle-averaged normal stress.

    The normal momentum-flux difference is
    ``I * ((1 + R_I) / c - T_I / c_backing)`` with
    ``I = p_peak**2 / (2 rho c)``. ``R_I=1`` and ``T_I=0`` give the
    ideal-reflection result ``p_peak**2 / (rho c**2)``. Any fraction
    ``1 - R_I - T_I`` is treated as absorbed at the interface. The input must
    be the incident forward pressure envelope; do not pass the total pressure
    at a pressure-release surface or an RMS pressure without converting its
    convention.
    """

    pressure = np.asarray(incident_peak_pressure_envelope_pa)
    if np.iscomplexobj(pressure):
        magnitude_squared = np.abs(pressure) ** 2
    else:
        pressure = pressure.astype(float)
        if np.any(~np.isfinite(pressure)):
            raise ValueError("incident pressure envelope must be finite")
        magnitude_squared = pressure**2
    if np.any(~np.isfinite(magnitude_squared)):
        raise ValueError("incident pressure envelope must be finite")
    density = _positive("density_kg_m3", density_kg_m3)
    sound_speed = _positive("sound_speed_m_s", sound_speed_m_s)
    reflected = float(reflected_intensity_fraction)
    if not np.isfinite(reflected) or not 0.0 <= reflected <= 1.0:
        raise ValueError("reflected_intensity_fraction must lie in [0, 1]")
    transmitted = float(transmitted_intensity_fraction)
    if not np.isfinite(transmitted) or not 0.0 <= transmitted <= 1.0:
        raise ValueError("transmitted_intensity_fraction must lie in [0, 1]")
    if reflected + transmitted > 1.0 + 1.0e-12:
        raise ValueError(
            "reflected and transmitted intensity fractions must sum to <= 1"
        )
    if transmitted > 0.0:
        if backing_sound_speed_m_s is None:
            raise ValueError(
                "backing_sound_speed_m_s is required when transmitted "
                "intensity is nonzero"
            )
        backing_speed = _positive(
            "backing_sound_speed_m_s", backing_sound_speed_m_s
        )
    else:
        backing_speed = np.inf
    intensity = magnitude_squared / (2.0 * density * sound_speed)
    momentum_factor = (
        (1.0 + reflected) / sound_speed - transmitted / backing_speed
    )
    return (intensity * momentum_factor).astype(float)


def raised_cosine_tone_envelope(
    time_s: ArrayLike,
    *,
    start_time_s: float,
    duration_s: float,
    edge_time_s: float = 0.0,
) -> NDArray[np.float64]:
    """Return a unit envelope with optional raised-cosine rise and fall."""

    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or np.any(~np.isfinite(time)):
        raise ValueError("time_s must be a finite 1D array")
    start = _nonnegative("start_time_s", start_time_s)
    duration = _positive("duration_s", duration_s)
    edge = _nonnegative("edge_time_s", edge_time_s)
    if 2.0 * edge > duration:
        raise ValueError("2 * edge_time_s must not exceed duration_s")
    local = time - start
    envelope = np.zeros_like(time)
    active = (local >= 0.0) & (local <= duration)
    envelope[active] = 1.0
    if edge > 0.0:
        rising = (local >= 0.0) & (local < edge)
        falling = (local > duration - edge) & (local <= duration)
        envelope[rising] = 0.5 * (
            1.0 - np.cos(np.pi * local[rising] / edge)
        )
        envelope[falling] = 0.5 * (
            1.0
            - np.cos(np.pi * (duration - local[falling]) / edge)
        )
    return envelope


def _axisymmetric_curvature(
    elevation_m: NDArray[np.float64],
    radius_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    slope = np.gradient(elevation_m, radius_m, axis=1, edge_order=2)
    normalized_slope = slope / np.sqrt(1.0 + slope**2)
    radial_flux = normalized_slope * radius_m[None, :]
    derivative = np.gradient(radial_flux, radius_m, axis=1, edge_order=2)
    curvature = np.empty_like(elevation_m)
    curvature[:, 1:] = derivative[:, 1:] / radius_m[None, 1:]
    curvature[:, 0] = 2.0 * (
        normalized_slope[:, 1] - normalized_slope[:, 0]
    ) / (radius_m[1] - radius_m[0])
    return curvature, slope


def simulate_axisymmetric_free_surface(
    time_s: ArrayLike,
    radius_m: ArrayLike,
    normal_stress_pa: ArrayLike,
    *,
    liquid: FreeSurfaceLiquid,
    liquid_depth_m: float,
    contact_line: ContactLineModel = ContactLineModel(),
    mode_count: int = 24,
    gravity_m_s2: float = 9.80665,
    initial_dynamic_elevation_m: ArrayLike | None = None,
    initial_vertical_velocity_m_s: ArrayLike | None = None,
    forcing_is_absolute: bool = False,
    acoustic_wavelength_m: float | None = None,
    focal_spot_radius_m: float | None = None,
    linear_slope_limit: float = 0.25,
    equilibrium_slope_limit: float = 0.30,
    weak_viscosity_limit: float = 0.20,
) -> AxisymmetricFreeSurfaceResult:
    """Simulate a one-way coupled transient axisymmetric meniscus response.

    ``normal_stress_pa`` has shape ``(time, radius)`` and is positive upward,
    toward the gas.  It should be a cycle-averaged acoustic radiation stress
    or another slow normal load.  A spatially uniform pressure is removed by
    the closed-well volume constraint and therefore cannot deform the surface.

    Average-acceleration Newmark integration is used for the uncoupled modal
    oscillators.  The method is stable for this linear problem and does not
    require resolving the MHz acoustic carrier.
    """

    time, delta_t = _validate_time(time_s)
    radius = _validate_radius(radius_m)
    stress = np.asarray(normal_stress_pa, dtype=float)
    if stress.shape != (time.size, radius.size):
        raise ValueError("normal_stress_pa must have shape (time, radius)")
    if np.any(~np.isfinite(stress)):
        raise ValueError("normal_stress_pa must be finite")
    slope_limit = _positive("linear_slope_limit", linear_slope_limit)
    static_slope_limit = _positive(
        "equilibrium_slope_limit", equilibrium_slope_limit
    )
    viscosity_limit = _positive("weak_viscosity_limit", weak_viscosity_limit)
    if acoustic_wavelength_m is not None:
        acoustic_wavelength_m = _positive(
            "acoustic_wavelength_m", acoustic_wavelength_m
        )
    if focal_spot_radius_m is not None:
        focal_spot_radius_m = _positive(
            "focal_spot_radius_m", focal_spot_radius_m
        )

    modes = axisymmetric_surface_modes(
        radius,
        liquid=liquid,
        liquid_depth_m=liquid_depth_m,
        contact_line=contact_line,
        mode_count=mode_count,
        gravity_m_s2=gravity_m_s2,
    )
    equilibrium = equilibrium_meniscus_profile(
        radius,
        liquid=liquid,
        equilibrium_contact_angle_deg=(
            contact_line.equilibrium_contact_angle_deg
        ),
        gravity_m_s2=gravity_m_s2,
    )
    mass, _ = _axisymmetric_fem_matrices(radius)
    mass_modes = mass @ modes.mode_shapes_m_inv
    modal_stress = stress @ mass_modes
    wavenumber = modes.radial_wavenumber_m_inv
    depth_factor = np.tanh(wavenumber * liquid_depth_m)
    forcing = modal_stress * (
        wavenumber * depth_factor / liquid.density_kg_m3
    )[None, :]
    omega = modes.natural_angular_frequency_rad_s
    stiffness = omega**2
    damping = 2.0 * modes.viscous_amplitude_decay_rate_s

    def initial_field(
        values: ArrayLike | None,
        *,
        name: str,
    ) -> NDArray[np.float64]:
        if values is None:
            return np.zeros(radius.size, dtype=float)
        field = np.asarray(values, dtype=float)
        if field.shape != radius.shape or np.any(~np.isfinite(field)):
            raise ValueError(f"{name} must be a finite array matching radius_m")
        return field

    initial_elevation = initial_field(
        initial_dynamic_elevation_m,
        name="initial_dynamic_elevation_m",
    )
    initial_velocity = initial_field(
        initial_vertical_velocity_m_s,
        name="initial_vertical_velocity_m_s",
    )
    modal_displacement = np.zeros((time.size, mode_count), dtype=float)
    modal_velocity = np.zeros_like(modal_displacement)
    modal_acceleration = np.zeros_like(modal_displacement)
    modal_displacement[0] = modes.mode_shapes_m_inv.T @ mass @ initial_elevation
    modal_velocity[0] = modes.mode_shapes_m_inv.T @ mass @ initial_velocity
    modal_acceleration[0] = (
        forcing[0]
        - damping * modal_velocity[0]
        - stiffness * modal_displacement[0]
    )

    beta = 0.25
    newmark_gamma = 0.5
    denominator = (
        1.0
        + newmark_gamma * delta_t * damping
        + beta * delta_t**2 * stiffness
    )
    for index in range(time.size - 1):
        displacement_predictor = (
            modal_displacement[index]
            + delta_t * modal_velocity[index]
            + delta_t**2 * (0.5 - beta) * modal_acceleration[index]
        )
        velocity_predictor = (
            modal_velocity[index]
            + delta_t
            * (1.0 - newmark_gamma)
            * modal_acceleration[index]
        )
        modal_acceleration[index + 1] = (
            forcing[index + 1]
            - damping * velocity_predictor
            - stiffness * displacement_predictor
        ) / denominator
        modal_displacement[index + 1] = (
            displacement_predictor
            + beta * delta_t**2 * modal_acceleration[index + 1]
        )
        modal_velocity[index + 1] = (
            velocity_predictor
            + newmark_gamma * delta_t * modal_acceleration[index + 1]
        )

    dynamic_elevation = modal_displacement @ modes.mode_shapes_m_inv.T
    vertical_velocity = modal_velocity @ modes.mode_shapes_m_inv.T
    surface_elevation = equilibrium[None, :] + dynamic_elevation
    _, dynamic_slope = _axisymmetric_curvature(dynamic_elevation, radius)
    _, equilibrium_slope = _axisymmetric_curvature(
        equilibrium[None, :], radius
    )
    total_curvature, _ = _axisymmetric_curvature(surface_elevation, radius)
    curvature = total_curvature
    volume_vector = mass @ np.ones(radius.size)
    volume_residual = 2.0 * np.pi * (dynamic_elevation @ volume_vector)
    positive_integrand = np.maximum(dynamic_elevation, 0.0) * radius[None, :]
    positive_mound_volume = 2.0 * np.pi * np.sum(
        0.5
        * (positive_integrand[:, 1:] + positive_integrand[:, :-1])
        * np.diff(radius)[None, :],
        axis=1,
    )

    effective_modal_mass = (
        2.0
        * np.pi
        * liquid.density_kg_m3
        / (wavenumber * depth_factor)
    )
    mechanical_energy = 0.5 * np.sum(
        effective_modal_mass[None, :]
        * (
            modal_velocity**2
            + stiffness[None, :] * modal_displacement**2
        ),
        axis=1,
    )
    acoustic_power = 2.0 * np.pi * np.einsum(
        "ti,ij,tj->t", stress, mass, vertical_velocity
    )
    cumulative_work = np.zeros(time.size, dtype=float)
    cumulative_work[1:] = np.cumsum(
        0.5 * delta_t * (acoustic_power[1:] + acoustic_power[:-1])
    )

    maximum_slope = float(np.max(np.abs(dynamic_slope)))
    maximum_equilibrium_slope = float(np.max(np.abs(equilibrium_slope)))
    maximum_elevation = float(np.max(np.abs(dynamic_elevation)))
    maximum_equilibrium_elevation = float(np.max(np.abs(equilibrium)))
    minimum_local_depth = float(
        liquid_depth_m + np.min(surface_elevation)
    )
    slope_exceeded = maximum_slope > slope_limit
    equilibrium_slope_exceeded = maximum_equilibrium_slope > static_slope_limit
    nonpositive_depth = minimum_local_depth <= 0.0
    modal_energy = 0.5 * effective_modal_mass[None, :] * (
        modal_velocity**2
        + stiffness[None, :] * modal_displacement**2
    )
    modal_participation = np.max(modal_energy, axis=0)
    maximum_participation = float(np.max(modal_participation))
    if maximum_participation > 0.0:
        active_modes = modal_participation >= 1.0e-8 * maximum_participation
    else:
        maximum_forcing = np.max(np.abs(forcing), axis=0)
        forcing_scale = float(np.max(maximum_forcing))
        active_modes = (
            maximum_forcing >= 1.0e-8 * forcing_scale
            if forcing_scale > 0.0
            else np.zeros(mode_count, dtype=bool)
        )
    viscosity_exceeded = bool(
        np.any(active_modes)
        and np.max(modes.damping_ratio[active_modes]) > viscosity_limit
    )
    frozen_feedback = False
    feedback_reasons: list[str] = []
    if acoustic_wavelength_m is not None:
        dynamic_ratio = maximum_elevation / acoustic_wavelength_m
        if dynamic_ratio > 0.05:
            frozen_feedback = True
            feedback_reasons.append(
                "dynamic deformation/acoustic wavelength reached "
                f"{dynamic_ratio:.3g}"
            )
        static_relief_ratio = (
            float(np.ptp(equilibrium)) / acoustic_wavelength_m
        )
        if static_relief_ratio > 0.05:
            frozen_feedback = True
            feedback_reasons.append(
                "static meniscus relief/acoustic wavelength reached "
                f"{static_relief_ratio:.3g}"
            )
    if focal_spot_radius_m is not None:
        dynamic_ratio = maximum_elevation / focal_spot_radius_m
        if dynamic_ratio > 0.10:
            frozen_feedback = True
            feedback_reasons.append(
                "dynamic deformation/focal-spot radius reached "
                f"{dynamic_ratio:.3g}"
            )
        static_ratio = maximum_equilibrium_elevation / focal_spot_radius_m
        if static_ratio > 0.10:
            frozen_feedback = True
            feedback_reasons.append(
                "static meniscus elevation/focal-spot radius reached "
                f"{static_ratio:.3g}"
            )
    depth_ratio = maximum_elevation / liquid_depth_m
    if depth_ratio > 0.10:
        frozen_feedback = True
        feedback_reasons.append(
            f"deformation/liquid depth reached {depth_ratio:.3g}"
        )

    limitations = [
        "linearized one-way capillary-gravity dynamics with a frozen acoustic field",
        "single-valued axisymmetric surface; no overturning, pinch-off, detached drop, or satellites",
        "positive_mound_volume_m3 is displaced mound volume, not droplet volume",
        "weak-viscosity modal damping omits wall/bottom boundary layers and viscoelastic effects",
        "dynamic modal operator is flat and contact-angle independent; wetting changes only the static baseline",
        "no acoustic streaming, cavitation, thermal accumulation, or contact-angle hysteresis",
    ]
    if not forcing_is_absolute:
        limitations.append(
            "normal-stress amplitude was not asserted to have absolute calibration"
        )
    if slope_exceeded:
        limitations.append(
            f"dynamic slope {maximum_slope:.3g} exceeded the configured linear limit {slope_limit:.3g}"
        )
    if equilibrium_slope_exceeded:
        limitations.append(
            "static wetting slope exceeds the flat-mode equilibrium limit: "
            f"{maximum_equilibrium_slope:.3g} > {static_slope_limit:.3g}"
        )
    if nonpositive_depth:
        limitations.append(
            "the reconstructed surface reached the rigid bottom; results are invalid"
        )
    if viscosity_exceeded:
        limitations.append(
            "at least one materially participating mode exceeds the "
            "weak-viscosity damping-ratio limit"
        )
    if feedback_reasons:
        limitations.append(
            "moving-surface acoustic feedback is likely: "
            + "; ".join(feedback_reasons)
        )

    return AxisymmetricFreeSurfaceResult(
        time_s=time,
        radius_m=radius,
        applied_normal_stress_pa=stress,
        equilibrium_elevation_m=equilibrium,
        dynamic_elevation_m=dynamic_elevation,
        surface_elevation_m=surface_elevation,
        vertical_velocity_m_s=vertical_velocity,
        curvature_1_m=curvature,
        modal_displacement_m2=modal_displacement,
        modal_velocity_m2_s=modal_velocity,
        radial_wavenumber_m_inv=wavenumber,
        natural_frequency_hz=modes.natural_frequency_hz,
        viscous_amplitude_decay_rate_s=modes.viscous_amplitude_decay_rate_s,
        damping_ratio=modes.damping_ratio,
        active_mode_mask=active_modes,
        positive_mound_volume_m3=positive_mound_volume,
        volume_residual_m3=volume_residual,
        mechanical_energy_j=mechanical_energy,
        cumulative_acoustic_work_j=cumulative_work,
        maximum_dynamic_abs_slope=maximum_slope,
        maximum_equilibrium_abs_slope=maximum_equilibrium_slope,
        maximum_abs_dynamic_elevation_m=maximum_elevation,
        maximum_abs_equilibrium_elevation_m=maximum_equilibrium_elevation,
        minimum_local_liquid_depth_m=minimum_local_depth,
        linear_slope_limit=slope_limit,
        equilibrium_slope_limit=static_slope_limit,
        weak_viscosity_limit=viscosity_limit,
        linear_slope_limit_exceeded=slope_exceeded,
        equilibrium_slope_limit_exceeded=equilibrium_slope_exceeded,
        nonpositive_depth_reached=nonpositive_depth,
        weak_viscosity_limit_exceeded=viscosity_exceeded,
        frozen_acoustic_feedback_likely=frozen_feedback,
        forcing_is_absolute=bool(forcing_is_absolute),
        contact_line=contact_line,
        limitations=tuple(limitations),
    )
