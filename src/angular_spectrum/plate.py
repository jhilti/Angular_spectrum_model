"""Plane-wave scattering by a fluid-loaded isotropic elastic plate.

Every transverse spatial-frequency component is rotated into its sagittal
plane.  The solid displacement is represented by forward/backward P and SV
potentials.  Six boundary conditions are enforced:

* normal displacement continuity at both faces,
* normal traction continuity at both faces,
* zero tangential traction at both fluid/solid interfaces.

The four solid-wave amplitudes include all P/SV mode conversions and all
internal reverberations.  SH motion is not driven by an acoustic fluid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .materials import ElasticPlate, Fluid


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def vertical_wavenumber(
    total_wavenumber: complex, transverse_wavenumber: ArrayLike
) -> ComplexArray:
    """Return the outgoing/decaying square-root branch of ``sqrt(k²-q²)``."""

    q = np.asarray(transverse_wavenumber, dtype=float)
    kz = np.sqrt(total_wavenumber**2 - q**2 + 0j).astype(
        np.complex128, copy=False
    )
    flip = (np.imag(kz) < 0.0) | (
        np.isclose(np.imag(kz), 0.0, atol=1e-14)
        & (np.real(kz) < 0.0)
    )
    return np.where(flip, -kz, kz)


def _solve_batched(matrix: ComplexArray, rhs: ComplexArray) -> ComplexArray:
    try:
        # The explicit trailing singleton keeps NumPy 1.x and 2.x batch
        # semantics identical.
        return np.linalg.solve(matrix, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        # A lossless plate can contain an exactly decoupled standing mode at an
        # isolated q.  Least squares keeps the physically unique fluid solution.
        result = np.empty_like(rhs)
        for index in range(matrix.shape[0]):
            result[index] = np.linalg.lstsq(
                matrix[index], rhs[index], rcond=1e-12
            )[0]
        return result


def _normal_incidence_scattering(
    frequency_hz: float,
    left: Fluid,
    plate: ElasticPlate,
    right: Fluid,
) -> tuple[complex, complex]:
    """Four-equation normal-incidence solution without decoupled SV modes."""

    omega = 2.0 * np.pi * frequency_hz
    omega2 = omega**2
    rho1 = left.density_kg_m3
    rho2 = plate.solid.density_kg_m3
    rho3 = right.density_kg_m3
    kz1 = left.wavenumber(frequency_hz)
    k_l = plate.solid.longitudinal_wavenumber(frequency_hz)
    kz3 = right.wavenumber(frequency_hz)
    lam, mu = plate.solid.complex_lame_parameters(frequency_hz)
    normal_modulus_term = (lam + 2.0 * mu) * k_l**2
    phase_l = np.exp(1j * k_l * plate.thickness_m)

    # Unknown potentials: reflected fluid, P+, P-, transmitted fluid.
    matrix = np.array(
        [
            [1j * kz1, 1j * k_l, -1j * k_l * phase_l, 0.0],
            [
                rho1 * omega2,
                -normal_modulus_term,
                -normal_modulus_term * phase_l,
                0.0,
            ],
            [0.0, 1j * k_l * phase_l, -1j * k_l, -1j * kz3],
            [
                0.0,
                -normal_modulus_term * phase_l,
                -normal_modulus_term,
                rho3 * omega2,
            ],
        ],
        dtype=np.complex128,
    )
    rhs = np.array(
        [1j * kz1, -rho1 * omega2, 0.0, 0.0],
        dtype=np.complex128,
    )
    reflected_potential, _, _, transmitted_potential = np.linalg.solve(matrix, rhs)
    reflected_pressure = reflected_potential
    transmitted_pressure = (rho3 / rho1) * transmitted_potential
    return complex(reflected_pressure), complex(transmitted_pressure)


def elastic_plate_scattering(
    transverse_wavenumber_rad_m: ArrayLike,
    frequency_hz: float,
    left: Fluid,
    plate: ElasticPlate,
    right: Fluid,
    *,
    chunk_size: int = 32768,
) -> tuple[ComplexArray, ComplexArray]:
    """Return complex pressure reflection and transmission coefficients.

    The coefficients are referenced at the two plate faces.  Input can have
    any shape; outputs have the same shape.  Pressure transmission is
    ``p_transmitted / p_incident``.
    """

    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and > 0")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    q_input = np.asarray(transverse_wavenumber_rad_m, dtype=float)
    if np.any(~np.isfinite(q_input)) or np.any(q_input < 0.0):
        raise ValueError("transverse wavenumbers must be finite and >= 0")
    original_shape = q_input.shape
    q_flat = q_input.reshape(-1)
    reflection = np.empty(q_flat.size, dtype=np.complex128)
    transmission = np.empty(q_flat.size, dtype=np.complex128)

    omega = 2.0 * np.pi * frequency_hz
    omega2 = omega**2
    rho1 = left.density_kg_m3
    rho3 = right.density_kg_m3
    k1 = left.wavenumber(frequency_hz)
    k_l = plate.solid.longitudinal_wavenumber(frequency_hz)
    k_s = plate.solid.shear_wavenumber(frequency_hz)
    k3 = right.wavenumber(frequency_hz)
    lam, mu = plate.solid.complex_lame_parameters(frequency_hz)
    thickness = plate.thickness_m

    normal_threshold = max(abs(k1), 1.0) * 1e-13
    normal_mask = q_flat <= normal_threshold
    if np.any(normal_mask):
        r0, t0 = _normal_incidence_scattering(
            frequency_hz, left, plate, right
        )
        reflection[normal_mask] = r0
        transmission[normal_mask] = t0

    oblique_indices = np.flatnonzero(~normal_mask)
    for start in range(0, oblique_indices.size, chunk_size):
        indices = oblique_indices[start : start + chunk_size]
        q = q_flat[indices]
        count = q.size

        kz1 = vertical_wavenumber(k1, q)
        kz_l = vertical_wavenumber(k_l, q)
        kz_s = vertical_wavenumber(k_s, q)
        kz3 = vertical_wavenumber(k3, q)

        phase_l = np.exp(1j * kz_l * thickness)
        phase_s = np.exp(1j * kz_s * thickness)
        normal_l = lam * k_l**2 + 2.0 * mu * kz_l**2
        shear_s = mu * (kz_s**2 - q**2)
        p_to_sv = 2.0 * mu * q * kz_s
        p_shear = 2.0 * mu * q * kz_l

        # Unknowns: R, P+, P-, SV+, SV-, T (all displacement potentials).
        matrix = np.zeros((count, 6, 6), dtype=np.complex128)
        rhs = np.zeros((count, 6), dtype=np.complex128)

        # Front face: normal displacement, normal traction, shear traction.
        matrix[:, 0, 0] = 1j * kz1
        matrix[:, 0, 1] = 1j * kz_l
        matrix[:, 0, 2] = -1j * kz_l * phase_l
        matrix[:, 0, 3] = 1j * q
        matrix[:, 0, 4] = 1j * q * phase_s
        rhs[:, 0] = 1j * kz1

        matrix[:, 1, 0] = rho1 * omega2
        matrix[:, 1, 1] = -normal_l
        matrix[:, 1, 2] = -normal_l * phase_l
        matrix[:, 1, 3] = -p_to_sv
        matrix[:, 1, 4] = p_to_sv * phase_s
        rhs[:, 1] = -rho1 * omega2

        matrix[:, 2, 1] = -p_shear
        matrix[:, 2, 2] = p_shear * phase_l
        matrix[:, 2, 3] = shear_s
        matrix[:, 2, 4] = shear_s * phase_s

        # Rear face: normal displacement, normal traction, shear traction.
        matrix[:, 3, 1] = 1j * kz_l * phase_l
        matrix[:, 3, 2] = -1j * kz_l
        matrix[:, 3, 3] = 1j * q * phase_s
        matrix[:, 3, 4] = 1j * q
        matrix[:, 3, 5] = -1j * kz3

        matrix[:, 4, 1] = -normal_l * phase_l
        matrix[:, 4, 2] = -normal_l
        matrix[:, 4, 3] = -p_to_sv * phase_s
        matrix[:, 4, 4] = p_to_sv
        matrix[:, 4, 5] = rho3 * omega2

        matrix[:, 5, 1] = -p_shear * phase_l
        matrix[:, 5, 2] = p_shear
        matrix[:, 5, 3] = shear_s * phase_s
        matrix[:, 5, 4] = shear_s

        solution = _solve_batched(matrix, rhs)
        reflection[indices] = solution[:, 0]
        transmission[indices] = (rho3 / rho1) * solution[:, 5]

    return (
        reflection.reshape(original_shape),
        transmission.reshape(original_shape),
    )


def elastic_plate_transfer_map(
    transverse_wavenumber_rad_m: ArrayLike,
    frequency_hz: float,
    left: Fluid,
    plate: ElasticPlate,
    right: Fluid,
    *,
    radial_samples: int | None = 4096,
) -> ComplexArray:
    """Return a 2D-ready plate transfer map.

    For an isotropic parallel plate the coefficient depends only on ``q``.
    Sampling it densely on a one-dimensional radial grid is much faster than
    solving a 6x6 system independently at every Cartesian FFT sample.
    Set ``radial_samples=None`` for a direct solve at all supplied values.
    """

    q = np.asarray(transverse_wavenumber_rad_m, dtype=float)
    if radial_samples is None:
        return elastic_plate_scattering(q, frequency_hz, left, plate, right)[1]
    if radial_samples < 128:
        raise ValueError("radial_samples must be >= 128 or None")
    q_max = float(np.max(q))
    if q_max == 0.0:
        return np.full(
            q.shape,
            _normal_incidence_scattering(frequency_hz, left, plate, right)[1],
            dtype=np.complex128,
        )
    q_radial = np.linspace(0.0, q_max, radial_samples, dtype=float)
    transmission_radial = elastic_plate_scattering(
        q_radial, frequency_hz, left, plate, right
    )[1]
    real = np.interp(q.reshape(-1), q_radial, transmission_radial.real)
    imag = np.interp(q.reshape(-1), q_radial, transmission_radial.imag)
    return (real + 1j * imag).reshape(q.shape)


def fluid_interface_scattering(
    transverse_wavenumber_rad_m: ArrayLike,
    frequency_hz: float,
    left: Fluid,
    right: Fluid,
) -> tuple[ComplexArray, ComplexArray]:
    """Return pressure scattering coefficients of a single fluid interface."""

    q = np.asarray(transverse_wavenumber_rad_m, dtype=float)
    kz1 = vertical_wavenumber(left.wavenumber(frequency_hz), q)
    kz3 = vertical_wavenumber(right.wavenumber(frequency_hz), q)
    admittance1 = kz1 / left.density_kg_m3
    admittance3 = kz3 / right.density_kg_m3
    denominator = admittance1 + admittance3
    reflection = (admittance1 - admittance3) / denominator
    transmission = 2.0 * admittance1 / denominator
    return reflection, transmission


def normal_power_transmission(
    transverse_wavenumber_rad_m: ArrayLike,
    frequency_hz: float,
    left: Fluid,
    right: Fluid,
    pressure_transmission: ArrayLike,
) -> FloatArray:
    """Convert a pressure coefficient to forward normal power transmission."""

    q = np.asarray(transverse_wavenumber_rad_m, dtype=float)
    coefficient = np.asarray(pressure_transmission, dtype=np.complex128)
    kz1 = vertical_wavenumber(left.wavenumber(frequency_hz), q)
    kz3 = vertical_wavenumber(right.wavenumber(frequency_hz), q)
    input_admittance = np.real(kz1 / left.density_kg_m3)
    output_admittance = np.real(kz3 / right.density_kg_m3)
    with np.errstate(divide="ignore", invalid="ignore"):
        power = np.abs(coefficient) ** 2 * output_admittance / input_admittance
    return np.where(input_admittance > 0.0, power, np.nan).astype(float)
