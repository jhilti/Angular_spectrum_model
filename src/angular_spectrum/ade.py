"""Linear screening metrics for acoustic droplet-ejection experiments.

These helpers translate a focal-field calculation into useful length and time
scales.  They do **not** predict ejection threshold, drop volume, velocity, or
satellite formation: those require a calibrated pressure field and a
transient, nonlinear free-surface model with surface tension and viscosity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ADELinearScreening:
    """Homogeneous-medium diffraction proxy and ideal timing scales."""

    frequency_hz: float
    wavelength_m: float
    f_number: float
    diffraction_spot_diameter_m: float
    spot_equivalent_sphere_volume_m3: float
    tone_duration_s: float
    liquid_cavity_round_trip_s: float
    liquid_cavity_round_trip_cycles: float
    first_cavity_return_after_drive: bool

    @property
    def homogeneous_intensity_fwhm_proxy_m(self) -> float:
        """Explicit name for the legacy ``diffraction_spot_diameter_m``."""

        return self.diffraction_spot_diameter_m

    @property
    def nominal_cavity_delay_exceeds_ideal_burst(self) -> bool:
        """Whether ``2h/c`` exceeds ``cycles/f``, excluding ring-down."""

        return self.first_cavity_return_after_drive


def linear_ade_screening(
    *,
    frequency_hz: float,
    sound_speed_m_s: float,
    focal_length_m: float,
    aperture_diameter_m: float,
    liquid_height_m: float,
    tone_cycles: float,
) -> ADELinearScreening:
    """Return diffraction and cavity-delay scales for ADE design screening.

    The spot estimate ``1.02 * F# * wavelength`` is the conventional one-way
    intensity-FWHM diameter proxy for a homogeneous circular-aperture focus.
    A layered refracting system should use the simulated intensity FWHM
    instead. The equivalent-sphere volume is only a geometric scale built from
    this proxy; it is deliberately not a predicted droplet volume.

    The cavity/burst boolean compares ideal durations only. Probe/electronics
    ring-down, pulse spreading, and oblique paths are not included.
    """

    parameters = {
        "frequency_hz": frequency_hz,
        "sound_speed_m_s": sound_speed_m_s,
        "focal_length_m": focal_length_m,
        "aperture_diameter_m": aperture_diameter_m,
        "liquid_height_m": liquid_height_m,
        "tone_cycles": tone_cycles,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in parameters.values()):
        raise ValueError("all ADE screening parameters must be finite and > 0")

    wavelength = sound_speed_m_s / frequency_hz
    f_number = focal_length_m / aperture_diameter_m
    spot_diameter = 1.02 * f_number * wavelength
    equivalent_volume = np.pi * spot_diameter**3 / 6.0
    tone_duration = tone_cycles / frequency_hz
    cavity_round_trip = 2.0 * liquid_height_m / sound_speed_m_s
    return ADELinearScreening(
        frequency_hz=float(frequency_hz),
        wavelength_m=float(wavelength),
        f_number=float(f_number),
        diffraction_spot_diameter_m=float(spot_diameter),
        spot_equivalent_sphere_volume_m3=float(equivalent_volume),
        tone_duration_s=float(tone_duration),
        liquid_cavity_round_trip_s=float(cavity_round_trip),
        liquid_cavity_round_trip_cycles=float(cavity_round_trip * frequency_hz),
        first_cavity_return_after_drive=bool(cavity_round_trip > tone_duration),
    )


def plane_wave_intensity_w_m2(
    peak_pressure_pa: float,
    *,
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> float:
    """Cycle-averaged intensity of a harmonic plane wave from peak pressure."""

    if not np.isfinite(peak_pressure_pa) or peak_pressure_pa < 0.0:
        raise ValueError("peak_pressure_pa must be finite and >= 0")
    if not np.isfinite(density_kg_m3) or density_kg_m3 <= 0.0:
        raise ValueError("density_kg_m3 must be finite and > 0")
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s must be finite and > 0")
    return float(
        peak_pressure_pa**2 / (2.0 * density_kg_m3 * sound_speed_m_s)
    )


def perfect_reflector_radiation_pressure_pa(
    incident_intensity_w_m2: float,
    *,
    sound_speed_m_s: float,
) -> float:
    """Ideal normal-incidence radiation-pressure scale ``2 I / c``.

    This is an upper-bound momentum-flux scale for perfect reflection, not a
    free-surface ejection threshold.
    """

    if (
        not np.isfinite(incident_intensity_w_m2)
        or incident_intensity_w_m2 < 0.0
    ):
        raise ValueError("incident_intensity_w_m2 must be finite and >= 0")
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s must be finite and > 0")
    return float(2.0 * incident_intensity_w_m2 / sound_speed_m_s)


def capillary_pressure_scale_pa(
    *,
    surface_tension_n_m: float,
    diameter_m: float,
) -> float:
    """Laplace-pressure scale ``4 sigma / diameter`` for a spherical drop."""

    if not np.isfinite(surface_tension_n_m) or surface_tension_n_m <= 0.0:
        raise ValueError("surface_tension_n_m must be finite and > 0")
    if not np.isfinite(diameter_m) or diameter_m <= 0.0:
        raise ValueError("diameter_m must be finite and > 0")
    return float(4.0 * surface_tension_n_m / diameter_m)
