"""Material descriptions and attenuation conventions.

The phasor convention used throughout the package is ``exp(-i omega t)``.
Forward waves therefore vary as ``exp(+i k_z z)``.  A positive imaginary
part of the wavenumber produces attenuation in the forward direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_DB_TO_NP = np.log(10.0) / 20.0


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")


def _require_nonnegative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")


@dataclass(frozen=True)
class Fluid:
    """Homogeneous acoustic fluid.

    ``attenuation_db_per_m`` is an amplitude loss in dB/m at
    ``attenuation_reference_hz``.  It follows a power law in frequency.
    """

    name: str
    density_kg_m3: float
    sound_speed_m_s: float
    attenuation_db_per_m: float = 0.0
    attenuation_power: float = 1.0
    attenuation_reference_hz: float = 10.0e6

    def __post_init__(self) -> None:
        _require_positive("density_kg_m3", self.density_kg_m3)
        _require_positive("sound_speed_m_s", self.sound_speed_m_s)
        _require_nonnegative("attenuation_db_per_m", self.attenuation_db_per_m)
        _require_nonnegative("attenuation_power", self.attenuation_power)
        _require_positive(
            "attenuation_reference_hz", self.attenuation_reference_hz
        )

    def attenuation_np_m(self, frequency_hz: float) -> float:
        """Return amplitude attenuation in nepers per metre."""

        _require_positive("frequency_hz", frequency_hz)
        scale = (frequency_hz / self.attenuation_reference_hz) ** (
            self.attenuation_power
        )
        return self.attenuation_db_per_m * _DB_TO_NP * scale

    def wavenumber(self, frequency_hz: float) -> complex:
        """Return ``omega/c + i alpha`` for the package phasor convention."""

        omega = 2.0 * np.pi * frequency_hz
        return complex(
            omega / self.sound_speed_m_s,
            self.attenuation_np_m(frequency_hz),
        )


@dataclass(frozen=True)
class ElasticSolid:
    """Homogeneous isotropic elastic solid with P and SV propagation."""

    name: str
    density_kg_m3: float
    longitudinal_speed_m_s: float
    shear_speed_m_s: float
    longitudinal_attenuation_db_per_m: float = 0.0
    shear_attenuation_db_per_m: float = 0.0
    attenuation_power: float = 1.0
    attenuation_reference_hz: float = 10.0e6

    def __post_init__(self) -> None:
        _require_positive("density_kg_m3", self.density_kg_m3)
        _require_positive("longitudinal_speed_m_s", self.longitudinal_speed_m_s)
        _require_positive("shear_speed_m_s", self.shear_speed_m_s)
        if self.longitudinal_speed_m_s <= np.sqrt(4.0 / 3.0) * self.shear_speed_m_s:
            raise ValueError(
                "longitudinal_speed_m_s is too small for a stable isotropic solid"
            )
        _require_nonnegative(
            "longitudinal_attenuation_db_per_m",
            self.longitudinal_attenuation_db_per_m,
        )
        _require_nonnegative(
            "shear_attenuation_db_per_m", self.shear_attenuation_db_per_m
        )
        _require_nonnegative("attenuation_power", self.attenuation_power)
        _require_positive(
            "attenuation_reference_hz", self.attenuation_reference_hz
        )

    @classmethod
    def from_longitudinal_speed_and_poisson(
        cls,
        *,
        name: str,
        density_kg_m3: float,
        longitudinal_speed_m_s: float,
        poisson_ratio: float,
        longitudinal_attenuation_db_per_m: float = 0.0,
        shear_attenuation_db_per_m: float = 0.0,
        attenuation_power: float = 1.0,
        attenuation_reference_hz: float = 10.0e6,
    ) -> "ElasticSolid":
        """Construct a solid when the shear speed has not yet been measured."""

        if not np.isfinite(poisson_ratio) or not (-1.0 < poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must lie between -1 and 0.5")
        ratio = (1.0 - 2.0 * poisson_ratio) / (2.0 * (1.0 - poisson_ratio))
        shear_speed = longitudinal_speed_m_s * np.sqrt(ratio)
        return cls(
            name=name,
            density_kg_m3=density_kg_m3,
            longitudinal_speed_m_s=longitudinal_speed_m_s,
            shear_speed_m_s=float(shear_speed),
            longitudinal_attenuation_db_per_m=longitudinal_attenuation_db_per_m,
            shear_attenuation_db_per_m=shear_attenuation_db_per_m,
            attenuation_power=attenuation_power,
            attenuation_reference_hz=attenuation_reference_hz,
        )

    @property
    def poisson_ratio(self) -> float:
        cl2 = self.longitudinal_speed_m_s**2
        cs2 = self.shear_speed_m_s**2
        return (cl2 - 2.0 * cs2) / (2.0 * (cl2 - cs2))

    def _attenuation_np_m(self, frequency_hz: float, value_db_m: float) -> float:
        _require_positive("frequency_hz", frequency_hz)
        scale = (frequency_hz / self.attenuation_reference_hz) ** (
            self.attenuation_power
        )
        return value_db_m * _DB_TO_NP * scale

    def longitudinal_wavenumber(self, frequency_hz: float) -> complex:
        omega = 2.0 * np.pi * frequency_hz
        alpha = self._attenuation_np_m(
            frequency_hz, self.longitudinal_attenuation_db_per_m
        )
        return complex(omega / self.longitudinal_speed_m_s, alpha)

    def shear_wavenumber(self, frequency_hz: float) -> complex:
        omega = 2.0 * np.pi * frequency_hz
        alpha = self._attenuation_np_m(
            frequency_hz, self.shear_attenuation_db_per_m
        )
        return complex(omega / self.shear_speed_m_s, alpha)

    def complex_lame_parameters(self, frequency_hz: float) -> tuple[complex, complex]:
        """Return frequency-consistent complex ``(lambda, mu)`` parameters."""

        omega = 2.0 * np.pi * frequency_hz
        k_l = self.longitudinal_wavenumber(frequency_hz)
        k_s = self.shear_wavenumber(frequency_hz)
        c_l_complex = omega / k_l
        c_s_complex = omega / k_s
        mu = self.density_kg_m3 * c_s_complex**2
        lam = self.density_kg_m3 * c_l_complex**2 - 2.0 * mu
        return lam, mu


@dataclass(frozen=True)
class ElasticPlate:
    """Finite isotropic elastic plate between two fluids."""

    solid: ElasticSolid
    thickness_m: float

    def __post_init__(self) -> None:
        _require_positive("thickness_m", self.thickness_m)
