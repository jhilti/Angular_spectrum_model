"""Measured DMSO/water mixture properties near room temperature.

Data are from Table 2 of:

M. M. Palaiologou, G. K. Arianas, N. G. Tsierkezos,
"Thermodynamic Investigation of Dimethyl Sulfoxide Binary Mixtures at
293.15 and 313.15 K", Journal of Solution Chemistry 35 (2006), 1551-1565.
https://doi.org/10.1007/s10953-006-9082-5

Composition interpolation is performed in DMSO mole fraction at each measured
temperature, followed by linear temperature interpolation.  This preserves the
strongly non-linear speed-of-sound curve of water/DMSO mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


ConcentrationBasis = Literal["volume", "mass", "mole"]

_DMSO_MOLAR_MASS_G_MOL = 78.133
_WATER_MOLAR_MASS_G_MOL = 18.01528

# DMSO mole fraction, density [g/cm3], speed of sound [m/s].
_DATA_20C = np.array(
    [
        [0.0000, 0.99822, 1483.1],
        [0.0541, 1.02328, 1582.4],
        [0.0906, 1.04118, 1642.7],
        [0.1326, 1.05627, 1684.5],
        [0.1896, 1.07291, 1715.6],
        [0.2561, 1.08618, 1721.0],
        [0.2866, 1.09058, 1716.0],
        [0.3769, 1.09866, 1687.7],
        [0.4793, 1.10262, 1647.5],
        [0.6076, 1.10358, 1599.9],
        [0.7492, 1.10266, 1556.9],
        [0.8406, 1.10179, 1534.4],
        [1.0000, 1.10035, 1502.6],
    ],
    dtype=float,
)

_DATA_40C = np.array(
    [
        [0.0000, 0.99225, 1529.2],
        [0.0990, 1.03353, 1636.0],
        [0.1983, 1.06043, 1670.8],
        [0.2952, 1.07509, 1656.0],
        [0.3896, 1.08190, 1621.1],
        [0.5016, 1.08465, 1575.0],
        [0.6168, 1.08465, 1531.7],
        [0.6739, 1.08416, 1512.8],
        [0.7954, 1.08272, 1478.2],
        [0.8706, 1.08167, 1460.7],
        [1.0000, 1.08022, 1434.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class DMSOWaterProperties:
    """Interpolated acoustic properties for a DMSO/water mixture."""

    dmso_fraction: float
    basis: ConcentrationBasis
    temperature_c: float
    dmso_mole_fraction: float
    density_kg_m3: float
    sound_speed_m_s: float


def _validate_fraction(value: ArrayLike) -> NDArray[np.float64]:
    fraction = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(fraction)) or np.any(
        (fraction < 0.0) | (fraction > 1.0)
    ):
        raise ValueError("DMSO fraction must be finite and lie in [0, 1]")
    return fraction


def _pure_density_g_cm3(temperature_c: float, component: str) -> float:
    temperature_weight = (temperature_c - 20.0) / 20.0
    column = 1
    if component == "DMSO":
        value_20 = _DATA_20C[-1, column]
        value_40 = _DATA_40C[-1, column]
    else:
        value_20 = _DATA_20C[0, column]
        value_40 = _DATA_40C[0, column]
    return float(value_20 + temperature_weight * (value_40 - value_20))


def dmso_concentration_to_mole_fraction(
    dmso_fraction: ArrayLike,
    *,
    basis: ConcentrationBasis = "volume",
    temperature_c: float = 22.0,
) -> NDArray[np.float64]:
    """Convert DMSO volume, mass, or mole fraction to mole fraction.

    For ``basis="volume"``, volumes of the two neat components before mixing
    are used.  Volume contraction on mixing makes this an approximation if the
    laboratory percentage is defined against the final solution volume.
    """

    fraction = _validate_fraction(dmso_fraction)
    if basis == "mole":
        return fraction
    if basis == "mass":
        dmso_moles = fraction / _DMSO_MOLAR_MASS_G_MOL
        water_moles = (1.0 - fraction) / _WATER_MOLAR_MASS_G_MOL
    elif basis == "volume":
        if not 20.0 <= temperature_c <= 40.0:
            raise ValueError("temperature_c must lie between 20 and 40 degC")
        rho_dmso = _pure_density_g_cm3(temperature_c, "DMSO")
        rho_water = _pure_density_g_cm3(temperature_c, "water")
        dmso_moles = fraction * rho_dmso / _DMSO_MOLAR_MASS_G_MOL
        water_moles = (
            (1.0 - fraction) * rho_water / _WATER_MOLAR_MASS_G_MOL
        )
    else:
        raise ValueError("basis must be 'volume', 'mass', or 'mole'")
    total_moles = dmso_moles + water_moles
    return np.divide(
        dmso_moles,
        total_moles,
        out=np.zeros_like(total_moles),
        where=total_moles > 0.0,
    )


def dmso_water_properties(
    dmso_fraction: float,
    *,
    basis: ConcentrationBasis = "volume",
    temperature_c: float = 22.0,
) -> DMSOWaterProperties:
    """Interpolate density and speed of sound for a DMSO/water mixture."""

    if not np.isfinite(temperature_c) or not 20.0 <= temperature_c <= 40.0:
        raise ValueError("temperature_c must lie between 20 and 40 degC")
    fraction = float(_validate_fraction(dmso_fraction))
    mole_fraction = float(
        dmso_concentration_to_mole_fraction(
            fraction, basis=basis, temperature_c=temperature_c
        )
    )
    temperature_weight = (temperature_c - 20.0) / 20.0
    density_20 = np.interp(
        mole_fraction, _DATA_20C[:, 0], _DATA_20C[:, 1]
    )
    density_40 = np.interp(
        mole_fraction, _DATA_40C[:, 0], _DATA_40C[:, 1]
    )
    speed_20 = np.interp(
        mole_fraction, _DATA_20C[:, 0], _DATA_20C[:, 2]
    )
    speed_40 = np.interp(
        mole_fraction, _DATA_40C[:, 0], _DATA_40C[:, 2]
    )
    density = density_20 + temperature_weight * (density_40 - density_20)
    speed = speed_20 + temperature_weight * (speed_40 - speed_20)
    return DMSOWaterProperties(
        dmso_fraction=fraction,
        basis=basis,
        temperature_c=temperature_c,
        dmso_mole_fraction=mole_fraction,
        density_kg_m3=float(density * 1000.0),
        sound_speed_m_s=float(speed),
    )
