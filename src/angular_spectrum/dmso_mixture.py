"""Measured DMSO/water mixture properties near room temperature.

Data are from Table 2 of:

M. M. Palaiologou, G. K. Arianas, N. G. Tsierkezos,
"Thermodynamic Investigation of Dimethyl Sulfoxide Binary Mixtures at
293.15 and 313.15 K", Journal of Solution Chemistry 35 (2006), 1551-1565.
https://doi.org/10.1007/s10953-006-9082-5

Pure-water sound speed is evaluated with the Marczak polynomial:

A. Marczak, "Water as a standard in the measurements of speed of sound in
liquids", Journal of the Acoustical Society of America 102 (1997), 2776-2779.
https://doi.org/10.1121/1.420332

Pure-water density is evaluated with the atmospheric-pressure correlation of:

G. S. Kell, "Density, Thermal Expansivity, and Compressibility of Liquid Water
from 0 deg to 150 deg C", Journal of Chemical & Engineering Data 20 (1975),
97-105. https://doi.org/10.1021/je60064a005

Dynamic viscosity and liquid/air surface tension are supplied by the
companion :mod:`angular_spectrum.dmso_transport` module.  Those properties
come from independent measured concentration series and are returned on the
same composition and temperature basis as the acoustic properties.

Composition interpolation is performed in DMSO mole fraction at each measured
temperature, followed by linear temperature interpolation. The pure-water
table endpoints are anchored to the Marczak/Kell correlations so the mixture
model is continuous at zero DMSO. This preserves the strongly non-linear
speed-of-sound curve of water/DMSO mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dmso_transport import (
    dynamic_viscosity_from_mole_fraction_pa_s,
    surface_tension_from_mole_fraction_n_m,
)


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
    """Interpolated acoustic and free-surface properties of DMSO/water."""

    dmso_fraction: float
    basis: ConcentrationBasis
    temperature_c: float
    dmso_mole_fraction: float
    density_kg_m3: float
    sound_speed_m_s: float
    dynamic_viscosity_pa_s: float
    surface_tension_n_m: float
    surface_tension_temperature_extrapolated: bool


@dataclass(frozen=True)
class WaterProperties:
    """Atmospheric-pressure properties of pure water."""

    temperature_c: float
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
        return _water_density_kg_m3(temperature_c) / 1000.0
    return float(value_20 + temperature_weight * (value_40 - value_20))


def _water_sound_speed_m_s(temperature_c: float) -> float:
    """Marczak polynomial for pure water at atmospheric pressure."""

    temperature = float(temperature_c)
    return float(
        1402.385
        + 5.038813 * temperature
        - 5.799136e-2 * temperature**2
        + 3.287156e-4 * temperature**3
        - 1.398845e-6 * temperature**4
        + 2.787860e-9 * temperature**5
    )


def _water_density_kg_m3(temperature_c: float) -> float:
    """Kell atmospheric-pressure density correlation for pure water."""

    temperature = float(temperature_c)
    return float(
        1000.0
        * (
            1.0
            - (temperature + 288.9414)
            / (508929.2 * (temperature + 68.12963))
            * (temperature - 3.9863) ** 2
        )
    )


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
    """Return acoustic, viscosity, and surface-tension mixture properties.

    Surface tension at 20 to below 25 degC is extrapolated from measurements
    at 25--55 degC and is explicitly identified by
    ``surface_tension_temperature_extrapolated``.  At exactly zero DMSO the
    IAPWS pure-water correlation is used directly and no extrapolation flag is
    set.
    """

    if not np.isfinite(temperature_c) or not 20.0 <= temperature_c <= 40.0:
        raise ValueError("temperature_c must lie between 20 and 40 degC")
    fraction = float(_validate_fraction(dmso_fraction))
    mole_fraction = float(
        dmso_concentration_to_mole_fraction(
            fraction, basis=basis, temperature_c=temperature_c
        )
    )
    if fraction == 0.0:
        pure_water = water_properties(temperature_c)
        dynamic_viscosity = dynamic_viscosity_from_mole_fraction_pa_s(
            0.0,
            temperature_c,
        )
        surface_tension, surface_tension_extrapolated = (
            surface_tension_from_mole_fraction_n_m(0.0, temperature_c)
        )
        return DMSOWaterProperties(
            dmso_fraction=0.0,
            basis=basis,
            temperature_c=temperature_c,
            dmso_mole_fraction=0.0,
            density_kg_m3=pure_water.density_kg_m3,
            sound_speed_m_s=pure_water.sound_speed_m_s,
            dynamic_viscosity_pa_s=dynamic_viscosity,
            surface_tension_n_m=surface_tension,
            surface_tension_temperature_extrapolated=(
                surface_tension_extrapolated
            ),
        )
    temperature_weight = (temperature_c - 20.0) / 20.0
    composition_grid = np.union1d(_DATA_20C[:, 0], _DATA_40C[:, 0])
    density_20_grid = np.interp(
        composition_grid,
        _DATA_20C[:, 0],
        _DATA_20C[:, 1],
    )
    density_40_grid = np.interp(
        composition_grid,
        _DATA_40C[:, 0],
        _DATA_40C[:, 1],
    )
    speed_20_grid = np.interp(
        composition_grid,
        _DATA_20C[:, 0],
        _DATA_20C[:, 2],
    )
    speed_40_grid = np.interp(
        composition_grid,
        _DATA_40C[:, 0],
        _DATA_40C[:, 2],
    )
    density_data = density_20_grid + temperature_weight * (
        density_40_grid - density_20_grid
    )
    speed_data = speed_20_grid + temperature_weight * (
        speed_40_grid - speed_20_grid
    )
    # Linear 20--40 °C interpolation is appropriate for the measured mixture
    # points, but pure-water c(T) is visibly curved. Anchor the endpoint at the
    # requested temperature before composition interpolation to avoid a jump
    # between exactly zero and an infinitesimal DMSO fraction.
    density_data[0] = _water_density_kg_m3(temperature_c) / 1000.0
    speed_data[0] = _water_sound_speed_m_s(temperature_c)
    density = np.interp(mole_fraction, composition_grid, density_data)
    speed = np.interp(mole_fraction, composition_grid, speed_data)
    dynamic_viscosity = dynamic_viscosity_from_mole_fraction_pa_s(
        mole_fraction,
        temperature_c,
    )
    surface_tension, surface_tension_extrapolated = (
        surface_tension_from_mole_fraction_n_m(
            mole_fraction,
            temperature_c,
        )
    )
    return DMSOWaterProperties(
        dmso_fraction=fraction,
        basis=basis,
        temperature_c=temperature_c,
        dmso_mole_fraction=mole_fraction,
        density_kg_m3=float(density * 1000.0),
        sound_speed_m_s=float(speed),
        dynamic_viscosity_pa_s=dynamic_viscosity,
        surface_tension_n_m=surface_tension,
        surface_tension_temperature_extrapolated=(
            surface_tension_extrapolated
        ),
    )


def dmso_water_dynamic_viscosity_pa_s(
    dmso_fraction: float,
    *,
    basis: ConcentrationBasis = "volume",
    temperature_c: float = 22.0,
) -> float:
    """Return measured/interpolated dynamic viscosity in Pa s.

    The source isotherms are at 20, 30, and 40 degC.  No temperature
    extrapolation is permitted.
    """

    if not np.isfinite(temperature_c) or not 20.0 <= temperature_c <= 40.0:
        raise ValueError("temperature_c must lie between 20 and 40 degC")
    fraction = float(_validate_fraction(dmso_fraction))
    mole_fraction = float(
        dmso_concentration_to_mole_fraction(
            fraction,
            basis=basis,
            temperature_c=temperature_c,
        )
    )
    return dynamic_viscosity_from_mole_fraction_pa_s(
        mole_fraction,
        temperature_c,
    )


def dmso_water_surface_tension_n_m(
    dmso_fraction: float,
    *,
    basis: ConcentrationBasis = "volume",
    temperature_c: float = 22.0,
) -> float:
    """Return DMSO/water liquid/air surface tension in N/m.

    Measurements cover 25--55 degC.  Values at 20 to below 25 degC are a
    continuity-preserving short extrapolation, except for pure water, which
    uses IAPWS directly.  Volume-basis conversion remains limited to 20--40
    degC because its neat-component density model is only validated there;
    mole and mass basis may use the full 20--55 degC surface-tension range.
    """

    if not np.isfinite(temperature_c) or not 20.0 <= temperature_c <= 55.0:
        raise ValueError("temperature_c must lie between 20 and 55 degC")
    fraction = float(_validate_fraction(dmso_fraction))
    mole_fraction = float(
        dmso_concentration_to_mole_fraction(
            fraction,
            basis=basis,
            temperature_c=temperature_c,
        )
    )
    surface_tension, _ = surface_tension_from_mole_fraction_n_m(
        mole_fraction,
        temperature_c,
    )
    return surface_tension


def water_properties(temperature_c: float = 22.0) -> WaterProperties:
    """Return temperature-consistent pure-water density and sound speed.

    The Marczak sound-speed polynomial is used over its stated 0--95 degC
    range. Density follows the Kell atmospheric-pressure correlation. The
    DMSO-mixture interpolation remains limited to 20--40 degC.
    """

    if not np.isfinite(temperature_c) or not 0.0 <= temperature_c <= 95.0:
        raise ValueError("temperature_c must lie between 0 and 95 degC")
    return WaterProperties(
        temperature_c=temperature_c,
        density_kg_m3=_water_density_kg_m3(temperature_c),
        sound_speed_m_s=_water_sound_speed_m_s(temperature_c),
    )
