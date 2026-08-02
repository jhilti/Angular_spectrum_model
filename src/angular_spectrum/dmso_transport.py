"""Transport and free-surface properties of DMSO/water mixtures.

Dynamic-viscosity data are from Table 2 of Omota et al., *Revue
Roumaine de Chimie* 53 (2008), 977--988.  The source reports viscosity at
293.15, 303.15, and 313.15 K as a function of water mole fraction; the table
below is stored as DMSO mole fraction.
https://revroum.lew.ro/wp-content/uploads/2008/RRCh_11_2008/Art%2001.pdf

Surface-tension data are the DMSO/water maximal-bubble-pressure measurements
of Markarian and Terzyan, *Journal of Chemical & Engineering Data* 52 (2007),
1704--1709, DOI 10.1021/je7001013, as distributed by NIST ThermoML.  The
mixture table contains 112 measurements from 298.15 to 328.15 K.  Pure-DMSO
values come from the same source.  Pure-water endpoints use IAPWS
R1-76(2014).
https://trc.nist.gov/ThermoML/10.1021/je7001013.html
https://www.iapws.org/relguide/Surf-H2O.html

Viscosity is interpolated in log(eta) versus composition and reciprocal
absolute temperature.  Surface tension is interpolated linearly in
composition on each measured isotherm and then linearly in temperature.
For 20--25 degC, mixture surface tension is a short, explicitly flagged
extrapolation from the 25 degC isotherm using the slope fitted over all seven
measured temperatures.  It must not be mistaken for a measurement at 22 degC.
"""

from __future__ import annotations

import numpy as np


# DMSO mole fraction followed by dynamic viscosity [mPa s] at 20, 30, 40 C.
_VISCOSITY_DATA = np.array(
    [
        [0.0000, 1.0050, 0.8007, 0.6560],
        [0.0351, 1.3067, 1.0476, 0.8404],
        [0.0782, 1.8706, 1.4343, 1.1383],
        [0.1363, 2.7242, 1.9906, 1.5702],
        [0.2158, 3.7056, 2.6427, 2.0738],
        [0.2979, 4.2602, 3.0584, 2.3885],
        [0.3459, 4.3588, 3.1679, 2.4679],
        [0.4331, 4.2124, 3.1509, 2.4516],
        [0.5205, 3.8322, 2.9441, 2.3048],
        [0.6406, 3.2743, 2.5523, 2.0457],
        [0.6962, 3.0658, 2.3833, 1.9388],
        [0.7585, 2.8714, 2.2237, 1.8400],
        [0.8287, 2.6765, 2.0820, 1.7480],
        [0.9085, 2.4515, 1.9550, 1.6545],
        [1.0000, 2.2255, 1.8180, 1.5553],
    ],
    dtype=float,
)
_VISCOSITY_TEMPERATURE_K = np.array([293.15, 303.15, 313.15])


# Temperature [degC], DMSO mole fraction, surface tension [N/m].  Composition
# grids around x=0.3 differ between the lower and upper four isotherms, so the
# authoritative long-form layout is preserved rather than reshaped.
_SURFACE_TENSION_DATA = np.array(
    [
        [25, 0.0121, 0.0710],
        [25, 0.0202, 0.0697],
        [25, 0.1044, 0.0612],
        [25, 0.1979, 0.0567],
        [25, 0.2507, 0.0554],
        [25, 0.2999, 0.0533],
        [25, 0.3187, 0.0531],
        [25, 0.3536, 0.0526],
        [25, 0.3802, 0.0520],
        [25, 0.4191, 0.0510],
        [25, 0.4523, 0.0503],
        [25, 0.5010, 0.0483],
        [25, 0.5900, 0.0460],
        [25, 0.7027, 0.0440],
        [25, 0.8088, 0.0426],
        [25, 0.8945, 0.0420],
        [30, 0.0121, 0.0701],
        [30, 0.0202, 0.0687],
        [30, 0.1044, 0.0606],
        [30, 0.1979, 0.0558],
        [30, 0.2507, 0.0545],
        [30, 0.2999, 0.0524],
        [30, 0.3187, 0.0523],
        [30, 0.3536, 0.0517],
        [30, 0.3802, 0.0512],
        [30, 0.4191, 0.0500],
        [30, 0.4523, 0.0499],
        [30, 0.5010, 0.0474],
        [30, 0.5900, 0.0452],
        [30, 0.7027, 0.0434],
        [30, 0.8088, 0.0421],
        [30, 0.8945, 0.0414],
        [35, 0.0121, 0.0692],
        [35, 0.0202, 0.0679],
        [35, 0.1044, 0.0598],
        [35, 0.1979, 0.0548],
        [35, 0.2507, 0.0532],
        [35, 0.2999, 0.0515],
        [35, 0.3187, 0.0514],
        [35, 0.3536, 0.0506],
        [35, 0.3802, 0.0503],
        [35, 0.4191, 0.0492],
        [35, 0.4523, 0.0491],
        [35, 0.5010, 0.0467],
        [35, 0.5900, 0.0446],
        [35, 0.7027, 0.0426],
        [35, 0.8088, 0.0415],
        [35, 0.8945, 0.0409],
        [40, 0.0121, 0.0676],
        [40, 0.0202, 0.0669],
        [40, 0.1044, 0.0587],
        [40, 0.1979, 0.0539],
        [40, 0.2507, 0.0523],
        [40, 0.2999, 0.0506],
        [40, 0.3187, 0.0504],
        [40, 0.3536, 0.0500],
        [40, 0.3802, 0.0495],
        [40, 0.4191, 0.0484],
        [40, 0.4523, 0.0482],
        [40, 0.5010, 0.0460],
        [40, 0.5900, 0.0438],
        [40, 0.7027, 0.0421],
        [40, 0.8088, 0.0409],
        [40, 0.8945, 0.0403],
        [45, 0.0121, 0.0664],
        [45, 0.0202, 0.0658],
        [45, 0.1044, 0.0579],
        [45, 0.1979, 0.0530],
        [45, 0.2507, 0.0512],
        [45, 0.3218, 0.0507],
        [45, 0.3506, 0.0497],
        [45, 0.3549, 0.0494],
        [45, 0.3802, 0.0486],
        [45, 0.4191, 0.0475],
        [45, 0.4523, 0.0473],
        [45, 0.5010, 0.0451],
        [45, 0.5900, 0.0431],
        [45, 0.7027, 0.0414],
        [45, 0.8088, 0.0404],
        [45, 0.8945, 0.0398],
        [50, 0.0121, 0.0653],
        [50, 0.0202, 0.0648],
        [50, 0.1044, 0.0570],
        [50, 0.1979, 0.0522],
        [50, 0.2507, 0.0503],
        [50, 0.3218, 0.0499],
        [50, 0.3506, 0.0487],
        [50, 0.3549, 0.0484],
        [50, 0.3802, 0.0477],
        [50, 0.4191, 0.0467],
        [50, 0.4523, 0.0463],
        [50, 0.5010, 0.0444],
        [50, 0.5900, 0.0423],
        [50, 0.7027, 0.0408],
        [50, 0.8088, 0.0398],
        [50, 0.8945, 0.0392],
        [55, 0.0121, 0.0651],
        [55, 0.0202, 0.0638],
        [55, 0.1044, 0.0551],
        [55, 0.1979, 0.0511],
        [55, 0.2507, 0.0494],
        [55, 0.3218, 0.0490],
        [55, 0.3506, 0.0477],
        [55, 0.3549, 0.0474],
        [55, 0.3802, 0.0468],
        [55, 0.4191, 0.0459],
        [55, 0.4523, 0.0453],
        [55, 0.5010, 0.0435],
        [55, 0.5900, 0.0417],
        [55, 0.7027, 0.0402],
        [55, 0.8088, 0.0392],
        [55, 0.8945, 0.0387],
    ],
    dtype=float,
)
_SURFACE_TENSION_TEMPERATURE_C = np.arange(25.0, 56.0, 5.0)
_PURE_DMSO_SURFACE_TENSION_N_M = np.array(
    [0.0417, 0.0410, 0.0403, 0.0397, 0.0391, 0.0385, 0.0379],
    dtype=float,
)


def _validate_mole_fraction(dmso_mole_fraction: float) -> float:
    fraction = float(dmso_mole_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("DMSO mole fraction must be finite and lie in [0, 1]")
    return fraction


def dynamic_viscosity_from_mole_fraction_pa_s(
    dmso_mole_fraction: float,
    temperature_c: float,
) -> float:
    """Return DMSO/water dynamic viscosity in Pa s for 20--40 degC."""

    fraction = _validate_mole_fraction(dmso_mole_fraction)
    if not np.isfinite(temperature_c) or not 20.0 <= temperature_c <= 40.0:
        raise ValueError("temperature_c must lie between 20 and 40 degC")

    log_eta_by_temperature = np.array(
        [
            np.interp(
                fraction,
                _VISCOSITY_DATA[:, 0],
                np.log(_VISCOSITY_DATA[:, column]),
            )
            for column in range(1, 4)
        ]
    )
    inverse_temperature = 1.0 / _VISCOSITY_TEMPERATURE_K
    log_eta_mpa_s = np.interp(
        1.0 / (temperature_c + 273.15),
        inverse_temperature[::-1],
        log_eta_by_temperature[::-1],
    )
    return float(np.exp(log_eta_mpa_s) * 1.0e-3)


def water_surface_tension_n_m(temperature_c: float) -> float:
    """IAPWS R1-76(2014) surface tension of water against its vapor."""

    if not np.isfinite(temperature_c) or not 0.01 <= temperature_c <= 373.946:
        raise ValueError(
            "temperature_c must lie between the water triple and critical points"
        )
    temperature_k = temperature_c + 273.15
    tau = 1.0 - temperature_k / 647.096
    return float(0.2358 * tau**1.256 * (1.0 - 0.625 * tau))


def surface_tension_from_mole_fraction_n_m(
    dmso_mole_fraction: float,
    temperature_c: float,
) -> tuple[float, bool]:
    """Return liquid/air surface tension and a temperature-extrapolation flag.

    Mixture measurements cover 25--55 degC.  Values from 20 to below 25 degC
    use the short extrapolation described in the module documentation.  Pure
    water is evaluated directly with IAPWS and therefore is not extrapolated.
    """

    fraction = _validate_mole_fraction(dmso_mole_fraction)
    if not np.isfinite(temperature_c) or not 20.0 <= temperature_c <= 55.0:
        raise ValueError("temperature_c must lie between 20 and 55 degC")
    if fraction == 0.0:
        return water_surface_tension_n_m(temperature_c), False

    values_by_temperature = []
    for index, measured_temperature_c in enumerate(
        _SURFACE_TENSION_TEMPERATURE_C
    ):
        isotherm = _SURFACE_TENSION_DATA[
            _SURFACE_TENSION_DATA[:, 0] == measured_temperature_c
        ]
        composition = np.concatenate(
            ([0.0], isotherm[:, 1], [1.0])
        )
        surface_tension = np.concatenate(
            (
                [water_surface_tension_n_m(measured_temperature_c)],
                isotherm[:, 2],
                [_PURE_DMSO_SURFACE_TENSION_N_M[index]],
            )
        )
        values_by_temperature.append(
            np.interp(fraction, composition, surface_tension)
        )
    values = np.asarray(values_by_temperature)

    if temperature_c >= 25.0:
        return (
            float(
                np.interp(
                    temperature_c,
                    _SURFACE_TENSION_TEMPERATURE_C,
                    values,
                )
            ),
            False,
        )

    temperature_slope = np.polyfit(
        _SURFACE_TENSION_TEMPERATURE_C,
        values,
        deg=1,
    )[0]
    extrapolated_value = values[0] + temperature_slope * (
        temperature_c - 25.0
    )
    return float(extrapolated_value), True
