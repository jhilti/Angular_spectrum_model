"""Sweep 70--100 vol.% DMSO at a fixed 150-cycle tone length.

The aperture-pressure drive is held constant across the sweep.  One common
illustrative scale is chosen so that the 80 vol.% reference case has 1 MPa
incident peak pressure on axis at the 4.22 mm meniscus.  Every concentration
then receives its own density, sound speed, PP transmission, refracted focus,
spatial meniscus field, radiation stress, and free-surface simulation.

Density, sound speed, dynamic viscosity, and liquid/air surface tension vary
with concentration and temperature through the repository's measured DMSO
property model.  A fixed-transport control repeats the hydrodynamic solve with
the 80 vol.% viscosity and surface tension so their incremental effect can be
separated from the acoustic changes.  Contact angle, liquid attenuation, and
electrical transducer response remain fixed or omitted.  Finite liquid-cavity
build-up is also omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from angular_spectrum import (
    AngularSpectrumModel,
    CartesianGrid,
    ContactLineModel,
    DMSOWaterProperties,
    ElasticPlate,
    ElasticSolid,
    Fluid,
    FocusedCircularAperture,
    FreeSurfaceLiquid,
    dmso_water_properties,
    radiation_stress_from_incident_pressure_envelope,
    raised_cosine_tone_envelope,
    simulate_axisymmetric_free_surface,
    water_properties,
)


FREQUENCY_HZ = 10.0e6
TONE_CYCLES = 150.0
TEMPERATURE_C = 22.0
LIQUID_DEPTH_M = 4.22e-3
WELL_RADIUS_M = 1.65e-3
WATER_PATH_M = 19.1e-3
PP_THICKNESS_M = 0.78e-3
REFERENCE_DMSO_PERCENT = 80.0
REFERENCE_MENISCUS_PRESSURE_PA = 1.0e6
FIXED_CONTACT_ANGLE_DEG = 90.0
POST_BURST_TIME_S = 600.0e-6
TIME_STEP_S = 0.20e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-percent", type=float, default=70.0)
    parser.add_argument("--max-percent", type=float, default=100.0)
    parser.add_argument("--step-percent", type=float, default=1.0)
    parser.add_argument(
        "--reference-meniscus-pressure-mpa",
        type=float,
        default=REFERENCE_MENISCUS_PRESSURE_PA * 1.0e-6,
        help="80%% reference pressure used to set one common aperture scale",
    )
    parser.add_argument(
        "--absolute-calibration",
        action="store_true",
        help="assert that the 80%% pressure reference has absolute calibration",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results"),
    )
    return parser.parse_args()


def concentration_values(
    minimum_percent: float,
    maximum_percent: float,
    step_percent: float,
) -> np.ndarray:
    """Return an inclusive concentration grid."""

    if not 0.0 <= minimum_percent <= maximum_percent <= 100.0:
        raise ValueError("require 0 <= min-percent <= max-percent <= 100")
    if step_percent <= 0.0:
        raise ValueError("step-percent must be > 0")
    count = int(np.floor((maximum_percent - minimum_percent) / step_percent)) + 1
    values = minimum_percent + step_percent * np.arange(count, dtype=float)
    if values[-1] < maximum_percent - 1.0e-12 * max(maximum_percent, 1.0):
        values = np.append(values, maximum_percent)
    return values


def quadratic_peak_position(
    coordinate_m: np.ndarray,
    amplitude: np.ndarray,
) -> tuple[float, bool]:
    """Estimate a sampled peak with a local quadratic interpolation."""

    index = int(np.argmax(amplitude))
    if index == 0 or index == coordinate_m.size - 1:
        return float(coordinate_m[index]), True
    denominator = (
        amplitude[index - 1]
        - 2.0 * amplitude[index]
        + amplitude[index + 1]
    )
    if denominator == 0.0:
        return float(coordinate_m[index]), False
    offset = 0.5 * (
        amplitude[index - 1] - amplitude[index + 1]
    ) / denominator
    return float(
        coordinate_m[index]
        + offset * (coordinate_m[index + 1] - coordinate_m[index])
    ), False


def build_model(
    properties: DMSOWaterProperties,
    *,
    grid: CartesianGrid,
    aperture: FocusedCircularAperture,
    water: Fluid,
    plate: ElasticPlate,
) -> AngularSpectrumModel:
    mixture = Fluid(
        f"{properties.dmso_fraction * 100:g}volpct_DMSO",
        properties.density_kg_m3,
        properties.sound_speed_m_s,
    )
    return AngularSpectrumModel(
        grid=grid,
        aperture=aperture,
        incident_fluid=water,
        plate=plate,
        transmitted_fluid=mixture,
        water_path_m=WATER_PATH_M,
        plate_radial_samples=4096,
    )


def radial_pressure_from_field(
    field: np.ndarray,
    *,
    grid: CartesianGrid,
    radius_m: np.ndarray,
    common_pressure_scale: float,
) -> tuple[np.ndarray, float]:
    """Convert the symmetric Cartesian ASM field to a radial magnitude."""

    centre_y, centre_x = grid.centre_index
    positive_x = grid.x_m >= 0.0
    positive_y = grid.y_m >= 0.0
    samples_x = grid.x_m[positive_x]
    samples_y = grid.y_m[positive_y]
    power_x = np.abs(field[centre_y, positive_x]) ** 2
    power_y = np.abs(field[positive_y, centre_x]) ** 2
    power = 0.5 * (power_x + np.interp(samples_x, samples_y, power_y))
    radial_pressure_pa = common_pressure_scale * np.sqrt(
        np.maximum(np.interp(radius_m, samples_x, power), 0.0)
    )
    relative_power = power / power[0]
    crossings = np.flatnonzero(relative_power <= 0.5)
    if crossings.size == 0 or crossings[0] == 0:
        raise RuntimeError("the ASM field does not contain its half-intensity radius")
    crossing = int(crossings[0])
    half_intensity_radius_m = float(
        np.interp(
            0.5,
            relative_power[crossing - 1 : crossing + 1][::-1],
            samples_x[crossing - 1 : crossing + 1][::-1],
        )
    )
    return radial_pressure_pa, half_intensity_radius_m


def main() -> None:
    args = parse_args()
    concentrations = concentration_values(
        args.min_percent,
        args.max_percent,
        args.step_percent,
    )
    if args.reference_meniscus_pressure_mpa <= 0.0:
        raise ValueError("reference-meniscus-pressure-mpa must be > 0")

    radius_m = np.linspace(0.0, WELL_RADIUS_M, 129)
    grid = CartesianGrid(nx=384, ny=384, dx_m=50.0e-6)
    aperture = FocusedCircularAperture(
        diameter_m=13.0e-3,
        focal_length_m=25.4e-3,
    )
    water_data = water_properties(TEMPERATURE_C)
    water = Fluid(
        f"water_{TEMPERATURE_C:g}C",
        water_data.density_kg_m3,
        water_data.sound_speed_m_s,
    )
    polypropylene = ElasticSolid.from_longitudinal_speed_and_poisson(
        name="polypropylene",
        density_kg_m3=900.0,
        longitudinal_speed_m_s=2732.0,
        poisson_ratio=0.42,
    )
    plate = ElasticPlate(polypropylene, PP_THICKNESS_M)

    reference_properties = dmso_water_properties(
        REFERENCE_DMSO_PERCENT / 100.0,
        basis="volume",
        temperature_c=TEMPERATURE_C,
    )
    reference_model = build_model(
        reference_properties,
        grid=grid,
        aperture=aperture,
        water=water,
        plate=plate,
    )
    reference_field = reference_model.field_after_plate(
        FREQUENCY_HZ,
        LIQUID_DEPTH_M,
    )
    reference_centre = reference_model.grid.centre_index
    unscaled_reference_pressure = abs(reference_field[reference_centre])
    common_pressure_scale = (
        args.reference_meniscus_pressure_mpa * 1.0e6
        / unscaled_reference_pressure
    )
    reference_dynamic_viscosity_pa_s = (
        reference_properties.dynamic_viscosity_pa_s
    )
    reference_surface_tension_n_m = reference_properties.surface_tension_n_m

    tone_duration_s = TONE_CYCLES / FREQUENCY_HZ
    end_time_s = tone_duration_s
    time_s = np.arange(
        0.0,
        end_time_s + POST_BURST_TIME_S + 0.5 * TIME_STEP_S,
        TIME_STEP_S,
    )
    envelope = raised_cosine_tone_envelope(
        time_s,
        start_time_s=0.0,
        duration_s=tone_duration_s,
        edge_time_s=0.5e-6,
    )
    focus_scan_m = np.linspace(0.0, 6.0e-3, 601)

    rows: list[dict[str, float | bool]] = []
    case_limitations: list[tuple[str, ...]] = []
    for concentration_percent in concentrations:
        properties = dmso_water_properties(
            concentration_percent / 100.0,
            basis="volume",
            temperature_c=TEMPERATURE_C,
        )
        model = build_model(
            properties,
            grid=grid,
            aperture=aperture,
            water=water,
            plate=plate,
        )
        field = model.field_after_plate(FREQUENCY_HZ, LIQUID_DEPTH_M)
        radial_pressure_pa, half_intensity_radius_m = radial_pressure_from_field(
            field,
            grid=grid,
            radius_m=radius_m,
            common_pressure_scale=common_pressure_scale,
        )
        axial_amplitude = np.abs(
            model.on_axis_scan_after_plate(FREQUENCY_HZ, focus_scan_m)
        )
        focus_after_plate_m, focus_boundary_limited = quadratic_peak_position(
            focus_scan_m,
            axial_amplitude,
        )
        pressure_envelope_pa = envelope[:, None] * radial_pressure_pa[None, :]
        normal_stress_pa = radiation_stress_from_incident_pressure_envelope(
            pressure_envelope_pa,
            density_kg_m3=properties.density_kg_m3,
            sound_speed_m_s=properties.sound_speed_m_s,
            reflected_intensity_fraction=1.0,
        )
        result = simulate_axisymmetric_free_surface(
            time_s,
            radius_m,
            normal_stress_pa,
            liquid=FreeSurfaceLiquid(
                density_kg_m3=properties.density_kg_m3,
                dynamic_viscosity_pa_s=properties.dynamic_viscosity_pa_s,
                surface_tension_n_m=properties.surface_tension_n_m,
            ),
            liquid_depth_m=LIQUID_DEPTH_M,
            contact_line=ContactLineModel(
                mode="fixed_contact_angle",
                equilibrium_contact_angle_deg=FIXED_CONTACT_ANGLE_DEG,
            ),
            mode_count=32,
            forcing_is_absolute=args.absolute_calibration,
            acoustic_wavelength_m=properties.sound_speed_m_s / FREQUENCY_HZ,
            focal_spot_radius_m=half_intensity_radius_m,
        )
        fixed_transport_result = simulate_axisymmetric_free_surface(
            time_s,
            radius_m,
            normal_stress_pa,
            liquid=FreeSurfaceLiquid(
                density_kg_m3=properties.density_kg_m3,
                dynamic_viscosity_pa_s=(
                    reference_dynamic_viscosity_pa_s
                ),
                surface_tension_n_m=reference_surface_tension_n_m,
            ),
            liquid_depth_m=LIQUID_DEPTH_M,
            contact_line=ContactLineModel(
                mode="fixed_contact_angle",
                equilibrium_contact_angle_deg=FIXED_CONTACT_ANGLE_DEG,
            ),
            mode_count=32,
            forcing_is_absolute=args.absolute_calibration,
            acoustic_wavelength_m=properties.sound_speed_m_s / FREQUENCY_HZ,
            focal_spot_radius_m=half_intensity_radius_m,
        )
        case_limitations.append(result.limitations)
        apex_m = result.apex_dynamic_elevation_m
        fixed_transport_apex_m = (
            fixed_transport_result.apex_dynamic_elevation_m
        )
        peak_index = int(np.argmax(apex_m))
        fixed_transport_peak_um = float(
            np.max(fixed_transport_apex_m) * 1.0e6
        )
        during_burst = time_s <= end_time_s + 0.5 * TIME_STEP_S
        end_index = int(np.argmin(np.abs(time_s - end_time_s)))
        rows.append(
            {
                "dmso_volume_percent": float(concentration_percent),
                "dmso_mole_fraction": properties.dmso_mole_fraction,
                "density_kg_m3": properties.density_kg_m3,
                "sound_speed_m_s": properties.sound_speed_m_s,
                "dynamic_viscosity_mpa_s": (
                    properties.dynamic_viscosity_pa_s * 1.0e3
                ),
                "surface_tension_mn_m": (
                    properties.surface_tension_n_m * 1.0e3
                ),
                "surface_tension_temperature_extrapolated": (
                    properties.surface_tension_temperature_extrapolated
                ),
                "half_wavelength_um": (
                    0.5 * properties.sound_speed_m_s / FREQUENCY_HZ * 1.0e6
                ),
                "liquid_cavity_round_trip_us": (
                    2.0 * LIQUID_DEPTH_M / properties.sound_speed_m_s * 1.0e6
                ),
                "focus_after_pp_mm": focus_after_plate_m * 1.0e3,
                "focus_below_meniscus_mm": (
                    LIQUID_DEPTH_M - focus_after_plate_m
                )
                * 1.0e3,
                "focus_scan_boundary_limited": focus_boundary_limited,
                "incident_peak_pressure_at_meniscus_mpa": (
                    radial_pressure_pa[0] * 1.0e-6
                ),
                "focal_intensity_fwhm_diameter_mm": (
                    2.0 * half_intensity_radius_m * 1.0e3
                ),
                "peak_apex_height_um": float(apex_m[peak_index] * 1.0e6),
                "peak_apex_height_fixed_80pct_transport_um": (
                    fixed_transport_peak_um
                ),
                "transport_property_effect_percent": float(
                    100.0
                    * (
                        apex_m[peak_index] * 1.0e6
                        / fixed_transport_peak_um
                        - 1.0
                    )
                ),
                "peak_during_burst_um": float(
                    np.max(apex_m[during_burst]) * 1.0e6
                ),
                "apex_at_burst_end_um": float(apex_m[end_index] * 1.0e6),
                "peak_time_us": float(time_s[peak_index] * 1.0e6),
                "peak_delay_after_burst_us": float(
                    (time_s[peak_index] - end_time_s) * 1.0e6
                ),
                "peak_is_window_clipped": bool(
                    peak_index >= time_s.size - 2
                ),
                "peak_positive_mound_volume_pl": float(
                    np.max(result.positive_mound_volume_m3) * 1.0e15
                ),
                "within_reduced_model_validity": bool(
                    result.within_reduced_model_validity
                ),
            }
        )

    peak_height_um = np.array(
        [float(row["peak_apex_height_um"]) for row in rows]
    )
    meniscus_pressure_mpa = np.array(
        [float(row["incident_peak_pressure_at_meniscus_mpa"]) for row in rows]
    )
    focus_below_meniscus_mm = np.array(
        [float(row["focus_below_meniscus_mm"]) for row in rows]
    )
    fixed_transport_peak_height_um = np.array(
        [
            float(row["peak_apex_height_fixed_80pct_transport_um"])
            for row in rows
        ]
    )
    dynamic_viscosity_mpa_s = np.array(
        [float(row["dynamic_viscosity_mpa_s"]) for row in rows]
    )
    surface_tension_mn_m = np.array(
        [float(row["surface_tension_mn_m"]) for row in rows]
    )
    validity = np.array(
        [bool(row["within_reduced_model_validity"]) for row in rows]
    )
    clipped = np.array([bool(row["peak_is_window_clipped"]) for row in rows])
    reference_index = int(
        np.argmin(np.abs(concentrations - REFERENCE_DMSO_PERCENT))
    )
    highest_index = int(np.argmax(peak_height_um))
    reference_height_um = peak_height_um[reference_index]
    for row, height_um in zip(rows, peak_height_um, strict=True):
        row["peak_apex_height_relative_to_nearest_80pct"] = float(
            height_um / reference_height_um
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    stem = args.output_directory / "free_surface_dmso_concentration_sweep"
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_scope": (
            "single-pass acoustic concentration sensitivity coupled to a "
            "linearized one-way pre-ejection free-surface response"
        ),
        "can_predict_detached_drop_volume": False,
        "input": {
            "frequency_hz": FREQUENCY_HZ,
            "cycles": TONE_CYCLES,
            "tone_duration_s": tone_duration_s,
            "temperature_c": TEMPERATURE_C,
            "liquid_depth_m": LIQUID_DEPTH_M,
            "water_path_m": WATER_PATH_M,
            "pp_thickness_m": PP_THICKNESS_M,
            "reference_dmso_volume_percent": REFERENCE_DMSO_PERCENT,
            "reference_incident_meniscus_pressure_pa": (
                args.reference_meniscus_pressure_mpa * 1.0e6
            ),
            "reference_pressure_has_absolute_calibration": bool(
                args.absolute_calibration
            ),
            "common_aperture_pressure_scale": common_pressure_scale,
            "finite_cavity_build_up_in_forcing": False,
            "transport_property_model": {
                "dynamic_viscosity": (
                    "Omota et al. measured 20/30/40 degC isotherms"
                ),
                "surface_tension": (
                    "Markarian and Terzyan / NIST ThermoML measured "
                    "25--55 degC isotherms"
                ),
                "surface_tension_temperature_extrapolated": bool(
                    TEMPERATURE_C < 25.0
                ),
            },
            "fixed_transport_control": {
                "reference_dmso_volume_percent": REFERENCE_DMSO_PERCENT,
                "dynamic_viscosity_pa_s": (
                    reference_dynamic_viscosity_pa_s
                ),
                "surface_tension_n_m": reference_surface_tension_n_m,
            },
            "fixed_equilibrium_contact_angle_deg": FIXED_CONTACT_ANGLE_DEG,
            "composition_dependent_hydrodynamic_properties_available": True,
        },
        "output": {
            "largest_peak_dmso_volume_percent": float(
                concentrations[highest_index]
            ),
            "largest_peak_apex_height_m": float(
                peak_height_um[highest_index] * 1.0e-6
            ),
            "nearest_80_percent_case": rows[reference_index],
            "all_cases_within_reduced_model_validity": bool(np.all(validity)),
            "any_peak_at_observation_boundary": bool(np.any(clipped)),
            "invalid_dmso_volume_percent": concentrations[~validity].tolist(),
        },
        "base_case_limitations": list(case_limitations[0]),
        "largest_response_case_limitations": list(
            case_limitations[highest_index]
        ),
        "additional_sweep_limitations": [
            "surface tension at 22 degC is a flagged 3 K extrapolation below the measured range",
            "contact angle and liquid attenuation are fixed across concentration",
            "one common aperture-pressure scale is a drive proxy, not an electroacoustic voltage calibration",
            "finite DMSO-air cavity build-up is omitted even though multiple returns fit within 150 cycles",
        ],
    }
    stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.2, 9.5),
        sharex=True,
        height_ratios=(1.7, 1.0, 1.0),
        constrained_layout=True,
    )
    axes[0].plot(
        concentrations,
        peak_height_um,
        color="#003b5c",
        linewidth=2.2,
        label="concentration-dependent viscosity + surface tension",
    )
    axes[0].plot(
        concentrations,
        fixed_transport_peak_height_um,
        color="#64748b",
        linestyle="--",
        linewidth=1.7,
        label="fixed 80% transport-property control",
    )
    if np.any(~validity):
        axes[0].scatter(
            concentrations[~validity],
            peak_height_um[~validity],
            marker="x",
            color="#b42318",
            label="reduced-model validity flag",
            zorder=4,
        )
    if np.any(clipped):
        axes[0].scatter(
            concentrations[clipped],
            peak_height_um[clipped],
            marker="s",
            facecolors="none",
            edgecolors="#b42318",
            label="maximum reached observation boundary",
            zorder=5,
        )
    axes[0].scatter(
        concentrations[reference_index],
        peak_height_um[reference_index],
        color="#f59e0b",
        edgecolor="#7c4a03",
        zorder=5,
    )
    axes[0].annotate(
        f"{concentrations[reference_index]:g}% reference\n"
        f"{peak_height_um[reference_index]:.2f} µm",
        xy=(
            concentrations[reference_index],
            peak_height_um[reference_index],
        ),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0].set(
        title="Maximum pre-ejection mound height",
        ylabel="Dynamic apex elevation [µm]",
    )
    relative_height_axis = axes[0].secondary_yaxis(
        "right",
        functions=(
            lambda height_um: 100.0 * (height_um / reference_height_um - 1.0),
            lambda relative_percent: reference_height_um
            * (1.0 + relative_percent / 100.0),
        ),
    )
    relative_height_axis.set_ylabel("Change from 80% reference [%]")
    axes[0].legend(frameon=False)

    pressure_line = axes[1].plot(
        concentrations,
        meniscus_pressure_mpa,
        color="#00a67d",
        linewidth=2.0,
        label="incident pressure at meniscus",
    )[0]
    axes[1].set(ylabel="Incident peak pressure [MPa]")
    focus_axis = axes[1].twinx()
    focus_line = focus_axis.plot(
        concentrations,
        focus_below_meniscus_mm,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.7,
        label="focus below meniscus",
    )[0]
    focus_axis.axhline(0.0, color="0.55", linewidth=0.8)
    focus_axis.set_ylabel("Focus below meniscus [mm]")
    axes[1].legend(
        [pressure_line, focus_line],
        [pressure_line.get_label(), focus_line.get_label()],
        frameon=False,
        loc="best",
    )
    viscosity_line = axes[2].plot(
        concentrations,
        dynamic_viscosity_mpa_s,
        color="#d97706",
        linewidth=2.0,
        label="dynamic viscosity",
    )[0]
    axes[2].set(
        xlabel="DMSO [vol.%]",
        ylabel="Dynamic viscosity [mPa·s]",
    )
    tension_axis = axes[2].twinx()
    tension_line = tension_axis.plot(
        concentrations,
        surface_tension_mn_m,
        color="#dc2626",
        linewidth=1.8,
        label="surface tension (3 K extrapolation)",
    )[0]
    tension_axis.set_ylabel("Surface tension [mN/m]")
    axes[2].legend(
        [viscosity_line, tension_line],
        [viscosity_line.get_label(), tension_line.get_label()],
        frameon=False,
        loc="best",
    )
    for axis in axes:
        axis.axvline(
            REFERENCE_DMSO_PERCENT,
            color="#f59e0b",
            linewidth=0.9,
            alpha=0.7,
        )
        axis.grid(alpha=0.18)
    calibration_label = (
        "absolute 80% pressure asserted"
        if args.absolute_calibration
        else "illustrative 80% pressure reference"
    )
    fig.suptitle(
        "DMSO concentration sweep · fixed 150-cycle aperture drive\n"
        f"single-pass field, {calibration_label}",
        fontsize=13,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)

    print(f"Saved {stem.with_suffix('.png')}")
    print(
        f"80% reference: {peak_height_um[reference_index]:.3f} µm peak apex height"
    )
    print(
        "Largest swept single-pass response: "
        f"{concentrations[highest_index]:g}% DMSO, "
        f"{peak_height_um[highest_index]:.3f} µm"
    )
    print(f"All reduced-model validity checks: {bool(np.all(validity))}")
    print(f"Any maximum clipped by observation window: {bool(np.any(clipped))}")


if __name__ == "__main__":
    main()
