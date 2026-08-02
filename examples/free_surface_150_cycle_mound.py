"""Transient mound response to a focused 150-cycle, 10 MHz tone burst.

The example converts an explicitly supplied *incident peak-pressure envelope*
into cycle-averaged radiation stress and drives the optional axisymmetric
free-surface solver.  The default 1 MPa scale is illustrative, not derived
from 150 V or from the qualitative survey ADC data.  Use
``--absolute-calibration`` only after replacing it with a traceable incident
pressure calibration at the meniscus plane.

This solver predicts a pre-ejection mound/capillary-wave response.  It does
not model pinch-off, and the reported positive mound volume is not a drop
volume.
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
DMSO_VOLUME_FRACTION = 0.80
TEMPERATURE_C = 22.0
DEFAULT_CYCLES = 150.0
LIQUID_DEPTH_M = 4.22e-3
WELL_RADIUS_M = 1.65e-3
WATER_PATH_M = 19.1e-3
PP_THICKNESS_M = 0.78e-3
CONTACT_ANGLE_DEG = 90.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incident-pressure-mpa",
        type=float,
        default=1.0,
        help="assumed incident peak pressure at the focal centre (default 1)",
    )
    parser.add_argument(
        "--cycles",
        type=float,
        default=DEFAULT_CYCLES,
        help="electrical tone length in cycles (default 150)",
    )
    parser.add_argument(
        "--absolute-calibration",
        action="store_true",
        help="assert that the supplied incident pressure has absolute calibration",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.incident_pressure_mpa <= 0.0 or args.cycles <= 0.0:
        raise ValueError("incident pressure and cycles must be > 0")

    acoustic = dmso_water_properties(
        DMSO_VOLUME_FRACTION,
        basis="volume",
        temperature_c=TEMPERATURE_C,
    )
    liquid = FreeSurfaceLiquid(
        density_kg_m3=acoustic.density_kg_m3,
        dynamic_viscosity_pa_s=acoustic.dynamic_viscosity_pa_s,
        surface_tension_n_m=acoustic.surface_tension_n_m,
    )
    tone_duration_s = args.cycles / FREQUENCY_HZ
    time_step_s = 0.10e-6
    time_s = np.arange(0.0, 250.0e-6 + 0.5 * time_step_s, time_step_s)
    radius_m = np.linspace(0.0, WELL_RADIUS_M, 129)
    start_time_s = 5.0e-6
    envelope = raised_cosine_tone_envelope(
        time_s,
        start_time_s=start_time_s,
        duration_s=tone_duration_s,
        edge_time_s=min(0.5e-6, 0.2 * tone_duration_s),
    )

    water_data = water_properties(TEMPERATURE_C)
    water = Fluid(
        f"water_{TEMPERATURE_C:g}C",
        water_data.density_kg_m3,
        water_data.sound_speed_m_s,
    )
    dmso = Fluid(
        f"{DMSO_VOLUME_FRACTION * 100:g}volpct_DMSO",
        acoustic.density_kg_m3,
        acoustic.sound_speed_m_s,
    )
    polypropylene = ElasticSolid.from_longitudinal_speed_and_poisson(
        name="polypropylene",
        density_kg_m3=900.0,
        longitudinal_speed_m_s=2732.0,
        poisson_ratio=0.42,
    )
    acoustic_model = AngularSpectrumModel(
        grid=CartesianGrid(nx=384, ny=384, dx_m=50.0e-6),
        aperture=FocusedCircularAperture(
            diameter_m=13.0e-3,
            focal_length_m=25.4e-3,
        ),
        incident_fluid=water,
        plate=ElasticPlate(polypropylene, PP_THICKNESS_M),
        transmitted_fluid=dmso,
        water_path_m=WATER_PATH_M,
        plate_radial_samples=4096,
    )
    incident_field = acoustic_model.field_after_plate(
        FREQUENCY_HZ, LIQUID_DEPTH_M
    )
    centre_y, centre_x = acoustic_model.grid.centre_index
    positive_x = acoustic_model.grid.x_m >= 0.0
    positive_y = acoustic_model.grid.y_m >= 0.0
    radial_samples_x = acoustic_model.grid.x_m[positive_x]
    radial_samples_y = acoustic_model.grid.y_m[positive_y]
    # The configured geometry is axisymmetric. Averaging |p|² along the two
    # centre axes reduces Cartesian-grid anisotropy without averaging complex
    # pressure phases before squaring.
    power_x = np.abs(incident_field[centre_y, positive_x]) ** 2
    power_y = np.abs(incident_field[positive_y, centre_x]) ** 2
    power_y_on_x = np.interp(radial_samples_x, radial_samples_y, power_y)
    relative_power = 0.5 * (power_x + power_y_on_x)
    relative_power /= relative_power[0]
    radial_relative_pressure = np.sqrt(
        np.interp(radius_m, radial_samples_x, relative_power)
    )
    peak_pressure_pa = args.incident_pressure_mpa * 1.0e6
    radial_peak_pressure_pa = peak_pressure_pa * radial_relative_pressure
    half_intensity_crossings = np.flatnonzero(relative_power <= 0.5)
    if half_intensity_crossings.size == 0:
        raise RuntimeError("the ASM field does not contain its half-intensity radius")
    crossing = int(half_intensity_crossings[0])
    half_intensity_radius_m = float(
        np.interp(
            0.5,
            relative_power[crossing - 1 : crossing + 1][::-1],
            radial_samples_x[crossing - 1 : crossing + 1][::-1],
        )
    )
    incident_pressure_envelope_pa = (
        envelope[:, None] * radial_peak_pressure_pa[None, :]
    )
    normal_stress_pa = radiation_stress_from_incident_pressure_envelope(
        incident_pressure_envelope_pa,
        density_kg_m3=acoustic.density_kg_m3,
        sound_speed_m_s=acoustic.sound_speed_m_s,
        reflected_intensity_fraction=1.0,
    )
    result = simulate_axisymmetric_free_surface(
        time_s,
        radius_m,
        normal_stress_pa,
        liquid=liquid,
        liquid_depth_m=LIQUID_DEPTH_M,
        contact_line=ContactLineModel(
            mode="fixed_contact_angle",
            equilibrium_contact_angle_deg=CONTACT_ANGLE_DEG,
        ),
        mode_count=32,
        forcing_is_absolute=args.absolute_calibration,
        acoustic_wavelength_m=acoustic.sound_speed_m_s / FREQUENCY_HZ,
        focal_spot_radius_m=half_intensity_radius_m,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    stem = args.output_directory / "free_surface_150_cycle_mound"
    np.savez_compressed(
        stem.with_suffix(".npz"),
        time_s=result.time_s,
        radius_m=result.radius_m,
        # Float32 keeps the example artifact compact; the solver itself and
        # CSV diagnostics remain float64.
        applied_normal_stress_pa=result.applied_normal_stress_pa.astype(
            np.float32
        ),
        equilibrium_elevation_m=result.equilibrium_elevation_m,
        dynamic_elevation_m=result.dynamic_elevation_m.astype(np.float32),
        surface_elevation_m=result.surface_elevation_m.astype(np.float32),
        vertical_velocity_m_s=result.vertical_velocity_m_s.astype(np.float32),
        curvature_1_m=result.curvature_1_m.astype(np.float32),
        positive_mound_volume_m3=result.positive_mound_volume_m3,
        volume_residual_m3=result.volume_residual_m3,
        mechanical_energy_j=result.mechanical_energy_j,
        cumulative_acoustic_work_j=result.cumulative_acoustic_work_j,
        radial_wavenumber_m_inv=result.radial_wavenumber_m_inv,
        natural_frequency_hz=result.natural_frequency_hz,
        damping_ratio=result.damping_ratio,
        active_mode_mask=result.active_mode_mask,
    )
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_us",
                "centre_radiation_stress_pa",
                "apex_dynamic_elevation_um",
                "apex_velocity_m_s",
                "positive_mound_volume_pl",
                "volume_residual_fl",
                "mechanical_energy_pj",
            ]
        )
        for index in range(time_s.size):
            writer.writerow(
                [
                    time_s[index] * 1.0e6,
                    normal_stress_pa[index, 0],
                    result.apex_dynamic_elevation_m[index] * 1.0e6,
                    result.apex_vertical_velocity_m_s[index],
                    result.positive_mound_volume_m3[index] * 1.0e15,
                    result.volume_residual_m3[index] * 1.0e18,
                    result.mechanical_energy_j[index] * 1.0e12,
                ]
            )

    summary = {
        "model_scope": "linearized one-way pre-ejection free-surface response",
        "can_predict_detached_drop_volume": False,
        "input": {
            "frequency_hz": FREQUENCY_HZ,
            "cycles": args.cycles,
            "tone_duration_s": tone_duration_s,
            "incident_peak_pressure_pa": peak_pressure_pa,
            "incident_pressure_has_absolute_calibration": bool(
                args.absolute_calibration
            ),
            "dmso_volume_fraction": DMSO_VOLUME_FRACTION,
            "temperature_c": TEMPERATURE_C,
            "liquid_depth_m": LIQUID_DEPTH_M,
            "well_radius_m": WELL_RADIUS_M,
            "water_path_m": WATER_PATH_M,
            "pp_thickness_m": PP_THICKNESS_M,
            "focal_intensity_fwhm_diameter_m": 2.0
            * half_intensity_radius_m,
            "spatial_pressure_profile": (
                "single-pass angular-spectrum field, annularly symmetric "
                "centre-axis |p|^2 average, scaled to supplied peak pressure"
            ),
            "finite_cavity_build_up_in_forcing": False,
            "density_kg_m3": acoustic.density_kg_m3,
            "sound_speed_m_s": acoustic.sound_speed_m_s,
            "dynamic_viscosity_pa_s": acoustic.dynamic_viscosity_pa_s,
            "surface_tension_n_m": acoustic.surface_tension_n_m,
            "surface_tension_temperature_extrapolated": (
                acoustic.surface_tension_temperature_extrapolated
            ),
            "equilibrium_contact_angle_deg": CONTACT_ANGLE_DEG,
            "hydrodynamic_property_basis": (
                "measured concentration isotherms with interpolation; "
                "surface tension at 22 degC is a flagged 3 K extrapolation"
            ),
        },
        "output": {
            "peak_positive_apex_displacement_m": (
                result.peak_positive_apex_displacement_m
            ),
            "peak_positive_mound_volume_m3": float(
                np.max(result.positive_mound_volume_m3)
            ),
            "maximum_dynamic_abs_slope": result.maximum_dynamic_abs_slope,
            "maximum_equilibrium_abs_slope": (
                result.maximum_equilibrium_abs_slope
            ),
            "minimum_local_liquid_depth_m": (
                result.minimum_local_liquid_depth_m
            ),
            "active_mode_count": int(np.count_nonzero(result.active_mode_mask)),
            "frozen_acoustic_feedback_likely": (
                result.frozen_acoustic_feedback_likely
            ),
            "maximum_volume_residual_m3": float(
                np.max(np.abs(result.volume_residual_m3))
            ),
            "within_reduced_model_validity": (
                result.within_reduced_model_validity
            ),
        },
        "limitations": list(result.limitations),
    }
    stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    peak_index = int(np.argmax(result.apex_dynamic_elevation_m))
    profile_indices = sorted(
        {
            0,
            int(np.searchsorted(time_s, start_time_s + tone_duration_s)),
            peak_index,
            time_s.size - 1,
        }
    )
    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    axes[0, 0].plot(time_us, normal_stress_pa[:, 0], color="#009b77")
    axes[0, 0].set(
        title="Cycle-averaged focal radiation stress",
        xlabel="Time [µs]",
        ylabel="Upward stress [Pa]",
    )
    axes[0, 0].axvspan(
        start_time_s * 1.0e6,
        (start_time_s + tone_duration_s) * 1.0e6,
        color="#00b388",
        alpha=0.12,
        label=f"{args.cycles:g} cycles",
    )
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        time_us,
        result.apex_dynamic_elevation_m * 1.0e6,
        color="#003b5c",
        label="apex elevation",
    )
    axes[0, 1].axhline(0.0, color="0.65", linewidth=0.8)
    axes[0, 1].set(
        title="Transient mound at the acoustic axis",
        xlabel="Time [µs]",
        ylabel="Dynamic elevation [µm]",
    )
    axes[0, 1].legend(frameon=False)

    for index in profile_indices:
        axes[1, 0].plot(
            radius_m * 1.0e3,
            result.dynamic_elevation_m[index] * 1.0e6,
            label=f"{time_us[index]:.1f} µs",
        )
    axes[1, 0].set(
        title="Axisymmetric dynamic surface profiles",
        xlabel="Radius [mm]",
        ylabel="Elevation relative to equilibrium [µm]",
    )
    axes[1, 0].legend(frameon=False, ncol=2)

    axes[1, 1].plot(
        time_us,
        result.positive_mound_volume_m3 * 1.0e15,
        color="#7c3aed",
        label="positive mound volume",
    )
    axes[1, 1].set(
        title="Displaced liquid above equilibrium",
        xlabel="Time [µs]",
        ylabel="Mound volume [pL]",
    )
    axes[1, 1].text(
        0.98,
        0.95,
        "Not detached-drop volume",
        ha="right",
        va="top",
        transform=axes[1, 1].transAxes,
        color="#b42318",
        fontsize=9,
    )
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    calibration_label = (
        "absolute pressure asserted"
        if args.absolute_calibration
        else "illustrative pressure scale"
    )
    fig.suptitle(
        "Optional free-surface model · 10 MHz focused burst\n"
        f"{args.incident_pressure_mpa:g} MPa peak, {calibration_label}",
        fontsize=13,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)

    print(f"Saved {stem.with_suffix('.png')}")
    print(
        "Peak apex displacement: "
        f"{result.peak_positive_apex_displacement_m * 1.0e6:.3f} µm"
    )
    print(
        "Peak positive mound volume (not drop volume): "
        f"{np.max(result.positive_mound_volume_m3) * 1.0e15:.3f} pL"
    )
    print(
        "Reduced-model validity checks: "
        f"{result.within_reduced_model_validity}"
    )


if __name__ == "__main__":
    main()
