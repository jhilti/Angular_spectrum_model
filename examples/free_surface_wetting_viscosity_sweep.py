"""Compare static wetting, viscosity, and surface-tension assumptions.

The baseline viscosity and surface tension come from the DMSO property model;
the surrounding sensitivity values and 0.8 MPa incident pressure remain
illustrative.  The example is intended to show which parameters need
laboratory calibration; it does not predict detached droplet volume.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from angular_spectrum import (
    ContactLineModel,
    FreeSurfaceLiquid,
    dmso_water_properties,
    equilibrium_meniscus_profile,
    radiation_stress_from_incident_pressure_envelope,
    raised_cosine_tone_envelope,
    simulate_axisymmetric_free_surface,
)


FREQUENCY_HZ = 10.0e6
CYCLES = 150.0
INCIDENT_PEAK_PRESSURE_PA = 0.8e6
LIQUID_DEPTH_M = 4.22e-3
WELL_RADIUS_M = 1.65e-3
FOCAL_INTENSITY_FWHM_DIAMETER_M = 0.303e-3
DMSO_VOLUME_FRACTION = 0.80
TEMPERATURE_C = 22.0
OUTPUT_DIRECTORY = Path("results")


def main() -> None:
    acoustic = dmso_water_properties(
        DMSO_VOLUME_FRACTION,
        basis="volume",
        temperature_c=TEMPERATURE_C,
    )
    base_viscosity_pa_s = acoustic.dynamic_viscosity_pa_s
    base_surface_tension_n_m = acoustic.surface_tension_n_m
    time_step_s = 0.20e-6
    time_s = np.arange(0.0, 350.0e-6 + 0.5 * time_step_s, time_step_s)
    radius_m = np.linspace(0.0, WELL_RADIUS_M, 97)
    tone_duration_s = CYCLES / FREQUENCY_HZ
    envelope = raised_cosine_tone_envelope(
        time_s,
        start_time_s=5.0e-6,
        duration_s=tone_duration_s,
        edge_time_s=0.5e-6,
    )
    radial_pressure = INCIDENT_PEAK_PRESSURE_PA * np.exp(
        -2.0
        * np.log(2.0)
        * (radius_m / FOCAL_INTENSITY_FWHM_DIAMETER_M) ** 2
    )
    stress_pa = radiation_stress_from_incident_pressure_envelope(
        envelope[:, None] * radial_pressure[None, :],
        density_kg_m3=acoustic.density_kg_m3,
        sound_speed_m_s=acoustic.sound_speed_m_s,
    )

    def run_case(
        *,
        viscosity_pa_s: float | None = None,
        surface_tension_n_m: float | None = None,
        contact_angle_deg: float = 90.0,
    ):
        if viscosity_pa_s is None:
            viscosity_pa_s = base_viscosity_pa_s
        if surface_tension_n_m is None:
            surface_tension_n_m = base_surface_tension_n_m
        return simulate_axisymmetric_free_surface(
            time_s,
            radius_m,
            stress_pa,
            liquid=FreeSurfaceLiquid(
                density_kg_m3=acoustic.density_kg_m3,
                dynamic_viscosity_pa_s=viscosity_pa_s,
                surface_tension_n_m=surface_tension_n_m,
            ),
            liquid_depth_m=LIQUID_DEPTH_M,
            contact_line=ContactLineModel(
                mode="fixed_contact_angle",
                equilibrium_contact_angle_deg=contact_angle_deg,
            ),
            mode_count=24,
            forcing_is_absolute=False,
            acoustic_wavelength_m=acoustic.sound_speed_m_s / FREQUENCY_HZ,
            focal_spot_radius_m=0.5 * FOCAL_INTENSITY_FWHM_DIAMETER_M,
        )

    viscosity_values = (1.0e-3, base_viscosity_pa_s, 5.0e-3)
    viscosity_results = {
        value: run_case(viscosity_pa_s=value) for value in viscosity_values
    }
    tension_values = (0.035, base_surface_tension_n_m, 0.055)
    tension_results = {
        value: run_case(surface_tension_n_m=value) for value in tension_values
    }
    dynamic_wetting_angles = (75.0, 90.0, 105.0)
    wetting_results = {
        angle: run_case(contact_angle_deg=angle)
        for angle in dynamic_wetting_angles
    }
    wetting_angles = (60.0, 90.0, 120.0)
    equilibrium_profiles = {
        angle: equilibrium_meniscus_profile(
            radius_m,
            liquid=FreeSurfaceLiquid(
                acoustic.density_kg_m3,
                base_viscosity_pa_s,
                base_surface_tension_n_m,
            ),
            equilibrium_contact_angle_deg=angle,
        )
        for angle in wetting_angles
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIRECTORY / "free_surface_wetting_viscosity_sweep"
    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), constrained_layout=True)
    for viscosity, result in viscosity_results.items():
        axes[0, 0].plot(
            time_us,
            result.apex_dynamic_elevation_m * 1.0e6,
            label=f"{viscosity * 1e3:g} mPa·s",
        )
    axes[0, 0].set(
        title="Viscosity controls recovery and damping",
        xlabel="Time [µs]",
        ylabel="Apex displacement [µm]",
    )
    axes[0, 0].legend(frameon=False)

    for tension, result in tension_results.items():
        axes[0, 1].plot(
            time_us,
            result.apex_dynamic_elevation_m * 1.0e6,
            label=f"{tension * 1e3:g} mN/m",
        )
    axes[0, 1].set(
        title="Surface tension changes capillary stiffness",
        xlabel="Time [µs]",
        ylabel="Apex displacement [µm]",
    )
    axes[0, 1].legend(frameon=False)

    for angle, profile in equilibrium_profiles.items():
        axes[1, 0].plot(
            radius_m * 1.0e3,
            profile * 1.0e6,
            label=f"θ = {angle:g}°",
        )
    axes[1, 0].set(
        title="Nonlinear Young–Laplace wetting equilibrium",
        xlabel="Radius [mm]",
        ylabel="Static elevation from mean plane [µm]",
    )
    axes[1, 0].legend(frameon=False)

    for angle, result in wetting_results.items():
        axes[1, 1].plot(
            time_us,
            result.apex_surface_elevation_m * 1.0e6,
            label=f"θ = {angle:g}°",
        )
    axes[1, 1].set(
        title="Static wetting offset + flat-mode perturbation",
        xlabel="Time [µs]",
        ylabel="Total apex elevation [µm]",
    )
    axes[1, 1].legend(frameon=False)
    axes[1, 1].text(
        0.02,
        0.04,
        "Curved cases trigger the frozen-acoustic-field validity flag",
        transform=axes[1, 1].transAxes,
        fontsize=8.5,
        color="#b42318",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.axhline(0.0, color="0.7", linewidth=0.7)
    fig.suptitle(
        "Optional free-surface sensitivity study\n"
        "DMSO-property baseline, illustrative brackets; no pinch-off model",
        fontsize=13,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)

    rows: list[dict[str, float | str | bool]] = []
    for family, cases in (
        ("viscosity", viscosity_results),
        ("surface_tension", tension_results),
        ("wetting_angle", wetting_results),
    ):
        for value, result in cases.items():
            rows.append(
                {
                    "family": family,
                    "case": str(value),
                    "peak_apex_displacement_um": (
                        result.peak_positive_apex_displacement_m * 1.0e6
                    ),
                    "peak_positive_mound_volume_pl": float(
                        np.max(result.positive_mound_volume_m3) * 1.0e15
                    ),
                    "maximum_dynamic_abs_slope": (
                        result.maximum_dynamic_abs_slope
                    ),
                    "maximum_retained_mode_damping_ratio": float(
                        np.max(result.damping_ratio)
                    ),
                    "within_reduced_model_validity": (
                        result.within_reduced_model_validity
                    ),
                    "frozen_acoustic_feedback_likely": (
                        result.frozen_acoustic_feedback_likely
                    ),
                }
            )
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_scope": "linearized one-way pre-ejection free-surface sensitivity",
        "can_predict_detached_drop_volume": False,
        "forcing_has_absolute_calibration": False,
        "input": {
            "frequency_hz": FREQUENCY_HZ,
            "cycles": CYCLES,
            "incident_peak_pressure_pa": INCIDENT_PEAK_PRESSURE_PA,
            "liquid_depth_m": LIQUID_DEPTH_M,
            "well_radius_m": WELL_RADIUS_M,
            "baseline_dynamic_viscosity_pa_s": base_viscosity_pa_s,
            "baseline_surface_tension_n_m": base_surface_tension_n_m,
            "baseline_surface_tension_temperature_extrapolated": (
                acoustic.surface_tension_temperature_extrapolated
            ),
            "viscosity_values_pa_s": list(viscosity_values),
            "surface_tension_values_n_m": list(tension_values),
            "wetting_angles_deg": list(wetting_angles),
            "dynamic_wetting_angles_deg": list(dynamic_wetting_angles),
            "contact_line_mode": "fixed_contact_angle",
            "sensitivity_bracket_values_are_measured": False,
        },
        "cases": rows,
        "shared_limitations": list(
            viscosity_results[base_viscosity_pa_s].limitations
        ),
    }
    stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {stem.with_suffix('.png')}")
    print("Mound volumes in the CSV are not detached-drop volumes.")


if __name__ == "__main__":
    main()
