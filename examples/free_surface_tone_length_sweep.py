"""Sweep tone length and report the maximum pre-ejection mound height.

The acoustic peak-pressure scale is held constant while the 10 MHz burst
duration is varied.  Each case is observed until a fixed time after its own
excitation end, so shorter bursts do not receive a longer search window than
longer bursts.  The main metric is the maximum dynamic apex elevation above
the static meniscus, not a detached-drop height or volume.

The default 1 MPa pressure scale is illustrative.  Viscosity and surface
tension come from the measured DMSO property model; at the default 22 degC,
surface tension is a flagged 3 K extrapolation below its measured range.
Finite liquid-cavity build-up is not included, so the sweep isolates the
linearized free-surface response to tone duration at a fixed spatial load.
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
DMSO_VOLUME_FRACTION = 0.80
TEMPERATURE_C = 22.0
LIQUID_DEPTH_M = 4.22e-3
WELL_RADIUS_M = 1.65e-3
WATER_PATH_M = 19.1e-3
PP_THICKNESS_M = 0.78e-3
CONTACT_ANGLE_DEG = 90.0
START_TIME_S = 0.0
TIME_STEP_S = 0.20e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incident-pressure-mpa",
        type=float,
        default=1.0,
        help="assumed incident peak pressure at the focal centre (default 1)",
    )
    parser.add_argument(
        "--min-cycles",
        type=float,
        default=10.0,
        help="shortest electrical tone length in cycles (default 10)",
    )
    parser.add_argument(
        "--max-cycles",
        type=float,
        default=300.0,
        help="longest electrical tone length in cycles (default 300)",
    )
    parser.add_argument(
        "--cycle-step",
        type=float,
        default=5.0,
        help="tone-length increment in cycles (default 5)",
    )
    parser.add_argument(
        "--post-burst-us",
        type=float,
        default=600.0,
        help="equal observation time after every burst (default 600 us)",
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


def tone_lengths(
    minimum_cycles: float,
    maximum_cycles: float,
    cycle_step: float,
) -> np.ndarray:
    """Return an inclusive, uniformly spaced tone-length array."""

    if minimum_cycles <= 0.0 or maximum_cycles < minimum_cycles:
        raise ValueError("require 0 < min-cycles <= max-cycles")
    if cycle_step <= 0.0:
        raise ValueError("cycle-step must be > 0")
    count = int(np.floor((maximum_cycles - minimum_cycles) / cycle_step)) + 1
    values = minimum_cycles + cycle_step * np.arange(count, dtype=float)
    if values[-1] < maximum_cycles - 1.0e-12 * maximum_cycles:
        values = np.append(values, maximum_cycles)
    return values


def focused_radial_pressure_profile(
    peak_pressure_pa: float,
    radius_m: np.ndarray,
) -> tuple[np.ndarray, float, DMSOWaterProperties]:
    """Return the default ASM pressure magnitude and half-intensity radius."""

    acoustic = dmso_water_properties(
        DMSO_VOLUME_FRACTION,
        basis="volume",
        temperature_c=TEMPERATURE_C,
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
    model = AngularSpectrumModel(
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
    incident_field = model.field_after_plate(FREQUENCY_HZ, LIQUID_DEPTH_M)
    centre_y, centre_x = model.grid.centre_index
    positive_x = model.grid.x_m >= 0.0
    positive_y = model.grid.y_m >= 0.0
    samples_x = model.grid.x_m[positive_x]
    samples_y = model.grid.y_m[positive_y]
    power_x = np.abs(incident_field[centre_y, positive_x]) ** 2
    power_y = np.abs(incident_field[positive_y, centre_x]) ** 2
    relative_power = 0.5 * (
        power_x + np.interp(samples_x, samples_y, power_y)
    )
    relative_power /= relative_power[0]
    radial_pressure_pa = peak_pressure_pa * np.sqrt(
        np.maximum(np.interp(radius_m, samples_x, relative_power), 0.0)
    )
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
    return radial_pressure_pa, half_intensity_radius_m, acoustic


def main() -> None:
    args = parse_args()
    if args.incident_pressure_mpa <= 0.0 or args.post_burst_us <= 0.0:
        raise ValueError("incident pressure and post-burst-us must be > 0")
    cycles = tone_lengths(args.min_cycles, args.max_cycles, args.cycle_step)
    radius_m = np.linspace(0.0, WELL_RADIUS_M, 129)
    peak_pressure_pa = args.incident_pressure_mpa * 1.0e6
    (
        radial_peak_pressure_pa,
        half_intensity_radius_m,
        acoustic,
    ) = focused_radial_pressure_profile(peak_pressure_pa, radius_m)
    liquid = FreeSurfaceLiquid(
        density_kg_m3=acoustic.density_kg_m3,
        dynamic_viscosity_pa_s=acoustic.dynamic_viscosity_pa_s,
        surface_tension_n_m=acoustic.surface_tension_n_m,
    )
    post_burst_s = args.post_burst_us * 1.0e-6

    rows: list[dict[str, float | bool]] = []
    case_limitations: list[tuple[str, ...]] = []
    for cycle_count in cycles:
        duration_s = float(cycle_count / FREQUENCY_HZ)
        end_time_s = START_TIME_S + duration_s
        time_s = np.arange(
            0.0,
            end_time_s + post_burst_s + 0.5 * TIME_STEP_S,
            TIME_STEP_S,
        )
        envelope = raised_cosine_tone_envelope(
            time_s,
            start_time_s=START_TIME_S,
            duration_s=duration_s,
            edge_time_s=min(0.5e-6, 0.2 * duration_s),
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
        case_limitations.append(result.limitations)
        apex_m = result.apex_dynamic_elevation_m
        peak_index = int(np.argmax(apex_m))
        during_burst = time_s <= end_time_s + 0.5 * TIME_STEP_S
        end_index = int(np.argmin(np.abs(time_s - end_time_s)))
        rows.append(
            {
                "cycles": float(cycle_count),
                "duration_us": duration_s * 1.0e6,
                "peak_apex_height_um": float(apex_m[peak_index] * 1.0e6),
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

    args.output_directory.mkdir(parents=True, exist_ok=True)
    stem = args.output_directory / "free_surface_tone_length_sweep"
    fieldnames = list(rows[0])
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    peak_height_um = np.array(
        [float(row["peak_apex_height_um"]) for row in rows]
    )
    during_height_um = np.array(
        [float(row["peak_during_burst_um"]) for row in rows]
    )
    peak_delay_us = np.array(
        [float(row["peak_delay_after_burst_us"]) for row in rows]
    )
    validity = np.array(
        [bool(row["within_reduced_model_validity"]) for row in rows]
    )
    clipped = np.array([bool(row["peak_is_window_clipped"]) for row in rows])
    best_index = int(np.argmax(peak_height_um))
    reference_index = int(np.argmin(np.abs(cycles - 150.0)))
    summary = {
        "model_scope": "linearized one-way pre-ejection free-surface response",
        "can_predict_detached_drop_volume": False,
        "metric": (
            "maximum dynamic apex elevation above the static meniscus from "
            "excitation start through a fixed post-burst observation window"
        ),
        "input": {
            "frequency_hz": FREQUENCY_HZ,
            "incident_peak_pressure_pa": peak_pressure_pa,
            "incident_pressure_has_absolute_calibration": bool(
                args.absolute_calibration
            ),
            "minimum_cycles": float(cycles[0]),
            "maximum_cycles": float(cycles[-1]),
            "cycle_step": args.cycle_step,
            "post_burst_observation_s": post_burst_s,
            "dmso_volume_fraction": DMSO_VOLUME_FRACTION,
            "temperature_c": TEMPERATURE_C,
            "liquid_depth_m": LIQUID_DEPTH_M,
            "well_radius_m": WELL_RADIUS_M,
            "water_path_m": WATER_PATH_M,
            "pp_thickness_m": PP_THICKNESS_M,
            "dynamic_viscosity_pa_s": acoustic.dynamic_viscosity_pa_s,
            "surface_tension_n_m": acoustic.surface_tension_n_m,
            "surface_tension_temperature_extrapolated": (
                acoustic.surface_tension_temperature_extrapolated
            ),
            "equilibrium_contact_angle_deg": CONTACT_ANGLE_DEG,
            "finite_cavity_build_up_in_forcing": False,
        },
        "output": {
            "largest_peak_cycles": float(cycles[best_index]),
            "largest_peak_apex_height_m": float(
                peak_height_um[best_index] * 1.0e-6
            ),
            "nearest_150_cycle_case": rows[reference_index],
            "all_cases_within_reduced_model_validity": bool(np.all(validity)),
            "any_peak_at_observation_boundary": bool(np.any(clipped)),
            "invalid_cycle_counts": cycles[~validity].tolist(),
        },
        "base_case_limitations": list(case_limitations[0]),
        "largest_response_case_limitations": list(
            case_limitations[best_index]
        ),
    }
    stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.0),
        sharex=True,
        height_ratios=(2.2, 1.0),
        constrained_layout=True,
    )
    axes[0].plot(
        cycles,
        peak_height_um,
        color="#003b5c",
        linewidth=2.2,
        label=f"maximum through +{args.post_burst_us:g} µs",
    )
    axes[0].plot(
        cycles,
        during_height_um,
        color="#00a67d",
        linewidth=1.7,
        linestyle="--",
        label="maximum while tone is active",
    )
    if np.any(~validity):
        axes[0].scatter(
            cycles[~validity],
            peak_height_um[~validity],
            marker="x",
            color="#b42318",
            label="reduced-model validity flag",
            zorder=4,
        )
    if np.any(clipped):
        axes[0].scatter(
            cycles[clipped],
            peak_height_um[clipped],
            marker="s",
            facecolors="none",
            edgecolors="#b42318",
            label="maximum reached observation boundary",
            zorder=5,
        )
    axes[0].scatter(
        cycles[reference_index],
        peak_height_um[reference_index],
        color="#f59e0b",
        edgecolor="#7c4a03",
        zorder=5,
    )
    axes[0].annotate(
        f"{cycles[reference_index]:g} cycles\n"
        f"{peak_height_um[reference_index]:.2f} µm",
        xy=(cycles[reference_index], peak_height_um[reference_index]),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0].set(
        title="Maximum pre-ejection mound height",
        ylabel="Dynamic apex elevation [µm]",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(cycles, peak_delay_us, color="#7c3aed", linewidth=1.8)
    axes[1].axhline(0.0, color="0.55", linewidth=0.8)
    axes[1].set(
        title="When the maximum occurs",
        xlabel="Tone length [cycles at 10 MHz]",
        ylabel="Delay after tone end [µs]",
    )
    duration_axis = axes[0].secondary_xaxis(
        "top",
        functions=(
            lambda cycle_count: cycle_count / (FREQUENCY_HZ * 1.0e-6),
            lambda duration_us: duration_us * FREQUENCY_HZ * 1.0e-6,
        ),
    )
    duration_axis.set_xlabel("Tone duration [µs]")
    for axis in axes:
        axis.axvline(150.0, color="#f59e0b", linewidth=0.9, alpha=0.7)
        axis.grid(alpha=0.18)
    calibration_label = (
        "absolute pressure asserted"
        if args.absolute_calibration
        else "illustrative pressure scale"
    )
    fig.suptitle(
        "Tone-length sweep · fixed focal pressure and spatial field\n"
        f"{args.incident_pressure_mpa:g} MPa peak, {calibration_label}",
        fontsize=13,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)

    print(f"Saved {stem.with_suffix('.png')}")
    print(
        f"Nearest 150-cycle case: {cycles[reference_index]:g} cycles, "
        f"{peak_height_um[reference_index]:.3f} µm peak apex height"
    )
    print(
        f"Largest swept response: {cycles[best_index]:g} cycles, "
        f"{peak_height_um[best_index]:.3f} µm"
    )
    print(f"All reduced-model validity checks: {bool(np.all(validity))}")
    print(f"Any maximum clipped by observation window: {bool(np.any(clipped))}")


if __name__ == "__main__":
    main()
