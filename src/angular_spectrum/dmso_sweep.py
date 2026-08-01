"""Concentration sweep for the 4.22 mm DMSO/water layer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dmso_mixture import dmso_water_properties, water_properties
from .grid import CartesianGrid
from .materials import ElasticPlate, ElasticSolid, Fluid
from .model import (
    AngularSpectrumModel,
    FocusedCircularAperture,
    validate_focused_grid_support,
)


def _quadratic_peak_position(
    x: np.ndarray,
    amplitude: np.ndarray,
) -> tuple[float, bool]:
    index = int(np.argmax(amplitude))
    if index == 0 or index == x.size - 1:
        return float(x[index]), True
    denominator = (
        amplitude[index - 1]
        - 2.0 * amplitude[index]
        + amplitude[index + 1]
    )
    if denominator == 0.0:
        return float(x[index]), False
    offset = 0.5 * (
        amplitude[index - 1] - amplitude[index + 1]
    ) / denominator
    return float(x[index] + offset * (x[1] - x[0])), False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Focus sweep for a DMSO/water concentration range."
    )
    parser.add_argument("--basis", choices=("volume", "mass", "mole"), default="volume")
    parser.add_argument("--min-percent", type=float, default=70.0)
    parser.add_argument("--max-percent", type=float, default=100.0)
    parser.add_argument("--step-percent", type=float, default=1.0)
    parser.add_argument("--dmso-height-mm", type=float, default=4.22)
    parser.add_argument("--water-path-mm", type=float, default=20.0)
    parser.add_argument("--plate-thickness-mm", type=float, default=0.78)
    parser.add_argument("--frequency-mhz", type=float, default=10.0)
    parser.add_argument("--diameter-mm", type=float, default=13.0)
    parser.add_argument("--focal-length-mm", type=float, default=25.4)
    parser.add_argument("--temperature-c", type=float, default=22.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--dx-um", type=float, default=40.0)
    parser.add_argument("--radial-samples", type=int, default=4096)
    return parser


def run(args: argparse.Namespace) -> list[dict[str, float | str | bool]]:
    if args.step_percent <= 0.0 or args.max_percent < args.min_percent:
        raise ValueError("invalid concentration range")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_hz = args.frequency_mhz * 1e6
    dmso_height_m = args.dmso_height_mm * 1e-3
    concentrations = np.arange(
        args.min_percent,
        args.max_percent + 0.5 * args.step_percent,
        args.step_percent,
    )
    z_after_plate = np.linspace(
        0.0, dmso_height_m, int(round(args.dmso_height_mm * 100.0)) + 1
    )

    water_data = water_properties(args.temperature_c)
    water = Fluid(
        f"water_{args.temperature_c:g}C",
        water_data.density_kg_m3,
        water_data.sound_speed_m_s,
    )
    polypropylene = ElasticSolid.from_longitudinal_speed_and_poisson(
        name="polypropylene",
        density_kg_m3=900.0,
        longitudinal_speed_m_s=2732.0,
        poisson_ratio=0.42,
    )
    grid = CartesianGrid(args.nx, args.nx, args.dx_um * 1e-6)
    aperture = FocusedCircularAperture(
        args.diameter_mm * 1e-3,
        args.focal_length_mm * 1e-3,
    )
    plate = ElasticPlate(
        polypropylene, args.plate_thickness_mm * 1e-3
    )

    rows: list[dict[str, float | str | bool]] = []
    for concentration_percent in concentrations:
        properties = dmso_water_properties(
            concentration_percent / 100.0,
            basis=args.basis,
            temperature_c=args.temperature_c,
        )
        mixture = Fluid(
            f"{concentration_percent:g}% DMSO",
            properties.density_kg_m3,
            properties.sound_speed_m_s,
        )
        model = AngularSpectrumModel(
            grid=grid,
            aperture=aperture,
            incident_fluid=water,
            plate=plate,
            transmitted_fluid=mixture,
            water_path_m=args.water_path_mm * 1e-3,
            plate_radial_samples=args.radial_samples,
        )
        validate_focused_grid_support(
            model,
            maximum_frequency_hz=frequency_hz,
            propagation_segments=(
                (
                    "one-way water path",
                    water,
                    args.water_path_mm * 1e-3,
                ),
                (
                    "one-way mixture-height scan",
                    mixture,
                    dmso_height_m,
                ),
            ),
        )
        axial = model.on_axis_scan_after_plate(
            frequency_hz, z_after_plate
        )
        amplitude = np.abs(axial)
        focus_in_mixture_m, focus_boundary_limited = _quadratic_peak_position(
            z_after_plate, amplitude
        )
        peak_amplitude = float(np.max(amplitude))
        rows.append(
            {
                "dmso_percent": float(concentration_percent),
                "basis": args.basis,
                "temperature_c": float(args.temperature_c),
                "transducer_diameter_mm": float(args.diameter_mm),
                "transducer_focal_length_mm": float(args.focal_length_mm),
                "dmso_mole_fraction": properties.dmso_mole_fraction,
                "density_kg_m3": properties.density_kg_m3,
                "sound_speed_m_s": properties.sound_speed_m_s,
                "focus_from_plate_exit_mm": focus_in_mixture_m * 1e3,
                "focus_from_aperture_mm": (
                    args.water_path_mm
                    + args.plate_thickness_mm
                    + focus_in_mixture_m * 1e3
                ),
                "distance_focus_to_mixture_top_mm": (
                    args.dmso_height_mm - focus_in_mixture_m * 1e3
                ),
                "focus_scan_boundary_limited": focus_boundary_limited,
                "focus_inside_mixture": (
                    not focus_boundary_limited
                    and 0.0 < focus_in_mixture_m < dmso_height_m
                ),
                "peak_for_1Pa_aperture_pa": peak_amplitude,
            }
        )

    reference_peak = float(rows[-1]["peak_for_1Pa_aperture_pa"])
    reference_concentration = float(rows[-1]["dmso_percent"])
    for row in rows:
        row["reference_concentration_percent"] = reference_concentration
        row["peak_relative_to_max_sweep_concentration"] = (
            float(row["peak_for_1Pa_aperture_pa"]) / reference_peak
        )

    csv_path = output_dir / "dmso_concentration_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "dmso_concentration_sweep.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    concentration = np.array([float(row["dmso_percent"]) for row in rows])
    focus = np.array(
        [float(row["focus_from_plate_exit_mm"]) for row in rows]
    )
    distance_to_top = np.array(
        [float(row["distance_focus_to_mixture_top_mm"]) for row in rows]
    )
    speed = np.array([float(row["sound_speed_m_s"]) for row in rows])

    fig, axes = plt.subplots(
        2, 1, figsize=(7.3, 6.6), sharex=True, constrained_layout=True
    )
    axes[0].plot(concentration, focus, marker="o", markersize=3)
    axes[0].axhline(
        args.dmso_height_mm,
        color="0.45",
        linestyle="--",
        label=f"top of {args.dmso_height_mm:.2f} mm layer",
    )
    axes[0].set(
        ylabel="Focus after PP exit [mm]",
        title=f"Focus position for {args.dmso_height_mm:.2f} mm DMSO/water",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(concentration, distance_to_top, marker="o", markersize=3)
    axes[1].set(
        xlabel=f"DMSO [{args.basis}%]",
        ylabel="Focus below liquid surface [mm]",
    )
    axes[1].grid(alpha=0.25)
    speed_axis = axes[1].twinx()
    speed_axis.plot(concentration, speed, color="tab:orange", alpha=0.65)
    speed_axis.set_ylabel("Sound speed [m/s]", color="tab:orange")
    fig.savefig(output_dir / "dmso_concentration_focus.png", dpi=180)
    plt.close(fig)
    return rows


def main() -> None:
    args = _parser().parse_args()
    rows = run(args)
    print(
        "DMSO [%]  c [m/s]  focus after PP [mm]  "
        "below surface [mm]  boundary-limited"
    )
    for row in rows:
        if float(row["dmso_percent"]) % 5.0 == 0.0:
            print(
                f"{float(row['dmso_percent']):7.1f}"
                f"  {float(row['sound_speed_m_s']):7.1f}"
                f"  {float(row['focus_from_plate_exit_mm']):17.3f}"
                f"  {float(row['distance_focus_to_mixture_top_mm']):20.3f}"
                f"  {str(bool(row['focus_scan_boundary_limited'])):>16s}"
            )


if __name__ == "__main__":
    main()
