"""Command-line runner for the 10 MHz water–PP–DMSO case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import amplitude_db, fwhm
from .grid import CartesianGrid
from .materials import ElasticPlate, ElasticSolid, Fluid
from .model import AngularSpectrumModel, FocusedCircularAperture
from .plate import (
    elastic_plate_scattering,
    normal_power_transmission,
)
from .pulse import (
    gaussian_transducer_response,
    propagate_pulse_on_axis,
    square_burst,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exact angular-spectrum simulation of a focused transducer "
            "through a water–polypropylene–DMSO stack."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--frequency-mhz", type=float, default=10.0)
    parser.add_argument("--diameter-mm", type=float, default=13.0)
    parser.add_argument("--focal-length-mm", type=float, default=25.0)
    parser.add_argument("--water-path-mm", type=float, default=20.0)
    parser.add_argument("--plate-thickness-mm", type=float, default=0.78)
    parser.add_argument("--z-after-max-mm", type=float, default=10.0)
    parser.add_argument("--z-samples", type=int, default=161)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--dx-um", type=float, default=40.0)
    parser.add_argument("--radial-samples", type=int, default=4096)
    parser.add_argument("--no-bandlimit", action="store_true")

    parser.add_argument("--water-density", type=float, default=997.77)
    parser.add_argument("--water-speed", type=float, default=1488.4)
    parser.add_argument("--dmso-density", type=float, default=1098.4)
    parser.add_argument("--dmso-speed", type=float, default=1499.0)
    parser.add_argument("--pp-density", type=float, default=900.0)
    parser.add_argument(
        "--pp-longitudinal-speed",
        type=float,
        default=2732.0,
        help="Measured value from the shared conversation.",
    )
    parser.add_argument(
        "--pp-shear-speed",
        type=float,
        default=None,
        help="Measured shear speed. If omitted it is inferred from Poisson ratio.",
    )
    parser.add_argument("--pp-poisson", type=float, default=0.42)
    parser.add_argument(
        "--pp-alpha-l-db-m",
        type=float,
        default=0.0,
        help="Longitudinal amplitude attenuation at the simulated frequency.",
    )
    parser.add_argument(
        "--pp-alpha-s-db-m",
        type=float,
        default=0.0,
        help="Shear amplitude attenuation at the simulated frequency.",
    )
    parser.add_argument(
        "--pulse",
        action="store_true",
        help="Also reconstruct a filtered three-cycle square burst.",
    )
    return parser


def _build_model(args: argparse.Namespace) -> AngularSpectrumModel:
    frequency_hz = args.frequency_mhz * 1e6
    water = Fluid(
        "water_22C",
        density_kg_m3=args.water_density,
        sound_speed_m_s=args.water_speed,
    )
    dmso = Fluid(
        "DMSO_22C_estimate",
        density_kg_m3=args.dmso_density,
        sound_speed_m_s=args.dmso_speed,
    )
    if args.pp_shear_speed is None:
        polypropylene = ElasticSolid.from_longitudinal_speed_and_poisson(
            name="polypropylene",
            density_kg_m3=args.pp_density,
            longitudinal_speed_m_s=args.pp_longitudinal_speed,
            poisson_ratio=args.pp_poisson,
            longitudinal_attenuation_db_per_m=args.pp_alpha_l_db_m,
            shear_attenuation_db_per_m=args.pp_alpha_s_db_m,
            attenuation_reference_hz=frequency_hz,
        )
    else:
        polypropylene = ElasticSolid(
            "polypropylene",
            density_kg_m3=args.pp_density,
            longitudinal_speed_m_s=args.pp_longitudinal_speed,
            shear_speed_m_s=args.pp_shear_speed,
            longitudinal_attenuation_db_per_m=args.pp_alpha_l_db_m,
            shear_attenuation_db_per_m=args.pp_alpha_s_db_m,
            attenuation_reference_hz=frequency_hz,
        )
    return AngularSpectrumModel(
        grid=CartesianGrid(
            nx=args.nx,
            ny=args.nx,
            dx_m=args.dx_um * 1e-6,
        ),
        aperture=FocusedCircularAperture(
            diameter_m=args.diameter_mm * 1e-3,
            focal_length_m=args.focal_length_mm * 1e-3,
        ),
        incident_fluid=water,
        plate=ElasticPlate(
            polypropylene, thickness_m=args.plate_thickness_mm * 1e-3
        ),
        transmitted_fluid=dmso,
        water_path_m=args.water_path_mm * 1e-3,
        bandlimit=not args.no_bandlimit,
        plate_radial_samples=args.radial_samples,
    )


def _save_monochromatic_results(
    model: AngularSpectrumModel,
    frequency_hz: float,
    output_dir: Path,
    z_after_max_m: float,
    z_samples: int,
) -> dict[str, object]:
    z_after = np.linspace(0.0, z_after_max_m, z_samples)
    z_physical = model.water_path_m + model.plate.thickness_m + z_after
    axial = model.on_axis_scan_after_plate(frequency_hz, z_after)
    axial_reference = model.reference_on_axis_scan(frequency_hz, z_after)

    focus_index = int(np.argmax(np.abs(axial)))
    reference_focus_index = int(np.argmax(np.abs(axial_reference)))
    focus_after = float(z_after[focus_index])
    focus_physical = float(z_physical[focus_index])
    field = model.field_after_plate(frequency_hz, focus_after)
    centre_y, _ = model.grid.centre_index
    lateral = field[centre_y, :]
    lateral_width = fwhm(model.grid.x_m, lateral)
    axial_width = fwhm(z_physical, axial)

    peak = float(np.max(np.abs(axial)))
    reference_peak = float(np.max(np.abs(axial_reference)))
    shared_scale = max(peak, reference_peak)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(
        z_physical * 1e3,
        np.abs(axial_reference) / shared_scale,
        label="ohne PP (durch Wasser ersetzt)",
    )
    ax.plot(
        z_physical * 1e3,
        np.abs(axial) / shared_scale,
        label="elastische PP-Platte",
    )
    ax.axvline(
        model.aperture.focal_length_m * 1e3,
        color="0.5",
        linestyle="--",
        linewidth=1.0,
        label="geometrischer Fokus",
    )
    ax.set(xlabel="z ab Aperturebene [mm]", ylabel="Druckamplitude / gemeinsames Maximum")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "axial_scan.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(model.grid.x_m * 1e3, amplitude_db(lateral, -50.0))
    ax.axhline(-6.0206, color="0.5", linestyle="--", linewidth=1.0)
    ax.set(
        xlabel="x [mm]",
        ylabel="normierte Druckamplitude [dB]",
        xlim=(-2.0, 2.0),
        ylim=(-50.0, 1.0),
        title=f"Laterales Profil bei z = {focus_physical * 1e3:.3f} mm",
    )
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "lateral_profile.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.1), constrained_layout=True)
    extent_mm = [
        model.grid.x_m[0] * 1e3,
        model.grid.x_m[-1] * 1e3,
        model.grid.y_m[0] * 1e3,
        model.grid.y_m[-1] * 1e3,
    ]
    image = ax.imshow(
        amplitude_db(field, -40.0),
        extent=extent_mm,
        origin="lower",
        cmap="magma",
        vmin=-40.0,
        vmax=0.0,
    )
    ax.set(
        xlabel="x [mm]",
        ylabel="y [mm]",
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        title="Fokalebene mit elastischer PP-Transmission",
    )
    fig.colorbar(image, ax=ax, label="normierte Druckamplitude [dB]")
    fig.savefig(output_dir / "focal_plane.png", dpi=180)
    plt.close(fig)

    frequency_sweep = np.linspace(
        0.7 * frequency_hz, 1.3 * frequency_hz, 401
    )
    normal_transmission = np.empty(frequency_sweep.size, dtype=np.complex128)
    for index, sweep_frequency in enumerate(frequency_sweep):
        normal_transmission[index] = elastic_plate_scattering(
            np.array([0.0]),
            float(sweep_frequency),
            model.incident_fluid,
            model.plate,
            model.transmitted_fluid,
        )[1][0]
    normal_power = np.array(
        [
            normal_power_transmission(
                np.array([0.0]),
                float(sweep_frequency),
                model.incident_fluid,
                model.transmitted_fluid,
                np.array([normal_transmission[index]]),
            )[0]
            for index, sweep_frequency in enumerate(frequency_sweep)
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(frequency_sweep * 1e-6, normal_power)
    ax.axvline(frequency_hz * 1e-6, color="0.5", linestyle="--", linewidth=1.0)
    ax.set(
        xlabel="Frequenz [MHz]",
        ylabel="normale Leistungstransmission",
        ylim=(0.0, max(1.05, float(np.nanmax(normal_power)) * 1.05)),
        title="PP-Plattenresonanzen bei Normalinzidenz",
    )
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "transmission_vs_frequency.png", dpi=180)
    plt.close(fig)

    angle_deg = np.linspace(0.0, 30.0, 301)
    incident_k_real = (
        2.0 * np.pi * frequency_hz / model.incident_fluid.sound_speed_m_s
    )
    q_angle = incident_k_real * np.sin(np.radians(angle_deg))
    reflection_angle, transmission_angle = elastic_plate_scattering(
        q_angle,
        frequency_hz,
        model.incident_fluid,
        model.plate,
        model.transmitted_fluid,
    )
    transmitted_power_angle = normal_power_transmission(
        q_angle,
        frequency_hz,
        model.incident_fluid,
        model.transmitted_fluid,
        transmission_angle,
    )
    reflected_power_angle = np.abs(reflection_angle) ** 2
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(angle_deg, transmitted_power_angle, label="Transmission")
    ax.plot(angle_deg, reflected_power_angle, label="Reflexion", alpha=0.8)
    ax.axvline(
        model.sampling_report(frequency_hz)["aperture_edge_angle_deg"],
        color="0.5",
        linestyle="--",
        linewidth=1.0,
        label="Aperturrandwinkel",
    )
    ax.set(
        xlabel="Einfallswinkel in Wasser [°]",
        ylabel="Leistungskoeffizient",
        ylim=(0.0, 1.05),
        title=f"Winkelabhängigkeit bei {frequency_hz * 1e-6:.1f} MHz",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "transmission_vs_angle.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "monochromatic_results.npz",
        x_m=model.grid.x_m,
        y_m=model.grid.y_m,
        z_after_plate_m=z_after,
        z_physical_m=z_physical,
        axial_pressure=axial,
        axial_reference_pressure=axial_reference,
        focal_plane_pressure=field,
        frequency_sweep_hz=frequency_sweep,
        normal_pressure_transmission=normal_transmission,
        angle_deg=angle_deg,
        angular_pressure_transmission=transmission_angle,
    )

    edge_water = np.radians(
        model.sampling_report(frequency_hz)["aperture_edge_angle_deg"]
    )
    pp_l_argument = (
        model.plate.solid.longitudinal_speed_m_s
        / model.incident_fluid.sound_speed_m_s
        * np.sin(edge_water)
    )
    pp_s_argument = (
        model.plate.solid.shear_speed_m_s
        / model.incident_fluid.sound_speed_m_s
        * np.sin(edge_water)
    )
    return {
        "frequency_hz": frequency_hz,
        "focus_physical_m": focus_physical,
        "focus_after_plate_m": focus_after,
        "reference_focus_physical_m": float(
            z_physical[reference_focus_index]
        ),
        "lateral_fwhm_m": lateral_width,
        "axial_fwhm_m": axial_width,
        "peak_pressure_for_1Pa_aperture_pa": peak,
        "reference_peak_pressure_for_1Pa_aperture_pa": reference_peak,
        "peak_pressure_ratio_plate_to_reference": (
            peak / reference_peak if reference_peak > 0.0 else None
        ),
        "normal_power_transmission_at_center_frequency": float(
            normal_power[len(normal_power) // 2]
        ),
        "maximum_lossless_energy_balance_error": float(
            np.nanmax(
                np.abs(
                    reflected_power_angle + transmitted_power_angle - 1.0
                )
            )
        ),
        "pp_longitudinal_refraction_at_aperture_edge_deg": (
            float(np.degrees(np.arcsin(pp_l_argument)))
            if abs(pp_l_argument) <= 1.0
            else None
        ),
        "pp_shear_refraction_at_aperture_edge_deg": (
            float(np.degrees(np.arcsin(pp_s_argument)))
            if abs(pp_s_argument) <= 1.0
            else None
        ),
    }


def _save_pulse_result(
    model: AngularSpectrumModel,
    frequency_hz: float,
    focus_after_plate_m: float,
    output_dir: Path,
) -> dict[str, object]:
    time, drive = square_burst(
        center_frequency_hz=frequency_hz,
        cycles=3.0,
        sample_rate_hz=80.0e6,
        record_length_s=32.0e-6,
    )
    response = lambda f: gaussian_transducer_response(
        f,
        center_frequency_hz=frequency_hz,
        fractional_bandwidth_6db=0.4,
    )
    result = propagate_pulse_on_axis(
        model,
        time,
        drive,
        z_after_plate_m=focus_after_plate_m,
        transducer_response=response,
        relative_spectrum_threshold=2.0e-2,
        minimum_frequency_hz=0.5 * frequency_hz,
        maximum_frequency_hz=1.5 * frequency_hz,
    )
    normalization = max(float(np.max(np.abs(result.output_signal))), 1e-30)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), constrained_layout=True)
    axes[0].plot(result.time_s * 1e6, result.input_signal)
    axes[0].set(xlabel="Zeit [µs]", ylabel="Ansteuerung [rel.]")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        result.time_s * 1e6, result.output_signal / normalization
    )
    axes[1].set(
        xlabel="Zeit [µs]",
        ylabel="Druck [normiert]",
        title="Rekonstruiertes Signal im Fokus hinter PP",
    )
    axes[1].grid(alpha=0.25)
    fig.savefig(output_dir / "pulse_response.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(
        output_dir / "pulse_results.npz",
        time_s=result.time_s,
        input_signal=result.input_signal,
        output_signal=result.output_signal,
        frequency_hz=result.frequency_hz,
        input_spectrum=result.input_spectrum,
        output_spectrum=result.output_spectrum,
        simulated_bin_mask=result.simulated_bin_mask,
    )
    return {
        "pulse_simulated_frequency_bins": int(
            np.count_nonzero(result.simulated_bin_mask)
        ),
        "pulse_output_peak_relative": float(
            np.max(np.abs(result.output_signal))
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_hz = args.frequency_mhz * 1e6
    model = _build_model(args)
    summary: dict[str, object] = {
        "model": "3D FFT angular spectrum with exact kz and isotropic P-SV plate solve",
        "phasor_convention": "exp(-i omega t)",
        "materials": {
            "water": {
                "density_kg_m3": model.incident_fluid.density_kg_m3,
                "sound_speed_m_s": model.incident_fluid.sound_speed_m_s,
            },
            "polypropylene": {
                "density_kg_m3": model.plate.solid.density_kg_m3,
                "longitudinal_speed_m_s": (
                    model.plate.solid.longitudinal_speed_m_s
                ),
                "shear_speed_m_s": model.plate.solid.shear_speed_m_s,
                "poisson_ratio": model.plate.solid.poisson_ratio,
                "longitudinal_attenuation_db_per_m": (
                    model.plate.solid.longitudinal_attenuation_db_per_m
                ),
                "shear_attenuation_db_per_m": (
                    model.plate.solid.shear_attenuation_db_per_m
                ),
            },
            "DMSO": {
                "density_kg_m3": model.transmitted_fluid.density_kg_m3,
                "sound_speed_m_s": model.transmitted_fluid.sound_speed_m_s,
            },
        },
        "geometry": {
            "aperture_diameter_m": model.aperture.diameter_m,
            "geometric_focal_length_m": model.aperture.focal_length_m,
            "water_path_to_plate_m": model.water_path_m,
            "plate_thickness_m": model.plate.thickness_m,
        },
        "sampling": model.sampling_report(frequency_hz),
        "assumptions_requiring_measurement": [
            "PP shear speed (currently inferred from Poisson ratio unless supplied)",
            "PP density unless supplied",
            "PP longitudinal and shear attenuation (zero by default)",
            "DMSO sound speed, density and attenuation for the actual concentration",
            "absolute electro-acoustic source calibration",
        ],
    }
    summary.update(
        _save_monochromatic_results(
            model,
            frequency_hz,
            output_dir,
            args.z_after_max_mm * 1e-3,
            args.z_samples,
        )
    )
    if args.pulse:
        summary.update(
            _save_pulse_result(
                model,
                frequency_hz,
                float(summary["focus_after_plate_m"]),
                output_dir,
            )
        )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = _parser().parse_args()
    summary = run(args)
    print(
        "Fokus:",
        f"{float(summary['focus_physical_m']) * 1e3:.3f} mm;",
        "laterale -6-dB-Breite:",
        f"{float(summary['lateral_fwhm_m']) * 1e3:.3f} mm;",
        "Ergebnisse:",
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
