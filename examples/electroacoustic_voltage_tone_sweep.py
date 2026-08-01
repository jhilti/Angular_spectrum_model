"""Voltage and tone-length sweep using the optional electro-acoustic layer.

The default 50-ohm probe impedance and unit transmit/receive sensitivities are
placeholders.  They demonstrate scaling and pulse shaping but are not an
absolute calibration of the Doppler I2-10P13F25-H probe.  Replace them with a
measured complex impedance and hydrophone/receiver calibration before treating
the Pa, V, or ADC axes as absolute.
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
    ElasticPlate,
    ElasticSolid,
    ElectroAcousticCalibration,
    Fluid,
    FocusedCircularAperture,
    asymmetric_gaussian_response,
    dmso_water_properties,
    simulate_electroacoustic_pulse_echo,
    sine_burst,
    smooth_dc_block_response,
    validate_focused_grid_support,
    water_properties,
)


TRANSDUCER_CENTER_FREQUENCY_HZ = 9.97e6
TRANSDUCER_PEAK_FREQUENCY_HZ = 11.29e6
TRANSDUCER_FRACTIONAL_BANDWIDTH_6DB = 1.0822
TRANSDUCER_LOWER_FREQUENCY_6DB_HZ = TRANSDUCER_CENTER_FREQUENCY_HZ * (
    1.0 - TRANSDUCER_FRACTIONAL_BANDWIDTH_6DB / 2.0
)
TRANSDUCER_UPPER_FREQUENCY_6DB_HZ = TRANSDUCER_CENTER_FREQUENCY_HZ * (
    1.0 + TRANSDUCER_FRACTIONAL_BANDWIDTH_6DB / 2.0
)
BURST_START_S = 0.25e-6
MINIMUM_ANALYSIS_BAND_HZ = 25.0e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep voltage and burst length through the optional calibrated "
            "electro-acoustic pulse-echo model."
        )
    )
    parser.add_argument("--source-voltage-v", type=float, default=20.0)
    parser.add_argument(
        "--cycles",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 3.0, 5.0],
    )
    parser.add_argument(
        "--voltage-sweep-v",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 40.0],
    )
    parser.add_argument("--source-impedance-ohm", type=float, default=50.0)
    parser.add_argument("--transducer-impedance-ohm", type=float, default=50.0)
    parser.add_argument("--transmit-pa-per-v", type=float, default=1.0)
    parser.add_argument("--receive-v-per-pa", type=float, default=1.0)
    parser.add_argument("--receiver-gain", type=float, default=1.0)
    parser.add_argument("--adc-counts-per-v", type=float)
    parser.add_argument(
        "--absolute-calibration",
        action="store_true",
        help="assert that all supplied sensitivities and gains are calibrated",
    )
    parser.add_argument("--dmso-percent", type=float, default=80.0)
    parser.add_argument("--temperature-c", type=float, default=22.0)
    parser.add_argument("--water-path-mm", type=float, default=25.3)
    parser.add_argument("--pp-thickness-mm", type=float, default=0.78)
    parser.add_argument("--fluid-height-mm", type=float, default=4.22)
    parser.add_argument("--frequency-mhz", type=float, default=10.0)
    parser.add_argument("--sample-rate-mhz", type=float, default=80.0)
    parser.add_argument("--grid-size", type=int, default=320)
    parser.add_argument("--grid-spacing-um", type=float, default=100.0)
    parser.add_argument("--radial-samples", type=int, default=768)
    parser.add_argument("--relative-threshold", type=float, default=7.0e-3)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results"),
    )
    return parser.parse_args()


def require_positive(name: str, values: np.ndarray | float) -> None:
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only finite values > 0")


def main() -> None:
    args = parse_args()
    for name, values in {
        "source voltage": args.source_voltage_v,
        "cycles": args.cycles,
        "voltage sweep": args.voltage_sweep_v,
        "source impedance": args.source_impedance_ohm,
        "transducer impedance": args.transducer_impedance_ohm,
        "transmit sensitivity": args.transmit_pa_per_v,
        "receive sensitivity": args.receive_v_per_pa,
        "receiver gain": args.receiver_gain,
        "water path": args.water_path_mm,
        "PP thickness": args.pp_thickness_mm,
        "fluid height": args.fluid_height_mm,
        "frequency": args.frequency_mhz,
        "sample rate": args.sample_rate_mhz,
        "grid size": args.grid_size,
        "grid spacing": args.grid_spacing_um,
        "radial samples": args.radial_samples,
    }.items():
        require_positive(name, values)
    if args.adc_counts_per_v is not None:
        require_positive("ADC counts per volt", args.adc_counts_per_v)
    if not 0.0 <= args.dmso_percent <= 100.0:
        raise ValueError("dmso-percent must lie between 0 and 100")
    if not 0.0 <= args.relative_threshold < 1.0:
        raise ValueError("relative-threshold must lie in [0, 1)")

    dmso_properties = dmso_water_properties(
        args.dmso_percent / 100.0,
        basis="volume",
        temperature_c=args.temperature_c,
    )
    water_data = water_properties(args.temperature_c)
    water = Fluid(
        f"water_{args.temperature_c:g}C",
        water_data.density_kg_m3,
        water_data.sound_speed_m_s,
    )
    dmso = Fluid(
        f"{args.dmso_percent:g}volpct_DMSO",
        dmso_properties.density_kg_m3,
        dmso_properties.sound_speed_m_s,
    )
    air = Fluid("air_22C", 1.196, 344.0)
    polypropylene = ElasticSolid.from_longitudinal_speed_and_poisson(
        name="polypropylene",
        density_kg_m3=900.0,
        longitudinal_speed_m_s=2732.0,
        poisson_ratio=0.42,
    )
    model = AngularSpectrumModel(
        grid=CartesianGrid(
            nx=args.grid_size,
            ny=args.grid_size,
            dx_m=args.grid_spacing_um * 1e-6,
        ),
        aperture=FocusedCircularAperture(
            diameter_m=13.0e-3,
            focal_length_m=25.4e-3,
            pressure_amplitude_pa=1.0,
        ),
        incident_fluid=water,
        plate=ElasticPlate(
            polypropylene,
            thickness_m=args.pp_thickness_mm * 1e-3,
        ),
        transmitted_fluid=dmso,
        water_path_m=args.water_path_mm * 1e-3,
        plate_radial_samples=args.radial_samples,
    )
    maximum_frequency_hz = max(
        MINIMUM_ANALYSIS_BAND_HZ,
        1.5 * args.frequency_mhz * 1e6,
    )
    validate_focused_grid_support(
        model,
        maximum_frequency_hz=maximum_frequency_hz,
        propagation_segments=(
            (
                "water pulse-echo round trip",
                water,
                2.0 * args.water_path_mm * 1e-3,
            ),
            (
                "liquid-layer cavity round trip",
                dmso,
                2.0 * args.fluid_height_mm * 1e-3,
            ),
        ),
    )

    def one_way_certificate_shape(frequency_hz: np.ndarray) -> np.ndarray:
        pulse_echo_shape = asymmetric_gaussian_response(
            frequency_hz,
            peak_frequency_hz=TRANSDUCER_PEAK_FREQUENCY_HZ,
            lower_frequency_6db_hz=TRANSDUCER_LOWER_FREQUENCY_6DB_HZ,
            upper_frequency_6db_hz=TRANSDUCER_UPPER_FREQUENCY_6DB_HZ,
        ) * smooth_dc_block_response(frequency_hz)
        return np.sqrt(pulse_echo_shape)

    calibration = ElectroAcousticCalibration(
        transmit_pressure_pa_per_v=lambda frequency: (
            args.transmit_pa_per_v * one_way_certificate_shape(frequency)
        ),
        receive_voltage_v_per_pa=lambda frequency: (
            args.receive_v_per_pa * one_way_certificate_shape(frequency)
        ),
        receiver_response=args.receiver_gain,
        adc_counts_per_v=args.adc_counts_per_v,
        absolute=args.absolute_calibration,
    )

    water_path_m = args.water_path_mm * 1e-3
    plate_thickness_m = args.pp_thickness_mm * 1e-3
    fluid_height_m = args.fluid_height_mm * 1e-3
    first_echo_s = (
        BURST_START_S + 2.0 * water_path_m / water.sound_speed_m_s
    )
    surface_echo_s = first_echo_s + 2.0 * (
        plate_thickness_m / polypropylene.longitudinal_speed_m_s
        + fluid_height_m / dmso.sound_speed_m_s
    )
    liquid_cavity_round_trip_s = 2.0 * fluid_height_m / dmso.sound_speed_m_s
    longest_drive_duration_s = max(args.cycles) / (args.frequency_mhz * 1e6)
    record_length_s = surface_echo_s + max(
        8.0e-6,
        liquid_cavity_round_trip_s,
    ) + 2.0 * longest_drive_duration_s
    results = []
    for cycles in args.cycles:
        time_s, open_circuit_voltage = sine_burst(
            center_frequency_hz=args.frequency_mhz * 1e6,
            cycles=float(cycles),
            sample_rate_hz=args.sample_rate_mhz * 1e6,
            record_length_s=record_length_s,
            start_time_s=BURST_START_S,
            amplitude=args.source_voltage_v,
        )
        result = simulate_electroacoustic_pulse_echo(
            model,
            time_s,
            open_circuit_voltage,
            source_impedance_ohm=args.source_impedance_ohm,
            transducer_impedance_ohm=args.transducer_impedance_ohm,
            calibration=calibration,
            fluid_layer_thickness_m=fluid_height_m,
            backing_fluid=air,
            relative_spectrum_threshold=args.relative_threshold,
            minimum_frequency_hz=0.0,
            maximum_frequency_hz=maximum_frequency_hz,
            fluid_cavity_echo_count=1,
        )
        results.append((float(cycles), result))

    cycle_values = np.asarray([item[0] for item in results])
    composite_received_peaks_v = np.asarray(
        [item[1].peak_received_voltage_v for item in results]
    )
    plate_front_peaks_v = np.asarray(
        [float(np.max(np.abs(item[1].plate_front_voltage_v))) for item in results]
    )
    first_surface_peaks_v = []
    for cycles, result in results:
        # Backing voltage contains the retained first DMSO-air return. Gate it
        # around its causal arrival as an extra guard against the small
        # non-causal tails introduced by the provisional zero-phase response.
        drive_duration_s = cycles / (args.frequency_mhz * 1e6)
        gate = (
            (result.electrical.time_s >= surface_echo_s - 0.5e-6)
            & (
                result.electrical.time_s
                <= surface_echo_s + drive_duration_s + 1.0e-6
            )
        )
        first_surface_peaks_v.append(
            float(np.max(np.abs(result.backing_voltage_v[gate])))
        )
    first_surface_peaks_v = np.asarray(first_surface_peaks_v)
    aperture_peaks_pa = np.asarray(
        [item[1].peak_aperture_pressure_pa for item in results]
    )
    terminal_peaks_v = np.asarray(
        [item[1].electrical.peak_terminal_voltage_v for item in results]
    )
    delivered_energy_uj = np.asarray(
        [item[1].electrical.delivered_energy_j * 1e6 for item in results]
    )
    reference_index = int(np.argmin(np.abs(cycle_values - 1.0)))
    voltage_values = np.asarray(args.voltage_sweep_v, dtype=float)
    voltage_sweep_peaks = (
        first_surface_peaks_v[reference_index]
        * voltage_values
        / args.source_voltage_v
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    receive_unit = "V" if args.absolute_calibration else "a.u."
    receive_ylabel = (
        "Received voltage [V]"
        if args.absolute_calibration
        else "Scaled receiver output [a.u.]"
    )
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9.0, 9.2),
        constrained_layout=True,
    )
    for cycles, result in results:
        time_since_excitation_us = (
            result.electrical.time_s - BURST_START_S
        ) * 1e6
        cycle_label = f"{cycles:g} cycle" + ("" if cycles == 1.0 else "s")
        axes[0].plot(
            time_since_excitation_us,
            result.received_voltage_v,
            label=cycle_label,
        )
    axes[0].set_xlim(0.0, (surface_echo_s - BURST_START_S) * 1e6 + 2.0)
    axes[0].set(
        xlabel="Time since excitation start [µs]",
        ylabel=receive_ylabel,
        title="Open-circuit-voltage-driven pulse echo",
    )
    axes[0].legend(ncols=3)

    axes[1].plot(
        cycle_values,
        first_surface_peaks_v,
        "o-",
        label="first surface echo (gated)",
    )
    axes[1].plot(
        cycle_values,
        composite_received_peaks_v,
        "o:",
        color="#7b6d8d",
        label="global composite peak",
    )
    axes[1].set(
        xlabel="Burst length [cycles]",
        ylabel=f"Peak receiver output [{receive_unit}]",
        title="Receiver metrics — not meniscus forcing or ejection efficiency",
    )
    energy_axis = axes[1].twinx()
    energy_axis.plot(
        cycle_values,
        delivered_energy_uj,
        "s--",
        color="#d1495b",
        label="net absorbed load energy",
    )
    energy_axis.set_ylabel("Net load energy [µJ]", color="#d1495b")
    axes[1].legend(loc="upper left")
    energy_axis.legend(loc="upper right")

    axes[2].plot(voltage_values, voltage_sweep_peaks, "o-")
    axes[2].set(
        xlabel="Open-circuit source peak voltage [V]",
        ylabel=f"Peak receiver output [{receive_unit}]",
        title=(
            "Linear first-surface echo scaling at "
            f"{cycle_values[reference_index]:g} cycle"
            + ("" if cycle_values[reference_index] == 1.0 else "s")
        ),
    )
    for axis in axes:
        axis.grid(alpha=0.23)
    if args.absolute_calibration:
        figure.suptitle(
            "USER-ASSERTED ABSOLUTE SCALE — verify calibration provenance",
            color="#8a5a00",
            fontsize=11,
        )
    else:
        figure.suptitle(
            "PROVISIONAL SCALE — replace impedance and Pa/V, V/Pa calibration",
            color="#b23a48",
            fontsize=11,
        )
    figure_path = args.output_directory / "electroacoustic_voltage_tone_sweep.png"
    figure.savefig(figure_path, dpi=190)
    plt.close(figure)

    csv_path = args.output_directory / "electroacoustic_voltage_tone_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "cycles",
                "terminal_peak_v",
                (
                    "aperture_peak_pa"
                    if args.absolute_calibration
                    else "aperture_peak_provisional_au"
                ),
                (
                    "first_surface_gated_peak_v"
                    if args.absolute_calibration
                    else "first_surface_gated_peak_provisional_au"
                ),
                (
                    "global_composite_peak_v"
                    if args.absolute_calibration
                    else "global_composite_peak_provisional_au"
                ),
                (
                    "plate_front_peak_v"
                    if args.absolute_calibration
                    else "plate_front_peak_provisional_au"
                ),
                "net_absorbed_load_energy_uj",
                "absolute_calibration",
            ]
        )
        for index, cycles in enumerate(cycle_values):
            writer.writerow(
                [
                    cycles,
                    terminal_peaks_v[index],
                    aperture_peaks_pa[index],
                    first_surface_peaks_v[index],
                    composite_received_peaks_v[index],
                    plate_front_peaks_v[index],
                    delivered_energy_uj[index],
                    args.absolute_calibration,
                ]
            )

    json_path = args.output_directory / "electroacoustic_voltage_tone_sweep.json"
    summary = {
        "scope": {
            "absolute_calibration": args.absolute_calibration,
            "absolute_calibration_is_user_asserted": bool(
                args.absolute_calibration
            ),
            "voltage_reference": "open-circuit Thevenin peak voltage",
            "amplitude_scale": (
                "user-asserted absolute"
                if args.absolute_calibration
                else "provisional"
            ),
            "warning": (
                "absolute status is a user assertion; preserve calibration "
                "files, reference planes, loading, gain, and uncertainty"
                if args.absolute_calibration
                else "50 ohm probe and unit Pa/V and V/Pa are placeholders"
            ),
            "tone_metric_warning": (
                "receiver echo peaks do not predict meniscus forcing or "
                "droplet-ejection efficiency"
            ),
        },
        "drive": {
            "frequency_hz": args.frequency_mhz * 1e6,
            "open_circuit_source_peak_v": args.source_voltage_v,
            "source_impedance_ohm": args.source_impedance_ohm,
            "probe_impedance_ohm": args.transducer_impedance_ohm,
            "cycles": cycle_values.tolist(),
            "voltage_sweep_v": voltage_values.tolist(),
        },
        "geometry": {
            "water_path_m": water_path_m,
            "pp_thickness_m": plate_thickness_m,
            "fluid_height_m": fluid_height_m,
        },
        "materials": {
            "temperature_c": args.temperature_c,
            "dmso_volume_percent": args.dmso_percent,
            "water_density_kg_m3": water.density_kg_m3,
            "water_sound_speed_m_s": water.sound_speed_m_s,
            "fluid_density_kg_m3": dmso.density_kg_m3,
            "fluid_sound_speed_m_s": dmso.sound_speed_m_s,
        },
        "numerics": {
            "grid_size": args.grid_size,
            "grid_spacing_m": args.grid_spacing_um * 1e-6,
            "grid_window_m": model.grid.extent_x_m,
            "grid_validation_frequency_hz": maximum_frequency_hz,
            "sample_rate_hz": args.sample_rate_mhz * 1e6,
            "record_length_s": record_length_s,
        },
        "tone_sweep": [
            {
                "cycles": float(cycle_values[index]),
                "terminal_peak_v": float(terminal_peaks_v[index]),
                "aperture_peak": float(aperture_peaks_pa[index]),
                "first_surface_gated_peak": float(
                    first_surface_peaks_v[index]
                ),
                "global_composite_received_peak": float(
                    composite_received_peaks_v[index]
                ),
                "plate_front_peak": float(plate_front_peaks_v[index]),
                "net_absorbed_load_energy_j": float(
                    delivered_energy_uj[index] * 1e-6
                ),
                "fluid_cavity_echo_count": int(
                    results[index][1].acoustic.fluid_cavity_echo_count
                ),
            }
            for index in range(cycle_values.size)
        ],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    scale_label = (
        "user-asserted absolute"
        if args.absolute_calibration
        else "provisional"
    )
    print(f"Calibration scale: {scale_label}")
    aperture_unit = "Pa" if args.absolute_calibration else "a.u."
    print(
        "cycles  terminal[V]  "
        f"aperture[{aperture_unit}]  surface[{receive_unit}]  "
        f"composite[{receive_unit}]  "
        "net-load-energy[µJ]"
    )
    for index, cycles in enumerate(cycle_values):
        print(
            f"{cycles:6.2f}  {terminal_peaks_v[index]:11.5g}  "
            f"{aperture_peaks_pa[index]:12.5g}  "
            f"{first_surface_peaks_v[index]:10.5g}  "
            f"{composite_received_peaks_v[index]:12.5g}  "
            f"{delivered_energy_uj[index]:10.5g}"
        )
    print(f"Saved {figure_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
