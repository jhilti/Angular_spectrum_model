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
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--grid-spacing-um", type=float, default=62.5)
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
    water = Fluid("water_22C", 997.77, 1488.4)
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

    def one_way_certificate_shape(frequency_hz: np.ndarray) -> np.ndarray:
        pulse_echo_shape = asymmetric_gaussian_response(
            frequency_hz,
            peak_frequency_hz=TRANSDUCER_PEAK_FREQUENCY_HZ,
            lower_frequency_6db_hz=TRANSDUCER_LOWER_FREQUENCY_6DB_HZ,
            upper_frequency_6db_hz=TRANSDUCER_UPPER_FREQUENCY_6DB_HZ,
        )
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
    first_echo_s = 2.0 * water_path_m / water.sound_speed_m_s
    surface_echo_s = first_echo_s + 2.0 * (
        plate_thickness_m / polypropylene.longitudinal_speed_m_s
        + fluid_height_m / dmso.sound_speed_m_s
    )
    record_length_s = surface_echo_s + 4.0e-6
    results = []
    for cycles in args.cycles:
        time_s, open_circuit_voltage = sine_burst(
            center_frequency_hz=args.frequency_mhz * 1e6,
            cycles=float(cycles),
            sample_rate_hz=args.sample_rate_mhz * 1e6,
            record_length_s=record_length_s,
            start_time_s=0.25e-6,
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
            minimum_frequency_hz=2.5e6,
            maximum_frequency_hz=20.0e6,
        )
        results.append((float(cycles), result))

    cycle_values = np.asarray([item[0] for item in results])
    received_peaks_v = np.asarray(
        [item[1].peak_received_voltage_v for item in results]
    )
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
        received_peaks_v[reference_index]
        * voltage_values
        / args.source_voltage_v
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9.0, 9.2),
        constrained_layout=True,
    )
    for cycles, result in results:
        relative_time_us = (result.electrical.time_s - first_echo_s) * 1e6
        cycle_label = f"{cycles:g} cycle" + ("" if cycles == 1.0 else "s")
        axes[0].plot(
            relative_time_us,
            result.received_voltage_v,
            label=cycle_label,
        )
    axes[0].set_xlim(-1.0, (surface_echo_s - first_echo_s) * 1e6 + 2.0)
    axes[0].set(
        xlabel="Time relative to water–PP echo [µs]",
        ylabel="Received voltage [V]",
        title="Voltage-driven pulse echo",
    )
    axes[0].legend(ncols=3)

    axes[1].plot(cycle_values, received_peaks_v, "o-", label="received peak")
    axes[1].set(
        xlabel="Burst length [cycles]",
        ylabel="Peak received voltage [V]",
        title="Tone-length response",
    )
    energy_axis = axes[1].twinx()
    energy_axis.plot(
        cycle_values,
        delivered_energy_uj,
        "s--",
        color="#d1495b",
        label="delivered electrical energy",
    )
    energy_axis.set_ylabel("Delivered energy [µJ]", color="#d1495b")

    axes[2].plot(voltage_values, voltage_sweep_peaks, "o-")
    axes[2].set(
        xlabel="Open-circuit source peak voltage [V]",
        ylabel="Peak received voltage [V]",
        title=(
            "Linear voltage sweep at "
            f"{cycle_values[reference_index]:g} cycle"
            + ("" if cycle_values[reference_index] == 1.0 else "s")
        ),
    )
    for axis in axes:
        axis.grid(alpha=0.23)
    if not args.absolute_calibration:
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
                "aperture_peak_pa",
                "received_peak_v",
                "delivered_energy_uj",
                "absolute_calibration",
            ]
        )
        for index, cycles in enumerate(cycle_values):
            writer.writerow(
                [
                    cycles,
                    terminal_peaks_v[index],
                    aperture_peaks_pa[index],
                    received_peaks_v[index],
                    delivered_energy_uj[index],
                    args.absolute_calibration,
                ]
            )

    scale_label = "absolute" if args.absolute_calibration else "provisional"
    print(f"Calibration scale: {scale_label}")
    print("cycles  terminal[V]  aperture[Pa]  receive[V]  energy[µJ]")
    for index, cycles in enumerate(cycle_values):
        print(
            f"{cycles:6.2f}  {terminal_peaks_v[index]:11.5g}  "
            f"{aperture_peaks_pa[index]:12.5g}  "
            f"{received_peaks_v[index]:10.5g}  "
            f"{delivered_energy_uj[index]:10.5g}"
        )
    print(f"Saved {figure_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
