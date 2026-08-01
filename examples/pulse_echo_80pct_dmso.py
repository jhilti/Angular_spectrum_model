"""Measurement-like pulse echo for water -> PP -> DMSO -> air.

The default case remains the requested 80 vol.% DMSO and 4.22 mm fill
height.  The electro-acoustic response now uses the *round-trip* bandwidth
reported on the Doppler I2-10P13F25-H pulse-echo certificate.  An optional
survey JSON can be overlaid after independent ADC normalization; its unknown
fluid, temperature, height and absolute amplitude are never used to fit the
physical model.
"""

from __future__ import annotations

import argparse
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
    Fluid,
    FocusedCircularAperture,
    asymmetric_gaussian_response,
    dmso_water_properties,
    simulate_monostatic_pulse_echo,
    sine_burst,
    smooth_dc_block_response,
    validate_focused_grid_support,
    water_properties,
)


DRIVE_FREQUENCY_HZ = 10.0e6
TRANSDUCER_CENTER_FREQUENCY_HZ = 9.97e6
TRANSDUCER_PEAK_FREQUENCY_HZ = 11.29e6
TRANSDUCER_FRACTIONAL_BANDWIDTH_6DB_ROUND_TRIP = 1.0822
TRANSDUCER_LOWER_FREQUENCY_6DB_HZ = (
    TRANSDUCER_CENTER_FREQUENCY_HZ
    * (1.0 - TRANSDUCER_FRACTIONAL_BANDWIDTH_6DB_ROUND_TRIP / 2.0)
)
TRANSDUCER_UPPER_FREQUENCY_6DB_HZ = (
    TRANSDUCER_CENTER_FREQUENCY_HZ
    * (1.0 + TRANSDUCER_FRACTIONAL_BANDWIDTH_6DB_ROUND_TRIP / 2.0)
)
TRANSDUCER_PULSE_DURATION_6DB_S = 66.0e-9
TRANSDUCER_FOCAL_LENGTH_M = 25.40e-3
TRANSDUCER_DIAMETER_M = 13.0e-3
GRID_VALIDATION_FREQUENCY_HZ = 25.0e6

WATER_PATH_M = 25.3e-3
PP_THICKNESS_M = 0.78e-3
DEFAULT_DMSO_HEIGHT_MM = 4.22
DEFAULT_DMSO_VOLUME_PERCENT = 80.0
DEFAULT_TEMPERATURE_C = 22.0

SAMPLE_RATE_HZ = 160.0e6
RECORD_LENGTH_S = 60.0e-6
OUTPUT_DIRECTORY = Path("results")


def analytic_envelope(signal: np.ndarray) -> np.ndarray:
    """Return the magnitude of the FFT-based analytic signal."""

    spectrum = np.fft.fft(signal)
    multiplier = np.zeros(signal.size)
    multiplier[0] = 1.0
    if signal.size % 2 == 0:
        multiplier[1 : signal.size // 2] = 2.0
        multiplier[signal.size // 2] = 1.0
    else:
        multiplier[1 : (signal.size + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier))


def peak_time(
    time_s: np.ndarray,
    signal: np.ndarray,
    centre_s: float,
    half_width_s: float,
) -> float:
    mask = np.abs(time_s - centre_s) <= half_width_s
    local_indices = np.flatnonzero(mask)
    return float(time_s[local_indices[np.argmax(np.abs(signal[mask]))]])


def load_normalized_survey(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], list[str]]:
    """Load a 250 MHz survey trace for qualitative overlay only."""

    with path.open("r", encoding="utf-8") as handle:
        survey = json.load(handle)
    super_pings = survey.get("SurveyPingsSuper") or []
    if not super_pings:
        raise ValueError("survey JSON contains no SurveyPingsSuper trace")
    ping = super_pings[0]
    signal = np.asarray(ping.get("SignalData"), dtype=float)
    sample_rate_hz = float(ping.get("SampleFrequency", 0.0))
    start_us = float(survey.get("SampleRangeAnalysisStartUSecs"))
    if (
        signal.ndim != 1
        or signal.size < 32
        or np.any(~np.isfinite(signal))
        or not np.isfinite(sample_rate_hz)
        or sample_rate_hz <= 0.0
    ):
        raise ValueError("invalid SurveyPingsSuper signal or sample rate")

    edge_count = max(8, signal.size // 20)
    baseline = float(
        np.median(np.concatenate((signal[:edge_count], signal[-edge_count:])))
    )
    signal = signal - baseline
    normalization = float(np.max(np.abs(signal)))
    if normalization <= 0.0:
        raise ValueError("survey trace has zero ADC range")
    signal /= normalization

    survey_result = survey.get("SurveyResult") or {}
    time_keys = {
        "water_pp": "PlateBaseSampleTimeUSecs",
        "pp_fluid": "WellBaseSampleTimeUSecs",
        "fluid_top": "FluidTopSampleTimeUSecs",
    }
    interface_times_us = {
        name: float(survey_result[key])
        for name, key in time_keys.items()
        if survey_result.get(key) is not None
    }
    if "water_pp" not in interface_times_us:
        raise ValueError("survey JSON has no PlateBaseSampleTimeUSecs")
    absolute_time_us = start_us + np.arange(signal.size) / sample_rate_hz * 1e6
    relative_time_us = absolute_time_us - interface_times_us["water_pp"]
    relative_interfaces_us = {
        name: value - interface_times_us["water_pp"]
        for name, value in interface_times_us.items()
    }

    warnings = [
        "ADC independently normalized; no absolute amplitude calibration",
    ]
    if survey.get("FluidMaterial") is None:
        warnings.append("FluidMaterial missing; no fluid property was inferred")
    warnings.append(
        "temperature and independently measured fill height unavailable"
    )
    return relative_time_us, signal, relative_interfaces_us, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the broadband monostatic echo and optionally overlay a "
            "normalized survey trace."
        )
    )
    parser.add_argument(
        "--dmso-percent",
        type=float,
        default=DEFAULT_DMSO_VOLUME_PERCENT,
        help="DMSO concentration in volume percent (default: 80)",
    )
    parser.add_argument(
        "--dmso-height-mm",
        type=float,
        default=DEFAULT_DMSO_HEIGHT_MM,
        help="DMSO fill height in mm (default: 4.22)",
    )
    parser.add_argument(
        "--temperature-c",
        type=float,
        default=DEFAULT_TEMPERATURE_C,
        help="temperature in degree Celsius (default: 22)",
    )
    parser.add_argument(
        "--survey-json",
        type=Path,
        help=(
            "optional survey JSON; used only as independently normalized "
            "timing/pulse-shape overlay"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="directory for PNG, NPZ and JSON outputs",
    )
    parser.add_argument(
        "--output-stem",
        help="output filename stem without extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.dmso_percent <= 100.0:
        raise ValueError("--dmso-percent must lie between 0 and 100")
    if args.dmso_height_mm <= 0.0:
        raise ValueError("--dmso-height-mm must be > 0")

    dmso_fraction = args.dmso_percent / 100.0
    dmso_height_m = args.dmso_height_mm * 1e-3
    dmso_properties = dmso_water_properties(
        dmso_fraction,
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
        # A two-way 25.3 mm water path needs a wider FFT window than a
        # one-way focal-field plot. The 28.8 mm window retains the intended
        # aperture-edge rays over the active pulse band.
        grid=CartesianGrid(nx=384, ny=384, dx_m=85e-6),
        aperture=FocusedCircularAperture(
            diameter_m=TRANSDUCER_DIAMETER_M,
            focal_length_m=TRANSDUCER_FOCAL_LENGTH_M,
        ),
        incident_fluid=water,
        plate=ElasticPlate(polypropylene, PP_THICKNESS_M),
        transmitted_fluid=dmso,
        water_path_m=WATER_PATH_M,
        plate_radial_samples=2048,
    )
    validate_focused_grid_support(
        model,
        maximum_frequency_hz=GRID_VALIDATION_FREQUENCY_HZ,
        propagation_segments=(
            ("water pulse-echo round trip", water, 2.0 * WATER_PATH_M),
            (
                "liquid-layer cavity round trip",
                dmso,
                2.0 * dmso_height_m,
            ),
        ),
    )

    front_arrival_s = 2.0 * WATER_PATH_M / water.sound_speed_m_s
    pp_back_arrival_s = (
        front_arrival_s
        + 2.0
        * PP_THICKNESS_M
        / polypropylene.longitudinal_speed_m_s
    )
    air_arrival_s = (
        pp_back_arrival_s
        + 2.0 * dmso_height_m / dmso.sound_speed_m_s
    )
    cavity_round_trip_s = 2.0 * dmso_height_m / dmso.sound_speed_m_s
    drive_duration_s = 1.0 / DRIVE_FREQUENCY_HZ
    record_length_s = max(
        RECORD_LENGTH_S,
        air_arrival_s
        + 2.0 * drive_duration_s
        + cavity_round_trip_s,
    )

    time_s, drive = sine_burst(
        center_frequency_hz=DRIVE_FREQUENCY_HZ,
        cycles=1.0,
        sample_rate_hz=SAMPLE_RATE_HZ,
        record_length_s=record_length_s,
        start_time_s=0.0,
    )

    def certificate_round_trip_response(frequency: np.ndarray) -> np.ndarray:
        return asymmetric_gaussian_response(
            frequency,
            peak_frequency_hz=TRANSDUCER_PEAK_FREQUENCY_HZ,
            lower_frequency_6db_hz=TRANSDUCER_LOWER_FREQUENCY_6DB_HZ,
            upper_frequency_6db_hz=TRANSDUCER_UPPER_FREQUENCY_6DB_HZ,
        ) * smooth_dc_block_response(frequency)

    result = simulate_monostatic_pulse_echo(
        model,
        time_s,
        drive,
        fluid_layer_thickness_m=dmso_height_m,
        backing_fluid=air,
        round_trip_response=certificate_round_trip_response,
        relative_spectrum_threshold=2.0e-3,
        minimum_frequency_hz=0.0,
        maximum_frequency_hz=GRID_VALIDATION_FREQUENCY_HZ,
        fluid_cavity_echo_count=1,
    )

    front_peak_s = peak_time(
        result.time_s,
        result.plate_front_signal,
        front_arrival_s,
        0.25e-6,
    )
    pp_back_peak_s = peak_time(
        result.time_s,
        result.plate_front_signal,
        pp_back_arrival_s,
        0.20e-6,
    )
    air_peak_s = peak_time(
        result.time_s,
        result.backing_signal,
        air_arrival_s,
        0.60e-6,
    )

    normalization = max(float(np.max(np.abs(result.received_signal))), 1e-30)
    received = result.received_signal / normalization
    plate_response = result.plate_front_signal / normalization
    backing = result.backing_signal / normalization
    envelope = analytic_envelope(received)
    plate_envelope = analytic_envelope(plate_response)
    backing_envelope = analytic_envelope(backing)
    relative_time_us = (result.time_s - front_arrival_s) * 1e6
    pp_back_relative_us = (pp_back_arrival_s - front_arrival_s) * 1e6
    air_relative_us = (air_arrival_s - front_arrival_s) * 1e6

    survey_overlay = None
    survey_warnings: list[str] = []
    if args.survey_json is not None:
        (
            survey_time_us,
            survey_signal,
            survey_interfaces_us,
            survey_warnings,
        ) = load_normalized_survey(args.survey_json)
        survey_overlay = (
            survey_time_us,
            survey_signal,
            survey_interfaces_us,
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    percent_tag = f"{args.dmso_percent:g}".replace(".", "p")
    output_stem = args.output_stem or f"pulse_echo_{percent_tag}pct_dmso"

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 9.2),
        constrained_layout=True,
    )
    axes[0].plot(
        relative_time_us,
        received,
        color="#355f8a",
        linewidth=0.9,
        label="Simulation, normalized",
    )
    axes[0].plot(
        relative_time_us,
        envelope,
        color="#d1495b",
        linewidth=1.0,
        alpha=0.85,
        label="Simulation envelope",
    )
    axes[0].plot(
        relative_time_us,
        -envelope,
        color="#d1495b",
        linewidth=1.0,
        alpha=0.85,
    )
    if survey_overlay is not None:
        survey_time_us, survey_signal, _ = survey_overlay
        axes[0].plot(
            survey_time_us,
            survey_signal,
            color="0.25",
            linewidth=0.75,
            alpha=0.72,
            label="Measurement, ADC normalized separately",
        )
    axes[0].axvline(
        0.0,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label="water–PP",
    )
    axes[0].axvline(
        pp_back_relative_us,
        color="#d1495b",
        linestyle=":",
        linewidth=1.2,
        label="PP–DMSO",
    )
    axes[0].set(
        xlim=(-0.5, 1.5),
        ylim=(-1.2, 1.2),
        xlabel="Time relative to water–PP [µs]",
        ylabel="Signal [normalized separately]",
        title=(
            "Close view of the two PP interfaces "
            f"({args.dmso_percent:g} vol.% DMSO)"
        ),
    )
    axes[0].legend(loc="upper right", ncols=2)

    axes[1].plot(
        relative_time_us,
        received,
        color="#355f8a",
        linewidth=0.75,
        label="received signal",
    )
    axes[1].plot(
        relative_time_us,
        envelope,
        color="#d1495b",
        linewidth=1.1,
        label="envelope",
    )
    axes[1].plot(
        relative_time_us,
        -envelope,
        color="#d1495b",
        linewidth=1.1,
    )
    interface_markers = [
        (0.0, "--", "0.45", "water–PP"),
        (pp_back_relative_us, ":", "#d1495b", "PP–DMSO"),
        (air_relative_us, "-.", "#2a9d8f", "DMSO–air"),
    ]
    for arrival_us, linestyle, color, label in interface_markers:
        axes[1].axvline(
            arrival_us,
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            label=label,
        )
    full_view_end_us = max(air_relative_us + 2.0, 8.0)
    axes[1].set(
        xlim=(-0.5, full_view_end_us),
        xlabel="Time relative to water–PP [µs]",
        ylabel="Received signal [normalized]",
        title="Complete pulse-echo response with three interfaces",
    )
    axes[1].legend(loc="upper right", ncols=2)

    axes[2].plot(
        relative_time_us,
        plate_envelope,
        label="complete PP plate response",
        color="#355f8a",
    )
    axes[2].plot(
        relative_time_us,
        backing_envelope,
        label="first retained DMSO–air return",
        color="#d1495b",
    )
    for arrival_us, linestyle, color, _ in interface_markers:
        axes[2].axvline(
            arrival_us,
            color=color,
            linestyle=linestyle,
            linewidth=1.0,
        )
    axes[2].set(
        xlim=(-0.5, full_view_end_us),
        xlabel="Time relative to water–PP [µs]",
        ylabel="Envelope [normalized]",
        title="Physical model components",
    )
    axes[2].legend(loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.23)

    figure_path = args.output_directory / f"{output_stem}.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    np.savez_compressed(
        args.output_directory / f"{output_stem}.npz",
        time_s=result.time_s,
        time_relative_to_water_pp_s=result.time_s - front_arrival_s,
        drive_signal=result.drive_signal,
        received_signal=result.received_signal,
        plate_response_signal=result.plate_front_signal,
        backing_signal=result.backing_signal,
        frequency_hz=result.frequency_hz,
        received_spectrum=result.received_spectrum,
        electroacoustic_response=result.electroacoustic_response,
        physical_round_trip_phasor=result.physical_round_trip_phasor,
        applied_round_trip_transfer=result.applied_round_trip_transfer,
        round_trip_transfer=result.round_trip_transfer,
        simulated_bin_mask=result.simulated_bin_mask,
        fluid_cavity_echo_count=result.fluid_cavity_echo_count,
    )

    front_peak = float(np.max(np.abs(result.plate_front_signal)))
    backing_peak = float(np.max(np.abs(result.backing_signal)))
    summary = {
        "excitation": {
            "center_frequency_hz": DRIVE_FREQUENCY_HZ,
            "cycles": 1.0,
            "start_phase_deg": 0.0,
            "time_zero": "positive-going zero crossing",
        },
        "geometry": {
            "water_path_to_pp_front_m": WATER_PATH_M,
            "pp_thickness_m": PP_THICKNESS_M,
            "dmso_height_m": dmso_height_m,
            "backing": "air",
        },
        "dmso": {
            "volume_fraction": dmso_fraction,
            "temperature_c": args.temperature_c,
            "density_kg_m3": dmso.density_kg_m3,
            "sound_speed_m_s": dmso.sound_speed_m_s,
        },
        "incident_water": {
            "temperature_c": args.temperature_c,
            "density_kg_m3": water.density_kg_m3,
            "sound_speed_m_s": water.sound_speed_m_s,
        },
        "numerics": {
            "grid_size": model.grid.nx,
            "grid_spacing_m": model.grid.dx_m,
            "grid_window_m": model.grid.extent_x_m,
            "grid_validation_frequency_hz": GRID_VALIDATION_FREQUENCY_HZ,
            "record_length_s": record_length_s,
            "fluid_cavity_echo_count": result.fluid_cavity_echo_count,
        },
        "transducer_certificate": {
            "model": "Doppler I2-10P13F25-H",
            "diameter_m": model.aperture.diameter_m,
            "focal_length_m": model.aperture.focal_length_m,
            "center_frequency_hz": TRANSDUCER_CENTER_FREQUENCY_HZ,
            "peak_frequency_hz": TRANSDUCER_PEAK_FREQUENCY_HZ,
            "fractional_bandwidth_6db_round_trip": (
                TRANSDUCER_FRACTIONAL_BANDWIDTH_6DB_ROUND_TRIP
            ),
            "lower_frequency_6db_hz": TRANSDUCER_LOWER_FREQUENCY_6DB_HZ,
            "upper_frequency_6db_hz": TRANSDUCER_UPPER_FREQUENCY_6DB_HZ,
            "pulse_duration_6db_s": TRANSDUCER_PULSE_DURATION_6DB_S,
            "response_application": "pulse-echo response applied once",
        },
        "timing": {
            "geometric_water_pp_arrival_s": front_arrival_s,
            "geometric_pp_dmso_arrival_s": pp_back_arrival_s,
            "geometric_dmso_air_arrival_s": air_arrival_s,
            "simulated_water_pp_peak_s": front_peak_s,
            "simulated_pp_dmso_local_peak_s": pp_back_peak_s,
            "simulated_dmso_air_component_peak_s": air_peak_s,
        },
        "relative_amplitudes": {
            "plate_component_peak": front_peak,
            "dmso_air_component_peak": backing_peak,
            "dmso_air_to_plate_peak_ratio": (
                backing_peak / front_peak if front_peak > 0.0 else None
            ),
            "calibrated_absolute_units": False,
        },
        "survey_overlay": {
            "used": survey_overlay is not None,
            "uses_absolute_adc_amplitude": False,
            "used_to_fit_fluid_properties": False,
            "warnings": survey_warnings,
        },
        "assumptions": [
            (
                "certificate magnitude represented by an asymmetric zero-phase "
                "Gaussian; certificate phase was unavailable"
            ),
            "same reciprocal aperture used for transmit and receive",
            "only the first robust DMSO-air surface order is retained",
            "zero material attenuation because measured values were not supplied",
            "no absolute ADC-to-pressure or receive-sensitivity calibration",
        ],
    }
    with (args.output_directory / f"{output_stem}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Geometric water–PP arrival: {front_arrival_s * 1e6:.3f} µs")
    print(f"Geometric PP–DMSO arrival: {pp_back_arrival_s * 1e6:.3f} µs")
    print(f"Geometric DMSO–air arrival: {air_arrival_s * 1e6:.3f} µs")
    if survey_overlay is not None:
        _, _, survey_interfaces_us = survey_overlay
        print(
            "Survey times relative to water–PP:",
            ", ".join(
                f"{name}={value:.3f} µs"
                for name, value in survey_interfaces_us.items()
            ),
        )
        for warning in survey_warnings:
            print(f"Survey note: {warning}")
    print(f"Plot: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
