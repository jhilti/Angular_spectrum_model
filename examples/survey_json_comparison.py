"""Compare a survey ADC trace with a pulse-echo simulation.

The survey's interface times are used for time-of-flight-equivalent geometry.
Absolute ADC amplitudes, the stored fluid label and the stored fluid height
are not treated as calibrated measurements.
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
    interpret_survey_geometry,
    load_survey_pulse_echo,
    simulate_monostatic_pulse_echo,
    sine_burst,
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
TRANSDUCER_FOCAL_LENGTH_M = 25.40e-3
TRANSDUCER_DIAMETER_M = 13.0e-3

WATER_SOUND_SPEED_M_S = 1488.4
WATER_DENSITY_KG_M3 = 997.77
PP_LONGITUDINAL_SPEED_M_S = 2732.0
PP_DENSITY_KG_M3 = 900.0
PP_POISSON_RATIO = 0.42

SAMPLE_RATE_HZ = 160.0e6
RECORD_LENGTH_S = 42.0e-6
DEFAULT_OUTPUT_DIRECTORY = Path("results/private")


def analytic_envelope(signal: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(signal)
    multiplier = np.zeros(signal.size)
    multiplier[0] = 1.0
    if signal.size % 2 == 0:
        multiplier[1 : signal.size // 2] = 2.0
        multiplier[signal.size // 2] = 1.0
    else:
        multiplier[1 : (signal.size + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier))


def echo_match(
    measured_time_s: np.ndarray,
    measured_signal: np.ndarray,
    simulated_time_s: np.ndarray,
    simulated_signal: np.ndarray,
    *,
    measured_arrival_s: float,
    simulated_arrival_s: float,
    half_window_s: float = 0.30e-6,
    maximum_lag_s: float = 0.15e-6,
) -> dict[str, float | np.ndarray]:
    """Find the best local timing shift after independent normalization."""

    measured_local = measured_time_s - measured_arrival_s
    simulated_local = simulated_time_s - simulated_arrival_s
    mask = np.abs(measured_local) <= half_window_s
    local_time = measured_local[mask]
    measured = measured_signal[mask]
    measured = measured - float(np.mean(measured))

    sample_interval_s = float(np.median(np.diff(local_time)))
    lags = np.arange(
        -maximum_lag_s,
        maximum_lag_s + 0.5 * sample_interval_s,
        sample_interval_s,
    )
    best_correlation = -np.inf
    best_lag_s = 0.0
    best_simulated = np.zeros_like(measured)
    best_rms = np.inf
    for lag_s in lags:
        candidate = np.interp(
            local_time - lag_s,
            simulated_local,
            simulated_signal,
            left=0.0,
            right=0.0,
        )
        candidate = candidate - float(np.mean(candidate))
        denominator = float(np.linalg.norm(measured) * np.linalg.norm(candidate))
        if denominator <= 0.0:
            continue
        correlation = float(np.dot(measured, candidate) / denominator)
        if correlation > best_correlation:
            measured_unit = measured / max(float(np.linalg.norm(measured)), 1e-30)
            candidate_unit = candidate / max(float(np.linalg.norm(candidate)), 1e-30)
            best_correlation = correlation
            best_lag_s = float(lag_s)
            best_simulated = candidate
            best_rms = float(
                np.sqrt(np.mean((measured_unit - candidate_unit) ** 2))
            )

    measured_scale = max(float(np.max(np.abs(measured))), 1e-30)
    simulated_scale = max(float(np.max(np.abs(best_simulated))), 1e-30)
    return {
        "local_time_s": local_time,
        "measured_normalized": measured / measured_scale,
        "simulated_normalized": best_simulated / simulated_scale,
        "best_lag_s": best_lag_s,
        "correlation": best_correlation,
        "normalized_rms": best_rms,
    }


def window_envelope_peak(
    time_s: np.ndarray,
    envelope: np.ndarray,
    centre_s: float,
    half_window_s: float,
) -> float:
    mask = np.abs(time_s - centre_s) <= half_window_s
    return float(np.max(envelope[mask]))


def gated_spectrum(
    time_s: np.ndarray,
    signal: np.ndarray,
    *,
    arrival_s: float,
    start_relative_s: float = -0.22e-6,
    end_relative_s: float = 0.38e-6,
) -> dict[str, float | np.ndarray]:
    """Return a locally normalized Hann-gated echo spectrum."""

    relative_time_s = time_s - arrival_s
    mask = (
        (relative_time_s >= start_relative_s)
        & (relative_time_s <= end_relative_s)
    )
    gated = signal[mask]
    gated = (gated - float(np.mean(gated))) * np.hanning(gated.size)
    sample_interval_s = float(np.median(np.diff(time_s)))
    fft_size = max(65536, 2 ** int(np.ceil(np.log2(gated.size * 32))))
    frequency_hz = np.fft.rfftfreq(fft_size, sample_interval_s)
    magnitude = np.abs(np.fft.rfft(gated, fft_size))
    analysis_band = (frequency_hz >= 2.0e6) & (frequency_hz <= 20.0e6)
    peak_index = np.flatnonzero(analysis_band)[
        np.argmax(magnitude[analysis_band])
    ]
    peak_magnitude = max(float(magnitude[peak_index]), 1e-30)
    normalized_magnitude = magnitude / peak_magnitude
    half_amplitude = 0.5
    left_crossings = np.where(
        normalized_magnitude[:peak_index] <= half_amplitude
    )[0]
    right_crossings = np.where(
        normalized_magnitude[peak_index:] <= half_amplitude
    )[0]
    lower_6db_hz = (
        float(frequency_hz[left_crossings[-1]])
        if left_crossings.size
        else float("nan")
    )
    upper_6db_hz = (
        float(frequency_hz[peak_index + right_crossings[0]])
        if right_crossings.size
        else float("nan")
    )
    peak_frequency_hz = float(frequency_hz[peak_index])
    return {
        "frequency_hz": frequency_hz,
        "magnitude_db": 20.0 * np.log10(
            np.maximum(normalized_magnitude, 1e-8)
        ),
        "peak_frequency_hz": peak_frequency_hz,
        "lower_frequency_6db_hz": lower_6db_hz,
        "upper_frequency_6db_hz": upper_6db_hz,
        "fractional_bandwidth_6db": (
            (upper_6db_hz - lower_6db_hz) / peak_frequency_hz
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive TOF-equivalent geometry from a survey JSON and compare its "
            "normalized ADC trace with the angular-spectrum simulation."
        )
    )
    parser.add_argument("survey_json", type=Path)
    parser.add_argument(
        "--dmso-percent",
        type=float,
        default=73.0,
        help="assumed DMSO volume percent; metadata does not contain it",
    )
    parser.add_argument(
        "--temperature-c",
        type=float,
        default=22.0,
        help="assumed temperature; metadata does not contain it",
    )
    parser.add_argument(
        "--known-water-path-mm",
        type=float,
        help=(
            "optional independently measured water path; used for focusing "
            "instead of the stored/TOF-derived distance"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--output-stem",
        default="survey_json_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.dmso_percent <= 100.0:
        raise ValueError("--dmso-percent must lie between 0 and 100")
    if args.known_water_path_mm is not None and args.known_water_path_mm <= 0.0:
        raise ValueError("--known-water-path-mm must be > 0")

    survey = load_survey_pulse_echo(args.survey_json)
    dmso_properties = dmso_water_properties(
        args.dmso_percent / 100.0,
        basis="volume",
        temperature_c=args.temperature_c,
    )
    geometry = interpret_survey_geometry(
        survey,
        incident_sound_speed_m_s=WATER_SOUND_SPEED_M_S,
        plate_longitudinal_speed_m_s=PP_LONGITUDINAL_SPEED_M_S,
        fluid_sound_speed_m_s=dmso_properties.sound_speed_m_s,
    )

    if args.known_water_path_mm is not None:
        simulation_water_path_m = args.known_water_path_mm * 1e-3
        water_path_source = "independent command-line value"
    elif geometry.stored_probe_to_plate_m is not None:
        simulation_water_path_m = geometry.stored_probe_to_plate_m
        water_path_source = "stored SurveyResult value"
    else:
        simulation_water_path_m = geometry.tof_probe_to_plate_m
        water_path_source = "TOF equivalent with assumed water sound speed"
    simulation_plate_thickness_m = (
        geometry.stored_plate_thickness_m
        if geometry.stored_plate_thickness_m is not None
        else geometry.tof_plate_thickness_m
    )
    simulation_fluid_height_m = geometry.tof_fluid_height_m

    water = Fluid(
        "water",
        WATER_DENSITY_KG_M3,
        WATER_SOUND_SPEED_M_S,
    )
    dmso = Fluid(
        f"assumed_{args.dmso_percent:g}volpct_DMSO",
        dmso_properties.density_kg_m3,
        dmso_properties.sound_speed_m_s,
    )
    air = Fluid("air", 1.196, 344.0)
    polypropylene = ElasticSolid.from_longitudinal_speed_and_poisson(
        name="polypropylene",
        density_kg_m3=PP_DENSITY_KG_M3,
        longitudinal_speed_m_s=PP_LONGITUDINAL_SPEED_M_S,
        poisson_ratio=PP_POISSON_RATIO,
    )
    model = AngularSpectrumModel(
        grid=CartesianGrid(nx=320, ny=320, dx_m=50e-6),
        aperture=FocusedCircularAperture(
            diameter_m=TRANSDUCER_DIAMETER_M,
            focal_length_m=TRANSDUCER_FOCAL_LENGTH_M,
        ),
        incident_fluid=water,
        plate=ElasticPlate(polypropylene, simulation_plate_thickness_m),
        transmitted_fluid=dmso,
        water_path_m=simulation_water_path_m,
        plate_radial_samples=1536,
    )
    time_s, drive = sine_burst(
        center_frequency_hz=DRIVE_FREQUENCY_HZ,
        cycles=1.0,
        sample_rate_hz=SAMPLE_RATE_HZ,
        record_length_s=RECORD_LENGTH_S,
        start_time_s=0.0,
    )

    def certificate_round_trip_response(frequency: np.ndarray) -> np.ndarray:
        return asymmetric_gaussian_response(
            frequency,
            peak_frequency_hz=TRANSDUCER_PEAK_FREQUENCY_HZ,
            lower_frequency_6db_hz=TRANSDUCER_LOWER_FREQUENCY_6DB_HZ,
            upper_frequency_6db_hz=TRANSDUCER_UPPER_FREQUENCY_6DB_HZ,
        )

    simulated = simulate_monostatic_pulse_echo(
        model,
        time_s,
        drive,
        fluid_layer_thickness_m=simulation_fluid_height_m,
        backing_fluid=air,
        round_trip_response=certificate_round_trip_response,
        relative_spectrum_threshold=2.0e-3,
        minimum_frequency_hz=2.5e6,
        maximum_frequency_hz=20.0e6,
    )

    simulated_water_pp_s = (
        2.0 * simulation_water_path_m / WATER_SOUND_SPEED_M_S
    )
    simulated_pp_fluid_s = (
        simulated_water_pp_s
        + 2.0
        * simulation_plate_thickness_m
        / PP_LONGITUDINAL_SPEED_M_S
    )
    simulated_fluid_top_s = (
        simulated_pp_fluid_s
        + 2.0
        * simulation_fluid_height_m
        / dmso_properties.sound_speed_m_s
    )
    measured_arrivals = {
        "water_pp": survey.water_pp_time_s,
        "pp_fluid": survey.pp_fluid_time_s,
        "fluid_top": survey.fluid_top_time_s,
    }
    simulated_arrivals = {
        "water_pp": simulated_water_pp_s,
        "pp_fluid": simulated_pp_fluid_s,
        "fluid_top": simulated_fluid_top_s,
    }

    simulated_scale = max(
        float(np.max(np.abs(simulated.received_signal))),
        1e-30,
    )
    simulated_normalized = simulated.received_signal / simulated_scale
    simulated_relative_s = simulated.time_s - simulated_water_pp_s
    measured_relative_s = survey.relative_time_s

    matches = {
        name: echo_match(
            survey.time_s,
            survey.normalized_signal,
            simulated.time_s,
            simulated_normalized,
            measured_arrival_s=measured_arrivals[name],
            simulated_arrival_s=simulated_arrivals[name],
        )
        for name in measured_arrivals
    }

    measured_envelope = analytic_envelope(survey.normalized_signal)
    simulated_envelope = analytic_envelope(simulated_normalized)
    measured_envelope /= max(float(np.max(measured_envelope)), 1e-30)
    simulated_envelope /= max(float(np.max(simulated_envelope)), 1e-30)
    measured_peak_ratios = {
        name: window_envelope_peak(
            survey.time_s,
            measured_envelope,
            arrival_s,
            0.30e-6,
        )
        for name, arrival_s in measured_arrivals.items()
    }
    simulated_peak_ratios = {
        name: window_envelope_peak(
            simulated.time_s,
            simulated_envelope,
            arrival_s,
            0.30e-6,
        )
        for name, arrival_s in simulated_arrivals.items()
    }
    spectra = {
        name: {
            "measured": gated_spectrum(
                survey.time_s,
                survey.normalized_signal,
                arrival_s=measured_arrivals[name],
            ),
            "simulated": gated_spectrum(
                simulated.time_s,
                simulated_normalized,
                arrival_s=simulated_arrivals[name],
            ),
        }
        for name in measured_arrivals
    }

    measured_relative_arrivals_us = {
        name: (arrival_s - survey.water_pp_time_s) * 1e6
        for name, arrival_s in measured_arrivals.items()
    }
    stored_height_surface_us = None
    if geometry.stored_height_fluid_delay_s is not None:
        stored_height_surface_us = (
            survey.pp_round_trip_time_s
            + geometry.stored_height_fluid_delay_s
        ) * 1e6

    fig = plt.figure(figsize=(13.0, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.25, 1.0))
    full_axis = fig.add_subplot(grid[0, :])
    full_axis.plot(
        measured_relative_s * 1e6,
        survey.normalized_signal,
        color="0.25",
        linewidth=0.8,
        label="Survey ADC, normiert",
    )
    full_axis.plot(
        simulated_relative_s * 1e6,
        simulated_normalized,
        color="#355f8a",
        linewidth=0.85,
        alpha=0.9,
        label="ASM-Simulation, normiert",
    )
    full_axis.plot(
        measured_relative_s * 1e6,
        measured_envelope,
        color="0.45",
        linewidth=1.2,
        alpha=0.65,
        label="Survey-Hüllkurve",
    )
    full_axis.plot(
        simulated_relative_s * 1e6,
        simulated_envelope,
        color="#d1495b",
        linewidth=1.2,
        alpha=0.78,
        label="Simulationshüllkurve",
    )
    marker_styles = {
        "water_pp": ("--", "0.45", "Wasser–PP"),
        "pp_fluid": (":", "#d1495b", "PP–Flüssigkeit"),
        "fluid_top": ("-.", "#2a9d8f", "Flüssigkeitsoberfläche"),
    }
    for name, relative_us in measured_relative_arrivals_us.items():
        linestyle, color, label = marker_styles[name]
        full_axis.axvline(
            relative_us,
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            label=label,
        )
    if stored_height_surface_us is not None:
        full_axis.axvline(
            stored_height_surface_us,
            color="#f4a261",
            linestyle=(0, (2, 2)),
            linewidth=1.4,
            label=(
                f"Oberfläche mit gespeicherten "
                f"{geometry.stored_fluid_height_m * 1e3:.3f} mm"
            ),
        )
    full_axis.set(
        xlim=(-0.35, 2.75),
        ylim=(-1.12, 1.18),
        xlabel="Zeit relativ zu Wasser–PP [µs]",
        ylabel="Signal/Hüllkurve [separat normiert]",
        title=(
            "Survey und TOF-angepasste Simulation "
            f"(Hypothese: {args.dmso_percent:g} Vol.-% DMSO, "
            f"{args.temperature_c:g} °C)"
        ),
    )
    full_axis.grid(alpha=0.22)
    full_axis.legend(loc="upper right", ncols=3, fontsize=9)

    display_names = {
        "water_pp": "Wasser–PP",
        "pp_fluid": "PP–Flüssigkeit",
        "fluid_top": "Flüssigkeitsoberfläche",
    }
    for axis, name in zip(
        (fig.add_subplot(grid[1, index]) for index in range(3)),
        ("water_pp", "pp_fluid", "fluid_top"),
    ):
        match = matches[name]
        local_time_us = np.asarray(match["local_time_s"]) * 1e6
        axis.plot(
            local_time_us,
            np.asarray(match["measured_normalized"]),
            color="0.25",
            linewidth=1.0,
            label="Survey",
        )
        axis.plot(
            local_time_us,
            np.asarray(match["simulated_normalized"]),
            color="#355f8a",
            linewidth=1.0,
            label="Simulation, bestverschoben",
        )
        axis.axvline(0.0, color="0.55", linestyle="--", linewidth=0.9)
        axis.set(
            xlim=(-0.25, 0.30),
            ylim=(-1.1, 1.1),
            xlabel="lokale Zeit [µs]",
            ylabel="lokal normiert",
            title=(
                f"{display_names[name]}\n"
                f"Korrelation {float(match['correlation']):.3f}, "
                f"Shift {float(match['best_lag_s']) * 1e9:+.0f} ns"
            ),
        )
        axis.grid(alpha=0.22)
        axis.legend(loc="lower right", fontsize=8)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_directory / f"{args.output_stem}.png"
    spectrum_figure_path = (
        args.output_directory / f"{args.output_stem}_spectrum.png"
    )
    summary_path = args.output_directory / f"{args.output_stem}.json"
    data_path = args.output_directory / f"{args.output_stem}.npz"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)

    spectrum_figure, spectrum_axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 4.2),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for axis, name in zip(
        spectrum_axes,
        ("water_pp", "pp_fluid", "fluid_top"),
    ):
        measured_spectrum = spectra[name]["measured"]
        simulated_spectrum = spectra[name]["simulated"]
        axis.plot(
            np.asarray(measured_spectrum["frequency_hz"]) * 1e-6,
            np.asarray(measured_spectrum["magnitude_db"]),
            color="0.25",
            label="Survey",
        )
        axis.plot(
            np.asarray(simulated_spectrum["frequency_hz"]) * 1e-6,
            np.asarray(simulated_spectrum["magnitude_db"]),
            color="#355f8a",
            label="Simulation",
        )
        axis.axhline(-6.0, color="0.6", linestyle="--", linewidth=0.9)
        axis.set(
            xlim=(2.0, 18.0),
            ylim=(-40.0, 2.0),
            xlabel="Frequenz [MHz]",
            ylabel="lokales Spektrum [dB]",
            title=(
                f"{display_names[name]}\n"
                f"Peak Messung "
                f"{float(measured_spectrum['peak_frequency_hz']) / 1e6:.2f} MHz, "
                f"Simulation "
                f"{float(simulated_spectrum['peak_frequency_hz']) / 1e6:.2f} MHz"
            ),
        )
        axis.grid(alpha=0.22)
        axis.legend(loc="lower left")
    spectrum_figure.savefig(spectrum_figure_path, dpi=190)
    plt.close(spectrum_figure)

    known_water_path_trigger_delay_s = None
    if args.known_water_path_mm is not None:
        known_water_path_trigger_delay_s = (
            survey.water_pp_time_s
            - 2.0
            * args.known_water_path_mm
            * 1e-3
            / WATER_SOUND_SPEED_M_S
        )
    summary = {
        "survey": {
            "date_time": survey.date_time,
            "fluid_material": survey.fluid_material,
            "sample_rate_hz": survey.sample_rate_hz,
            "absolute_adc_calibrated": False,
            "interface_times_s": measured_arrivals,
        },
        "hypothesis": {
            "fluid": f"{args.dmso_percent:g} vol.% DMSO",
            "temperature_c": args.temperature_c,
            "sound_speed_m_s": dmso_properties.sound_speed_m_s,
            "density_kg_m3": dmso_properties.density_kg_m3,
            "metadata_supports_hypothesis": False,
        },
        "geometry": {
            "stored": {
                "probe_to_plate_m": geometry.stored_probe_to_plate_m,
                "plate_thickness_m": geometry.stored_plate_thickness_m,
                "fluid_height_m": geometry.stored_fluid_height_m,
            },
            "implied_speeds_m_s": {
                "incident": geometry.implied_incident_speed_m_s,
                "plate": geometry.implied_plate_speed_m_s,
                "fluid": geometry.implied_fluid_speed_m_s,
            },
            "tof_equivalent_for_hypothesis": {
                "probe_to_plate_m": geometry.tof_probe_to_plate_m,
                "plate_thickness_m": geometry.tof_plate_thickness_m,
                "fluid_height_m": geometry.tof_fluid_height_m,
            },
            "simulation": {
                "water_path_m": simulation_water_path_m,
                "water_path_source": water_path_source,
                "plate_thickness_m": simulation_plate_thickness_m,
                "fluid_height_m": simulation_fluid_height_m,
            },
            "stored_height_timing_error_s_for_hypothesis": (
                geometry.stored_height_timing_error_s
            ),
            "known_water_path_trigger_delay_s": (
                known_water_path_trigger_delay_s
            ),
        },
        "comparison": {
            "signals_normalized_independently": True,
            "echo_shape": {
                name: {
                    "best_lag_s": float(match["best_lag_s"]),
                    "correlation": float(match["correlation"]),
                    "normalized_rms": float(match["normalized_rms"]),
                }
                for name, match in matches.items()
            },
            "measured_envelope_peak_ratios": measured_peak_ratios,
            "simulated_envelope_peak_ratios": simulated_peak_ratios,
            "echo_spectra": {
                name: {
                    source: {
                        key: float(value)
                        for key, value in spectrum.items()
                        if key not in {"frequency_hz", "magnitude_db"}
                    }
                    for source, spectrum in pair.items()
                }
                for name, pair in spectra.items()
            },
        },
        "limitations": [
            "FluidMaterial is missing from the survey JSON",
            "temperature is not stored",
            "stored distances are TOF-derived, not independent measurements",
            "absolute ADC-to-pressure calibration is unavailable",
            "certificate phase response and analogue electronics are unavailable",
            "PP and fluid attenuation are not calibrated",
        ],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    np.savez_compressed(
        data_path,
        measured_time_relative_s=measured_relative_s,
        measured_signal_normalized=survey.normalized_signal,
        simulated_time_relative_s=simulated_relative_s,
        simulated_signal_normalized=simulated_normalized,
        measured_envelope_normalized=measured_envelope,
        simulated_envelope_normalized=simulated_envelope,
    )

    print(
        "Stored distances [mm]:",
        f"water={geometry.stored_probe_to_plate_m * 1e3:.6f},",
        f"PP={geometry.stored_plate_thickness_m * 1e3:.6f},",
        f"fluid={geometry.stored_fluid_height_m * 1e3:.6f}",
    )
    print(
        "Implied stored sound speeds [m/s]:",
        f"water={geometry.implied_incident_speed_m_s:.3f},",
        f"PP={geometry.implied_plate_speed_m_s:.3f},",
        f"fluid={geometry.implied_fluid_speed_m_s:.3f}",
    )
    print(
        "TOF-equivalent distances for hypothesis [mm]:",
        f"water={geometry.tof_probe_to_plate_m * 1e3:.6f},",
        f"PP={geometry.tof_plate_thickness_m * 1e3:.6f},",
        f"fluid={geometry.tof_fluid_height_m * 1e3:.6f}",
    )
    if geometry.stored_height_timing_error_s is not None:
        print(
            "Stored-height surface timing error for hypothesis:",
            f"{geometry.stored_height_timing_error_s * 1e6:+.6f} µs",
        )
    if known_water_path_trigger_delay_s is not None:
        print(
            "Fixed delay relative to known water path:",
            f"{known_water_path_trigger_delay_s * 1e6:+.6f} µs",
        )
    for name, match in matches.items():
        print(
            f"{name}: correlation={float(match['correlation']):.4f}, "
            f"best shift={float(match['best_lag_s']) * 1e9:+.1f} ns"
        )
    print(f"Plot: {figure_path.resolve()}")
    print(f"Spectrum plot: {spectrum_figure_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
