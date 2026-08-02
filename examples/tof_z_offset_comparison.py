"""Cross-check a probe-z survey sweep against the focused acoustic model.

The survey ADC values remain qualitative.  Unlike a single-file overlay, this
analysis preserves the relative ADC scale across the sweep and uses one common
plate thickness and liquid delay for the unchanged well.  Fluid composition
and temperature are explicit hypotheses because the JSON does not store them.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

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
    PulseEchoResult,
    SurveyPulseEcho,
    apply_reference_transfer,
    asymmetric_gaussian_response,
    dmso_water_properties,
    estimate_reference_transfer,
    load_survey_pulse_echo,
    simulate_monostatic_pulse_echo,
    sine_burst,
    smooth_dc_block_response,
    water_properties,
)
from angular_spectrum.app_model import (
    NUMERICAL_PRESETS,
    PP_DENSITY_KG_M3,
    PP_LONGITUDINAL_SPEED_M_S,
    PP_POISSON_RATIO,
    SimulationInputs,
    TRANSDUCER_LOWER_FREQUENCY_6DB_HZ,
    TRANSDUCER_PEAK_FREQUENCY_HZ,
    TRANSDUCER_UPPER_FREQUENCY_6DB_HZ,
    analytic_envelope,
    run_interactive_simulation,
)
from angular_spectrum.labware import get_labcyte_plate


DEFAULT_INPUT_DIRECTORY = Path("tof_z_offset_results")
DEFAULT_OUTPUT_DIRECTORY = Path("results/private/tof_z_offset")
OFFSET_PATTERN = re.compile(r"z_offset_([+-]?\d+(?:\.\d+)?)mm$")


@dataclass(frozen=True)
class SweepRecord:
    """One z position with its parsed trace and detector results."""

    offset_mm: float
    path: Path
    survey: SurveyPulseEcho
    plate_front_amplitude_adc: float
    plate_back_amplitude_adc: float
    surface_amplitude_adc: float
    probe_position_command_mm: float
    probe_position_actual_mm: float
    probe_position_abs_actual_mm: float
    bubbler_position_mm: float | None


@dataclass(frozen=True)
class PulseSweepSimulation:
    """Broadband monostatic results with one shared z=0 correction."""

    time_s: np.ndarray
    received_signals: tuple[np.ndarray, ...]
    plate_front_signals: tuple[np.ndarray, ...]
    backing_signals: tuple[np.ndarray, ...]
    water_arrivals_s: np.ndarray
    plate_back_arrivals_s: np.ndarray
    surface_arrivals_s: np.ndarray
    common_reference_applied: bool
    common_reference_error: str | None
    common_reference_offset_mm: float
    common_reference_supported_frequency_hz: tuple[float, float] | None
    common_reference_gate_relative_s: tuple[float, float] | None
    common_reference_regularization: float | None
    common_reference_maximum_correction_db: float | None
    sample_rate_hz: float
    grid_size: int
    grid_spacing_m: float
    radial_samples: int
    active_frequency_bin_counts: tuple[int, ...]


def _finite_float(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def offset_from_path(path: Path) -> float:
    """Extract the signed millimetre offset from one survey filename."""

    match = OFFSET_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"cannot read z offset from {path.name!r}")
    return float(match.group(1))


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def load_sweep(input_directory: Path) -> tuple[dict[str, Any], list[SweepRecord]]:
    """Load and validate a complete sweep without changing the raw files."""

    summary_path = input_directory / "tof_z_offset_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing sweep summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(summary.get("rows"), list):
        raise ValueError("sweep summary must contain a rows array")

    if any(not isinstance(row, dict) for row in summary["rows"]):
        raise ValueError("every sweep summary row must be an object")
    summary_offsets = [
        _finite_float(row.get("probe_z_offset_mm"), "summary z offset")
        for row in summary["rows"]
    ]
    if len(summary_offsets) != len(set(summary_offsets)):
        raise ValueError("sweep summary contains duplicate z offsets")
    row_offsets = set(summary_offsets)
    expected_well = str(summary.get("well", "")).strip()
    if not expected_well:
        raise ValueError("sweep summary must identify the well")
    paths = sorted(input_directory.glob("survey_*_z_offset_*mm.json"))
    if not paths:
        raise ValueError(f"no z-offset survey JSON files found in {input_directory}")

    records: list[SweepRecord] = []
    for path in paths:
        offset_mm = offset_from_path(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"survey root must be an object: {path.name}")
        result = document.get("SurveyResult")
        if not isinstance(result, dict):
            raise ValueError(f"survey has no SurveyResult: {path.name}")
        if str(document.get("WellName", "")).strip() != expected_well:
            raise ValueError(f"survey well does not match summary: {path.name}")
        survey = load_survey_pulse_echo(path)
        records.append(
            SweepRecord(
                offset_mm=offset_mm,
                path=path,
                survey=survey,
                plate_front_amplitude_adc=_finite_float(
                    result.get("PlateBaseSampleAmplitude"),
                    "plate-front amplitude",
                ),
                plate_back_amplitude_adc=_finite_float(
                    result.get("WellBaseSampleAmplitude"),
                    "plate-back amplitude",
                ),
                surface_amplitude_adc=_finite_float(
                    result.get("FluidTopSampleAmplitude"),
                    "surface amplitude",
                ),
                probe_position_command_mm=_finite_float(
                    document.get("ProbePositionCommand"),
                    "probe command",
                ),
                probe_position_actual_mm=_finite_float(
                    document.get("ProbePositionActual"),
                    "probe position",
                ),
                probe_position_abs_actual_mm=_finite_float(
                    document.get("ProbePositionAbsActual"),
                    "absolute probe position",
                ),
                bubbler_position_mm=_optional_finite_float(
                    document.get("BubblerPosition")
                ),
            )
        )

    records.sort(key=lambda record: record.offset_mm)
    if len(records) < 3:
        raise ValueError(
            "a focus cross-check requires at least three survey positions"
        )
    offsets = [record.offset_mm for record in records]
    if len(offsets) != len(set(offsets)):
        raise ValueError("survey filenames contain duplicate z offsets")
    zero_count = sum(
        np.isclose(offset, 0.0, rtol=0.0, atol=1.0e-9)
        for offset in offsets
    )
    if zero_count != 1:
        raise ValueError("the sweep must contain exactly one z=0 survey")
    if set(offsets) != row_offsets:
        raise ValueError(
            "summary z offsets do not match the available survey filenames"
        )

    first = records[0].survey
    required_first_metadata = {
        "plate type": first.plate_type_id,
        "probe frequency": first.probe_frequency_hz,
        "tone length": first.tone_length_cycles,
        "voltage setting": first.probe_voltage_setting_v,
        "sample rate": first.sample_rate_hz,
    }
    for name, value in required_first_metadata.items():
        if value is None:
            raise ValueError(f"missing constant {name} metadata")
    for record in records[1:]:
        survey = record.survey
        comparable = {
            "plate type": (first.plate_type_id, survey.plate_type_id),
            "probe frequency": (
                first.probe_frequency_hz,
                survey.probe_frequency_hz,
            ),
            "tone length": (
                first.tone_length_cycles,
                survey.tone_length_cycles,
            ),
            "voltage setting": (
                first.probe_voltage_setting_v,
                survey.probe_voltage_setting_v,
            ),
            "sample rate": (first.sample_rate_hz, survey.sample_rate_hz),
        }
        for name, (reference, candidate) in comparable.items():
            if reference is None or candidate is None:
                raise ValueError(f"missing constant {name} metadata")
            if isinstance(reference, str):
                matches = reference == candidate
            else:
                matches = bool(np.isclose(reference, candidate))
            if not matches:
                raise ValueError(f"{name} changes across the z sweep")

    if first.plate_type_id is None:
        raise ValueError("survey does not identify the plate")
    plate = get_labcyte_plate(first.plate_type_id)
    if plate.material != "polypropylene":
        raise ValueError(
            "the z-sweep example currently supports polypropylene plates only"
        )
    summary_plate = summary.get("plate_id")
    if summary_plate is not None and get_labcyte_plate(str(summary_plate)) != plate:
        raise ValueError("summary plate does not match the survey files")
    configured_frequency = summary.get("configured_probe_frequency_hz")
    if configured_frequency is not None and not np.isclose(
        _finite_float(configured_frequency, "configured probe frequency"),
        float(first.probe_frequency_hz),
    ):
        raise ValueError("summary probe frequency does not match survey files")
    for record in records:
        _baseline_corrected_trace(record)
    return summary, records


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return slope, intercept, R² and maximum residual for one trend."""

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    denominator = float(np.sum((y - float(np.mean(y))) ** 2))
    if denominator <= 0.0:
        raise ValueError("linear fit requires non-constant response values")
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
    }


def _normalized(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("cannot normalize an empty or zero response")
    return values / scale


def _simulate_pulse_sweep(
    inputs: SimulationInputs,
    water_paths_mm: np.ndarray,
    records: list[SweepRecord],
) -> PulseSweepSimulation:
    """Simulate every z position and apply one shared z=0 waveform correction."""

    preset = NUMERICAL_PRESETS[inputs.numerical_preset]
    water_data = water_properties(inputs.temperature_c)
    fluid_data = dmso_water_properties(
        inputs.dmso_volume_percent / 100.0,
        basis="volume",
        temperature_c=inputs.temperature_c,
    )
    water = Fluid("water", water_data.density_kg_m3, water_data.sound_speed_m_s)
    liquid = Fluid(
        "DMSO-water hypothesis",
        fluid_data.density_kg_m3,
        fluid_data.sound_speed_m_s,
        attenuation_db_per_m=inputs.fluid_attenuation_db_per_m,
        attenuation_power=inputs.attenuation_power,
    )
    air = Fluid("air", 1.196, 344.0)
    solid = ElasticSolid.from_longitudinal_speed_and_poisson(
        name="polypropylene",
        density_kg_m3=inputs.plate_density_kg_m3,
        longitudinal_speed_m_s=inputs.plate_longitudinal_speed_m_s,
        poisson_ratio=inputs.plate_poisson_ratio,
        longitudinal_attenuation_db_per_m=(
            inputs.pp_longitudinal_attenuation_db_per_m
        ),
        shear_attenuation_db_per_m=inputs.pp_shear_attenuation_db_per_m,
        attenuation_power=inputs.attenuation_power,
    )
    plate = ElasticPlate(solid, inputs.plate_thickness_mm * 1e-3)
    echo_grid_size = preset.echo_grid_size or preset.grid_size
    echo_grid_spacing_m = (
        preset.echo_grid_spacing_m or preset.grid_spacing_m
    )
    grid = CartesianGrid(
        nx=echo_grid_size,
        ny=echo_grid_size,
        dx_m=echo_grid_spacing_m,
    )
    aperture = FocusedCircularAperture(
        diameter_m=inputs.transducer_diameter_mm * 1e-3,
        focal_length_m=inputs.transducer_focal_length_mm * 1e-3,
    )
    water_arrivals_s = (
        2.0 * water_paths_mm * 1e-3 / water.sound_speed_m_s
    )
    plate_back_arrivals_s = (
        water_arrivals_s
        + 2.0
        * inputs.plate_thickness_mm
        * 1e-3
        / inputs.plate_longitudinal_speed_m_s
    )
    surface_arrivals_s = (
        plate_back_arrivals_s
        + 2.0
        * inputs.fluid_height_mm
        * 1e-3
        / liquid.sound_speed_m_s
    )
    drive_frequency_hz = inputs.excitation_frequency_mhz * 1e6
    drive_duration_s = inputs.excitation_cycles / drive_frequency_hz
    record_length_s = max(
        48.0e-6,
        float(np.max(surface_arrivals_s)) + 2.0 * drive_duration_s + 8.0e-6,
    )
    time_s, drive = sine_burst(
        center_frequency_hz=drive_frequency_hz,
        cycles=inputs.excitation_cycles,
        sample_rate_hz=preset.sample_rate_hz,
        record_length_s=record_length_s,
        start_time_s=0.0,
    )

    def certificate_round_trip_response(
        frequency_hz: np.ndarray,
    ) -> np.ndarray:
        return asymmetric_gaussian_response(
            frequency_hz,
            peak_frequency_hz=TRANSDUCER_PEAK_FREQUENCY_HZ,
            lower_frequency_6db_hz=TRANSDUCER_LOWER_FREQUENCY_6DB_HZ,
            upper_frequency_6db_hz=TRANSDUCER_UPPER_FREQUENCY_6DB_HZ,
        ) * smooth_dc_block_response(frequency_hz)

    results: list[PulseEchoResult] = []
    for water_path_mm in water_paths_mm:
        model = AngularSpectrumModel(
            grid=grid,
            aperture=aperture,
            incident_fluid=water,
            plate=plate,
            transmitted_fluid=liquid,
            water_path_m=float(water_path_mm) * 1e-3,
            plate_radial_samples=preset.radial_samples,
        )
        results.append(
            simulate_monostatic_pulse_echo(
                model,
                time_s,
                drive,
                fluid_layer_thickness_m=inputs.fluid_height_mm * 1e-3,
                backing_fluid=air,
                round_trip_response=certificate_round_trip_response,
                relative_spectrum_threshold=(
                    preset.relative_spectrum_threshold
                ),
                minimum_frequency_hz=0.0,
                maximum_frequency_hz=max(25.0e6, 1.5 * drive_frequency_hz),
                fluid_cavity_echo_count=1,
            )
        )

    received = tuple(result.received_signal for result in results)
    plate_front = tuple(result.plate_front_signal for result in results)
    backing = tuple(result.backing_signal for result in results)
    offsets_mm = np.asarray([record.offset_mm for record in records])
    reference_candidates = np.flatnonzero(
        np.isclose(offsets_mm, 0.0, rtol=0.0, atol=1.0e-9)
    )
    if reference_candidates.size != 1:
        raise ValueError(
            "the shared waveform correction requires exactly one z=0 survey"
        )
    reference_index = int(reference_candidates[0])
    common_reference_applied = False
    common_reference_error: str | None = None
    supported_frequency_hz: tuple[float, float] | None = None
    gate_relative_s: tuple[float, float] | None = None
    calibration_regularization: float | None = None
    calibration_maximum_correction_db: float | None = None
    try:
        calibration = estimate_reference_transfer(
            records[reference_index].survey.time_s,
            _baseline_corrected_trace(records[reference_index]),
            time_s,
            received[reference_index],
            measured_arrival_s=(
                records[reference_index].survey.water_pp_time_s
            ),
            simulated_arrival_s=float(water_arrivals_s[reference_index]),
            target_time_s=time_s,
        )
        received = tuple(
            apply_reference_transfer(time_s, signal, calibration)
            for signal in received
        )
        plate_front = tuple(
            apply_reference_transfer(time_s, signal, calibration)
            for signal in plate_front
        )
        backing = tuple(
            apply_reference_transfer(time_s, signal, calibration)
            for signal in backing
        )
        supported_frequency_hz = (
            calibration.minimum_frequency_hz,
            calibration.maximum_frequency_hz,
        )
        gate_relative_s = (
            calibration.gate_start_s,
            calibration.gate_end_s,
        )
        calibration_regularization = calibration.regularization
        calibration_maximum_correction_db = (
            calibration.maximum_correction_db
        )
        common_reference_applied = True
    except ValueError as exc:
        common_reference_error = str(exc)

    return PulseSweepSimulation(
        time_s=time_s,
        received_signals=received,
        plate_front_signals=plate_front,
        backing_signals=backing,
        water_arrivals_s=water_arrivals_s,
        plate_back_arrivals_s=plate_back_arrivals_s,
        surface_arrivals_s=surface_arrivals_s,
        common_reference_applied=common_reference_applied,
        common_reference_error=common_reference_error,
        common_reference_offset_mm=float(offsets_mm[reference_index]),
        common_reference_supported_frequency_hz=supported_frequency_hz,
        common_reference_gate_relative_s=gate_relative_s,
        common_reference_regularization=calibration_regularization,
        common_reference_maximum_correction_db=(
            calibration_maximum_correction_db
        ),
        sample_rate_hz=preset.sample_rate_hz,
        grid_size=echo_grid_size,
        grid_spacing_m=echo_grid_spacing_m,
        radial_samples=preset.radial_samples,
        active_frequency_bin_counts=tuple(
            int(np.count_nonzero(result.simulated_bin_mask))
            for result in results
        ),
    )


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.ndim != 1 or first.size < 2:
        raise ValueError("correlation requires equal one-dimensional arrays")
    if np.any(~np.isfinite(first)) or np.any(~np.isfinite(second)):
        raise ValueError("correlation values must be finite")
    if np.ptp(first) <= 1.0e-12 or np.ptp(second) <= 1.0e-12:
        raise ValueError("correlation is undefined for a constant response")
    return float(np.corrcoef(first, second)[0, 1])


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _baseline_corrected_trace(record: SweepRecord) -> np.ndarray:
    survey = record.survey
    baseline_gate = (
        (survey.time_s >= survey.water_pp_time_s - 4.0e-6)
        & (survey.time_s <= survey.water_pp_time_s - 2.0e-6)
    )
    if np.count_nonzero(baseline_gate) < 8:
        raise ValueError(
            f"survey has no usable pre-echo baseline gate: {record.path.name}"
        )
    baseline = float(np.median(survey.signal_adc[baseline_gate]))
    return survey.signal_adc - baseline


def _gated_envelope_peak(
    time_s: np.ndarray,
    signal: np.ndarray,
    arrival_s: float,
    *,
    half_width_s: float = 0.26e-6,
) -> float:
    """Return an envelope maximum in a narrow, interface-specific gate."""

    mask = np.abs(time_s - arrival_s) <= half_width_s
    if np.count_nonzero(mask) < 8:
        raise ValueError("pulse-echo interface gate contains too few samples")
    return float(np.max(analytic_envelope(signal)[mask]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-check a multi-position probe-z survey against timing and "
            "focused angular-spectrum predictions."
        )
    )
    parser.add_argument(
        "input_directory",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
    )
    parser.add_argument("--dmso-percent", type=float, default=80.0)
    parser.add_argument("--temperature-c", type=float, default=22.0)
    parser.add_argument(
        "--base-water-path-mm",
        type=float,
        help=(
            "nominal water gap at z=0; defaults to focal_distance_mm in the "
            "sweep summary. Pass an independently measured value when known"
        ),
    )
    parser.add_argument(
        "--transducer-focal-length-mm",
        type=float,
        default=25.4,
    )
    parser.add_argument(
        "--transducer-diameter-mm",
        type=float,
        default=13.0,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--output-stem", default="tof_z_offset_crosscheck")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.dmso_percent <= 100.0:
        raise ValueError("--dmso-percent must lie between 0 and 100")
    if not 20.0 <= args.temperature_c <= 40.0:
        raise ValueError("--temperature-c must lie between 20 and 40")
    if args.transducer_focal_length_mm <= 0.0:
        raise ValueError("--transducer-focal-length-mm must be > 0")
    if args.transducer_diameter_mm <= 0.0:
        raise ValueError("--transducer-diameter-mm must be > 0")

    summary, records = load_sweep(args.input_directory)
    offsets_mm = np.asarray([record.offset_mm for record in records])
    if args.base_water_path_mm is None:
        base_water_path_mm = _finite_float(
            summary.get("focal_distance_mm"),
            "summary focal distance",
        )
        base_path_source = (
            "nominal z=0 geometry inferred from summary focal_distance_mm"
        )
    else:
        base_water_path_mm = float(args.base_water_path_mm)
        base_path_source = "command-line independent geometry"
    if base_water_path_mm <= float(np.max(offsets_mm)):
        raise ValueError("base water path must remain positive over the sweep")
    model_water_paths_mm = base_water_path_mm - offsets_mm

    water_pp_us = np.asarray(
        [record.survey.water_pp_time_s * 1e6 for record in records]
    )
    plate_back_us = np.asarray(
        [record.survey.pp_fluid_time_s * 1e6 for record in records]
    )
    fluid_top_us = np.asarray(
        [record.survey.fluid_top_time_s * 1e6 for record in records]
    )
    pp_delay_us = plate_back_us - water_pp_us
    fluid_delay_us = fluid_top_us - plate_back_us
    pp_delay_median_us = float(np.median(pp_delay_us))
    fluid_delay_median_us = float(np.median(fluid_delay_us))
    plate_thickness_mm = (
        pp_delay_median_us * 1e-6 * PP_LONGITUDINAL_SPEED_M_S * 0.5 * 1e3
    )
    fluid_data = dmso_water_properties(
        args.dmso_percent / 100.0,
        basis="volume",
        temperature_c=args.temperature_c,
    )
    fluid_height_mm = (
        fluid_delay_median_us
        * 1e-6
        * fluid_data.sound_speed_m_s
        * 0.5
        * 1e3
    )

    first = records[0].survey
    frequency_mhz = float(first.probe_frequency_hz) * 1e-6
    cycles = float(first.tone_length_cycles)
    inputs = SimulationInputs(
        dmso_volume_percent=args.dmso_percent,
        temperature_c=args.temperature_c,
        water_path_mm=base_water_path_mm,
        plate_thickness_mm=plate_thickness_mm,
        fluid_height_mm=fluid_height_mm,
        excitation_frequency_mhz=frequency_mhz,
        excitation_cycles=cycles,
        transducer_diameter_mm=args.transducer_diameter_mm,
        transducer_focal_length_mm=args.transducer_focal_length_mm,
        fill_height_uncertainty_mm=min(0.05, 0.25 * fluid_height_mm),
        numerical_preset="Fast",
    )
    simulation = run_interactive_simulation(inputs)

    measured_plate = _normalized(
        np.asarray([record.plate_front_amplitude_adc for record in records])
    )
    measured_plate_back = _normalized(
        np.asarray([record.plate_back_amplitude_adc for record in records])
    )
    measured_surface = _normalized(
        np.asarray([record.surface_amplitude_adc for record in records])
    )
    if (
        float(np.min(model_water_paths_mm))
        < float(np.min(simulation.water_path_scan_mm))
        or float(np.max(model_water_paths_mm))
        > float(np.max(simulation.water_path_scan_mm))
    ):
        raise ValueError("modeled z positions lie outside the focus scan")
    modeled_surface_cw = _normalized(
        np.interp(
            model_water_paths_mm,
            simulation.water_path_scan_mm,
            simulation.meniscus_intensity_normalized,
        )
    )
    pulse_sweep = _simulate_pulse_sweep(inputs, model_water_paths_mm, records)
    modeled_plate = _normalized(
        np.asarray(
            [
                _gated_envelope_peak(
                    pulse_sweep.time_s,
                    signal,
                    float(arrival),
                )
                for signal, arrival in zip(
                    pulse_sweep.plate_front_signals,
                    pulse_sweep.water_arrivals_s,
                )
            ]
        )
    )
    modeled_plate_back = _normalized(
        np.asarray(
            [
                _gated_envelope_peak(
                    pulse_sweep.time_s,
                    signal,
                    float(arrival),
                )
                # The library's plate_front_signal is the complete elastic-
                # plate reflection component; its delayed part contains the
                # PP-liquid interface return. backing_signal is reserved for
                # the liquid-air cavity return.
                for signal, arrival in zip(
                    pulse_sweep.plate_front_signals,
                    pulse_sweep.plate_back_arrivals_s,
                )
            ]
        )
    )
    modeled_surface_pulse = _normalized(
        np.asarray(
            [
                _gated_envelope_peak(
                    pulse_sweep.time_s,
                    signal,
                    float(arrival),
                )
                for signal, arrival in zip(
                    pulse_sweep.backing_signals,
                    pulse_sweep.surface_arrivals_s,
                )
            ]
        )
    )

    water_fit = linear_fit(offsets_mm, water_pp_us)
    plate_back_fit = linear_fit(offsets_mm, plate_back_us)
    surface_fit = linear_fit(offsets_mm, fluid_top_us)
    probe_command_fit = linear_fit(
        offsets_mm,
        np.asarray([record.probe_position_command_mm for record in records]),
    )
    probe_actual_fit = linear_fit(
        offsets_mm,
        np.asarray([record.probe_position_actual_mm for record in records]),
    )
    probe_absolute_fit = linear_fit(
        offsets_mm,
        np.asarray(
            [record.probe_position_abs_actual_mm for record in records]
        ),
    )
    bubbler_positions = np.asarray(
        [
            np.nan
            if record.bubbler_position_mm is None
            else record.bubbler_position_mm
            for record in records
        ]
    )
    if water_fit["slope"] >= -1.0e-9:
        raise ValueError(
            "water-PP arrival must decrease as the positive z offset closes "
            "the water gap"
        )
    apparent_speed_m_s = -2000.0 / water_fit["slope"]
    water_data = water_properties(args.temperature_c)
    expected_water_slope_us_per_mm = -2000.0 / water_data.sound_speed_m_s
    expected_water_intercept_us = float(
        np.mean(water_pp_us - expected_water_slope_us_per_mm * offsets_mm)
    )
    nominal_water_round_trip_us = (
        2.0
        * model_water_paths_mm
        * 1e-3
        / water_data.sound_speed_m_s
        * 1e6
    )
    residual_delay_us = water_pp_us - nominal_water_round_trip_us
    optimal_z_offset_mm = (
        base_water_path_mm - simulation.optimal_water_path_mm
    )
    measured_surface_peak_index = int(np.argmax(measured_surface))
    measured_surface_peak_at_boundary = measured_surface_peak_index in {
        0,
        len(records) - 1,
    }
    measured_surface_peak_description = (
        "measured maximum at scan boundary"
        if measured_surface_peak_at_boundary
        else "measured maximum inside scan"
    )

    figure = plt.figure(figsize=(13.0, 12.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.2))
    timing_axis = figure.add_subplot(grid[0, 0])
    for values, fit, label, color in (
        (water_pp_us, water_fit, "water–PP", "#1c2d57"),
        (plate_back_us, plate_back_fit, "PP–liquid", "#cf5f4b"),
        (fluid_top_us, surface_fit, "liquid–air", "#007a4e"),
    ):
        timing_axis.plot(offsets_mm, values, "o", color=color)
        timing_axis.plot(
            offsets_mm,
            fit["slope"] * offsets_mm + fit["intercept"],
            color=color,
            label=f"{label}: {fit['slope']:+.4f} µs/mm",
        )
    timing_axis.plot(
        offsets_mm,
        expected_water_slope_us_per_mm * offsets_mm
        + expected_water_intercept_us,
        color="#1c2d57",
        linestyle="--",
        alpha=0.65,
        label=(
            f"{args.temperature_c:g} °C water slope expectation "
            "(intercept aligned): "
            f"{expected_water_slope_us_per_mm:+.4f} µs/mm"
        ),
    )
    timing_axis.set(
        title="All absolute echoes move together",
        xlabel="Probe z offset [mm]",
        ylabel="Arrival since excitation [µs]",
    )
    timing_axis.grid(alpha=0.22)
    timing_axis.legend()

    internal_axis = figure.add_subplot(grid[0, 1])
    internal_axis.plot(
        offsets_mm,
        (pp_delay_us - np.median(pp_delay_us)) * 1e3,
        "o-",
        label="PP round trip",
        color="#cf5f4b",
    )
    internal_axis.plot(
        offsets_mm,
        (fluid_delay_us - np.median(fluid_delay_us)) * 1e3,
        "o-",
        label="liquid round trip",
        color="#007a4e",
    )
    internal_axis.axhline(0.0, color="0.55", linewidth=0.8)
    internal_axis.set(
        title="Internal stack delays remain nearly fixed",
        xlabel="Probe z offset [mm]",
        ylabel="Delay deviation from median [ns]",
    )
    internal_axis.grid(alpha=0.22)
    internal_axis.legend()
    internal_axis.text(
        0.02,
        0.04,
        (
            f"Common PP: {plate_thickness_mm:.4f} mm\n"
            f"Liquid hypothesis: {fluid_height_mm:.4f} mm"
        ),
        transform=internal_axis.transAxes,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    plate_axis = figure.add_subplot(grid[1, 0])
    plate_axis.plot(
        offsets_mm,
        measured_plate,
        "o-",
        color="#1c2d57",
        label="survey water–PP detector amplitude",
    )
    plate_axis.plot(
        offsets_mm,
        modeled_plate,
        "s--",
        color="#e77925",
        label="monostatic pulse model: water–PP",
    )
    plate_axis.plot(
        offsets_mm,
        measured_plate_back,
        "^-",
        color="#cf5f4b",
        label="survey PP–liquid detector amplitude",
    )
    plate_axis.plot(
        offsets_mm,
        modeled_plate_back,
        "v--",
        color="#8c3e33",
        label="monostatic pulse model: PP–liquid",
    )
    plate_axis.set(
        title=(
            "Interface amplitudes · each curve normalized over z\n"
            "water–PP r="
            f"{_correlation(measured_plate, modeled_plate):.3f}, "
            "PP–liquid r="
            f"{_correlation(measured_plate_back, modeled_plate_back):.3f}"
        ),
        xlabel="Probe z offset [mm]",
        ylabel="Response normalized over sweep",
        ylim=(-0.03, 1.08),
    )
    plate_axis.grid(alpha=0.22)
    plate_axis.legend()

    surface_axis = figure.add_subplot(grid[1, 1])
    z_scan_mm = base_water_path_mm - simulation.water_path_scan_mm
    order = np.argsort(z_scan_mm)
    surface_axis.plot(
        z_scan_mm[order],
        simulation.meniscus_intensity_normalized[order],
        color="#176c82",
        label="ASM on-axis single-pass intensity proxy",
    )
    surface_axis.plot(
        offsets_mm,
        measured_surface,
        "o",
        color="#007a4e",
        markersize=7,
        label="survey surface detector amplitude",
    )
    surface_axis.plot(
        offsets_mm,
        modeled_surface_pulse,
        "s--",
        color="#e77925",
        label="monostatic broadband surface echo",
    )
    surface_axis.axvline(
        optimal_z_offset_mm,
        color="#e77925",
        linestyle="--",
        label=f"nominal-geometry ASM optimum {optimal_z_offset_mm:+.2f} mm",
    )
    surface_axis.set(
        xlim=(
            min(float(np.min(offsets_mm)), optimal_z_offset_mm) - 0.4,
            max(float(np.max(offsets_mm)), optimal_z_offset_mm) + 0.6,
        ),
        ylim=(-0.03, 1.08),
        title=(
            f"Meniscus trend · n={len(records)} and "
            f"{measured_surface_peak_description}\n"
            "pulse r="
            f"{_correlation(measured_surface, modeled_surface_pulse):.3f}, "
            "single-pass proxy r="
            f"{_correlation(measured_surface, modeled_surface_cw):.3f}"
        ),
        xlabel="Probe z offset [mm]",
        ylabel="Response normalized over sweep",
    )
    surface_axis.grid(alpha=0.22)
    surface_axis.legend()

    waveform_axis = figure.add_subplot(grid[2, :])
    corrected = [_baseline_corrected_trace(record) for record in records]
    waveform_start_us = -0.35
    waveform_end_us = max(
        2.85,
        pp_delay_median_us + fluid_delay_median_us + 0.45,
    )
    display_window = [
        (
            (record.survey.relative_time_s >= waveform_start_us * 1e-6)
            & (record.survey.relative_time_s <= waveform_end_us * 1e-6)
        )
        for record in records
    ]
    global_scale = max(
        float(np.max(np.abs(signal[mask])))
        for signal, mask in zip(corrected, display_window)
    )
    simulated_global_scale = max(
        float(np.max(np.abs(signal)))
        for signal in pulse_sweep.received_signals
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(records)))
    separation = 1.25
    for index, (record, signal, color) in enumerate(
        zip(records, corrected, colors)
    ):
        relative_us = record.survey.relative_time_s * 1e6
        mask = (
            (relative_us >= waveform_start_us)
            & (relative_us <= waveform_end_us)
        )
        waveform_axis.plot(
            relative_us[mask],
            signal[mask] / global_scale + index * separation,
            color=color,
            linewidth=0.85,
            label="survey ADC" if index == 0 else None,
        )
        simulated_relative_us = (
            pulse_sweep.time_s - pulse_sweep.water_arrivals_s[index]
        ) * 1e6
        simulated_mask = (
            (simulated_relative_us >= waveform_start_us)
            & (simulated_relative_us <= waveform_end_us)
        )
        waveform_axis.plot(
            simulated_relative_us[simulated_mask],
            pulse_sweep.received_signals[index][simulated_mask]
            / simulated_global_scale
            + index * separation,
            color="#e77925",
            linestyle="--",
            linewidth=0.95,
            alpha=0.9,
            label=(
                "shared-correction pulse model" if index == 0 else None
            ),
        )
        waveform_axis.text(
            waveform_start_us + 0.02,
            index * separation + 0.38,
            f"z={record.offset_mm:+g} mm",
            color=color,
            fontsize=9,
        )
    waveform_axis.axvline(0.0, color="0.45", linestyle="--", linewidth=0.9)
    waveform_axis.axvline(
        pp_delay_median_us,
        color="#cf5f4b",
        linestyle=":",
        linewidth=1.0,
    )
    waveform_axis.axvline(
        pp_delay_median_us + fluid_delay_median_us,
        color="#007a4e",
        linestyle="-.",
        linewidth=1.0,
    )
    waveform_axis.set(
        xlim=(waveform_start_us, waveform_end_us),
        title=(
            "Waveforms use one scale per source; model correction is shared "
            "across z"
        ),
        xlabel="Time relative to water–PP echo [µs]",
        ylabel="Global ADC scale + vertical offset",
        yticks=[],
    )
    waveform_axis.grid(axis="x", alpha=0.2)
    waveform_axis.legend(loc="upper right")

    figure.suptitle(
        "Probe-z sweep cross-check · "
        f"{args.dmso_percent:g} vol.% DMSO hypothesis at "
        f"{args.temperature_c:g} °C",
        fontsize=15,
        fontweight="bold",
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_directory / f"{args.output_stem}.png"
    csv_path = args.output_directory / f"{args.output_stem}.csv"
    json_path = args.output_directory / f"{args.output_stem}.json"
    figure.savefig(plot_path, dpi=190)
    plt.close(figure)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "probe_z_offset_mm",
                "model_water_path_mm",
                "water_pp_time_us",
                "pp_delay_us",
                "fluid_delay_us",
                "plate_front_adc",
                "plate_back_adc",
                "surface_adc",
                "measured_plate_normalized",
                "modeled_pulse_plate_normalized",
                "measured_plate_back_normalized",
                "modeled_pulse_plate_back_normalized",
                "measured_surface_normalized",
                "modeled_pulse_surface_normalized",
                "modeled_single_pass_surface_normalized",
            )
        )
        for index, record in enumerate(records):
            writer.writerow(
                (
                    record.offset_mm,
                    model_water_paths_mm[index],
                    water_pp_us[index],
                    pp_delay_us[index],
                    fluid_delay_us[index],
                    record.plate_front_amplitude_adc,
                    record.plate_back_amplitude_adc,
                    record.surface_amplitude_adc,
                    measured_plate[index],
                    modeled_plate[index],
                    measured_plate_back[index],
                    modeled_plate_back[index],
                    measured_surface[index],
                    modeled_surface_pulse[index],
                    modeled_surface_cw[index],
                )
            )

    output = {
        "data": {
            "input_directory_name": args.input_directory.name,
            "well": summary.get("well"),
            "plate": get_labcyte_plate(str(first.plate_type_id)).id,
            "absolute_adc_calibrated": False,
            "offset_count": len(records),
        },
        "constant_acquisition_settings": {
            "frequency_hz": first.probe_frequency_hz,
            "cycles": first.tone_length_cycles,
            "voltage_setting_v": first.probe_voltage_setting_v,
            "sample_rate_hz": first.sample_rate_hz,
        },
        "hypothesis": {
            "dmso_volume_percent": args.dmso_percent,
            "temperature_c": args.temperature_c,
            "fluid_sound_speed_m_s": fluid_data.sound_speed_m_s,
            "fluid_metadata_supports_hypothesis": False,
            "base_water_path_mm": base_water_path_mm,
            "base_water_path_source": base_path_source,
            "transducer_focal_length_mm": args.transducer_focal_length_mm,
            "transducer_diameter_mm": args.transducer_diameter_mm,
        },
        "common_geometry": {
            "pp_delay_median_us": pp_delay_median_us,
            "plate_thickness_mm": plate_thickness_mm,
            "fluid_delay_median_us": fluid_delay_median_us,
            "fluid_height_mm_for_hypothesis": fluid_height_mm,
        },
        "pulse_model": {
            "sample_rate_hz": pulse_sweep.sample_rate_hz,
            "grid_size": pulse_sweep.grid_size,
            "grid_spacing_m": pulse_sweep.grid_spacing_m,
            "plate_radial_samples": pulse_sweep.radial_samples,
            "active_frequency_bin_counts": list(
                pulse_sweep.active_frequency_bin_counts
            ),
            "shared_reference": {
                "reference_offset_mm": (
                    pulse_sweep.common_reference_offset_mm
                ),
                "applied": pulse_sweep.common_reference_applied,
                "error": pulse_sweep.common_reference_error,
                "supported_frequency_hz": (
                    pulse_sweep.common_reference_supported_frequency_hz
                ),
                "gate_relative_s": (
                    pulse_sweep.common_reference_gate_relative_s
                ),
                "regularization": (
                    pulse_sweep.common_reference_regularization
                ),
                "maximum_correction_db": (
                    pulse_sweep.common_reference_maximum_correction_db
                ),
            },
        },
        "timing": {
            "water_pp_fit_us_per_mm": water_fit,
            "plate_back_fit_us_per_mm": plate_back_fit,
            "surface_fit_us_per_mm": surface_fit,
            "expected_water_slope_us_per_mm": (
                expected_water_slope_us_per_mm
            ),
            "apparent_tof_speed_m_s": apparent_speed_m_s,
            "apparent_speed_is_not_independently_calibrated": True,
            "residual_delay_vs_nominal_geometry_us": residual_delay_us.tolist(),
            "residual_delay_mean_us": float(np.mean(residual_delay_us)),
            "residual_delay_range_us": float(np.ptp(residual_delay_us)),
        },
        "motion_crosscheck": {
            "probe_command_fit_mm_per_mm": probe_command_fit,
            "probe_primary_encoder_fit_mm_per_mm": probe_actual_fit,
            "probe_absolute_encoder_fit_mm_per_mm": probe_absolute_fit,
            "bubbler_position_range_mm": (
                None
                if np.all(np.isnan(bubbler_positions))
                else float(
                    np.nanmax(bubbler_positions)
                    - np.nanmin(bubbler_positions)
                )
            ),
        },
        "focus_crosscheck": {
            "modeled_optimal_water_path_mm": simulation.optimal_water_path_mm,
            "modeled_optimal_probe_z_offset_mm": optimal_z_offset_mm,
            "optimum_z_shift_per_base_path_shift_mm_per_mm": 1.0,
            "pulse_model_common_reference_applied": (
                pulse_sweep.common_reference_applied
            ),
            "pulse_model_common_reference_error": (
                pulse_sweep.common_reference_error
            ),
            "water_pp_pulse_response_correlation": _correlation(
                measured_plate, modeled_plate
            ),
            "water_pp_pulse_response_rmse": _rmse(
                measured_plate, modeled_plate
            ),
            "pp_liquid_pulse_response_correlation": _correlation(
                measured_plate_back, modeled_plate_back
            ),
            "pp_liquid_pulse_response_rmse": _rmse(
                measured_plate_back, modeled_plate_back
            ),
            "surface_pulse_response_correlation": _correlation(
                measured_surface, modeled_surface_pulse
            ),
            "surface_pulse_response_rmse": _rmse(
                measured_surface, modeled_surface_pulse
            ),
            "surface_single_pass_proxy_correlation": _correlation(
                measured_surface, modeled_surface_cw
            ),
            "surface_single_pass_proxy_rmse": _rmse(
                measured_surface, modeled_surface_cw
            ),
        },
        "limitations": [
            "Fluid identity and temperature are absent from the survey JSON.",
            "ADC increments are qualitative and not calibrated to pressure.",
            (
                "The summary focal_distance_mm is treated as the nominal "
                "z=0 water path and should be checked against the stage datum."
            ),
            (
                f"Only {len(records)} positions were measured and all response "
                "curves are normalized separately. "
                + (
                    "The measured meniscus maximum lies at a scan boundary, "
                    "so both sides of the maximum are not bracketed. "
                    if measured_surface_peak_at_boundary
                    else "The measured maximum is inside the sampled range. "
                )
                + "Correlations are descriptive."
            ),
            "Sequential acquisition is confounded with drift and bubbler motion.",
            (
                "Survey amplitudes are positive detector estimates rather "
                "than independently recomputed waveform-envelope peaks."
            ),
            (
                "One bounded correction derived only from the z=0 water-PP "
                "waveform is applied to every simulated z trace. It does not "
                "calibrate absolute gain or interface-specific attenuation."
            ),
            (
                "The single-pass on-axis intensity is a focus proxy; the "
                "finite, possibly curved meniscus requires a receive integral."
            ),
            (
                "The pulse model assumes a planar, parallel meniscus and an "
                "isotropic PP plate. This run omits attenuation in water, PP, "
                "and the DMSO-water hypothesis."
            ),
            "The apparent TOF speed can include focused broadband group-delay bias.",
        ],
    }
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(
        "Absolute echo slope:",
        f"{water_fit['slope']:+.6f} µs/mm,",
        f"R²={water_fit['r_squared']:.6f},",
        f"apparent TOF speed={apparent_speed_m_s:.2f} m/s",
    )
    print(
        "Water-model expectation:",
        f"{expected_water_slope_us_per_mm:+.6f} µs/mm,",
        f"nominal-geometry residual delay "
        f"{float(np.min(residual_delay_us)):.3f}–"
        f"{float(np.max(residual_delay_us)):.3f} µs",
    )
    print(
        "Probe encoder slope:",
        f"{probe_actual_fit['slope']:+.6f} mm/mm,",
        f"R²={probe_actual_fit['r_squared']:.6f}",
    )
    print(
        "Common geometry:",
        f"PP={plate_thickness_mm:.6f} mm,",
        f"fluid={fluid_height_mm:.6f} mm for the stated hypothesis",
    )
    print(
        "Pulse responses:",
        f"water-PP r={_correlation(measured_plate, modeled_plate):.4f},",
        "PP-liquid r="
        f"{_correlation(measured_plate_back, modeled_plate_back):.4f},",
        "shared reference="
        f"{pulse_sweep.common_reference_applied}",
    )
    print(
        "Surface response:",
        "pulse r="
        f"{_correlation(measured_surface, modeled_surface_pulse):.4f},",
        "single-pass proxy r="
        f"{_correlation(measured_surface, modeled_surface_cw):.4f},",
        f"modeled optimum z={optimal_z_offset_mm:+.3f} mm",
    )
    print(f"Plot: {plot_path.resolve()}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Summary: {json_path.resolve()}")


if __name__ == "__main__":
    main()
