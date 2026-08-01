"""Conservative import and time-of-flight interpretation of survey JSON data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SurveyPulseEcho:
    """One high-rate survey trace and its detected interface times.

    ADC values are retained as relative values only.  ``normalized_signal`` is
    baseline-corrected and scaled by its largest absolute sample.
    """

    time_s: NDArray[np.float64]
    signal_adc: NDArray[np.float64]
    normalized_signal: NDArray[np.float64]
    sample_rate_hz: float
    water_pp_time_s: float
    pp_fluid_time_s: float
    fluid_top_time_s: float
    stored_probe_to_plate_m: float | None
    stored_plate_thickness_m: float | None
    stored_fluid_height_m: float | None
    fluid_material: str | None
    date_time: str | None
    probe_frequency_hz: float | None
    tone_length_cycles: float | None
    probe_voltage_setting_v: float | None
    sample_index_start: int | None

    @property
    def time_since_excitation_s(self) -> NDArray[np.float64]:
        """Recorded sample times measured from the ping excitation start."""

        return self.time_s

    @property
    def relative_time_s(self) -> NDArray[np.float64]:
        """Sample times relative to the detected water-PP interface."""

        return self.time_s - self.water_pp_time_s

    @property
    def pp_round_trip_time_s(self) -> float:
        return self.pp_fluid_time_s - self.water_pp_time_s

    @property
    def fluid_round_trip_time_s(self) -> float:
        return self.fluid_top_time_s - self.pp_fluid_time_s

    @property
    def excitation_metadata_is_calibrated(self) -> bool:
        """Survey settings are metadata, not measured connector waveforms."""

        return False


@dataclass(frozen=True)
class SurveyGeometryInterpretation:
    """Stored and time-of-flight-equivalent geometry for chosen sound speeds."""

    stored_probe_to_plate_m: float | None
    stored_plate_thickness_m: float | None
    stored_fluid_height_m: float | None
    implied_incident_speed_m_s: float | None
    implied_plate_speed_m_s: float | None
    implied_fluid_speed_m_s: float | None
    tof_probe_to_plate_m: float
    tof_plate_thickness_m: float
    tof_fluid_height_m: float
    stored_height_fluid_delay_s: float | None
    stored_height_timing_error_s: float | None


def _positive_optional_m(value: object) -> float | None:
    if value is None:
        return None
    result = float(value) * 1e-3
    if not np.isfinite(result) or result <= 0.0:
        return None
    return result


def _positive_optional(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        return None
    return result


def parse_survey_pulse_echo(
    document: Mapping[str, Any],
) -> SurveyPulseEcho:
    """Parse the first high-rate ``SurveyPingsSuper`` trace from a mapping.

    This in-memory entry point is useful for uploaded survey files: callers do
    not need to persist raw ADC data to disk before interpreting it.
    """

    super_pings = document.get("SurveyPingsSuper") or []
    if not super_pings:
        raise ValueError("survey JSON contains no SurveyPingsSuper trace")
    ping = super_pings[0]
    signal = np.asarray(ping.get("SignalData"), dtype=float)
    sample_rate_hz = float(ping.get("SampleFrequency", 0.0))
    start_us = float(document.get("SampleRangeAnalysisStartUSecs"))
    if (
        signal.ndim != 1
        or signal.size < 32
        or np.any(~np.isfinite(signal))
        or not np.isfinite(sample_rate_hz)
        or sample_rate_hz <= 0.0
        or not np.isfinite(start_us)
    ):
        raise ValueError("invalid SurveyPingsSuper signal or sampling metadata")

    result = document.get("SurveyResult") or {}
    required_times = {
        "water_pp_time_s": result.get("PlateBaseSampleTimeUSecs"),
        "pp_fluid_time_s": result.get("WellBaseSampleTimeUSecs"),
        "fluid_top_time_s": result.get("FluidTopSampleTimeUSecs"),
    }
    if any(value is None for value in required_times.values()):
        raise ValueError("survey JSON is missing one or more interface times")
    interface_times_s = {
        name: float(value) * 1e-6
        for name, value in required_times.items()
    }
    if not (
        interface_times_s["water_pp_time_s"]
        < interface_times_s["pp_fluid_time_s"]
        < interface_times_s["fluid_top_time_s"]
    ):
        raise ValueError("survey interface times must be strictly increasing")

    time_s = start_us * 1e-6 + np.arange(signal.size) / sample_rate_hz
    edge_count = max(8, signal.size // 20)
    baseline = float(
        np.median(np.concatenate((signal[:edge_count], signal[-edge_count:])))
    )
    baseline_corrected = signal - baseline
    scale = float(np.max(np.abs(baseline_corrected)))
    if scale <= 0.0:
        raise ValueError("survey trace has zero ADC range")

    fluid_material = document.get("FluidMaterial")
    if fluid_material is not None:
        fluid_material = str(fluid_material)
    date_time = document.get("DateTime")
    if date_time is not None:
        date_time = str(date_time)
    sample_index_start_raw = ping.get("SampleIndexStart")
    sample_index_start = None
    if sample_index_start_raw is not None:
        candidate = int(sample_index_start_raw)
        if candidate >= 0:
            sample_index_start = candidate

    return SurveyPulseEcho(
        time_s=time_s,
        signal_adc=signal,
        normalized_signal=baseline_corrected / scale,
        sample_rate_hz=sample_rate_hz,
        stored_probe_to_plate_m=_positive_optional_m(
            result.get("ProbeToPlateBaseDistance")
        ),
        stored_plate_thickness_m=_positive_optional_m(
            result.get("WellBaseThickness")
        ),
        stored_fluid_height_m=_positive_optional_m(
            result.get("FluidHeight")
        ),
        fluid_material=fluid_material,
        date_time=date_time,
        probe_frequency_hz=_positive_optional(ping.get("ProbeFrequency")),
        tone_length_cycles=_positive_optional(ping.get("ToneLength")),
        probe_voltage_setting_v=_positive_optional(ping.get("ProbeVoltage")),
        sample_index_start=sample_index_start,
        **interface_times_s,
    )


def load_survey_pulse_echo(path: str | Path) -> SurveyPulseEcho:
    """Load the first high-rate ``SurveyPingsSuper`` trace from JSON."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise ValueError("survey JSON root must be an object")
    return parse_survey_pulse_echo(document)


def interpret_survey_geometry(
    survey: SurveyPulseEcho,
    *,
    incident_sound_speed_m_s: float,
    plate_longitudinal_speed_m_s: float,
    fluid_sound_speed_m_s: float,
) -> SurveyGeometryInterpretation:
    """Interpret stored distances without treating them as independent truth.

    The TOF-equivalent distances assume zero fixed electronic delay.  Implied
    speeds expose which sound speeds were used to generate stored distances.
    """

    speeds = {
        "incident_sound_speed_m_s": incident_sound_speed_m_s,
        "plate_longitudinal_speed_m_s": plate_longitudinal_speed_m_s,
        "fluid_sound_speed_m_s": fluid_sound_speed_m_s,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in speeds.values()):
        raise ValueError("sound speeds must be finite and > 0")

    incident_time_s = survey.water_pp_time_s
    plate_time_s = survey.pp_round_trip_time_s
    fluid_time_s = survey.fluid_round_trip_time_s

    def implied_speed(distance_m: float | None, time_s: float) -> float | None:
        if distance_m is None:
            return None
        return 2.0 * distance_m / time_s

    stored_fluid_delay_s = None
    stored_height_timing_error_s = None
    if survey.stored_fluid_height_m is not None:
        stored_fluid_delay_s = (
            2.0 * survey.stored_fluid_height_m / fluid_sound_speed_m_s
        )
        stored_height_timing_error_s = stored_fluid_delay_s - fluid_time_s

    return SurveyGeometryInterpretation(
        stored_probe_to_plate_m=survey.stored_probe_to_plate_m,
        stored_plate_thickness_m=survey.stored_plate_thickness_m,
        stored_fluid_height_m=survey.stored_fluid_height_m,
        implied_incident_speed_m_s=implied_speed(
            survey.stored_probe_to_plate_m,
            incident_time_s,
        ),
        implied_plate_speed_m_s=implied_speed(
            survey.stored_plate_thickness_m,
            plate_time_s,
        ),
        implied_fluid_speed_m_s=implied_speed(
            survey.stored_fluid_height_m,
            fluid_time_s,
        ),
        tof_probe_to_plate_m=(
            0.5 * incident_sound_speed_m_s * incident_time_s
        ),
        tof_plate_thickness_m=(
            0.5 * plate_longitudinal_speed_m_s * plate_time_s
        ),
        tof_fluid_height_m=0.5 * fluid_sound_speed_m_s * fluid_time_s,
        stored_height_fluid_delay_s=stored_fluid_delay_s,
        stored_height_timing_error_s=stored_height_timing_error_s,
    )
