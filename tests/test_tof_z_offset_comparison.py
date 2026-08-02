"""Tests for the reproducible probe-z sweep example."""

import json
from pathlib import Path

import numpy as np
import pytest

from examples.tof_z_offset_comparison import (
    _correlation,
    linear_fit,
    load_sweep,
    offset_from_path,
)


def _survey_document(offset_mm: float, *, frequency_hz: float = 10.0e6) -> dict:
    signal = np.zeros(200)
    signal[80:85] = [0.0, 1.0, -2.0, 1.0, 0.0]
    water_time_us = 35.0 - 1.3 * offset_mm
    return {
        "PlateTypeId": "2a09adfa-a468-4327-b5b1-5f8296136782",
        "WellName": "I18",
        "FluidMaterial": None,
        "SampleRangeAnalysisStartUSecs": 0.0,
        "ProbePositionCommand": 10.0 + offset_mm,
        "ProbePositionActual": 10.0 + offset_mm,
        "ProbePositionAbsActual": 10.0 + offset_mm,
        "BubblerPosition": 0.9,
        "SurveyResult": {
            "PlateBaseSampleTimeUSecs": water_time_us,
            "WellBaseSampleTimeUSecs": water_time_us + 0.57,
            "FluidTopSampleTimeUSecs": water_time_us + 2.40,
            "ProbeToPlateBaseDistance": water_time_us * 1.5 / 2.0,
            "WellBaseThickness": 0.78,
            "FluidHeight": 1.37,
            "PlateBaseSampleAmplitude": 1000.0,
            "WellBaseSampleAmplitude": 300.0,
            "FluidTopSampleAmplitude": 800.0,
        },
        "SurveyPingsSuper": [
            {
                "SampleFrequency": 5.0e6,
                "SampleIndexStart": 0,
                "ProbeFrequency": frequency_hz,
                "ToneLength": 1.0,
                "ProbeVoltage": 110.0,
                "SignalData": signal.tolist(),
            }
        ],
    }


def _write_sweep(directory: Path, *, second_frequency_hz: float = 10.0e6) -> None:
    offsets = (-1.0, 0.0, 1.0)
    summary = {
        "plate_id": "2a09adfa-a468-4327-b5b1-5f8296136782",
        "well": "I18",
        "configured_probe_frequency_hz": 10.0e6,
        "focal_distance_mm": 25.0,
        "rows": [{"probe_z_offset_mm": value} for value in offsets],
    }
    (directory / "tof_z_offset_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    for offset in offsets:
        frequency = second_frequency_hz if offset == 0.0 else 10.0e6
        path = directory / f"survey_I18_z_offset_{offset:+.1f}mm.json"
        path.write_text(
            json.dumps(_survey_document(offset, frequency_hz=frequency)),
            encoding="utf-8",
        )


def test_offset_from_path_preserves_sign() -> None:
    assert offset_from_path(Path("survey_I18_z_offset_-1.0mm.json")) == -1.0
    assert offset_from_path(Path("survey_I18_z_offset_+3.0mm.json")) == 3.0
    with pytest.raises(ValueError, match="cannot read z offset"):
        offset_from_path(Path("survey.json"))


def test_linear_fit_reports_exact_slope_and_residual() -> None:
    result = linear_fit(
        np.asarray([-1.0, 0.0, 1.0, 2.0]),
        np.asarray([36.0, 34.7, 33.4, 32.1]),
    )
    assert result["slope"] == pytest.approx(-1.3)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["maximum_absolute_residual"] == pytest.approx(0.0, abs=1e-12)


def test_correlation_rejects_constant_response() -> None:
    with pytest.raises(ValueError, match="constant response"):
        _correlation(np.ones(3), np.asarray([0.0, 0.5, 1.0]))


def test_load_sweep_matches_summary_and_rejects_setting_changes(
    tmp_path: Path,
) -> None:
    _write_sweep(tmp_path)
    summary, records = load_sweep(tmp_path)
    assert summary["well"] == "I18"
    assert [record.offset_mm for record in records] == [-1.0, 0.0, 1.0]
    assert records[0].survey.plate_type_id == (
        "2a09adfa-a468-4327-b5b1-5f8296136782"
    )

    _write_sweep(tmp_path, second_frequency_hz=9.0e6)
    with pytest.raises(ValueError, match="probe frequency changes"):
        load_sweep(tmp_path)


def test_load_sweep_requires_exact_zero_reference(tmp_path: Path) -> None:
    _write_sweep(tmp_path)
    zero_path = tmp_path / "survey_I18_z_offset_+0.0mm.json"
    zero_path.rename(tmp_path / "survey_I18_z_offset_+2.0mm.json")
    summary_path = tmp_path / "tof_z_offset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["rows"][1]["probe_z_offset_mm"] = 2.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one z=0"):
        load_sweep(tmp_path)
