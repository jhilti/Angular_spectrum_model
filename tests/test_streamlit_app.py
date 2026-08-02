"""App-level regression tests for the interactive Streamlit flow."""

from dataclasses import dataclass
import json
from pathlib import Path

import angular_spectrum
import pytest

from angular_spectrum import dmso_water_properties, water_properties
from angular_spectrum.labware import get_labcyte_plate


pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=180).run()
    assert not app.exception
    assert not app.error
    return app


def _markdown_containing(app: AppTest, text: str) -> list[str]:
    return [element.value for element in app.markdown if text in element.value]


def _survey_upload() -> tuple[str, bytes, str]:
    signal = [0.0] * 200
    signal[60:65] = [0.0, 1.0, -2.0, 1.0, 0.0]
    document = {
        "DateTime": "2026-01-01T00:00:00Z",
        "PlateTypeId": "dcf061d9-9470-455b-ba0d-8ea085de5810",
        "FluidMaterial": "unknown",
        "SampleRangeAnalysisStartUSecs": 10.0,
        "SurveyResult": {
            "PlateBaseSampleTimeUSecs": 12.0,
            "WellBaseSampleTimeUSecs": 12.5,
            "FluidTopSampleTimeUSecs": 14.0,
            "ProbeToPlateBaseDistance": 9.0,
            "WellBaseThickness": 0.65,
            "FluidHeight": 1.125,
        },
        "SurveyPingsSuper": [
            {
                "SampleFrequency": 20.0e6,
                "SampleIndexStart": 7,
                "ProbeFrequency": 8.0e6,
                "ToneLength": 2.0,
                "ProbeVoltage": 150.0,
                "SignalData": signal,
            }
        ],
    }
    return (
        "survey.json",
        json.dumps(document).encode("utf-8"),
        "application/json",
    )


def _number_input(app: AppTest, label: str):
    return next(item for item in app.number_input if item.label == label)


def _selectbox(app: AppTest, label: str):
    return next(item for item in app.selectbox if item.label == label)


def _button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def test_mixed_survey_schema_stops_with_reboot_guidance(monkeypatch) -> None:
    @dataclass(frozen=True)
    class LegacySurveyPulseEcho:
        """Minimal pre-plate-ID schema retained by a hot-reloaded process."""

        probe_frequency_hz: float | None = None
        tone_length_cycles: float | None = None
        probe_voltage_setting_v: float | None = None

    monkeypatch.setattr(
        angular_spectrum,
        "SurveyPulseEcho",
        LegacySurveyPulseEcho,
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=180).run()

    assert not app.exception
    assert any(
        "different deployment revisions" in item.value for item in app.error
    )


def test_first_load_shows_labware_and_estimated_stack_preview() -> None:
    app = _app()

    assert app.selectbox[0].label == "Labcyte plate"
    assert app.selectbox[0].value == "PP-0200"
    filling_input = next(
        item for item in app.radio if item.label == "Filling input"
    )
    assert filling_input.value == "Height [mm]"
    fill_height = next(
        item
        for item in app.number_input
        if item.label == "Liquid fill height [mm]"
    )
    assert fill_height.value == pytest.approx(4.22)
    assert any("55.87 µL per well" in item.value for item in app.caption)
    assert len(_markdown_containing(app, "Acoustic stack cross-section")) == 1
    assert len(_markdown_containing(app, "Estimated ray focus")) == 1
    assert any("not the ASM solution" in item.value for item in app.caption)
    assert any("local catalogue-scaled cutaway" in item.value for item in app.caption)
    assert any("physical plate continues" in item.value for item in app.caption)
    assert any(
        item.label == "Fill-height sensitivity ± [mm]"
        for item in app.number_input
    )
    assert len(app.image) == 1
    assert len(app.tabs) == 0
    assert not any(
        button.label == "Apply survey values to inputs"
        for button in app.button
    )


def test_survey_upload_previews_then_copies_safe_values() -> None:
    app = _app()

    app.file_uploader[0].set_value(_survey_upload()).run()

    assert not app.exception
    assert _number_input(app, "Water gap to plate [mm]").value == pytest.approx(
        25.3
    )
    assert _number_input(app, "Liquid fill height [mm]").value == pytest.approx(
        4.22
    )
    assert _selectbox(app, "Geometry source").value == (
        "Survey TOF · keep manual water gap"
    )
    assert _button(app, "Apply survey values to inputs")
    assert _markdown_containing(app, "Survey → inputs")

    _button(app, "Apply survey values to inputs").click().run()

    fluid_speed_m_s = dmso_water_properties(
        0.80,
        basis="volume",
        temperature_c=22.0,
    ).sound_speed_m_s
    expected_height_mm = fluid_speed_m_s * 1.5e-6 * 0.5 * 1e3
    assert not app.exception
    assert _selectbox(app, "Labcyte plate").value == "PP-0200-BC"
    assert _selectbox(app, "Geometry source").value == "Manual geometry"
    assert _number_input(app, "Water gap to plate [mm]").value == pytest.approx(
        25.3
    )
    assert _number_input(
        app, "Plate bottom thickness [mm]"
    ).value == pytest.approx(0.65)
    assert _number_input(
        app, "Liquid fill height [mm]"
    ).value == pytest.approx(expected_height_mm)
    assert expected_height_mm != pytest.approx(1.125)
    assert _number_input(
        app, "Excitation frequency [MHz]"
    ).value == pytest.approx(8.0)
    assert _number_input(app, "Pulse cycles").value == pytest.approx(2.0)
    assert any("Survey values copied" in item.value for item in app.success)
    assert len(app.tabs) == 0


def test_survey_timestamp_button_overwrites_complete_geometry_only() -> None:
    app = _app()
    app.file_uploader[0].set_value(_survey_upload()).run()

    _button(app, "Calculate all distances from timestamps").click().run()

    plate = get_labcyte_plate("PP-0200-BC")
    expected_water_mm = (
        0.5 * water_properties(22.0).sound_speed_m_s * 12.0e-6 * 1e3
    )
    expected_plate_mm = (
        0.5 * plate.inferred_longitudinal_speed_m_s * 0.5e-6 * 1e3
    )
    fluid_speed_m_s = dmso_water_properties(
        0.80,
        basis="volume",
        temperature_c=22.0,
    ).sound_speed_m_s
    expected_fill_mm = 0.5 * fluid_speed_m_s * 1.5e-6 * 1e3
    expected_volume_ul = plate.estimated_fill_volume_ul(expected_fill_mm)

    assert not app.exception
    assert _selectbox(app, "Labcyte plate").value == "PP-0200-BC"
    assert _selectbox(app, "Geometry source").value == "Manual geometry"
    assert _number_input(
        app, "Water gap to plate [mm]"
    ).value == pytest.approx(expected_water_mm)
    assert _number_input(
        app, "Plate bottom thickness [mm]"
    ).value == pytest.approx(expected_plate_mm)
    assert _number_input(
        app, "Liquid fill height [mm]"
    ).value == pytest.approx(expected_fill_mm)
    assert _number_input(app, "DMSO [vol.%]").value == pytest.approx(80.0)
    assert _number_input(app, "Temperature [°C]").value == pytest.approx(22.0)
    assert _number_input(
        app, "Excitation frequency [MHz]"
    ).value == pytest.approx(10.0)
    assert _number_input(app, "Pulse cycles").value == pytest.approx(1.0)
    assert any(
        f"{expected_fill_mm:.3f} mm" in item.value
        and f"{expected_volume_ul:.2f} µL per well" in item.value
        for item in app.caption
    )
    assert expected_water_mm != pytest.approx(9.0)
    assert expected_plate_mm != pytest.approx(0.65)
    assert expected_fill_mm != pytest.approx(1.125)
    assert any("Timestamp-derived" in item.value for item in app.success)


def test_survey_actions_render_in_json_section() -> None:
    app = _app()
    app.file_uploader[0].set_value(_survey_upload()).run()

    expected_labels = {
        "Apply survey values to inputs",
        "Calculate all distances from timestamps",
    }
    first_divider_index = min(
        index
        for index, child in app.sidebar.children.items()
        if child.type == "divider"
    )
    matching_indices: list[int] = []
    for index, child in app.sidebar.children.items():
        try:
            labels = {button.label for button in child.button}
        except AttributeError:
            continue
        if expected_labels <= labels:
            matching_indices.append(index)

    assert len(matching_indices) == 1
    assert matching_indices[0] < first_divider_index


def test_survey_all_distances_mode_can_copy_water_gap_explicitly() -> None:
    app = _app()
    app.file_uploader[0].set_value(_survey_upload()).run()

    _selectbox(app, "Geometry source").set_value(
        "Survey metadata · all distances"
    ).run()
    _button(app, "Apply survey values to inputs").click().run()

    assert not app.exception
    assert _number_input(app, "Water gap to plate [mm]").value == pytest.approx(
        9.0
    )
    assert _selectbox(app, "Geometry source").value == "Manual geometry"


def test_survey_copy_keeps_volume_and_height_inputs_synchronized() -> None:
    app = _app()
    filling_input = next(
        item for item in app.radio if item.label == "Filling input"
    )
    filling_input.set_value("Volume [µL]").run()
    app.file_uploader[0].set_value(_survey_upload()).run()

    _button(app, "Apply survey values to inputs").click().run()

    fluid_speed_m_s = dmso_water_properties(
        0.80,
        basis="volume",
        temperature_c=22.0,
    ).sound_speed_m_s
    expected_height_mm = fluid_speed_m_s * 1.5e-6 * 0.5 * 1e3
    expected_volume_ul = get_labcyte_plate(
        "PP-0200-BC"
    ).estimated_fill_volume_ul(expected_height_mm)
    assert not app.exception
    assert _number_input(
        app, "Liquid fill volume [µL]"
    ).value == pytest.approx(expected_volume_ul)
    assert any(
        f"{expected_height_mm:.3f} mm" in item.value
        and f"{expected_volume_ul:.2f} µL per well" in item.value
        for item in app.caption
    )


def test_filling_can_be_entered_as_volume_and_keeps_both_units_visible() -> None:
    app = _app()
    filling_input = next(
        item for item in app.radio if item.label == "Filling input"
    )

    filling_input.set_value("Volume [µL]").run()

    assert not app.exception
    fill_volume = next(
        item
        for item in app.number_input
        if item.label == "Liquid fill volume [µL]"
    )
    assert fill_volume.value == pytest.approx(55.87, abs=0.01)
    assert any(
        "4.220 mm" in item.value and "55.87 µL per well" in item.value
        for item in app.caption
    )

    fill_volume.set_value(65.0).run()

    assert not app.exception
    assert any(
        "4.893 mm" in item.value and "65.00 µL per well" in item.value
        for item in app.caption
    )

    filling_input = next(
        item for item in app.radio if item.label == "Filling input"
    )
    filling_input.set_value("Height [mm]").run()

    assert not app.exception
    fill_height = next(
        item
        for item in app.number_input
        if item.label == "Liquid fill height [mm]"
    )
    assert fill_height.value == pytest.approx(4.8925, abs=0.0006)


def test_submit_replaces_preview_with_one_exact_asm_stack() -> None:
    app = _app()
    submit = next(
        button
        for button in app.button
        if button.label == "Simulate and optimize focus"
    )

    submit.click().run(timeout=180)

    assert not app.exception
    assert len(_markdown_containing(app, "Acoustic stack cross-section")) == 1
    assert len(_markdown_containing(app, "Calculated angular-spectrum focus")) == 1
    assert not _markdown_containing(app, "Estimated ray focus")
    assert len(_markdown_containing(app, "Cavity-return exposure")) == 1
    assert len(_markdown_containing(app, "Liquid filling")) == 1
    assert len(_markdown_containing(app, "55.87 µL")) >= 1
    assert len(_markdown_containing(app, "narrow-band separated-pass")) == 1
    assert len(_markdown_containing(app, "acoustic overlap unverified")) == 1
    assert any("most recent simulation" in item.value for item in app.caption)
    assert any(
        "always frames all three interface reflections" in item.value
        for item in app.caption
    )
    assert [tab.label for tab in app.tabs] == [
        "Pulse response",
        "Focus optimization",
        "Spectrum & exports",
    ]

    fill_height = next(
        item
        for item in app.number_input
        if item.label == "Liquid fill height [mm]"
    )
    fill_height.set_value(4.30).run()

    assert not app.exception
    assert any(
        "sidebar contains pending values" in item.value for item in app.info
    )


def test_scientific_instrument_theme_uses_hamilton_navy_and_green() -> None:
    app = _app()
    stylesheet = app.markdown[0].value.lower()

    assert "--navy: #1c2d57" in stylesheet
    assert "--accent: #00f091" in stylesheet
    assert "--accent-soft: #e8fff6" in stylesheet
    assert "border-radius: .3rem" in stylesheet
    assert '[data-testid="stformsubmitbutton"] > button p' in stylesheet
    assert "color: #ffffff" in stylesheet
