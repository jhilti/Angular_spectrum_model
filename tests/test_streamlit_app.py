"""App-level regression tests for the interactive Streamlit flow."""

from pathlib import Path

import pytest


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
