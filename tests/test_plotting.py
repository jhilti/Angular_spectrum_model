"""Tests for view-window helpers used by the Streamlit plots."""

import numpy as np
import pytest

from angular_spectrum.plotting import (
    interface_reflection_window_us,
    modeled_echo_right_padding_us,
)


SIMULATED = {
    "Water–plate": 33.9972,
    "Plate–DMSO": 34.5682,
    "DMSO–air": 39.7398,
}


def test_interface_window_frames_all_three_simulated_reflections() -> None:
    lower, upper = interface_reflection_window_us(SIMULATED)

    assert lower == pytest.approx(33.1972)
    assert upper == pytest.approx(40.5398)
    assert lower > 0.0


def test_interface_window_also_contains_survey_picks() -> None:
    measured = {
        "Water–plate": 34.75,
        "Plate–DMSO": 35.32,
        "DMSO–air": 40.91,
    }

    lower, upper = interface_reflection_window_us(SIMULATED, measured)

    assert lower < min(SIMULATED.values())
    assert upper > max(measured.values())


def test_interface_window_rejects_incomplete_arrivals() -> None:
    with pytest.raises(ValueError, match="DMSO–air"):
        interface_reflection_window_us(
            {"Water–plate": 1.0, "Plate–DMSO": 2.0}
        )


def test_interface_window_never_precedes_excitation_start() -> None:
    arrivals = {
        "Water–plate": 0.1,
        "Plate–DMSO": 0.2,
        "DMSO–air": 0.4,
    }

    lower, upper = interface_reflection_window_us(arrivals)

    assert lower == 0.0
    assert upper > max(arrivals.values())


def test_interface_window_retains_required_long_pulse_padding() -> None:
    lower, upper = interface_reflection_window_us(
        SIMULATED,
        minimum_right_padding_us=4.2,
    )

    assert lower == pytest.approx(33.1972)
    assert upper == pytest.approx(SIMULATED["DMSO–air"] + 4.2)


def test_modeled_echo_padding_includes_drive_and_ringdown_margin() -> None:
    time_us = np.linspace(0.0, 20.0, 2001)
    envelope = np.zeros_like(time_us)

    padding = modeled_echo_right_padding_us(
        time_us,
        envelope,
        last_arrival_us=10.0,
        drive_duration_us=10.0 / 3.0,
    )

    assert padding == pytest.approx(10.0 / 3.0 + 0.8)


def test_modeled_echo_padding_retains_contiguous_response_tail() -> None:
    time_us = np.linspace(0.0, 20.0, 2001)
    envelope = np.zeros_like(time_us)
    envelope[(time_us >= 10.0) & (time_us < 12.0)] = 1.0

    padding = modeled_echo_right_padding_us(
        time_us,
        envelope,
        last_arrival_us=10.0,
        drive_duration_us=0.1,
        quiet_interval_us=0.25,
    )

    assert padding == pytest.approx(2.0)
