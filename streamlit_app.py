"""Interactive pulse-echo and focus dashboard."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from io import BytesIO, StringIO
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from angular_spectrum import (
    ReferenceTransferCalibration,
    SurveyPulseEcho,
    apply_reference_transfer,
    dmso_water_properties,
    estimate_reference_transfer,
    interpret_survey_geometry,
    parse_survey_pulse_echo,
    water_properties,
)
from angular_spectrum.app_model import (
    NUMERICAL_PRESETS,
    PP_DENSITY_KG_M3,
    PP_POISSON_RATIO,
    InteractiveSimulationResult,
    SimulationInputs,
    analytic_envelope,
    run_interactive_simulation,
)
from angular_spectrum.labware import (
    DEFAULT_LABCYTE_PLATE_ID,
    get_labcyte_plate,
    labcyte_plate_choice_ids,
    labcyte_plate_choice_label,
)
from angular_spectrum.plotting import (
    interface_reflection_window_us,
    modeled_echo_right_padding_us,
)
from angular_spectrum.schematic import (
    acoustic_stack_geometry,
    acoustic_stack_schematic_figure,
    refracted_ray_preview,
)


COLORS = {
    "ink": "#1c2d57",
    "blue": "#176c82",
    "red": "#cf5f4b",
    "green": "#007a4e",
    "gold": "#e77925",
    "muted": "#667085",
    "grid": "#d6dbe2",
    "paper": "#ffffff",
}

GEOMETRY_MANUAL = "Manual geometry"
GEOMETRY_SURVEY_KEEP_WATER = "Survey TOF · keep manual water gap"
GEOMETRY_SURVEY_ALL = "Survey metadata · all distances"
GEOMETRY_MODES = (
    GEOMETRY_MANUAL,
    GEOMETRY_SURVEY_KEEP_WATER,
    GEOMETRY_SURVEY_ALL,
)

COC_DENSITY_KG_M3 = 1020.0
COC_POISSON_RATIO_ASSUMPTION = 0.40


def _plate_material_defaults(material: str) -> tuple[str, float, float]:
    """Return explicit elastic defaults for one catalogue material."""

    if material == "polypropylene":
        return "polypropylene", PP_DENSITY_KG_M3, PP_POISSON_RATIO
    if material == "coc":
        return (
            "cyclic olefin copolymer",
            COC_DENSITY_KG_M3,
            COC_POISSON_RATIO_ASSUMPTION,
        )
    raise ValueError(f"unsupported catalogue plate material: {material}")


@dataclass(frozen=True)
class DisplaySignals:
    """Signals shown after an optional in-situ survey correction."""

    received: np.ndarray
    plate: np.ndarray
    surface: np.ndarray
    envelope: np.ndarray
    spectrum_db: np.ndarray
    reference_calibration: ReferenceTransferCalibration | None
    calibration_error: str | None


st.set_page_config(
    page_title="Pulse Echo Focus Lab",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --ink: #1c2d57;
        --ink-soft: #475467;
        --navy: #1c2d57;
        --navy-deep: #13203f;
        --accent: #00f091;
        --accent-deep: #007a4e;
        --accent-soft: #e8fff6;
        --paper: #f7f8fa;
        --surface: #ffffff;
        --surface-soft: #f0f2f5;
        --line: #d6dbe2;
        --line-strong: #b8c0cc;
        --shadow: 0 5px 18px rgba(16, 24, 40, .055);
    }
    html, body, [class*="css"] {
        font-family: Outfit, "Avenir Next", Inter, ui-sans-serif, -apple-system,
            BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .stApp {
        background: var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(247, 248, 250, .94);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--line);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1280px;
        padding-top: 2.4rem;
        padding-bottom: 5rem;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebarContent"] {
        padding-top: 1.35rem;
    }
    [data-testid="stSidebar"] [data-testid="stForm"] {
        border: 0;
        padding: 0;
        background: transparent;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--line);
        margin: .85rem 0 1rem;
    }
    [data-testid="stSidebar"] label {
        color: #344054;
        font-size: .82rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff;
        border-color: var(--line-strong);
        border-radius: .3rem;
        min-height: 2.65rem;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] > div:focus-within,
    [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
        border-color: var(--accent);
        box-shadow: 0 0 0 2px rgba(0, 240, 145, .18);
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #f8f9fb;
        border: 1px dashed var(--line-strong);
        border-radius: .35rem;
        padding: .7rem;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
        background: #ffffff;
    }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button {
        border: 0;
        border-radius: .3rem;
        min-height: 2.9rem;
        background: var(--navy);
        color: #ffffff;
        font-weight: 700;
        letter-spacing: .005em;
        box-shadow: 0 3px 8px rgba(28, 45, 87, .18);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        background: var(--navy-deep);
        color: #ffffff;
        box-shadow: 0 5px 12px rgba(28, 45, 87, .22);
    }
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button p {
        color: #ffffff;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -.025em;
    }
    p, li {
        color: var(--ink-soft);
    }
    a {
        color: var(--accent-deep);
    }
    .app-hero {
        position: relative;
        overflow: hidden;
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(250px, .75fr);
        gap: 3rem;
        align-items: center;
        margin: 0 0 1.7rem;
        padding: 2.8rem 3rem;
        border: 1px solid var(--line);
        border-top: 3px solid var(--accent);
        border-radius: .4rem;
        background: #ffffff;
        box-shadow: var(--shadow);
    }
    .app-hero::after {
        content: "";
        position: absolute;
        width: 12rem;
        height: 1px;
        right: -2.5rem;
        top: 2.4rem;
        background: var(--accent);
        opacity: .22;
        transform: rotate(-35deg);
        pointer-events: none;
    }
    .hero-copy,
    .layer-card {
        position: relative;
        z-index: 1;
    }
    .eyebrow {
        margin-bottom: .7rem;
        color: var(--accent-deep);
        font-size: .8rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .app-hero h1 {
        max-width: 760px;
        margin: 0 0 .8rem;
        color: var(--navy);
        font-size: clamp(2.15rem, 4vw, 3.45rem);
        line-height: 1.02;
        letter-spacing: -.055em;
    }
    .hero-copy > p {
        max-width: 720px;
        margin: 0;
        color: #667085;
        font-size: 1rem;
        line-height: 1.65;
    }
    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: .5rem;
        margin-top: 1.2rem;
    }
    .hero-tag {
        padding: .38rem .64rem;
        border: 1px solid var(--line);
        border-radius: .25rem;
        background: #f7f8fa;
        color: #475467;
        font-size: .8rem;
        font-weight: 600;
    }
    .layer-card {
        padding: 1.25rem;
        border: 1px solid var(--line);
        border-radius: .35rem;
        background: #f8f9fb;
    }
    .layer-card-label {
        margin-bottom: .75rem;
        color: #667085;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .layer-flow {
        display: grid;
        grid-template-columns: 1fr auto 1fr auto 1.2fr auto .8fr;
        align-items: center;
        gap: .35rem;
    }
    .layer {
        padding: .55rem .35rem;
        border-radius: .2rem;
        color: var(--navy);
        font-size: .8rem;
        font-weight: 700;
        text-align: center;
    }
    .layer.water { background: #dcecf2; }
    .layer.pp { background: #eff0f0; }
    .layer.dmso { background: #b2fade; }
    .layer.air { background: #e9ecf0; }
    .layer-arrow { color: #98a2b3; font-size: .8rem; }
    .hero-foot {
        display: flex;
        align-items: center;
        gap: .5rem;
        margin-top: .85rem;
        color: #667085;
        font-size: .78rem;
    }
    .pulse-dot {
        width: .45rem;
        height: .45rem;
        border-radius: .05rem;
        background: var(--accent);
        box-shadow: 0 0 0 .22rem rgba(0, 240, 145, .12);
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: .75rem;
        margin: .2rem 0 1rem;
    }
    .sidebar-mark {
        display: grid;
        place-items: center;
        width: 2.45rem;
        height: 2.45rem;
        border-radius: .25rem;
        background: var(--navy);
        color: white;
        font-size: 1rem;
        box-shadow: none;
    }
    .sidebar-brand strong {
        display: block;
        color: var(--ink);
        font-size: .95rem;
        letter-spacing: -.02em;
    }
    .sidebar-brand span {
        display: block;
        margin-top: .1rem;
        color: #6d8084;
        font-size: .76rem;
    }
    .sidebar-kicker {
        margin: 1.15rem 0 .65rem;
        color: #667085;
        font-size: .78rem;
        font-weight: 760;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .survey-status {
        margin: -.1rem 0 .85rem;
        padding: .62rem .72rem;
        border: 1px solid #b2fade;
        border-radius: .3rem;
        background: var(--accent-soft);
        color: var(--accent-deep);
        font-size: .8rem;
    }
    .section-head {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 1.5rem;
        margin: 2.35rem 0 1rem;
    }
    .section-head .section-kicker {
        margin-bottom: .28rem;
        color: var(--accent-deep);
        font-size: .78rem;
        font-weight: 760;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .section-head h2 {
        margin: 0;
        font-size: 1.45rem;
        letter-spacing: -.035em;
    }
    .section-head p {
        max-width: 590px;
        margin: 0;
        color: #6a7c80;
        font-size: .82rem;
        line-height: 1.5;
        text-align: right;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .9rem;
        margin-bottom: 1.15rem;
    }
    .metric-card {
        min-height: 8.1rem;
        padding: 1.15rem 1.2rem 1rem;
        border: 1px solid var(--line);
        border-radius: .35rem;
        background: #ffffff;
        box-shadow: none;
    }
    .metric-label {
        color: #667b80;
        font-size: .78rem;
        font-weight: 720;
        letter-spacing: .075em;
        text-transform: uppercase;
    }
    .metric-value {
        margin-top: .72rem;
        color: var(--ink);
        font-size: clamp(1.2rem, 2vw, 1.65rem);
        font-weight: 720;
        letter-spacing: -.04em;
        line-height: 1.05;
    }
    .metric-hint {
        margin-top: .55rem;
        color: #789096;
        font-size: .72rem;
        line-height: 1.35;
    }
    .metric-hint.positive { color: var(--accent-deep); }
    .metric-hint.negative { color: #b54708; }
    .metric-card.reflection-card {
        grid-column: 1 / -1;
        min-height: auto;
        display: grid;
        grid-template-columns: minmax(220px, .72fr) minmax(0, 1.75fr);
        gap: 1.5rem;
        align-items: center;
        border-left: 3px solid var(--accent);
    }
    .reflection-card-copy {
        color: #5f7378;
        font-size: .78rem;
        line-height: 1.55;
    }
    .reflection-card-copy strong { color: var(--ink); }
    .model-note {
        display: flex;
        align-items: flex-start;
        gap: .75rem;
        margin: .35rem 0 1.2rem;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-left: 3px solid var(--accent);
        border-radius: .25rem;
        background: #ffffff;
        color: #475467;
        font-size: .82rem;
        line-height: 1.5;
    }
    .model-note::before {
        content: "i";
        flex: 0 0 auto;
        display: grid;
        place-items: center;
        width: 1.35rem;
        height: 1.35rem;
        margin-top: .05rem;
        border-radius: .2rem;
        background: var(--accent-soft);
        color: var(--accent-deep);
        font-size: .72rem;
        font-weight: 750;
    }
    .empty-state {
        margin-top: 1.25rem;
        padding: 1.3rem 1.5rem;
        border: 1px solid var(--line);
        border-left: 3px solid var(--accent);
        border-radius: .3rem;
        background: #ffffff;
        text-align: center;
    }
    .empty-mark {
        display: grid;
        place-items: center;
        width: 3rem;
        height: 3rem;
        margin: 0 auto .9rem;
        border-radius: .25rem;
        background: var(--accent-soft);
        color: var(--accent-deep);
        font-size: 1.2rem;
    }
    .empty-state h3 { margin: 0 0 .35rem; }
    .empty-state p {
        max-width: 480px;
        margin: 0 auto;
        color: #6a7d81;
        font-size: .86rem;
    }
    .focus-state {
        display: flex;
        align-items: flex-start;
        gap: .9rem;
        margin: 0 0 1rem;
        padding: .9rem 1rem;
        border: 1px solid var(--line);
        border-left: 3px solid var(--accent);
        border-radius: .3rem;
        background: #ffffff;
    }
    .focus-state.exact {
        border-left-color: var(--navy);
    }
    .focus-state-tag {
        flex: 0 0 auto;
        padding: .24rem .38rem;
        border-radius: .18rem;
        background: var(--accent-soft);
        color: var(--accent-deep);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .09em;
        text-transform: uppercase;
    }
    .focus-state.exact .focus-state-tag {
        background: #e9edf2;
        color: var(--navy);
    }
    .focus-state strong {
        display: block;
        margin-bottom: .15rem;
        color: var(--navy);
        font-size: .83rem;
    }
    .focus-state span:last-child {
        color: #667085;
        font-size: .8rem;
        line-height: 1.45;
    }
    [data-baseweb="tab-list"] {
        gap: .3rem;
        padding: .3rem;
        border: 1px solid var(--line);
        border-radius: .3rem;
        background: #eef0f3;
    }
    [data-baseweb="tab"] {
        height: 2.7rem;
        padding: 0 1rem;
        border-radius: .2rem;
        color: #5c7378;
        font-weight: 620;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: white;
        color: var(--ink);
        box-shadow: none;
    }
    [data-baseweb="tab-highlight"] {
        display: none;
    }
    div[data-testid="stPlotlyChart"],
    div[data-testid="stImage"],
    [data-testid="stDataFrame"],
    [data-testid="stPyplot"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: .35rem;
        background: var(--surface);
        box-shadow: var(--shadow);
    }
    .st-key-acoustic_stack_visual [data-testid="stPyplot"] {
        max-width: 760px;
        margin-inline: auto;
    }
    [data-testid="stDataFrame"] {
        margin-top: .25rem;
    }
    [data-testid="stAlert"] {
        border-radius: .3rem;
        border-width: 1px;
    }
    [data-testid="stDownloadButton"] > button {
        min-height: 2.8rem;
        border-color: #cbdad6;
        border-radius: .3rem;
        background: white;
        color: #23474e;
        font-weight: 620;
    }
    [data-testid="stDownloadButton"] > button:hover {
        border-color: var(--accent);
        color: var(--accent-deep);
        background: var(--accent-soft);
    }
    [data-testid="stExpander"] {
        border-color: var(--line);
        border-radius: .3rem;
        background: #ffffff;
    }
    .footer-note {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--line);
        color: #7b8d91;
        font-size: .72rem;
        text-align: center;
    }
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }
        .app-hero { grid-template-columns: 1fr; padding: 2rem; }
        .layer-card { display: none; }
        .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .metric-card.reflection-card { grid-template-columns: 1fr; gap: .65rem; }
        .section-head { display: block; }
        .section-head p { margin-top: .35rem; text-align: left; }
    }
    @media (max-width: 540px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem .85rem 3rem;
        }
        .app-hero { padding: 1.45rem; border-radius: .35rem; }
        .app-hero h1 { font-size: 2rem; }
        .metric-grid { grid-template-columns: 1fr; }
        .metric-card { min-height: auto; }
        .focus-state { display: block; }
        .focus-state-tag { display: inline-block; margin-bottom: .55rem; }
        [data-baseweb="tab"] { padding: 0 .55rem; font-size: .75rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_uploaded_survey(data: bytes) -> SurveyPulseEcho:
    document = json.loads(data.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("survey JSON root must be an object")
    return parse_survey_pulse_echo(document)


@st.cache_data(show_spinner=False, max_entries=12)
def simulate_cached(inputs: SimulationInputs) -> InteractiveSimulationResult:
    return run_interactive_simulation(inputs)


def _optional_mm(value_m: float | None) -> str:
    return "not stored" if value_m is None else f"{value_m * 1e3:.3f} mm"


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["paper"])
    axis.grid(
        visible=True,
        color=COLORS["grid"],
        linewidth=0.7,
        alpha=0.72,
    )
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#b9cbc7")
    axis.tick_params(colors=COLORS["muted"], labelsize=9)
    axis.xaxis.label.set_color(COLORS["muted"])
    axis.yaxis.label.set_color(COLORS["muted"])
    axis.title.set_color(COLORS["ink"])
    axis.title.set_fontweight(700)


def _used_geometry(
    manual_inputs: SimulationInputs,
    survey: SurveyPulseEcho | None,
    geometry_mode: str,
) -> tuple[SimulationInputs, str]:
    if survey is None or geometry_mode == GEOMETRY_MANUAL:
        return manual_inputs, "Manual geometry fields"
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(f"unknown geometry mode: {geometry_mode}")
    properties = dmso_water_properties(
        manual_inputs.dmso_volume_percent / 100.0,
        basis="volume",
        temperature_c=manual_inputs.temperature_c,
    )
    water = water_properties(manual_inputs.temperature_c)
    geometry = interpret_survey_geometry(
        survey,
        incident_sound_speed_m_s=water.sound_speed_m_s,
        plate_longitudinal_speed_m_s=(
            manual_inputs.plate_longitudinal_speed_m_s
        ),
        fluid_sound_speed_m_s=properties.sound_speed_m_s,
    )
    if geometry_mode == GEOMETRY_SURVEY_KEEP_WATER:
        water_path_m = manual_inputs.water_path_mm * 1e-3
        water_source = "manual water gap"
    else:
        water_path_m = (
            geometry.stored_probe_to_plate_m
            if geometry.stored_probe_to_plate_m is not None
            else geometry.tof_probe_to_plate_m
        )
        water_source = "stored/TOF water distance"
    plate_thickness_m = (
        geometry.stored_plate_thickness_m
        if geometry.stored_plate_thickness_m is not None
        else geometry.tof_plate_thickness_m
    )
    used = replace(
        manual_inputs,
        water_path_mm=water_path_m * 1e3,
        plate_thickness_mm=plate_thickness_m * 1e3,
        fluid_height_mm=geometry.tof_fluid_height_m * 1e3,
    )
    source = (
        f"Survey timing with {water_source}; plate thickness from stored/TOF "
        "timing and fluid height from the selected DMSO sound speed"
    )
    return used, source


def _measured_arrivals_since_excitation_us(
    survey: SurveyPulseEcho,
) -> dict[str, float]:
    return {
        "Water–plate": survey.water_pp_time_s * 1e6,
        "Plate–DMSO": survey.pp_fluid_time_s * 1e6,
        "DMSO–air": survey.fluid_top_time_s * 1e6,
    }


def local_waveform_correlation(
    measured_time_s: np.ndarray,
    measured_signal: np.ndarray,
    simulated_time_s: np.ndarray,
    simulated_signal: np.ndarray,
    *,
    measured_arrival_s: float,
    simulated_arrival_s: float,
    half_window_s: float = 0.30e-6,
    maximum_lag_s: float = 0.15e-6,
) -> float:
    """Return the best local normalized correlation around one echo."""

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
    best = -1.0
    for lag_s in lags:
        candidate = np.interp(
            local_time - lag_s,
            simulated_local,
            simulated_signal,
            left=0.0,
            right=0.0,
        )
        candidate -= float(np.mean(candidate))
        denominator = float(np.linalg.norm(measured) * np.linalg.norm(candidate))
        if denominator > 0.0:
            best = max(best, float(np.dot(measured, candidate) / denominator))
    return best


def display_signals(
    result: InteractiveSimulationResult,
    survey: SurveyPulseEcho | None,
    use_reference_calibration: bool,
) -> DisplaySignals:
    """Return raw or water–plate-referenced traces for display."""

    received = result.received_normalized.copy()
    plate = result.plate_normalized.copy()
    surface = result.surface_normalized.copy()
    calibration = None
    calibration_error = None
    if survey is not None and use_reference_calibration:
        simulation_time_s = result.time_since_excitation_us * 1e-6
        try:
            calibration = estimate_reference_transfer(
                survey.time_since_excitation_s,
                survey.normalized_signal,
                simulation_time_s,
                received,
                measured_arrival_s=survey.water_pp_time_s,
                simulated_arrival_s=result.arrivals.water_pp_s,
                target_time_s=simulation_time_s,
            )
            received = apply_reference_transfer(
                simulation_time_s,
                received,
                calibration,
            )
            plate = apply_reference_transfer(
                simulation_time_s,
                plate,
                calibration,
            )
            surface = apply_reference_transfer(
                simulation_time_s,
                surface,
                calibration,
            )
        except ValueError as exc:
            calibration = None
            calibration_error = str(exc)
    scale = max(float(np.max(np.abs(received))), 1e-30)
    received /= scale
    plate /= scale
    surface /= scale
    envelope = analytic_envelope(received)
    envelope /= max(float(np.max(envelope)), 1e-30)
    spectrum = np.abs(np.fft.rfft(received))
    spectrum /= max(float(np.max(spectrum)), 1e-30)
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-6))
    return DisplaySignals(
        received=received,
        plate=plate,
        surface=surface,
        envelope=envelope,
        spectrum_db=spectrum_db,
        reference_calibration=calibration,
        calibration_error=calibration_error,
    )


def pulse_figure(
    result: InteractiveSimulationResult,
    survey: SurveyPulseEcho | None,
    signals: DisplaySignals,
) -> plt.Figure:
    arrivals_since_excitation = result.arrivals.since_excitation_us
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 6.8),
        sharex=False,
        constrained_layout=True,
    )
    figure.patch.set_facecolor(COLORS["paper"])
    axes[0].plot(
        result.time_since_excitation_us,
        signals.received,
        color=COLORS["blue"],
        linewidth=1.15,
        label=(
            "ASM simulation · water–plate referenced"
            if signals.reference_calibration is not None
            else "ASM simulation"
        ),
    )
    measured_relative_us = None
    measured_envelope = None
    if survey is not None:
        measured_relative_us = survey.time_since_excitation_s * 1e6
        axes[0].plot(
            measured_relative_us,
            survey.normalized_signal,
            color=COLORS["ink"],
            linewidth=0.9,
            alpha=0.68,
            label="Survey ADC (independently normalized)",
        )
        measured_envelope = analytic_envelope(survey.normalized_signal)
        measured_envelope /= max(float(np.max(measured_envelope)), 1e-30)

    marker_styles = {
        "Water–plate": ("--", COLORS["muted"]),
        "Plate–DMSO": (":", COLORS["red"]),
        "DMSO–air": ("-.", COLORS["green"]),
    }
    axes[0].axvline(
        0.0,
        color=COLORS["gold"],
        linestyle="-",
        linewidth=1.0,
        alpha=0.8,
        label="Excitation start",
    )
    for name, arrival_us in arrivals_since_excitation.items():
        linestyle, color = marker_styles[name]
        axes[0].axvline(
            arrival_us,
            color=color,
            linestyle=linestyle,
            linewidth=1.15,
            label=f"Simulated {name}",
        )
    if survey is not None:
        for name, arrival_us in (
            _measured_arrivals_since_excitation_us(survey).items()
        ):
            _, color = marker_styles[name]
            axes[0].axvline(
                arrival_us,
                color=color,
                linestyle=(0, (1, 2)),
                linewidth=1.0,
                alpha=0.68,
            )

    axes[0].set_ylabel("RF signal [relative]")
    axes[0].set_title("Received pulse echo at the transmitting aperture")
    axes[0].legend(
        loc="upper right",
        ncols=2,
        fontsize=8.2,
        frameon=False,
    )

    axes[1].fill_between(
        result.time_since_excitation_us,
        0.0,
        signals.envelope,
        color=COLORS["red"],
        alpha=0.08,
    )
    axes[1].plot(
        result.time_since_excitation_us,
        signals.envelope,
        color=COLORS["red"],
        linewidth=1.35,
        label="Simulation envelope",
    )
    axes[1].plot(
        result.time_since_excitation_us,
        np.abs(signals.plate),
        color=COLORS["blue"],
        linewidth=0.8,
        alpha=0.46,
        label="Plate component",
    )
    axes[1].plot(
        result.time_since_excitation_us,
        np.abs(signals.surface),
        color=COLORS["green"],
        linewidth=0.9,
        alpha=0.75,
        label="DMSO–air component",
    )
    if measured_relative_us is not None and measured_envelope is not None:
        axes[1].plot(
            measured_relative_us,
            measured_envelope,
            color=COLORS["ink"],
            linewidth=1.05,
            alpha=0.65,
            label="Survey envelope",
        )
    for name, arrival_us in arrivals_since_excitation.items():
        linestyle, color = marker_styles[name]
        axes[1].axvline(
            arrival_us,
            color=color,
            linestyle=linestyle,
            linewidth=1.0,
        )
    axes[1].set(
        xlabel="Time since excitation start [µs]",
        ylabel="Envelope / component [relative]",
    )
    axes[1].legend(
        loc="upper right",
        ncols=2,
        fontsize=8.2,
        frameon=False,
    )

    measured_arrivals = (
        None
        if survey is None
        else _measured_arrivals_since_excitation_us(survey)
    )
    last_simulated_arrival_us = max(arrivals_since_excitation.values())
    drive_duration_us = (
        result.inputs.excitation_cycles
        / result.inputs.excitation_frequency_mhz
    )
    right_padding_us = modeled_echo_right_padding_us(
        result.time_since_excitation_us,
        analytic_envelope(signals.surface),
        last_arrival_us=last_simulated_arrival_us,
        drive_duration_us=drive_duration_us,
    )
    echo_xlim = interface_reflection_window_us(
        arrivals_since_excitation,
        measured_arrivals,
        minimum_right_padding_us=right_padding_us,
    )
    axes[0].set_xlim(*echo_xlim)
    axes[0].set_xlabel("Time since excitation start [µs]")
    axes[1].set_xlim(*echo_xlim)
    for axis in axes:
        _style_axis(axis)
    return figure


def focus_figure(result: InteractiveSimulationResult) -> plt.Figure:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.5),
        constrained_layout=True,
    )
    figure.patch.set_facecolor(COLORS["paper"])
    axes[0].fill_between(
        result.axial_position_after_pp_mm,
        0.0,
        result.axial_intensity_normalized,
        color=COLORS["blue"],
        alpha=0.09,
    )
    axes[0].plot(
        result.axial_position_after_pp_mm,
        result.axial_intensity_normalized,
        color=COLORS["blue"],
        linewidth=1.7,
    )
    axes[0].axvline(
        result.inputs.fluid_height_mm,
        color=COLORS["green"],
        linestyle="--",
        label="Meniscus",
    )
    axes[0].axvline(
        result.focus_after_pp_mm,
        color=COLORS["gold"],
        linestyle=":",
        label="Calculated focus",
    )
    axes[0].set(
        xlabel="Position after plate [mm]",
        ylabel="Relative on-axis |p|² [normalized]",
        title="Monochromatic current focus",
    )
    axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].fill_between(
        result.water_path_scan_mm,
        0.0,
        result.meniscus_intensity_normalized,
        color=COLORS["green"],
        alpha=0.09,
    )
    axes[1].plot(
        result.water_path_scan_mm,
        result.meniscus_intensity_normalized,
        color=COLORS["green"],
        linewidth=1.7,
    )
    axes[1].axvline(
        result.inputs.water_path_mm,
        color=COLORS["muted"],
        linestyle="--",
        label="Current water gap",
    )
    axes[1].axvline(
        result.optimal_water_path_mm,
        color=COLORS["gold"],
        linestyle=":",
        label="Best water gap",
    )
    axes[1].set(
        xlabel="Water gap to plate [mm]",
        ylabel="Relative meniscus |p|² [normalized]",
        title="Monochromatic focus optimization",
    )
    axes[1].legend(frameon=False, fontsize=8.5)
    for axis in axes:
        _style_axis(axis)
    return figure


def spectrum_figure(
    result: InteractiveSimulationResult,
    signals: DisplaySignals,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9.5, 4.0), constrained_layout=True)
    figure.patch.set_facecolor(COLORS["paper"])
    mask = (
        (result.frequency_mhz >= 1.0)
        & (result.frequency_mhz <= 24.0)
    )
    axis.plot(
        result.frequency_mhz[mask],
        signals.spectrum_db[mask],
        color=COLORS["blue"],
        linewidth=1.35,
    )
    axis.fill_between(
        result.frequency_mhz[mask],
        -60.0,
        signals.spectrum_db[mask],
        color=COLORS["blue"],
        alpha=0.08,
    )
    axis.axhline(-6.0, color=COLORS["muted"], linestyle="--", linewidth=1.0)
    axis.set(
        xlabel="Frequency [MHz]",
        ylabel="Received spectrum [dB, normalized]",
        ylim=(-60.0, 2.0),
        title="Simulated received spectrum",
    )
    _style_axis(axis)
    return figure


def figure_png(figure: plt.Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    return buffer.getvalue()


def result_csv(
    result: InteractiveSimulationResult,
    survey: SurveyPulseEcho | None,
    signals: DisplaySignals,
) -> bytes:
    measured = np.full(result.time_since_excitation_us.shape, np.nan)
    if survey is not None:
        measured_time_us = survey.time_since_excitation_s * 1e6
        inside = (
            (result.time_since_excitation_us >= measured_time_us[0])
            & (result.time_since_excitation_us <= measured_time_us[-1])
        )
        measured[inside] = np.interp(
            result.time_since_excitation_us[inside],
            measured_time_us,
            survey.normalized_signal,
        )
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "time_since_excitation_start_us",
            "simulation_raw_normalized",
            "simulation_displayed_normalized",
            "displayed_envelope_normalized",
            "displayed_pp_plate_component_normalized",
            "displayed_dmso_air_component_normalized",
            "survey_adc_normalized_interpolated",
        ]
    )
    writer.writerows(
        zip(
            result.time_since_excitation_us,
            result.received_normalized,
            signals.received,
            signals.envelope,
            signals.plate,
            signals.surface,
            measured,
        )
    )
    return output.getvalue().encode("utf-8")


def result_summary(
    result: InteractiveSimulationResult,
    survey: SurveyPulseEcho | None,
    geometry_source: str,
    signals: DisplaySignals,
) -> dict[str, Any]:
    plate = get_labcyte_plate(result.inputs.plate_part_number)
    return {
        "inputs": asdict(result.inputs),
        "geometry_source": geometry_source,
        "labware": {
            "part_number": plate.id,
            "name": plate.name,
            "guid": plate.guid,
            "family": plate.family,
            "material": plate.material,
            "well_count": plate.well_count,
            "well_depth_mm": plate.well_depth_mm,
            "well_top_width_mm": plate.well_top_width_mm,
            "well_bottom_width_mm": plate.well_bottom_width_mm,
            "well_pitch_mm": plate.well_pitch_mm,
            "well_volume_ul": plate.well_volume_ul,
            "catalogue_bottom_thickness_mm": plate.bottom_thickness_mm,
            "simulation_bottom_thickness_mm": (
                result.inputs.plate_thickness_mm
            ),
            "catalogue_raw_longitudinal_speed": (
                plate.raw_longitudinal_speed
            ),
            "catalogue_inferred_longitudinal_speed_m_s": (
                plate.inferred_longitudinal_speed_m_s
            ),
            "source_url": plate.source_url,
            "limitations": list(plate.limitations),
        },
        "water_properties": asdict(result.water_properties),
        "dmso_properties": asdict(result.dmso_properties),
        "arrivals_relative_to_water_pp_us": (
            result.arrivals.relative_to_water_pp_us
        ),
        "arrivals_since_excitation_start_us": (
            result.arrivals.since_excitation_us
        ),
        "focus": {
            "focus_after_pp_mm": result.focus_after_pp_mm,
            "focus_from_aperture_mm": result.focus_from_aperture_mm,
            "focus_offset_from_meniscus_mm": (
                result.focus_offset_from_meniscus_mm
            ),
            "focus_scan_boundary_limited": (
                result.focus_scan_boundary_limited
            ),
            "optimal_water_path_mm_for_meniscus": (
                result.optimal_water_path_mm
            ),
            "optimal_meniscus_intensity_fwhm_mm": (
                result.optimal_meniscus_intensity_fwhm_mm
            ),
            "optimal_water_path_boundary_limited": (
                result.optimal_water_path_boundary_limited
            ),
            "predicted_gain_from_water_path_adjustment_db": (
                result.optimal_water_path_gain_db
            ),
            "criterion": (
                "maximum center-frequency single-pass on-axis pressure squared "
                "at the planar meniscus"
            ),
        },
        "meniscus_reflection_exposure": {
            "reference": "first forward pass across the meniscus plane",
            "frequency_hz": result.meniscus_cavity.frequency_hz,
            "electrical_overlap_regime": (
                result.meniscus_cavity.electrical_overlap_regime
            ),
            "electrical_burst_duration_us": (
                result.meniscus_cavity.electrical_burst_duration_s * 1e6
            ),
            "cavity_round_trip_us": (
                result.meniscus_cavity.cavity_round_trip_s * 1e6
            ),
            "narrowband_separated_pass_exposure_gain": (
                result.meniscus_cavity.narrowband_separated_pass_exposure_gain
            ),
            "narrowband_separated_pass_percent_change": (
                result.meniscus_cavity.narrowband_separated_pass_percent_change
            ),
            "coherent_cw_power_gain": (
                result.meniscus_cavity.coherent_power_gain
            ),
            "coherent_cw_percent_change": (
                result.meniscus_cavity.coherent_percent_change
            ),
            "fill_height_sensitivity_plus_minus_mm": (
                result.meniscus_cavity.height_uncertainty_m * 1e3
            ),
            "coherent_cw_gain_sensitivity_range": [
                result.meniscus_cavity.coherent_gain_height_min,
                result.meniscus_cavity.coherent_gain_height_max,
            ],
            "narrowband_separated_gain_sensitivity_range": [
                result.meniscus_cavity.narrowband_gain_height_min,
                result.meniscus_cavity.narrowband_gain_height_max,
            ],
            "cavity_orders_retained": (
                result.meniscus_cavity.cavity_orders_retained
            ),
            "cavity_series_converged": (
                result.meniscus_cavity.cavity_series_converged
            ),
            "sensitivity_series_converged": (
                result.meniscus_cavity.sensitivity_series_converged
            ),
            "cavity_relative_tolerance": (
                result.meniscus_cavity.cavity_relative_tolerance
            ),
            "absolute_energy_calibrated": False,
            "limitations": list(result.meniscus_cavity.limitations),
        },
        "survey": {
            "used": survey is not None,
            "date_time": None if survey is None else survey.date_time,
            "fluid_material": (
                None if survey is None else survey.fluid_material
            ),
            "probe_frequency_hz": (
                None if survey is None else survey.probe_frequency_hz
            ),
            "tone_length_cycles": (
                None if survey is None else survey.tone_length_cycles
            ),
            "probe_voltage_setting_v": (
                None if survey is None else survey.probe_voltage_setting_v
            ),
            "sample_index_start": (
                None if survey is None else survey.sample_index_start
            ),
            "excitation_metadata_is_calibrated": False,
            "absolute_adc_calibrated": False,
            "signals_normalized_independently": survey is not None,
            "water_pp_reference_calibration": {
                "applied": signals.reference_calibration is not None,
                "absolute_gain_calibrated": False,
                "calibration_error": signals.calibration_error,
                "gate_start_s": (
                    None
                    if signals.reference_calibration is None
                    else signals.reference_calibration.gate_start_s
                ),
                "gate_end_s": (
                    None
                    if signals.reference_calibration is None
                    else signals.reference_calibration.gate_end_s
                ),
                "minimum_frequency_hz": (
                    None
                    if signals.reference_calibration is None
                    else signals.reference_calibration.minimum_frequency_hz
                ),
                "maximum_frequency_hz": (
                    None
                    if signals.reference_calibration is None
                    else signals.reference_calibration.maximum_frequency_hz
                ),
            },
        },
        "numerics": {
            "simulated_frequency_bins": result.simulated_frequency_bin_count,
            "fluid_cavity_echo_count": result.fluid_cavity_echo_count,
            "preset": result.inputs.numerical_preset,
        },
        "assumptions": [
            "survey ADC amplitudes are qualitative and independently normalized",
            "uploaded survey data are parsed in memory and are not committed",
            "DMSO concentration and temperature are user hypotheses",
            "the focus optimization excludes DMSO-air cavity interference",
            (
                "the separate meniscus reflection metric is a relative "
                "forward-exposure proxy, not net energy into air"
            ),
            (
                "the transducer certificate magnitude is represented by a "
                "zero-phase asymmetric Gaussian; small symmetric pre-ringing "
                "is numerical/model uncertainty, not a physical precursor"
            ),
            (
                "a water-plate reference correction, when enabled, is a common "
                "qualitative system filter and not an ADC pressure calibration"
            ),
            (
                "plate and fluid attenuation are user inputs and default to zero"
            ),
            "the meniscus is planar and parallel to the plate",
        ],
    }


st.markdown(
    """
    <section class="app-hero">
        <div class="hero-copy">
            <div class="eyebrow">Ultrasound simulation workspace</div>
            <h1>Pulse Echo<br>Focus Lab</h1>
            <p>
                Explore broadband echoes, layered-media timing, and focal
                alignment in one focused acoustic workspace.
            </p>
            <div class="hero-tags">
                <span class="hero-tag">10 MHz broadband</span>
                <span class="hero-tag">Monostatic pulse echo</span>
                <span class="hero-tag">Elastic plate model</span>
            </div>
        </div>
        <div class="layer-card">
            <div class="layer-card-label">Acoustic path</div>
            <div class="layer-flow">
                <span class="layer water">H₂O</span>
                <span class="layer-arrow">→</span>
                <span class="layer pp">plate</span>
                <span class="layer-arrow">→</span>
                <span class="layer dmso">DMSO</span>
                <span class="layer-arrow">→</span>
                <span class="layer air">Air</span>
            </div>
            <div class="hero-foot">
                <span class="pulse-dot"></span>
                Angular-spectrum propagation with elastic plate response
            </div>
        </div>
    </section>
    <div class="model-note">
        Enter the known geometry, optionally add a survey JSON, and run the
        model. ADC amplitudes are independently normalized and remain
        qualitative.
    </div>
    """,
    unsafe_allow_html=True,
)

survey: SurveyPulseEcho | None = None
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-mark">◉</div>
            <div>
                <strong>Simulation setup</strong>
                <span>Configure the acoustic stack</span>
            </div>
        </div>
        <div class="sidebar-kicker">Optional measurement</div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Survey JSON",
        type=["json"],
        help=(
            "The file is parsed in memory. Raw ADC data are not saved by the app."
        ),
    )
    if uploaded_file is not None:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("The survey file must be smaller than 10 MB.")
        else:
            try:
                survey = parse_uploaded_survey(uploaded_file.getvalue())
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                st.error(f"Could not read this survey: {exc}")
    if survey is None:
        st.caption("Optional · JSON · up to 10 MB · processed in memory")
    else:
        st.markdown(
            '<div class="survey-status">Survey loaded · ready for overlay</div>',
            unsafe_allow_html=True,
        )
    survey_token = (
        None
        if survey is None or uploaded_file is None
        else (uploaded_file.name, uploaded_file.size)
    )
    if st.session_state.get("_survey_token") != survey_token:
        st.session_state["_survey_token"] = survey_token
        st.session_state["geometry_mode"] = (
            GEOMETRY_SURVEY_KEEP_WATER
            if survey is not None
            else GEOMETRY_MANUAL
        )
        st.session_state["use_reference_calibration"] = survey is not None
    elif "geometry_mode" not in st.session_state:
        st.session_state["geometry_mode"] = GEOMETRY_MANUAL
    if "use_reference_calibration" not in st.session_state:
        st.session_state["use_reference_calibration"] = False
    if survey is not None:
        with st.expander("Extracted survey metadata", expanded=False):
            st.write(f"Sample rate: {survey.sample_rate_hz / 1e6:.1f} MHz")
            st.write(
                "Probe → PP: "
                f"{_optional_mm(survey.stored_probe_to_plate_m)}"
            )
            st.write(
                "Stored plate thickness: "
                f"{_optional_mm(survey.stored_plate_thickness_m)}"
            )
            st.write(
                f"Stored fluid height: {_optional_mm(survey.stored_fluid_height_m)}"
            )
            st.write(f"Stored fluid label: {survey.fluid_material or 'missing'}")
    use_reference_calibration = st.checkbox(
        "Match waveform to water–plate reference",
        disabled=survey is None,
        key="use_reference_calibration",
        help=(
            "Derives one bounded complex system-response correction only from "
            "the measured water–plate echo, then applies it equally to all "
            "simulated echoes. ADC amplitude remains qualitative."
        ),
    )

    st.divider()
    st.markdown(
        '<div class="sidebar-kicker">Labware</div>',
        unsafe_allow_html=True,
    )
    plate_part_number = st.selectbox(
        "Labcyte plate",
        options=labcyte_plate_choice_ids(),
        index=labcyte_plate_choice_ids().index(DEFAULT_LABCYTE_PLATE_ID),
        format_func=labcyte_plate_choice_label,
        key="labcyte_plate_part_number",
        help=(
            "Offline snapshot of the resolved UK Robotics labware catalogue. "
            "Commercial barcode, colour, and sterile variants share their "
            "family's acoustic bottom profile."
        ),
    )
    plate_record = get_labcyte_plate(plate_part_number)
    plate_material_name, plate_density_default, plate_poisson_default = (
        _plate_material_defaults(plate_record.material)
    )
    st.caption(
        f"{plate_record.name} · {plate_record.well_depth_mm:g} mm well depth · "
        f"{plate_record.well_volume_ul:g} µL nominal volume"
    )
    st.markdown(
        f"[Open catalogue record ↗]({plate_record.source_url})",
        help="Source geometry used by the bundled offline snapshot.",
    )
    if plate_record.material == "coc":
        st.warning(
            "COC density and Poisson ratio are modeling assumptions because "
            "the labware catalogue does not provide them. Verify the actual "
            "plate material before quantitative use."
        )

    st.divider()
    with st.form("simulation_form"):
        st.markdown(
            '<div class="sidebar-kicker">Fluid</div>',
            unsafe_allow_html=True,
        )
        dmso_percent = st.number_input(
            "DMSO [vol.%]",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
        )
        temperature_c = st.number_input(
            "Temperature [°C]",
            min_value=20.0,
            max_value=40.0,
            value=22.0,
            step=0.5,
        )
        fluid_height_mm = st.number_input(
            "DMSO fill height [mm]",
            min_value=0.1,
            max_value=30.0,
            value=4.22,
            step=0.1,
        )

        st.markdown(
            '<div class="sidebar-kicker">Geometry</div>',
            unsafe_allow_html=True,
        )
        water_path_mm = st.number_input(
            "Water gap to plate [mm]",
            min_value=0.1,
            max_value=60.0,
            value=25.3,
            step=0.1,
        )
        plate_thickness_mm = st.number_input(
            "Plate bottom thickness [mm]",
            min_value=0.05,
            max_value=5.0,
            value=float(plate_record.bottom_thickness_mm),
            step=0.01,
            key=f"plate_thickness_mm_{plate_record.family}",
            help=(
                "Initialized from the selected catalogue profile. Enter your "
                "own measured value when available."
            ),
        )
        geometry_mode = st.selectbox(
            "Geometry source",
            options=GEOMETRY_MODES,
            disabled=survey is None,
            key="geometry_mode",
            help=(
                "The recommended survey mode preserves an independently known "
                "manual water gap while deriving plate and fluid thickness from "
                "echo differences. The all-distances mode also uses the stored "
                "water distance."
            ),
        )

        st.markdown(
            '<div class="sidebar-kicker">Transducer</div>',
            unsafe_allow_html=True,
        )
        excitation_frequency_mhz = st.number_input(
            "Excitation frequency [MHz]",
            min_value=3.0,
            max_value=12.0,
            value=10.0,
            step=0.1,
        )
        excitation_cycles = st.number_input(
            "Pulse cycles",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
        )
        transducer_diameter_mm = st.number_input(
            "Aperture diameter [mm]",
            min_value=1.0,
            max_value=15.0,
            value=13.0,
            step=0.1,
        )
        transducer_focal_length_mm = st.number_input(
            "Nominal focal length [mm]",
            min_value=2.0,
            max_value=60.0,
            value=25.4,
            step=0.1,
        )
        with st.expander(
            "Advanced acoustics, loss & uncertainty",
            expanded=False,
        ):
            st.caption(
                "Catalogue sound speed is inferred from its raw unit field. "
                "Density and Poisson ratio are not catalogue measurements. "
                "Leave losses at zero until measured or independently fitted."
            )
            fill_height_uncertainty_mm = st.number_input(
                "Fill-height sensitivity ± [mm]",
                min_value=0.0,
                max_value=float(max(0.001, fluid_height_mm - 0.001)),
                value=float(min(0.05, fluid_height_mm - 0.001)),
                step=0.01,
                help=(
                    "Deterministic range used to show how the coherent 10 MHz "
                    "cavity limit changes around the entered fill height. It "
                    "is not a statistical uncertainty unless it matches your "
                    "measurement uncertainty."
                ),
            )
            plate_longitudinal_speed_m_s = st.number_input(
                "Plate longitudinal speed [m/s]",
                min_value=500.0,
                max_value=10000.0,
                value=float(plate_record.inferred_longitudinal_speed_m_s),
                step=1.0,
                key=f"plate_speed_{plate_record.family}",
            )
            plate_density_kg_m3 = st.number_input(
                "Plate density [kg/m³]",
                min_value=100.0,
                max_value=10000.0,
                value=float(plate_density_default),
                step=10.0,
                key=f"plate_density_{plate_record.family}",
            )
            plate_poisson_ratio = st.number_input(
                "Plate Poisson ratio",
                min_value=0.0,
                max_value=0.49,
                value=float(plate_poisson_default),
                step=0.01,
                format="%.2f",
                key=f"plate_poisson_{plate_record.family}",
            )
            pp_alpha_l_db_m = st.number_input(
                "Plate longitudinal loss [dB/m]",
                min_value=0.0,
                max_value=100000.0,
                value=0.0,
                step=100.0,
            )
            pp_alpha_s_db_m = st.number_input(
                "Plate shear loss [dB/m]",
                min_value=0.0,
                max_value=100000.0,
                value=0.0,
                step=100.0,
            )
            fluid_alpha_db_m = st.number_input(
                "Fluid loss [dB/m]",
                min_value=0.0,
                max_value=20000.0,
                value=0.0,
                step=25.0,
            )
            attenuation_power = st.number_input(
                "Frequency exponent",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Attenuation scales as (frequency / 10 MHz)^exponent.",
            )
        numerical_preset = st.selectbox(
            "Calculation quality",
            options=list(NUMERICAL_PRESETS),
            index=0,
            help="Fast is intended for exploration; Accurate uses the validated grid.",
        )
        submitted = st.form_submit_button(
            "Simulate and optimize focus",
            type="primary",
            width="stretch",
        )

manual_inputs = SimulationInputs(
    dmso_volume_percent=dmso_percent,
    temperature_c=temperature_c,
    water_path_mm=water_path_mm,
    plate_thickness_mm=plate_thickness_mm,
    fluid_height_mm=fluid_height_mm,
    excitation_frequency_mhz=excitation_frequency_mhz,
    excitation_cycles=excitation_cycles,
    transducer_diameter_mm=transducer_diameter_mm,
    transducer_focal_length_mm=transducer_focal_length_mm,
    plate_part_number=plate_record.id,
    plate_material_name=plate_material_name,
    plate_longitudinal_speed_m_s=plate_longitudinal_speed_m_s,
    plate_density_kg_m3=plate_density_kg_m3,
    plate_poisson_ratio=plate_poisson_ratio,
    pp_longitudinal_attenuation_db_per_m=pp_alpha_l_db_m,
    pp_shear_attenuation_db_per_m=pp_alpha_s_db_m,
    fluid_attenuation_db_per_m=fluid_alpha_db_m,
    attenuation_power=attenuation_power,
    fill_height_uncertainty_mm=fill_height_uncertainty_mm,
    numerical_preset=numerical_preset,
)

if submitted:
    try:
        used_inputs, geometry_source = _used_geometry(
            manual_inputs,
            survey,
            geometry_mode,
        )
        with st.spinner(
            "Calculating the broadband echo and searching the focus…"
        ):
            st.session_state["simulation_result"] = simulate_cached(used_inputs)
        st.session_state["simulation_survey"] = survey
        st.session_state["geometry_source"] = geometry_source
    except (ValueError, FloatingPointError) as exc:
        st.error(f"Simulation could not be completed: {exc}")

result = st.session_state.get("simulation_result")
result_survey = st.session_state.get("simulation_survey")
geometry_source = st.session_state.get(
    "geometry_source",
    "Manual geometry fields",
)

if result is None:
    try:
        stack_inputs, stack_geometry_source = _used_geometry(
            manual_inputs,
            survey,
            geometry_mode,
        )
    except (ValueError, FloatingPointError) as exc:
        stack_inputs = manual_inputs
        stack_geometry_source = "Manual geometry fields (survey preview unavailable)"
        st.warning(
            "The survey-derived geometry could not be previewed, so the "
            f"manual fields are shown instead: {exc}"
        )
    stack_plate = plate_record
    stack_asm_focus_mm: float | None = None
    stack_geometry_focus_mm = stack_inputs.transducer_focal_length_mm
    focus_state_class = "preview"
    focus_state_tag = "Preview"
    focus_state_title = "Estimated ray focus"
    focus_state_copy = (
        "Geometry follows the current inputs. The ray focus is a "
        "geometric-acoustics estimate through the material stack; no "
        "angular-spectrum result has been calculated yet."
    )
    section_kicker = "Geometry preview"
    section_copy = (
        "Upward-facing transducer, selected well bottom, liquid layer, "
        "meniscus, and an estimated refracted focus."
    )
    stack_caption = (
        "Preview from the current sidebar values. The ray focus is estimated "
        "from the nominal transducer geometry and material sound speeds; it "
        "is not the ASM solution. The meniscus is planar in the physics model."
    )
else:
    stack_inputs = result.inputs
    stack_plate = get_labcyte_plate(result.inputs.plate_part_number)
    stack_geometry_source = geometry_source
    stack_asm_focus_mm = result.focus_from_aperture_mm
    stack_geometry_focus_mm = result.focus_from_aperture_mm
    focus_state_class = "exact"
    focus_state_tag = "ASM result"
    focus_state_title = "Calculated angular-spectrum focus"
    focus_state_copy = (
        "This cross-section uses the most recently submitted inputs and the "
        "focus returned by the full angular-spectrum calculation."
    )
    section_kicker = "Solved geometry"
    section_copy = (
        "Upward-facing transducer, selected well bottom, liquid layer, "
        "meniscus, refracted rays, and the calculated ASM focus."
    )
    stack_caption = (
        "Axial distances and well dimensions use the most recent simulation. "
        "The meniscus is planar in the physics model; the ray bundle explains "
        "refraction and is not a calculated pressure map."
    )

stack_ray = refracted_ray_preview(stack_inputs)
if stack_ray.ray_focus_y_mm is None:
    if stack_ray.critical_interface is None:
        ray_focus_summary = (
            "No propagating longitudinal edge-ray focus is available"
        )
    else:
        ray_focus_summary = (
            "The longitudinal edge ray reaches its critical angle at "
            f"{stack_ray.critical_interface}"
        )
else:
    ray_focus_summary = (
        f"Snell estimate {stack_ray.ray_focus_y_mm:.2f} mm from the aperture "
        f"({stack_ray.ray_focus_medium})"
    )

if result is None:
    if stack_ray.ray_focus_y_mm is not None:
        focus_state_title = (
            f"Estimated ray focus · {stack_ray.ray_focus_y_mm:.2f} mm"
        )
    focus_state_copy += f" {ray_focus_summary}."
else:
    focus_state_title = (
        f"Calculated angular-spectrum focus · "
        f"{result.focus_from_aperture_mm:.2f} mm"
    )
    if stack_ray.ray_focus_y_mm is not None:
        focus_delta_mm = (
            result.focus_from_aperture_mm - stack_ray.ray_focus_y_mm
        )
        focus_state_copy += (
            f" {ray_focus_summary}; the ASM maximum is shifted by "
            f"{focus_delta_mm:+.2f} mm."
        )
    else:
        focus_state_copy += f" {ray_focus_summary}."

if result is not None and plate_record.id != stack_plate.id:
    st.info(
        f"The visible result still uses {stack_plate.id}. Select "
        "**Simulate and optimize focus** to apply the new plate choice."
    )

stack_geometry = acoustic_stack_geometry(
    stack_inputs,
    stack_geometry_focus_mm,
    stack_plate,
)
if stack_plate.material == "coc":
    st.warning(
        "This view uses a COC plate. Its bottom geometry and longitudinal "
        "speed come from the labware catalogue; density and Poisson ratio are "
        "editable modeling assumptions, and loss still defaults to zero."
    )
if stack_geometry.fill_exceeds_well_depth:
    st.warning(
        f"The entered {stack_inputs.fluid_height_mm:.2f} mm fill height "
        f"exceeds the catalogue well depth of {stack_plate.well_depth_mm:.2f} "
        "mm. The hatched region in the schematic indicates this geometrically "
        "inconsistent input."
    )
st.markdown(
    f"""
    <div class="section-head">
        <div>
            <div class="section-kicker">{section_kicker}</div>
            <h2>Acoustic stack cross-section</h2>
        </div>
        <p>{section_copy}</p>
    </div>
    <div class="focus-state {focus_state_class}">
        <span class="focus-state-tag">{focus_state_tag}</span>
        <div>
            <strong>{focus_state_title}</strong>
            <span>{focus_state_copy} Geometry source: {stack_geometry_source}.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
stack_plot = acoustic_stack_schematic_figure(
    stack_inputs,
    stack_asm_focus_mm,
    stack_plate,
)
with st.container(key="acoustic_stack_visual"):
    st.pyplot(stack_plot, width="stretch")
plt.close(stack_plot)
st.caption(
    stack_caption
    + " The active well and its pitch-repeated neighbours form a local "
    "catalogue-scaled "
    f"cutaway at {stack_plate.well_pitch_mm:g} mm pitch; well widths, depth, "
    "and the entered acoustic floor thickness are drawn to scale. Break "
    "marks indicate that the physical plate continues beyond the view."
)
st.markdown(
    f"Labware geometry: [{stack_plate.id} catalogue record ↗]"
    f"({stack_plate.source_url}) · offline snapshot bundled for reproducible "
    "simulation."
)

if result is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-mark">↗</div>
            <h3>Preview ready</h3>
            <p>
                Review the geometry above, then select
                <strong>Simulate and optimize focus</strong>. The preview will
                be replaced by the calculated ASM focus, pulse response, and
                water-gap optimization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

shown_signals = display_signals(
    result,
    result_survey,
    bool(st.session_state.get("use_reference_calibration", False)),
)
if shown_signals.calibration_error is not None:
    st.warning(
        "The water–plate reference correction could not be applied: "
        f"{shown_signals.calibration_error}. Raw simulation traces are shown."
    )
if result_survey is not None:
    mismatch_notes = []
    if result_survey.probe_frequency_hz is not None and not np.isclose(
        result_survey.probe_frequency_hz,
        result.inputs.excitation_frequency_mhz * 1e6,
        rtol=0.01,
    ):
        mismatch_notes.append(
            f"survey frequency {result_survey.probe_frequency_hz / 1e6:g} MHz"
        )
    if result_survey.tone_length_cycles is not None and not np.isclose(
        result_survey.tone_length_cycles,
        result.inputs.excitation_cycles,
        rtol=0.01,
    ):
        mismatch_notes.append(
            f"survey tone length {result_survey.tone_length_cycles:g} cycles"
        )
    if mismatch_notes:
        st.warning(
            "The simulation excitation differs from the uploaded survey: "
            + ", ".join(mismatch_notes)
            + ". Timing can still be compared, but waveform shape is not a "
            "like-for-like validation."
        )

offset = result.focus_offset_from_meniscus_mm
offset_label = "below" if offset < 0.0 else "above"
water_gap_delta = result.optimal_water_path_mm - result.inputs.water_path_mm
cavity = result.meniscus_cavity
coherent_min_percent = 100.0 * (cavity.coherent_gain_height_min - 1.0)
coherent_max_percent = 100.0 * (cavity.coherent_gain_height_max - 1.0)
narrowband_percent = cavity.narrowband_separated_pass_percent_change
if cavity.electrical_overlap_regime == "electrical-burst-long":
    reflection_percent = cavity.coherent_percent_change
    reflection_value = f"{reflection_percent:+.1f}% CW limit"
    reflection_basis = (
        f"{result.inputs.excitation_frequency_mhz:g} MHz coherent forward power"
    )
    reflection_regime = "electrical burst spans at least five round trips"
    reflection_hint_class = (
        "positive" if reflection_percent >= 0.0 else "negative"
    )
elif cavity.electrical_overlap_regime == "electrical-burst-intermediate":
    reflection_value = "No single %"
    reflection_basis = "time-domain overlap unresolved"
    reflection_regime = (
        f"{narrowband_percent:+.1f}% separated-pass proxy · "
        f"{cavity.coherent_percent_change:+.1f}% CW limit"
    )
    reflection_hint_class = ""
else:
    reflection_value = f"{narrowband_percent:+.1f}% proxy"
    reflection_basis = (
        f"{result.inputs.excitation_frequency_mhz:g} MHz narrow-band "
        "separated-pass exposure"
    )
    reflection_regime = (
        "electrical burst is shorter; acoustic overlap unverified"
    )
    reflection_hint_class = "positive"
if cavity.height_uncertainty_m > 0.0:
    coherent_sensitivity_copy = (
        f"over ±{cavity.height_uncertainty_m * 1e3:.2f} mm: "
        f"{coherent_min_percent:+.1f}% to {coherent_max_percent:+.1f}%"
    )
else:
    coherent_sensitivity_copy = "at the entered nominal height"
st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-kicker">Simulation overview</div>
            <h2>Focal alignment at a glance</h2>
        </div>
        <p>
            Focus results use a single-pass criterion. Cavity returns are
            reported separately as single-frequency proxies and limits.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Focus after plate</div>
            <div class="metric-value">{result.focus_after_pp_mm:.3f} mm</div>
            <div class="metric-hint">
                {result.focus_from_aperture_mm:.3f} mm from aperture
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Meniscus alignment</div>
            <div class="metric-value">{abs(offset):.3f} mm</div>
            <div class="metric-hint">{offset_label} the fluid surface</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Modeled water-gap optimum</div>
            <div class="metric-value">{result.optimal_water_path_mm:.2f} mm</div>
            <div class="metric-hint positive">
                Adjust by {water_gap_delta:+.2f} mm · modeled intensity FWHM
                {result.optimal_meniscus_intensity_fwhm_mm:.3f} mm
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">DMSO sound speed</div>
            <div class="metric-value">
                {result.dmso_properties.sound_speed_m_s:.1f} m/s
            </div>
            <div class="metric-hint">
                {result.inputs.dmso_volume_percent:.0f} vol.% at
                {result.inputs.temperature_c:.1f} °C
            </div>
        </div>
        <div class="metric-card reflection-card">
            <div>
                <div class="metric-label">Cavity-return exposure</div>
                <div class="metric-value">{reflection_value}</div>
                <div class="metric-hint {reflection_hint_class}">
                    {reflection_basis} · {reflection_regime}
                </div>
            </div>
            <div class="reflection-card-copy">
                At the nominal height, the <strong>single-frequency
                separated-pass proxy is {narrowband_percent:+.1f}%</strong>
                and the coherent CW limit is
                <strong>{cavity.coherent_percent_change:+.1f}%</strong>;
                {coherent_sensitivity_copy}. These are not the spectrally
                integrated fluence of the configured finite pulse. Cavity round trip
                {cavity.cavity_round_trip_s * 1e6:.3f} µs versus
                {cavity.electrical_burst_duration_s * 1e6:.3f} µs electrical
                burst; causal acoustic ring-down is uncalibrated. The values
                describe repeated forward crossings, not net energy into air
                or ejection efficiency.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if result.focus_scan_boundary_limited:
    st.warning(
        "The strongest point of the current axial scan lies on its boundary. "
        "The transducer is probably focused at or before the plate exit; the "
        "recommended water-gap search is the more useful focus result."
    )
if result.optimal_water_path_boundary_limited:
    st.warning(
        "The best water gap lies on the search boundary, so it is not a "
        "resolved optimum. Use a larger-window custom calculation before "
        "changing the hardware geometry."
    )
if not cavity.cavity_series_converged:
    st.warning(
        "The nominal meniscus cavity series did not reach its convergence "
        "criterion. Treat the reflection percentage as unresolved."
    )
if not cavity.sensitivity_series_converged:
    st.warning(
        "At least one fill-height sensitivity point did not converge. The "
        "reported coherent range may be incomplete."
    )
st.markdown(
    f"""
    <div class="model-note">
        Geometry: {geometry_source}. The modeled optimum changes the relative
        single-pass focus metric by
        <strong>{result.optimal_water_path_gain_db:.2f} dB</strong>.
        This is monochromatic, uncalibrated |p|²—not an ADE efficiency or
        ejection-threshold prediction. DMSO–air cavity returns are excluded
        from the focus optimum and reported separately above.
    </div>
    """,
    unsafe_allow_html=True,
)
if shown_signals.reference_calibration is not None:
    calibration = shown_signals.reference_calibration
    st.info(
        "Waveform reference active: a regularized common-system correction "
        "derived only from the water–plate echo is applied to every displayed "
        f"simulation trace from {calibration.minimum_frequency_hz / 1e6:.1f} "
        f"to {calibration.maximum_frequency_hz / 1e6:.1f} MHz. Geometry, "
        "focus, and absolute gain are unchanged."
    )

pulse_tab, focus_tab, data_tab = st.tabs(
    ["Pulse response", "Focus optimization", "Spectrum & exports"]
)
with pulse_tab:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-kicker">Time domain</div>
                <h2>Received pulse response</h2>
            </div>
            <p>
                Compare the simulated RF signal, envelope, and individual
                interface contributions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pulse_plot = pulse_figure(result, result_survey, shown_signals)
    st.pyplot(pulse_plot, width="stretch")
    st.caption(
        "The visible time window always frames all three interface "
        "reflections and retains the selected drive plus the modeled response "
        "tail. Tick values remain absolute times from the excitation start; "
        "a survey overlay is included in the same window."
    )
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-kicker">Arrival markers</div>
                <h2>Interface timing</h2>
            </div>
            <p>All times are measured from the excitation start.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    simulated_arrivals = result.arrivals.since_excitation_us
    measured_arrivals = (
        None
        if result_survey is None
        else _measured_arrivals_since_excitation_us(result_survey)
    )
    timing_rows = []
    for name, simulated_us in simulated_arrivals.items():
        measured_us = (
            None if measured_arrivals is None else measured_arrivals[name]
        )
        raw_correlation: float | str = "—"
        displayed_correlation: float | str = "—"
        if result_survey is not None and measured_us is not None:
            raw_correlation = round(
                local_waveform_correlation(
                    result_survey.time_since_excitation_s,
                    result_survey.normalized_signal,
                    result.time_since_excitation_us * 1e-6,
                    result.received_normalized,
                    measured_arrival_s=measured_us * 1e-6,
                    simulated_arrival_s=simulated_us * 1e-6,
                ),
                3,
            )
            displayed_correlation = round(
                local_waveform_correlation(
                    result_survey.time_since_excitation_s,
                    result_survey.normalized_signal,
                    result.time_since_excitation_us * 1e-6,
                    shown_signals.received,
                    measured_arrival_s=measured_us * 1e-6,
                    simulated_arrival_s=simulated_us * 1e-6,
                ),
                3,
            )
        timing_rows.append(
            {
                "Interface": name,
                "Simulation [µs]": round(simulated_us, 4),
                "Survey [µs]": (
                    "—" if measured_us is None else round(measured_us, 4)
                ),
                "Survey − simulation [µs]": (
                    "—"
                    if measured_us is None
                    else round(measured_us - simulated_us, 4)
                ),
                "Correlation raw": raw_correlation,
                "Correlation displayed": displayed_correlation,
            }
        )
    st.dataframe(
        timing_rows,
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Times are measured from the excitation start. A timing residual can "
        "arise from uncertain geometry, fluid concentration/temperature, PP "
        "sound speed, interface picking, or fixed electronic delay. Correlations "
        "use independently normalized 0.6 µs windows and a ±0.15 µs local lag "
        "search."
    )

with focus_tab:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-kicker">Spatial response</div>
                <h2>Focus and water-gap optimization</h2>
            </div>
            <p>
                Separate the current axial maximum from the water gap that
                maximizes center-frequency pressure squared at the meniscus.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    focus_plot = focus_figure(result)
    st.pyplot(focus_plot, width="stretch")
    st.markdown(
        f"""
        <div class="model-note">
            The current on-axis maximum is
            <strong>{result.focus_from_aperture_mm:.3f} mm from the
            aperture</strong>. For the {result.inputs.fluid_height_mm:.3f} mm
            meniscus, center a low-drive experimental scan near
            <strong>{result.optimal_water_path_mm:.3f} mm</strong>. The search
            includes the PP transmission phase and temperature-dependent water
            and DMSO sound speeds. For a one-cycle ejection pulse, verify this
            recommendation with a broadband pressure/energy measurement.
        </div>
        """,
        unsafe_allow_html=True,
    )

with data_tab:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-kicker">Frequency domain</div>
                <h2>Received spectrum</h2>
            </div>
            <p>
                Inspect the normalized spectral response and export the
                calculation for further analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    spectral_plot = spectrum_figure(result, shown_signals)
    st.pyplot(spectral_plot, width="stretch")
    summary = result_summary(
        result,
        result_survey,
        geometry_source,
        shown_signals,
    )
    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download pulse plot (PNG)",
        data=figure_png(pulse_plot),
        file_name="pulse_echo_focus_lab.png",
        mime="image/png",
        width="stretch",
    )
    download_columns[1].download_button(
        "Download signal data (CSV)",
        data=result_csv(result, result_survey, shown_signals),
        file_name="pulse_echo_focus_lab.csv",
        mime="text/csv",
        width="stretch",
    )
    download_columns[2].download_button(
        "Download summary (JSON)",
        data=json.dumps(summary, indent=2, ensure_ascii=False),
        file_name="pulse_echo_focus_lab.json",
        mime="application/json",
        width="stretch",
    )
    with st.expander("Model assumptions"):
        for assumption in summary["assumptions"]:
            st.write(f"• {assumption}")

st.markdown(
    """
    <div class="footer-note">
        Pulse Echo Focus Lab · Relative acoustic model · Survey ADC values are
        not an absolute pressure calibration
    </div>
    """,
    unsafe_allow_html=True,
)
