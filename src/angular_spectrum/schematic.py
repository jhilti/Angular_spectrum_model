"""Scaled cross-section drawing for the interactive acoustic stack."""

from __future__ import annotations

from dataclasses import dataclass
import math
import textwrap

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from .app_model import SimulationInputs
from .labware import LabcytePlate


@dataclass(frozen=True, slots=True)
class AcousticStackGeometry:
    """Key axial coordinates in millimetres from the aperture centre."""

    aperture_y_mm: float
    water_plate_y_mm: float
    plate_fluid_y_mm: float
    meniscus_y_mm: float
    well_rim_y_mm: float
    focus_y_mm: float
    aperture_radius_mm: float
    well_bottom_radius_mm: float
    well_top_radius_mm: float

    @property
    def focus_offset_from_meniscus_mm(self) -> float:
        """Positive when the focus is above the liquid surface."""

        return self.focus_y_mm - self.meniscus_y_mm

    @property
    def fill_exceeds_well_depth(self) -> bool:
        return self.meniscus_y_mm > self.well_rim_y_mm


def acoustic_stack_geometry(
    inputs: SimulationInputs,
    focus_from_aperture_mm: float,
    plate: LabcytePlate,
) -> AcousticStackGeometry:
    """Return the exact coordinates used by the cross-section drawing."""

    inputs.validate()
    if not math.isfinite(focus_from_aperture_mm):
        raise ValueError("focus_from_aperture_mm must be finite")
    water_plate = inputs.water_path_mm
    plate_fluid = water_plate + inputs.plate_thickness_mm
    return AcousticStackGeometry(
        aperture_y_mm=0.0,
        water_plate_y_mm=water_plate,
        plate_fluid_y_mm=plate_fluid,
        meniscus_y_mm=plate_fluid + inputs.fluid_height_mm,
        well_rim_y_mm=plate_fluid + plate.well_depth_mm,
        focus_y_mm=float(focus_from_aperture_mm),
        aperture_radius_mm=inputs.transducer_diameter_mm / 2.0,
        well_bottom_radius_mm=plate.well_bottom_width_mm / 2.0,
        well_top_radius_mm=plate.well_top_width_mm / 2.0,
    )


def _dimension(
    axis: plt.Axes,
    *,
    x_mm: float,
    low_mm: float,
    high_mm: float,
    label: str,
    color: str,
) -> None:
    axis.annotate(
        "",
        xy=(x_mm, high_mm),
        xytext=(x_mm, low_mm),
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "linewidth": 1.15,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    axis.text(
        x_mm + 0.22,
        (low_mm + high_mm) / 2.0,
        label,
        color=color,
        fontsize=8.4,
        va="center",
        ha="left",
        rotation=90,
    )


def _well_half_width_mm(
    y_mm: np.ndarray | float,
    geometry: AcousticStackGeometry,
) -> np.ndarray:
    values = np.asarray(y_mm, dtype=float)
    fraction = np.clip(
        (values - geometry.plate_fluid_y_mm)
        / max(geometry.well_rim_y_mm - geometry.plate_fluid_y_mm, 1.0e-9),
        0.0,
        1.0,
    )
    return (
        geometry.well_bottom_radius_mm
        + fraction
        * (geometry.well_top_radius_mm - geometry.well_bottom_radius_mm)
    )


def acoustic_stack_schematic_figure(
    inputs: SimulationInputs,
    focus_from_aperture_mm: float,
    plate: LabcytePlate,
) -> Figure:
    """Draw the upward-facing transducer, plate well, and planar meniscus.

    Axial coordinates and all displayed dimensions use the simulation values.
    The translucent cone only communicates convergence and is deliberately not
    presented as a calculated pressure field.
    """

    geometry = acoustic_stack_geometry(inputs, focus_from_aperture_mm, plate)
    colors = {
        "ink": "#10282f",
        "muted": "#657b80",
        "water": "#b9e2ea",
        "water_line": "#3b95a3",
        "pp": "#f0bd73",
        "coc": "#c7b7df",
        "dmso": "#9bd8c6",
        "dmso_line": "#16846f",
        "air": "#f7f9f8",
        "focus": "#d49a35",
        "beam": "#147d88",
        "transducer": "#244f59",
        "paper": "#fbfcfb",
        "line": "#d7e4e1",
    }
    plate_color = colors["pp"] if plate.material == "polypropylene" else colors["coc"]

    radius = geometry.aperture_radius_mm
    x_limit = max(radius + 2.7, geometry.well_top_radius_mm + 4.4)
    lower_y = -max(2.0, 0.13 * inputs.transducer_focal_length_mm)
    upper_y = max(
        geometry.well_rim_y_mm,
        geometry.meniscus_y_mm,
        geometry.focus_y_mm,
    ) + 1.6

    figure = plt.figure(figsize=(11.8, 6.7), constrained_layout=True)
    figure.patch.set_facecolor(colors["paper"])
    grid = figure.add_gridspec(1, 2, width_ratios=(1.45, 0.78), wspace=0.05)
    axis = figure.add_subplot(grid[0, 0])
    details = figure.add_subplot(grid[0, 1])
    axis.set_facecolor(colors["air"])
    details.set_facecolor(colors["paper"])

    axis.add_patch(
        Rectangle(
            (-x_limit, 0.0),
            2.0 * x_limit,
            geometry.water_plate_y_mm,
            facecolor=colors["water"],
            edgecolor="none",
            alpha=0.52,
            zorder=0,
        )
    )

    # Elastic plate bottom and the two sidewalls around the selected well.
    axis.add_patch(
        Rectangle(
            (-x_limit, geometry.water_plate_y_mm),
            2.0 * x_limit,
            inputs.plate_thickness_mm,
            facecolor=plate_color,
            edgecolor=colors["ink"],
            linewidth=1.0,
            alpha=0.86,
            zorder=3,
        )
    )
    left_wall = np.array(
        [
            [-x_limit, geometry.plate_fluid_y_mm],
            [-geometry.well_bottom_radius_mm, geometry.plate_fluid_y_mm],
            [-geometry.well_top_radius_mm, geometry.well_rim_y_mm],
            [-x_limit, geometry.well_rim_y_mm],
        ]
    )
    right_wall = left_wall.copy()
    right_wall[:, 0] *= -1.0
    for wall in (left_wall, right_wall):
        axis.add_patch(
            Polygon(
                wall,
                closed=True,
                facecolor=plate_color,
                edgecolor=colors["ink"],
                linewidth=1.0,
                alpha=0.74,
                zorder=2,
            )
        )

    liquid_top = min(geometry.meniscus_y_mm, geometry.well_rim_y_mm)
    if liquid_top > geometry.plate_fluid_y_mm:
        y_liquid = np.linspace(geometry.plate_fluid_y_mm, liquid_top, 80)
        half_width = _well_half_width_mm(y_liquid, geometry)
        liquid_polygon = np.column_stack(
            (
                np.concatenate((-half_width, half_width[::-1])),
                np.concatenate((y_liquid, y_liquid[::-1])),
            )
        )
        axis.add_patch(
            Polygon(
                liquid_polygon,
                closed=True,
                facecolor=colors["dmso"],
                edgecolor="none",
                alpha=0.72,
                zorder=1,
            )
        )
    if geometry.fill_exceeds_well_depth:
        axis.add_patch(
            Rectangle(
                (-geometry.well_top_radius_mm, geometry.well_rim_y_mm),
                2.0 * geometry.well_top_radius_mm,
                geometry.meniscus_y_mm - geometry.well_rim_y_mm,
                facecolor=colors["dmso"],
                edgecolor=colors["dmso_line"],
                hatch="///",
                linewidth=0.8,
                alpha=0.38,
                zorder=1,
            )
        )

    meniscus_half_width = float(
        _well_half_width_mm(geometry.meniscus_y_mm, geometry).item()
    )
    axis.plot(
        [-meniscus_half_width, meniscus_half_width],
        [geometry.meniscus_y_mm, geometry.meniscus_y_mm],
        color=colors["dmso_line"],
        linewidth=3.0,
        solid_capstyle="round",
        zorder=7,
    )

    # Spherical-cap transducer face with its concavity pointing upward.
    focal_radius = max(inputs.transducer_focal_length_mm, radius + 0.05)
    transducer_x = np.linspace(-radius, radius, 240)
    transducer_y = focal_radius - np.sqrt(
        np.maximum(focal_radius**2 - transducer_x**2, 0.0)
    )
    axis.fill_between(
        transducer_x,
        transducer_y,
        transducer_y - 0.75,
        color=colors["transducer"],
        alpha=0.95,
        zorder=8,
    )
    axis.plot(
        transducer_x,
        transducer_y,
        color="#0e343d",
        linewidth=1.7,
        zorder=9,
    )

    beam_start_y = float(np.max(transducer_y))
    beam_end_y = max(geometry.meniscus_y_mm, geometry.focus_y_mm)
    if beam_end_y > beam_start_y and geometry.focus_y_mm > 0.0:
        beam_y = np.linspace(beam_start_y, beam_end_y, 260)
        beam_half_width = radius * np.abs(
            1.0 - beam_y / geometry.focus_y_mm
        )
        beam_half_width = np.minimum(beam_half_width, x_limit)
        axis.fill_betweenx(
            beam_y,
            -beam_half_width,
            beam_half_width,
            color=colors["beam"],
            alpha=0.11,
            zorder=4,
        )
        axis.plot(
            beam_half_width,
            beam_y,
            color=colors["beam"],
            linewidth=1.1,
            alpha=0.72,
            zorder=5,
        )
        axis.plot(
            -beam_half_width,
            beam_y,
            color=colors["beam"],
            linewidth=1.1,
            alpha=0.72,
            zorder=5,
        )

    axis.scatter(
        [0.0],
        [geometry.focus_y_mm],
        s=115,
        marker="*",
        color=colors["focus"],
        edgecolor="#8d641f",
        linewidth=0.8,
        zorder=10,
    )
    axis.annotate(
        "Modeled focus",
        xy=(0.0, geometry.focus_y_mm),
        xytext=(0.65, geometry.focus_y_mm + 0.8),
        color=colors["ink"],
        fontsize=8.8,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": colors["focus"], "lw": 1.0},
        zorder=11,
    )

    dimension_x = x_limit - 1.05
    _dimension(
        axis,
        x_mm=dimension_x,
        low_mm=0.0,
        high_mm=geometry.water_plate_y_mm,
        label=f"water {inputs.water_path_mm:.2f} mm",
        color=colors["water_line"],
    )
    _dimension(
        axis,
        x_mm=dimension_x,
        low_mm=geometry.plate_fluid_y_mm,
        high_mm=geometry.meniscus_y_mm,
        label=f"fill {inputs.fluid_height_mm:.2f} mm",
        color=colors["dmso_line"],
    )

    axis.text(
        -x_limit + 0.35,
        geometry.water_plate_y_mm / 2.0,
        "WATER",
        color=colors["water_line"],
        fontsize=8.4,
        fontweight="bold",
        alpha=0.9,
        rotation=90,
        va="center",
    )
    axis.text(
        0.0,
        geometry.plate_fluid_y_mm
        + max(0.25, min(inputs.fluid_height_mm, plate.well_depth_mm) / 2.0),
        f"{inputs.dmso_volume_percent:.0f}% DMSO",
        color="#116956",
        fontsize=8.0,
        fontweight=700,
        ha="center",
        va="center",
        rotation=90,
        zorder=6,
    )
    axis.text(
        -x_limit + 0.35,
        (geometry.water_plate_y_mm + geometry.plate_fluid_y_mm) / 2.0,
        f"{plate.id} bottom · {inputs.plate_thickness_mm:.2f} mm",
        color="#704e1c",
        fontsize=7.4,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=7,
    )
    axis.axhline(
        geometry.water_plate_y_mm,
        color=colors["ink"],
        linewidth=0.75,
        alpha=0.7,
        zorder=6,
    )
    axis.set(
        xlim=(-x_limit, x_limit),
        ylim=(lower_y, upper_y),
        xlabel="Lateral position [mm]",
        ylabel="Distance from aperture centre [mm]",
    )
    axis.set_aspect(0.43)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#b9cbc7")
    axis.tick_params(colors=colors["muted"], labelsize=8.5)
    axis.grid(axis="y", color=colors["line"], linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)

    offset = geometry.focus_offset_from_meniscus_mm
    alignment = (
        "aligned"
        if abs(offset) < 0.05
        else f"{abs(offset):.2f} mm {'above' if offset > 0 else 'below'} surface"
    )
    material_label = "polypropylene" if plate.material == "polypropylene" else "COC"
    details.text(
        0.0,
        0.96,
        plate.id,
        transform=details.transAxes,
        color=colors["ink"],
        fontsize=19,
        fontweight="bold",
        va="top",
    )
    details.text(
        0.0,
        0.88,
        textwrap.fill(plate.name, width=41),
        transform=details.transAxes,
        color=colors["muted"],
        fontsize=9.3,
        va="top",
        linespacing=1.35,
    )

    rows = (
        ("Plate family", plate.family),
        ("Material", material_label),
        ("Well format", f"{plate.well_count} wells · {plate.well_volume_ul:g} µL"),
        ("Well opening", f"{plate.well_top_width_mm:g} mm · {plate.well_pitch_mm:g} mm pitch"),
        ("Bottom", f"{inputs.plate_thickness_mm:.2f} mm"),
        ("Current focus", f"{geometry.focus_y_mm:.2f} mm from aperture"),
        ("Meniscus", f"{geometry.meniscus_y_mm:.2f} mm from aperture"),
        ("Alignment", alignment),
    )
    row_y = 0.67
    for label, value in rows:
        details.text(
            0.0,
            row_y,
            label.upper(),
            transform=details.transAxes,
            color=colors["muted"],
            fontsize=7.2,
            fontweight="bold",
            va="top",
        )
        details.text(
            0.0,
            row_y - 0.030,
            value,
            transform=details.transAxes,
            color=colors["ink"],
            fontsize=9.4,
            fontweight="bold",
            va="top",
        )
        details.plot(
            [0.0, 0.92],
            [row_y - 0.066, row_y - 0.066],
            transform=details.transAxes,
            color=colors["line"],
            linewidth=0.7,
        )
        row_y -= 0.078

    details.text(
        0.0,
        0.012,
        "Scaled axial geometry · planar modeled meniscus\n"
        "Beam cone is illustrative, not a pressure field.",
        transform=details.transAxes,
        color=colors["muted"],
        fontsize=7.6,
        va="bottom",
        linespacing=1.35,
    )
    details.set_axis_off()
    figure.suptitle(
        "Acoustic stack cross-section",
        x=0.02,
        ha="left",
        color=colors["ink"],
        fontsize=15,
        fontweight="bold",
    )
    return figure
