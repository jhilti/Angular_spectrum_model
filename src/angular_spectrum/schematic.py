"""Scaled cross-section drawing for the interactive acoustic stack."""

from __future__ import annotations

from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from .app_model import SimulationInputs
from .dmso_mixture import dmso_water_properties, water_properties
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


@dataclass(frozen=True, slots=True)
class RefractedRayPreview:
    """Longitudinal aperture-edge ray through the three modeled layers.

    The preview uses geometric Snell refraction, with the conserved transverse
    slowness ``sin(theta) / c``.  It is intentionally separate from the exact
    angular-spectrum result: diffraction, the aperture weighting, elastic
    P/SV mode conversion, and plate resonances do not have a single ray path.

    ``right_edge_x_mm`` contains the signed lateral coordinate of the ray
    launched from the positive aperture edge.  The left ray is its mirror.
    If a longitudinal branch is beyond its critical angle, the coordinates
    stop at that interface and ``critical_interface`` identifies it.
    """

    axial_y_mm: tuple[float, ...]
    right_edge_x_mm: tuple[float, ...]
    transverse_slowness_s_m: float
    water_angle_deg: float
    plate_longitudinal_angle_deg: float | None
    liquid_angle_deg: float | None
    ray_focus_y_mm: float | None
    ray_focus_medium: str | None
    focus_is_extrapolated_beyond_meniscus: bool
    critical_interface: str | None


def refracted_ray_preview(inputs: SimulationInputs) -> RefractedRayPreview:
    """Return a geometric aperture-edge ray with Snell-law refraction.

    The aperture is the planar equivalent used by the numerical source model.
    Its edge ray is aimed at the nominal homogeneous-water focus.  Only the
    propagating longitudinal branch in the plate is followed; this is a
    physically interpretable orientation aid, not a replacement for the
    angular-spectrum/elastic-plate calculation.
    """

    inputs.validate()
    radius_mm = 0.5 * inputs.transducer_diameter_mm
    water_angle = math.atan2(radius_mm, inputs.transducer_focal_length_mm)
    water_speed = water_properties(inputs.temperature_c).sound_speed_m_s
    liquid_speed = dmso_water_properties(
        inputs.dmso_volume_percent / 100.0,
        basis="volume",
        temperature_c=inputs.temperature_c,
    ).sound_speed_m_s
    transverse_slowness = math.sin(water_angle) / water_speed

    def transmitted_angle(sound_speed_m_s: float) -> float | None:
        sine = transverse_slowness * sound_speed_m_s
        # Exactly grazing propagation has an unbounded geometric walkoff and
        # is not a useful finite ray for this preview.
        if sine >= 1.0:
            return None
        return math.asin(max(0.0, sine))

    plate_angle = transmitted_angle(inputs.plate_longitudinal_speed_m_s)
    liquid_angle = transmitted_angle(liquid_speed)
    axial_y = [0.0]
    right_x = [radius_mm]
    ray_focus_y: float | None = None
    ray_focus_medium: str | None = None

    def append_segment(
        length_mm: float,
        angle_rad: float,
        medium: str,
    ) -> None:
        nonlocal ray_focus_y, ray_focus_medium
        y0 = axial_y[-1]
        x0 = right_x[-1]
        y1 = y0 + length_mm
        x1 = x0 - length_mm * math.tan(angle_rad)
        axial_y.append(y1)
        right_x.append(x1)
        if (
            ray_focus_y is None
            and x0 >= 0.0
            and x1 <= 0.0
            and not math.isclose(x0, x1)
        ):
            fraction = x0 / (x0 - x1)
            ray_focus_y = y0 + fraction * length_mm
            ray_focus_medium = medium

    append_segment(inputs.water_path_mm, water_angle, "water")
    critical_interface: str | None = None
    if plate_angle is None:
        critical_interface = "water–plate"
    else:
        append_segment(
            inputs.plate_thickness_mm,
            plate_angle,
            "plate (longitudinal branch)",
        )
        if liquid_angle is None:
            critical_interface = "plate–liquid"
        else:
            append_segment(inputs.fluid_height_mm, liquid_angle, "liquid")

    extrapolated = False
    if (
        ray_focus_y is None
        and critical_interface is None
        and liquid_angle is not None
        and right_x[-1] > 0.0
        and math.tan(liquid_angle) > 0.0
    ):
        # This is useful for seeing whether the geometric focus lies above the
        # entered fill, but the dotted continuation is explicitly labelled as
        # an unbounded-liquid extrapolation: the real free surface interrupts
        # the ray at the meniscus.
        ray_focus_y = axial_y[-1] + right_x[-1] / math.tan(liquid_angle)
        ray_focus_medium = "liquid (extrapolated beyond meniscus)"
        extrapolated = True

    return RefractedRayPreview(
        axial_y_mm=tuple(float(value) for value in axial_y),
        right_edge_x_mm=tuple(float(value) for value in right_x),
        transverse_slowness_s_m=float(transverse_slowness),
        water_angle_deg=math.degrees(water_angle),
        plate_longitudinal_angle_deg=(
            None if plate_angle is None else math.degrees(plate_angle)
        ),
        liquid_angle_deg=(
            None if liquid_angle is None else math.degrees(liquid_angle)
        ),
        ray_focus_y_mm=ray_focus_y,
        ray_focus_medium=ray_focus_medium,
        focus_is_extrapolated_beyond_meniscus=extrapolated,
        critical_interface=critical_interface,
    )


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
    focus_from_aperture_mm: float | None,
    plate: LabcytePlate,
) -> Figure:
    """Draw a portrait preview of the upward-facing acoustic stack.

    The blue path is the longitudinal aperture-edge ray returned by
    :func:`refracted_ray_preview`; its visible interface kinks follow Snell's
    law.  ``focus_from_aperture_mm`` may be ``None`` for a ray-only preview or
    the independently calculated angular-spectrum focus for a distinct marker.
    """

    ray = refracted_ray_preview(inputs)
    display_focus_mm = (
        inputs.transducer_focal_length_mm
        if focus_from_aperture_mm is None
        else focus_from_aperture_mm
    )
    geometry = acoustic_stack_geometry(inputs, display_focus_mm, plate)
    colors = {
        "ink": "#17343d",
        "muted": "#687d84",
        "water": "#dceff5",
        "water_line": "#318da2",
        "plate": "#d9e3e6",
        "plate_dark": "#45646d",
        "liquid": "#f7b455",
        "liquid_light": "#fbd8a2",
        "liquid_line": "#e77925",
        "air": "#fcfdfd",
        "asm_focus": "#e77925",
        "ray": "#176c82",
        "transducer": "#244f59",
        "paper": "#fcfdfd",
        "line": "#d9e4e7",
    }

    radius = geometry.aperture_radius_mm
    x_limit = max(radius + 3.0, geometry.well_top_radius_mm + 4.6)
    focus_candidates = [
        geometry.well_rim_y_mm,
        geometry.meniscus_y_mm,
    ]
    if focus_from_aperture_mm is not None:
        focus_candidates.append(focus_from_aperture_mm)
    if ray.ray_focus_y_mm is not None:
        focus_candidates.append(ray.ray_focus_y_mm)
    lower_y = -max(2.0, min(4.0, 0.13 * inputs.transducer_focal_length_mm))
    if focus_from_aperture_mm is not None:
        lower_y = min(lower_y, focus_from_aperture_mm - 1.0)
    upper_y = max(focus_candidates) + 3.2

    figure, axis = plt.subplots(
        figsize=(5.8, 8.2),
        constrained_layout=True,
    )
    figure.patch.set_facecolor(colors["paper"])
    axis.set_facecolor(colors["air"])

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
            facecolor=colors["plate"],
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
                facecolor=colors["plate"],
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
                facecolor=colors["liquid"],
                edgecolor="none",
                alpha=0.82,
                zorder=1,
            )
        )
    if geometry.fill_exceeds_well_depth:
        axis.add_patch(
            Rectangle(
                (-geometry.well_top_radius_mm, geometry.well_rim_y_mm),
                2.0 * geometry.well_top_radius_mm,
                geometry.meniscus_y_mm - geometry.well_rim_y_mm,
                facecolor=colors["liquid_light"],
                edgecolor=colors["liquid_line"],
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
        color=colors["liquid_line"],
        linewidth=3.0,
        solid_capstyle="round",
        zorder=7,
    )

    # A spherical-cap icon sits below the planar equivalent aperture at y=0.
    # Subtracting the edge sag keeps the rim on that reference plane.
    focal_radius = max(inputs.transducer_focal_length_mm, radius + 0.05)
    transducer_x = np.linspace(-radius, radius, 240)
    raw_transducer_y = focal_radius - np.sqrt(
        np.maximum(focal_radius**2 - transducer_x**2, 0.0)
    )
    transducer_y = raw_transducer_y - float(raw_transducer_y[-1])
    body_top_y = float(np.min(transducer_y)) - 0.18
    body_bottom_y = body_top_y - 2.15
    body_half_width = min(3.15, 0.50 * radius)
    lower_y = min(lower_y, body_bottom_y - 0.28)
    axis.add_patch(
        Rectangle(
            (-body_half_width, body_bottom_y),
            2.0 * body_half_width,
            body_top_y - body_bottom_y,
            facecolor="#1c2d57",
            edgecolor="#0e343d",
            linewidth=1.1,
            zorder=7,
        )
    )
    for rib_y in np.linspace(body_bottom_y + 0.38, body_top_y - 0.38, 4):
        axis.plot(
            [-body_half_width, body_half_width],
            [rib_y, rib_y],
            color="#78909a",
            linewidth=0.55,
            alpha=0.78,
            zorder=8,
        )
    axis.text(
        0.0,
        (body_bottom_y + body_top_y) / 2.0,
        f"{inputs.excitation_frequency_mhz:g} MHz\nTRANSDUCER",
        color="#ffffff",
        fontsize=6.5,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=9,
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
    axis.text(
        body_half_width + 0.2,
        body_bottom_y + 0.12,
        f"{inputs.transducer_diameter_mm:.1f} mm aperture",
        color=colors["muted"],
        fontsize=6.8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # Draw the two signed edge rays.  Inserting the focus into the polyline
    # makes the filled preview close to zero cleanly if the rays cross inside
    # one of the layers.
    ray_y = list(ray.axial_y_mm)
    ray_x = list(ray.right_edge_x_mm)
    if (
        ray.ray_focus_y_mm is not None
        and not ray.focus_is_extrapolated_beyond_meniscus
    ):
        for index in range(len(ray_y) - 1):
            if ray_y[index] < ray.ray_focus_y_mm < ray_y[index + 1]:
                ray_y.insert(index + 1, ray.ray_focus_y_mm)
                ray_x.insert(index + 1, 0.0)
                break
    for index in range(len(ray_y) - 1):
        segment_y = np.asarray(ray_y[index : index + 2])
        segment_half_width = np.abs(
            np.asarray(ray_x[index : index + 2])
        )
        axis.fill_betweenx(
            segment_y,
            -segment_half_width,
            segment_half_width,
            color=colors["ray"],
            alpha=0.09,
            zorder=4,
        )
    axis.plot(
        ray_x,
        ray_y,
        color=colors["ray"],
        linewidth=1.45,
        marker="o",
        markersize=2.8,
        label="Snell edge rays",
        zorder=6,
    )
    axis.plot(
        -np.asarray(ray_x),
        ray_y,
        color=colors["ray"],
        linewidth=1.45,
        marker="o",
        markersize=2.8,
        zorder=6,
    )

    if ray.focus_is_extrapolated_beyond_meniscus:
        axis.plot(
            [ray.right_edge_x_mm[-1], 0.0],
            [ray.axial_y_mm[-1], ray.ray_focus_y_mm],
            color=colors["ray"],
            linewidth=1.0,
            linestyle=":",
            alpha=0.7,
            zorder=5,
        )
        axis.plot(
            [-ray.right_edge_x_mm[-1], 0.0],
            [ray.axial_y_mm[-1], ray.ray_focus_y_mm],
            color=colors["ray"],
            linewidth=1.0,
            linestyle=":",
            alpha=0.7,
            zorder=5,
        )

    if ray.ray_focus_y_mm is not None:
        ray_focus_label = (
            "Ray focus (liquid extrap.)"
            if ray.focus_is_extrapolated_beyond_meniscus
            else "Snell ray focus"
        )
        axis.scatter(
            [0.0],
            [ray.ray_focus_y_mm],
            s=70,
            marker="o",
            facecolor=colors["paper"],
            edgecolor=colors["ray"],
            linewidth=1.5,
            label=ray_focus_label,
            zorder=10,
        )
        axis.annotate(
            ray_focus_label,
            xy=(0.0, ray.ray_focus_y_mm),
            xytext=(-2.05, ray.ray_focus_y_mm + 0.18),
            color=colors["ray"],
            fontsize=7.8,
            fontweight="bold",
            ha="right",
            arrowprops={"arrowstyle": "-", "color": colors["ray"], "lw": 0.8},
            zorder=11,
        )

    if ray.critical_interface is not None:
        critical_y = ray.axial_y_mm[-1]
        critical_x = abs(ray.right_edge_x_mm[-1])
        axis.scatter(
            [-critical_x, critical_x],
            [critical_y, critical_y],
            marker="x",
            s=34,
            color=colors["liquid_line"],
            linewidth=1.3,
            zorder=11,
        )
        axis.annotate(
            f"Longitudinal edge ray beyond critical angle\n"
            f"at {ray.critical_interface}",
            xy=(critical_x, critical_y),
            xytext=(0.55 * x_limit, critical_y + 1.0),
            color=colors["muted"],
            fontsize=7.2,
            ha="center",
            arrowprops={"arrowstyle": "-", "color": colors["muted"], "lw": 0.8},
            zorder=12,
        )

    if focus_from_aperture_mm is not None:
        axis.scatter(
            [0.0],
            [focus_from_aperture_mm],
            s=125,
            marker="*",
            color=colors["asm_focus"],
            edgecolor="#9b491b",
            linewidth=0.7,
            label="ASM axial focus",
            zorder=12,
        )
        axis.annotate(
            "ASM axial focus",
            xy=(0.0, focus_from_aperture_mm),
            xytext=(2.05, focus_from_aperture_mm + 0.18),
            color=colors["asm_focus"],
            fontsize=7.8,
            fontweight="bold",
            ha="left",
            arrowprops={
                "arrowstyle": "-",
                "color": colors["asm_focus"],
                "lw": 0.9,
            },
            zorder=13,
        )

    dimension_x = x_limit - 0.72
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
        color=colors["liquid_line"],
    )

    axis.text(
        -x_limit + 0.42,
        geometry.water_plate_y_mm / 2.0,
        "COUPLING WATER",
        color=colors["water_line"],
        fontsize=7.8,
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
        color="#7d4217",
        fontsize=7.8,
        fontweight=700,
        ha="center",
        va="center",
        rotation=90,
        zorder=6,
    )
    material_short = "PP" if plate.material == "polypropylene" else "COC"
    axis.text(
        -x_limit + 0.4,
        geometry.plate_fluid_y_mm
        + min(2.0, 0.35 * plate.well_depth_mm),
        f"{plate.id} · {plate.well_count}-WELL SOURCE PLATE\n"
        f"{inputs.plate_thickness_mm:.2f} mm {material_short} BOTTOM",
        color=colors["plate_dark"],
        fontsize=6.9,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=7,
    )
    axis.text(
        0.0,
        geometry.meniscus_y_mm + 0.24,
        "PLANAR MENISCUS",
        color=colors["liquid_line"],
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=8,
    )
    axis.axhline(
        geometry.water_plate_y_mm,
        color=colors["ink"],
        linewidth=0.75,
        alpha=0.7,
        zorder=6,
    )
    axis.axhline(
        geometry.plate_fluid_y_mm,
        color=colors["plate_dark"],
        linewidth=0.7,
        alpha=0.72,
        zorder=6,
    )
    axis.axvline(
        0.0,
        color=colors["line"],
        linewidth=0.7,
        linestyle="--",
        zorder=0,
    )
    axis.annotate(
        "EJECTION / AIR",
        xy=(0.0, upper_y - 1.35),
        xytext=(0.0, upper_y - 2.25),
        color=colors["muted"],
        fontsize=7.5,
        fontweight="bold",
        ha="center",
        arrowprops={
            "arrowstyle": "-|>",
            "color": colors["liquid_line"],
            "lw": 1.0,
        },
    )
    axis.set(
        xlim=(-x_limit, x_limit),
        ylim=(lower_y, upper_y),
        xlabel="Lateral position [mm]",
        ylabel="Axial distance from aperture [mm]  ↑",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#b7c8cd")
    axis.tick_params(colors=colors["muted"], labelsize=8.0)
    axis.grid(axis="y", color=colors["line"], linewidth=0.55, alpha=0.62)
    axis.set_axisbelow(True)
    angle_text = (
        f"Snell edge ray (longitudinal branch): "
        f"water {ray.water_angle_deg:.1f}°"
    )
    if ray.plate_longitudinal_angle_deg is not None:
        angle_text += f" → plate {ray.plate_longitudinal_angle_deg:.1f}°"
    if ray.liquid_angle_deg is not None:
        angle_text += f" → liquid {ray.liquid_angle_deg:.1f}°"
    axis.text(
        0.01,
        0.095,
        angle_text
        + "\nRay preview excludes diffraction, P/SV conversion, and plate resonances.",
        transform=axis.transAxes,
        color=colors["muted"],
        fontsize=6.9,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": colors["paper"],
            "edgecolor": colors["line"],
            "alpha": 0.93,
        },
        zorder=20,
    )
    handles, labels = axis.get_legend_handles_labels()
    unique_handles: list[object] = []
    unique_labels: list[str] = []
    for handle, label in zip(handles, labels, strict=True):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    axis.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=6.6,
        ncols=max(1, len(unique_labels)),
        columnspacing=1.0,
        handletextpad=0.45,
    )
    axis.set_title(
        "Upward acoustic path",
        color=colors["ink"],
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    return figure
