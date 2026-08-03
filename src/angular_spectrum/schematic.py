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


# Streamlit Community Cloud can retain an imported project module while
# hot-reloading the entrypoint.  The app checks this revision before rendering
# so an old schematic cannot silently survive a deployment update.
SCHEMATIC_RENDERER_REVISION = "2026-08-03-clean-technical-v2"


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
    well_pitch_mm: float
    displayed_well_centres_mm: tuple[float, ...]

    @property
    def focus_offset_from_meniscus_mm(self) -> float:
        """Positive when the focus is above the liquid surface."""

        return self.focus_y_mm - self.meniscus_y_mm

    @property
    def fill_exceeds_well_depth(self) -> bool:
        return self.meniscus_y_mm > self.well_rim_y_mm

    @property
    def top_interwell_web_mm(self) -> float:
        """Nominal material width between adjacent well openings."""

        return self.well_pitch_mm - 2.0 * self.well_top_radius_mm

    @property
    def bottom_interwell_web_mm(self) -> float:
        """Nominal material width between adjacent well floors."""

        return self.well_pitch_mm - 2.0 * self.well_bottom_radius_mm


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
    aperture_radius = inputs.transducer_diameter_mm / 2.0
    # Show enough repeated wells that the local plate cut extends beyond the
    # full transducer aperture.  Two neighbours on either side is the minimum;
    # small-pitch plates naturally show more wells at the same physical scale.
    wells_per_side = max(
        2,
        math.ceil(
            (aperture_radius + plate.well_top_width_mm / 2.0)
            / plate.well_pitch_mm
        ),
    )
    displayed_centres = tuple(
        well_index * plate.well_pitch_mm
        for well_index in range(-wells_per_side, wells_per_side + 1)
    )
    return AcousticStackGeometry(
        aperture_y_mm=0.0,
        water_plate_y_mm=water_plate,
        plate_fluid_y_mm=plate_fluid,
        meniscus_y_mm=plate_fluid + inputs.fluid_height_mm,
        well_rim_y_mm=plate_fluid + plate.well_depth_mm,
        focus_y_mm=float(focus_from_aperture_mm),
        aperture_radius_mm=aperture_radius,
        well_bottom_radius_mm=plate.well_bottom_width_mm / 2.0,
        well_top_radius_mm=plate.well_top_width_mm / 2.0,
        well_pitch_mm=plate.well_pitch_mm,
        displayed_well_centres_mm=displayed_centres,
    )


def _sketch(artist: object) -> object:
    """Give structural outlines a restrained hand-drawn character."""

    set_sketch_params = getattr(artist, "set_sketch_params", None)
    if set_sketch_params is not None:
        set_sketch_params(scale=0.42, length=150.0, randomness=1.1)
    return artist


def _dimension_rail_entry(
    axis: plt.Axes,
    *,
    key: str,
    x_mm: float,
    low_mm: float,
    high_mm: float,
    label_x_mm: float,
    label_y_mm: float,
    label: str,
    color: str,
) -> None:
    """Draw one exact layer span and an uncluttered horizontal callout."""

    arrow = axis.annotate(
        "",
        xy=(x_mm, high_mm),
        xytext=(x_mm, low_mm),
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "linewidth": 1.2,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 7.5,
        },
        annotation_clip=False,
    )
    arrow.set_gid(f"dimension-{key}-arrow")
    for boundary_index, boundary_y in enumerate((low_mm, high_mm)):
        tick = axis.plot(
            [x_mm - 0.22, x_mm + 0.22],
            [boundary_y, boundary_y],
            color=color,
            linewidth=0.95,
            solid_capstyle="round",
            zorder=20,
        )[0]
        tick.set_gid(f"dimension-{key}-tick-{boundary_index}")
    midpoint_y = 0.5 * (low_mm + high_mm)
    leader = axis.plot(
        [x_mm + 0.28, label_x_mm - 0.14, label_x_mm - 0.14],
        [midpoint_y, midpoint_y, label_y_mm],
        color=color,
        linewidth=0.75,
        alpha=0.72,
        zorder=19,
    )[0]
    leader.set_gid(f"dimension-{key}-leader")
    text = axis.text(
        label_x_mm,
        label_y_mm,
        label,
        color="#1c2d57",
        fontsize=11.2,
        fontweight="bold",
        va="center",
        ha="left",
        linespacing=1.12,
        zorder=21,
    )
    text.set_gid(f"dimension-{key}-label")


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


def _well_cavity_polygon(
    centre_x_mm: float,
    geometry: AcousticStackGeometry,
) -> np.ndarray:
    """Return a catalogue-scaled trapezoidal well cavity."""

    return np.asarray(
        [
            [
                centre_x_mm - geometry.well_bottom_radius_mm,
                geometry.plate_fluid_y_mm,
            ],
            [
                centre_x_mm + geometry.well_bottom_radius_mm,
                geometry.plate_fluid_y_mm,
            ],
            [
                centre_x_mm + geometry.well_top_radius_mm,
                geometry.well_rim_y_mm,
            ],
            [
                centre_x_mm - geometry.well_top_radius_mm,
                geometry.well_rim_y_mm,
            ],
        ],
        dtype=float,
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
        "ink": "#1c2d57",
        "muted": "#667085",
        "water": "#dceff5",
        "water_line": "#176c82",
        "plate": "#e3e8ea",
        "plate_dark": "#45646d",
        "liquid": "#f7b455",
        "liquid_light": "#fbd8a2",
        "liquid_line": "#e77925",
        "air": "#fcfdfd",
        "asm_focus": "#e77925",
        "ray": "#176c82",
        "transducer": "#244f59",
        "paper": "#fcfdfd",
        "line": "#d6dbe2",
        "accent": "#00a86b",
    }

    radius = geometry.aperture_radius_mm
    # A pitch-accurate run of wells forms a local cutaway wide enough to cover
    # the transducer.  The plate continues beyond the two crop marks; these are
    # deliberately not rendered as physical SBS plate sidewalls.
    outer_well_centre = max(
        abs(value) for value in geometry.displayed_well_centres_mm
    )
    local_plate_half_width = (
        outer_well_centre + 0.5 * geometry.well_pitch_mm
    )
    drawing_left = -max(radius + 2.0, local_plate_half_width + 0.8)
    drawing_right = max(radius + 2.0, local_plate_half_width + 0.8)
    rail_x = drawing_right + 1.25
    label_x = rail_x + 0.72
    x_limits = (drawing_left - 0.8, label_x + 4.6)
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
    upper_y = max(focus_candidates) + 6.0

    figure, axis = plt.subplots(
        figsize=(6.6, 8.2),
        constrained_layout=True,
    )
    figure.patch.set_facecolor(colors["paper"])
    axis.set_facecolor(colors["air"])

    chamber_bottom_half_width = max(radius + 1.3, 0.72 * local_plate_half_width)
    water_chamber = Polygon(
        [
            (-chamber_bottom_half_width, 0.0),
            (-local_plate_half_width, geometry.water_plate_y_mm),
            (local_plate_half_width, geometry.water_plate_y_mm),
            (chamber_bottom_half_width, 0.0),
        ],
        closed=True,
        facecolor=colors["water"],
        edgecolor="none",
        alpha=0.72,
        zorder=0,
    )
    water_chamber.set_gid("coupling-water")
    axis.add_patch(water_chamber)
    for side in (-1.0, 1.0):
        chamber_wall = axis.plot(
            [
                side * chamber_bottom_half_width,
                side * local_plate_half_width,
            ],
            [0.0, geometry.water_plate_y_mm],
            color=colors["water_line"],
            linewidth=1.05,
            alpha=0.62,
            zorder=1,
        )[0]
        chamber_wall.set_gid(
            "coupling-chamber-left" if side < 0.0 else "coupling-chamber-right"
        )
        _sketch(chamber_wall)

    # Pitch-accurate local source-plate cutaway.  The catalogue's SVG is a
    # useful visual reference but contains only one well; using its numeric
    # JSON fields here lets the neighbouring wells change with the selected
    # plate family while preserving the exact acoustic floor thickness.
    plate_body = Rectangle(
        (-local_plate_half_width, geometry.water_plate_y_mm),
        2.0 * local_plate_half_width,
        geometry.well_rim_y_mm - geometry.water_plate_y_mm,
        facecolor=colors["plate"],
        edgecolor="none",
        alpha=0.95,
        zorder=2,
    )
    plate_body.set_gid("plate-body")
    axis.add_patch(plate_body)

    for centre_x in geometry.displayed_well_centres_mm:
        well_index = int(round(centre_x / geometry.well_pitch_mm))
        cavity = Polygon(
            _well_cavity_polygon(centre_x, geometry),
            closed=True,
            facecolor=colors["air"],
            edgecolor=colors["ink"],
            linewidth=0.9,
            alpha=1.0,
            zorder=3,
        )
        cavity.set_gid(f"well-cavity-{well_index:+d}")
        if not math.isclose(centre_x, 0.0):
            cavity.set_alpha(0.68)
        _sketch(cavity)
        axis.add_patch(cavity)

    # Outline the cropped local plate body without drawing a false line across
    # the open well mouths.  In particular, do not draw vertical lines at the
    # crop boundaries: paired diagonal break marks say explicitly that the
    # physical plate continues beyond this local section.
    plate_floor = axis.plot(
        [-local_plate_half_width, local_plate_half_width],
        [geometry.water_plate_y_mm, geometry.water_plate_y_mm],
        color=colors["ink"],
        linewidth=1.15,
        zorder=4,
    )[0]
    plate_floor.set_gid("plate-water-interface")
    _sketch(plate_floor)
    top_edges = [
        -local_plate_half_width,
        *[
            edge
            for centre in geometry.displayed_well_centres_mm
            for edge in (
                centre - geometry.well_top_radius_mm,
                centre + geometry.well_top_radius_mm,
            )
        ],
        local_plate_half_width,
    ]
    for left_edge, right_edge in zip(
        top_edges[0::2],
        top_edges[1::2],
        strict=True,
    ):
        rim_segment = axis.plot(
            [left_edge, right_edge],
            [geometry.well_rim_y_mm, geometry.well_rim_y_mm],
            color=colors["ink"],
            linewidth=1.05,
            zorder=4,
        )[0]
        _sketch(rim_segment)

    crop_mark_y = geometry.plate_fluid_y_mm + 0.62 * (
        geometry.well_rim_y_mm - geometry.plate_fluid_y_mm
    )
    crop_mark_height = min(
        0.55,
        0.10 * (geometry.well_rim_y_mm - geometry.plate_fluid_y_mm),
    )
    crop_mark_width = min(0.46, 0.14 * geometry.well_pitch_mm)
    for side_name, sign in (("left", -1.0), ("right", 1.0)):
        edge_x = sign * local_plate_half_width
        for mark_index, vertical_offset in enumerate((-0.42, 0.42)):
            line = axis.plot(
                [edge_x - crop_mark_width, edge_x + crop_mark_width],
                [
                    crop_mark_y + vertical_offset - crop_mark_height,
                    crop_mark_y + vertical_offset + crop_mark_height,
                ],
                color=colors["ink"],
                linewidth=1.25,
                solid_capstyle="round",
                clip_on=False,
                zorder=6,
            )[0]
            line.set_gid(f"plate-crop-break-{side_name}-{mark_index}")
            _sketch(line)

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
                zorder=4,
            )
        )
        axis.patches[-1].set_gid("active-well-liquid")
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
                zorder=4,
            )
        )

    meniscus_half_width = float(
        _well_half_width_mm(geometry.meniscus_y_mm, geometry).item()
    )
    meniscus_line = axis.plot(
        [-meniscus_half_width, meniscus_half_width],
        [geometry.meniscus_y_mm, geometry.meniscus_y_mm],
        color=colors["liquid_line"],
        linewidth=2.5,
        solid_capstyle="round",
        zorder=7,
    )[0]
    meniscus_line.set_gid("meniscus")
    _sketch(meniscus_line)

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
    transducer_body = Rectangle(
        (-body_half_width, body_bottom_y),
        2.0 * body_half_width,
        body_top_y - body_bottom_y,
        facecolor=colors["ink"],
        edgecolor="#13203f",
        linewidth=1.1,
        zorder=7,
    )
    transducer_body.set_gid("transducer-body")
    _sketch(transducer_body)
    axis.add_patch(transducer_body)
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
        fontsize=7.8,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=9,
    )
    transducer_cap = axis.fill_between(
        transducer_x,
        transducer_y,
        transducer_y - 0.75,
        color=colors["transducer"],
        alpha=0.95,
        zorder=8,
    )
    transducer_cap.set_gid("transducer-cap")
    transducer_face = axis.plot(
        transducer_x,
        transducer_y,
        color=colors["ink"],
        linewidth=1.7,
        zorder=9,
    )[0]
    transducer_face.set_gid("transducer-face")
    _sketch(transducer_face)
    axis.text(
        body_half_width + 0.2,
        body_bottom_y + 0.12,
        f"Ø {inputs.transducer_diameter_mm:.1f} mm",
        color=colors["muted"],
        fontsize=8.6,
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
    right_ray = axis.plot(
        ray_x,
        ray_y,
        color=colors["ray"],
        linewidth=1.65,
        label="Snell edge rays",
        zorder=6,
    )[0]
    right_ray.set_gid("snell-edge-ray-right")
    _sketch(right_ray)
    left_ray = axis.plot(
        -np.asarray(ray_x),
        ray_y,
        color=colors["ray"],
        linewidth=1.65,
        zorder=6,
    )[0]
    left_ray.set_gid("snell-edge-ray-left")
    _sketch(left_ray)

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
        ray_focus_marker = axis.scatter(
            [0.0],
            [ray.ray_focus_y_mm],
            s=58,
            marker="o",
            facecolor=colors["paper"],
            edgecolor=colors["ray"],
            linewidth=1.7,
            label=ray_focus_label,
            zorder=10,
        )
        ray_focus_marker.set_gid("snell-ray-focus")

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
            f"Critical angle · {ray.critical_interface}",
            xy=(critical_x, critical_y),
            xytext=(drawing_left + 1.0, critical_y + 1.0),
            color=colors["muted"],
            fontsize=7.5,
            ha="left",
            arrowprops={"arrowstyle": "-", "color": colors["muted"], "lw": 0.8},
            zorder=12,
        )

    if focus_from_aperture_mm is not None:
        asm_focus_marker = axis.scatter(
            [0.0],
            [focus_from_aperture_mm],
            s=118,
            marker="*",
            color=colors["asm_focus"],
            edgecolor="#9b491b",
            linewidth=0.7,
            label="ASM axial focus",
            zorder=12,
        )
        asm_focus_marker.set_gid("asm-axial-focus")

    material_short = "PP" if plate.material == "polypropylene" else "COC"
    fill_volume_label = (
        f"≈{plate.estimated_fill_volume_ul(inputs.fluid_height_mm):.2f} µL"
        if 0.0 <= inputs.fluid_height_mm <= plate.well_depth_mm
        else "outside nominal well"
    )
    natural_label_y = [
        0.5 * geometry.water_plate_y_mm,
        0.5 * (geometry.water_plate_y_mm + geometry.plate_fluid_y_mm),
        0.5 * (geometry.plate_fluid_y_mm + geometry.meniscus_y_mm),
    ]
    label_y = [natural_label_y[0]]
    for candidate_y in natural_label_y[1:]:
        label_y.append(max(candidate_y, label_y[-1] + 1.55))
    _dimension_rail_entry(
        axis,
        key="water",
        x_mm=rail_x,
        low_mm=0.0,
        high_mm=geometry.water_plate_y_mm,
        label_x_mm=label_x,
        label_y_mm=label_y[0],
        label=f"WATER  {inputs.water_path_mm:.2f} mm",
        color=colors["water_line"],
    )
    _dimension_rail_entry(
        axis,
        key="plate",
        x_mm=rail_x,
        low_mm=geometry.water_plate_y_mm,
        high_mm=geometry.plate_fluid_y_mm,
        label_x_mm=label_x,
        label_y_mm=label_y[1],
        label=f"{material_short}  {inputs.plate_thickness_mm:.2f} mm",
        color=colors["plate_dark"],
    )
    _dimension_rail_entry(
        axis,
        key="liquid",
        x_mm=rail_x,
        low_mm=geometry.plate_fluid_y_mm,
        high_mm=geometry.meniscus_y_mm,
        label_x_mm=label_x,
        label_y_mm=label_y[2],
        label=(
            f"DMSO {inputs.dmso_volume_percent:.0f} vol.%\n"
            f"{inputs.fluid_height_mm:.2f} mm · {fill_volume_label}"
        ),
        color=colors["liquid_line"],
    )

    rail_separator = axis.plot(
        [rail_x - 0.62, rail_x - 0.62],
        [lower_y + 0.2, upper_y - 0.2],
        color=colors["line"],
        linewidth=0.85,
        zorder=18,
    )[0]
    rail_separator.set_gid("dimension-rail-separator")
    axis.text(
        label_x,
        upper_y - 1.0,
        "LAYER DISTANCES",
        color=colors["muted"],
        fontsize=8.6,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=21,
    )

    axis.text(
        drawing_left + 0.6,
        max(1.2, 0.22 * geometry.water_plate_y_mm),
        "COUPLING WATER",
        color=colors["water_line"],
        fontsize=9.2,
        fontweight="bold",
        alpha=0.78,
        ha="left",
        va="center",
    )
    axis.text(
        0.0,
        geometry.well_rim_y_mm - 0.62,
        "ACTIVE",
        color=colors["ink"],
        fontsize=8.8,
        fontweight="bold",
        ha="center",
        va="top",
        zorder=15,
    )
    axis.annotate(
        "MENISCUS",
        xy=(-meniscus_half_width, geometry.meniscus_y_mm),
        xytext=(-meniscus_half_width - 0.7, geometry.meniscus_y_mm + 0.45),
        color=colors["liquid_line"],
        fontsize=8.6,
        fontweight="bold",
        ha="right",
        va="bottom",
        arrowprops={
            "arrowstyle": "-",
            "color": colors["liquid_line"],
            "lw": 0.8,
        },
        zorder=15,
    )
    ejection_arrow = axis.annotate(
        "",
        xy=(0.0, geometry.meniscus_y_mm + 2.0),
        xytext=(0.0, geometry.meniscus_y_mm + 0.65),
        arrowprops={
            "arrowstyle": "-|>",
            "color": colors["accent"],
            "lw": 1.1,
        },
        zorder=14,
    )
    ejection_arrow.set_gid("ejection-arrow")
    axis.text(
        0.55,
        geometry.meniscus_y_mm + 1.45,
        "AIR / EJECTION",
        color=colors["muted"],
        fontsize=8.6,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=14,
    )

    title_y = upper_y - 0.45
    axis.text(
        drawing_left,
        title_y,
        "ACOUSTIC STACK",
        color=colors["ink"],
        fontsize=16.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    # Keep the symbol key visually separate from the large title at both
    # desktop and phone-scaled rendering sizes.
    key_y = upper_y - 2.25
    key_x = drawing_left
    key_ray = axis.plot(
        [key_x, key_x + 0.85],
        [key_y, key_y],
        color=colors["ray"],
        linewidth=1.6,
        zorder=20,
    )[0]
    _sketch(key_ray)
    axis.text(
        key_x + 1.05,
        key_y,
        "SNELL RAYS",
        color=colors["muted"],
        fontsize=7.8,
        fontweight="bold",
        ha="left",
        va="center",
    )
    focus_key_x = key_x + 5.5
    axis.scatter(
        [focus_key_x],
        [key_y],
        s=34,
        marker="o",
        facecolor=colors["paper"],
        edgecolor=colors["ray"],
        linewidth=1.4,
        zorder=20,
    )
    axis.text(
        focus_key_x + 0.38,
        key_y,
        "RAY FOCUS",
        color=colors["muted"],
        fontsize=7.8,
        fontweight="bold",
        ha="left",
        va="center",
    )
    if focus_from_aperture_mm is not None:
        asm_key_x = focus_key_x + 5.0
        axis.scatter(
            [asm_key_x],
            [key_y],
            s=58,
            marker="*",
            color=colors["asm_focus"],
            edgecolor="#9b491b",
            linewidth=0.5,
            zorder=20,
        )
        axis.text(
            asm_key_x + 0.42,
            key_y,
            "ASM FOCUS",
            color=colors["muted"],
            fontsize=7.8,
            fontweight="bold",
            ha="left",
            va="center",
        )
    axis.text(
        -local_plate_half_width,
        geometry.well_rim_y_mm + 0.28,
        f"{plate.id} · {plate.well_count}-WELL {material_short} · "
        f"{plate.well_pitch_mm:.2f} mm pitch",
        color=colors["plate_dark"],
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=15,
    )

    axis.set(xlim=x_limits, ylim=(lower_y, upper_y))
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    return figure
