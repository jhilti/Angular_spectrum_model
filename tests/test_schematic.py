from io import BytesIO
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from angular_spectrum.app_model import SimulationInputs
from angular_spectrum.dmso_mixture import (
    dmso_water_properties,
    water_properties,
)
from angular_spectrum.labware import get_labcyte_plate
from angular_spectrum.schematic import (
    acoustic_stack_geometry,
    acoustic_stack_schematic_figure,
    refracted_ray_preview,
)


def test_stack_geometry_tracks_all_selected_dimensions() -> None:
    inputs = SimulationInputs(
        water_path_mm=23.4,
        plate_thickness_mm=1.0,
        fluid_height_mm=3.2,
        transducer_diameter_mm=9.0,
        plate_part_number="LP-0200",
        plate_material_name="cyclic olefin copolymer",
        plate_longitudinal_speed_m_s=2500.0,
        plate_density_kg_m3=1020.0,
        plate_poisson_ratio=0.40,
    )
    geometry = acoustic_stack_geometry(
        inputs,
        focus_from_aperture_mm=27.2,
        plate=get_labcyte_plate("LP-0200"),
    )

    assert geometry.aperture_y_mm == 0.0
    assert geometry.water_plate_y_mm == pytest.approx(23.4)
    assert geometry.plate_fluid_y_mm == pytest.approx(24.4)
    assert geometry.meniscus_y_mm == pytest.approx(27.6)
    assert geometry.focus_y_mm == pytest.approx(27.2)
    assert geometry.aperture_radius_mm == pytest.approx(4.5)
    assert geometry.focus_offset_from_meniscus_mm == pytest.approx(-0.4)
    assert geometry.well_bottom_radius_mm == pytest.approx(1.216)
    assert geometry.well_pitch_mm == pytest.approx(4.5)
    assert geometry.displayed_well_centres_mm == pytest.approx(
        (-9.0, -4.5, 0.0, 4.5, 9.0)
    )
    assert (
        max(abs(value) for value in geometry.displayed_well_centres_mm)
        + geometry.well_top_radius_mm
        >= geometry.aperture_radius_mm
    )
    assert geometry.top_interwell_web_mm == pytest.approx(4.5 - 2.432)
    assert geometry.bottom_interwell_web_mm == pytest.approx(4.5 - 2.432)


@pytest.mark.parametrize("asm_focus_mm", [None, 27.0])
def test_stack_figure_renders_to_portrait_png(
    asm_focus_mm: float | None,
) -> None:
    inputs = SimulationInputs()
    figure = acoustic_stack_schematic_figure(
        inputs,
        focus_from_aperture_mm=asm_focus_mm,
        plate=get_labcyte_plate(),
    )
    legend_labels = figure.axes[0].get_legend_handles_labels()[1]
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=80)
    plt.close(figure)

    assert buffer.getvalue().startswith(b"\x89PNG")
    assert figure.get_figheight() > figure.get_figwidth()
    assert "Snell edge rays" in legend_labels
    assert ("ASM axial focus" in legend_labels) is (asm_focus_mm is not None)
    assert not figure.axes[0].axison
    assert any(
        "DMSO 80 vol.%\n4.22 mm · ≈55.87 µL" in text.get_text()
        for text in figure.axes[0].texts
    )


def test_dimension_rail_tracks_all_three_layer_inputs() -> None:
    inputs = SimulationInputs(
        water_path_mm=24.5,
        plate_thickness_mm=0.78,
        fluid_height_mm=2.0,
        dmso_volume_percent=80.0,
    )
    plate = get_labcyte_plate("PP-0200")
    figure = acoustic_stack_schematic_figure(
        inputs,
        focus_from_aperture_mm=27.0,
        plate=plate,
    )
    axis = figure.axes[0]
    artists = {
        artist.get_gid(): artist
        for artist in axis.get_children()
        if artist.get_gid() is not None
    }

    expected_spans = {
        "water": (0.0, 24.5),
        "plate": (24.5, 25.28),
        "liquid": (25.28, 27.28),
    }
    arrow_x: float | None = None
    for key, (low_mm, high_mm) in expected_spans.items():
        arrow = artists[f"dimension-{key}-arrow"]
        current_x = float(arrow.xy[0])
        if arrow_x is None:
            arrow_x = current_x
        assert current_x == pytest.approx(arrow_x)
        assert arrow.xy == pytest.approx((arrow_x, high_mm))
        assert arrow.xyann == pytest.approx((arrow_x, low_mm))

    assert arrow_x is not None
    geometry = acoustic_stack_geometry(inputs, 27.0, plate)
    cutaway_right = (
        max(geometry.displayed_well_centres_mm)
        + geometry.well_top_radius_mm
    )
    assert cutaway_right < arrow_x < axis.get_xlim()[1]
    assert "WATER  24.50 mm" in artists["dimension-water-label"].get_text()
    assert "PP  0.78 mm" in artists["dimension-plate-label"].get_text()
    assert (
        "DMSO 80 vol.%\n2.00 mm"
        in artists["dimension-liquid-label"].get_text()
    )
    plt.close(figure)


def test_stack_geometry_flags_fill_above_catalogued_well_depth() -> None:
    geometry = acoustic_stack_geometry(
        SimulationInputs(fluid_height_mm=12.0),
        focus_from_aperture_mm=27.0,
        plate=get_labcyte_plate("PP-0200"),
    )
    assert geometry.fill_exceeds_well_depth


@pytest.mark.parametrize(
    (
        "plate_id",
        "expected_pitch_mm",
        "expected_top_width_mm",
        "expected_bottom_width_mm",
    ),
    [
        ("PP-0200", 4.5, 3.8, 3.6),
        ("LP-0200", 4.5, 2.432, 2.432),
        ("LP-0400", 2.25, 1.7, 1.4),
    ],
)
def test_stack_figure_draws_repeated_catalogue_scaled_wells_and_crop_marks(
    plate_id: str,
    expected_pitch_mm: float,
    expected_top_width_mm: float,
    expected_bottom_width_mm: float,
) -> None:
    plate = get_labcyte_plate(plate_id)
    inputs = SimulationInputs(
        plate_part_number=plate.id,
        plate_material_name=plate.material,
        plate_thickness_mm=plate.bottom_thickness_mm,
        plate_longitudinal_speed_m_s=plate.inferred_longitudinal_speed_m_s,
    )
    figure = acoustic_stack_schematic_figure(
        inputs,
        focus_from_aperture_mm=None,
        plate=plate,
    )
    axis = figure.axes[0]
    cavities = [
        patch
        for patch in axis.patches
        if (patch.get_gid() or "").startswith("well-cavity-")
    ]
    liquid = [
        patch
        for patch in axis.patches
        if patch.get_gid() == "active-well-liquid"
    ]

    expected_wells_per_side = max(
        2,
        math.ceil(
            (
                inputs.transducer_diameter_mm / 2.0
                + expected_top_width_mm / 2.0
            )
            / expected_pitch_mm
        ),
    )
    expected_indices = range(
        -expected_wells_per_side,
        expected_wells_per_side + 1,
    )
    assert {patch.get_gid() for patch in cavities} == {
        f"well-cavity-{index:+d}" for index in expected_indices
    }
    assert len(liquid) == 1
    cavity_centres = sorted(
        float(patch.get_xy()[:-1, 0].mean()) for patch in cavities
    )
    assert cavity_centres == pytest.approx(
        [index * expected_pitch_mm for index in expected_indices]
    )
    assert (
        max(abs(value) for value in cavity_centres)
        + expected_top_width_mm / 2.0
        >= inputs.transducer_diameter_mm / 2.0
    )
    for patch in cavities:
        vertices = patch.get_xy()[:-1]
        bottom = vertices[vertices[:, 1] == vertices[:, 1].min(), 0]
        top = vertices[vertices[:, 1] == vertices[:, 1].max(), 0]
        assert max(bottom) - min(bottom) == pytest.approx(
            expected_bottom_width_mm
        )
        assert max(top) - min(top) == pytest.approx(expected_top_width_mm)
    assert any(
        f"{expected_pitch_mm:.2f} mm pitch" in text.get_text()
        for text in axis.texts
    )
    crop_marks = {
        line.get_gid() for line in axis.lines if line.get_gid() is not None
    }
    assert {
        "plate-crop-break-left-0",
        "plate-crop-break-left-1",
        "plate-crop-break-right-0",
        "plate-crop-break-right-1",
    }.issubset(crop_marks)
    plt.close(figure)


def test_refracted_ray_obeys_snell_law_and_kinks_at_interfaces() -> None:
    inputs = SimulationInputs(water_path_mm=19.07)
    ray = refracted_ray_preview(inputs)

    assert ray.critical_interface is None
    assert ray.axial_y_mm == pytest.approx(
        (
            0.0,
            inputs.water_path_mm,
            inputs.water_path_mm + inputs.plate_thickness_mm,
            inputs.water_path_mm
            + inputs.plate_thickness_mm
            + inputs.fluid_height_mm,
        )
    )
    assert len(ray.right_edge_x_mm) == 4
    segment_slopes = [
        abs((x1 - x0) / (y1 - y0))
        for x0, x1, y0, y1 in zip(
            ray.right_edge_x_mm[:-1],
            ray.right_edge_x_mm[1:],
            ray.axial_y_mm[:-1],
            ray.axial_y_mm[1:],
            strict=True,
        )
    ]
    # The faster longitudinal plate branch bends farther from the normal;
    # the DMSO angle lies between the plate and water angles.
    assert segment_slopes[1] > segment_slopes[2] > segment_slopes[0]

    water_speed = water_properties(inputs.temperature_c).sound_speed_m_s
    liquid_speed = dmso_water_properties(
        inputs.dmso_volume_percent / 100.0,
        basis="volume",
        temperature_c=inputs.temperature_c,
    ).sound_speed_m_s
    assert math.sin(math.radians(ray.water_angle_deg)) / water_speed == (
        pytest.approx(ray.transverse_slowness_s_m)
    )
    assert ray.plate_longitudinal_angle_deg is not None
    assert math.sin(
        math.radians(ray.plate_longitudinal_angle_deg)
    ) / inputs.plate_longitudinal_speed_m_s == pytest.approx(
        ray.transverse_slowness_s_m
    )
    assert ray.liquid_angle_deg is not None
    assert math.sin(math.radians(ray.liquid_angle_deg)) / liquid_speed == (
        pytest.approx(ray.transverse_slowness_s_m)
    )
    assert ray.ray_focus_y_mm is not None
    assert ray.focus_is_extrapolated_beyond_meniscus


def test_refracted_ray_stops_when_longitudinal_branch_is_critical() -> None:
    inputs = SimulationInputs(
        water_path_mm=1.0,
        transducer_diameter_mm=15.0,
        transducer_focal_length_mm=2.0,
    )
    ray = refracted_ray_preview(inputs)

    assert ray.critical_interface == "water–plate"
    assert ray.plate_longitudinal_angle_deg is None
    assert ray.liquid_angle_deg is None
    assert ray.ray_focus_y_mm is None
    assert len(ray.axial_y_mm) == 2

    figure = acoustic_stack_schematic_figure(
        inputs,
        focus_from_aperture_mm=None,
        plate=get_labcyte_plate(),
    )
    plt.close(figure)
