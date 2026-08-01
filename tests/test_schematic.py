from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from angular_spectrum.app_model import SimulationInputs
from angular_spectrum.labware import get_labcyte_plate
from angular_spectrum.schematic import (
    acoustic_stack_geometry,
    acoustic_stack_schematic_figure,
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


def test_stack_figure_renders_to_png() -> None:
    inputs = SimulationInputs()
    figure = acoustic_stack_schematic_figure(
        inputs,
        focus_from_aperture_mm=27.0,
        plate=get_labcyte_plate(),
    )
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=80)
    plt.close(figure)

    assert buffer.getvalue().startswith(b"\x89PNG")


def test_stack_geometry_flags_fill_above_catalogued_well_depth() -> None:
    geometry = acoustic_stack_geometry(
        SimulationInputs(fluid_height_mm=12.0),
        focus_from_aperture_mm=27.0,
        plate=get_labcyte_plate("PP-0200"),
    )
    assert geometry.fill_exceeds_well_depth
