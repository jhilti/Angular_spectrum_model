"""Minimal editable example for the measured polypropylene case."""

import numpy as np

from angular_spectrum import (
    AngularSpectrumModel,
    CartesianGrid,
    ElasticPlate,
    ElasticSolid,
    Fluid,
    FocusedCircularAperture,
    fwhm,
)


frequency_hz = 10.0e6
water_path_m = 20.0e-3  # Replace with the measured front-face position.

water = Fluid("water_22C", density_kg_m3=997.77, sound_speed_m_s=1488.4)
dmso = Fluid("DMSO_22C", density_kg_m3=1098.4, sound_speed_m_s=1499.0)
polypropylene = ElasticSolid.from_longitudinal_speed_and_poisson(
    name="polypropylene",
    density_kg_m3=900.0,
    longitudinal_speed_m_s=2732.0,
    poisson_ratio=0.42,
)

model = AngularSpectrumModel(
    grid=CartesianGrid(nx=512, ny=512, dx_m=40e-6),
    aperture=FocusedCircularAperture(
        diameter_m=13.0e-3,
        focal_length_m=25.0e-3,
    ),
    incident_fluid=water,
    plate=ElasticPlate(polypropylene, thickness_m=0.78e-3),
    transmitted_fluid=dmso,
    water_path_m=water_path_m,
)

z_after_plate_m = np.linspace(0.0, 10.0e-3, 161)
axis_pressure = model.on_axis_scan_after_plate(
    frequency_hz, z_after_plate_m
)
focus_index = int(np.argmax(np.abs(axis_pressure)))
focus_after_plate_m = float(z_after_plate_m[focus_index])
focus_physical_m = (
    water_path_m + model.plate.thickness_m + focus_after_plate_m
)

focal_plane = model.field_after_plate(
    frequency_hz, focus_after_plate_m
)
centre_y, _ = model.grid.centre_index
lateral_width_m = fwhm(model.grid.x_m, focal_plane[centre_y, :])

print(f"Fokus: {focus_physical_m * 1e3:.3f} mm ab Aperturebene")
print(f"Laterale -6-dB-Breite: {lateral_width_m * 1e3:.3f} mm")
