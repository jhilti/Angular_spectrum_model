import unittest

import numpy as np

from angular_spectrum import (
    AngularSpectrumModel,
    CartesianGrid,
    ElasticPlate,
    ElasticSolid,
    Fluid,
    FocusedCircularAperture,
    optimal_meniscus_intensity_sweep,
)


class MeniscusIntensityTests(unittest.TestCase):
    def test_focused_sweep_returns_finite_positive_intensities(self) -> None:
        water = Fluid("water", 1000.0, 1500.0)
        layer = Fluid("layer", 1050.0, 1600.0)
        air = Fluid("air", 1.2, 344.0)
        solid = ElasticSolid.from_longitudinal_speed_and_poisson(
            name="solid",
            density_kg_m3=900.0,
            longitudinal_speed_m_s=2600.0,
            poisson_ratio=0.40,
        )
        model = AngularSpectrumModel(
            grid=CartesianGrid(nx=64, ny=64, dx_m=0.1e-3),
            aperture=FocusedCircularAperture(2.0e-3, 5.0e-3),
            incident_fluid=water,
            plate=ElasticPlate(solid, 0.2e-3),
            transmitted_fluid=layer,
            water_path_m=3.0e-3,
            plate_radial_samples=128,
        )
        heights = np.linspace(0.8e-3, 1.2e-3, 7)
        result = optimal_meniscus_intensity_sweep(
            model,
            1.0e6,
            heights,
            backing_fluid=air,
        )
        self.assertEqual(result.height_m.shape, heights.shape)
        self.assertTrue(
            np.all(np.isfinite(result.intensity_with_reverberation_w_m2))
        )
        self.assertTrue(np.all(result.single_pass_intensity_w_m2 > 0.0))
        self.assertTrue(np.all(result.interference_gain > 0.0))


if __name__ == "__main__":
    unittest.main()
