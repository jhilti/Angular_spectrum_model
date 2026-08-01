import unittest

import numpy as np

from angular_spectrum import (
    capillary_pressure_scale_pa,
    linear_ade_screening,
    perfect_reflector_radiation_pressure_pa,
    plane_wave_intensity_w_m2,
)


class ADELinearScreeningTests(unittest.TestCase):
    def test_spot_and_cavity_scales(self) -> None:
        result = linear_ade_screening(
            frequency_hz=10.0e6,
            sound_speed_m_s=1600.0,
            focal_length_m=25.4e-3,
            aperture_diameter_m=13.0e-3,
            liquid_height_m=2.5e-3,
            tone_cycles=1.0,
        )
        self.assertAlmostEqual(result.wavelength_m, 160.0e-6)
        self.assertAlmostEqual(result.f_number, 25.4 / 13.0)
        self.assertAlmostEqual(
            result.diffraction_spot_diameter_m,
            1.02 * (25.4 / 13.0) * 160.0e-6,
        )
        self.assertAlmostEqual(result.liquid_cavity_round_trip_s, 3.125e-6)
        self.assertAlmostEqual(result.liquid_cavity_round_trip_cycles, 31.25)
        self.assertTrue(result.first_cavity_return_after_drive)

    def test_pressure_intensity_and_capillary_scales(self) -> None:
        intensity = plane_wave_intensity_w_m2(
            2.0e6,
            density_kg_m3=1000.0,
            sound_speed_m_s=1500.0,
        )
        self.assertAlmostEqual(intensity, 2.0e6**2 / (2.0 * 1000.0 * 1500.0))
        self.assertAlmostEqual(
            perfect_reflector_radiation_pressure_pa(
                intensity,
                sound_speed_m_s=1500.0,
            ),
            2.0 * intensity / 1500.0,
        )
        self.assertAlmostEqual(
            capillary_pressure_scale_pa(
                surface_tension_n_m=0.05,
                diameter_m=0.5e-3,
            ),
            400.0,
        )


if __name__ == "__main__":
    unittest.main()
