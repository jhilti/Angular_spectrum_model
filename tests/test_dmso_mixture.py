import unittest

from angular_spectrum.dmso_mixture import (
    dmso_concentration_to_mole_fraction,
    dmso_water_properties,
)


class DMSOMixturePropertyTests(unittest.TestCase):
    def test_pure_dmso_interpolates_temperature(self) -> None:
        properties = dmso_water_properties(
            1.0, basis="volume", temperature_c=22.0
        )
        self.assertAlmostEqual(properties.sound_speed_m_s, 1495.74, places=2)
        self.assertAlmostEqual(properties.density_kg_m3, 1098.337, places=3)

    def test_volume_fraction_is_converted_to_mole_fraction(self) -> None:
        mole_fraction = float(
            dmso_concentration_to_mole_fraction(
                0.70, basis="volume", temperature_c=22.0
            )
        )
        self.assertGreater(mole_fraction, 0.36)
        self.assertLess(mole_fraction, 0.38)

    def test_speed_decreases_between_70_and_100_volume_percent(self) -> None:
        speeds = [
            dmso_water_properties(
                fraction, basis="volume", temperature_c=22.0
            ).sound_speed_m_s
            for fraction in (0.70, 0.80, 0.90, 1.00)
        ]
        self.assertTrue(all(a > b for a, b in zip(speeds, speeds[1:])))


if __name__ == "__main__":
    unittest.main()
