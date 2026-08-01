import unittest

from angular_spectrum.dmso_mixture import (
    dmso_concentration_to_mole_fraction,
    dmso_water_properties,
    water_properties,
)


class DMSOMixturePropertyTests(unittest.TestCase):
    def test_pure_water_properties_follow_reference_correlations(self) -> None:
        water_20 = water_properties(20.0)
        water_22 = water_properties(22.0)
        water_40 = water_properties(40.0)
        self.assertAlmostEqual(water_20.density_kg_m3, 998.23363614)
        self.assertAlmostEqual(water_20.sound_speed_m_s, 1482.379546752)
        self.assertAlmostEqual(water_22.density_kg_m3, 997.800320317)
        self.assertAlmostEqual(water_22.sound_speed_m_s, 1488.357911239)
        self.assertAlmostEqual(water_40.density_kg_m3, 992.247318629)
        self.assertAlmostEqual(water_40.sound_speed_m_s, 1528.893576064)
        self.assertGreater(water_40.sound_speed_m_s, water_20.sound_speed_m_s)

    def test_pure_water_rejects_temperature_outside_sound_speed_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 and 95"):
            water_properties(96.0)

    def test_zero_percent_mixture_matches_incident_pure_water(self) -> None:
        for temperature_c in (20.0, 22.0, 40.0):
            water = water_properties(temperature_c)
            mixture = dmso_water_properties(
                0.0,
                basis="volume",
                temperature_c=temperature_c,
            )
            self.assertEqual(mixture.sound_speed_m_s, water.sound_speed_m_s)
            self.assertEqual(mixture.density_kg_m3, water.density_kg_m3)

    def test_mixture_properties_are_continuous_at_zero_dmso(self) -> None:
        water = dmso_water_properties(
            0.0,
            basis="volume",
            temperature_c=22.0,
        )
        trace_dmso = dmso_water_properties(
            1.0e-9,
            basis="volume",
            temperature_c=22.0,
        )
        self.assertAlmostEqual(
            trace_dmso.sound_speed_m_s,
            water.sound_speed_m_s,
            places=4,
        )
        self.assertAlmostEqual(
            trace_dmso.density_kg_m3,
            water.density_kg_m3,
            places=4,
        )

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
