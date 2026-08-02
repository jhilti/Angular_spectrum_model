import unittest

from angular_spectrum.dmso_mixture import (
    dmso_concentration_to_mole_fraction,
    dmso_water_dynamic_viscosity_pa_s,
    dmso_water_properties,
    dmso_water_surface_tension_n_m,
    water_properties,
)
from angular_spectrum.dmso_transport import (
    _PURE_DMSO_SURFACE_TENSION_N_M,
    _SURFACE_TENSION_DATA,
    _SURFACE_TENSION_TEMPERATURE_C,
    _VISCOSITY_DATA,
    water_surface_tension_n_m,
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
            self.assertGreater(mixture.dynamic_viscosity_pa_s, 0.0)
            self.assertGreater(mixture.surface_tension_n_m, 0.0)
            self.assertFalse(
                mixture.surface_tension_temperature_extrapolated
            )

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

    def test_viscosity_reproduces_all_source_table_nodes(self) -> None:
        for row in _VISCOSITY_DATA:
            dmso_mole_fraction = float(row[0])
            for temperature_c, expected_mpa_s in zip(
                (20.0, 30.0, 40.0),
                row[1:],
                strict=True,
            ):
                with self.subTest(
                    dmso_mole_fraction=dmso_mole_fraction,
                    temperature_c=temperature_c,
                ):
                    actual = dmso_water_dynamic_viscosity_pa_s(
                        dmso_mole_fraction,
                        basis="mole",
                        temperature_c=temperature_c,
                    )
                    self.assertAlmostEqual(
                        actual,
                        float(expected_mpa_s) * 1.0e-3,
                        places=12,
                    )

    def test_viscosity_22c_volume_fraction_regression(self) -> None:
        expected_mpa_s = {
            0.70: 4.0479235319,
            0.80: 3.6926120015,
            0.90: 2.9136196500,
            0.95: 2.5430418078,
            1.00: 2.1349352626,
        }
        for fraction, expected in expected_mpa_s.items():
            with self.subTest(fraction=fraction):
                actual = dmso_water_dynamic_viscosity_pa_s(
                    fraction,
                    basis="volume",
                    temperature_c=22.0,
                )
                self.assertAlmostEqual(actual * 1.0e3, expected, places=9)

    def test_viscosity_has_measured_nonmonotonic_composition_peak(self) -> None:
        viscosities = [
            dmso_water_dynamic_viscosity_pa_s(
                float(row[0]),
                basis="mole",
                temperature_c=20.0,
            )
            for row in _VISCOSITY_DATA
        ]
        maximum_index = max(
            range(len(viscosities)), key=viscosities.__getitem__
        )
        self.assertAlmostEqual(_VISCOSITY_DATA[maximum_index, 0], 0.3459)

    def test_surface_tension_reproduces_all_mixture_nodes(self) -> None:
        for temperature_c, dmso_mole_fraction, expected in (
            _SURFACE_TENSION_DATA
        ):
            with self.subTest(
                dmso_mole_fraction=dmso_mole_fraction,
                temperature_c=temperature_c,
            ):
                actual = dmso_water_surface_tension_n_m(
                    float(dmso_mole_fraction),
                    basis="mole",
                    temperature_c=float(temperature_c),
                )
                self.assertAlmostEqual(actual, float(expected), places=12)

    def test_surface_tension_reproduces_pure_dmso_nodes(self) -> None:
        for temperature_c, expected in zip(
            _SURFACE_TENSION_TEMPERATURE_C,
            _PURE_DMSO_SURFACE_TENSION_N_M,
            strict=True,
        ):
            actual = dmso_water_surface_tension_n_m(
                1.0,
                basis="mole",
                temperature_c=float(temperature_c),
            )
            self.assertAlmostEqual(actual, float(expected), places=12)

    def test_surface_tension_interpolates_nonrectangular_grid(self) -> None:
        self.assertAlmostEqual(
            dmso_water_surface_tension_n_m(
                0.5455, basis="mole", temperature_c=25.0
            ),
            0.04715,
            places=12,
        )
        self.assertAlmostEqual(
            dmso_water_surface_tension_n_m(
                0.7027, basis="mole", temperature_c=27.5
            ),
            0.0437,
            places=12,
        )
        self.assertAlmostEqual(
            dmso_water_surface_tension_n_m(
                0.3506, basis="mole", temperature_c=45.0
            ),
            0.0497,
            places=12,
        )

    def test_surface_tension_22c_volume_fraction_regression_is_flagged(
        self,
    ) -> None:
        expected_n_m = {
            0.70: 0.0526996326,
            0.80: 0.0486985047,
            0.90: 0.0445122104,
            0.95: 0.0428021394,
            1.00: 0.0420771429,
        }
        for fraction, expected in expected_n_m.items():
            with self.subTest(fraction=fraction):
                properties = dmso_water_properties(
                    fraction,
                    basis="volume",
                    temperature_c=22.0,
                )
                self.assertAlmostEqual(
                    properties.surface_tension_n_m,
                    expected,
                    places=9,
                )
                self.assertTrue(
                    properties.surface_tension_temperature_extrapolated
                )

    def test_iapws_water_surface_tension_anchors(self) -> None:
        expected_n_m = {
            20.0: 0.072736140422,
            22.0: 0.072432270479,
            25.0: 0.071972205230,
            40.0: 0.069596312354,
            55.0: 0.067097653684,
        }
        for temperature_c, expected in expected_n_m.items():
            with self.subTest(temperature_c=temperature_c):
                self.assertAlmostEqual(
                    water_surface_tension_n_m(temperature_c),
                    expected,
                    places=11,
                )

    def test_transport_property_temperature_ranges_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "20 and 40"):
            dmso_water_dynamic_viscosity_pa_s(
                0.8, basis="volume", temperature_c=19.9
            )
        with self.assertRaisesRegex(ValueError, "20 and 55"):
            dmso_water_surface_tension_n_m(
                0.8, basis="mole", temperature_c=55.1
            )

    def test_invalid_basis_is_rejected_even_for_pure_water(self) -> None:
        with self.assertRaisesRegex(ValueError, "basis"):
            dmso_water_properties(0.0, basis="invalid", temperature_c=22.0)


if __name__ == "__main__":
    unittest.main()
