import unittest
from unittest.mock import patch

import numpy as np

from angular_spectrum.materials import ElasticPlate, ElasticSolid, Fluid
from angular_spectrum.plate import (
    elastic_plate_scattering,
    elastic_plate_scattering_map,
    fluid_interface_scattering,
    normal_power_transmission,
    vertical_wavenumber,
)


class ElasticPlateScatteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.water = Fluid("water", 1000.0, 1480.0)
        self.copper = ElasticSolid("copper", 8900.0, 4660.0, 2260.0)
        self.plate = ElasticPlate(self.copper, 1.0e-3)
        self.frequency_hz = 5.0e6

    def test_vertical_wavenumber_uses_decaying_branch(self) -> None:
        k = 100.0
        kz = vertical_wavenumber(k, np.array([0.0, 50.0, 120.0]))
        self.assertGreater(kz[0].real, 0.0)
        self.assertGreater(kz[1].real, 0.0)
        self.assertGreater(kz[2].imag, 0.0)

    def test_independent_loss_inputs_cannot_create_active_bulk_modulus(self) -> None:
        non_passive = ElasticSolid.from_longitudinal_speed_and_poisson(
            name="non-passive PP",
            density_kg_m3=900.0,
            longitudinal_speed_m_s=2732.0,
            poisson_ratio=0.42,
            longitudinal_attenuation_db_per_m=0.0,
            shear_attenuation_db_per_m=4000.0,
        )
        with self.assertRaisesRegex(ValueError, "non-passive"):
            non_passive.complex_lame_parameters(10.0e6)

        passive = ElasticSolid.from_longitudinal_speed_and_poisson(
            name="passive PP",
            density_kg_m3=900.0,
            longitudinal_speed_m_s=2732.0,
            poisson_ratio=0.42,
            longitudinal_attenuation_db_per_m=2500.0,
            shear_attenuation_db_per_m=4000.0,
        )
        lam, mu = passive.complex_lame_parameters(10.0e6)
        self.assertLessEqual(mu.imag, 0.0)
        self.assertLessEqual((lam + 2.0 * mu / 3.0).imag, 0.0)

    def test_normal_incidence_matches_acoustic_layer_formula(self) -> None:
        transmission = elastic_plate_scattering(
            np.array([0.0]),
            self.frequency_hz,
            self.water,
            self.plate,
            self.water,
        )[1][0]

        z1 = self.water.density_kg_m3 * self.water.sound_speed_m_s
        z2 = self.copper.density_kg_m3 * self.copper.longitudinal_speed_m_s
        k2 = 2.0 * np.pi * self.frequency_hz / self.copper.longitudinal_speed_m_s
        t12 = 2.0 * z2 / (z1 + z2)
        t23 = 2.0 * z1 / (z2 + z1)
        r21 = (z1 - z2) / (z1 + z2)
        r23 = (z1 - z2) / (z1 + z2)
        phase = np.exp(1j * k2 * self.plate.thickness_m)
        expected = t12 * t23 * phase / (1.0 - r21 * r23 * phase**2)
        self.assertLess(abs(transmission - expected), 1e-12)

    def test_lossless_energy_is_conserved_at_oblique_incidence(self) -> None:
        angle = np.linspace(0.0, 30.0, 121)
        q = (
            2.0
            * np.pi
            * self.frequency_hz
            / self.water.sound_speed_m_s
            * np.sin(np.radians(angle))
        )
        reflection, transmission = elastic_plate_scattering(
            q,
            self.frequency_hz,
            self.water,
            self.plate,
            self.water,
        )
        transmitted_power = normal_power_transmission(
            q,
            self.frequency_hz,
            self.water,
            self.water,
            transmission,
        )
        balance = np.abs(reflection) ** 2 + transmitted_power
        self.assertLess(float(np.max(np.abs(balance - 1.0))), 2e-11)

    def test_asymmetric_fluids_conserve_energy_in_both_directions(self) -> None:
        other_fluid = Fluid("other fluid", 1250.0, 1700.0)
        maximum_q = 0.75 * min(
            abs(self.water.wavenumber(self.frequency_hz)),
            abs(other_fluid.wavenumber(self.frequency_hz)),
        )
        q = np.linspace(0.0, maximum_q, 121)

        for left, right in (
            (self.water, other_fluid),
            (other_fluid, self.water),
        ):
            reflection, transmission = elastic_plate_scattering(
                q,
                self.frequency_hz,
                left,
                self.plate,
                right,
            )
            transmitted_power = normal_power_transmission(
                q,
                self.frequency_hz,
                left,
                right,
                transmission,
            )
            balance = np.abs(reflection) ** 2 + transmitted_power
            self.assertLess(float(np.max(np.abs(balance - 1.0))), 2e-11)

    def test_asymmetric_pressure_transmission_is_reciprocal(self) -> None:
        other_fluid = Fluid("other fluid", 1250.0, 1700.0)
        maximum_q = 0.75 * min(
            abs(self.water.wavenumber(self.frequency_hz)),
            abs(other_fluid.wavenumber(self.frequency_hz)),
        )
        q = np.linspace(0.0, maximum_q, 121)
        _, transmission_forward = elastic_plate_scattering(
            q,
            self.frequency_hz,
            self.water,
            self.plate,
            other_fluid,
        )
        _, transmission_reverse = elastic_plate_scattering(
            q,
            self.frequency_hz,
            other_fluid,
            self.plate,
            self.water,
        )
        admittance_water = vertical_wavenumber(
            self.water.wavenumber(self.frequency_hz), q
        ) / self.water.density_kg_m3
        admittance_other = vertical_wavenumber(
            other_fluid.wavenumber(self.frequency_hz), q
        ) / other_fluid.density_kg_m3

        np.testing.assert_allclose(
            transmission_forward * admittance_other,
            transmission_reverse * admittance_water,
            rtol=2e-12,
            atol=2e-12,
        )

    def test_direct_map_solves_unique_radii_and_matches_direct_result(self) -> None:
        q = np.array(
            [
                [0.0, 1000.0, 2000.0, 1000.0],
                [2000.0, 3000.0, 0.0, 3000.0],
            ]
        )
        expected = elastic_plate_scattering(
            q,
            self.frequency_hz,
            self.water,
            self.plate,
            self.water,
        )

        with patch(
            "angular_spectrum.plate.elastic_plate_scattering",
            wraps=elastic_plate_scattering,
        ) as solve:
            actual = elastic_plate_scattering_map(
                q,
                self.frequency_hz,
                self.water,
                self.plate,
                self.water,
                radial_samples=None,
            )

        solved_q = np.asarray(solve.call_args.args[0])
        self.assertEqual(solved_q.size, np.unique(q).size)
        self.assertEqual(actual[0].shape, q.shape)
        self.assertEqual(actual[1].shape, q.shape)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_identical_fluid_interface_is_finite_at_exact_grazing(self) -> None:
        q = np.array([self.water.wavenumber(self.frequency_hz).real])
        reflection, transmission = fluid_interface_scattering(
            q,
            self.frequency_hz,
            self.water,
            self.water,
        )

        np.testing.assert_array_equal(reflection, np.array([0.0 + 0.0j]))
        np.testing.assert_array_equal(transmission, np.array([1.0 + 0.0j]))

    def test_same_speed_interface_has_density_limit_at_exact_grazing(self) -> None:
        dense_fluid = Fluid("dense fluid", 1250.0, self.water.sound_speed_m_s)
        q = np.array([self.water.wavenumber(self.frequency_hz).real])
        reflection, transmission = fluid_interface_scattering(
            q,
            self.frequency_hz,
            self.water,
            dense_fluid,
        )
        expected_reflection = (
            dense_fluid.density_kg_m3 - self.water.density_kg_m3
        ) / (dense_fluid.density_kg_m3 + self.water.density_kg_m3)
        expected_transmission = (
            2.0 * dense_fluid.density_kg_m3
            / (dense_fluid.density_kg_m3 + self.water.density_kg_m3)
        )

        self.assertTrue(np.isfinite(reflection[0]))
        self.assertTrue(np.isfinite(transmission[0]))
        self.assertAlmostEqual(reflection[0].real, expected_reflection)
        self.assertAlmostEqual(reflection[0].imag, 0.0)
        self.assertAlmostEqual(transmission[0].real, expected_transmission)
        self.assertAlmostEqual(transmission[0].imag, 0.0)


if __name__ == "__main__":
    unittest.main()
