import unittest

import numpy as np

from angular_spectrum.materials import ElasticPlate, ElasticSolid, Fluid
from angular_spectrum.plate import (
    elastic_plate_scattering,
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


if __name__ == "__main__":
    unittest.main()
