import unittest

import numpy as np

from angular_spectrum.analysis import fwhm
from angular_spectrum.grid import CartesianGrid
from angular_spectrum.materials import ElasticPlate, ElasticSolid, Fluid
from angular_spectrum.model import (
    AngularSpectrumModel,
    FocusedCircularAperture,
    validate_focused_grid_support,
)
from angular_spectrum.pulse import propagate_pulse_on_axis


class AngularSpectrumModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        water = Fluid("water", 997.77, 1488.4)
        dmso = Fluid("DMSO", 1098.4, 1499.0)
        polypropylene = ElasticSolid.from_longitudinal_speed_and_poisson(
            name="PP",
            density_kg_m3=900.0,
            longitudinal_speed_m_s=2732.0,
            poisson_ratio=0.42,
        )
        cls.model = AngularSpectrumModel(
            grid=CartesianGrid(nx=256, ny=256, dx_m=60e-6),
            aperture=FocusedCircularAperture(13e-3, 25e-3),
            incident_fluid=water,
            plate=ElasticPlate(polypropylene, 0.78e-3),
            transmitted_fluid=dmso,
            water_path_m=20e-3,
            plate_radial_samples=1024,
        )

    def test_axial_fields_are_finite_and_focus_near_design_range(self) -> None:
        z_after = np.linspace(0.0, 10e-3, 61)
        plate_field = self.model.on_axis_scan_after_plate(10e6, z_after)
        reference = self.model.reference_on_axis_scan(10e6, z_after)
        self.assertTrue(np.all(np.isfinite(plate_field)))
        self.assertTrue(np.all(np.isfinite(reference)))
        reference_focus = (
            self.model.water_path_m
            + self.model.plate.thickness_m
            + z_after[int(np.argmax(np.abs(reference)))]
        )
        self.assertGreater(reference_focus, 23e-3)
        self.assertLess(reference_focus, 27e-3)

    def test_focal_plane_has_expected_shape(self) -> None:
        field = self.model.field_after_plate(10e6, 4e-3)
        self.assertEqual(field.shape, (256, 256))
        self.assertTrue(np.all(np.isfinite(field)))

    def test_axis_scan_matches_full_field_at_zero_distance(self) -> None:
        centre_y, centre_x = self.model.grid.centre_index
        field = self.model.field_after_plate(10.0e6, 0.0)
        scan = self.model.on_axis_scan_after_plate(10.0e6, [0.0])
        self.assertTrue(np.allclose(scan[0], field[centre_y, centre_x]))

        reference_field = self.model.reference_field(10.0e6, 0.0)
        reference_scan = self.model.reference_on_axis_scan(10.0e6, [0.0])
        self.assertTrue(
            np.allclose(
                reference_scan[0],
                reference_field[centre_y, centre_x],
            )
        )

    def test_fwhm_interpolates_gaussian_crossings(self) -> None:
        x = np.linspace(-5.0, 5.0, 10001)
        y = np.exp(-0.5 * x**2)
        self.assertAlmostEqual(fwhm(x, y), 2.35482, places=4)

    def test_grid_support_distinguishes_one_way_from_round_trip(self) -> None:
        validate_focused_grid_support(
            self.model,
            maximum_frequency_hz=10.0e6,
            propagation_segments=(
                (
                    "one-way water path",
                    self.model.incident_fluid,
                    self.model.water_path_m,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "FFT window"):
            validate_focused_grid_support(
                self.model,
                maximum_frequency_hz=10.0e6,
                propagation_segments=(
                    (
                        "water round trip",
                        self.model.incident_fluid,
                        2.0 * self.model.water_path_m,
                    ),
                ),
            )

    def test_combined_mask_matches_unsplit_path_in_one_medium(self) -> None:
        split = self.model._combined_bandlimit_mask(
            10.0e6,
            (
                (self.model.incident_fluid, 10.0e-3),
                (self.model.incident_fluid, 15.0e-3),
            ),
        )
        combined = self.model._combined_bandlimit_mask(
            10.0e6,
            ((self.model.incident_fluid, 25.0e-3),),
        )
        self.assertTrue(np.array_equal(split, combined))

    def test_grid_support_rejects_cumulative_layered_walkoff(self) -> None:
        segment = (
            "individually supported water segment",
            self.model.incident_fluid,
            20.0e-3,
        )
        validate_focused_grid_support(
            self.model,
            maximum_frequency_hz=10.0e6,
            propagation_segments=(segment,),
        )
        with self.assertRaisesRegex(ValueError, "combined layered path"):
            validate_focused_grid_support(
                self.model,
                maximum_frequency_hz=10.0e6,
                propagation_segments=(segment, segment),
            )


class PulseConventionTests(unittest.TestCase):
    def test_numpy_reconstruction_delays_the_signal(self) -> None:
        sample_count = 128
        delta_t = 1e-8
        delay_samples = 17
        time = np.arange(sample_count) * delta_t
        drive = np.zeros(sample_count)
        drive[3] = 1.0

        class DelayModel:
            def on_axis_value_after_plate(self, frequency_hz, z_after_plate_m):
                delay_s = delay_samples * delta_t
                return np.exp(1j * 2.0 * np.pi * frequency_hz * delay_s)

        result = propagate_pulse_on_axis(
            DelayModel(),
            time,
            drive,
            z_after_plate_m=0.0,
            relative_spectrum_threshold=0.0,
            minimum_frequency_hz=1.0,
        )
        self.assertEqual(int(np.argmax(result.output_signal)), 3 + delay_samples)


if __name__ == "__main__":
    unittest.main()
