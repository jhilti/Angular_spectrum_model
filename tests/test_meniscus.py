import unittest

import numpy as np

from angular_spectrum import (
    AngularSpectrumModel,
    CartesianGrid,
    ElasticPlate,
    ElasticSolid,
    Fluid,
    FocusedCircularAperture,
    configured_meniscus_cavity_metric,
    optimal_meniscus_intensity_sweep,
)


class MeniscusIntensityTests(unittest.TestCase):
    @staticmethod
    def _model(*, pressure_amplitude_pa: float = 1.0) -> AngularSpectrumModel:
        water = Fluid("water", 1000.0, 1500.0)
        layer = Fluid("layer", 1050.0, 1600.0)
        solid = ElasticSolid.from_longitudinal_speed_and_poisson(
            name="solid",
            density_kg_m3=900.0,
            longitudinal_speed_m_s=2600.0,
            poisson_ratio=0.40,
        )
        return AngularSpectrumModel(
            grid=CartesianGrid(nx=64, ny=64, dx_m=0.1e-3),
            aperture=FocusedCircularAperture(
                2.0e-3,
                5.0e-3,
                pressure_amplitude_pa=pressure_amplitude_pa,
            ),
            incident_fluid=water,
            plate=ElasticPlate(solid, 0.2e-3),
            transmitted_fluid=layer,
            water_path_m=3.0e-3,
            plate_radial_samples=128,
        )

    def test_focused_sweep_returns_finite_positive_intensities(self) -> None:
        air = Fluid("air", 1.2, 344.0)
        model = self._model()
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
        self.assertEqual(result.total_interface_pressure_pa.shape, heights.shape)
        self.assertEqual(
            result.total_interface_normal_velocity_m_s.shape,
            heights.shape,
        )
        self.assertEqual(
            result.total_interface_normal_displacement_m.shape,
            heights.shape,
        )
        self.assertTrue(result.phase_optimized_independently_per_height)
        self.assertTrue(result.is_ideal_phase_conjugate_upper_bound)
        self.assertTrue(
            result.pressure_phase_optimized_independently_per_height
        )
        self.assertTrue(result.is_phase_only_pressure_upper_bound)
        self.assertTrue(np.all(result.cavity_orders_retained >= 2))
        self.assertTrue(np.all(result.cavity_series_converged))

    def test_intensity_scales_with_aperture_pressure_squared(self) -> None:
        air = Fluid("air", 1.2, 344.0)
        heights = [1.0e-3]
        unit_pressure = optimal_meniscus_intensity_sweep(
            self._model(pressure_amplitude_pa=1.0),
            1.0e6,
            heights,
            backing_fluid=air,
        )
        double_pressure = optimal_meniscus_intensity_sweep(
            self._model(pressure_amplitude_pa=2.0),
            1.0e6,
            heights,
            backing_fluid=air,
        )
        np.testing.assert_allclose(
            double_pressure.single_pass_intensity_w_m2,
            4.0 * unit_pressure.single_pass_intensity_w_m2,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            double_pressure.intensity_with_reverberation_w_m2,
            4.0 * unit_pressure.intensity_with_reverberation_w_m2,
            rtol=1e-12,
        )

    def test_pressure_release_has_near_zero_pressure_and_finite_velocity(
        self,
    ) -> None:
        model = self._model()
        pressure_release = Fluid("near-vacuum", 1.0e-9, 344.0)
        result = optimal_meniscus_intensity_sweep(
            model,
            1.0e6,
            [1.0e-3],
            backing_fluid=pressure_release,
        )
        pressure = result.total_interface_pressure_pa[0]
        velocity = result.total_interface_normal_velocity_m_s[0]
        displacement = result.total_interface_normal_displacement_m[0]

        self.assertLess(abs(pressure), 1.0e-8)
        self.assertTrue(np.isfinite(velocity))
        self.assertGreater(abs(velocity), 0.0)
        self.assertTrue(np.isfinite(displacement))
        np.testing.assert_allclose(
            displacement,
            velocity / (-1j * 2.0 * np.pi * 1.0e6),
            rtol=1e-13,
        )

    def test_matched_backing_has_no_cavity_exposure_change(self) -> None:
        model = self._model()
        result = configured_meniscus_cavity_metric(
            model,
            1.0e6,
            1.0e-3,
            backing_fluid=model.transmitted_fluid,
            excitation_cycles=1.0,
        )

        self.assertAlmostEqual(result.coherent_power_gain, 1.0, places=12)
        self.assertAlmostEqual(
            result.narrowband_separated_pass_exposure_gain,
            1.0,
            places=12,
        )
        self.assertAlmostEqual(result.coherent_percent_change, 0.0, places=10)
        self.assertAlmostEqual(
            result.narrowband_separated_pass_percent_change,
            0.0,
            places=10,
        )
        self.assertEqual(result.cavity_orders_retained, 2)
        self.assertTrue(result.cavity_series_converged)

    def test_metric_separates_short_pulse_and_coherent_cw_limits(self) -> None:
        model = self._model()
        air = Fluid("air", 1.2, 344.0)
        result = configured_meniscus_cavity_metric(
            model,
            1.0e6,
            1.0e-3,
            backing_fluid=air,
            excitation_cycles=1.0,
            height_uncertainty_m=0.05e-3,
            height_sensitivity_samples=9,
        )

        self.assertEqual(
            result.electrical_overlap_regime,
            "electrical-burst-shorter",
        )
        self.assertLess(
            result.electrical_burst_duration_s,
            result.cavity_round_trip_s,
        )
        self.assertGreaterEqual(
            result.narrowband_separated_pass_exposure_gain,
            1.0,
        )
        self.assertGreater(result.coherent_power_gain, 0.0)
        self.assertLessEqual(
            result.coherent_gain_height_min,
            result.coherent_power_gain,
        )
        self.assertGreaterEqual(
            result.coherent_gain_height_max,
            result.coherent_power_gain,
        )
        self.assertLessEqual(
            result.narrowband_gain_height_min,
            result.narrowband_separated_pass_exposure_gain,
        )
        self.assertGreaterEqual(
            result.narrowband_gain_height_max,
            result.narrowband_separated_pass_exposure_gain,
        )
        self.assertTrue(result.sensitivity_series_converged)
        self.assertTrue(any("not net energy" in item for item in result.limitations))

    def test_metric_power_scales_but_dimensionless_gains_do_not(self) -> None:
        air = Fluid("air", 1.2, 344.0)
        unit = configured_meniscus_cavity_metric(
            self._model(pressure_amplitude_pa=1.0),
            1.0e6,
            1.0e-3,
            backing_fluid=air,
            excitation_cycles=10.0,
        )
        doubled = configured_meniscus_cavity_metric(
            self._model(pressure_amplitude_pa=2.0),
            1.0e6,
            1.0e-3,
            backing_fluid=air,
            excitation_cycles=10.0,
        )

        self.assertEqual(
            unit.electrical_overlap_regime,
            "electrical-burst-long",
        )
        self.assertAlmostEqual(
            doubled.first_pass_forward_power_w,
            4.0 * unit.first_pass_forward_power_w,
            places=15,
        )
        self.assertAlmostEqual(
            doubled.coherent_power_gain,
            unit.coherent_power_gain,
            places=12,
        )
        self.assertAlmostEqual(
            doubled.narrowband_separated_pass_exposure_gain,
            unit.narrowband_separated_pass_exposure_gain,
            places=12,
        )

    def test_metric_rejects_height_uncertainty_larger_than_fill(self) -> None:
        with self.assertRaisesRegex(ValueError, "height_uncertainty_m"):
            configured_meniscus_cavity_metric(
                self._model(),
                1.0e6,
                1.0e-3,
                backing_fluid=Fluid("air", 1.2, 344.0),
                excitation_cycles=1.0,
                height_uncertainty_m=1.0e-3,
            )


if __name__ == "__main__":
    unittest.main()
