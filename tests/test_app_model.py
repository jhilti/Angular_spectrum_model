import unittest

import numpy as np

from angular_spectrum.app_model import (
    NUMERICAL_PRESETS,
    NumericalPreset,
    SimulationInputs,
    _build_model,
    _paraxial_focus_after_plate_m,
    _paraxial_water_path_for_target_m,
    interface_arrivals,
    run_interactive_simulation,
)
from angular_spectrum.dmso_mixture import water_properties


class InteractiveAppModelTests(unittest.TestCase):
    def test_material_attenuation_inputs_are_validated(self) -> None:
        SimulationInputs(
            pp_longitudinal_attenuation_db_per_m=2500.0,
            pp_shear_attenuation_db_per_m=4000.0,
            fluid_attenuation_db_per_m=150.0,
            attenuation_power=1.2,
        ).validate()
        with self.assertRaisesRegex(ValueError, "fluid attenuation"):
            SimulationInputs(fluid_attenuation_db_per_m=-1.0).validate()

    def test_interface_arrivals_are_strictly_ordered(self) -> None:
        inputs = SimulationInputs()
        arrivals = interface_arrivals(
            inputs,
            fluid_sound_speed_m_s=1600.0,
        )
        self.assertLess(arrivals.water_pp_s, arrivals.pp_fluid_s)
        self.assertLess(arrivals.pp_fluid_s, arrivals.fluid_air_s)
        self.assertEqual(
            arrivals.relative_to_water_pp_us["Water–PP"],
            0.0,
        )

    def test_water_arrival_uses_selected_temperature(self) -> None:
        cold = interface_arrivals(
            SimulationInputs(temperature_c=20.0),
            fluid_sound_speed_m_s=1600.0,
        )
        warm = interface_arrivals(
            SimulationInputs(temperature_c=40.0),
            fluid_sound_speed_m_s=1600.0,
        )
        self.assertAlmostEqual(
            cold.water_pp_s,
            2.0 * 25.3e-3 / water_properties(20.0).sound_speed_m_s,
        )
        self.assertAlmostEqual(
            warm.water_pp_s,
            2.0 * 25.3e-3 / water_properties(40.0).sound_speed_m_s,
        )
        self.assertLess(warm.water_pp_s, cold.water_pp_s)

    def test_focus_scan_includes_a_short_entered_water_path(self) -> None:
        NUMERICAL_PRESETS["_TestShortGap"] = NumericalPreset(
            grid_size=64,
            grid_spacing_m=100e-6,
            radial_samples=128,
            sample_rate_hz=40.0e6,
            relative_spectrum_threshold=2.0e-2,
        )
        try:
            inputs = SimulationInputs(
                water_path_mm=0.1,
                plate_thickness_mm=0.20,
                fluid_height_mm=1.0,
                excitation_frequency_mhz=5.0,
                transducer_diameter_mm=2.0,
                transducer_focal_length_mm=5.0,
                numerical_preset="_TestShortGap",
            )
            model = _build_model(inputs)[0]
            from angular_spectrum.app_model import _focus_scans

            result = _focus_scans(model, inputs, 5.0e6)
        finally:
            del NUMERICAL_PRESETS["_TestShortGap"]

        water_path_scan_mm = result[4]
        self.assertLessEqual(water_path_scan_mm[0], inputs.water_path_mm)

    def test_paraxial_layer_focus_uses_slowness_conservation(self) -> None:
        inputs = SimulationInputs(
            water_path_mm=20.0,
            plate_thickness_mm=0.78,
            fluid_height_mm=4.22,
            transducer_focal_length_mm=25.4,
        )
        model = _build_model(inputs)[0]
        expected_focus = (
            (25.4e-3 - 20.0e-3) * model.incident_fluid.sound_speed_m_s
            - 0.78e-3 * model.plate.solid.longitudinal_speed_m_s
        ) / model.transmitted_fluid.sound_speed_m_s
        expected_water = 25.4e-3 - (
            0.78e-3 * model.plate.solid.longitudinal_speed_m_s
            + 4.22e-3 * model.transmitted_fluid.sound_speed_m_s
        ) / model.incident_fluid.sound_speed_m_s
        self.assertAlmostEqual(
            _paraxial_focus_after_plate_m(model, inputs),
            expected_focus,
        )
        self.assertAlmostEqual(
            _paraxial_water_path_for_target_m(model, inputs),
            expected_water,
        )

    def test_small_interactive_run_returns_normalized_outputs(self) -> None:
        NUMERICAL_PRESETS["_Test"] = NumericalPreset(
            grid_size=64,
            grid_spacing_m=100e-6,
            radial_samples=128,
            sample_rate_hz=40.0e6,
            relative_spectrum_threshold=2.0e-2,
        )
        try:
            result = run_interactive_simulation(
                SimulationInputs(
                    water_path_mm=3.0,
                    plate_thickness_mm=0.20,
                    fluid_height_mm=1.0,
                    excitation_frequency_mhz=5.0,
                    transducer_diameter_mm=2.0,
                    transducer_focal_length_mm=5.0,
                    numerical_preset="_Test",
                )
            )
        finally:
            del NUMERICAL_PRESETS["_Test"]

        self.assertTrue(np.all(np.isfinite(result.received_normalized)))
        self.assertAlmostEqual(
            float(np.max(np.abs(result.received_normalized))),
            1.0,
        )
        self.assertTrue(np.all(np.isfinite(result.axial_intensity_normalized)))
        self.assertGreater(result.optimal_water_path_mm, 0.0)
        self.assertGreater(result.optimal_meniscus_intensity_fwhm_mm, 0.0)
        self.assertFalse(result.optimal_water_path_boundary_limited)
        self.assertGreater(result.simulated_frequency_bin_count, 0)


if __name__ == "__main__":
    unittest.main()
