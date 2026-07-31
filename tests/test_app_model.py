import unittest

import numpy as np

from angular_spectrum.app_model import (
    NUMERICAL_PRESETS,
    NumericalPreset,
    SimulationInputs,
    interface_arrivals,
    run_interactive_simulation,
)


class InteractiveAppModelTests(unittest.TestCase):
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
        self.assertGreater(result.simulated_frequency_bin_count, 0)


if __name__ == "__main__":
    unittest.main()
