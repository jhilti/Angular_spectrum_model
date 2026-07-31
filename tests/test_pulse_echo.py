import unittest

import numpy as np

from angular_spectrum import (
    AngularSpectrumModel,
    CartesianGrid,
    ElasticPlate,
    ElasticSolid,
    Fluid,
    FocusedCircularAperture,
    asymmetric_gaussian_response,
    simulate_monostatic_pulse_echo,
    sine_burst,
)


class PulseEchoTests(unittest.TestCase):
    def test_asymmetric_response_matches_reported_6db_frequencies(self) -> None:
        frequencies = np.array([4.5e6, 11.0e6, 16.0e6])
        response = asymmetric_gaussian_response(
            frequencies,
            peak_frequency_hz=11.0e6,
            lower_frequency_6db_hz=4.5e6,
            upper_frequency_6db_hz=16.0e6,
        )
        self.assertTrue(np.allclose(response, [0.5, 1.0, 0.5]))

    def test_sine_burst_starts_at_positive_going_zero_crossing(self) -> None:
        time, signal = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=1.0,
            sample_rate_hz=20.0e6,
            record_length_s=3.0e-6,
        )
        self.assertEqual(signal[0], 0.0)
        self.assertGreater(signal[1], 0.0)
        self.assertTrue(np.all(signal[time >= 1.0e-6] == 0.0))
        self.assertAlmostEqual(float(np.sum(signal)), 0.0, places=12)

    def test_monostatic_signal_is_sum_of_reflection_components(self) -> None:
        water = Fluid("water", 1000.0, 1500.0)
        layer = Fluid("layer", 1050.0, 1600.0)
        air = Fluid("air", 1.2, 344.0)
        solid = ElasticSolid.from_longitudinal_speed_and_poisson(
            name="solid",
            density_kg_m3=900.0,
            longitudinal_speed_m_s=2600.0,
            poisson_ratio=0.40,
        )
        model = AngularSpectrumModel(
            grid=CartesianGrid(nx=64, ny=64, dx_m=0.1e-3),
            aperture=FocusedCircularAperture(2.0e-3, 5.0e-3),
            incident_fluid=water,
            plate=ElasticPlate(solid, 0.2e-3),
            transmitted_fluid=layer,
            water_path_m=3.0e-3,
            plate_radial_samples=128,
        )
        time, drive = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=1.0,
            sample_rate_hz=20.0e6,
            record_length_s=12.0e-6,
        )
        result = simulate_monostatic_pulse_echo(
            model,
            time,
            drive,
            fluid_layer_thickness_m=1.0e-3,
            backing_fluid=air,
            relative_spectrum_threshold=0.0,
            minimum_frequency_hz=0.5e6,
            maximum_frequency_hz=1.5e6,
        )
        self.assertTrue(np.all(np.isfinite(result.received_signal)))
        self.assertTrue(
            np.allclose(
                result.received_signal,
                result.plate_front_signal + result.backing_signal,
            )
        )
        self.assertGreater(
            int(np.count_nonzero(result.simulated_bin_mask)),
            0,
        )
        self.assertTrue(
            np.allclose(result.electroacoustic_response, 1.0)
        )

        round_trip = simulate_monostatic_pulse_echo(
            model,
            time,
            drive,
            fluid_layer_thickness_m=1.0e-3,
            backing_fluid=air,
            round_trip_response=np.full(
                np.fft.rfftfreq(time.size, time[1] - time[0]).shape,
                0.5,
            ),
            relative_spectrum_threshold=0.0,
            minimum_frequency_hz=0.5e6,
            maximum_frequency_hz=1.5e6,
        )
        self.assertTrue(
            np.allclose(round_trip.received_signal, 0.5 * result.received_signal)
        )

        with self.assertRaisesRegex(ValueError, "either"):
            simulate_monostatic_pulse_echo(
                model,
                time,
                drive,
                fluid_layer_thickness_m=1.0e-3,
                backing_fluid=air,
                transducer_response=np.ones_like(result.frequency_hz),
                round_trip_response=np.ones_like(result.frequency_hz),
            )


if __name__ == "__main__":
    unittest.main()
