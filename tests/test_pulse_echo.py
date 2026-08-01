import unittest
from unittest.mock import patch

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
    smooth_dc_block_response,
)
from angular_spectrum.pulse_echo import _monostatic_transfer_components


class PulseEchoTests(unittest.TestCase):
    def test_smooth_dc_block_has_no_nonphysical_dc_tail(self) -> None:
        response = smooth_dc_block_response(
            np.array([0.0, 0.5e6, 1.0e6, 5.0e6])
        )
        self.assertEqual(response[0], 0.0)
        self.assertTrue(np.all(np.diff(response) > 0.0))
        self.assertGreater(response[-1], 1.0 - 1.0e-12)

    @staticmethod
    def _model() -> AngularSpectrumModel:
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
            aperture=FocusedCircularAperture(2.0e-3, 5.0e-3),
            incident_fluid=water,
            plate=ElasticPlate(solid, 0.2e-3),
            transmitted_fluid=layer,
            water_path_m=3.0e-3,
            plate_radial_samples=128,
        )

    def test_asymmetric_response_matches_reported_6db_frequencies(self) -> None:
        frequencies = np.array([4.5e6, 11.0e6, 16.0e6])
        response = asymmetric_gaussian_response(
            frequencies,
            peak_frequency_hz=11.0e6,
            lower_frequency_6db_hz=4.5e6,
            upper_frequency_6db_hz=16.0e6,
        )
        self.assertTrue(np.allclose(response, [0.5, 1.0, 0.5]))

    def test_callable_scalar_response_is_broadcast(self) -> None:
        model = self._model()
        air = Fluid("air", 1.2, 344.0)
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
            round_trip_response=lambda frequency: 0.5,
            minimum_frequency_hz=0.5e6,
            maximum_frequency_hz=1.5e6,
        )
        self.assertTrue(np.allclose(result.electroacoustic_response, 0.5))

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

    def test_sine_burst_rejects_silent_record_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete sine burst"):
            sine_burst(
                center_frequency_hz=1.0e6,
                cycles=4.0,
                sample_rate_hz=20.0e6,
                record_length_s=3.0e-6,
            )

    def test_monostatic_signal_is_sum_of_reflection_components(self) -> None:
        model = self._model()
        air = Fluid("air", 1.2, 344.0)
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
        self.assertEqual(result.fluid_cavity_echo_count, 4)
        self.assertTrue(
            np.allclose(
                result.applied_round_trip_transfer,
                np.conj(result.physical_round_trip_phasor),
            )
        )
        self.assertTrue(
            np.allclose(
                result.received_spectrum,
                result.drive_spectrum
                * result.electroacoustic_response
                * result.applied_round_trip_transfer,
            )
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

    def test_short_record_rejects_wrapped_surface_echo(self) -> None:
        model = self._model()
        air = Fluid("air", 1.2, 344.0)
        time, drive = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=1.0,
            sample_rate_hz=20.0e6,
            record_length_s=5.0e-6,
        )
        with self.assertRaisesRegex(ValueError, "plate-front"):
            simulate_monostatic_pulse_echo(
                model,
                time,
                drive,
                fluid_layer_thickness_m=1.0e-3,
                backing_fluid=air,
                relative_spectrum_threshold=0.0,
                minimum_frequency_hz=0.5e6,
                maximum_frequency_hz=1.5e6,
            )

    def test_front_only_record_omits_surface_without_wrapping(self) -> None:
        model = self._model()
        air = Fluid("air", 1.2, 344.0)
        time, drive = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=1.0,
            sample_rate_hz=20.0e6,
            record_length_s=7.0e-6,
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
        self.assertEqual(result.fluid_cavity_echo_count, 0)
        self.assertTrue(np.allclose(result.backing_signal, 0.0))
        self.assertGreater(float(np.max(np.abs(result.plate_front_signal))), 0.0)

    def test_explicit_cavity_count_cannot_wrap(self) -> None:
        model = self._model()
        air = Fluid("air", 1.2, 344.0)
        time, drive = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=1.0,
            sample_rate_hz=20.0e6,
            record_length_s=12.0e-6,
        )
        zero = simulate_monostatic_pulse_echo(
            model,
            time,
            drive,
            fluid_layer_thickness_m=1.0e-3,
            backing_fluid=air,
            fluid_cavity_echo_count=0,
            relative_spectrum_threshold=0.0,
            minimum_frequency_hz=0.5e6,
            maximum_frequency_hz=1.5e6,
        )
        self.assertEqual(zero.fluid_cavity_echo_count, 0)
        self.assertTrue(np.allclose(zero.backing_signal, 0.0))
        with self.assertRaisesRegex(ValueError, "wrap"):
            simulate_monostatic_pulse_echo(
                model,
                time,
                drive,
                fluid_layer_thickness_m=1.0e-3,
                backing_fluid=air,
                fluid_cavity_echo_count=100,
                relative_spectrum_threshold=0.0,
                minimum_frequency_hz=0.5e6,
                maximum_frequency_hz=1.5e6,
            )

    def test_monostatic_components_bandlimit_full_round_trip_distances(
        self,
    ) -> None:
        model = self._model()
        frequency_hz = 1.0e6
        layer_thickness_m = 2.0e-3
        _, _, q = model.grid.spectral_mesh()
        zeros = np.zeros_like(q, dtype=np.complex128)
        ones = np.ones_like(q, dtype=np.complex128)

        # Isolate a unit-transmission plate and a perfectly reflecting backing
        # so that the expected transfer contains only the two round-trip
        # propagators.  This makes the anti-alias masks directly observable.
        with patch(
            "angular_spectrum.pulse_echo.elastic_plate_scattering_map",
            side_effect=[(zeros, ones), (ones, ones)],
        ), patch(
            "angular_spectrum.pulse_echo.fluid_interface_scattering",
            return_value=(ones, ones),
        ):
            front, backing = _monostatic_transfer_components(
                model,
                frequency_hz,
                fluid_layer_thickness_m=layer_thickness_m,
                backing_fluid=Fluid("backing", 1200.0, 1700.0),
                fluid_cavity_echo_count=1,
            )

        water_round_trip = model._propagator(
            model.incident_fluid,
            frequency_hz,
            2.0 * model.water_path_m,
        )
        layer_round_trip = model._propagator(
            model.transmitted_fluid,
            frequency_hz,
            2.0 * layer_thickness_m,
        )
        source_pressure = model.source_pressure(frequency_hz)
        source_spectrum = np.fft.fft2(source_pressure)
        receiver_weight = (
            source_pressure / model.aperture.pressure_amplitude_pa
        )
        receiver_normalization = float(
            np.sum(np.abs(receiver_weight) ** 2)
        )
        expected_field = np.fft.ifft2(
            source_spectrum
            * water_round_trip
            * layer_round_trip
            * model._combined_bandlimit_mask(
                frequency_hz,
                (
                    (model.incident_fluid, 2.0 * model.water_path_m),
                    (model.transmitted_fluid, 2.0 * layer_thickness_m),
                ),
            )
        )
        expected_backing = np.sum(receiver_weight * expected_field) / (
            receiver_normalization
        )

        water_one_way = model._propagator(
            model.incident_fluid,
            frequency_hz,
            model.water_path_m,
        )
        self.assertTrue(
            np.any(
                (np.abs(water_one_way) > 0.0)
                & (np.abs(water_round_trip) == 0.0)
            )
        )
        self.assertAlmostEqual(abs(front), 0.0, places=14)
        self.assertTrue(np.allclose(backing, expected_backing))

    def test_each_cavity_order_uses_its_total_liquid_distance(self) -> None:
        model = self._model()
        frequency_hz = 1.0e6
        layer_thickness_m = 2.0e-3
        _, _, q = model.grid.spectral_mesh()
        zeros = np.zeros_like(q, dtype=np.complex128)
        ones = np.ones_like(q, dtype=np.complex128)
        with patch(
            "angular_spectrum.pulse_echo.elastic_plate_scattering_map",
            side_effect=[(zeros, ones), (ones, ones)],
        ), patch(
            "angular_spectrum.pulse_echo.fluid_interface_scattering",
            return_value=(ones, ones),
        ):
            _, backing = _monostatic_transfer_components(
                model,
                frequency_hz,
                fluid_layer_thickness_m=layer_thickness_m,
                backing_fluid=Fluid("backing", 1200.0, 1700.0),
                fluid_cavity_echo_count=2,
            )

        expected_map = model._propagator(
            model.transmitted_fluid,
            frequency_hz,
            2.0 * layer_thickness_m,
        ) * model._combined_bandlimit_mask(
            frequency_hz,
            (
                (model.incident_fluid, 2.0 * model.water_path_m),
                (model.transmitted_fluid, 2.0 * layer_thickness_m),
            ),
        ) + model._propagator(
            model.transmitted_fluid,
            frequency_hz,
            4.0 * layer_thickness_m,
        ) * model._combined_bandlimit_mask(
            frequency_hz,
            (
                (model.incident_fluid, 2.0 * model.water_path_m),
                (model.transmitted_fluid, 4.0 * layer_thickness_m),
            ),
        )
        source_pressure = model.source_pressure(frequency_hz)
        returned = np.fft.ifft2(
            np.fft.fft2(source_pressure)
            * model._propagator(
                model.incident_fluid,
                frequency_hz,
                2.0 * model.water_path_m,
            )
            * model._combined_bandlimit_mask(
                frequency_hz,
                ((model.incident_fluid, 2.0 * model.water_path_m),),
            )
            * expected_map
        )
        receiver_weight = source_pressure / model.aperture.pressure_amplitude_pa
        expected = np.sum(receiver_weight * returned) / np.sum(
            np.abs(receiver_weight) ** 2
        )
        self.assertTrue(np.allclose(backing, expected))


if __name__ == "__main__":
    unittest.main()
