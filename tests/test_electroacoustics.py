import unittest

import numpy as np

from angular_spectrum import (
    AngularSpectrumModel,
    ButterworthVanDyke,
    CartesianGrid,
    ElasticPlate,
    ElasticSolid,
    ElectroAcousticCalibration,
    Fluid,
    FocusedCircularAperture,
    simulate_electroacoustic_pulse_echo,
    simulate_monostatic_pulse_echo,
    sine_burst,
    solve_thevenin_drive,
)


class ElectricalDriveTests(unittest.TestCase):
    def test_resistive_thevenin_divider_voltage_current_and_energy(self) -> None:
        time, source_voltage = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=2.0,
            sample_rate_hz=40.0e6,
            record_length_s=4.0e-6,
            amplitude=10.0,
        )
        result = solve_thevenin_drive(
            time,
            source_voltage,
            source_impedance_ohm=50.0,
            transducer_impedance_ohm=50.0,
        )

        self.assertTrue(
            np.allclose(result.terminal_voltage_v, 0.5 * source_voltage)
        )
        self.assertTrue(
            np.allclose(result.terminal_current_a, source_voltage / 100.0)
        )
        self.assertAlmostEqual(result.peak_terminal_voltage_v, 5.0)
        self.assertAlmostEqual(result.peak_terminal_current_a, 0.1)
        self.assertAlmostEqual(result.delivered_energy_j, 0.5e-6, places=12)

    def test_bvd_model_has_requested_series_resonance(self) -> None:
        motional_capacitance = 10.0e-12
        target_resonance = 10.0e6
        motional_inductance = 1.0 / (
            (2.0 * np.pi * target_resonance) ** 2 * motional_capacitance
        )
        bvd = ButterworthVanDyke(
            static_capacitance_f=100.0e-12,
            motional_resistance_ohm=20.0,
            motional_inductance_h=motional_inductance,
            motional_capacitance_f=motional_capacitance,
            series_resistance_ohm=1.0,
        )
        impedance = bvd.impedance(np.array([0.0, target_resonance, 15.0e6]))

        self.assertAlmostEqual(bvd.series_resonance_hz, target_resonance)
        self.assertTrue(np.isinf(impedance[0]))
        self.assertTrue(np.all(np.isfinite(impedance[1:])))
        self.assertTrue(np.all(impedance[1:].real > 0.0))

    def test_active_load_impedance_is_rejected(self) -> None:
        time, source_voltage = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=1.0,
            sample_rate_hz=20.0e6,
            record_length_s=3.0e-6,
        )
        with self.assertRaisesRegex(ValueError, "passive"):
            solve_thevenin_drive(
                time,
                source_voltage,
                source_impedance_ohm=50.0,
                transducer_impedance_ohm=-10.0,
            )


class ElectroAcousticPulseEchoTests(unittest.TestCase):
    @staticmethod
    def _model() -> tuple[AngularSpectrumModel, Fluid]:
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
            aperture=FocusedCircularAperture(
                2.0e-3,
                5.0e-3,
                pressure_amplitude_pa=2.0,
            ),
            incident_fluid=water,
            plate=ElasticPlate(solid, 0.2e-3),
            transmitted_fluid=layer,
            water_path_m=3.0e-3,
            plate_radial_samples=128,
        )
        return model, air

    def test_calibrated_wrapper_matches_explicit_pressure_chain(self) -> None:
        model, air = self._model()
        time, source_voltage = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=1.0,
            sample_rate_hz=20.0e6,
            record_length_s=12.0e-6,
            amplitude=1.0,
        )
        calibration = ElectroAcousticCalibration(
            transmit_pressure_pa_per_v=4.0,
            receive_voltage_v_per_pa=0.25,
            receiver_response=2.0,
            adc_counts_per_v=1000.0,
            absolute=True,
        )
        result = simulate_electroacoustic_pulse_echo(
            model,
            time,
            source_voltage,
            source_impedance_ohm=50.0,
            transducer_impedance_ohm=50.0,
            calibration=calibration,
            fluid_layer_thickness_m=1.0e-3,
            backing_fluid=air,
            relative_spectrum_threshold=0.0,
            minimum_frequency_hz=0.5e6,
            maximum_frequency_hz=1.5e6,
        )
        explicit = simulate_monostatic_pulse_echo(
            model,
            time,
            source_voltage,
            fluid_layer_thickness_m=1.0e-3,
            backing_fluid=air,
            relative_spectrum_threshold=0.0,
            minimum_frequency_hz=0.5e6,
            maximum_frequency_hz=1.5e6,
        )

        self.assertTrue(
            np.allclose(result.aperture_pressure_pa, 2.0 * source_voltage)
        )
        self.assertTrue(
            np.allclose(result.returned_pressure_pa, explicit.received_signal)
        )
        self.assertTrue(
            np.allclose(result.received_voltage_v, 0.5 * explicit.received_signal)
        )
        self.assertTrue(
            np.allclose(
                result.received_adc_counts,
                1000.0 * result.received_voltage_v,
            )
        )
        self.assertTrue(result.absolute_calibration)

    def test_received_voltage_scales_linearly_with_source_voltage(self) -> None:
        model, air = self._model()
        time, source_voltage = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=1.0,
            sample_rate_hz=20.0e6,
            record_length_s=12.0e-6,
        )
        calibration = ElectroAcousticCalibration(
            transmit_pressure_pa_per_v=1.0,
            receive_voltage_v_per_pa=1.0,
        )
        common = {
            "source_impedance_ohm": 50.0,
            "transducer_impedance_ohm": 50.0,
            "calibration": calibration,
            "fluid_layer_thickness_m": 1.0e-3,
            "backing_fluid": air,
            "relative_spectrum_threshold": 0.0,
            "minimum_frequency_hz": 0.5e6,
            "maximum_frequency_hz": 1.5e6,
        }
        low = simulate_electroacoustic_pulse_echo(
            model,
            time,
            source_voltage,
            **common,
        )
        high = simulate_electroacoustic_pulse_echo(
            model,
            time,
            3.0 * source_voltage,
            **common,
        )

        self.assertTrue(
            np.allclose(high.received_voltage_v, 3.0 * low.received_voltage_v)
        )
        self.assertAlmostEqual(
            high.electrical.delivered_energy_j,
            9.0 * low.electrical.delivered_energy_j,
        )

    def test_frequency_shaped_transmit_uses_unfiltered_source_support(self) -> None:
        model, air = self._model()
        time, source_voltage = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=1.0,
            sample_rate_hz=20.0e6,
            record_length_s=12.0e-6,
        )
        calibration = ElectroAcousticCalibration(
            transmit_pressure_pa_per_v=lambda frequency: np.exp(
                -0.5 * ((frequency - 1.0e6) / 0.35e6) ** 2
            ),
            receive_voltage_v_per_pa=1.0,
        )
        result = simulate_electroacoustic_pulse_echo(
            model,
            time,
            source_voltage,
            source_impedance_ohm=50.0,
            transducer_impedance_ohm=50.0,
            calibration=calibration,
            fluid_layer_thickness_m=1.0e-3,
            backing_fluid=air,
            relative_spectrum_threshold=1.0e-3,
            minimum_frequency_hz=0.2e6,
            maximum_frequency_hz=1.8e6,
        )
        self.assertGreater(result.acoustic.fluid_cavity_echo_count, 0)
        self.assertGreater(abs(result.aperture_pressure_pa[-1]), 0.0)

    def test_receive_input_impedance_applies_explicit_loading(self) -> None:
        model, air = self._model()
        time, source_voltage = sine_burst(
            center_frequency_hz=1.0e6,
            cycles=1.0,
            sample_rate_hz=20.0e6,
            record_length_s=12.0e-6,
        )
        common = {
            "source_impedance_ohm": 50.0,
            "transducer_impedance_ohm": 50.0,
            "calibration": ElectroAcousticCalibration(
                transmit_pressure_pa_per_v=1.0,
                receive_voltage_v_per_pa=1.0,
            ),
            "fluid_layer_thickness_m": 1.0e-3,
            "backing_fluid": air,
            "relative_spectrum_threshold": 0.0,
            "minimum_frequency_hz": 0.5e6,
            "maximum_frequency_hz": 1.5e6,
        }
        open_circuit = simulate_electroacoustic_pulse_echo(
            model,
            time,
            source_voltage,
            **common,
        )
        loaded = simulate_electroacoustic_pulse_echo(
            model,
            time,
            source_voltage,
            receiver_input_impedance_ohm=50.0,
            **common,
        )
        self.assertTrue(
            np.allclose(
                loaded.received_voltage_v,
                0.5 * open_circuit.received_voltage_v,
            )
        )
        self.assertTrue(np.allclose(loaded.receiver_loading_response, 0.5))


if __name__ == "__main__":
    unittest.main()
