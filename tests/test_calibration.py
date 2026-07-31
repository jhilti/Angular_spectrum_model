import unittest

import numpy as np

from angular_spectrum import (
    apply_reference_transfer,
    estimate_reference_transfer,
)


class ReferenceCalibrationTests(unittest.TestCase):
    def test_reference_transfer_recovers_common_waveform_filter(self) -> None:
        sample_rate_hz = 80.0e6
        time_s = np.arange(4096, dtype=float) / sample_rate_hz
        arrival_s = 12.0e-6
        local_s = time_s - arrival_s
        simulated = (
            np.exp(-0.5 * (local_s / 0.095e-6) ** 2)
            * np.sin(2.0 * np.pi * 10.0e6 * local_s)
        )
        frequency_hz = np.fft.rfftfreq(time_s.size, 1.0 / sample_rate_hz)
        known_response = (
            np.exp(-((frequency_hz / 13.0e6) ** 4))
            * np.exp(-2j * np.pi * frequency_hz * 28.0e-9)
        )
        measured = np.fft.irfft(
            np.fft.rfft(simulated) * known_response,
            n=time_s.size,
        )

        calibration = estimate_reference_transfer(
            time_s,
            measured,
            time_s,
            simulated,
            measured_arrival_s=arrival_s,
            simulated_arrival_s=arrival_s,
            target_time_s=time_s,
            minimum_frequency_hz=3.0e6,
            maximum_frequency_hz=18.0e6,
        )
        corrected = apply_reference_transfer(
            time_s,
            simulated,
            calibration,
        )
        gate = np.abs(local_s) <= 0.30e-6
        raw_correlation = float(np.corrcoef(measured[gate], simulated[gate])[0, 1])
        corrected_correlation = float(
            np.corrcoef(measured[gate], corrected[gate])[0, 1]
        )

        self.assertGreater(corrected_correlation, raw_correlation + 0.05)
        self.assertGreater(corrected_correlation, 0.95)
        self.assertTrue(np.all(np.isfinite(calibration.response)))
        self.assertLessEqual(
            float(np.max(np.abs(calibration.magnitude_correction_db))),
            calibration.maximum_correction_db + 1e-9,
        )

    def test_calibration_must_match_target_time_grid(self) -> None:
        time_s = np.arange(512, dtype=float) / 40.0e6
        local_s = time_s - 4.0e-6
        signal = np.exp(-0.5 * (local_s / 0.12e-6) ** 2) * np.sin(
            2.0 * np.pi * 6.0e6 * local_s
        )
        calibration = estimate_reference_transfer(
            time_s,
            signal,
            time_s,
            signal,
            measured_arrival_s=4.0e-6,
            simulated_arrival_s=4.0e-6,
            target_time_s=time_s,
            minimum_frequency_hz=2.0e6,
            maximum_frequency_hz=12.0e6,
        )
        with self.assertRaisesRegex(ValueError, "time grid"):
            apply_reference_transfer(time_s[:-2], signal[:-2], calibration)


if __name__ == "__main__":
    unittest.main()
