import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from angular_spectrum import (
    interpret_survey_geometry,
    load_survey_pulse_echo,
    parse_survey_pulse_echo,
)


class SurveyImportTests(unittest.TestCase):
    @staticmethod
    def _document() -> dict:
        sample_rate_hz = 20.0e6
        signal = np.zeros(200)
        signal[60:65] = [0.0, 1.0, -2.0, 1.0, 0.0]
        return {
            "DateTime": "2026-01-01T00:00:00Z",
            "FluidMaterial": None,
            "SampleRangeAnalysisStartUSecs": 10.0,
            "SurveyResult": {
                "PlateBaseSampleTimeUSecs": 12.0,
                "WellBaseSampleTimeUSecs": 12.5,
                "FluidTopSampleTimeUSecs": 14.0,
                "ProbeToPlateBaseDistance": 9.0,
                "WellBaseThickness": 0.65,
                "FluidHeight": 1.125,
            },
            "SurveyPingsSuper": [
                {
                    "SampleFrequency": sample_rate_hz,
                    "SampleIndexStart": 7,
                    "ProbeFrequency": 10.0e6,
                    "ToneLength": 1,
                    "ProbeVoltage": 100.0,
                    "SignalData": signal.tolist(),
                }
            ],
        }

    def test_in_memory_parser_matches_file_loader(self) -> None:
        document = self._document()
        parsed = parse_survey_pulse_echo(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "survey.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = load_survey_pulse_echo(path)
        self.assertTrue(np.array_equal(parsed.signal_adc, loaded.signal_adc))
        self.assertEqual(parsed.sample_rate_hz, loaded.sample_rate_hz)
        self.assertEqual(parsed.sample_index_start, 7)
        self.assertEqual(parsed.probe_frequency_hz, 10.0e6)
        self.assertEqual(parsed.tone_length_cycles, 1.0)
        self.assertEqual(parsed.probe_voltage_setting_v, 100.0)
        self.assertFalse(parsed.excitation_metadata_is_calibrated)
        self.assertAlmostEqual(parsed.time_s[0], 10.0e-6)
        self.assertAlmostEqual(parsed.time_since_excitation_s[0], 10.0e-6)
        self.assertAlmostEqual(parsed.relative_time_s[0], -2.0e-6)

    def test_geometry_exposes_stored_sound_speed_assumptions(self) -> None:
        document = self._document()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "survey.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            survey = load_survey_pulse_echo(path)

        self.assertAlmostEqual(float(np.max(np.abs(survey.normalized_signal))), 1.0)
        self.assertIsNone(survey.fluid_material)
        geometry = interpret_survey_geometry(
            survey,
            incident_sound_speed_m_s=1480.0,
            plate_longitudinal_speed_m_s=2600.0,
            fluid_sound_speed_m_s=1600.0,
        )
        self.assertAlmostEqual(geometry.implied_incident_speed_m_s, 1500.0)
        self.assertAlmostEqual(geometry.implied_plate_speed_m_s, 2600.0)
        self.assertAlmostEqual(geometry.implied_fluid_speed_m_s, 1500.0)
        self.assertAlmostEqual(geometry.tof_probe_to_plate_m, 8.88e-3)
        self.assertAlmostEqual(geometry.tof_fluid_height_m, 1.2e-3)
        self.assertAlmostEqual(geometry.stored_height_timing_error_s, -0.09375e-6)


if __name__ == "__main__":
    unittest.main()
