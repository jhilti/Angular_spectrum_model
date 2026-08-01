import unittest

from angular_spectrum import square_burst


class PulseTests(unittest.TestCase):
    def test_square_burst_rejects_silent_record_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete square burst"):
            square_burst(
                center_frequency_hz=1.0e6,
                cycles=4.0,
                sample_rate_hz=20.0e6,
                record_length_s=3.0e-6,
            )


if __name__ == "__main__":
    unittest.main()
