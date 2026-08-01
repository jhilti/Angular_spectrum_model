import unittest

import numpy as np

from angular_spectrum.dmso_sweep import _quadratic_peak_position


class DMSOSweepTests(unittest.TestCase):
    def test_peak_position_marks_scan_boundaries(self) -> None:
        coordinate = np.linspace(0.0, 1.0, 11)
        position, boundary_limited = _quadratic_peak_position(
            coordinate,
            coordinate,
        )
        self.assertEqual(position, 1.0)
        self.assertTrue(boundary_limited)

    def test_peak_position_refines_interior_maximum(self) -> None:
        coordinate = np.linspace(0.0, 1.0, 11)
        amplitude = 1.0 - (coordinate - 0.43) ** 2
        position, boundary_limited = _quadratic_peak_position(
            coordinate,
            amplitude,
        )
        self.assertAlmostEqual(position, 0.43)
        self.assertFalse(boundary_limited)


if __name__ == "__main__":
    unittest.main()
