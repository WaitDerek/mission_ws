import unittest

from mission_runtime.box_carry import BoxCarryMixin


class TestPlaceBoxForceUnload(unittest.TestCase):
    def test_measured_box_unload_exceeds_default_thresholds(self):
        # Baseline is the carried-box reading; current is the unloaded,
        # table-supported reading.
        left = BoxCarryMixin._place_box_test_fz_unload_metric(
            -855.0, -11790.0, 1.0
        )
        right = BoxCarryMixin._place_box_test_fz_unload_metric(
            -1797.0, -12331.0, 1.0
        )
        self.assertAlmostEqual(left, 10935.0)
        self.assertAlmostEqual(right, 10534.0)
        self.assertGreater(left, 7000.0)
        self.assertGreater(right, 7000.0)

    def test_force_x_is_not_an_input(self):
        self.assertEqual(
            BoxCarryMixin._place_box_test_fz_unload_metric(-1000.0, -12000.0, 1.0),
            11000.0,
        )


if __name__ == "__main__":
    unittest.main()
