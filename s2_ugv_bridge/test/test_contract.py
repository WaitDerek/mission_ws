import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s2_ugv_bridge.contract import validate_request


class TranslationRequestContractTests(unittest.TestCase):
    def test_zero_speed_preserves_the_existing_ros1_default_behavior(self):
        request = validate_request("forward", 5.0, 0.0)

        self.assertEqual("forward", request.direction)
        self.assertEqual(5.0, request.duration_s)
        self.assertEqual(0.0, request.speed_mps)

    def test_rejects_a_direction_not_supported_by_the_s2_action(self):
        for direction in ("diagonal", "", "FORWARD"):
            with self.subTest(direction=direction):
                with self.assertRaisesRegex(ValueError, "direction"):
                    validate_request(direction, 1.0, 0.0)

    def test_rejects_nonpositive_or_nonfinite_duration_before_docker_is_called(self):
        for duration_s in (0.0, -0.1, math.inf, math.nan):
            with self.subTest(duration_s=duration_s):
                with self.assertRaisesRegex(ValueError, "duration_s"):
                    validate_request("forward", duration_s, 0.0)

    def test_rejects_negative_or_nonfinite_speed_before_docker_is_called(self):
        for speed_mps in (-0.01, math.inf, math.nan):
            with self.subTest(speed_mps=speed_mps):
                with self.assertRaisesRegex(ValueError, "speed_mps"):
                    validate_request("forward", 1.0, speed_mps)
