import threading
import time
import unittest

from mission_runtime.box_carry import BoxCarryMixin
from mission_runtime.common import MissionError
from mission_runtime.realman_sdk_adapter import RealManSdkError


class FakeGoal:
    is_cancel_requested = False


class FakeAdapter:
    def __init__(self, fail_retreat=False):
        self.calls = []
        self.fail_retreat = fail_retreat

    def execute_dual(self, left, right, mode, speed, blocking, **kwargs):
        self.calls.append(("retreat", left, right, mode, speed, kwargs))
        if self.fail_retreat:
            raise RealManSdkError("simulated SDK failure")
        return "dual retreat complete"

    def execute_dual_movej(self, left, right, speed, **kwargs):
        self.calls.append(("joint2", left, right, speed))
        return "dual Joint2 complete"


class FakePlace(BoxCarryMixin):
    def __init__(self):
        self.values = {
            "place_box_test_post_release_tool_y_retreat_enabled": True,
            "place_box_test_post_release_arm_movej_enabled": True,
            "place_box_test_post_release_body_home_enabled": False,
            "place_box_test_post_release_arm_home_enabled": False,
            "place_box_test_post_release_tool_y_retreat_m": 0.001,
            "place_box_test_post_release_tool_y_retreat_velocity_percent": 3.0,
            "place_box_test_post_release_tool_y_retreat_timeout_sec": 30.0,
            "place_box_test_post_release_arm_joint2_angle_deg": 60.0,
            "place_box_test_post_release_arm_movej_velocity_percent": 15.0,
            "place_box_test_post_release_arm_movej_timeout_sec": 30.0,
            "place_box_test_post_release_arm_movej_feedback_max_age_sec": 1.0,
        }
        self.joint_state_lock = threading.Lock()
        self.latest_slave_arm_positions = {"left": [0.0] * 7, "right": [0.0] * 7}
        self.latest_slave_arm_state_times = {
            "left": time.monotonic(), "right": time.monotonic()
        }
        self.latest_slave_arm_state_sequences = {"left": 1, "right": 1}

    def _boolean(self, name):
        return bool(self.values[name])

    def _float(self, name):
        return float(self.values[name])

    def _float_array(self, name):
        if name == "place_box_test_post_release_body_home_joint_units":
            return [0.0] * 4
        raise AssertionError(name)

    def _publish_place_box_test_feedback(self, *args):
        pass

    def _wait_for_post_arm_joint_targets(self, *args, **kwargs):
        pass


class TestPlaceBoxReleaseRetreat(unittest.TestCase):
    def test_tool_y_retreat_finishes_before_joint2(self):
        place = FakePlace()
        adapter = FakeAdapter()
        detail = place._execute_place_box_test_post_release(
            FakeGoal(), adapter, False
        )
        self.assertEqual([call[0] for call in adapter.calls], ["retreat", "joint2"])
        retreat = adapter.calls[0]
        self.assertEqual(retreat[1], [0.0, -0.001, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(retreat[2], [0.0, 0.001, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(retreat[3], "movel_offset")
        self.assertEqual(retreat[5]["offset_frame_type"], 1)
        self.assertIn("tool_y_retreat_m=0.0010", detail)

    def test_retreat_failure_prevents_joint2(self):
        place = FakePlace()
        adapter = FakeAdapter(fail_retreat=True)
        with self.assertRaisesRegex(MissionError, "Joint2 was not commanded"):
            place._execute_place_box_test_post_release(FakeGoal(), adapter, False)
        self.assertEqual([call[0] for call in adapter.calls], ["retreat"])


if __name__ == "__main__":
    unittest.main()
