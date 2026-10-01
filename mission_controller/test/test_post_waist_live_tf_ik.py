"""Regression checks for the first TF grasp MoveJ after waist motion."""

import unittest
from types import SimpleNamespace

from geometry_msgs.msg import Pose

from mission_runtime.box_waist_planning import BoxWaistPlanningMixin
from mission_runtime.common import MissionError


def target(x):
    pose = Pose()
    pose.position.x = x
    pose.orientation.w = 1.0
    return pose


class Harness(BoxWaistPlanningMixin):
    def __init__(self):
        self.events = []

    def _float(self, name):
        assert name == "waist_workspace_minimum_margin_deg"
        return 0.1

    def _boolean(self, name):
        assert name == "box_ik_joint4_negative_required"
        return True

    def _float_array(self, name):
        if name.endswith("_min_deg"):
            return [-175.0] * 7
        if name.endswith("_max_deg"):
            return [175.0] * 7
        raise AssertionError(name)

    def _publish_box_grasp_feedback(self, _, stage, detail):
        self.events.append((stage, detail))


class Adapter:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def solve_ik(self, arm, pose, seed):
        self.calls.append((arm, list(pose), list(seed)))
        return self.responses.get(arm)


class PostWaistLiveTfIkTest(unittest.TestCase):
    def setUp(self):
        self.planner = Harness()
        self.predicted = SimpleNamespace(
            left_joint_deg=[11.0, 12.0, 13.0, -50.0, 15.0, 16.0, 17.0],
            right_joint_deg=[21.0, 22.0, 23.0, -60.0, 25.0, 26.0, 27.0],
        )

    def test_live_pose_is_solved_instead_of_sending_cached_ik(self):
        left_result = [31.0, 32.0, 33.0, -40.0, 35.0, 36.0, 37.0]
        right_result = [41.0, 42.0, 43.0, -45.0, 45.0, 46.0, 47.0]
        adapter = Adapter({"left": left_result, "right": right_result})
        solved = self.planner._solve_post_waist_live_tf_movej_ik(
            None, adapter, target(0.31), target(0.42), self.predicted,
            right_arm_only=False,
        )
        self.assertEqual(solved, {"left": left_result, "right": right_result})
        self.assertEqual([call[0] for call in adapter.calls], ["left", "right"])
        self.assertAlmostEqual(adapter.calls[0][1][0], 0.31)
        self.assertAlmostEqual(adapter.calls[1][1][0], 0.42)
        self.assertEqual(adapter.calls[0][2], self.predicted.left_joint_deg)
        self.assertEqual(adapter.calls[1][2], self.predicted.right_joint_deg)

    def test_drag_initial_phase_only_solves_right(self):
        right_result = [41.0, 42.0, 43.0, -45.0, 45.0, 46.0, 47.0]
        adapter = Adapter({"right": right_result})
        solved = self.planner._solve_post_waist_live_tf_movej_ik(
            None, adapter, target(0.31), target(0.42), self.predicted,
            right_arm_only=True,
        )
        self.assertEqual(solved, {"right": right_result})
        self.assertEqual([call[0] for call in adapter.calls], ["right"])

    def test_positive_joint4_accepted_when_filter_disabled(self):
        self.planner._boolean = lambda name: False
        left = [0.0, 0.0, 0.0, -37.106, 0.0, 0.0, 0.0]
        right = [0.0, 0.0, 0.0, 4.845, 0.0, 0.0, 0.0]
        adapter = Adapter({"left": left, "right": right})
        solved = self.planner._solve_post_waist_live_tf_movej_ik(
            None, adapter, target(0.31), target(0.42), self.predicted,
            right_arm_only=False,
        )
        self.assertEqual(solved["right"], right)

    def test_invalid_joint4_is_rejected_before_motion(self):
        adapter = Adapter({"left": [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0]})
        with self.assertRaisesRegex(MissionError, "no acceptable offline IK"):
            self.planner._solve_post_waist_live_tf_movej_ik(
                None, adapter, target(0.31), target(0.42), self.predicted,
                right_arm_only=False,
            )
        self.assertGreater(len(adapter.calls), 2)
        self.assertTrue(all(call[0] == "left" for call in adapter.calls))
        self.assertTrue(all(call[1] == adapter.calls[0][1] for call in adapter.calls))


if __name__ == "__main__":
    unittest.main()
