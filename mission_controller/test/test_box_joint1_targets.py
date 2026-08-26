import math
import unittest
from copy import deepcopy

from geometry_msgs.msg import Pose

from mission_runtime.mission_controller import MissionController


class _Harness:
    VALUES = {
        "box_joint1_axis_xyz": [0.0, 0.0, 1.0],
        "box_joint1_feedback_to_geometric_sign": -1.0,
        "box_joint1_to_left_base_xyz": [0.0, 1.05, -0.05],
        "box_joint1_to_right_base_xyz": [0.0, 1.05, 0.05],
        "box_joint1_to_left_base_rotation": [
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
        ],
        "box_joint1_to_right_base_rotation": [
            0.0, -1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        ],
    }

    def _float_array(self, name):
        return self.VALUES[name]

    def _float(self, name):
        return float(self.VALUES[name])


class TestBoxJoint1Targets(unittest.TestCase):
    def test_clockwise_feedback_reexpresses_frozen_targets(self):
        target = Pose()
        target.orientation.w = 1.0
        delta = math.radians(18.677)

        left = MissionController._reexpress_link8_target_after_joint1_rotation(
            _Harness(), target, "left", delta
        )
        right = MissionController._reexpress_link8_target_after_joint1_rotation(
            _Harness(), target, "right", delta
        )

        self.assertAlmostEqual(left.position.x, -0.055294, places=5)
        self.assertAlmostEqual(left.position.y, -0.336244, places=5)
        self.assertAlmostEqual(left.position.z, 0.0, places=6)
        self.assertAlmostEqual(left.orientation.z, -0.162267, places=5)
        self.assertAlmostEqual(left.orientation.w, 0.986747, places=5)

        self.assertAlmostEqual(right.position.x, -0.055294, places=5)
        self.assertAlmostEqual(right.position.y, 0.336244, places=5)
        self.assertAlmostEqual(right.position.z, 0.0, places=6)
        self.assertAlmostEqual(right.orientation.z, 0.162267, places=5)
        self.assertAlmostEqual(right.orientation.w, 0.986747, places=5)

    def test_zero_rotation_keeps_target_unchanged(self):
        target = Pose()
        target.position.x = 0.1
        target.position.y = -0.2
        target.position.z = 0.3
        target.orientation.x = 0.1
        target.orientation.y = 0.2
        target.orientation.z = 0.3
        target.orientation.w = math.sqrt(0.86)

        result = MissionController._reexpress_link8_target_after_joint1_rotation(
            _Harness(), target, "left", 0.0
        )

        expected = [
            target.position.x,
            target.position.y,
            target.position.z,
            target.orientation.x,
            target.orientation.y,
            target.orientation.z,
            target.orientation.w,
        ]
        actual = [
            result.position.x,
            result.position.y,
            result.position.z,
            result.orientation.x,
            result.orientation.y,
            result.orientation.z,
            result.orientation.w,
        ]
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=9)

    def test_orientation_only_mode_keeps_position_exactly(self):
        target = Pose()
        target.position.x = 0.072
        target.position.y = 0.549
        target.position.z = 0.447
        target.orientation.x = -0.490
        target.orientation.y = -0.501
        target.orientation.z = -0.486
        target.orientation.w = 0.522

        result = MissionController._rotate_link8_orientation_after_joint1_rotation(
            _Harness(), target, "left", math.radians(18.119)
        )

        self.assertEqual(
            [result.position.x, result.position.y, result.position.z],
            [0.072, 0.549, 0.447],
        )
        self.assertNotEqual(
            [
                result.orientation.x,
                result.orientation.y,
                result.orientation.z,
                result.orientation.w,
            ],
            [-0.490, -0.501, -0.486, 0.522],
        )

    def test_post_movel_targets_are_cumulative_and_keep_orientation(self):
        class PostMoveHarness:
            VALUES = {
                "box_post_movel_left_step1_xyz": [0.0, 0.0, -0.08],
                "box_post_movel_right_step1_xyz": [0.0, 0.0, -0.05],
                "box_post_movel_left_step2_xyz": [0.09, 0.0, 0.0],
                "box_post_movel_right_step2_xyz": [0.09, 0.0, 0.0],
                "direct_movel_left_box_to_link8_orientation": [-0.550, -0.402, -0.584, 0.442],
                "direct_movel_right_box_to_link8_orientation": [0.571, -0.410, 0.569, 0.428],
            }

            def _float_array(self, name):
                return self.VALUES[name]

            @staticmethod
            def _integer(name):
                assert name == "box_post_movel_step_count"
                return 2

            def _translate_pose_in_box_frame(self, target, delta_xyz, arm):
                return MissionController._translate_pose_in_box_frame(
                    self, target, delta_xyz, arm
                )

            @staticmethod
            def _string(name):
                assert name == "direct_movel_target_mode"
                return "camera_offset_box_orientation"

        left = Pose()
        left.position.x = 0.324
        left.position.y = 0.539
        left.position.z = 0.446
        left.orientation.x = -0.550
        left.orientation.y = -0.402
        left.orientation.z = -0.584
        left.orientation.w = 0.442
        right = Pose()
        right.position.x = 0.344
        right.position.y = -0.536
        right.position.z = 0.367
        right.orientation.x = 0.571
        right.orientation.y = -0.410
        right.orientation.z = 0.569
        right.orientation.w = 0.428

        targets = MissionController._post_movel_targets(
            PostMoveHarness(), left, right
        )
        left_step1, right_step1 = targets[0]
        left_step2, right_step2 = targets[1]

        self.assertEqual(
            [left_step1.position.x, left_step1.position.y, left_step1.position.z],
            [0.324, 0.539, 0.366],
        )
        self.assertEqual(
            [right_step1.position.x, right_step1.position.y, right_step1.position.z],
            [0.344, -0.536, 0.317],
        )
        for actual, expected in zip(
            [left_step2.position.x, left_step2.position.y, left_step2.position.z],
            [0.414, 0.539, 0.366],
        ):
            self.assertAlmostEqual(actual, expected, places=9)
        for actual, expected in zip(
            [right_step2.position.x, right_step2.position.y, right_step2.position.z],
            [0.434, -0.536, 0.317],
        ):
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertEqual(left_step2.orientation, left.orientation)
        self.assertEqual(right_step2.orientation, right.orientation)

    def test_post_movel_box_delta_rotates_into_arm_base(self):
        class RotatedHarness:
            VALUES = {
                "box_post_movel_left_step1_xyz": [1.0, 0.0, 0.0],
                "box_post_movel_right_step1_xyz": [1.0, 0.0, 0.0],
                "box_post_movel_left_step2_xyz": [0.0, 0.0, 0.0],
                "box_post_movel_right_step2_xyz": [0.0, 0.0, 0.0],
                "direct_movel_left_box_to_link8_orientation": [0.0, 0.0, 0.0, 1.0],
                "direct_movel_right_box_to_link8_orientation": [0.0, 0.0, 0.0, 1.0],
            }

            def _float_array(self, name):
                return self.VALUES[name]

            @staticmethod
            def _integer(name):
                assert name == "box_post_movel_step_count"
                return 2

            def _translate_pose_in_box_frame(self, target, delta_xyz, arm):
                return MissionController._translate_pose_in_box_frame(
                    self, target, delta_xyz, arm
                )

            @staticmethod
            def _string(name):
                assert name == "direct_movel_target_mode"
                return "camera_offset_box_orientation"

        target = Pose()
        target.orientation.z = 2.0 ** -0.5
        target.orientation.w = 2.0 ** -0.5
        targets = MissionController._post_movel_targets(
            RotatedHarness(), target, deepcopy(target)
        )
        left_step1, right_step1 = targets[0]
        self.assertAlmostEqual(left_step1.position.x, 0.0, places=9)
        self.assertAlmostEqual(left_step1.position.y, 1.0, places=9)
        self.assertAlmostEqual(right_step1.position.x, 0.0, places=9)
        self.assertAlmostEqual(right_step1.position.y, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
