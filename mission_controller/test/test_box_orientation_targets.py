import unittest
from copy import deepcopy

from geometry_msgs.msg import PoseStamped

from mission_runtime.mission_controller import MissionController


class _Harness:
    VALUES = {
        "direct_movel_left_box_to_link8_orientation": [
            0.559715925759,
            0.501217990744,
            -0.485326226240,
            0.447165587148,
        ],
        "direct_movel_right_box_to_link8_orientation": [
            0.495125041989,
            -0.526225123507,
            0.461789015629,
            0.514479559584,
        ],
        "left_fixture_center_in_link8_xyz": [-0.07, -0.11, 0.03],
        "right_fixture_center_in_link8_xyz": [-0.07, 0.11, 0.03],
        "direct_movel_left_offset_xyz": [1.0, 0.0, 0.0],
    }

    def _float_array(self, name):
        return self.VALUES[name]

    @staticmethod
    def _string(name):
        assert name == "direct_movel_target_mode"
        return "camera_offset_box_orientation"

    @staticmethod
    def _boolean(name):
        assert name == "direct_movel_fixture_compensation_enabled"
        return True


class TestBoxOrientationTargets(unittest.TestCase):
    def test_box_frame_offset_follows_box_orientation(self):
        harness = _Harness()
        camera_pose = PoseStamped()
        camera_pose.pose.orientation.w = 1.0
        first = MissionController._apply_box_frame_grasp_offset(
            harness, camera_pose, "direct_movel_left_offset_xyz"
        )

        rotated_box_pose = deepcopy(camera_pose)
        rotated_box_pose.pose.orientation.z = 2.0 ** -0.5
        rotated_box_pose.pose.orientation.w = 2.0 ** -0.5
        second = MissionController._apply_box_frame_grasp_offset(
            harness, rotated_box_pose, "direct_movel_left_offset_xyz"
        )

        self.assertAlmostEqual(first.pose.position.x, 1.0, places=9)
        self.assertAlmostEqual(first.pose.position.y, 0.0, places=9)
        self.assertAlmostEqual(second.pose.position.x, 0.0, places=9)
        self.assertAlmostEqual(second.pose.position.y, 1.0, places=9)
        self.assertAlmostEqual(second.pose.position.z, 0.0, places=9)

    def test_box_orientation_does_not_change_position_without_offset(self):
        harness = _Harness()
        box_pose = PoseStamped()
        box_pose.pose.orientation.w = 1.0
        arm_pose = PoseStamped()
        arm_pose.pose.position.x = 0.1
        arm_pose.pose.position.y = 0.2
        arm_pose.pose.position.z = 0.3
        arm_pose.pose.orientation.w = 1.0

        first = MissionController._make_camera_offset_box_orientation_pose(
            harness, arm_pose, box_pose, "left"
        )
        rotated_box_pose = deepcopy(box_pose)
        rotated_box_pose.pose.orientation.z = 2.0 ** -0.5
        rotated_box_pose.pose.orientation.w = 2.0 ** -0.5
        second = MissionController._make_camera_offset_box_orientation_pose(
            harness, arm_pose, rotated_box_pose, "left"
        )

        first_position = [
            first.pose.position.x,
            first.pose.position.y,
            first.pose.position.z,
        ]
        second_position = [
            second.pose.position.x,
            second.pose.position.y,
            second.pose.position.z,
        ]
        self.assertEqual(first_position, [0.1, 0.2, 0.3])
        self.assertEqual(second_position, first_position)
        first_q = [
            first.pose.orientation.x,
            first.pose.orientation.y,
            first.pose.orientation.z,
            first.pose.orientation.w,
        ]
        second_q = [
            second.pose.orientation.x,
            second.pose.orientation.y,
            second.pose.orientation.z,
            second.pose.orientation.w,
        ]
        self.assertLess(
            abs(sum(a * b for a, b in zip(first_q, second_q))),
            0.8,
        )


if __name__ == "__main__":
    unittest.main()
