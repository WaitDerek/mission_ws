import math
import unittest

from geometry_msgs.msg import Pose, PoseStamped

from mission_runtime.drag_right_target_calibration import (
    _absolute_right_correction,
    _profile_parameter,
    _quaternion_to_rpy_degrees,
)


class TestDragRightTargetCalibration(unittest.TestCase):
    def test_absolute_correction_is_solved_in_box_frame(self):
        box_pose = PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        ideal = Pose()
        ideal.position.x = 0.1
        ideal.position.y = -0.2
        ideal.position.z = 0.8
        ideal.orientation.w = 1.0

        result = _absolute_right_correction(
            box_pose,
            ideal,
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            False,
        )

        for actual, expected in zip(
            result, (0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0)
        ):
            self.assertAlmostEqual(actual, expected)

    def test_orientation_display_uses_degrees(self):
        half_angle = math.radians(45.0)
        roll, pitch, yaw = _quaternion_to_rpy_degrees(
            (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))
        )
        self.assertAlmostEqual(roll, 0.0)
        self.assertAlmostEqual(pitch, 0.0)
        self.assertAlmostEqual(yaw, 90.0)

    def test_profile_parameters_are_model_and_layer_specific(self):
        self.assertEqual(
            _profile_parameter("offset", "bigbox", 3),
            "drag_box_tf_direct_movel_right_offset_xyz_bigbox_layer3",
        )
        self.assertEqual(
            _profile_parameter("correction", "smallbox", 4),
            "drag_box_tf_joint123_right_target_correction_pose_box_smallbox_layer4",
        )


if __name__ == "__main__":
    unittest.main()
