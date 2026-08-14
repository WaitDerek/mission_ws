import math
import unittest
from copy import deepcopy

from geometry_msgs.msg import Pose, PoseStamped

from mission_runtime.adaptive_box_support import (
    AdaptiveBoxSupportMixin,
    compose_pose,
)


class _GraspHarness(AdaptiveBoxSupportMixin):
    def __init__(self):
        self.parameters = {
            "adaptive_grasp_span_axis_object": [0.0, 0.0, 1.0],
            "adaptive_grasp_height_axis_object": [-1.0, 0.0, 0.0],
            "adaptive_grasp_correction_rpy": [0.0, 0.0, 0.0],
            "adaptive_left_grasp_extra_rpy": [0.0, 0.0, 0.0],
            "adaptive_right_grasp_extra_rpy": [0.0, 0.0, 0.0],
            "box_width": 0.4,
            "adaptive_grasp_side_clearance_m": 0.01,
            "adaptive_grasp_height_offset_m": 0.05,
            "direct_movel_fixture_compensation_enabled": True,
            "left_fixture_center_in_link8_xyz": [-0.07, -0.11, 0.03],
            "right_fixture_center_in_link8_xyz": [-0.07, 0.11, 0.03],
            "left_arm_base_frame": "L_base_Link",
            "right_arm_base_frame": "R_base_Link",
        }

    def _float_array(self, name):
        return list(self.parameters[name])

    def _float(self, name):
        return float(self.parameters[name])

    def _boolean(self, name):
        return bool(self.parameters[name])

    def _string(self, name):
        return str(self.parameters[name])

    @staticmethod
    def _transform_pose_latest(pose, target_frame):
        result = deepcopy(pose)
        result.header.frame_id = target_frame
        return result

    @staticmethod
    def _quaternion_from_rpy(roll, pitch, yaw):
        half_roll = 0.5 * roll
        half_pitch = 0.5 * pitch
        half_yaw = 0.5 * yaw
        cr, sr = math.cos(half_roll), math.sin(half_roll)
        cp, sp = math.cos(half_pitch), math.sin(half_pitch)
        cy, sy = math.cos(half_yaw), math.sin(half_yaw)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )


class TestAdaptiveBoxGeometry(unittest.TestCase):
    def test_compose_pose_rotates_object_local_translation(self):
        parent = Pose()
        parent.position.x = 1.0
        parent.position.y = 2.0
        parent.position.z = 3.0
        root_half = 2.0**-0.5
        parent.orientation.z = root_half
        parent.orientation.w = root_half

        result = compose_pose(
            parent,
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )

        self.assertAlmostEqual(result.position.x, 1.0, places=6)
        self.assertAlmostEqual(result.position.y, 3.0, places=6)
        self.assertAlmostEqual(result.position.z, 3.0, places=6)
        self.assertAlmostEqual(result.orientation.z, root_half, places=6)
        self.assertAlmostEqual(result.orientation.w, root_half, places=6)

    def test_grasp_targets_are_symmetric_and_keep_base_header(self):
        object_pose = PoseStamped()
        object_pose.header.frame_id = "base_link"
        object_pose.header.stamp.sec = 12
        object_pose.header.stamp.nanosec = 34
        object_pose.pose.position.x = 1.0
        object_pose.pose.position.y = 2.0
        object_pose.pose.position.z = 3.0
        object_pose.pose.orientation.w = 1.0

        left, right = _GraspHarness()._compute_adaptive_grasp_poses(
            object_pose
        )

        self.assertEqual(left.header.frame_id, "base_link")
        self.assertEqual(right.header.frame_id, "base_link")
        self.assertEqual(left.header.stamp.sec, 12)
        self.assertAlmostEqual(left.pose.position.x, 0.95)
        self.assertAlmostEqual(right.pose.position.x, 0.95)
        self.assertAlmostEqual(left.pose.position.z, 2.79)
        self.assertAlmostEqual(right.pose.position.z, 3.21)

    def test_sdk_targets_apply_lift_and_fixture_center_compensation(self):
        left = PoseStamped()
        left.header.frame_id = "base_link"
        left.pose.position.x = 1.0
        left.pose.position.y = 2.0
        left.pose.position.z = 3.0
        left.pose.orientation.w = 1.0
        right = deepcopy(left)

        left_target, right_target = _GraspHarness()._adaptive_sdk_targets(
            left, right, 0.1
        )

        self.assertAlmostEqual(left_target.position.x, 1.07)
        self.assertAlmostEqual(left_target.position.y, 2.11)
        self.assertAlmostEqual(left_target.position.z, 3.07)
        self.assertAlmostEqual(right_target.position.x, 1.07)
        self.assertAlmostEqual(right_target.position.y, 1.89)
        self.assertAlmostEqual(right_target.position.z, 3.07)


if __name__ == "__main__":
    unittest.main()
