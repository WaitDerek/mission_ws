import threading
import time
import unittest

from geometry_msgs.msg import PoseStamped

from mission_runtime.mission_controller import MissionController


class _Harness:
    VALUES = {
        "camera_dynamic_link8_extrinsics_enabled": True,
        "camera_detection_arm": "right",
        "camera_eepose_max_age_sec": 1.0,
        "camera_tf_timeout_sec": 0.1,
        "camera_fixed_cross_arm_transform_enabled": True,
        "camera_right_base_to_left_base_xyz": [0.024, 0.0, 0.0],
        "camera_right_base_to_left_base_quaternion_xyzw": [0.0, 0.0, 1.0, 0.0],
        "left_arm_base_frame": "L_base_Link",
        "right_arm_base_frame": "R_base_Link",
        "right_arm_base_frame": "R_base_Link",
        "camera_right_link8_to_rgb_camera_xyz": [
            0.097293583276,
            0.000243459470,
            0.053076686984,
        ],
        "camera_right_link8_to_rgb_camera_quaternion_xyzw": [
            0.153045932190,
            -0.153045932190,
            -0.690345524096,
            0.690345524097,
        ],
    }

    def __init__(self):
        self.joint_state_lock = threading.Lock()
        self.latest_slave_arm_poses = {
            "left": None,
            "right": (
                -0.185504,
                -0.000006,
                -0.691,
                -0.707,
                0.707,
                0.0,
                0.0,
            ),
        }
        self.latest_slave_arm_pose_times = {
            "left": 0.0,
            "right": time.monotonic(),
        }

    def _float_array(self, name):
        return self.VALUES[name]

    def _float(self, name):
        return float(self.VALUES[name])

    def _boolean(self, name):
        return bool(self.VALUES[name])

    def _string(self, name):
        return self.VALUES[name]

    def _slave_arm_pose_snapshot(self, arm):
        pose = self.latest_slave_arm_poses.get(arm)
        state_time = self.latest_slave_arm_pose_times.get(arm, 0.0)
        return pose, time.monotonic() - state_time

    def _arm_base_frame(self, arm):
        return self.VALUES[
            "left_arm_base_frame" if arm == "left" else "right_arm_base_frame"
        ]

    def _dynamic_camera_pose_in_arm_base(
        self, camera_pose, target_arm, detection_arm
    ):
        return MissionController._dynamic_camera_pose_in_arm_base(
            self, camera_pose, target_arm, detection_arm
        )

    @staticmethod
    def _normalize_quaternion(quaternion):
        return MissionController._normalize_quaternion(quaternion)

    @staticmethod
    def _compose_transform(lhs, rhs):
        return MissionController._compose_transform(lhs, rhs)

    @staticmethod
    def _inverse_transform(transform):
        return MissionController._inverse_transform(transform)


class TestWristCameraExtrinsics(unittest.TestCase):
    def test_right_camera_is_composed_from_live_link8_pose(self):
        harness = _Harness()
        camera_box = PoseStamped()
        camera_box.pose.orientation.w = 1.0

        result = MissionController._measured_camera_pose_in_arm_base(
            harness, camera_box, "right", "right"
        )

        self.assertEqual(result.header.frame_id, "R_base_Link")
        self.assertAlmostEqual(result.pose.position.x, -0.185747459470, places=6)
        self.assertAlmostEqual(result.pose.position.y, -0.097299583276, places=6)
        self.assertAlmostEqual(result.pose.position.z, -0.744076686984, places=6)
        # q and -q represent the same rotation; accept either convention.
        self.assertAlmostEqual(abs(result.pose.orientation.x), 0.976296002901, places=6)
        self.assertAlmostEqual(result.pose.orientation.y, 0.0, places=6)
        self.assertAlmostEqual(result.pose.orientation.z, 0.0, places=6)
        self.assertAlmostEqual(abs(result.pose.orientation.w), 0.216439632969, places=6)

    def test_right_camera_pose_uses_fixed_right_to_left_base_transform(self):
        harness = _Harness()
        camera_box = PoseStamped()
        camera_box.pose.orientation.w = 1.0

        result = MissionController._measured_camera_pose_in_arm_base(
            harness, camera_box, "left", "right"
        )

        self.assertEqual(result.header.frame_id, "L_base_Link")
        self.assertAlmostEqual(result.pose.position.x, 0.209747459470, places=6)
        self.assertAlmostEqual(result.pose.position.y, 0.097299583276, places=6)
        self.assertAlmostEqual(result.pose.position.z, -0.744076686984, places=6)


if __name__ == "__main__":
    unittest.main()
