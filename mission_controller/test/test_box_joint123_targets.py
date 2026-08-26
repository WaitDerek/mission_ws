import math
import unittest

from geometry_msgs.msg import Pose

from mission_runtime.mission_controller import MissionController


class _Harness:
    VALUES = {
        "box_joint1_axis_xyz": [0.0, 0.0, 1.0],
        "box_joint1_feedback_to_geometric_sign": 1.0,
        "box_joint2_axis_xyz": [0.0, 0.0, -1.0],
        "box_joint3_axis_xyz": [0.0, 0.0, 1.0],
        "box_joint2_feedback_to_urdf_axis_sign": 1.0,
        "box_joint3_feedback_to_urdf_axis_sign": 1.0,
        "box_waist1_origin_xyz": [-0.080814986, 0.135049308, 0.266],
        "box_waist1_origin_rpy": [math.pi, math.pi / 2.0, 0.0],
        "box_waist2_origin_xyz": [-0.384, 0.0, -0.0074],
        "box_waist2_origin_rpy": [0.0, 0.0, 0.0],
        "box_waist3_origin_xyz": [-0.277703995, 0.0, 0.0024],
        "box_waist3_origin_rpy": [0.0, 0.0, 0.0],
        "box_waist3_to_chest_xyz": [-0.123796005, 0.0, -0.0755],
        "box_waist3_to_chest_rpy": [0.0, math.pi / 2.0, 0.0],
        "box_chest_to_left_arm_base_xyz": [0.012, 0.0, -0.2975],
        "box_chest_to_left_arm_base_rpy": [0.0, math.pi, 0.0],
        "box_chest_to_right_arm_base_xyz": [-0.012, 0.0, -0.2975],
        "box_chest_to_right_arm_base_rpy": [math.pi, 0.0, 0.0],
    }

    def _float_array(self, name):
        return self.VALUES[name]

    def _float(self, name):
        return float(self.VALUES[name])


def _pose_values(pose):
    return [
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]


def _normalized_pose_values(pose):
    values = _pose_values(pose)
    norm = math.sqrt(sum(value * value for value in values[3:]))
    values[3:] = [value / norm for value in values[3:]]
    return values


class TestBoxJoint123Targets(unittest.TestCase):
    def setUp(self):
        self.harness = _Harness()
        self.target = Pose()
        self.target.position.x = 0.21
        self.target.position.y = -0.37
        self.target.position.z = 0.44
        self.target.orientation.x = -0.49
        self.target.orientation.y = -0.50
        self.target.orientation.z = -0.48
        self.target.orientation.w = 0.53

    def test_urdf_zero_pose_matches_realbots29(self):
        right = MissionController._joint123_arm_base_transform(
            self.harness,
            "right", [0.0, 0.0, 0.0]
        )
        left = MissionController._joint123_arm_base_transform(
            self.harness,
            "left", [0.0, 0.0, 0.0]
        )
        self.assertAlmostEqual(right[0][0], -0.012315, places=5)
        self.assertAlmostEqual(right[0][1], 0.135049, places=5)
        self.assertAlmostEqual(right[0][2], 1.349000, places=5)
        self.assertAlmostEqual(left[0][0], 0.011685, places=5)
        self.assertAlmostEqual(left[0][1], 0.135049, places=5)
        self.assertAlmostEqual(left[0][2], 1.349000, places=5)
        self.assertAlmostEqual(abs(right[1][3]), 1.0, places=6)
        self.assertAlmostEqual(abs(left[1][2]), 1.0, places=6)

    def test_zero_joint_change_keeps_complete_pose(self):
        result = MissionController._reexpress_link8_target_after_joint123_motion(
            self.harness,
            self.target,
            "left",
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        for actual, expected in zip(
            _pose_values(result), _normalized_pose_values(self.target)
        ):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_joint123_change_then_reverse_recovers_pose(self):
        detected = [0.0, 0.0, 0.0]
        moved = [math.radians(18.677), math.radians(4.0), math.radians(7.0)]
        moved_target = MissionController._reexpress_link8_target_after_joint123_motion(
            self.harness, self.target, "right", detected, moved
        )
        recovered = MissionController._reexpress_link8_target_after_joint123_motion(
            self.harness, moved_target, "right", moved, detected
        )
        for actual, expected in zip(
            _pose_values(recovered), _normalized_pose_values(self.target)
        ):
            self.assertAlmostEqual(actual, expected, places=7)


if __name__ == "__main__":
    unittest.main()
