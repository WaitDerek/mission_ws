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
        "direct_movel_target_mode": "camera_offset_box_orientation",
        "direct_movel_left_box_to_link8_orientation": [0.0, 0.0, 0.0, 1.0],
        "direct_movel_right_box_to_link8_orientation": [0.0, 0.0, 0.0, 1.0],
        "box_post_movel_left_step3_xyz": [-0.14, 0.0, 0.0],
        "box_post_movel_right_step3_xyz": [-0.14, 0.0, 0.0],
        "box_post_movel_left_step4_xyz": [0.0, 0.0, -0.1],
        "box_post_movel_right_step4_xyz": [0.0, 0.0, 0.1],
    }

    def _float_array(self, name):
        return self.VALUES[name]

    def _float(self, name):
        return float(self.VALUES[name])

    def _string(self, name):
        return str(self.VALUES[name])


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

    def test_arm_base_transform_is_chest_transform_plus_fixed_mount(self):
        angles = [math.radians(-45.0), math.radians(-85.0), math.radians(-55.0)]
        chest = MissionController._joint123_chest_transform(
            self.harness, angles
        )
        mount = MissionController._configured_rpy_transform(
            self.harness,
            "box_chest_to_right_arm_base_xyz",
            "box_chest_to_right_arm_base_rpy",
        )
        expected = MissionController._compose_transform(chest, mount)
        actual = MissionController._joint123_arm_base_transform(
            self.harness, "right", angles
        )
        for actual_value, expected_value in zip(actual[0], expected[0]):
            self.assertAlmostEqual(actual_value, expected_value, places=9)
        self.assertAlmostEqual(
            abs(sum(actual[1][i] * expected[1][i] for i in range(4))),
            1.0,
            places=9,
        )

    def test_tf_carry_keeps_box_orientation_and_box_to_link7_relation(self):
        current_box = ((0.2, -0.4, 0.5), (0.0, 0.0, 0.0, 1.0))
        box_to_link = ((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))
        future_carrier = (
            (0.1, 0.2, 0.3),
            MissionController._quaternion_from_rpy(0.3, -0.2, 0.1),
        )
        carrier_point = (0.4, -0.1, 0.2)
        future_position = MissionController._compose_transform(
            future_carrier,
            (carrier_point, (0.0, 0.0, 0.0, 1.0)),
        )[0]
        future_box = (future_position, current_box[1])
        future_link = MissionController._compose_transform(
            future_box, box_to_link
        )
        recovered_relation = MissionController._compose_transform(
            MissionController._inverse_transform(future_box), future_link
        )
        for actual, expected in zip(recovered_relation[0], box_to_link[0]):
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertEqual(future_box[1], current_box[1])

    def test_place_box_slerp_preserves_normalization_and_endpoints(self):
        start = MissionController._quaternion_from_rpy(0.0, 0.0, 0.0)
        target = MissionController._quaternion_from_rpy(0.15, -0.08, 0.2)
        first = MissionController._slerp_quaternion(start, target, 0.0)
        middle = MissionController._slerp_quaternion(start, target, 0.5)
        last = MissionController._slerp_quaternion(start, target, 1.0)
        self.assertAlmostEqual(
            abs(sum(first[index] * start[index] for index in range(4))),
            1.0,
            places=9,
        )
        self.assertAlmostEqual(
            abs(sum(last[index] * target[index] for index in range(4))),
            1.0,
            places=9,
        )
        self.assertAlmostEqual(sum(value * value for value in middle), 1.0, places=9)

    def test_place_box_slerp_treats_negated_quaternion_as_same_pose(self):
        start = MissionController._quaternion_from_rpy(0.2, -0.1, 0.3)
        target = tuple(-value for value in start)
        result = MissionController._slerp_quaternion(start, target, 0.5)
        self.assertAlmostEqual(
            abs(sum(result[index] * start[index] for index in range(4))),
            1.0,
            places=9,
        )

    def test_post_steps_are_rebased_after_tf_carry(self):
        left = Pose()
        left.orientation.w = 1.0
        right = Pose()
        right.orientation.w = 1.0
        self.harness._last_tf_body_home_carry_arm_targets = {
            "left": left,
            "right": right,
        }
        targets = [
            ("step1", Pose(), Pose()),
            ("step2", Pose(), Pose()),
            ("step3", Pose(), Pose()),
            ("step4", Pose(), Pose()),
        ]
        MissionController._rebase_post_movel_targets_after_tf_carry(
            self.harness, targets, 2
        )
        self.assertAlmostEqual(targets[2][1].position.x, -0.14)
        self.assertAlmostEqual(targets[2][2].position.x, -0.14)
        self.assertAlmostEqual(targets[3][1].position.z, -0.1)
        self.assertAlmostEqual(targets[3][2].position.z, 0.1)

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
