import unittest

from mission_runtime.grasp_dual_target_calibration import _profile_parameter


class TestGraspDualTargetCalibration(unittest.TestCase):
    def test_profile_parameters_are_arm_model_and_layer_specific(self):
        self.assertEqual(
            _profile_parameter("left", "offset", "smallbox", 2),
            "grasp_box_tf_direct_movel_left_offset_xyz_smallbox_layer2",
        )
        self.assertEqual(
            _profile_parameter("right", "correction", "bigbox", 4),
            "grasp_box_tf_joint123_right_target_correction_pose_box_bigbox_layer4",
        )

    def test_invalid_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            _profile_parameter("centre", "offset", "bigbox", 1)


if __name__ == "__main__":
    unittest.main()
