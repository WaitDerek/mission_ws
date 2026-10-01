import unittest

from mission_runtime.drag_left_after_pull_calibration import (
    _parse_reanchored_box_pose,
    _profile_parameter,
)


class TestDragLeftAfterPullCalibration(unittest.TestCase):
    def test_parses_full_reanchored_box_pose_from_feedback(self):
        output = (
            "detail: drag_tf_reanchor_after_drag3=completed; "
            "inferred_box_position=[0.123456789,-0.234567891,0.345678912]; "
            "inferred_box_orientation=[0.010000000,-0.020000000,"
            "0.030000000,0.999300000]; left_join_target=recomputed"
        )
        pose = _parse_reanchored_box_pose(output, "base_link")
        self.assertEqual(pose.header.frame_id, "base_link")
        self.assertAlmostEqual(pose.pose.position.x, 0.123456789)
        self.assertAlmostEqual(pose.pose.position.y, -0.234567891)
        self.assertAlmostEqual(pose.pose.position.z, 0.345678912)
        self.assertAlmostEqual(pose.pose.orientation.w, 0.9993)

    def test_missing_reanchored_orientation_is_rejected(self):
        with self.assertRaises(RuntimeError):
            _parse_reanchored_box_pose(
                "inferred_box_position=[0.1,0.2,0.3]", "base_link"
            )

    def test_profile_is_left_arm_specific(self):
        self.assertEqual(
            _profile_parameter("correction", "bigbox", 3),
            "drag_box_tf_joint123_left_target_correction_pose_box_bigbox_layer3",
        )


if __name__ == "__main__":
    unittest.main()
