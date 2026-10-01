import unittest

from geometry_msgs.msg import PoseStamped

from mission_runtime.mission_controller import MissionController
from mission_runtime.tf_layer_profiles import TfLayerProfilesMixin


class _TargetHarness:
    VALUES = {
        "drag_box_tf_right_backward_offset_m_layer1": 0.0,
        "drag_box_tf_right_backward_offset_m_layer2": 0.0,
        "drag_box_tf_left_forward_offset_m_layer3": 0.0,
        "drag_box_tf_right_forward_offset_m_layer3": 0.0,
        "drag_box_tf_right_forward_offset_m_layer4": 0.0,
        "grasp_box_tf_left_backward_offset_m_layer1": 0.0,
        "grasp_box_tf_right_forward_offset_m_layer1": 0.0,
        "grasp_box_tf_right_forward_offset_m_layer2": 0.0,
        "grasp_box_tf_right_forward_offset_m_layer3": 0.0,
        "grasp_box_tf_right_forward_offset_m_layer4": 0.0,

        "grasp_box_tf_contact_height_offset_m": 0.0,
        "drag_box_tf_contact_height_offset_m": 0.0,
        "grasp_box_tf_left_contact_forward_delta_scale": 1.0,
        "grasp_box_tf_right_contact_forward_delta_scale": 1.0,
        "drag_box_tf_left_contact_forward_delta_scale": 1.0,
        "drag_box_tf_right_contact_forward_delta_scale": 1.0,
        "direct_movel_target_mode": "camera_offset_box_orientation",
        "direct_movel_fixture_compensation_enabled": False,
        "direct_movel_left_box_to_link8_orientation": [0.0, 0.0, 0.0, 1.0],
        "direct_movel_right_box_to_link8_orientation": [0.0, 0.0, 0.0, 1.0],
        "grasp_box_tf_direct_movel_left_offset_xyz_smallbox_layer2": [
            0.0,
            0.0,
            -0.5,
        ],
        "grasp_box_tf_direct_movel_right_offset_xyz_smallbox_layer2": [
            0.0,
            0.0,
            0.5,
        ],
        "grasp_box_tf_joint123_left_target_correction_pose_box_smallbox_layer2": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "grasp_box_tf_joint123_right_target_correction_pose_box_smallbox_layer2": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "drag_box_tf_direct_movel_left_offset_xyz_smallbox_layer2": [
            0.0,
            0.0,
            -0.5,
        ],
        "drag_box_tf_direct_movel_right_offset_xyz_smallbox_layer2": [
            0.0,
            0.0,
            0.5,
        ],
        "drag_box_tf_joint123_left_target_correction_pose_box_smallbox_layer2": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "drag_box_tf_joint123_right_target_correction_pose_box_smallbox_layer2": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    }

    _raise_drag_contact_target = MissionController._raise_drag_contact_target
    _make_tf_link8_target_poses = MissionController._make_tf_link8_target_poses
    _apply_box_frame_target_correction = (
        MissionController._apply_box_frame_target_correction
    )
    _direct_movel_offset_parameter_name = staticmethod(
        MissionController._direct_movel_offset_parameter_name
    )
    _joint123_target_correction_parameter_name = (
        MissionController._joint123_target_correction_parameter_name
    )
    _make_camera_offset_box_orientation_pose = (
        MissionController._make_camera_offset_box_orientation_pose
    )
    _box_oriented_link8_orientation = MissionController._box_oriented_link8_orientation
    _make_direct_movel_pose = MissionController._make_direct_movel_pose

    _scale_grasp_contact_forward_delta = MissionController._scale_grasp_contact_forward_delta

    def _float(self, name):
        return float(self.VALUES[name])

    def _string(self, name):
        return str(self.VALUES[name])

    def _float_array(self, name):
        return list(self.VALUES[name])

    def _boolean(self, name):
        return bool(self.VALUES[name])


class TestTfBoxTargetCorrection(unittest.TestCase):
    def setUp(self):
        self.box_pose = PoseStamped()
        self.box_pose.header.frame_id = "base_link"
        self.box_pose.pose.orientation.w = 1.0

    def test_grasp_uses_per_arm_offsets_without_shared_center_shift(self):
        left, right = _TargetHarness()._make_tf_link8_target_poses(
            self.box_pose,
            2,
            "smallbox",
            drag_mode=False,
        )

        self.assertAlmostEqual(left.pose.position.z, -0.5, places=9)
        self.assertAlmostEqual(right.pose.position.z, 0.5, places=9)
        self.assertAlmostEqual(right.pose.position.z - left.pose.position.z, 1.0)
        self.assertAlmostEqual(
            (left.pose.position.z + right.pose.position.z) / 2.0,
            0.0,
        )

    def test_drag_does_not_apply_grasp_center_correction(self):
        left, right = _TargetHarness()._make_tf_link8_target_poses(
            self.box_pose,
            2,
            "smallbox",
            drag_mode=True,
        )

        self.assertAlmostEqual(left.pose.position.z, -0.5, places=9)
        self.assertAlmostEqual(right.pose.position.z, 0.5, places=9)

    def test_generated_defaults_include_grasp_smallbox_calibrations(self):
        values = dict(TfLayerProfilesMixin._tf_layer_parameter_defaults())
        self.assertNotIn(
            "grasp_box_tf_box_center_correction_xyz_smallbox_layer2",
            values,
        )
        expected = {
            1: (
                [0.0, 0.0, 0.0,
                 -0.024384405, -0.023875088, -0.037291703, 0.998721538],
                [0.0, 0.0, 0.0,
                 -0.051357703, 0.004360193, 0.012754161, 0.998589358],
            ),
            2: (
                [0.0, 0.0, 0.0,
                 -0.016304739, -0.002712095, 0.007963239, 0.999831679],
                [0.0, 0.0, 0.0,
                 -0.039304169, 0.050625186, 0.030148091, 0.997488529],
            ),
            3: (
                [0.0, 0.0, 0.0,
                 -0.009310303, -0.008333252, 0.003572466, 0.999915553],
                [0.0, 0.0, 0.0,
                 -0.038994183, 0.003851522, 0.040366668, 0.998416322],
            ),
            4: (
                [0.0, 0.0, 0.0,
                 -0.005119032, -0.013035214, 0.004660018, 0.999891076],
                [0.0, 0.0, 0.0,
                 -0.044762070, 0.002591940, 0.031502585, 0.998497484],
            ),
        }
        for layer, (left, right) in expected.items():
            self.assertEqual(
                values[
                    "grasp_box_tf_joint123_left_target_correction_pose_box_"
                    f"smallbox_layer{layer}"
                ],
                left,
            )
            self.assertEqual(
                values[
                    "grasp_box_tf_joint123_right_target_correction_pose_box_"
                    f"smallbox_layer{layer}"
                ],
                right,
            )

    def test_generated_defaults_include_drag_bigbox_right_calibrations(self):
        values = dict(TfLayerProfilesMixin._tf_layer_parameter_defaults())
        expected = {
            1: [
                0.0,
                0.0,
                0.0,
                -0.078806036,
                0.031130706,
                0.078378699,
                0.993316298,
            ],
            2: [
                0.0,
                0.0,
                0.0,
                -0.064522892,
                0.019208153,
                0.050748447,
                0.996439882,
            ],
            3: [
                0.0,
                0.0,
                0.0,
                -0.081903855,
                0.048541701,
                0.073411578,
                0.992746797,
            ],
            4: [
                0.0,
                0.0,
                0.0,
                -0.059377345,
                0.012520900,
                0.047450144,
                0.997028606,
            ],
        }
        for layer, correction in expected.items():
            self.assertEqual(
                values[
                    "drag_box_tf_joint123_right_target_correction_pose_box_"
                    f"bigbox_layer{layer}"
                ],
                correction,
            )


if __name__ == "__main__":
    unittest.main()
