import unittest

from geometry_msgs.msg import PoseStamped

from mission_runtime.mission_controller import (
    MissionController,
    MissionError,
    rotate_vector,
)


class _Logger:
    def info(self, _message):
        pass


class _ConstraintHarness:
    def __init__(self, min_dot=0.5, model_label="f320"):
        self.min_dot = min_dot
        self.model_label = model_label
        self.logger = _Logger()

    def _float(self, name):
        assert name == "box_camera_pose_axis_min_dot"
        return self.min_dot

    def _string(self, name):
        assert name == "box_object_pose_model_label"
        return self.model_label

    def _quaternion_from_rpy(self, roll, pitch, yaw):
        return MissionController._quaternion_from_rpy(roll, pitch, yaw)

    def get_logger(self):
        return self.logger


def _make_pose(quaternion):
    pose = PoseStamped()
    pose.header.frame_id = "camera_optical_frame"
    pose.pose.position.x = 0.1
    pose.pose.position.y = -0.2
    pose.pose.position.z = 0.7
    (
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z,
        pose.pose.orientation.w,
    ) = quaternion
    return pose


def _axes(pose):
    quaternion = (
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z,
        pose.pose.orientation.w,
    )
    return (
        rotate_vector((1.0, 0.0, 0.0), quaternion),
        rotate_vector((0.0, 1.0, 0.0), quaternion),
        rotate_vector((0.0, 0.0, 1.0), quaternion),
    )


class TestBoxCameraPoseConstraint(unittest.TestCase):
    def test_flips_local_y_and_z_when_x_down_y_backward_z_left(self):
        # Columns of this quaternion's rotation are:
        # X=(0,+1,0), Y=(0,0,-1), Z=(-1,0,0) in optical coordinates.
        pose = _make_pose((0.5, 0.5, -0.5, -0.5))
        constrained = MissionController._constrain_box_camera_pose(
            _ConstraintHarness(), pose
        )

        object_x, object_y, object_z = _axes(constrained)
        self.assertAlmostEqual(object_x[1], 1.0, places=6)
        self.assertAlmostEqual(object_y[2], 1.0, places=6)
        self.assertAlmostEqual(object_z[0], 1.0, places=6)
        self.assertEqual(constrained.header.frame_id, pose.header.frame_id)
        self.assertAlmostEqual(constrained.pose.position.x, 0.1)
        self.assertAlmostEqual(constrained.pose.position.y, -0.2)
        self.assertAlmostEqual(constrained.pose.position.z, 0.7)

    def test_keeps_nonmatching_orientation(self):
        pose = _make_pose((0.0, 0.0, 0.0, 1.0))
        constrained = MissionController._constrain_box_camera_pose(
            _ConstraintHarness(), pose
        )
        self.assertIs(constrained, pose)

    def test_f455_z_forward_is_rotated_to_f320_canonical_axes(self):
        # F455 observed axes: X=down, Y=left, Z=forward.
        root_half = 2.0**-0.5
        pose = _make_pose((0.0, 0.0, root_half, root_half))
        constrained = MissionController._constrain_box_camera_pose(
            _ConstraintHarness(model_label="f455"), pose
        )

        object_x, object_y, object_z = _axes(constrained)
        self.assertAlmostEqual(object_x[1], 1.0, places=6)
        self.assertAlmostEqual(object_y[2], 1.0, places=6)
        self.assertAlmostEqual(object_z[0], 1.0, places=6)

    def test_f455_z_backward_is_rotated_to_f320_canonical_axes(self):
        # The symmetric F455 solution: X=down, Y=right, Z=backward.
        root_half = 2.0**-0.5
        pose = _make_pose((root_half, root_half, 0.0, 0.0))
        constrained = MissionController._constrain_box_camera_pose(
            _ConstraintHarness(model_label="f455"), pose
        )

        object_x, object_y, object_z = _axes(constrained)
        self.assertAlmostEqual(object_x[1], 1.0, places=6)
        self.assertAlmostEqual(object_y[2], 1.0, places=6)
        self.assertAlmostEqual(object_z[0], 1.0, places=6)

    def test_rejects_grossly_inverted_x_up_pose(self):
        # A -90-degree camera-Z rotation maps object X to optical -Y (up).
        root_half = 2.0**-0.5
        pose = _make_pose((0.0, 0.0, -root_half, root_half))
        with self.assertRaisesRegex(MissionError, "object X points up"):
            MissionController._constrain_box_camera_pose(
                _ConstraintHarness(model_label="f455"), pose
            )


if __name__ == "__main__":
    unittest.main()
