import math

from geometry_msgs.msg import Pose

from mission_controller.common import pose_from_transform
from mission_controller.tracking_action import (
    OBJECT_TO_TARGET_MATRIX,
    compute_target_poses,
)


def _pose(x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def test_obj_to_target_is_composed_in_badge_frame():
    ee = _pose(0.2, 0.3, 0.4)
    badge_camera = _pose(0.1, 0.0, 0.5)
    identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    camera, badge, target = compute_target_poses(
        ee, badge_camera, identity, OBJECT_TO_TARGET_MATRIX
    )

    assert math.isclose(camera.position.x, 0.2, abs_tol=1e-9)
    assert math.isclose(camera.position.y, 0.3, abs_tol=1e-9)
    assert math.isclose(camera.position.z, 0.4, abs_tol=1e-9)
    assert math.isclose(badge.position.x, 0.3, abs_tol=1e-9)
    assert math.isclose(badge.position.y, 0.3, abs_tol=1e-9)
    assert math.isclose(badge.position.z, 0.9, abs_tol=1e-9)
    assert math.isclose(target.position.x, 0.25, abs_tol=1e-9)
    assert math.isclose(target.position.y, 0.3, abs_tol=1e-9)
    assert math.isclose(target.position.z, 0.9, abs_tol=1e-9)
    assert math.isclose(abs(target.orientation.x), 0.5, abs_tol=1e-12)
    assert math.isclose(abs(target.orientation.y), 0.5, abs_tol=1e-12)
    assert math.isclose(abs(target.orientation.z), 0.5, abs_tol=1e-12)
    assert math.isclose(abs(target.orientation.w), 0.5, abs_tol=1e-12)


def test_obj_to_target_contains_local_negative_five_centimeter_offset():
    transform = pose_from_transform(OBJECT_TO_TARGET_MATRIX)
    assert math.isclose(transform.position.x, -0.05, abs_tol=1e-12)
    assert math.isclose(transform.position.y, 0.0, abs_tol=1e-12)
    assert math.isclose(transform.position.z, 0.0, abs_tol=1e-12)

    # The matrix maps badge-local X/Y/Z to target X/Y/Z as
    # target X <- badge Y, target Y <- badge Z, target Z <- badge X.
    for component in (
        transform.orientation.x,
        transform.orientation.y,
        transform.orientation.z,
        transform.orientation.w,
    ):
        # q and -q represent the same rotation.
        assert math.isclose(abs(component), 0.5, abs_tol=1e-12)
