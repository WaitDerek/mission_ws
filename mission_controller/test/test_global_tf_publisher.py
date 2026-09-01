import math
from pathlib import Path

import yaml

from mission_runtime.global_tf_kinematics import (
    RigidTransform,
    UrdfKinematics,
    compatibility_arm_base_transforms,
    inverse,
)


def test_realbots_urdf_contains_the_requested_camera_chain():
    source_root = Path(__file__).resolve().parents[2]
    urdf = source_root / "realbots2" / "urdf" / "realbots29.urdf"
    model = UrdfKinematics(str(urdf))

    zero_pose = {}
    for endpoint in (
        "left_arm_8_Link",
        "right_arm_8_Link",
        "left_camera_Link",
        "right_camera_Link",
        "left_binocular_left_camera_Link",
        "right_binocular_left_camera_Link",
    ):
        transform = model.transform_from_root("base_footprint", endpoint, zero_pose)

        assert all(math.isfinite(value) for value in transform.translation)
        assert math.sqrt(sum(value * value for value in transform.translation)) > 0.1
        recovered = inverse(inverse(transform))
        assert math.isclose(
            recovered.translation[0], transform.translation[0], abs_tol=1.0e-9
        )
        assert math.isclose(
            recovered.translation[1], transform.translation[1], abs_tol=1.0e-9
        )
        assert math.isclose(
            recovered.translation[2], transform.translation[2], abs_tol=1.0e-9
        )


def test_left_camera_children_apply_local_optical_roll_correction():
    source_root = Path(__file__).resolve().parents[2]
    urdf = source_root / "realbots2" / "urdf" / "realbots29.urdf"
    model = UrdfKinematics(str(urdf))

    correction = math.radians(-5.0)
    expected = (math.sin(correction / 2.0), 0.0, 0.0, math.cos(correction / 2.0))
    for joint_name in (
        "left_binocular_right_camera_joint",
        "left_speckle_camera_joint",
        "left_rgb_camera_Link",
        "left_binocular_left_camera_joint",
    ):
        actual = model.joints[joint_name].origin.rotation
        assert math.isclose(
            abs(sum(a * b for a, b in zip(actual, expected))),
            1.0,
            abs_tol=1.0e-9,
        )


def test_left_depth_camera_mount_moves_back_1cm_z_and_09cm_y():
    source_root = Path(__file__).resolve().parents[2]
    urdf = source_root / "realbots2" / "urdf" / "realbots29.urdf"
    model = UrdfKinematics(str(urdf))
    rgb_origin = model.joints["left_rgb_camera_Link"].origin.translation

    config = yaml.safe_load(
        (source_root / "mission_controller" / "config" / "global_tf.yaml").read_text(
            encoding="utf-8"
        )
    )
    parameters = config["realbots_global_tf"]["ros__parameters"]
    configured = tuple(
        float(value)
        for value in parameters["camera_mount_xyz"]
    )
    quaternion = tuple(
        float(value) for value in parameters["camera_mount_quaternion_xyzw"]
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    x, y, z, w = (value / norm for value in quaternion)
    y_axis = (
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
    )
    z_axis = (
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    )
    expected = tuple(
        rgb_origin[index] - 0.01 * z_axis[index] - 0.009 * y_axis[index]
        for index in range(3)
    )

    for actual, target in zip(configured, expected):
        assert math.isclose(actual, target, abs_tol=1.0e-12)


def test_left_depth_camera_axes_apply_requested_local_remap():
    source_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (source_root / "mission_controller" / "config" / "global_tf.yaml").read_text(
            encoding="utf-8"
        )
    )
    configured = tuple(
        float(value)
        for value in config["realbots_global_tf"]["ros__parameters"][
            "camera_mount_quaternion_xyzw"
        ]
    )
    old_mount = (0.500617260, 0.499457371, 0.499528938, -0.500395377)
    correction = math.radians(-5.0)
    roll_correction = (
        math.sin(correction / 2.0),
        0.0,
        0.0,
        math.cos(correction / 2.0),
    )
    axis_remap = (0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0))

    def multiply(lhs, rhs):
        lx, ly, lz, lw = lhs
        rx, ry, rz, rw = rhs
        return (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )

    expected = multiply(multiply(roll_correction, old_mount), axis_remap)
    configured_norm = math.sqrt(sum(value * value for value in configured))
    expected_norm = math.sqrt(sum(value * value for value in expected))
    configured = tuple(value / configured_norm for value in configured)
    expected = tuple(value / expected_norm for value in expected)

    assert math.isclose(
        abs(sum(a * b for a, b in zip(configured, expected))),
        1.0,
        abs_tol=1.0e-9,
    )


def test_urdf_joint_names_match_hardware_feedback_mapping():
    source_root = Path(__file__).resolve().parents[2]
    urdf = source_root / "realbots2" / "urdf" / "realbots29.urdf"
    model = UrdfKinematics(str(urdf))

    expected = {
        "waist_1_joint",
        "waist_2_joint",
        "waist_3_joint",
        "chest_joint",
        "left_arm_1_joint",
        "left_arm_7_joint",
    }
    assert expected.issubset(model.joints)


def test_compatibility_arm_base_transforms_follow_live_waist_state():
    source_root = Path(__file__).resolve().parents[2]
    urdf = source_root / "realbots2" / "urdf" / "realbots29.urdf"
    model = UrdfKinematics(str(urdf))
    left_chest_to_base = RigidTransform((0.012, 0.0, -0.2975), (0.0, 1.0, 0.0, 0.0))
    right_chest_to_base = RigidTransform((-0.012, 0.0, -0.2975), (1.0, 0.0, 0.0, 0.0))
    zero = {
        "waist_1_joint": 0.0,
        "waist_2_joint": 0.0,
        "waist_3_joint": 0.0,
        "chest_joint": 0.0,
    }
    moved = dict(zero)
    moved["waist_2_joint"] = 0.4
    zero_left, zero_right = compatibility_arm_base_transforms(
        model,
        "base_Link",
        "chest_Link",
        zero,
        left_chest_to_base,
        right_chest_to_base,
    )
    moved_left, moved_right = compatibility_arm_base_transforms(
        model,
        "base_Link",
        "chest_Link",
        moved,
        left_chest_to_base,
        right_chest_to_base,
    )
    assert (
        max(abs(a - b) for a, b in zip(zero_left.translation, moved_left.translation))
        > 1.0e-3
    )
    assert (
        max(abs(a - b) for a, b in zip(zero_right.translation, moved_right.translation))
        > 1.0e-3
    )
