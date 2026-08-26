import math
from pathlib import Path

from mission_runtime.global_tf_publisher import (
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
        transform = model.transform_from_root(
            "base_footprint", endpoint, zero_pose
        )

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
    left_chest_to_base = RigidTransform(
        (0.012, 0.0, -0.2975), (0.0, 1.0, 0.0, 0.0)
    )
    right_chest_to_base = RigidTransform(
        (-0.012, 0.0, -0.2975), (1.0, 0.0, 0.0, 0.0)
    )
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
    assert max(
        abs(a - b) for a, b in zip(zero_left.translation, moved_left.translation)
    ) > 1.0e-3
    assert max(
        abs(a - b) for a, b in zip(zero_right.translation, moved_right.translation)
    ) > 1.0e-3
