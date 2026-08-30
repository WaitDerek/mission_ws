import json
import math
from pathlib import Path

import numpy as np
from geometry_msgs.msg import Pose

from mission_controller.manipulation_math import (
    assembly_targets,
    grip_base_targets,
    local_y_approach,
    matrix_to_pose_array,
    peel_base_above_target,
    peel_withdraw_targets,
    pose_to_matrix,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]
MISSION_CONFIG = SOURCE_ROOT / "mission_controller" / "config"
VENDOR_CONFIG = (
    SOURCE_ROOT / "g1d-task-master" / "execute_grasp_script_runner" / "config"
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_pose() -> Pose:
    pose = Pose()
    pose.orientation.w = 1.0
    return pose


def test_pipeline_configs_are_exact_vendor_copies():
    assert _load(MISSION_CONFIG / "grip.json") == _load(
        VENDOR_CONFIG / "grip_config.json"
    )
    assert _load(MISSION_CONFIG / "peel.json") == _load(
        VENDOR_CONFIG / "peel_config.json"
    )


def test_grip_targets_preserve_original_transform_order():
    config = _load(MISSION_CONFIG / "grip.json")
    identity = np.eye(4)
    down, up = grip_base_targets(identity, identity, config)

    left_ee_to_camera = pose_to_matrix(_pose_from_config(config["left_ee_T_cam"]))
    rotation = _original_object_target_rotation(config["correct_matrix"])
    down_offset = np.eye(4)
    down_offset[1, 3] = -0.27
    up_offset = np.eye(4)
    up_offset[1, 3] = -0.29
    assert np.allclose(down, left_ee_to_camera @ rotation @ down_offset)
    assert np.allclose(up, left_ee_to_camera @ rotation @ up_offset)
    pose_array = matrix_to_pose_array(down)
    assert len(pose_array) == 7
    assert math.isclose(sum(value * value for value in pose_array[3:]), 1.0)


def test_peel_targets_preserve_original_local_motion():
    config = _load(MISSION_CONFIG / "peel.json")
    identity = np.eye(4)
    above = peel_base_above_target(identity, identity, config)
    right_ee_to_camera = pose_to_matrix(_pose_from_config(config["right_ee_T_cam"]))
    rotation = _original_object_target_rotation(config["correct_matrix"])
    above_offset = np.eye(4)
    above_offset[1, 3] = -0.155
    assert np.allclose(above, right_ee_to_camera @ rotation @ above_offset)

    left, right = peel_withdraw_targets(identity, identity, config)
    assert np.allclose(left[:3, 3], [0.02, -0.03, 0.0])
    assert np.allclose(right[:3, 3], [0.02, -0.03, 0.0])
    theta = math.radians(15.0)
    assert math.isclose(left[0, 1], math.sin(theta), abs_tol=1e-12)
    assert math.isclose(right[0, 1], -math.sin(theta), abs_tol=1e-12)


def test_approach_translation_is_in_end_effector_local_y():
    base = np.eye(4)
    base[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    target = local_y_approach(base, 0.1)
    assert np.allclose(target[:3, 3], [-0.1, 0.0, 0.0])


def test_assembly_target_uses_camera_task_and_local_preoffset():
    config = _load(MISSION_CONFIG / "assembly.json")
    target_config = config["targets"]["connector"]
    identity = np.eye(4)
    pre_target, final_target = assembly_targets(
        identity,
        identity,
        config,
        target_config,
    )
    expected_final = pose_to_matrix(_pose_from_config(config["left_ee_T_cam"]))
    assert np.allclose(final_target, expected_final)
    assert np.allclose(
        np.linalg.inv(final_target) @ pre_target,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def _pose_from_config(value) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = value["position"]
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = value["orientation"]
    return pose


def _original_object_target_rotation(correct_matrix):
    first = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    second = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return first @ second @ np.array(correct_matrix)
