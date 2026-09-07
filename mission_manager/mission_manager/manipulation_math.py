"""Transform helpers shared by grip, peel, and install tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
from geometry_msgs.msg import Pose

from .common import pose_from_transform, pose_to_array


def pose_to_matrix(pose: Pose) -> np.ndarray:
    quaternion = np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=float,
    )
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        raise ValueError("pose quaternion has zero norm")
    x, y, z, w = quaternion / norm
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return transform


def pose_dict_to_matrix(value: dict[str, Any]) -> np.ndarray:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = value["position"]
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = value["orientation"]
    return pose_to_matrix(pose)


def matrix_to_pose_array(transform: np.ndarray) -> list[float]:
    return pose_to_array(pose_from_transform(transform.reshape(-1).tolist()))


def _object_target_rotation(correct_matrix: list[list[float]]) -> np.ndarray:
    first_rotation = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    second_rotation = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return first_rotation @ second_rotation @ np.array(correct_matrix)


def grip_object_targets(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rotation = _object_target_rotation(config["correct_matrix"])
    down_offset = np.eye(4)
    up_offset = np.eye(4)
    down_offset[1, 3] = -config["bottom_to_left_ee"] - config["down_offset"]
    up_offset[1, 3] = -config["bottom_to_left_ee"] - config["up_offset"]
    return rotation @ down_offset, rotation @ up_offset


def grip_base_targets(
    base_to_left_ee: np.ndarray,
    camera_to_object: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    left_ee_to_camera = pose_dict_to_matrix(config["left_ee_T_cam"])
    base_to_object = base_to_left_ee @ left_ee_to_camera @ camera_to_object
    object_to_down, object_to_up = grip_object_targets(config)
    return base_to_object @ object_to_down, base_to_object @ object_to_up


def peel_object_above_target(config: dict[str, Any]) -> np.ndarray:
    rotation = _object_target_rotation(config["correct_matrix"])
    above_offset = np.eye(4)
    above_offset[1, 3] = -config["bottom_to_right_ee"] - config["above_dist"]
    return rotation @ above_offset


def peel_base_above_target(
    base_to_right_ee: np.ndarray,
    camera_to_object: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    right_ee_to_camera = pose_dict_to_matrix(config["right_ee_T_cam"])
    return (
        base_to_right_ee
        @ right_ee_to_camera
        @ camera_to_object
        @ peel_object_above_target(config)
    )


def local_y_approach(base_to_ee: np.ndarray, distance: float) -> np.ndarray:
    approach = np.eye(4)
    approach[1, 3] = distance
    return base_to_ee @ approach


def peel_withdraw_targets(
    base_to_left_ee: np.ndarray,
    base_to_right_ee: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    left_local_target = np.eye(4)
    right_local_target = np.eye(4)
    left_local_target[:3, 3] = config["withdraw_dist_l"]
    right_local_target[:3, 3] = config["withdraw_dist_r"]

    theta = float(config["withdraw_theta"]) / 180.0 * np.pi
    left_local_target[:2, :2] = [
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ]
    right_local_target[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
    return (
        base_to_left_ee @ left_local_target,
        base_to_right_ee @ right_local_target,
    )


def assembly_targets(
    base_to_left_ee: np.ndarray,
    camera_to_task: np.ndarray,
    config: dict[str, Any],
    target_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return pre-assembly and final tool targets in the arm base frame."""
    left_ee_to_camera = pose_dict_to_matrix(config["left_ee_T_cam"])
    task_to_tool = np.asarray(target_config["task_T_tool"], dtype=float)
    if task_to_tool.shape != (4, 4):
        raise ValueError("task_T_tool must be a 4x4 matrix")
    base_to_final_tool = (
        base_to_left_ee @ left_ee_to_camera @ camera_to_task @ task_to_tool
    )
    pre_offset = np.eye(4)
    offset = np.asarray(target_config["preinstall_offset_xyz"], dtype=float)
    if offset.shape != (3,):
        raise ValueError("preinstall_offset_xyz must contain three values")
    pre_offset[:3, 3] = offset
    return base_to_final_tool @ pre_offset, base_to_final_tool
