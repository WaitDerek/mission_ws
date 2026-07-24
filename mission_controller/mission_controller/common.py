import math
from dataclasses import dataclass
from typing import Optional

from geometry_msgs.msg import Pose, PoseStamped


VALID_ARMS = {"left", "right"}
CHASSIS_DIRECTIONS = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "clockwise": (0.0, 0.0, -1.0),
    "counterclockwise": (0.0, 0.0, 1.0),
}
ANGULAR_CHASSIS_DIRECTIONS = {"clockwise", "counterclockwise"}


class MissionError(RuntimeError):
    pass


class MissionCanceled(MissionError):
    pass


class TaskActionError(MissionError):
    def __init__(self, message: str, error_code: int) -> None:
        super().__init__(message)
        self.error_code = error_code


class PickupAttemptError(MissionError):
    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[int],
        motion_started: bool,
        first_segment_completed: bool,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.motion_started = motion_started
        self.first_segment_completed = first_segment_completed


class TwoStageMotionError(MissionError):
    def __init__(self, stage: int, stage_one_completed: bool, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.stage_one_completed = stage_one_completed


@dataclass
class GraspCandidate:
    pose: PoseStamped
    score: float
    width: float
    height: float
    depth: float
    object_id: int


def pose_to_array(pose: Pose) -> list[float]:
    values = [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]
    if not all(math.isfinite(value) for value in values):
        raise MissionError("grasp pose contains NaN or Inf")

    quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
    if quaternion_norm < 1e-8:
        raise MissionError("grasp pose quaternion has zero norm")
    values[3:] = [value / quaternion_norm for value in values[3:]]
    return values


def quaternion_multiply(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def rotate_vector(
    vector: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def compose_poses(parent_pose: Pose, child_pose: Pose) -> Pose:
    """Compose T_reference_parent with T_parent_child."""
    parent_values = pose_to_array(parent_pose)
    child_values = pose_to_array(child_pose)
    parent_orientation = tuple(parent_values[3:])
    child_in_reference = rotate_vector(
        tuple(child_values[:3]), parent_orientation
    )
    orientation = quaternion_multiply(
        parent_orientation, tuple(child_values[3:])
    )
    orientation_norm = math.sqrt(sum(value * value for value in orientation))

    result = Pose()
    result.position.x = parent_values[0] + child_in_reference[0]
    result.position.y = parent_values[1] + child_in_reference[1]
    result.position.z = parent_values[2] + child_in_reference[2]
    result.orientation.x = orientation[0] / orientation_norm
    result.orientation.y = orientation[1] / orientation_norm
    result.orientation.z = orientation[2] / orientation_norm
    result.orientation.w = orientation[3] / orientation_norm
    return result


def interpolate_pose(start_pose: Pose, target_pose: Pose, fraction: float) -> Pose:
    """Interpolate position linearly and orientation along the shortest arc."""
    if not 0.0 <= fraction <= 1.0:
        raise MissionError("pose interpolation fraction must be in [0, 1]")

    start = pose_to_array(start_pose)
    target = pose_to_array(target_pose)
    start_quaternion = start[3:]
    target_quaternion = target[3:]
    dot = sum(a * b for a, b in zip(start_quaternion, target_quaternion))
    if dot < 0.0:
        target_quaternion = [-value for value in target_quaternion]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        orientation = [
            a + fraction * (b - a)
            for a, b in zip(start_quaternion, target_quaternion)
        ]
    else:
        theta = math.acos(dot)
        scale = math.sin(theta)
        start_scale = math.sin((1.0 - fraction) * theta) / scale
        target_scale = math.sin(fraction * theta) / scale
        orientation = [
            start_scale * a + target_scale * b
            for a, b in zip(start_quaternion, target_quaternion)
        ]
    orientation_norm = math.sqrt(sum(value * value for value in orientation))

    result = Pose()
    result.position.x = start[0] + fraction * (target[0] - start[0])
    result.position.y = start[1] + fraction * (target[1] - start[1])
    result.position.z = start[2] + fraction * (target[2] - start[2])
    result.orientation.x = orientation[0] / orientation_norm
    result.orientation.y = orientation[1] / orientation_norm
    result.orientation.z = orientation[2] / orientation_norm
    result.orientation.w = orientation[3] / orientation_norm
    return result
