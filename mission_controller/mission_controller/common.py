import math
from typing import Optional

from geometry_msgs.msg import Pose


VALID_ARMS = {"left", "right"}
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
