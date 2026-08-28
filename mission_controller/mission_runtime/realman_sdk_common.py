"""Direct RealMan Python SDK motion backend."""

from __future__ import annotations

import math
from typing import Sequence


class RealManSdkError(RuntimeError):
    """A connection, SDK, or motion-command failure."""


class RealManSdkCanceled(RealManSdkError):
    """The mission was canceled while an SDK motion was active."""


def quaternion_to_rpy(quaternion: Sequence[float]) -> list[float]:
    """Convert an x/y/z/w quaternion to RealMan roll/pitch/yaw radians."""
    if len(quaternion) != 4:
        raise ValueError("quaternion must contain [x, y, z, w]")
    qx, qy, qz, qw = (float(value) for value in quaternion)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion norm must be finite and non-zero")
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def pose_to_sdk_target(pose) -> list[float]:
    """Convert geometry_msgs/Pose to [x, y, z, rx, ry, rz]."""
    position = [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    ]
    if not all(math.isfinite(value) for value in position):
        raise ValueError("pose position contains NaN or Inf")
    orientation = [
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]
    if not all(math.isfinite(value) for value in orientation):
        raise ValueError("pose orientation contains NaN or Inf")
    return position + quaternion_to_rpy(orientation)
