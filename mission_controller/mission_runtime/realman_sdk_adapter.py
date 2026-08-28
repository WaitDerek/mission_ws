"""Public RealMan SDK adapter facade."""

from .realman_sdk_common import (
    RealManSdkCanceled,
    RealManSdkError,
    pose_to_sdk_target,
    quaternion_to_rpy,
)
from .realman_sdk_connected_motion import RealManSdkConnectedMotionMixin
from .realman_sdk_connection import RealManSdkConnectionMixin
from .realman_sdk_motion import RealManSdkMotionMixin


class RealManSdkAdapter(
    RealManSdkConnectionMixin,
    RealManSdkMotionMixin,
    RealManSdkConnectedMotionMixin,
):
    """Own two SDK connections and expose the legacy public API."""

    pass


__all__ = [
    "RealManSdkAdapter",
    "RealManSdkCanceled",
    "RealManSdkError",
    "pose_to_sdk_target",
    "quaternion_to_rpy",
]
