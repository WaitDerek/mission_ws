"""ROS2 主机底盘命令的输入契约。"""

from __future__ import annotations

from dataclasses import dataclass
import math


SUPPORTED_DIRECTIONS = frozenset({"forward", "backward", "left", "right"})


@dataclass(frozen=True)
class TranslationRequest:
    """一个将被转发给 ROS1 timed_translate Action 的平移请求。"""

    direction: str
    duration_s: float
    speed_mps: float


def validate_request(
    direction: str, duration_s: float, speed_mps: float
) -> TranslationRequest:
    """校验并规范化用户输入，避免无效请求到达 Docker 或底盘。"""

    normalized_direction = str(direction).strip()
    if normalized_direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(
            "direction must be one of: " + ", ".join(sorted(SUPPORTED_DIRECTIONS))
        )

    normalized_duration = float(duration_s)
    if not math.isfinite(normalized_duration) or normalized_duration <= 0.0:
        raise ValueError("duration_s must be finite and greater than 0")

    normalized_speed = float(speed_mps)
    if not math.isfinite(normalized_speed) or normalized_speed < 0.0:
        raise ValueError("speed_mps must be finite and greater than or equal to 0")

    return TranslationRequest(
        direction=normalized_direction,
        duration_s=normalized_duration,
        speed_mps=normalized_speed,
    )
