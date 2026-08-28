"""提供 ``ros2 run s2_ugv_bridge timed_translate`` 命令。"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Sequence, TextIO

from .contract import validate_request
from .docker_ros1 import DockerRos1Config, run_translation


def _positive_float(value: float, name: str) -> float:
    normalized = float(value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 ROS2 主机调用 Docker 内的 ROS1 timed_translate Action。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("direction", help="forward、backward、left 或 right")
    parser.add_argument("duration_s", type=float, help="运动持续时间，单位秒")
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=0.0,
        help="速度 m/s；省略或设为 0 时交给既有 ROS1 Action 使用默认速度",
    )
    parser.add_argument("--container", default="unitree", help="Docker 容器名称")
    parser.add_argument(
        "--server-timeout-s",
        type=float,
        default=10.0,
        help="等待 ROS1 Action Server 出现的最长时间，单位秒",
    )
    parser.add_argument(
        "--completion-margin-s",
        type=float,
        default=15.0,
        help="运动时长之外允许的模式切换与收尾余量，单位秒",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    run_translation_fn: Callable[..., int] = run_translation,
    stderr: TextIO | None = None,
) -> int:
    """解析命令并返回进程退出码，便于测试和 ``console_scripts`` 使用。"""

    if stderr is None:
        stderr = sys.stderr

    args = build_parser().parse_args(argv)
    try:
        request = validate_request(args.direction, args.duration_s, args.speed_mps)
        container = str(args.container).strip()
        if not container:
            raise ValueError("container must not be empty")
        config = DockerRos1Config(
            container=container,
            server_timeout_s=_positive_float(
                args.server_timeout_s, "server_timeout_s"
            ),
            completion_margin_s=_positive_float(
                args.completion_margin_s, "completion_margin_s"
            ),
        )
    except (TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=stderr)
        return 2

    return int(run_translation_fn(request, config))


if __name__ == "__main__":
    raise SystemExit(main())
