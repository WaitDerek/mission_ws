"""将 ROS2 主机命令安全转发到 Docker 中的 ROS1 Action。"""

from __future__ import annotations

from dataclasses import dataclass
import shlex
import subprocess
import sys
import time
from typing import Callable, TextIO

from .contract import TranslationRequest


@dataclass(frozen=True)
class DockerRos1Config:
    """Docker 内既有 ROS1 timed_translate Action 的连接配置。"""

    container: str = "unitree"
    ros_setup: str = "/opt/ros/noetic/setup.bash"
    ugv_msg_setup: str = "/workspace/ugv_msg_patch/devel/setup.bash"
    mission_setup: str = "/workspace/s2_mission_ws/devel/setup.bash"
    action_name: str = "/timed_translate"
    server_timeout_s: float = 10.0
    completion_margin_s: float = 15.0
    cancel_wait_s: float = 4.0


def _format_number(value: float) -> str:
    return format(float(value), "g")


def _ros1_prefix(config: DockerRos1Config) -> str:
    setup_scripts = (config.ros_setup, config.ugv_msg_setup, config.mission_setup)
    return "set -e; " + "; ".join(
        f"source {shlex.quote(path)}" for path in setup_scripts
    )


def _docker_exec_argv(shell_command: str, config: DockerRos1Config) -> list[str]:
    return ["docker", "exec", config.container, "/bin/bash", "-lc", shell_command]


def build_server_check_argv(config: DockerRos1Config) -> list[str]:
    """构造有时间上限的 Action Server 可用性检查命令。"""

    wait_script = (
        f"until rostopic list | grep -Fxq {shlex.quote(config.action_name + '/goal')}; "
        "do sleep 0.2; done"
    )
    shell_command = (
        f"{_ros1_prefix(config)}; timeout {_format_number(config.server_timeout_s)} "
        f"bash -lc {shlex.quote(wait_script)}"
    )
    return _docker_exec_argv(shell_command, config)


def build_goal_argv(
    request: TranslationRequest, config: DockerRos1Config
) -> list[str]:
    """构造对既有 ROS1 客户端的单次调用，不直接发布底盘 Topic。"""

    shell_command = (
        f"{_ros1_prefix(config)}; exec rosrun s2_ugv_mission "
        "timed_translate_client.py "
        f"{shlex.quote(request.direction)} {_format_number(request.duration_s)} "
        f"--speed-mps {_format_number(request.speed_mps)}"
    )
    return _docker_exec_argv(shell_command, config)


def build_cancel_argv(config: DockerRos1Config) -> list[str]:
    """构造取消当前 Action 的命令；服务端 finally 块会再发布底盘停止命令。"""

    shell_command = (
        f"{_ros1_prefix(config)}; rostopic pub -1 "
        f"{shlex.quote(config.action_name + '/cancel')} actionlib_msgs/GoalID '{{}}'"
    )
    return _docker_exec_argv(shell_command, config)


def _cancel_then_terminate(
    process: subprocess.Popen[bytes],
    config: DockerRos1Config,
    run: Callable[..., subprocess.CompletedProcess[bytes]],
) -> None:
    try:
        run(build_cancel_argv(config), check=False, timeout=config.cancel_wait_s)
    except (OSError, subprocess.TimeoutExpired):
        pass

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=config.cancel_wait_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_translation(
    request: TranslationRequest,
    config: DockerRos1Config,
    *,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    stderr: TextIO | None = None,
) -> int:
    """执行一次平移；Ctrl+C 或超时会先取消 ROS1 Action 再终止本地进程。"""

    if stderr is None:
        stderr = sys.stderr

    try:
        preflight = run(
            build_server_check_argv(config),
            check=False,
            timeout=config.server_timeout_s + 5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"错误：无法检查 ROS1 Action 服务端：{exc}", file=stderr)
        return 70

    if preflight.returncode != 0:
        print(
            "错误：Docker 内未在限定时间内发现 /timed_translate/goal，未发送运动请求。",
            file=stderr,
        )
        return 70

    try:
        process = popen(build_goal_argv(request, config))
    except OSError as exc:
        print(f"错误：无法启动 Docker 内的 ROS1 Action 客户端：{exc}", file=stderr)
        return 71

    deadline = monotonic() + request.duration_s + config.completion_margin_s
    try:
        while process.poll() is None:
            if monotonic() >= deadline:
                _cancel_then_terminate(process, config, run)
                print("错误：ROS1 Action 等待超时，已请求取消动作。", file=stderr)
                return 124
            sleep(0.1)
    except KeyboardInterrupt:
        _cancel_then_terminate(process, config, run)
        print("已取消 ROS1 timed_translate Action。", file=stderr)
        return 130

    return int(process.returncode)
