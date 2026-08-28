"""ROS 2 action server for distance-based S2 base translation."""

from __future__ import annotations

import math
import subprocess
import threading
import time
from typing import Optional

import rclpy
from mission_interfaces.action import MoveBaseDistance
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .contract import SUPPORTED_DIRECTIONS, TranslationRequest
from .docker_ros1 import (
    DockerRos1Config,
    _cancel_then_terminate,
    build_cancel_argv,
    build_goal_argv,
    build_server_check_argv,
)


class MoveBaseDistanceActionServer(Node):
    """Convert a requested physical distance into a ROS1 timed translation.

    The ROS1 action is time based.  The two calibration ratios below convert
    its ideal ``speed * duration`` distance to the measured real distance:
    0.17/0.30 for forward/backward and 0.15/0.30 for lateral motion.
    """

    def __init__(self) -> None:
        super().__init__("move_base_distance_action_server")
        self.declare_parameter("default_speed_mps", 0.05)
        self.declare_parameter("forward_actual_per_commanded_ratio", 17.0 / 30.0)
        self.declare_parameter("lateral_actual_per_commanded_ratio", 15.0 / 30.0)
        self.declare_parameter("max_distance_m", 2.0)
        self.declare_parameter("container", "unitree")
        self.declare_parameter("server_timeout_s", 10.0)
        self.declare_parameter("completion_margin_s", 15.0)
        self.declare_parameter("action_name", "/move_base_distance")

        self._active_lock = threading.Lock()
        self._active = False
        action_name = str(self.get_parameter("action_name").value).strip()
        if not action_name:
            raise ValueError("action_name must not be empty")
        self._server = ActionServer(
            self,
            MoveBaseDistance,
            action_name,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )
        self.get_logger().info(
            f"distance base action ready: {action_name}; "
            f"default_speed_mps={self._float_parameter('default_speed_mps'):.3f}; "
            "calibration=forward 0.17/0.30, lateral 0.15/0.30"
        )

    def _float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _goal_values(self, goal: MoveBaseDistance.Goal) -> tuple[str, float, float]:
        direction = str(goal.direction).strip().lower()
        distance = float(goal.distance_m)
        speed = float(goal.speed_mps)
        if direction not in SUPPORTED_DIRECTIONS:
            raise ValueError(
                "direction must be one of: "
                + ", ".join(sorted(SUPPORTED_DIRECTIONS))
            )
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("distance_m must be finite and greater than 0")
        max_distance = self._float_parameter("max_distance_m")
        if max_distance <= 0.0 or distance > max_distance:
            raise ValueError(
                f"distance_m must be in (0, {max_distance:g}]"
            )
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError("speed_mps must be finite and greater than or equal to 0")
        if speed == 0.0:
            speed = self._float_parameter("default_speed_mps")
        if speed <= 0.0:
            raise ValueError("default_speed_mps must be greater than 0")
        return direction, distance, speed

    def _goal_callback(self, goal: MoveBaseDistance.Goal) -> GoalResponse:
        try:
            self._goal_values(goal)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"rejecting invalid base-distance goal: {exc}")
            return GoalResponse.REJECT
        with self._active_lock:
            if self._active:
                self.get_logger().warning("rejecting base-distance goal: another move is active")
                return GoalResponse.REJECT
            self._active = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel_callback(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _feedback(self, goal_handle, stage: str, progress: float) -> None:
        feedback = MoveBaseDistance.Feedback()
        feedback.stage = stage
        feedback.progress = float(max(0.0, min(1.0, progress)))
        goal_handle.publish_feedback(feedback)

    def _config(self) -> DockerRos1Config:
        return DockerRos1Config(
            container=str(self.get_parameter("container").value).strip(),
            server_timeout_s=self._float_parameter("server_timeout_s"),
            completion_margin_s=self._float_parameter("completion_margin_s"),
        )

    def _ratio(self, direction: str) -> float:
        name = (
            "forward_actual_per_commanded_ratio"
            if direction in {"forward", "backward"}
            else "lateral_actual_per_commanded_ratio"
        )
        ratio = self._float_parameter(name)
        if ratio <= 0.0:
            raise ValueError(f"{name} must be greater than 0")
        return ratio

    def _cancel_process(self, process: subprocess.Popen[bytes], config: DockerRos1Config) -> None:
        _cancel_then_terminate(process, config, subprocess.run)

    def _execute_callback(self, goal_handle):
        result = MoveBaseDistance.Result()
        process: Optional[subprocess.Popen[bytes]] = None
        try:
            direction, distance, speed = self._goal_values(goal_handle.request)
            ratio = self._ratio(direction)
            duration = distance / (speed * ratio)
            estimated_distance = speed * duration * ratio
            request = TranslationRequest(direction, duration, speed)
            config = self._config()

            result.commanded_duration_s = duration
            result.estimated_distance_m = estimated_distance
            self._feedback(goal_handle, "waiting_for_ros1_server", 0.0)
            try:
                preflight = subprocess.run(
                    build_server_check_argv(config),
                    check=False,
                    timeout=config.server_timeout_s + 5.0,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"无法检查 ROS1 timed_translate 服务端：{exc}") from exc
            if preflight.returncode != 0:
                raise RuntimeError(
                    "Docker 内未在限定时间内发现 /timed_translate/goal"
                )

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = "base translation canceled before start"
                return result

            self._feedback(goal_handle, "translating", 0.0)
            try:
                process = subprocess.Popen(
                    build_goal_argv(request, config),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                raise RuntimeError(f"无法启动 Docker 内的 ROS1 timed_translate 客户端：{exc}") from exc

            deadline = time.monotonic() + duration + config.completion_margin_s
            started = time.monotonic()
            while process.poll() is None:
                if goal_handle.is_cancel_requested:
                    self._cancel_process(process, config)
                    goal_handle.canceled()
                    result.success = False
                    result.message = "base translation canceled"
                    return result
                now = time.monotonic()
                if now >= deadline:
                    self._cancel_process(process, config)
                    raise RuntimeError("ROS1 timed_translate 等待超时，已请求取消动作")
                self._feedback(
                    goal_handle,
                    "translating",
                    min(0.99, (now - started) / duration),
                )
                time.sleep(0.1)

            return_code = int(process.returncode)
            if return_code != 0:
                raise RuntimeError(f"ROS1 timed_translate failed with exit_code={return_code}")
            self._feedback(goal_handle, "complete", 1.0)
            goal_handle.succeed()
            result.success = True
            result.message = (
                f"{direction} {distance:.3f} m completed; "
                f"commanded_duration={duration:.3f} s, speed={speed:.3f} m/s, "
                f"calibration_ratio={ratio:.6f}"
            )
            return result
        except Exception as exc:  # noqa: BLE001
            if process is not None and process.poll() is None:
                try:
                    self._cancel_process(process, self._config())
                except Exception as stop_exc:  # noqa: BLE001
                    self.get_logger().error(f"failed to cancel base process: {stop_exc}")
            if not goal_handle.is_cancel_requested:
                goal_handle.abort()
            result.success = False
            result.message = str(exc)
            return result
        finally:
            with self._active_lock:
                self._active = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MoveBaseDistanceActionServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
