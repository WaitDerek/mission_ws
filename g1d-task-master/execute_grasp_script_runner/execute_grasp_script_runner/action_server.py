#!/usr/bin/env python3
"""Expose a Python script as a mission_interfaces/ExecuteGrasp action."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
from ament_index_python.packages import get_package_share_directory
from mission_interfaces.action import ExecuteGrasp
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


PACKAGE_NAME = "execute_grasp_script_runner"


class ExecuteGraspScriptServer(Node):
    """Run exactly one configured Python script for each accepted goal."""

    def __init__(self) -> None:
        super().__init__("execute_grasp_script_server")

        action_name = self.declare_parameter(
            "action_name", "/execute_grasp"
        ).value
        configured_script = self.declare_parameter("script_path", "").value
        self._poll_interval_sec = max(
            0.05,
            float(self.declare_parameter("poll_interval_sec", 0.1).value),
        )
        self._terminate_timeout_sec = max(
            0.0,
            float(self.declare_parameter("terminate_timeout_sec", 5.0).value),
        )

        default_script = (
            Path(get_package_share_directory(PACKAGE_NAME)) / "main.py"
        )
        self._script_path = Path(configured_script or default_script).expanduser()
        self._script_path = self._script_path.resolve()

        self._state_lock = threading.Lock()
        self._goal_reserved = False
        self._process: Optional[subprocess.Popen[bytes]] = None

        self._action_server = ActionServer(
            self,
            ExecuteGrasp,
            action_name,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            f"ready: action={action_name} script={self._script_path}"
        )

    def _goal_callback(self, _goal_request: ExecuteGrasp.Goal) -> GoalResponse:
        with self._state_lock:
            if self._goal_reserved:
                self.get_logger().warning(
                    "rejecting goal: the configured script is already running"
                )
                return GoalResponse.REJECT
            self._goal_reserved = True

        self.get_logger().info(
            "accepted goal; all ExecuteGrasp goal fields are ignored"
        )
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel_callback(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    @staticmethod
    def _make_result(
        *,
        success: bool,
        message: str,
    ) -> ExecuteGrasp.Result:
        result = ExecuteGrasp.Result()
        result.success = success
        result.message = message
        result.arm_message = message
        result.gripper_command_published = False
        result.torso_reset_command_published = False
        return result

    def _publish_feedback(
        self,
        goal_handle,
        *,
        stage: str,
        detail: str,
    ) -> None:
        feedback = ExecuteGrasp.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    def _execute_callback(self, goal_handle) -> ExecuteGrasp.Result:
        process: Optional[subprocess.Popen[bytes]] = None
        try:
            self._publish_feedback(
                goal_handle,
                stage="STARTING_SCRIPT",
                detail=str(self._script_path),
            )
            if not self._script_path.is_file():
                message = f"script does not exist: {self._script_path}"
                self.get_logger().error(message)
                goal_handle.abort()
                return self._make_result(
                    success=False, message=message
                )

            process = subprocess.Popen(
                [sys.executable, str(self._script_path)],
                cwd=str(self._script_path.parent),
            )
            with self._state_lock:
                self._process = process

            self._publish_feedback(
                goal_handle,
                stage="SCRIPT_RUNNING",
                detail=f"pid={process.pid}",
            )
            while process.poll() is None:
                if goal_handle.is_cancel_requested:
                    self._terminate_process(process)
                    message = "script execution canceled"
                    goal_handle.canceled()
                    return self._make_result(
                        success=False, message=message
                    )
                time.sleep(self._poll_interval_sec)

            return_code = process.returncode
            if return_code == 0:
                message = "script completed successfully"
                goal_handle.succeed()
                return self._make_result(
                    success=True, message=message
                )

            message = f"script failed with exit code {return_code}"
            self.get_logger().error(message)
            goal_handle.abort()
            return self._make_result(
                success=False, message=message
            )
        except OSError as exc:
            message = f"failed to start script: {exc}"
            self.get_logger().error(message)
            goal_handle.abort()
            return self._make_result(
                success=False, message=message
            )
        finally:
            with self._state_lock:
                if self._process is process:
                    self._process = None
                self._goal_reserved = False

    def _terminate_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=self._terminate_timeout_sec)
        except subprocess.TimeoutExpired:
            self.get_logger().warning(
                f"script pid={process.pid} ignored SIGTERM; sending SIGKILL"
            )
            process.kill()
            process.wait()

    def destroy_node(self) -> None:
        with self._state_lock:
            process = self._process
        if process is not None:
            self._terminate_process(process)
        self._action_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExecuteGraspScriptServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
