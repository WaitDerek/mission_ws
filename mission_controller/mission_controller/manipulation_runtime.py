"""Shared ROS clients, pose state, and preparation for manipulation tasks."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from object_pose_interfaces.action import EstimateObjectPose
from rclpy.action import ActionClient
from task_interfaces.action import MoveArmJoints, MoveArmPose

from .common import MissionCanceled, MissionError

try:
    from object_pose_interfaces.action import EstimateFrontBumperPose
except ImportError:  # The source interface may be newer than the sourced install.
    EstimateFrontBumperPose = None


class ManipulationRuntimeMixin:
    """Provide the motion, perception, and EE-pose primitives shared by both actions."""

    def _initialize_pipeline_runtime(self) -> None:
        self._grip_config = self._load_pipeline_config("grip_config_file")
        self._connector_grip_config = self._load_pipeline_config(
            "connector_grip_config_file"
        )
        self._peel_config = self._load_pipeline_config("peel_config_file")
        self._assembly_config = self._load_pipeline_config("assembly_config_file")

        self._pipeline_condition = threading.Condition()
        self._pipeline_left_ee_pose: Optional[PoseStamped] = None
        self._pipeline_right_ee_pose: Optional[PoseStamped] = None
        self._pipeline_left_ee_sequence = 0
        self._pipeline_right_ee_sequence = 0

        self.pipeline_left_ee_subscription = self.create_subscription(
            PoseStamped,
            self._string("pipeline_left_ee_pose_topic"),
            self._pipeline_left_ee_callback,
            10,
            callback_group=self._callback_group,
        )
        self.pipeline_right_ee_subscription = self.create_subscription(
            PoseStamped,
            self._string("pipeline_right_ee_pose_topic"),
            self._pipeline_right_ee_callback,
            10,
            callback_group=self._callback_group,
        )
        self.arm_linear_client = ActionClient(
            self,
            MoveArmPose,
            self._string("move_arm_linear_action_name"),
            callback_group=self._callback_group,
        )
        self.front_bumper_pose_client = None
        if EstimateFrontBumperPose is not None:
            self.front_bumper_pose_client = ActionClient(
                self,
                EstimateFrontBumperPose,
                self._string("front_bumper_pose_action_name"),
                callback_group=self._callback_group,
            )
        else:
            self.get_logger().warning(
                "EstimateFrontBumperPose is unavailable; /execute_assembly "
                "requires rebuilding and sourcing the current vision interfaces"
            )

    def _load_pipeline_config(self, parameter_name: str) -> dict[str, Any]:
        configured_path = Path(self._string(parameter_name)).expanduser()
        if not configured_path.is_absolute():
            configured_path = (
                Path(get_package_share_directory("mission_controller"))
                / "config"
                / configured_path
            )
        try:
            with configured_path.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"unable to load {parameter_name} '{configured_path}': {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise ValueError(
                f"{parameter_name} '{configured_path}' must contain an object"
            )
        self.get_logger().info(f"loaded {parameter_name} from '{configured_path.name}'")
        return document

    def _pipeline_left_ee_callback(self, message: PoseStamped) -> None:
        with self._pipeline_condition:
            self._pipeline_left_ee_pose = message
            self._pipeline_left_ee_sequence += 1
            self._pipeline_condition.notify_all()

    def _pipeline_right_ee_callback(self, message: PoseStamped) -> None:
        with self._pipeline_condition:
            self._pipeline_right_ee_pose = message
            self._pipeline_right_ee_sequence += 1
            self._pipeline_condition.notify_all()

    def _cancelable_sleep(self, goal_handle, seconds: float, context: str) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, context)
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _wait_for_pipeline_poses(
        self,
        goal_handle,
        *,
        need_left: bool,
        need_right: bool,
        require_fresh: bool,
    ) -> tuple[Optional[PoseStamped], Optional[PoseStamped]]:
        with self._pipeline_condition:
            left_before = self._pipeline_left_ee_sequence
            right_before = self._pipeline_right_ee_sequence
        deadline = time.monotonic() + self._float("ee_pose_timeout_sec")
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for EE poses")
            with self._pipeline_condition:
                left_ready = not need_left or (
                    self._pipeline_left_ee_pose is not None
                    and (
                        not require_fresh
                        or self._pipeline_left_ee_sequence > left_before
                    )
                )
                right_ready = not need_right or (
                    self._pipeline_right_ee_pose is not None
                    and (
                        not require_fresh
                        or self._pipeline_right_ee_sequence > right_before
                    )
                )
                if left_ready and right_ready:
                    return self._pipeline_left_ee_pose, self._pipeline_right_ee_pose
            time.sleep(0.02)
        requested = (
            "left/right"
            if need_left and need_right
            else "left" if need_left else "right"
        )
        freshness = "fresh " if require_fresh else ""
        raise MissionError(f"no {freshness}{requested} EE pose before timeout")

    def _send_pipeline_action(
        self,
        goal_handle,
        *,
        client,
        action_name: str,
        request,
        handle_key: str,
        result_timeout: float,
    ):
        self._wait_for_server(client, action_name, goal_handle)
        child = self._wait_future(
            client.send_goal_async(request),
            goal_handle,
            f"sending {action_name}",
            self._float("dependency_wait_timeout_sec"),
        )
        if child is None or not child.accepted:
            raise MissionError(f"{action_name} goal was rejected")
        with self._state_lock:
            self._active_child_handles[handle_key] = child
        try:
            try:
                wrapped = self._wait_future(
                    child.get_result_async(),
                    goal_handle,
                    f"waiting for {action_name} result",
                    result_timeout,
                )
            except Exception:
                try:
                    child.cancel_goal_async()
                except Exception:  # noqa: BLE001 - preserve the root failure.
                    pass
                raise
        finally:
            with self._state_lock:
                self._active_child_handles.pop(handle_key, None)
        result = wrapped.result
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
            raise MissionError(
                f"{action_name} failed: error_code={result.error_code}, "
                f"message={result.message}"
            )
        return result

    def _pipeline_move_joints(
        self, goal_handle, left: list[float], right: list[float]
    ) -> None:
        request = MoveArmJoints.Goal()
        request.left_joints = [float(value) for value in left]
        request.right_joints = [float(value) for value in right]
        request.dry_run = False
        request.duration = 0.0
        self._send_pipeline_action(
            goal_handle,
            client=self.arm_joints_client,
            action_name=self._string("move_arm_joints_action_name"),
            request=request,
            handle_key="pipeline_move_arm_j",
            result_timeout=self._float("arm_pose_timeout_sec"),
        )

    def _pipeline_move_pose(
        self,
        goal_handle,
        *,
        left: list[float],
        right: list[float],
        linear: bool = False,
    ) -> None:
        request = MoveArmPose.Goal()
        request.left_pose = [float(value) for value in left]
        request.right_pose = [float(value) for value in right]
        request.dry_run = False
        client = self.arm_linear_client if linear else self.arm_pose_client
        action_name = self._string(
            "move_arm_linear_action_name" if linear else "move_arm_pose_action_name"
        )
        self._send_pipeline_action(
            goal_handle,
            client=client,
            action_name=action_name,
            request=request,
            handle_key="pipeline_move_arm_l" if linear else "pipeline_move_arm_p",
            result_timeout=self._float("arm_pose_timeout_sec"),
        )

    def _estimate_pipeline_object(self, goal_handle, model_label: str):
        action_name = self._string("object_pose_action_name")
        self._wait_for_server(self.object_pose_client, action_name, goal_handle)
        request = EstimateObjectPose.Goal()
        request.model_label = str(model_label)
        request.instance_index = 0
        request.confidence_threshold = 0.0
        child = self._wait_future(
            self.object_pose_client.send_goal_async(request),
            goal_handle,
            f"sending {action_name} for {model_label}",
            self._float("dependency_wait_timeout_sec"),
        )
        if child is None or not child.accepted:
            raise MissionError(f"{action_name} goal for {model_label} was rejected")
        with self._state_lock:
            self._active_child_handles["pipeline_object_pose"] = child
        try:
            try:
                wrapped = self._wait_future(
                    child.get_result_async(),
                    goal_handle,
                    f"waiting for {model_label} detection",
                    self._float("object_pose_timeout_sec"),
                )
            except Exception:
                try:
                    child.cancel_goal_async()
                except Exception:  # noqa: BLE001 - preserve the root failure.
                    pass
                raise
        finally:
            with self._state_lock:
                self._active_child_handles.pop("pipeline_object_pose", None)
        result = wrapped.result
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
            raise MissionError(
                f"{action_name} failed for {model_label}: {result.message}"
            )
        return result.pose

    def _estimate_front_bumper_task(self, goal_handle, patch_name: str):
        if EstimateFrontBumperPose is None or self.front_bumper_pose_client is None:
            raise MissionError(
                "EstimateFrontBumperPose is unavailable; rebuild and source vision_ws"
            )
        action_name = self._string("front_bumper_pose_action_name")
        self._wait_for_server(self.front_bumper_pose_client, action_name, goal_handle)
        request = EstimateFrontBumperPose.Goal()
        request.patch_name = str(patch_name)
        request.require_full_pose = True
        request.roi_x = 0
        request.roi_y = 0
        request.roi_width = 0
        request.roi_height = 0
        child = self._wait_future(
            self.front_bumper_pose_client.send_goal_async(request),
            goal_handle,
            f"sending {action_name} for {patch_name}",
            self._float("dependency_wait_timeout_sec"),
        )
        if child is None or not child.accepted:
            raise MissionError(f"{action_name} goal for {patch_name} was rejected")
        with self._state_lock:
            self._active_child_handles["front_bumper_pose"] = child
        try:
            try:
                wrapped = self._wait_future(
                    child.get_result_async(),
                    goal_handle,
                    f"waiting for front-bumper patch {patch_name}",
                    self._float("object_pose_timeout_sec"),
                )
            except Exception:
                try:
                    child.cancel_goal_async()
                except Exception:  # noqa: BLE001 - preserve the root failure.
                    pass
                raise
        finally:
            with self._state_lock:
                self._active_child_handles.pop("front_bumper_pose", None)
        result = wrapped.result
        if (
            wrapped.status != GoalStatus.STATUS_SUCCEEDED
            or not result.success
            or not result.has_task_pose
        ):
            raise MissionError(
                f"{action_name} failed for {patch_name}: "
                f"status={result.status}, message={result.message}"
            )
        actual_frame = str(result.task_pose.header.frame_id).strip().lstrip("/")
        expected_frame = self._string("camera_frame").lstrip("/")
        if actual_frame and actual_frame != expected_frame:
            raise MissionError(
                f"{action_name} returned frame {actual_frame!r}; "
                f"expected {expected_frame!r}"
            )
        return result.task_pose

    def _prepare_pipeline(
        self,
        goal_handle,
        action_type,
        config: dict[str, Any],
    ) -> None:
        if "if_prepare" in config and not config["if_prepare"]:
            return
        trajectory = config["prepare_traj"]
        if config.get("test_mode", False):
            trajectory = config["test_traj"]["prepare"]
        left_trajectory = trajectory["left"]
        right_trajectory = trajectory["right"]
        if len(left_trajectory) != len(right_trajectory):
            raise MissionError(
                "prepare trajectory left/right waypoint counts differ: "
                f"{len(left_trajectory)} != {len(right_trajectory)}"
            )
        for index, (left, right) in enumerate(
            zip(left_trajectory, right_trajectory), start=1
        ):
            self._pipeline_feedback(
                goal_handle,
                action_type,
                "PREPARING",
                f"executing joint waypoint {index}/{len(left_trajectory)}",
            )
            self._pipeline_move_joints(goal_handle, left, right)
        self._cancelable_sleep(goal_handle, 2.0, "after preparation trajectory")
