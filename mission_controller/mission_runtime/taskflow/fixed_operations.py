"""ROS adapters for the deterministic fixed box workflow."""

from __future__ import annotations

import threading
import time
import math
from collections.abc import Callable

from action_msgs.msg import GoalStatus
from mission_interfaces.action import ExecuteBoxGrasp, ExecuteDragBoxGrasp, PlaceBoxTest
from object_pose_interfaces.action import GlobalObservation
from task_interfaces.action import MoveArmJoints

from .fixed_model import FixedBoxTask
from .model import NavigationRequest, StepResult
from .observation import (
    FrontStackPoseValidation,
    ObservationValidationError,
    adapt_global_observation_result,
)
from .model import ObservationResult
from .ros_operations import ObservationGoalConfig


class FixedRosWorkflowOperations:
    """Serialize navigation, grasp, and place child Actions for one workflow."""

    def __init__(
        self,
        *,
        node,
        navigation_gateway,
        direct_grasp_client,
        drag_grasp_client,
        place_test_client,
        observation_client=None,
        observation_goal: ObservationGoalConfig | None = None,
        arm_joints_client=None,
        arm_joints_action_name: str = "/move_arm_j",
        arm_joints_duration: float = 0.0,
        sdk_adapter=None,
        arm_joints_speed_percent: float = 10.0,
        arm_joints_timeout_sec: float = 120.0,
        cancel_event: threading.Event,
        direct_action_name: str,
        drag_action_name: str,
        target_label: int,
        dry_run: bool,
        server_wait_timeout_sec: float,
        result_timeout_sec: float,
        child_feedback_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self._node = node
        self._navigation_gateway = navigation_gateway
        self._direct_grasp_client = direct_grasp_client
        self._drag_grasp_client = drag_grasp_client
        self._place_test_client = place_test_client
        self._observation_client = observation_client
        self._observation_goal = observation_goal
        self._arm_joints_client = arm_joints_client
        self._arm_joints_action_name = str(arm_joints_action_name)
        self._arm_joints_duration = float(arm_joints_duration)
        self._sdk_adapter = sdk_adapter
        self._arm_joints_speed_percent = float(arm_joints_speed_percent)
        self._arm_joints_timeout_sec = float(arm_joints_timeout_sec)
        self._cancel_event = cancel_event
        self._direct_action_name = str(direct_action_name)
        self._drag_action_name = str(drag_action_name)
        self._target_label = int(target_label)
        self._dry_run = bool(dry_run)
        self._server_wait_timeout_sec = float(server_wait_timeout_sec)
        self._result_timeout_sec = float(result_timeout_sec)
        self._child_feedback_callback = child_feedback_callback
        self._active_lock = threading.Lock()
        self._active_child_goal_handle = None

    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def cancel_active(self) -> None:
        self._cancel_event.set()
        self._navigation_gateway.cancel_active()
        with self._active_lock:
            goal_handle = self._active_child_goal_handle
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warning(
                    f"failed to cancel fixed workflow child Action: {exc}"
                )

    def close(self) -> None:
        self._navigation_gateway.close()
        if self._sdk_adapter is not None:
            try:
                self._sdk_adapter.close()
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warning(
                    f"failed to close fixed workflow Python SDK adapter: {exc}"
                )

    def navigate(self, request: NavigationRequest) -> StepResult:
        if self.is_cancel_requested():
            return StepResult(False, "fixed workflow navigation canceled")
        if self._dry_run:
            return StepResult(
                True,
                "fixed workflow navigation skipped in dry-run: "
                f"point={request.point_id}, pos={list(request.pos or ())}",
            )
        result = self._navigation_gateway.navigate(request, self.is_cancel_requested)
        return StepResult(result.success, result.message or result.status)

    def grasp(self, action_name: str, request_id: str, task: FixedBoxTask) -> StepResult:
        if action_name == self._drag_action_name:
            client = self._drag_grasp_client
            goal = ExecuteDragBoxGrasp.Goal()
        elif action_name == self._direct_action_name:
            client = self._direct_grasp_client
            goal = ExecuteBoxGrasp.Goal()
        else:
            return StepResult(False, f"unsupported fixed grasp Action {action_name}")
        goal.request_id = request_id
        goal.target_label = self._target_label
        goal.box_layer = int(task.box_layer)
        goal.box_type = str(task.box_type)
        goal.dry_run = self._dry_run
        call = self._call_action(
            client,
            goal,
            f"{action_name} {task.box_type} layer {task.box_layer}",
        )
        if not call.success:
            return StepResult(False, call.message)
        result = call.result
        if not bool(getattr(result, "success", False)):
            return StepResult(False, str(getattr(result, "message", "grasp failed")))
        return StepResult(True, str(getattr(result, "message", "grasp succeeded")))

    def observe(self, point_id: str) -> ObservationResult:
        if self._observation_client is None or self._observation_goal is None:
            return ObservationResult(
                False, message="global observation is not configured for fixed workflow"
            )
        if self._dry_run:
            return ObservationResult(
                False,
                message="dry-run cannot synthesize a Vision global observation plan",
            )
        goal = GlobalObservation.Goal()
        goal.camera_side = self._observation_goal.camera_side
        goal.max_front_stacks = self._observation_goal.max_front_stacks
        goal.model_label = self._observation_goal.model_label
        goal.confidence_threshold = self._observation_goal.confidence_threshold
        call = self._call_action(
            self._observation_client,
            goal,
            f"Vision global observation at point {point_id}",
        )
        if not call.success:
            return ObservationResult(False, message=call.message)
        try:
            return adapt_global_observation_result(
                point_id,
                call.result,
                front_stack_validation=FrontStackPoseValidation(
                    enabled=self._observation_goal.verify_front_stack_poses,
                    expected_count=self._observation_goal.max_front_stacks,
                    min_lateral_separation_m=(
                        self._observation_goal.front_min_lateral_separation_m
                    ),
                    max_depth_spread_m=(
                        self._observation_goal.front_max_depth_spread_m
                    ),
                    # Mission does not impose an absolute camera-depth gate.
                    max_camera_depth_m=0.0,
                ),
            )
        except ObservationValidationError as exc:
            return ObservationResult(False, message=str(exc))

    def move_left_joints(
        self, left_joints: list[float] | tuple[float, ...], description: str
    ) -> StepResult:
        if self._sdk_adapter is not None:
            try:
                # Mission joint-state values are radians; RealMan's rm_movej
                # API expects seven joint angles in degrees.
                joint_degrees = [math.degrees(float(value)) for value in left_joints]
                message = self._sdk_adapter.execute_single_movej(
                    arm="left",
                    joint_degrees=joint_degrees,
                    speed_percent=self._arm_joints_speed_percent,
                    cancel_requested=self.is_cancel_requested,
                    timeout_sec=self._arm_joints_timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001
                return StepResult(False, f"{description} Python SDK MoveJ failed: {exc}")
            return StepResult(True, message)
        if self._arm_joints_client is None:
            return StepResult(False, "left-arm MoveJ client is unavailable")
        goal = MoveArmJoints.Goal()
        goal.left_joints = [float(value) for value in left_joints]
        goal.right_joints = []
        goal.dry_run = self._dry_run
        goal.duration = self._arm_joints_duration
        call = self._call_action(
            self._arm_joints_client,
            goal,
            f"{description} ({self._arm_joints_action_name})",
        )
        if not call.success:
            return StepResult(False, call.message)
        result = call.result
        if not bool(getattr(result, "success", False)):
            return StepResult(False, str(getattr(result, "message", "MoveJ failed")))
        return StepResult(True, str(getattr(result, "message", "MoveJ completed")))

    def place(self, request_id: str, box_type: str) -> StepResult:
        goal = PlaceBoxTest.Goal()
        goal.request_id = request_id
        goal.box_type = str(box_type)
        goal.release_after_place = True
        goal.dry_run = self._dry_run
        call = self._call_action(self._place_test_client, goal, "/place_box_test")
        if not call.success:
            return StepResult(False, call.message)
        result = call.result
        if not bool(getattr(result, "success", False)):
            return StepResult(False, str(getattr(result, "message", "place failed")))
        return StepResult(True, str(getattr(result, "message", "place succeeded")))

    def _call_action(self, client, goal, description: str):
        if not self._wait_for_server(client, description):
            return _ActionCall(False, None, f"{description} Action server unavailable")
        if self.is_cancel_requested():
            return _ActionCall(False, None, f"{description} canceled")
        try:
            send_future = client.send_goal_async(
                goal,
                feedback_callback=lambda message: self._forward_feedback(
                    description, message
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return _ActionCall(False, None, f"failed to send {description}: {exc}")
        if not self._wait_future(
            send_future,
            self._server_wait_timeout_sec,
            description,
            ignore_cancel=True,
        ):
            send_future.add_done_callback(self._cancel_late_goal)
            return _ActionCall(False, None, f"{description} goal response timed out")
        try:
            goal_handle = send_future.result()
        except Exception as exc:  # noqa: BLE001
            return _ActionCall(False, None, f"{description} goal response failed: {exc}")
        if goal_handle is None or not goal_handle.accepted:
            return _ActionCall(False, None, f"{description} goal was rejected")
        with self._active_lock:
            self._active_child_goal_handle = goal_handle
        try:
            result_future = goal_handle.get_result_async()
            if not self._wait_future(
                result_future, self._result_timeout_sec, description
            ):
                try:
                    goal_handle.cancel_goal_async()
                except Exception:  # noqa: BLE001
                    pass
                if self.is_cancel_requested():
                    self._wait_future(
                        result_future,
                        min(30.0, self._result_timeout_sec),
                        description,
                        ignore_cancel=True,
                    )
                    return _ActionCall(False, None, f"{description} canceled")
                return _ActionCall(False, None, f"{description} result timed out")
            wrapped = result_future.result()
            if wrapped is None:
                return _ActionCall(False, None, f"{description} returned no result")
            if int(wrapped.status) != int(GoalStatus.STATUS_SUCCEEDED):
                child_result = wrapped.result
                details = []
                for field in ("message", "stage", "detail"):
                    value = str(getattr(child_result, field, "")).strip()
                    if value and value not in details:
                        details.append(value)
                failure = f"{description} ended with status {int(wrapped.status)}"
                if details:
                    failure += ": " + "; ".join(details)
                return _ActionCall(
                    False,
                    child_result,
                    failure,
                )
            return _ActionCall(True, wrapped.result, "")
        finally:
            with self._active_lock:
                if self._active_child_goal_handle is goal_handle:
                    self._active_child_goal_handle = None

    def _wait_for_server(self, client, description: str) -> bool:
        deadline = time.monotonic() + self._server_wait_timeout_sec
        while not self.is_cancel_requested():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._node.get_logger().error(
                    f"timed out waiting for {description} Action server"
                )
                return False
            if client.wait_for_server(timeout_sec=min(0.1, remaining)):
                return True
        return False

    def _wait_future(
        self,
        future,
        timeout_sec: float,
        description: str,
        *,
        ignore_cancel: bool = False,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if not ignore_cancel and self.is_cancel_requested():
                return False
            if time.monotonic() >= deadline:
                self._node.get_logger().error(f"{description} timed out")
                return False
            time.sleep(0.01)
        return True

    @staticmethod
    def _cancel_late_goal(future) -> None:
        try:
            goal_handle = future.result()
            if goal_handle is not None and goal_handle.accepted:
                goal_handle.cancel_goal_async()
        except Exception:  # noqa: BLE001
            pass

    def _forward_feedback(self, description: str, feedback_message) -> None:
        if self._child_feedback_callback is None:
            return
        feedback = getattr(feedback_message, "feedback", None)
        stage = str(getattr(feedback, "stage", description))
        detail = str(getattr(feedback, "detail", ""))
        self._child_feedback_callback(stage, detail)


class _ActionCall:
    __slots__ = ("success", "result", "message")

    def __init__(self, success: bool, result: object, message: str) -> None:
        self.success = bool(success)
        self.result = result
        self.message = str(message)
