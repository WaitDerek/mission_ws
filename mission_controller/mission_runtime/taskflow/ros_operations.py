"""ROS Action/Service adapters used by the pure workflow engine."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from action_msgs.msg import GoalStatus
from mission_interfaces.action import (
    ExecuteBoxGrasp,
    ExecuteBoxPlace,
    ExecuteDragBoxGrasp,
)
from object_pose_interfaces.action import GlobalObservation

from .mapping import DIRECT_GRASP_ACTION, DRAG_GRASP_ACTION
from .model import NavigationRequest, ObservationResult, ObservationTask, StepResult
from .navigation import NavigationGateway
from .observation import (
    FrontStackPoseValidation,
    ObservationValidationError,
    adapt_global_observation_result,
)


@dataclass(frozen=True)
class ObservationGoalConfig:
    camera_side: str
    max_front_stacks: int
    model_label: str
    confidence_threshold: float
    verify_front_stack_poses: bool
    front_min_lateral_separation_m: float
    front_max_depth_spread_m: float
    front_max_camera_depth_m: float


class RosWorkflowOperations:
    """Concrete workflow effects with one active ROS child Action at a time."""

    def __init__(
        self,
        *,
        node,
        navigation_gateway: NavigationGateway,
        observation_client,
        direct_grasp_client,
        drag_grasp_client,
        place_client,
        cancel_event: threading.Event,
        observation_goal: ObservationGoalConfig,
        target_label: int,
        dry_run: bool,
        server_wait_timeout_sec: float,
        result_timeout_sec: float,
        child_feedback_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self._node = node
        self._navigation_gateway = navigation_gateway
        self._observation_client = observation_client
        self._direct_grasp_client = direct_grasp_client
        self._drag_grasp_client = drag_grasp_client
        self._place_client = place_client
        self._cancel_event = cancel_event
        self._observation_goal = observation_goal
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
                    f"failed to request child Action cancellation: {exc}"
                )

    def close(self) -> None:
        self._navigation_gateway.close()

    def navigate(self, request: NavigationRequest) -> StepResult:
        result = self._navigation_gateway.navigate(request, self.is_cancel_requested)
        return StepResult(result.success, result.message or result.status)

    def observe(self, point_id: str) -> ObservationResult:
        goal = GlobalObservation.Goal()
        goal.camera_side = self._observation_goal.camera_side
        goal.max_front_stacks = self._observation_goal.max_front_stacks
        goal.model_label = self._observation_goal.model_label
        goal.confidence_threshold = self._observation_goal.confidence_threshold
        call = self._call_action(
            self._observation_client,
            goal,
            f"Vision observation at point {point_id}",
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
                    max_camera_depth_m=(
                        self._observation_goal.front_max_camera_depth_m
                    ),
                ),
            )
        except ObservationValidationError as exc:
            return ObservationResult(False, message=str(exc))

    def grasp(
        self,
        action_name: str,
        request_id: str,
        task: ObservationTask,
        operation_point_id: str,
    ) -> StepResult:
        if action_name == DRAG_GRASP_ACTION:
            goal = ExecuteDragBoxGrasp.Goal()
            client = self._drag_grasp_client
        elif action_name == DIRECT_GRASP_ACTION:
            goal = ExecuteBoxGrasp.Goal()
            client = self._direct_grasp_client
        else:
            return StepResult(False, f"unsupported grasp Action {action_name}")
        goal.request_id = request_id
        goal.target_label = self._target_label
        goal.box_layer = task.layer
        goal.box_type = task.box_type
        goal.dry_run = self._dry_run
        call = self._call_action(
            client,
            goal,
            f"grasp at operation point {operation_point_id}",
        )
        if not call.success:
            return StepResult(False, call.message)
        return StepResult(
            bool(call.result.success),
            str(call.result.message),
        )

    def place(self, request_id: str, task: ObservationTask) -> StepResult:
        del task
        goal = ExecuteBoxPlace.Goal()
        goal.request_id = request_id
        goal.dry_run = self._dry_run
        call = self._call_action(self._place_client, goal, "box placement")
        if not call.success:
            return StepResult(False, call.message)
        return StepResult(
            bool(call.result.success),
            str(call.result.message),
        )

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
        goal_handle = send_future.result()
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
                return _ActionCall(
                    False,
                    wrapped.result,
                    f"{description} ended with status {wrapped.status}",
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


@dataclass(frozen=True)
class _ActionCall:
    success: bool
    result: object
    message: str
