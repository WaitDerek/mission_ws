"""ROS node exposing the platform-facing depalletizing workflow Action."""

from __future__ import annotations

import threading
import time
from typing import Optional

import rclpy
from mission_interfaces.action import (
    ExecuteBoxGrasp,
    ExecuteBoxPlace,
    ExecuteDragBoxGrasp,
    ExecuteWorkflow,
)
from mission_interfaces.srv import AcquireMissionLease, ReleaseMissionLease
from object_pose_interfaces.action import GlobalObservation
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .identifiers import new_workflow_id
from .mqtt_navigation import MqttNavigationGateway, parse_navigation_points_json
from .mqtt_start import MqttStartRequest, MqttWorkflowStartBridge
from .navigation import DisabledNavigationGateway
from .ros_operations import ObservationGoalConfig, RosWorkflowOperations
from .state_machine import DepalletizingWorkflowEngine


class DepalletizingWorkflowNode(Node):
    def __init__(self) -> None:
        super().__init__("execute_workflow")
        self._declare_parameters()
        self._validate_parameters()
        self._workflow_lock = threading.Lock()
        self._workflow_reserved = False
        self._cancel_event = threading.Event()
        self._active_operations: Optional[RosWorkflowOperations] = None
        self._active_goal_handle = None
        self._mqtt_start_lock = threading.Lock()
        self._mqtt_start_busy = False
        self._mqtt_pending_start = None
        self._mqtt_start_bridge = None

        self._server_group = ReentrantCallbackGroup()
        self._client_group = ReentrantCallbackGroup()
        self._observation_client = ActionClient(
            self,
            GlobalObservation,
            self._string("global_observation_action_name"),
            callback_group=self._client_group,
        )
        self._direct_grasp_client = ActionClient(
            self,
            ExecuteBoxGrasp,
            self._string("grasp_box_tf_action_name"),
            callback_group=self._client_group,
        )
        self._drag_grasp_client = ActionClient(
            self,
            ExecuteDragBoxGrasp,
            self._string("execute_drag_box_grasp_tf_action_name"),
            callback_group=self._client_group,
        )
        self._place_client = ActionClient(
            self,
            ExecuteBoxPlace,
            self._string("execute_box_place_action_name"),
            callback_group=self._client_group,
        )
        self._acquire_lease_client = self.create_client(
            AcquireMissionLease,
            self._string("acquire_mission_lease_service_name"),
            callback_group=self._client_group,
        )
        self._release_lease_client = self.create_client(
            ReleaseMissionLease,
            self._string("release_mission_lease_service_name"),
            callback_group=self._client_group,
        )
        self._action_server = ActionServer(
            self,
            ExecuteWorkflow,
            self._string("workflow_action_name"),
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._server_group,
        )
        self._mqtt_workflow_client = ActionClient(
            self,
            ExecuteWorkflow,
            self._string("workflow_action_name"),
            callback_group=self._client_group,
        )
        self._mqtt_start_timer = self.create_timer(
            0.05,
            self._dispatch_mqtt_start,
            callback_group=self._client_group,
        )
        if self._boolean("mqtt_start_enabled"):
            self._mqtt_start_bridge = MqttWorkflowStartBridge(
                host=self._string("mqtt_host"),
                port=self._integer("mqtt_port"),
                start_topic=self._string("mqtt_start_topic"),
                status_topic=self._string("mqtt_status_topic"),
                client_id=self._string("mqtt_trigger_client_id"),
                qos=self._integer("mqtt_qos"),
                keepalive_sec=self._integer("mqtt_keepalive_sec"),
                on_start=self._queue_mqtt_start,
            )
        self.get_logger().info(
            "depalletizing workflow ready: "
            f"action={self._string('workflow_action_name')} "
            f"navigation_adapter={self._string('navigation_adapter')} "
            f"mqtt_start_enabled={self._boolean('mqtt_start_enabled')}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameters(
            namespace="",
            parameters=[
                ("workflow_action_name", "/execute_workflow"),
                ("global_observation_action_name", "/depalletizing/observe"),
                ("grasp_box_tf_action_name", "/grasp_box_tf"),
                (
                    "execute_drag_box_grasp_tf_action_name",
                    "/execute_drag_box_grasp_tf",
                ),
                ("execute_box_place_action_name", "/execute_box_place"),
                (
                    "acquire_mission_lease_service_name",
                    "/mission/acquire_workflow_lease",
                ),
                (
                    "release_mission_lease_service_name",
                    "/mission/release_workflow_lease",
                ),
                ("navigation_adapter", "disabled"),
                ("mqtt_host", "127.0.0.1"),
                ("mqtt_port", 1883),
                ("mqtt_request_topic", "mission/navigation/request"),
                ("mqtt_result_topic", "mission/navigation/result"),
                ("mqtt_client_id", ""),
                ("mqtt_qos", 1),
                ("mqtt_keepalive_sec", 60),
                ("mqtt_connect_timeout_sec", 10.0),
                ("mqtt_navigation_timeout_sec", 300.0),
                ("mqtt_navigation_frame_id", "map"),
                ("mqtt_navigation_points_json", "{}"),
                ("mqtt_start_enabled", False),
                ("mqtt_start_topic", "mission/workflow/start"),
                ("mqtt_status_topic", "mission/workflow/status"),
                ("mqtt_trigger_client_id", ""),
                ("mqtt_start_action_wait_timeout_sec", 5.0),
                ("global_observation_camera_side", "left"),
                ("global_observation_max_front_stacks", 2),
                ("global_observation_model_label", ""),
                ("global_observation_confidence_threshold", 0.0),
                ("global_observation_verify_front_stack_poses", True),
                ("global_observation_front_min_lateral_separation_m", 0.20),
                ("global_observation_front_max_depth_spread_m", 0.35),
                ("global_observation_front_max_camera_depth_m", 1.20),
                ("target_label", 0),
                ("dry_run", False),
                ("server_wait_timeout_sec", 10.0),
                ("child_result_timeout_sec", 1200.0),
                ("lease_service_timeout_sec", 5.0),
            ],
        )

    def _validate_parameters(self) -> None:
        if self._string("navigation_adapter") not in ("disabled", "mqtt"):
            raise ValueError(
                "navigation_adapter must be disabled or mqtt"
            )
        if self._string("global_observation_camera_side").lower() not in (
            "left",
            "right",
        ):
            raise ValueError("global_observation_camera_side must be left or right")
        if self._integer("global_observation_max_front_stacks") <= 0:
            raise ValueError("global_observation_max_front_stacks must be positive")
        for name in (
            "global_observation_front_min_lateral_separation_m",
            "global_observation_front_max_depth_spread_m",
        ):
            if self._float(name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self._float("global_observation_front_max_camera_depth_m") < 0.0:
            raise ValueError(
                "global_observation_front_max_camera_depth_m cannot be negative"
            )
        for name in (
            "server_wait_timeout_sec",
            "child_result_timeout_sec",
            "lease_service_timeout_sec",
            "mqtt_connect_timeout_sec",
            "mqtt_navigation_timeout_sec",
            "mqtt_start_action_wait_timeout_sec",
        ):
            if self._float(name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self._integer("mqtt_port") <= 65535:
            raise ValueError("mqtt_port must be in [1, 65535]")
        if self._integer("mqtt_qos") not in (0, 1, 2):
            raise ValueError("mqtt_qos must be 0, 1, or 2")
        if self._integer("mqtt_keepalive_sec") <= 0:
            raise ValueError("mqtt_keepalive_sec must be positive")
        mqtt_used = self._string("navigation_adapter") == "mqtt" or self._boolean(
            "mqtt_start_enabled"
        )
        if mqtt_used:
            for name in ("mqtt_host",):
                if not self._string(name):
                    raise ValueError(f"{name} must not be empty")
        if self._string("navigation_adapter") == "mqtt":
            for name in ("mqtt_request_topic", "mqtt_result_topic"):
                if not self._string(name):
                    raise ValueError(f"{name} must not be empty")
            if self._string("mqtt_request_topic") == self._string(
                "mqtt_result_topic"
            ):
                raise ValueError("MQTT request and result topics must differ")
            if not self._string("mqtt_navigation_frame_id").lstrip("/"):
                raise ValueError("mqtt_navigation_frame_id must not be empty")
            parse_navigation_points_json(
                self._string("mqtt_navigation_points_json")
            )
        if self._boolean("mqtt_start_enabled"):
            for name in ("mqtt_start_topic", "mqtt_status_topic"):
                if not self._string(name):
                    raise ValueError(f"{name} must not be empty")
            topics = (
                self._string("mqtt_start_topic"),
                self._string("mqtt_status_topic"),
                self._string("mqtt_request_topic"),
                self._string("mqtt_result_topic"),
            )
            if len(set(topics)) != len(topics):
                raise ValueError("MQTT workflow and navigation topics must differ")

    def _queue_mqtt_start(self, request: MqttStartRequest) -> None:
        normalized = MqttStartRequest(
            request_id=request.request_id or f"mqtt-{new_workflow_id()}",
            id=request.id,
        )
        if normalized.id != 0:
            bridge = self._mqtt_start_bridge
            if bridge is not None:
                bridge.publish_status(
                    {
                        "event": "rejected",
                        "id": normalized.id,
                        "request_id": normalized.request_id,
                        "message": "workflow MQTT id must be 0",
                    }
                )
            return
        with self._mqtt_start_lock:
            if self._mqtt_start_busy:
                rejected = True
            else:
                rejected = False
                self._mqtt_start_busy = True
                self._mqtt_pending_start = (
                    normalized,
                    time.monotonic()
                    + self._float("mqtt_start_action_wait_timeout_sec"),
                )
        if rejected:
            self._publish_mqtt_workflow_status(
                "rejected",
                normalized.request_id,
                message="another MQTT workflow request is active",
            )
            return
        self._publish_mqtt_workflow_status(
            "received",
            normalized.request_id,
            message="MQTT start request queued",
        )

    def _dispatch_mqtt_start(self) -> None:
        with self._mqtt_start_lock:
            pending = self._mqtt_pending_start
        if pending is None:
            return
        request, deadline = pending
        if not self._mqtt_workflow_client.server_is_ready():
            if time.monotonic() < deadline:
                return
            self._finish_mqtt_start(
                "rejected",
                request.request_id,
                message="workflow Action server is unavailable",
            )
            return
        with self._mqtt_start_lock:
            if self._mqtt_pending_start != pending:
                return
            self._mqtt_pending_start = None
        goal = ExecuteWorkflow.Goal()
        goal.start = True
        try:
            future = self._mqtt_workflow_client.send_goal_async(
                goal,
                feedback_callback=lambda message: self._mqtt_workflow_feedback(
                    request.request_id, message
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._finish_mqtt_start(
                "rejected",
                request.request_id,
                message=f"failed to send workflow Action goal: {exc}",
            )
            return
        future.add_done_callback(
            lambda result: self._mqtt_workflow_goal_response(request.request_id, result)
        )

    def _mqtt_workflow_goal_response(self, request_id: str, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self._finish_mqtt_start(
                "rejected",
                request_id,
                message=f"workflow Action goal failed: {exc}",
            )
            return
        if not goal_handle.accepted:
            self._finish_mqtt_start(
                "rejected",
                request_id,
                message="workflow Action goal was rejected",
            )
            return
        self._publish_mqtt_workflow_status(
            "accepted",
            request_id,
            message="workflow Action goal accepted",
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result: self._mqtt_workflow_result(request_id, result)
        )

    def _mqtt_workflow_feedback(self, request_id: str, wrapped_feedback) -> None:
        feedback = wrapped_feedback.feedback
        self._publish_mqtt_workflow_status(
            "feedback",
            request_id,
            workflow_id=feedback.workflow_id,
            stage=feedback.stage,
            point_id=feedback.current_point_id,
            stack_id=feedback.current_stack_id,
            current_order_index=int(feedback.current_order_index),
            total_order_items=int(feedback.total_order_items),
            detail=feedback.detail,
        )

    def _mqtt_workflow_result(self, request_id: str, future) -> None:
        try:
            wrapped = future.result()
            result = wrapped.result
        except Exception as exc:  # noqa: BLE001
            self._finish_mqtt_start(
                "result",
                request_id,
                success=False,
                message=f"workflow Action result failed: {exc}",
            )
            return
        self._finish_mqtt_start(
            "result",
            request_id,
            success=bool(result.success),
            workflow_id=result.workflow_id,
            message=result.message,
            completed_observation_count=int(result.completed_observation_count),
            completed_box_count=int(result.completed_box_count),
            final_stage=result.final_stage,
            ros_status=int(wrapped.status),
        )

    def _publish_mqtt_workflow_status(
        self,
        event: str,
        request_id: str,
        **values,
    ) -> bool:
        bridge = self._mqtt_start_bridge
        if bridge is None:
            return False
        return bridge.publish_status(
            {"event": event, "id": 0, "request_id": request_id, **values}
        )

    def _finish_mqtt_start(
        self,
        event: str,
        request_id: str,
        **values,
    ) -> None:
        with self._mqtt_start_lock:
            self._mqtt_start_busy = False
            self._mqtt_pending_start = None
        self._publish_mqtt_workflow_status(event, request_id, **values)

    def _goal_callback(self, request) -> GoalResponse:
        if not request.start:
            self.get_logger().warning("rejecting workflow goal: start must be true")
            return GoalResponse.REJECT
        with self._workflow_lock:
            if self._workflow_reserved:
                self.get_logger().warning(
                    "rejecting workflow goal: another workflow is active"
                )
                return GoalResponse.REJECT
            self._workflow_reserved = True
            self._cancel_event.clear()
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        self._cancel_event.set()
        with self._workflow_lock:
            operations = self._active_operations
        if operations is not None:
            operations.cancel_active()
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        workflow_id = new_workflow_id()
        lease_token = ""
        outcome = None
        release_error = ""
        with self._workflow_lock:
            self._active_goal_handle = goal_handle
        try:
            acquired = self._acquire_lease(workflow_id)
            if not acquired.success:
                outcome = self._failure_outcome(
                    workflow_id, "ACQUIRE_LEASE", acquired.message
                )
            else:
                lease_token = acquired.lease_token
                operations = self._make_operations(goal_handle, workflow_id)
                with self._workflow_lock:
                    self._active_operations = operations
                engine = DepalletizingWorkflowEngine(
                    operations,
                    progress_callback=lambda progress: self._publish_progress(
                        goal_handle, progress
                    ),
                )
                outcome = engine.run(workflow_id, lease_token)
                if not outcome.success:
                    self.get_logger().error(
                        "depalletizing workflow stopped "
                        f"workflow_id={workflow_id} "
                        f"stage={outcome.final_stage} message={outcome.message}"
                    )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                "unexpected depalletizing workflow error "
                f"workflow_id={workflow_id}: {exc}"
            )
            outcome = self._failure_outcome(
                workflow_id,
                "INTERNAL_ERROR",
                f"unexpected workflow error: {exc}",
            )
        finally:
            with self._workflow_lock:
                operations = self._active_operations
                self._active_operations = None
            if self._cancel_event.is_set() and operations is not None:
                operations.cancel_active()
            if operations is not None:
                close_operations = getattr(operations, "close", None)
                if callable(close_operations):
                    close_operations()
            if lease_token:
                released = self._release_lease(workflow_id, lease_token)
                if not released.success:
                    release_error = released.message
            with self._workflow_lock:
                self._active_goal_handle = None
                self._workflow_reserved = False

        if outcome is None:
            outcome = self._failure_outcome(
                workflow_id, "INTERNAL_ERROR", "workflow returned no outcome"
            )
        if release_error:
            outcome = self._failure_outcome(
                workflow_id,
                "RELEASE_LEASE",
                f"workflow stopped but Mission lease release failed: {release_error}",
                completed_observations=outcome.completed_observation_count,
                completed_boxes=outcome.completed_box_count,
            )

        result = ExecuteWorkflow.Result()
        result.success = outcome.success
        result.workflow_id = workflow_id
        result.message = outcome.message
        result.completed_observation_count = outcome.completed_observation_count
        result.completed_box_count = outcome.completed_box_count
        result.final_stage = outcome.final_stage
        if self._cancel_event.is_set() or outcome.final_stage == "CANCELED":
            goal_handle.canceled()
        elif outcome.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _make_operations(self, goal_handle, workflow_id: str) -> RosWorkflowOperations:
        return RosWorkflowOperations(
            node=self,
            navigation_gateway=self._make_navigation_gateway(),
            observation_client=self._observation_client,
            direct_grasp_client=self._direct_grasp_client,
            drag_grasp_client=self._drag_grasp_client,
            place_client=self._place_client,
            cancel_event=self._cancel_event,
            observation_goal=ObservationGoalConfig(
                camera_side=self._string("global_observation_camera_side"),
                max_front_stacks=self._integer("global_observation_max_front_stacks"),
                model_label=self._string("global_observation_model_label"),
                confidence_threshold=self._float(
                    "global_observation_confidence_threshold"
                ),
                verify_front_stack_poses=self._boolean(
                    "global_observation_verify_front_stack_poses"
                ),
                front_min_lateral_separation_m=self._float(
                    "global_observation_front_min_lateral_separation_m"
                ),
                front_max_depth_spread_m=self._float(
                    "global_observation_front_max_depth_spread_m"
                ),
                front_max_camera_depth_m=self._float(
                    "global_observation_front_max_camera_depth_m"
                ),
            ),
            target_label=self._integer("target_label"),
            dry_run=self._boolean("dry_run"),
            server_wait_timeout_sec=self._float("server_wait_timeout_sec"),
            result_timeout_sec=self._float("child_result_timeout_sec"),
            child_feedback_callback=lambda stage, detail: self._publish_child_feedback(
                goal_handle, workflow_id=workflow_id, stage=stage, detail=detail
            ),
        )

    def _make_navigation_gateway(self):
        if self._string("navigation_adapter") == "disabled":
            return DisabledNavigationGateway()
        return MqttNavigationGateway(
            host=self._string("mqtt_host"),
            port=self._integer("mqtt_port"),
            request_topic=self._string("mqtt_request_topic"),
            result_topic=self._string("mqtt_result_topic"),
            client_id=self._string("mqtt_client_id"),
            qos=self._integer("mqtt_qos"),
            keepalive_sec=self._integer("mqtt_keepalive_sec"),
            connect_timeout_sec=self._float("mqtt_connect_timeout_sec"),
            navigation_timeout_sec=self._float("mqtt_navigation_timeout_sec"),
            frame_id=self._string("mqtt_navigation_frame_id"),
            point_poses=parse_navigation_points_json(
                self._string("mqtt_navigation_points_json")
            ),
        )

    def _publish_progress(self, goal_handle, progress) -> None:
        self.get_logger().info(
            "workflow transition "
            f"workflow_id={progress.workflow_id} stage={progress.stage} "
            f"point_id={progress.current_point_id or '<none>'} "
            f"stack_id={progress.current_stack_id or '<none>'} "
            f"order_index={progress.current_order_index}/"
            f"{progress.total_order_items} detail={progress.detail}"
        )
        feedback = ExecuteWorkflow.Feedback()
        feedback.workflow_id = progress.workflow_id
        feedback.stage = progress.stage
        feedback.current_point_id = progress.current_point_id
        feedback.current_stack_id = progress.current_stack_id
        feedback.current_order_index = progress.current_order_index
        feedback.total_order_items = progress.total_order_items
        feedback.detail = progress.detail
        goal_handle.publish_feedback(feedback)

    def _publish_child_feedback(
        self, goal_handle, workflow_id: str, stage: str, detail: str
    ) -> None:
        feedback = ExecuteWorkflow.Feedback()
        feedback.workflow_id = workflow_id
        feedback.stage = f"CHILD_{stage}"
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    def _acquire_lease(self, workflow_id: str):
        request = AcquireMissionLease.Request()
        request.workflow_id = workflow_id
        response = self._call_service(
            self._acquire_lease_client,
            request,
            "acquire Mission workflow lease",
            ignore_cancel=True,
        )
        if response is None:
            fallback = AcquireMissionLease.Response()
            fallback.success = False
            fallback.message = "Mission lease service unavailable or timed out"
            return fallback
        return response

    def _release_lease(self, workflow_id: str, lease_token: str):
        request = ReleaseMissionLease.Request()
        request.workflow_id = workflow_id
        request.lease_token = lease_token
        response = self._call_service(
            self._release_lease_client,
            request,
            "release Mission workflow lease",
            ignore_cancel=True,
            secret=lease_token,
        )
        if response is None:
            fallback = ReleaseMissionLease.Response()
            fallback.success = False
            fallback.message = "Mission lease release service unavailable or timed out"
            return fallback
        return response

    def _call_service(
        self,
        client,
        request,
        description: str,
        *,
        ignore_cancel: bool = False,
        secret: str = "",
    ):
        timeout = self._float("lease_service_timeout_sec")
        deadline = time.monotonic() + timeout
        while not client.wait_for_service(timeout_sec=0.1):
            if not ignore_cancel and self._cancel_event.is_set():
                return None
            if time.monotonic() >= deadline:
                self.get_logger().error(f"timed out waiting to {description}")
                return None
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            if not ignore_cancel and self._cancel_event.is_set():
                return None
            if time.monotonic() >= deadline:
                self.get_logger().error(f"timed out trying to {description}")
                return None
            time.sleep(0.01)
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
            if secret:
                detail = detail.replace(secret, "<redacted>")
            self.get_logger().error(f"failed to {description}: {detail}")
            return None

    @staticmethod
    def _failure_outcome(
        workflow_id: str,
        stage: str,
        message: str,
        *,
        completed_observations: int = 0,
        completed_boxes: int = 0,
    ):
        from .model import WorkflowOutcome

        return WorkflowOutcome(
            False,
            workflow_id,
            message,
            stage,
            completed_observations,
            completed_boxes,
        )

    def _string(self, name: str) -> str:
        return str(self.get_parameter(name).value).strip()

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _integer(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _boolean(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def destroy_node(self):
        bridge = self._mqtt_start_bridge
        self._mqtt_start_bridge = None
        if bridge is not None:
            bridge.close()
        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = DepalletizingWorkflowNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                executor.shutdown()
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
