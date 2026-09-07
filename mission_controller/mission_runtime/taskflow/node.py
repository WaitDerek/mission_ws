"""ROS node exposing the platform-facing depalletizing workflow Action."""

from __future__ import annotations

import threading
import time
import math
from typing import Optional

import rclpy
from mission_interfaces.action import (
    ExecuteBoxGrasp,
    ExecuteBoxPlace,
    ExecuteDragBoxGrasp,
    ExecuteFixedBoxWorkflow,
    ExecuteObservationNavigation,
    ExecuteWorkflow,
    NavigateToPoint,
    PlaceBoxTest,
)
from mission_interfaces.srv import AcquireMissionLease, ReleaseMissionLease
from object_pose_interfaces.action import GlobalObservation
from task_interfaces.action import MoveArmJoints
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .identifiers import new_workflow_id
from .lease import WorkflowLeaseManager
from .mqtt_navigation import (
    MqttNavigationGateway,
    NavigationPoint,
    parse_navigation_points_json,
)
from .mqtt_start import MqttStartRequest, MqttWorkflowStartBridge
from .model import NavigationRequest
from .navigation import DisabledNavigationGateway
from .observation_navigation import (
    ObservationNavigationOutcome,
    ObservationNavigationWorkflowEngine,
)
from .ros_operations import ObservationGoalConfig, RosWorkflowOperations
from .state_machine import DepalletizingWorkflowEngine
from .fixed_model import (
    build_fixed_box_tasks,
    build_observed_box_tasks,
    build_smallbox_tasks,
)
from .fixed_operations import FixedRosWorkflowOperations
from .fixed_state_machine import FixedBoxWorkflowEngine
from ..realman_sdk_adapter import RealManSdkAdapter


class DepalletizingWorkflowNode(Node):
    def __init__(self) -> None:
        super().__init__("execute_workflow")
        self._declare_parameters()
        self._validate_parameters()
        self._workflow_lock = threading.Lock()
        self._workflow_reserved = False
        # Keep standalone child-action goals (notably NavigateToPoint) mutually
        # exclusive and compatible with the workflow lease identity rules.
        self.mission_lease_manager = WorkflowLeaseManager()
        self._cancel_event = threading.Event()
        self._active_operations: Optional[RosWorkflowOperations] = None
        self._active_goal_handle = None
        self._mqtt_start_lock = threading.Lock()
        self._mqtt_start_busy = False
        self._mqtt_pending_start = None
        self._mqtt_start_bridge = None
        self._active_navigation_gateway = None
        self._fixed_total_item_count = FixedBoxWorkflowEngine.TOTAL_ITEMS

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
        self._fixed_place_client = ActionClient(
            self,
            PlaceBoxTest,
            self._string("place_box_test_action_name"),
            callback_group=self._client_group,
        )
        self._fixed_arm_joints_client = ActionClient(
            self,
            MoveArmJoints,
            self._string("fixed_workflow_arm_joints_action_name"),
            callback_group=self._client_group,
        )
        # Global-observation posture changes use the same direct Python SDK
        # path as the box motion code, so they do not depend on /move_arm_j.
        self._fixed_sdk_adapter = RealManSdkAdapter(
            sdk_root=self._string("fixed_workflow_arm_joints_sdk_root"),
            left_ip=self._string("fixed_workflow_arm_joints_sdk_left_ip"),
            right_ip=self._string("fixed_workflow_arm_joints_sdk_right_ip"),
            port=self._integer("fixed_workflow_arm_joints_sdk_port"),
            connect_level=self._integer("fixed_workflow_arm_joints_sdk_connect_level"),
            logger=self.get_logger(),
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
        self._observation_navigation_action_server = ActionServer(
            self,
            ExecuteObservationNavigation,
            self._string("observation_navigation_action_name"),
            execute_callback=self._execute_observation_navigation,
            goal_callback=self._observation_navigation_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._server_group,
        )
        self._navigate_to_point_action_server = ActionServer(
            self,
            NavigateToPoint,
            self._string("navigate_to_point_action_name"),
            execute_callback=self._execute_navigate_to_point,
            goal_callback=self._navigate_to_point_goal_callback,
            cancel_callback=self._navigate_to_point_cancel_callback,
            callback_group=self._server_group,
        )
        self._fixed_action_server = None
        if self._boolean("fixed_workflow_enabled"):
            self._fixed_action_server = ActionServer(
                self,
                ExecuteFixedBoxWorkflow,
                self._string("fixed_workflow_action_name"),
                execute_callback=self._execute_fixed_box_workflow,
                goal_callback=self._fixed_goal_callback,
                cancel_callback=self._fixed_cancel_callback,
                callback_group=self._server_group,
            )
        self._smallbox_action_server = None
        if self._boolean("smallbox_workflow_enabled"):
            self._smallbox_action_server = ActionServer(
                self,
                ExecuteFixedBoxWorkflow,
                self._string("smallbox_workflow_action_name"),
                execute_callback=self._execute_smallbox_workflow,
                goal_callback=self._smallbox_goal_callback,
                cancel_callback=self._fixed_cancel_callback,
                callback_group=self._server_group,
            )
        self._mqtt_workflow_client = ActionClient(
            self,
            ExecuteWorkflow,
            self._string("workflow_action_name"),
            callback_group=self._client_group,
        )
        self._mqtt_observation_navigation_client = ActionClient(
            self,
            ExecuteObservationNavigation,
            self._string("observation_navigation_action_name"),
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
                (
                    "observation_navigation_action_name",
                    "/execute_observation_navigation",
                ),
                ("navigate_to_point_action_name", "/navigate_to_point"),
                ("global_observation_action_name", "/depalletizing/observe"),
                ("grasp_box_tf_action_name", "/grasp_box_tf"),
                (
                    "execute_drag_box_grasp_tf_action_name",
                    "/execute_drag_box_grasp_tf",
                ),
                ("execute_box_place_action_name", "/execute_box_place"),
                ("place_box_test_action_name", "/place_box_test"),
                ("fixed_workflow_action_name", "/execute_fixed_box_workflow"),
                ("fixed_workflow_enabled", True),
                ("smallbox_workflow_action_name", "/execute_smallbox_workflow"),
                ("smallbox_workflow_enabled", True),
                ("fixed_workflow_target_label", 0),
                ("fixed_workflow_global_observation_point_id", 4),
                (
                    "fixed_workflow_global_observation_pos",
                    [-0.17, -0.59, 1.12],
                ),
                (
                    "fixed_workflow_global_observation_left_joints",
                    [
                        0.9552012463239765,
                        0.20020671849626956,
                        -0.023596851486963336,
                        -1.6503309808082782,
                        -0.23349014733180137,
                        0.05307546255314755,
                        0.9764593566132677,
                    ],
                ),
                (
                    "fixed_workflow_global_observation_left_home_joints",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "fixed_workflow_arm_joints_action_name",
                    "/move_arm_j",
                ),
                ("fixed_workflow_arm_joints_duration", 0.0),
                ("fixed_workflow_arm_joints_speed_percent", 10.0),
                ("fixed_workflow_arm_joints_timeout_sec", 120.0),
                (
                    "fixed_workflow_arm_joints_sdk_root",
                    "/rm_nvme/recordings/code/RM_API2/Demo/RMDemo_Python/RMDemo_SimpleProcess",
                ),
                ("fixed_workflow_arm_joints_sdk_left_ip", "192.168.127.18"),
                ("fixed_workflow_arm_joints_sdk_right_ip", "192.168.127.19"),
                ("fixed_workflow_arm_joints_sdk_port", 8080),
                ("fixed_workflow_arm_joints_sdk_connect_level", 3),
                ("fixed_workflow_place_point_id", 2),
                ("fixed_workflow_place_pos", [-0.81, -0.77, -1.89]),
                ("fixed_workflow_drag_bigbox_point_id", 1),
                (
                    "fixed_workflow_drag_bigbox_pickup_pos",
                    [0.65, -0.05, 1.24],
                ),
                (
                    "fixed_workflow_drag_bigbox_layer4_pickup_pos",
                    [0.65, -0.05, 1.24],
                ),
                ("fixed_workflow_direct_smallbox_point_id", 3),
                (
                    "fixed_workflow_direct_smallbox_pickup_pos",
                    [-0.10, 0.20, 1.23],
                ),
                (
                    "fixed_workflow_direct_smallbox_layer4_pickup_pos",
                    [-0.10, 0.30, 1.23],
                ),
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
                ("mqtt_robot_id", "realman-001"),
                ("mqtt_navigation_robot_id", "realman-001"),
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
        for name in (
            "workflow_action_name",
            "observation_navigation_action_name",
            "navigate_to_point_action_name",
            "global_observation_action_name",
            "grasp_box_tf_action_name",
            "execute_drag_box_grasp_tf_action_name",
            "execute_box_place_action_name",
            "place_box_test_action_name",
            "fixed_workflow_action_name",
            "smallbox_workflow_action_name",
        ):
            if not self._string(name):
                raise ValueError(f"{name} must not be empty")
        action_names = tuple(
            self._string(name)
            for name in (
                "workflow_action_name",
                "observation_navigation_action_name",
                "navigate_to_point_action_name",
                "fixed_workflow_action_name",
                "smallbox_workflow_action_name",
            )
        )
        if len(set(action_names)) != len(action_names):
            raise ValueError("top-level workflow Action names must be unique")
        for name in (
            "fixed_workflow_place_pos",
            "fixed_workflow_global_observation_pos",
            "fixed_workflow_drag_bigbox_pickup_pos",
            "fixed_workflow_drag_bigbox_layer4_pickup_pos",
            "fixed_workflow_direct_smallbox_pickup_pos",
            "fixed_workflow_direct_smallbox_layer4_pickup_pos",
        ):
            self._parse_custom_navigation_pos(self._float_array(name))
        for name in (
            "fixed_workflow_global_observation_left_joints",
            "fixed_workflow_global_observation_left_home_joints",
        ):
            values = self._float_array(name)
            if len(values) != 7:
                raise ValueError(f"{name} must contain exactly 7 joint values")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain finite joint values")
        if self._float("fixed_workflow_arm_joints_duration") < 0.0:
            raise ValueError("fixed_workflow_arm_joints_duration cannot be negative")
        if not 1.0 <= self._float("fixed_workflow_arm_joints_speed_percent") <= 100.0:
            raise ValueError("fixed_workflow_arm_joints_speed_percent must be in [1,100]")
        if self._float("fixed_workflow_arm_joints_timeout_sec") <= 0.0:
            raise ValueError("fixed_workflow_arm_joints_timeout_sec must be positive")
        for name in (
            "fixed_workflow_arm_joints_sdk_root",
            "fixed_workflow_arm_joints_sdk_left_ip",
            "fixed_workflow_arm_joints_sdk_right_ip",
        ):
            if not self._string(name).strip():
                raise ValueError(f"{name} must not be empty")
        if self._integer("fixed_workflow_arm_joints_sdk_port") <= 0:
            raise ValueError("fixed_workflow_arm_joints_sdk_port must be positive")
        if self._integer("fixed_workflow_arm_joints_sdk_connect_level") < 0:
            raise ValueError("fixed_workflow_arm_joints_sdk_connect_level must be nonnegative")
        for name in (
            "fixed_workflow_place_point_id",
            "fixed_workflow_global_observation_point_id",
            "fixed_workflow_drag_bigbox_point_id",
            "fixed_workflow_direct_smallbox_point_id",
        ):
            if not 1 <= self._integer(name) <= 16:
                raise ValueError(f"{name} must be in [1,16]")
        if self._integer("fixed_workflow_target_label") < 0:
            raise ValueError("fixed_workflow_target_label must be nonnegative")
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
            for name in ("mqtt_host", "mqtt_robot_id"):
                if not self._string(name):
                    raise ValueError(f"{name} must not be empty")
        if self._string("navigation_adapter") == "mqtt":
            if not self._string("mqtt_navigation_robot_id"):
                raise ValueError("mqtt_navigation_robot_id must not be empty")
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
            robot_id=str(request.robot_id).strip(),
            workflow=str(request.workflow).strip().lower() or "full",
            observation_point_id=int(request.observation_point_id),
        )
        if normalized.workflow not in {"full", "observation_navigation"}:
            self._publish_mqtt_workflow_status(
                "rejected",
                normalized.request_id,
                message=f"unsupported workflow {normalized.workflow!r}",
            )
            return
        if (
            normalized.workflow == "observation_navigation"
            and not 1 <= normalized.observation_point_id <= 4
        ):
            self._publish_mqtt_workflow_status(
                "rejected",
                normalized.request_id,
                workflow=normalized.workflow,
                message="observation_point_id must be in [1,4]",
            )
            return
        expected_robot_id = self._string("mqtt_robot_id")
        if normalized.robot_id != expected_robot_id:
            bridge = self._mqtt_start_bridge
            if bridge is not None:
                bridge.publish_status(
                    {
                        "event": "rejected",
                        "robot_id": normalized.robot_id,
                        "request_id": normalized.request_id,
                        "message": (
                            f"workflow MQTT robot_id must be {expected_robot_id}"
                        ),
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
            workflow=normalized.workflow,
            observation_point_id=(
                normalized.observation_point_id
                if normalized.workflow == "observation_navigation"
                else 0
            ),
            message=f"MQTT {normalized.workflow} start request queued",
        )

    def _dispatch_mqtt_start(self) -> None:
        with self._mqtt_start_lock:
            pending = self._mqtt_pending_start
        if pending is None:
            return
        request, deadline = pending
        observation_navigation = request.workflow == "observation_navigation"
        action_client = (
            self._mqtt_observation_navigation_client
            if observation_navigation
            else self._mqtt_workflow_client
        )
        if not action_client.server_is_ready():
            if time.monotonic() < deadline:
                return
            self._finish_mqtt_start(
                "rejected",
                request.request_id,
                workflow=request.workflow,
                message=f"{request.workflow} Action server is unavailable",
            )
            return
        with self._mqtt_start_lock:
            if self._mqtt_pending_start != pending:
                return
            self._mqtt_pending_start = None
        if observation_navigation:
            goal = ExecuteObservationNavigation.Goal()
            goal.request_id = request.request_id
            goal.start = True
            goal.observation_point_id = request.observation_point_id
            feedback_callback = (
                lambda message: self._mqtt_observation_navigation_feedback(
                    request.request_id, message
                )
            )
        else:
            goal = ExecuteWorkflow.Goal()
            goal.start = True
            feedback_callback = lambda message: self._mqtt_workflow_feedback(
                request.request_id, message
            )
        try:
            future = action_client.send_goal_async(
                goal,
                feedback_callback=feedback_callback,
            )
        except Exception as exc:  # noqa: BLE001
            self._finish_mqtt_start(
                "rejected",
                request.request_id,
                message=f"failed to send workflow Action goal: {exc}",
            )
            return
        future.add_done_callback(
            lambda result: self._mqtt_workflow_goal_response(
                request.request_id,
                result,
                workflow=request.workflow,
            )
        )

    def _mqtt_workflow_goal_response(
        self, request_id: str, future, *, workflow: str = "full"
    ) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self._finish_mqtt_start(
                "rejected",
                request_id,
                workflow=workflow,
                message=f"workflow Action goal failed: {exc}",
            )
            return
        if not goal_handle.accepted:
            self._finish_mqtt_start(
                "rejected",
                request_id,
                workflow=workflow,
                message="workflow Action goal was rejected",
            )
            return
        self._publish_mqtt_workflow_status(
            "accepted",
            request_id,
            workflow=workflow,
            message="workflow Action goal accepted",
        )
        result_future = goal_handle.get_result_async()
        if workflow == "observation_navigation":
            result_future.add_done_callback(
                lambda result: self._mqtt_observation_navigation_result(
                    request_id, result
                )
            )
        else:
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

    def _mqtt_observation_navigation_feedback(
        self, request_id: str, wrapped_feedback
    ) -> None:
        feedback = wrapped_feedback.feedback
        self._publish_mqtt_workflow_status(
            "feedback",
            request_id,
            workflow="observation_navigation",
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

    def _mqtt_observation_navigation_result(self, request_id: str, future) -> None:
        try:
            wrapped = future.result()
            result = wrapped.result
        except Exception as exc:  # noqa: BLE001
            self._finish_mqtt_start(
                "result",
                request_id,
                workflow="observation_navigation",
                success=False,
                message=f"observation navigation Action result failed: {exc}",
            )
            return
        self._finish_mqtt_start(
            "result",
            request_id,
            workflow="observation_navigation",
            success=bool(result.success),
            workflow_id=result.workflow_id,
            message=result.message,
            completed_navigation_count=int(result.completed_navigation_count),
            completed_observation_count=int(result.completed_observation_count),
            planned_box_count=int(result.planned_box_count),
            selected_operation_point_id=int(result.selected_operation_point_id),
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
            {
                "event": event,
                "robot_id": self._string("mqtt_robot_id"),
                "request_id": request_id,
                **values,
            }
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

    def _reserve_goal(self, mission: str, request_id: object) -> GoalResponse:
        reservation = self.mission_lease_manager.reserve_goal(mission, request_id)
        if not reservation.accepted:
            self.get_logger().warning(
                f"rejecting {mission} goal: {reservation.message}; "
                f"request_id={reservation.sanitized_request_id}"
            )
            return GoalResponse.REJECT
        self.get_logger().info(
            f"accepted {mission} goal "
            f"request_id={reservation.sanitized_request_id}"
        )
        return GoalResponse.ACCEPT

    def _observation_navigation_goal_callback(self, request) -> GoalResponse:
        if not request.start:
            self.get_logger().warning(
                "rejecting observation navigation goal: start must be true"
            )
            return GoalResponse.REJECT
        if not 1 <= int(request.observation_point_id) <= 4:
            self.get_logger().warning(
                "rejecting observation navigation goal: "
                "observation_point_id must be in [1,4]"
            )
            return GoalResponse.REJECT
        with self._workflow_lock:
            if self._workflow_reserved:
                self.get_logger().warning(
                    "rejecting observation navigation goal: another workflow is active"
                )
                return GoalResponse.REJECT
            self._workflow_reserved = True
            self._cancel_event.clear()
        return GoalResponse.ACCEPT

    def _navigate_to_point_goal_callback(self, request) -> GoalResponse:
        if not request.start:
            self.get_logger().warning(
                "rejecting navigate_to_point goal: start must be true"
            )
            return GoalResponse.REJECT
        if request.point_id < 1 or request.point_id > 16:
            self.get_logger().warning(
                "rejecting navigate_to_point goal: point_id must be in [1, 16]"
            )
            return GoalResponse.REJECT
        if request.use_custom_pos:
            try:
                self._parse_custom_navigation_pos(request.pos)
            except ValueError as exc:
                self.get_logger().warning(
                    f"rejecting navigate_to_point goal: {exc}"
                )
                return GoalResponse.REJECT
        return self._reserve_goal("navigate_to_point", request.request_id)

    def _navigate_to_point_cancel_callback(self, _goal_handle) -> CancelResponse:
        gateway = self._active_navigation_gateway
        if gateway is not None:
            gateway.cancel_active()
        return CancelResponse.ACCEPT

    def _fixed_goal_callback(self, request) -> GoalResponse:
        if not request.start:
            self.get_logger().warning(
                "rejecting fixed box workflow goal: start must be true"
            )
            return GoalResponse.REJECT
        if not 1 <= int(request.start_item_index) <= FixedBoxWorkflowEngine.TOTAL_ITEMS:
            self.get_logger().warning(
                "rejecting fixed box workflow goal: start_item_index must be in [1,8]"
            )
            return GoalResponse.REJECT
        stop = int(request.stop_after_item_index)
        if stop and not int(request.start_item_index) <= stop <= FixedBoxWorkflowEngine.TOTAL_ITEMS:
            self.get_logger().warning(
                "rejecting fixed box workflow goal: stop_after_item_index must "
                "be 0 or in [start_item_index,8]"
            )
            return GoalResponse.REJECT
        with self._workflow_lock:
            if self._workflow_reserved:
                self.get_logger().warning(
                    "rejecting fixed box workflow goal: another workflow is active"
                )
                return GoalResponse.REJECT
            self._workflow_reserved = True
            self._cancel_event.clear()
        return GoalResponse.ACCEPT

    def _fixed_cancel_callback(self, _goal_handle) -> CancelResponse:
        self._cancel_event.set()
        with self._workflow_lock:
            operations = self._active_operations
        if operations is not None:
            operations.cancel_active()
        return CancelResponse.ACCEPT

    def _smallbox_goal_callback(self, request) -> GoalResponse:
        if not request.start:
            self.get_logger().warning(
                "rejecting smallbox workflow goal: start must be true"
            )
            return GoalResponse.REJECT
        start = int(request.start_item_index)
        stop = int(request.stop_after_item_index)
        if not 1 <= start <= 4:
            self.get_logger().warning(
                "rejecting smallbox workflow goal: start_item_index must be in [1,4]"
            )
            return GoalResponse.REJECT
        if stop and not start <= stop <= 4:
            self.get_logger().warning(
                "rejecting smallbox workflow goal: stop_after_item_index must "
                "be 0 or in [start_item_index,4]"
            )
            return GoalResponse.REJECT
        with self._workflow_lock:
            if self._workflow_reserved:
                self.get_logger().warning(
                    "rejecting smallbox workflow goal: another workflow is active"
                )
                return GoalResponse.REJECT
            self._workflow_reserved = True
            self._cancel_event.clear()
        return GoalResponse.ACCEPT

    def _execute_fixed_box_workflow(self, goal_handle):
        request = goal_handle.request
        workflow_id = new_workflow_id()
        lease_token = ""
        operations = None
        outcome = None
        release_error = ""
        with self._workflow_lock:
            self._active_goal_handle = goal_handle
        try:
            self._publish_fixed_progress(
                goal_handle,
                workflow_id,
                "INITIALIZING",
                None,
                "validating the fixed workflow request",
            )
            acquired = self._acquire_lease(workflow_id)
            if not acquired.success:
                outcome = self._fixed_failure_outcome(
                    workflow_id, "ACQUIRE_LEASE", acquired.message
                )
            else:
                lease_token = acquired.lease_token
                operations = self._make_fixed_operations(
                    goal_handle, workflow_id, bool(request.dry_run)
                )
                with self._workflow_lock:
                    self._active_operations = operations
                if bool(request.use_global_observation):
                    if bool(request.dry_run):
                        outcome = self._fixed_failure_outcome(
                            workflow_id,
                            "GLOBAL_OBSERVATION",
                            "use_global_observation requires dry_run=false",
                        )
                    else:
                        observation_point = str(
                            self._integer(
                                "fixed_workflow_global_observation_point_id"
                            )
                        )
                        observation_pos = tuple(
                            self._float_array("fixed_workflow_global_observation_pos")
                        )
                        self._publish_fixed_progress(
                            goal_handle,
                            workflow_id,
                            "GLOBAL_OBSERVATION_NAVIGATION",
                            None,
                            "requesting navigation to global observation point "
                            f"{observation_point} with pos={list(observation_pos)}",
                        )
                        navigation = operations.navigate(
                            NavigationRequest(
                                workflow_id=workflow_id,
                                step_id=f"{workflow_id}:global-observation",
                                point_id=observation_point,
                                pos=observation_pos,
                            )
                        )
                        if not navigation.success:
                            outcome = self._fixed_failure_outcome(
                                workflow_id,
                                "GLOBAL_OBSERVATION_NAVIGATION",
                                navigation.message,
                            )
                        else:
                            left_observation_joints = self._float_array(
                                "fixed_workflow_global_observation_left_joints"
                            )
                            self._publish_fixed_progress(
                                goal_handle,
                                workflow_id,
                                "GLOBAL_OBSERVATION_LEFT_ARM_MOVEJ",
                                None,
                                "moving left arm to global observation joint target "
                                f"{left_observation_joints}",
                            )
                            arm_move = operations.move_left_joints(
                                left_observation_joints,
                                "global observation left-arm MoveJ",
                            )
                            if arm_move.success:
                                self._publish_fixed_progress(
                                    goal_handle,
                                    workflow_id,
                                    "GLOBAL_OBSERVATION",
                                    None,
                                    "calling Vision GlobalObservation Action",
                                )
                                observation = operations.observe(observation_point)
                                self._publish_fixed_progress(
                                    goal_handle,
                                    workflow_id,
                                    "GLOBAL_OBSERVATION_LEFT_ARM_HOME",
                                    None,
                                    "returning left arm to zero joints after global observation",
                                )
                                home_move = operations.move_left_joints(
                                    self._float_array(
                                        "fixed_workflow_global_observation_left_home_joints"
                                    ),
                                    "global observation left-arm zero reset",
                                )
                            else:
                                observation = None
                                home_move = None
                            if not arm_move.success:
                                outcome = self._fixed_failure_outcome(
                                    workflow_id,
                                    "GLOBAL_OBSERVATION_LEFT_ARM_MOVEJ",
                                    arm_move.message,
                                )
                            elif home_move is not None and not home_move.success:
                                outcome = self._fixed_failure_outcome(
                                    workflow_id,
                                    "GLOBAL_OBSERVATION_LEFT_ARM_HOME",
                                    home_move.message,
                                )
                            elif (
                                not observation.success
                                or observation.plan is None
                                or not observation.plan.actionable
                            ):
                                outcome = self._fixed_failure_outcome(
                                    workflow_id,
                                    "GLOBAL_OBSERVATION",
                                    observation.message
                                    or "Vision returned no actionable plan",
                                )
                            else:
                                try:
                                    tasks = build_observed_box_tasks(
                                        observation.plan,
                                        drag_point_id=self._integer(
                                            "fixed_workflow_drag_bigbox_point_id"
                                        ),
                                        drag_pickup_pos=self._float_array(
                                            "fixed_workflow_drag_bigbox_pickup_pos"
                                        ),
                                        drag_layer4_pickup_pos=self._float_array(
                                            "fixed_workflow_drag_bigbox_layer4_pickup_pos"
                                        ),
                                        direct_point_id=self._integer(
                                            "fixed_workflow_direct_smallbox_point_id"
                                        ),
                                        direct_pickup_pos=self._float_array(
                                            "fixed_workflow_direct_smallbox_pickup_pos"
                                        ),
                                        direct_layer4_pickup_pos=self._float_array(
                                            "fixed_workflow_direct_smallbox_layer4_pickup_pos"
                                        ),
                                        drag_action_name=self._string(
                                            "execute_drag_box_grasp_tf_action_name"
                                        ),
                                        direct_action_name=self._string(
                                            "grasp_box_tf_action_name"
                                        ),
                                    )
                                except (TypeError, ValueError) as exc:
                                    outcome = self._fixed_failure_outcome(
                                        workflow_id,
                                        "GLOBAL_OBSERVATION",
                                        f"invalid Vision task plan: {exc}",
                                    )
                                else:
                                    if len(tasks) > FixedBoxWorkflowEngine.TOTAL_ITEMS:
                                        outcome = self._fixed_failure_outcome(
                                            workflow_id,
                                            "GLOBAL_OBSERVATION",
                                            "Vision plan contains more than 8 items",
                                        )
                                    else:
                                        self._publish_fixed_progress(
                                            goal_handle,
                                            workflow_id,
                                            "GLOBAL_OBSERVATION_PLAN_READY",
                                            None,
                                            "Vision plan accepted: "
                                            f"{len(tasks)} items; "
                                            + ", ".join(
                                                f"{task.box_type}/L{task.box_layer}/"
                                                f"{'drag' if task.grasp_action == self._string('execute_drag_box_grasp_tf_action_name') else 'direct'}"
                                                for task in tasks
                                            ),
                                        )
                else:
                    tasks = build_fixed_box_tasks(
                        drag_point_id=self._integer("fixed_workflow_drag_bigbox_point_id"),
                        drag_pickup_pos=self._float_array(
                            "fixed_workflow_drag_bigbox_pickup_pos"
                        ),
                        drag_layer4_pickup_pos=self._float_array(
                            "fixed_workflow_drag_bigbox_layer4_pickup_pos"
                        ),
                        direct_point_id=self._integer(
                            "fixed_workflow_direct_smallbox_point_id"
                        ),
                        direct_pickup_pos=self._float_array(
                            "fixed_workflow_direct_smallbox_pickup_pos"
                        ),
                        direct_layer4_pickup_pos=self._float_array(
                            "fixed_workflow_direct_smallbox_layer4_pickup_pos"
                        ),
                        drag_action_name=self._string(
                            "execute_drag_box_grasp_tf_action_name"
                        ),
                        direct_action_name=self._string("grasp_box_tf_action_name"),
                    )
                if outcome is None:
                    self._fixed_total_item_count = len(tasks)
                    engine = FixedBoxWorkflowEngine(
                        operations,
                        tasks,
                        place_point_id=self._integer("fixed_workflow_place_point_id"),
                        place_pos=tuple(
                            self._float_array("fixed_workflow_place_pos")
                        ),
                        progress_callback=lambda stage, task, detail: self._publish_fixed_progress(
                            goal_handle, workflow_id, stage, task, detail
                        ),
                    )
                    outcome = engine.run(
                        workflow_id,
                        lease_token,
                        start_item_index=int(request.start_item_index),
                        stop_after_item_index=int(request.stop_after_item_index),
                    )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"unexpected fixed box workflow error workflow_id={workflow_id}: {exc}"
            )
            outcome = self._fixed_failure_outcome(
                workflow_id, "INTERNAL_ERROR", f"unexpected workflow error: {exc}"
            )
        finally:
            with self._workflow_lock:
                active_operations = self._active_operations
                self._active_operations = None
            if self._cancel_event.is_set() and active_operations is not None:
                active_operations.cancel_active()
            if active_operations is not None:
                close_operations = getattr(active_operations, "close", None)
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
            outcome = self._fixed_failure_outcome(
                workflow_id, "INTERNAL_ERROR", "fixed workflow returned no outcome"
            )
        if release_error:
            outcome = self._fixed_failure_outcome(
                workflow_id,
                "RELEASE_LEASE",
                f"workflow stopped but Mission lease release failed: {release_error}",
                completed=outcome.completed_box_count,
                last_completed=outcome.last_completed_item_index,
            )
        result = ExecuteFixedBoxWorkflow.Result()
        result.success = bool(outcome.success)
        result.workflow_id = workflow_id
        result.completed_box_count = int(outcome.completed_box_count)
        result.last_completed_item_index = int(outcome.last_completed_item_index)
        result.final_stage = outcome.final_stage
        result.message = outcome.message
        if self._cancel_event.is_set() or outcome.final_stage == "CANCELED":
            goal_handle.canceled()
        elif outcome.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _execute_smallbox_workflow(self, goal_handle):
        request = goal_handle.request
        workflow_id = new_workflow_id()
        lease_token = ""
        operations = None
        outcome = None
        release_error = ""
        with self._workflow_lock:
            self._active_goal_handle = goal_handle
        try:
            self._publish_smallbox_progress(
                goal_handle,
                workflow_id,
                "INITIALIZING",
                None,
                "validating the four-layer smallbox workflow",
            )
            acquired = self._acquire_lease(workflow_id)
            if not acquired.success:
                outcome = self._fixed_failure_outcome(
                    workflow_id, "ACQUIRE_LEASE", acquired.message
                )
            else:
                lease_token = acquired.lease_token
                operations = self._make_smallbox_operations(
                    goal_handle, workflow_id, bool(request.dry_run)
                )
                with self._workflow_lock:
                    self._active_operations = operations
                tasks = build_smallbox_tasks(
                    direct_point_id=self._integer(
                        "fixed_workflow_direct_smallbox_point_id"
                    ),
                    direct_pickup_pos=self._float_array(
                        "fixed_workflow_direct_smallbox_pickup_pos"
                    ),
                    direct_layer4_pickup_pos=self._float_array(
                        "fixed_workflow_direct_smallbox_layer4_pickup_pos"
                    ),
                    direct_action_name=self._string("grasp_box_tf_action_name"),
                )
                engine = FixedBoxWorkflowEngine(
                    operations,
                    tasks,
                    place_point_id=self._integer("fixed_workflow_place_point_id"),
                    place_pos=tuple(self._float_array("fixed_workflow_place_pos")),
                    progress_callback=lambda stage, task, detail: self._publish_smallbox_progress(
                        goal_handle, workflow_id, stage, task, detail
                    ),
                )
                outcome = engine.run(
                    workflow_id,
                    lease_token,
                    start_item_index=int(request.start_item_index),
                    stop_after_item_index=int(request.stop_after_item_index),
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"unexpected smallbox workflow error workflow_id={workflow_id}: {exc}"
            )
            outcome = self._fixed_failure_outcome(
                workflow_id, "INTERNAL_ERROR", f"unexpected workflow error: {exc}"
            )
        finally:
            with self._workflow_lock:
                active_operations = self._active_operations
                self._active_operations = None
            if self._cancel_event.is_set() and active_operations is not None:
                active_operations.cancel_active()
            if active_operations is not None:
                close_operations = getattr(active_operations, "close", None)
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
            outcome = self._fixed_failure_outcome(
                workflow_id, "INTERNAL_ERROR", "smallbox workflow returned no outcome"
            )
        if release_error:
            outcome = self._fixed_failure_outcome(
                workflow_id,
                "RELEASE_LEASE",
                f"workflow stopped but Mission lease release failed: {release_error}",
                completed=outcome.completed_box_count,
                last_completed=outcome.last_completed_item_index,
            )
        result = ExecuteFixedBoxWorkflow.Result()
        result.success = bool(outcome.success)
        result.workflow_id = workflow_id
        result.completed_box_count = int(outcome.completed_box_count)
        result.last_completed_item_index = int(outcome.last_completed_item_index)
        result.final_stage = outcome.final_stage
        result.message = outcome.message
        if self._cancel_event.is_set() or outcome.final_stage == "CANCELED":
            goal_handle.canceled()
        elif outcome.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _make_fixed_operations(self, goal_handle, workflow_id: str, dry_run: bool):
        return FixedRosWorkflowOperations(
            node=self,
            navigation_gateway=self._make_navigation_gateway(),
            direct_grasp_client=self._direct_grasp_client,
            drag_grasp_client=self._drag_grasp_client,
            place_test_client=self._fixed_place_client,
            observation_client=self._observation_client,
            observation_goal=self._global_observation_goal_config(),
            arm_joints_client=self._fixed_arm_joints_client,
            arm_joints_action_name=self._string(
                "fixed_workflow_arm_joints_action_name"
            ),
            arm_joints_duration=self._float(
                "fixed_workflow_arm_joints_duration"
            ),
            sdk_adapter=self._fixed_sdk_adapter,
            arm_joints_speed_percent=self._float(
                "fixed_workflow_arm_joints_speed_percent"
            ),
            arm_joints_timeout_sec=self._float(
                "fixed_workflow_arm_joints_timeout_sec"
            ),
            cancel_event=self._cancel_event,
            direct_action_name=self._string("grasp_box_tf_action_name"),
            drag_action_name=self._string(
                "execute_drag_box_grasp_tf_action_name"
            ),
            target_label=self._integer("fixed_workflow_target_label"),
            dry_run=dry_run,
            server_wait_timeout_sec=self._float("server_wait_timeout_sec"),
            result_timeout_sec=self._float("child_result_timeout_sec"),
            child_feedback_callback=lambda stage, detail: self._publish_fixed_child_feedback(
                goal_handle, workflow_id, stage, detail
            ),
        )

    def _global_observation_goal_config(self) -> ObservationGoalConfig:
        return ObservationGoalConfig(
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
            # Absolute camera-depth gating is intentionally disabled. The
            # relative front-row depth-spread check remains active.
            front_max_camera_depth_m=0.0,
        )

    def _make_smallbox_operations(self, goal_handle, workflow_id: str, dry_run: bool):
        return FixedRosWorkflowOperations(
            node=self,
            navigation_gateway=self._make_navigation_gateway(),
            direct_grasp_client=self._direct_grasp_client,
            drag_grasp_client=self._drag_grasp_client,
            place_test_client=self._fixed_place_client,
            observation_client=self._observation_client,
            observation_goal=self._global_observation_goal_config(),
            cancel_event=self._cancel_event,
            direct_action_name=self._string("grasp_box_tf_action_name"),
            drag_action_name=self._string("execute_drag_box_grasp_tf_action_name"),
            target_label=self._integer("fixed_workflow_target_label"),
            dry_run=dry_run,
            server_wait_timeout_sec=self._float("server_wait_timeout_sec"),
            result_timeout_sec=self._float("child_result_timeout_sec"),
            child_feedback_callback=lambda stage, detail: self._publish_smallbox_child_feedback(
                goal_handle, workflow_id, stage, detail
            ),
        )

    def _publish_fixed_progress(
        self, goal_handle, workflow_id: str, stage: str, task, detail: str
    ) -> None:
        item_index = int(getattr(task, "item_index", 0))
        box_type = str(getattr(task, "box_type", ""))
        box_layer = int(getattr(task, "box_layer", 0))
        self.get_logger().info(
            f"fixed workflow transition workflow_id={workflow_id} "
            f"stage={stage} item={item_index}/{self._fixed_total_item_count} "
            f"detail={detail}"
        )
        feedback = ExecuteFixedBoxWorkflow.Feedback()
        feedback.workflow_id = workflow_id
        feedback.stage = str(stage)
        feedback.current_item_index = item_index
        feedback.total_item_count = self._fixed_total_item_count
        feedback.box_type = box_type
        feedback.box_layer = box_layer
        feedback.detail = str(detail)
        goal_handle.publish_feedback(feedback)

    def _publish_fixed_child_feedback(
        self, goal_handle, workflow_id: str, stage: str, detail: str
    ) -> None:
        self._publish_fixed_progress(
            goal_handle, workflow_id, f"CHILD_{stage}", None, detail
        )

    def _publish_smallbox_progress(
        self, goal_handle, workflow_id: str, stage: str, task, detail: str
    ) -> None:
        item_index = int(getattr(task, "item_index", 0))
        box_type = str(getattr(task, "box_type", ""))
        box_layer = int(getattr(task, "box_layer", 0))
        self.get_logger().info(
            f"smallbox workflow transition workflow_id={workflow_id} "
            f"stage={stage} item={item_index}/4 detail={detail}"
        )
        feedback = ExecuteFixedBoxWorkflow.Feedback()
        feedback.workflow_id = workflow_id
        feedback.stage = str(stage)
        feedback.current_item_index = item_index
        feedback.total_item_count = 4
        feedback.box_type = box_type
        feedback.box_layer = box_layer
        feedback.detail = str(detail)
        goal_handle.publish_feedback(feedback)

    def _publish_smallbox_child_feedback(
        self, goal_handle, workflow_id: str, stage: str, detail: str
    ) -> None:
        self._publish_smallbox_progress(
            goal_handle, workflow_id, f"CHILD_{stage}", None, detail
        )

    @staticmethod
    def _fixed_failure_outcome(
        workflow_id: str,
        stage: str,
        message: str,
        *,
        completed: int = 0,
        last_completed: int = 0,
    ):
        from .fixed_model import FixedWorkflowOutcome

        return FixedWorkflowOutcome(
            False,
            workflow_id,
            int(completed),
            int(last_completed),
            stage,
            str(message),
        )

    @staticmethod
    def _navigation_result_message(status: str, message: str) -> str:
        detail = str(message).strip()
        return detail or str(status).strip() or "navigation failed"

    @staticmethod
    def _parse_custom_navigation_pos(values: object) -> tuple[float, float, float]:
        try:
            raw_values = list(values)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "custom navigation pos must be a numeric [x, y, yaw] array"
            ) from exc
        if len(raw_values) != 3:
            raise ValueError(
                "custom navigation pos must contain exactly 3 values [x, y, yaw]"
            )
        if any(isinstance(value, bool) for value in raw_values):
            raise ValueError("custom navigation pos values must be numeric")
        try:
            coordinates = tuple(float(value) for value in raw_values)
        except (TypeError, ValueError) as exc:
            raise ValueError("custom navigation pos values must be numeric") from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("custom navigation pos values must be finite")
        return coordinates  # type: ignore[return-value]

    def _execute_observation_navigation(self, goal_handle):
        request = goal_handle.request
        workflow_id = new_workflow_id()
        lease_token = ""
        operations = None
        outcome = None
        release_error = ""
        with self._workflow_lock:
            self._active_goal_handle = goal_handle
        try:
            acquired = self._acquire_lease(workflow_id)
            if not acquired.success:
                outcome = self._observation_navigation_failure_outcome(
                    workflow_id, "ACQUIRE_LEASE", acquired.message
                )
            else:
                lease_token = acquired.lease_token
                operations = self._make_observation_navigation_operations(
                    goal_handle, workflow_id
                )
                with self._workflow_lock:
                    self._active_operations = operations
                engine = ObservationNavigationWorkflowEngine(
                    operations,
                    place_point_id=str(
                        self._integer("fixed_workflow_place_point_id")
                    ),
                    place_pos=tuple(
                        self._float_array("fixed_workflow_place_pos")
                    ),
                    progress_callback=lambda progress: (
                        self._publish_observation_navigation_progress(
                            goal_handle, progress
                        )
                    ),
                )
                outcome = engine.run(
                    workflow_id,
                    lease_token,
                    int(request.observation_point_id),
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                "unexpected observation navigation workflow error "
                f"workflow_id={workflow_id}: {exc}"
            )
            outcome = self._observation_navigation_failure_outcome(
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
            outcome = self._observation_navigation_failure_outcome(
                workflow_id, "INTERNAL_ERROR", "workflow returned no outcome"
            )
        if release_error:
            outcome = self._observation_navigation_failure_outcome(
                workflow_id,
                "RELEASE_LEASE",
                f"workflow stopped but Mission lease release failed: {release_error}",
                completed_navigation_count=outcome.completed_navigation_count,
                completed_observation_count=outcome.completed_observation_count,
                planned_box_count=outcome.planned_box_count,
                selected_operation_point_id=outcome.selected_operation_point_id,
            )

        result = ExecuteObservationNavigation.Result()
        result.success = bool(outcome.success)
        result.workflow_id = workflow_id
        result.message = outcome.message
        result.completed_navigation_count = int(outcome.completed_navigation_count)
        result.completed_observation_count = int(outcome.completed_observation_count)
        result.planned_box_count = int(outcome.planned_box_count)
        result.selected_operation_point_id = int(
            outcome.selected_operation_point_id or 0
        )
        result.final_stage = outcome.final_stage
        if self._cancel_event.is_set() or outcome.final_stage == "CANCELED":
            goal_handle.canceled()
        elif outcome.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _execute_navigate_to_point(self, goal_handle):
        request = goal_handle.request
        result = NavigateToPoint.Result()
        result.point_id = int(request.point_id)
        gateway = None

        def feedback(stage: str, detail: str) -> None:
            message = NavigateToPoint.Feedback()
            message.stage = stage
            message.detail = detail
            goal_handle.publish_feedback(message)

        try:
            if request.use_custom_pos:
                custom_pos = self._parse_custom_navigation_pos(request.pos)
                target = NavigationPoint(*custom_pos)
                feedback(
                    "VALIDATING",
                    f"using custom navigation pos="
                    f"[{target.x:g},{target.y:g},{target.yaw:g}]",
                )
            else:
                custom_pos = None
                feedback(
                    "VALIDATING",
                    f"loading configured navigation point {request.point_id}",
                )
                points = parse_navigation_points_json(
                    self._string("mqtt_navigation_points_json")
                )
                target = points.get(str(request.point_id))
                if target is None:
                    result.success = False
                    result.message = (
                        f"navigation point {request.point_id} has no configured coordinates"
                    )
                    result.pos = []
                    feedback("FAILED", result.message)
                    goal_handle.abort()
                    return result

            result.pos = [target.x, target.y, target.yaw]
            if request.dry_run:
                result.success = True
                result.message = (
                    f"dry-run navigation point {request.point_id}: "
                    f"pos=[{target.x:g},{target.y:g},{target.yaw:g}]"
                )
                feedback("DRY_RUN_COMPLETE", result.message)
                goal_handle.succeed()
                return result

            if self._string("navigation_adapter").strip().lower() != "mqtt":
                result.success = False
                result.message = "navigation_adapter must be mqtt for physical navigation"
                feedback("FAILED", result.message)
                goal_handle.abort()
                return result

            feedback("CONNECTING", "connecting to MQTT navigation broker")
            gateway = self._make_navigation_gateway()
            self._active_navigation_gateway = gateway
            feedback(
                "WAITING_FOR_NAVIGATION",
                f"requesting point {request.point_id} with pos="
                f"[{target.x:g},{target.y:g},{target.yaw:g}]",
            )
            navigation_result = gateway.navigate(
                NavigationRequest(
                    workflow_id="standalone-navigation",
                    step_id=str(request.request_id).strip() or "navigate-to-point",
                    point_id=str(request.point_id),
                    pos=custom_pos,
                ),
                lambda: bool(goal_handle.is_cancel_requested),
            )
            result.success = bool(navigation_result.success)
            result.message = self._navigation_result_message(
                navigation_result.status,
                navigation_result.message,
            )
            if goal_handle.is_cancel_requested or navigation_result.status == "canceled":
                feedback("CANCELED", result.message)
                goal_handle.canceled()
            elif result.success:
                feedback("SUCCEEDED", result.message)
                goal_handle.succeed()
            else:
                feedback("FAILED", result.message)
                goal_handle.abort()
            return result
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.message = f"unexpected standalone navigation error: {exc}"
            feedback("FAILED", result.message)
            goal_handle.abort()
            return result
        finally:
            self._active_navigation_gateway = None
            if gateway is not None:
                gateway.close()
            self.mission_lease_manager.release_goal()

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
        return self._new_ros_workflow_operations(
            child_feedback_callback=lambda stage, detail: self._publish_child_feedback(
                goal_handle, workflow_id=workflow_id, stage=stage, detail=detail
            )
        )

    def _make_observation_navigation_operations(
        self, goal_handle, workflow_id: str
    ) -> RosWorkflowOperations:
        return self._new_ros_workflow_operations(
            child_feedback_callback=lambda stage, detail: (
                self._publish_observation_navigation_child_feedback(
                    goal_handle,
                    workflow_id=workflow_id,
                    stage=stage,
                    detail=detail,
                )
            )
        )

    def _new_ros_workflow_operations(
        self, *, child_feedback_callback
    ) -> RosWorkflowOperations:
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
                # Absolute camera-depth gating is intentionally disabled.
                front_max_camera_depth_m=0.0,
            ),
            target_label=self._integer("target_label"),
            dry_run=self._boolean("dry_run"),
            server_wait_timeout_sec=self._float("server_wait_timeout_sec"),
            result_timeout_sec=self._float("child_result_timeout_sec"),
            child_feedback_callback=child_feedback_callback,
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
            robot_id=self._string("mqtt_navigation_robot_id"),
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

    def _publish_observation_navigation_progress(
        self, goal_handle, progress
    ) -> None:
        self.get_logger().info(
            "observation navigation transition "
            f"workflow_id={progress.workflow_id} stage={progress.stage} "
            f"point_id={progress.current_point_id or '<none>'} "
            f"stack_id={progress.current_stack_id or '<none>'} "
            f"order_index={progress.current_order_index}/"
            f"{progress.total_order_items} detail={progress.detail}"
        )
        feedback = ExecuteObservationNavigation.Feedback()
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

    def _publish_observation_navigation_child_feedback(
        self, goal_handle, workflow_id: str, stage: str, detail: str
    ) -> None:
        feedback = ExecuteObservationNavigation.Feedback()
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
    def _observation_navigation_failure_outcome(
        workflow_id: str,
        stage: str,
        message: str,
        *,
        completed_navigation_count: int = 0,
        completed_observation_count: int = 0,
        planned_box_count: int = 0,
        selected_operation_point_id: str = "",
    ) -> ObservationNavigationOutcome:
        return ObservationNavigationOutcome(
            success=False,
            workflow_id=workflow_id,
            message=message,
            final_stage=stage,
            completed_navigation_count=completed_navigation_count,
            completed_observation_count=completed_observation_count,
            planned_box_count=planned_box_count,
            selected_operation_point_id=selected_operation_point_id,
        )

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

    def _float_array(self, name: str) -> list[float]:
        return [float(value) for value in list(self.get_parameter(name).value)]

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
