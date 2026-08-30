"""ROS node exposing the assembly workflow and its MQTT trigger bridge."""

from __future__ import annotations

import threading
import uuid

import rclpy
from mission_interfaces.action import (
    ExecuteAssembly,
    ExecuteGrip,
    ExecutePeel,
    ExecuteWorkflow,
)
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .model import WorkflowProgress
from .mqtt_navigation import (
    MqttNavigationGateway,
    parse_navigation_points_json,
    require_all_navigation_points,
)
from .mqtt_trigger import WorkflowMqttTriggerMixin
from .operations import RosWorkflowOperations
from .state_machine import AssemblyWorkflowEngine


class WorkflowNode(WorkflowMqttTriggerMixin, Node):
    """Run one 1 -> 3 -> 2 -> 3 workflow at a time."""

    def __init__(self) -> None:
        super().__init__("execute_workflow")
        self._declare_parameters()
        self._validate_parameters()
        self._server_group = ReentrantCallbackGroup()
        self._client_group = ReentrantCallbackGroup()
        self._workflow_lock = threading.Lock()
        self._workflow_reserved = False
        self._cancel_event = threading.Event()
        self._active_operations = None

        self._grip_client = ActionClient(
            self,
            ExecuteGrip,
            self._string("execute_grip_action_name"),
            callback_group=self._client_group,
        )
        self._peel_client = ActionClient(
            self,
            ExecutePeel,
            self._string("execute_peel_action_name"),
            callback_group=self._client_group,
        )
        self._assembly_client = ActionClient(
            self,
            ExecuteAssembly,
            self._string("execute_assembly_action_name"),
            callback_group=self._client_group,
        )
        self._navigation_gateway = self._create_navigation_gateway()
        self._action_server = ActionServer(
            self,
            ExecuteWorkflow,
            self._string("workflow_action_name"),
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._server_group,
        )

        self._initialize_mqtt_trigger()
        self.get_logger().info(
            "workflow ready: action=%s sequence=1->3->2->3"
            % self._string("workflow_action_name")
        )

    def _declare_parameters(self) -> None:
        self.declare_parameters(
            namespace="",
            parameters=[
                ("workflow_action_name", "/execute_workflow"),
                ("execute_grip_action_name", "/execute_grip"),
                ("execute_peel_action_name", "/execute_peel"),
                ("execute_assembly_action_name", "/execute_assembly"),
                ("connector_point_id", "1"),
                ("badge_point_id", "2"),
                ("assembly_point_id", "3"),
                ("robot_id", "g1d"),
                ("navigation_adapter", "mqtt"),
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
                ("mqtt_start_enabled", True),
                ("mqtt_start_topic", "mission/workflow/start"),
                ("mqtt_status_topic", "mission/workflow/status"),
                ("mqtt_trigger_client_id", ""),
                ("mqtt_start_action_wait_timeout_sec", 5.0),
                ("server_wait_timeout_sec", 10.0),
                ("child_result_timeout_sec", 1200.0),
            ],
        )

    def _validate_parameters(self) -> None:
        for name in (
            "workflow_action_name",
            "execute_grip_action_name",
            "execute_peel_action_name",
            "execute_assembly_action_name",
            "robot_id",
            "mqtt_host",
            "mqtt_request_topic",
            "mqtt_result_topic",
            "mqtt_navigation_frame_id",
            "mqtt_start_topic",
            "mqtt_status_topic",
        ):
            if not self._string(name):
                raise ValueError(f"{name} must not be empty")
        if self._string("navigation_adapter") != "mqtt":
            raise ValueError("navigation_adapter must be mqtt")
        if not 1 <= self._integer("mqtt_port") <= 65535:
            raise ValueError("mqtt_port must be in [1, 65535]")
        if self._integer("mqtt_qos") not in (0, 1, 2):
            raise ValueError("mqtt_qos must be 0, 1, or 2")
        if self._integer("mqtt_keepalive_sec") <= 0:
            raise ValueError("mqtt_keepalive_sec must be positive")
        for name in (
            "mqtt_connect_timeout_sec",
            "mqtt_navigation_timeout_sec",
            "mqtt_start_action_wait_timeout_sec",
            "server_wait_timeout_sec",
            "child_result_timeout_sec",
        ):
            if self._float(name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        topics = (
            self._string("mqtt_request_topic"),
            self._string("mqtt_result_topic"),
            self._string("mqtt_start_topic"),
            self._string("mqtt_status_topic"),
        )
        if len(set(topics)) != len(topics):
            raise ValueError("MQTT workflow and navigation topics must differ")
        navigation_points = parse_navigation_points_json(
            self._string("mqtt_navigation_points_json")
        )
        require_all_navigation_points(navigation_points)
        workflow_points = {
            self._string("connector_point_id"),
            self._string("badge_point_id"),
            self._string("assembly_point_id"),
        }
        if len(workflow_points) != 3 or not workflow_points.issubset(
            {"1", "2", "3", "4"}
        ):
            raise ValueError(
                "connector, badge, and assembly points must be distinct ids in 1..4"
            )

    def _create_navigation_gateway(self):
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

    def _string(self, name: str) -> str:
        return str(self.get_parameter(name).value).strip()

    def _integer(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _boolean(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _goal_callback(self, request) -> GoalResponse:
        if not request.start:
            return GoalResponse.REJECT
        with self._workflow_lock:
            if self._workflow_reserved:
                return GoalResponse.REJECT
            self._workflow_reserved = True
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        self._cancel_event.set()
        operations = self._active_operations
        if operations is not None:
            operations.cancel_active()
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle) -> ExecuteWorkflow.Result:
        workflow_id = f"workflow-{uuid.uuid4().hex[:12]}"
        self._cancel_event.clear()
        operations = RosWorkflowOperations(
            node=self,
            navigation_gateway=self._navigation_gateway,
            grip_client=self._grip_client,
            peel_client=self._peel_client,
            assembly_client=self._assembly_client,
            cancel_event=self._cancel_event,
            server_wait_timeout_sec=self._float("server_wait_timeout_sec"),
            result_timeout_sec=self._float("child_result_timeout_sec"),
        )
        self._active_operations = operations
        try:
            engine = AssemblyWorkflowEngine(
                operations,
                progress_callback=lambda progress: self._publish_feedback(
                    goal_handle, progress
                ),
                connector_point_id=self._string("connector_point_id"),
                badge_point_id=self._string("badge_point_id"),
                assembly_point_id=self._string("assembly_point_id"),
            )
            outcome = engine.run(workflow_id)
            result = ExecuteWorkflow.Result()
            result.success = outcome.success
            result.workflow_id = outcome.workflow_id
            result.message = outcome.message
            result.completed_task_count = outcome.completed_task_count
            result.final_stage = outcome.final_stage
            if outcome.final_stage == "CANCELED":
                goal_handle.canceled()
            elif outcome.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result
        finally:
            self._active_operations = None
            with self._workflow_lock:
                self._workflow_reserved = False

    @staticmethod
    def _publish_feedback(goal_handle, progress: WorkflowProgress) -> None:
        feedback = ExecuteWorkflow.Feedback()
        feedback.workflow_id = progress.workflow_id
        feedback.stage = progress.stage
        feedback.current_point_id = progress.current_point_id
        feedback.current_task = progress.current_task
        feedback.current_step = progress.current_step
        feedback.total_steps = progress.total_steps
        feedback.detail = progress.detail
        goal_handle.publish_feedback(feedback)

    def destroy_node(self) -> None:
        self._close_mqtt_trigger()
        self._navigation_gateway.close()
        self._action_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WorkflowNode()
    executor = MultiThreadedExecutor(num_threads=4)
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
