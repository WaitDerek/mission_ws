"""Bridge MQTT workflow-start messages to the local workflow Action."""

from __future__ import annotations

import threading
import time
import uuid

from mission_interfaces.action import ExecuteWorkflow
from rclpy.action import ActionClient

from .mqtt_start import (
    MqttStartRequest,
    MqttWorkflowStartBridge,
    normalize_robot_id,
)


class WorkflowMqttTriggerMixin:
    def _initialize_mqtt_trigger(self) -> None:
        self._mqtt_start_bridge = None
        self._mqtt_start_lock = threading.Lock()
        self._mqtt_start_busy = False
        self._mqtt_pending_start = None
        self._mqtt_workflow_client = ActionClient(
            self,
            ExecuteWorkflow,
            self._string("workflow_action_name"),
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
        self._mqtt_start_timer = self.create_timer(
            0.05,
            self._dispatch_mqtt_start,
            callback_group=self._client_group,
        )

    def _queue_mqtt_start(self, request: MqttStartRequest) -> None:
        normalized = MqttStartRequest(
            request_id=request.request_id or f"mqtt-{uuid.uuid4().hex[:12]}",
            robot_id=request.robot_id,
        )
        expected_robot_id = self._string("robot_id")
        if normalize_robot_id(normalized.robot_id) != expected_robot_id:
            bridge = self._mqtt_start_bridge
            if bridge is not None:
                bridge.publish_status(
                    {
                        "event": "rejected",
                        "robot_id": normalized.robot_id,
                        "request_id": normalized.request_id,
                        "message": (
                            f"workflow MQTT robot id does not match {expected_robot_id!r}"
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
            self._publish_mqtt_status(
                "rejected",
                normalized.request_id,
                normalized.robot_id,
                message="another MQTT workflow request is active",
            )
            return
        self._publish_mqtt_status(
            "received",
            normalized.request_id,
            normalized.robot_id,
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
                request.robot_id,
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
                    request.request_id, request.robot_id, message
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._finish_mqtt_start(
                "rejected",
                request.request_id,
                request.robot_id,
                message=f"failed to send workflow Action goal: {exc}",
            )
            return
        future.add_done_callback(
            lambda result: self._mqtt_goal_response(
                request.request_id, request.robot_id, result
            )
        )

    def _mqtt_goal_response(self, request_id: str, robot_id, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self._finish_mqtt_start(
                "rejected",
                request_id,
                robot_id,
                message=f"workflow goal failed: {exc}",
            )
            return
        if not goal_handle.accepted:
            self._finish_mqtt_start(
                "rejected",
                request_id,
                robot_id,
                message="workflow goal was rejected",
            )
            return
        self._publish_mqtt_status(
            "accepted",
            request_id,
            robot_id,
            message="workflow Action goal accepted",
        )
        goal_handle.get_result_async().add_done_callback(
            lambda result: self._mqtt_workflow_result(request_id, robot_id, result)
        )

    def _mqtt_workflow_feedback(
        self, request_id: str, robot_id, wrapped_feedback
    ) -> None:
        feedback = wrapped_feedback.feedback
        self._publish_mqtt_status(
            "feedback",
            request_id,
            robot_id,
            workflow_id=feedback.workflow_id,
            stage=feedback.stage,
            point_id=feedback.current_point_id,
            task=feedback.current_task,
            current_step=int(feedback.current_step),
            total_steps=int(feedback.total_steps),
            detail=feedback.detail,
        )

    def _mqtt_workflow_result(self, request_id: str, robot_id, future) -> None:
        try:
            wrapped = future.result()
            result = wrapped.result
        except Exception as exc:  # noqa: BLE001
            self._finish_mqtt_start(
                "result",
                request_id,
                robot_id,
                success=False,
                message=f"workflow result failed: {exc}",
            )
            return
        self._finish_mqtt_start(
            "result",
            request_id,
            robot_id,
            success=bool(result.success),
            workflow_id=result.workflow_id,
            message=result.message,
            completed_task_count=int(result.completed_task_count),
            final_stage=result.final_stage,
            ros_status=int(wrapped.status),
        )

    def _publish_mqtt_status(
        self, event: str, request_id: str, robot_id, **values
    ) -> bool:
        bridge = self._mqtt_start_bridge
        if bridge is None:
            return False
        return bridge.publish_status(
            {
                "event": event,
                "robot_id": robot_id,
                "request_id": request_id,
                **values,
            }
        )

    def _finish_mqtt_start(
        self, event: str, request_id: str, robot_id, **values
    ) -> None:
        with self._mqtt_start_lock:
            self._mqtt_start_busy = False
            self._mqtt_pending_start = None
        self._publish_mqtt_status(event, request_id, robot_id, **values)

    def _close_mqtt_trigger(self) -> None:
        bridge = self._mqtt_start_bridge
        if bridge is not None:
            bridge.close()
