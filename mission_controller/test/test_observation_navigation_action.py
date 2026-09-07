import threading
import time
from types import SimpleNamespace
import unittest

try:
    import rclpy
    from mission_interfaces.action import ExecuteObservationNavigation
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    from mission_runtime.taskflow.model import (
        ObservationPlan,
        ObservationResult,
        ObservationTask,
        StepResult,
    )
    from mission_runtime.taskflow.mqtt_start import MqttStartRequest
    from mission_runtime.taskflow.node import DepalletizingWorkflowNode
    from mission_runtime.taskflow.ros_operations import RosWorkflowOperations
except (ImportError, ModuleNotFoundError):
    ROS_WORKFLOW_AVAILABLE = False
else:
    ROS_WORKFLOW_AVAILABLE = True


class _Operations:
    def __init__(self):
        self.calls = []
        self.canceled = False

    def is_cancel_requested(self):
        return self.canceled

    def cancel_active(self):
        self.canceled = True

    def close(self):
        return None

    def navigate(self, request):
        self.calls.append(f"nav:{request.point_id}")
        return StepResult(True, "arrived")

    def observe(self, point_id):
        self.calls.append(f"observe:{point_id}")
        plan = ObservationPlan(
            point_id,
            (
                ObservationTask(
                    stack_id="front-left",
                    stack_index=0,
                    column=0,
                    layer=1,
                    box_type="smallbox",
                    order_index=0,
                ),
                ObservationTask(
                    stack_id="front-right",
                    stack_index=1,
                    column=1,
                    layer=2,
                    box_type="bigbox",
                    order_index=1,
                ),
            ),
            "global plan ready",
        )
        return ObservationResult(True, plan, "global plan ready")


class _StatusBridge:
    def __init__(self):
        self.statuses = []

    def publish_status(self, values):
        self.statuses.append(dict(values))
        return True

    def close(self):
        return None


@unittest.skipUnless(ROS_WORKFLOW_AVAILABLE, "ROS workflow interfaces unavailable")
class TestObservationNavigationAction(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.workflow_node = DepalletizingWorkflowNode()
        self.client_node = Node("observation_navigation_test_client")
        self.operations = _Operations()
        self.workflow_node._acquire_lease = lambda workflow_id: SimpleNamespace(
            success=True,
            lease_token="test-token",
            message=f"acquired {workflow_id}",
        )
        self.workflow_node._release_lease = lambda workflow_id, token: SimpleNamespace(
            success=True, message=f"released {workflow_id}"
        )
        self.workflow_node._make_observation_navigation_operations = (
            lambda goal_handle, workflow_id: self.operations
        )
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.workflow_node)
        self.executor.add_node(self.client_node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.client = ActionClient(
            self.client_node,
            ExecuteObservationNavigation,
            "/execute_observation_navigation",
        )
        self.assertTrue(self.client.wait_for_server(timeout_sec=2.0))

    def tearDown(self):
        self.executor.shutdown(timeout_sec=2.0)
        self.spin_thread.join(timeout=2.0)
        self.client_node.destroy_node()
        self.workflow_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    @staticmethod
    def _wait(future, timeout=3.0):
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("ROS Action future timed out")
            time.sleep(0.01)
        return future.result()

    def test_action_runs_only_navigation_observation_navigation(self):
        goal = ExecuteObservationNavigation.Goal()
        goal.request_id = "test-1"
        goal.start = True
        goal.observation_point_id = 1

        goal_handle = self._wait(self.client.send_goal_async(goal))
        self.assertTrue(goal_handle.accepted)
        wrapped = self._wait(goal_handle.get_result_async())

        self.assertTrue(wrapped.result.success)
        self.assertEqual(wrapped.result.final_stage, "COMPLETE")
        self.assertEqual(wrapped.result.completed_navigation_count, 5)
        self.assertEqual(wrapped.result.completed_observation_count, 1)
        self.assertEqual(wrapped.result.selected_operation_point_id, 3)
        self.assertEqual(
            self.operations.calls,
            ["nav:1", "observe:1", "nav:3", "nav:2", "nav:1", "nav:2"],
        )

    def test_mqtt_mode_routes_to_the_new_action(self):
        bridge = _StatusBridge()
        self.workflow_node._mqtt_start_bridge = bridge

        self.workflow_node._queue_mqtt_start(
            MqttStartRequest(
                request_id="platform-vision-1",
                workflow="observation_navigation",
                observation_point_id=1,
            )
        )
        deadline = time.monotonic() + 3.0
        while not any(status["event"] == "result" for status in bridge.statuses):
            if time.monotonic() >= deadline:
                raise TimeoutError("MQTT observation navigation result timed out")
            time.sleep(0.01)

        result = next(
            status for status in bridge.statuses if status["event"] == "result"
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["workflow"], "observation_navigation")
        self.assertEqual(result["selected_operation_point_id"], 3)
        self.assertEqual(
            self.operations.calls,
            ["nav:1", "observe:1", "nav:3", "nav:2", "nav:1", "nav:2"],
        )

    def test_global_observation_feedback_fields_are_forwarded(self):
        forwarded = []
        operations = RosWorkflowOperations.__new__(RosWorkflowOperations)
        operations._child_feedback_callback = lambda stage, detail: forwarded.append(
            (stage, detail)
        )
        message = SimpleNamespace(
            feedback=SimpleNamespace(
                stage="planning",
                progress=0.72,
                completed_top_boxes=2,
                current_stack_id="front-left",
            )
        )

        operations._forward_feedback("global observation", message)

        self.assertEqual(forwarded[0][0], "planning")
        self.assertIn("progress=72%", forwarded[0][1])
        self.assertIn("completed_top_boxes=2", forwarded[0][1])
        self.assertIn("current_stack_id=front-left", forwarded[0][1])


if __name__ == "__main__":
    unittest.main()
