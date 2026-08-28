import threading
import time
from types import SimpleNamespace
import unittest

try:
    import rclpy
    from mission_interfaces.action import ExecuteWorkflow
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    from mission_runtime.taskflow.model import (
        ObservationPlan,
        ObservationResult,
        ObservationTask,
        StepResult,
    )
    from mission_runtime.taskflow.node import DepalletizingWorkflowNode
    from mission_runtime.taskflow.mqtt_start import MqttStartRequest
except (ImportError, ModuleNotFoundError):
    ROS_WORKFLOW_AVAILABLE = False
else:
    ROS_WORKFLOW_AVAILABLE = True


class _SuccessfulOperations:
    def __init__(self):
        self.calls = []

    def is_cancel_requested(self):
        return False

    def cancel_active(self):
        return None

    def navigate(self, request):
        self.calls.append(f"nav:{request.point_id}")
        return StepResult(True, "arrived")

    def observe(self, point_id):
        self.calls.append(f"observe:{point_id}")
        task = ObservationTask(
            stack_id=f"stack-{point_id}",
            stack_index=0,
            column=0,
            layer=1,
            box_type="smallbox",
            order_index=0,
        )
        return ObservationResult(True, ObservationPlan(point_id, (task,), "ok"), "ok")

    def grasp(self, action_name, request_id, task, operation_point_id):
        del request_id, task
        self.calls.append(f"grasp:{operation_point_id}:{action_name}")
        return StepResult(True, "grasped")

    def place(self, request_id, task):
        del request_id, task
        self.calls.append("place")
        return StepResult(True, "placed")


class _BlockingOperations(_SuccessfulOperations):
    def __init__(self):
        super().__init__()
        self.grasp_started = threading.Event()
        self.canceled = threading.Event()

    def is_cancel_requested(self):
        return self.canceled.is_set()

    def cancel_active(self):
        self.canceled.set()

    def grasp(self, action_name, request_id, task, operation_point_id):
        del request_id, task
        self.calls.append(f"grasp:{operation_point_id}:{action_name}")
        self.grasp_started.set()
        self.canceled.wait(timeout=2.0)
        return StepResult(False, "grasp canceled")


class _StatusBridge:
    def __init__(self):
        self.statuses = []

    def publish_status(self, values):
        self.statuses.append(dict(values))
        return True

    def close(self):
        return None


@unittest.skipUnless(ROS_WORKFLOW_AVAILABLE, "ROS workflow interfaces unavailable")
class TestDepalletizingWorkflowAction(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.workflow_node = DepalletizingWorkflowNode()
        self.client_node = Node("execute_workflow_test_client")
        self.operations = _SuccessfulOperations()
        self.workflow_node._acquire_lease = lambda workflow_id: SimpleNamespace(
            success=True,
            lease_token="test-token",
            message=f"acquired {workflow_id}",
        )
        self.workflow_node._release_lease = lambda workflow_id, token: SimpleNamespace(
            success=True, message=f"released {workflow_id}"
        )
        self.workflow_node._make_operations = (
            lambda goal_handle, workflow_id: self.operations
        )
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.workflow_node)
        self.executor.add_node(self.client_node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.client = ActionClient(
            self.client_node,
            ExecuteWorkflow,
            "/execute_workflow",
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

    def test_start_true_executes_the_complete_top_level_action(self):
        goal = ExecuteWorkflow.Goal()
        goal.start = True
        goal_handle = self._wait(self.client.send_goal_async(goal))

        self.assertTrue(goal_handle.accepted)
        wrapped = self._wait(goal_handle.get_result_async())

        self.assertTrue(wrapped.result.success)
        self.assertEqual(wrapped.result.completed_observation_count, 2)
        self.assertEqual(wrapped.result.completed_box_count, 2)
        self.assertEqual(wrapped.result.final_stage, "COMPLETE")
        self.assertEqual(
            self.operations.calls,
            [
                "nav:1",
                "observe:1",
                "nav:6",
                "grasp:6:/grasp_box_tf",
                "nav:16",
                "place",
                "nav:3",
                "observe:3",
                "nav:10",
                "grasp:10:/grasp_box_tf",
                "nav:16",
                "place",
            ],
        )

    def test_start_false_is_rejected(self):
        goal = ExecuteWorkflow.Goal()
        goal.start = False

        goal_handle = self._wait(self.client.send_goal_async(goal))

        self.assertFalse(goal_handle.accepted)

    def test_mqtt_start_invokes_the_top_level_action(self):
        bridge = _StatusBridge()
        self.workflow_node._mqtt_start_bridge = bridge

        self.workflow_node._queue_mqtt_start(MqttStartRequest("platform-1"))
        deadline = time.monotonic() + 3.0
        while not any(status["event"] == "result" for status in bridge.statuses):
            if time.monotonic() >= deadline:
                raise TimeoutError("MQTT-triggered workflow result timed out")
            time.sleep(0.01)

        events = [status["event"] for status in bridge.statuses]
        result = next(
            status for status in bridge.statuses if status["event"] == "result"
        )
        self.assertEqual(events[0], "received")
        self.assertIn("accepted", events)
        self.assertIn("feedback", events)
        self.assertTrue(result["success"])
        self.assertEqual(result["request_id"], "platform-1")
        self.assertEqual(result["final_stage"], "COMPLETE")

    def test_cancel_stops_the_active_child_and_blocks_later_steps(self):
        blocking = _BlockingOperations()
        self.workflow_node._make_operations = lambda goal_handle, workflow_id: blocking
        goal = ExecuteWorkflow.Goal()
        goal.start = True
        goal_handle = self._wait(self.client.send_goal_async(goal))
        self.assertTrue(goal_handle.accepted)
        self.assertTrue(blocking.grasp_started.wait(timeout=2.0))

        self._wait(goal_handle.cancel_goal_async())
        wrapped = self._wait(goal_handle.get_result_async())

        self.assertFalse(wrapped.result.success)
        self.assertEqual(wrapped.result.final_stage, "CANCELED")
        self.assertNotIn("nav:16", blocking.calls)
        self.assertNotIn("place", blocking.calls)


if __name__ == "__main__":
    unittest.main()
