import threading
import time
import unittest

from mission_runtime.taskflow.model import (
    ObservationPlan,
    ObservationResult,
    ObservationTask,
    StepResult,
)
from mission_runtime.taskflow.state_machine import DepalletizingWorkflowEngine


def _plan(point_id, *columns):
    tasks = tuple(
        ObservationTask(
            stack_id=f"stack-{point_id}-{column}",
            stack_index=column,
            column=column,
            layer=index + 1,
            box_type="bigbox" if column else "smallbox",
            order_index=index,
        )
        for index, column in enumerate(columns)
    )
    return ObservationResult(True, ObservationPlan(str(point_id), tasks), "ok")


class _Operations:
    def __init__(self, observations):
        self.observations = dict(observations)
        self.calls = []
        self.cancel_requested = False
        self.fail_at = None
        self.grasp_release = None

    def is_cancel_requested(self):
        return self.cancel_requested

    def _result(self, label):
        if self.fail_at == label:
            return StepResult(False, f"{label} failed")
        return StepResult(True, f"{label} complete")

    def navigate(self, request):
        label = f"nav:{request.point_id}"
        self.calls.append(label)
        return self._result(label)

    def observe(self, point_id):
        label = f"observe:{point_id}"
        self.calls.append(label)
        return self.observations.get(
            point_id, ObservationResult(False, message="no plan")
        )

    def grasp(self, action_name, request_id, task, operation_point_id):
        del request_id, task
        label = f"grasp:{operation_point_id}:{action_name}"
        self.calls.append(label)
        if self.grasp_release is not None:
            self.grasp_release.wait(timeout=1.0)
        return self._result(label)

    def place(self, request_id, task):
        del request_id, task
        label = "place"
        self.calls.append(label)
        return self._result(label)


class TestDepalletizingStateMachine(unittest.TestCase):
    def test_primary_path_processes_one_then_three(self):
        operations = _Operations({"1": _plan("1", 0, 1), "3": _plan("3", 1)})

        outcome = DepalletizingWorkflowEngine(operations).run("workflow", "token")

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.completed_observation_count, 2)
        self.assertEqual(outcome.completed_box_count, 3)
        self.assertEqual(
            operations.calls,
            [
                "nav:1",
                "observe:1",
                "nav:6",
                "grasp:6:/grasp_box_tf",
                "nav:16",
                "place",
                "nav:5",
                "grasp:5:/execute_drag_box_grasp_tf",
                "nav:16",
                "place",
                "nav:3",
                "observe:3",
                "nav:9",
                "grasp:9:/execute_drag_box_grasp_tf",
                "nav:16",
                "place",
            ],
        )

    def test_point_one_no_plan_falls_back_to_two_then_four(self):
        operations = _Operations({"2": _plan("2", 0), "4": _plan("4", 1)})

        outcome = DepalletizingWorkflowEngine(operations).run("workflow", "token")

        self.assertTrue(outcome.success)
        self.assertEqual(
            [call for call in operations.calls if call.startswith("observe")],
            ["observe:1", "observe:2", "observe:4"],
        )
        self.assertIn("nav:8", operations.calls)
        self.assertIn("nav:11", operations.calls)

    def test_point_two_no_plan_aborts_without_later_calls(self):
        operations = _Operations({})

        outcome = DepalletizingWorkflowEngine(operations).run("workflow", "token")

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.final_stage, "OBSERVE_2")
        self.assertEqual(
            operations.calls,
            ["nav:1", "observe:1", "nav:2", "observe:2"],
        )

    def test_failed_child_stops_before_next_operation(self):
        operations = _Operations({"1": _plan("1", 0), "3": _plan("3", 0)})
        operations.fail_at = "grasp:6:/grasp_box_tf"

        outcome = DepalletizingWorkflowEngine(operations).run("workflow", "token")

        self.assertFalse(outcome.success)
        self.assertNotIn("nav:16", operations.calls)
        self.assertNotIn("place", operations.calls)

    def test_place_completion_gates_next_pickup_navigation(self):
        operations = _Operations({"1": _plan("1", 0), "3": _plan("3", 0)})
        operations.grasp_release = threading.Event()
        outcomes = []
        thread = threading.Thread(
            target=lambda: outcomes.append(
                DepalletizingWorkflowEngine(operations).run("workflow", "token")
            )
        )
        thread.start()
        deadline = time.monotonic() + 1.0
        while "grasp:6:/grasp_box_tf" not in operations.calls:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        self.assertNotIn("nav:16", operations.calls)
        self.assertNotIn("place", operations.calls)
        operations.grasp_release.set()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(outcomes[0].success)


if __name__ == "__main__":
    unittest.main()
