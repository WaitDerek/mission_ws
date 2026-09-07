import unittest

from mission_runtime.taskflow.model import (
    ObservationPlan,
    ObservationResult,
    ObservationTask,
    StepResult,
)
from mission_runtime.taskflow.observation_navigation import (
    ObservationNavigationWorkflowEngine,
)


def _plan(point_id="1", stack_index=0):
    tasks = (
        ObservationTask(
            stack_id="front-stack-left",
            stack_index=stack_index,
            column=stack_index,
            layer=1,
            box_type="smallbox",
            order_index=0,
        ),
        ObservationTask(
            stack_id="front-stack-right",
            stack_index=1 - stack_index,
            column=1 - stack_index,
            layer=2,
            box_type="bigbox",
            order_index=1,
        ),
    )
    return ObservationPlan(point_id, tasks, "plan ready")


class _Operations:
    def __init__(self, *, fail_at="", stack_index=0):
        self.fail_at = fail_at
        self.stack_index = stack_index
        self.calls = []
        self.canceled = False

    def is_cancel_requested(self):
        return self.canceled

    def navigate(self, request):
        call = f"nav:{request.point_id}"
        self.calls.append(call)
        second_navigation_failure = (
            self.fail_at == "second_navigation"
            and sum(item.startswith("nav:") for item in self.calls) == 2
        )
        return StepResult(
            call != self.fail_at and not second_navigation_failure,
            "arrived"
            if call != self.fail_at and not second_navigation_failure
            else "blocked",
        )

    def observe(self, point_id):
        call = f"observe:{point_id}"
        self.calls.append(call)
        if call == self.fail_at:
            return ObservationResult(False, message="no plan")
        plan = _plan(point_id, self.stack_index)
        return ObservationResult(True, plan, "plan ready")


class TestObservationNavigationWorkflow(unittest.TestCase):
    def test_success_requires_both_navigation_results_and_observation(self):
        operations = _Operations(stack_index=0)

        outcome = ObservationNavigationWorkflowEngine(operations).run(
            "workflow", "lease", 1
        )

        self.assertTrue(outcome.success)
        self.assertEqual(
            operations.calls,
            ["nav:1", "observe:1", "nav:3", "nav:2", "nav:1", "nav:2"],
        )
        self.assertEqual(outcome.completed_navigation_count, 5)
        self.assertEqual(outcome.completed_observation_count, 1)
        self.assertEqual(outcome.planned_box_count, 2)
        self.assertEqual(outcome.selected_operation_point_id, "3")

    def test_first_navigation_failure_blocks_observation(self):
        operations = _Operations(fail_at="nav:1")

        outcome = ObservationNavigationWorkflowEngine(operations).run(
            "workflow", "lease", 1
        )

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.final_stage, "NAVIGATE_OBSERVATION_1")
        self.assertEqual(operations.calls, ["nav:1"])

    def test_observation_failure_blocks_operation_navigation(self):
        operations = _Operations(fail_at="observe:1")

        outcome = ObservationNavigationWorkflowEngine(operations).run(
            "workflow", "lease", 1
        )

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.final_stage, "OBSERVE_1")
        self.assertEqual(operations.calls, ["nav:1", "observe:1"])

    def test_second_navigation_failure_is_reported(self):
        operations = _Operations(fail_at="second_navigation", stack_index=1)

        outcome = ObservationNavigationWorkflowEngine(operations).run(
            "workflow", "lease", 1
        )

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.final_stage, "NAVIGATE_OPERATION_1")
        self.assertEqual(operations.calls, ["nav:1", "observe:1", "nav:1"])

    def test_invalid_observation_point_fails_without_side_effects(self):
        operations = _Operations()

        outcome = ObservationNavigationWorkflowEngine(operations).run(
            "workflow", "lease", 5
        )

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.final_stage, "VALIDATING")
        self.assertEqual(operations.calls, [])


if __name__ == "__main__":
    unittest.main()
