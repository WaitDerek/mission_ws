import unittest

from mission_runtime.taskflow.fixed_model import build_fixed_box_tasks
from mission_runtime.taskflow.fixed_state_machine import FixedBoxWorkflowEngine
from mission_runtime.taskflow.model import NavigationRequest, StepResult


class _FakeOperations:
    def __init__(self, *, fail_stage=""):
        self.calls = []
        self.fail_stage = fail_stage
        self.canceled = False

    def is_cancel_requested(self):
        return self.canceled

    def navigate(self, request: NavigationRequest):
        self.calls.append(("navigate", request.point_id, request.pos))
        if self.fail_stage == "navigate":
            return StepResult(False, "platform refused navigation")
        return StepResult(True, "arrived")

    def grasp(self, action_name, request_id, task):
        self.calls.append(("grasp", action_name, task.box_type, task.box_layer))
        if self.fail_stage == "grasp":
            return StepResult(False, "grasp failed")
        return StepResult(True, "grasped")

    def place(self, request_id, box_type):
        self.calls.append(("place", request_id, box_type))
        if self.fail_stage == "place":
            return StepResult(False, "place failed")
        return StepResult(True, "placed")


def _engine(operations, progress=None):
    tasks = build_fixed_box_tasks(
        drag_point_id=5,
        drag_pickup_pos=(3.069969, 0.759915, 3.077476),
        direct_point_id=4,
        direct_pickup_pos=(-2.95, -0.02, 3.11),
    )
    return FixedBoxWorkflowEngine(
        operations,
        tasks,
        place_point_id=6,
        place_pos=(4.6, -0.13, 0.01),
        progress_callback=progress,
    )


class TestFixedBoxWorkflow(unittest.TestCase):
    def test_runs_exact_eight_item_order(self):
        operations = _FakeOperations()
        stages = []
        outcome = _engine(
            operations,
            lambda stage, task, detail: stages.append((stage, getattr(task, "item_index", 0))),
        ).run("workflow", "lease")

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.completed_box_count, 8)
        self.assertEqual(outcome.last_completed_item_index, 8)
        self.assertEqual(outcome.final_stage, "COMPLETE")
        self.assertEqual(
            [call[0] for call in operations.calls],
            ["navigate", "grasp", "navigate", "place"] * 8,
        )
        self.assertEqual(
            [call[3] for call in operations.calls if call[0] == "grasp"],
            [1, 2, 1, 3, 2, 4, 3, 4],
        )
        self.assertEqual(stages[-1], ("COMPLETE", 0))

    def test_resume_and_stop_range(self):
        operations = _FakeOperations()
        outcome = _engine(operations).run(
            "workflow", "lease", start_item_index=3, stop_after_item_index=5
        )

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.completed_box_count, 3)
        self.assertEqual(outcome.last_completed_item_index, 5)
        self.assertEqual(
            [call[3] for call in operations.calls if call[0] == "grasp"],
            [1, 3, 2],
        )

    def test_failure_stops_before_next_item(self):
        operations = _FakeOperations(fail_stage="place")
        outcome = _engine(operations).run("workflow", "lease")

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.final_stage, "PLACE_1")
        self.assertEqual(outcome.completed_box_count, 0)
        self.assertEqual(len([call for call in operations.calls if call[0] == "place"]), 1)

    def test_cancellation_is_reported(self):
        operations = _FakeOperations()
        operations.canceled = True
        outcome = _engine(operations).run("workflow", "lease")

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.final_stage, "CANCELED")
        self.assertEqual(outcome.completed_box_count, 0)


if __name__ == "__main__":
    unittest.main()
