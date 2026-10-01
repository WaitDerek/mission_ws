import unittest
from types import SimpleNamespace

from mission_runtime.taskflow.fixed_model import (
    build_fixed_box_tasks,
    build_observed_box_tasks,
)
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

    def micro_navigate(self, request: NavigationRequest):
        self.calls.append(("micro_navigate", request.point_id, request.pos))
        if self.fail_stage == "micro_navigate":
            return StepResult(False, "micro retreat failed")
        return StepResult(True, "micro retreat completed")

    def place(self, request_id, box_type):
        self.calls.append(("place", request_id, box_type))
        if self.fail_stage == "place":
            return StepResult(False, "place failed")
        return StepResult(True, "placed")


def _engine(operations, progress=None):
    tasks = build_fixed_box_tasks(
        drag_point_id=1,
        drag_pickup_pos=(0.24, -0.24, 0.11),
        direct_point_id=3,
        direct_pickup_pos=(0.20, 0.50, 0.06),
        north_drag_point_id=5,
        north_drag_pickup_pos=(1.82, 0.77, -3.03),
        north_direct_point_id=6,
        north_direct_pickup_pos=(1.93, 0.00, -3.02),
    )
    return FixedBoxWorkflowEngine(
        operations,
        tasks,
        place_point_id=6,
        place_pos=(4.6, -0.13, 0.01),
        micro_retreat_pos=(-0.5, 0.0, 0.0),
        progress_callback=progress,
    )


class TestFixedBoxWorkflow(unittest.TestCase):
    def test_runs_exact_front_then_back_16_item_order(self):
        operations = _FakeOperations()
        stages = []
        outcome = _engine(
            operations,
            lambda stage, task, detail: stages.append((stage, getattr(task, "item_index", 0))),
        ).run("workflow", "lease")

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.completed_box_count, 16)
        self.assertEqual(outcome.last_completed_item_index, 16)
        self.assertEqual(outcome.final_stage, "COMPLETE")
        self.assertEqual(
            [call[0] for call in operations.calls],
            ["navigate", "grasp", "micro_navigate", "navigate", "place"] * 16,
        )
        self.assertEqual(
            [call[3] for call in operations.calls if call[0] == "grasp"],
            [1, 2, 1, 3, 2, 4, 3, 4] * 2,
        )
        self.assertEqual(stages[-1], ("COMPLETE", 0))

    def test_uses_distinct_front_and_back_pickup_points(self):
        operations = _FakeOperations()
        outcome = _engine(operations).run("workflow", "lease")

        self.assertTrue(outcome.success)
        pickup_calls = [
            call
            for index, call in enumerate(operations.calls)
            if index % 5 == 0
        ]
        self.assertEqual(
            [call[1] for call in pickup_calls[:8]],
            ["1", "1", "3", "1", "3", "1", "3", "3"],
        )
        self.assertEqual(
            [call[1] for call in pickup_calls[8:]],
            ["5", "5", "6", "5", "6", "5", "6", "6"],
        )
        self.assertEqual(pickup_calls[0][2], (0.24, -0.24, 0.11))
        self.assertEqual(pickup_calls[8][2], (1.82, 0.77, -3.03))

    def test_observed_back_tasks_can_continue_indexes_after_front(self):
        plan = SimpleNamespace(
            tasks=tuple(
                SimpleNamespace(
                    stack_index=index % 2,
                    grasp_mode="direct_grasp" if index % 2 == 0 else "drag",
                    layer=(index % 4) + 1,
                    box_type="smallbox" if index % 2 == 0 else "bigbox",
                )
                for index in range(8)
            )
        )
        tasks = build_observed_box_tasks(
            plan,
            drag_point_id=5,
            drag_pickup_pos=(1.82, 0.77, -3.03),
            direct_point_id=6,
            direct_pickup_pos=(1.93, 0.00, -3.02),
            start_index=9,
            pickup_zone="back",
        )

        self.assertEqual([task.item_index for task in tasks], list(range(9, 17)))
        self.assertTrue(all(task.pickup_zone == "back" for task in tasks))

    def test_six_observed_front_boxes_then_dynamic_back_indexes(self):
        def plan(count):
            return SimpleNamespace(
                tasks=tuple(
                    SimpleNamespace(
                        stack_index=index % 2,
                        grasp_mode="direct_grasp" if index % 2 == 0 else "drag",
                        layer=(index % 4) + 1,
                        box_type="smallbox" if index % 2 == 0 else "bigbox",
                    )
                    for index in range(count)
                )
            )

        common = dict(
            drag_point_id=1,
            drag_pickup_pos=(0.24, -0.24, 0.11),
            direct_point_id=3,
            direct_pickup_pos=(0.20, 0.50, 0.06),
        )
        front = build_observed_box_tasks(
            plan(6), start_index=1, pickup_zone="front", **common
        )
        back = build_observed_box_tasks(
            plan(3), start_index=len(front) + 1, pickup_zone="back", **common
        )
        operations = _FakeOperations()
        options = dict(
            place_point_id=2,
            place_pos=(-0.78, 1.16, -3.07),
            micro_retreat_pos=(-0.5, 0.0, 0.0),
        )
        front_result = FixedBoxWorkflowEngine(operations, front, **options).run(
            "workflow", "lease", completion_stage="FRONT_COMPLETE"
        )
        back_result = FixedBoxWorkflowEngine(operations, front + back, **options).run(
            "workflow", "lease", start_item_index=7
        )

        self.assertTrue(front_result.success)
        self.assertEqual(front_result.final_stage, "FRONT_COMPLETE")
        self.assertEqual(front_result.completed_box_count, 6)
        self.assertTrue(back_result.success)
        self.assertEqual(back_result.completed_box_count, 3)
        self.assertEqual(back_result.last_completed_item_index, 9)
        self.assertEqual(len([call for call in operations.calls if call[0] == "grasp"]), 9)

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

    def test_micro_retreat_failure_stops_before_place_navigation(self):
        operations = _FakeOperations(fail_stage="micro_navigate")
        outcome = _engine(operations).run("workflow", "lease")

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.final_stage, "MICRO_RETREAT_1")
        self.assertEqual(
            [call[0] for call in operations.calls],
            ["navigate", "grasp", "micro_navigate"],
        )

    def test_cancellation_is_reported(self):
        operations = _FakeOperations()
        operations.canceled = True
        outcome = _engine(operations).run("workflow", "lease")

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.final_stage, "CANCELED")
        self.assertEqual(outcome.completed_box_count, 0)


if __name__ == "__main__":
    unittest.main()


def test_later_start_item_still_navigates_to_pickup():
    operations = _FakeOperations()
    outcome = _engine(operations).run("full", "lease", start_item_index=14, stop_after_item_index=14)
    assert outcome.success
    assert operations.calls[0][0] == "navigate"
    assert operations.calls[1][0] == "grasp"
