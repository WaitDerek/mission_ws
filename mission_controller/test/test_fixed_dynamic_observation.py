import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace

from mission_runtime.taskflow.fixed_model import build_fixed_box_tasks
from mission_runtime.taskflow.node import DepalletizingWorkflowNode
from mission_runtime.taskflow.model import ObservationResult
from mission_runtime.taskflow.model import StepResult


class _Operations:
    def __init__(self, events):
        self.events = events

    def is_cancel_requested(self):
        return False

    def navigate(self, request):
        self.events.append(("navigate", request.point_id))
        return StepResult(True, "arrived")

    def grasp(self, action_name, request_id, task):
        self.events.append(("grasp", task.item_index))
        return StepResult(True, "grasped")

    def micro_navigate(self, request):
        self.events.append(("retreat", request.point_id))
        return StepResult(True, "retreated")

    def place(self, request_id, box_type):
        self.events.append(("place", box_type))
        return StepResult(True, "placed")

    def close(self):
        pass


class _Goal:
    def __init__(self):
        self.request = SimpleNamespace(
            request_id="test-dynamic-observation",
            start=True,
            start_item_index=1,
            stop_after_item_index=0,
            dry_run=False,
            use_global_observation=True,
        )
        self.status = None

    def succeed(self):
        self.status = "succeeded"

    def abort(self):
        self.status = "aborted"

    def canceled(self):
        self.status = "canceled"


class _Node:
    _fixed_failure_outcome = staticmethod(DepalletizingWorkflowNode._fixed_failure_outcome)
    _execute_fixed_box_workflow = DepalletizingWorkflowNode._execute_fixed_box_workflow

    def __init__(self, front_count, back_count, *, fail_back=False):
        self.events = []
        self._workflow_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._active_goal_handle = None
        self._active_operations = None
        self._workflow_reserved = True
        self._fixed_total_item_count = 16
        self.fail_back = fail_back
        self.operations = _Operations(self.events)
        fixed = build_fixed_box_tasks(
            drag_point_id=1,
            drag_pickup_pos=(0.24, -0.24, 0.11),
            direct_point_id=3,
            direct_pickup_pos=(0.20, 0.50, 0.06),
            north_drag_point_id=5,
            north_drag_pickup_pos=(1.82, 0.77, -3.03),
            north_direct_point_id=6,
            north_direct_pickup_pos=(1.93, 0.00, -3.02),
        )
        self.front = fixed[:front_count]
        self.back = tuple(
            replace(task, item_index=front_count + index + 1)
            for index, task in enumerate(fixed[8 : 8 + back_count])
        )

    def _publish_fixed_progress(self, goal, workflow_id, stage, task, detail):
        self.events.append(("stage", stage))

    def _acquire_lease(self, workflow_id):
        return SimpleNamespace(success=True, lease_token="lease")

    def _release_lease(self, workflow_id, token):
        return SimpleNamespace(success=True)

    def _make_fixed_operations(self, goal, workflow_id, dry_run):
        return self.operations

    def _observe_fixed_face(self, goal, workflow_id, operations, **kwargs):
        face = kwargs["stage_label"]
        self.events.append(("observe", face))
        if face == "BACK" and self.fail_back:
            return (), self._fixed_failure_outcome(
                workflow_id, "GLOBAL_OBSERVATION_BACK", "Vision failed"
            )
        tasks = self.front if face == "FRONT" else self.back
        self.assert_start_index(kwargs["start_index"], tasks)
        return tasks, None

    @staticmethod
    def assert_start_index(start, tasks):
        if tasks:
            assert start == tasks[0].item_index

    def _integer(self, name):
        return 2

    def _float_array(self, name):
        return [-0.5, 0.0, 0.0]

    def get_logger(self):
        return SimpleNamespace(error=lambda message: self.events.append(("error", message)))


class TestDynamicObservationOrder(unittest.TestCase):
    def test_confirmed_empty_face_is_skipped_only_after_two_observations(self):
        node = _Node(front_count=6, back_count=0)
        node._string = lambda name: "right"
        node._float = lambda name: 30.0
        marker = (
            "global observation viewpoint precheck failed: status=none; "
            "recommended_viewpoint=hold; YOLO did not detect a tote"
        )

        class EmptyOperations:
            observe_count = 0

            def navigate(self, request):
                return StepResult(True, "arrived")

            j2_prepared = False

            def prepare_right_observation_joint2(self, degrees, description):
                assert degrees == 30.0
                self.j2_prepared = True
                return StepResult(True, "J2 reached")

            def move_right_joints(self, joints, description):
                assert self.j2_prepared
                return StepResult(True, "moved")

            def observe(self, point_id):
                self.observe_count += 1
                return ObservationResult(False, message=marker)

        operations = EmptyOperations()
        tasks, failure = DepalletizingWorkflowNode._observe_fixed_face(
            node,
            _Goal(),
            "workflow",
            operations,
            stage_label="BACK",
            pickup_zone="back",
            start_index=7,
            observation_point_parameter="point",
            observation_pos_parameter="pos",
            drag_point_parameter="point",
            drag_pos_parameter="pos",
            drag_layer4_pos_parameter="pos",
            direct_point_parameter="point",
            direct_pos_parameter="pos",
            direct_layer4_pos_parameter="pos",
        )

        self.assertIsNone(failure)
        self.assertEqual(tasks, ())
        self.assertEqual(operations.observe_count, 2)

    def test_front_six_are_placed_before_back_observation(self):
        node = _Node(front_count=6, back_count=3)
        goal = _Goal()

        result = node._execute_fixed_box_workflow(goal)

        self.assertTrue(result.success)
        self.assertEqual(goal.status, "succeeded")
        self.assertEqual(result.completed_box_count, 9)
        self.assertEqual(result.last_completed_item_index, 9)
        back_observe_index = node.events.index(("observe", "BACK"))
        self.assertEqual(
            sum(kind == "place" for kind, _value in node.events[:back_observe_index]),
            6,
        )
        self.assertEqual(node._fixed_total_item_count, 9)

    def test_explicit_front_stop_skips_back_observation(self):
        node = _Node(front_count=6, back_count=3)
        goal = _Goal()
        goal.request.stop_after_item_index = 6

        result = node._execute_fixed_box_workflow(goal)

        self.assertTrue(result.success)
        self.assertEqual(result.completed_box_count, 6)
        self.assertNotIn(("observe", "BACK"), node.events)

    def test_back_observation_failure_preserves_completed_front_count(self):
        node = _Node(front_count=6, back_count=3, fail_back=True)
        goal = _Goal()

        result = node._execute_fixed_box_workflow(goal)

        self.assertFalse(result.success)
        self.assertEqual(result.final_stage, "GLOBAL_OBSERVATION_BACK")
        self.assertEqual(result.completed_box_count, 6)
        self.assertEqual(result.last_completed_item_index, 6)


if __name__ == "__main__":
    unittest.main()


def test_item_range_does_not_skip_front_global_observation():
    node = _Node(front_count=8, back_count=8)
    goal = _Goal()
    goal.request.start_item_index = 9
    result = node._execute_fixed_box_workflow(goal)
    assert result.success
    assert [value for kind,value in node.events if kind=="observe"] == ["FRONT","BACK"]
    assert [value for kind,value in node.events if kind=="grasp"] == list(range(9,17))
