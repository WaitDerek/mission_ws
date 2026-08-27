import unittest

from mission_interfaces.action import ExecuteDragBoxGrasp
from rclpy.action import GoalResponse

from mission_runtime.box_actions import BoxActionsMixin
from mission_runtime.mission_controller import MissionController


class _DispatchHarness(BoxActionsMixin):
    def __init__(self):
        self.calls = []

    def _execute_box_grasp_with_action_type(
        self, goal_handle, action_type, action_name, *, tf_mode=False
    ):
        self.calls.append((goal_handle, action_type, action_name, tf_mode))
        return action_name


class _Logger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)


class _GoalHarness:
    def __init__(self, **overrides):
        self.values = {
            "box_direct_movel_enabled": True,
            "drag_box_post_movel_enabled": True,
            "direct_movel_target_mode": "camera_offset_box_orientation",
            "direct_motion_backend": "python_sdk",
        }
        self.values.update(overrides)
        self.logger = _Logger()
        self.delegated = []

    def _boolean(self, name):
        return bool(self.values[name])

    def _string(self, name):
        return str(self.values[name])

    def get_logger(self):
        return self.logger

    def _tf_grasp_goal_prerequisites(self, label):
        return MissionController._tf_grasp_goal_prerequisites(self, label)

    def _drag_box_grasp_goal_callback_for_mission(
        self, request, mission_name, *, require_tf
    ):
        return MissionController._drag_box_grasp_goal_callback_for_mission(
            self,
            request,
            mission_name,
            require_tf=require_tf,
        )

    def _box_grasp_goal_callback_for_mission(self, request, mission_name):
        self.delegated.append((request, mission_name))
        return GoalResponse.ACCEPT


class _Goal:
    def __init__(self, *, dry_run=False):
        self.dry_run = dry_run


class TestDragBoxTfAction(unittest.TestCase):
    def test_regular_drag_uses_non_tf_path(self):
        harness = _DispatchHarness()
        goal_handle = object()

        result = harness._execute_drag_box_grasp(goal_handle)

        self.assertEqual(result, "execute_drag_box_grasp")
        self.assertEqual(
            harness.calls,
            [
                (
                    goal_handle,
                    ExecuteDragBoxGrasp,
                    "execute_drag_box_grasp",
                    False,
                )
            ],
        )

    def test_tf_drag_uses_frozen_tf_path(self):
        harness = _DispatchHarness()
        goal_handle = object()

        result = harness._execute_drag_box_grasp_tf(goal_handle)

        self.assertEqual(result, "execute_drag_box_grasp_tf")
        self.assertEqual(
            harness.calls,
            [
                (
                    goal_handle,
                    ExecuteDragBoxGrasp,
                    "execute_drag_box_grasp_tf",
                    True,
                )
            ],
        )

    def test_tf_drag_goal_delegates_after_all_prerequisites_pass(self):
        harness = _GoalHarness()
        request = _Goal()

        response = MissionController._drag_box_grasp_tf_goal_callback(
            harness, request
        )

        self.assertEqual(response, GoalResponse.ACCEPT)
        self.assertEqual(harness.delegated, [(request, "drag_box_grasp_tf")])

    def test_tf_drag_goal_rejects_non_tf_target_mode(self):
        harness = _GoalHarness(direct_movel_target_mode="camera_offset")

        response = MissionController._drag_box_grasp_tf_goal_callback(
            harness, _Goal()
        )

        self.assertEqual(response, GoalResponse.REJECT)
        self.assertFalse(harness.delegated)

    def test_physical_drag_goal_rejects_non_sdk_backend(self):
        harness = _GoalHarness(direct_motion_backend="ros_service")

        response = MissionController._drag_box_grasp_goal_callback(
            harness, _Goal(dry_run=False)
        )

        self.assertEqual(response, GoalResponse.REJECT)
        self.assertFalse(harness.delegated)

    def test_drag_goal_rejects_disabled_drag_sequence(self):
        harness = _GoalHarness(drag_box_post_movel_enabled=False)

        response = MissionController._drag_box_grasp_goal_callback(
            harness, _Goal(dry_run=True)
        )

        self.assertEqual(response, GoalResponse.REJECT)
        self.assertFalse(harness.delegated)


if __name__ == "__main__":
    unittest.main()
