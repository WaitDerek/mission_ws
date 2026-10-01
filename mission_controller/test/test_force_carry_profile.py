"""The dedicated full workflow must not change standalone grasp defaults."""

import unittest
import threading
from types import SimpleNamespace

from geometry_msgs.msg import Pose
from mission_runtime.mission_controller import MissionController
from mission_runtime.box_force_clamp import BoxForceClampMixin
from mission_runtime.taskflow.fixed_operations import FixedRosWorkflowOperations


class TestForceCarryProfile(unittest.TestCase):
    def test_post_drag3_targets_are_independent_and_read_at_clamp_start(self):
        values = {
            "drag_box_tf_force_carry_post_drag3_target_force_left_n": -8.0,
            "drag_box_tf_force_carry_post_drag3_target_force_right_n": 9.0,
            "grasp_box_tf_force_carry_target_force_left_n": -75.0,
            "grasp_box_tf_force_carry_target_force_right_n": 75.0,
        }
        harness = SimpleNamespace(_float=values.__getitem__)

        def targets(profile, drag, label):
            return BoxForceClampMixin._force_carry_clamp_target_override(
                harness, force_carry_profile=profile, drag_mode=drag, label=label
            )

        self.assertEqual(targets(True, True, "step1_left"), {"left": -8.0, "right": 9.0})
        values["drag_box_tf_force_carry_post_drag3_target_force_left_n"] = -6.0
        self.assertEqual(targets(True, True, "step1_left"), {"left": -6.0, "right": 9.0})
        self.assertIsNone(targets(True, True, "step1"))
        self.assertIsNone(targets(False, True, "step1_left"))
        self.assertEqual(targets(True, False, "step1"), {"left": -75.0, "right": 75.0})
        values["grasp_box_tf_force_carry_target_force_left_n"] = -70.0
        self.assertEqual(targets(True, False, "step1"), {"left": -70.0, "right": 75.0})

    def _operation(self, enabled):
        operation = FixedRosWorkflowOperations.__new__(FixedRosWorkflowOperations)
        operation._drag_action_name = "/execute_drag_box_grasp_tf"
        operation._direct_action_name = "/grasp_box_tf"
        operation._drag_grasp_client = object()
        operation._direct_grasp_client = object()
        operation._place_test_client = object()
        operation._target_label = 0
        operation._dry_run = False
        operation._force_carry_profile = enabled
        calls = []

        def call_action(_client, goal, _label):
            calls.append(goal)
            return SimpleNamespace(
                success=True,
                result=SimpleNamespace(success=True, message="ok"),
            )

        operation._call_action = call_action
        return operation, calls

    def test_new_workflow_marks_both_tf_grasp_actions(self):
        operation, calls = self._operation(True)
        task = SimpleNamespace(box_layer=3, box_type="bigbox")
        self.assertTrue(operation.grasp("/execute_drag_box_grasp_tf", "drag", task).success)
        self.assertTrue(operation.grasp("/grasp_box_tf", "grasp", task).success)
        self.assertEqual([goal.force_carry_profile for goal in calls], [True, True])

    def test_existing_workflow_keeps_original_profile(self):
        operation, calls = self._operation(False)
        task = SimpleNamespace(box_layer=1, box_type="smallbox")
        self.assertTrue(operation.grasp("/grasp_box_tf", "grasp", task).success)
        self.assertFalse(calls[0].force_carry_profile)

    def test_place_profile_is_scoped_to_the_dedicated_workflow(self):
        for enabled in (False, True):
            operation, calls = self._operation(enabled)
            self.assertTrue(operation.place("test-place", "bigbox").success)
            self.assertEqual(calls[0].force_carry_profile, enabled)
            self.assertTrue(calls[0].release_after_place)

    def test_grasp_profile_skips_step2_and_starts_carry_after_clamp(self):
        class DryRunHarness:
            def __init__(self):
                self.joint_state_lock = threading.Lock()
                self.latest_slave_arm_pose_sequences = {"left": 0, "right": 0}
                self.events = []

            def _boolean(self, name):
                return {
                    "grasp_box_tf_body_home_carry_enabled": True,
                    "box_post_movel_enabled": True,
                    "box_tf_equalize_dual_target_z_enabled": True,
                }[name]

            def _integer(self, name):
                assert name == "box_post_movel_step_count"
                return 2

            def _string(self, name):
                assert name == "grasp_box_tf_post_movel_sdk_motion_mode"
                return "movel"

            def _force_clamp_mode(self, _prefix):
                return "closed_loop"

            def _post_movel_targets_with_labels(self, left, right, **_kwargs):
                return [("step1", left, right), ("step2", left, right)]

            def _publish_box_grasp_feedback(self, _goal, stage, _detail):
                self.events.append(stage)

            def _execute_tf_body_home_carry(self, *_args, **_kwargs):
                self.events.append("CARRY_CALLED")
                return "carry"

        harness = DryRunHarness()
        goal = SimpleNamespace(request=SimpleNamespace(force_carry_profile=True))
        left = Pose()
        right = Pose()
        left.orientation.w = right.orientation.w = 1.0
        detail = MissionController._execute_post_movel_sequence(
            harness, goal, None, left, right, True,
            box_layer=1, drag_mode=False, model_label="smallbox", tf_mode=True,
        )
        self.assertIn("step2=skipped", detail)
        self.assertEqual(harness.events, ["POST_MOVEL_STEP1_TARGETS", "CARRY_CALLED"])


if __name__ == "__main__":
    unittest.main()
