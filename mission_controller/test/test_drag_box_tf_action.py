import unittest
import threading

from geometry_msgs.msg import Pose, PoseStamped
from mission_interfaces.action import ExecuteDragBoxGrasp
from rclpy.action import GoalResponse

from mission_runtime.common import MissionError
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
            "drag_box_tf_body_home_carry_enabled": False,
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


class _MotionGoal:
    is_cancel_requested = False


class _SequenceAdapter:
    def __init__(self, events):
        self.events = events

    def execute_single(self, arm, _target, _mode, _velocity, _blocking, **_kwargs):
        self.events.append(("single", arm))
        return f"single_{arm}"

    def execute_dual(self, _left, _right, _mode, _velocity, _blocking, **_kwargs):
        self.events.append(("dual",))
        return "dual"


class _DragTfSequenceHarness:
    def __init__(self, carry_enabled):
        self.events = []
        self.joint_state_lock = threading.Lock()
        self.latest_slave_arm_pose_sequences = {"left": 0, "right": 0}
        self._last_tf_body_home_carry_arm_targets = None
        self.values = {
            "box_post_movel_enabled": False,
            "drag_box_post_movel_enabled": True,
            "drag_box_tf_body_home_carry_enabled": carry_enabled,
            "grasp_box_tf_body_home_carry_enabled": True,
            "direct_movel_blocking": True,
            "box_post_movel_step4_motion_mode": "movej",
            "box_post_movel_velocity_percent": 10.0,
            "direct_sdk_motion_timeout_sec": 10.0,
        }

    def _boolean(self, name):
        return bool(self.values.get(name, False))

    def _float(self, name):
        return float(self.values[name])

    def _string(self, name):
        return str(self.values[name])

    def _post_movel_targets_with_labels(self, *_args, **_kwargs):
        pose = Pose()
        pose.orientation.w = 1.0
        return [
            ("step1", pose, pose),
            ("step_drag1", pose, pose),
            ("step_drag2", pose, pose),
            ("step_drag3", pose, pose),
            ("step1_left", pose, pose),
            ("step2", pose, pose),
            ("step3", pose, pose),
        ]

    def _publish_box_grasp_feedback(self, _goal_handle, stage, _detail):
        self.events.append(("feedback", stage))

    def _rebase_post_movel_targets_after_tf_carry(self, *_args, **_kwargs):
        self.events.append(("rebase",))

    def _execute_drag_box_left_join(
        self, _goal_handle, _adapter, _left_target, _dry_run
    ):
        self.events.append(("left_join",))
        return "left_join"

    def _execute_tf_body_home_carry(
        self,
        _goal_handle,
        _adapter,
        _dry_run,
        *,
        parameter_prefix,
        **_kwargs,
    ):
        self.events.append(("carry", parameter_prefix))
        return "carry"


class _NonTfCarryHarness:
    def __init__(self, *, drag_mode, carry_enabled):
        self.values = {
            "drag_box_tf_body_home_carry_enabled": False,
            "grasp_box_tf_body_home_carry_enabled": False,
        }
        self.values[
            "drag_box_tf_body_home_carry_enabled"
            if drag_mode
            else "grasp_box_tf_body_home_carry_enabled"
        ] = carry_enabled

    def _boolean(self, name):
        return bool(self.values[name])


class _DragTfCallHarness:
    def __init__(self):
        self.values = {
            "drag_box_tf_body_home_carry_enabled": True,
            "grasp_box_tf_body_home_carry_enabled": False,
            "box_step2_waist_endpoint_sync_enabled": False,
            "drag_box_left_arm_enabled": True,
            "box_grasp_execution_mode": "joint123_then_arms",
            "drag_box_post_movel_enabled": True,
            "box_post_movel_step_count": 2,
        }

    def _boolean(self, name):
        return bool(self.values[name])

    def _string(self, name):
        return str(self.values[name])

    def _integer(self, name):
        return int(self.values[name])


class _DragTfReanchorHarness:
    def __init__(self):
        self.values = {
            "box_tf_equalize_dual_target_z_enabled": True,
            "grasp_box_tf_freeze_frame": "base_link",
            "left_arm_base_frame": "left_base",
            "right_arm_base_frame": "right_base",
            "left_link8_frame": "left_link",
            "right_link8_frame": "right_link",
            "left_step2": [0.14, 0.0, 0.0],
            "right_step2": [0.14, 0.0, 0.0],
        }
        frozen = PoseStamped()
        frozen.header.frame_id = "base_link"
        frozen.pose.position.x = 1.0
        frozen.pose.position.y = 2.0
        frozen.pose.position.z = 3.0
        frozen.pose.orientation.w = 1.0
        self._last_grasp_box_tf_box_pose = frozen
        self._last_grasp_box_tf_box_to_link7_targets = {
            "left": ((-0.5, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "right": ((0.5, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        }
        self.transforms = {
            "left_base": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "right_base": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "left_link": ((1.5, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)),
            "right_link": ((1.5, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)),
        }

    def _string(self, name):
        return str(self.values[name])

    def _boolean(self, name):
        return bool(self.values[name])

    def _float_array(self, name):
        return list(self.values[name])

    def _post_movel_xyz_parameter_name(
        self, arm, step, _model_label, **_kwargs
    ):
        self.assert_step2(step)
        return f"{arm}_step2"

    @staticmethod
    def assert_step2(step):
        if step != 2:
            raise AssertionError(f"unexpected step: {step}")

    def _lookup_tf_carry_transform(
        self, _target_frame, source_frame, *, parameter_prefix=None
    ):
        del parameter_prefix
        return self.transforms[source_frame]

    @staticmethod
    def _pose_stamped_to_transform(pose):
        return (
            (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z),
            (
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ),
        )

    @staticmethod
    def _endpoint_sync_transform_to_pose(transform):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = transform[0]
        pose.orientation.x, pose.orientation.y = transform[1][:2]
        pose.orientation.z, pose.orientation.w = transform[1][2:]
        return pose

    _drag_tf_world_transform_to_arm_pose = (
        MissionController._drag_tf_world_transform_to_arm_pose
    )


class TestDragBoxTfAction(unittest.TestCase):
    def test_tf_dual_target_z_equalization_uses_mean_for_dual_motion(self):
        harness = _GoalHarness(box_tf_equalize_dual_target_z_enabled=True)
        left = Pose()
        right = Pose()
        left.position.z = 0.10
        right.position.z = 0.18

        left_result, right_result, detail = (
            MissionController._equalize_tf_dual_target_z(
                harness, left, right, reference="average"
            )
        )

        self.assertAlmostEqual(left_result.position.z, 0.14)
        self.assertAlmostEqual(right_result.position.z, 0.14)
        self.assertIn("reference=average", detail)
        self.assertAlmostEqual(left.position.z, 0.10)
        self.assertAlmostEqual(right.position.z, 0.18)

    def test_tf_delayed_left_join_uses_actual_right_z(self):
        harness = _GoalHarness(box_tf_equalize_dual_target_z_enabled=True)
        left = Pose()
        right = Pose()
        left.position.z = 0.10
        right.position.z = 0.18

        left_result, right_result, detail = (
            MissionController._equalize_tf_dual_target_z(
                harness, left, right, reference="right"
            )
        )

        self.assertAlmostEqual(left_result.position.z, 0.18)
        self.assertAlmostEqual(right_result.position.z, 0.18)
        self.assertIn("reference=right", detail)

    def test_drag3_reanchor_uses_actual_right_link_for_left_join_and_step2(self):
        harness = _DragTfReanchorHarness()

        MissionController._capture_drag_tf_right_grasp_relation(harness)
        self.assertAlmostEqual(
            harness._last_drag_box_tf_right_grasp_relation[0][0], 0.5
        )

        # The right arm moved the box one metre along the common box X axis.
        harness.transforms["right_link"] = (
            (2.5, 2.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        left_join, _detail = MissionController._reanchor_drag_tf_left_join_after_drag3(
            harness
        )
        self.assertAlmostEqual(left_join.position.x, 1.5)
        self.assertAlmostEqual(left_join.position.y, 2.0)
        self.assertAlmostEqual(left_join.position.z, 3.0)

        # Once both arms are physically holding the box, Step2 must move both
        # targets by the same 0.14 m common-box transform.
        harness.transforms["left_link"] = (
            (1.5, 2.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        targets = [("step2", Pose(), Pose())]
        MissionController._reanchor_drag_tf_step2_from_actual_grasp(
            harness,
            targets,
            0,
            box_layer=1,
            model_label="bigbox",
        )
        _label, left_step2, right_step2 = targets[0]
        self.assertAlmostEqual(left_step2.position.x, 1.64)
        self.assertAlmostEqual(right_step2.position.x, 2.64)
        self.assertAlmostEqual(left_step2.position.z, right_step2.position.z)

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

    def test_drag_tf_carry_switch_off_does_not_trigger(self):
        harness = _DragTfSequenceHarness(carry_enabled=False)
        adapter = _SequenceAdapter(harness.events)

        MissionController._execute_post_movel_sequence(
            harness,
            _MotionGoal(),
            adapter,
            Pose(),
            Pose(),
            False,
            drag_mode=True,
            right_arm_only=True,
            delayed_left_join=True,
            tf_mode=True,
        )

        self.assertNotIn(("carry", "drag_box_tf_body_home_carry"), harness.events)

    def test_drag_tf_carry_runs_after_left_join_and_dual_step2(self):
        harness = _DragTfSequenceHarness(carry_enabled=True)
        adapter = _SequenceAdapter(harness.events)

        MissionController._execute_post_movel_sequence(
            harness,
            _MotionGoal(),
            adapter,
            Pose(),
            Pose(),
            False,
            drag_mode=True,
            right_arm_only=True,
            delayed_left_join=True,
            tf_mode=True,
        )

        carry_index = harness.events.index(("carry", "drag_box_tf_body_home_carry"))
        left_join_index = harness.events.index(("left_join",))
        left_step1_index = harness.events.index(("single", "left"))
        step2_index = max(
            index
            for index, event in enumerate(harness.events[:carry_index])
            if event == ("dual",)
        )
        rebase_index = harness.events.index(("rebase",), carry_index + 1)
        step3_index = next(
            index
            for index, event in enumerate(harness.events[rebase_index + 1 :], rebase_index + 1)
            if event == ("dual",)
        )
        self.assertLess(left_join_index, left_step1_index)
        self.assertLess(left_step1_index, step2_index)
        self.assertLess(step2_index, carry_index)
        self.assertLess(carry_index, rebase_index)
        self.assertLess(rebase_index, step3_index)

    def test_non_tf_drag_carry_switch_is_rejected(self):
        harness = _NonTfCarryHarness(drag_mode=True, carry_enabled=True)

        with self.assertRaisesRegex(
            MissionError, "drag_box_tf_body_home_carry_enabled"
        ):
            MissionController._call_direct_box_movel(
                harness,
                _MotionGoal(),
                Pose(),
                True,
                1,
                drag_mode=True,
                tf_mode=False,
            )

    def test_non_tf_drag_goal_rejects_carry_switch(self):
        harness = _GoalHarness(drag_box_tf_body_home_carry_enabled=True)

        response = MissionController._drag_box_grasp_goal_callback(
            harness, _Goal(dry_run=True)
        )

        self.assertEqual(response, GoalResponse.REJECT)
        self.assertFalse(harness.delegated)

    def test_drag_tf_carry_uses_drag_parameters_and_dual_arm_preconditions(self):
        harness = _DragTfCallHarness()
        # The precondition path must accept Drag TF using only its own switch
        # and the DragBox post sequence, without consulting GraspBox carry.
        self.assertTrue(harness._boolean("drag_box_tf_body_home_carry_enabled"))
        self.assertFalse(harness._boolean("grasp_box_tf_body_home_carry_enabled"))
        self.assertEqual(
            harness._string("box_grasp_execution_mode"), "joint123_then_arms"
        )
        self.assertTrue(harness._boolean("drag_box_post_movel_enabled"))


if __name__ == "__main__":
    unittest.main()
