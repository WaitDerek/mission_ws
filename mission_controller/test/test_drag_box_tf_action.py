import json
import unittest
import threading
import math
import time

from geometry_msgs.msg import Pose, PoseStamped
from mission_interfaces.action import ExecuteDragBoxGrasp
from rclpy.action import GoalResponse

from mission_runtime.common import MissionError
from mission_runtime.box_actions import BoxActionsMixin
from mission_runtime.box_execution import BoxExecutionMixin
from mission_runtime.box_perception import BoxPerceptionMixin
from mission_runtime.box_preparation import BoxPreparationMixin
from mission_runtime.mission_controller import MissionController
from mission_runtime.realman_sdk_adapter import RealManSdkError


class TestDragParallelArmPreparation(unittest.TestCase):
    def test_left_avoidance_and_right_preparation_overlap(self):
        barrier = threading.Barrier(2)

        class Harness(BoxPerceptionMixin):
            direct_sdk_adapter = object()

            def _execute_drag_box_tf_post_detection_left_movej(self, *_args):
                barrier.wait(timeout=2.0)
                return "left reached"

            def _execute_drag_box_tf_right_grasp_preparation(self, *_args):
                barrier.wait(timeout=2.0)
                return "right reached"

            def _publish_box_grasp_feedback(self, *_args):
                pass

        self.assertEqual(
            Harness()._execute_drag_box_parallel_arm_preparation(
                object(), False, 2, "bigbox"
            ),
            ("left reached", "right reached"),
        )


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
        self.calls = []

    def execute_single(self, arm, target, mode, _velocity, _blocking, **kwargs):
        self.calls.append(("single", arm, list(target), mode, dict(kwargs)))
        self.events.append(("single", arm))
        return f"single_{arm}"

    def execute_dual(self, left, right, mode, _velocity, _blocking, **kwargs):
        self.calls.append(("dual", list(left), list(right), mode, dict(kwargs)))
        self.events.append(("dual",))
        return "dual"


class _PostWaistMoveJAdapter:
    def __init__(self):
        self.calls = []

    def execute_dual_movej(self, left, right, velocity, **kwargs):
        self.calls.append(("dual", left, right, velocity, kwargs))
        return "dual_movej"

    def execute_single_movej(self, arm, target, velocity, **kwargs):
        self.calls.append(("single", arm, target, velocity, kwargs))
        return "single_movej"


class _StagedLeftJoinAdapter:
    def __init__(self, solution, failures_before_success=0):
        self.solution = list(solution)
        self.failures_before_success = failures_before_success
        self.ik_calls = []
        self.movej_calls = []
        self.path_checks = []

    def inspect_joint_path(self, arm, start, end, lower, upper):
        self.path_checks.append((arm, list(start), list(end)))
        return "offline path passed"

    def solve_ik(self, arm, target, seed):
        self.ik_calls.append((arm, list(target), list(seed)))
        if len(self.ik_calls) <= self.failures_before_success:
            return None
        return list(self.solution)

    def execute_single_movej(self, arm, target, velocity, **kwargs):
        self.movej_calls.append((arm, list(target), velocity, kwargs))
        return f"movej_{len(self.movej_calls)}"

    def execute_single(self, *_args, **_kwargs):
        raise AssertionError("staged IK join must not call MoveJ_P or MoveL")


class _StagedLeftJoinHarness(BoxExecutionMixin):
    def __init__(self):
        self.values = {
            "drag_box_left_join_motion_mode": "staged_ik_movej",
            "drag_box_left_join_ik_max_attempts": 10,
            "drag_box_left_join_ik_seed_joint_deg": [
                0.0,
                40.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "drag_box_left_join_velocity_percent": 10.0,
            "drag_box_left_join_timeout_sec": 60.0,
            "box_pre_target_arm_movej_feedback_max_age_sec": 1.0,
            "box_pre_target_arm_movej_position_tolerance_rad": 0.01,
            "box_pre_target_arm_movej_velocity_tolerance_rad_sec": 0.01,
            "box_pre_target_arm_movej_stable_samples": 3,
            "box_ik_joint4_negative_required": True,
            "drag_box_left_join_joint4_negative_required": True,
            "drag_box_left_join_joint4_preference_enabled": True,
            "drag_box_tf_calibration_left_join_joint_override_enabled": False,
            "drag_box_tf_calibration_left_join_joint_target_deg": [
                -62.412,
                25.861,
                24.668,
                -40.876,
                -35.381,
                -8.514,
                -3.713,
            ],
            "drag_box_left_join_ik_random_seed_attempts": 0,
            "waist_workspace_left_arm_joint_min_deg": [
                -175.0, -15.0, -175.0, -105.0, -170.0, -65.0, -100.0
            ],
            "waist_workspace_left_arm_joint_max_deg": [
                175.0, 175.0, 175.0, 105.0, 175.0, 20.0, 90.0
            ],
        }
        self.joint_state_lock = threading.Lock()
        self.latest_slave_arm_positions = {
            "left": [math.radians(value) for value in [10, 20, 30, 40, 50, 60, 70]],
            "right": [math.radians(value) for value in [-10, 20, -30, -40, -50, 60, -70]],
        }
        self.latest_slave_arm_state_times = {
            "left": time.monotonic(),
            "right": time.monotonic(),
        }
        self.latest_slave_arm_state_sequences = {"left": 1, "right": 1}
        self.latest_slave_arm_velocities = {
            "left": [0.0] * 7,
            "right": [0.0] * 7,
        }
        self.feedback = []

    def _string(self, name):
        return str(self.values[name])

    def _integer(self, name):
        return int(self.values[name])

    def _float(self, name):
        return float(self.values[name])

    def _float_array(self, name):
        return [float(value) for value in self.values[name]]

    def _boolean(self, name):
        return bool(self.values[name])

    def _drag_left_join_joint4_preference(self, _box_layer, _model_label):
        return None

    def _check_canceled(self, _goal_handle, _description):
        return None

    def _publish_box_grasp_feedback(self, _goal_handle, stage, detail):
        self.feedback.append((stage, detail))


class _AvoidancePathHarness(BoxPreparationMixin):
    def __init__(self, adapter):
        self.direct_sdk_adapter = adapter
        self.joint_state_lock = threading.Lock()
        self.latest_slave_arm_positions = {"left": [math.radians(v) for v in
            [-72.452, 133.071, 70.555, -100.099, -114.234, -49.202, -25.357]]}
        self.latest_slave_arm_state_times = {"left": time.monotonic()}
        self.values = {
            "drag_box_tf_post_detection_left_movej_enabled": True,
            "drag_box_tf_post_detection_left_transition_joint_units_bigbox":
                [1104, 83030, -93968, -4985, 99573, 5457, 2099],
            "drag_box_tf_post_detection_left_transition_joint_units_smallbox":
                [1104, 83030, -93968, -4985, 99573, 5457, 2099],
            "box_pre_detection_left_movej_command_units_per_degree": 1000.0,
            "box_pre_detection_left_movej_device": 0,
            "box_pre_target_arm_movej_feedback_max_age_sec": 1.0,
            "waist_workspace_left_arm_joint_min_deg":
                [-175, -15, -175, -105, -170, -65, -100],
            "waist_workspace_left_arm_joint_max_deg":
                [175, 175, 175, 105, 175, 20, 90],
        }
        self.feedback = []

    def _boolean(self, name):
        return bool(self.values[name])

    def _float(self, name):
        return float(self.values[name])

    def _integer(self, name):
        return int(self.values[name])

    def _float_array(self, name):
        return list(self.values[name])

    def _drag_box_tf_post_detection_left_movej_joint_units(self, _layer, _model):
        return [-171982, -204, 93820, -25000, 4401, 5999, -4935]

    def _publish_box_grasp_feedback(self, _goal, stage, detail):
        self.feedback.append((stage, detail))

    def _check_canceled(self, _goal, _description):
        return None

    def _execute_pre_detection_arm_intermediate_movej(self, *_args, **_kwargs):
        raise AssertionError("path check must precede the first MoveJ")


class _PostWaistMoveJHarness(BoxExecutionMixin):
    def __init__(self, *, action_prefix="grasp_box_tf", enabled=True):
        prefix = f"{action_prefix}_post_waist_pre_movej"
        self.values = {
            f"{prefix}_enabled": enabled,
            f"{prefix}_command_units_per_degree": 1000.0,
            f"{prefix}_velocity_percent": 10.0,
            f"{prefix}_timeout_sec": 120.0,
            "box_pre_target_arm_movej_feedback_max_age_sec": 1.0,
            f"{prefix}_left_joint_units_bigbox_layer1": [
                1000,
                2000,
                3000,
                4000,
                5000,
                6000,
                7000,
            ],
            f"{prefix}_right_joint_units_bigbox_layer1": [
                -1000,
                -2000,
                -3000,
                -4000,
                -5000,
                -6000,
                -7000,
            ],
        }
        self.feedback = []
        self.joint_state_lock = threading.Lock()
        self.latest_slave_arm_positions = {
            "left": [math.radians(value) for value in [11, 12, 13, 14, 15, 16, 17]],
            "right": [math.radians(value) for value in [-11, -12, -13, -14, -15, -16, -17]],
        }
        self.latest_slave_arm_state_times = {
            "left": time.monotonic(),
            "right": time.monotonic(),
        }

    def _boolean(self, name):
        return bool(self.values[name])

    def _float(self, name):
        return float(self.values[name])

    def _float_array(self, name):
        return [float(value) for value in self.values[name]]

    def _publish_box_grasp_feedback(self, _goal_handle, stage, detail):
        self.feedback.append((stage, detail))


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
            "drag_box_tf_post_movel_sdk_motion_mode": "movel_offset",
            "drag_box_tf_post_movel_step_drag1_left_xyz_bigbox_layer1": [0.14, 0.0, 0.0],
            "drag_box_tf_post_movel_step_drag1_right_xyz_bigbox_layer1": [0.14, 0.0, 0.0],
            "drag_box_tf_post_movel_step_drag2_left_xyz_bigbox_layer1": [0.0, 0.20, 0.0],
            "drag_box_tf_post_movel_step_drag2_right_xyz_bigbox_layer1": [0.0, 0.20, 0.0],
            "drag_box_tf_post_movel_step_drag3_left_xyz_bigbox_layer1": [-0.14, 0.0, 0.0],
            "drag_box_tf_post_movel_step_drag3_right_xyz_bigbox_layer1": [-0.14, 0.0, 0.0],
        }

    def _boolean(self, name):
        return bool(self.values.get(name, False))

    def _float(self, name):
        return float(self.values[name])

    def _string(self, name):
        return str(self.values[name])

    def _float_array(self, name):
        return [float(value) for value in self.values[name]]

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
        self, _goal_handle, _adapter, _left_target, _dry_run, **_kwargs
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
            "drag_box_tf_left_join_forward_offset_m_layer1": -1.0,
            "drag_box_tf_left_join_forward_offset_m_layer4": -1.0,
            "drag_box_tf_left_join_contact_span_m_bigbox": 0.0,
            "drag_box_tf_left_join_forward_offset_m": 0.0,
            "box_tf_equalize_dual_target_z_enabled": True,
            "drag_box_tf_left_contact_forward_delta_scale": 1.0,
            "grasp_box_tf_freeze_frame": "base_link",
            "left_arm_base_frame": "left_base",
            "right_arm_base_frame": "right_base",
            "left_link8_frame": "left_link",
            "right_link8_frame": "right_link",
            "box_tf_step2_contact_height_tolerance_m": 0.002,
            "box_tf_step2_grasp_span_tolerance_m": 0.002,
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

    def _float(self, name):
        return float(self.values[name])

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
    _endpoint_sync_pose_position_error = staticmethod(
        MissionController._endpoint_sync_pose_position_error
    )


class TestDragBoxTfAction(unittest.TestCase):
    def test_predicted_drag3_left_join_uses_tool_offsets_and_right_height(self):
        harness = _DragTfReanchorHarness()
        harness.values.update({
            "drag_box_tf_post_movel_step_drag1_right_xyz_bigbox_layer1": [0.1, 0.0, 0.0],
            "drag_box_tf_post_movel_step_drag2_right_xyz_bigbox_layer1": [0.0, 0.2, 0.0],
            "drag_box_tf_post_movel_step_drag3_right_xyz_bigbox_layer1": [-0.05, 0.0, 0.0],
        })
        harness._equalize_tf_dual_target_z = lambda left, right, reference: (
            MissionController._equalize_tf_dual_target_z(
                harness, left, right, reference=reference
            )
        )
        frozen = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        left_world = ((0.0, 1.0, 0.0), frozen[1])
        identity = frozen
        for contact_x in (0.5, 0.6):
            right_contact = ((contact_x, -1.0, 0.1), frozen[1])
            result = MissionController._predict_drag_tf_left_join_target(
                harness, frozen, left_world, right_contact, identity, identity,
                box_layer=1, model_label="bigbox",
            )
            self.assertAlmostEqual(result.position.x, 0.05)
            self.assertAlmostEqual(result.position.y, 1.2)
            self.assertAlmostEqual(result.position.z, 0.1)

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
        # targets by the same base_link +Z 0.14 m rigid transform.
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
        self.assertAlmostEqual(left_step2.position.x, 1.5)
        self.assertAlmostEqual(right_step2.position.x, 2.5)
        self.assertAlmostEqual(left_step2.position.z, 3.14)
        self.assertAlmostEqual(right_step2.position.z, 3.14)

    def test_rigid_step2_rejects_unequal_contact_height(self):
        harness = _DragTfReanchorHarness()
        harness.transforms["left_link"] = (
            (1.5, 2.0, 3.01),
            (0.0, 0.0, 0.0, 1.0),
        )
        harness.transforms["right_link"] = (
            (2.5, 2.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        with self.assertRaisesRegex(Exception, "contact height mismatch"):
            MissionController._reanchor_drag_tf_step2_from_actual_grasp(
                harness,
                [("step2", Pose(), Pose())],
                0,
                box_layer=1,
                model_label="bigbox",
            )

    def test_grasp_tf_uses_the_same_rigid_base_z_step2(self):
        harness = _DragTfReanchorHarness()
        harness.transforms["left_link"] = (
            (0.4, -0.3, 1.2),
            (0.0, 0.0, 0.0, 1.0),
        )
        harness.transforms["right_link"] = (
            (-0.4, -0.3, 1.2),
            (0.0, 0.0, 0.0, 1.0),
        )
        targets = [("step2", Pose(), Pose())]
        MissionController._reanchor_drag_tf_step2_from_actual_grasp(
            harness,
            targets,
            0,
            box_layer=1,
            model_label="smallbox",
            drag_mode=False,
        )
        _label, left_step2, right_step2 = targets[0]
        self.assertAlmostEqual(left_step2.position.z, 1.34)
        self.assertAlmostEqual(right_step2.position.z, 1.34)
        self.assertAlmostEqual(left_step2.position.x - right_step2.position.x, 0.8)

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
            model_label="bigbox",
            box_layer=1,
        )

        self.assertNotIn(("carry", "drag_box_tf_body_home_carry"), harness.events)
        tool_offsets = [
            call for call in adapter.calls if call[-1].get("offset_frame_type") == 1
        ]
        self.assertEqual(
            [(call[1], call[2], call[3]) for call in tool_offsets],
            [
                ("right", [0.14, 0.0, 0.0, 0.0, 0.0, 0.0], "movel_offset"),
                ("right", [0.0, 0.20, 0.0, 0.0, 0.0, 0.0], "movel_offset"),
                ("right", [-0.14, 0.0, 0.0, 0.0, 0.0, 0.0], "movel_offset"),
            ],
        )

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
            model_label="bigbox",
            box_layer=1,
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


class TestPostWaistPreMoveJ(unittest.TestCase):
    def test_direct_grasp_executes_layer_specific_dual_movej(self):
        harness = _PostWaistMoveJHarness()
        adapter = _PostWaistMoveJAdapter()

        detail = harness._execute_post_waist_pre_movej(
            _MotionGoal(),
            adapter,
            False,
            1,
            "bigbox",
            drag_mode=False,
            right_arm_only=False,
        )

        self.assertIn("grasp_box_tf_post_waist_pre_movej=enabled", detail)
        self.assertEqual(len(adapter.calls), 1)
        mode, left, right, velocity, kwargs = adapter.calls[0]
        self.assertEqual(mode, "dual")
        self.assertEqual(left, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(right, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0])
        self.assertEqual(velocity, 10.0)
        self.assertEqual(kwargs["blend_radius"], 0)
        self.assertEqual(kwargs["trajectory_connect"], 0)
        self.assertEqual(
            [stage for stage, _detail in harness.feedback],
            [
                "POST_WAIST_PRE_MOVEJ_TARGETS",
                "MOVING_POST_WAIST_PRE_MOVEJ",
                "POST_WAIST_PRE_MOVEJ_REACHED",
            ],
        )

    def test_direct_grasp_can_run_preparation_parallel_before_waist(self):
        harness = _PostWaistMoveJHarness()
        adapter = _PostWaistMoveJAdapter()

        detail = harness._execute_post_waist_pre_movej(
            _MotionGoal(),
            adapter,
            False,
            1,
            "bigbox",
            drag_mode=False,
            right_arm_only=False,
            timing="parallel_before_waist",
        )

        self.assertIn("timing=parallel_before_waist", detail)
        self.assertIn("preparation_order=single_stage", detail)
        self.assertEqual(len(adapter.calls), 1)
        mode, left, right, _, _ = adapter.calls[0]
        self.assertEqual(mode, "dual")
        self.assertEqual(left, [1., 2., 3., 4., 5., 6., 7.])
        self.assertEqual(right, [-1., -2., -3., -4., -5., -6., -7.])
        self.assertEqual(
            [stage for stage, _detail in harness.feedback],
            [
                "PARALLEL_PREPARATION_TARGETS",
                "MOVING_PARALLEL_PREPARATION",
                "PARALLEL_PREPARATION_REACHED",
            ],
        )

    def test_drag_right_reuses_grasp_profile_with_distal_first(self):
        harness = _PostWaistMoveJHarness()
        harness.values["drag_box_tf_right_grasp_preparation_enabled"] = True
        harness.values["grasp_box_tf_post_waist_pre_movej_velocity_percent"] = 15.0
        harness.values["drag_box_tf_post_waist_pre_movej_velocity_percent"] = 10.0
        adapter = _PostWaistMoveJAdapter()

        detail = harness._execute_drag_box_tf_right_grasp_preparation(
            _MotionGoal(),
            adapter,
            False,
            1,
            "bigbox",
        )

        self.assertIn("drag_box_tf_right_grasp_preparation=enabled", detail)
        self.assertIn("target_profile=grasp_box_tf", detail)
        self.assertEqual(len(adapter.calls), 2)
        first_mode, first_arm, first_target, first_velocity, _ = adapter.calls[0]
        self.assertEqual((first_mode, first_arm), ("single", "right"))
        self.assertEqual(first_velocity, 10.0)
        for actual, expected in zip(
            first_target, [-11.0, -12.0, -3.0, -4.0, -5.0, -6.0, -7.0]
        ):
            self.assertAlmostEqual(actual, expected)
        second_mode, second_arm, second_target, second_velocity, _ = adapter.calls[1]
        self.assertEqual((second_mode, second_arm), ("single", "right"))
        self.assertEqual(second_velocity, 10.0)
        self.assertEqual(second_target, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0])
        self.assertIn("movej_velocity_percent=10", detail)

    def test_completed_parallel_preparation_is_not_executed_again(self):
        harness = _PostWaistMoveJHarness()
        adapter = _PostWaistMoveJAdapter()

        detail = harness._execute_post_waist_pre_movej(
            _MotionGoal(),
            adapter,
            False,
            1,
            "bigbox",
            drag_mode=False,
            right_arm_only=False,
            already_completed=True,
        )

        self.assertEqual(
            detail,
            "grasp_box_tf_post_waist_pre_movej=completed_parallel_before_waist",
        )
        self.assertEqual(adapter.calls, [])

    def test_drag_interface_is_independent_and_disabled(self):
        harness = _PostWaistMoveJHarness(
            action_prefix="drag_box_tf",
            enabled=False,
        )
        adapter = _PostWaistMoveJAdapter()

        detail = harness._execute_post_waist_pre_movej(
            _MotionGoal(),
            adapter,
            False,
            1,
            "bigbox",
            drag_mode=True,
            right_arm_only=True,
        )

        self.assertEqual(detail, "drag_box_tf_post_waist_pre_movej=disabled")
        self.assertEqual(adapter.calls, [])


class TestAvoidancePath(unittest.TestCase):
    def test_box_models_send_three_ros_movej_stages_without_offline_check(self):
        class RecordOfflineCheck:
            def __init__(self):
                self.calls = []

            def inspect_joint_path(self, arm, start, end, lower, upper):
                self.calls.append((arm, list(start), list(end)))
                return "offline path passed"

        class CommandClient:
            def __init__(self):
                self.requests = []

            def call_async(self, request):
                self.requests.append(request)
                return object()

        class DirectHarness(_AvoidancePathHarness):
            def __init__(self):
                super().__init__(RecordOfflineCheck())
                self.values.update({
                    "box_joint1_command_service_name": "/robot/command",
                    "box_pre_detection_left_movej_velocity": 10,
                    "box_pre_detection_left_movej_blend_radius": 0,
                    "box_pre_detection_left_movej_trajectory_connect": 0,
                    "dependency_wait_timeout_sec": 5.0,
                })
                self.body_command_client = CommandClient()
                self.waited_targets = []
                self.latest_slave_arm_state_sequences = {"left": 3}

            def _string(self, name):
                return str(self.values[name])

            def _wait_for_service(self, *_args):
                pass

            def _wait_future(self, *_args, **_kwargs):
                return object()

            def _parse_string_command_response(self, *_args):
                pass

            def _wait_for_post_arm_joint_targets(self, *args, **kwargs):
                self.waited_targets.append((args, kwargs))

        for model in ("bigbox", "smallbox"):
            for layer in range(1, 5):
                harness = DirectHarness()
                result = harness._execute_drag_box_tf_post_detection_left_movej(
                    _MotionGoal(), False, layer, model
                )
                expected_targets = [
                    [1104, 83030, -93968, -4985, 99573, 5457, 2099],
                    [-171982, 83030, -93968, -4985, 99573, 5457, 2099],
                    [-171982, -204, 93820, -25000, 4401, 5999, -4935],
                ]
                self.assertEqual(len(harness.body_command_client.requests), 3)
                for request, expected in zip(
                    harness.body_command_client.requests, expected_targets
                ):
                    payload = json.loads(request.data)
                    self.assertEqual(payload, {
                        "device": 0,
                        "payload": {
                            "command": "movej", "joint": expected,
                            "v": 10, "r": 0, "trajectory_connect": 0,
                        },
                    })
                self.assertEqual(len(harness.waited_targets), 3)
                for (_args, kwargs), expected in zip(
                    harness.waited_targets, expected_targets
                ):
                    self.assertEqual(kwargs["active_arms"], ("left",))
                    self.assertEqual(
                        [round(math.degrees(value) * 1000) for value in _args[1]],
                        expected,
                    )
                self.assertIn("order=transition_then_joint1_then_remaining", result)
                self.assertEqual(
                    [
                        stage for stage, _ in harness.feedback
                        if stage.endswith("_REACHED")
                    ],
                    [
                        "POST_DETECTION_LEFT_ARM_TRANSITION_REACHED",
                        "POST_DETECTION_LEFT_ARM_JOINT1_REACHED",
                        "POST_DETECTION_LEFT_ARM_REMAINING_REACHED",
                        "POST_DETECTION_LEFT_ARM_REACHED",
                    ],
                )
                self.assertEqual(harness.direct_sdk_adapter.calls, [])
                self.assertNotIn(
                    "POST_DETECTION_LEFT_ARM_PATH_CHECKING",
                    [stage for stage, _ in harness.feedback],
                )


class TestDragDetectionArmSelection(unittest.TestCase):
    def test_bigbox_and_smallbox_use_separate_right_camera_parameters(self):
        class Harness:
            values = {
                "drag_box_tf_detection_arm": "right",
                "drag_box_tf_detection_arm_smallbox": "right",
            }

            def _string(self, name):
                return self.values[name]

        harness = Harness()
        for model in ("bigbox", "smallbox"):
            self.assertEqual(
                MissionController._box_detection_arm(
                    harness, tf_mode=True, drag_mode=True, model_label=model
                ),
                "right",
            )
        harness.values["drag_box_tf_detection_arm"] = "left"
        self.assertEqual(
            MissionController._box_detection_arm(
                harness, tf_mode=True, drag_mode=True, model_label="bigbox"
            ),
            "left",
        )
        self.assertEqual(
            MissionController._box_detection_arm(
                harness, tf_mode=True, drag_mode=True, model_label="smallbox"
            ),
            "right",
        )


class TestStagedLeftJoinIkMoveJ(unittest.TestCase):
    @staticmethod
    def _retry_adapter(solution, failures, harness=None):
        class RetryAdapter(_StagedLeftJoinAdapter):
            def execute_single_movej(self, arm, target, velocity, **kwargs):
                self.movej_calls.append((arm, list(target), velocity, kwargs))
                if len(self.movej_calls) <= failures:
                    if harness is not None:
                        with harness.joint_state_lock:
                            harness.latest_slave_arm_positions["left"] = [
                                math.radians(value) for value in target
                            ]
                            harness.latest_slave_arm_velocities["left"] = [0.0] * 7
                            harness.latest_slave_arm_state_sequences["left"] += 1
                            harness.latest_slave_arm_state_times["left"] = (
                                time.monotonic()
                            )
                    raise RealManSdkError(
                        "direct MoveJ left failed: return_code=1"
                    )
                return f"movej_{len(self.movej_calls)}"

        return RetryAdapter(solution)

    def test_return_code_one_with_reached_feedback_moves_joint1_next(self):
        harness = _StagedLeftJoinHarness()
        harness.values["box_pre_target_arm_movej_stable_samples"] = 1
        solution = [-11.0, 22.0, 33.0, -44.0, 55.0, -16.0, 27.0]
        adapter = self._retry_adapter(solution, 1, harness)
        target = Pose()
        target.orientation.w = 1.0

        detail = harness._execute_drag_box_left_join(
            _MotionGoal(), adapter, target, False
        )

        self.assertEqual(len(adapter.movej_calls), 2)
        self.assertEqual(adapter.movej_calls[1][1], solution)
        self.assertIn("fresh stationary Joint2-through-Joint7 feedback", detail)
        self.assertIn(
            "DRAG_LEFT_JOIN_JOINT2_TO_7_CONFIRMED_AFTER_ERROR",
            [stage for stage, _detail in harness.feedback],
        )

    def test_return_code_one_retries_five_times_when_stopped_short(self):
        class StoppedShortHarness(_StagedLeftJoinHarness):
            def _drag_left_join_stage_after_movej_error(
                self, _goal_handle, _target, _sequence, _timeout_sec
            ):
                return False, "stationary short of target"

        harness = StoppedShortHarness()
        solution = [-11.0, 22.0, 33.0, -44.0, 55.0, -16.0, 27.0]
        adapter = self._retry_adapter(solution, 6)
        target = Pose()
        target.orientation.w = 1.0

        with self.assertRaisesRegex(MissionError, "after 5 retries"):
            harness._execute_drag_box_left_join(
                _MotionGoal(), adapter, target, False
            )

        self.assertEqual(len(adapter.movej_calls), 6)
        self.assertEqual(
            sum(
                stage == "DRAG_LEFT_JOIN_JOINT2_TO_7_RETRY"
                for stage, _detail in harness.feedback
            ),
            5,
        )

    def test_return_code_one_waits_for_position_progress_before_retry(self):
        harness = _StagedLeftJoinHarness()
        harness.values["box_pre_target_arm_movej_stable_samples"] = 2
        target_deg = [10.0, 30.0, 30.0, -40.0, 50.0, 10.0, 5.0]
        with harness.joint_state_lock:
            sequence_before = harness.latest_slave_arm_state_sequences["left"]
        def advance_arm():
            for fraction in range(1, 9):
                time.sleep(0.12)
                with harness.joint_state_lock:
                    harness.latest_slave_arm_positions["left"] = [
                        math.radians(start + (end - start) * fraction / 8)
                        for start, end in zip(
                            [10, 20, 30, 40, 50, 60, 70], target_deg
                        )
                    ]
                    # The field that was used for the old stop decision is
                    # smaller than the configured tolerance while moving.
                    harness.latest_slave_arm_velocities["left"] = [0.004] * 7
                    harness.latest_slave_arm_state_sequences["left"] += 1
                    harness.latest_slave_arm_state_times["left"] = time.monotonic()
            for _ in range(30):
                time.sleep(0.02)
                with harness.joint_state_lock:
                    harness.latest_slave_arm_state_sequences["left"] += 1
                    harness.latest_slave_arm_state_times["left"] = time.monotonic()

        worker = threading.Thread(target=advance_arm)
        worker.start()
        try:
            reached, detail = harness._drag_left_join_stage_after_movej_error(
                _MotionGoal(), target_deg, sequence_before, 2.5
            )
        finally:
            worker.join()
        self.assertTrue(reached)
        self.assertIn("position_change_0p5s_rad", detail)

    def test_return_code_one_retry_then_first_stage_succeeds(self):
        class StoppedShortHarness(_StagedLeftJoinHarness):
            def _drag_left_join_stage_after_movej_error(
                self, _goal_handle, _target, _sequence, _timeout_sec
            ):
                return False, "stationary short of target"

        harness = StoppedShortHarness()
        solution = [-11.0, 22.0, 33.0, -44.0, 55.0, -16.0, 27.0]
        adapter = self._retry_adapter(solution, 2)
        target = Pose()
        target.orientation.w = 1.0

        detail = harness._execute_drag_box_left_join(
            _MotionGoal(), adapter, target, False
        )

        self.assertIn("left_join=staged_ik_movej", detail)
        self.assertEqual(len(adapter.movej_calls), 4)
        self.assertEqual(adapter.movej_calls[2][1][0], 10.0)
        self.assertEqual(adapter.movej_calls[3][1], solution)

    def test_bigbox_layer1_joint4_preference_rejects_weak_negative_branch(self):
        self.assertFalse(BoxExecutionMixin._drag_left_join_joint4_acceptable(
            [0.0, 0.0, 0.0, -17.994, 0.0, 0.0, 0.0],
            (-50.0, 15.0), negative_required=True,
        ))
        self.assertTrue(BoxExecutionMixin._drag_left_join_joint4_acceptable(
            [0.0, 0.0, 0.0, -52.0, 0.0, 0.0, 0.0],
            (-50.0, 15.0), negative_required=True,
        ))

    def test_bigbox_layer1_chooses_joint4_nearest_minus_fifty(self):
        class PreferredHarness(_StagedLeftJoinHarness):
            def _drag_left_join_joint4_preference(self, _layer, _model):
                return -50.0, 15.0

        class VaryingIkAdapter(_StagedLeftJoinAdapter):
            def solve_ik(self, arm, target, seed):
                self.ik_calls.append((arm, list(target), list(seed)))
                solution = [-80.0, 23.0, 24.0, -51.0, -31.0, -21.0, -14.0]
                if len(self.ik_calls) == 1:
                    solution[3] = -17.994
                elif len(self.ik_calls) == 2:
                    solution[3] = -44.0
                return solution

        harness = PreferredHarness()
        adapter = VaryingIkAdapter([0.0] * 7)
        target = Pose()
        target.orientation.w = 1.0
        harness._execute_drag_box_left_join(
            _MotionGoal(), adapter, target, False,
            box_layer=1, model_label="bigbox",
        )
        self.assertEqual(len(adapter.ik_calls), 10)
        self.assertEqual(adapter.movej_calls[-1][1][3], -51.0)

    def test_retries_ik_then_moves_joint2_to_7_before_joint1(self):
        harness = _StagedLeftJoinHarness()
        solution = [-11.0, 22.0, 33.0, -44.0, 55.0, -16.0, 27.0]
        adapter = _StagedLeftJoinAdapter(solution, failures_before_success=2)
        target = Pose()
        target.position.x = 0.4
        target.position.y = 0.5
        target.position.z = 0.6
        target.orientation.w = 1.0

        detail = harness._execute_drag_box_left_join(
            _MotionGoal(),
            adapter,
            target,
            False,
        )

        self.assertIn("left_join=staged_ik_movej", detail)
        self.assertEqual(len(adapter.ik_calls), 10)
        self.assertEqual(
            adapter.ik_calls[0][2],
            [0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(len(adapter.movej_calls), 2)
        self.assertEqual(adapter.path_checks, [])
        first_target = adapter.movej_calls[0][1]
        final_target = adapter.movej_calls[1][1]
        for actual, expected in zip(
            first_target,
            [10.0, 22.0, 33.0, -44.0, 55.0, -16.0, 27.0],
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(final_target, solution)
        stages = [stage for stage, _detail in harness.feedback]
        self.assertEqual(stages.count("DRAG_LEFT_JOIN_IK_SOLVING"), 10)
        self.assertLess(
            stages.index("DRAG_LEFT_JOIN_JOINT2_TO_7_REACHED"),
            stages.index("DRAG_LEFT_JOIN_FINAL_MOVEJ_REACHED"),
        )

    def test_chooses_preferred_ik_solution_without_offline_path_checks(self):
        class PathAwareAdapter(_StagedLeftJoinAdapter):
            def solve_ik(self, arm, target, seed):
                self.ik_calls.append((arm, list(target), list(seed)))
                solution = [-11.0, 22.0, 33.0, -44.0, 55.0, -16.0, 27.0]
                if len(self.ik_calls) > 1:
                    solution[3] = -50.0
                return solution

            def inspect_joint_path(self, arm, start, end, lower, upper):
                self.path_checks.append((arm, list(start), list(end)))
                if end[3] == -44.0:
                    raise RealManSdkError("sample=17/42: singularity return_code=-1")
                return "offline path passed"

        harness = _StagedLeftJoinHarness()
        adapter = PathAwareAdapter([0.0] * 7)
        target = Pose()
        target.orientation.w = 1.0

        detail = harness._execute_drag_box_left_join(
            _MotionGoal(), adapter, target, False
        )

        self.assertIn("left_join=staged_ik_movej", detail)
        self.assertEqual(adapter.movej_calls[-1][1][3], -50.0)
        self.assertEqual(adapter.path_checks, [])

    def test_ranks_ik_solutions_by_minimum_joint_margin(self):
        class TwoBranches(_StagedLeftJoinAdapter):
            def solve_ik(self, arm, target, seed):
                self.ik_calls.append((arm, list(target), list(seed)))
                return [-11.0,
                        20.0 if len(self.ik_calls) == 1 else 35.0,
                        33.0, -50.0, 55.0, -16.0, 27.0]

        harness = _StagedLeftJoinHarness()
        adapter = TwoBranches([0.0] * 7)
        target = Pose()
        target.orientation.w = 1.0

        harness._execute_drag_box_left_join(_MotionGoal(), adapter, target, False)

        self.assertEqual(adapter.movej_calls[-1][1][1], 35.0)
        self.assertEqual(adapter.path_checks, [])

    def test_does_not_call_offline_path_checker_before_motion(self):
        class InvalidPathAdapter(_StagedLeftJoinAdapter):
            def inspect_joint_path(self, arm, start, end, lower, upper):
                raise RealManSdkError("sample=1/5: self-collision return_code=1")

        harness = _StagedLeftJoinHarness()
        adapter = InvalidPathAdapter([-11, 22, 33, -44, 55, -16, 27])
        target = Pose()
        target.orientation.w = 1.0

        harness._execute_drag_box_left_join(
            _MotionGoal(), adapter, target, False
        )

        self.assertEqual(adapter.path_checks, [])
        self.assertEqual(len(adapter.movej_calls), 2)

    def test_fails_only_after_all_ten_ik_attempts(self):
        harness = _StagedLeftJoinHarness()
        adapter = _StagedLeftJoinAdapter([0.0] * 7, failures_before_success=10)
        target = Pose()
        target.orientation.w = 1.0

        with self.assertRaisesRegex(
            MissionError, "after 10 distinct-seed attempts"
        ):
            harness._execute_drag_box_left_join(
                _MotionGoal(),
                adapter,
                target,
                False,
            )

        self.assertEqual(len(adapter.ik_calls), 10)
        self.assertEqual(adapter.movej_calls, [])

    def test_rejects_nonnegative_joint4_after_all_attempts(self):
        harness = _StagedLeftJoinHarness()
        adapter = _StagedLeftJoinAdapter(
            [0.0, 20.0, 30.0, 4.0, 50.0, 10.0, 5.0]
        )
        target = Pose()
        target.orientation.w = 1.0

        with self.assertRaisesRegex(
            MissionError, "every returned solution had non-negative Joint4"
        ):
            harness._execute_drag_box_left_join(
                _MotionGoal(),
                adapter,
                target,
                False,
            )

        self.assertEqual(len(adapter.ik_calls), 10)
        self.assertEqual(adapter.movej_calls, [])


if __name__ == "__main__":
    unittest.main()


class TestDisabledLeftJoinJoint4Filters(unittest.TestCase):
    def test_positive_joint4_is_executed_when_left_join_filters_disabled(self):
        class PositiveAdapter(_StagedLeftJoinAdapter):
            def solve_ik(self, arm, target, seed):
                self.ik_calls.append((arm, list(target), list(seed)))
                return [-11.0, 22.0, 33.0, 30.754, 55.0, -16.0, 27.0]

        harness = _StagedLeftJoinHarness()
        harness.values["drag_box_left_join_joint4_preference_enabled"] = False
        harness.values["drag_box_left_join_joint4_negative_required"] = False
        self.assertTrue(harness.values["box_ik_joint4_negative_required"])
        self.assertIsNone(harness._drag_left_join_joint4_preference(1, "bigbox"))
        adapter = PositiveAdapter([0.0] * 7)
        target = Pose()
        target.orientation.w = 1.0
        harness._execute_drag_box_left_join(_MotionGoal(), adapter, target, False)
        self.assertAlmostEqual(adapter.movej_calls[-1][1][3], 30.754)
