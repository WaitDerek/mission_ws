import threading
import time
import unittest
from types import SimpleNamespace

from mission_runtime.realman_sdk_adapter import RealManSdkAdapter, RealManSdkError


class _SlowSuccessfulRobot:
    def __init__(self):
        self.calls = []

    def rm_movel(self, *_args):
        self.calls.append("movel")
        time.sleep(0.06)
        return 0

    def rm_movej_p(self, *_args):
        self.calls.append("movej_p")
        time.sleep(0.06)
        return 0

    def rm_movel_offset(self, *_args):
        self.calls.append(("movel_offset", _args))
        time.sleep(0.06)
        return 0

    def rm_movej(self, *_args):
        self.calls.append(("movej", _args))
        time.sleep(0.06)
        return 0


class _IkRobot:
    def __init__(self):
        self.params = []

    def rm_algo_inverse_kinematics(self, params):
        self.params.append(params)
        return 0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


class _IkParams:
    def __init__(self, seed, target, flag):
        self.seed = seed
        self.target = target
        self.flag = flag


class _CompactMatrix:
    def __init__(self, value):
        self.data = [0.0] * 16
        self.data[3] = value


class _RemoteMatrix:
    def __init__(self, rows, columns, data):
        self.rows = rows
        self.columns = columns
        self.data = data


class _ContinuousIkRobot:
    def __init__(self):
        self.initializations = []
        self.seeds = []

    def rm_algo_ik_remote_init(self, dt_sec, frame):
        self.initializations.append((dt_sec, frame))

    def rm_algo_pos2matrix(self, target):
        return _CompactMatrix(target[0])

    def rm_algo_ik_remote(self, matrix, seed, output):
        del matrix
        self.seeds.append(list(seed))
        for index, value in enumerate(seed):
            output[index] = value + 1.0
        return 0


class _ForceMoveParam:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _IdentityMatrix:
    data = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


class _ForceRobot:
    def __init__(self, sign):
        self.sign = float(sign)
        self.commands = []
        self.start_calls = 0
        self.stop_calls = 0

    def rm_get_current_tool_frame(self):
        return 0, {"name": "Arm_Tip", "pose": [0.0] * 6}

    def rm_get_current_work_frame(self):
        return 0, {"name": "World", "pose": [0.0] * 6}

    def rm_get_current_arm_state(self):
        return 0, {"joint": [0.0] * 7}

    def rm_algo_forward_kinematics(self, _joint, flag):
        return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0] if flag == 0 else [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]

    def rm_algo_pos2matrix(self, _pose):
        return _IdentityMatrix()

    def rm_get_force_data(self):
        # A one-shot SDK command is enough to let the controller's internal
        # force/position loop reach the target; the test models that result.
        fy = self.sign * (1.2 if self.commands else 0.0)
        return 0, {"tool_zero_force_data": [0.0, fy, 0.0, 0.0, 0.0, 0.0]}

    def rm_start_force_position_move(self):
        self.start_calls += 1
        return 0

    def rm_force_position_move(self, command):
        self.commands.append(command)
        return 0

    def rm_stop_force_position_move(self):
        self.stop_calls += 1
        return 0


class _SettlingForceRobot(_ForceRobot):
    def __init__(self, pre_force_values):
        super().__init__(1.0)
        self.pre_force_values = list(pre_force_values)
        self.pre_start_reads = 0
        self.reads_at_start = None

    def rm_get_force_data(self):
        if self.commands:
            return super().rm_get_force_data()
        index = min(self.pre_start_reads, len(self.pre_force_values) - 1)
        fy = self.pre_force_values[index]
        self.pre_start_reads += 1
        return 0, {
            "tool_zero_force_data": [0.0, fy, 0.0, 0.0, 0.0, 0.0]
        }

    def rm_start_force_position_move(self):
        self.reads_at_start = self.pre_start_reads
        return super().rm_start_force_position_move()


class _DelayedContactForceRobot(_ForceRobot):
    def __init__(self):
        super().__init__(1.0)
        self.delayed = False

    def rm_get_force_data(self):
        if len(self.commands) == 1 and not self.delayed:
            self.delayed = True
            time.sleep(0.08)
            return 0, {"tool_zero_force_data": [0.0] * 6}
        return super().rm_get_force_data()


class _NeverStableForceRobot(_ForceRobot):
    def __init__(self):
        super().__init__(1.0)
        self.pre_start_reads = 0

    def rm_get_force_data(self):
        self.pre_start_reads += 1
        fy = 1.0 if self.pre_start_reads % 2 else -1.0
        return 0, {
            "tool_zero_force_data": [0.0, fy, 0.0, 0.0, 0.0, 0.0]
        }


class _TransientStartupForceRobot(_ForceRobot):
    def __init__(self):
        super().__init__(1.0)
        self.force_mode_active = False

    def rm_start_force_position_move(self):
        self.force_mode_active = True
        return super().rm_start_force_position_move()

    def rm_stop_force_position_move(self):
        self.force_mode_active = False
        return super().rm_stop_force_position_move()

    def rm_get_force_data(self):
        fy = 3.2 if self.force_mode_active and self.commands else 0.0
        return 0, {
            "tool_zero_force_data": [0.0, fy, 0.0, 0.0, 0.0, 0.0]
        }


class _SingleSpikeThenContactRobot(_TransientStartupForceRobot):
    def __init__(self):
        super().__init__()
        self.startup_spike_sent = False

    def rm_get_force_data(self):
        fy = 0.0
        if self.force_mode_active and self.commands:
            if not self.startup_spike_sent:
                self.startup_spike_sent = True
                fy = 3.2
            elif len(self.commands) >= 5:
                fy = 2.5
        elif not self.force_mode_active and len(self.commands) >= 5:
            fy = 2.5
        return 0, {
            "tool_zero_force_data": [0.0, fy, 0.0, 0.0, 0.0, 0.0]
        }


class _ContactOnSecondAttemptRobot(_TransientStartupForceRobot):
    def rm_get_force_data(self):
        if self.force_mode_active and self.commands:
            fy = 3.2 if self.start_calls == 1 else 2.5
        elif not self.force_mode_active and self.start_calls >= 2:
            fy = 2.5
        else:
            fy = 0.0
        return 0, {
            "tool_zero_force_data": [0.0, fy, 0.0, 0.0, 0.0, 0.0]
        }


class _ContactDipsDuringHoldRobot(_ForceRobot):
    def __init__(self):
        super().__init__(1.0)
        self.post_stop_reads = 0

    def rm_get_force_data(self):
        if self.stop_calls:
            self.post_stop_reads += 1
            fy = 1.5 if self.post_stop_reads == 3 else 2.5
        else:
            fy = 2.5 if self.commands else 0.0
        return 0, {
            "tool_zero_force_data": [0.0, fy, 0.0, 0.0, 0.0, 0.0]
        }


class _StableContactForceRobot(_ForceRobot):
    def __init__(self):
        super().__init__(1.0)

    def rm_get_force_data(self):
        fy = 2.5 if self.commands else 0.0
        return 0, {
            "tool_zero_force_data": [0.0, fy, 0.0, 0.0, 0.0, 0.0]
        }


class TestRealManSdkAdapter(unittest.TestCase):
    def test_offline_algorithm_switches_to_each_arms_controller_dh(self):
        class ProfileRobot:
            def __init__(self, model, eighth_offset):
                self.model = model
                self.eighth_offset = eighth_offset

            def rm_get_robot_info(self):
                return 0, {"arm_dof": 7, "arm_model": self.model, "force_type": "6FB"}

            def rm_get_DH_data(self):
                return 0, {
                    "d": [0.0] * 7 + [self.eighth_offset],
                    "a": [0.0] * 8,
                    "alpha": [0.0] * 8,
                    "offset": [0.0] * 8,
                }

            def rm_get_install_pose(self):
                return {"return_code": 0, "x": 0, "y": -90, "z": 0}

            def rm_get_joint_min_pos(self):
                return 0, [-170.0] * 7

            def rm_get_joint_max_pos(self):
                return 0, [170.0] * 7

            def rm_get_current_tool_frame(self):
                return 0, {"pose": [0.0] * 6}

            rm_get_current_work_frame = rm_get_current_tool_frame

        adapter = object.__new__(RealManSdkAdapter)
        adapter._algo_profiles = {}
        adapter._logger = None
        adapter._arm_model_type = SimpleNamespace(
            RM_MODEL_RXL75_E=1, RM_MODEL_RXR75_E=2,
        )
        adapter._force_type = SimpleNamespace(
            RM_MODEL_RM_B_E=0, RM_MODEL_RM_ZF_E=1,
            RM_MODEL_RM_SF_E=2, RM_MODEL_RM_ISF_E=3,
            RM_MODEL_RM_BV_E=4, RM_MODEL_RM_ISFV_E=5,
        )
        adapter._dh_type = lambda **values: values
        adapter._frame_type = lambda **values: values
        activations = []

        class NativeAlgo:
            # No dh/arm_dof kwargs: those would silently select universal model.
            def __init__(self, model, force):
                self.model, self.force = model, force

            def rm_algo_set_dh(self, dh):
                activations.append((self.model, self.force, dh['d'][-1]))

            def rm_algo_set_angle(self, *angle):
                assert angle == (0.0, -90.0, 0.0)

            def rm_algo_set_joint_min_limit(self, values):
                assert values == [-170.0] * 7

            def rm_algo_set_joint_max_limit(self, values):
                assert values == [170.0] * 7

            def rm_algo_set_toolframe(self, frame):
                assert frame['pose'] == [0.0] * 6

            rm_algo_set_workframe = rm_algo_set_toolframe

        adapter._algo_type = NativeAlgo
        left = ProfileRobot("RXL75", 0.105)
        right = ProfileRobot("RXR75", 0.205)

        adapter._activate_arm_algorithm("left", left)
        adapter._activate_arm_algorithm("right", right)
        adapter._activate_arm_algorithm("left", left)
        # Connected-handle FK can replace the native process-global profile
        # without changing the cached arm name; the same arm must reload DH.
        adapter._activate_arm_algorithm("left", left)

        self.assertEqual(
            activations,
            [(1, 3, 0.105), (2, 3, 0.205), (1, 3, 0.105), (1, 3, 0.105)],
        )

    def test_dual_tool_y_force_clamp_uses_mirrored_motion_and_force(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        left = _ForceRobot(-1.0)
        right = _ForceRobot(1.0)
        adapter._robots = lambda: (left, right)
        adapter._force_position_move_type = _ForceMoveParam
        adapter.stop_all = lambda: None

        result = adapter.execute_tool_y_force_clamp(
            ("left", "right"),
            {"left": -1.0, "right": 1.0},
            speed_mm_s=1.0,
            max_travel_m={"left": 0.05, "right": 0.05},
            timeout_sec=1.0,
            control_period_sec=0.01,
            baseline_stability_window_sec=0.04,
            baseline_stability_max_span_n=0.2,
            baseline_stability_timeout_sec=0.3,
        )

        self.assertEqual(left.start_calls, 1)
        self.assertEqual(right.start_calls, 1)
        self.assertEqual(left.stop_calls, 1)
        self.assertEqual(right.stop_calls, 1)
        self.assertTrue(left.commands)
        self.assertTrue(right.commands)
        self.assertLess(left.commands[-1].desired_force[1], 0.0)
        self.assertGreater(right.commands[-1].desired_force[1], 0.0)
        self.assertGreater(left.commands[-1].pose[1], 0.2)
        self.assertLess(right.commands[-1].pose[1], 0.2)
        self.assertAlmostEqual(left.commands[-1].limit_vel[1], 0.001, places=6)
        self.assertAlmostEqual(right.commands[-1].limit_vel[1], 0.001, places=6)
        self.assertLessEqual(result["final_tool_wrench"]["left"][1], -1.0)
        self.assertGreaterEqual(result["final_tool_wrench"]["right"][1], 1.0)

    def test_delayed_force_read_does_not_catch_up_with_a_large_pose_step(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        right = _DelayedContactForceRobot()
        adapter._robots = lambda: (_ForceRobot(-1.0), right)
        adapter._force_position_move_type = _ForceMoveParam
        adapter.stop_all = lambda: None

        adapter.execute_tool_y_force_clamp(
            ("right",),
            {"right": 1.0},
            speed_mm_s=3.0,
            max_travel_m={"right": 0.05},
            timeout_sec=1.0,
            control_period_sec=0.01,
            baseline_stability_window_sec=0.04,
            baseline_stability_max_span_n=0.2,
            baseline_stability_timeout_sec=0.3,
        )

        self.assertTrue(right.delayed)
        self.assertEqual(len(right.commands), 2)
        self.assertAlmostEqual(right.commands[-1].limit_vel[1], 0.003)
        self.assertLessEqual(
            abs(right.commands[1].pose[1] - right.commands[0].pose[1]),
            3.0 / 1000.0 * 0.01 + 1e-8,
        )

    def test_right_only_tool_y_force_clamp_supports_drag_contact(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        left = _ForceRobot(-1.0)
        right = _ForceRobot(1.0)
        adapter._robots = lambda: (left, right)
        adapter._force_position_move_type = _ForceMoveParam
        adapter.stop_all = lambda: None

        result = adapter.execute_tool_y_force_clamp(
            ("right",),
            {"right": 1.0},
            speed_mm_s=1.0,
            max_travel_m={"right": 0.05},
            timeout_sec=1.0,
            control_period_sec=0.01,
            baseline_stability_window_sec=0.04,
            baseline_stability_max_span_n=0.2,
            baseline_stability_timeout_sec=0.3,
            post_stop_confirmation_sec=0.08,
            post_stop_min_force_n=0.5,
        )

        self.assertFalse(left.commands)
        self.assertTrue(right.commands)
        self.assertEqual(result["arms"], ("right",))
        self.assertGreaterEqual(result["post_stop_confirm_delta_n"], 0.5)

    def test_initial_right_contact_requires_continuous_post_stop_force(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        right = _ContactDipsDuringHoldRobot()
        adapter._robots = lambda: (_ForceRobot(-1.0), right)
        adapter._force_position_move_type = _ForceMoveParam
        adapter.stop_all = lambda: None

        with self.assertRaisesRegex(
            RealManSdkError, "initial right Tool-Y contact was not sustained"
        ):
            adapter.execute_tool_y_force_clamp(
                ("right",),
                {"right": 2.0},
                speed_mm_s=3.0,
                max_travel_m={"right": 0.05},
                timeout_sec=1.0,
                control_period_sec=0.01,
                baseline_stability_window_sec=0.04,
                baseline_stability_max_span_n=0.2,
                baseline_stability_timeout_sec=0.3,
                post_stop_confirmation_sec=0.08,
                post_stop_min_force_n=2.0,
                contact_consecutive_samples=1,
            )
        self.assertEqual(right.stop_calls, 1)
        self.assertEqual(len(right.commands), 1)

    def test_initial_right_contact_holds_for_half_second_without_more_motion(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        right = _StableContactForceRobot()
        adapter._robots = lambda: (_ForceRobot(-1.0), right)
        adapter._force_position_move_type = _ForceMoveParam
        adapter.stop_all = lambda: None

        result = adapter.execute_tool_y_force_clamp(
            ("right",),
            {"right": 2.0},
            speed_mm_s=3.0,
            max_travel_m={"right": 0.05},
            timeout_sec=2.0,
            control_period_sec=0.02,
            baseline_stability_window_sec=0.04,
            baseline_stability_max_span_n=0.2,
            baseline_stability_timeout_sec=0.3,
            post_stop_confirmation_sec=0.5,
            post_stop_min_force_n=2.0,
            contact_consecutive_samples=1,
        )
        self.assertGreaterEqual(result["elapsed_sec"], 0.5)
        self.assertGreaterEqual(result["post_stop_confirm_delta_n"], 2.0)
        self.assertEqual(right.stop_calls, 1)
        self.assertEqual(len(right.commands), 1)

    def test_initial_drag_contact_rejects_force_startup_spike(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        right = _TransientStartupForceRobot()
        adapter._robots = lambda: (_ForceRobot(-1.0), right)
        adapter._force_position_move_type = _ForceMoveParam
        adapter.stop_all = lambda: None

        with self.assertRaisesRegex(
            RealManSdkError, "initial right Tool-Y contact was not sustained"
        ):
            adapter.execute_tool_y_force_clamp(
                ("right",),
                {"right": 2.0},
                speed_mm_s=3.0,
                max_travel_m={"right": 0.05},
                timeout_sec=1.0,
                control_period_sec=0.01,
                baseline_stability_window_sec=0.04,
                baseline_stability_max_span_n=0.2,
                baseline_stability_timeout_sec=0.3,
                post_stop_confirmation_sec=0.08,
                post_stop_min_force_n=1.0,
                contact_consecutive_samples=4,
            )
        self.assertEqual(right.start_calls, 1)
        self.assertEqual(right.stop_calls, 1)
        self.assertTrue(right.commands)

    def test_initial_drag_contact_ignores_one_spike_then_reaches_real_contact(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        right = _SingleSpikeThenContactRobot()
        adapter._robots = lambda: (_ForceRobot(-1.0), right)
        adapter._force_position_move_type = _ForceMoveParam
        adapter.stop_all = lambda: None

        result = adapter.execute_tool_y_force_clamp(
            ("right",),
            {"right": 2.0},
            speed_mm_s=3.0,
            max_travel_m={"right": 0.05},
            timeout_sec=1.0,
            control_period_sec=0.01,
            baseline_stability_window_sec=0.04,
            baseline_stability_max_span_n=0.2,
            baseline_stability_timeout_sec=0.3,
            post_stop_confirmation_sec=0.08,
            post_stop_min_force_n=1.0,
            contact_consecutive_samples=4,
        )
        self.assertGreaterEqual(len(right.commands), 8)
        self.assertGreaterEqual(result["post_stop_confirm_delta_n"], 1.0)

    def test_initial_drag_contact_retries_once_from_current_state(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        right = _ContactOnSecondAttemptRobot()
        adapter._robots = lambda: (_ForceRobot(-1.0), right)
        adapter._force_position_move_type = _ForceMoveParam
        adapter.stop_all = lambda: None
        retries = []

        result = adapter.execute_tool_y_force_clamp_retry_initial_right(
            ("right",),
            {"right": 2.0},
            max_attempts=3,
            speed_mm_s=3.0,
            max_travel_m={"right": 0.05},
            timeout_sec=2.0,
            control_period_sec=0.01,
            baseline_stability_window_sec=0.04,
            baseline_stability_max_span_n=0.2,
            baseline_stability_timeout_sec=0.3,
            post_stop_confirmation_sec=0.08,
            post_stop_min_force_n=1.0,
            contact_consecutive_samples=4,
            on_retry=lambda attempt, maximum, reason: retries.append(
                (attempt, maximum, reason)
            ),
        )
        self.assertEqual(result["contact_attempts"], 2)
        self.assertEqual(right.start_calls, 2)
        self.assertEqual(right.stop_calls, 2)
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0][:2], (2, 3))
        self.assertGreaterEqual(result["post_stop_confirm_delta_n"], 1.0)

    def test_initial_drag_contact_stops_after_three_false_attempts(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        right = _TransientStartupForceRobot()
        adapter._robots = lambda: (_ForceRobot(-1.0), right)
        adapter._force_position_move_type = _ForceMoveParam
        adapter.stop_all = lambda: None

        with self.assertRaisesRegex(
            RealManSdkError, "not confirmed after 3 attempt"
        ):
            adapter.execute_tool_y_force_clamp_retry_initial_right(
                ("right",),
                {"right": 2.0},
                max_attempts=3,
                speed_mm_s=3.0,
                max_travel_m={"right": 0.05},
                timeout_sec=2.0,
                control_period_sec=0.01,
                baseline_stability_window_sec=0.04,
                baseline_stability_max_span_n=0.2,
                baseline_stability_timeout_sec=0.3,
                post_stop_confirmation_sec=0.08,
                post_stop_min_force_n=1.0,
                contact_consecutive_samples=4,
            )
        self.assertEqual(right.start_calls, 3)
        self.assertEqual(right.stop_calls, 3)

    def test_tool_y_force_control_starts_only_after_stable_baseline(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        right = _SettlingForceRobot(
            [0.0, 1.0, -1.0, 0.7, -0.8] + [0.1] * 10
        )
        adapter._robots = lambda: (_ForceRobot(-1.0), right)
        adapter._force_position_move_type = _ForceMoveParam
        adapter.stop_all = lambda: None

        result = adapter.execute_tool_y_force_clamp(
            ("right",),
            {"right": 1.0},
            speed_mm_s=1.0,
            max_travel_m={"right": 0.05},
            timeout_sec=1.0,
            control_period_sec=0.01,
            baseline_stability_window_sec=0.04,
            baseline_stability_max_span_n=0.2,
            baseline_stability_timeout_sec=0.3,
        )

        self.assertGreaterEqual(right.reads_at_start, 10)
        self.assertEqual(right.start_calls, 1)
        self.assertAlmostEqual(result["initial_tool_wrench"]["right"][1], 0.1)
        self.assertLessEqual(result["baseline_stability_spans_n"]["right"], 0.2)

    def test_tool_y_force_control_rejects_unstable_baseline_without_motion(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._connect = lambda: None
        right = _NeverStableForceRobot()
        adapter._robots = lambda: (_ForceRobot(-1.0), right)

        with self.assertRaisesRegex(
            RealManSdkError, "baselines did not stabilize"
        ):
            adapter.execute_tool_y_force_clamp(
                ("right",),
                {"right": 1.0},
                speed_mm_s=1.0,
                max_travel_m={"right": 0.05},
                timeout_sec=1.0,
                control_period_sec=0.01,
                baseline_stability_window_sec=0.02,
                baseline_stability_max_span_n=0.2,
                baseline_stability_timeout_sec=0.08,
            )
        self.assertEqual(right.start_calls, 0)
        self.assertFalse(right.commands)

    def test_offline_ik_does_not_dispatch_motion(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._sdk_lock = threading.RLock()
        adapter._connect = lambda: None
        left = _IkRobot()
        right = _IkRobot()
        adapter._robots = lambda: (left, right)
        adapter._ik_params_type = _IkParams
        adapter._activate_arm_algorithm = lambda arm, robot: None

        solution = adapter.solve_ik(
            "left",
            [0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
            [0.0] * 7,
        )

        self.assertEqual(solution, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(len(left.params), 1)
        self.assertEqual(len(right.params), 0)
        self.assertEqual(left.params[0].flag, 1)

    def test_offline_joint_path_samples_both_segments_without_motion(self):
        class InspectionRobot:
            def __init__(self):
                self.samples = []

            def rm_algo_universal_singularity_analyse(self, joints, threshold):
                self.samples.append(list(joints))
                return 0

            def rm_algo_safety_robot_self_collision_detection(self, joints):
                return 0

        adapter = object.__new__(RealManSdkAdapter)
        adapter._sdk_lock = threading.RLock()
        adapter._connect = lambda: None
        left = InspectionRobot()
        adapter._robots = lambda: (left, InspectionRobot())
        adapter._activate_arm_algorithm = lambda arm, robot: None
        limits_min = [-175.0] * 7
        limits_max = [175.0] * 7
        start = [0.0] * 7
        middle = [0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0]
        end = [12.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0]

        adapter.inspect_joint_path("left", start, middle, limits_min, limits_max)
        adapter.inspect_joint_path("left", middle, end, limits_min, limits_max)

        self.assertEqual(len(left.samples), 4 + 7)
        self.assertEqual(left.samples[0], start)
        self.assertEqual(left.samples[3], middle)
        self.assertEqual(left.samples[-1], end)

    def test_offline_joint_path_rejects_singularity_before_dispatch(self):
        class InspectionRobot:
            def rm_algo_universal_singularity_analyse(self, joints, threshold):
                return -1 if joints[3] <= -2.0 else 0

            def rm_algo_safety_robot_self_collision_detection(self, joints):
                return 0

        adapter = object.__new__(RealManSdkAdapter)
        adapter._sdk_lock = threading.RLock()
        adapter._connect = lambda: None
        adapter._robots = lambda: (InspectionRobot(), InspectionRobot())
        adapter._activate_arm_algorithm = lambda arm, robot: None

        with self.assertRaisesRegex(RealManSdkError, "singularity check at sample=1/1"):
            adapter.inspect_joint_path(
                "left", [0.0] * 7, [0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
                [-175.0] * 7, [175.0] * 7,
            )

    def test_continuous_ik_chains_each_solution_without_motion(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._sdk_lock = threading.RLock()
        adapter._connect = lambda: None
        left = _ContinuousIkRobot()
        right = _ContinuousIkRobot()
        adapter._robots = lambda: (left, right)
        adapter._activate_arm_algorithm = lambda arm, robot: None
        adapter._ik_matrix_type = _RemoteMatrix

        solutions = adapter.solve_continuous_ik(
            "left",
            [
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [0.0] * 7,
            0.5,
        )

        self.assertEqual(left.initializations, [(0.5, 1)])
        self.assertEqual(left.seeds, [[0.0] * 7, [1.0] * 7])
        self.assertEqual(solutions, [[1.0] * 7, [2.0] * 7])
        self.assertEqual(right.initializations, [])
    def test_connected_waypoints_accepts_and_runs_progress_callback(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        progress_samples = []
        target = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
        result = adapter.execute_dual_movel_connected_waypoints(
            [target],
            [target],
            5.0,
            5.0,
            progress_callback=lambda: progress_samples.append(True),
        )

        self.assertTrue(progress_samples)
        self.assertIn("connected", result)

    def test_connected_waypoints_allow_successful_progress_stop(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        stop_calls = []
        abort_calls = []
        adapter.stop_all = lambda: stop_calls.append(True)

        target = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
        result = adapter.execute_dual_movel_connected_waypoints(
            [target],
            [target],
            5.0,
            5.0,
            progress_callback=lambda: False,
            abort_callback=lambda: abort_calls.append(True),
        )

        self.assertTrue(stop_calls)
        self.assertEqual(abort_calls, [True])
        self.assertIn("stopped_by_progress_callback=true", result)

    def test_endpoint_movej_p_dispatches_both_arm_commands(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        target = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
        result = adapter.execute_dual_movel_endpoint(
            target,
            target,
            5.0,
            5.0,
            motion_mode="movej_p",
        )

        self.assertEqual(robot.calls, ["movej_p", "movej_p"])
        self.assertIn("movej_p", result)

    def test_connected_movej_p_dispatches_final_waypoint(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        target = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
        result = adapter.execute_dual_movel_connected_waypoints(
            [target],
            [target],
            5.0,
            5.0,
            motion_mode="movej_p",
        )

        self.assertEqual(robot.calls, ["movej_p", "movej_p"])
        self.assertIn("movej_p", result)

    def test_connected_movel_offset_dispatches_work_frame_waypoints(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        offsets = [
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        result = adapter.execute_dual_movel_connected_waypoints(
            offsets,
            offsets,
            12.0,
            12.0,
            motion_mode="movel_offset",
            offset_frame_type=0,
        )

        calls = [call for call in robot.calls if call[0] == "movel_offset"]
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call[1][4] == 0 for call in calls))
        self.assertIn("movel_offset", result)

    def test_dual_movej_dispatches_both_arms(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        joints = [0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = adapter.execute_dual_movej(joints, joints, 12.0)

        calls = [call for call in robot.calls if call[0] == "movej"]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[1][0][1] == 60.0 for call in calls))
        self.assertIn("dual-arm movej", result)

    def test_mission_movej_uses_ros_transport_not_sdk(self):
        adapter = object.__new__(RealManSdkAdapter)
        adapter._motion_lock = threading.Lock()
        adapter._motion_active = False
        connect_calls = []
        adapter._connect = lambda: connect_calls.append(True)
        calls = []
        adapter._ros_movej_transport = SimpleNamespace(
            execute=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or "ROS MoveJ completed"
            )
        )
        result = adapter.execute_single_movej(
            "left", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 13.0
        )
        self.assertEqual(result, "ROS MoveJ completed")
        self.assertEqual(len(connect_calls), 1)
        self.assertEqual(calls[0][0][0], {"left": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        self.assertEqual(calls[0][0][1], 13)
        self.assertFalse(adapter._motion_active)

    def test_single_movel_offset_uses_work_frame(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_arm = lambda _arm: None

        offset = [0.01, -0.02, 0.03, 0.0, 0.0, 0.0]
        result = adapter.execute_single(
            "right",
            offset,
            "movel_offset",
            12.0,
            True,
            offset_frame_type=0,
        )

        name, args = robot.calls[0]
        self.assertEqual(name, "movel_offset")
        self.assertEqual(args, (offset, 12, 0, 0, 0, 1))
        self.assertIn("movel_offset", result)

    def test_dual_movel_offset_dispatches_both_arms(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        offset = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = adapter.execute_dual(
            offset,
            offset,
            "movel_offset",
            10.0,
            True,
            offset_frame_type=0,
        )

        self.assertEqual([call[0] for call in robot.calls], ["movel_offset"] * 2)
        self.assertIn("movel_offset", result)


if __name__ == "__main__":
    unittest.main()
