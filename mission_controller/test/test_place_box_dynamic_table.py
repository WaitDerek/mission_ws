import math
import threading
import time
import unittest

from mission_runtime.box_carry import BoxCarryMixin
from mission_runtime.box_support import BoxSupportMixin
from mission_runtime.common import MissionError


class TestDynamicTablePlacement(unittest.TestCase):
    def test_descent_uses_fixed_origin_after_small_movel_overshoot(self):
        class FakePlace(BoxCarryMixin):
            def __init__(self):
                self.measured = ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))

            def _string(self, name):
                return name

            def _float(self, name):
                if name == 'place_box_test_descent_work_y_step_m':
                    return 0.0
                if name == 'place_box_test_descent_work_y_max_travel_m':
                    return 0.1
                assert name == "place_box_test_timeout_sec"
                return 10.0

            def _publish_place_box_test_feedback(self, *_args):
                pass

            def _boolean(self, name):
                assert name == "place_box_test_descent_y_search_enabled"
                return False

            def _lookup_tf_carry_transform(self, _base, _frame):
                if _frame.endswith("_link8_frame"):
                    return self.measured
                return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

            def _wait_for_place_box_test_world_targets(self, _goal, _targets):
                raise AssertionError("descent must not wait for geometric convergence")

            def _place_box_test_verify_descent_z(self, _goal, _adapter, _base, _ref, _targets):
                return dict.fromkeys(('left', 'right'), self.measured)

            def _place_box_test_current_box(self, _base, _relations):
                raise AssertionError("descent must not run rigid-grasp target checks")

        class FakeAdapter:
            def __init__(self, place):
                self.place = place
                self.commanded_z = []

            def execute_dual_movel_endpoint(self, left, right, *_args, **_kwargs):
                self.commanded_z.append(left[2])
                self.place.measured = (
                    (0.0, 0.0, left[2] - (0.0025 if len(self.commanded_z) == 1 else 0.0)),
                    (0.0, 0.0, 0.0, 1.0),
                )

        place = FakePlace()
        adapter = FakeAdapter(place)
        relations = {
            arm: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            for arm in ("left", "right")
        }
        reference, _ = place._place_box_test_capture_descent_reference("base_link", relations)
        reference.maximum_drop_m = 0.1
        first = place._place_box_test_move_box_z(
            None, adapter, "base_link", relations, reference, 0.005, 5.0
        )
        self.assertAlmostEqual(first[0][2], 0.9925)
        second = place._place_box_test_move_box_z(
            None, adapter, "base_link", relations, reference, 0.010, 5.0
        )
        self.assertAlmostEqual(adapter.commanded_z[1], 0.9875)
        self.assertAlmostEqual(second[0][2], 0.9875)

    def test_descent_preserves_actual_unequal_heights_and_orientations(self):
        from mission_runtime.realman_sdk_adapter import pose_to_sdk_target

        identity = (0.0, 0.0, 0.0, 1.0)
        tilt = (math.sin(0.1), 0.0, 0.0, math.cos(0.1))

        class FakePlace(BoxCarryMixin):
            actual = {
                "left": ((0.20, 0.45, 0.80), tilt),
                "right": ((0.22, -0.44, 0.84), identity),
            }
            # Work Z is intentionally not aligned with common base_link Z.
            bases = {
                "left": ((0.0, 0.4, 0.3), tilt),
                "right": ((0.0, -0.4, 0.3), tilt),
            }

            def _string(self, name):
                return name

            def _float(self, name):
                if name == 'place_box_test_descent_work_y_step_m':
                    return 0.0
                if name == 'place_box_test_descent_work_y_max_travel_m':
                    return 0.1
                assert name == "place_box_test_timeout_sec"
                return 10.0

            def _publish_place_box_test_feedback(self, *_args):
                pass

            def _lookup_tf_carry_transform(self, _base, frame):
                arm = frame.split("_")[0]
                return self.bases[arm] if frame.endswith("_arm_base_frame") else self.actual[arm]

            def _wait_for_place_box_test_world_targets(self, *_args):
                raise AssertionError("unexpected geometric target wait")

            def _place_box_test_verify_descent_z(self, _goal, _adapter, _base, _ref, _targets):
                return self.actual

            def _place_box_test_current_box(self, *_args):
                raise AssertionError("unexpected rigid-grasp consistency check")

        class FakeAdapter:
            calls = []

            def execute_dual_movel_endpoint(self, left, right, *_args, **_kwargs):
                self.calls.append((left, right))

        place, adapter = FakePlace(), FakeAdapter()
        place._boolean = lambda _name: False
        relations = {arm: ((0.0, 0.0, 0.0), identity) for arm in ("left", "right")}
        reference, _ = place._place_box_test_capture_descent_reference("base_link", relations)
        reference.maximum_drop_m = 0.1
        place._place_box_test_move_box_z(
            None, adapter, "base_link", relations,
            reference, 0.005, 5.0,
        )
        self.assertEqual(len(adapter.calls), 1)
        for arm, target in zip(("left", "right"), adapter.calls[0]):
            original = place.actual[arm]
            initial_local = BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(place.bases[arm]), original)
            expected_local = ((*initial_local[0][:2], initial_local[0][2] - 0.005), initial_local[1])
            expected = pose_to_sdk_target(place._endpoint_sync_transform_to_pose(expected_local))
            for value, expected_value in zip(target, expected):
                self.assertAlmostEqual(value, expected_value)

    def test_descent_rejects_upward_or_invalid_step_before_commanding(self):
        from unittest.mock import Mock

        for step in (-0.005, 0.0, float('nan'), float('inf')):
            adapter = Mock()
            with self.assertRaisesRegex(MissionError, "positive cumulative"):
                BoxCarryMixin._place_box_test_move_box_z(
                    object(), None, adapter, "base_link", {}, None, step, 5.0
                )
            adapter.execute_dual_movel_endpoint.assert_not_called()

    def test_box_bottom_and_common_vertical_shift(self):
        half = math.sqrt(0.5)
        box = ((1.0, 2.0, 0.95), (0.0, -half, 0.0, half))
        self.assertAlmostEqual(
            BoxCarryMixin._place_box_test_bottom_z(box, 0.14), 0.81
        )
        raised = BoxCarryMixin._place_box_test_shift_box_z(box, 0.06)
        self.assertEqual(raised[0][:2], box[0][:2])
        self.assertEqual(raised[1], box[1])
        self.assertAlmostEqual(
            BoxCarryMixin._place_box_test_bottom_z(raised, 0.14), 0.87
        )

    def test_non_upright_box_fails_closed(self):
        with self.assertRaises(MissionError):
            BoxCarryMixin._place_box_test_bottom_z(
                ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)), 0.14
            )

    def test_both_arm_targets_keep_measured_grasp_span(self):
        half = math.sqrt(0.5)
        box = ((0.0, 0.0, 1.0), (0.0, -half, 0.0, half))
        relation = {
            "left": ((0.0, 0.0, -0.5), (0.0, 0.0, 0.0, 1.0)),
            "right": ((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
        }
        before = {
            arm: BoxSupportMixin._compose_transform(box, relation[arm])
            for arm in relation
        }
        lowered = BoxCarryMixin._place_box_test_shift_box_z(box, -0.2)
        after = {
            arm: BoxSupportMixin._compose_transform(lowered, relation[arm])
            for arm in relation
        }
        for arm in relation:
            self.assertAlmostEqual(after[arm][0][2] - before[arm][0][2], -0.2)
        span_before = [before["right"][0][i] - before["left"][0][i]
                       for i in range(3)]
        span_after = [after["right"][0][i] - after["left"][0][i]
                      for i in range(3)]
        for actual, expected in zip(span_after, span_before):
            self.assertAlmostEqual(actual, expected)

    def test_waist_path_search_moves_one_box_in_y_and_keeps_grasp(self):
        class FakePlace(BoxCarryMixin):
            def __init__(self):
                self.joint_state_lock = threading.RLock()
                self.latest_slave_arm_positions = {
                    "left": [0.0] * 7, "right": [0.0] * 7,
                }
                self.latest_slave_arm_state_times = {
                    "left": time.monotonic(), "right": time.monotonic(),
                }
                self.params = {
                    "place_box_test_waist_workspace_enabled": True,
                    "place_box_test_waist_y_search_half_range_m": 0.10,
                    "place_box_test_waist_y_search_step_m": 0.02,
                    "place_box_test_waist_z_max_drop_m": 0.08,
                    "place_box_test_waist_z_search_step_m": 0.04,
                    "place_box_test_waist_ik_min_margin_deg": 2.0,
                    "place_box_test_waist_ik_max_step_deg": 75.0,
                    "place_box_test_waist_clearance_m": 0.04,
                    "box_pre_target_arm_movej_feedback_max_age_sec": 2.0,
                    "box_ik_joint4_negative_required": True,
                    "waist_workspace_left_arm_joint_min_deg": [-175.0] * 7,
                    "waist_workspace_left_arm_joint_max_deg": [175.0] * 7,
                    "waist_workspace_right_arm_joint_min_deg": [-175.0] * 7,
                    "waist_workspace_right_arm_joint_max_deg": [175.0] * 7,
                }
                self.feedback = []

            def _float(self, name):
                return float(self.params[name])

            def _boolean(self, name):
                return bool(self.params[name])

            def _float_array(self, name):
                return self.params[name]

            def _check_canceled(self, goal_handle, detail):
                return None

            def _publish_place_box_test_feedback(self, goal_handle, stage, detail):
                self.feedback.append((stage, detail))

            def _joint123_arm_base_transform(self, arm, angles):
                # The synthetic waist bend moves each arm base in common +Y.
                return ((0.0, -0.10 * angles[1], 0.0), (0.0, 0.0, 0.0, 1.0))

        class FakeAdapter:
            def solve_ik(self, arm, target, seed):
                if target[1] < -0.08:
                    return None
                return [0.0, 20.0, 0.0, -40.0, 0.0, 0.0, 0.0]

        place = FakePlace()
        half = math.sqrt(0.5)
        current_box = ((0.0, 0.0, 1.0), (0.0, -half, 0.0, half))
        relations = {
            "left": ((0.0, 0.0, -0.4), (0.0, 0.0, 0.0, 1.0)),
            "right": ((0.0, 0.0, 0.4), (0.0, 0.0, 0.0, 1.0)),
        }
        kwargs = dict(
            goal_handle=None, adapter=FakeAdapter(), current_box=current_box,
            relations=relations, body_start=[0.0] * 4,
            target_units=[-40000, -60000, -40000, 0],
            units_per_degree=[1000.0] * 4,
            live_arm_base={
                "left": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                "right": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            },
            segments=6, table_z=0.72, half_height=0.14,
        )
        selected, detail = place._place_box_test_optimize_waist_box_path(**kwargs)
        self.assertGreater(selected[0][1], 0.0)
        self.assertIn("ik_points=6x2", detail)
        self.assertGreaterEqual(
            place._place_box_test_bottom_z(selected, 0.14), 0.76 - 1e-9
        )
        body, arm, transforms, world = place._place_box_test_waist_path(
            current_box, selected, relations, kwargs["body_start"],
            kwargs["target_units"], kwargs["units_per_degree"],
            kwargs["live_arm_base"], kwargs["segments"],
        )
        self.assertEqual(len(body), 6)
        self.assertEqual(len(arm["left"]), 6)
        for waypoint in world:
            span = [
                waypoint["right"][0][axis] - waypoint["left"][0][axis]
                for axis in range(3)
            ]
            self.assertAlmostEqual(sum(value * value for value in span), 0.64)

        class NoSolution:
            def solve_ik(self, arm, target, seed):
                return None

        kwargs["adapter"] = NoSolution()
        with self.assertRaisesRegex(MissionError, "before motion"):
            place._place_box_test_optimize_waist_box_path(**kwargs)

    def test_table_early_contact_tolerance_is_fifty_centimeters(self):
        class FakePlace(BoxCarryMixin):
            def _place_box_test_wait_arms_still(self, _goal):
                pass

            def _place_box_test_capture_descent_reference(self, _frame, _relations):
                from mission_runtime.box_placement import _PlaceDescentReference
                return _PlaceDescentReference(dict.fromkeys(('left', 'right'), self.start_box), {}), self.start_box

            def _float(self, name):
                return {
                    "place_box_test_table_support_delta_fz_n": 3.0,
                    "place_box_test_force_unload_baseline_timeout_sec": 3.0,
                    "place_box_test_table_unloaded_abs_fz_n": 5.0,
                    "place_box_test_table_support_sign_left": 1.0,
                    "place_box_test_table_support_sign_right": 1.0,
                    "place_box_test_table_max_overtravel_m": 0.005,
                    "place_box_test_table_early_contact_tolerance_m": 0.500,
                    "place_box_test_table_fine_distance_m": 0.030,
                    "place_box_test_table_coarse_step_m": 0.005,
                    "place_box_test_table_fine_step_m": 0.001,
                    "place_box_test_table_descent_velocity_percent": 5.0,
                    "place_box_test_timeout_sec": 10.0,
                }[name]

            def _check_canceled(self, *_args):
                pass

            def _publish_place_box_test_feedback(self, *_args):
                pass

            def _place_box_test_move_box_z(
                self, goal_handle, adapter, base_frame, relations, reference,
                descent_m, speed,
            ):
                reference.commanded_drop_m = descent_m
                return self._place_box_test_shift_box_z(reference.tcp_by_arm['left'], -descent_m)

        class FakeAdapter:
            def __init__(self):
                self.samples = 0

            def read_bilateral_work_fz(self):
                self.samples += 1
                force = 0.0 if self.samples <= 8 else 4.0
                return {"left": force, "right": force}

        place = FakePlace()
        upright = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))
        place.start_box = ((0.0, 0.0, 1.235), upright)
        with self.assertRaisesRegex(MissionError, "tolerance=0.5000m"):
            place._place_box_test_descend_to_table_segmented(
                None, FakeAdapter(), "base_link", {},
                ((0.0, 0.0, 1.235), upright), 0.72, 0.0,
            )
        place.start_box = ((0.0, 0.0, 0.765), upright)
        _, detail = place._place_box_test_descend_to_table_segmented(
            None, FakeAdapter(), "base_link", {},
            ((0.0, 0.0, 0.765), upright), 0.72, 0.0,
        )
        self.assertIn("table support confirmed", detail)

    def test_unilateral_force_continues_until_both_sides_are_stable(self):
        class FakePlace(BoxCarryMixin):
            def __init__(self):
                self.moves = 0
                self.feedback = []
                self.start_z = 0.760

            def _place_box_test_wait_arms_still(self, _goal):
                pass

            def _place_box_test_capture_descent_reference(self, _frame, _relations):
                from mission_runtime.box_placement import _PlaceDescentReference
                box = ((0.0, 0.0, self.start_z), (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5)))
                return _PlaceDescentReference(dict.fromkeys(('left', 'right'), box), {}), box

            def _float(self, name):
                return {
                    "place_box_test_table_support_delta_fz_n": 3.0,
                    "place_box_test_force_unload_baseline_timeout_sec": 3.0,
                    "place_box_test_table_unloaded_abs_fz_n": 5.0,
                    "place_box_test_table_support_sign_left": 1.0,
                    "place_box_test_table_support_sign_right": 1.0,
                    "place_box_test_table_max_overtravel_m": 0.005,
                    "place_box_test_table_early_contact_tolerance_m": 0.500,
                    "place_box_test_table_fine_distance_m": 0.030,
                    "place_box_test_table_coarse_step_m": 0.005,
                    "place_box_test_table_fine_step_m": 0.001,
                    "place_box_test_table_descent_velocity_percent": 5.0,
                    "place_box_test_timeout_sec": 10.0,
                }[name]

            def _check_canceled(self, *_args):
                pass

            def _publish_place_box_test_feedback(self, _goal, stage, detail):
                self.feedback.append((stage, detail))

            def _place_box_test_move_box_z(
                self, goal_handle, adapter, base_frame, relations, reference,
                descent_m, speed,
            ):
                self.moves += 1
                reference.commanded_drop_m = descent_m
                return self._place_box_test_shift_box_z(reference.tcp_by_arm['left'], -descent_m)

        class FakeAdapter:
            def __init__(self):
                self.samples = 0

            def read_bilateral_work_fz(self):
                self.samples += 1
                if self.samples <= 8:
                    return {"left": 0.0, "right": 0.0}
                if self.samples <= 13:
                    return {"left": 0.0, "right": 4.0}
                return {"left": 4.0, "right": 4.0}

        place = FakePlace()
        upright = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))
        _, detail = place._place_box_test_descend_to_table_segmented(
            None, FakeAdapter(), "base_link", {},
            ((0.0, 0.0, 0.760), upright), 0.72, 0.0,
        )
        self.assertGreaterEqual(place.moves, 6)
        self.assertIn("table support confirmed", detail)
        self.assertEqual(place.feedback[0][0], "PLACE_DESCENT_BASELINE")
        self.assertEqual(place.feedback[1][0], "PLACE_DESCENT_REFERENCE")
        self.assertEqual(place.feedback[2][0], "PLACE_TABLE_UNILATERAL_FORCE")
        self.assertEqual(place.feedback[-1][0], "TABLE_SUPPORT_CONFIRMED")

        class NeverBilateral(FakeAdapter):
            def read_bilateral_work_fz(self):
                self.samples += 1
                if self.samples <= 8:
                    return {"left": 0.0, "right": 0.0}
                return {"left": 0.0, "right": 4.0}

        never_place = FakePlace()
        never_place.start_z = 0.725
        with self.assertRaisesRegex(MissionError, "table-height limit"):
            never_place._place_box_test_descend_to_table_segmented(
                None, NeverBilateral(), "base_link", {},
                ((0.0, 0.0, 0.725), upright), 0.72, 0.0,
            )

    def test_post_support_descent_uses_each_arm_base_negative_z(self):
        class FakeGoal:
            is_cancel_requested = False

        class FakePlace(BoxCarryMixin):
            def __init__(self):
                self.bases = {
                    "left": ((0.0, 0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
                    "right": ((0.0, -0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
                }
                self.actual = {
                    arm: ((base[0][0], base[0][1], 1.0), base[1])
                    for arm, base in self.bases.items()
                }
                self.feedback = []

            def _float(self, name):
                return {
                    "place_box_test_post_support_arm_base_descent_m": 0.030,
                    "place_box_test_table_coarse_step_m": 0.005,
                    "place_box_test_table_descent_velocity_percent": 5.0,
                    "place_box_test_table_unloaded_abs_fz_n": 5.0,
                    "place_box_test_post_support_max_abs_work_fz_n": 200.0,
                    "place_box_test_position_tolerance_m": 0.005,
                    "place_box_test_timeout_sec": 10.0,
                }[name]

            def _string(self, name):
                return name

            def _lookup_tf_carry_transform(self, _base, frame):
                if frame.endswith("_arm_base_frame"):
                    return self.bases[frame.split("_")[0]]
                return self.actual[frame.split("_")[0]]

            def _check_canceled(self, *_args):
                pass

            def _publish_place_box_test_feedback(self, _goal, stage, detail):
                self.feedback.append((stage, detail))

            def _wait_for_place_box_test_world_targets(self, _goal, targets):
                for arm in ("left", "right"):
                    self.assert_target(arm, targets[arm])

            def _place_box_test_verify_descent_z(self, _goal, _adapter, _base, ref, targets):
                from mission_runtime.place_descent_z_guard import DescentZSettings, evaluate_work_z
                passed, detail = evaluate_work_z(
                    ref.base_by_arm, ref.tcp_by_arm, targets, self.actual,
                    DescentZSettings(0.005, 0.005, 1.0, 0.25, 3),
                )
                assert passed, detail
                self.feedback.append(('PLACE_DESCENT_Z_VERIFIED', detail))
                return self.actual

            def assert_target(self, arm, target):
                if any(
                    abs(actual - expected) > 1e-9
                    for actual, expected in zip(self.actual[arm][0], target[0])
                ):
                    raise AssertionError(f"{arm} did not reach target")

        class FakeAdapter:
            def __init__(self, place):
                self.place = place
                self.calls = []

            def execute_dual_movel_endpoint(
                self, left, right, left_speed, right_speed, **_kwargs
            ):
                self.calls.append((list(left), list(right)))
                for arm, target in (("left", left), ("right", right)):
                    base = self.place.bases[arm]
                    self.place.actual[arm] = (
                        (base[0][0] + target[0], base[0][1] + target[1],
                         base[0][2] + target[2]),
                        (0.0, 0.0, 0.0, 1.0),
                    )

            def read_bilateral_work_fz(self):
                return {"left": 0.0, "right": 0.0}

        place = FakePlace()
        adapter = FakeAdapter(place)
        final_pose, detail = place._place_box_test_post_support_arm_base_descent(
            FakeGoal(), adapter, "base_link"
        )
        self.assertEqual(len(adapter.calls), 6)
        self.assertEqual(sum(stage == 'PLACE_DESCENT_Z_VERIFIED' for stage, _ in place.feedback), 6)
        self.assertAlmostEqual(final_pose["left"].position.z, 0.970)
        self.assertAlmostEqual(final_pose["right"].position.z, 0.970)
        self.assertAlmostEqual(final_pose["left"].position.y, 0.0)
        self.assertAlmostEqual(final_pose["right"].position.y, 0.0)
        self.assertAlmostEqual(place.actual["left"][0][1], 0.5)
        self.assertAlmostEqual(place.actual["right"][0][1], -0.5)
        self.assertIn("0.0300m", detail)

        class ModerateForceAdapter(FakeAdapter):
            def read_bilateral_work_fz(self):
                return {"left": 0.0, "right": 6.0}

        moderate_place = FakePlace()
        moderate_adapter = ModerateForceAdapter(moderate_place)
        moderate_place._place_box_test_post_support_arm_base_descent(
            FakeGoal(), moderate_adapter, "base_link"
        )
        self.assertEqual(len(moderate_adapter.calls), 6)

        class HighForceAdapter(FakeAdapter):
            def read_bilateral_work_fz(self):
                return {"left": 0.0, "right": 201.0}

        overloaded_place = FakePlace()
        overloaded_adapter = HighForceAdapter(overloaded_place)
        with self.assertRaisesRegex(MissionError, "stopping before release"):
            overloaded_place._place_box_test_post_support_arm_base_descent(
                FakeGoal(), overloaded_adapter, "base_link"
            )
        self.assertEqual(len(overloaded_adapter.calls), 1)


if __name__ == "__main__":
    unittest.main()
