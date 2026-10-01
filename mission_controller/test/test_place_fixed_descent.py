"""Fixed-origin placement regressions; all motion and sensor interfaces mocked."""
import math
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mission_runtime.box_carry import BoxCarryMixin
from mission_runtime.box_support import BoxSupportMixin
from mission_runtime.common import MissionError


IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


class FixedDescent(BoxCarryMixin):
    def __init__(self):
        self.actual = {
            'left': ((0.1, 0.4, 1.0), IDENTITY[1]),
            'right': ((0.2, -0.4, 1.02), IDENTITY[1]),
        }
        self.feedback = []
        self.joint_state_lock = threading.Lock()
        self.latest_slave_arm_positions = {'left': [0.1] * 7, 'right': [-0.1] * 7}
        self.latest_slave_arm_state_sequences = {'left': 31, 'right': 40}
        self._wait_for_post_arm_joint_targets = Mock()

    def _string(self, name):
        return name

    def _check_canceled(self, *_args):
        pass

    def _boolean(self, name):
        assert name == 'place_box_test_descent_y_search_enabled'
        return False

    def _float(self, name):
        if name == 'place_box_test_descent_work_y_step_m':
            return 0.0  # Z-only geometry tests; explicit Y tests below.
        if name == 'place_box_test_descent_work_y_max_travel_m':
            return 0.1
        assert name == 'place_box_test_timeout_sec'
        return 10.0

    def _lookup_tf_carry_transform(self, _base, frame):
        return IDENTITY if frame.endswith('_arm_base_frame') else self.actual[frame.split('_')[0]]

    def _publish_place_box_test_feedback(self, _goal, stage, detail):
        self.feedback.append((stage, detail))

    def _place_box_test_verify_descent_z(self, _goal, _adapter, _base, _ref, _targets):
        # Geometry-only fixture. The real guard has dedicated behavior/TF tests.
        return self.actual

    def _place_box_test_capture_descent_reference(self, base, relations):
        reference, box = super()._place_box_test_capture_descent_reference(base, relations)
        reference.maximum_drop_m = 0.1
        return reference, box


def test_asymmetric_tracking_drift_is_not_new_origin():
    place, adapter = FixedDescent(), Mock()
    relations = dict.fromkeys(('left', 'right'), IDENTITY)
    reference, _ = place._place_box_test_capture_descent_reference('base_link', relations)
    goal = SimpleNamespace(is_cancel_requested=False)
    place._place_box_test_move_box_z(goal, adapter, 'base_link', relations, reference, 0.005, 5.0)
    # Right overshot by 2.5mm, left by 0.5mm; XY/orientation also noisy.
    tilt = (math.sin(0.002), 0.0, 0.0, math.cos(0.002))
    place.actual = {
        'left': ((0.101, 0.399, 0.9945), tilt),
        'right': ((0.198, -0.402, 1.0125), tilt),
    }
    place._place_box_test_move_box_z(goal, adapter, 'base_link', relations, reference, 0.010, 5.0)
    left, right, lspeed, rspeed = adapter.execute_dual_movel_endpoint.call_args.args
    assert left[:3] == pytest.approx([0.1, 0.4, 0.9875])
    assert right[:3] == pytest.approx([0.2, -0.4, 1.0075])
    assert left[3:] == pytest.approx([0.0, 0.0, 0.0])
    assert right[3:] == pytest.approx([0.0, 0.0, 0.0])
    assert lspeed == rspeed == 5.0
    assert 'progress_callback' not in adapter.execute_dual_movel_endpoint.call_args.kwargs
    adapter.read_bilateral_work_fz.assert_not_called()
    assert 'common_commanded_drop=0.012500m' in place.feedback[-1][1]
    assert 'drop_difference=+0.002000m' in place.feedback[-1][1]
    assert reference.tcp_by_arm['right'][0][2] == 1.02


def test_reported_5789um_overshoot_advances_both_without_upward_command():
    place, adapter = FixedDescent(), Mock()
    relations = dict.fromkeys(('left', 'right'), IDENTITY)
    reference, _ = place._place_box_test_capture_descent_reference('base_link', relations)
    reference.commanded_drop_m = 0.015
    place.actual['left'] = ((0.1, 0.4, 1.0 - 0.018875), IDENTITY[1])
    place.actual['right'] = ((0.2, -0.4, 1.02 - 0.020789), IDENTITY[1])
    place._place_box_test_move_box_z(None, adapter, 'base_link', relations, reference, 0.020, 5.0)
    left, right, *_ = adapter.execute_dual_movel_endpoint.call_args.args
    assert reference.commanded_drop_m == pytest.approx(0.025789)
    assert left[2] < place.actual['left'][0][2]
    assert right[2] == pytest.approx(place.actual['right'][0][2] - 0.005)
    assert any(s == 'PLACE_DESCENT_PROGRESS_REBASED' for s, _ in place.feedback)


def test_progress_near_fixed_limit_clips_then_refuses_more_motion():
    place, adapter = FixedDescent(), Mock()
    relations = dict.fromkeys(('left', 'right'), IDENTITY)
    reference, _ = place._place_box_test_capture_descent_reference('base_link', relations)
    reference.commanded_drop_m = 0.090
    place.actual['left'] = ((0.1, 0.4, 0.903), IDENTITY[1])
    place.actual['right'] = ((0.2, -0.4, 0.922), IDENTITY[1])
    place._place_box_test_move_box_z(None, adapter, 'base_link', relations, reference, 0.095, 5.0)
    assert reference.commanded_drop_m == pytest.approx(0.100)
    assert adapter.execute_dual_movel_endpoint.call_args.args[1][2] == pytest.approx(0.920)
    adapter.reset_mock()
    with pytest.raises(MissionError, match='fixed travel limit'):
        place._place_box_test_move_box_z(None, adapter, 'base_link', relations, reference, 0.105, 5.0)
    adapter.execute_dual_movel_endpoint.assert_not_called()


def test_repeated_overshoot_updates_progress_but_never_reanchors_tcp():
    place, adapter = FixedDescent(), Mock()
    relations = dict.fromkeys(('left', 'right'), IDENTITY)
    reference, _ = place._place_box_test_capture_descent_reference('base_link', relations)
    for expected in (0.005, 0.016, 0.027):
        place._place_box_test_move_box_z(None, adapter, 'base_link', relations, reference,
                                       reference.commanded_drop_m + 0.005, 5.0)
        assert reference.commanded_drop_m == pytest.approx(expected)
        for arm in ('left', 'right'):
            origin = reference.tcp_by_arm[arm]
            place.actual[arm] = ((origin[0][0], origin[0][1], origin[0][2] - expected - 0.006), origin[1])
    assert reference.tcp_by_arm['left'][0][2] == 1.0
    assert reference.maximum_drop_m == 0.1


def test_sdk_failure_prevents_completion_report():
    place, adapter = FixedDescent(), Mock()
    relations = dict.fromkeys(('left', 'right'), IDENTITY)
    reference, _ = place._place_box_test_capture_descent_reference('base_link', relations)
    adapter.execute_dual_movel_endpoint.side_effect = RuntimeError('right failed')
    with pytest.raises(RuntimeError, match='right failed'):
        place._place_box_test_move_box_z(None, adapter, 'base_link', relations, reference, 0.005, 5.0)
    assert place.feedback == []
    assert reference.commanded_drop_m == 0.0
    assert reference.y_offset_m == 0.0


@pytest.mark.parametrize('limit', [0.0, -0.1, float('nan'), float('inf')])
def test_invalid_travel_budget_never_commands_motion(limit):
    place, adapter = FixedDescent(), Mock()
    relations = dict.fromkeys(('left', 'right'), IDENTITY)
    reference, _ = place._place_box_test_capture_descent_reference('base_link', relations)
    reference.maximum_drop_m = limit
    with pytest.raises(MissionError, match='fixed travel limit'):
        place._place_box_test_move_box_z(None, adapter, 'base_link', relations, reference, 0.005, 5.0)
    adapter.execute_dual_movel_endpoint.assert_not_called()


def test_progress_uses_own_work_z_not_world_z_with_y_displacement():
    from mission_runtime.common import rotate_vector
    place, adapter = FixedDescent(), Mock()
    relations = dict.fromkeys(('left', 'right'), IDENTITY)
    reference, _ = place._place_box_test_capture_descent_reference('base_link', relations)
    reference.commanded_drop_m = 0.015
    for arm in ('left', 'right'):
        q = (math.sin(0.2), 0.0, 0.0, math.cos(0.2))
        reference.base_by_arm[arm] = (IDENTITY[0], q)
        origin = reference.tcp_by_arm[arm]
        delta = rotate_vector((0.0, 0.03, -0.021), q)
        place.actual[arm] = (tuple(v + d for v, d in zip(origin[0], delta)), origin[1])
    place._place_box_test_move_box_z(None, adapter, 'base_link', relations, reference, 0.020, 5.0)
    assert reference.commanded_drop_m == pytest.approx(0.026)


def test_joint_stability_check_only_observes_current_angles():
    place = FixedDescent()
    place._place_box_test_wait_arms_still(None)
    args = place._wait_for_post_arm_joint_targets.call_args.args
    assert args[1] == (0.1,) * 7
    assert args[2] == (-0.1,) * 7
    assert args[3] == {'left': 31, 'right': 40}


@pytest.mark.parametrize('q', [[], [float('nan')] * 7])
def test_bad_joint_feedback_rejected_before_descent(q):
    place = FixedDescent()
    place.latest_slave_arm_positions['left'] = q
    with pytest.raises(MissionError, match='valid dual-arm joint feedback'):
        place._place_box_test_wait_arms_still(None)
    place._wait_for_post_arm_joint_targets.assert_not_called()


def test_direct_movel_does_not_call_ik_even_if_legacy_search_enabled():
    place, adapter = FixedDescent(), Mock()
    place._boolean = lambda name: True
    adapter.solve_ik.side_effect = AssertionError('offline IK must not run')
    relations = dict.fromkeys(('left', 'right'), IDENTITY)
    reference, _ = place._place_box_test_capture_descent_reference('base_link', relations)
    place._place_box_test_move_box_z(None, adapter, 'base_link', relations, reference, 0.005, 5.0)
    left, right, *_ = adapter.execute_dual_movel_endpoint.call_args.args
    assert left[:3] == pytest.approx([0.1, 0.4, 0.995])
    assert right[:3] == pytest.approx([0.2, -0.4, 1.015])
    assert right[1] - left[1] == pytest.approx(-0.8)
    adapter.read_bilateral_work_fz.assert_not_called()
    adapter.solve_ik.assert_not_called()


def test_y_two_mm_per_segment_is_absolute_capped_and_frozen_after_contact():
    place, adapter = FixedDescent(), Mock()
    values = {'place_box_test_timeout_sec': 10.0,
              'place_box_test_descent_work_y_step_m': 0.002,
              'place_box_test_descent_work_y_max_travel_m': 0.005}
    place._float = values.__getitem__
    relations = dict.fromkeys(('left', 'right'), IDENTITY)
    reference, _ = place._place_box_test_capture_descent_reference('base_link', relations)
    for index, expected_y in enumerate((0.002, 0.004, 0.005), 1):
        place._place_box_test_move_box_z(None, adapter, 'base_link', relations, reference, index * 0.005, 5.0)
        left, right, *_ = adapter.execute_dual_movel_endpoint.call_args.args
        assert left[:3] == pytest.approx([0.1, 0.4 + expected_y, 1.0 - index * 0.005])
        assert right[:3] == pytest.approx([0.2, -0.4 - expected_y, 1.02 - index * 0.005])
        assert reference.y_offset_m == pytest.approx(expected_y)
    reference.y_offset_m = 0.002
    reference.allow_y_change = False
    place._place_box_test_move_box_z(None, adapter, 'base_link', relations, reference, 0.020, 5.0)
    assert reference.y_offset_m == pytest.approx(0.002)
    adapter.solve_ik.assert_not_called()


def test_each_work_z_not_world_z_and_no_y_axis_parallel_requirement():
    place = FixedDescent()
    relations = dict.fromkeys(('left', 'right'), IDENTITY)
    bases = {
        'left': ((0.0, 0.0, 0.0), (math.sin(0.15), 0.0, 0.0, math.cos(0.15))),
        'right': ((0.0, 0.0, 0.0), (0.0, math.sin(-0.1), 0.0, math.cos(-0.1))),
    }
    place._lookup_tf_carry_transform = lambda _base, frame: (
        bases[frame.split('_')[0]] if frame.endswith('_arm_base_frame')
        else place.actual[frame.split('_')[0]])
    reference, _ = place._place_box_test_capture_descent_reference('base_link', relations)
    world = place._place_box_test_descent_world_targets(reference, 0.005)
    from mission_runtime.box_support import BoxSupportMixin
    for arm in ('left', 'right'):
        inverse = BoxSupportMixin._inverse_transform(reference.base_by_arm[arm])
        start = BoxSupportMixin._compose_transform(inverse, reference.tcp_by_arm[arm])
        target = BoxSupportMixin._compose_transform(inverse, world[arm])
        assert target[0][0] == pytest.approx(start[0][0])
        assert target[0][1] == pytest.approx(start[0][1])
        assert target[0][2] - start[0][2] == pytest.approx(-0.005)
        assert target[1] == pytest.approx(start[1])
    adapter = Mock()
    place._place_box_test_move_box_z(
        SimpleNamespace(is_cancel_requested=False), adapter, 'base_link', relations, reference, 0.005, 5.0)
    adapter.solve_ik.assert_not_called()
    for arm, target in zip(('left', 'right'), adapter.execute_dual_movel_endpoint.call_args.args[:2]):
        local_start = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(bases[arm]), reference.tcp_by_arm[arm])
        assert target[:3] == pytest.approx((*local_start[0][:2], local_start[0][2] - 0.005))
