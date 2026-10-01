"""Work-Z guards and fresh synchronized TF tests. No hardware connections."""
import math
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from tf2_ros import TransformException

from mission_runtime.common import MissionCanceled, MissionError
from mission_runtime.box_placement import _PlaceDescentReference
from mission_runtime.place_descent_z_guard import (
    DescentZSettings, PlaceDescentZGuardMixin, evaluate_work_z, work_z,
    choose_downward_correction,
)


Q = (0.0, 0.0, 0.0, 1.0)
BASES = {'left': ((0.0, 0.4, 0.2), Q), 'right': ((0.0, -0.4, 0.3), Q)}
START = {'left': ((0.1, 0.5, 1.0), Q), 'right': ((0.1, -0.5, 1.12), Q)}
TARGET = {a: ((t[0][0], t[0][1], t[0][2] - 0.01), Q) for a, t in START.items()}
SETTINGS = DescentZSettings(0.005, 0.005, 1.0, 0.25, 3)


def error_actual(left, right):
    return {a: ((t[0][0], t[0][1], t[0][2] + e), Q)
            for (a, t), e in zip(TARGET.items(), (left, right))}


def test_initial_height_difference_is_preserved_not_forced_to_zero():
    passed, detail = evaluate_work_z(BASES, START, TARGET, TARGET, SETTINGS)
    assert passed
    assert 'initial_left_minus_right_z=-0.020000m' in detail
    assert 'z_difference_change=+0.000000m' in detail


@pytest.mark.parametrize('errors,passed', [
    ((0.0025, 0.0), True), ((0.005, 0.005), True),
    ((0.006, 0.006), False),  # equal heights are insufficient if both missed target
    ((0.004, -0.004), False),  # individually within5mm, but relative error8mm
])
def test_absolute_and_relative_z_checks_are_independent(errors, passed):
    result, _ = evaluate_work_z(BASES, START, TARGET, error_actual(*errors), SETTINGS)
    assert result is passed


def test_work_z_uses_rotated_axis_and_origin_not_common_world_z():
    # 90deg about Work X: Work Z points along common -Y.
    base = ((0.2, 0.4, 0.6), (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
    assert work_z(base, ((0.2, 0.1, 0.6), Q)) == pytest.approx(0.3)


def test_nonfinite_readings_cannot_pass():
    with pytest.raises(MissionError, match='non-finite'):
        evaluate_work_z(BASES, START, TARGET, error_actual(float('nan'), 0), SETTINGS)


class Guard(PlaceDescentZGuardMixin):
    def __init__(self):
        self.feedback = []
        self.ns = 10_000_000_000
        self.tf_buffer = Mock()

    def _float(self, name):
        return {
            'place_box_test_descent_z_target_tolerance_m': 0.005,
            'place_box_test_descent_z_difference_tolerance_m': 0.005,
            'place_box_test_descent_z_check_timeout_sec': 0.2,
            'place_box_test_descent_z_feedback_max_age_sec': 0.25,
            'place_box_test_descent_z_correction_step_m': 0.002,
            'place_box_test_descent_z_correction_max_travel_m': 0.005,
        }[name]

    def _integer(self, _name):
        return 3

    def _string(self, name):
        return name

    def get_clock(self):
        return NS(now=lambda: NS(nanoseconds=self.ns))

    def _check_canceled(self, *_args):
        pass

    def _publish_place_box_test_feedback(self, _goal, stage, detail):
        self.feedback.append((stage, detail))


def invoke(guard, adapter):
    now = [0.0]
    def sleep(dt):
        now[0] += dt
    with patch('mission_runtime.place_descent_z_guard.time.monotonic', side_effect=lambda: now[0]), \
         patch('mission_runtime.place_descent_z_guard.time.sleep', side_effect=sleep):
        return guard._place_box_test_verify_descent_z(
            None, adapter, 'base_link', _PlaceDescentReference(START, BASES), TARGET
        )


def test_waits_for_three_distinct_samples_no_motion_or_force_queries():
    guard, adapter = Guard(), Mock()
    stamps = iter([11, 11, 12, 13])
    guard._place_box_test_read_z_check_tf = Mock(
        side_effect=lambda *_: (TARGET, next(stamps) * 10**9, '')
    )
    assert invoke(guard, adapter) == TARGET
    assert guard._place_box_test_read_z_check_tf.call_count == 4
    assert adapter.mock_calls == []
    assert guard.feedback[-1][0] == 'PLACE_DESCENT_Z_VERIFIED'


@pytest.mark.parametrize('kind', ['z_error', 'stale', 'tf_exception', 'repeated'])
def test_failure_stops_both_without_correction_or_release(kind):
    guard, adapter = Guard(), Mock()
    n = [11]
    def read(*_args):
        n[0] += 1
        if kind == 'tf_exception':
            raise TransformException('missing frame')
        if kind == 'stale':
            return None, 0, 'stale TF'
        if kind == 'repeated':
            return TARGET, 11 * 10**9, ''
        return error_actual(-0.006, 0.0), n[0] * 10**9, ''
    guard._place_box_test_read_z_check_tf = read
    with pytest.raises(MissionError, match='no next segment or release'):
        invoke(guard, adapter)
    adapter.stop_all.assert_called_once()
    assert [call[0] for call in adapter.mock_calls] == ['stop_all']
    assert guard.feedback[-1][0] == 'PLACE_DESCENT_Z_CHECK_FAILED'


def test_cancel_stops_both_and_propagates():
    guard, adapter = Guard(), Mock()
    guard._check_canceled = Mock(side_effect=MissionCanceled('cancelled'))
    with pytest.raises(MissionCanceled):
        invoke(guard, adapter)
    adapter.stop_all.assert_called_once()


def message(stamp, z=0.9):
    return NS(header=NS(stamp=NS(sec=stamp // 10**9, nanosec=stamp % 10**9)),
              transform=NS(translation=NS(x=0.0, y=0.0, z=z),
                           rotation=NS(x=0.0, y=0.0, z=0.0, w=1.0)))


def test_both_tf_lookups_use_same_post_completion_timestamp():
    guard = Guard()
    guard.ns += 50_000_000
    stamps = {'left': 10_010_000_000, 'right': 10_020_000_000}
    exact_queries = []
    def lookup(_base, frame, when):
        arm = frame.split('_')[0]
        if when.nanoseconds:
            exact_queries.append(when.nanoseconds)
        return message(when.nanoseconds or stamps[arm])
    guard.tf_buffer.lookup_transform.side_effect = lookup
    actual, stamp, _ = guard._place_box_test_read_z_check_tf('base_link', 10_000_000_000, 0.25)
    assert set(actual) == {'left', 'right'}
    assert exact_queries == [stamp, stamp] == [10_010_000_000] * 2


@pytest.mark.parametrize('stamp', [9_990_000_000, 9_500_000_000, 11_000_000_000])
def test_pre_completion_stale_or_future_tf_is_not_accepted(stamp):
    guard = Guard()
    guard.tf_buffer.lookup_transform.return_value = message(stamp)
    actual, _, _ = guard._place_box_test_read_z_check_tf('base_link', 9_995_000_000, 0.25)
    assert actual is None
    assert guard.tf_buffer.lookup_transform.call_count == 2


def test_corrects_only_lagging_side_then_verifies_without_dual_motion():
    guard, adapter = Guard(), Mock()
    actual = error_actual(0.006, 0.0)
    corrections = []
    def read(*_args):
        guard.ns += 10_000_000
        return dict(actual), guard.ns, ''
    def correct(_goal, _adapter, _ref, measured, arm, distance):
        corrections.append((arm, distance))
        pose = measured[arm]
        actual[arm] = ((pose[0][0], pose[0][1], pose[0][2] - distance), pose[1])
    guard._place_box_test_read_z_check_tf = read
    guard._place_box_test_correct_descent_z = correct
    result = invoke(guard, adapter)
    assert corrections == [('left', 0.002)]
    assert result['right'] == TARGET['right']
    assert result['left'][0][2] - TARGET['left'][0][2] == pytest.approx(0.004)
    adapter.stop_all.assert_not_called()
    assert [s for s, _ in guard.feedback] == ['PLACE_DESCENT_Z_CORRECTING', 'PLACE_DESCENT_Z_VERIFIED']


def test_pair_error_corrects_high_side_only_as_far_as_its_original_target():
    actual = error_actual(0.002, -0.004)
    arm, distance = choose_downward_correction(
        BASES, TARGET, actual, SETTINGS, 0.005, dict.fromkeys(ARMS, 0.005)
    )
    assert arm == 'left'
    assert distance == pytest.approx(0.002)  # not 6mm to chase the right arm


ARMS = ('left', 'right')


def test_unresponsive_correction_is_bounded_and_does_not_release():
    guard, adapter = Guard(), Mock()
    def read(*_args):
        guard.ns += 10_000_000
        return error_actual(0.009, 0), guard.ns, ''
    guard._place_box_test_read_z_check_tf = read
    guard._place_box_test_correct_descent_z = Mock()  # no measured progress
    with pytest.raises(MissionError, match='no next segment or release'):
        invoke(guard, adapter)
    distances = [c.args[-1] for c in guard._place_box_test_correct_descent_z.call_args_list]
    assert distances == pytest.approx([0.002, 0.002, 0.001])
    assert sum(distances) == pytest.approx(0.005)
    adapter.stop_all.assert_called_once()


def test_still_moving_feedback_does_not_trigger_correction():
    guard, adapter = Guard(), Mock()
    count = [0]
    def read(*_args):
        count[0] += 1
        guard.ns += 10_000_000
        return error_actual(0.006 + 0.003 * (count[0] % 2), 0), guard.ns, ''
    guard._place_box_test_read_z_check_tf = read
    guard._place_box_test_correct_descent_z = Mock()
    with pytest.raises(MissionError):
        invoke(guard, adapter)
    guard._place_box_test_correct_descent_z.assert_not_called()
    adapter.stop_all.assert_called_once()


def test_correction_sdk_error_stops_both_and_is_not_retried():
    guard, adapter = Guard(), Mock()
    def read(*_args):
        guard.ns += 10_000_000
        return error_actual(0.006, 0), guard.ns, ''
    guard._place_box_test_read_z_check_tf = read
    guard._place_box_test_correct_descent_z = Mock(side_effect=RuntimeError('SDK failed'))
    with pytest.raises(RuntimeError, match='SDK failed'):
        invoke(guard, adapter)
    guard._place_box_test_correct_descent_z.assert_called_once()
    adapter.stop_all.assert_called_once()


def test_correction_command_keeps_other_arm_untouched_and_local_xy_orientation():
    from mission_runtime.box_carry import BoxCarryMixin
    place, adapter = BoxCarryMixin(), Mock()
    place._float = lambda key: 5.0 if key.endswith('velocity_percent') else 120.0
    actual = error_actual(0.006, 0)
    ref = _PlaceDescentReference(START, BASES)
    place._place_box_test_correct_descent_z(NS(is_cancel_requested=False), adapter, ref, actual, 'left', 0.002)
    kwargs = adapter.execute_single.call_args.kwargs
    assert kwargs['arm'] == 'left'
    assert kwargs['motion_mode'] == 'movel'
    assert kwargs['target'][:3] == pytest.approx([0.1, 0.1, actual['left'][0][2] - 0.2 - 0.002])
    assert kwargs['target'][3:] == pytest.approx([0, 0, 0])
    adapter.execute_dual_movel_endpoint.assert_not_called()
    assert [c[0] for c in adapter.mock_calls] == ['execute_single']
