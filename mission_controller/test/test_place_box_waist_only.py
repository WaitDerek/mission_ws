"""No real SDK/ROS motion: waist-only dispatch and force-settling regressions."""
import math
import threading
from unittest.mock import Mock, patch

import pytest

from mission_runtime.box_carry import BoxCarryMixin
from mission_runtime.box_placement import BoxPlacementMixin
from mission_runtime.common import MissionError


IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
UPRIGHT = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))


class WaistOnlyPlace(BoxCarryMixin):
    def __init__(self):
        self.events = []
        self.feedback = []
        self.waist_done = False
        self.fail_body = False
        self.drop_scale = 0.1
        self.post_distance = 0.0
        self.post_wrench = {'left': (0.0,) * 6, 'right': (0.0,) * 6}
        self.actual = {arm: ((0.0, 0.0, 1.2), UPRIGHT) for arm in ('left', 'right')}
        self._last_grasp_box_tf_box_to_link7_targets = {arm: IDENTITY for arm in self.actual}
        self.body_command_client = Mock()
        self.body_command_client.call_async.side_effect = self.send_body

    def send_body(self, request):
        self.events.append('body_send')
        assert request == ('body', [-40000, -80000, -40000, 0])
        return object()

    def _float(self, name):
        return {
            'place_box_test_waist_clearance_m': 0.04,
            'place_box_test_post_support_arm_base_descent_m': self.post_distance,
            'dependency_wait_timeout_sec': 2.0,
            'place_box_test_start_body_tolerance_rad': 0.035,
            'box_joint1_velocity_tolerance_rad_sec': 0.02,
            'place_box_test_target_consistency_position_tolerance_m': 0.15,
            'place_box_test_target_consistency_orientation_tolerance_rad': 0.35,
            'place_box_test_bigbox_half_height_m': 0.1,
            'place_box_test_table_support_delta_fz_n': 3.0,
            'place_box_test_table_support_sign_left': 1.0,
            'place_box_test_table_support_sign_right': 1.0,
        }[name]

    def _float_array(self, name):
        return {
            'box_body_command_units_per_degree': [1000.0] * 4,
            'place_box_test_start_body_joint_units': [0.0] * 4,
            'place_box_test_body_joint_units': [-40000, -80000, -40000, 0],
        }[name]

    def _boolean(self, name):
        assert name == 'place_box_test_dynamic_table_enabled'
        return True

    def _integer(self, name):
        assert name == 'place_box_test_segments'
        return 6

    def _string(self, name):
        return 'base_link' if name == 'grasp_box_tf_freeze_frame' else name

    def _wait_for_fresh_body_feedback(self, _goal):
        return [0.0] * 4, [0.0] * 4, 42

    def _place_box_test_table_z_in_base(self, _frame):
        return 0.72

    def _lookup_tf_carry_transform(self, _frame, name):
        return IDENTITY if name.endswith('_arm_base_frame') else self.actual[name.split('_')[0]]

    def _joint123_arm_base_transform(self, _arm, angles):
        return ((0.0, 0.0, angles[1] * self.drop_scale), IDENTITY[1])

    def _check_canceled(self, *_args):
        pass

    def _publish_place_box_test_feedback(self, _goal, stage, detail):
        self.feedback.append((stage, detail))

    def _wait_for_service(self, *_args):
        pass

    def _place_box_test_body_request(self, units, **kwargs):
        assert kwargs == {'trajectory_connect': 0, 'blend_radius': 0}
        return 'body', units

    def _wait_future(self, *_args, **_kwargs):
        self.events.append('body_ack')
        return object()

    def _parse_string_command_response(self, *_args):
        pass

    def _wait_for_body_joints_target(self, _goal, angles, **kwargs):
        self.events.append('body_wait')
        assert kwargs['sequence_after'] == 42
        if self.fail_body:
            raise MissionError('body target failed')
        self.waist_done = True
        self.actual = {arm: ((0.0, 0.0, 1.06), UPRIGHT) for arm in self.actual}

    def _place_box_test_descend_to_table(self, _goal, _adapter, _frame, _relations, box, *_args):
        assert self.waist_done
        assert box[0][2] == pytest.approx(1.06)
        self.events.append('force_and_descent')
        return box, 'table support confirmed'

    def _place_box_test_wait_arms_still(self, _goal):
        pass

    def _place_box_test_stable_common_wrench(self, _goal, _adapter, _frame):
        return self.post_wrench if self.waist_done else {'left': (0.0,) * 6, 'right': (0.0,) * 6}

    def _place_box_test_waist_contact_monitor(self, *_args):
        pass

    def _place_box_test_post_support_arm_base_descent(self, *_args, **_kwargs):
        self.events.append('post_support')
        return {arm: self._endpoint_sync_transform_to_pose(pose) for arm, pose in self.actual.items()}, 'post_support'

    def _place_box_test_stop_body(self):
        self.events.append('stop_body')

    def _place_box_test_optimize_waist_box_path(self, *_args):
        raise AssertionError('must not search an arm compensation path')

    def _place_box_test_waist_path(self, *_args):
        raise AssertionError('must not create any arm compensation waypoints')


def run(place, adapter, dry=False):
    return place._execute_place_box_test_motion(None, adapter, 'bigbox', dry)


def test_dynamic_dispatch_sends_only_waist_before_measured_descent():
    place, adapter = WaistOnlyPlace(), Mock()
    result = run(place, adapter)
    assert place.events == ['body_send', 'body_ack', 'body_wait', 'force_and_descent']
    assert 'arm_compensation=disabled' in result[0]
    assert 'waist_workspace=not_used' in result[0]
    assert adapter.mock_calls == []
    assert place.body_command_client.call_async.call_count == 1


def test_waist_only_dry_run_sends_no_commands():
    place, adapter = WaistOnlyPlace(), Mock()
    assert 'dry-run' in run(place, adapter, dry=True)[0]
    assert place.events == []
    assert adapter.mock_calls == []
    place.body_command_client.call_async.assert_not_called()


def test_waist_failure_does_not_start_descent_or_arm_compensation():
    place, adapter = WaistOnlyPlace(), Mock()
    place.fail_body = True
    with pytest.raises(MissionError, match='body target failed'):
        run(place, adapter)
    assert place.events == ['body_send', 'body_ack', 'body_wait', 'stop_body']
    assert adapter.mock_calls == []


def test_predicted_early_table_contact_rejects_before_motion_without_lifting():
    place, adapter = WaistOnlyPlace(), Mock()
    place.drop_scale = 0.4
    with pytest.raises(MissionError, match='waist-only bend would violate'):
        run(place, adapter)
    assert place.events == []
    assert adapter.mock_calls == []


def test_existing_post_support_step_is_only_after_support_confirmation():
    place, adapter = WaistOnlyPlace(), Mock()
    place.post_distance = 0.03
    run(place, adapter)
    assert place.events[-2:] == ['force_and_descent', 'post_support']


def test_unilateral_contact_after_waist_skips_search_and_runs_three_cm():
    place, adapter = WaistOnlyPlace(), Mock()
    place.post_distance = 0.03
    place.post_wrench = {
        'left': (0.0, 0.0, 4.0, 0.0, 0.0, 0.0),
        'right': (0.0,) * 6,
    }
    result = run(place, adapter)
    assert 'force_and_descent' not in place.events
    assert place.events[-1] == 'post_support'
    assert 'table-search descent skipped' in result[0]


def test_work_force_is_rotated_to_common_vertical_before_comparison():
    place, adapter = WaistOnlyPlace(), Mock()
    quarter_turn_y = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
    place._lookup_tf_carry_transform = lambda _base, _arm: ((0.0, 0.0, 0.0), quarter_turn_y)
    adapter.read_bilateral_work_wrench.return_value = {
        'left': (-4.0, 0.0, 0.0, 0.0, 2.0, 0.0),
        'right': (-3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
    sample = place._place_box_test_common_wrench_sample(adapter, 'base_link')
    assert sample['left'][2] == pytest.approx(4.0)
    assert sample['right'][2] == pytest.approx(3.0)
    assert sample['left'][3:] == pytest.approx((0.0, 2.0, 0.0))


def test_waist_monitor_stops_on_sustained_unilateral_force():
    place, adapter = WaistOnlyPlace(), Mock()
    place._place_box_test_common_wrench_sample = lambda *_args: {
        'left': (0.0, 0.0, 12.0, 0.0, 0.0, 0.0),
        'right': (0.0,) * 6,
    }
    stop, state = threading.Event(), {}
    BoxPlacementMixin._place_box_test_waist_contact_monitor(
        place,
        adapter, 'base_link',
        {'left': (0.0,) * 6, 'right': (0.0,) * 6}, stop, state,
    )
    assert state['contact']['left'] == pytest.approx(12.0)
    assert place.events == ['stop_body']


class BaselinePlace(BoxCarryMixin):
    def __init__(self):
        self.feedback = []

    def _float(self, name):
        assert name == 'place_box_test_force_unload_baseline_timeout_sec'
        return 3.0

    def _check_canceled(self, *_args):
        pass

    def _publish_place_box_test_feedback(self, _goal, stage, detail):
        self.feedback.append((stage, detail))


@pytest.mark.parametrize('settles', [True, False])
def test_baseline_waits_for_settling_without_motion_or_sensor_zero(settles):
    place, adapter = BaselinePlace(), Mock()
    now = [0.0]
    count = [0]

    def read_force():
        count[0] += 1
        value = (1.709 if count[0] % 2 else 0.0) if count[0] <= 8 or not settles else 0.2
        return {'left': 0.0, 'right': value}

    def sleep(duration):
        now[0] += duration

    adapter.read_bilateral_work_fz.side_effect = read_force
    with patch('mission_runtime.box_placement.time.monotonic', side_effect=lambda: now[0]), \
         patch('mission_runtime.box_placement.time.sleep', side_effect=sleep):
        if settles:
            result = place._place_box_test_stable_work_fz_baseline(None, adapter, 3.0)
            assert result == {'left': 0.0, 'right': 0.2}
            assert count[0] > 8
        else:
            with pytest.raises(MissionError, match='no descent started'):
                place._place_box_test_stable_work_fz_baseline(None, adapter, 3.0)
    assert all(call[0] == 'read_bilateral_work_fz' for call in adapter.mock_calls)
    assert place.feedback[0][0] == 'WAITING_FOR_PLACE_FORCE_STABILITY'
