import threading
import pytest
from mission_runtime.realman_sdk_adapter import RealManSdkAdapter
from mission_runtime.realman_sdk_common import RealManSdkError
from test_realman_sdk_adapter import _ForceRobot, _ForceMoveParam, _SingleSpikeThenContactRobot

class LostAfterStop(_ForceRobot):
    def rm_get_force_data(self):
        if self.stop_calls:
            return 0, {'tool_zero_force_data': [0., self.sign*.069, 0., 0., 0., 0.]}
        return super().rm_get_force_data()

class SpikeThenContact(_SingleSpikeThenContactRobot):
    def __init__(self, sign):
        super().__init__()
        self.sign = sign
    def rm_get_force_data(self):
        code, data = super().rm_get_force_data()
        data['tool_zero_force_data'][1] *= self.sign
        return code, data

def run(left, right, threshold=1.):
    adapter = object.__new__(RealManSdkAdapter)
    adapter._motion_lock = threading.Lock()
    adapter._stop_event = threading.Event()
    adapter._motion_active = False
    adapter._connect = lambda: None
    adapter._robots = lambda: (left,right)
    adapter._force_position_move_type = _ForceMoveParam
    stops=[]
    adapter.stop_all=lambda: stops.append(True)
    args=dict(speed_mm_s=1., max_travel_m={'left':.05,'right':.05}, timeout_sec=1.,
              control_period_sec=.01, baseline_stability_window_sec=.04,
              baseline_stability_max_span_n=.2, baseline_stability_timeout_sec=.3,
              contact_consecutive_samples=3, contact_min_duration_sec=.3,
              dual_post_stop_confirmation_sec=.3)
    return adapter, stops, lambda: adapter.execute_tool_y_force_clamp(
        ('left','right'), {'left':-threshold,'right':threshold}, **args)

@pytest.mark.parametrize('lost_arm',['left','right','both'])
def test_contact_loss_blocks_success_and_stops_both(lost_arm):
    left=(LostAfterStop if lost_arm in ('left','both') else _ForceRobot)(-1.)
    right=(LostAfterStop if lost_arm in ('right','both') else _ForceRobot)(1.)
    _,stops,execute=run(left,right)
    with pytest.raises(RealManSdkError,match='lift forbidden'):
        execute()
    assert stops

def test_sustained_bilateral_contact_succeeds():
    _,stops,execute=run(_ForceRobot(-1.),_ForceRobot(1.))
    result=execute()
    assert result['force_delta_n']['left'] <= -1.
    assert result['force_delta_n']['right'] >= 1.
    assert not stops

def test_both_startup_spikes_are_ignored_until_continuous_contact():
    left,right=SpikeThenContact(-1.),SpikeThenContact(1.)
    _,stops,execute=run(left,right,threshold=2.)
    execute()
    assert len(left.commands)>=5 and len(right.commands)>=5
    assert not stops

class ShortPulse(_ForceRobot):
    def __init__(self, sign):
        super().__init__(sign)
        self.first_force_at = None
    def rm_get_force_data(self):
        import time
        if self.commands:
            if self.first_force_at is None:
                self.first_force_at = time.monotonic()
            if time.monotonic() - self.first_force_at > .12:
                return 0, {'tool_zero_force_data': [0.]*6}
        return super().rm_get_force_data()

def test_pulse_longer_than_three_samples_but_under_300ms_cannot_confirm():
    left,right=ShortPulse(-1.),ShortPulse(1.)
    _,stops,execute=run(left,right)
    with pytest.raises(RealManSdkError,match='timed out'):
        execute()
    assert stops
