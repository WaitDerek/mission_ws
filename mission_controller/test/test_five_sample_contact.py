from test_bilateral_contact_guard import run
from test_realman_sdk_adapter import _ForceRobot

class ResetThenContact(_ForceRobot):
    def __init__(self, sign):
        super().__init__(sign)
        self.active_reads = 0
        self.reads_at_stop = None
    def rm_get_force_data(self):
        fy = 0.
        if self.commands and not self.stop_calls:
            self.active_reads += 1
            # Four hits, a miss, then five hits. The miss must reset counting.
            fy = self.sign * (0. if self.active_reads == 5 else 1.2)
        return 0, {'tool_zero_force_data': [0.,fy,0.,0.,0.,0.]}
    def rm_stop_force_position_move(self):
        self.reads_at_stop = self.active_reads
        return super().rm_stop_force_position_move()

def test_five_consecutive_hits_reset_on_miss_and_no_hold_after_stop():
    left,right=ResetThenContact(-1.),ResetThenContact(1.)
    adapter,stops,_=run(left,right)
    result=adapter.execute_tool_y_force_clamp(
        ('left','right'), {'left':-1.,'right':1.},
        speed_mm_s=1.,max_travel_m={'left':.05,'right':.05},timeout_sec=2.,
        control_period_sec=.01,baseline_stability_window_sec=.04,
        baseline_stability_max_span_n=.2,baseline_stability_timeout_sec=.3,
        contact_consecutive_samples=5,contact_min_duration_sec=0.,
        dual_post_stop_confirmation_sec=0.)
    assert left.reads_at_stop == 10
    assert right.reads_at_stop == 10
    assert result['contact_force_delta_n']['left'] <= -1.
    assert result['contact_force_delta_n']['right'] >= 1.
    assert not stops
