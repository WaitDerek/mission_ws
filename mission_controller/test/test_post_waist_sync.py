import math
from types import SimpleNamespace
import threading
import time

import pytest
from geometry_msgs.msg import PoseStamped, TransformStamped
from mission_runtime import post_waist_sync as sync
from mission_runtime.common import MissionError
from mission_runtime.box_support import BoxSupportMixin
from test_post_waist_live_tf_ik import Harness, Adapter, target


def test_failed_run_angles_do_not_pass_settled_gate():
    desired = [math.radians(v) for v in [-4, 7, -6]]
    inferred = [math.radians(v) for v in [-3.629, 6.600, -5.439]] + [0.]
    assert all(abs(a-b) < .01 for a,b in zip(inferred,desired))
    assert not sync.feedback_is_settled(inferred, [0.]*4, desired, .01)
    assert sync.feedback_is_settled(desired+[0.], [0.]*4, desired, .01)
    assert not sync.feedback_is_settled(desired+[0.], [0.]*4, desired, 1.)


class Clock:
    value = 0.
    def monotonic(self): return self.value
    def sleep(self, value): self.value += value
    def now(self): return SimpleNamespace(nanoseconds=int((10+self.value)*1e9))


class SyncHarness(BoxSupportMixin):
    def __init__(self, clock, stale_tf=False, wrong_body=False):
        self.clock = clock
        self.seq = 0
        self.events = []
        self.stale_tf = stale_tf
        self.wrong_body = wrong_body
        self.queries = []
        self.tf_buffer = SimpleNamespace(lookup_transform=self.lookup)
    def _float(self, name): return 1.
    def _string(self, name): return name
    def _check_canceled(self, *args): pass
    def _publish_box_grasp_feedback(self, _, stage, detail): self.events.append(stage)
    def get_clock(self): return self.clock
    def _body_feedback_snapshot(self):
        self.seq += 1
        return ([.005 if self.wrong_body else 0.,0.,0.,0.], [0.]*4, self.clock.value, self.seq)
    def _joint123_chest_transform(self, q): return ((0.,0.,0.),(0.,0.,0.,1.))
    def _configured_rpy_transform(self, *args): return ((0.,0.,0.),(0.,0.,0.,1.))
    def lookup(self, frame, arm, stamp):
        self.queries.append((arm,stamp.nanoseconds))
        msg=TransformStamped()
        now=self.clock.now().nanoseconds
        msg.header.stamp.sec=now//10**9
        msg.header.stamp.nanosec=now%10**9
        msg.transform.rotation.w=1.
        if self.stale_tf:
            msg.transform.translation.y=.02
        return msg


def frozen():
    p=PoseStamped();p.header.frame_id='base_link';p.pose.position.x=.5;p.pose.orientation.w=1.
    return p


def test_gate_returns_only_after_stability_and_uses_one_tf_time(monkeypatch):
    clock=Clock();monkeypatch.setattr(sync,'time',clock)
    h=SyncHarness(clock)
    left,right=sync.wait_for_settled_tf_targets(h,None,frozen(),frozen(),[0.]*3)
    assert clock.value >= sync.SETTLE_SEC
    assert left.position.x == right.position.x == .5
    assert h.events[-1] == 'POST_WAIST_TF_SYNCED'
    for i in range(0,len(h.queries),4):
        assert h.queries[i+2][1] == h.queries[i+3][1] > 0


@pytest.mark.parametrize('wrong_body,stale_tf', [(True,False),(False,True)])
def test_wrong_feedback_or_tf_times_out_without_returning_targets(monkeypatch,wrong_body,stale_tf):
    clock=Clock();monkeypatch.setattr(sync,'time',clock)
    h=SyncHarness(clock,stale_tf,wrong_body)
    with pytest.raises(MissionError,match='synchronization timed out'):
        sync.wait_for_settled_tf_targets(h,None,frozen(),frozen(),[0.]*3)
    assert 'POST_WAIST_TF_SYNCED' not in h.events


def test_current_joint_seed_is_used_first():
    h=Harness();h.joint_state_lock=threading.Lock()
    h.latest_slave_arm_positions={'left':[.1]*7,'right':[.2]*7}
    h.latest_slave_arm_state_times={'left':time.monotonic(),'right':time.monotonic()}
    adapter=Adapter({'left':[0.,0.,0.,-30.,0.,0.,0.],'right':[0.,0.,0.,-30.,0.,0.,0.]})
    predicted=SimpleNamespace(left_joint_deg=[0.]*7,right_joint_deg=[0.]*7)
    h._solve_post_waist_live_tf_movej_ik(None,adapter,target(.3),target(.4),predicted,right_arm_only=False)
    assert adapter.calls[0][2] == pytest.approx([math.degrees(.1)]*7)
    assert adapter.calls[1][2] == pytest.approx([math.degrees(.2)]*7)


def test_redundancy_seed_recovers_without_changing_target_pose():
    h=Harness()
    class SeedAdapter(Adapter):
        def solve_ik(self, arm, pose, seed):
            self.calls.append((arm,list(pose),list(seed)))
            return [0.,0.,0.,-30.,0.,0.,0.] if seed[2] == -30. else None
    adapter=SeedAdapter({})
    predicted=SimpleNamespace(left_joint_deg=[0.]*7,right_joint_deg=[0.]*7)
    solved=h._solve_post_waist_live_tf_movej_ik(None,adapter,target(.3),target(.4),predicted,right_arm_only=False)
    assert set(solved) == {'left','right'}
    for arm in solved:
        poses=[call[1] for call in adapter.calls if call[0]==arm]
        assert all(p == poses[0] for p in poses)
