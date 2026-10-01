import json
import threading
from types import SimpleNamespace

import pytest
from rclpy.action import GoalResponse
from mission_runtime.taskflow.navigation_test import NavigationTestMixin
from mission_runtime.taskflow.lease import WorkflowLeaseManager
from mission_runtime.taskflow.navigation import FakeNavigationGateway
from mission_runtime.taskflow.model import NavigationResult


class Harness(NavigationTestMixin):
    def __init__(self):
        self._workflow_lock = threading.Lock()
        self._workflow_reserved = False
        self._cancel_event = threading.Event()
        self.mission_lease_manager = WorkflowLeaseManager()
        self._active_navigation_gateway = None
        self.points = {str(i): dict(x=float(i), y=0., yaw=.1) for i in range(1,8)}
        self.gateway = FakeNavigationGateway()
        self.acquired = 0
        self.released = 0
        self.created = 0

    def _string(self, key):
        assert key == 'mqtt_navigation_points_json'
        return json.dumps(self.points)

    def get_logger(self):
        return SimpleNamespace(warning=lambda *_: None)

    def _reserve_goal(self, mission, request_id):
        r = self.mission_lease_manager.reserve_goal(mission, request_id)
        return GoalResponse.ACCEPT if r.accepted else GoalResponse.REJECT

    def _acquire_lease(self, workflow_id):
        self.acquired += 1
        return SimpleNamespace(success=True, lease_token='test-token')

    def _release_lease(self, workflow_id, token):
        self.released += 1
        return SimpleNamespace(success=True)

    def _make_navigation_gateway(self):
        self.created += 1
        return self.gateway

    def _audit_feedback(self, *args):
        pass


class Goal:
    def __init__(self, dry_run=False, rounds=0):
        self.request = SimpleNamespace(start=True, request_id='test', dry_run=dry_run, rounds=rounds)
        self.is_cancel_requested = False
        self.feedback = []
        self.status = None

    def publish_feedback(self, message): self.feedback.append(message)
    def succeed(self): self.status = 'success'
    def abort(self): self.status = 'failed'
    def canceled(self): self.status = 'canceled'


def run(h, g):
    assert h._navigation_test_goal_callback(g.request) == GoalResponse.ACCEPT
    result = h._execute_navigation_test(g)
    assert not h._workflow_reserved
    assert not h.mission_lease_manager.goal_reserved
    assert h._active_navigation_gateway is None
    return result


def test_seven_points_in_order_and_pose_snapshot():
    h, g = Harness(), Goal()
    r = run(h,g)
    assert r.success and g.status == 'success'
    assert [n.point_id for n in h.gateway.requests] == [str(i) for i in range(1,8)]
    assert [n.pos for n in h.gateway.requests] == [(float(i),0.,.1) for i in range(1,8)]
    assert len(set(n.step_id for n in h.gateway.requests)) == 7
    assert r.completed_rounds == 1 and r.completed_navigation_count == 7
    assert len(r.leg_elapsed_sec) == len(r.attempted_point_ids) == 7
    assert h.acquired == h.released == h.created == 1


def test_multiple_rounds_unique_request_ids():
    h,g = Harness(),Goal(rounds=2)
    r=run(h,g)
    assert r.success and r.completed_rounds == 2
    assert list(r.attempted_point_ids) == list(range(1,8))*2
    assert len(set(n.step_id for n in h.gateway.requests)) == 14


def test_dry_run_never_connects_or_acquires_remote_lease():
    h,g=Harness(),Goal(dry_run=True)
    r=run(h,g)
    assert r.success and r.final_stage == 'DRY_RUN_COMPLETE'
    assert h.acquired == h.released == h.created == 0
    assert not h.gateway.requests


def test_failure_stops_before_next_point():
    h,g=Harness(),Goal()
    h.gateway=FakeNavigationGateway([NavigationResult(True,'succeeded','ok'), NavigationResult(False,'timeout','late')])
    r=run(h,g)
    assert not r.success and r.failed_point_id == 2
    assert r.completed_navigation_count == 1 and r.completed_rounds == 0
    assert len(h.gateway.requests) == 2 and h.released == 1


def test_cancel_during_navigation_stops_remaining_route():
    h,g=Harness(),Goal()
    def navigate(request, cancel):
        h._navigation_test_cancel_callback(g)
        return NavigationResult(False,'canceled','user canceled')
    h.gateway.navigate=navigate
    r=run(h,g)
    assert not r.success and g.status == 'canceled'
    assert list(r.attempted_point_ids) == [1]
    assert h.released == 1


def test_cancel_before_execute_sends_nothing():
    h,g=Harness(),Goal()
    assert h._navigation_test_goal_callback(g.request) == GoalResponse.ACCEPT
    h._navigation_test_cancel_callback(g)
    r=h._execute_navigation_test(g)
    assert g.status == 'canceled' and not r.success
    assert h.acquired == h.created == 0
    assert not h._workflow_reserved


def test_missing_point_rejects_before_reservation():
    h,g=Harness(),Goal()
    del h.points['7']
    assert h._navigation_test_goal_callback(g.request) == GoalResponse.REJECT
    assert not h._workflow_reserved and not h.mission_lease_manager.goal_reserved


def test_existing_workflow_blocks_test():
    h,g=Harness(),Goal()
    h._workflow_reserved=True
    assert h._navigation_test_goal_callback(g.request) == GoalResponse.REJECT


def test_existing_standalone_navigation_blocks_test():
    h,g=Harness(),Goal()
    h.mission_lease_manager.reserve_goal('navigate_to_point','existing')
    assert h._navigation_test_goal_callback(g.request) == GoalResponse.REJECT
    assert not h._workflow_reserved


def test_reserved_test_blocks_standalone_navigation():
    h,g=Harness(),Goal()
    assert h._navigation_test_goal_callback(g.request) == GoalResponse.ACCEPT
    assert not h.mission_lease_manager.reserve_goal('navigate_to_point','other').accepted


def test_close_failure_still_releases_lease_and_reservation():
    h,g=Harness(),Goal()
    def bad_close(): raise RuntimeError('close failed')
    h.gateway.close=bad_close
    r=run(h,g)
    assert not r.success and r.final_stage == 'CLEANUP_FAILED'
    assert h.released == 1


def test_remote_lease_failure_does_not_navigate():
    h,g=Harness(),Goal()
    h._acquire_lease=lambda _: SimpleNamespace(success=False,message='busy')
    r=run(h,g)
    assert not r.success and h.created == h.released == 0
