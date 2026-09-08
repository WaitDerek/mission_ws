"""G1D navigation tests exercise routing without moving hardware."""

import threading
from types import SimpleNamespace

import pytest

from mission_manager.taskflow.model import StepResult
from mission_manager.taskflow.mqtt_navigation import MqttNavigationGateway
from mission_manager.taskflow.state_machine import AssemblyWorkflowEngine
from mission_manager.taskflow.test_actions import TestActionsMixin as ActionMixin


@pytest.mark.parametrize("failed_point", [None, "3"])
def test_navigation_route_never_calls_perception_or_manipulation(failed_point):
    calls = []
    def navigate(request):
        calls.append(request.point_id)
        return StepResult(request.point_id != failed_point, "navigation result")
    def forbidden(*args):
        pytest.fail("navigation-only route called a manipulation step")
    operations = SimpleNamespace(navigate=navigate, is_cancel_requested=lambda: False,
                                 grip=forbidden, peel=forbidden, assemble=forbidden)
    progress = []
    outcome = AssemblyWorkflowEngine(
        operations, navigation_only=True, progress_callback=progress.append
    ).run("test")
    assert calls == (["1", "3", "2", "3"] if failed_point is None else ["1", "3"])
    assert outcome.success == (failed_point is None)
    assert outcome.completed_task_count == (4 if failed_point is None else 1)
    assert progress[-1].total_steps == 4


def test_custom_navigation_dry_run_never_publishes():
    harness = ActionMixin()
    harness._workflow_lock = threading.Lock()
    harness._workflow_reserved = True
    harness._cancel_event = threading.Event()
    harness._navigation_gateway = SimpleNamespace(_point_poses={})
    statuses = []
    goal = SimpleNamespace(
        request=SimpleNamespace(point_id=4, request_id="test", use_custom_pos=True,
                                pos=[1., 2., 3.], dry_run=True),
        publish_feedback=lambda feedback: None,
        succeed=lambda: statuses.append("success"),
        abort=lambda: statuses.append("abort"),
        canceled=lambda: statuses.append("canceled"))
    result = harness._execute_navigation(goal)
    assert result.success
    assert list(result.pos) == [1., 2., 3.]
    assert statuses == ["success"]
    assert not harness._workflow_reserved


def test_invalid_custom_navigation_is_rejected_before_broker_access():
    from mission_manager.taskflow.model import NavigationRequest
    gateway = object.__new__(MqttNavigationGateway)
    gateway._point_poses = {}
    result = gateway.navigate(NavigationRequest("test", "step", "1", (float("nan"), 0, 0)), lambda: False)
    assert not result.success
    assert result.status == "invalid"
