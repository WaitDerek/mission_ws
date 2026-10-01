"""No-motion checks for global observation J2 preparation."""
import math
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from mission_runtime.taskflow.fixed_operations import FixedRosWorkflowOperations
from mission_runtime.taskflow.model import StepResult
from mission_runtime.ros_arm_movej import RosArmMoveJ
from mission_runtime.realman_sdk_common import RealManSdkError
from test_ros_arm_movej import _Node, _Client, _state

def operation():
    o = FixedRosWorkflowOperations.__new__(FixedRosWorkflowOperations)
    o._dry_run = False
    o.is_cancel_requested = lambda: False
    o.move_right_joints = Mock(return_value=StepResult(True, "reached"))
    return o

def test_only_joint2_changes_from_measured_pose():
    o = operation()
    measured = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    transport = Mock()
    transport.current_joint_positions.return_value = measured.copy()
    o._sdk_adapter = SimpleNamespace(_ros_movej_transport=transport)
    assert o.prepare_right_observation_joint2(30., "prepare").success
    target = o.move_right_joints.call_args.args[0]
    assert target[1] == pytest.approx(math.radians(30))
    assert [target[i] for i in (0,2,3,4,5,6)] == [measured[i] for i in (0,2,3,4,5,6)]

def test_missing_feedback_does_not_command_motion():
    o = operation()
    transport = Mock()
    transport.current_joint_positions.side_effect = RealManSdkError("stale feedback")
    o._sdk_adapter = SimpleNamespace(_ros_movej_transport=transport)
    assert not o.prepare_right_observation_joint2(30., "prepare").success
    o.move_right_joints.assert_not_called()

def test_snapshot_rejects_stale_or_moving_feedback_without_commands():
    node, client = _Node(), _Client()
    t = RosArmMoveJ(node, client=client)
    t._on_feedback("right", _state([10.]*7))
    assert t.current_joint_positions("right") == pytest.approx([math.radians(10.)]*7)
    seq, stamp, joints, speeds = t._feedback["right"]
    t._feedback["right"] = (seq, time.monotonic()-10, joints, speeds)
    with pytest.raises(RealManSdkError): t.current_joint_positions("right", timeout_sec=.03)
    t._feedback["right"] = (seq, time.monotonic(), joints, [1.]*7)
    with pytest.raises(RealManSdkError): t.current_joint_positions("right", timeout_sec=.03)
    assert not client.requests
