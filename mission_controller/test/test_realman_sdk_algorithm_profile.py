"""Offline profile/serialization regressions; no physical robot required."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mission_runtime.realman_sdk_algorithm import (
    ControllerAlgorithmProfile, SDK_ALGORITHM_LOCK,
)
from mission_runtime.realman_sdk_adapter import RealManSdkAdapter, RealManSdkError


def robot_profile():
    robot = Mock()
    robot.rm_get_robot_info.return_value = (
        0, {"arm_model": "RXL75", "arm_dof": 7, "force_type": "6FB"})
    robot.rm_get_DH_data.return_value = (0, {
        "d": [0.1855, 0, 0.292, 0, 0.294, 0, 0, 0.105],
        "a": [0] * 8, "alpha": [0] * 8, "offset": [0] * 8})
    robot.rm_get_install_pose.return_value = {
        "return_code": 0, "x": 0, "y": -90, "z": 0}
    robot.rm_get_joint_min_pos.return_value = (0, [-175] * 7)
    robot.rm_get_joint_max_pos.return_value = (0, [175] * 7)
    robot.rm_get_current_tool_frame.return_value = (
        0, {"pose": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    robot.rm_get_current_work_frame.return_value = (
        0, {"pose": [-0.1, 0.2, 0, 0, 0, 0.3]})
    return robot


def read_profile(robot):
    return ControllerAlgorithmProfile.read(
        "left", robot, SimpleNamespace(RM_MODEL_RXL75_E=17),
        SimpleNamespace(RM_MODEL_RM_ISF_E=3),
        lambda **values: values, lambda **values: values,
    )


def test_full_profile_preserves_model_and_nonzero_frames():
    robot = robot_profile()
    profile = read_profile(robot)
    constructor = Mock()
    algo = profile.activate(constructor)
    constructor.assert_called_once_with(17, 3)
    algo.rm_algo_set_dh.assert_called_once_with(robot.rm_get_DH_data.return_value[1])
    algo.rm_algo_set_angle.assert_called_once_with(0.0, -90.0, 0.0)
    algo.rm_algo_set_joint_min_limit.assert_called_once_with([-175] * 7)
    algo.rm_algo_set_joint_max_limit.assert_called_once_with([175] * 7)
    algo.rm_algo_set_toolframe.assert_called_once_with(
        robot.rm_get_current_tool_frame.return_value[1])
    algo.rm_algo_set_workframe.assert_called_once_with(
        robot.rm_get_current_work_frame.return_value[1])
    assert all(not call[0].startswith('rm_set_') for call in robot.mock_calls)


@pytest.mark.parametrize("method", [
    "rm_get_robot_info", "rm_get_DH_data", "rm_get_joint_min_pos",
    "rm_get_joint_max_pos", "rm_get_current_tool_frame", "rm_get_current_work_frame",
])
def test_profile_read_failure_never_falls_back_to_factory(method):
    robot = robot_profile()
    getattr(robot, method).return_value = (-2, {})
    with pytest.raises(RealManSdkError, match="cannot read"):
        read_profile(robot)


@pytest.mark.parametrize("bad", ["dh_rows", "dh_nan", "angle", "min_length", "bounds", "tool"])
def test_rejects_incomplete_or_invalid_profile(bad):
    robot = robot_profile()
    if bad == "dh_rows":
        robot.rm_get_DH_data.return_value[1]["d"] = [0] * 7
    elif bad == "dh_nan":
        robot.rm_get_DH_data.return_value[1]["offset"][3] = float('nan')
    elif bad == "angle":
        robot.rm_get_install_pose.return_value = {"return_code": -1}
    elif bad == "min_length":
        robot.rm_get_joint_min_pos.return_value = (0, [-175] * 6)
    elif bad == "bounds":
        robot.rm_get_joint_max_pos.return_value = (0, [-180] * 7)
    else:
        robot.rm_get_current_tool_frame.return_value = (0, {"pose": [float('inf')] * 6})
    with pytest.raises(RealManSdkError):
        read_profile(robot)


def test_force_start_fk_waits_for_offline_algorithm_lock():
    adapter = object.__new__(RealManSdkAdapter)
    robot = Mock()
    robot.rm_get_current_tool_frame.return_value = (0, {"name": "Arm_Tip", "pose": [0] * 6})
    robot.rm_get_current_work_frame.return_value = (0, {"name": "World", "pose": [0] * 6})
    robot.rm_get_current_arm_state.return_value = (0, {"joint": [0] * 7})
    robot.rm_algo_forward_kinematics.side_effect = [[0, 0, 0, 1, 0, 0, 0], [0] * 6]
    robot.rm_algo_pos2matrix.return_value = SimpleNamespace(
        data=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    robot.rm_get_force_data.return_value = (0, {"tool_zero_force_data": [0] * 6})
    ready = threading.Event()
    robot.rm_get_current_arm_state.side_effect = lambda: (ready.set() or (0, {"joint": [0] * 7}))
    results = []
    errors = []

    def run():
        try:
            results.append(adapter._tool_y_force_start_state("left", robot))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    with SDK_ALGORITHM_LOCK:
        thread.start()
        assert ready.wait(1.0)
        robot.rm_algo_forward_kinematics.assert_not_called()
    thread.join(1.0)
    assert not thread.is_alive()
    assert not errors
    assert len(results) == 1
    assert robot.rm_algo_forward_kinematics.call_count == 2
