import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped

from mission_controller.common import MissionCanceled, MissionError
from mission_controller.manipulation import ManipulationMixin
from mission_controller.manipulation_runtime import ManipulationRuntimeMixin
from mission_controller import suction_runtime


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


class _Logger:
    def error(self, _message):
        pass


def test_execute_and_run_pose_topics_are_kept_separate():
    mission_config = (CONFIG_ROOT / "mission.yaml").read_text(encoding="utf-8")

    assert (
        "pipeline_left_ee_pose_topic: /pinocchio_g1d/left_ee_pose"
        in mission_config
    )
    assert (
        "pipeline_right_ee_pose_topic: /pinocchio_g1d/right_ee_pose"
        in mission_config
    )
    assert "run_left_ee_pose_topic: /left_ee_pose" in mission_config
    assert "run_right_ee_pose_topic: /right_ee_pose" in mission_config


class _GoalHandle:
    def __init__(self, *, target_type=""):
        self.status = None
        self.is_cancel_requested = False
        self.request = SimpleNamespace(target_type=target_type)

    def publish_feedback(self, _feedback):
        pass

    def succeed(self):
        self.status = "succeeded"

    def abort(self):
        self.status = "aborted"

    def canceled(self):
        self.status = "canceled"


class _Harness(ManipulationMixin):
    def __init__(self):
        self._grip_config = json.loads(
            (CONFIG_ROOT / "grip.json").read_text(encoding="utf-8")
        )
        self._connector_grip_config = json.loads(
            (CONFIG_ROOT / "connector_grip.json").read_text(encoding="utf-8")
        )
        self._peel_config = json.loads(
            (CONFIG_ROOT / "peel.json").read_text(encoding="utf-8")
        )
        self._assembly_config = json.loads(
            (CONFIG_ROOT / "assembly.json").read_text(encoding="utf-8")
        )
        self._state_lock = threading.Lock()
        self._active_child_handles = {}
        self._active = True
        self.events = []
        self.pose = PoseStamped()
        self.pose.pose.orientation.w = 1.0

    def get_logger(self):
        return _Logger()

    @staticmethod
    def _float(_name):
        return 10.0

    @staticmethod
    def _string(_name):
        return "torso_link"

    @staticmethod
    def get_clock():
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time))

    def _require_hardware_messages(self):
        self.events.append(("require_hardware",))

    def _prepare_pipeline(self, _goal, action_type, config):
        self.events.append(
            ("prepare", action_type.__name__, config.get("model_label", ""))
        )

    @staticmethod
    def _pipeline_feedback(_goal, _action_type, _stage, _detail):
        pass

    def _wait_for_pipeline_poses(self, _goal, *, need_left, need_right, require_fresh):
        self.events.append(("wait_pose", need_left, need_right, require_fresh))
        return (
            self.pose if need_left else None,
            self.pose if need_right else None,
        )

    def _wait_for_run_poses(self, _goal, *, need_left, need_right, require_fresh):
        self.events.append(("wait_run_pose", need_left, need_right, require_fresh))
        return (
            self.pose if need_left else None,
            self.pose if need_right else None,
        )

    def _estimate_pipeline_object(self, _goal, model_label):
        self.events.append(("detect", model_label))
        return self.pose

    def _estimate_front_bumper_task(self, _goal, patch_name):
        self.events.append(("detect_front_bumper", patch_name))
        return self.pose

    def _pipeline_move_pose(self, _goal, *, left, right, linear=False):
        self.events.append(("move_l" if linear else "move_p", bool(left), bool(right)))

    def _set_suction(self, _goal, selector, enabled):
        self.events.append(("suction", selector, enabled))

    def _wait_for_next_force(self, _goal):
        self.events.append(("force_baseline",))
        return (0.0, 0.0, 1.0), 7

    def _pipeline_linear_contact(
        self,
        _goal,
        *,
        left,
        right,
        threshold,
        baseline_sample=None,
        force_sequence=None,
    ):
        self.events.append(("contact", bool(left), bool(right), threshold))
        return True, threshold + 0.5

    def _cancel_children(self):
        self.events.append(("cancel_children",))


def test_execute_grip_keeps_original_run_order():
    harness = _Harness()
    harness.run_grip = lambda *_args: pytest.fail("execute_grip called run_grip")
    goal = _GoalHandle()
    result = harness._execute_grip(goal)

    assert result.success
    assert result.contact_detected
    assert goal.status == "succeeded"
    assert harness.events == [
        ("require_hardware",),
        ("prepare", "ExecuteGrip", "badge"),
        ("wait_pose", True, False, True),
        ("detect", "badge"),
        ("move_p", True, False),
        ("suction", "left", True),
        ("wait_pose", True, False, True),
        ("contact", True, False, 5.0),
        ("move_l", True, False),
    ]


def test_execute_grip_fails_closed_until_connector_is_calibrated():
    harness = _Harness()
    goal = _GoalHandle(target_type="connector")
    result = harness._execute_grip(goal)

    assert not result.success
    assert goal.status == "aborted"
    assert "connector grip calibration is disabled" in result.message


def test_execute_grip_selects_connector_vision_model_after_calibration():
    harness = _Harness()
    harness._connector_grip_config["calibrated"] = True
    goal = _GoalHandle(target_type="connector")
    result = harness._execute_grip(goal)

    assert result.success
    assert ("prepare", "ExecuteGrip", "badge_connector") in harness.events
    assert ("detect", "badge_connector") in harness.events


def test_execute_peel_keeps_original_run_order():
    harness = _Harness()
    harness.run_peel = lambda *_args: pytest.fail("execute_peel called run_peel")
    goal = _GoalHandle()
    result = harness._execute_peel(goal)

    assert result.success
    assert result.contact_detected
    assert goal.status == "succeeded"
    assert harness.events == [
        ("require_hardware",),
        ("prepare", "ExecutePeel", "badge_back"),
        ("wait_pose", True, True, True),
        ("detect", "badge_back"),
        ("move_p", False, True),
        ("suction", "right", True),
        ("force_baseline",),
        ("wait_pose", True, True, True),
        ("contact", False, True, 3.0),
        ("wait_pose", True, True, True),
        ("move_p", True, True),
        ("suction", "right", False),
    ]


def test_run_grip_remains_a_separate_one_shot_flow():
    harness = _Harness()
    goal = _GoalHandle()

    outcome = harness.run_grip(goal, "badge")

    assert outcome.contact_detected
    assert goal.status is None
    assert ("wait_run_pose", True, False, False) in harness.events
    assert not any(event[0] == "wait_pose" for event in harness.events)


def test_mission_node_keeps_legacy_run_methods_during_migration():
    harness = _Harness()

    assert callable(harness.run_grip)
    assert callable(harness.run_peel)


def test_run_peel_remains_a_separate_one_shot_flow():
    harness = _Harness()
    goal = _GoalHandle()

    outcome = harness.run_peel(goal)

    assert outcome.contact_detected
    assert goal.status is None
    assert ("wait_run_pose", True, True, True) in harness.events
    assert not any(event[0] == "wait_pose" for event in harness.events)


def test_run_grip_releases_suction_after_post_contact_failure():
    harness = _Harness()
    original_move = harness._pipeline_move_pose

    def fail_withdraw(goal, *, left, right, linear=False):
        if linear:
            raise MissionError("legacy grip withdraw failed")
        return original_move(goal, left=left, right=right, linear=linear)

    harness._pipeline_move_pose = fail_withdraw
    with pytest.raises(MissionError, match="legacy grip withdraw failed"):
        harness.run_grip(_GoalHandle(), "badge")
    assert harness.events[-1] == ("suction", "left", False)


def test_run_peel_releases_suction_after_force_failure():
    harness = _Harness()

    def fail_force(_goal):
        raise MissionError("legacy peel force failed")

    harness._wait_for_next_force = fail_force
    with pytest.raises(MissionError, match="legacy peel force failed"):
        harness.run_peel(_GoalHandle())
    assert harness.events[-1] == ("suction", "right", False)


def test_execute_grip_fails_without_contact_and_releases_suction():
    harness = _Harness()

    def no_contact(*_args, **_kwargs):
        harness.events.append(("contact", True, False, 5.0))
        return False, 0.25

    harness._pipeline_linear_contact = no_contact
    goal = _GoalHandle()
    result = harness._execute_grip(goal)

    assert not result.success
    assert goal.status == "aborted"
    assert "without detecting force contact" in result.message
    assert harness.events[-1] == ("suction", "left", False)
    assert ("move_l", True, False) not in harness.events


def test_execute_peel_fails_without_contact_and_releases_suction():
    harness = _Harness()

    def no_contact(*_args, **_kwargs):
        harness.events.append(("contact", False, True, 3.0))
        return False, 0.25

    harness._pipeline_linear_contact = no_contact
    goal = _GoalHandle()
    result = harness._execute_peel(goal)

    assert not result.success
    assert goal.status == "aborted"
    assert "without detecting force contact" in result.message
    assert harness.events[-1] == ("suction", "right", False)
    assert ("move_p", True, True) not in harness.events


def test_child_motion_is_canceled_before_outer_cancel_propagates():
    harness = _Harness()
    child = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: object(),
        cancel_goal_async=lambda: setattr(child, "canceled", True),
        canceled=False,
    )
    client = SimpleNamespace(send_goal_async=lambda _request: object())
    harness._wait_for_server = lambda *_args: None
    waits = iter([child, MissionCanceled("canceled")])

    def wait_future(*_args):
        value = next(waits)
        if isinstance(value, Exception):
            raise value
        return value

    harness._wait_future = wait_future
    with pytest.raises(MissionCanceled):
        harness._send_pipeline_action(
            _GoalHandle(),
            client=client,
            action_name="/move_arm_j",
            request=object(),
            handle_key="test_child",
            result_timeout=1.0,
        )
    assert child.canceled
    assert "test_child" not in harness._active_child_handles


def test_child_motion_is_canceled_after_result_timeout():
    harness = _Harness()
    child = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: object(),
        cancel_goal_async=lambda: setattr(child, "canceled", True),
        canceled=False,
    )
    client = SimpleNamespace(send_goal_async=lambda _request: object())
    harness._wait_for_server = lambda *_args: None
    waits = iter([child, MissionError("result timeout")])

    def wait_future(*_args):
        value = next(waits)
        if isinstance(value, Exception):
            raise value
        return value

    harness._wait_future = wait_future
    with pytest.raises(MissionError, match="result timeout"):
        harness._send_pipeline_action(
            _GoalHandle(),
            client=client,
            action_name="/move_arm_j",
            request=object(),
            handle_key="test_child",
            result_timeout=1.0,
        )
    assert child.canceled
    assert "test_child" not in harness._active_child_handles


def test_contact_motion_is_canceled_after_terminal_wait_failure():
    harness = _Harness()
    harness._pipeline_condition = threading.Condition()
    harness._force_sequence = 1
    harness._force_sample = (0.0, 0.0, 1.0)
    result_future = SimpleNamespace(done=lambda: True)
    child = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: result_future,
        cancel_goal_async=lambda: setattr(child, "canceled", True),
        canceled=False,
    )
    harness.arm_linear_client = SimpleNamespace(
        send_goal_async=lambda _request: object()
    )
    harness._wait_for_server = lambda *_args: None
    waits = iter([child, MissionError("terminal result timeout")])

    def wait_future(*_args):
        value = next(waits)
        if isinstance(value, Exception):
            raise value
        return value

    harness._wait_future = wait_future
    with pytest.raises(MissionError, match="terminal result timeout"):
        suction_runtime.SuctionRuntimeMixin._pipeline_linear_contact(
            harness,
            _GoalHandle(),
            left=[],
            right=[0.0] * 7,
            threshold=3.0,
            baseline_sample=(0.0, 0.0, 1.0),
            force_sequence=1,
        )
    assert child.canceled
    assert "pipeline_contact_move_arm_l" not in harness._active_child_handles


def test_prepare_rejects_mismatched_dual_arm_waypoint_counts():
    harness = _Harness()
    config = {
        "prepare_traj": {
            "left": [[0.0] * 7],
            "right": [],
        }
    }

    with pytest.raises(MissionError, match="waypoint counts differ"):
        ManipulationRuntimeMixin._prepare_pipeline(
            harness, _GoalHandle(), object, config
        )


def test_relay_verification_requires_a_new_result_callback(monkeypatch):
    harness = object.__new__(ManipulationMixin)
    harness._pipeline_condition = threading.Condition()
    harness._relay_expected_channel = None
    harness._relay_expected_state = None
    harness._relay_verified = False
    harness._relay_result_sequence = 4
    published = threading.Event()
    harness._relay_goal_publisher = SimpleNamespace(
        publish=lambda _message: published.set()
    )
    harness._wait_for_relay_driver = lambda _goal: None
    harness._check_canceled = lambda *_args: None
    monkeypatch.setattr(
        suction_runtime,
        "SerialioCtrl",
        SimpleNamespace(Goal=lambda: SimpleNamespace()),
    )

    thread = threading.Thread(
        target=harness._set_relay_channel,
        args=(_GoalHandle(), 1, True),
    )
    thread.start()
    assert published.wait(timeout=0.5)
    with harness._pipeline_condition:
        harness._relay_verified = True
        harness._pipeline_condition.notify_all()
    time.sleep(0.05)
    assert thread.is_alive()

    harness._relay_result_callback(SimpleNamespace(success_array=[True]))
    thread.join(timeout=0.5)
    assert not thread.is_alive()


def test_execute_grip_can_run_twice_on_the_same_node():
    harness = _Harness()

    for _ in range(2):
        harness._active = True
        goal = _GoalHandle()
        result = harness._execute_grip(goal)
        assert result.success
        assert goal.status == "succeeded"
        assert not harness._active


def test_force_sequence_only_advances_for_a_new_timestamp():
    harness = object.__new__(ManipulationMixin)
    harness._pipeline_condition = threading.Condition()
    harness._force_sample = None
    harness._force_stamp_ns = None
    harness._force_sequence = 0

    def message(nanosec):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=nanosec)),
            fx=1.0,
            fy=2.0,
            fz=3.0,
        )

    harness._force_callback(message(10))
    harness._force_callback(message(10))
    assert harness._force_sequence == 1
    harness._force_callback(message(11))
    assert harness._force_sequence == 2


def test_missing_hardware_messages_abort_grip_goal():
    harness = _Harness()

    def reject_missing_driver_overlay():
        raise MissionError("brsd_msgs unavailable")

    harness._require_hardware_messages = reject_missing_driver_overlay
    goal = _GoalHandle()
    result = harness._execute_grip(goal)

    assert not result.success
    assert goal.status == "aborted"
    assert "brsd_msgs unavailable" in result.message


def test_outer_cancel_sets_canceled_terminal_state_and_cancels_children():
    harness = _Harness()

    def cancel_during_prepare(*_args):
        raise MissionCanceled("mission canceled during preparation")

    harness._prepare_pipeline = cancel_during_prepare
    goal = _GoalHandle()
    result = harness._execute_peel(goal)

    assert not result.success
    assert goal.status == "canceled"
    assert harness.events == [("require_hardware",), ("cancel_children",)]


def test_assembly_fails_closed_until_calibrated():
    harness = _Harness()
    goal = _GoalHandle(target_type="connector")
    result = harness._execute_assembly(goal)

    assert not result.success
    assert goal.status == "aborted"
    assert "calibration is disabled" in result.message
    assert harness.events == [("require_hardware",)]


def test_execute_assembly_uses_front_bumper_pose_then_force_insertion():
    harness = _Harness()
    harness._assembly_config["calibrated"] = True
    goal = _GoalHandle(target_type="connector")
    result = harness._execute_assembly(goal)

    assert result.success
    assert result.contact_detected
    assert goal.status == "succeeded"
    assert harness.events == [
        ("require_hardware",),
        ("prepare", "ExecuteAssembly", ""),
        ("wait_pose", True, False, True),
        ("detect_front_bumper", "connector"),
        ("move_p", True, False),
        ("force_baseline",),
        ("contact", True, False, 3.0),
        ("suction", "left", False),
    ]
