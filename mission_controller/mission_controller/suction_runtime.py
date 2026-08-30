"""Suction relay and force-contact behavior for manipulation tasks."""

from __future__ import annotations

import time
from typing import Optional

from action_msgs.msg import GoalStatus
from task_interfaces.action import MoveArmPose

from .common import MissionCanceled, MissionError

try:
    from brsd_msgs.action import SerialioCtrl
    from brsd_msgs.msg import ForceData
except ImportError:  # The driver overlay is only present on the hardware host.
    SerialioCtrl = None
    ForceData = None


class _CleanupGoalHandle:
    is_cancel_requested = False


class SuctionRuntimeMixin:
    """Provide the original relay sequencing and fresh-force contact detection."""

    def _initialize_suction_runtime(self) -> None:
        self._force_sample: Optional[tuple[float, float, float]] = None
        self._force_stamp_ns: Optional[int] = None
        self._force_sequence = 0
        self._relay_expected_channel: Optional[int] = None
        self._relay_expected_state: Optional[bool] = None
        self._relay_verified = False
        self._relay_result_sequence = 0

        self._relay_goal_publisher = None
        self._relay_result_subscription = None
        self._force_subscription = None
        if SerialioCtrl is not None and ForceData is not None:
            self._relay_goal_publisher = self.create_publisher(
                SerialioCtrl.Goal,
                self._string("suction_relay_goal_topic"),
                10,
            )
            self._relay_result_subscription = self.create_subscription(
                SerialioCtrl.Result,
                self._string("suction_relay_result_topic"),
                self._relay_result_callback,
                10,
                callback_group=self._callback_group,
            )
            self._force_subscription = self.create_subscription(
                ForceData,
                self._string("force_torque_topic"),
                self._force_callback,
                10,
                callback_group=self._callback_group,
            )
        else:
            self.get_logger().warning(
                "brsd_msgs is unavailable; /execute_grip and /execute_peel "
                "will reject execution until the hardware driver overlay is sourced"
            )

    def _require_hardware_messages(self) -> None:
        if (
            SerialioCtrl is None
            or ForceData is None
            or self._relay_goal_publisher is None
            or self._force_subscription is None
        ):
            raise MissionError(
                "brsd_msgs is unavailable; source the driver workspace before "
                "starting mission_controller"
            )

    def _force_callback(self, message) -> None:
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        with self._pipeline_condition:
            self._force_sample = (
                float(message.fx),
                float(message.fy),
                float(message.fz),
            )
            if stamp_ns != self._force_stamp_ns:
                self._force_stamp_ns = stamp_ns
                self._force_sequence += 1
            self._pipeline_condition.notify_all()

    def _relay_result_callback(self, message) -> None:
        with self._pipeline_condition:
            self._relay_result_sequence += 1
            channel = self._relay_expected_channel
            expected_state = self._relay_expected_state
            if channel is None or expected_state is None:
                return
            states = list(message.success_array)
            index = channel - 1
            if len(states) > index and bool(states[index]) == expected_state:
                self._relay_verified = True
                self._pipeline_condition.notify_all()

    def _wait_for_relay_driver(self, goal_handle) -> None:
        self._require_hardware_messages()
        deadline = time.monotonic() + 3.0
        while self._relay_goal_publisher.get_subscription_count() == 0:
            self._check_canceled(
                goal_handle, "while waiting for the suction relay driver"
            )
            if time.monotonic() >= deadline:
                raise MissionError(
                    "no relay driver subscribes to "
                    f"{self._string('suction_relay_goal_topic')}"
                )
            time.sleep(0.05)

    def _set_relay_channel(self, goal_handle, channel: int, enabled: bool) -> None:
        self._wait_for_relay_driver(goal_handle)
        with self._pipeline_condition:
            result_sequence_before = self._relay_result_sequence
            self._relay_expected_channel = channel
            self._relay_expected_state = enabled
            self._relay_verified = False
        message = SerialioCtrl.Goal()
        message.deveice_id_array = [channel]
        message.state_array = [enabled]
        message.control_mode = False
        self._relay_goal_publisher.publish(message)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while verifying suction relay state")
            with self._pipeline_condition:
                if (
                    self._relay_verified
                    and self._relay_result_sequence > result_sequence_before
                ):
                    return
            time.sleep(0.02)
        raise MissionError(
            f"relay channel {channel} was not verified within 5.0 seconds"
        )

    def _set_suction(self, goal_handle, selector: str, enabled: bool) -> None:
        if selector == "left":
            sequence = ((1, True), (2, True)) if enabled else ((2, False), (1, False))
        elif selector == "right":
            sequence = ((3, False), (4, True)) if enabled else ((4, False), (3, True))
        else:
            raise MissionError(f"unsupported suction selector '{selector}'")
        for index, (channel, state) in enumerate(sequence):
            self._set_relay_channel(goal_handle, channel, state)
            if index == 0:
                self._cancelable_sleep(
                    goal_handle, 2.0, "between suction relay channel commands"
                )

    def _set_suction_best_effort(self, selector: str, enabled: bool) -> None:
        """Restore relay state after a failed/canceled Action without re-canceling."""
        try:
            self._set_suction(_CleanupGoalHandle(), selector, enabled)
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask root failure.
            self.get_logger().error(
                f"failed to restore {selector} suction to {enabled}: {exc}"
            )

    def _wait_for_next_force(
        self, goal_handle
    ) -> tuple[tuple[float, float, float], int]:
        self._require_hardware_messages()
        with self._pipeline_condition:
            sequence_before = self._force_sequence
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for force sensor data")
            with self._pipeline_condition:
                if (
                    self._force_sample is not None
                    and self._force_sequence > sequence_before
                ):
                    return self._force_sample, self._force_sequence
            time.sleep(0.01)
        raise MissionError("no new force sensor data received within 2.0 seconds")

    def _pipeline_linear_contact(
        self,
        goal_handle,
        *,
        left: list[float],
        right: list[float],
        threshold: float,
        baseline_sample: Optional[tuple[float, float, float]] = None,
        force_sequence: Optional[int] = None,
    ) -> tuple[bool, float]:
        if baseline_sample is None or force_sequence is None:
            baseline_sample, force_sequence = self._wait_for_next_force(goal_handle)
        initial_fz = baseline_sample[2]
        request = MoveArmPose.Goal()
        request.left_pose = [float(value) for value in left]
        request.right_pose = [float(value) for value in right]
        request.dry_run = False
        action_name = self._string("move_arm_linear_action_name")
        self._wait_for_server(self.arm_linear_client, action_name, goal_handle)
        child = self._wait_future(
            self.arm_linear_client.send_goal_async(request),
            goal_handle,
            f"sending {action_name}",
            self._float("dependency_wait_timeout_sec"),
        )
        if child is None or not child.accepted:
            raise MissionError(f"{action_name} goal was rejected")
        with self._state_lock:
            self._active_child_handles["pipeline_contact_move_arm_l"] = child

        result_future = child.get_result_async()
        with self._pipeline_condition:
            force_sequence = self._force_sequence
        contact_detected = False
        peak_delta = 0.0
        deadline = time.monotonic() + self._float("arm_pose_timeout_sec")
        try:
            while not result_future.done():
                self._check_canceled(goal_handle, f"while waiting for {action_name}")
                if time.monotonic() >= deadline:
                    child.cancel_goal_async()
                    raise MissionError(f"timeout waiting for {action_name} result")
                with self._pipeline_condition:
                    if (
                        self._force_sequence > force_sequence
                        and self._force_sample is not None
                    ):
                        force_sequence = self._force_sequence
                        delta = abs(self._force_sample[2] - initial_fz)
                        peak_delta = max(peak_delta, delta)
                        if delta >= threshold:
                            contact_detected = True
                if contact_detected:
                    self.get_logger().info(
                        f"suction contact detected: delta_fz={peak_delta:.4f} "
                        f">= {threshold:.4f}"
                    )
                    self._wait_future(
                        child.cancel_goal_async(),
                        goal_handle,
                        f"canceling {action_name} after contact",
                        self._float("dependency_wait_timeout_sec"),
                    )
                    break
                time.sleep(0.01)

            wrapped = self._wait_future(
                result_future,
                goal_handle,
                f"waiting for {action_name} terminal result",
                self._float("dependency_wait_timeout_sec"),
            )
        except Exception:
            try:
                child.cancel_goal_async()
            except Exception:  # noqa: BLE001 - preserve the root failure.
                pass
            raise
        finally:
            with self._state_lock:
                self._active_child_handles.pop("pipeline_contact_move_arm_l", None)

        if contact_detected:
            return True, peak_delta
        result = wrapped.result
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED and result.success:
            self.get_logger().warning(
                "maximum suction approach distance reached without detecting contact"
            )
            return False, peak_delta
        raise MissionError(
            f"{action_name} failed: error_code={result.error_code}, "
            f"message={result.message}"
        )
