import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import JointState
from task_interfaces.action import GoReady, Home, MoveArmJoints

from .common import (
    VALID_ARMS,
    MissionCanceled,
    MissionError,
    TaskActionError,
)


class ActionRuntimeMixin:
    """Shared command, feedback, readiness, and dependency primitives."""

    def _finalize_action_result(
        self, result, started_at: float, action_name: str
    ) -> None:
        if "elapsed_sec=" in str(result.message):
            return
        elapsed_sec = max(0.0, time.monotonic() - started_at)
        base_message = str(result.message).strip() or f"{action_name} finished"
        result.message = f"{base_message} (elapsed_sec={elapsed_sec:.3f})"
        self.get_logger().info(
            f"{action_name} finished: success={bool(result.success)}, "
            f"elapsed_sec={elapsed_sec:.3f}"
        )

    @staticmethod
    def _check_canceled(goal_handle, context: str) -> None:
        if goal_handle.is_cancel_requested:
            raise MissionCanceled(f"mission canceled {context}")

    def _wait_delay(self, goal_handle, duration_sec: float, context: str) -> None:
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, context)
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _wait_for_service(self, client, service_name: str, goal_handle) -> None:
        timeout_sec = self._float("dependency_wait_timeout_sec")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, f"while waiting for {service_name}")
            remaining = max(0.0, deadline - time.monotonic())
            if client.wait_for_service(timeout_sec=min(0.5, remaining)):
                return
        raise MissionError(
            f"timeout waiting for service {service_name} after {timeout_sec:.1f}s"
        )

    def _wait_for_action_server(self, goal_handle) -> None:
        action_name = self._string("arm_pose_action_name")
        timeout_sec = self._float("dependency_wait_timeout_sec")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, f"while waiting for {action_name}")
            remaining = max(0.0, deadline - time.monotonic())
            if self.arm_pose_client.wait_for_server(timeout_sec=min(0.5, remaining)):
                return
        raise MissionError(
            f"timeout waiting for action {action_name} after {timeout_sec:.1f}s"
        )

    def _wait_future(
        self,
        future,
        goal_handle,
        description: str,
        timeout_sec: float,
        cancel_local_future: bool,
    ):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if goal_handle.is_cancel_requested and cancel_local_future:
                future.cancel()
                raise MissionCanceled(f"mission canceled while {description}")
            if time.monotonic() >= deadline:
                future.cancel()
                raise MissionError(
                    f"timeout while {description} after {timeout_sec:.1f}s"
                )
            time.sleep(0.05)
        if not rclpy.ok():
            raise MissionError(f"ROS shutdown while {description}")
        try:
            value = future.result()
        except Exception as exc:  # noqa: BLE001
            raise MissionError(f"{description} failed: {exc}") from exc
        if not cancel_local_future:
            self._check_canceled(goal_handle, f"after {description}")
        return value

    def _wait_for_publisher(self, publisher, topic: str, goal_handle) -> None:
        if not self._boolean("wait_for_command_subscribers"):
            return
        timeout_sec = self._float("command_subscriber_wait_timeout_sec")
        deadline = time.monotonic() + timeout_sec
        while publisher.get_subscription_count() == 0:
            self._check_canceled(goal_handle, f"while waiting for subscriber on {topic}")
            if time.monotonic() >= deadline:
                detail = (
                    f"no subscriber matched on {topic} within {timeout_sec:.1f}s"
                )
                if self._boolean("require_command_subscribers"):
                    raise MissionError(detail)
                self.get_logger().warning(f"{detail}; publishing anyway")
                return
            time.sleep(0.05)

    def _publish_joint_command(
        self,
        publisher,
        topic: str,
        positions: list[float],
        velocities: list[float],
        goal_handle,
        *,
        require_subscriber: bool = True,
        honor_cancel: bool = True,
    ) -> None:
        if require_subscriber:
            self._wait_for_publisher(publisher, topic, goal_handle)
        message = JointState()
        message.position = positions
        message.velocity = velocities
        repeat_count = self._integer("command_repeat_count")
        interval_sec = self._float("command_repeat_interval_sec")
        for _ in range(repeat_count):
            if honor_cancel:
                self._check_canceled(goal_handle, f"while publishing {topic}")
            message.header.stamp = self.get_clock().now().to_msg()
            publisher.publish(message)
            if interval_sec > 0.0:
                time.sleep(interval_sec)

    def _publish_torso(
        self,
        goal_handle,
        positions: list[float],
        *,
        velocities: Optional[list[float]] = None,
        require_subscriber: bool = True,
        honor_cancel: bool = True,
    ) -> None:
        self._publish_joint_command(
            self.torso_publisher,
            self._string("torso_topic"),
            positions,
            (
                list(velocities)
                if velocities is not None
                else self._float_array("torso_velocities")
            ),
            goal_handle,
            require_subscriber=require_subscriber,
            honor_cancel=honor_cancel,
        )

    def _publish_gripper(self, goal_handle, arm: str, position: float) -> None:
        if arm == "left":
            publisher = self.left_gripper_publisher
            topic = self._string("left_gripper_topic")
        else:
            publisher = self.right_gripper_publisher
            topic = self._string("right_gripper_topic")
        self._publish_joint_command(
            publisher, topic, [position], [], goal_handle
        )

    def _joint_state_callback(self, message: JointState) -> None:
        if len(message.name) != len(message.position):
            return
        positions = {
            name: float(position)
            for name, position in zip(message.name, message.position)
            if name and math.isfinite(float(position))
        }
        if not positions:
            return
        with self.joint_state_lock:
            self.latest_joint_positions.update(positions)
            self.latest_joint_state_time = time.monotonic()
            self.latest_joint_state_sequence += 1

    def _torso_feedback_callback(self, message: JointState) -> None:
        if len(message.position) < 4:
            return
        positions = [float(value) for value in message.position[:4]]
        if not all(math.isfinite(value) for value in positions):
            return
        with self.joint_state_lock:
            self.latest_torso_positions = positions
            self.latest_torso_state_time = time.monotonic()
            self.latest_torso_state_sequence += 1

    def _gripper_feedback_callback(
        self, arm: str, message: JointState
    ) -> None:
        if len(message.position) != 1:
            return
        position = float(message.position[0])
        if not math.isfinite(position):
            return
        with self.joint_state_lock:
            self.latest_gripper_positions[arm] = position
            self.latest_gripper_state_times[arm] = time.monotonic()
            self.latest_gripper_state_sequences[arm] += 1

    def _gripper_feedback_sequence(self, arm: str) -> int:
        with self.joint_state_lock:
            return self.latest_gripper_state_sequences[arm]

    def _gripper_close_ratio(self, measured_position: float) -> float:
        open_position = self._float("gripper_open_position")
        closed_position = self._float("gripper_closed_position")
        ratio = (
            (measured_position - open_position)
            / (closed_position - open_position)
        )
        return min(1.0, max(0.0, ratio))

    def _wait_for_gripper_close_feedback(
        self,
        goal_handle,
        arm: str,
        sequence_before_close: int,
    ) -> float:
        timeout_sec = self._float("grasp_gripper_feedback_timeout_sec")
        max_age_sec = self._float("grasp_gripper_feedback_max_age_sec")
        deadline = time.monotonic() + timeout_sec
        latest_position: Optional[float] = None
        latest_sequence = sequence_before_close
        latest_age = math.inf

        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while waiting for gripper close feedback"
            )
            now = time.monotonic()
            with self.joint_state_lock:
                latest_sequence = self.latest_gripper_state_sequences[arm]
                latest_position = self.latest_gripper_positions.get(arm)
                latest_age = now - self.latest_gripper_state_times[arm]
            if (
                latest_sequence > sequence_before_close
                and latest_position is not None
                and latest_age <= max_age_sec
            ):
                return latest_position
            time.sleep(0.02)

        raise MissionError(
            f"no fresh {arm} gripper feedback after close command within "
            f"{timeout_sec:.1f}s (topic="
            f"{self._string(f'{arm}_gripper_feedback_topic')}, "
            f"sequence={latest_sequence}, age_sec={latest_age:.3f})"
        )

    def _wait_for_torso_target(
        self,
        goal_handle,
        targets: list[float],
        context: str,
    ) -> None:
        tolerance = self._float("torso_target_tolerance")
        timeout_sec = self._float("torso_target_wait_timeout_sec")
        required_stable = self._integer("torso_target_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequence = -1
        last_errors: list[float] = []

        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, f"while {context}")
            with self.joint_state_lock:
                sequence = self.latest_torso_state_sequence
                measured = list(self.latest_torso_positions)
            if sequence == last_sequence or len(measured) < len(targets):
                time.sleep(0.02)
                continue
            last_sequence = sequence
            last_errors = [
                abs(actual - target)
                for actual, target in zip(measured, targets)
            ]
            if max(last_errors, default=0.0) <= tolerance:
                stable_samples += 1
                if stable_samples >= required_stable:
                    self.get_logger().info(
                        f"verified torso target while {context}: "
                        f"max_error={max(last_errors, default=0.0):.4f}"
                    )
                    return
            else:
                stable_samples = 0
            time.sleep(0.02)

        raise MissionError(
            f"torso target was not confirmed while {context} within "
            f"{timeout_sec:.1f}s: max_error="
            f"{max(last_errors, default=math.inf):.4f}, "
            f"tolerance={tolerance:.4f}, stable_samples={required_stable}"
        )

    def _observation_ready(
        self,
        *,
        enabled_parameter: str,
        left_target_parameter: str,
        right_target_parameter: str,
        torso_target_parameter: str,
        max_age_parameter: str,
        torso_tolerance_parameter: str,
        label: str,
    ) -> tuple[bool, str]:
        if not self._boolean(enabled_parameter):
            return False, f"{label} observation readiness check is disabled"

        now = time.monotonic()
        max_age = self._float(max_age_parameter)
        with self.joint_state_lock:
            arm_positions = dict(self.latest_joint_positions)
            arm_age = now - self.latest_joint_state_time
            torso_positions = list(self.latest_torso_positions)
            torso_age = now - self.latest_torso_state_time

        left_names = self._string_array("left_arm_joint_names")
        right_names = self._string_array("right_arm_joint_names")
        arm_targets = dict(
            zip(
                left_names,
                self._float_array(left_target_parameter),
            )
        )
        arm_targets.update(
            zip(
                right_names,
                self._float_array(right_target_parameter),
            )
        )
        missing = [name for name in arm_targets if name not in arm_positions]
        if missing:
            return False, f"missing arm feedback joints={missing}"
        if not torso_positions:
            return False, "torso feedback has not been received"
        if arm_age > max_age or torso_age > max_age:
            return (
                False,
                "feedback is stale: "
                f"arms={arm_age:.2f}s, torso={torso_age:.2f}s, "
                f"limit={max_age:.2f}s",
            )

        arm_errors = {
            name: abs(arm_positions[name] - target)
            for name, target in arm_targets.items()
        }
        torso_targets = self._float_array(torso_target_parameter)
        torso_errors = [
            abs(actual - target)
            for actual, target in zip(torso_positions, torso_targets)
        ]
        max_arm_error = max(arm_errors.values(), default=0.0)
        max_torso_error = max(torso_errors, default=0.0)
        arm_tolerance = self._float("arm_joint_target_tolerance")
        torso_tolerance = self._float(torso_tolerance_parameter)
        ready = (
            max_arm_error <= arm_tolerance
            and max_torso_error <= torso_tolerance
        )
        return (
            ready,
            f"max arm error={max_arm_error:.4f} rad "
            f"(limit={arm_tolerance:.4f}), max torso error="
            f"{max_torso_error:.4f} (limit={torso_tolerance:.4f})",
        )

    def _box_observation_ready(self) -> tuple[bool, str]:
        return self._observation_ready(
            enabled_parameter="box_observation_ready_check_enabled",
            left_target_parameter="box_grasp_left_observation_joint_positions",
            right_target_parameter="box_grasp_right_observation_joint_positions",
            torso_target_parameter="box_grasp_torso_prepare_positions",
            max_age_parameter="box_observation_feedback_max_age_sec",
            torso_tolerance_parameter="box_observation_torso_tolerance",
            label="box",
        )

    def _grasp_observation_ready(self) -> tuple[bool, str]:
        return self._observation_ready(
            enabled_parameter="grasp_observation_ready_check_enabled",
            left_target_parameter="grasp_left_joint_positions",
            right_target_parameter="grasp_right_joint_positions",
            torso_target_parameter="torso_prepare_positions",
            max_age_parameter="grasp_observation_feedback_max_age_sec",
            torso_tolerance_parameter="grasp_observation_torso_tolerance",
            label="grasp",
        )

    def _stack_default_ready(self) -> tuple[bool, str]:
        if not self._boolean("stack_start_ready_check_enabled"):
            return True, "stack default readiness check is disabled"

        now = time.monotonic()
        max_age = self._float("stack_start_feedback_max_age_sec")
        with self.joint_state_lock:
            arm_positions = dict(self.latest_joint_positions)
            arm_age = now - self.latest_joint_state_time
            torso_positions = list(self.latest_torso_positions)
            torso_age = now - self.latest_torso_state_time

        left_names = self._string_array("left_arm_joint_names")
        right_names = self._string_array("right_arm_joint_names")
        targets = dict(
            zip(
                left_names,
                self._float_array("stack_default_left_joint_positions"),
            )
        )
        targets.update(
            zip(
                right_names,
                self._float_array("stack_default_right_joint_positions"),
            )
        )
        missing = [name for name in targets if name not in arm_positions]
        if missing:
            return False, f"missing arm feedback joints={missing}"
        if arm_age > max_age:
            return (
                False,
                f"arm feedback is stale: age={arm_age:.2f}s, limit={max_age:.2f}s",
            )
        if len(torso_positions) < 4:
            return False, "missing torso feedback"
        if torso_age > max_age:
            return (
                False,
                f"torso feedback is stale: age={torso_age:.2f}s, "
                f"limit={max_age:.2f}s",
            )

        errors = {
            name: abs(arm_positions[name] - target)
            for name, target in targets.items()
        }
        max_error = max(errors.values(), default=0.0)
        arm_tolerance = self._float("arm_joint_target_tolerance")
        torso_targets = self._float_array("stack_pickup_torso_positions")
        torso_error = max(
            (
                abs(actual - target)
                for actual, target in zip(torso_positions, torso_targets)
            ),
            default=0.0,
        )
        torso_tolerance = self._float("torso_target_tolerance")
        return (
            max_error <= arm_tolerance and torso_error <= torso_tolerance,
            f"max arm error={max_error:.4f} rad "
            f"(limit={arm_tolerance:.4f}), max torso error="
            f"{torso_error:.4f} (limit={torso_tolerance:.4f})",
        )

    def _wait_for_arm_joint_targets(
        self,
        goal_handle,
        left_positions: list[float],
        right_positions: list[float],
    ) -> None:
        if not self._boolean("verify_arm_joint_targets"):
            return
        targets = dict(
            zip(self._string_array("left_arm_joint_names"), left_positions)
        )
        targets.update(
            zip(self._string_array("right_arm_joint_names"), right_positions)
        )
        tolerance = self._float("arm_joint_target_tolerance")
        timeout_sec = self._float("arm_joint_target_wait_timeout_sec")
        required_stable = self._integer("arm_joint_target_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequence = -1
        last_errors: dict[str, float] = {}
        missing = list(targets)

        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while verifying the arm observation posture"
            )
            with self.joint_state_lock:
                sequence = self.latest_joint_state_sequence
                measured = dict(self.latest_joint_positions)
            if sequence == last_sequence:
                time.sleep(0.02)
                continue
            last_sequence = sequence
            missing = [name for name in targets if name not in measured]
            if missing:
                stable_samples = 0
                continue
            last_errors = {
                name: abs(measured[name] - target)
                for name, target in targets.items()
            }
            if max(last_errors.values(), default=0.0) <= tolerance:
                stable_samples += 1
                if stable_samples >= required_stable:
                    self.get_logger().info(
                        "verified arm joint target from /joint_states: "
                        f"max_error={max(last_errors.values(), default=0.0):.4f}"
                    )
                    return
            else:
                stable_samples = 0
            time.sleep(0.02)

        if missing:
            detail = f"missing joints={missing}"
        else:
            worst = sorted(
                last_errors.items(), key=lambda item: item[1], reverse=True
            )[:3]
            detail = f"largest errors={dict(worst)}"
        raise MissionError(
            "arm joint action returned before the commanded posture was "
            f"confirmed within {tolerance:.3f} rad for {required_stable} fresh "
            f"samples ({detail})"
        )

    def _publish_both_grippers(self, goal_handle, position: float) -> None:
        for gripper_arm in ("left", "right"):
            self._publish_gripper(goal_handle, gripper_arm, position)

    def _open_grippers(
        self,
        goal_handle,
        arms: tuple[str, ...],
        settle_context: str,
    ) -> None:
        if not arms:
            raise MissionError("at least one gripper is required")
        invalid_arms = [arm for arm in arms if arm not in VALID_ARMS]
        if invalid_arms:
            raise MissionError(
                f"invalid gripper arms: {', '.join(invalid_arms)}"
            )
        open_position = self._float("gripper_open_position")
        for arm in arms:
            self._publish_gripper(goal_handle, arm, open_position)
        self._wait_delay(
            goal_handle,
            self._float("gripper_settle_sec"),
            settle_context,
        )

    def _close_grippers_and_measure(
        self,
        goal_handle,
        arms: tuple[str, ...],
        settle_context: str,
        *,
        require_feedback: bool = True,
    ) -> dict[str, float]:
        """Close grippers once and optionally return fresh feedback values."""
        if not arms:
            raise MissionError("at least one gripper is required")
        invalid_arms = [arm for arm in arms if arm not in VALID_ARMS]
        if invalid_arms:
            raise MissionError(
                f"invalid gripper arms: {', '.join(invalid_arms)}"
            )

        feedback_sequences = (
            {
                arm: self._gripper_feedback_sequence(arm)
                for arm in arms
            }
            if require_feedback
            else {}
        )
        closed_position = self._float("gripper_closed_position")
        for arm in arms:
            self._publish_gripper(goal_handle, arm, closed_position)
        self._wait_delay(
            goal_handle,
            self._float("gripper_settle_sec"),
            settle_context,
        )
        if not require_feedback:
            return {}
        return {
            arm: self._wait_for_gripper_close_feedback(
                goal_handle,
                arm,
                feedback_sequences[arm],
            )
            for arm in arms
        }

    def _prepare_grasp_grippers(self, goal_handle) -> None:
        self._open_grippers(
            goal_handle,
            ("left", "right"),
            "while waiting for gripper preparation",
        )

    def _prepare_grasp_torso(self, goal_handle) -> None:
        self._publish_torso(
            goal_handle, self._float_array("torso_prepare_positions")
        )
        self._wait_delay(
            goal_handle,
            self._float("torso_settle_sec"),
            "while waiting for torso preparation",
        )

    def _prepare_grasp_arms_and_torso(self, goal_handle) -> None:
        self._call_arm_joints(
            goal_handle,
            self._float_array("grasp_left_joint_positions"),
            self._float_array("grasp_right_joint_positions"),
            False,
            goal_accepted_callback=lambda: self._prepare_grasp_torso(goal_handle),
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while waiting for synchronized arm and torso preparation",
        )

    def _prepare_observation_intermediate_arms(self, goal_handle) -> None:
        self._call_arm_joints(
            goal_handle,
            self._float_array("observation_intermediate_left_joint_positions"),
            self._float_array("observation_intermediate_right_joint_positions"),
            False,
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while waiting for intermediate arm preparation",
        )

    def _prepare_grasp_concurrently(
        self, goal_handle, open_grippers: bool
    ) -> None:
        intermediate_tasks = [self._prepare_observation_intermediate_arms]
        if open_grippers:
            intermediate_tasks.insert(0, self._prepare_grasp_grippers)

        with ThreadPoolExecutor(
            max_workers=len(intermediate_tasks),
            thread_name_prefix="grasp_intermediate",
        ) as executor:
            futures = [
                executor.submit(task, goal_handle) for task in intermediate_tasks
            ]
            for future in futures:
                future.result()

        # Dispatch the torso target as soon as the final /move_arm_j goal is
        # accepted.  The arm trajectory and waist motion therefore begin in
        # the same phase instead of two independent worker threads racing.
        self._prepare_grasp_arms_and_torso(goal_handle)

    def _prepare_box_grasp_grippers(self, goal_handle) -> None:
        self._open_grippers(
            goal_handle,
            ("left", "right"),
            "while waiting for box gripper preparation",
        )

    def _prepare_box_grasp_torso(self, goal_handle) -> None:
        self._publish_torso(
            goal_handle,
            self._float_array("box_grasp_torso_prepare_positions"),
        )
        self._wait_delay(
            goal_handle,
            self._float("torso_settle_sec"),
            "while waiting for box torso preparation",
        )

    def _prepare_box_grasp_arms_and_torso(self, goal_handle) -> None:
        self._call_arm_joints(
            goal_handle,
            self._float_array("box_grasp_left_observation_joint_positions"),
            self._float_array("box_grasp_right_observation_joint_positions"),
            False,
            goal_accepted_callback=lambda: self._prepare_box_grasp_torso(
                goal_handle
            ),
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while waiting for synchronized box arm and torso preparation",
        )

    def _prepare_box_grasp_concurrently(self, goal_handle) -> None:
        self._prepare_observation_intermediate_arms(goal_handle)
        self._prepare_box_grasp_arms_and_torso(goal_handle)

    def _wait_for_box_detection_posture(self, goal_handle) -> None:
        self._publish_box_grasp_feedback(
            goal_handle,
            "WAITING_FOR_BOX_OBSERVATION",
            "waiting for measured arm and torso feedback to confirm the box "
            "observation posture",
        )
        self._wait_for_arm_joint_targets(
            goal_handle,
            self._float_array("box_grasp_left_observation_joint_positions"),
            self._float_array("box_grasp_right_observation_joint_positions"),
        )
        self._wait_for_torso_target(
            goal_handle,
            self._float_array("box_grasp_torso_prepare_positions"),
            "confirming the box observation torso before detection",
        )
        settle_sec = self._float("box_detection_posture_settle_sec")
        self._publish_box_grasp_feedback(
            goal_handle,
            "SETTLING_BOX_OBSERVATION",
            "box observation arms and torso confirmed; holding still for "
            f"{settle_sec:.1f}s before FoundationPose",
        )
        self._wait_delay(
            goal_handle,
            settle_sec,
            "while holding the confirmed box observation posture before detection",
        )

    def _prepare_box_pickup_clearance_arms(self, goal_handle) -> None:
        self._call_arm_joints(
            goal_handle,
            self._float_array("box_pickup_clearance_left_joint_positions"),
            self._float_array("box_pickup_clearance_right_joint_positions"),
            False,
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while waiting for the box pickup clearance posture",
        )

    def _call_task_action(
        self,
        goal_handle,
        client,
        action_name: str,
        action_goal,
        timeout_sec: float,
        active_handle_attribute: str,
        feedback_callback=None,
        goal_accepted_callback=None,
    ):
        wait_deadline = time.monotonic() + self._float(
            "dependency_wait_timeout_sec"
        )
        while time.monotonic() < wait_deadline:
            self._check_canceled(goal_handle, f"while waiting for {action_name}")
            remaining = max(0.0, wait_deadline - time.monotonic())
            if client.wait_for_server(timeout_sec=min(0.5, remaining)):
                break
        else:
            raise MissionError(
                f"timeout waiting for action {action_name} after "
                f"{self._float('dependency_wait_timeout_sec'):.1f}s"
            )

        send_future = client.send_goal_async(
            action_goal, feedback_callback=feedback_callback
        )
        action_handle = self._wait_future(
            send_future,
            goal_handle,
            f"sending {action_name} goal",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        if action_handle is None or not action_handle.accepted:
            raise MissionError(f"{action_name} goal was rejected")

        with self.state_lock:
            setattr(self, active_handle_attribute, action_handle)
        try:
            if goal_accepted_callback is not None:
                try:
                    goal_accepted_callback()
                except Exception:
                    action_handle.cancel_goal_async()
                    raise

            result_future = action_handle.get_result_async()
            deadline = time.monotonic() + timeout_sec
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested:
                    action_handle.cancel_goal_async()
                    raise MissionCanceled(f"mission canceled during {action_name}")
                if time.monotonic() >= deadline:
                    action_handle.cancel_goal_async()
                    raise MissionError(
                        f"timeout waiting for {action_name} result after "
                        f"{timeout_sec:.1f}s"
                    )
                time.sleep(0.05)
            if not rclpy.ok():
                raise MissionError(f"ROS shutdown while waiting for {action_name}")
            wrapped_result = result_future.result()
        finally:
            with self.state_lock:
                setattr(self, active_handle_attribute, None)

        action_result = wrapped_result.result
        succeeded = wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
        if not succeeded or not action_result.success:
            raise TaskActionError(
                f"{action_name} failed: {action_result.message} "
                f"(error_code={action_result.error_code})",
                int(action_result.error_code),
            )
        return action_result

    def _call_arm_joints(
        self,
        goal_handle,
        left_positions: list[float],
        right_positions: list[float],
        dry_run: bool,
        goal_accepted_callback=None,
        duration: float = 0.0,
    ) -> str:
        if dry_run:
            return "dry run: skipped non-planning /move_arm_j action"
        action_name = self._string("arm_joints_service_name")
        action_goal = MoveArmJoints.Goal()
        action_goal.left_joints = left_positions
        action_goal.right_joints = right_positions
        action_goal.dry_run = False
        action_goal.duration = float(duration)
        try:
            response = self._call_task_action(
                goal_handle,
                self.arm_joints_client,
                action_name,
                action_goal,
                self._float("arm_joints_result_timeout_sec"),
                "active_arm_joints_goal_handle",
                goal_accepted_callback=goal_accepted_callback,
            )
        except MissionError as exc:
            if "hardware trajectory execution failed or runtime state froze" not in str(
                exc
            ):
                raise
            self.get_logger().warning(
                f"{action_name} reported a transient runtime failure; "
                "checking measured joints before deciding whether motion failed"
            )
            self._wait_for_arm_joint_targets(
                goal_handle, left_positions, right_positions
            )
            self.get_logger().warning(
                f"{action_name} runtime error overridden because measured "
                "joints reached the commanded posture"
            )
            return f"measured target reached after transient result: {exc}"
        self._wait_for_arm_joint_targets(
            goal_handle, left_positions, right_positions
        )
        return str(response.message)

    def _call_home(self, goal_handle, dry_run: bool) -> str:
        action_name = self._string("home_service_name")
        action_goal = Home.Goal()
        action_goal.dry_run = dry_run
        action_goal.duration = 0.0
        response = self._call_task_action(
            goal_handle,
            self.home_client,
            action_name,
            action_goal,
            self._float("home_result_timeout_sec"),
            "active_home_goal_handle",
        )
        return str(response.message)

    def _call_go_ready(self, goal_handle, dry_run: bool) -> str:
        action_name = self._string("go_ready_action_name")
        action_goal = GoReady.Goal()
        action_goal.dry_run = dry_run
        action_goal.duration = 0.0
        response = self._call_task_action(
            goal_handle,
            self.go_ready_client,
            action_name,
            action_goal,
            self._float("go_ready_result_timeout_sec"),
            "active_go_ready_goal_handle",
        )
        return str(response.message)

    def _safe_pre_arm_torso_reset(self, goal_handle) -> bool:
        try:
            self._publish_torso(
                goal_handle,
                self._float_array("torso_reset_positions"),
                require_subscriber=False,
                honor_cancel=False,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"failed to publish torso cleanup command: {exc}")
            return False
