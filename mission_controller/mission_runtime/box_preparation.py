import json
import math
import time

from geometry_msgs.msg import Pose
from rm_robot_interfaces.srv import StringCmd

try:
    from task_interfaces.srv import MoveCartesian
except ImportError:
    MoveCartesian = None
try:
    from object_pose_interfaces.action import EstimateObjectPose
except ModuleNotFoundError:
    EstimateObjectPose = None

from .common import MissionError


class BoxPreparationMixin:
    """Body/arm feedback waits and pre-detection preparation motion."""

    def _joint1_feedback_snapshot(self):
        with self.joint_state_lock:
            return (
                self.latest_body_joint1_position,
                self.latest_body_joint1_velocity,
                self.latest_body_joint1_state_time,
                self.latest_body_joint1_state_sequence,
            )

    def _body_feedback_snapshot(self):
        with self.joint_state_lock:
            names = [self._string(f"box_joint{index}_name") for index in range(1, 5)]
            return (
                [self.latest_body_joint_positions.get(name) for name in names],
                [self.latest_body_joint_velocities.get(name) for name in names],
                self.latest_body_state_time,
                self.latest_body_state_sequence,
            )

    def _wait_for_body_joints_target(
        self,
        goal_handle,
        target_angles_rad,
        *,
        sequence_after=-1,
        timeout_parameter="box_joint1_wait_timeout_sec",
    ):
        timeout_sec = self._float(timeout_parameter)
        position_tolerance = self._float("box_joint1_position_tolerance_rad")
        velocity_tolerance = self._float("box_joint1_velocity_tolerance_rad_sec")
        max_age_sec = self._float("box_joint1_feedback_max_age_sec")
        required_stable = self._integer("box_joint1_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequence = -1
        latest_positions = [None] * 4
        latest_velocities = [None] * 4
        latest_age = math.inf
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for body joints")
            positions, velocities, state_time, sequence = self._body_feedback_snapshot()
            latest_positions, latest_velocities = positions, velocities
            latest_age = time.monotonic() - state_time
            if sequence != last_sequence:
                last_sequence = sequence
                sample_matches = (
                    sequence > sequence_after
                    and latest_age <= max_age_sec
                    and all(
                        position is not None
                        and velocity is not None
                        and abs(position - target) <= position_tolerance
                        and abs(velocity) <= velocity_tolerance
                        for position, velocity, target in zip(
                            positions[: len(target_angles_rad)],
                            velocities[: len(target_angles_rad)],
                            target_angles_rad,
                        )
                    )
                )
                stable_samples = stable_samples + 1 if sample_matches else 0
                if stable_samples >= required_stable:
                    return [float(value) for value in positions]
            time.sleep(0.02)
        raise MissionError(
            "body joints did not reach requested targets: "
            f"targets={target_angles_rad}, measured={latest_positions}, "
            f"velocities={latest_velocities}, feedback_age_sec={latest_age:.3f}, "
            f"timeout_sec={timeout_sec:.1f}"
        )

    def _wait_for_fresh_body_feedback(
        self,
        goal_handle,
        *,
        sequence_after: int = -1,
    ) -> tuple[list[float], list[float], int]:
        """Return a fresh complete four-joint body sample.

        The detection posture is not assumed to be the configured zero pose.
        A replacement robot may be left at any valid Joint1-4 state when a
        GraspBox goal starts, so motion planning must use the measured state
        and only use the configured angles as the subsequent target.
        """
        timeout_sec = self._float("box_joint1_wait_timeout_sec")
        max_age_sec = self._float("box_joint1_feedback_max_age_sec")
        deadline = time.monotonic() + timeout_sec
        latest_positions = [None] * 4
        latest_velocities = [None] * 4
        latest_age = math.inf
        latest_sequence = -1
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for fresh body feedback")
            positions, velocities, state_time, sequence = self._body_feedback_snapshot()
            latest_positions = positions
            latest_velocities = velocities
            latest_sequence = sequence
            latest_age = time.monotonic() - state_time
            if (
                sequence > sequence_after
                and latest_age <= max_age_sec
                and len(positions) >= 4
                and len(velocities) >= 4
                and all(value is not None for value in positions[:4])
                and all(value is not None for value in velocities[:4])
            ):
                return (
                    [float(value) for value in positions[:4]],
                    [float(value) for value in velocities[:4]],
                    sequence,
                )
            time.sleep(0.02)
        raise MissionError(
            "fresh /mcap/body feedback did not contain all four joints: "
            f"measured={latest_positions}, velocities={latest_velocities}, "
            f"sequence={latest_sequence}, feedback_age_sec={latest_age:.3f}, "
            f"timeout_sec={timeout_sec:.1f}"
        )

    def _wait_for_post_arm_joint_targets(
        self,
        goal_handle,
        left_target_rad,
        right_target_rad,
        sequence_after,
        # Legacy post-arm feedback parameters omit the ``_movej`` segment,
        # while the new pre-target block keeps it in the parameter names.
        parameter_prefix="box_post_arm",
        description="post-grasp arm MoveJ",
        active_arms=("left", "right"),
    ) -> None:
        """Require selected arm feedback streams to reach seven-joint targets."""
        tolerance = self._float(f"{parameter_prefix}_position_tolerance_rad")
        velocity_tolerance = self._float(
            f"{parameter_prefix}_velocity_tolerance_rad_sec"
        )
        max_age = self._float(f"{parameter_prefix}_feedback_max_age_sec")
        timeout_parameter = (
            "box_post_arm_movej_timeout_sec"
            if parameter_prefix == "box_post_arm"
            else f"{parameter_prefix}_timeout_sec"
        )
        timeout_sec = self._float(timeout_parameter)
        stable_required = self._integer(f"{parameter_prefix}_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequences = {arm: -1 for arm in active_arms}
        latest_detail = "no feedback"
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, f"while verifying {description}")
            now = time.monotonic()
            with self.joint_state_lock:
                sequences = dict(self.latest_slave_arm_state_sequences)
                positions = {
                    arm: list(values)
                    for arm, values in self.latest_slave_arm_positions.items()
                }
                velocities = {
                    arm: list(values)
                    for arm, values in self.latest_slave_arm_velocities.items()
                }
                times = dict(self.latest_slave_arm_state_times)
            if sequences == last_sequences:
                time.sleep(0.02)
                continue
            last_sequences = sequences
            matches = True
            error_detail = []
            targets = {"left": left_target_rad, "right": right_target_rad}
            for arm in active_arms:
                target = targets[arm]
                measured = positions.get(arm, [])
                measured_velocity = velocities.get(arm, [])
                age = now - times.get(arm, 0.0)
                if (
                    sequences.get(arm, 0) <= sequence_after.get(arm, -1)
                    or len(measured) < 7
                    or len(measured_velocity) < 7
                    or age > max_age
                ):
                    matches = False
                    error_detail.append(
                        f"{arm}=missing_or_stale(seq={sequences.get(arm, 0)}, age={age:.3f})"
                    )
                    continue
                position_error = max(
                    abs(measured[index] - target[index]) for index in range(7)
                )
                velocity_error = max(
                    abs(measured_velocity[index]) for index in range(7)
                )
                error_detail.append(
                    f"{arm}=pos_err={position_error:.4f}, vel={velocity_error:.4f}"
                )
                if position_error > tolerance or velocity_error > velocity_tolerance:
                    matches = False
            latest_detail = "; ".join(error_detail)
            stable_samples = stable_samples + 1 if matches else 0
            if stable_samples >= stable_required:
                return
            time.sleep(0.02)
        raise MissionError(
            f"{description} targets were not confirmed: "
            f"{latest_detail}; timeout_sec={timeout_sec:.1f}"
        )

    @staticmethod
    def _parse_string_command_response(response, description: str) -> None:
        try:
            response_data = json.loads(str(response.data).strip())
        except (TypeError, ValueError) as exc:
            raise MissionError(
                f"invalid {description} service response: {response.data!r}"
            ) from exc
        if not response_data.get("receive_state", False):
            raise MissionError(
                f"{description} command was not accepted: {response.data}"
            )

    def _post_arm_movej_targets(self):
        units_per_degree = self._float("box_post_arm_movej_command_units_per_degree")
        left_units = [
            int(value)
            for value in self._float_array("box_post_arm_movej_left_joint_units")
        ]
        right_units = [
            int(value)
            for value in self._float_array("box_post_arm_movej_right_joint_units")
        ]
        target_left = [
            math.radians(float(value) / units_per_degree) for value in left_units
        ]
        target_right = [
            math.radians(float(value) / units_per_degree) for value in right_units
        ]
        return left_units, right_units, target_left, target_right

    def _configured_dual_arm_movej_targets(self, prefix: str):
        """Read a dual-arm MoveJ configuration and convert units to radians."""
        units_per_degree = self._float(f"{prefix}_command_units_per_degree")
        left_units = [
            int(value) for value in self._float_array(f"{prefix}_left_joint_units")
        ]
        right_units = [
            int(value) for value in self._float_array(f"{prefix}_right_joint_units")
        ]
        target_left = [
            math.radians(float(value) / units_per_degree) for value in left_units
        ]
        target_right = [
            math.radians(float(value) / units_per_degree) for value in right_units
        ]
        return left_units, right_units, target_left, target_right

    def _current_dual_arm_prepare_targets(self, active_arms=("left", "right")):
        """Build stage-one targets from fresh seven-joint arm feedback."""
        units_per_degree = self._float(
            "box_pre_target_arm_movej_command_units_per_degree"
        )
        joint2_units = self._integer("box_pre_target_arm_movej_stage1_joint2_units")
        max_age = self._float("box_pre_target_arm_movej_feedback_max_age_sec")
        now = time.monotonic()
        units_by_arm = {}
        targets_by_arm = {}
        detail_by_arm = {}
        with self.joint_state_lock:
            for arm in active_arms:
                positions = list(self.latest_slave_arm_positions.get(arm, []))
                sequence = self.latest_slave_arm_state_sequences.get(arm, 0)
                age = now - self.latest_slave_arm_state_times.get(arm, 0.0)
                if sequence <= 0 or len(positions) < 7 or age > max_age:
                    raise MissionError(
                        "fresh arm feedback required for prepare stage 1: "
                        f"{arm}=sequence={sequence}, joints={len(positions)}, "
                        f"feedback_age_sec={age:.3f}, timeout_sec={max_age:.1f}"
                    )
                if not all(math.isfinite(value) for value in positions[:7]):
                    raise MissionError(
                        f"arm feedback for prepare stage 1 contains invalid values: {arm}"
                    )
                units = [
                    int(round(math.degrees(value) * units_per_degree))
                    for value in positions[:7]
                ]
                units[1] = joint2_units
                units_by_arm[arm] = units
                targets_by_arm[arm] = [
                    math.radians(float(value) / units_per_degree) for value in units
                ]
                detail_by_arm[arm] = (
                    f"{arm}=seq={sequence}, age={age:.3f}, " f"joint_units={units}"
                )
        return units_by_arm, targets_by_arm, detail_by_arm

    def _current_step4_movej_targets(self, active_arms=("left", "right")):
        """Preserve live arm joints while overriding Joint2 for Step4."""
        units_per_degree = self._float(
            "box_post_movel_step4_movej_command_units_per_degree"
        )
        joint2_units = self._integer("box_post_movel_step4_movej_joint2_units")
        max_age = self._float("box_post_movel_step4_movej_feedback_max_age_sec")
        now = time.monotonic()
        units_by_arm = {}
        targets_by_arm = {}
        detail_by_arm = {}
        with self.joint_state_lock:
            for arm in active_arms:
                positions = list(self.latest_slave_arm_positions.get(arm, []))
                sequence = self.latest_slave_arm_state_sequences.get(arm, 0)
                age = now - self.latest_slave_arm_state_times.get(arm, 0.0)
                if sequence <= 0 or len(positions) < 7 or age > max_age:
                    raise MissionError(
                        "fresh arm feedback required for Step4 MoveJ: "
                        f"{arm}=sequence={sequence}, joints={len(positions)}, "
                        f"feedback_age_sec={age:.3f}, timeout_sec={max_age:.1f}"
                    )
                if not all(math.isfinite(value) for value in positions[:7]):
                    raise MissionError(
                        f"arm feedback for Step4 MoveJ contains invalid values: {arm}"
                    )
                units = [
                    int(round(math.degrees(value) * units_per_degree))
                    for value in positions[:7]
                ]
                units[1] = joint2_units
                units_by_arm[arm] = units
                targets_by_arm[arm] = [
                    math.radians(float(value) / units_per_degree) for value in units
                ]
                detail_by_arm[arm] = (
                    f"{arm}=seq={sequence}, age={age:.3f}, " f"joint_units={units}"
                )
        return units_by_arm, targets_by_arm, detail_by_arm

    def _execute_dual_arm_movej_targets(
        self,
        goal_handle,
        dry_run: bool,
        prefix: str,
        left_units,
        right_units,
        left_target,
        right_target,
        detail_prefix: str,
        feedback_prefix: str,
        feedback_move_stage: str,
        feedback_wait_stage: str,
        feedback_reached_stage: str,
        description: str,
        active_arms=("left", "right"),
        velocity_parameter=None,
    ) -> str:
        """Send selected arm MoveJ targets and verify their feedback."""
        active_arms = tuple(active_arms)
        if not active_arms or any(arm not in ("left", "right") for arm in active_arms):
            raise MissionError(f"invalid active arms for MoveJ: {active_arms}")
        units_by_arm = {"left": left_units, "right": right_units}
        devices_by_arm = {
            arm: self._integer(f"{prefix}_{arm}_device") for arm in active_arms
        }
        detail = f"{detail_prefix}: " + ", ".join(
            f"{arm}_device={devices_by_arm[arm]}, "
            f"{arm}_joint_units={units_by_arm[arm]}"
            for arm in active_arms
        )
        self._publish_box_grasp_feedback(
            goal_handle, f"{feedback_prefix}_TARGETS", detail
        )
        if dry_run:
            return f"{detail}; skipped in dry-run"

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        if velocity_parameter is None:
            velocity_parameter = (
                f"{prefix}_velocity"
                if prefix == "box_post_movel_step4_movej"
                else "box_preparation_movej_velocity"
            )
        common = {
            "command": "movej",
            "v": self._integer(velocity_parameter),
            "r": self._integer(f"{prefix}_blend_radius"),
            "trajectory_connect": self._integer(f"{prefix}_trajectory_connect"),
        }
        requests = {}
        for arm in active_arms:
            payload = {
                "device": devices_by_arm[arm],
                "payload": {**common, "joint": units_by_arm[arm]},
            }
            request = StringCmd.Request()
            request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
            requests[arm] = request
        with self.joint_state_lock:
            sequence_before = dict(self.latest_slave_arm_state_sequences)
        self._publish_box_grasp_feedback(
            goal_handle,
            feedback_move_stage,
            f"sending {description} for {', '.join(active_arms)} arm(s)",
        )
        futures = {
            arm: self.body_command_client.call_async(requests[arm])
            for arm in active_arms
        }
        for arm in active_arms:
            response = self._wait_future(
                futures[arm],
                goal_handle,
                f"calling {description} {arm} arm MoveJ",
                self._float("dependency_wait_timeout_sec"),
                cancel_local_future=False,
            )
            self._parse_string_command_response(
                response, f"{description} {arm} arm MoveJ"
            )
        self._publish_box_grasp_feedback(
            goal_handle,
            feedback_wait_stage,
            f"{description} commands accepted; waiting for fresh position and zero-velocity feedback for {', '.join(active_arms)} arm(s)",
        )
        self._wait_for_post_arm_joint_targets(
            goal_handle,
            left_target,
            right_target,
            sequence_before,
            parameter_prefix=prefix,
            description=description,
            active_arms=active_arms,
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            feedback_reached_stage,
            f"{', '.join(active_arms)} arm(s) reached the {description} targets with stable feedback",
        )
        return f"{detail}; arm_feedback=confirmed"

    def _execute_configured_dual_arm_movej(
        self,
        goal_handle,
        dry_run: bool,
        prefix: str,
        detail_prefix: str,
        feedback_prefix: str,
        feedback_move_stage: str,
        feedback_wait_stage: str,
        feedback_reached_stage: str,
        description: str,
        active_arms=("left", "right"),
    ) -> str:
        """Send two configured arm MoveJ commands concurrently and verify them."""
        left_units, right_units, left_target, right_target = (
            self._configured_dual_arm_movej_targets(prefix)
        )
        return self._execute_dual_arm_movej_targets(
            goal_handle,
            dry_run,
            prefix,
            left_units,
            right_units,
            left_target,
            right_target,
            detail_prefix,
            feedback_prefix,
            feedback_move_stage,
            feedback_wait_stage,
            feedback_reached_stage,
            description,
            active_arms=active_arms,
        )

    def _execute_pre_target_arm_movej(
        self, goal_handle, dry_run: bool, right_arm_only: bool = False
    ) -> str:
        if not self._boolean("box_pre_target_arm_movej_enabled"):
            return "pre_target_arm_movej=disabled"
        if not self._boolean("box_pre_target_arm_movej_two_stage_enabled"):
            return self._execute_configured_dual_arm_movej(
                goal_handle,
                dry_run,
                "box_pre_target_arm_movej",
                "pre-target dual arm MoveJ",
                "PRE_TARGET_ARM_MOVEJ",
                "MOVING_PRE_TARGET_ARM_JOINTS",
                "WAITING_FOR_PRE_TARGET_ARM_JOINTS",
                "PRE_TARGET_ARM_JOINTS_REACHED",
                "pre-target dual arm MoveJ",
                active_arms=("right",) if right_arm_only else ("left", "right"),
            )
        return self._execute_two_stage_prepare_arm_movej(
            goal_handle,
            dry_run,
            "pre-target",
            "PRE_TARGET_ARM_MOVEJ",
            "pre-target dual arm MoveJ",
            right_arm_only=right_arm_only,
        )

    def _execute_pre_detection_arm_movej(
        self,
        goal_handle,
        dry_run: bool,
        right_arm_only: bool = False,
        active_arms=None,
    ) -> str:
        if not self._boolean("box_pre_detection_arm_movej_enabled"):
            return "pre_detection_arm_movej=disabled"
        if active_arms is None:
            active_arms = ("right",) if right_arm_only else ("left", "right")
        else:
            active_arms = tuple(active_arms)
            if not active_arms or any(
                arm not in ("left", "right") for arm in active_arms
            ):
                raise MissionError(
                    f"invalid active arms for pre-detection MoveJ: {active_arms}"
                )
        if not self._boolean("box_pre_target_arm_movej_two_stage_enabled"):
            return self._execute_configured_dual_arm_movej(
                goal_handle,
                dry_run,
                "box_pre_target_arm_movej",
                "pre-detection dual arm MoveJ",
                "PRE_DETECTION_ARM_MOVEJ",
                "MOVING_PRE_DETECTION_ARM_JOINTS",
                "WAITING_FOR_PRE_DETECTION_ARM_JOINTS",
                "PRE_DETECTION_ARM_JOINTS_REACHED",
                "pre-detection dual arm MoveJ",
                active_arms=active_arms,
            )
        return self._execute_two_stage_prepare_arm_movej(
            goal_handle,
            dry_run,
            "pre-detection",
            "PRE_DETECTION_ARM_MOVEJ",
            "pre-detection dual arm MoveJ",
            right_arm_only=right_arm_only,
            active_arms=active_arms,
        )

    def _execute_two_stage_prepare_arm_movej(
        self,
        goal_handle,
        dry_run: bool,
        phase_label: str,
        feedback_prefix: str,
        description: str,
        right_arm_only: bool = False,
        active_arms=None,
    ) -> str:
        """Prepare selected arms in two stages while preserving live joints first."""
        prefix = "box_pre_target_arm_movej"
        if active_arms is None:
            active_arms = ("right",) if right_arm_only else ("left", "right")
        else:
            active_arms = tuple(active_arms)
        if not active_arms or any(arm not in ("left", "right") for arm in active_arms):
            raise MissionError(
                f"invalid active arms for two-stage MoveJ: {active_arms}"
            )
        if dry_run:
            stage1_detail = (
                f"{phase_label} stage1 dynamic {', '.join(active_arms)} arm MoveJ: "
                "live joint feedback read at execution time; skipped in dry-run"
            )
            stage2_details = []
            for arm in active_arms:
                stage2_units = [
                    int(value)
                    for value in self._float_array(f"{prefix}_{arm}_joint_units")
                ]
                stage2_details.append(f"{arm}_joint_units={stage2_units}")
            return (
                f"{stage1_detail}; {phase_label} stage2 {', '.join(active_arms)} arm MoveJ: "
                f"{', '.join(stage2_details)}; skipped in dry-run"
            )

        units_by_arm, targets_by_arm, feedback_detail = (
            self._current_dual_arm_prepare_targets(active_arms=active_arms)
        )
        stage1_detail = (
            f"{phase_label} stage1 dynamic {', '.join(active_arms)} arm MoveJ: "
            f"{'; '.join(feedback_detail[arm] for arm in active_arms)}"
        )
        stage1_result = self._execute_dual_arm_movej_targets(
            goal_handle,
            False,
            prefix,
            units_by_arm.get("left", []),
            units_by_arm.get("right", []),
            targets_by_arm.get("left", []),
            targets_by_arm.get("right", []),
            stage1_detail,
            f"{feedback_prefix}_STAGE1",
            f"MOVING_{feedback_prefix}_STAGE1",
            f"WAITING_FOR_{feedback_prefix}_STAGE1",
            f"{feedback_prefix}_STAGE1_REACHED",
            f"{description} stage1",
            active_arms=active_arms,
        )
        left_units, right_units, left_target, right_target = (
            self._configured_dual_arm_movej_targets(prefix)
        )
        stage2_result = self._execute_dual_arm_movej_targets(
            goal_handle,
            False,
            prefix,
            left_units,
            right_units,
            left_target,
            right_target,
            f"{phase_label} stage2 dual arm MoveJ",
            f"{feedback_prefix}_STAGE2",
            f"MOVING_{feedback_prefix}_STAGE2",
            f"WAITING_FOR_{feedback_prefix}_STAGE2",
            f"{feedback_prefix}_STAGE2_REACHED",
            f"{description} stage2",
            active_arms=active_arms,
        )
        return f"{stage1_result}; {stage2_result}"

    def _execute_pre_detection_arm_intermediate_movej(
        self, goal_handle, dry_run: bool, final_units: list[int], arm: str
    ) -> str:
        """Move detection Joint1 first while preserving live Joints 2-7."""
        if arm not in ("left", "right"):
            raise MissionError(f"invalid detection arm: {arm}")
        prefix = f"box_pre_detection_{arm}_movej"
        units_per_degree = self._float(f"{prefix}_command_units_per_degree")
        with self.joint_state_lock:
            positions = list(self.latest_slave_arm_positions.get(arm, []))
            sequence_before = self.latest_slave_arm_state_sequences.get(arm, 0)
            pose_sequence_before = self.latest_slave_arm_pose_sequences.get(arm, 0)
            state_time = self.latest_slave_arm_state_times.get(arm, 0.0)
        age = time.monotonic() - state_time
        max_age = self._float(f"{prefix}_feedback_max_age_sec")
        if not dry_run and (
            sequence_before <= 0 or len(positions) < 7 or age > max_age
        ):
            raise MissionError(
                f"fresh {arm}-arm feedback required before detection intermediate MoveJ: "
                f"sequence={sequence_before}, joints={len(positions)}, "
                f"feedback_age_sec={age:.3f}, timeout_sec={max_age:.1f}"
            )
        units = [int(value) for value in final_units]
        if not dry_run:
            units[1:] = [
                int(round(math.degrees(value) * units_per_degree))
                for value in positions[1:7]
            ]
        target = [math.radians(float(value) / units_per_degree) for value in units]
        detail = (
            f"pre-detection {arm} arm intermediate MoveJ: "
            f"device={self._integer(f'{prefix}_device')}, joint_units={units}"
        )
        self._publish_box_grasp_feedback(
            goal_handle, f"PRE_DETECTION_{arm.upper()}_INTERMEDIATE_TARGETS", detail
        )
        if dry_run:
            return f"{detail}; skipped in dry-run"

        self._wait_for_service(
            self.body_command_client,
            self._string("box_joint1_command_service_name"),
            goal_handle,
        )
        payload = {
            "device": self._integer(f"{prefix}_device"),
            "payload": {
                "command": "movej",
                "joint": units,
                "v": self._integer(f"{prefix}_velocity"),
                "r": self._integer(f"{prefix}_blend_radius"),
                "trajectory_connect": self._integer(f"{prefix}_trajectory_connect"),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            f"MOVING_{arm.upper()}_ARM_TO_DETECTION_INTERMEDIATE_POSE",
            f"sending the {arm}-arm intermediate detection MoveJ command",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            f"calling pre-detection {arm} arm intermediate MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        self._parse_string_command_response(
            response, f"pre-detection {arm} arm intermediate MoveJ"
        )
        tolerance = self._float(f"{prefix}_position_tolerance_rad")
        velocity_tolerance = self._float(f"{prefix}_velocity_tolerance_rad_sec")
        stable_required = self._integer(f"{prefix}_stable_samples")
        deadline = time.monotonic() + self._float(f"{prefix}_timeout_sec")
        stable_samples = 0
        latest_detail = f"no {arm}-arm feedback"
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle,
                f"while verifying the {arm}-arm intermediate detection MoveJ",
            )
            now = time.monotonic()
            with self.joint_state_lock:
                sequence = self.latest_slave_arm_state_sequences.get(arm, 0)
                measured = list(self.latest_slave_arm_positions.get(arm, []))
                velocity = list(self.latest_slave_arm_velocities.get(arm, []))
                state_time = self.latest_slave_arm_state_times.get(arm, 0.0)
                pose_sequence = self.latest_slave_arm_pose_sequences.get(arm, 0)
            age = now - state_time
            position_error = (
                max(abs(measured[index] - target[index]) for index in range(7))
                if len(measured) >= 7
                else float("inf")
            )
            velocity_error = (
                max(abs(value) for value in velocity[:7])
                if len(velocity) >= 7
                else float("inf")
            )
            matches = (
                sequence > sequence_before
                and len(measured) >= 7
                and len(velocity) >= 7
                and age <= max_age
                and position_error <= tolerance
                and velocity_error <= velocity_tolerance
                and pose_sequence > pose_sequence_before
            )
            latest_detail = (
                f"seq={sequence}, pose_seq={pose_sequence}, age={age:.3f}, "
                f"pos_err={position_error:.4f}, vel={velocity_error:.4f}"
            )
            stable_samples = stable_samples + 1 if matches else 0
            if stable_samples >= stable_required:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    f"{arm.upper()}_ARM_DETECTION_INTERMEDIATE_REACHED",
                    f"{arm} arm reached the intermediate detection pose",
                )
                return f"{detail}; arm_feedback=confirmed"
            time.sleep(0.02)
        raise MissionError(
            f"{arm} arm did not reach the intermediate detection pose: "
            f"{latest_detail}; timeout_sec={self._float(f'{prefix}_timeout_sec'):.1f}"
        )

    def _box_layer_pre_detection_arm_movej_joint_units(
        self,
        box_layer: int,
        model_label: str | None = None,
        *,
        arm: str = "right",
        tf_mode: bool = False,
        drag_mode: bool = False,
    ) -> list[int]:
        """Return configured detection-arm joints for a model/layer.

        ``model_label`` is supplied by the regular GraspBox mission so
        smallbox and bigbox can have independent observation poses.  The
        original generic table remains a fallback for DragBox and other
        callers that do not select a model explicitly.
        """
        if arm not in ("left", "right"):
            raise MissionError(f"invalid detection arm: {arm}")
        if box_layer < 1 or box_layer > 4:
            raise MissionError("box_layer must be in [1, 4]")
        configured = self._boolean_array(
            f"box_layer_pre_detection_{arm}_movej_configured"
        )
        if len(configured) != 4:
            raise MissionError(
                f"box_layer_pre_detection_{arm}_movej_configured must contain "
                "four values"
            )
        if not configured[box_layer - 1]:
            raise MissionError(
                f"box_layer {box_layer} {arm}-arm detection pose is not configured yet"
            )
        parameter_name = f"box_layer_pre_detection_{arm}_movej_joint_units"
        if tf_mode:
            action_prefix = "drag_box_tf" if drag_mode else "grasp_box_tf"
            parameter_name = self._tf_layer_parameter_name(
                action_prefix,
                f"box_layer_pre_detection_{arm}_movej_joint_units",
                model_label,
                box_layer,
            )
            values = self._float_array(parameter_name)
            if len(values) != 7:
                raise MissionError(f"{parameter_name} must contain seven joint values")
            if not all(math.isfinite(value) for value in values):
                raise MissionError(f"{parameter_name} contains invalid values")
            return [int(round(value)) for value in values]
        if model_label:
            normalized_model = str(model_label).strip().lower()
            if normalized_model in ("bigbox", "smallbox"):
                parameter_name = (
                    f"box_layer_pre_detection_{arm}_movej_joint_units_"
                    f"{normalized_model}"
                )
        values = self._float_array(parameter_name)
        if len(values) != 28:
            raise MissionError(
                f"{parameter_name} must contain 28 values "
                "(four layers x seven joints)"
            )
        start = (box_layer - 1) * 7
        selected = values[start : start + 7]
        if not all(math.isfinite(value) for value in selected):
            raise MissionError(
                f"box_layer {box_layer} {arm}-arm detection pose contains invalid values"
            )
        return [int(round(value)) for value in selected]

    def _execute_pre_detection_arm_movej_fixed(
        self,
        goal_handle,
        dry_run: bool,
        box_layer: int,
        model_label: str | None = None,
        *,
        arm: str = "right",
        tf_mode: bool = False,
        drag_mode: bool = False,
    ) -> str:
        """Move the selected wrist camera to its fixed detection configuration."""
        if arm not in ("left", "right"):
            raise MissionError(f"invalid detection arm: {arm}")
        prefix = f"box_pre_detection_{arm}_movej"
        if not self._boolean(f"{prefix}_enabled"):
            return f"{prefix}=disabled"
        units = self._box_layer_pre_detection_arm_movej_joint_units(
            box_layer,
            model_label,
            arm=arm,
            tf_mode=tf_mode,
            drag_mode=drag_mode,
        )
        intermediate_detail = self._execute_pre_detection_arm_intermediate_movej(
            goal_handle, dry_run, units, arm
        )
        units_per_degree = self._float(f"{prefix}_command_units_per_degree")
        target = [math.radians(float(value) / units_per_degree) for value in units]
        device = self._integer(f"{prefix}_device")
        detail = (
            f"pre-detection {arm} arm MoveJ: " f"device={device}, joint_units={units}"
        )
        self._publish_box_grasp_feedback(
            goal_handle, f"PRE_DETECTION_{arm.upper()}_MOVEJ_TARGETS", detail
        )
        if dry_run:
            return f"{intermediate_detail}; {detail}; skipped in dry-run"

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        with self.joint_state_lock:
            sequence_before = self.latest_slave_arm_state_sequences.get(arm, 0)
            pose_sequence_before = self.latest_slave_arm_pose_sequences.get(arm, 0)
        payload = {
            "device": device,
            "payload": {
                "command": "movej",
                "joint": units,
                "v": self._integer(f"{prefix}_velocity"),
                "r": self._integer(f"{prefix}_blend_radius"),
                "trajectory_connect": self._integer(f"{prefix}_trajectory_connect"),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            f"MOVING_{arm.upper()}_ARM_TO_DETECTION_POSE",
            f"sending the fixed {arm}-arm detection MoveJ command",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            f"calling pre-detection {arm} arm MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        self._parse_string_command_response(response, f"pre-detection {arm} arm MoveJ")
        self._publish_box_grasp_feedback(
            goal_handle,
            f"WAITING_FOR_{arm.upper()}_ARM_DETECTION_POSE",
            f"{arm}-arm detection MoveJ accepted; waiting for fresh position, zero velocity, and EEPose feedback",
        )
        deadline = time.monotonic() + self._float(f"{prefix}_timeout_sec")
        timeout_sec = self._float(f"{prefix}_timeout_sec")
        tolerance = self._float(f"{prefix}_position_tolerance_rad")
        velocity_tolerance = self._float(f"{prefix}_velocity_tolerance_rad_sec")
        max_age = self._float(f"{prefix}_feedback_max_age_sec")
        stable_required = self._integer(f"{prefix}_stable_samples")
        stable_samples = 0
        latest_detail = f"no {arm}-arm feedback"
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, f"while verifying the {arm}-arm detection MoveJ"
            )
            now = time.monotonic()
            with self.joint_state_lock:
                sequence = self.latest_slave_arm_state_sequences.get(arm, 0)
                measured = list(self.latest_slave_arm_positions.get(arm, []))
                velocity = list(self.latest_slave_arm_velocities.get(arm, []))
                state_time = self.latest_slave_arm_state_times.get(arm, 0.0)
                pose_sequence = self.latest_slave_arm_pose_sequences.get(arm, 0)
            age = now - state_time
            matches = (
                sequence > sequence_before
                and len(measured) >= 7
                and len(velocity) >= 7
                and age <= max_age
                and max(abs(measured[i] - target[i]) for i in range(7)) <= tolerance
                and max(abs(velocity[i]) for i in range(7)) <= velocity_tolerance
                and pose_sequence > pose_sequence_before
            )
            latest_detail = (
                f"seq={sequence}, pose_seq={pose_sequence}, age={age:.3f}, "
                f"pos_err={max(abs(measured[i] - target[i]) for i in range(7)) if len(measured) >= 7 else float('inf'):.4f}, "
                f"vel={max(abs(value) for value in velocity) if len(velocity) >= 7 else float('inf'):.4f}"
            )
            stable_samples = stable_samples + 1 if matches else 0
            if stable_samples >= stable_required:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    f"{arm.upper()}_ARM_DETECTION_POSE_REACHED",
                    f"{arm} arm reached the fixed detection pose with fresh EEPose feedback",
                )
                return f"{detail}; arm_feedback=confirmed"
            time.sleep(0.02)
        raise MissionError(
            f"{arm} arm did not reach the fixed detection pose: "
            f"{latest_detail}; timeout_sec={timeout_sec:.1f}"
        )

    def _drag_box_tf_post_detection_left_movej_joint_units(
        self, box_layer: int, model_label: str | None = None
    ) -> list[int]:
        """Return the DragBox-TF left-arm pose used after left-camera detection."""
        if box_layer < 1 or box_layer > 4:
            raise MissionError("box_layer must be in [1, 4]")
        model = str(model_label or "bigbox").strip().lower()
        if model not in ("bigbox", "smallbox"):
            model = "bigbox"
        parameter_name = (
            "drag_box_tf_box_layer_post_detection_left_movej_joint_units_"
            f"{model}_layer{box_layer}"
        )
        values = self._float_array(parameter_name)
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise MissionError(
                f"{parameter_name} must contain seven finite joint values"
            )
        return [int(round(value)) for value in values]

    def _execute_drag_box_tf_post_detection_left_movej(
        self,
        goal_handle,
        dry_run: bool,
        box_layer: int,
        model_label: str | None = None,
    ) -> str:
        """Move the left arm away from the camera pose after DragBox TF detection."""
        if not self._boolean("drag_box_tf_post_detection_left_movej_enabled"):
            return "drag_box_tf_post_detection_left_movej=disabled"
        prefix = "box_pre_detection_left_movej"
        units = self._drag_box_tf_post_detection_left_movej_joint_units(
            box_layer, model_label
        )
        units_per_degree = self._float(f"{prefix}_command_units_per_degree")
        target = [math.radians(float(value) / units_per_degree) for value in units]
        detail = (
            "post-detection left arm MoveJ (DragBox TF): "
            f"device={self._integer(f'{prefix}_device')}, "
            f"joint_units={units}"
        )
        self._publish_box_grasp_feedback(
            goal_handle, "POST_DETECTION_LEFT_ARM_MOVEJ_TARGETS", detail
        )
        if dry_run:
            return f"{detail}; skipped in dry-run"

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        with self.joint_state_lock:
            sequence_before = self.latest_slave_arm_state_sequences.get("left", 0)
        payload = {
            "device": self._integer(f"{prefix}_device"),
            "payload": {
                "command": "movej",
                "joint": units,
                "v": self._integer(f"{prefix}_velocity"),
                "r": self._integer(f"{prefix}_blend_radius"),
                "trajectory_connect": self._integer(f"{prefix}_trajectory_connect"),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_LEFT_ARM_TO_POST_DETECTION_POSE",
            "sending the configured left-arm post-detection MoveJ",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            "calling DragBox TF post-detection left arm MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        self._parse_string_command_response(
            response, "DragBox TF post-detection left arm MoveJ"
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "WAITING_FOR_POST_DETECTION_LEFT_ARM",
            "post-detection left-arm MoveJ accepted; waiting for stable feedback",
        )
        self._wait_for_post_arm_joint_targets(
            goal_handle,
            target,
            [],
            {"left": sequence_before},
            parameter_prefix=prefix,
            description="DragBox TF post-detection left arm MoveJ",
            active_arms=("left",),
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "POST_DETECTION_LEFT_ARM_REACHED",
            "left arm reached the configured post-detection pose with stable feedback",
        )
        return f"{detail}; arm_feedback=confirmed"

    # Backward-compatible right-arm entry points retained for existing
    # callers and tests.  New code should use the arm-generic helpers above.
    def _box_layer_pre_detection_right_movej_joint_units(
        self,
        box_layer: int,
        model_label: str | None = None,
        *,
        tf_mode: bool = False,
        drag_mode: bool = False,
    ) -> list[int]:
        # Some downstream unit-test harnesses bind only this legacy helper;
        # keep its original standalone behavior in that case.
        if not hasattr(self, "_box_layer_pre_detection_arm_movej_joint_units"):
            if box_layer < 1 or box_layer > 4:
                raise MissionError("box_layer must be in [1, 4]")
            configured = self._boolean_array(
                "box_layer_pre_detection_right_movej_configured"
            )
            if not configured[box_layer - 1]:
                raise MissionError(
                    f"box_layer {box_layer} right-arm detection pose is not configured yet"
                )
            parameter_name = "box_layer_pre_detection_right_movej_joint_units"
            if model_label and str(model_label).strip().lower() in (
                "bigbox",
                "smallbox",
            ):
                parameter_name += f"_{str(model_label).strip().lower()}"
            values = self._float_array(parameter_name)
            start = (box_layer - 1) * 7
            return [int(round(value)) for value in values[start : start + 7]]
        return self._box_layer_pre_detection_arm_movej_joint_units(
            box_layer, model_label, arm="right", tf_mode=tf_mode, drag_mode=drag_mode
        )

    def _execute_pre_detection_right_movej(
        self,
        goal_handle,
        dry_run: bool,
        box_layer: int,
        model_label: str | None = None,
        *,
        tf_mode: bool = False,
        drag_mode: bool = False,
    ) -> str:
        return self._execute_pre_detection_arm_movej_fixed(
            goal_handle,
            dry_run,
            box_layer,
            model_label,
            arm="right",
            tf_mode=tf_mode,
            drag_mode=drag_mode,
        )

    def _execute_post_arm_movej(
        self, goal_handle, dry_run: bool, right_arm_only: bool = False
    ) -> str:
        if not self._boolean("box_post_arm_movej_enabled"):
            return "post_arm_movej=disabled"
        left_units, right_units, left_target, right_target = (
            self._post_arm_movej_targets()
        )
        if right_arm_only:
            return self._execute_dual_arm_movej_targets(
                goal_handle,
                dry_run,
                "box_post_arm_movej",
                left_units,
                right_units,
                left_target,
                right_target,
                "post-grasp right arm MoveJ",
                "POST_ARM_MOVEJ",
                "MOVING_POST_GRASP_ARM_JOINTS",
                "WAITING_FOR_POST_GRASP_ARM_JOINTS",
                "POST_GRASP_ARM_JOINTS_REACHED",
                "post-grasp right arm MoveJ",
                active_arms=("right",),
                velocity_parameter="box_post_arm_movej_velocity",
            )
        detail = (
            "post-grasp dual arm MoveJ: "
            f"left_device={self._integer('box_post_arm_movej_left_device')}, "
            f"right_device={self._integer('box_post_arm_movej_right_device')}, "
            f"left_joint_units={left_units}, right_joint_units={right_units}"
        )
        self._publish_box_grasp_feedback(goal_handle, "POST_ARM_MOVEJ_TARGETS", detail)
        if dry_run:
            return f"{detail}; skipped in dry-run"

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        common = {
            "command": "movej",
            "v": self._integer("box_post_arm_movej_velocity"),
            "r": self._integer("box_post_arm_movej_blend_radius"),
            "trajectory_connect": self._integer(
                "box_post_arm_movej_trajectory_connect"
            ),
        }
        left_payload = {
            "device": self._integer("box_post_arm_movej_left_device"),
            "payload": {
                **common,
                "joint": left_units,
            },
        }
        right_payload = {
            "device": self._integer("box_post_arm_movej_right_device"),
            "payload": {
                **common,
                "joint": right_units,
            },
        }
        left_request = StringCmd.Request()
        right_request = StringCmd.Request()
        left_request.data = json.dumps(left_payload, separators=(",", ":")) + "\r\n"
        right_request.data = json.dumps(right_payload, separators=(",", ":")) + "\r\n"
        with self.joint_state_lock:
            sequence_before = dict(self.latest_slave_arm_state_sequences)
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_POST_GRASP_ARM_JOINTS",
            "sending left and right arm MoveJ commands concurrently",
        )
        left_future = self.body_command_client.call_async(left_request)
        right_future = self.body_command_client.call_async(right_request)
        left_response = self._wait_future(
            left_future,
            goal_handle,
            "calling post-grasp left arm MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        right_response = self._wait_future(
            right_future,
            goal_handle,
            "calling post-grasp right arm MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        self._parse_string_command_response(left_response, "post-grasp left arm MoveJ")
        self._parse_string_command_response(
            right_response, "post-grasp right arm MoveJ"
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "WAITING_FOR_POST_GRASP_ARM_JOINTS",
            "arm MoveJ commands accepted; waiting for fresh position and zero-velocity feedback",
        )
        self._wait_for_post_arm_joint_targets(
            goal_handle, left_target, right_target, sequence_before
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "POST_GRASP_ARM_JOINTS_REACHED",
            "both arm MoveJ targets reached with stable feedback",
        )
        return f"{detail}; arm_feedback=confirmed"

    def _execute_body_home(self, goal_handle, dry_run: bool) -> str:
        if not self._boolean("box_body_return_home_enabled"):
            return "body_home=disabled"
        units = [int(value) for value in self._float_array("box_body_home_joint_units")]
        target_angles = [
            math.radians(
                float(value)
                / self._float_array("box_body_command_units_per_degree")[index]
            )
            for index, value in enumerate(units)
        ]
        detail = f"body home MoveJ joint_units={units}"
        self._publish_box_grasp_feedback(goal_handle, "BODY_HOME_TARGETS", detail)
        if dry_run:
            return f"{detail}; skipped in dry-run"
        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        with self.joint_state_lock:
            sequence_before = self.latest_body_state_sequence
        payload = {
            "device": self._integer("box_joint1_device"),
            "payload": {
                "command": "movej",
                "device": self._integer("box_joint1_device"),
                "joint": units,
                "v": self._integer("box_body_home_velocity"),
                "r": self._integer("box_body_home_blend_radius"),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            "RETURNING_BODY_HOME",
            "sending body Joint1-4 home command and waiting for fresh feedback",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            "calling body home MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        self._parse_string_command_response(response, "body home MoveJ")
        self._publish_box_grasp_feedback(
            goal_handle,
            "WAITING_FOR_BODY_HOME",
            "waiting for all four body joints to reach zero",
        )
        self._wait_for_body_joints_target(
            goal_handle,
            target_angles,
            sequence_after=sequence_before,
            timeout_parameter="box_body_home_timeout_sec",
        )
        self._publish_box_grasp_feedback(
            goal_handle, "BODY_HOME_REACHED", "all four body joints reached home"
        )
        return f"{detail}; body_feedback=confirmed"

    def _box_layer_joint123_approach_angles_deg(
        self,
        box_layer: int,
        model_label: str | None = None,
        *,
        tf_mode: bool = False,
        drag_mode: bool = False,
    ) -> tuple[float, float, float]:
        if box_layer < 1 or box_layer > 4:
            raise MissionError("box_layer must be in [1, 4]")
        configured = self._boolean_array("box_layer_joint123_configured")
        if len(configured) != 4:
            raise MissionError("box_layer_joint123_configured must contain four values")
        if not configured[box_layer - 1]:
            raise MissionError(
                f"box_layer {box_layer} joint123 target is not configured yet"
            )
        angles = []
        normalized_model = (
            str(model_label).strip().lower() if model_label is not None else ""
        )
        if normalized_model not in ("", "bigbox", "smallbox"):
            raise MissionError("model_label must be 'bigbox', 'smallbox', or empty")
        if tf_mode:
            action_prefix = "drag_box_tf" if drag_mode else "grasp_box_tf"
            return tuple(
                self._float(
                    self._tf_layer_parameter_name(
                        action_prefix,
                        f"box_layer_joint{index}_approach_angle_deg",
                        normalized_model,
                        box_layer,
                    )
                )
                for index in range(1, 4)
            )
        for index in range(1, 4):
            name = f"box_layer_joint{index}_approach_angles_deg"
            if normalized_model:
                name += f"_{normalized_model}"
            values = self._float_array(name)
            if len(values) != 4:
                raise MissionError(f"{name} must contain four values")
            angle_deg = float(values[box_layer - 1])
            if not math.isfinite(angle_deg):
                raise MissionError(f"{name}[{box_layer - 1}] is invalid")
            angles.append(angle_deg)
        return tuple(angles)

    def _box_layer_joint1_approach_angle_deg(
        self,
        box_layer: int,
        model_label: str | None = None,
        *,
        tf_mode: bool = False,
        drag_mode: bool = False,
    ) -> float:
        return self._box_layer_joint123_approach_angles_deg(
            box_layer,
            model_label,
            tf_mode=tf_mode,
            drag_mode=drag_mode,
        )[0]

    def _move_body_joints_after_detection(
        self,
        goal_handle,
        box_layer: int,
        model_label: str | None = None,
        *,
        tf_mode: bool = False,
        drag_mode: bool = False,
    ):
        """Move J1/J2/J3 and preserve measured J4/J5 values."""
        approach_angles = [
            math.radians(angle_deg)
            for angle_deg in self._box_layer_joint123_approach_angles_deg(
                box_layer,
                model_label,
                tf_mode=tf_mode,
                drag_mode=drag_mode,
            )
        ]
        initial, _, sequence_before = self._wait_for_fresh_body_feedback(goal_handle)
        units_per_degree = self._float_array("box_body_command_units_per_degree")
        command_angles = approach_angles + initial[3:4]
        command_units = [
            int(round(math.degrees(angle) * units_per_degree[index]))
            for index, angle in enumerate(command_angles)
        ]
        device = self._integer("box_joint1_device")
        payload = {
            "device": device,
            "payload": {
                "command": "movej",
                "device": device,
                "joint": command_units,
                "v": self._integer("box_body_movej_velocity"),
                "r": self._integer("box_joint1_blend_radius"),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_BODY_JOINT123",
            "moving body joint1/joint2/joint3 while preserving joint4",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            f"calling body joints service {self._string('box_joint1_command_service_name')}",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        try:
            response_data = json.loads(str(response.data).strip())
        except (TypeError, ValueError) as exc:
            raise MissionError(
                f"invalid body joints service response: {response.data!r}"
            ) from exc
        if not response_data.get("receive_state", False):
            raise MissionError(f"body joints command was not accepted: {response.data}")
        final = self._wait_for_body_joints_target(
            goal_handle, approach_angles, sequence_after=sequence_before
        )
        return initial, final

    def _wait_for_body_joint1_target(
        self,
        goal_handle,
        target_rad: float,
        *,
        sequence_after: int = -1,
    ) -> float:
        timeout_sec = self._float("box_joint1_wait_timeout_sec")
        position_tolerance = self._float("box_joint1_position_tolerance_rad")
        velocity_tolerance = self._float("box_joint1_velocity_tolerance_rad_sec")
        max_age_sec = self._float("box_joint1_feedback_max_age_sec")
        required_stable = self._integer("box_joint1_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequence = -1
        latest_position = None
        latest_velocity = None
        latest_age = math.inf

        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for body joint1")
            position, velocity, state_time, sequence = self._joint1_feedback_snapshot()
            latest_position = position
            latest_velocity = velocity
            latest_age = time.monotonic() - state_time
            if sequence != last_sequence:
                last_sequence = sequence
                sample_matches = (
                    sequence > sequence_after
                    and position is not None
                    and velocity is not None
                    and latest_age <= max_age_sec
                    and abs(position - target_rad) <= position_tolerance
                    and abs(velocity) <= velocity_tolerance
                )
                stable_samples = stable_samples + 1 if sample_matches else 0
                if stable_samples >= required_stable:
                    return float(position)
            time.sleep(0.02)

        raise MissionError(
            "body joint1 did not reach the requested target: "
            f"target={target_rad:.6f} rad, measured={latest_position}, "
            f"velocity={latest_velocity}, feedback_age_sec={latest_age:.3f}, "
            f"timeout_sec={timeout_sec:.1f}"
        )

    def _move_body_joint1_after_detection(
        self, goal_handle, approach_angle_deg: float | None = None
    ) -> float:
        """Move Joint1 once and return its measured feedback-angle delta."""
        approach_deg = (
            self._float("box_joint1_approach_angle_deg")
            if approach_angle_deg is None
            else float(approach_angle_deg)
        )
        approach_rad = math.radians(approach_deg)
        initial, _, sequence_before = self._wait_for_fresh_body_feedback(goal_handle)
        initial_position = initial[0]

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        joint_units = int(
            round(approach_deg * self._float("box_joint1_command_units_per_degree"))
        )
        payload = {
            "device": self._integer("box_joint1_device"),
            "payload": {
                "command": "movej",
                "device": self._integer("box_joint1_device"),
                "joint": [joint_units, 0, 0, 0, 0],
                "v": self._integer("box_body_movej_velocity"),
                "r": self._integer("box_joint1_blend_radius"),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_BODY_JOINT1",
            f"moving body joint1 from {math.degrees(initial_position):.3f} "
            f"deg to {approach_deg:.3f} deg",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            f"calling body joint1 service {service_name}",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        try:
            response_data = json.loads(str(response.data).strip())
        except (TypeError, ValueError) as exc:
            raise MissionError(
                f"invalid body joint1 service response: {response.data!r}"
            ) from exc
        if not response_data.get("receive_state", False):
            raise MissionError(f"body joint1 command was not accepted: {response.data}")

        self._publish_box_grasp_feedback(
            goal_handle,
            "WAITING_FOR_BODY_JOINT1",
            "body joint1 command accepted; waiting for fresh /mcap/body "
            "position and zero-velocity feedback",
        )
        final_position = self._wait_for_body_joint1_target(
            goal_handle,
            approach_rad,
            sequence_after=sequence_before,
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "BODY_JOINT1_REACHED",
            f"body joint1 reached {math.degrees(final_position):.3f} deg",
        )
        return final_position - initial_position

    def _apply_joint1_execution_mode(
        self,
        goal_handle,
        left_target: Pose,
        right_target: Pose,
        dry_run: bool,
        box_layer: int,
        model_label: str | None = None,
    ) -> tuple[Pose, Pose, str]:
        execution_mode = self._string("box_grasp_execution_mode").lower()
        if execution_mode == "arms_only":
            return left_target, right_target, "execution=arms_only"

        if execution_mode == "joint123_then_arms":
            detection_angles = [
                math.radians(self._float(f"box_joint{index}_detection_angle_deg"))
                for index in range(1, 4)
            ]
            approach_angles = [
                math.radians(angle_deg)
                for angle_deg in self._box_layer_joint123_approach_angles_deg(
                    box_layer, model_label
                )
            ]
            if dry_run:
                movement_detail = "joint1/2/3 motion planned but skipped in dry-run"
                final_angles = approach_angles
                detection_angles_used = detection_angles
            else:
                initial_feedback, final_feedback = (
                    self._move_body_joints_after_detection(
                        goal_handle, box_layer, model_label
                    )
                )
                movement_detail = (
                    "joint1/2/3 motion completed from measured /mcap/body feedback"
                )
                # Use the fresh pre-motion feedback as the source frame. The
                # configured detection angles are only nominal and may differ
                # from the actual body pose when a goal starts.
                detection_angles_used = initial_feedback[:3]
                final_angles = final_feedback[:3]
            left_moved = self._reexpress_link8_target_after_joint123_motion(
                left_target, "left", detection_angles_used, final_angles
            )
            right_moved = self._reexpress_link8_target_after_joint123_motion(
                right_target, "right", detection_angles_used, final_angles
            )
            delta_detail = ",".join(
                f"{math.degrees(value):.3f}" for value in final_angles
            )
            return (
                left_moved,
                right_moved,
                "execution=joint123_then_arms; "
                "position_and_orientation=joint123_compensated; "
                f"approach_angles_deg=[{delta_detail}]; {movement_detail}",
            )

        configured_delta = math.radians(
            self._box_layer_joint1_approach_angle_deg(box_layer, model_label)
            - self._float("box_joint1_detection_angle_deg")
        )
        if dry_run:
            feedback_delta = configured_delta
            movement_detail = "joint1 motion planned but skipped in dry-run"
        else:
            feedback_delta = self._move_body_joint1_after_detection(
                goal_handle,
                self._box_layer_joint1_approach_angle_deg(box_layer, model_label),
            )
            movement_detail = (
                "joint1 motion completed from measured /mcap/body feedback"
            )
        if execution_mode == "joint1_then_arms_keep_position":
            left_moved = self._rotate_link8_orientation_after_joint1_rotation(
                left_target, "left", feedback_delta
            )
            right_moved = self._rotate_link8_orientation_after_joint1_rotation(
                right_target, "right", feedback_delta
            )
            target_detail = "position=unchanged, orientation=joint1_compensated"
        else:
            left_moved = self._reexpress_link8_target_after_joint1_rotation(
                left_target, "left", feedback_delta
            )
            right_moved = self._reexpress_link8_target_after_joint1_rotation(
                right_target, "right", feedback_delta
            )
            target_detail = "position_and_orientation=joint1_compensated"
        return (
            left_moved,
            right_moved,
            f"execution={execution_mode}; {target_detail}; {movement_detail}; "
            f"feedback_delta_deg={math.degrees(feedback_delta):.3f}",
        )
