from .common import MissionError


class StackSupportMixin:
    """Closed-loop torso helpers used by box stacking."""

    def _stack_level_torso_target(self, level: int) -> list[float]:
        values = self._float_array("stack_level_torso_positions")
        start = (level - 1) * 4
        return values[start : start + 4]

    def _move_stack_torso(
        self,
        goal_handle,
        targets: list[float],
        context: str,
    ) -> None:
        velocities, estimated_duration = self._stack_torso_motion_profile(targets)
        self.get_logger().info(
            "stack torso synchronized profile: "
            f"duration={estimated_duration:.3f}s, velocities="
            f"{[round(value, 6) for value in velocities]}"
        )
        self._publish_torso(goal_handle, targets, velocities=velocities)
        self._wait_for_torso_target(goal_handle, targets, context)

    def _stack_torso_motion_profile(
        self,
        targets: list[float],
    ) -> tuple[list[float], float]:
        with self.joint_state_lock:
            current = list(self.latest_torso_positions)
        if len(current) < 4:
            raise MissionError(
                "cannot calculate synchronized stack torso speeds without "
                "four measured torso positions"
            )

        deltas = [
            abs(target - actual)
            for actual, target in zip(current[:4], targets)
        ]
        torso1_speed = self._float("stack_torso1_speed")
        if deltas[0] <= 1e-6:
            if max(deltas, default=0.0) <= 1e-6:
                return [0.0, 0.0, 0.0, 0.0], 0.0
            raise MissionError(
                "cannot synchronize stack torso motion to Torso1 because "
                "Torso1 has no remaining motion while another torso joint does"
            )

        duration = deltas[0] / torso1_speed
        velocities = [delta / duration for delta in deltas]
        velocities[2] = (
            self._float("stack_torso3_speed")
            if deltas[2] > 1e-6
            else 0.0
        )
        return velocities, duration

    def _prepare_stack_start_pose(self, goal_handle) -> str:
        pickup_torso = self._float_array("stack_pickup_torso_positions")
        pickup_velocities, pickup_duration = self._stack_torso_motion_profile(
            pickup_torso
        )
        self.get_logger().info(
            "stack start torso synchronized profile: "
            f"duration={pickup_duration:.3f}s, velocities="
            f"{[round(value, 6) for value in pickup_velocities]}"
        )
        arm_message = self._call_arm_joints(
            goal_handle,
            self._float_array("stack_default_left_joint_positions"),
            self._float_array("stack_default_right_joint_positions"),
            False,
            goal_accepted_callback=lambda: self._publish_torso(
                goal_handle,
                pickup_torso,
                velocities=pickup_velocities,
            ),
            duration=self._float("stack_start_arm_duration_sec"),
        )
        self._wait_for_torso_target(
            goal_handle,
            pickup_torso,
            "preparing the highest box-loading torso posture",
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while settling at the fixed box-loading posture",
        )
        return arm_message
