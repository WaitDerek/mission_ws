import time

from mission_interfaces.action import ExecuteBoxStack

from .common import MissionCanceled, MissionError


class StackActionMixin:
    """Fixed-arm box stacking orchestration."""

    @staticmethod
    def _publish_box_stack_feedback(
        goal_handle, stage: str, detail: str, level: int
    ) -> None:
        feedback = ExecuteBoxStack.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.level = level
        goal_handle.publish_feedback(feedback)

    def _execute_box_stack(self, goal_handle) -> ExecuteBoxStack.Result:
        started_at = time.monotonic()
        level = int(goal_handle.request.level)
        result = ExecuteBoxStack.Result()
        result.level = level
        target_torso = self._stack_level_torso_target(level)
        pickup_torso = self._float_array("stack_pickup_torso_positions")
        result.target_torso_positions = target_torso

        try:
            self._publish_box_stack_feedback(
                goal_handle,
                "INITIALIZING",
                "checking the fixed highest box-loading arm and torso posture",
                level,
            )
            ready, readiness_detail = self._stack_default_ready()
            if not ready:
                self._publish_box_stack_feedback(
                    goal_handle,
                    "PREPARING_START_POSE",
                    "current posture is not the fixed highest loading posture; "
                    f"moving both arms and torso into place ({readiness_detail})",
                    level,
                )
                result.arm_message = self._prepare_stack_start_pose(goal_handle)

            self._publish_box_stack_feedback(
                goal_handle,
                "CLOSING_GRIPPERS",
                "closing both grippers around the manually loaded box",
                level,
            )
            self._publish_both_grippers(
                goal_handle, self._float("gripper_closed_position")
            )
            result.gripper_closed = True
            self._wait_delay(
                goal_handle,
                self._float("gripper_settle_sec"),
                "while waiting for the manually loaded box to be gripped",
            )

            self._publish_box_stack_feedback(
                goal_handle,
                "LOWERING_TORSO",
                f"holding both arms fixed while lowering the torso to level {level}",
                level,
            )
            self._move_stack_torso(
                goal_handle,
                target_torso,
                f"lowering the held box to stack level {level}",
            )
            fixed_message = "both arms remained at the fixed loading posture"
            result.arm_message = (
                f"start={result.arm_message}; {fixed_message}"
                if result.arm_message
                else fixed_message
            )

            self._publish_box_stack_feedback(
                goal_handle,
                "WAITING_TO_RELEASE",
                "torso target confirmed; holding the box for "
                f"{self._float('stack_release_delay_sec'):.1f}s before release",
                level,
            )
            self._wait_delay(
                goal_handle,
                self._float("stack_release_delay_sec"),
                "while holding the confirmed stack posture before release",
            )

            self._publish_box_stack_feedback(
                goal_handle,
                "OPENING_GRIPPERS",
                "opening both grippers after the place trajectory completed",
                level,
            )
            self._publish_both_grippers(
                goal_handle, self._float("gripper_open_position")
            )
            result.gripper_opened = True
            self._wait_delay(
                goal_handle,
                self._float("gripper_settle_sec"),
                "while waiting for the stacked box to be released",
            )

            self._publish_box_stack_feedback(
                goal_handle,
                "RETREATING",
                "raising the torso back to the highest loading posture so both "
                "grippers clear the released box",
                level,
            )
            self._move_stack_torso(
                goal_handle,
                pickup_torso,
                "raising both fixed arms clear of the released box",
            )
            result.retreat_completed = True

            self._publish_box_stack_feedback(
                goal_handle,
                "RETURNING_READY",
                "highest manual-load posture restored; arms stayed fixed",
                level,
            )
            result.ready_completed = True

            result.success = True
            result.message = (
                f"box stack level {level} completed by closed-loop torso motion"
            )
            self._finalize_action_result(result, started_at, "execute_box_stack")
            self._publish_box_stack_feedback(
                goal_handle, "DONE", result.message, level
            )
            goal_handle.succeed()
            return result
        except MissionCanceled as exc:
            result.success = False
            result.message = str(exc)
            self._publish_box_stack_feedback(
                goal_handle, "CANCELLED", result.message, level
            )
            goal_handle.canceled()
            return result
        except MissionError as exc:
            result.success = False
            result.message = str(exc)
            self.get_logger().error(result.message)
            self._publish_box_stack_feedback(
                goal_handle, "FAILED", result.message, level
            )
            goal_handle.abort()
            return result
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.message = f"unexpected box stack mission error: {exc}"
            self.get_logger().error(result.message)
            self._publish_box_stack_feedback(
                goal_handle, "FAILED", result.message, level
            )
            goal_handle.abort()
            return result
        finally:
            self._release_goal()
            self._finalize_action_result(
                result, started_at, "execute_box_stack"
            )
