import time

from mission_interfaces.action import MoveChassis

from .common import (
    ANGULAR_CHASSIS_DIRECTIONS,
    MissionCanceled,
    MissionError,
)


class ChassisActionMixin:
    """Timed chassis movement orchestration."""

    @staticmethod
    def _publish_move_chassis_feedback(
        goal_handle, stage: str, detail: str, progress: float
    ) -> None:
        feedback = MoveChassis.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.progress = max(0.0, min(1.0, progress))
        goal_handle.publish_feedback(feedback)

    def _execute_move_chassis(self, goal_handle) -> MoveChassis.Result:
        started_at = time.monotonic()
        request = goal_handle.request
        direction = request.direction.strip().lower()
        speed = self._float(
            "chassis_angular_speed"
            if direction in ANGULAR_CHASSIS_DIRECTIONS
            else "chassis_linear_speed"
        )
        duration_sec = self._float("chassis_move_duration_sec")
        result = MoveChassis.Result()
        result.direction = direction
        result.speed = speed
        result.duration_sec = duration_sec

        try:
            self._publish_move_chassis_feedback(
                goal_handle,
                "STARTING",
                f"starting {direction} chassis motion at "
                f"{speed:.3f} for {duration_sec:.3f}s",
                0.0,
            )
            self._move_chassis_for_duration(
                goal_handle,
                direction,
                speed,
                duration_sec,
            )

            result.success = True
            result.message = "chassis motion completed"
            self._finalize_action_result(result, started_at, "move_chassis")
            self._publish_move_chassis_feedback(
                goal_handle, "DONE", result.message, 1.0
            )
            goal_handle.succeed()
            return result
        except MissionCanceled as exc:
            result.success = False
            result.message = str(exc)
            goal_handle.canceled()
            return result
        except MissionError as exc:
            result.success = False
            result.message = str(exc)
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.message = f"unexpected chassis motion error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            self._publish_zero_chassis()
            self._release_goal()
            self._finalize_action_result(result, started_at, "move_chassis")
