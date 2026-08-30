"""Independent ROS Action implementation for robust film peeling."""

from __future__ import annotations

import time

from mission_interfaces.action import ExecutePeel

from .common import MissionCanceled, MissionError
from .manipulation_math import (
    local_y_approach,
    matrix_to_pose_array,
    peel_base_above_target,
    peel_withdraw_targets,
    pose_to_matrix,
)


class ExecutePeelMixin:
    """Execute film peeling as a checked, cancelable ROS Action."""

    def _execute_peel(self, goal_handle) -> ExecutePeel.Result:
        result = ExecutePeel.Result()
        started = time.monotonic()
        suction_may_be_enabled = False
        try:
            self._require_hardware_messages()
            config = self._peel_config
            self._prepare_pipeline(goal_handle, ExecutePeel, config)
            self._pipeline_feedback(
                goal_handle,
                ExecutePeel,
                "WAITING_FOR_POSES",
                "waiting for fresh left/right EE poses",
            )
            _, right_pose = self._wait_for_pipeline_poses(
                goal_handle,
                need_left=True,
                need_right=True,
                require_fresh=True,
            )
            self._pipeline_feedback(
                goal_handle,
                ExecutePeel,
                "DETECTING_BADGE_BACK",
                f"requesting model {config['model_label']}",
            )
            object_pose = self._estimate_pipeline_object(
                goal_handle, config["model_label"]
            )
            above_target = peel_base_above_target(
                pose_to_matrix(right_pose.pose),
                pose_to_matrix(object_pose.pose),
                config,
            )
            self._pipeline_feedback(
                goal_handle,
                ExecutePeel,
                "MOVING_ABOVE",
                "moving right arm above badge back",
            )
            self._pipeline_move_pose(
                goal_handle,
                left=[],
                right=matrix_to_pose_array(above_target),
            )
            self._pipeline_feedback(
                goal_handle,
                ExecutePeel,
                "STARTING_SUCTION",
                "turning on right suction",
            )
            suction_may_be_enabled = True
            self._set_suction(goal_handle, "right", True)
            baseline_sample, force_sequence = self._wait_for_next_force(goal_handle)
            _, right_pose = self._wait_for_pipeline_poses(
                goal_handle,
                need_left=True,
                need_right=True,
                require_fresh=True,
            )
            approach_target = local_y_approach(
                pose_to_matrix(right_pose.pose), config["max_approach_dist"]
            )
            self._pipeline_feedback(
                goal_handle,
                ExecutePeel,
                "APPROACHING",
                "moving right arm along local +Y with force monitoring",
            )
            contact_detected, force_delta = self._pipeline_linear_contact(
                goal_handle,
                left=[],
                right=matrix_to_pose_array(approach_target),
                threshold=float(config["suction_force_threshold"]),
                baseline_sample=baseline_sample,
                force_sequence=force_sequence,
            )
            if not contact_detected:
                raise MissionError(
                    "peel approach completed without detecting force contact"
                )
            left_pose, right_pose = self._wait_for_pipeline_poses(
                goal_handle,
                need_left=True,
                need_right=True,
                require_fresh=True,
            )
            left_target, right_target = peel_withdraw_targets(
                pose_to_matrix(left_pose.pose),
                pose_to_matrix(right_pose.pose),
                config,
            )
            self._pipeline_feedback(
                goal_handle,
                ExecutePeel,
                "PEELING",
                "executing dual-arm peel trajectory",
            )
            self._pipeline_move_pose(
                goal_handle,
                left=matrix_to_pose_array(left_target),
                right=matrix_to_pose_array(right_target),
            )
            self._pipeline_feedback(
                goal_handle,
                ExecutePeel,
                "DETACHING",
                "turning off right suction",
            )
            self._set_suction(goal_handle, "right", False)
            suction_may_be_enabled = False
            result.success = True
            result.contact_detected = True
            result.force_delta = force_delta
            result.message = "peel pipeline completed"
            self._pipeline_feedback(goal_handle, ExecutePeel, "DONE", result.message)
            goal_handle.succeed()
            return result
        except MissionCanceled as exc:
            self._cancel_children()
            result.success = False
            result.message = str(exc)
            goal_handle.canceled()
            return result
        except (MissionError, ValueError, KeyError, TypeError) as exc:
            result.success = False
            result.message = f"peel pipeline failed: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.message = f"unexpected peel pipeline error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            if suction_may_be_enabled:
                self._set_suction_best_effort("right", False)
            if result.message:
                result.message += f" (elapsed_sec={time.monotonic() - started:.3f})"
            with self._state_lock:
                self._active = False
                self._active_child_handles.clear()
