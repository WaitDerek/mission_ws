"""Independent ROS Action implementation for robust suction grip."""

from __future__ import annotations

import time

from mission_interfaces.action import ExecuteGrip

from .common import MissionCanceled, MissionError
from .manipulation_math import (
    grip_base_targets,
    local_y_approach,
    matrix_to_pose_array,
    pose_to_matrix,
)


class ExecuteGripMixin:
    """Execute a grip as a checked, cancelable ROS Action."""

    def _execute_grip(self, goal_handle) -> ExecuteGrip.Result:
        result = ExecuteGrip.Result()
        started = time.monotonic()
        suction_may_be_enabled = False
        completed = False
        try:
            self._require_hardware_messages()
            target_type = (
                str(goal_handle.request.target_type).strip().lower() or "badge"
            )
            if target_type not in ("badge", "connector"):
                raise MissionError(
                    f"target_type must be badge or connector, got {target_type!r}"
                )
            config = dict(self._grip_config)
            if target_type == "connector":
                connector_config = self._connector_grip_config
                if not bool(connector_config.get("calibrated", False)):
                    raise MissionError(
                        "connector grip calibration is disabled; validate the "
                        "connector prepare trajectory and object target offsets "
                        "before enabling it"
                    )
                overrides = connector_config.get("overrides", {})
                if not isinstance(overrides, dict):
                    raise MissionError("connector grip overrides must be an object")
                config.update(overrides)
                config["model_label"] = str(connector_config["model_label"])

            self._prepare_pipeline(goal_handle, ExecuteGrip, config)
            self._pipeline_feedback(
                goal_handle,
                ExecuteGrip,
                "WAITING_FOR_POSE",
                "waiting for a fresh left EE pose",
            )
            left_pose, _ = self._wait_for_pipeline_poses(
                goal_handle,
                need_left=True,
                need_right=False,
                require_fresh=True,
            )
            self._pipeline_feedback(
                goal_handle,
                ExecuteGrip,
                f"DETECTING_{target_type.upper()}",
                f"requesting model {config['model_label']}",
            )
            object_pose = self._estimate_pipeline_object(
                goal_handle, config["model_label"]
            )
            down_target, up_target = grip_base_targets(
                pose_to_matrix(left_pose.pose),
                pose_to_matrix(object_pose.pose),
                config,
            )
            self._pipeline_feedback(
                goal_handle,
                ExecuteGrip,
                "MOVING_TO_GRIP",
                "moving left arm to down target",
            )
            self._pipeline_move_pose(
                goal_handle,
                left=matrix_to_pose_array(down_target),
                right=[],
            )
            self._pipeline_feedback(
                goal_handle,
                ExecuteGrip,
                "STARTING_SUCTION",
                "turning on left suction",
            )
            suction_may_be_enabled = True
            self._set_suction(goal_handle, "left", True)
            left_pose, _ = self._wait_for_pipeline_poses(
                goal_handle,
                need_left=True,
                need_right=False,
                require_fresh=True,
            )
            approach_target = local_y_approach(
                pose_to_matrix(left_pose.pose), config["max_approach_distance"]
            )
            self._pipeline_feedback(
                goal_handle,
                ExecuteGrip,
                "APPROACHING",
                "moving left arm along local +Y with force monitoring",
            )
            contact_detected, force_delta = self._pipeline_linear_contact(
                goal_handle,
                left=matrix_to_pose_array(approach_target),
                right=[],
                threshold=float(config["suction_force_threshold"]),
            )
            if not contact_detected:
                raise MissionError(
                    "grip approach completed without detecting force contact"
                )
            self._pipeline_feedback(
                goal_handle,
                ExecuteGrip,
                "WITHDRAWING",
                "withdrawing to up target",
            )
            self._pipeline_move_pose(
                goal_handle,
                left=matrix_to_pose_array(up_target),
                right=[],
                linear=True,
            )
            result.success = True
            result.contact_detected = True
            result.force_delta = force_delta
            result.message = "grip pipeline completed"
            self._pipeline_feedback(goal_handle, ExecuteGrip, "DONE", result.message)
            completed = True
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
            result.message = f"grip pipeline failed: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.message = f"unexpected grip pipeline error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            if suction_may_be_enabled and not completed:
                self._set_suction_best_effort("left", False)
            if result.message:
                result.message += f" (elapsed_sec={time.monotonic() - started:.3f})"
            with self._state_lock:
                self._active = False
                self._active_child_handles.clear()
