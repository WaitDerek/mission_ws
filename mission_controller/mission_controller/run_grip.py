"""Legacy-style one-shot grip sequence."""

from __future__ import annotations

from dataclasses import dataclass
import time

from mission_interfaces.action import ExecuteGrip

from .common import MissionCanceled, MissionError
from .manipulation_math import (
    grip_base_targets,
    local_y_approach,
    matrix_to_pose_array,
    pose_to_matrix,
)


@dataclass(frozen=True)
class GripRunResult:
    """Output of one legacy-style grip sequence."""

    contact_detected: bool
    force_delta: float
    message: str = "grip pipeline completed"


class RunGripMixin:
    """Run the original left-suction grip sequence exactly once."""

    def _execute_run_grip(self, goal_handle) -> ExecuteGrip.Result:
        """Expose the legacy-compatible grip sequence as a ROS Action."""
        result = ExecuteGrip.Result()
        started = time.monotonic()
        try:
            target_type = (
                str(goal_handle.request.target_type).strip().lower() or "badge"
            )
            outcome = self.run_grip(goal_handle, target_type)
            result.success = True
            result.contact_detected = outcome.contact_detected
            result.force_delta = outcome.force_delta
            result.message = outcome.message
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
            result.message = f"legacy grip pipeline failed: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.message = f"unexpected legacy grip pipeline error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            if result.message:
                result.message += (
                    f" (elapsed_sec={time.monotonic() - started:.3f})"
                )
            with self._state_lock:
                self._active = False
                self._active_child_handles.clear()

    def run_grip(self, goal_handle, target_type: str) -> GripRunResult:
        self._require_hardware_messages()
        normalized_target = str(target_type).strip().lower() or "badge"
        if normalized_target not in ("badge", "connector"):
            raise MissionError(
                "target_type must be badge or connector, "
                f"got {normalized_target!r}"
            )

        config = dict(self._grip_config)
        if normalized_target == "connector":
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
            "waiting for left EE pose",
        )
        left_pose, _ = self._wait_for_run_poses(
            goal_handle,
            need_left=True,
            need_right=False,
            require_fresh=False,
        )
        self._pipeline_feedback(
            goal_handle,
            ExecuteGrip,
            f"DETECTING_{normalized_target.upper()}",
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
        completed = False
        try:
            self._set_suction(goal_handle, "left", True)
            left_pose, _ = self._wait_for_run_poses(
                goal_handle,
                need_left=True,
                need_right=False,
                require_fresh=False,
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
            outcome = GripRunResult(contact_detected, force_delta)
            self._pipeline_feedback(
                goal_handle, ExecuteGrip, "DONE", outcome.message
            )
            completed = True
            return outcome
        finally:
            if not completed:
                self._set_suction_best_effort("left", False)
