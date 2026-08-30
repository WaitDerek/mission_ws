"""Front-bumper connector and badge assembly Action."""

from __future__ import annotations

import time

from geometry_msgs.msg import PoseStamped
from mission_interfaces.action import ExecuteAssembly

from .common import MissionCanceled, MissionError
from .manipulation_math import (
    assembly_targets,
    matrix_to_pose_array,
    pose_to_matrix,
)


class AssemblyActionMixin:
    """Detect a bumper task pose and insert the held part with force limiting."""

    def _execute_assembly(self, goal_handle) -> ExecuteAssembly.Result:
        result = ExecuteAssembly.Result()
        started = time.monotonic()
        try:
            self._require_hardware_messages()
            target_type = str(goal_handle.request.target_type).strip().lower()
            if target_type not in ("connector", "badge"):
                raise MissionError(
                    f"target_type must be connector or badge, got {target_type!r}"
                )
            config = self._assembly_config
            if not bool(config.get("calibrated", False)):
                raise MissionError(
                    "assembly calibration is disabled; set calibrated=true only "
                    "after validating front-bumper and task_T_tool transforms"
                )
            target_config = config["targets"][target_type]

            self._prepare_pipeline(goal_handle, ExecuteAssembly, config)
            self._pipeline_feedback(
                goal_handle,
                ExecuteAssembly,
                "WAITING_FOR_POSE",
                "waiting for a fresh left EE pose",
            )
            left_pose, _ = self._wait_for_pipeline_poses(
                goal_handle,
                need_left=True,
                need_right=False,
                require_fresh=True,
            )
            patch_name = str(target_config["patch_name"])
            self._pipeline_feedback(
                goal_handle,
                ExecuteAssembly,
                "DETECTING_FRONT_BUMPER",
                f"requesting front-bumper patch {patch_name}",
            )
            task_pose = self._estimate_front_bumper_task(goal_handle, patch_name)
            pre_target, final_target = assembly_targets(
                pose_to_matrix(left_pose.pose),
                pose_to_matrix(task_pose.pose),
                config,
                target_config,
            )

            target_message = PoseStamped()
            target_message.header.frame_id = left_pose.header.frame_id or self._string(
                "execution_frame"
            )
            target_message.header.stamp = self.get_clock().now().to_msg()
            final_values = matrix_to_pose_array(final_target)
            target_message.pose.position.x = final_values[0]
            target_message.pose.position.y = final_values[1]
            target_message.pose.position.z = final_values[2]
            target_message.pose.orientation.x = final_values[3]
            target_message.pose.orientation.y = final_values[4]
            target_message.pose.orientation.z = final_values[5]
            target_message.pose.orientation.w = final_values[6]
            result.target_pose = target_message

            self._pipeline_feedback(
                goal_handle,
                ExecuteAssembly,
                "MOVING_TO_PREASSEMBLY",
                "moving the held part to the calibrated pre-assembly target",
            )
            self._pipeline_move_pose(
                goal_handle,
                left=matrix_to_pose_array(pre_target),
                right=[],
            )
            baseline_sample, force_sequence = self._wait_for_next_force(goal_handle)
            self._pipeline_feedback(
                goal_handle,
                ExecuteAssembly,
                "FORCE_CONTROLLED_INSERTION",
                "inserting with fresh force feedback",
            )
            contact_detected, force_delta = self._pipeline_linear_contact(
                goal_handle,
                left=final_values,
                right=[],
                threshold=float(target_config["force_threshold"]),
                baseline_sample=baseline_sample,
                force_sequence=force_sequence,
            )
            if (
                bool(target_config.get("require_contact", True))
                and not contact_detected
            ):
                raise MissionError(
                    "assembly insertion reached its target without force contact"
                )
            self._pipeline_feedback(
                goal_handle,
                ExecuteAssembly,
                "RELEASING_PART",
                "turning off left suction",
            )
            self._set_suction(goal_handle, "left", False)

            result.success = True
            result.contact_detected = contact_detected
            result.force_delta = force_delta
            result.message = f"{target_type} assembly completed"
            self._pipeline_feedback(
                goal_handle, ExecuteAssembly, "DONE", result.message
            )
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
            result.message = f"assembly failed: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.message = f"unexpected assembly error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            if result.message:
                result.message += f" (elapsed_sec={time.monotonic() - started:.3f})"
            with self._state_lock:
                self._active = False
                self._active_child_handles.clear()
