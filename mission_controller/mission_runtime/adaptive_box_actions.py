"""Direct-SDK adaptive box Action orchestration."""

from __future__ import annotations

import time
import uuid

from mission_interfaces.action import ExecuteAdaptiveBoxGrasp

from .common import MissionCanceled, MissionError


class AdaptiveBoxActionsMixin:
    """Freeze targets, execute native MoveL, and lift without dual_arm."""

    @staticmethod
    def _publish_adaptive_feedback(
        goal_handle, stage: str, progress: float, detail: str
    ) -> None:
        feedback = ExecuteAdaptiveBoxGrasp.Feedback()
        feedback.stage = stage
        feedback.progress = float(max(0.0, min(1.0, progress)))
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _adaptive_error_code(stage: str) -> int:
        result_type = ExecuteAdaptiveBoxGrasp.Result
        if stage.startswith("DETECTING"):
            return result_type.ERROR_DETECTION
        if stage == "FREEZING_TARGET_IN_BASE":
            return result_type.ERROR_TF_FREEZE
        if stage in {"COMPUTING_GRASP_POSES", "PREPARING_SDK_TARGETS"}:
            return result_type.ERROR_GRASP_TARGET
        if stage == "MOVING_TO_GRASP_WITH_MOVEL":
            return result_type.ERROR_GRASP_MOVEL
        if stage == "LIFTING_WITH_MOVEL":
            return result_type.ERROR_LIFT_MOVEL
        return result_type.ERROR_INTERNAL

    def _adaptive_safe_recovery(self, goal_handle, reason: str) -> None:
        self._publish_adaptive_feedback(
            goal_handle,
            "CANCELING_ACTIVE_MOTION",
            0.99,
            f"requesting native SDK slow-stop: {reason}",
        )
        self._cancel_callback(goal_handle)
        self._publish_adaptive_feedback(
            goal_handle,
            "SAFE_RECOVERY",
            0.99,
            "native SDK motions canceled; holding the current state without "
            "a torso, ready, or gripper command",
        )

    def _execute_adaptive_box_grasp(
        self, goal_handle
    ) -> ExecuteAdaptiveBoxGrasp.Result:
        started_at = time.monotonic()
        request = goal_handle.request
        result = ExecuteAdaptiveBoxGrasp.Result()
        result.task_id = request.task_id.strip() or str(uuid.uuid4())
        result.error_code = ExecuteAdaptiveBoxGrasp.Result.ERROR_NONE
        stage = "DETECTING_OBJECT"

        try:
            self._publish_adaptive_feedback(
                goal_handle,
                stage,
                0.01,
                f"requesting object detection for task_id={result.task_id}",
            )
            detection = self._call_adaptive_detection(goal_handle, request)
            result.detection_confidence = float(detection.detection_score)

            stage = "FREEZING_TARGET_IN_BASE"
            self._publish_adaptive_feedback(
                goal_handle,
                stage,
                0.20,
                "using the detection PoseStamped timestamp for the one-time "
                "camera-to-base_link TF conversion",
            )
            constrained_camera_pose = self._constrain_box_camera_pose(
                detection.pose
            )
            frozen_object_pose = self._freeze_detection_pose(
                constrained_camera_pose
            )
            object_pose_base = self._make_pickup_box_pose(frozen_object_pose)
            result.object_pose_base = object_pose_base
            self.box_object_pose_raw_publisher.publish(frozen_object_pose)
            self.box_object_pose_publisher.publish(object_pose_base)

            stage = "COMPUTING_GRASP_POSES"
            self._publish_adaptive_feedback(
                goal_handle,
                stage,
                0.30,
                "computing frozen left/right fixture-center targets from the "
                "box dimensions and object-frame grasp geometry",
            )
            left_grasp, right_grasp = self._compute_adaptive_grasp_poses(
                object_pose_base
            )
            result.left_grasp_pose_base = left_grasp
            result.right_grasp_pose_base = right_grasp

            stage = "PREPARING_SDK_TARGETS"
            self._publish_adaptive_feedback(
                goal_handle,
                stage,
                0.45,
                "converting the frozen base_link fixture-center targets to "
                "fixture-compensated Link8 targets in each arm base frame",
            )
            grasp_message = self._execute_adaptive_grasp_movel(
                goal_handle,
                left_grasp,
                right_grasp,
                True,
            )

            if request.dry_run:
                lift_message = self._execute_adaptive_movel_lift(
                    goal_handle,
                    left_grasp,
                    right_grasp,
                    True,
                )
                result.success = True
                result.message = (
                    "direct-SDK dry-run completed; no dual_arm planning and no "
                    f"physical command were used; {grasp_message}; {lift_message}"
                )
                self._publish_adaptive_feedback(
                    goal_handle, "DRY_RUN_COMPLETE", 1.0, result.message
                )
                self._finalize_action_result(
                    result, started_at, "execute_adaptive_box_grasp"
                )
                goal_handle.succeed()
                return result

            stage = "MOVING_TO_GRASP_WITH_MOVEL"
            self._publish_adaptive_feedback(
                goal_handle,
                stage,
                0.55,
                "executing both frozen grasp targets directly through native "
                "RealMan rm_movel without MoveIt planning or collision checking",
            )
            # PREPARING_SDK_TARGETS validated the conversion in dry-run mode;
            # perform the physical call only after publishing the motion stage.
            grasp_message = self._execute_adaptive_grasp_movel(
                goal_handle,
                left_grasp,
                right_grasp,
                False,
            )
            result.grasp_movel_executed = True

            stage = "LIFTING_WITH_MOVEL"
            self._publish_adaptive_feedback(
                goal_handle,
                stage,
                0.82,
                "executing a second native rm_movel to the frozen grasp targets "
                "offset along base_link +Z; no gripper command is included",
            )
            lift_message = self._execute_adaptive_movel_lift(
                goal_handle,
                left_grasp,
                right_grasp,
                False,
            )
            result.lift_executed = True

            result.success = True
            result.error_code = ExecuteAdaptiveBoxGrasp.Result.ERROR_NONE
            result.failure_stage = ""
            result.message = (
                "direct native-SDK grasp MoveL and lift MoveL completed; "
                f"{grasp_message}; {lift_message}"
            )
            self._publish_adaptive_feedback(
                goal_handle, "COMPLETED", 1.0, result.message
            )
            self._finalize_action_result(
                result, started_at, "execute_adaptive_box_grasp"
            )
            goal_handle.succeed()
            return result
        except MissionCanceled as exc:
            result.success = False
            result.error_code = ExecuteAdaptiveBoxGrasp.Result.ERROR_CANCELED
            result.failure_stage = stage
            result.message = str(exc)
            self._adaptive_safe_recovery(goal_handle, result.message)
            self._publish_adaptive_feedback(
                goal_handle, "CANCELED", 1.0, result.message
            )
            goal_handle.canceled()
            return result
        except MissionError as exc:
            result.success = False
            result.error_code = self._adaptive_error_code(stage)
            result.failure_stage = stage
            result.message = str(exc)
            self.get_logger().error(
                f"adaptive direct-SDK mission failed at {stage}: {result.message}"
            )
            self._adaptive_safe_recovery(goal_handle, result.message)
            self._publish_adaptive_feedback(
                goal_handle, "FAILED", 1.0, result.message
            )
            goal_handle.abort()
            return result
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.error_code = ExecuteAdaptiveBoxGrasp.Result.ERROR_INTERNAL
            result.failure_stage = stage
            result.message = f"unexpected adaptive direct-SDK error: {exc}"
            self.get_logger().error(result.message)
            self._adaptive_safe_recovery(goal_handle, result.message)
            self._publish_adaptive_feedback(
                goal_handle, "FAILED", 1.0, result.message
            )
            goal_handle.abort()
            return result
        finally:
            self._release_goal()
            self._finalize_action_result(
                result, started_at, "execute_adaptive_box_grasp"
            )
