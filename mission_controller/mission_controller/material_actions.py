import time
from typing import Optional

from mission_interfaces.action import ExecuteGrasp, ExecutePlace

from .common import MissionCanceled, MissionError


class MaterialActionsMixin:
    """Mission-level single-material grasp and placement orchestration."""

    @staticmethod
    def _publish_grasp_feedback(goal_handle, stage: str, detail: str, arm: str) -> None:
        feedback = ExecuteGrasp.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.arm = arm
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _publish_place_feedback(goal_handle, stage: str, detail: str, arm: str) -> None:
        feedback = ExecutePlace.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.arm = arm
        goal_handle.publish_feedback(feedback)

    def _execute_grasp(self, goal_handle) -> ExecuteGrasp.Result:
        started_at = time.monotonic()
        request = goal_handle.request
        arm = self._resolve_arm(request.arm)
        result = ExecuteGrasp.Result()
        result.arm = arm
        torso_prepared = False
        joint_preparation_started = False
        joint_preparation_complete = False
        motion_state = {"started": False}
        completed = False

        try:
            self._publish_grasp_feedback(
                goal_handle, "INITIALIZING", "preparing gripper, torso, and arms", arm
            )
            if request.dry_run:
                self._publish_grasp_feedback(
                    goal_handle,
                    "DRY_RUN_INITIALIZATION",
                    "skipping direct gripper, torso, and arm-joint commands",
                    arm,
                )
            else:
                if self._boolean("open_gripper_before_grasp"):
                    self._publish_grasp_feedback(
                        goal_handle,
                        "OPENING_INITIAL_GRIPPERS",
                        "opening both grippers before moving to the grasp "
                        "observation posture",
                        arm,
                    )
                    self._prepare_grasp_grippers(goal_handle)
                observation_ready, readiness_detail = (
                    self._grasp_observation_ready()
                )
                if observation_ready:
                    self._publish_grasp_feedback(
                        goal_handle,
                        "GRASP_OBSERVATION_READY",
                        "arms and torso are already at the grasp observation "
                        f"posture; skipping preparation ({readiness_detail})",
                        arm,
                    )
                else:
                    self._publish_grasp_feedback(
                        goal_handle,
                        "PREPARING",
                        "grasp observation posture is not ready; moving "
                        "through the intermediate arms, then entering the "
                        f"final observation posture ({readiness_detail})",
                        arm,
                    )
                    torso_prepared = True
                    joint_preparation_started = True
                    self._prepare_grasp_concurrently(
                        goal_handle,
                        False,
                    )
                    joint_preparation_complete = True

            close_check_enabled = (
                not request.dry_run
                and self._boolean("grasp_close_check_enabled")
            )
            max_grasp_attempts = (
                self._integer("grasp_max_empty_close_attempts")
                if close_check_enabled
                else 1
            )
            unlimited_empty_grasp_retries = (
                close_check_enabled and max_grasp_attempts == 0
            )
            measured_gripper_position: Optional[float] = None
            grasp_attempt = 0

            while True:
                grasp_attempt += 1
                attempt_label = (
                    str(grasp_attempt)
                    if unlimited_empty_grasp_retries
                    else f"{grasp_attempt}/{max_grasp_attempts}"
                )
                detection, grasp_pose_execution, arm_message = (
                    self._detect_and_execute_grasp(
                        goal_handle, request, arm, motion_state
                    )
                )
                result.grasp_pose = grasp_pose_execution
                result.score = float(detection.score)
                result.width = float(detection.width)
                result.height = float(detection.height)
                result.depth = float(detection.depth)
                result.object_id = int(detection.object_id)
                if request.publish_pose:
                    self.grasp_pose_publisher.publish(grasp_pose_execution)
                result.arm_message = arm_message

                if request.dry_run:
                    self._publish_grasp_feedback(
                        goal_handle,
                        "DRY_RUN_COMPLETE",
                        "arm plan succeeded; direct close and observation "
                        "return were skipped",
                        arm,
                    )
                    break

                self._publish_grasp_feedback(
                    goal_handle,
                    "CLOSING_GRIPPER",
                    f"closing selected gripper (grasp attempt "
                    f"{attempt_label})",
                    arm,
                )
                result.gripper_command_published = True
                close_measurements = self._close_grippers_and_measure(
                    goal_handle,
                    (arm,),
                    "while waiting for gripper close",
                    require_feedback=close_check_enabled,
                )

                if close_check_enabled:
                    measured_gripper_position = close_measurements[arm]
                    close_ratio = self._gripper_close_ratio(
                        measured_gripper_position
                    )
                    empty_close_ratio_threshold = self._float(
                        "grasp_empty_close_ratio_threshold"
                    )
                    if close_ratio > empty_close_ratio_threshold:
                        detail = (
                            f"{arm} gripper closed too far without retaining "
                            f"material: "
                            f"measured={measured_gripper_position:.3f}, "
                            f"close_ratio={close_ratio:.1%}, "
                            f"empty_threshold="
                            f"{empty_close_ratio_threshold:.1%}, grasp "
                            f"attempt={attempt_label}"
                        )
                        self.get_logger().warning(detail)
                        self._publish_grasp_feedback(
                            goal_handle,
                            "EMPTY_GRASP_DETECTED",
                            detail,
                            arm,
                        )
                        self._recover_grasp_observation(
                            goal_handle,
                            arm,
                            "empty grasp detected",
                            False,
                        )
                        result.torso_reset_command_published = True
                        if (
                            not unlimited_empty_grasp_retries
                            and grasp_attempt >= max_grasp_attempts
                        ):
                            raise MissionError(
                                f"grasp failed after {max_grasp_attempts} empty "
                                f"grasp attempts; {detail}"
                            )
                        self._publish_grasp_feedback(
                            goal_handle,
                            "RETRYING_EMPTY_GRASP",
                            "observation posture restored; requesting a fresh "
                            "detection for the next grasp attempt",
                            arm,
                        )
                        continue

                    self._publish_grasp_feedback(
                        goal_handle,
                        "GRASP_CONFIRMED",
                        f"material retained an opening in the {arm} gripper "
                        f"(measured={measured_gripper_position:.3f}, "
                        f"close_ratio={close_ratio:.1%}, "
                        f"empty_threshold="
                        f"{empty_close_ratio_threshold:.1%})",
                        arm,
                    )

                self._publish_grasp_feedback(
                    goal_handle,
                    "RETURNING_TO_OBSERVATION",
                    "returning directly to the final grasp observation posture",
                    arm,
                )
                self._prepare_grasp_arms_and_torso(goal_handle)
                result.torso_reset_command_published = True

                if close_check_enabled:
                    self._publish_grasp_feedback(
                        goal_handle,
                        "FINAL_GRASP_VERIFICATION",
                        "observation posture restored; reasserting the closed "
                        "gripper command and reading fresh feedback before "
                        "completing the action",
                        arm,
                    )
                    measured_gripper_position = self._close_grippers_and_measure(
                        goal_handle,
                        (arm,),
                        "while settling the final grasp verification",
                    )[arm]
                    close_ratio = self._gripper_close_ratio(
                        measured_gripper_position
                    )
                    empty_close_ratio_threshold = self._float(
                        "grasp_empty_close_ratio_threshold"
                    )
                    if close_ratio > empty_close_ratio_threshold:
                        detail = (
                            f"final grasp verification detected an empty grasp "
                            f"after returning to observation: {arm} gripper "
                            f"closed more than "
                            f"{empty_close_ratio_threshold:.1%}, indicating "
                            f"that no material remains between the fingers "
                            f"(measured={measured_gripper_position:.3f}, "
                            f"close_ratio={close_ratio:.1%})"
                        )
                        self.get_logger().warning(detail)
                        self._publish_grasp_feedback(
                            goal_handle,
                            "FINAL_EMPTY_GRASP_DETECTED",
                            detail,
                            arm,
                        )
                        self._publish_gripper(
                            goal_handle,
                            arm,
                            self._float("gripper_open_position"),
                        )
                        self._wait_delay(
                            goal_handle,
                            self._float("gripper_settle_sec"),
                            "while reopening after final empty-grasp "
                            "verification",
                        )
                        if (
                            not unlimited_empty_grasp_retries
                            and grasp_attempt >= max_grasp_attempts
                        ):
                            raise MissionError(
                                f"grasp failed after {max_grasp_attempts} "
                                f"empty grasp attempts; {detail}"
                            )
                        self._publish_grasp_feedback(
                            goal_handle,
                            "RETRYING_FINAL_EMPTY_GRASP",
                            "gripper reopened at the observation posture; "
                            "requesting a fresh detection for the next grasp "
                            "attempt",
                            arm,
                        )
                        continue

                    self._publish_grasp_feedback(
                        goal_handle,
                        "FINAL_GRASP_CONFIRMED",
                        f"material remains between the gripper fingers after "
                        f"returning to observation "
                        f"(measured={measured_gripper_position:.3f}, "
                        f"close_ratio={close_ratio:.1%}, "
                        f"empty_threshold="
                        f"{empty_close_ratio_threshold:.1%})",
                        arm,
                    )
                break

            result.success = True
            result.message = (
                "grasp dry run completed"
                if request.dry_run
                else (
                    "grasp mission completed"
                    if measured_gripper_position is None
                    else "grasp mission completed; retained-object gripper "
                    f"close ratio confirmed at {close_ratio:.1%} "
                    f"(measured={measured_gripper_position:.3f})"
                )
            )
            self._finalize_action_result(result, started_at, "execute_grasp")
            self._publish_grasp_feedback(goal_handle, "DONE", result.message, arm)
            goal_handle.succeed()
            completed = True
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
            result.message = f"unexpected grasp mission error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            safe_to_reset = all(
                (
                    torso_prepared,
                    not motion_state["started"],
                    not joint_preparation_started or joint_preparation_complete,
                )
            )
            if not completed and not request.dry_run and safe_to_reset:
                result.torso_reset_command_published = self._safe_pre_arm_torso_reset(
                    goal_handle
                )
            self._release_goal()
            self._finalize_action_result(result, started_at, "execute_grasp")

    def _execute_place(self, goal_handle) -> ExecutePlace.Result:
        started_at = time.monotonic()
        request = goal_handle.request
        arm = self._resolve_arm(request.arm)
        result = ExecutePlace.Result()
        result.arm = arm

        try:
            if request.dry_run:
                self._publish_place_feedback(
                    goal_handle,
                    "DRY_RUN_POSITIONING",
                    "skipping direct torso, arm-joint, and gripper commands",
                    arm,
                )
            else:
                self._publish_place_feedback(
                    goal_handle, "PREPARING_TORSO", "publishing place torso target", arm
                )
                self._publish_torso(
                    goal_handle, self._float_array("torso_prepare_positions")
                )
                self._wait_delay(
                    goal_handle,
                    self._float("torso_settle_sec"),
                    "while waiting for place torso preparation",
                )

                self._publish_place_feedback(
                    goal_handle,
                    "PREPARING_ARMS",
                    "calling configured right-arm place joint target",
                    arm,
                )
                self._call_arm_joints(
                    goal_handle,
                    [],
                    self._float_array("place_right_joint_positions"),
                    False,
                )
                self._wait_delay(
                    goal_handle,
                    self._float("arm_settle_sec"),
                    "while waiting for place arm preparation",
                )

                self._publish_place_feedback(
                    goal_handle, "OPENING_GRIPPER", "opening selected gripper", arm
                )
                self._publish_gripper(
                    goal_handle, arm, self._float("gripper_open_position")
                )
                result.gripper_command_published = True
                self._publish_place_feedback(
                    goal_handle,
                    "RELEASE_COMPLETE",
                    "gripper open command published; post-release arm and "
                    "torso motions are disabled",
                    arm,
                )

            result.success = True
            result.message = (
                "place dry run completed; post-release motions skipped"
                if request.dry_run
                else "place mission completed immediately after gripper release; "
                "post-release motions skipped"
            )
            self._finalize_action_result(result, started_at, "execute_place")
            self._publish_place_feedback(goal_handle, "DONE", result.message, arm)
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
            result.message = f"unexpected place mission error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            self._release_goal()
            self._finalize_action_result(result, started_at, "execute_place")
