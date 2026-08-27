import time

from mission_interfaces.action import (
    ExecuteBoxGrasp,
    ExecuteBoxPlace,
    ExecuteDragBoxGrasp,
)

from .common import MissionCanceled, MissionError


class BoxActionsMixin:
    """Mission-level box pickup and placement orchestration."""

    @staticmethod
    def _publish_box_grasp_feedback(
        goal_handle, stage: str, detail: str
    ) -> None:
        request = getattr(goal_handle, "request", None)
        feedback_type = (
            ExecuteDragBoxGrasp.Feedback
            if isinstance(request, ExecuteDragBoxGrasp.Goal)
            else ExecuteBoxGrasp.Feedback
        )
        feedback = feedback_type()
        feedback.stage = stage
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _publish_box_place_feedback(
        goal_handle, stage: str, detail: str
    ) -> None:
        feedback = ExecuteBoxPlace.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    def _execute_box_grasp(self, goal_handle) -> ExecuteBoxGrasp.Result:
        return self._execute_box_grasp_with_action_type(
            goal_handle, ExecuteBoxGrasp, "execute_box_grasp", tf_mode=False
        )

    def _execute_grasp_box_tf(self, goal_handle) -> ExecuteBoxGrasp.Result:
        """Execute GraspBox with the TF-based frozen-world target path."""
        return self._execute_box_grasp_with_action_type(
            goal_handle, ExecuteBoxGrasp, "grasp_box_tf", tf_mode=True
        )

    def _execute_drag_box_grasp(
        self, goal_handle
    ) -> ExecuteDragBoxGrasp.Result:
        return self._execute_box_grasp_with_action_type(
            goal_handle, ExecuteDragBoxGrasp, "execute_drag_box_grasp", tf_mode=False
        )

    def _execute_drag_box_grasp_tf(
        self, goal_handle
    ) -> ExecuteDragBoxGrasp.Result:
        """Execute DragBox with a detector-timestamp-frozen TF target."""
        return self._execute_box_grasp_with_action_type(
            goal_handle,
            ExecuteDragBoxGrasp,
            "execute_drag_box_grasp_tf",
            tf_mode=True,
        )

    def _execute_box_grasp_with_action_type(
        self, goal_handle, action_type, action_name: str, *, tf_mode: bool = False
    ):
        started_at = time.monotonic()
        request = goal_handle.request
        result = action_type.Result()
        motion_state = {
            "started": False,
            "gripper_command_published": False,
        }

        try:
            drag_mode = action_type is ExecuteDragBoxGrasp
            left_arm_enabled = drag_mode and self._boolean(
                "drag_box_left_arm_enabled"
            )
            left_join_mode = (
                self._string("drag_box_left_join_mode").strip().lower()
                if drag_mode
                else "immediate"
            )
            if drag_mode and left_join_mode not in ("immediate", "after_drag3"):
                raise MissionError(
                    "drag_box_left_join_mode must be 'immediate' or 'after_drag3'"
                )
            delayed_left_join = (
                left_arm_enabled and left_join_mode == "after_drag3"
            )
            right_arm_only = drag_mode and (
                not left_arm_enabled or delayed_left_join
            )
            if (
                right_arm_only
                and not request.dry_run
                and not self._boolean("box_direct_movel_enabled")
            ):
                raise MissionError(
                    "right-only or delayed-left DragBox execution requires "
                    "box_direct_movel_enabled=true"
                )
            self._publish_box_grasp_feedback(
                goal_handle,
                "INITIALIZING",
                "preparing box observation pose, torso, and grippers",
            )
            if request.dry_run:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "DRY_RUN_INITIALIZATION",
                    "skipping direct box preparation commands",
                )
            elif self._boolean("box_direct_movel_enabled"):
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "DIRECT_MOVEL_INITIALIZATION",
                    "direct rm_movel mode: using the current camera view; "
                    "skipping grippers, torso, and observation posture",
                )
            else:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "OPENING_INITIAL_GRIPPERS",
                    "opening both grippers before moving to the box "
                    "observation posture",
                )
                self._prepare_box_grasp_grippers(goal_handle)
                observation_ready, readiness_detail = (
                    self._box_observation_ready()
                )
                if observation_ready:
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "BOX_OBSERVATION_READY",
                        "arms and torso are already at the box observation "
                        f"posture; skipping preparation ({readiness_detail})",
                    )
                else:
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "PREPARING_BOX_OBSERVATION",
                        "box observation posture is not ready; moving through "
                        "the intermediate arms, then the final observation "
                        f"posture ({readiness_detail})",
                    )
                    self._prepare_box_grasp_concurrently(goal_handle)

            detection, box_pose, result.pickup_message = (
                self._detect_and_execute_box_pickup(
                    goal_handle,
                    request,
                    motion_state,
                    drag_mode=drag_mode,
                    right_arm_only=right_arm_only,
                    delayed_left_join=delayed_left_join,
                    tf_mode=tf_mode,
                )
            )
            result.box_pose = box_pose
            result.score = float(detection.detection_score)
            result.width = self._float("box_width")
            result.height = self._float("box_height")
            result.object_id = int(request.target_label)
            result.gripper_command_published = motion_state[
                "gripper_command_published"
            ]

            if request.dry_run:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "DRY_RUN_COMPLETE",
                    "box pickup planning succeeded; direct execution was skipped",
                )
            elif not self._boolean("box_direct_movel_enabled"):
                torso_lift_target = self._float_array(
                    "box_grasp_torso_lift_positions"
                )
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "LIFTING_BOX_TORSO1_CLEARANCE",
                    "verified box retention; moving Torso1 0.20 rad toward "
                    "zero while holding Torso2/3/4 at the pickup posture",
                )
                self._publish_torso(goal_handle, torso_lift_target)
                result.torso_lift_command_published = True
                self._wait_for_torso_target(
                    goal_handle,
                    torso_lift_target,
                    "confirming the Torso1 box-clearance lift",
                )

            result.success = True
            result.message = (
                "box grasp dry run completed"
                if request.dry_run
                else "box grasp mission completed"
            )
            self._finalize_action_result(
                result, started_at, action_name
            )
            self._publish_box_grasp_feedback(goal_handle, "DONE", result.message)
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
            result.message = f"unexpected box grasp mission error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            result.gripper_command_published = motion_state[
                "gripper_command_published"
            ]
            # A failed box pickup deliberately retains the validated box
            # observation posture. In particular, an IK/planning failure must
            # not publish torso_reset_positions=[0, 0, 0, 0]. A subsequent
            # action can detect the retained posture and skip initialization.
            self._release_goal()
            self._finalize_action_result(
                result, started_at, action_name
            )

    def _execute_box_place(self, goal_handle) -> ExecuteBoxPlace.Result:
        started_at = time.monotonic()
        request = goal_handle.request
        result = ExecuteBoxPlace.Result()

        try:
            if request.dry_run:
                self._publish_box_place_feedback(
                    goal_handle,
                    "DRY_RUN_POSITIONING",
                    "skipping direct box torso and gripper commands",
                )
            else:
                self._publish_box_place_feedback(
                    goal_handle,
                    "BENDING_TORSO",
                    "bending torso for box placement",
                )
                self._publish_torso(
                    goal_handle, self._float_array("box_place_torso_positions")
                )
                self._publish_box_place_feedback(
                    goal_handle,
                    "WAITING_FOR_TORSO_BEND",
                    "waiting for measured torso feedback to confirm the box "
                    "place bend before releasing the grippers",
                )
                self._wait_for_torso_target(
                    goal_handle,
                    self._float_array("box_place_torso_positions"),
                    "confirming the box place torso bend",
                )
                self._publish_box_place_feedback(
                    goal_handle,
                    "HOLDING_AT_BOX_PLACE_HEIGHT",
                    "box place torso target confirmed; holding for "
                    f"{self._float('box_place_release_delay_sec'):.1f}s "
                    "before releasing the grippers",
                )
                self._wait_delay(
                    goal_handle,
                    self._float("box_place_release_delay_sec"),
                    "while holding the confirmed box place height before release",
                )

                self._publish_box_place_feedback(
                    goal_handle,
                    "OPENING_BOX_GRIPPERS",
                    "opening both grippers to release the box",
                )
                self._publish_both_grippers(
                    goal_handle, self._float("gripper_open_position")
                )
                result.gripper_command_published = True
                self._wait_delay(
                    goal_handle,
                    self._float("gripper_settle_sec"),
                    "while waiting for box release",
                )

                self._publish_box_place_feedback(
                    goal_handle,
                    "LIFTING_ARMS_AFTER_RELEASE",
                    "moving both released arms to the box pickup clearance "
                    "posture while keeping the torso bent",
                )
                self._prepare_box_pickup_clearance_arms(goal_handle)

            if not request.dry_run:
                torso_intermediate = self._float_array(
                    "box_place_torso_straighten_intermediate_positions"
                )
                self._publish_box_place_feedback(
                    goal_handle,
                    "POSITIONING_TORSO2_FOR_STRAIGHTENING",
                    "moving torso2 from the box-place bend to the configured "
                    "intermediate posture before full straightening",
                )
                self._publish_torso(goal_handle, torso_intermediate)
                self._wait_for_torso_target(
                    goal_handle,
                    torso_intermediate,
                    "confirming the torso2 box-place intermediate posture",
                )

                self._publish_box_place_feedback(
                    goal_handle,
                    "STRAIGHTENING_TORSO",
                    "torso2 intermediate posture confirmed; straightening "
                    "and verifying all torso joints before moving the arms "
                    "to the fixed ready posture",
                )
                self._publish_torso(
                    goal_handle, self._float_array("torso_reset_positions")
                )
                result.torso_reset_command_published = True
                self._wait_for_torso_target(
                    goal_handle,
                    self._float_array("torso_reset_positions"),
                    "confirming the straight torso posture",
                )

            self._publish_box_place_feedback(
                goal_handle,
                "RETURNING_ARMS_TO_READY",
                "returning both arms to the fixed ready posture after the "
                "torso is straight",
            )
            self._call_go_ready(goal_handle, request.dry_run)
            result.ready_completed = True

            result.success = True
            result.message = (
                "box place dry run completed"
                if request.dry_run
                else "box place mission completed"
            )
            self._finalize_action_result(
                result, started_at, "execute_box_place"
            )
            self._publish_box_place_feedback(goal_handle, "DONE", result.message)
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
            result.message = f"unexpected box place mission error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            self._release_goal()
            self._finalize_action_result(
                result, started_at, "execute_box_place"
            )
