import math
from copy import deepcopy
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from task_interfaces.action import PickupTask

try:
    from task_interfaces.srv import MoveCartesian
except ImportError:
    MoveCartesian = None
try:
    from object_pose_interfaces.action import EstimateObjectPose
except ModuleNotFoundError:
    EstimateObjectPose = None

from .common import (
    MissionCanceled,
    MissionError,
    PickupAttemptError,
    pose_to_array,
    quaternion_multiply,
    rotate_vector,
)


class BoxPerceptionMixin:
    """FoundationPose validation, child calls, retries, and grasp closure."""

    def _constrain_box_camera_pose(self, camera_pose: PoseStamped) -> PoseStamped:
        """Normalize F320/F455 axes before camera-to-robot TF conversion.

        ROS optical coordinates use +X image-right, +Y image-down, and +Z
        camera-forward.  Downstream dual-arm pickup always expects the
        canonical F320 convention: object X down, Y camera-forward, and Z
        image-right.

        F320 only has a 180-degree local-X symmetry to resolve.  F455 is placed
        90 degrees around object X relative to F320: its Z points forward or
        backward while Y points left or right.  Apply the corresponding local
        +/-90-degree X rotation so both models reach the same canonical frame.
        """
        values = pose_to_array(camera_pose.pose)
        orientation = tuple(values[3:])
        object_x = rotate_vector((1.0, 0.0, 0.0), orientation)
        object_y = rotate_vector((0.0, 1.0, 0.0), orientation)
        object_z = rotate_vector((0.0, 0.0, 1.0), orientation)
        min_dot = self._float("box_camera_pose_axis_min_dot")
        model_label = self._string("box_object_pose_model_label").strip().lower()
        x_up_alignment = -object_x[1]
        if x_up_alignment >= min_dot:
            raise MissionError(
                "rejected FoundationPose camera-frame orientation: object X "
                f"points up (alignment={x_up_alignment:.3f}, "
                f"threshold={min_dot:.3f}, model={model_label or 'unknown'}); "
                "requesting a fresh detection instead of planning from a "
                "grossly inverted pose"
            )

        if model_label == "f455":
            forward_alignment = {
                "x_down": object_x[1],
                "z_forward": object_z[2],
                "y_left": -object_y[0],
            }
            backward_alignment = {
                "x_down": object_x[1],
                "z_backward": -object_z[2],
                "y_right": object_y[0],
            }
            if all(score >= min_dot for score in forward_alignment.values()):
                correction_roll = math.pi / 2.0
                alignment = forward_alignment
                source_axes = "X down, Z forward, Y left"
            elif all(score >= min_dot for score in backward_alignment.values()):
                correction_roll = -math.pi / 2.0
                alignment = backward_alignment
                source_axes = "X down, Z backward, Y right"
            else:
                self.get_logger().info(
                    "kept F455 FoundationPose camera-frame orientation; "
                    f"forward_alignment={forward_alignment}, "
                    f"backward_alignment={backward_alignment}, "
                    f"threshold={min_dot:.3f}"
                )
                return camera_pose
        else:
            alignment = {
                "x_down": object_x[1],
                "y_backward": -object_y[2],
                "z_left": -object_z[0],
            }
            if not all(score >= min_dot for score in alignment.values()):
                self.get_logger().info(
                    "kept F320 FoundationPose camera-frame orientation; "
                    f"symmetry alignment={alignment}, threshold={min_dot:.3f}"
                )
                return camera_pose
            correction_roll = math.pi
            source_axes = "X down, Y backward, Z left"

        local_x_correction = self._quaternion_from_rpy(correction_roll, 0.0, 0.0)
        corrected_orientation = quaternion_multiply(orientation, local_x_correction)
        orientation_norm = math.sqrt(
            sum(value * value for value in corrected_orientation)
        )

        corrected = PoseStamped()
        corrected.header = camera_pose.header
        corrected.pose.position.x = values[0]
        corrected.pose.position.y = values[1]
        corrected.pose.position.z = values[2]
        corrected.pose.orientation.x = corrected_orientation[0] / orientation_norm
        corrected.pose.orientation.y = corrected_orientation[1] / orientation_norm
        corrected.pose.orientation.z = corrected_orientation[2] / orientation_norm
        corrected.pose.orientation.w = corrected_orientation[3] / orientation_norm
        self.get_logger().info(
            "normalized FoundationPose camera-frame orientation for "
            f"{model_label or 'f320'} from [{source_axes}] with local "
            f"Rx({math.degrees(correction_roll):.1f} deg): "
            f"alignment={alignment}, threshold={min_dot:.3f}; "
            "canonical X is down, Y is forward, Z is right"
        )
        return corrected

    def _box_model_label_for_request(self, request) -> str:
        """Resolve the requested box model, with a configured fallback.

        Both box-grasp actions carry ``box_type``.  When it is present, the
        action goal selects the FoundationPose model for that run, so callers
        do not need to mutate a ROS parameter between bigbox and smallbox
        goals.  An empty field keeps the configured model for compatibility
        with older clients.
        """
        requested = str(getattr(request, "box_type", "") or "").strip().lower()
        aliases = {
            "big": "bigbox",
            "big_box": "bigbox",
            "small": "smallbox",
            "small_box": "smallbox",
        }
        model_label = aliases.get(
            requested,
            requested or self._string("box_object_pose_model_label").strip().lower(),
        )
        if model_label not in ("bigbox", "smallbox"):
            raise MissionError(
                "box_type must be 'bigbox' or 'smallbox' "
                f"(received '{requested or model_label}')"
            )
        return model_label

    def _call_box_object_pose(
        self,
        goal_handle,
        request,
        *,
        tf_mode: bool = False,
        drag_mode: bool = False,
        detection_arm: str | None = None,
    ):
        if self.box_object_pose_client is None or EstimateObjectPose is None:
            raise MissionError("box grasp requires the object_pose_interfaces package")
        action_name = self._string("box_object_pose_action_name")
        pre_settle_sec = self._float("box_foundation_pose_pre_settle_sec")
        if pre_settle_sec > 0.0:
            self._publish_box_grasp_feedback(
                goal_handle,
                "FOUNDATION_PRE_SETTLE",
                "holding the confirmed camera/robot posture for "
                f"{pre_settle_sec:.1f}s before FoundationPose",
            )
            self._wait_delay(
                goal_handle,
                pre_settle_sec,
                "while holding the camera/robot posture before FoundationPose",
            )
        # A configured value of zero disables the result deadline so a
        # first-time FoundationPose model load cannot trigger an immediate,
        # still-busy retry.
        timeout_sec = self._float("box_object_pose_result_timeout_sec")
        wait_deadline = time.monotonic() + self._float("dependency_wait_timeout_sec")
        while time.monotonic() < wait_deadline:
            self._check_canceled(goal_handle, f"while waiting for {action_name}")
            remaining = max(0.0, wait_deadline - time.monotonic())
            if self.box_object_pose_client.wait_for_server(
                timeout_sec=min(0.5, remaining)
            ):
                break
        else:
            raise MissionError(
                f"timeout waiting for action {action_name} after "
                f"{self._float('dependency_wait_timeout_sec'):.1f}s"
            )

        foundation_goal = EstimateObjectPose.Goal()
        model_label = self._box_model_label_for_request(request)
        if detection_arm is None:
            detection_arm = self._box_detection_arm(
                tf_mode=tf_mode, drag_mode=drag_mode
            )
        foundation_goal.camera_side = detection_arm
        foundation_goal.model_label = model_label
        configured_instance = self._integer("box_object_pose_instance_index")
        foundation_goal.instance_index = (
            int(request.target_label)
            if request.target_label >= 0
            else configured_instance
        )
        foundation_goal.confidence_threshold = self._float(
            "box_object_pose_confidence_threshold"
        )
        send_future = self.box_object_pose_client.send_goal_async(
            foundation_goal,
            feedback_callback=lambda message: self._forward_box_object_pose_feedback(
                goal_handle, message
            ),
        )
        foundation_handle = self._wait_future(
            send_future,
            goal_handle,
            f"sending {action_name} goal",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        if foundation_handle is None or not foundation_handle.accepted:
            raise MissionError(f"{action_name} goal was rejected")

        with self.state_lock:
            self.active_box_object_pose_goal_handle = foundation_handle
        result_future = foundation_handle.get_result_async()
        deadline = time.monotonic() + timeout_sec if timeout_sec > 0.0 else None
        try:
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested:
                    foundation_handle.cancel_goal_async()
                    raise MissionCanceled(f"mission canceled during {action_name}")
                if deadline is not None and time.monotonic() >= deadline:
                    foundation_handle.cancel_goal_async()
                    raise MissionError(
                        f"timeout waiting for {action_name} result after "
                        f"{timeout_sec:.1f}s"
                    )
                time.sleep(0.05)
            if not rclpy.ok():
                raise MissionError(f"ROS shutdown while waiting for {action_name}")
            wrapped_result = result_future.result()
        finally:
            with self.state_lock:
                self.active_box_object_pose_goal_handle = None

        foundation_result = wrapped_result.result
        succeeded = wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
        if not succeeded or not foundation_result.success:
            raise MissionError(f"{action_name} failed: {foundation_result.message}")

        post_settle_sec = self._float("box_foundation_pose_post_settle_sec")
        if post_settle_sec > 0.0:
            self._publish_box_grasp_feedback(
                goal_handle,
                "FOUNDATION_POST_SETTLE",
                "FoundationPose result received; holding the camera/robot "
                f"posture for {post_settle_sec:.1f}s before using the pose",
            )
            self._wait_delay(
                goal_handle,
                post_settle_sec,
                "while holding the camera/robot posture after FoundationPose",
            )

        target_frame = self._string("arm_execution_frame").lstrip("/")
        # TF GraspBox freezes the camera result in the chassis-fixed frame at
        # the detector timestamp.  All later waist/arm conversion uses live
        # TF; no ArmSlaveData EEPose or hand-entered camera extrinsic is used.
        camera_pose = foundation_result.pose
        if tf_mode:
            frozen_box_pose = self._transform_foundation_pose_to_tf_freeze_frame(
                camera_pose
            )
            self._last_grasp_box_tf_box_pose = frozen_box_pose
            raw_foundation_center_pose = deepcopy(frozen_box_pose)
            foundation_center_pose = deepcopy(frozen_box_pose)
            self._last_box_pose_by_arm = {}
        elif self._boolean("camera_measured_extrinsics_enabled"):
            target_mode = self._string("direct_movel_target_mode").strip().lower()
            raw_right_box_pose = self._measured_camera_pose_in_arm_base(
                camera_pose, "right"
            )
            left_correction_name = self._joint123_target_correction_parameter_name(
                "left", request.box_layer
            )
            right_correction_name = self._joint123_target_correction_parameter_name(
                "right", request.box_layer
            )
            left_offset_name = self._direct_movel_offset_parameter_name(
                "left", request.box_layer, model_label
            )
            right_offset_name = self._direct_movel_offset_parameter_name(
                "right", request.box_layer, model_label
            )
            left_camera_target, left_corrected_camera_box = (
                self._apply_box_frame_target_correction(
                    camera_pose,
                    left_offset_name,
                    left_correction_name,
                )
            )
            right_camera_target, right_corrected_camera_box = (
                self._apply_box_frame_target_correction(
                    camera_pose,
                    right_offset_name,
                    right_correction_name,
                )
            )
            left_box_pose = self._measured_camera_pose_in_arm_base(
                left_corrected_camera_box, "left"
            )
            right_box_pose = self._measured_camera_pose_in_arm_base(
                right_corrected_camera_box, "right"
            )
            left_arm_pose = self._measured_camera_pose_in_arm_base(
                left_camera_target, "left"
            )
            right_arm_pose = self._measured_camera_pose_in_arm_base(
                right_camera_target, "right"
            )
            if target_mode == "camera_offset_box_orientation":
                left_arm_pose = self._make_camera_offset_box_orientation_pose(
                    left_arm_pose, left_box_pose, "left"
                )
                right_arm_pose = self._make_camera_offset_box_orientation_pose(
                    right_arm_pose, right_box_pose, "right"
                )
            self._last_box_pose_by_arm = {
                "left": left_arm_pose,
                "right": right_arm_pose,
            }
            if self._boolean("box_direct_movel_enabled"):
                # In direct mode report the right-arm target in its measured
                # base frame.  For a wrist camera the dynamic Link8 EEPose is
                # used first, and the opposite-arm target additionally relies
                # on the live Base-to-Base TF relation.
                raw_foundation_center_pose = raw_right_box_pose
                foundation_center_pose = raw_right_box_pose
            else:
                raw_foundation_center_pose = self._arm_pose_in_execution_frame(
                    self._measured_camera_pose_in_arm_base(
                        foundation_result.pose, "right"
                    )
                )
                foundation_center_pose = self._arm_pose_in_execution_frame(
                    self._measured_camera_pose_in_arm_base(camera_pose, "right")
                )
        else:
            raw_foundation_center_pose = self._transform_detection_pose(
                foundation_result.pose, target_frame
            )
            foundation_center_pose = self._transform_detection_pose(
                camera_pose, target_frame
            )
        self.box_object_pose_raw_publisher.publish(raw_foundation_center_pose)
        pickup_box_pose = self._make_pickup_box_pose(foundation_center_pose)
        self.box_object_pose_publisher.publish(pickup_box_pose)
        self.get_logger().info(
            "prepared FoundationPose geometric-centre pose for pickup; "
            "camera-to-body TF is complete and any configured model-axis "
            "correction was applied exactly once"
        )
        return foundation_result, pickup_box_pose

    def _forward_pickup_task_feedback(
        self,
        goal_handle,
        detection_attempt: int,
        detection_attempts: int,
        attempt_state: dict[str, bool],
        feedback_message,
    ) -> None:
        feedback = feedback_message.feedback
        if feedback.stage == "APPROACHING":
            attempt_state["motion_started"] = True
            if "segment 2/" in feedback.detail:
                # PickupSkill only reports segment 2 after segment 1 returned
                # successfully, so this is a reliable recovery boundary.
                attempt_state["first_segment_completed"] = True
        self._publish_box_grasp_feedback(
            goal_handle,
            f"PICKUP_{feedback.stage}",
            f"detection {detection_attempt}/{detection_attempts}: "
            f"{feedback.detail} "
            f"(progress={feedback.progress:.0%})",
        )

    def _call_pickup_task(
        self,
        goal_handle,
        box_pose: PoseStamped,
        dry_run: bool,
        detection_attempt: int,
        detection_attempts: int,
    ) -> str:
        action_name = self._string("pickup_task_action_name")
        attempt_state = {
            "motion_started": False,
            "first_segment_completed": False,
        }
        self._publish_box_grasp_feedback(
            goal_handle,
            "PICKUP_ATTEMPT",
            f"calling {action_name} for detection "
            f"{detection_attempt}/{detection_attempts}",
        )
        pickup_goal = PickupTask.Goal()
        pickup_goal.box_pose = box_pose
        pickup_goal.box_width = self._float("box_width")
        pickup_goal.box_height = self._float("box_height")
        pickup_goal.box_type = self._string("box_type")
        pickup_goal.dry_run = dry_run
        try:
            pickup_result = self._call_task_action(
                goal_handle,
                self.pickup_task_client,
                action_name,
                pickup_goal,
                self._float("pickup_task_result_timeout_sec"),
                "active_pickup_task_goal_handle",
                feedback_callback=lambda message: (
                    self._forward_pickup_task_feedback(
                        goal_handle,
                        detection_attempt,
                        detection_attempts,
                        attempt_state,
                        message,
                    )
                ),
            )
        except MissionCanceled:
            raise
        except MissionError as exc:
            raise PickupAttemptError(
                str(exc),
                error_code=getattr(exc, "error_code", None),
                motion_started=attempt_state["motion_started"],
                first_segment_completed=attempt_state["first_segment_completed"],
            ) from exc
        return str(pickup_result.message)

    def _recover_box_observation(self, goal_handle, reason: str, dry_run: bool) -> None:
        self._publish_box_grasp_feedback(
            goal_handle,
            "RECOVERING_BOX_OBSERVATION",
            f"{reason}; returning directly to the box observation posture "
            "before re-detection",
        )
        if not dry_run:
            # Do not revisit the broad initialization waypoint here. The arms
            # are already on the pickup path, so return directly to the
            # validated final box observation joints while restoring the torso.
            self._prepare_box_grasp_arms_and_torso(goal_handle)

    def _open_box_grippers_for_redetection(
        self,
        goal_handle,
        reason: str,
        dry_run: bool,
    ) -> None:
        self._publish_box_grasp_feedback(
            goal_handle,
            "OPENING_BOX_GRIPPERS_FOR_REDETECTION",
            f"{reason}; opening both grippers before returning to detection",
        )
        if not dry_run and not self._boolean("box_direct_movel_enabled"):
            self._open_grippers(
                goal_handle,
                ("left", "right"),
                "while opening both box grippers before re-detection",
            )

    def _close_and_confirm_box_grasp(
        self,
        goal_handle,
        motion_state: dict[str, bool],
    ) -> None:
        self._publish_box_grasp_feedback(
            goal_handle,
            "CLOSING_BOX_GRIPPERS",
            "closing both grippers after pickup execution",
        )
        motion_state["gripper_command_published"] = True
        measured_positions = self._close_grippers_and_measure(
            goal_handle,
            ("left", "right"),
            "while waiting for both box grippers to close",
        )

        self._publish_box_grasp_feedback(
            goal_handle,
            "VERIFYING_BOX_GRASP",
            "checking both grippers for retained material before allowing "
            "any torso lift",
        )
        close_ratios = {
            gripper_arm: self._gripper_close_ratio(position)
            for gripper_arm, position in measured_positions.items()
        }
        empty_threshold = self._float("box_empty_close_ratio_threshold")
        empty_grippers = [
            gripper_arm
            for gripper_arm, close_ratio in close_ratios.items()
            if close_ratio > empty_threshold
        ]
        close_detail = ", ".join(
            f"{gripper_arm}: measured="
            f"{measured_positions[gripper_arm]:.3f}, "
            f"close_ratio={close_ratios[gripper_arm]:.1%}"
            for gripper_arm in ("left", "right")
        )
        if empty_grippers:
            detail = (
                "box grasp failed because "
                f"{'/'.join(empty_grippers)} gripper closed beyond "
                f"the {empty_threshold:.1%} empty-grasp threshold; "
                f"{close_detail}"
            )
            self._publish_box_grasp_feedback(
                goal_handle,
                "EMPTY_BOX_GRASP_DETECTED",
                detail,
            )
            raise MissionError(detail)

        self._publish_box_grasp_feedback(
            goal_handle,
            "BOX_GRASP_CONFIRMED",
            "both grippers retained the box below the empty-grasp threshold "
            f"before the Torso1 clearance lift ({close_detail})",
        )

    def _detect_and_execute_box_pickup(
        self,
        goal_handle,
        request,
        motion_state: dict[str, bool],
        *,
        drag_mode: bool = False,
        right_arm_only: bool = False,
        delayed_left_join: bool = False,
        tf_mode: bool = False,
    ):
        detection_attempts = self._integer("box_detection_attempts")
        failures: list[str] = []
        model_label = self._box_model_label_for_request(request)
        detection_arm = self._box_detection_arm(tf_mode=tf_mode, drag_mode=drag_mode)

        # Put the selected wrist camera at its calibrated observation
        # configuration before requesting FoundationPose.  Manipulation
        # ownership (for example, DragBox right-arm-only execution) is kept
        # independent from the arm that carries the detection camera.
        if self._boolean("box_direct_movel_enabled"):
            self._execute_pre_detection_arm_movej(
                goal_handle,
                request.dry_run,
                active_arms=(detection_arm,),
            )
            # Select the per-model detection table from the action goal for
            # both GraspBox and DragBox.  Bigbox keeps the existing generic
            # values, while a smallbox goal now uses its smallbox-specific
            # layer pose instead of silently falling back to the bigbox pose.
            detection_model_label = self._box_model_label_for_request(request)
            self._execute_pre_detection_arm_movej_fixed(
                goal_handle,
                request.dry_run,
                request.box_layer,
                detection_model_label,
                arm=detection_arm,
                tf_mode=tf_mode,
                drag_mode=drag_mode,
            )

        for detection_attempt in range(1, detection_attempts + 1):
            clearance_active = False
            if not request.dry_run and not self._boolean("box_direct_movel_enabled"):
                self._wait_for_box_detection_posture(goal_handle)
            self._publish_box_grasp_feedback(
                goal_handle,
                "DETECTING_BOX",
                "requesting FoundationPose object pose estimation "
                f"(attempt {detection_attempt}/{detection_attempts})",
            )
            try:
                detection, box_pose = self._call_box_object_pose(
                    goal_handle,
                    request,
                    tf_mode=tf_mode,
                    drag_mode=drag_mode,
                    detection_arm=detection_arm,
                )
            except MissionCanceled:
                raise
            except MissionError as exc:
                failure = (
                    f"detection {detection_attempt}/{detection_attempts} "
                    f"failed: {exc}"
                )
                failures.append(failure)
                self.get_logger().warning(failure)
                if detection_attempt < detection_attempts:
                    self._open_box_grippers_for_redetection(
                        goal_handle,
                        "FoundationPose detection failed",
                        request.dry_run,
                    )
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "REDETECTING_BOX",
                        "FoundationPose failed; requesting one fresh detection",
                    )
                continue

            self._check_canceled(goal_handle, "after FoundationPose estimation")
            if not request.dry_run and not self._boolean("box_direct_movel_enabled"):
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "MOVING_TO_PICKUP_CLEARANCE",
                    "moving both arms from the observation posture to the "
                    "recorded collision-clearance posture before pickup "
                    "planning",
                )
                try:
                    self._prepare_box_pickup_clearance_arms(goal_handle)
                    clearance_active = True
                except MissionCanceled:
                    raise
                except MissionError as exc:
                    failure = (
                        f"detection {detection_attempt}/{detection_attempts} "
                        f"clearance posture failed: {exc}"
                    )
                    failures.append(failure)
                    self.get_logger().warning(failure)
                    if detection_attempt < detection_attempts:
                        self._open_box_grippers_for_redetection(
                            goal_handle,
                            "box pickup-clearance motion failed",
                            request.dry_run,
                        )
                    self._recover_box_observation(
                        goal_handle,
                        "clearance posture failed or stopped before pickup",
                        request.dry_run,
                    )
                    if detection_attempt < detection_attempts:
                        self._publish_box_grasp_feedback(
                            goal_handle,
                            "REDETECTING_BOX",
                            "observation posture was restored after the "
                            "clearance move failed; capturing a fresh "
                            "FoundationPose estimate",
                        )
                    continue

            self._publish_box_grasp_feedback(
                goal_handle,
                "PLANNING_BOX_PICKUP",
                (
                    f"preparing detection {detection_attempt}/{detection_attempts} "
                    "for direct Link8 targets"
                    if self._boolean("box_direct_movel_enabled")
                    else (
                        f"sending detection {detection_attempt}/"
                        f"{detection_attempts} torso-frame box pose to "
                        f"{self._string('pickup_task_action_name')}"
                    )
                ),
            )
            if self._boolean("box_direct_movel_enabled"):
                try:
                    post_detection_left_detail = (
                        "drag_box_tf_post_detection_left_movej=not_applicable"
                    )
                    if tf_mode and drag_mode and detection_arm == "left":
                        post_detection_left_detail = (
                            self._execute_drag_box_tf_post_detection_left_movej(
                                goal_handle,
                                request.dry_run,
                                request.box_layer,
                                model_label,
                            )
                        )
                    pre_target_detail = self._execute_pre_target_arm_movej(
                        goal_handle,
                        request.dry_run,
                        right_arm_only=right_arm_only,
                    )
                    pickup_message = self._call_direct_box_movel(
                        goal_handle,
                        box_pose,
                        request.dry_run,
                        request.box_layer,
                        drag_mode=drag_mode,
                        right_arm_only=right_arm_only,
                        delayed_left_join=delayed_left_join,
                        tf_mode=tf_mode,
                        model_label=model_label,
                    )
                    pickup_message = (
                        f"{post_detection_left_detail}; "
                        f"{pre_target_detail}; {pickup_message}"
                    )
                    if not request.dry_run:
                        motion_state["started"] = True
                    return detection, box_pose, pickup_message
                except MissionCanceled:
                    raise
                except MissionError as exc:
                    failure = (
                        f"detection {detection_attempt}/{detection_attempts} "
                        f"direct {self._string('direct_movel_motion_mode')} failed: {exc}"
                    )
                    failures.append(failure)
                    self.get_logger().warning(failure)
                    # A native MoveL failure may leave either arm between
                    # waypoints; never retry perception and issue another
                    # Cartesian command without an explicit recovery posture.
                    raise MissionError("; ".join(failures)) from exc
            try:
                pickup_message = self._call_pickup_task(
                    goal_handle,
                    box_pose,
                    request.dry_run,
                    detection_attempt,
                    detection_attempts,
                )
                if not request.dry_run:
                    motion_state["started"] = True
                    try:
                        self._close_and_confirm_box_grasp(
                            goal_handle,
                            motion_state,
                        )
                    except MissionCanceled:
                        raise
                    except MissionError as exc:
                        failure = (
                            f"detection {detection_attempt}/"
                            f"{detection_attempts} box retention failed: {exc}"
                        )
                        failures.append(failure)
                        self.get_logger().warning(failure)
                        if detection_attempt < detection_attempts:
                            self._open_box_grippers_for_redetection(
                                goal_handle,
                                "box retention check failed",
                                request.dry_run,
                            )
                            self._recover_box_observation(
                                goal_handle,
                                "box retention check failed",
                                request.dry_run,
                            )
                            self._publish_box_grasp_feedback(
                                goal_handle,
                                "REDETECTING_BOX",
                                "box retention was not confirmed; observation "
                                "posture restored for a fresh FoundationPose "
                                "estimate",
                            )
                        continue
                return detection, box_pose, pickup_message
            except MissionCanceled:
                raise
            except PickupAttemptError as exc:
                motion_state["started"] = motion_state["started"] or exc.motion_started
                failure = (
                    f"detection {detection_attempt}/{detection_attempts} "
                    f"pickup failed: {exc}"
                )
                failures.append(failure)
                self.get_logger().warning(failure)

                if detection_attempt < detection_attempts:
                    self._open_box_grippers_for_redetection(
                        goal_handle,
                        "pickup execution failed",
                        request.dry_run,
                    )
                if clearance_active or exc.motion_started:
                    recovery_reason = (
                        "pickup stage 2 failed after the 10 cm pre-grasp "
                        "segment completed"
                        if exc.first_segment_completed
                        else (
                            "pickup execution failed after arm motion started"
                            if exc.motion_started
                            else "pickup IK/planning failed from the clearance "
                            "posture"
                        )
                    )
                    self._recover_box_observation(
                        goal_handle,
                        recovery_reason,
                        request.dry_run,
                    )

                if detection_attempt < detection_attempts:
                    if exc.error_code == 1 and not exc.motion_started:
                        retry_reason = (
                            "pickup IK/planning failed before arm execution; "
                            "capturing a fresh FoundationPose estimate"
                        )
                    elif exc.motion_started:
                        retry_reason = (
                            "pickup execution failed and observation posture "
                            "was restored; capturing a fresh FoundationPose "
                            "estimate"
                        )
                    else:
                        retry_reason = (
                            "pickup failed before arm execution; capturing a "
                            "fresh FoundationPose estimate"
                        )
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "REDETECTING_BOX",
                        retry_reason,
                    )

        raise MissionError(
            "box grasp exhausted fresh-detection attempts: " + " | ".join(failures)
        )
