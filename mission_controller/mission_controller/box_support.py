import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from task_interfaces.action import PickupTask
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import TransformException

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


class BoxSupportMixin:
    """FoundationPose normalization and dual-arm pickup delegation."""

    @staticmethod
    def _quaternion_from_rpy(
        roll: float, pitch: float, yaw: float
    ) -> tuple[float, float, float, float]:
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    def _publish_camera_mount_tf(self) -> None:
        if not self._boolean("camera_mount_tf_enabled"):
            self.get_logger().info("camera mount TF publication disabled")
            return

        xyz = self._float_array("camera_mount_xyz")
        mount_quaternion = self._quaternion_from_rpy(
            *self._float_array("camera_mount_rpy")
        )
        correction_quaternion = self._quaternion_from_rpy(
            *self._float_array("camera_mount_correction_rpy")
        )
        qx, qy, qz, qw = quaternion_multiply(
            correction_quaternion, mount_quaternion
        )
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self._string("camera_mount_parent_frame").lstrip("/")
        transform.child_frame_id = self._string("camera_mount_child_frame").lstrip("/")
        transform.transform.translation.x = xyz[0]
        transform.transform.translation.y = xyz[1]
        transform.transform.translation.z = xyz[2]
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.camera_static_broadcaster.sendTransform(transform)

    def _transform_detection_pose(
        self, pose: PoseStamped, target_frame: str
    ) -> PoseStamped:
        source_frame = pose.header.frame_id.strip().lstrip("/")
        target_frame = target_frame.strip().lstrip("/")
        if not source_frame:
            raise MissionError("box detector returned an empty source frame")
        if not target_frame or source_frame == target_frame:
            pose.header.frame_id = target_frame or source_frame
            return pose

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self._float("camera_tf_timeout_sec")),
            )
            transformed = do_transform_pose_stamped(pose, transform)
        except TransformException as exc:
            raise MissionError(
                f"box pose transform {source_frame} -> {target_frame} failed: {exc}"
            ) from exc

        transformed.header.frame_id = target_frame
        transformed.header.stamp = self.get_clock().now().to_msg()
        return transformed

    def _box_object_pose_camera_callback(self, pose: PoseStamped) -> None:
        """Publish the raw box pose after camera->execution-frame TF only."""
        try:
            transformed = self._transform_detection_pose(
                pose,
                self._string("arm_execution_frame"),
            )
        except MissionError as exc:
            self.get_logger().warning(
                f"cannot publish raw box pose in robot frame: {exc}"
            )
            return
        self.box_object_pose_raw_publisher.publish(transformed)

    def _forward_box_object_pose_feedback(
        self, goal_handle, feedback_message
    ) -> None:
        feedback = feedback_message.feedback
        self._publish_box_grasp_feedback(
            goal_handle,
            f"FOUNDATION_{feedback.stage}",
            f"FoundationPose progress={feedback.progress:.0%}",
        )

    def _make_pickup_box_pose(self, center_pose: PoseStamped) -> PoseStamped:
        """Apply an optional profile remap while preserving the geometric centre."""
        center_values = pose_to_array(center_pose.pose)
        model_to_pickup = self._quaternion_from_rpy(
            *self._float_array("box_foundation_to_pickup_rpy")
        )
        pickup_orientation = quaternion_multiply(
            tuple(center_values[3:]), model_to_pickup
        )
        orientation_norm = math.sqrt(
            sum(value * value for value in pickup_orientation)
        )
        pickup_orientation = tuple(
            value / orientation_norm for value in pickup_orientation
        )

        result = PoseStamped()
        result.header = center_pose.header
        result.pose.position.x = center_values[0]
        result.pose.position.y = center_values[1]
        result.pose.position.z = center_values[2]
        result.pose.orientation.x = pickup_orientation[0]
        result.pose.orientation.y = pickup_orientation[1]
        result.pose.orientation.z = pickup_orientation[2]
        result.pose.orientation.w = pickup_orientation[3]
        return result

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
        if not self._boolean("box_camera_pose_constraint_enabled"):
            return camera_pose

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
            if all(
                score >= min_dot for score in forward_alignment.values()
            ):
                correction_roll = math.pi / 2.0
                alignment = forward_alignment
                source_axes = "X down, Z forward, Y left"
            elif all(
                score >= min_dot for score in backward_alignment.values()
            ):
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

        local_x_correction = self._quaternion_from_rpy(
            correction_roll, 0.0, 0.0
        )
        corrected_orientation = quaternion_multiply(
            orientation, local_x_correction
        )
        orientation_norm = math.sqrt(
            sum(value * value for value in corrected_orientation)
        )

        corrected = PoseStamped()
        corrected.header = camera_pose.header
        corrected.pose.position.x = values[0]
        corrected.pose.position.y = values[1]
        corrected.pose.position.z = values[2]
        corrected.pose.orientation.x = (
            corrected_orientation[0] / orientation_norm
        )
        corrected.pose.orientation.y = (
            corrected_orientation[1] / orientation_norm
        )
        corrected.pose.orientation.z = (
            corrected_orientation[2] / orientation_norm
        )
        corrected.pose.orientation.w = (
            corrected_orientation[3] / orientation_norm
        )
        self.get_logger().info(
            "normalized FoundationPose camera-frame orientation for "
            f"{model_label or 'f320'} from [{source_axes}] with local "
            f"Rx({math.degrees(correction_roll):.1f} deg): "
            f"alignment={alignment}, threshold={min_dot:.3f}; "
            "canonical X is down, Y is forward, Z is right"
        )
        return corrected

    def _call_box_object_pose(self, goal_handle, request):
        if self.box_object_pose_client is None or EstimateObjectPose is None:
            raise MissionError(
                "box grasp requires the object_pose_interfaces package"
            )
        action_name = self._string("box_object_pose_action_name")
        # A configured value of zero disables the result deadline so a
        # first-time FoundationPose model load cannot trigger an immediate,
        # still-busy retry.
        timeout_sec = self._float("box_object_pose_result_timeout_sec")
        wait_deadline = time.monotonic() + self._float(
            "dependency_wait_timeout_sec"
        )
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
        foundation_goal.model_label = self._string("box_object_pose_model_label")
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
        deadline = (
            time.monotonic() + timeout_sec
            if timeout_sec > 0.0
            else None
        )
        try:
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested:
                    foundation_handle.cancel_goal_async()
                    raise MissionCanceled(
                        f"mission canceled during {action_name}"
                    )
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
            raise MissionError(
                f"{action_name} failed: {foundation_result.message}"
            )

        target_frame = self._string("arm_execution_frame").lstrip("/")
        constrained_camera_pose = self._constrain_box_camera_pose(
            foundation_result.pose
        )
        raw_foundation_center_pose = self._transform_detection_pose(
            foundation_result.pose, target_frame
        )
        foundation_center_pose = self._transform_detection_pose(
            constrained_camera_pose, target_frame
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
                first_segment_completed=attempt_state[
                    "first_segment_completed"
                ],
            ) from exc
        return str(pickup_result.message)

    def _recover_box_observation(
        self, goal_handle, reason: str, dry_run: bool
    ) -> None:
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
        if not dry_run:
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
        self, goal_handle, request, motion_state: dict[str, bool]
    ):
        detection_attempts = self._integer("box_detection_attempts")
        failures: list[str] = []

        for detection_attempt in range(1, detection_attempts + 1):
            clearance_active = False
            if not request.dry_run:
                self._wait_for_box_detection_posture(goal_handle)
            self._publish_box_grasp_feedback(
                goal_handle,
                "DETECTING_BOX",
                "requesting FoundationPose object pose estimation "
                f"(attempt {detection_attempt}/{detection_attempts})",
            )
            try:
                detection, box_pose = self._call_box_object_pose(
                    goal_handle, request
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
            if not request.dry_run:
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
                f"sending detection {detection_attempt}/{detection_attempts} "
                f"torso-frame box pose to "
                f"{self._string('pickup_task_action_name')}",
            )
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
                motion_state["started"] = (
                    motion_state["started"] or exc.motion_started
                )
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
            "box grasp exhausted fresh-detection attempts: "
            + " | ".join(failures)
        )
