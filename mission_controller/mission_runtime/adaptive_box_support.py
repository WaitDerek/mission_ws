"""Support primitives for the timestamp-frozen direct-SDK box mission."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Sequence

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import TransformException

try:
    from object_pose_interfaces.action import EstimateObjectPose
except ModuleNotFoundError:
    EstimateObjectPose = None

from .common import (
    MissionCanceled,
    MissionError,
    pose_to_array,
    quaternion_multiply,
    rotate_vector,
)
from .realman_sdk_adapter import (
    RealManSdkCanceled,
    RealManSdkError,
    pose_to_sdk_target,
)


def normalize_vector(values: Sequence[float], label: str) -> tuple[float, float, float]:
    """Return one finite normalized 3-vector."""
    if len(values) != 3:
        raise MissionError(f"{label} must contain three values")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise MissionError(f"{label} contains NaN or Inf")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise MissionError(f"{label} has zero norm")
    return tuple(value / norm for value in vector)


def normalize_quaternion(
    values: Sequence[float], label: str
) -> tuple[float, float, float, float]:
    """Return one finite normalized x/y/z/w quaternion."""
    if len(values) != 4:
        raise MissionError(f"{label} must contain four values")
    quaternion = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in quaternion):
        raise MissionError(f"{label} contains NaN or Inf")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise MissionError(f"{label} has zero norm")
    return tuple(value / norm for value in quaternion)


def compose_pose(
    parent_pose: Pose,
    child_xyz: Sequence[float],
    child_quaternion: Sequence[float],
) -> Pose:
    """Compose T_parent_object with T_object_child."""
    parent = pose_to_array(parent_pose)
    parent_q = tuple(parent[3:])
    local_xyz = tuple(float(value) for value in child_xyz)
    local_q = normalize_quaternion(child_quaternion, "child quaternion")
    rotated_xyz = rotate_vector(local_xyz, parent_q)
    result_q = normalize_quaternion(
        quaternion_multiply(parent_q, local_q), "composed quaternion"
    )

    result = Pose()
    result.position.x = parent[0] + rotated_xyz[0]
    result.position.y = parent[1] + rotated_xyz[1]
    result.position.z = parent[2] + rotated_xyz[2]
    result.orientation.x = result_q[0]
    result.orientation.y = result_q[1]
    result.orientation.z = result_q[2]
    result.orientation.w = result_q[3]
    return result


class AdaptiveBoxSupportMixin:
    """Detection, frozen geometry, and native dual-arm MoveL helpers."""

    def _forward_adaptive_detection_feedback(
        self, goal_handle, feedback_message
    ) -> None:
        feedback = feedback_message.feedback
        self._publish_adaptive_feedback(
            goal_handle,
            f"DETECTING_{feedback.stage}",
            min(0.18, max(0.01, 0.18 * float(feedback.progress))),
            f"FoundationPose progress={feedback.progress:.0%}",
        )

    def _call_adaptive_detection(self, goal_handle, request):
        if self.box_object_pose_client is None or EstimateObjectPose is None:
            raise MissionError(
                "adaptive box grasp requires object_pose_interfaces"
            )
        action_name = self._string("box_object_pose_action_name")
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

        detection_goal = EstimateObjectPose.Goal()
        detection_goal.model_label = self._string("box_object_pose_model_label")
        configured_instance = self._integer("box_object_pose_instance_index")
        detection_goal.instance_index = (
            int(request.target_instance_index)
            if request.target_instance_index >= 0
            else configured_instance
        )
        detection_goal.confidence_threshold = self._float(
            "box_object_pose_confidence_threshold"
        )
        send_future = self.box_object_pose_client.send_goal_async(
            detection_goal,
            feedback_callback=lambda message: (
                self._forward_adaptive_detection_feedback(goal_handle, message)
            ),
        )
        detection_handle = self._wait_future(
            send_future,
            goal_handle,
            f"sending {action_name} goal",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        if detection_handle is None or not detection_handle.accepted:
            raise MissionError(f"{action_name} goal was rejected")

        with self.state_lock:
            self.active_box_object_pose_goal_handle = detection_handle
        result_future = detection_handle.get_result_async()
        timeout_sec = self._float("box_object_pose_result_timeout_sec")
        deadline = time.monotonic() + timeout_sec if timeout_sec > 0.0 else None
        try:
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested:
                    detection_handle.cancel_goal_async()
                    raise MissionCanceled(
                        f"mission canceled during {action_name}"
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    detection_handle.cancel_goal_async()
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

        result = wrapped_result.result
        if (
            wrapped_result.status != GoalStatus.STATUS_SUCCEEDED
            or not result.success
        ):
            raise MissionError(f"{action_name} failed: {result.message}")
        return result

    def _freeze_detection_pose(self, detection_pose: PoseStamped) -> PoseStamped:
        source_frame = detection_pose.header.frame_id.strip().lstrip("/")
        target_frame = self._string("adaptive_freeze_frame").lstrip("/")
        if target_frame != "base_link":
            raise MissionError(
                "adaptive_freeze_frame must remain base_link for this mission"
            )
        if not source_frame:
            raise MissionError("detector returned an empty camera frame")
        stamp = detection_pose.header.stamp
        if (
            self._boolean("adaptive_require_detection_timestamp")
            and stamp.sec == 0
            and stamp.nanosec == 0
        ):
            raise MissionError(
                "detector returned a zero timestamp; cannot freeze historical TF"
            )

        pose = deepcopy(detection_pose)
        pose.header.frame_id = source_frame
        if source_frame == target_frame:
            pose.header.frame_id = target_frame
            return pose

        query_time = Time.from_msg(stamp)
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                query_time,
                timeout=Duration(
                    seconds=self._float("adaptive_detection_tf_timeout_sec")
                ),
            )
            frozen = do_transform_pose_stamped(pose, transform)
        except TransformException as exc:
            raise MissionError(
                "timestamped target freeze failed: "
                f"{source_frame} -> {target_frame} at "
                f"{stamp.sec}.{stamp.nanosec:09d}: {exc}"
            ) from exc
        frozen.header.frame_id = target_frame
        frozen.header.stamp = stamp
        return frozen

    def _compute_adaptive_grasp_poses(
        self, object_pose_base: PoseStamped
    ) -> tuple[PoseStamped, PoseStamped]:
        span_axis = normalize_vector(
            self._float_array("adaptive_grasp_span_axis_object"),
            "adaptive_grasp_span_axis_object",
        )
        height_axis = normalize_vector(
            self._float_array("adaptive_grasp_height_axis_object"),
            "adaptive_grasp_height_axis_object",
        )
        half_span = (
            0.5 * self._float("box_width")
            + self._float("adaptive_grasp_side_clearance_m")
        )
        height_offset = self._float("adaptive_grasp_height_offset_m")
        left_xyz = tuple(
            -half_span * span + height_offset * height
            for span, height in zip(span_axis, height_axis)
        )
        right_xyz = tuple(
            half_span * span + height_offset * height
            for span, height in zip(span_axis, height_axis)
        )

        base_correction = self._quaternion_from_rpy(
            *self._float_array("adaptive_grasp_correction_rpy")
        )
        left_extra = self._quaternion_from_rpy(
            *self._float_array("adaptive_left_grasp_extra_rpy")
        )
        right_extra = self._quaternion_from_rpy(
            *self._float_array("adaptive_right_grasp_extra_rpy")
        )
        left_local_q = normalize_quaternion(
            quaternion_multiply(base_correction, left_extra),
            "left object-to-grasp quaternion",
        )
        right_local_q = normalize_quaternion(
            quaternion_multiply(base_correction, right_extra),
            "right object-to-grasp quaternion",
        )

        left = PoseStamped()
        left.header = deepcopy(object_pose_base.header)
        left.pose = compose_pose(object_pose_base.pose, left_xyz, left_local_q)
        right = PoseStamped()
        right.header = deepcopy(object_pose_base.header)
        right.pose = compose_pose(object_pose_base.pose, right_xyz, right_local_q)
        return left, right

    def _transform_pose_latest(
        self, pose: PoseStamped, target_frame: str
    ) -> PoseStamped:
        target_frame = target_frame.lstrip("/")
        source_frame = pose.header.frame_id.lstrip("/")
        if target_frame == source_frame:
            return deepcopy(pose)
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(
                    seconds=self._float("adaptive_runtime_tf_timeout_sec")
                ),
            )
            result = do_transform_pose_stamped(pose, transform)
        except TransformException as exc:
            raise MissionError(
                f"runtime pose transform {source_frame} -> {target_frame} failed: {exc}"
            ) from exc
        result.header.frame_id = target_frame
        result.header.stamp = self.get_clock().now().to_msg()
        return result

    def _fixture_center_to_sdk_link8_target(
        self, grasp_pose_base: PoseStamped, arm: str
    ) -> Pose:
        prefix = "left" if arm == "left" else "right"
        arm_target = self._transform_pose_latest(
            grasp_pose_base, self._string(f"{prefix}_arm_base_frame")
        )
        values = pose_to_array(arm_target.pose)
        quaternion = tuple(values[3:])
        if self._boolean("direct_movel_fixture_compensation_enabled"):
            fixture_xyz = self._float_array(
                f"{prefix}_fixture_center_in_link8_xyz"
            )
            fixture_offset = rotate_vector(tuple(fixture_xyz), quaternion)
            arm_target.pose.position.x -= fixture_offset[0]
            arm_target.pose.position.y -= fixture_offset[1]
            arm_target.pose.position.z -= fixture_offset[2]
        return arm_target.pose

    def _adaptive_sdk_targets(
        self,
        left_grasp_pose_base: PoseStamped,
        right_grasp_pose_base: PoseStamped,
        lift_offset_m: float,
    ) -> tuple[Pose, Pose]:
        left = deepcopy(left_grasp_pose_base)
        right = deepcopy(right_grasp_pose_base)
        left.pose.position.z += float(lift_offset_m)
        right.pose.position.z += float(lift_offset_m)
        return (
            self._fixture_center_to_sdk_link8_target(left, "left"),
            self._fixture_center_to_sdk_link8_target(right, "right"),
        )

    def _execute_adaptive_dual_movel(
        self,
        goal_handle,
        left_grasp_pose_base: PoseStamped,
        right_grasp_pose_base: PoseStamped,
        *,
        lift_offset_m: float,
        velocity_parameter: str,
        timeout_parameter: str,
        phase: str,
        dry_run: bool,
    ) -> str:
        left_target, right_target = self._adaptive_sdk_targets(
            left_grasp_pose_base,
            right_grasp_pose_base,
            lift_offset_m,
        )
        detail = (
            f"native dual-arm rm_movel {phase} targets prepared in the "
            "left/right arm base frames from frozen base_link poses"
        )
        if lift_offset_m:
            detail += f" with base_link +Z offset={lift_offset_m:.3f}m"
        if dry_run:
            return f"{detail}; SDK execution skipped"
        if self._string("direct_motion_backend").lower() != "python_sdk":
            raise MissionError(
                "adaptive direct MoveL requires direct_motion_backend=python_sdk"
            )
        if self.direct_sdk_adapter is None:
            raise MissionError("RealMan Python SDK adapter is not initialized")
        try:
            message = self.direct_sdk_adapter.execute_dual(
                pose_to_sdk_target(left_target),
                pose_to_sdk_target(right_target),
                "movel",
                self._float(velocity_parameter),
                True,
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=self._float(timeout_parameter),
            )
        except RealManSdkCanceled as exc:
            raise MissionCanceled(str(exc)) from exc
        except (RealManSdkError, ValueError) as exc:
            raise MissionError(str(exc)) from exc
        return f"{detail}; {message}"

    def _execute_adaptive_grasp_movel(
        self,
        goal_handle,
        left_grasp_pose_base: PoseStamped,
        right_grasp_pose_base: PoseStamped,
        dry_run: bool,
    ) -> str:
        return self._execute_adaptive_dual_movel(
            goal_handle,
            left_grasp_pose_base,
            right_grasp_pose_base,
            lift_offset_m=0.0,
            velocity_parameter="adaptive_grasp_velocity_percent",
            timeout_parameter="adaptive_grasp_timeout_sec",
            phase="grasp approach",
            dry_run=dry_run,
        )

    def _execute_adaptive_movel_lift(
        self,
        goal_handle,
        left_grasp_pose_base: PoseStamped,
        right_grasp_pose_base: PoseStamped,
        dry_run: bool,
    ) -> str:
        return self._execute_adaptive_dual_movel(
            goal_handle,
            left_grasp_pose_base,
            right_grasp_pose_base,
            lift_offset_m=self._float("adaptive_lift_distance_m"),
            velocity_parameter="adaptive_lift_velocity_percent",
            timeout_parameter="adaptive_lift_timeout_sec",
            phase="vertical lift",
            dry_run=dry_run,
        )
