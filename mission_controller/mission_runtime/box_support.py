import json
import math
from copy import deepcopy
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.duration import Duration
from rm_robot_interfaces.srv import StringCmd
from task_interfaces.action import PickupTask
try:
    from task_interfaces.srv import MoveCartesian
except ImportError:
    MoveCartesian = None
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
from .realman_sdk_adapter import (
    RealManSdkCanceled,
    RealManSdkError,
    pose_to_sdk_target,
)


class BoxSupportMixin:
    """FoundationPose normalization and dual-arm pickup delegation."""

    def _slave_arm_feedback_callback(self, arm: str, message) -> None:
        """Cache fresh RX arm joint position/velocity feedback."""
        joint_state = message.joint_state
        if (
            len(joint_state.name) != len(joint_state.position)
            or len(joint_state.name) != len(joint_state.velocity)
        ):
            return
        names = [str(name) for name in joint_state.name]
        raw_positions = [float(value) for value in joint_state.position]
        raw_velocities = [float(value) for value in joint_state.velocity]
        expected_names = [f"joint{index}" for index in range(1, 8)]
        if all(name in names for name in expected_names):
            positions_by_name = dict(zip(names, raw_positions))
            velocities_by_name = dict(zip(names, raw_velocities))
            positions = [positions_by_name[name] for name in expected_names]
            velocities = [velocities_by_name[name] for name in expected_names]
        else:
            positions = raw_positions
            velocities = raw_velocities
        if not positions or len(velocities) != len(positions):
            return
        if not all(
            math.isfinite(value) for value in (*positions, *velocities)
        ):
            return
        if arm not in ("left", "right"):
            return

        # The replacement robot reports the Link8 pose in ArmSlaveData.pose.
        # Link8 is also the controller TCP origin on this hardware, but the
        # user-installed fixture remains a separate local offset and is
        # compensated later by _make_direct_movel_pose().  Do not reject an
        # otherwise valid joint sample just because a driver omits pose; the
        # dynamic camera path will report a clear error when it is required.
        link8_pose = getattr(message, "pose", None)
        pose_frame = ""
        header = getattr(message, "header", None)
        if header is not None:
            pose_frame = str(getattr(header, "frame_id", "")).strip().lstrip("/")
        pose_values = None
        if link8_pose is not None:
            pose_values = (
                float(link8_pose.position.x),
                float(link8_pose.position.y),
                float(link8_pose.position.z),
                float(link8_pose.orientation.x),
                float(link8_pose.orientation.y),
                float(link8_pose.orientation.z),
                float(link8_pose.orientation.w),
            )
            pose_norm = math.sqrt(sum(value * value for value in pose_values[3:]))
            if (
                not all(math.isfinite(value) for value in pose_values)
                or pose_norm <= 1e-12
            ):
                pose_values = None
            else:
                pose_values = pose_values[:3] + tuple(
                    value / pose_norm for value in pose_values[3:]
                )
        with self.joint_state_lock:
            state_time = time.monotonic()
            self.latest_slave_arm_positions[arm] = positions
            self.latest_slave_arm_velocities[arm] = velocities
            self.latest_slave_arm_state_times[arm] = state_time
            self.latest_slave_arm_state_sequences[arm] += 1
            if pose_values is not None:
                self.latest_slave_arm_poses[arm] = pose_values
                self.latest_slave_arm_pose_times[arm] = state_time
                self.latest_slave_arm_pose_sequences[arm] += 1
                if pose_frame:
                    self.latest_slave_arm_pose_frames[arm] = pose_frame

    def _body_joint1_feedback_callback(self, message) -> None:
        joint_state = message.joint_state
        if len(joint_state.name) != len(joint_state.position):
            return
        names = [
            self._string(f"box_joint{index}_name")
            for index in range(1, 5)
        ]
        positions = {}
        velocities = {}
        for joint_name in names:
            try:
                index = list(joint_state.name).index(joint_name)
            except ValueError:
                continue
            position = float(joint_state.position[index])
            velocity = (
                float(joint_state.velocity[index])
                if index < len(joint_state.velocity)
                else math.inf
            )
            if math.isfinite(position) and math.isfinite(velocity):
                positions[joint_name] = position
                velocities[joint_name] = velocity
        joint1_name = names[0]
        if joint1_name not in positions:
            return
        with self.joint_state_lock:
            state_time = time.monotonic()
            self.latest_body_joint_positions = positions
            self.latest_body_joint_velocities = velocities
            self.latest_body_state_time = state_time
            self.latest_body_state_sequence += 1
            self.latest_body_joint1_position = positions[joint1_name]
            self.latest_body_joint1_velocity = velocities[joint1_name]
            self.latest_body_joint1_state_time = state_time
            self.latest_body_joint1_state_sequence += 1

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

    def _slave_arm_pose_snapshot(self, arm: str):
        """Return the latest Link8 EEPose and its monotonic age."""
        if arm not in ("left", "right"):
            raise MissionError(f"unsupported camera arm '{arm}'")
        with self.joint_state_lock:
            pose_values = self.latest_slave_arm_poses.get(arm)
            state_time = self.latest_slave_arm_pose_times.get(arm, 0.0)
        return pose_values, time.monotonic() - state_time

    def _arm_base_frame(self, arm: str) -> str:
        """Prefer the feedback frame_id, falling back to the configured name."""
        prefix = "left" if arm == "left" else "right"
        with self.joint_state_lock:
            feedback_frame = getattr(
                self, "latest_slave_arm_pose_frames", {}
            ).get(arm, "")
        return (
            feedback_frame
            or self._string(f"{prefix}_arm_base_frame").strip().lstrip("/")
        )

    def _dynamic_camera_pose_in_arm_base(
        self,
        camera_pose: PoseStamped,
        target_arm: str,
        detection_arm: str,
    ) -> PoseStamped:
        """Convert a wrist-camera detection using live Link8 EEPose.

        The replacement robot mounts the RGB camera on the arm Link8.  The
        corresponding ArmSlaveData.pose is the Link8/TCP pose in that arm's
        Base frame.  The configured Link8->RGB transform is fixed, while the
        Base->Link8 part is read for every detection.  The cross-arm conversion
        uses the configured fixed Base-to-Base transform so this path does not
        depend on a TF tree published by the new robot.
        """
        target_arm = target_arm.strip().lower()
        detection_arm = detection_arm.strip().lower()
        if target_arm not in ("left", "right"):
            raise MissionError(f"unsupported target arm '{target_arm}'")
        if detection_arm not in ("left", "right"):
            raise MissionError(f"unsupported detection arm '{detection_arm}'")

        link8_values, pose_age = self._slave_arm_pose_snapshot(detection_arm)
        max_age = self._float("camera_eepose_max_age_sec")
        if link8_values is None:
            raise MissionError(
                f"no Link8 EEPose has been received from "
                f"{detection_arm} arm feedback topic"
            )
        if pose_age > max_age:
            raise MissionError(
                f"{detection_arm} Link8 EEPose is stale: "
                f"age={pose_age:.3f}s, limit={max_age:.3f}s"
            )

        prefix = "left" if detection_arm == "left" else "right"
        link8_to_camera = (
            tuple(
                self._float_array(
                    f"camera_{prefix}_link8_to_rgb_camera_xyz"
                )
            ),
            BoxSupportMixin._normalize_quaternion(
                tuple(
                    self._float_array(
                        f"camera_{prefix}_link8_to_rgb_camera_quaternion_xyzw"
                    )
                )
            ),
        )
        base_to_link8 = (
            tuple(link8_values[:3]),
            BoxSupportMixin._normalize_quaternion(
                tuple(link8_values[3:])
            ),
        )
        base_to_camera = BoxSupportMixin._compose_transform(
            base_to_link8, link8_to_camera
        )
        camera_to_box = (
            tuple(pose_to_array(camera_pose.pose)[:3]),
            BoxSupportMixin._normalize_quaternion(
                tuple(pose_to_array(camera_pose.pose)[3:])
            ),
        )
        detection_base_to_box = BoxSupportMixin._compose_transform(
            base_to_camera, camera_to_box
        )

        detection_base_frame = self._arm_base_frame(detection_arm)
        target_base_frame = self._arm_base_frame(target_arm)
        result = PoseStamped()
        result.header = deepcopy(camera_pose.header)
        result.header.frame_id = detection_base_frame
        result.pose.position.x = detection_base_to_box[0][0]
        result.pose.position.y = detection_base_to_box[0][1]
        result.pose.position.z = detection_base_to_box[0][2]
        result.pose.orientation.x = detection_base_to_box[1][0]
        result.pose.orientation.y = detection_base_to_box[1][1]
        result.pose.orientation.z = detection_base_to_box[1][2]
        result.pose.orientation.w = detection_base_to_box[1][3]

        if target_arm == detection_arm:
            return result

        if self._boolean("camera_fixed_cross_arm_transform_enabled"):
            # The configured transform is T_left_right: it maps coordinates
            # expressed in the right Base frame into the left Base frame.
            # The opposite direction is its exact rigid-transform inverse.
            right_to_left = (
                tuple(
                    self._float_array("camera_right_base_to_left_base_xyz")
                ),
                self._normalize_quaternion(
                    tuple(
                        self._float_array(
                            "camera_right_base_to_left_base_quaternion_xyzw"
                        )
                    )
                ),
            )
            cross_arm_transform = (
                right_to_left
                if detection_arm == "right" and target_arm == "left"
                else self._inverse_transform(right_to_left)
            )
            transformed_values = self._compose_transform(
                cross_arm_transform,
                (
                    (
                        result.pose.position.x,
                        result.pose.position.y,
                        result.pose.position.z,
                    ),
                    self._normalize_quaternion(
                        (
                            result.pose.orientation.x,
                            result.pose.orientation.y,
                            result.pose.orientation.z,
                            result.pose.orientation.w,
                        )
                    ),
                ),
            )
            transformed = PoseStamped()
            transformed.header = deepcopy(result.header)
            transformed.header.frame_id = target_base_frame
            transformed.pose.position.x = transformed_values[0][0]
            transformed.pose.position.y = transformed_values[0][1]
            transformed.pose.position.z = transformed_values[0][2]
            transformed.pose.orientation.x = transformed_values[1][0]
            transformed.pose.orientation.y = transformed_values[1][1]
            transformed.pose.orientation.z = transformed_values[1][2]
            transformed.pose.orientation.w = transformed_values[1][3]
            return transformed

        try:
            transform = self.tf_buffer.lookup_transform(
                target_base_frame,
                detection_base_frame,
                rclpy.time.Time(),
                timeout=Duration(
                    seconds=self._float("camera_tf_timeout_sec")
                ),
            )
            transformed = do_transform_pose_stamped(result, transform)
        except TransformException as exc:
            raise MissionError(
                f"dynamic camera pose requires TF "
                f"{detection_base_frame} -> {target_base_frame}: {exc}"
            ) from exc
        transformed.header.frame_id = target_base_frame
        return transformed

    def _transform_foundation_pose_to_tf_freeze_frame(
        self, camera_pose: PoseStamped
    ) -> PoseStamped:
        """Transform a detector pose into the chassis-fixed TF frame.

        The lookup is performed at FoundationPose's capture timestamp.  This
        freezes the physical box before any waist motion; using ``Time()``
        here would silently mix a past camera sample with a newer robot pose.
        """
        target_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        source_frame = camera_pose.header.frame_id.strip().lstrip("/")
        if not target_frame:
            raise MissionError("grasp_box_tf_freeze_frame must not be empty")
        if not source_frame:
            raise MissionError(
                "FoundationPose returned an empty camera frame for TF GraspBox"
            )
        stamp = camera_pose.header.stamp
        stamp_is_zero = int(stamp.sec) == 0 and int(stamp.nanosec) == 0
        if stamp_is_zero and self._boolean(
            "grasp_box_tf_require_detection_timestamp"
        ):
            raise MissionError(
                "TF GraspBox requires a non-zero FoundationPose detection timestamp"
            )
        lookup_time = (
            rclpy.time.Time.from_msg(stamp)
            if not stamp_is_zero
            else rclpy.time.Time()
        )
        if source_frame == target_frame:
            transformed = deepcopy(camera_pose)
            transformed.header.frame_id = target_frame
            return transformed
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                lookup_time,
                timeout=Duration(
                    seconds=self._float("grasp_box_tf_detection_tf_timeout_sec")
                ),
            )
            transformed = do_transform_pose_stamped(camera_pose, transform)
        except TransformException as exc:
            raise MissionError(
                "TF GraspBox detection transform "
                f"{source_frame} -> {target_frame} failed at detector timestamp: {exc}"
            ) from exc
        transformed.header.frame_id = target_frame
        return transformed

    def _tf_target_pose_in_arm_base(
        self, target: PoseStamped, arm: str
    ) -> Pose:
        """Convert a frozen base-frame Link8 target using the live arm-base TF."""
        prefix = "left" if arm == "left" else "right"
        target_frame = self._string(f"{prefix}_arm_base_frame").strip().lstrip("/")
        source_frame = target.header.frame_id.strip().lstrip("/")
        if not source_frame:
            raise MissionError("TF GraspBox target has an empty source frame")
        if source_frame == target_frame:
            transformed = deepcopy(target)
        else:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=Duration(
                        seconds=self._float("grasp_box_tf_runtime_tf_timeout_sec")
                    ),
                )
                transformed = do_transform_pose_stamped(target, transform)
            except TransformException as exc:
                raise MissionError(
                    "TF GraspBox runtime target transform "
                    f"{source_frame} -> {target_frame} failed: {exc}"
                ) from exc
        result = Pose()
        result.position = deepcopy(transformed.pose.position)
        result.orientation = deepcopy(transformed.pose.orientation)
        return result

    def _make_tf_link8_target_poses(
        self,
        frozen_box_pose: PoseStamped,
        box_layer: int,
        model_label: str | None = None,
    ) -> tuple[PoseStamped, PoseStamped]:
        """Build both fixture-center Link8 targets in the frozen TF frame."""
        if self._string("direct_movel_target_mode").strip().lower() != (
            "camera_offset_box_orientation"
        ):
            raise MissionError(
                "TF GraspBox requires direct_movel_target_mode="
                "camera_offset_box_orientation"
            )
        targets = []
        for arm in ("left", "right"):
            center_pose, corrected_box_pose = (
                self._apply_box_frame_target_correction(
                    frozen_box_pose,
                    self._direct_movel_offset_parameter_name(
                        arm, box_layer, model_label
                    ),
                    self._joint123_target_correction_parameter_name(
                        arm, box_layer
                    ),
                )
            )
            link8_pose = self._make_camera_offset_box_orientation_pose(
                center_pose, corrected_box_pose, arm
            )
            compensated = self._make_direct_movel_pose(link8_pose, arm)
            link8_pose.pose = compensated
            targets.append(link8_pose)
        return targets[0], targets[1]

    def _apply_tf_execution_mode(
        self,
        goal_handle,
        left_target: PoseStamped,
        right_target: PoseStamped,
        dry_run: bool,
        box_layer: int,
        model_label: str | None = None,
    ) -> tuple[Pose, Pose, str]:
        """Move the waist as configured, then express frozen targets via TF."""
        execution_mode = self._string("box_grasp_execution_mode").strip().lower()
        if execution_mode not in (
            "arms_only",
            "joint1_then_arms",
            "joint1_then_arms_keep_position",
            "joint123_then_arms",
        ):
            raise MissionError(
                f"unsupported box_grasp_execution_mode for TF GraspBox: {execution_mode}"
            )
        movement_detail = "waist motion skipped in dry-run"
        if not dry_run:
            if execution_mode == "joint123_then_arms":
                self._move_body_joints_after_detection(
                    goal_handle, box_layer, model_label
                )
                movement_detail = "joint1/2/3 motion completed from measured /mcap/body feedback"
            elif execution_mode in (
                "joint1_then_arms",
                "joint1_then_arms_keep_position",
            ):
                self._move_body_joint1_after_detection(
                    goal_handle,
                    self._box_layer_joint1_approach_angle_deg(
                        box_layer, model_label
                    ),
                )
                movement_detail = "joint1 motion completed from measured /mcap/body feedback"
            else:
                movement_detail = "waist motion disabled (arms_only)"
        else:
            if execution_mode == "arms_only":
                movement_detail = "waist motion disabled in dry-run"
            else:
                movement_detail = (
                    f"{execution_mode} waist motion planned but skipped in dry-run"
                )
        left_moved = self._tf_target_pose_in_arm_base(left_target, "left")
        right_moved = self._tf_target_pose_in_arm_base(right_target, "right")
        return (
            left_moved,
            right_moved,
            f"execution={execution_mode}; target_frame=live_TF_arm_base; "
            f"frozen_frame={left_target.header.frame_id}; {movement_detail}",
        )

    def _measured_camera_pose_in_arm_base(
        self,
        camera_pose: PoseStamped,
        arm: str,
        detection_arm: str | None = None,
    ) -> PoseStamped:
        """Convert camera pose using live wrist-camera or legacy fixed extrinsics."""
        if self._boolean("camera_dynamic_link8_extrinsics_enabled"):
            selected_detection_arm = detection_arm or self._string(
                "camera_detection_arm"
            )
            return self._dynamic_camera_pose_in_arm_base(
                camera_pose, arm, selected_detection_arm
            )

        # Backward-compatible path for a camera rigidly mounted to the robot
        # body. It is intentionally not used for the wrist-camera profile.
        prefix = "left" if arm == "left" else "right"
        xyz = self._float_array(f"camera_{prefix}_base_xyz")
        q_ext = self._quaternion_from_rpy(
            *self._float_array(f"camera_{prefix}_base_rpy")
        )
        values = pose_to_array(camera_pose.pose)
        rotated = rotate_vector(tuple(values[:3]), q_ext)
        result = PoseStamped()
        result.header = deepcopy(camera_pose.header)
        result.header.frame_id = self._string(
            f"{prefix}_arm_base_frame"
        ).lstrip("/")
        result.header.stamp = self.get_clock().now().to_msg()
        result.pose.position.x = xyz[0] + rotated[0]
        result.pose.position.y = xyz[1] + rotated[1]
        result.pose.position.z = xyz[2] + rotated[2]
        q = quaternion_multiply(q_ext, tuple(values[3:]))
        norm = math.sqrt(sum(value * value for value in q))
        result.pose.orientation.x = q[0] / norm
        result.pose.orientation.y = q[1] / norm
        result.pose.orientation.z = q[2] / norm
        result.pose.orientation.w = q[3] / norm
        return result

    def _apply_box_frame_grasp_offset(
        self, camera_pose: PoseStamped, offset_name: str
    ) -> PoseStamped:
        """Apply an XYZ grasp offset expressed in the FoundationPose box frame."""
        offset_box = tuple(self._float_array(offset_name))
        box_quaternion = (
            float(camera_pose.pose.orientation.x),
            float(camera_pose.pose.orientation.y),
            float(camera_pose.pose.orientation.z),
            float(camera_pose.pose.orientation.w),
        )
        quaternion_norm = math.sqrt(
            sum(value * value for value in box_quaternion)
        )
        if quaternion_norm <= 1e-12 or not all(
            math.isfinite(value)
            for value in (*offset_box, *box_quaternion)
        ):
            raise MissionError(
                f"invalid FoundationPose orientation or offset for {offset_name}"
            )
        box_quaternion = tuple(
            value / quaternion_norm for value in box_quaternion
        )
        offset_camera = rotate_vector(offset_box, box_quaternion)
        result = PoseStamped()
        result.header = deepcopy(camera_pose.header)
        result.pose = deepcopy(camera_pose.pose)
        result.pose.position.x += offset_camera[0]
        result.pose.position.y += offset_camera[1]
        result.pose.position.z += offset_camera[2]
        return result

    def _apply_box_frame_target_correction(
        self,
        camera_pose: PoseStamped,
        offset_name: str,
        correction_name: str,
    ) -> tuple[PoseStamped, PoseStamped]:
        """Apply a box-frame offset and full Pose correction."""
        offset_box = tuple(self._float_array(offset_name))
        correction = tuple(self._float_array(correction_name))
        if len(offset_box) != 3 or len(correction) != 7:
            raise MissionError(
                f"invalid box-frame target correction dimensions for "
                f"{correction_name}"
            )
        box_q = BoxSupportMixin._normalize_quaternion(
            (
                float(camera_pose.pose.orientation.x),
                float(camera_pose.pose.orientation.y),
                float(camera_pose.pose.orientation.z),
                float(camera_pose.pose.orientation.w),
            )
        )
        correction_q = BoxSupportMixin._normalize_quaternion(
            tuple(float(value) for value in correction[3:])
        )
        box_translation = tuple(
            float(offset_box[index]) + float(correction[index])
            for index in range(3)
        )
        rotated_translation = rotate_vector(box_translation, box_q)

        target = PoseStamped()
        target.header = deepcopy(camera_pose.header)
        target.pose = deepcopy(camera_pose.pose)
        target.pose.position.x += rotated_translation[0]
        target.pose.position.y += rotated_translation[1]
        target.pose.position.z += rotated_translation[2]

        corrected_box = PoseStamped()
        corrected_box.header = deepcopy(camera_pose.header)
        corrected_box.pose = deepcopy(camera_pose.pose)
        corrected_box_q = BoxSupportMixin._normalize_quaternion(
            quaternion_multiply(box_q, correction_q)
        )
        corrected_box.pose.orientation.x = corrected_box_q[0]
        corrected_box.pose.orientation.y = corrected_box_q[1]
        corrected_box.pose.orientation.z = corrected_box_q[2]
        corrected_box.pose.orientation.w = corrected_box_q[3]
        return target, corrected_box

    @staticmethod
    def _direct_movel_offset_parameter_name(
        arm: str,
        box_layer: int,
        model_label: str | None = None,
    ) -> str:
        """Resolve the independent initial grasp offset for a model/layer."""
        if arm not in ("left", "right") or box_layer < 1 or box_layer > 4:
            raise MissionError(
                "arm must be left/right and box_layer must be in [1, 4]"
            )
        parameter_name = f"direct_movel_{arm}_offset_xyz"
        normalized_model = str(model_label or "").strip().lower()
        if normalized_model in ("bigbox", "smallbox"):
            parameter_name += f"_{normalized_model}_layer{box_layer}"
        return parameter_name

    @staticmethod
    def _joint123_target_correction_parameter_name(
        arm: str, box_layer: int
    ) -> str:
        if arm not in ("left", "right") or box_layer < 1 or box_layer > 4:
            raise MissionError("arm must be left/right and box_layer must be in [1, 4]")
        return f"joint123_layer{box_layer}_{arm}_target_correction_pose_box"

    def _arm_pose_in_execution_frame(self, arm_pose: PoseStamped) -> PoseStamped:
        target_frame = self._string("arm_execution_frame").lstrip("/")
        source_frame = arm_pose.header.frame_id.lstrip("/")
        if source_frame == target_frame:
            return arm_pose
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self._float("camera_tf_timeout_sec")),
            )
            result = do_transform_pose_stamped(arm_pose, transform)
        except TransformException as exc:
            raise MissionError(
                f"arm-base pose transform {source_frame} -> {target_frame} failed: {exc}"
            ) from exc
        result.header.frame_id = target_frame
        result.header.stamp = self.get_clock().now().to_msg()
        return result

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
            if (
                self._boolean("camera_measured_extrinsics_enabled")
                and self._boolean("camera_dynamic_link8_extrinsics_enabled")
            ):
                detection_arm = self._string("camera_detection_arm")
                transformed = self._measured_camera_pose_in_arm_base(
                    pose, detection_arm, detection_arm
                )
                transformed = self._arm_pose_in_execution_frame(transformed)
            else:
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

    def _current_link8_orientation(self, arm: str) -> tuple[float, float, float, float]:
        prefix = "left" if arm == "left" else "right"
        base_frame = self._string(f"{prefix}_arm_base_frame").lstrip("/")
        link8_frame = self._string(f"{prefix}_link8_frame").lstrip("/")
        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                link8_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self._float("camera_tf_timeout_sec")),
            )
        except TransformException as exc:
            raise MissionError(
                f"current {link8_frame} orientation lookup failed: {exc}"
            ) from exc
        q = (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        )
        norm = math.sqrt(sum(value * value for value in q))
        if norm <= 1e-12:
            raise MissionError(f"current {link8_frame} orientation is invalid")
        return tuple(value / norm for value in q)

    def _make_camera_offset_box_orientation_pose(
        self,
        arm_pose: PoseStamped,
        box_pose: PoseStamped,
        arm: str,
    ) -> PoseStamped:
        """Keep the camera-offset position and orient Link8 with the box."""
        result = PoseStamped()
        result.header = deepcopy(arm_pose.header)
        result.pose.position = deepcopy(arm_pose.pose.position)
        result.pose.orientation = (
            BoxSupportMixin._box_oriented_link8_orientation(
                self, box_pose, arm
            )
        )
        return result

    def _box_oriented_link8_orientation(
        self,
        box_pose: PoseStamped,
        arm: str,
    ):
        """Return Link8 orientation from box orientation and calibration."""
        prefix = "left" if arm == "left" else "right"
        box_q = tuple(pose_to_array(box_pose.pose)[3:])
        relative_q_values = self._float_array(
            f"direct_movel_{prefix}_box_to_link8_orientation"
        )
        relative_q_norm = math.sqrt(
            sum(value * value for value in relative_q_values)
        )
        if relative_q_norm <= 1e-12:
            raise MissionError(
                f"configured {prefix} box-to-Link8 orientation is invalid"
            )
        relative_q = tuple(
            value / relative_q_norm for value in relative_q_values
        )
        target_q = quaternion_multiply(box_q, relative_q)
        target_q_norm = math.sqrt(sum(value * value for value in target_q))
        orientation = Pose().orientation
        orientation.x = target_q[0] / target_q_norm
        orientation.y = target_q[1] / target_q_norm
        orientation.z = target_q[2] / target_q_norm
        orientation.w = target_q[3] / target_q_norm
        return orientation

    def _make_direct_movel_pose(
        self,
        box_pose: PoseStamped,
        arm: str,
    ) -> Pose:
        """Return a fixture-compensated Link8 target in its arm base frame."""
        prefix = "left" if arm == "left" else "right"
        source_q = (
            box_pose.pose.orientation.x,
            box_pose.pose.orientation.y,
            box_pose.pose.orientation.z,
            box_pose.pose.orientation.w,
        )
        norm = math.sqrt(sum(value * value for value in source_q))
        if norm <= 1e-12:
            raise MissionError(f"invalid {prefix} Link8 target orientation")
        target_q = tuple(value / norm for value in source_q)
        target_mode = self._string("direct_movel_target_mode").strip().lower()
        if target_mode == "camera_offset_box_orientation":
            # box_pose already contains the composed box-relative Link8
            # orientation. Its position is the requested fixture center.
            pass
        elif self._boolean("direct_movel_use_current_fixture_orientation"):
            target_q = self._current_link8_orientation(arm)
        else:
            fixed_q = self._float_array(
                f"direct_movel_{prefix}_fixed_link8_orientation"
            )
            fixed_norm = math.sqrt(sum(value * value for value in fixed_q))
            if fixed_norm <= 1e-12:
                raise MissionError(
                    f"configured {prefix} fixed Link8 orientation is invalid"
                )
            target_q = tuple(value / fixed_norm for value in fixed_q)
        fixture_compensation_enabled = self._boolean(
            "direct_movel_fixture_compensation_enabled"
        )
        if fixture_compensation_enabled:
            fixture_xyz = self._float_array(
                f"{prefix}_fixture_center_in_link8_xyz"
            )
            fixture_offset_base = rotate_vector(
                tuple(fixture_xyz), target_q
            )
        else:
            fixture_offset_base = (0.0, 0.0, 0.0)
        result = Pose()
        # The requested pose is the fixture center. Move Link8 to the
        # inverse offset so that Link8 + rotated(fixture_xyz) reaches it.
        result.position.x = box_pose.pose.position.x - fixture_offset_base[0]
        result.position.y = box_pose.pose.position.y - fixture_offset_base[1]
        result.position.z = box_pose.pose.position.z - fixture_offset_base[2]
        result.orientation.x = target_q[0]
        result.orientation.y = target_q[1]
        result.orientation.z = target_q[2]
        result.orientation.w = target_q[3]
        return result

    @staticmethod
    def _quaternion_conjugate(
        quaternion: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        return (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])

    @staticmethod
    def _quaternion_from_rotation_matrix(
        values: list[float],
    ) -> tuple[float, float, float, float]:
        """Convert a row-major proper rotation matrix to x,y,z,w."""
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = values
        trace = m00 + m11 + m22
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = (
                (m21 - m12) / scale,
                (m02 - m20) / scale,
                (m10 - m01) / scale,
                0.25 * scale,
            )
        elif m00 > m11 and m00 > m22:
            scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            quaternion = (
                0.25 * scale,
                (m01 + m10) / scale,
                (m02 + m20) / scale,
                (m21 - m12) / scale,
            )
        elif m11 > m22:
            scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            quaternion = (
                (m01 + m10) / scale,
                0.25 * scale,
                (m12 + m21) / scale,
                (m02 - m20) / scale,
            )
        else:
            scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            quaternion = (
                (m02 + m20) / scale,
                (m12 + m21) / scale,
                0.25 * scale,
                (m10 - m01) / scale,
            )
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 1e-12:
            raise MissionError("joint1-to-arm-base rotation is invalid")
        return tuple(value / norm for value in quaternion)

    def _reexpress_link8_target_after_joint1_rotation(
        self,
        target: Pose,
        arm: str,
        feedback_delta_rad: float,
    ) -> Pose:
        """Keep a frozen physical Link8 target while its arm base rotates."""
        prefix = "left" if arm == "left" else "right"
        mount_position = tuple(
            self._float_array(f"box_joint1_to_{prefix}_base_xyz")
        )
        mount_orientation = BoxSupportMixin._quaternion_from_rotation_matrix(
            self._float_array(f"box_joint1_to_{prefix}_base_rotation")
        )
        axis = self._float_array("box_joint1_axis_xyz")
        axis_norm = math.sqrt(sum(value * value for value in axis))
        if axis_norm <= 1e-12:
            raise MissionError("box_joint1_axis_xyz has zero length")
        signed_delta = (
            feedback_delta_rad
            * self._float("box_joint1_feedback_to_geometric_sign")
        )
        half_angle = 0.5 * signed_delta
        sine = math.sin(half_angle) / axis_norm
        joint_rotation = (
            axis[0] * sine,
            axis[1] * sine,
            axis[2] * sine,
            math.cos(half_angle),
        )
        moved_mount_position = rotate_vector(mount_position, joint_rotation)
        moved_mount_orientation = quaternion_multiply(
            joint_rotation, mount_orientation
        )

        target_position = (
            target.position.x,
            target.position.y,
            target.position.z,
        )
        target_in_joint1 = rotate_vector(target_position, mount_orientation)
        target_in_joint1 = tuple(
            mount_position[index] + target_in_joint1[index]
            for index in range(3)
        )
        target_relative_to_moved_base = tuple(
            target_in_joint1[index] - moved_mount_position[index]
            for index in range(3)
        )
        moved_base_inverse = BoxSupportMixin._quaternion_conjugate(
            moved_mount_orientation
        )
        moved_target_position = rotate_vector(
            target_relative_to_moved_base, moved_base_inverse
        )

        target_orientation = tuple(pose_to_array(target)[3:])
        old_to_new_base_orientation = quaternion_multiply(
            moved_base_inverse, mount_orientation
        )
        moved_target_orientation = quaternion_multiply(
            old_to_new_base_orientation, target_orientation
        )
        orientation_norm = math.sqrt(
            sum(value * value for value in moved_target_orientation)
        )

        result = Pose()
        result.position.x = moved_target_position[0]
        result.position.y = moved_target_position[1]
        result.position.z = moved_target_position[2]
        result.orientation.x = moved_target_orientation[0] / orientation_norm
        result.orientation.y = moved_target_orientation[1] / orientation_norm
        result.orientation.z = moved_target_orientation[2] / orientation_norm
        result.orientation.w = moved_target_orientation[3] / orientation_norm
        return result

    def _rotate_link8_orientation_after_joint1_rotation(
        self,
        target: Pose,
        arm: str,
        feedback_delta_rad: float,
    ) -> Pose:
        """Keep arm-base XYZ while preserving physical Link8 orientation."""
        fully_reexpressed = (
            BoxSupportMixin._reexpress_link8_target_after_joint1_rotation(
                self,
                target, arm, feedback_delta_rad
            )
        )
        result = Pose()
        result.position.x = target.position.x
        result.position.y = target.position.y
        result.position.z = target.position.z
        result.orientation = fully_reexpressed.orientation
        return result

    @staticmethod
    def _normalize_quaternion(quaternion):
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 1e-12:
            raise MissionError("rotation quaternion has zero norm")
        return tuple(value / norm for value in quaternion)

    @staticmethod
    def _compose_transform(lhs, rhs):
        lhs_position, lhs_orientation = lhs
        rhs_position, rhs_orientation = rhs
        orientation = quaternion_multiply(lhs_orientation, rhs_orientation)
        return (
            tuple(
                lhs_position[index]
                + rotate_vector(rhs_position, lhs_orientation)[index]
                for index in range(3)
            ),
            BoxSupportMixin._normalize_quaternion(orientation),
        )

    @staticmethod
    def _inverse_transform(transform):
        position, orientation = transform
        inverse_orientation = BoxSupportMixin._quaternion_conjugate(orientation)
        rotated = rotate_vector(position, inverse_orientation)
        return (
            tuple(-value for value in rotated),
            inverse_orientation,
        )

    @staticmethod
    def _rotation_transform(axis, angle):
        norm = math.sqrt(sum(value * value for value in axis))
        if norm <= 1e-12:
            raise MissionError("joint rotation axis has zero length")
        half_angle = angle * 0.5
        sine = math.sin(half_angle) / norm
        return (
            (0.0, 0.0, 0.0),
            BoxSupportMixin._normalize_quaternion(
                (
                    axis[0] * sine,
                    axis[1] * sine,
                    axis[2] * sine,
                    math.cos(half_angle),
                )
            ),
        )

    def _configured_joint_transform(self, xyz_name, rotation_name, axis_name, angle):
        origin = (
            tuple(self._float_array(xyz_name)),
            BoxSupportMixin._quaternion_from_rotation_matrix(
                self._float_array(rotation_name)
            ),
        )
        return BoxSupportMixin._compose_transform(
            origin,
            BoxSupportMixin._rotation_transform(
                self._float_array(axis_name), angle
            ),
        )

    def _configured_rpy_transform(self, xyz_name, rpy_name):
        """Build a fixed URDF origin transform from XYZ and RPY parameters."""
        return (
            tuple(self._float_array(xyz_name)),
            BoxSupportMixin._normalize_quaternion(
                BoxSupportMixin._quaternion_from_rpy(
                    *self._float_array(rpy_name)
                )
            ),
        )

    def _joint123_arm_base_transform(self, arm: str, angles_rad):
        """Return the URDF root->arm-base transform for waist J1/J2/J3."""
        joint1_angle, joint2_angle, joint3_angle = angles_rad
        joint1_origin = BoxSupportMixin._configured_rpy_transform(
            self,
            "box_waist1_origin_xyz", "box_waist1_origin_rpy"
        )
        waist1_rotation = BoxSupportMixin._rotation_transform(
            self._float_array("box_joint1_axis_xyz"),
            joint1_angle
            * self._float("box_joint1_feedback_to_geometric_sign"),
        )
        waist1_to_waist2 = BoxSupportMixin._compose_transform(
            BoxSupportMixin._configured_rpy_transform(
                self,
                "box_waist2_origin_xyz", "box_waist2_origin_rpy"
            ),
            BoxSupportMixin._rotation_transform(
                self._float_array("box_joint2_axis_xyz"),
                joint2_angle
                * self._float("box_joint2_feedback_to_urdf_axis_sign"),
            ),
        )
        waist2_to_waist3 = BoxSupportMixin._compose_transform(
            BoxSupportMixin._configured_rpy_transform(
                self,
                "box_waist3_origin_xyz", "box_waist3_origin_rpy"
            ),
            BoxSupportMixin._rotation_transform(
                self._float_array("box_joint3_axis_xyz"),
                joint3_angle
                * self._float("box_joint3_feedback_to_urdf_axis_sign"),
            ),
        )
        prefix = "left" if arm == "left" else "right"
        waist3_to_chest = BoxSupportMixin._configured_rpy_transform(
            self,
            "box_waist3_to_chest_xyz", "box_waist3_to_chest_rpy"
        )
        chest_to_arm_base = BoxSupportMixin._configured_rpy_transform(
            self,
            f"box_chest_to_{prefix}_arm_base_xyz",
            f"box_chest_to_{prefix}_arm_base_rpy",
        )
        return BoxSupportMixin._compose_transform(
            joint1_origin,
            BoxSupportMixin._compose_transform(
                waist1_rotation,
                BoxSupportMixin._compose_transform(
                    waist1_to_waist2,
                    BoxSupportMixin._compose_transform(
                        waist2_to_waist3,
                        BoxSupportMixin._compose_transform(
                            waist3_to_chest, chest_to_arm_base
                        ),
                    ),
                ),
            ),
        )

    def _reexpress_link8_target_after_joint123_motion(
        self,
        target: Pose,
        arm: str,
        detection_angles_rad,
        target_angles_rad,
    ) -> Pose:
        """Re-express a frozen physical target after J1/J2/J3 move."""
        detected_base = BoxSupportMixin._joint123_arm_base_transform(
            self,
            arm, detection_angles_rad
        )
        target_base = BoxSupportMixin._joint123_arm_base_transform(
            self, arm, target_angles_rad
        )
        relative_base = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(target_base), detected_base
        )
        moved = BoxSupportMixin._compose_transform(
            relative_base,
            (
                (target.position.x, target.position.y, target.position.z),
                BoxSupportMixin._normalize_quaternion(
                    (
                        target.orientation.x,
                        target.orientation.y,
                        target.orientation.z,
                        target.orientation.w,
                    )
                ),
            ),
        )
        result = Pose()
        result.position.x, result.position.y, result.position.z = moved[0]
        result.orientation.x, result.orientation.y = moved[1][0:2]
        result.orientation.z, result.orientation.w = moved[1][2:4]
        return result

    def _joint1_feedback_snapshot(self):
        with self.joint_state_lock:
            return (
                self.latest_body_joint1_position,
                self.latest_body_joint1_velocity,
                self.latest_body_joint1_state_time,
                self.latest_body_joint1_state_sequence,
            )

    def _body_feedback_snapshot(self):
        with self.joint_state_lock:
            names = [
                self._string(f"box_joint{index}_name")
                for index in range(1, 5)
            ]
            return (
                [self.latest_body_joint_positions.get(name) for name in names],
                [self.latest_body_joint_velocities.get(name) for name in names],
                self.latest_body_state_time,
                self.latest_body_state_sequence,
            )

    def _wait_for_body_joints_target(
        self,
        goal_handle,
        target_angles_rad,
        *,
        sequence_after=-1,
        timeout_parameter="box_joint1_wait_timeout_sec",
    ):
        timeout_sec = self._float(timeout_parameter)
        position_tolerance = self._float("box_joint1_position_tolerance_rad")
        velocity_tolerance = self._float("box_joint1_velocity_tolerance_rad_sec")
        max_age_sec = self._float("box_joint1_feedback_max_age_sec")
        required_stable = self._integer("box_joint1_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequence = -1
        latest_positions = [None] * 4
        latest_velocities = [None] * 4
        latest_age = math.inf
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for body joints")
            positions, velocities, state_time, sequence = self._body_feedback_snapshot()
            latest_positions, latest_velocities = positions, velocities
            latest_age = time.monotonic() - state_time
            if sequence != last_sequence:
                last_sequence = sequence
                sample_matches = (
                    sequence > sequence_after
                    and latest_age <= max_age_sec
                    and all(
                        position is not None
                        and velocity is not None
                        and abs(position - target) <= position_tolerance
                        and abs(velocity) <= velocity_tolerance
                        for position, velocity, target in zip(
                            positions[: len(target_angles_rad)],
                            velocities[: len(target_angles_rad)],
                            target_angles_rad,
                        )
                    )
                )
                stable_samples = stable_samples + 1 if sample_matches else 0
                if stable_samples >= required_stable:
                    return [float(value) for value in positions]
            time.sleep(0.02)
        raise MissionError(
            "body joints did not reach requested targets: "
            f"targets={target_angles_rad}, measured={latest_positions}, "
            f"velocities={latest_velocities}, feedback_age_sec={latest_age:.3f}, "
            f"timeout_sec={timeout_sec:.1f}"
        )

    def _wait_for_fresh_body_feedback(
        self,
        goal_handle,
        *,
        sequence_after: int = -1,
    ) -> tuple[list[float], list[float], int]:
        """Return a fresh complete four-joint body sample.

        The detection posture is not assumed to be the configured zero pose.
        A replacement robot may be left at any valid Joint1-4 state when a
        GraspBox goal starts, so motion planning must use the measured state
        and only use the configured angles as the subsequent target.
        """
        timeout_sec = self._float("box_joint1_wait_timeout_sec")
        max_age_sec = self._float("box_joint1_feedback_max_age_sec")
        deadline = time.monotonic() + timeout_sec
        latest_positions = [None] * 4
        latest_velocities = [None] * 4
        latest_age = math.inf
        latest_sequence = -1
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for fresh body feedback")
            positions, velocities, state_time, sequence = (
                self._body_feedback_snapshot()
            )
            latest_positions = positions
            latest_velocities = velocities
            latest_sequence = sequence
            latest_age = time.monotonic() - state_time
            if (
                sequence > sequence_after
                and latest_age <= max_age_sec
                and len(positions) >= 4
                and len(velocities) >= 4
                and all(value is not None for value in positions[:4])
                and all(value is not None for value in velocities[:4])
            ):
                return (
                    [float(value) for value in positions[:4]],
                    [float(value) for value in velocities[:4]],
                    sequence,
                )
            time.sleep(0.02)
        raise MissionError(
            "fresh /mcap/body feedback did not contain all four joints: "
            f"measured={latest_positions}, velocities={latest_velocities}, "
            f"sequence={latest_sequence}, feedback_age_sec={latest_age:.3f}, "
            f"timeout_sec={timeout_sec:.1f}"
        )

    def _wait_for_post_arm_joint_targets(
        self,
        goal_handle,
        left_target_rad,
        right_target_rad,
        sequence_after,
        # Legacy post-arm feedback parameters omit the ``_movej`` segment,
        # while the new pre-target block keeps it in the parameter names.
        parameter_prefix="box_post_arm",
        description="post-grasp arm MoveJ",
        active_arms=("left", "right"),
    ) -> None:
        """Require selected arm feedback streams to reach seven-joint targets."""
        tolerance = self._float(
            f"{parameter_prefix}_position_tolerance_rad"
        )
        velocity_tolerance = self._float(
            f"{parameter_prefix}_velocity_tolerance_rad_sec"
        )
        max_age = self._float(f"{parameter_prefix}_feedback_max_age_sec")
        timeout_parameter = (
            "box_post_arm_movej_timeout_sec"
            if parameter_prefix == "box_post_arm"
            else f"{parameter_prefix}_timeout_sec"
        )
        timeout_sec = self._float(timeout_parameter)
        stable_required = self._integer(f"{parameter_prefix}_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequences = {arm: -1 for arm in active_arms}
        latest_detail = "no feedback"
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, f"while verifying {description}"
            )
            now = time.monotonic()
            with self.joint_state_lock:
                sequences = dict(self.latest_slave_arm_state_sequences)
                positions = {
                    arm: list(values)
                    for arm, values in self.latest_slave_arm_positions.items()
                }
                velocities = {
                    arm: list(values)
                    for arm, values in self.latest_slave_arm_velocities.items()
                }
                times = dict(self.latest_slave_arm_state_times)
            if sequences == last_sequences:
                time.sleep(0.02)
                continue
            last_sequences = sequences
            matches = True
            error_detail = []
            targets = {"left": left_target_rad, "right": right_target_rad}
            for arm in active_arms:
                target = targets[arm]
                measured = positions.get(arm, [])
                measured_velocity = velocities.get(arm, [])
                age = now - times.get(arm, 0.0)
                if (
                    sequences.get(arm, 0) <= sequence_after.get(arm, -1)
                    or len(measured) < 7
                    or len(measured_velocity) < 7
                    or age > max_age
                ):
                    matches = False
                    error_detail.append(
                        f"{arm}=missing_or_stale(seq={sequences.get(arm, 0)}, age={age:.3f})"
                    )
                    continue
                position_error = max(
                    abs(measured[index] - target[index]) for index in range(7)
                )
                velocity_error = max(
                    abs(measured_velocity[index]) for index in range(7)
                )
                error_detail.append(
                    f"{arm}=pos_err={position_error:.4f}, vel={velocity_error:.4f}"
                )
                if position_error > tolerance or velocity_error > velocity_tolerance:
                    matches = False
            latest_detail = "; ".join(error_detail)
            stable_samples = stable_samples + 1 if matches else 0
            if stable_samples >= stable_required:
                return
            time.sleep(0.02)
        raise MissionError(
            f"{description} targets were not confirmed: "
            f"{latest_detail}; timeout_sec={timeout_sec:.1f}"
        )

    @staticmethod
    def _parse_string_command_response(response, description: str) -> None:
        try:
            response_data = json.loads(str(response.data).strip())
        except (TypeError, ValueError) as exc:
            raise MissionError(
                f"invalid {description} service response: {response.data!r}"
            ) from exc
        if not response_data.get("receive_state", False):
            raise MissionError(
                f"{description} command was not accepted: {response.data}"
            )

    def _post_arm_movej_targets(self):
        units_per_degree = self._float(
            "box_post_arm_movej_command_units_per_degree"
        )
        left_units = [
            int(value)
            for value in self._float_array("box_post_arm_movej_left_joint_units")
        ]
        right_units = [
            int(value)
            for value in self._float_array("box_post_arm_movej_right_joint_units")
        ]
        target_left = [
            math.radians(float(value) / units_per_degree) for value in left_units
        ]
        target_right = [
            math.radians(float(value) / units_per_degree) for value in right_units
        ]
        return left_units, right_units, target_left, target_right

    def _configured_dual_arm_movej_targets(self, prefix: str):
        """Read a dual-arm MoveJ configuration and convert units to radians."""
        units_per_degree = self._float(
            f"{prefix}_command_units_per_degree"
        )
        left_units = [
            int(value)
            for value in self._float_array(f"{prefix}_left_joint_units")
        ]
        right_units = [
            int(value)
            for value in self._float_array(f"{prefix}_right_joint_units")
        ]
        target_left = [
            math.radians(float(value) / units_per_degree)
            for value in left_units
        ]
        target_right = [
            math.radians(float(value) / units_per_degree)
            for value in right_units
        ]
        return left_units, right_units, target_left, target_right

    def _current_dual_arm_prepare_targets(self, active_arms=("left", "right")):
        """Build stage-one targets from fresh seven-joint arm feedback."""
        units_per_degree = self._float(
            "box_pre_target_arm_movej_command_units_per_degree"
        )
        joint2_units = self._integer(
            "box_pre_target_arm_movej_stage1_joint2_units"
        )
        max_age = self._float("box_pre_target_arm_movej_feedback_max_age_sec")
        now = time.monotonic()
        units_by_arm = {}
        targets_by_arm = {}
        detail_by_arm = {}
        with self.joint_state_lock:
            for arm in active_arms:
                positions = list(self.latest_slave_arm_positions.get(arm, []))
                sequence = self.latest_slave_arm_state_sequences.get(arm, 0)
                age = now - self.latest_slave_arm_state_times.get(arm, 0.0)
                if sequence <= 0 or len(positions) < 7 or age > max_age:
                    raise MissionError(
                        "fresh arm feedback required for prepare stage 1: "
                        f"{arm}=sequence={sequence}, joints={len(positions)}, "
                        f"feedback_age_sec={age:.3f}, timeout_sec={max_age:.1f}"
                    )
                if not all(math.isfinite(value) for value in positions[:7]):
                    raise MissionError(
                        f"arm feedback for prepare stage 1 contains invalid values: {arm}"
                    )
                units = [
                    int(round(math.degrees(value) * units_per_degree))
                    for value in positions[:7]
                ]
                units[1] = joint2_units
                units_by_arm[arm] = units
                targets_by_arm[arm] = [
                    math.radians(float(value) / units_per_degree)
                    for value in units
                ]
                detail_by_arm[arm] = (
                    f"{arm}=seq={sequence}, age={age:.3f}, "
                    f"joint_units={units}"
                )
        return units_by_arm, targets_by_arm, detail_by_arm

    def _current_step4_movej_targets(self, active_arms=("left", "right")):
        """Preserve live arm joints while overriding Joint2 for Step4."""
        units_per_degree = self._float(
            "box_post_movel_step4_movej_command_units_per_degree"
        )
        joint2_units = self._integer("box_post_movel_step4_movej_joint2_units")
        max_age = self._float(
            "box_post_movel_step4_movej_feedback_max_age_sec"
        )
        now = time.monotonic()
        units_by_arm = {}
        targets_by_arm = {}
        detail_by_arm = {}
        with self.joint_state_lock:
            for arm in active_arms:
                positions = list(self.latest_slave_arm_positions.get(arm, []))
                sequence = self.latest_slave_arm_state_sequences.get(arm, 0)
                age = now - self.latest_slave_arm_state_times.get(arm, 0.0)
                if sequence <= 0 or len(positions) < 7 or age > max_age:
                    raise MissionError(
                        "fresh arm feedback required for Step4 MoveJ: "
                        f"{arm}=sequence={sequence}, joints={len(positions)}, "
                        f"feedback_age_sec={age:.3f}, timeout_sec={max_age:.1f}"
                    )
                if not all(math.isfinite(value) for value in positions[:7]):
                    raise MissionError(
                        f"arm feedback for Step4 MoveJ contains invalid values: {arm}"
                    )
                units = [
                    int(round(math.degrees(value) * units_per_degree))
                    for value in positions[:7]
                ]
                units[1] = joint2_units
                units_by_arm[arm] = units
                targets_by_arm[arm] = [
                    math.radians(float(value) / units_per_degree)
                    for value in units
                ]
                detail_by_arm[arm] = (
                    f"{arm}=seq={sequence}, age={age:.3f}, "
                    f"joint_units={units}"
                )
        return units_by_arm, targets_by_arm, detail_by_arm

    def _execute_dual_arm_movej_targets(
        self,
        goal_handle,
        dry_run: bool,
        prefix: str,
        left_units,
        right_units,
        left_target,
        right_target,
        detail_prefix: str,
        feedback_prefix: str,
        feedback_move_stage: str,
        feedback_wait_stage: str,
        feedback_reached_stage: str,
        description: str,
        active_arms=("left", "right"),
        velocity_parameter=None,
    ) -> str:
        """Send selected arm MoveJ targets and verify their feedback."""
        active_arms = tuple(active_arms)
        if not active_arms or any(arm not in ("left", "right") for arm in active_arms):
            raise MissionError(f"invalid active arms for MoveJ: {active_arms}")
        units_by_arm = {"left": left_units, "right": right_units}
        targets_by_arm = {"left": left_target, "right": right_target}
        devices_by_arm = {
            arm: self._integer(f"{prefix}_{arm}_device") for arm in active_arms
        }
        detail = f"{detail_prefix}: " + ", ".join(
            f"{arm}_device={devices_by_arm[arm]}, "
            f"{arm}_joint_units={units_by_arm[arm]}"
            for arm in active_arms
        )
        self._publish_box_grasp_feedback(
            goal_handle, f"{feedback_prefix}_TARGETS", detail
        )
        if dry_run:
            return f"{detail}; skipped in dry-run"

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        if velocity_parameter is None:
            velocity_parameter = (
                f"{prefix}_velocity"
                if prefix == "box_post_movel_step4_movej"
                else "box_preparation_movej_velocity"
            )
        common = {
            "command": "movej",
            "v": self._integer(velocity_parameter),
            "r": self._integer(f"{prefix}_blend_radius"),
            "trajectory_connect": self._integer(
                f"{prefix}_trajectory_connect"
            ),
        }
        requests = {}
        for arm in active_arms:
            payload = {
                "device": devices_by_arm[arm],
                "payload": {**common, "joint": units_by_arm[arm]},
            }
            request = StringCmd.Request()
            request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
            requests[arm] = request
        with self.joint_state_lock:
            sequence_before = dict(self.latest_slave_arm_state_sequences)
        self._publish_box_grasp_feedback(
            goal_handle,
            feedback_move_stage,
            f"sending {description} for {', '.join(active_arms)} arm(s)",
        )
        futures = {
            arm: self.body_command_client.call_async(requests[arm])
            for arm in active_arms
        }
        for arm in active_arms:
            response = self._wait_future(
                futures[arm],
                goal_handle,
                f"calling {description} {arm} arm MoveJ",
                self._float("dependency_wait_timeout_sec"),
                cancel_local_future=False,
            )
            self._parse_string_command_response(
                response, f"{description} {arm} arm MoveJ"
            )
        self._publish_box_grasp_feedback(
            goal_handle,
            feedback_wait_stage,
            f"{description} commands accepted; waiting for fresh position and zero-velocity feedback for {', '.join(active_arms)} arm(s)",
        )
        self._wait_for_post_arm_joint_targets(
            goal_handle,
            left_target,
            right_target,
            sequence_before,
            parameter_prefix=prefix,
            description=description,
            active_arms=active_arms,
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            feedback_reached_stage,
            f"{', '.join(active_arms)} arm(s) reached the {description} targets with stable feedback",
        )
        return f"{detail}; arm_feedback=confirmed"

    def _execute_configured_dual_arm_movej(
        self,
        goal_handle,
        dry_run: bool,
        prefix: str,
        detail_prefix: str,
        feedback_prefix: str,
        feedback_move_stage: str,
        feedback_wait_stage: str,
        feedback_reached_stage: str,
        description: str,
        active_arms=("left", "right"),
    ) -> str:
        """Send two configured arm MoveJ commands concurrently and verify them."""
        left_units, right_units, left_target, right_target = (
            self._configured_dual_arm_movej_targets(prefix)
        )
        return self._execute_dual_arm_movej_targets(
            goal_handle,
            dry_run,
            prefix,
            left_units,
            right_units,
            left_target,
            right_target,
            detail_prefix,
            feedback_prefix,
            feedback_move_stage,
            feedback_wait_stage,
            feedback_reached_stage,
            description,
            active_arms=active_arms,
        )

    def _execute_pre_target_arm_movej(
        self, goal_handle, dry_run: bool, right_arm_only: bool = False
    ) -> str:
        if not self._boolean("box_pre_target_arm_movej_enabled"):
            return "pre_target_arm_movej=disabled"
        if not self._boolean("box_pre_target_arm_movej_two_stage_enabled"):
            return self._execute_configured_dual_arm_movej(
                goal_handle,
                dry_run,
                "box_pre_target_arm_movej",
                "pre-target dual arm MoveJ",
                "PRE_TARGET_ARM_MOVEJ",
                "MOVING_PRE_TARGET_ARM_JOINTS",
                "WAITING_FOR_PRE_TARGET_ARM_JOINTS",
                "PRE_TARGET_ARM_JOINTS_REACHED",
                "pre-target dual arm MoveJ",
                active_arms=("right",) if right_arm_only else ("left", "right"),
            )
        return self._execute_two_stage_prepare_arm_movej(
            goal_handle,
            dry_run,
            "pre-target",
            "PRE_TARGET_ARM_MOVEJ",
            "pre-target dual arm MoveJ",
            right_arm_only=right_arm_only,
        )

    def _execute_pre_detection_arm_movej(
        self, goal_handle, dry_run: bool, right_arm_only: bool = False
    ) -> str:
        if not self._boolean("box_pre_detection_arm_movej_enabled"):
            return "pre_detection_arm_movej=disabled"
        if not self._boolean("box_pre_target_arm_movej_two_stage_enabled"):
            return self._execute_configured_dual_arm_movej(
                goal_handle,
                dry_run,
                "box_pre_target_arm_movej",
                "pre-detection dual arm MoveJ",
                "PRE_DETECTION_ARM_MOVEJ",
                "MOVING_PRE_DETECTION_ARM_JOINTS",
                "WAITING_FOR_PRE_DETECTION_ARM_JOINTS",
                "PRE_DETECTION_ARM_JOINTS_REACHED",
                "pre-detection dual arm MoveJ",
                active_arms=("right",) if right_arm_only else ("left", "right"),
            )
        return self._execute_two_stage_prepare_arm_movej(
            goal_handle,
            dry_run,
            "pre-detection",
            "PRE_DETECTION_ARM_MOVEJ",
            "pre-detection dual arm MoveJ",
            right_arm_only=right_arm_only,
        )

    def _execute_two_stage_prepare_arm_movej(
        self,
        goal_handle,
        dry_run: bool,
        phase_label: str,
        feedback_prefix: str,
        description: str,
        right_arm_only: bool = False,
    ) -> str:
        """Prepare selected arms in two stages while preserving live joints first."""
        prefix = "box_pre_target_arm_movej"
        active_arms = ("right",) if right_arm_only else ("left", "right")
        if dry_run:
            stage1_detail = (
                f"{phase_label} stage1 dynamic {', '.join(active_arms)} arm MoveJ: "
                "live joint feedback read at execution time; skipped in dry-run"
            )
            stage2_right_units = [
                int(value)
                for value in self._float_array(f"{prefix}_right_joint_units")
            ]
            stage2_detail = f"right_joint_units={stage2_right_units}"
            if not right_arm_only:
                stage2_units = [
                    int(value)
                    for value in self._float_array(f"{prefix}_left_joint_units")
                ]
                stage2_detail = (
                    f"left_joint_units={stage2_units}, {stage2_detail}"
                )
            return (
                f"{stage1_detail}; {phase_label} stage2 {', '.join(active_arms)} arm MoveJ: "
                f"{stage2_detail}; skipped in dry-run"
            )

        units_by_arm, targets_by_arm, feedback_detail = (
            self._current_dual_arm_prepare_targets(active_arms=active_arms)
        )
        stage1_detail = (
            f"{phase_label} stage1 dynamic {', '.join(active_arms)} arm MoveJ: "
            f"{'; '.join(feedback_detail[arm] for arm in active_arms)}"
        )
        stage1_result = self._execute_dual_arm_movej_targets(
            goal_handle,
            False,
            prefix,
            units_by_arm.get("left", []),
            units_by_arm["right"],
            targets_by_arm.get("left", []),
            targets_by_arm["right"],
            stage1_detail,
            f"{feedback_prefix}_STAGE1",
            f"MOVING_{feedback_prefix}_STAGE1",
            f"WAITING_FOR_{feedback_prefix}_STAGE1",
            f"{feedback_prefix}_STAGE1_REACHED",
            f"{description} stage1",
            active_arms=active_arms,
        )
        left_units, right_units, left_target, right_target = (
            self._configured_dual_arm_movej_targets(prefix)
        )
        stage2_result = self._execute_dual_arm_movej_targets(
            goal_handle,
            False,
            prefix,
            left_units,
            right_units,
            left_target,
            right_target,
            f"{phase_label} stage2 dual arm MoveJ",
            f"{feedback_prefix}_STAGE2",
            f"MOVING_{feedback_prefix}_STAGE2",
            f"WAITING_FOR_{feedback_prefix}_STAGE2",
            f"{feedback_prefix}_STAGE2_REACHED",
            f"{description} stage2",
            active_arms=active_arms,
        )
        return f"{stage1_result}; {stage2_result}"

    def _execute_pre_detection_right_intermediate_movej(
        self, goal_handle, dry_run: bool, final_units: list[int]
    ) -> str:
        """Move only detection joint1 first while preserving live joints 2-7."""
        prefix = "box_pre_detection_right_movej"
        units_per_degree = self._float(f"{prefix}_command_units_per_degree")
        with self.joint_state_lock:
            positions = list(self.latest_slave_arm_positions.get("right", []))
            sequence_before = self.latest_slave_arm_state_sequences.get("right", 0)
            pose_sequence_before = self.latest_slave_arm_pose_sequences.get("right", 0)
            state_time = self.latest_slave_arm_state_times.get("right", 0.0)
        age = time.monotonic() - state_time
        max_age = self._float(f"{prefix}_feedback_max_age_sec")
        if not dry_run and (
            sequence_before <= 0 or len(positions) < 7 or age > max_age
        ):
            raise MissionError(
                "fresh right-arm feedback required before detection intermediate MoveJ: "
                f"sequence={sequence_before}, joints={len(positions)}, "
                f"feedback_age_sec={age:.3f}, timeout_sec={max_age:.1f}"
            )
        units = [int(value) for value in final_units]
        if not dry_run:
            units[1:] = [
                int(round(math.degrees(value) * units_per_degree))
                for value in positions[1:7]
            ]
        target = [
            math.radians(float(value) / units_per_degree) for value in units
        ]
        detail = (
            "pre-detection right arm intermediate MoveJ: "
            f"device={self._integer(f'{prefix}_device')}, joint_units={units}"
        )
        self._publish_box_grasp_feedback(
            goal_handle, "PRE_DETECTION_RIGHT_INTERMEDIATE_TARGETS", detail
        )
        if dry_run:
            return f"{detail}; skipped in dry-run"

        self._wait_for_service(
            self.body_command_client,
            self._string("box_joint1_command_service_name"),
            goal_handle,
        )
        payload = {
            "device": self._integer(f"{prefix}_device"),
            "payload": {
                "command": "movej",
                "joint": units,
                "v": self._integer("box_preparation_movej_velocity"),
                "r": self._integer(f"{prefix}_blend_radius"),
                "trajectory_connect": self._integer(
                    f"{prefix}_trajectory_connect"
                ),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_RIGHT_ARM_TO_DETECTION_INTERMEDIATE_POSE",
            "sending the right-arm intermediate detection MoveJ command",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            "calling pre-detection right arm intermediate MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        self._parse_string_command_response(
            response, "pre-detection right arm intermediate MoveJ"
        )
        tolerance = self._float(f"{prefix}_position_tolerance_rad")
        velocity_tolerance = self._float(
            f"{prefix}_velocity_tolerance_rad_sec"
        )
        stable_required = self._integer(f"{prefix}_stable_samples")
        deadline = time.monotonic() + self._float(f"{prefix}_timeout_sec")
        stable_samples = 0
        latest_detail = "no right-arm feedback"
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle,
                "while verifying the right-arm intermediate detection MoveJ",
            )
            now = time.monotonic()
            with self.joint_state_lock:
                sequence = self.latest_slave_arm_state_sequences.get("right", 0)
                measured = list(self.latest_slave_arm_positions.get("right", []))
                velocity = list(self.latest_slave_arm_velocities.get("right", []))
                state_time = self.latest_slave_arm_state_times.get("right", 0.0)
                pose_sequence = self.latest_slave_arm_pose_sequences.get("right", 0)
            age = now - state_time
            position_error = (
                max(abs(measured[index] - target[index]) for index in range(7))
                if len(measured) >= 7
                else float("inf")
            )
            velocity_error = (
                max(abs(value) for value in velocity[:7])
                if len(velocity) >= 7
                else float("inf")
            )
            matches = (
                sequence > sequence_before
                and len(measured) >= 7
                and len(velocity) >= 7
                and age <= max_age
                and position_error <= tolerance
                and velocity_error <= velocity_tolerance
                and pose_sequence > pose_sequence_before
            )
            latest_detail = (
                f"seq={sequence}, pose_seq={pose_sequence}, age={age:.3f}, "
                f"pos_err={position_error:.4f}, vel={velocity_error:.4f}"
            )
            stable_samples = stable_samples + 1 if matches else 0
            if stable_samples >= stable_required:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "RIGHT_ARM_DETECTION_INTERMEDIATE_REACHED",
                    "right arm reached the intermediate detection pose",
                )
                return f"{detail}; arm_feedback=confirmed"
            time.sleep(0.02)
        raise MissionError(
            "right arm did not reach the intermediate detection pose: "
            f"{latest_detail}; timeout_sec={self._float(f'{prefix}_timeout_sec'):.1f}"
        )

    def _box_layer_pre_detection_right_movej_joint_units(
        self, box_layer: int, model_label: str | None = None
    ) -> list[int]:
        """Return the configured right-arm detection joints for a layer.

        ``model_label`` is supplied by the regular GraspBox mission so
        smallbox and bigbox can have independent observation poses.  The
        original generic table remains a fallback for DragBox and other
        callers that do not select a model explicitly.
        """
        if box_layer < 1 or box_layer > 4:
            raise MissionError("box_layer must be in [1, 4]")
        configured = self._boolean_array(
            "box_layer_pre_detection_right_movej_configured"
        )
        if len(configured) != 4:
            raise MissionError(
                "box_layer_pre_detection_right_movej_configured must contain "
                "four values"
            )
        if not configured[box_layer - 1]:
            raise MissionError(
                f"box_layer {box_layer} right-arm detection pose is not configured yet"
            )
        parameter_name = "box_layer_pre_detection_right_movej_joint_units"
        if model_label:
            normalized_model = str(model_label).strip().lower()
            if normalized_model in ("bigbox", "smallbox"):
                parameter_name = (
                    "box_layer_pre_detection_right_movej_joint_units_"
                    f"{normalized_model}"
                )
        values = self._float_array(parameter_name)
        if len(values) != 28:
            raise MissionError(
                f"{parameter_name} must contain 28 values "
                "(four layers x seven joints)"
            )
        start = (box_layer - 1) * 7
        selected = values[start : start + 7]
        if not all(math.isfinite(value) for value in selected):
            raise MissionError(
                f"box_layer {box_layer} right-arm detection pose contains invalid values"
            )
        return [int(round(value)) for value in selected]

    def _execute_pre_detection_right_movej(
        self,
        goal_handle,
        dry_run: bool,
        box_layer: int,
        model_label: str | None = None,
    ) -> str:
        """Move the right wrist camera to its fixed detection configuration."""
        if not self._boolean("box_pre_detection_right_movej_enabled"):
            return "pre_detection_right_movej=disabled"
        prefix = "box_pre_detection_right_movej"
        units = self._box_layer_pre_detection_right_movej_joint_units(
            box_layer, model_label
        )
        intermediate_detail = self._execute_pre_detection_right_intermediate_movej(
            goal_handle, dry_run, units
        )
        units_per_degree = self._float(f"{prefix}_command_units_per_degree")
        target = [
            math.radians(float(value) / units_per_degree) for value in units
        ]
        device = self._integer(f"{prefix}_device")
        detail = (
            "pre-detection right arm MoveJ: "
            f"device={device}, joint_units={units}"
        )
        self._publish_box_grasp_feedback(
            goal_handle, "PRE_DETECTION_RIGHT_MOVEJ_TARGETS", detail
        )
        if dry_run:
            return f"{intermediate_detail}; {detail}; skipped in dry-run"

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        with self.joint_state_lock:
            sequence_before = self.latest_slave_arm_state_sequences.get("right", 0)
            pose_sequence_before = self.latest_slave_arm_pose_sequences.get("right", 0)
        payload = {
            "device": device,
            "payload": {
                "command": "movej",
                "joint": units,
                "v": self._integer("box_preparation_movej_velocity"),
                "r": self._integer(f"{prefix}_blend_radius"),
                "trajectory_connect": self._integer(
                    f"{prefix}_trajectory_connect"
                ),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_RIGHT_ARM_TO_DETECTION_POSE",
            "sending the fixed right-arm detection MoveJ command",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            "calling pre-detection right arm MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        self._parse_string_command_response(
            response, "pre-detection right arm MoveJ"
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "WAITING_FOR_RIGHT_ARM_DETECTION_POSE",
            "right-arm detection MoveJ accepted; waiting for fresh position, zero velocity, and EEPose feedback",
        )
        deadline = time.monotonic() + self._float(
            f"{prefix}_timeout_sec"
        )
        timeout_sec = self._float(f"{prefix}_timeout_sec")
        tolerance = self._float(
            f"{prefix}_position_tolerance_rad"
        )
        velocity_tolerance = self._float(
            f"{prefix}_velocity_tolerance_rad_sec"
        )
        max_age = self._float(f"{prefix}_feedback_max_age_sec")
        stable_required = self._integer(f"{prefix}_stable_samples")
        stable_samples = 0
        latest_detail = "no right-arm feedback"
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while verifying the right-arm detection MoveJ"
            )
            now = time.monotonic()
            with self.joint_state_lock:
                sequence = self.latest_slave_arm_state_sequences.get("right", 0)
                measured = list(self.latest_slave_arm_positions.get("right", []))
                velocity = list(self.latest_slave_arm_velocities.get("right", []))
                state_time = self.latest_slave_arm_state_times.get("right", 0.0)
                pose_sequence = self.latest_slave_arm_pose_sequences.get("right", 0)
            age = now - state_time
            matches = (
                sequence > sequence_before
                and len(measured) >= 7
                and len(velocity) >= 7
                and age <= max_age
                and max(abs(measured[i] - target[i]) for i in range(7)) <= tolerance
                and max(abs(velocity[i]) for i in range(7)) <= velocity_tolerance
                and pose_sequence > pose_sequence_before
            )
            latest_detail = (
                f"seq={sequence}, pose_seq={pose_sequence}, age={age:.3f}, "
                f"pos_err={max(abs(measured[i] - target[i]) for i in range(7)) if len(measured) >= 7 else float('inf'):.4f}, "
                f"vel={max(abs(value) for value in velocity) if len(velocity) >= 7 else float('inf'):.4f}"
            )
            stable_samples = stable_samples + 1 if matches else 0
            if stable_samples >= stable_required:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "RIGHT_ARM_DETECTION_POSE_REACHED",
                    "right arm reached the fixed detection pose with fresh EEPose feedback",
                )
                return f"{detail}; arm_feedback=confirmed"
            time.sleep(0.02)
        raise MissionError(
            "right arm did not reach the fixed detection pose: "
            f"{latest_detail}; timeout_sec={timeout_sec:.1f}"
        )

    def _execute_post_arm_movej(
        self, goal_handle, dry_run: bool, right_arm_only: bool = False
    ) -> str:
        if not self._boolean("box_post_arm_movej_enabled"):
            return "post_arm_movej=disabled"
        left_units, right_units, left_target, right_target = (
            self._post_arm_movej_targets()
        )
        if right_arm_only:
            return self._execute_dual_arm_movej_targets(
                goal_handle,
                dry_run,
                "box_post_arm_movej",
                left_units,
                right_units,
                left_target,
                right_target,
                "post-grasp right arm MoveJ",
                "POST_ARM_MOVEJ",
                "MOVING_POST_GRASP_ARM_JOINTS",
                "WAITING_FOR_POST_GRASP_ARM_JOINTS",
                "POST_GRASP_ARM_JOINTS_REACHED",
                "post-grasp right arm MoveJ",
                active_arms=("right",),
                velocity_parameter="box_post_arm_movej_velocity",
            )
        detail = (
            "post-grasp dual arm MoveJ: "
            f"left_device={self._integer('box_post_arm_movej_left_device')}, "
            f"right_device={self._integer('box_post_arm_movej_right_device')}, "
            f"left_joint_units={left_units}, right_joint_units={right_units}"
        )
        self._publish_box_grasp_feedback(
            goal_handle, "POST_ARM_MOVEJ_TARGETS", detail
        )
        if dry_run:
            return f"{detail}; skipped in dry-run"

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        common = {
            "command": "movej",
            "v": self._integer("box_post_arm_movej_velocity"),
            "r": self._integer("box_post_arm_movej_blend_radius"),
            "trajectory_connect": self._integer(
                "box_post_arm_movej_trajectory_connect"
            ),
        }
        left_payload = {
            "device": self._integer("box_post_arm_movej_left_device"),
            "payload": {
                **common,
                "joint": left_units,
            },
        }
        right_payload = {
            "device": self._integer("box_post_arm_movej_right_device"),
            "payload": {
                **common,
                "joint": right_units,
            },
        }
        left_request = StringCmd.Request()
        right_request = StringCmd.Request()
        left_request.data = json.dumps(left_payload, separators=(",", ":")) + "\r\n"
        right_request.data = json.dumps(right_payload, separators=(",", ":")) + "\r\n"
        with self.joint_state_lock:
            sequence_before = dict(self.latest_slave_arm_state_sequences)
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_POST_GRASP_ARM_JOINTS",
            "sending left and right arm MoveJ commands concurrently",
        )
        left_future = self.body_command_client.call_async(left_request)
        right_future = self.body_command_client.call_async(right_request)
        left_response = self._wait_future(
            left_future,
            goal_handle,
            "calling post-grasp left arm MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        right_response = self._wait_future(
            right_future,
            goal_handle,
            "calling post-grasp right arm MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        self._parse_string_command_response(left_response, "post-grasp left arm MoveJ")
        self._parse_string_command_response(right_response, "post-grasp right arm MoveJ")
        self._publish_box_grasp_feedback(
            goal_handle,
            "WAITING_FOR_POST_GRASP_ARM_JOINTS",
            "arm MoveJ commands accepted; waiting for fresh position and zero-velocity feedback",
        )
        self._wait_for_post_arm_joint_targets(
            goal_handle, left_target, right_target, sequence_before
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "POST_GRASP_ARM_JOINTS_REACHED",
            "both arm MoveJ targets reached with stable feedback",
        )
        return f"{detail}; arm_feedback=confirmed"

    def _execute_body_home(self, goal_handle, dry_run: bool) -> str:
        if not self._boolean("box_body_return_home_enabled"):
            return "body_home=disabled"
        units = [
            int(value) for value in self._float_array("box_body_home_joint_units")
        ]
        target_angles = [
            math.radians(
                float(value)
                / self._float_array("box_body_command_units_per_degree")[index]
            )
            for index, value in enumerate(units)
        ]
        detail = f"body home MoveJ joint_units={units}"
        self._publish_box_grasp_feedback(goal_handle, "BODY_HOME_TARGETS", detail)
        if dry_run:
            return f"{detail}; skipped in dry-run"
        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        with self.joint_state_lock:
            sequence_before = self.latest_body_state_sequence
        payload = {
            "device": self._integer("box_joint1_device"),
            "payload": {
                "command": "movej",
                "device": self._integer("box_joint1_device"),
                "joint": units,
                "v": self._integer("box_body_home_velocity"),
                "r": self._integer("box_body_home_blend_radius"),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            "RETURNING_BODY_HOME",
            "sending body Joint1-4 home command and waiting for fresh feedback",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            "calling body home MoveJ",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        self._parse_string_command_response(response, "body home MoveJ")
        self._publish_box_grasp_feedback(
            goal_handle, "WAITING_FOR_BODY_HOME", "waiting for all four body joints to reach zero"
        )
        self._wait_for_body_joints_target(
            goal_handle,
            target_angles,
            sequence_after=sequence_before,
            timeout_parameter="box_body_home_timeout_sec",
        )
        self._publish_box_grasp_feedback(
            goal_handle, "BODY_HOME_REACHED", "all four body joints reached home"
        )
        return f"{detail}; body_feedback=confirmed"

    def _box_layer_joint123_approach_angles_deg(
        self, box_layer: int, model_label: str | None = None
    ) -> tuple[float, float, float]:
        if box_layer < 1 or box_layer > 4:
            raise MissionError("box_layer must be in [1, 4]")
        configured = self._boolean_array("box_layer_joint123_configured")
        if len(configured) != 4:
            raise MissionError(
                "box_layer_joint123_configured must contain four values"
            )
        if not configured[box_layer - 1]:
            raise MissionError(
                f"box_layer {box_layer} joint123 target is not configured yet"
            )
        angles = []
        normalized_model = (
            str(model_label).strip().lower() if model_label is not None else ""
        )
        if normalized_model not in ("", "bigbox", "smallbox"):
            raise MissionError(
                "model_label must be 'bigbox', 'smallbox', or empty"
            )
        for index in range(1, 4):
            name = f"box_layer_joint{index}_approach_angles_deg"
            if normalized_model:
                name += f"_{normalized_model}"
            values = self._float_array(name)
            if len(values) != 4:
                raise MissionError(f"{name} must contain four values")
            angle_deg = float(values[box_layer - 1])
            if not math.isfinite(angle_deg):
                raise MissionError(
                    f"{name}[{box_layer - 1}] is invalid"
                )
            angles.append(angle_deg)
        return tuple(angles)

    def _box_layer_joint1_approach_angle_deg(
        self, box_layer: int, model_label: str | None = None
    ) -> float:
        return self._box_layer_joint123_approach_angles_deg(
            box_layer, model_label
        )[0]

    def _move_body_joints_after_detection(
        self,
        goal_handle,
        box_layer: int,
        model_label: str | None = None,
    ):
        """Move J1/J2/J3 and preserve measured J4/J5 values."""
        approach_angles = [
            math.radians(angle_deg)
            for angle_deg in self._box_layer_joint123_approach_angles_deg(
                box_layer, model_label
            )
        ]
        initial, _, sequence_before = self._wait_for_fresh_body_feedback(
            goal_handle
        )
        units_per_degree = self._float_array("box_body_command_units_per_degree")
        command_angles = approach_angles + initial[3:4]
        command_units = [
            int(round(math.degrees(angle) * units_per_degree[index]))
            for index, angle in enumerate(command_angles)
        ]
        device = self._integer("box_joint1_device")
        payload = {
            "device": device,
            "payload": {
                "command": "movej",
                "device": device,
                "joint": command_units,
                "v": self._integer("box_body_movej_velocity"),
                "r": self._integer("box_joint1_blend_radius"),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_BODY_JOINT123",
            "moving body joint1/joint2/joint3 while preserving joint4",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            f"calling body joints service {self._string('box_joint1_command_service_name')}",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        try:
            response_data = json.loads(str(response.data).strip())
        except (TypeError, ValueError) as exc:
            raise MissionError(f"invalid body joints service response: {response.data!r}") from exc
        if not response_data.get("receive_state", False):
            raise MissionError(f"body joints command was not accepted: {response.data}")
        final = self._wait_for_body_joints_target(
            goal_handle, approach_angles, sequence_after=sequence_before
        )
        return initial, final

    def _wait_for_body_joint1_target(
        self,
        goal_handle,
        target_rad: float,
        *,
        sequence_after: int = -1,
    ) -> float:
        timeout_sec = self._float("box_joint1_wait_timeout_sec")
        position_tolerance = self._float("box_joint1_position_tolerance_rad")
        velocity_tolerance = self._float("box_joint1_velocity_tolerance_rad_sec")
        max_age_sec = self._float("box_joint1_feedback_max_age_sec")
        required_stable = self._integer("box_joint1_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequence = -1
        latest_position = None
        latest_velocity = None
        latest_age = math.inf

        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for body joint1")
            position, velocity, state_time, sequence = (
                self._joint1_feedback_snapshot()
            )
            latest_position = position
            latest_velocity = velocity
            latest_age = time.monotonic() - state_time
            if sequence != last_sequence:
                last_sequence = sequence
                sample_matches = (
                    sequence > sequence_after
                    and position is not None
                    and velocity is not None
                    and latest_age <= max_age_sec
                    and abs(position - target_rad) <= position_tolerance
                    and abs(velocity) <= velocity_tolerance
                )
                stable_samples = stable_samples + 1 if sample_matches else 0
                if stable_samples >= required_stable:
                    return float(position)
            time.sleep(0.02)

        raise MissionError(
            "body joint1 did not reach the requested target: "
            f"target={target_rad:.6f} rad, measured={latest_position}, "
            f"velocity={latest_velocity}, feedback_age_sec={latest_age:.3f}, "
            f"timeout_sec={timeout_sec:.1f}"
        )

    def _move_body_joint1_after_detection(
        self, goal_handle, approach_angle_deg: float | None = None
    ) -> float:
        """Move Joint1 once and return its measured feedback-angle delta."""
        approach_deg = (
            self._float("box_joint1_approach_angle_deg")
            if approach_angle_deg is None
            else float(approach_angle_deg)
        )
        approach_rad = math.radians(approach_deg)
        initial, _, sequence_before = self._wait_for_fresh_body_feedback(
            goal_handle
        )
        initial_position = initial[0]

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        joint_units = int(
            round(
                approach_deg
                * self._float("box_joint1_command_units_per_degree")
            )
        )
        payload = {
            "device": self._integer("box_joint1_device"),
            "payload": {
                "command": "movej",
                "device": self._integer("box_joint1_device"),
                "joint": [joint_units, 0, 0, 0, 0],
                "v": self._integer("box_body_movej_velocity"),
                "r": self._integer("box_joint1_blend_radius"),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_BODY_JOINT1",
            f"moving body joint1 from {math.degrees(initial_position):.3f} "
            f"deg to {approach_deg:.3f} deg",
        )
        response = self._wait_future(
            self.body_command_client.call_async(request),
            goal_handle,
            f"calling body joint1 service {service_name}",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        try:
            response_data = json.loads(str(response.data).strip())
        except (TypeError, ValueError) as exc:
            raise MissionError(
                f"invalid body joint1 service response: {response.data!r}"
            ) from exc
        if not response_data.get("receive_state", False):
            raise MissionError(
                f"body joint1 command was not accepted: {response.data}"
            )

        self._publish_box_grasp_feedback(
            goal_handle,
            "WAITING_FOR_BODY_JOINT1",
            "body joint1 command accepted; waiting for fresh /mcap/body "
            "position and zero-velocity feedback",
        )
        final_position = self._wait_for_body_joint1_target(
            goal_handle,
            approach_rad,
            sequence_after=sequence_before,
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "BODY_JOINT1_REACHED",
            f"body joint1 reached {math.degrees(final_position):.3f} deg",
        )
        return final_position - initial_position

    def _apply_joint1_execution_mode(
        self,
        goal_handle,
        left_target: Pose,
        right_target: Pose,
        dry_run: bool,
        box_layer: int,
        model_label: str | None = None,
    ) -> tuple[Pose, Pose, str]:
        execution_mode = self._string("box_grasp_execution_mode").lower()
        if execution_mode == "arms_only":
            return left_target, right_target, "execution=arms_only"

        if execution_mode == "joint123_then_arms":
            detection_angles = [
                math.radians(
                    self._float(f"box_joint{index}_detection_angle_deg")
                )
                for index in range(1, 4)
            ]
            approach_angles = [
                math.radians(angle_deg)
                for angle_deg in self._box_layer_joint123_approach_angles_deg(
                    box_layer, model_label
                )
            ]
            if dry_run:
                movement_detail = "joint1/2/3 motion planned but skipped in dry-run"
                final_angles = approach_angles
                detection_angles_used = detection_angles
            else:
                initial_feedback, final_feedback = self._move_body_joints_after_detection(
                    goal_handle, box_layer, model_label
                )
                movement_detail = (
                    "joint1/2/3 motion completed from measured /mcap/body feedback"
                )
                # Use the fresh pre-motion feedback as the source frame. The
                # configured detection angles are only nominal and may differ
                # from the actual body pose when a goal starts.
                detection_angles_used = initial_feedback[:3]
                final_angles = final_feedback[:3]
            left_moved = self._reexpress_link8_target_after_joint123_motion(
                left_target, "left", detection_angles_used, final_angles
            )
            right_moved = self._reexpress_link8_target_after_joint123_motion(
                right_target, "right", detection_angles_used, final_angles
            )
            delta_detail = ",".join(
                f"{math.degrees(value):.3f}"
                for value in final_angles
            )
            return (
                left_moved,
                right_moved,
                "execution=joint123_then_arms; "
                "position_and_orientation=joint123_compensated; "
                f"approach_angles_deg=[{delta_detail}]; {movement_detail}",
            )

        configured_delta = math.radians(
            self._box_layer_joint1_approach_angle_deg(box_layer, model_label)
            - self._float("box_joint1_detection_angle_deg")
        )
        if dry_run:
            feedback_delta = configured_delta
            movement_detail = "joint1 motion planned but skipped in dry-run"
        else:
            feedback_delta = self._move_body_joint1_after_detection(
                goal_handle,
                self._box_layer_joint1_approach_angle_deg(
                    box_layer, model_label
                ),
            )
            movement_detail = (
                "joint1 motion completed from measured /mcap/body feedback"
            )
        if execution_mode == "joint1_then_arms_keep_position":
            left_moved = self._rotate_link8_orientation_after_joint1_rotation(
                left_target, "left", feedback_delta
            )
            right_moved = self._rotate_link8_orientation_after_joint1_rotation(
                right_target, "right", feedback_delta
            )
            target_detail = "position=unchanged, orientation=joint1_compensated"
        else:
            left_moved = self._reexpress_link8_target_after_joint1_rotation(
                left_target, "left", feedback_delta
            )
            right_moved = self._reexpress_link8_target_after_joint1_rotation(
                right_target, "right", feedback_delta
            )
            target_detail = "position_and_orientation=joint1_compensated"
        return (
            left_moved,
            right_moved,
            f"execution={execution_mode}; {target_detail}; {movement_detail}; "
            f"feedback_delta_deg={math.degrees(feedback_delta):.3f}",
        )

    def _translate_pose_in_box_frame(
        self,
        target: Pose,
        delta_box_xyz: list[float],
        arm: str,
    ) -> Pose:
        """Translate a Link8 target by a delta expressed in the box frame."""
        if self._string("direct_movel_target_mode").strip().lower() != (
            "camera_offset_box_orientation"
        ):
            raise MissionError(
                "box-frame post MoveL requires "
                "direct_movel_target_mode=camera_offset_box_orientation"
            )
        prefix = "left" if arm == "left" else "right"
        link8_orientation = BoxSupportMixin._normalize_quaternion(
            (
                float(target.orientation.x),
                float(target.orientation.y),
                float(target.orientation.z),
                float(target.orientation.w),
            )
        )
        box_to_link8 = BoxSupportMixin._normalize_quaternion(
            tuple(
                self._float_array(
                    f"direct_movel_{prefix}_box_to_link8_orientation"
                )
            )
        )
        base_to_box = BoxSupportMixin._normalize_quaternion(
            quaternion_multiply(
                link8_orientation,
                BoxSupportMixin._quaternion_conjugate(box_to_link8),
            )
        )
        delta_box = tuple(float(value) for value in delta_box_xyz)
        if len(delta_box) != 3 or not all(math.isfinite(value) for value in delta_box):
            raise MissionError(f"invalid box-frame post MoveL delta for {arm}")
        delta_base = rotate_vector(delta_box, base_to_box)
        result = deepcopy(target)
        result.position.x += delta_base[0]
        result.position.y += delta_base[1]
        result.position.z += delta_base[2]
        return result

    def _post_movel_targets(
        self,
        left_target: Pose,
        right_target: Pose,
    ) -> list[tuple[Pose, Pose]]:
        """Return standard cumulative MoveL targets from box-frame deltas."""
        return [
            (left_step, right_step)
            for _label, left_step, right_step in BoxSupportMixin._post_movel_targets_with_labels(
                self,
                left_target, right_target, include_drag_steps=False
            )
        ]

    def _post_movel_targets_with_labels(
        self,
        left_target: Pose,
        right_target: Pose,
        *,
        include_drag_steps: bool,
        defer_left_step1: bool = False,
        model_label: str | None = None,
    ) -> list[tuple[str, Pose, Pose]]:
        """Return cumulative targets, optionally inserting drag-only steps.

        DragBox can defer the left arm's Step1 until after the delayed left
        join. In that mode the left cumulative target is built from the drag
        deltas only, then a separate ``step1_left`` target is inserted before
        Step2 so Step1 is executed after the left arm has joined.
        """
        targets = []
        left_current = deepcopy(left_target)
        right_current = deepcopy(right_target)
        step_count = self._integer("box_post_movel_step_count")
        for index in range(1, step_count + 1):
            left_parameter = f"box_post_movel_left_step{index}_xyz"
            right_parameter = f"box_post_movel_right_step{index}_xyz"
            # Keep existing generic values for bigbox. Only smallbox Step1
            # currently has a model-specific calibration.
            if (
                index == 1
                and str(model_label or "").strip().lower() == "smallbox"
            ):
                left_parameter += "_smallbox"
                right_parameter += "_smallbox"
            if not (defer_left_step1 and include_drag_steps and index == 1):
                left_current = self._translate_pose_in_box_frame(
                    left_current,
                    self._float_array(left_parameter),
                    "left",
                )
            right_current = self._translate_pose_in_box_frame(
                right_current,
                self._float_array(right_parameter),
                "right",
            )
            targets.append((f"step{index}", left_current, right_current))
            if include_drag_steps and index == 1:
                for drag_index in range(1, 4):
                    left_current = self._translate_pose_in_box_frame(
                        left_current,
                        self._float_array(
                            f"drag_box_post_movel_step_drag{drag_index}_left_xyz"
                        ),
                        "left",
                    )
                    right_current = self._translate_pose_in_box_frame(
                        right_current,
                        self._float_array(
                            f"drag_box_post_movel_step_drag{drag_index}_right_xyz"
                        ),
                        "right",
                    )
                    targets.append(
                        (
                            f"step_drag{drag_index}",
                            left_current,
                            right_current,
                        )
                    )
                if defer_left_step1:
                    left_current = self._translate_pose_in_box_frame(
                        left_current,
                        self._float_array(left_parameter),
                        "left",
                    )
                    targets.append(("step1_left", left_current, right_current))
        return targets

    @staticmethod
    def _endpoint_sync_pose_values_to_transform(values):
        if values is None or len(values) != 7:
            raise MissionError("Link7 EEPose must contain seven values")
        return (
            tuple(float(value) for value in values[:3]),
            BoxSupportMixin._normalize_quaternion(values[3:7]),
        )

    @staticmethod
    def _endpoint_sync_transform_to_pose(transform) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = transform[0]
        pose.orientation.x, pose.orientation.y = transform[1][:2]
        pose.orientation.z, pose.orientation.w = transform[1][2:]
        return pose

    @staticmethod
    def _endpoint_sync_pose_position_error(lhs, rhs) -> float:
        return math.sqrt(
            sum((float(lhs[index]) - float(rhs[index])) ** 2 for index in range(3))
        )

    @staticmethod
    def _endpoint_sync_pose_orientation_error(lhs, rhs) -> float:
        lhs_q = BoxSupportMixin._normalize_quaternion(lhs)
        rhs_q = BoxSupportMixin._normalize_quaternion(rhs)
        dot = abs(sum(lhs_q[index] * rhs_q[index] for index in range(4)))
        return 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    def _step2_endpoint_sync_layer_config(self, box_layer: int):
        if box_layer < 1 or box_layer > 4:
            raise MissionError("box_layer must be in [1, 4]")
        prefix = f"box_step2_waist_endpoint_sync_layer{box_layer}_"
        if not self._boolean(f"{prefix}configured"):
            raise MissionError(
                f"box_layer {box_layer} endpoint waist sync is not configured"
            )
        values = {
            "segments": self._integer(f"{prefix}segments"),
            "forward_body_velocity": self._integer(
                f"{prefix}forward_body_velocity"
            ),
            "forward_left_speed": self._float(
                f"{prefix}forward_left_movel_velocity_percent"
            ),
            "forward_right_speed": self._float(
                f"{prefix}forward_right_movel_velocity_percent"
            ),
            "reverse_body_velocity": self._integer(
                f"{prefix}reverse_body_velocity"
            ),
            "reverse_left_speed": self._float(
                f"{prefix}reverse_left_movel_velocity_percent"
            ),
            "reverse_right_speed": self._float(
                f"{prefix}reverse_right_movel_velocity_percent"
            ),
        }
        if not 1 <= values["forward_body_velocity"] <= 100:
            raise MissionError(f"{prefix}forward_body_velocity must be in [1,100]")
        if not 1 <= values["reverse_body_velocity"] <= 100:
            raise MissionError(f"{prefix}reverse_body_velocity must be in [1,100]")
        if values["segments"] not in (1, 2):
            raise MissionError(f"{prefix}segments must be 1 or 2")
        for key in ("forward_left_speed", "forward_right_speed", "reverse_left_speed", "reverse_right_speed"):
            if not 1.0 <= values[key] <= 100.0:
                raise MissionError(f"{prefix}{key} must be in [1,100]")
        return values

    def _wait_for_fresh_endpoint_arm_poses(self, goal_handle, sequence_after):
        timeout_sec = self._float("box_step2_waist_endpoint_sync_timeout_sec")
        max_age_sec = self._float(
            "box_step2_waist_endpoint_sync_feedback_max_age_sec"
        )
        deadline = time.monotonic() + timeout_sec
        latest = {}
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while reading Step2 endpoint EEPose")
            complete = True
            now = time.monotonic()
            with self.joint_state_lock:
                for arm in ("left", "right"):
                    values = self.latest_slave_arm_poses.get(arm)
                    sequence = self.latest_slave_arm_pose_sequences.get(arm, 0)
                    age = now - self.latest_slave_arm_pose_times.get(arm, 0.0)
                    latest[arm] = (values, sequence, age)
                    if (
                        sequence <= sequence_after.get(arm, -1)
                        or values is None
                        or age > max_age_sec
                    ):
                        complete = False
            if complete:
                return {
                    arm: tuple(latest[arm][0]) for arm in ("left", "right")
                }, {
                    arm: latest[arm][1] for arm in ("left", "right")
                }
            time.sleep(0.02)
        detail = ", ".join(
            f"{arm}=sequence={item[1]}, pose={'present' if item[0] is not None else 'missing'}, age={item[2]:.3f}"
            for arm, item in latest.items()
        )
        raise MissionError(
            "fresh Step2 endpoint EEPose was not available for both arms: "
            f"{detail}, timeout_sec={timeout_sec:.1f}"
        )

    def _step2_endpoint_sync_body_request(self, body_units, velocity):
        device = self._integer("box_joint1_device")
        payload = {
            "device": device,
            "payload": {
                "command": "movej",
                "device": device,
                "joint": [int(value) for value in body_units],
                "v": int(velocity),
                "r": self._integer(
                    "box_step2_waist_endpoint_sync_body_blend_radius"
                ),
                "trajectory_connect": 0,
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        return request

    def _step2_endpoint_sync_stop_body(self):
        if not self._boolean("box_step2_waist_endpoint_sync_body_stop_enabled"):
            return
        try:
            if not self.body_command_client.service_is_ready():
                self.body_command_client.wait_for_service(timeout_sec=0.5)
            if not self.body_command_client.service_is_ready():
                raise MissionError("body command service is unavailable")
            device = self._integer("box_joint1_device")
            payload = {
                "device": device,
                "payload": {
                    "command": self._string(
                        "box_step2_waist_endpoint_sync_body_stop_command"
                    ),
                    "device": device,
                },
            }
            request = StringCmd.Request()
            request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
            self.body_command_client.call_async(request)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Step2 endpoint body stop failed: {exc}")

    def _wait_for_endpoint_arm_targets(
        self, goal_handle, targets_by_arm, sequence_after
    ):
        timeout_sec = self._float("box_step2_waist_endpoint_sync_timeout_sec")
        max_age_sec = self._float(
            "box_step2_waist_endpoint_sync_feedback_max_age_sec"
        )
        position_limit = self._float(
            "box_step2_waist_endpoint_sync_final_position_tolerance_m"
        )
        orientation_limit = self._float(
            "box_step2_waist_endpoint_sync_final_orientation_tolerance_rad"
        )
        required_stable = self._integer(
            "box_step2_waist_endpoint_sync_stable_samples"
        )
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        latest_detail = "no feedback"
        last_sequences = dict(sequence_after)
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while verifying Step2 endpoint EEPose")
            now = time.monotonic()
            all_match = True
            details = []
            with self.joint_state_lock:
                for arm in ("left", "right"):
                    values = self.latest_slave_arm_poses.get(arm)
                    sequence = self.latest_slave_arm_pose_sequences.get(arm, 0)
                    age = now - self.latest_slave_arm_pose_times.get(arm, 0.0)
                    if values is None or sequence <= sequence_after.get(arm, -1) or age > max_age_sec:
                        all_match = False
                        details.append(f"{arm}=stale/missing")
                        continue
                    target = targets_by_arm[arm]
                    if isinstance(target, Pose):
                        target = self._endpoint_sync_pose_values_to_transform(
                            (
                                target.position.x,
                                target.position.y,
                                target.position.z,
                                target.orientation.x,
                                target.orientation.y,
                                target.orientation.z,
                                target.orientation.w,
                            )
                        )
                    position_error = self._endpoint_sync_pose_position_error(
                        values[:3], target[0]
                    )
                    orientation_error = self._endpoint_sync_pose_orientation_error(
                        values[3:7], target[1]
                    )
                    details.append(
                        f"{arm}_position_error={position_error:.4f}m,"
                        f"{arm}_orientation_error={orientation_error:.4f}rad"
                    )
                    if position_error > position_limit or orientation_error > orientation_limit:
                        all_match = False
                    last_sequences[arm] = sequence
            latest_detail = "; ".join(details)
            if all_match:
                stable_samples += 1
                if stable_samples >= required_stable:
                    return
            else:
                stable_samples = 0
            time.sleep(0.02)
        raise MissionError(
            "Step2 endpoint EEPose did not reach targets: "
            f"{latest_detail}, timeout_sec={timeout_sec:.1f}"
        )

    def _execute_step2_waist_endpoint_sync(
        self,
        goal_handle,
        adapter,
        box_layer: int,
        dry_run: bool,
        sequence_after=None,
    ) -> str:
        if not self._boolean("box_step2_waist_endpoint_sync_enabled"):
            return "step2_waist_endpoint_sync=disabled"
        speeds = self._step2_endpoint_sync_layer_config(box_layer)
        prefix = "box_step2_waist_endpoint_sync_"
        home_units = [
            int(value)
            for value in self._float_array(f"{prefix}home_joint_units")
        ]
        if len(home_units) != 4:
            raise MissionError(f"{prefix}home_joint_units must contain four values")
        if dry_run:
            return (
                "step2_waist_endpoint_sync=enabled; "
                f"layer={box_layer}; home_joint_units={home_units}; "
                f"forward_body_velocity={speeds['forward_body_velocity']}; "
                f"reverse_body_velocity={speeds['reverse_body_velocity']}; "
                "fresh Step2 EEPose/body feedback required at runtime; skipped in dry-run"
            )
        if adapter is None:
            raise MissionError(
                "Step2 endpoint waist sync requires direct_motion_backend=python_sdk"
            )
        sequence_after = sequence_after or {"left": -1, "right": -1}
        body_start, _body_velocities, body_sequence = self._wait_for_fresh_body_feedback(
            goal_handle
        )
        arm_values, arm_sequences = self._wait_for_fresh_endpoint_arm_poses(
            goal_handle, sequence_after
        )
        units_per_degree = self._float_array("box_body_command_units_per_degree")
        if len(units_per_degree) != 4 or any(value <= 0.0 for value in units_per_degree):
            raise MissionError("box_body_command_units_per_degree must contain four positive values")
        start_units = [
            int(round(math.degrees(float(body_start[index])) * units_per_degree[index]))
            for index in range(4)
        ]
        start_angles = [float(value) for value in body_start[:3]]
        home_angles = [
            math.radians(float(home_units[index]) / units_per_degree[index])
            for index in range(4)
        ]
        hold_by_arm = {
            arm: BoxSupportMixin._compose_transform(
                self._joint123_arm_base_transform(arm, start_angles),
                self._endpoint_sync_pose_values_to_transform(arm_values[arm]),
            )
            for arm in ("left", "right")
        }
        def pose_targets_for_body_angles(body_angles):
            return {
                arm: self._endpoint_sync_transform_to_pose(
                    BoxSupportMixin._compose_transform(
                        BoxSupportMixin._inverse_transform(
                            self._joint123_arm_base_transform(arm, body_angles[:3])
                        ),
                        hold_by_arm[arm],
                    )
                )
                for arm in ("left", "right")
            }

        home_targets = {
            arm: self._endpoint_sync_transform_to_pose(
                BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(
                        self._joint123_arm_base_transform(arm, home_angles[:3])
                    ),
                    hold_by_arm[arm],
                )
            )
            for arm in ("left", "right")
        }
        self._publish_box_grasp_feedback(
            goal_handle,
            "STEP2_WAIST_ENDPOINT_SYNC_TARGETS",
            "Step2 complete; computed endpoint MoveL targets at waist home "
            f"from measured Link7 EEPose, layer={box_layer}, "
            f"home_joint_units={home_units}; segments={speeds['segments']}; "
            f"targets={self._dual_target_position_detail(home_targets['left'], home_targets['right'])}",
        )
        endpoint_timeout = self._float(f"{prefix}timeout_sec")
        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)

        def build_leg_plan(leg_start_units, leg_end_units, leg_start_angles, leg_end_angles):
            plan = []
            segment_count = speeds["segments"]
            for segment_index in range(1, segment_count + 1):
                fraction = float(segment_index) / float(segment_count)
                if segment_index == segment_count:
                    units = [int(value) for value in leg_end_units]
                    body_angles = [float(value) for value in leg_end_angles]
                else:
                    units = [
                        int(round(
                            float(leg_start_units[index])
                            + fraction * (float(leg_end_units[index]) - float(leg_start_units[index]))
                        ))
                        for index in range(4)
                    ]
                    body_angles = [
                        float(leg_start_angles[index])
                        + fraction * (float(leg_end_angles[index]) - float(leg_start_angles[index]))
                        for index in range(4)
                    ]
                command_angles = [
                    math.radians(float(units[index]) / units_per_degree[index])
                    for index in range(4)
                ]
                plan.append({
                    "index": segment_index,
                    "units": units,
                    "angles": command_angles,
                    "poses": pose_targets_for_body_angles(command_angles),
                })
            return plan

        def execute_leg(
            label,
            leg_start_units,
            leg_end_units,
            leg_start_angles,
            leg_end_angles,
            body_velocity,
            left_speed,
            right_speed,
            arm_sequences_before,
            body_sequence_before,
        ):
            current_arm_sequences = dict(arm_sequences_before)
            current_body_sequence = body_sequence_before
            results = []
            plan = build_leg_plan(
                leg_start_units,
                leg_end_units,
                leg_start_angles,
                leg_end_angles,
            )
            for item in plan:
                command_future = {}

                def release_body():
                    command_future["future"] = self.body_command_client.call_async(
                        self._step2_endpoint_sync_body_request(
                            item["units"], body_velocity
                        )
                    )

                self._publish_box_grasp_feedback(
                    goal_handle,
                    f"STEP2_WAIST_ENDPOINT_{label.upper()}_{item['index']}",
                    f"sending MoveL segment {item['index']}/{len(plan)} with body MoveJ, "
                    f"left_speed={left_speed:.1f}, right_speed={right_speed:.1f}; "
                    f"targets={self._dual_target_position_detail(item['poses']['left'], item['poses']['right'])}",
                )
                try:
                    motion_result = adapter.execute_dual_movel_endpoint(
                        pose_to_sdk_target(item["poses"]["left"]),
                        pose_to_sdk_target(item["poses"]["right"]),
                        left_speed,
                        right_speed,
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=endpoint_timeout,
                        before_start=release_body,
                        abort_callback=self._step2_endpoint_sync_stop_body,
                    )
                    if "future" not in command_future:
                        raise MissionError(
                            f"Step2 endpoint {label} segment {item['index']} body command was not released"
                        )
                    response = self._wait_future(
                        command_future["future"],
                        goal_handle,
                        f"starting Step2 endpoint waist {label} segment {item['index']} MoveJ",
                        self._float("dependency_wait_timeout_sec"),
                        cancel_local_future=False,
                    )
                    self._parse_string_command_response(
                        response,
                        f"starting Step2 endpoint waist {label} segment {item['index']} MoveJ",
                    )
                    self._wait_for_body_joints_target(
                        goal_handle,
                        item["angles"],
                        sequence_after=current_body_sequence,
                        timeout_parameter=f"{prefix}timeout_sec",
                    )
                    self._wait_for_endpoint_arm_targets(
                        goal_handle, item["poses"], current_arm_sequences
                    )
                    results.append(
                        f"segment={item['index']}/{len(plan)}; {motion_result}"
                    )
                    with self.joint_state_lock:
                        current_body_sequence = self.latest_body_state_sequence
                        current_arm_sequences = dict(
                            self.latest_slave_arm_pose_sequences
                        )
                except (RealManSdkCanceled, MissionCanceled):
                    self._step2_endpoint_sync_stop_body()
                    raise
                except (RealManSdkError, MissionError, ValueError) as exc:
                    self._step2_endpoint_sync_stop_body()
                    raise MissionError(
                        f"Step2 waist endpoint {label} segment {item['index']} failed: {exc}"
                    ) from exc
            return "; ".join(results)

        try:
            forward_result = execute_leg(
                "to_home",
                start_units,
                home_units,
                [float(value) for value in body_start],
                home_angles,
                speeds["forward_body_velocity"],
                speeds["forward_left_speed"],
                speeds["forward_right_speed"],
                arm_sequences,
                body_sequence,
            )
            with self.joint_state_lock:
                reverse_body_sequence = self.latest_body_state_sequence
                reverse_arm_sequences = dict(self.latest_slave_arm_pose_sequences)
            reverse_result = execute_leg(
                "to_start",
                home_units,
                start_units,
                home_angles,
                [float(value) for value in body_start],
                speeds["reverse_body_velocity"],
                speeds["reverse_left_speed"],
                speeds["reverse_right_speed"],
                reverse_arm_sequences,
                reverse_body_sequence,
            )
        except (RealManSdkCanceled, MissionCanceled):
            self._step2_endpoint_sync_stop_body()
            raise
        except (RealManSdkError, MissionError, ValueError) as exc:
            self._step2_endpoint_sync_stop_body()
            raise MissionError(f"Step2 waist endpoint sync failed: {exc}") from exc
        self._last_step2_endpoint_sync_completed = True
        return (
            "step2_waist_endpoint_sync=completed; "
            f"layer={box_layer}; home_joint_units={home_units}; "
            f"return_joint_units={start_units}; {forward_result}; {reverse_result}; "
            "body_and_Link7_feedback=confirmed"
        )

    @staticmethod
    def _dual_target_position_detail(left_target: Pose, right_target: Pose) -> str:
        return (
            f"left=[{left_target.position.x:.3f}, "
            f"{left_target.position.y:.3f}, {left_target.position.z:.3f}] m, "
            f"right=[{right_target.position.x:.3f}, "
            f"{right_target.position.y:.3f}, {right_target.position.z:.3f}] m"
        )

    def _execute_drag_box_left_join_pre_movej(
        self,
        goal_handle,
        dry_run: bool,
    ) -> str:
        """Move the left arm to its configured posture before the join.

        DragBox intentionally keeps the left arm stationary while the right
        arm performs Drag1--Drag3.  Immediately before the delayed left-arm
        MoveJ_P join, this optional MoveJ places the left arm in a known,
        reachable posture.  The existing pre-target-arm MoveJ command and
        feedback parameters are reused for device, speed, tolerances and
        timeout so the two MoveJ paths have identical safety checks.
        """
        if not self._boolean("drag_box_left_join_pre_movej_enabled"):
            return "left_join_pre_movej=disabled"

        units = [
            int(round(value))
            for value in self._float_array(
                "drag_box_left_join_pre_movej_joint_units"
            )
        ]
        if len(units) != 7:
            raise MissionError(
                "drag_box_left_join_pre_movej_joint_units must contain 7 values"
            )
        units_per_degree = self._float(
            "box_pre_target_arm_movej_command_units_per_degree"
        )
        if not math.isfinite(units_per_degree) or units_per_degree <= 0.0:
            raise MissionError(
                "box_pre_target_arm_movej_command_units_per_degree must be positive"
            )
        left_target = [
            math.radians(float(value) / units_per_degree) for value in units
        ]
        return self._execute_dual_arm_movej_targets(
            goal_handle,
            dry_run,
            "box_pre_target_arm_movej",
            units,
            [0] * 7,
            left_target,
            [0.0] * 7,
            "DragBox left-arm join pre-MoveJ",
            "DRAG_LEFT_JOIN_PRE_MOVEJ",
            "MOVING_DRAG_LEFT_JOIN_PRE_MOVEJ",
            "WAITING_FOR_DRAG_LEFT_JOIN_PRE_MOVEJ",
            "DRAG_LEFT_JOIN_PRE_MOVEJ_REACHED",
            "DragBox left-arm join pre-MoveJ",
            active_arms=("left",),
            velocity_parameter="box_pre_target_arm_movej_velocity",
        )

    def _execute_drag_box_left_join(
        self,
        goal_handle,
        adapter,
        left_target: Pose,
        dry_run: bool,
    ) -> str:
        """Move the delayed left arm to its cumulative post-drag target."""
        motion_mode = self._string(
            "drag_box_left_join_motion_mode"
        ).strip().lower()
        if motion_mode not in ("movel", "movej_p"):
            raise MissionError(
                "drag_box_left_join_motion_mode must be 'movel' or 'movej_p'"
            )
        detail = (
            "delayed left-arm join after Drag3: "
            f"{motion_mode} target="
            f"[{left_target.position.x:.3f}, {left_target.position.y:.3f}, "
            f"{left_target.position.z:.3f}] m, "
            f"q=[{left_target.orientation.x:.3f}, "
            f"{left_target.orientation.y:.3f}, "
            f"{left_target.orientation.z:.3f}, "
            f"{left_target.orientation.w:.3f}], "
            "target_frame=left_arm_base"
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "POST_MOVEL_LEFT_JOIN_TARGETS",
            detail,
        )
        if adapter is None:
            if not dry_run:
                raise MissionError(
                    "delayed left-arm join requires direct_motion_backend=python_sdk"
                )
        pre_movej_result = self._execute_drag_box_left_join_pre_movej(
            goal_handle,
            dry_run,
        )
        if dry_run:
            return f"{pre_movej_result}; {detail}; skipped in dry-run"
        try:
            motion_result = adapter.execute_single(
                "left",
                pose_to_sdk_target(left_target),
                motion_mode,
                self._float("drag_box_left_join_velocity_percent"),
                self._boolean("direct_movel_blocking"),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=self._float("drag_box_left_join_timeout_sec"),
            )
        except RealManSdkCanceled as exc:
            raise MissionCanceled(str(exc)) from exc
        except (RealManSdkError, ValueError) as exc:
            raise MissionError(f"delayed left-arm join failed: {exc}") from exc
        return (
            f"{pre_movej_result}; {detail}; {motion_result}; "
            "left_join=confirmed"
        )

    def _execute_post_movel_sequence(
        self,
        goal_handle,
        adapter,
        left_target: Pose,
        right_target: Pose,
        dry_run: bool,
        *,
        box_layer: int = 1,
        drag_mode: bool = False,
        right_arm_only: bool = False,
        delayed_left_join: bool = False,
        model_label: str | None = None,
    ) -> str:
        self._last_step2_endpoint_sync_completed = False
        standard_post_movel_enabled = self._boolean("box_post_movel_enabled")
        drag_post_movel_enabled = drag_mode and self._boolean(
            "drag_box_post_movel_enabled"
        )
        if not standard_post_movel_enabled and not drag_post_movel_enabled:
            if delayed_left_join:
                raise MissionError(
                    "delayed left-arm join requires "
                    "drag_box_post_movel_enabled=true"
                )
            return "post_movel=disabled"

        include_drag_steps = drag_post_movel_enabled
        if delayed_left_join and not include_drag_steps:
            raise MissionError(
                "delayed left-arm join requires drag_box_post_movel_enabled=true"
            )
        left_joined = not delayed_left_join
        active_arms = ("right",) if right_arm_only else ("left", "right")
        targets = self._post_movel_targets_with_labels(
            left_target,
            right_target,
            include_drag_steps=include_drag_steps,
            defer_left_step1=delayed_left_join,
            model_label=model_label,
        )
        with self.joint_state_lock:
            step2_endpoint_sequence_after = {
                "left": self.latest_slave_arm_pose_sequences.get("left", 0),
                "right": self.latest_slave_arm_pose_sequences.get("right", 0),
            }
        results = []
        for sequence_index, (label, left_step, right_step) in enumerate(
            targets, start=1
        ):
            is_step4 = label == "step4"
            is_left_only_step = label == "step1_left"
            motion_mode = "movel"
            if is_step4:
                motion_mode = self._string(
                    "box_post_movel_step4_motion_mode"
                ).strip().lower()
                if motion_mode not in ("movel", "movej_p", "movej"):
                    raise MissionError(
                        "box_post_movel_step4_motion_mode must be "
                        "'movel', 'movej_p', or 'movej'"
                    )
            detail_arms = ("left",) if is_left_only_step else active_arms
            detail = (
                f"post-grasp {', '.join(detail_arms)} arm {motion_mode} {label} "
                f"({sequence_index}/{len(targets)}) "
                "in left/right arm "
                f"base frames: "
                f"{BoxSupportMixin._dual_target_position_detail(left_step, right_step)}; "
                "delta_frame=foundationpose_box; "
                "orientation unchanged from the initial Link8 targets"
            )
            self._publish_box_grasp_feedback(
                goal_handle,
                f"POST_MOVEL_{label.upper()}_TARGETS",
                detail,
            )
            if dry_run:
                results.append(f"{detail}; skipped in dry-run")
                if delayed_left_join and label == "step_drag3":
                    results.append(
                        self._execute_drag_box_left_join(
                            goal_handle,
                            None,
                            left_step,
                            True,
                        )
                    )
                    left_joined = True
                    active_arms = ("left", "right")
                if (
                    label == "step2"
                    and not drag_mode
                    and not right_arm_only
                    and self._boolean("box_step2_waist_endpoint_sync_enabled")
                ):
                    results.append(
                        self._execute_step2_waist_endpoint_sync(
                            goal_handle,
                            None,
                            box_layer,
                            True,
                            step2_endpoint_sequence_after,
                        )
                    )
                continue
            if is_step4 and motion_mode == "movej":
                units_by_arm, targets_by_arm, _feedback_detail = (
                    self._current_step4_movej_targets(active_arms=active_arms)
                )
                movej_detail = self._execute_dual_arm_movej_targets(
                    goal_handle,
                    False,
                    "box_post_movel_step4_movej",
                    units_by_arm.get("left", []),
                    units_by_arm["right"],
                    targets_by_arm.get("left", []),
                    targets_by_arm["right"],
                    "post-grasp Step4 dual arm MoveJ",
                    "POST_MOVEL_STEP4_MOVEJ",
                    "MOVING_POST_MOVEL_STEP4_MOVEJ",
                    "WAITING_FOR_POST_MOVEL_STEP4_MOVEJ",
                    "POST_MOVEL_STEP4_MOVEJ_REACHED",
                    "post-grasp Step4 dual arm MoveJ",
                    active_arms=active_arms,
                )
                results.append(f"{detail}; {movej_detail}")
                if (
                    label == "step2"
                    and not drag_mode
                    and not right_arm_only
                    and self._boolean("box_step2_waist_endpoint_sync_enabled")
                ):
                    results.append(
                        self._execute_step2_waist_endpoint_sync(
                            goal_handle,
                            adapter,
                            box_layer,
                            dry_run,
                            step2_endpoint_sequence_after,
                        )
                    )
                continue
            try:
                if is_left_only_step:
                    motion_result = adapter.execute_single(
                        "left",
                        pose_to_sdk_target(left_step),
                        motion_mode,
                        self._float("box_post_movel_velocity_percent"),
                        self._boolean("direct_movel_blocking"),
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                    )
                elif right_arm_only and not left_joined:
                    motion_result = adapter.execute_single(
                        "right",
                        pose_to_sdk_target(right_step),
                        motion_mode,
                        self._float("box_post_movel_velocity_percent"),
                        self._boolean("direct_movel_blocking"),
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                    )
                else:
                    motion_result = adapter.execute_dual(
                        pose_to_sdk_target(left_step),
                        pose_to_sdk_target(right_step),
                        motion_mode,
                        self._float("box_post_movel_velocity_percent"),
                        self._boolean("direct_movel_blocking"),
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                    )
            except RealManSdkCanceled as exc:
                raise MissionCanceled(str(exc)) from exc
            except (RealManSdkError, ValueError) as exc:
                raise MissionError(
                    f"post-grasp {', '.join(detail_arms)} {motion_mode} {label} "
                    f"({sequence_index}/{len(targets)}) "
                    f"failed: {exc}"
                ) from exc
            results.append(f"{detail}; {motion_result}")
            if delayed_left_join and label == "step_drag3":
                results.append(
                    self._execute_drag_box_left_join(
                        goal_handle,
                        adapter,
                        left_step,
                        False,
                    )
                )
                left_joined = True
                active_arms = ("left", "right")
            if (
                label == "step2"
                and not drag_mode
                and not right_arm_only
                and self._boolean("box_step2_waist_endpoint_sync_enabled")
            ):
                results.append(
                    self._execute_step2_waist_endpoint_sync(
                        goal_handle,
                        adapter,
                        box_layer,
                        dry_run,
                        step2_endpoint_sequence_after,
                    )
                )
        return " | ".join(results) if results else "post_movel=no_steps"

    def _call_direct_box_movel(
        self,
        goal_handle,
        box_pose: PoseStamped,
        dry_run: bool,
        box_layer: int,
        *,
        drag_mode: bool = False,
        right_arm_only: bool = False,
        delayed_left_join: bool = False,
        tf_mode: bool = False,
        model_label: str | None = None,
    ) -> str:
        """Send final Link8 targets, optionally for the right arm only."""
        if (
            self._boolean("box_step2_waist_endpoint_sync_enabled")
            and not drag_mode
            and not right_arm_only
        ):
            self._step2_endpoint_sync_layer_config(box_layer)
        if tf_mode:
            frozen_box_pose = getattr(self, "_last_grasp_box_tf_box_pose", None)
            if frozen_box_pose is None:
                raise MissionError(
                    "TF GraspBox has no frozen base-frame box pose after detection"
                )
            left_tf_target, right_tf_target = self._make_tf_link8_target_poses(
                frozen_box_pose, box_layer, model_label
            )
            left_target, right_target, execution_detail = (
                self._apply_tf_execution_mode(
                    goal_handle,
                    left_tf_target,
                    right_tf_target,
                    dry_run,
                    box_layer,
                    model_label,
                )
            )
        else:
            arm_poses = getattr(self, "_last_box_pose_by_arm", {})
            left_target = self._make_direct_movel_pose(
                arm_poses.get("left", box_pose),
                "left",
            )
            right_target = self._make_direct_movel_pose(
                arm_poses.get("right", box_pose),
                "right",
            )
            left_target, right_target, execution_detail = (
                self._apply_joint1_execution_mode(
                    goal_handle,
                    left_target,
                    right_target,
                    dry_run,
                    box_layer,
                    model_label,
                )
            )
        if self._string("box_grasp_execution_mode").lower() != "arms_only":
            self._publish_box_grasp_feedback(
                goal_handle,
                "FROZEN_PRE_JOINT1_TARGETS",
                "freezing the detected left/right Link8 targets before moving "
                "body joint1; FoundationPose will not be called again",
            )
        left_values = (
            left_target.position.x,
            left_target.position.y,
            left_target.position.z,
        )
        right_values = (
            right_target.position.x,
            right_target.position.y,
            right_target.position.z,
        )
        motion_mode = self._string("direct_movel_motion_mode").strip().lower()
        compensation_detail = (
            "fixture-center compensated"
            if self._boolean("direct_movel_fixture_compensation_enabled")
            else "fixture compensation disabled"
        )
        target_mode = self._string("direct_movel_target_mode").strip().lower()
        orientation_mode = target_mode
        if target_mode == "camera_offset":
            orientation_mode = (
                "current_tf"
                if self._boolean("direct_movel_use_current_fixture_orientation")
                else "fixed_config"
            )
        left_correction_name = self._joint123_target_correction_parameter_name(
            "left", box_layer
        )
        right_correction_name = self._joint123_target_correction_parameter_name(
            "right", box_layer
        )
        left_correction = self._float_array(left_correction_name)
        right_correction = self._float_array(right_correction_name)
        detail = (
            f"direct {motion_mode} Link8 targets ({compensation_detail}) in "
            "left/right arm base frames: "
            f"left=[{left_values[0]:.3f}, {left_values[1]:.3f}, "
            f"{left_values[2]:.3f}] m, "
            f"right=[{right_values[0]:.3f}, {right_values[1]:.3f}, "
            f"{right_values[2]:.3f}] m; orientation={orientation_mode}, "
            f"left_q=[{left_target.orientation.x:.3f}, "
            f"{left_target.orientation.y:.3f}, "
            f"{left_target.orientation.z:.3f}, "
            f"{left_target.orientation.w:.3f}], "
            f"right_q=[{right_target.orientation.x:.3f}, "
            f"{right_target.orientation.y:.3f}, "
            f"{right_target.orientation.z:.3f}, "
            f"{right_target.orientation.w:.3f}]; "
            "position_offset_frame=foundationpose_box; "
            + (
                "camera_extrinsics=live_tf_tree"
                if tf_mode
                else (
                    "camera_extrinsics=live_link8_eepose"
                    if self._boolean("camera_dynamic_link8_extrinsics_enabled")
                    else "camera_extrinsics=legacy_fixed_base"
                )
            )
            + "; "
            f"box_target_correction_left=[{','.join(f'{value:.4f}' for value in left_correction)}]; "
            f"box_target_correction_right=[{','.join(f'{value:.4f}' for value in right_correction)}]; "
            f"target_correction_frame=foundationpose_box; "
            f"target_correction_layer={box_layer}; "
            f"arm_motion={('right_only_until_drag3' if delayed_left_join else ('right_only' if right_arm_only else 'dual'))}; "
            f"{execution_detail}"
        )
        self._publish_box_grasp_feedback(goal_handle, "DIRECT_MOVEL_TARGETS", detail)
        post_right_arm_only = right_arm_only and not delayed_left_join
        if dry_run:
            post_detail = self._execute_post_movel_sequence(
                goal_handle,
                None,
                left_target,
                right_target,
                True,
                box_layer=box_layer,
                drag_mode=drag_mode,
                right_arm_only=right_arm_only,
                delayed_left_join=delayed_left_join,
                model_label=model_label,
            )
            post_arm_detail = self._execute_post_arm_movej(
                goal_handle, True, right_arm_only=post_right_arm_only
            )
            body_home_detail = (
                "body_home=skipped_after_step2_endpoint_sync"
                if (
                    getattr(self, "_last_step2_endpoint_sync_completed", False)
                    and self._boolean(
                        "box_step2_waist_endpoint_sync_skip_final_body_home"
                    )
                )
                else self._execute_body_home(goal_handle, True)
            )
            return (
                f"{detail}; direct {motion_mode} skipped in dry-run; "
                f"{post_detail}; {post_arm_detail}; {body_home_detail}"
            )

        backend = self._string("direct_motion_backend").strip().lower()
        if (
            self._boolean("box_post_movel_enabled")
            or (drag_mode and self._boolean("drag_box_post_movel_enabled"))
            or self._boolean("box_post_arm_movej_enabled")
            or self._boolean("box_body_return_home_enabled")
        ) and backend != "python_sdk":
            raise MissionError(
                "post-grasp MoveL/MoveJ and body-home actions require "
                "direct_motion_backend=python_sdk"
            )
        if backend == "python_sdk":
            adapter = getattr(self, "direct_sdk_adapter", None)
            if adapter is None:
                raise MissionError(
                    "direct_motion_backend=python_sdk but the SDK adapter "
                    "is not initialized"
                )
            try:
                if right_arm_only:
                    motion_result = adapter.execute_single(
                        "right",
                        pose_to_sdk_target(right_target),
                        motion_mode,
                        self._float("direct_movel_velocity_percent"),
                        self._boolean("direct_movel_blocking"),
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                    )
                else:
                    motion_result = adapter.execute_dual(
                        pose_to_sdk_target(left_target),
                        pose_to_sdk_target(right_target),
                        motion_mode,
                        self._float("direct_movel_velocity_percent"),
                        self._boolean("direct_movel_blocking"),
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                    )
                post_detail = self._execute_post_movel_sequence(
                    goal_handle,
                    adapter,
                    left_target,
                    right_target,
                    False,
                    box_layer=box_layer,
                    drag_mode=drag_mode,
                    right_arm_only=right_arm_only,
                    delayed_left_join=delayed_left_join,
                    model_label=model_label,
                )
                post_arm_detail = self._execute_post_arm_movej(
                    goal_handle, False, right_arm_only=post_right_arm_only
                )
                body_home_detail = (
                    "body_home=skipped_after_step2_endpoint_sync"
                    if (
                        getattr(self, "_last_step2_endpoint_sync_completed", False)
                        and self._boolean(
                            "box_step2_waist_endpoint_sync_skip_final_body_home"
                        )
                    )
                    else self._execute_body_home(goal_handle, False)
                )
                return (
                    f"{detail}; {motion_result}; {post_detail}; "
                    f"{post_arm_detail}; {body_home_detail}"
                )
            except RealManSdkCanceled as exc:
                raise MissionCanceled(str(exc)) from exc
            except (RealManSdkError, ValueError) as exc:
                raise MissionError(str(exc)) from exc

        if backend != "ros_service":
            raise MissionError(f"unsupported direct_motion_backend: {backend}")
        if MoveCartesian is None:
            raise MissionError(
                "direct_motion_backend=ros_service requires "
                "task_interfaces.srv.MoveCartesian"
            )
        if self.direct_movel_client is None:
            raise MissionError("ROS service motion backend is not initialized")
        if right_arm_only:
            raise MissionError(
                "right-only or delayed-left DragBox requires "
                "direct_motion_backend=python_sdk so arm phases can be sequenced"
            )
        service_name = self._string("direct_movel_service_name")
        self._wait_for_service(self.direct_movel_client, service_name, goal_handle)
        request = MoveCartesian.Request()
        request.left_pose = left_target
        request.right_pose = right_target
        request.arm = MoveCartesian.Request.DUAL
        request.velocity = self._float("direct_movel_velocity_percent")
        request.blocking = self._boolean("direct_movel_blocking")
        request.dry_run = False
        request.motion_type = (
            MoveCartesian.Request.MOVEJ_P
            if motion_mode == "movej_p"
            else MoveCartesian.Request.MOVEL
        )
        future = self.direct_movel_client.call_async(request)
        response = self._wait_future(
            future,
            goal_handle,
            f"direct {motion_mode} service {service_name}",
            self._float("pickup_task_result_timeout_sec"),
            cancel_local_future=False,
        )
        if not response.success:
            raise MissionError(str(response.message))
        return f"{detail}; {response.message}"

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
            requested or self._string("box_object_pose_model_label")
            .strip()
            .lower(),
        )
        if model_label not in ("bigbox", "smallbox"):
            raise MissionError(
                "box_type must be 'bigbox' or 'smallbox' "
                f"(received '{requested or model_label}')"
            )
        return model_label

    def _call_box_object_pose(self, goal_handle, request, *, tf_mode: bool = False):
        if self.box_object_pose_client is None or EstimateObjectPose is None:
            raise MissionError(
                "box grasp requires the object_pose_interfaces package"
            )
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
        model_label = self._box_model_label_for_request(request)
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
                    self._measured_camera_pose_in_arm_base(
                        camera_pose, "right"
                    )
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

        # The camera is mounted on the right wrist.  Put that wrist at the
        # calibrated observation configuration before requesting FoundationPose
        # so the camera-to-Link7 transform is reproducible for every goal.
        if self._boolean("box_direct_movel_enabled"):
            self._execute_pre_detection_arm_movej(
                goal_handle, request.dry_run, right_arm_only=right_arm_only
            )
            # The per-model detection table is used by the regular GraspBox
            # mission.  DragBox must retain its existing generic observation
            # poses because its drag-specific path has separate calibration.
            detection_model_label = (
                None
                if drag_mode
                else self._box_model_label_for_request(request)
            )
            self._execute_pre_detection_right_movej(
                goal_handle,
                request.dry_run,
                request.box_layer,
                detection_model_label,
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
                    goal_handle, request, tf_mode=tf_mode
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
                    pickup_message = f"{pre_target_detail}; {pickup_message}"
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
