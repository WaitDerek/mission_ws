import math
from copy import deepcopy
import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from rclpy.duration import Duration

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
    MissionError,
    pose_to_array,
    quaternion_multiply,
    rotate_vector,
)


# Bound to the composed compatibility facade by box_support.py.
BoxSupportMixin = None


class BoxGeometryMixin:
    """Feedback, TF, quaternion, and geometric target helpers."""

    def _slave_arm_feedback_callback(self, arm: str, message) -> None:
        """Cache fresh RX arm joint position/velocity feedback."""
        joint_state = message.joint_state
        if len(joint_state.name) != len(joint_state.position) or len(
            joint_state.name
        ) != len(joint_state.velocity):
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
        if not all(math.isfinite(value) for value in (*positions, *velocities)):
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
        names = [self._string(f"box_joint{index}_name") for index in range(1, 5)]
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
        qx, qy, qz, qw = quaternion_multiply(correction_quaternion, mount_quaternion)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self._string("camera_mount_parent_frame").lstrip(
            "/"
        )
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
            feedback_frame = getattr(self, "latest_slave_arm_pose_frames", {}).get(
                arm, ""
            )
        return feedback_frame or self._string(
            f"{prefix}_arm_base_frame"
        ).strip().lstrip("/")

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
            tuple(self._float_array(f"camera_{prefix}_link8_to_rgb_camera_xyz")),
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
            BoxSupportMixin._normalize_quaternion(tuple(link8_values[3:])),
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
                tuple(self._float_array("camera_right_base_to_left_base_xyz")),
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
                timeout=Duration(seconds=self._float("camera_tf_timeout_sec")),
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
        if stamp_is_zero and self._boolean("grasp_box_tf_require_detection_timestamp"):
            raise MissionError(
                "TF GraspBox requires a non-zero FoundationPose detection timestamp"
            )
        lookup_time = (
            rclpy.time.Time.from_msg(stamp) if not stamp_is_zero else rclpy.time.Time()
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

    def _tf_target_pose_in_arm_base(self, target: PoseStamped, arm: str) -> Pose:
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
        *,
        drag_mode: bool = False,
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
            center_pose, corrected_box_pose = self._apply_box_frame_target_correction(
                frozen_box_pose,
                self._direct_movel_offset_parameter_name(
                    arm,
                    box_layer,
                    model_label,
                    tf_mode=True,
                    drag_mode=drag_mode,
                ),
                self._joint123_target_correction_parameter_name(
                    arm,
                    box_layer,
                    model_label,
                    tf_mode=True,
                    drag_mode=drag_mode,
                ),
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
        *,
        drag_mode: bool = False,
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
                    goal_handle,
                    box_layer,
                    model_label,
                    tf_mode=True,
                    drag_mode=drag_mode,
                )
                movement_detail = (
                    "joint1/2/3 motion completed from measured /mcap/body feedback"
                )
            elif execution_mode in (
                "joint1_then_arms",
                "joint1_then_arms_keep_position",
            ):
                self._move_body_joint1_after_detection(
                    goal_handle,
                    self._box_layer_joint1_approach_angle_deg(
                        box_layer,
                        model_label,
                        tf_mode=True,
                        drag_mode=drag_mode,
                    ),
                )
                movement_detail = (
                    "joint1 motion completed from measured /mcap/body feedback"
                )
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
        result.header.frame_id = self._string(f"{prefix}_arm_base_frame").lstrip("/")
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
        quaternion_norm = math.sqrt(sum(value * value for value in box_quaternion))
        if quaternion_norm <= 1e-12 or not all(
            math.isfinite(value) for value in (*offset_box, *box_quaternion)
        ):
            raise MissionError(
                f"invalid FoundationPose orientation or offset for {offset_name}"
            )
        box_quaternion = tuple(value / quaternion_norm for value in box_quaternion)
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
            float(offset_box[index]) + float(correction[index]) for index in range(3)
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
        *,
        tf_mode: bool = False,
        drag_mode: bool = False,
    ) -> str:
        """Resolve the independent initial grasp offset for a model/layer."""
        if arm not in ("left", "right") or box_layer < 1 or box_layer > 4:
            raise MissionError("arm must be left/right and box_layer must be in [1, 4]")
        if tf_mode:
            action_prefix = "drag_box_tf" if drag_mode else "grasp_box_tf"
            normalized_model = str(model_label or "").strip().lower()
            if normalized_model not in ("bigbox", "smallbox"):
                raise MissionError(
                    "TF box target requires model_label='bigbox' or 'smallbox'"
                )
            return (
                f"{action_prefix}_direct_movel_{arm}_offset_xyz_"
                f"{normalized_model}_layer{box_layer}"
            )
        parameter_name = f"direct_movel_{arm}_offset_xyz"
        normalized_model = str(model_label or "").strip().lower()
        if normalized_model in ("bigbox", "smallbox"):
            parameter_name += f"_{normalized_model}_layer{box_layer}"
        return parameter_name

    def _joint123_target_correction_parameter_name(
        self,
        arm: str,
        box_layer: int,
        model_label: str | None = None,
        *,
        tf_mode: bool = False,
        drag_mode: bool = False,
    ) -> str:
        if arm not in ("left", "right") or box_layer < 1 or box_layer > 4:
            raise MissionError("arm must be left/right and box_layer must be in [1, 4]")
        if tf_mode:
            action_prefix = "drag_box_tf" if drag_mode else "grasp_box_tf"
            normalized_model = str(model_label or "").strip().lower()
            if normalized_model not in ("bigbox", "smallbox"):
                raise MissionError(
                    "TF box target requires model_label='bigbox' or 'smallbox'"
                )
            return (
                f"{action_prefix}_joint123_{arm}_target_correction_pose_box_"
                f"{normalized_model}_layer{box_layer}"
            )
        return f"joint123_layer{box_layer}_{arm}_target_correction_pose_box"

    @staticmethod
    def _tf_layer_parameter_name(
        action_prefix: str,
        stem: str,
        model_label: str | None,
        box_layer: int,
    ) -> str:
        normalized_model = str(model_label or "").strip().lower()
        if action_prefix not in ("grasp_box_tf", "drag_box_tf"):
            raise MissionError("invalid TF box action prefix")
        if normalized_model not in ("bigbox", "smallbox"):
            raise MissionError(
                "TF box target requires model_label='bigbox' or 'smallbox'"
            )
        if box_layer < 1 or box_layer > 4:
            raise MissionError("box_layer must be in [1, 4]")
        return f"{action_prefix}_{stem}_{normalized_model}_layer{box_layer}"

    def _box_detection_arm(
        self, *, tf_mode: bool = False, drag_mode: bool = False
    ) -> str:
        """Resolve the camera/detection arm for the current action path."""
        if tf_mode:
            parameter_name = (
                "drag_box_tf_detection_arm"
                if drag_mode
                else "grasp_box_tf_detection_arm"
            )
        else:
            parameter_name = "camera_detection_arm"
        arm = self._string(parameter_name).strip().lower()
        if arm not in ("left", "right"):
            raise MissionError(
                f"{parameter_name} must be 'left' or 'right' (got '{arm}')"
            )
        return arm

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
            if self._boolean("camera_measured_extrinsics_enabled") and self._boolean(
                "camera_dynamic_link8_extrinsics_enabled"
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

    def _forward_box_object_pose_feedback(self, goal_handle, feedback_message) -> None:
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
        orientation_norm = math.sqrt(sum(value * value for value in pickup_orientation))
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
        result.pose.orientation = BoxSupportMixin._box_oriented_link8_orientation(
            self, box_pose, arm
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
        relative_q_norm = math.sqrt(sum(value * value for value in relative_q_values))
        if relative_q_norm <= 1e-12:
            raise MissionError(
                f"configured {prefix} box-to-Link8 orientation is invalid"
            )
        relative_q = tuple(value / relative_q_norm for value in relative_q_values)
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
            fixture_xyz = self._float_array(f"{prefix}_fixture_center_in_link8_xyz")
            fixture_offset_base = rotate_vector(tuple(fixture_xyz), target_q)
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
        quaternion: tuple[float, float, float, float],
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
        mount_position = tuple(self._float_array(f"box_joint1_to_{prefix}_base_xyz"))
        mount_orientation = BoxSupportMixin._quaternion_from_rotation_matrix(
            self._float_array(f"box_joint1_to_{prefix}_base_rotation")
        )
        axis = self._float_array("box_joint1_axis_xyz")
        axis_norm = math.sqrt(sum(value * value for value in axis))
        if axis_norm <= 1e-12:
            raise MissionError("box_joint1_axis_xyz has zero length")
        signed_delta = feedback_delta_rad * self._float(
            "box_joint1_feedback_to_geometric_sign"
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
        moved_mount_orientation = quaternion_multiply(joint_rotation, mount_orientation)

        target_position = (
            target.position.x,
            target.position.y,
            target.position.z,
        )
        target_in_joint1 = rotate_vector(target_position, mount_orientation)
        target_in_joint1 = tuple(
            mount_position[index] + target_in_joint1[index] for index in range(3)
        )
        target_relative_to_moved_base = tuple(
            target_in_joint1[index] - moved_mount_position[index] for index in range(3)
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
                self, target, arm, feedback_delta_rad
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
    def _slerp_quaternion(start, target, fraction):
        """Shortest-path normalized quaternion interpolation."""
        start_q = BoxSupportMixin._normalize_quaternion(start)
        target_q = BoxSupportMixin._normalize_quaternion(target)
        fraction = max(0.0, min(1.0, float(fraction)))
        dot = sum(start_q[index] * target_q[index] for index in range(4))
        if dot < 0.0:
            target_q = tuple(-value for value in target_q)
            dot = -dot
        dot = max(-1.0, min(1.0, dot))
        if dot > 0.9995:
            return BoxSupportMixin._normalize_quaternion(
                tuple(
                    start_q[index] + fraction * (target_q[index] - start_q[index])
                    for index in range(4)
                )
            )
        theta = math.acos(dot)
        sine = math.sin(theta)
        if abs(sine) <= 1e-12:
            return start_q
        start_weight = math.sin((1.0 - fraction) * theta) / sine
        target_weight = math.sin(fraction * theta) / sine
        return BoxSupportMixin._normalize_quaternion(
            tuple(
                start_weight * start_q[index] + target_weight * target_q[index]
                for index in range(4)
            )
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
            BoxSupportMixin._rotation_transform(self._float_array(axis_name), angle),
        )

    def _configured_rpy_transform(self, xyz_name, rpy_name):
        """Build a fixed URDF origin transform from XYZ and RPY parameters."""
        return (
            tuple(self._float_array(xyz_name)),
            BoxSupportMixin._normalize_quaternion(
                BoxSupportMixin._quaternion_from_rpy(*self._float_array(rpy_name))
            ),
        )

    def _joint123_chest_transform(self, angles_rad):
        """Return the configured URDF root->chest transform for waist J1/J2/J3."""
        joint1_angle, joint2_angle, joint3_angle = angles_rad
        joint1_origin = BoxSupportMixin._configured_rpy_transform(
            self, "box_waist1_origin_xyz", "box_waist1_origin_rpy"
        )
        waist1_rotation = BoxSupportMixin._rotation_transform(
            self._float_array("box_joint1_axis_xyz"),
            joint1_angle * self._float("box_joint1_feedback_to_geometric_sign"),
        )
        waist1_to_waist2 = BoxSupportMixin._compose_transform(
            BoxSupportMixin._configured_rpy_transform(
                self, "box_waist2_origin_xyz", "box_waist2_origin_rpy"
            ),
            BoxSupportMixin._rotation_transform(
                self._float_array("box_joint2_axis_xyz"),
                joint2_angle * self._float("box_joint2_feedback_to_urdf_axis_sign"),
            ),
        )
        waist2_to_waist3 = BoxSupportMixin._compose_transform(
            BoxSupportMixin._configured_rpy_transform(
                self, "box_waist3_origin_xyz", "box_waist3_origin_rpy"
            ),
            BoxSupportMixin._rotation_transform(
                self._float_array("box_joint3_axis_xyz"),
                joint3_angle * self._float("box_joint3_feedback_to_urdf_axis_sign"),
            ),
        )
        waist3_to_chest = BoxSupportMixin._configured_rpy_transform(
            self, "box_waist3_to_chest_xyz", "box_waist3_to_chest_rpy"
        )
        return BoxSupportMixin._compose_transform(
            joint1_origin,
            BoxSupportMixin._compose_transform(
                waist1_rotation,
                BoxSupportMixin._compose_transform(
                    waist1_to_waist2,
                    BoxSupportMixin._compose_transform(
                        waist2_to_waist3,
                        waist3_to_chest,
                    ),
                ),
            ),
        )

    def _joint123_arm_base_transform(self, arm: str, angles_rad):
        """Return the configured URDF root->arm-base transform for waist J1/J2/J3."""
        prefix = "left" if arm == "left" else "right"
        chest_to_arm_base = BoxSupportMixin._configured_rpy_transform(
            self,
            f"box_chest_to_{prefix}_arm_base_xyz",
            f"box_chest_to_{prefix}_arm_base_rpy",
        )
        return BoxSupportMixin._compose_transform(
            BoxSupportMixin._joint123_chest_transform(self, angles_rad),
            chest_to_arm_base,
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
            self, arm, detection_angles_rad
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
