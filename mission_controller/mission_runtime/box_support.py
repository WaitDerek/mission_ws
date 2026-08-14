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
        if len(joint_state.name) != len(joint_state.position):
            return
        positions = [float(value) for value in joint_state.position]
        velocities = [float(value) for value in joint_state.velocity]
        if not positions or len(velocities) != len(positions):
            return
        if not all(
            math.isfinite(value) for value in (*positions, *velocities)
        ):
            return
        if arm not in ("left", "right"):
            return
        with self.joint_state_lock:
            self.latest_slave_arm_positions[arm] = positions
            self.latest_slave_arm_velocities[arm] = velocities
            self.latest_slave_arm_state_times[arm] = time.monotonic()
            self.latest_slave_arm_state_sequences[arm] += 1

    def _body_joint1_feedback_callback(self, message) -> None:
        joint_state = message.joint_state
        if len(joint_state.name) != len(joint_state.position):
            return
        names = [
            self._string(f"box_joint{index}_name")
            for index in range(1, 6)
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

    def _measured_camera_pose_in_arm_base(
        self, camera_pose: PoseStamped, arm: str
    ) -> PoseStamped:
        """Apply the measured camera-to-arm-base extrinsic directly."""
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

    def _joint123_arm_base_transform(self, arm: str, angles_rad):
        """Return the measured root->arm-base transform for J1/J2/J3 angles."""
        joint1_angle, joint2_angle, joint3_angle = angles_rad
        joint1_axis = self._float_array("box_joint1_axis_xyz")
        joint1_signed = joint1_angle * self._float(
            "box_joint1_feedback_to_geometric_sign"
        )
        root_to_joint1 = BoxSupportMixin._rotation_transform(
            joint1_axis, joint1_signed
        )
        joint1_to_joint2 = BoxSupportMixin._configured_joint_transform(
            self,
            "box_joint1_to_joint2_xyz",
            "box_joint1_to_joint2_rotation",
            "box_joint2_axis_xyz",
            joint2_angle * self._float("box_joint2_feedback_to_urdf_axis_sign"),
        )
        joint2_to_joint3 = BoxSupportMixin._configured_joint_transform(
            self,
            "box_joint2_to_joint3_xyz",
            "box_joint2_to_joint3_rotation",
            "box_joint3_axis_xyz",
            joint3_angle * self._float("box_joint3_feedback_to_urdf_axis_sign"),
        )
        prefix = "left" if arm == "left" else "right"
        joint1_to_base_zero = (
            tuple(self._float_array(f"box_joint1_to_{prefix}_base_xyz")),
            BoxSupportMixin._quaternion_from_rotation_matrix(
                self._float_array(f"box_joint1_to_{prefix}_base_rotation")
            ),
        )
        fixed_joint3_to_base = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(
                BoxSupportMixin._compose_transform(
                    BoxSupportMixin._configured_joint_transform(
                        self,
                        "box_joint1_to_joint2_xyz",
                        "box_joint1_to_joint2_rotation",
                        "box_joint2_axis_xyz",
                        0.0,
                    ),
                    BoxSupportMixin._configured_joint_transform(
                        self,
                        "box_joint2_to_joint3_xyz",
                        "box_joint2_to_joint3_rotation",
                        "box_joint3_axis_xyz",
                        0.0,
                    ),
                )
            ),
            joint1_to_base_zero,
        )
        return BoxSupportMixin._compose_transform(
            root_to_joint1,
            BoxSupportMixin._compose_transform(
                joint1_to_joint2,
                BoxSupportMixin._compose_transform(
                    joint2_to_joint3, fixed_joint3_to_base
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
                for index in range(1, 6)
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
        latest_positions = [None] * 5
        latest_velocities = [None] * 5
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

    def _wait_for_post_arm_joint_targets(
        self,
        goal_handle,
        left_target_rad,
        right_target_rad,
        sequence_after,
    ) -> None:
        """Require both arm feedback streams to reach all seven targets."""
        tolerance = self._float("box_post_arm_position_tolerance_rad")
        velocity_tolerance = self._float(
            "box_post_arm_velocity_tolerance_rad_sec"
        )
        max_age = self._float("box_post_arm_feedback_max_age_sec")
        timeout_sec = self._float("box_post_arm_movej_timeout_sec")
        stable_required = self._integer("box_post_arm_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequences = {"left": -1, "right": -1}
        latest_detail = "no feedback"
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while verifying post-grasp arm MoveJ"
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
            for arm, target in (("left", left_target_rad), ("right", right_target_rad)):
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
            "post-grasp arm MoveJ targets were not confirmed: "
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

    def _execute_post_arm_movej(self, goal_handle, dry_run: bool) -> str:
        if not self._boolean("box_post_arm_movej_enabled"):
            return "post_arm_movej=disabled"
        left_units, right_units, left_target, right_target = (
            self._post_arm_movej_targets()
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
            "sending body Joint1-5 home command and waiting for fresh feedback",
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
            goal_handle, "WAITING_FOR_BODY_HOME", "waiting for all five body joints to reach zero"
        )
        self._wait_for_body_joints_target(
            goal_handle,
            target_angles,
            sequence_after=sequence_before,
            timeout_parameter="box_body_home_timeout_sec",
        )
        self._publish_box_grasp_feedback(
            goal_handle, "BODY_HOME_REACHED", "all five body joints reached home"
        )
        return f"{detail}; body_feedback=confirmed"

    def _move_body_joints_after_detection(self, goal_handle):
        """Move J1/J2/J3 and preserve measured J4/J5 values."""
        detection_angles = [
            math.radians(self._float(f"box_joint{index}_detection_angle_deg"))
            for index in range(1, 4)
        ]
        approach_angles = [
            math.radians(self._float(f"box_joint{index}_approach_angle_deg"))
            for index in range(1, 4)
        ]
        initial = self._wait_for_body_joints_target(goal_handle, detection_angles)
        if any(value is None for value in initial[3:5]):
            raise MissionError(
                "body feedback did not provide joint4/joint5 positions; "
                "refusing to command them to an unknown value"
            )
        _, _, _, sequence_before = self._body_feedback_snapshot()
        units_per_degree = self._float_array("box_body_command_units_per_degree")
        command_angles = approach_angles + initial[3:5]
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
                "v": self._integer("box_joint1_velocity"),
                "r": self._integer("box_joint1_blend_radius"),
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_BODY_JOINT123",
            "moving body joint1/joint2/joint3 while preserving joint4/joint5",
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

    def _move_body_joint1_after_detection(self, goal_handle) -> float:
        """Move Joint1 once and return its measured feedback-angle delta."""
        detection_rad = math.radians(
            self._float("box_joint1_detection_angle_deg")
        )
        approach_deg = self._float("box_joint1_approach_angle_deg")
        approach_rad = math.radians(approach_deg)
        initial_position = self._wait_for_body_joint1_target(
            goal_handle, detection_rad
        )
        _, _, _, sequence_before = self._joint1_feedback_snapshot()

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
                "v": self._integer("box_joint1_velocity"),
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
                math.radians(
                    self._float(f"box_joint{index}_approach_angle_deg")
                )
                for index in range(1, 4)
            ]
            if dry_run:
                movement_detail = "joint1/2/3 motion planned but skipped in dry-run"
                final_angles = approach_angles
                detection_angles_used = detection_angles
            else:
                initial_feedback, final_feedback = self._move_body_joints_after_detection(
                    goal_handle
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
            self._float("box_joint1_approach_angle_deg")
            - self._float("box_joint1_detection_angle_deg")
        )
        if dry_run:
            feedback_delta = configured_delta
            movement_detail = "joint1 motion planned but skipped in dry-run"
        else:
            feedback_delta = self._move_body_joint1_after_detection(goal_handle)
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
        """Return cumulative MoveL targets from box-frame deltas."""
        targets = []
        left_current = deepcopy(left_target)
        right_current = deepcopy(right_target)
        step_count = self._integer("box_post_movel_step_count")
        for index in range(1, step_count + 1):
            left_current = self._translate_pose_in_box_frame(
                left_current,
                self._float_array(f"box_post_movel_left_step{index}_xyz"),
                "left",
            )
            right_current = self._translate_pose_in_box_frame(
                right_current,
                self._float_array(f"box_post_movel_right_step{index}_xyz"),
                "right",
            )
            targets.append((left_current, right_current))
        return targets

    @staticmethod
    def _dual_target_position_detail(left_target: Pose, right_target: Pose) -> str:
        return (
            f"left=[{left_target.position.x:.3f}, "
            f"{left_target.position.y:.3f}, {left_target.position.z:.3f}] m, "
            f"right=[{right_target.position.x:.3f}, "
            f"{right_target.position.y:.3f}, {right_target.position.z:.3f}] m"
        )

    def _execute_post_movel_sequence(
        self,
        goal_handle,
        adapter,
        left_target: Pose,
        right_target: Pose,
        dry_run: bool,
    ) -> str:
        if not self._boolean("box_post_movel_enabled"):
            return "post_movel=disabled"

        targets = self._post_movel_targets(left_target, right_target)
        results = []
        for index, (left_step, right_step) in enumerate(targets, start=1):
            detail = (
                f"post-grasp dual MoveL step {index}/{len(targets)} in left/right arm "
                f"base frames: "
                f"{BoxSupportMixin._dual_target_position_detail(left_step, right_step)}; "
                "delta_frame=foundationpose_box; "
                "orientation unchanged from the initial Link8 targets"
            )
            self._publish_box_grasp_feedback(
                goal_handle,
                f"POST_MOVEL_STEP_{index}_TARGETS",
                detail,
            )
            if dry_run:
                results.append(f"{detail}; skipped in dry-run")
                continue
            try:
                motion_result = adapter.execute_dual(
                    pose_to_sdk_target(left_step),
                    pose_to_sdk_target(right_step),
                    "movel",
                    self._float("box_post_movel_velocity_percent"),
                    self._boolean("direct_movel_blocking"),
                    cancel_requested=lambda: goal_handle.is_cancel_requested,
                    timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                )
            except RealManSdkCanceled as exc:
                raise MissionCanceled(str(exc)) from exc
            except (RealManSdkError, ValueError) as exc:
                raise MissionError(
                    f"post-grasp dual MoveL step {index}/{len(targets)} failed: {exc}"
                ) from exc
            results.append(f"{detail}; {motion_result}")
        return " | ".join(results) if results else "post_movel=no_steps"

    def _call_direct_box_movel(
        self, goal_handle, box_pose: PoseStamped, dry_run: bool
    ) -> str:
        """Send both final Link8 targets through the selected motion backend."""
        arm_poses = getattr(self, "_last_box_pose_by_arm", {})
        left_target = self._make_direct_movel_pose(
            arm_poses.get("left", box_pose),
            "left",
        )
        right_target = self._make_direct_movel_pose(
            arm_poses.get("right", box_pose),
            "right",
        )
        if self._string("box_grasp_execution_mode").lower() != "arms_only":
            self._publish_box_grasp_feedback(
                goal_handle,
                "FROZEN_PRE_JOINT1_TARGETS",
                "freezing the detected left/right Link8 targets before moving "
                "body joint1; FoundationPose will not be called again",
            )
        left_target, right_target, execution_detail = (
            self._apply_joint1_execution_mode(
                goal_handle, left_target, right_target, dry_run
            )
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
            f"{execution_detail}"
        )
        self._publish_box_grasp_feedback(goal_handle, "DIRECT_MOVEL_TARGETS", detail)
        if dry_run:
            post_detail = self._execute_post_movel_sequence(
                goal_handle,
                None,
                left_target,
                right_target,
                True,
            )
            post_arm_detail = self._execute_post_arm_movej(goal_handle, True)
            body_home_detail = self._execute_body_home(goal_handle, True)
            return (
                f"{detail}; direct {motion_mode} skipped in dry-run; "
                f"{post_detail}; {post_arm_detail}; {body_home_detail}"
            )

        backend = self._string("direct_motion_backend").strip().lower()
        if (
            self._boolean("box_post_movel_enabled")
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
                )
                post_arm_detail = self._execute_post_arm_movej(
                    goal_handle, False
                )
                body_home_detail = self._execute_body_home(goal_handle, False)
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
        model_label = self._string("box_object_pose_model_label").strip()
        if (
            self._string("direct_movel_target_mode").strip().lower()
            == "camera_offset_box_orientation"
            and model_label.lower()
            != self._string("direct_movel_box_relative_model_label")
            .strip()
            .lower()
        ):
            calibration_label = self._string(
                "direct_movel_box_relative_model_label"
            )
            raise MissionError(
                "box-orientation calibration model does not match the requested "
                f"model: calibration={calibration_label}, requested={model_label}"
            )
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

        target_frame = self._string("arm_execution_frame").lstrip("/")
        # ExecuteBoxGrasp uses the detector's camera-frame pose directly.
        # The adaptive action performs its own explicit orientation check.
        camera_pose = foundation_result.pose
        if self._boolean("camera_measured_extrinsics_enabled"):
            target_mode = self._string("direct_movel_target_mode").strip().lower()
            left_box_pose = self._measured_camera_pose_in_arm_base(
                camera_pose, "left"
            )
            right_box_pose = self._measured_camera_pose_in_arm_base(
                camera_pose, "right"
            )
            left_camera_target = self._apply_box_frame_grasp_offset(
                camera_pose, "direct_movel_left_offset_xyz"
            )
            right_camera_target = self._apply_box_frame_grasp_offset(
                camera_pose, "direct_movel_right_offset_xyz"
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
                # base frame; no unmeasured camera->base_link extrinsic is used.
                raw_foundation_center_pose = right_box_pose
                foundation_center_pose = right_box_pose
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
        self, goal_handle, request, motion_state: dict[str, bool]
    ):
        detection_attempts = self._integer("box_detection_attempts")
        failures: list[str] = []

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
                    pickup_message = self._call_direct_box_movel(
                        goal_handle, box_pose, request.dry_run
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
