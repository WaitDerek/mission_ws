import math
import time
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
from grasp_orchestrator_interfaces.srv import DetectGraspPose
from rclpy.duration import Duration
from task_interfaces.action import MoveArmPose
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import TransformException
from visualization_msgs.msg import Marker, MarkerArray

from .common import (
    GraspCandidate,
    MissionCanceled,
    MissionError,
    TwoStageMotionError,
    compose_poses,
    interpolate_pose,
    pose_to_array,
    quaternion_multiply,
    rotate_vector,
)


class GraspSupportMixin:
    """Grasp perception transforms, visualization, and arm pose execution."""

    @staticmethod
    def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
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

    def _grasp_center_to_gripper_pose(self, grasp_center: Pose) -> Pose:
        center_values = pose_to_array(grasp_center)
        grasp_orientation = tuple(center_values[3:])
        center_to_gripper = tuple(
            self._float_array("grasp_center_to_gripper_xyz")
        )
        center_to_gripper_in_target = rotate_vector(
            center_to_gripper, grasp_orientation
        )
        grasp_to_gripper_orientation = self._quaternion_from_rpy(
            *self._float_array("grasp_to_gripper_rpy")
        )
        gripper_orientation = quaternion_multiply(
            grasp_orientation, grasp_to_gripper_orientation
        )
        gripper_target_post_orientation = self._quaternion_from_rpy(
            *self._float_array("gripper_target_post_rpy")
        )
        gripper_orientation = quaternion_multiply(
            gripper_orientation, gripper_target_post_orientation
        )
        orientation_norm = math.sqrt(
            sum(value * value for value in gripper_orientation)
        )
        gripper_orientation = tuple(
            value / orientation_norm for value in gripper_orientation
        )

        gripper_pose = Pose()
        gripper_pose.position.x = (
            center_values[0] + center_to_gripper_in_target[0]
        )
        gripper_pose.position.y = (
            center_values[1] + center_to_gripper_in_target[1]
        )
        gripper_pose.position.z = (
            center_values[2] + center_to_gripper_in_target[2]
        )
        gripper_pose.orientation.x = gripper_orientation[0]
        gripper_pose.orientation.y = gripper_orientation[1]
        gripper_pose.orientation.z = gripper_orientation[2]
        gripper_pose.orientation.w = gripper_orientation[3]

        self.get_logger().info(
            "converted grasp center to gripper_link target: "
            f"center=[{center_values[0]:.4f}, {center_values[1]:.4f}, "
            f"{center_values[2]:.4f}], "
            f"target=[{gripper_pose.position.x:.4f}, "
            f"{gripper_pose.position.y:.4f}, "
            f"{gripper_pose.position.z:.4f}], "
            f"center_to_gripper={list(center_to_gripper)}, "
            "grasp_to_gripper_rpy="
            f"{self._float_array('grasp_to_gripper_rpy')}, "
            "gripper_target_post_rpy="
            f"{self._float_array('gripper_target_post_rpy')}"
        )
        return gripper_pose

    @staticmethod
    def _quaternion_angular_distance(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if first_norm <= 1e-12 or second_norm <= 1e-12:
            raise MissionError("cannot compare a zero-norm quaternion")
        dot = abs(
            sum(a * b for a, b in zip(first, second))
            / (first_norm * second_norm)
        )
        return 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    def _normalize_grasp_symmetry(
        self,
        grasp_pose: PoseStamped,
        execution_frame: str,
        gripper_frame: str,
    ) -> tuple[PoseStamped, Pose]:
        primary_gripper = self._grasp_center_to_gripper_pose(grasp_pose.pose)
        if not self._boolean("grasp_symmetry_normalization_enabled"):
            return grasp_pose, primary_gripper

        values = pose_to_array(grasp_pose.pose)
        symmetry = self._quaternion_from_rpy(
            *self._float_array("grasp_symmetry_rpy")
        )
        alternative_orientation = quaternion_multiply(tuple(values[3:]), symmetry)
        orientation_norm = math.sqrt(
            sum(value * value for value in alternative_orientation)
        )

        alternative_grasp = PoseStamped()
        alternative_grasp.header = grasp_pose.header
        alternative_grasp.pose.position.x = values[0]
        alternative_grasp.pose.position.y = values[1]
        alternative_grasp.pose.position.z = values[2]
        alternative_grasp.pose.orientation.x = (
            alternative_orientation[0] / orientation_norm
        )
        alternative_grasp.pose.orientation.y = (
            alternative_orientation[1] / orientation_norm
        )
        alternative_grasp.pose.orientation.z = (
            alternative_orientation[2] / orientation_norm
        )
        alternative_grasp.pose.orientation.w = (
            alternative_orientation[3] / orientation_norm
        )
        alternative_gripper = self._grasp_center_to_gripper_pose(
            alternative_grasp.pose
        )

        try:
            current_transform = self.tf_buffer.lookup_transform(
                execution_frame,
                gripper_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self._float("camera_tf_timeout_sec")),
            )
        except TransformException as exc:
            self.get_logger().warning(
                "cannot normalize grasp symmetry without current "
                f"{gripper_frame} TF: {exc}; keeping the primary branch"
            )
            return grasp_pose, primary_gripper

        current_orientation = current_transform.transform.rotation
        current_quaternion = (
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
            current_orientation.w,
        )
        primary_values = pose_to_array(primary_gripper)
        alternative_values = pose_to_array(alternative_gripper)
        primary_distance = self._quaternion_angular_distance(
            current_quaternion, tuple(primary_values[3:])
        )
        alternative_distance = self._quaternion_angular_distance(
            current_quaternion, tuple(alternative_values[3:])
        )
        use_alternative = alternative_distance + 1e-6 < primary_distance
        self.get_logger().info(
            "grasp symmetry normalization: "
            f"primary={math.degrees(primary_distance):.1f} deg, "
            f"alternative={math.degrees(alternative_distance):.1f} deg, "
            f"selected={'alternative' if use_alternative else 'primary'}"
        )
        if use_alternative:
            return alternative_grasp, alternative_gripper
        return grasp_pose, primary_gripper

    def _gripper_target_to_ee_target(
        self,
        gripper_target: PoseStamped,
        gripper_frame: str,
        ee_frame: str,
    ) -> PoseStamped:
        try:
            transform = self.tf_buffer.lookup_transform(
                gripper_frame,
                ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self._float("camera_tf_timeout_sec")),
            )
        except TransformException as exc:
            raise MissionError(
                f"URDF transform {gripper_frame} -> {ee_frame} failed: {exc}"
            ) from exc

        gripper_to_ee = Pose()
        gripper_to_ee.position.x = transform.transform.translation.x
        gripper_to_ee.position.y = transform.transform.translation.y
        gripper_to_ee.position.z = transform.transform.translation.z
        gripper_to_ee.orientation = transform.transform.rotation

        ee_target = PoseStamped()
        ee_target.header = gripper_target.header
        ee_target.pose = compose_poses(gripper_target.pose, gripper_to_ee)
        values = pose_to_array(ee_target.pose)
        self.get_logger().info(
            f"applied URDF {gripper_frame} -> {ee_frame}: "
            f"xyz=[{transform.transform.translation.x:.4f}, "
            f"{transform.transform.translation.y:.4f}, "
            f"{transform.transform.translation.z:.4f}], "
            f"arm target=[{values[0]:.4f}, {values[1]:.4f}, "
            f"{values[2]:.4f}]"
        )
        return ee_target

    def _apply_grasp_pose_correction(self, pose_stamped: PoseStamped) -> PoseStamped:
        values = pose_to_array(pose_stamped.pose)
        correction = self._quaternion_from_rpy(
            *self._float_array("grasp_pose_correction_rpy")
        )
        orientation = quaternion_multiply(tuple(values[3:]), correction)
        orientation_norm = math.sqrt(sum(value * value for value in orientation))

        corrected = PoseStamped()
        corrected.header = pose_stamped.header
        corrected.pose.position.x = values[0]
        corrected.pose.position.y = values[1]
        corrected.pose.position.z = values[2]
        corrected.pose.orientation.x = orientation[0] / orientation_norm
        corrected.pose.orientation.y = orientation[1] / orientation_norm
        corrected.pose.orientation.z = orientation[2] / orientation_norm
        corrected.pose.orientation.w = orientation[3] / orientation_norm
        return corrected

    @staticmethod
    def _point(x: float, y: float, z: float) -> Point:
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        return point

    def _pose_markers(
        self,
        pose_stamped: PoseStamped,
        label: str,
        marker_id_base: int,
        sphere_color: tuple[float, float, float],
    ) -> list[Marker]:
        values = pose_to_array(pose_stamped.pose)
        position = tuple(values[:3])
        orientation = tuple(values[3:])
        header = pose_stamped.header
        header.stamp = self.get_clock().now().to_msg()

        sphere = Marker()
        sphere.header = header
        sphere.ns = label
        sphere.id = marker_id_base
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = self._point(*position)
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.04
        sphere.color.r, sphere.color.g, sphere.color.b = sphere_color
        sphere.color.a = 0.9

        markers = [sphere]
        axis_length = 0.12
        axis_colors = (
            (1.0, 0.1, 0.1),
            (0.1, 1.0, 0.1),
            (0.1, 0.35, 1.0),
        )
        for axis_index, (axis, color) in enumerate(
            zip(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), axis_colors)
        ):
            direction = rotate_vector(axis, orientation)
            line = Marker()
            line.header = header
            line.ns = f"{label}_axes"
            line.id = marker_id_base + axis_index + 1
            line.type = Marker.LINE_LIST
            line.action = Marker.ADD
            line.scale.x = 0.012
            line.color.r, line.color.g, line.color.b = color
            line.color.a = 1.0
            line.points = [
                self._point(*position),
                self._point(
                    position[0] + axis_length * direction[0],
                    position[1] + axis_length * direction[1],
                    position[2] + axis_length * direction[2],
                ),
            ]
            markers.append(line)

        text = Marker()
        text.header = header
        text.ns = f"{label}_label"
        text.id = marker_id_base + 4
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = self._point(
            position[0], position[1], position[2] + 0.08
        )
        text.pose.orientation.w = 1.0
        text.scale.z = 0.035
        text.color.r = text.color.g = text.color.b = 1.0
        text.color.a = 1.0
        text.text = (
            f"{label}\n"
            f"x={position[0]:.3f} y={position[1]:.3f} z={position[2]:.3f}"
        )
        markers.append(text)
        return markers

    def _gripper_mesh_marker(
        self,
        pose_stamped: PoseStamped,
        mesh_frame: str,
        marker_id: int,
        namespace: Optional[str] = None,
        color: tuple[float, float, float] = (0.1, 1.0, 0.2),
        alpha: float = 0.65,
    ) -> Marker:
        marker = Marker()
        marker.header = pose_stamped.header
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace or f"{mesh_frame}_target_mesh"
        marker.id = marker_id
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose = pose_stamped.pose
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = alpha
        marker.mesh_resource = (
            "package://r1_pro_with_gripper/meshes/"
            f"{mesh_frame}.STL"
        )
        marker.mesh_use_embedded_materials = False
        return marker

    def _gripper_mesh_markers(
        self,
        pose_stamped: PoseStamped,
        gripper_frame: str,
        marker_id_base: int,
        namespace: str,
        color: tuple[float, float, float],
        alpha: float,
    ) -> list[Marker]:
        markers = [
            self._gripper_mesh_marker(
                pose_stamped,
                gripper_frame,
                marker_id_base,
                namespace=namespace,
                color=color,
                alpha=alpha,
            )
        ]
        frame_prefix = gripper_frame.removesuffix("_gripper_link")
        for finger_index in (1, 2):
            finger_frame = (
                f"{frame_prefix}_gripper_finger_link{finger_index}"
            )
            try:
                transform = self.tf_buffer.lookup_transform(
                    gripper_frame,
                    finger_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.05),
                )
            except TransformException as exc:
                self.get_logger().debug(
                    f"cannot render {finger_frame} target mesh yet: {exc}"
                )
                continue

            gripper_to_finger = Pose()
            gripper_to_finger.position.x = transform.transform.translation.x
            gripper_to_finger.position.y = transform.transform.translation.y
            gripper_to_finger.position.z = transform.transform.translation.z
            gripper_to_finger.orientation = transform.transform.rotation
            finger_pose = PoseStamped()
            finger_pose.header = pose_stamped.header
            finger_pose.pose = compose_poses(
                pose_stamped.pose, gripper_to_finger
            )
            markers.append(
                self._gripper_mesh_marker(
                    finger_pose,
                    finger_frame,
                    marker_id_base + finger_index,
                    namespace=f"{namespace}_finger{finger_index}",
                    color=color,
                    alpha=alpha,
                )
            )
        return markers

    def _publish_gripper_target_tf(self) -> None:
        if (
            self.latest_gripper_target_pose is None
            or self.latest_gripper_target_frame is None
        ):
            return
        transform = TransformStamped()
        transform.header = self.latest_gripper_target_pose.header
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.child_frame_id = (
            f"mission_target/{self.latest_gripper_target_frame}"
        )
        transform.transform.translation.x = (
            self.latest_gripper_target_pose.pose.position.x
        )
        transform.transform.translation.y = (
            self.latest_gripper_target_pose.pose.position.y
        )
        transform.transform.translation.z = (
            self.latest_gripper_target_pose.pose.position.z
        )
        transform.transform.rotation = (
            self.latest_gripper_target_pose.pose.orientation
        )
        self.target_tf_broadcaster.sendTransform(transform)

    def _publish_grasp_visualization(self) -> None:
        marker_array = MarkerArray()
        gripper_frame = self.latest_gripper_target_frame or "gripper_link"
        if self.latest_grasp_pose is not None:
            marker_array.markers.extend(
                self._pose_markers(
                    self.latest_grasp_pose,
                    "grasp_center",
                    0,
                    (1.0, 0.55, 0.05),
                )
            )
            # Render the corrected Graspness pose with the real gripper-link
            # geometry, but before grasp_to_gripper_rpy. This makes convention
            # errors (for example a 90-degree approach-axis mismatch) visible.
            marker_array.markers.extend(
                self._gripper_mesh_markers(
                    self.latest_grasp_pose,
                    gripper_frame,
                    5,
                    "grasp_pose_gripper_mesh",
                    color=(1.0, 0.55, 0.05),
                    alpha=0.55,
                )
            )
        if self.latest_gripper_target_pose is not None:
            marker_array.markers.extend(
                self._pose_markers(
                    self.latest_gripper_target_pose,
                    f"{gripper_frame}_target",
                    10,
                    (0.1, 1.0, 0.2),
                )
            )
            marker_array.markers.extend(
                self._gripper_mesh_markers(
                    self.latest_gripper_target_pose,
                    gripper_frame,
                    15,
                    f"{gripper_frame}_target_mesh",
                    color=(0.1, 1.0, 0.2),
                    alpha=0.65,
                )
            )
        if self.latest_arm_target_pose is not None:
            marker_array.markers.extend(
                self._pose_markers(
                    self.latest_arm_target_pose,
                    "arm_link7_target",
                    20,
                    (0.0, 0.9, 1.0),
                )
            )
        if self.latest_arm_intermediate_pose is not None:
            marker_array.markers.extend(
                self._pose_markers(
                    self.latest_arm_intermediate_pose,
                    "arm_link7_intermediate",
                    30,
                    (1.0, 0.2, 0.9),
                )
            )
        if marker_array.markers:
            self.grasp_visualization_publisher.publish(marker_array)
        self._publish_gripper_target_tf()

    def _republish_grasp_visualization(self) -> None:
        if self.latest_grasp_pose_camera is not None:
            self.grasp_pose_camera_publisher.publish(
                self.latest_grasp_pose_camera
            )
        if self.latest_grasp_pose_ee is not None:
            self.grasp_pose_ee_publisher.publish(self.latest_grasp_pose_ee)
        if self.latest_grasp_pose is not None:
            self.grasp_pose_publisher.publish(self.latest_grasp_pose)
        if self.latest_gripper_target_pose is not None:
            self.gripper_target_pose_publisher.publish(
                self.latest_gripper_target_pose
            )
        if self.latest_arm_target_pose is not None:
            self.arm_target_pose_publisher.publish(self.latest_arm_target_pose)
        if self.latest_arm_intermediate_pose is not None:
            self.arm_intermediate_pose_publisher.publish(
                self.latest_arm_intermediate_pose
            )
        self._publish_grasp_visualization()

    def _preview_grasp_pose_callback(self, pose: PoseStamped) -> None:
        try:
            corrected_pose = self._apply_grasp_pose_correction(pose)
            self.latest_grasp_pose_camera = corrected_pose
            self.grasp_pose_camera_publisher.publish(corrected_pose)
            self._prepare_grasp_target(
                corrected_pose, self._string("preview_arm").lower()
            )
        except (MissionError, ValueError) as exc:
            self.get_logger().warning(f"grasp preview failed: {exc}")

    def _prepare_grasp_target(
        self, grasp_pose_camera: PoseStamped, arm: str
    ) -> tuple[PoseStamped, PoseStamped]:
        ee_frame = self._string(
            "left_ee_frame" if arm == "left" else "right_ee_frame"
        ).lstrip("/")
        gripper_frame = self._string(
            "left_gripper_frame" if arm == "left" else "right_gripper_frame"
        ).lstrip("/")
        execution_frame = self._string("arm_execution_frame").lstrip("/")

        # Retain this diagnostic topic so the camera-to-robot calibration can
        # be inspected at the actual initial joint state.
        grasp_pose_ee = self._transform_detection_pose(
            grasp_pose_camera, ee_frame
        )
        self.latest_grasp_pose_ee = grasp_pose_ee
        self.grasp_pose_ee_publisher.publish(grasp_pose_ee)

        grasp_pose_execution = self._transform_detection_pose(
            grasp_pose_camera, execution_frame
        )
        grasp_pose_execution, normalized_gripper_pose = (
            self._normalize_grasp_symmetry(
                grasp_pose_execution, execution_frame, gripper_frame
            )
        )
        gripper_target_execution = PoseStamped()
        gripper_target_execution.header = grasp_pose_execution.header
        gripper_target_execution.pose = normalized_gripper_pose
        arm_target_execution = self._gripper_target_to_ee_target(
            gripper_target_execution, gripper_frame, ee_frame
        )

        self.latest_grasp_pose = grasp_pose_execution
        self.latest_gripper_target_pose = gripper_target_execution
        self.latest_gripper_target_frame = gripper_frame
        self.latest_arm_target_pose = arm_target_execution
        self.grasp_pose_publisher.publish(grasp_pose_execution)
        self.gripper_target_pose_publisher.publish(gripper_target_execution)
        self.arm_target_pose_publisher.publish(arm_target_execution)
        self._publish_grasp_visualization()
        self.get_logger().info(
            "prepared move_arm_p target in mission: "
            f"camera={grasp_pose_camera.header.frame_id} -> "
            f"grasp center in {execution_frame} -> {gripper_frame} target "
            f"-> URDF {ee_frame} target"
        )
        return grasp_pose_execution, arm_target_execution

    def _publish_camera_mount_tf(self) -> None:
        if not self._boolean("camera_mount_tf_enabled"):
            self.get_logger().info("camera mount TF publication disabled")
            return

        xyz = self._float_array("camera_mount_xyz")
        rpy = self._float_array("camera_mount_rpy")
        mount_quaternion = self._quaternion_from_rpy(*rpy)
        correction_rpy = self._float_array("camera_mount_correction_rpy")
        correction_quaternion = self._quaternion_from_rpy(*correction_rpy)
        # The correction is expressed in the parent/gripper frame, so it must
        # pre-multiply the existing camera mount rotation.
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
        self.get_logger().info(
            "published mission-owned camera mount TF "
            f"{transform.header.frame_id} -> {transform.child_frame_id}; "
            f"parent-frame correction_rpy={correction_rpy}"
        )

    def _transform_detection_pose(self, pose: PoseStamped, target_frame: str) -> PoseStamped:
        source_frame = pose.header.frame_id.strip().lstrip("/")
        target_frame = target_frame.strip().lstrip("/")
        if not source_frame:
            raise MissionError("grasp detector returned an empty source frame")
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
                f"camera pose transform {source_frame} -> {target_frame} failed: {exc}"
            ) from exc

        transformed.header.frame_id = target_frame
        transformed.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(
            f"converted grasp pose {source_frame} -> {target_frame} using current robot TF/joint state"
        )
        return transformed

    def _call_detect(
        self,
        goal_handle,
        request,
        *,
        client=None,
        service_name: Optional[str] = None,
    ):
        client = client or self.detect_client
        service_name = service_name or self._string("detect_service_name")
        self._wait_for_service(client, service_name, goal_handle)
        detect_request = DetectGraspPose.Request()
        # Graspness is camera-local. Mission owns the robot/camera relationship
        # and performs the complete conversion after receiving this response.
        detect_request.target_frame = ""
        detect_request.target_label = int(request.target_label)
        requested_timeout_sec = (
            float(request.detection_timeout_sec)
            if request.detection_timeout_sec > 0.0
            else self._float("default_detection_timeout_sec")
        )
        detect_request.timeout_sec = max(
            requested_timeout_sec,
            self._float("grasp_detection_min_timeout_sec"),
        )
        response = self._wait_future(
            client.call_async(detect_request),
            goal_handle,
            f"calling {service_name}",
            detect_request.timeout_sec + self._float("dependency_wait_timeout_sec"),
            cancel_local_future=True,
        )
        if not response.success:
            raise MissionError(f"grasp detection failed: {response.message}")
        candidate_poses = list(
            getattr(response, "candidate_poses", [])
        ) or [response.grasp_pose]

        def metadata(values, index: int, fallback):
            return values[index] if index < len(values) else fallback

        candidate_scores = list(getattr(response, "candidate_scores", []))
        candidate_widths = list(getattr(response, "candidate_widths", []))
        candidate_heights = list(getattr(response, "candidate_heights", []))
        candidate_depths = list(getattr(response, "candidate_depths", []))
        candidate_object_ids = list(
            getattr(response, "candidate_object_ids", [])
        )
        candidates: list[GraspCandidate] = []
        for index, pose in enumerate(candidate_poses):
            corrected_pose = self._apply_grasp_pose_correction(pose)
            candidates.append(
                GraspCandidate(
                    pose=corrected_pose,
                    score=float(
                        metadata(candidate_scores, index, response.score)
                    ),
                    width=float(
                        metadata(candidate_widths, index, response.width)
                    ),
                    height=float(
                        metadata(candidate_heights, index, response.height)
                    ),
                    depth=float(
                        metadata(candidate_depths, index, response.depth)
                    ),
                    object_id=int(
                        metadata(
                            candidate_object_ids,
                            index,
                            response.object_id,
                        )
                    ),
                )
            )
        if not candidates:
            raise MissionError("grasp detector returned no candidates")
        self.latest_grasp_pose_camera = candidates[0].pose
        self.grasp_pose_camera_publisher.publish(candidates[0].pose)
        return candidates

    def _forward_arm_feedback(self, goal_handle, arm: str, feedback_message) -> None:
        feedback = feedback_message.feedback
        detail = f"{feedback.detail} (progress={feedback.progress:.0%})"
        self._publish_grasp_feedback(
            goal_handle, f"ARM_{feedback.stage}", detail, arm
        )

    def _call_arm_pose(
        self, goal_handle, arm: str, pose: Pose, dry_run: bool
    ) -> str:
        action_name = self._string("arm_pose_action_name")
        self._wait_for_action_server(goal_handle)
        pose_values = pose_to_array(pose)
        arm_goal = MoveArmPose.Goal()
        arm_goal.left_pose = pose_values if arm == "left" else []
        arm_goal.right_pose = pose_values if arm == "right" else []
        arm_goal.dry_run = dry_run

        send_future = self.arm_pose_client.send_goal_async(
            arm_goal,
            feedback_callback=lambda message: self._forward_arm_feedback(
                goal_handle, arm, message
            ),
        )
        arm_goal_handle = self._wait_future(
            send_future,
            goal_handle,
            f"sending {action_name} goal",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        if arm_goal_handle is None or not arm_goal_handle.accepted:
            raise MissionError(f"{action_name} goal was rejected")

        with self.state_lock:
            self.active_arm_goal_handle = arm_goal_handle
        if goal_handle.is_cancel_requested:
            arm_goal_handle.cancel_goal_async()

        result_future = arm_goal_handle.get_result_async()
        deadline = time.monotonic() + self._float("arm_pose_result_timeout_sec")
        cancel_sent = False
        try:
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested and not cancel_sent:
                    arm_goal_handle.cancel_goal_async()
                    cancel_sent = True
                if time.monotonic() >= deadline:
                    arm_goal_handle.cancel_goal_async()
                    raise MissionError(
                        f"timeout waiting for {action_name} result after "
                        f"{self._float('arm_pose_result_timeout_sec'):.1f}s"
                    )
                time.sleep(0.05)
            if not rclpy.ok():
                raise MissionError(f"ROS shutdown while waiting for {action_name}")
            wrapped_result = result_future.result()
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, MissionError):
                raise
            raise MissionError(f"waiting for {action_name} result failed: {exc}") from exc
        finally:
            with self.state_lock:
                self.active_arm_goal_handle = None

        if goal_handle.is_cancel_requested:
            raise MissionCanceled(f"mission canceled during {action_name}")
        arm_result = wrapped_result.result
        arm_succeeded = wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
        arm_succeeded = arm_succeeded and arm_result.success
        if not arm_succeeded:
            raise MissionError(
                f"{action_name} failed: {arm_result.message} "
                f"(error_code={arm_result.error_code})"
            )
        return str(arm_result.message)

    def _current_arm_pose(self, arm: str) -> Pose:
        execution_frame = self._string("arm_execution_frame").lstrip("/")
        ee_frame = self._string(
            "left_ee_frame" if arm == "left" else "right_ee_frame"
        ).lstrip("/")
        try:
            transform = self.tf_buffer.lookup_transform(
                execution_frame,
                ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self._float("camera_tf_timeout_sec")),
            )
        except TransformException as exc:
            raise MissionError(
                f"current arm pose {execution_frame} <- {ee_frame} failed: {exc}"
            ) from exc

        pose = Pose()
        pose.position.x = transform.transform.translation.x
        pose.position.y = transform.transform.translation.y
        pose.position.z = transform.transform.translation.z
        pose.orientation = transform.transform.rotation
        pose_to_array(pose)
        return pose

    def _call_two_stage_grasp_pose(
        self, goal_handle, arm: str, target_pose: Pose, dry_run: bool
    ) -> str:
        current_pose = self._current_arm_pose(arm)
        intermediate_pose = interpolate_pose(current_pose, target_pose, 0.5)

        intermediate_stamped = PoseStamped()
        intermediate_stamped.header.stamp = self.get_clock().now().to_msg()
        intermediate_stamped.header.frame_id = self._string(
            "arm_execution_frame"
        ).lstrip("/")
        intermediate_stamped.pose = intermediate_pose
        self.latest_arm_intermediate_pose = intermediate_stamped
        self.arm_intermediate_pose_publisher.publish(intermediate_stamped)
        self._publish_grasp_visualization()

        self._publish_grasp_feedback(
            goal_handle,
            "EXECUTING_INTERMEDIATE_GRASP_POSE",
            "sending halfway arm pose (stage 1/2)",
            arm,
        )
        try:
            intermediate_message = self._call_arm_pose(
                goal_handle, arm, intermediate_pose, dry_run
            )
        except MissionCanceled:
            raise
        except MissionError as exc:
            intermediate_message = f"failed but continued: {exc}"
            self.get_logger().warning(
                "grasp stage 1/2 failed; continuing directly to the final "
                f"pose of the same candidate: {exc}"
            )
            self._publish_grasp_feedback(
                goal_handle,
                "INTERMEDIATE_GRASP_POSE_FAILED_CONTINUING",
                "halfway pose failed; directly trying the final pose of the "
                "same candidate",
                arm,
            )
        self._check_canceled(goal_handle, "after intermediate grasp pose")

        self._publish_grasp_feedback(
            goal_handle,
            "EXECUTING_FINAL_GRASP_POSE",
            "sending final detected arm pose (stage 2/2)",
            arm,
        )
        try:
            final_message = self._call_arm_pose(
                goal_handle, arm, target_pose, dry_run
            )
        except MissionCanceled:
            raise
        except MissionError as exc:
            raise TwoStageMotionError(
                2,
                True,
                "grasp stage 2/2 failed after the intermediate pose was "
                f"attempted: {exc}",
            ) from exc
        return f"stage 1/2: {intermediate_message}; stage 2/2: {final_message}"

    def _recover_grasp_observation(
        self, goal_handle, arm: str, reason: str, dry_run: bool
    ) -> None:
        self._publish_grasp_feedback(
            goal_handle,
            "OPENING_GRIPPER_FOR_REDETECTION",
            f"{reason}; opening the {arm} gripper before returning to the "
            "observation posture",
            arm,
        )
        if not dry_run:
            self._open_grippers(
                goal_handle,
                (arm,),
                "while opening the selected gripper before re-detection",
            )
        self._publish_grasp_feedback(
            goal_handle,
            "RECOVERING_OBSERVATION",
            f"{reason}; returning directly to the final arm observation posture "
            "before re-detection",
            arm,
        )
        if not dry_run:
            # A failed grasp attempt leaves the arms on, or close to, the
            # Cartesian approach path.  Going through the broad intermediate
            # posture adds an unnecessary detour here; command the validated
            # final observation posture directly while restoring the torso.
            recovery_attempt = 0
            while True:
                recovery_attempt += 1
                try:
                    self._prepare_grasp_arms_and_torso(goal_handle)
                    return
                except MissionCanceled:
                    raise
                except MissionError as exc:
                    error_text = str(exc).lower()
                    transient_cancel = (
                        "trajectory execution canceled" in error_text
                        or "error_code=17" in error_text
                    )
                    if not transient_cancel:
                        raise
                    detail = (
                        "observation recovery /move_arm_j was canceled "
                        f"transiently (attempt {recovery_attempt}): {exc}; "
                        "retrying the same observation target"
                    )
                    self.get_logger().warning(detail)
                    self._publish_grasp_feedback(
                        goal_handle,
                        "RETRYING_OBSERVATION_RECOVERY",
                        detail,
                        arm,
                    )
                    self._wait_delay(
                        goal_handle,
                        self._float("grasp_recovery_retry_delay_sec"),
                        "before retrying canceled observation recovery",
                    )

    def _detect_and_execute_grasp(
        self, goal_handle, request, arm: str, motion_state: dict[str, bool]
    ) -> tuple[GraspCandidate, PoseStamped, str]:
        max_detection_attempts = self._integer("grasp_detection_attempts")
        unlimited_detection_retries = max_detection_attempts == 0
        candidates_per_detection = self._integer(
            "grasp_candidates_per_detection"
        )
        failures: list[str] = []
        detection_attempt = 0

        while True:
            detection_attempt += 1
            detection_attempt_label = (
                str(detection_attempt)
                if unlimited_detection_retries
                else f"{detection_attempt}/{max_detection_attempts}"
            )
            self._publish_grasp_feedback(
                goal_handle,
                "DETECTING",
                f"requesting grasp pose detection "
                f"(attempt {detection_attempt_label})",
                arm,
            )
            try:
                candidates = self._call_detect(goal_handle, request)[
                    :candidates_per_detection
                ]
            except MissionCanceled:
                raise
            except MissionError as exc:
                failure = f"detection {detection_attempt} failed: {exc}"
                failures.append(failure)
                failures[:] = failures[-20:]
                self.get_logger().warning(failure)
                if (
                    unlimited_detection_retries
                    or detection_attempt < max_detection_attempts
                ):
                    if not request.dry_run:
                        self._open_grippers(
                            goal_handle,
                            (arm,),
                            "while opening the selected gripper after a "
                            "failed detection",
                        )
                    self._publish_grasp_feedback(
                        goal_handle,
                        "REDETECTING",
                        "detection failed or timed out; requesting a fresh "
                        "detection instead of aborting the grasp action",
                        arm,
                    )
                    continue
                break
            for candidate_index, candidate in enumerate(candidates, start=1):
                self._check_canceled(goal_handle, "before grasp candidate execution")
                try:
                    grasp_pose_execution, arm_target_stamped = (
                        self._prepare_grasp_target(candidate.pose, arm)
                    )
                except MissionError as exc:
                    failure = (
                        f"detection {detection_attempt} candidate "
                        f"{candidate_index} target preparation failed: {exc}"
                    )
                    failures.append(failure)
                    failures[:] = failures[-20:]
                    self.get_logger().warning(failure)
                    continue
                self._publish_grasp_feedback(
                    goal_handle,
                    "EXECUTING_GRASP_CANDIDATE",
                    f"detection {detection_attempt_label}, "
                    f"candidate {candidate_index}/{len(candidates)}, "
                    f"score={candidate.score:.4f}",
                    arm,
                )
                try:
                    motion_state["started"] = True
                    message = self._call_two_stage_grasp_pose(
                        goal_handle,
                        arm,
                        arm_target_stamped.pose,
                        request.dry_run,
                    )
                    return candidate, grasp_pose_execution, message
                except MissionCanceled:
                    raise
                except TwoStageMotionError as exc:
                    failure = (
                        f"detection {detection_attempt} candidate "
                        f"{candidate_index} failed: {exc}"
                    )
                    failures.append(failure)
                    failures[:] = failures[-20:]
                    self.get_logger().warning(failure)

                    self._recover_grasp_observation(
                        goal_handle,
                        arm,
                        "candidate final pose failed after its intermediate "
                        "pose was attempted",
                        request.dry_run,
                    )

                    if candidate_index < len(candidates):
                        self._publish_grasp_feedback(
                            goal_handle,
                            "TRYING_NEXT_CANDIDATE",
                            "candidate final pose failed; observation posture "
                            "restored, trying the next-ranked candidate",
                            arm,
                        )

            if (
                unlimited_detection_retries
                or detection_attempt < max_detection_attempts
            ):
                if not request.dry_run:
                    self._open_grippers(
                        goal_handle,
                        (arm,),
                        "while opening the selected gripper before a fresh "
                        "detection",
                    )
                self._publish_grasp_feedback(
                    goal_handle,
                    "REDETECTING",
                    "available candidates failed; capturing a fresh detection",
                    arm,
                )
                continue
            break

        raise MissionError(
            "grasp execution exhausted detection/candidate retries: "
            + " | ".join(failures)
        )
