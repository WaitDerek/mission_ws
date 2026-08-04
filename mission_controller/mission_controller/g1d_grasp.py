import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseStamped
from mission_interfaces.action import ExecuteGrasp
from object_pose_interfaces.action import EstimateObjectPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from task_interfaces.action import MoveArmJoints, MoveArmPose
from visualization_msgs.msg import Marker, MarkerArray

from .common import (
    MissionCanceled,
    MissionError,
    compose_poses,
    pose_from_transform,
    pose_to_array,
    rotate_vector,
)


LEFT_JOINT_WAYPOINTS = [
    [
        0.4035448133945465,
        0.08892294764518738,
        0.9569523930549622,
        -0.20204205811023712,
        1.1267688274383545,
        0.466677725315094,
        -1.609840989112854,
    ]
]

OBJECT_TO_TARGET_MATRIX = [
    0.0,
    1.0,
    0.0,
    -0.05,
    0.0,
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def compute_target_poses(
    ee_pose: Pose,
    camera_object_pose: Pose,
    ee_to_camera: list[float],
    object_to_target: list[float],
) -> tuple[Pose, Pose, Pose]:
    """Return camera, raw badge, and target poses in the torso frame."""
    camera_pose = compose_poses(ee_pose, pose_from_transform(ee_to_camera))
    badge_pose = compose_poses(camera_pose, camera_object_pose)
    target_pose = compose_poses(
        badge_pose, pose_from_transform(object_to_target)
    )
    return camera_pose, badge_pose, target_pose


class G1DGraspController(Node):
    """Run the G1-D badge tracking sequence behind /execute_grasp."""

    def __init__(self) -> None:
        super().__init__("mission_controller")
        self._declare_parameters()
        self._handeye_matrix = self._load_handeye_matrix()
        self._validate_parameters()

        self._callback_group = ReentrantCallbackGroup()
        self._state_lock = threading.Lock()
        self._latest_ee_pose: Optional[PoseStamped] = None
        self._ee_pose_sequence = 0
        self._active_child_handles: dict[str, object] = {}
        self._active = False

        visualization_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.ee_pose_subscription = self.create_subscription(
            PoseStamped,
            self._string("ee_pose_topic"),
            self._ee_pose_callback,
            10,
            callback_group=self._callback_group,
        )
        self.object_pose_camera_publisher = self.create_publisher(
            PoseStamped,
            self._string("object_pose_camera_topic"),
            visualization_qos,
        )
        self.object_pose_publisher = self.create_publisher(
            PoseStamped,
            self._string("object_pose_topic"),
            visualization_qos,
        )
        self.target_pose_publisher = self.create_publisher(
            PoseStamped,
            self._string("target_pose_topic"),
            visualization_qos,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            self._string("visualization_topic"),
            visualization_qos,
        )

        self.arm_joints_client = ActionClient(
            self,
            MoveArmJoints,
            self._string("move_arm_joints_action_name"),
            callback_group=self._callback_group,
        )
        self.arm_pose_client = ActionClient(
            self,
            MoveArmPose,
            self._string("move_arm_pose_action_name"),
            callback_group=self._callback_group,
        )
        self.object_pose_client = ActionClient(
            self,
            EstimateObjectPose,
            self._string("object_pose_action_name"),
            callback_group=self._callback_group,
        )
        self.action_server = ActionServer(
            self,
            ExecuteGrasp,
            self._string("execute_grasp_action_name"),
            execute_callback=self._execute_grasp,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            "G1-D mission ready: only %s; observation=LEFT_JOINT_WAYPOINTS, "
            "ee_topic=%s, object_action=%s, move_pose=%s"
            % (
                self._string("execute_grasp_action_name"),
                self._string("ee_pose_topic"),
                self._string("object_pose_action_name"),
                self._string("move_arm_pose_action_name"),
            )
        )

    def _declare_parameters(self) -> None:
        self.declare_parameters(
            namespace="",
            parameters=[
                ("execute_grasp_action_name", "/execute_grasp"),
                ("move_arm_joints_action_name", "/move_arm_j"),
                ("move_arm_pose_action_name", "/move_arm_p"),
                ("object_pose_action_name", "/object_pose/estimate"),
                ("object_model_label", "badge"),
                ("object_instance_index", 0),
                ("object_confidence_threshold", 0.0),
                ("ee_pose_topic", "/pinocchio_g1d/left_ee_pose"),
                ("execution_frame", "torso_link"),
                ("ee_frame", "left_gripper_base_link"),
                ("camera_frame", "camera_color_optical_frame"),
                ("object_pose_camera_topic", "/mission/badge_pose_camera"),
                ("object_pose_topic", "/mission/badge_pose"),
                ("target_pose_topic", "/mission/badge_target_pose"),
                ("visualization_topic", "/mission/grasp_visualization"),
                ("left_observation_joint_positions", LEFT_JOINT_WAYPOINTS[0]),
                ("right_observation_joint_positions", []),
                ("handeye_file", "handeye_result_12.yaml"),
                ("object_to_target_matrix", OBJECT_TO_TARGET_MATRIX),
                ("joint_motion_duration_sec", 5.0),
                ("dependency_wait_timeout_sec", 10.0),
                ("object_pose_timeout_sec", 120.0),
                ("arm_pose_timeout_sec", 120.0),
                ("ee_pose_timeout_sec", 10.0),
                ("restore_observation", True),
            ],
        )

    def _validate_parameters(self) -> None:
        for name in (
            "execute_grasp_action_name",
            "move_arm_joints_action_name",
            "move_arm_pose_action_name",
            "object_pose_action_name",
            "object_model_label",
            "ee_pose_topic",
            "execution_frame",
            "ee_frame",
            "camera_frame",
            "object_pose_camera_topic",
            "object_pose_topic",
            "target_pose_topic",
            "visualization_topic",
        ):
            if not self._string(name):
                raise ValueError(f"parameter '{name}' must not be empty")
        if len(self._float_array("left_observation_joint_positions")) != 7:
            raise ValueError("left_observation_joint_positions must contain 7 values")
        right = self._float_array("right_observation_joint_positions")
        if right and len(right) != 7:
            raise ValueError("right_observation_joint_positions must be empty or 7 values")
        if len(self._handeye_matrix) != 16:
            raise ValueError("handeye_file must contain a 4x4 ee_to_camera matrix")
        if len(self._float_array("object_to_target_matrix")) != 16:
            raise ValueError("object_to_target_matrix must contain 16 values")
        if self._float("joint_motion_duration_sec") <= 0.0:
            raise ValueError("joint_motion_duration_sec must be positive")

    def _load_handeye_matrix(self) -> list[float]:
        configured_path = Path(self._string("handeye_file")).expanduser()
        if not configured_path.is_absolute():
            configured_path = (
                Path(get_package_share_directory("mission_controller"))
                / "config"
                / configured_path
            )
        try:
            with configured_path.open("r", encoding="utf-8") as stream:
                document = yaml.safe_load(stream)
        except OSError as exc:
            raise ValueError(f"unable to read handeye_file '{configured_path}': {exc}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid handeye YAML '{configured_path}': {exc}") from exc

        if not isinstance(document, dict):
            raise ValueError(f"handeye_file '{configured_path}' must contain a YAML mapping")
        ee_to_camera = document.get("ee_to_camera")
        matrix = ee_to_camera.get("matrix") if isinstance(ee_to_camera, dict) else None
        if not isinstance(matrix, list) or len(matrix) != 4:
            raise ValueError(
                f"handeye_file '{configured_path}' must define ee_to_camera.matrix as 4 rows"
            )
        if not all(isinstance(row, list) and len(row) == 4 for row in matrix):
            raise ValueError(
                f"handeye_file '{configured_path}' ee_to_camera.matrix must be 4x4"
            )
        try:
            flattened = [float(value) for row in matrix for value in row]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"handeye_file '{configured_path}' contains a non-numeric matrix value"
            ) from exc
        if len(flattened) != 16:
            raise ValueError(
                f"handeye_file '{configured_path}' ee_to_camera.matrix must be 4x4"
            )
        self.get_logger().info(
            f"loaded hand-eye calibration from config file '{self._string('handeye_file')}'"
        )
        return flattened

    def _string(self, name: str) -> str:
        return str(self.get_parameter(name).value).strip()

    def _integer(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _boolean(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _float_array(self, name: str) -> list[float]:
        try:
            value = self.get_parameter(name).value
        except ParameterUninitializedException:
            # ROS 2 cannot initialize an empty typed array from YAML.  An
            # omitted right-arm waypoint intentionally means "hold position".
            if name == "right_observation_joint_positions":
                return []
            raise
        return [float(item) for item in value]

    def _ee_pose_callback(self, message: PoseStamped) -> None:
        with self._state_lock:
            self._latest_ee_pose = message
            self._ee_pose_sequence += 1

    def _goal_callback(self, _goal) -> GoalResponse:
        with self._state_lock:
            if self._active:
                self.get_logger().warning("rejecting execute_grasp: another goal is active")
                return GoalResponse.REJECT
            self._active = True
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    @staticmethod
    def _feedback(goal_handle, stage: str, detail: str) -> None:
        feedback = ExecuteGrasp.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.arm = "left"
        goal_handle.publish_feedback(feedback)

    def _check_canceled(self, goal_handle, context: str) -> None:
        if goal_handle.is_cancel_requested:
            raise MissionCanceled(f"mission canceled {context}")

    def _wait_future(self, future, goal_handle, description: str, timeout: float):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            self._check_canceled(goal_handle, f"while {description}")
            if time.monotonic() >= deadline:
                future.cancel()
                raise MissionError(f"timeout while {description} after {timeout:.1f}s")
            time.sleep(0.02)
        if not rclpy.ok():
            raise MissionError(f"ROS shutdown while {description}")
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001
            raise MissionError(f"{description} failed: {exc}") from exc

    def _wait_for_server(self, client, name: str, goal_handle) -> None:
        timeout = self._float("dependency_wait_timeout_sec")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, f"while waiting for {name}")
            if client.wait_for_server(timeout_sec=0.5):
                return
        raise MissionError(f"timeout waiting for action server {name}")

    def _send_move_joints(
        self, goal_handle, left: list[float], right: list[float], dry_run: bool
    ) -> None:
        name = self._string("move_arm_joints_action_name")
        self._wait_for_server(self.arm_joints_client, name, goal_handle)
        request = MoveArmJoints.Goal()
        request.left_joints = left
        request.right_joints = right
        request.dry_run = dry_run
        request.duration = self._float("joint_motion_duration_sec")
        send_future = self.arm_joints_client.send_goal_async(request)
        child = self._wait_future(send_future, goal_handle, f"sending {name}", self._float("dependency_wait_timeout_sec"))
        if child is None or not child.accepted:
            raise MissionError(f"{name} goal was rejected")
        with self._state_lock:
            self._active_child_handles["move_arm_j"] = child
        try:
            result_future = child.get_result_async()
            wrapped = self._wait_future(
                result_future,
                goal_handle,
                f"waiting for {name} result",
                self._float("arm_pose_timeout_sec"),
            )
        finally:
            with self._state_lock:
                self._active_child_handles.pop("move_arm_j", None)
        result = wrapped.result
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
            raise MissionError(f"{name} failed: {result.message}")

    def _send_move_pose(self, goal_handle, target: Pose, dry_run: bool) -> None:
        name = self._string("move_arm_pose_action_name")
        self._wait_for_server(self.arm_pose_client, name, goal_handle)
        request = MoveArmPose.Goal()
        request.left_pose = pose_to_array(target)
        request.right_pose = []
        request.dry_run = dry_run
        send_future = self.arm_pose_client.send_goal_async(request)
        child = self._wait_future(send_future, goal_handle, f"sending {name}", self._float("dependency_wait_timeout_sec"))
        if child is None or not child.accepted:
            raise MissionError(f"{name} goal was rejected")
        with self._state_lock:
            self._active_child_handles["move_arm_p"] = child
        try:
            result_future = child.get_result_async()
            wrapped = self._wait_future(
                result_future,
                goal_handle,
                f"waiting for {name} result",
                self._float("arm_pose_timeout_sec"),
            )
        finally:
            with self._state_lock:
                self._active_child_handles.pop("move_arm_p", None)
        result = wrapped.result
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
            raise MissionError(f"{name} failed: {result.message}")

    def _wait_for_ee_pose(
        self, goal_handle, sequence_before: int, require_fresh: bool = True
    ) -> PoseStamped:
        deadline = time.monotonic() + self._float("ee_pose_timeout_sec")
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for Pinocchio EE pose")
            with self._state_lock:
                pose = self._latest_ee_pose
                sequence = self._ee_pose_sequence
            if pose is not None and (not require_fresh or sequence > sequence_before):
                return pose
            time.sleep(0.02)
        if require_fresh:
            detail = "no fresh Pinocchio pose"
        else:
            detail = "no Pinocchio pose"
        raise MissionError(
            f"{detail} on {self._string('ee_pose_topic')} after "
            f"{self._float('ee_pose_timeout_sec'):.1f}s"
        )

    def _estimate_badge(self, goal_handle):
        name = self._string("object_pose_action_name")
        self._wait_for_server(self.object_pose_client, name, goal_handle)
        request = EstimateObjectPose.Goal()
        request.model_label = self._string("object_model_label")
        request.instance_index = self._integer("object_instance_index")
        request.confidence_threshold = self._float("object_confidence_threshold")
        send_future = self.object_pose_client.send_goal_async(
            request,
            feedback_callback=lambda message: self._feedback(
                goal_handle,
                "DETECTING_" + str(message.feedback.stage),
                f"FoundationPose progress={message.feedback.progress:.0%}",
            ),
        )
        child = self._wait_future(send_future, goal_handle, f"sending {name}", self._float("dependency_wait_timeout_sec"))
        if child is None or not child.accepted:
            raise MissionError(f"{name} goal was rejected")
        with self._state_lock:
            self._active_child_handles["object_pose"] = child
        try:
            wrapped = self._wait_future(
                child.get_result_async(),
                goal_handle,
                f"waiting for {name} result",
                self._float("object_pose_timeout_sec"),
            )
        finally:
            with self._state_lock:
                self._active_child_handles.pop("object_pose", None)
        result = wrapped.result
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
            raise MissionError(f"{name} failed: {result.message}")
        if result.pose.header.frame_id and result.pose.header.frame_id.strip("/") != self._string("camera_frame").strip("/"):
            self.get_logger().warning(
                f"FoundationPose frame '{result.pose.header.frame_id}' differs from "
                f"configured camera frame '{self._string('camera_frame')}'; using configured frame"
            )
        return result

    def _publish_poses(
        self,
        ee_pose: PoseStamped,
        camera_pose: Pose,
        badge_pose: Pose,
        target_pose: Pose,
    ) -> None:
        frame = self._string("execution_frame")
        camera_message = PoseStamped()
        camera_message.header.frame_id = frame
        camera_message.header.stamp = self.get_clock().now().to_msg()
        camera_message.pose = camera_pose
        badge_message = PoseStamped()
        badge_message.header = camera_message.header
        badge_message.pose = badge_pose
        target_message = PoseStamped()
        target_message.header = camera_message.header
        target_message.pose = target_pose
        self.object_pose_publisher.publish(badge_message)
        self.target_pose_publisher.publish(target_message)
        self._publish_markers(ee_pose.pose, camera_pose, badge_pose, target_pose)

    def _publish_raw_pose(self, pose: PoseStamped) -> None:
        message = PoseStamped()
        message.header = pose.header
        if not message.header.frame_id:
            message.header.frame_id = self._string("camera_frame")
        message.pose = pose.pose
        self.object_pose_camera_publisher.publish(message)

    def _publish_markers(self, ee: Pose, camera: Pose, badge: Pose, target: Pose) -> None:
        def point(x: float, y: float, z: float) -> Point:
            value = Point()
            value.x, value.y, value.z = x, y, z
            return value

        frame = self._string("execution_frame")
        marker_array = MarkerArray()
        clear = Marker()
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)
        marker_id = 1
        for name, pose in (
            (self._string("ee_frame"), ee),
            (self._string("camera_frame"), camera),
            ("badge_preview_frame", badge),
            ("badge_target_preview_frame", target),
        ):
            origin = point(pose.position.x, pose.position.y, pose.position.z)
            quaternion = (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            axes = ((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0, 0.0, 1.0))
            for axis_x, axis_y, axis_z, red, green, blue in axes:
                direction = rotate_vector((axis_x, axis_y, axis_z), quaternion)
                marker = Marker()
                marker.header.frame_id = frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "g1d_grasp"
                marker.id = marker_id
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                marker.scale.x = 0.006
                marker.scale.y = 0.012
                marker.scale.z = 0.018
                marker.color.r = red
                marker.color.g = green
                marker.color.b = blue
                marker.color.a = 1.0
                marker.points = [
                    origin,
                    point(
                        origin.x + 0.10 * direction[0],
                        origin.y + 0.10 * direction[1],
                        origin.z + 0.10 * direction[2],
                    ),
                ]
                marker_array.markers.append(marker)
                marker_id += 1
            label = Marker()
            label.header.frame_id = frame
            label.header.stamp = self.get_clock().now().to_msg()
            label.ns = "g1d_grasp_labels"
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = origin
            label.pose.position.z += 0.025
            label.pose.orientation.w = 1.0
            label.scale.z = 0.035
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = name
            marker_array.markers.append(label)
            marker_id += 1
        self.marker_publisher.publish(marker_array)

    def _cancel_children(self) -> None:
        with self._state_lock:
            handles = list(self._active_child_handles.values())
        for handle in handles:
            try:
                handle.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass

    def _execute_grasp(self, goal_handle) -> ExecuteGrasp.Result:
        result = ExecuteGrasp.Result()
        result.arm = "left"
        started = time.monotonic()
        observation_commanded = False
        restore_error: Optional[str] = None
        try:
            request = goal_handle.request
            dry_run = bool(request.dry_run)
            if request.arm and request.arm.strip().lower() != "left":
                self.get_logger().warning(
                    f"execute_grasp arm='{request.arm}' ignored; G1-D tracking uses the left arm"
                )
            self._feedback(goal_handle, "INITIALIZING", "preparing G1-D observation waypoint")
            with self._state_lock:
                sequence_before = self._ee_pose_sequence
            self._feedback(goal_handle, "MOVING_TO_OBSERVATION", "sending LEFT_JOINT_WAYPOINTS to /move_arm_j")
            self._send_move_joints(
                goal_handle,
                self._float_array("left_observation_joint_positions"),
                self._float_array("right_observation_joint_positions"),
                dry_run,
            )
            observation_commanded = not dry_run
            ee_pose = self._wait_for_ee_pose(
                goal_handle, sequence_before, require_fresh=not dry_run
            )
            self._feedback(goal_handle, "DETECTING_BADGE", "requesting the badge pose from /object_pose/estimate")
            detection = self._estimate_badge(goal_handle)
            self._publish_raw_pose(detection.pose)
            self._feedback(goal_handle, "CALCULATING_TARGET", "applying hand-eye and obj_T_tar to the left gripper target")
            camera_pose, badge_pose, target_pose = compute_target_poses(
                ee_pose.pose,
                detection.pose.pose,
                self._handeye_matrix,
                self._float_array("object_to_target_matrix"),
            )
            self._publish_poses(ee_pose, camera_pose, badge_pose, target_pose)
            target_message = PoseStamped()
            target_message.header.frame_id = self._string("execution_frame")
            target_message.header.stamp = self.get_clock().now().to_msg()
            target_message.pose = target_pose
            result.grasp_pose = target_message
            result.score = float(detection.detection_score)
            result.object_id = -1
            self._feedback(goal_handle, "EXECUTING_TARGET", "calling /move_arm_p for left_gripper_base_link")
            try:
                self._send_move_pose(goal_handle, target_pose, dry_run)
            finally:
                if observation_commanded and self._boolean("restore_observation"):
                    self._feedback(goal_handle, "RETURNING_TO_OBSERVATION", "returning to LEFT_JOINT_WAYPOINTS")
                    try:
                        self._send_move_joints(
                            goal_handle,
                            self._float_array("left_observation_joint_positions"),
                            self._float_array("right_observation_joint_positions"),
                            False,
                        )
                    except MissionCanceled:
                        raise
                    except MissionError as exc:
                        restore_error = str(exc)
            result.success = restore_error is None
            result.message = (
                "G1-D badge grasp flow completed"
                if restore_error is None
                else f"target executed but observation recovery failed: {restore_error}"
            )
            result.arm_message = (
                "observation -> badge detection -> obj_T_tar target -> move_arm_p -> observation"
            )
            result.torso_reset_command_published = False
            result.gripper_command_published = False
            if result.success:
                self._feedback(goal_handle, "DONE", result.message)
                goal_handle.succeed()
            else:
                self._feedback(goal_handle, "FAILED", result.message)
                goal_handle.abort()
            return result
        except MissionCanceled as exc:
            self._cancel_children()
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
            result.message = f"unexpected G1-D grasp error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            elapsed = time.monotonic() - started
            if result.message:
                result.message = f"{result.message} (elapsed_sec={elapsed:.3f})"
            with self._state_lock:
                self._active = False
                self._active_child_handles.clear()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = G1DGraspController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop callbacks before destroying entities.  This avoids a second
        # SIGINT interrupting rclpy's entity teardown during launch shutdown.
        try:
            executor.shutdown()
        except KeyboardInterrupt:
            pass
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()
