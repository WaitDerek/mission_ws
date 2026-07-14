import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, TwistStamped
from grasp_orchestrator_interfaces.srv import DetectGraspPose
from mission_interfaces.action import (
    ExecuteBinGrasp,
    ExecuteBinPlace,
    ExecuteGrasp,
    ExecutePlace,
)
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from task_interfaces.action import MoveArmPose
from task_interfaces.srv import ArmJoints, Home
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener


VALID_ARMS = {"left", "right"}


class MissionError(RuntimeError):
    pass


class MissionCanceled(MissionError):
    pass


def pose_to_array(pose: Pose) -> list[float]:
    values = [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]
    if not all(math.isfinite(value) for value in values):
        raise MissionError("grasp pose contains NaN or Inf")

    quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
    if quaternion_norm < 1e-8:
        raise MissionError("grasp pose quaternion has zero norm")
    values[3:] = [value / quaternion_norm for value in values[3:]]
    return values


class MissionController(Node):
    def __init__(self) -> None:
        super().__init__("mission_controller")
        self._declare_parameters()
        self._validate_parameters()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_camera_mount_tf()

        # Match the command transport used by dual_arm_manipulation/tools/r1pro_test.
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.torso_publisher = self.create_publisher(
            JointState, self._string("torso_topic"), command_qos
        )
        self.left_gripper_publisher = self.create_publisher(
            JointState, self._string("left_gripper_topic"), command_qos
        )
        self.right_gripper_publisher = self.create_publisher(
            JointState, self._string("right_gripper_topic"), command_qos
        )
        self.chassis_publisher = self.create_publisher(
            TwistStamped, self._string("chassis_topic"), command_qos
        )
        self.grasp_pose_publisher = self.create_publisher(
            PoseStamped, self._string("grasp_pose_topic"), 10
        )

        self.client_group = ReentrantCallbackGroup()
        self.server_group = ReentrantCallbackGroup()
        self.detect_client = self.create_client(
            DetectGraspPose,
            self._string("detect_service_name"),
            callback_group=self.client_group,
        )
        self.bin_detect_client = self.create_client(
            DetectGraspPose,
            self._string("bin_detect_service_name"),
            callback_group=self.client_group,
        )
        self.arm_joints_client = self.create_client(
            ArmJoints,
            self._string("arm_joints_service_name"),
            callback_group=self.client_group,
        )
        self.home_client = self.create_client(
            Home,
            self._string("home_service_name"),
            callback_group=self.client_group,
        )
        self.arm_pose_client = ActionClient(
            self,
            MoveArmPose,
            self._string("arm_pose_action_name"),
            callback_group=self.client_group,
        )

        self.state_lock = threading.Lock()
        self.mission_reserved = False
        self.active_mission = ""
        self.active_arm_goal_handle = None

        self.grasp_action_server = ActionServer(
            self,
            ExecuteGrasp,
            self._string("execute_grasp_action_name"),
            execute_callback=self._execute_grasp,
            goal_callback=self._grasp_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.place_action_server = ActionServer(
            self,
            ExecutePlace,
            self._string("execute_place_action_name"),
            execute_callback=self._execute_place,
            goal_callback=self._place_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.bin_grasp_action_server = ActionServer(
            self,
            ExecuteBinGrasp,
            self._string("execute_bin_grasp_action_name"),
            execute_callback=self._execute_bin_grasp,
            goal_callback=self._bin_grasp_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.bin_place_action_server = ActionServer(
            self,
            ExecuteBinPlace,
            self._string("execute_bin_place_action_name"),
            execute_callback=self._execute_bin_place,
            goal_callback=self._bin_place_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.get_logger().info(
            "mission controller ready: "
            f"grasp={self._string('execute_grasp_action_name')} "
            f"place={self._string('execute_place_action_name')} "
            f"bin_grasp={self._string('execute_bin_grasp_action_name')} "
            f"bin_place={self._string('execute_bin_place_action_name')}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameters(
            namespace="",
            parameters=[
                ("execute_grasp_action_name", "/execute_grasp"),
                ("execute_place_action_name", "/execute_place"),
                ("execute_bin_grasp_action_name", "/execute_bin_grasp"),
                ("execute_bin_place_action_name", "/execute_bin_place"),
                ("detect_service_name", "/detect_grasp_pose"),
                ("bin_detect_service_name", "/detect_bin_grasp_pose"),
                ("bin_mission_enabled", False),
                ("arm_pose_action_name", "/move_arm_p"),
                ("arm_joints_service_name", "/move_arm_j"),
                ("home_service_name", "/home"),
                ("grasp_pose_topic", "/mission/grasp_pose"),
                ("torso_topic", "/motion_target/target_joint_state_torso"),
                (
                    "left_gripper_topic",
                    "/motion_target/target_position_gripper_left",
                ),
                (
                    "right_gripper_topic",
                    "/motion_target/target_position_gripper_right",
                ),
                ("chassis_topic", "/motion_target/target_speed_chassis"),
                ("default_arm", "right"),
                ("default_target_frame", "torso_link4"),
                ("camera_mount_tf_enabled", True),
                ("camera_mount_parent_frame", "right_gripper_link"),
                ("camera_mount_child_frame", "hdas/camera_wrist_right_link"),
                ("camera_mount_xyz", [0.074703, 0.009582, 0.019134]),
                ("camera_mount_rpy", [2.4435, 0.0, -1.563]),
                ("camera_tf_timeout_sec", 2.0),
                ("default_detection_timeout_sec", 20.0),
                ("dependency_wait_timeout_sec", 10.0),
                ("arm_joints_result_timeout_sec", 60.0),
                ("arm_pose_result_timeout_sec", 120.0),
                ("home_result_timeout_sec", 60.0),
                ("wait_for_command_subscribers", True),
                ("require_command_subscribers", False),
                ("command_subscriber_wait_timeout_sec", 3.0),
                ("command_repeat_count", 10),
                ("command_repeat_interval_sec", 0.005),
                ("torso_settle_sec", 1.0),
                ("arm_settle_sec", 1.0),
                ("gripper_settle_sec", 1.0),
                ("torso_prepare_positions", [0.61, -0.81, -0.21, 0.0]),
                ("torso_reset_positions", [0.0, 0.0, 0.0, 0.0]),
                ("torso_velocities", [0.2, 0.2, 0.2, 0.2]),
                (
                    "grasp_left_joint_positions",
                    [-0.88, 0.84, -1.13, -1.80, 1.25, 0.29, 0.13],
                ),
                (
                    "grasp_right_joint_positions",
                    [-0.98, -0.64, 1.13, -1.60, -1.25, 0.6, -0.13],
                ),
                (
                    "place_right_joint_positions",
                    [-0.911, -0.270, 1.095, -1.193, -2.356, 0.801, -1.470],
                ),
                # Fill these bin-specific values before enabling bin missions.
                (
                    "bin_grasp_left_observation_joint_positions",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "bin_grasp_right_observation_joint_positions",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                ("bin_grasp_torso_prepare_positions", [0.0, 0.0, 0.0, 0.0]),
                ("bin_grasp_torso_lift_positions", [0.0, 0.0, 0.0, 0.0]),
                ("bin_place_torso_positions", [0.0, 0.0, 0.0, 0.0]),
                ("bin_place_chassis_distance_x", 0.0),
                ("bin_place_chassis_distance_y", 0.0),
                ("bin_place_chassis_yaw", 0.0),
                ("bin_place_chassis_duration_sec", 5.0),
                ("gripper_open_position", 100.0),
                ("gripper_closed_position", 0.0),
                ("open_gripper_before_grasp", True),
                ("place_chassis_distance_x", -0.5),
                ("place_chassis_distance_y", 0.0),
                ("place_chassis_yaw", 0.0),
                ("place_chassis_duration_sec", 5.0),
                ("chassis_publish_hz", 10.0),
                ("chassis_stop_repeat_count", 1),
                ("max_chassis_linear_speed", 0.2),
                ("max_chassis_angular_speed", 0.4),
                ("home_velocity", 0.05),
            ],
        )

    def _validate_parameters(self) -> None:
        for name in (
            "execute_grasp_action_name",
            "execute_place_action_name",
            "execute_bin_grasp_action_name",
            "execute_bin_place_action_name",
            "detect_service_name",
            "bin_detect_service_name",
            "arm_pose_action_name",
            "arm_joints_service_name",
            "home_service_name",
            "torso_topic",
            "left_gripper_topic",
            "right_gripper_topic",
            "chassis_topic",
            "default_target_frame",
        ):
            if not self._string(name):
                raise ValueError(f"parameter '{name}' must not be empty")

        if self._string("default_arm").lower() not in VALID_ARMS:
            raise ValueError("default_arm must be 'left' or 'right'")

        if self._boolean("camera_mount_tf_enabled"):
            for name in ("camera_mount_parent_frame", "camera_mount_child_frame"):
                if not self._string(name):
                    raise ValueError(f"parameter '{name}' must not be empty")

        for name, expected_length in (
            ("torso_prepare_positions", 4),
            ("torso_reset_positions", 4),
            ("torso_velocities", 4),
            ("grasp_left_joint_positions", 7),
            ("grasp_right_joint_positions", 7),
            ("place_right_joint_positions", 7),
            ("bin_grasp_left_observation_joint_positions", 7),
            ("bin_grasp_right_observation_joint_positions", 7),
            ("bin_grasp_torso_prepare_positions", 4),
            ("bin_grasp_torso_lift_positions", 4),
            ("bin_place_torso_positions", 4),
            ("camera_mount_xyz", 3),
            ("camera_mount_rpy", 3),
        ):
            values = self._float_array(name)
            if len(values) != expected_length:
                raise ValueError(
                    f"parameter '{name}' must contain {expected_length} values"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"parameter '{name}' contains NaN or Inf")

        for name in ("gripper_open_position", "gripper_closed_position"):
            value = self._float(name)
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"parameter '{name}' must be in [0, 100]")

        positive_parameters = (
            "default_detection_timeout_sec",
            "dependency_wait_timeout_sec",
            "arm_joints_result_timeout_sec",
            "arm_pose_result_timeout_sec",
            "home_result_timeout_sec",
            "command_subscriber_wait_timeout_sec",
            "place_chassis_duration_sec",
            "bin_place_chassis_duration_sec",
            "chassis_publish_hz",
            "max_chassis_linear_speed",
            "max_chassis_angular_speed",
            "home_velocity",
            "camera_tf_timeout_sec",
        )
        for name in positive_parameters:
            if not math.isfinite(self._float(name)) or self._float(name) <= 0.0:
                raise ValueError(f"parameter '{name}' must be finite and positive")

        nonnegative_parameters = (
            "command_repeat_interval_sec",
            "torso_settle_sec",
            "arm_settle_sec",
            "gripper_settle_sec",
        )
        for name in nonnegative_parameters:
            if not math.isfinite(self._float(name)) or self._float(name) < 0.0:
                raise ValueError(f"parameter '{name}' must be finite and nonnegative")

        if self._integer("command_repeat_count") <= 0:
            raise ValueError("command_repeat_count must be positive")
        if self._integer("chassis_stop_repeat_count") <= 0:
            raise ValueError("chassis_stop_repeat_count must be positive")

        duration = self._float("place_chassis_duration_sec")
        vx = self._float("place_chassis_distance_x") / duration
        vy = self._float("place_chassis_distance_y") / duration
        wz = self._float("place_chassis_yaw") / duration
        linear_speed = math.hypot(vx, vy)
        if linear_speed > self._float("max_chassis_linear_speed") + 1e-9:
            raise ValueError(
                "configured place chassis linear speed exceeds max_chassis_linear_speed"
            )
        if abs(wz) > self._float("max_chassis_angular_speed") + 1e-9:
            raise ValueError(
                "configured place chassis angular speed exceeds max_chassis_angular_speed"
            )

        bin_duration = self._float("bin_place_chassis_duration_sec")
        bin_vx = self._float("bin_place_chassis_distance_x") / bin_duration
        bin_vy = self._float("bin_place_chassis_distance_y") / bin_duration
        bin_wz = self._float("bin_place_chassis_yaw") / bin_duration
        if math.hypot(bin_vx, bin_vy) > self._float(
            "max_chassis_linear_speed"
        ) + 1e-9:
            raise ValueError(
                "configured bin place chassis linear speed exceeds "
                "max_chassis_linear_speed"
            )
        if abs(bin_wz) > self._float("max_chassis_angular_speed") + 1e-9:
            raise ValueError(
                "configured bin place chassis angular speed exceeds "
                "max_chassis_angular_speed"
            )

        if self._boolean("bin_mission_enabled"):
            for name in (
                "bin_grasp_left_observation_joint_positions",
                "bin_grasp_right_observation_joint_positions",
                "bin_grasp_torso_prepare_positions",
                "bin_grasp_torso_lift_positions",
                "bin_place_torso_positions",
            ):
                if all(abs(value) < 1e-9 for value in self._float_array(name)):
                    raise ValueError(
                        f"bin_mission_enabled requires configured '{name}'"
                    )

    def _string(self, name: str) -> str:
        return str(self.get_parameter(name).value).strip()

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _integer(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _boolean(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _float_array(self, name: str) -> list[float]:
        return [float(value) for value in self.get_parameter(name).value]

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

    def _publish_camera_mount_tf(self) -> None:
        if not self._boolean("camera_mount_tf_enabled"):
            self.get_logger().info("camera mount TF publication disabled")
            return

        xyz = self._float_array("camera_mount_xyz")
        rpy = self._float_array("camera_mount_rpy")
        qx, qy, qz, qw = self._quaternion_from_rpy(*rpy)
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
            f"{transform.header.frame_id} -> {transform.child_frame_id}"
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

    def _resolve_arm(self, requested_arm: str) -> str:
        arm = requested_arm.strip().lower()
        return arm or self._string("default_arm").lower()

    def _reserve_goal(self, mission: str, request_id: str, arm: str) -> GoalResponse:
        if arm not in VALID_ARMS:
            self.get_logger().error(
                f"rejecting {mission} goal: arm must be left or right, got '{arm}'"
            )
            return GoalResponse.REJECT
        with self.state_lock:
            if self.mission_reserved:
                self.get_logger().warning(
                    f"rejecting {mission} goal: {self.active_mission} mission is active"
                )
                return GoalResponse.REJECT
            self.mission_reserved = True
            self.active_mission = mission
        self.get_logger().info(
            f"accepted {mission} goal request_id={request_id or '<empty>'} arm={arm}"
        )
        return GoalResponse.ACCEPT

    def _grasp_goal_callback(self, request: ExecuteGrasp.Goal) -> GoalResponse:
        return self._reserve_goal(
            "grasp", request.request_id, self._resolve_arm(request.arm)
        )

    def _place_goal_callback(self, request: ExecutePlace.Goal) -> GoalResponse:
        return self._reserve_goal(
            "place", request.request_id, self._resolve_arm(request.arm)
        )

    def _bin_grasp_goal_callback(
        self, request: ExecuteBinGrasp.Goal
    ) -> GoalResponse:
        if not self._boolean("bin_mission_enabled"):
            self.get_logger().warning(
                "rejecting bin grasp goal: bin_mission_enabled is false; "
                "configure bin preparation targets and perception first"
            )
            return GoalResponse.REJECT
        return self._reserve_goal(
            "bin_grasp", request.request_id, self._resolve_arm(request.arm)
        )

    def _bin_place_goal_callback(
        self, request: ExecuteBinPlace.Goal
    ) -> GoalResponse:
        if not self._boolean("bin_mission_enabled"):
            self.get_logger().warning(
                "rejecting bin place goal: bin_mission_enabled is false; "
                "configure bin preparation targets first"
            )
            return GoalResponse.REJECT
        return self._reserve_goal(
            "bin_place", request.request_id, self._resolve_arm(request.arm)
        )

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        with self.state_lock:
            arm_goal_handle = self.active_arm_goal_handle
        if arm_goal_handle is not None:
            arm_goal_handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    def _release_goal(self) -> None:
        with self.state_lock:
            self.mission_reserved = False
            self.active_mission = ""
            self.active_arm_goal_handle = None

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

    @staticmethod
    def _publish_bin_grasp_feedback(
        goal_handle, stage: str, detail: str, arm: str
    ) -> None:
        feedback = ExecuteBinGrasp.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.arm = arm
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _publish_bin_place_feedback(
        goal_handle, stage: str, detail: str, arm: str
    ) -> None:
        feedback = ExecuteBinPlace.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.arm = arm
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _check_canceled(goal_handle, context: str) -> None:
        if goal_handle.is_cancel_requested:
            raise MissionCanceled(f"mission canceled {context}")

    def _wait_delay(self, goal_handle, duration_sec: float, context: str) -> None:
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, context)
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _wait_for_service(self, client, service_name: str, goal_handle) -> None:
        timeout_sec = self._float("dependency_wait_timeout_sec")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, f"while waiting for {service_name}")
            remaining = max(0.0, deadline - time.monotonic())
            if client.wait_for_service(timeout_sec=min(0.5, remaining)):
                return
        raise MissionError(
            f"timeout waiting for service {service_name} after {timeout_sec:.1f}s"
        )

    def _wait_for_action_server(self, goal_handle) -> None:
        action_name = self._string("arm_pose_action_name")
        timeout_sec = self._float("dependency_wait_timeout_sec")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, f"while waiting for {action_name}")
            remaining = max(0.0, deadline - time.monotonic())
            if self.arm_pose_client.wait_for_server(timeout_sec=min(0.5, remaining)):
                return
        raise MissionError(
            f"timeout waiting for action {action_name} after {timeout_sec:.1f}s"
        )

    def _wait_future(
        self,
        future,
        goal_handle,
        description: str,
        timeout_sec: float,
        cancel_local_future: bool,
    ):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if goal_handle.is_cancel_requested and cancel_local_future:
                future.cancel()
                raise MissionCanceled(f"mission canceled while {description}")
            if time.monotonic() >= deadline:
                future.cancel()
                raise MissionError(
                    f"timeout while {description} after {timeout_sec:.1f}s"
                )
            time.sleep(0.05)
        if not rclpy.ok():
            raise MissionError(f"ROS shutdown while {description}")
        try:
            value = future.result()
        except Exception as exc:  # noqa: BLE001
            raise MissionError(f"{description} failed: {exc}") from exc
        if not cancel_local_future:
            self._check_canceled(goal_handle, f"after {description}")
        return value

    def _wait_for_publisher(self, publisher, topic: str, goal_handle) -> None:
        if not self._boolean("wait_for_command_subscribers"):
            return
        timeout_sec = self._float("command_subscriber_wait_timeout_sec")
        deadline = time.monotonic() + timeout_sec
        while publisher.get_subscription_count() == 0:
            self._check_canceled(goal_handle, f"while waiting for subscriber on {topic}")
            if time.monotonic() >= deadline:
                detail = (
                    f"no subscriber matched on {topic} within {timeout_sec:.1f}s"
                )
                if self._boolean("require_command_subscribers"):
                    raise MissionError(detail)
                self.get_logger().warning(f"{detail}; publishing anyway")
                return
            time.sleep(0.05)

    def _publish_joint_command(
        self,
        publisher,
        topic: str,
        positions: list[float],
        velocities: list[float],
        goal_handle,
        *,
        require_subscriber: bool = True,
        honor_cancel: bool = True,
    ) -> None:
        if require_subscriber:
            self._wait_for_publisher(publisher, topic, goal_handle)
        message = JointState()
        message.position = positions
        message.velocity = velocities
        repeat_count = self._integer("command_repeat_count")
        interval_sec = self._float("command_repeat_interval_sec")
        for _ in range(repeat_count):
            if honor_cancel:
                self._check_canceled(goal_handle, f"while publishing {topic}")
            message.header.stamp = self.get_clock().now().to_msg()
            publisher.publish(message)
            if interval_sec > 0.0:
                time.sleep(interval_sec)

    def _publish_torso(
        self,
        goal_handle,
        positions: list[float],
        *,
        require_subscriber: bool = True,
        honor_cancel: bool = True,
    ) -> None:
        self._publish_joint_command(
            self.torso_publisher,
            self._string("torso_topic"),
            positions,
            self._float_array("torso_velocities"),
            goal_handle,
            require_subscriber=require_subscriber,
            honor_cancel=honor_cancel,
        )

    def _publish_gripper(self, goal_handle, arm: str, position: float) -> None:
        if arm == "left":
            publisher = self.left_gripper_publisher
            topic = self._string("left_gripper_topic")
        else:
            publisher = self.right_gripper_publisher
            topic = self._string("right_gripper_topic")
        self._publish_joint_command(
            publisher, topic, [position], [], goal_handle
        )

    def _prepare_grasp_grippers(self, goal_handle) -> None:
        for gripper_arm in ("left", "right"):
            self._publish_gripper(
                goal_handle,
                gripper_arm,
                self._float("gripper_open_position"),
            )
        self._wait_delay(
            goal_handle,
            self._float("gripper_settle_sec"),
            "while waiting for gripper preparation",
        )

    def _prepare_grasp_torso(self, goal_handle) -> None:
        self._publish_torso(
            goal_handle, self._float_array("torso_prepare_positions")
        )
        self._wait_delay(
            goal_handle,
            self._float("torso_settle_sec"),
            "while waiting for torso preparation",
        )

    def _prepare_grasp_arms(self, goal_handle) -> None:
        self._call_arm_joints(
            goal_handle,
            self._float_array("grasp_left_joint_positions"),
            self._float_array("grasp_right_joint_positions"),
            False,
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while waiting for arm preparation",
        )

    def _prepare_grasp_concurrently(
        self, goal_handle, open_grippers: bool
    ) -> None:
        tasks = [
            self._prepare_grasp_torso,
            self._prepare_grasp_arms,
        ]
        if open_grippers:
            tasks.insert(0, self._prepare_grasp_grippers)

        with ThreadPoolExecutor(
            max_workers=len(tasks), thread_name_prefix="grasp_prepare"
        ) as executor:
            futures = [executor.submit(task, goal_handle) for task in tasks]
            for future in futures:
                future.result()

    def _prepare_bin_grasp_grippers(self, goal_handle) -> None:
        for gripper_arm in ("left", "right"):
            self._publish_gripper(
                goal_handle,
                gripper_arm,
                self._float("gripper_open_position"),
            )
        self._wait_delay(
            goal_handle,
            self._float("gripper_settle_sec"),
            "while waiting for bin gripper preparation",
        )

    def _prepare_bin_grasp_torso(self, goal_handle) -> None:
        self._publish_torso(
            goal_handle,
            self._float_array("bin_grasp_torso_prepare_positions"),
        )
        self._wait_delay(
            goal_handle,
            self._float("torso_settle_sec"),
            "while waiting for bin torso preparation",
        )

    def _prepare_bin_grasp_arms(self, goal_handle) -> None:
        self._call_arm_joints(
            goal_handle,
            self._float_array("bin_grasp_left_observation_joint_positions"),
            self._float_array("bin_grasp_right_observation_joint_positions"),
            False,
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while waiting for bin arm preparation",
        )

    def _prepare_bin_grasp_concurrently(self, goal_handle) -> None:
        tasks = [
            self._prepare_bin_grasp_grippers,
            self._prepare_bin_grasp_torso,
            self._prepare_bin_grasp_arms,
        ]
        with ThreadPoolExecutor(
            max_workers=len(tasks), thread_name_prefix="bin_grasp_prepare"
        ) as executor:
            futures = [executor.submit(task, goal_handle) for task in tasks]
            for future in futures:
                future.result()

    def _make_chassis_message(
        self, linear_x: float, linear_y: float, angular_z: float
    ) -> TwistStamped:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.twist.linear.x = linear_x
        message.twist.linear.y = linear_y
        message.twist.linear.z = 0.0
        message.twist.angular.x = 0.0
        message.twist.angular.y = 0.0
        message.twist.angular.z = angular_z
        return message

    def _publish_zero_chassis(self) -> None:
        if not rclpy.ok():
            return
        repeat_count = self._integer("chassis_stop_repeat_count")
        interval_sec = self._float("command_repeat_interval_sec")
        for _ in range(repeat_count):
            if not rclpy.ok():
                return
            try:
                self.chassis_publisher.publish(
                    self._make_chassis_message(0.0, 0.0, 0.0)
                )
            except Exception:  # ROS context may become invalid during shutdown.
                return
            if interval_sec > 0.0:
                time.sleep(interval_sec)

    def _move_chassis(self, goal_handle, dry_run: bool) -> None:
        duration = self._float("place_chassis_duration_sec")
        distance_x = self._float("place_chassis_distance_x")
        distance_y = self._float("place_chassis_distance_y")
        yaw = self._float("place_chassis_yaw")
        if dry_run:
            return

        topic = self._string("chassis_topic")
        self._wait_for_publisher(self.chassis_publisher, topic, goal_handle)
        vx = distance_x / duration
        vy = distance_y / duration
        wz = yaw / duration
        period = 1.0 / self._float("chassis_publish_hz")
        deadline = time.monotonic() + duration
        try:
            while time.monotonic() < deadline:
                self._check_canceled(goal_handle, "during chassis motion")
                self.chassis_publisher.publish(self._make_chassis_message(vx, vy, wz))
                time.sleep(min(period, max(0.0, deadline - time.monotonic())))
        finally:
            self._publish_zero_chassis()

    def _move_bin_chassis(self, goal_handle, dry_run: bool) -> None:
        duration = self._float("bin_place_chassis_duration_sec")
        distance_x = self._float("bin_place_chassis_distance_x")
        distance_y = self._float("bin_place_chassis_distance_y")
        yaw = self._float("bin_place_chassis_yaw")
        if dry_run:
            return

        topic = self._string("chassis_topic")
        self._wait_for_publisher(self.chassis_publisher, topic, goal_handle)
        vx = distance_x / duration
        vy = distance_y / duration
        wz = yaw / duration
        period = 1.0 / self._float("chassis_publish_hz")
        deadline = time.monotonic() + duration
        try:
            while time.monotonic() < deadline:
                self._check_canceled(goal_handle, "during bin chassis motion")
                self.chassis_publisher.publish(self._make_chassis_message(vx, vy, wz))
                time.sleep(min(period, max(0.0, deadline - time.monotonic())))
        finally:
            self._publish_zero_chassis()

    def _call_arm_joints(
        self,
        goal_handle,
        left_positions: list[float],
        right_positions: list[float],
        dry_run: bool,
    ) -> str:
        if dry_run:
            return "dry run: skipped non-planning /move_arm_j service"
        service_name = self._string("arm_joints_service_name")
        self._wait_for_service(self.arm_joints_client, service_name, goal_handle)
        request = ArmJoints.Request()
        request.left_joints = left_positions
        request.right_joints = right_positions
        response = self._wait_future(
            self.arm_joints_client.call_async(request),
            goal_handle,
            f"calling {service_name}",
            self._float("arm_joints_result_timeout_sec"),
            cancel_local_future=False,
        )
        if not response.success:
            raise MissionError(f"{service_name} failed: {response.message}")
        return str(response.message)

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
        target_frame = (
            request.target_frame.strip() or self._string("default_target_frame")
        ).lstrip("/")
        # Graspness is camera-local. Mission owns the robot/camera relationship
        # and performs the conversion from the current robot TF state below.
        detect_request.target_frame = ""
        detect_request.target_label = int(request.target_label)
        detect_request.timeout_sec = (
            float(request.detection_timeout_sec)
            if request.detection_timeout_sec > 0.0
            else self._float("default_detection_timeout_sec")
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
        response.grasp_pose = self._transform_detection_pose(
            response.grasp_pose, target_frame
        )
        self.grasp_pose_publisher.publish(response.grasp_pose)
        return response

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

    def _call_home(self, goal_handle, dry_run: bool) -> str:
        service_name = self._string("home_service_name")
        self._wait_for_service(self.home_client, service_name, goal_handle)
        request = Home.Request()
        request.dry_run = dry_run
        request.velocity = self._float("home_velocity")
        response = self._wait_future(
            self.home_client.call_async(request),
            goal_handle,
            f"calling {service_name}",
            self._float("home_result_timeout_sec"),
            cancel_local_future=False,
        )
        if not response.success:
            raise MissionError(
                f"{service_name} failed: {response.message} "
                f"(error_code={response.error_code})"
            )
        return str(response.message)

    def _safe_pre_arm_torso_reset(self, goal_handle) -> bool:
        try:
            self._publish_torso(
                goal_handle,
                self._float_array("torso_reset_positions"),
                require_subscriber=False,
                honor_cancel=False,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"failed to publish torso cleanup command: {exc}")
            return False

    def _execute_bin_grasp(self, goal_handle) -> ExecuteBinGrasp.Result:
        request = goal_handle.request
        arm = self._resolve_arm(request.arm)
        result = ExecuteBinGrasp.Result()
        result.arm = arm
        torso_prepared = False
        joint_preparation_started = False
        joint_preparation_complete = False
        cartesian_motion_started = False
        completed = False

        try:
            self._publish_bin_grasp_feedback(
                goal_handle,
                "INITIALIZING",
                "preparing bin observation pose, torso, and grippers",
                arm,
            )
            if request.dry_run:
                self._publish_bin_grasp_feedback(
                    goal_handle,
                    "DRY_RUN_INITIALIZATION",
                    "skipping direct bin preparation commands",
                    arm,
                )
            else:
                self._publish_bin_grasp_feedback(
                    goal_handle,
                    "PREPARING_BIN_OBSERVATION",
                    "moving bin observation arms, torso, and grippers concurrently",
                    arm,
                )
                torso_prepared = True
                joint_preparation_started = True
                self._prepare_bin_grasp_concurrently(goal_handle)
                joint_preparation_complete = True

            self._publish_bin_grasp_feedback(
                goal_handle,
                "DETECTING_BIN",
                "requesting bin grasp pose detection",
                arm,
            )
            detection = self._call_detect(
                goal_handle,
                request,
                client=self.bin_detect_client,
                service_name=self._string("bin_detect_service_name"),
            )
            result.grasp_pose = detection.grasp_pose
            result.score = float(detection.score)
            result.width = float(detection.width)
            result.height = float(detection.height)
            result.depth = float(detection.depth)
            result.object_id = int(detection.object_id)
            if request.publish_pose:
                self.grasp_pose_publisher.publish(detection.grasp_pose)

            self._check_canceled(goal_handle, "after bin grasp detection")
            self._publish_bin_grasp_feedback(
                goal_handle,
                "EXECUTING_BIN_GRASP",
                f"sending detected bin pose to {self._string('arm_pose_action_name')}",
                arm,
            )
            cartesian_motion_started = True
            result.arm_message = self._call_arm_pose(
                goal_handle, arm, detection.grasp_pose.pose, request.dry_run
            )

            if request.dry_run:
                self._publish_bin_grasp_feedback(
                    goal_handle,
                    "DRY_RUN_COMPLETE",
                    "bin grasp plan succeeded; direct close and lift were skipped",
                    arm,
                )
            else:
                self._publish_bin_grasp_feedback(
                    goal_handle, "CLOSING_GRIPPER", "closing bin gripper", arm
                )
                self._publish_gripper(
                    goal_handle, arm, self._float("gripper_closed_position")
                )
                result.gripper_command_published = True
                self._wait_delay(
                    goal_handle,
                    self._float("gripper_settle_sec"),
                    "while waiting for bin gripper close",
                )

                self._publish_bin_grasp_feedback(
                    goal_handle,
                    "LIFTING_TORSO",
                    "lifting torso while keeping the arm at the grasp pose",
                    arm,
                )
                self._publish_torso(
                    goal_handle,
                    self._float_array("bin_grasp_torso_lift_positions"),
                )
                result.torso_lift_command_published = True
                self._wait_delay(
                    goal_handle,
                    self._float("torso_settle_sec"),
                    "while waiting for bin torso lift",
                )

            result.success = True
            result.message = (
                "bin grasp dry run completed"
                if request.dry_run
                else "bin grasp mission completed"
            )
            self._publish_bin_grasp_feedback(goal_handle, "DONE", result.message, arm)
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
            result.message = f"unexpected bin grasp mission error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            safe_to_reset = all(
                (
                    torso_prepared,
                    not cartesian_motion_started,
                    not joint_preparation_started or joint_preparation_complete,
                )
            )
            if not completed and not request.dry_run and safe_to_reset:
                self._safe_pre_arm_torso_reset(goal_handle)
            self._release_goal()

    def _execute_bin_place(self, goal_handle) -> ExecuteBinPlace.Result:
        request = goal_handle.request
        arm = self._resolve_arm(request.arm)
        result = ExecuteBinPlace.Result()
        result.arm = arm
        result.chassis_distance_x = self._float("bin_place_chassis_distance_x")
        result.chassis_distance_y = self._float("bin_place_chassis_distance_y")
        result.chassis_yaw = self._float("bin_place_chassis_yaw")
        result.chassis_duration_sec = self._float("bin_place_chassis_duration_sec")

        try:
            self._publish_bin_place_feedback(
                goal_handle,
                "MOVING_CHASSIS",
                "moving chassis to the configured bin place location",
                arm,
            )
            self._move_bin_chassis(goal_handle, request.dry_run)

            if request.dry_run:
                self._publish_bin_place_feedback(
                    goal_handle,
                    "DRY_RUN_POSITIONING",
                    "skipping direct bin torso and gripper commands",
                    arm,
                )
            else:
                self._publish_bin_place_feedback(
                    goal_handle,
                    "BENDING_TORSO",
                    "bending torso for bin placement",
                    arm,
                )
                self._publish_torso(
                    goal_handle, self._float_array("bin_place_torso_positions")
                )
                self._wait_delay(
                    goal_handle,
                    self._float("torso_settle_sec"),
                    "while waiting for bin place torso bend",
                )

                self._publish_bin_place_feedback(
                    goal_handle, "OPENING_GRIPPER", "releasing bin", arm
                )
                self._publish_gripper(
                    goal_handle, arm, self._float("gripper_open_position")
                )
                result.gripper_command_published = True
                self._wait_delay(
                    goal_handle,
                    self._float("gripper_settle_sec"),
                    "while waiting for bin release",
                )

            self._publish_bin_place_feedback(
                goal_handle,
                "RESTORING_ROBOT",
                "homing arms and resetting torso",
                arm,
            )
            self._call_home(goal_handle, request.dry_run)
            result.home_completed = True

            if not request.dry_run:
                self._publish_torso(
                    goal_handle, self._float_array("torso_reset_positions")
                )
                result.torso_reset_command_published = True

            result.success = True
            result.message = (
                "bin place dry run completed"
                if request.dry_run
                else "bin place mission completed"
            )
            self._publish_bin_place_feedback(goal_handle, "DONE", result.message, arm)
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
            result.message = f"unexpected bin place mission error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            if not request.dry_run:
                self._publish_zero_chassis()
            self._release_goal()

    def _execute_grasp(self, goal_handle) -> ExecuteGrasp.Result:
        request = goal_handle.request
        arm = self._resolve_arm(request.arm)
        result = ExecuteGrasp.Result()
        result.arm = arm
        torso_prepared = False
        joint_preparation_started = False
        joint_preparation_complete = False
        cartesian_motion_started = False
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
                self._publish_grasp_feedback(
                    goal_handle,
                    "PREPARING",
                    "starting grippers, torso, and arms concurrently"
                    if self._boolean("open_gripper_before_grasp")
                    else "starting torso and arms concurrently",
                    arm,
                )
                torso_prepared = True
                joint_preparation_started = True
                self._prepare_grasp_concurrently(
                    goal_handle,
                    self._boolean("open_gripper_before_grasp"),
                )
                joint_preparation_complete = True

            self._publish_grasp_feedback(
                goal_handle, "DETECTING", "requesting grasp pose detection", arm
            )
            detection = self._call_detect(goal_handle, request)
            result.grasp_pose = detection.grasp_pose
            result.score = float(detection.score)
            result.width = float(detection.width)
            result.height = float(detection.height)
            result.depth = float(detection.depth)
            result.object_id = int(detection.object_id)
            if request.publish_pose:
                self.grasp_pose_publisher.publish(detection.grasp_pose)

            self._check_canceled(goal_handle, "after grasp detection")
            self._publish_grasp_feedback(
                goal_handle,
                "EXECUTING_GRASP_POSE",
                f"sending detected pose to {self._string('arm_pose_action_name')}",
                arm,
            )
            cartesian_motion_started = True
            result.arm_message = self._call_arm_pose(
                goal_handle, arm, detection.grasp_pose.pose, request.dry_run
            )

            if request.dry_run:
                self._publish_grasp_feedback(
                    goal_handle,
                    "DRY_RUN_COMPLETE",
                    "arm plan succeeded; direct close and torso reset were skipped",
                    arm,
                )
            else:
                self._publish_grasp_feedback(
                    goal_handle, "CLOSING_GRIPPER", "closing selected gripper", arm
                )
                self._publish_gripper(
                    goal_handle, arm, self._float("gripper_closed_position")
                )
                result.gripper_command_published = True
                self._wait_delay(
                    goal_handle,
                    self._float("gripper_settle_sec"),
                    "while waiting for gripper close",
                )

                self._publish_grasp_feedback(
                    goal_handle, "RESETTING_TORSO", "publishing torso reset target", arm
                )
                self._publish_torso(
                    goal_handle, self._float_array("torso_reset_positions")
                )
                result.torso_reset_command_published = True

            result.success = True
            result.message = (
                "grasp dry run completed"
                if request.dry_run
                else "grasp mission completed"
            )
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
                    not cartesian_motion_started,
                    not joint_preparation_started or joint_preparation_complete,
                )
            )
            if not completed and not request.dry_run and safe_to_reset:
                result.torso_reset_command_published = self._safe_pre_arm_torso_reset(
                    goal_handle
                )
            self._release_goal()

    def _execute_place(self, goal_handle) -> ExecutePlace.Result:
        request = goal_handle.request
        arm = self._resolve_arm(request.arm)
        result = ExecutePlace.Result()
        result.arm = arm
        result.chassis_distance_x = self._float("place_chassis_distance_x")
        result.chassis_distance_y = self._float("place_chassis_distance_y")
        result.chassis_yaw = self._float("place_chassis_yaw")
        result.chassis_duration_sec = self._float("place_chassis_duration_sec")

        try:
            self._publish_place_feedback(
                goal_handle,
                "MOVING_CHASSIS",
                "moving chassis to the configured fixed place location",
                arm,
            )
            self._move_chassis(goal_handle, request.dry_run)

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
                self._wait_delay(
                    goal_handle,
                    self._float("gripper_settle_sec"),
                    "while waiting for object release",
                )

            if not request.dry_run:
                self._publish_place_feedback(
                    goal_handle, "RESETTING_TORSO", "publishing torso reset target", arm
                )
                self._publish_torso(
                    goal_handle, self._float_array("torso_reset_positions")
                )
                result.torso_reset_command_published = True

            self._publish_place_feedback(
                goal_handle, "HOMING_ARMS", "calling robot arm home service", arm
            )
            self._call_home(goal_handle, request.dry_run)
            result.home_completed = True

            result.success = True
            result.message = (
                "place dry run completed"
                if request.dry_run
                else "place mission completed"
            )
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
            if not request.dry_run:
                self._publish_zero_chassis()
            self._release_goal()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = MissionController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node._publish_zero_chassis()
            executor.shutdown()
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
