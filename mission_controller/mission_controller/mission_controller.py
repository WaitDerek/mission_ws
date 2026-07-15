import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped, TwistStamped
from grasp_orchestrator_interfaces.srv import DetectGraspPose
from mission_interfaces.action import (
    ExecuteBinGrasp,
    ExecuteBinPlace,
    ExecuteGrasp,
    ExecutePlace,
)
from object_pose_interfaces.action import EstimateObjectPose
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
from task_interfaces.action import Home, MoveArmJoints, MoveArmPose
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformBroadcaster,
    TransformException,
    TransformListener,
)
from visualization_msgs.msg import Marker, MarkerArray


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


def quaternion_multiply(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def rotate_vector(
    vector: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def compose_poses(parent_pose: Pose, child_pose: Pose) -> Pose:
    """Compose T_reference_parent with T_parent_child."""
    parent_values = pose_to_array(parent_pose)
    child_values = pose_to_array(child_pose)
    parent_orientation = tuple(parent_values[3:])
    child_in_reference = rotate_vector(
        tuple(child_values[:3]), parent_orientation
    )
    orientation = quaternion_multiply(
        parent_orientation, tuple(child_values[3:])
    )
    orientation_norm = math.sqrt(sum(value * value for value in orientation))

    result = Pose()
    result.position.x = parent_values[0] + child_in_reference[0]
    result.position.y = parent_values[1] + child_in_reference[1]
    result.position.z = parent_values[2] + child_in_reference[2]
    result.orientation.x = orientation[0] / orientation_norm
    result.orientation.y = orientation[1] / orientation_norm
    result.orientation.z = orientation[2] / orientation_norm
    result.orientation.w = orientation[3] / orientation_norm
    return result


class MissionController(Node):
    def __init__(self) -> None:
        super().__init__("mission_controller")
        self._declare_parameters()
        self._validate_parameters()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_static_broadcaster = StaticTransformBroadcaster(self)
        self.target_tf_broadcaster = TransformBroadcaster(self)
        self._publish_camera_mount_tf()

        # Match the command transport used by dual_arm_manipulation/tools/r1pro_test.
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        visualization_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
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
            PoseStamped, self._string("grasp_pose_topic"), visualization_qos
        )
        self.grasp_pose_camera_publisher = self.create_publisher(
            PoseStamped,
            self._string("grasp_pose_camera_topic"),
            visualization_qos,
        )
        self.grasp_pose_ee_publisher = self.create_publisher(
            PoseStamped,
            self._string("grasp_pose_ee_topic"),
            visualization_qos,
        )
        self.gripper_target_pose_publisher = self.create_publisher(
            PoseStamped,
            self._string("gripper_target_pose_topic"),
            visualization_qos,
        )
        self.arm_target_pose_publisher = self.create_publisher(
            PoseStamped, self._string("arm_target_pose_topic"), visualization_qos
        )
        self.grasp_visualization_publisher = self.create_publisher(
            MarkerArray,
            self._string("grasp_visualization_topic"),
            visualization_qos,
        )
        self.latest_grasp_pose: Optional[PoseStamped] = None
        self.latest_grasp_pose_camera: Optional[PoseStamped] = None
        self.latest_grasp_pose_ee: Optional[PoseStamped] = None
        self.latest_gripper_target_pose: Optional[PoseStamped] = None
        self.latest_gripper_target_frame: Optional[str] = None
        self.latest_arm_target_pose: Optional[PoseStamped] = None
        self.preview_grasp_subscription = self.create_subscription(
            PoseStamped,
            self._string("preview_grasp_pose_topic"),
            self._preview_grasp_pose_callback,
            visualization_qos,
        )
        self.visualization_timer = self.create_timer(
            0.5, self._republish_grasp_visualization
        )
        self.bin_object_pose_publisher = self.create_publisher(
            PoseStamped, self._string("bin_object_pose_topic"), 10
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
        self.arm_joints_client = ActionClient(
            self,
            MoveArmJoints,
            self._string("arm_joints_service_name"),
            callback_group=self.client_group,
        )
        self.home_client = ActionClient(
            self,
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
        self.bin_object_pose_client = ActionClient(
            self,
            EstimateObjectPose,
            self._string("bin_object_pose_action_name"),
            callback_group=self.client_group,
        )

        self.state_lock = threading.Lock()
        self.mission_reserved = False
        self.active_mission = ""
        self.active_arm_goal_handle = None
        self.active_arm_joints_goal_handle = None
        self.active_home_goal_handle = None
        self.active_bin_object_pose_goal_handle = None

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
                ("bin_object_pose_action_name", "/object_pose/estimate"),
                ("bin_object_pose_topic", "/mission/bin_object_pose"),
                ("bin_object_pose_model_label", "f320"),
                ("bin_object_pose_instance_index", 0),
                ("bin_object_pose_confidence_threshold", 0.0),
                ("bin_object_pose_result_timeout_sec", 60.0),
                ("bin_grasp_object_to_ee_xyz", [0.0, 0.0, 0.0]),
                ("bin_grasp_object_to_ee_rpy", [0.0, 0.0, 0.0]),
                ("bin_grasp_offset_configured", False),
                ("arm_pose_action_name", "/move_arm_p"),
                ("arm_joints_service_name", "/move_arm_j"),
                ("home_service_name", "/home"),
                ("grasp_pose_topic", "/mission/grasp_pose"),
                (
                    "grasp_pose_camera_topic",
                    "/mission/grasp_pose_camera",
                ),
                ("grasp_pose_ee_topic", "/mission/grasp_pose_ee"),
                (
                    "gripper_target_pose_topic",
                    "/mission/gripper_link_target",
                ),
                ("arm_target_pose_topic", "/mission/arm_link7_target"),
                (
                    "grasp_visualization_topic",
                    "/mission/grasp_visualization",
                ),
                ("preview_grasp_pose_topic", "/mission/preview_grasp_pose"),
                ("preview_arm", "right"),
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
                ("arm_execution_frame", "torso_link4"),
                ("left_ee_frame", "left_arm_link7"),
                ("right_ee_frame", "right_arm_link7"),
                ("left_gripper_frame", "left_gripper_link"),
                ("right_gripper_frame", "right_gripper_link"),
                # The 15 cm retreat is from the detected grasp center to the
                # gripper link, expressed in the corrected grasp frame.
                ("grasp_center_to_gripper_xyz", [-0.15, 0.0, 0.0]),
                # Preserve GraspNet +X while flipping Y/Z to match the
                # physical gripper convention.
                ("grasp_pose_correction_rpy", [3.141592653589793, 0.0, 0.0]),
                # Map GraspNet axes to the physical gripper convention.
                ("grasp_to_gripper_rpy", [0.0, -1.5707963267948966, 0.0]),
                # Apply after the axis mapping, in the target gripper's local
                # frame. The physical gripper is flipped 180 degrees about Z.
                ("gripper_target_post_rpy", [0.0, 0.0, 3.141592653589793]),
                ("camera_mount_tf_enabled", True),
                ("camera_mount_parent_frame", "right_D405_link"),
                ("camera_mount_child_frame", "hdas/camera_wrist_right_link"),
                # The mechanical flange-to-camera transform comes from URDF.
                # This zero-translation rotation only maps the CAD D405 axes
                # to the ROS camera_link convention: camera +X = D405 +Z.
                ("camera_mount_xyz", [0.0, 0.0, 0.0]),
                (
                    "camera_mount_rpy",
                    [-1.5707963267948966, -1.5707963267948966, 0.0],
                ),
                ("camera_mount_correction_rpy", [0.0, 0.0, 0.0]),
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
            "bin_object_pose_action_name",
            "bin_object_pose_topic",
            "bin_object_pose_model_label",
            "arm_pose_action_name",
            "arm_joints_service_name",
            "home_service_name",
            "grasp_pose_topic",
            "grasp_pose_camera_topic",
            "grasp_pose_ee_topic",
            "gripper_target_pose_topic",
            "arm_target_pose_topic",
            "grasp_visualization_topic",
            "preview_grasp_pose_topic",
            "torso_topic",
            "left_gripper_topic",
            "right_gripper_topic",
            "chassis_topic",
            "default_target_frame",
            "arm_execution_frame",
            "left_ee_frame",
            "right_ee_frame",
            "left_gripper_frame",
            "right_gripper_frame",
        ):
            if not self._string(name):
                raise ValueError(f"parameter '{name}' must not be empty")

        if self._string("default_arm").lower() not in VALID_ARMS:
            raise ValueError("default_arm must be 'left' or 'right'")
        if self._string("preview_arm").lower() not in VALID_ARMS:
            raise ValueError("preview_arm must be 'left' or 'right'")

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
            ("camera_mount_correction_rpy", 3),
            ("grasp_center_to_gripper_xyz", 3),
            ("grasp_pose_correction_rpy", 3),
            ("grasp_to_gripper_rpy", 3),
            ("gripper_target_post_rpy", 3),
            ("bin_grasp_object_to_ee_xyz", 3),
            ("bin_grasp_object_to_ee_rpy", 3),
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
            "bin_object_pose_result_timeout_sec",
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
        if self._integer("bin_object_pose_instance_index") < 0:
            raise ValueError("bin_object_pose_instance_index must be nonnegative")
        bin_confidence = self._float("bin_object_pose_confidence_threshold")
        if not 0.0 <= bin_confidence <= 1.0:
            raise ValueError(
                "bin_object_pose_confidence_threshold must be in [0, 1]"
            )

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

    def _make_bin_grasp_pose(self, object_pose: PoseStamped) -> PoseStamped:
        object_values = pose_to_array(object_pose.pose)
        object_quaternion = tuple(object_values[3:])
        offset_xyz = tuple(self._float_array("bin_grasp_object_to_ee_xyz"))
        offset_quaternion = self._quaternion_from_rpy(
            *self._float_array("bin_grasp_object_to_ee_rpy")
        )
        rotated_offset = rotate_vector(offset_xyz, object_quaternion)
        grasp_quaternion = quaternion_multiply(
            object_quaternion, offset_quaternion
        )
        quaternion_norm = math.sqrt(sum(value * value for value in grasp_quaternion))

        grasp_pose = PoseStamped()
        grasp_pose.header = object_pose.header
        grasp_pose.pose.position.x = object_pose.pose.position.x + rotated_offset[0]
        grasp_pose.pose.position.y = object_pose.pose.position.y + rotated_offset[1]
        grasp_pose.pose.position.z = object_pose.pose.position.z + rotated_offset[2]
        grasp_pose.pose.orientation.x = grasp_quaternion[0] / quaternion_norm
        grasp_pose.pose.orientation.y = grasp_quaternion[1] / quaternion_norm
        grasp_pose.pose.orientation.z = grasp_quaternion[2] / quaternion_norm
        grasp_pose.pose.orientation.w = grasp_quaternion[3] / quaternion_norm
        return grasp_pose

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
        gripper_target_execution = PoseStamped()
        gripper_target_execution.header = grasp_pose_execution.header
        gripper_target_execution.pose = self._grasp_center_to_gripper_pose(
            grasp_pose_execution.pose
        )
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
        if not self._boolean("bin_mission_enabled") and not request.dry_run:
            self.get_logger().warning(
                "rejecting bin grasp goal: bin_mission_enabled is false; "
                "configure bin preparation targets and perception first"
            )
            return GoalResponse.REJECT
        if not self._boolean("bin_mission_enabled"):
            self.get_logger().info(
                "accepting perception-only bin grasp dry run while "
                "bin_mission_enabled is false"
            )
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
            arm_joints_goal_handle = self.active_arm_joints_goal_handle
            home_goal_handle = self.active_home_goal_handle
            bin_object_pose_goal_handle = self.active_bin_object_pose_goal_handle
        if arm_goal_handle is not None:
            arm_goal_handle.cancel_goal_async()
        if arm_joints_goal_handle is not None:
            arm_joints_goal_handle.cancel_goal_async()
        if home_goal_handle is not None:
            home_goal_handle.cancel_goal_async()
        if bin_object_pose_goal_handle is not None:
            bin_object_pose_goal_handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    def _release_goal(self) -> None:
        with self.state_lock:
            self.mission_reserved = False
            self.active_mission = ""
            self.active_arm_goal_handle = None
            self.active_arm_joints_goal_handle = None
            self.active_home_goal_handle = None
            self.active_bin_object_pose_goal_handle = None

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

    def _call_task_action(
        self,
        goal_handle,
        client,
        action_name: str,
        action_goal,
        timeout_sec: float,
        active_handle_attribute: str,
    ):
        wait_deadline = time.monotonic() + self._float(
            "dependency_wait_timeout_sec"
        )
        while time.monotonic() < wait_deadline:
            self._check_canceled(goal_handle, f"while waiting for {action_name}")
            remaining = max(0.0, wait_deadline - time.monotonic())
            if client.wait_for_server(timeout_sec=min(0.5, remaining)):
                break
        else:
            raise MissionError(
                f"timeout waiting for action {action_name} after "
                f"{self._float('dependency_wait_timeout_sec'):.1f}s"
            )

        action_handle = self._wait_future(
            client.send_goal_async(action_goal),
            goal_handle,
            f"sending {action_name} goal",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        if action_handle is None or not action_handle.accepted:
            raise MissionError(f"{action_name} goal was rejected")

        with self.state_lock:
            setattr(self, active_handle_attribute, action_handle)
        result_future = action_handle.get_result_async()
        deadline = time.monotonic() + timeout_sec
        try:
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested:
                    action_handle.cancel_goal_async()
                    raise MissionCanceled(f"mission canceled during {action_name}")
                if time.monotonic() >= deadline:
                    action_handle.cancel_goal_async()
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
                setattr(self, active_handle_attribute, None)

        action_result = wrapped_result.result
        succeeded = wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
        if not succeeded or not action_result.success:
            raise MissionError(
                f"{action_name} failed: {action_result.message} "
                f"(error_code={action_result.error_code})"
            )
        return action_result

    def _call_arm_joints(
        self,
        goal_handle,
        left_positions: list[float],
        right_positions: list[float],
        dry_run: bool,
    ) -> str:
        if dry_run:
            return "dry run: skipped non-planning /move_arm_j action"
        action_name = self._string("arm_joints_service_name")
        action_goal = MoveArmJoints.Goal()
        action_goal.left_joints = left_positions
        action_goal.right_joints = right_positions
        action_goal.dry_run = False
        action_goal.duration = 0.0
        response = self._call_task_action(
            goal_handle,
            self.arm_joints_client,
            action_name,
            action_goal,
            self._float("arm_joints_result_timeout_sec"),
            "active_arm_joints_goal_handle",
        )
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
        # Graspness is camera-local. Mission owns the robot/camera relationship
        # and performs the complete conversion after receiving this response.
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
        response.grasp_pose = self._apply_grasp_pose_correction(
            response.grasp_pose
        )
        self.latest_grasp_pose_camera = response.grasp_pose
        self.grasp_pose_camera_publisher.publish(response.grasp_pose)
        return response

    def _forward_bin_object_pose_feedback(
        self, goal_handle, arm: str, feedback_message
    ) -> None:
        feedback = feedback_message.feedback
        self._publish_bin_grasp_feedback(
            goal_handle,
            f"FOUNDATION_{feedback.stage}",
            f"FoundationPose progress={feedback.progress:.0%}",
            arm,
        )

    def _call_bin_object_pose(self, goal_handle, request, arm: str):
        action_name = self._string("bin_object_pose_action_name")
        timeout_sec = (
            float(request.detection_timeout_sec)
            if request.detection_timeout_sec > 0.0
            else self._float("bin_object_pose_result_timeout_sec")
        )
        wait_deadline = time.monotonic() + self._float(
            "dependency_wait_timeout_sec"
        )
        while time.monotonic() < wait_deadline:
            self._check_canceled(goal_handle, f"while waiting for {action_name}")
            remaining = max(0.0, wait_deadline - time.monotonic())
            if self.bin_object_pose_client.wait_for_server(
                timeout_sec=min(0.5, remaining)
            ):
                break
        else:
            raise MissionError(
                f"timeout waiting for action {action_name} after "
                f"{self._float('dependency_wait_timeout_sec'):.1f}s"
            )

        foundation_goal = EstimateObjectPose.Goal()
        foundation_goal.model_label = self._string("bin_object_pose_model_label")
        configured_instance = self._integer("bin_object_pose_instance_index")
        foundation_goal.instance_index = (
            int(request.target_label)
            if request.target_label >= 0
            else configured_instance
        )
        foundation_goal.confidence_threshold = self._float(
            "bin_object_pose_confidence_threshold"
        )
        send_future = self.bin_object_pose_client.send_goal_async(
            foundation_goal,
            feedback_callback=lambda message: self._forward_bin_object_pose_feedback(
                goal_handle, arm, message
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
            self.active_bin_object_pose_goal_handle = foundation_handle
        result_future = foundation_handle.get_result_async()
        deadline = time.monotonic() + timeout_sec
        try:
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested:
                    foundation_handle.cancel_goal_async()
                    raise MissionCanceled(
                        f"mission canceled during {action_name}"
                    )
                if time.monotonic() >= deadline:
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
                self.active_bin_object_pose_goal_handle = None

        foundation_result = wrapped_result.result
        succeeded = wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
        if not succeeded or not foundation_result.success:
            raise MissionError(
                f"{action_name} failed: {foundation_result.message}"
            )

        target_frame = (
            request.target_frame.strip() or self._string("default_target_frame")
        ).lstrip("/")
        object_pose = self._transform_detection_pose(
            foundation_result.pose, target_frame
        )
        self.bin_object_pose_publisher.publish(object_pose)
        grasp_pose = self._make_bin_grasp_pose(object_pose)
        self.grasp_pose_publisher.publish(grasp_pose)
        return foundation_result, object_pose, grasp_pose

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
        action_name = self._string("home_service_name")
        action_goal = Home.Goal()
        action_goal.dry_run = dry_run
        action_goal.duration = 0.0
        response = self._call_task_action(
            goal_handle,
            self.home_client,
            action_name,
            action_goal,
            self._float("home_result_timeout_sec"),
            "active_home_goal_handle",
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
                "requesting FoundationPose object pose estimation",
                arm,
            )
            detection, object_pose, grasp_pose = self._call_bin_object_pose(
                goal_handle, request, arm
            )
            result.grasp_pose = grasp_pose
            result.score = float(detection.detection_score)
            result.width = 0.0
            result.height = 0.0
            result.depth = 0.0
            result.object_id = int(request.target_label)
            if request.publish_pose:
                self.bin_object_pose_publisher.publish(object_pose)
                self.grasp_pose_publisher.publish(grasp_pose)

            self._check_canceled(goal_handle, "after FoundationPose estimation")
            if not self._boolean("bin_grasp_offset_configured"):
                if not request.dry_run:
                    raise MissionError(
                        "bin_grasp_offset_configured is false; calibrate "
                        "bin_grasp_object_to_ee_xyz/rpy before real motion"
                    )
                result.success = True
                result.message = (
                    "bin perception dry run completed; object-to-EE offset "
                    "is not configured, so arm planning was skipped"
                )
                self._publish_bin_grasp_feedback(
                    goal_handle,
                    "OFFSET_REQUIRED",
                    result.message,
                    arm,
                )
                goal_handle.succeed()
                completed = True
                return result

            self._publish_bin_grasp_feedback(
                goal_handle,
                "EXECUTING_BIN_GRASP",
                f"sending calibrated bin grasp pose to {self._string('arm_pose_action_name')}",
                arm,
            )
            cartesian_motion_started = True
            result.arm_message = self._call_arm_pose(
                goal_handle, arm, grasp_pose.pose, request.dry_run
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
            grasp_pose_execution, arm_target_stamped = (
                self._prepare_grasp_target(detection.grasp_pose, arm)
            )
            result.grasp_pose = grasp_pose_execution
            result.score = float(detection.score)
            result.width = float(detection.width)
            result.height = float(detection.height)
            result.depth = float(detection.depth)
            result.object_id = int(detection.object_id)
            if request.publish_pose:
                self.grasp_pose_publisher.publish(grasp_pose_execution)

            self._check_canceled(goal_handle, "after grasp detection")
            self._publish_grasp_feedback(
                goal_handle,
                "EXECUTING_GRASP_POSE",
                f"sending detected pose to {self._string('arm_pose_action_name')}",
                arm,
            )
            cartesian_motion_started = True
            result.arm_message = self._call_arm_pose(
                goal_handle, arm, arm_target_stamped.pose, request.dry_run
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
