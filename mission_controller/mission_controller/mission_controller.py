import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped, TwistStamped
from grasp_orchestrator_interfaces.srv import DetectGraspPose
from mission_interfaces.action import (
    ExecuteBoxGrasp,
    ExecuteBoxPlace,
    ExecuteBoxStack,
    ExecuteGrasp,
    ExecutePlace,
    MoveChassis,
)
try:
    from object_pose_interfaces.action import EstimateObjectPose
except ModuleNotFoundError as exc:
    EstimateObjectPose = None
    OBJECT_POSE_IMPORT_ERROR: Optional[ModuleNotFoundError] = exc
else:
    OBJECT_POSE_IMPORT_ERROR = None
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
from task_interfaces.action import (
    GoReady,
    Home,
    MoveArmJoints,
    MoveArmPose,
    PickupTask,
)
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
CHASSIS_DIRECTIONS = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "clockwise": (0.0, 0.0, -1.0),
    "counterclockwise": (0.0, 0.0, 1.0),
}
ANGULAR_CHASSIS_DIRECTIONS = {"clockwise", "counterclockwise"}


class MissionError(RuntimeError):
    pass


class MissionCanceled(MissionError):
    pass


class TaskActionError(MissionError):
    def __init__(self, message: str, error_code: int) -> None:
        super().__init__(message)
        self.error_code = error_code


class PickupAttemptError(MissionError):
    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[int],
        motion_started: bool,
        first_segment_completed: bool,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.motion_started = motion_started
        self.first_segment_completed = first_segment_completed


class TwoStageMotionError(MissionError):
    def __init__(self, stage: int, stage_one_completed: bool, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.stage_one_completed = stage_one_completed


@dataclass
class GraspCandidate:
    pose: PoseStamped
    score: float
    width: float
    height: float
    depth: float
    object_id: int


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


def interpolate_pose(start_pose: Pose, target_pose: Pose, fraction: float) -> Pose:
    """Interpolate position linearly and orientation along the shortest arc."""
    if not 0.0 <= fraction <= 1.0:
        raise MissionError("pose interpolation fraction must be in [0, 1]")

    start = pose_to_array(start_pose)
    target = pose_to_array(target_pose)
    start_quaternion = start[3:]
    target_quaternion = target[3:]
    dot = sum(a * b for a, b in zip(start_quaternion, target_quaternion))
    if dot < 0.0:
        target_quaternion = [-value for value in target_quaternion]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        orientation = [
            a + fraction * (b - a)
            for a, b in zip(start_quaternion, target_quaternion)
        ]
    else:
        theta = math.acos(dot)
        scale = math.sin(theta)
        start_scale = math.sin((1.0 - fraction) * theta) / scale
        target_scale = math.sin(fraction * theta) / scale
        orientation = [
            start_scale * a + target_scale * b
            for a, b in zip(start_quaternion, target_quaternion)
        ]
    orientation_norm = math.sqrt(sum(value * value for value in orientation))

    result = Pose()
    result.position.x = start[0] + fraction * (target[0] - start[0])
    result.position.y = start[1] + fraction * (target[1] - start[1])
    result.position.z = start[2] + fraction * (target[2] - start[2])
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
        feedback_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
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
        self.joint_state_lock = threading.Lock()
        self.latest_joint_positions: dict[str, float] = {}
        self.latest_joint_state_time = 0.0
        self.latest_joint_state_sequence = 0
        self.latest_torso_positions: list[float] = []
        self.latest_torso_state_time = 0.0
        self.latest_torso_state_sequence = 0
        self.latest_gripper_positions: dict[str, float] = {}
        self.latest_gripper_state_times = {"left": 0.0, "right": 0.0}
        self.latest_gripper_state_sequences = {"left": 0, "right": 0}
        self.joint_state_subscription = self.create_subscription(
            JointState,
            self._string("joint_state_topic"),
            self._joint_state_callback,
            20,
        )
        self.torso_feedback_subscription = self.create_subscription(
            JointState,
            self._string("torso_feedback_topic"),
            self._torso_feedback_callback,
            20,
        )
        self.left_gripper_feedback_subscription = self.create_subscription(
            JointState,
            self._string("left_gripper_feedback_topic"),
            lambda message: self._gripper_feedback_callback("left", message),
            feedback_qos,
        )
        self.right_gripper_feedback_subscription = self.create_subscription(
            JointState,
            self._string("right_gripper_feedback_topic"),
            lambda message: self._gripper_feedback_callback("right", message),
            feedback_qos,
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
        self.arm_intermediate_pose_publisher = self.create_publisher(
            PoseStamped,
            self._string("arm_intermediate_pose_topic"),
            visualization_qos,
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
        self.latest_arm_intermediate_pose: Optional[PoseStamped] = None
        self.preview_grasp_subscription = self.create_subscription(
            PoseStamped,
            self._string("preview_grasp_pose_topic"),
            self._preview_grasp_pose_callback,
            visualization_qos,
        )
        self.visualization_timer = self.create_timer(
            0.5, self._republish_grasp_visualization
        )
        self.box_object_pose_publisher = self.create_publisher(
            PoseStamped, self._string("box_object_pose_topic"), 10
        )
        self.box_object_pose_raw_publisher = self.create_publisher(
            PoseStamped,
            self._string("box_object_pose_raw_topic"),
            visualization_qos,
        )
        self.box_object_pose_camera_subscription = self.create_subscription(
            PoseStamped,
            self._string("box_object_pose_camera_topic"),
            self._box_object_pose_camera_callback,
            10,
        )

        self.client_group = ReentrantCallbackGroup()
        self.server_group = ReentrantCallbackGroup()
        self.detect_client = self.create_client(
            DetectGraspPose,
            self._string("detect_service_name"),
            callback_group=self.client_group,
        )
        self.box_detect_client = self.create_client(
            DetectGraspPose,
            self._string("box_detect_service_name"),
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
        self.go_ready_client = ActionClient(
            self,
            GoReady,
            self._string("go_ready_action_name"),
            callback_group=self.client_group,
        )
        self.arm_pose_client = ActionClient(
            self,
            MoveArmPose,
            self._string("arm_pose_action_name"),
            callback_group=self.client_group,
        )
        self.box_object_pose_client = None
        if EstimateObjectPose is not None:
            self.box_object_pose_client = ActionClient(
                self,
                EstimateObjectPose,
                self._string("box_object_pose_action_name"),
                callback_group=self.client_group,
            )
        else:
            self.get_logger().warning(
                "object_pose_interfaces is unavailable; box grasp goals will "
                f"be rejected, while material grasp remains available: "
                f"{OBJECT_POSE_IMPORT_ERROR}"
            )
        self.pickup_task_client = ActionClient(
            self,
            PickupTask,
            self._string("pickup_task_action_name"),
            callback_group=self.client_group,
        )
        self.state_lock = threading.Lock()
        self.mission_reserved = False
        self.active_mission = ""
        self.active_arm_goal_handle = None
        self.active_arm_joints_goal_handle = None
        self.active_home_goal_handle = None
        self.active_go_ready_goal_handle = None
        self.active_box_object_pose_goal_handle = None
        self.active_pickup_task_goal_handle = None

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
        self.box_grasp_action_server = ActionServer(
            self,
            ExecuteBoxGrasp,
            self._string("execute_box_grasp_action_name"),
            execute_callback=self._execute_box_grasp,
            goal_callback=self._box_grasp_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.box_place_action_server = ActionServer(
            self,
            ExecuteBoxPlace,
            self._string("execute_box_place_action_name"),
            execute_callback=self._execute_box_place,
            goal_callback=self._box_place_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.box_stack_action_server = ActionServer(
            self,
            ExecuteBoxStack,
            self._string("execute_box_stack_action_name"),
            execute_callback=self._execute_box_stack,
            goal_callback=self._box_stack_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.move_chassis_action_server = ActionServer(
            self,
            MoveChassis,
            self._string("move_chassis_action_name"),
            execute_callback=self._execute_move_chassis,
            goal_callback=self._move_chassis_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.get_logger().info(
            "mission controller ready: "
            f"grasp={self._string('execute_grasp_action_name')} "
            f"place={self._string('execute_place_action_name')} "
            f"box_grasp={self._string('execute_box_grasp_action_name')} "
            f"box_place={self._string('execute_box_place_action_name')} "
            f"box_stack={self._string('execute_box_stack_action_name')} "
            f"chassis={self._string('move_chassis_action_name')}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameters(
            namespace="",
            parameters=[
                ("execute_grasp_action_name", "/execute_grasp"),
                ("execute_place_action_name", "/execute_place"),
                ("execute_box_grasp_action_name", "/execute_box_grasp"),
                ("execute_box_place_action_name", "/execute_box_place"),
                ("execute_box_stack_action_name", "/execute_box_stack"),
                ("move_chassis_action_name", "/move_chassis"),
                ("detect_service_name", "/detect_grasp_pose"),
                ("box_detect_service_name", "/detect_box_grasp_pose"),
                ("box_mission_enabled", False),
                ("box_object_pose_action_name", "/object_pose/estimate"),
                ("box_object_pose_topic", "/mission/box_object_pose"),
                ("box_object_pose_camera_topic", "/object_pose/pose"),
                ("box_object_pose_raw_topic", "/mission/box_object_pose_raw"),
                ("box_object_pose_model_label", "f320"),
                ("box_object_pose_instance_index", 0),
                ("box_object_pose_confidence_threshold", 0.0),
                ("box_object_pose_result_timeout_sec", 60.0),
                ("box_camera_pose_constraint_enabled", True),
                ("box_camera_pose_axis_min_dot", 0.5),
                ("pickup_task_action_name", "/pickup_task"),
                ("pickup_task_result_timeout_sec", 120.0),
                ("box_detection_attempts", 2),
                ("box_width", 0.357),
                ("box_height", 0.127),
                ("box_type", "f320"),
                # FoundationPose publishes the oriented-bounding-box centre
                # with F320 axes X=down, Y=depth, Z=width. Keep those axes by
                # default; each robot pickup profile owns the fixed transform
                # from model axes to its operation/end-effector convention.
                (
                    "box_foundation_to_pickup_rpy",
                    [0.0, 0.0, 0.0],
                ),
                ("arm_pose_action_name", "/move_arm_p"),
                ("arm_joints_service_name", "/move_arm_j"),
                ("home_service_name", "/home"),
                ("go_ready_action_name", "/go_ready"),
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
                    "arm_intermediate_pose_topic",
                    "/mission/arm_link7_intermediate",
                ),
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
                ("joint_state_topic", "/joint_states"),
                ("torso_feedback_topic", "/hdas/feedback_torso"),
                (
                    "left_gripper_feedback_topic",
                    "/hdas/feedback_gripper_left",
                ),
                (
                    "right_gripper_feedback_topic",
                    "/hdas/feedback_gripper_right",
                ),
                (
                    "left_arm_joint_names",
                    [f"left_arm_joint{index}" for index in range(1, 8)],
                ),
                (
                    "right_arm_joint_names",
                    [f"right_arm_joint{index}" for index in range(1, 8)],
                ),
                ("verify_arm_joint_targets", True),
                ("arm_joint_target_tolerance", 0.10),
                ("arm_joint_target_wait_timeout_sec", 20.0),
                ("arm_joint_target_stable_samples", 3),
                ("box_observation_ready_check_enabled", True),
                ("box_observation_feedback_max_age_sec", 2.0),
                ("box_observation_torso_tolerance", 0.10),
                ("grasp_observation_ready_check_enabled", True),
                ("grasp_observation_feedback_max_age_sec", 2.0),
                ("grasp_observation_torso_tolerance", 0.10),
                ("torso_target_tolerance", 0.03),
                ("torso_target_wait_timeout_sec", 40.0),
                ("torso_target_stable_samples", 3),
                ("default_arm", "right"),
                ("default_target_frame", "torso_link4"),
                ("arm_execution_frame", "torso_link4"),
                ("left_ee_frame", "left_arm_link7"),
                ("right_ee_frame", "right_arm_link7"),
                ("left_gripper_frame", "left_gripper_link"),
                ("right_gripper_frame", "right_gripper_link"),
                # Retreat 3 cm from the detected grasp centre along corrected
                # local -X; gripper_link -> arm_link7 still comes from URDF TF.
                ("grasp_center_to_gripper_xyz", [-0.03, 0.0, 0.0]),
                # Preserve GraspNet +X while flipping Y/Z to match the
                # physical gripper convention.
                ("grasp_pose_correction_rpy", [3.141592653589793, 0.0, 0.0]),
                ("grasp_symmetry_normalization_enabled", True),
                ("grasp_symmetry_rpy", [3.141592653589793, 0.0, 0.0]),
                # Map GraspNet axes to the physical gripper convention.
                ("grasp_to_gripper_rpy", [0.0, -1.5707963267948966, 0.0]),
                # Apply after the axis mapping in the target gripper's local
                # frame: tilt 20 degrees backward about Y while retaining the
                # physical gripper's 180-degree palm flip about Z.
                (
                    "gripper_target_post_rpy",
                    [0.0, -0.3490658503988659, 3.141592653589793],
                ),
                ("camera_mount_tf_enabled", True),
                ("camera_mount_parent_frame", "right_D405_link"),
                (
                    "camera_mount_child_frame",
                    "hdas/camera_wrist_right_link",
                ),
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
                ("grasp_detection_min_timeout_sec", 60.0),
                ("grasp_detection_attempts", 0),
                ("grasp_candidates_per_detection", 1),
                ("dependency_wait_timeout_sec", 10.0),
                ("arm_joints_result_timeout_sec", 60.0),
                ("arm_pose_result_timeout_sec", 120.0),
                ("home_result_timeout_sec", 60.0),
                ("go_ready_result_timeout_sec", 60.0),
                ("wait_for_command_subscribers", True),
                ("require_command_subscribers", False),
                ("command_subscriber_wait_timeout_sec", 3.0),
                ("command_repeat_count", 10),
                ("command_repeat_interval_sec", 0.005),
                ("torso_settle_sec", 1.0),
                ("arm_settle_sec", 1.0),
                ("gripper_settle_sec", 1.0),
                ("box_detection_posture_settle_sec", 2.0),
                ("box_place_release_delay_sec", 2.0),
                ("place_torso_straighten_step_delay_sec", 2.0),
                ("torso_prepare_positions", [0.61, -0.81, -0.60, 0.0]),
                ("torso_reset_positions", [0.0, 0.0, 0.0, 0.0]),
                ("torso_velocities", [0.1, 0.1, 0.1, 0.1]),
                (
                    "observation_intermediate_left_joint_positions",
                    [1.30, 0.6, 0.0, -1.5, 0.0, 0.0, 0.0],
                ),
                (
                    "observation_intermediate_right_joint_positions",
                    [1.30, -0.6, 0.0, -1.5, 0.0, 0.0, 0.0],
                ),
                (
                    "grasp_left_joint_positions",
                    [-0.98, 0.84, -0.83, -2.00, 1.25, 0.29, 0.13],
                ),
                (
                    "grasp_right_joint_positions",
                    [-0.98, -0.84, 0.93, -2.00, -1.25, 0.60, -0.13],
                ),
                (
                    "place_right_joint_positions",
                    [-1.011, 0.040, 0.835, -0.9513, -1.956, 0.901, -1.370],
                ),
                # Fill these box-specific values before enabling box missions.
                (
                    "box_grasp_left_observation_joint_positions",
                    [-0.88, 1.24, -0.70, -2.0, 1.25, 0.1, 0.0],
                ),
                (
                    "box_grasp_right_observation_joint_positions",
                    [
                        0.86,
                        -0.24,
                        0.20,
                        -2.0944,
                        0.174647,
                        -0.618606,
                        0.104098,
                    ],
                ),
                (
                    "box_pickup_clearance_left_joint_positions",
                    [
                        -1.413830,
                        0.687872,
                        -1.236596,
                        -1.839149,
                        1.905532,
                        0.525745,
                        1.146383,
                    ],
                ),
                (
                    "box_pickup_clearance_right_joint_positions",
                    [
                        -1.403617,
                        -0.668723,
                        1.238298,
                        -1.802128,
                        -1.944043,
                        0.435319,
                        -1.307021,
                    ],
                ),
                ("box_grasp_torso_prepare_positions", [0.61, -0.81, -0.6, 0.0]),
                (
                    "box_grasp_torso_lift_positions",
                    [0.41, -0.81, -0.6, 0.0],
                ),
                ("box_place_torso_positions", [0.61, -0.81, -0.6, 0.0]),
                (
                    "box_place_torso_straighten_intermediate_positions",
                    [0.61, -1.2, -0.6, 0.0],
                ),
                # ExecuteBoxStack starts and ends at this fixed arm posture
                # with a fully upright torso. The arms remain fixed while the
                # torso selects one of four calibrated stacking heights.
                (
                    "stack_default_left_joint_positions",
                    [
                        -1.133818,
                        0.120475,
                        -1.197170,
                        -0.672971,
                        2.354960,
                        1.046860,
                        1.240171,
                    ],
                ),
                (
                    "stack_default_right_joint_positions",
                    [
                        -1.129491,
                        -0.125152,
                        1.203598,
                        -0.676157,
                        -2.343158,
                        1.040511,
                        -1.249240,
                    ],
                ),
                (
                    "stack_pickup_torso_positions",
                    [0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "stack_level_torso_positions",
                    [
                        1.630000, -2.500000, -0.920000, 0.0,
                        1.356518, -2.506617, -1.049991, 0.0,
                        1.154311, -2.139907, -0.885489, 0.0,
                        0.906723, -1.646855, -0.640023, 0.0,
                    ],
                ),
                ("stack_start_arm_duration_sec", 8.0),
                ("stack_torso1_speed", 0.1),
                ("stack_torso3_speed", 0.13),
                ("stack_release_delay_sec", 2.0),
                ("stack_start_ready_check_enabled", True),
                ("stack_start_feedback_max_age_sec", 2.0),
                ("gripper_open_position", 100.0),
                ("gripper_closed_position", 0.0),
                ("open_gripper_before_grasp", True),
                ("grasp_close_check_enabled", True),
                ("grasp_empty_close_ratio_threshold", 0.95),
                ("grasp_gripper_feedback_timeout_sec", 2.0),
                ("grasp_gripper_feedback_max_age_sec", 0.5),
                ("grasp_max_empty_close_attempts", 0),
                ("grasp_recovery_retry_delay_sec", 1.0),
                ("chassis_linear_speed", 0.3),
                ("chassis_angular_speed", 0.2),
                ("chassis_move_duration_sec", 3.0),
                ("chassis_publish_hz", 10.0),
                ("chassis_stop_repeat_count", 1),
                ("max_chassis_linear_speed", 0.3),
                ("max_chassis_angular_speed", 0.4),
                ("home_velocity", 0.05),
            ],
        )

    def _validate_parameters(self) -> None:
        for name in (
            "execute_grasp_action_name",
            "execute_place_action_name",
            "execute_box_grasp_action_name",
            "execute_box_place_action_name",
            "execute_box_stack_action_name",
            "move_chassis_action_name",
            "detect_service_name",
            "box_detect_service_name",
            "box_object_pose_action_name",
            "box_object_pose_topic",
            "box_object_pose_camera_topic",
            "box_object_pose_raw_topic",
            "box_object_pose_model_label",
            "pickup_task_action_name",
            "box_type",
            "arm_pose_action_name",
            "arm_joints_service_name",
            "home_service_name",
            "go_ready_action_name",
            "grasp_pose_topic",
            "grasp_pose_camera_topic",
            "grasp_pose_ee_topic",
            "gripper_target_pose_topic",
            "arm_target_pose_topic",
            "arm_intermediate_pose_topic",
            "grasp_visualization_topic",
            "preview_grasp_pose_topic",
            "torso_topic",
            "left_gripper_topic",
            "right_gripper_topic",
            "chassis_topic",
            "joint_state_topic",
            "torso_feedback_topic",
            "left_gripper_feedback_topic",
            "right_gripper_feedback_topic",
            "default_target_frame",
            "arm_execution_frame",
            "left_ee_frame",
            "right_ee_frame",
            "left_gripper_frame",
            "right_gripper_frame",
        ):
            if not self._string(name):
                raise ValueError(f"parameter '{name}' must not be empty")

        for name in ("left_arm_joint_names", "right_arm_joint_names"):
            joint_names = self._string_array(name)
            if len(joint_names) != 7 or len(set(joint_names)) != 7:
                raise ValueError(
                    f"parameter '{name}' must contain 7 unique joint names"
                )

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
            ("observation_intermediate_left_joint_positions", 7),
            ("observation_intermediate_right_joint_positions", 7),
            ("place_right_joint_positions", 7),
            ("box_grasp_left_observation_joint_positions", 7),
            ("box_grasp_right_observation_joint_positions", 7),
            ("box_pickup_clearance_left_joint_positions", 7),
            ("box_pickup_clearance_right_joint_positions", 7),
            ("box_grasp_torso_prepare_positions", 4),
            ("box_grasp_torso_lift_positions", 4),
            ("box_place_torso_positions", 4),
            ("box_place_torso_straighten_intermediate_positions", 4),
            ("stack_default_left_joint_positions", 7),
            ("stack_default_right_joint_positions", 7),
            ("stack_pickup_torso_positions", 4),
            ("stack_level_torso_positions", 16),
            ("camera_mount_xyz", 3),
            ("camera_mount_rpy", 3),
            ("camera_mount_correction_rpy", 3),
            ("grasp_center_to_gripper_xyz", 3),
            ("grasp_pose_correction_rpy", 3),
            ("grasp_symmetry_rpy", 3),
            ("grasp_to_gripper_rpy", 3),
            ("gripper_target_post_rpy", 3),
            ("box_foundation_to_pickup_rpy", 3),
        ):
            values = self._float_array(name)
            if len(values) != expected_length:
                raise ValueError(
                    f"parameter '{name}' must contain {expected_length} values"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"parameter '{name}' contains NaN or Inf")

        for name in (
            "gripper_open_position",
            "gripper_closed_position",
        ):
            value = self._float(name)
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"parameter '{name}' must be in [0, 100]")

        if math.isclose(
            self._float("gripper_open_position"),
            self._float("gripper_closed_position"),
        ):
            raise ValueError(
                "gripper_open_position and gripper_closed_position must differ"
            )
        close_ratio = self._float("grasp_empty_close_ratio_threshold")
        if not 0.0 <= close_ratio <= 1.0:
            raise ValueError(
                "grasp_empty_close_ratio_threshold must be in [0, 1]"
            )

        positive_parameters = (
            "default_detection_timeout_sec",
            "grasp_detection_min_timeout_sec",
            "dependency_wait_timeout_sec",
            "arm_joints_result_timeout_sec",
            "arm_pose_result_timeout_sec",
            "home_result_timeout_sec",
            "go_ready_result_timeout_sec",
            "command_subscriber_wait_timeout_sec",
            "chassis_linear_speed",
            "chassis_angular_speed",
            "chassis_move_duration_sec",
            "chassis_publish_hz",
            "max_chassis_linear_speed",
            "max_chassis_angular_speed",
            "home_velocity",
            "camera_tf_timeout_sec",
            "pickup_task_result_timeout_sec",
            "box_width",
            "box_height",
            "arm_joint_target_tolerance",
            "arm_joint_target_wait_timeout_sec",
            "box_observation_feedback_max_age_sec",
            "box_observation_torso_tolerance",
            "grasp_observation_feedback_max_age_sec",
            "grasp_observation_torso_tolerance",
            "torso_target_tolerance",
            "torso_target_wait_timeout_sec",
            "stack_start_feedback_max_age_sec",
            "stack_start_arm_duration_sec",
            "stack_torso1_speed",
            "stack_torso3_speed",
            "grasp_gripper_feedback_timeout_sec",
            "grasp_gripper_feedback_max_age_sec",
        )
        for name in positive_parameters:
            if not math.isfinite(self._float(name)) or self._float(name) <= 0.0:
                raise ValueError(f"parameter '{name}' must be finite and positive")

        if self._float("chassis_linear_speed") > self._float(
            "max_chassis_linear_speed"
        ) + 1e-9:
            raise ValueError(
                "chassis_linear_speed exceeds max_chassis_linear_speed"
            )
        if self._float("chassis_angular_speed") > self._float(
            "max_chassis_angular_speed"
        ) + 1e-9:
            raise ValueError(
                "chassis_angular_speed exceeds max_chassis_angular_speed"
            )

        nonnegative_parameters = (
            "command_repeat_interval_sec",
            "torso_settle_sec",
            "arm_settle_sec",
            "gripper_settle_sec",
            "box_object_pose_result_timeout_sec",
            "box_detection_posture_settle_sec",
            "box_place_release_delay_sec",
            "stack_release_delay_sec",
            "grasp_recovery_retry_delay_sec",
            "place_torso_straighten_step_delay_sec",
        )
        for name in nonnegative_parameters:
            if not math.isfinite(self._float(name)) or self._float(name) < 0.0:
                raise ValueError(f"parameter '{name}' must be finite and nonnegative")

        if self._integer("command_repeat_count") <= 0:
            raise ValueError("command_repeat_count must be positive")
        if self._integer("arm_joint_target_stable_samples") <= 0:
            raise ValueError("arm_joint_target_stable_samples must be positive")
        if self._integer("torso_target_stable_samples") <= 0:
            raise ValueError("torso_target_stable_samples must be positive")
        if self._integer("grasp_detection_attempts") < 0:
            raise ValueError(
                "grasp_detection_attempts must be nonnegative; zero means "
                "unlimited retries"
            )
        if self._integer("box_detection_attempts") <= 0:
            raise ValueError("box_detection_attempts must be positive")
        if self._integer("grasp_candidates_per_detection") <= 0:
            raise ValueError("grasp_candidates_per_detection must be positive")
        if self._integer("grasp_max_empty_close_attempts") < 0:
            raise ValueError(
                "grasp_max_empty_close_attempts must be nonnegative; "
                "zero means unlimited retries"
            )
        if self._integer("chassis_stop_repeat_count") <= 0:
            raise ValueError("chassis_stop_repeat_count must be positive")
        if self._integer("box_object_pose_instance_index") < 0:
            raise ValueError("box_object_pose_instance_index must be nonnegative")

        box_confidence = self._float("box_object_pose_confidence_threshold")
        if not 0.0 <= box_confidence <= 1.0:
            raise ValueError(
                "box_object_pose_confidence_threshold must be in [0, 1]"
            )
        box_axis_min_dot = self._float("box_camera_pose_axis_min_dot")
        if not 0.0 <= box_axis_min_dot <= 1.0:
            raise ValueError("box_camera_pose_axis_min_dot must be in [0, 1]")

        if self._boolean("box_mission_enabled"):
            for name in (
                "box_grasp_left_observation_joint_positions",
                "box_grasp_right_observation_joint_positions",
                "box_grasp_torso_prepare_positions",
            ):
                if all(abs(value) < 1e-9 for value in self._float_array(name)):
                    raise ValueError(
                        f"box_mission_enabled requires configured '{name}'"
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

    def _string_array(self, name: str) -> list[str]:
        return [str(value).strip() for value in self.get_parameter(name).value]

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

    def _box_grasp_goal_callback(
        self, request: ExecuteBoxGrasp.Goal
    ) -> GoalResponse:
        if self.box_object_pose_client is None:
            self.get_logger().warning(
                "rejecting box grasp goal: object_pose_interfaces is unavailable"
            )
            return GoalResponse.REJECT
        if not self._boolean("box_mission_enabled") and not request.dry_run:
            self.get_logger().warning(
                "rejecting box grasp goal: box_mission_enabled is false; "
                "configure box preparation targets and perception first"
            )
            return GoalResponse.REJECT
        if not self._boolean("box_mission_enabled"):
            self.get_logger().info(
                "accepting perception-only box grasp dry run while "
                "box_mission_enabled is false"
            )
        return self._reserve_goal(
            "box_grasp", request.request_id, self._resolve_arm(request.arm)
        )

    def _box_place_goal_callback(
        self, request: ExecuteBoxPlace.Goal
    ) -> GoalResponse:
        if not self._boolean("box_mission_enabled"):
            self.get_logger().warning(
                "rejecting box place goal: box_mission_enabled is false; "
                "configure box preparation targets first"
            )
            return GoalResponse.REJECT
        if not request.dry_run and all(
            abs(value) < 1e-9
            for value in self._float_array("box_place_torso_positions")
        ):
            self.get_logger().warning(
                "rejecting box place goal: box_place_torso_positions is not "
                "configured"
            )
            return GoalResponse.REJECT
        return self._reserve_goal(
            "box_place", request.request_id, self._resolve_arm(request.arm)
        )

    def _box_stack_goal_callback(
        self, request: ExecuteBoxStack.Goal
    ) -> GoalResponse:
        if request.level < 1 or request.level > 4:
            self.get_logger().error(
                "rejecting box_stack goal: level must be an integer in [1, 4], "
                f"got {request.level}"
            )
            return GoalResponse.REJECT
        if not self._boolean("box_mission_enabled"):
            self.get_logger().warning(
                "rejecting box_stack goal: box_mission_enabled is false"
            )
            return GoalResponse.REJECT
        with self.state_lock:
            if self.mission_reserved:
                self.get_logger().warning(
                    "rejecting box_stack goal: "
                    f"{self.active_mission} mission is active"
                )
                return GoalResponse.REJECT
            self.mission_reserved = True
            self.active_mission = "box_stack"
        self.get_logger().info(f"accepted box_stack goal level={request.level}")
        return GoalResponse.ACCEPT

    def _move_chassis_goal_callback(
        self, request: MoveChassis.Goal
    ) -> GoalResponse:
        direction = request.direction.strip().lower()
        if direction not in CHASSIS_DIRECTIONS:
            self.get_logger().error(
                "rejecting move_chassis goal: direction must be one of "
                f"{sorted(CHASSIS_DIRECTIONS)}, got '{request.direction}'"
            )
            return GoalResponse.REJECT
        with self.state_lock:
            if self.mission_reserved:
                self.get_logger().warning(
                    "rejecting move_chassis goal: "
                    f"{self.active_mission} mission is active"
                )
                return GoalResponse.REJECT
            self.mission_reserved = True
            self.active_mission = "move_chassis"
        self.get_logger().info(
            f"accepted move_chassis goal direction={direction}"
        )
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        with self.state_lock:
            arm_goal_handle = self.active_arm_goal_handle
            arm_joints_goal_handle = self.active_arm_joints_goal_handle
            home_goal_handle = self.active_home_goal_handle
            go_ready_goal_handle = self.active_go_ready_goal_handle
            box_object_pose_goal_handle = self.active_box_object_pose_goal_handle
            pickup_task_goal_handle = self.active_pickup_task_goal_handle
        if arm_goal_handle is not None:
            arm_goal_handle.cancel_goal_async()
        if arm_joints_goal_handle is not None:
            arm_joints_goal_handle.cancel_goal_async()
        if home_goal_handle is not None:
            home_goal_handle.cancel_goal_async()
        if go_ready_goal_handle is not None:
            go_ready_goal_handle.cancel_goal_async()
        if box_object_pose_goal_handle is not None:
            box_object_pose_goal_handle.cancel_goal_async()
        if pickup_task_goal_handle is not None:
            pickup_task_goal_handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    def _release_goal(self) -> None:
        with self.state_lock:
            self.mission_reserved = False
            self.active_mission = ""
            self.active_arm_goal_handle = None
            self.active_arm_joints_goal_handle = None
            self.active_home_goal_handle = None
            self.active_go_ready_goal_handle = None
            self.active_box_object_pose_goal_handle = None
            self.active_pickup_task_goal_handle = None

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
    def _publish_box_grasp_feedback(
        goal_handle, stage: str, detail: str, arm: str
    ) -> None:
        feedback = ExecuteBoxGrasp.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.arm = arm
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _publish_box_place_feedback(
        goal_handle, stage: str, detail: str, arm: str
    ) -> None:
        feedback = ExecuteBoxPlace.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.arm = arm
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _publish_box_stack_feedback(
        goal_handle, stage: str, detail: str, level: int
    ) -> None:
        feedback = ExecuteBoxStack.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.level = level
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _publish_move_chassis_feedback(
        goal_handle, stage: str, detail: str, progress: float
    ) -> None:
        feedback = MoveChassis.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        feedback.progress = max(0.0, min(1.0, progress))
        goal_handle.publish_feedback(feedback)

    def _finalize_action_result(
        self, result, started_at: float, action_name: str
    ) -> None:
        if "elapsed_sec=" in str(result.message):
            return
        elapsed_sec = max(0.0, time.monotonic() - started_at)
        base_message = str(result.message).strip() or f"{action_name} finished"
        result.message = f"{base_message} (elapsed_sec={elapsed_sec:.3f})"
        self.get_logger().info(
            f"{action_name} finished: success={bool(result.success)}, "
            f"elapsed_sec={elapsed_sec:.3f}"
        )

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
        velocities: Optional[list[float]] = None,
        require_subscriber: bool = True,
        honor_cancel: bool = True,
    ) -> None:
        self._publish_joint_command(
            self.torso_publisher,
            self._string("torso_topic"),
            positions,
            (
                list(velocities)
                if velocities is not None
                else self._float_array("torso_velocities")
            ),
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

    def _joint_state_callback(self, message: JointState) -> None:
        if len(message.name) != len(message.position):
            return
        positions = {
            name: float(position)
            for name, position in zip(message.name, message.position)
            if name and math.isfinite(float(position))
        }
        if not positions:
            return
        with self.joint_state_lock:
            self.latest_joint_positions.update(positions)
            self.latest_joint_state_time = time.monotonic()
            self.latest_joint_state_sequence += 1

    def _torso_feedback_callback(self, message: JointState) -> None:
        if len(message.position) < 4:
            return
        positions = [float(value) for value in message.position[:4]]
        if not all(math.isfinite(value) for value in positions):
            return
        with self.joint_state_lock:
            self.latest_torso_positions = positions
            self.latest_torso_state_time = time.monotonic()
            self.latest_torso_state_sequence += 1

    def _gripper_feedback_callback(
        self, arm: str, message: JointState
    ) -> None:
        if len(message.position) != 1:
            return
        position = float(message.position[0])
        if not math.isfinite(position):
            return
        with self.joint_state_lock:
            self.latest_gripper_positions[arm] = position
            self.latest_gripper_state_times[arm] = time.monotonic()
            self.latest_gripper_state_sequences[arm] += 1

    def _gripper_feedback_sequence(self, arm: str) -> int:
        with self.joint_state_lock:
            return self.latest_gripper_state_sequences[arm]

    def _gripper_close_ratio(self, measured_position: float) -> float:
        open_position = self._float("gripper_open_position")
        closed_position = self._float("gripper_closed_position")
        ratio = (
            (measured_position - open_position)
            / (closed_position - open_position)
        )
        return min(1.0, max(0.0, ratio))

    def _wait_for_gripper_close_feedback(
        self,
        goal_handle,
        arm: str,
        sequence_before_close: int,
    ) -> float:
        timeout_sec = self._float("grasp_gripper_feedback_timeout_sec")
        max_age_sec = self._float("grasp_gripper_feedback_max_age_sec")
        deadline = time.monotonic() + timeout_sec
        latest_position: Optional[float] = None
        latest_sequence = sequence_before_close
        latest_age = math.inf

        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while waiting for gripper close feedback"
            )
            now = time.monotonic()
            with self.joint_state_lock:
                latest_sequence = self.latest_gripper_state_sequences[arm]
                latest_position = self.latest_gripper_positions.get(arm)
                latest_age = now - self.latest_gripper_state_times[arm]
            if (
                latest_sequence > sequence_before_close
                and latest_position is not None
                and latest_age <= max_age_sec
            ):
                return latest_position
            time.sleep(0.02)

        raise MissionError(
            f"no fresh {arm} gripper feedback after close command within "
            f"{timeout_sec:.1f}s (topic="
            f"{self._string(f'{arm}_gripper_feedback_topic')}, "
            f"sequence={latest_sequence}, age_sec={latest_age:.3f})"
        )

    def _wait_for_torso_target(
        self,
        goal_handle,
        targets: list[float],
        context: str,
    ) -> None:
        tolerance = self._float("torso_target_tolerance")
        timeout_sec = self._float("torso_target_wait_timeout_sec")
        required_stable = self._integer("torso_target_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequence = -1
        last_errors: list[float] = []

        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, f"while {context}")
            with self.joint_state_lock:
                sequence = self.latest_torso_state_sequence
                measured = list(self.latest_torso_positions)
            if sequence == last_sequence or len(measured) < len(targets):
                time.sleep(0.02)
                continue
            last_sequence = sequence
            last_errors = [
                abs(actual - target)
                for actual, target in zip(measured, targets)
            ]
            if max(last_errors, default=0.0) <= tolerance:
                stable_samples += 1
                if stable_samples >= required_stable:
                    self.get_logger().info(
                        f"verified torso target while {context}: "
                        f"max_error={max(last_errors, default=0.0):.4f}"
                    )
                    return
            else:
                stable_samples = 0
            time.sleep(0.02)

        raise MissionError(
            f"torso target was not confirmed while {context} within "
            f"{timeout_sec:.1f}s: max_error="
            f"{max(last_errors, default=math.inf):.4f}, "
            f"tolerance={tolerance:.4f}, stable_samples={required_stable}"
        )

    def _observation_ready(
        self,
        *,
        enabled_parameter: str,
        left_target_parameter: str,
        right_target_parameter: str,
        torso_target_parameter: str,
        max_age_parameter: str,
        torso_tolerance_parameter: str,
        label: str,
    ) -> tuple[bool, str]:
        if not self._boolean(enabled_parameter):
            return False, f"{label} observation readiness check is disabled"

        now = time.monotonic()
        max_age = self._float(max_age_parameter)
        with self.joint_state_lock:
            arm_positions = dict(self.latest_joint_positions)
            arm_age = now - self.latest_joint_state_time
            torso_positions = list(self.latest_torso_positions)
            torso_age = now - self.latest_torso_state_time

        left_names = self._string_array("left_arm_joint_names")
        right_names = self._string_array("right_arm_joint_names")
        arm_targets = dict(
            zip(
                left_names,
                self._float_array(left_target_parameter),
            )
        )
        arm_targets.update(
            zip(
                right_names,
                self._float_array(right_target_parameter),
            )
        )
        missing = [name for name in arm_targets if name not in arm_positions]
        if missing:
            return False, f"missing arm feedback joints={missing}"
        if not torso_positions:
            return False, "torso feedback has not been received"
        if arm_age > max_age or torso_age > max_age:
            return (
                False,
                "feedback is stale: "
                f"arms={arm_age:.2f}s, torso={torso_age:.2f}s, "
                f"limit={max_age:.2f}s",
            )

        arm_errors = {
            name: abs(arm_positions[name] - target)
            for name, target in arm_targets.items()
        }
        torso_targets = self._float_array(torso_target_parameter)
        torso_errors = [
            abs(actual - target)
            for actual, target in zip(torso_positions, torso_targets)
        ]
        max_arm_error = max(arm_errors.values(), default=0.0)
        max_torso_error = max(torso_errors, default=0.0)
        arm_tolerance = self._float("arm_joint_target_tolerance")
        torso_tolerance = self._float(torso_tolerance_parameter)
        ready = (
            max_arm_error <= arm_tolerance
            and max_torso_error <= torso_tolerance
        )
        return (
            ready,
            f"max arm error={max_arm_error:.4f} rad "
            f"(limit={arm_tolerance:.4f}), max torso error="
            f"{max_torso_error:.4f} (limit={torso_tolerance:.4f})",
        )

    def _box_observation_ready(self) -> tuple[bool, str]:
        return self._observation_ready(
            enabled_parameter="box_observation_ready_check_enabled",
            left_target_parameter="box_grasp_left_observation_joint_positions",
            right_target_parameter="box_grasp_right_observation_joint_positions",
            torso_target_parameter="box_grasp_torso_prepare_positions",
            max_age_parameter="box_observation_feedback_max_age_sec",
            torso_tolerance_parameter="box_observation_torso_tolerance",
            label="box",
        )

    def _grasp_observation_ready(self) -> tuple[bool, str]:
        return self._observation_ready(
            enabled_parameter="grasp_observation_ready_check_enabled",
            left_target_parameter="grasp_left_joint_positions",
            right_target_parameter="grasp_right_joint_positions",
            torso_target_parameter="torso_prepare_positions",
            max_age_parameter="grasp_observation_feedback_max_age_sec",
            torso_tolerance_parameter="grasp_observation_torso_tolerance",
            label="grasp",
        )

    def _stack_default_ready(self) -> tuple[bool, str]:
        if not self._boolean("stack_start_ready_check_enabled"):
            return True, "stack default readiness check is disabled"

        now = time.monotonic()
        max_age = self._float("stack_start_feedback_max_age_sec")
        with self.joint_state_lock:
            arm_positions = dict(self.latest_joint_positions)
            arm_age = now - self.latest_joint_state_time
            torso_positions = list(self.latest_torso_positions)
            torso_age = now - self.latest_torso_state_time

        left_names = self._string_array("left_arm_joint_names")
        right_names = self._string_array("right_arm_joint_names")
        targets = dict(
            zip(
                left_names,
                self._float_array("stack_default_left_joint_positions"),
            )
        )
        targets.update(
            zip(
                right_names,
                self._float_array("stack_default_right_joint_positions"),
            )
        )
        missing = [name for name in targets if name not in arm_positions]
        if missing:
            return False, f"missing arm feedback joints={missing}"
        if arm_age > max_age:
            return (
                False,
                f"arm feedback is stale: age={arm_age:.2f}s, limit={max_age:.2f}s",
            )
        if len(torso_positions) < 4:
            return False, "missing torso feedback"
        if torso_age > max_age:
            return (
                False,
                f"torso feedback is stale: age={torso_age:.2f}s, "
                f"limit={max_age:.2f}s",
            )

        errors = {
            name: abs(arm_positions[name] - target)
            for name, target in targets.items()
        }
        max_error = max(errors.values(), default=0.0)
        arm_tolerance = self._float("arm_joint_target_tolerance")
        torso_targets = self._float_array("stack_pickup_torso_positions")
        torso_error = max(
            (
                abs(actual - target)
                for actual, target in zip(torso_positions, torso_targets)
            ),
            default=0.0,
        )
        torso_tolerance = self._float("torso_target_tolerance")
        return (
            max_error <= arm_tolerance and torso_error <= torso_tolerance,
            f"max arm error={max_error:.4f} rad "
            f"(limit={arm_tolerance:.4f}), max torso error="
            f"{torso_error:.4f} (limit={torso_tolerance:.4f})",
        )

    def _wait_for_arm_joint_targets(
        self,
        goal_handle,
        left_positions: list[float],
        right_positions: list[float],
    ) -> None:
        if not self._boolean("verify_arm_joint_targets"):
            return
        targets = dict(
            zip(self._string_array("left_arm_joint_names"), left_positions)
        )
        targets.update(
            zip(self._string_array("right_arm_joint_names"), right_positions)
        )
        tolerance = self._float("arm_joint_target_tolerance")
        timeout_sec = self._float("arm_joint_target_wait_timeout_sec")
        required_stable = self._integer("arm_joint_target_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        last_sequence = -1
        last_errors: dict[str, float] = {}
        missing = list(targets)

        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while verifying the arm observation posture"
            )
            with self.joint_state_lock:
                sequence = self.latest_joint_state_sequence
                measured = dict(self.latest_joint_positions)
            if sequence == last_sequence:
                time.sleep(0.02)
                continue
            last_sequence = sequence
            missing = [name for name in targets if name not in measured]
            if missing:
                stable_samples = 0
                continue
            last_errors = {
                name: abs(measured[name] - target)
                for name, target in targets.items()
            }
            if max(last_errors.values(), default=0.0) <= tolerance:
                stable_samples += 1
                if stable_samples >= required_stable:
                    self.get_logger().info(
                        "verified arm joint target from /joint_states: "
                        f"max_error={max(last_errors.values(), default=0.0):.4f}"
                    )
                    return
            else:
                stable_samples = 0
            time.sleep(0.02)

        if missing:
            detail = f"missing joints={missing}"
        else:
            worst = sorted(
                last_errors.items(), key=lambda item: item[1], reverse=True
            )[:3]
            detail = f"largest errors={dict(worst)}"
        raise MissionError(
            "arm joint action returned before the commanded posture was "
            f"confirmed within {tolerance:.3f} rad for {required_stable} fresh "
            f"samples ({detail})"
        )

    def _publish_both_grippers(self, goal_handle, position: float) -> None:
        for gripper_arm in ("left", "right"):
            self._publish_gripper(goal_handle, gripper_arm, position)

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

    def _prepare_grasp_arms_and_torso(self, goal_handle) -> None:
        self._call_arm_joints(
            goal_handle,
            self._float_array("grasp_left_joint_positions"),
            self._float_array("grasp_right_joint_positions"),
            False,
            goal_accepted_callback=lambda: self._prepare_grasp_torso(goal_handle),
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while waiting for synchronized arm and torso preparation",
        )

    def _prepare_observation_intermediate_arms(self, goal_handle) -> None:
        self._call_arm_joints(
            goal_handle,
            self._float_array("observation_intermediate_left_joint_positions"),
            self._float_array("observation_intermediate_right_joint_positions"),
            False,
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while waiting for intermediate arm preparation",
        )

    def _prepare_grasp_concurrently(
        self, goal_handle, open_grippers: bool
    ) -> None:
        intermediate_tasks = [self._prepare_observation_intermediate_arms]
        if open_grippers:
            intermediate_tasks.insert(0, self._prepare_grasp_grippers)

        with ThreadPoolExecutor(
            max_workers=len(intermediate_tasks),
            thread_name_prefix="grasp_intermediate",
        ) as executor:
            futures = [
                executor.submit(task, goal_handle) for task in intermediate_tasks
            ]
            for future in futures:
                future.result()

        # Dispatch the torso target as soon as the final /move_arm_j goal is
        # accepted.  The arm trajectory and waist motion therefore begin in
        # the same phase instead of two independent worker threads racing.
        self._prepare_grasp_arms_and_torso(goal_handle)

    def _prepare_box_grasp_grippers(self, goal_handle) -> None:
        self._publish_both_grippers(
            goal_handle, self._float("gripper_open_position")
        )
        self._wait_delay(
            goal_handle,
            self._float("gripper_settle_sec"),
            "while waiting for box gripper preparation",
        )

    def _prepare_box_grasp_torso(self, goal_handle) -> None:
        self._publish_torso(
            goal_handle,
            self._float_array("box_grasp_torso_prepare_positions"),
        )
        self._wait_delay(
            goal_handle,
            self._float("torso_settle_sec"),
            "while waiting for box torso preparation",
        )

    def _prepare_box_grasp_arms_and_torso(self, goal_handle) -> None:
        self._call_arm_joints(
            goal_handle,
            self._float_array("box_grasp_left_observation_joint_positions"),
            self._float_array("box_grasp_right_observation_joint_positions"),
            False,
            goal_accepted_callback=lambda: self._prepare_box_grasp_torso(
                goal_handle
            ),
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while waiting for synchronized box arm and torso preparation",
        )

    def _prepare_box_grasp_concurrently(self, goal_handle) -> None:
        self._prepare_observation_intermediate_arms(goal_handle)
        self._prepare_box_grasp_arms_and_torso(goal_handle)

    def _wait_for_box_detection_posture(
        self, goal_handle, arm: str
    ) -> None:
        self._publish_box_grasp_feedback(
            goal_handle,
            "WAITING_FOR_BOX_OBSERVATION",
            "waiting for measured arm and torso feedback to confirm the box "
            "observation posture",
            arm,
        )
        self._wait_for_arm_joint_targets(
            goal_handle,
            self._float_array("box_grasp_left_observation_joint_positions"),
            self._float_array("box_grasp_right_observation_joint_positions"),
        )
        self._wait_for_torso_target(
            goal_handle,
            self._float_array("box_grasp_torso_prepare_positions"),
            "confirming the box observation torso before detection",
        )
        settle_sec = self._float("box_detection_posture_settle_sec")
        self._publish_box_grasp_feedback(
            goal_handle,
            "SETTLING_BOX_OBSERVATION",
            "box observation arms and torso confirmed; holding still for "
            f"{settle_sec:.1f}s before FoundationPose",
            arm,
        )
        self._wait_delay(
            goal_handle,
            settle_sec,
            "while holding the confirmed box observation posture before detection",
        )

    def _prepare_box_pickup_clearance_arms(self, goal_handle) -> None:
        self._call_arm_joints(
            goal_handle,
            self._float_array("box_pickup_clearance_left_joint_positions"),
            self._float_array("box_pickup_clearance_right_joint_positions"),
            False,
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while waiting for the box pickup clearance posture",
        )

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

    def _move_chassis_for_duration(
        self,
        goal_handle,
        direction: str,
        speed: float,
        duration_sec: float,
    ) -> None:
        direction_scale = CHASSIS_DIRECTIONS[direction]
        vx = direction_scale[0] * speed
        vy = direction_scale[1] * speed
        wz = direction_scale[2] * speed
        topic = self._string("chassis_topic")
        self._wait_for_publisher(self.chassis_publisher, topic, goal_handle)

        period = 1.0 / self._float("chassis_publish_hz")
        started_at = time.monotonic()
        deadline = started_at + duration_sec
        last_feedback_at = started_at - 1.0
        try:
            while time.monotonic() < deadline:
                self._check_canceled(goal_handle, "during chassis motion")
                now = time.monotonic()
                self.chassis_publisher.publish(
                    self._make_chassis_message(vx, vy, wz)
                )
                if now - last_feedback_at >= 0.5:
                    progress = min(1.0, (now - started_at) / duration_sec)
                    self._publish_move_chassis_feedback(
                        goal_handle,
                        "MOVING",
                        f"moving {direction} at {speed:.3f} for "
                        f"{duration_sec:.3f}s ({progress * 100.0:.0f}%)",
                        progress,
                    )
                    last_feedback_at = now
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
        feedback_callback=None,
        goal_accepted_callback=None,
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

        send_future = client.send_goal_async(
            action_goal, feedback_callback=feedback_callback
        )
        action_handle = self._wait_future(
            send_future,
            goal_handle,
            f"sending {action_name} goal",
            self._float("dependency_wait_timeout_sec"),
            cancel_local_future=False,
        )
        if action_handle is None or not action_handle.accepted:
            raise MissionError(f"{action_name} goal was rejected")

        with self.state_lock:
            setattr(self, active_handle_attribute, action_handle)
        try:
            if goal_accepted_callback is not None:
                try:
                    goal_accepted_callback()
                except Exception:
                    action_handle.cancel_goal_async()
                    raise

            result_future = action_handle.get_result_async()
            deadline = time.monotonic() + timeout_sec
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
            raise TaskActionError(
                f"{action_name} failed: {action_result.message} "
                f"(error_code={action_result.error_code})",
                int(action_result.error_code),
            )
        return action_result

    def _stack_level_torso_target(self, level: int) -> list[float]:
        values = self._float_array("stack_level_torso_positions")
        start = (level - 1) * 4
        return values[start : start + 4]

    def _move_stack_torso(
        self,
        goal_handle,
        targets: list[float],
        context: str,
    ) -> None:
        velocities, estimated_duration = self._stack_torso_motion_profile(targets)
        self.get_logger().info(
            "stack torso synchronized profile: "
            f"duration={estimated_duration:.3f}s, velocities="
            f"{[round(value, 6) for value in velocities]}"
        )
        self._publish_torso(goal_handle, targets, velocities=velocities)
        self._wait_for_torso_target(goal_handle, targets, context)

    def _stack_torso_motion_profile(
        self,
        targets: list[float],
    ) -> tuple[list[float], float]:
        with self.joint_state_lock:
            current = list(self.latest_torso_positions)
        if len(current) < 4:
            raise MissionError(
                "cannot calculate synchronized stack torso speeds without "
                "four measured torso positions"
            )

        deltas = [
            abs(target - actual)
            for actual, target in zip(current[:4], targets)
        ]
        torso1_speed = self._float("stack_torso1_speed")
        if deltas[0] <= 1e-6:
            if max(deltas, default=0.0) <= 1e-6:
                return [0.0, 0.0, 0.0, 0.0], 0.0
            raise MissionError(
                "cannot synchronize stack torso motion to Torso1 because "
                "Torso1 has no remaining motion while another torso joint does"
            )

        duration = deltas[0] / torso1_speed
        velocities = [delta / duration for delta in deltas]
        velocities[2] = (
            self._float("stack_torso3_speed")
            if deltas[2] > 1e-6
            else 0.0
        )
        return velocities, duration

    def _prepare_stack_start_pose(self, goal_handle) -> str:
        pickup_torso = self._float_array("stack_pickup_torso_positions")
        pickup_velocities, pickup_duration = self._stack_torso_motion_profile(
            pickup_torso
        )
        self.get_logger().info(
            "stack start torso synchronized profile: "
            f"duration={pickup_duration:.3f}s, velocities="
            f"{[round(value, 6) for value in pickup_velocities]}"
        )
        arm_message = self._call_arm_joints(
            goal_handle,
            self._float_array("stack_default_left_joint_positions"),
            self._float_array("stack_default_right_joint_positions"),
            False,
            goal_accepted_callback=lambda: self._publish_torso(
                goal_handle,
                pickup_torso,
                velocities=pickup_velocities,
            ),
            duration=self._float("stack_start_arm_duration_sec"),
        )
        self._wait_for_torso_target(
            goal_handle,
            pickup_torso,
            "preparing the highest box-loading torso posture",
        )
        self._wait_delay(
            goal_handle,
            self._float("arm_settle_sec"),
            "while settling at the fixed box-loading posture",
        )
        return arm_message

    def _call_arm_joints(
        self,
        goal_handle,
        left_positions: list[float],
        right_positions: list[float],
        dry_run: bool,
        goal_accepted_callback=None,
        duration: float = 0.0,
    ) -> str:
        if dry_run:
            return "dry run: skipped non-planning /move_arm_j action"
        action_name = self._string("arm_joints_service_name")
        action_goal = MoveArmJoints.Goal()
        action_goal.left_joints = left_positions
        action_goal.right_joints = right_positions
        action_goal.dry_run = False
        action_goal.duration = float(duration)
        try:
            response = self._call_task_action(
                goal_handle,
                self.arm_joints_client,
                action_name,
                action_goal,
                self._float("arm_joints_result_timeout_sec"),
                "active_arm_joints_goal_handle",
                goal_accepted_callback=goal_accepted_callback,
            )
        except MissionError as exc:
            if "hardware trajectory execution failed or runtime state froze" not in str(
                exc
            ):
                raise
            self.get_logger().warning(
                f"{action_name} reported a transient runtime failure; "
                "checking measured joints before deciding whether motion failed"
            )
            self._wait_for_arm_joint_targets(
                goal_handle, left_positions, right_positions
            )
            self.get_logger().warning(
                f"{action_name} runtime error overridden because measured "
                "joints reached the commanded posture"
            )
            return f"measured target reached after transient result: {exc}"
        self._wait_for_arm_joint_targets(
            goal_handle, left_positions, right_positions
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
        candidate_poses = list(response.candidate_poses) or [response.grasp_pose]

        def metadata(values, index: int, fallback):
            return values[index] if index < len(values) else fallback

        candidates: list[GraspCandidate] = []
        for index, pose in enumerate(candidate_poses):
            corrected_pose = self._apply_grasp_pose_correction(pose)
            candidates.append(
                GraspCandidate(
                    pose=corrected_pose,
                    score=float(
                        metadata(response.candidate_scores, index, response.score)
                    ),
                    width=float(
                        metadata(response.candidate_widths, index, response.width)
                    ),
                    height=float(
                        metadata(response.candidate_heights, index, response.height)
                    ),
                    depth=float(
                        metadata(response.candidate_depths, index, response.depth)
                    ),
                    object_id=int(
                        metadata(
                            response.candidate_object_ids,
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

    def _forward_box_object_pose_feedback(
        self, goal_handle, arm: str, feedback_message
    ) -> None:
        feedback = feedback_message.feedback
        self._publish_box_grasp_feedback(
            goal_handle,
            f"FOUNDATION_{feedback.stage}",
            f"FoundationPose progress={feedback.progress:.0%}",
            arm,
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

    def _call_box_object_pose(self, goal_handle, request, arm: str):
        if self.box_object_pose_client is None or EstimateObjectPose is None:
            raise MissionError(
                "box grasp requires the object_pose_interfaces package"
            )
        action_name = self._string("box_object_pose_action_name")
        # ExecuteBoxGrasp.detection_timeout_sec remains in the action interface
        # for compatibility, but box perception deliberately ignores it. A
        # configured value of zero disables the local result deadline so a
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
        requested_frame = request.target_frame.strip().lstrip("/")
        if requested_frame and requested_frame != target_frame:
            self.get_logger().warning(
                f"ignoring box target_frame '{requested_frame}'; "
                f"pickup_task requires robot body frame '{target_frame}'"
            )
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
        arm: str,
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
            arm,
        )

    def _call_pickup_task(
        self,
        goal_handle,
        box_pose: PoseStamped,
        dry_run: bool,
        arm: str,
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
            arm,
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
                        arm,
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
        self, goal_handle, arm: str, reason: str, dry_run: bool
    ) -> None:
        self._publish_box_grasp_feedback(
            goal_handle,
            "RECOVERING_BOX_OBSERVATION",
            f"{reason}; returning directly to the box observation posture "
            "before re-detection",
            arm,
        )
        if not dry_run:
            # Do not revisit the broad initialization waypoint here. The arms
            # are already on the pickup path, so return directly to the
            # validated final box observation joints while restoring the torso.
            self._prepare_box_grasp_arms_and_torso(goal_handle)

    def _detect_and_execute_box_pickup(
        self, goal_handle, request, arm: str, motion_state: dict[str, bool]
    ):
        detection_attempts = self._integer("box_detection_attempts")
        failures: list[str] = []

        for detection_attempt in range(1, detection_attempts + 1):
            clearance_active = False
            if not request.dry_run:
                self._wait_for_box_detection_posture(goal_handle, arm)
            self._publish_box_grasp_feedback(
                goal_handle,
                "DETECTING_BOX",
                "requesting FoundationPose object pose estimation "
                f"(attempt {detection_attempt}/{detection_attempts})",
                arm,
            )
            try:
                detection, object_pose = self._call_box_object_pose(
                    goal_handle, request, arm
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
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "REDETECTING_BOX",
                        "FoundationPose failed; requesting one fresh detection",
                        arm,
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
                    arm,
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
                    self._recover_box_observation(
                        goal_handle,
                        arm,
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
                            arm,
                        )
                    continue

            self._publish_box_grasp_feedback(
                goal_handle,
                "PLANNING_BOX_PICKUP",
                f"sending detection {detection_attempt}/{detection_attempts} "
                f"torso-frame box pose to "
                f"{self._string('pickup_task_action_name')}",
                arm,
            )
            try:
                arm_message = self._call_pickup_task(
                    goal_handle,
                    object_pose,
                    request.dry_run,
                    arm,
                    detection_attempt,
                    detection_attempts,
                )
                if not request.dry_run:
                    motion_state["started"] = True
                return detection, object_pose, arm_message
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
                        arm,
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
                        arm,
                    )

        raise MissionError(
            "box pickup exhausted fresh-detection attempts: "
            + " | ".join(failures)
        )

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

    def _call_go_ready(self, goal_handle, dry_run: bool) -> str:
        action_name = self._string("go_ready_action_name")
        action_goal = GoReady.Goal()
        action_goal.dry_run = dry_run
        action_goal.duration = 0.0
        response = self._call_task_action(
            goal_handle,
            self.go_ready_client,
            action_name,
            action_goal,
            self._float("go_ready_result_timeout_sec"),
            "active_go_ready_goal_handle",
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

    def _execute_box_grasp(self, goal_handle) -> ExecuteBoxGrasp.Result:
        started_at = time.monotonic()
        request = goal_handle.request
        arm = self._resolve_arm(request.arm)
        result = ExecuteBoxGrasp.Result()
        result.arm = arm
        motion_state = {"started": False}

        try:
            self._publish_box_grasp_feedback(
                goal_handle,
                "INITIALIZING",
                "preparing box observation pose, torso, and grippers",
                arm,
            )
            if request.dry_run:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "DRY_RUN_INITIALIZATION",
                    "skipping direct box preparation commands",
                    arm,
                )
            else:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "OPENING_INITIAL_GRIPPERS",
                    "opening both grippers before moving to the box "
                    "observation posture",
                    arm,
                )
                self._prepare_box_grasp_grippers(goal_handle)
                observation_ready, readiness_detail = (
                    self._box_observation_ready()
                )
                if observation_ready:
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "BOX_OBSERVATION_READY",
                        "arms and torso are already at the box observation "
                        f"posture; skipping preparation ({readiness_detail})",
                        arm,
                    )
                else:
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "PREPARING_BOX_OBSERVATION",
                        "box observation posture is not ready; moving through "
                        "the intermediate arms, then the final observation "
                        f"posture ({readiness_detail})",
                        arm,
                    )
                    self._prepare_box_grasp_concurrently(goal_handle)

            detection, object_pose, result.arm_message = (
                self._detect_and_execute_box_pickup(
                    goal_handle, request, arm, motion_state
                )
            )
            # ExecuteBoxGrasp predates the box-level pickup delegation. Keep
            # this field populated with the transformed box pose for callers.
            result.grasp_pose = object_pose
            result.score = float(detection.detection_score)
            result.width = self._float("box_width")
            result.height = self._float("box_height")
            result.depth = 0.0
            result.object_id = int(request.target_label)
            if request.publish_pose:
                self.box_object_pose_publisher.publish(object_pose)

            if request.dry_run:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "DRY_RUN_COMPLETE",
                    "box pickup planning succeeded; direct execution was skipped",
                    arm,
                )
            else:
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "CLOSING_BOX_GRIPPERS",
                    "closing both grippers after pickup execution",
                    arm,
                )
                feedback_sequences = {
                    gripper_arm: self._gripper_feedback_sequence(gripper_arm)
                    for gripper_arm in ("left", "right")
                }
                self._publish_both_grippers(
                    goal_handle, self._float("gripper_closed_position")
                )
                result.gripper_command_published = True
                self._wait_delay(
                    goal_handle,
                    self._float("gripper_settle_sec"),
                    "while waiting for both box grippers to close",
                )

                torso_lift_target = self._float_array(
                    "box_grasp_torso_lift_positions"
                )
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "LIFTING_BOX_TORSO1_CLEARANCE",
                    "moving Torso1 0.20 rad toward zero while holding "
                    "Torso2/3/4 at the pickup posture",
                    arm,
                )
                self._publish_torso(goal_handle, torso_lift_target)
                result.torso_lift_command_published = True
                self._wait_for_torso_target(
                    goal_handle,
                    torso_lift_target,
                    "confirming the Torso1 box-clearance lift",
                )

                self._publish_box_grasp_feedback(
                    goal_handle,
                    "VERIFYING_BOX_GRASP",
                    "Torso1 clearance lift confirmed; checking both grippers "
                    "for retained material",
                    arm,
                )
                measured_positions = {
                    gripper_arm: self._wait_for_gripper_close_feedback(
                        goal_handle,
                        gripper_arm,
                        feedback_sequences[gripper_arm],
                    )
                    for gripper_arm in ("left", "right")
                }
                close_ratios = {
                    gripper_arm: self._gripper_close_ratio(position)
                    for gripper_arm, position in measured_positions.items()
                }
                empty_threshold = self._float(
                    "grasp_empty_close_ratio_threshold"
                )
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
                        arm,
                    )
                    raise MissionError(detail)

                self._publish_box_grasp_feedback(
                    goal_handle,
                    "BOX_GRASP_CONFIRMED",
                    "both grippers retained the box below the empty-grasp "
                    f"threshold after the Torso1 clearance lift "
                    f"({close_detail})",
                    arm,
                )

            result.success = True
            result.message = (
                "box grasp dry run completed"
                if request.dry_run
                else "box grasp mission completed"
            )
            self._finalize_action_result(
                result, started_at, "execute_box_grasp"
            )
            self._publish_box_grasp_feedback(goal_handle, "DONE", result.message, arm)
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
            result.message = f"unexpected box grasp mission error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            # A failed box pickup deliberately retains the validated box
            # observation posture. In particular, an IK/planning failure must
            # not publish torso_reset_positions=[0, 0, 0, 0]. A subsequent
            # action can detect the retained posture and skip initialization.
            self._release_goal()
            self._finalize_action_result(
                result, started_at, "execute_box_grasp"
            )

    def _execute_box_place(self, goal_handle) -> ExecuteBoxPlace.Result:
        started_at = time.monotonic()
        request = goal_handle.request
        arm = self._resolve_arm(request.arm)
        result = ExecuteBoxPlace.Result()
        result.arm = arm

        try:
            if request.dry_run:
                self._publish_box_place_feedback(
                    goal_handle,
                    "DRY_RUN_POSITIONING",
                    "skipping direct box torso and gripper commands",
                    arm,
                )
            else:
                self._publish_box_place_feedback(
                    goal_handle,
                    "BENDING_TORSO",
                    "bending torso for box placement",
                    arm,
                )
                self._publish_torso(
                    goal_handle, self._float_array("box_place_torso_positions")
                )
                self._publish_box_place_feedback(
                    goal_handle,
                    "WAITING_FOR_TORSO_BEND",
                    "waiting for measured torso feedback to confirm the box "
                    "place bend before releasing the grippers",
                    arm,
                )
                self._wait_for_torso_target(
                    goal_handle,
                    self._float_array("box_place_torso_positions"),
                    "confirming the box place torso bend",
                )
                self._publish_box_place_feedback(
                    goal_handle,
                    "HOLDING_AT_BOX_PLACE_HEIGHT",
                    "box place torso target confirmed; holding for "
                    f"{self._float('box_place_release_delay_sec'):.1f}s "
                    "before releasing the grippers",
                    arm,
                )
                self._wait_delay(
                    goal_handle,
                    self._float("box_place_release_delay_sec"),
                    "while holding the confirmed box place height before release",
                )

                self._publish_box_place_feedback(
                    goal_handle,
                    "OPENING_BOX_GRIPPERS",
                    "opening both grippers to release the box",
                    arm,
                )
                self._publish_both_grippers(
                    goal_handle, self._float("gripper_open_position")
                )
                result.gripper_command_published = True
                self._wait_delay(
                    goal_handle,
                    self._float("gripper_settle_sec"),
                    "while waiting for box release",
                )

                self._publish_box_place_feedback(
                    goal_handle,
                    "LIFTING_ARMS_AFTER_RELEASE",
                    "moving both released arms to the box pickup clearance "
                    "posture while keeping the torso bent",
                    arm,
                )
                self._prepare_box_pickup_clearance_arms(goal_handle)

            if not request.dry_run:
                torso_intermediate = self._float_array(
                    "box_place_torso_straighten_intermediate_positions"
                )
                self._publish_box_place_feedback(
                    goal_handle,
                    "POSITIONING_TORSO2_FOR_STRAIGHTENING",
                    "moving torso2 from the box-place bend to the configured "
                    "intermediate posture before full straightening",
                    arm,
                )
                self._publish_torso(goal_handle, torso_intermediate)
                self._wait_for_torso_target(
                    goal_handle,
                    torso_intermediate,
                    "confirming the torso2 box-place intermediate posture",
                )

                self._publish_box_place_feedback(
                    goal_handle,
                    "STRAIGHTENING_TORSO",
                    "torso2 intermediate posture confirmed; straightening "
                    "and verifying all torso joints before moving the arms "
                    "to the fixed ready posture",
                    arm,
                )
                self._publish_torso(
                    goal_handle, self._float_array("torso_reset_positions")
                )
                result.torso_reset_command_published = True
                self._wait_for_torso_target(
                    goal_handle,
                    self._float_array("torso_reset_positions"),
                    "confirming the straight torso posture",
                )

            self._publish_box_place_feedback(
                goal_handle,
                "RETURNING_ARMS_TO_READY",
                "returning both arms to the fixed ready posture after the "
                "torso is straight",
                arm,
            )
            self._call_go_ready(goal_handle, request.dry_run)
            result.ready_completed = True

            result.success = True
            result.message = (
                "box place dry run completed"
                if request.dry_run
                else "box place mission completed"
            )
            self._finalize_action_result(
                result, started_at, "execute_box_place"
            )
            self._publish_box_place_feedback(goal_handle, "DONE", result.message, arm)
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
            result.message = f"unexpected box place mission error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            self._release_goal()
            self._finalize_action_result(
                result, started_at, "execute_box_place"
            )

    def _execute_box_stack(self, goal_handle) -> ExecuteBoxStack.Result:
        started_at = time.monotonic()
        level = int(goal_handle.request.level)
        result = ExecuteBoxStack.Result()
        result.level = level
        target_torso = self._stack_level_torso_target(level)
        pickup_torso = self._float_array("stack_pickup_torso_positions")
        result.target_torso_positions = target_torso

        try:
            self._publish_box_stack_feedback(
                goal_handle,
                "INITIALIZING",
                "checking the fixed highest box-loading arm and torso posture",
                level,
            )
            ready, readiness_detail = self._stack_default_ready()
            if not ready:
                self._publish_box_stack_feedback(
                    goal_handle,
                    "PREPARING_START_POSE",
                    "current posture is not the fixed highest loading posture; "
                    f"moving both arms and torso into place ({readiness_detail})",
                    level,
                )
                result.arm_message = self._prepare_stack_start_pose(goal_handle)

            self._publish_box_stack_feedback(
                goal_handle,
                "CLOSING_GRIPPERS",
                "closing both grippers around the manually loaded box",
                level,
            )
            self._publish_both_grippers(
                goal_handle, self._float("gripper_closed_position")
            )
            result.gripper_closed = True
            self._wait_delay(
                goal_handle,
                self._float("gripper_settle_sec"),
                "while waiting for the manually loaded box to be gripped",
            )

            self._publish_box_stack_feedback(
                goal_handle,
                "LOWERING_TORSO",
                f"holding both arms fixed while lowering the torso to level {level}",
                level,
            )
            self._move_stack_torso(
                goal_handle,
                target_torso,
                f"lowering the held box to stack level {level}",
            )
            fixed_message = "both arms remained at the fixed loading posture"
            result.arm_message = (
                f"start={result.arm_message}; {fixed_message}"
                if result.arm_message
                else fixed_message
            )

            self._publish_box_stack_feedback(
                goal_handle,
                "WAITING_TO_RELEASE",
                "torso target confirmed; holding the box for "
                f"{self._float('stack_release_delay_sec'):.1f}s before release",
                level,
            )
            self._wait_delay(
                goal_handle,
                self._float("stack_release_delay_sec"),
                "while holding the confirmed stack posture before release",
            )

            self._publish_box_stack_feedback(
                goal_handle,
                "OPENING_GRIPPERS",
                "opening both grippers after the place trajectory completed",
                level,
            )
            self._publish_both_grippers(
                goal_handle, self._float("gripper_open_position")
            )
            result.gripper_opened = True
            self._wait_delay(
                goal_handle,
                self._float("gripper_settle_sec"),
                "while waiting for the stacked box to be released",
            )

            self._publish_box_stack_feedback(
                goal_handle,
                "RETREATING",
                "raising the torso back to the highest loading posture so both "
                "grippers clear the released box",
                level,
            )
            self._move_stack_torso(
                goal_handle,
                pickup_torso,
                "raising both fixed arms clear of the released box",
            )
            result.retreat_completed = True

            self._publish_box_stack_feedback(
                goal_handle,
                "RETURNING_READY",
                "highest manual-load posture restored; arms stayed fixed",
                level,
            )
            result.ready_completed = True

            result.success = True
            result.message = (
                f"box stack level {level} completed by closed-loop torso motion"
            )
            self._finalize_action_result(result, started_at, "execute_box_stack")
            self._publish_box_stack_feedback(
                goal_handle, "DONE", result.message, level
            )
            goal_handle.succeed()
            return result
        except MissionCanceled as exc:
            result.success = False
            result.message = str(exc)
            self._publish_box_stack_feedback(
                goal_handle, "CANCELLED", result.message, level
            )
            goal_handle.canceled()
            return result
        except MissionError as exc:
            result.success = False
            result.message = str(exc)
            self.get_logger().error(result.message)
            self._publish_box_stack_feedback(
                goal_handle, "FAILED", result.message, level
            )
            goal_handle.abort()
            return result
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.message = f"unexpected box stack mission error: {exc}"
            self.get_logger().error(result.message)
            self._publish_box_stack_feedback(
                goal_handle, "FAILED", result.message, level
            )
            goal_handle.abort()
            return result
        finally:
            self._release_goal()
            self._finalize_action_result(
                result, started_at, "execute_box_stack"
            )

    def _recover_grasp_observation(
        self, goal_handle, arm: str, reason: str, dry_run: bool
    ) -> None:
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

    def _execute_grasp(self, goal_handle) -> ExecuteGrasp.Result:
        started_at = time.monotonic()
        request = goal_handle.request
        arm = self._resolve_arm(request.arm)
        result = ExecuteGrasp.Result()
        result.arm = arm
        torso_prepared = False
        joint_preparation_started = False
        joint_preparation_complete = False
        motion_state = {"started": False}
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
                if self._boolean("open_gripper_before_grasp"):
                    self._publish_grasp_feedback(
                        goal_handle,
                        "OPENING_INITIAL_GRIPPERS",
                        "opening both grippers before moving to the grasp "
                        "observation posture",
                        arm,
                    )
                    self._prepare_grasp_grippers(goal_handle)
                observation_ready, readiness_detail = (
                    self._grasp_observation_ready()
                )
                if observation_ready:
                    self._publish_grasp_feedback(
                        goal_handle,
                        "GRASP_OBSERVATION_READY",
                        "arms and torso are already at the grasp observation "
                        f"posture; skipping preparation ({readiness_detail})",
                        arm,
                    )
                else:
                    self._publish_grasp_feedback(
                        goal_handle,
                        "PREPARING",
                        "grasp observation posture is not ready; moving "
                        "through the intermediate arms, then entering the "
                        f"final observation posture ({readiness_detail})",
                        arm,
                    )
                    torso_prepared = True
                    joint_preparation_started = True
                    self._prepare_grasp_concurrently(
                        goal_handle,
                        False,
                    )
                    joint_preparation_complete = True

            close_check_enabled = (
                not request.dry_run
                and self._boolean("grasp_close_check_enabled")
            )
            max_grasp_attempts = (
                self._integer("grasp_max_empty_close_attempts")
                if close_check_enabled
                else 1
            )
            unlimited_empty_grasp_retries = (
                close_check_enabled and max_grasp_attempts == 0
            )
            measured_gripper_position: Optional[float] = None
            grasp_attempt = 0

            while True:
                grasp_attempt += 1
                attempt_label = (
                    str(grasp_attempt)
                    if unlimited_empty_grasp_retries
                    else f"{grasp_attempt}/{max_grasp_attempts}"
                )
                detection, grasp_pose_execution, arm_message = (
                    self._detect_and_execute_grasp(
                        goal_handle, request, arm, motion_state
                    )
                )
                result.grasp_pose = grasp_pose_execution
                result.score = float(detection.score)
                result.width = float(detection.width)
                result.height = float(detection.height)
                result.depth = float(detection.depth)
                result.object_id = int(detection.object_id)
                if request.publish_pose:
                    self.grasp_pose_publisher.publish(grasp_pose_execution)
                result.arm_message = arm_message

                if request.dry_run:
                    self._publish_grasp_feedback(
                        goal_handle,
                        "DRY_RUN_COMPLETE",
                        "arm plan succeeded; direct close and observation "
                        "return were skipped",
                        arm,
                    )
                    break

                self._publish_grasp_feedback(
                    goal_handle,
                    "CLOSING_GRIPPER",
                    f"closing selected gripper (grasp attempt "
                    f"{attempt_label})",
                    arm,
                )
                feedback_sequence = self._gripper_feedback_sequence(arm)
                self._publish_gripper(
                    goal_handle, arm, self._float("gripper_closed_position")
                )
                result.gripper_command_published = True
                self._wait_delay(
                    goal_handle,
                    self._float("gripper_settle_sec"),
                    "while waiting for gripper close",
                )

                if close_check_enabled:
                    measured_gripper_position = (
                        self._wait_for_gripper_close_feedback(
                            goal_handle, arm, feedback_sequence
                        )
                    )
                    close_ratio = self._gripper_close_ratio(
                        measured_gripper_position
                    )
                    empty_close_ratio_threshold = self._float(
                        "grasp_empty_close_ratio_threshold"
                    )
                    if close_ratio > empty_close_ratio_threshold:
                        detail = (
                            f"{arm} gripper closed too far without retaining "
                            f"material: "
                            f"measured={measured_gripper_position:.3f}, "
                            f"close_ratio={close_ratio:.1%}, "
                            f"empty_threshold="
                            f"{empty_close_ratio_threshold:.1%}, grasp "
                            f"attempt={attempt_label}"
                        )
                        self.get_logger().warning(detail)
                        self._publish_grasp_feedback(
                            goal_handle,
                            "EMPTY_GRASP_DETECTED",
                            detail,
                            arm,
                        )
                        self._publish_gripper(
                            goal_handle,
                            arm,
                            self._float("gripper_open_position"),
                        )
                        self._wait_delay(
                            goal_handle,
                            self._float("gripper_settle_sec"),
                            "while reopening after an empty grasp",
                        )
                        self._recover_grasp_observation(
                            goal_handle,
                            arm,
                            "empty grasp detected",
                            False,
                        )
                        result.torso_reset_command_published = True
                        if (
                            not unlimited_empty_grasp_retries
                            and grasp_attempt >= max_grasp_attempts
                        ):
                            raise MissionError(
                                f"grasp failed after {max_grasp_attempts} empty "
                                f"grasp attempts; {detail}"
                            )
                        self._publish_grasp_feedback(
                            goal_handle,
                            "RETRYING_EMPTY_GRASP",
                            "observation posture restored; requesting a fresh "
                            "detection for the next grasp attempt",
                            arm,
                        )
                        continue

                    self._publish_grasp_feedback(
                        goal_handle,
                        "GRASP_CONFIRMED",
                        f"material retained an opening in the {arm} gripper "
                        f"(measured={measured_gripper_position:.3f}, "
                        f"close_ratio={close_ratio:.1%}, "
                        f"empty_threshold="
                        f"{empty_close_ratio_threshold:.1%})",
                        arm,
                    )

                self._publish_grasp_feedback(
                    goal_handle,
                    "RETURNING_TO_OBSERVATION",
                    "returning directly to the final grasp observation posture",
                    arm,
                )
                self._prepare_grasp_arms_and_torso(goal_handle)
                result.torso_reset_command_published = True

                if close_check_enabled:
                    self._publish_grasp_feedback(
                        goal_handle,
                        "FINAL_GRASP_VERIFICATION",
                        "observation posture restored; reasserting the closed "
                        "gripper command and reading fresh feedback before "
                        "completing the action",
                        arm,
                    )
                    final_feedback_sequence = (
                        self._gripper_feedback_sequence(arm)
                    )
                    self._publish_gripper(
                        goal_handle,
                        arm,
                        self._float("gripper_closed_position"),
                    )
                    self._wait_delay(
                        goal_handle,
                        self._float("gripper_settle_sec"),
                        "while settling the final grasp verification",
                    )
                    measured_gripper_position = (
                        self._wait_for_gripper_close_feedback(
                            goal_handle,
                            arm,
                            final_feedback_sequence,
                        )
                    )
                    close_ratio = self._gripper_close_ratio(
                        measured_gripper_position
                    )
                    empty_close_ratio_threshold = self._float(
                        "grasp_empty_close_ratio_threshold"
                    )
                    if close_ratio > empty_close_ratio_threshold:
                        detail = (
                            f"final grasp verification detected an empty grasp "
                            f"after returning to observation: {arm} gripper "
                            f"closed more than "
                            f"{empty_close_ratio_threshold:.1%}, indicating "
                            f"that no material remains between the fingers "
                            f"(measured={measured_gripper_position:.3f}, "
                            f"close_ratio={close_ratio:.1%})"
                        )
                        self.get_logger().warning(detail)
                        self._publish_grasp_feedback(
                            goal_handle,
                            "FINAL_EMPTY_GRASP_DETECTED",
                            detail,
                            arm,
                        )
                        self._publish_gripper(
                            goal_handle,
                            arm,
                            self._float("gripper_open_position"),
                        )
                        self._wait_delay(
                            goal_handle,
                            self._float("gripper_settle_sec"),
                            "while reopening after final empty-grasp "
                            "verification",
                        )
                        if (
                            not unlimited_empty_grasp_retries
                            and grasp_attempt >= max_grasp_attempts
                        ):
                            raise MissionError(
                                f"grasp failed after {max_grasp_attempts} "
                                f"empty grasp attempts; {detail}"
                            )
                        self._publish_grasp_feedback(
                            goal_handle,
                            "RETRYING_FINAL_EMPTY_GRASP",
                            "gripper reopened at the observation posture; "
                            "requesting a fresh detection for the next grasp "
                            "attempt",
                            arm,
                        )
                        continue

                    self._publish_grasp_feedback(
                        goal_handle,
                        "FINAL_GRASP_CONFIRMED",
                        f"material remains between the gripper fingers after "
                        f"returning to observation "
                        f"(measured={measured_gripper_position:.3f}, "
                        f"close_ratio={close_ratio:.1%}, "
                        f"empty_threshold="
                        f"{empty_close_ratio_threshold:.1%})",
                        arm,
                    )
                break

            result.success = True
            result.message = (
                "grasp dry run completed"
                if request.dry_run
                else (
                    "grasp mission completed"
                    if measured_gripper_position is None
                    else "grasp mission completed; retained-object gripper "
                    f"close ratio confirmed at {close_ratio:.1%} "
                    f"(measured={measured_gripper_position:.3f})"
                )
            )
            self._finalize_action_result(result, started_at, "execute_grasp")
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
                    not motion_state["started"],
                    not joint_preparation_started or joint_preparation_complete,
                )
            )
            if not completed and not request.dry_run and safe_to_reset:
                result.torso_reset_command_published = self._safe_pre_arm_torso_reset(
                    goal_handle
                )
            self._release_goal()
            self._finalize_action_result(result, started_at, "execute_grasp")

    def _execute_place(self, goal_handle) -> ExecutePlace.Result:
        started_at = time.monotonic()
        request = goal_handle.request
        arm = self._resolve_arm(request.arm)
        result = ExecutePlace.Result()
        result.arm = arm

        try:
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
                self._publish_place_feedback(
                    goal_handle,
                    "RELEASE_COMPLETE",
                    "gripper open command published; post-release arm and "
                    "torso motions are disabled",
                    arm,
                )

            result.home_completed = False
            result.torso_reset_command_published = False

            result.success = True
            result.message = (
                "place dry run completed; post-release motions skipped"
                if request.dry_run
                else "place mission completed immediately after gripper release; "
                "post-release motions skipped"
            )
            self._finalize_action_result(result, started_at, "execute_place")
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
            self._release_goal()
            self._finalize_action_result(result, started_at, "execute_place")

    def _execute_move_chassis(self, goal_handle) -> MoveChassis.Result:
        started_at = time.monotonic()
        request = goal_handle.request
        direction = request.direction.strip().lower()
        speed = self._float(
            "chassis_angular_speed"
            if direction in ANGULAR_CHASSIS_DIRECTIONS
            else "chassis_linear_speed"
        )
        duration_sec = self._float("chassis_move_duration_sec")
        result = MoveChassis.Result()
        result.direction = direction
        result.speed = speed
        result.duration_sec = duration_sec

        try:
            self._publish_move_chassis_feedback(
                goal_handle,
                "STARTING",
                f"starting {direction} chassis motion at "
                f"{speed:.3f} for {duration_sec:.3f}s",
                0.0,
            )
            self._move_chassis_for_duration(
                goal_handle,
                direction,
                speed,
                duration_sec,
            )

            result.success = True
            result.message = "chassis motion completed"
            self._finalize_action_result(result, started_at, "move_chassis")
            self._publish_move_chassis_feedback(
                goal_handle, "DONE", result.message, 1.0
            )
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
            result.message = f"unexpected chassis motion error: {exc}"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result
        finally:
            self._publish_zero_chassis()
            self._release_goal()
            self._finalize_action_result(result, started_at, "move_chassis")


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
