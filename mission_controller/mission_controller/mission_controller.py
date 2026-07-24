import math
import threading
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
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
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformBroadcaster,
    TransformListener,
)
from visualization_msgs.msg import MarkerArray

from .common import (
    CHASSIS_DIRECTIONS,
    VALID_ARMS,
    GraspCandidate,
    MissionCanceled,
    MissionError,
    PickupAttemptError,
    TaskActionError,
    TwoStageMotionError,
    compose_poses,
    interpolate_pose,
    pose_to_array,
    quaternion_multiply,
    rotate_vector,
)
from .box_actions import BoxActionsMixin
from .box_support import BoxSupportMixin
from .chassis_action import ChassisActionMixin
from .chassis_support import ChassisSupportMixin
from .action_runtime import ActionRuntimeMixin
from .grasp_support import GraspSupportMixin
from .material_actions import MaterialActionsMixin
from .stack_action import StackActionMixin
from .stack_support import StackSupportMixin

__all__ = [
    "MissionController",
    "MissionCanceled",
    "MissionError",
    "PickupAttemptError",
    "TaskActionError",
    "TwoStageMotionError",
    "GraspCandidate",
    "compose_poses",
    "interpolate_pose",
    "pose_to_array",
    "quaternion_multiply",
    "rotate_vector",
    "main",
]


class MissionController(
    ActionRuntimeMixin,
    BoxSupportMixin,
    GraspSupportMixin,
    StackSupportMixin,
    ChassisSupportMixin,
    BoxActionsMixin,
    MaterialActionsMixin,
    StackActionMixin,
    ChassisActionMixin,
    Node,
):
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



















    def _resolve_arm(self, requested_arm: str) -> str:
        arm = requested_arm.strip().lower()
        return arm or self._string("default_arm").lower()

    def _reserve_goal(
        self,
        mission: str,
        request_id: str,
        arm: Optional[str] = None,
    ) -> GoalResponse:
        if arm is not None and arm not in VALID_ARMS:
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
        detail = f"accepted {mission} goal request_id={request_id or '<empty>'}"
        if arm is not None:
            detail += f" arm={arm}"
        self.get_logger().info(detail)
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
        return self._reserve_goal("box_grasp", request.request_id)

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
        return self._reserve_goal("box_place", request.request_id)

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
