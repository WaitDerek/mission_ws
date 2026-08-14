import math
import threading
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from mission_interfaces.action import (
    ExecuteAdaptiveBoxGrasp,
    ExecuteBoxGrasp,
    ExecuteBoxPlace,
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
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rm_robot_interfaces.msg import ArmSlaveData, BodyData
from rm_robot_interfaces.srv import StringCmd
from sensor_msgs.msg import JointState
from task_interfaces.action import (
    GoReady,
    MoveArmJoints,
    PickupTask,
)
try:
    from task_interfaces.srv import MoveCartesian
except ImportError:
    MoveCartesian = None
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformListener,
)
from .common import (
    MissionCanceled,
    MissionError,
    PickupAttemptError,
    TaskActionError,
    pose_to_array,
    quaternion_multiply,
    rotate_vector,
)
from .box_actions import BoxActionsMixin
from .box_support import BoxSupportMixin
from .adaptive_box_actions import AdaptiveBoxActionsMixin
from .adaptive_box_support import AdaptiveBoxSupportMixin
from .action_runtime import ActionRuntimeMixin
from .realman_sdk_adapter import RealManSdkAdapter

__all__ = [
    "MissionController",
    "MissionCanceled",
    "MissionError",
    "PickupAttemptError",
    "TaskActionError",
    "pose_to_array",
    "quaternion_multiply",
    "rotate_vector",
    "main",
]


class MissionController(
    ActionRuntimeMixin,
    BoxSupportMixin,
    AdaptiveBoxSupportMixin,
    BoxActionsMixin,
    AdaptiveBoxActionsMixin,
    Node,
):
    def __init__(self) -> None:
        super().__init__("mission_controller")
        self._declare_parameters()
        self._validate_parameters()
        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=self._float("adaptive_tf_cache_time_sec"))
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_camera_mount_tf()

        # Match the command transport used by dual_arm_manipulation.
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
        self.latest_body_joint1_position: Optional[float] = None
        self.latest_body_joint1_velocity: Optional[float] = None
        self.latest_body_joint1_state_time = 0.0
        self.latest_body_joint1_state_sequence = 0
        self.latest_body_joint_positions: dict[str, float] = {}
        self.latest_body_joint_velocities: dict[str, float] = {}
        self.latest_body_state_time = 0.0
        self.latest_body_state_sequence = 0
        self.latest_slave_arm_positions = {"left": [], "right": []}
        self.latest_slave_arm_velocities = {"left": [], "right": []}
        self.latest_slave_arm_state_times = {"left": 0.0, "right": 0.0}
        self.latest_slave_arm_state_sequences = {"left": 0, "right": 0}
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
        self.body_joint1_feedback_subscription = self.create_subscription(
            BodyData,
            self._string("box_joint1_feedback_topic"),
            self._body_joint1_feedback_callback,
            feedback_qos,
        )
        self.left_slave_arm_feedback_subscription = self.create_subscription(
            ArmSlaveData,
            self._string("box_post_arm_left_feedback_topic"),
            lambda message: self._slave_arm_feedback_callback("left", message),
            feedback_qos,
        )
        self.right_slave_arm_feedback_subscription = self.create_subscription(
            ArmSlaveData,
            self._string("box_post_arm_right_feedback_topic"),
            lambda message: self._slave_arm_feedback_callback("right", message),
            feedback_qos,
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
        self.arm_joints_client = ActionClient(
            self,
            MoveArmJoints,
            self._string("arm_joints_service_name"),
            callback_group=self.client_group,
        )
        self.go_ready_client = ActionClient(
            self,
            GoReady,
            self._string("go_ready_action_name"),
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
                f"be rejected: "
                f"{OBJECT_POSE_IMPORT_ERROR}"
            )
        self.pickup_task_client = ActionClient(
            self,
            PickupTask,
            self._string("pickup_task_action_name"),
            callback_group=self.client_group,
        )
        self.direct_movel_client = None
        self.body_command_client = self.create_client(
            StringCmd,
            self._string("box_joint1_command_service_name"),
            callback_group=self.client_group,
        )
        self.direct_sdk_adapter = None
        motion_backend = self._string("direct_motion_backend").lower()
        if motion_backend == "ros_service":
            if MoveCartesian is None:
                raise RuntimeError(
                    "direct_motion_backend=ros_service requires "
                    "task_interfaces.srv.MoveCartesian"
                )
            self.direct_movel_client = self.create_client(
                MoveCartesian,
                self._string("direct_movel_service_name"),
                callback_group=self.client_group,
            )
        elif motion_backend == "python_sdk" and (
            self._boolean("box_direct_movel_enabled")
            or self._boolean("adaptive_box_action_enabled")
        ):
            self.direct_sdk_adapter = RealManSdkAdapter(
                sdk_root=self._string("direct_sdk_root"),
                left_ip=self._string("direct_sdk_left_ip"),
                right_ip=self._string("direct_sdk_right_ip"),
                port=self._integer("direct_sdk_port"),
                connect_level=self._integer("direct_sdk_connect_level"),
                logger=self.get_logger(),
            )
        self.state_lock = threading.Lock()
        self.mission_reserved = False
        self.active_mission = ""
        self.active_arm_joints_goal_handle = None
        self.active_go_ready_goal_handle = None
        self.active_box_object_pose_goal_handle = None
        self.active_pickup_task_goal_handle = None

        self.adaptive_box_grasp_action_server = ActionServer(
            self,
            ExecuteAdaptiveBoxGrasp,
            self._string("execute_adaptive_box_grasp_action_name"),
            execute_callback=self._execute_adaptive_box_grasp,
            goal_callback=self._adaptive_box_grasp_goal_callback,
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
        self.get_logger().info(
            "mission controller ready: "
            f"adaptive_box_grasp="
            f"{self._string('execute_adaptive_box_grasp_action_name')} "
            f"box_grasp={self._string('execute_box_grasp_action_name')} "
            f"box_place={self._string('execute_box_place_action_name')}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    "execute_adaptive_box_grasp_action_name",
                    "/execute_adaptive_box_grasp",
                ),
                ("execute_box_grasp_action_name", "/execute_box_grasp"),
                ("execute_box_place_action_name", "/execute_box_place"),
                ("adaptive_box_action_enabled", True),
                ("adaptive_freeze_frame", "base_link"),
                ("adaptive_require_detection_timestamp", True),
                ("adaptive_tf_cache_time_sec", 180.0),
                ("adaptive_detection_tf_timeout_sec", 5.0),
                ("adaptive_runtime_tf_timeout_sec", 2.0),
                # Canonical object axes after pose normalization are X=down,
                # Y=forward, Z=right. Grasp from the two object-Z side faces.
                ("adaptive_grasp_span_axis_object", [0.0, 0.0, 1.0]),
                ("adaptive_grasp_height_axis_object", [-1.0, 0.0, 0.0]),
                ("adaptive_grasp_side_clearance_m", 0.0),
                ("adaptive_grasp_height_offset_m", 0.0),
                (
                    "adaptive_grasp_correction_rpy",
                    [-1.5707963267948966, 1.5707963267948966, 0.0],
                ),
                (
                    "adaptive_left_grasp_extra_rpy",
                    [0.0, 0.0, 3.141592653589793],
                ),
                ("adaptive_right_grasp_extra_rpy", [0.0, 0.0, 0.0]),
                ("adaptive_grasp_velocity_percent", 5.0),
                ("adaptive_grasp_timeout_sec", 120.0),
                ("adaptive_lift_distance_m", 0.10),
                ("adaptive_lift_velocity_percent", 5.0),
                ("adaptive_lift_timeout_sec", 120.0),
                ("box_mission_enabled", True),
                ("box_object_pose_action_name", "/object_pose/estimate"),
                ("box_object_pose_topic", "/mission/box_object_pose"),
                ("box_object_pose_camera_topic", "/object_pose/pose"),
                ("box_object_pose_raw_topic", "/mission/box_object_pose_raw"),
                ("box_object_pose_model_label", "smallbox"),
                ("box_object_pose_instance_index", 0),
                ("box_object_pose_confidence_threshold", 0.25),
                ("box_object_pose_result_timeout_sec", 120.0),
                ("box_camera_pose_axis_min_dot", 0.5),
                ("pickup_task_action_name", "/pickup_task"),
                ("pickup_task_result_timeout_sec", 120.0),
                ("box_direct_movel_enabled", True),
                ("direct_motion_backend", "python_sdk"),
                ("direct_movel_service_name", "/realbot/movel"),
                (
                    "direct_sdk_root",
                    "RM_API2/Demo/RMDemo_Python/RMDemo_SimpleProcess",
                ),
                ("direct_sdk_left_ip", "192.168.127.18"),
                ("direct_sdk_right_ip", "192.168.127.19"),
                ("direct_sdk_port", 8080),
                ("direct_sdk_connect_level", 3),
                ("direct_sdk_motion_timeout_sec", 120.0),
                ("box_grasp_execution_mode", "joint123_then_arms"),
                ("box_joint1_command_service_name", "/robot/command"),
                ("box_joint1_feedback_topic", "/mcap/body"),
                ("box_joint1_name", "joint1"),
                ("box_joint2_name", "joint2"),
                ("box_joint3_name", "joint3"),
                ("box_joint4_name", "joint4"),
                ("box_joint5_name", "joint5"),
                ("box_joint1_device", 2),
                ("box_joint1_detection_angle_deg", 0.0),
                ("box_joint1_approach_angle_deg", 12.0),
                ("box_joint2_detection_angle_deg", 0.0),
                ("box_joint2_approach_angle_deg", 0.0),
                ("box_joint3_detection_angle_deg", 0.0),
                ("box_joint3_approach_angle_deg", 0.0),
                ("box_joint1_command_units_per_degree", 1000.0),
                ("box_body_command_units_per_degree", [1000.0] * 5),
                ("box_joint1_velocity", 10),
                ("box_joint1_blend_radius", 0),
                ("box_joint1_axis_xyz", [0.0, 0.0, 1.0]),
                ("box_joint1_feedback_to_geometric_sign", -1.0),
                ("box_joint2_axis_xyz", [0.0, 0.0000118, 1.0]),
                ("box_joint3_axis_xyz", [0.0, -0.0000118, -1.0]),
                ("box_joint2_feedback_to_urdf_axis_sign", 1.0),
                ("box_joint3_feedback_to_urdf_axis_sign", 1.0),
                ("box_joint1_to_joint2_xyz", [0.0, 0.255, 0.0]),
                (
                    "box_joint1_to_joint2_rotation",
                    [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                ),
                ("box_joint2_to_joint3_xyz", [0.0, 0.255, 0.0]),
                (
                    "box_joint2_to_joint3_rotation",
                    [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                ),
                ("box_joint1_to_left_base_xyz", [0.0, 1.05, -0.05]),
                ("box_joint1_to_right_base_xyz", [0.0, 1.05, 0.05]),
                (
                    "box_joint1_to_left_base_rotation",
                    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                ),
                (
                    "box_joint1_to_right_base_rotation",
                    [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                ("box_joint1_position_tolerance_rad", 0.01),
                ("box_joint1_velocity_tolerance_rad_sec", 0.01),
                ("box_joint1_feedback_max_age_sec", 1.0),
                ("box_joint1_wait_timeout_sec", 40.0),
                ("box_joint1_stable_samples", 3),
                (
                    "direct_movel_target_mode",
                    "camera_offset_box_orientation",
                ),
                ("direct_movel_box_relative_model_label", "smallbox"),
                ("direct_movel_motion_mode", "movej_p"),
                ("direct_movel_velocity_percent", 5.0),
                ("direct_movel_blocking", True),
                ("box_post_movel_enabled", False),
                ("box_post_movel_velocity_percent", 5.0),
                ("box_post_movel_step_count", 4),
                ("box_post_movel_left_step1_xyz", [0.0, 0.0, 0.08]),
                ("box_post_movel_right_step1_xyz", [0.0, 0.0, -0.03]),
                ("box_post_movel_left_step2_xyz", [0.12, 0.0, 0.0]),
                ("box_post_movel_right_step2_xyz", [0.12, 0.0, 0.0]),
                ("box_post_movel_left_step3_xyz", [-0.12, 0.0, 0.0]),
                ("box_post_movel_right_step3_xyz", [-0.12, 0.0, 0.0]),
                ("box_post_movel_left_step4_xyz", [0.0, 0.0, -0.1]),
                ("box_post_movel_right_step4_xyz", [0.0, 0.0, 0.15]),
                ("box_post_movel_left_step5_xyz", [0.0, 0.0, 0.0]),
                ("box_post_movel_right_step5_xyz", [0.0, 0.0, 0.0]),
                ("box_post_arm_movej_enabled", False),
                ("box_post_arm_movej_left_device", 0),
                ("box_post_arm_movej_right_device", 1),
                (
                    "box_post_arm_movej_left_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                (
                    "box_post_arm_movej_right_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                ("box_post_arm_movej_command_units_per_degree", 1000.0),
                ("box_post_arm_movej_velocity", 5),
                ("box_post_arm_movej_blend_radius", 0),
                ("box_post_arm_movej_trajectory_connect", 0),
                ("box_post_arm_movej_timeout_sec", 40.0),
                ("box_post_arm_left_feedback_topic", "/mcap/slave_arm_left"),
                ("box_post_arm_right_feedback_topic", "/mcap/slave_arm_right"),
                ("box_post_arm_position_tolerance_rad", 0.01),
                ("box_post_arm_velocity_tolerance_rad_sec", 0.01),
                ("box_post_arm_feedback_max_age_sec", 1.0),
                ("box_post_arm_stable_samples", 3),
                ("box_body_return_home_enabled", False),
                ("box_body_home_joint_units", [0, 0, 0, 0, 0]),
                ("box_body_home_velocity", 10),
                ("box_body_home_blend_radius", 0),
                ("box_body_home_timeout_sec", 40.0),
                ("direct_movel_use_current_fixture_orientation", False),
                (
                    "direct_movel_left_fixed_link8_orientation",
                    [-0.497, -0.503, -0.488, 0.509],
                ),
                (
                    "direct_movel_right_fixed_link8_orientation",
                    [0.482, -0.463, 0.522, 0.528],
                ),
                ("direct_movel_fixture_compensation_enabled", True),
                ("direct_movel_left_offset_xyz", [0.0, 0.05, -0.46]),
                ("direct_movel_right_offset_xyz", [0.0, 0.0, 0.35]),
                (
                    "direct_movel_left_box_to_link8_orientation",
                    [0.529174705702, 0.468855850183, -0.519042875374, 0.480356967388],
                ),
                (
                    "direct_movel_right_box_to_link8_orientation",
                    [0.495125041989, -0.526225123507, 0.461789015629, 0.514479559584],
                ),
                (
                    "left_fixture_center_in_link8_xyz",
                    [-0.07, -0.11, 0.03],
                ),
                (
                    "right_fixture_center_in_link8_xyz",
                    [-0.07, 0.11, 0.03],
                ),
                ("box_detection_attempts", 2),
                ("box_width", 0.357),
                ("box_height", 0.127),
                ("box_type", "f455"),
                # FoundationPose publishes the oriented-bounding-box centre
                # with F320 axes X=down, Y=depth, Z=width. Keep those axes by
                # default; each robot pickup profile owns the fixed transform
                # from model axes to its operation/end-effector convention.
                (
                    "box_foundation_to_pickup_rpy",
                    [0.0, 0.0, 0.0],
                ),
                ("arm_joints_service_name", "/move_arm_j"),
                ("go_ready_action_name", "/go_ready"),
                ("torso_topic", "/realbot/motion_target/torso"),
                (
                    "left_gripper_topic",
                    "/realbot/motion_target/gripper_left",
                ),
                (
                    "right_gripper_topic",
                    "/realbot/motion_target/gripper_right",
                ),
                ("joint_state_topic", "/joint_states"),
                ("torso_feedback_topic", "/realbot/feedback/torso"),
                (
                    "left_gripper_feedback_topic",
                    "/realbot/feedback/gripper_left",
                ),
                (
                    "right_gripper_feedback_topic",
                    "/realbot/feedback/gripper_right",
                ),
                (
                    "left_arm_joint_names",
                    [f"L_JOINT_{index}" for index in range(1, 8)],
                ),
                (
                    "right_arm_joint_names",
                    [f"R_JOINT_{index}" for index in range(1, 8)],
                ),
                ("verify_arm_joint_targets", True),
                ("arm_joint_target_tolerance", 0.10),
                ("arm_joint_target_wait_timeout_sec", 20.0),
                ("arm_joint_target_stable_samples", 3),
                ("box_observation_ready_check_enabled", True),
                ("box_observation_feedback_max_age_sec", 2.0),
                ("box_observation_torso_tolerance", 0.10),
                ("torso_target_tolerance", 0.03),
                ("torso_target_wait_timeout_sec", 40.0),
                ("torso_target_stable_samples", 3),
                ("arm_execution_frame", "base_link"),
                ("left_ee_frame", "L_Link_7"),
                ("right_ee_frame", "R_Link_7"),
                ("left_gripper_frame", "L_Link_8"),
                ("right_gripper_frame", "R_Link_8"),
                ("camera_mount_tf_enabled", False),
                ("camera_mount_parent_frame", "R_Link_8"),
                (
                    "camera_mount_child_frame",
                    "realbot_camera_link",
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
                ("camera_measured_extrinsics_enabled", True),
                ("left_arm_base_frame", "L_base_Link"),
                ("right_arm_base_frame", "R_base_Link"),
                ("left_link8_frame", "L_Link_8"),
                ("right_link8_frame", "R_Link_8"),
                # Use the measured camera origin directly. The J1-height
                # difference between CAD models is not an external-camera
                # calibration measurement.
                ("camera_left_base_xyz", [0.045, 0.08, -0.05]),
                ("camera_right_base_xyz", [0.045, -0.08, -0.08]),
                (
                    "camera_left_base_rpy",
                    [0.0, 1.5707963267948966, 1.5707963267948966],
                ),
                (
                    "camera_right_base_rpy",
                    [0.0, -1.5707963267948966, 1.5707963267948966],
                ),
                ("camera_tf_timeout_sec", 2.0),
                ("dependency_wait_timeout_sec", 10.0),
                ("arm_joints_result_timeout_sec", 60.0),
                ("go_ready_result_timeout_sec", 60.0),
                ("wait_for_command_subscribers", True),
                ("require_command_subscribers", True),
                ("command_subscriber_wait_timeout_sec", 3.0),
                ("command_repeat_count", 10),
                ("command_repeat_interval_sec", 0.005),
                ("torso_settle_sec", 1.0),
                ("arm_settle_sec", 1.0),
                ("gripper_settle_sec", 1.0),
                ("box_detection_posture_settle_sec", 2.0),
                ("box_place_release_delay_sec", 2.0),
                ("torso_reset_positions", [0.0, 0.0, 0.0, 0.0]),
                ("torso_velocities", [0.1, 0.1, 0.1, 0.1]),
                # RealBot dual-arm preparation and clearance postures.
                (
                    "box_grasp_intermediate_left_joint_positions",
                    [1.30, 0.6, 0.0, -1.5, 0.0, 0.0, 0.0],
                ),
                (
                    "box_grasp_intermediate_right_joint_positions",
                    [1.30, -0.6, 0.0, -1.5, 0.0, 0.0, 0.0],
                ),
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
                ("gripper_open_position", 100.0),
                ("gripper_closed_position", 0.0),
                ("box_empty_close_ratio_threshold", 0.95),
                ("box_gripper_feedback_timeout_sec", 2.0),
                ("box_gripper_feedback_max_age_sec", 0.5),
            ],
        )

    def _validate_parameters(self) -> None:
        for name in (
            "execute_adaptive_box_grasp_action_name",
            "execute_box_grasp_action_name",
            "execute_box_place_action_name",
            "adaptive_freeze_frame",
            "box_object_pose_action_name",
            "box_object_pose_topic",
            "box_object_pose_camera_topic",
            "box_object_pose_raw_topic",
            "box_object_pose_model_label",
            "pickup_task_action_name",
            "direct_motion_backend",
            "direct_movel_service_name",
            "direct_sdk_root",
            "direct_sdk_left_ip",
            "direct_sdk_right_ip",
            "direct_movel_target_mode",
            "direct_movel_box_relative_model_label",
            "direct_movel_motion_mode",
            "box_grasp_execution_mode",
            "box_joint1_command_service_name",
            "box_joint1_feedback_topic",
            "box_joint1_name",
            "box_joint2_name",
            "box_joint3_name",
            "box_joint4_name",
            "box_joint5_name",
            "box_type",
            "arm_joints_service_name",
            "go_ready_action_name",
            "torso_topic",
            "left_gripper_topic",
            "right_gripper_topic",
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

        if self._string("adaptive_freeze_frame").lstrip("/") != "base_link":
            raise ValueError(
                "adaptive_freeze_frame must be base_link while the chassis is fixed"
            )

        for name in ("left_arm_joint_names", "right_arm_joint_names"):
            joint_names = self._string_array(name)
            if len(joint_names) != 7 or len(set(joint_names)) != 7:
                raise ValueError(
                    f"parameter '{name}' must contain 7 unique joint names"
                )

        motion_mode = self._string("direct_movel_motion_mode").lower()
        if motion_mode not in ("movel", "movej_p"):
            raise ValueError(
                "parameter 'direct_movel_motion_mode' must be 'movel' or 'movej_p'"
            )

        execution_mode = self._string("box_grasp_execution_mode").lower()
        if execution_mode not in (
            "arms_only",
            "joint1_then_arms",
            "joint1_then_arms_keep_position",
            "joint123_then_arms",
        ):
            raise ValueError(
                "parameter 'box_grasp_execution_mode' must be 'arms_only' "
                "or 'joint1_then_arms' or "
                "'joint1_then_arms_keep_position' or 'joint123_then_arms'"
            )

        target_mode = self._string("direct_movel_target_mode").lower()
        if target_mode not in (
            "camera_offset",
            "camera_offset_box_orientation",
        ):
            raise ValueError(
                "parameter 'direct_movel_target_mode' must be "
                "'camera_offset' or 'camera_offset_box_orientation'"
            )
        if target_mode == "camera_offset_box_orientation" and not self._boolean(
            "camera_measured_extrinsics_enabled"
        ):
            raise ValueError(
                f"direct_movel_target_mode={target_mode} requires "
                "camera_measured_extrinsics_enabled=true"
            )
        if (
            target_mode == "camera_offset_box_orientation"
            and self._string("box_object_pose_model_label").strip().lower()
            != self._string("direct_movel_box_relative_model_label")
            .strip()
            .lower()
        ):
            raise ValueError(
                "box-orientation calibration model does not match "
                "box_object_pose_model_label"
            )

        motion_backend = self._string("direct_motion_backend").lower()
        if motion_backend not in ("ros_service", "python_sdk"):
            raise ValueError(
                "parameter 'direct_motion_backend' must be 'ros_service' "
                "or 'python_sdk'"
            )

        if self._boolean("camera_mount_tf_enabled"):
            for name in ("camera_mount_parent_frame", "camera_mount_child_frame"):
                if not self._string(name):
                    raise ValueError(f"parameter '{name}' must not be empty")

        for name, expected_length in (
            ("torso_reset_positions", 4),
            ("torso_velocities", 4),
            ("box_grasp_intermediate_left_joint_positions", 7),
            ("box_grasp_intermediate_right_joint_positions", 7),
            ("box_grasp_left_observation_joint_positions", 7),
            ("box_grasp_right_observation_joint_positions", 7),
            ("box_pickup_clearance_left_joint_positions", 7),
            ("box_pickup_clearance_right_joint_positions", 7),
            ("box_grasp_torso_prepare_positions", 4),
            ("box_grasp_torso_lift_positions", 4),
            ("box_place_torso_positions", 4),
            ("box_place_torso_straighten_intermediate_positions", 4),
            ("camera_mount_xyz", 3),
            ("camera_mount_rpy", 3),
            ("camera_mount_correction_rpy", 3),
            ("camera_left_base_xyz", 3),
            ("camera_right_base_xyz", 3),
            ("camera_left_base_rpy", 3),
            ("camera_right_base_rpy", 3),
            ("box_foundation_to_pickup_rpy", 3),
            ("direct_movel_left_offset_xyz", 3),
            ("direct_movel_right_offset_xyz", 3),
            ("box_post_movel_left_step1_xyz", 3),
            ("box_post_movel_right_step1_xyz", 3),
            ("box_post_movel_left_step2_xyz", 3),
            ("box_post_movel_right_step2_xyz", 3),
            ("box_post_movel_left_step3_xyz", 3),
            ("box_post_movel_right_step3_xyz", 3),
            ("box_post_movel_left_step4_xyz", 3),
            ("box_post_movel_right_step4_xyz", 3),
            ("box_post_movel_left_step5_xyz", 3),
            ("box_post_movel_right_step5_xyz", 3),
            ("direct_movel_left_box_to_link8_orientation", 4),
            ("direct_movel_right_box_to_link8_orientation", 4),
            ("direct_movel_left_fixed_link8_orientation", 4),
            ("direct_movel_right_fixed_link8_orientation", 4),
            ("left_fixture_center_in_link8_xyz", 3),
            ("right_fixture_center_in_link8_xyz", 3),
            ("box_joint1_axis_xyz", 3),
            ("box_joint1_to_left_base_xyz", 3),
            ("box_joint1_to_right_base_xyz", 3),
            ("box_joint1_to_left_base_rotation", 9),
            ("box_joint1_to_right_base_rotation", 9),
            ("box_joint1_to_joint2_xyz", 3),
            ("box_joint1_to_joint2_rotation", 9),
            ("box_joint2_to_joint3_xyz", 3),
            ("box_joint2_to_joint3_rotation", 9),
            ("box_joint2_axis_xyz", 3),
            ("box_joint3_axis_xyz", 3),
            ("box_body_command_units_per_degree", 5),
            ("adaptive_grasp_span_axis_object", 3),
            ("adaptive_grasp_height_axis_object", 3),
            ("adaptive_grasp_correction_rpy", 3),
            ("adaptive_left_grasp_extra_rpy", 3),
            ("adaptive_right_grasp_extra_rpy", 3),
            ("box_post_arm_movej_left_joint_units", 7),
            ("box_post_arm_movej_right_joint_units", 7),
            ("box_body_home_joint_units", 5),
        ):
            values = self._float_array(name)
            if len(values) != expected_length:
                raise ValueError(
                    f"parameter '{name}' must contain {expected_length} values"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"parameter '{name}' contains NaN or Inf")

        for name in ("adaptive_grasp_height_offset_m",):
            if not math.isfinite(self._float(name)):
                raise ValueError(f"parameter '{name}' must be finite")

        for name in (
            "box_joint1_detection_angle_deg",
            "box_joint1_approach_angle_deg",
            "box_joint1_feedback_to_geometric_sign",
            "box_joint2_detection_angle_deg",
            "box_joint2_approach_angle_deg",
            "box_joint3_detection_angle_deg",
            "box_joint3_approach_angle_deg",
            "box_joint2_feedback_to_urdf_axis_sign",
            "box_joint3_feedback_to_urdf_axis_sign",
        ):
            if not math.isfinite(self._float(name)):
                raise ValueError(f"parameter '{name}' must be finite")
        if self._float("box_joint1_feedback_to_geometric_sign") not in (
            -1.0,
            1.0,
        ):
            raise ValueError(
                "box_joint1_feedback_to_geometric_sign must be -1.0 or 1.0"
            )
        for name in (
            "box_joint2_feedback_to_urdf_axis_sign",
            "box_joint3_feedback_to_urdf_axis_sign",
        ):
            if self._float(name) not in (-1.0, 1.0):
                raise ValueError(f"{name} must be -1.0 or 1.0")
        if any(
            value <= 0.0
            for value in self._float_array("box_body_command_units_per_degree")
        ):
            raise ValueError(
                "box_body_command_units_per_degree values must be positive"
            )

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
        close_ratio = self._float("box_empty_close_ratio_threshold")
        if not 0.0 <= close_ratio <= 1.0:
            raise ValueError(
                "box_empty_close_ratio_threshold must be in [0, 1]"
            )

        positive_parameters = (
            "adaptive_tf_cache_time_sec",
            "adaptive_detection_tf_timeout_sec",
            "adaptive_runtime_tf_timeout_sec",
            "adaptive_grasp_velocity_percent",
            "adaptive_grasp_timeout_sec",
            "adaptive_lift_distance_m",
            "adaptive_lift_velocity_percent",
            "adaptive_lift_timeout_sec",
            "dependency_wait_timeout_sec",
            "arm_joints_result_timeout_sec",
            "go_ready_result_timeout_sec",
            "command_subscriber_wait_timeout_sec",
            "camera_tf_timeout_sec",
            "pickup_task_result_timeout_sec",
            "direct_movel_velocity_percent",
            "box_post_movel_velocity_percent",
            "direct_sdk_motion_timeout_sec",
            "box_width",
            "box_height",
            "arm_joint_target_tolerance",
            "arm_joint_target_wait_timeout_sec",
            "box_observation_feedback_max_age_sec",
            "box_observation_torso_tolerance",
            "torso_target_tolerance",
            "torso_target_wait_timeout_sec",
            "box_gripper_feedback_timeout_sec",
            "box_gripper_feedback_max_age_sec",
            "box_joint1_command_units_per_degree",
            "box_joint1_position_tolerance_rad",
            "box_joint1_velocity_tolerance_rad_sec",
            "box_joint1_feedback_max_age_sec",
            "box_joint1_wait_timeout_sec",
            "box_post_arm_movej_command_units_per_degree",
            "box_post_arm_position_tolerance_rad",
            "box_post_arm_velocity_tolerance_rad_sec",
            "box_post_arm_feedback_max_age_sec",
            "box_post_arm_movej_timeout_sec",
            "box_body_home_timeout_sec",
        )
        for name in positive_parameters:
            if not math.isfinite(self._float(name)) or self._float(name) <= 0.0:
                raise ValueError(f"parameter '{name}' must be finite and positive")

        for name in (
            "adaptive_grasp_velocity_percent",
            "adaptive_lift_velocity_percent",
            "box_post_movel_velocity_percent",
        ):
            if self._float(name) > 100.0:
                raise ValueError(f"{name} must be in (0, 100]")

        for name in (
            "direct_sdk_port",
            "direct_sdk_connect_level",
        ):
            if self._integer(name) <= 0:
                raise ValueError(f"parameter '{name}' must be positive")

        nonnegative_parameters = (
            "adaptive_grasp_side_clearance_m",
            "command_repeat_interval_sec",
            "torso_settle_sec",
            "arm_settle_sec",
            "gripper_settle_sec",
            "box_object_pose_result_timeout_sec",
            "box_detection_posture_settle_sec",
            "box_place_release_delay_sec",
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
        if self._integer("box_detection_attempts") <= 0:
            raise ValueError("box_detection_attempts must be positive")
        if self._integer("box_joint1_stable_samples") <= 0:
            raise ValueError("box_joint1_stable_samples must be positive")
        if not 0 <= self._integer("box_post_movel_step_count") <= 5:
            raise ValueError("box_post_movel_step_count must be in [0, 5]")
        if self._integer("box_post_arm_stable_samples") <= 0:
            raise ValueError("box_post_arm_stable_samples must be positive")
        if self._float("box_post_arm_movej_command_units_per_degree") <= 0.0:
            raise ValueError(
                "box_post_arm_movej_command_units_per_degree must be positive"
            )
        if self._integer("box_post_arm_movej_velocity") <= 0:
            raise ValueError("box_post_arm_movej_velocity must be positive")
        if self._integer("box_post_arm_movej_velocity") > 100:
            raise ValueError("box_post_arm_movej_velocity must be in (0, 100]")
        if self._integer("box_post_arm_movej_blend_radius") < 0:
            raise ValueError("box_post_arm_movej_blend_radius must be nonnegative")
        if self._integer("box_post_arm_movej_trajectory_connect") not in (0, 1):
            raise ValueError("box_post_arm_movej_trajectory_connect must be 0 or 1")
        if self._integer("box_post_arm_movej_left_device") < 0:
            raise ValueError("box_post_arm_movej_left_device must be nonnegative")
        if self._integer("box_post_arm_movej_right_device") < 0:
            raise ValueError("box_post_arm_movej_right_device must be nonnegative")
        if self._integer("box_body_home_velocity") <= 0:
            raise ValueError("box_body_home_velocity must be positive")
        if self._integer("box_body_home_blend_radius") < 0:
            raise ValueError("box_body_home_blend_radius must be nonnegative")
        if self._integer("box_joint1_device") <= 0:
            raise ValueError("box_joint1_device must be positive")
        if self._integer("box_joint1_velocity") <= 0:
            raise ValueError("box_joint1_velocity must be positive")
        if self._integer("box_joint1_velocity") > 100:
            raise ValueError("box_joint1_velocity must be in (0, 100]")
        if self._integer("box_joint1_blend_radius") < 0:
            raise ValueError("box_joint1_blend_radius must be nonnegative")
        if self._integer("box_object_pose_instance_index") < 0:
            raise ValueError("box_object_pose_instance_index must be nonnegative")

        box_confidence = self._float("box_object_pose_confidence_threshold")
        if not 0.0 <= box_confidence <= 1.0:
            raise ValueError(
                "box_object_pose_confidence_threshold must be in [0, 1]"
            )
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



















    def _reserve_goal(
        self,
        mission: str,
        request_id: str,
    ) -> GoalResponse:
        with self.state_lock:
            if self.mission_reserved:
                self.get_logger().warning(
                    f"rejecting {mission} goal: {self.active_mission} mission is active"
                )
                return GoalResponse.REJECT
            self.mission_reserved = True
            self.active_mission = mission
        self.get_logger().info(
            f"accepted {mission} goal request_id={request_id or '<empty>'}"
        )
        return GoalResponse.ACCEPT

    def _adaptive_box_grasp_goal_callback(
        self, request: ExecuteAdaptiveBoxGrasp.Goal
    ) -> GoalResponse:
        if request.target_instance_index < -1:
            self.get_logger().warning(
                "rejecting adaptive box grasp: target_instance_index must "
                "be -1 or nonnegative"
            )
            return GoalResponse.REJECT
        if self.box_object_pose_client is None:
            self.get_logger().warning(
                "rejecting adaptive box grasp: object_pose_interfaces is unavailable"
            )
            return GoalResponse.REJECT
        if not self._boolean("adaptive_box_action_enabled") and not request.dry_run:
            self.get_logger().warning(
                "rejecting adaptive box grasp: adaptive_box_action_enabled is false"
            )
            return GoalResponse.REJECT
        if not self._boolean("adaptive_box_action_enabled"):
            self.get_logger().info(
                "accepting adaptive box grasp dry-run while physical execution "
                "is disabled"
            )
        if (
            not request.dry_run
            and self._string("direct_motion_backend").lower() != "python_sdk"
        ):
            self.get_logger().warning(
                "rejecting adaptive box grasp: physical execution requires "
                "direct_motion_backend=python_sdk"
            )
            return GoalResponse.REJECT
        return self._reserve_goal("adaptive_box_grasp", request.task_id)

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


    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        if self.direct_sdk_adapter is not None:
            self.direct_sdk_adapter.stop_all()
        with self.state_lock:
            arm_joints_goal_handle = self.active_arm_joints_goal_handle
            go_ready_goal_handle = self.active_go_ready_goal_handle
            box_object_pose_goal_handle = self.active_box_object_pose_goal_handle
            pickup_task_goal_handle = self.active_pickup_task_goal_handle
        if arm_joints_goal_handle is not None:
            arm_joints_goal_handle.cancel_goal_async()
        if go_ready_goal_handle is not None:
            go_ready_goal_handle.cancel_goal_async()
        if box_object_pose_goal_handle is not None:
            box_object_pose_goal_handle.cancel_goal_async()
        if pickup_task_goal_handle is not None:
            pickup_task_goal_handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    def _close_direct_sdk(self) -> None:
        if self.direct_sdk_adapter is not None:
            self.direct_sdk_adapter.close()

    def _release_goal(self) -> None:
        with self.state_lock:
            self.mission_reserved = False
            self.active_mission = ""
            self.active_arm_joints_goal_handle = None
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
            node._close_direct_sdk()
            if rclpy.ok():
                executor.shutdown()
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
