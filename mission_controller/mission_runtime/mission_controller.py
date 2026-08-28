import math
import threading
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from mission_interfaces.action import (
    ExecuteAdaptiveBoxGrasp,
    ExecuteBoxGrasp,
    ExecuteBoxPlace,
    ExecuteDragBoxGrasp,
    PlaceBoxTest,
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
        # ArmSlaveData.pose is the Link8 pose expressed in the configured
        # arm-base frame.  Keep it separately from joint feedback because the
        # wrist-mounted camera extrinsic is composed from this live pose.
        self.latest_slave_arm_poses = {"left": None, "right": None}
        self.latest_slave_arm_pose_times = {"left": 0.0, "right": 0.0}
        self.latest_slave_arm_pose_sequences = {"left": 0, "right": 0}
        self.latest_slave_arm_pose_frames = {"left": "", "right": ""}
        self._last_grasp_box_tf_box_pose = None
        self._last_grasp_box_tf_box_to_link7_targets = None
        self._last_tf_body_home_carry_completed = False
        self._last_tf_body_home_carry_arm_targets = None
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
        # TF-based GraspBox uses the same public action schema as the legacy
        # endpoint, but has an isolated callback and target-conversion path.
        self.grasp_box_tf_action_server = ActionServer(
            self,
            ExecuteBoxGrasp,
            self._string("grasp_box_tf_action_name"),
            execute_callback=self._execute_grasp_box_tf,
            goal_callback=self._grasp_box_tf_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.drag_box_grasp_action_server = ActionServer(
            self,
            ExecuteDragBoxGrasp,
            self._string("execute_drag_box_grasp_action_name"),
            execute_callback=self._execute_drag_box_grasp,
            goal_callback=self._drag_box_grasp_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.drag_box_grasp_tf_action_server = ActionServer(
            self,
            ExecuteDragBoxGrasp,
            self._string("execute_drag_box_grasp_tf_action_name"),
            execute_callback=self._execute_drag_box_grasp_tf,
            goal_callback=self._drag_box_grasp_tf_goal_callback,
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
        self.place_box_test_action_server = ActionServer(
            self,
            PlaceBoxTest,
            self._string("place_box_test_action_name"),
            execute_callback=self._execute_place_box_test,
            goal_callback=self._place_box_test_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.server_group,
        )
        self.get_logger().info(
            "mission controller ready: "
            f"adaptive_box_grasp="
            f"{self._string('execute_adaptive_box_grasp_action_name')} "
            f"box_grasp={self._string('execute_box_grasp_action_name')} "
            f"grasp_box_tf={self._string('grasp_box_tf_action_name')} "
            f"drag_box_grasp={self._string('execute_drag_box_grasp_action_name')} "
            f"drag_box_grasp_tf="
            f"{self._string('execute_drag_box_grasp_tf_action_name')} "
            f"box_place={self._string('execute_box_place_action_name')} "
            f"place_box_test={self._string('place_box_test_action_name')}"
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
                ("grasp_box_tf_action_name", "/grasp_box_tf"),
                (
                    "execute_drag_box_grasp_action_name",
                    "/execute_drag_box_grasp",
                ),
                (
                    "execute_drag_box_grasp_tf_action_name",
                    "/execute_drag_box_grasp_tf",
                ),
                ("execute_box_place_action_name", "/execute_box_place"),
                ("place_box_test_action_name", "/place_box_test"),
                # Small-box placement test taught at body joint1=-15 deg.
                # The action requires a preceding /grasp_box_tf goal in the
                # same controller process so the rigid box->Link7 transforms
                # remain available after mobile-base transport.
                ("place_box_test_enabled", True),
                ("place_box_test_box_type", "smallbox"),
                ("place_box_test_start_body_joint_units", [0, 0, 0, 0]),
                ("place_box_test_body_joint_units", [-15000, 0, 0, 0]),
                (
                    "place_box_test_left_target_pose_arm_base",
                    [
                        -0.446381,
                        0.569104,
                        -0.128351,
                        -0.395,
                        -0.461,
                        -0.577,
                        0.545,
                    ],
                ),
                (
                    "place_box_test_right_target_pose_arm_base",
                    [
                        -0.260070,
                        -0.611426,
                        -0.094985,
                        0.463,
                        -0.444,
                        0.566,
                        0.516,
                    ],
                ),
                ("place_box_test_segments", 6),
                ("place_box_test_body_velocity", 2),
                ("place_box_test_body_blend_radius", 5),
                ("place_box_test_arm_blend_radius", 5),
                ("place_box_test_left_movel_velocity_percent", 3.0),
                ("place_box_test_right_movel_velocity_percent", 3.0),
                ("place_box_test_final_correction_enabled", True),
                ("place_box_test_final_correction_velocity_percent", 3.0),
                ("place_box_test_timeout_sec", 180.0),
                ("place_box_test_start_body_tolerance_rad", 0.035),
                ("place_box_test_position_tolerance_m", 0.015),
                ("place_box_test_orientation_tolerance_rad", 0.10),
                ("place_box_test_stable_samples", 3),
                (
                    "place_box_test_target_consistency_position_tolerance_m",
                    0.05,
                ),
                (
                    "place_box_test_target_consistency_orientation_tolerance_rad",
                    0.35,
                ),
                ("place_box_test_body_stop_enabled", True),
                ("place_box_test_body_stop_command", "stop"),
                ("adaptive_box_action_enabled", True),
                ("adaptive_freeze_frame", "base_link"),
                ("adaptive_require_detection_timestamp", True),
                ("adaptive_tf_cache_time_sec", 180.0),
                ("adaptive_detection_tf_timeout_sec", 5.0),
                ("adaptive_runtime_tf_timeout_sec", 2.0),
                # TF GraspBox freezes the detected box in the chassis-fixed
                # frame, then re-expresses the target in each live arm base
                # after the waist has reached its layer pose.
                ("grasp_box_tf_freeze_frame", "base_link"),
                ("grasp_box_tf_detection_tf_timeout_sec", 5.0),
                ("grasp_box_tf_runtime_tf_timeout_sec", 5.0),
                ("grasp_box_tf_require_detection_timestamp", True),
                ("grasp_box_tf_detection_arm", "right"),
                ("drag_box_tf_detection_arm", "left"),
                # After a left-camera DragBox TF detection, move that arm to
                # its configured safe/standby joint pose before pickup
                # planning.  The target is independent for model and layer.
                ("drag_box_tf_post_detection_left_movej_enabled", True),
                # Optional post-Step2 carry controller for /grasp_box_tf.
                # The waist uses MoveJ while both arms receive synchronized,
                # segmented SDK MoveL endpoints.  Box translation follows the
                # common chest frame, while its base_link orientation and both
                # box->controller-TCP transforms remain fixed.
                ("grasp_box_tf_body_home_carry_enabled", False),
                ("grasp_box_tf_body_home_carry_carrier_frame", "chest_Link"),
                ("grasp_box_tf_body_home_carry_joint_units", [0, 0, 0, 0]),
                ("grasp_box_tf_body_home_carry_segments", 6),
                ("grasp_box_tf_body_home_carry_continuous_enabled", False),
                ("grasp_box_tf_body_home_carry_body_velocity", 2),
                ("grasp_box_tf_body_home_carry_body_blend_radius", 0),
                ("grasp_box_tf_body_home_carry_arm_blend_radius", 5),
                (
                    "grasp_box_tf_body_home_carry_left_movel_velocity_percent",
                    3.0,
                ),
                (
                    "grasp_box_tf_body_home_carry_right_movel_velocity_percent",
                    3.0,
                ),
                ("grasp_box_tf_body_home_carry_timeout_sec", 180.0),
                ("grasp_box_tf_body_home_carry_tf_timeout_sec", 5.0),
                ("grasp_box_tf_body_home_carry_position_tolerance_m", 0.01),
                (
                    "grasp_box_tf_body_home_carry_orientation_tolerance_rad",
                    0.0872665,
                ),
                ("grasp_box_tf_body_home_carry_stable_samples", 3),
                ("grasp_box_tf_body_home_carry_final_correction_enabled", True),
                (
                    "grasp_box_tf_body_home_carry_final_correction_velocity_percent",
                    3.0,
                ),
                ("grasp_box_tf_body_home_carry_body_stop_enabled", True),
                ("grasp_box_tf_body_home_carry_body_stop_command", "stop"),
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
                (
                    "box_object_pose_action_name",
                    "/object_pose/estimate",
                ),
                ("box_object_pose_camera_side", "right"),
                ("box_object_pose_topic", "/mission/box_object_pose"),
                ("box_object_pose_camera_topic", "/object_pose/pose"),
                ("box_object_pose_raw_topic", "/mission/box_object_pose_raw"),
                # Temporary smallbox profile: reuse the current bigbox
                # geometric calibration values until smallbox is calibrated.
                ("box_object_pose_model_label", "smallbox"),
                ("box_object_pose_instance_index", 0),
                ("box_object_pose_confidence_threshold", 0.25),
                ("box_object_pose_result_timeout_sec", 120.0),
                # Hold the confirmed detection posture before and after each
                # FoundationPose request so RGB-D frames and robot TF settle.
                ("box_foundation_pose_pre_settle_sec", 5.0),
                ("box_foundation_pose_post_settle_sec", 5.0),
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
                ("box_grasp_execution_mode", "arms_only"),
                ("box_joint1_command_service_name", "/robot/command"),
                ("box_joint1_feedback_topic", "/mcap/body"),
                ("box_joint1_name", "joint1"),
                ("box_joint2_name", "joint2"),
                ("box_joint3_name", "joint3"),
                ("box_joint4_name", "joint4"),
                ("box_joint1_device", 2),
                ("box_joint1_detection_angle_deg", 0.0),
                ("box_joint1_approach_angle_deg", -13.0),
                (
                    "box_layer_joint1_approach_angles_deg",
                    [-13.0, -45.0, -70.0, -89.0],
                ),
                (
                    "box_layer_joint2_approach_angles_deg",
                    [0.0, -85.0, -120.0, -149.0],
                ),
                (
                    "box_layer_joint3_approach_angles_deg",
                    [0.0, -55.0, -73.0, -89.0],
                ),
                (
                    "box_layer_joint1_approach_angles_deg_bigbox",
                    [-13.0, -45.0, -70.0, -89.0],
                ),
                (
                    "box_layer_joint2_approach_angles_deg_bigbox",
                    [0.0, -85.0, -120.0, -149.0],
                ),
                (
                    "box_layer_joint3_approach_angles_deg_bigbox",
                    [0.0, -55.0, -73.0, -89.0],
                ),
                (
                    "box_layer_joint1_approach_angles_deg_smallbox",
                    [-13.0, -45.0, -70.0, -89.0],
                ),
                (
                    "box_layer_joint2_approach_angles_deg_smallbox",
                    [0.0, -85.0, -120.0, -149.0],
                ),
                (
                    "box_layer_joint3_approach_angles_deg_smallbox",
                    [0.0, -70.0, -73.0, -89.0],
                ),
                (
                    "box_layer_joint123_configured",
                    [True, True, True, True],
                ),
                ("box_joint2_detection_angle_deg", 0.0),
                ("box_joint2_approach_angle_deg", 0.0),
                ("box_joint3_detection_angle_deg", 0.0),
                ("box_joint3_approach_angle_deg", 0.0),
                ("box_joint1_command_units_per_degree", 1000.0),
                ("box_body_command_units_per_degree", [1000.0] * 4),
                ("box_joint1_velocity", 5),
                ("box_body_movej_velocity", 5),
                ("box_joint1_blend_radius", 0),
                ("box_joint1_axis_xyz", [0.0, 0.0, 1.0]),
                ("box_joint1_feedback_to_geometric_sign", 1.0),
                ("box_joint2_axis_xyz", [0.0, 0.0, -1.0]),
                ("box_joint3_axis_xyz", [0.0, 0.0, 1.0]),
                ("box_joint2_feedback_to_urdf_axis_sign", 1.0),
                ("box_joint3_feedback_to_urdf_axis_sign", 1.0),
                ("box_waist1_origin_xyz", [-0.080814986, 0.135049308, 0.266]),
                ("box_waist1_origin_rpy", [math.pi, math.pi / 2.0, 0.0]),
                ("box_waist2_origin_xyz", [-0.384, 0.0, -0.0074]),
                ("box_waist2_origin_rpy", [0.0, 0.0, 0.0]),
                ("box_waist3_origin_xyz", [-0.277703995, 0.0, 0.0024]),
                ("box_waist3_origin_rpy", [0.0, 0.0, 0.0]),
                ("box_waist3_to_chest_xyz", [-0.123796005, 0.0, -0.0755]),
                ("box_waist3_to_chest_rpy", [0.0, math.pi / 2.0, 0.0]),
                ("box_chest_to_left_arm_base_xyz", [0.012, 0.0, -0.2975]),
                ("box_chest_to_left_arm_base_rpy", [0.0, math.pi, 0.0]),
                ("box_chest_to_right_arm_base_xyz", [-0.012, 0.0, -0.2975]),
                ("box_chest_to_right_arm_base_rpy", [math.pi, 0.0, 0.0]),
                ("box_joint1_position_tolerance_rad", 0.01),
                ("box_joint1_velocity_tolerance_rad_sec", 0.01),
                ("box_joint1_feedback_max_age_sec", 1.0),
                ("box_joint1_wait_timeout_sec", 80.0),
                ("box_joint1_stable_samples", 3),
                (
                    "direct_movel_target_mode",
                    "camera_offset_box_orientation",
                ),
                ("direct_movel_box_relative_model_label", "smallbox"),
                ("direct_movel_motion_mode", "movej_p"),
                ("direct_movel_velocity_percent", 10.0),
                ("direct_movel_blocking", True),
                ("box_post_movel_enabled", False),
                ("box_post_movel_velocity_percent", 10.0),
                ("drag_box_post_movel_enabled", True),
                # DragBox moves the right arm through Drag3 first, then joins
                # the left arm at its cumulative target before Step2.
                ("drag_box_left_arm_enabled", True),
                ("drag_box_left_join_mode", "after_drag3"),
                ("drag_box_left_join_motion_mode", "movej_p"),
                ("drag_box_left_join_velocity_percent", 10.0),
                ("drag_box_left_join_timeout_sec", 60.0),
                # Before the delayed left-arm MoveJ_P join, move the left arm
                # to a configured posture. Values are RealMan command units
                # (1000 units = 1 degree).
                ("drag_box_left_join_pre_movej_enabled", True),
                (
                    "drag_box_left_join_pre_movej_joint_units",
                    [-22817, 92009, -98469, -100366, -81197, 5123, 9078],
                ),
                ("drag_box_post_movel_step_drag1_left_xyz", [0.0, 0.0, 0.0]),
                ("drag_box_post_movel_step_drag1_right_xyz", [0.14, 0.0, 0.0]),
                ("drag_box_post_movel_step_drag2_left_xyz", [0.0, 0.0, 0.10]),
                ("drag_box_post_movel_step_drag2_right_xyz", [0.0, 0.0, 0.10]),
                ("drag_box_post_movel_step_drag3_left_xyz", [0.0, 0.0, 0.0]),
                ("drag_box_post_movel_step_drag3_right_xyz", [-0.14, 0.0, 0.0]),
                ("box_post_movel_step4_motion_mode", "movej"),
                ("box_post_movel_step4_movej_joint2_units", 40000),
                ("box_post_movel_step4_movej_left_device", 0),
                ("box_post_movel_step4_movej_right_device", 1),
                (
                    "box_post_movel_step4_movej_left_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                (
                    "box_post_movel_step4_movej_right_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                ("box_post_movel_step4_movej_command_units_per_degree", 1000.0),
                ("box_post_movel_step4_movej_velocity", 10),
                ("box_post_movel_step4_movej_blend_radius", 0),
                ("box_post_movel_step4_movej_trajectory_connect", 0),
                ("box_post_movel_step4_movej_timeout_sec", 40.0),
                ("box_post_movel_step4_movej_position_tolerance_rad", 0.01),
                ("box_post_movel_step4_movej_velocity_tolerance_rad_sec", 0.01),
                ("box_post_movel_step4_movej_feedback_max_age_sec", 1.0),
                ("box_post_movel_step4_movej_stable_samples", 3),
                ("box_post_movel_step_count", 4),
                ("box_post_movel_left_step1_xyz", [0.0, 0.0, 0.025]),
                ("box_post_movel_right_step1_xyz", [0.0, 0.0, -0.028]),
                # Smallbox-specific Step1 deltas. Bigbox and callers that do
                # not select a model continue using the generic values.
                ("box_post_movel_left_step1_xyz_smallbox", [0.0, 0.0, -0.035]),
                ("box_post_movel_right_step1_xyz_smallbox", [0.0, 0.0, 0.02]),
                ("box_post_movel_left_step2_xyz", [0.14, 0.0, 0.0]),
                ("box_post_movel_right_step2_xyz", [0.14, 0.0, 0.0]),
                ("box_post_movel_left_step3_xyz", [-0.14, 0.0, 0.0]),
                ("box_post_movel_right_step3_xyz", [-0.14, 0.0, 0.0]),
                ("box_post_movel_left_step4_xyz", [0.0, 0.0, -0.1]),
                ("box_post_movel_right_step4_xyz", [0.0, 0.0, 0.1]),
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
                ("box_post_arm_movej_velocity", 10),
                ("box_post_arm_movej_blend_radius", 0),
                ("box_post_arm_movej_trajectory_connect", 0),
                ("box_post_arm_movej_timeout_sec", 40.0),
                # Place the wrist-mounted right camera at a known joint pose
                # before FoundationPose detection.
                ("box_pre_detection_arm_movej_enabled", True),
                ("box_pre_detection_right_movej_enabled", True),
                ("box_pre_detection_right_movej_device", 1),
                (
                    "box_pre_detection_right_movej_joint_units",
                    [144725, -5335, 7032, 9843, 7540, -5611, 85414],
                ),
                (
                    "box_layer_pre_detection_right_movej_joint_units",
                    [
                        144725, -5335, 7032, 9843, 7540, -5611, 85414,
                        -12083, 5105, -17961, -50575, 9150, -5641, -66298,
                        23227, 13389, -62736, -51630, 44662, -12958, -22269,
                        21382, -4978, -274, 1282, -5472, 3628, -32636,
                    ],
                ),
                # Per-model detection poses.  The generic flattened table is
                # retained as a backward-compatible fallback; GraspBox
                # selects the table for its configured model label.
                (
                    "box_layer_pre_detection_right_movej_joint_units_bigbox",
                    [
                        144725, -5335, 7032, 9843, 7540, -5611, 85414,
                        -12083, 5105, -17961, -50575, 9150, -5641, -66298,
                        23227, 13389, -62736, -51630, 44662, -12958, -22269,
                        21382, -4978, -274, 1282, -5472, 3628, -32636,
                    ],
                ),
                (
                    "box_layer_pre_detection_right_movej_joint_units_smallbox",
                    [
                        144725, -5335, 7032, 9843, 7540, -5611, 85414,
                        -12083, 5105, -17961, -50575, 9150, -5641, -66298,
                        23227, 13389, -62736, -51630, 44662, -12958, -22269,
                        1538, 252, 146, -25260, -8872, 92, -23259,
                    ],
                ),
                (
                    "box_layer_pre_detection_right_movej_configured",
                    [True, True, True, True],
                ),
                ("box_pre_detection_right_movej_command_units_per_degree", 1000.0),
                ("box_pre_detection_right_movej_velocity", 10),
                ("box_pre_detection_right_movej_blend_radius", 0),
                ("box_pre_detection_right_movej_trajectory_connect", 0),
                ("box_pre_detection_right_movej_timeout_sec", 40.0),
                ("box_pre_detection_right_movej_position_tolerance_rad", 0.01),
                ("box_pre_detection_right_movej_velocity_tolerance_rad_sec", 0.01),
                ("box_pre_detection_right_movej_feedback_max_age_sec", 1.0),
                ("box_pre_detection_right_movej_stable_samples", 3),
                # Left-camera counterpart. Defaults mirror the current right
                # table until each left observation pose is calibrated.
                ("box_pre_detection_left_movej_enabled", True),
                ("box_pre_detection_left_movej_device", 0),
                (
                    "box_pre_detection_left_movej_joint_units",
                    [144725, -5335, 7032, 9843, 7540, -5611, 85414],
                ),
                (
                    "box_layer_pre_detection_left_movej_joint_units",
                    [
                        144725, -5335, 7032, 9843, 7540, -5611, 85414,
                        -12083, 5105, -17961, -50575, 9150, -5641, -66298,
                        23227, 13389, -62736, -51630, 44662, -12958, -22269,
                        21382, -4978, -274, 1282, -5472, 3628, -32636,
                    ],
                ),
                (
                    "box_layer_pre_detection_left_movej_joint_units_bigbox",
                    [
                        144725, -5335, 7032, 9843, 7540, -5611, 85414,
                        -12083, 5105, -17961, -50575, 9150, -5641, -66298,
                        23227, 13389, -62736, -51630, 44662, -12958, -22269,
                        21382, -4978, -274, 1282, -5472, 3628, -32636,
                    ],
                ),
                (
                    "box_layer_pre_detection_left_movej_joint_units_smallbox",
                    [
                        144725, -5335, 7032, 9843, 7540, -5611, 85414,
                        -12083, 5105, -17961, -50575, 9150, -5641, -66298,
                        23227, 13389, -62736, -51630, 44662, -12958, -22269,
                        1538, 252, 146, -25260, -8872, 92, -23259,
                    ],
                ),
                (
                    "box_layer_pre_detection_left_movej_configured",
                    [True, True, True, True],
                ),
                ("box_pre_detection_left_movej_command_units_per_degree", 1000.0),
                ("box_pre_detection_left_movej_velocity", 10),
                ("box_pre_detection_left_movej_blend_radius", 0),
                ("box_pre_detection_left_movej_trajectory_connect", 0),
                ("box_pre_detection_left_movej_timeout_sec", 40.0),
                ("box_pre_detection_left_movej_position_tolerance_rad", 0.01),
                ("box_pre_detection_left_movej_velocity_tolerance_rad_sec", 0.01),
                ("box_pre_detection_left_movej_feedback_max_age_sec", 1.0),
                ("box_pre_detection_left_movej_stable_samples", 3),
                # Move both arms to this intermediate pose after detection,
                # then send the computed Link8 movej_p targets.
                ("box_pre_target_arm_movej_enabled", True),
                ("box_pre_target_arm_movej_two_stage_enabled", True),
                ("box_pre_target_arm_movej_stage1_joint2_units", 40000),
                ("box_preparation_movej_velocity", 10),
                ("box_pre_target_arm_movej_left_device", 0),
                ("box_pre_target_arm_movej_right_device", 1),
                (
                    "box_pre_target_arm_movej_left_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                (
                    "box_pre_target_arm_movej_right_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                ("box_pre_target_arm_movej_command_units_per_degree", 1000.0),
                ("box_pre_target_arm_movej_velocity", 10),
                ("box_pre_target_arm_movej_blend_radius", 0),
                ("box_pre_target_arm_movej_trajectory_connect", 0),
                ("box_pre_target_arm_movej_timeout_sec", 40.0),
                ("box_pre_target_arm_movej_position_tolerance_rad", 0.01),
                ("box_pre_target_arm_movej_velocity_tolerance_rad_sec", 0.01),
                ("box_pre_target_arm_movej_feedback_max_age_sec", 1.0),
                ("box_pre_target_arm_movej_stable_samples", 3),
                ("box_post_arm_left_feedback_topic", "/mcap/slave_arm_left"),
                ("box_post_arm_right_feedback_topic", "/mcap/slave_arm_right"),
                ("box_post_arm_position_tolerance_rad", 0.01),
                ("box_post_arm_velocity_tolerance_rad_sec", 0.01),
                ("box_post_arm_feedback_max_age_sec", 1.0),
                ("box_post_arm_stable_samples", 3),
                ("box_body_return_home_enabled", True),
                ("box_body_home_joint_units", [0, 0, 0, 0]),
                ("box_body_home_velocity", 5),
                ("box_body_home_blend_radius", 0),
                ("box_body_home_timeout_sec", 40.0),
                # After post-grasp Step2, optionally send one body MoveJ to
                # home while both arms send one-shot MoveL endpoint targets.
                # The captured Link7 EEPose is preserved in the fixed root
                # frame; the reverse endpoint then restores the measured
                # Step2 state before Step3/Step4 continue.
                ("box_step2_waist_endpoint_sync_enabled", False),
                ("box_step2_waist_endpoint_sync_home_joint_units", [0, 0, 0, 0]),
                ("box_step2_waist_endpoint_sync_body_blend_radius", 0),
                ("box_step2_waist_endpoint_sync_timeout_sec", 180.0),
                ("box_step2_waist_endpoint_sync_feedback_max_age_sec", 0.5),
                ("box_step2_waist_endpoint_sync_final_position_tolerance_m", 0.01),
                ("box_step2_waist_endpoint_sync_final_orientation_tolerance_rad", 0.0872665),
                ("box_step2_waist_endpoint_sync_stable_samples", 3),
                ("box_step2_waist_endpoint_sync_body_stop_enabled", True),
                ("box_step2_waist_endpoint_sync_body_stop_command", "stop"),
                ("box_step2_waist_endpoint_sync_skip_final_body_home", True),
                ("box_step2_waist_endpoint_sync_layer1_configured", False),
                ("box_step2_waist_endpoint_sync_layer1_segments", 1),
                ("box_step2_waist_endpoint_sync_layer1_forward_body_velocity", 5),
                ("box_step2_waist_endpoint_sync_layer1_forward_left_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer1_forward_right_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer1_reverse_body_velocity", 5),
                ("box_step2_waist_endpoint_sync_layer1_reverse_left_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer1_reverse_right_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer2_configured", False),
                ("box_step2_waist_endpoint_sync_layer2_segments", 1),
                ("box_step2_waist_endpoint_sync_layer2_forward_body_velocity", 5),
                ("box_step2_waist_endpoint_sync_layer2_forward_left_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer2_forward_right_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer2_reverse_body_velocity", 5),
                ("box_step2_waist_endpoint_sync_layer2_reverse_left_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer2_reverse_right_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer3_configured", False),
                ("box_step2_waist_endpoint_sync_layer3_segments", 2),
                ("box_step2_waist_endpoint_sync_layer3_forward_body_velocity", 5),
                ("box_step2_waist_endpoint_sync_layer3_forward_left_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer3_forward_right_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer3_reverse_body_velocity", 5),
                ("box_step2_waist_endpoint_sync_layer3_reverse_left_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer3_reverse_right_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer4_configured", False),
                ("box_step2_waist_endpoint_sync_layer4_segments", 1),
                ("box_step2_waist_endpoint_sync_layer4_forward_body_velocity", 5),
                ("box_step2_waist_endpoint_sync_layer4_forward_left_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer4_forward_right_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer4_reverse_body_velocity", 5),
                ("box_step2_waist_endpoint_sync_layer4_reverse_left_movel_velocity_percent", 10.0),
                ("box_step2_waist_endpoint_sync_layer4_reverse_right_movel_velocity_percent", 10.0),
                ("direct_movel_use_current_fixture_orientation", False),
                (
                    "direct_movel_left_fixed_link8_orientation",
                    [-0.497, -0.503, -0.488, 0.509],
                ),
                (
                    "direct_movel_right_fixed_link8_orientation",
                    [0.482, -0.463, 0.522, 0.528],
                ),
                # During the current Realbots2 calibration, direct targets
                # are Link8 EEPose targets.  Fixture-center compensation is
                # opt-in and remains disabled until the fixture transform is
                # independently verified.
                ("direct_movel_fixture_compensation_enabled", False),
                ("direct_movel_left_offset_xyz", [0.0, 0.0, -0.51]),
                ("direct_movel_right_offset_xyz", [0.0, 0.0, 0.45]),
                (
                    "direct_movel_left_offset_xyz_bigbox_layer1",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_bigbox_layer1",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_bigbox_layer2",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_bigbox_layer2",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_bigbox_layer3",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_bigbox_layer3",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_bigbox_layer4",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_bigbox_layer4",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_smallbox_layer1",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_smallbox_layer1",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_smallbox_layer2",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_smallbox_layer2",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_smallbox_layer3",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_smallbox_layer3",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_smallbox_layer4",
                    [0.0, -0.025, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_smallbox_layer4",
                    [0.0, -0.025, 0.50],
                ),
                (
                    "direct_movel_left_box_to_link8_orientation",
                    [-0.666064, -0.026103, 0.005935, 0.745414],
                ),
                (
                    "direct_movel_right_box_to_link8_orientation",
                    [-0.694193, -0.007865, 0.013341, 0.719622],
                ),
                (
                    "joint123_layer1_left_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer1_right_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer2_left_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer2_right_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer3_left_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer3_right_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer4_left_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer4_right_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "left_fixture_center_in_link8_xyz",
                    [-0.12, -0.08, 0.05],
                ),
                (
                    "right_fixture_center_in_link8_xyz",
                    [-0.12, 0.08, 0.05],
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
                ("left_ee_frame", "left_arm_8_Link"),
                ("right_ee_frame", "right_arm_8_Link"),
                ("left_gripper_frame", "left_arm_8_Link"),
                ("right_gripper_frame", "right_arm_8_Link"),
                ("camera_mount_tf_enabled", False),
                ("camera_mount_parent_frame", "right_arm_8_Link"),
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
                # Wrist-camera profile: the camera is rigidly mounted to
                # Link8, so Base->camera is composed from live EEPose and the
                # fixed URDF Link8->RGB optical-center transform.
                ("camera_dynamic_link8_extrinsics_enabled", True),
                ("camera_detection_arm", "right"),
                ("camera_eepose_max_age_sec", 1.0),
                # Fixed transform T_left_right for the new robot.  It maps
                # right-arm-base coordinates into left-arm-base coordinates.
                # The left Base origin is [0.024, 0, 0] in the right Base;
                # left x/y axes are the negatives of right x/y, and z is
                # shared. The inverse is used for left-camera detection.
                ("camera_fixed_cross_arm_transform_enabled", True),
                ("camera_right_base_to_left_base_xyz", [0.024, 0.0, 0.0]),
                (
                    "camera_right_base_to_left_base_quaternion_xyzw",
                    [0.0, 0.0, 1.0, 0.0],
                ),
                ("left_arm_base_frame", "L_base_Link"),
                ("right_arm_base_frame", "R_base_Link"),
                ("left_link8_frame", "left_arm_8_Link"),
                ("right_link8_frame", "right_arm_8_Link"),
                (
                    "camera_left_link8_to_rgb_camera_xyz",
                    [0.097294396234, 0.000243365421, 0.053076686984],
                ),
                (
                    "camera_left_link8_to_rgb_camera_quaternion_xyzw",
                    [
                        0.153045932190,
                        -0.153045932190,
                        -0.690345524096,
                        0.690345524097,
                    ],
                ),
                (
                    "camera_right_link8_to_rgb_camera_xyz",
                    [0.096527, -0.016012, 0.046146],
                ),
                (
                    "camera_right_link8_to_rgb_camera_quaternion_xyzw",
                    [
                        -0.152364,
                        -0.111601,
                        0.682613,
                        0.705953,
                    ],
                ),
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

        # TF GraspBox and TF DragBox have independent per-model/per-layer
        # profiles.  Keep the legacy parameters above for the original
        # actions, while these generated declarations provide a complete,
        # explicit tuning surface for /grasp_box_tf and
        # /execute_drag_box_grasp_tf.
        self.declare_parameters(
            namespace="",
            parameters=self._tf_layer_parameter_defaults(),
        )

    @staticmethod
    def _tf_layer_parameter_defaults():
        """Return independent TF-action/model/layer defaults.

        Values intentionally mirror the currently deployed bigbox/smallbox
        profiles.  Each layer receives its own scalar/vector parameters so a
        later calibration change cannot affect another layer or the other TF
        action.
        """
        detection = {
            "bigbox": [
                [144725, -5335, 7032, 9843, 7540, -5611, 85414],
                [170647, 3018, 18744, 95121, -1950, -8903, 30524],
                [-39570, 16276, -17721, -95245, 14032, -14558, -23838],
                [21382, -4978, -274, 1282, -5472, 3628, -32636],
            ],
            "smallbox": [
                [172102, -3751, 14348, 95105, 7730, -5615, 31841],
                [-20762, 1539, -12837, -102889, 1366, -8774, 3529],
                [6894, -3485, -20092, -19433, 15367, -7336, -41454],
                [2341, 248, -3624, -16956, 2290, 10183, -45681],
            ],
        }
        # DragBox TF uses the left camera for detection.  Keep its calibrated
        # bigbox poses independent from the GraspBox/right-camera profiles.
        # The values are controller joint units (1000 units = 1 degree).
        drag_left_detection = {
            "bigbox": [
                [-172278, 7319, 20124, 51808, -17856, -23263, -70442],
                [-161722, 5480, 933, 104186, 5631, -2342, -2903],
                [18442, 3544, 2870, -104500, -2568, 5250, -24076],
                [-19238, 3482, 215, -91196, -2634, 5268, -78519],
            ],
            # Until separately calibrated, retain the existing left/smallbox
            # defaults rather than coupling them to the bigbox calibration.
            "smallbox": detection["smallbox"],
        }
        post_detection_left = {
            model: [
                [-171982, -204, 93820, 89651, 4401, 5999, -4935]
                for _layer in range(1, 5)
            ]
            for model in ("bigbox", "smallbox")
        }
        angles = {
            "bigbox": {
                1: (-13.0, 0.0, 0.0),
                2: (-45.0, -85.0, -55.0),
                3: (-70.0, -120.0, -73.0),
                4: (-89.0, -149.0, -89.0),
            },
            "smallbox": {
                1: (-13.0, 0.0, 0.0),
                2: (-45.0, -85.0, -70.0),
                3: (-70.0, -120.0, -73.0),
                4: (-89.0, -149.0, -89.0),
            },
        }
        offsets = {
            "bigbox": {
                layer: ([0.0, 0.0, -0.5], [0.0, 0.0, 0.5])
                for layer in range(1, 5)
            },
            "smallbox": {
                layer: (
                    [0.0, -0.025, -0.5],
                    [0.0, -0.025, 0.5],
                )
                if layer == 4
                else ([0.0, 0.0, -0.5], [0.0, 0.0, 0.5])
                for layer in range(1, 5)
            },
        }
        left_correction = [
            0.064762,
            -0.049358,
            0.060595,
            -0.058164,
            -0.006476,
            0.081596,
            0.994946,
        ]
        right_correction = [
            0.081444,
            -0.049338,
            -0.020083,
            0.012614,
            -0.032172,
            0.081927,
            0.996039,
        ]
        standard_steps = {
            "left": {
                1: [0.0, 0.0, 0.025],
                2: [0.14, 0.0, 0.0],
                3: [-0.14, 0.0, 0.0],
                4: [0.0, 0.0, -0.1],
                5: [0.0, 0.0, 0.0],
            },
            "right": {
                1: [0.0, 0.0, -0.028],
                2: [0.14, 0.0, 0.0],
                3: [-0.14, 0.0, 0.0],
                4: [0.0, 0.0, 0.1],
                5: [0.0, 0.0, 0.0],
            },
        }
        smallbox_step1 = {
            "left": [0.0, 0.0, 0.03],
            "right": [0.0, 0.0, -0.02],
        }
        drag_steps = {
            "left": {
                1: [0.0, 0.0, 0.0],
                2: [0.0, 0.0, 0.2],
                3: [0.0, 0.0, 0.0],
            },
            "right": {
                1: [0.14, 0.0, 0.0],
                2: [0.0, 0.0, 0.2],
                3: [-0.14, 0.0, 0.0],
            },
        }
        parameters = []
        for action_prefix in ("grasp_box_tf", "drag_box_tf"):
            for model in ("bigbox", "smallbox"):
                for layer in range(1, 5):
                    for arm in ("left", "right"):
                        profile = (
                            drag_left_detection[model][layer - 1]
                            if action_prefix == "drag_box_tf" and arm == "left"
                            else detection[model][layer - 1]
                        )
                        parameters.append(
                            (
                                f"{action_prefix}_box_layer_pre_detection_{arm}_movej_joint_units_"
                                f"{model}_layer{layer}",
                                list(profile),
                            )
                        )
                    if action_prefix == "drag_box_tf":
                        parameters.append(
                            (
                                f"drag_box_tf_box_layer_post_detection_left_movej_joint_units_"
                                f"{model}_layer{layer}",
                                list(post_detection_left[model][layer - 1]),
                            )
                        )
                    for joint_index, angle in enumerate(
                        angles[model][layer], start=1
                    ):
                        parameters.append(
                            (
                                f"{action_prefix}_box_layer_joint{joint_index}_"
                                f"approach_angle_deg_{model}_layer{layer}",
                                float(angle),
                            )
                        )
                    parameters.extend(
                        [
                            (
                                f"{action_prefix}_direct_movel_left_offset_xyz_"
                                f"{model}_layer{layer}",
                                list(offsets[model][layer][0]),
                            ),
                            (
                                f"{action_prefix}_direct_movel_right_offset_xyz_"
                                f"{model}_layer{layer}",
                                list(offsets[model][layer][1]),
                            ),
                            (
                                f"{action_prefix}_joint123_left_target_correction_pose_box_"
                                f"{model}_layer{layer}",
                                list(left_correction),
                            ),
                            (
                                f"{action_prefix}_joint123_right_target_correction_pose_box_"
                                f"{model}_layer{layer}",
                                list(right_correction),
                            ),
                        ]
                    )
                    for arm in ("left", "right"):
                        for step in range(1, 6):
                            delta = standard_steps[arm][step]
                            if model == "smallbox" and step == 1:
                                delta = smallbox_step1[arm]
                            parameters.append(
                                (
                                    f"{action_prefix}_post_movel_{arm}_step{step}_xyz_"
                                    f"{model}_layer{layer}",
                                    list(delta),
                                )
                            )
                    if action_prefix == "drag_box_tf":
                        for arm in ("left", "right"):
                            for drag_index in range(1, 4):
                                parameters.append(
                                    (
                                        f"drag_box_tf_post_movel_step_drag{drag_index}_"
                                        f"{arm}_xyz_{model}_layer{layer}",
                                        list(drag_steps[arm][drag_index]),
                                    )
                                )
        return parameters

    def _validate_parameters(self) -> None:
        for name in (
            "execute_adaptive_box_grasp_action_name",
            "execute_box_grasp_action_name",
            "grasp_box_tf_action_name",
            "execute_drag_box_grasp_action_name",
            "execute_drag_box_grasp_tf_action_name",
            "execute_box_place_action_name",
            "adaptive_freeze_frame",
            "box_object_pose_action_name",
            "box_object_pose_camera_side",
            "grasp_box_tf_detection_arm",
            "drag_box_tf_detection_arm",
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
            "camera_detection_arm",
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
        step4_motion_mode = self._string(
            "box_post_movel_step4_motion_mode"
        ).lower()
        if step4_motion_mode not in ("movel", "movej_p", "movej"):
            raise ValueError(
                "parameter 'box_post_movel_step4_motion_mode' must be "
                "'movel', 'movej_p', or 'movej'"
            )

        left_join_mode = self._string("drag_box_left_join_mode").strip().lower()
        if left_join_mode not in ("immediate", "after_drag3"):
            raise ValueError(
                "parameter 'drag_box_left_join_mode' must be "
                "'immediate' or 'after_drag3'"
            )
        left_join_motion_mode = self._string(
            "drag_box_left_join_motion_mode"
        ).strip().lower()
        if left_join_motion_mode not in ("movel", "movej_p"):
            raise ValueError(
                "parameter 'drag_box_left_join_motion_mode' must be "
                "'movel' or 'movej_p'"
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
        if self._string("camera_detection_arm").strip().lower() not in (
            "left",
            "right",
        ):
            raise ValueError("camera_detection_arm must be 'left' or 'right'")
        for name in ("grasp_box_tf_detection_arm", "drag_box_tf_detection_arm"):
            if self._string(name).strip().lower() not in ("left", "right"):
                raise ValueError(f"{name} must be 'left' or 'right'")
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
            ("camera_left_link8_to_rgb_camera_xyz", 3),
            ("camera_right_link8_to_rgb_camera_xyz", 3),
            ("camera_left_link8_to_rgb_camera_quaternion_xyzw", 4),
            ("camera_right_link8_to_rgb_camera_quaternion_xyzw", 4),
            ("camera_right_base_to_left_base_xyz", 3),
            ("camera_right_base_to_left_base_quaternion_xyzw", 4),
            ("box_foundation_to_pickup_rpy", 3),
            ("direct_movel_left_offset_xyz", 3),
            ("direct_movel_right_offset_xyz", 3),
            ("direct_movel_left_offset_xyz_bigbox_layer1", 3),
            ("direct_movel_right_offset_xyz_bigbox_layer1", 3),
            ("direct_movel_left_offset_xyz_bigbox_layer2", 3),
            ("direct_movel_right_offset_xyz_bigbox_layer2", 3),
            ("direct_movel_left_offset_xyz_bigbox_layer3", 3),
            ("direct_movel_right_offset_xyz_bigbox_layer3", 3),
            ("direct_movel_left_offset_xyz_bigbox_layer4", 3),
            ("direct_movel_right_offset_xyz_bigbox_layer4", 3),
            ("direct_movel_left_offset_xyz_smallbox_layer1", 3),
            ("direct_movel_right_offset_xyz_smallbox_layer1", 3),
            ("direct_movel_left_offset_xyz_smallbox_layer2", 3),
            ("direct_movel_right_offset_xyz_smallbox_layer2", 3),
            ("direct_movel_left_offset_xyz_smallbox_layer3", 3),
            ("direct_movel_right_offset_xyz_smallbox_layer3", 3),
            ("direct_movel_left_offset_xyz_smallbox_layer4", 3),
            ("direct_movel_right_offset_xyz_smallbox_layer4", 3),
            ("box_post_movel_left_step1_xyz", 3),
            ("box_post_movel_right_step1_xyz", 3),
            ("box_post_movel_left_step1_xyz_smallbox", 3),
            ("box_post_movel_right_step1_xyz_smallbox", 3),
            ("box_post_movel_left_step2_xyz", 3),
            ("box_post_movel_right_step2_xyz", 3),
            ("box_post_movel_left_step3_xyz", 3),
            ("box_post_movel_right_step3_xyz", 3),
            ("box_post_movel_left_step4_xyz", 3),
            ("box_post_movel_right_step4_xyz", 3),
            ("box_post_movel_left_step5_xyz", 3),
            ("box_post_movel_right_step5_xyz", 3),
            ("drag_box_post_movel_step_drag1_left_xyz", 3),
            ("drag_box_post_movel_step_drag1_right_xyz", 3),
            ("drag_box_post_movel_step_drag2_left_xyz", 3),
            ("drag_box_post_movel_step_drag2_right_xyz", 3),
            ("drag_box_post_movel_step_drag3_left_xyz", 3),
            ("drag_box_post_movel_step_drag3_right_xyz", 3),
            ("box_post_movel_step4_movej_left_joint_units", 7),
            ("box_post_movel_step4_movej_right_joint_units", 7),
            ("direct_movel_left_box_to_link8_orientation", 4),
            ("direct_movel_right_box_to_link8_orientation", 4),
            ("joint123_layer1_left_target_correction_pose_box", 7),
            ("joint123_layer1_right_target_correction_pose_box", 7),
            ("joint123_layer2_left_target_correction_pose_box", 7),
            ("joint123_layer2_right_target_correction_pose_box", 7),
            ("joint123_layer3_left_target_correction_pose_box", 7),
            ("joint123_layer3_right_target_correction_pose_box", 7),
            ("joint123_layer4_left_target_correction_pose_box", 7),
            ("joint123_layer4_right_target_correction_pose_box", 7),
            ("direct_movel_left_fixed_link8_orientation", 4),
            ("direct_movel_right_fixed_link8_orientation", 4),
            ("left_fixture_center_in_link8_xyz", 3),
            ("right_fixture_center_in_link8_xyz", 3),
            ("box_joint1_axis_xyz", 3),
            ("box_joint2_axis_xyz", 3),
            ("box_joint3_axis_xyz", 3),
            ("box_waist1_origin_xyz", 3),
            ("box_waist1_origin_rpy", 3),
            ("box_waist2_origin_xyz", 3),
            ("box_waist2_origin_rpy", 3),
            ("box_waist3_origin_xyz", 3),
            ("box_waist3_origin_rpy", 3),
            ("box_waist3_to_chest_xyz", 3),
            ("box_waist3_to_chest_rpy", 3),
            ("box_chest_to_left_arm_base_xyz", 3),
            ("box_chest_to_left_arm_base_rpy", 3),
            ("box_chest_to_right_arm_base_xyz", 3),
            ("box_chest_to_right_arm_base_rpy", 3),
            ("box_body_command_units_per_degree", 4),
            ("box_layer_joint1_approach_angles_deg", 4),
            ("box_layer_joint2_approach_angles_deg", 4),
            ("box_layer_joint3_approach_angles_deg", 4),
            ("box_layer_joint1_approach_angles_deg_bigbox", 4),
            ("box_layer_joint2_approach_angles_deg_bigbox", 4),
            ("box_layer_joint3_approach_angles_deg_bigbox", 4),
            ("box_layer_joint1_approach_angles_deg_smallbox", 4),
            ("box_layer_joint2_approach_angles_deg_smallbox", 4),
            ("box_layer_joint3_approach_angles_deg_smallbox", 4),
            ("adaptive_grasp_span_axis_object", 3),
            ("adaptive_grasp_height_axis_object", 3),
            ("adaptive_grasp_correction_rpy", 3),
            ("adaptive_left_grasp_extra_rpy", 3),
            ("adaptive_right_grasp_extra_rpy", 3),
            ("box_post_arm_movej_left_joint_units", 7),
            ("box_post_arm_movej_right_joint_units", 7),
            ("box_pre_detection_right_movej_joint_units", 7),
            ("box_pre_detection_left_movej_joint_units", 7),
            ("box_layer_pre_detection_right_movej_joint_units", 28),
            ("box_layer_pre_detection_right_movej_joint_units_bigbox", 28),
            ("box_layer_pre_detection_right_movej_joint_units_smallbox", 28),
            ("box_layer_pre_detection_left_movej_joint_units", 28),
            ("box_layer_pre_detection_left_movej_joint_units_bigbox", 28),
            ("box_layer_pre_detection_left_movej_joint_units_smallbox", 28),
            ("box_pre_target_arm_movej_left_joint_units", 7),
            ("box_pre_target_arm_movej_right_joint_units", 7),
            ("drag_box_left_join_pre_movej_joint_units", 7),
            ("box_body_home_joint_units", 4),
            ("box_step2_waist_endpoint_sync_home_joint_units", 4),
            ("grasp_box_tf_body_home_carry_joint_units", 4),
            ("place_box_test_body_joint_units", 4),
            ("place_box_test_start_body_joint_units", 4),
            ("place_box_test_left_target_pose_arm_base", 7),
            ("place_box_test_right_target_pose_arm_base", 7),
        ):
            values = self._float_array(name)
            if len(values) != expected_length:
                raise ValueError(
                    f"parameter '{name}' must contain {expected_length} values"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"parameter '{name}' contains NaN or Inf")

        for name in (
            "joint123_layer1_left_target_correction_pose_box",
            "joint123_layer1_right_target_correction_pose_box",
            "joint123_layer2_left_target_correction_pose_box",
            "joint123_layer2_right_target_correction_pose_box",
            "joint123_layer3_left_target_correction_pose_box",
            "joint123_layer3_right_target_correction_pose_box",
            "joint123_layer4_left_target_correction_pose_box",
            "joint123_layer4_right_target_correction_pose_box",
            "place_box_test_left_target_pose_arm_base",
            "place_box_test_right_target_pose_arm_base",
        ):
            values = self._float_array(name)
            quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
            if quaternion_norm <= 1e-12:
                raise ValueError(
                    f"parameter '{name}' contains a zero quaternion"
                )

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
        for name in (
            "box_layer_joint1_approach_angles_deg",
            "box_layer_joint2_approach_angles_deg",
            "box_layer_joint3_approach_angles_deg",
            "box_layer_joint1_approach_angles_deg_bigbox",
            "box_layer_joint2_approach_angles_deg_bigbox",
            "box_layer_joint3_approach_angles_deg_bigbox",
            "box_layer_joint1_approach_angles_deg_smallbox",
            "box_layer_joint2_approach_angles_deg_smallbox",
            "box_layer_joint3_approach_angles_deg_smallbox",
        ):
            layer_angles = self._float_array(name)
            if not all(math.isfinite(value) for value in layer_angles):
                raise ValueError(
                    f"parameter '{name}' contains NaN or Inf"
                )
        layer_configured = self._boolean_array("box_layer_joint123_configured")
        if len(layer_configured) != 4:
            raise ValueError(
                "parameter 'box_layer_joint123_configured' must contain four values"
            )
        detection_configured = self._boolean_array(
            "box_layer_pre_detection_right_movej_configured"
        )
        if len(detection_configured) != 4:
            raise ValueError(
                "parameter 'box_layer_pre_detection_right_movej_configured' "
                "must contain four values"
            )
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
        for name in (
            "camera_left_link8_to_rgb_camera_quaternion_xyzw",
            "camera_right_link8_to_rgb_camera_quaternion_xyzw",
            "camera_right_base_to_left_base_quaternion_xyzw",
        ):
            if math.sqrt(
                sum(value * value for value in self._float_array(name))
            ) <= 1e-12:
                raise ValueError(f"parameter '{name}' has zero norm")
        if self._float("camera_eepose_max_age_sec") <= 0.0:
            raise ValueError("camera_eepose_max_age_sec must be positive")
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
            "drag_box_left_join_velocity_percent",
            "drag_box_left_join_timeout_sec",
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
            "box_post_movel_step4_movej_timeout_sec",
            "box_post_movel_step4_movej_position_tolerance_rad",
            "box_post_movel_step4_movej_velocity_tolerance_rad_sec",
            "box_post_movel_step4_movej_feedback_max_age_sec",
            "box_pre_detection_right_movej_command_units_per_degree",
            "box_pre_detection_right_movej_timeout_sec",
            "box_pre_detection_right_movej_position_tolerance_rad",
            "box_pre_detection_right_movej_velocity_tolerance_rad_sec",
            "box_pre_detection_right_movej_feedback_max_age_sec",
            "box_pre_detection_left_movej_command_units_per_degree",
            "box_pre_detection_left_movej_timeout_sec",
            "box_pre_detection_left_movej_position_tolerance_rad",
            "box_pre_detection_left_movej_velocity_tolerance_rad_sec",
            "box_pre_detection_left_movej_feedback_max_age_sec",
            "box_pre_target_arm_movej_command_units_per_degree",
            "box_pre_target_arm_movej_position_tolerance_rad",
            "box_pre_target_arm_movej_velocity_tolerance_rad_sec",
            "box_pre_target_arm_movej_feedback_max_age_sec",
            "box_pre_target_arm_movej_timeout_sec",
            "box_body_home_timeout_sec",
            "box_step2_waist_endpoint_sync_feedback_max_age_sec",
            "box_step2_waist_endpoint_sync_timeout_sec",
            "box_step2_waist_endpoint_sync_stable_samples",
            "box_step2_waist_endpoint_sync_final_position_tolerance_m",
            "box_step2_waist_endpoint_sync_final_orientation_tolerance_rad",
            "grasp_box_tf_body_home_carry_timeout_sec",
            "grasp_box_tf_body_home_carry_tf_timeout_sec",
            "grasp_box_tf_body_home_carry_position_tolerance_m",
            "grasp_box_tf_body_home_carry_orientation_tolerance_rad",
            "grasp_box_tf_body_home_carry_stable_samples",
            "grasp_box_tf_body_home_carry_left_movel_velocity_percent",
            "grasp_box_tf_body_home_carry_right_movel_velocity_percent",
            "grasp_box_tf_body_home_carry_final_correction_velocity_percent",
            "place_box_test_left_movel_velocity_percent",
            "place_box_test_right_movel_velocity_percent",
            "place_box_test_final_correction_velocity_percent",
            "place_box_test_timeout_sec",
            "place_box_test_start_body_tolerance_rad",
            "place_box_test_position_tolerance_m",
            "place_box_test_orientation_tolerance_rad",
            "place_box_test_target_consistency_position_tolerance_m",
            "place_box_test_target_consistency_orientation_tolerance_rad",
        )
        for name in positive_parameters:
            if not math.isfinite(self._float(name)) or self._float(name) <= 0.0:
                raise ValueError(f"parameter '{name}' must be finite and positive")

        for name in (
            "adaptive_grasp_velocity_percent",
            "adaptive_lift_velocity_percent",
            "box_post_movel_velocity_percent",
            "drag_box_left_join_velocity_percent",
            "grasp_box_tf_body_home_carry_left_movel_velocity_percent",
            "grasp_box_tf_body_home_carry_right_movel_velocity_percent",
            "grasp_box_tf_body_home_carry_final_correction_velocity_percent",
            "place_box_test_left_movel_velocity_percent",
            "place_box_test_right_movel_velocity_percent",
            "place_box_test_final_correction_velocity_percent",
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
            "box_foundation_pose_pre_settle_sec",
            "box_foundation_pose_post_settle_sec",
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
        if self._float("box_post_movel_step4_movej_command_units_per_degree") <= 0.0:
            raise ValueError(
                "box_post_movel_step4_movej_command_units_per_degree must be positive"
            )
        if not math.isfinite(
            float(self._integer("box_post_movel_step4_movej_joint2_units"))
        ):
            raise ValueError(
                "box_post_movel_step4_movej_joint2_units must be finite"
            )
        if not 1 <= self._integer("box_post_movel_step4_movej_velocity") <= 100:
            raise ValueError(
                "box_post_movel_step4_movej_velocity must be in [1, 100]"
            )
        for name in (
            "box_post_movel_step4_movej_left_device",
            "box_post_movel_step4_movej_right_device",
        ):
            if self._integer(name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self._integer("box_post_movel_step4_movej_blend_radius") < 0:
            raise ValueError(
                "box_post_movel_step4_movej_blend_radius must be nonnegative"
            )
        if self._integer("box_post_movel_step4_movej_trajectory_connect") not in (0, 1):
            raise ValueError(
                "box_post_movel_step4_movej_trajectory_connect must be 0 or 1"
            )
        if self._integer("box_post_movel_step4_movej_stable_samples") <= 0:
            raise ValueError(
                "box_post_movel_step4_movej_stable_samples must be positive"
            )
        for name in (
            "box_pre_detection_right_movej_device",
            "box_pre_detection_left_movej_device",
        ):
            if self._integer(name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "box_pre_target_arm_movej_left_device",
            "box_pre_target_arm_movej_right_device",
        ):
            if self._integer(name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "box_pre_detection_right_movej_velocity",
            "box_pre_detection_left_movej_velocity",
            "box_pre_target_arm_movej_velocity",
            "box_preparation_movej_velocity",
        ):
            if not 1 <= self._integer(name) <= 100:
                raise ValueError(f"{name} must be in [1, 100]")
        for name in (
            "box_pre_detection_right_movej_blend_radius",
            "box_pre_detection_left_movej_blend_radius",
            "box_pre_target_arm_movej_blend_radius",
        ):
            if self._integer(name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "box_pre_detection_right_movej_trajectory_connect",
            "box_pre_detection_left_movej_trajectory_connect",
            "box_pre_target_arm_movej_trajectory_connect",
        ):
            if self._integer(name) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if self._integer("box_pre_target_arm_movej_stable_samples") <= 0:
            raise ValueError(
                "box_pre_target_arm_movej_stable_samples must be positive"
            )
        for name in (
            "box_pre_detection_right_movej_stable_samples",
            "box_pre_detection_left_movej_stable_samples",
        ):
            if self._integer(name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self._integer("box_body_home_velocity") <= 0:
            raise ValueError("box_body_home_velocity must be positive")
        if self._integer("box_body_home_blend_radius") < 0:
            raise ValueError("box_body_home_blend_radius must be nonnegative")
        if self._integer("grasp_box_tf_body_home_carry_segments") <= 0:
            raise ValueError(
                "grasp_box_tf_body_home_carry_segments must be positive"
            )
        if not 1 <= self._integer(
            "grasp_box_tf_body_home_carry_body_velocity"
        ) <= 100:
            raise ValueError(
                "grasp_box_tf_body_home_carry_body_velocity must be in [1, 100]"
            )
        if not 0 <= self._integer(
            "grasp_box_tf_body_home_carry_body_blend_radius"
        ) <= 100:
            raise ValueError(
                "grasp_box_tf_body_home_carry_body_blend_radius must be in [0, 100]"
            )
        if not 0 <= self._integer(
            "grasp_box_tf_body_home_carry_arm_blend_radius"
        ) <= 100:
            raise ValueError(
                "grasp_box_tf_body_home_carry_arm_blend_radius must be in [0, 100]"
            )
        if not self._string(
            "grasp_box_tf_body_home_carry_carrier_frame"
        ).strip().lstrip("/"):
            raise ValueError(
                "grasp_box_tf_body_home_carry_carrier_frame must not be empty"
            )
        if self._string("place_box_test_box_type").strip().lower() != "smallbox":
            raise ValueError("place_box_test_box_type must be 'smallbox'")
        if self._integer("place_box_test_segments") <= 0:
            raise ValueError("place_box_test_segments must be positive")
        if not 1 <= self._integer("place_box_test_body_velocity") <= 100:
            raise ValueError("place_box_test_body_velocity must be in [1, 100]")
        for name in (
            "place_box_test_body_blend_radius",
            "place_box_test_arm_blend_radius",
        ):
            if not 0 <= self._integer(name) <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        if self._integer("place_box_test_stable_samples") <= 0:
            raise ValueError("place_box_test_stable_samples must be positive")
        if (
            self._boolean("grasp_box_tf_body_home_carry_enabled")
            and self._boolean("box_step2_waist_endpoint_sync_enabled")
        ):
            raise ValueError(
                "grasp_box_tf_body_home_carry_enabled and "
                "box_step2_waist_endpoint_sync_enabled are mutually exclusive"
            )
        if not 0 <= self._integer("box_step2_waist_endpoint_sync_body_blend_radius") <= 100:
            raise ValueError(
                "box_step2_waist_endpoint_sync_body_blend_radius must be in [0, 100]"
            )
        for layer in range(1, 5):
            prefix = f"box_step2_waist_endpoint_sync_layer{layer}_"
            if self._integer(f"{prefix}segments") not in (1, 2):
                raise ValueError(f"{prefix}segments must be 1 or 2")
            for name in (
                f"{prefix}forward_body_velocity",
                f"{prefix}reverse_body_velocity",
            ):
                if not 1 <= self._integer(name) <= 100:
                    raise ValueError(f"{name} must be in [1, 100]")
            for name in (
                f"{prefix}forward_left_movel_velocity_percent",
                f"{prefix}forward_right_movel_velocity_percent",
                f"{prefix}reverse_left_movel_velocity_percent",
                f"{prefix}reverse_right_movel_velocity_percent",
            ):
                if not 1.0 <= self._float(name) <= 100.0:
                    raise ValueError(f"{name} must be in [1, 100]")
        if self._integer("box_joint1_device") <= 0:
            raise ValueError("box_joint1_device must be positive")
        if self._integer("box_joint1_velocity") <= 0:
            raise ValueError("box_joint1_velocity must be positive")
        if self._integer("box_joint1_velocity") > 100:
            raise ValueError("box_joint1_velocity must be in (0, 100]")
        if self._integer("box_body_movej_velocity") <= 0:
            raise ValueError("box_body_movej_velocity must be positive")
        if self._integer("box_body_movej_velocity") > 100:
            raise ValueError("box_body_movej_velocity must be in (0, 100]")
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

        # Validate every generated TF action/model/layer profile at startup.
        # This catches a missing or malformed layer value before an action is
        # accepted, while retaining the legacy parameter validation above.
        for name, _default in self._tf_layer_parameter_defaults():
            if "approach_angle_deg" in name:
                if not math.isfinite(self._float(name)):
                    raise ValueError(f"parameter '{name}' must be finite")
                continue
            expected_length = (
                7
                if (
                    "pre_detection_" in name and "_movej_joint_units" in name
                    or "post_detection_" in name and "_movej_joint_units" in name
                    or "target_correction_pose_box" in name
                )
                else 3
            )
            values = self._float_array(name)
            if len(values) != expected_length:
                raise ValueError(
                    f"parameter '{name}' must contain {expected_length} values"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"parameter '{name}' contains NaN or Inf")
            if "target_correction_pose_box" in name and math.sqrt(
                sum(value * value for value in values[3:])
            ) <= 1e-12:
                raise ValueError(f"parameter '{name}' contains a zero quaternion")

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

    def _boolean_array(self, name: str) -> list[bool]:
        return [bool(value) for value in self.get_parameter(name).value]

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
        return self._box_grasp_goal_callback_for_mission(request, "box_grasp")

    def _tf_grasp_goal_prerequisites(self, label: str) -> bool:
        if not self._boolean("box_direct_movel_enabled"):
            self.get_logger().warning(
                f"rejecting {label} goal: box_direct_movel_enabled must be true"
            )
            return False
        if self._string("direct_movel_target_mode").strip().lower() != (
            "camera_offset_box_orientation"
        ):
            self.get_logger().warning(
                f"rejecting {label} goal: direct_movel_target_mode must be "
                "camera_offset_box_orientation"
            )
            return False
        return True

    def _grasp_box_tf_goal_callback(
        self, request: ExecuteBoxGrasp.Goal
    ) -> GoalResponse:
        if not self._tf_grasp_goal_prerequisites("TF GraspBox"):
            return GoalResponse.REJECT
        return self._box_grasp_goal_callback_for_mission(
            request, "grasp_box_tf"
        )

    def _drag_box_grasp_goal_callback(
        self, request: ExecuteDragBoxGrasp.Goal
    ) -> GoalResponse:
        return self._drag_box_grasp_goal_callback_for_mission(
            request, "drag_box_grasp", require_tf=False
        )

    def _drag_box_grasp_tf_goal_callback(
        self, request: ExecuteDragBoxGrasp.Goal
    ) -> GoalResponse:
        return self._drag_box_grasp_goal_callback_for_mission(
            request, "drag_box_grasp_tf", require_tf=True
        )

    def _drag_box_grasp_goal_callback_for_mission(
        self,
        request: ExecuteDragBoxGrasp.Goal,
        mission_name: str,
        *,
        require_tf: bool,
    ) -> GoalResponse:
        if require_tf and not self._tf_grasp_goal_prerequisites("TF DragBox"):
            return GoalResponse.REJECT
        if not self._boolean("box_direct_movel_enabled"):
            self.get_logger().warning(
                f"rejecting {mission_name} goal: "
                "box_direct_movel_enabled must be true"
            )
            return GoalResponse.REJECT
        if (
            not request.dry_run
            and self._string("direct_motion_backend").strip().lower()
            != "python_sdk"
        ):
            self.get_logger().warning(
                f"rejecting {mission_name} goal: physical DragBox execution "
                "requires direct_motion_backend=python_sdk"
            )
            return GoalResponse.REJECT
        if not self._boolean("drag_box_post_movel_enabled"):
            self.get_logger().warning(
                f"rejecting {mission_name} goal: "
                "drag_box_post_movel_enabled must be true"
            )
            return GoalResponse.REJECT
        return self._box_grasp_goal_callback_for_mission(
            request, mission_name
        )

    def _box_grasp_goal_callback_for_mission(
        self,
        request,
        mission_name: str,
        *,
        tf_mode: bool = False,
        drag_mode: bool = False,
    ) -> GoalResponse:
        tf_mode = tf_mode or mission_name in (
            "grasp_box_tf",
            "drag_box_grasp_tf",
        )
        drag_mode = drag_mode or mission_name in (
            "drag_box_grasp",
            "drag_box_grasp_tf",
        )
        if request.box_layer < 1 or request.box_layer > 4:
            self.get_logger().warning(
                "rejecting box grasp goal: box_layer must be in [1, 4]"
            )
            return GoalResponse.REJECT
        try:
            # DragBox may select bigbox/smallbox per goal.  Empty values remain
            # backward-compatible and use the configured default model.
            model_label = self._box_model_label_for_request(request)
            self._box_layer_joint1_approach_angle_deg(
                request.box_layer,
                model_label,
                tf_mode=tf_mode,
                drag_mode=drag_mode,
            )
            detection_arm = self._box_detection_arm(
                tf_mode=tf_mode, drag_mode=drag_mode
            )
            if self._boolean(f"box_pre_detection_{detection_arm}_movej_enabled"):
                self._box_layer_pre_detection_arm_movej_joint_units(
                    request.box_layer,
                    model_label,
                    arm=detection_arm,
                    tf_mode=tf_mode,
                    drag_mode=drag_mode,
                )
        except MissionError as exc:
            self.get_logger().warning(f"rejecting box grasp goal: {exc}")
            return GoalResponse.REJECT
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
        return self._reserve_goal(mission_name, request.request_id)

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

    def _place_box_test_goal_callback(
        self, request: PlaceBoxTest.Goal
    ) -> GoalResponse:
        if not self._boolean("place_box_test_enabled") and not request.dry_run:
            self.get_logger().warning(
                "rejecting place_box_test goal: place_box_test_enabled is false"
            )
            return GoalResponse.REJECT
        if (
            not request.dry_run
            and self._string("direct_motion_backend").strip().lower()
            != "python_sdk"
        ):
            self.get_logger().warning(
                "rejecting place_box_test goal: physical execution requires "
                "direct_motion_backend=python_sdk"
            )
            return GoalResponse.REJECT
        if not self._last_grasp_box_tf_box_to_link7_targets:
            self.get_logger().warning(
                "rejecting place_box_test goal: no rigid box->Link7 state is "
                "available; run /grasp_box_tf in this controller process first"
            )
            return GoalResponse.REJECT
        return self._reserve_goal("place_box_test", request.request_id)


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
