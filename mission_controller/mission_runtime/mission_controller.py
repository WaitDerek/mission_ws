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
from mission_interfaces.srv import AcquireMissionLease, ReleaseMissionLease

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
from .parameters import MissionParametersMixin
from .taskflow.lease import WorkflowLeaseManager

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
    MissionParametersMixin,
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
        # ArmSlaveData.wrench_stamped is the raw wrist force sensor signal.
        # Keep it independent from pose/joint feedback so missing force data
        # cannot invalidate an otherwise usable arm sample.
        self.latest_slave_arm_wrenches = {"left": None, "right": None}
        self.latest_slave_arm_wrench_times = {"left": 0.0, "right": 0.0}
        self.latest_slave_arm_wrench_sequences = {"left": 0, "right": 0}
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
        self.mission_lease_manager = WorkflowLeaseManager()
        self.active_arm_joints_goal_handle = None
        self.active_go_ready_goal_handle = None
        self.active_box_object_pose_goal_handle = None
        self.active_pickup_task_goal_handle = None
        self.acquire_mission_lease_service = self.create_service(
            AcquireMissionLease,
            self._string("acquire_mission_lease_service_name"),
            self._acquire_mission_lease_callback,
            callback_group=self.server_group,
        )
        self.release_mission_lease_service = self.create_service(
            ReleaseMissionLease,
            self._string("release_mission_lease_service_name"),
            self._release_mission_lease_callback,
            callback_group=self.server_group,
        )

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

    def _reserve_goal(
        self,
        mission: str,
        request_id: str,
    ) -> GoalResponse:
        reservation = self.mission_lease_manager.reserve_goal(mission, request_id)
        if not reservation.accepted:
            self.get_logger().warning(
                f"rejecting {mission} goal: {reservation.message}; "
                f"request_id={reservation.sanitized_request_id}"
            )
            return GoalResponse.REJECT
        self.get_logger().info(
            f"accepted {mission} goal " f"request_id={reservation.sanitized_request_id}"
        )
        return GoalResponse.ACCEPT

    def _acquire_mission_lease_callback(self, request, response):
        result = self.mission_lease_manager.acquire(request.workflow_id)
        response.success = result.success
        response.lease_token = result.lease_token
        response.message = result.message
        if result.success:
            self.get_logger().info(
                f"workflow lease acquired workflow_id={request.workflow_id}"
            )
        else:
            self.get_logger().warning(
                "workflow lease rejected "
                f"workflow_id={request.workflow_id}: {result.message}"
            )
        return response

    def _release_mission_lease_callback(self, request, response):
        result = self.mission_lease_manager.release(
            request.workflow_id, request.lease_token
        )
        response.success = result.success
        response.message = result.message
        if result.success:
            self.get_logger().info(
                f"workflow lease released workflow_id={request.workflow_id}"
            )
        else:
            self.get_logger().warning(
                "workflow lease release rejected "
                f"workflow_id={request.workflow_id}: {result.message}"
            )
        return response

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

    def _box_grasp_goal_callback(self, request: ExecuteBoxGrasp.Goal) -> GoalResponse:
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
        return self._box_grasp_goal_callback_for_mission(request, "grasp_box_tf")

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
        if not require_tf and self._boolean(
            "drag_box_tf_body_home_carry_enabled"
        ):
            self.get_logger().warning(
                f"rejecting {mission_name} goal: "
                "drag_box_tf_body_home_carry_enabled is only valid for "
                "/execute_drag_box_grasp_tf"
            )
            return GoalResponse.REJECT
        if not self._boolean("box_direct_movel_enabled"):
            self.get_logger().warning(
                f"rejecting {mission_name} goal: "
                "box_direct_movel_enabled must be true"
            )
            return GoalResponse.REJECT
        if (
            not request.dry_run
            and self._string("direct_motion_backend").strip().lower() != "python_sdk"
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
        return self._box_grasp_goal_callback_for_mission(request, mission_name)

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

    def _box_place_goal_callback(self, request: ExecuteBoxPlace.Goal) -> GoalResponse:
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

    def _place_box_test_goal_callback(self, request: PlaceBoxTest.Goal) -> GoalResponse:
        if not self._boolean("place_box_test_enabled") and not request.dry_run:
            self.get_logger().warning(
                "rejecting place_box_test goal: place_box_test_enabled is false"
            )
            return GoalResponse.REJECT
        if (
            not request.dry_run
            and self._string("direct_motion_backend").strip().lower() != "python_sdk"
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
        self.mission_lease_manager.release_goal()
        with self.state_lock:
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
