import math
import threading
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from grasp_orchestrator_interfaces.srv import DetectGraspPose
from mission_interfaces.action import (
    ExecuteBoxGrasp,
    ExecuteBoxPlace,
    ExecuteGrasp,
    ExecutePlace,
)
from object_pose_interfaces.action import EstimateObjectPose
from rclpy.action import ActionClient, ActionServer
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


class MockMissionSystem(Node):
    def __init__(self) -> None:
        super().__init__("mock_mission_system")
        self.events: list[str] = []
        self.events_lock = threading.Lock()
        self.arm_joint_call_count = 0
        self.arm_pose_call_count = 0
        self.arm_pose_failure_stages: list[int] = []
        self.pickup_call_count = 0
        self.pickup_failures_remaining = 0
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            JointState,
            "/motion_target/target_joint_state_torso",
            self._on_torso,
            command_qos,
        )
        self.create_subscription(
            JointState,
            "/motion_target/target_position_gripper_left",
            lambda message: self._on_gripper("left", message),
            command_qos,
        )
        self.create_subscription(
            JointState,
            "/motion_target/target_position_gripper_right",
            lambda message: self._on_gripper("right", message),
            command_qos,
        )
        self.create_subscription(
            TwistStamped,
            "/motion_target/target_speed_chassis",
            self._on_chassis,
            command_qos,
        )

        self.create_service(DetectGraspPose, "/detect_grasp_pose", self._detect)
        self.arm_joints_server = ActionServer(
            self, MoveArmJoints, "/move_arm_j", self._move_arm_joints
        )
        self.home_server = ActionServer(self, Home, "/home", self._home)
        self.go_ready_server = ActionServer(
            self, GoReady, "/go_ready", self._go_ready
        )
        self.arm_pose_server = ActionServer(
            self, MoveArmPose, "/move_arm_p", self._move_arm_pose
        )
        self.object_pose_server = ActionServer(
            self,
            EstimateObjectPose,
            "/object_pose/estimate",
            self._estimate_object_pose,
        )
        self.pickup_server = ActionServer(
            self, PickupTask, "/pickup_task", self._pickup_task
        )
        self.grasp_client = ActionClient(self, ExecuteGrasp, "/execute_grasp")
        self.place_client = ActionClient(self, ExecutePlace, "/execute_place")
        self.box_grasp_client = ActionClient(
            self, ExecuteBoxGrasp, "/execute_box_grasp"
        )
        self.box_place_client = ActionClient(
            self, ExecuteBoxPlace, "/execute_box_place"
        )

    def _record(self, event: str) -> None:
        with self.events_lock:
            self.events.append(event)

    def snapshot(self) -> list[str]:
        with self.events_lock:
            return list(self.events)

    def clear_events(self) -> None:
        with self.events_lock:
            self.events.clear()

    def configure_pickup_failures(self, count: int) -> None:
        self.pickup_call_count = 0
        self.pickup_failures_remaining = count

    def configure_arm_pose_failures(self, stages: list[int]) -> None:
        self.arm_pose_failure_stages = list(stages)

    def _on_torso(self, message: JointState) -> None:
        if not message.position:
            return
        positions = list(message.position)
        if all(abs(value) < 1e-8 for value in positions):
            self._record("torso:reset")
        elif all(
            math.isclose(actual, expected, abs_tol=1e-8)
            for actual, expected in zip(
                positions, [0.61, -0.81, -0.21, 0.0]
            )
        ):
            self._record("torso:grasp_observation")
        elif all(
            math.isclose(actual, expected, abs_tol=1e-8)
            for actual, expected in zip(
                positions, [0.61, -0.81, -0.6, 0.0]
            )
        ):
            self._record("torso:deep_observation")
        else:
            self._record("torso:prepare")

    def _on_gripper(self, arm: str, message: JointState) -> None:
        if message.position:
            self._record(f"gripper:{arm}:{message.position[0]:.1f}")

    def _on_chassis(self, message: TwistStamped) -> None:
        speed = math.hypot(message.twist.linear.x, message.twist.linear.y)
        speed += abs(message.twist.angular.z)
        if speed <= 1e-8:
            self._record("chassis:stopped")
        elif (
            message.twist.linear.y < 0.0
            and abs(message.twist.linear.x) <= 1e-8
            and abs(message.twist.angular.z) <= 1e-8
        ):
            self._record("chassis:right")
        else:
            self._record("chassis:moving")

    def _detect(self, _request, response):
        self._record("detect")
        response.success = True
        response.message = "mock grasp"
        response.grasp_pose.header.frame_id = "torso_link"
        response.grasp_pose.header.stamp = self.get_clock().now().to_msg()
        response.grasp_pose.pose.position.x = 0.30
        response.grasp_pose.pose.position.y = -0.10
        response.grasp_pose.pose.position.z = 0.15
        response.grasp_pose.pose.orientation.x = 0.7071
        response.grasp_pose.pose.orientation.w = 0.7071
        response.score = 0.9
        response.width = 0.08
        response.height = 0.02
        response.depth = 0.03
        response.object_id = 1
        response.source_frame = "torso_link"
        response.candidate_poses.extend(
            [response.grasp_pose, response.grasp_pose]
        )
        response.candidate_scores.extend([0.9, 0.8])
        response.candidate_widths.extend([0.08, 0.08])
        response.candidate_heights.extend([0.02, 0.02])
        response.candidate_depths.extend([0.03, 0.03])
        response.candidate_object_ids.extend([1, 1])
        return response

    def _move_arm_joints(self, goal_handle):
        request = goal_handle.request
        self.arm_joint_call_count += 1
        left = list(request.left_joints)
        right = list(request.right_joints)
        intermediate_left = [1.30, 0.6, 0.0, -1.5, 0.0, 0.0, 0.0]
        intermediate_right = [1.30, -0.6, 0.0, -1.5, 0.0, 0.0, 0.0]
        grasp_left = [-0.98, 0.84, -0.83, -2.00, 1.25, 0.29, 0.13]
        grasp_right = [-0.98, -0.84, 0.93, -2.00, -1.25, 0.60, -0.13]
        box_left = [-0.88, 1.24, -0.70, -2.0, 1.25, 0.1, 0.0]
        box_right = [
            0.86,
            -0.24,
            0.20,
            -2.0944,
            0.174647,
            -0.618606,
            0.104098,
        ]
        if left == intermediate_left and right == intermediate_right:
            pass
        elif left == grasp_left and right == grasp_right:
            pass
        elif not left:
            if request.left_joints:
                raise AssertionError("place flow should leave left_joints empty")
            if len(request.right_joints) != 7:
                raise AssertionError("right_joints must contain seven positions")
            expected_place_right = [
                -1.011,
                0.040,
                0.835,
                -0.9513,
                -1.956,
                0.901,
                -1.370,
            ]
            if right != expected_place_right:
                raise AssertionError(
                    f"unexpected right-arm place target: {right}"
                )
        elif left == box_left and right == box_right:
            pass
        else:
            raise AssertionError(
                f"unexpected move_arm_j target left={left} right={right}"
            )
        self._record("move_arm_j")
        result = MoveArmJoints.Result()
        result.success = True
        result.error_code = 0
        result.message = "mock joints complete"
        goal_handle.succeed()
        return result

    def _home(self, goal_handle):
        request = goal_handle.request
        self._record(f"home:dry_run={str(request.dry_run).lower()}")
        result = Home.Result()
        result.success = True
        result.error_code = 0
        result.message = "mock home complete"
        goal_handle.succeed()
        return result

    def _go_ready(self, goal_handle):
        request = goal_handle.request
        self._record(f"go_ready:dry_run={str(request.dry_run).lower()}")
        result = GoReady.Result()
        result.success = True
        result.error_code = 0
        result.message = "mock ready complete"
        goal_handle.succeed()
        return result

    def _move_arm_pose(self, goal_handle):
        request = goal_handle.request
        self.arm_pose_call_count += 1
        stage = 1 if math.isclose(request.right_pose[0], 0.135, abs_tol=1e-6) else 2
        expected_right_pose = (
            [
                0.135,
                -0.05,
                0.075,
                0.2286958501,
                -0.3589805774,
                -0.3589805774,
                0.8306407757,
            ]
            if stage == 1
            else [
                0.27,
                -0.10,
                0.15,
                0.3799281966,
                -0.5963678105,
                -0.5963678105,
                0.3799281966,
            ]
        )
        if len(request.right_pose) != 7:
            raise AssertionError("grasp should send a seven-value right-arm pose")
        for actual, expected in zip(request.right_pose, expected_right_pose):
            if not math.isclose(actual, expected, abs_tol=1e-6):
                raise AssertionError(
                    "grasp target conversion produced an unexpected right-arm pose: "
                    f"actual={list(request.right_pose)}"
                )
        event = f"move_arm_p:{stage}/2"
        result = MoveArmPose.Result()
        if self.arm_pose_failure_stages and self.arm_pose_failure_stages[0] == stage:
            self.arm_pose_failure_stages.pop(0)
            self._record(f"{event}:failed")
            result.success = False
            result.error_code = 13
            result.message = f"mock stage {stage} failure"
            goal_handle.abort()
            return result
        self._record(event)
        result.success = True
        result.error_code = 0
        result.message = "mock pose complete"
        goal_handle.succeed()
        return result

    def _estimate_object_pose(self, goal_handle):
        request = goal_handle.request
        if request.model_label != "f320" or request.instance_index != 0:
            raise AssertionError("unexpected FoundationPose goal")
        self._record("foundation_pose")
        result = EstimateObjectPose.Result()
        result.success = True
        result.message = "mock box detected"
        result.model_label = request.model_label
        result.detection_score = 0.95
        result.pose.header.frame_id = "torso_link"
        result.pose.header.stamp = self.get_clock().now().to_msg()
        result.pose.pose.position.x = 0.40
        result.pose.pose.position.z = 0.20
        result.pose.pose.orientation.w = 1.0
        goal_handle.succeed()
        return result

    def _pickup_task(self, goal_handle):
        request = goal_handle.request
        if request.box_pose.header.frame_id != "torso_link":
            raise AssertionError("pickup box pose must be in the body frame")
        if not math.isclose(request.box_width, 0.357, abs_tol=1e-9):
            raise AssertionError("unexpected pickup box width")
        if not math.isclose(request.box_height, 0.127, abs_tol=1e-9):
            raise AssertionError("unexpected pickup box height")
        expected_center = [0.40, 0.0, 0.20]
        actual_center = [
            request.box_pose.pose.position.x,
            request.box_pose.pose.position.y,
            request.box_pose.pose.position.z,
        ]
        for actual, expected in zip(actual_center, expected_center):
            if not math.isclose(actual, expected, abs_tol=1e-6):
                raise AssertionError(
                    "FoundationPose geometric centre changed before pickup: "
                    f"actual={actual_center}"
                )
        expected_orientation = [0.0, 0.0, 0.0, 1.0]
        actual_orientation = [
            request.box_pose.pose.orientation.x,
            request.box_pose.pose.orientation.y,
            request.box_pose.pose.orientation.z,
            request.box_pose.pose.orientation.w,
        ]
        for actual, expected in zip(actual_orientation, expected_orientation):
            if not math.isclose(actual, expected, abs_tol=1e-6):
                raise AssertionError(
                    "FoundationPose axes changed before pickup: "
                    f"actual={actual_orientation}"
                )
        if request.box_type != "f320":
            raise AssertionError("unexpected pickup task metadata")
        self.pickup_call_count += 1
        self._record(
            f"pickup_task:{self.pickup_call_count}:"
            f"dry_run={str(request.dry_run).lower()}"
        )
        result = PickupTask.Result()
        if self.pickup_failures_remaining > 0:
            self.pickup_failures_remaining -= 1
            result.success = False
            result.error_code = 13
            result.message = f"mock pickup failure {self.pickup_call_count}"
            goal_handle.abort()
            return result
        result.success = True
        result.error_code = 0
        result.message = "mock pickup plan complete"
        goal_handle.succeed()
        return result


def wait_future(future, timeout_sec: float):
    deadline = time.monotonic() + timeout_sec
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not future.done():
        raise TimeoutError("timed out waiting for ROS future")
    return future.result()


def assert_in_order(events: list[str], expected: list[str]) -> None:
    cursor = 0
    for event in expected:
        try:
            cursor = events.index(event, cursor) + 1
        except ValueError as exc:
            raise AssertionError(
                f"event '{event}' missing or out of order; events={events}"
            ) from exc


def assert_all_before(events: list[str], expected: list[str], marker: str) -> None:
    marker_index = events.index(marker)
    for event in expected:
        if event not in events[:marker_index]:
            raise AssertionError(
                f"event '{event}' did not occur before '{marker}'; events={events}"
            )


def run_grasp(node: MockMissionSystem) -> None:
    if not node.grasp_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("/execute_grasp action server not available")
    time.sleep(0.1)
    node.clear_events()
    goal = ExecuteGrasp.Goal()
    goal.request_id = "mock_grasp"
    goal.target_frame = "torso_link"
    goal.target_label = 0
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = False
    goal_handle = wait_future(node.grasp_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock grasp goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    if not result.torso_reset_command_published:
        raise AssertionError("grasp must report its torso observation-return command")
    time.sleep(0.1)
    events = node.snapshot()
    assert_all_before(
        events,
        [
            "gripper:left:100.0",
            "gripper:right:100.0",
            "torso:deep_observation",
            "move_arm_j",
        ],
        "detect",
    )
    detect_index = events.index("detect")
    first_arm_move_index = events.index("move_arm_j")
    if (
        events.index("gripper:left:100.0") > first_arm_move_index
        or events.index("gripper:right:100.0") > first_arm_move_index
    ):
        raise AssertionError(
            f"both grippers must open before initial arm movement; events={events}"
        )
    if events[:detect_index].count("move_arm_j") != 2:
        raise AssertionError(
            "grasp preparation must use intermediate and final arm targets; "
            f"events={events}"
        )
    assert_in_order(
        events,
        [
            "detect",
            "move_arm_p:1/2",
            "move_arm_p:2/2",
            "gripper:right:0.0",
        ],
    )
    gripper_closed_index = events.index("gripper:right:0.0")
    return_events = events[gripper_closed_index + 1 :]
    for expected in ("torso:deep_observation", "move_arm_j"):
        if expected not in return_events:
            raise AssertionError(
                f"grasp did not return to observation posture; events={events}"
            )
    if return_events.count("move_arm_j") != 1:
        raise AssertionError(
            "grasp return must go directly to the final observation target; "
            f"events={events}"
        )
    if "torso:reset" in events:
        raise AssertionError("grasp must retain the deep observation torso posture")


def run_grasp_stage_one_failure_continues_same_candidate(
    node: MockMissionSystem,
) -> None:
    node.clear_events()
    node.configure_arm_pose_failures([1])
    goal = ExecuteGrasp.Goal()
    goal.request_id = "mock_grasp_stage_one_continue"
    goal.target_frame = "torso_link"
    goal.target_label = 0
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = False
    goal_handle = wait_future(node.grasp_client.send_goal_async(goal), 5.0)
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    if not math.isclose(result.score, 0.9, abs_tol=1e-9):
        raise AssertionError("stage-1 failure must retain the highest-ranked grasp")
    events = node.snapshot()
    failed_index = events.index("move_arm_p:1/2:failed")
    final_index = events.index("move_arm_p:2/2", failed_index + 1)
    recovery_events = events[failed_index + 1 : final_index]
    if recovery_events.count("move_arm_j") != 0:
        raise AssertionError(
            "the same candidate final pose must be attempted directly after "
            "its intermediate pose fails; "
            f"events={events}"
        )
    assert_in_order(
        events,
        [
            "detect",
            "move_arm_p:1/2:failed",
            "move_arm_p:2/2",
        ],
    )


def run_grasp_second_candidate_after_stage_two_failure(
    node: MockMissionSystem,
) -> None:
    node.clear_events()
    node.configure_arm_pose_failures([2])
    goal = ExecuteGrasp.Goal()
    goal.request_id = "mock_grasp_stage_two_second_candidate"
    goal.target_frame = "torso_link"
    goal.target_label = 0
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = False
    goal_handle = wait_future(node.grasp_client.send_goal_async(goal), 5.0)
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    if not math.isclose(result.score, 0.8, abs_tol=1e-9):
        raise AssertionError("stage-2 failure must select the second-ranked grasp")
    events = node.snapshot()
    failed_index = events.index("move_arm_p:2/2:failed")
    next_candidate_index = events.index("move_arm_p:1/2", failed_index + 1)
    recovery_events = events[failed_index + 1 : next_candidate_index]
    if recovery_events.count("move_arm_j") != 1:
        raise AssertionError(
            "stage-2 failure must return directly to the final observation "
            f"joints before candidate 2; events={events}"
        )
    if events.count("detect") != 1:
        raise AssertionError(f"candidate 2 must reuse the same detection: {events}")
    assert_in_order(
        events,
        [
            "move_arm_p:2/2:failed",
            "move_arm_j",
            "move_arm_p:1/2",
            "move_arm_p:2/2",
        ],
    )


def run_grasp_redetect_after_both_candidate_final_failures(
    node: MockMissionSystem,
) -> None:
    node.clear_events()
    node.configure_arm_pose_failures([2, 2])
    goal = ExecuteGrasp.Goal()
    goal.request_id = "mock_grasp_both_candidate_finals_fail"
    goal.target_frame = "torso_link"
    goal.target_label = 0
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = False
    goal_handle = wait_future(node.grasp_client.send_goal_async(goal), 5.0)
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    events = node.snapshot()
    first_detect_index = events.index("detect")
    second_detect_index = events.index("detect", first_detect_index + 1)
    failed_indices = [
        index
        for index, event in enumerate(events[:second_detect_index])
        if event == "move_arm_p:2/2:failed"
    ]
    if len(failed_indices) != 2:
        raise AssertionError(
            f"both first-detection candidate final poses must fail: {events}"
        )
    between_failures = events[failed_indices[0] + 1 : failed_indices[1]]
    if between_failures.count("move_arm_j") != 1:
        raise AssertionError(
            f"candidate 2 must start after direct observation recovery: {events}"
        )
    before_redetect = events[failed_indices[1] + 1 : second_detect_index]
    if before_redetect.count("move_arm_j") != 1:
        raise AssertionError(
            "both candidate final poses failing must restore the observation "
            f"joints before re-detection; events={events}"
        )


def run_place(node: MockMissionSystem) -> None:
    node.clear_events()
    if not node.place_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("/execute_place action server not available")
    goal = ExecutePlace.Goal()
    goal.request_id = "mock_place"
    goal.arm = "right"
    goal.dry_run = False
    goal_handle = wait_future(node.place_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock place goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    if result.home_completed:
        raise AssertionError("material place must not call home after release")
    if not result.torso_reset_command_published:
        raise AssertionError("material place must report observation return")
    time.sleep(0.1)
    events = node.snapshot()
    assert_in_order(
        events,
        [
            "chassis:moving",
            "chassis:stopped",
            "torso:deep_observation",
            "move_arm_j",
            "gripper:right:100.0",
        ],
    )
    release_index = events.index("gripper:right:100.0")
    return_events = events[release_index + 1 :]
    if (
        "move_arm_j" not in return_events
        or "torso:deep_observation" not in return_events
    ):
        raise AssertionError(
            f"material place must return directly to observation; events={events}"
        )
    if "torso:reset" in return_events or "home:dry_run=false" in return_events:
        raise AssertionError(
            f"material place must not reset/home after release; events={events}"
        )


def run_box_grasp(node: MockMissionSystem) -> None:
    node.clear_events()
    node.configure_pickup_failures(1)
    if not node.box_grasp_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("/execute_box_grasp action server not available")
    goal = ExecuteBoxGrasp.Goal()
    goal.request_id = "mock_box_grasp"
    goal.target_label = -1
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = False
    goal_handle = wait_future(node.box_grasp_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock box grasp goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    if result.grasp_pose.header.frame_id != "torso_link":
        raise AssertionError("box result must expose the transformed body-frame pose")
    if not result.gripper_command_published:
        raise AssertionError("box grasp must close both grippers after pickup")
    if not result.torso_lift_command_published:
        raise AssertionError("box grasp must lift the torso after closing grippers")
    events = node.snapshot()
    assert_all_before(
        events,
        [
            "gripper:left:100.0",
            "gripper:right:100.0",
            "torso:deep_observation",
            "move_arm_j",
        ],
        "foundation_pose",
    )
    foundation_pose_index = events.index("foundation_pose")
    first_arm_move_index = events.index("move_arm_j")
    if (
        events.index("gripper:left:100.0") > first_arm_move_index
        or events.index("gripper:right:100.0") > first_arm_move_index
    ):
        raise AssertionError(
            "both box grippers must open before initial arm movement; "
            f"events={events}"
        )
    if events[:foundation_pose_index].count("move_arm_j") != 2:
        raise AssertionError(
            "box preparation must use intermediate and final arm targets; "
            f"events={events}"
        )
    assert_in_order(
        events,
        [
            "foundation_pose",
            "pickup_task:1:dry_run=false",
            "pickup_task:2:dry_run=false",
            "gripper:left:0.0",
            "gripper:right:0.0",
            "torso:grasp_observation",
        ],
    )


def run_box_grasp_double_failure(node: MockMissionSystem) -> None:
    node.clear_events()
    node.configure_pickup_failures(2)
    goal = ExecuteBoxGrasp.Goal()
    goal.request_id = "mock_box_grasp_double_failure"
    goal.target_label = -1
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = True
    goal_handle = wait_future(node.box_grasp_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock box grasp failure goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if result.success:
        raise AssertionError("box grasp must fail after two pickup failures")
    if "failed twice" not in result.message:
        raise AssertionError(f"unexpected double-failure message: {result.message}")
    events = node.snapshot()
    assert_in_order(
        events,
        [
            "foundation_pose",
            "pickup_task:1:dry_run=true",
            "pickup_task:2:dry_run=true",
        ],
    )
    if any(event.startswith("gripper:") for event in events):
        raise AssertionError("failed pickup must not command either gripper")
    if any(event.startswith("torso:") for event in events):
        raise AssertionError("failed dry-run pickup must not command the torso")


def run_box_place(node: MockMissionSystem) -> None:
    node.clear_events()
    if not node.box_place_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("/execute_box_place action server not available")
    goal = ExecuteBoxPlace.Goal()
    goal.request_id = "mock_box_place"
    goal.arm = "right"
    goal.dry_run = False
    goal_handle = wait_future(node.box_place_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock box place goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    if not result.gripper_command_published:
        raise AssertionError("box place must open both grippers")
    if not result.ready_completed:
        raise AssertionError("box place must restore the ready arm posture")
    if not result.torso_reset_command_published:
        raise AssertionError("box place must reset the torso after release")
    time.sleep(0.1)
    assert_in_order(
        node.snapshot(),
        [
            "chassis:right",
            "chassis:stopped",
            "torso:deep_observation",
            "gripper:left:100.0",
            "gripper:right:100.0",
            "go_ready:dry_run=false",
            "torso:reset",
        ],
    )


def main() -> None:
    rclpy.init()
    node = MockMissionSystem()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        run_grasp(node)
        run_grasp_stage_one_failure_continues_same_candidate(node)
        run_grasp_second_candidate_after_stage_two_failure(node)
        run_grasp_redetect_after_both_candidate_final_failures(node)
        run_place(node)
        run_box_grasp(node)
        run_box_grasp_double_failure(node)
        run_box_place(node)
        print("mock material and complete box grasp/place missions passed")
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
