import math
import threading
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from grasp_orchestrator_interfaces.srv import DetectGraspPose
from mission_interfaces.action import ExecuteBinGrasp, ExecuteGrasp, ExecutePlace
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
from task_interfaces.action import Home, MoveArmJoints, MoveArmPose, PickupTask


class MockMissionSystem(Node):
    def __init__(self) -> None:
        super().__init__("mock_mission_system")
        self.events: list[str] = []
        self.events_lock = threading.Lock()
        self.arm_joint_call_count = 0
        self.arm_pose_call_count = 0
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
        self.bin_grasp_client = ActionClient(
            self, ExecuteBinGrasp, "/execute_bin_grasp"
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

    def _on_torso(self, message: JointState) -> None:
        if not message.position:
            return
        if all(abs(value) < 1e-8 for value in message.position):
            self._record("torso:reset")
        else:
            self._record("torso:prepare")

    def _on_gripper(self, arm: str, message: JointState) -> None:
        if message.position:
            self._record(f"gripper:{arm}:{message.position[0]:.1f}")

    def _on_chassis(self, message: TwistStamped) -> None:
        speed = math.hypot(message.twist.linear.x, message.twist.linear.y)
        speed += abs(message.twist.angular.z)
        self._record("chassis:moving" if speed > 1e-8 else "chassis:stopped")

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
        return response

    def _move_arm_joints(self, goal_handle):
        request = goal_handle.request
        self.arm_joint_call_count += 1
        if self.arm_joint_call_count == 1:
            if len(request.left_joints) != 7 or len(request.right_joints) != 7:
                raise AssertionError("grasp preparation should target both arms")
            expected_left = [-0.88, 0.84, -1.13, -1.80, 1.25, 0.29, 0.13]
            expected_right = [-0.98, -0.64, 1.13, -1.60, -1.25, 0.6, -0.13]
            if list(request.left_joints) != expected_left:
                raise AssertionError("unexpected grasp left preparation target")
            if list(request.right_joints) != expected_right:
                raise AssertionError("unexpected grasp right preparation target")
        else:
            if request.left_joints:
                raise AssertionError("place flow should leave left_joints empty")
            if len(request.right_joints) != 7:
                raise AssertionError("right_joints must contain seven positions")
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

    def _move_arm_pose(self, goal_handle):
        request = goal_handle.request
        self.arm_pose_call_count += 1
        expected_right_pose = (
            [
                0.075,
                -0.05,
                0.075,
                0.2886751346,
                -0.2886751346,
                -0.2886751346,
                0.8660254038,
            ]
            if self.arm_pose_call_count == 1
            else [0.15, -0.10, 0.15, 0.5, -0.5, -0.5, 0.5]
        )
        if len(request.right_pose) != 7:
            raise AssertionError("grasp should send a seven-value right-arm pose")
        for actual, expected in zip(request.right_pose, expected_right_pose):
            if not math.isclose(actual, expected, abs_tol=1e-6):
                raise AssertionError(
                    "grasp-center extrinsic was not applied to right-arm target: "
                    f"actual={list(request.right_pose)}"
                )
        self._record(f"move_arm_p:{self.arm_pose_call_count}/2")
        result = MoveArmPose.Result()
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
        if request.box_type != "f320" or not request.dry_run:
            raise AssertionError("unexpected pickup task metadata")
        self._record("pickup_task:dry_run=true")
        result = PickupTask.Result()
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
    if not wrapped_result.result.success:
        raise AssertionError(wrapped_result.result.message)
    time.sleep(0.1)
    events = node.snapshot()
    assert_all_before(
        events,
        ["gripper:left:100.0", "gripper:right:100.0", "torso:prepare", "move_arm_j"],
        "detect",
    )
    assert_in_order(
        events,
        [
            "detect",
            "move_arm_p:1/2",
            "move_arm_p:2/2",
            "gripper:right:0.0",
            "torso:reset",
        ],
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
    if not wrapped_result.result.success:
        raise AssertionError(wrapped_result.result.message)
    time.sleep(0.1)
    assert_in_order(
        node.snapshot(),
        [
            "chassis:moving",
            "chassis:stopped",
            "torso:prepare",
            "move_arm_j",
            "gripper:right:100.0",
            "torso:reset",
            "home:dry_run=false",
        ],
    )


def run_bin_grasp(node: MockMissionSystem) -> None:
    node.clear_events()
    if not node.bin_grasp_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("/execute_bin_grasp action server not available")
    goal = ExecuteBinGrasp.Goal()
    goal.request_id = "mock_bin_grasp"
    goal.target_label = -1
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = True
    goal_handle = wait_future(node.bin_grasp_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock bin grasp goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    if result.grasp_pose.header.frame_id != "torso_link":
        raise AssertionError("bin result must expose the transformed body-frame pose")
    assert_in_order(
        node.snapshot(), ["foundation_pose", "pickup_task:dry_run=true"]
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
        run_place(node)
        run_bin_grasp(node)
        print("mock grasp, place, and bin-pickup missions passed")
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
