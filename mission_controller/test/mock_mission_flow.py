import math
import threading
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from grasp_orchestrator_interfaces.srv import DetectGraspPose
from mission_interfaces.action import ExecuteGrasp, ExecutePlace
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
from task_interfaces.action import MoveArmPose
from task_interfaces.srv import ArmJoints, Home


class MockMissionSystem(Node):
    def __init__(self) -> None:
        super().__init__("mock_mission_system")
        self.events: list[str] = []
        self.events_lock = threading.Lock()
        self.arm_joint_call_count = 0
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
        self.create_service(ArmJoints, "/move_arm_j", self._move_arm_joints)
        self.create_service(Home, "/home", self._home)
        self.arm_pose_server = ActionServer(
            self, MoveArmPose, "/move_arm_p", self._move_arm_pose
        )
        self.grasp_client = ActionClient(self, ExecuteGrasp, "/execute_grasp")
        self.place_client = ActionClient(self, ExecutePlace, "/execute_place")

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

    def _move_arm_joints(self, request, response):
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
        response.success = True
        response.message = "mock joints complete"
        return response

    def _home(self, request, response):
        self._record(f"home:dry_run={str(request.dry_run).lower()}")
        response.success = True
        response.error_code = 0
        response.message = "mock home complete"
        return response

    def _move_arm_pose(self, goal_handle):
        self._record("move_arm_p")
        result = MoveArmPose.Result()
        result.success = True
        result.error_code = 0
        result.message = "mock pose complete"
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
        ["detect", "move_arm_p", "gripper:right:0.0", "torso:reset"],
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
        print("mock grasp and place missions passed")
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
