import math
import re
import threading
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from grasp_orchestrator_interfaces.srv import DetectGraspPose
from mission_interfaces.action import (
    ExecuteBoxGrasp,
    ExecuteBoxPlace,
    ExecuteBoxStack,
    ExecuteGrasp,
    ExecutePlace,
    MoveChassis,
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
        self.pickup_failure_segment = 0
        self.gripper_close_feedback_sequences = {
            "left": [],
            "right": [],
        }
        self.active_gripper_close_feedback = {"left": 10.0, "right": 10.0}
        self.last_gripper_commands = {"left": None, "right": None}
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        feedback_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.joint_state_publisher = self.create_publisher(
            JointState, "/joint_states", 10
        )
        self.torso_feedback_publisher = self.create_publisher(
            JointState, "/hdas/feedback_torso", 10
        )
        self.left_gripper_feedback_publisher = self.create_publisher(
            JointState, "/hdas/feedback_gripper_left", feedback_qos
        )
        self.right_gripper_feedback_publisher = self.create_publisher(
            JointState, "/hdas/feedback_gripper_right", feedback_qos
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
        self.box_stack_client = ActionClient(
            self, ExecuteBoxStack, "/execute_box_stack"
        )
        self.move_chassis_client = ActionClient(
            self, MoveChassis, "/move_chassis"
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

    def configure_pickup_failures(
        self, count: int, approach_segment: int = 0
    ) -> None:
        self.pickup_call_count = 0
        self.pickup_failures_remaining = count
        self.pickup_failure_segment = approach_segment

    def configure_arm_pose_failures(self, stages: list[int]) -> None:
        self.arm_pose_failure_stages = list(stages)

    def configure_gripper_close_feedback(
        self, arm: str, positions: list[float]
    ) -> None:
        self.gripper_close_feedback_sequences[arm] = list(positions)
        self.active_gripper_close_feedback[arm] = 10.0
        self.last_gripper_commands[arm] = None

    def publish_box_observation_feedback(self) -> None:
        arm_message = JointState()
        arm_message.name = [
            *[f"left_arm_joint{index}" for index in range(1, 8)],
            *[f"right_arm_joint{index}" for index in range(1, 8)],
        ]
        arm_message.position = [
            -0.88,
            1.24,
            -0.70,
            -2.0,
            1.25,
            0.1,
            0.0,
            0.86,
            -0.24,
            0.20,
            -2.0944,
            0.174647,
            -0.618606,
            0.104098,
        ]
        torso_message = JointState()
        torso_message.position = [0.61, -0.81, -0.6, 0.0]
        for _ in range(5):
            stamp = self.get_clock().now().to_msg()
            arm_message.header.stamp = stamp
            torso_message.header.stamp = stamp
            self.joint_state_publisher.publish(arm_message)
            self.torso_feedback_publisher.publish(torso_message)
            time.sleep(0.02)

    def publish_grasp_observation_feedback(self) -> None:
        arm_message = JointState()
        arm_message.name = [
            *[f"left_arm_joint{index}" for index in range(1, 8)],
            *[f"right_arm_joint{index}" for index in range(1, 8)],
        ]
        arm_message.position = [
            -0.98,
            0.84,
            -0.83,
            -2.00,
            1.25,
            0.29,
            0.13,
            -0.98,
            -0.84,
            0.93,
            -2.00,
            -1.25,
            0.60,
            -0.13,
        ]
        torso_message = JointState()
        torso_message.position = [0.61, -0.81, -0.60, 0.0]
        for _ in range(5):
            stamp = self.get_clock().now().to_msg()
            arm_message.header.stamp = stamp
            torso_message.header.stamp = stamp
            self.joint_state_publisher.publish(arm_message)
            self.torso_feedback_publisher.publish(torso_message)
            time.sleep(0.02)

    def publish_stack_ready_feedback(self) -> None:
        arm_message = JointState()
        arm_message.name = [
            *[f"left_arm_joint{index}" for index in range(1, 8)],
            *[f"right_arm_joint{index}" for index in range(1, 8)],
        ]
        arm_message.position = [
            -1.133818,
            0.120475,
            -1.197170,
            -0.672971,
            2.354960,
            1.046860,
            1.240171,
            -1.129491,
            -0.125152,
            1.203598,
            -0.676157,
            -2.343158,
            1.040511,
            -1.249240,
        ]
        torso_message = JointState()
        torso_message.position = [0.0, 0.0, 0.0, 0.0]
        for _ in range(5):
            stamp = self.get_clock().now().to_msg()
            arm_message.header.stamp = stamp
            torso_message.header.stamp = stamp
            self.joint_state_publisher.publish(arm_message)
            self.torso_feedback_publisher.publish(torso_message)
            time.sleep(0.02)

    def publish_stack_not_ready_feedback(self) -> None:
        arm_message = JointState()
        arm_message.name = [
            *[f"left_arm_joint{index}" for index in range(1, 8)],
            *[f"right_arm_joint{index}" for index in range(1, 8)],
        ]
        arm_message.position = [0.0] * 14
        torso_message = JointState()
        torso_message.position = [0.0, 0.0, 0.0, 0.0]
        for _ in range(5):
            stamp = self.get_clock().now().to_msg()
            arm_message.header.stamp = stamp
            torso_message.header.stamp = stamp
            self.joint_state_publisher.publish(arm_message)
            self.torso_feedback_publisher.publish(torso_message)
            time.sleep(0.02)

    def _on_torso(self, message: JointState) -> None:
        if not message.position:
            return
        positions = list(message.position)
        level2_target = [1.407070, -2.598294, -1.091117, 0.0]
        if all(
            math.isclose(actual, expected, abs_tol=1e-8)
            for actual, expected in zip(positions, level2_target)
        ):
            duration = abs(level2_target[0]) / 0.1
            expected_velocities = [
                abs(value) / duration for value in level2_target
            ]
            if len(message.velocity) != 4 or any(
                not math.isclose(actual, expected, abs_tol=1e-8)
                for actual, expected in zip(
                    message.velocity, expected_velocities
                )
            ):
                raise AssertionError(
                    "stack torso velocities must synchronize every joint to "
                    f"Torso1 time: actual={list(message.velocity)}, "
                    f"expected={expected_velocities}"
                )
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
        elif all(
            math.isclose(actual, expected, abs_tol=1e-8)
            for actual, expected in zip(
                positions, [0.41, -0.81, -0.6, 0.0]
            )
        ):
            self._record("torso:box_clearance_lift")
        else:
            self._record("torso:prepare")
        feedback = JointState()
        feedback.header.stamp = self.get_clock().now().to_msg()
        feedback.position = positions
        self.torso_feedback_publisher.publish(feedback)

    def _on_gripper(self, arm: str, message: JointState) -> None:
        if not message.position:
            return
        command = float(message.position[0])
        self._record(f"gripper:{arm}:{command:.1f}")
        previous_command = self.last_gripper_commands[arm]
        if command <= 1.0:
            if previous_command is None or previous_command > 1.0:
                responses = self.gripper_close_feedback_sequences[arm]
                self.active_gripper_close_feedback[arm] = (
                    responses.pop(0) if responses else 10.0
                )
            feedback_position = self.active_gripper_close_feedback[arm]
        else:
            feedback_position = command
        self.last_gripper_commands[arm] = command

        feedback = JointState()
        feedback.header.stamp = self.get_clock().now().to_msg()
        feedback.position = [feedback_position]
        self._record(f"gripper_feedback:{arm}:{feedback_position:.1f}")
        publisher = (
            self.left_gripper_feedback_publisher
            if arm == "left"
            else self.right_gripper_feedback_publisher
        )
        publisher.publish(feedback)

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
        box_clearance_left = [
            -1.413830,
            0.687872,
            -1.236596,
            -1.839149,
            1.905532,
            0.525745,
            1.146383,
        ]
        box_clearance_right = [
            -1.403617,
            -0.668723,
            1.238298,
            -1.802128,
            -1.944043,
            0.435319,
            -1.307021,
        ]
        stack_left = [
            -1.133818,
            0.120475,
            -1.197170,
            -0.672971,
            2.354960,
            1.046860,
            1.240171,
        ]
        stack_right = [
            -1.129491,
            -0.125152,
            1.203598,
            -0.676157,
            -2.343158,
            1.040511,
            -1.249240,
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
        elif left == box_clearance_left and right == box_clearance_right:
            pass
        elif left == stack_left and right == stack_right:
            if not math.isclose(request.duration, 8.0, abs_tol=1e-9):
                raise AssertionError(
                    f"stack start motion must use 8.0s, got {request.duration}"
                )
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
                0.2418987109,
                -0.3454671618,
                -0.3454671618,
                0.8383256490,
            ]
            if stage == 1
            else [
                0.27,
                -0.10,
                0.15,
                0.4055797877,
                -0.5792279653,
                -0.5792279653,
                0.4055797877,
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
            if self.pickup_failure_segment > 0:
                feedback = PickupTask.Feedback()
                feedback.stage = "APPROACHING"
                feedback.progress = (
                    0.5 if self.pickup_failure_segment == 2 else 0.15
                )
                feedback.detail = (
                    f"executing pickup approach segment "
                    f"{self.pickup_failure_segment}/2"
                )
                goal_handle.publish_feedback(feedback)
                self._record(
                    f"pickup_feedback:{self.pickup_failure_segment}/2"
                )
            result.success = False
            result.error_code = 3 if self.pickup_failure_segment > 0 else 1
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


def assert_elapsed_result(result, action_name: str) -> None:
    if re.search(r"\(elapsed_sec=\d+\.\d{3}\)$", result.message) is None:
        raise AssertionError(
            f"{action_name} result does not contain a three-decimal elapsed "
            f"time: {result.message!r}"
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
    assert_elapsed_result(result, "execute_grasp")
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


def run_grasp_retries_after_empty_close(node: MockMissionSystem) -> None:
    node.clear_events()
    node.configure_arm_pose_failures([])
    node.configure_gripper_close_feedback("right", [2.0, 10.0])
    goal = ExecuteGrasp.Goal()
    goal.request_id = "mock_grasp_empty_then_success"
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
    if "10.000" not in result.message:
        raise AssertionError(
            f"successful retry must report measured gripper opening: {result.message}"
        )
    events = node.snapshot()
    if events.count("detect") != 2:
        raise AssertionError(
            f"an empty first close must trigger one fresh detection: {events}"
        )
    first_close = events.index("gripper:right:0.0")
    second_detection = events.index("detect", events.index("detect") + 1)
    recovery_events = events[first_close + 1 : second_detection]
    for expected in ("gripper:right:100.0", "move_arm_j"):
        if expected not in recovery_events:
            raise AssertionError(
                "close-verification retry must reopen and restore observation before "
                f"re-detection; events={events}"
            )
    if "gripper_feedback:right:2.0" not in recovery_events:
        raise AssertionError(
            f"test did not expose an empty first close: {events}"
        )


def run_grasp_continues_after_ten_empty_closes(
    node: MockMissionSystem,
) -> None:
    node.clear_events()
    node.configure_arm_pose_failures([])
    node.configure_gripper_close_feedback("right", [2.0] * 10 + [10.0])
    goal = ExecuteGrasp.Goal()
    goal.request_id = "mock_grasp_ten_empty_then_success"
    goal.target_frame = "torso_link"
    goal.target_label = 0
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = False
    goal_handle = wait_future(node.grasp_client.send_goal_async(goal), 5.0)
    wrapped_result = wait_future(goal_handle.get_result_async(), 30.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(
            "empty-grasp feedback must keep retrying until success: "
            f"{result.message}"
        )
    if "10.000" not in result.message:
        raise AssertionError(
            f"successful retry must report retained opening: {result.message}"
        )
    events = node.snapshot()
    if events.count("detect") != 11:
        raise AssertionError(
            "ten empty closes followed by success must perform eleven "
            f"detections: {events}"
        )
    if events.count("gripper:right:0.0") != 24:
        raise AssertionError(
            "eleven close attempts plus final verification must publish the "
            f"expected command samples; events={events}"
        )


def run_grasp_reuses_ready_observation(node: MockMissionSystem) -> None:
    node.clear_events()
    node.configure_arm_pose_failures([])
    node.publish_grasp_observation_feedback()
    time.sleep(0.1)
    goal = ExecuteGrasp.Goal()
    goal.request_id = "mock_grasp_ready_reuse"
    goal.target_frame = "torso_link"
    goal.target_label = 0
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = False
    goal_handle = wait_future(node.grasp_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock ready-reuse grasp goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    assert_elapsed_result(result, "execute_grasp ready reuse")
    time.sleep(0.1)
    events = node.snapshot()
    detection_index = events.index("detect")
    if "move_arm_j" in events[:detection_index]:
        raise AssertionError(
            "an already-ready grasp posture must skip intermediate/final arm "
            f"preparation; events={events}"
        )
    if "torso:deep_observation" in events[:detection_index]:
        raise AssertionError(
            "an already-ready grasp torso must not be commanded again before "
            f"detection; events={events}"
        )
    if "gripper:left:100.0" not in events[:detection_index] or (
        "gripper:right:100.0" not in events[:detection_index]
    ):
        raise AssertionError(
            f"ready-pose reuse must still open both grippers; events={events}"
        )


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
    assert_elapsed_result(result, "execute_grasp stage-one retry")
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


def run_grasp_redetect_after_stage_two_failure(
    node: MockMissionSystem,
) -> None:
    node.clear_events()
    node.configure_arm_pose_failures([2])
    goal = ExecuteGrasp.Goal()
    goal.request_id = "mock_grasp_stage_two_redetect"
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
        raise AssertionError("stage-2 failure must use a fresh top-ranked grasp")
    events = node.snapshot()
    failed_index = events.index("move_arm_p:2/2:failed")
    second_detection_index = events.index("detect", events.index("detect") + 1)
    recovery_events = events[failed_index + 1 : second_detection_index]
    if recovery_events.count("move_arm_j") != 1:
        raise AssertionError(
            "stage-2 failure must return directly to the final observation "
            f"joints before re-detection; events={events}"
        )
    if events.count("detect") != 2:
        raise AssertionError(
            f"a failed top candidate must trigger a fresh detection: {events}"
        )
    assert_in_order(
        events,
        [
            "move_arm_p:2/2:failed",
            "move_arm_j",
            "detect",
            "move_arm_p:1/2",
            "move_arm_p:2/2",
        ],
    )


def run_grasp_redetect_after_repeated_top_candidate_failures(
    node: MockMissionSystem,
) -> None:
    node.clear_events()
    node.configure_arm_pose_failures([2, 2])
    goal = ExecuteGrasp.Goal()
    goal.request_id = "mock_grasp_repeated_top_candidate_failures"
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
    detect_indices = [
        index for index, event in enumerate(events) if event == "detect"
    ]
    failed_indices = [
        index for index, event in enumerate(events)
        if event == "move_arm_p:2/2:failed"
    ]
    if len(failed_indices) != 2:
        raise AssertionError(
            f"the first two top candidates must fail: {events}"
        )
    if len(detect_indices) != 3:
        raise AssertionError(
            f"each failed top candidate must cause a fresh detection: {events}"
        )
    for failure_index, next_detection_index in zip(
        failed_indices, detect_indices[1:]
    ):
        recovery_events = events[failure_index + 1 : next_detection_index]
        if recovery_events.count("move_arm_j") != 1:
            raise AssertionError(
                "each failed top candidate must restore observation before "
                f"re-detection; events={events}"
            )


def run_move_chassis(node: MockMissionSystem) -> None:
    node.clear_events()
    if not node.move_chassis_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("/move_chassis action server not available")
    goal = MoveChassis.Goal()
    goal.direction = "right"
    goal_handle = wait_future(
        node.move_chassis_client.send_goal_async(goal), 5.0
    )
    if not goal_handle.accepted:
        raise AssertionError("mock move_chassis goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 5.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    assert_elapsed_result(result, "move_chassis")
    if result.direction != "right":
        raise AssertionError(f"unexpected chassis direction: {result.direction}")
    if not math.isclose(result.speed, 0.1, abs_tol=1e-9):
        raise AssertionError(f"unexpected chassis speed: {result.speed}")
    if not math.isclose(result.duration_sec, 0.2, abs_tol=1e-9):
        raise AssertionError(
            f"unexpected chassis duration: {result.duration_sec}"
        )
    time.sleep(0.1)
    assert_in_order(node.snapshot(), ["chassis:right", "chassis:stopped"])


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
    assert_elapsed_result(result, "execute_place")
    if result.home_completed:
        raise AssertionError("material place must not call home after release")
    if result.torso_reset_command_published:
        raise AssertionError(
            "material place must not reset the torso after release"
        )
    time.sleep(0.1)
    events = node.snapshot()
    assert_in_order(
        events,
        [
            "torso:deep_observation",
            "move_arm_j",
            "gripper:right:100.0",
        ],
    )
    if any(event.startswith("chassis:") for event in events):
        raise AssertionError(
            f"material place must not command the chassis; events={events}"
        )
    release_index = events.index("gripper:right:100.0")
    return_events = events[release_index + 1 :]
    if any(
        event == "move_arm_j" or event.startswith("torso:")
        for event in return_events
    ):
        raise AssertionError(
            "material place must return success immediately after gripper "
            f"release without post-release arm or torso commands; events={events}"
        )
    if "home:dry_run=false" in return_events:
        raise AssertionError(
            f"material place must not call home after release; events={events}"
        )


def run_box_grasp(node: MockMissionSystem) -> None:
    node.clear_events()
    node.configure_pickup_failures(1)
    node.configure_gripper_close_feedback("left", [10.0])
    node.configure_gripper_close_feedback("right", [10.0])
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
    assert_elapsed_result(result, "execute_box_grasp")
    if result.grasp_pose.header.frame_id != "torso_link":
        raise AssertionError("box result must expose the transformed body-frame pose")
    if not result.gripper_command_published:
        raise AssertionError("box grasp must close both grippers after pickup")
    if not result.torso_lift_command_published:
        raise AssertionError(
            "box grasp must publish the Torso1 clearance lift after closing"
        )
    events = node.snapshot()
    post_close_events = events[
        events.index("gripper:right:0.0") + 1 :
    ]
    if post_close_events.count("torso:box_clearance_lift") < 1:
        raise AssertionError(
            "box grasp must move to the configured Torso1 clearance target "
            f"before gripper verification; events={events}"
        )
    unexpected_torso_events = [
        event
        for event in post_close_events
        if event.startswith("torso:")
        and event != "torso:box_clearance_lift"
    ]
    if unexpected_torso_events:
        raise AssertionError(
            "box grasp must not perform additional post-grasp torso motions; "
            f"events={events}"
        )
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
            "foundation_pose",
            "pickup_task:2:dry_run=false",
            "gripper:left:0.0",
            "gripper:right:0.0",
            "torso:box_clearance_lift",
        ],
    )
    if events.count("foundation_pose") != 2:
        raise AssertionError(
            "a failed pickup plan must trigger a fresh FoundationPose estimate; "
            f"events={events}"
        )


def run_box_grasp_rejects_empty_gripper(node: MockMissionSystem) -> None:
    node.clear_events()
    node.configure_pickup_failures(0)
    node.configure_gripper_close_feedback("left", [2.0])
    node.configure_gripper_close_feedback("right", [10.0])
    goal = ExecuteBoxGrasp.Goal()
    goal.request_id = "mock_box_grasp_empty_left"
    goal.target_label = -1
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = False
    goal_handle = wait_future(node.box_grasp_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock empty box-grasp goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if result.success:
        raise AssertionError("a >95% closed box gripper must fail the action")
    if "left gripper closed beyond" not in result.message:
        raise AssertionError(
            f"box empty-grasp failure did not identify the left gripper: "
            f"{result.message}"
        )
    if not result.torso_lift_command_published:
        raise AssertionError(
            "empty-grasp verification must follow the Torso1 clearance lift"
        )
    events = node.snapshot()
    post_close_events = events[
        events.index("gripper:right:0.0") + 1 :
    ]
    if post_close_events.count("torso:box_clearance_lift") < 1:
        raise AssertionError(
            "empty-grasp verification must run after the Torso1 clearance "
            f"target is confirmed; events={events}"
        )
    unexpected_torso_events = [
        event
        for event in post_close_events
        if event.startswith("torso:")
        and event != "torso:box_clearance_lift"
    ]
    if unexpected_torso_events:
        raise AssertionError(
            "empty-grasp verification must not include any additional torso "
            f"motion; events={events}"
        )
    node.configure_gripper_close_feedback("left", [10.0])
    node.configure_gripper_close_feedback("right", [10.0])


def run_box_grasp_recovers_after_pregrasp_failure(
    node: MockMissionSystem,
) -> None:
    node.clear_events()
    node.configure_pickup_failures(1, approach_segment=2)
    goal = ExecuteBoxGrasp.Goal()
    goal.request_id = "mock_box_grasp_pregrasp_recovery"
    goal.target_label = -1
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = False
    goal_handle = wait_future(node.box_grasp_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock box recovery goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)

    events = node.snapshot()
    first_pickup = events.index("pickup_task:1:dry_run=false")
    second_detection = events.index("foundation_pose", first_pickup + 1)
    recovery_events = events[first_pickup + 1 : second_detection]
    if "pickup_feedback:2/2" not in recovery_events:
        raise AssertionError(f"mock did not enter pickup segment 2: {events}")
    if recovery_events.count("move_arm_j") != 1:
        raise AssertionError(
            "a failure after the pre-grasp segment must return directly to "
            f"the final box observation posture; events={events}"
        )
    if "torso:deep_observation" not in recovery_events:
        raise AssertionError(
            f"box recovery must restore the observation torso; events={events}"
        )
    assert_in_order(
        events,
        [
            "pickup_task:1:dry_run=false",
            "pickup_feedback:2/2",
            "move_arm_j",
            "foundation_pose",
            "pickup_task:2:dry_run=false",
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
    if "exhausted fresh-detection attempts" not in result.message:
        raise AssertionError(f"unexpected double-failure message: {result.message}")
    assert_elapsed_result(result, "failed execute_box_grasp")
    events = node.snapshot()
    assert_in_order(
        events,
        [
            "foundation_pose",
            "pickup_task:1:dry_run=true",
            "foundation_pose",
            "pickup_task:2:dry_run=true",
        ],
    )
    if events.count("foundation_pose") != 2:
        raise AssertionError(
            f"both pickup failures must use fresh detections: events={events}"
        )
    if any(event.startswith("gripper:") for event in events):
        raise AssertionError("failed pickup must not command either gripper")
    if any(event.startswith("torso:") for event in events):
        raise AssertionError("failed dry-run pickup must not command the torso")


def run_box_grasp_reuses_ready_observation_on_failure(
    node: MockMissionSystem,
) -> None:
    node.clear_events()
    node.configure_pickup_failures(2)
    node.publish_box_observation_feedback()
    time.sleep(0.1)
    goal = ExecuteBoxGrasp.Goal()
    goal.request_id = "mock_box_grasp_ready_reuse"
    goal.target_label = -1
    goal.arm = "right"
    goal.publish_pose = True
    goal.detection_timeout_sec = 2.0
    goal.dry_run = False
    goal_handle = wait_future(node.box_grasp_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock ready-reuse box goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if result.success:
        raise AssertionError("ready-reuse goal must expose the mocked failures")
    time.sleep(0.1)
    events = node.snapshot()
    first_detection = events.index("foundation_pose")
    if "move_arm_j" in events[:first_detection]:
        raise AssertionError(
            "an already-ready box posture must skip intermediate/final arm "
            f"preparation; events={events}"
        )
    if "torso:deep_observation" in events[:first_detection]:
        raise AssertionError(
            "an already-ready torso must not be commanded again before "
            f"detection; events={events}"
        )
    if "torso:reset" in events:
        raise AssertionError(
            "a box IK/planning failure must retain the observation torso "
            f"instead of publishing all-zero reset; events={events}"
        )


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
    assert_elapsed_result(result, "execute_box_place")
    if not result.gripper_command_published:
        raise AssertionError("box place must open both grippers")
    if not result.ready_completed:
        raise AssertionError("box place must restore the ready arm posture")
    if not result.torso_reset_command_published:
        raise AssertionError("box place must reset the torso after release")
    time.sleep(0.1)
    events = node.snapshot()
    assert_in_order(
        events,
        [
            "torso:deep_observation",
            "gripper:left:100.0",
            "gripper:right:100.0",
            "move_arm_j",
            "torso:reset",
            "go_ready:dry_run=false",
        ],
    )
    if any(event.startswith("chassis:") for event in events):
        raise AssertionError(
            f"box place must not command the chassis; events={events}"
        )


def run_box_stack(node: MockMissionSystem) -> None:
    node.clear_events()
    node.publish_stack_not_ready_feedback()
    if not node.box_stack_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("/execute_box_stack action server not available")
    goal = ExecuteBoxStack.Goal()
    goal.level = 2
    goal_handle = wait_future(node.box_stack_client.send_goal_async(goal), 5.0)
    if not goal_handle.accepted:
        raise AssertionError("mock box stack goal was rejected")
    wrapped_result = wait_future(goal_handle.get_result_async(), 10.0)
    result = wrapped_result.result
    if not result.success:
        raise AssertionError(result.message)
    assert_elapsed_result(result, "execute_box_stack")
    if result.level != 2:
        raise AssertionError(f"unexpected completed stack level: {result.level}")
    expected_torso = [1.407070, -2.598294, -1.091117, 0.0]
    if any(
        not math.isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(
            result.target_torso_positions, expected_torso
        )
    ):
        raise AssertionError(
            "level 2 must use the configured waist target: "
            f"{list(result.target_torso_positions)}"
        )
    if not all(
        (
            result.gripper_closed,
            result.gripper_opened,
            result.retreat_completed,
            result.ready_completed,
        )
    ):
        raise AssertionError(
            "box stack did not report close/place/release/retreat/ready completion"
        )
    time.sleep(0.1)
    events = node.snapshot()
    first_gripper_close = events.index("gripper:left:0.0")
    preparation_events = events[:first_gripper_close]
    if "move_arm_j" not in preparation_events:
        raise AssertionError(
            f"stack must prepare fixed arms before closing; events={events}"
        )
    if "torso:reset" not in preparation_events:
        raise AssertionError(
            f"stack must prepare the upright torso before closing; events={events}"
        )
    assert_in_order(
        events,
        [
            "gripper:left:0.0",
            "gripper:right:0.0",
            "torso:prepare",
            "gripper:left:100.0",
            "gripper:right:100.0",
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
        run_grasp_retries_after_empty_close(node)
        run_grasp_continues_after_ten_empty_closes(node)
        run_grasp_reuses_ready_observation(node)
        run_grasp_stage_one_failure_continues_same_candidate(node)
        run_grasp_redetect_after_stage_two_failure(node)
        run_grasp_redetect_after_repeated_top_candidate_failures(node)
        run_move_chassis(node)
        run_place(node)
        run_box_grasp(node)
        run_box_grasp_rejects_empty_gripper(node)
        run_box_grasp_recovers_after_pregrasp_failure(node)
        run_box_grasp_double_failure(node)
        run_box_grasp_reuses_ready_observation_on_failure(node)
        run_box_place(node)
        run_box_stack(node)
        print("mock chassis and complete grasp/place/stack missions passed")
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
