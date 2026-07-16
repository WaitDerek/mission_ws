import math

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from task_interfaces.action import MoveArmPose
from tf2_ros import Buffer, TransformException, TransformListener


VALID_ARMS = {"left", "right"}


class GraspTargetExecutor(Node):
    """Forward one visualized Mission target to move_arm_p."""

    def __init__(self) -> None:
        super().__init__("grasp_target_executor")
        self.declare_parameter("target_topic", "/mission/arm_link7_target")
        self.declare_parameter("action_name", "/move_arm_p")
        self.declare_parameter("arm", "right")
        self.declare_parameter("expected_frame", "torso_link4")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("execute_confirmed", False)
        self.declare_parameter("server_timeout_sec", 10.0)
        self.declare_parameter("interpolation_steps", 2)
        self.declare_parameter("current_pose_timeout_sec", 2.0)
        self.declare_parameter("segment_pause_sec", 0.5)

        self.exit_code = 1
        self._goal_sent = False
        self._pending_target = None
        self._start_deadline_ns = 0
        self._start_timer = None
        self._next_step_timer = None
        self._interpolated_poses: list[list[float]] = []
        self._step_index = 0
        self.arm = str(self.get_parameter("arm").value).strip().lower()
        self.dry_run = bool(self.get_parameter("dry_run").value)
        execute_confirmed = bool(self.get_parameter("execute_confirmed").value)
        if self.arm not in VALID_ARMS:
            raise ValueError("arm must be 'left' or 'right'")
        if not self.dry_run and not execute_confirmed:
            raise ValueError("real execution requires execute_confirmed:=true")

        action_name = str(self.get_parameter("action_name").value).strip()
        self.action_client = ActionClient(self, MoveArmPose, action_name)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.target_subscription = self.create_subscription(
            PoseStamped,
            str(self.get_parameter("target_topic").value),
            self._target_callback,
            qos,
        )
        mode = "PLAN ONLY" if self.dry_run else "REAL EXECUTION"
        self.get_logger().info(
            f"waiting for Mission arm_link7 target; mode={mode}, arm={self.arm}"
        )

    @staticmethod
    def _pose_array(target: PoseStamped) -> list[float]:
        values = [
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
            target.pose.orientation.x,
            target.pose.orientation.y,
            target.pose.orientation.z,
            target.pose.orientation.w,
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("target pose contains NaN or Inf")
        quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
        if quaternion_norm < 1e-8:
            raise ValueError("target pose quaternion has zero norm")
        values[3:] = [value / quaternion_norm for value in values[3:]]
        return values

    def _target_callback(self, target: PoseStamped) -> None:
        if self._goal_sent:
            return
        expected_frame = (
            str(self.get_parameter("expected_frame").value).strip().lstrip("/")
        )
        actual_frame = target.header.frame_id.strip().lstrip("/")
        if actual_frame != expected_frame:
            self.get_logger().error(
                f"refusing target in '{actual_frame}'; expected '{expected_frame}'"
            )
            self._finish(2)
            return

        try:
            self._pose_array(target)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            self._finish(2)
            return

        self._goal_sent = True
        self._pending_target = target
        timeout = float(self.get_parameter("current_pose_timeout_sec").value)
        if timeout <= 0.0:
            self.get_logger().error("current_pose_timeout_sec must be positive")
            self._finish(2)
            return
        self._start_deadline_ns = self.get_clock().now().nanoseconds + int(
            timeout * 1e9
        )
        self._start_timer = self.create_timer(0.05, self._try_start_sequence)

    @staticmethod
    def _slerp(
        start: list[float], target: list[float], fraction: float
    ) -> list[float]:
        dot = sum(a * b for a, b in zip(start, target))
        if dot < 0.0:
            target = [-value for value in target]
            dot = -dot
        dot = max(-1.0, min(1.0, dot))
        if dot > 0.9995:
            result = [
                a + fraction * (b - a) for a, b in zip(start, target)
            ]
        else:
            theta = math.acos(dot)
            scale = math.sin(theta)
            start_scale = math.sin((1.0 - fraction) * theta) / scale
            target_scale = math.sin(fraction * theta) / scale
            result = [
                start_scale * a + target_scale * b
                for a, b in zip(start, target)
            ]
        norm = math.sqrt(sum(value * value for value in result))
        return [value / norm for value in result]

    @classmethod
    def _interpolate_pose(
        cls, start: list[float], target: list[float], fraction: float
    ) -> list[float]:
        position = [
            start[index] + fraction * (target[index] - start[index])
            for index in range(3)
        ]
        return position + cls._slerp(start[3:], target[3:], fraction)

    def _try_start_sequence(self) -> None:
        if self._pending_target is None:
            return
        expected_frame = (
            str(self.get_parameter("expected_frame").value).strip().lstrip("/")
        )
        target_link = f"{self.arm}_arm_link7"
        try:
            transform = self.tf_buffer.lookup_transform(
                expected_frame, target_link, rclpy.time.Time()
            )
        except TransformException as exc:
            if self.get_clock().now().nanoseconds < self._start_deadline_ns:
                return
            self.get_logger().error(
                f"current pose TF {expected_frame} <- {target_link} failed: {exc}"
            )
            self._finish(2)
            return

        if self._start_timer is not None:
            self._start_timer.cancel()
        target = self._pose_array(self._pending_target)
        start = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
        steps = int(self.get_parameter("interpolation_steps").value)
        if steps < 1:
            self.get_logger().error("interpolation_steps must be at least one")
            self._finish(2)
            return
        self._interpolated_poses = [
            self._interpolate_pose(start, target, step / steps)
            for step in range(1, steps + 1)
        ]
        timeout = float(self.get_parameter("server_timeout_sec").value)
        if timeout <= 0.0 or not self.action_client.wait_for_server(
            timeout_sec=timeout
        ):
            self.get_logger().error("move_arm_p action server is unavailable")
            self._finish(3)
            return
        if self.dry_run and steps > 1:
            self.get_logger().warning(
                "dry-run plans every segment from the unchanged measured state; "
                "only real execution advances the robot between segments"
            )
        self._step_index = 0
        self._send_step()

    def _send_step(self) -> None:
        pose = self._interpolated_poses[self._step_index]
        goal = MoveArmPose.Goal()
        goal.left_pose = pose if self.arm == "left" else []
        goal.right_pose = pose if self.arm == "right" else []
        goal.dry_run = self.dry_run
        self.get_logger().info(
            f"sending step {self._step_index + 1}/"
            f"{len(self._interpolated_poses)} {self.arm} target: {pose}"
        )
        future = self.action_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback
        )
        future.add_done_callback(self._goal_response_callback)

    def _feedback_callback(self, feedback_message) -> None:
        feedback = feedback_message.feedback
        self.get_logger().info(
            f"move_arm_p {feedback.stage} {feedback.progress:.0%}: "
            f"{feedback.detail}"
        )

    def _goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("move_arm_p goal was rejected")
            self._finish(4)
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future) -> None:
        wrapped = future.result()
        result = wrapped.result
        succeeded = wrapped.status == GoalStatus.STATUS_SUCCEEDED and result.success
        message = (
            f"move_arm_p step {self._step_index + 1}/"
            f"{len(self._interpolated_poses)} result: success={result.success}, "
            f"error_code={result.error_code}, message='{result.message}'"
        )
        if succeeded:
            self.get_logger().info(message)
        else:
            self.get_logger().error(message)
        if not succeeded:
            self._finish(5)
            return
        self._step_index += 1
        if self._step_index >= len(self._interpolated_poses):
            self._finish(0)
            return
        pause = float(self.get_parameter("segment_pause_sec").value)
        if pause < 0.0:
            self.get_logger().error("segment_pause_sec must be non-negative")
            self._finish(2)
            return
        if pause == 0.0:
            self._send_step()
            return
        self._next_step_timer = self.create_timer(pause, self._send_next_step)

    def _send_next_step(self) -> None:
        if self._next_step_timer is not None:
            self._next_step_timer.cancel()
            self._next_step_timer = None
        self._send_step()

    def _finish(self, exit_code: int) -> None:
        self.exit_code = exit_code
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    exit_code = 1
    try:
        node = GraspTargetExecutor()
        rclpy.spin(node)
        exit_code = node.exit_code
    except (KeyboardInterrupt, ValueError) as exc:
        if node is not None and str(exc):
            node.get_logger().error(str(exc))
        elif str(exc):
            print(f"grasp_target_executor: {exc}")
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
