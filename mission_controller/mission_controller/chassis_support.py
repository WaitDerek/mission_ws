import time

import rclpy
from geometry_msgs.msg import TwistStamped

from .common import CHASSIS_DIRECTIONS


class ChassisSupportMixin:
    """Chassis command construction and timed movement."""

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
