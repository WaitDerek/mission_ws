import rclpy
import threading
from rclpy.node import Node

from brsd_msgs.msg import ForceData

FORCE_TORQUE_TOPIC = "/force_torque/data"
DATA_TIMEOUT_SEC = 2


class ForceTorque(Node):

    def __init__(self):
        super().__init__('force_torque_node')

        self._lock = threading.Lock()

        self.fx = None
        self.fy = None
        self.fz = None

        self.mx = None
        self.my = None
        self.mz = None

        self.force_sub = self.create_subscription(
            ForceData,
            FORCE_TORQUE_TOPIC,
            self.force_callback,
            10
        )
        
        # Timestamp of the latest received sensor message, in nanoseconds.
        self._stamp_ns = None

        # Create a dedicated executor for this node
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self)

        # Continuously process sensor callbacks in the background
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._executor.spin,
            daemon=True
        )
        self._thread.start()

        self.get_logger().info(
            'ForceTorque sensor started'
        )

    def force_callback(self, msg: ForceData):
        stamp_ns = (
            msg.header.stamp.sec * 1_000_000_000
            + msg.header.stamp.nanosec
        )

        with self._condition:
            self.fx = msg.fx
            self.fy = msg.fy
            self.fz = msg.fz

            self.mx = msg.mx
            self.my = msg.my
            self.mz = msg.mz

            self._stamp_ns = stamp_ns

            self._condition.notify_all()
    
    def get_next_force(self, timeout_sec=DATA_TIMEOUT_SEC):
        """
        Wait for the next sensor message whose timestamp differs
        from the timestamp observed when this function was called.

        Returns:
            (fx, fy, fz), or None if no new sample arrives before timeout.
        """
        with self._condition:
            previous_stamp_ns = self._stamp_ns

            if not self._condition.wait_for(
                lambda: (
                    self._stamp_ns is not None
                    and self._stamp_ns != previous_stamp_ns
                ),
                timeout=timeout_sec
            ):
                return None

            return self.fx, self.fy, self.fz
    
    def shutdown(self):
        self._executor.shutdown()

        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self.destroy_node()
    