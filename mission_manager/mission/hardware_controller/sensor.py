import threading
from brsd_msgs.msg import ForceData
from rclpy.callback_groups import ReentrantCallbackGroup

FORCE_TORQUE_TOPIC = "/force_torque/data"


class ForceTorque:

    def __init__(self, node):
        self._node = node
        self._condition = threading.Condition()
        self._requested = False
        self._latest_force = None

        # Allow the sensor callback to run while another callback
        # from the same node is waiting in get_next_force().
        self._callback_group = ReentrantCallbackGroup()

        self.force_sub = node.create_subscription(
            ForceData,
            "/force_torque/data",
            self.force_callback,
            10,
            callback_group=self._callback_group,
        )

    def force_callback(self, msg):
        with self._condition:
            if not self._requested:
                return

            self._latest_force = (msg.fx, msg.fy, msg.fz)
            self._requested = False
            self._condition.notify()

    def get_next_force(self, timeout_sec=3.0):
        with self._condition:
            self._requested = True
            self._latest_force = None

            if not self._condition.wait_for(
                lambda: self._latest_force is not None,
                timeout=timeout_sec,
            ):
                self._requested = False
                return None

            return self._latest_force