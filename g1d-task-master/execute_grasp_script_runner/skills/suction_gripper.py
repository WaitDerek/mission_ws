import time
import rclpy
from rclpy.node import Node

from brsd_msgs.action import SerialioCtrl


class SuctionGripper(Node):
    """
    Control a 4-channel USB relay.

    Channels:
        1, 2, 3, 4

    ROS topics:
        Goal:   /arto/usb_relay_ctrl_goal
        Result: /arto/usb_relay_ctrl_result
    """
    def __init__(self):
        super().__init__("suction_gripper")

        goal_topic = "/arto/usb_relay_ctrl_goal"
        result_topic = "/arto/usb_relay_ctrl_result"

        self.goal_topic = goal_topic
        self.result_topic = result_topic

        # The relay has four channels.
        self.num_channels = 4

        # Keep track of the expected state of each channel.
        self.channel_states = [False] * self.num_channels

        # Verification flags.
        self._command_sent = False
        self._verified = False
        self._expected_channel = None
        self._expected_state = None

        # Publisher
        self.publisher = self.create_publisher(
            SerialioCtrl.Goal,
            self.goal_topic,
            10,
        )

        # Subscriber
        self.subscription = self.create_subscription(
            SerialioCtrl.Result,
            self.result_topic,
            self._result_callback,
            10,
        )

        self.get_logger().info("SuctionGripper initialized with 4 relay channels.")

    def _result_callback(self, message):
        """
        Receive relay state feedback.
        """
        if not self._command_sent:
            return

        states = list(message.success_array)
        device_ids = list(message.deveice_id_array)

        self.get_logger().info(
            f"Relay result: devices={device_ids}, states={states}"
        )

        channel = self._expected_channel
        expected_state = self._expected_state

        if channel is None:
            return

        # Channel numbers are 1~4, while Python arrays are 0~3.
        index = channel - 1

        if len(states) > index:
            actual_state = states[index]

            if actual_state == expected_state:
                self._verified = True
                self.channel_states[index] = actual_state
    
    def _wait_for_driver(self, timeout=3.0):
        """
        Wait until the relay driver subscribes to the goal topic.
        """

        deadline = time.monotonic() + timeout

        while (
            self.publisher.get_subscription_count() == 0
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.publisher.get_subscription_count() == 0:
            raise RuntimeError(
                f"No relay driver subscribes to {self.goal_topic}"
            )

    def set_channel(
        self,
        channel: int,
        enabled: bool,
        verify: bool = True,
        timeout: float = 5.0,
    ):
        """
        Set one relay channel ON or OFF.

        Args:
            channel:
                Relay channel, 1~4.

            enabled:
                True  -> ON
                False -> OFF

            verify:
                Wait for the relay result and verify the state.

            timeout:
                Verification timeout in seconds.

        Returns:
            True if successful.
        """

        # Validate channel.
        if channel not in range(1, self.num_channels + 1):
            raise ValueError(
                f"Channel must be between 1 and {self.num_channels}, "
                f"got {channel}"
            )

        self._wait_for_driver()

        # Reset verification state.
        self._command_sent = False
        self._verified = False
        self._expected_channel = channel
        self._expected_state = enabled

        # Construct the relay command.
        goal = SerialioCtrl.Goal()

        goal.deveice_id_array = [channel]
        goal.state_array = [enabled]
        goal.control_mode = False

        # Publish.
        self._command_sent = True
        self.publisher.publish(goal)

        state_string = "ON" if enabled else "OFF"

        self.get_logger().info(
            f"Set relay channel {channel} -> {state_string}"
        )

        # No verification requested.
        if not verify:
            self.channel_states[channel - 1] = enabled
            return True

        # Wait for verification.
        deadline = time.monotonic() + timeout

        while (
            not self._verified
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self._verified:
            raise RuntimeError(
                f"Channel {channel} state was not verified "
                f"within {timeout:.1f} seconds"
            )

        self.get_logger().info(
            f"Verified relay channel {channel} -> {state_string}"
        )

        return True

    def get_channel_state(self, channel: int) -> bool:
            """
            Return the last verified state of a channel.
            """
    
            if channel not in range(1, self.num_channels + 1):
                raise ValueError(
                    f"Channel must be between 1 and {self.num_channels}"
                )
    
            return self.channel_states[channel - 1]

    def turn_on(self, selector):
        """
        Turn on left/right suction gripper.

        Args:
            selector: 'left'/'right'

        Returns:
            True if successful.
        """
        if selector == 'left':
            res1 = self.set_channel(1, True)
            time.sleep(2.0)
            res2 = self.set_channel(2, True)
            return res1 and res2

        if selector == 'right':
            res1 = self.set_channel(3, False)
            time.sleep(2.0)
            res2 = self.set_channel(4, True)
            return res1 and res2

        return False

    def turn_off(self, selector):
        """
        Turn off left/right suction gripper.

        Args:
            selector: 'left'/'right'

        Returns:
            True if successful.
        """
        if selector == 'left':
            res1 = self.set_channel(2, False)
            time.sleep(2.0)
            res2 = self.set_channel(1, False)
            return res1 and res2
            

        if selector == 'right':
            res1 = self.set_channel(4, False)
            time.sleep(2.0)
            res2 = self.set_channel(3, True) 
            return res1 and res2

        return False
