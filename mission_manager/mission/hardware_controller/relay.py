import time
import threading

from brsd_msgs.action import SerialioCtrl


USB_RELAY_GOAL_TOPIC = "/arto/usb_relay_ctrl_goal"
USB_RELAY_RESULT_TOPIC = "/arto/usb_relay_ctrl_result"


class USBRelay:
    def __init__(self, node):
        self._node = node

        self._publisher = node.create_publisher(
            SerialioCtrl.Goal,
            USB_RELAY_GOAL_TOPIC,
            10,
        )

        self._subscription = node.create_subscription(
            SerialioCtrl.Result,
            USB_RELAY_RESULT_TOPIC,
            self._result_callback,
            10,
        )

        # Verification state.
        self._requested_states = {}
        self._sent_at = None
        self._matching_samples = 0

        # Used to block set_channels() until verification succeeds.
        self._verification_event = threading.Event()

        # Result of the current command.
        self._verification_result = False

        # Protect shared state accessed by set_channels() and
        # _result_callback() running on different executor threads.
        self._lock = threading.Lock()

    def _result_callback(self, message):
        with self._lock:
            if self._sent_at is None:
                return

            ids = list(message.deveice_id_array)
            states = list(message.success_array)

            # Validate the result message.
            if (
                len(ids) != len(states)
                or len(set(ids)) != len(ids)
                or any(channel not in (1, 2, 3, 4) for channel in ids)
            ):
                self._matching_samples = 0
                return

            observed = dict(zip(ids, states))

            self._node.get_logger().info(
                f"Observed relay states: {observed}"
            )

            # Ignore results received during the first 0.5 s.
            if time.monotonic() - self._sent_at < 0.5:
                return

            # Check whether all requested channels have the desired state.
            if all(
                observed.get(channel) == state
                for channel, state in self._requested_states.items()
            ):
                self._matching_samples += 1
            else:
                self._matching_samples = 0

            # Require two consecutive matching samples.
            if self._matching_samples >= 2:
                self._verification_result = True
                self._verification_event.set()

    def set_channels(self, states, timeout=5.0):
        """
        Set relay channels and block until the requested states
        are verified or timeout expires.

        Args:
            states: Dictionary mapping channel -> bool.
                    Example: {1: False, 2: False}
            timeout: Maximum time to wait for verification.

        Returns:
            True  -- requested states were verified.
            False -- verification failed or timed out.
        """
        if not states:
            self._node.get_logger().error("States cannot be empty")
            return False

        for channel, state in states.items():
            if channel not in (1, 2, 3, 4):
                self._node.get_logger().error(f"Invalid relay channel: {channel}")
                return False

            if not isinstance(state, bool):
                self._node.get_logger().error(f"State for channel {channel} must be bool")
                return False

        with self._lock:
            # Prevent another command from overwriting the current one.
            if self._sent_at is not None:
                self._node.get_logger().error("Another USB relay command is being verified.")
                return False

            self._requested_states = dict(states)
            self._sent_at = time.monotonic()
            self._matching_samples = 0
            self._verification_result = False
            self._verification_event.clear()

            goal = SerialioCtrl.Goal()
            goal.deveice_id_array = list(states.keys())
            goal.state_array = list(states.values())
            goal.control_mode = False

            self._publisher.publish(goal)

        self._node.get_logger().info(
            f"Published USB relay command: {states}"
        )

        # Block this thread until:
        #   1. two matching result samples are received, or
        #   2. timeout expires.
        verified = self._verification_event.wait(timeout=timeout)

        with self._lock:
            result = verified and self._verification_result

            # Clear the current command.
            self._requested_states = {}
            self._sent_at = None
            self._matching_samples = 0

        if result:
            self._node.get_logger().info(
                f"USB relay states verified: {states}"
            )
        else:
            self._node.get_logger().error(
                f"USB relay states were not verified: {states}"
            )
        return result

    def set_channel(self, channel, state, timeout=8.0):
        """
        Convenience method for controlling one channel.
        """
        return self.set_channels(
            {channel: state},
            timeout=timeout,
        )

    def turn_on(self, selector):
        if selector == 'left':
            return self.set_channels({1: True, 2: True})

        if selector == 'right':
            return self.set_channels({3: True, 4: True})

        return False

    def turn_off(self, selector):
        if selector == 'left':
            if not self.set_channels({2: False}):
                return False
            time.sleep(2.0)
            return self.set_channels({1: False})
            

        if selector == 'right':
            if not self.set_channels({4: False}):
                return False

            time.sleep(2.0)
            return self.set_channels({3: False})
        
        return False