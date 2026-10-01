"""No-motion checks for the ROS arm MoveJ command transport."""

import json
import threading
import time
import unittest
from types import SimpleNamespace

from mission_runtime.realman_sdk_common import RealManSdkCanceled, RealManSdkError
from mission_runtime.ros_arm_movej import RosArmMoveJ


class _Future:
    def __init__(self, accepted=True):
        self._accepted = accepted

    def done(self):
        return True

    def result(self):
        return SimpleNamespace(data=json.dumps({"receive_state": self._accepted}))


class _Client:
    def __init__(self, accepted=True):
        self.requests = []
        self.accepted = accepted

    def service_is_ready(self):
        return True

    def call_async(self, request):
        self.requests.append(json.loads(request.data))
        return _Future(self.accepted)


class _Node:
    def __init__(self):
        self.callbacks = {}

    def create_subscription(self, _type, topic, callback, _qos, callback_group=None):
        del callback_group
        self.callbacks[topic] = callback
        return callback


def _state(degrees):
    return SimpleNamespace(
        joint_state=SimpleNamespace(
            name=[f"joint{index}" for index in range(1, 8)],
            position=[value * 3.141592653589793 / 180.0 for value in degrees],
            velocity=[0.0] * 7,
        )
    )


class RosArmMoveJTests(unittest.TestCase):
    def test_dual_movej_preserves_speed_and_waits_for_both_arms(self):
        node = _Node()
        client = _Client()
        transport = RosArmMoveJ(node, client=client)
        left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        right = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]

        def publish():
            time.sleep(0.05)
            for _ in range(3):
                node.callbacks["/mcap/slave_arm_left"](_state(left))
                node.callbacks["/mcap/slave_arm_right"](_state(right))
                time.sleep(0.04)

        thread = threading.Thread(target=publish)
        thread.start()
        try:
            result = transport.execute(
                {"left": left, "right": right}, 15, 0, 0, timeout_sec=1.0
            )
        finally:
            thread.join()
        self.assertIn("completed", result)
        self.assertEqual([item["device"] for item in client.requests], [0, 1])
        self.assertEqual(client.requests[0]["payload"]["joint"], [1000, 2000, 3000, 4000, 5000, 6000, 7000])
        self.assertEqual(client.requests[1]["payload"]["joint"], [-1000, -2000, -3000, -4000, -5000, -6000, -7000])
        self.assertTrue(all(item["payload"]["v"] == 15 for item in client.requests))

    def test_service_rejection_stops_commanded_arm(self):
        stops = []
        client = _Client(accepted=False)
        transport = RosArmMoveJ(
            _Node(), client=client, stop_arm=stops.append
        )
        with self.assertRaisesRegex(RealManSdkError, "rejected"):
            transport.execute({"left": [0.0] * 7}, 10, 0, 0, timeout_sec=0.5)
        self.assertEqual(stops, ["left"])

    def test_cancel_before_submission_sends_nothing(self):
        client = _Client()
        transport = RosArmMoveJ(_Node(), client=client)
        with self.assertRaises(RealManSdkCanceled):
            transport.execute(
                {"right": [0.0] * 7}, 10, 0, 0,
                cancel_requested=lambda: True, timeout_sec=0.5,
            )
        self.assertEqual(client.requests, [])


if __name__ == "__main__":
    unittest.main()
