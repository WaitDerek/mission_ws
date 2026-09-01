import json
import threading
import time
import unittest
from types import SimpleNamespace

from mission_runtime.taskflow.model import NavigationRequest
from mission_runtime.taskflow.mqtt_navigation import (
    MqttNavigationGateway,
    NavigationPoint,
    parse_navigation_points_json,
)


class _PublishInfo:
    rc = 0

    def wait_for_publish(self, timeout=None):
        del timeout
        return None


class _FakeClient:
    def __init__(self, *, connect=True):
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None
        self.connect_on_start = connect
        self.published = []
        self.subscriptions = []
        self.disconnected = False
        self.loop_stopped = False

    def connect_async(self, host, port, keepalive):
        self.connection = (host, port, keepalive)
        return 0

    def loop_start(self):
        if self.connect_on_start:
            self.on_connect(self, None, {}, 0)

    def subscribe(self, topic, qos):
        self.subscriptions.append((topic, qos))
        return 0, 1

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))
        return _PublishInfo()

    def emit(self, payload, *, topic="mission/navigation/result", retain=False):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.on_message(
            self,
            None,
            SimpleNamespace(topic=topic, payload=payload, retain=retain),
        )

    def drop_connection(self):
        self.on_disconnect(self, None, 1)

    def disconnect(self):
        self.disconnected = True

    def loop_stop(self):
        self.loop_stopped = True


def _gateway(client, **overrides):
    values = {
        "host": "broker",
        "connect_timeout_sec": 0.1,
        "navigation_timeout_sec": 0.2,
        "point_poses": {
            str(point_id): NavigationPoint(
                x=float(point_id),
                y=-float(point_id),
                yaw=float(point_id) / 10.0,
            )
            for point_id in range(1, 17)
        },
        "client": client,
    }
    values.update(overrides)
    return MqttNavigationGateway(**values)


def _navigate_in_thread(gateway, point_id):
    results = []
    thread = threading.Thread(
        target=lambda: results.append(
            gateway.navigate(
                NavigationRequest("workflow", "step", point_id),
                lambda: False,
            )
        )
    )
    thread.start()
    return thread, results


class TestMqttNavigationGateway(unittest.TestCase):
    def test_navigation_request_and_platform_result_protocol(self):
        client = _FakeClient()
        gateway = _gateway(client)
        thread, results = _navigate_in_thread(gateway, "5")
        self._wait_for_publish(client)

        client.emit(
            json.dumps(
                {"robot_id": "6", "success": True, "message": "arrived"}
            )
        )
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(results[0].success)
        self.assertEqual(
            client.published,
            [
                (
                    "mission/navigation/request",
                    '{"robot_id":"6","point_id":5,"frame_id":"map","x":5.0,"y":-5.0,"yaw":0.5}',
                    1,
                    False,
                )
            ],
        )
        self.assertEqual(client.subscriptions, [("mission/navigation/result", 1)])
        gateway.close()

    def test_json_failure_is_returned_to_workflow(self):
        client = _FakeClient()
        gateway = _gateway(client)
        thread, results = _navigate_in_thread(gateway, "8")
        self._wait_for_publish(client)

        client.emit(
            json.dumps(
                {
                    "robot_id": "6",
                    "success": False,
                    "message": "blocked",
                }
            )
        )
        thread.join(timeout=1.0)

        self.assertFalse(results[0].success)
        self.assertEqual(results[0].status, "failed")
        self.assertEqual(results[0].message, "blocked")
        gateway.close()

    def test_legacy_plain_and_point_id_results_are_rejected(self):
        client = _FakeClient()
        gateway = _gateway(client, navigation_timeout_sec=0.05)
        thread, results = _navigate_in_thread(gateway, "4")
        self._wait_for_publish(client)

        client.emit("4")
        client.emit(json.dumps({"id": "4", "success": True}))
        client.emit(json.dumps({"point_id": "4", "success": True}))
        thread.join(timeout=1.0)

        self.assertEqual(results[0].status, "timeout")
        gateway.close()

    def test_mismatched_and_retained_results_are_ignored(self):
        client = _FakeClient()
        gateway = _gateway(client, navigation_timeout_sec=0.05)
        thread, results = _navigate_in_thread(gateway, "16")
        self._wait_for_publish(client)

        client.emit(json.dumps({"robot_id": "7", "success": True}))
        client.emit(
            json.dumps({"robot_id": "6", "success": True}),
            retain=True,
        )
        thread.join(timeout=1.0)

        self.assertEqual(results[0].status, "timeout")
        gateway.close()

    def test_cancel_releases_waiting_navigation(self):
        client = _FakeClient()
        gateway = _gateway(client)
        thread, results = _navigate_in_thread(gateway, "1")
        self._wait_for_publish(client)

        gateway.cancel_active()
        thread.join(timeout=1.0)

        self.assertEqual(results[0].status, "canceled")
        gateway.close()
        self.assertTrue(client.disconnected)
        self.assertTrue(client.loop_stopped)

    def test_disconnect_fails_active_navigation_immediately(self):
        client = _FakeClient()
        gateway = _gateway(client)
        thread, results = _navigate_in_thread(gateway, "3")
        self._wait_for_publish(client)

        client.drop_connection()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].status, "unavailable")
        self.assertIn("disconnected", results[0].message)
        gateway.close()

    def test_unconnected_broker_fails_closed(self):
        gateway = _gateway(_FakeClient(connect=False), connect_timeout_sec=0.03)

        result = gateway.navigate(
            NavigationRequest("workflow", "step", "1"), lambda: False
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, "unavailable")
        gateway.close()

    def test_point_id_is_limited_to_configured_map(self):
        gateway = _gateway(_FakeClient())

        result = gateway.navigate(
            NavigationRequest("workflow", "step", "17"), lambda: False
        )

        self.assertEqual(result.status, "invalid")
        gateway.close()

    def test_missing_point_coordinates_fail_without_publish(self):
        client = _FakeClient()
        gateway = _gateway(client, point_poses={})

        result = gateway.navigate(
            NavigationRequest("workflow", "step", "5"), lambda: False
        )

        self.assertEqual(result.status, "invalid")
        self.assertIn("no configured coordinates", result.message)
        self.assertEqual(client.published, [])
        gateway.close()

    def test_navigation_points_json_is_validated(self):
        points = parse_navigation_points_json(
            '{"1":{"x":1.25,"y":-2.5,"yaw":1.57}}'
        )

        self.assertEqual(points["1"], NavigationPoint(1.25, -2.5, 1.57))
        with self.assertRaisesRegex(ValueError, "missing yaw"):
            parse_navigation_points_json('{"1":{"x":1.0,"y":2.0}}')
        with self.assertRaisesRegex(ValueError, "must be finite"):
            parse_navigation_points_json(
                '{"1":{"x":1e999,"y":2.0,"yaw":0.0}}'
            )

    def _wait_for_publish(self, client):
        deadline = time.monotonic() + 1.0
        while not client.published:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)


if __name__ == "__main__":
    unittest.main()
