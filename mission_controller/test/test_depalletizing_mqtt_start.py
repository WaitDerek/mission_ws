import json
import unittest
from types import SimpleNamespace

from mission_runtime.taskflow.mqtt_start import MqttWorkflowStartBridge


class _PublishInfo:
    rc = 0


class _FakeClient:
    def __init__(self):
        self.on_connect = None
        self.on_message = None
        self.subscriptions = []
        self.published = []
        self.closed = False

    def connect_async(self, host, port, keepalive):
        self.connection = (host, port, keepalive)
        return 0

    def loop_start(self):
        self.on_connect(self, None, {}, 0)

    def subscribe(self, topic, qos):
        self.subscriptions.append((topic, qos))
        return 0, 1

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))
        return _PublishInfo()

    def emit(self, payload, *, retain=False, topic="mission/workflow/start"):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.on_message(
            self,
            None,
            SimpleNamespace(topic=topic, payload=payload, retain=retain),
        )

    def disconnect(self):
        self.closed = True

    def loop_stop(self):
        return None


def _bridge(client, requests):
    return MqttWorkflowStartBridge(
        host="broker",
        start_topic="mission/workflow/start",
        status_topic="mission/workflow/status",
        on_start=requests.append,
        client=client,
    )


class TestMqttWorkflowStartBridge(unittest.TestCase):
    def test_plain_start_is_rejected(self):
        client = _FakeClient()
        requests = []
        bridge = _bridge(client, requests)

        client.emit("start")

        self.assertEqual(requests, [])
        status = json.loads(client.published[-1][1])
        self.assertEqual(status["event"], "rejected")
        self.assertIn("JSON object", status["message"])
        self.assertEqual(client.subscriptions, [("mission/workflow/start", 1)])
        bridge.close()

    def test_json_start_preserves_platform_request_id(self):
        client = _FakeClient()
        requests = []
        bridge = _bridge(client, requests)

        client.emit(
            json.dumps(
                {"robot_id": "6", "start": True, "request_id": "platform-7"}
            )
        )

        self.assertEqual(requests[0].request_id, "platform-7")
        self.assertEqual(requests[0].robot_id, "6")
        bridge.close()

    def test_json_start_requires_robot_id(self):
        client = _FakeClient()
        requests = []
        bridge = _bridge(client, requests)

        client.emit(json.dumps({"start": True}))

        self.assertEqual(requests, [])
        status = json.loads(client.published[-1][1])
        self.assertIn("requires robot_id", status["message"])
        bridge.close()

    def test_other_robot_id_is_preserved_for_node_validation(self):
        client = _FakeClient()
        requests = []
        bridge = _bridge(client, requests)

        client.emit(json.dumps({"robot_id": "7", "start": True}))

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].robot_id, "7")
        bridge.close()

    def test_json_robot_id_must_be_scalar(self):
        client = _FakeClient()
        requests = []
        bridge = _bridge(client, requests)

        client.emit(json.dumps({"robot_id": {"value": 0}, "start": True}))

        self.assertEqual(requests, [])
        status = json.loads(client.published[-1][1])
        self.assertIn("string or integer", status["message"])
        bridge.close()

    def test_legacy_id_field_is_not_used(self):
        client = _FakeClient()
        requests = []
        bridge = _bridge(client, requests)

        client.emit(json.dumps({"id": 0, "start": True}))

        self.assertEqual(requests, [])
        status = json.loads(client.published[-1][1])
        self.assertIn("requires robot_id", status["message"])
        bridge.close()

    def test_invalid_start_is_rejected_on_status_topic(self):
        client = _FakeClient()
        bridge = _bridge(client, [])

        client.emit(json.dumps({"start": False}))

        topic, payload, qos, retained = client.published[-1]
        status = json.loads(payload)
        self.assertEqual(topic, "mission/workflow/status")
        self.assertEqual(status["event"], "rejected")
        self.assertIn("start=true", status["message"])
        self.assertEqual(qos, 1)
        self.assertFalse(retained)
        bridge.close()

    def test_retained_start_is_ignored(self):
        client = _FakeClient()
        requests = []
        bridge = _bridge(client, requests)

        client.emit("start", retain=True)

        self.assertEqual(requests, [])
        bridge.close()

    def test_status_payload_is_compact_json(self):
        client = _FakeClient()
        bridge = _bridge(client, [])

        published = bridge.publish_status(
            {"event": "result", "request_id": "p1", "success": True}
        )

        self.assertTrue(published)
        self.assertEqual(
            json.loads(client.published[-1][1]),
            {"event": "result", "request_id": "p1", "success": True},
        )
        bridge.close()
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
