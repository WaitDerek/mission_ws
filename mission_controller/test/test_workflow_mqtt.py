import json
import threading
import time
from types import SimpleNamespace

import pytest

from mission_controller.taskflow.model import NavigationRequest
from mission_controller.taskflow.mqtt_navigation import (
    MqttNavigationGateway,
    NavigationPoint,
    parse_navigation_points_json,
    require_all_navigation_points,
)
from mission_controller.taskflow.mqtt_start import (
    MqttStartRequest,
    MqttWorkflowStartBridge,
)
from mission_controller.taskflow.mqtt_support import mqtt_call_succeeded
from mission_controller.taskflow.mqtt_trigger import WorkflowMqttTriggerMixin


class _PublishInfo:
    rc = 0


class _MqttClient:
    def __init__(self):
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None
        self.published = []
        self.subscribed = []

    def connect_async(self, *_args):
        return 0

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))
        return 0, 1

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))
        return _PublishInfo()


def test_navigation_points_accept_only_ids_1_through_4():
    points = parse_navigation_points_json(
        '{"1":{"x":1,"y":2,"yaw":0.5},"4":{"x":4,"y":5,"yaw":1}}'
    )
    assert points["1"] == NavigationPoint(1.0, 2.0, 0.5)
    with pytest.raises(ValueError, match="1..4"):
        parse_navigation_points_json('{"5":{"x":0,"y":0,"yaw":0}}')
    with pytest.raises(ValueError, match="missing ids: 2, 3"):
        require_all_navigation_points(points)


def test_navigation_publishes_pos_json_and_waits_for_matching_result():
    client = _MqttClient()
    gateway = MqttNavigationGateway(
        host="localhost",
        point_poses={"1": NavigationPoint(1.2, 3.4, 0.5)},
        client=client,
        connect_timeout_sec=0.5,
        navigation_timeout_sec=0.5,
    )
    gateway._on_connect(client, None, None, 0)
    holder = {}

    def navigate():
        holder["result"] = gateway.navigate(
            NavigationRequest("wf", "step", "1"), lambda: False
        )

    thread = threading.Thread(target=navigate)
    thread.start()
    deadline = time.monotonic() + 0.5
    while not client.published and time.monotonic() < deadline:
        time.sleep(0.01)
    payload = json.loads(client.published[0][1])
    assert payload == {
        "id": 1,
        "frame_id": "map",
        "pos": [1.2, 3.4, 0.5],
    }
    gateway._on_message(
        client,
        None,
        SimpleNamespace(
            topic="mission/navigation/result",
            retain=False,
            payload=b"2",
        ),
    )
    time.sleep(0.05)
    assert thread.is_alive()
    gateway._on_message(
        client,
        None,
        SimpleNamespace(
            topic="mission/navigation/result",
            retain=False,
            payload=b'{"id":1,"success":true,"message":"arrived"}',
        ),
    )
    thread.join(timeout=1.0)
    assert holder["result"].success
    assert holder["result"].message == "arrived"
    gateway.close()


def test_mqtt_start_requires_robot_id_and_start_true():
    decoded = MqttWorkflowStartBridge._decode_start(
        b'{"robot_id":"g1d-01","start":true,"request_id":"platform-1"}'
    )
    assert decoded == MqttStartRequest("platform-1", "g1d-01")
    with pytest.raises(ValueError, match="JSON object"):
        MqttWorkflowStartBridge._decode_start(b"start")
    with pytest.raises(ValueError, match="start=true"):
        MqttWorkflowStartBridge._decode_start(
            b'{"robot_id":"g1d-01","start":false}'
        )
    with pytest.raises(ValueError, match="robot id"):
        MqttWorkflowStartBridge._decode_start(b'{"start":true}')
    with pytest.raises(ValueError, match="string or integer"):
        MqttWorkflowStartBridge._decode_start(
            b'{"robot_id":true,"start":true}'
        )


def test_navigation_result_accepts_realbot_plain_or_json_protocol():
    assert MqttNavigationGateway._decode_result(b"1") == (
        "1",
        True,
        "platform reported arrival",
    )
    with pytest.raises(ValueError, match="success must be boolean"):
        MqttNavigationGateway._decode_result(b'{"id":1}')


def test_mqtt_start_bridge_dispatches_and_publishes_json_status():
    client = _MqttClient()
    requests = []
    bridge = MqttWorkflowStartBridge(
        host="localhost",
        start_topic="mission/workflow/start",
        status_topic="mission/workflow/status",
        on_start=requests.append,
        client=client,
    )
    bridge._on_message(
        client,
        None,
        SimpleNamespace(
            topic="mission/workflow/start",
            retain=False,
            payload=b'{"robot_id":"g1d-01","start":true,"request_id":"r1"}',
        ),
    )
    assert requests == [MqttStartRequest("r1", "g1d-01")]
    assert bridge.publish_status(
        {"event": "accepted", "robot_id": "g1d-01"}
    )
    assert json.loads(client.published[-1][1]) == {
        "event": "accepted",
        "robot_id": "g1d-01",
    }
    bridge.close()


class _TriggerHarness(WorkflowMqttTriggerMixin):
    def __init__(self):
        self.statuses = []
        self._mqtt_start_bridge = SimpleNamespace(
            publish_status=lambda values: self.statuses.append(values) or True
        )
        self._mqtt_start_lock = threading.Lock()
        self._mqtt_start_busy = False
        self._mqtt_pending_start = None

    @staticmethod
    def _float(_name):
        return 5.0

    @staticmethod
    def _string(name):
        assert name == "robot_id"
        return "g1d-01"


def test_workflow_trigger_accepts_only_configured_robot_id():
    harness = _TriggerHarness()
    harness._queue_mqtt_start(MqttStartRequest("bad", "other-robot"))
    assert harness.statuses[-1]["event"] == "rejected"
    assert harness.statuses[-1]["robot_id"] == "other-robot"
    assert not harness._mqtt_start_busy

    harness._queue_mqtt_start(MqttStartRequest("good", "g1d-01"))
    assert harness._mqtt_start_busy
    assert harness._mqtt_pending_start[0] == MqttStartRequest("good", "g1d-01")
    assert harness.statuses[-1] == {
        "event": "received",
        "robot_id": "g1d-01",
        "request_id": "good",
        "message": "MQTT start request queued",
    }


def test_paho_async_none_return_is_treated_as_accepted():
    assert mqtt_call_succeeded(None)
    assert mqtt_call_succeeded(0)
    assert not mqtt_call_succeeded(1)
