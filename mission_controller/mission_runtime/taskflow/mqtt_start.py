"""MQTT ingress and status egress for starting the top-level workflow Action."""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
from collections.abc import Callable, Mapping
from typing import Any

from .mqtt_support import create_paho_client, mqtt_reason_is_failure


@dataclass(frozen=True)
class MqttStartRequest:
    request_id: str = ""
    robot_id: str = "6"


class MqttWorkflowStartBridge:
    """Translate MQTT start messages into callbacks owned by the ROS node."""

    def __init__(
        self,
        *,
        host: str,
        start_topic: str,
        status_topic: str,
        on_start: Callable[[MqttStartRequest], None],
        port: int = 1883,
        client_id: str = "",
        qos: int = 1,
        keepalive_sec: int = 60,
        client: Any | None = None,
    ) -> None:
        self._host = str(host).strip()
        self._port = int(port)
        self._start_topic = str(start_topic).strip()
        self._status_topic = str(status_topic).strip()
        self._qos = int(qos)
        self._keepalive_sec = int(keepalive_sec)
        self._on_start_callback = on_start
        self._closed = False
        self._lock = threading.Lock()
        self._mqtt_success = 0
        self._validate_configuration()

        if client is None:
            self._client, self._mqtt_success = create_paho_client(client_id)
        else:
            self._client = client
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        connect_result = self._client.connect_async(
            self._host,
            self._port,
            self._keepalive_sec,
        )
        if int(connect_result) != self._mqtt_success:
            raise RuntimeError(
                f"MQTT start connect_async returned error {int(connect_result)}"
            )
        self._client.loop_start()

    def _validate_configuration(self) -> None:
        if not self._host:
            raise ValueError("mqtt_host must not be empty")
        if not 1 <= self._port <= 65535:
            raise ValueError("mqtt_port must be in [1, 65535]")
        if not self._start_topic or not self._status_topic:
            raise ValueError("MQTT workflow topics must not be empty")
        if self._start_topic == self._status_topic:
            raise ValueError("MQTT workflow start and status topics must differ")
        if self._qos not in (0, 1, 2):
            raise ValueError("mqtt_qos must be 0, 1, or 2")
        if self._keepalive_sec <= 0:
            raise ValueError("mqtt_keepalive_sec must be positive")

    def _on_connect(
        self,
        client,
        _userdata,
        _flags,
        reason_code,
        _properties=None,
    ) -> None:
        if mqtt_reason_is_failure(reason_code):
            return
        client.subscribe(self._start_topic, qos=self._qos)

    @staticmethod
    def _decode_start(payload: bytes) -> MqttStartRequest:
        try:
            text = payload.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("workflow start payload is not valid UTF-8") from exc
        if not text:
            raise ValueError("workflow start payload is empty")
        if not text.startswith("{"):
            raise ValueError("workflow start payload must be a JSON object")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("workflow start payload is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("workflow start JSON must be an object")
        if value.get("start") is not True:
            raise ValueError("workflow start JSON requires start=true")
        if "robot_id" not in value:
            raise ValueError("workflow start JSON requires robot_id")
        raw_robot_id = value["robot_id"]
        if isinstance(raw_robot_id, bool) or not isinstance(raw_robot_id, (str, int)):
            raise ValueError("workflow start JSON robot_id must be a string or integer")
        robot_id = str(raw_robot_id).strip()
        if not robot_id:
            raise ValueError("workflow start JSON robot_id must not be empty")
        return MqttStartRequest(
            request_id=str(value.get("request_id", "")).strip(),
            robot_id=robot_id,
        )

    def _on_message(self, _client, _userdata, message) -> None:
        if str(getattr(message, "topic", "")) != self._start_topic:
            return
        if bool(getattr(message, "retain", False)):
            return
        try:
            request = self._decode_start(message.payload)
        except (AttributeError, TypeError, ValueError) as exc:
            self.publish_status({"event": "rejected", "message": str(exc)})
            return
        self._on_start_callback(request)

    def publish_status(self, values: Mapping[str, object]) -> bool:
        payload = json.dumps(
            dict(values),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            if self._closed:
                return False
            info = self._client.publish(
                self._status_topic,
                payload=payload,
                qos=self._qos,
                retain=False,
            )
        return int(info.rc) == self._mqtt_success

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._client.disconnect()
        finally:
            self._client.loop_stop()
