"""MQTT request/result navigation adapter for four configured map points."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .model import NavigationRequest, NavigationResult
from .mqtt_support import (
    create_paho_client,
    mqtt_call_succeeded,
    mqtt_reason_is_failure,
)


VALID_POINT_IDS = frozenset({"1", "2", "3", "4"})


@dataclass(frozen=True)
class NavigationPoint:
    x: float
    y: float
    yaw: float


def parse_navigation_points_json(payload: str) -> dict[str, NavigationPoint]:
    text = str(payload).strip()
    if not text:
        raise ValueError("mqtt_navigation_points_json must not be empty")
    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("mqtt_navigation_points_json is invalid JSON") from exc
    if not isinstance(values, dict):
        raise ValueError("mqtt_navigation_points_json must be a JSON object")

    points: dict[str, NavigationPoint] = {}
    for raw_id, raw_pose in values.items():
        point_id = str(raw_id).strip()
        if point_id not in VALID_POINT_IDS:
            raise ValueError(
                f"MQTT navigation coordinate id must be in 1..4, got {point_id!r}"
            )
        if not isinstance(raw_pose, dict):
            raise ValueError(f"MQTT navigation point {point_id} must be an object")
        missing = {"x", "y", "yaw"}.difference(raw_pose)
        if missing:
            raise ValueError(
                f"MQTT navigation point {point_id} is missing "
                + ", ".join(sorted(missing))
            )
        coordinates = []
        for name in ("x", "y", "yaw"):
            value = raw_pose[name]
            if isinstance(value, bool):
                raise ValueError(
                    f"MQTT navigation point {point_id} {name} must be numeric"
                )
            try:
                coordinate = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"MQTT navigation point {point_id} {name} must be numeric"
                ) from exc
            if not math.isfinite(coordinate):
                raise ValueError(
                    f"MQTT navigation point {point_id} {name} must be finite"
                )
            coordinates.append(coordinate)
        points[point_id] = NavigationPoint(*coordinates)
    return points


def require_all_navigation_points(points: Mapping[str, NavigationPoint]) -> None:
    missing = VALID_POINT_IDS.difference(points)
    if missing:
        raise ValueError(
            "MQTT navigation coordinates are missing ids: " + ", ".join(sorted(missing))
        )


class MqttNavigationGateway:
    def __init__(
        self,
        *,
        host: str,
        port: int = 1883,
        request_topic: str = "mission/navigation/request",
        result_topic: str = "mission/navigation/result",
        client_id: str = "",
        qos: int = 1,
        keepalive_sec: int = 60,
        connect_timeout_sec: float = 10.0,
        navigation_timeout_sec: float = 300.0,
        frame_id: str = "map",
        point_poses: Mapping[str, NavigationPoint] | None = None,
        client: Any | None = None,
    ) -> None:
        self._host = str(host).strip()
        self._port = int(port)
        self._request_topic = str(request_topic).strip()
        self._result_topic = str(result_topic).strip()
        self._qos = int(qos)
        self._keepalive_sec = int(keepalive_sec)
        self._connect_timeout_sec = float(connect_timeout_sec)
        self._navigation_timeout_sec = float(navigation_timeout_sec)
        self._frame_id = str(frame_id).strip().lstrip("/")
        self._point_poses = dict(point_poses or {})
        self._validate_configuration()

        self._connected = threading.Event()
        self._response_ready = threading.Event()
        self._cancel_requested = threading.Event()
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending_point_id = ""
        self._response: NavigationResult | None = None
        self._connection_error = ""
        self._closed = False

        self._mqtt_success = 0
        if client is None:
            self._client, self._mqtt_success = create_paho_client(client_id)
        else:
            self._client = client
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        connect_result = self._client.connect_async(
            self._host,
            self._port,
            self._keepalive_sec,
        )
        if not mqtt_call_succeeded(connect_result, self._mqtt_success):
            self._connection_error = (
                f"MQTT connect_async returned error {connect_result!r}"
            )
        self._client.loop_start()

    def _validate_configuration(self) -> None:
        if not self._host:
            raise ValueError("mqtt_host must not be empty")
        if not 1 <= self._port <= 65535:
            raise ValueError("mqtt_port must be in [1, 65535]")
        if not self._request_topic or not self._result_topic:
            raise ValueError("MQTT navigation topics must not be empty")
        if self._request_topic == self._result_topic:
            raise ValueError("MQTT request and result topics must differ")
        if self._qos not in (0, 1, 2):
            raise ValueError("mqtt_qos must be 0, 1, or 2")
        if self._keepalive_sec <= 0:
            raise ValueError("mqtt_keepalive_sec must be positive")
        if self._connect_timeout_sec <= 0.0:
            raise ValueError("mqtt_connect_timeout_sec must be positive")
        if self._navigation_timeout_sec <= 0.0:
            raise ValueError("mqtt_navigation_timeout_sec must be positive")
        if not self._frame_id:
            raise ValueError("mqtt_navigation_frame_id must not be empty")
        invalid_ids = set(self._point_poses).difference(VALID_POINT_IDS)
        if invalid_ids:
            raise ValueError(
                "MQTT navigation coordinates contain invalid ids: "
                + ", ".join(sorted(invalid_ids))
            )

    def _on_connect(
        self,
        client,
        _userdata,
        _flags,
        reason_code,
        _properties=None,
    ) -> None:
        if mqtt_reason_is_failure(reason_code):
            self._connection_error = f"MQTT connection rejected: {reason_code}"
            self._connected.clear()
            return
        subscribe_result, _message_id = client.subscribe(
            self._result_topic,
            qos=self._qos,
        )
        if int(subscribe_result) != self._mqtt_success:
            self._connection_error = (
                f"MQTT subscribe returned error {int(subscribe_result)}"
            )
            self._connected.clear()
            return
        self._connection_error = ""
        self._connected.set()

    def _on_disconnect(self, _client, _userdata, *_args) -> None:
        self._connected.clear()
        with self._state_lock:
            if self._pending_point_id and not self._cancel_requested.is_set():
                self._response = NavigationResult(
                    False,
                    "unavailable",
                    "MQTT broker disconnected during navigation",
                )
                self._response_ready.set()

    @staticmethod
    def _decode_result(payload: bytes) -> tuple[str, bool, str]:
        try:
            text = payload.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("navigation result is not valid UTF-8") from exc
        if not text:
            raise ValueError("navigation result payload is empty")
        if not text.startswith("{"):
            return text, True, "platform reported arrival"
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("navigation result is invalid JSON") from exc
        if not isinstance(result, dict):
            raise ValueError("navigation result JSON must be an object")
        point_id = str(result.get("id", result.get("point_id", ""))).strip()
        success = result.get("success")
        if not point_id:
            raise ValueError("navigation result JSON is missing id")
        if not isinstance(success, bool):
            raise ValueError("navigation result JSON success must be boolean")
        return point_id, success, str(result.get("message", "")).strip()

    def _on_message(self, _client, _userdata, message) -> None:
        if str(getattr(message, "topic", "")) != self._result_topic:
            return
        if bool(getattr(message, "retain", False)):
            return
        try:
            point_id, success, detail = self._decode_result(message.payload)
        except (AttributeError, TypeError, ValueError):
            return
        with self._state_lock:
            if (
                self._cancel_requested.is_set()
                or self._response_ready.is_set()
                or point_id != self._pending_point_id
            ):
                return
            self._response = NavigationResult(
                success,
                "succeeded" if success else "failed",
                detail
                or (
                    f"platform reached point {point_id}"
                    if success
                    else f"platform failed to reach point {point_id}"
                ),
            )
            self._response_ready.set()

    @staticmethod
    def _wait_event(event, timeout_sec, canceled: Callable[[], bool]) -> bool:
        deadline = time.monotonic() + timeout_sec
        while True:
            if canceled():
                return False
            if event.wait(0.05):
                return not canceled()
            if time.monotonic() >= deadline:
                return False

    def navigate(
        self, request: NavigationRequest, cancel_requested
    ) -> NavigationResult:
        point_id = str(request.point_id).strip()
        if point_id not in VALID_POINT_IDS:
            return NavigationResult(
                False,
                "invalid",
                f"MQTT navigation point must be in 1..4, got {point_id!r}",
            )
        target = self._point_poses.get(point_id)
        if target is None:
            return NavigationResult(
                False,
                "invalid",
                f"MQTT navigation point {point_id} has no configured coordinates",
            )
        with self._request_lock:
            if self._closed:
                return NavigationResult(False, "unavailable", "MQTT gateway is closed")
            self._cancel_requested.clear()

            def canceled():
                return self._cancel_requested.is_set() or cancel_requested()

            if not self._wait_event(
                self._connected,
                self._connect_timeout_sec,
                canceled,
            ):
                if canceled():
                    return NavigationResult(False, "canceled", "navigation canceled")
                return NavigationResult(
                    False,
                    "unavailable",
                    self._connection_error or "MQTT broker connection timed out",
                )

            with self._state_lock:
                self._pending_point_id = point_id
                self._response = None
                self._response_ready.clear()
            publish_info = self._client.publish(
                self._request_topic,
                payload=json.dumps(
                    {
                        "id": int(point_id),
                        "frame_id": self._frame_id,
                        "pos": [target.x, target.y, target.yaw],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                qos=self._qos,
                retain=False,
            )
            if int(publish_info.rc) != self._mqtt_success:
                self._clear_pending()
                return NavigationResult(
                    False,
                    "unavailable",
                    f"MQTT publish returned error {int(publish_info.rc)}",
                )

            if not self._wait_event(
                self._response_ready,
                self._navigation_timeout_sec,
                canceled,
            ):
                self._clear_pending()
                if canceled():
                    return NavigationResult(False, "canceled", "navigation canceled")
                return NavigationResult(
                    False,
                    "timeout",
                    f"platform navigation to point {point_id} timed out",
                )
            with self._state_lock:
                response = self._response
                self._pending_point_id = ""
                self._response = None
                self._response_ready.clear()
            return response or NavigationResult(
                False,
                "failed",
                "MQTT navigation completed without a result",
            )

    def _clear_pending(self) -> None:
        with self._state_lock:
            self._pending_point_id = ""
            self._response = None
            self._response_ready.clear()

    def cancel_active(self) -> None:
        self._cancel_requested.set()
        self._response_ready.set()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self.cancel_active()
        try:
            self._client.disconnect()
        finally:
            self._client.loop_stop()
