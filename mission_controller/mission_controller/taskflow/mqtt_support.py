"""Lazy Paho MQTT compatibility helpers."""


def create_paho_client(client_id: str):
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("paho-mqtt is required for MQTT taskflow features") from exc

    kwargs = {"client_id": str(client_id).strip(), "protocol": mqtt.MQTTv311}
    callback_versions = getattr(mqtt, "CallbackAPIVersion", None)
    if callback_versions is None:
        client = mqtt.Client(**kwargs)
    else:
        client = mqtt.Client(callback_versions.VERSION2, **kwargs)
    return client, int(mqtt.MQTT_ERR_SUCCESS)


def mqtt_reason_is_failure(reason_code: object) -> bool:
    is_failure = getattr(reason_code, "is_failure", None)
    if is_failure is not None:
        return bool(is_failure)
    try:
        return int(reason_code) != 0
    except (TypeError, ValueError):
        return True


def mqtt_call_succeeded(result: object, success_code: int = 0) -> bool:
    """Accept Paho 2.x async calls returning None and older integer codes."""
    if result is None:
        return True
    try:
        return int(result) == int(success_code)
    except (TypeError, ValueError):
        return False
