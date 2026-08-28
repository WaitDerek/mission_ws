"""Correlation identifiers and redaction for workflow-owned child goals."""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass
from urllib.parse import quote, unquote


_SAFE = "-._~"


def _encode(value: object) -> str:
    return quote(str(value), safe=_SAFE)


def _decode(value: str) -> str:
    return unquote(value)


def new_workflow_id() -> str:
    return str(uuid.uuid4())


def new_lease_token() -> str:
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class StepIdentity:
    stage: str
    point_id: str
    order_index: int
    sequence: int


@dataclass(frozen=True)
class RequestIdentity:
    workflow_id: str
    lease_token: str
    step_id: str


def build_step_id(
    stage: object, point_id: object, order_index: int, sequence: int
) -> str:
    if int(order_index) < 0 or int(sequence) < 0:
        raise ValueError("order_index and sequence must be nonnegative")
    return "|".join(
        (
            "v1",
            f"stage={_encode(stage)}",
            f"p={_encode(point_id)}",
            f"o={int(order_index)}",
            f"n={int(sequence)}",
        )
    )


def parse_step_id(step_id: object) -> StepIdentity:
    parts = str(step_id).split("|")
    if len(parts) != 5 or parts[0] != "v1":
        raise ValueError("invalid step_id format")
    values = {}
    for part in parts[1:]:
        key, separator, value = part.partition("=")
        if not separator or key in values:
            raise ValueError("invalid step_id field")
        values[key] = value
    if set(values) != {"stage", "p", "o", "n"}:
        raise ValueError("step_id fields are incomplete")
    try:
        order_index = int(values["o"])
        sequence = int(values["n"])
    except ValueError as exc:
        raise ValueError("step_id numeric fields are invalid") from exc
    if order_index < 0 or sequence < 0:
        raise ValueError("step_id numeric fields must be nonnegative")
    return StepIdentity(
        stage=_decode(values["stage"]),
        point_id=_decode(values["p"]),
        order_index=order_index,
        sequence=sequence,
    )


def build_request_id(workflow_id: object, lease_token: object, step_id: object) -> str:
    workflow = str(workflow_id).strip()
    token = str(lease_token).strip()
    step = str(step_id).strip()
    if not workflow or not token or not step:
        raise ValueError("workflow_id, lease_token, and step_id are required")
    parse_step_id(step)
    return "|".join(
        (
            "v1",
            f"w={_encode(workflow)}",
            f"l={_encode(token)}",
            f"s={_encode(step)}",
        )
    )


def parse_request_id(request_id: object) -> RequestIdentity:
    parts = str(request_id).split("|")
    if len(parts) != 4 or parts[0] != "v1":
        raise ValueError("request_id is not a workflow request")
    values = {}
    for part in parts[1:]:
        key, separator, value = part.partition("=")
        if not separator or key in values:
            raise ValueError("invalid request_id field")
        values[key] = value
    if set(values) != {"w", "l", "s"}:
        raise ValueError("request_id fields are incomplete")
    identity = RequestIdentity(
        workflow_id=_decode(values["w"]),
        lease_token=_decode(values["l"]),
        step_id=_decode(values["s"]),
    )
    if not identity.workflow_id or not identity.lease_token:
        raise ValueError("request_id identity fields must not be empty")
    parse_step_id(identity.step_id)
    return identity


def sanitize_request_id(request_id: object) -> str:
    """Return trace information without exposing the lease token."""

    raw = str(request_id).strip()
    if not raw:
        return "<empty>"
    try:
        identity = parse_request_id(raw)
    except ValueError:
        return raw
    return "|".join(
        (
            "v1",
            f"w={_encode(identity.workflow_id)}",
            f"s={_encode(identity.step_id)}",
            "l=<redacted>",
        )
    )


class IdentifierFactory:
    def __init__(self, workflow_id: str, lease_token: str) -> None:
        self.workflow_id = workflow_id
        self.lease_token = lease_token
        self._sequence = 0

    def request_id(
        self, stage: str, point_id: str, order_index: int = 0
    ) -> tuple[str, str]:
        step_id = build_step_id(stage, point_id, order_index, self._sequence)
        self._sequence += 1
        return step_id, build_request_id(self.workflow_id, self.lease_token, step_id)
