"""Validation adapter for Vision ``GlobalObservation`` results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from .mapping import normalize_box_type, normalize_stack_member
from .model import ObservationPlan, ObservationResult, ObservationTask


class ObservationValidationError(ValueError):
    pass


_ORDER_FIELDS = (
    "order_stack_ids",
    "order_stack_indices",
    "order_layer_numbers",
    "order_columns",
    "order_box_sizes",
)


@dataclass(frozen=True)
class FrontStackPoseValidation:
    enabled: bool = True
    expected_count: int = 2
    min_lateral_separation_m: float = 0.20
    max_depth_spread_m: float = 0.35
    max_camera_depth_m: float = 1.20

    def validate(self) -> None:
        if self.expected_count <= 0:
            raise ValueError("expected front stack count must be positive")
        if self.min_lateral_separation_m <= 0.0:
            raise ValueError("front stack lateral separation must be positive")
        if self.max_depth_spread_m <= 0.0:
            raise ValueError("front stack depth spread must be positive")
        if self.max_camera_depth_m < 0.0:
            raise ValueError("front stack maximum camera depth cannot be negative")

def _sequence(result: object, name: str) -> Sequence:
    value = getattr(result, name, None)
    if value is None or isinstance(value, (str, bytes)):
        raise ObservationValidationError(f"Vision result field {name} is missing")
    return value


def _camera_position(pose_stamped: object, index: int) -> tuple[str, float, float]:
    try:
        frame_id = str(pose_stamped.header.frame_id).strip().lstrip("/")
        x = float(pose_stamped.pose.position.x)
        z = float(pose_stamped.pose.position.z)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ObservationValidationError(
            f"Vision top_box_camera_poses[{index}] is invalid"
        ) from exc
    if not frame_id:
        raise ObservationValidationError(
            f"Vision top_box_camera_poses[{index}] has an empty frame_id"
        )
    if not math.isfinite(x) or not math.isfinite(z) or z <= 0.0:
        raise ObservationValidationError(
            f"Vision top_box_camera_poses[{index}] has an invalid camera position"
        )
    return frame_id, x, z


def _validate_front_stack_poses(
    result: object,
    config: FrontStackPoseValidation,
) -> None:
    if not config.enabled:
        return
    try:
        config.validate()
    except ValueError as exc:
        raise ObservationValidationError(str(exc)) from exc
    poses = _sequence(result, "top_box_camera_poses")
    sides = _sequence(result, "stack_sides")
    columns = _sequence(result, "stack_columns")
    expected = config.expected_count
    lengths = {
        "top_box_camera_poses": len(poses),
        "stack_sides": len(sides),
        "stack_columns": len(columns),
    }
    if any(length != expected for length in lengths.values()):
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ObservationValidationError(
            f"Vision must return exactly {expected} front stacks: {detail}"
        )
    try:
        normalized_sides = tuple(int(side) for side in sides)
        normalized_columns = tuple(int(column) for column in columns)
    except (TypeError, ValueError) as exc:
        raise ObservationValidationError(
            "Vision front stack side/column values must be integers"
        ) from exc
    if any(side != 0 for side in normalized_sides):
        raise ObservationValidationError("Vision result contains a non-front stack")
    if normalized_columns != tuple(range(expected)):
        raise ObservationValidationError(
            "Vision front stack columns must be ordered left-to-right"
        )

    positions = [_camera_position(pose, index) for index, pose in enumerate(poses)]
    frames = {frame_id for frame_id, _x, _z in positions}
    if len(frames) != 1:
        raise ObservationValidationError(
            "Vision top-box camera poses must use the same camera frame"
        )
    lateral = [x for _frame, x, _z in positions]
    for left, right in zip(lateral, lateral[1:]):
        if right - left < config.min_lateral_separation_m:
            raise ObservationValidationError(
                "Vision top-box poses do not represent distinct left/right stacks"
            )
    depths = [z for _frame, _x, z in positions]
    if max(depths) - min(depths) > config.max_depth_spread_m:
        raise ObservationValidationError(
            "Vision top-box depths are inconsistent with one front row"
        )
    if config.max_camera_depth_m > 0.0 and max(depths) > config.max_camera_depth_m:
        raise ObservationValidationError(
            "Vision top-box poses are beyond the configured front-row depth"
        )


def adapt_global_observation_result(
    point_id: object,
    result: object,
    *,
    front_stack_validation: FrontStackPoseValidation | None = None,
) -> ObservationResult:
    """Convert a ROS result-like object to validated immutable tasks.

    The function intentionally accepts a generic object so the complete
    mapping/validation contract can be tested without importing ROS messages.
    """

    normalized_point = str(point_id).strip()
    message = str(getattr(result, "message", "")).strip()
    if not bool(getattr(result, "success", False)):
        return ObservationResult(False, message=message or "Vision observation failed")
    if not bool(getattr(result, "plan_valid", False)):
        return ObservationResult(False, message=message or "Vision returned no plan")

    _validate_front_stack_poses(
        result,
        front_stack_validation or FrontStackPoseValidation(),
    )

    fields = {name: _sequence(result, name) for name in _ORDER_FIELDS}
    lengths = {name: len(values) for name, values in fields.items()}
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ObservationValidationError(
            f"Vision order arrays must have equal lengths: {detail}"
        )
    count = next(iter(lengths.values()), 0)
    if count <= 0:
        raise ObservationValidationError(
            "Vision marked the plan valid but returned no order items"
        )

    tasks = []
    for order_index in range(count):
        stack_id = str(fields["order_stack_ids"][order_index]).strip()
        if not stack_id:
            raise ObservationValidationError(
                f"Vision order item {order_index} has an empty stack_id"
            )
        try:
            stack_index = normalize_stack_member(
                fields["order_stack_indices"][order_index]
            )
            column = normalize_stack_member(fields["order_columns"][order_index])
        except ValueError as exc:
            raise ObservationValidationError(
                f"Vision order item {order_index}: {exc}"
            ) from exc
        if stack_index != column:
            raise ObservationValidationError(
                "Vision order item "
                f"{order_index} disagrees: stack_index={stack_index}, column={column}"
            )
        try:
            layer = int(fields["order_layer_numbers"][order_index])
        except (TypeError, ValueError) as exc:
            raise ObservationValidationError(
                f"Vision order item {order_index} has an invalid layer"
            ) from exc
        if layer not in (1, 2, 3, 4):
            raise ObservationValidationError(
                f"Vision order item {order_index} layer must be in 1..4, got {layer}"
            )
        try:
            box_type = normalize_box_type(fields["order_box_sizes"][order_index])
        except ValueError as exc:
            raise ObservationValidationError(
                f"Vision order item {order_index}: {exc}"
            ) from exc
        tasks.append(
            ObservationTask(
                stack_id=stack_id,
                stack_index=stack_index,
                column=column,
                layer=layer,
                box_type=box_type,
                order_index=order_index,
            )
        )

    return ObservationResult(
        True,
        ObservationPlan(normalized_point, tuple(tasks), message),
        message,
    )
