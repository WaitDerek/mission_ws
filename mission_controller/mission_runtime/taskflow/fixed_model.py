"""Data model for the deterministic front-then-back 16-box workflow."""

from __future__ import annotations

from dataclasses import dataclass

from .mapping import DIRECT_GRASP_ACTION, DRAG_GRASP_ACTION


@dataclass(frozen=True)
class FixedBoxTask:
    """One fixed pickup/place item, independent of Vision ordering."""

    item_index: int
    pickup_point_id: str
    pickup_pos: tuple[float, float, float]
    grasp_action: str
    box_type: str
    box_layer: int
    pickup_zone: str = "front"


@dataclass(frozen=True)
class FixedWorkflowOutcome:
    success: bool
    workflow_id: str
    completed_box_count: int
    last_completed_item_index: int
    final_stage: str
    message: str
    trace: tuple[str, ...] = ()


def _pos(values: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("fixed workflow navigation poses require [x, y, yaw]")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def build_fixed_box_tasks(
    *,
    drag_point_id: int,
    drag_pickup_pos: tuple[float, float, float] | list[float],
    drag_layer4_pickup_pos: tuple[float, float, float] | list[float] | None = None,
    direct_point_id: int,
    direct_pickup_pos: tuple[float, float, float] | list[float],
    direct_layer4_pickup_pos: tuple[float, float, float] | list[float] | None = None,
    north_drag_point_id: int,
    north_drag_pickup_pos: tuple[float, float, float] | list[float],
    north_drag_layer4_pickup_pos: (
        tuple[float, float, float] | list[float] | None
    ) = None,
    north_direct_point_id: int,
    north_direct_pickup_pos: tuple[float, float, float] | list[float],
    north_direct_layer4_pickup_pos: (
        tuple[float, float, float] | list[float] | None
    ) = None,
    drag_action_name: str = DRAG_GRASP_ACTION,
    direct_action_name: str = DIRECT_GRASP_ACTION,
) -> tuple[FixedBoxTask, ...]:
    """Build eight front tasks followed by the same eight-task back order."""
    front = _build_fixed_face_tasks(
        start_index=1,
        pickup_zone="front",
        drag_point_id=drag_point_id,
        drag_pickup_pos=drag_pickup_pos,
        drag_layer4_pickup_pos=drag_layer4_pickup_pos,
        direct_point_id=direct_point_id,
        direct_pickup_pos=direct_pickup_pos,
        direct_layer4_pickup_pos=direct_layer4_pickup_pos,
        drag_action_name=drag_action_name,
        direct_action_name=direct_action_name,
    )
    back = _build_fixed_face_tasks(
        start_index=9,
        pickup_zone="back",
        drag_point_id=north_drag_point_id,
        drag_pickup_pos=north_drag_pickup_pos,
        drag_layer4_pickup_pos=north_drag_layer4_pickup_pos,
        direct_point_id=north_direct_point_id,
        direct_pickup_pos=north_direct_pickup_pos,
        direct_layer4_pickup_pos=north_direct_layer4_pickup_pos,
        drag_action_name=drag_action_name,
        direct_action_name=direct_action_name,
    )
    return front + back


def _build_fixed_face_tasks(
    *,
    start_index: int,
    pickup_zone: str,
    drag_point_id: int,
    drag_pickup_pos: tuple[float, float, float] | list[float],
    drag_layer4_pickup_pos: tuple[float, float, float] | list[float] | None,
    direct_point_id: int,
    direct_pickup_pos: tuple[float, float, float] | list[float],
    direct_layer4_pickup_pos: tuple[float, float, float] | list[float] | None,
    drag_action_name: str,
    direct_action_name: str,
) -> tuple[FixedBoxTask, ...]:
    """Build one face in the established big/small interleaved layer order."""
    drag_point = str(int(drag_point_id))
    direct_point = str(int(direct_point_id))
    drag_pos = _pos(drag_pickup_pos)
    drag_layer4_pos = (
        _pos(drag_layer4_pickup_pos)
        if drag_layer4_pickup_pos is not None
        else drag_pos
    )
    direct_pos = _pos(direct_pickup_pos)
    direct_layer4_pos = (
        _pos(direct_layer4_pickup_pos)
        if direct_layer4_pickup_pos is not None
        else direct_pos
    )
    task_specs = (
        (drag_point, drag_pos, drag_action_name, "bigbox", 1),
        (drag_point, drag_pos, drag_action_name, "bigbox", 2),
        (direct_point, direct_pos, direct_action_name, "smallbox", 1),
        (drag_point, drag_pos, drag_action_name, "bigbox", 3),
        (direct_point, direct_pos, direct_action_name, "smallbox", 2),
        (drag_point, drag_layer4_pos, drag_action_name, "bigbox", 4),
        (direct_point, direct_pos, direct_action_name, "smallbox", 3),
        (direct_point, direct_layer4_pos, direct_action_name, "smallbox", 4),
    )
    return tuple(
        FixedBoxTask(
            start_index + offset,
            point_id,
            position,
            action_name,
            box_type,
            layer,
            pickup_zone,
        )
        for offset, (point_id, position, action_name, box_type, layer) in enumerate(
            task_specs
        )
    )


def build_smallbox_tasks(
    *,
    direct_point_id: int,
    direct_pickup_pos: tuple[float, float, float] | list[float],
    direct_layer4_pickup_pos: tuple[float, float, float] | list[float] | None = None,
    direct_action_name: str = DIRECT_GRASP_ACTION,
) -> tuple[FixedBoxTask, ...]:
    """Build a direct-grasp workflow containing smallbox layers 1 through 4."""
    direct_point = str(int(direct_point_id))
    direct_pos = _pos(direct_pickup_pos)
    direct_layer4_pos = (
        _pos(direct_layer4_pickup_pos)
        if direct_layer4_pickup_pos is not None
        else direct_pos
    )
    return tuple(
        FixedBoxTask(
            layer,
            direct_point,
            direct_layer4_pos if layer == 4 else direct_pos,
            direct_action_name,
            "smallbox",
            layer,
        )
        for layer in range(1, 5)
    )


def build_observed_box_tasks(
    plan,
    *,
    drag_point_id: int,
    drag_pickup_pos: tuple[float, float, float] | list[float],
    drag_layer4_pickup_pos: tuple[float, float, float] | list[float] | None = None,
    direct_point_id: int,
    direct_pickup_pos: tuple[float, float, float] | list[float],
    direct_layer4_pickup_pos: tuple[float, float, float] | list[float] | None = None,
    start_index: int = 1,
    pickup_zone: str = "observed",
    drag_action_name: str = DRAG_GRASP_ACTION,
    direct_action_name: str = DIRECT_GRASP_ACTION,
) -> tuple[FixedBoxTask, ...]:
    """Translate a validated Vision plan into executable fixed-workflow tasks."""
    if plan is None or not getattr(plan, "tasks", None):
        raise ValueError("global observation returned no actionable tasks")
    drag_pos = _pos(drag_pickup_pos)
    drag_layer4_pos = (
        _pos(drag_layer4_pickup_pos)
        if drag_layer4_pickup_pos is not None
        else drag_pos
    )
    direct_pos = _pos(direct_pickup_pos)
    direct_layer4_pos = (
        _pos(direct_layer4_pickup_pos)
        if direct_layer4_pickup_pos is not None
        else direct_pos
    )
    tasks = []
    for item_index, observed in enumerate(plan.tasks, start=int(start_index)):
        stack_index = int(observed.stack_index)
        if stack_index not in (0, 1):
            raise ValueError(
                f"global observation task {item_index} has unsupported stack_index "
                f"{stack_index}; expected 0 or 1"
            )
        grasp_mode = str(getattr(observed, "grasp_mode", "")).strip().lower()
        if grasp_mode == "drag":
            action_name = drag_action_name
        elif grasp_mode == "direct_grasp":
            action_name = direct_action_name
        else:
            raise ValueError(
                f"global observation task {item_index} has unsupported grasp_mode "
                f"{grasp_mode!r}"
            )
        layer = int(observed.layer)
        if stack_index == 0:
            pickup_point = str(int(direct_point_id))
            pickup_pos = direct_layer4_pos if layer == 4 else direct_pos
        else:
            pickup_point = str(int(drag_point_id))
            pickup_pos = drag_layer4_pos if layer == 4 else drag_pos
        tasks.append(
            FixedBoxTask(
                item_index,
                pickup_point,
                pickup_pos,
                action_name,
                str(observed.box_type),
                layer,
                str(pickup_zone),
            )
        )
    return tuple(tasks)
