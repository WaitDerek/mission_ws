"""Pure point, grasp-mode, and Vision value mappings."""

from __future__ import annotations


OBSERVATION_POINT_IDS = ("1", "2", "3", "4")
OBSERVATION_PAIRS = {"1": "3", "2": "4", "3": "1", "4": "2"}
# Operation IDs run clockwise around the pallet from 5 through 12. Vision
# columns are camera-relative at every observation pose: index 0 is the visible
# left stack (even/direct), while index 1 is the visible right stack (odd/drag).
# The workflow uses only the opposite observation pair that sees the box long
# edge: 1/3 or 2/4.
OPERATION_POINTS = {
    "1": ("6", "5"),
    "2": ("8", "7"),
    "3": ("10", "9"),
    "4": ("12", "11"),
}
PLACE_POINT_ID = "16"
RESERVED_POINT_IDS = frozenset({"13", "14", "15"})
DRAG_GRASP_ACTION = "/execute_drag_box_grasp_tf"
DIRECT_GRASP_ACTION = "/grasp_box_tf"


def normalize_point_id(value: object) -> str:
    point_id = str(value).strip()
    if not point_id:
        raise ValueError("point_id must not be empty")
    return point_id


def normalize_stack_member(value: object) -> int:
    try:
        member = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stack member must be 0 or 1, got {value!r}") from exc
    if member not in (0, 1):
        raise ValueError(f"stack member must be 0 or 1, got {member}")
    return member


def operation_point_for_stack(
    observation_point_id: object, stack_member: object
) -> str:
    observation_point = normalize_point_id(observation_point_id)
    try:
        pair = OPERATION_POINTS[observation_point]
    except KeyError as exc:
        raise ValueError(
            f"unknown observation point {observation_point!r}; expected 1..4"
        ) from exc
    return pair[normalize_stack_member(stack_member)]


def grasp_action_for_operation_point(operation_point_id: object) -> str:
    point_id = normalize_point_id(operation_point_id)
    if point_id in RESERVED_POINT_IDS or point_id == PLACE_POINT_ID:
        raise ValueError(f"point {point_id} is not a pickup operation point")
    valid_points = {point for pair in OPERATION_POINTS.values() for point in pair}
    if point_id not in valid_points:
        raise ValueError(f"unknown pickup operation point {point_id!r}")
    return DRAG_GRASP_ACTION if int(point_id) % 2 else DIRECT_GRASP_ACTION


def normalize_box_type(value: object) -> str:
    normalized = str(value).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "big": "bigbox",
        "bigbox": "bigbox",
        "large": "bigbox",
        "largebox": "bigbox",
        "small": "smallbox",
        "smallbox": "smallbox",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported box size {value!r}") from exc
