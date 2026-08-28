"""Pure URDF kinematics and rigid-transform helpers."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class RigidTransform:
    """A transform represented as translation plus xyzw quaternion."""

    translation: Vector3
    rotation: Quaternion


IDENTITY = RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _finite_vector(values: Iterable[float], length: int, label: str) -> Vector3:
    values = tuple(float(value) for value in values)
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain {length} finite values")
    return values  # type: ignore[return-value]


def _normalize_quaternion(values: Iterable[float], label: str) -> Quaternion:
    raw = tuple(float(value) for value in values)
    if len(raw) != 4 or not all(math.isfinite(value) for value in raw):
        raise ValueError(f"{label} must contain 4 finite values")
    norm = math.sqrt(sum(value * value for value in raw))
    if norm <= 1.0e-12:
        raise ValueError(f"{label} must not be a zero quaternion")
    return tuple(value / norm for value in raw)  # type: ignore[return-value]


def _quaternion_multiply(lhs: Quaternion, rhs: Quaternion) -> Quaternion:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quaternion_conjugate(value: Quaternion) -> Quaternion:
    return (-value[0], -value[1], -value[2], value[3])


def _rotate_vector(rotation: Quaternion, vector: Vector3) -> Vector3:
    vector_quaternion = (vector[0], vector[1], vector[2], 0.0)
    rotated = _quaternion_multiply(
        _quaternion_multiply(rotation, vector_quaternion),
        _quaternion_conjugate(rotation),
    )
    return rotated[:3]  # type: ignore[return-value]


def compose(lhs: RigidTransform, rhs: RigidTransform) -> RigidTransform:
    """Return lhs * rhs, where rhs is expressed in lhs's child frame."""

    rotated = _rotate_vector(lhs.rotation, rhs.translation)
    return RigidTransform(
        (
            lhs.translation[0] + rotated[0],
            lhs.translation[1] + rotated[1],
            lhs.translation[2] + rotated[2],
        ),
        _normalize_quaternion(
            _quaternion_multiply(lhs.rotation, rhs.rotation), "composed rotation"
        ),
    )


def inverse(transform: RigidTransform) -> RigidTransform:
    rotation = _quaternion_conjugate(transform.rotation)
    translated = _rotate_vector(
        rotation,
        (
            -transform.translation[0],
            -transform.translation[1],
            -transform.translation[2],
        ),
    )
    return RigidTransform(translated, rotation)


def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _axis_angle_to_quaternion(axis: Vector3, angle: float) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1.0e-12:
        return IDENTITY.rotation
    half_angle = angle * 0.5
    scale = math.sin(half_angle) / norm
    return _normalize_quaternion(
        (
            axis[0] * scale,
            axis[1] * scale,
            axis[2] * scale,
            math.cos(half_angle),
        ),
        "joint rotation",
    )


def _xml_vector(element: Optional[ET.Element], attribute: str, default: str) -> Vector3:
    if element is None:
        raw = default
    else:
        raw = element.get(attribute, default)
    return _finite_vector(raw.split(), 3, f"URDF {attribute}")


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: RigidTransform
    axis: Vector3


class UrdfKinematics:
    """Small URDF FK evaluator covering the fixed/revolute RealBot tree."""

    def __init__(self, urdf_file: str) -> None:
        self.urdf_file = str(Path(urdf_file).expanduser().resolve())
        root = ET.parse(self.urdf_file).getroot()
        if root.tag != "robot":
            raise ValueError(f"{self.urdf_file} is not a URDF robot document")

        self.joints: dict[str, UrdfJoint] = {}
        self.joint_by_child: dict[str, UrdfJoint] = {}
        self.links = {link.get("name", "") for link in root.findall("link")}
        for element in root.findall("joint"):
            name = element.get("name", "").strip()
            joint_type = element.get("type", "").strip()
            parent_element = element.find("parent")
            child_element = element.find("child")
            if (
                not name
                or not joint_type
                or parent_element is None
                or child_element is None
            ):
                raise ValueError(f"malformed joint in {self.urdf_file}")
            parent = parent_element.get("link", "").strip()
            child = child_element.get("link", "").strip()
            if not parent or not child:
                raise ValueError(f"joint {name} has an empty parent or child")
            origin_element = element.find("origin")
            xyz = _xml_vector(origin_element, "xyz", "0 0 0")
            rpy = _xml_vector(origin_element, "rpy", "0 0 0")
            axis = _xml_vector(element.find("axis"), "xyz", "0 0 0")
            joint = UrdfJoint(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin=RigidTransform(xyz, _rpy_to_quaternion(*rpy)),
                axis=axis,
            )
            if name in self.joints or child in self.joint_by_child:
                raise ValueError(f"duplicate URDF joint name or child link: {name}")
            self.joints[name] = joint
            self.joint_by_child[child] = joint

    def _joint_transform(
        self, joint: UrdfJoint, joint_positions: Mapping[str, float]
    ) -> RigidTransform:
        if joint.joint_type in ("fixed", "floating", "planar"):
            return joint.origin
        if joint.joint_type not in ("revolute", "continuous"):
            raise ValueError(
                f"unsupported URDF joint type {joint.joint_type!r} for {joint.name}"
            )
        angle = float(joint_positions.get(joint.name, 0.0))
        motion = RigidTransform(
            (0.0, 0.0, 0.0),
            _axis_angle_to_quaternion(joint.axis, angle),
        )
        return compose(joint.origin, motion)

    def transform_from_root(
        self,
        root_frame: str,
        child_frame: str,
        joint_positions: Mapping[str, float],
    ) -> RigidTransform:
        """Return T_root_child by following the URDF parent chain."""

        root_frame = root_frame.lstrip("/")
        child_frame = child_frame.lstrip("/")
        if root_frame not in self.links or child_frame not in self.links:
            raise ValueError(
                f"unknown URDF frame: root={root_frame!r}, child={child_frame!r}"
            )
        path: list[UrdfJoint] = []
        current = child_frame
        visited: set[str] = set()
        while current != root_frame:
            if current in visited:
                raise ValueError(f"cycle while tracing URDF frame {child_frame}")
            visited.add(current)
            joint = self.joint_by_child.get(current)
            if joint is None:
                raise ValueError(
                    f"{root_frame!r} is not an ancestor of {child_frame!r}"
                )
            path.append(joint)
            current = joint.parent

        result = IDENTITY
        for joint in reversed(path):
            result = compose(result, self._joint_transform(joint, joint_positions))
        return result


def compatibility_arm_base_transforms(
    kinematics: UrdfKinematics,
    root_frame: str,
    chest_frame: str,
    joint_positions: Mapping[str, float],
    left_chest_to_arm_base: RigidTransform,
    right_chest_to_arm_base: RigidTransform,
) -> tuple[RigidTransform, RigidTransform]:
    """Build live root-to-SDK-arm-base transforms.

    The current ``realbots29`` URDF models the physical shoulder branches as
    ``left_arm_*``/``right_arm_*`` links, while the mission SDK targets are
    expressed in the legacy ``L_base_Link``/``R_base_Link`` frames.  Keep the
    calibrated chest-to-SDK-base transforms explicit, but obtain the
    root-to-chest part from the current joint state so waist motion is not
    frozen at the URDF zero pose.
    """

    root_to_chest = kinematics.transform_from_root(
        root_frame, chest_frame, joint_positions
    )
    return (
        compose(root_to_chest, left_chest_to_arm_base),
        compose(root_to_chest, right_chest_to_arm_base),
    )


def _clean_frame(value: str) -> str:
    return str(value).strip().lstrip("/")


def _as_string_list(value: object, parameter_name: str) -> list[str]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        try:
            values = [str(part).strip() for part in value]  # type: ignore[union-attr]
        except TypeError as exc:
            raise ValueError(f"{parameter_name} must be a string array") from exc
    if not values or any(not value for value in values):
        raise ValueError(f"{parameter_name} must not be empty")
    return values
