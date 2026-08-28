"""Bridge RealBot feedback into the URDF TF tree.

The hardware publishes torso and arm feedback on the rm_robot_interfaces
topics rather than on one complete joint-state stream. This node normalizes
those streams and publishes a complete ``JointState`` stream for
``robot_state_publisher``. The cameras are attached with fixed extrinsics:

    base_footprint -> ... -> left_arm_8_Link
                    -> left_arm_depth_cam_link -> ... -> optical frame
                  and right_arm_8_Link
                    -> right_arm_depth_cam_link -> ... -> optical frame

The camera drivers own the links below each ``*_depth_cam_link``. Keeping the
mounts at the camera-link roots avoids creating a second parent for either
optical frame.
"""

from __future__ import annotations

import math
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rm_robot_interfaces.msg import ArmSlaveData, BodyData
from sensor_msgs.msg import JointState
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


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


class RealbotsGlobalTf(Node):
    """Read RealBot feedback and bridge it into the URDF/camera TF tree."""

    def __init__(self) -> None:
        super().__init__("realbots_global_tf")

        self.declare_parameter("urdf_file", "")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("output_frame", "base_footprint")
        self.declare_parameter(
            "camera_optical_frame", "left_arm_depth_cam_color_optical_frame"
        )
        self.declare_parameter("camera_mount_parent_link", "left_camera_Link")
        self.declare_parameter(
            "camera_mount_child_frame", "left_arm_depth_cam_link"
        )
        self.declare_parameter(
            "camera_mount_xyz", [0.049094570020, 0.0, -0.000100000000]
        )
        self.declare_parameter(
            "camera_mount_quaternion_xyzw",
            [0.500617260, 0.499457371, 0.499528938, -0.500395377],
        )
        self.declare_parameter("right_camera_mount_parent_link", "right_camera_Link")
        self.declare_parameter(
            "right_camera_mount_child_frame", "right_arm_depth_cam_link"
        )
        self.declare_parameter(
            "right_camera_mount_xyz",
            [0.049094570020, 0.0, -0.000100000000],
        )
        self.declare_parameter(
            "right_camera_mount_quaternion_xyzw",
            [-0.497028867, -0.503278286, -0.498134947, 0.501532498],
        )
        self.declare_parameter("body_feedback_topic", "/mcap/body")
        self.declare_parameter("left_arm_feedback_topic", "/mcap/slave_arm_left")
        self.declare_parameter("right_arm_feedback_topic", "/mcap/slave_arm_right")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter(
            "body_input_joint_names", ["joint1", "joint2", "joint3", "joint4"]
        )
        self.declare_parameter(
            "body_urdf_joint_names",
            ["waist_1_joint", "waist_2_joint", "waist_3_joint", "chest_joint"],
        )
        self.declare_parameter("neck_input_joint_names", ["joint1", "joint2"])
        self.declare_parameter(
            "neck_urdf_joint_names", ["neck_Link", "head_joint"]
        )
        self.declare_parameter(
            "left_arm_input_joint_names",
            [f"joint{index}" for index in range(1, 8)],
        )
        self.declare_parameter(
            "right_arm_input_joint_names",
            [f"joint{index}" for index in range(1, 8)],
        )
        self.declare_parameter(
            "left_arm_urdf_joint_names",
            [f"left_arm_{index}_joint" for index in range(1, 8)],
        )
        self.declare_parameter(
            "right_arm_urdf_joint_names",
            [f"right_arm_{index}_joint" for index in range(1, 8)],
        )
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("max_feedback_age_sec", 0.5)
        self.declare_parameter("publish_complete_joint_states", True)
        self.declare_parameter("default_joint_position", 0.0)
        self.declare_parameter("compatibility_frames_enabled", True)
        self.declare_parameter("compatibility_urdf_root_frame", "base_Link")
        self.declare_parameter("compatibility_base_frame", "base_link")
        self.declare_parameter("compatibility_chest_frame", "chest_Link")
        self.declare_parameter(
            "compatibility_left_arm_base_frame", "L_base_Link"
        )
        self.declare_parameter(
            "compatibility_right_arm_base_frame", "R_base_Link"
        )
        self.declare_parameter(
            "compatibility_left_chest_to_arm_base_xyz", [0.012, 0.0, -0.2975]
        )
        self.declare_parameter(
            "compatibility_left_chest_to_arm_base_rpy",
            [0.0, math.pi, 0.0],
        )
        self.declare_parameter(
            "compatibility_right_chest_to_arm_base_xyz", [-0.012, 0.0, -0.2975]
        )
        self.declare_parameter(
            "compatibility_right_chest_to_arm_base_rpy",
            [math.pi, 0.0, 0.0],
        )

        urdf_file = str(self.get_parameter("urdf_file").value).strip()
        if not urdf_file:
            try:
                from ament_index_python.packages import (
                    get_package_share_directory,
                )

                urdf_file = str(
                    Path(get_package_share_directory("realbots29"))
                    / "urdf"
                    / "realbots29.urdf"
                )
            except Exception as exc:
                raise RuntimeError(
                    "urdf_file is empty and the realbots29 package is unavailable"
                ) from exc

        self.base_frame = _clean_frame(self.get_parameter("base_frame").value)
        self.output_frame = _clean_frame(self.get_parameter("output_frame").value)
        self.camera_frame = _clean_frame(
            self.get_parameter("camera_optical_frame").value
        )
        self.camera_mount_parent = _clean_frame(
            self.get_parameter("camera_mount_parent_link").value
        )
        self.camera_mount_child = _clean_frame(
            self.get_parameter("camera_mount_child_frame").value
        )
        self.right_camera_mount_parent = _clean_frame(
            self.get_parameter("right_camera_mount_parent_link").value
        )
        self.right_camera_mount_child = _clean_frame(
            self.get_parameter("right_camera_mount_child_frame").value
        )
        self.body_input_names = _as_string_list(
            self.get_parameter("body_input_joint_names").value,
            "body_input_joint_names",
        )
        self.body_urdf_names = _as_string_list(
            self.get_parameter("body_urdf_joint_names").value,
            "body_urdf_joint_names",
        )
        self.neck_input_names = _as_string_list(
            self.get_parameter("neck_input_joint_names").value,
            "neck_input_joint_names",
        )
        self.neck_urdf_names = _as_string_list(
            self.get_parameter("neck_urdf_joint_names").value,
            "neck_urdf_joint_names",
        )
        self.left_input_names = _as_string_list(
            self.get_parameter("left_arm_input_joint_names").value,
            "left_arm_input_joint_names",
        )
        self.right_input_names = _as_string_list(
            self.get_parameter("right_arm_input_joint_names").value,
            "right_arm_input_joint_names",
        )
        self.left_urdf_names = _as_string_list(
            self.get_parameter("left_arm_urdf_joint_names").value,
            "left_arm_urdf_joint_names",
        )
        self.right_urdf_names = _as_string_list(
            self.get_parameter("right_arm_urdf_joint_names").value,
            "right_arm_urdf_joint_names",
        )
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.max_feedback_age_sec = float(
            self.get_parameter("max_feedback_age_sec").value
        )
        self.publish_complete_joint_states = bool(
            self.get_parameter("publish_complete_joint_states").value
        )
        self.default_joint_position = float(
            self.get_parameter("default_joint_position").value
        )
        self.compatibility_frames_enabled = bool(
            self.get_parameter("compatibility_frames_enabled").value
        )
        self.compatibility_urdf_root_frame = _clean_frame(
            self.get_parameter("compatibility_urdf_root_frame").value
        )
        self.compatibility_base_frame = _clean_frame(
            self.get_parameter("compatibility_base_frame").value
        )
        self.compatibility_chest_frame = _clean_frame(
            self.get_parameter("compatibility_chest_frame").value
        )
        self.compatibility_left_arm_base_frame = _clean_frame(
            self.get_parameter("compatibility_left_arm_base_frame").value
        )
        self.compatibility_right_arm_base_frame = _clean_frame(
            self.get_parameter("compatibility_right_arm_base_frame").value
        )
        if self.publish_rate_hz <= 0.0 or self.max_feedback_age_sec <= 0.0:
            raise ValueError("publish_rate_hz and max_feedback_age_sec must be positive")
        if not math.isfinite(self.default_joint_position):
            raise ValueError("default_joint_position must be finite")
        if len(self.body_input_names) != len(self.body_urdf_names):
            raise ValueError("body input and URDF joint name arrays must have equal size")
        if len(self.neck_input_names) != len(self.neck_urdf_names):
            raise ValueError("neck input and URDF joint name arrays must have equal size")
        if len(self.left_input_names) != len(self.left_urdf_names):
            raise ValueError(
                "left arm input and URDF joint name arrays must have equal size"
            )
        if len(self.right_input_names) != len(self.right_urdf_names):
            raise ValueError(
                "right arm input and URDF joint name arrays must have equal size"
            )

        self.kinematics = UrdfKinematics(urdf_file)
        if self.compatibility_frames_enabled:
            for frame in (
                self.compatibility_urdf_root_frame,
                self.compatibility_chest_frame,
            ):
                if frame not in self.kinematics.links:
                    raise ValueError(
                        "compatibility frame must be present in the URDF: "
                        f"{frame}"
                    )
            left_xyz = _finite_vector(
                self.get_parameter("compatibility_left_chest_to_arm_base_xyz").value,
                3,
                "compatibility_left_chest_to_arm_base_xyz",
            )
            left_rpy = _finite_vector(
                self.get_parameter("compatibility_left_chest_to_arm_base_rpy").value,
                3,
                "compatibility_left_chest_to_arm_base_rpy",
            )
            right_xyz = _finite_vector(
                self.get_parameter(
                    "compatibility_right_chest_to_arm_base_xyz"
                ).value,
                3,
                "compatibility_right_chest_to_arm_base_xyz",
            )
            right_rpy = _finite_vector(
                self.get_parameter(
                    "compatibility_right_chest_to_arm_base_rpy"
                ).value,
                3,
                "compatibility_right_chest_to_arm_base_rpy",
            )
            self.compatibility_left_chest_to_arm_base = RigidTransform(
                left_xyz, _rpy_to_quaternion(*left_rpy)
            )
            self.compatibility_right_chest_to_arm_base = RigidTransform(
                right_xyz, _rpy_to_quaternion(*right_rpy)
            )
        self.movable_joint_names = tuple(
            joint.name
            for joint in self.kinematics.joints.values()
            if joint.joint_type in ("revolute", "continuous")
        )
        mount_xyz = _finite_vector(
            self.get_parameter("camera_mount_xyz").value,
            3,
            "camera_mount_xyz",
        )
        mount_quaternion = _normalize_quaternion(
            self.get_parameter("camera_mount_quaternion_xyzw").value,
            "camera_mount_quaternion_xyzw",
        )
        self.camera_mount = RigidTransform(mount_xyz, mount_quaternion)
        right_mount_xyz = _finite_vector(
            self.get_parameter("right_camera_mount_xyz").value,
            3,
            "right_camera_mount_xyz",
        )
        right_mount_quaternion = _normalize_quaternion(
            self.get_parameter("right_camera_mount_quaternion_xyzw").value,
            "right_camera_mount_quaternion_xyzw",
        )
        self.right_camera_mount = RigidTransform(
            right_mount_xyz, right_mount_quaternion
        )

        self._lock = threading.Lock()
        self._joint_positions: dict[str, float] = {}
        self._feedback_times = {
            "body": 0.0,
            "neck": 0.0,
            "left": 0.0,
            "right": 0.0,
        }
        self._last_warning_time = 0.0

        feedback_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Hardware feedback topics are best-effort, but robot_state_publisher
        # subscribes to /joint_states with reliable QoS.  Use a dedicated
        # reliable publisher profile so normalized joint states enter the URDF
        # FK tree instead of being rejected as an incompatible endpoint.
        joint_state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.body_subscription = self.create_subscription(
            BodyData,
            str(self.get_parameter("body_feedback_topic").value),
            self._body_callback,
            feedback_qos,
        )
        self.left_subscription = self.create_subscription(
            ArmSlaveData,
            str(self.get_parameter("left_arm_feedback_topic").value),
            self._left_callback,
            feedback_qos,
        )
        self.right_subscription = self.create_subscription(
            ArmSlaveData,
            str(self.get_parameter("right_arm_feedback_topic").value),
            self._right_callback,
            feedback_qos,
        )

        joint_state_topic = str(self.get_parameter("joint_state_topic").value).strip()
        self.joint_state_publisher = None
        self.joint_state_subscription = None
        if joint_state_topic:
            self.joint_state_publisher = self.create_publisher(
                JointState,
                joint_state_topic,
                joint_state_qos,
            )
            self.joint_state_subscription = self.create_subscription(
                JointState,
                joint_state_topic,
                self._joint_state_callback,
                feedback_qos,
            )

        self.static_transform_broadcaster = StaticTransformBroadcaster(self)
        self.transform_broadcaster = TransformBroadcaster(self)
        self._publish_camera_mount_tf()
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_timer)
        self.get_logger().info(
            f"Publishing static camera mounts "
            f"{self.camera_mount_parent} -> {self.camera_mount_child} and "
            f"{self.right_camera_mount_parent} -> {self.right_camera_mount_child}; "
            f"using {urdf_file}"
        )
        if self.compatibility_frames_enabled:
            self.get_logger().info(
                "Publishing compatibility TF: "
                f"{self.compatibility_urdf_root_frame} -> "
                f"{self.compatibility_base_frame} (identity), and live "
                f"{self.compatibility_base_frame} -> "
                f"{self.compatibility_left_arm_base_frame}/"
                f"{self.compatibility_right_arm_base_frame}"
            )
        if self.joint_state_publisher is not None:
            self.get_logger().info(
                f"Republishing normalized body/arm feedback on {joint_state_topic}"
            )
            if self.publish_complete_joint_states:
                self.get_logger().info(
                    f"Publishing all {len(self.movable_joint_names)} URDF movable "
                    f"joints; unavailable groups use {self.default_joint_position:.3f} rad; "
                    f"neck mapping={self.neck_urdf_names}"
                )

    def _publish_camera_mount_tf(self) -> None:
        """Attach both camera-driver roots to their URDF Link8 frames."""

        transforms = (
            (
                self.camera_mount_parent,
                self.camera_mount_child,
                self.camera_mount,
            ),
            (
                self.right_camera_mount_parent,
                self.right_camera_mount_child,
                self.right_camera_mount,
            ),
        )
        messages = []
        for parent, child, transform in transforms:
            message = TransformStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = parent
            message.child_frame_id = child
            message.transform.translation.x = transform.translation[0]
            message.transform.translation.y = transform.translation[1]
            message.transform.translation.z = transform.translation[2]
            message.transform.rotation.x = transform.rotation[0]
            message.transform.rotation.y = transform.rotation[1]
            message.transform.rotation.z = transform.rotation[2]
            message.transform.rotation.w = transform.rotation[3]
            messages.append(message)
        if (
            self.compatibility_frames_enabled
            and self.compatibility_urdf_root_frame
            != self.compatibility_base_frame
        ):
            message = TransformStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.compatibility_urdf_root_frame
            message.child_frame_id = self.compatibility_base_frame
            message.transform.rotation.w = 1.0
            messages.append(message)
        self.static_transform_broadcaster.sendTransform(messages)

    def _publish_compatibility_arm_base_tf(
        self, joint_positions: Mapping[str, float]
    ) -> None:
        """Publish live aliases for the two SDK arm-base coordinate frames."""

        if not self.compatibility_frames_enabled:
            return
        left_transform, right_transform = compatibility_arm_base_transforms(
            self.kinematics,
            self.compatibility_urdf_root_frame,
            self.compatibility_chest_frame,
            joint_positions,
            self.compatibility_left_chest_to_arm_base,
            self.compatibility_right_chest_to_arm_base,
        )
        stamp = self.get_clock().now().to_msg()
        messages = []
        for child, transform in (
            (self.compatibility_left_arm_base_frame, left_transform),
            (self.compatibility_right_arm_base_frame, right_transform),
        ):
            message = TransformStamped()
            message.header.stamp = stamp
            message.header.frame_id = self.compatibility_base_frame
            message.child_frame_id = child
            message.transform.translation.x = transform.translation[0]
            message.transform.translation.y = transform.translation[1]
            message.transform.translation.z = transform.translation[2]
            message.transform.rotation.x = transform.rotation[0]
            message.transform.rotation.y = transform.rotation[1]
            message.transform.rotation.z = transform.rotation[2]
            message.transform.rotation.w = transform.rotation[3]
            messages.append(message)
        self.transform_broadcaster.sendTransform(messages)

    @staticmethod
    def _position_map(message: JointState) -> Optional[dict[str, float]]:
        if len(message.name) != len(message.position):
            return None
        values: dict[str, float] = {}
        for name, position in zip(message.name, message.position):
            value = float(position)
            if not math.isfinite(value):
                return None
            values[str(name)] = value
        return values

    @staticmethod
    def _ordered_values(
        positions: Mapping[str, float],
        input_names: Sequence[str],
        urdf_names: Sequence[str],
    ) -> Optional[list[float]]:
        candidates = [list(input_names), list(urdf_names)]
        for names in candidates:
            if names and all(name in positions for name in names):
                return [float(positions[name]) for name in names]
        if len(positions) == len(urdf_names):
            return [float(value) for value in positions.values()]
        return None

    def _store_group(
        self,
        group: str,
        urdf_names: Sequence[str],
        values: Optional[Sequence[float]],
    ) -> None:
        if values is None or len(values) != len(urdf_names):
            return
        if not all(math.isfinite(float(value)) for value in values):
            return
        with self._lock:
            self._joint_positions.update(
                {name: float(value) for name, value in zip(urdf_names, values)}
            )
            self._feedback_times[group] = time.monotonic()

    def _body_callback(self, message: BodyData) -> None:
        positions = self._position_map(message.joint_state)
        self._store_group(
            "body",
            self.body_urdf_names,
            self._ordered_values(
                positions or {},
                self.body_input_names,
                self.body_urdf_names,
            ),
        )
        neck_positions = self._position_map(message.neck_joint_state)
        self._store_group(
            "neck",
            self.neck_urdf_names,
            self._ordered_values(
                neck_positions or {},
                self.neck_input_names,
                self.neck_urdf_names,
            ),
        )

    def _left_callback(self, message: ArmSlaveData) -> None:
        positions = self._position_map(message.joint_state)
        self._store_group(
            "left",
            self.left_urdf_names,
            self._ordered_values(
                positions or {},
                self.left_input_names,
                self.left_urdf_names,
            ),
        )

    def _right_callback(self, message: ArmSlaveData) -> None:
        positions = self._position_map(message.joint_state)
        self._store_group(
            "right",
            self.right_urdf_names,
            self._ordered_values(
                positions or {},
                self.right_input_names,
                self.right_urdf_names,
            ),
        )

    def _joint_state_callback(self, message: JointState) -> None:
        positions = self._position_map(message)
        if positions is None:
            return

        body_aliases = [
            ("waist_1_joint", "joint1"),
            ("waist_2_joint", "joint2"),
            ("waist_3_joint", "joint3"),
            ("chest_joint", "joint4"),
        ]
        neck_aliases = [
            ("neck_Link", "neck_joint1", "joint5"),
            ("head_joint", "head_joint2", "joint6"),
        ]
        left_aliases = [
            (f"left_arm_{index}_joint", f"L_JOINT_{index}")
            for index in range(1, 8)
        ]
        right_aliases = [
            (f"right_arm_{index}_joint", f"R_JOINT_{index}")
            for index in range(1, 8)
        ]
        self._store_group(
            "body",
            self.body_urdf_names,
            self._values_from_aliases(positions, body_aliases),
        )
        self._store_group(
            "neck",
            self.neck_urdf_names,
            self._values_from_aliases(positions, neck_aliases),
        )
        self._store_group(
            "left",
            self.left_urdf_names,
            self._values_from_aliases(positions, left_aliases),
        )
        self._store_group(
            "right",
            self.right_urdf_names,
            self._values_from_aliases(positions, right_aliases),
        )

    @staticmethod
    def _values_from_aliases(
        positions: Mapping[str, float],
        aliases: Sequence[Sequence[str]],
    ) -> Optional[list[float]]:
        result: list[float] = []
        for names in aliases:
            value = next(
                (positions[name] for name in names if name in positions), None
            )
            if value is None:
                return None
            result.append(float(value))
        return result

    def _warn_waiting(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_time >= 5.0:
            self.get_logger().warning(message)
            self._last_warning_time = now

    def _publish_joint_states(
        self,
        joint_positions: Mapping[str, float],
        right_is_fresh: bool,
        neck_is_fresh: bool,
    ) -> None:
        """Expose split RealBot feedback to robot_state_publisher.

        The hardware publishes BodyData and ArmSlaveData, while
        robot_state_publisher consumes sensor_msgs/JointState. Keep the
        bridge in this node so the URDF FK and RViz model use the same joint
        name mapping as the global camera TF.
        """
        if self.joint_state_publisher is None:
            return

        if self.publish_complete_joint_states:
            # robot_state_publisher drops an entire branch when a revolute joint
            # has never appeared in JointState.  The hardware feedback topics do
            # not contain every URDF joint (for example steering and head joints),
            # so publish a complete vector.  Measured body/arm values win; an
            # unavailable group is held at the explicit visualization default.
            joint_names = list(self.movable_joint_names)
            values = []
            for name in joint_names:
                if (
                    name in self.right_urdf_names
                    and not right_is_fresh
                ) or (
                    name in self.neck_urdf_names
                    and not neck_is_fresh
                ):
                    values.append(self.default_joint_position)
                else:
                    values.append(
                        float(
                            joint_positions.get(name, self.default_joint_position)
                        )
                    )
        else:
            joint_names = list(self.body_urdf_names) + list(self.left_urdf_names)
            if right_is_fresh:
                joint_names.extend(self.right_urdf_names)
            if any(name not in joint_positions for name in joint_names):
                return
            values = [float(joint_positions[name]) for name in joint_names]

        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = joint_names
        message.position = values
        self.joint_state_publisher.publish(message)

    def _publish_timer(self) -> None:
        required = tuple(self.body_urdf_names) + tuple(self.left_urdf_names)
        with self._lock:
            missing = [name for name in required if name not in self._joint_positions]
            ages = {
                group: time.monotonic() - state_time
                for group, state_time in self._feedback_times.items()
            }
            if missing:
                self._warn_waiting(
                    "waiting for body/left-arm joint feedback; missing "
                    + ", ".join(missing)
                )
                return
            if ages["body"] > self.max_feedback_age_sec:
                self._warn_waiting(
                    f"body feedback is stale ({ages['body']:.3f}s); "
                    "global TF is paused"
                )
                return
            if ages["left"] > self.max_feedback_age_sec:
                self._warn_waiting(
                    f"left-arm feedback is stale ({ages['left']:.3f}s); "
                    "global TF is paused"
                )
                return
            joint_positions = dict(self._joint_positions)
            right_is_fresh = ages["right"] <= self.max_feedback_age_sec
            neck_is_fresh = ages["neck"] <= self.max_feedback_age_sec

        self._publish_joint_states(joint_positions, right_is_fresh, neck_is_fresh)
        self._publish_compatibility_arm_base_tf(joint_positions)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = RealbotsGlobalTf()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        # rclpy can raise during shutdown if a DDS sample is being converted
        # at the same time as Ctrl-C. Do not turn a normal stop into a second
        # shutdown error.
        if rclpy.ok():
            if node is not None:
                node.get_logger().error(f"ROS executor stopped: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


__all__ = [
    "IDENTITY",
    "RealbotsGlobalTf",
    "RigidTransform",
    "UrdfKinematics",
    "compose",
    "compatibility_arm_base_transforms",
    "inverse",
    "main",
]


if __name__ == "__main__":
    main()
