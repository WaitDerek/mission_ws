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
from pathlib import Path
from typing import Mapping, Optional, Sequence

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


from .global_tf_kinematics import (
    IDENTITY,
    RigidTransform,
    UrdfKinematics,
    _as_string_list,
    _clean_frame,
    _finite_vector,
    _normalize_quaternion,
    _rpy_to_quaternion,
    compatibility_arm_base_transforms,
    compose,
    inverse,
)


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
        self.declare_parameter("camera_mount_child_frame", "left_arm_depth_cam_link")
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
        self.declare_parameter("neck_urdf_joint_names", ["neck_Link", "head_joint"])
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
        self.declare_parameter("compatibility_left_arm_base_frame", "L_base_Link")
        self.declare_parameter("compatibility_right_arm_base_frame", "R_base_Link")
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
            raise ValueError(
                "publish_rate_hz and max_feedback_age_sec must be positive"
            )
        if not math.isfinite(self.default_joint_position):
            raise ValueError("default_joint_position must be finite")
        if len(self.body_input_names) != len(self.body_urdf_names):
            raise ValueError(
                "body input and URDF joint name arrays must have equal size"
            )
        if len(self.neck_input_names) != len(self.neck_urdf_names):
            raise ValueError(
                "neck input and URDF joint name arrays must have equal size"
            )
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
                        "compatibility frame must be present in the URDF: " f"{frame}"
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
                self.get_parameter("compatibility_right_chest_to_arm_base_xyz").value,
                3,
                "compatibility_right_chest_to_arm_base_xyz",
            )
            right_rpy = _finite_vector(
                self.get_parameter("compatibility_right_chest_to_arm_base_rpy").value,
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
            and self.compatibility_urdf_root_frame != self.compatibility_base_frame
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
            (f"left_arm_{index}_joint", f"L_JOINT_{index}") for index in range(1, 8)
        ]
        right_aliases = [
            (f"right_arm_{index}_joint", f"R_JOINT_{index}") for index in range(1, 8)
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
            value = next((positions[name] for name in names if name in positions), None)
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
                if (name in self.right_urdf_names and not right_is_fresh) or (
                    name in self.neck_urdf_names and not neck_is_fresh
                ):
                    values.append(self.default_joint_position)
                else:
                    values.append(
                        float(joint_positions.get(name, self.default_joint_position))
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
