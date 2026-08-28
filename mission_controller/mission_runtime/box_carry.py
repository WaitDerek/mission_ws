import json
import math
from copy import deepcopy
import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.duration import Duration
from rm_robot_interfaces.srv import StringCmd

try:
    from task_interfaces.srv import MoveCartesian
except ImportError:
    MoveCartesian = None
from tf2_ros import TransformException

try:
    from object_pose_interfaces.action import EstimateObjectPose
except ModuleNotFoundError:
    EstimateObjectPose = None

from .common import (
    MissionCanceled,
    MissionError,
    quaternion_multiply,
    rotate_vector,
)
from .realman_sdk_adapter import (
    RealManSdkCanceled,
    RealManSdkError,
    pose_to_sdk_target,
)


# Bound to the composed compatibility facade by box_support.py.
BoxSupportMixin = None


class BoxCarryMixin:
    """Post targets, endpoint synchronization, TF carry, and test place."""

    def _translate_pose_in_box_frame(
        self,
        target: Pose,
        delta_box_xyz: list[float],
        arm: str,
    ) -> Pose:
        """Translate a Link8 target by a delta expressed in the box frame."""
        if self._string("direct_movel_target_mode").strip().lower() != (
            "camera_offset_box_orientation"
        ):
            raise MissionError(
                "box-frame post MoveL requires "
                "direct_movel_target_mode=camera_offset_box_orientation"
            )
        prefix = "left" if arm == "left" else "right"
        link8_orientation = BoxSupportMixin._normalize_quaternion(
            (
                float(target.orientation.x),
                float(target.orientation.y),
                float(target.orientation.z),
                float(target.orientation.w),
            )
        )
        box_to_link8 = BoxSupportMixin._normalize_quaternion(
            tuple(self._float_array(f"direct_movel_{prefix}_box_to_link8_orientation"))
        )
        base_to_box = BoxSupportMixin._normalize_quaternion(
            quaternion_multiply(
                link8_orientation,
                BoxSupportMixin._quaternion_conjugate(box_to_link8),
            )
        )
        delta_box = tuple(float(value) for value in delta_box_xyz)
        if len(delta_box) != 3 or not all(math.isfinite(value) for value in delta_box):
            raise MissionError(f"invalid box-frame post MoveL delta for {arm}")
        delta_base = rotate_vector(delta_box, base_to_box)
        result = deepcopy(target)
        result.position.x += delta_base[0]
        result.position.y += delta_base[1]
        result.position.z += delta_base[2]
        return result

    def _post_movel_targets(
        self,
        left_target: Pose,
        right_target: Pose,
    ) -> list[tuple[Pose, Pose]]:
        """Return standard cumulative MoveL targets from box-frame deltas."""
        return [
            (left_step, right_step)
            for _label, left_step, right_step in BoxSupportMixin._post_movel_targets_with_labels(
                self, left_target, right_target, include_drag_steps=False
            )
        ]

    def _post_movel_xyz_parameter_name(
        self,
        arm: str,
        step: int,
        model_label: str | None,
        *,
        box_layer: int,
        tf_mode: bool,
        drag_mode: bool,
    ) -> str:
        """Resolve a standard post-grasp delta parameter.

        The legacy GraspBox/DragBox paths intentionally keep their existing
        parameter names.  TF paths use an action/model/layer-specific name so
        their four layer profiles can be tuned independently.
        """
        if arm not in ("left", "right") or not 1 <= step <= 5:
            raise MissionError("arm must be left/right and step must be in [1, 5]")
        if tf_mode:
            action_prefix = "drag_box_tf" if drag_mode else "grasp_box_tf"
            return BoxSupportMixin._tf_layer_parameter_name(
                action_prefix,
                f"post_movel_{arm}_step{step}_xyz",
                model_label,
                box_layer,
            )
        parameter_name = f"box_post_movel_{arm}_step{step}_xyz"
        if step == 1 and str(model_label or "").strip().lower() == "smallbox":
            parameter_name += "_smallbox"
        return parameter_name

    def _drag_post_movel_xyz_parameter_name(
        self,
        arm: str,
        drag_index: int,
        model_label: str | None,
        *,
        box_layer: int,
        tf_mode: bool,
        drag_mode: bool,
    ) -> str:
        """Resolve a DragBox drag delta parameter with TF isolation."""
        if arm not in ("left", "right") or not 1 <= drag_index <= 3:
            raise MissionError(
                "arm must be left/right and drag_index must be in [1, 3]"
            )
        if tf_mode and drag_mode:
            return BoxSupportMixin._tf_layer_parameter_name(
                "drag_box_tf",
                f"post_movel_step_drag{drag_index}_{arm}_xyz",
                model_label,
                box_layer,
            )
        return f"drag_box_post_movel_step_drag{drag_index}_{arm}_xyz"

    def _post_movel_targets_with_labels(
        self,
        left_target: Pose,
        right_target: Pose,
        *,
        include_drag_steps: bool,
        defer_left_step1: bool = False,
        model_label: str | None = None,
        box_layer: int = 1,
        tf_mode: bool = False,
    ) -> list[tuple[str, Pose, Pose]]:
        """Return cumulative targets, optionally inserting drag-only steps.

        DragBox can defer the left arm's Step1 until after the delayed left
        join. In that mode the left cumulative target is built from the drag
        deltas only, then a separate ``step1_left`` target is inserted before
        Step2 so Step1 is executed after the left arm has joined.
        """
        targets = []
        left_current = deepcopy(left_target)
        right_current = deepcopy(right_target)
        step_count = self._integer("box_post_movel_step_count")
        for index in range(1, step_count + 1):
            left_parameter = BoxSupportMixin._post_movel_xyz_parameter_name(
                self,
                "left",
                index,
                model_label,
                box_layer=box_layer,
                tf_mode=tf_mode,
                drag_mode=include_drag_steps,
            )
            right_parameter = BoxSupportMixin._post_movel_xyz_parameter_name(
                self,
                "right",
                index,
                model_label,
                box_layer=box_layer,
                tf_mode=tf_mode,
                drag_mode=include_drag_steps,
            )
            if not (defer_left_step1 and include_drag_steps and index == 1):
                left_current = self._translate_pose_in_box_frame(
                    left_current,
                    self._float_array(left_parameter),
                    "left",
                )
            right_current = self._translate_pose_in_box_frame(
                right_current,
                self._float_array(right_parameter),
                "right",
            )
            targets.append((f"step{index}", left_current, right_current))
            if include_drag_steps and index == 1:
                for drag_index in range(1, 4):
                    left_current = self._translate_pose_in_box_frame(
                        left_current,
                        self._float_array(
                            BoxSupportMixin._drag_post_movel_xyz_parameter_name(
                                self,
                                "left",
                                drag_index,
                                model_label,
                                box_layer=box_layer,
                                tf_mode=tf_mode,
                                drag_mode=include_drag_steps,
                            )
                        ),
                        "left",
                    )
                    right_current = self._translate_pose_in_box_frame(
                        right_current,
                        self._float_array(
                            BoxSupportMixin._drag_post_movel_xyz_parameter_name(
                                self,
                                "right",
                                drag_index,
                                model_label,
                                box_layer=box_layer,
                                tf_mode=tf_mode,
                                drag_mode=include_drag_steps,
                            )
                        ),
                        "right",
                    )
                    targets.append(
                        (
                            f"step_drag{drag_index}",
                            left_current,
                            right_current,
                        )
                    )
                if defer_left_step1:
                    left_current = self._translate_pose_in_box_frame(
                        left_current,
                        self._float_array(left_parameter),
                        "left",
                    )
                    targets.append(("step1_left", left_current, right_current))
        return targets

    @staticmethod
    def _endpoint_sync_pose_values_to_transform(values):
        if values is None or len(values) != 7:
            raise MissionError("Link7 EEPose must contain seven values")
        return (
            tuple(float(value) for value in values[:3]),
            BoxSupportMixin._normalize_quaternion(values[3:7]),
        )

    @staticmethod
    def _endpoint_sync_transform_to_pose(transform) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = transform[0]
        pose.orientation.x, pose.orientation.y = transform[1][:2]
        pose.orientation.z, pose.orientation.w = transform[1][2:]
        return pose

    @staticmethod
    def _endpoint_sync_pose_position_error(lhs, rhs) -> float:
        return math.sqrt(
            sum((float(lhs[index]) - float(rhs[index])) ** 2 for index in range(3))
        )

    @staticmethod
    def _endpoint_sync_pose_orientation_error(lhs, rhs) -> float:
        lhs_q = BoxSupportMixin._normalize_quaternion(lhs)
        rhs_q = BoxSupportMixin._normalize_quaternion(rhs)
        dot = abs(sum(lhs_q[index] * rhs_q[index] for index in range(4)))
        return 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    def _step2_endpoint_sync_layer_config(self, box_layer: int):
        if box_layer < 1 or box_layer > 4:
            raise MissionError("box_layer must be in [1, 4]")
        prefix = f"box_step2_waist_endpoint_sync_layer{box_layer}_"
        if not self._boolean(f"{prefix}configured"):
            raise MissionError(
                f"box_layer {box_layer} endpoint waist sync is not configured"
            )
        values = {
            "segments": self._integer(f"{prefix}segments"),
            "forward_body_velocity": self._integer(f"{prefix}forward_body_velocity"),
            "forward_left_speed": self._float(
                f"{prefix}forward_left_movel_velocity_percent"
            ),
            "forward_right_speed": self._float(
                f"{prefix}forward_right_movel_velocity_percent"
            ),
            "reverse_body_velocity": self._integer(f"{prefix}reverse_body_velocity"),
            "reverse_left_speed": self._float(
                f"{prefix}reverse_left_movel_velocity_percent"
            ),
            "reverse_right_speed": self._float(
                f"{prefix}reverse_right_movel_velocity_percent"
            ),
        }
        if not 1 <= values["forward_body_velocity"] <= 100:
            raise MissionError(f"{prefix}forward_body_velocity must be in [1,100]")
        if not 1 <= values["reverse_body_velocity"] <= 100:
            raise MissionError(f"{prefix}reverse_body_velocity must be in [1,100]")
        if values["segments"] not in (1, 2):
            raise MissionError(f"{prefix}segments must be 1 or 2")
        for key in (
            "forward_left_speed",
            "forward_right_speed",
            "reverse_left_speed",
            "reverse_right_speed",
        ):
            if not 1.0 <= values[key] <= 100.0:
                raise MissionError(f"{prefix}{key} must be in [1,100]")
        return values

    def _wait_for_fresh_endpoint_arm_poses(self, goal_handle, sequence_after):
        timeout_sec = self._float("box_step2_waist_endpoint_sync_timeout_sec")
        max_age_sec = self._float("box_step2_waist_endpoint_sync_feedback_max_age_sec")
        deadline = time.monotonic() + timeout_sec
        latest = {}
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while reading Step2 endpoint EEPose")
            complete = True
            now = time.monotonic()
            with self.joint_state_lock:
                for arm in ("left", "right"):
                    values = self.latest_slave_arm_poses.get(arm)
                    sequence = self.latest_slave_arm_pose_sequences.get(arm, 0)
                    age = now - self.latest_slave_arm_pose_times.get(arm, 0.0)
                    latest[arm] = (values, sequence, age)
                    if (
                        sequence <= sequence_after.get(arm, -1)
                        or values is None
                        or age > max_age_sec
                    ):
                        complete = False
            if complete:
                return {arm: tuple(latest[arm][0]) for arm in ("left", "right")}, {
                    arm: latest[arm][1] for arm in ("left", "right")
                }
            time.sleep(0.02)
        detail = ", ".join(
            f"{arm}=sequence={item[1]}, pose={'present' if item[0] is not None else 'missing'}, age={item[2]:.3f}"
            for arm, item in latest.items()
        )
        raise MissionError(
            "fresh Step2 endpoint EEPose was not available for both arms: "
            f"{detail}, timeout_sec={timeout_sec:.1f}"
        )

    def _step2_endpoint_sync_body_request(self, body_units, velocity):
        device = self._integer("box_joint1_device")
        payload = {
            "device": device,
            "payload": {
                "command": "movej",
                "device": device,
                "joint": [int(value) for value in body_units],
                "v": int(velocity),
                "r": self._integer("box_step2_waist_endpoint_sync_body_blend_radius"),
                "trajectory_connect": 0,
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        return request

    def _step2_endpoint_sync_stop_body(self):
        if not self._boolean("box_step2_waist_endpoint_sync_body_stop_enabled"):
            return
        try:
            if not self.body_command_client.service_is_ready():
                self.body_command_client.wait_for_service(timeout_sec=0.5)
            if not self.body_command_client.service_is_ready():
                raise MissionError("body command service is unavailable")
            device = self._integer("box_joint1_device")
            payload = {
                "device": device,
                "payload": {
                    "command": self._string(
                        "box_step2_waist_endpoint_sync_body_stop_command"
                    ),
                    "device": device,
                },
            }
            request = StringCmd.Request()
            request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
            self.body_command_client.call_async(request)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Step2 endpoint body stop failed: {exc}")

    @staticmethod
    def _pose_stamped_to_transform(pose: PoseStamped):
        return (
            (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            ),
            BoxSupportMixin._normalize_quaternion(
                (
                    float(pose.pose.orientation.x),
                    float(pose.pose.orientation.y),
                    float(pose.pose.orientation.z),
                    float(pose.pose.orientation.w),
                )
            ),
        )

    def _lookup_tf_carry_transform(self, target_frame: str, source_frame: str):
        """Read one latest transform used by the TF waist-carry controller."""
        target_frame = target_frame.strip().lstrip("/")
        source_frame = source_frame.strip().lstrip("/")
        try:
            message = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(
                    seconds=self._float("grasp_box_tf_body_home_carry_tf_timeout_sec")
                ),
            )
        except TransformException as exc:
            raise MissionError(
                f"TF waist carry requires {target_frame} -> {source_frame}: {exc}"
            ) from exc
        translation = message.transform.translation
        rotation = message.transform.rotation
        return (
            (float(translation.x), float(translation.y), float(translation.z)),
            BoxSupportMixin._normalize_quaternion(
                (
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                    float(rotation.w),
                )
            ),
        )

    @staticmethod
    def _mean_rigid_transforms(lhs, rhs):
        """Return an equal-weight pose mean, resolving quaternion sign first."""
        position = tuple((lhs[0][i] + rhs[0][i]) * 0.5 for i in range(3))
        rhs_q = rhs[1]
        if sum(lhs[1][i] * rhs_q[i] for i in range(4)) < 0.0:
            rhs_q = tuple(-value for value in rhs_q)
        quaternion = BoxSupportMixin._normalize_quaternion(
            tuple(lhs[1][i] + rhs_q[i] for i in range(4))
        )
        return position, quaternion

    def _tf_carry_body_request(
        self,
        body_units,
        velocity,
        *,
        blend_radius=None,
        trajectory_connect=0,
    ):
        if blend_radius is None:
            blend_radius = self._integer(
                "grasp_box_tf_body_home_carry_body_blend_radius"
            )
        blend_radius = int(round(float(blend_radius)))
        if not 0 <= blend_radius <= 100:
            raise MissionError(
                "grasp_box_tf_body_home_carry_body_blend_radius must be in [0,100]"
            )
        trajectory_connect = int(trajectory_connect)
        if trajectory_connect not in (0, 1):
            raise MissionError("TF waist-carry trajectory_connect must be 0 or 1")
        device = self._integer("box_joint1_device")
        payload = {
            "device": device,
            "payload": {
                "command": "movej",
                "device": device,
                "joint": [int(value) for value in body_units],
                "v": int(velocity),
                "r": blend_radius,
                "trajectory_connect": trajectory_connect,
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        return request

    def _tf_carry_stop_body(self):
        if not self._boolean("grasp_box_tf_body_home_carry_body_stop_enabled"):
            return
        try:
            if not self.body_command_client.service_is_ready():
                self.body_command_client.wait_for_service(timeout_sec=0.5)
            if not self.body_command_client.service_is_ready():
                return
            device = self._integer("box_joint1_device")
            payload = {
                "device": device,
                "payload": {
                    "command": self._string(
                        "grasp_box_tf_body_home_carry_body_stop_command"
                    ),
                    "device": device,
                },
            }
            request = StringCmd.Request()
            request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
            self.body_command_client.call_async(request)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"TF waist carry body stop failed: {exc}")

    def _wait_for_tf_carry_world_targets(self, goal_handle, targets_by_arm):
        """Verify actual configured TCP world TF against both rigid-grasp targets."""
        timeout_sec = self._float("grasp_box_tf_body_home_carry_timeout_sec")
        position_limit = self._float(
            "grasp_box_tf_body_home_carry_position_tolerance_m"
        )
        orientation_limit = self._float(
            "grasp_box_tf_body_home_carry_orientation_tolerance_rad"
        )
        required_stable = self._integer("grasp_box_tf_body_home_carry_stable_samples")
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        latest_detail = "no TF sample"
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while verifying TF waist-carry TCP targets"
            )
            matches = True
            details = []
            for arm in ("left", "right"):
                tcp_frame = self._string(f"{arm}_link8_frame").strip().lstrip("/")
                actual = self._lookup_tf_carry_transform(base_frame, tcp_frame)
                target = targets_by_arm[arm]
                position_error = self._endpoint_sync_pose_position_error(
                    actual[0], target[0]
                )
                orientation_error = self._endpoint_sync_pose_orientation_error(
                    actual[1], target[1]
                )
                details.append(
                    f"{arm}_position_error={position_error:.4f}m,"
                    f"{arm}_orientation_error={orientation_error:.4f}rad"
                )
                if (
                    position_error > position_limit
                    or orientation_error > orientation_limit
                ):
                    matches = False
            latest_detail = "; ".join(details)
            stable_samples = stable_samples + 1 if matches else 0
            if stable_samples >= required_stable:
                return latest_detail
            time.sleep(0.02)
        raise MissionError(
            "TF waist-carry TCP targets were not reached: "
            f"{latest_detail}, timeout_sec={timeout_sec:.1f}"
        )

    def _execute_tf_body_home_carry_continuous(
        self,
        goal_handle,
        adapter,
        *,
        segments,
        home_units,
        body_start,
        body_sequence,
        relation_by_arm,
        current_carrier,
        actual_link,
        current_box,
        base_frame,
        carrier_frame,
        units_per_degree,
        left_speed,
        right_speed,
        body_velocity,
        timeout_sec,
    ) -> str:
        """Execute one connected waist/dual-arm trajectory without stops.

        All intermediate waist and arm waypoints are queued with
        ``trajectory_connect=1``.  The final waypoint is submitted with
        ``trajectory_connect=0`` behind a barrier, after the final waist
        command has been accepted.  The box orientation and box-to-TCP poses
        remain fixed in the chassis/world frame for the complete path.
        """
        if segments < 1:
            raise MissionError(
                "grasp_box_tf_body_home_carry_segments must be at least 1"
            )
        arm_blend_radius = self._integer(
            "grasp_box_tf_body_home_carry_arm_blend_radius"
        )
        body_blend_radius = self._integer(
            "grasp_box_tf_body_home_carry_body_blend_radius"
        )
        if not 0 <= arm_blend_radius <= 100:
            raise MissionError(
                "grasp_box_tf_body_home_carry_arm_blend_radius must be in [0,100]"
            )
        if not 0 <= body_blend_radius <= 100:
            raise MissionError(
                "grasp_box_tf_body_home_carry_body_blend_radius must be in [0,100]"
            )

        live_carrier = current_carrier
        live_arm_base = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        box_world_orientation = current_box[1]
        carrier_to_box_position = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(live_carrier), current_box
        )[0]
        # Keep all four body joints for command-space interpolation.  Joint4 is
        # preserved by the home trajectory even though only Joint1--3 affect
        # the configured chest/arm-base FK chain.
        start_body_angles = [float(value) for value in body_start[:4]]
        start_joint123_angles = start_body_angles[:3]
        left_targets = []
        right_targets = []
        world_targets_by_segment = []
        body_targets = []
        for segment_index in range(1, segments + 1):
            fraction = float(segment_index) / float(segments)
            target_units = [
                (
                    home_units[index]
                    if segment_index == segments
                    else int(
                        round(
                            math.degrees(start_body_angles[index])
                            * units_per_degree[index]
                            + fraction
                            * (
                                home_units[index]
                                - math.degrees(start_body_angles[index])
                                * units_per_degree[index]
                            )
                        )
                    )
                )
                for index in range(4)
            ]
            target_angles = [
                math.radians(float(target_units[index]) / units_per_degree[index])
                for index in range(4)
            ]
            fk_current_chest = self._joint123_chest_transform(start_joint123_angles)
            fk_target_chest = self._joint123_chest_transform(target_angles[:3])
            future_carrier = BoxSupportMixin._compose_transform(
                live_carrier,
                BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(fk_current_chest),
                    fk_target_chest,
                ),
            )
            future_box = (
                BoxSupportMixin._compose_transform(
                    future_carrier,
                    (carrier_to_box_position, (0.0, 0.0, 0.0, 1.0)),
                )[0],
                box_world_orientation,
            )
            world_targets = {
                arm: BoxSupportMixin._compose_transform(
                    future_box, relation_by_arm[arm]
                )
                for arm in ("left", "right")
            }
            arm_targets = {}
            for arm in ("left", "right"):
                fk_current_arm = self._joint123_arm_base_transform(
                    arm, start_joint123_angles
                )
                fk_target_arm = self._joint123_arm_base_transform(
                    arm, target_angles[:3]
                )
                future_arm_base = BoxSupportMixin._compose_transform(
                    live_arm_base[arm],
                    BoxSupportMixin._compose_transform(
                        BoxSupportMixin._inverse_transform(fk_current_arm),
                        fk_target_arm,
                    ),
                )
                arm_targets[arm] = self._endpoint_sync_transform_to_pose(
                    BoxSupportMixin._compose_transform(
                        BoxSupportMixin._inverse_transform(future_arm_base),
                        world_targets[arm],
                    )
                )
            left_targets.append(pose_to_sdk_target(arm_targets["left"]))
            right_targets.append(pose_to_sdk_target(arm_targets["right"]))
            world_targets_by_segment.append(world_targets)
            body_targets.append((target_units, target_angles))

        self._publish_box_grasp_feedback(
            goal_handle,
            "TF_BODY_HOME_CARRY_CONNECTED",
            "queuing connected waist/dual-arm trajectory: "
            f"segments={segments}; arm_blend_radius={arm_blend_radius}; "
            f"body_blend_radius={body_blend_radius}; "
            "trajectory_connect=1 for intermediate points, 0 for final; "
            "intermediate_stop=false",
        )
        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)

        # Queue the body intermediate points before the arm adapter starts the
        # final point.  A connect=1 request is accepted immediately and is
        # intentionally not waited on as a physical stop.
        for index in range(max(0, segments - 1)):
            self._check_canceled(
                goal_handle, "while queuing connected waist trajectory"
            )
            future = self.body_command_client.call_async(
                self._tf_carry_body_request(
                    body_targets[index][0],
                    body_velocity,
                    blend_radius=body_blend_radius,
                    trajectory_connect=1,
                )
            )
            response = self._wait_future(
                future,
                goal_handle,
                f"queuing TF waist-carry waypoint {index + 1}",
                self._float("dependency_wait_timeout_sec"),
                cancel_local_future=False,
            )
            self._parse_string_command_response(
                response, f"TF waist-carry waypoint {index + 1} MoveJ"
            )

        final_body_future = {}

        def release_body():
            final_body_future["future"] = self.body_command_client.call_async(
                self._tf_carry_body_request(
                    body_targets[-1][0],
                    body_velocity,
                    blend_radius=0,
                    trajectory_connect=0,
                )
            )

        # A lightweight rigid-grasp guard runs while the connected arm path is
        # active.  Transient TF lookup gaps are tolerated; the final guard is
        # authoritative and still requires fresh stable samples at the home
        # endpoint.
        last_monitor_time = [0.0]
        position_limit = self._float(
            "grasp_box_tf_body_home_carry_position_tolerance_m"
        )
        orientation_limit = self._float(
            "grasp_box_tf_body_home_carry_orientation_tolerance_rad"
        )

        def monitor_connected_path():
            now = time.monotonic()
            if now - last_monitor_time[0] < 0.1:
                return
            last_monitor_time[0] = now
            try:
                actual = {
                    arm: self._lookup_tf_carry_transform(
                        base_frame,
                        self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                    )
                    for arm in ("left", "right")
                }
            except MissionError:
                return
            inferred = {
                arm: BoxSupportMixin._compose_transform(
                    actual[arm],
                    BoxSupportMixin._inverse_transform(relation_by_arm[arm]),
                )
                for arm in ("left", "right")
            }
            position_error = self._endpoint_sync_pose_position_error(
                inferred["left"][0], inferred["right"][0]
            )
            orientation_error = self._endpoint_sync_pose_orientation_error(
                inferred["left"][1], inferred["right"][1]
            )
            if position_error > max(0.05, position_limit * 5.0):
                raise MissionError(
                    "connected TF waist-carry rigid-grasp monitor exceeded "
                    f"position tolerance: {position_error:.4f}m"
                )
            if orientation_error > max(0.5, orientation_limit * 5.0):
                raise MissionError(
                    "connected TF waist-carry rigid-grasp monitor exceeded "
                    f"orientation tolerance: {orientation_error:.4f}rad"
                )

        try:
            motion_result = adapter.execute_dual_movel_connected_waypoints(
                left_targets,
                right_targets,
                left_speed,
                right_speed,
                blend_radius=arm_blend_radius,
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=timeout_sec,
                before_start=release_body,
                abort_callback=self._tf_carry_stop_body,
                progress_callback=monitor_connected_path,
            )
            if "future" not in final_body_future:
                raise MissionError("TF waist-carry final body command was not released")
            response = self._wait_future(
                final_body_future["future"],
                goal_handle,
                "starting connected TF waist-carry body MoveJ",
                self._float("dependency_wait_timeout_sec"),
                cancel_local_future=False,
            )
            self._parse_string_command_response(
                response, "connected TF waist-carry final body MoveJ"
            )
            final_angles = body_targets[-1][1]
            self._wait_for_body_joints_target(
                goal_handle,
                final_angles,
                sequence_after=body_sequence,
                timeout_parameter="grasp_box_tf_body_home_carry_timeout_sec",
            )
            last_world_targets = world_targets_by_segment[-1]
            verification = self._wait_for_tf_carry_world_targets(
                goal_handle, last_world_targets
            )
            live_arm_base = {
                arm: self._lookup_tf_carry_transform(
                    base_frame,
                    self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
                )
                for arm in ("left", "right")
            }
            correction_targets = {
                arm: self._endpoint_sync_transform_to_pose(
                    BoxSupportMixin._compose_transform(
                        BoxSupportMixin._inverse_transform(live_arm_base[arm]),
                        last_world_targets[arm],
                    )
                )
                for arm in ("left", "right")
            }
            if self._boolean("grasp_box_tf_body_home_carry_final_correction_enabled"):
                correction_speed = self._float(
                    "grasp_box_tf_body_home_carry_final_correction_velocity_percent"
                )
                adapter.execute_dual_movel_endpoint(
                    pose_to_sdk_target(correction_targets["left"]),
                    pose_to_sdk_target(correction_targets["right"]),
                    correction_speed,
                    correction_speed,
                    cancel_requested=lambda: goal_handle.is_cancel_requested,
                    timeout_sec=timeout_sec,
                )
                verification = self._wait_for_tf_carry_world_targets(
                    goal_handle, last_world_targets
                )
            self._last_tf_body_home_carry_arm_targets = correction_targets
            self._last_tf_body_home_carry_completed = True
            return (
                "tf_body_home_carry=completed; mode=continuous; "
                f"segments={segments}; home_joint_units={home_units}; "
                f"{motion_result}; final_guard={verification}; "
                "box_translation=follows_chest; box_world_orientation=fixed; "
                "box_to_TCP=fixed; intermediate_stop=false"
            )
        except (RealManSdkCanceled, MissionCanceled):
            self._tf_carry_stop_body()
            raise
        except (RealManSdkError, MissionError, ValueError) as exc:
            self._tf_carry_stop_body()
            raise MissionError(f"TF waist home carry failed: {exc}") from exc

    def _place_box_test_body_request(
        self,
        body_units,
        *,
        trajectory_connect,
        blend_radius,
    ):
        trajectory_connect = int(trajectory_connect)
        blend_radius = int(blend_radius)
        if trajectory_connect not in (0, 1):
            raise MissionError("place_box_test trajectory_connect must be 0 or 1")
        if not 0 <= blend_radius <= 100:
            raise MissionError("place_box_test body blend radius must be in [0,100]")
        device = self._integer("box_joint1_device")
        payload = {
            "device": device,
            "payload": {
                "command": "movej",
                "device": device,
                "joint": [int(value) for value in body_units],
                "v": self._integer("place_box_test_body_velocity"),
                "r": blend_radius,
                "trajectory_connect": trajectory_connect,
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        return request

    def _place_box_test_stop_body(self):
        if not self._boolean("place_box_test_body_stop_enabled"):
            return
        try:
            if not self.body_command_client.service_is_ready():
                self.body_command_client.wait_for_service(timeout_sec=0.5)
            if not self.body_command_client.service_is_ready():
                return
            device = self._integer("box_joint1_device")
            payload = {
                "device": device,
                "payload": {
                    "command": self._string("place_box_test_body_stop_command"),
                    "device": device,
                },
            }
            request = StringCmd.Request()
            request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
            self.body_command_client.call_async(request)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"place_box_test body stop failed: {exc}")

    def _wait_for_place_box_test_world_targets(self, goal_handle, targets_by_arm):
        timeout_sec = self._float("place_box_test_timeout_sec")
        position_limit = self._float("place_box_test_position_tolerance_m")
        orientation_limit = self._float("place_box_test_orientation_tolerance_rad")
        required_stable = self._integer("place_box_test_stable_samples")
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        latest_detail = "no TF sample"
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while verifying place_box_test targets")
            matches = True
            details = []
            for arm in ("left", "right"):
                actual = self._lookup_tf_carry_transform(
                    base_frame,
                    self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                )
                target = targets_by_arm[arm]
                position_error = self._endpoint_sync_pose_position_error(
                    actual[0], target[0]
                )
                orientation_error = self._endpoint_sync_pose_orientation_error(
                    actual[1], target[1]
                )
                details.append(
                    f"{arm}_position_error={position_error:.4f}m,"
                    f"{arm}_orientation_error={orientation_error:.4f}rad"
                )
                if (
                    position_error > position_limit
                    or orientation_error > orientation_limit
                ):
                    matches = False
            latest_detail = "; ".join(details)
            stable_samples = stable_samples + 1 if matches else 0
            if stable_samples >= required_stable:
                return latest_detail
            time.sleep(0.02)
        raise MissionError(
            "place_box_test Link7 targets were not reached: "
            f"{latest_detail}, timeout_sec={timeout_sec:.1f}"
        )

    def _execute_place_box_test_motion(self, goal_handle, adapter, dry_run: bool):
        """Move the waist and both TCPs to the taught small-box place pose.

        A single virtual box pose is interpolated from the transported pose to
        the taught placement pose.  Both Link7 targets are derived from that
        same box pose, so the arms cannot independently twist the held box.
        """
        relation_by_arm = getattr(self, "_last_grasp_box_tf_box_to_link7_targets", None)
        if not relation_by_arm:
            raise MissionError(
                "place_box_test requires a preceding /grasp_box_tf goal in "
                "the same mission_controller process"
            )
        if adapter is None and not dry_run:
            raise MissionError(
                "place_box_test requires direct_motion_backend=python_sdk"
            )

        body_start, body_velocities, body_sequence = self._wait_for_fresh_body_feedback(
            goal_handle
        )
        units_per_degree = self._float_array("box_body_command_units_per_degree")
        start_home_units = [
            int(round(value))
            for value in self._float_array("place_box_test_start_body_joint_units")
        ]
        start_home_angles = [
            math.radians(float(start_home_units[index]) / units_per_degree[index])
            for index in range(4)
        ]
        start_tolerance = self._float("place_box_test_start_body_tolerance_rad")
        if any(
            abs(body_start[index] - start_home_angles[index]) > start_tolerance
            for index in range(4)
        ):
            raise MissionError(
                "place_box_test must start with the waist at its carried-home "
                f"pose: expected={start_home_angles}, measured={body_start}, "
                f"tolerance_rad={start_tolerance:.4f}"
            )
        velocity_limit = self._float("box_joint1_velocity_tolerance_rad_sec")
        if any(abs(value) > velocity_limit for value in body_velocities):
            raise MissionError(
                "place_box_test requires a stationary waist: "
                f"velocities={body_velocities}, limit={velocity_limit:.4f}"
            )

        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        actual_link = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_link8_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        inferred_current_box = {
            arm: BoxSupportMixin._compose_transform(
                actual_link[arm],
                BoxSupportMixin._inverse_transform(relation_by_arm[arm]),
            )
            for arm in ("left", "right")
        }
        current_position_error = self._endpoint_sync_pose_position_error(
            inferred_current_box["left"][0], inferred_current_box["right"][0]
        )
        current_orientation_error = self._endpoint_sync_pose_orientation_error(
            inferred_current_box["left"][1], inferred_current_box["right"][1]
        )
        consistency_position_limit = self._float(
            "place_box_test_target_consistency_position_tolerance_m"
        )
        consistency_orientation_limit = self._float(
            "place_box_test_target_consistency_orientation_tolerance_rad"
        )
        if (
            current_position_error > consistency_position_limit
            or current_orientation_error > consistency_orientation_limit
        ):
            raise MissionError(
                "saved grasp is inconsistent with the current dual-arm TF: "
                f"box_position_disagreement={current_position_error:.4f}m, "
                f"box_orientation_disagreement={current_orientation_error:.4f}rad"
            )
        current_box = BoxSupportMixin._mean_rigid_transforms(
            inferred_current_box["left"], inferred_current_box["right"]
        )
        # Re-capture the physical rigid grasp at the destination.  This
        # removes small endpoint tracking error accumulated during transport.
        relation_by_arm = {
            arm: BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(current_box), actual_link[arm]
            )
            for arm in ("left", "right")
        }
        self._last_grasp_box_tf_box_to_link7_targets = relation_by_arm

        target_units = [
            int(round(value))
            for value in self._float_array("place_box_test_body_joint_units")
        ]
        target_angles = [
            math.radians(float(target_units[index]) / units_per_degree[index])
            for index in range(4)
        ]
        live_arm_base = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        start_angles = [float(value) for value in body_start[:3]]
        future_arm_base = {}
        for arm in ("left", "right"):
            fk_current = self._joint123_arm_base_transform(arm, start_angles)
            fk_target = self._joint123_arm_base_transform(arm, target_angles[:3])
            future_arm_base[arm] = BoxSupportMixin._compose_transform(
                live_arm_base[arm],
                BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(fk_current), fk_target
                ),
            )

        configured_target = {
            arm: self._endpoint_sync_pose_values_to_transform(
                self._float_array(f"place_box_test_{arm}_target_pose_arm_base")
            )
            for arm in ("left", "right")
        }
        target_world_link = {
            arm: BoxSupportMixin._compose_transform(
                future_arm_base[arm], configured_target[arm]
            )
            for arm in ("left", "right")
        }
        inferred_target_box = {
            arm: BoxSupportMixin._compose_transform(
                target_world_link[arm],
                BoxSupportMixin._inverse_transform(relation_by_arm[arm]),
            )
            for arm in ("left", "right")
        }
        target_position_error = self._endpoint_sync_pose_position_error(
            inferred_target_box["left"][0], inferred_target_box["right"][0]
        )
        target_orientation_error = self._endpoint_sync_pose_orientation_error(
            inferred_target_box["left"][1], inferred_target_box["right"][1]
        )
        if (
            target_position_error > consistency_position_limit
            or target_orientation_error > consistency_orientation_limit
        ):
            raise MissionError(
                "taught left/right place poses do not describe one rigid box pose: "
                f"position_disagreement={target_position_error:.4f}m, "
                f"orientation_disagreement={target_orientation_error:.4f}rad, "
                f"limits=[{consistency_position_limit:.4f}m,"
                f"{consistency_orientation_limit:.4f}rad]"
            )
        target_box = BoxSupportMixin._mean_rigid_transforms(
            inferred_target_box["left"], inferred_target_box["right"]
        )

        segments = self._integer("place_box_test_segments")
        arm_targets = {"left": [], "right": []}
        world_targets_by_segment = []
        body_targets = []
        start_units = [
            int(round(math.degrees(body_start[index]) * units_per_degree[index]))
            for index in range(4)
        ]
        for segment_index in range(1, segments + 1):
            fraction = float(segment_index) / float(segments)
            body_units = [
                (
                    target_units[index]
                    if segment_index == segments
                    else int(
                        round(
                            start_units[index]
                            + fraction * (target_units[index] - start_units[index])
                        )
                    )
                )
                for index in range(4)
            ]
            body_angles = [
                math.radians(float(body_units[index]) / units_per_degree[index])
                for index in range(4)
            ]
            box_pose = (
                tuple(
                    current_box[0][index]
                    + fraction * (target_box[0][index] - current_box[0][index])
                    for index in range(3)
                ),
                BoxSupportMixin._slerp_quaternion(
                    current_box[1], target_box[1], fraction
                ),
            )
            world_targets = {
                arm: BoxSupportMixin._compose_transform(box_pose, relation_by_arm[arm])
                for arm in ("left", "right")
            }
            world_targets_by_segment.append(world_targets)
            body_targets.append((body_units, body_angles))
            for arm in ("left", "right"):
                fk_current = self._joint123_arm_base_transform(arm, start_angles)
                fk_waypoint = self._joint123_arm_base_transform(arm, body_angles[:3])
                waypoint_arm_base = BoxSupportMixin._compose_transform(
                    live_arm_base[arm],
                    BoxSupportMixin._compose_transform(
                        BoxSupportMixin._inverse_transform(fk_current), fk_waypoint
                    ),
                )
                local_target = BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(waypoint_arm_base),
                    world_targets[arm],
                )
                arm_targets[arm].append(
                    pose_to_sdk_target(
                        self._endpoint_sync_transform_to_pose(local_target)
                    )
                )

        final_pose = {
            arm: self._endpoint_sync_transform_to_pose(
                BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(future_arm_base[arm]),
                    world_targets_by_segment[-1][arm],
                )
            )
            for arm in ("left", "right")
        }
        planning_detail = (
            f"smallbox placement path prepared: segments={segments}; "
            f"body_target_units={target_units}; "
            f"left_target=[{final_pose['left'].position.x:.3f},"
            f"{final_pose['left'].position.y:.3f},"
            f"{final_pose['left'].position.z:.3f}]; "
            f"right_target=[{final_pose['right'].position.x:.3f},"
            f"{final_pose['right'].position.y:.3f},"
            f"{final_pose['right'].position.z:.3f}]; "
            "box_pose=single_rigid_interpolation"
        )
        self._publish_place_box_test_feedback(
            goal_handle, "PLACE_BOX_TEST_TARGETS", planning_detail
        )
        if dry_run:
            return (
                f"{planning_detail}; skipped in dry-run",
                final_pose["left"],
                final_pose["right"],
            )

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        body_blend_radius = self._integer("place_box_test_body_blend_radius")
        for index in range(max(0, segments - 1)):
            self._check_canceled(goal_handle, "while queuing place_box_test waist path")
            future = self.body_command_client.call_async(
                self._place_box_test_body_request(
                    body_targets[index][0],
                    trajectory_connect=1,
                    blend_radius=body_blend_radius,
                )
            )
            response = self._wait_future(
                future,
                goal_handle,
                f"queuing place_box_test waist waypoint {index + 1}",
                self._float("dependency_wait_timeout_sec"),
                cancel_local_future=False,
            )
            self._parse_string_command_response(
                response, f"place_box_test waist waypoint {index + 1} MoveJ"
            )

        final_body_future = {}

        def release_body():
            final_body_future["future"] = self.body_command_client.call_async(
                self._place_box_test_body_request(
                    body_targets[-1][0],
                    trajectory_connect=0,
                    blend_radius=0,
                )
            )

        last_monitor_time = [0.0]

        def monitor_rigid_grasp():
            now = time.monotonic()
            if now - last_monitor_time[0] < 0.1:
                return
            last_monitor_time[0] = now
            try:
                actual = {
                    arm: self._lookup_tf_carry_transform(
                        base_frame,
                        self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                    )
                    for arm in ("left", "right")
                }
            except MissionError:
                return
            inferred = {
                arm: BoxSupportMixin._compose_transform(
                    actual[arm],
                    BoxSupportMixin._inverse_transform(relation_by_arm[arm]),
                )
                for arm in ("left", "right")
            }
            position_error = self._endpoint_sync_pose_position_error(
                inferred["left"][0], inferred["right"][0]
            )
            orientation_error = self._endpoint_sync_pose_orientation_error(
                inferred["left"][1], inferred["right"][1]
            )
            if position_error > consistency_position_limit:
                raise MissionError(
                    "place_box_test rigid-grasp position disagreement: "
                    f"{position_error:.4f}m"
                )
            if orientation_error > consistency_orientation_limit:
                raise MissionError(
                    "place_box_test rigid-grasp orientation disagreement: "
                    f"{orientation_error:.4f}rad"
                )

        self._publish_place_box_test_feedback(
            goal_handle,
            "MOVING_TO_PLACE_POSE",
            "starting connected waist MoveJ and dual-arm SDK MoveL; "
            "intermediate trajectory_connect=1, final=0",
        )
        try:
            motion_result = adapter.execute_dual_movel_connected_waypoints(
                arm_targets["left"],
                arm_targets["right"],
                self._float("place_box_test_left_movel_velocity_percent"),
                self._float("place_box_test_right_movel_velocity_percent"),
                blend_radius=self._integer("place_box_test_arm_blend_radius"),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=self._float("place_box_test_timeout_sec"),
                before_start=release_body,
                abort_callback=self._place_box_test_stop_body,
                progress_callback=monitor_rigid_grasp,
            )
            if "future" not in final_body_future:
                raise MissionError(
                    "place_box_test final waist command was not released"
                )
            response = self._wait_future(
                final_body_future["future"],
                goal_handle,
                "starting place_box_test final waist MoveJ",
                self._float("dependency_wait_timeout_sec"),
                cancel_local_future=False,
            )
            self._parse_string_command_response(
                response, "place_box_test final waist MoveJ"
            )
            self._wait_for_body_joints_target(
                goal_handle,
                body_targets[-1][1],
                sequence_after=body_sequence,
                timeout_parameter="place_box_test_timeout_sec",
            )
            if self._boolean("place_box_test_final_correction_enabled"):
                live_final_arm_base = {
                    arm: self._lookup_tf_carry_transform(
                        base_frame,
                        self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
                    )
                    for arm in ("left", "right")
                }
                final_pose = {
                    arm: self._endpoint_sync_transform_to_pose(
                        BoxSupportMixin._compose_transform(
                            BoxSupportMixin._inverse_transform(
                                live_final_arm_base[arm]
                            ),
                            world_targets_by_segment[-1][arm],
                        )
                    )
                    for arm in ("left", "right")
                }
                correction_speed = self._float(
                    "place_box_test_final_correction_velocity_percent"
                )
                adapter.execute_dual_movel_endpoint(
                    pose_to_sdk_target(final_pose["left"]),
                    pose_to_sdk_target(final_pose["right"]),
                    correction_speed,
                    correction_speed,
                    cancel_requested=lambda: goal_handle.is_cancel_requested,
                    timeout_sec=self._float("place_box_test_timeout_sec"),
                )
            verification = self._wait_for_place_box_test_world_targets(
                goal_handle, world_targets_by_segment[-1]
            )
        except (RealManSdkCanceled, MissionCanceled):
            self._place_box_test_stop_body()
            raise
        except (RealManSdkError, MissionError, ValueError) as exc:
            self._place_box_test_stop_body()
            raise MissionError(f"place_box_test motion failed: {exc}") from exc

        return (
            f"{planning_detail}; {motion_result}; final_guard={verification}",
            final_pose["left"],
            final_pose["right"],
        )

    def _execute_tf_body_home_carry(self, goal_handle, adapter, dry_run: bool) -> str:
        """Return the waist home while carrying the box level with dual MoveL.

        Translation follows a point fixed in the common chest frame, while the
        box orientation in the chassis-fixed frame and both box->TCP rigid
        transforms remain fixed.  Future chest/arm-base transforms are
        predicted from the configured URDF chain and re-anchored to live TF at
        every segment so model error cannot accumulate unchecked.
        """
        if not self._boolean("grasp_box_tf_body_home_carry_enabled"):
            return "tf_body_home_carry=disabled"
        if self._boolean("box_step2_waist_endpoint_sync_enabled"):
            raise MissionError(
                "grasp_box_tf_body_home_carry_enabled and "
                "box_step2_waist_endpoint_sync_enabled are mutually exclusive"
            )
        if (
            self._string("grasp_box_tf_body_home_carry_carrier_frame")
            .strip()
            .lstrip("/")
            != "chest_Link"
        ):
            raise MissionError(
                "grasp_box_tf_body_home_carry_carrier_frame must be 'chest_Link' "
                "because future carrier TF is currently predicted by the waist URDF chain"
            )
        segments = self._integer("grasp_box_tf_body_home_carry_segments")
        home_units = [
            int(round(value))
            for value in self._float_array("grasp_box_tf_body_home_carry_joint_units")
        ]
        if dry_run:
            self._last_tf_body_home_carry_completed = True
            self._last_tf_body_home_carry_arm_targets = None
            continuous = self._boolean(
                "grasp_box_tf_body_home_carry_continuous_enabled"
            )
            return (
                "tf_body_home_carry=enabled; trigger=step2; "
                f"segments={segments}; mode={'continuous' if continuous else 'segmented'}; "
                f"home_joint_units={home_units}; "
                "box_translation=follows_chest; box_world_orientation=fixed; "
                "box_to_TCP=fixed; skipped in dry-run"
            )
        if adapter is None:
            raise MissionError(
                "TF waist carry requires direct_motion_backend=python_sdk"
            )
        relation_by_arm = getattr(self, "_last_grasp_box_tf_box_to_link7_targets", None)
        if not relation_by_arm:
            raise MissionError(
                "TF waist carry has no captured box->TCP grasp transforms"
            )

        body_start, _velocities, body_sequence = self._wait_for_fresh_body_feedback(
            goal_handle
        )
        units_per_degree = self._float_array("box_body_command_units_per_degree")
        start_units = [
            int(round(math.degrees(body_start[i]) * units_per_degree[i]))
            for i in range(4)
        ]
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        carrier_frame = (
            self._string("grasp_box_tf_body_home_carry_carrier_frame")
            .strip()
            .lstrip("/")
        )
        current_carrier = self._lookup_tf_carry_transform(base_frame, carrier_frame)
        actual_link = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_link8_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        inferred_box = {
            arm: BoxSupportMixin._compose_transform(
                actual_link[arm],
                BoxSupportMixin._inverse_transform(relation_by_arm[arm]),
            )
            for arm in ("left", "right")
        }
        current_box = BoxSupportMixin._mean_rigid_transforms(
            inferred_box["left"], inferred_box["right"]
        )
        box_world_orientation = current_box[1]
        carrier_to_box_position = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(current_carrier), current_box
        )[0]
        # Re-capture both rigid grasp transforms from actual Link7 TF so the
        # first carry segment starts without a commanded discontinuity.
        relation_by_arm = {
            arm: BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(current_box), actual_link[arm]
            )
            for arm in ("left", "right")
        }

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        left_speed = self._float(
            "grasp_box_tf_body_home_carry_left_movel_velocity_percent"
        )
        right_speed = self._float(
            "grasp_box_tf_body_home_carry_right_movel_velocity_percent"
        )
        body_velocity = self._integer("grasp_box_tf_body_home_carry_body_velocity")
        timeout_sec = self._float("grasp_box_tf_body_home_carry_timeout_sec")
        if self._boolean("grasp_box_tf_body_home_carry_continuous_enabled"):
            return self._execute_tf_body_home_carry_continuous(
                goal_handle,
                adapter,
                segments=segments,
                home_units=home_units,
                body_start=body_start,
                body_sequence=body_sequence,
                relation_by_arm=relation_by_arm,
                current_carrier=current_carrier,
                actual_link=actual_link,
                current_box=current_box,
                base_frame=base_frame,
                carrier_frame=carrier_frame,
                units_per_degree=units_per_degree,
                left_speed=left_speed,
                right_speed=right_speed,
                body_velocity=body_velocity,
                timeout_sec=timeout_sec,
            )
        last_world_targets = None

        try:
            for segment_index in range(1, segments + 1):
                fraction = float(segment_index) / float(segments)
                target_units = [
                    (
                        home_units[i]
                        if segment_index == segments
                        else int(
                            round(
                                start_units[i]
                                + fraction * (home_units[i] - start_units[i])
                            )
                        )
                    )
                    for i in range(4)
                ]
                target_angles = [
                    math.radians(float(target_units[i]) / units_per_degree[i])
                    for i in range(4)
                ]
                measured_body, _measured_velocity, current_body_sequence = (
                    self._wait_for_fresh_body_feedback(
                        goal_handle,
                        sequence_after=(body_sequence if segment_index > 1 else -1),
                    )
                )
                current_angles = [float(value) for value in measured_body[:3]]
                live_carrier = self._lookup_tf_carry_transform(
                    base_frame, carrier_frame
                )
                live_arm_base = {
                    arm: self._lookup_tf_carry_transform(
                        base_frame,
                        self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
                    )
                    for arm in ("left", "right")
                }
                fk_current_chest = self._joint123_chest_transform(current_angles)
                fk_target_chest = self._joint123_chest_transform(target_angles[:3])
                future_carrier = BoxSupportMixin._compose_transform(
                    live_carrier,
                    BoxSupportMixin._compose_transform(
                        BoxSupportMixin._inverse_transform(fk_current_chest),
                        fk_target_chest,
                    ),
                )
                future_box_position = BoxSupportMixin._compose_transform(
                    future_carrier,
                    (carrier_to_box_position, (0.0, 0.0, 0.0, 1.0)),
                )[0]
                future_box = (future_box_position, box_world_orientation)
                world_targets = {
                    arm: BoxSupportMixin._compose_transform(
                        future_box, relation_by_arm[arm]
                    )
                    for arm in ("left", "right")
                }
                arm_targets = {}
                for arm in ("left", "right"):
                    fk_current_arm = self._joint123_arm_base_transform(
                        arm, current_angles
                    )
                    fk_target_arm = self._joint123_arm_base_transform(
                        arm, target_angles[:3]
                    )
                    future_arm_base = BoxSupportMixin._compose_transform(
                        live_arm_base[arm],
                        BoxSupportMixin._compose_transform(
                            BoxSupportMixin._inverse_transform(fk_current_arm),
                            fk_target_arm,
                        ),
                    )
                    arm_targets[arm] = self._endpoint_sync_transform_to_pose(
                        BoxSupportMixin._compose_transform(
                            BoxSupportMixin._inverse_transform(future_arm_base),
                            world_targets[arm],
                        )
                    )
                command_future = {}

                def release_body(units=target_units):
                    command_future["future"] = self.body_command_client.call_async(
                        self._tf_carry_body_request(units, body_velocity)
                    )

                self._publish_box_grasp_feedback(
                    goal_handle,
                    "TF_BODY_HOME_CARRY_SEGMENT",
                    f"segment {segment_index}/{segments}: body MoveJ + dual-arm MoveL; "
                    f"body_joint_units={target_units}; "
                    f"{self._dual_target_position_detail(arm_targets['left'], arm_targets['right'])}",
                )
                motion_result = adapter.execute_dual_movel_endpoint(
                    pose_to_sdk_target(arm_targets["left"]),
                    pose_to_sdk_target(arm_targets["right"]),
                    left_speed,
                    right_speed,
                    cancel_requested=lambda: goal_handle.is_cancel_requested,
                    timeout_sec=timeout_sec,
                    before_start=release_body,
                    abort_callback=self._tf_carry_stop_body,
                )
                if "future" not in command_future:
                    raise MissionError("TF waist-carry body command was not released")
                response = self._wait_future(
                    command_future["future"],
                    goal_handle,
                    f"starting TF waist-carry segment {segment_index} body MoveJ",
                    self._float("dependency_wait_timeout_sec"),
                    cancel_local_future=False,
                )
                self._parse_string_command_response(
                    response, f"TF waist-carry segment {segment_index} body MoveJ"
                )
                self._wait_for_body_joints_target(
                    goal_handle,
                    target_angles,
                    sequence_after=current_body_sequence,
                    timeout_parameter="grasp_box_tf_body_home_carry_timeout_sec",
                )
                verification = self._wait_for_tf_carry_world_targets(
                    goal_handle, world_targets
                )
                body_sequence = self.latest_body_state_sequence
                last_world_targets = world_targets
                self.get_logger().info(
                    f"TF waist-carry segment {segment_index}/{segments} complete: "
                    f"{motion_result}; {verification}"
                )

            live_arm_base = {
                arm: self._lookup_tf_carry_transform(
                    base_frame,
                    self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
                )
                for arm in ("left", "right")
            }
            correction_targets = {
                arm: self._endpoint_sync_transform_to_pose(
                    BoxSupportMixin._compose_transform(
                        BoxSupportMixin._inverse_transform(live_arm_base[arm]),
                        last_world_targets[arm],
                    )
                )
                for arm in ("left", "right")
            }
            if self._boolean("grasp_box_tf_body_home_carry_final_correction_enabled"):
                correction_speed = self._float(
                    "grasp_box_tf_body_home_carry_final_correction_velocity_percent"
                )
                adapter.execute_dual_movel_endpoint(
                    pose_to_sdk_target(correction_targets["left"]),
                    pose_to_sdk_target(correction_targets["right"]),
                    correction_speed,
                    correction_speed,
                    cancel_requested=lambda: goal_handle.is_cancel_requested,
                    timeout_sec=timeout_sec,
                )
                self._wait_for_tf_carry_world_targets(goal_handle, last_world_targets)
            self._last_tf_body_home_carry_arm_targets = correction_targets
        except (RealManSdkCanceled, MissionCanceled):
            self._tf_carry_stop_body()
            raise
        except (RealManSdkError, MissionError, ValueError) as exc:
            self._tf_carry_stop_body()
            raise MissionError(f"TF waist home carry failed: {exc}") from exc

        self._last_tf_body_home_carry_completed = True
        return (
            "tf_body_home_carry=completed; "
            f"segments={segments}; home_joint_units={home_units}; "
            "box_translation=follows_chest; box_world_orientation=fixed; "
            "box_to_TCP=fixed; live_TF_final_guard=confirmed"
        )
