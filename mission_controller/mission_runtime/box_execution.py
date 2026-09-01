import math
from copy import deepcopy
import time

from geometry_msgs.msg import Pose, PoseStamped

try:
    from task_interfaces.srv import MoveCartesian
except ImportError:
    MoveCartesian = None
try:
    from object_pose_interfaces.action import EstimateObjectPose
except ModuleNotFoundError:
    EstimateObjectPose = None

from .common import (
    MissionCanceled,
    MissionError,
)
from .realman_sdk_adapter import (
    RealManSdkCanceled,
    RealManSdkError,
    pose_to_sdk_target,
)


# Bound to the composed compatibility facade by box_support.py.
BoxSupportMixin = None


class BoxExecutionMixin:
    """Post-grasp trajectories, drag joins, and direct motion execution."""

    def _equalize_tf_dual_target_z(
        self,
        left_target: Pose,
        right_target: Pose,
        *,
        reference: str = "average",
    ):
        """Return TF-action targets with one shared arm-base Z coordinate.

        The left/right SDK poses are expressed in their respective arm-base
        frames.  On this robot those bases share the same configured height,
        so equal numeric Z values are an explicit grasp-level constraint.
        ``right`` is used for the delayed DragBox left join because the right
        arm is already holding the physical box; simultaneous dual-arm moves
        use the mean to minimize the correction applied to either arm.
        """
        left_result = deepcopy(left_target)
        right_result = deepcopy(right_target)
        if not self._boolean("box_tf_equalize_dual_target_z_enabled"):
            return left_result, right_result, "target_z_equalization=disabled"
        reference = str(reference).strip().lower()
        if reference == "average":
            target_z = 0.5 * (
                float(left_target.position.z) + float(right_target.position.z)
            )
        elif reference == "left":
            target_z = float(left_target.position.z)
        elif reference == "right":
            target_z = float(right_target.position.z)
        else:
            raise MissionError(
                "TF dual-target Z reference must be average, left, or right"
            )
        original_left_z = float(left_target.position.z)
        original_right_z = float(right_target.position.z)
        left_result.position.z = target_z
        right_result.position.z = target_z
        return left_result, right_result, (
            "target_z_equalization=enabled; "
            f"reference={reference}; target_z={target_z:.4f}; "
            f"original_left_z={original_left_z:.4f}; "
            f"original_right_z={original_right_z:.4f}"
        )

    @staticmethod
    def _drag_tf_reanchor_active(
        *,
        enabled: bool,
        tf_mode: bool,
        drag_mode: bool,
        delayed_left_join: bool,
    ) -> bool:
        """Return whether the runtime Drag3 re-anchor path is applicable."""
        return bool(enabled and tf_mode and drag_mode and delayed_left_join)

    def _drag_tf_world_transform_to_arm_pose(self, transform, arm: str) -> Pose:
        """Express one frozen-frame Link7 transform in the live arm base."""
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        arm_base_frame = self._string(f"{arm}_arm_base_frame").strip().lstrip("/")
        base_to_arm_base = self._lookup_tf_carry_transform(
            base_frame,
            arm_base_frame,
            parameter_prefix="drag_box_tf_body_home_carry",
        )
        arm_base_to_target = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(base_to_arm_base),
            transform,
        )
        return self._endpoint_sync_transform_to_pose(arm_base_to_target)

    def _capture_drag_tf_right_grasp_relation(self) -> str:
        """Capture the physical box->right-Link7 relation before Drag1.

        The box is still supported by the shelf after the right-arm Step1
        contact search.  The frozen FoundationPose box transform can therefore
        be paired with the actual right Link7 TF to obtain the rigid relation
        used to infer the box pose after Drag3.
        """
        frozen_box_pose = getattr(self, "_last_grasp_box_tf_box_pose", None)
        relation_by_arm = getattr(
            self, "_last_grasp_box_tf_box_to_link7_targets", None
        )
        if frozen_box_pose is None or not relation_by_arm:
            raise MissionError(
                "DragBox TF re-anchor has no frozen box pose or box->Link7 targets"
            )
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        pose_frame = frozen_box_pose.header.frame_id.strip().lstrip("/")
        if pose_frame != base_frame:
            raise MissionError(
                "DragBox TF re-anchor frozen box frame mismatch: "
                f"pose_frame={pose_frame}, expected={base_frame}"
            )
        frozen_box = self._pose_stamped_to_transform(frozen_box_pose)
        actual_right = self._lookup_tf_carry_transform(
            base_frame,
            self._string("right_link8_frame").strip().lstrip("/"),
            parameter_prefix="drag_box_tf_body_home_carry",
        )
        right_relation = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(frozen_box),
            actual_right,
        )
        self._last_drag_box_tf_desired_box_to_link7_targets = deepcopy(
            relation_by_arm
        )
        self._last_drag_box_tf_right_grasp_relation = right_relation
        return (
            "drag_tf_right_grasp_relation=captured_after_step1; "
            f"box_to_right_link7_translation="
            f"[{right_relation[0][0]:.4f},{right_relation[0][1]:.4f},"
            f"{right_relation[0][2]:.4f}]"
        )

    def _reanchor_drag_tf_left_join_after_drag3(self):
        """Infer the moved box from actual right Link7 and rebuild left join."""
        desired_relations = getattr(
            self, "_last_drag_box_tf_desired_box_to_link7_targets", None
        )
        right_relation = getattr(
            self, "_last_drag_box_tf_right_grasp_relation", None
        )
        if not desired_relations or right_relation is None:
            raise MissionError(
                "DragBox TF re-anchor has no captured right-arm grasp relation"
            )
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        actual_right = self._lookup_tf_carry_transform(
            base_frame,
            self._string("right_link8_frame").strip().lstrip("/"),
            parameter_prefix="drag_box_tf_body_home_carry",
        )
        current_box = BoxSupportMixin._compose_transform(
            actual_right,
            BoxSupportMixin._inverse_transform(right_relation),
        )
        left_world_target = BoxSupportMixin._compose_transform(
            current_box,
            desired_relations["left"],
        )
        left_target = self._drag_tf_world_transform_to_arm_pose(
            left_world_target, "left"
        )
        right_target = self._drag_tf_world_transform_to_arm_pose(
            actual_right, "right"
        )
        left_target, _right_target, z_detail = (
            BoxExecutionMixin._equalize_tf_dual_target_z(
                self,
                left_target,
                right_target,
                reference="right",
            )
        )
        self._last_drag_box_tf_reanchored_box_after_drag3 = current_box
        relation_by_arm = deepcopy(desired_relations)
        relation_by_arm["right"] = right_relation
        self._last_grasp_box_tf_box_to_link7_targets = relation_by_arm
        return left_target, (
            "drag_tf_reanchor_after_drag3=completed; authority=actual_right_link7; "
            f"inferred_box_position=[{current_box[0][0]:.4f},"
            f"{current_box[0][1]:.4f},{current_box[0][2]:.4f}]; "
            f"left_join_target=recomputed_from_common_box; {z_detail}"
        )

    def _reanchor_drag_tf_step2_from_actual_grasp(
        self,
        targets,
        next_target_index: int,
        *,
        box_layer: int,
        model_label: str | None,
    ) -> str:
        """Rebuild Step2 from one box frame after the delayed left clamp.

        Both TCP targets are generated from the same translated box transform.
        The actual post-clamp box->TCP relations are captured first, preserving
        the physical grasp while preventing the two arms from using separately
        reconstructed box axes.
        """
        right_relation = getattr(
            self, "_last_drag_box_tf_right_grasp_relation", None
        )
        if right_relation is None:
            raise MissionError(
                "DragBox TF Step2 re-anchor has no right-arm grasp relation"
            )
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        actual_link = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                parameter_prefix="drag_box_tf_body_home_carry",
            )
            for arm in ("left", "right")
        }
        current_box = BoxSupportMixin._compose_transform(
            actual_link["right"],
            BoxSupportMixin._inverse_transform(right_relation),
        )
        relation_by_arm = {
            arm: BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(current_box),
                actual_link[arm],
            )
            for arm in ("left", "right")
        }

        left_parameter = self._post_movel_xyz_parameter_name(
            "left",
            2,
            model_label,
            box_layer=box_layer,
            tf_mode=True,
            drag_mode=True,
        )
        right_parameter = self._post_movel_xyz_parameter_name(
            "right",
            2,
            model_label,
            box_layer=box_layer,
            tf_mode=True,
            drag_mode=True,
        )
        left_delta = tuple(float(value) for value in self._float_array(left_parameter))
        right_delta = tuple(
            float(value) for value in self._float_array(right_parameter)
        )
        if len(left_delta) != 3 or len(right_delta) != 3:
            raise MissionError("DragBox TF Step2 deltas must contain three values")
        if any(
            abs(left_delta[index] - right_delta[index]) > 1.0e-6
            for index in range(3)
        ):
            raise MissionError(
                "DragBox TF rigid Step2 requires identical left/right box-frame "
                f"deltas: left={left_delta}, right={right_delta}"
            )

        step2_box = BoxSupportMixin._compose_transform(
            current_box,
            (left_delta, (0.0, 0.0, 0.0, 1.0)),
        )
        step2_poses = {
            arm: self._drag_tf_world_transform_to_arm_pose(
                BoxSupportMixin._compose_transform(
                    step2_box, relation_by_arm[arm]
                ),
                arm,
            )
            for arm in ("left", "right")
        }
        step2_index = next(
            (
                index
                for index in range(next_target_index, len(targets))
                if targets[index][0] == "step2"
            ),
            None,
        )
        if step2_index is None:
            raise MissionError("DragBox TF re-anchor could not find Step2 target")
        targets[step2_index] = (
            "step2",
            step2_poses["left"],
            step2_poses["right"],
        )
        self._last_grasp_box_tf_box_to_link7_targets = relation_by_arm
        return (
            "drag_tf_step2_reanchor=completed; common_box_frame=true; "
            f"box_delta=[{left_delta[0]:.4f},{left_delta[1]:.4f},"
            f"{left_delta[2]:.4f}]; post_clamp_box_to_link7=recaptured"
        )

    def _rebase_post_movel_targets_after_tf_carry(
        self,
        targets,
        next_target_index: int,
        *,
        box_layer: int = 1,
        model_label: str | None = None,
        drag_mode: bool = False,
        tf_mode: bool = False,
    ) -> None:
        """Rebuild post-Step2 targets in the new live arm-base frames."""
        rebased = getattr(self, "_last_tf_body_home_carry_arm_targets", None)
        if not rebased:
            return
        left_current = deepcopy(rebased["left"])
        right_current = deepcopy(rebased["right"])
        for index in range(next_target_index, len(targets)):
            label = targets[index][0]
            if not label.startswith("step") or not label[4:].isdigit():
                continue
            step_number = int(label[4:])
            left_parameter = (
                self._post_movel_xyz_parameter_name(
                    "left",
                    step_number,
                    model_label,
                    box_layer=box_layer,
                    tf_mode=True,
                    drag_mode=drag_mode,
                )
                if tf_mode
                else f"box_post_movel_left_step{step_number}_xyz"
            )
            right_parameter = (
                self._post_movel_xyz_parameter_name(
                    "right",
                    step_number,
                    model_label,
                    box_layer=box_layer,
                    tf_mode=True,
                    drag_mode=drag_mode,
                )
                if tf_mode
                else f"box_post_movel_right_step{step_number}_xyz"
            )
            left_current = BoxSupportMixin._translate_pose_in_box_frame(
                self,
                left_current,
                self._float_array(left_parameter),
                "left",
            )
            right_current = BoxSupportMixin._translate_pose_in_box_frame(
                self,
                right_current,
                self._float_array(right_parameter),
                "right",
            )
            targets[index] = (label, left_current, right_current)

    def _wait_for_endpoint_arm_targets(
        self, goal_handle, targets_by_arm, sequence_after
    ):
        timeout_sec = self._float("box_step2_waist_endpoint_sync_timeout_sec")
        max_age_sec = self._float("box_step2_waist_endpoint_sync_feedback_max_age_sec")
        position_limit = self._float(
            "box_step2_waist_endpoint_sync_final_position_tolerance_m"
        )
        orientation_limit = self._float(
            "box_step2_waist_endpoint_sync_final_orientation_tolerance_rad"
        )
        required_stable = self._integer("box_step2_waist_endpoint_sync_stable_samples")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        latest_detail = "no feedback"
        last_sequences = dict(sequence_after)
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while verifying Step2 endpoint EEPose")
            now = time.monotonic()
            all_match = True
            details = []
            with self.joint_state_lock:
                for arm in ("left", "right"):
                    values = self.latest_slave_arm_poses.get(arm)
                    sequence = self.latest_slave_arm_pose_sequences.get(arm, 0)
                    age = now - self.latest_slave_arm_pose_times.get(arm, 0.0)
                    if (
                        values is None
                        or sequence <= sequence_after.get(arm, -1)
                        or age > max_age_sec
                    ):
                        all_match = False
                        details.append(f"{arm}=stale/missing")
                        continue
                    target = targets_by_arm[arm]
                    if isinstance(target, Pose):
                        target = self._endpoint_sync_pose_values_to_transform(
                            (
                                target.position.x,
                                target.position.y,
                                target.position.z,
                                target.orientation.x,
                                target.orientation.y,
                                target.orientation.z,
                                target.orientation.w,
                            )
                        )
                    position_error = self._endpoint_sync_pose_position_error(
                        values[:3], target[0]
                    )
                    orientation_error = self._endpoint_sync_pose_orientation_error(
                        values[3:7], target[1]
                    )
                    details.append(
                        f"{arm}_position_error={position_error:.4f}m,"
                        f"{arm}_orientation_error={orientation_error:.4f}rad"
                    )
                    if (
                        position_error > position_limit
                        or orientation_error > orientation_limit
                    ):
                        all_match = False
                    last_sequences[arm] = sequence
            latest_detail = "; ".join(details)
            if all_match:
                stable_samples += 1
                if stable_samples >= required_stable:
                    return
            else:
                stable_samples = 0
            time.sleep(0.02)
        raise MissionError(
            "Step2 endpoint EEPose did not reach targets: "
            f"{latest_detail}, timeout_sec={timeout_sec:.1f}"
        )

    def _execute_step2_waist_endpoint_sync(
        self,
        goal_handle,
        adapter,
        box_layer: int,
        dry_run: bool,
        sequence_after=None,
    ) -> str:
        if not self._boolean("box_step2_waist_endpoint_sync_enabled"):
            return "step2_waist_endpoint_sync=disabled"
        speeds = self._step2_endpoint_sync_layer_config(box_layer)
        prefix = "box_step2_waist_endpoint_sync_"
        home_units = [
            int(value) for value in self._float_array(f"{prefix}home_joint_units")
        ]
        if len(home_units) != 4:
            raise MissionError(f"{prefix}home_joint_units must contain four values")
        if dry_run:
            return (
                "step2_waist_endpoint_sync=enabled; "
                f"layer={box_layer}; home_joint_units={home_units}; "
                f"forward_body_velocity={speeds['forward_body_velocity']}; "
                f"reverse_body_velocity={speeds['reverse_body_velocity']}; "
                "fresh Step2 EEPose/body feedback required at runtime; skipped in dry-run"
            )
        if adapter is None:
            raise MissionError(
                "Step2 endpoint waist sync requires direct_motion_backend=python_sdk"
            )
        sequence_after = sequence_after or {"left": -1, "right": -1}
        body_start, _body_velocities, body_sequence = (
            self._wait_for_fresh_body_feedback(goal_handle)
        )
        arm_values, arm_sequences = self._wait_for_fresh_endpoint_arm_poses(
            goal_handle, sequence_after
        )
        units_per_degree = self._float_array("box_body_command_units_per_degree")
        if len(units_per_degree) != 4 or any(
            value <= 0.0 for value in units_per_degree
        ):
            raise MissionError(
                "box_body_command_units_per_degree must contain four positive values"
            )
        start_units = [
            int(round(math.degrees(float(body_start[index])) * units_per_degree[index]))
            for index in range(4)
        ]
        start_angles = [float(value) for value in body_start[:3]]
        home_angles = [
            math.radians(float(home_units[index]) / units_per_degree[index])
            for index in range(4)
        ]
        hold_by_arm = {
            arm: BoxSupportMixin._compose_transform(
                self._joint123_arm_base_transform(arm, start_angles),
                self._endpoint_sync_pose_values_to_transform(arm_values[arm]),
            )
            for arm in ("left", "right")
        }

        def pose_targets_for_body_angles(body_angles):
            return {
                arm: self._endpoint_sync_transform_to_pose(
                    BoxSupportMixin._compose_transform(
                        BoxSupportMixin._inverse_transform(
                            self._joint123_arm_base_transform(arm, body_angles[:3])
                        ),
                        hold_by_arm[arm],
                    )
                )
                for arm in ("left", "right")
            }

        home_targets = {
            arm: self._endpoint_sync_transform_to_pose(
                BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(
                        self._joint123_arm_base_transform(arm, home_angles[:3])
                    ),
                    hold_by_arm[arm],
                )
            )
            for arm in ("left", "right")
        }
        self._publish_box_grasp_feedback(
            goal_handle,
            "STEP2_WAIST_ENDPOINT_SYNC_TARGETS",
            "Step2 complete; computed endpoint MoveL targets at waist home "
            f"from measured Link7 EEPose, layer={box_layer}, "
            f"home_joint_units={home_units}; segments={speeds['segments']}; "
            f"targets={self._dual_target_position_detail(home_targets['left'], home_targets['right'])}",
        )
        endpoint_timeout = self._float(f"{prefix}timeout_sec")
        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)

        def build_leg_plan(leg_start_units, leg_end_units):
            plan = []
            segment_count = speeds["segments"]
            for segment_index in range(1, segment_count + 1):
                fraction = float(segment_index) / float(segment_count)
                if segment_index == segment_count:
                    units = [int(value) for value in leg_end_units]
                else:
                    units = [
                        int(
                            round(
                                float(leg_start_units[index])
                                + fraction
                                * (
                                    float(leg_end_units[index])
                                    - float(leg_start_units[index])
                                )
                            )
                        )
                        for index in range(4)
                    ]
                command_angles = [
                    math.radians(float(units[index]) / units_per_degree[index])
                    for index in range(4)
                ]
                plan.append(
                    {
                        "index": segment_index,
                        "units": units,
                        "angles": command_angles,
                        "poses": pose_targets_for_body_angles(command_angles),
                    }
                )
            return plan

        def execute_leg(
            label,
            leg_start_units,
            leg_end_units,
            leg_start_angles,
            leg_end_angles,
            body_velocity,
            left_speed,
            right_speed,
            arm_sequences_before,
            body_sequence_before,
        ):
            current_arm_sequences = dict(arm_sequences_before)
            current_body_sequence = body_sequence_before
            results = []
            plan = build_leg_plan(leg_start_units, leg_end_units)
            for item in plan:
                command_future = {}

                def release_body():
                    command_future["future"] = self.body_command_client.call_async(
                        self._step2_endpoint_sync_body_request(
                            item["units"], body_velocity
                        )
                    )

                self._publish_box_grasp_feedback(
                    goal_handle,
                    f"STEP2_WAIST_ENDPOINT_{label.upper()}_{item['index']}",
                    f"sending MoveL segment {item['index']}/{len(plan)} with body MoveJ, "
                    f"left_speed={left_speed:.1f}, right_speed={right_speed:.1f}; "
                    f"targets={self._dual_target_position_detail(item['poses']['left'], item['poses']['right'])}",
                )
                try:
                    motion_result = adapter.execute_dual_movel_endpoint(
                        pose_to_sdk_target(item["poses"]["left"]),
                        pose_to_sdk_target(item["poses"]["right"]),
                        left_speed,
                        right_speed,
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=endpoint_timeout,
                        before_start=release_body,
                        abort_callback=self._step2_endpoint_sync_stop_body,
                    )
                    if "future" not in command_future:
                        raise MissionError(
                            f"Step2 endpoint {label} segment {item['index']} body command was not released"
                        )
                    response = self._wait_future(
                        command_future["future"],
                        goal_handle,
                        f"starting Step2 endpoint waist {label} segment {item['index']} MoveJ",
                        self._float("dependency_wait_timeout_sec"),
                        cancel_local_future=False,
                    )
                    self._parse_string_command_response(
                        response,
                        f"starting Step2 endpoint waist {label} segment {item['index']} MoveJ",
                    )
                    self._wait_for_body_joints_target(
                        goal_handle,
                        item["angles"],
                        sequence_after=current_body_sequence,
                        timeout_parameter=f"{prefix}timeout_sec",
                    )
                    self._wait_for_endpoint_arm_targets(
                        goal_handle, item["poses"], current_arm_sequences
                    )
                    results.append(
                        f"segment={item['index']}/{len(plan)}; {motion_result}"
                    )
                    with self.joint_state_lock:
                        current_body_sequence = self.latest_body_state_sequence
                        current_arm_sequences = dict(
                            self.latest_slave_arm_pose_sequences
                        )
                except (RealManSdkCanceled, MissionCanceled):
                    self._step2_endpoint_sync_stop_body()
                    raise
                except (RealManSdkError, MissionError, ValueError) as exc:
                    self._step2_endpoint_sync_stop_body()
                    raise MissionError(
                        f"Step2 waist endpoint {label} segment {item['index']} failed: {exc}"
                    ) from exc
            return "; ".join(results)

        try:
            forward_result = execute_leg(
                "to_home",
                start_units,
                home_units,
                [float(value) for value in body_start],
                home_angles,
                speeds["forward_body_velocity"],
                speeds["forward_left_speed"],
                speeds["forward_right_speed"],
                arm_sequences,
                body_sequence,
            )
            with self.joint_state_lock:
                reverse_body_sequence = self.latest_body_state_sequence
                reverse_arm_sequences = dict(self.latest_slave_arm_pose_sequences)
            reverse_result = execute_leg(
                "to_start",
                home_units,
                start_units,
                home_angles,
                [float(value) for value in body_start],
                speeds["reverse_body_velocity"],
                speeds["reverse_left_speed"],
                speeds["reverse_right_speed"],
                reverse_arm_sequences,
                reverse_body_sequence,
            )
        except (RealManSdkCanceled, MissionCanceled):
            self._step2_endpoint_sync_stop_body()
            raise
        except (RealManSdkError, MissionError, ValueError) as exc:
            self._step2_endpoint_sync_stop_body()
            raise MissionError(f"Step2 waist endpoint sync failed: {exc}") from exc
        self._last_step2_endpoint_sync_completed = True
        return (
            "step2_waist_endpoint_sync=completed; "
            f"layer={box_layer}; home_joint_units={home_units}; "
            f"return_joint_units={start_units}; {forward_result}; {reverse_result}; "
            "body_and_Link7_feedback=confirmed"
        )

    @staticmethod
    def _dual_target_position_detail(left_target: Pose, right_target: Pose) -> str:
        return (
            f"left=[{left_target.position.x:.3f}, "
            f"{left_target.position.y:.3f}, {left_target.position.z:.3f}] m, "
            f"right=[{right_target.position.x:.3f}, "
            f"{right_target.position.y:.3f}, {right_target.position.z:.3f}] m"
        )

    def _execute_drag_box_left_join_pre_movej(
        self,
        goal_handle,
        adapter,
        dry_run: bool,
    ) -> str:
        """Move the left arm to its configured posture before the join.

        DragBox intentionally keeps the left arm stationary while the right
        arm performs Drag1--Drag3. Immediately before the delayed left-arm
        MoveJ_P join, this optional MoveJ places the left arm in a known,
        reachable posture. Both commands use the same Python SDK adapter and
        the same left-arm SDK handle; the ROS ``/robot/command`` path is not
        used here.
        """
        if not self._boolean("drag_box_left_join_pre_movej_enabled"):
            return "left_join_pre_movej=disabled"

        units = [
            int(round(value))
            for value in self._float_array("drag_box_left_join_pre_movej_joint_units")
        ]
        if len(units) != 7:
            raise MissionError(
                "drag_box_left_join_pre_movej_joint_units must contain 7 values"
            )
        units_per_degree = self._float(
            "box_pre_target_arm_movej_command_units_per_degree"
        )
        if not math.isfinite(units_per_degree) or units_per_degree <= 0.0:
            raise MissionError(
                "box_pre_target_arm_movej_command_units_per_degree must be positive"
            )
        left_target_degrees = [float(value) / units_per_degree for value in units]
        detail = (
            "DragBox left-arm join pre-MoveJ: "
            "backend=python_sdk, arm=left, "
            f"joint_units={units}, "
            f"joint_degrees={[round(value, 3) for value in left_target_degrees]}"
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "DRAG_LEFT_JOIN_PRE_MOVEJ_TARGETS",
            detail,
        )
        if dry_run:
            return f"{detail}; skipped in dry-run"
        if adapter is None:
            raise MissionError(
                "DragBox left-arm join pre-MoveJ requires "
                "direct_motion_backend=python_sdk"
            )

        with self.joint_state_lock:
            sequence_before = {
                "left": self.latest_slave_arm_state_sequences.get("left", 0)
            }
        self._publish_box_grasp_feedback(
            goal_handle,
            "MOVING_DRAG_LEFT_JOIN_PRE_MOVEJ",
            "sending DragBox left-arm join pre-MoveJ through Python SDK",
        )
        try:
            motion_result = adapter.execute_single_movej(
                "left",
                left_target_degrees,
                self._float("box_pre_target_arm_movej_velocity"),
                blend_radius=self._integer("box_pre_target_arm_movej_blend_radius"),
                trajectory_connect=self._integer(
                    "box_pre_target_arm_movej_trajectory_connect"
                ),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=self._float("box_pre_target_arm_movej_timeout_sec"),
            )
        except RealManSdkCanceled as exc:
            raise MissionCanceled(str(exc)) from exc
        except (RealManSdkError, ValueError) as exc:
            raise MissionError(
                f"DragBox left-arm join pre-MoveJ failed: {exc}"
            ) from exc

        self._publish_box_grasp_feedback(
            goal_handle,
            "WAITING_FOR_DRAG_LEFT_JOIN_PRE_MOVEJ",
            "Python SDK pre-MoveJ completed; waiting for fresh stable left-arm feedback",
        )
        self._wait_for_post_arm_joint_targets(
            goal_handle,
            left_target_rad=[math.radians(value) for value in left_target_degrees],
            right_target_rad=[0.0] * 7,
            sequence_after=sequence_before,
            parameter_prefix="box_pre_target_arm_movej",
            description="DragBox left-arm join pre-MoveJ",
            active_arms=("left",),
        )
        settle_sec = self._float("arm_settle_sec")
        if settle_sec > 0.0:
            time.sleep(settle_sec)
        self._publish_box_grasp_feedback(
            goal_handle,
            "DRAG_LEFT_JOIN_PRE_MOVEJ_REACHED",
            f"left arm reached the pre-MoveJ target with stable feedback; "
            f"backend=python_sdk; settle_sec={settle_sec:.3f}",
        )
        return f"{detail}; {motion_result}; arm_feedback=confirmed; settle_sec={settle_sec:.3f}"

    def _execute_drag_box_left_join(
        self,
        goal_handle,
        adapter,
        left_target: Pose,
        dry_run: bool,
    ) -> str:
        """Move the delayed left arm to its cumulative post-drag target."""
        motion_mode = self._string("drag_box_left_join_motion_mode").strip().lower()
        if motion_mode not in ("movel", "movej_p"):
            raise MissionError(
                "drag_box_left_join_motion_mode must be 'movel' or 'movej_p'"
            )
        detail = (
            "delayed left-arm join after Drag3: "
            f"{motion_mode} target="
            f"[{left_target.position.x:.3f}, {left_target.position.y:.3f}, "
            f"{left_target.position.z:.3f}] m, "
            f"q=[{left_target.orientation.x:.3f}, "
            f"{left_target.orientation.y:.3f}, "
            f"{left_target.orientation.z:.3f}, "
            f"{left_target.orientation.w:.3f}], "
            "target_frame=left_arm_base"
        )
        self._publish_box_grasp_feedback(
            goal_handle,
            "POST_MOVEL_LEFT_JOIN_TARGETS",
            detail,
        )
        if adapter is None:
            if not dry_run:
                raise MissionError(
                    "delayed left-arm join requires direct_motion_backend=python_sdk"
                )
        pre_movej_result = self._execute_drag_box_left_join_pre_movej(
            goal_handle,
            adapter,
            dry_run,
        )
        if dry_run:
            return f"{pre_movej_result}; {detail}; skipped in dry-run"
        try:
            motion_result = adapter.execute_single(
                "left",
                pose_to_sdk_target(left_target),
                motion_mode,
                self._float("drag_box_left_join_velocity_percent"),
                self._boolean("direct_movel_blocking"),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=self._float("drag_box_left_join_timeout_sec"),
            )
        except RealManSdkCanceled as exc:
            raise MissionCanceled(str(exc)) from exc
        except (RealManSdkError, ValueError) as exc:
            raise MissionError(f"delayed left-arm join failed: {exc}") from exc
        return f"{pre_movej_result}; {detail}; {motion_result}; " "left_join=confirmed"

    def _execute_post_movel_sequence(
        self,
        goal_handle,
        adapter,
        left_target: Pose,
        right_target: Pose,
        dry_run: bool,
        *,
        box_layer: int = 1,
        drag_mode: bool = False,
        right_arm_only: bool = False,
        delayed_left_join: bool = False,
        model_label: str | None = None,
        tf_mode: bool = False,
    ) -> str:
        self._last_step2_endpoint_sync_completed = False
        self._last_tf_body_home_carry_completed = False
        self._last_tf_body_home_carry_arm_targets = None
        tf_carry_parameter_prefix = BoxSupportMixin._tf_body_home_carry_parameter_prefix(
            tf_mode=tf_mode,
            drag_mode=drag_mode,
        )
        tf_carry_enabled = bool(
            tf_carry_parameter_prefix
            and self._boolean(f"{tf_carry_parameter_prefix}_enabled")
        )
        standard_post_movel_enabled = self._boolean("box_post_movel_enabled")
        drag_post_movel_enabled = drag_mode and self._boolean(
            "drag_box_post_movel_enabled"
        )
        force_parameter_prefix = BoxSupportMixin._force_clamp_parameter_prefix(
            tf_mode=tf_mode,
            drag_mode=drag_mode,
        )
        # Small sequence harnesses used by the legacy unit tests compose only
        # BoxExecutionMixin.  Treat the absent optional force mixin as the
        # safe disabled mode; MissionController itself always has the method.
        force_mode = (
            self._force_clamp_mode(force_parameter_prefix)
            if hasattr(self, "_force_clamp_mode")
            else "disabled"
        )
        drag_tf_reanchor_enabled = BoxExecutionMixin._drag_tf_reanchor_active(
            enabled=(
                self._boolean("drag_box_tf_reanchor_after_drag3_enabled")
                if tf_mode and drag_mode and delayed_left_join
                else False
            ),
            tf_mode=tf_mode,
            drag_mode=drag_mode,
            delayed_left_join=delayed_left_join,
        )
        self._last_drag_box_tf_desired_box_to_link7_targets = None
        self._last_drag_box_tf_right_grasp_relation = None
        self._last_drag_box_tf_reanchored_box_after_drag3 = None
        if not standard_post_movel_enabled and not drag_post_movel_enabled:
            if force_mode == "closed_loop":
                raise MissionError(
                    f"{force_parameter_prefix}_mode=closed_loop requires the "
                    "corresponding post_movel sequence to be enabled"
                )
            if delayed_left_join:
                raise MissionError(
                    "delayed left-arm join requires " "drag_box_post_movel_enabled=true"
                )
            return "post_movel=disabled"

        include_drag_steps = drag_post_movel_enabled
        if delayed_left_join and not include_drag_steps:
            raise MissionError(
                "delayed left-arm join requires drag_box_post_movel_enabled=true"
            )
        left_joined = not delayed_left_join
        active_arms = ("right",) if right_arm_only else ("left", "right")
        post_target_kwargs = {
            "include_drag_steps": include_drag_steps,
            "defer_left_step1": delayed_left_join,
            "model_label": model_label,
        }
        if tf_mode:
            post_target_kwargs.update(
                box_layer=box_layer,
                tf_mode=True,
            )
        targets = self._post_movel_targets_with_labels(
            left_target,
            right_target,
            **post_target_kwargs,
        )
        force_session = None
        if force_mode == "closed_loop":
            if not tf_mode:
                raise MissionError(
                    "force clamp is only supported by grasp_box_tf and drag_box_tf"
                )
            if adapter is None and not dry_run:
                raise MissionError("force clamp requires direct_motion_backend=python_sdk")
            if self._integer("box_post_movel_step_count") < 2:
                raise MissionError(
                    f"{force_parameter_prefix}_mode=closed_loop requires "
                    "box_post_movel_step_count>=2 so Step2 follows force clamp"
                )
            if not dry_run:
                force_session = self._new_force_clamp_session(force_parameter_prefix)
        with self.joint_state_lock:
            step2_endpoint_sequence_after = {
                "left": self.latest_slave_arm_pose_sequences.get("left", 0),
                "right": self.latest_slave_arm_pose_sequences.get("right", 0),
            }
        results = []
        for sequence_index, (label, left_step, right_step) in enumerate(
            targets, start=1
        ):
            is_step4 = label == "step4"
            is_left_only_step = label == "step1_left"
            motion_mode = "movel"
            if is_step4:
                motion_mode = (
                    self._string("box_post_movel_step4_motion_mode").strip().lower()
                )
                if motion_mode not in ("movel", "movej_p", "movej"):
                    raise MissionError(
                        "box_post_movel_step4_motion_mode must be "
                        "'movel', 'movej_p', or 'movej'"
                    )
            z_equalization_detail = "target_z_equalization=not_applicable"
            sends_dual_pose = (
                not is_left_only_step
                and not (right_arm_only and not left_joined)
                and not (is_step4 and motion_mode == "movej")
            )
            if tf_mode and sends_dual_pose:
                left_step, right_step, z_equalization_detail = (
                    BoxExecutionMixin._equalize_tf_dual_target_z(
                        self,
                        left_step,
                        right_step,
                        reference="average",
                    )
                )
                targets[sequence_index - 1] = (label, left_step, right_step)
            detail_arms = ("left",) if is_left_only_step else active_arms
            detail = (
                f"post-grasp {', '.join(detail_arms)} arm {motion_mode} {label} "
                f"({sequence_index}/{len(targets)}) "
                "in left/right arm "
                f"base frames: "
                f"{BoxSupportMixin._dual_target_position_detail(left_step, right_step)}; "
                "delta_frame=foundationpose_box; "
                "orientation unchanged from the initial Link8 targets; "
                f"{z_equalization_detail}"
            )
            self._publish_box_grasp_feedback(
                goal_handle,
                f"POST_MOVEL_{label.upper()}_TARGETS",
                detail,
            )
            if dry_run:
                results.append(f"{detail}; skipped in dry-run")
                if delayed_left_join and label == "step_drag3":
                    if drag_tf_reanchor_enabled:
                        results.append(
                            "drag_tf_reanchor_after_drag3=skipped_in_dry_run; "
                            "nominal cumulative left target retained"
                        )
                    results.append(
                        self._execute_drag_box_left_join(
                            goal_handle,
                            None,
                            left_step,
                            True,
                        )
                    )
                    left_joined = True
                    active_arms = ("left", "right")
                if (
                    label == "step2"
                    and not drag_mode
                    and not right_arm_only
                    and self._boolean("box_step2_waist_endpoint_sync_enabled")
                ):
                    results.append(
                        self._execute_step2_waist_endpoint_sync(
                            goal_handle,
                            None,
                            box_layer,
                            True,
                            step2_endpoint_sequence_after,
                        )
                    )
                if (
                    label == "step2"
                    and tf_carry_enabled
                    and (
                        not right_arm_only
                        or (delayed_left_join and left_joined)
                    )
                ):
                    results.append(
                        self._execute_tf_body_home_carry(
                            goal_handle,
                            None,
                            True,
                            parameter_prefix=tf_carry_parameter_prefix,
                            box_layer=box_layer,
                            model_label=model_label,
                        )
                    )
                continue
            # Step1 is the force-controlled clamping motion.  DragBox first
            # searches contact with the right arm only; after Drag3 the
            # delayed left arm joins and both arms execute the Step1 force
            # clamp together.
            force_label = (
                label == "step1"
                and force_mode == "closed_loop"
            ) or (
                label == "step1_left"
                and force_mode == "closed_loop"
                and drag_mode
                and delayed_left_join
            )
            if force_label:
                if label == "step1" and drag_mode and delayed_left_join:
                    force_arms = ("right",)
                elif label == "step1_left":
                    force_arms = ("left", "right")
                else:
                    force_arms = tuple(detail_arms)
                current_poses = {}
                for arm in force_arms:
                    pose, _sequence, age = self._force_clamp_pose_snapshot(arm)
                    if pose is None or age > self._float(
                        f"{force_parameter_prefix}_sensor_max_age_sec"
                    ):
                        raise MissionError(
                            f"{force_parameter_prefix} {arm} Link7 pose is missing/stale "
                            f"before {label}: age={age:.3f}s"
                        )
                    current_poses[arm] = pose
                left_delta_name = self._post_movel_xyz_parameter_name(
                    "left",
                    1,
                    model_label,
                    box_layer=box_layer,
                    tf_mode=True,
                    drag_mode=drag_mode,
                )
                right_delta_name = self._post_movel_xyz_parameter_name(
                    "right",
                    1,
                    model_label,
                    box_layer=box_layer,
                    tf_mode=True,
                    drag_mode=drag_mode,
                )
                directions = {
                    "left": self._force_clamp_direction(
                        self._float_array(left_delta_name), "left"
                    ),
                    "right": self._force_clamp_direction(
                        self._float_array(right_delta_name), "right"
                    ),
                }
                if label == "step1" and drag_mode and delayed_left_join:
                    actual_right, force_detail = self._force_clamp_contact_only(
                        goal_handle,
                        adapter,
                        force_session,
                        "right",
                        current_poses["right"],
                        self._float_array(right_delta_name),
                    )
                    actual_by_arm = {"right": actual_right}
                else:
                    actual_by_arm, force_detail = self._force_clamp_bilateral(
                        goal_handle,
                        adapter,
                        force_session,
                        current_poses,
                        directions,
                    )
                    # A zero direction is passive: do not apply a correction
                    # based on a monitored arm that was not moved.
                    actual_by_arm = {
                        arm: pose
                        for arm, pose in actual_by_arm.items()
                        if directions.get(arm) is not None
                    }
                actual_left = actual_by_arm.get("left", left_step)
                actual_right = actual_by_arm.get("right", right_step)
                targets[sequence_index - 1] = (label, actual_left, actual_right)
                self._rebase_post_movel_targets_after_force_clamp(
                    targets,
                    sequence_index,
                    actual_by_arm,
                    {"left": left_step, "right": right_step},
                )
                results.append(f"{detail}; {force_detail}")
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "FORCE_CLAMP_CONFIRMED",
                    f"{force_detail}; nominal Step1 distance ignored; "
                    "Step2 will use the actual force-confirmed pose",
                )
                if (
                    drag_tf_reanchor_enabled
                    and label == "step1"
                    and drag_mode
                    and delayed_left_join
                ):
                    capture_detail = self._capture_drag_tf_right_grasp_relation()
                    results.append(capture_detail)
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "DRAG_TF_RIGHT_GRASP_RELATION_CAPTURED",
                        capture_detail,
                    )
                if label == "step1_left":
                    left_joined = True
                    active_arms = ("left", "right")
                    if drag_tf_reanchor_enabled:
                        reanchor_detail = (
                            self._reanchor_drag_tf_step2_from_actual_grasp(
                                targets,
                                sequence_index,
                                box_layer=box_layer,
                                model_label=model_label,
                            )
                        )
                        results.append(reanchor_detail)
                        self._publish_box_grasp_feedback(
                            goal_handle,
                            "DRAG_TF_STEP2_REANCHORED",
                            reanchor_detail,
                        )
                if self._boolean(
                    f"{force_parameter_prefix}_stop_after_clamp_confirmed"
                ):
                    results.append("force_clamp_stop_after_confirmed=true")
                    return " | ".join(results)
                continue
            if is_step4 and motion_mode == "movej":
                units_by_arm, targets_by_arm, _feedback_detail = (
                    self._current_step4_movej_targets(active_arms=active_arms)
                )
                movej_detail = self._execute_dual_arm_movej_targets(
                    goal_handle,
                    False,
                    "box_post_movel_step4_movej",
                    units_by_arm.get("left", []),
                    units_by_arm["right"],
                    targets_by_arm.get("left", []),
                    targets_by_arm["right"],
                    "post-grasp Step4 dual arm MoveJ",
                    "POST_MOVEL_STEP4_MOVEJ",
                    "MOVING_POST_MOVEL_STEP4_MOVEJ",
                    "WAITING_FOR_POST_MOVEL_STEP4_MOVEJ",
                    "POST_MOVEL_STEP4_MOVEJ_REACHED",
                    "post-grasp Step4 dual arm MoveJ",
                    active_arms=active_arms,
                )
                results.append(f"{detail}; {movej_detail}")
                if (
                    label == "step2"
                    and not drag_mode
                    and not right_arm_only
                    and self._boolean("box_step2_waist_endpoint_sync_enabled")
                ):
                    results.append(
                        self._execute_step2_waist_endpoint_sync(
                            goal_handle,
                            adapter,
                            box_layer,
                            dry_run,
                            step2_endpoint_sequence_after,
                        )
                    )
                continue
            try:
                if is_left_only_step:
                    motion_result = adapter.execute_single(
                        "left",
                        pose_to_sdk_target(left_step),
                        motion_mode,
                        self._float("box_post_movel_velocity_percent"),
                        self._boolean("direct_movel_blocking"),
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                    )
                elif right_arm_only and not left_joined:
                    motion_result = adapter.execute_single(
                        "right",
                        pose_to_sdk_target(right_step),
                        motion_mode,
                        self._float("box_post_movel_velocity_percent"),
                        self._boolean("direct_movel_blocking"),
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                    )
                else:
                    motion_result = adapter.execute_dual(
                        pose_to_sdk_target(left_step),
                        pose_to_sdk_target(right_step),
                        motion_mode,
                        self._float("box_post_movel_velocity_percent"),
                        self._boolean("direct_movel_blocking"),
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                    )
            except RealManSdkCanceled as exc:
                raise MissionCanceled(str(exc)) from exc
            except (RealManSdkError, ValueError) as exc:
                raise MissionError(
                    f"post-grasp {', '.join(detail_arms)} {motion_mode} {label} "
                    f"({sequence_index}/{len(targets)}) "
                    f"failed: {exc}"
                ) from exc
            results.append(f"{detail}; {motion_result}")
            if (
                drag_tf_reanchor_enabled
                and label == "step1"
                and drag_mode
                and delayed_left_join
            ):
                capture_detail = self._capture_drag_tf_right_grasp_relation()
                results.append(capture_detail)
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "DRAG_TF_RIGHT_GRASP_RELATION_CAPTURED",
                    capture_detail,
                )
            if delayed_left_join and label == "step_drag3":
                join_left_target = left_step
                if drag_tf_reanchor_enabled:
                    join_left_target, reanchor_detail = (
                        self._reanchor_drag_tf_left_join_after_drag3()
                    )
                    targets[sequence_index - 1] = (
                        label,
                        join_left_target,
                        right_step,
                    )
                    results.append(reanchor_detail)
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "DRAG_TF_BOX_REANCHORED_AFTER_DRAG3",
                        reanchor_detail,
                    )
                results.append(
                    self._execute_drag_box_left_join(
                        goal_handle,
                        adapter,
                        join_left_target,
                        False,
                    )
                )
                left_joined = True
                active_arms = ("left", "right")
            if drag_tf_reanchor_enabled and label == "step1_left":
                reanchor_detail = self._reanchor_drag_tf_step2_from_actual_grasp(
                    targets,
                    sequence_index,
                    box_layer=box_layer,
                    model_label=model_label,
                )
                results.append(reanchor_detail)
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "DRAG_TF_STEP2_REANCHORED",
                    reanchor_detail,
                )
            if (
                label == "step2"
                and not drag_mode
                and not right_arm_only
                and self._boolean("box_step2_waist_endpoint_sync_enabled")
            ):
                results.append(
                    self._execute_step2_waist_endpoint_sync(
                        goal_handle,
                        adapter,
                        box_layer,
                        dry_run,
                        step2_endpoint_sequence_after,
                    )
                )
            if (
                label == "step2"
                and tf_carry_enabled
                and (
                    not right_arm_only
                    or (delayed_left_join and left_joined)
                )
            ):
                results.append(
                    self._execute_tf_body_home_carry(
                        goal_handle,
                        adapter,
                        False,
                        parameter_prefix=tf_carry_parameter_prefix,
                        box_layer=box_layer,
                        model_label=model_label,
                    )
                )
                self._rebase_post_movel_targets_after_tf_carry(
                    targets,
                    sequence_index,
                    box_layer=box_layer,
                    model_label=model_label,
                    drag_mode=drag_mode,
                    tf_mode=tf_mode,
                )
        return " | ".join(results) if results else "post_movel=no_steps"

    def _call_direct_box_movel(
        self,
        goal_handle,
        box_pose: PoseStamped,
        dry_run: bool,
        box_layer: int,
        *,
        drag_mode: bool = False,
        right_arm_only: bool = False,
        delayed_left_join: bool = False,
        tf_mode: bool = False,
        model_label: str | None = None,
    ) -> str:
        """Send final Link8 targets, optionally for the right arm only."""
        tf_carry_parameter_prefix = BoxSupportMixin._tf_body_home_carry_parameter_prefix(
            tf_mode=tf_mode,
            drag_mode=drag_mode,
        )
        path_carry_parameter_prefix = (
            tf_carry_parameter_prefix
            if tf_mode
            else (
                "drag_box_tf_body_home_carry"
                if drag_mode
                else "grasp_box_tf_body_home_carry"
            )
        )
        carry_enabled = self._boolean(
            f"{path_carry_parameter_prefix}_enabled"
        )
        if not tf_mode and carry_enabled:
            raise MissionError(
                f"{path_carry_parameter_prefix}_enabled is only valid for "
                f"/{'execute_drag_box_grasp_tf' if drag_mode else 'grasp_box_tf'}"
            )
        if carry_enabled:
            if self._boolean("box_step2_waist_endpoint_sync_enabled"):
                raise MissionError(
                    f"{path_carry_parameter_prefix}_enabled and "
                    "box_step2_waist_endpoint_sync_enabled are mutually exclusive"
                )
            if drag_mode and (
                not self._boolean("drag_box_left_arm_enabled")
                or (right_arm_only and not delayed_left_join)
            ):
                raise MissionError(
                    "drag_box_tf_body_home_carry_enabled requires the dual-arm "
                    "DragBox path with drag_box_left_arm_enabled=true"
                )
            if not drag_mode and (right_arm_only or delayed_left_join):
                raise MissionError(
                    "grasp_box_tf_body_home_carry_enabled requires the dual-arm "
                    "non-Drag GraspBox path"
                )
            if (
                self._string("box_grasp_execution_mode").strip().lower()
                != "joint123_then_arms"
            ):
                raise MissionError(
                    "TF waist carry requires box_grasp_execution_mode=joint123_then_arms"
                )
            post_enabled_parameter = (
                "drag_box_post_movel_enabled"
                if drag_mode
                else "box_post_movel_enabled"
            )
            if (
                not self._boolean(post_enabled_parameter)
                or self._integer("box_post_movel_step_count") < 2
            ):
                raise MissionError(
                    f"{path_carry_parameter_prefix}_enabled runs after Step2 and "
                    f"requires {post_enabled_parameter}=true with "
                    "box_post_movel_step_count>=2"
                )
        if (
            self._boolean("box_step2_waist_endpoint_sync_enabled")
            and not drag_mode
            and not right_arm_only
        ):
            self._step2_endpoint_sync_layer_config(box_layer)
        if tf_mode:
            frozen_box_pose = getattr(self, "_last_grasp_box_tf_box_pose", None)
            if frozen_box_pose is None:
                raise MissionError(
                    "TF GraspBox has no frozen base-frame box pose after detection"
                )
            left_tf_target, right_tf_target = self._make_tf_link8_target_poses(
                frozen_box_pose,
                box_layer,
                model_label,
                drag_mode=drag_mode,
            )
            box_transform = self._pose_stamped_to_transform(frozen_box_pose)
            self._last_grasp_box_tf_box_to_link7_targets = {
                "left": BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(box_transform),
                    self._pose_stamped_to_transform(left_tf_target),
                ),
                "right": BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(box_transform),
                    self._pose_stamped_to_transform(right_tf_target),
                ),
            }
            left_target, right_target, execution_detail = self._apply_tf_execution_mode(
                goal_handle,
                left_tf_target,
                right_tf_target,
                dry_run,
                box_layer,
                model_label,
                drag_mode=drag_mode,
            )
        else:
            arm_poses = getattr(self, "_last_box_pose_by_arm", {})
            left_target = self._make_direct_movel_pose(
                arm_poses.get("left", box_pose),
                "left",
            )
            right_target = self._make_direct_movel_pose(
                arm_poses.get("right", box_pose),
                "right",
            )
            left_target, right_target, execution_detail = (
                self._apply_joint1_execution_mode(
                    goal_handle,
                    left_target,
                    right_target,
                    dry_run,
                    box_layer,
                    model_label,
                )
            )
        z_equalization_detail = "target_z_equalization=not_applicable"
        if tf_mode and not right_arm_only and not delayed_left_join:
            left_target, right_target, z_equalization_detail = (
                BoxExecutionMixin._equalize_tf_dual_target_z(
                    self,
                    left_target,
                    right_target,
                    reference="average",
                )
            )
            # Keep the saved box->Link7 relations consistent with the exact
            # Z-equalized targets that will be sent to the SDK.  TF carry and
            # place logic consume these relations later in the same process.
            base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
            adjusted_world_targets = {}
            for arm, target in (
                ("left", left_target),
                ("right", right_target),
            ):
                base_to_arm_base = self._lookup_tf_carry_transform(
                    base_frame,
                    self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
                    parameter_prefix=path_carry_parameter_prefix,
                )
                arm_target = (
                    (
                        float(target.position.x),
                        float(target.position.y),
                        float(target.position.z),
                    ),
                    BoxSupportMixin._normalize_quaternion(
                        (
                            float(target.orientation.x),
                            float(target.orientation.y),
                            float(target.orientation.z),
                            float(target.orientation.w),
                        )
                    ),
                )
                adjusted_world_targets[arm] = BoxSupportMixin._compose_transform(
                    base_to_arm_base,
                    arm_target,
                )
            self._last_grasp_box_tf_box_to_link7_targets = {
                arm: BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(box_transform),
                    adjusted_world_targets[arm],
                )
                for arm in ("left", "right")
            }
        if self._string("box_grasp_execution_mode").lower() != "arms_only":
            self._publish_box_grasp_feedback(
                goal_handle,
                "FROZEN_PRE_JOINT1_TARGETS",
                "freezing the detected left/right Link8 targets before moving "
                "body joint1; FoundationPose will not be called again",
            )
        left_values = (
            left_target.position.x,
            left_target.position.y,
            left_target.position.z,
        )
        right_values = (
            right_target.position.x,
            right_target.position.y,
            right_target.position.z,
        )
        motion_mode = self._string("direct_movel_motion_mode").strip().lower()
        compensation_detail = (
            "fixture-center compensated"
            if self._boolean("direct_movel_fixture_compensation_enabled")
            else "fixture compensation disabled"
        )
        target_mode = self._string("direct_movel_target_mode").strip().lower()
        orientation_mode = target_mode
        if target_mode == "camera_offset":
            orientation_mode = (
                "current_tf"
                if self._boolean("direct_movel_use_current_fixture_orientation")
                else "fixed_config"
            )
        left_correction_name = self._joint123_target_correction_parameter_name(
            "left",
            box_layer,
            model_label,
            tf_mode=tf_mode,
            drag_mode=drag_mode,
        )
        right_correction_name = self._joint123_target_correction_parameter_name(
            "right",
            box_layer,
            model_label,
            tf_mode=tf_mode,
            drag_mode=drag_mode,
        )
        left_correction = self._float_array(left_correction_name)
        right_correction = self._float_array(right_correction_name)
        detail = (
            f"direct {motion_mode} Link8 targets ({compensation_detail}) in "
            "left/right arm base frames: "
            f"left=[{left_values[0]:.3f}, {left_values[1]:.3f}, "
            f"{left_values[2]:.3f}] m, "
            f"right=[{right_values[0]:.3f}, {right_values[1]:.3f}, "
            f"{right_values[2]:.3f}] m; orientation={orientation_mode}, "
            f"left_q=[{left_target.orientation.x:.3f}, "
            f"{left_target.orientation.y:.3f}, "
            f"{left_target.orientation.z:.3f}, "
            f"{left_target.orientation.w:.3f}], "
            f"right_q=[{right_target.orientation.x:.3f}, "
            f"{right_target.orientation.y:.3f}, "
            f"{right_target.orientation.z:.3f}, "
            f"{right_target.orientation.w:.3f}]; "
            "position_offset_frame=foundationpose_box; "
            + (
                "camera_extrinsics=live_tf_tree"
                if tf_mode
                else (
                    "camera_extrinsics=live_link8_eepose"
                    if self._boolean("camera_dynamic_link8_extrinsics_enabled")
                    else "camera_extrinsics=legacy_fixed_base"
                )
            )
            + "; "
            f"box_target_correction_left=[{','.join(f'{value:.4f}' for value in left_correction)}]; "
            f"box_target_correction_right=[{','.join(f'{value:.4f}' for value in right_correction)}]; "
            f"target_correction_frame=foundationpose_box; "
            f"target_correction_layer={box_layer}; "
            f"arm_motion={('right_only_until_drag3' if delayed_left_join else ('right_only' if right_arm_only else 'dual'))}; "
            f"{z_equalization_detail}; {execution_detail}"
        )
        self._publish_box_grasp_feedback(goal_handle, "DIRECT_MOVEL_TARGETS", detail)
        post_right_arm_only = right_arm_only and not delayed_left_join
        if dry_run:
            post_detail = self._execute_post_movel_sequence(
                goal_handle,
                None,
                left_target,
                right_target,
                True,
                box_layer=box_layer,
                drag_mode=drag_mode,
                right_arm_only=right_arm_only,
                delayed_left_join=delayed_left_join,
                model_label=model_label,
                tf_mode=tf_mode,
            )
            post_arm_detail = self._execute_post_arm_movej(
                goal_handle, True, right_arm_only=post_right_arm_only
            )
            body_home_detail = (
                "body_home=skipped_after_tf_carry"
                if getattr(self, "_last_tf_body_home_carry_completed", False)
                else (
                    "body_home=skipped_after_step2_endpoint_sync"
                    if (
                        getattr(self, "_last_step2_endpoint_sync_completed", False)
                        and self._boolean(
                            "box_step2_waist_endpoint_sync_skip_final_body_home"
                        )
                    )
                    else self._execute_body_home(goal_handle, True)
                )
            )
            return (
                f"{detail}; direct {motion_mode} skipped in dry-run; "
                f"{post_detail}; {post_arm_detail}; {body_home_detail}"
            )

        backend = self._string("direct_motion_backend").strip().lower()
        if (
            self._boolean("box_post_movel_enabled")
            or (drag_mode and self._boolean("drag_box_post_movel_enabled"))
            or self._boolean("box_post_arm_movej_enabled")
            or self._boolean("box_body_return_home_enabled")
        ) and backend != "python_sdk":
            raise MissionError(
                "post-grasp MoveL/MoveJ and body-home actions require "
                "direct_motion_backend=python_sdk"
            )
        if backend == "python_sdk":
            adapter = getattr(self, "direct_sdk_adapter", None)
            if adapter is None:
                raise MissionError(
                    "direct_motion_backend=python_sdk but the SDK adapter "
                    "is not initialized"
                )
            try:
                if right_arm_only:
                    motion_result = adapter.execute_single(
                        "right",
                        pose_to_sdk_target(right_target),
                        motion_mode,
                        self._float("direct_movel_velocity_percent"),
                        self._boolean("direct_movel_blocking"),
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                    )
                else:
                    motion_result = adapter.execute_dual(
                        pose_to_sdk_target(left_target),
                        pose_to_sdk_target(right_target),
                        motion_mode,
                        self._float("direct_movel_velocity_percent"),
                        self._boolean("direct_movel_blocking"),
                        cancel_requested=lambda: goal_handle.is_cancel_requested,
                        timeout_sec=self._float("direct_sdk_motion_timeout_sec"),
                    )
                post_detail = self._execute_post_movel_sequence(
                    goal_handle,
                    adapter,
                    left_target,
                    right_target,
                    False,
                    box_layer=box_layer,
                    drag_mode=drag_mode,
                    right_arm_only=right_arm_only,
                    delayed_left_join=delayed_left_join,
                    model_label=model_label,
                    tf_mode=tf_mode,
                )
                post_arm_detail = self._execute_post_arm_movej(
                    goal_handle, False, right_arm_only=post_right_arm_only
                )
                body_home_detail = (
                    "body_home=skipped_after_tf_carry"
                    if getattr(self, "_last_tf_body_home_carry_completed", False)
                    else (
                        "body_home=skipped_after_step2_endpoint_sync"
                        if (
                            getattr(self, "_last_step2_endpoint_sync_completed", False)
                            and self._boolean(
                                "box_step2_waist_endpoint_sync_skip_final_body_home"
                            )
                        )
                        else self._execute_body_home(goal_handle, False)
                    )
                )
                return (
                    f"{detail}; {motion_result}; {post_detail}; "
                    f"{post_arm_detail}; {body_home_detail}"
                )
            except RealManSdkCanceled as exc:
                raise MissionCanceled(str(exc)) from exc
            except (RealManSdkError, ValueError) as exc:
                raise MissionError(str(exc)) from exc

        if backend != "ros_service":
            raise MissionError(f"unsupported direct_motion_backend: {backend}")
        if MoveCartesian is None:
            raise MissionError(
                "direct_motion_backend=ros_service requires "
                "task_interfaces.srv.MoveCartesian"
            )
        if self.direct_movel_client is None:
            raise MissionError("ROS service motion backend is not initialized")
        if right_arm_only:
            raise MissionError(
                "right-only or delayed-left DragBox requires "
                "direct_motion_backend=python_sdk so arm phases can be sequenced"
            )
        service_name = self._string("direct_movel_service_name")
        self._wait_for_service(self.direct_movel_client, service_name, goal_handle)
        request = MoveCartesian.Request()
        request.left_pose = left_target
        request.right_pose = right_target
        request.arm = MoveCartesian.Request.DUAL
        request.velocity = self._float("direct_movel_velocity_percent")
        request.blocking = self._boolean("direct_movel_blocking")
        request.dry_run = False
        request.motion_type = (
            MoveCartesian.Request.MOVEJ_P
            if motion_mode == "movej_p"
            else MoveCartesian.Request.MOVEL
        )
        future = self.direct_movel_client.call_async(request)
        response = self._wait_future(
            future,
            goal_handle,
            f"direct {motion_mode} service {service_name}",
            self._float("pickup_task_result_timeout_sec"),
            cancel_local_future=False,
        )
        if not response.success:
            raise MissionError(str(response.message))
        return f"{detail}; {response.message}"
