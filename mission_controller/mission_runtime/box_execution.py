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
        standard_post_movel_enabled = self._boolean("box_post_movel_enabled")
        drag_post_movel_enabled = drag_mode and self._boolean(
            "drag_box_post_movel_enabled"
        )
        if not standard_post_movel_enabled and not drag_post_movel_enabled:
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
            detail_arms = ("left",) if is_left_only_step else active_arms
            detail = (
                f"post-grasp {', '.join(detail_arms)} arm {motion_mode} {label} "
                f"({sequence_index}/{len(targets)}) "
                "in left/right arm "
                f"base frames: "
                f"{BoxSupportMixin._dual_target_position_detail(left_step, right_step)}; "
                "delta_frame=foundationpose_box; "
                "orientation unchanged from the initial Link8 targets"
            )
            self._publish_box_grasp_feedback(
                goal_handle,
                f"POST_MOVEL_{label.upper()}_TARGETS",
                detail,
            )
            if dry_run:
                results.append(f"{detail}; skipped in dry-run")
                if delayed_left_join and label == "step_drag3":
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
                    and tf_mode
                    and not drag_mode
                    and not right_arm_only
                    and self._boolean("grasp_box_tf_body_home_carry_enabled")
                ):
                    results.append(
                        self._execute_tf_body_home_carry(goal_handle, None, True)
                    )
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
            if delayed_left_join and label == "step_drag3":
                results.append(
                    self._execute_drag_box_left_join(
                        goal_handle,
                        adapter,
                        left_step,
                        False,
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
                        adapter,
                        box_layer,
                        dry_run,
                        step2_endpoint_sequence_after,
                    )
                )
            if (
                label == "step2"
                and tf_mode
                and not drag_mode
                and not right_arm_only
                and self._boolean("grasp_box_tf_body_home_carry_enabled")
            ):
                results.append(
                    self._execute_tf_body_home_carry(goal_handle, adapter, False)
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
        if self._boolean("grasp_box_tf_body_home_carry_enabled"):
            if not tf_mode:
                raise MissionError(
                    "grasp_box_tf_body_home_carry_enabled is only valid for /grasp_box_tf"
                )
            if drag_mode or right_arm_only or delayed_left_join:
                raise MissionError(
                    "TF waist carry currently requires the dual-arm non-Drag GraspBox path"
                )
            if (
                self._string("box_grasp_execution_mode").strip().lower()
                != "joint123_then_arms"
            ):
                raise MissionError(
                    "TF waist carry requires box_grasp_execution_mode=joint123_then_arms"
                )
            if (
                not self._boolean("box_post_movel_enabled")
                or self._integer("box_post_movel_step_count") < 2
            ):
                raise MissionError(
                    "TF waist carry runs after Step2 and requires "
                    "box_post_movel_enabled=true with box_post_movel_step_count>=2"
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
            f"{execution_detail}"
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
