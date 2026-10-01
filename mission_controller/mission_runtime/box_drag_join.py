"""DragBox TF reanchoring and delayed left-arm join operations."""

import math
import random
from collections import deque
from copy import deepcopy
import time

from geometry_msgs.msg import Pose, PoseStamped

from .common import MissionCanceled, MissionError, rotate_vector
from .arm_target_visualization import publish_drag_join_targets
from .realman_sdk_adapter import (
    RealManSdkCanceled,
    RealManSdkError,
    pose_to_sdk_target,
)


# Bound to the composed compatibility facade by box_support.py.
BoxSupportMixin = None


def offset_robot_forward(target, base_to_footprint, distance):
    """Translate along robot forward (-footprint Y), preserving orientation."""
    if not math.isfinite(distance):
        raise MissionError('Drag left forward offset must be finite')
    shift = rotate_vector((0.0, -distance, 0.0), base_to_footprint[1])
    return (tuple(target[0][i]+shift[i] for i in range(3)), target[1])


def fixed_contact_lateral_span(left_link, right_link, box_rotation,
                               left_fixture, right_fixture, span):
    """Set contact separation along box -Z; preserve orthogonal offsets."""
    if not math.isfinite(span) or span <= 0.0:
        raise MissionError("Drag contact span must be finite and positive")
    axis = rotate_vector((0.0, 0.0, -1.0), box_rotation)
    norm = math.sqrt(sum(v*v for v in axis))
    if norm < 1e-9:
        raise MissionError("Drag contact span requires a valid box orientation")
    axis = tuple(v/norm for v in axis)
    lc = tuple(a+b for a,b in zip(left_link[0], rotate_vector(left_fixture, left_link[1])))
    rc = tuple(a+b for a,b in zip(right_link[0], rotate_vector(right_fixture, right_link[1])))
    current = sum((a-b)*v for a,b,v in zip(lc,rc,axis))
    return (tuple(p+(span-current)*v for p,v in zip(left_link[0],axis)), left_link[1])


class BoxDragJoinMixin:
    """Reanchor the dragged box and join its left-arm grasp."""

    def _drag_tf_apply_fixed_contact_span(self, left_world, right_world, box, model_label):
        if str(model_label).strip().lower() != "bigbox":
            return left_world
        span = self._float("drag_box_tf_left_join_contact_span_m_bigbox")
        if span == 0.0:
            return left_world
        return fixed_contact_lateral_span(
            left_world, right_world, box[1],
            self._float_array("left_fixture_center_in_link8_xyz"),
            self._float_array("right_fixture_center_in_link8_xyz"), span,
        )

    def _drag_tf_scaled_left_join_world(self, current_box, fallback_target):
        """Rebuild left contact from its raw relation; apply k once after Drag3."""
        if self._float("drag_box_tf_left_contact_forward_delta_scale") == 1.0:
            return fallback_target
        relations = getattr(self, "_last_drag_box_tf_unscaled_contact_relations", None)
        if not relations or "left" not in relations:
            raise MissionError("DragBox left scaling requires the unscaled contact relation")
        contact_world = BoxSupportMixin._compose_transform(current_box, relations["left"])
        frozen_pose = self._last_grasp_box_tf_box_pose
        box_pose = PoseStamped()
        box_pose.header = deepcopy(frozen_pose.header)
        box_pose.pose = self._endpoint_sync_transform_to_pose(current_box)
        contact_pose = PoseStamped()
        contact_pose.header = deepcopy(frozen_pose.header)
        contact_pose.pose = self._endpoint_sync_transform_to_pose(contact_world)
        contact_pose = BoxSupportMixin._scale_grasp_contact_forward_delta(
            self, contact_pose, box_pose, "left", drag_mode=True
        )
        contact_pose.pose = BoxSupportMixin._make_direct_movel_pose(
            self, contact_pose, "left"
        )
        return BoxSupportMixin._pose_stamped_to_transform(contact_pose)

    def _drag_left_join_forward_offset(self, box_layer):
        distance = self._float("drag_box_tf_left_join_forward_offset_m")
        if box_layer in (1, 3, 4):
            override = self._float(f"drag_box_tf_left_join_forward_offset_m_layer{box_layer}")
            if not math.isfinite(override):
                raise MissionError(f"layer{box_layer} left-join forward offset must be finite")
            if override >= 0.0:
                distance = override
        return distance

    def _predict_drag_tf_left_join_target(
        self,
        frozen_box,
        desired_left_world,
        right_contact_world,
        left_arm_base_world,
        right_arm_base_world,
        *,
        box_layer: int,
        model_label: str | None,
    ) -> Pose:
        """Predict the runtime Drag3 re-anchor using SDK Tool-frame offsets.

        The right contact position cancels from the box displacement; its
        orientation determines the three Tool-frame translation directions.
        The same right-referenced arm-base Z adjustment is applied at runtime.
        """
        right_relation = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(frozen_box), right_contact_world
        )
        right_after_drag3 = right_contact_world
        for drag_index in range(1, 4):
            name = BoxSupportMixin._drag_post_movel_xyz_parameter_name(
                self, "right", drag_index, model_label,
                box_layer=box_layer, tf_mode=True, drag_mode=True,
            )
            delta = self._float_array(name)
            if len(delta) != 3 or not all(math.isfinite(value) for value in delta):
                raise MissionError(f"{name} must contain three finite offsets")
            right_after_drag3 = BoxSupportMixin._compose_transform(
                right_after_drag3,
                (tuple(float(value) for value in delta), (0.0, 0.0, 0.0, 1.0)),
            )
        predicted_box = BoxSupportMixin._compose_transform(
            right_after_drag3, BoxSupportMixin._inverse_transform(right_relation)
        )
        left_relation = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(frozen_box), desired_left_world
        )
        left_world = BoxSupportMixin._compose_transform(
            predicted_box, left_relation
        )
        left_world = BoxSupportMixin._drag_tf_scaled_left_join_world(
            self, predicted_box, left_world
        )
        left_pose = self._endpoint_sync_transform_to_pose(
            BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(left_arm_base_world), left_world
            )
        )
        right_pose = self._endpoint_sync_transform_to_pose(
            BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(right_arm_base_world),
                right_after_drag3,
            )
        )
        left_pose, _right_pose, _detail = self._equalize_tf_dual_target_z(
            left_pose, right_pose, reference="right"
        )
        left_world = BoxSupportMixin._compose_transform(
            left_arm_base_world,
            ((left_pose.position.x, left_pose.position.y, left_pose.position.z),
             (left_pose.orientation.x, left_pose.orientation.y, left_pose.orientation.z, left_pose.orientation.w)),
        )
        left_world = BoxDragJoinMixin._drag_tf_apply_fixed_contact_span(
            self, left_world, right_after_drag3, predicted_box, model_label
        )
        left_pose = self._endpoint_sync_transform_to_pose(BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(left_arm_base_world), left_world
        ))
        forward_offset = BoxDragJoinMixin._drag_left_join_forward_offset(self, box_layer)
        if not math.isfinite(forward_offset):
            raise MissionError("drag_box_tf_left_join_forward_offset_m must be finite")
        if forward_offset != 0.0:
            base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
            foot = self._lookup_tf_carry_transform(
                base_frame, "base_footprint", parameter_prefix="drag_box_tf_body_home_carry"
            )
            stamped = PoseStamped()
            stamped.pose = deepcopy(left_pose)
            world = BoxSupportMixin._compose_transform(
                left_arm_base_world, BoxSupportMixin._pose_stamped_to_transform(stamped)
            )
            world = offset_robot_forward(world, foot, forward_offset)
            left_pose = self._endpoint_sync_transform_to_pose(BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(left_arm_base_world), world
            ))
        return left_pose

    def _drag_left_join_base_seed_trials(
        self, right_joint_deg, *, preference=None
    ):
        """Use the same deterministic seed families before and after Drag3."""
        configured = self._float_array("drag_box_left_join_ik_seed_joint_deg")
        if len(configured) != 7 or not all(math.isfinite(x) for x in configured):
            raise MissionError(
                "drag_box_left_join_ik_seed_joint_deg must contain seven finite angles"
            )
        mirror_signs = (-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0)
        mirrored = [
            float(value) * mirror_signs[index]
            for index, value in enumerate(right_joint_deg)
        ]
        trials = [
            ("user_provided_fixed_joint_seed", configured),
            ("right_post_drag3_mirrored", mirrored),
            ("zero_fallback", [0.0] * 7),
        ]
        if preference is not None:
            for source, base_seed in (
                ("fixed_preferred_joint4", configured),
                ("mirrored_preferred_joint4", mirrored),
            ):
                seed = list(base_seed)
                seed[3] = preference[0]
                trials.append((source, seed))
        for index, label in ((1, "joint2"), (3, "joint4"),
                             (5, "joint6"), (6, "joint7")):
            for delta in (18.0, -18.0):
                seed = list(mirrored)
                seed[index] += delta
                if index == 3 and self._boolean("drag_box_left_join_joint4_negative_required"):
                    seed[index] = min(seed[index], -1.0)
                trials.append((f"mirrored_{label}_{delta:+.0f}deg", seed))
        unique = {}
        for source, seed in trials:
            unique.setdefault(tuple(round(value, 3) for value in seed),
                              (source, seed))
        return list(unique.values())

    def _drag_left_join_joint4_preference(
        self, box_layer: int, model_label: str | None
    ) -> tuple[float, float] | None:
        """Optional calibrated bigbox layer-1 J4 branch constraint."""
        if not self._boolean("drag_box_left_join_joint4_preference_enabled"):
            return None
        if str(model_label or "bigbox").strip().lower() != "bigbox" or box_layer != 1:
            return None
        preferred = self._float(
            "drag_box_left_join_preferred_joint4_deg_bigbox_layer1"
        )
        tolerance = self._float(
            "drag_box_left_join_joint4_tolerance_deg_bigbox_layer1"
        )
        if not math.isfinite(preferred) or not math.isfinite(tolerance) or tolerance <= 0:
            raise MissionError("bigbox layer-1 left-join Joint4 preference is invalid")
        return preferred, tolerance

    @staticmethod
    def _drag_left_join_joint4_acceptable(
        solution_deg, preference: tuple[float, float] | None,
        *, negative_required: bool,
    ) -> bool:
        joint4 = float(solution_deg[3])
        return (
            (not negative_required or joint4 < 0.0)
            and (preference is None or abs(joint4 - preference[0]) <= preference[1])
        )

    def _drag_left_join_seed_trials(
        self, right_joint_deg, *, preference=None
    ):
        """Build the full bounded seed list shared by prediction and execution."""
        trials = self._drag_left_join_base_seed_trials(
            right_joint_deg, preference=preference
        )
        seen = {tuple(round(value, 3) for value in seed) for _, seed in trials}
        random_count = self._integer("drag_box_left_join_ik_random_seed_attempts")
        lower = self._float_array("waist_workspace_left_arm_joint_min_deg")
        upper = self._float_array("waist_workspace_left_arm_joint_max_deg")
        if len(lower) != 7 or len(upper) != 7 or any(
            not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi
            for lo, hi in zip(lower, upper)
        ):
            raise MissionError("left-arm joint limits must contain seven finite ranges")
        random_upper = list(upper)
        if self._boolean("drag_box_left_join_joint4_negative_required"):
            random_upper[3] = min(random_upper[3], -0.001)
        if lower[3] > random_upper[3]:
            raise MissionError("left-arm Joint4 limits have no negative range")
        rng = random.Random(time.monotonic_ns())
        added = 0
        generated = 0
        while added < random_count and generated < max(1, random_count * 3):
            generated += 1
            seed = [rng.uniform(lo, hi) for lo, hi in zip(lower, random_upper)]
            key = tuple(round(value, 3) for value in seed)
            if key in seen:
                continue
            seen.add(key)
            added += 1
            trials.append((f"random_{added:03d}", seed))
        return trials, added, random_count

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

    def _drag_tf_world_transform_to_arm_pose(
        self,
        transform,
        arm: str,
        *,
        parameter_prefix: str = "drag_box_tf_body_home_carry",
    ) -> Pose:
        """Express one frozen-frame Link8 transform in the live arm base."""
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        arm_base_frame = self._string(f"{arm}_arm_base_frame").strip().lstrip("/")
        base_to_arm_base = self._lookup_tf_carry_transform(
            base_frame,
            arm_base_frame,
            parameter_prefix=parameter_prefix,
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

    def _reanchor_drag_tf_left_join_after_drag3(self, *, model_label=None, box_layer=None):
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
        left_world_target = BoxSupportMixin._drag_tf_scaled_left_join_world(
            self, current_box, left_world_target
        )
        left_target = self._drag_tf_world_transform_to_arm_pose(
            left_world_target, "left"
        )
        right_target = self._drag_tf_world_transform_to_arm_pose(
            actual_right, "right"
        )
        left_target, _right_target, z_detail = (
            BoxSupportMixin._equalize_tf_dual_target_z(
                self,
                left_target,
                right_target,
                reference="right",
            )
        )
        # Apply AFTER equalizing arm-base Z. Otherwise a tilted waist could
        # remove part of a footprint-forward shift. The desired pre-drag
        # relation is not changed, so rebuilding never accumulates the offset.
        left_base = self._lookup_tf_carry_transform(
            base_frame, self._string('left_arm_base_frame').strip().lstrip('/'),
            parameter_prefix='drag_box_tf_body_home_carry',
        )
        stamped = PoseStamped()
        stamped.pose = deepcopy(left_target)
        left_world_target = BoxSupportMixin._compose_transform(
            left_base, BoxSupportMixin._pose_stamped_to_transform(stamped)
        )
        left_world_target = BoxDragJoinMixin._drag_tf_apply_fixed_contact_span(
            self, left_world_target, actual_right, current_box, model_label
        )
        left_target = self._endpoint_sync_transform_to_pose(BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(left_base), left_world_target
        ))
        if str(model_label).strip().lower() == "bigbox":
            z_detail += (
                f"; left_join_contact_lateral_span_m={self._float('drag_box_tf_left_join_contact_span_m_bigbox'):.4f}; "
                "span_reference=actual_right_fixture_center; span_axis=box_negative_Z; "
                "span_applied_after_z_equalization_before_forward_trim=true"
            )
        forward_offset = BoxDragJoinMixin._drag_left_join_forward_offset(self, box_layer)
        if not math.isfinite(forward_offset):
            raise MissionError('drag_box_tf_left_join_forward_offset_m must be finite')
        if forward_offset != 0.0:
            left_base = self._lookup_tf_carry_transform(
                base_frame, self._string('left_arm_base_frame').strip().lstrip('/'),
                parameter_prefix='drag_box_tf_body_home_carry',
            )
            stamped = PoseStamped()
            stamped.pose = deepcopy(left_target)
            world_after_equalization = BoxSupportMixin._compose_transform(
                left_base, BoxSupportMixin._pose_stamped_to_transform(stamped)
            )
            foot = self._lookup_tf_carry_transform(
                base_frame, 'base_footprint', parameter_prefix='drag_box_tf_body_home_carry'
            )
            left_world_target = offset_robot_forward(world_after_equalization, foot, forward_offset)
            left_target = self._endpoint_sync_transform_to_pose(BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(left_base), left_world_target
            ))
        z_detail += (f'; left_forward_offset={forward_offset:.4f}m; '
                     'offset_frame=base_footprint_negative_Y; applied_after_z_equalization=true')
        publish_drag_join_targets(self, current_box, left_target, actual_right)
        self._last_drag_box_tf_reanchored_box_after_drag3 = current_box
        relation_by_arm = deepcopy(desired_relations)
        relation_by_arm["left"] = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(current_box), left_world_target
        )
        relation_by_arm["right"] = right_relation
        self._last_grasp_box_tf_box_to_link7_targets = relation_by_arm
        return left_target, (
            "drag_tf_reanchor_after_drag3=completed; authority=actual_right_link7; "
            f"inferred_box_position=[{current_box[0][0]:.9f},"
            f"{current_box[0][1]:.9f},{current_box[0][2]:.9f}]; "
            f"inferred_box_orientation=[{current_box[1][0]:.9f},"
            f"{current_box[1][1]:.9f},{current_box[1][2]:.9f},"
            f"{current_box[1][3]:.9f}]; "
            f"left_join_target=recomputed_from_common_box; {z_detail}"
        )

    def _precheck_drag_tf_left_join_after_contact(
        self, goal_handle, adapter, *, box_layer: int, model_label: str | None
    ) -> str:
        """Check the predicted left join after real right contact, before Drag1."""
        frozen_pose = getattr(self, "_last_grasp_box_tf_box_pose", None)
        desired = getattr(self, "_last_drag_box_tf_desired_box_to_link7_targets", None)
        if frozen_pose is None or not desired or "left" not in desired:
            raise MissionError("DragBox left-join precheck has no captured grasp")
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        frozen_box = self._pose_stamped_to_transform(frozen_pose)
        desired_left_world = BoxSupportMixin._compose_transform(
            frozen_box, desired["left"]
        )
        right_contact_world = self._lookup_tf_carry_transform(
            base_frame, self._string("right_link8_frame").strip().lstrip("/"),
            parameter_prefix="drag_box_tf_body_home_carry",
        )
        arm_bases = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
                parameter_prefix="drag_box_tf_body_home_carry",
            )
            for arm in ("left", "right")
        }
        target_pose = self._predict_drag_tf_left_join_target(
            frozen_box, desired_left_world, right_contact_world,
            arm_bases["left"], arm_bases["right"],
            box_layer=box_layer, model_label=model_label,
        )
        target = pose_to_sdk_target(target_pose)
        now = time.monotonic()
        max_age = self._float("box_pre_target_arm_movej_feedback_max_age_sec")
        with self.joint_state_lock:
            right_joint_rad = list(self.latest_slave_arm_positions.get("right", []))
            age = now - self.latest_slave_arm_state_times.get("right", 0.0)
        if len(right_joint_rad) < 7 or age > max_age:
            raise MissionError(
                "DragBox left-join precheck requires fresh right joint feedback: "
                f"age={age:.3f}s, limit={max_age:.3f}s"
            )
        preference = self._drag_left_join_joint4_preference(
            box_layer, model_label
        )
        trials, _random_added, _random_count = self._drag_left_join_seed_trials(
            [math.degrees(value) for value in right_joint_rad[:7]],
            preference=preference,
        )
        limit = min(self._integer("drag_box_left_join_ik_max_attempts"), len(trials))
        joint4_negative = self._boolean("drag_box_left_join_joint4_negative_required")
        for attempt, (source, seed) in enumerate(trials[:limit], start=1):
            self._check_canceled(goal_handle, "while checking drag left join before Drag1")
            solution = adapter.solve_ik("left", target, seed)
            if solution is None or len(solution) != 7 or not all(
                math.isfinite(value) for value in solution
            ):
                continue
            if not self._drag_left_join_joint4_acceptable(
                solution, preference, negative_required=joint4_negative
            ):
                continue
            detail = (
                "predicted Drag3 left join IK is available before dragging; "
                f"target={[round(value, 5) for value in target]}; "
                f"attempt={attempt}/{limit}; seed_source={source}"
            )
            self._publish_box_grasp_feedback(
                goal_handle, "DRAG_LEFT_JOIN_PRECHECK_PASSED", detail
            )
            return detail
        raise MissionError(
            "predicted Drag3 left join has no IK after actual right contact; "
            f"target={[round(value, 6) for value in target]}; "
            f"attempts={limit}; Drag1/2/3 were not started"
        )

    def _reanchor_drag_tf_step2_from_actual_grasp(
        self,
        targets,
        next_target_index: int,
        *,
        box_layer: int,
        model_label: str | None,
        drag_mode: bool = True,
    ) -> str:
        """Build an equal-height rigid Step2 lift from actual bilateral TF.

        The configured Step2 X component is the lift distance, but the motion
        is deliberately applied along the common ``base_link`` +Z axis.  Both
        Link8 targets are derived from one shared reference transform so their
        contact span cannot shrink during the lift.
        """
        action_name = "DragBox TF" if drag_mode else "GraspBox TF"
        action_prefix = "drag_box_tf" if drag_mode else "grasp_box_tf"
        carry_prefix = f"{action_prefix}_body_home_carry"
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        actual_link = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                parameter_prefix=carry_prefix,
            )
            for arm in ("left", "right")
        }
        height_error = abs(
            float(actual_link["left"][0][2])
            - float(actual_link["right"][0][2])
        )
        height_tolerance = self._float(
            "box_tf_step2_contact_height_tolerance_m"
        )
        if height_error > height_tolerance:
            raise MissionError(
                f"{action_name} rigid Step2 contact height mismatch: "
                f"left_z={actual_link['left'][0][2]:.6f}, "
                f"right_z={actual_link['right'][0][2]:.6f}, "
                f"error={height_error:.6f}m > {height_tolerance:.6f}m"
            )

        common_z = 0.5 * (
            float(actual_link["left"][0][2])
            + float(actual_link["right"][0][2])
        )
        leveled_link = {
            arm: (
                (
                    float(actual_link[arm][0][0]),
                    float(actual_link[arm][0][1]),
                    common_z,
                ),
                actual_link[arm][1],
            )
            for arm in ("left", "right")
        }
        midpoint = tuple(
            0.5 * (
                float(leveled_link["left"][0][axis])
                + float(leveled_link["right"][0][axis])
            )
            for axis in range(3)
        )
        # The orientation of this shared reference cancels algebraically; use
        # the current left Link8 orientation to keep the transform well posed.
        current_box = (midpoint, leveled_link["left"][1])
        relation_by_arm = {
            arm: BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(current_box),
                leveled_link[arm],
            )
            for arm in ("left", "right")
        }

        left_parameter = self._post_movel_xyz_parameter_name(
            "left",
            2,
            model_label,
            box_layer=box_layer,
            tf_mode=True,
            drag_mode=drag_mode,
        )
        right_parameter = self._post_movel_xyz_parameter_name(
            "right",
            2,
            model_label,
            box_layer=box_layer,
            tf_mode=True,
            drag_mode=drag_mode,
        )
        left_delta = tuple(float(value) for value in self._float_array(left_parameter))
        right_delta = tuple(
            float(value) for value in self._float_array(right_parameter)
        )
        if len(left_delta) != 3 or len(right_delta) != 3:
            raise MissionError(f"{action_name} Step2 deltas must contain three values")
        if any(
            abs(left_delta[index] - right_delta[index]) > 1.0e-6
            for index in range(3)
        ):
            raise MissionError(
                f"{action_name} rigid Step2 requires identical left/right "
                f"deltas: left={left_delta}, right={right_delta}"
            )
        if abs(left_delta[1]) > 1.0e-9 or abs(left_delta[2]) > 1.0e-9:
            raise MissionError(
                f"{action_name} rigid Step2 expects [lift,0,0], got {left_delta}"
            )
        lift_distance = float(left_delta[0])
        if not math.isfinite(lift_distance) or lift_distance <= 0.0:
            raise MissionError(
                f"{action_name} rigid Step2 lift must be positive, "
                f"got {lift_distance}"
            )

        # Left-multiplying a base-frame translation moves both TCPs through
        # exactly the same world-space vector without changing their span.
        step2_box = (
            (midpoint[0], midpoint[1], midpoint[2] + lift_distance),
            current_box[1],
        )
        world_targets = {
            arm: BoxSupportMixin._compose_transform(
                step2_box, relation_by_arm[arm]
            )
            for arm in ("left", "right")
        }
        target_height_error = abs(
            world_targets["left"][0][2] - world_targets["right"][0][2]
        )
        contact_span = self._endpoint_sync_pose_position_error(
            leveled_link["left"][0], leveled_link["right"][0]
        )
        target_span = self._endpoint_sync_pose_position_error(
            world_targets["left"][0], world_targets["right"][0]
        )
        span_error = abs(target_span - contact_span)
        span_tolerance = self._float("box_tf_step2_grasp_span_tolerance_m")
        if target_height_error > height_tolerance or span_error > span_tolerance:
            raise MissionError(
                f"{action_name} rigid Step2 invariant failed: "
                f"height_error={target_height_error:.9f}m, "
                f"span_error={span_error:.9f}m"
            )
        step2_poses = {
            arm: self._drag_tf_world_transform_to_arm_pose(
                world_targets[arm],
                arm,
                parameter_prefix=carry_prefix,
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
            raise MissionError(f"{action_name} could not find Step2 target")
        targets[step2_index] = (
            "step2",
            step2_poses["left"],
            step2_poses["right"],
        )
        self._last_grasp_box_tf_box_to_link7_targets = relation_by_arm
        return (
            f"{action_prefix}_step2_rigid_lift=completed; "
            "common_frame=base_link; common_box_frame=true; "
            f"lift_z={lift_distance:.4f}m; common_target_z="
            f"{world_targets['left'][0][2]:.6f}m; "
            f"contact_height_error={height_error:.6f}m; "
            f"grasp_span={contact_span:.6f}m; span_error={span_error:.9f}m; "
            f"measured_link8_frame={base_frame}; "
            f"actual_left_link8_xyz={list(actual_link['left'][0])}; "
            f"actual_left_link8_q={list(actual_link['left'][1])}; "
            f"actual_right_link8_xyz={list(actual_link['right'][0])}; "
            f"actual_right_link8_q={list(actual_link['right'][1])}"
        )

    def _drag_left_join_stage_after_movej_error(
        self,
        goal_handle,
        target_deg: list[float],
        sequence_before: int,
        timeout_sec: float,
    ) -> tuple[bool, str]:
        """Reconcile a rejected MoveJ with fresh, stationary arm feedback.

        A controller error can arrive after the arm has reached the target.
        Never resend while the first command may still be moving.
        """
        target_rad = [math.radians(value) for value in target_deg]
        position_tolerance = self._float(
            "box_pre_target_arm_movej_position_tolerance_rad"
        )
        velocity_tolerance = self._float(
            "box_pre_target_arm_movej_velocity_tolerance_rad_sec"
        )
        max_age = self._float("box_pre_target_arm_movej_feedback_max_age_sec")
        stable_required = self._integer("box_pre_target_arm_movej_stable_samples")
        started = time.monotonic()
        deadline = started + timeout_sec
        last_sequence = sequence_before
        reached_samples = 0
        stopped_elsewhere_since = None
        position_window = deque()
        latest_detail = "no fresh left-arm feedback"
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while reconciling delayed left-arm join MoveJ"
            )
            now = time.monotonic()
            with self.joint_state_lock:
                sequence = self.latest_slave_arm_state_sequences.get("left", 0)
                measured = list(self.latest_slave_arm_positions.get("left", []))
                velocity = list(self.latest_slave_arm_velocities.get("left", []))
                age = now - self.latest_slave_arm_state_times.get("left", 0.0)
            if sequence <= last_sequence:
                time.sleep(0.02)
                continue
            last_sequence = sequence
            if (
                len(measured) < 7
                or age > max_age
                or not all(math.isfinite(v) for v in measured[:7])
            ):
                reached_samples = 0
                stopped_elsewhere_since = None
                position_window.clear()
                latest_detail = f"invalid_or_stale_feedback(seq={sequence}, age={age:.3f}s)"
                time.sleep(0.02)
                continue
            position_window.append((now, measured[:7]))
            while len(position_window) > 1 and now - position_window[0][0] > 0.5:
                position_window.popleft()
            position_error = max(
                abs(measured[index] - target_rad[index]) for index in range(7)
            )
            max_velocity = (
                max(abs(value) for value in velocity[:7])
                if len(velocity) >= 7 and all(math.isfinite(v) for v in velocity[:7])
                else float("nan")
            )
            position_change = max(
                abs(measured[index] - position_window[0][1][index])
                for index in range(7)
            )
            moving = position_change > max(0.001, position_tolerance * 0.2)
            if math.isfinite(max_velocity) and max_velocity > velocity_tolerance:
                moving = True
            latest_detail = (
                f"measured_deg={[round(math.degrees(v), 3) for v in measured[:7]]}; "
                f"position_error_rad={position_error:.4f}; "
                f"position_change_0p5s_rad={position_change:.4f}; "
                f"max_velocity_rad_sec={max_velocity:.4f}; age_sec={age:.3f}"
            )
            if moving:
                reached_samples = 0
                stopped_elsewhere_since = None
            elif position_error <= position_tolerance:
                stopped_elsewhere_since = None
                reached_samples += 1
                if reached_samples >= stable_required:
                    return True, latest_detail
            else:
                reached_samples = 0
                if stopped_elsewhere_since is None:
                    stopped_elsewhere_since = now
                # A controller may return 1 before an accepted MoveJ begins.
                # A low reported velocity alone is not evidence of a stop.
                if now - started >= 2.0 and now - stopped_elsewhere_since >= 1.0:
                    return False, latest_detail
            time.sleep(0.02)
        raise MissionError(
            "delayed left-arm join MoveJ did not reach its target within "
            f"{timeout_sec:.1f}s after return_code=1; no retry was sent: "
            f"{latest_detail}"
        )

    def _execute_drag_box_left_join_staged_ik_movej(
        self,
        goal_handle,
        adapter,
        left_target: Pose,
        dry_run: bool,
        *,
        one_shot: bool = False,
        box_layer: int = 1,
        model_label: str | None = None,
    ) -> str:
        """Solve the reanchored join from live, mirrored, and random IK seeds.

        Joint2 through Joint7 move first while Joint1 retains its measured
        value. A second blocking MoveJ sends the complete seven-axis IK
        solution, so only Joint1 still needs to move. No configured safety
        posture and no controller-side MoveJ_P re-solve are used in this mode.
        """
        target = pose_to_sdk_target(left_target)
        if dry_run:
            detail = (
                "delayed left-arm staged IK MoveJ skipped in dry-run; "
                f"target={[round(value, 6) for value in target]}"
            )
            self._publish_box_grasp_feedback(
                goal_handle,
                "DRAG_LEFT_JOIN_STAGED_IK_MOVEJ",
                detail,
            )
            return detail
        if adapter is None:
            raise MissionError(
                "delayed left-arm staged IK MoveJ requires "
                "direct_motion_backend=python_sdk"
            )

        now = time.monotonic()
        max_age = self._float("box_pre_target_arm_movej_feedback_max_age_sec")
        with self.joint_state_lock:
            current_joint_rad = list(
                self.latest_slave_arm_positions.get("left", [])
            )
            right_current_joint_rad = list(
                self.latest_slave_arm_positions.get("right", [])
            )
            feedback_age = now - self.latest_slave_arm_state_times.get("left", 0.0)
            right_feedback_age = (
                now - self.latest_slave_arm_state_times.get("right", 0.0)
            )
        if (
            len(current_joint_rad) < 7
            or feedback_age > max_age
            or not all(math.isfinite(value) for value in current_joint_rad[:7])
        ):
            raise MissionError(
                "delayed left-arm join requires fresh seven-joint feedback for "
                "the Joint2-through-Joint7 stage: "
                f"joints={len(current_joint_rad)}, age={feedback_age:.3f}s, "
                f"limit={max_age:.3f}s"
            )
        if (
            len(right_current_joint_rad) < 7
            or right_feedback_age > max_age
            or not all(
                math.isfinite(value) for value in right_current_joint_rad[:7]
            )
        ):
            raise MissionError(
                "delayed left-arm join requires fresh seven-joint feedback from "
                "the right arm after Drag3 to build its mirrored IK seed: "
                f"joints={len(right_current_joint_rad)}, "
                f"age={right_feedback_age:.3f}s, limit={max_age:.3f}s"
            )
        current_joint_deg = [
            math.degrees(value) for value in current_joint_rad[:7]
        ]
        joint_min_deg = self._float_array("waist_workspace_left_arm_joint_min_deg")
        joint_max_deg = self._float_array("waist_workspace_left_arm_joint_max_deg")
        if len(joint_min_deg) != 7 or len(joint_max_deg) != 7:
            raise MissionError("left-arm IK ranking requires seven joint limits")
        max_ik_attempts = self._integer("drag_box_left_join_ik_max_attempts")
        joint4_negative_required = self._boolean(
            "drag_box_left_join_joint4_negative_required"
        )
        preference = self._drag_left_join_joint4_preference(
            box_layer, model_label
        )
        seed_trials, random_added, random_seed_attempts = (
            self._drag_left_join_seed_trials(
                [math.degrees(value) for value in right_current_joint_rad[:7]],
                preference=preference,
            )
        )
        attempt_limit = min(max_ik_attempts, len(seed_trials))
        solved_joint_deg = None
        solved_seed_deg = None
        solved_seed_source = ""
        solved_attempt = 0
        last_nonnegative_joint4 = None
        accepted_solutions = 0
        checked_solutions = set()
        best_rank = None
        for attempt, (seed_source, seed_joint_deg) in enumerate(
            seed_trials[:attempt_limit], start=1
        ):
            self._check_canceled(goal_handle, "while solving delayed left-arm join IK")
            self._publish_box_grasp_feedback(
                goal_handle,
                "DRAG_LEFT_JOIN_IK_SOLVING",
                "solving the reanchored left join Pose with offline SDK IK; "
                f"attempt={attempt}/{attempt_limit}; seed_source={seed_source}; "
                f"seed_deg={[round(value, 3) for value in seed_joint_deg]}; "
                "this step sends no robot motion",
            )
            try:
                candidate_joint_deg = adapter.solve_ik(
                    "left", target, seed_joint_deg
                )
            except (RealManSdkError, ValueError) as exc:
                raise MissionError(f"delayed left-arm join IK failed: {exc}") from exc
            if candidate_joint_deg is None:
                continue
            candidate_joint_deg = [float(value) for value in candidate_joint_deg]
            if len(candidate_joint_deg) != 7 or not all(
                math.isfinite(value) for value in candidate_joint_deg
            ):
                raise MissionError(
                    "delayed left-arm join IK did not return seven finite joints"
                )
            if not self._drag_left_join_joint4_acceptable(
                candidate_joint_deg, preference,
                negative_required=joint4_negative_required,
            ):
                if joint4_negative_required and candidate_joint_deg[3] >= 0.0:
                    last_nonnegative_joint4 = candidate_joint_deg[3]
                continue
            accepted_solutions += 1
            solution_key = tuple(round(value, 3) for value in candidate_joint_deg)
            if solution_key in checked_solutions:
                continue
            checked_solutions.add(solution_key)
            margin = min(
                min(value - lower, upper - value)
                for value, lower, upper in zip(
                    candidate_joint_deg, joint_min_deg, joint_max_deg
                )
            )
            joint_change = [
                abs(value - current)
                for value, current in zip(candidate_joint_deg, current_joint_deg)
            ]
            preferred_joint4 = (
                preference[0] if preference else
                -50.0 if self._boolean("drag_box_left_join_joint4_preference_enabled") else None
            )
            rank = (
                -margin,
                abs(candidate_joint_deg[3] - preferred_joint4) if preferred_joint4 is not None else 0.0,
                max(joint_change),
                sum(joint_change),
            )
            if best_rank is not None and rank >= best_rank:
                continue
            best_rank = rank
            solved_joint_deg = candidate_joint_deg
            solved_seed_deg = seed_joint_deg
            solved_seed_source = seed_source
            solved_attempt = attempt
        if solved_joint_deg is None:
            joint4_detail = (
                "; every returned solution had non-negative Joint4, "
                f"last_joint4={last_nonnegative_joint4:.3f}deg"
                if last_nonnegative_joint4 is not None
                else ""
            )
            raise MissionError(
                "delayed left-arm join IK has no acceptable IK solution after "
                f"{attempt_limit} distinct-seed attempts for the reanchored "
                f"target={target}; seed_sources="
                f"{[source for source, _ in seed_trials[:attempt_limit]]}; "
                f"random_seeds_generated={random_added}/{random_seed_attempts}"
                f"; joint4_preference={preference}"
                f"; ik_accepted={accepted_solutions}; "
                f"unique_ik_solutions={len(checked_solutions)}"
                f"{joint4_detail}"
            )
        self._publish_box_grasp_feedback(
            goal_handle,
            "DRAG_LEFT_JOIN_IK_SOLVED",
            "offline left-arm IK solved; "
            f"attempt={solved_attempt}/{attempt_limit}; "
            f"seed_source={solved_seed_source}; "
            f"seed_deg={[round(value, 3) for value in solved_seed_deg]}; "
            f"solution_deg={[round(value, 3) for value in solved_joint_deg]}; "
            f"joint4_preference={preference}; "
            f"acceptable_solutions={accepted_solutions}; "
            f"unique_ik_solutions={len(checked_solutions)}; "
            f"score_joint_margin_deg={-best_rank[0]:.3f}; "
            f"joint4_distance_from_preferred_deg={best_rank[1]:.3f}; "
            f"max_joint_change_deg={best_rank[2]:.3f}",
        )

        speed = self._float("drag_box_left_join_velocity_percent")
        timeout_sec = self._float("drag_box_left_join_timeout_sec")
        common_kwargs = {
            "blend_radius": 0,
            "trajectory_connect": 0,
            "cancel_requested": lambda: goal_handle.is_cancel_requested,
            "timeout_sec": timeout_sec,
        }
        if one_shot:
            self._publish_box_grasp_feedback(
                goal_handle,
                "DRAG_LEFT_JOIN_ONE_SHOT_MOVEJ_TARGETS",
                "moving directly from the avoidance posture to the complete "
                f"offline-IK joint solution: {[round(value, 3) for value in solved_joint_deg]}",
            )
            try:
                result = adapter.execute_single_movej(
                    "left", solved_joint_deg, speed, **common_kwargs
                )
            except RealManSdkCanceled as exc:
                raise MissionCanceled(str(exc)) from exc
            except (RealManSdkError, ValueError) as exc:
                raise MissionError(f"delayed left-arm one-shot MoveJ failed: {exc}") from exc
            return (
                "left_join=ik_movej; "
                f"ik_solution_deg={[round(value, 3) for value in solved_joint_deg]}; "
                f"motion={result}"
            )

        joint2_to_7_target_deg = list(solved_joint_deg)
        joint2_to_7_target_deg[0] = current_joint_deg[0]
        self._publish_box_grasp_feedback(
            goal_handle,
            "DRAG_LEFT_JOIN_JOINT2_TO_7_TARGETS",
            "first staged left-arm MoveJ keeps Joint1 fixed and moves Joint2 "
            "through Joint7 to the IK solution; "
            f"target_deg={[round(value, 3) for value in joint2_to_7_target_deg]}",
        )
        max_retries = 5
        for attempt in range(max_retries + 1):
            self._check_canceled(
                goal_handle, "before delayed left-arm join Joint2-through-Joint7 MoveJ"
            )
            with self.joint_state_lock:
                sequence_before = self.latest_slave_arm_state_sequences.get("left", 0)
            try:
                joint2_to_7_result = adapter.execute_single_movej(
                    "left",
                    joint2_to_7_target_deg,
                    speed,
                    **common_kwargs,
                )
                break
            except RealManSdkCanceled as exc:
                raise MissionCanceled(str(exc)) from exc
            except (RealManSdkError, ValueError) as exc:
                if not isinstance(exc, RealManSdkError) or not str(exc).endswith(
                    "return_code=1"
                ):
                    raise MissionError(
                        "delayed left-arm join Joint2-through-Joint7 MoveJ failed: "
                        f"{exc}"
                    ) from exc
                reached, state_detail = (
                    self._drag_left_join_stage_after_movej_error(
                        goal_handle, joint2_to_7_target_deg, sequence_before,
                        timeout_sec,
                    )
                )
                if reached:
                    joint2_to_7_result = (
                        "controller returned 1, but fresh stationary Joint2-through-"
                        f"Joint7 feedback confirmed the target; {state_detail}"
                    )
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "DRAG_LEFT_JOIN_JOINT2_TO_7_CONFIRMED_AFTER_ERROR",
                        joint2_to_7_result,
                    )
                    break
                if attempt >= max_retries:
                    raise MissionError(
                        "delayed left-arm join Joint2-through-Joint7 MoveJ failed "
                        f"after {max_retries} retries: {exc}; {state_detail}"
                    ) from exc
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "DRAG_LEFT_JOIN_JOINT2_TO_7_RETRY",
                    f"MoveJ return_code=1; arm stopped short of target; "
                    f"retry={attempt + 1}/{max_retries}; {state_detail}",
                )
        self._publish_box_grasp_feedback(
            goal_handle,
            "DRAG_LEFT_JOIN_JOINT2_TO_7_REACHED",
            "Joint2 through Joint7 reached the IK values; sending the complete "
            "solved joint target so only Joint1 still moves",
        )

        with self.joint_state_lock:
            sequence_before = self.latest_slave_arm_state_sequences.get("left", 0)
        try:
            final_result = adapter.execute_single_movej(
                "left",
                solved_joint_deg,
                speed,
                **common_kwargs,
            )
        except RealManSdkCanceled as exc:
            raise MissionCanceled(str(exc)) from exc
        except (RealManSdkError, ValueError) as exc:
            if not isinstance(exc, RealManSdkError) or not str(exc).endswith(
                "return_code=1"
            ):
                raise MissionError(
                    f"delayed left-arm join final MoveJ failed: {exc}"
                ) from exc
            reached, state_detail = self._drag_left_join_stage_after_movej_error(
                goal_handle, solved_joint_deg, sequence_before, timeout_sec
            )
            if not reached:
                raise MissionError(
                    "delayed left-arm join final MoveJ stopped short after "
                    f"return_code=1: {state_detail}"
                ) from exc
            final_result = (
                "controller returned 1, but fresh stationary feedback "
                f"confirmed the complete joint target; {state_detail}"
            )
        self._publish_box_grasp_feedback(
            goal_handle,
            "DRAG_LEFT_JOIN_FINAL_MOVEJ_REACHED",
            "left arm reached the complete offline-IK joint solution",
        )
        return (
            "left_join=staged_ik_movej; "
            f"ik_solution_deg={[round(value, 3) for value in solved_joint_deg]}; "
            f"joint2_to_7_motion={joint2_to_7_result}; final_motion={final_result}"
        )

    def _execute_drag_box_left_join(
        self,
        goal_handle,
        adapter,
        left_target: Pose,
        dry_run: bool,
        *,
        box_layer: int = 1,
        model_label: str | None = None,
    ) -> str:
        """Move the delayed left arm to its cumulative post-drag target."""
        if self._boolean(
            "drag_box_tf_calibration_left_join_joint_override_enabled"
        ):
            joint_target_deg = self._float_array(
                "drag_box_tf_calibration_left_join_joint_target_deg"
            )
            if len(joint_target_deg) != 7 or not all(
                math.isfinite(value) for value in joint_target_deg
            ):
                raise MissionError(
                    "calibration left-join joint target must contain seven "
                    "finite degree values"
                )
            joint_min_deg = self._float_array(
                "waist_workspace_left_arm_joint_min_deg"
            )
            joint_max_deg = self._float_array(
                "waist_workspace_left_arm_joint_max_deg"
            )
            if len(joint_min_deg) != 7 or len(joint_max_deg) != 7 or any(
                value < lower or value > upper
                for value, lower, upper in zip(
                    joint_target_deg, joint_min_deg, joint_max_deg
                )
            ):
                raise MissionError(
                    "calibration left-join joint target is outside configured "
                    "left-arm joint limits"
                )
            detail = (
                "calibration-only Drag3 left-join target is a ROS MoveJ "
                "joint posture (pose IK bypassed); "
                f"joint_deg={[round(value, 3) for value in joint_target_deg]}"
            )
            self._publish_box_grasp_feedback(
                goal_handle,
                "DRAG_LEFT_JOIN_CALIBRATION_JOINT_TARGET",
                detail,
            )
            if dry_run:
                return f"{detail}; skipped in dry-run"
            if adapter is None:
                raise MissionError(
                    "calibration left-join joint override requires "
                    "direct_motion_backend=python_sdk"
                )
            try:
                motion_result = adapter.execute_single_movej(
                    "left",
                    joint_target_deg,
                    self._float("drag_box_left_join_velocity_percent"),
                    blend_radius=0,
                    trajectory_connect=0,
                    cancel_requested=lambda: goal_handle.is_cancel_requested,
                    timeout_sec=self._float("drag_box_left_join_timeout_sec"),
                )
            except (RealManSdkCanceled, MissionCanceled):
                raise
            except (RealManSdkError, ValueError) as exc:
                raise MissionError(
                    f"calibration-only Drag3 left-join MoveJ failed: {exc}"
                ) from exc
            self._publish_box_grasp_feedback(
                goal_handle,
                "DRAG_LEFT_JOIN_CALIBRATION_JOINT_REACHED",
                f"left arm reached the temporary calibration joint target; "
                f"{motion_result}",
            )
            return f"{detail}; {motion_result}"

        motion_mode = self._string("drag_box_left_join_motion_mode").strip().lower()
        if motion_mode not in ("movel", "movej_p", "staged_ik_movej", "ik_movej"):
            raise MissionError(
                "drag_box_left_join_motion_mode must be 'movel', 'movej_p', "
                "'staged_ik_movej', or 'ik_movej'"
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
        if motion_mode in ("staged_ik_movej", "ik_movej"):
            staged_result = self._execute_drag_box_left_join_staged_ik_movej(
                goal_handle,
                adapter,
                left_target,
                dry_run,
                one_shot=motion_mode == "ik_movej",
                box_layer=box_layer,
                model_label=model_label,
            )
            return f"{detail}; {staged_result}; left_join=confirmed"
        if adapter is None:
            if not dry_run:
                raise MissionError(
                    "delayed left-arm join requires direct_motion_backend=python_sdk"
                )
        if dry_run:
            return f"{detail}; skipped in dry-run"
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
        return f"{detail}; {motion_result}; left_join=confirmed"
