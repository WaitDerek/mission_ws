"""TF box-grasp waist search and offline endpoint IK planning."""

import math
from concurrent.futures import Future
import threading
import time

from geometry_msgs.msg import Pose, PoseStamped

from .common import MissionError
from .realman_sdk_adapter import RealManSdkError, pose_to_sdk_target
from .waist_workspace_optimizer import WaistWorkspaceOptimizer


# Bound to the composed compatibility facade by box_support.py.
BoxSupportMixin = None


class BoxWaistPlanningMixin:
    """Score candidate waist angles without moving the robot."""

    def _solve_post_waist_live_tf_movej_ik(
        self,
        goal_handle,
        adapter,
        left_target: Pose,
        right_target: Pose,
        optimized_workspace,
        *,
        right_arm_only: bool,
    ) -> dict[str, list[float]]:
        """Re-solve the actual post-waist arm-base targets before any MoveJ.

        The workspace optimizer uses configured waist FK for *hypothetical*
        angles. After the waist reaches its selected angle, the targets have
        already been re-expressed through the live arm-base TF. Its cached IK
        solutions are useful seeds, but must not be sent as final commands.
        """
        arms = ("right",) if right_arm_only else ("left", "right")
        targets = {"left": left_target, "right": right_target}
        minimum_margin = self._float("waist_workspace_minimum_margin_deg")
        require_negative_joint4 = self._boolean("box_ik_joint4_negative_required")
        solved_targets: dict[str, list[float]] = {}
        for arm in arms:
            pose = pose_to_sdk_target(targets[arm])
            predicted = list(getattr(optimized_workspace, f"{arm}_joint_deg"))
            if len(predicted) != 7 or not all(math.isfinite(v) for v in predicted):
                raise MissionError(
                    f"{arm} predicted waist-workspace IK seed is invalid"
                )
            lower = self._float_array(
                f"waist_workspace_{arm}_arm_joint_min_deg"
            )
            upper = self._float_array(
                f"waist_workspace_{arm}_arm_joint_max_deg"
            )
            if len(lower) != 7 or len(upper) != 7:
                raise MissionError(f"{arm} arm joint limits must contain seven values")
            self._publish_box_grasp_feedback(
                goal_handle,
                "POST_WAIST_LIVE_TF_IK_SOLVING",
                f"{arm} arm: solving the post-waist live-TF Link8 target "
                f"in its current arm base; target={pose}; no arm motion sent",
            )
            current = None
            if hasattr(self, "joint_state_lock"):
                with self.joint_state_lock:
                    measured = list(self.latest_slave_arm_positions.get(arm, []))
                    age = time.monotonic() - self.latest_slave_arm_state_times.get(arm, 0.0)
                if len(measured) == 7 and age <= 1.0 and all(math.isfinite(v) for v in measured):
                    current = [math.degrees(value) for value in measured]
            attempts = []
            if current is not None:
                attempts.append(("current_joint_feedback", current))
            attempts.extend([("optimized_solution", predicted), ("zero_fallback", [0.0] * 7)])
            # Explore redundant-axis seeds, never alter the requested target pose.
            for source, base in (("current", current), ("optimized", predicted)):
                if base is None:
                    continue
                for delta in (-30.0, -15.0, 15.0, 30.0):
                    seed = list(base)
                    seed[2] += delta
                    seed[4] -= delta
                    seed = [max(lo, min(hi, value)) for value, lo, hi in zip(seed, lower, upper)]
                    attempts.append((f"{source}_redundancy_{delta:+g}", seed))
            seen = set()
            unique_attempts = []
            for source, seed in attempts:
                key = tuple(round(v, 6) for v in seed)
                if key not in seen:
                    seen.add(key)
                    unique_attempts.append((source, seed))
            attempts = unique_attempts
            rejection_details = []
            for source, seed in attempts:
                if hasattr(self, "_check_canceled"):
                    self._check_canceled(goal_handle, "while solving post-waist arm IK")
                solution = adapter.solve_ik(arm, pose, seed)
                if solution is None:
                    rejection_details.append(f"{source}: no IK solution")
                    continue
                joints = [float(value) for value in solution]
                if len(joints) != 7 or not all(math.isfinite(v) for v in joints):
                    rejection_details.append(f"{source}: invalid seven-joint solution")
                    continue
                if require_negative_joint4 and joints[3] >= 0.0:
                    rejection_details.append(
                        f"{source}: Joint4={joints[3]:.3f}deg is not negative"
                    )
                    continue
                margin = min(
                    min(value - lo, hi - value)
                    for value, lo, hi in zip(joints, lower, upper)
                )
                if margin < minimum_margin:
                    rejection_details.append(
                        f"{source}: joint margin={margin:.3f}deg "
                        f"< {minimum_margin:.3f}deg"
                    )
                    continue
                solved_targets[arm] = joints
                self._publish_box_grasp_feedback(
                    goal_handle,
                    "POST_WAIST_LIVE_TF_IK_SOLVED",
                    f"{arm} arm: source={source}; "
                    f"joint_deg=[{','.join(f'{value:.3f}' for value in joints)}]; "
                    f"minimum_margin={margin:.3f}deg; no arm motion sent",
                )
                break
            else:
                raise MissionError(
                    f"{arm} post-waist live-TF target has no acceptable offline IK: "
                    + "; ".join(rejection_details)
                    + "; no MoveJ command sent"
                )
        return solved_targets

    def _waist_workspace_candidate_poses(
        self,
        left_frozen_target: PoseStamped,
        right_frozen_target: PoseStamped,
        candidate_waist,
        *,
        equalize_target_z: bool,
    ) -> tuple[Pose, Pose]:
        """Express the frozen dual-arm target in candidate arm-base frames."""
        left_pose = self._tf_target_pose_for_joint123_angles(
            left_frozen_target, "left", candidate_waist
        )
        right_pose = self._tf_target_pose_for_joint123_angles(
            right_frozen_target, "right", candidate_waist
        )
        if equalize_target_z:
            left_pose, right_pose, _ = self._equalize_tf_dual_target_z(
                left_pose, right_pose, reference="average"
            )
        return left_pose, right_pose

    @staticmethod
    def _waist_workspace_pose_transform(pose: Pose):
        return (
            (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ),
            BoxSupportMixin._normalize_quaternion(
                (
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                )
            ),
        )

    def _waist_workspace_carry_final_endpoints(
        self,
        left_start: Pose,
        right_start: Pose,
        candidate_waist,
        *,
        box_layer: int,
        model_label: str | None,
        drag_mode: bool,
        right_arm_only: bool,
        delayed_left_join: bool,
        direct_carry_after_clamp: bool = False,
    ) -> tuple[list[Pose], list[Pose]]:
        """Predict only the final arm poses after the waist returns home.

        The optimizer deliberately checks two poses per evaluated arm: the
        initial MoveJ_P target and this final waist-carry target.  Step1,
        Drag1--Drag3, Step2, and carry intermediate poses are not IK sampled.
        """
        standard_enabled = self._boolean("box_post_movel_enabled")
        drag_enabled = drag_mode and self._boolean("drag_box_post_movel_enabled")
        if not standard_enabled and not drag_enabled:
            raise MissionError(
                "waist-home final-Pose IK requires the post-movel sequence"
            )
        carry_prefix = BoxSupportMixin._tf_body_home_carry_parameter_prefix(
            tf_mode=True,
            drag_mode=drag_mode,
        )
        if not carry_prefix or not self._boolean(f"{carry_prefix}_enabled"):
            raise MissionError(
                "waist-home final-Pose IK requires TF body-home carry to be enabled"
            )
        targets = self._post_movel_targets_with_labels(
            left_start,
            right_start,
            include_drag_steps=drag_enabled,
            defer_left_step1=delayed_left_join,
            model_label=model_label,
            box_layer=box_layer,
            tf_mode=True,
        )
        carry_start_label = "step1" if direct_carry_after_clamp else "step2"
        try:
            _label, left_step2, right_step2 = next(
                target for target in targets if target[0] == carry_start_label
            )
        except StopIteration as exc:
            raise MissionError(
                f"waist-home final-Pose IK could not find the {carry_start_label} trigger"
            ) from exc

        # Step2 is dual-arm for both direct grasp and delayed-left DragBox.
        # Match the runtime's shared numeric arm-base Z constraint.
        left_step2, right_step2, _ = self._equalize_tf_dual_target_z(
            left_step2,
            right_step2,
            reference="average",
        )
        candidate_waist = tuple(float(value) for value in candidate_waist)
        candidate_arm_base = {
            arm: self._joint123_arm_base_transform(arm, candidate_waist)
            for arm in ("left", "right")
        }
        initial_pose = {"left": left_start, "right": right_start}
        step2_pose = {"left": left_step2, "right": right_step2}
        initial_world = {
            arm: BoxSupportMixin._compose_transform(
                candidate_arm_base[arm],
                self._waist_workspace_pose_transform(initial_pose[arm]),
            )
            for arm in ("left", "right")
        }
        step2_world = {
            arm: BoxSupportMixin._compose_transform(
                candidate_arm_base[arm],
                self._waist_workspace_pose_transform(step2_pose[arm]),
            )
            for arm in ("left", "right")
        }

        frozen_box_pose = getattr(self, "_last_grasp_box_tf_box_pose", None)
        if frozen_box_pose is None:
            raise MissionError(
                "waist-home final-Pose IK has no frozen FoundationPose box"
            )
        frozen_box = self._pose_stamped_to_transform(frozen_box_pose)
        initial_relation = {
            arm: BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(frozen_box),
                initial_world[arm],
            )
            for arm in ("left", "right")
        }
        inferred_box = {
            arm: BoxSupportMixin._compose_transform(
                step2_world[arm],
                BoxSupportMixin._inverse_transform(initial_relation[arm]),
            )
            for arm in ("left", "right")
        }
        # DragBox uses the right arm as the physical authority through Drag3;
        # direct GraspBox uses the same bilateral mean as the runtime carry.
        current_box = (
            inferred_box["right"]
            if drag_mode and right_arm_only
            else BoxSupportMixin._mean_rigid_transforms(
                inferred_box["left"], inferred_box["right"]
            )
        )
        current_relation = {
            arm: BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(current_box),
                step2_world[arm],
            )
            for arm in ("left", "right")
        }

        current_carrier = self._joint123_chest_transform(candidate_waist)
        units_per_degree = self._float_array("box_body_command_units_per_degree")
        home_units = self._float_array(f"{carry_prefix}_joint_units")
        if len(units_per_degree) != 4 or len(home_units) != 4:
            raise MissionError(
                "waist-home final-Pose IK requires four body unit scales and targets"
            )
        home_waist = tuple(
            math.radians(float(home_units[index]) / float(units_per_degree[index]))
            for index in range(3)
        )
        home_carrier = self._joint123_chest_transform(home_waist)
        carrier_to_box_position = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(current_carrier), current_box
        )[0]
        final_box_position = BoxSupportMixin._compose_transform(
            home_carrier,
            (carrier_to_box_position, (0.0, 0.0, 0.0, 1.0)),
        )[0]
        final_box = (final_box_position, current_box[1])
        final_world = {
            arm: BoxSupportMixin._compose_transform(
                final_box, current_relation[arm]
            )
            for arm in ("left", "right")
        }
        final_pose = {}
        for arm in ("left", "right"):
            home_arm_base = self._joint123_arm_base_transform(arm, home_waist)
            final_pose[arm] = self._endpoint_sync_transform_to_pose(
                BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(home_arm_base),
                    final_world[arm],
                )
            )
        return [final_pose["left"]], [final_pose["right"]]

    def _optimize_tf_waist_target(
        self,
        goal_handle,
        left_frozen_target: PoseStamped,
        right_frozen_target: PoseStamped,
        box_layer: int,
        model_label: str | None,
        *,
        drag_mode: bool,
        equalize_target_z: bool,
        right_arm_only: bool,
        delayed_left_join: bool,
        direct_carry_after_clamp: bool = False,
        abort_event=None,
    ):
        """Optimize waist using only initial and waist-home-final Pose IK."""
        if not self._boolean("waist_workspace_optimization_enabled"):
            return None, None, "waist_workspace_optimization=disabled"
        # Fixed profiles validate one pose through the same IK pipeline.
        # Keeping a workspace result also retains post-waist live-TF IK.
        fixed_waist = (
            drag_mode
            and str(model_label).strip().lower() == "bigbox"
            and box_layer in (1, 2)
            and self._boolean(f"drag_box_tf_fixed_waist_enabled_bigbox_layer{box_layer}")
        )
        adapter = getattr(self, "direct_sdk_adapter", None)
        if adapter is None:
            raise MissionError(
                "waist workspace optimization requires direct_motion_backend=python_sdk"
            )
        now = time.monotonic()
        max_age = self._float("box_pre_target_arm_movej_feedback_max_age_sec")
        with self.joint_state_lock:
            left_seed = list(self.latest_slave_arm_positions.get("left", []))
            right_seed = list(self.latest_slave_arm_positions.get("right", []))
            left_age = now - self.latest_slave_arm_state_times.get("left", 0.0)
            right_age = now - self.latest_slave_arm_state_times.get("right", 0.0)
        if (
            len(left_seed) < 7
            or len(right_seed) < 7
            or left_age > max_age
            or right_age > max_age
        ):
            raise MissionError(
                "waist workspace optimization requires fresh seven-joint feedback: "
                f"left_joints={len(left_seed)}, left_age={left_age:.3f}s; "
                f"right_joints={len(right_seed)}, right_age={right_age:.3f}s; "
                f"limit={max_age:.3f}s"
            )

        ik_seed_mode = self._string("waist_workspace_ik_seed_mode").strip().lower()
        if ik_seed_mode == "zero":
            left_seed = [0.0] * 7
            right_seed = [0.0] * 7
        else:
            configured_seed = self._post_waist_pre_movej_targets(
                box_layer,
                model_label,
                drag_mode=drag_mode,
            )
            if configured_seed is not None:
                left_seed = [math.radians(value) for value in configured_seed[0]]
                right_seed = [math.radians(value) for value in configured_seed[1]]
        right_initial_joint4_negative_required = (
            drag_mode
            and right_arm_only
            and ik_seed_mode != "zero"
            and self._boolean(
                "drag_box_tf_right_initial_ik_joint4_negative_required"
            )
        )
        right_initial_joint4_seed_deg = self._float(
            "drag_box_tf_right_initial_ik_joint4_seed_deg"
        )
        joint4_negative_required = self._boolean(
            "box_ik_joint4_negative_required"
        )
        if right_initial_joint4_negative_required:
            # RM75's single-solution IK follows its seed branch.  Bias the
            # DragBox right-arm MoveJ_P solve toward the negative-Joint4
            # branch, then reject any solver result that crosses zero.
            right_seed[3] = math.radians(right_initial_joint4_seed_deg)

        def initial_solution_validator(arm, solution):
            if joint4_negative_required and float(solution[3]) >= 0.0:
                return False
            if right_initial_joint4_negative_required and arm == "right":
                return float(solution[3]) < 0.0
            return True

        def endpoint_solution_validator(_arm, solution):
            return not joint4_negative_required or float(solution[3]) < 0.0

        nominal_waist = tuple(
            math.radians(value)
            for value in self._box_layer_joint123_approach_angles_deg(
                box_layer,
                model_label,
                tf_mode=True,
                drag_mode=drag_mode,
            )
        )

        def candidate_poses(candidate_waist):
            return self._waist_workspace_candidate_poses(
                left_frozen_target,
                right_frozen_target,
                candidate_waist,
                equalize_target_z=equalize_target_z,
            )

        def target_pose_provider(candidate_waist):
            left_pose, right_pose = candidate_poses(candidate_waist)
            return pose_to_sdk_target(left_pose), pose_to_sdk_target(right_pose)

        check_predicted_left_join = drag_mode and right_arm_only and delayed_left_join

        def validate_predicted_left_join(candidate_waist, solutions):
            self._check_canceled(goal_handle, "while checking predicted Drag3 left join")
            if abort_event is not None and abort_event.is_set():
                raise MissionError("parallel waist search canceled during left-join check")
            frozen_box = BoxSupportMixin._pose_stamped_to_transform(
                self._last_grasp_box_tf_box_pose
            )
            left_world = BoxSupportMixin._pose_stamped_to_transform(left_frozen_target)
            right_world = BoxSupportMixin._pose_stamped_to_transform(right_frozen_target)
            target = self._predict_drag_tf_left_join_target(
                frozen_box, left_world, right_world,
                self._joint123_arm_base_transform("left", candidate_waist),
                self._joint123_arm_base_transform("right", candidate_waist),
                box_layer=box_layer, model_label=model_label,
            )
            preference = self._drag_left_join_joint4_preference(box_layer, model_label)
            trials = self._drag_left_join_base_seed_trials(
                solutions["right"], preference=preference
            )
            lower = self._float_array("waist_workspace_left_arm_joint_min_deg")
            upper = self._float_array("waist_workspace_left_arm_joint_max_deg")
            minimum_margin = self._float("waist_workspace_minimum_margin_deg")
            best_solution = None
            best_margin = -math.inf
            target_values = pose_to_sdk_target(target)
            for _source, seed in trials:
                solution = adapter.solve_ik("left", target_values, seed)
                if solution is None or len(solution) != 7 or not all(math.isfinite(v) for v in solution):
                    continue
                if not self._drag_left_join_joint4_acceptable(
                    solution, preference,
                    negative_required=self._boolean("drag_box_left_join_joint4_negative_required"),
                ):
                    continue
                margin = min(min(v-lo, hi-v) for v,lo,hi in zip(solution,lower,upper))
                if margin < minimum_margin or margin <= best_margin:
                    continue
                position_error, rotation_error = adapter.ik_pose_residual(
                    "left", target_values, solution
                )
                if position_error > 0.001 or rotation_error > math.radians(0.5):
                    continue
                best_solution, best_margin = solution, margin
            return {"left": best_solution} if best_solution is not None else False

        def carry_final_endpoint_poses(candidate_waist):
            left_pose, right_pose = candidate_poses(candidate_waist)
            return self._waist_workspace_carry_final_endpoints(
                left_pose,
                right_pose,
                candidate_waist,
                box_layer=box_layer,
                model_label=model_label,
                drag_mode=drag_mode,
                right_arm_only=right_arm_only,
                delayed_left_join=delayed_left_join,
                direct_carry_after_clamp=direct_carry_after_clamp,
            )

        def carry_final_endpoint_provider(candidate_waist):
            left_endpoints, right_endpoints = carry_final_endpoint_poses(
                candidate_waist
            )
            return (
                [pose_to_sdk_target(pose) for pose in left_endpoints],
                [pose_to_sdk_target(pose) for pose in right_endpoints],
            )

        carry_final_pose_ik_enabled = self._boolean(
            "waist_workspace_movel_endpoint_ik_enabled"
        )
        optimization_arms = (
            ("right",) if drag_mode and right_arm_only else ("left", "right")
        )
        optimizer = WaistWorkspaceOptimizer(
            solve_ik=adapter.solve_ik,
            left_joint_limits_deg=(
                self._float_array("waist_workspace_left_arm_joint_min_deg"),
                self._float_array("waist_workspace_left_arm_joint_max_deg"),
            ),
            right_joint_limits_deg=(
                self._float_array("waist_workspace_right_arm_joint_min_deg"),
                self._float_array("waist_workspace_right_arm_joint_max_deg"),
            ),
            waist_limits_deg=(
                self._float_array("waist_workspace_joint_min_deg"),
                self._float_array("waist_workspace_joint_max_deg"),
            ),
            search_mode=("exhaustive" if fixed_waist else self._string("waist_workspace_search_mode")),
            candidate_step_deg=self._float("waist_workspace_candidate_step_deg"),
            candidate_delta_deg=(0.0 if fixed_waist else self._float("waist_workspace_candidate_delta_deg")),
            minimum_margin_deg=self._float("waist_workspace_minimum_margin_deg"),
            coarse_step_deg=self._float("waist_workspace_coarse_step_deg"),
            coarse_top_k=self._integer("waist_workspace_coarse_top_k"),
            candidate_diversity_deg=self._float(
                "waist_workspace_candidate_diversity_deg"
            ),
            refine_radius_deg=self._float("waist_workspace_refine_radius_deg"),
            refine_step_deg=self._float("waist_workspace_refine_step_deg"),
            sobol_sample_count=self._integer(
                "waist_workspace_sobol_sample_count"
            ),
            sobol_seed=self._integer("waist_workspace_sobol_seed"),
        )

        def report_optimization_progress(detail: str) -> None:
            if abort_event is not None and abort_event.is_set():
                raise MissionError(
                    "parallel waist workspace optimization was stopped because "
                    "the concurrent arm preparation did not complete"
                )
            self._check_canceled(
                goal_handle, "during waist workspace IK optimization"
            )
            self.get_logger().info(f"waist workspace optimization: {detail}")
            self._publish_box_grasp_feedback(
                goal_handle,
                "WAIST_WORKSPACE_OPTIMIZING",
                detail,
            )

        try:
            result = optimizer.optimize(
                nominal_waist_rad=nominal_waist,
                left_seed_rad=left_seed[:7],
                right_seed_rad=right_seed[:7],
                target_pose_provider=target_pose_provider,
                movel_endpoint_provider=(
                    carry_final_endpoint_provider
                    if carry_final_pose_ik_enabled
                    else None
                ),
                endpoint_ik_label="waist-home-final-Pose",
                endpoint_seed_mode=(
                    "initial_seed" if ik_seed_mode == "zero" else "previous_solution"
                ),
                active_arms=optimization_arms,
                initial_solution_validator=(
                    initial_solution_validator
                    if joint4_negative_required
                    or right_initial_joint4_negative_required
                    else None
                ),
                endpoint_solution_validator=(
                    endpoint_solution_validator
                    if joint4_negative_required
                    else None
                ),
                priority_joint_arm=(
                    "right" if drag_mode and right_arm_only and not check_predicted_left_join else None
                ),
                priority_joint_index=(
                    3 if drag_mode and right_arm_only and not check_predicted_left_join else None
                ),
                priority_joint_upper_bound_deg=(
                    0.0
                    if drag_mode and right_arm_only and joint4_negative_required and not check_predicted_left_join
                    else None
                ),
                auxiliary_candidate_validator=(validate_predicted_left_join if check_predicted_left_join else None),
                progress_callback=report_optimization_progress,
            )
        except (RealManSdkError, RuntimeError, ValueError) as exc:
            raise MissionError(f"waist workspace optimization failed: {exc}") from exc
        left_result_detail = (
            f"left_ik_deg=[{','.join(f'{value:.3f}' for value in result.left_joint_deg)}]; "
            f"left_min_margin_deg={result.left_margin_deg:.3f}; "
            if "left" in optimization_arms
            else (
                "left_initial_ik=not_required; left_join_ik=checked_per_candidate; "
                f"left_join_ik_deg={[round(v, 3) for v in result.left_joint_deg]}; "
                f"left_join_min_margin_deg={result.left_margin_deg:.3f}; "
                "left_join_fk_tolerance=0.001m/0.5deg; "
            )
        )
        detail = (
            f"waist_workspace_optimization={'disabled_fixed_pose' if fixed_waist else 'enabled'}; "
            f"search_mode={result.search_mode}; "
            f"evaluated_arms={','.join(optimization_arms)}; "
            + (
                "search_objective=right_joint4_negative_branch_margin; "
                f"right_joint4_search_margin_deg={result.priority_joint_margin_deg:.3f}; "
                if result.priority_joint_margin_deg is not None
                and joint4_negative_required
                else (
                    "search_objective=right_joint4_limit_margin; "
                    f"right_joint4_search_margin_deg={result.priority_joint_margin_deg:.3f}; "
                    if result.priority_joint_margin_deg is not None
                    else ""
                )
                )
            + ("search_objective=bilateral_minimum_joint_margin; " if check_predicted_left_join else "")
            + f"ik_seed_mode={ik_seed_mode}; "
            "joint4_negative_required="
            f"{str(joint4_negative_required).lower()}; "
            "right_initial_joint4_negative_required="
            f"{str(right_initial_joint4_negative_required).lower()}; "
            f"right_initial_joint4_seed_deg={right_initial_joint4_seed_deg:.3f}; "
            "movej_p_target_ik=true; "
            f"drag_left_join_precheck={str(check_predicted_left_join).lower()}; "
            "drag_left_join_actual_ik=rechecked_after_drag3; "
            "validation_points=initial_movej_p,waist_home_final_pose; "
            f"waist_home_final_pose_ik={str(carry_final_pose_ik_enabled).lower()}; "
            "movel_step_endpoint_ik=false; "
            "movel_intermediate_ik=false; "
            f"candidates={result.candidate_count}; valid={result.valid_count}; "
            f"initial_candidates={result.initial_candidate_count}; "
            f"refined_candidates={result.refined_candidate_count}; "
            f"selected_waist_deg=[{','.join(f'{math.degrees(value):.3f}' for value in result.waist_angles_rad)}]; "
            + left_result_detail
            + f"right_ik_deg=[{','.join(f'{value:.3f}' for value in result.right_joint_deg)}]; "
            f"right_min_margin_deg={result.right_margin_deg:.3f}; "
            f"score_deg={result.score_deg:.3f}"
        )
        self._publish_box_grasp_feedback(
            goal_handle, "WAIST_WORKSPACE_OPTIMIZED", detail
        )
        return result.waist_angles_rad, result, detail

    def _start_tf_waist_optimization_parallel(
        self,
        goal_handle,
        box_layer: int,
        model_label: str | None,
        *,
        drag_mode: bool,
        equalize_target_z: bool,
        right_arm_only: bool,
        delayed_left_join: bool,
        direct_carry_after_clamp: bool = False,
    ):
        """Start frozen-target waist IK while the arms move to preparation poses."""
        frozen_box_pose = getattr(self, "_last_grasp_box_tf_box_pose", None)
        if frozen_box_pose is None:
            raise MissionError(
                "TF waist optimization cannot start before FoundationPose is frozen"
            )
        left_target, right_target = self._make_tf_link8_target_poses(
            frozen_box_pose,
            box_layer,
            model_label,
            drag_mode=drag_mode,
        )
        result_future = Future()
        abort_event = threading.Event()
        started_event = threading.Event()

        def run_optimization() -> None:
            started_event.set()
            try:
                result_future.set_result(
                    self._optimize_tf_waist_target(
                        goal_handle,
                        left_target,
                        right_target,
                        box_layer,
                        model_label,
                        drag_mode=drag_mode,
                        equalize_target_z=equalize_target_z,
                        right_arm_only=right_arm_only,
                        delayed_left_join=delayed_left_join,
                        direct_carry_after_clamp=direct_carry_after_clamp,
                        abort_event=abort_event,
                    )
                )
            except BaseException as exc:  # noqa: BLE001
                result_future.set_exception(exc)

        worker = threading.Thread(
            target=run_optimization,
            name=(
                "drag-box-waist-optimizer"
                if drag_mode
                else "grasp-box-waist-optimizer"
            ),
            daemon=True,
        )
        worker.start()
        started_event.wait(timeout=1.0)
        detail = (
            "waist workspace optimization started in parallel with "
            + (
                "the DragBox left-arm avoidance motion"
                if drag_mode
                else "the GraspBox layer-specific dual-arm preparation"
            )
        )
        self.get_logger().info(detail)
        self._publish_box_grasp_feedback(
            goal_handle,
            "WAIST_WORKSPACE_PARALLEL_STARTED",
            detail,
        )
        return {
            "left_target": left_target,
            "right_target": right_target,
            "future": result_future,
            "abort_event": abort_event,
            "worker": worker,
        }

    def _wait_tf_waist_optimization_parallel(self, goal_handle, context):
        """Join a previously started waist optimization without blocking ROS cancel."""
        future = context["future"]
        while not future.done():
            self._check_canceled(
                goal_handle, "while waiting for parallel waist workspace optimization"
            )
            time.sleep(0.02)
        context["worker"].join(timeout=0.1)
        return future.result()

    @staticmethod
    def _abort_tf_waist_optimization_parallel(context) -> None:
        if context is not None:
            context["abort_event"].set()
            context["worker"].join(timeout=2.0)
