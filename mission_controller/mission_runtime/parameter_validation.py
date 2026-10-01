"""Mission parameter validation and typed accessors."""

import math


class ParameterValidationMixin:
    """Validate calibrated inputs before Action servers accept goals."""

    def _validate_parameters(self) -> None:
        for prefix in ("grasp_box_tf", "drag_box_tf"):
            for arm in ("left", "right"):
                name = f"{prefix}_{arm}_contact_forward_delta_scale"
                value = self._float(name)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and >= 0")
        for name in (
            "execute_adaptive_box_grasp_action_name",
            "execute_box_grasp_action_name",
            "grasp_box_tf_action_name",
            "execute_drag_box_grasp_action_name",
            "execute_drag_box_grasp_tf_action_name",
            "execute_box_place_action_name",
            "adaptive_freeze_frame",
            "box_object_pose_action_name",
            "box_object_pose_camera_side",
            "grasp_box_tf_detection_arm",
            "drag_box_tf_detection_arm",
            "drag_box_tf_detection_arm_smallbox",
            "box_object_pose_topic",
            "box_object_pose_camera_topic",
            "box_object_pose_raw_topic",
            "box_object_pose_model_label",
            "pickup_task_action_name",
            "direct_motion_backend",
            "direct_movel_service_name",
            "direct_sdk_root",
            "direct_sdk_left_ip",
            "direct_sdk_right_ip",
            "direct_movel_target_mode",
            "direct_movel_box_relative_model_label",
            "direct_movel_motion_mode",
            "box_grasp_execution_mode",
            "box_joint1_command_service_name",
            "box_joint1_feedback_topic",
            "box_joint1_name",
            "box_joint2_name",
            "box_joint3_name",
            "box_joint4_name",
            "box_type",
            "arm_joints_service_name",
            "go_ready_action_name",
            "torso_topic",
            "left_gripper_topic",
            "right_gripper_topic",
            "joint_state_topic",
            "torso_feedback_topic",
            "left_gripper_feedback_topic",
            "right_gripper_feedback_topic",
            "arm_execution_frame",
            "left_ee_frame",
            "right_ee_frame",
            "left_gripper_frame",
            "right_gripper_frame",
            "camera_detection_arm",
            "grasp_box_tf_force_clamp_mode",
            "drag_box_tf_force_clamp_mode",
            "grasp_box_tf_post_movel_sdk_motion_mode",
            "drag_box_tf_post_movel_sdk_motion_mode",
        ):
            if not self._string(name):
                raise ValueError(f"parameter '{name}' must not be empty")

        if self._string("adaptive_freeze_frame").lstrip("/") != "base_link":
            raise ValueError(
                "adaptive_freeze_frame must be base_link while the chassis is fixed"
            )

        for name in ("left_arm_joint_names", "right_arm_joint_names"):
            joint_names = self._string_array(name)
            if len(joint_names) != 7 or len(set(joint_names)) != 7:
                raise ValueError(
                    f"parameter '{name}' must contain 7 unique joint names"
                )

        motion_mode = self._string("direct_movel_motion_mode").lower()
        if motion_mode not in ("movel", "movej_p"):
            raise ValueError(
                "parameter 'direct_movel_motion_mode' must be 'movel' or 'movej_p'"
            )
        step4_motion_mode = self._string("box_post_movel_step4_motion_mode").lower()
        if step4_motion_mode not in ("movel", "movej_p", "movej"):
            raise ValueError(
                "parameter 'box_post_movel_step4_motion_mode' must be "
                "'movel', 'movej_p', or 'movej'"
            )

        for name in (
            "grasp_box_tf_post_movel_sdk_motion_mode",
            "drag_box_tf_post_movel_sdk_motion_mode",
        ):
            post_movel_mode = self._string(name).strip().lower()
            if post_movel_mode not in ("movel", "movel_offset"):
                raise ValueError(
                    f"parameter '{name}' must be 'movel' or 'movel_offset'"
                )
        for name in (
            "box_tf_step2_contact_height_tolerance_m",
            "box_tf_step2_grasp_span_tolerance_m",
        ):
            value = self._float(name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"parameter '{name}' must be finite and positive")

        left_join_mode = self._string("drag_box_left_join_mode").strip().lower()
        if left_join_mode not in ("immediate", "after_drag3"):
            raise ValueError(
                "parameter 'drag_box_left_join_mode' must be "
                "'immediate' or 'after_drag3'"
            )
        left_join_motion_mode = (
            self._string("drag_box_left_join_motion_mode").strip().lower()
        )
        if left_join_motion_mode not in ("movel", "movej_p", "staged_ik_movej"):
            raise ValueError(
                "parameter 'drag_box_left_join_motion_mode' must be "
                "'movel', 'movej_p', or 'staged_ik_movej'"
            )
        if self._integer("drag_box_left_join_ik_max_attempts") <= 0:
            raise ValueError(
                "parameter 'drag_box_left_join_ik_max_attempts' must be positive"
            )
        if self._integer("drag_box_left_join_ik_random_seed_attempts") < 0:
            raise ValueError(
                "parameter 'drag_box_left_join_ik_random_seed_attempts' "
                "must be non-negative"
            )

        execution_mode = self._string("box_grasp_execution_mode").lower()
        if execution_mode not in (
            "arms_only",
            "joint1_then_arms",
            "joint1_then_arms_keep_position",
            "joint123_then_arms",
        ):
            raise ValueError(
                "parameter 'box_grasp_execution_mode' must be 'arms_only' "
                "or 'joint1_then_arms' or "
                "'joint1_then_arms_keep_position' or 'joint123_then_arms'"
            )

        for name, expected_size in (
            ("waist_workspace_joint_min_deg", 3),
            ("waist_workspace_joint_max_deg", 3),
            ("waist_workspace_left_arm_joint_min_deg", 7),
            ("waist_workspace_left_arm_joint_max_deg", 7),
            ("waist_workspace_right_arm_joint_min_deg", 7),
            ("waist_workspace_right_arm_joint_max_deg", 7),
        ):
            values = self._float_array(name)
            if len(values) != expected_size or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError(
                    f"parameter '{name}' must contain {expected_size} finite values"
                )
        for prefix in (
            "waist_workspace_joint",
            "waist_workspace_left_arm_joint",
            "waist_workspace_right_arm_joint",
        ):
            lower = self._float_array(f"{prefix}_min_deg")
            upper = self._float_array(f"{prefix}_max_deg")
            if any(lo >= hi for lo, hi in zip(lower, upper)):
                raise ValueError(
                    f"parameters '{prefix}_min_deg/max_deg' contain invalid limits"
                )
        for action_prefix in ("grasp_box_tf", "drag_box_tf"):
            prefix = f"{action_prefix}_post_waist_pre_movej"
            for model in ("bigbox", "smallbox"):
                for layer in range(1, 5):
                    for arm in ("left", "right"):
                        parameter_name = (
                            f"{prefix}_{arm}_joint_units_{model}_layer{layer}"
                        )
                        values = self._float_array(parameter_name)
                        if len(values) != 7 or not all(
                            math.isfinite(value) for value in values
                        ):
                            raise ValueError(
                                f"parameter '{parameter_name}' must contain "
                                "seven finite joint values"
                            )
        candidate_step = self._float("waist_workspace_candidate_step_deg")
        candidate_delta = self._float("waist_workspace_candidate_delta_deg")
        minimum_margin = self._float("waist_workspace_minimum_margin_deg")
        coarse_step = self._float("waist_workspace_coarse_step_deg")
        coarse_top_k = self._integer("waist_workspace_coarse_top_k")
        candidate_diversity = self._float(
            "waist_workspace_candidate_diversity_deg"
        )
        refine_radius = self._float("waist_workspace_refine_radius_deg")
        refine_step = self._float("waist_workspace_refine_step_deg")
        sobol_sample_count = self._integer(
            "waist_workspace_sobol_sample_count"
        )
        sobol_seed = self._integer("waist_workspace_sobol_seed")
        ik_seed_mode = self._string("waist_workspace_ik_seed_mode").strip().lower()
        search_mode = self._string("waist_workspace_search_mode").strip().lower()
        if search_mode not in ("exhaustive", "coarse_to_fine", "sobol_refine"):
            raise ValueError(
                "parameter 'waist_workspace_search_mode' must be "
                "'exhaustive', 'coarse_to_fine', or 'sobol_refine'"
            )
        if not math.isfinite(candidate_step) or candidate_step <= 0.0:
            raise ValueError(
                "parameter 'waist_workspace_candidate_step_deg' must be positive"
            )
        if not math.isfinite(candidate_delta) or candidate_delta < 0.0:
            raise ValueError(
                "parameter 'waist_workspace_candidate_delta_deg' must be non-negative"
            )
        if not math.isfinite(minimum_margin) or minimum_margin < 0.0:
            raise ValueError(
                "parameter 'waist_workspace_minimum_margin_deg' must be non-negative"
            )
        if not math.isfinite(coarse_step) or coarse_step <= 0.0:
            raise ValueError(
                "parameter 'waist_workspace_coarse_step_deg' must be positive"
            )
        if coarse_top_k <= 0:
            raise ValueError(
                "parameter 'waist_workspace_coarse_top_k' must be positive"
            )
        if not math.isfinite(candidate_diversity) or candidate_diversity < 0.0:
            raise ValueError(
                "parameter 'waist_workspace_candidate_diversity_deg' "
                "must be non-negative"
            )
        if not math.isfinite(refine_radius) or refine_radius < 0.0:
            raise ValueError(
                "parameter 'waist_workspace_refine_radius_deg' must be non-negative"
            )
        if not math.isfinite(refine_step) or refine_step <= 0.0:
            raise ValueError(
                "parameter 'waist_workspace_refine_step_deg' must be positive"
            )
        if sobol_sample_count <= 0:
            raise ValueError(
                "parameter 'waist_workspace_sobol_sample_count' must be positive"
            )
        if sobol_seed < 0:
            raise ValueError(
                "parameter 'waist_workspace_sobol_seed' must be non-negative"
            )
        if ik_seed_mode not in ("preparation", "zero"):
            raise ValueError(
                "parameter 'waist_workspace_ik_seed_mode' must be "
                "'preparation' or 'zero'"
            )
        drag_right_joint4_seed = self._float(
            "drag_box_tf_right_initial_ik_joint4_seed_deg"
        )
        if not math.isfinite(drag_right_joint4_seed):
            raise ValueError(
                "parameter 'drag_box_tf_right_initial_ik_joint4_seed_deg' "
                "must be finite"
            )
        if (
            self._boolean(
                "drag_box_tf_right_initial_ik_joint4_negative_required"
            )
            and drag_right_joint4_seed >= 0.0
        ):
            raise ValueError(
                "parameter 'drag_box_tf_right_initial_ik_joint4_seed_deg' "
                "must be negative when the negative-Joint4 requirement is enabled"
            )

        target_mode = self._string("direct_movel_target_mode").lower()
        if target_mode not in (
            "camera_offset",
            "camera_offset_box_orientation",
        ):
            raise ValueError(
                "parameter 'direct_movel_target_mode' must be "
                "'camera_offset' or 'camera_offset_box_orientation'"
            )
        if self._string("camera_detection_arm").strip().lower() not in (
            "left",
            "right",
        ):
            raise ValueError("camera_detection_arm must be 'left' or 'right'")
        for name in (
            "grasp_box_tf_detection_arm",
            "drag_box_tf_detection_arm",
            "drag_box_tf_detection_arm_smallbox",
        ):
            if self._string(name).strip().lower() not in ("left", "right"):
                raise ValueError(f"{name} must be 'left' or 'right'")
        if target_mode == "camera_offset_box_orientation" and not self._boolean(
            "camera_measured_extrinsics_enabled"
        ):
            raise ValueError(
                f"direct_movel_target_mode={target_mode} requires "
                "camera_measured_extrinsics_enabled=true"
            )
        if (
            target_mode == "camera_offset_box_orientation"
            and self._string("box_object_pose_model_label").strip().lower()
            != self._string("direct_movel_box_relative_model_label").strip().lower()
        ):
            raise ValueError(
                "box-orientation calibration model does not match "
                "box_object_pose_model_label"
            )

        motion_backend = self._string("direct_motion_backend").lower()
        if motion_backend not in ("ros_service", "python_sdk"):
            raise ValueError(
                "parameter 'direct_motion_backend' must be 'ros_service' "
                "or 'python_sdk'"
            )

        if self._boolean("camera_mount_tf_enabled"):
            for name in ("camera_mount_parent_frame", "camera_mount_child_frame"):
                if not self._string(name):
                    raise ValueError(f"parameter '{name}' must not be empty")

        for name, expected_length in (
            ("torso_reset_positions", 4),
            ("torso_velocities", 4),
            ("box_grasp_intermediate_left_joint_positions", 7),
            ("box_grasp_intermediate_right_joint_positions", 7),
            ("box_grasp_left_observation_joint_positions", 7),
            ("box_grasp_right_observation_joint_positions", 7),
            ("box_pickup_clearance_left_joint_positions", 7),
            ("box_pickup_clearance_right_joint_positions", 7),
            ("box_grasp_torso_prepare_positions", 4),
            ("box_grasp_torso_lift_positions", 4),
            ("box_place_torso_positions", 4),
            ("box_place_torso_straighten_intermediate_positions", 4),
            ("camera_mount_xyz", 3),
            ("camera_mount_rpy", 3),
            ("camera_mount_correction_rpy", 3),
            ("camera_left_base_xyz", 3),
            ("camera_right_base_xyz", 3),
            ("camera_left_base_rpy", 3),
            ("camera_right_base_rpy", 3),
            ("camera_left_link8_to_rgb_camera_xyz", 3),
            ("camera_right_link8_to_rgb_camera_xyz", 3),
            ("camera_left_link8_to_rgb_camera_quaternion_xyzw", 4),
            ("camera_right_link8_to_rgb_camera_quaternion_xyzw", 4),
            ("camera_right_base_to_left_base_xyz", 3),
            ("camera_right_base_to_left_base_quaternion_xyzw", 4),
            ("box_foundation_to_pickup_rpy", 3),
            ("direct_movel_left_offset_xyz", 3),
            ("direct_movel_right_offset_xyz", 3),
            ("direct_movel_left_offset_xyz_bigbox_layer1", 3),
            ("direct_movel_right_offset_xyz_bigbox_layer1", 3),
            ("direct_movel_left_offset_xyz_bigbox_layer2", 3),
            ("direct_movel_right_offset_xyz_bigbox_layer2", 3),
            ("direct_movel_left_offset_xyz_bigbox_layer3", 3),
            ("direct_movel_right_offset_xyz_bigbox_layer3", 3),
            ("direct_movel_left_offset_xyz_bigbox_layer4", 3),
            ("direct_movel_right_offset_xyz_bigbox_layer4", 3),
            ("direct_movel_left_offset_xyz_smallbox_layer1", 3),
            ("direct_movel_right_offset_xyz_smallbox_layer1", 3),
            ("direct_movel_left_offset_xyz_smallbox_layer2", 3),
            ("direct_movel_right_offset_xyz_smallbox_layer2", 3),
            ("direct_movel_left_offset_xyz_smallbox_layer3", 3),
            ("direct_movel_right_offset_xyz_smallbox_layer3", 3),
            ("direct_movel_left_offset_xyz_smallbox_layer4", 3),
            ("direct_movel_right_offset_xyz_smallbox_layer4", 3),
            ("box_post_movel_left_step1_xyz", 3),
            ("box_post_movel_right_step1_xyz", 3),
            ("box_post_movel_left_step1_xyz_smallbox", 3),
            ("box_post_movel_right_step1_xyz_smallbox", 3),
            ("box_post_movel_left_step2_xyz", 3),
            ("box_post_movel_right_step2_xyz", 3),
            ("box_post_movel_left_step3_xyz", 3),
            ("box_post_movel_right_step3_xyz", 3),
            ("box_post_movel_left_step4_xyz", 3),
            ("box_post_movel_right_step4_xyz", 3),
            ("box_post_movel_left_step5_xyz", 3),
            ("box_post_movel_right_step5_xyz", 3),
            ("drag_box_post_movel_step_drag1_left_xyz", 3),
            ("drag_box_post_movel_step_drag1_right_xyz", 3),
            ("drag_box_post_movel_step_drag2_left_xyz", 3),
            ("drag_box_post_movel_step_drag2_right_xyz", 3),
            ("drag_box_post_movel_step_drag3_left_xyz", 3),
            ("drag_box_post_movel_step_drag3_right_xyz", 3),
            ("box_post_movel_step4_movej_left_joint_units", 7),
            ("box_post_movel_step4_movej_right_joint_units", 7),
            ("direct_movel_left_box_to_link8_orientation", 4),
            ("direct_movel_right_box_to_link8_orientation", 4),
            ("joint123_layer1_left_target_correction_pose_box", 7),
            ("joint123_layer1_right_target_correction_pose_box", 7),
            ("joint123_layer2_left_target_correction_pose_box", 7),
            ("joint123_layer2_right_target_correction_pose_box", 7),
            ("joint123_layer3_left_target_correction_pose_box", 7),
            ("joint123_layer3_right_target_correction_pose_box", 7),
            ("joint123_layer4_left_target_correction_pose_box", 7),
            ("joint123_layer4_right_target_correction_pose_box", 7),
            ("direct_movel_left_fixed_link8_orientation", 4),
            ("direct_movel_right_fixed_link8_orientation", 4),
            ("left_fixture_center_in_link8_xyz", 3),
            ("right_fixture_center_in_link8_xyz", 3),
            ("box_joint1_axis_xyz", 3),
            ("box_joint2_axis_xyz", 3),
            ("box_joint3_axis_xyz", 3),
            ("box_waist1_origin_xyz", 3),
            ("box_waist1_origin_rpy", 3),
            ("box_waist2_origin_xyz", 3),
            ("box_waist2_origin_rpy", 3),
            ("box_waist3_origin_xyz", 3),
            ("box_waist3_origin_rpy", 3),
            ("box_waist3_to_chest_xyz", 3),
            ("box_waist3_to_chest_rpy", 3),
            ("box_chest_to_left_arm_base_xyz", 3),
            ("box_chest_to_left_arm_base_rpy", 3),
            ("box_chest_to_right_arm_base_xyz", 3),
            ("box_chest_to_right_arm_base_rpy", 3),
            ("box_body_command_units_per_degree", 4),
            ("box_layer_joint1_approach_angles_deg", 4),
            ("box_layer_joint2_approach_angles_deg", 4),
            ("box_layer_joint3_approach_angles_deg", 4),
            ("box_layer_joint1_approach_angles_deg_bigbox", 4),
            ("box_layer_joint2_approach_angles_deg_bigbox", 4),
            ("box_layer_joint3_approach_angles_deg_bigbox", 4),
            ("box_layer_joint1_approach_angles_deg_smallbox", 4),
            ("box_layer_joint2_approach_angles_deg_smallbox", 4),
            ("box_layer_joint3_approach_angles_deg_smallbox", 4),
            ("adaptive_grasp_span_axis_object", 3),
            ("adaptive_grasp_height_axis_object", 3),
            ("adaptive_grasp_correction_rpy", 3),
            ("adaptive_left_grasp_extra_rpy", 3),
            ("adaptive_right_grasp_extra_rpy", 3),
            ("box_post_arm_movej_left_joint_units", 7),
            ("box_post_arm_movej_right_joint_units", 7),
            ("box_pre_detection_right_movej_joint_units", 7),
            ("box_pre_detection_left_movej_joint_units", 7),
            ("box_layer_pre_detection_right_movej_joint_units", 28),
            ("box_layer_pre_detection_right_movej_joint_units_bigbox", 28),
            ("box_layer_pre_detection_right_movej_joint_units_smallbox", 28),
            ("box_layer_pre_detection_left_movej_joint_units", 28),
            ("box_layer_pre_detection_left_movej_joint_units_bigbox", 28),
            ("box_layer_pre_detection_left_movej_joint_units_smallbox", 28),
            ("box_pre_target_arm_movej_left_joint_units", 7),
            ("box_pre_target_arm_movej_right_joint_units", 7),
            ("drag_box_left_join_ik_seed_joint_deg", 7),
            ("drag_box_tf_calibration_left_join_joint_target_deg", 7),
            ("box_body_home_joint_units", 4),
            ("box_step2_waist_endpoint_sync_home_joint_units", 4),
            ("grasp_box_tf_body_home_carry_joint_units", 4),
            ("drag_box_tf_body_home_carry_joint_units", 4),
            ("place_box_test_body_joint_units", 4),
            ("place_box_test_start_body_joint_units", 4),
            ("place_box_test_post_release_body_home_joint_units", 4),
            ("place_box_test_post_release_arm_home_left_joint_units", 7),
            ("place_box_test_post_release_arm_home_right_joint_units", 7),
            ("place_box_test_left_target_pose_arm_base", 7),
            ("place_box_test_right_target_pose_arm_base", 7),
            ("place_box_test_left_target_pose_arm_base_smallbox", 7),
            ("place_box_test_right_target_pose_arm_base_smallbox", 7),
            ("place_box_test_left_target_pose_arm_base_bigbox", 7),
            ("place_box_test_right_target_pose_arm_base_bigbox", 7),
        ):
            values = self._float_array(name)
            if len(values) != expected_length:
                raise ValueError(
                    f"parameter '{name}' must contain {expected_length} values"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"parameter '{name}' contains NaN or Inf")

        for name in (
            "joint123_layer1_left_target_correction_pose_box",
            "joint123_layer1_right_target_correction_pose_box",
            "joint123_layer2_left_target_correction_pose_box",
            "joint123_layer2_right_target_correction_pose_box",
            "joint123_layer3_left_target_correction_pose_box",
            "joint123_layer3_right_target_correction_pose_box",
            "joint123_layer4_left_target_correction_pose_box",
            "joint123_layer4_right_target_correction_pose_box",
            "place_box_test_left_target_pose_arm_base",
            "place_box_test_right_target_pose_arm_base",
            "place_box_test_left_target_pose_arm_base_smallbox",
            "place_box_test_right_target_pose_arm_base_smallbox",
            "place_box_test_left_target_pose_arm_base_bigbox",
            "place_box_test_right_target_pose_arm_base_bigbox",
        ):
            values = self._float_array(name)
            quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
            if quaternion_norm <= 1e-12:
                raise ValueError(f"parameter '{name}' contains a zero quaternion")

        for name in ("adaptive_grasp_height_offset_m",):
            if not math.isfinite(self._float(name)):
                raise ValueError(f"parameter '{name}' must be finite")

        for name in (
            "box_joint1_detection_angle_deg",
            "box_joint1_approach_angle_deg",
            "box_joint1_feedback_to_geometric_sign",
            "box_joint2_detection_angle_deg",
            "box_joint2_approach_angle_deg",
            "box_joint3_detection_angle_deg",
            "box_joint3_approach_angle_deg",
            "box_joint2_feedback_to_urdf_axis_sign",
            "box_joint3_feedback_to_urdf_axis_sign",
        ):
            if not math.isfinite(self._float(name)):
                raise ValueError(f"parameter '{name}' must be finite")
        for name in (
            "box_layer_joint1_approach_angles_deg",
            "box_layer_joint2_approach_angles_deg",
            "box_layer_joint3_approach_angles_deg",
            "box_layer_joint1_approach_angles_deg_bigbox",
            "box_layer_joint2_approach_angles_deg_bigbox",
            "box_layer_joint3_approach_angles_deg_bigbox",
            "box_layer_joint1_approach_angles_deg_smallbox",
            "box_layer_joint2_approach_angles_deg_smallbox",
            "box_layer_joint3_approach_angles_deg_smallbox",
        ):
            layer_angles = self._float_array(name)
            if not all(math.isfinite(value) for value in layer_angles):
                raise ValueError(f"parameter '{name}' contains NaN or Inf")
        layer_configured = self._boolean_array("box_layer_joint123_configured")
        if len(layer_configured) != 4:
            raise ValueError(
                "parameter 'box_layer_joint123_configured' must contain four values"
            )
        detection_configured = self._boolean_array(
            "box_layer_pre_detection_right_movej_configured"
        )
        if len(detection_configured) != 4:
            raise ValueError(
                "parameter 'box_layer_pre_detection_right_movej_configured' "
                "must contain four values"
            )
        if self._float("box_joint1_feedback_to_geometric_sign") not in (
            -1.0,
            1.0,
        ):
            raise ValueError(
                "box_joint1_feedback_to_geometric_sign must be -1.0 or 1.0"
            )
        for name in (
            "box_joint2_feedback_to_urdf_axis_sign",
            "box_joint3_feedback_to_urdf_axis_sign",
        ):
            if self._float(name) not in (-1.0, 1.0):
                raise ValueError(f"{name} must be -1.0 or 1.0")
        for name in (
            "camera_left_link8_to_rgb_camera_quaternion_xyzw",
            "camera_right_link8_to_rgb_camera_quaternion_xyzw",
            "camera_right_base_to_left_base_quaternion_xyzw",
        ):
            if (
                math.sqrt(sum(value * value for value in self._float_array(name)))
                <= 1e-12
            ):
                raise ValueError(f"parameter '{name}' has zero norm")
        if self._float("camera_eepose_max_age_sec") <= 0.0:
            raise ValueError("camera_eepose_max_age_sec must be positive")
        if any(
            value <= 0.0
            for value in self._float_array("box_body_command_units_per_degree")
        ):
            raise ValueError(
                "box_body_command_units_per_degree values must be positive"
            )

        for name in (
            "gripper_open_position",
            "gripper_closed_position",
        ):
            value = self._float(name)
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"parameter '{name}' must be in [0, 100]")

        if math.isclose(
            self._float("gripper_open_position"),
            self._float("gripper_closed_position"),
        ):
            raise ValueError(
                "gripper_open_position and gripper_closed_position must differ"
            )
        close_ratio = self._float("box_empty_close_ratio_threshold")
        if not 0.0 <= close_ratio <= 1.0:
            raise ValueError("box_empty_close_ratio_threshold must be in [0, 1]")

        positive_parameters = (
            "adaptive_tf_cache_time_sec",
            "adaptive_detection_tf_timeout_sec",
            "adaptive_runtime_tf_timeout_sec",
            "adaptive_grasp_velocity_percent",
            "adaptive_grasp_timeout_sec",
            "adaptive_lift_distance_m",
            "adaptive_lift_velocity_percent",
            "adaptive_lift_timeout_sec",
            "dependency_wait_timeout_sec",
            "arm_joints_result_timeout_sec",
            "go_ready_result_timeout_sec",
            "command_subscriber_wait_timeout_sec",
            "camera_tf_timeout_sec",
            "pickup_task_result_timeout_sec",
            "direct_movel_velocity_percent",
            "box_post_movel_velocity_percent",
            "drag_box_left_join_velocity_percent",
            "drag_box_left_join_timeout_sec",
            "direct_sdk_motion_timeout_sec",
            "grasp_box_tf_post_waist_pre_movej_command_units_per_degree",
            "grasp_box_tf_post_waist_pre_movej_velocity_percent",
            "grasp_box_tf_post_waist_pre_movej_timeout_sec",
            "drag_box_tf_post_waist_pre_movej_command_units_per_degree",
            "drag_box_tf_post_waist_pre_movej_velocity_percent",
            "drag_box_tf_post_waist_pre_movej_timeout_sec",
            "box_width",
            "box_height",
            "arm_joint_target_tolerance",
            "arm_joint_target_wait_timeout_sec",
            "box_observation_feedback_max_age_sec",
            "box_observation_torso_tolerance",
            "torso_target_tolerance",
            "torso_target_wait_timeout_sec",
            "box_gripper_feedback_timeout_sec",
            "box_gripper_feedback_max_age_sec",
            "box_joint1_command_units_per_degree",
            "box_joint1_position_tolerance_rad",
            "box_joint1_velocity_tolerance_rad_sec",
            "box_joint1_feedback_max_age_sec",
            "box_joint1_wait_timeout_sec",
            "box_post_arm_movej_command_units_per_degree",
            "box_post_arm_position_tolerance_rad",
            "box_post_arm_velocity_tolerance_rad_sec",
            "box_post_arm_feedback_max_age_sec",
            "box_post_arm_movej_timeout_sec",
            "box_post_movel_step4_movej_timeout_sec",
            "box_post_movel_step4_movej_position_tolerance_rad",
            "box_post_movel_step4_movej_velocity_tolerance_rad_sec",
            "box_post_movel_step4_movej_feedback_max_age_sec",
            "box_pre_detection_right_movej_command_units_per_degree",
            "box_pre_detection_right_movej_timeout_sec",
            "box_pre_detection_right_movej_position_tolerance_rad",
            "box_pre_detection_right_movej_velocity_tolerance_rad_sec",
            "box_pre_detection_right_movej_feedback_max_age_sec",
            "box_pre_detection_left_movej_command_units_per_degree",
            "box_pre_detection_left_movej_timeout_sec",
            "box_pre_detection_left_movej_position_tolerance_rad",
            "box_pre_detection_left_movej_velocity_tolerance_rad_sec",
            "box_pre_detection_left_movej_feedback_max_age_sec",
            "box_pre_target_arm_movej_command_units_per_degree",
            "box_pre_target_arm_movej_position_tolerance_rad",
            "box_pre_target_arm_movej_velocity_tolerance_rad_sec",
            "box_pre_target_arm_movej_feedback_max_age_sec",
            "box_pre_target_arm_movej_timeout_sec",
            "box_body_home_timeout_sec",
            "box_step2_waist_endpoint_sync_feedback_max_age_sec",
            "box_step2_waist_endpoint_sync_timeout_sec",
            "box_step2_waist_endpoint_sync_stable_samples",
            "box_step2_waist_endpoint_sync_final_position_tolerance_m",
            "box_step2_waist_endpoint_sync_final_orientation_tolerance_rad",
            "grasp_box_tf_body_home_carry_timeout_sec",
            "grasp_box_tf_body_home_carry_tf_timeout_sec",
            "grasp_box_tf_body_home_carry_position_tolerance_m",
            "grasp_box_tf_body_home_carry_orientation_tolerance_rad",
            "grasp_box_tf_body_home_carry_stable_samples",
            "grasp_box_tf_body_home_carry_left_movel_velocity_percent",
            "grasp_box_tf_body_home_carry_right_movel_velocity_percent",
            "grasp_box_tf_body_home_carry_final_correction_velocity_percent",
            "drag_box_tf_body_home_carry_timeout_sec",
            "drag_box_tf_body_home_carry_tf_timeout_sec",
            "drag_box_tf_body_home_carry_position_tolerance_m",
            "drag_box_tf_body_home_carry_orientation_tolerance_rad",
            "drag_box_tf_body_home_carry_stable_samples",
            "drag_box_tf_body_home_carry_left_movel_velocity_percent",
            "drag_box_tf_body_home_carry_right_movel_velocity_percent",
            "drag_box_tf_body_home_carry_final_correction_velocity_percent",
            "place_box_test_left_movel_velocity_percent",
            "place_box_test_right_movel_velocity_percent",
            "place_box_test_table_height_base_footprint_m",
            "place_box_test_bigbox_half_height_m",
            "place_box_test_smallbox_half_height_m",
            "place_box_test_waist_clearance_m",
            "place_box_test_waist_y_search_step_m",
            "place_box_test_waist_z_search_step_m",
            "place_box_test_waist_ik_max_step_deg",
            "place_box_test_descent_y_search_step_m",
            "place_box_test_descent_y_max_step_m",
            "place_box_test_descent_ik_max_joint_step_deg",
            "place_box_test_descent_z_target_tolerance_m",
            "place_box_test_descent_z_difference_tolerance_m",
            "place_box_test_descent_z_check_timeout_sec",
            "place_box_test_descent_z_feedback_max_age_sec",
            "place_box_test_descent_z_correction_step_m",
            "place_box_test_descent_z_correction_max_travel_m",
            "place_box_test_table_descent_velocity_percent",
            "place_box_test_table_coarse_step_m",
            "place_box_test_table_fine_step_m",
            "place_box_test_table_fine_distance_m",
            "place_box_test_table_max_overtravel_m",
            "place_box_test_table_early_contact_tolerance_m",
            "place_box_test_table_support_delta_fz_n",
            "place_box_test_table_unloaded_abs_fz_n",
            "place_box_test_post_support_max_abs_work_fz_n",
            "place_box_test_timeout_sec",
            "place_box_test_start_body_tolerance_rad",
            "place_box_test_position_tolerance_m",
            "place_box_test_orientation_tolerance_rad",
            "place_box_test_target_consistency_position_tolerance_m",
            "place_box_test_target_consistency_orientation_tolerance_rad",
            "place_box_test_force_unload_baseline_duration_sec",
            "place_box_test_force_unload_baseline_timeout_sec",
            "place_box_test_force_unload_sensor_max_age_sec",
            "place_box_test_force_unload_threshold_left_counts",
            "place_box_test_force_unload_threshold_right_counts",
            "place_box_test_force_unload_required_duration_sec",
            "place_box_test_post_support_z_equalization_velocity_percent",
            "place_box_test_post_support_z_equalization_max_correction_m",
            "place_box_test_post_support_z_equalization_min_high_arm_downward_m",
            "place_box_test_post_support_z_equalization_tolerance_m",
            "place_box_test_post_support_z_equalization_timeout_sec",
            "place_box_test_post_release_arm_joint2_angle_deg",
            "place_box_test_post_release_tool_y_retreat_m",
            "place_box_test_post_release_tool_y_retreat_velocity_percent",
            "place_box_test_post_release_tool_y_retreat_timeout_sec",
            "place_box_test_post_release_arm_movej_velocity_percent",
            "place_box_test_post_release_arm_movej_position_tolerance_rad",
            "place_box_test_post_release_arm_movej_velocity_tolerance_rad_sec",
            "place_box_test_post_release_arm_movej_feedback_max_age_sec",
            "place_box_test_post_release_arm_movej_timeout_sec",
            "place_box_test_post_release_arm_home_command_units_per_degree",
            "place_box_test_post_release_arm_home_velocity_percent",
            "place_box_test_post_release_arm_home_position_tolerance_rad",
            "place_box_test_post_release_arm_home_velocity_tolerance_rad_sec",
            "place_box_test_post_release_arm_home_feedback_max_age_sec",
            "place_box_test_post_release_arm_home_timeout_sec",
        )
        for name in positive_parameters:
            if not math.isfinite(self._float(name)) or self._float(name) <= 0.0:
                raise ValueError(f"parameter '{name}' must be finite and positive")

        for name in (
            "adaptive_grasp_velocity_percent",
            "adaptive_lift_velocity_percent",
            "box_post_movel_velocity_percent",
            "drag_box_left_join_velocity_percent",
            "grasp_box_tf_post_waist_pre_movej_velocity_percent",
            "drag_box_tf_post_waist_pre_movej_velocity_percent",
            "grasp_box_tf_body_home_carry_left_movel_velocity_percent",
            "grasp_box_tf_body_home_carry_right_movel_velocity_percent",
            "grasp_box_tf_body_home_carry_final_correction_velocity_percent",
            "drag_box_tf_body_home_carry_left_movel_velocity_percent",
            "drag_box_tf_body_home_carry_right_movel_velocity_percent",
            "drag_box_tf_body_home_carry_final_correction_velocity_percent",
            "place_box_test_left_movel_velocity_percent",
            "place_box_test_right_movel_velocity_percent",
            "place_box_test_table_descent_velocity_percent",
            "place_box_test_post_support_z_equalization_velocity_percent",
            "place_box_test_post_release_arm_movej_velocity_percent",
            "place_box_test_post_release_tool_y_retreat_velocity_percent",
            "place_box_test_post_release_arm_home_velocity_percent",
        ):
            if self._float(name) > 100.0:
                raise ValueError(f"{name} must be in (0, 100]")

        for name in (
            "direct_sdk_port",
            "direct_sdk_connect_level",
        ):
            if self._integer(name) <= 0:
                raise ValueError(f"parameter '{name}' must be positive")

        nonnegative_parameters = (
            "adaptive_grasp_side_clearance_m",
            "command_repeat_interval_sec",
            "torso_settle_sec",
            "arm_settle_sec",
            "gripper_settle_sec",
            "box_object_pose_result_timeout_sec",
            "box_foundation_pose_pre_settle_sec",
            "box_foundation_pose_post_settle_sec",
            "box_detection_posture_settle_sec",
            "box_place_release_delay_sec",
            "grasp_box_tf_body_home_carry_arm_start_delay_sec",
            "drag_box_tf_body_home_carry_arm_start_delay_sec",
            "grasp_box_tf_body_home_carry_arm_start_lead_sec",
            "drag_box_tf_body_home_carry_arm_start_lead_sec",
        )
        for name in nonnegative_parameters:
            if not math.isfinite(self._float(name)) or self._float(name) < 0.0:
                raise ValueError(f"parameter '{name}' must be finite and nonnegative")

        if self._integer("command_repeat_count") <= 0:
            raise ValueError("command_repeat_count must be positive")
        if self._integer("arm_joint_target_stable_samples") <= 0:
            raise ValueError("arm_joint_target_stable_samples must be positive")
        if self._integer("torso_target_stable_samples") <= 0:
            raise ValueError("torso_target_stable_samples must be positive")
        if self._integer("box_detection_attempts") <= 0:
            raise ValueError("box_detection_attempts must be positive")
        if self._integer("box_joint1_stable_samples") <= 0:
            raise ValueError("box_joint1_stable_samples must be positive")
        if not 0 <= self._integer("box_post_movel_step_count") <= 5:
            raise ValueError("box_post_movel_step_count must be in [0, 5]")
        if self._integer("box_post_arm_stable_samples") <= 0:
            raise ValueError("box_post_arm_stable_samples must be positive")
        if self._float("box_post_arm_movej_command_units_per_degree") <= 0.0:
            raise ValueError(
                "box_post_arm_movej_command_units_per_degree must be positive"
            )
        if self._integer("box_post_arm_movej_velocity") <= 0:
            raise ValueError("box_post_arm_movej_velocity must be positive")
        if self._integer("box_post_arm_movej_velocity") > 100:
            raise ValueError("box_post_arm_movej_velocity must be in (0, 100]")
        if self._integer("box_post_arm_movej_blend_radius") < 0:
            raise ValueError("box_post_arm_movej_blend_radius must be nonnegative")
        if self._integer("box_post_arm_movej_trajectory_connect") not in (0, 1):
            raise ValueError("box_post_arm_movej_trajectory_connect must be 0 or 1")
        if self._integer("box_post_arm_movej_left_device") < 0:
            raise ValueError("box_post_arm_movej_left_device must be nonnegative")
        if self._integer("box_post_arm_movej_right_device") < 0:
            raise ValueError("box_post_arm_movej_right_device must be nonnegative")
        if self._float("box_post_movel_step4_movej_command_units_per_degree") <= 0.0:
            raise ValueError(
                "box_post_movel_step4_movej_command_units_per_degree must be positive"
            )
        if not math.isfinite(
            float(self._integer("box_post_movel_step4_movej_joint2_units"))
        ):
            raise ValueError("box_post_movel_step4_movej_joint2_units must be finite")
        if not 1 <= self._integer("box_post_movel_step4_movej_velocity") <= 100:
            raise ValueError("box_post_movel_step4_movej_velocity must be in [1, 100]")
        for name in (
            "box_post_movel_step4_movej_left_device",
            "box_post_movel_step4_movej_right_device",
        ):
            if self._integer(name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self._integer("box_post_movel_step4_movej_blend_radius") < 0:
            raise ValueError(
                "box_post_movel_step4_movej_blend_radius must be nonnegative"
            )
        if self._integer("box_post_movel_step4_movej_trajectory_connect") not in (0, 1):
            raise ValueError(
                "box_post_movel_step4_movej_trajectory_connect must be 0 or 1"
            )
        if self._integer("box_post_movel_step4_movej_stable_samples") <= 0:
            raise ValueError(
                "box_post_movel_step4_movej_stable_samples must be positive"
            )
        for name in (
            "box_pre_detection_right_movej_device",
            "box_pre_detection_left_movej_device",
        ):
            if self._integer(name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "box_pre_target_arm_movej_left_device",
            "box_pre_target_arm_movej_right_device",
        ):
            if self._integer(name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "box_pre_detection_right_movej_velocity",
            "box_pre_detection_left_movej_velocity",
            "box_pre_target_arm_movej_velocity",
            "box_preparation_movej_velocity",
        ):
            if not 1 <= self._integer(name) <= 100:
                raise ValueError(f"{name} must be in [1, 100]")
        for name in (
            "box_pre_detection_right_movej_blend_radius",
            "box_pre_detection_left_movej_blend_radius",
            "box_pre_target_arm_movej_blend_radius",
        ):
            if self._integer(name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "box_pre_detection_right_movej_trajectory_connect",
            "box_pre_detection_left_movej_trajectory_connect",
            "box_pre_target_arm_movej_trajectory_connect",
        ):
            if self._integer(name) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if self._integer("box_pre_target_arm_movej_stable_samples") <= 0:
            raise ValueError("box_pre_target_arm_movej_stable_samples must be positive")
        for name in (
            "box_pre_detection_right_movej_stable_samples",
            "box_pre_detection_left_movej_stable_samples",
        ):
            if self._integer(name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self._integer("box_body_home_velocity") <= 0:
            raise ValueError("box_body_home_velocity must be positive")
        if self._integer("box_body_home_blend_radius") < 0:
            raise ValueError("box_body_home_blend_radius must be nonnegative")
        for prefix in (
            "grasp_box_tf_body_home_carry",
            "drag_box_tf_body_home_carry",
        ):
            motion_mode = self._string(
                f"{prefix}_arm_motion_mode"
            ).strip().lower()
            if motion_mode not in ("movel", "movej_p"):
                raise ValueError(
                    f"{prefix}_arm_motion_mode must be 'movel' or 'movej_p'"
                )
            if self._integer(f"{prefix}_segments") <= 0:
                raise ValueError(f"{prefix}_segments must be positive")
            if not 1 <= self._integer(f"{prefix}_body_velocity") <= 100:
                raise ValueError(f"{prefix}_body_velocity must be in [1, 100]")
            for name in ("body_blend_radius", "arm_blend_radius"):
                if not 0 <= self._integer(f"{prefix}_{name}") <= 100:
                    raise ValueError(f"{prefix}_{name} must be in [0, 100]")
            if not self._string(f"{prefix}_carrier_frame").strip().lstrip("/"):
                raise ValueError(f"{prefix}_carrier_frame must not be empty")
        for prefix in ("grasp_box_tf_force_clamp", "drag_box_tf_force_clamp"):
            mode = self._string(f"{prefix}_mode").strip().lower()
            if mode not in ("disabled", "closed_loop"):
                raise ValueError(
                    f"{prefix}_mode must be disabled or closed_loop"
                )
            left_target_n = self._float(
                f"{prefix}_sdk_target_force_left_n"
            )
            right_target_n = self._float(
                f"{prefix}_sdk_target_force_right_n"
            )
            if not math.isfinite(left_target_n) or left_target_n >= 0.0:
                raise ValueError(
                    f"{prefix}_sdk_target_force_left_n must be finite and negative"
                )
            if not math.isfinite(right_target_n) or right_target_n <= 0.0:
                raise ValueError(
                    f"{prefix}_sdk_target_force_right_n must be finite and positive"
                )
            sdk_speed = self._float(f"{prefix}_sdk_speed_mm_s")
            if not math.isfinite(sdk_speed) or not 0.1 <= sdk_speed <= 10.0:
                raise ValueError(
                    f"{prefix}_sdk_speed_mm_s must be in [0.1, 10.0]"
                )
            sdk_period = self._float(f"{prefix}_sdk_control_period_sec")
            if not math.isfinite(sdk_period) or not 0.01 <= sdk_period <= 0.1:
                raise ValueError(
                    f"{prefix}_sdk_control_period_sec must be in [0.01, 0.1]"
                )
            baseline_window = self._float(
                f"{prefix}_sdk_baseline_stability_window_sec"
            )
            baseline_max_span = self._float(
                f"{prefix}_sdk_baseline_stability_max_span_n"
            )
            baseline_timeout = self._float(
                f"{prefix}_sdk_baseline_stability_timeout_sec"
            )
            if not math.isfinite(baseline_window) or baseline_window < sdk_period:
                raise ValueError(
                    f"{prefix}_sdk_baseline_stability_window_sec "
                    "must be at least one SDK control period"
                )
            if not math.isfinite(baseline_max_span) or baseline_max_span <= 0.0:
                raise ValueError(
                    f"{prefix}_sdk_baseline_stability_max_span_n "
                    "must be finite and positive"
                )
            if baseline_max_span >= min(abs(left_target_n), right_target_n):
                raise ValueError(
                    f"{prefix}_sdk_baseline_stability_max_span_n "
                    "must be below both contact thresholds"
                )
            if not math.isfinite(baseline_timeout) or baseline_timeout <= baseline_window:
                raise ValueError(
                    f"{prefix}_sdk_baseline_stability_timeout_sec "
                    "must exceed the stability window"
                )
            if prefix == "drag_box_tf_force_clamp":
                confirm_sec = self._float(
                    "drag_box_tf_force_clamp_initial_right_post_stop_confirm_sec"
                )
                min_delta_n = self._float(
                    "drag_box_tf_force_clamp_initial_right_post_stop_min_delta_n"
                )
                if not math.isfinite(confirm_sec) or not 2 * sdk_period <= confirm_sec <= 2.0:
                    raise ValueError(
                        "drag_box_tf_force_clamp_initial_right_post_stop_confirm_sec "
                        "must be at least two SDK periods and at most 2 seconds"
                    )
                if not math.isfinite(min_delta_n) or not 0.0 < min_delta_n <= right_target_n:
                    raise ValueError(
                        "drag_box_tf_force_clamp_initial_right_post_stop_min_delta_n "
                        "must be above zero and no greater than the right contact target"
                    )
                contact_samples = self._integer(
                    "drag_box_tf_force_clamp_initial_right_contact_consecutive_samples"
                )
                if not 1 <= contact_samples <= 10:
                    raise ValueError(
                        "drag_box_tf_force_clamp_initial_right_contact_consecutive_samples "
                        "must be in [1, 10]"
                    )
                retry_attempts = self._integer(
                    "drag_box_tf_force_clamp_initial_right_max_attempts"
                )
                if not 1 <= retry_attempts <= 3:
                    raise ValueError(
                        "drag_box_tf_force_clamp_initial_right_max_attempts "
                        "must be in [1, 3]"
                    )
            for arm in ("left", "right"):
                max_travel = self._float(
                    f"{prefix}_sdk_max_travel_{arm}_m"
                )
                if not math.isfinite(max_travel) or max_travel <= 0.0:
                    raise ValueError(
                        f"{prefix}_sdk_max_travel_{arm}_m must be positive"
                    )
            for suffix in (
                "motion_timeout_sec",
                "timeout_sec",
                "sensor_max_age_sec",
            ):
                name = f"{prefix}_{suffix}"
                if not math.isfinite(self._float(name)) or self._float(name) <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
        if self._string("place_box_test_box_type").strip().lower() not in (
            "smallbox",
            "bigbox",
        ):
            raise ValueError("place_box_test_box_type must be 'smallbox' or 'bigbox'")
        if self._integer("place_box_test_segments") <= 0:
            raise ValueError("place_box_test_segments must be positive")
        for name in (
            "place_box_test_waist_y_search_half_range_m",
            "place_box_test_waist_z_max_drop_m",
            "place_box_test_waist_ik_min_margin_deg",
            "place_box_test_post_support_arm_base_descent_m",
        ):
            if not math.isfinite(self._float(name)) or self._float(name) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        from .place_descent_search import DescentYSettings
        y_step = self._float("place_box_test_descent_work_y_step_m")
        y_limit = self._float("place_box_test_descent_work_y_max_travel_m")
        if not (math.isfinite(y_step) and math.isfinite(y_limit)
                and 0 <= y_step <= 0.005 and y_step <= y_limit <= 0.100):
            raise ValueError("placement Work-Y requires 0 <= step <= 0.005m and step <= travel <= 0.100m")
        if not 1 <= self._integer("place_box_test_descent_z_correction_max_attempts") <= 5:
            raise ValueError("descent Z correction attempts must be within [1,5]")
        if not (0 < self._float("place_box_test_descent_z_correction_step_m")
                <= self._float("place_box_test_descent_z_correction_max_travel_m") <= 0.010):
            raise ValueError("descent Z correction step must be <= travel budget <= 0.010m")
        DescentYSettings(
            self._float("place_box_test_descent_y_search_half_range_m"),
            self._float("place_box_test_descent_y_search_step_m"),
            self._float("place_box_test_descent_y_max_step_m"),
            self._float("place_box_test_waist_ik_min_margin_deg"),
            self._float("place_box_test_descent_ik_max_joint_step_deg"),
        ).validate()
        if self._float("place_box_test_post_release_tool_y_retreat_m") > 0.005:
            raise ValueError(
                "place_box_test_post_release_tool_y_retreat_m must not exceed 0.005m"
            )
        if self._float("place_box_test_post_support_arm_base_descent_m") > 0.030:
            raise ValueError(
                "place_box_test_post_support_arm_base_descent_m must not exceed 0.030m"
            )
        if self._float("place_box_test_table_fine_step_m") > self._float(
            "place_box_test_table_coarse_step_m"
        ):
            raise ValueError("place_box_test fine descent step must not exceed coarse step")
        for arm in ("left", "right"):
            sign = self._float(f"place_box_test_table_support_sign_{arm}")
            if not math.isfinite(sign) or abs(sign) != 1.0:
                raise ValueError(
                    f"place_box_test_table_support_sign_{arm} must be +1 or -1"
                )
        if not 1 <= self._integer("place_box_test_body_velocity") <= 100:
            raise ValueError("place_box_test_body_velocity must be in [1, 100]")
        for name in (
            "place_box_test_body_blend_radius",
            "place_box_test_arm_blend_radius",
        ):
            if not 0 <= self._integer(name) <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        if self._integer("place_box_test_stable_samples") <= 0:
            raise ValueError("place_box_test_stable_samples must be positive")
        if self._integer("place_box_test_force_unload_baseline_min_samples") <= 0:
            raise ValueError(
                "place_box_test_force_unload_baseline_min_samples must be positive"
            )
        if self._integer("place_box_test_force_unload_filter_samples") <= 0:
            raise ValueError(
                "place_box_test_force_unload_filter_samples must be positive"
            )
        for arm in ("left", "right"):
            sign = self._float(f"place_box_test_force_unload_sign_{arm}")
            if not math.isfinite(sign) or abs(sign) <= 1e-12:
                raise ValueError(
                    f"place_box_test_force_unload_sign_{arm} must be finite and nonzero"
                )
        if self._string("place_box_test_arm_motion_mode").strip().lower() not in (
            "movel",
            "movel_offset",
        ):
            raise ValueError(
                "place_box_test_arm_motion_mode must be 'movel' or 'movel_offset'"
            )
        if self._integer("place_box_test_arm_offset_frame_type") not in (0, 1):
            raise ValueError(
                "place_box_test_arm_offset_frame_type must be 0 (work) or 1 (tool)"
            )
        if self._integer("place_box_test_post_release_arm_movej_stable_samples") <= 0:
            raise ValueError(
                "place_box_test_post_release_arm_movej_stable_samples must be positive"
            )
        if self._integer("place_box_test_post_release_arm_home_stable_samples") <= 0:
            raise ValueError(
                "place_box_test_post_release_arm_home_stable_samples must be positive"
            )
        if self._boolean("box_step2_waist_endpoint_sync_enabled"):
            for prefix in (
                "grasp_box_tf_body_home_carry",
                "drag_box_tf_body_home_carry",
            ):
                if self._boolean(f"{prefix}_enabled"):
                    raise ValueError(
                        f"{prefix}_enabled and "
                        "box_step2_waist_endpoint_sync_enabled are mutually exclusive"
                    )
        if (
            not 0
            <= self._integer("box_step2_waist_endpoint_sync_body_blend_radius")
            <= 100
        ):
            raise ValueError(
                "box_step2_waist_endpoint_sync_body_blend_radius must be in [0, 100]"
            )
        for layer in range(1, 5):
            prefix = f"box_step2_waist_endpoint_sync_layer{layer}_"
            if self._integer(f"{prefix}segments") not in (1, 2):
                raise ValueError(f"{prefix}segments must be 1 or 2")
            for name in (
                f"{prefix}forward_body_velocity",
                f"{prefix}reverse_body_velocity",
            ):
                if not 1 <= self._integer(name) <= 100:
                    raise ValueError(f"{name} must be in [1, 100]")
            for name in (
                f"{prefix}forward_left_movel_velocity_percent",
                f"{prefix}forward_right_movel_velocity_percent",
                f"{prefix}reverse_left_movel_velocity_percent",
                f"{prefix}reverse_right_movel_velocity_percent",
            ):
                if not 1.0 <= self._float(name) <= 100.0:
                    raise ValueError(f"{name} must be in [1, 100]")
        if self._integer("box_joint1_device") <= 0:
            raise ValueError("box_joint1_device must be positive")
        if self._integer("box_joint1_velocity") <= 0:
            raise ValueError("box_joint1_velocity must be positive")
        if self._integer("box_joint1_velocity") > 100:
            raise ValueError("box_joint1_velocity must be in (0, 100]")
        if self._integer("box_body_movej_velocity") <= 0:
            raise ValueError("box_body_movej_velocity must be positive")
        if self._integer("box_body_movej_velocity") > 100:
            raise ValueError("box_body_movej_velocity must be in (0, 100]")
        if self._integer("box_joint1_blend_radius") < 0:
            raise ValueError("box_joint1_blend_radius must be nonnegative")
        if self._integer("box_object_pose_instance_index") < 0:
            raise ValueError("box_object_pose_instance_index must be nonnegative")

        box_confidence = self._float("box_object_pose_confidence_threshold")
        if not 0.0 <= box_confidence <= 1.0:
            raise ValueError("box_object_pose_confidence_threshold must be in [0, 1]")
        if not math.isfinite(self._float("drag_box_tf_workflow_reference_base_x_m")):
            raise ValueError("drag_box_tf_workflow_reference_base_x_m must be finite")
        tolerance = self._float("drag_box_tf_workflow_base_x_tolerance_m")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "drag_box_tf_workflow_base_x_tolerance_m must be nonnegative and finite"
            )
        if self._boolean("box_mission_enabled"):
            for name in (
                "box_grasp_left_observation_joint_positions",
                "box_grasp_right_observation_joint_positions",
                "box_grasp_torso_prepare_positions",
            ):
                if all(abs(value) < 1e-9 for value in self._float_array(name)):
                    raise ValueError(
                        f"box_mission_enabled requires configured '{name}'"
                    )

        # Validate every generated TF action/model/layer profile at startup.
        # This catches a missing or malformed layer value before an action is
        # accepted, while retaining the legacy parameter validation above.
        for name, _default in self._tf_layer_parameter_defaults():
            if "approach_angle_deg" in name:
                if not math.isfinite(self._float(name)):
                    raise ValueError(f"parameter '{name}' must be finite")
                continue
            if (
                "body_home_carry_" in name
                and "_movel_velocity_percent_" in name
            ):
                speed = self._float(name)
                if not math.isfinite(speed) or not 1.0 <= speed <= 100.0:
                    raise ValueError(
                        f"{name} must be finite and in [1, 100]"
                    )
                continue
            if "body_home_carry_body_velocity_" in name:
                speed = self._integer(name)
                if not 1 <= speed <= 100:
                    raise ValueError(f"{name} must be in [1, 100]")
                continue
            expected_length = (
                7
                if (
                    "pre_detection_" in name
                    and "_movej_joint_units" in name
                    or "post_detection_" in name
                    and "_movej_joint_units" in name
                    or "target_correction_pose_box" in name
                )
                else 3
            )
            values = self._float_array(name)
            if len(values) != expected_length:
                raise ValueError(
                    f"parameter '{name}' must contain {expected_length} values"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"parameter '{name}' contains NaN or Inf")
            if (
                "target_correction_pose_box" in name
                and math.sqrt(sum(value * value for value in values[3:])) <= 1e-12
            ):
                raise ValueError(f"parameter '{name}' contains a zero quaternion")

    def _string(self, name: str) -> str:
        return str(self.get_parameter(name).value).strip()

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _integer(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _boolean(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _float_array(self, name: str) -> list[float]:
        return [float(value) for value in self.get_parameter(name).value]

    def _boolean_array(self, name: str) -> list[bool]:
        return [bool(value) for value in self.get_parameter(name).value]

    def _string_array(self, name: str) -> list[str]:
        return [str(value).strip() for value in self.get_parameter(name).value]
