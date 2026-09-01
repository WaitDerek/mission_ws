"""Mission parameter validation and typed accessors."""

import math


class ParameterValidationMixin:
    """Validate calibrated inputs before Action servers accept goals."""

    def _validate_parameters(self) -> None:
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

        left_join_mode = self._string("drag_box_left_join_mode").strip().lower()
        if left_join_mode not in ("immediate", "after_drag3"):
            raise ValueError(
                "parameter 'drag_box_left_join_mode' must be "
                "'immediate' or 'after_drag3'"
            )
        left_join_motion_mode = (
            self._string("drag_box_left_join_motion_mode").strip().lower()
        )
        if left_join_motion_mode not in ("movel", "movej_p"):
            raise ValueError(
                "parameter 'drag_box_left_join_motion_mode' must be "
                "'movel' or 'movej_p'"
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
        for name in ("grasp_box_tf_detection_arm", "drag_box_tf_detection_arm"):
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
            ("drag_box_left_join_pre_movej_joint_units", 7),
            ("box_body_home_joint_units", 4),
            ("box_step2_waist_endpoint_sync_home_joint_units", 4),
            ("grasp_box_tf_body_home_carry_joint_units", 4),
            ("drag_box_tf_body_home_carry_joint_units", 4),
            ("place_box_test_body_joint_units", 4),
            ("place_box_test_start_body_joint_units", 4),
            ("place_box_test_left_target_pose_arm_base", 7),
            ("place_box_test_right_target_pose_arm_base", 7),
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
            "place_box_test_final_correction_velocity_percent",
            "place_box_test_timeout_sec",
            "place_box_test_start_body_tolerance_rad",
            "place_box_test_position_tolerance_m",
            "place_box_test_orientation_tolerance_rad",
            "place_box_test_target_consistency_position_tolerance_m",
            "place_box_test_target_consistency_orientation_tolerance_rad",
        )
        for name in positive_parameters:
            if not math.isfinite(self._float(name)) or self._float(name) <= 0.0:
                raise ValueError(f"parameter '{name}' must be finite and positive")

        for name in (
            "adaptive_grasp_velocity_percent",
            "adaptive_lift_velocity_percent",
            "box_post_movel_velocity_percent",
            "drag_box_left_join_velocity_percent",
            "grasp_box_tf_body_home_carry_left_movel_velocity_percent",
            "grasp_box_tf_body_home_carry_right_movel_velocity_percent",
            "grasp_box_tf_body_home_carry_final_correction_velocity_percent",
            "drag_box_tf_body_home_carry_left_movel_velocity_percent",
            "drag_box_tf_body_home_carry_right_movel_velocity_percent",
            "drag_box_tf_body_home_carry_final_correction_velocity_percent",
            "place_box_test_left_movel_velocity_percent",
            "place_box_test_right_movel_velocity_percent",
            "place_box_test_final_correction_velocity_percent",
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
            if mode not in ("disabled", "monitor_only", "closed_loop"):
                raise ValueError(
                    f"{prefix}_mode must be disabled, monitor_only, or closed_loop"
                )
            for arm in ("left", "right"):
                contact_value = self._float(
                    f"{prefix}_contact_threshold_{arm}_counts"
                )
                clamped_value = self._float(
                    f"{prefix}_clamped_threshold_{arm}_counts"
                )
                hold_value = self._float(f"{prefix}_hold_threshold_{arm}_counts")
                emergency_value = self._float(
                    f"{prefix}_emergency_threshold_{arm}_counts"
                )
                for suffix in (
                    "contact_threshold",
                    "clamped_threshold",
                    "hold_threshold",
                    "emergency_threshold",
                ):
                    name = f"{prefix}_{suffix}_{arm}_counts"
                    if not math.isfinite(self._float(name)) or self._float(name) < 0.0:
                        raise ValueError(f"{name} must be finite and nonnegative")
                if not contact_value <= clamped_value <= emergency_value:
                    raise ValueError(
                        f"{prefix} {arm} thresholds must satisfy "
                        "contact <= clamped <= emergency"
                    )
                if hold_value > clamped_value:
                    raise ValueError(
                        f"{prefix}_hold_threshold_{arm}_counts must not exceed "
                        f"{prefix}_clamped_threshold_{arm}_counts"
                    )
                sign = self._float(f"{prefix}_force_sign_{arm}")
                if not math.isfinite(sign) or abs(sign) <= 1e-12:
                    raise ValueError(f"{prefix}_force_sign_{arm} must be nonzero")
                max_distance = self._float(f"{prefix}_max_distance_{arm}_m")
                if not math.isfinite(max_distance) or max_distance <= 0.0:
                    raise ValueError(
                        f"{prefix}_max_distance_{arm}_m must be finite and positive"
                    )
            for suffix in (
                "baseline_duration_sec",
                "baseline_timeout_sec",
                "arm_velocity_tolerance_rad_sec",
                "search_step_m",
                "fine_step_m",
                "movel_velocity_percent",
                "motion_timeout_sec",
                "timeout_sec",
                "sensor_max_age_sec",
                "contact_required_duration_sec",
                "clamped_required_duration_sec",
                "hold_wait_sec",
                "hold_required_duration_sec",
            ):
                name = f"{prefix}_{suffix}"
                if not math.isfinite(self._float(name)) or self._float(name) <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
            if self._float(f"{prefix}_movel_velocity_percent") > 100.0:
                raise ValueError(f"{prefix}_movel_velocity_percent must be in (0, 100]")
            if self._integer(f"{prefix}_baseline_min_samples") <= 0:
                raise ValueError(f"{prefix}_baseline_min_samples must be positive")
            if self._integer(f"{prefix}_filter_samples") <= 0:
                raise ValueError(f"{prefix}_filter_samples must be positive")
            if self._integer(f"{prefix}_max_correction_count") < 0:
                raise ValueError(f"{prefix}_max_correction_count must be nonnegative")
        if self._string("place_box_test_box_type").strip().lower() != "smallbox":
            raise ValueError("place_box_test_box_type must be 'smallbox'")
        if self._integer("place_box_test_segments") <= 0:
            raise ValueError("place_box_test_segments must be positive")
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
