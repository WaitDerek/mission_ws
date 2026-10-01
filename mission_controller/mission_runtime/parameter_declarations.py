"""Mission parameter declarations and calibrated defaults."""

import math


_DIRECT_GRASP_POST_WAIST_PRE_MOVEJ_UNITS = {
    1: {
        "left": [68910, 40432, -82821, -101837, -82739, -30193, 9473],
        "right": [-82466, 50035, 86176, -92648, 76926, 38450, -18016],
    },
    2: {
        "left": [28810, 50943, -22412, -94677, -50758, -10833, 33468],
        "right": [-15865, 32806, 42268, -94310, 40166, 31301, -6061],
    },
    3: {
        "left": [33802, 45067, -26518, -91134, -49283, -18163, 31401],
        "right": [-30930, 41897, 37382, -91322, 48187, 24447, -24760],
    },
    4: {
        "left": [14473, 45035, -3736, -82846, -49272, -8368, 38850],
        "right": [-10593, 40398, 16217, -76297, 41544, 24117, -33964],
    },
}


class ParameterDeclarationsMixin:
    """Declare the existing public Mission parameter surface."""

    def _declare_parameters(self) -> None:
        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    "execute_adaptive_box_grasp_action_name",
                    "/execute_adaptive_box_grasp",
                ),
                ("execute_box_grasp_action_name", "/execute_box_grasp"),
                ("grasp_box_tf_action_name", "/grasp_box_tf"),
                (
                    "execute_drag_box_grasp_action_name",
                    "/execute_drag_box_grasp",
                ),
                (
                    "execute_drag_box_grasp_tf_action_name",
                    "/execute_drag_box_grasp_tf",
                ),
                ("execute_box_place_action_name", "/execute_box_place"),
                ("place_box_test_action_name", "/place_box_test"),
                (
                    "acquire_mission_lease_service_name",
                    "/mission/acquire_workflow_lease",
                ),
                (
                    "release_mission_lease_service_name",
                    "/mission/release_workflow_lease",
                ),
                # Small-box placement test taught at body joint1/2/3
                # [-40, -60, -40] deg.
                # The action requires a preceding /grasp_box_tf goal in the
                # same controller process so the rigid box->Link7 transforms
                # remain available after mobile-base transport.
                ("place_box_test_enabled", True),
                ("place_box_test_box_type", "smallbox"),
                ("place_box_test_start_body_joint_units", [0, 0, 0, 0]),
                ("place_box_test_body_joint_units", [-40000, -80000, -40000, 0]),
                ("place_box_test_dynamic_table_enabled", True),
                ("place_box_test_table_height_base_footprint_m", 0.755),
                ("place_box_test_bigbox_half_height_m", 0.16865),
                ("place_box_test_smallbox_half_height_m", 0.13980),
                ("place_box_test_waist_clearance_m", 0.040),
                ("place_box_test_waist_workspace_enabled", True),
                ("place_box_test_waist_y_search_half_range_m", 0.100),
                ("place_box_test_waist_y_search_step_m", 0.020),
                ("place_box_test_waist_z_max_drop_m", 0.080),
                ("place_box_test_waist_z_search_step_m", 0.040),
                ("place_box_test_waist_ik_min_margin_deg", 2.0),
                ("place_box_test_waist_ik_max_step_deg", 75.0),
                ("place_box_test_table_descent_velocity_percent", 5.0),
                ("place_box_test_descent_mode", "segmented"),
                ("place_box_test_continuous_left_min_z_footprint_m", 0.975131),
                ("place_box_test_continuous_right_min_z_footprint_m", 0.966566),
                ("place_box_test_continuous_left_velocity_percent", 10.0),
                ("place_box_test_continuous_right_velocity_percent", 15.0),
                ("place_box_test_continuous_max_drop_difference_m", 0.03),
                ("place_box_test_descent_y_search_enabled", False),  # Legacy; direct descent ignores search.
                ("place_box_test_descent_work_y_step_m", 0.002),
                ("place_box_test_descent_work_y_max_travel_m", 0.100),
                ("place_box_test_descent_y_search_half_range_m", 0.100),
                ("place_box_test_descent_y_search_step_m", 0.001),
                ("place_box_test_descent_y_max_step_m", 0.002),
                ("place_box_test_descent_ik_max_joint_step_deg", 10.0),
                ("place_box_test_descent_z_target_tolerance_m", 0.020),
                ("place_box_test_descent_z_difference_tolerance_m", 0.005),
                ("place_box_test_descent_z_check_timeout_sec", 1.0),
                ("place_box_test_descent_z_feedback_max_age_sec", 0.25),
                ("place_box_test_descent_z_correction_max_attempts", 3),
                ("place_box_test_descent_z_correction_step_m", 0.002),
                ("place_box_test_descent_z_correction_max_travel_m", 0.005),
                ("place_box_test_table_coarse_step_m", 0.005),
                ("place_box_test_table_fine_step_m", 0.001),
                ("place_box_test_table_fine_distance_m", 0.030),
                ("place_box_test_table_max_overtravel_m", 0.005),
                ("place_box_test_table_early_contact_tolerance_m", 0.500),
                ("place_box_test_table_support_delta_fz_n", 3.0),
                ("place_box_test_table_unloaded_abs_fz_n", 100.0),
                ("place_box_test_table_support_sign_left", 1.0),
                ("place_box_test_table_support_sign_right", 1.0),
                ("place_box_test_post_support_arm_base_descent_m", 0.030),
                ("place_box_test_post_support_max_abs_work_fz_n", 200.0),
                (
                    "place_box_test_left_target_pose_arm_base",
                    [
                        -0.484519,
                        0.564599,
                        -0.102020,
                        -0.435,
                        -0.446,
                        -0.566,
                        0.539,
                    ],
                ),
                (
                    "place_box_test_right_target_pose_arm_base",
                    [
                        -0.261094,
                        -0.591816,
                        -0.098310,
                        0.394,
                        -0.409,
                        0.568,
                        0.594,
                    ],
                ),
                # Explicit per-box-type placement targets. The legacy
                # unsuffixed targets above remain small-box aliases.
                (
                    "place_box_test_left_target_pose_arm_base_smallbox",
                    [
                        -0.484519,
                        0.564599,
                        -0.102020,
                        -0.435,
                        -0.446,
                        -0.566,
                        0.539,
                    ],
                ),
                (
                    "place_box_test_right_target_pose_arm_base_smallbox",
                    [
                        -0.261094,
                        -0.591816,
                        -0.098310,
                        0.394,
                        -0.409,
                        0.568,
                        0.594,
                    ],
                ),
                (
                    "place_box_test_left_target_pose_arm_base_bigbox",
                    [
                        -0.387499,
                        0.568216,
                        0.014259,
                        -0.413,
                        -0.426,
                        -0.575,
                        0.562,
                    ],
                ),
                (
                    "place_box_test_right_target_pose_arm_base_bigbox",
                    [
                        -0.327162,
                        -0.578776,
                        0.012840,
                        0.416,
                        -0.402,
                        0.612,
                        0.537,
                    ],
                ),
                ("place_box_test_segments", 6),
                ("place_box_test_body_velocity", 12),
                ("place_box_test_body_blend_radius", 5),
                ("place_box_test_arm_blend_radius", 5),
                ("place_box_test_left_movel_velocity_percent", 10.0),
                ("place_box_test_right_movel_velocity_percent", 10.0),
                ("place_box_test_arm_motion_mode", "movel"),
                ("place_box_test_arm_offset_frame_type", 0),
                ("place_box_test_timeout_sec", 180.0),
                ("place_box_test_start_body_tolerance_rad", 0.035),
                ("place_box_test_position_tolerance_m", 0.020),
                ("place_box_test_orientation_tolerance_rad", 0.10),
                ("place_box_test_stable_samples", 3),
                (
                    "place_box_test_target_consistency_position_tolerance_m",
                    0.15,
                ),
                (
                    "place_box_test_target_consistency_orientation_tolerance_rad",
                    0.35,
                ),
                ("place_box_test_body_stop_enabled", True),
                ("place_box_test_body_stop_command", "stop"),
                # Placement release interlock.  Capture the carried-box Fz
                # baseline before motion and require both wrists to unload as
                # the table takes the weight.  Fx is deliberately ignored.
                ("place_box_test_force_unload_enabled", False),
                ("place_box_test_force_unload_baseline_duration_sec", 0.5),
                ("place_box_test_force_unload_baseline_min_samples", 20),
                ("place_box_test_force_unload_baseline_timeout_sec", 3.0),
                ("place_box_test_force_unload_filter_samples", 15),
                ("place_box_test_force_unload_sensor_max_age_sec", 0.2),
                ("place_box_test_force_unload_threshold_left_counts", 7000.0),
                ("place_box_test_force_unload_threshold_right_counts", 7000.0),
                ("place_box_test_force_unload_sign_left", 1.0),
                ("place_box_test_force_unload_sign_right", 1.0),
                ("place_box_test_force_unload_required_duration_sec", 0.2),
                # Once table support is detected, equalize the physical
                # Link7 heights in the common base_link frame before opening
                # the grippers and moving Joint2 outward.
                ("place_box_test_post_support_z_equalization_enabled", True),
                (
                    "place_box_test_post_support_z_equalization_velocity_percent",
                    5.0,
                ),
                (
                    "place_box_test_post_support_z_equalization_max_correction_m",
                    0.05,
                ),
                (
                    "place_box_test_post_support_z_equalization_min_high_arm_downward_m",
                    0.02,
                ),
                (
                    "place_box_test_post_support_z_equalization_tolerance_m",
                    0.005,
                ),
                (
                    "place_box_test_post_support_z_equalization_timeout_sec",
                    30.0,
                ),
                ("place_box_test_post_release_arm_movej_enabled", True),
                ("place_box_test_post_release_tool_y_retreat_enabled", True),
                ("place_box_test_post_release_tool_y_retreat_m", 0.001),
                ("place_box_test_post_release_tool_y_retreat_velocity_percent", 3.0),
                ("place_box_test_post_release_tool_y_retreat_timeout_sec", 30.0),
                ("place_box_test_post_release_arm_joint2_angle_deg", 60.0),
                ("place_box_test_post_release_arm_movej_velocity_percent", 15.0),
                ("place_box_test_post_release_arm_movej_position_tolerance_rad", 0.035),
                (
                    "place_box_test_post_release_arm_movej_velocity_tolerance_rad_sec",
                    0.035,
                ),
                ("place_box_test_post_release_arm_movej_feedback_max_age_sec", 1.0),
                ("place_box_test_post_release_arm_movej_timeout_sec", 180.0),
                ("place_box_test_post_release_arm_movej_stable_samples", 3),
                ("place_box_test_post_release_body_home_enabled", True),
                ("place_box_test_post_release_body_home_joint_units", [0, 0, 0, 0]),
                ("place_box_test_post_release_arm_home_enabled", True),
                (
                    "place_box_test_post_release_arm_home_left_joint_units",
                    [0, 0, 0, 0, 0, 0, 0],
                ),
                (
                    "place_box_test_post_release_arm_home_right_joint_units",
                    [0, 0, 0, 0, 0, 0, 0],
                ),
                (
                    "place_box_test_post_release_arm_home_command_units_per_degree",
                    1000.0,
                ),
                (
                    "place_box_test_post_release_arm_home_velocity_percent",
                    15.0,
                ),
                (
                    "place_box_test_post_release_arm_home_position_tolerance_rad",
                    0.035,
                ),
                (
                    "place_box_test_post_release_arm_home_velocity_tolerance_rad_sec",
                    0.035,
                ),
                (
                    "place_box_test_post_release_arm_home_feedback_max_age_sec",
                    1.0,
                ),
                ("place_box_test_post_release_arm_home_timeout_sec", 180.0),
                ("place_box_test_post_release_arm_home_stable_samples", 3),
                ("adaptive_box_action_enabled", True),
                ("adaptive_freeze_frame", "base_link"),
                ("adaptive_require_detection_timestamp", True),
                ("adaptive_tf_cache_time_sec", 180.0),
                ("adaptive_detection_tf_timeout_sec", 5.0),
                ("adaptive_runtime_tf_timeout_sec", 2.0),
                # TF GraspBox freezes the detected box in the chassis-fixed
                # frame, then re-expresses the target in each live arm base
                # after the waist has reached its layer pose.
                ("grasp_box_tf_freeze_frame", "base_link"),
                ("grasp_box_tf_left_contact_forward_delta_scale", 1.0),
                ("grasp_box_tf_right_contact_forward_delta_scale", 1.0),
                ("drag_box_tf_left_contact_forward_delta_scale", 1.0),
                ("drag_box_tf_right_contact_forward_delta_scale", 1.0),
                ("grasp_box_tf_detection_tf_timeout_sec", 5.0),
                ("grasp_box_tf_runtime_tf_timeout_sec", 5.0),
                ("grasp_box_tf_require_detection_timestamp", True),
                ("grasp_box_tf_detection_arm", "right"),
                ("drag_box_tf_detection_arm", "right"),
                ("grasp_box_tf_horizontal_constraint_enabled", False),
                ("drag_box_tf_horizontal_constraint_enabled", False),
                ("grasp_box_tf_contact_height_offset_m", 0.0),
                ("grasp_box_tf_left_backward_offset_m_layer1", 0.0),
                ("grasp_box_tf_right_forward_offset_m_layer1", 0.0),
                ("grasp_box_tf_right_forward_offset_m_layer2", 0.0),
                ("grasp_box_tf_right_forward_offset_m_layer3", 0.0),
                ("grasp_box_tf_right_forward_offset_m_layer4", 0.0),

                ("drag_box_tf_contact_height_offset_m", 0.0),
                ("drag_box_tf_right_backward_offset_m_layer1", 0.0),
                ("drag_box_tf_right_backward_offset_m_layer2", 0.0),
                ("drag_box_tf_left_join_forward_offset_m", 0.0),
                ("drag_box_tf_left_join_forward_offset_m_layer1", -1.0),
                ("drag_box_tf_right_forward_offset_m_layer3", 0.0),
                ("drag_box_tf_right_forward_offset_m_layer4", 0.0),
                ("drag_box_tf_left_join_forward_offset_m_layer3", -1.0),
                ("drag_box_tf_left_join_forward_offset_m_layer4", -1.0),
                ("drag_box_tf_left_join_contact_span_m_bigbox", 0.0),
                ("drag_box_tf_detection_arm_smallbox", "right"),
                ("drag_box_tf_workflow_reference_base_x_m", 0.110135),
                ("drag_box_tf_workflow_base_x_tolerance_m", 0.010),
                # After DragBox TF detection, move the left arm
                # through its configured avoidance joint sequence before
                # pickup planning.  The target is independent by model/layer.
                ("drag_box_tf_post_detection_left_movej_enabled", True),
                # Left-arm transition poses for the staged DragBox TF
                # detection-to-avoidance MoveJ sequence.
                (
                    "drag_box_tf_post_detection_left_transition_joint_units_bigbox",
                    [1104, 83030, -93968, -4985, 99573, 5457, 2099],
                ),
                (
                    "drag_box_tf_post_detection_left_transition_joint_units_smallbox",
                    [1104, 83030, -93968, -4985, 99573, 5457, 2099],
                ),
                # DragBox begins with the right arm.  RM75 IK is a seeded
                # single-solution solver, so bias and constrain the initial
                # MoveJ_P branch to a negative Joint4 configuration.
                (
                    "drag_box_tf_right_initial_ik_joint4_negative_required",
                    True,
                ),
                ("drag_box_tf_right_initial_ik_joint4_seed_deg", -60.0),
                # Optional post-Step2 carry controller for /grasp_box_tf.
                # The waist uses MoveJ while both arms receive synchronized,
                # segmented SDK MoveL endpoints.  Box translation follows the
                # common chest frame, while its base_link orientation and both
                # box->controller-TCP transforms remain fixed.
                ("grasp_box_tf_body_home_carry_enabled", True),
                ("grasp_box_tf_body_home_carry_carrier_frame", "chest_Link"),
                ("grasp_box_tf_body_home_carry_joint_units", [0, 0, 0, 0]),
                ("grasp_box_tf_body_home_carry_segments", 6),
                ("grasp_box_tf_body_home_carry_continuous_enabled", True),
                # Arm-side command used while carrying the box back with the
                # waist.  Both arms intentionally share one mode so their
                # endpoint commands stay synchronized.
                ("grasp_box_tf_body_home_carry_arm_motion_mode", "movel"),
                ("grasp_box_tf_body_home_carry_body_velocity", 12),
                ("grasp_box_tf_body_home_carry_body_blend_radius", 0),
                ("grasp_box_tf_body_home_carry_arm_blend_radius", 5),
                (
                    "grasp_box_tf_body_home_carry_left_movel_velocity_percent",
                    12.0,
                ),
                (
                    "grasp_box_tf_body_home_carry_right_movel_velocity_percent",
                    12.0,
                ),
                # Backward-compatible body-first delay; normally zero when
                # arm_start_lead_sec is used.
                ("grasp_box_tf_body_home_carry_arm_start_delay_sec", 0.0),
                # Positive lead releases both arms first, then submits the
                # waist command after this delay.
                ("grasp_box_tf_body_home_carry_arm_start_lead_sec", 0.0),
                ("grasp_box_tf_body_home_carry_timeout_sec", 180.0),
                ("grasp_box_tf_body_home_carry_tf_timeout_sec", 5.0),
                ("grasp_box_tf_body_home_carry_position_tolerance_m", 0.01),
                (
                    "grasp_box_tf_body_home_carry_orientation_tolerance_rad",
                    0.0872665,
                ),
                ("grasp_box_tf_body_home_carry_stable_samples", 3),
                ("grasp_box_tf_body_home_carry_final_correction_enabled", True),
                (
                    "grasp_box_tf_body_home_carry_final_correction_velocity_percent",
                    12.0,
                ),
                ("grasp_box_tf_body_home_carry_body_stop_enabled", True),
                ("grasp_box_tf_body_home_carry_body_stop_command", "stop"),
                # Optional post-Step2 carry controller for /execute_drag_box_grasp_tf.
                # Keep the complete tuning surface independent from GraspBox;
                # the switch is off by default for backward compatibility.
                ("drag_box_tf_body_home_carry_enabled", True),
                ("drag_box_tf_body_home_carry_carrier_frame", "chest_Link"),
                ("drag_box_tf_body_home_carry_joint_units", [0, 0, 0, 0]),
                ("drag_box_tf_body_home_carry_segments", 6),
                ("drag_box_tf_body_home_carry_continuous_enabled", True),
                # DragBox TF has its own mode switch; it does not inherit the
                # GraspBox TF setting.
                ("drag_box_tf_body_home_carry_arm_motion_mode", "movel"),
                ("drag_box_tf_body_home_carry_body_velocity", 12),
                ("drag_box_tf_body_home_carry_body_blend_radius", 0),
                ("drag_box_tf_body_home_carry_arm_blend_radius", 5),
                (
                    "drag_box_tf_body_home_carry_left_movel_velocity_percent",
                    12.0,
                ),
                (
                    "drag_box_tf_body_home_carry_right_movel_velocity_percent",
                    12.0,
                ),
                # Kept independent so DragBox can be tuned separately.
                ("drag_box_tf_body_home_carry_arm_start_delay_sec", 0.0),
                ("drag_box_tf_body_home_carry_arm_start_lead_sec", 0.0),
                ("drag_box_tf_body_home_carry_timeout_sec", 180.0),
                ("drag_box_tf_body_home_carry_tf_timeout_sec", 5.0),
                ("drag_box_tf_body_home_carry_position_tolerance_m", 0.01),
                (
                    "drag_box_tf_body_home_carry_orientation_tolerance_rad",
                    0.0872665,
                ),
                ("drag_box_tf_body_home_carry_stable_samples", 3),
                ("drag_box_tf_body_home_carry_final_correction_enabled", True),
                (
                    "drag_box_tf_body_home_carry_final_correction_velocity_percent",
                    12.0,
                ),
                ("drag_box_tf_body_home_carry_body_stop_enabled", True),
                ("drag_box_tf_body_home_carry_body_stop_command", "stop"),
                # Native SDK Tool-Y force-position clamping for TF workflows.
                ("grasp_box_tf_force_clamp_mode", "closed_loop"),
                ("grasp_box_tf_force_clamp_sdk_target_force_left_n", -2.0),
                ("grasp_box_tf_force_clamp_sdk_target_force_right_n", 2.0),
                ("grasp_box_tf_force_carry_target_force_left_n", -75.0),
                ("grasp_box_tf_force_carry_target_force_right_n", 75.0),
                ("grasp_box_tf_force_clamp_sdk_speed_mm_s", 3.0),
                ("grasp_box_tf_force_clamp_sdk_max_travel_left_m", 0.20),
                ("grasp_box_tf_force_clamp_sdk_max_travel_right_m", 0.20),
                ("grasp_box_tf_force_clamp_sdk_control_period_sec", 0.02),
                ("grasp_box_tf_force_clamp_sdk_baseline_stability_window_sec", 0.3),
                ("grasp_box_tf_force_clamp_sdk_baseline_stability_max_span_n", 0.4),
                ("grasp_box_tf_force_clamp_sdk_baseline_stability_timeout_sec", 5.0),
                ("grasp_box_tf_force_clamp_motion_timeout_sec", 15.0),
                ("grasp_box_tf_force_clamp_timeout_sec", 60.0),
                ("grasp_box_tf_force_clamp_sensor_max_age_sec", 0.2),
                ("grasp_box_tf_force_clamp_stop_after_clamp_confirmed", False),
                ("drag_box_tf_force_clamp_mode", "closed_loop"),
                ("drag_box_tf_force_clamp_sdk_target_force_left_n", -2.0),
                ("drag_box_tf_force_clamp_sdk_target_force_right_n", 2.0),
                ("drag_box_tf_force_carry_post_drag3_target_force_left_n", -75.0),
                ("drag_box_tf_force_carry_post_drag3_target_force_right_n", 75.0),
                ("drag_box_tf_force_clamp_sdk_speed_mm_s", 3.0),
                ("drag_box_tf_force_clamp_sdk_max_travel_left_m", 0.20),
                ("drag_box_tf_force_clamp_sdk_max_travel_right_m", 0.20),
                ("drag_box_tf_force_clamp_sdk_control_period_sec", 0.02),
                ("drag_box_tf_force_clamp_sdk_baseline_stability_window_sec", 0.3),
                ("drag_box_tf_force_clamp_sdk_baseline_stability_max_span_n", 0.4),
                ("drag_box_tf_force_clamp_sdk_baseline_stability_timeout_sec", 5.0),
                ("drag_box_tf_force_clamp_initial_right_post_stop_confirm_sec", 0.5),
                ("drag_box_tf_force_clamp_initial_right_post_stop_min_delta_n", 1.0),
                ("drag_box_tf_force_clamp_initial_right_contact_consecutive_samples", 1),
                ("drag_box_tf_force_clamp_initial_right_max_attempts", 3),
                ("drag_box_tf_force_clamp_motion_timeout_sec", 15.0),
                ("drag_box_tf_force_clamp_timeout_sec", 60.0),
                ("drag_box_tf_force_clamp_sensor_max_age_sec", 0.2),
                ("drag_box_tf_force_clamp_stop_after_clamp_confirmed", False),
                # Canonical object axes after pose normalization are X=down,
                # Y=forward, Z=right. Grasp from the two object-Z side faces.
                ("adaptive_grasp_span_axis_object", [0.0, 0.0, 1.0]),
                ("adaptive_grasp_height_axis_object", [-1.0, 0.0, 0.0]),
                ("adaptive_grasp_side_clearance_m", 0.0),
                ("adaptive_grasp_height_offset_m", 0.0),
                (
                    "adaptive_grasp_correction_rpy",
                    [-1.5707963267948966, 1.5707963267948966, 0.0],
                ),
                (
                    "adaptive_left_grasp_extra_rpy",
                    [0.0, 0.0, 3.141592653589793],
                ),
                ("adaptive_right_grasp_extra_rpy", [0.0, 0.0, 0.0]),
                ("adaptive_grasp_velocity_percent", 12.0),
                ("adaptive_grasp_timeout_sec", 120.0),
                ("adaptive_lift_distance_m", 0.10),
                ("adaptive_lift_velocity_percent", 12.0),
                ("adaptive_lift_timeout_sec", 120.0),
                ("box_mission_enabled", True),
                (
                    "box_object_pose_action_name",
                    "/object_pose/estimate",
                ),
                ("box_object_pose_camera_side", "right"),
                ("box_object_pose_topic", "/mission/box_object_pose"),
                ("box_object_pose_camera_topic", "/object_pose/pose"),
                ("box_object_pose_raw_topic", "/mission/box_object_pose_raw"),
                # Temporary smallbox profile: reuse the current bigbox
                # geometric calibration values until smallbox is calibrated.
                ("box_object_pose_model_label", "smallbox"),
                ("box_object_pose_instance_index", 0),
                ("box_object_pose_confidence_threshold", 0.8),
                ("box_object_pose_result_timeout_sec", 120.0),
                # Hold the confirmed detection posture before and after each
                # FoundationPose request so RGB-D frames and robot TF settle.
                ("box_foundation_pose_pre_settle_sec", 5.0),
                ("box_foundation_pose_post_settle_sec", 5.0),
                ("box_camera_pose_axis_min_dot", 0.5),
                ("pickup_task_action_name", "/pickup_task"),
                ("pickup_task_result_timeout_sec", 120.0),
                ("box_direct_movel_enabled", True),
                ("direct_motion_backend", "python_sdk"),
                ("direct_movel_service_name", "/realbot/movel"),
                (
                    "direct_sdk_root",
                    "RM_API2/Demo/RMDemo_Python/RMDemo_SimpleProcess",
                ),
                ("direct_sdk_left_ip", "192.168.127.18"),
                ("direct_sdk_right_ip", "192.168.127.19"),
                ("direct_sdk_port", 8080),
                ("direct_sdk_connect_level", 3),
                ("direct_sdk_motion_timeout_sec", 120.0),
                ("box_grasp_execution_mode", "arms_only"),
                # Endpoint-only offline IK search around the configured
                # per-action/per-layer waist target.
                ("waist_workspace_optimization_enabled", False),
                ("drag_box_tf_fixed_waist_enabled_bigbox_layer1", False),
                ("drag_box_tf_fixed_waist_enabled_bigbox_layer2", False),
                ("waist_workspace_movel_endpoint_ik_enabled", True),
                ("waist_workspace_ik_seed_mode", "zero"),
                ("box_ik_joint4_negative_required", True),
                ("waist_workspace_search_mode", "coarse_to_fine"),
                ("waist_workspace_candidate_step_deg", 1.0),
                ("waist_workspace_candidate_delta_deg", 10.0),
                ("waist_workspace_minimum_margin_deg", 0.1),
                ("waist_workspace_coarse_step_deg", 5.0),
                ("waist_workspace_coarse_top_k", 8),
                ("waist_workspace_candidate_diversity_deg", 5.0),
                ("waist_workspace_refine_radius_deg", 3.0),
                ("waist_workspace_refine_step_deg", 2.0),
                ("waist_workspace_sobol_sample_count", 1024),
                ("waist_workspace_sobol_seed", 0),
                (
                    "waist_workspace_joint_min_deg",
                    [-89.0, -149.0, -89.0],
                ),
                (
                    "waist_workspace_joint_max_deg",
                    [4.0, 149.0, 4.0],
                ),
                (
                    "waist_workspace_left_arm_joint_min_deg",
                    [
                        -174.0,
                        -14.0,
                        -174.0,
                        -104.0,
                        -169.0,
                        -64.0,
                        -99.0,
                    ],
                ),
                (
                    "waist_workspace_left_arm_joint_max_deg",
                    [
                        174.0,
                        174.0,
                        174.0,
                        104.0,
                        174.0,
                        19.0,
                        89.0,
                    ],
                ),
                (
                    "waist_workspace_right_arm_joint_min_deg",
                    [
                        -174.0,
                        -14.0,
                        -174.0,
                        -104.0,
                        -169.0,
                        -19.0,
                        -89.0,
                    ],
                ),
                (
                    "waist_workspace_right_arm_joint_max_deg",
                    [
                        174.0,
                        174.0,
                        174.0,
                        104.0,
                        174.0,
                        64.0,
                        99.0,
                    ],
                ),
                # Once FoundationPose is frozen, direct GraspBox starts waist
                # IK in a background worker and concurrently moves both arms
                # to this layer-specific posture before moving the waist.
                # Bigbox and smallbox remain independent calibration surfaces
                # even while their initial taught values are identical.
                ("grasp_box_tf_post_waist_pre_movej_enabled", True),
                # Calibration-only hold point. The interactive dual-arm
                # calibrator enables it temporarily and always restores it.
                (
                    "grasp_box_tf_calibration_stop_after_initial_dual_target_enabled",
                    False,
                ),
                (
                    "grasp_box_tf_post_waist_pre_movej_command_units_per_degree",
                    1000.0,
                ),
                ("grasp_box_tf_post_waist_pre_movej_velocity_percent", 15.0),
                ("grasp_box_tf_post_waist_pre_movej_timeout_sec", 120.0),
                *[
                    (
                        "grasp_box_tf_post_waist_pre_movej_"
                        f"{arm}_joint_units_{model}_layer{layer}",
                        list(
                            _DIRECT_GRASP_POST_WAIST_PRE_MOVEJ_UNITS[layer][arm]
                        ),
                    )
                    for model in ("bigbox", "smallbox")
                    for layer in range(1, 5)
                    for arm in ("left", "right")
                ],
                # DragBox owns a separate interface. It is deliberately
                # disabled and its zero arrays mean "not calibrated"; it never
                # falls back to the direct-grasp postures above.
                ("drag_box_tf_post_waist_pre_movej_enabled", False),
                # DragBox keeps its left-arm avoidance profile, but prepares
                # the right arm with the matching GraspBox model/layer target.
                ("drag_box_tf_right_grasp_preparation_enabled", True),
                # Calibration-only hold point. Normal DragBox execution must
                # leave this false; the interactive calibrator enables it
                # temporarily and restores it when the script exits.
                (
                    "drag_box_tf_calibration_stop_after_initial_right_target_enabled",
                    False,
                ),
                (
                    "drag_box_tf_calibration_stop_after_left_join_enabled",
                    False,
                ),
                (
                    "drag_box_tf_calibration_left_join_joint_override_enabled",
                    False,
                ),
                (
                    "drag_box_tf_calibration_left_join_joint_target_deg",
                    [
                        -62.412,
                        25.861,
                        24.668,
                        -40.876,
                        -35.381,
                        -8.514,
                        -3.713,
                    ],
                ),
                (
                    "drag_box_tf_calibration_pause_after_initial_right_target_enabled",
                    False,
                ),
                (
                    "drag_box_tf_calibration_continue_after_initial_right_target",
                    False,
                ),
                ("drag_box_tf_calibration_pause_timeout_sec", 1800.0),
                (
                    "drag_box_tf_post_waist_pre_movej_command_units_per_degree",
                    1000.0,
                ),
                ("drag_box_tf_post_waist_pre_movej_velocity_percent", 10.0),
                ("drag_box_tf_post_waist_pre_movej_timeout_sec", 120.0),
                *[
                    (
                        "drag_box_tf_post_waist_pre_movej_"
                        f"{arm}_joint_units_{model}_layer{layer}",
                        [0, 0, 0, 0, 0, 0, 0],
                    )
                    for model in ("bigbox", "smallbox")
                    for layer in range(1, 5)
                    for arm in ("left", "right")
                ],
                ("box_joint1_command_service_name", "/robot/command"),
                ("box_joint1_feedback_topic", "/mcap/body"),
                ("box_joint1_name", "joint1"),
                ("box_joint2_name", "joint2"),
                ("box_joint3_name", "joint3"),
                ("box_joint4_name", "joint4"),
                ("box_joint1_device", 2),
                ("box_joint1_detection_angle_deg", 0.0),
                ("box_joint1_approach_angle_deg", -13.0),
                (
                    "box_layer_joint1_approach_angles_deg",
                    [-13.0, -45.0, -70.0, -89.0],
                ),
                (
                    "box_layer_joint2_approach_angles_deg",
                    [0.0, -85.0, -120.0, -149.0],
                ),
                (
                    "box_layer_joint3_approach_angles_deg",
                    [0.0, -55.0, -73.0, -89.0],
                ),
                (
                    "box_layer_joint1_approach_angles_deg_bigbox",
                    [-13.0, -45.0, -70.0, -89.0],
                ),
                (
                    "box_layer_joint2_approach_angles_deg_bigbox",
                    [0.0, -85.0, -120.0, -149.0],
                ),
                (
                    "box_layer_joint3_approach_angles_deg_bigbox",
                    [0.0, -55.0, -73.0, -89.0],
                ),
                (
                    "box_layer_joint1_approach_angles_deg_smallbox",
                    [-13.0, -45.0, -70.0, -89.0],
                ),
                (
                    "box_layer_joint2_approach_angles_deg_smallbox",
                    [0.0, -85.0, -120.0, -149.0],
                ),
                (
                    "box_layer_joint3_approach_angles_deg_smallbox",
                    [0.0, -70.0, -73.0, -89.0],
                ),
                (
                    "box_layer_joint123_configured",
                    [True, True, True, True],
                ),
                ("box_joint2_detection_angle_deg", 0.0),
                ("box_joint2_approach_angle_deg", 0.0),
                ("box_joint3_detection_angle_deg", 0.0),
                ("box_joint3_approach_angle_deg", 0.0),
                ("box_joint1_command_units_per_degree", 1000.0),
                ("box_body_command_units_per_degree", [1000.0] * 4),
                ("box_joint1_velocity", 12),
                ("box_body_movej_velocity", 12),
                ("box_joint1_blend_radius", 0),
                ("box_joint1_axis_xyz", [0.0, 0.0, 1.0]),
                ("box_joint1_feedback_to_geometric_sign", 1.0),
                ("box_joint2_axis_xyz", [0.0, 0.0, -1.0]),
                ("box_joint3_axis_xyz", [0.0, 0.0, 1.0]),
                ("box_joint2_feedback_to_urdf_axis_sign", 1.0),
                ("box_joint3_feedback_to_urdf_axis_sign", 1.0),
                ("box_waist1_origin_xyz", [-0.080814986, 0.135049308, 0.266]),
                ("box_waist1_origin_rpy", [math.pi, math.pi / 2.0, 0.0]),
                ("box_waist2_origin_xyz", [-0.384, 0.0, -0.0074]),
                ("box_waist2_origin_rpy", [0.0, 0.0, 0.0]),
                ("box_waist3_origin_xyz", [-0.277703995, 0.0, 0.0024]),
                ("box_waist3_origin_rpy", [0.0, 0.0, 0.0]),
                ("box_waist3_to_chest_xyz", [-0.123796005, 0.0, -0.0755]),
                ("box_waist3_to_chest_rpy", [0.0, math.pi / 2.0, 0.0]),
                ("box_chest_to_left_arm_base_xyz", [0.012, 0.0, -0.2975]),
                ("box_chest_to_left_arm_base_rpy", [0.0, math.pi, 0.0]),
                ("box_chest_to_right_arm_base_xyz", [-0.012, 0.0, -0.2975]),
                ("box_chest_to_right_arm_base_rpy", [math.pi, 0.0, 0.0]),
                ("box_joint1_position_tolerance_rad", 0.01),
                ("box_joint1_velocity_tolerance_rad_sec", 0.01),
                ("box_joint1_feedback_max_age_sec", 1.0),
                ("box_joint1_wait_timeout_sec", 80.0),
                ("box_joint1_stable_samples", 3),
                (
                    "direct_movel_target_mode",
                    "camera_offset_box_orientation",
                ),
                ("direct_movel_box_relative_model_label", "smallbox"),
                ("direct_movel_motion_mode", "movej_p"),
                ("direct_movel_velocity_percent", 15.0),
                ("direct_movel_blocking", True),
                ("box_post_movel_enabled", False),
                ("box_post_movel_velocity_percent", 12.0),
                ("box_post_lift_left_velocity_percent", 10.0),
                ("box_post_lift_right_velocity_percent", 15.0),
                # TF Step/Drag Cartesian translations use absolute rm_movel
                # targets by default.  rm_movel_offset remains selectable for
                # controlled compatibility testing.
                ("grasp_box_tf_post_movel_sdk_motion_mode", "movel"),
                ("drag_box_tf_post_movel_sdk_motion_mode", "movel"),
                # For TF GraspBox/DragBox dual-arm Cartesian commands, force
                # both arm-base target poses to use the same numeric Z value.
                ("box_tf_equalize_dual_target_z_enabled", True),
                # Step2 is rebuilt from actual bilateral Link8 TF and lifted
                # along base_link +Z.  This preserves grasp span and equal
                # physical height instead of equalizing unrelated arm-base Z.
                ("box_tf_step2_rigid_base_z_lift_enabled", True),
                ("box_tf_step2_contact_height_tolerance_m", 0.10),
                ("box_tf_step2_grasp_span_tolerance_m", 0.002),
                ("drag_box_post_movel_enabled", True),
                # DragBox moves the right arm through Drag3 first, then joins
                # the left arm at its cumulative target before Step2.
                ("drag_box_left_arm_enabled", True),
                ("drag_box_left_join_mode", "after_drag3"),
                ("drag_box_left_join_motion_mode", "staged_ik_movej"),
                ("drag_box_left_join_velocity_percent", 10.0),
                ("drag_box_left_join_timeout_sec", 60.0),
                ("drag_box_left_join_ik_max_attempts", 113),
                ("drag_box_left_join_ik_random_seed_attempts", 100),
                ("drag_box_left_join_joint4_preference_enabled", True),
                ("drag_box_left_join_joint4_negative_required", True),
                ("drag_box_left_join_preferred_joint4_deg_bigbox_layer1", -50.0),
                ("drag_box_left_join_joint4_tolerance_deg_bigbox_layer1", 15.0),
                (
                    "drag_box_left_join_ik_seed_joint_deg",
                    [
                        -62.412,
                        25.861,
                        24.668,
                        -40.876,
                        -35.381,
                        -8.514,
                        -3.713,
                    ],
                ),
                # Reconstruct the moved box from the actual right Link7 TF
                # after Drag3, then derive the delayed left join and rigid
                # dual-arm Step2 from that one common box frame.
                ("drag_box_tf_reanchor_after_drag3_enabled", True),
                ("drag_box_post_movel_step_drag1_left_xyz", [0.0, 0.0, 0.0]),
                ("drag_box_post_movel_step_drag1_right_xyz", [0.14, 0.0, 0.0]),
                ("drag_box_post_movel_step_drag2_left_xyz", [0.0, 0.0, 0.10]),
                ("drag_box_post_movel_step_drag2_right_xyz", [0.0, 0.0, 0.15]),
                ("drag_box_post_movel_step_drag3_left_xyz", [0.0, 0.0, 0.0]),
                ("drag_box_post_movel_step_drag3_right_xyz", [-0.14, 0.0, 0.0]),
                ("box_post_movel_step4_motion_mode", "movej"),
                ("box_post_movel_step4_movej_joint2_units", 40000),
                ("box_post_movel_step4_movej_left_device", 0),
                ("box_post_movel_step4_movej_right_device", 1),
                (
                    "box_post_movel_step4_movej_left_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                (
                    "box_post_movel_step4_movej_right_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                ("box_post_movel_step4_movej_command_units_per_degree", 1000.0),
                ("box_post_movel_step4_movej_velocity", 15),
                ("box_post_movel_step4_movej_blend_radius", 0),
                ("box_post_movel_step4_movej_trajectory_connect", 0),
                ("box_post_movel_step4_movej_timeout_sec", 40.0),
                ("box_post_movel_step4_movej_position_tolerance_rad", 0.01),
                ("box_post_movel_step4_movej_velocity_tolerance_rad_sec", 0.01),
                ("box_post_movel_step4_movej_feedback_max_age_sec", 1.0),
                ("box_post_movel_step4_movej_stable_samples", 3),
                ("box_post_movel_step_count", 2),
                ("box_post_movel_left_step1_xyz", [0.0, 0.0, 0.025]),
                ("box_post_movel_right_step1_xyz", [0.0, 0.0, -0.028]),
                # Smallbox-specific Step1 deltas. Bigbox and callers that do
                # not select a model continue using the generic values.
                ("box_post_movel_left_step1_xyz_smallbox", [0.0, 0.0, -0.035]),
                ("box_post_movel_right_step1_xyz_smallbox", [0.0, 0.0, 0.02]),
                ("box_post_movel_left_step2_xyz", [0.14, 0.0, 0.0]),
                ("box_post_movel_right_step2_xyz", [0.14, 0.0, 0.0]),
                ("box_post_movel_left_step3_xyz", [-0.14, 0.0, 0.0]),
                ("box_post_movel_right_step3_xyz", [-0.14, 0.0, 0.0]),
                ("box_post_movel_left_step4_xyz", [0.0, 0.0, -0.1]),
                ("box_post_movel_right_step4_xyz", [0.0, 0.0, 0.1]),
                ("box_post_movel_left_step5_xyz", [0.0, 0.0, 0.0]),
                ("box_post_movel_right_step5_xyz", [0.0, 0.0, 0.0]),
                ("box_post_arm_movej_enabled", False),
                ("box_post_arm_movej_left_device", 0),
                ("box_post_arm_movej_right_device", 1),
                (
                    "box_post_arm_movej_left_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                (
                    "box_post_arm_movej_right_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                ("box_post_arm_movej_command_units_per_degree", 1000.0),
                ("box_post_arm_movej_velocity", 15),
                ("box_post_arm_movej_blend_radius", 0),
                ("box_post_arm_movej_trajectory_connect", 0),
                ("box_post_arm_movej_timeout_sec", 40.0),
                # Place the wrist-mounted right camera at a known joint pose
                # before FoundationPose detection.
                ("box_pre_detection_arm_movej_enabled", True),
                ("box_pre_detection_right_movej_enabled", True),
                ("box_pre_detection_right_movej_device", 1),
                (
                    "box_pre_detection_right_movej_joint_units",
                    [144725, -5335, 7032, 9843, 7540, -5611, 85414],
                ),
                (
                    "box_layer_pre_detection_right_movej_joint_units",
                    [
                        144725,
                        -5335,
                        7032,
                        9843,
                        7540,
                        -5611,
                        85414,
                        -12083,
                        5105,
                        -17961,
                        -50575,
                        9150,
                        -5641,
                        -66298,
                        23227,
                        13389,
                        -62736,
                        -51630,
                        44662,
                        -12958,
                        -22269,
                        21382,
                        -4978,
                        -274,
                        1282,
                        -5472,
                        3628,
                        -32636,
                    ],
                ),
                # Per-model detection poses.  The generic flattened table is
                # retained as a backward-compatible fallback; GraspBox
                # selects the table for its configured model label.
                (
                    "box_layer_pre_detection_right_movej_joint_units_bigbox",
                    [
                        144725,
                        -5335,
                        7032,
                        9843,
                        7540,
                        -5611,
                        85414,
                        -12083,
                        5105,
                        -17961,
                        -50575,
                        9150,
                        -5641,
                        -66298,
                        23227,
                        13389,
                        -62736,
                        -51630,
                        44662,
                        -12958,
                        -22269,
                        21382,
                        -4978,
                        -274,
                        1282,
                        -5472,
                        3628,
                        -32636,
                    ],
                ),
                (
                    "box_layer_pre_detection_right_movej_joint_units_smallbox",
                    [
                        144725,
                        -5335,
                        7032,
                        9843,
                        7540,
                        -5611,
                        85414,
                        -12083,
                        5105,
                        -17961,
                        -50575,
                        9150,
                        -5641,
                        -66298,
                        23227,
                        13389,
                        -62736,
                        -51630,
                        44662,
                        -12958,
                        -22269,
                        1538,
                        252,
                        146,
                        -25260,
                        -8872,
                        92,
                        -23259,
                    ],
                ),
                (
                    "box_layer_pre_detection_right_movej_configured",
                    [True, True, True, True],
                ),
                ("box_pre_detection_right_movej_command_units_per_degree", 1000.0),
                ("box_pre_detection_right_movej_velocity", 15),
                ("box_pre_detection_right_movej_blend_radius", 0),
                ("box_pre_detection_right_movej_trajectory_connect", 0),
                ("box_pre_detection_right_movej_timeout_sec", 40.0),
                ("box_pre_detection_right_movej_position_tolerance_rad", 0.01),
                ("box_pre_detection_right_movej_velocity_tolerance_rad_sec", 0.01),
                ("box_pre_detection_right_movej_feedback_max_age_sec", 1.0),
                ("box_pre_detection_right_movej_stable_samples", 3),
                # Left-camera counterpart. Defaults mirror the current right
                # table until each left observation pose is calibrated.
                ("box_pre_detection_left_movej_enabled", True),
                ("box_pre_detection_left_movej_device", 0),
                (
                    "box_pre_detection_left_movej_joint_units",
                    [144725, -5335, 7032, 9843, 7540, -5611, 85414],
                ),
                (
                    "box_layer_pre_detection_left_movej_joint_units",
                    [
                        144725,
                        -5335,
                        7032,
                        9843,
                        7540,
                        -5611,
                        85414,
                        -12083,
                        5105,
                        -17961,
                        -50575,
                        9150,
                        -5641,
                        -66298,
                        23227,
                        13389,
                        -62736,
                        -51630,
                        44662,
                        -12958,
                        -22269,
                        21382,
                        -4978,
                        -274,
                        1282,
                        -5472,
                        3628,
                        -32636,
                    ],
                ),
                (
                    "box_layer_pre_detection_left_movej_joint_units_bigbox",
                    [
                        144725,
                        -5335,
                        7032,
                        9843,
                        7540,
                        -5611,
                        85414,
                        -12083,
                        5105,
                        -17961,
                        -50575,
                        9150,
                        -5641,
                        -66298,
                        23227,
                        13389,
                        -62736,
                        -51630,
                        44662,
                        -12958,
                        -22269,
                        21382,
                        -4978,
                        -274,
                        1282,
                        -5472,
                        3628,
                        -32636,
                    ],
                ),
                (
                    "box_layer_pre_detection_left_movej_joint_units_smallbox",
                    [
                        144725,
                        -5335,
                        7032,
                        9843,
                        7540,
                        -5611,
                        85414,
                        -12083,
                        5105,
                        -17961,
                        -50575,
                        9150,
                        -5641,
                        -66298,
                        23227,
                        13389,
                        -62736,
                        -51630,
                        44662,
                        -12958,
                        -22269,
                        1538,
                        252,
                        146,
                        -25260,
                        -8872,
                        92,
                        -23259,
                    ],
                ),
                (
                    "box_layer_pre_detection_left_movej_configured",
                    [True, True, True, True],
                ),
                ("box_pre_detection_left_movej_command_units_per_degree", 1000.0),
                ("box_pre_detection_left_movej_velocity", 10),
                ("box_pre_detection_left_movej_blend_radius", 0),
                ("box_pre_detection_left_movej_trajectory_connect", 0),
                ("box_pre_detection_left_movej_timeout_sec", 40.0),
                ("box_pre_detection_left_movej_position_tolerance_rad", 0.01),
                ("box_pre_detection_left_movej_velocity_tolerance_rad_sec", 0.01),
                ("box_pre_detection_left_movej_feedback_max_age_sec", 1.0),
                ("box_pre_detection_left_movej_stable_samples", 3),
                # Move both arms to this intermediate pose after detection,
                # then send the computed Link8 movej_p targets.
                ("box_pre_target_arm_movej_enabled", True),
                ("box_pre_target_arm_movej_two_stage_enabled", True),
                ("box_pre_target_arm_movej_stage1_joint2_units", 40000),
                ("box_preparation_movej_velocity", 15),
                ("box_pre_target_arm_movej_left_device", 0),
                ("box_pre_target_arm_movej_right_device", 1),
                (
                    "box_pre_target_arm_movej_left_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                (
                    "box_pre_target_arm_movej_right_joint_units",
                    [0, 40000, 0, 0, 0, 0, 0],
                ),
                ("box_pre_target_arm_movej_command_units_per_degree", 1000.0),
                ("box_pre_target_arm_movej_velocity", 15),
                ("box_pre_target_arm_movej_blend_radius", 0),
                ("box_pre_target_arm_movej_trajectory_connect", 0),
                ("box_pre_target_arm_movej_timeout_sec", 40.0),
                ("box_pre_target_arm_movej_position_tolerance_rad", 0.01),
                ("box_pre_target_arm_movej_velocity_tolerance_rad_sec", 0.01),
                ("box_pre_target_arm_movej_feedback_max_age_sec", 1.0),
                ("box_pre_target_arm_movej_stable_samples", 3),
                ("box_post_arm_left_feedback_topic", "/mcap/slave_arm_left"),
                ("box_post_arm_right_feedback_topic", "/mcap/slave_arm_right"),
                ("box_post_arm_position_tolerance_rad", 0.01),
                ("box_post_arm_velocity_tolerance_rad_sec", 0.01),
                ("box_post_arm_feedback_max_age_sec", 1.0),
                ("box_post_arm_stable_samples", 3),
                ("box_body_return_home_enabled", True),
                ("box_body_home_joint_units", [0, 0, 0, 0]),
                ("box_body_home_velocity", 12),
                ("box_body_home_blend_radius", 0),
                ("box_body_home_timeout_sec", 40.0),
                # After post-grasp Step2, optionally send one body MoveJ to
                # home while both arms send one-shot MoveL endpoint targets.
                # The captured Link7 EEPose is preserved in the fixed root
                # frame; the reverse endpoint then restores the measured
                # Step2 state before Step3/Step4 continue.
                ("box_step2_waist_endpoint_sync_enabled", False),
                ("box_step2_waist_endpoint_sync_home_joint_units", [0, 0, 0, 0]),
                ("box_step2_waist_endpoint_sync_body_blend_radius", 0),
                ("box_step2_waist_endpoint_sync_timeout_sec", 180.0),
                ("box_step2_waist_endpoint_sync_feedback_max_age_sec", 0.5),
                ("box_step2_waist_endpoint_sync_final_position_tolerance_m", 0.01),
                (
                    "box_step2_waist_endpoint_sync_final_orientation_tolerance_rad",
                    0.0872665,
                ),
                ("box_step2_waist_endpoint_sync_stable_samples", 3),
                ("box_step2_waist_endpoint_sync_body_stop_enabled", True),
                ("box_step2_waist_endpoint_sync_body_stop_command", "stop"),
                ("box_step2_waist_endpoint_sync_skip_final_body_home", True),
                ("box_step2_waist_endpoint_sync_layer1_configured", False),
                ("box_step2_waist_endpoint_sync_layer1_segments", 1),
                ("box_step2_waist_endpoint_sync_layer1_forward_body_velocity", 12),
                (
                    "box_step2_waist_endpoint_sync_layer1_forward_left_movel_velocity_percent",
                    12.0,
                ),
                (
                    "box_step2_waist_endpoint_sync_layer1_forward_right_movel_velocity_percent",
                    12.0,
                ),
                ("box_step2_waist_endpoint_sync_layer1_reverse_body_velocity", 12),
                (
                    "box_step2_waist_endpoint_sync_layer1_reverse_left_movel_velocity_percent",
                    12.0,
                ),
                (
                    "box_step2_waist_endpoint_sync_layer1_reverse_right_movel_velocity_percent",
                    12.0,
                ),
                ("box_step2_waist_endpoint_sync_layer2_configured", False),
                ("box_step2_waist_endpoint_sync_layer2_segments", 1),
                ("box_step2_waist_endpoint_sync_layer2_forward_body_velocity", 12),
                (
                    "box_step2_waist_endpoint_sync_layer2_forward_left_movel_velocity_percent",
                    12.0,
                ),
                (
                    "box_step2_waist_endpoint_sync_layer2_forward_right_movel_velocity_percent",
                    12.0,
                ),
                ("box_step2_waist_endpoint_sync_layer2_reverse_body_velocity", 12),
                (
                    "box_step2_waist_endpoint_sync_layer2_reverse_left_movel_velocity_percent",
                    12.0,
                ),
                (
                    "box_step2_waist_endpoint_sync_layer2_reverse_right_movel_velocity_percent",
                    12.0,
                ),
                ("box_step2_waist_endpoint_sync_layer3_configured", False),
                ("box_step2_waist_endpoint_sync_layer3_segments", 2),
                ("box_step2_waist_endpoint_sync_layer3_forward_body_velocity", 12),
                (
                    "box_step2_waist_endpoint_sync_layer3_forward_left_movel_velocity_percent",
                    12.0,
                ),
                (
                    "box_step2_waist_endpoint_sync_layer3_forward_right_movel_velocity_percent",
                    12.0,
                ),
                ("box_step2_waist_endpoint_sync_layer3_reverse_body_velocity", 12),
                (
                    "box_step2_waist_endpoint_sync_layer3_reverse_left_movel_velocity_percent",
                    12.0,
                ),
                (
                    "box_step2_waist_endpoint_sync_layer3_reverse_right_movel_velocity_percent",
                    12.0,
                ),
                ("box_step2_waist_endpoint_sync_layer4_configured", False),
                ("box_step2_waist_endpoint_sync_layer4_segments", 1),
                ("box_step2_waist_endpoint_sync_layer4_forward_body_velocity", 12),
                (
                    "box_step2_waist_endpoint_sync_layer4_forward_left_movel_velocity_percent",
                    12.0,
                ),
                (
                    "box_step2_waist_endpoint_sync_layer4_forward_right_movel_velocity_percent",
                    12.0,
                ),
                ("box_step2_waist_endpoint_sync_layer4_reverse_body_velocity", 12),
                (
                    "box_step2_waist_endpoint_sync_layer4_reverse_left_movel_velocity_percent",
                    12.0,
                ),
                (
                    "box_step2_waist_endpoint_sync_layer4_reverse_right_movel_velocity_percent",
                    12.0,
                ),
                ("direct_movel_use_current_fixture_orientation", False),
                (
                    "direct_movel_left_fixed_link8_orientation",
                    [-0.497, -0.503, -0.488, 0.509],
                ),
                (
                    "direct_movel_right_fixed_link8_orientation",
                    [0.482, -0.463, 0.522, 0.528],
                ),
                # During the current Realbots2 calibration, direct targets
                # are Link8 EEPose targets.  Fixture-center compensation is
                # opt-in and remains disabled until the fixture transform is
                # independently verified.
                ("direct_movel_fixture_compensation_enabled", False),
                ("direct_movel_left_offset_xyz", [0.0, 0.0, -0.51]),
                ("direct_movel_right_offset_xyz", [0.0, 0.0, 0.45]),
                (
                    "direct_movel_left_offset_xyz_bigbox_layer1",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_bigbox_layer1",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_bigbox_layer2",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_bigbox_layer2",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_bigbox_layer3",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_bigbox_layer3",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_bigbox_layer4",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_bigbox_layer4",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_smallbox_layer1",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_smallbox_layer1",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_smallbox_layer2",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_smallbox_layer2",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_smallbox_layer3",
                    [0.0, 0.0, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_smallbox_layer3",
                    [0.0, 0.0, 0.50],
                ),
                (
                    "direct_movel_left_offset_xyz_smallbox_layer4",
                    [0.0, -0.025, -0.50],
                ),
                (
                    "direct_movel_right_offset_xyz_smallbox_layer4",
                    [0.0, -0.025, 0.50],
                ),
                (
                    "direct_movel_left_box_to_link8_orientation",
                    [-0.666064, -0.026103, 0.005935, 0.745414],
                ),
                (
                    "direct_movel_right_box_to_link8_orientation",
                    [-0.694193, -0.007865, 0.013341, 0.719622],
                ),
                (
                    "joint123_layer1_left_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer1_right_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer2_left_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer2_right_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer3_left_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer3_right_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer4_left_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "joint123_layer4_right_target_correction_pose_box",
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ),
                (
                    "left_fixture_center_in_link8_xyz",
                    [-0.12, -0.10, 0.05],
                ),
                (
                    "right_fixture_center_in_link8_xyz",
                    [-0.12, 0.10, 0.05],
                ),
                ("box_detection_attempts", 2),
                ("box_width", 0.357),
                ("box_height", 0.127),
                ("box_type", "f455"),
                # FoundationPose publishes the oriented-bounding-box centre
                # with F320 axes X=down, Y=depth, Z=width. Keep those axes by
                # default; each robot pickup profile owns the fixed transform
                # from model axes to its operation/end-effector convention.
                (
                    "box_foundation_to_pickup_rpy",
                    [0.0, 0.0, 0.0],
                ),
                ("arm_joints_service_name", "/move_arm_j"),
                ("go_ready_action_name", "/go_ready"),
                ("torso_topic", "/realbot/motion_target/torso"),
                (
                    "left_gripper_topic",
                    "/realbot/motion_target/gripper_left",
                ),
                (
                    "right_gripper_topic",
                    "/realbot/motion_target/gripper_right",
                ),
                ("joint_state_topic", "/joint_states"),
                ("torso_feedback_topic", "/realbot/feedback/torso"),
                (
                    "left_gripper_feedback_topic",
                    "/realbot/feedback/gripper_left",
                ),
                (
                    "right_gripper_feedback_topic",
                    "/realbot/feedback/gripper_right",
                ),
                (
                    "left_arm_joint_names",
                    [f"L_JOINT_{index}" for index in range(1, 8)],
                ),
                (
                    "right_arm_joint_names",
                    [f"R_JOINT_{index}" for index in range(1, 8)],
                ),
                ("verify_arm_joint_targets", True),
                ("arm_joint_target_tolerance", 0.10),
                ("arm_joint_target_wait_timeout_sec", 20.0),
                ("arm_joint_target_stable_samples", 3),
                ("box_observation_ready_check_enabled", True),
                ("box_observation_feedback_max_age_sec", 2.0),
                ("box_observation_torso_tolerance", 0.10),
                ("torso_target_tolerance", 0.03),
                ("torso_target_wait_timeout_sec", 40.0),
                ("torso_target_stable_samples", 3),
                ("arm_execution_frame", "base_link"),
                ("left_ee_frame", "left_arm_8_Link"),
                ("right_ee_frame", "right_arm_8_Link"),
                ("left_gripper_frame", "left_arm_8_Link"),
                ("right_gripper_frame", "right_arm_8_Link"),
                ("camera_mount_tf_enabled", False),
                ("camera_mount_parent_frame", "right_arm_8_Link"),
                (
                    "camera_mount_child_frame",
                    "realbot_camera_link",
                ),
                # The mechanical flange-to-camera transform comes from URDF.
                # This zero-translation rotation only maps the CAD D405 axes
                # to the ROS camera_link convention: camera +X = D405 +Z.
                ("camera_mount_xyz", [0.0, 0.0, 0.0]),
                (
                    "camera_mount_rpy",
                    [-1.5707963267948966, -1.5707963267948966, 0.0],
                ),
                ("camera_mount_correction_rpy", [0.0, 0.0, 0.0]),
                ("camera_measured_extrinsics_enabled", True),
                # Wrist-camera profile: the camera is rigidly mounted to
                # Link8, so Base->camera is composed from live EEPose and the
                # fixed URDF Link8->RGB optical-center transform.
                ("camera_dynamic_link8_extrinsics_enabled", True),
                ("camera_detection_arm", "right"),
                ("camera_eepose_max_age_sec", 1.0),
                # Fixed transform T_left_right for the new robot.  It maps
                # right-arm-base coordinates into left-arm-base coordinates.
                # The left Base origin is [0.024, 0, 0] in the right Base;
                # left x/y axes are the negatives of right x/y, and z is
                # shared. The inverse is used for left-camera detection.
                ("camera_fixed_cross_arm_transform_enabled", True),
                ("camera_right_base_to_left_base_xyz", [0.024, 0.0, 0.0]),
                (
                    "camera_right_base_to_left_base_quaternion_xyzw",
                    [0.0, 0.0, 1.0, 0.0],
                ),
                ("left_arm_base_frame", "L_base_Link"),
                ("right_arm_base_frame", "R_base_Link"),
                ("left_link8_frame", "left_arm_8_Link"),
                ("right_link8_frame", "right_arm_8_Link"),
                (
                    "camera_left_link8_to_rgb_camera_xyz",
                    [0.097294396234, 0.000243365421, 0.053076686984],
                ),
                (
                    "camera_left_link8_to_rgb_camera_quaternion_xyzw",
                    [
                        0.153045932190,
                        -0.153045932190,
                        -0.690345524096,
                        0.690345524097,
                    ],
                ),
                (
                    "camera_right_link8_to_rgb_camera_xyz",
                    [0.096527, -0.016012, 0.046146],
                ),
                (
                    "camera_right_link8_to_rgb_camera_quaternion_xyzw",
                    [
                        -0.152364,
                        -0.111601,
                        0.682613,
                        0.705953,
                    ],
                ),
                # Use the measured camera origin directly. The J1-height
                # difference between CAD models is not an external-camera
                # calibration measurement.
                ("camera_left_base_xyz", [0.045, 0.08, -0.05]),
                ("camera_right_base_xyz", [0.045, -0.08, -0.08]),
                (
                    "camera_left_base_rpy",
                    [0.0, 1.5707963267948966, 1.5707963267948966],
                ),
                (
                    "camera_right_base_rpy",
                    [0.0, -1.5707963267948966, 1.5707963267948966],
                ),
                ("camera_tf_timeout_sec", 2.0),
                ("dependency_wait_timeout_sec", 10.0),
                ("arm_joints_result_timeout_sec", 60.0),
                ("go_ready_result_timeout_sec", 60.0),
                ("wait_for_command_subscribers", True),
                ("require_command_subscribers", True),
                ("command_subscriber_wait_timeout_sec", 3.0),
                ("command_repeat_count", 10),
                ("command_repeat_interval_sec", 0.005),
                ("torso_settle_sec", 1.0),
                ("arm_settle_sec", 1.0),
                ("gripper_settle_sec", 1.0),
                ("box_detection_posture_settle_sec", 2.0),
                ("box_place_release_delay_sec", 2.0),
                ("torso_reset_positions", [0.0, 0.0, 0.0, 0.0]),
                ("torso_velocities", [0.1, 0.1, 0.1, 0.1]),
                # RealBot dual-arm preparation and clearance postures.
                (
                    "box_grasp_intermediate_left_joint_positions",
                    [1.30, 0.6, 0.0, -1.5, 0.0, 0.0, 0.0],
                ),
                (
                    "box_grasp_intermediate_right_joint_positions",
                    [1.30, -0.6, 0.0, -1.5, 0.0, 0.0, 0.0],
                ),
                (
                    "box_grasp_left_observation_joint_positions",
                    [-0.88, 1.24, -0.70, -2.0, 1.25, 0.1, 0.0],
                ),
                (
                    "box_grasp_right_observation_joint_positions",
                    [
                        0.86,
                        -0.24,
                        0.20,
                        -2.0944,
                        0.174647,
                        -0.618606,
                        0.104098,
                    ],
                ),
                (
                    "box_pickup_clearance_left_joint_positions",
                    [
                        -1.413830,
                        0.687872,
                        -1.236596,
                        -1.839149,
                        1.905532,
                        0.525745,
                        1.146383,
                    ],
                ),
                (
                    "box_pickup_clearance_right_joint_positions",
                    [
                        -1.403617,
                        -0.668723,
                        1.238298,
                        -1.802128,
                        -1.944043,
                        0.435319,
                        -1.307021,
                    ],
                ),
                ("box_grasp_torso_prepare_positions", [0.61, -0.81, -0.6, 0.0]),
                (
                    "box_grasp_torso_lift_positions",
                    [0.41, -0.81, -0.6, 0.0],
                ),
                ("box_place_torso_positions", [0.61, -0.81, -0.6, 0.0]),
                (
                    "box_place_torso_straighten_intermediate_positions",
                    [0.61, -1.2, -0.6, 0.0],
                ),
                ("gripper_open_position", 100.0),
                ("gripper_closed_position", 0.0),
                ("box_empty_close_ratio_threshold", 0.95),
                ("box_gripper_feedback_timeout_sec", 2.0),
                ("box_gripper_feedback_max_age_sec", 0.5),
            ],
        )

        # TF GraspBox and TF DragBox have independent per-model/per-layer
        # profiles.  Keep the legacy parameters above for the original
        # actions, while these generated declarations provide a complete,
        # explicit tuning surface for /grasp_box_tf and
        # /execute_drag_box_grasp_tf.
        self.declare_parameters(
            namespace="",
            parameters=self._tf_layer_parameter_defaults(),
        )
