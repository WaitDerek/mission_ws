"""Generated TF grasp/drag parameter profile names and defaults."""


class TfLayerProfilesMixin:
    """Generate model/layer-specific parameter defaults."""

    @staticmethod
    def _tf_layer_parameter_defaults():
        """Return independent TF-action/model/layer defaults.

        Values intentionally mirror the currently deployed bigbox/smallbox
        profiles.  Each layer receives its own scalar/vector parameters so a
        later calibration change cannot affect another layer or the other TF
        action.
        """
        detection = {
            "bigbox": [
                [144725, -5335, 7032, 9843, 7540, -5611, 85414],
                [170647, 3018, 18744, 95121, -1950, -8903, 30524],
                [-39570, 16276, -17721, -95245, 14032, -14558, -23838],
                [21382, -4978, -274, 1282, -5472, 3628, -32636],
            ],
            "smallbox": [
                [172102, -3751, 14348, 95105, 7730, -5615, 31841],
                [-20762, 1539, -12837, -102889, 1366, -8774, 3529],
                [6894, -3485, -20092, -19433, 15367, -7336, -41454],
                [2341, 248, -3624, -16956, 2290, 10183, -45681],
            ],
        }
        # DragBox TF uses the left camera for detection.  Keep its calibrated
        # bigbox poses independent from the GraspBox/right-camera profiles.
        # The values are controller joint units (1000 units = 1 degree).
        drag_left_detection = {
            "bigbox": [
                [-172278, 7319, 20124, 51808, -17856, -23263, -70442],
                [-161722, 5480, 933, 104186, 5631, -2342, -2903],
                [18442, 3544, 2870, -104500, -2568, 5250, -24076],
                [-19238, 3482, 215, -91196, -2634, 5268, -78519],
            ],
            # Until separately calibrated, retain the existing left/smallbox
            # defaults rather than coupling them to the bigbox calibration.
            "smallbox": detection["smallbox"],
        }
        post_detection_left = {
            model: [
                [-171982, -204, 93820, 89651, 4401, 5999, -4935]
                for _layer in range(1, 5)
            ]
            for model in ("bigbox", "smallbox")
        }
        angles = {
            "bigbox": {
                1: (-13.0, 0.0, 0.0),
                2: (-45.0, -85.0, -55.0),
                3: (-70.0, -120.0, -73.0),
                4: (-89.0, -149.0, -89.0),
            },
            "smallbox": {
                1: (-13.0, 0.0, 0.0),
                2: (-45.0, -85.0, -70.0),
                3: (-70.0, -120.0, -73.0),
                4: (-89.0, -149.0, -89.0),
            },
        }
        offsets = {
            "bigbox": {
                layer: ([0.0, 0.0, -0.5], [0.0, 0.0, 0.5]) for layer in range(1, 5)
            },
            "smallbox": {
                layer: (
                    (
                        [0.0, -0.025, -0.5],
                        [0.0, -0.025, 0.5],
                    )
                    if layer == 4
                    else ([0.0, 0.0, -0.5], [0.0, 0.0, 0.5])
                )
                for layer in range(1, 5)
            },
        }
        left_correction = [
            0.064762,
            -0.049358,
            0.060595,
            -0.058164,
            -0.006476,
            0.081596,
            0.994946,
        ]
        right_correction = [
            0.081444,
            -0.049338,
            -0.020083,
            0.012614,
            -0.032172,
            0.081927,
            0.996039,
        ]
        standard_steps = {
            "left": {
                1: [0.0, 0.0, 0.025],
                2: [0.14, 0.0, 0.0],
                3: [-0.14, 0.0, 0.0],
                4: [0.0, 0.0, -0.1],
                5: [0.0, 0.0, 0.0],
            },
            "right": {
                1: [0.0, 0.0, -0.028],
                2: [0.14, 0.0, 0.0],
                3: [-0.14, 0.0, 0.0],
                4: [0.0, 0.0, 0.1],
                5: [0.0, 0.0, 0.0],
            },
        }
        smallbox_step1 = {
            "left": [0.0, 0.0, 0.03],
            "right": [0.0, 0.0, -0.02],
        }
        drag_steps = {
            "left": {
                1: [0.0, 0.0, 0.0],
                2: [0.0, 0.0, 0.2],
                3: [0.0, 0.0, 0.0],
            },
            "right": {
                1: [0.14, 0.0, 0.0],
                2: [0.0, 0.0, 0.2],
                3: [-0.14, 0.0, 0.0],
            },
        }
        parameters = []
        for action_prefix in ("grasp_box_tf", "drag_box_tf"):
            for model in ("bigbox", "smallbox"):
                for layer in range(1, 5):
                    for arm in ("left", "right"):
                        profile = (
                            drag_left_detection[model][layer - 1]
                            if action_prefix == "drag_box_tf" and arm == "left"
                            else detection[model][layer - 1]
                        )
                        parameters.append(
                            (
                                f"{action_prefix}_box_layer_pre_detection_{arm}_movej_joint_units_"
                                f"{model}_layer{layer}",
                                list(profile),
                            )
                        )
                    if action_prefix == "drag_box_tf":
                        parameters.append(
                            (
                                f"drag_box_tf_box_layer_post_detection_left_movej_joint_units_"
                                f"{model}_layer{layer}",
                                list(post_detection_left[model][layer - 1]),
                            )
                        )
                    for joint_index, angle in enumerate(angles[model][layer], start=1):
                        parameters.append(
                            (
                                f"{action_prefix}_box_layer_joint{joint_index}_"
                                f"approach_angle_deg_{model}_layer{layer}",
                                float(angle),
                            )
                        )
                    parameters.extend(
                        [
                            (
                                f"{action_prefix}_direct_movel_left_offset_xyz_"
                                f"{model}_layer{layer}",
                                list(offsets[model][layer][0]),
                            ),
                            (
                                f"{action_prefix}_direct_movel_right_offset_xyz_"
                                f"{model}_layer{layer}",
                                list(offsets[model][layer][1]),
                            ),
                            (
                                f"{action_prefix}_joint123_left_target_correction_pose_box_"
                                f"{model}_layer{layer}",
                                list(left_correction),
                            ),
                            (
                                f"{action_prefix}_joint123_right_target_correction_pose_box_"
                                f"{model}_layer{layer}",
                                list(right_correction),
                            ),
                        ]
                    )
                    # TF waist-carry arm speeds are independently tunable
                    # for each action, box model, and layer.  Initialize every
                    # profile from the current unified 12% defaults while
                    # keeping the legacy action-wide parameters as fallback
                    # for callers that do not provide a model/layer.
                    parameters.extend(
                        [
                            (
                                f"{action_prefix}_body_home_carry_left_movel_velocity_percent_"
                                f"{model}_layer{layer}",
                                12.0,
                            ),
                            (
                                f"{action_prefix}_body_home_carry_right_movel_velocity_percent_"
                                f"{model}_layer{layer}",
                                12.0,
                            ),
                        ]
                    )
                    for arm in ("left", "right"):
                        for step in range(1, 6):
                            delta = standard_steps[arm][step]
                            if model == "smallbox" and step == 1:
                                delta = smallbox_step1[arm]
                            parameters.append(
                                (
                                    f"{action_prefix}_post_movel_{arm}_step{step}_xyz_"
                                    f"{model}_layer{layer}",
                                    list(delta),
                                )
                            )
                    if action_prefix == "drag_box_tf":
                        for arm in ("left", "right"):
                            for drag_index in range(1, 4):
                                parameters.append(
                                    (
                                        f"drag_box_tf_post_movel_step_drag{drag_index}_"
                                        f"{arm}_xyz_{model}_layer{layer}",
                                        list(drag_steps[arm][drag_index]),
                                    )
                                )
        return parameters
