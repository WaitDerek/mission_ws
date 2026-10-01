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
        # GraspBox uses the right wrist camera.  Keep its calibrated detection
        # poses independent from the generic/DragBox profiles.
        grasp_right_detection_overrides = {
            ("smallbox", 1): [
                -43681,
                64069,
                -32832,
                -82829,
                52815,
                -5888,
                -85689,
            ],
            ("smallbox", 2): [
                -43666,
                34670,
                20100,
                -82382,
                11944,
                37569,
                -34208,
            ],
            ("smallbox", 3): [
                -7650,
                14669,
                8212,
                -6785,
                -28607,
                10575,
                -76223,
            ],
            ("smallbox", 4): [
                -5483,
                9744,
                12854,
                -2708,
                -19532,
                1808,
                -70085,
            ],
        }
        # DragBox TF uses the right camera for detection.  Keep the left arm
        # at its configured layer-specific clearance posture during imaging.
        # The values are controller joint units (1000 units = 1 degree).
        drag_left_detection = {
            "bigbox": [
                [57192, -409, -8583, -95717, 1770, -12286, 62542],
                [57192, -409, -8583, -95717, 1770, -12286, 62542],
                [28101, 10100, -5186, -67156, -3504, -12579, 31431],
                [-11175, 19393, -96084, 17486, 96830, -4823, 36800],
            ],
            # Layers 1-2 share the measured left-arm standby posture while
            # the right wrist camera observes either box size.
            "smallbox": [
                [57192, -409, -8583, -95717, 1770, -12286, 62542],
                [57192, -409, -8583, -95717, 1770, -12286, 62542],
                *detection["smallbox"][2:],
            ],
        }
        post_detection_left = {
            model: [
                [-171982, -204, 93820, 89651, 4401, 5999, -4935]
                for _layer in range(1, 5)
            ]
            for model in ("bigbox", "smallbox")
        }
        # Big-box DragBox joins use a negative-Joint4 IK branch.  Retaining a
        # positive-Joint4 avoidance posture forced another sign crossing after
        # Drag3.  Keep the other six calibrated avoidance axes independent.
        for layer_profile in post_detection_left["bigbox"]:
            layer_profile[3] = -25000
        for layer_profile in post_detection_left["smallbox"]:
            layer_profile[3] = -25000
        angles = {
            "grasp_box_tf": {
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
            },
            "drag_box_tf": {
                "bigbox": {
                    1: (-13.0, 0.0, 0.0),
                    2: (-45.0, -85.0, -55.0),
                    3: (-60.0, -115.0, -70.0),
                    4: (-89.0, -149.0, -89.0),
                },
                "smallbox": {
                    1: (-13.0, 0.0, 0.0),
                    2: (-45.0, -85.0, -70.0),
                    3: (-70.0, -120.0, -73.0),
                    4: (-89.0, -149.0, -89.0),
                },
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
            0.0,
            0.0,
            0.0,
            -0.058164,
            -0.006476,
            0.081596,
            0.994946,
        ]
        right_correction = [
            0.0,
            0.0,
            0.0,
            0.012614,
            -0.032172,
            0.081927,
            0.996039,
        ]
        # Absolute left/right corrections measured with the interactive
        # GraspBox TF calibrator. Keep every smallbox layer independent.
        grasp_smallbox_corrections = {
            1: (
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.024384405,
                    -0.023875088,
                    -0.037291703,
                    0.998721538,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.051357703,
                    0.004360193,
                    0.012754161,
                    0.998589358,
                ],
            ),
            2: (
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.016304739,
                    -0.002712095,
                    0.007963239,
                    0.999831679,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.039304169,
                    0.050625186,
                    0.030148091,
                    0.997488529,
                ],
            ),
            3: (
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.009310303,
                    -0.008333252,
                    0.003572466,
                    0.999915553,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.038994183,
                    0.003851522,
                    0.040366668,
                    0.998416322,
                ],
            ),
            4: (
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.005119032,
                    -0.013035214,
                    0.004660018,
                    0.999891076,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.044762070,
                    0.002591940,
                    0.031502585,
                    0.998497484,
                ],
            ),
        }
        # Absolute left-side corrections measured after Drag3 re-anchoring.
        # Keep every bigbox layer independent from the initial right target.
        drag_bigbox_left_corrections = {
            1: [
                0.0,
                0.0,
                0.0,
                -0.035042439,
                -0.034710191,
                0.024586835,
                0.998480204,
            ],
            2: [
                0.0,
                0.0,
                0.0,
                -0.054370455,
                0.016881237,
                0.023892927,
                0.998092183,
            ],
            3: [
                0.0,
                0.0,
                0.0,
                -0.054209689,
                -0.006481642,
                0.038319026,
                0.997772995,
            ],
            4: [
                0.0,
                0.0,
                0.0,
                -0.040472304,
                -0.027567708,
                -0.003524540,
                0.998794069,
            ],
        }
        # Absolute right-side corrections measured with the interactive
        # DragBox TF calibrator. Keep every bigbox layer independent.
        drag_bigbox_right_corrections = {
            1: [
                0.0,
                0.0,
                0.0,
                -0.078806036,
                0.031130706,
                0.078378699,
                0.993316298,
            ],
            2: [
                0.0,
                0.0,
                0.0,
                -0.064522892,
                0.019208153,
                0.050748447,
                0.996439882,
            ],
            3: [
                0.0,
                0.0,
                0.0,
                -0.081903855,
                0.048541701,
                0.073411578,
                0.992746797,
            ],
            4: [
                0.0,
                0.0,
                0.0,
                -0.059377345,
                0.012520900,
                0.047450144,
                0.997028606,
            ],
        }
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
                1: [0.14, 0.0, 0.0],
                2: [0.0, 0.20, 0.0],
                3: [-0.14, 0.0, 0.0],
            },
            "right": {
                1: [0.14, 0.0, 0.0],
                2: [0.0, 0.20, 0.0],
                3: [-0.14, 0.0, 0.0],
            },
        }
        parameters = []
        for action_prefix in ("grasp_box_tf", "drag_box_tf"):
            for model in ("bigbox", "smallbox"):
                for layer in range(1, 5):
                    layer_left_correction = left_correction
                    layer_right_correction = right_correction
                    if action_prefix == "grasp_box_tf" and model == "smallbox":
                        (
                            layer_left_correction,
                            layer_right_correction,
                        ) = grasp_smallbox_corrections[layer]
                    elif action_prefix == "drag_box_tf" and model == "bigbox":
                        layer_left_correction = drag_bigbox_left_corrections[layer]
                        layer_right_correction = drag_bigbox_right_corrections[layer]
                    for arm in ("left", "right"):
                        if action_prefix == "drag_box_tf" and arm == "left":
                            profile = drag_left_detection[model][layer - 1]
                        elif action_prefix == "grasp_box_tf" and arm == "right":
                            profile = grasp_right_detection_overrides.get(
                                (model, layer), detection[model][layer - 1]
                            )
                        else:
                            profile = detection[model][layer - 1]
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
                    for joint_index, angle in enumerate(
                        angles[action_prefix][model][layer], start=1
                    ):
                        parameters.append(
                            (
                                f"{action_prefix}_box_layer_joint{joint_index}_"
                                f"approach_angle_deg_{model}_layer{layer}",
                                float(angle),
                            )
                        )
                    left_offset = list(offsets[model][layer][0])
                    right_offset = list(offsets[model][layer][1])
                    if action_prefix == "grasp_box_tf" and model == "smallbox":
                        left_offset = [0.0, 0.0, -0.54]
                    elif action_prefix == "drag_box_tf" and model == "bigbox":
                        left_offset = [0.0, 0.0, -0.54]
                        right_offset = [0.0, 0.0, 0.54]
                    parameters.extend(
                        [
                            (
                                f"{action_prefix}_direct_movel_left_offset_xyz_"
                                f"{model}_layer{layer}",
                                left_offset,
                            ),
                            (
                                f"{action_prefix}_direct_movel_right_offset_xyz_"
                                f"{model}_layer{layer}",
                                right_offset,
                            ),
                            (
                                f"{action_prefix}_joint123_left_target_correction_pose_box_"
                                f"{model}_layer{layer}",
                                [0.0, 0.0, 0.0, *layer_left_correction[3:]],
                            ),
                            (
                                f"{action_prefix}_joint123_right_target_correction_pose_box_"
                                f"{model}_layer{layer}",
                                [0.0, 0.0, 0.0, *layer_right_correction[3:]],
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
                    # The waist MoveJ speed is independently tunable for
                    # every action, box model, and layer.  Layer 1 is
                    # intentionally slower for initial commissioning; the
                    # remaining defaults preserve the existing action-wide
                    # speeds (GraspBox 12, DragBox 20).
                    parameters.append(
                        (
                            f"{action_prefix}_body_home_carry_body_velocity_"
                            f"{model}_layer{layer}",
                            3 if layer == 1 else (12 if action_prefix == "grasp_box_tf" else 20),
                        )
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
                                drag_step = drag_steps[arm][drag_index]
                                parameters.append(
                                    (
                                        f"drag_box_tf_post_movel_step_drag{drag_index}_"
                                        f"{arm}_xyz_{model}_layer{layer}",
                                        list(drag_step),
                                    )
                                )
        return parameters
