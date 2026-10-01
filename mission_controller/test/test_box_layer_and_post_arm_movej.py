import json
import threading
import time
import unittest

from mission_runtime.common import MissionError
from mission_runtime.mission_controller import MissionController


class _LayerHarness:
    def _float_array(self, name):
        return {
            "box_layer_joint1_approach_angles_deg": [-13.0, -45.0, -70.0, -89.0],
            "box_layer_joint2_approach_angles_deg": [0.0, -85.0, -120.0, -149.0],
            "box_layer_joint3_approach_angles_deg": [0.0, -55.0, -78.0, -89.0],
        }[name]

    def _boolean_array(self, name):
        return {
            "box_layer_joint123_configured": [True, True, True, True],
            "box_layer_pre_detection_right_movej_configured": [
                True, True, True, True
            ],
        }[name]


class _ImmediateFuture:
    def __init__(self, response):
        self.response = response


class _CommandClient:
    def __init__(self, harness):
        self.harness = harness
        self.requests = []

    def call_async(self, request):
        self.requests.append(request)
        sequence = len(self.requests)
        now = time.monotonic()
        with self.harness.joint_state_lock:
            self.harness.latest_slave_arm_state_sequences = {
                "left": sequence,
                "right": sequence,
            }
            self.harness.latest_slave_arm_positions = {
                "left": list(self.harness.left_target),
                "right": list(self.harness.right_target),
            }
            self.harness.latest_slave_arm_velocities = {
                "left": [0.0] * 7,
                "right": [0.0] * 7,
            }
            self.harness.latest_slave_arm_state_times = {
                "left": now,
                "right": now,
            }
        response = type(
            "Response",
            (),
            {"data": json.dumps({"receive_state": True})},
        )()
        return _ImmediateFuture(response)


class _PostArmHarness:
    def __init__(self):
        self.values = {
            "box_post_arm_movej_enabled": True,
            "box_post_arm_movej_left_device": 0,
            "box_post_arm_movej_right_device": 1,
            "box_post_arm_movej_velocity": 5,
            "box_post_arm_movej_blend_radius": 0,
            "box_post_arm_movej_trajectory_connect": 0,
            "box_post_arm_movej_timeout_sec": 2.0,
            "box_post_arm_position_tolerance_rad": 0.01,
            "box_post_arm_velocity_tolerance_rad_sec": 0.01,
            "box_post_arm_feedback_max_age_sec": 1.0,
            "box_post_arm_stable_samples": 1,
            "box_post_arm_movej_command_units_per_degree": 1000.0,
            "box_post_arm_movej_left_joint_units": [0] * 7,
            "box_post_arm_movej_right_joint_units": [0] * 7,
            "dependency_wait_timeout_sec": 1.0,
        }
        self.queried = []
        self.joint_state_lock = threading.Lock()
        self.latest_slave_arm_state_sequences = {"left": 0, "right": 0}
        self.latest_slave_arm_positions = {"left": [], "right": []}
        self.latest_slave_arm_velocities = {"left": [], "right": []}
        self.latest_slave_arm_state_times = {"left": 0.0, "right": 0.0}
        self.left_target = [0.0] * 7
        self.right_target = [0.0] * 7
        self.body_command_client = _CommandClient(self)
        self.feedback = []

    def _boolean(self, name):
        return bool(self.values[name])

    def _integer(self, name):
        self.queried.append(name)
        return int(self.values[name])

    def _float(self, name):
        self.queried.append(name)
        return float(self.values[name])

    def _float_array(self, name):
        self.queried.append(name)
        return list(self.values[name])

    def _string(self, name):
        self.queried.append(name)
        return "/robot/command"

    def _post_arm_movej_targets(self):
        left_units = list(self.values["box_post_arm_movej_left_joint_units"])
        right_units = list(self.values["box_post_arm_movej_right_joint_units"])
        self.left_target = [0.0] * 7
        self.right_target = [0.0] * 7
        return left_units, right_units, self.left_target, self.right_target

    def _publish_box_grasp_feedback(self, _goal_handle, stage, _detail):
        self.feedback.append(stage)

    def _wait_for_service(self, *_args):
        return None

    def _wait_future(self, future, *_args, **_kwargs):
        return future.response

    def _check_canceled(self, *_args):
        return None

    def _wait_for_post_arm_joint_targets(self, *args, **kwargs):
        return MissionController._wait_for_post_arm_joint_targets(
            self, *args, **kwargs
        )

    @staticmethod
    def _parse_string_command_response(response, description):
        return MissionController._parse_string_command_response(
            response, description
        )


class _Goal:
    is_cancel_requested = False


class _DetectionSdkAdapter:
    def __init__(self):
        self.calls = []

    def execute_single_movej(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "sdk movej ok"


class _DetectionOrderHarness:
    def __init__(self):
        self.intermediate_calls = []
        self.feedback = []
        self.direct_sdk_adapter = _DetectionSdkAdapter()

    def _boolean(self, name):
        assert name == "box_pre_detection_right_movej_enabled"
        return True

    def _float(self, name):
        return {
            "box_pre_detection_right_movej_command_units_per_degree": 1000.0,
            "box_pre_detection_right_movej_timeout_sec": 40.0,
        }[name]

    def _integer(self, name):
        return {
            "box_pre_detection_right_movej_device": 1,
            "box_pre_detection_right_movej_velocity": 15,
            "box_pre_detection_right_movej_blend_radius": 0,
            "box_pre_detection_right_movej_trajectory_connect": 0,
        }[name]

    def _box_layer_pre_detection_arm_movej_joint_units(self, *args, **kwargs):
        del args, kwargs
        return [1000, 2000, 3000, 4000, 5000, 6000, 7000]

    def _execute_pre_detection_arm_intermediate_movej(self, *args, **kwargs):
        self.intermediate_calls.append((args, kwargs))
        return "joint2 stage"

    def _publish_box_grasp_feedback(self, _goal_handle, stage, detail):
        self.feedback.append((stage, detail))


class TestBoxLayerAndPostArmMoveJ(unittest.TestCase):
    def test_grasp_tf_right_detection_moves_joint2_first_with_sdk(self):
        harness = _DetectionOrderHarness()
        detail = MissionController._execute_pre_detection_arm_movej_fixed(
            harness,
            _Goal(),
            True,
            2,
            "smallbox",
            arm="right",
            tf_mode=True,
            drag_mode=False,
        )

        self.assertEqual(len(harness.intermediate_calls), 1)
        args, kwargs = harness.intermediate_calls[0]
        self.assertEqual(args[3], "right")
        self.assertEqual(kwargs["target_joint_indices"], (1,))
        self.assertTrue(kwargs["use_python_sdk"])
        self.assertIn("command_backend=python_sdk", detail)

    def test_detection_sdk_movej_converts_joint_units_to_degrees(self):
        harness = _DetectionOrderHarness()
        result = MissionController._execute_pre_detection_arm_sdk_movej(
            harness,
            _Goal(),
            "right",
            [1000, 2000, 3000, 4000, 5000, 6000, 7000],
            "box_pre_detection_right_movej",
            "test detection MoveJ",
        )

        self.assertEqual(result, "sdk movej ok")
        self.assertEqual(len(harness.direct_sdk_adapter.calls), 1)
        args, kwargs = harness.direct_sdk_adapter.calls[0]
        self.assertEqual(args[0], "right")
        self.assertEqual(args[1], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(args[2:5], (15, 0, 0))
        self.assertEqual(kwargs["timeout_sec"], 40.0)

    def test_layer_detection_pose_selects_second_layer_units(self):
        harness = _LayerHarness()
        harness._float_array = lambda name: {
            "box_layer_pre_detection_right_movej_joint_units": [
                144725, -5335, 7032, 9843, 7540, -5611, 85414,
                -12083, 5105, -17961, -50575, 9150, -5641, -66298,
                144725, -5335, 7032, 9843, 7540, -5611, 85414,
                144725, -5335, 7032, 9843, 7540, -5611, 85414,
            ]
        }[name]
        self.assertEqual(
            MissionController._box_layer_pre_detection_right_movej_joint_units(
                harness, 2
            ),
            [-12083, 5105, -17961, -50575, 9150, -5641, -66298],
        )
        self.assertEqual(
            MissionController._box_layer_pre_detection_right_movej_joint_units(
                harness, 3
            ),
            [144725, -5335, 7032, 9843, 7540, -5611, 85414],
        )

    def test_box_layers_use_configured_joint123_angles(self):
        harness = _LayerHarness()
        self.assertEqual(
            MissionController._box_layer_joint123_approach_angles_deg(harness, 1),
            (-13.0, 0.0, 0.0),
        )
        self.assertEqual(
            MissionController._box_layer_joint123_approach_angles_deg(harness, 2),
            (-45.0, -85.0, -55.0),
        )
        self.assertEqual(
            MissionController._box_layer_joint123_approach_angles_deg(harness, 3),
            (-70.0, -120.0, -78.0),
        )
        self.assertEqual(
            MissionController._box_layer_joint123_approach_angles_deg(harness, 4),
            (-89.0, -149.0, -89.0),
        )
        with self.assertRaises(MissionError):
            MissionController._box_layer_joint123_approach_angles_deg(harness, 0)

    def test_post_arm_movej_confirms_success_with_movej_timeout_parameter(self):
        harness = _PostArmHarness()
        detail = MissionController._execute_post_arm_movej(
            harness, _Goal(), False
        )
        self.assertIn("arm_feedback=confirmed", detail)
        self.assertIn("box_post_arm_movej_timeout_sec", harness.queried)
        self.assertNotIn("box_post_arm_timeout_sec", harness.queried)
        self.assertEqual(len(harness.body_command_client.requests), 2)
        for request in harness.body_command_client.requests:
            payload = json.loads(request.data.strip())["payload"]
            self.assertEqual(payload["command"], "movej")


if __name__ == "__main__":
    unittest.main()
