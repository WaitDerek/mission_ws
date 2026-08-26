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


class TestBoxLayerAndPostArmMoveJ(unittest.TestCase):
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
