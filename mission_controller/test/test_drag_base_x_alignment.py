"""No-hardware checks for the workflow-only DragBox X alignment."""

from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from geometry_msgs.msg import PoseStamped, TransformStamped

from mission_runtime.box_perception import BoxPerceptionMixin


class TestDragBaseXAlignment(TestCase):
    def setUp(self):
        self.pose = PoseStamped()
        self.pose.header.frame_id = "base_link"
        self.pose.header.stamp.sec = 1
        self.pose.pose.position.x = 0.140135
        self.pose.pose.orientation.w = 1.0
        transform = TransformStamped()
        transform.header.frame_id = "base_footprint"
        transform.child_frame_id = "base_link"
        transform.transform.rotation.w = 1.0

        self.node = SimpleNamespace(
            mission_lease_manager=SimpleNamespace(workflow_id="workflow-test"),
            _last_grasp_box_tf_box_pose=deepcopy(self.pose),
            tf_buffer=SimpleNamespace(lookup_transform=MagicMock(return_value=transform)),
            _float=lambda name: {
                "drag_box_tf_workflow_reference_base_x_m": 0.110135,
                "drag_box_tf_workflow_base_x_tolerance_m": 0.010,
                "grasp_box_tf_detection_tf_timeout_sec": 10.0,
            }[name],
            _publish_box_grasp_feedback=MagicMock(),
            _call_box_object_pose=MagicMock(),
        )
        self.goal = SimpleNamespace(is_cancel_requested=False)
        self.request = SimpleNamespace(dry_run=False)
        self.detection = object()

    def test_workflow_micro_uses_pos_one_and_second_detection(self):
        gateway = MagicMock()
        gateway.navigate.return_value = SimpleNamespace(success=True)
        fresh_detection = object()
        fresh_pose = deepcopy(self.pose)
        fresh_pose.pose.position.x = 0.111
        self.node._call_box_object_pose.return_value = (
            fresh_detection, fresh_pose
        )
        with patch(
            "mission_runtime.box_perception.MqttNavigationGateway",
            return_value=gateway,
        ):
            result_detection, result_pose = (
                BoxPerceptionMixin._align_workflow_drag_box_base_x(
                    self.node, self.goal, self.request,
                    self.detection, self.pose, detection_arm="left",
                )
            )
        request = gateway.navigate.call_args.args[0]
        self.assertEqual(request.pos[0], 0.0)
        self.assertAlmostEqual(request.pos[1], 0.03)
        self.assertEqual(request.pos[2], 0.0)
        self.assertIs(result_detection, fresh_detection)
        self.assertIs(result_pose, fresh_pose)
        self.assertAlmostEqual(
            self.node._last_grasp_box_tf_box_pose.pose.position.x, 0.140135
        )
        self.node._call_box_object_pose.assert_called_once_with(
            self.goal, self.request, tf_mode=True, drag_mode=True,
            detection_arm="left",
        )
        gateway.close.assert_called_once()

    def test_standalone_drag_does_not_micro_navigate(self):
        self.node.mission_lease_manager.workflow_id = ""
        with patch("mission_runtime.box_perception.MqttNavigationGateway") as factory:
            result_detection, result_pose = (
                BoxPerceptionMixin._align_workflow_drag_box_base_x(
                    self.node, self.goal, self.request,
                    self.detection, self.pose, detection_arm="left",
                )
            )
        self.assertIs(result_detection, self.detection)
        self.assertIs(result_pose, self.pose)
        factory.assert_not_called()
        self.node._call_box_object_pose.assert_not_called()

    def test_within_one_centimeter_does_not_micro_navigate(self):
        self.node._last_grasp_box_tf_box_pose.pose.position.x = 0.115
        with patch("mission_runtime.box_perception.MqttNavigationGateway") as factory:
            result_detection, result_pose = (
                BoxPerceptionMixin._align_workflow_drag_box_base_x(
                    self.node, self.goal, self.request,
                    self.detection, self.pose, detection_arm="left",
                )
            )
        self.assertIs(result_detection, self.detection)
        self.assertIs(result_pose, self.pose)
        factory.assert_not_called()
        self.node._call_box_object_pose.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
