import math
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_geometry_msgs import do_transform_pose_stamped

from mission_runtime.box_geometry import BoxGeometryMixin
from mission_runtime.common import MissionError, rotate_vector
from test_tf_box_center_correction import _TargetHarness


def pose(frame, xyz):
    result = PoseStamped()
    result.header.frame_id = frame
    result.header.stamp.sec = 123
    result.pose.position.x, result.pose.position.y, result.pose.position.z = map(float, xyz)
    result.pose.orientation.w = 1.0
    return result


def harness(scale, transform=None):
    return SimpleNamespace(
        _float=lambda name: 5.0 if "timeout" in name else scale,
        tf_buffer=SimpleNamespace(lookup_transform=Mock(return_value=transform)),
    )


def apply(h, contact, box):
    return BoxGeometryMixin._scale_grasp_contact_forward_delta(h, contact, box, "left")


def test_identity_is_exact_and_needs_no_tf():
    h = harness(1.0)
    contact = pose("base_link", (0.8, 0.3, 0.7))
    assert apply(h, contact, pose("base_link", (0.5, 0, 0.7))) is contact
    h.tf_buffer.lookup_transform.assert_not_called()


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.5])
def test_only_forward_delta_changes(scale):
    h = harness(scale)
    contact = pose("base_footprint", (0.8, -0.3, 0.7))
    result = apply(h, contact, pose("base_footprint", (0.5, 0, 0.6)))
    assert result.pose.position.x == contact.pose.position.x
    assert result.pose.position.y == pytest.approx(scale * -0.3)
    assert result.pose.position.z == contact.pose.position.z
    assert result.pose.orientation == contact.pose.orientation
    assert result.header == contact.header
    assert contact.pose.position.x == 0.8


def test_rotated_frozen_frame_preserves_footprint_x_z_and_orientation():
    tf = TransformStamped()
    tf.header.frame_id = "base_footprint"
    tf.child_frame_id = "frozen"
    tf.transform.translation.x = 2.0
    tf.transform.translation.y = -0.4
    tf.transform.translation.z = 0.8
    tf.transform.rotation.x = math.sin(0.4)
    tf.transform.rotation.w = math.cos(0.4)
    h = harness(0.5, tf)
    contact = pose("frozen", (0.8, -0.3, 0.7))
    box = pose("frozen", (0.5, 0.0, 0.6))
    before = do_transform_pose_stamped(contact, tf)
    box_foot = do_transform_pose_stamped(box, tf)
    result = apply(h, contact, box)
    after = do_transform_pose_stamped(result, tf)
    assert after.pose.position.y == pytest.approx(
        box_foot.pose.position.y + 0.5 * (before.pose.position.y - box_foot.pose.position.y)
    )
    assert after.pose.position.x == pytest.approx(before.pose.position.x)
    assert after.pose.position.z == pytest.approx(before.pose.position.z)
    assert result.pose.orientation == contact.pose.orientation
    args = h.tf_buffer.lookup_transform.call_args.args
    assert args[:2] == ("base_footprint", "frozen")
    assert args[2].nanoseconds == 123000000000


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_invalid_scale_rejected(scale):
    with pytest.raises(MissionError):
        apply(harness(scale), pose("base_footprint", (1, 0, 0)), pose("base_footprint", (0, 0, 0)))


@pytest.mark.parametrize("drag_mode", [False, True])
def test_scaling_precedes_fixture_compensation_and_action_parameters_are_independent(drag_mode):
    h = _TargetHarness()
    h.VALUES = deepcopy(h.VALUES)
    h.VALUES.update({
        "direct_movel_fixture_compensation_enabled": True,
        "left_fixture_center_in_link8_xyz": [-0.12, -0.1, 0.04],
        "right_fixture_center_in_link8_xyz": [-0.12, 0.1, 0.04],
    })
    box = pose("base_footprint", (0.6, 0.1, 0.7))
    box.pose.orientation.x = math.sin(0.3)
    box.pose.orientation.w = math.cos(0.3)
    baseline = h._make_tf_link8_target_poses(box, 2, "smallbox", drag_mode=drag_mode)
    other_baseline = h._make_tf_link8_target_poses(box, 2, "smallbox", drag_mode=not drag_mode)
    prefix = "drag_box_tf" if drag_mode else "grasp_box_tf"
    h.VALUES[f"{prefix}_left_contact_forward_delta_scale"] = 0.5
    h.VALUES[f"{prefix}_right_contact_forward_delta_scale"] = 0.0
    scaled = h._make_tf_link8_target_poses(box, 2, "smallbox", drag_mode=drag_mode)
    for arm, old, new, k in zip(("left", "right"), baseline, scaled, (0.5, 0.0)):
        q = old.pose.orientation
        fixture = rotate_vector(tuple(h.VALUES[f"{arm}_fixture_center_in_link8_xyz"]), (q.x, q.y, q.z, q.w))
        old_contact_y = old.pose.position.y + fixture[1]
        new_contact_y = new.pose.position.y + fixture[1]
        assert new_contact_y == pytest.approx(0.1 + k * (old_contact_y - 0.1))
        assert new.pose.position.x == old.pose.position.x
        assert new.pose.position.z == old.pose.position.z
        assert new.pose.orientation == old.pose.orientation
    assert h._make_tf_link8_target_poses(box, 2, "smallbox", drag_mode=not drag_mode) == other_baseline


def test_recorded_box_pose_keeps_left_contact_ahead_without_narrowing_span():
    h = _TargetHarness()
    h.VALUES = deepcopy(h.VALUES)
    h.VALUES["drag_box_tf_left_contact_forward_delta_scale"] = 0.5
    h.VALUES["drag_box_tf_right_contact_forward_delta_scale"] = 0.5
    # Exact detection quaternion from the 2026-09-30 11:51 DragBox action.
    box = pose("base_footprint", (0.128076708, -0.598076125, 0.909653154))
    q = (-0.006560350, -0.701233752, -0.017833001, 0.712678168)
    box.pose.orientation.x, box.pose.orientation.y = q[:2]
    box.pose.orientation.z, box.pose.orientation.w = q[2:]
    h.VALUES["drag_box_tf_direct_movel_left_offset_xyz_smallbox_layer2"] = [0., 0., -0.54]
    h.VALUES["drag_box_tf_direct_movel_right_offset_xyz_smallbox_layer2"] = [0., 0., 0.54]
    left, right = h._make_tf_link8_target_poses(box, 2, "smallbox", drag_mode=True)
    assert right.pose.position.y - left.pose.position.y == pytest.approx(0.018554962, abs=1e-8)
    assert left.pose.position.x - right.pose.position.x == pytest.approx(1.079215909, abs=1e-8)


def reanchor_harness():
    from test_drag_box_tf_action import _DragTfReanchorHarness
    h = _DragTfReanchorHarness()
    h.values.update({
        "drag_box_tf_left_contact_forward_delta_scale": 0.5,
        "grasp_box_tf_freeze_frame": "base_footprint",
        "direct_movel_target_mode": "camera_offset_box_orientation",
        "direct_movel_fixture_compensation_enabled": False,
    })
    h._last_grasp_box_tf_box_pose.header.frame_id = "base_footprint"
    h._last_drag_box_tf_unscaled_contact_relations = {
        "left": ((-0.5, 0.4, 0.0), (0.0, 0.0, 0.0, 1.0))
    }
    h._last_grasp_box_tf_box_to_link7_targets["left"] = (
        (-0.5, 0.2, 0.0), (0.0, 0.0, 0.0, 1.0)
    )
    return h


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2])
def test_drag3_reanchor_scales_raw_contact_once_in_current_footprint(yaw):
    from mission_runtime.mission_controller import MissionController
    h = reanchor_harness()
    MissionController._capture_drag_tf_right_grasp_relation(h)
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    moved_box = ((2.0, 2.0, 3.0), q)
    h.transforms["right_link"] = MissionController._compose_transform(
        moved_box, h._last_drag_box_tf_right_grasp_relation
    )
    target, _ = MissionController._reanchor_drag_tf_left_join_after_drag3(h)
    raw_delta = rotate_vector((-0.5, 0.4, 0.0), q)
    assert target.position.x == pytest.approx(2.0 + raw_delta[0])
    assert target.position.y == pytest.approx(2.0 + 0.5 * raw_delta[1])
    assert target.position.z == pytest.approx(3.0)
    relation = h._last_grasp_box_tf_box_to_link7_targets["left"]
    rebuilt = MissionController._compose_transform(moved_box, relation)
    assert rebuilt[0] == pytest.approx((target.position.x, target.position.y, target.position.z))


def test_predicted_drag_join_uses_same_single_scaling_as_runtime():
    from mission_runtime.mission_controller import MissionController
    h = reanchor_harness()
    h.values.update({
        "drag_box_tf_post_movel_step_drag1_right_xyz_bigbox_layer1": [0.1, 0., 0.],
        "drag_box_tf_post_movel_step_drag2_right_xyz_bigbox_layer1": [0., 0.2, 0.],
        "drag_box_tf_post_movel_step_drag3_right_xyz_bigbox_layer1": [0., 0., 0.],
    })
    h._equalize_tf_dual_target_z = lambda left, right, reference: (
        MissionController._equalize_tf_dual_target_z(h, left, right, reference=reference)
    )
    identity = ((0., 0., 0.), (0., 0., 0., 1.))
    result = MissionController._predict_drag_tf_left_join_target(
        h, ((1., 2., 3.), identity[1]),
        ((0.5, 2.2, 3.), identity[1]), ((1.5, 2., 3.), identity[1]),
        identity, identity, box_layer=1, model_label="bigbox"
    )
    assert result.position.x == pytest.approx(0.6)
    assert result.position.y == pytest.approx(2.4)
    assert result.position.z == pytest.approx(3.0)
