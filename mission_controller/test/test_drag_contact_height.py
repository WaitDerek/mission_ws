"""Offline checks for the shared Drag contact-height offset."""
import math
from copy import deepcopy
import pytest
from geometry_msgs.msg import TransformStamped
from tf2_geometry_msgs import do_transform_pose_stamped
from mission_runtime.box_geometry import BoxGeometryMixin
from mission_runtime.mission_controller import MissionController
from test_grasp_contact_forward_scale import pose, harness
from test_tf_box_center_correction import _TargetHarness

def test_height_uses_footprint_vertical_even_for_rotated_source():
    tf=TransformStamped()
    tf.header.frame_id="base_footprint";tf.child_frame_id="frozen"
    tf.transform.rotation.x=math.sin(.4);tf.transform.rotation.w=math.cos(.4)
    h=harness(.04,tf)
    before=pose("frozen",(.2,-.3,.7))
    after=BoxGeometryMixin._raise_drag_contact_target(h,before)
    a=do_transform_pose_stamped(before,tf);b=do_transform_pose_stamped(after,tf)
    assert b.pose.position.x==pytest.approx(a.pose.position.x)
    assert b.pose.position.y==pytest.approx(a.pose.position.y)
    assert b.pose.position.z-a.pose.position.z==pytest.approx(.04)
    assert before.pose.orientation==after.pose.orientation

@pytest.mark.parametrize("k",[1.,.5])
def test_initial_and_rebuilt_contacts_inherit_four_cm_once(k):
    h=_TargetHarness();h.VALUES=deepcopy(h.VALUES)
    h.VALUES.update({"direct_movel_fixture_compensation_enabled": True,
                    "left_fixture_center_in_link8_xyz":[-.12,-.1,.04],
                    "right_fixture_center_in_link8_xyz":[-.12,.1,.04],
                    "drag_box_tf_left_contact_forward_delta_scale":k,
                    "drag_box_tf_right_contact_forward_delta_scale":k})
    box=pose("base_footprint",(.2,-.6,.7))
    old=h._make_tf_link8_target_poses(box,2,"smallbox",drag_mode=True)
    grasp=h._make_tf_link8_target_poses(box,2,"smallbox",drag_mode=False)
    h.VALUES["drag_box_tf_contact_height_offset_m"]=.04
    new=h._make_tf_link8_target_poses(box,2,"smallbox",drag_mode=True)
    for a,b in zip(old,new):
        assert b.pose.position.z-a.pose.position.z==pytest.approx(.04)
        assert b.pose.position.x==a.pose.position.x
        assert b.pose.position.y==a.pose.position.y
        assert b.pose.orientation==a.pose.orientation
    assert h._make_tf_link8_target_poses(box,2,"smallbox",drag_mode=False)==grasp
    h._last_grasp_box_tf_box_pose=box
    h._endpoint_sync_transform_to_pose=MissionController._endpoint_sync_transform_to_pose
    world=MissionController._pose_stamped_to_transform(box)
    fallback=MissionController._pose_stamped_to_transform(new[0])
    rebuilt=MissionController._drag_tf_scaled_left_join_world(h,world,fallback)
    assert rebuilt[0]==pytest.approx(fallback[0])
    # Rebuilding an initial target from the same detection never accumulates height.
    again=h._make_tf_link8_target_poses(box,2,"smallbox",drag_mode=True)
    assert again==new
