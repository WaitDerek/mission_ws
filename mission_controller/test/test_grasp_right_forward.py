from copy import deepcopy
import math
import pytest
from geometry_msgs.msg import TransformStamped
from tf2_geometry_msgs import do_transform_pose_stamped
from mission_runtime.box_geometry import BoxGeometryMixin
from test_tf_box_center_correction import _TargetHarness
from test_grasp_contact_forward_scale import pose, harness

@pytest.mark.parametrize('box_type',['smallbox','bigbox'])
@pytest.mark.parametrize('layer,distance',[(1,0.),(2,0.),(3,.012),(4,.02)])
def test_layer_offset_only_changes_grasp_right_forward(box_type,layer,distance):
    h=_TargetHarness();h.VALUES=deepcopy(h.VALUES)
    for k,v in list(h.VALUES.items()):
        if k.endswith('_smallbox_layer2'):
            h.VALUES[k.replace('_smallbox_layer2',f'_{box_type}_layer{layer}')]=deepcopy(v)
    box=pose('base_footprint',(.2,-.6,.7))
    box.pose.orientation.z=math.sin(.3);box.pose.orientation.w=math.cos(.3)
    old=h._make_tf_link8_target_poses(box,layer,box_type)
    drag=h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)
    h.VALUES[f'grasp_box_tf_right_forward_offset_m_layer{layer}']=distance
    new=h._make_tf_link8_target_poses(box,layer,box_type)
    assert new[0]==old[0]
    assert new[1].pose.position.y==pytest.approx(old[1].pose.position.y-distance)
    assert new[1].pose.position.x==pytest.approx(old[1].pose.position.x)
    assert new[1].pose.position.z==pytest.approx(old[1].pose.position.z)
    assert new[1].pose.orientation==old[1].pose.orientation
    assert h._make_tf_link8_target_poses(box,layer,box_type)==new
    assert h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)==drag

def test_forward_is_footprint_direction_in_rotated_frame():
    tf=TransformStamped();tf.header.frame_id='base_footprint';tf.child_frame_id='frozen'
    tf.transform.rotation.x=math.sin(.4);tf.transform.rotation.w=math.cos(.4)
    h=harness(.02,tf)
    before=pose('frozen',(.2,-.3,.7))
    after=BoxGeometryMixin._raise_drag_contact_target(h,before,
        parameter_name='grasp_box_tf_right_forward_offset_m_layer4',footprint_direction=(0.,-1.,0.))
    a,b=do_transform_pose_stamped(before,tf),do_transform_pose_stamped(after,tf)
    assert b.pose.position.x==pytest.approx(a.pose.position.x)
    assert b.pose.position.y==pytest.approx(a.pose.position.y-.02)
    assert b.pose.position.z==pytest.approx(a.pose.position.z)
