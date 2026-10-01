from copy import deepcopy
import math
import pytest
from test_tf_box_center_correction import _TargetHarness
from test_grasp_contact_forward_scale import pose

@pytest.mark.parametrize("box_type",["smallbox","bigbox"])
@pytest.mark.parametrize("layer",[1,2,3,4])
@pytest.mark.parametrize("yaw",[0.,.7])
@pytest.mark.parametrize("adjusted_layer",[3,4])
def test_right_offset_is_in_initial_pose_only(box_type,layer,yaw,adjusted_layer):
    h=_TargetHarness();h.VALUES=deepcopy(h.VALUES)
    for k,v in list(h.VALUES.items()):
        if k.endswith("_smallbox_layer2"):
            h.VALUES[k.replace("_smallbox_layer2",f"_{box_type}_layer{layer}")]=deepcopy(v)
    box=pose("base_footprint",(.2,-.6,.7))
    box.pose.orientation.z=math.sin(yaw/2);box.pose.orientation.w=math.cos(yaw/2)
    old=h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)
    grasp=h._make_tf_link8_target_poses(box,layer,box_type)
    h.VALUES[f"drag_box_tf_right_forward_offset_m_layer{adjusted_layer}"]=.01
    new=h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)
    for arm,(before,after) in enumerate(zip(old,new)):
        assert after.pose.position.y==pytest.approx(before.pose.position.y - (.01 if layer==adjusted_layer and arm==1 else 0.))
        assert after.pose.position.x==pytest.approx(before.pose.position.x)
        assert after.pose.position.z==pytest.approx(before.pose.position.z)
        assert after.pose.orientation==before.pose.orientation
    assert h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)==new
    assert h._make_tf_link8_target_poses(box,layer,box_type)==grasp
