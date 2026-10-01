from copy import deepcopy
import pytest
from test_tf_box_center_correction import _TargetHarness
from test_grasp_contact_forward_scale import pose

@pytest.mark.parametrize("box_type",["smallbox","bigbox"])
@pytest.mark.parametrize("layer",[1,2,3,4])
def test_left_backward_only_layer1_and_grasp(box_type,layer):
    h=_TargetHarness();h.VALUES=deepcopy(h.VALUES)
    for k,v in list(h.VALUES.items()):
        if k.endswith("_smallbox_layer2"):
            h.VALUES[k.replace("_smallbox_layer2",f"_{box_type}_layer{layer}")]=deepcopy(v)
    box=pose("base_footprint",(.2,-.6,.7))
    old=h._make_tf_link8_target_poses(box,layer,box_type)
    drag=h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)
    h.VALUES["grasp_box_tf_left_backward_offset_m_layer1"]=.012
    new=h._make_tf_link8_target_poses(box,layer,box_type)
    assert new[1]==old[1]
    assert new[0].pose.position.y==pytest.approx(old[0].pose.position.y+(.012 if layer==1 else 0.))
    assert new[0].pose.position.x==pytest.approx(old[0].pose.position.x)
    assert new[0].pose.position.z==pytest.approx(old[0].pose.position.z)
    assert new[0].pose.orientation==old[0].pose.orientation
    assert h._make_tf_link8_target_poses(box,layer,box_type)==new
    assert h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)==drag
