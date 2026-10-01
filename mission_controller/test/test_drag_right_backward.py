from copy import deepcopy
import pytest
from test_tf_box_center_correction import _TargetHarness
from test_grasp_contact_forward_scale import pose

@pytest.mark.parametrize("box_type",["smallbox","bigbox"])
@pytest.mark.parametrize("layer",[1,2,3,4])
@pytest.mark.parametrize("adjusted_layer,distance",[(1,.04),(2,.015)])
def test_right_backward_only_selected_layer_and_drag(box_type,layer,adjusted_layer,distance):
    h=_TargetHarness();h.VALUES=deepcopy(h.VALUES)
    for k,v in list(h.VALUES.items()):
        if k.endswith("_smallbox_layer2"):
            h.VALUES[k.replace("_smallbox_layer2",f"_{box_type}_layer{layer}")]=deepcopy(v)
    box=pose("base_footprint",(.2,-.6,.7))
    old=h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)
    grasp=h._make_tf_link8_target_poses(box,layer,box_type)
    h.VALUES[f"drag_box_tf_right_backward_offset_m_layer{adjusted_layer}"]=distance
    new=h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)
    assert new[0]==old[0]
    assert new[1].pose.position.y==pytest.approx(old[1].pose.position.y+(distance if layer==adjusted_layer else 0.))
    assert new[1].pose.position.x==pytest.approx(old[1].pose.position.x)
    assert new[1].pose.position.z==pytest.approx(old[1].pose.position.z)
    assert new[1].pose.orientation==old[1].pose.orientation
    assert h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)==new
    assert h._make_tf_link8_target_poses(box,layer,box_type)==grasp
