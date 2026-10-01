from copy import deepcopy
import pytest
from test_tf_box_center_correction import _TargetHarness
from test_grasp_contact_forward_scale import pose

@pytest.mark.parametrize("box_type", ["smallbox", "bigbox"])
@pytest.mark.parametrize("layer", [1,2,3,4])
def test_grasp_both_targets_rise_once_without_changing_drag(box_type, layer):
    h = _TargetHarness()
    h.VALUES = deepcopy(h.VALUES)
    for key, value in list(h.VALUES.items()):
        if key.endswith("_smallbox_layer2"):
            h.VALUES[key.replace("_smallbox_layer2", f"_{box_type}_layer{layer}")] = deepcopy(value)
    h.VALUES.update({"direct_movel_fixture_compensation_enabled": True,
                     "left_fixture_center_in_link8_xyz": [-.12,-.1,.04],
                     "right_fixture_center_in_link8_xyz": [-.12,.1,.04]})
    box = pose("base_footprint", (.2,-.6,.7))
    old = h._make_tf_link8_target_poses(box,layer,box_type)
    drag = h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True)
    h.VALUES["grasp_box_tf_contact_height_offset_m"] = .04
    new = h._make_tf_link8_target_poses(box,layer,box_type)
    for a,b in zip(old,new):
        assert b.pose.position.z == pytest.approx(a.pose.position.z+.04)
        assert b.pose.position.x == pytest.approx(a.pose.position.x)
        assert b.pose.position.y == pytest.approx(a.pose.position.y)
        assert b.pose.orientation == a.pose.orientation
    assert h._make_tf_link8_target_poses(box,layer,box_type) == new
    assert h._make_tf_link8_target_poses(box,layer,box_type,drag_mode=True) == drag
