import pytest
from mission_runtime.box_drag_join import BoxDragJoinMixin as B

@pytest.mark.parametrize("layer,expected",[(1,.015),(2,.02),(3,.03),(4,.02)])
def test_left_join_offsets_are_selected_before_ik(layer,expected):
    class H:
        def _float(self,name):
            return {"drag_box_tf_left_join_forward_offset_m":.02,
                    "drag_box_tf_left_join_forward_offset_m_layer1":.015,
                    "drag_box_tf_left_join_forward_offset_m_layer3":.03,
                    "drag_box_tf_left_join_forward_offset_m_layer4":-1.}[name]
    assert B._drag_left_join_forward_offset(H(),layer)==pytest.approx(expected)
