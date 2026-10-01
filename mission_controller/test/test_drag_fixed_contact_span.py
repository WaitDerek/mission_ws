import math
import pytest
from mission_runtime.box_drag_join import fixed_contact_lateral_span, BoxDragJoinMixin
from mission_runtime.common import rotate_vector, MissionError
from mission_runtime.mission_controller import MissionController as M
from test_drag_box_tf_action import _DragTfReanchorHarness

I=(0.,0.,0.,1.)
@pytest.mark.parametrize("yaw", [0.,.4,1.2])
def test_actual_contacts_have_one_meter_lateral_span(yaw):
    # Rotate box -Z into horizontal left, including yaw.
    q=(math.sin(yaw/2)/math.sqrt(2),-math.cos(yaw/2)/math.sqrt(2),
       math.sin(yaw/2)/math.sqrt(2),math.cos(yaw/2)/math.sqrt(2))
    left=((.7,.2,.9),I); right=((-0.3,.1,.8),I)
    lf=(-.12,-.1,.04);rf=(-.12,.1,.04)
    result=fixed_contact_lateral_span(left,right,q,lf,rf,1.)
    axis=rotate_vector((0.,0.,-1.),q)
    lc=tuple(a+b for a,b in zip(result[0],lf))
    rc=tuple(a+b for a,b in zip(right[0],rf))
    assert sum((a-b)*v for a,b,v in zip(lc,rc,axis))==pytest.approx(1.)
    assert result[1]==left[1]
    again=fixed_contact_lateral_span(result,right,q,lf,rf,1.)
    assert again[0]==pytest.approx(result[0])

def test_disabled_and_smallbox_leave_target_unchanged():
    h=_DragTfReanchorHarness()
    target=((1.,2.,3.),I)
    assert BoxDragJoinMixin._drag_tf_apply_fixed_contact_span(h,target,target,target,'smallbox')==target
    assert BoxDragJoinMixin._drag_tf_apply_fixed_contact_span(h,target,target,target,'bigbox')==target

@pytest.mark.parametrize("span",[-1.,float('nan'),float('inf')])
def test_invalid_span_fails(span):
    with pytest.raises(MissionError):
        fixed_contact_lateral_span(((0.,0.,0.),I),((0.,0.,0.),I),I,(0.,0.,0.),(0.,0.,0.),span)

def test_prediction_and_runtime_use_same_fixed_span():
    h=_DragTfReanchorHarness()
    h.values.update({
        'drag_box_tf_left_join_contact_span_m_bigbox':1.,
        'drag_box_tf_left_join_forward_offset_m':.02,
        'left_fixture_center_in_link8_xyz':[-.12,-.1,.04],
        'right_fixture_center_in_link8_xyz':[-.12,.1,.04],
    })
    identity=((0.,0.,0.),I)
    h.transforms['base_footprint']=identity
    for step in (1,2,3):
        h.values[f'drag_box_tf_post_movel_step_drag{step}_right_xyz_bigbox_layer1']=[0.,0.,0.]
    h._equalize_tf_dual_target_z=lambda l,r,reference:M._equalize_tf_dual_target_z(h,l,r,reference=reference)
    M._capture_drag_tf_right_grasp_relation(h)
    frozen=M._pose_stamped_to_transform(h._last_grasp_box_tf_box_pose)
    left=M._compose_transform(frozen,h._last_drag_box_tf_desired_box_to_link7_targets['left'])
    predicted=M._predict_drag_tf_left_join_target(h,frozen,left,h.transforms['right_link'],
        h.transforms['left_base'],h.transforms['right_base'],box_layer=1,model_label='bigbox')
    actual,detail=M._reanchor_drag_tf_left_join_after_drag3(h,model_label='bigbox')
    assert [actual.position.x,actual.position.y,actual.position.z]==pytest.approx(
        [predicted.position.x,predicted.position.y,predicted.position.z])
    assert 'left_join_contact_lateral_span_m=1.0000' in detail
