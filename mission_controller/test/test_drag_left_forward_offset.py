"""Forward correction stays robot-relative after box reanchoring and Z alignment."""
import math
from copy import deepcopy
import pytest
from mission_runtime.mission_controller import MissionController as M
from mission_runtime.box_drag_join import offset_robot_forward
from mission_runtime.common import MissionError
from test_grasp_contact_forward_scale import reanchor_harness

I=((0.,0.,0.),(0.,0.,0.,1.))

@pytest.mark.parametrize('yaw',[0.,.6,1.57])
@pytest.mark.parametrize('k',[1.,.5])
def test_final_left_target_moves_two_cm_once_and_retains_carry_relation(yaw,k):
    h=reanchor_harness()
    h.values['drag_box_tf_left_contact_forward_delta_scale']=k
    h.transforms['base_footprint']=I
    h.transforms['left_base']=((.1,.2,.3),(math.sin(.3),0.,0.,math.cos(.3)))
    M._capture_drag_tf_right_grasp_relation(h)
    desired=deepcopy(h._last_drag_box_tf_desired_box_to_link7_targets)
    box=((2.,2.,3.),(0.,0.,math.sin(yaw/2),math.cos(yaw/2)))
    h.transforms['right_link']=M._compose_transform(box,h._last_drag_box_tf_right_grasp_relation)
    right_before=deepcopy(h.transforms['right_link'])
    old,_=M._reanchor_drag_tf_left_join_after_drag3(h)
    h.values['drag_box_tf_left_join_forward_offset_m']=.02
    new,detail=M._reanchor_drag_tf_left_join_after_drag3(h)
    def world(p):
        return M._compose_transform(h.transforms['left_base'],
            ((p.position.x,p.position.y,p.position.z),(p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w)))
    a,b=world(old),world(new)
    assert b[0]==pytest.approx((a[0][0],a[0][1]-.02,a[0][2]))
    assert b[1]==pytest.approx(a[1])
    assert h.transforms['right_link']==right_before
    assert h._last_drag_box_tf_desired_box_to_link7_targets==desired
    cached=M._compose_transform(h._last_drag_box_tf_reanchored_box_after_drag3,
                                h._last_grasp_box_tf_box_to_link7_targets['left'])
    assert cached[0]==pytest.approx(b[0])
    again,_=M._reanchor_drag_tf_left_join_after_drag3(h)
    assert world(again)[0]==pytest.approx(b[0])
    assert 'left_forward_offset=0.0200m' in detail

def test_rotated_freeze_frame_preserves_robot_forward_direction():
    foot=((.2,.3,.4),(0.,0.,math.sin(.4),math.cos(.4)))
    target=((.1,.2,.3),I[1])
    shifted=offset_robot_forward(target,foot,.02)
    inverse=M._inverse_transform(foot)
    old=M._compose_transform(inverse,target)
    new=M._compose_transform(inverse,shifted)
    assert new[0]==pytest.approx((old[0][0],old[0][1]-.02,old[0][2]))
    assert shifted[1]==target[1]

def test_invalid_offset_rejected():
    with pytest.raises(MissionError):offset_robot_forward(I,I,float('nan'))

def test_precheck_prediction_includes_same_forward_offset():
    from test_drag_box_tf_action import _DragTfReanchorHarness
    h = _DragTfReanchorHarness()
    h.transforms['base_footprint'] = I
    for step in (1, 2, 3):
        h.values[f'drag_box_tf_post_movel_step_drag{step}_right_xyz_bigbox_layer1'] = [0., 0., 0.]
    h._equalize_tf_dual_target_z = lambda left, right, reference: M._equalize_tf_dual_target_z(h, left, right, reference=reference)
    args = (I, ((0., 1., 0.), I[1]), ((.5, -1., .1), I[1]), I, I)
    old = M._predict_drag_tf_left_join_target(h, *args, box_layer=1, model_label='bigbox')
    h.values['drag_box_tf_left_join_forward_offset_m'] = .02
    new = M._predict_drag_tf_left_join_target(h, *args, box_layer=1, model_label='bigbox')
    assert new.position.x == pytest.approx(old.position.x)
    assert new.position.y == pytest.approx(old.position.y - .02)
    assert new.position.z == pytest.approx(old.position.z)

def test_layer1_override_changes_runtime_join_by_half_cm_only():
    from mission_runtime.box_drag_join import BoxDragJoinMixin as B
    h=reanchor_harness()
    h.transforms['base_footprint']=I
    h.values['drag_box_tf_left_join_forward_offset_m']=.02
    h.values['drag_box_tf_left_join_forward_offset_m_layer1']=-1.
    M._capture_drag_tf_right_grasp_relation(h)
    old,_=M._reanchor_drag_tf_left_join_after_drag3(h,box_layer=1)
    h.values['drag_box_tf_left_join_forward_offset_m_layer1']=.015
    new,detail=M._reanchor_drag_tf_left_join_after_drag3(h,box_layer=1)
    assert new.position.y==pytest.approx(old.position.y+.005)
    assert new.position.x==pytest.approx(old.position.x)
    assert new.position.z==pytest.approx(old.position.z)
    assert 'left_forward_offset=0.0150m' in detail
    assert B._drag_left_join_forward_offset(h,1)==.015
    h.values['drag_box_tf_left_join_forward_offset_m_layer3']=-1.
    for layer in (2,3,4):
        assert B._drag_left_join_forward_offset(h,layer)==.02
        unchanged,_=M._reanchor_drag_tf_left_join_after_drag3(h,box_layer=layer)
        assert unchanged.position.y==pytest.approx(old.position.y)


def test_layer3_offset_is_in_reanchored_target_without_extra_move():
    h=reanchor_harness()
    h.transforms['base_footprint']=I
    h.values['drag_box_tf_left_join_forward_offset_m']=.02
    h.values['drag_box_tf_left_join_forward_offset_m_layer3']=-1.
    M._capture_drag_tf_right_grasp_relation(h)
    old,_=M._reanchor_drag_tf_left_join_after_drag3(h,box_layer=3)
    h.values['drag_box_tf_left_join_forward_offset_m_layer3']=.03
    new,detail=M._reanchor_drag_tf_left_join_after_drag3(h,box_layer=3)
    assert new.position.y==pytest.approx(old.position.y-.01)
    assert new.position.x==pytest.approx(old.position.x)
    assert new.position.z==pytest.approx(old.position.z)
    assert new.orientation==old.orientation
    assert 'left_forward_offset=0.0300m' in detail

def test_layer4_override_adds_one_cm_after_drag_only():
    h=reanchor_harness()
    h.transforms['base_footprint']=I
    h.values['drag_box_tf_left_join_forward_offset_m']=.02
    h.values['drag_box_tf_left_join_forward_offset_m_layer4']=-1.
    M._capture_drag_tf_right_grasp_relation(h)
    old,_=M._reanchor_drag_tf_left_join_after_drag3(h,box_layer=4)
    h.values['drag_box_tf_left_join_forward_offset_m_layer4']=.03
    new,detail=M._reanchor_drag_tf_left_join_after_drag3(h,box_layer=4)
    assert new.position.y==pytest.approx(old.position.y-.01)
    assert new.position.x==pytest.approx(old.position.x)
    assert new.position.z==pytest.approx(old.position.z)
    assert new.orientation==old.orientation
    assert 'left_forward_offset=0.0300m' in detail
    again,_=M._reanchor_drag_tf_left_join_after_drag3(h,box_layer=4)
    assert again==new
