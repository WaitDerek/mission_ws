import math
from types import SimpleNamespace
import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from mission_runtime.box_horizontal import constrain_box_horizontal, apply_box_horizontal
from mission_runtime.common import MissionError

def make_pose(rotation):
    p=PoseStamped();p.header.frame_id="base_link"
    p.pose.position.x=.2;p.pose.position.y=-.6;p.pose.position.z=.8
    p.pose.orientation.x,p.pose.orientation.y,p.pose.orientation.z,p.pose.orientation.w=map(float,rotation.as_quat())
    return p

def rotation(p):
    q=p.pose.orientation
    return R.from_quat([q.x,q.y,q.z,q.w])

@pytest.mark.parametrize("heading",[-2.,0.,.7,2.8])
@pytest.mark.parametrize("source_rpy",[(0.,0.,0.),(.3,-.2,.8)])
def test_level_preserves_center_heading_and_is_idempotent(heading,source_rpy):
    source_foot=R.from_euler("xyz",source_rpy)
    upright=R.from_euler("z",heading)*R.from_euler("y",-math.pi/2)
    tilted=R.from_euler("xyz",[.15,.1,0.])*upright
    p=make_pose(source_foot*tilted)
    foot=((.3,.2,.1),source_foot.as_quat())
    result,tilt=constrain_box_horizontal(p,foot)
    matrix=(source_foot.inv()*rotation(result)).as_matrix()
    assert matrix[:,0]==pytest.approx([0.,0.,1.],abs=1e-12)
    lateral=tilted.as_matrix()[:,2];lateral[2]=0.;lateral/=np.linalg.norm(lateral)
    assert matrix[:,2]==pytest.approx(lateral)
    assert np.linalg.det(matrix)==pytest.approx(1.)
    assert result.pose.position==p.pose.position
    again,_=constrain_box_horizontal(result,foot)
    assert (rotation(again).inv()*rotation(result)).magnitude()<1e-12
    assert tilt>0.

@pytest.mark.parametrize("drag",[False,True])
def test_workflow_flag_and_shared_corrected_pose(drag):
    prefix="drag_box_tf" if drag else "grasp_box_tf"
    seen=[]
    node=SimpleNamespace(
        _boolean=lambda k: seen.append(k) or True,
        _lookup_tf_carry_transform=lambda *a,**k: ((0.,0.,0.),(0.,0.,0.,1.)))
    p=make_pose(R.from_euler("xyz",[.2,-1.4,.3]))
    out,detail=apply_box_horizontal(node,p,drag_mode=drag)
    assert seen==[prefix+"_horizontal_constraint_enabled"]
    assert rotation(out).as_matrix()[:,0]==pytest.approx([0.,0.,1.],abs=1e-12)
    assert "center=unchanged" in detail
    node._boolean=lambda k:False
    same,detail=apply_box_horizontal(node,p,drag_mode=drag)
    assert same==p and same is not p and detail==""

def test_vertical_lateral_axis_rejected_instead_of_inventing_heading():
    with pytest.raises(MissionError):
        constrain_box_horizontal(make_pose(R.identity()),((0.,0.,0.),(0.,0.,0.,1.)))
