import threading,time
from types import SimpleNamespace
import pytest
from geometry_msgs.msg import PoseStamped,Pose
import mission_runtime.box_waist_planning as planning
from mission_runtime.mission_controller import MissionController

class Done(Exception): pass

@pytest.mark.parametrize('residual',[(0.,0.),(.002,0.),(0.,.02)])
@pytest.mark.parametrize('left_solution,expected',[(None,False),([0.,0.,0.,4.845,0.,0.,0.],True),([200.]*7,False)])
def test_search_actually_checks_predicted_left_join(monkeypatch,left_solution,expected,residual):
    expected = expected and residual == (0.,0.)
    calls=[]
    class Optimizer:
        def __init__(self,**kwargs):pass
        def optimize(self,**kwargs):
            gate=kwargs['auxiliary_candidate_validator']
            assert callable(gate)
            result = gate((0.,0.,0.),{'right':[0.]*7})
            assert bool(result) is expected
            if expected: assert result == {'left': left_solution}
            assert kwargs['priority_joint_arm'] is None
            raise Done()
    monkeypatch.setattr(planning,'WaistWorkspaceOptimizer',Optimizer)
    box=PoseStamped();box.header.frame_id='base_link';box.pose.orientation.w=1.
    h=SimpleNamespace(
        direct_sdk_adapter=SimpleNamespace(solve_ik=lambda arm,target,seed: calls.append((arm,target)) or left_solution, ik_pose_residual=lambda *a: residual),
        joint_state_lock=threading.Lock(),latest_slave_arm_positions={'left':[0.]*7,'right':[0.]*7},
        latest_slave_arm_state_times={'left':time.monotonic(),'right':time.monotonic()},
        _boolean=lambda n:n=='waist_workspace_optimization_enabled',
        _float=lambda n: .1 if n=='waist_workspace_minimum_margin_deg' else 10.,
        _integer=lambda n:8,
        _float_array=lambda n:([-180.]*7 if 'min_deg' in n else [180.]*7),
        _string=lambda n:'zero' if n=='waist_workspace_ik_seed_mode' else 'coarse_to_fine',
        _box_layer_joint123_approach_angles_deg=lambda *a,**k:[0.]*3,
        _check_canceled=lambda *a:None,
        _last_grasp_box_tf_box_pose=box,
        _joint123_arm_base_transform=lambda *a:((0.,0.,0.),(0.,0.,0.,1.)),
        _predict_drag_tf_left_join_target=lambda *a,**k:Pose(),
        _drag_left_join_joint4_preference=lambda *a:None,
        _drag_left_join_base_seed_trials=lambda *a,**k:[('zero',[0.]*7)],
        _drag_left_join_joint4_acceptable=lambda *a,**k:True,
    )
    with pytest.raises(Done):
        planning.BoxWaistPlanningMixin._optimize_tf_waist_target(h,None,box,box,1,'bigbox',
            drag_mode=True,equalize_target_z=False,right_arm_only=True,delayed_left_join=True)
    assert calls and calls[0][0]=='left'
