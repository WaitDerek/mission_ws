import math
import pytest
from mission_runtime.waist_workspace_optimizer import WaistWorkspaceOptimizer

def test_deferred_left_clearance_changes_right_only_winner():
    opt=WaistWorkspaceOptimizer(
        solve_ik=lambda arm,pose,seed: [0. if pose[0]<.5 else 10.]*7,
        left_joint_limits_deg=([-100.]*7,[100.]*7),
        right_joint_limits_deg=([-100.]*7,[100.]*7),
        waist_limits_deg=([0.,0.,0.],[1.,1.,1.]),
        search_mode='exhaustive',candidate_step_deg=1.,candidate_delta_deg=1.,minimum_margin_deg=.1)
    args=dict(nominal_waist_rad=[0.]*3,left_seed_rad=[0.]*7,right_seed_rad=[0.]*7,
        target_pose_provider=lambda w: ([0.]*6,[math.degrees(w[0])]+[0.]*5),active_arms=('right',))
    old=opt.optimize(**args,auxiliary_candidate_validator=lambda *a: True)
    assert old.waist_angles_rad[0]==0.
    result=opt.optimize(**args,auxiliary_candidate_validator=lambda w,s:
        {'left': [99. if math.degrees(w[0])<.5 else 20.]*7})
    assert math.degrees(result.waist_angles_rad[0])==pytest.approx(1.)
    assert result.left_margin_deg==80.
    assert result.right_margin_deg==90.
    assert result.score_deg==80.
    assert result.left_joint_deg==(20.,)*7
