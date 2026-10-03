import math
import pytest
from mission_runtime.waist_workspace_optimizer import WaistWorkspaceOptimizer

def optimizer(solve=None, residual=None, delta=0.1, span=1):
    return WaistWorkspaceOptimizer(
        solve_ik=solve or (lambda arm, pose, seed: [abs(pose[0])] * 7),
        ik_pose_residual=residual,
        robustness_delta_deg=delta,
        left_joint_limits_deg=([-180]*7, [180]*7),
        right_joint_limits_deg=([-180]*7, [180]*7),
        waist_limits_deg=([-5]*3, [5]*3),
        search_mode="exhaustive", candidate_delta_deg=span,
        candidate_step_deg=1, minimum_margin_deg=0)

def run(opt, **kwargs):
    return opt.optimize(nominal_waist_rad=[0]*3, left_seed_rad=[0]*7,
        right_seed_rad=[0]*7,
        target_pose_provider=lambda w: ([math.degrees(x) for x in w]+[0]*3,)*2,
        **kwargs)

def test_reject_false_success_and_nonfinite_fk():
    for error in [(0.00271, 0), (0, math.radians(.2)), (float("nan"), 0)]:
        with pytest.raises(RuntimeError, match="no .*valid"):
            run(optimizer(residual=lambda *args: error))

def test_skip_boundary_candidate_then_select_next():
    def solve(arm, pose, seed):
        if arm == "left" and 0 < pose[0] < .2:
            return None
        return [abs(pose[0])]*7
    result = run(optimizer(solve=solve))
    assert abs(math.degrees(result.waist_angles_rad[0])) == pytest.approx(1)

def test_all_six_axis_probes_and_progress():
    messages = []
    seen = []
    def solve(arm, pose, seed):
        seen.append((arm, tuple(round(x, 4) for x in pose[:3])))
        return [0]*7
    run(optimizer(solve=solve, span=0), progress_callback=messages.append)
    for axis in range(3):
        for sign in (-1, 1):
            target = [0.0]*3
            target[axis] = sign*.1
            assert ("left", tuple(target)) in seen
            assert ("right", tuple(target)) in seen
    assert any("probes=6/6" in m for m in messages)

def test_no_robust_candidate_fails_closed():
    def solve(arm, pose, seed):
        return None if any(abs(x)>0 for x in pose[:3]) else [0]*7
    with pytest.raises(RuntimeError, match="no robust waist target"):
        run(optimizer(solve=solve, span=0))

def test_auxiliary_drag_join_is_rechecked_at_perturbed_waist():
    def auxiliary(waist, solutions):
        return False if waist[1] > 0 else {"left": [0]*7}
    with pytest.raises(RuntimeError, match="no robust waist target"):
        run(optimizer(span=0), active_arms=("right",),
            auxiliary_candidate_validator=auxiliary)

def test_carry_endpoint_is_rechecked():
    def solve(arm, pose, seed):
        return None if pose[5] == 99 and pose[1] > 0 else [0]*7
    def endpoints(waist):
        return ([[0, waist[1], 0, 0, 0, 99]],)*2
    with pytest.raises(RuntimeError, match="no robust waist target"):
        run(optimizer(solve=solve, span=0), movel_endpoint_provider=endpoints)
