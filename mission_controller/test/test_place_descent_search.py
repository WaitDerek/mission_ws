"""Pure offline planner tests: no ROS, SDK connection or motion."""
from dataclasses import replace

import pytest

from mission_runtime.place_descent_search import DescentYSettings, select_common_y


SETTINGS = DescentYSettings(0.1, 0.001, 0.002, 2.0, 10.0)
Q = [0.0, 20.0, 0.0, -40.0, 0.0, 0.0, 0.0]


def run(solve, **kwargs):
    options = dict(
        previous_y=0.0, settings=SETTINGS,
        seeds={a: list(Q) for a in ('left', 'right')},
        limits={a: ([-100.0] * 7, [100.0] * 7) for a in ('left', 'right')},
        solve_ik=solve,
        target_builder=lambda y, f: {a: [y, f] for a in ('left', 'right')},
        check_canceled=lambda: None, negative_joint4=True,
    )
    options.update(kwargs)
    return select_common_y(**options)


def test_no_lateral_motion_when_straight_descent_is_valid():
    seen = []
    def solve(a, p, q):
        seen.append((a, p))
        return q
    y, detail = run(solve)
    assert y == 0.0
    assert len(seen) == 4
    assert 'checked=1' in detail


def test_small_common_shift_not_independent_opposing_arm_shifts():
    def solve(a, p, q):
        return None if p[0] == 0.0 else q
    y, detail = run(solve)
    assert y == -0.001
    assert 'checked=2' in detail


def test_candidates_reset_seeds_but_samples_chain_solutions():
    calls = []
    def solve(arm, p, q):
        calls.append((arm, p[:], q[:]))
        if p[0] == 0 and arm == 'right':
            return None
        result = q[:]
        result[0] += 1
        return result
    run(solve)
    shifted = [c for c in calls if c[1][0] == -0.001]
    assert shifted[0][2][0] == 0
    assert shifted[2][2][0] == 1


def test_midpoint_failure_rejected_even_when_endpoint_solvable():
    with pytest.raises(ValueError, match='no bilateral IK-valid'):
        run(lambda a, p, q: None if p[1] == 0.5 else q)


@pytest.mark.parametrize('index,value', [(0, 99), (3, 1), (0, 11)])
def test_limits_joint4_and_branch_jumps_rejected(index, value):
    def solve(a, p, q):
        result = q[:]
        result[index] = value
        return result
    with pytest.raises(ValueError, match='no bilateral IK-valid'):
        run(solve)


def test_no_lateral_search_after_endpoint_contact():
    visited = []
    def solve(a, p, q):
        visited.append(p[0])
        return None
    with pytest.raises(ValueError, match='y_change_allowed=False'):
        run(solve, previous_y=0.003, allow_change=False)
    assert visited == [0.003]


def test_search_stays_inside_total_and_per_step_bounds():
    seen = []
    def solve(a, p, q):
        seen.append(p[0])
        return None
    with pytest.raises(ValueError):
        run(solve, previous_y=0.1)
    assert all(0.098 - 1e-9 <= y <= 0.1 for y in seen)
    assert len(seen) <= 5


def test_cancel_propagates_before_any_solver_call():
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        run(lambda *_: pytest.fail('IK must not run'), check_canceled=cancel)


@pytest.mark.parametrize('settings', [
    replace(SETTINGS, grid_m=0), replace(SETTINGS, half_range_m=float('nan')),
    replace(SETTINGS, max_step_m=0.02), replace(SETTINGS, grid_m=0.00001),
])
def test_invalid_settings_reject_before_ik(settings):
    with pytest.raises(ValueError):
        run(lambda *_: pytest.fail('IK must not run'), settings=settings)
