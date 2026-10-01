"""Bounded, motion-free paired Work-Y search for a rigid dual-arm descent step.

This is a local IK feasibility filter, not a collision or controller-path proof.
The caller supplies Work-frame targets at the midpoint and endpoint of MoveL.
The scalar Y offset is measured along LEFT Work Y; mirrored right Work Y has
the opposite numerical sign. It is not robot base_link Y.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DescentYSettings:
    half_range_m: float
    grid_m: float
    max_step_m: float
    min_margin_deg: float
    max_joint_step_deg: float

    def validate(self):
        values = (self.half_range_m, self.grid_m, self.max_step_m,
                  self.min_margin_deg, self.max_joint_step_deg)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("descent Y search settings must be finite")
        if not 0.0 <= self.half_range_m <= 0.1:
            raise ValueError("descent common Y range must be within 0..0.1m")
        if not 0.0 < self.grid_m <= self.max_step_m <= 0.005:
            raise ValueError("descent Y grid must be positive and <= max step <= 0.005m")
        if self.min_margin_deg < 0 or not 0 < self.max_joint_step_deg <= 30:
            raise ValueError("invalid descent IK margin or continuity bound")
        if self.max_step_m / self.grid_m > 10:
            raise ValueError("descent Y search exceeds 21 local candidates")


def select_common_y(*, previous_y, settings, seeds, limits, solve_ik,
                    target_builder, check_canceled, negative_joint4, allow_change=True):
    """Prefer unchanged Y; otherwise the smallest feasible common-Y change.

    Each candidate starts from the same measured joints. Solutions are chained
    midpoint -> endpoint, never between rejected candidates. Reject any branch
    jump relative to the measured start as well as either sampled interval.
    """
    settings.validate()
    if not math.isfinite(previous_y) or abs(previous_y) > settings.half_range_m + 1e-9:
        raise ValueError("current descent Y exceeds configured search range")
    for arm in ('left', 'right'):
        lo, hi = limits[arm]
        if any(len(v) != 7 for v in (seeds[arm], lo, hi)) or not all(
            math.isfinite(v) for a in (seeds[arm], lo, hi) for v in a
        ) or any(a >= b for a, b in zip(lo, hi)):
            raise ValueError("descent IK requires finite seven-joint seeds and limits")
    offsets = {previous_y}
    if allow_change:
        n = int(settings.max_step_m / settings.grid_m + 1e-9)
        offsets.update(round(previous_y + i * settings.grid_m, 9) for i in range(-n, n + 1))
    offsets = sorted((v for v in offsets if abs(v) <= settings.half_range_m + 1e-9),
                     key=lambda v: (abs(v - previous_y), abs(v), v))
    checked = 0
    for y in offsets:
        check_canceled()
        checked += 1
        current = {arm: list(seeds[arm]) for arm in seeds}
        margin = math.inf
        valid = True
        for fraction in (0.5, 1.0):
            targets = target_builder(y, fraction)
            for arm in ('left', 'right'):
                check_canceled()
                solution = solve_ik(arm, targets[arm], current[arm])
                if solution is None:
                    valid = False
                    break
                q = [float(v) for v in solution]
                if len(q) != 7 or not all(math.isfinite(v) for v in q):
                    raise ValueError("descent IK returned invalid joint solution")
                lo, hi = limits[arm]
                arm_margin = min(min(v - a, b - v) for v, a, b in zip(q, lo, hi))
                jump = max(abs(v - old) for v, old in zip(q, current[arm]))
                total_jump = max(abs(v - old) for v, old in zip(q, seeds[arm]))
                if ((negative_joint4 and q[3] >= 0) or arm_margin < settings.min_margin_deg
                        or max(jump, total_jump) > settings.max_joint_step_deg):
                    valid = False
                    break
                margin = min(margin, arm_margin)
                current[arm] = q
            if not valid:
                break
        if valid:
            return y, f"checked={checked}; ik_samples=2x2; min_joint_margin={margin:.3f}deg"
    raise ValueError(
        f"no bilateral IK-valid descent step: checked={checked}; "
        f"previous_y={previous_y:+.4f}m; max_y_step={settings.max_step_m:.4f}m; "
        f"y_change_allowed={allow_change}; no motion commanded"
    )
