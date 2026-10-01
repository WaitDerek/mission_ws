"""Post-MoveL Work-Z verification with bounded downward-only catch-up."""
import math
import time
from dataclasses import dataclass

from rclpy.time import Time
from tf2_ros import TransformException

from .common import MissionError, rotate_vector


ARMS = ('left', 'right')


@dataclass(frozen=True)
class DescentZSettings:
    target_tolerance_m: float
    difference_tolerance_m: float
    timeout_sec: float
    max_age_sec: float
    stable_samples: int

    def validate(self):
        if any(not math.isfinite(v) or v <= 0 for v in (
            self.target_tolerance_m, self.difference_tolerance_m,
            self.timeout_sec, self.max_age_sec,
        )) or self.stable_samples < 1:
            raise ValueError('descent Z check tolerances/timeouts/sample count must be positive')


def work_z(base, world):
    """Project position onto that arm's Work-Z, including its own origin."""
    axis = rotate_vector((0.0, 0.0, 1.0), base[1])
    return sum(a * (p - origin) for a, p, origin in zip(axis, world[0], base[0]))


def evaluate_work_z(bases, starts, targets, actual, settings):
    settings.validate()
    z0 = {a: work_z(bases[a], starts[a]) for a in ARMS}
    goal = {a: work_z(bases[a], targets[a]) for a in ARMS}
    measured = {a: work_z(bases[a], actual[a]) for a in ARMS}
    if not all(math.isfinite(v) for d in (z0, goal, measured) for v in d.values()):
        raise MissionError('placement Work-Z check received non-finite transforms')
    errors = {a: measured[a] - goal[a] for a in ARMS}
    initial_difference = z0['left'] - z0['right']
    actual_difference = measured['left'] - measured['right']
    change = actual_difference - initial_difference
    passed = (all(abs(e) <= settings.target_tolerance_m + 1e-9 for e in errors.values())
              and abs(change) <= settings.difference_tolerance_m + 1e-9)
    detail = (
        f"left_work_z={measured['left']:.6f}m; right_work_z={measured['right']:.6f}m; "
        f"left_target_z={goal['left']:.6f}m; right_target_z={goal['right']:.6f}m; "
        f"left_z_error={errors['left']:+.6f}m; right_z_error={errors['right']:+.6f}m; "
        f"initial_left_minus_right_z={initial_difference:+.6f}m; "
        f"actual_left_minus_right_z={actual_difference:+.6f}m; "
        f"z_difference_change={change:+.6f}m; "
        f"target_tolerance={settings.target_tolerance_m:.6f}m; "
        f"difference_tolerance={settings.difference_tolerance_m:.6f}m"
    )
    return passed, detail


def choose_downward_correction(bases, targets, actual, settings, step_m, budgets):
    """Correct only positive target error, never chase the lower arm's overshoot."""
    errors = {a: work_z(bases[a], actual[a]) - work_z(bases[a], targets[a]) for a in ARMS}
    if any(not math.isfinite(v) for v in errors.values()):
        raise MissionError('invalid Work-Z error; no correction commanded')
    if any(v < -settings.target_tolerance_m - 1e-9 for v in errors.values()):
        raise MissionError('an arm already overshot its Z target; refusing to chase it downward')
    arm = max(ARMS, key=lambda a: errors[a])
    distance = min(errors[arm], step_m, budgets[arm])
    if distance <= 1e-6:
        raise MissionError('no bounded downward correction remains toward the original Z targets')
    return arm, distance


class PlaceDescentZGuardMixin:
    def _place_box_test_read_z_check_tf(self, base_frame, after_ns, max_age_sec):
        """Require fresh post-completion data, then query both at one TF time."""
        frames = {a: self._string(f'{a}_link8_frame').strip().lstrip('/') for a in ARMS}
        latest = {a: self.tf_buffer.lookup_transform(base_frame, frames[a], Time()) for a in ARMS}
        stamps = {a: int(m.header.stamp.sec) * 10**9 + int(m.header.stamp.nanosec)
                  for a, m in latest.items()}
        stamp = min(stamps.values())
        now_ns = self.get_clock().now().nanoseconds
        age = (now_ns - stamp) / 1e9
        if stamp <= after_ns or age < 0 or age > max_age_sec:
            return None, stamp, f'waiting for post-MoveL TF: age={age:.3f}s, stamps={stamps}'
        result = {}
        for arm in ARMS:
            m = self.tf_buffer.lookup_transform(base_frame, frames[arm], Time(nanoseconds=stamp))
            t, q = m.transform.translation, m.transform.rotation
            xyz, quat = (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)
            if not all(math.isfinite(v) for v in xyz + quat):
                raise MissionError('non-finite placement TF')
            norm = math.sqrt(sum(v * v for v in quat))
            if norm < 1e-12:
                raise MissionError('invalid placement TF quaternion')
            result[arm] = (xyz, tuple(v / norm for v in quat))
        return result, stamp, ''

    def _place_box_test_verify_descent_z(self, goal_handle, adapter, base_frame, reference, targets):
        """Verify, catch up only the high/lagging side, then verify again."""
        settings = DescentZSettings(
            self._float('place_box_test_descent_z_target_tolerance_m'),
            self._float('place_box_test_descent_z_difference_tolerance_m'),
            self._float('place_box_test_descent_z_check_timeout_sec'),
            self._float('place_box_test_descent_z_feedback_max_age_sec'),
            self._integer('place_box_test_stable_samples'),
        )
        latest_detail = 'no new TF samples'
        try:
            settings.validate()
            max_corrections = self._integer('place_box_test_descent_z_correction_max_attempts')
            step_m = self._float('place_box_test_descent_z_correction_step_m')
            max_travel = self._float('place_box_test_descent_z_correction_max_travel_m')
            if (not 1 <= max_corrections <= 5 or not math.isfinite(step_m)
                    or not math.isfinite(max_travel) or not 0 < step_m <= max_travel <= 0.010):
                raise ValueError('invalid bounded descent Z correction settings')
            traveled = dict.fromkeys(ARMS, 0.0)
            for correction_count in range(max_corrections + 1):
                after_ns = self.get_clock().now().nanoseconds
                deadline = time.monotonic() + settings.timeout_sec
                stable = 0
                last_stamp = after_ns
                latest_actual = None
                bad_samples = []
                while time.monotonic() < deadline:
                    self._check_canceled(goal_handle, 'while verifying bilateral placement Work-Z')
                    try:
                        actual, stamp, waiting = self._place_box_test_read_z_check_tf(
                            base_frame, after_ns, settings.max_age_sec
                        )
                    except TransformException as exc:
                        actual, stamp, waiting = None, last_stamp, str(exc)
                    if actual is None:
                        stable = 0
                        latest_actual = None
                        bad_samples.clear()
                        latest_detail = waiting
                    elif stamp > last_stamp:
                        last_stamp = stamp
                        latest_actual = actual
                        passed, latest_detail = evaluate_work_z(
                            reference.base_by_arm, reference.tcp_by_arm, targets, actual, settings
                        )
                        stable = stable + 1 if passed else 0
                        if passed:
                            bad_samples.clear()
                        else:
                            bad_samples.append({a: work_z(reference.base_by_arm[a], actual[a]) for a in ARMS})
                            bad_samples = bad_samples[-settings.stable_samples:]
                        if stable >= settings.stable_samples:
                            self._publish_place_box_test_feedback(
                                goal_handle, 'PLACE_DESCENT_Z_VERIFIED',
                                f'{latest_detail}; stable_samples={stable}; '
                                f'tf_stamp_ns={stamp}; corrections={correction_count}; '
                                f'correction_travel_m={traveled}',
                            )
                            return actual
                    time.sleep(0.02)
                age = (self.get_clock().now().nanoseconds - last_stamp) / 1e9
                settled_bad = (len(bad_samples) == settings.stable_samples and all(
                    max(s[a] for s in bad_samples) - min(s[a] for s in bad_samples) <= 0.001
                    for a in ARMS
                ))
                if (correction_count == max_corrections or latest_actual is None
                        or not 0 <= age <= settings.max_age_sec or not settled_bad):
                    raise MissionError(
                        'placement bilateral Work-Z verification failed; no next segment or release: '
                        f'{latest_detail}; corrections={correction_count}/{max_corrections}; '
                        f'settled_fresh_error={settled_bad}; correction_travel_m={traveled}'
                    )
                try:
                    arm, distance = choose_downward_correction(
                        reference.base_by_arm, targets, latest_actual, settings, step_m,
                        {a: max_travel - traveled[a] for a in ARMS},
                    )
                except MissionError as exc:
                    raise MissionError(f'{exc}; {latest_detail}; no next segment or release') from exc
                self._publish_place_box_test_feedback(
                    goal_handle, 'PLACE_DESCENT_Z_CORRECTING',
                    f'arm={arm}; downward={distance:.6f}m; attempt={correction_count + 1}/{max_corrections}; '
                    f'{latest_detail}; other_arm=hold; target=original_segment_Z',
                )
                self._check_canceled(goal_handle, 'before downward-only Z correction')
                self._place_box_test_correct_descent_z(
                    goal_handle, adapter, reference, latest_actual, arm, distance
                )
                traveled[arm] += distance
        except Exception as exc:
            try:
                adapter.stop_all()
            except Exception as stop_exc:
                self.get_logger().error(f'placement Z guard stop failed: {stop_exc}')
            self._publish_place_box_test_feedback(goal_handle, 'PLACE_DESCENT_Z_CHECK_FAILED', str(exc))
            raise
