"""Force-limited box-clamping helpers for the TF grasp paths.

The force controller deliberately lives beside the normal post-grasp
trajectory code.  It only participates when a TF path explicitly selects
``closed_loop``; the legacy/adaptive paths and the default TF configuration
remain unchanged.  The sensor signal is the calibrated change in the raw
Link7 wrench force-X value::

    S = force_sign * (filtered_Fx - baseline_Fx)

The default sign is -1 because a push in the measured convention lowers Fx.
All movement directions are expressed in the FoundationPose box frame and
converted to each arm base by the existing geometry helper.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from typing import Optional, Sequence

from geometry_msgs.msg import Pose

from .common import MissionCanceled, MissionError
from .realman_sdk_adapter import RealManSdkCanceled, RealManSdkError, pose_to_sdk_target


# Bound to the composed compatibility facade by box_support.py.
BoxSupportMixin = None


class BoxForceClampMixin:
    """Closed-loop, wrench-triggered Step1 clamping for TF workflows."""

    @staticmethod
    def _force_clamp_parameter_prefix(*, tf_mode: bool, drag_mode: bool):
        if not tf_mode:
            return None
        return "drag_box_tf_force_clamp" if drag_mode else "grasp_box_tf_force_clamp"

    def _force_clamp_mode(self, parameter_prefix: Optional[str]) -> str:
        if not parameter_prefix:
            return "disabled"
        mode = self._string(f"{parameter_prefix}_mode").strip().lower()
        if mode not in ("disabled", "monitor_only", "closed_loop"):
            raise MissionError(
                f"{parameter_prefix}_mode must be disabled, monitor_only, or closed_loop"
            )
        return mode

    def _new_force_clamp_session(self, parameter_prefix: str) -> dict:
        arms = ("left", "right")
        filter_size = max(1, self._integer(f"{parameter_prefix}_filter_samples"))
        return {
            "prefix": parameter_prefix,
            "baseline_fx": {},
            "baseline_sequences": {},
            "windows": {arm: deque(maxlen=filter_size) for arm in arms},
            "travelled_m": {arm: 0.0 for arm in arms},
            "corrections": {arm: 0 for arm in arms},
        }

    def _force_clamp_wrench_snapshot(self, arm: str):
        with self.joint_state_lock:
            wrench = getattr(self, "latest_slave_arm_wrenches", {}).get(arm)
            sequence = getattr(self, "latest_slave_arm_wrench_sequences", {}).get(
                arm, 0
            )
            stamp = getattr(self, "latest_slave_arm_wrench_times", {}).get(arm, 0.0)
        age = time.monotonic() - stamp if stamp > 0.0 else math.inf
        return wrench, sequence, age

    def _force_clamp_pose_snapshot(self, arm: str):
        with self.joint_state_lock:
            values = getattr(self, "latest_slave_arm_poses", {}).get(arm)
            sequence = getattr(self, "latest_slave_arm_pose_sequences", {}).get(
                arm, 0
            )
            stamp = getattr(self, "latest_slave_arm_pose_times", {}).get(arm, 0.0)
        age = time.monotonic() - stamp if stamp > 0.0 else math.inf
        if values is None:
            return None, sequence, age
        return BoxSupportMixin._endpoint_sync_transform_to_pose(
            BoxSupportMixin._endpoint_sync_pose_values_to_transform(values)
        ), sequence, age

    def _force_clamp_velocity_is_zero(self, arm: str, tolerance: float) -> bool:
        with self.joint_state_lock:
            velocities = getattr(self, "latest_slave_arm_velocities", {}).get(arm, [])
        return bool(velocities) and all(abs(float(value)) <= tolerance for value in velocities)

    def _force_clamp_ensure_baseline(
        self,
        goal_handle,
        session: dict,
        arms: Sequence[str],
    ) -> None:
        """Collect only missing baselines, retaining DragBox's right baseline."""
        missing_arms = [arm for arm in arms if arm not in session["baseline_fx"]]
        if not missing_arms:
            return
        prefix = session["prefix"]
        duration = self._float(f"{prefix}_baseline_duration_sec")
        minimum = self._integer(f"{prefix}_baseline_min_samples")
        timeout = self._float(f"{prefix}_baseline_timeout_sec")
        sensor_max_age = self._float(f"{prefix}_sensor_max_age_sec")
        velocity_tolerance = self._float(f"{prefix}_arm_velocity_tolerance_rad_sec")
        deadline = time.monotonic() + timeout
        started = None
        collected = {arm: [] for arm in missing_arms}
        last_detail = "no wrench feedback"
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while collecting force baseline")
            now = time.monotonic()
            if started is None and all(
                self._force_clamp_velocity_is_zero(arm, velocity_tolerance)
                for arm in missing_arms
            ):
                started = now
            if started is not None:
                for arm in missing_arms:
                    wrench, sequence, age = self._force_clamp_wrench_snapshot(arm)
                    if wrench is None or age > sensor_max_age:
                        last_detail = f"{arm}=missing/stale wrench age={age:.3f}s"
                        continue
                    if sequence <= session["baseline_sequences"].get(arm, -1):
                        continue
                    fx = float(wrench[0])
                    if not math.isfinite(fx):
                        continue
                    collected[arm].append(fx)
                    session["baseline_sequences"][arm] = sequence
                if all(
                    len(collected[arm]) >= minimum and now - started >= duration
                    for arm in missing_arms
                ):
                    break
            time.sleep(0.01)
        if not all(
            len(collected[arm]) >= minimum for arm in missing_arms
        ):
            detail = ", ".join(
                f"{arm} samples={len(collected[arm])}" for arm in missing_arms
            )
            raise MissionError(
                f"{prefix} baseline collection timed out: {detail}; {last_detail}; "
                f"timeout_sec={timeout:.1f}"
            )
        for arm, values in collected.items():
            baseline = float(statistics.median(values))
            session["baseline_fx"][arm] = baseline
            session["windows"][arm].clear()
            session["windows"][arm].extend(values[-session["windows"][arm].maxlen :])

    def _force_clamp_metric(self, session: dict, arm: str) -> tuple[float, float]:
        prefix = session["prefix"]
        wrench, _sequence, age = self._force_clamp_wrench_snapshot(arm)
        if wrench is None:
            raise MissionError(f"{prefix} {arm} wrench feedback is unavailable")
        sensor_max_age = self._float(f"{prefix}_sensor_max_age_sec")
        if age > sensor_max_age:
            raise MissionError(
                f"{prefix} {arm} wrench feedback is stale: age={age:.3f}s, "
                f"max_age={sensor_max_age:.3f}s"
            )
        fx = float(wrench[0])
        if not math.isfinite(fx):
            raise MissionError(f"{prefix} {arm} wrench force-x is not finite")
        window = session["windows"][arm]
        window.append(fx)
        filtered = float(statistics.median(window))
        sign = self._float(f"{prefix}_force_sign_{arm}")
        value = sign * (filtered - session["baseline_fx"][arm])
        return value, age

    def _force_clamp_progress_callback(self, session: dict, arms: Sequence[str]):
        prefix = session["prefix"]
        emergency = {
            arm: self._float(f"{prefix}_emergency_threshold_{arm}_counts")
            for arm in arms
        }

        def progress():
            for arm in arms:
                value, _age = self._force_clamp_metric(session, arm)
                if value >= emergency[arm]:
                    raise MissionError(
                        f"{prefix} emergency threshold on {arm}: "
                        f"S={value:.1f} >= {emergency[arm]:.1f}"
                    )

        return progress

    @staticmethod
    def _force_clamp_direction(delta_box_xyz: Sequence[float], arm: str):
        delta = tuple(float(value) for value in delta_box_xyz)
        if len(delta) != 3 or not all(math.isfinite(value) for value in delta):
            raise MissionError(f"{arm} force-clamp direction is invalid")
        norm = math.sqrt(sum(value * value for value in delta))
        if norm <= 1e-9:
            return None
        return tuple(value / norm for value in delta)

    def _force_clamp_step_pose(
        self, current_pose: Pose, arm: str, direction: Sequence[float], distance: float
    ) -> Pose:
        return self._translate_pose_in_box_frame(
            current_pose,
            [float(direction[index]) * float(distance) for index in range(3)],
            arm,
        )

    def _force_clamp_move_arms(
        self,
        goal_handle,
        adapter,
        session: dict,
        current_poses: dict[str, Pose],
        move_arms: Sequence[str],
        directions: dict[str, Optional[Sequence[float]]],
    ) -> dict[str, Pose]:
        prefix = session["prefix"]
        speed = self._float(f"{prefix}_movel_velocity_percent")
        blocking = self._boolean("direct_movel_blocking")
        step = self._float(f"{prefix}_search_step_m")
        targets = {}
        previous_sequences = {
            arm: self.latest_slave_arm_pose_sequences.get(arm, 0)
            for arm in move_arms
        }
        for arm in move_arms:
            direction = directions.get(arm)
            if direction is None:
                continue
            value, _age = self._force_clamp_metric(session, arm)
            contact = self._float(f"{prefix}_contact_threshold_{arm}_counts")
            distance = (
                self._float(f"{prefix}_fine_step_m")
                if value >= contact
                else step
            )
            remaining = self._float(f"{prefix}_max_distance_{arm}_m") - session[
                "travelled_m"
            ][arm]
            if remaining <= 1e-9:
                raise MissionError(
                    f"{prefix} {arm} reached max clamp travel "
                    f"{self._float(f'{prefix}_max_distance_{arm}_m'):.3f}m"
                )
            distance = min(distance, remaining)
            targets[arm] = self._force_clamp_step_pose(
                current_poses[arm], arm, direction, distance
            )

        if not targets:
            return current_poses
        try:
            if len(targets) == 2:
                motion_result = adapter.execute_dual(
                    pose_to_sdk_target(targets["left"]),
                    pose_to_sdk_target(targets["right"]),
                    "movel",
                    speed,
                    blocking,
                    cancel_requested=lambda: goal_handle.is_cancel_requested,
                    timeout_sec=self._float(f"{prefix}_motion_timeout_sec"),
                    progress_callback=self._force_clamp_progress_callback(
                        session, tuple(current_poses)
                    ),
                )
            else:
                arm = next(iter(targets))
                motion_result = adapter.execute_single(
                    arm,
                    pose_to_sdk_target(targets[arm]),
                    "movel",
                    speed,
                    blocking,
                    cancel_requested=lambda: goal_handle.is_cancel_requested,
                    timeout_sec=self._float(f"{prefix}_motion_timeout_sec"),
                    progress_callback=self._force_clamp_progress_callback(
                        session, tuple(current_poses)
                    ),
                )
        except RealManSdkCanceled as exc:
            raise MissionCanceled(str(exc)) from exc
        except (RealManSdkError, ValueError, MissionError) as exc:
            raise MissionError(f"{prefix} clamp MoveL failed: {exc}") from exc
        for arm in targets:
            session["travelled_m"][arm] += distance if len(targets) == 1 else math.sqrt(
                sum(
                    (
                        targets[arm].position.__getattribute__(axis)
                        - current_poses[arm].position.__getattribute__(axis)
                    )
                    ** 2
                    for axis in ("x", "y", "z")
                )
            )
        del motion_result
        # The SDK call is blocking, but use a fresh pose sequence before
        # computing the next short correction.  This prevents stale feedback
        # from making a correction in the wrong direction.
        updated = dict(current_poses)
        deadline = time.monotonic() + self._float(f"{prefix}_motion_timeout_sec")
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for force-clamp pose feedback")
            all_fresh = True
            for arm in targets:
                pose, sequence, age = self._force_clamp_pose_snapshot(arm)
                if pose is None or sequence <= previous_sequences[arm] or age > self._float(
                    f"{prefix}_sensor_max_age_sec"
                ):
                    all_fresh = False
                    continue
                updated[arm] = pose
            if all_fresh:
                return updated
            time.sleep(0.01)
        raise MissionError(f"{prefix} did not receive fresh Link7 pose after clamp MoveL")

    def _force_clamp_wait_hold(
        self, goal_handle, session: dict, arms: Sequence[str], start_time: float
    ) -> bool:
        prefix = session["prefix"]
        hold_wait = self._float(f"{prefix}_hold_wait_sec")
        hold_duration = self._float(f"{prefix}_hold_required_duration_sec")
        hold_thresholds = {
            arm: self._float(f"{prefix}_hold_threshold_{arm}_counts") for arm in arms
        }
        if hold_wait > 0.0:
            time.sleep(hold_wait)
        deadline = time.monotonic() + max(hold_duration * 3.0, 1.0)
        held_since = None
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while holding force clamp")
            metrics = {
                arm: self._force_clamp_metric(session, arm)[0] for arm in arms
            }
            if all(metrics[arm] >= hold_thresholds[arm] for arm in arms):
                held_since = held_since or time.monotonic()
                if time.monotonic() - held_since >= hold_duration:
                    return True
            else:
                # Return to the main loop so it can issue at most the
                # configured number of fine corrections instead of silently
                # holding a weak contact.
                return False
            time.sleep(0.01)
        raise MissionError(
            f"{prefix} hold confirmation timed out after "
            f"{time.monotonic() - start_time:.2f}s"
        )

    def _force_clamp_contact_only(
        self,
        goal_handle,
        adapter,
        session: dict,
        arm: str,
        current_pose: Pose,
        delta_box_xyz: Sequence[float],
    ) -> tuple[Pose, str]:
        prefix = session["prefix"]
        self._force_clamp_ensure_baseline(goal_handle, session, (arm,))
        direction = self._force_clamp_direction(delta_box_xyz, arm)
        if direction is None:
            raise MissionError(f"{prefix} {arm} contact direction is zero")
        contact = self._float(f"{prefix}_contact_threshold_{arm}_counts")
        required = self._float(f"{prefix}_contact_required_duration_sec")
        emergency = self._float(f"{prefix}_emergency_threshold_{arm}_counts")
        timeout = self._float(f"{prefix}_timeout_sec")
        deadline = time.monotonic() + timeout
        contact_since = None
        current = current_pose
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "during force contact search")
            metric, _age = self._force_clamp_metric(session, arm)
            if metric >= emergency:
                raise MissionError(
                    f"{prefix} emergency threshold on {arm}: "
                    f"S={metric:.1f} >= {emergency:.1f}"
                )
            if metric >= contact:
                contact_since = contact_since or time.monotonic()
                if time.monotonic() - contact_since >= required:
                    return current, (
                        f"force_contact arm={arm}; S={metric:.1f}; "
                        f"baseline_Fx={session['baseline_fx'][arm]:.1f}; "
                        f"travel={session['travelled_m'][arm]:.4f}m"
                    )
            else:
                contact_since = None
                current = self._force_clamp_move_arms(
                    goal_handle,
                    adapter,
                    session,
                    {arm: current},
                    (arm,),
                    {arm: direction},
                )[arm]
            time.sleep(0.01)
        raise MissionError(f"{prefix} {arm} contact search timed out")

    def _force_clamp_bilateral(
        self,
        goal_handle,
        adapter,
        session: dict,
        current_poses: dict[str, Pose],
        directions: dict[str, Optional[Sequence[float]]],
    ) -> tuple[dict[str, Pose], str]:
        prefix = session["prefix"]
        arms = tuple(arm for arm in ("left", "right") if arm in current_poses)
        self._force_clamp_ensure_baseline(goal_handle, session, arms)
        clamped = {
            arm: self._float(f"{prefix}_clamped_threshold_{arm}_counts") for arm in arms
        }
        emergency = {
            arm: self._float(f"{prefix}_emergency_threshold_{arm}_counts") for arm in arms
        }
        required = self._float(f"{prefix}_clamped_required_duration_sec")
        deadline = time.monotonic() + self._float(f"{prefix}_timeout_sec")
        clamped_since = None
        current = dict(current_poses)
        start_time = time.monotonic()
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "during force clamp")
            metrics = {
                arm: self._force_clamp_metric(session, arm)[0] for arm in arms
            }
            for arm in arms:
                if metrics[arm] >= emergency[arm]:
                    raise MissionError(
                        f"{prefix} emergency threshold on {arm}: "
                        f"S={metrics[arm]:.1f} >= {emergency[arm]:.1f}"
                    )
            if all(metrics[arm] >= clamped[arm] for arm in arms):
                clamped_since = clamped_since or time.monotonic()
                if time.monotonic() - clamped_since >= required:
                    if self._force_clamp_wait_hold(
                        goal_handle, session, arms, start_time
                    ):
                        return current, (
                            "force_clamp confirmed; "
                            + ", ".join(
                                f"{arm}_S={metrics[arm]:.1f}" for arm in arms
                            )
                            + "; "
                            + ", ".join(
                                f"{arm}_travel={session['travelled_m'][arm]:.4f}m"
                                for arm in arms
                            )
                        )
                    low_arms = [
                        arm
                        for arm in arms
                        if metrics[arm]
                        < self._float(f"{prefix}_hold_threshold_{arm}_counts")
                    ]
                    for arm in low_arms:
                        session["corrections"][arm] += 1
                    if any(
                        session["corrections"][arm]
                        > self._integer(f"{prefix}_max_correction_count")
                        for arm in low_arms
                    ):
                        raise MissionError(
                            f"{prefix} hold force dropped below threshold: "
                            + ", ".join(
                                f"{arm}={metrics[arm]:.1f}" for arm in arms
                            )
                        )
                    correction_arms = [
                        arm for arm in low_arms if directions.get(arm) is not None
                    ]
                    if len(correction_arms) != len(low_arms):
                        raise MissionError(
                            f"{prefix} passive arm cannot correct hold-force drop: "
                            + ", ".join(low_arms)
                        )
                    current = self._force_clamp_move_arms(
                        goal_handle,
                        adapter,
                        session,
                        current,
                        correction_arms,
                        directions,
                    )
                    clamped_since = None
            else:
                clamped_since = None
                move_arms = [
                    arm
                    for arm in arms
                    if metrics[arm] < clamped[arm] and directions.get(arm) is not None
                ]
                # A zero direction means that arm is already at its nominal
                # Step1 pose; it is monitored for hold but never moved blindly.
                if not move_arms:
                    raise MissionError(
                        f"{prefix} clamp needs motion but all low-force arms are passive: "
                        + ", ".join(
                            f"{arm}={metrics[arm]:.1f}" for arm in arms
                        )
                    )
                current = self._force_clamp_move_arms(
                    goal_handle, adapter, session, current, move_arms, directions
                )
            time.sleep(0.01)
        raise MissionError(
            f"{prefix} clamp timed out: "
            + ", ".join(
                f"{arm}_S={self._force_clamp_metric(session, arm)[0]:.1f}"
                for arm in arms
            )
        )

    def _rebase_post_movel_targets_after_force_clamp(
        self,
        targets,
        next_target_index: int,
        actual_by_arm: dict[str, Pose],
        nominal_by_arm: dict[str, Pose],
    ) -> None:
        """Apply actual-vs-nominal clamp transforms to all later targets."""
        corrections = {}
        for arm, actual in actual_by_arm.items():
            nominal = nominal_by_arm.get(arm)
            if nominal is None:
                continue
            corrections[arm] = BoxSupportMixin._compose_transform(
                BoxSupportMixin._endpoint_sync_pose_values_to_transform(
                    (
                        actual.position.x,
                        actual.position.y,
                        actual.position.z,
                        actual.orientation.x,
                        actual.orientation.y,
                        actual.orientation.z,
                        actual.orientation.w,
                    )
                ),
                BoxSupportMixin._inverse_transform(
                    BoxSupportMixin._endpoint_sync_pose_values_to_transform(
                        (
                            nominal.position.x,
                            nominal.position.y,
                            nominal.position.z,
                            nominal.orientation.x,
                            nominal.orientation.y,
                            nominal.orientation.z,
                            nominal.orientation.w,
                        )
                    )
                ),
            )
        if not corrections:
            return
        for index in range(next_target_index, len(targets)):
            label, left, right = targets[index]
            updated = {"left": left, "right": right}
            for arm, correction in corrections.items():
                pose = updated[arm]
                transform = BoxSupportMixin._compose_transform(
                    correction,
                    BoxSupportMixin._endpoint_sync_pose_values_to_transform(
                        (
                            pose.position.x,
                            pose.position.y,
                            pose.position.z,
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        )
                    ),
                )
                updated[arm] = BoxSupportMixin._endpoint_sync_transform_to_pose(transform)
            targets[index] = (label, updated["left"], updated["right"])
