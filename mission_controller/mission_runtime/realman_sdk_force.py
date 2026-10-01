"""RealMan SDK Tool-Y force-position incremental backend."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from statistics import median
from typing import Callable, Optional, Sequence

from .realman_sdk_common import RealManSdkCanceled, RealManSdkError
from .realman_sdk_algorithm import SDK_ALGORITHM_LOCK


class RealManSdkContactNotSustained(RealManSdkError):
    """The initial right contact vanished after force-position mode stopped."""

    def __init__(self, message: str, *, travel_m: float):
        super().__init__(message)
        self.travel_m = float(travel_m)


class RealManSdkForceMixin:
    """Coordinated one/two-arm Tool-Y force-position motion."""

    def read_bilateral_work_wrench(self) -> dict[str, tuple[float, ...]]:
        """Read both complete compensated Work-frame wrenches without motion."""
        with self._motion_lock:
            self._connect()
            robots = dict(zip(("left", "right"), self._robots()))
            result = {}
            for arm, robot in robots.items():
                if robot is None:
                    raise RealManSdkError(f"{arm} SDK connection unavailable")
                frame_code, frame = robot.rm_get_current_work_frame()
                if (int(frame_code) != 0 or frame.get("name") != "World"
                        or not self._zero_sdk_frame(frame)):
                    raise RealManSdkError(
                        f"{arm} placement requires zero-offset SDK Work frame World"
                    )
                code, payload = robot.rm_get_force_data()
                if int(code) != 0:
                    raise RealManSdkError(f"{arm} rm_get_force_data returned {code}")
                try:
                    wrench = tuple(float(value) for value in payload["work_zero_force_data"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RealManSdkError(f"{arm} missing work_zero_force_data") from exc
                if len(wrench) != 6 or not all(math.isfinite(value) for value in wrench):
                    raise RealManSdkError(f"{arm} invalid SDK Work wrench")
                result[arm] = wrench
            return result

    def read_bilateral_work_fz(self) -> dict[str, float]:
        """Read compensated vertical force in each controller's default Work frame.

        Placement uses the same SDK connections as arm motion.  No force-control
        mode is enabled and this method sends no motion command.
        """
        with self._motion_lock:
            self._connect()
            robots = dict(zip(("left", "right"), self._robots()))
            result = {}
            for arm, robot in robots.items():
                if robot is None:
                    raise RealManSdkError(f"{arm} SDK connection unavailable")
                frame_code, frame = robot.rm_get_current_work_frame()
                if (int(frame_code) != 0 or frame.get("name") != "World"
                        or not self._zero_sdk_frame(frame)):
                    raise RealManSdkError(
                        f"{arm} placement requires zero-offset SDK Work frame World"
                    )
                code, payload = robot.rm_get_force_data()
                if int(code) != 0:
                    raise RealManSdkError(f"{arm} rm_get_force_data returned {code}")
                try:
                    wrench = [float(value) for value in payload["work_zero_force_data"]]
                except (KeyError, TypeError, ValueError) as exc:
                    raise RealManSdkError(
                        f"{arm} missing work_zero_force_data"
                    ) from exc
                if len(wrench) != 6 or not all(math.isfinite(value) for value in wrench):
                    raise RealManSdkError(f"{arm} invalid SDK Work wrench")
                result[arm] = wrench[2]
            return result

    @staticmethod
    def _zero_sdk_frame(frame: dict) -> bool:
        pose = frame.get("pose", []) if isinstance(frame, dict) else []
        return len(pose) == 6 and max(abs(float(value)) for value in pose) <= 1e-6

    @staticmethod
    def _tool_wrench(robot) -> list[float]:
        return_code, payload = robot.rm_get_force_data()
        if int(return_code) != 0:
            raise RealManSdkError(
                f"rm_get_force_data return_code={int(return_code)}"
            )
        try:
            values = [float(value) for value in payload["tool_zero_force_data"]]
        except Exception as exc:  # noqa: BLE001
            raise RealManSdkError(
                "SDK force payload has no valid tool_zero_force_data"
            ) from exc
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise RealManSdkError(
                "SDK tool_zero_force_data must contain six finite values"
            )
        return values

    def _tool_y_force_start_state(self, arm: str, robot) -> dict:
        tool_code, tool_frame = robot.rm_get_current_tool_frame()
        work_code, work_frame = robot.rm_get_current_work_frame()
        if int(tool_code) != 0 or int(work_code) != 0:
            raise RealManSdkError(
                f"{arm} failed to read Tool/Work frame: "
                f"tool={int(tool_code)}, work={int(work_code)}"
            )
        if tool_frame.get("name") != "Arm_Tip" or not self._zero_sdk_frame(
            tool_frame
        ):
            raise RealManSdkError(
                f"{arm} SDK Tool frame must be zero-offset Arm_Tip"
            )
        if work_frame.get("name") != "World" or not self._zero_sdk_frame(
            work_frame
        ):
            raise RealManSdkError(
                f"{arm} SDK Work frame must be zero-offset World"
            )

        state_code, state = robot.rm_get_current_arm_state()
        if int(state_code) != 0:
            raise RealManSdkError(
                f"{arm} rm_get_current_arm_state return_code={int(state_code)}"
            )
        joints = [float(value) for value in state.get("joint", [])]
        if len(joints) != 7 or not all(math.isfinite(value) for value in joints):
            raise RealManSdkError(f"{arm} SDK joint state is invalid")

        # Do not retransmit the controller's rounded Euler pose. FK from the
        # current joints preserves the exact quaternion near pitch singularity.
        # Connected FK selects controller configuration in process-global SDK
        # state. Serialize it with both arms' offline IK and path inspections.
        with SDK_ALGORITHM_LOCK:
            pose_quaternion = [
                float(value)
                for value in robot.rm_algo_forward_kinematics(joints, flag=0)
            ]
            pose_euler = [
                float(value)
                for value in robot.rm_algo_forward_kinematics(joints, flag=1)
            ]
            if (
                len(pose_quaternion) != 7
                or len(pose_euler) != 6
                or not all(
                    math.isfinite(value) for value in pose_quaternion + pose_euler
                )
            ):
                raise RealManSdkError(f"{arm} SDK FK returned an invalid pose")
            matrix = robot.rm_algo_pos2matrix(pose_euler)
        data = [float(value) for value in matrix.data]
        if len(data) < 12:
            raise RealManSdkError(f"{arm} SDK pose matrix is incomplete")
        tool_y_work = [data[1], data[5], data[9]]
        norm = math.sqrt(sum(value * value for value in tool_y_work))
        if not math.isfinite(norm) or norm <= 1e-9:
            raise RealManSdkError(f"{arm} SDK Tool-Y direction is invalid")
        tool_y_work = [value / norm for value in tool_y_work]
        return {
            "pose": pose_quaternion,
            "tool_y_work": tool_y_work,
            "initial_wrench": self._tool_wrench(robot),
        }

    def _capture_stable_tool_y_baselines(
        self,
        robot_by_arm: dict,
        active_arms: Sequence[str],
        *,
        window_sec: float,
        max_span_n: float,
        timeout_sec: float,
        sample_period_sec: float,
        cancel_requested: Optional[Callable[[], bool]],
    ) -> tuple[dict[str, list[float]], dict[str, float], float, int]:
        """Wait for simultaneous quiet Tool-Y windows before starting motion."""
        sample_count = max(2, math.ceil(window_sec / sample_period_sec) + 1)
        samples = {
            arm: deque(maxlen=sample_count) for arm in active_arms
        }
        spans = {arm: float("inf") for arm in active_arms}
        started_at = time.monotonic()
        while True:
            if cancel_requested is not None and cancel_requested():
                raise RealManSdkCanceled(
                    "mission canceled while waiting for stable Tool-Y baselines"
                )
            elapsed = time.monotonic() - started_at
            if elapsed >= timeout_sec:
                span_detail = ", ".join(
                    f"{arm}={spans[arm]:.3f}N" for arm in active_arms
                )
                raise RealManSdkError(
                    "Tool-Y baselines did not stabilize before force control: "
                    f"elapsed={elapsed:.3f}s, spans=[{span_detail}], "
                    f"limit={max_span_n:.3f}N over {window_sec:.3f}s"
                )

            for arm in active_arms:
                samples[arm].append(self._tool_wrench(robot_by_arm[arm]))
            if all(len(samples[arm]) == sample_count for arm in active_arms):
                spans = {
                    arm: max(row[1] for row in samples[arm])
                    - min(row[1] for row in samples[arm])
                    for arm in active_arms
                }
                if all(spans[arm] <= max_span_n for arm in active_arms):
                    baselines = {
                        arm: [
                            float(median(row[index] for row in samples[arm]))
                            for index in range(6)
                        ]
                        for arm in active_arms
                    }
                    return (
                        baselines,
                        spans,
                        time.monotonic() - started_at,
                        sample_count,
                    )
            time.sleep(min(sample_period_sec, max(0.0, timeout_sec - elapsed)))

    def _confirm_bilateral_tool_y_contact(
        self, robots, baselines, targets, *, duration_sec, period_sec,
        cancel_requested=None,
    ):
        """Require both signed force deltas throughout a stationary interval."""
        started = time.monotonic()
        samples = 0
        while True:
            if cancel_requested is not None and cancel_requested():
                raise RealManSdkCanceled("mission canceled during bilateral contact confirmation")
            deltas = {
                arm: self._tool_wrench(robots[arm])[1] - baselines[arm]
                for arm in ("left", "right")
            }
            for arm, delta in deltas.items():
                target = targets[arm]
                if not math.isfinite(delta) or (delta > target if target < 0 else delta < target):
                    raise RealManSdkError(
                        "bilateral contact not sustained; lift forbidden: "
                        f"{arm} delta_Fy={delta:+.3f}N, required={target:+.3f}N; "
                        f"held={time.monotonic()-started:.3f}s/{duration_sec:.3f}s"
                    )
            samples += 1
            if time.monotonic() - started >= duration_sec and samples >= 3:
                return deltas
            time.sleep(period_sec)

    def execute_tool_y_force_clamp(
        self,
        arms: Sequence[str],
        target_force_n: dict[str, float],
        *,
        speed_mm_s: float,
        max_travel_m: dict[str, float],
        timeout_sec: float,
        control_period_sec: float,
        baseline_stability_window_sec: float,
        baseline_stability_max_span_n: float,
        baseline_stability_timeout_sec: float,
        post_stop_confirmation_sec: float = 0.0,
        post_stop_min_force_n: float = 0.0,
        contact_consecutive_samples: int = 1,
        dual_post_stop_confirmation_sec: float = 0.0,
        contact_min_duration_sec: float = 0.0,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Wait for stable baselines, then stop each arm at first Tool-Y contact."""
        active_arms = tuple(dict.fromkeys(str(arm) for arm in arms))
        if not active_arms or any(arm not in ("left", "right") for arm in active_arms):
            raise RealManSdkError("Tool-Y force clamp requires left and/or right arm")
        targets = {arm: float(target_force_n[arm]) for arm in active_arms}
        if any(not math.isfinite(value) for value in targets.values()):
            raise RealManSdkError("Tool-Y target force must be finite")
        if "left" in targets and targets["left"] >= 0.0:
            raise RealManSdkError("left +Tool-Y target force must be negative")
        if "right" in targets and targets["right"] <= 0.0:
            raise RealManSdkError("right -Tool-Y target force must be positive")
        speed = float(speed_mm_s)
        period = float(control_period_sec)
        timeout = float(timeout_sec)
        baseline_window = float(baseline_stability_window_sec)
        baseline_max_span = float(baseline_stability_max_span_n)
        baseline_timeout = float(baseline_stability_timeout_sec)
        confirmation_sec = float(post_stop_confirmation_sec)
        confirmation_min_force = float(post_stop_min_force_n)
        required_contact_samples = int(contact_consecutive_samples)
        if not math.isfinite(speed) or not 0.1 <= speed <= 10.0:
            raise RealManSdkError("Tool-Y speed must be in [0.1, 10.0] mm/s")
        if not math.isfinite(period) or not 0.01 <= period <= 0.1:
            raise RealManSdkError("Tool-Y control period must be in [0.01, 0.1] s")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise RealManSdkError("Tool-Y timeout must be finite and positive")
        if not math.isfinite(baseline_window) or baseline_window < period:
            raise RealManSdkError(
                "Tool-Y baseline stability window must be at least one sample period"
            )
        if not math.isfinite(baseline_max_span) or baseline_max_span <= 0.0:
            raise RealManSdkError(
                "Tool-Y baseline stability max span must be finite and positive"
            )
        if baseline_max_span >= min(abs(value) for value in targets.values()):
            raise RealManSdkError(
                "Tool-Y baseline stability max span must be below every "
                "active arm's contact threshold"
            )
        if not math.isfinite(baseline_timeout) or baseline_timeout <= baseline_window:
            raise RealManSdkError(
                "Tool-Y baseline stability timeout must exceed the window"
            )
        if (
            not math.isfinite(confirmation_sec)
            or confirmation_sec < 0.0
            or (0.0 < confirmation_sec < 2.0 * period)
        ):
            raise RealManSdkError(
                "Tool-Y post-stop confirmation duration must be zero or at "
                "least two control periods"
            )
        if confirmation_sec > 0.0:
            if active_arms != ("right",):
                raise RealManSdkError(
                    "Tool-Y post-stop confirmation applies to initial right contact only"
                )
            if (
                not math.isfinite(confirmation_min_force)
                or not 0.0 < confirmation_min_force <= targets["right"]
            ):
                raise RealManSdkError(
                    "Tool-Y post-stop confirmation force must be above zero "
                    "and no greater than the right contact target"
                )
        if not 1 <= required_contact_samples <= 10:
            raise RealManSdkError(
                "Tool-Y contact consecutive samples must be in [1, 10]"
            )
        contact_duration = float(contact_min_duration_sec)
        if not math.isfinite(contact_duration) or contact_duration < 0.0:
            raise RealManSdkError("contact duration must be finite and non-negative")
        dual_confirmation = float(dual_post_stop_confirmation_sec)
        if not math.isfinite(dual_confirmation) or dual_confirmation < 0.0:
            raise RealManSdkError("dual contact confirmation duration must be finite and non-negative")
        if dual_confirmation > 0.0 and (
            set(active_arms) != {"left", "right"} or dual_confirmation < 2.0 * period
        ):
            raise RealManSdkError("dual contact confirmation requires both arms and at least two periods")
        travels = {arm: float(max_travel_m[arm]) for arm in active_arms}
        if any(
            not math.isfinite(value) or value <= 0.0 for value in travels.values()
        ):
            raise RealManSdkError("Tool-Y max travel must be finite and positive")

        with self._motion_lock:
            self._connect()
            left_robot, right_robot = self._robots()
            robot_by_arm = {"left": left_robot, "right": right_robot}
            # The prior arm/waist motion can leave a transient Tool-Y reading.
            # No force-position mode or approach command is active while
            # collecting the stable per-arm windows.
            starts = {
                arm: self._tool_y_force_start_state(arm, robot_by_arm[arm])
                for arm in active_arms
            }
            (
                baselines,
                baseline_spans,
                baseline_elapsed,
                baseline_samples,
            ) = self._capture_stable_tool_y_baselines(
                robot_by_arm,
                active_arms,
                window_sec=baseline_window,
                max_span_n=baseline_max_span,
                timeout_sec=baseline_timeout,
                sample_period_sec=period,
                cancel_requested=cancel_requested,
            )
            for arm in active_arms:
                starts[arm]["initial_wrench"] = baselines[arm]
            controller_target_force = {
                arm: starts[arm]["initial_wrench"][1] + targets[arm]
                for arm in active_arms
            }
            self._stop_event.clear()
            done_event = threading.Event()
            failure_event = threading.Event()
            state_lock = threading.Lock()
            barrier = threading.Barrier(len(active_arms) + 1)
            errors: dict[str, Exception] = {}
            latest_wrench = {
                arm: list(starts[arm]["initial_wrench"]) for arm in active_arms
            }
            latest_force_delta = {arm: 0.0 for arm in active_arms}
            reached = {arm: False for arm in active_arms}
            contact_sample_counts = {arm: 0 for arm in active_arms}
            contact_since = {arm: None for arm in active_arms}
            # Once an arm has made contact, latch it.  Re-sending the same
            # force/pose command after contact makes the controller fight the
            # already-contacted box and is the source of the observed chatter.
            contact_force_delta: dict[str, Optional[float]] = {
                arm: None for arm in active_arms
            }
            distances = {arm: 0.0 for arm in active_arms}
            started_at = time.monotonic()

            def target_reached(arm: str, force_delta_y: float) -> bool:
                return (
                    force_delta_y <= targets[arm]
                    if arm == "left"
                    else force_delta_y >= targets[arm]
                )

            def move_one(arm: str) -> None:
                robot = robot_by_arm[arm]
                force_mode_started = False
                approach_sign = 1.0 if arm == "left" else -1.0
                distance = 0.0
                try:
                    start_code = int(robot.rm_start_force_position_move())
                    if start_code != 0:
                        raise RealManSdkError(
                            f"{arm} rm_start_force_position_move "
                            f"return_code={start_code}"
                        )
                    force_mode_started = True
                    barrier.wait(timeout=5.0)
                    next_tick = time.monotonic()
                    previous_tick = next_tick
                    while not done_event.is_set() and not failure_event.is_set():
                        now = time.monotonic()
                        if cancel_requested is not None and cancel_requested():
                            raise RealManSdkCanceled(
                                "mission canceled during Tool-Y force clamp"
                            )
                        elapsed = now - started_at
                        if elapsed >= timeout:
                            raise RealManSdkError(
                                f"Tool-Y force clamp timed out after {timeout:.1f}s"
                            )
                        wrench = self._tool_wrench(robot)
                        force_delta_y = (
                            wrench[1] - starts[arm]["initial_wrench"][1]
                        )
                        arm_reached = target_reached(arm, force_delta_y)
                        with state_lock:
                            latest_wrench[arm] = wrench
                            latest_force_delta[arm] = force_delta_y
                            contact_sample_counts[arm] = (
                                contact_sample_counts[arm] + 1
                                if arm_reached
                                else 0
                            )
                            sampled_at = time.monotonic()
                            if not arm_reached:
                                contact_since[arm] = None
                            elif contact_since[arm] is None:
                                contact_since[arm] = sampled_at
                            if (
                                contact_sample_counts[arm] >= required_contact_samples
                                and contact_since[arm] is not None
                                and sampled_at - contact_since[arm] >= contact_duration
                                and not reached[arm]
                            ):
                                reached[arm] = True
                                contact_force_delta[arm] = force_delta_y
                            is_latched = reached[arm]
                            if all(reached.values()):
                                done_event.set()

                        # A latched arm remains stopped while its partner
                        # finishes; it is never commanded farther inward.
                        if is_latched:
                            if force_mode_started:
                                stop_code = int(robot.rm_stop_force_position_move())
                                if stop_code != 0:
                                    raise RealManSdkError(
                                        f"{arm} rm_stop_force_position_move "
                                        f"return_code={stop_code}"
                                    )
                                force_mode_started = False
                            if done_event.is_set():
                                break
                            next_tick += period
                            delay = next_tick - time.monotonic()
                            if delay > 0.0:
                                time.sleep(delay)
                            else:
                                next_tick = time.monotonic()
                            continue

                        # A delayed SDK/force read must not turn into one large
                        # Cartesian target jump. Drop missed stream periods
                        # instead of trying to catch up in a single packet.
                        dt = min(max(0.0, now - previous_tick), period)
                        previous_tick = now
                        distance += (speed / 1000.0) * dt
                        if distance >= travels[arm]:
                            raise RealManSdkError(
                                f"{arm} reached Tool-Y max travel "
                                f"{travels[arm]:.3f}m without reaching "
                                "baseline-relative target force"
                            )
                        with state_lock:
                            distances[arm] = distance

                        start_pose = starts[arm]["pose"]
                        tool_y = starts[arm]["tool_y_work"]
                        target_pose = list(start_pose)
                        for index in range(3):
                            target_pose[index] += (
                                approach_sign * tool_y[index] * distance
                            )
                        command = self._force_position_move_type(
                            flag=1,
                            pose=target_pose,
                            sensor=1,
                            mode=1,
                            follow=False,
                            control_mode=[0, 7, 0, 0, 0, 0],
                            desired_force=[
                                0.0, controller_target_force[arm], 0.0,
                                0.0, 0.0, 0.0,
                            ],
                            # The SDK limit_vel linear axis is in m/s;
                            # 0.001 m/s is its documented precision, not
                            # the unit of each numeric value.
                            limit_vel=[0.0, speed / 1000.0, 0.0, 0.0, 0.0, 0.0],
                            trajectory_mode=0,
                            radio=0,
                        )
                        command_code = int(robot.rm_force_position_move(command))
                        if command_code != 0:
                            raise RealManSdkError(
                                f"{arm} rm_force_position_move "
                                f"return_code={command_code}"
                            )
                        next_tick += period
                        delay = next_tick - time.monotonic()
                        if delay > 0.0:
                            time.sleep(delay)
                        else:
                            # Do not send burst packets for missed ticks. The
                            # next target increment remains capped above.
                            next_tick = time.monotonic()
                except Exception as exc:  # noqa: BLE001
                    with state_lock:
                        errors[arm] = exc
                    failure_event.set()
                    done_event.set()
                finally:
                    if force_mode_started:
                        try:
                            stop_code = int(robot.rm_stop_force_position_move())
                            if stop_code != 0 and arm not in errors:
                                with state_lock:
                                    errors[arm] = RealManSdkError(
                                        f"{arm} rm_stop_force_position_move "
                                        f"return_code={stop_code}"
                                    )
                                failure_event.set()
                        except Exception as exc:  # noqa: BLE001
                            with state_lock:
                                errors.setdefault(arm, exc)
                            failure_event.set()

            threads = [
                threading.Thread(
                    target=move_one,
                    args=(arm,),
                    name=f"realman-{arm}-tool-y-force",
                )
                for arm in active_arms
            ]
            self._motion_active = True
            for thread in threads:
                thread.start()
            try:
                try:
                    barrier.wait(timeout=5.0)
                except Exception as exc:  # noqa: BLE001
                    failure_event.set()
                    done_event.set()
                    errors.setdefault(
                        "coordinator",
                        RealManSdkError(
                            f"Tool-Y force start synchronization failed: {exc}"
                        ),
                    )
                for thread in threads:
                    thread.join(timeout=timeout + 7.0)
                if any(thread.is_alive() for thread in threads):
                    failure_event.set()
                    done_event.set()
                    errors.setdefault(
                        "coordinator",
                        RealManSdkError("Tool-Y force workers did not stop"),
                    )
            finally:
                self._motion_active = False

            if failure_event.is_set() or errors:
                self.stop_all()
                first = next(iter(errors.values()), RealManSdkError("unknown failure"))
                if isinstance(first, RealManSdkCanceled):
                    raise first
                raise RealManSdkError(f"Tool-Y force clamp failed: {first}") from first
            if dual_confirmation > 0.0:
                try:
                    self._confirm_bilateral_tool_y_contact(
                        robot_by_arm,
                        {arm: starts[arm]["initial_wrench"][1] for arm in active_arms},
                        targets, duration_sec=dual_confirmation, period_sec=period,
                        cancel_requested=cancel_requested,
                    )
                except Exception:
                    self.stop_all()
                    raise
            post_stop_confirm_delta_n = None
            if confirmation_sec > 0.0:
                # Stop at first contact. Verify the force stays above the
                # configured threshold for the full stationary interval;
                # a brief startup spike must not be accepted before Drag1.
                right_robot = robot_by_arm["right"]
                right_baseline_fy = starts["right"]["initial_wrench"][1]
                confirmation_deltas = []
                confirmation_started_at = time.monotonic()
                confirmation_deadline = confirmation_started_at + confirmation_sec
                while True:
                    if cancel_requested is not None and cancel_requested():
                        raise RealManSdkCanceled(
                            "mission canceled during initial right contact confirmation"
                        )
                    delta = self._tool_wrench(right_robot)[1] - right_baseline_fy
                    confirmation_deltas.append(delta)
                    if delta < confirmation_min_force:
                        raise RealManSdkContactNotSustained(
                            "initial right Tool-Y contact was not sustained after "
                            "force control stopped: "
                            f"trigger_delta={contact_force_delta['right']:+.3f}N, "
                            f"settled_delta={delta:+.3f}N, "
                            f"required={confirmation_min_force:.3f}N for "
                            f"{confirmation_sec:.3f}s, "
                            f"held={time.monotonic() - confirmation_started_at:.3f}s, "
                            f"travel={distances['right']:.4f}m",
                            travel_m=distances["right"],
                        )
                    remaining = confirmation_deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    time.sleep(min(period, remaining))
                post_stop_confirm_delta_n = min(confirmation_deltas)
            elapsed = time.monotonic() - started_at
            return {
                "arms": active_arms,
                "target_force_n": targets,
                "baseline_stability_elapsed_sec": baseline_elapsed,
                "baseline_stability_spans_n": baseline_spans,
                "baseline_stability_samples": baseline_samples,
                "controller_target_force_n": controller_target_force,
                "initial_tool_wrench": {
                    arm: list(starts[arm]["initial_wrench"]) for arm in active_arms
                },
                "final_tool_wrench": latest_wrench,
                "force_delta_n": latest_force_delta,
                "contact_force_delta_n": contact_force_delta,
                "post_stop_confirm_delta_n": post_stop_confirm_delta_n,
                "travel_m": distances,
                "elapsed_sec": elapsed,
            }

    def execute_tool_y_force_clamp_retry_initial_right(
        self,
        arms: Sequence[str],
        target_force_n: dict[str, float],
        *,
        max_attempts: int,
        max_travel_m: dict[str, float],
        timeout_sec: float,
        on_retry: Optional[Callable[[int, int, str], None]] = None,
        **kwargs,
    ) -> dict:
        """Retry only a vanished initial right contact, within one motion budget."""
        if tuple(arms) != ("right",):
            raise RealManSdkError(
                "initial right Tool-Y contact retry requires the right arm only"
            )
        attempts = int(max_attempts)
        if not 1 <= attempts <= 3:
            raise RealManSdkError(
                "initial right Tool-Y contact attempts must be in [1, 3]"
            )
        travel_limit = float(max_travel_m["right"])
        total_timeout = float(timeout_sec)
        if not math.isfinite(travel_limit) or travel_limit <= 0.0:
            raise RealManSdkError("initial right Tool-Y travel limit must be positive")
        if not math.isfinite(total_timeout) or total_timeout <= 0.0:
            raise RealManSdkError("initial right Tool-Y timeout must be positive")

        started_at = time.monotonic()
        used_travel_m = 0.0
        failed_contacts = []
        for attempt in range(1, attempts + 1):
            remaining_travel_m = travel_limit - used_travel_m
            remaining_timeout_sec = total_timeout - (time.monotonic() - started_at)
            if remaining_travel_m <= 0.0 or remaining_timeout_sec <= 0.0:
                raise RealManSdkError(
                    "initial right Tool-Y contact exhausted its shared travel "
                    f"or time budget after {attempt - 1} attempt(s): "
                    f"travel={used_travel_m:.4f}/{travel_limit:.4f}m, "
                    f"elapsed={time.monotonic() - started_at:.3f}/{total_timeout:.3f}s"
                )
            try:
                result = self.execute_tool_y_force_clamp(
                    arms,
                    target_force_n,
                    max_travel_m={"right": remaining_travel_m},
                    timeout_sec=remaining_timeout_sec,
                    **kwargs,
                )
            except RealManSdkContactNotSustained as exc:
                used_travel_m += exc.travel_m
                failed_contacts.append(str(exc))
                if attempt == attempts:
                    raise RealManSdkError(
                        f"initial right Tool-Y contact not confirmed after "
                        f"{attempts} attempt(s); total_travel={used_travel_m:.4f}m; "
                        f"last_attempt={exc}"
                    ) from exc
                if on_retry is not None:
                    on_retry(attempt + 1, attempts, str(exc))
                continue

            result["travel_m"]["right"] += used_travel_m
            result["elapsed_sec"] = time.monotonic() - started_at
            result["contact_attempts"] = attempt
            result["failed_contact_checks"] = failed_contacts
            return result

        raise RealManSdkError("initial right Tool-Y contact retry ended unexpectedly")
