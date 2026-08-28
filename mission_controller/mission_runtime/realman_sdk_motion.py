"""RealMan SDK adapter responsibility mixin."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional, Sequence

from .realman_sdk_common import RealManSdkCanceled, RealManSdkError


class RealManSdkMotionMixin:
    """Single/dual MoveL and MoveJ execution."""

    def execute_dual(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
        motion_mode: str,
        speed_percent: float,
        blocking: bool,
        cancel_requested: Optional[Callable[[], bool]] = None,
        timeout_sec: float = 120.0,
    ) -> str:
        mode = str(motion_mode).strip().lower()
        if mode not in ("movel", "movej_p"):
            raise RealManSdkError(f"unsupported RealMan motion mode: {mode}")
        if not blocking:
            raise RealManSdkError(
                "direct Python SDK requires blocking=true so Action success "
                "means both arm motions have completed"
            )
        speed = int(round(float(speed_percent)))
        if speed < 1 or speed > 100:
            raise RealManSdkError(f"SDK speed must be in [1, 100], got {speed}")
        if len(left_target) != 6 or len(right_target) != 6:
            raise RealManSdkError("SDK targets must contain six values")
        if not all(
            math.isfinite(float(value))
            for value in list(left_target) + list(right_target)
        ):
            raise RealManSdkError("SDK target contains NaN or Inf")
        if timeout_sec <= 0.0 or not math.isfinite(float(timeout_sec)):
            raise RealManSdkError("SDK motion timeout must be finite and positive")

        with self._motion_lock:
            self._connect()
            left_robot, right_robot = self._robots()
            self._stop_event.clear()
            failure_event = threading.Event()
            results: dict[str, int] = {}
            errors: dict[str, str] = {}
            barrier = threading.Barrier(3)

            def move_one(name: str, robot, target: Sequence[float]) -> None:
                try:
                    barrier.wait(timeout=5.0)
                    if self._stop_event.is_set():
                        return
                    if mode == "movel":
                        return_code = robot.rm_movel(list(target), speed, 0, 0, 1)
                    else:
                        return_code = robot.rm_movej_p(list(target), speed, 0, 0, 1)
                    results[name] = int(return_code)
                    if int(return_code) != 0:
                        errors[name] = f"return_code={int(return_code)}"
                        failure_event.set()
                except Exception as exc:  # noqa: BLE001
                    errors[name] = str(exc)
                    failure_event.set()

            threads = [
                threading.Thread(
                    target=move_one,
                    args=("left", left_robot, left_target),
                    name="realman-left-motion",
                ),
                threading.Thread(
                    target=move_one,
                    args=("right", right_robot, right_target),
                    name="realman-right-motion",
                ),
            ]
            self._motion_active = True
            for thread in threads:
                thread.start()
            motion_error = None
            stop_requested = False
            try:
                try:
                    barrier.wait(timeout=5.0)
                except Exception as exc:  # noqa: BLE001
                    self.stop_all()
                    motion_error = RealManSdkError(
                        f"direct {mode} start synchronization failed: {exc}"
                    )
                deadline = time.monotonic() + float(timeout_sec)
                while motion_error is None and any(
                    thread.is_alive() for thread in threads
                ):
                    if cancel_requested is not None and cancel_requested():
                        self.stop_all()
                        motion_error = RealManSdkCanceled(
                            "mission canceled during direct Python SDK motion"
                        )
                        break
                    if failure_event.is_set():
                        if not stop_requested:
                            self.stop_all()
                            stop_requested = True
                    if time.monotonic() >= deadline:
                        self.stop_all()
                        motion_error = RealManSdkError(
                            f"direct {mode} timed out after {timeout_sec:.1f}s"
                        )
                        break
                    time.sleep(0.02)
                join_deadline = time.monotonic() + 5.0
                while any(thread.is_alive() for thread in threads):
                    remaining = join_deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    for thread in threads:
                        thread.join(timeout=min(0.05, remaining))
                if any(thread.is_alive() for thread in threads):
                    self.stop_all()
                    if motion_error is None:
                        motion_error = RealManSdkError(
                            f"direct {mode} worker threads did not stop after "
                            "slow-stop"
                        )
            finally:
                self._motion_active = False

            if motion_error is not None:
                raise motion_error

            if errors or any(results.get(name, -1) != 0 for name in ("left", "right")):
                left_detail = errors.get(
                    "left", f"return_code={results.get('left', -1)}"
                )
                right_detail = errors.get(
                    "right", f"return_code={results.get('right', -1)}"
                )
                raise RealManSdkError(
                    f"direct {mode} failed: left {left_detail}; "
                    f"right {right_detail}"
                )
            return (
                f"direct Python SDK {mode} completed: "
                f"left_return_code={results['left']}, "
                f"right_return_code={results['right']}"
            )

    def execute_single(
        self,
        arm: str,
        target: Sequence[float],
        motion_mode: str,
        speed_percent: float,
        blocking: bool,
        cancel_requested: Optional[Callable[[], bool]] = None,
        timeout_sec: float = 120.0,
    ) -> str:
        """Execute a blocking Cartesian motion on exactly one arm."""
        mode = str(motion_mode).strip().lower()
        if arm not in ("left", "right"):
            raise RealManSdkError(f"invalid RealMan arm: {arm}")
        if mode not in ("movel", "movej_p"):
            raise RealManSdkError(f"unsupported RealMan motion mode: {mode}")
        if not blocking:
            raise RealManSdkError(
                "direct Python SDK requires blocking=true so Action success "
                "means the arm motion has completed"
            )
        speed = int(round(float(speed_percent)))
        if speed < 1 or speed > 100:
            raise RealManSdkError(f"SDK speed must be in [1, 100], got {speed}")
        if len(target) != 6 or not all(math.isfinite(float(value)) for value in target):
            raise RealManSdkError("SDK target must contain six finite values")
        if timeout_sec <= 0.0 or not math.isfinite(float(timeout_sec)):
            raise RealManSdkError("SDK motion timeout must be finite and positive")

        with self._motion_lock:
            self._connect()
            left_robot, right_robot = self._robots()
            robot = left_robot if arm == "left" else right_robot
            if robot is None:
                raise RealManSdkError(f"RealMan SDK {arm} connection is unavailable")
            self._stop_event.clear()
            result: dict[str, int] = {}
            errors: dict[str, str] = {}

            def move_one() -> None:
                try:
                    if self._stop_event.is_set():
                        return
                    if mode == "movel":
                        return_code = robot.rm_movel(list(target), speed, 0, 0, 1)
                    else:
                        return_code = robot.rm_movej_p(list(target), speed, 0, 0, 1)
                    result[arm] = int(return_code)
                    if int(return_code) != 0:
                        errors[arm] = f"return_code={int(return_code)}"
                except Exception as exc:  # noqa: BLE001
                    errors[arm] = str(exc)

            thread = threading.Thread(target=move_one, name=f"realman-{arm}-motion")
            self._motion_active = True
            thread.start()
            motion_error = None
            try:
                deadline = time.monotonic() + float(timeout_sec)
                while thread.is_alive():
                    if cancel_requested is not None and cancel_requested():
                        self.stop_arm(arm)
                        motion_error = RealManSdkCanceled(
                            f"mission canceled during direct Python SDK {arm} motion"
                        )
                        break
                    if time.monotonic() >= deadline:
                        self.stop_arm(arm)
                        motion_error = RealManSdkError(
                            f"direct {mode} {arm} timed out after {timeout_sec:.1f}s"
                        )
                        break
                    thread.join(timeout=0.02)
                if motion_error is not None:
                    thread.join(timeout=5.0)
            finally:
                self._motion_active = False
            if motion_error is not None:
                raise motion_error
            if errors or result.get(arm, -1) != 0:
                raise RealManSdkError(
                    f"direct {mode} {arm} failed: "
                    f"{errors.get(arm, f'return_code={result.get(arm, -1)}')}"
                )
            return (
                f"direct Python SDK {arm} {mode} completed: return_code={result[arm]}"
            )

    def execute_single_movej(
        self,
        arm: str,
        joint_degrees: Sequence[float],
        speed_percent: float,
        blend_radius: int = 0,
        trajectory_connect: int = 0,
        cancel_requested: Optional[Callable[[], bool]] = None,
        timeout_sec: float = 120.0,
    ) -> str:
        """Execute a blocking joint-space MoveJ on one SDK connection.

        ``rm_movej`` expects seven joint angles in degrees.  This method is
        intentionally kept on the same adapter and motion lock as
        ``execute_single`` so a DragBox pre-join MoveJ and its following
        MoveJ_P use the same SDK robot handle and command channel.
        """
        if arm not in ("left", "right"):
            raise RealManSdkError(f"invalid RealMan arm: {arm}")
        joints = [float(value) for value in joint_degrees]
        if len(joints) != 7 or not all(math.isfinite(value) for value in joints):
            raise RealManSdkError(
                "SDK MoveJ target must contain seven finite joint angles in degrees"
            )
        speed = int(round(float(speed_percent)))
        if speed < 1 or speed > 100:
            raise RealManSdkError(f"SDK speed must be in [1, 100], got {speed}")
        blend = int(round(float(blend_radius)))
        if blend < 0 or blend > 100:
            raise RealManSdkError(
                f"SDK MoveJ blend radius must be in [0, 100], got {blend}"
            )
        connect = int(round(float(trajectory_connect)))
        if connect not in (0, 1):
            raise RealManSdkError(
                f"SDK MoveJ trajectory_connect must be 0 or 1, got {connect}"
            )
        if timeout_sec <= 0.0 or not math.isfinite(float(timeout_sec)):
            raise RealManSdkError("SDK motion timeout must be finite and positive")

        with self._motion_lock:
            self._connect()
            left_robot, right_robot = self._robots()
            robot = left_robot if arm == "left" else right_robot
            if robot is None:
                raise RealManSdkError(f"RealMan SDK {arm} connection is unavailable")
            self._stop_event.clear()
            result: dict[str, int] = {}
            errors: dict[str, str] = {}

            def move_one() -> None:
                try:
                    if self._stop_event.is_set():
                        return
                    return_code = robot.rm_movej(joints, speed, blend, connect, 1)
                    result[arm] = int(return_code)
                    if int(return_code) != 0:
                        errors[arm] = f"return_code={int(return_code)}"
                except Exception as exc:  # noqa: BLE001
                    errors[arm] = str(exc)

            thread = threading.Thread(target=move_one, name=f"realman-{arm}-movej")
            self._motion_active = True
            thread.start()
            motion_error = None
            try:
                deadline = time.monotonic() + float(timeout_sec)
                while thread.is_alive():
                    if cancel_requested is not None and cancel_requested():
                        self.stop_arm(arm)
                        motion_error = RealManSdkCanceled(
                            f"mission canceled during direct Python SDK {arm} MoveJ"
                        )
                        break
                    if time.monotonic() >= deadline:
                        self.stop_arm(arm)
                        motion_error = RealManSdkError(
                            f"direct MoveJ {arm} timed out after {timeout_sec:.1f}s"
                        )
                        break
                    thread.join(timeout=0.02)
                if motion_error is not None:
                    thread.join(timeout=5.0)
            finally:
                self._motion_active = False
            if motion_error is not None:
                raise motion_error
            if errors or result.get(arm, -1) != 0:
                raise RealManSdkError(
                    f"direct MoveJ {arm} failed: "
                    f"{errors.get(arm, f'return_code={result.get(arm, -1)}')}"
                )
            return f"direct Python SDK {arm} movej completed: return_code={result[arm]}"

    def execute_dual_movel_endpoint(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
        left_speed_percent: float,
        right_speed_percent: float,
        cancel_requested: Optional[Callable[[], bool]] = None,
        timeout_sec: float = 180.0,
        before_start: Optional[Callable[[], object]] = None,
        abort_callback: Optional[Callable[[], object]] = None,
        progress_callback: Optional[Callable[[], object]] = None,
    ) -> str:
        """Execute one synchronized dual-arm SDK MoveL endpoint.

        The body MoveJ is released by ``before_start`` while both arm worker
        threads are waiting at the same barrier.  Each arm receives its own
        configured MoveL speed.  This intentionally sends one endpoint per
        arm (``trajectory_connect=0``) and never queues a connected or
        interpolated path.
        """
        left = list(left_target)
        right = list(right_target)
        if len(left) != 6 or len(right) != 6:
            raise RealManSdkError("endpoint SDK targets must contain six values")
        if not all(math.isfinite(float(value)) for value in left + right):
            raise RealManSdkError("endpoint SDK target contains NaN or Inf")
        left_speed = int(round(float(left_speed_percent)))
        right_speed = int(round(float(right_speed_percent)))
        if not 1 <= left_speed <= 100 or not 1 <= right_speed <= 100:
            raise RealManSdkError(
                "endpoint SDK speeds must be in [1, 100], got "
                f"left={left_speed}, right={right_speed}"
            )
        if timeout_sec <= 0.0 or not math.isfinite(float(timeout_sec)):
            raise RealManSdkError("endpoint SDK timeout must be finite and positive")

        with self._motion_lock:
            self._connect()
            left_robot, right_robot = self._robots()
            if left_robot is None or right_robot is None:
                raise RealManSdkError(
                    "dual-arm endpoint MoveL requires both SDK connections"
                )
            self._stop_event.clear()
            barrier = threading.Barrier(3)
            release_event = threading.Event()
            failure_event = threading.Event()
            results: dict[str, int] = {}
            errors: dict[str, str] = {}

            def move_one(name: str, robot, target: Sequence[float], speed: int) -> None:
                try:
                    barrier.wait(timeout=5.0)
                    release_event.wait(timeout=5.0)
                    if self._stop_event.is_set():
                        return
                    code = int(robot.rm_movel(list(target), speed, 0, 0, 1))
                    results[name] = code
                    if code != 0:
                        errors[name] = f"return_code={code}"
                        failure_event.set()
                except Exception as exc:  # noqa: BLE001
                    errors[name] = str(exc)
                    failure_event.set()

            threads = [
                threading.Thread(
                    target=move_one,
                    args=("left", left_robot, left, left_speed),
                    name="realman-endpoint-left",
                ),
                threading.Thread(
                    target=move_one,
                    args=("right", right_robot, right, right_speed),
                    name="realman-endpoint-right",
                ),
            ]
            self._motion_active = True
            for thread in threads:
                thread.start()
            motion_error = None
            abort_called = False

            def stop_after_failure() -> None:
                nonlocal abort_called
                if abort_called:
                    return
                abort_called = True
                self.stop_all()
                if abort_callback is not None:
                    try:
                        abort_callback()
                    except Exception as exc:  # noqa: BLE001
                        self._log(
                            "error", f"endpoint body abort callback failed: {exc}"
                        )

            try:
                try:
                    barrier.wait(timeout=5.0)
                    if before_start is not None:
                        before_start()
                    release_event.set()
                except Exception as exc:  # noqa: BLE001
                    release_event.set()
                    stop_after_failure()
                    motion_error = RealManSdkError(
                        f"endpoint MoveL start synchronization failed: {exc}"
                    )
                deadline = time.monotonic() + float(timeout_sec)
                while motion_error is None and any(
                    thread.is_alive() for thread in threads
                ):
                    if cancel_requested is not None and cancel_requested():
                        stop_after_failure()
                        motion_error = RealManSdkCanceled(
                            "mission canceled during endpoint dual-arm MoveL"
                        )
                        break
                    if failure_event.is_set():
                        stop_after_failure()
                        worker_detail = "; ".join(
                            f"{name}={errors.get(name, results.get(name, 'unknown'))}"
                            for name in ("left", "right")
                        )
                        motion_error = RealManSdkError(
                            "endpoint dual-arm MoveL worker failed: " + worker_detail
                        )
                        break
                    if time.monotonic() >= deadline:
                        stop_after_failure()
                        motion_error = RealManSdkError(
                            f"endpoint dual-arm MoveL timed out after {timeout_sec:.1f}s"
                        )
                        break
                    time.sleep(0.02)
                join_deadline = time.monotonic() + 5.0
                while any(thread.is_alive() for thread in threads):
                    remaining = join_deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    for thread in threads:
                        thread.join(timeout=min(0.05, remaining))
                if any(thread.is_alive() for thread in threads):
                    stop_after_failure()
                    if motion_error is None:
                        motion_error = RealManSdkError(
                            "endpoint dual-arm MoveL workers did not stop"
                        )
            finally:
                self._motion_active = False

            if motion_error is not None:
                raise motion_error
            if errors or set(results) != {"left", "right"}:
                stop_after_failure()
                raise RealManSdkError(
                    "endpoint dual-arm MoveL failed: "
                    + "; ".join(
                        f"{name}={errors.get(name, results.get(name, 'missing'))}"
                        for name in ("left", "right")
                    )
                )
            return (
                "direct Python SDK endpoint dual-arm MoveL completed: "
                f"left_speed={left_speed}, right_speed={right_speed}, "
                "trajectory_connect=0"
            )
