"""RealMan SDK adapter responsibility mixin."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional, Sequence

from .realman_sdk_common import RealManSdkCanceled, RealManSdkError


class RealManSdkConnectedMotionMixin:
    """Connected waypoint execution and adapter shutdown."""

    def execute_dual_movel_connected_waypoints(
        self,
        left_targets: Sequence[Sequence[float]],
        right_targets: Sequence[Sequence[float]],
        left_speed_percent: float,
        right_speed_percent: float,
        blend_radius: int = 0,
        cancel_requested: Optional[Callable[[], bool]] = None,
        timeout_sec: float = 180.0,
        before_start: Optional[Callable[[], object]] = None,
        abort_callback: Optional[Callable[[], object]] = None,
        progress_callback: Optional[Callable[[], object]] = None,
    ) -> str:
        """Queue a connected dual-arm MoveL path and start its final point.

        RealMan's ``trajectory_connect=1`` queues an intermediate trajectory
        without executing it.  The final point is submitted with
        ``trajectory_connect=0`` and starts the queued path.  The two final
        arm commands are released together behind a barrier; ``before_start``
        can submit a matching waist command before the release.  This method
        intentionally does not wait at intermediate waypoints.
        """
        left = [list(target) for target in left_targets]
        right = [list(target) for target in right_targets]
        if not left or len(left) != len(right):
            raise RealManSdkError(
                "connected dual-arm MoveL requires equally sized non-empty paths"
            )
        if any(
            len(target) != 6 or not all(math.isfinite(float(value)) for value in target)
            for target in left + right
        ):
            raise RealManSdkError(
                "connected dual-arm MoveL targets must contain six finite values"
            )
        left_speed = int(round(float(left_speed_percent)))
        right_speed = int(round(float(right_speed_percent)))
        if not 1 <= left_speed <= 100 or not 1 <= right_speed <= 100:
            raise RealManSdkError(
                "SDK speeds must be in [1, 100], got "
                f"left={left_speed}, right={right_speed}"
            )
        radius = int(round(float(blend_radius)))
        if not 0 <= radius <= 100:
            raise RealManSdkError(f"SDK blend_radius must be in [0, 100], got {radius}")
        if timeout_sec <= 0.0 or not math.isfinite(float(timeout_sec)):
            raise RealManSdkError("SDK motion timeout must be finite and positive")

        with self._motion_lock:
            self._connect()
            left_robot, right_robot = self._robots()
            if left_robot is None or right_robot is None:
                raise RealManSdkError(
                    "connected dual-arm MoveL requires both SDK connections"
                )
            self._stop_event.clear()
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
                            "error",
                            f"connected endpoint abort callback failed: {exc}",
                        )

            self._motion_active = True
            try:
                # Queue all intermediate points before either arm is allowed
                # to execute the final point.  Connect=1 returns immediately
                # in the SDK's multi-thread mode.
                for point_index in range(len(left) - 1):
                    if cancel_requested is not None and cancel_requested():
                        stop_after_failure()
                        raise RealManSdkCanceled(
                            "mission canceled while queuing connected MoveL"
                        )
                    left_code = int(
                        left_robot.rm_movel(left[point_index], left_speed, radius, 1, 1)
                    )
                    if left_code != 0:
                        raise RealManSdkError(
                            "connected left-arm MoveL queue failed: "
                            f"waypoint={point_index + 1}, return_code={left_code}"
                        )
                    right_code = int(
                        right_robot.rm_movel(
                            right[point_index], right_speed, radius, 1, 1
                        )
                    )
                    if right_code != 0:
                        raise RealManSdkError(
                            "connected right-arm MoveL queue failed: "
                            f"waypoint={point_index + 1}, return_code={right_code}"
                        )

                barrier = threading.Barrier(3)
                release_event = threading.Event()
                failure_event = threading.Event()
                results: dict[str, int] = {}
                errors: dict[str, str] = {}

                def move_final(
                    name: str, robot, target: Sequence[float], speed: int
                ) -> None:
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
                        target=move_final,
                        args=("left", left_robot, left[-1], left_speed),
                        name="realman-connected-final-left",
                    ),
                    threading.Thread(
                        target=move_final,
                        args=("right", right_robot, right[-1], right_speed),
                        name="realman-connected-final-right",
                    ),
                ]
                for thread in threads:
                    thread.start()

                motion_error = None
                try:
                    barrier.wait(timeout=5.0)
                    if before_start is not None:
                        before_start()
                    release_event.set()
                except Exception as exc:  # noqa: BLE001
                    release_event.set()
                    stop_after_failure()
                    motion_error = RealManSdkError(
                        f"connected MoveL start synchronization failed: {exc}"
                    )

                deadline = time.monotonic() + float(timeout_sec)
                while motion_error is None and any(
                    thread.is_alive() for thread in threads
                ):
                    if cancel_requested is not None and cancel_requested():
                        stop_after_failure()
                        motion_error = RealManSdkCanceled(
                            "mission canceled during connected dual-arm MoveL"
                        )
                        break
                    if failure_event.is_set():
                        stop_after_failure()
                        detail = "; ".join(
                            f"{name}={errors.get(name, results.get(name, 'unknown'))}"
                            for name in ("left", "right")
                        )
                        motion_error = RealManSdkError(
                            "connected dual-arm MoveL worker failed: " + detail
                        )
                        break
                    if progress_callback is not None:
                        try:
                            progress_callback()
                        except Exception as exc:  # noqa: BLE001
                            stop_after_failure()
                            motion_error = RealManSdkError(
                                "connected dual-arm MoveL progress check failed: "
                                f"{exc}"
                            )
                            break
                    if time.monotonic() >= deadline:
                        stop_after_failure()
                        motion_error = RealManSdkError(
                            "connected dual-arm MoveL timed out after "
                            f"{timeout_sec:.1f}s"
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
                            "connected dual-arm MoveL workers did not stop"
                        )

                if motion_error is not None:
                    raise motion_error
                if errors or set(results) != {"left", "right"}:
                    stop_after_failure()
                    raise RealManSdkError(
                        "connected dual-arm MoveL failed: "
                        + "; ".join(
                            f"{name}={errors.get(name, results.get(name, 'missing'))}"
                            for name in ("left", "right")
                        )
                    )
            except (RealManSdkCanceled, RealManSdkError):
                stop_after_failure()
                raise
            except Exception as exc:  # noqa: BLE001
                stop_after_failure()
                raise RealManSdkError(
                    f"connected dual-arm MoveL setup failed: {exc}"
                ) from exc
            finally:
                self._motion_active = False

            return (
                "direct Python SDK connected dual-arm MoveL completed: "
                f"waypoints={len(left)}, left_speed={left_speed}, "
                f"right_speed={right_speed}, blend_radius={radius}, "
                "trajectory_connect=1..1,0"
            )

    def close(self) -> None:
        self.stop_all()
        with self._sdk_lock:
            for robot in (self._left_robot, self._right_robot):
                if robot is None:
                    continue
                try:
                    robot.rm_delete_robot_arm()
                except Exception as exc:  # noqa: BLE001
                    self._log("error", f"RealMan SDK disconnect failed: {exc}")
            self._left_robot = None
            self._right_robot = None
            self._motion_active = False
