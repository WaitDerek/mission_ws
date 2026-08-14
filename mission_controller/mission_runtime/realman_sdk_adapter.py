"""Direct RealMan Python SDK motion backend."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence


class RealManSdkError(RuntimeError):
    """A connection, SDK, or motion-command failure."""


class RealManSdkCanceled(RealManSdkError):
    """The mission was canceled while an SDK motion was active."""


def quaternion_to_rpy(quaternion: Sequence[float]) -> list[float]:
    """Convert an x/y/z/w quaternion to RealMan roll/pitch/yaw radians."""
    if len(quaternion) != 4:
        raise ValueError("quaternion must contain [x, y, z, w]")
    qx, qy, qz, qw = (float(value) for value in quaternion)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion norm must be finite and non-zero")
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def pose_to_sdk_target(pose) -> list[float]:
    """Convert geometry_msgs/Pose to [x, y, z, rx, ry, rz]."""
    position = [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    ]
    if not all(math.isfinite(value) for value in position):
        raise ValueError("pose position contains NaN or Inf")
    orientation = [
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]
    if not all(math.isfinite(value) for value in orientation):
        raise ValueError("pose orientation contains NaN or Inf")
    return position + quaternion_to_rpy(orientation)


class RealManSdkAdapter:
    """Own two RealMan SDK connections and execute one dual-arm motion."""

    def __init__(
        self,
        sdk_root: str,
        left_ip: str,
        right_ip: str,
        port: int,
        connect_level: int,
        logger=None,
    ) -> None:
        self._sdk_root = str(Path(sdk_root).expanduser())
        self._left_ip = str(left_ip).strip()
        self._right_ip = str(right_ip).strip()
        self._port = int(port)
        self._connect_level = int(connect_level)
        self._logger = logger
        self._sdk_lock = threading.RLock()
        self._motion_lock = threading.Lock()
        self._left_robot = None
        self._right_robot = None
        self._robot_type = None
        self._thread_mode_type = None
        self._sdk_loaded = False
        self._stop_event = threading.Event()
        self._motion_active = False

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        try:
            if level == "debug":
                self._logger.debug(message)
            elif level == "info":
                self._logger.info(message)
            elif level == "warning":
                self._logger.warning(message)
            elif level == "error":
                self._logger.error(message)
            else:
                self._logger.info(f"[{level}] {message}")
        except Exception:  # noqa: BLE001
            # Logging must never mask an SDK connection or motion result.
            return

    def _load_sdk(self) -> None:
        if self._sdk_loaded:
            return
        requested = Path(self._sdk_root).expanduser()
        candidates: list[Path] = []
        if requested.is_absolute():
            candidates.append(requested)
        else:
            configured_root = os.environ.get("REALMAN_SDK_ROOT", "").strip()
            if configured_root:
                environment_path = Path(configured_root).expanduser()
                candidates.extend(
                    (environment_path, environment_path / requested)
                )
            candidates.append(Path.cwd() / requested)
            candidates.extend(
                parent / requested for parent in Path(__file__).resolve().parents
            )

        root = next(
            (
                candidate.resolve()
                for candidate in candidates
                if (
                    candidate
                    / "src"
                    / "Robotic_Arm"
                    / "rm_robot_interface.py"
                ).is_file()
            ),
            None,
        )
        if root is None:
            raise RealManSdkError(
                "RealMan SDK root could not be resolved from "
                f"'{self._sdk_root}'; set direct_sdk_root or "
                "REALMAN_SDK_ROOT"
            )
        module_file = root / "src" / "Robotic_Arm" / "rm_robot_interface.py"
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        try:
            from src.Robotic_Arm.rm_robot_interface import (  # type: ignore
                RoboticArm,
                rm_thread_mode_e,
            )
        except Exception as exc:  # noqa: BLE001
            raise RealManSdkError(
                f"failed to import RealMan Python SDK from {root}: {exc}"
            ) from exc
        self._robot_type = RoboticArm
        self._thread_mode_type = rm_thread_mode_e
        self._sdk_loaded = True

    @staticmethod
    def _handle_id(handle) -> int:
        try:
            return int(handle.id)
        except Exception as exc:  # noqa: BLE001
            raise RealManSdkError(
                f"RealMan SDK returned an invalid robot handle: {handle!r}"
            ) from exc

    def _connect(self) -> None:
        with self._sdk_lock:
            if self._left_robot is not None and self._right_robot is not None:
                return
            self._load_sdk()
            left_robot = None
            right_robot = None
            try:
                # Keep UDP ownership with aloha_slave_node so ROS arm-state
                # topics continue receiving the controller's realtime push.
                mode = self._thread_mode_type(1)
                left_robot = self._robot_type(mode)
                left_handle = left_robot.rm_create_robot_arm(
                    self._left_ip, self._port, self._connect_level
                )
                left_id = self._handle_id(left_handle)
                if left_id < 0:
                    raise RealManSdkError(
                        f"left arm connection failed: handle_id={left_id}"
                    )
                # rm_init() is process-global in the RealMan SDK. Only the
                # first RoboticArm instance may receive the thread mode;
                # constructing the second one with mode reinitializes the
                # global logger and raises the SDK severity error.
                right_robot = self._robot_type()
                right_handle = right_robot.rm_create_robot_arm(
                    self._right_ip, self._port, self._connect_level
                )
                right_id = self._handle_id(right_handle)
                if right_id < 0:
                    raise RealManSdkError(
                        f"right arm connection failed: handle_id={right_id}"
                    )
            except Exception:
                for robot in (left_robot, right_robot):
                    if robot is not None:
                        try:
                            robot.rm_delete_robot_arm()
                        except Exception:  # noqa: BLE001
                            pass
                raise
            self._left_robot = left_robot
            self._right_robot = right_robot
            self._log(
                "info",
                f"RealMan SDK connected: left={self._left_ip}, right={self._right_ip}",
            )

    def _robots(self):
        with self._sdk_lock:
            return self._left_robot, self._right_robot

    def stop_all(self) -> None:
        """Request a slow stop for both arms without waiting for motion."""
        self._stop_event.set()
        left_robot, right_robot = self._robots()
        for name, robot in (("left", left_robot), ("right", right_robot)):
            if robot is None:
                continue
            try:
                return_code = int(robot.rm_set_arm_slow_stop())
                self._log(
                    "warning",
                    f"RealMan SDK {name} slow-stop return code={return_code}",
                )
            except Exception as exc:  # noqa: BLE001
                self._log("error", f"RealMan SDK {name} slow-stop failed: {exc}")

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
                        return_code = robot.rm_movej_p(
                            list(target), speed, 0, 0, 1
                        )
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

            if errors or any(
                results.get(name, -1) != 0 for name in ("left", "right")
            ):
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
