"""RealMan SDK adapter responsibility mixin."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from .realman_sdk_common import RealManSdkError


class RealManSdkConnectionMixin:
    """SDK loading, connection lifetime, and stop operations."""

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
                candidates.extend((environment_path, environment_path / requested))
            candidates.append(Path.cwd() / requested)
            candidates.extend(
                parent / requested for parent in Path(__file__).resolve().parents
            )

        root = None
        for candidate in candidates:
            candidate = candidate.resolve()
            # Demo workspaces contain src/Robotic_Arm, while the consolidated
            # SDK package contains Robotic_Arm directly.
            if (candidate / "src" / "Robotic_Arm" / "rm_robot_interface.py").is_file():
                root = candidate
                break
            if (candidate / "Robotic_Arm" / "rm_robot_interface.py").is_file():
                root = candidate
                break
        if root is None:
            raise RealManSdkError(
                "RealMan SDK root could not be resolved from "
                f"'{self._sdk_root}'; set direct_sdk_root or "
                "REALMAN_SDK_ROOT"
            )
        module_root = root / "src" if (root / "src").is_dir() else root
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        if str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))
        try:
            try:
                from src.Robotic_Arm.rm_robot_interface import (  # type: ignore
                    RoboticArm,
                    rm_thread_mode_e,
                )
            except ModuleNotFoundError:
                from Robotic_Arm.rm_robot_interface import (  # type: ignore
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
            self._stop_robot(name, robot)

    def _stop_robot(self, name: str, robot) -> None:
        if robot is None:
            return
        try:
            return_code = int(robot.rm_set_arm_slow_stop())
            self._log(
                "warning",
                f"RealMan SDK {name} slow-stop return code={return_code}",
            )
        except Exception as exc:  # noqa: BLE001
            self._log("error", f"RealMan SDK {name} slow-stop failed: {exc}")

    def stop_arm(self, arm: str) -> None:
        """Request a slow stop for one arm without commanding the other arm."""
        if arm not in ("left", "right"):
            raise RealManSdkError(f"invalid arm for slow-stop: {arm}")
        self._stop_event.set()
        left_robot, right_robot = self._robots()
        self._stop_robot(arm, left_robot if arm == "left" else right_robot)
