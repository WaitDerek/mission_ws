"""RealMan SDK adapter responsibility mixin."""

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
from pathlib import Path

from .realman_sdk_common import RealManSdkError
from .realman_sdk_algorithm import ControllerAlgorithmProfile, SDK_ALGORITHM_LOCK


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
        self._ik_params_type = None
        self._ik_matrix_type = None
        self._force_position_move_type = None
        self._algo_type = None
        self._arm_model_type = None
        self._force_type = None
        self._dh_type = None
        self._frame_type = None
        self._algo_profiles = {}
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
                    Algo,
                    rm_Mat_t,
                    rm_dh_t,
                    rm_frame_t,
                    rm_robot_arm_model_e,
                    rm_force_type_e,
                    rm_inverse_kinematics_params_t,
                    rm_force_position_move_t,
                    rm_thread_mode_e,
                )
            except ModuleNotFoundError:
                from Robotic_Arm.rm_robot_interface import (  # type: ignore
                    RoboticArm,
                    Algo,
                    rm_Mat_t,
                    rm_dh_t,
                    rm_frame_t,
                    rm_robot_arm_model_e,
                    rm_force_type_e,
                    rm_inverse_kinematics_params_t,
                    rm_force_position_move_t,
                    rm_thread_mode_e,
                )
        except Exception as exc:  # noqa: BLE001
            raise RealManSdkError(
                f"failed to import RealMan Python SDK from {root}: {exc}"
            ) from exc
        self._robot_type = RoboticArm
        self._algo_type = Algo
        self._arm_model_type = rm_robot_arm_model_e
        self._force_type = rm_force_type_e
        self._dh_type = rm_dh_t
        self._frame_type = rm_frame_t
        self._thread_mode_type = rm_thread_mode_e
        self._ik_params_type = rm_inverse_kinematics_params_t
        self._ik_matrix_type = rm_Mat_t
        self._force_position_move_type = rm_force_position_move_t
        self._sdk_loaded = True

    def _activate_arm_algorithm(self, arm: str, robot):
        """Reapply a complete controller profile under SDK_ALGORITHM_LOCK.

        Connected-handle FK/IK load controller configuration into shared state,
        so even consecutive calls for one arm must reapply the cached profile.
        """
        profile = self._algo_profiles.get(arm)
        if profile is None:
            profile = ControllerAlgorithmProfile.read(
                arm, robot, self._arm_model_type, self._force_type,
                self._dh_type, self._frame_type,
            )
            self._algo_profiles[arm] = profile
            self._log("info", f"SDK {arm} offline profile loaded from controller: "
                      f"installation_deg={profile.angle}; joint_min_deg={profile.lower}; "
                      f"joint_max_deg={profile.upper}; model/DH/Tool/Work synchronized")
        return profile.activate(self._algo_type)

    def solve_ik(
        self,
        arm: str,
        target_pose_euler,
        seed_joint_deg,
    ) -> list[float] | None:
        """Solve one 7-DOF target without sending any motion command.

        The connected arm handle lets the SDK use that controller's actual
        kinematic parameters.  This method only calls the local algorithm
        interface; it never calls MoveJ, MoveJ_P, MoveL, or MoveL_Offset.
        """
        if arm not in ("left", "right"):
            raise RealManSdkError(f"invalid arm for offline IK: {arm}")
        target = [float(value) for value in target_pose_euler]
        seed = [float(value) for value in seed_joint_deg]
        if len(target) != 6 or len(seed) != 7:
            raise RealManSdkError(
                "offline IK requires [x,y,z,rx,ry,rz] and seven seed joints"
            )
        if not all(math.isfinite(value) for value in target + seed):
            raise RealManSdkError("offline IK input contains NaN or Inf")
        with SDK_ALGORITHM_LOCK, self._sdk_lock:
            self._connect()
            left_robot, right_robot = self._robots()
            robot = left_robot if arm == "left" else right_robot
            self._activate_arm_algorithm(arm, robot)
            params = self._ik_params_type(seed, target, 1)
            return_code, solution = robot.rm_algo_inverse_kinematics(params)
        if int(return_code) != 0:
            return None
        result = [float(value) for value in solution]
        if len(result) != 7 or not all(math.isfinite(value) for value in result):
            raise RealManSdkError(
                f"offline 7DOF IK returned {len(result)} joints for {arm}"
            )
        return result

    def ik_pose_residual(self, arm, target_pose_euler, joint_deg):
        """Return offline FK position (m) and rotation (rad) errors; no motion."""
        from scipy.spatial.transform import Rotation

        if arm not in ("left", "right"):
            raise RealManSdkError(f"invalid arm for offline FK: {arm}")
        with SDK_ALGORITHM_LOCK, self._sdk_lock:
            self._connect()
            robot = self._robots()[0 if arm == "left" else 1]
            self._activate_arm_algorithm(arm, robot)
            actual = list(robot.rm_algo_forward_kinematics(joint_deg, flag=1))
        if len(actual) != 6 or not all(math.isfinite(v) for v in actual):
            return math.inf, math.inf
        target = list(target_pose_euler)
        position_error = math.dist(actual[:3], target[:3])
        rotation_error = (
            Rotation.from_euler("xyz", actual[3:]).inv()
            * Rotation.from_euler("xyz", target[3:])
        ).magnitude()
        return position_error, float(rotation_error)

    def inspect_joint_path(
        self,
        arm: str,
        start_joint_deg,
        target_joint_deg,
        joint_min_deg,
        joint_max_deg,
        *,
        max_sample_step_deg: float = 2.0,
        singularity_limit: float = 0.01,
    ) -> str:
        """Check a sampled joint-space segment without commanding the arm.

        The SDK checks singularity and robot self-collision. This does not
        check the carried box, fixtures, or other obstacles in the room.
        """
        if arm not in ("left", "right"):
            raise RealManSdkError(f"invalid arm for joint path inspection: {arm}")
        vectors = [
            [float(value) for value in vector]
            for vector in (
                start_joint_deg, target_joint_deg, joint_min_deg, joint_max_deg
            )
        ]
        if any(len(vector) != 7 for vector in vectors) or not all(
            math.isfinite(value) for vector in vectors for value in vector
        ):
            raise RealManSdkError("joint path inspection requires four finite 7-joint arrays")
        if (
            not math.isfinite(max_sample_step_deg)
            or max_sample_step_deg <= 0.0
            or not math.isfinite(singularity_limit)
            or not 0.0 < singularity_limit < 1.0
        ):
            raise RealManSdkError("joint path inspection thresholds are invalid")
        start, target, lower, upper = vectors
        steps = max(
            1,
            math.ceil(
                max(abs(end - begin) for begin, end in zip(start, target))
                / max_sample_step_deg
            ),
        )
        with SDK_ALGORITHM_LOCK, self._sdk_lock:
            self._connect()
            left_robot, right_robot = self._robots()
            robot = left_robot if arm == "left" else right_robot
            self._activate_arm_algorithm(arm, robot)
            for index in range(steps + 1):
                fraction = index / steps
                sample = [
                    begin + (end - begin) * fraction
                    for begin, end in zip(start, target)
                ]
                location = f"sample={index}/{steps}, fraction={fraction:.3f}"
                for joint_index, (value, minimum, maximum) in enumerate(
                    zip(sample, lower, upper), start=1
                ):
                    if minimum > maximum or value < minimum or value > maximum:
                        raise RealManSdkError(
                            f"{arm} joint path limit at {location}: "
                            f"joint{joint_index}={value:.3f}deg, "
                            f"allowed=[{minimum:.3f},{maximum:.3f}]deg"
                        )
                singularity = int(
                    robot.rm_algo_universal_singularity_analyse(
                        sample, singularity_limit
                    )
                )
                if singularity != 0:
                    raise RealManSdkError(
                        f"{arm} joint path singularity check at {location}: "
                        f"return_code={singularity}, joint_deg="
                        f"{[round(value, 3) for value in sample]}"
                    )
                collision = int(
                    robot.rm_algo_safety_robot_self_collision_detection(sample)
                )
                if collision != 0:
                    raise RealManSdkError(
                        f"{arm} joint path self-collision check at {location}: "
                        f"return_code={collision}, joint_deg="
                        f"{[round(value, 3) for value in sample]}"
                    )
        return (
            f"{arm} joint path offline checks passed: samples={steps + 1}, "
            f"max_step_deg={max_sample_step_deg:.3f}, "
            f"singularity_limit={singularity_limit:.3f}"
        )

    def solve_continuous_ik(
        self,
        arm: str,
        target_poses_euler,
        seed_joint_deg,
        dt_sec: float,
    ) -> list[list[float]] | None:
        """Solve one Cartesian pose sequence without dispatching motion.

        ``rm_algo_ik_remote`` is the SDK solver intended for continuous
        Cartesian poses.  The complete sequence is solved while holding the
        process-global SDK lock because its initialization and solver state
        are shared by all connected handles.
        """
        if arm not in ("left", "right"):
            raise RealManSdkError(f"invalid arm for continuous IK: {arm}")
        seed = [float(value) for value in seed_joint_deg]
        targets = [
            [float(value) for value in target]
            for target in target_poses_euler
        ]
        dt = float(dt_sec)
        if len(seed) != 7 or not all(math.isfinite(value) for value in seed):
            raise RealManSdkError("continuous IK requires seven finite seed joints")
        if not math.isfinite(dt) or dt <= 0.0:
            raise RealManSdkError("continuous IK dt_sec must be positive")
        if any(
            len(target) != 6
            or not all(math.isfinite(value) for value in target)
            for target in targets
        ):
            raise RealManSdkError(
                "continuous IK targets must be finite [x,y,z,rx,ry,rz] poses"
            )
        if not targets:
            return []

        with SDK_ALGORITHM_LOCK, self._sdk_lock:
            self._connect()
            left_robot, right_robot = self._robots()
            robot = left_robot if arm == "left" else right_robot
            self._activate_arm_algorithm(arm, robot)
            robot.rm_algo_ik_remote_init(dt, 1)
            previous = seed
            solutions: list[list[float]] = []
            for index, target in enumerate(targets):
                compact_matrix = robot.rm_algo_pos2matrix(target)
                matrix_rows = [
                    [
                        float(compact_matrix.data[row * 4 + column])
                        for column in range(4)
                    ]
                    for row in range(4)
                ]
                target_matrix = self._ik_matrix_type(4, 4, matrix_rows)
                output = (ctypes.c_float * 7)()
                return_code = int(
                    robot.rm_algo_ik_remote(target_matrix, previous, output)
                )
                if return_code != 0:
                    self._log(
                        "debug",
                        f"continuous IK rejected {arm} sample {index + 1}/"
                        f"{len(targets)}: return_code={return_code}",
                    )
                    return None
                solution = [float(value) for value in output]
                if len(solution) != 7 or not all(
                    math.isfinite(value) for value in solution
                ):
                    raise RealManSdkError(
                        f"continuous IK returned invalid 7DOF solution for {arm}"
                    )
                solutions.append(solution)
                previous = solution
        return solutions

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
            # Controller calibration/frames may have changed since disconnect.
            self._algo_profiles.clear()
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
