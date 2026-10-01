"""Controller-derived profiles for RealMan's process-global offline algorithms.

Only rm_algo_* setters are used: these change local algorithm state, never
the controller's calibration. Profiles are cached for one SDK connection;
reconnect/restart Mission after changing controller calibration or frames.
"""

from dataclasses import dataclass
import math
import threading

from .realman_sdk_common import RealManSdkError


# Connected-handle FK/IK also overwrite native algorithm state. Every Mission
# algorithm call (including force-start FK) must share this process-wide lock.
SDK_ALGORITHM_LOCK = threading.RLock()


def _vector(values, length, label):
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise RealManSdkError(f"invalid {label}") from exc
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise RealManSdkError(f"{label} requires {length} finite values")
    return result


def _read(result, label):
    code, data = result
    if int(code) != 0:
        raise RealManSdkError(f"cannot read {label}: return_code={code}")
    return data


@dataclass(frozen=True)
class ControllerAlgorithmProfile:
    model: object
    force: object
    dh: object
    angle: list
    lower: list
    upper: list
    tool: object
    work: object

    @classmethod
    def read(cls, arm, robot, model_type, force_type, dh_type, frame_type):
        """Fail closed on missing/invalid controller data; no factory fallback."""
        info = _read(robot.rm_get_robot_info(), f"{arm} robot info")
        model_names = {"RXL75": "RM_MODEL_RXL75_E", "RXR75": "RM_MODEL_RXR75_E"}
        force_names = {
            "B": "RM_MODEL_RM_B_E", "ZF": "RM_MODEL_RM_ZF_E",
            "6F": "RM_MODEL_RM_SF_E", "6FB": "RM_MODEL_RM_ISF_E",
            "B-V": "RM_MODEL_RM_BV_E", "6FB-V": "RM_MODEL_RM_ISFV_E",
        }
        model = getattr(model_type, model_names.get(info.get("arm_model"), ""), None)
        force = getattr(force_type, force_names.get(info.get("force_type"), ""), None)
        if model is None or force is None or int(info.get("arm_dof", 0)) != 7:
            raise RealManSdkError(f"unsupported {arm} offline algorithm profile: {info}")

        dh = _read(robot.rm_get_DH_data(), f"{arm} controller DH")
        dh = dh_type(**{
            key: _vector(dh.get(key), 8, f"{arm} DH {key}")
            for key in ("d", "a", "alpha", "offset")
        })
        install = robot.rm_get_install_pose()
        if int(install.get("return_code", -1)) != 0:
            raise RealManSdkError(f"cannot read {arm} installation angle: {install}")
        angle = _vector([install.get(k) for k in ("x", "y", "z")], 3,
                        f"{arm} installation angle (degrees)")
        lower = _vector(_read(robot.rm_get_joint_min_pos(), f"{arm} min limits"),
                        7, f"{arm} min limits")
        upper = _vector(_read(robot.rm_get_joint_max_pos(), f"{arm} max limits"),
                        7, f"{arm} max limits")
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise RealManSdkError(f"{arm} controller joint limits are invalid")

        def frame(result, label):
            data = _read(result, f"{arm} {label}")
            # The algorithm only needs pose; name/payload/COM do not enter FK.
            return frame_type(pose=_vector(data.get("pose"), 6, f"{arm} {label}"))

        tool = frame(robot.rm_get_current_tool_frame(), "Tool frame")
        work = frame(robot.rm_get_current_work_frame(), "Work frame")
        return cls(model, force, dh, angle, lower, upper, tool, work)

    def activate(self, algo_type):
        """Caller holds SDK_ALGORITHM_LOCK through the subsequent calculation."""
        # Passing dh to Algo.__init__ silently selects RM_MODEL_UNIVERSAL_E.
        # Keep the actual arm model, then apply its calibrated DH explicitly.
        algo = algo_type(self.model, self.force)
        algo.rm_algo_set_dh(self.dh)
        algo.rm_algo_set_angle(*self.angle)
        algo.rm_algo_set_joint_min_limit(self.lower)
        algo.rm_algo_set_joint_max_limit(self.upper)
        algo.rm_algo_set_toolframe(self.tool)
        algo.rm_algo_set_workframe(self.work)
        return algo
