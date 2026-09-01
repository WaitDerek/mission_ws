#!/usr/bin/env python3
"""Interactive real-robot calibration for TF box target corrections.

The script calibrates exactly one selected layer.  It drives the selected TF
action only after an explicit confirmation, disables all motion after the
initial MoveJ_P, captures the frozen box pose and live arm-base TF, and uses
manually taught ideal Link8 poses to solve the absolute correction in the
FoundationPose box frame.  No other layer is modified.

Run only after sourcing setup_mission_env.sh.
"""

from __future__ import annotations

import argparse
import ast
import math
import subprocess
import sys
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rm_robot_interfaces.msg import ArmSlaveData
from tf2_ros import Buffer, TransformException, TransformListener


MISSION_NODE = "/mission_controller"
IDENTITY_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _normalize_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    q = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in q))
    if len(q) != 4 or norm <= 1e-12 or not all(math.isfinite(value) for value in q):
        raise ValueError(f"invalid quaternion: {values}")
    q = tuple(value / norm for value in q)
    # Canonicalize only for stable display. q and -q are the same rotation.
    if q[3] < 0.0:
        q = tuple(-value for value in q)
    return q


def _quaternion_conjugate(q: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = _normalize_quaternion(q)
    return (-x, -y, -z, w)


def _quaternion_multiply(
    lhs: Sequence[float], rhs: Sequence[float]
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return _normalize_quaternion(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def _rotate_vector(vector: Sequence[float], quaternion: Sequence[float]):
    x, y, z, w = _normalize_quaternion(quaternion)
    vx, vy, vz = (float(value) for value in vector)
    # Expanded q * v * q^-1 avoids normalizing the pure-vector quaternion.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _pose_values(pose: Pose) -> tuple[float, ...]:
    return (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )


def _format_array(values: Iterable[float], precision: int = 9) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(command, text=True, capture_output=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"command failed: {' '.join(command)}\n{detail}")
    return result


def _param_get(name: str):
    output = _run(["ros2", "param", "get", MISSION_NODE, name]).stdout.strip()
    if "array(" in output:
        start = output.find("[")
        end = output.rfind("]")
        if start < 0 or end < start:
            raise RuntimeError(f"cannot parse parameter {name}: {output}")
        return list(ast.literal_eval(output[start : end + 1]))
    value = output.split(":", 1)[-1].strip()
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _param_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def _param_set(name: str, value) -> None:
    result = _run(
        ["ros2", "param", "set", MISSION_NODE, name, _param_text(value)],
        check=False,
    )
    combined = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0 or "successful" not in combined.lower():
        raise RuntimeError(f"failed to set {name}: {combined}")


class CalibrationObserver(Node):
    def __init__(self) -> None:
        super().__init__("tf_target_correction_calibrator")
        qos = QoSProfile(depth=20)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE
        self.lock = threading.Lock()
        self.raw_box_pose: PoseStamped | None = None
        self.raw_box_received_monotonic = 0.0
        self.arm_data: dict[str, ArmSlaveData | None] = {"left": None, "right": None}
        self.arm_received_monotonic = {"left": 0.0, "right": 0.0}
        self.create_subscription(
            PoseStamped,
            "/mission/box_object_pose_raw",
            self._raw_box_callback,
            qos,
        )
        self.create_subscription(
            ArmSlaveData,
            "/mcap/slave_arm_left",
            lambda message: self._arm_callback("left", message),
            qos,
        )
        self.create_subscription(
            ArmSlaveData,
            "/mcap/slave_arm_right",
            lambda message: self._arm_callback("right", message),
            qos,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _raw_box_callback(self, message: PoseStamped) -> None:
        with self.lock:
            self.raw_box_pose = deepcopy(message)
            self.raw_box_received_monotonic = time.monotonic()

    def _arm_callback(self, arm: str, message: ArmSlaveData) -> None:
        with self.lock:
            self.arm_data[arm] = deepcopy(message)
            self.arm_received_monotonic[arm] = time.monotonic()

    def clear_raw_box(self) -> None:
        with self.lock:
            self.raw_box_pose = None
            self.raw_box_received_monotonic = 0.0

    def box_snapshot(self) -> PoseStamped:
        with self.lock:
            if self.raw_box_pose is None:
                raise RuntimeError("no fresh /mission/box_object_pose_raw was received")
            return deepcopy(self.raw_box_pose)

    def arm_snapshot(self, max_age_sec: float = 1.0) -> dict[str, Pose]:
        now = time.monotonic()
        result: dict[str, Pose] = {}
        with self.lock:
            for arm in ("left", "right"):
                message = self.arm_data[arm]
                age = now - self.arm_received_monotonic[arm]
                if message is None or age > max_age_sec:
                    raise RuntimeError(
                        f"{arm} ArmSlaveData is missing/stale (age={age:.3f}s)"
                    )
                result[arm] = deepcopy(message.pose)
        return result

    def base_transform(self, freeze_frame: str, arm_base_frame: str):
        try:
            transform = self.tf_buffer.lookup_transform(
                freeze_frame,
                arm_base_frame,
                Time(),
                timeout=Duration(seconds=5.0),
            ).transform
        except TransformException as exc:
            raise RuntimeError(
                f"TF lookup {freeze_frame} -> {arm_base_frame} failed: {exc}"
            ) from exc
        return (
            (
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ),
            _normalize_quaternion(
                (
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                )
            ),
        )


def _absolute_correction(
    box_pose: PoseStamped,
    ideal_pose_in_arm_base: Pose,
    freeze_to_arm_base,
    offset_box: Sequence[float],
    box_to_link_orientation: Sequence[float],
    fixture_xyz: Sequence[float],
    fixture_enabled: bool,
) -> list[float]:
    box_values = _pose_values(box_pose.pose)
    p_box = box_values[:3]
    q_box = _normalize_quaternion(box_values[3:])

    p_f_a, q_f_a = freeze_to_arm_base
    ideal_values = _pose_values(ideal_pose_in_arm_base)
    p_a_l = ideal_values[:3]
    q_a_l = _normalize_quaternion(ideal_values[3:])
    rotated_p_a_l = _rotate_vector(p_a_l, q_f_a)
    p_f_l = tuple(p_f_a[index] + rotated_p_a_l[index] for index in range(3))
    q_f_l = _quaternion_multiply(q_f_a, q_a_l)

    if fixture_enabled:
        fixture_in_freeze = _rotate_vector(fixture_xyz, q_f_l)
    else:
        fixture_in_freeze = (0.0, 0.0, 0.0)
    p_fixture = tuple(p_f_l[index] + fixture_in_freeze[index] for index in range(3))
    box_to_fixture_freeze = tuple(
        p_fixture[index] - p_box[index] for index in range(3)
    )
    box_to_fixture_box = _rotate_vector(
        box_to_fixture_freeze, _quaternion_conjugate(q_box)
    )
    translation = [
        box_to_fixture_box[index] - float(offset_box[index]) for index in range(3)
    ]

    correction_q = _quaternion_multiply(
        _quaternion_multiply(_quaternion_conjugate(q_box), q_f_l),
        _quaternion_conjugate(box_to_link_orientation),
    )
    return [*translation, *correction_q]


def _profile_parameter(prefix: str, stem: str, arm: str, model: str, layer: int) -> str:
    if stem == "offset":
        return f"{prefix}_direct_movel_{arm}_offset_xyz_{model}_layer{layer}"
    if stem == "correction":
        return (
            f"{prefix}_joint123_{arm}_target_correction_pose_box_"
            f"{model}_layer{layer}"
        )
    raise ValueError(stem)


def _action_spec(target_action: str):
    if target_action == "grasp_box_tf":
        return (
            "grasp_box_tf",
            "/grasp_box_tf",
            "mission_interfaces/action/ExecuteBoxGrasp",
        )
    return (
        "drag_box_tf",
        "/execute_drag_box_grasp_tf",
        "mission_interfaces/action/ExecuteDragBoxGrasp",
    )


def _wait_for_token(prompt: str, expected: str) -> None:
    """Wait for an explicit token; blank input must never advance the workflow."""
    while True:
        try:
            value = input(prompt).strip()
        except EOFError as exc:
            raise RuntimeError("interactive terminal input is required") from exc
        if value == expected:
            return
        print(f"Please type {expected} exactly; no action was taken.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drive one TF box-layer calibration and solve/apply its absolute "
            "box-frame target correction without modifying other layers."
        )
    )
    parser.add_argument(
        "--target-action",
        choices=("grasp_box_tf", "drag_box_tf"),
        default="grasp_box_tf",
    )
    parser.add_argument("--box-type", choices=("bigbox", "smallbox"), required=True)
    parser.add_argument("--box-layer", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--apply-runtime",
        action="store_true",
        help="apply the solved result to the selected layer after confirmation",
    )
    args = parser.parse_args()

    prefix, action_name, action_type = _action_spec(args.target_action)
    model = args.box_type
    layer = args.box_layer
    correction_names = {
        arm: _profile_parameter(prefix, "correction", arm, model, layer)
        for arm in ("left", "right")
    }
    offsets = {
        arm: _param_get(_profile_parameter(prefix, "offset", arm, model, layer))
        for arm in ("left", "right")
    }

    temporary_values = {
        "grasp_box_tf_force_clamp_mode": "disabled",
        "drag_box_tf_force_clamp_mode": "disabled",
        "box_post_movel_enabled": False,
        "drag_box_post_movel_enabled": False,
        "grasp_box_tf_body_home_carry_enabled": False,
        "drag_box_tf_body_home_carry_enabled": False,
        "box_step2_waist_endpoint_sync_enabled": False,
        "box_body_return_home_enabled": False,
        "box_post_arm_movej_enabled": False,
        correction_names["left"]: IDENTITY_POSE,
        correction_names["right"]: IDENTITY_POSE,
    }
    saved_values = {name: _param_get(name) for name in temporary_values}

    print("\n=== TF target correction real-robot calibration ===")
    print(f"target action : {args.target_action}")
    print(f"box profile  : {model}, layer{layer} (only this layer will be modified)")
    print(f"left offset  : {offsets['left']}")
    print(f"right offset : {offsets['right']}")
    print("\nThe action WILL move the waist and robot arm(s).")
    print("Clear the workspace, enable the robot, and keep an emergency stop ready.")
    if input("Type MOVE to continue: ").strip() != "MOVE":
        print("Canceled; no parameters were changed.")
        return 2

    rclpy.init()
    observer = CalibrationObserver()
    spin_thread = threading.Thread(target=rclpy.spin, args=(observer,), daemon=True)
    spin_thread.start()
    applied = False
    try:
        for name, value in temporary_values.items():
            _param_set(name, value)
        observer.clear_raw_box()
        request_id = (
            f"cal-{args.target_action}-{model}-l{layer}-{uuid.uuid4().hex[:8]}"
        )
        goal = (
            "{request_id: '"
            + request_id
            + f"', target_label: {args.target_label}, box_layer: {layer}, "
            + f"box_type: '{model}', dry_run: false}}"
        )
        print("\nSending real action; all motion after the initial MoveJ_P is disabled.\n")
        process = subprocess.Popen(
            ["ros2", "action", "send_goal", "--feedback", action_name, action_type, goal],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        action_output: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            action_output.append(line)
        return_code = process.wait()
        output_text = "".join(action_output)
        if return_code != 0 or "Goal finished with status: SUCCEEDED" not in output_text:
            raise RuntimeError("calibration action did not finish with SUCCEEDED")
        time.sleep(0.5)
        box_pose = observer.box_snapshot()
        actual_poses = observer.arm_snapshot()
        print("\nActual Link8 poses after the calibration action:")
        print(f"left : {_format_array(_pose_values(actual_poses['left']))}")
        print(f"right: {_format_array(_pose_values(actual_poses['right']))}")
        print(f"frozen box frame: {box_pose.header.frame_id}")
        print(f"frozen box pose : {_format_array(_pose_values(box_pose.pose))}")

        _wait_for_token(
            "\nManually jog BOTH Link8 origins to the ideal poses. "
            "Do not move the box, chassis, or waist. "
            "Type IDEAL when stable (blank Enter does not continue): ",
            "IDEAL",
        )
        time.sleep(0.5)
        ideal_poses = observer.arm_snapshot()
        freeze_frame = box_pose.header.frame_id.lstrip("/")
        if not freeze_frame:
            raise RuntimeError("frozen box pose has an empty frame_id")
        arm_base_frames = {
            "left": str(_param_get("left_arm_base_frame")).lstrip("/"),
            "right": str(_param_get("right_arm_base_frame")).lstrip("/"),
        }
        fixture_enabled = bool(_param_get("direct_movel_fixture_compensation_enabled"))
        results: dict[str, list[float]] = {}
        for arm in ("left", "right"):
            results[arm] = _absolute_correction(
                box_pose,
                ideal_poses[arm],
                observer.base_transform(freeze_frame, arm_base_frames[arm]),
                offsets[arm],
                _param_get(f"direct_movel_{arm}_box_to_link8_orientation"),
                _param_get(f"{arm}_fixture_center_in_link8_xyz"),
                fixture_enabled,
            )

        print("\nIdeal Link8 poses read from live feedback:")
        print(f"left : {_format_array(_pose_values(ideal_poses['left']))}")
        print(f"right: {_format_array(_pose_values(ideal_poses['right']))}")
        print("\nSolved absolute target_correction_pose_box:")
        print(f"left : {_format_array(results['left'])}")
        print(f"right: {_format_array(results['right'])}")
        print(f"\nRuntime commands for {model} layer{layer}:")
        for arm in ("left", "right"):
            name = _profile_parameter(prefix, "correction", arm, model, layer)
            print(
                f"ros2 param set {MISSION_NODE} {name} "
                f"\"{_format_array(results[arm])}\""
            )

        if args.apply_runtime:
            if input(f"\nType APPLY to set these values on {model} layer{layer}: ").strip() == "APPLY":
                for arm in ("left", "right"):
                    _param_set(
                        _profile_parameter(prefix, "correction", arm, model, layer),
                        results[arm],
                    )
                applied = True
                print("Applied only to the selected action/model/layer.")
            else:
                print("Result was not applied.")
        return 0
    finally:
        # Restore every temporary flow setting. Keep solved target corrections only
        # when the user explicitly applied them.
        for name, value in saved_values.items():
            if applied and name in correction_names.values():
                continue
            try:
                _param_set(name, value)
            except Exception as exc:  # best-effort restoration must remain visible
                print(f"WARNING: failed to restore {name}: {exc}", file=sys.stderr)
        observer.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCanceled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
