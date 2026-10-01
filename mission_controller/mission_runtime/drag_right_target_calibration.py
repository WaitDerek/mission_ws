"""Interactive calibration of the DragBox TF initial right-arm target.

The normal DragBox action performs detection and reaches its initial right-side
target.  A calibration-only hold flag makes the action return successfully at
that exact point, before force clamp or any Drag motion.  After the operator
jogs the right Link8 to the desired pose, this tool solves the absolute
box-frame target correction for the selected box type and layer.
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

from .tf_calibration_defaults import apply_and_save, default_config_root


MISSION_NODE = "/mission_controller"
ACTION_NAME = "/execute_drag_box_grasp_tf"
ACTION_TYPE = "mission_interfaces/action/ExecuteDragBoxGrasp"
HOLD_PARAMETER = (
    "drag_box_tf_calibration_stop_after_initial_right_target_enabled"
)


def _normalize_quaternion(
    values: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError(f"quaternion must contain four values: {values}")
    quaternion = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12 or not all(math.isfinite(value) for value in quaternion):
        raise ValueError(f"invalid quaternion: {values}")
    quaternion = tuple(value / norm for value in quaternion)
    if quaternion[3] < 0.0:
        quaternion = tuple(-value for value in quaternion)
    return quaternion


def _quaternion_conjugate(
    quaternion: Sequence[float],
) -> tuple[float, float, float, float]:
    x, y, z, w = _normalize_quaternion(quaternion)
    return (-x, -y, -z, w)


def _quaternion_multiply(
    lhs: Sequence[float], rhs: Sequence[float]
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = _normalize_quaternion(lhs)
    x2, y2, z2, w2 = _normalize_quaternion(rhs)
    return _normalize_quaternion(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def _rotate_vector(
    vector: Sequence[float], quaternion: Sequence[float]
) -> tuple[float, float, float]:
    x, y, z, w = _normalize_quaternion(quaternion)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _quaternion_to_rpy_degrees(
    quaternion: Sequence[float],
) -> tuple[float, float, float]:
    x, y, z, w = _normalize_quaternion(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


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
    return "[" + ", ".join(
        f"{float(value):.{precision}f}" for value in values
    ) + "]"


def _run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(command, text=True, capture_output=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"命令执行失败：{' '.join(command)}\n{detail}")
    return result


def _param_get(name: str):
    output = _run(["ros2", "param", "get", MISSION_NODE, name]).stdout.strip()
    if "array(" in output:
        start = output.find("[")
        end = output.rfind("]")
        if start < 0 or end < start:
            raise RuntimeError(f"无法解析参数 {name}: {output}")
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
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def _param_set(name: str, value) -> None:
    result = _run(
        ["ros2", "param", "set", MISSION_NODE, name, _param_text(value)],
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0 or "successful" not in output.lower():
        raise RuntimeError(f"设置参数 {name} 失败：{output}")


def _wait_for_one(prompt: str) -> None:
    while True:
        try:
            answer = input(prompt).strip()
        except EOFError as exc:
            raise RuntimeError("该标定必须在交互式终端中运行") from exc
        if answer == "1":
            return
        if answer == "0":
            raise KeyboardInterrupt
        print("输入无效：请输入 1 继续，或输入 0 退出。")


class CalibrationObserver(Node):
    """Capture the frozen raw box pose, dual-arm EEPose, and live TF."""

    def __init__(self) -> None:
        super().__init__("drag_right_target_calibrator")
        qos = QoSProfile(depth=20)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE
        self.lock = threading.Lock()
        self.raw_box_pose: PoseStamped | None = None
        self.raw_box_time = 0.0
        self.arm_data: dict[str, ArmSlaveData | None] = {
            "left": None,
            "right": None,
        }
        self.arm_time = {"left": 0.0, "right": 0.0}
        self.create_subscription(
            PoseStamped,
            "/mission/box_object_pose_raw",
            self._box_callback,
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

    def _box_callback(self, message: PoseStamped) -> None:
        with self.lock:
            self.raw_box_pose = deepcopy(message)
            self.raw_box_time = time.monotonic()

    def _arm_callback(self, arm: str, message: ArmSlaveData) -> None:
        with self.lock:
            self.arm_data[arm] = deepcopy(message)
            self.arm_time[arm] = time.monotonic()

    def clear_box(self) -> None:
        with self.lock:
            self.raw_box_pose = None
            self.raw_box_time = 0.0

    def box_snapshot(self) -> PoseStamped:
        with self.lock:
            if self.raw_box_pose is None:
                raise RuntimeError("没有收到 /mission/box_object_pose_raw")
            return deepcopy(self.raw_box_pose)

    def arm_pose_snapshot(self, arm: str, max_age_sec: float = 1.0) -> Pose:
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        now = time.monotonic()
        with self.lock:
            age = now - self.arm_time[arm]
            if self.arm_data[arm] is None or age > max_age_sec:
                raise RuntimeError(
                    f"{arm} ArmSlaveData 数据缺失或过期：age={age:.3f}s"
                )
            return deepcopy(self.arm_data[arm].pose)

    def right_pose_snapshot(self, max_age_sec: float = 1.0) -> Pose:
        return self.arm_pose_snapshot("right", max_age_sec)

    def freeze_to_arm_base(self, freeze_frame: str, arm_base_frame: str):
        try:
            transform = self.tf_buffer.lookup_transform(
                freeze_frame,
                arm_base_frame,
                Time(),
                timeout=Duration(seconds=5.0),
            ).transform
        except TransformException as exc:
            raise RuntimeError(
                f"查询 TF {freeze_frame} <- {arm_base_frame} 失败：{exc}"
            ) from exc
        return (
            (
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
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


def _absolute_target_correction(
    box_pose: PoseStamped,
    ideal_pose_in_arm_base: Pose,
    freeze_to_arm_base,
    offset_box: Sequence[float],
    box_to_link_orientation: Sequence[float],
    fixture_xyz: Sequence[float],
    fixture_enabled: bool,
) -> list[float]:
    """Solve the absolute configured correction [xyz, quaternion] in box frame."""
    box_values = _pose_values(box_pose.pose)
    box_position = box_values[:3]
    box_orientation = _normalize_quaternion(box_values[3:])
    freeze_to_arm_position, freeze_to_arm_orientation = freeze_to_arm_base

    ideal_values = _pose_values(ideal_pose_in_arm_base)
    ideal_position_in_freeze = _rotate_vector(
        ideal_values[:3], freeze_to_arm_orientation
    )
    link_position_in_freeze = tuple(
        freeze_to_arm_position[index] + ideal_position_in_freeze[index]
        for index in range(3)
    )
    link_orientation_in_freeze = _quaternion_multiply(
        freeze_to_arm_orientation, ideal_values[3:]
    )

    fixture_in_freeze = (
        _rotate_vector(fixture_xyz, link_orientation_in_freeze)
        if fixture_enabled
        else (0.0, 0.0, 0.0)
    )
    fixture_position = tuple(
        link_position_in_freeze[index] + fixture_in_freeze[index]
        for index in range(3)
    )
    box_to_fixture_in_freeze = tuple(
        fixture_position[index] - box_position[index] for index in range(3)
    )
    box_to_fixture_in_box = _rotate_vector(
        box_to_fixture_in_freeze,
        _quaternion_conjugate(box_orientation),
    )
    translation = [
        box_to_fixture_in_box[index] - float(offset_box[index])
        for index in range(3)
    ]
    correction_orientation = _quaternion_multiply(
        _quaternion_multiply(
            _quaternion_conjugate(box_orientation),
            link_orientation_in_freeze,
        ),
        _quaternion_conjugate(box_to_link_orientation),
    )
    return [*translation, *correction_orientation]


# Backward-compatible internal name used by the first calibration tests.
_absolute_right_correction = _absolute_target_correction


def _profile_parameter(stem: str, model: str, layer: int) -> str:
    if stem == "offset":
        return f"drag_box_tf_direct_movel_right_offset_xyz_{model}_layer{layer}"
    if stem == "correction":
        return (
            "drag_box_tf_joint123_right_target_correction_pose_box_"
            f"{model}_layer{layer}"
        )
    raise ValueError(stem)


def _send_drag_action(box_type: str, box_layer: int, target_label: int) -> str:
    request_id = f"cal-drag-right-{box_type}-l{box_layer}-{uuid.uuid4().hex[:8]}"
    goal = (
        "{request_id: '"
        + request_id
        + f"', target_label: {target_label}, box_layer: {box_layer}, "
        + f"box_type: '{box_type}', dry_run: false}}"
    )
    process = subprocess.Popen(
        ["ros2", "action", "send_goal", "--feedback", ACTION_NAME, ACTION_TYPE, goal],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    return_code = process.wait()
    output = "".join(output_lines)
    if return_code != 0 or "Goal finished with status: SUCCEEDED" not in output:
        raise RuntimeError("DragBox TF 标定动作没有成功结束，请查看上方 Action 报错")
    if "DRAG_RIGHT_TARGET_CALIBRATION_HOLD" not in output:
        raise RuntimeError(
            "动作没有到达专用标定停点；请确认 Mission 已重启并加载最新构建"
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="识别料箱、停在右侧，并计算 DragBox TF 右臂目标修正"
    )
    parser.add_argument("--box-type", choices=("bigbox", "smallbox"), required=True)
    parser.add_argument("--box-layer", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--apply-runtime",
        action="store_true",
        help="计算完成后允许确认并临时写入 /mission_controller",
    )
    parser.add_argument(
        "--save-defaults", action="store_true",
        help="确认后同步写入运行参数及源 YAML 默认值（自动备份）",
    )
    parser.add_argument(
        "--config-root", type=Path, default=default_config_root(),
        help="Mission 源码 config/mission 目录",
    )
    args = parser.parse_args(argv)

    correction_name = _profile_parameter(
        "correction", args.box_type, args.box_layer
    )
    offset_name = _profile_parameter("offset", args.box_type, args.box_layer)
    saved_hold = _param_get(HOLD_PARAMETER)
    saved_force_mode = _param_get("drag_box_tf_force_clamp_mode")
    current_correction = [float(value) for value in _param_get(correction_name)]
    offset = [float(value) for value in _param_get(offset_name)]
    if str(_param_get("drag_box_left_join_mode")).strip() != "after_drag3":
        raise RuntimeError(
            "当前 drag_box_left_join_mode 不是 after_drag3，无法保证初始阶段只有右臂"
        )

    print("\n========== DragBox TF 右臂目标标定 ==========")
    print(f"料箱配置：{args.box_type}，第 {args.box_layer} 层")
    print(f"当前右臂 offset：{_format_array(offset)}")
    print(f"当前右臂 correction：{_format_array(current_correction)}")
    print("\n脚本将真实运行识别、腰部自适应和右臂初始目标运动。")
    print("到达右侧后会正常结束 Action，不会执行力控、Drag1/2/3或左臂加入。")
    _wait_for_one(
        "步骤1：清空机器人周围空间，确认急停可用、料箱稳定。"
        "输入 1 开始；输入 0 退出："
    )

    rclpy.init()
    observer = CalibrationObserver()
    spin_thread = threading.Thread(target=rclpy.spin, args=(observer,), daemon=True)
    spin_thread.start()
    try:
        _param_set("drag_box_tf_force_clamp_mode", "disabled")
        _param_set(HOLD_PARAMETER, True)
        observer.clear_box()
        print("\n步骤2：正在调用 FoundationPose 并让右臂运动到料箱右侧……\n")
        _send_drag_action(args.box_type, args.box_layer, args.target_label)
        time.sleep(0.5)

        box_pose = observer.box_snapshot()
        initial_pose = observer.right_pose_snapshot()
        print("\n右臂已停在当前计算目标，后续拖拽没有执行。")
        print(f"识别料箱 frame：{box_pose.header.frame_id}")
        print(f"识别料箱 pose：{_format_array(_pose_values(box_pose.pose))}")
        print(f"调整前右臂 pose：{_format_array(_pose_values(initial_pose))}")
        _wait_for_one(
            "\n步骤3：使用示教器低速调整右臂，使末端 position 和 orientation "
            "达到理想的料箱右侧姿态。不要移动料箱、底盘或腰部。"
            "调整完成且右臂静止后输入 1；输入 0 退出："
        )
        time.sleep(0.5)
        ideal_pose = observer.right_pose_snapshot()

        freeze_frame = box_pose.header.frame_id.lstrip("/")
        if not freeze_frame:
            raise RuntimeError("识别料箱 Pose 的 frame_id 为空")
        right_arm_base_frame = str(_param_get("right_arm_base_frame")).lstrip("/")
        new_correction = _absolute_target_correction(
            box_pose,
            ideal_pose,
            observer.freeze_to_arm_base(freeze_frame, right_arm_base_frame),
            offset,
            _param_get("direct_movel_right_box_to_link8_orientation"),
            _param_get("right_fixture_center_in_link8_xyz"),
            bool(_param_get("direct_movel_fixture_compensation_enabled")),
        )
        translation_delta = [
            new_correction[index] - current_correction[index] for index in range(3)
        ]
        orientation_delta = _quaternion_multiply(
            _quaternion_conjugate(current_correction[3:]),
            new_correction[3:],
        )

        print("\n========== 标定计算结果 ==========")
        print(f"调整后右臂 pose：{_format_array(_pose_values(ideal_pose))}")
        print("料箱坐标系下 position 增量 [x,y,z] m：")
        print(_format_array(translation_delta))
        print("料箱坐标系下 orientation 增量四元数 [x,y,z,w]：")
        print(_format_array(orientation_delta))
        print("orientation 增量 RPY [deg]（仅用于理解）：")
        print(_format_array(_quaternion_to_rpy_degrees(orientation_delta), 4))
        print("\n新的绝对 correction 参数 [x,y,z,qx,qy,qz,qw]：")
        print(_format_array(new_correction))
        command = (
            f"ros2 param set {MISSION_NODE} {correction_name} "
            f"\"{_format_array(new_correction)}\""
        )
        print("\n临时应用命令：")
        print(command)
        print("\n配置文件字段：")
        print(f"{correction_name}: {_format_array(new_correction)}")

        if args.save_defaults:
            config_file = args.config_root / "drag.yaml"
            _wait_for_one(
                f"\n步骤4：确认把右臂结果写入运行参数和默认配置 {config_file}。"
                "输入 1 保存；输入 0 退出："
            )
            backup = apply_and_save(
                config_file=config_file,
                results={correction_name: new_correction},
                previous_runtime={correction_name: current_correction},
                param_set=_param_set,
            )
            print(f"已写入当前 Mission 和默认配置；原配置备份：{backup}")
        elif args.apply_runtime:
            _wait_for_one(
                "\n步骤4：确认要把结果临时写入当前 Mission。输入 1 应用；输入 0 退出："
            )
            _param_set(correction_name, new_correction)
            print("已临时应用；没有修改其他料箱类型或层数。")
        return 0
    finally:
        for name, value in (
            (HOLD_PARAMETER, saved_hold),
            ("drag_box_tf_force_clamp_mode", saved_force_mode),
        ):
            try:
                _param_set(name, value)
            except Exception as exc:
                print(f"警告：恢复参数 {name} 失败：{exc}", file=sys.stderr)
        observer.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n标定已退出。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"标定失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
