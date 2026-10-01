"""Interactive calibration of both initial GraspBox TF arm targets."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Sequence

import rclpy

from .drag_right_target_calibration import (
    MISSION_NODE,
    CalibrationObserver,
    _absolute_target_correction,
    _format_array,
    _param_get,
    _param_set,
    _pose_values,
    _quaternion_conjugate,
    _quaternion_multiply,
    _quaternion_to_rpy_degrees,
    _wait_for_one,
)
from .tf_calibration_defaults import apply_and_save, default_config_root


ACTION_NAME = "/grasp_box_tf"
ACTION_TYPE = "mission_interfaces/action/ExecuteBoxGrasp"
HOLD_PARAMETER = "grasp_box_tf_calibration_stop_after_initial_dual_target_enabled"


def _profile_parameter(arm: str, stem: str, model: str, layer: int) -> str:
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    if stem == "offset":
        return f"grasp_box_tf_direct_movel_{arm}_offset_xyz_{model}_layer{layer}"
    if stem == "correction":
        return (
            f"grasp_box_tf_joint123_{arm}_target_correction_pose_box_"
            f"{model}_layer{layer}"
        )
    raise ValueError(stem)


def _send_grasp_action(box_type: str, box_layer: int, target_label: int) -> None:
    request_id = f"cal-grasp-dual-{box_type}-l{box_layer}-{uuid.uuid4().hex[:8]}"
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
        raise RuntimeError("GraspBox TF 标定动作没有成功结束，请查看上方 Action 报错")
    if "GRASP_DUAL_TARGET_CALIBRATION_HOLD" not in output:
        raise RuntimeError(
            "动作没有到达专用标定停点；请确认 Mission 已重启并加载最新构建"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="识别料箱、双臂停在两侧，并计算 GraspBox TF 双臂目标修正"
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

    correction_names = {
        arm: _profile_parameter(
            arm, "correction", args.box_type, args.box_layer
        )
        for arm in ("left", "right")
    }
    offset_names = {
        arm: _profile_parameter(arm, "offset", args.box_type, args.box_layer)
        for arm in ("left", "right")
    }
    current_corrections = {
        arm: [float(value) for value in _param_get(correction_names[arm])]
        for arm in ("left", "right")
    }
    offsets = {
        arm: [float(value) for value in _param_get(offset_names[arm])]
        for arm in ("left", "right")
    }
    saved_hold = _param_get(HOLD_PARAMETER)
    saved_force_mode = _param_get("grasp_box_tf_force_clamp_mode")

    print("\n========== GraspBox TF 双臂目标标定 ==========")
    print(f"料箱配置：{args.box_type}，第 {args.box_layer} 层")
    for arm in ("left", "right"):
        print(f"当前{arm} offset：{_format_array(offsets[arm])}")
        print(f"当前{arm} correction：{_format_array(current_corrections[arm])}")
    print("\n脚本将真实运行识别、腰部自适应和双臂初始目标运动。")
    print("双臂到达料箱两侧后会成功结束 Action，不会执行力控、抬升或携箱回零。")
    _wait_for_one(
        "步骤1：清空机器人周围空间，确认急停可用、料箱稳定。"
        "输入 1 开始；输入 0 退出："
    )

    rclpy.init()
    observer = CalibrationObserver()
    spin_thread = threading.Thread(target=rclpy.spin, args=(observer,), daemon=True)
    spin_thread.start()
    try:
        _param_set("grasp_box_tf_force_clamp_mode", "disabled")
        _param_set(HOLD_PARAMETER, True)
        observer.clear_box()
        print("\n步骤2：正在调用 FoundationPose 并让双臂运动到料箱两侧……\n")
        _send_grasp_action(args.box_type, args.box_layer, args.target_label)
        time.sleep(0.5)

        box_pose = observer.box_snapshot()
        initial_poses = {
            arm: observer.arm_pose_snapshot(arm) for arm in ("left", "right")
        }
        print("\n双臂已经停在当前计算目标，后续夹紧与抬升没有执行。")
        print(f"识别料箱 frame：{box_pose.header.frame_id}")
        print(f"识别料箱 pose：{_format_array(_pose_values(box_pose.pose))}")
        for arm in ("left", "right"):
            print(f"调整前{arm}臂 pose：{_format_array(_pose_values(initial_poses[arm]))}")

        _wait_for_one(
            "\n步骤3：使用示教器低速分别调整左右臂，使两个末端的 position 和 "
            "orientation 都达到理想料箱侧面姿态。不要移动料箱、底盘或腰部。"
            "双臂均静止后输入 1；输入 0 退出："
        )
        time.sleep(0.5)
        ideal_poses = {
            arm: observer.arm_pose_snapshot(arm) for arm in ("left", "right")
        }

        freeze_frame = box_pose.header.frame_id.lstrip("/")
        if not freeze_frame:
            raise RuntimeError("识别料箱 Pose 的 frame_id 为空")
        fixture_enabled = bool(
            _param_get("direct_movel_fixture_compensation_enabled")
        )
        results: dict[str, list[float]] = {}
        for arm in ("left", "right"):
            arm_base_frame = str(_param_get(f"{arm}_arm_base_frame")).lstrip("/")
            results[arm] = _absolute_target_correction(
                box_pose,
                ideal_poses[arm],
                observer.freeze_to_arm_base(freeze_frame, arm_base_frame),
                offsets[arm],
                _param_get(f"direct_movel_{arm}_box_to_link8_orientation"),
                _param_get(f"{arm}_fixture_center_in_link8_xyz"),
                fixture_enabled,
            )

        print("\n========== 标定计算结果 ==========")
        for arm in ("left", "right"):
            translation_delta = [
                results[arm][index] - current_corrections[arm][index]
                for index in range(3)
            ]
            orientation_delta = _quaternion_multiply(
                _quaternion_conjugate(current_corrections[arm][3:]),
                results[arm][3:],
            )
            print(f"\n{arm}臂调整后 pose：{_format_array(_pose_values(ideal_poses[arm]))}")
            print(
                f"{arm}臂料箱坐标系 position 增量 [m]："
                f"{_format_array(translation_delta)}"
            )
            print(
                f"{arm}臂 orientation 增量四元数："
                f"{_format_array(orientation_delta)}"
            )
            print(
                f"{arm}臂 orientation 增量 RPY [deg]："
                f"{_format_array(_quaternion_to_rpy_degrees(orientation_delta), 4)}"
            )
            print(f"{arm}臂新绝对 correction：{_format_array(results[arm])}")

        print("\n临时应用命令：")
        for arm in ("left", "right"):
            print(
                f"ros2 param set {MISSION_NODE} {correction_names[arm]} "
                f"\"{_format_array(results[arm])}\""
            )
        print("\n配置文件字段：")
        for arm in ("left", "right"):
            print(f"{correction_names[arm]}: {_format_array(results[arm])}")

        if args.save_defaults:
            config_file = args.config_root / "grasp_tf.yaml"
            _wait_for_one(
                f"\n步骤4：确认把双臂结果写入运行参数和默认配置 {config_file}。"
                "输入 1 保存；输入 0 退出："
            )
            updates = {correction_names[arm]: results[arm] for arm in ("left", "right")}
            previous = {
                correction_names[arm]: current_corrections[arm]
                for arm in ("left", "right")
            }
            backup = apply_and_save(
                config_file=config_file, results=updates,
                previous_runtime=previous, param_set=_param_set,
            )
            print(f"已写入当前 Mission 和默认配置；原配置备份：{backup}")
        elif args.apply_runtime:
            _wait_for_one(
                "\n步骤4：确认要把左右结果临时写入当前 Mission。"
                "输入 1 应用；输入 0 退出："
            )
            for arm in ("left", "right"):
                _param_set(correction_names[arm], results[arm])
            print("已临时应用，仅修改所选料箱类型和层数。")
        return 0
    finally:
        for name, value in (
            (HOLD_PARAMETER, saved_hold),
            ("grasp_box_tf_force_clamp_mode", saved_force_mode),
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
