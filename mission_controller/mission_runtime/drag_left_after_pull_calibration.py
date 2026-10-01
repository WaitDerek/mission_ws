"""Calibrate the DragBox TF left-arm join target after the right-arm pull."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Sequence

import rclpy
from geometry_msgs.msg import PoseStamped

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


ACTION_NAME = "/execute_drag_box_grasp_tf"
ACTION_TYPE = "mission_interfaces/action/ExecuteDragBoxGrasp"
HOLD_PARAMETER = "drag_box_tf_calibration_stop_after_left_join_enabled"


def _profile_parameter(stem: str, model: str, layer: int) -> str:
    if stem == "offset":
        return f"drag_box_tf_direct_movel_left_offset_xyz_{model}_layer{layer}"
    if stem == "correction":
        return (
            "drag_box_tf_joint123_left_target_correction_pose_box_"
            f"{model}_layer{layer}"
        )
    raise ValueError(stem)


def _send_drag_action(box_type: str, box_layer: int, target_label: int) -> str:
    request_id = f"cal-drag-left-{box_type}-l{box_layer}-{uuid.uuid4().hex[:8]}"
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
        raise RuntimeError("DragBox TF 左臂加入标定动作失败，请查看上方 Action 报错")
    if "DRAG_LEFT_TARGET_CALIBRATION_HOLD" not in output:
        raise RuntimeError(
            "动作没有到达左臂加入标定停点；请确认 Mission 已重启并加载最新构建"
        )
    return output


def _parse_reanchored_box_pose(output: str, frame_id: str) -> PoseStamped:
    position_matches = re.findall(r"inferred_box_position=\[([^\]]+)\]", output)
    orientation_matches = re.findall(
        r"inferred_box_orientation=\[([^\]]+)\]", output
    )
    if not position_matches or not orientation_matches:
        raise RuntimeError("Action反馈中没有完整的Drag3后实际料箱Pose")
    position = [float(value.strip()) for value in position_matches[-1].split(",")]
    orientation = [
        float(value.strip()) for value in orientation_matches[-1].split(",")
    ]
    if len(position) != 3 or len(orientation) != 4:
        raise RuntimeError("Action反馈中的Drag3后料箱Pose维度错误")
    result = PoseStamped()
    result.header.frame_id = frame_id
    result.pose.position.x, result.pose.position.y, result.pose.position.z = position
    (
        result.pose.orientation.x,
        result.pose.orientation.y,
        result.pose.orientation.z,
        result.pose.orientation.w,
    ) = orientation
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="右臂完成拖出、左臂加入后，标定DragBox TF左臂目标"
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

    if str(_param_get("drag_box_left_join_mode")).strip() != "after_drag3":
        raise RuntimeError("标定要求 drag_box_left_join_mode=after_drag3")
    if not bool(_param_get("drag_box_tf_reanchor_after_drag3_enabled")):
        raise RuntimeError("标定要求 drag_box_tf_reanchor_after_drag3_enabled=true")
    if str(_param_get("drag_box_tf_force_clamp_mode")).strip() != "closed_loop":
        raise RuntimeError("标定要求右臂初始接触力控为 closed_loop")

    correction_name = _profile_parameter(
        "correction", args.box_type, args.box_layer
    )
    offset_name = _profile_parameter("offset", args.box_type, args.box_layer)
    current_correction = [float(value) for value in _param_get(correction_name)]
    offset = [float(value) for value in _param_get(offset_name)]
    saved_hold = _param_get(HOLD_PARAMETER)

    print("\n========== DragBox TF 拖出后左臂目标标定 ==========")
    print(f"料箱配置：{args.box_type}，第 {args.box_layer} 层")
    print(f"当前左臂 offset：{_format_array(offset)}")
    print(f"当前左臂 correction：{_format_array(current_correction)}")
    print("\n本脚本会真实完成右臂接触和Drag1/Drag2/Drag3。")
    print("随后根据右臂实际Link8重建料箱Pose，让左臂到达当前加入目标并停止。")
    print("不会执行左臂加入后的双臂力控、Step2、携箱回零或回零动作。")
    _wait_for_one(
        "步骤1：确认料箱、机器人周围空间和拖出路径安全，急停可用。"
        "输入 1 开始；输入 0 退出："
    )

    rclpy.init()
    observer = CalibrationObserver()
    spin_thread = threading.Thread(target=rclpy.spin, args=(observer,), daemon=True)
    spin_thread.start()
    try:
        _param_set(HOLD_PARAMETER, True)
        observer.clear_box()
        print("\n步骤2：正在识别料箱、执行右臂拖出并让左臂加入……\n")
        action_output = _send_drag_action(
            args.box_type, args.box_layer, args.target_label
        )
        time.sleep(0.5)

        original_box_pose = observer.box_snapshot()
        reanchored_box_pose = _parse_reanchored_box_pose(
            action_output, original_box_pose.header.frame_id
        )
        initial_left_pose = observer.arm_pose_snapshot("left")
        print("\n左臂已经停在Drag3后的当前加入目标。")
        print(
            "Drag3后实际料箱Pose："
            f"{_format_array(_pose_values(reanchored_box_pose.pose))}"
        )
        print(f"调整前左臂Pose：{_format_array(_pose_values(initial_left_pose))}")
        _wait_for_one(
            "\n步骤3：使用示教器低速调整左臂，使其position和orientation达到"
            "理想的料箱左侧姿态。保持右臂不动，不要移动料箱、底盘或腰部。"
            "左臂静止后输入 1；输入 0 退出："
        )
        time.sleep(0.5)
        ideal_left_pose = observer.arm_pose_snapshot("left")

        freeze_frame = reanchored_box_pose.header.frame_id.lstrip("/")
        if not freeze_frame:
            raise RuntimeError("Drag3后料箱Pose的frame_id为空")
        left_arm_base_frame = str(_param_get("left_arm_base_frame")).lstrip("/")
        new_correction = _absolute_target_correction(
            reanchored_box_pose,
            ideal_left_pose,
            observer.freeze_to_arm_base(freeze_frame, left_arm_base_frame),
            offset,
            _param_get("direct_movel_left_box_to_link8_orientation"),
            _param_get("left_fixture_center_in_link8_xyz"),
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
        print(f"调整后左臂Pose：{_format_array(_pose_values(ideal_left_pose))}")
        print("料箱坐标系下position增量 [x,y,z] m：")
        print(_format_array(translation_delta))
        print("料箱坐标系下orientation增量四元数 [x,y,z,w]：")
        print(_format_array(orientation_delta))
        print("orientation增量RPY [deg]（仅用于理解）：")
        print(_format_array(_quaternion_to_rpy_degrees(orientation_delta), 4))
        print("\n新的绝对左臂correction [x,y,z,qx,qy,qz,qw]：")
        print(_format_array(new_correction))
        print("\n临时应用命令：")
        print(
            f"ros2 param set {MISSION_NODE} {correction_name} "
            f"\"{_format_array(new_correction)}\""
        )
        print("\n配置文件字段：")
        print(f"{correction_name}: {_format_array(new_correction)}")

        if args.save_defaults:
            config_file = args.config_root / "drag.yaml"
            _wait_for_one(
                f"\n步骤4：确认把左臂结果写入运行参数和默认配置 {config_file}。"
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
                "\n步骤4：确认把结果临时写入当前Mission。"
                "输入 1 应用；输入 0 退出："
            )
            _param_set(correction_name, new_correction)
            print("已临时应用，只修改所选料箱类型和层数的左臂参数。")
        print("\n注意：机器人和料箱仍停在拖出后的标定状态，请按安全流程恢复。")
        return 0
    finally:
        try:
            _param_set(HOLD_PARAMETER, saved_hold)
        except Exception as exc:
            print(f"警告：恢复参数 {HOLD_PARAMETER} 失败：{exc}", file=sys.stderr)
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
