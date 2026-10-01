"""Calibrate both DragBox targets within one uninterrupted grasp Action."""

from __future__ import annotations

import argparse
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Sequence

import rclpy
from action_msgs.msg import GoalStatus
from mission_interfaces.action import ExecuteDragBoxGrasp
from rclpy.action import ActionClient

from .drag_left_after_pull_calibration import _parse_reanchored_box_pose
from .drag_right_target_calibration import (
    CalibrationObserver, MISSION_NODE, _absolute_target_correction,
    _format_array, _param_get, _param_set, _pose_values, _wait_for_one,
)
from .tf_calibration_defaults import apply_and_save, default_config_root

PAUSE = "drag_box_tf_calibration_pause_after_initial_right_target_enabled"
CONTINUE = "drag_box_tf_calibration_continue_after_initial_right_target"
LEFT_HOLD = "drag_box_tf_calibration_stop_after_left_join_enabled"
RIGHT_HOLD = "drag_box_tf_calibration_stop_after_initial_right_target_enabled"
LEFT_MOTION = "drag_box_left_join_motion_mode"
LEFT_JOINT_OVERRIDE = "drag_box_tf_calibration_left_join_joint_override_enabled"
LEFT_JOINT_TARGET = "drag_box_tf_calibration_left_join_joint_target_deg"
CALIBRATION_JOINT_TARGET_DEG = [
    -62.412,
    25.861,
    24.668,
    -40.876,
    -35.381,
    -8.514,
    -3.713,
]


def _wait_future(future, seconds: float, label: str):
    deadline = time.monotonic() + seconds
    while not future.done():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"等待{label}超时（{seconds:.0f}s）")
        time.sleep(0.05)
    return future.result()


class _OneDragGoal:
    def __init__(self, node):
        self.client = ActionClient(node, ExecuteDragBoxGrasp, "/execute_drag_box_grasp_tf")
        self.handle = None
        self.result_future = None
        self.condition = threading.Condition()
        self.stages: set[str] = set()
        self.details: list[str] = []

    def _feedback(self, msg):
        stage, detail = str(msg.feedback.stage), str(msg.feedback.detail)
        with self.condition:
            self.stages.add(stage)
            self.details.append(f"{stage}: {detail}")
            self.condition.notify_all()
        print(f"[{stage}] {detail}", flush=True)

    def start(self, layer: int, label: int):
        if not self.client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("/execute_drag_box_grasp_tf Action server 不可用")
        goal = ExecuteDragBoxGrasp.Goal()
        goal.request_id = f"cal-drag-bigbox-l{layer}-{uuid.uuid4().hex[:8]}"
        goal.target_label, goal.box_layer = label, layer
        goal.box_type, goal.dry_run = "bigbox", False
        self.handle = _wait_future(
            self.client.send_goal_async(goal, feedback_callback=self._feedback),
            20.0, "Goal 接收",
        )
        if not self.handle.accepted:
            raise RuntimeError("DragBox Goal 被拒绝；请检查 Mission lease")
        self.result_future = self.handle.get_result_async()

    def wait_stage(self, stage: str, seconds: float):
        deadline = time.monotonic() + seconds
        with self.condition:
            while stage not in self.stages:
                if self.result_future is not None and self.result_future.done():
                    result = self.result_future.result().result
                    raise RuntimeError(f"Action 未到 {stage} 就结束：{result.message}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"等待 {stage} 超时")
                self.condition.wait(min(remaining, 0.2))

    def wait_result(self):
        wrapped = _wait_future(self.result_future, 900.0, "拖箱和左臂加入")
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not wrapped.result.success:
            raise RuntimeError(
                f"DragBox 失败 status={wrapped.status}: {wrapped.result.message}; "
                f"pickup_message={wrapped.result.pickup_message}"
            )
        self.wait_stage("DRAG_LEFT_TARGET_CALIBRATION_HOLD", 2.0)
        return wrapped.result

    def cancel_if_running(self):
        if self.handle is None or self.result_future is None or self.result_future.done():
            return True
        try:
            _wait_future(self.handle.cancel_goal_async(), 5.0, "Action 取消确认")
            _wait_future(self.result_future, 10.0, "Action 取消完成")
            return True
        except Exception as exc:
            print(f"警告：取消未获确认：{exc}", file=sys.stderr)
            return False


def _calibrate_side(observer, side: str, layer: int, box_pose, config_root: Path):
    correction_name = (
        f"drag_box_tf_joint123_{side}_target_correction_pose_box_bigbox_layer{layer}"
    )
    offset_name = f"drag_box_tf_direct_movel_{side}_offset_xyz_bigbox_layer{layer}"
    previous = [float(x) for x in _param_get(correction_name)]
    offset = [float(x) for x in _param_get(offset_name)]
    print(f"\n{side}臂调整前 Pose：{_format_array(_pose_values(observer.arm_pose_snapshot(side)))}")
    _wait_for_one(
        f"示教器低速调整{side}臂到理想料箱侧面，保持料箱、底盘、腰部和另一机械臂不动。"
        "静止后输入 1；退出输入 0："
    )
    time.sleep(0.5)
    ideal = observer.arm_pose_snapshot(side)
    frame = box_pose.header.frame_id.lstrip("/")
    if not frame:
        raise RuntimeError("冻结料箱 Pose 的 frame_id 为空")
    arm_base = str(_param_get(f"{side}_arm_base_frame")).lstrip("/")
    correction = _absolute_target_correction(
        box_pose, ideal, observer.freeze_to_arm_base(frame, arm_base), offset,
        _param_get(f"direct_movel_{side}_box_to_link8_orientation"),
        _param_get(f"{side}_fixture_center_in_link8_xyz"),
        bool(_param_get("direct_movel_fixture_compensation_enabled")),
    )
    print(f"示教 Pose：{_format_array(_pose_values(ideal))}")
    print(f"{correction_name}: {_format_array(correction)}")
    _wait_for_one(
        f"确认把{side}臂修正写入当前参数和 {config_root / 'drag.yaml'}？"
        "输入 1 保存；0 退出："
    )
    backup = apply_and_save(
        config_file=config_root / "drag.yaml",
        results={correction_name: correction},
        previous_runtime={correction_name: previous}, param_set=_param_set,
    )
    print(f"{side}臂修正已保存，备份：{backup}")


def _calibrate_layer(layer: int, label: int, config_root: Path):
    if str(_param_get("drag_box_left_join_mode")).strip() != "after_drag3":
        raise RuntimeError("要求 drag_box_left_join_mode=after_drag3")
    if not bool(_param_get("drag_box_tf_reanchor_after_drag3_enabled")):
        raise RuntimeError("要求 drag_box_tf_reanchor_after_drag3_enabled=true")
    if str(_param_get("drag_box_tf_force_clamp_mode")).strip() != "closed_loop":
        raise RuntimeError("要求 drag_box_tf_force_clamp_mode=closed_loop")
    print(f"\n========== Bigbox 第 {layer} 层连续标定 ==========")
    print(
        "同一次 Action：左臂检测并避让 → 右臂目标标定 → 右臂拖出 → "
        "左臂通过 Python SDK MoveJ 到指定临时关节姿态并暂停；本轮绕过左臂加入 Pose IK。"
    )
    print("一次 MoveJ 未验证整个关节路径，必须确认避让姿态到目标间的扫掠空间安全。")
    _wait_for_one("确认机器人周围安全且急停可用。输入 1 启动；0 退出：")

    rclpy.init()
    observer = CalibrationObserver()
    spin_thread = threading.Thread(target=rclpy.spin, args=(observer,), daemon=True)
    spin_thread.start()
    goal = _OneDragGoal(observer)
    saved = {}
    try:
        for name in (
            PAUSE,
            CONTINUE,
            LEFT_HOLD,
            RIGHT_HOLD,
            LEFT_MOTION,
            LEFT_JOINT_OVERRIDE,
            LEFT_JOINT_TARGET,
        ):
            saved[name] = _param_get(name)
        if any(
            bool(saved[name])
            for name in (PAUSE, CONTINUE, LEFT_HOLD, RIGHT_HOLD, LEFT_JOINT_OVERRIDE)
        ):
            raise RuntimeError(
                "检测到上一轮标定标志仍为 true；请先确认没有活动 Action，"
                "再将标定标志恢复为 false"
            )
        for name, value in (
            (CONTINUE, False), (RIGHT_HOLD, False), (LEFT_HOLD, True),
            (LEFT_MOTION, "staged_ik_movej"), (PAUSE, True),
            (LEFT_JOINT_TARGET, CALIBRATION_JOINT_TARGET_DEG),
            (LEFT_JOINT_OVERRIDE, True),
        ):
            _param_set(name, value)
        observer.clear_box()
        goal.start(layer, label)
        goal.wait_stage("DRAG_RIGHT_TARGET_CALIBRATION_WAITING", 900.0)
        raw_box = observer.box_snapshot()
        print("\n右臂停在拖前目标；左臂仍在避让姿态，不会重新观测。")
        _calibrate_side(observer, "right", layer, raw_box, config_root)
        _wait_for_one(
            "确认右臂接触及 Drag1/2/3 路径安全；继续后会真实拖出料箱。"
            "输入 1 继续；0 退出："
        )
        _param_set(CONTINUE, True)
        result = goal.wait_result()
        with goal.condition:
            output = "\n".join([*goal.details, result.pickup_message])
        reanchored = _parse_reanchored_box_pose(output, raw_box.header.frame_id)
        print("\n右臂 Drag3 完成，左臂已通过 SDK MoveJ 到临时指定关节姿态。")
        print(f"实际料箱 Pose：{_format_array(_pose_values(reanchored.pose))}")
        _calibrate_side(observer, "left", layer, reanchored, config_root)
        print("本层完成。请人工安全恢复料箱和双臂后再标定下一层。")
    finally:
        goal_stopped = goal.cancel_if_running()
        if not goal_stopped:
            print(
                "警告：Action 可能仍在执行；保持标定暂停标志，"
                "请在示教器确认机器人安全并人工取消 Action。",
                file=sys.stderr,
            )
        for name in (
            CONTINUE,
            PAUSE,
            LEFT_HOLD,
            RIGHT_HOLD,
            LEFT_MOTION,
            LEFT_JOINT_OVERRIDE,
            LEFT_JOINT_TARGET,
        ):
            if name in saved:
                if not goal_stopped and name != CONTINUE:
                    continue
                try:
                    _param_set(name, False if name == CONTINUE else saved[name])
                except Exception as exc:
                    print(f"警告：恢复 {name} 失败：{exc}", file=sys.stderr)
        observer.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DragBox bigbox 单 Action 连续标定")
    parser.add_argument("start_layer", nargs="?", type=int, default=1)
    parser.add_argument("end_layer", nargs="?", type=int)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--config-root", type=Path, default=default_config_root())
    args = parser.parse_args(argv)
    end = args.end_layer if args.end_layer is not None else args.start_layer
    if not (1 <= args.start_layer <= end <= 4):
        parser.error("要求 1 <= 起始层 <= 结束层 <= 4")
    for layer in range(args.start_layer, end + 1):
        _calibrate_layer(layer, args.target_label, args.config_root)
        if layer < end:
            _wait_for_one(f"人工恢复到第 {layer + 1} 层可标定状态后，输入 1；0 退出：")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n标定已取消。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"标定失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
