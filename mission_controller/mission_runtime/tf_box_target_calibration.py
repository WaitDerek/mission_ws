"""One entry point for all GraspBox/DragBox TF box-side target calibrations.

GraspBox reaches both box sides before contact. DragBox reaches only the right
side before the pull; its left-side target exists only after Drag3/reanchoring.
Each run calibrates one model/layer and requires operator confirmation before
moving the robot and before updating live parameters and source defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import (
    drag_left_after_pull_calibration,
    drag_right_target_calibration,
    grasp_dual_target_calibration,
)
from .tf_calibration_defaults import default_config_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="通用料箱侧面目标标定：手动示教后写入当前参数和 YAML 默认值"
    )
    parser.add_argument("--action", choices=("grasp", "drag"), required=True)
    parser.add_argument("--box-type", choices=("bigbox", "smallbox"), required=True)
    parser.add_argument("--box-layer", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--arm", choices=("both", "left", "right"))
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--config-root", type=Path, default=default_config_root(),
        help="Mission 源码 config/mission 目录；默认按脚本源码位置寻找",
    )
    args = parser.parse_args(argv)

    if args.action == "grasp":
        if args.arm not in (None, "both"):
            parser.error("GraspBox 初始侧面目标是双臂同时标定；使用 --arm both")
        calibrator = grasp_dual_target_calibration
        print("模式：GraspBox 双臂初始目标；到位后停住，不执行夹紧。")
    else:
        if args.arm not in ("left", "right"):
            parser.error("DragBox 请指定 --arm right（拖前目标）或 --arm left（Drag3后加入目标）")
        calibrator = (
            drag_right_target_calibration
            if args.arm == "right"
            else drag_left_after_pull_calibration
        )
        if args.arm == "right":
            print("模式：DragBox 右臂初始目标；到位后停住，不执行拖拽。")
        else:
            print("模式：DragBox Drag3 后左臂加入目标；会真实执行右臂接触与拖拽。")
    print("只有您两次确认后才会启动实机动作及保存；不会自动连续运行四层。", flush=True)

    forwarded = [
        "--box-type", args.box_type,
        "--box-layer", str(args.box_layer),
        "--target-label", str(args.target_label),
        "--config-root", str(args.config_root),
        "--save-defaults",
    ]
    return calibrator.main(forwarded)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n标定已退出。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"标定失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
