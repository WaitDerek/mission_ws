#!/usr/bin/env python3
"""Detect one box and print its model-origin X in base_footprint."""

from __future__ import annotations

import argparse
import sys
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from object_pose_interfaces.action import EstimateObjectPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener


def wait_future(future, timeout_sec: float, description: str):
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and not future.done():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"等待{description}超过 {timeout_sec:g} 秒")
        time.sleep(0.05)
    if not future.done():
        raise RuntimeError(f"等待{description}时 ROS 已停止")
    return future.result()


def main() -> int:
    parser = argparse.ArgumentParser(description="检测料箱并输出 base_footprint 下的 X")
    parser.add_argument("--camera-side", choices=("left", "right"), default="left")
    parser.add_argument("--model-label", default="bigbox")
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--target-frame", default="base_footprint")
    parser.add_argument("--result-timeout-sec", type=float, default=180.0)
    args = parser.parse_args()

    rclpy.init()
    node = Node("measure_box_basefoot_x")
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        buffer = Buffer(cache_time=Duration(seconds=600.0))
        listener = TransformListener(buffer, node)
        client = ActionClient(node, EstimateObjectPose, "/object_pose/estimate")
        if not client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("/object_pose/estimate Action 服务不可用")

        # Start caching live TF before the camera image is captured.
        time.sleep(1.0)
        goal = EstimateObjectPose.Goal()
        goal.camera_side = args.camera_side
        goal.model_label = args.model_label
        goal.instance_index = args.instance_index
        goal.confidence_threshold = args.confidence_threshold
        goal.detection_domain = ""
        print(
            f"正在检测：{args.camera_side} 相机，{args.model_label}，"
            f"instance_index={args.instance_index} ...",
            flush=True,
        )

        goal_handle = wait_future(
            client.send_goal_async(goal), 20.0, "Action 接收目标"
        )
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("检测目标被 Action 拒绝")

        result_future = goal_handle.get_result_async()
        try:
            wrapped = wait_future(
                result_future, args.result_timeout_sec, "FoundationPose 检测结果"
            )
        except TimeoutError:
            goal_handle.cancel_goal_async()
            raise
        result = wrapped.result
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
            raise RuntimeError(f"检测失败：{result.message}")

        detected = result.pose
        source_frame = detected.header.frame_id.strip().lstrip("/")
        if not source_frame:
            raise RuntimeError("检测结果 pose.header.frame_id 为空，不能转换坐标")
        capture_time = Time.from_msg(detected.header.stamp)
        if capture_time.nanoseconds == 0:
            raise RuntimeError("检测结果没有有效时间戳，不能可靠匹配检测时的 TF")
        try:
            transform = buffer.lookup_transform(
                args.target_frame,
                source_frame,
                capture_time,
                timeout=Duration(seconds=10.0),
            )
        except TransformException as exc:
            raise RuntimeError(
                f"无法取得检测时刻的 {source_frame} → {args.target_frame} TF：{exc}"
            ) from exc
        box_pose = do_transform_pose(detected.pose, transform)
        x = box_pose.position.x
        print(f"检测帧：{source_frame}")
        print(f"目标帧：{args.target_frame}")
        print(f"料箱 Pose 原点 X = {x:.6f} m ({x * 1000.0:.1f} mm)")
        return 0
    except (RuntimeError, TimeoutError) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
