#!/usr/bin/env python3
"""Right-arm Tool-Y force-only test; sends no MoveL or streaming targets.

The fixture must already be in light contact with a compliant test object.
Default invocation is read-only. Real force mode requires --execute and an
interactive confirmation. This does not prove that force mode can seek an
object across a gap: the SDK documents this mode for Cartesian trajectories.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time


SDK_ROOT = "/rm_nvme/recordings/code/RM_API2/Python"
RIGHT_IP = "192.168.127.19"
RIGHT_PORT = 8080


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable real force control")
    parser.add_argument("--target-force-n", type=float, default=1.0,
                        help="positive right-arm Tool Fy target; default 1 N")
    parser.add_argument("--limit-vel-m-s", type=float, default=0.003,
                        help="Tool-Y force-axis speed limit; default 0.003 m/s")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--max-displacement-mm", type=float, default=50.0,
                        help="software-monitored actual TCP displacement limit, not a controller travel command")
    parser.add_argument("--max-tool-fy-n", type=float, default=3.0)
    return parser.parse_args()


def checked_values(args: argparse.Namespace) -> None:
    if not 0.0 < args.target_force_n <= 2.0:
        raise ValueError("target force must be in (0, 2] N for this test")
    if not 0.0 < args.limit_vel_m_s <= 0.003:
        raise ValueError("limit velocity must be in (0, 0.003] m/s")
    if not 0.5 <= args.timeout_sec <= 60.0:
        raise ValueError("timeout must be between 0.5 and 60 s")
    if not 0.0 < args.max_displacement_mm <= 50.0:
        raise ValueError("maximum displacement must be in (0, 50] mm")
    if not args.target_force_n < args.max_tool_fy_n <= 5.0:
        raise ValueError("force ceiling must exceed target and be <= 5 N")


def sdk_force(robot) -> float:
    code, data = robot.rm_get_force_data()
    if code != 0:
        raise RuntimeError(f"rm_get_force_data returned {code}")
    value = float(data["tool_zero_force_data"][1])
    if not math.isfinite(value):
        raise RuntimeError("Tool Fy is not finite")
    return value


def sdk_pose(robot) -> list[float]:
    code, state = robot.rm_get_current_arm_state()
    if code != 0:
        raise RuntimeError(f"rm_get_current_arm_state returned {code}")
    pose = [float(value) for value in state["pose"]]
    if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
        raise RuntimeError(f"unexpected SDK EEPose: {pose}")
    return pose


def check_joint_alarm(robot) -> None:
    state = robot.rm_get_joint_err_flag()
    if state.get("return_code") != 0:
        raise RuntimeError(f"rm_get_joint_err_flag failed: {state}")
    if any(int(code) != 0 for code in state.get("err_flag", [])):
        raise RuntimeError(f"controller joint alarm: {state['err_flag']}")


def displacement_mm(a: list[float], b: list[float]) -> float:
    return 1000.0 * math.dist(a[:3], b[:3])


def main() -> int:
    args = arguments()
    checked_values(args)
    if SDK_ROOT not in sys.path:
        sys.path.insert(0, SDK_ROOT)
    from Robotic_Arm.rm_robot_interface import (
        RoboticArm,
        rm_force_position_t,
        rm_thread_mode_e,
    )

    robot = RoboticArm(rm_thread_mode_e(2))
    handle = robot.rm_create_robot_arm(RIGHT_IP, RIGHT_PORT, 3)
    if handle.id < 0:
        raise RuntimeError(f"could not connect to right arm: handle={handle.id}")

    mode_started = False
    fault = False
    try:
        check_joint_alarm(robot)
        tool_code, tool_frame = robot.rm_get_current_tool_frame()
        if tool_code != 0 or tool_frame.get("name") != "Arm_Tip":
            raise RuntimeError(f"需要 Arm_Tip 工具坐标系，当前为：{tool_frame}")
        start_pose = sdk_pose(robot)
        samples = []
        for _ in range(6):
            samples.append((sdk_pose(robot), sdk_force(robot)))
            time.sleep(0.05)
        baseline = statistics.median(force for _, force in samples)
        noise_span = max(force for _, force in samples) - min(force for _, force in samples)
        pose_span = max(displacement_mm(start_pose, pose) for pose, _ in samples)
        print(f"右臂 Tool Fy={baseline:.3f} N; 采样跨度={noise_span:.3f} N; "
              f"末端漂移={pose_span:.3f} mm")
        print(f"计划：仅启用一次 Tool-Y 力跟踪；目标 +{args.target_force_n:.2f} N；"
              f"速度上限 {args.limit_vel_m_s * 1000:.1f} mm/s；"
              f"监测位移上限 {args.max_displacement_mm:.1f} mm；"
              f"总超时 {args.timeout_sec:.1f} s。")
        print("注意：位移上限由脚本轮询监测，不是控制器内置的硬行程限制。")
        if noise_span > 0.3 or pose_span > 0.3:
            raise RuntimeError("力或末端位置尚未稳定，不启用力控")
        if abs(baseline) >= args.max_tool_fy_n:
            raise RuntimeError("当前 Tool Fy 已超过测试上限")
        if not args.execute:
            print("只读检查结束。加 --execute 并在现场确认后才会启用力控。")
            return 0

        phrase = input(
            "确认右夹具已轻触软质测试物、周围无人、急停可用；"
            "不会发送 MoveL。输入 FORCE RIGHT 确认："
        )
        if phrase != "FORCE RIGHT":
            print("未确认；没有启用力控。")
            return 2

        param = rm_force_position_t()
        param.sensor = 1
        param.mode = 1                 # Tool frame
        param.control_mode[:] = [0, 4, 0, 0, 0, 0]  # Tool-Y force tracking
        param.desired_force[:] = [0.0, args.target_force_n, 0.0, 0.0, 0.0, 0.0]
        param.limit_vel[:] = [0.0, args.limit_vel_m_s, 0.0, 0.0, 0.0, 0.0]

        print("启用一次 rm_set_force_position_new；不发送任何运动轨迹。")
        code = robot.rm_set_force_position_new(param)
        if code != 0:
            raise RuntimeError(f"rm_set_force_position_new returned {code}")
        mode_started = True
        started = time.monotonic()
        reached_since = None
        max_seen_mm = 0.0
        last_print = -1.0
        while time.monotonic() - started < args.timeout_sec:
            check_joint_alarm(robot)
            fy = sdk_force(robot)
            pose = sdk_pose(robot)
            moved_mm = displacement_mm(start_pose, pose)
            max_seen_mm = max(max_seen_mm, moved_mm)
            elapsed = time.monotonic() - started
            if elapsed - last_print >= 0.5:
                print(f"{elapsed:.2f}s Tool Fy={fy:+.3f} N, 末端位移={moved_mm:.3f} mm")
                last_print = elapsed
            if abs(fy) >= args.max_tool_fy_n:
                fault = True
                raise RuntimeError(f"Tool Fy 达到监测上限：{fy:.3f} N")
            if moved_mm >= args.max_displacement_mm:
                fault = True
                raise RuntimeError(f"实际末端位移达到监测上限：{moved_mm:.3f} mm")
            if abs(fy - args.target_force_n) <= 0.3:
                reached_since = reached_since or time.monotonic()
                if time.monotonic() - reached_since >= 0.2:
                    print(f"目标力已维持约 0.2 s；最大实际位移 {max_seen_mm:.3f} mm。")
                    return 0
            else:
                reached_since = None
            time.sleep(0.1)
        print(f"超时：没有证明单独开启力控能自主达到目标力；最大实际位移 {max_seen_mm:.3f} mm。")
        return 1
    except KeyboardInterrupt:
        fault = True
        print("收到 Ctrl-C，停止测试。")
        return 130
    except Exception as exc:
        fault = True
        print(f"测试失败：{exc}")
        return 1
    finally:
        if mode_started:
            try:
                print("rm_stop_force_position 返回：", robot.rm_stop_force_position())
            except Exception as exc:
                print(f"停止力控失败：{exc}；请立即使用实体急停/示教器确认状态")
        if fault:
            try:
                print("rm_set_arm_slow_stop 返回：", robot.rm_set_arm_slow_stop())
            except Exception as exc:
                print(f"慢停失败：{exc}；请使用实体急停")
        print("断开右臂连接：", robot.rm_delete_robot_arm())


if __name__ == "__main__":
    raise SystemExit(main())
