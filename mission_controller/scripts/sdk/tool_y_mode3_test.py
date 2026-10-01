#!/usr/bin/env python3
"""Single-arm Tool-Y mode-3 diagnostic; no motion without --execute.

This is a bounded position-streaming test, NOT force-controlled grasping.
Contact is detected from SDK Tool Fy and immediately ends the stream.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

SDK_ROOT = "/rm_nvme/recordings/code/RM_API2/Python"
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from Robotic_Arm.rm_robot_interface import (  # noqa: E402
    RoboticArm,
    rm_force_position_move_t,
    rm_thread_mode_e,
)

ARMS = {"left": "192.168.127.18", "right": "192.168.127.19"}
PERIOD_SEC = 0.020
SPEED_M_S = 0.003
MAX_COMMAND_M = 0.050
TIMEOUT_SEC = 20.0
CONTACT_DELTA_N = 1.0
MAX_FORCE_DELTA_N = 5.0
MAX_TORQUE_DELTA_NM = 1.0


def get_force(robot):
    code, data = robot.rm_get_force_data()
    if code != 0:
        raise RuntimeError(f"rm_get_force_data failed: {code}")
    value = tuple(float(x) for x in data["tool_zero_force_data"])
    if len(value) != 6 or not all(math.isfinite(x) for x in value):
        raise RuntimeError(f"invalid SDK Tool wrench: {value}")
    return value


def zero_frame(frame):
    return (
        len(frame.get("pose", [])) == 6
        and max(abs(float(x)) for x in frame["pose"]) <= 1e-6
    )


def tool_y_in_work(pose):
    # SDK quaternion pose: [x,y,z,w,qx,qy,qz]. Rotate local +Y into Work.
    w, x, y, z = (float(v) for v in pose[3:7])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        raise RuntimeError("invalid FK quaternion")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    axis = (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w))
    length = math.sqrt(sum(v * v for v in axis))
    return tuple(v / length for v in axis)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-side", choices=ARMS, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    sign = 1.0 if args.arm_side == "left" else -1.0
    direction = "+Tool Y" if sign > 0 else "-Tool Y"

    robot = RoboticArm(rm_thread_mode_e(2))
    handle = robot.rm_create_robot_arm(ARMS[args.arm_side], 8080, 3)
    if handle.id < 0:
        raise RuntimeError(f"failed to connect {args.arm_side}: handle={handle.id}")

    started = False
    try:
        tool_code, tool = robot.rm_get_current_tool_frame()
        work_code, work = robot.rm_get_current_work_frame()
        if tool_code or work_code:
            raise RuntimeError(f"frame query failed: tool={tool_code}, work={work_code}")
        if tool.get("name") != "Arm_Tip" or not zero_frame(tool):
            raise RuntimeError(f"expected zero-offset Arm_Tip, got {tool}")
        if work.get("name") != "World" or not zero_frame(work):
            raise RuntimeError(f"expected zero-offset World, got {work}")

        samples = []
        baseline_end = time.monotonic() + 0.5
        while time.monotonic() < baseline_end:
            samples.append(get_force(robot))
            time.sleep(0.01)
        if len(samples) < 10:
            raise RuntimeError("too few force baseline samples")
        baseline = tuple(statistics.median(col) for col in zip(*samples))
        fy_noise = statistics.pstdev([s[1] for s in samples])
        if fy_noise > 0.20:
            raise RuntimeError(f"Tool Fy baseline is unstable: std={fy_noise:.3f} N")

        state_code, state = robot.rm_get_current_arm_state()
        if state_code:
            raise RuntimeError(f"arm state read failed: {state_code}")
        joints = [float(v) for v in state["joint"]]
        if len(joints) != 7:
            raise RuntimeError(f"expected 7 joints, got {joints}")
        start_pose = [float(v) for v in robot.rm_algo_forward_kinematics(joints, flag=0)]
        if len(start_pose) != 7 or not all(math.isfinite(v) for v in start_pose):
            raise RuntimeError(f"invalid FK pose: {start_pose}")
        axis = tuple(sign * v for v in tool_y_in_work(start_pose))
        print(f"arm={args.arm_side}, direction={direction}, axis_in_Work={axis}")
        print(f"speed=3 mm/s, commanded travel limit=50 mm, timeout=20 s")
        print(f"contact stop: Tool Fy change {CONTACT_DELTA_N} N opposite motion")
        print(f"baseline Tool Fy={baseline[1]:.3f} N; noise={fy_noise:.3f} N")
        if not args.execute:
            print("READ-ONLY: no movement. Add --execute for real movement.")
            return 0

        answer = input(
            f"确认 {args.arm_side} 臂沿 {direction} 的路径已清空、急停可用；"
            "接触即停止，最多指令位移50mm。输入 MOVE 执行："
        )
        if answer.strip() != "MOVE":
            print("未确认，不发送运动")
            return 2

        code = robot.rm_start_force_position_move()
        if code:
            raise RuntimeError(f"rm_start_force_position_move failed: {code}")
        started = True
        begin = time.monotonic()
        next_tick = begin
        last_log = begin
        while True:
            now = time.monotonic()
            elapsed = now - begin
            force = get_force(robot)
            delta = tuple(value - initial for value, initial in zip(force, baseline))
            force_norm = math.sqrt(sum(v * v for v in delta[:3]))
            torque_norm = math.sqrt(sum(v * v for v in delta[3:]))
            if force_norm >= MAX_FORCE_DELTA_N or torque_norm >= MAX_TORQUE_DELTA_NM:
                raise RuntimeError(
                    f"force/torque guard: delta={delta}, norms={force_norm:.3f} N/"
                    f"{torque_norm:.3f} Nm"
                )
            # Reaction force is opposite to the commanded Tool-Y direction.
            if -sign * delta[1] >= CONTACT_DELTA_N:
                print(f"CONTACT: Tool Fy delta={delta[1]:.3f} N; stopping")
                return 0
            if sign * delta[1] >= CONTACT_DELTA_N:
                raise RuntimeError(f"unexpected Tool Fy direction: delta={delta[1]:.3f} N")
            distance = SPEED_M_S * elapsed
            if distance >= MAX_COMMAND_M:
                raise RuntimeError("50 mm command limit reached without contact")
            if elapsed >= TIMEOUT_SEC:
                raise RuntimeError("20 s timeout reached")

            pose = list(start_pose)
            for i in range(3):
                pose[i] += distance * axis[i]
            param = rm_force_position_move_t(
                flag=1,
                pose=pose,
                sensor=1,
                mode=1,
                follow=False,
                control_mode=[0, 3, 0, 0, 0, 0],
                desired_force=[0.0] * 6,
                limit_vel=[0.0, SPEED_M_S, 0.0, 0.0, 0.0, 0.0],
                # In mode 3 this is not the speed limit; the pose ramp is.
                trajectory_mode=0,
                radio=0,
            )
            code = robot.rm_force_position_move(param)
            if code:
                raise RuntimeError(f"rm_force_position_move failed: {code}")
            if now - last_log >= 0.5:
                print(f"elapsed={elapsed:.2f}s, commanded={distance*1000:.2f}mm, "
                      f"Tool Fy delta={delta[1]:.3f}N")
                last_log = now
            next_tick += PERIOD_SEC
            lag = next_tick - time.monotonic()
            if lag > 0:
                time.sleep(lag)
            elif lag < -PERIOD_SEC:
                raise RuntimeError(f"stream overrun={-lag*1000:.1f} ms")
    except KeyboardInterrupt:
        print("Ctrl-C: stopping stream")
        return 130
    except Exception as exc:
        print("TEST FAILED:", exc)
        return 1
    finally:
        if started:
            try:
                print("rm_stop_force_position_move ret=", robot.rm_stop_force_position_move())
            except Exception as exc:
                print("stop force mode error:", exc)
            try:
                print("rm_set_arm_slow_stop ret=", robot.rm_set_arm_slow_stop())
            except Exception as exc:
                print("slow stop error:", exc)
        print("disconnect_ret=", robot.rm_delete_robot_arm())


if __name__ == "__main__":
    raise SystemExit(main())
