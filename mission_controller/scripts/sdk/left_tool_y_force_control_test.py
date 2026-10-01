#!/usr/bin/env python3
"""Mirrored Tool-Y force-position test using only the RealMan Python SDK.

The left arm advances along +Tool Y and expects negative Fy. The right arm
advances along -Tool Y and expects positive Fy. ROS feedback is not used.
Default mode is read-only; pass --execute to permit motion.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time


SDK_PYTHON_ROOT = "/rm_nvme/recordings/code/RM_API2/Python"
if SDK_PYTHON_ROOT not in sys.path:
    sys.path.insert(0, SDK_PYTHON_ROOT)

from Robotic_Arm.rm_robot_interface import (
    RoboticArm,
    rm_force_position_move_t,
    rm_thread_mode_e,
)


ARM_ENDPOINTS = {
    "left": ("192.168.127.18", 8080),
    "right": ("192.168.127.19", 8080),
}

# Internal trajectory settings. The operational goal exposed to the user is
# only the requested Tool-Y force.
CONTROL_PERIOD_SEC = 0.020

# Fault-containment guards only. They do not participate in normal contact
# detection. A position-driven stream must not run forever when an object is
# absent or the sensor fails.
HARD_TRAVEL_LIMIT_M = 0.050
HARD_TIMEOUT_SEC = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="allow real robot motion")
    parser.add_argument(
        "--arm-side",
        choices=("left", "right"),
        default="left",
        help="left moves +Tool Y; right moves -Tool Y",
    )
    parser.add_argument(
        "--target-force-n",
        type=float,
        default=None,
        help="signed SDK Tool-Y target force; default is left=-1 N, right=+1 N",
    )
    parser.add_argument(
        "--speed-mm-s",
        type=float,
        default=5.0,
        help="Tool-Y approach speed in mm/s (converted to m/s for SDK limit_vel)",
    )
    return parser.parse_args()


def _frame_is_zero(frame: dict) -> bool:
    pose = frame.get("pose", [])
    return len(pose) == 6 and max(abs(float(v)) for v in pose) <= 1e-6


def _read_tool_force(robot: RoboticArm) -> tuple[float, float, float, float, float, float]:
    ret, data = robot.rm_get_force_data()
    if ret != 0:
        raise RuntimeError(f"rm_get_force_data failed: {ret}")
    values = tuple(float(v) for v in data["tool_zero_force_data"])
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise RuntimeError(f"invalid SDK tool_zero_force_data: {values}")
    return values


def _collect_sdk_force_baseline(
    robot: RoboticArm,
    duration_sec: float = 1.0,
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    samples: list[tuple[float, ...]] = []
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        samples.append(_read_tool_force(robot))
        time.sleep(0.005)
    columns = list(zip(*samples))
    median = tuple(statistics.median(column) for column in columns)
    std = tuple(statistics.pstdev(column) for column in columns)
    return median, std, len(samples)


def _stop_force_mode(robot: RoboticArm, started: bool, slow_stop: bool) -> None:
    if started:
        try:
            print("rm_stop_force_position_move ret=", robot.rm_stop_force_position_move())
        except Exception as exc:
            print("rm_stop_force_position_move exception:", repr(exc))
    if slow_stop:
        try:
            print("rm_set_arm_slow_stop ret=", robot.rm_set_arm_slow_stop())
        except Exception as exc:
            print("rm_set_arm_slow_stop exception:", repr(exc))


def main() -> int:
    args = parse_args()
    approach_sign = 1.0 if args.arm_side == "left" else -1.0
    target_force_n = (
        float(args.target_force_n)
        if args.target_force_n is not None
        else (-1.0 if args.arm_side == "left" else 1.0)
    )
    if not math.isfinite(target_force_n):
        raise ValueError("target_force_n must be finite")
    if args.arm_side == "left" and target_force_n >= 0.0:
        raise ValueError("left-arm +Tool-Y approach requires a negative target force")
    if args.arm_side == "right" and target_force_n <= 0.0:
        raise ValueError("right-arm -Tool-Y approach requires a positive target force")
    if not math.isfinite(args.speed_mm_s) or not 0.1 <= args.speed_mm_s <= 10.0:
        raise ValueError("speed_mm_s must be between 0.1 and 10.0 mm/s")
    approach_speed_m_s = args.speed_mm_s / 1000.0

    arm_ip, arm_port = ARM_ENDPOINTS[args.arm_side]
    robot = RoboticArm(rm_thread_mode_e(2))
    handle = robot.rm_create_robot_arm(arm_ip, arm_port, 3)
    if handle.id < 0:
        raise RuntimeError(
            f"failed to connect {args.arm_side} arm at {arm_ip}:{arm_port}: "
            f"handle={handle.id}"
        )

    force_started = False
    hard_guard_triggered = False
    try:
        tool_ret, tool = robot.rm_get_current_tool_frame()
        work_ret, work = robot.rm_get_current_work_frame()
        print("current_tool_frame:", tool_ret, tool)
        print("current_work_frame:", work_ret, work)
        if tool_ret != 0 or work_ret != 0:
            raise RuntimeError("failed to read current Tool/Work frame")
        if tool.get("name") != "Arm_Tip" or not _frame_is_zero(tool):
            raise RuntimeError("this test requires zero-offset Arm_Tip Tool frame")
        if work.get("name") != "World" or not _frame_is_zero(work):
            raise RuntimeError("this test requires zero-offset World Work frame")

        force_median, force_std, sample_count = _collect_sdk_force_baseline(robot)
        print(
            "SDK tool_zero_force_data baseline={} N/Nm, std={}, samples={}".format(
                tuple(round(v, 4) for v in force_median),
                tuple(round(v, 4) for v in force_std),
                sample_count,
            )
        )
        target_already_reached = (
            force_median[1] <= target_force_n
            if args.arm_side == "left"
            else force_median[1] >= target_force_n
        )
        if target_already_reached:
            raise RuntimeError(
                f"Tool Fy is already at/past target: {force_median[1]:.3f} N"
            )

        state_ret, state = robot.rm_get_current_arm_state()
        if state_ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed: {state_ret}")
        start_joint_deg = [float(v) for v in state["joint"]]
        if len(start_joint_deg) != 7:
            raise RuntimeError(f"unexpected SDK joint state: {start_joint_deg}")

        # Do not retransmit rm_get_current_arm_state()["pose"] here. Its Euler
        # angles are rounded to milliradians and this arm can operate near the
        # Euler pitch singularity (+/- pi/2), where that rounding can create a
        # large orientation/IK jump. Derive a precise pose from current joints
        # and stream it in quaternion form instead.
        start_pose_quat = [
            float(v) for v in robot.rm_algo_forward_kinematics(start_joint_deg, flag=0)
        ]
        start_pose_euler = [
            float(v) for v in robot.rm_algo_forward_kinematics(start_joint_deg, flag=1)
        ]
        if len(start_pose_quat) != 7 or len(start_pose_euler) != 6:
            raise RuntimeError(
                f"unexpected SDK FK result: quat={start_pose_quat}, "
                f"euler={start_pose_euler}"
            )

        matrix = robot.rm_algo_pos2matrix(start_pose_euler)
        data = [float(v) for v in matrix.data]
        tool_y_work = (data[1], data[5], data[9])
        axis_norm = math.sqrt(sum(v * v for v in tool_y_work))
        if axis_norm < 1e-9:
            raise RuntimeError("invalid Tool-Y direction from SDK pose matrix")
        tool_y_work = tuple(v / axis_norm for v in tool_y_work)

        print("start_joint_deg=", [round(v, 6) for v in start_joint_deg])
        print("start_pose_quat_wxyz=", [round(v, 6) for v in start_pose_quat])
        approach_axis_work = tuple(approach_sign * v for v in tool_y_work)
        direction_label = "+Tool Y" if approach_sign > 0.0 else "-Tool Y"
        print("approach_axis_in_Work=", [round(v, 6) for v in approach_axis_work])
        print(
            f"plan: arm={args.arm_side}; direction={direction_label}; "
            f"SDK Tool Fy target={target_force_n:.3f} N; "
            f"speed={args.speed_mm_s:.3f} mm/s; "
            "controller mode=Tool-Y force-tracking+motion"
        )

        if not args.execute:
            print("READ-ONLY CHECK COMPLETE. Add --execute to allow motion.")
            return 0

        phrase = input(
            f"Confirm a fixed box/soft pad is on {direction_label}, hands are clear, "
            "and E-stop is ready. Enter 1 to start: "
        )
        if phrase.strip() != "1":
            print("confirmation mismatch; no motion sent")
            return 2

        start_ret = robot.rm_start_force_position_move()
        print("rm_start_force_position_move ret=", start_ret)
        if start_ret != 0:
            raise RuntimeError(f"failed to start force-position streaming: {start_ret}")
        force_started = True

        begin = time.monotonic()
        next_tick = begin
        last_print = 0.0
        while True:
            now = time.monotonic()
            elapsed = now - begin
            tool_wrench = _read_tool_force(robot)
            tool_fy = tool_wrench[1]

            # The only normal completion condition: the SDK itself reports that
            # absolute Tool Fy has reached the requested negative force.
            force_target_reached = (
                tool_fy <= target_force_n
                if args.arm_side == "left"
                else tool_fy >= target_force_n
            )
            if force_target_reached:
                print(
                    f"SUCCESS: SDK Tool Fy reached target: "
                    f"Fy={tool_fy:.3f} N, target={target_force_n:.3f} N"
                )
                return 0

            distance = approach_speed_m_s * elapsed
            if distance >= HARD_TRAVEL_LIMIT_M:
                hard_guard_triggered = True
                raise RuntimeError(
                    "internal 50 mm fault-containment guard reached without "
                    f"SDK Tool Fy reaching {target_force_n:.3f} N"
                )
            if elapsed >= HARD_TIMEOUT_SEC:
                hard_guard_triggered = True
                raise RuntimeError(
                    "internal 60 s fault-containment timeout reached without "
                    f"SDK Tool Fy reaching {target_force_n:.3f} N"
                )

            target_pose = list(start_pose_quat)
            target_pose[0] += approach_axis_work[0] * distance
            target_pose[1] += approach_axis_work[1] * distance
            target_pose[2] += approach_axis_work[2] * distance
            param = rm_force_position_move_t(
                flag=1,
                pose=target_pose,
                sensor=1,
                mode=1,
                follow=False,
                control_mode=[0, 7, 0, 0, 0, 0],
                desired_force=[0.0, target_force_n, 0.0, 0.0, 0.0, 0.0],
                limit_vel=[0.0, approach_speed_m_s, 0.0, 0.0, 0.0, 0.0],
                trajectory_mode=0,
                radio=0,
            )
            command_ret = robot.rm_force_position_move(param)
            if command_ret != 0:
                hard_guard_triggered = True
                raise RuntimeError(f"rm_force_position_move failed: {command_ret}")

            if now - last_print >= 0.10:
                print(
                    f"t={elapsed:.2f}s commanded={distance * 1000.0:.3f}mm "
                    f"SDK_tool_fy={tool_fy:.3f}N target={target_force_n:.3f}N"
                )
                last_print = now

            next_tick += CONTROL_PERIOD_SEC
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif delay < -CONTROL_PERIOD_SEC:
                hard_guard_triggered = True
                raise RuntimeError(f"SDK streaming loop overrun: {-delay * 1000.0:.1f} ms")

    except KeyboardInterrupt:
        hard_guard_triggered = True
        print("Ctrl-C received")
        return 130
    except Exception as exc:
        print("TEST FAILED:", repr(exc))
        return 1
    finally:
        _stop_force_mode(robot, force_started, hard_guard_triggered)
        print("disconnect_ret=", robot.rm_delete_robot_arm())


if __name__ == "__main__":
    raise SystemExit(main())
