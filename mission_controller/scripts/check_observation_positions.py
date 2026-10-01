#!/usr/bin/env python3
"""Interactive right-arm observation check. No vision/grasp/waist commands."""
import argparse
import math
from pathlib import Path
import signal
import sys
import threading
import time

import yaml

PACKAGE = Path('/rm_nvme/recordings/code/mission_ws/src/mission_controller')


def load_profiles(package=PACKAGE):
    params = {}
    for name in ('core', 'direct_motion', 'box_common', 'grasp_tf', 'drag'):
        params.update(yaml.safe_load((package / 'config/mission' / (name + '.yaml')).read_text())['mission_controller']['ros__parameters'])
    unit = float(params['box_pre_detection_right_movej_command_units_per_degree'])
    profiles = {}
    for number in range(1, 9):
        drag = number <= 4
        layer = number if drag else number - 4
        prefix, model = ('drag_box_tf', 'bigbox') if drag else ('grasp_box_tf', 'smallbox')
        key = f'{prefix}_box_layer_pre_detection_right_movej_joint_units_{model}_layer{layer}'
        angles = [float(v) / unit for v in params[key]]
        validate_angles(angles, params)
        profiles[number] = (f'{"Drag 大箱" if drag else "Grasp 小箱"} 第 {layer} 层', angles, drag)
    return params, profiles


def validate_angles(angles, params):
    if len(angles) != 7 or not all(math.isfinite(v) for v in angles):
        raise ValueError('需要七个有限关节角度')
    lower = params['waist_workspace_right_arm_joint_min_deg']
    upper = params['waist_workspace_right_arm_joint_max_deg']
    for i, (v, lo, hi) in enumerate(zip(angles, lower, upper), 1):
        if not lo <= v <= hi:
            raise ValueError(f'J{i}={v:.3f}° 超过配置限位 [{lo}, {hi}]')


def stage_indices(drag):
    return ((1,), (0, 1), tuple(range(7))) if drag else ((1,), tuple(range(7)))


def competing_controllers():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = [v.decode() for v in (path / 'cmdline').read_bytes().split(b'\0') if v]
        except (OSError, UnicodeError):
            continue
        if any(v.endswith('/mission_controller') or v.endswith('/execute_workflow') for v in args):
            found.append(path.name)
    return found


def print_profiles(profiles):
    print('\n编号  观察位置            右臂 J1～J7（度）')
    for number, (label, angles, _) in profiles.items():
        print(f'{number}     {label}: {[round(v, 3) for v in angles]}')
    print('输入 1～8：仅移动到所选观察位；q：退出；Ctrl+C：停止并退出。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='仅列出八组角度，不连接机器人')
    parser.add_argument('--check', action='store_true', help='只检查服务和反馈，不发送运动')
    parser.add_argument('--speed', type=int, default=10, help='速度百分比，默认 10，范围 1～15')
    args = parser.parse_args()
    if not 1 <= args.speed <= 15:
        parser.error('--speed 必须在 1～15 之间')
    params, profiles = load_profiles()
    print_profiles(profiles)
    if args.list:
        return 0

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    sys.path.insert(0, str(PACKAGE))
    from mission_runtime.realman_sdk_adapter import RealManSdkAdapter
    from mission_runtime.ros_arm_movej import RosArmMoveJ

    canceled = threading.Event()
    def request_stop(_signal, _frame):
        canceled.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('check_right_observation_positions')
    adapter = RealManSdkAdapter(
        sdk_root=params['direct_sdk_root'], left_ip=params['direct_sdk_left_ip'],
        right_ip=params['direct_sdk_right_ip'], port=int(params['direct_sdk_port']),
        connect_level=int(params['direct_sdk_connect_level']), logger=node.get_logger())
    transport = RosArmMoveJ(
        node, stop_arm=adapter.stop_arm,
        position_tolerance_rad=0.01, velocity_tolerance_rad_sec=0.01,
        stable_samples=3)
    adapter._ros_movej_transport = transport
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    motion_started = False

    def current_angles():
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if canceled.is_set():
                raise KeyboardInterrupt
            with transport._lock:
                seq, stamp, positions, velocities = transport._feedback['right']
            if seq and len(positions) == 7 and time.monotonic() - stamp < 1.0:
                if max(abs(v) for v in velocities) > 0.05:
                    raise RuntimeError('右臂仍在运动，未发送新目标')
                return [math.degrees(v) for v in positions]
            time.sleep(0.05)
        raise RuntimeError('没有新鲜的右臂反馈 /mcap/slave_arm_right')

    try:
        if not transport._client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/robot/command 服务不可用')
        print('当前右臂角度：', [round(v, 3) for v in current_angles()])
        if args.check:
            print('服务和右臂反馈正常；没有发送运动指令。')
            return 0
        while not canceled.is_set():
            choice = input('\n观察位 [1-8 / q] > ').strip().lower()
            if choice in ('q', 'quit', 'exit'):
                break
            if choice not in tuple(str(i) for i in range(1, 9)):
                print('请输入 1～8 或 q。')
                continue
            blockers = competing_controllers()
            if blockers:
                raise RuntimeError(f'mission/全流程节点仍在运行，先关闭：PID {blockers}')
            # Reload so subsequent manual calibration edits take effect.
            params, profiles = load_profiles()
            label, final, drag = profiles[int(choice)]
            print(f'执行 {label}，右臂目标（度）：{final}，速度 {args.speed}%')
            for indices in stage_indices(drag):
                target = current_angles()
                for index in indices:
                    target[index] = final[index]
                validate_angles(target, params)
                blockers = competing_controllers()
                if blockers:
                    raise RuntimeError(f'mission/全流程节点重新启动，停止测试：PID {blockers}')
                print('  阶段', '+'.join(f'J{i+1}' for i in indices), flush=True)
                motion_started = True
                result = adapter.execute_single_movej(
                    arm='right', joint_degrees=target, speed_percent=args.speed,
                    cancel_requested=canceled.is_set, timeout_sec=120.0)
                motion_started = False
                print(' ', result)
            print(f'已到达 {label}。保持当前位置，等待下一次输入。')
        return 0
    except (KeyboardInterrupt, EOFError):
        canceled.set()
        if motion_started:
            adapter.stop_arm('right')
        print('\n已退出。' if not motion_started else '\n已请求右臂减速停止，退出测试。')
        return 130 if motion_started else 0
    except Exception as exc:
        canceled.set()
        if motion_started:
            adapter.stop_arm('right')
        print(f'测试停止：{exc}', file=sys.stderr)
        return 1
    finally:
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        adapter.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
