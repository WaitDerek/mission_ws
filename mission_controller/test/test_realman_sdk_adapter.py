import threading
import time
import unittest

from mission_runtime.realman_sdk_adapter import RealManSdkAdapter


class _SlowSuccessfulRobot:
    def __init__(self):
        self.calls = []

    def rm_movel(self, *_args):
        self.calls.append("movel")
        time.sleep(0.06)
        return 0

    def rm_movej_p(self, *_args):
        self.calls.append("movej_p")
        time.sleep(0.06)
        return 0

    def rm_movel_offset(self, *_args):
        self.calls.append(("movel_offset", _args))
        time.sleep(0.06)
        return 0

    def rm_movej(self, *_args):
        self.calls.append(("movej", _args))
        time.sleep(0.06)
        return 0


class TestRealManSdkAdapter(unittest.TestCase):
    def test_connected_waypoints_accepts_and_runs_progress_callback(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        progress_samples = []
        target = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
        result = adapter.execute_dual_movel_connected_waypoints(
            [target],
            [target],
            5.0,
            5.0,
            progress_callback=lambda: progress_samples.append(True),
        )

        self.assertTrue(progress_samples)
        self.assertIn("connected", result)

    def test_connected_waypoints_allow_successful_progress_stop(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        stop_calls = []
        abort_calls = []
        adapter.stop_all = lambda: stop_calls.append(True)

        target = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
        result = adapter.execute_dual_movel_connected_waypoints(
            [target],
            [target],
            5.0,
            5.0,
            progress_callback=lambda: False,
            abort_callback=lambda: abort_calls.append(True),
        )

        self.assertTrue(stop_calls)
        self.assertEqual(abort_calls, [True])
        self.assertIn("stopped_by_progress_callback=true", result)

    def test_endpoint_movej_p_dispatches_both_arm_commands(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        target = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
        result = adapter.execute_dual_movel_endpoint(
            target,
            target,
            5.0,
            5.0,
            motion_mode="movej_p",
        )

        self.assertEqual(robot.calls, ["movej_p", "movej_p"])
        self.assertIn("movej_p", result)

    def test_connected_movej_p_dispatches_final_waypoint(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        target = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
        result = adapter.execute_dual_movel_connected_waypoints(
            [target],
            [target],
            5.0,
            5.0,
            motion_mode="movej_p",
        )

        self.assertEqual(robot.calls, ["movej_p", "movej_p"])
        self.assertIn("movej_p", result)

    def test_connected_movel_offset_dispatches_work_frame_waypoints(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        offsets = [
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        result = adapter.execute_dual_movel_connected_waypoints(
            offsets,
            offsets,
            12.0,
            12.0,
            motion_mode="movel_offset",
            offset_frame_type=0,
        )

        calls = [call for call in robot.calls if call[0] == "movel_offset"]
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call[1][4] == 0 for call in calls))
        self.assertIn("movel_offset", result)

    def test_dual_movej_dispatches_both_arms(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        joints = [0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = adapter.execute_dual_movej(joints, joints, 12.0)

        calls = [call for call in robot.calls if call[0] == "movej"]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[1][0][1] == 60.0 for call in calls))
        self.assertIn("dual-arm movej", result)

    def test_single_movel_offset_uses_work_frame(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_arm = lambda _arm: None

        offset = [0.01, -0.02, 0.03, 0.0, 0.0, 0.0]
        result = adapter.execute_single(
            "right",
            offset,
            "movel_offset",
            12.0,
            True,
            offset_frame_type=0,
        )

        name, args = robot.calls[0]
        self.assertEqual(name, "movel_offset")
        self.assertEqual(args, (offset, 12, 0, 0, 0, 1))
        self.assertIn("movel_offset", result)

    def test_dual_movel_offset_dispatches_both_arms(self):
        adapter = object.__new__(RealManSdkAdapter)
        robot = _SlowSuccessfulRobot()
        adapter._motion_lock = threading.Lock()
        adapter._stop_event = threading.Event()
        adapter._motion_active = False
        adapter._connect = lambda: None
        adapter._robots = lambda: (robot, robot)
        adapter.stop_all = lambda: None

        offset = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = adapter.execute_dual(
            offset,
            offset,
            "movel_offset",
            10.0,
            True,
            offset_frame_type=0,
        )

        self.assertEqual([call[0] for call in robot.calls], ["movel_offset"] * 2)
        self.assertIn("movel_offset", result)


if __name__ == "__main__":
    unittest.main()
