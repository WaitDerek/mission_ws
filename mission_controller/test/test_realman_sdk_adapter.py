import threading
import time
import unittest

from mission_runtime.realman_sdk_adapter import RealManSdkAdapter


class _SlowSuccessfulRobot:
    def rm_movel(self, *_args):
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


if __name__ == "__main__":
    unittest.main()
