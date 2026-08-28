from io import StringIO
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s2_ugv_bridge.contract import validate_request
from s2_ugv_bridge.docker_ros1 import (
    DockerRos1Config,
    build_cancel_argv,
    build_goal_argv,
    build_server_check_argv,
    run_translation,
)


class _CompletedProcess:
    def __init__(self, returncode):
        self.returncode = returncode


class _RunningProcess:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


class DockerRos1CommandTests(unittest.TestCase):
    def setUp(self):
        self.config = DockerRos1Config()
        self.request = validate_request("forward", 5.0, 0.08)

    def test_preflight_wait_is_bounded_and_checks_the_ros1_action_goal_topic(self):
        argv = build_server_check_argv(self.config)

        self.assertEqual(["docker", "exec", "unitree", "/bin/bash", "-lc"], argv[:5])
        self.assertIn("timeout 10", argv[-1])
        self.assertIn("/timed_translate/goal", argv[-1])

    def test_goal_command_uses_the_existing_ros1_client_with_all_three_parameters(self):
        argv = build_goal_argv(self.request, self.config)

        self.assertEqual(["docker", "exec", "unitree", "/bin/bash", "-lc"], argv[:5])
        self.assertIn("rosrun s2_ugv_mission timed_translate_client.py forward 5 --speed-mps 0.08", argv[-1])
        self.assertIn("source /opt/ros/noetic/setup.bash", argv[-1])

    def test_cancel_command_publishes_to_the_ros1_action_cancel_topic(self):
        argv = build_cancel_argv(self.config)

        self.assertIn("/timed_translate/cancel", argv[-1])
        self.assertIn("actionlib_msgs/GoalID", argv[-1])

    def test_ctrl_c_cancels_the_ros1_action_before_terminating_the_local_docker_client(self):
        calls = []
        process = _RunningProcess()

        def fake_run(argv, **_kwargs):
            calls.append(argv)
            return _CompletedProcess(0)

        def fake_sleep(_seconds):
            raise KeyboardInterrupt

        exit_code = run_translation(
            self.request,
            self.config,
            run=fake_run,
            popen=lambda _argv, **_kwargs: process,
            monotonic=lambda: 0.0,
            sleep=fake_sleep,
            stderr=StringIO(),
        )

        self.assertEqual(130, exit_code)
        self.assertTrue(process.terminated)
        self.assertTrue(any("/timed_translate/cancel" in command[-1] for command in calls))

    def test_missing_ros1_action_returns_a_readable_nonzero_error_without_starting_client(self):
        stderr = StringIO()
        client_started = False

        exit_code = run_translation(
            self.request,
            self.config,
            run=lambda _argv, **_kwargs: _CompletedProcess(1),
            popen=lambda _argv, **_kwargs: self.fail("Docker client must not start"),
            stderr=stderr,
        )

        self.assertEqual(70, exit_code)
        self.assertIn("timed_translate", stderr.getvalue())

    def test_action_wait_timeout_cancels_then_returns_nonzero(self):
        calls = []
        process = _RunningProcess()
        times = iter((0.0, 21.0))

        def fake_run(argv, **_kwargs):
            calls.append(argv)
            return _CompletedProcess(0)

        exit_code = run_translation(
            self.request,
            DockerRos1Config(completion_margin_s=15.0),
            run=fake_run,
            popen=lambda _argv, **_kwargs: process,
            monotonic=lambda: next(times),
            sleep=lambda _seconds: None,
            stderr=StringIO(),
        )

        self.assertEqual(124, exit_code)
        self.assertTrue(process.terminated)
        self.assertTrue(any("/timed_translate/cancel" in command[-1] for command in calls))
