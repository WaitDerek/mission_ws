from io import StringIO
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s2_ugv_bridge.cli import main


class TimedTranslateCliTests(unittest.TestCase):
    def test_valid_command_forwards_all_user_parameters_to_the_gateway(self):
        calls = []

        def fake_run_translation(request, config):
            calls.append((request, config))
            return 0

        stderr = StringIO()
        exit_code = main(
            [
                "forward",
                "5",
                "--speed-mps",
                "0.08",
                "--container",
                "unitree-test",
                "--server-timeout-s",
                "12",
            ],
            run_translation_fn=fake_run_translation,
            stderr=stderr,
        )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        request, config = calls[0]
        self.assertEqual("forward", request.direction)
        self.assertEqual(5.0, request.duration_s)
        self.assertEqual(0.08, request.speed_mps)
        self.assertEqual("unitree-test", config.container)
        self.assertEqual(12.0, config.server_timeout_s)

    def test_omitted_speed_is_forwarded_as_zero_to_keep_the_ros1_default(self):
        calls = []

        exit_code = main(
            ["left", "2"],
            run_translation_fn=lambda request, config: calls.append((request, config)) or 0,
            stderr=StringIO(),
        )

        self.assertEqual(0, exit_code)
        self.assertEqual(0.0, calls[0][0].speed_mps)

    def test_invalid_motion_input_returns_two_without_calling_docker(self):
        called = False
        stderr = StringIO()

        def fake_run_translation(_request, _config):
            nonlocal called
            called = True
            return 0

        exit_code = main(
            ["diagonal", "0"],
            run_translation_fn=fake_run_translation,
            stderr=stderr,
        )

        self.assertEqual(2, exit_code)
        self.assertFalse(called)
        self.assertIn("错误", stderr.getvalue())

    def test_invalid_runtime_timeout_returns_two_without_calling_docker(self):
        called = False
        stderr = StringIO()

        def fake_run_translation(_request, _config):
            nonlocal called
            called = True
            return 0

        exit_code = main(
            ["forward", "1", "--server-timeout-s", "0"],
            run_translation_fn=fake_run_translation,
            stderr=stderr,
        )

        self.assertEqual(2, exit_code)
        self.assertFalse(called)
        self.assertIn("server_timeout_s", stderr.getvalue())

