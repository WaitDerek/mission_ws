import threading
import time
import unittest

from mission_runtime.taskflow.model import NavigationRequest, NavigationResult
from mission_runtime.taskflow.navigation import (
    DisabledNavigationGateway,
    FakeNavigationGateway,
)


class TestDepalletizingNavigationGateway(unittest.TestCase):
    def test_disabled_gateway_fails_closed(self):
        result = DisabledNavigationGateway().navigate(
            NavigationRequest("workflow", "step", "1"), lambda: False
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, "unavailable")

    def test_fake_gateway_records_scripted_results(self):
        gateway = FakeNavigationGateway(
            [NavigationResult(False, "failed", "platform refused")]
        )
        request = NavigationRequest("workflow", "step", "5")

        result = gateway.navigate(request, lambda: False)

        self.assertEqual(gateway.requests, [request])
        self.assertFalse(result.success)
        self.assertEqual(result.message, "platform refused")

    def test_cancel_releases_blocked_fake_navigation(self):
        gateway = FakeNavigationGateway(block=True)
        results = []
        thread = threading.Thread(
            target=lambda: results.append(
                gateway.navigate(
                    NavigationRequest("workflow", "step", "1"), lambda: False
                )
            )
        )
        thread.start()
        time.sleep(0.03)

        gateway.cancel_active()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].status, "canceled")


if __name__ == "__main__":
    unittest.main()
