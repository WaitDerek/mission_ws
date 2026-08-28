import unittest

from mission_runtime.taskflow.identifiers import build_request_id, build_step_id
from mission_runtime.taskflow.lease import WorkflowLeaseManager


class TestDepalletizingLease(unittest.TestCase):
    def setUp(self):
        self.manager = WorkflowLeaseManager()
        self.acquired = self.manager.acquire("workflow-1")
        self.step_id = build_step_id("GRASP", "5", 0, 0)

    def _workflow_request(self, token=None):
        return build_request_id(
            "workflow-1",
            token if token is not None else self.acquired.lease_token,
            self.step_id,
        )

    def test_active_lease_blocks_legacy_but_allows_matching_child(self):
        legacy = self.manager.reserve_goal("box_place", "external-request")
        child = self.manager.reserve_goal("grasp_box_tf", self._workflow_request())

        self.assertFalse(legacy.accepted)
        self.assertTrue(child.accepted)
        self.assertTrue(child.workflow_owned)

    def test_wrong_or_stale_token_is_rejected_without_token_echo(self):
        decision = self.manager.reserve_goal(
            "grasp_box_tf", self._workflow_request("wrong-secret")
        )

        self.assertFalse(decision.accepted)
        self.assertNotIn("wrong-secret", decision.message)
        self.assertNotIn("wrong-secret", decision.sanitized_request_id)

    def test_malformed_request_cannot_leak_the_active_token(self):
        decision = self.manager.reserve_goal(
            "grasp_box_tf", f"malformed|{self.acquired.lease_token}"
        )

        self.assertFalse(decision.accepted)
        self.assertNotIn(self.acquired.lease_token, decision.sanitized_request_id)

    def test_child_release_keeps_workflow_lease_until_explicit_release(self):
        self.assertTrue(
            self.manager.reserve_goal("grasp_box_tf", self._workflow_request()).accepted
        )
        self.assertFalse(
            self.manager.release("workflow-1", self.acquired.lease_token).success
        )

        self.manager.release_goal()
        released = self.manager.release("workflow-1", self.acquired.lease_token)

        self.assertTrue(released.success)
        self.assertTrue(
            self.manager.reserve_goal("box_place", "external-request").accepted
        )

    def test_acquire_is_rejected_while_legacy_goal_is_active(self):
        other = WorkflowLeaseManager()
        self.assertTrue(other.reserve_goal("box_place", "legacy").accepted)

        acquired = other.acquire("workflow")

        self.assertFalse(acquired.success)


if __name__ == "__main__":
    unittest.main()
