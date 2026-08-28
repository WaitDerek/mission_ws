import unittest

from mission_runtime.taskflow.identifiers import (
    IdentifierFactory,
    build_request_id,
    build_step_id,
    parse_request_id,
    parse_step_id,
    sanitize_request_id,
)


class TestDepalletizingIdentifiers(unittest.TestCase):
    def test_step_and_request_ids_round_trip_delimiters(self):
        step_id = build_step_id("GRASP|TF", "5/left", 3, 7)
        step = parse_step_id(step_id)
        self.assertEqual(step.stage, "GRASP|TF")
        self.assertEqual(step.point_id, "5/left")
        self.assertEqual(step.order_index, 3)
        self.assertEqual(step.sequence, 7)

        request_id = build_request_id("workflow|1", "secret/token", step_id)
        request = parse_request_id(request_id)
        self.assertEqual(request.workflow_id, "workflow|1")
        self.assertEqual(request.lease_token, "secret/token")
        self.assertEqual(request.step_id, step_id)

    def test_sanitizer_removes_token_but_retains_trace(self):
        step_id = build_step_id("PLACE", "16", 2, 9)
        request_id = build_request_id("workflow-1", "raw-secret", step_id)

        sanitized = sanitize_request_id(request_id)

        self.assertNotIn("raw-secret", sanitized)
        self.assertIn("workflow-1", sanitized)
        self.assertIn("%7Cstage%3DPLACE", sanitized)
        self.assertIn("<redacted>", sanitized)

    def test_factory_uses_monotonic_sequence_to_avoid_collisions(self):
        factory = IdentifierFactory("workflow", "token")
        first_step, first_request = factory.request_id("PLACE", "16", 0)
        second_step, second_request = factory.request_id("PLACE", "16", 0)

        self.assertNotEqual(first_step, second_step)
        self.assertNotEqual(first_request, second_request)
        self.assertEqual(parse_step_id(first_step).sequence, 0)
        self.assertEqual(parse_step_id(second_step).sequence, 1)


if __name__ == "__main__":
    unittest.main()
