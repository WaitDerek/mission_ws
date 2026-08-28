import unittest

from mission_runtime.taskflow.mapping import (
    DIRECT_GRASP_ACTION,
    DRAG_GRASP_ACTION,
    grasp_action_for_operation_point,
    normalize_box_type,
    operation_point_for_stack,
)


class TestDepalletizingMapping(unittest.TestCase):
    def test_observation_points_map_to_two_operation_points(self):
        expected = {
            "1": ("6", "5"),
            "2": ("8", "7"),
            "3": ("10", "9"),
            "4": ("12", "11"),
        }
        for observation_point, operation_points in expected.items():
            for stack_member, operation_point in enumerate(operation_points):
                with self.subTest(
                    observation_point=observation_point,
                    stack_member=stack_member,
                ):
                    self.assertEqual(
                        operation_point_for_stack(observation_point, stack_member),
                        operation_point,
                    )

    def test_odd_points_use_drag_and_even_points_use_direct_grasp(self):
        for point in ("5", "7", "9", "11"):
            self.assertEqual(grasp_action_for_operation_point(point), DRAG_GRASP_ACTION)
        for point in ("6", "8", "10", "12"):
            self.assertEqual(
                grasp_action_for_operation_point(point), DIRECT_GRASP_ACTION
            )

    def test_camera_visible_left_is_even_and_visible_right_is_odd(self):
        for observation_point in ("1", "2", "3", "4"):
            left = int(operation_point_for_stack(observation_point, 0))
            right = int(operation_point_for_stack(observation_point, 1))
            self.assertEqual(left % 2, 0)
            self.assertEqual(right % 2, 1)

    def test_reserved_and_unknown_points_are_rejected(self):
        for point in ("13", "14", "15", "16", "99"):
            with self.subTest(point=point), self.assertRaises(ValueError):
                grasp_action_for_operation_point(point)

    def test_box_size_aliases_normalize_to_existing_action_contract(self):
        self.assertEqual(normalize_box_type("big"), "bigbox")
        self.assertEqual(normalize_box_type("BIG_BOX"), "bigbox")
        self.assertEqual(normalize_box_type("small"), "smallbox")
        self.assertEqual(normalize_box_type("small-box"), "smallbox")


if __name__ == "__main__":
    unittest.main()
