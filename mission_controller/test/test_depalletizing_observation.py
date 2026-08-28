import unittest
from types import SimpleNamespace

from mission_runtime.taskflow.observation import (
    FrontStackPoseValidation,
    ObservationValidationError,
    adapt_global_observation_result,
)


def _camera_pose(x, z, frame_id="camera_optical_frame"):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        pose=SimpleNamespace(position=SimpleNamespace(x=x, z=z)),
    )


def _result(**overrides):
    values = {
        "success": True,
        "plan_valid": True,
        "message": "ok",
        "order_stack_ids": ["front_right", "front_left"],
        "order_stack_indices": [1, 0],
        "order_layer_numbers": [1, 2],
        "order_columns": [1, 0],
        "order_box_sizes": ["big", "smallbox"],
        "stack_sides": [0, 0],
        "stack_columns": [0, 1],
        "top_box_camera_poses": [
            _camera_pose(-0.30, 0.90),
            _camera_pose(0.30, 0.92),
        ],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDepalletizingObservation(unittest.TestCase):
    def test_valid_result_preserves_ids_layers_and_normalizes_sizes(self):
        adapted = adapt_global_observation_result("1", _result())

        self.assertTrue(adapted.success)
        self.assertEqual(adapted.plan.point_id, "1")
        self.assertEqual(
            [task.stack_id for task in adapted.plan.tasks],
            ["front_right", "front_left"],
        )
        self.assertEqual([task.layer for task in adapted.plan.tasks], [1, 2])
        self.assertEqual(
            [task.box_type for task in adapted.plan.tasks],
            ["bigbox", "smallbox"],
        )

    def test_unsuccessful_or_invalid_plan_is_non_actionable(self):
        failed = adapt_global_observation_result(
            "1", _result(success=False, message="camera failed")
        )
        empty = adapt_global_observation_result("1", _result(plan_valid=False))

        self.assertFalse(failed.success)
        self.assertEqual(failed.message, "camera failed")
        self.assertFalse(empty.success)

    def test_parallel_array_length_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ObservationValidationError, "equal lengths"):
            adapt_global_observation_result("1", _result(order_box_sizes=["big"]))

    def test_stack_index_and_column_disagreement_fails_closed(self):
        with self.assertRaisesRegex(ObservationValidationError, "disagrees"):
            adapt_global_observation_result("1", _result(order_columns=[0, 0]))

    def test_rear_or_mixed_row_is_rejected(self):
        cases = (
            ({"stack_sides": [0, 1]}, "non-front"),
            (
                {
                    "top_box_camera_poses": [
                        _camera_pose(-0.30, 0.80),
                        _camera_pose(0.30, 1.30),
                    ]
                },
                "one front row",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ObservationValidationError, message
            ):
                adapt_global_observation_result("1", _result(**overrides))

    def test_front_pair_must_be_distinct_and_left_to_right(self):
        with self.assertRaisesRegex(ObservationValidationError, "distinct"):
            adapt_global_observation_result(
                "1",
                _result(
                    top_box_camera_poses=[
                        _camera_pose(-0.05, 0.90),
                        _camera_pose(0.05, 0.90),
                    ]
                ),
            )

    def test_optional_absolute_depth_gate_rejects_rear_pair(self):
        with self.assertRaisesRegex(ObservationValidationError, "front-row depth"):
            adapt_global_observation_result(
                "1",
                _result(
                    top_box_camera_poses=[
                        _camera_pose(-0.30, 1.50),
                        _camera_pose(0.30, 1.52),
                    ]
                ),
                front_stack_validation=FrontStackPoseValidation(
                    max_camera_depth_m=1.20
                ),
            )

    def test_pose_validation_can_be_disabled_for_compatibility(self):
        adapted = adapt_global_observation_result(
            "1",
            _result(top_box_camera_poses=[]),
            front_stack_validation=FrontStackPoseValidation(enabled=False),
        )

        self.assertTrue(adapted.success)

    def test_invalid_layer_size_and_empty_id_fail_closed(self):
        cases = (
            ({"order_layer_numbers": [0, 2]}, "layer must be in 1..4"),
            ({"order_box_sizes": ["medium", "small"]}, "unsupported box size"),
            ({"order_stack_ids": ["", "front_left"]}, "empty stack_id"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ObservationValidationError, message
            ):
                adapt_global_observation_result("1", _result(**overrides))


if __name__ == "__main__":
    unittest.main()
