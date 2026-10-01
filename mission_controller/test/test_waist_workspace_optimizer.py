import math
import unittest

from mission_runtime.waist_workspace_optimizer import WaistWorkspaceOptimizer


LIMITS = ([-180.0] * 7, [180.0] * 7)


def make_optimizer(**overrides):
    parameters = {
        "solve_ik": lambda arm, pose, seed: [pose[0]] + [0.0] * 6,
        "left_joint_limits_deg": LIMITS,
        "right_joint_limits_deg": LIMITS,
        "waist_limits_deg": ([-90.0] * 3, [0.0] * 3),
        "search_mode": "exhaustive",
        "candidate_step_deg": 10.0,
        "candidate_delta_deg": 20.0,
        "minimum_margin_deg": 0.0,
    }
    parameters.update(overrides)
    return WaistWorkspaceOptimizer(**parameters)


class TestWaistWorkspaceOptimizer(unittest.TestCase):

    def test_fixed_waist_validates_only_requested_pose(self):
        requested = [-40.001, -90.998, -68.305]
        seen = []
        def poses(waist):
            seen.append([math.degrees(v) for v in waist])
            return [0.0] * 6, [0.0] * 6
        result = make_optimizer(
            waist_limits_deg=([-89., -149., -89.], [4., 149., 4.]),
            candidate_delta_deg=0.0,
        ).optimize(
            nominal_waist_rad=[math.radians(v) for v in requested],
            left_seed_rad=[0.0]*7, right_seed_rad=[0.0]*7,
            target_pose_provider=poses,
        )
        self.assertEqual(len(seen), 1)
        self.assertEqual(result.candidate_count, 1)
        for actual, expected in zip(seen[0], requested):
            self.assertAlmostEqual(actual, expected)

    def test_fixed_waist_invalid_ik_does_not_search_alternatives(self):
        seen = []
        def poses(waist):
            seen.append(tuple(waist))
            return [0.0]*6, [0.0]*6
        with self.assertRaises(RuntimeError):
            make_optimizer(candidate_delta_deg=0.0, solve_ik=lambda *args: None).optimize(
                nominal_waist_rad=[math.radians(-10.)]*3,
                left_seed_rad=[0.0]*7, right_seed_rad=[0.0]*7,
                target_pose_provider=poses,
            )
        self.assertEqual(len(seen), 1)

    def test_auxiliary_check_rejects_right_only_candidate_before_selection(self):
        checked = []

        def check_left_join(waist_rad, solutions):
            checked.append(math.degrees(waist_rad[0]))
            self.assertIn("right", solutions)
            return math.degrees(waist_rad[0]) < -1.0

        result = make_optimizer(
            candidate_delta_deg=10.0,
            candidate_step_deg=10.0,
        ).optimize(
            nominal_waist_rad=[0.0] * 3,
            left_seed_rad=[0.0] * 7,
            right_seed_rad=[0.0] * 7,
            target_pose_provider=lambda _waist: ([0.0] * 6, [0.0] * 6),
            active_arms=("right",),
            auxiliary_candidate_validator=check_left_join,
        )

        self.assertLess(math.degrees(result.waist_angles_rad[0]), -1.0)
        self.assertTrue(any(abs(value) < 1e-6 for value in checked))

    def test_initial_seed_mode_reuses_zero_seed_for_final_pose(self):
        calls = []

        def solve_ik(arm, pose, seed):
            calls.append((arm, pose[0], list(seed)))
            return [pose[0]] + [0.0] * 6

        optimizer = WaistWorkspaceOptimizer(
            solve_ik=solve_ik,
            left_joint_limits_deg=([-180.0] * 7, [180.0] * 7),
            right_joint_limits_deg=([-180.0] * 7, [180.0] * 7),
            waist_limits_deg=([0.0] * 3, [1.0] * 3),
            search_mode="exhaustive",
            candidate_step_deg=1.0,
            candidate_delta_deg=0.0,
            minimum_margin_deg=0.0,
        )
        optimizer.optimize(
            nominal_waist_rad=[0.0] * 3,
            left_seed_rad=[0.0] * 7,
            right_seed_rad=[0.0] * 7,
            target_pose_provider=lambda _waist: ([10.0] * 6, [20.0] * 6),
            movel_endpoint_provider=lambda _waist: (
                [[11.0] * 6],
                [[21.0] * 6],
            ),
            endpoint_seed_mode="initial_seed",
        )

        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0], ("left", 10.0, [0.0] * 7))
        self.assertEqual(calls[1], ("right", 20.0, [0.0] * 7))
        self.assertEqual(calls[2], ("left", 11.0, [0.0] * 7))
        self.assertEqual(calls[3], ("right", 21.0, [0.0] * 7))

    def test_exhaustive_selects_largest_bilateral_worst_joint_margin(self):
        def target_provider(candidate_rad):
            candidate_deg = [math.degrees(value) for value in candidate_rad]
            return candidate_deg + [0.0] * 3, candidate_deg + [0.0] * 3

        result = make_optimizer().optimize(
            nominal_waist_rad=[0.0, 0.0, 0.0],
            left_seed_rad=[0.0] * 7,
            right_seed_rad=[0.0] * 7,
            target_pose_provider=target_provider,
        )

        self.assertEqual(result.search_mode, "exhaustive")
        self.assertEqual(result.waist_angles_rad, (0.0, 0.0, 0.0))
        self.assertEqual(result.candidate_count, 27)
        self.assertEqual(result.valid_count, 27)
        self.assertEqual(result.refined_candidate_count, 0)
        self.assertAlmostEqual(result.score_deg, 180.0)

    def test_rejects_best_candidate_below_required_margin(self):
        optimizer = make_optimizer(
            solve_ik=lambda arm, pose, seed: [9.0] + [0.0] * 6,
            left_joint_limits_deg=([-10.0] * 7, [10.0] * 7),
            right_joint_limits_deg=([-10.0] * 7, [10.0] * 7),
            candidate_delta_deg=0.0,
            minimum_margin_deg=2.0,
        )
        with self.assertRaisesRegex(RuntimeError, "margin is too small"):
            optimizer.optimize(
                nominal_waist_rad=[0.0, 0.0, 0.0],
                left_seed_rad=[0.0] * 7,
                right_seed_rad=[0.0] * 7,
                target_pose_provider=lambda candidate: ([0.0] * 6, [0.0] * 6),
            )

    def test_skips_candidate_when_either_arm_ik_fails(self):
        def solve_ik(arm, pose, seed):
            if pose[0] < -1e-6:
                return None
            return [0.0] * 7

        optimizer = make_optimizer(solve_ik=solve_ik)
        result = optimizer.optimize(
            nominal_waist_rad=[0.0, 0.0, 0.0],
            left_seed_rad=[0.0] * 7,
            right_seed_rad=[0.0] * 7,
            target_pose_provider=lambda candidate: (
                [math.degrees(candidate[0])] + [0.0] * 5,
                [math.degrees(candidate[0])] + [0.0] * 5,
            ),
        )

        self.assertEqual(result.valid_count, 9)
        self.assertEqual(result.waist_angles_rad[0], 0.0)

    def test_checks_movej_p_and_movel_endpoints_only_and_chains_seeds(self):
        calls = []

        def solve_ik(arm, pose, seed):
            calls.append((arm, pose[0], list(seed)))
            return [pose[0]] + [0.0] * 6

        optimizer = make_optimizer(
            solve_ik=solve_ik,
            candidate_delta_deg=0.0,
        )
        optimizer.optimize(
            nominal_waist_rad=[0.0, 0.0, 0.0],
            left_seed_rad=[0.0] * 7,
            right_seed_rad=[0.0] * 7,
            target_pose_provider=lambda candidate: ([0.0] * 6, [0.0] * 6),
            movel_endpoint_provider=lambda candidate: (
                ([1.0] * 6, [2.0] * 6),
                ([3.0] * 6,),
            ),
        )

        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[2], ("left", 1.0, [0.0] * 7))
        self.assertEqual(calls[3], ("left", 2.0, [1.0] + [0.0] * 6))
        self.assertEqual(calls[4], ("right", 3.0, [0.0] * 7))

    def test_skips_movel_endpoint_ik_when_provider_is_disabled(self):
        calls = []

        def solve_ik(arm, pose, seed):
            calls.append((arm, pose[0], list(seed)))
            return [pose[0]] + [0.0] * 6

        make_optimizer(
            solve_ik=solve_ik,
            candidate_delta_deg=0.0,
        ).optimize(
            nominal_waist_rad=[0.0, 0.0, 0.0],
            left_seed_rad=[0.0] * 7,
            right_seed_rad=[0.0] * 7,
            target_pose_provider=lambda candidate: ([1.0] * 6, [2.0] * 6),
            movel_endpoint_provider=None,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][:2], ("left", 1.0))
        self.assertEqual(calls[1][:2], ("right", 2.0))

    def test_right_only_search_ignores_left_target_and_left_endpoints(self):
        calls = []

        def solve_ik(arm, pose, seed):
            calls.append((arm, pose[0], list(seed)))
            if arm == "left":
                raise AssertionError("left arm must not be evaluated")
            return [pose[0]] + [0.0] * 6

        result = make_optimizer(
            solve_ik=solve_ik,
            candidate_delta_deg=0.0,
        ).optimize(
            nominal_waist_rad=[0.0, 0.0, 0.0],
            left_seed_rad=[0.0] * 7,
            right_seed_rad=[0.0] * 7,
            target_pose_provider=lambda candidate: ([99.0] * 6, [1.0] * 6),
            movel_endpoint_provider=lambda candidate: (
                ([98.0] * 6,),
                ([2.0] * 6, [3.0] * 6),
            ),
            active_arms=("right",),
        )

        self.assertEqual([call[:2] for call in calls], [
            ("right", 1.0),
            ("right", 2.0),
            ("right", 3.0),
        ])
        self.assertEqual(result.left_joint_deg, (0.0,) * 7)
        self.assertTrue(math.isinf(result.left_margin_deg))
        self.assertEqual(result.right_joint_deg[0], 1.0)
        self.assertAlmostEqual(result.score_deg, 177.0)

    def test_initial_solution_validator_rejects_positive_joint4(self):
        optimizer = make_optimizer(
            solve_ik=lambda arm, pose, seed: [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            candidate_delta_deg=0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "no MoveJ_P-IK-valid"):
            optimizer.optimize(
                nominal_waist_rad=[0.0, 0.0, 0.0],
                left_seed_rad=[0.0] * 7,
                right_seed_rad=[0.0] * 7,
                target_pose_provider=lambda candidate: ([0.0] * 6, [0.0] * 6),
                active_arms=("right",),
                initial_solution_validator=lambda arm, solution: solution[3] < 0.0,
            )

    def test_initial_solution_validator_does_not_filter_movel_endpoints(self):
        calls = []

        def solve_ik(arm, pose, seed):
            calls.append(pose[0])
            joint4 = -5.0 if pose[0] == 1.0 else 5.0
            return [0.0, 0.0, 0.0, joint4, 0.0, 0.0, 0.0]

        result = make_optimizer(
            solve_ik=solve_ik,
            candidate_delta_deg=0.0,
        ).optimize(
            nominal_waist_rad=[0.0, 0.0, 0.0],
            left_seed_rad=[0.0] * 7,
            right_seed_rad=[0.0] * 7,
            target_pose_provider=lambda candidate: ([0.0] * 6, [1.0] * 6),
            movel_endpoint_provider=lambda candidate: ((), ([2.0] * 6,)),
            active_arms=("right",),
            initial_solution_validator=lambda arm, solution: solution[3] < 0.0,
        )

        self.assertEqual(calls, [1.0, 2.0])
        self.assertEqual(result.right_joint_deg[3], -5.0)

    def test_endpoint_solution_validator_rejects_positive_joint4(self):
        def solve_ik(arm, pose, seed):
            joint4 = -5.0 if pose[0] == 1.0 else 5.0
            return [0.0, 0.0, 0.0, joint4, 0.0, 0.0, 0.0]

        with self.assertRaisesRegex(
            RuntimeError, "no MoveJ_P/MoveL-endpoint-IK-valid"
        ):
            make_optimizer(
                solve_ik=solve_ik,
                candidate_delta_deg=0.0,
            ).optimize(
                nominal_waist_rad=[0.0, 0.0, 0.0],
                left_seed_rad=[0.0] * 7,
                right_seed_rad=[0.0] * 7,
                target_pose_provider=lambda candidate: ([0.0] * 6, [1.0] * 6),
                movel_endpoint_provider=lambda candidate: ((), ([2.0] * 6,)),
                active_arms=("right",),
                endpoint_solution_validator=(
                    lambda arm, solution: solution[3] < 0.0
                ),
            )

    def test_rejects_invalid_active_arm_selection(self):
        with self.assertRaisesRegex(ValueError, "active_arms"):
            make_optimizer(candidate_delta_deg=0.0).optimize(
                nominal_waist_rad=[0.0, 0.0, 0.0],
                left_seed_rad=[0.0] * 7,
                right_seed_rad=[0.0] * 7,
                target_pose_provider=lambda candidate: ([0.0] * 6, [0.0] * 6),
                active_arms=(),
            )

    def test_coarse_to_fine_finds_optimum_between_coarse_grid_points(self):
        def target_provider(candidate_rad):
            candidate_deg = [math.degrees(value) for value in candidate_rad]
            return candidate_deg + [0.0] * 3, candidate_deg + [0.0] * 3

        def solve_ik(arm, pose, seed):
            return [abs(pose[0] - 2.0), abs(pose[1]), abs(pose[2])] + [0.0] * 4

        optimizer = make_optimizer(
            solve_ik=solve_ik,
            search_mode="coarse_to_fine",
            waist_limits_deg=([-5.0] * 3, [5.0] * 3),
            candidate_delta_deg=5.0,
            coarse_step_deg=5.0,
            coarse_top_k=1,
            candidate_diversity_deg=0.0,
            refine_radius_deg=3.0,
            refine_step_deg=1.0,
        )
        result = optimizer.optimize(
            nominal_waist_rad=[0.0, 0.0, 0.0],
            left_seed_rad=[0.0] * 7,
            right_seed_rad=[0.0] * 7,
            target_pose_provider=target_provider,
        )

        self.assertEqual(result.search_mode, "coarse_to_fine")
        self.assertAlmostEqual(math.degrees(result.waist_angles_rad[0]), 2.0)
        self.assertGreater(result.refined_candidate_count, 0)

    def test_sobol_refine_is_reproducible_with_fixed_seed(self):
        def target_provider(candidate_rad):
            candidate_deg = [math.degrees(value) for value in candidate_rad]
            return candidate_deg + [0.0] * 3, candidate_deg + [0.0] * 3

        parameters = {
            "search_mode": "sobol_refine",
            "waist_limits_deg": ([-10.0] * 3, [10.0] * 3),
            "candidate_delta_deg": 10.0,
            "coarse_top_k": 3,
            "candidate_diversity_deg": 2.0,
            "refine_radius_deg": 1.0,
            "refine_step_deg": 1.0,
            "sobol_sample_count": 32,
            "sobol_seed": 7,
        }
        results = []
        for _ in range(2):
            results.append(
                make_optimizer(**parameters).optimize(
                    nominal_waist_rad=[0.0, 0.0, 0.0],
                    left_seed_rad=[0.0] * 7,
                    right_seed_rad=[0.0] * 7,
                    target_pose_provider=target_provider,
                )
            )

        self.assertEqual(results[0].waist_angles_rad, results[1].waist_angles_rad)
        self.assertEqual(results[0].candidate_count, results[1].candidate_count)
        self.assertGreater(results[0].refined_candidate_count, 0)

    def test_reports_search_mode_and_phase_progress(self):
        progress = []
        make_optimizer(
            search_mode="coarse_to_fine",
            candidate_delta_deg=0.0,
            coarse_step_deg=5.0,
            coarse_top_k=1,
            refine_radius_deg=0.0,
        ).optimize(
            nominal_waist_rad=[0.0, 0.0, 0.0],
            left_seed_rad=[0.0] * 7,
            right_seed_rad=[0.0] * 7,
            target_pose_provider=lambda candidate: ([0.0] * 6, [0.0] * 6),
            progress_callback=progress.append,
        )

        self.assertTrue(any("mode=coarse_to_fine" in item for item in progress))
        self.assertTrue(any("coarse MoveJ_P IK" in item for item in progress))
        self.assertTrue(any("MoveL intermediate IK=disabled" in item for item in progress))

    def test_rejects_unknown_search_mode(self):
        with self.assertRaisesRegex(ValueError, "search_mode"):
            make_optimizer(search_mode="gradient")


if __name__ == "__main__":
    unittest.main()
