"""Deterministic waist search using dual-arm endpoint IK clearance."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class WaistWorkspaceResult:
    """Selected waist target and predicted 7-DOF arm solutions, in degrees."""

    search_mode: str
    waist_angles_rad: tuple[float, float, float]
    left_joint_deg: tuple[float, ...]
    right_joint_deg: tuple[float, ...]
    left_margin_deg: float
    right_margin_deg: float
    candidate_count: int
    valid_count: int
    initial_candidate_count: int
    refined_candidate_count: int
    priority_joint_margin_deg: float | None = None

    @property
    def score_deg(self) -> float:
        """Worst evaluated-arm joint clearance used to rank candidates."""
        return min(self.left_margin_deg, self.right_margin_deg)


@dataclass(frozen=True)
class _EvaluatedCandidate:
    rank: tuple[float, ...]
    waist_deg: tuple[float, float, float]
    left_joint_deg: tuple[float, ...]
    right_joint_deg: tuple[float, ...]
    left_margin_deg: float
    right_margin_deg: float
    priority_joint_margin_deg: float | None = None


class WaistWorkspaceOptimizer:
    """Pick the endpoint-IK-valid waist target with the best joint margin."""

    SEARCH_MODES = ("exhaustive", "coarse_to_fine", "sobol_refine")

    def __init__(
        self,
        *,
        solve_ik: Callable[
            [str, Sequence[float], Sequence[float]], Sequence[float] | None
        ],
        left_joint_limits_deg: tuple[Sequence[float], Sequence[float]],
        right_joint_limits_deg: tuple[Sequence[float], Sequence[float]],
        waist_limits_deg: tuple[Sequence[float], Sequence[float]],
        search_mode: str = "coarse_to_fine",
        candidate_step_deg: float = 1.0,
        candidate_delta_deg: float = 20.0,
        minimum_margin_deg: float = 8.0,
        coarse_step_deg: float = 5.0,
        coarse_top_k: int = 8,
        candidate_diversity_deg: float = 5.0,
        refine_radius_deg: float = 3.0,
        refine_step_deg: float = 1.0,
        sobol_sample_count: int = 1024,
        sobol_seed: int = 0,
        ik_pose_residual=None,
        robustness_delta_deg: float = 0.0,
    ) -> None:
        self._raw_solve_ik = solve_ik
        self._ik_pose_residual = ik_pose_residual
        self._robustness_delta_deg = float(robustness_delta_deg)
        if not math.isfinite(self._robustness_delta_deg) or self._robustness_delta_deg < 0:
            raise ValueError("robustness_delta_deg must be finite and non-negative")
        self._left_limits = self._validated_limits(
            left_joint_limits_deg, 7, "left arm"
        )
        self._right_limits = self._validated_limits(
            right_joint_limits_deg, 7, "right arm"
        )
        self._waist_limits = self._validated_limits(
            waist_limits_deg, 3, "waist"
        )
        self._search_mode = str(search_mode).strip().lower()
        self._candidate_step_deg = float(candidate_step_deg)
        self._candidate_delta_deg = float(candidate_delta_deg)
        self._minimum_margin_deg = float(minimum_margin_deg)
        self._coarse_step_deg = float(coarse_step_deg)
        self._coarse_top_k = int(coarse_top_k)
        self._candidate_diversity_deg = float(candidate_diversity_deg)
        self._refine_radius_deg = float(refine_radius_deg)
        self._refine_step_deg = float(refine_step_deg)
        self._sobol_sample_count = int(sobol_sample_count)
        self._sobol_seed = int(sobol_seed)
        self._validate_configuration()

    def _solve_ik(self, arm, pose, seed):
        solution = self._raw_solve_ik(arm, pose, seed)
        if solution is None:
            return None
        if len(solution) != 7 or not all(math.isfinite(v) for v in solution):
            return None
        if self._ik_pose_residual is not None:
            position, rotation = self._ik_pose_residual(arm, pose, solution)
            if (not math.isfinite(position) or not math.isfinite(rotation)
                    or position > 0.0005 or rotation > math.radians(0.1)):
                return None
        return solution

    def _validate_configuration(self) -> None:
        if self._search_mode not in self.SEARCH_MODES:
            raise ValueError(
                "search_mode must be one of " + ", ".join(self.SEARCH_MODES)
            )
        for name, value in (
            ("candidate_step_deg", self._candidate_step_deg),
            ("coarse_step_deg", self._coarse_step_deg),
            ("refine_step_deg", self._refine_step_deg),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("candidate_delta_deg", self._candidate_delta_deg),
            ("minimum_margin_deg", self._minimum_margin_deg),
            ("candidate_diversity_deg", self._candidate_diversity_deg),
            ("refine_radius_deg", self._refine_radius_deg),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self._coarse_top_k <= 0:
            raise ValueError("coarse_top_k must be positive")
        if self._sobol_sample_count <= 0:
            raise ValueError("sobol_sample_count must be positive")
        if self._sobol_seed < 0:
            raise ValueError("sobol_seed must be non-negative")

    @staticmethod
    def _validated_limits(limits, dof: int, label: str):
        lower = tuple(float(value) for value in limits[0])
        upper = tuple(float(value) for value in limits[1])
        if len(lower) != dof or len(upper) != dof:
            raise ValueError(f"{label} limits must contain {dof} values")
        if not all(
            math.isfinite(lo) and math.isfinite(hi) and lo < hi
            for lo, hi in zip(lower, upper)
        ):
            raise ValueError(f"{label} limits are invalid")
        return lower, upper

    @staticmethod
    def _axis_values(
        start_deg: float,
        stop_deg: float,
        step_deg: float,
        *required_values: float,
    ) -> tuple[float, ...]:
        if start_deg > stop_deg:
            return ()
        count = int(math.floor((stop_deg - start_deg) / step_deg + 1e-9))
        values = [start_deg + index * step_deg for index in range(count + 1)]
        values.append(stop_deg)
        values.extend(
            value for value in required_values if start_deg <= value <= stop_deg
        )
        return tuple(sorted({round(float(value), 9) for value in values}))

    @staticmethod
    def _minimum_joint_margin(joints_deg, limits) -> float:
        lower, upper = limits
        return min(
            min(float(joint) - lo, hi - float(joint))
            for joint, lo, hi in zip(joints_deg, lower, upper)
        )

    @staticmethod
    def _candidate_key(candidate_deg: Sequence[float]) -> tuple[float, float, float]:
        if len(candidate_deg) != 3:
            raise ValueError("waist candidate must contain three joints")
        return tuple(round(float(value), 9) for value in candidate_deg)

    def _search_bounds(self, nominal_deg):
        waist_lower, waist_upper = self._waist_limits
        lower = tuple(
            max(waist_lower[index], nominal_deg[index] - self._candidate_delta_deg)
            for index in range(3)
        )
        upper = tuple(
            min(waist_upper[index], nominal_deg[index] + self._candidate_delta_deg)
            for index in range(3)
        )
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise RuntimeError("nominal waist target is outside configured limits")
        return lower, upper

    def _grid_candidates(
        self,
        center_deg,
        lower_deg,
        upper_deg,
        step_deg: float,
    ) -> list[tuple[float, float, float]]:
        axes = tuple(
            self._axis_values(
                lower_deg[index], upper_deg[index], step_deg, center_deg[index]
            )
            for index in range(3)
        )
        return [self._candidate_key(candidate) for candidate in itertools.product(*axes)]

    def _refine_candidates(self, centers, search_lower, search_upper):
        candidates = set()
        for center in centers:
            local_lower = tuple(
                max(search_lower[index], center[index] - self._refine_radius_deg)
                for index in range(3)
            )
            local_upper = tuple(
                min(search_upper[index], center[index] + self._refine_radius_deg)
                for index in range(3)
            )
            candidates.update(
                self._grid_candidates(
                    center,
                    local_lower,
                    local_upper,
                    self._refine_step_deg,
                )
            )
        return sorted(candidates)

    def _sobol_candidates(self, nominal_deg, lower_deg, upper_deg):
        try:
            from scipy.stats import qmc
        except ImportError as exc:
            raise RuntimeError(
                "sobol_refine requires scipy.stats.qmc.Sobol"
            ) from exc

        exponent = int(math.ceil(math.log2(self._sobol_sample_count)))
        samples = qmc.Sobol(
            d=3, scramble=True, seed=self._sobol_seed
        ).random_base2(exponent)
        candidates = {self._candidate_key(nominal_deg)}
        for sample in samples[: self._sobol_sample_count]:
            candidates.add(
                self._candidate_key(
                    tuple(
                        lower_deg[index]
                        + float(sample[index])
                        * (upper_deg[index] - lower_deg[index])
                        for index in range(3)
                    )
                )
            )
        for corner in itertools.product((0, 1), repeat=3):
            candidates.add(
                self._candidate_key(
                    tuple(
                        upper_deg[index] if corner[index] else lower_deg[index]
                        for index in range(3)
                    )
                )
            )
        return sorted(candidates)

    def _evaluate_candidate(
        self,
        candidate_deg,
        nominal_deg,
        left_seed_deg,
        right_seed_deg,
        target_pose_provider,
        movel_endpoint_provider,
        endpoint_seed_mode,
        active_arms,
        initial_solution_validator,
        endpoint_solution_validator,
        priority_joint_arm,
        priority_joint_index,
        priority_joint_upper_bound_deg,
        auxiliary_candidate_validator,
    ) -> _EvaluatedCandidate | None:
        candidate_rad = tuple(math.radians(value) for value in candidate_deg)
        left_pose, right_pose = target_pose_provider(candidate_rad)
        arm_inputs = {
            "left": (left_pose, left_seed_deg, self._left_limits),
            "right": (right_pose, right_seed_deg, self._right_limits),
        }
        solutions = {
            "left": tuple(float(value) for value in left_seed_deg),
            "right": tuple(float(value) for value in right_seed_deg),
        }
        margins = {"left": math.inf, "right": math.inf}
        priority_joint_margin = math.inf
        for arm in active_arms:
            pose, seed, limits = arm_inputs[arm]
            solution = self._solve_ik(arm, pose, seed)
            if solution is None:
                return None
            solution = tuple(float(value) for value in solution)
            if len(solution) != 7:
                raise RuntimeError("SDK IK returned a non-7DOF solution")
            if (
                initial_solution_validator is not None
                and not initial_solution_validator(arm, solution)
            ):
                return None
            solutions[arm] = solution
            margins[arm] = self._minimum_joint_margin(solution, limits)
            if arm == priority_joint_arm:
                lower = limits[0][priority_joint_index]
                upper = (
                    limits[1][priority_joint_index]
                    if priority_joint_upper_bound_deg is None
                    else priority_joint_upper_bound_deg
                )
                priority_joint_margin = min(
                    priority_joint_margin,
                    solution[priority_joint_index] - lower,
                    upper - solution[priority_joint_index],
                )

        if movel_endpoint_provider is not None:
            left_endpoints, right_endpoints = movel_endpoint_provider(candidate_rad)
            endpoints_by_arm = {
                "left": (left_endpoints, self._left_limits),
                "right": (right_endpoints, self._right_limits),
            }
            for arm in active_arms:
                endpoints, limits = endpoints_by_arm[arm]
                configured_seed = arm_inputs[arm][1]
                previous = (
                    solutions[arm]
                    if endpoint_seed_mode == "previous_solution"
                    else configured_seed
                )
                for endpoint in endpoints:
                    solution = self._solve_ik(arm, endpoint, previous)
                    if solution is None:
                        return None
                    solution = tuple(float(value) for value in solution)
                    if len(solution) != 7:
                        raise RuntimeError("SDK IK returned a non-7DOF solution")
                    if (
                        endpoint_solution_validator is not None
                        and not endpoint_solution_validator(arm, solution)
                    ):
                        return None
                    margins[arm] = min(
                        margins[arm],
                        self._minimum_joint_margin(solution, limits),
                    )
                    if arm == priority_joint_arm:
                        lower = limits[0][priority_joint_index]
                        upper = (
                            limits[1][priority_joint_index]
                            if priority_joint_upper_bound_deg is None
                            else priority_joint_upper_bound_deg
                        )
                        priority_joint_margin = min(
                            priority_joint_margin,
                            solution[priority_joint_index] - lower,
                            upper - solution[priority_joint_index],
                        )
                    if endpoint_seed_mode == "previous_solution":
                        previous = solution

        scored_arms = set(active_arms)
        if auxiliary_candidate_validator is not None:
            auxiliary = auxiliary_candidate_validator(candidate_rad, solutions)
            if not auxiliary:
                return None
            # A mapping contributes deferred-arm solutions to the same
            # worst-arm clearance score; bool callbacks remain supported.
            if isinstance(auxiliary, dict):
                for arm, values in auxiliary.items():
                    if arm not in arm_inputs:
                        raise ValueError(f"unknown auxiliary arm: {arm}")
                    values = tuple(float(v) for v in values)
                    if len(values) != 7 or not all(math.isfinite(v) for v in values):
                        return None
                    margin = self._minimum_joint_margin(values, arm_inputs[arm][2])
                    if margin < self._minimum_margin_deg:
                        return None
                    solutions[arm] = values
                    margins[arm] = min(margins[arm], margin)
                    scored_arms.add(arm)

        score = min(margins[arm] for arm in scored_arms)
        waist_offset = sum(
            abs(value - nominal)
            for value, nominal in zip(candidate_deg, nominal_deg)
        )
        arm_motion = sum(
            abs(value - seed_value)
            for arm, seed_values in (
                ("left", left_seed_deg),
                ("right", right_seed_deg),
            )
            if arm in active_arms
            for value, seed_value in zip(solutions[arm], seed_values)
        )
        if priority_joint_arm is None:
            rank = (score, -waist_offset, -arm_motion)
            priority_joint_margin_result = None
        else:
            # Rank the right-arm negative-Joint4 branch first, while retaining
            # the configured all-joint clearance as a hard feasibility priority.
            safe_candidate = int(
                score >= self._minimum_margin_deg
                and priority_joint_margin >= self._minimum_margin_deg
            )
            rank = (
                safe_candidate,
                priority_joint_margin,
                score,
                -waist_offset,
                -arm_motion,
            )
            priority_joint_margin_result = priority_joint_margin
        return _EvaluatedCandidate(
            rank=rank,
            waist_deg=self._candidate_key(candidate_deg),
            left_joint_deg=solutions["left"],
            right_joint_deg=solutions["right"],
            left_margin_deg=margins["left"],
            right_margin_deg=margins["right"],
            priority_joint_margin_deg=priority_joint_margin_result,
        )

    def _evaluate_candidates(
        self,
        candidates: Iterable[Sequence[float]],
        *,
        phase: str,
        nominal_deg,
        left_seed_deg,
        right_seed_deg,
        target_pose_provider,
        movel_endpoint_provider,
        endpoint_ik_label,
        endpoint_seed_mode,
        active_arms,
        initial_solution_validator,
        endpoint_solution_validator,
        priority_joint_arm,
        priority_joint_index,
        priority_joint_upper_bound_deg,
        auxiliary_candidate_validator,
        seen,
        progress_callback,
    ) -> tuple[list[_EvaluatedCandidate], int]:
        pending = []
        for candidate in candidates:
            key = self._candidate_key(candidate)
            if key not in seen:
                seen.add(key)
                pending.append(key)
        total = len(pending)
        valid = []
        best = None
        progress_interval = max(1, total // 100) if total else 1
        ik_scope = (
            f"MoveJ_P/{endpoint_ik_label} IK"
            if movel_endpoint_provider is not None
            else "MoveJ_P IK"
        )
        if progress_callback is not None:
            progress_callback(
                f"{phase} {ik_scope} started: checked=0/{total}, valid=0"
            )
        for checked, candidate in enumerate(pending, start=1):
            try:
                evaluated = self._evaluate_candidate(
                    candidate,
                    nominal_deg,
                    left_seed_deg,
                    right_seed_deg,
                    target_pose_provider,
                    movel_endpoint_provider,
                    endpoint_seed_mode,
                    active_arms,
                    initial_solution_validator,
                    endpoint_solution_validator,
                    priority_joint_arm,
                    priority_joint_index,
                    priority_joint_upper_bound_deg,
                    auxiliary_candidate_validator,
                )
                if evaluated is not None:
                    valid.append(evaluated)
                    if best is None or evaluated.rank > best.rank:
                        best = evaluated
            finally:
                if progress_callback is not None and (
                    checked == 1
                    or checked == total
                    or checked % progress_interval == 0
                ):
                    best_detail = "best=none"
                    if best is not None:
                        waist = ",".join(f"{value:.3f}" for value in best.waist_deg)
                        best_detail = (
                            f"best_waist_deg=[{waist}], "
                            f"best_joint{priority_joint_index + 1}_margin="
                            f"{best.priority_joint_margin_deg:.3f}deg"
                            if priority_joint_arm is not None
                            else f"best_waist_deg=[{waist}], "
                            f"best_margin={best.rank[0]:.3f}deg"
                        )
                    progress_callback(
                        f"{phase} {ik_scope} progress: "
                        f"checked={checked}/{total}, valid={len(valid)}, "
                        f"{best_detail}"
                    )
        return valid, total

    def _select_diverse_candidates(self, candidates):
        selected = []
        for candidate in sorted(candidates, key=lambda item: item.rank, reverse=True):
            if all(
                math.dist(candidate.waist_deg, existing.waist_deg)
                >= self._candidate_diversity_deg
                for existing in selected
            ):
                selected.append(candidate)
                if len(selected) >= self._coarse_top_k:
                    break
        return selected

    def optimize(
        self,
        *,
        nominal_waist_rad: Sequence[float],
        left_seed_rad: Sequence[float],
        right_seed_rad: Sequence[float],
        target_pose_provider: Callable[
            [Sequence[float]], tuple[Sequence[float], Sequence[float]]
        ],
        movel_endpoint_provider: Callable[
            [Sequence[float]],
            tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]],
        ]
        | None = None,
        endpoint_ik_label: str = "MoveL-endpoint",
        endpoint_seed_mode: str = "previous_solution",
        active_arms: Sequence[str] = ("left", "right"),
        initial_solution_validator: Callable[
            [str, Sequence[float]], bool
        ]
        | None = None,
        endpoint_solution_validator: Callable[
            [str, Sequence[float]], bool
        ]
        | None = None,
        priority_joint_arm: str | None = None,
        priority_joint_index: int | None = None,
        priority_joint_upper_bound_deg: float | None = None,
        auxiliary_candidate_validator: Callable[
            [Sequence[float], dict[str, tuple[float, ...]]], bool | dict[str, Sequence[float]]
        ] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> WaistWorkspaceResult:
        """Search candidates using initial IK and one optional final endpoint."""
        if len(nominal_waist_rad) != 3:
            raise ValueError("nominal waist target must contain three joints")
        if len(left_seed_rad) != 7 or len(right_seed_rad) != 7:
            raise ValueError("arm IK seeds must contain seven joints")
        active_arms = tuple(str(arm).strip().lower() for arm in active_arms)
        if not active_arms or len(set(active_arms)) != len(active_arms):
            raise ValueError("active_arms must contain unique arm names")
        if any(arm not in ("left", "right") for arm in active_arms):
            raise ValueError("active_arms may contain only 'left' and 'right'")
        endpoint_ik_label = str(endpoint_ik_label).strip()
        if not endpoint_ik_label:
            raise ValueError("endpoint_ik_label must not be empty")
        endpoint_seed_mode = str(endpoint_seed_mode).strip().lower()
        if endpoint_seed_mode not in ("previous_solution", "initial_seed"):
            raise ValueError(
                "endpoint_seed_mode must be 'previous_solution' or 'initial_seed'"
            )
        if priority_joint_arm is None:
            if priority_joint_index is not None or priority_joint_upper_bound_deg is not None:
                raise ValueError("priority joint settings require priority_joint_arm")
        else:
            priority_joint_arm = str(priority_joint_arm).strip().lower()
            if priority_joint_arm not in active_arms:
                raise ValueError("priority_joint_arm must be one of active_arms")
            if priority_joint_index is None or not 0 <= int(priority_joint_index) < 7:
                raise ValueError("priority_joint_index must be in [0, 6]")
            priority_joint_index = int(priority_joint_index)
            if priority_joint_upper_bound_deg is not None:
                lower = (
                    self._left_limits[0][priority_joint_index]
                    if priority_joint_arm == "left"
                    else self._right_limits[0][priority_joint_index]
                )
                upper = (
                    self._left_limits[1][priority_joint_index]
                    if priority_joint_arm == "left"
                    else self._right_limits[1][priority_joint_index]
                )
                priority_joint_upper_bound_deg = float(priority_joint_upper_bound_deg)
                if not lower < priority_joint_upper_bound_deg <= upper:
                    raise ValueError("priority joint upper bound is outside joint limits")

        nominal_deg = tuple(math.degrees(float(value)) for value in nominal_waist_rad)
        left_seed_deg = tuple(math.degrees(float(value)) for value in left_seed_rad)
        right_seed_deg = tuple(math.degrees(float(value)) for value in right_seed_rad)
        search_lower, search_upper = self._search_bounds(nominal_deg)

        if self._search_mode == "exhaustive":
            initial_candidates = self._grid_candidates(
                nominal_deg,
                search_lower,
                search_upper,
                self._candidate_step_deg,
            )
            initial_phase = "exhaustive"
        elif self._search_mode == "coarse_to_fine":
            initial_candidates = self._grid_candidates(
                nominal_deg,
                search_lower,
                search_upper,
                self._coarse_step_deg,
            )
            initial_phase = "coarse"
        else:
            initial_candidates = self._sobol_candidates(
                nominal_deg,
                search_lower,
                search_upper,
            )
            initial_phase = "sobol"

        if progress_callback is not None:
            endpoint_status = (
                "enabled" if movel_endpoint_provider is not None else "disabled"
            )
            progress_callback(
                f"waist search mode={self._search_mode}; "
                f"evaluated arms={','.join(active_arms)}; "
                "MoveJ_P target IK=enabled; "
                f"{endpoint_ik_label} IK={endpoint_status}; "
                f"endpoint IK seed mode={endpoint_seed_mode}; "
                "MoveL intermediate IK=disabled; "
                "MoveJ_P solution filter="
                f"{'enabled' if initial_solution_validator is not None else 'disabled'}; "
                f"{endpoint_ik_label} solution filter="
                f"{'enabled' if endpoint_solution_validator is not None else 'disabled'}; "
                "auxiliary candidate check="
                f"{'enabled' if auxiliary_candidate_validator is not None else 'disabled'}; "
                "search objective="
                + (
                    f"{priority_joint_arm} joint {priority_joint_index + 1} margin"
                    + (
                        f" within [{self._left_limits[0][priority_joint_index]:.3f},"
                        f"{priority_joint_upper_bound_deg:.3f}]deg"
                        if priority_joint_arm == "left"
                        and priority_joint_upper_bound_deg is not None
                        else (
                            f" within [{self._right_limits[0][priority_joint_index]:.3f},"
                            f"{priority_joint_upper_bound_deg:.3f}]deg"
                            if priority_joint_upper_bound_deg is not None
                            else ""
                        )
                    )
                    if priority_joint_arm is not None
                    else "minimum evaluated joint margin"
                )
            )
        seen = set()
        initial_valid, initial_count = self._evaluate_candidates(
            initial_candidates,
            phase=initial_phase,
            nominal_deg=nominal_deg,
            left_seed_deg=left_seed_deg,
            right_seed_deg=right_seed_deg,
            target_pose_provider=target_pose_provider,
            movel_endpoint_provider=movel_endpoint_provider,
            endpoint_ik_label=endpoint_ik_label,
            endpoint_seed_mode=endpoint_seed_mode,
            active_arms=active_arms,
            initial_solution_validator=initial_solution_validator,
            endpoint_solution_validator=endpoint_solution_validator,
            priority_joint_arm=priority_joint_arm,
            priority_joint_index=priority_joint_index,
            priority_joint_upper_bound_deg=priority_joint_upper_bound_deg,
            auxiliary_candidate_validator=auxiliary_candidate_validator,
            seen=seen,
            progress_callback=progress_callback,
        )
        if not initial_valid:
            validation_scope = (
                f"MoveJ_P/{endpoint_ik_label}-IK"
                if movel_endpoint_provider is not None
                else "MoveJ_P-IK"
            )
            raise RuntimeError(
                f"no {validation_scope}-valid waist target in {initial_phase} search "
                f"among {initial_count} candidates"
            )

        refined_valid = []
        refined_count = 0
        if self._search_mode != "exhaustive":
            refine_centers = self._select_diverse_candidates(initial_valid)
            refine_candidates = self._refine_candidates(
                [candidate.waist_deg for candidate in refine_centers],
                search_lower,
                search_upper,
            )
            refined_valid, refined_count = self._evaluate_candidates(
                refine_candidates,
                phase="refine",
                nominal_deg=nominal_deg,
                left_seed_deg=left_seed_deg,
                right_seed_deg=right_seed_deg,
                target_pose_provider=target_pose_provider,
                movel_endpoint_provider=movel_endpoint_provider,
                endpoint_ik_label=endpoint_ik_label,
                endpoint_seed_mode=endpoint_seed_mode,
                active_arms=active_arms,
                initial_solution_validator=initial_solution_validator,
                endpoint_solution_validator=endpoint_solution_validator,
                priority_joint_arm=priority_joint_arm,
                priority_joint_index=priority_joint_index,
                priority_joint_upper_bound_deg=priority_joint_upper_bound_deg,
                auxiliary_candidate_validator=auxiliary_candidate_validator,
                seen=seen,
                progress_callback=progress_callback,
            )

        all_valid = initial_valid + refined_valid
        best = max(all_valid, key=lambda item: item.rank)
        robustness_checked = 0
        if self._robustness_delta_deg > 0:
            best = None
            for candidate in sorted(all_valid, key=lambda item: item.rank, reverse=True):
                robustness_checked += 1
                if progress_callback is not None:
                    progress_callback(
                        f"waist robustness check: candidate={robustness_checked}/{len(all_valid)}; "
                        f"waist_deg={candidate.waist_deg}; "
                        f"axis_delta_deg=+/-{self._robustness_delta_deg:.3f}"
                    )
                if min(candidate.left_margin_deg, candidate.right_margin_deg) < self._minimum_margin_deg:
                    continue
                passed = True
                for axis, sign in itertools.product(range(3), (-1, 1)):
                    perturbed = list(candidate.waist_deg)
                    perturbed[axis] += sign * self._robustness_delta_deg
                    if not self._waist_limits[0][axis] <= perturbed[axis] <= self._waist_limits[1][axis]:
                        passed = False
                        break
                    if progress_callback is not None:
                        progress_callback(
                            f"waist robustness probe: candidate={robustness_checked}; "
                            f"joint={axis + 1}; delta_deg={sign * self._robustness_delta_deg:+.3f}"
                        )
                    probe = self._evaluate_candidate(
                        perturbed, nominal_deg,
                        candidate.left_joint_deg, candidate.right_joint_deg,
                        target_pose_provider, movel_endpoint_provider,
                        endpoint_seed_mode, active_arms,
                        initial_solution_validator, endpoint_solution_validator,
                        priority_joint_arm, priority_joint_index,
                        priority_joint_upper_bound_deg, auxiliary_candidate_validator,
                    )
                    if (probe is None
                            or min(probe.left_margin_deg, probe.right_margin_deg) < self._minimum_margin_deg
                            or (probe.priority_joint_margin_deg is not None
                                and probe.priority_joint_margin_deg < self._minimum_margin_deg)):
                        passed = False
                        break
                if passed:
                    best = candidate
                    break
            if best is None:
                raise RuntimeError(
                    f"no robust waist target among {len(all_valid)} nominal IK-valid candidates; "
                    f"required J1/J2/J3 +/-{self._robustness_delta_deg:.3f}deg"
                )
            if progress_callback is not None:
                progress_callback(
                    f"waist robustness passed: checked_candidates={robustness_checked}; "
                    f"probes=6/6; axis_delta_deg=+/-{self._robustness_delta_deg:.3f}; "
                    "nominal ranking unchanged; valid count describes nominal candidates"
                )
        result = WaistWorkspaceResult(
            search_mode=self._search_mode,
            waist_angles_rad=tuple(math.radians(value) for value in best.waist_deg),
            left_joint_deg=best.left_joint_deg,
            right_joint_deg=best.right_joint_deg,
            left_margin_deg=best.left_margin_deg,
            right_margin_deg=best.right_margin_deg,
            candidate_count=initial_count + refined_count,
            valid_count=len(all_valid),
            initial_candidate_count=initial_count,
            refined_candidate_count=refined_count,
            priority_joint_margin_deg=best.priority_joint_margin_deg,
        )
        if result.score_deg < self._minimum_margin_deg:
            validation_scope = (
                f"MoveJ_P/{endpoint_ik_label} IK"
                if movel_endpoint_provider is not None
                else "MoveJ_P IK"
            )
            arm_scope = (
                "right-arm" if active_arms == ("right",) else "dual-arm"
            )
            raise RuntimeError(
                f"best {arm_scope} {validation_scope} joint margin is too small: "
                f"score={result.score_deg:.3f}deg, "
                f"required={self._minimum_margin_deg:.3f}deg"
            )
        if (
            result.priority_joint_margin_deg is not None
            and result.priority_joint_margin_deg < self._minimum_margin_deg
        ):
            raise RuntimeError(
                "best candidate priority-joint margin is too small: "
                f"score={result.priority_joint_margin_deg:.3f}deg, "
                f"required={self._minimum_margin_deg:.3f}deg"
            )
        return result


__all__ = ["WaistWorkspaceOptimizer", "WaistWorkspaceResult"]
