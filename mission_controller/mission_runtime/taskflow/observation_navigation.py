"""Strict navigation/global-observation/navigation test workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .identifiers import IdentifierFactory
from .mapping import (
    DIRECT_GRASP_ACTION,
    DRAG_GRASP_ACTION,
    OBSERVATION_POINT_IDS,
)
from .model import (
    NavigationRequest,
    ObservationResult,
    StepResult,
    WorkflowCanceled,
    WorkflowError,
    WorkflowProgress,
)


class ObservationNavigationOperations(Protocol):
    def is_cancel_requested(self) -> bool: ...

    def navigate(self, request: NavigationRequest) -> StepResult: ...

    def observe(self, point_id: str) -> ObservationResult: ...


@dataclass(frozen=True)
class ObservationNavigationOutcome:
    success: bool
    workflow_id: str
    message: str
    final_stage: str
    completed_navigation_count: int = 0
    completed_observation_count: int = 0
    planned_box_count: int = 0
    selected_operation_point_id: str = ""
    trace: tuple[str, ...] = field(default_factory=tuple)


ProgressCallback = Callable[[WorkflowProgress], None]


# The platform station exposes four logical navigation IDs to Mission:
# 1=right drag pickup, 2=placement, 3=left direct pickup, 4=global view.
# The left layer-4 approach has a separate map pose and is sent as a custom
# navigation pose because it shares logical point 3 with layers 1-3.
_RIGHT_PICKUP_POINT_ID = "1"
_RIGHT_PICKUP_POS = (0.65, -0.05, 1.24)
_LEFT_PICKUP_POINT_ID = "3"
_LEFT_PICKUP_POS = (-0.10, 0.20, 1.23)
_LEFT_LAYER4_PICKUP_POS = (-0.10, 0.30, 1.23)


def _operation_navigation_target(task) -> tuple[str, tuple[float, float, float]]:
    """Resolve the new station pickup point for one Vision order item."""

    stack_index = int(task.stack_index)
    if stack_index == 0:
        position = (
            _LEFT_LAYER4_PICKUP_POS
            if int(task.layer) == 4
            and str(task.box_type).strip().lower() == "smallbox"
            else _LEFT_PICKUP_POS
        )
        return _LEFT_PICKUP_POINT_ID, position
    if stack_index == 1:
        return _RIGHT_PICKUP_POINT_ID, _RIGHT_PICKUP_POS
    raise ValueError(f"unsupported Vision stack index {stack_index}; expected 0 or 1")


class ObservationNavigationWorkflowEngine:
    """Run only global navigation and global observation effects.

    After observation, navigate to every operation point in Vision's validated
    order. The workflow deliberately performs no grasp or placement Action.
    """

    def __init__(
        self,
        operations: ObservationNavigationOperations,
        *,
        place_point_id: str = "2",
        place_pos: tuple[float, float, float] = (-0.81, -0.77, -1.89),
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._operations = operations
        self._progress_callback = progress_callback or (lambda _progress: None)
        self._trace: list[str] = []
        self._completed_navigation_count = 0
        self._completed_observation_count = 0
        self._planned_box_count = 0
        self._selected_operation_point_id = ""
        self._place_point_id = str(place_point_id)
        self._place_pos = tuple(float(value) for value in place_pos)
        if len(self._place_pos) != 3:
            raise ValueError("placement navigation pose requires [x, y, yaw]")

    def run(
        self,
        workflow_id: str,
        lease_token: str,
        observation_point_id: object,
    ) -> ObservationNavigationOutcome:
        point_id = str(observation_point_id).strip()
        final_stage = "VALIDATING"
        try:
            if point_id not in OBSERVATION_POINT_IDS:
                raise WorkflowError(
                    final_stage,
                    f"observation_point_id must be one of {OBSERVATION_POINT_IDS}",
                )
            identifiers = IdentifierFactory(workflow_id, lease_token)
            self._check_cancel(final_stage)

            stage = f"NAVIGATE_OBSERVATION_{point_id}"
            step_id, _ = identifiers.request_id(stage, point_id)
            self._publish(
                workflow_id,
                stage,
                point_id=point_id,
                detail=f"requesting navigation to observation point {point_id}",
            )
            self._require_success(
                self._operations.navigate(
                    NavigationRequest(workflow_id, step_id, point_id)
                ),
                stage,
            )
            self._completed_navigation_count += 1
            self._check_cancel(stage)

            stage = f"OBSERVE_{point_id}"
            self._publish(
                workflow_id,
                stage,
                point_id=point_id,
                detail="requesting global observation",
            )
            observation = self._operations.observe(point_id)
            self._check_cancel(stage)
            if (
                not observation.success
                or observation.plan is None
                or not observation.plan.actionable
            ):
                raise WorkflowError(
                    stage,
                    observation.message
                    or "global observation returned no actionable plan",
                )
            plan = observation.plan
            self._completed_observation_count = 1
            self._planned_box_count = len(plan.tasks)
            # Publish the complete Vision plan immediately after observation,
            # before the robot starts moving through the individual tasks.
            for plan_index, planned_task in enumerate(plan.tasks, start=1):
                planned_point_id, planned_pos = _operation_navigation_target(
                    planned_task
                )
                grasp_mode = str(
                    getattr(planned_task, "grasp_mode", "unknown")
                ).strip().lower()
                grasp_action = (
                    DRAG_GRASP_ACTION
                    if grasp_mode == "drag"
                    else DIRECT_GRASP_ACTION
                    if grasp_mode == "direct_grasp"
                    else "unknown"
                )
                self._publish(
                    workflow_id,
                    f"GLOBAL_PLAN_ITEM_{plan_index}",
                    point_id=planned_point_id,
                    stack_id=planned_task.stack_id,
                    current_order_index=planned_task.order_index,
                    total_order_items=len(plan.tasks),
                    detail=(
                        f"global plan item {plan_index}/{len(plan.tasks)}: "
                        f"pickup_point={planned_point_id}, "
                        f"pickup_pos=[{planned_pos[0]:g},{planned_pos[1]:g},"
                        f"{planned_pos[2]:g}], "
                        f"grasp_action={grasp_action}, grasp_mode={grasp_mode}, "
                        f"box_type={planned_task.box_type}, "
                        f"layer={planned_task.layer}, "
                        f"stack_id={planned_task.stack_id}"
                    ),
                )

            for task_index, task in enumerate(plan.tasks):
                operation_point_id, operation_pos = _operation_navigation_target(task)
                if not self._selected_operation_point_id:
                    self._selected_operation_point_id = operation_point_id
                stage = (
                    f"NAVIGATE_OPERATION_{operation_point_id}"
                    if task_index == 0
                    else f"NAVIGATE_OPERATION_{operation_point_id}_ITEM_{task_index + 1}"
                )
                step_id, _ = identifiers.request_id(
                    stage, operation_point_id, task.order_index
                )
                self._publish(
                    workflow_id,
                    stage,
                    point_id=operation_point_id,
                    stack_id=task.stack_id,
                    current_order_index=task.order_index,
                    total_order_items=len(plan.tasks),
                    detail=(
                        f"requesting navigation to planned operation point "
                        f"{operation_point_id} ({task_index + 1}/{len(plan.tasks)}); "
                        f"grasp_info=mode={getattr(task, 'grasp_mode', 'unknown')}, "
                        f"box_type={task.box_type}, layer={task.layer}, "
                        f"stack_id={task.stack_id}"
                    ),
                )
                self._require_success(
                    self._operations.navigate(
                        NavigationRequest(
                            workflow_id,
                            step_id,
                            operation_point_id,
                            pos=operation_pos,
                        )
                    ),
                    stage,
                )
                self._completed_navigation_count += 1
                self._check_cancel(stage)
                self._publish(
                    workflow_id,
                    f"PICKUP_REACHED_{task_index + 1}",
                    point_id=operation_point_id,
                    stack_id=task.stack_id,
                    current_order_index=task.order_index,
                    total_order_items=len(plan.tasks),
                    detail=(
                        f"arrived at pickup point {operation_point_id}; "
                        f"grasp_info=mode={getattr(task, 'grasp_mode', 'unknown')}, "
                        f"box_type={task.box_type}, layer={task.layer}, "
                        f"target_pos=[{operation_pos[0]:g},{operation_pos[1]:g},{operation_pos[2]:g}]"
                    ),
                )

                place_stage = f"NAVIGATE_PLACE_{task_index + 1}"
                place_step, _ = identifiers.request_id(
                    place_stage, self._place_point_id, task.order_index
                )
                self._publish(
                    workflow_id,
                    place_stage,
                    point_id=self._place_point_id,
                    stack_id=task.stack_id,
                    current_order_index=task.order_index,
                    total_order_items=len(plan.tasks),
                    detail=(
                        f"requesting navigation to placement point "
                        f"{self._place_point_id} after pickup {task_index + 1}/"
                        f"{len(plan.tasks)} with pos=[{self._place_pos[0]:g},"
                        f"{self._place_pos[1]:g},{self._place_pos[2]:g}]"
                    ),
                )
                self._require_success(
                    self._operations.navigate(
                        NavigationRequest(
                            workflow_id,
                            place_step,
                            self._place_point_id,
                            pos=self._place_pos,
                        )
                    ),
                    place_stage,
                )
                self._completed_navigation_count += 1
                self._check_cancel(place_stage)
                self._publish(
                    workflow_id,
                    f"PLACE_REACHED_{task_index + 1}",
                    point_id=self._place_point_id,
                    stack_id=task.stack_id,
                    current_order_index=task.order_index,
                    total_order_items=len(plan.tasks),
                    detail=(
                        f"arrived at placement point {self._place_point_id}; "
                        f"completed pickup/place navigation {task_index + 1}/"
                        f"{len(plan.tasks)}"
                    ),
                )

            final_stage = "COMPLETE"
            self._publish(
                workflow_id,
                final_stage,
                point_id=self._selected_operation_point_id,
                stack_id=plan.tasks[-1].stack_id,
                current_order_index=plan.tasks[-1].order_index,
                total_order_items=len(plan.tasks),
                detail=(
                    f"global observation, pickup, and placement navigation "
                    f"completed for all {len(plan.tasks)} planned tasks"
                ),
            )
            return self._outcome(
                True,
                workflow_id,
                "navigation/global-observation/navigation workflow completed",
                final_stage,
            )
        except WorkflowCanceled as exc:
            return self._outcome(False, workflow_id, str(exc), "CANCELED")
        except WorkflowError as exc:
            return self._outcome(False, workflow_id, str(exc), exc.stage)

    def _publish(
        self,
        workflow_id: str,
        stage: str,
        *,
        point_id: str = "",
        stack_id: str = "",
        current_order_index: int = 0,
        total_order_items: int = 0,
        detail: str = "",
    ) -> None:
        self._trace.append(stage)
        self._progress_callback(
            WorkflowProgress(
                workflow_id=workflow_id,
                stage=stage,
                current_point_id=point_id,
                current_stack_id=stack_id,
                current_order_index=current_order_index,
                total_order_items=total_order_items,
                detail=detail,
            )
        )

    def _check_cancel(self, stage: str) -> None:
        if self._operations.is_cancel_requested():
            raise WorkflowCanceled(stage, f"workflow canceled during {stage}")

    def _require_success(self, result: StepResult, stage: str) -> None:
        if not result.success:
            self._check_cancel(stage)
            raise WorkflowError(stage, result.message or f"{stage} failed")

    def _outcome(
        self,
        success: bool,
        workflow_id: str,
        message: str,
        final_stage: str,
    ) -> ObservationNavigationOutcome:
        return ObservationNavigationOutcome(
            success=success,
            workflow_id=workflow_id,
            message=message,
            final_stage=final_stage,
            completed_navigation_count=self._completed_navigation_count,
            completed_observation_count=self._completed_observation_count,
            planned_box_count=self._planned_box_count,
            selected_operation_point_id=self._selected_operation_point_id,
            trace=tuple(self._trace),
        )
