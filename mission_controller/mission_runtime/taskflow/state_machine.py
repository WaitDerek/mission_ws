"""Strictly serial, transport-independent depalletizing workflow engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .identifiers import IdentifierFactory
from .mapping import (
    PLACE_POINT_ID,
    grasp_action_for_operation_point,
    operation_point_for_stack,
)
from .model import (
    NavigationRequest,
    ObservationResult,
    ObservationTask,
    StepResult,
    WorkflowCanceled,
    WorkflowError,
    WorkflowOutcome,
    WorkflowProgress,
)


class WorkflowOperations(Protocol):
    def is_cancel_requested(self) -> bool: ...

    def navigate(self, request: NavigationRequest) -> StepResult: ...

    def observe(self, point_id: str) -> ObservationResult: ...

    def grasp(
        self,
        action_name: str,
        request_id: str,
        task: ObservationTask,
        operation_point_id: str,
    ) -> StepResult: ...

    def place(self, request_id: str, task: ObservationTask) -> StepResult: ...


ProgressCallback = Callable[[WorkflowProgress], None]


class DepalletizingWorkflowEngine:
    def __init__(
        self,
        operations: WorkflowOperations,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._operations = operations
        self._progress_callback = progress_callback or (lambda _progress: None)
        self._trace: list[str] = []
        self._completed_observations = 0
        self._completed_boxes = 0

    def run(self, workflow_id: str, lease_token: str) -> WorkflowOutcome:
        identifiers = IdentifierFactory(workflow_id, lease_token)
        final_stage = "INITIALIZING"
        try:
            self._check_cancel(final_stage)
            first = self._navigate_and_observe(
                workflow_id, identifiers, "1", allow_observation_failure=True
            )
            if first is not None:
                self._process_plan(workflow_id, identifiers, first)
                opposite = self._navigate_and_observe(
                    workflow_id, identifiers, "3", allow_observation_failure=False
                )
                if opposite is None:  # defensive; failure is not allowed here
                    raise WorkflowError("OBSERVE_3", "point 3 returned no plan")
                self._process_plan(workflow_id, identifiers, opposite)
            else:
                fallback = self._navigate_and_observe(
                    workflow_id, identifiers, "2", allow_observation_failure=True
                )
                if fallback is None:
                    raise WorkflowError(
                        "OBSERVE_2", "points 1 and 2 returned no actionable plan"
                    )
                self._process_plan(workflow_id, identifiers, fallback)
                opposite = self._navigate_and_observe(
                    workflow_id, identifiers, "4", allow_observation_failure=False
                )
                if opposite is None:
                    raise WorkflowError("OBSERVE_4", "point 4 returned no plan")
                self._process_plan(workflow_id, identifiers, opposite)

            final_stage = "COMPLETE"
            self._publish(workflow_id, final_stage, detail="workflow completed")
            return WorkflowOutcome(
                True,
                workflow_id,
                "depalletizing workflow completed",
                final_stage,
                self._completed_observations,
                self._completed_boxes,
                tuple(self._trace),
            )
        except WorkflowCanceled as exc:
            return WorkflowOutcome(
                False,
                workflow_id,
                str(exc),
                "CANCELED",
                self._completed_observations,
                self._completed_boxes,
                tuple(self._trace),
            )
        except WorkflowError as exc:
            return WorkflowOutcome(
                False,
                workflow_id,
                str(exc),
                exc.stage,
                self._completed_observations,
                self._completed_boxes,
                tuple(self._trace),
            )

    def _navigate_and_observe(
        self,
        workflow_id: str,
        identifiers: IdentifierFactory,
        point_id: str,
        *,
        allow_observation_failure: bool,
    ):
        stage = f"NAVIGATE_OBSERVATION_{point_id}"
        step_id, _request_id = identifiers.request_id(stage, point_id)
        self._publish(workflow_id, stage, point_id, detail="requesting navigation")
        self._check_cancel(stage)
        result = self._operations.navigate(
            NavigationRequest(workflow_id, step_id, point_id)
        )
        self._require_success(result, stage)
        self._check_cancel(stage)

        stage = f"OBSERVE_{point_id}"
        self._publish(
            workflow_id, stage, point_id, detail="requesting global observation"
        )
        observation = self._operations.observe(point_id)
        self._check_cancel(stage)
        if (
            not observation.success
            or observation.plan is None
            or not observation.plan.actionable
        ):
            if allow_observation_failure:
                self._publish(
                    workflow_id,
                    f"{stage}_NO_PLAN",
                    point_id,
                    detail=observation.message or "no actionable plan",
                )
                return None
            raise WorkflowError(stage, observation.message or "no actionable plan")
        self._completed_observations += 1
        return observation.plan

    def _process_plan(
        self, workflow_id: str, identifiers: IdentifierFactory, plan
    ) -> None:
        total = len(plan.tasks)
        for task in plan.tasks:
            self._check_cancel("PROCESS_PLAN")
            operation_point = operation_point_for_stack(plan.point_id, task.stack_index)
            nav_stage = f"NAVIGATE_PICKUP_{operation_point}"
            nav_step_id, _ = identifiers.request_id(
                nav_stage, operation_point, task.order_index
            )
            self._publish(
                workflow_id,
                nav_stage,
                operation_point,
                task,
                total,
                "requesting pickup navigation",
            )
            nav_result = self._operations.navigate(
                NavigationRequest(workflow_id, nav_step_id, operation_point)
            )
            self._require_success(nav_result, nav_stage)
            self._check_cancel(nav_stage)

            action_name = grasp_action_for_operation_point(operation_point)
            grasp_stage = (
                "GRASP_DRAG_TF" if int(operation_point) % 2 else "GRASP_DIRECT_TF"
            )
            _, grasp_request_id = identifiers.request_id(
                grasp_stage, operation_point, task.order_index
            )
            self._publish(
                workflow_id,
                grasp_stage,
                operation_point,
                task,
                total,
                f"calling {action_name}",
            )
            grasp_result = self._operations.grasp(
                action_name,
                grasp_request_id,
                task,
                operation_point,
            )
            self._require_success(grasp_result, grasp_stage)
            self._check_cancel(grasp_stage)

            place_nav_stage = "NAVIGATE_PLACE_16"
            place_nav_step_id, _ = identifiers.request_id(
                place_nav_stage, PLACE_POINT_ID, task.order_index
            )
            self._publish(
                workflow_id,
                place_nav_stage,
                PLACE_POINT_ID,
                task,
                total,
                "requesting placement navigation",
            )
            place_nav_result = self._operations.navigate(
                NavigationRequest(workflow_id, place_nav_step_id, PLACE_POINT_ID)
            )
            self._require_success(place_nav_result, place_nav_stage)
            self._check_cancel(place_nav_stage)

            place_stage = "PLACE_BOX"
            _, place_request_id = identifiers.request_id(
                place_stage, PLACE_POINT_ID, task.order_index
            )
            self._publish(
                workflow_id,
                place_stage,
                PLACE_POINT_ID,
                task,
                total,
                "calling /execute_box_place",
            )
            place_result = self._operations.place(place_request_id, task)
            self._require_success(place_result, place_stage)
            self._completed_boxes += 1
            self._check_cancel(place_stage)

    def _publish(
        self,
        workflow_id: str,
        stage: str,
        point_id: str = "",
        task: ObservationTask | None = None,
        total: int = 0,
        detail: str = "",
    ) -> None:
        self._trace.append(stage)
        self._progress_callback(
            WorkflowProgress(
                workflow_id=workflow_id,
                stage=stage,
                current_point_id=point_id,
                current_stack_id=task.stack_id if task else "",
                current_order_index=task.order_index if task else 0,
                total_order_items=total,
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
