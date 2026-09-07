"""Strict serial state machine for connector and badge assembly."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .model import (
    NavigationRequest,
    StepResult,
    WorkflowCanceled,
    WorkflowError,
    WorkflowOutcome,
    WorkflowProgress,
)


CONNECTOR_POINT_ID = "1"
BADGE_POINT_ID = "2"
ASSEMBLY_POINT_ID = "3"
RESERVED_POINT_ID = "4"
TOTAL_STEPS = 9
VALID_POINT_IDS = frozenset({"1", "2", "3", "4"})


class WorkflowOperations(Protocol):
    def is_cancel_requested(self) -> bool: ...

    def navigate(self, request: NavigationRequest) -> StepResult: ...

    def grip(self, request_id: str, target_type: str) -> StepResult: ...

    def peel(self, request_id: str) -> StepResult: ...

    def assemble(self, request_id: str, target_type: str) -> StepResult: ...


ProgressCallback = Callable[[WorkflowProgress], None]


class AssemblyWorkflowEngine:
    """Execute point sequence 1 -> 3 -> 2 -> 3 with strict success gating."""

    def __init__(
        self,
        operations: WorkflowOperations,
        *,
        progress_callback: ProgressCallback | None = None,
        connector_point_id: str = CONNECTOR_POINT_ID,
        badge_point_id: str = BADGE_POINT_ID,
        assembly_point_id: str = ASSEMBLY_POINT_ID,
    ) -> None:
        self._operations = operations
        self._progress_callback = progress_callback or (lambda _progress: None)
        self._connector_point_id = str(connector_point_id)
        self._badge_point_id = str(badge_point_id)
        self._assembly_point_id = str(assembly_point_id)
        configured_points = {
            self._connector_point_id,
            self._badge_point_id,
            self._assembly_point_id,
        }
        if len(configured_points) != 3 or not configured_points.issubset(
            VALID_POINT_IDS
        ):
            raise ValueError(
                "connector, badge, and assembly points must be distinct ids in 1..4"
            )
        self._trace: list[str] = []
        self._completed_tasks = 0
        self._step_index = 0

    def run(self, workflow_id: str) -> WorkflowOutcome:
        try:
            self._navigate(workflow_id, self._connector_point_id, "connector")
            self._task(workflow_id, "GRIP_CONNECTOR", "connector", self._grip)
            self._navigate(workflow_id, self._assembly_point_id, "connector")
            self._task(
                workflow_id,
                "ASSEMBLY_CONNECTOR",
                "connector",
                self._assemble,
            )
            self._navigate(workflow_id, self._badge_point_id, "badge")
            self._task(workflow_id, "GRIP_BADGE", "badge", self._grip)
            self._task(workflow_id, "PEEL_BADGE", "badge", self._peel)
            self._navigate(workflow_id, self._assembly_point_id, "badge")
            self._task(workflow_id, "ASSEMBLY_BADGE", "badge", self._assemble)
            self._publish(
                workflow_id,
                "COMPLETE",
                task="badge",
                point_id=self._assembly_point_id,
                detail="workflow completed",
                advance=False,
            )
            return WorkflowOutcome(
                True,
                workflow_id,
                "connector and badge assembly workflow completed",
                "COMPLETE",
                self._completed_tasks,
                tuple(self._trace),
            )
        except WorkflowCanceled as exc:
            return WorkflowOutcome(
                False,
                workflow_id,
                str(exc),
                "CANCELED",
                self._completed_tasks,
                tuple(self._trace),
            )
        except WorkflowError as exc:
            return WorkflowOutcome(
                False,
                workflow_id,
                str(exc),
                exc.stage,
                self._completed_tasks,
                tuple(self._trace),
            )

    def _request_id(self, workflow_id: str, stage: str) -> str:
        return f"{workflow_id}:{self._step_index:02d}:{stage.lower()}"

    def _navigate(self, workflow_id: str, point_id: str, task: str) -> None:
        stage = f"NAVIGATE_{point_id}_{task.upper()}"
        self._publish(
            workflow_id,
            stage,
            task=task,
            point_id=point_id,
            detail=f"requesting navigation to point {point_id}",
        )
        self._check_cancel(stage)
        result = self._operations.navigate(
            NavigationRequest(
                workflow_id,
                self._request_id(workflow_id, stage),
                point_id,
            )
        )
        self._require_success(result, stage)

    def _task(self, workflow_id, stage, target_type, callback) -> None:
        self._publish(
            workflow_id,
            stage,
            task=target_type,
            detail=f"executing {stage.lower()}",
        )
        self._check_cancel(stage)
        result = callback(self._request_id(workflow_id, stage), target_type)
        self._require_success(result, stage)
        self._completed_tasks += 1

    def _grip(self, request_id: str, target_type: str) -> StepResult:
        return self._operations.grip(request_id, target_type)

    def _peel(self, request_id: str, _target_type: str) -> StepResult:
        return self._operations.peel(request_id)

    def _assemble(self, request_id: str, target_type: str) -> StepResult:
        return self._operations.assemble(request_id, target_type)

    def _publish(
        self,
        workflow_id: str,
        stage: str,
        *,
        task: str,
        point_id: str = "",
        detail: str = "",
        advance: bool = True,
    ) -> None:
        if advance:
            self._step_index += 1
        self._trace.append(stage)
        self._progress_callback(
            WorkflowProgress(
                workflow_id=workflow_id,
                stage=stage,
                current_point_id=point_id,
                current_task=task,
                current_step=self._step_index,
                total_steps=TOTAL_STEPS,
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
