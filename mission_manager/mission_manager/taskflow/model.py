"""Transport-independent workflow data models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NavigationRequest:
    workflow_id: str
    step_id: str
    point_id: str


@dataclass(frozen=True)
class NavigationResult:
    success: bool
    status: str
    message: str = ""


@dataclass(frozen=True)
class StepResult:
    success: bool
    message: str = ""


@dataclass(frozen=True)
class WorkflowProgress:
    workflow_id: str
    stage: str
    current_point_id: str = ""
    current_task: str = ""
    current_step: int = 0
    total_steps: int = 9
    detail: str = ""


@dataclass(frozen=True)
class WorkflowOutcome:
    success: bool
    workflow_id: str
    message: str
    final_stage: str
    completed_task_count: int = 0
    trace: tuple[str, ...] = field(default_factory=tuple)


class WorkflowError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class WorkflowCanceled(WorkflowError):
    pass
