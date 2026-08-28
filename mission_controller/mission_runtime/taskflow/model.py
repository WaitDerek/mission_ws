"""Transport-independent data models for the depalletizing workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ObservationTask:
    """One Vision order item, normalized for Mission execution."""

    stack_id: str
    stack_index: int
    column: int
    layer: int
    box_type: str
    order_index: int


@dataclass(frozen=True)
class ObservationPlan:
    """Validated order returned for one global observation point."""

    point_id: str
    tasks: tuple[ObservationTask, ...] = ()
    message: str = ""

    @property
    def actionable(self) -> bool:
        return bool(self.tasks)


@dataclass(frozen=True)
class ObservationResult:
    """Outcome of invoking the Vision GlobalObservation Action."""

    success: bool
    plan: Optional[ObservationPlan] = None
    message: str = ""


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
    current_stack_id: str = ""
    current_order_index: int = 0
    total_order_items: int = 0
    detail: str = ""


@dataclass(frozen=True)
class WorkflowOutcome:
    success: bool
    workflow_id: str
    message: str
    final_stage: str
    completed_observation_count: int = 0
    completed_box_count: int = 0
    trace: tuple[str, ...] = field(default_factory=tuple)


class WorkflowError(RuntimeError):
    """Base error carrying the stage that stopped the workflow."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class WorkflowCanceled(WorkflowError):
    pass
