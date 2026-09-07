"""Pure serial state machine for the fixed eight-box workflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Protocol

from .fixed_model import FixedBoxTask, FixedWorkflowOutcome
from .identifiers import IdentifierFactory
from .model import NavigationRequest, StepResult, WorkflowCanceled, WorkflowError


class FixedWorkflowOperations(Protocol):
    def is_cancel_requested(self) -> bool: ...

    def navigate(self, request: NavigationRequest) -> StepResult: ...

    def grasp(self, action_name: str, request_id: str, task: FixedBoxTask) -> StepResult: ...

    def place(self, request_id: str, box_type: str) -> StepResult: ...


FixedProgressCallback = Callable[[str, Optional[FixedBoxTask], str], None]


class FixedBoxWorkflowEngine:
    """Execute one item completely before starting the next item."""

    TOTAL_ITEMS = 8

    def __init__(
        self,
        operations: FixedWorkflowOperations,
        tasks: tuple[FixedBoxTask, ...],
        *,
        place_point_id: int,
        place_pos: tuple[float, float, float],
        progress_callback: FixedProgressCallback | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("fixed workflow must contain at least one task")
        if tuple(task.item_index for task in tasks) != tuple(
            range(1, len(tasks) + 1)
        ):
            raise ValueError("fixed workflow task indexes must be contiguous from 1")
        self._operations = operations
        self._tasks = tasks
        self.total_items = len(tasks)
        self._place_point_id = str(int(place_point_id))
        self._place_pos = tuple(float(value) for value in place_pos)
        if len(self._place_pos) != 3:
            raise ValueError("fixed workflow placement pose requires [x, y, yaw]")
        self._progress_callback = progress_callback or (lambda *_args: None)
        self._trace: list[str] = []
        self._completed = 0
        self._last_completed = 0
        self._current_workflow_id = ""

    def run(
        self,
        workflow_id: str,
        lease_token: str,
        *,
        start_item_index: int = 1,
        stop_after_item_index: int = 0,
    ) -> FixedWorkflowOutcome:
        final_stage = "INITIALIZING"
        try:
            self._validate_range(start_item_index, stop_after_item_index)
            self._current_workflow_id = str(workflow_id)
            identifiers = IdentifierFactory(workflow_id, lease_token)
            stop = stop_after_item_index or self.total_items
            self._check_cancel(final_stage)
            for task in self._tasks[start_item_index - 1 : stop]:
                self._run_item(task, identifiers)
            final_stage = "COMPLETE"
            self._publish(final_stage, None, "fixed workflow completed")
            return FixedWorkflowOutcome(
                True,
                workflow_id,
                self._completed,
                self._last_completed,
                final_stage,
                "fixed box workflow completed",
                tuple(self._trace),
            )
        except WorkflowCanceled as exc:
            return FixedWorkflowOutcome(
                False,
                workflow_id,
                self._completed,
                self._last_completed,
                "CANCELED",
                str(exc),
                tuple(self._trace),
            )
        except WorkflowError as exc:
            return FixedWorkflowOutcome(
                False,
                workflow_id,
                self._completed,
                self._last_completed,
                exc.stage,
                str(exc),
                tuple(self._trace),
            )

    def _run_item(self, task: FixedBoxTask, identifiers: IdentifierFactory) -> None:
        self._check_cancel(f"ITEM_{task.item_index}")
        pickup_stage = f"NAVIGATE_PICKUP_{task.item_index}"
        pickup_step, _ = identifiers.request_id(
            pickup_stage, task.pickup_point_id, task.item_index
        )
        self._publish(
            pickup_stage,
            task,
            f"requesting pickup navigation point {task.pickup_point_id} "
            f"with pos=[{task.pickup_pos[0]:g},{task.pickup_pos[1]:g},{task.pickup_pos[2]:g}]",
        )
        self._require(
            self._operations.navigate(
                NavigationRequest(
                    workflow_id=self._workflow_id,
                    step_id=pickup_step,
                    point_id=task.pickup_point_id,
                    pos=task.pickup_pos,
                )
            ),
            pickup_stage,
        )

        grasp_stage = f"GRASP_{task.item_index}"
        _, grasp_request = identifiers.request_id(
            grasp_stage, task.pickup_point_id, task.item_index
        )
        self._publish(
            grasp_stage,
            task,
            f"calling {task.grasp_action} for {task.box_type} layer {task.box_layer}",
        )
        self._require(
            self._operations.grasp(task.grasp_action, grasp_request, task),
            grasp_stage,
        )

        place_nav_stage = f"NAVIGATE_PLACE_{task.item_index}"
        place_step, _ = identifiers.request_id(
            place_nav_stage, self._place_point_id, task.item_index
        )
        self._publish(
            place_nav_stage,
            task,
            f"requesting placement navigation point {self._place_point_id} "
            f"with pos=[{self._place_pos[0]:g},{self._place_pos[1]:g},{self._place_pos[2]:g}]",
        )
        self._require(
            self._operations.navigate(
                NavigationRequest(
                    workflow_id=self._workflow_id,
                    step_id=place_step,
                    point_id=self._place_point_id,
                    pos=self._place_pos,
                )
            ),
            place_nav_stage,
        )

        place_stage = f"PLACE_{task.item_index}"
        _, place_request = identifiers.request_id(
            place_stage, self._place_point_id, task.item_index
        )
        self._publish(place_stage, task, "calling /place_box_test")
        self._require(
            self._operations.place(place_request, task.box_type), place_stage
        )
        self._completed += 1
        self._last_completed = task.item_index
        self._publish(
            f"ITEM_{task.item_index}_COMPLETE",
            task,
            "pickup, grasp, placement navigation, and placement completed",
        )

    def _publish(self, stage: str, task: FixedBoxTask | None, detail: str) -> None:
        self._trace.append(stage)
        self._progress_callback(stage, task, detail)

    def _check_cancel(self, stage: str) -> None:
        if self._operations.is_cancel_requested():
            raise WorkflowCanceled(stage, f"fixed workflow canceled during {stage}")

    def _require(self, result: StepResult, stage: str) -> None:
        self._check_cancel(stage)
        if not result.success:
            raise WorkflowError(stage, result.message or f"{stage} failed")

    def _validate_range(self, start: int, stop: int) -> None:
        try:
            start = int(start)
            stop = int(stop)
        except (TypeError, ValueError) as exc:
            raise WorkflowError("VALIDATING", "item indexes must be integers") from exc
        if not 1 <= start <= self.total_items:
            raise WorkflowError(
                "VALIDATING", f"start_item_index must be in [1,{self.total_items}]"
            )
        if stop and not start <= stop <= self.total_items:
            raise WorkflowError(
                "VALIDATING",
                "stop_after_item_index must be 0 or in "
                f"[start_item_index,{self.total_items}]",
            )

    @property
    def _workflow_id(self) -> str:
        return self._current_workflow_id
