"""Thread-safe Mission ownership shared by legacy and workflow goals."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .identifiers import new_lease_token, parse_request_id, sanitize_request_id


@dataclass(frozen=True)
class LeaseResult:
    success: bool
    message: str
    lease_token: str = ""


@dataclass(frozen=True)
class GoalReservation:
    accepted: bool
    message: str
    sanitized_request_id: str
    workflow_owned: bool = False


class WorkflowLeaseManager:
    """Own one workflow lease and, within it, one child goal at a time.

    There is deliberately no TTL. If the workflow process dies, ownership
    remains fail-closed until the MissionController process is restarted.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workflow_id = ""
        self._lease_token = ""
        self._goal_reserved = False
        self._active_mission = ""
        self._active_goal_workflow_owned = False

    @property
    def workflow_id(self) -> str:
        with self._lock:
            return self._workflow_id

    @property
    def active_mission(self) -> str:
        with self._lock:
            return self._active_mission

    @property
    def goal_reserved(self) -> bool:
        with self._lock:
            return self._goal_reserved

    def acquire(self, workflow_id: object) -> LeaseResult:
        identity = str(workflow_id).strip()
        if not identity:
            return LeaseResult(False, "workflow_id must not be empty")
        with self._lock:
            if self._workflow_id:
                return LeaseResult(
                    False,
                    f"workflow lease {self._workflow_id} is already active",
                )
            if self._goal_reserved:
                return LeaseResult(
                    False,
                    f"legacy mission {self._active_mission} is active",
                )
            self._workflow_id = identity
            self._lease_token = new_lease_token()
            return LeaseResult(
                True,
                "workflow lease acquired",
                self._lease_token,
            )

    def reserve_goal(self, mission: str, request_id: object) -> GoalReservation:
        sanitized = sanitize_request_id(request_id)
        try:
            identity = parse_request_id(request_id)
        except ValueError:
            identity = None

        with self._lock:
            if self._lease_token and self._lease_token in sanitized:
                sanitized = sanitized.replace(self._lease_token, "<redacted>")
            if self._goal_reserved:
                return GoalReservation(
                    False,
                    f"{self._active_mission} mission is active",
                    sanitized,
                )

            if self._workflow_id:
                if identity is None:
                    return GoalReservation(
                        False,
                        f"workflow {self._workflow_id} owns Mission",
                        sanitized,
                    )
                if (
                    identity.workflow_id != self._workflow_id
                    or identity.lease_token != self._lease_token
                ):
                    return GoalReservation(
                        False,
                        "workflow ownership did not match the active lease",
                        sanitized,
                    )
                workflow_owned = True
            else:
                if identity is not None:
                    return GoalReservation(
                        False,
                        "workflow request has no active Mission lease",
                        sanitized,
                    )
                workflow_owned = False

            self._goal_reserved = True
            self._active_mission = str(mission)
            self._active_goal_workflow_owned = workflow_owned
            return GoalReservation(
                True,
                "goal reserved",
                sanitized,
                workflow_owned,
            )

    def release_goal(self) -> None:
        with self._lock:
            self._goal_reserved = False
            self._active_mission = ""
            self._active_goal_workflow_owned = False

    def release(self, workflow_id: object, lease_token: object) -> LeaseResult:
        identity = str(workflow_id).strip()
        token = str(lease_token).strip()
        with self._lock:
            if not self._workflow_id:
                # Idempotent cleanup after a successfully released workflow.
                return LeaseResult(True, "no workflow lease is active")
            if identity != self._workflow_id or token != self._lease_token:
                return LeaseResult(False, "workflow lease identity did not match")
            if self._goal_reserved and self._active_goal_workflow_owned:
                return LeaseResult(
                    False,
                    f"workflow child {self._active_mission} is still active",
                )
            if self._goal_reserved:
                return LeaseResult(
                    False,
                    f"legacy mission {self._active_mission} is still active",
                )
            self._workflow_id = ""
            self._lease_token = ""
            return LeaseResult(True, "workflow lease released")
