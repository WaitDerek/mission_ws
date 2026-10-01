"""Durable, per-goal audit records for Mission-owned ROS actions.

This supplements ROS console logs: Action feedback and results are not retained by
the action protocol after the normal result timeout.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from rclpy.action import GoalResponse
from rosidl_runtime_py.convert import message_to_ordereddict


def _message_dict(message):
    try:
        return message_to_ordereddict(message)
    except Exception:  # A malformed ROS message must not change the action outcome.
        return str(message)


def _goal_id(goal_handle) -> str:
    return bytes(goal_handle.goal_id.uuid).hex()


class ActionAuditStore:
    """Append one JSON line per goal event without affecting robot motion on I/O errors."""

    def __init__(self, node_name: str, logger, directory: str | None = None):
        root = directory or os.environ.get("MISSION_ACTION_AUDIT_DIR")
        default_root = (
            Path("/rm_nvme/recordings/mission_logs/action_audit")
            if Path("/rm_nvme/recordings").is_dir()
            else Path.home() / ".ros/log/mission_action_audit"
        )
        self._root = Path(root) if root else default_root
        self._node_name = re.sub(r"[^A-Za-z0-9_-]", "_", node_name)
        self._logger = logger
        self._lock = threading.Lock()
        self._goal_files: dict[str, Path] = {}
        self._last_warning = 0.0

    def _write(self, path: Path, event: str, *, sync: bool = False, **fields) -> None:
        record = {
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "node": self._node_name,
            "event": event,
            **fields,
        }
        try:
            data = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
            with self._lock:
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(descriptor, data)
                    if sync:
                        os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        except Exception as exc:  # Audit I/O must never alter a motion outcome.
            now = time.monotonic()
            if now - self._last_warning >= 60.0:
                self._last_warning = now
                self._logger.error(f"Mission action audit write failed: {exc}")

    def decision(self, action: str, request, accepted: bool, error: str = "") -> None:
        day = datetime.now().astimezone().strftime("%Y-%m-%d")
        self._write(
            self._root / day / self._node_name / "goal_decisions.jsonl",
            "goal_decision",
            sync=True,
            action=action,
            request_id=str(getattr(request, "request_id", "")),
            accepted=accepted,
            error=error,
            goal=_message_dict(request),
        )

    def started(self, action: str, goal_handle) -> str:
        goal_id = _goal_id(goal_handle)
        day = datetime.now().astimezone().strftime("%Y-%m-%d")
        safe_action = re.sub(r"[^A-Za-z0-9_-]", "_", action).strip("_")
        path = self._root / day / self._node_name / f"{safe_action}_{goal_id}.jsonl"
        with self._lock:
            self._goal_files[goal_id] = path
        self._write(
            path,
            "start",
            sync=True,
            action=action,
            goal_id=goal_id,
            request_id=str(getattr(goal_handle.request, "request_id", "")),
            goal=_message_dict(goal_handle.request),
        )
        return goal_id

    def feedback(self, goal_handle, message) -> None:
        goal_id = _goal_id(goal_handle)
        with self._lock:
            path = self._goal_files.get(goal_id)
        if path is not None:
            self._write(path, "feedback", goal_id=goal_id, feedback=_message_dict(message))

    def finished(self, goal_handle, result=None, error: str = "") -> None:
        goal_id = _goal_id(goal_handle)
        with self._lock:
            path = self._goal_files.pop(goal_id, None)
        if path is not None:
            self._write(
                path,
                "result" if not error else "exception",
                sync=True,
                goal_id=goal_id,
                goal_status=int(goal_handle.status),
                success=getattr(result, "success", None),
                message=str(getattr(result, "message", "")),
                result=_message_dict(result) if result is not None else None,
                error=error,
            )


class ActionAuditMixin:
    """Keep action registration unchanged except for its two callback wrappers."""

    def _audit_goal_callback(self, action: str, callback):
        def wrapped(request):
            try:
                response = callback(request)
            except Exception as exc:
                self._action_audit.decision(action, request, False, str(exc))
                raise
            self._action_audit.decision(action, request, response == GoalResponse.ACCEPT)
            return response

        return wrapped

    def _audit_execute_callback(self, action: str, callback):
        def wrapped(goal_handle):
            self._action_audit.started(action, goal_handle)
            try:
                result = callback(goal_handle)
            except Exception as exc:
                self._action_audit.finished(goal_handle, error=repr(exc))
                raise
            self._action_audit.finished(goal_handle, result=result)
            return result

        return wrapped

    def _audit_feedback(self, goal_handle, message) -> None:
        audit = getattr(self, "_action_audit", None)
        if audit is not None:
            audit.feedback(goal_handle, message)
