"""Legacy EE-pose subscriptions used only by staged-migration run methods."""

from __future__ import annotations

import threading
import time
from typing import Optional

from geometry_msgs.msg import PoseStamped

from .common import MissionError


class RunPoseRuntimeMixin:
    """Keep the original G1D pose topics separate from Execute* subscriptions."""

    def _initialize_run_pose_runtime(self) -> None:
        self._run_pose_condition = threading.Condition()
        self._run_left_ee_pose: Optional[PoseStamped] = None
        self._run_right_ee_pose: Optional[PoseStamped] = None
        self._run_left_ee_sequence = 0
        self._run_right_ee_sequence = 0

        self.run_left_ee_subscription = self.create_subscription(
            PoseStamped,
            self._string("run_left_ee_pose_topic"),
            self._run_left_ee_callback,
            10,
            callback_group=self._callback_group,
        )
        self.run_right_ee_subscription = self.create_subscription(
            PoseStamped,
            self._string("run_right_ee_pose_topic"),
            self._run_right_ee_callback,
            10,
            callback_group=self._callback_group,
        )

    def _run_left_ee_callback(self, message: PoseStamped) -> None:
        with self._run_pose_condition:
            self._run_left_ee_pose = message
            self._run_left_ee_sequence += 1
            self._run_pose_condition.notify_all()

    def _run_right_ee_callback(self, message: PoseStamped) -> None:
        with self._run_pose_condition:
            self._run_right_ee_pose = message
            self._run_right_ee_sequence += 1
            self._run_pose_condition.notify_all()

    def _wait_for_run_poses(
        self,
        goal_handle,
        *,
        need_left: bool,
        need_right: bool,
        require_fresh: bool,
    ) -> tuple[Optional[PoseStamped], Optional[PoseStamped]]:
        with self._run_pose_condition:
            left_before = self._run_left_ee_sequence
            right_before = self._run_right_ee_sequence
        deadline = time.monotonic() + self._float("ee_pose_timeout_sec")
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for legacy run EE poses")
            with self._run_pose_condition:
                left_ready = not need_left or (
                    self._run_left_ee_pose is not None
                    and (
                        not require_fresh
                        or self._run_left_ee_sequence > left_before
                    )
                )
                right_ready = not need_right or (
                    self._run_right_ee_pose is not None
                    and (
                        not require_fresh
                        or self._run_right_ee_sequence > right_before
                    )
                )
                if left_ready and right_ready:
                    return self._run_left_ee_pose, self._run_right_ee_pose
            time.sleep(0.02)
        requested = (
            "left/right"
            if need_left and need_right
            else "left" if need_left else "right"
        )
        freshness = "fresh " if require_fresh else ""
        raise MissionError(
            f"no {freshness}{requested} legacy run EE pose before timeout"
        )
