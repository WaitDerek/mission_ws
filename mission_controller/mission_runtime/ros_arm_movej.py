"""Blocking arm MoveJ over the robot's ROS StringCmd service.

The service acknowledges receipt, not completion.  Require fresh measured
joint position and velocity feedback before a caller may start its next stage.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rm_robot_interfaces.msg import ArmSlaveData
from rm_robot_interfaces.srv import StringCmd

from .realman_sdk_common import RealManSdkCanceled, RealManSdkError


class RosArmMoveJ:
    """Send arm MoveJ to /robot/command and verify both selected arms."""

    def __init__(
        self,
        node,
        *,
        service_name: str = "/robot/command",
        left_feedback_topic: str = "/mcap/slave_arm_left",
        right_feedback_topic: str = "/mcap/slave_arm_right",
        callback_group=None,
        client=None,
        stop_arm: Callable[[str], None] | None = None,
        stop_all: Callable[[], None] | None = None,
        position_tolerance_rad: float = 0.02,
        velocity_tolerance_rad_sec: float = 0.05,
        feedback_max_age_sec: float = 1.0,
        stable_samples: int = 3,
    ) -> None:
        self._client = client or node.create_client(
            StringCmd, service_name, callback_group=callback_group
        )
        self._service_name = service_name
        self._stop_arm = stop_arm
        self._stop_all = stop_all
        self._position_tolerance = float(position_tolerance_rad)
        self._velocity_tolerance = float(velocity_tolerance_rad_sec)
        self._feedback_max_age = float(feedback_max_age_sec)
        self._stable_samples = int(stable_samples)
        self._lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._feedback = {
            "left": (0, 0.0, [], []),
            "right": (0, 0.0, [], []),
        }
        feedback_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscriptions = [
            node.create_subscription(
                ArmSlaveData,
                topic,
                lambda message, side=arm: self._on_feedback(side, message),
                feedback_qos,
                callback_group=callback_group,
            )
            for arm, topic in (
                ("left", left_feedback_topic),
                ("right", right_feedback_topic),
            )
        ]

    def _on_feedback(self, arm: str, message: ArmSlaveData) -> None:
        state = message.joint_state
        names = list(state.name)
        positions = list(state.position)
        velocities = list(state.velocity)
        if len(names) != len(positions) or len(names) != len(velocities):
            return
        expected = [f"joint{index}" for index in range(1, 8)]
        if all(name in names for name in expected):
            indices = [names.index(name) for name in expected]
        elif len(positions) >= 7:
            indices = list(range(7))
        else:
            return
        joints = [float(positions[index]) for index in indices]
        speeds = [float(velocities[index]) for index in indices]
        if not all(math.isfinite(value) for value in joints + speeds):
            return
        with self._lock:
            sequence = self._feedback[arm][0] + 1
            self._feedback[arm] = (sequence, time.monotonic(), joints, speeds)

    def current_joint_positions(
        self, arm: str, *, timeout_sec: float = 5.0, cancel_requested=None
    ) -> list[float]:
        """Return fresh, stationary measured joints in radians; never command motion."""
        if arm not in self._feedback:
            raise RealManSdkError(f"unknown feedback arm: {arm}")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self._check_cancel(cancel_requested, "joint snapshot")
            with self._lock:
                sequence, stamp, joints, speeds = self._feedback[arm]
                joints, speeds = list(joints), list(speeds)
            if (sequence > 0 and len(joints) == 7 and len(speeds) == 7
                    and time.monotonic() - stamp <= self._feedback_max_age
                    and max(abs(v) for v in speeds) <= self._velocity_tolerance):
                return joints
            time.sleep(0.02)
        raise RealManSdkError(f"no fresh stationary {arm}-arm joint feedback")

    def _stop(self, arms: Sequence[str]) -> None:
        try:
            if len(arms) == 2 and self._stop_all is not None:
                self._stop_all()
            elif self._stop_arm is not None:
                for arm in arms:
                    self._stop_arm(arm)
        except Exception:  # noqa: BLE001
            # Preserve the original service/feedback failure for the caller.
            pass

    @staticmethod
    def _check_cancel(cancel_requested, stage: str) -> None:
        if cancel_requested is not None and cancel_requested():
            raise RealManSdkCanceled(f"mission canceled during ROS MoveJ {stage}")

    def execute(
        self,
        targets_deg: Mapping[str, Sequence[float]],
        speed_percent: int,
        blend_radius: int,
        trajectory_connect: int,
        *,
        cancel_requested=None,
        timeout_sec: float,
    ) -> str:
        arms = tuple(targets_deg)
        if not arms or any(arm not in ("left", "right") for arm in arms):
            raise RealManSdkError("ROS MoveJ requires one or both named arms")
        targets = {
            arm: [float(value) for value in targets_deg[arm]] for arm in arms
        }
        if any(len(values) != 7 or not all(math.isfinite(v) for v in values)
               for values in targets.values()):
            raise RealManSdkError("ROS MoveJ requires seven finite degrees per arm")
        deadline = time.monotonic() + float(timeout_sec)
        with self._command_lock:
            while not self._client.service_is_ready():
                self._check_cancel(cancel_requested, "service wait")
                if time.monotonic() >= deadline:
                    raise RealManSdkError(
                        f"ROS MoveJ service {self._service_name} unavailable"
                    )
                self._client.wait_for_service(timeout_sec=0.1)

            with self._lock:
                prior = {arm: self._feedback[arm][0] for arm in arms}
            target_rad = {
                arm: [math.radians(value) for value in values]
                for arm, values in targets.items()
            }
            futures = {}
            try:
                for arm in arms:
                    self._check_cancel(cancel_requested, "command submission")
                    request = StringCmd.Request()
                    request.data = json.dumps(
                        {
                            "device": 0 if arm == "left" else 1,
                            "payload": {
                                "command": "movej",
                                "joint": [int(round(value * 1000.0))
                                          for value in targets[arm]],
                                "v": int(speed_percent),
                                "r": int(blend_radius),
                                "trajectory_connect": int(trajectory_connect),
                            },
                        },
                        separators=(",", ":"),
                    ) + "\r\n"
                    futures[arm] = self._client.call_async(request)

                for arm, future in futures.items():
                    while not future.done():
                        self._check_cancel(cancel_requested, "service response")
                        if time.monotonic() >= deadline:
                            raise RealManSdkError(
                                f"ROS MoveJ {arm} service response timed out"
                            )
                        time.sleep(0.02)
                    response = future.result()
                    try:
                        data = json.loads(str(response.data).strip())
                    except (AttributeError, TypeError, ValueError) as exc:
                        raise RealManSdkError(
                            f"ROS MoveJ {arm} invalid service response: {response!r}"
                        ) from exc
                    if not data.get("receive_state", False):
                        raise RealManSdkError(
                            f"ROS MoveJ {arm} rejected: {response.data}"
                        )

                stable = 0
                last_sequences = {arm: prior[arm] for arm in arms}
                last_detail = "no fresh arm feedback"
                while time.monotonic() < deadline:
                    self._check_cancel(cancel_requested, "feedback wait")
                    with self._lock:
                        feedback = {arm: self._feedback[arm] for arm in arms}
                    if any(feedback[arm][0] <= last_sequences[arm] for arm in arms):
                        time.sleep(0.02)
                        continue
                    last_sequences = {arm: feedback[arm][0] for arm in arms}
                    matches = True
                    details = []
                    for arm in arms:
                        seq, stamp, joints, speeds = feedback[arm]
                        age = time.monotonic() - stamp
                        if seq <= prior[arm] or len(joints) != 7 or age > self._feedback_max_age:
                            matches = False
                            details.append(f"{arm}=missing_or_stale(seq={seq},age={age:.3f})")
                            continue
                        error = max(abs(actual - target)
                                    for actual, target in zip(joints, target_rad[arm]))
                        velocity = max(abs(value) for value in speeds)
                        details.append(f"{arm}=position_error={error:.4f}rad,velocity={velocity:.4f}rad/s")
                        if error > self._position_tolerance or velocity > self._velocity_tolerance:
                            matches = False
                    last_detail = "; ".join(details)
                    stable = stable + 1 if matches else 0
                    if stable >= self._stable_samples:
                        return (
                            "ROS /robot/command MoveJ completed from fresh feedback: "
                            + last_detail
                        )
                    time.sleep(0.02)
                raise RealManSdkError(
                    f"ROS MoveJ targets not reached before {timeout_sec:.1f}s: {last_detail}"
                )
            except Exception:
                if futures:
                    self._stop(arms)
                raise
