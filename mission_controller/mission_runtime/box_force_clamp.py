"""Native RealMan SDK Tool-Y force clamping for TF box workflows."""

from __future__ import annotations

import math
import time
from typing import Optional

from geometry_msgs.msg import Pose

from .common import MissionCanceled, MissionError
from .realman_sdk_adapter import RealManSdkCanceled, RealManSdkError


# Bound to the composed compatibility facade by box_support.py.
BoxSupportMixin = None


class BoxForceClampMixin:
    """Apply mirrored Tool-Y force control and preserve later TF targets."""

    def _force_carry_clamp_target_override(
        self, *, force_carry_profile: bool, drag_mode: bool, label: str
    ) -> Optional[dict[str, float]]:
        """Read the dedicated profile targets when its bilateral clamp starts."""
        if not force_carry_profile:
            return None
        if not drag_mode:
            return {
                arm: self._float(f"grasp_box_tf_force_carry_target_force_{arm}_n")
                for arm in ("left", "right")
            }
        if label != "step1_left":
            return None
        return {
            arm: self._float(
                f"drag_box_tf_force_carry_post_drag3_target_force_{arm}_n"
            )
            for arm in ("left", "right")
        }

    @staticmethod
    def _force_clamp_parameter_prefix(*, tf_mode: bool, drag_mode: bool):
        if not tf_mode:
            return None
        return "drag_box_tf_force_clamp" if drag_mode else "grasp_box_tf_force_clamp"

    def _force_clamp_mode(self, parameter_prefix: Optional[str]) -> str:
        if not parameter_prefix:
            return "disabled"
        mode = self._string(f"{parameter_prefix}_mode").strip().lower()
        if mode not in ("disabled", "closed_loop"):
            raise MissionError(
                f"{parameter_prefix}_mode must be disabled or closed_loop"
            )
        return mode

    @staticmethod
    def _new_force_clamp_session(parameter_prefix: str) -> dict:
        return {
            "prefix": parameter_prefix,
            "travelled_m": {"left": 0.0, "right": 0.0},
        }

    def _force_clamp_pose_snapshot(self, arm: str):
        with self.joint_state_lock:
            values = getattr(self, "latest_slave_arm_poses", {}).get(arm)
            sequence = getattr(self, "latest_slave_arm_pose_sequences", {}).get(
                arm, 0
            )
            stamp = getattr(self, "latest_slave_arm_pose_times", {}).get(arm, 0.0)
        age = time.monotonic() - stamp if stamp > 0.0 else math.inf
        if values is None:
            return None, sequence, age
        return BoxSupportMixin._endpoint_sync_transform_to_pose(
            BoxSupportMixin._endpoint_sync_pose_values_to_transform(values)
        ), sequence, age

    def _force_clamp_sdk_tool_y_stream(
        self,
        goal_handle,
        adapter,
        session: dict,
        current_poses: dict[str, Pose],
        *,
        initial_drag_right_contact: bool = False,
        target_force_override_n: Optional[dict[str, float]] = None,
    ) -> tuple[dict[str, Pose], str]:
        """Run SDK Tool-Y force-position control and read final Link7 poses."""
        prefix = session["prefix"]
        arms = tuple(arm for arm in ("left", "right") if arm in current_poses)
        target_force_n = {
            arm: (
                float(target_force_override_n[arm])
                if target_force_override_n is not None and arm in target_force_override_n
                else self._float(f"{prefix}_sdk_target_force_{arm}_n")
            )
            for arm in arms
        }
        self._publish_box_grasp_feedback(
            goal_handle,
            "SDK_TOOL_Y_FORCE_CLAMP",
            "starting incremental SDK Tool-Y force-position control: "
            + ", ".join(
                f"{arm}_target_delta={target_force_n[arm]:.3f}N"
                for arm in arms
            )
            + f"; speed={self._float(f'{prefix}_sdk_speed_mm_s'):.3f}mm/s; "
            "waiting for simultaneous stable per-arm Tool-Y baselines: "
            f"window={self._float(f'{prefix}_sdk_baseline_stability_window_sec'):.3f}s, "
            f"max_span={self._float(f'{prefix}_sdk_baseline_stability_max_span_n'):.3f}N, "
            f"timeout={self._float(f'{prefix}_sdk_baseline_stability_timeout_sec'):.3f}s; "
            + (
                "initial right stops at first threshold crossing, then "
                "must hold at least "
                f"{self._float('drag_box_tf_force_clamp_initial_right_post_stop_min_delta_n'):.3f}N "
                "for "
                f"{self._float('drag_box_tf_force_clamp_initial_right_post_stop_confirm_sec'):.3f}s "
                "while stationary; at most "
                f"{self._integer('drag_box_tf_force_clamp_initial_right_max_attempts')} "
                "attempts"
                if initial_drag_right_contact
                else "each arm requires 5 consecutive contact samples (reset on any miss); each confirmed arm stops inward motion; lift proceeds once both arms have confirmed; no post-stop force hold required"
            ),
        )
        previous_sequences = {
            arm: self.latest_slave_arm_pose_sequences.get(arm, 0) for arm in arms
        }
        retry_options = {}
        clamp_executor = adapter.execute_tool_y_force_clamp
        if initial_drag_right_contact:
            clamp_executor = adapter.execute_tool_y_force_clamp_retry_initial_right
            retry_options = {
                "max_attempts": self._integer(
                    "drag_box_tf_force_clamp_initial_right_max_attempts"
                ),
                "on_retry": lambda attempt, maximum, reason: (
                    self._publish_box_grasp_feedback(
                        goal_handle,
                        "SDK_TOOL_Y_FORCE_CLAMP_RETRY",
                        f"initial right contact retry {attempt}/{maximum}; {reason}",
                    )
                ),
            }
        try:
            result = clamp_executor(
                arms,
                target_force_n,
                speed_mm_s=self._float(f"{prefix}_sdk_speed_mm_s"),
                max_travel_m={
                    arm: self._float(f"{prefix}_sdk_max_travel_{arm}_m")
                    for arm in arms
                },
                timeout_sec=self._float(f"{prefix}_timeout_sec"),
                control_period_sec=self._float(
                    f"{prefix}_sdk_control_period_sec"
                ),
                baseline_stability_window_sec=self._float(
                    f"{prefix}_sdk_baseline_stability_window_sec"
                ),
                baseline_stability_max_span_n=self._float(
                    f"{prefix}_sdk_baseline_stability_max_span_n"
                ),
                baseline_stability_timeout_sec=self._float(
                    f"{prefix}_sdk_baseline_stability_timeout_sec"
                ),
                post_stop_confirmation_sec=(
                    self._float(
                        "drag_box_tf_force_clamp_initial_right_post_stop_confirm_sec"
                    )
                    if initial_drag_right_contact
                    else 0.0
                ),
                post_stop_min_force_n=(
                    self._float(
                        "drag_box_tf_force_clamp_initial_right_post_stop_min_delta_n"
                    )
                    if initial_drag_right_contact
                    else 0.0
                ),
                contact_consecutive_samples=(
                    self._integer(
                        "drag_box_tf_force_clamp_initial_right_contact_consecutive_samples"
                    )
                    if initial_drag_right_contact
                    else 5
                ),
                dual_post_stop_confirmation_sec=0.0,
                contact_min_duration_sec=0.0,
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                **retry_options,
            )
        except RealManSdkCanceled as exc:
            raise MissionCanceled(str(exc)) from exc
        except (RealManSdkError, ValueError) as exc:
            raise MissionError(f"{prefix} SDK Tool-Y force clamp failed: {exc}") from exc

        for arm, distance in result["travel_m"].items():
            session["travelled_m"][arm] += float(distance)
        updated = dict(current_poses)
        deadline = time.monotonic() + self._float(f"{prefix}_motion_timeout_sec")
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while waiting for SDK Tool-Y force-clamp feedback"
            )
            all_fresh = True
            for arm in arms:
                pose, sequence, age = self._force_clamp_pose_snapshot(arm)
                if (
                    pose is None
                    or sequence <= previous_sequences[arm]
                    or age > self._float(f"{prefix}_sensor_max_age_sec")
                ):
                    all_fresh = False
                    continue
                updated[arm] = pose
            if all_fresh:
                break
            time.sleep(0.01)
        else:
            raise MissionError(
                f"{prefix} did not receive fresh Link7 pose after SDK Tool-Y clamp"
            )

        final = result["final_tool_wrench"]
        initial = result["initial_tool_wrench"]
        force_delta = result["force_delta_n"]
        contact_delta = result["contact_force_delta_n"]
        return updated, (
            "incremental SDK Tool-Y force clamp confirmed; "
            + ("bilateral_contact=confirmed_5_consecutive_samples_per_arm; post_stop_force_hold=disabled; " if len(arms) == 2 else "")
            + ", ".join(
                f"{arm}_Fy={float(final[arm][1]):.3f}N,"
                f"baseline_Fy={float(initial[arm][1]):.3f}N,"
                f"delta_Fy={float(force_delta[arm]):+.3f}N,"
                f"trigger_delta_Fy={float(contact_delta[arm]):+.3f}N,"
                f"target_delta={float(result['target_force_n'][arm]):+.3f}N,"
                f"sdk_target_Fy="
                f"{float(result['controller_target_force_n'][arm]):+.3f}N,"
                f"travel={float(result['travel_m'][arm]):.4f}m"
                for arm in arms
            )
            + "; baseline_stability="
            + ",".join(
                f"{arm}:{float(result['baseline_stability_spans_n'][arm]):.3f}N"
                for arm in arms
            )
            + f" over {float(result['baseline_stability_elapsed_sec']):.3f}s"
            + f" ({int(result['baseline_stability_samples'])} samples); "
            + (
                "post_stop_confirm_delta_Fy="
                f"{float(result['post_stop_confirm_delta_n']):+.3f}N; "
                if result["post_stop_confirm_delta_n"] is not None
                else ""
            )
            + (
                f"contact_attempts={int(result['contact_attempts'])}; "
                if "contact_attempts" in result
                else ""
            )
            + f"elapsed={float(result['elapsed_sec']):.3f}s; "
            "orientation=locked_by_sdk_fk"
        )

    def _rebase_post_movel_targets_after_force_clamp(
        self,
        targets,
        next_target_index: int,
        actual_by_arm: dict[str, Pose],
        nominal_by_arm: dict[str, Pose],
    ) -> None:
        """Apply actual-vs-nominal clamp transforms to all later targets."""
        corrections = {}
        for arm, actual in actual_by_arm.items():
            nominal = nominal_by_arm.get(arm)
            if nominal is None:
                continue
            corrections[arm] = BoxSupportMixin._compose_transform(
                BoxSupportMixin._endpoint_sync_pose_values_to_transform(
                    (
                        actual.position.x,
                        actual.position.y,
                        actual.position.z,
                        actual.orientation.x,
                        actual.orientation.y,
                        actual.orientation.z,
                        actual.orientation.w,
                    )
                ),
                BoxSupportMixin._inverse_transform(
                    BoxSupportMixin._endpoint_sync_pose_values_to_transform(
                        (
                            nominal.position.x,
                            nominal.position.y,
                            nominal.position.z,
                            nominal.orientation.x,
                            nominal.orientation.y,
                            nominal.orientation.z,
                            nominal.orientation.w,
                        )
                    )
                ),
            )
        if not corrections:
            return
        for index in range(next_target_index, len(targets)):
            label, left, right = targets[index]
            updated = {"left": left, "right": right}
            for arm, correction in corrections.items():
                pose = updated[arm]
                transform = BoxSupportMixin._compose_transform(
                    correction,
                    BoxSupportMixin._endpoint_sync_pose_values_to_transform(
                        (
                            pose.position.x,
                            pose.position.y,
                            pose.position.z,
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        )
                    ),
                )
                updated[arm] = BoxSupportMixin._endpoint_sync_transform_to_pose(
                    transform
                )
            targets[index] = (label, updated["left"], updated["right"])
