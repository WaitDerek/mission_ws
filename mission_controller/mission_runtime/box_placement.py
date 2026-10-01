"""Dynamic table placement, support sensing, and release operations.

The methods remain available through BoxCarryMixin for existing callers.
"""

import json
import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass
import time

from rm_robot_interfaces.srv import StringCmd

from .common import MissionCanceled, MissionError, quaternion_multiply, rotate_vector
from .place_descent_z_guard import PlaceDescentZGuardMixin, work_z
from .realman_sdk_adapter import (
    RealManSdkCanceled,
    RealManSdkError,
    pose_to_sdk_target,
    quaternion_to_rpy,
)


# Bound to the composed compatibility facade by box_support.py.
BoxSupportMixin = None


@dataclass
class _PlaceDescentReference:
    """Action-local, post-waist TCP origins and fixed Work-frame transforms."""

    tcp_by_arm: dict
    base_by_arm: dict
    y_offset_m: float = 0.0
    allow_y_change: bool = True
    commanded_drop_m: float = 0.0
    maximum_drop_m: float = 0.0


class BoxPlacementMixin(PlaceDescentZGuardMixin):
    """Plan and execute test placement while preserving the rigid box grasp."""

    def _place_box_test_common_wrench_sample(self, adapter, base_frame):
        """Express both complete SDK Work wrenches on the common base axes.

        Torque is only rotated: the SDK reports torque about its sensor, not
        about the base_link origin. No sensor zeroing or movement occurs here.
        """
        work = adapter.read_bilateral_work_wrench()
        result = {}
        for arm in ("left", "right"):
            arm_base = self._lookup_tf_carry_transform(
                base_frame, self._string(f"{arm}_arm_base_frame").strip().lstrip("/")
            )
            wrench = work[arm]
            force = rotate_vector(tuple(wrench[:3]), arm_base[1])
            torque = rotate_vector(tuple(wrench[3:]), arm_base[1])
            result[arm] = force + torque
        return result

    def _place_box_test_stable_common_wrench(self, goal_handle, adapter, base_frame):
        """Capture a still, eight-sample full-wrench reference or fail closed."""
        threshold = self._float("place_box_test_table_support_delta_fz_n")
        timeout = self._float("place_box_test_force_unload_baseline_timeout_sec")
        deadline = time.monotonic() + timeout
        samples = deque(maxlen=8)
        spans = None
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while sampling placement force")
            samples.append(self._place_box_test_common_wrench_sample(adapter, base_frame))
            if len(samples) == samples.maxlen:
                spans = {
                    arm: max(sample[arm][2] for sample in samples)
                    - min(sample[arm][2] for sample in samples)
                    for arm in ("left", "right")
                }
                if all(span <= threshold * 0.5 for span in spans.values()):
                    return {
                        arm: tuple(statistics.median(sample[arm][axis] for sample in samples)
                                   for axis in range(6))
                        for arm in ("left", "right")
                    }
            time.sleep(0.05)
        raise MissionError(
            "common-frame placement wrench did not stabilize: "
            f"z_spans={spans}; limit={threshold * 0.5:.3f}N; "
            f"timeout={timeout:.1f}s; no placement motion started"
        )

    def _place_box_test_waist_contact_monitor(self, adapter, base_frame, baseline, stop, state):
        """Stop an unfinished waist bend on sustained, substantial load transfer."""
        threshold = max(10.0, 3.0 * self._float("place_box_test_table_support_delta_fz_n"))
        signs = {
            arm: self._float(f"place_box_test_table_support_sign_{arm}")
            for arm in ("left", "right")
        }
        consecutive = 0
        while not stop.is_set():
            try:
                current = self._place_box_test_common_wrench_sample(adapter, base_frame)
                delta = {
                    arm: signs[arm] * (current[arm][2] - baseline[arm][2])
                    for arm in ("left", "right")
                }
                # A bend can redistribute weight between wrists without the
                # table taking any weight. Require a bilateral *total* change
                # as well as a change on at least one individual wrist.
                candidate = (
                    any(v >= threshold for v in delta.values())
                    and sum(delta.values()) >= threshold
                )
                consecutive = consecutive + 1 if candidate else 0
                if consecutive >= 3:
                    state["contact"] = delta
                    self._place_box_test_stop_body()
                    return
            except Exception as exc:  # noqa: BLE001
                state["error"] = exc
                self._place_box_test_stop_body()
                return
            stop.wait(0.1)

    def _place_box_test_body_request(
        self,
        body_units,
        *,
        trajectory_connect,
        blend_radius,
    ):
        trajectory_connect = int(trajectory_connect)
        blend_radius = int(blend_radius)
        if trajectory_connect not in (0, 1):
            raise MissionError("place_box_test trajectory_connect must be 0 or 1")
        if not 0 <= blend_radius <= 100:
            raise MissionError("place_box_test body blend radius must be in [0,100]")
        device = self._integer("box_joint1_device")
        payload = {
            "device": device,
            "payload": {
                "command": "movej",
                "device": device,
                "joint": [int(value) for value in body_units],
                "v": self._integer("place_box_test_body_velocity"),
                "r": blend_radius,
                "trajectory_connect": trajectory_connect,
            },
        }
        request = StringCmd.Request()
        request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
        return request

    def _place_box_test_stop_body(self):
        if not self._boolean("place_box_test_body_stop_enabled"):
            return
        try:
            if not self.body_command_client.service_is_ready():
                self.body_command_client.wait_for_service(timeout_sec=0.5)
            if not self.body_command_client.service_is_ready():
                return
            device = self._integer("box_joint1_device")
            payload = {
                "device": device,
                "payload": {
                    "command": self._string("place_box_test_body_stop_command"),
                    "device": device,
                },
            }
            request = StringCmd.Request()
            request.data = json.dumps(payload, separators=(",", ":")) + "\r\n"
            self.body_command_client.call_async(request)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"place_box_test body stop failed: {exc}")

    def _wait_for_place_box_test_world_targets(self, goal_handle, targets_by_arm):
        timeout_sec = self._float("place_box_test_timeout_sec")
        position_limit = self._float("place_box_test_position_tolerance_m")
        orientation_limit = self._float("place_box_test_orientation_tolerance_rad")
        required_stable = self._integer("place_box_test_stable_samples")
        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        deadline = time.monotonic() + timeout_sec
        stable_samples = 0
        latest_detail = "no TF sample"
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while verifying place_box_test targets")
            matches = True
            details = []
            for arm in ("left", "right"):
                actual = self._lookup_tf_carry_transform(
                    base_frame,
                    self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                )
                target = targets_by_arm[arm]
                position_error = self._endpoint_sync_pose_position_error(
                    actual[0], target[0]
                )
                orientation_error = self._endpoint_sync_pose_orientation_error(
                    actual[1], target[1]
                )
                details.append(
                    f"{arm}_position_error={position_error:.4f}m,"
                    f"{arm}_orientation_error={orientation_error:.4f}rad"
                )
                if (
                    position_error > position_limit
                    or orientation_error > orientation_limit
                ):
                    matches = False
            latest_detail = "; ".join(details)
            stable_samples = stable_samples + 1 if matches else 0
            if stable_samples >= required_stable:
                return latest_detail
            time.sleep(0.02)
        raise MissionError(
            "place_box_test Link7 targets were not reached: "
            f"{latest_detail}, timeout_sec={timeout_sec:.1f}"
        )

    @staticmethod
    def _place_box_test_fz_unload_metric(
        current_fz: float,
        baseline_fz: float,
        force_sign: float,
    ) -> float:
        """Return the table-support signal from force-Z only.

        With the current wrists, a carried box makes Fz more negative.  When
        the table takes the load Fz moves back toward zero, so the calibrated
        default sign is +1.  Fx is intentionally not part of the placement
        release decision; Joint2=60 deg removes the lateral clamp afterwards.
        """
        return float(force_sign) * (float(current_fz) - float(baseline_fz))

    def _place_box_test_fz_snapshot(self, arm: str):
        with self.joint_state_lock:
            wrench = getattr(self, "latest_slave_arm_wrenches", {}).get(arm)
            sequence = getattr(
                self, "latest_slave_arm_wrench_sequences", {}
            ).get(arm, 0)
            stamp = getattr(self, "latest_slave_arm_wrench_times", {}).get(
                arm, 0.0
            )
        age = time.monotonic() - stamp if stamp > 0.0 else math.inf
        if wrench is None or len(wrench) < 3:
            return None, sequence, age
        fz = float(wrench[2])
        if not math.isfinite(fz):
            return None, sequence, age
        return fz, sequence, age

    def _collect_place_box_test_fz_baseline(self, goal_handle):
        """Capture a fresh, loaded-box Fz baseline for both wrists."""
        duration = self._float("place_box_test_force_unload_baseline_duration_sec")
        minimum = self._integer("place_box_test_force_unload_baseline_min_samples")
        timeout = self._float("place_box_test_force_unload_baseline_timeout_sec")
        max_age = self._float("place_box_test_force_unload_sensor_max_age_sec")
        samples = {"left": [], "right": []}
        sequences = {"left": -1, "right": -1}
        started_at = time.monotonic()
        deadline = started_at + timeout
        latest_detail = "no fresh Fz samples"
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while collecting place_box_test force-Z baseline"
            )
            for arm in ("left", "right"):
                fz, sequence, age = self._place_box_test_fz_snapshot(arm)
                if fz is None or age > max_age or sequence <= sequences[arm]:
                    latest_detail = (
                        f"{arm}=missing/stale Fz, sequence={sequence}, age={age:.3f}s"
                    )
                    continue
                samples[arm].append(fz)
                sequences[arm] = sequence
            if (
                time.monotonic() - started_at >= duration
                and all(len(samples[arm]) >= minimum for arm in ("left", "right"))
            ):
                baselines = {
                    arm: float(statistics.median(samples[arm]))
                    for arm in ("left", "right")
                }
                return baselines, sequences
            time.sleep(0.01)
        raise MissionError(
            "place_box_test force-Z baseline collection timed out: "
            f"left_samples={len(samples['left'])}, "
            f"right_samples={len(samples['right'])}; {latest_detail}; "
            f"timeout_sec={timeout:.1f}"
        )

    def _new_place_box_test_fz_unload_monitor(
        self,
        baselines,
        baseline_sequences,
    ) -> dict:
        filter_samples = self._integer("place_box_test_force_unload_filter_samples")
        return {
            "baselines": dict(baselines),
            "sequences": dict(baseline_sequences),
            "windows": {
                arm: deque(maxlen=filter_samples) for arm in ("left", "right")
            },
            "confirmed_since": None,
            "confirmed": False,
            "detail": "force_z_table_support=pending",
            "latest_metrics": {"left": 0.0, "right": 0.0},
            "latest_fz": dict(baselines),
        }

    def _update_place_box_test_fz_unload_monitor(self, monitor: dict) -> bool:
        """Update the in-motion Fz monitor and report bilateral support."""
        max_age = self._float("place_box_test_force_unload_sensor_max_age_sec")
        required_duration = self._float(
            "place_box_test_force_unload_required_duration_sec"
        )
        thresholds = {
            arm: self._float(
                f"place_box_test_force_unload_threshold_{arm}_counts"
            )
            for arm in ("left", "right")
        }
        signs = {
            arm: self._float(f"place_box_test_force_unload_sign_{arm}")
            for arm in ("left", "right")
        }
        fresh = True
        for arm in ("left", "right"):
            fz, sequence, age = self._place_box_test_fz_snapshot(arm)
            if fz is None or age > max_age:
                fresh = False
                continue
            if sequence > monitor["sequences"][arm]:
                monitor["windows"][arm].append(fz)
                monitor["sequences"][arm] = sequence
            if not monitor["windows"][arm]:
                fresh = False
                continue
            monitor["latest_fz"][arm] = float(
                statistics.median(monitor["windows"][arm])
            )
            monitor["latest_metrics"][arm] = (
                self._place_box_test_fz_unload_metric(
                    monitor["latest_fz"][arm],
                    monitor["baselines"][arm],
                    signs[arm],
                )
            )
        supported = fresh and all(
            monitor["latest_metrics"][arm] >= thresholds[arm]
            for arm in ("left", "right")
        )
        if not supported:
            monitor["confirmed_since"] = None
            return False
        monitor["confirmed_since"] = monitor["confirmed_since"] or time.monotonic()
        if time.monotonic() - monitor["confirmed_since"] < required_duration:
            return False
        monitor["confirmed"] = True
        monitor["detail"] = (
            "force_z_table_support=confirmed_in_motion; "
            f"left_baseline_Fz={monitor['baselines']['left']:.1f},"
            f"left_Fz={monitor['latest_fz']['left']:.1f},"
            f"left_unload={monitor['latest_metrics']['left']:.1f}; "
            f"right_baseline_Fz={monitor['baselines']['right']:.1f},"
            f"right_Fz={monitor['latest_fz']['right']:.1f},"
            f"right_unload={monitor['latest_metrics']['right']:.1f}; "
            f"stable_sec={required_duration:.2f}; force_x=ignored"
        )
        return True

    def _reset_place_box_test_fz_unload_monitor(self, monitor: dict) -> None:
        """Require fresh bilateral Fz support after a corrective motion."""
        filter_samples = self._integer("place_box_test_force_unload_filter_samples")
        monitor["windows"] = {
            arm: deque(maxlen=filter_samples) for arm in ("left", "right")
        }
        monitor["confirmed_since"] = None
        monitor["confirmed"] = False
        monitor["detail"] = "force_z_table_support=reconfirming_after_z_equalization"
        for arm in ("left", "right"):
            _fz, sequence, _age = self._place_box_test_fz_snapshot(arm)
            monitor["sequences"][arm] = sequence

    def _execute_place_box_test_post_support_z_equalization(
        self,
        goal_handle,
        adapter,
        base_frame: str,
        force_monitor: dict,
    ):
        """Equalize physical Link7 heights after table support is detected.

        Z is compared and corrected in ``base_frame`` (normally base_link),
        never in the two rotated arm-base frames.  Each arm keeps its current
        world X/Y and orientation.  World Z starts from the arithmetic mean,
        then is lowered as needed so the initially higher arm descends by at
        least the configured safety distance before Joint2 opens outward.
        """
        live_arm_base = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        actual_link = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_link8_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        left_z = float(actual_link["left"][0][2])
        right_z = float(actual_link["right"][0][2])
        average_z = 0.5 * (left_z + right_z)
        highest_z = max(left_z, right_z)
        minimum_high_arm_downward = self._float(
            "place_box_test_post_support_z_equalization_min_high_arm_downward_m"
        )
        target_z = min(average_z, highest_z - minimum_high_arm_downward)
        corrections = {
            "left": target_z - left_z,
            "right": target_z - right_z,
        }
        max_correction = self._float(
            "place_box_test_post_support_z_equalization_max_correction_m"
        )
        if any(abs(value) > max_correction for value in corrections.values()):
            raise MissionError(
                "place_box_test post-support Link7 Z equalization exceeds the "
                f"safety limit: left_delta={corrections['left']:.4f}m, "
                f"right_delta={corrections['right']:.4f}m, "
                f"max_correction_m={max_correction:.4f}"
            )

        world_targets = {}
        local_target_poses = {}
        for arm in ("left", "right"):
            position, orientation = actual_link[arm]
            world_targets[arm] = (
                (float(position[0]), float(position[1]), target_z),
                BoxSupportMixin._normalize_quaternion(orientation),
            )
            local_target = BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(live_arm_base[arm]),
                world_targets[arm],
            )
            local_target_poses[arm] = self._endpoint_sync_transform_to_pose(
                local_target
            )

        detail = (
            f"common_frame={base_frame}; "
            "reference=average_with_min_high_arm_downward; "
            f"left_z={left_z:.4f}m; right_z={right_z:.4f}m; "
            f"average_z={average_z:.4f}m; "
            f"min_high_arm_downward={minimum_high_arm_downward:.4f}m; "
            f"target_z={target_z:.4f}m; "
            f"left_delta={corrections['left']:.4f}m; "
            f"right_delta={corrections['right']:.4f}m"
        )
        timeout = self._float(
            "place_box_test_post_support_z_equalization_timeout_sec"
        )
        tolerance = self._float(
            "place_box_test_post_support_z_equalization_tolerance_m"
        )
        if max(abs(value) for value in corrections.values()) <= tolerance:
            motion_detail = (
                "post_support_z_equalization_motion=skipped_already_equal"
            )
            self._publish_place_box_test_feedback(
                goal_handle,
                "POST_SUPPORT_LINK7_Z_ALREADY_EQUAL",
                detail + f"; z_tolerance={tolerance:.4f}m; no arm motion required",
            )
        else:
            self._publish_place_box_test_feedback(
                goal_handle,
                "EQUALIZING_POST_SUPPORT_LINK7_Z",
                detail + "; executing low-speed dual-arm SDK MoveL",
            )
            speed = self._float(
                "place_box_test_post_support_z_equalization_velocity_percent"
            )
            adapter.execute_dual_movel_endpoint(
                pose_to_sdk_target(local_target_poses["left"]),
                pose_to_sdk_target(local_target_poses["right"]),
                speed,
                speed,
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=timeout,
                motion_mode="movel",
            )
            required_stable = self._integer("place_box_test_stable_samples")
            deadline = time.monotonic() + timeout
            stable_samples = 0
            latest_error = math.inf
            while time.monotonic() < deadline:
                self._check_canceled(
                    goal_handle, "while verifying post-support Link7 Z equalization"
                )
                actual_link = {
                    arm: self._lookup_tf_carry_transform(
                        base_frame,
                        self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                    )
                    for arm in ("left", "right")
                }
                latest_error = abs(
                    float(actual_link["left"][0][2])
                    - float(actual_link["right"][0][2])
                )
                stable_samples = (
                    stable_samples + 1 if latest_error <= tolerance else 0
                )
                if stable_samples >= required_stable:
                    break
                time.sleep(0.02)
            else:
                raise MissionError(
                    "place_box_test post-support Link7 Z equalization was not "
                    f"reached: z_error={latest_error:.4f}m, "
                    f"tolerance={tolerance:.4f}m, timeout_sec={timeout:.1f}"
                )
            motion_detail = "post_support_z_equalization_motion=completed"

        self._reset_place_box_test_fz_unload_monitor(force_monitor)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_canceled(
                goal_handle, "while reconfirming table support after Z equalization"
            )
            if self._update_place_box_test_fz_unload_monitor(force_monitor):
                break
            time.sleep(0.02)
        else:
            raise MissionError(
                "place_box_test lost bilateral force-Z table support after Link7 "
                f"Z equalization, timeout_sec={timeout:.1f}"
            )

        live_arm_base = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        actual_link = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_link8_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        final_error = abs(
            float(actual_link["left"][0][2])
            - float(actual_link["right"][0][2])
        )
        result_detail = (
            f"post_support_z_equalization=completed; {detail}; "
            f"{motion_detail}; final_z_error={final_error:.4f}m; "
            f"{force_monitor['detail']}"
        )
        self._publish_place_box_test_feedback(
            goal_handle,
            "POST_SUPPORT_LINK7_Z_EQUALIZED",
            result_detail,
        )
        return live_arm_base, actual_link, result_detail

    @staticmethod
    def _place_box_test_work_frame_offset(start_transform, target_transform):
        """Return the 6D work-frame offset from one Link7 pose to another."""
        start_position, start_orientation = start_transform
        target_position, target_orientation = target_transform
        rotation_offset = BoxSupportMixin._normalize_quaternion(
            quaternion_multiply(
                target_orientation,
                BoxSupportMixin._quaternion_conjugate(start_orientation),
            )
        )
        return [
            float(target_position[index]) - float(start_position[index])
            for index in range(3)
        ] + quaternion_to_rpy(rotation_offset)

    @staticmethod
    def _place_box_test_bottom_z(box_pose, half_height_m: float) -> float:
        """Height of the box bottom for the upright box model (local X is up)."""
        vertical = abs(rotate_vector((1.0, 0.0, 0.0), box_pose[1])[2])
        if vertical < math.cos(math.radians(15.0)):
            raise MissionError(
                "place_box_test box is too tilted for the upright table-height model"
            )
        return float(box_pose[0][2]) - half_height_m * vertical

    @staticmethod
    def _place_box_test_shift_box_z(box_pose, delta_z: float):
        return (
            (box_pose[0][0], box_pose[0][1], box_pose[0][2] + delta_z),
            box_pose[1],
        )

    def _place_box_test_table_z_in_base(self, base_frame: str) -> float:
        base_to_foot = self._lookup_tf_carry_transform(base_frame, "base_footprint")
        foot_up = rotate_vector((0.0, 0.0, 1.0), base_to_foot[1])
        if foot_up[2] < 0.99:
            raise MissionError("base_footprint Z is not aligned with base_link Z")
        return (
            base_to_foot[0][2]
            + self._float("place_box_test_table_height_base_footprint_m") * foot_up[2]
        )

    def _place_box_test_box_world_targets(self, box_pose, relations):
        return {
            arm: BoxSupportMixin._compose_transform(box_pose, relations[arm])
            for arm in ("left", "right")
        }

    def _place_box_test_current_box(self, base_frame: str, relations):
        inferred = {
            arm: BoxSupportMixin._compose_transform(
                self._lookup_tf_carry_transform(
                    base_frame, self._string(f"{arm}_link8_frame").strip().lstrip("/")
                ),
                BoxSupportMixin._inverse_transform(relations[arm]),
            )
            for arm in ("left", "right")
        }
        position_error = self._endpoint_sync_pose_position_error(
            inferred["left"][0], inferred["right"][0]
        )
        orientation_error = self._endpoint_sync_pose_orientation_error(
            inferred["left"][1], inferred["right"][1]
        )
        if (
            position_error > self._float("place_box_test_target_consistency_position_tolerance_m")
            or orientation_error > self._float("place_box_test_target_consistency_orientation_tolerance_rad")
        ):
            raise MissionError(
                "place_box_test rigid grasp lost: "
                f"box_position_disagreement={position_error:.4f}m, "
                f"box_orientation_disagreement={orientation_error:.4f}rad"
            )
        return BoxSupportMixin._mean_rigid_transforms(
            inferred["left"], inferred["right"]
        )

    def _place_box_test_waist_path(
        self, current_box, target_box, relations, body_start, target_units,
        units_per_degree, live_arm_base, segments,
    ):
        """Derive both arm paths from one rigid box path and the same waist path."""
        start_angles = [float(value) for value in body_start[:3]]
        start_units = [
            int(round(math.degrees(body_start[index]) * units_per_degree[index]))
            for index in range(4)
        ]
        arm_targets = {"left": [], "right": []}
        arm_transforms = {"left": [], "right": []}
        world_targets_by_segment = []
        body_targets = []
        fk_start = {
            arm: self._joint123_arm_base_transform(arm, start_angles)
            for arm in ("left", "right")
        }
        for segment_index in range(1, segments + 1):
            fraction = float(segment_index) / float(segments)
            body_units = [
                target_units[index] if segment_index == segments else int(round(
                    start_units[index]
                    + fraction * (target_units[index] - start_units[index])
                ))
                for index in range(4)
            ]
            body_angles = [
                math.radians(float(body_units[index]) / units_per_degree[index])
                for index in range(4)
            ]
            box_pose = (
                tuple(
                    current_box[0][axis]
                    + fraction * (target_box[0][axis] - current_box[0][axis])
                    for axis in range(3)
                ),
                BoxSupportMixin._slerp_quaternion(
                    current_box[1], target_box[1], fraction
                ),
            )
            world_targets = self._place_box_test_box_world_targets(box_pose, relations)
            world_targets_by_segment.append(world_targets)
            body_targets.append((body_units, body_angles))
            for arm in ("left", "right"):
                fk_waypoint = self._joint123_arm_base_transform(
                    arm, body_angles[:3]
                )
                waypoint_arm_base = BoxSupportMixin._compose_transform(
                    live_arm_base[arm],
                    BoxSupportMixin._compose_transform(
                        BoxSupportMixin._inverse_transform(fk_start[arm]),
                        fk_waypoint,
                    ),
                )
                local_target = BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(waypoint_arm_base),
                    world_targets[arm],
                )
                arm_transforms[arm].append(local_target)
                arm_targets[arm].append(pose_to_sdk_target(
                    self._endpoint_sync_transform_to_pose(local_target)
                ))
        return body_targets, arm_targets, arm_transforms, world_targets_by_segment

    def _place_box_test_optimize_waist_box_path(
        self, goal_handle, adapter, current_box, relations, body_start,
        target_units, units_per_degree, live_arm_base, segments,
        table_z, half_height,
    ):
        """Search common base_link Y/Z shifts before commanding the waist."""
        if not self._boolean("place_box_test_waist_workspace_enabled"):
            return current_box, "waist_workspace=disabled; box_held_at_fixed_pose"
        if adapter is None:
            raise MissionError("place waist workspace IK needs a Python SDK adapter")

        now = time.monotonic()
        max_age = self._float("box_pre_target_arm_movej_feedback_max_age_sec")
        with self.joint_state_lock:
            joint_rad = {
                arm: list(self.latest_slave_arm_positions.get(arm, []))
                for arm in ("left", "right")
            }
            ages = {
                arm: now - self.latest_slave_arm_state_times.get(arm, 0.0)
                for arm in ("left", "right")
            }
        if any(
            len(joint_rad[arm]) < 7 or ages[arm] > max_age
            or not all(math.isfinite(value) for value in joint_rad[arm][:7])
            for arm in ("left", "right")
        ):
            raise MissionError(
                "place waist workspace needs fresh bilateral seven-joint feedback: "
                f"ages={ages}, limit={max_age:.3f}s"
            )
        limits = {
            arm: (
                self._float_array(f"waist_workspace_{arm}_arm_joint_min_deg"),
                self._float_array(f"waist_workspace_{arm}_arm_joint_max_deg"),
            )
            for arm in ("left", "right")
        }
        max_y = self._float("place_box_test_waist_y_search_half_range_m")
        y_step = self._float("place_box_test_waist_y_search_step_m")
        max_drop = min(
            self._float("place_box_test_waist_z_max_drop_m"),
            max(0.0, self._place_box_test_bottom_z(current_box, half_height)
                - table_z - self._float("place_box_test_waist_clearance_m")),
        )
        z_step = self._float("place_box_test_waist_z_search_step_m")
        minimum_margin = self._float("place_box_test_waist_ik_min_margin_deg")
        max_joint_step = self._float("place_box_test_waist_ik_max_step_deg")
        if not all(math.isfinite(value) for value in (
            max_y, y_step, max_drop, z_step, minimum_margin, max_joint_step
        )) or y_step <= 0.0 or z_step <= 0.0 or max_y < 0.0 or max_drop < 0.0:
            raise MissionError("place waist workspace search parameters are invalid")
        if (2 * int(max_y / y_step) + 3) * (int(max_drop / z_step) + 2) > 250:
            raise MissionError(
                "place waist workspace Y/Z search exceeds 250 candidates; "
                "increase the search steps or reduce the ranges"
            )
        y_values = sorted(
            {0.0, -max_y, max_y} | {
                round(index * y_step, 9)
                for index in range(-int(max_y / y_step), int(max_y / y_step) + 1)
            },
            key=lambda value: (abs(value), value),
        )
        z_values = sorted(
            {0.0, -max_drop} | {
                round(-index * z_step, 9)
                for index in range(int(max_drop / z_step) + 1)
            },
            key=lambda value: abs(value),
        )
        best = None
        checked = 0
        valid = 0
        ik_rejected = 0
        joint_rejected = 0
        for dy in y_values:
            for dz in z_values:
                self._check_canceled(goal_handle, "while checking place waist Y/Z IK")
                checked += 1
                candidate_box = (
                    (current_box[0][0], current_box[0][1] + dy,
                     current_box[0][2] + dz),
                    current_box[1],
                )
                _, candidate_targets, _, _ = self._place_box_test_waist_path(
                    current_box, candidate_box, relations, body_start,
                    target_units, units_per_degree, live_arm_base, segments,
                )
                seeds = {
                    arm: [math.degrees(value) for value in joint_rad[arm][:7]]
                    for arm in ("left", "right")
                }
                worst_margin = math.inf
                rejected = False
                for index in range(segments):
                    for arm in ("left", "right"):
                        solution = adapter.solve_ik(
                            arm, candidate_targets[arm][index], seeds[arm]
                        )
                        if solution is None:
                            ik_rejected += 1
                            rejected = True
                            break
                        solution = [float(value) for value in solution]
                        if len(solution) != 7 or not all(
                            math.isfinite(value) for value in solution
                        ):
                            raise MissionError(
                                f"place waist {arm} offline IK returned invalid joints"
                            )
                        if (
                            self._boolean("box_ik_joint4_negative_required")
                            and solution[3] >= 0.0
                        ) or max(
                            abs(value - previous)
                            for value, previous in zip(solution, seeds[arm])
                        ) > max_joint_step:
                            joint_rejected += 1
                            rejected = True
                            break
                        lower, upper = limits[arm]
                        margin = min(
                            min(value - lo, hi - value)
                            for value, lo, hi in zip(solution, lower, upper)
                        )
                        if margin < minimum_margin:
                            joint_rejected += 1
                            rejected = True
                            break
                        worst_margin = min(worst_margin, margin)
                        seeds[arm] = solution
                    if rejected:
                        break
                if rejected:
                    continue
                valid += 1
                score = worst_margin - 20.0 * abs(dy) - 10.0 * abs(dz)
                rank = (score, worst_margin, -abs(dy), -abs(dz))
                if best is None or rank > best[0]:
                    best = (rank, candidate_box, dy, dz, worst_margin)
        if best is None:
            raise MissionError(
                "place waist Y/Z search found no bilateral waypoint-IK path "
                f"before motion: checked={checked}, ik_rejected={ik_rejected}, "
                f"joint_rejected={joint_rejected}; y_range=±{max_y:.3f}m, "
                f"max_drop={max_drop:.3f}m"
            )
        _, selected_box, dy, dz, margin = best
        detail = (
            f"waist_workspace=enabled; frame=base_link; "
            f"box_delta_y={dy:+.4f}m; box_delta_z={dz:+.4f}m; "
            f"checked={checked}; valid={valid}; min_joint_margin={margin:.3f}deg; "
            f"ik_points={segments}x2; box_orientation=fixed; "
            f"rigid_box_to_arms=fixed; table_clearance={self._place_box_test_bottom_z(selected_box, half_height) - table_z:.4f}m"
        )
        self._publish_place_box_test_feedback(
            goal_handle, "PLACE_WAIST_WORKSPACE_SELECTED", detail
        )
        return selected_box, detail

    def _place_box_test_live_descent_state(self, base_frame, relations):
        """Read actual TCPs; infer box height for the bounded force search only.

        During descent we must not reconstruct arm targets from an averaged
        box pose and historical grasp relations: that can lift the lower arm
        back toward a theoretical height. These are latest available TF values;
        no geometric target/rigid-grasp comparison runs here.
        """
        actual = {
            arm: self._lookup_tf_carry_transform(
                base_frame, self._string(f"{arm}_link8_frame").strip().lstrip("/")
            )
            for arm in ("left", "right")
        }
        return actual, self._place_box_test_box_from_actual(actual, relations)

    @staticmethod
    def _place_box_test_box_from_actual(actual, relations):
        inferred = {
            arm: BoxSupportMixin._compose_transform(
                actual[arm], BoxSupportMixin._inverse_transform(relations[arm])
            )
            for arm in ("left", "right")
        }
        return BoxSupportMixin._mean_rigid_transforms(inferred["left"], inferred["right"])

    def _place_box_test_wait_arms_still(self, goal_handle):
        """Observe fresh joint feedback only; never issue a preparation MoveJ."""
        with self.joint_state_lock:
            targets = {
                arm: tuple(self.latest_slave_arm_positions.get(arm, ()))[:7]
                for arm in ("left", "right")
            }
            sequences = dict(self.latest_slave_arm_state_sequences)
        if any(len(q) != 7 or not all(math.isfinite(v) for v in q)
               for q in targets.values()):
            raise MissionError("placement requires valid dual-arm joint feedback before descent")
        self._wait_for_post_arm_joint_targets(
            goal_handle, targets["left"], targets["right"], sequences,
            parameter_prefix="box_pre_target_arm_movej",
            description="stationary placement arms (no motion commanded)",
        )

    def _place_box_test_capture_descent_reference(self, base_frame, relations):
        actual, box_pose = self._place_box_test_live_descent_state(base_frame, relations)
        bases = {
            arm: self._lookup_tf_carry_transform(
                base_frame, self._string(f"{arm}_arm_base_frame").strip().lstrip("/")
            )
            for arm in ("left", "right")
        }
        return _PlaceDescentReference(actual, bases), box_pose

    def _place_box_test_descent_world_targets(self, reference, descent_m, y_offset_m=0.0):
        # Ordinary MoveL needs a complete Pose. Keep X/orientation at the
        # measured descent origin, add mirrored Work-Y offsets, and subtract
        # the shared distance from each OWN Work Z. No offline IK/search.
        return {
            arm: (tuple(tcp[0][i] + rotate_vector(
                            (0.0, y_offset_m if arm == 'left' else -y_offset_m, -descent_m),
                            reference.base_by_arm[arm][1])[i]
                        for i in range(3)), tcp[1])
            for arm, tcp in reference.tcp_by_arm.items()
        }

    def _place_box_test_move_box_z(
        self, goal_handle, adapter, base_frame, relations, reference, descent_m, speed
    ):
        """Send absolute targets from ONE origin pair and a shared descent.

        Subsequent feedback verifies Work-Z, never becomes a new trajectory origin.
        No force polling callback or upward recovery motion is introduced.
        """
        if not math.isfinite(descent_m) or descent_m <= 0.0:
            raise MissionError("table descent requires a finite positive cumulative distance")
        y_step = self._float("place_box_test_descent_work_y_step_m")
        y_limit = self._float("place_box_test_descent_work_y_max_travel_m")
        if not (math.isfinite(y_step) and math.isfinite(y_limit)
                and 0 <= y_step <= 0.005 and y_step <= y_limit <= 0.100):
            raise MissionError("invalid placement Work-Y step/travel limits")
        y_offset = (min(y_limit, reference.y_offset_m + y_step)
                    if reference.allow_y_change else reference.y_offset_m)
        before, _ = self._place_box_test_live_descent_state(base_frame, relations)
        measured_before = {
            arm: work_z(reference.base_by_arm[arm], reference.tcp_by_arm[arm])
                 - work_z(reference.base_by_arm[arm], before[arm])
            for arm in ("left", "right")
        }
        requested_drop = descent_m
        step = requested_drop - reference.commanded_drop_m
        limit = reference.maximum_drop_m
        if (not all(math.isfinite(v) for v in (*measured_before.values(), step, limit))
                or step <= 0.0 or limit <= 0.0):
            raise MissionError("invalid measured descent progress, step or fixed travel limit")
        progress = max(reference.commanded_drop_m, *measured_before.values(), 0.0)
        # Advance the common scalar, not either TCP origin. The same increment
        # starts below the farther-descended arm; a lagging arm catches up.
        # Clip against the ORIGINAL total travel budget, never reset it.
        descent_m = min(limit, progress + step)
        if progress >= limit - 1e-6 or descent_m <= progress + 1e-6:
            raise MissionError(
                "placement measured descent reached the fixed travel limit; "
                f"measured_drop={measured_before}; commanded_drop={reference.commanded_drop_m:.6f}m; "
                f"maximum_drop={limit:.6f}m; no next segment or release commanded"
            )
        if descent_m > requested_drop + 1e-6:
            self._publish_place_box_test_feedback(
                goal_handle, "PLACE_DESCENT_PROGRESS_REBASED",
                f"nominal_drop={requested_drop:.6f}m; measured_work_drop={measured_before}; "
                f"next_common_drop={descent_m:.6f}m; step={step:.6f}m; "
                f"maximum_drop={limit:.6f}m; origin=unchanged; upward_motion=false",
            )
        world_targets = self._place_box_test_descent_world_targets(reference, descent_m, y_offset)
        # Do not pull an overshot arm upward to recover the nominal trajectory.
        # Final defensive check; updating progress never resets the origin.
        if any(sum((world_targets[arm][0][i] - before[arm][0][i]) * rotate_vector(
                       (0.0, 0.0, 1.0), reference.base_by_arm[arm][1])[i]
                   for i in range(3)) > 1e-6
               for arm in ("left", "right")):
            raise MissionError(
                "placement cumulative target would move an arm upward after overshoot; "
                "no corrective motion or release commanded"
            )
        local_targets = {}
        for arm in ("left", "right"):
            arm_target = BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(reference.base_by_arm[arm]), world_targets[arm]
            )
            local_targets[arm] = pose_to_sdk_target(
                self._endpoint_sync_transform_to_pose(arm_target)
            )
        started = time.monotonic()
        adapter.execute_dual_movel_endpoint(
            local_targets["left"], local_targets["right"], speed, speed,
            cancel_requested=lambda: goal_handle.is_cancel_requested,
            timeout_sec=self._float("place_box_test_timeout_sec"),
        )
        reference.y_offset_m = y_offset
        reference.commanded_drop_m = descent_m
        actual = self._place_box_test_verify_descent_z(
            goal_handle, adapter, base_frame, reference, world_targets
        )
        measured_box = self._place_box_test_box_from_actual(actual, relations)
        measured_drop = {
            arm: work_z(reference.base_by_arm[arm], reference.tcp_by_arm[arm])
                 - work_z(reference.base_by_arm[arm], actual[arm])
            for arm in ("left", "right")
        }
        self._publish_place_box_test_feedback(
            goal_handle, "PLACE_DESCENT_PROGRESS",
            f"common_commanded_drop={descent_m:.6f}m; "
            "descent_frame=each_arm_base_negative_Z; offline_ik=not_used; y_search=not_used; "
            f"left_work_y_offset=+{y_offset:.6f}m; right_work_y_offset=-{y_offset:.6f}m; "
            f"left_measured_drop={measured_drop['left']:.6f}m; "
            f"right_measured_drop={measured_drop['right']:.6f}m; "
            f"drop_difference={measured_drop['right'] - measured_drop['left']:+.6f}m; "
            f"left_actual_xyz={actual['left'][0]}; right_actual_xyz={actual['right'][0]}; "
            f"step_elapsed={time.monotonic() - started:.3f}s; speed={speed:.1f}%; "
            "origin=fixed_after_waist; both_movel_completed=true; "
            "work_z_verified=true; feedback=synchronized_fresh_TF",
        )
        return measured_box

    def _place_box_test_correct_descent_z(
        self, goal_handle, adapter, reference, actual, arm, distance
    ):
        """One ordinary MoveL on the lagging arm; leave the other arm alone."""
        base = reference.base_by_arm[arm]
        local = BoxSupportMixin._compose_transform(
            BoxSupportMixin._inverse_transform(base), actual[arm]
        )
        target = ((local[0][0], local[0][1], local[0][2] - distance), local[1])
        world = BoxSupportMixin._compose_transform(base, target)
        if distance <= 0.0 or world[0][2] >= actual[arm][0][2]:
            raise MissionError('Z catch-up must move downward in Work and common frame')
        adapter.execute_single(
            arm=arm,
            target=pose_to_sdk_target(self._endpoint_sync_transform_to_pose(target)),
            motion_mode="movel",
            speed_percent=self._float("place_box_test_table_descent_velocity_percent"),
            blocking=True,
            cancel_requested=lambda: goal_handle.is_cancel_requested,
            timeout_sec=min(5.0, self._float("place_box_test_timeout_sec")),
        )

    def _place_box_test_stable_work_fz_baseline(self, goal_handle, adapter, threshold):
        """Wait for a quiet rolling window; never move or clear force sensors."""
        timeout = self._float("place_box_test_force_unload_baseline_timeout_sec")
        if not math.isfinite(timeout) or timeout <= 0.0 or threshold <= 0.0:
            raise MissionError("invalid placement force baseline settings")
        deadline = time.monotonic() + timeout
        samples = deque(maxlen=8)
        spans = {"left": math.inf, "right": math.inf}
        waiting_reported = False
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "while waiting for stable loaded place force")
            force = adapter.read_bilateral_work_fz()
            if not all(math.isfinite(force[arm]) for arm in ("left", "right")):
                raise MissionError("placement force baseline contains invalid data")
            samples.append(force)
            if len(samples) == samples.maxlen:
                spans = {
                    arm: max(s[arm] for s in samples) - min(s[arm] for s in samples)
                    for arm in ("left", "right")
                }
                if all(span <= threshold * 0.5 for span in spans.values()):
                    return {
                        arm: statistics.median(s[arm] for s in samples)
                        for arm in ("left", "right")
                    }
                if not waiting_reported:
                    self._publish_place_box_test_feedback(
                        goal_handle, "WAITING_FOR_PLACE_FORCE_STABILITY",
                        "waist stopped; arms held stationary, no motion commands; "
                        f"waiting for 8 stable samples: spans={spans}, "
                        f"limit={threshold * 0.5:.3f}N, timeout={timeout:.1f}s",
                    )
                    waiting_reported = True
            time.sleep(0.05)
        raise MissionError(
            "SDK Work-Fz baseline did not stabilize before timeout; "
            f"left_span={spans['left']:.3f}N, right_span={spans['right']:.3f}N, "
            f"limit={threshold * 0.5:.3f}N, timeout={timeout:.1f}s; no descent started"
        )

    def _place_box_test_descend_to_table(
        self, goal_handle, adapter, base_frame, relations, box_pose, table_z, half_height
    ):
        if self._string("place_box_test_descent_mode").strip().lower() == "continuous":
            return self._place_box_test_descend_continuous(
                goal_handle, adapter, base_frame, relations, box_pose, table_z, half_height
            )
        return self._place_box_test_descend_to_table_segmented(
            goal_handle, adapter, base_frame, relations, box_pose, table_z, half_height
        )

    def _place_box_test_descend_continuous(
        self, goal_handle, adapter, base_frame, relations, box_pose, table_z, half_height
    ):
        """One world-vertical MoveL per arm; stop on contact, never auto-release.

        Empty-arm tuning supplied 10/15 percent, not a real-time speed servo.
        Fixed origins preserve the initial height difference and orientation.
        The old Work-Z/Y segmented implementation remains selectable below.
        """
        arms = ('left', 'right')
        floors = {a: self._float(f'place_box_test_continuous_{a}_min_z_footprint_m') for a in arms}
        speeds = {a: self._float(f'place_box_test_continuous_{a}_velocity_percent') for a in arms}
        difference_limit = self._float('place_box_test_continuous_max_drop_difference_m')
        if (any(not math.isfinite(v) or not 0 <= v <= 2. for v in floors.values())
                or not math.isfinite(difference_limit) or not 0 < difference_limit <= 0.10
                or any(not math.isfinite(v) or not 1 <= v <= 100 or v != int(v) for v in speeds.values())):
            raise MissionError('invalid continuous Place distance, speeds or difference limit')
        self._place_box_test_wait_arms_still(goal_handle)
        threshold = self._float('place_box_test_table_support_delta_fz_n')
        force_limit = self._float('place_box_test_post_support_max_abs_work_fz_n')
        unloaded_limit = self._float('place_box_test_table_unloaded_abs_fz_n')
        signs = {a: self._float(f'place_box_test_table_support_sign_{a}') for a in arms}
        baseline = self._place_box_test_stable_work_fz_baseline(goal_handle, adapter, threshold)
        reference, box_pose = self._place_box_test_capture_descent_reference(base_frame, relations)
        foot = self._lookup_tf_carry_transform(base_frame, 'base_footprint')
        up = rotate_vector((0., 0., 1.), foot[1])
        if up[2] < 0.99:
            raise MissionError('continuous Place requires common frame Z aligned with footprint Z')
        def footprint_z(pose):
            return sum((pose[0][i]-foot[0][i])*up[i] for i in range(3))
        height_budget = min(footprint_z(reference.tcp_by_arm[a])-floors[a] for a in arms)
        distance = min(height_budget, self._place_box_test_bottom_z(box_pose, half_height)-table_z
                       + self._float('place_box_test_table_max_overtravel_m'))
        if not math.isfinite(distance) or distance <= 0:
            raise MissionError('already at/below minimum TCP Z or no table clearance; no descent/release')
        targets = {a: (tuple(reference.tcp_by_arm[a][0][i]-distance*up[i] for i in range(3)),
                       reference.tcp_by_arm[a][1]) for a in arms}
        local = {a: pose_to_sdk_target(self._endpoint_sync_transform_to_pose(
            BoxSupportMixin._compose_transform(BoxSupportMixin._inverse_transform(reference.base_by_arm[a]), targets[a])
        )) for a in arms}
        detail = (f'continuous MoveL; frame=base_footprint_negative_Z; distance={distance:.4f}m; '
                  f'min_z_footprint={floors}; left_speed={speeds["left"]:.0f}%; '
                  f'right_speed={speeds["right"]:.0f}%; work_y_offset=disabled; '
                  f'drop_difference_limit={difference_limit:.4f}m; stop_on_contact=true; '
                  'release_requires_stationary_bilateral_support=true')
        self._publish_place_box_test_feedback(goal_handle, 'PLACE_CONTINUOUS_DESCENT_TARGETS', detail)
        self._publish_place_box_test_feedback(goal_handle, 'PLACE_CONTINUOUS_DESCENT_REFERENCE',
            f'frame={base_frame}; measured_starts={reference.tcp_by_arm}; world_targets={targets}; baseline={baseline}')
        stop = threading.Event()
        state = {}
        def watch():
            contact_count = 0
            last_log = 0.
            try:
                while not stop.is_set():
                    actual, _stamp, why = self._place_box_test_read_z_check_tf(
                        base_frame, 0, self._float('place_box_test_descent_z_feedback_max_age_sec'))
                    if actual is None:
                        raise MissionError(f'continuous descent TF unavailable: {why}')
                    drops = {a: sum((reference.tcp_by_arm[a][0][i]-actual[a][0][i])*up[i]
                                    for i in range(3)) for a in arms}
                    if abs(drops['left']-drops['right']) > difference_limit:
                        raise MissionError(f'continuous descent mismatch: drops={drops}')
                    if any(footprint_z(actual[a]) <= floors[a]+.005 for a in arms):
                        state['height_stop'] = {a: footprint_z(actual[a]) for a in arms}
                        adapter.stop_all()
                        return
                    force = adapter.read_bilateral_work_fz()
                    if any(not math.isfinite(force[a]) or abs(force[a]) > force_limit for a in arms):
                        raise MissionError(f'continuous descent force limit: {force}')
                    delta = {a: signs[a]*(force[a]-baseline[a]) for a in arms}
                    # Stop on persistent first-side contact; never push farther
                    # just to make the other side reach its release threshold.
                    contact = any(v >= threshold for v in delta.values()) and sum(delta.values()) >= threshold
                    contact_count = contact_count+1 if contact else 0
                    if time.monotonic()-last_log >= .2:
                        self._publish_place_box_test_feedback(goal_handle, 'PLACE_CONTINUOUS_DESCENT_PROGRESS',
                            f'drops={drops}; left_minus_right={drops["left"]-drops["right"]:+.6f}m; delta_Fz={delta}')
                        last_log = time.monotonic()
                    if contact_count >= 3:
                        state['contact_stop'] = delta
                        adapter.stop_all()
                        return
                    stop.wait(.05)
            except Exception as exc:
                state['error'] = exc
                adapter.stop_all()
        monitor = threading.Thread(target=watch, name='place-continuous-monitor', daemon=True)
        motion_error = None
        try:
            self._check_canceled(goal_handle, 'before continuous descent')
            adapter.execute_dual_movel_endpoint(local['left'], local['right'], speeds['left'], speeds['right'],
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=self._float('place_box_test_timeout_sec'), after_start=monitor.start)
        except (RealManSdkCanceled, RealManSdkError) as exc:
            motion_error = exc
        finally:
            stop.set()
            if monitor.ident is not None:
                monitor.join(timeout=3.)
            if monitor.is_alive():
                adapter.stop_all()
                raise MissionError('continuous descent monitor did not stop; box was not released')
        self._check_canceled(goal_handle, 'after continuous descent')
        if 'error' in state:
            raise MissionError(str(state['error']))
        if motion_error is not None and not any(k in state for k in ('contact_stop','height_stop')):
            raise motion_error
        self._place_box_test_wait_arms_still(goal_handle)
        actual, measured_box = self._place_box_test_live_descent_state(base_frame, relations)
        drops = {a: sum((reference.tcp_by_arm[a][0][i]-actual[a][0][i])*up[i] for i in range(3)) for a in arms}
        if abs(drops['left']-drops['right']) > difference_limit:
            raise MissionError('continuous descent final height difference exceeded limit; no release')
        if any(footprint_z(actual[a]) < floors[a]-.005 for a in arms):
            raise MissionError('continuous descent minimum TCP Z exceeded (>5mm); no release')
        bottom = self._place_box_test_bottom_z(measured_box, half_height)
        if bottom > table_z+self._float('place_box_test_table_early_contact_tolerance_m'):
            raise MissionError('continuous descent stopped above table tolerance; no release')
        for _ in range(3):
            self._check_canceled(goal_handle, 'confirming continuous descent table support')
            force = adapter.read_bilateral_work_fz()
            if any(not math.isfinite(force[a]) or signs[a]*(force[a]-baseline[a]) < threshold
                   or abs(force[a]) > unloaded_limit for a in arms):
                raise MissionError('continuous descent ended without stable bilateral table support; '
                                   f'force={force}; baseline={baseline}; no segmented fallback or release')
            time.sleep(.05)
        detail += (f'; table_support=confirmed; drops={drops}; contact_stop={state.get("contact_stop")}; '
                   f'height_stop={state.get("height_stop")}; post_support_extra_descent=disabled')
        self._publish_place_box_test_feedback(goal_handle, 'TABLE_SUPPORT_CONFIRMED', detail)
        return measured_box, detail

    def _place_box_test_descend_to_table_segmented(
        self, goal_handle, adapter, base_frame, relations, box_pose, table_z, half_height
    ):
        """Lower a rigidly held box, stopping only after bilateral load transfer."""
        threshold = self._float("place_box_test_table_support_delta_fz_n")
        unloaded_limit = self._float("place_box_test_table_unloaded_abs_fz_n")
        signs = {
            arm: self._float(f"place_box_test_table_support_sign_{arm}")
            for arm in ("left", "right")
        }
        self._place_box_test_wait_arms_still(goal_handle)
        baseline = self._place_box_test_stable_work_fz_baseline(
            goal_handle, adapter, threshold
        )
        self._publish_place_box_test_feedback(
            goal_handle, "PLACE_DESCENT_BASELINE",
            f"SDK Work Fz baseline N: left={baseline['left']:.3f}, "
            f"right={baseline['right']:.3f}; threshold={threshold:.3f}N",
        )
        reference, box_pose = self._place_box_test_capture_descent_reference(base_frame, relations)
        self._publish_place_box_test_feedback(
            goal_handle, "PLACE_DESCENT_REFERENCE",
            f"fixed TCP origins in {base_frame}: left={reference.tcp_by_arm['left']}; "
            f"right={reference.tcp_by_arm['right']}; shared cumulative Work-Z descent; "
            "orientation=fixed; X=measured_origin; Y=left_positive_right_negative_each_Work; "
            "offline_ik=not_used; y_search=not_used; "
            "no in-motion force callback; endpoint support checks retained",
        )
        max_overtravel = self._float("place_box_test_table_max_overtravel_m")
        early_contact_tolerance = self._float(
            "place_box_test_table_early_contact_tolerance_m"
        )
        near_distance = self._float("place_box_test_table_fine_distance_m")
        coarse_step = self._float("place_box_test_table_coarse_step_m")
        fine_step = self._float("place_box_test_table_fine_step_m")
        speed = self._float("place_box_test_table_descent_velocity_percent")
        first_contact_z = {}
        commanded_drop = 0.0
        maximum_drop = max(0.0, self._place_box_test_bottom_z(box_pose, half_height)
                           - table_z + max_overtravel)
        reference.maximum_drop_m = maximum_drop
        deadline = time.monotonic() + self._float("place_box_test_timeout_sec")
        while time.monotonic() < deadline:
            self._check_canceled(goal_handle, "during table-support descent")
            bottom = self._place_box_test_bottom_z(box_pose, half_height)
            distance = bottom - table_z
            if distance < -max_overtravel:
                break
            # Do not descend in a large step near the expected table plane.
            step = (
                fine_step if distance <= near_distance + 1e-6
                else min(coarse_step, distance - near_distance)
            )
            step = min(step, max(0.0, distance + max_overtravel),
                       max(0.0, maximum_drop - commanded_drop))
            if step <= 1e-6:
                break
            commanded_drop += step
            box_pose = self._place_box_test_move_box_z(
                goal_handle, adapter, base_frame, relations, reference, commanded_drop, speed
            )
            commanded_drop = reference.commanded_drop_m
            time.sleep(0.05)
            force = adapter.read_bilateral_work_fz()
            changes = {
                arm: signs[arm] * (force[arm] - baseline[arm])
                for arm in ("left", "right")
            }
            bottom = self._place_box_test_bottom_z(box_pose, half_height)
            new_contacts = []
            for arm in ("left", "right"):
                if changes[arm] >= threshold and arm not in first_contact_z:
                    first_contact_z[arm] = bottom
                    new_contacts.append(arm)
            if first_contact_z:
                # Keep the existing no-lateral-slide policy after contact.
                reference.allow_y_change = False
            if first_contact_z and bottom > table_z + early_contact_tolerance:
                raise MissionError(
                    "place_box_test contact detected above the expected table "
                    "tolerance; stopping before release: "
                    f"bottom_z={bottom:.4f}m, table_z={table_z:.4f}m, "
                    f"tolerance={early_contact_tolerance:.4f}m, "
                    f"contact_arms={list(first_contact_z)}, "
                    f"left_delta_Fz={changes['left']:.3f}N, "
                    f"right_delta_Fz={changes['right']:.3f}N"
                )
            if len(first_contact_z) == 1 and new_contacts:
                self._publish_place_box_test_feedback(
                    goal_handle, "PLACE_TABLE_UNILATERAL_FORCE",
                    f"first_side={new_contacts[0]}; bottom_z={bottom:.4f}m; "
                    f"left_delta_Fz={changes['left']:.3f}N; "
                    f"right_delta_Fz={changes['right']:.3f}N; "
                    "continuing common base_link -Z descent until both sides "
                    "meet the force threshold or the table search ends",
                )
            if all(changes[arm] >= threshold for arm in ("left", "right")):
                # Stay stationary and verify the contact signal persists; do
                # not command additional downward travel during confirmation.
                for _ in range(3):
                    self._check_canceled(goal_handle, "while confirming table support")
                    time.sleep(0.05)
                    stable_force = adapter.read_bilateral_work_fz()
                    if any(
                        signs[arm] * (stable_force[arm] - baseline[arm]) < threshold
                        or abs(stable_force[arm]) > unloaded_limit
                        for arm in ("left", "right")
                    ):
                        raise MissionError(
                            "place_box_test bilateral table support/unloaded Fz "
                            "did not remain stable; box was not released: "
                            f"left={stable_force['left']:.3f}N, "
                            f"right={stable_force['right']:.3f}N, "
                            f"unloaded_limit={unloaded_limit:.3f}N"
                        )
                detail = (
                    f"table support confirmed at bottom_z={bottom:.4f}m, "
                    f"left_delta_Fz={changes['left']:.3f}N, "
                    f"right_delta_Fz={changes['right']:.3f}N"
                )
                self._publish_place_box_test_feedback(
                    goal_handle, "TABLE_SUPPORT_CONFIRMED", detail
                )
                return box_pose, detail
        raise MissionError(
            "place_box_test reached the table-height limit without bilateral "
            "SDK Work-Fz support confirmation; box was not released"
        )

    def _place_box_test_post_support_arm_base_descent(
        self, goal_handle, adapter, base_frame: str, *, support_source="bilateral"
    ):
        """Move each held TCP down its own arm-base Z after table support."""
        distance = self._float("place_box_test_post_support_arm_base_descent_m")
        if distance <= 0.0:
            raise MissionError("post-support arm-base descent must be positive")
        step_m = min(self._float("place_box_test_table_coarse_step_m"), 0.005)
        speed = self._float("place_box_test_table_descent_velocity_percent")
        arm_base = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        start_local = {
            arm: BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(arm_base[arm]),
                self._lookup_tf_carry_transform(
                    base_frame,
                    self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                ),
            )
            for arm in ("left", "right")
        }
        guard_reference = _PlaceDescentReference(
            {arm: BoxSupportMixin._compose_transform(arm_base[arm], start_local[arm])
             for arm in ("left", "right")}, arm_base,
        )
        steps = max(1, int(math.ceil(distance / step_m)))
        self._publish_place_box_test_feedback(
            goal_handle, "POST_SUPPORT_ARM_BASE_Z_DESCENT",
            f"support_source={support_source}; descending both arms along "
            f"their own arm-base -Z by {distance:.4f}m in {steps} "
            f"unconnected SDK MoveL steps; speed={speed:.1f}%",
        )
        target_local = None
        for index in range(steps):
            self._check_canceled(
                goal_handle, "during post-support arm-base Z descent"
            )
            traveled = min(distance, (index + 1) * step_m)
            target_local = {
                arm: (
                    (
                        start_local[arm][0][0],
                        start_local[arm][0][1],
                        start_local[arm][0][2] - traveled,
                    ),
                    start_local[arm][1],
                )
                for arm in ("left", "right")
            }
            world_targets = {
                arm: BoxSupportMixin._compose_transform(
                    arm_base[arm], target_local[arm]
                )
                for arm in ("left", "right")
            }
            sdk_targets = {
                arm: pose_to_sdk_target(
                    self._endpoint_sync_transform_to_pose(target_local[arm])
                )
                for arm in ("left", "right")
            }
            adapter.execute_dual_movel_endpoint(
                sdk_targets["left"], sdk_targets["right"],
                speed, speed,
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=self._float("place_box_test_timeout_sec"),
            )
            self._place_box_test_verify_descent_z(
                goal_handle, adapter, base_frame, guard_reference, world_targets
            )
            self._wait_for_place_box_test_world_targets(goal_handle, world_targets)
            time.sleep(0.05)
            force = adapter.read_bilateral_work_fz()
            force_limit = self._float("place_box_test_post_support_max_abs_work_fz_n")
            if any(abs(force[arm]) > force_limit for arm in ("left", "right")):
                raise MissionError(
                    "post-support arm-base Z descent increased wrist load; "
                    "stopping before release: "
                    f"step={index + 1}/{steps}, "
                    f"left_work_Fz={force['left']:.3f}N, "
                    f"right_work_Fz={force['right']:.3f}N, "
                    f"max_abs_work_Fz={force_limit:.3f}N"
                )
            self._publish_place_box_test_feedback(
                goal_handle, "POST_SUPPORT_ARM_BASE_Z_PROGRESS",
                f"step={index + 1}/{steps}; traveled={traveled:.4f}m; "
                f"left_work_Fz={force['left']:.3f}N; "
                f"right_work_Fz={force['right']:.3f}N",
            )
        final_pose = {}
        for arm in ("left", "right"):
            actual_local = BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(arm_base[arm]),
                self._lookup_tf_carry_transform(
                    base_frame,
                    self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                ),
            )
            error = self._endpoint_sync_pose_position_error(
                actual_local[0], target_local[arm][0]
            )
            if error > self._float("place_box_test_position_tolerance_m"):
                raise MissionError(
                    f"post-support {arm} arm-base -Z target not reached: "
                    f"position_error={error:.4f}m; "
                    f"tolerance={self._float('place_box_test_position_tolerance_m'):.4f}m; "
                    "box was not released"
                )
            final_pose[arm] = self._endpoint_sync_transform_to_pose(actual_local)
        return final_pose, (
            f"post_support_arm_base_z_descent={distance:.4f}m; "
            f"steps={steps}; speed={speed:.1f}%"
        )

    def _execute_place_box_test_waist_only(
        self, goal_handle, adapter, box_type, dry_run, *, base_frame,
        relations, actual_link, live_arm_base, body_start, body_sequence,
        target_units, target_angles, table_z, half_height,
    ):
        """Dynamic placement: fixed arm joints, waist only, then measured descent.

        The forecast checks clearance without generating any arm trajectory.
        Actual arm targets are generated only after waist completion, from TF.
        """
        arms = ("left", "right")
        fixed_local = {
            arm: BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(live_arm_base[arm]), actual_link[arm]
            )
            for arm in arms
        }
        fk_start = {
            arm: self._joint123_arm_base_transform(arm, body_start[:3]) for arm in arms
        }
        segments = max(1, self._integer("place_box_test_segments"))
        minimum_bottom = math.inf
        # Sample the waist-only geometry, not an independently moving arm path.
        for index in range(segments + 1):
            self._check_canceled(goal_handle, "while checking waist-only clearance")
            angles = [
                start + (end - start) * index / segments
                for start, end in zip(body_start[:3], target_angles[:3])
            ]
            predicted_base = {
                arm: BoxSupportMixin._compose_transform(
                    live_arm_base[arm],
                    BoxSupportMixin._compose_transform(
                        BoxSupportMixin._inverse_transform(fk_start[arm]),
                        self._joint123_arm_base_transform(arm, angles),
                    ),
                )
                for arm in arms
            }
            predicted_world = {
                arm: BoxSupportMixin._compose_transform(predicted_base[arm], fixed_local[arm])
                for arm in arms
            }
            inferred = {
                arm: BoxSupportMixin._compose_transform(
                    predicted_world[arm], BoxSupportMixin._inverse_transform(relations[arm])
                )
                for arm in arms
            }
            predicted_box = BoxSupportMixin._mean_rigid_transforms(inferred['left'], inferred['right'])
            minimum_bottom = min(
                minimum_bottom, self._place_box_test_bottom_z(predicted_box, half_height)
            )
        required_bottom = table_z + self._float("place_box_test_waist_clearance_m")
        if minimum_bottom < required_bottom:
            raise MissionError(
                "waist-only bend would violate predicted table clearance; "
                "no automatic lift or arm compensation was commanded: "
                f"minimum_bottom_z={minimum_bottom:.4f}m, required={required_bottom:.4f}m"
            )

        post_distance = self._float("place_box_test_post_support_arm_base_descent_m")
        # Dry-run estimates only: measured post-waist TF replaces these before motion.
        drop = table_z - self._place_box_test_bottom_z(predicted_box, half_height)
        descent_mode = self._string('place_box_test_descent_mode').strip().lower()
        continuous_detail = ''
        if descent_mode == 'continuous':
            floors = {a: self._float(f'place_box_test_continuous_{a}_min_z_footprint_m') for a in arms}
            speeds = [self._float(f'place_box_test_continuous_{a}_velocity_percent') for a in arms]
            if (any(not math.isfinite(v) or not 0 <= v <= 2. for v in floors.values())
                    or any(not math.isfinite(v) or not 1 <= v <= 100 or v != int(v) for v in speeds)):
                raise MissionError('invalid continuous placement parameters; no waist motion commanded')
            foot = self._lookup_tf_carry_transform(base_frame, 'base_footprint')
            height_budget = min(predicted_world[a][0][2]-foot[0][2]-floors[a] for a in arms)
            drop = -min(max(0.,height_budget), max(0., -drop+self._float('place_box_test_table_max_overtravel_m')))
            post_distance = 0.0  # Absolute floor applies to the whole new descent.
            continuous_detail = (f'continuous_min_z_footprint={floors}; '
                                 f'continuous_speeds={speeds}; continuous_frame=base_footprint_negative_Z; ')
        preview = {}
        for arm in arms:
            world = self._place_box_test_shift_box_z(predicted_world[arm], drop)
            local = BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(predicted_base[arm]), world
            )
            preview[arm] = self._endpoint_sync_transform_to_pose(local)
            preview[arm].position.z -= max(0.0, post_distance)
        detail = (
            f"{box_type} placement: waist_only=true; arm_compensation=disabled; "
            f"waist_workspace=not_used; body_target_units={target_units}; "
            f"table_z={table_z:.4f}m; clearance_samples={segments + 1}; "
            f"predicted_min_bottom_z={minimum_bottom:.4f}m; "
            "arms_hold_current_joints; box_follows_waist; "
            "descent_targets=measured_TCP_after_waist; automatic_pre_lift=disabled; "
            f"post_support_arm_base_z_descent={post_distance:.4f}m; "
            f"descent_mode={self._string('place_box_test_descent_mode')}; "
            f"{continuous_detail}"
            "preview_only=true; descent_IK_not_prevalidated"
        )
        self._publish_place_box_test_feedback(goal_handle, "PLACE_BOX_TEST_TARGETS", detail)
        if dry_run:
            return f"{detail}; skipped in dry-run", preview['left'], preview['right']

        try:
            self._wait_for_service(
                self.body_command_client, self._string("box_joint1_command_service_name"), goal_handle
            )
            self._check_canceled(goal_handle, "before waist-only placement motion")
            self._place_box_test_wait_arms_still(goal_handle)
            pre_waist_wrench = self._place_box_test_stable_common_wrench(
                goal_handle, adapter, base_frame
            )
            self._publish_place_box_test_feedback(
                goal_handle, "PLACE_PRE_WAIST_WRENCH",
                f"complete bilateral common-frame wrench recorded before waist: "
                f"left={pre_waist_wrench['left']}; right={pre_waist_wrench['right']}; "
                "SDK Work force and torque rotated onto base_link axes",
            )
            self._publish_place_box_test_feedback(
                goal_handle, "MOVING_PLACE_WAIST_ONLY",
                "moving waist only; both arms hold current joints; "
                "no arm MoveJ/MoveL/compensation commands are sent",
            )
            monitor_stop = threading.Event()
            monitor_state = {}
            monitor = threading.Thread(
                target=self._place_box_test_waist_contact_monitor,
                args=(adapter, base_frame, pre_waist_wrench, monitor_stop, monitor_state),
                name="place-waist-contact-monitor", daemon=True,
            )
            monitor.start()
            try:
                future = self.body_command_client.call_async(
                    self._place_box_test_body_request(target_units, trajectory_connect=0, blend_radius=0)
                )
                response = self._wait_future(
                    future, goal_handle, "starting waist-only placement MoveJ",
                    self._float("dependency_wait_timeout_sec"), cancel_local_future=False,
                )
                self._parse_string_command_response(response, "place waist-only MoveJ")
                self._wait_for_body_joints_target(
                    goal_handle, target_angles, sequence_after=body_sequence,
                    timeout_parameter="place_box_test_timeout_sec",
                )
            finally:
                monitor_stop.set()
                monitor.join(timeout=2.0)
            if monitor.is_alive():
                self._place_box_test_stop_body()
                raise MissionError("waist force monitor did not stop; refusing descent")
            if "error" in monitor_state:
                raise MissionError(f"waist force monitor failed: {monitor_state['error']}")
            self._place_box_test_wait_arms_still(goal_handle)
            post_waist_wrench = self._place_box_test_stable_common_wrench(
                goal_handle, adapter, base_frame
            )
            signs = {
                arm: self._float(f"place_box_test_table_support_sign_{arm}")
                for arm in arms
            }
            threshold = self._float("place_box_test_table_support_delta_fz_n")
            common_z_changes = {
                arm: signs[arm] * (post_waist_wrench[arm][2] - pre_waist_wrench[arm][2])
                for arm in arms
            }
            contact_arms = (
                [arm for arm in arms if common_z_changes[arm] >= threshold]
                if sum(common_z_changes.values()) >= threshold else []
            )
            if "contact" in monitor_state and not contact_arms:
                raise MissionError(
                    "waist-motion force indicated possible contact, but the "
                    "stationary post-waist wrench did not confirm it; "
                    f"moving_delta={monitor_state['contact']}; "
                    f"stationary_delta={common_z_changes}; refusing descent"
                )
            self._publish_place_box_test_feedback(
                goal_handle, "PLACE_POST_WAIST_CONTACT_CHECK",
                f"common-frame vertical unloading from pre-waist complete-wrench reference: "
                f"left={common_z_changes['left']:.3f}N; "
                f"right={common_z_changes['right']:.3f}N; "
                f"threshold={threshold:.3f}N; contact_arms={contact_arms}; "
                f"pre_wrench={pre_waist_wrench}; post_wrench={post_waist_wrench}",
            )
            _, measured_box = self._place_box_test_live_descent_state(base_frame, relations)
            if contact_arms:
                if descent_mode == 'continuous' and len(contact_arms) != 2:
                    raise MissionError('continuous Place requires bilateral post-waist support; '
                                       'unilateral contact detected, no descent or release')
                support_detail = (
                    "post-waist contact detected against pre-waist common-frame baseline; "
                    f"contact_arms={contact_arms}; common_z_changes={common_z_changes}; "
                    "table-search descent skipped"
                )
                self._publish_place_box_test_feedback(
                    goal_handle, "PLACE_ALREADY_CONTACTING_TABLE", support_detail
                )
            else:
                self._publish_place_box_test_feedback(
                    goal_handle, "DESCENDING_TO_TABLE",
                    "waist-only motion completed; no pre-waist-baseline contact detected; "
                    f"capturing TCP origins; descent_mode={self._string('place_box_test_descent_mode')}",
                )
                _, support_detail = self._place_box_test_descend_to_table(
                    goal_handle, adapter, base_frame, relations, measured_box, table_z, half_height
                )
            if post_distance > 0.0:
                final_pose, post_detail = self._place_box_test_post_support_arm_base_descent(
                    goal_handle, adapter, base_frame,
                    support_source="pre_waist_bilateral" if len(contact_arms) == 2
                    else "pre_waist_unilateral" if contact_arms else "descent_bilateral",
                )
                support_detail += f"; {post_detail}"
            else:
                actual, _ = self._place_box_test_live_descent_state(base_frame, relations)
                final_pose = {
                    arm: self._endpoint_sync_transform_to_pose(
                        BoxSupportMixin._compose_transform(
                            BoxSupportMixin._inverse_transform(self._lookup_tf_carry_transform(
                                base_frame, self._string(f"{arm}_arm_base_frame").strip().lstrip("/")
                            )), actual[arm],
                        )
                    )
                    for arm in arms
                }
        except (RealManSdkCanceled, MissionCanceled):
            self._place_box_test_stop_body()
            raise
        except (RealManSdkError, MissionError, ValueError) as exc:
            self._place_box_test_stop_body()
            raise MissionError(f"place_box_test motion failed: {exc}") from exc
        return f"{detail}; {support_detail}", final_pose['left'], final_pose['right']

    def _execute_force_carry_place_waist(self, goal_handle, adapter, dry_run):
        """Bend with fixed arm joints; the caller then releases and retracts."""
        if not dry_run and getattr(adapter, "_ros_movej_transport", None) is None:
            raise MissionError("force-carry placement requires ROS arm MoveJ transport")
        units = [int(round(v)) for v in self._float_array("place_box_test_body_joint_units")]
        scales = self._float_array("box_body_command_units_per_degree")
        angles = [math.radians(v / scale) for v, scale in zip(units, scales)]
        detail = (
            f"force-carry placement: waist_joint_units={units}; arms_hold_current_joints; "
            "descent=skipped; support_sensing=skipped; tool_retreat=skipped; "
            "release_then_arm_joint2=40deg; arm_movej_transport=ROS"
        )
        self._publish_place_box_test_feedback(goal_handle, "PLACE_BOX_TEST_TARGETS", detail)
        if not dry_run:
            _, _, sequence = self._wait_for_fresh_body_feedback(goal_handle)
            self._wait_for_service(
                self.body_command_client, self._string("box_joint1_command_service_name"), goal_handle
            )
            self._check_canceled(goal_handle, "before force-carry placement waist motion")
            self._publish_place_box_test_feedback(
                goal_handle, "MOVING_PLACE_WAIST_ONLY",
                "moving waist only; after arrival release and retract both Joint2 axes to 40deg",
            )
            try:
                future = self.body_command_client.call_async(
                    self._place_box_test_body_request(units, trajectory_connect=0, blend_radius=0)
                )
                response = self._wait_future(
                    future, goal_handle, "starting force-carry placement waist MoveJ",
                    self._float("dependency_wait_timeout_sec"), cancel_local_future=False,
                )
                self._parse_string_command_response(response, "force-carry placement waist MoveJ")
                self._wait_for_body_joints_target(
                    goal_handle, angles, sequence_after=sequence,
                    timeout_parameter="place_box_test_timeout_sec",
                )
            except Exception:
                self._place_box_test_stop_body()
                raise
        poses = {
            arm: self._endpoint_sync_transform_to_pose(self._lookup_tf_carry_transform(
                self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
                self._string(f"{arm}_link8_frame").strip().lstrip("/"),
            ))
            for arm in ("left", "right")
        }
        return detail, poses["left"], poses["right"]

    def _execute_place_box_test_motion(
        self, goal_handle, adapter, requested_box_type: str, dry_run: bool
    ):
        """Dispatch dynamic waist-only placement or legacy taught-pose placement.

        Dynamic placement holds arm joints while bending the waist and then
        descends from measured TCPs. Only the legacy taught-pose mode below
        interpolates a common virtual box pose with dual-arm compensation.
        """
        if bool(getattr(getattr(goal_handle, "request", None), "force_carry_profile", False)):
            return self._execute_force_carry_place_waist(goal_handle, adapter, dry_run)
        relation_by_arm = getattr(self, "_last_grasp_box_tf_box_to_link7_targets", None)
        if not relation_by_arm:
            raise MissionError(
                "place_box_test requires a preceding /grasp_box_tf goal in "
                "the same mission_controller process"
            )
        if adapter is None and not dry_run:
            raise MissionError(
                "place_box_test requires direct_motion_backend=python_sdk"
            )

        box_type = str(requested_box_type or self._string("place_box_test_box_type"))
        box_type = box_type.strip().lower()
        if box_type not in ("smallbox", "bigbox"):
            raise MissionError(
                "place_box_test box_type must be 'smallbox' or 'bigbox', "
                f"got {requested_box_type!r}"
            )

        body_start, body_velocities, body_sequence = self._wait_for_fresh_body_feedback(
            goal_handle
        )
        units_per_degree = self._float_array("box_body_command_units_per_degree")
        start_home_units = [
            int(round(value))
            for value in self._float_array("place_box_test_start_body_joint_units")
        ]
        start_home_angles = [
            math.radians(float(start_home_units[index]) / units_per_degree[index])
            for index in range(4)
        ]
        start_tolerance = self._float("place_box_test_start_body_tolerance_rad")
        if any(
            abs(body_start[index] - start_home_angles[index]) > start_tolerance
            for index in range(4)
        ):
            raise MissionError(
                "place_box_test must start with the waist at its carried-home "
                f"pose: expected={start_home_angles}, measured={body_start}, "
                f"tolerance_rad={start_tolerance:.4f}"
            )
        velocity_limit = self._float("box_joint1_velocity_tolerance_rad_sec")
        if any(abs(value) > velocity_limit for value in body_velocities):
            raise MissionError(
                "place_box_test requires a stationary waist: "
                f"velocities={body_velocities}, limit={velocity_limit:.4f}"
            )

        base_frame = self._string("grasp_box_tf_freeze_frame").strip().lstrip("/")
        actual_link = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_link8_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        inferred_current_box = {
            arm: BoxSupportMixin._compose_transform(
                actual_link[arm],
                BoxSupportMixin._inverse_transform(relation_by_arm[arm]),
            )
            for arm in ("left", "right")
        }
        current_position_error = self._endpoint_sync_pose_position_error(
            inferred_current_box["left"][0], inferred_current_box["right"][0]
        )
        current_orientation_error = self._endpoint_sync_pose_orientation_error(
            inferred_current_box["left"][1], inferred_current_box["right"][1]
        )
        consistency_position_limit = self._float(
            "place_box_test_target_consistency_position_tolerance_m"
        )
        consistency_orientation_limit = self._float(
            "place_box_test_target_consistency_orientation_tolerance_rad"
        )
        if (
            current_position_error > consistency_position_limit
            or current_orientation_error > consistency_orientation_limit
        ):
            raise MissionError(
                "saved grasp is inconsistent with the current dual-arm TF: "
                f"box_position_disagreement={current_position_error:.4f}m, "
                f"box_orientation_disagreement={current_orientation_error:.4f}rad"
            )
        current_box = BoxSupportMixin._mean_rigid_transforms(
            inferred_current_box["left"], inferred_current_box["right"]
        )
        # Re-capture the physical rigid grasp at the destination.  This
        # removes small endpoint tracking error accumulated during transport.
        relation_by_arm = {
            arm: BoxSupportMixin._compose_transform(
                BoxSupportMixin._inverse_transform(current_box), actual_link[arm]
            )
            for arm in ("left", "right")
        }
        self._last_grasp_box_tf_box_to_link7_targets = relation_by_arm

        dynamic_placement = self._boolean("place_box_test_dynamic_table_enabled")
        table_z = None
        half_height = None
        if dynamic_placement:
            table_z = self._place_box_test_table_z_in_base(base_frame)
            half_height = self._float(
                f"place_box_test_{box_type}_half_height_m"
            )
            bottom_z = self._place_box_test_bottom_z(current_box, half_height)
            required_bottom = table_z + self._float(
                "place_box_test_waist_clearance_m"
            )
            if bottom_z < required_bottom:
                raise MissionError(
                    "place_box_test box is below the required waist clearance; "
                    f"bottom_z={bottom_z:.4f}m, "
                    f"required_bottom_z={required_bottom:.4f}m"
                )
            self._publish_place_box_test_feedback(
                goal_handle, "PLACE_CLEARANCE_CHECK",
                f"box_bottom_z={bottom_z:.4f}m, table_z={table_z:.4f}m, "
                f"required_clearance={required_bottom:.4f}m; "
                "automatic_pre_lift=disabled",
            )

        target_units = [
            int(round(value))
            for value in self._float_array("place_box_test_body_joint_units")
        ]
        target_angles = [
            math.radians(float(target_units[index]) / units_per_degree[index])
            for index in range(4)
        ]
        live_arm_base = {
            arm: self._lookup_tf_carry_transform(
                base_frame,
                self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
            )
            for arm in ("left", "right")
        }
        if dynamic_placement:
            # Do not plan, queue or execute the legacy dual-arm waist-follow
            # trajectory. No arm command is sent until the waist has stopped.
            return self._execute_place_box_test_waist_only(
                goal_handle, adapter, box_type, dry_run,
                base_frame=base_frame, relations=relation_by_arm,
                actual_link=actual_link, live_arm_base=live_arm_base,
                body_start=body_start, body_sequence=body_sequence,
                target_units=target_units, target_angles=target_angles,
                table_z=table_z, half_height=half_height,
            )
        start_angles = [float(value) for value in body_start[:3]]
        future_arm_base = {}
        for arm in ("left", "right"):
            fk_current = self._joint123_arm_base_transform(arm, start_angles)
            fk_target = self._joint123_arm_base_transform(arm, target_angles[:3])
            future_arm_base[arm] = BoxSupportMixin._compose_transform(
                live_arm_base[arm],
                BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(fk_current), fk_target
                ),
            )

        configured_target = {
            arm: self._endpoint_sync_pose_values_to_transform(
                self._float_array(
                    f"place_box_test_{arm}_target_pose_arm_base_{box_type}"
                )
            )
            for arm in ("left", "right")
        } if not dynamic_placement else None
        target_world_link = {
            arm: BoxSupportMixin._compose_transform(
                future_arm_base[arm], configured_target[arm]
            )
            for arm in ("left", "right")
        } if not dynamic_placement else None
        inferred_target_box = {
            arm: BoxSupportMixin._compose_transform(
                target_world_link[arm],
                BoxSupportMixin._inverse_transform(relation_by_arm[arm]),
            )
            for arm in ("left", "right")
        } if not dynamic_placement else None
        target_position_error = self._endpoint_sync_pose_position_error(
            inferred_target_box["left"][0], inferred_target_box["right"][0]
        ) if not dynamic_placement else 0.0
        target_orientation_error = self._endpoint_sync_pose_orientation_error(
            inferred_target_box["left"][1], inferred_target_box["right"][1]
        ) if not dynamic_placement else 0.0
        if (
            target_position_error > consistency_position_limit
            or target_orientation_error > consistency_orientation_limit
        ):
            raise MissionError(
                "taught left/right place poses do not describe one rigid box pose: "
                f"position_disagreement={target_position_error:.4f}m, "
                f"orientation_disagreement={target_orientation_error:.4f}rad, "
                f"limits=[{consistency_position_limit:.4f}m,"
                f"{consistency_orientation_limit:.4f}rad]"
            )
        # Dynamic placement starts from the measured box pose.  The waist
        # search may shift the entire rigid box in common-frame Y/Z while
        # bending, then both arms descend only after the waist is stationary.
        target_box = (
            current_box if dynamic_placement
            else BoxSupportMixin._mean_rigid_transforms(
                inferred_target_box["left"], inferred_target_box["right"]
            )
        )

        segments = self._integer("place_box_test_segments")
        workspace_detail = "waist_workspace=not_applicable"
        if dynamic_placement:
            target_box, workspace_detail = (
                self._place_box_test_optimize_waist_box_path(
                    goal_handle, adapter, current_box, relation_by_arm,
                    body_start, target_units, units_per_degree, live_arm_base,
                    segments, table_z, half_height,
                )
            )
        (
            body_targets, arm_targets, arm_target_transforms,
            world_targets_by_segment,
        ) = self._place_box_test_waist_path(
            current_box, target_box, relation_by_arm, body_start,
            target_units, units_per_degree, live_arm_base, segments,
        )

        arm_motion_mode = self._string("place_box_test_arm_motion_mode").strip().lower()
        if arm_motion_mode == "movel_offset":
            initial_local_link = {
                arm: BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(live_arm_base[arm]),
                    actual_link[arm],
                )
                for arm in ("left", "right")
            }
            # rm_movel_offset connected waypoints are incremental: each
            # offset starts at the previous queued endpoint.
            for arm in ("left", "right"):
                previous = initial_local_link[arm]
                arm_targets[arm] = []
                for local_target in arm_target_transforms[arm]:
                    arm_targets[arm].append(
                        self._place_box_test_work_frame_offset(
                            previous, local_target
                        )
                    )
                    previous = local_target

        planned_place_box = (
            self._place_box_test_shift_box_z(
                target_box,
                table_z - self._place_box_test_bottom_z(target_box, half_height),
            )
            if dynamic_placement else target_box
        )
        planned_place_world_targets = (
            self._place_box_test_box_world_targets(
                planned_place_box, relation_by_arm
            )
            if dynamic_placement else world_targets_by_segment[-1]
        )
        final_pose = {
            arm: self._endpoint_sync_transform_to_pose(
                BoxSupportMixin._compose_transform(
                    BoxSupportMixin._inverse_transform(future_arm_base[arm]),
                    planned_place_world_targets[arm],
                )
            )
            for arm in ("left", "right")
        }
        post_support_descent = (
            self._float("place_box_test_post_support_arm_base_descent_m")
            if dynamic_placement else 0.0
        )
        if post_support_descent > 0.0:
            for arm in ("left", "right"):
                final_pose[arm].position.z -= post_support_descent
        table_detail = (
            f"table_z={table_z:.4f}m; automatic_pre_lift=disabled; "
            if dynamic_placement else ""
        )
        planning_detail = (
            f"{box_type} placement path prepared: segments={segments}; "
            f"body_target_units={target_units}; "
            f"{table_detail}"
            f"post_support_arm_base_z_descent={post_support_descent:.4f}m; "
            f"{workspace_detail}; "
            f"left_target=[{final_pose['left'].position.x:.3f},"
            f"{final_pose['left'].position.y:.3f},"
            f"{final_pose['left'].position.z:.3f}]; "
            f"right_target=[{final_pose['right'].position.x:.3f},"
            f"{final_pose['right'].position.y:.3f},"
            f"{final_pose['right'].position.z:.3f}]; "
            f"arm_motion_mode={arm_motion_mode}; "
            f"box_pose={'rigid_yz_waist_path_then_table_descent' if dynamic_placement else 'single_rigid_interpolation'}"
        )
        self._publish_place_box_test_feedback(
            goal_handle, "PLACE_BOX_TEST_TARGETS", planning_detail
        )
        if dry_run:
            return (
                f"{planning_detail}; skipped in dry-run",
                final_pose["left"],
                final_pose["right"],
            )

        force_unload_enabled = not dynamic_placement and self._boolean(
            "place_box_test_force_unload_enabled"
        )
        force_baselines = None
        force_baseline_sequences = None
        force_monitor = None
        if force_unload_enabled:
            self._publish_place_box_test_feedback(
                goal_handle,
                "CAPTURING_LOADED_FZ_BASELINE",
                "capturing fresh bilateral force-Z baselines while the box is held; "
                "force-X is not used for placement release",
            )
            force_baselines, force_baseline_sequences = (
                self._collect_place_box_test_fz_baseline(goal_handle)
            )
            self._publish_place_box_test_feedback(
                goal_handle,
                "LOADED_FZ_BASELINE_READY",
                f"left_Fz0={force_baselines['left']:.1f}, "
                f"right_Fz0={force_baselines['right']:.1f}",
            )
            force_monitor = self._new_place_box_test_fz_unload_monitor(
                force_baselines, force_baseline_sequences
            )

        service_name = self._string("box_joint1_command_service_name")
        self._wait_for_service(self.body_command_client, service_name, goal_handle)
        body_blend_radius = self._integer("place_box_test_body_blend_radius")
        for index in range(max(0, segments - 1)):
            self._check_canceled(goal_handle, "while queuing place_box_test waist path")
            future = self.body_command_client.call_async(
                self._place_box_test_body_request(
                    body_targets[index][0],
                    trajectory_connect=1,
                    blend_radius=body_blend_radius,
                )
            )
            response = self._wait_future(
                future,
                goal_handle,
                f"queuing place_box_test waist waypoint {index + 1}",
                self._float("dependency_wait_timeout_sec"),
                cancel_local_future=False,
            )
            self._parse_string_command_response(
                response, f"place_box_test waist waypoint {index + 1} MoveJ"
            )

        final_body_future = {}

        def release_body():
            final_body_future["future"] = self.body_command_client.call_async(
                self._place_box_test_body_request(
                    body_targets[-1][0],
                    trajectory_connect=0,
                    blend_radius=0,
                )
            )

        last_monitor_time = [0.0]
        support_feedback_published = [False]

        def monitor_rigid_grasp():
            if force_monitor is not None:
                if self._update_place_box_test_fz_unload_monitor(force_monitor):
                    if not support_feedback_published[0]:
                        support_feedback_published[0] = True
                        self._publish_place_box_test_feedback(
                            goal_handle,
                            "TABLE_SUPPORT_CONFIRMED_IN_MOTION",
                            force_monitor["detail"]
                            + "; stopping waist and both arms before release",
                        )
                    # A False return requests a coordinated *successful* stop
                    # from the SDK adapter and its body abort callback.
                    return False
            now = time.monotonic()
            if now - last_monitor_time[0] < 0.1:
                return True
            last_monitor_time[0] = now
            try:
                actual = {
                    arm: self._lookup_tf_carry_transform(
                        base_frame,
                        self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                    )
                    for arm in ("left", "right")
                }
            except MissionError:
                return True
            inferred = {
                arm: BoxSupportMixin._compose_transform(
                    actual[arm],
                    BoxSupportMixin._inverse_transform(relation_by_arm[arm]),
                )
                for arm in ("left", "right")
            }
            if dynamic_placement:
                measured_box = BoxSupportMixin._mean_rigid_transforms(
                    inferred["left"], inferred["right"]
                )
                if self._place_box_test_bottom_z(measured_box, half_height) < (
                    table_z + self._float("place_box_test_waist_clearance_m") * 0.5
                ):
                    raise MissionError(
                        "place_box_test box clearance fell during waist bend; "
                        "stopping before table contact"
                    )
            position_error = self._endpoint_sync_pose_position_error(
                inferred["left"][0], inferred["right"][0]
            )
            orientation_error = self._endpoint_sync_pose_orientation_error(
                inferred["left"][1], inferred["right"][1]
            )
            if position_error > consistency_position_limit:
                raise MissionError(
                    "place_box_test rigid-grasp position disagreement: "
                    f"{position_error:.4f}m"
                )
            if orientation_error > consistency_orientation_limit:
                raise MissionError(
                    "place_box_test rigid-grasp orientation disagreement: "
                    f"{orientation_error:.4f}rad"
                )
            return True

        self._publish_place_box_test_feedback(
            goal_handle,
            "MOVING_TO_PLACE_POSE",
            f"starting connected waist MoveJ and dual-arm SDK {arm_motion_mode}; "
            "intermediate trajectory_connect=1, final=0",
        )
        try:
            motion_result = adapter.execute_dual_movel_connected_waypoints(
                arm_targets["left"],
                arm_targets["right"],
                self._float("place_box_test_left_movel_velocity_percent"),
                self._float("place_box_test_right_movel_velocity_percent"),
                blend_radius=self._integer("place_box_test_arm_blend_radius"),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=self._float("place_box_test_timeout_sec"),
                before_start=release_body,
                abort_callback=self._place_box_test_stop_body,
                progress_callback=monitor_rigid_grasp,
                motion_mode=arm_motion_mode,
                offset_frame_type=self._integer(
                    "place_box_test_arm_offset_frame_type"
                ),
            )
            if "future" not in final_body_future:
                raise MissionError(
                    "place_box_test final waist command was not released"
                )
            response = self._wait_future(
                final_body_future["future"],
                goal_handle,
                "starting place_box_test final waist MoveJ",
                self._float("dependency_wait_timeout_sec"),
                cancel_local_future=False,
            )
            self._parse_string_command_response(
                response, "place_box_test final waist MoveJ"
            )
            force_stopped = bool(force_monitor and force_monitor["confirmed"])
            if force_unload_enabled and not force_stopped:
                raise MissionError(
                    "place_box_test reached the geometric endpoint without "
                    "bilateral force-Z table-support confirmation; Joint2 "
                    "release was not executed"
                )
            if force_stopped:
                self._place_box_test_stop_body()
                live_final_arm_base = {
                    arm: self._lookup_tf_carry_transform(
                        base_frame,
                        self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
                    )
                    for arm in ("left", "right")
                }
                actual_final_link = {
                    arm: self._lookup_tf_carry_transform(
                        base_frame,
                        self._string(f"{arm}_link8_frame").strip().lstrip("/"),
                    )
                    for arm in ("left", "right")
                }
                z_equalization_detail = "post_support_z_equalization=disabled"
                if self._boolean(
                    "place_box_test_post_support_z_equalization_enabled"
                ):
                    (
                        live_final_arm_base,
                        actual_final_link,
                        z_equalization_detail,
                    ) = self._execute_place_box_test_post_support_z_equalization(
                        goal_handle,
                        adapter,
                        base_frame,
                        force_monitor,
                    )
                final_pose = {
                    arm: self._endpoint_sync_transform_to_pose(
                        BoxSupportMixin._compose_transform(
                            BoxSupportMixin._inverse_transform(
                                live_final_arm_base[arm]
                            ),
                            actual_final_link[arm],
                        )
                    )
                    for arm in ("left", "right")
                }
                verification = "geometric_final_guard=skipped_after_force_stop"
                force_verification = (
                    f"{force_monitor['detail']}; {z_equalization_detail}"
                )
            else:
                self._wait_for_body_joints_target(
                    goal_handle,
                    body_targets[-1][1],
                    sequence_after=body_sequence,
                    timeout_parameter="place_box_test_timeout_sec",
                )
            if not force_stopped:
                if dynamic_placement:
                    verification = "post_waist_arm_target_check=skipped"
                else:
                    verification = self._wait_for_place_box_test_world_targets(
                        goal_handle, world_targets_by_segment[-1]
                    )
                force_verification = "force_z_table_support=disabled"
            if dynamic_placement:
                self._publish_place_box_test_feedback(
                    goal_handle, "DESCENDING_TO_TABLE",
                    "waist reached target; lowering each measured TCP equally "
                    "in common base_link -Z; XY/orientation held at actual values; "
                    "no post-waist height correction or per-step TF target guard; "
                    "SDK Work-Fz support checks remain active",
                )
                _, measured_box = self._place_box_test_live_descent_state(
                    base_frame, relation_by_arm
                )
                placed_box, force_verification = self._place_box_test_descend_to_table(
                    goal_handle, adapter, base_frame, relation_by_arm,
                    measured_box, table_z, half_height,
                )
                if post_support_descent > 0.0:
                    final_pose, descent_detail = (
                        self._place_box_test_post_support_arm_base_descent(
                            goal_handle, adapter, base_frame
                        )
                    )
                    force_verification += f"; {descent_detail}"
                else:
                    placed_world_targets, _ = self._place_box_test_live_descent_state(
                        base_frame, relation_by_arm
                    )
                    live_final_arm_base = {
                        arm: self._lookup_tf_carry_transform(
                            base_frame,
                            self._string(f"{arm}_arm_base_frame").strip().lstrip("/"),
                        )
                        for arm in ("left", "right")
                    }
                    final_pose = {
                        arm: self._endpoint_sync_transform_to_pose(
                            BoxSupportMixin._compose_transform(
                                BoxSupportMixin._inverse_transform(
                                    live_final_arm_base[arm]
                                ),
                                placed_world_targets[arm],
                            )
                        )
                        for arm in ("left", "right")
                    }
        except (RealManSdkCanceled, MissionCanceled):
            self._place_box_test_stop_body()
            raise
        except (RealManSdkError, MissionError, ValueError) as exc:
            self._place_box_test_stop_body()
            raise MissionError(f"place_box_test motion failed: {exc}") from exc

        return (
            f"{planning_detail}; {motion_result}; final_guard={verification}; "
            f"{force_verification}",
            final_pose["left"],
            final_pose["right"],
        )

    def _execute_place_box_test_post_release(self, goal_handle, adapter, dry_run: bool):
        """Retract both tools after release, then move Joint2 and home."""
        force_carry_profile = bool(getattr(getattr(goal_handle, "request", None), "force_carry_profile", False))
        retreat_enabled = self._boolean(
            "place_box_test_post_release_tool_y_retreat_enabled"
        )
        arm_enabled = self._boolean(
            "place_box_test_post_release_arm_movej_enabled"
        )
        body_enabled = self._boolean(
            "place_box_test_post_release_body_home_enabled"
        )
        arm_home_enabled = self._boolean(
            "place_box_test_post_release_arm_home_enabled"
        )
        if force_carry_profile:
            retreat_enabled = False
            arm_enabled = True
            arm_home_enabled = True
        if not retreat_enabled and not arm_enabled and not body_enabled and not arm_home_enabled:
            return "post-release arm/body motion disabled"

        joint2_deg = self._float(
            "place_box_test_post_release_arm_joint2_angle_deg"
        )
        if force_carry_profile:
            joint2_deg = 40.0
        home_units = [
            int(round(value))
            for value in self._float_array(
                "place_box_test_post_release_body_home_joint_units"
            )
        ]
        if dry_run:
            return (
                "post-release motion skipped in dry-run: "
                f"tool_y_retreat_enabled={retreat_enabled}; "
                f"tool_y_retreat_distance_m="
                f"{self._float('place_box_test_post_release_tool_y_retreat_m'):.4f}; "
                f"arm_joint2_target_deg={joint2_deg:.3f}; "
                f"body_home_joint_units={home_units}"
            )
        if adapter is None:
            raise MissionError(
                "place_box_test post-release arm MoveJ requires "
                "direct_motion_backend=python_sdk"
            )

        details = []
        if retreat_enabled:
            distance = self._float("place_box_test_post_release_tool_y_retreat_m")
            speed = self._float(
                "place_box_test_post_release_tool_y_retreat_velocity_percent"
            )
            self._publish_place_box_test_feedback(
                goal_handle,
                "POST_RELEASE_TOOL_Y_RETREAT",
                f"retracting both arms after gripper release: "
                f"left=-Tool Y {distance:.4f}m, "
                f"right=+Tool Y {distance:.4f}m; SDK speed={speed:.1f}%",
            )
            try:
                retreat_result = adapter.execute_dual(
                    [0.0, -distance, 0.0, 0.0, 0.0, 0.0],
                    [0.0, distance, 0.0, 0.0, 0.0, 0.0],
                    "movel_offset",
                    speed,
                    True,
                    cancel_requested=lambda: goal_handle.is_cancel_requested,
                    timeout_sec=self._float(
                        "place_box_test_post_release_tool_y_retreat_timeout_sec"
                    ),
                    offset_frame_type=1,
                )
            except RealManSdkCanceled as exc:
                raise MissionCanceled(
                    "place_box_test canceled during post-release Tool-Y retreat"
                ) from exc
            except RealManSdkError as exc:
                raise MissionError(
                    f"place_box_test post-release Tool-Y retreat failed: {exc}; "
                    "Joint2 was not commanded"
                ) from exc
            details.append(
                f"tool_y_retreat_m={distance:.4f}; {retreat_result}"
            )
        if arm_enabled:
            max_age = self._float(
                "place_box_test_post_release_arm_movej_feedback_max_age_sec"
            )
            with self.joint_state_lock:
                positions = {
                    arm: list(self.latest_slave_arm_positions.get(arm, []))
                    for arm in ("left", "right")
                }
                state_times = dict(self.latest_slave_arm_state_times)
                sequence_before = dict(self.latest_slave_arm_state_sequences)
            now = time.monotonic()
            for arm in ("left", "right"):
                if (
                    len(positions[arm]) < 7
                    or not all(
                        value is not None and math.isfinite(float(value))
                        for value in positions[arm][:7]
                    )
                ):
                    raise MissionError(
                        f"place_box_test post-release {arm} feedback does not "
                        "contain seven finite joints"
                    )
                age = now - state_times.get(arm, 0.0)
                if age > max_age:
                    raise MissionError(
                        f"place_box_test post-release {arm} feedback is stale: "
                        f"age={age:.3f}s, limit={max_age:.3f}s"
                    )
            target_rad = {
                arm: [float(value) for value in positions[arm][:7]]
                for arm in ("left", "right")
            }
            for arm in ("left", "right"):
                target_rad[arm][1] = math.radians(joint2_deg)
            target_deg = {
                arm: [math.degrees(value) for value in target_rad[arm]]
                for arm in ("left", "right")
            }
            self._publish_place_box_test_feedback(
                goal_handle,
                "POST_RELEASE_ARM_MOVEJ",
                "moving both arm Joint2 axes to "
                f"{joint2_deg:.3f} deg while preserving Joints1,3-7",
            )
            movej_result = adapter.execute_dual_movej(
                target_deg["left"],
                target_deg["right"],
                self._float(
                    "place_box_test_post_release_arm_movej_velocity_percent"
                ),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=self._float(
                    "place_box_test_post_release_arm_movej_timeout_sec"
                ),
            )
            self._wait_for_post_arm_joint_targets(
                goal_handle,
                target_rad["left"],
                target_rad["right"],
                sequence_before,
                parameter_prefix="place_box_test_post_release_arm_movej",
                description="place_box_test post-release dual-arm MoveJ",
            )
            details.append(
                f"arm_joint2_target_deg={joint2_deg:.3f}; {movej_result}"
            )

        if body_enabled:
            _, _, body_sequence = self._wait_for_fresh_body_feedback(goal_handle)
            units_per_degree = self._float_array(
                "box_body_command_units_per_degree"
            )
            home_angles = [
                math.radians(float(home_units[index]) / units_per_degree[index])
                for index in range(4)
            ]
            self._publish_place_box_test_feedback(
                goal_handle,
                "POST_RELEASE_BODY_HOME",
                f"returning body joints to {home_units} at velocity "
                f"{self._integer('place_box_test_body_velocity')}",
            )
            service_name = self._string("box_joint1_command_service_name")
            self._wait_for_service(
                self.body_command_client, service_name, goal_handle
            )
            future = self.body_command_client.call_async(
                self._place_box_test_body_request(
                    home_units,
                    trajectory_connect=0,
                    blend_radius=0,
                )
            )
            response = self._wait_future(
                future,
                goal_handle,
                "starting place_box_test post-release body home MoveJ",
                self._float("dependency_wait_timeout_sec"),
                cancel_local_future=False,
            )
            self._parse_string_command_response(
                response, "place_box_test post-release body home MoveJ"
            )
            self._wait_for_body_joints_target(
                goal_handle,
                home_angles,
                sequence_after=body_sequence,
                timeout_parameter="place_box_test_timeout_sec",
            )
            details.append(f"body_home_joint_units={home_units}")

        if arm_home_enabled:
            units_per_degree = self._float(
                "place_box_test_post_release_arm_home_command_units_per_degree"
            )
            left_units = [
                int(round(value))
                for value in self._float_array(
                    "place_box_test_post_release_arm_home_left_joint_units"
                )
            ]
            right_units = [
                int(round(value))
                for value in self._float_array(
                    "place_box_test_post_release_arm_home_right_joint_units"
                )
            ]
            if len(left_units) != 7 or len(right_units) != 7:
                raise MissionError(
                    "place_box_test post-release arm home targets must contain seven joints"
                )
            if not math.isfinite(units_per_degree) or units_per_degree <= 0.0:
                raise MissionError(
                    "place_box_test post-release arm home units-per-degree must be positive"
                )
            target_rad = {
                "left": [
                    math.radians(float(value) / units_per_degree)
                    for value in left_units
                ],
                "right": [
                    math.radians(float(value) / units_per_degree)
                    for value in right_units
                ],
            }
            target_deg = {
                arm: [math.degrees(value) for value in target_rad[arm]]
                for arm in ("left", "right")
            }
            with self.joint_state_lock:
                sequence_before = dict(self.latest_slave_arm_state_sequences)
            self._publish_place_box_test_feedback(
                goal_handle,
                "POST_RELEASE_ARM_HOME",
                "returning both arms to configured full-home MoveJ targets "
                f"left={left_units}, right={right_units}",
            )
            movej_result = adapter.execute_dual_movej(
                target_deg["left"],
                target_deg["right"],
                self._float(
                    "place_box_test_post_release_arm_home_velocity_percent"
                ),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
                timeout_sec=self._float(
                    "place_box_test_post_release_arm_home_timeout_sec"
                ),
            )
            self._wait_for_post_arm_joint_targets(
                goal_handle,
                target_rad["left"],
                target_rad["right"],
                sequence_before,
                parameter_prefix="place_box_test_post_release_arm_home",
                description="place_box_test post-release full arm home MoveJ",
            )
            details.append(f"arm_home={movej_result}")
            # A released box must never leave stale rigid grasp state that a
            # later placement could accidentally reuse.
            self._last_grasp_box_tf_box_to_link7_targets = None
            self._last_grasp_box_tf_box_pose = None

        return "; ".join(details)
