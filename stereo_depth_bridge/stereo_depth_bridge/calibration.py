"""Pure stereo-calibration helpers shared by the ROS node and tests."""

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


class CalibrationError(ValueError):
    """Raised when metric stereo depth cannot be produced safely."""


@dataclass(frozen=True)
class Rectification:
    """OpenCV remap tables and rectified projection matrices."""

    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    left_projection: np.ndarray
    right_projection: np.ndarray
    source: str
    baseline_m: float


def matrix(values: Sequence[float], rows: int, cols: int, name: str) -> np.ndarray:
    """Convert a flat ROS parameter or CameraInfo field to a finite matrix."""
    result = np.asarray(values, dtype=np.float64)
    if result.size != rows * cols:
        raise CalibrationError(
            f"{name} must contain {rows * cols} values, got {result.size}"
        )
    result = result.reshape(rows, cols)
    if not np.all(np.isfinite(result)):
        raise CalibrationError(f"{name} contains non-finite values")
    return result


def camera_info_matrices(info) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract K, D, R and P from a sensor_msgs/CameraInfo-like object."""
    K = matrix(info.k, 3, 3, "CameraInfo.k")
    D = np.asarray(info.d, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(D)):
        raise CalibrationError("CameraInfo.d contains non-finite values")
    R = matrix(info.r, 3, 3, "CameraInfo.r")
    P = matrix(info.p, 3, 4, "CameraInfo.p")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise CalibrationError("CameraInfo.k must contain positive focal lengths")
    return K, D, R, P


def distortion_model(info) -> str:
    """Normalize the ROS distortion-model name used by OpenCV."""
    model = str(getattr(info, "distortion_model", "") or "plumb_bob").lower()
    if model in {"plumb_bob", "rational_polynomial"}:
        return "pinhole"
    if model == "equidistant":
        return "fisheye"
    raise CalibrationError(f"unsupported distortion model: {model}")


def baseline_from_projections(
    left_projection: np.ndarray, right_projection: np.ndarray
) -> float:
    """Return the rectified metric baseline encoded by two ROS P matrices."""
    left = np.asarray(left_projection, dtype=np.float64).reshape(3, 4)
    right = np.asarray(right_projection, dtype=np.float64).reshape(3, 4)
    if left[0, 0] <= 0.0 or right[0, 0] <= 0.0:
        raise CalibrationError("projection matrices require positive fx")
    left_tx_m = left[0, 3] / left[0, 0]
    right_tx_m = right[0, 3] / right[0, 0]
    return abs(float(right_tx_m - left_tx_m))


def _validate_image_size(info, image_size: tuple[int, int], name: str) -> None:
    width, height = image_size
    info_width = int(getattr(info, "width", 0))
    info_height = int(getattr(info, "height", 0))
    if info_width not in (0, width) or info_height not in (0, height):
        raise CalibrationError(
            f"{name} dimensions {info_width}x{info_height} do not match "
            f"image dimensions {width}x{height}"
        )


def _maps(
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    image_size: tuple[int, int],
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    if model == "fisheye":
        if D.size != 4:
            raise CalibrationError(
                "equidistant CameraInfo.d must contain exactly four values"
            )
        return cv2.fisheye.initUndistortRectifyMap(
            K,
            D,
            R,
            P[:, :3],
            image_size,
            cv2.CV_32FC1,
        )
    return cv2.initUndistortRectifyMap(
        K,
        D,
        R,
        P[:, :3],
        image_size,
        cv2.CV_32FC1,
    )


def _from_camera_info(
    left_info,
    right_info,
    image_size: tuple[int, int],
    min_baseline_m: float,
) -> Rectification:
    left_K, left_D, left_R, left_P = camera_info_matrices(left_info)
    right_K, right_D, right_R, right_P = camera_info_matrices(right_info)
    left_model = distortion_model(left_info)
    right_model = distortion_model(right_info)
    if left_model != right_model:
        raise CalibrationError("left/right cameras use different distortion models")
    baseline_m = baseline_from_projections(left_P, right_P)
    if baseline_m < min_baseline_m:
        raise CalibrationError(
            "CameraInfo projection matrices do not contain a valid stereo baseline"
        )
    left_map_x, left_map_y = _maps(
        left_K, left_D, left_R, left_P, image_size, left_model
    )
    right_map_x, right_map_y = _maps(
        right_K, right_D, right_R, right_P, image_size, right_model
    )
    return Rectification(
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
        left_projection=left_P,
        right_projection=right_P,
        source="camera_info",
        baseline_m=baseline_m,
    )


def _from_fixed_extrinsic(
    left_info,
    right_info,
    image_size: tuple[int, int],
    rotation_left_to_right: Sequence[float],
    translation_left_to_right_m: Sequence[float],
    min_baseline_m: float,
    alpha: float,
) -> Rectification:
    left_K, left_D, _, _ = camera_info_matrices(left_info)
    right_K, right_D, _, _ = camera_info_matrices(right_info)
    left_model = distortion_model(left_info)
    right_model = distortion_model(right_info)
    if left_model != right_model:
        raise CalibrationError("left/right cameras use different distortion models")
    rotation = matrix(
        rotation_left_to_right, 3, 3, "rotation_left_to_right"
    )
    translation = np.asarray(translation_left_to_right_m, dtype=np.float64).reshape(-1)
    if translation.size != 3 or not np.all(np.isfinite(translation)):
        raise CalibrationError(
            "translation_left_to_right_m must contain three finite values"
        )
    baseline_m = float(np.linalg.norm(translation))
    if baseline_m < min_baseline_m:
        raise CalibrationError(
            "fixed stereo extrinsic is still a placeholder: provide a non-zero "
            "translation_left_to_right_m"
        )
    determinant = float(np.linalg.det(rotation))
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4) or not np.isclose(
        determinant, 1.0, atol=1e-4
    ):
        raise CalibrationError("rotation_left_to_right must be a proper rotation matrix")

    if left_model == "fisheye":
        left_R, right_R, left_P, right_P, _ = cv2.fisheye.stereoRectify(
            left_K,
            left_D,
            right_K,
            right_D,
            image_size,
            rotation,
            translation.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=image_size,
            balance=float(alpha),
            fov_scale=1.0,
        )
    else:
        left_R, right_R, left_P, right_P, _, _, _ = cv2.stereoRectify(
            left_K,
            left_D,
            right_K,
            right_D,
            image_size,
            rotation,
            translation.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=float(alpha),
        )
    left_map_x, left_map_y = _maps(
        left_K, left_D, left_R, left_P, image_size, left_model
    )
    right_map_x, right_map_y = _maps(
        right_K, right_D, right_R, right_P, image_size, right_model
    )
    encoded_baseline_m = baseline_from_projections(left_P, right_P)
    if encoded_baseline_m < min_baseline_m:
        raise CalibrationError(
            "fixed extrinsic does not produce horizontal stereo disparity"
        )
    return Rectification(
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
        left_projection=left_P,
        right_projection=right_P,
        source="fixed",
        baseline_m=encoded_baseline_m,
    )


def build_rectification(
    left_info,
    right_info,
    image_size: tuple[int, int],
    *,
    calibration_source: str,
    rotation_left_to_right: Sequence[float],
    translation_left_to_right_m: Sequence[float],
    min_baseline_m: float = 1e-4,
    alpha: float = 0.0,
) -> Rectification:
    """Build rectification maps from CameraInfo or a configured fixed transform.

    ``rotation_left_to_right`` and ``translation_left_to_right_m`` follow OpenCV's
    stereo convention: ``X_right = R * X_left + T``. For a parallel pair whose
    right camera is physically on the left camera's +X side, T is normally
    ``[-baseline, 0, 0]``.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise CalibrationError(f"invalid image dimensions {width}x{height}")
    _validate_image_size(left_info, image_size, "left CameraInfo")
    _validate_image_size(right_info, image_size, "right CameraInfo")

    source = str(calibration_source).strip().lower()
    if source not in {"auto", "camera_info", "fixed"}:
        raise CalibrationError(
            "calibration_source must be one of: auto, camera_info, fixed"
        )
    if min_baseline_m <= 0.0:
        raise CalibrationError("min_baseline_m must be positive")

    if source in {"auto", "camera_info"}:
        try:
            return _from_camera_info(
                left_info, right_info, image_size, float(min_baseline_m)
            )
        except CalibrationError:
            if source == "camera_info":
                raise

    return _from_fixed_extrinsic(
        left_info,
        right_info,
        image_size,
        rotation_left_to_right,
        translation_left_to_right_m,
        float(min_baseline_m),
        float(alpha),
    )
