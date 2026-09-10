from types import SimpleNamespace

import numpy as np
import pytest

from stereo_depth_bridge.calibration import (
    CalibrationError,
    baseline_from_projections,
    build_rectification,
)


def _info(width=640, height=480, tx=0.0, distortion_model="plumb_bob"):
    fx = 500.0
    fy = 500.0
    cx = width / 2.0
    cy = height / 2.0
    return SimpleNamespace(
        width=width,
        height=height,
        distortion_model=distortion_model,
        k=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        d=[0.0, 0.0, 0.0, 0.0, 0.0],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


def test_baseline_is_decoded_from_projection_matrices():
    left = np.asarray(_info().p).reshape(3, 4)
    right = np.asarray(_info(tx=-60.0).p).reshape(3, 4)
    assert baseline_from_projections(left, right) == pytest.approx(0.12)


def test_auto_uses_complete_camera_info_stereo_calibration():
    result = build_rectification(
        _info(),
        _info(tx=-60.0),
        (640, 480),
        calibration_source="auto",
        rotation_left_to_right=np.eye(3).reshape(-1),
        translation_left_to_right_m=[0.0, 0.0, 0.0],
    )
    assert result.source == "camera_info"
    assert result.baseline_m == pytest.approx(0.12)
    assert result.left_map_x.shape == (480, 640)
    assert result.right_map_y.shape == (480, 640)


def test_auto_falls_back_to_fixed_extrinsic():
    result = build_rectification(
        _info(),
        _info(),
        (640, 480),
        calibration_source="auto",
        rotation_left_to_right=np.eye(3).reshape(-1),
        translation_left_to_right_m=[-0.12, 0.0, 0.0],
    )
    assert result.source == "fixed"
    assert result.baseline_m == pytest.approx(0.12)
    assert result.right_projection[0, 3] < 0.0


def test_placeholder_extrinsic_is_rejected():
    with pytest.raises(CalibrationError, match="placeholder"):
        build_rectification(
            _info(),
            _info(),
            (640, 480),
            calibration_source="fixed",
            rotation_left_to_right=np.eye(3).reshape(-1),
            translation_left_to_right_m=[0.0, 0.0, 0.0],
        )


def test_mismatched_camera_info_resolution_is_rejected():
    with pytest.raises(CalibrationError, match="do not match"):
        build_rectification(
            _info(width=320),
            _info(width=320, tx=-30.0),
            (640, 480),
            calibration_source="camera_info",
            rotation_left_to_right=np.eye(3).reshape(-1),
            translation_left_to_right_m=[0.0, 0.0, 0.0],
        )


def test_equidistant_camera_info_is_supported():
    left = _info(distortion_model="equidistant")
    right = _info(tx=-60.0, distortion_model="equidistant")
    left.d = [0.0, 0.0, 0.0, 0.0]
    right.d = [0.0, 0.0, 0.0, 0.0]
    result = build_rectification(
        left,
        right,
        (640, 480),
        calibration_source="camera_info",
        rotation_left_to_right=np.eye(3).reshape(-1),
        translation_left_to_right_m=[0.0, 0.0, 0.0],
    )
    assert result.source == "camera_info"
    assert result.left_map_x.shape == (480, 640)


def test_vertical_only_fixed_baseline_is_rejected():
    with pytest.raises(CalibrationError, match="horizontal"):
        build_rectification(
            _info(),
            _info(),
            (640, 480),
            calibration_source="fixed",
            rotation_left_to_right=np.eye(3).reshape(-1),
            translation_left_to_right_m=[0.0, -0.12, 0.0],
        )
