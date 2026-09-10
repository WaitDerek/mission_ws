"""Synchronize, decode and rectify the robot's two compressed RGB streams."""

from __future__ import annotations

import hashlib

import cv2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from .calibration import CalibrationError, Rectification, build_rectification
from .contracts import (
    LEFT_CAMERA_INFO_TOPIC,
    LEFT_COMPRESSED_TOPIC,
    LEFT_RECTIFIED_INFO_TOPIC,
    LEFT_RECTIFIED_TOPIC,
    RIGHT_CAMERA_INFO_TOPIC,
    RIGHT_COMPRESSED_TOPIC,
    RIGHT_RECTIFIED_INFO_TOPIC,
    RIGHT_RECTIFIED_TOPIC,
)


class StereoRectifier(Node):
    """Prepare a metric, rectified stereo pair for ESS/FoundationStereo."""

    def __init__(self) -> None:
        super().__init__("stereo_rectifier")
        self._declare_parameters()
        self._rectification: Rectification | None = None
        self._calibration_signature = ""
        self._last_error = ""

        left_image = Subscriber(
            self,
            CompressedImage,
            self._string("left_compressed_topic"),
            qos_profile=qos_profile_sensor_data,
        )
        left_info = Subscriber(
            self,
            CameraInfo,
            self._string("left_camera_info_topic"),
            qos_profile=qos_profile_sensor_data,
        )
        right_image = Subscriber(
            self,
            CompressedImage,
            self._string("right_compressed_topic"),
            qos_profile=qos_profile_sensor_data,
        )
        right_info = Subscriber(
            self,
            CameraInfo,
            self._string("right_camera_info_topic"),
            qos_profile=qos_profile_sensor_data,
        )
        self._synchronizer = ApproximateTimeSynchronizer(
            [left_image, left_info, right_image, right_info],
            queue_size=self._integer("sync_queue_size"),
            slop=self._float("sync_slop_sec"),
        )
        self._synchronizer.registerCallback(self._on_stereo)

        self._left_image_publisher = self.create_publisher(
            Image, self._string("left_rectified_topic"), qos_profile_sensor_data
        )
        self._left_info_publisher = self.create_publisher(
            CameraInfo,
            self._string("left_rectified_info_topic"),
            qos_profile_sensor_data,
        )
        self._right_image_publisher = self.create_publisher(
            Image, self._string("right_rectified_topic"), qos_profile_sensor_data
        )
        self._right_info_publisher = self.create_publisher(
            CameraInfo,
            self._string("right_rectified_info_topic"),
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "waiting for synchronized left/right compressed RGB and CameraInfo"
        )

    def _declare_parameters(self) -> None:
        parameters = (
            ("left_compressed_topic", LEFT_COMPRESSED_TOPIC),
            ("left_camera_info_topic", LEFT_CAMERA_INFO_TOPIC),
            ("right_compressed_topic", RIGHT_COMPRESSED_TOPIC),
            ("right_camera_info_topic", RIGHT_CAMERA_INFO_TOPIC),
            ("left_rectified_topic", LEFT_RECTIFIED_TOPIC),
            ("left_rectified_info_topic", LEFT_RECTIFIED_INFO_TOPIC),
            ("right_rectified_topic", RIGHT_RECTIFIED_TOPIC),
            ("right_rectified_info_topic", RIGHT_RECTIFIED_INFO_TOPIC),
            ("left_output_frame_id", ""),
            ("right_output_frame_id", ""),
            ("calibration_source", "auto"),
            (
                "rotation_left_to_right",
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ),
            ("translation_left_to_right_m", [0.0, 0.0, 0.0]),
            ("min_baseline_m", 0.0001),
            ("rectification_alpha", 0.0),
            ("sync_queue_size", 10),
            ("sync_slop_sec", 0.05),
        )
        for name, default in parameters:
            self.declare_parameter(name, default)

    def _string(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _integer(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _array(self, name: str) -> list[float]:
        return [float(value) for value in self.get_parameter(name).value]

    @staticmethod
    def _decode(message: CompressedImage) -> np.ndarray:
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV could not decode the compressed image")
        return image

    def _signature(self, left_info: CameraInfo, right_info: CameraInfo, size) -> str:
        values = (
            size,
            tuple(left_info.k),
            tuple(left_info.d),
            str(left_info.distortion_model),
            tuple(left_info.r),
            tuple(left_info.p),
            int(left_info.width),
            int(left_info.height),
            tuple(right_info.k),
            tuple(right_info.d),
            str(right_info.distortion_model),
            tuple(right_info.r),
            tuple(right_info.p),
            int(right_info.width),
            int(right_info.height),
            self._string("calibration_source"),
            tuple(self._array("rotation_left_to_right")),
            tuple(self._array("translation_left_to_right_m")),
            self._float("min_baseline_m"),
            self._float("rectification_alpha"),
        )
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()

    def _get_rectification(
        self, left_info: CameraInfo, right_info: CameraInfo, size: tuple[int, int]
    ) -> Rectification:
        signature = self._signature(left_info, right_info, size)
        if self._rectification is not None and signature == self._calibration_signature:
            return self._rectification
        rectification = build_rectification(
            left_info,
            right_info,
            size,
            calibration_source=self._string("calibration_source"),
            rotation_left_to_right=self._array("rotation_left_to_right"),
            translation_left_to_right_m=self._array(
                "translation_left_to_right_m"
            ),
            min_baseline_m=self._float("min_baseline_m"),
            alpha=self._float("rectification_alpha"),
        )
        self._rectification = rectification
        self._calibration_signature = signature
        self.get_logger().info(
            f"stereo calibration ready: source={rectification.source}, "
            f"baseline={rectification.baseline_m:.6f} m, "
            f"input={size[0]}x{size[1]}"
        )
        return rectification

    @staticmethod
    def _image_message(image_bgr: np.ndarray, source, frame_id: str) -> Image:
        image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
        message = Image()
        message.header.stamp = source.header.stamp
        message.header.frame_id = frame_id or source.header.frame_id
        message.height = int(image.shape[0])
        message.width = int(image.shape[1])
        message.encoding = "bgr8"
        message.is_bigendian = 0
        message.step = int(image.shape[1] * 3)
        message.data = image.tobytes()
        return message

    @staticmethod
    def _camera_info_message(
        source: CameraInfo,
        projection: np.ndarray,
        stamp,
        frame_id: str,
        width: int,
        height: int,
    ) -> CameraInfo:
        P = np.asarray(projection, dtype=np.float64).reshape(3, 4)
        message = CameraInfo()
        message.header.stamp = stamp
        message.header.frame_id = frame_id or source.header.frame_id
        message.width = int(width)
        message.height = int(height)
        message.distortion_model = "plumb_bob"
        message.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        message.k = P[:, :3].reshape(-1).tolist()
        message.r = np.eye(3, dtype=np.float64).reshape(-1).tolist()
        message.p = P.reshape(-1).tolist()
        message.binning_x = 0
        message.binning_y = 0
        message.roi.do_rectify = False
        return message

    def _report_error(self, message: str) -> None:
        if message == self._last_error:
            return
        self._last_error = message
        self.get_logger().error(message)

    def _on_stereo(
        self,
        left_image_message: CompressedImage,
        left_info: CameraInfo,
        right_image_message: CompressedImage,
        right_info: CameraInfo,
    ) -> None:
        try:
            left = self._decode(left_image_message)
            right = self._decode(right_image_message)
            if left.shape != right.shape:
                raise ValueError(
                    f"left/right image shapes differ: {left.shape} vs {right.shape}"
                )
            height, width = left.shape[:2]
            rectification = self._get_rectification(
                left_info, right_info, (width, height)
            )
            left_rectified = cv2.remap(
                left,
                rectification.left_map_x,
                rectification.left_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            right_rectified = cv2.remap(
                right,
                rectification.right_map_x,
                rectification.right_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        except (CalibrationError, ValueError, cv2.error) as error:
            self._report_error(f"stereo frame rejected: {error}")
            return

        self._last_error = ""
        left_frame = self._string("left_output_frame_id")
        right_frame = self._string("right_output_frame_id")
        left_output = self._image_message(
            left_rectified, left_image_message, left_frame
        )
        right_output = self._image_message(
            right_rectified, right_image_message, right_frame
        )
        left_info_output = self._camera_info_message(
            left_info,
            rectification.left_projection,
            left_output.header.stamp,
            left_output.header.frame_id,
            width,
            height,
        )
        right_info_output = self._camera_info_message(
            right_info,
            rectification.right_projection,
            right_output.header.stamp,
            right_output.header.frame_id,
            width,
            height,
        )
        self._left_image_publisher.publish(left_output)
        self._left_info_publisher.publish(left_info_output)
        self._right_image_publisher.publish(right_output)
        self._right_info_publisher.publish(right_info_output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StereoRectifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
