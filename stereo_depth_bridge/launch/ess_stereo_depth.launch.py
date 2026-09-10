"""Run compressed stereo preparation, Isaac ROS ESS and metric depth output."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from stereo_depth_bridge.contracts import (
    DISPARITY_TOPIC,
    LEFT_CAMERA_INFO_OUTPUT_TOPIC,
    LEFT_CAMERA_INFO_TOPIC,
    LEFT_COLOR_OUTPUT_TOPIC,
    LEFT_COMPRESSED_TOPIC,
    LEFT_DEPTH_OUTPUT_TOPIC,
    LEFT_ESS_TOPIC,
    LEFT_RECTIFIED_INFO_TOPIC,
    LEFT_RECTIFIED_TOPIC,
    RIGHT_CAMERA_INFO_TOPIC,
    RIGHT_COMPRESSED_TOPIC,
    RIGHT_ESS_INFO_TOPIC,
    RIGHT_ESS_TOPIC,
    RIGHT_RECTIFIED_INFO_TOPIC,
    RIGHT_RECTIFIED_TOPIC,
)


def _validate_launch(context):
    engine_path = LaunchConfiguration("engine_file_path").perform(context).strip()
    if not engine_path or not Path(engine_path).expanduser().is_file():
        raise RuntimeError(
            "engine_file_path must point to an existing ESS TensorRT engine"
        )
    width = int(LaunchConfiguration("model_input_width").perform(context))
    height = int(LaunchConfiguration("model_input_height").perform(context))
    if width <= 0 or height <= 0:
        raise RuntimeError("model input width and height must be positive")
    threshold = float(
        LaunchConfiguration("confidence_threshold").perform(context)
    )
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError("confidence_threshold must be in [0, 1]")
    return [
        LogInfo(
            msg=(
                f"ESS stereo depth: engine={engine_path}, "
                f"model_input={width}x{height}, threshold={threshold:g}"
            )
        )
    ]


def generate_launch_description() -> LaunchDescription:
    width = LaunchConfiguration("model_input_width")
    height = LaunchConfiguration("model_input_height")

    rectifier = Node(
        package="stereo_depth_bridge",
        executable="stereo_rectifier",
        name="stereo_rectifier",
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
            {
                "left_compressed_topic": LaunchConfiguration(
                    "left_compressed_topic"
                ),
                "left_camera_info_topic": LaunchConfiguration(
                    "left_camera_info_topic"
                ),
                "right_compressed_topic": LaunchConfiguration(
                    "right_compressed_topic"
                ),
                "right_camera_info_topic": LaunchConfiguration(
                    "right_camera_info_topic"
                ),
            },
        ],
    )

    left_resize = ComposableNode(
        package="isaac_ros_image_proc",
        plugin="nvidia::isaac_ros::image_proc::ResizeNode",
        name="left_ess_resize",
        parameters=[
            {
                "output_width": ParameterValue(width, value_type=int),
                "output_height": ParameterValue(height, value_type=int),
                "keep_aspect_ratio": False,
                "encoding_desired": "rgb8",
            }
        ],
        remappings=[
            ("image", LEFT_RECTIFIED_TOPIC),
            ("camera_info", LEFT_RECTIFIED_INFO_TOPIC),
            ("resize/image", LEFT_ESS_TOPIC),
            ("resize/camera_info", LEFT_CAMERA_INFO_OUTPUT_TOPIC),
        ],
    )
    right_resize = ComposableNode(
        package="isaac_ros_image_proc",
        plugin="nvidia::isaac_ros::image_proc::ResizeNode",
        name="right_ess_resize",
        parameters=[
            {
                "output_width": ParameterValue(width, value_type=int),
                "output_height": ParameterValue(height, value_type=int),
                "keep_aspect_ratio": False,
                "encoding_desired": "rgb8",
            }
        ],
        remappings=[
            ("image", RIGHT_RECTIFIED_TOPIC),
            ("camera_info", RIGHT_RECTIFIED_INFO_TOPIC),
            ("resize/image", RIGHT_ESS_TOPIC),
            ("resize/camera_info", RIGHT_ESS_INFO_TOPIC),
        ],
    )
    ess = ComposableNode(
        package="isaac_ros_ess",
        plugin="nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode",
        name="ess_disparity",
        parameters=[
            {
                "engine_file_path": LaunchConfiguration("engine_file_path"),
                "threshold": ParameterValue(
                    LaunchConfiguration("confidence_threshold"), value_type=float
                ),
                "input_layer_width": ParameterValue(width, value_type=int),
                "input_layer_height": ParameterValue(height, value_type=int),
            }
        ],
        remappings=[
            ("left/image_rect", LEFT_ESS_TOPIC),
            ("left/camera_info", LEFT_CAMERA_INFO_OUTPUT_TOPIC),
            ("right/image_rect", RIGHT_ESS_TOPIC),
            ("right/camera_info", RIGHT_ESS_INFO_TOPIC),
            ("disparity", DISPARITY_TOPIC),
        ],
    )
    disparity_to_depth = ComposableNode(
        package="isaac_ros_stereo_image_proc",
        plugin="nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode",
        name="disparity_to_depth",
        remappings=[
            ("disparity", DISPARITY_TOPIC),
            ("depth", LEFT_DEPTH_OUTPUT_TOPIC),
        ],
    )
    left_bgr = ComposableNode(
        package="isaac_ros_image_proc",
        plugin="nvidia::isaac_ros::image_proc::ImageFormatConverterNode",
        name="left_bgr_output",
        parameters=[
            {
                "encoding_desired": "bgr8",
                "image_width": ParameterValue(width, value_type=int),
                "image_height": ParameterValue(height, value_type=int),
            }
        ],
        remappings=[
            ("image_raw", LEFT_ESS_TOPIC),
            ("image", LEFT_COLOR_OUTPUT_TOPIC),
        ],
    )
    container = ComposableNodeContainer(
        package="rclcpp_components",
        executable="component_container_mt",
        name="stereo_depth_container",
        namespace="",
        composable_node_descriptions=[
            left_resize,
            right_resize,
            ess,
            disparity_to_depth,
            left_bgr,
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("stereo_depth_bridge"),
                        "config",
                        "stereo_depth.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "left_compressed_topic", default_value=LEFT_COMPRESSED_TOPIC
            ),
            DeclareLaunchArgument(
                "left_camera_info_topic", default_value=LEFT_CAMERA_INFO_TOPIC
            ),
            DeclareLaunchArgument(
                "right_compressed_topic", default_value=RIGHT_COMPRESSED_TOPIC
            ),
            DeclareLaunchArgument(
                "right_camera_info_topic", default_value=RIGHT_CAMERA_INFO_TOPIC
            ),
            DeclareLaunchArgument(
                "engine_file_path",
                default_value="",
                description="Absolute path to the Isaac ROS 3.2 ESS TensorRT engine",
            ),
            DeclareLaunchArgument("model_input_width", default_value="960"),
            DeclareLaunchArgument("model_input_height", default_value="576"),
            DeclareLaunchArgument("confidence_threshold", default_value="0.4"),
            OpaqueFunction(function=_validate_launch),
            rectifier,
            container,
        ]
    )
