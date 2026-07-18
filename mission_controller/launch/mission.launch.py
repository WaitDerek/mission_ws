from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_file = LaunchConfiguration("config_file")
    start_perception = LaunchConfiguration("start_perception")
    start_detector_daemon = LaunchConfiguration("start_detector_daemon")
    require_command_subscribers = LaunchConfiguration(
        "require_command_subscribers"
    )

    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("grasp_orchestrator"),
                    "launch",
                    "grasp_detection.launch.py",
                ]
            )
        ),
        condition=IfCondition(start_perception),
        launch_arguments={"start_daemon": start_detector_daemon}.items(),
    )

    controller = Node(
        package="mission_controller",
        executable="mission_controller",
        name="mission_controller",
        output="screen",
        prefix=[FindExecutable(name="python3")],
        parameters=[
            config_file,
            {
                "require_command_subscribers": ParameterValue(
                    require_command_subscribers, value_type=bool
                )
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("mission_controller"), "config", "mission.yaml"]
                ),
            ),
            DeclareLaunchArgument("start_perception", default_value="true"),
            DeclareLaunchArgument("start_detector_daemon", default_value="true"),
            DeclareLaunchArgument(
                "require_command_subscribers",
                default_value="false",
                description=(
                    "Abort Mission commands when gripper or torso topics have "
                    "no subscribers. Hardware mode must set this to true."
                ),
            ),
            perception_launch,
            controller,
        ]
    )
