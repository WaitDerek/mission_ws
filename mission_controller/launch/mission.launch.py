"""Start the G1-D-only mission controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("mission_controller"), "config", "mission.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "handeye_file",
                default_value="handeye_result_12.yaml",
                description="Hand-eye YAML filename relative to the mission config directory.",
            ),
            Node(
                package="mission_controller",
                executable="mission_controller",
                name="mission_controller",
                output="screen",
                parameters=[
                    LaunchConfiguration("config_file"),
                    {"handeye_file": LaunchConfiguration("handeye_file")},
                ],
            ),
        ]
    )
