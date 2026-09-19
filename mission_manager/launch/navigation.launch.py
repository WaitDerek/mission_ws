"""Start manipulation actions and the MQTT-driven workflow."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "taskflow_config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("mission_manager"), "config", "taskflow.yaml"]
                ),
            ),
           
            Node(
                package="mission_manager",
                executable="run_navigation",
                name="execute_workflow",
                output="screen",
                parameters=[LaunchConfiguration("taskflow_config_file")],
            ),
        ]
    )
