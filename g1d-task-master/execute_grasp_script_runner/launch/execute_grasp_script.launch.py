from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    action_name = LaunchConfiguration("action_name")
    script_path = LaunchConfiguration("script_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "action_name",
                default_value="/execute_grasp",
                description=(
                    "Action endpoint. Do not run mission_controller with the "
                    "same endpoint."
                ),
            ),
            DeclareLaunchArgument(
                "script_path",
                default_value="",
                description=(
                    "Python script to execute. Empty uses the package main.py."
                ),
            ),
            Node(
                package="execute_grasp_script_runner",
                executable="execute_grasp_script_server",
                name="execute_grasp_script_server",
                output="screen",
                parameters=[
                    {
                        "action_name": action_name,
                        "script_path": script_path,
                    }
                ],
            ),
        ]
    )
