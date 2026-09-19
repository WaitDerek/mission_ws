"""Start manipulation actions"""

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_dir",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("mission_manager"), "config"]
                ),
            ),
            DeclareLaunchArgument(
                "fake_camera_pose",
                default_value="false",
                description="Whether to use a fake camera pose",
            ),
            Node(
                package="mission_manager",
                executable="run_mission",
                output="screen",
                parameters=[
                    {
                        "config_dir": LaunchConfiguration("config_dir"),
                        "fake_camera_pose": ParameterValue(
                            LaunchConfiguration("fake_camera_pose"),
                            value_type=bool,
                        ),
                    }
                ],
            )
        ]
    )
