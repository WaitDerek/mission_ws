"""Launch the RealBot box-grasp/box-place mission controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_file = LaunchConfiguration("config_file")
    require_command_subscribers = LaunchConfiguration(
        "require_command_subscribers"
    )
    direct_motion_backend = LaunchConfiguration("direct_motion_backend")

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
                ),
                "direct_motion_backend": direct_motion_backend,
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
            DeclareLaunchArgument(
                "require_command_subscribers",
                default_value="false",
                description=(
                    "Abort box mission commands when command topics have no "
                    "subscribers. Set true for hardware."
                ),
            ),
            DeclareLaunchArgument(
                "direct_motion_backend",
                default_value="python_sdk",
            ),
            controller,
        ]
    )
