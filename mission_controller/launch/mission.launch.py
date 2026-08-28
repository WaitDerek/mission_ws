"""Launch the RealBot box-grasp/box-place mission controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_file = LaunchConfiguration("config_file")
    fragment_names = (
        "core.yaml",
        "camera.yaml",
        "adaptive.yaml",
        "direct_motion.yaml",
        "box_common.yaml",
        "grasp_tf.yaml",
        "drag.yaml",
        "placement.yaml",
    )
    mission_fragments = [
        PathJoinSubstitution(
            [FindPackageShare("mission_controller"), "config", "mission", name]
        )
        for name in fragment_names
    ]
    require_command_subscribers = LaunchConfiguration("require_command_subscribers")
    direct_motion_backend = LaunchConfiguration("direct_motion_backend")
    enable_global_tf = LaunchConfiguration("enable_global_tf")
    enable_robot_state_publisher = LaunchConfiguration("enable_robot_state_publisher")

    controller = Node(
        package="mission_controller",
        executable="mission_controller",
        name="mission_controller",
        output="screen",
        prefix=[FindExecutable(name="python3")],
        parameters=[
            *mission_fragments,
            config_file,
            {
                "require_command_subscribers": ParameterValue(
                    require_command_subscribers, value_type=bool
                ),
                "direct_motion_backend": direct_motion_backend,
            },
        ],
    )

    global_tf = Node(
        package="mission_controller",
        executable="realbots_global_tf",
        name="realbots_global_tf",
        output="screen",
        condition=IfCondition(enable_global_tf),
        prefix=[FindExecutable(name="python3")],
        additional_env={
            "ROS_LOCALHOST_ONLY": "0",
            "RMW_IMPLEMENTATION": LaunchConfiguration("global_tf_rmw_implementation"),
            "CYCLONEDDS_URI": LaunchConfiguration("global_tf_cyclonedds_uri"),
        },
        parameters=[
            LaunchConfiguration("global_tf_config_file"),
            {
                "urdf_file": ParameterValue(
                    LaunchConfiguration("global_tf_urdf_file"),
                    value_type=str,
                )
            },
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        condition=IfCondition(enable_robot_state_publisher),
        parameters=[
            {
                "robot_description": ParameterValue(
                    Command(
                        [
                            FindExecutable(name="cat"),
                            " ",
                            LaunchConfiguration("global_tf_urdf_file"),
                        ]
                    ),
                    value_type=str,
                ),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
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
            DeclareLaunchArgument("enable_global_tf", default_value="true"),
            DeclareLaunchArgument("enable_robot_state_publisher", default_value="true"),
            DeclareLaunchArgument(
                "global_tf_config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("mission_controller"),
                        "config",
                        "global_tf.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "global_tf_urdf_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("realbots29"), "urdf", "realbots29.urdf"]
                ),
            ),
            DeclareLaunchArgument(
                "global_tf_rmw_implementation",
                default_value="rmw_cyclonedds_cpp",
            ),
            DeclareLaunchArgument(
                "global_tf_cyclonedds_uri",
                default_value="file:///rm_app/rm_robot_ws/script/cyclonedds.xml",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            robot_state_publisher,
            global_tf,
            controller,
        ]
    )
