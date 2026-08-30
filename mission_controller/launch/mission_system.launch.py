"""Start G1-D bringup, manipulation actions, and the MQTT workflow."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("robot_bringup"), "launch", "planning_only.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("simulation")),
        launch_arguments={
            "robot_profile": "g1_d",
            "planning_pipeline": LaunchConfiguration("planning_pipeline"),
            "dry_run": LaunchConfiguration("dry_run"),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "enable_fake_ros2_control": "true",
            "enable_fk_pose_publisher": "true",
        }.items(),
    )
    hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("robot_bringup"), "launch", "test.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("hardware")),
        launch_arguments={
            "robot_profile": "g1_d",
            "robot_adapter": "g1d",
            "robot_ip": LaunchConfiguration("robot_ip"),
            "dry_run": "false",
            "prefer_hardware": "true",
            "allow_mock_fallback": "false",
            "enable_robot_state_publisher": "true",
            "enable_move_group": "true",
            "enable_rviz": LaunchConfiguration("enable_rviz"),
        }.items(),
    )
    mission = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("mission_controller"), "launch", "mission.launch.py"]
            )
        ),
        launch_arguments={
            "config_file": LaunchConfiguration("config_file"),
            "handeye_file": LaunchConfiguration("handeye_file"),
            "taskflow_config_file": LaunchConfiguration("taskflow_config_file"),
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("simulation", default_value="true"),
            DeclareLaunchArgument("hardware", default_value="false"),
            DeclareLaunchArgument("planning_pipeline", default_value="stomp"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument("enable_rviz", default_value="true"),
            DeclareLaunchArgument("robot_ip", default_value="enP8p1s0"),
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
            DeclareLaunchArgument(
                "taskflow_config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("mission_controller"), "config", "taskflow.yaml"]
                ),
            ),
            simulation,
            hardware,
            mission,
        ]
    )
