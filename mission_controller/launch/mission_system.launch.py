"""Start RealBot dual-arm bringup, box perception, and Mission."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


def _validate_configuration(context):
    mode = LaunchConfiguration("mode").perform(context).strip().lower()
    if mode not in {"simulation", "hardware"}:
        raise RuntimeError(
            f"Unsupported mode '{mode}'; explicitly pass "
            "mode:=simulation or mode:=hardware"
        )

    perception_delay = float(
        LaunchConfiguration("perception_start_delay_sec").perform(context)
    )
    mission_delay = float(
        LaunchConfiguration("mission_start_delay_sec").perform(context)
    )
    if perception_delay < 0.0 or mission_delay <= perception_delay:
        raise RuntimeError(
            "mission_start_delay_sec must be greater than "
            "perception_start_delay_sec, and both delays must be valid"
        )

    return [
        LogInfo(
            msg=(
                f"Mission system: dual_arm + box perception + mission "
                f"(mode: {mode}); startup order: "
                f"dual_arm@0s -> box@{perception_delay:g}s -> "
                f"mission@{mission_delay:g}s"
            )
        )
    ]


def generate_launch_description() -> LaunchDescription:
    mode = LaunchConfiguration("mode")
    simulation_condition = IfCondition(
        PythonExpression(["'", mode, "'.lower() == 'simulation'"])
    )
    hardware_condition = IfCondition(
        PythonExpression(["'", mode, "'.lower() == 'hardware'"])
    )

    simulation_dual_arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("robot_bringup"), "launch", "planning_only.launch.py"]
            )
        ),
        condition=simulation_condition,
        launch_arguments={
            "robot_profile": LaunchConfiguration("robot_profile"),
            "planning_pipeline": LaunchConfiguration("planning_pipeline"),
            "dry_run": LaunchConfiguration("dry_run"),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "enable_fake_ros2_control": "true",
            "enable_fk_pose_publisher": LaunchConfiguration("enable_fk_pose_publisher"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "log_level": LaunchConfiguration("log_level"),
        }.items(),
    )

    hardware_dual_arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("robot_bringup"), "launch", "test.launch.py"]
            )
        ),
        condition=hardware_condition,
        launch_arguments={
            "robot_profile": LaunchConfiguration("robot_profile"),
            "robot_adapter": "realbot",
            "robot_ip": LaunchConfiguration("robot_ip"),
            "robot_port": LaunchConfiguration("robot_port"),
            "planning_pipeline": LaunchConfiguration("planning_pipeline"),
            "dry_run": LaunchConfiguration("dry_run"),
            "prefer_hardware": "true",
            "allow_mock_fallback": "false",
            "enable_robot_state_publisher": LaunchConfiguration("enable_robot_state_publisher"),
            "enable_move_group": "true",
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "enable_fake_ros2_control": "false",
            "enable_fk_pose_publisher": LaunchConfiguration("enable_fk_pose_publisher"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "log_level": LaunchConfiguration("log_level"),
        }.items(),
    )

    box_perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("object_pose_ros"), "launch", "object_pose_action.launch.py"]
            )
        ),
        launch_arguments={
            "config_file": PathJoinSubstitution(
                [
                    FindPackageShare("object_pose_ros"),
                    "config",
                    LaunchConfiguration("box_config_file"),
                ]
            ),
            "server_output": LaunchConfiguration("box_server_output"),
        }.items(),
    )

    mission = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("mission_controller"), "launch", "mission.launch.py"]
            )
        ),
        launch_arguments={
            "config_file": LaunchConfiguration("mission_config_file"),
            "require_command_subscribers": PythonExpression(
                ["'", mode, "'.lower() == 'hardware'"]
            ),
        }.items(),
    )

    runtime = GroupAction(
        actions=[
            simulation_dual_arm,
            hardware_dual_arm,
            TimerAction(
                period=LaunchConfiguration("perception_start_delay_sec"),
                actions=[box_perception],
            ),
            TimerAction(
                period=LaunchConfiguration("mission_start_delay_sec"),
                actions=[mission],
            ),
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="required"),
            DeclareLaunchArgument("robot_profile", default_value="realbot"),
            DeclareLaunchArgument("planning_pipeline", default_value="ompl"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument("enable_rviz", default_value="true"),
            DeclareLaunchArgument("robot_ip", default_value="192.168.127.18,192.168.127.19"),
            DeclareLaunchArgument("robot_port", default_value="8080"),
            DeclareLaunchArgument("enable_robot_state_publisher", default_value="true"),
            DeclareLaunchArgument("enable_fk_pose_publisher", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("log_level", default_value="info"),
            DeclareLaunchArgument("perception_start_delay_sec", default_value="0.0"),
            DeclareLaunchArgument("mission_start_delay_sec", default_value="30.0"),
            DeclareLaunchArgument(
                "mission_config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("mission_controller"), "config", "mission.yaml"]
                ),
            ),
            DeclareLaunchArgument("box_config_file", default_value="object_pose.yaml"),
            DeclareLaunchArgument("box_server_output", default_value="screen"),
            OpaqueFunction(function=_validate_configuration),
            runtime,
        ]
    )
