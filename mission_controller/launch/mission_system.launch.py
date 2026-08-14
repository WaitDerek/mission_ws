"""Start RealBot dual-arm bringup, box perception, and Mission."""

import os
from pathlib import Path
import shutil

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.substitutions import FindPackageShare


def _default_box_perception_root() -> str:
    """Find the perception checkout without binding the launch file to a user."""
    explicit_root = os.environ.get("OBJECT_POSE_ROOT", "").strip()
    if explicit_root:
        return str(Path(explicit_root).expanduser().resolve())

    launch_path = Path(__file__).resolve()
    marker = Path("scripts/start_object_pose_action.sh")
    for ancestor in launch_path.parents:
        if (ancestor / marker).is_file():
            return str(ancestor)

        vision_source = ancestor / "vision_ws" / "src"
        if not vision_source.is_dir():
            continue
        for candidate in sorted(vision_source.iterdir()):
            if candidate.is_dir() and (candidate / marker).is_file():
                return str(candidate.resolve())
    return ""


def _clean_runtime_path() -> str:
    """Keep ROS and system commands while excluding user environment paths."""
    entries: list[str] = []
    ros2_executable = shutil.which("ros2")
    if ros2_executable:
        entries.append(str(Path(ros2_executable).resolve().parent))
    entries.extend(os.defpath.split(os.pathsep))
    return os.pathsep.join(dict.fromkeys(entries))


def _validate_configuration(context):
    mode = LaunchConfiguration("mode").perform(context).strip().lower()
    if mode not in {"simulation", "hardware"}:
        raise RuntimeError(
            f"Unsupported mode '{mode}'; explicitly pass "
            "mode:=simulation or mode:=hardware"
        )

    direct_motion_backend = (
        LaunchConfiguration("direct_motion_backend").perform(context).strip().lower()
    )
    if direct_motion_backend not in {"python_sdk", "ros_service"}:
        raise RuntimeError(
            "direct_motion_backend must be python_sdk or ros_service"
        )
    if mode == "simulation" and direct_motion_backend == "python_sdk":
        raise RuntimeError(
            "direct_motion_backend=python_sdk is only valid with mode:=hardware"
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

    perception_enabled = (
        LaunchConfiguration("enable_box_perception")
        .perform(context)
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    if perception_enabled:
        required_files = {
            "box perception start script": LaunchConfiguration(
                "box_perception_script"
            ).perform(context),
            "box perception config": LaunchConfiguration("box_config_file").perform(
                context
            ),
        }
        missing = [
            f"{label}: {path or '<empty>'}"
            for label, path in required_files.items()
            if not path or not Path(path).is_file()
        ]
        if missing:
            raise RuntimeError(
                "Cannot locate box perception files ("
                + "; ".join(missing)
                + "). Set OBJECT_POSE_ROOT or pass box_perception_root:=..."
            )

    return [
        LogInfo(
            msg=(
                f"Mission system: dual_arm + optional box perception + mission "
                f"(mode: {mode}); startup order: "
                f"dual_arm@0s -> box@{perception_delay:g}s -> "
                f"mission@{mission_delay:g}s; direct_motion_backend="
                f"{direct_motion_backend}"
            )
        )
    ]


def generate_launch_description() -> LaunchDescription:
    box_perception_root_default = _default_box_perception_root()
    clean_runtime_path = _clean_runtime_path()
    mode = LaunchConfiguration("mode")
    direct_motion_backend = LaunchConfiguration("direct_motion_backend")
    simulation_condition = IfCondition(
        PythonExpression(["'", mode, "'.lower() == 'simulation'"])
    )
    hardware_condition = IfCondition(
        PythonExpression(
            [
                "'",
                mode,
                "'.lower() == 'hardware' and '",
                direct_motion_backend,
                "'.lower() == 'ros_service'",
            ]
        )
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

    box_perception = ExecuteProcess(
        cmd=[
            FindExecutable(name="env"),
            "-u",
            "PYTHONHOME",
            "-u",
            "PYTHONPATH",
            "-u",
            "CONDA_PREFIX",
            "-u",
            "CONDA_DEFAULT_ENV",
            f"PATH={clean_runtime_path}",
            "ROS_DOMAIN_ID=0",
            "ROS_LOCALHOST_ONLY=0",
            FindExecutable(name="bash"),
            LaunchConfiguration("box_perception_script"),
            LaunchConfiguration("box_camera_source"),
            ["config_file:=", LaunchConfiguration("box_config_file")],
            ["camera_model:=", LaunchConfiguration("box_camera_model")],
            ["server_output:=", LaunchConfiguration("box_server_output")],
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_box_perception")),
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
            "direct_motion_backend": direct_motion_backend,
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
            DeclareLaunchArgument(
                "direct_motion_backend",
                default_value="python_sdk",
                description=(
                    "Motion owner: python_sdk uses the RealMan Python SDK and "
                    "skips hardware_driver; ros_service uses /realbot/movel."
                ),
            ),
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
            DeclareLaunchArgument(
                "box_perception_root",
                default_value=box_perception_root_default,
                description=(
                    "Perception source checkout. Defaults to OBJECT_POSE_ROOT "
                    "or a sibling vision_ws checkout discovered from this file."
                ),
            ),
            DeclareLaunchArgument(
                "box_perception_script",
                default_value=PathJoinSubstitution(
                    [
                        LaunchConfiguration("box_perception_root"),
                        "scripts",
                        "start_object_pose_action.sh",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "box_config_file",
                default_value=PathJoinSubstitution(
                    [
                        LaunchConfiguration("box_perception_root"),
                        "ros2",
                        "object_pose_ros",
                        "config",
                        "object_pose_bigbox.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument("enable_box_perception", default_value="true"),
            DeclareLaunchArgument("box_camera_source", default_value="d435i"),
            DeclareLaunchArgument("box_camera_model", default_value="d435i"),
            DeclareLaunchArgument("box_server_output", default_value="screen"),
            OpaqueFunction(function=_validate_configuration),
            runtime,
        ]
    )
