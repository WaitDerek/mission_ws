"""Start the R1 Pro planning stack, Mission, and one perception pipeline."""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.substitutions import FindPackageShare


def _validate_configuration(context):
    mode = LaunchConfiguration("mode").perform(context).strip().lower()
    if mode not in {"simulation", "hardware"}:
        raise RuntimeError(
            f"Unsupported mode '{mode}'; explicitly pass "
            "mode:=simulation or mode:=hardware"
        )

    pipeline = LaunchConfiguration("pipeline").perform(context).strip().lower()
    if pipeline not in {"grasp", "box"}:
        raise RuntimeError(
            f"Unsupported pipeline '{pipeline}'; expected 'grasp' or 'box'"
        )

    conda_root = LaunchConfiguration("conda_root").perform(context)
    grasp_env = LaunchConfiguration("grasp_conda_env").perform(context)
    vision_env = LaunchConfiguration("vision_conda_env").perform(context)
    grasp_python = os.path.join(conda_root, "envs", grasp_env, "bin", "python")
    if not os.access(grasp_python, os.X_OK):
        raise RuntimeError(
            f"Python for Dual Arm, Mission, and Grasp is not executable: "
            f"{grasp_python}"
        )

    selected_env = grasp_env
    if pipeline == "box":
        vision_python = os.path.join(
            conda_root, "envs", vision_env, "bin", "python"
        )
        if not os.access(vision_python, os.X_OK):
            raise RuntimeError(
                f"Python for Vision is not executable: {vision_python}"
            )
        selected_env = vision_env

    return [
        LogInfo(
            msg=(
                f"Mission system: dual_arm + mission + {pipeline} "
                f"(mode: {mode}, conda env: {selected_env})"
            )
        )
    ]


def generate_launch_description() -> LaunchDescription:
    mode = LaunchConfiguration("mode")
    pipeline = LaunchConfiguration("pipeline")
    conda_root = LaunchConfiguration("conda_root")
    grasp_conda_env = LaunchConfiguration("grasp_conda_env")
    vision_conda_env = LaunchConfiguration("vision_conda_env")

    grasp_condition = IfCondition(
        PythonExpression(["'", pipeline, "'.lower() == 'grasp'"])
    )
    box_condition = IfCondition(
        PythonExpression(["'", pipeline, "'.lower() == 'box'"])
    )
    simulation_condition = IfCondition(
        PythonExpression(["'", mode, "'.lower() == 'simulation'"])
    )
    hardware_condition = IfCondition(
        PythonExpression(["'", mode, "'.lower() == 'hardware'"])
    )

    grasp_python_bin = PathJoinSubstitution(
        [conda_root, "envs", grasp_conda_env, "bin"]
    )
    grasp_config_file = PathJoinSubstitution(
        [
            FindPackageShare("grasp_orchestrator"),
            "config",
            LaunchConfiguration("grasp_config_file"),
        ]
    )
    vision_python = PathJoinSubstitution(
        [conda_root, "envs", vision_conda_env, "bin", "python"]
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
            "enable_fk_pose_publisher": LaunchConfiguration(
                "enable_fk_pose_publisher"
            ),
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
            "robot_adapter": "galaxy",
            "robot_ip": LaunchConfiguration("robot_ip"),
            "planning_pipeline": LaunchConfiguration("planning_pipeline"),
            "dry_run": LaunchConfiguration("dry_run"),
            "prefer_hardware": "true",
            "allow_mock_fallback": "false",
            "hardware_armed": LaunchConfiguration("hardware_armed"),
            "galaxy_enable_native_cartesian": LaunchConfiguration(
                "galaxy_enable_native_cartesian"
            ),
            "enable_robot_state_publisher": "true",
            "enable_move_group": "true",
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "enable_fake_ros2_control": "false",
            "enable_fk_pose_publisher": LaunchConfiguration(
                "enable_fk_pose_publisher"
            ),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "log_level": LaunchConfiguration("log_level"),
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
            "start_perception": "false",
            "start_detector_daemon": "false",
            "require_command_subscribers": PythonExpression(
                ["'", mode, "'.lower() == 'hardware'"]
            ),
        }.items(),
    )

    grasp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("grasp_orchestrator"),
                    "launch",
                    "grasp_detection.launch.py",
                ]
            )
        ),
        condition=grasp_condition,
        launch_arguments={
            "start_daemon": LaunchConfiguration("start_grasp_daemon"),
            "config_file": grasp_config_file,
            "visualize": LaunchConfiguration("grasp_visualize"),
            "visualization_grasps": LaunchConfiguration(
                "grasp_visualization_grasps"
            ),
        }.items(),
    )

    box = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("object_pose_ros"),
                    "launch",
                    "object_pose_action.launch.py",
                ]
            )
        ),
        condition=box_condition,
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

    runtime = GroupAction(
        actions=[
            # Dual Arm, Mission, and Grasp use changan. The Vision executable is
            # a wrapper that honors OBJECT_POSE_PYTHON, so it remains isolated.
            SetEnvironmentVariable(
                "PATH",
                [grasp_python_bin, os.pathsep, EnvironmentVariable("PATH")],
            ),
            SetEnvironmentVariable("PYTHONNOUSERSITE", "1"),
            SetEnvironmentVariable("OBJECT_POSE_PYTHON", vision_python),
            simulation_dual_arm,
            hardware_dual_arm,
            mission,
            grasp,
            box,
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mode",
                default_value="required",
                description=(
                    "Required execution mode: simulation or hardware. Hardware "
                    "remains locked unless hardware_armed:=true."
                ),
            ),
            DeclareLaunchArgument(
                "pipeline",
                default_value="grasp",
                description="Perception pipeline: grasp or box.",
            ),
            DeclareLaunchArgument(
                "conda_root",
                default_value=EnvironmentVariable(
                    "MISSION_CONDA_ROOT", default_value="/home/dekc/anaconda3"
                ),
            ),
            DeclareLaunchArgument(
                "grasp_conda_env",
                default_value=EnvironmentVariable(
                    "MISSION_GRASP_CONDA_ENV", default_value="changan"
                ),
            ),
            DeclareLaunchArgument(
                "vision_conda_env",
                default_value=EnvironmentVariable(
                    "MISSION_VISION_CONDA_ENV", default_value="foundationpose"
                ),
            ),
            DeclareLaunchArgument("robot_profile", default_value="r1_pro"),
            DeclareLaunchArgument("planning_pipeline", default_value="ompl"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument("enable_rviz", default_value="true"),
            DeclareLaunchArgument("robot_ip", default_value="auto"),
            DeclareLaunchArgument(
                "hardware_armed",
                default_value="false",
                description=(
                    "R1 Pro hardware motion gate. Keep false for preflight; set "
                    "true explicitly only after feedback and emergency-stop checks."
                ),
            ),
            DeclareLaunchArgument(
                "galaxy_enable_native_cartesian", default_value="false"
            ),
            DeclareLaunchArgument("enable_fk_pose_publisher", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("log_level", default_value="info"),
            DeclareLaunchArgument(
                "mission_config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("mission_controller"), "config", "mission.yaml"]
                ),
            ),
            DeclareLaunchArgument("start_grasp_daemon", default_value="true"),
            DeclareLaunchArgument(
                "grasp_config_file",
                default_value="camera_topics_r1pro.yaml",
                description=(
                    "Config filename under grasp_orchestrator/config, for example "
                    "camera_topics_local_d405.yaml."
                ),
            ),
            DeclareLaunchArgument("grasp_visualize", default_value="false"),
            DeclareLaunchArgument(
                "grasp_visualization_grasps", default_value="10"
            ),
            DeclareLaunchArgument(
                "box_config_file",
                default_value="object_pose_hd.yaml",
                description=(
                    "Config filename under object_pose_ros/config for the box "
                    "FoundationPose pipeline."
                ),
            ),
            DeclareLaunchArgument("box_server_output", default_value="screen"),
            OpaqueFunction(function=_validate_configuration),
            runtime,
        ]
    )
