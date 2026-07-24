"""Start the R1 Pro planning stack, Mission, and one perception pipeline."""

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
from launch.substitutions import (
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

    grasp_config = LaunchConfiguration("grasp_config_file").perform(context).strip()
    if (
        mode == "hardware"
        and pipeline == "grasp"
        and grasp_config != "camera_topics_r1pro.yaml"
    ):
        raise RuntimeError(
            "R1 Pro hardware grasp requires "
            "grasp_config_file:=camera_topics_r1pro.yaml"
        )

    try:
        perception_delay = float(
            LaunchConfiguration("perception_start_delay_sec").perform(context)
        )
        mission_delay = float(
            LaunchConfiguration("mission_start_delay_sec").perform(context)
        )
    except ValueError as exc:
        raise RuntimeError("staged startup delays must be numeric") from exc
    if perception_delay < 0.0:
        raise RuntimeError("perception_start_delay_sec must be nonnegative")
    if mission_delay <= perception_delay:
        raise RuntimeError(
            "mission_start_delay_sec must be greater than "
            "perception_start_delay_sec"
        )

    return [
        LogInfo(
            msg=(
                f"Mission system: dual_arm + {pipeline} + mission "
                f"(mode: {mode}); staged startup order: "
                f"dual_arm@0s -> {pipeline}@{perception_delay:g}s -> "
                f"mission@{mission_delay:g}s"
            )
        )
    ]


def generate_launch_description() -> LaunchDescription:
    mode = LaunchConfiguration("mode")
    pipeline = LaunchConfiguration("pipeline")

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

    grasp_config_file = PathJoinSubstitution(
        [
            FindPackageShare("grasp_orchestrator"),
            "config",
            LaunchConfiguration("grasp_config_file"),
        ]
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
            "enable_robot_state_publisher": LaunchConfiguration(
                "enable_robot_state_publisher"
            ),
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

    perception_start = TimerAction(
        period=LaunchConfiguration("perception_start_delay_sec"),
        actions=[grasp, box],
    )
    mission_start = TimerAction(
        period=LaunchConfiguration("mission_start_delay_sec"),
        actions=[mission],
    )

    runtime = GroupAction(
        actions=[
            # Start the planning/hardware stack first. MoveIt initialization is
            # intentionally given its own startup window before perception
            # loads a model and Mission begins creating its action clients.
            simulation_dual_arm,
            hardware_dual_arm,
            perception_start,
            mission_start,
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
            DeclareLaunchArgument("robot_profile", default_value="r1_pro"),
            DeclareLaunchArgument("planning_pipeline", default_value="stomp"),
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
            DeclareLaunchArgument(
                "enable_robot_state_publisher",
                default_value="true",
                description=(
                    "Start the Dual Arm robot_state_publisher. Set false when "
                    "the R1 Pro native bringup already publishes the required "
                    "robot TF tree."
                ),
            ),
            DeclareLaunchArgument("enable_fk_pose_publisher", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("log_level", default_value="info"),
            DeclareLaunchArgument(
                "perception_start_delay_sec",
                default_value="0.0",
                description=(
                    "Seconds after launch before starting the selected grasp or "
                    "box perception pipeline. Dual Arm starts immediately."
                ),
            ),
            DeclareLaunchArgument(
                "mission_start_delay_sec",
                default_value="30.0",
                description=(
                    "Seconds after launch before starting Mission. Keep this "
                    "greater than perception_start_delay_sec."
                ),
            ),
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
                    "Grasp camera config filename. Hardware mode requires "
                    "camera_topics_r1pro.yaml."
                ),
            ),
            DeclareLaunchArgument("grasp_visualize", default_value="false"),
            DeclareLaunchArgument(
                "grasp_visualization_grasps", default_value="10"
            ),
            DeclareLaunchArgument(
                "box_config_file",
                default_value="object_pose_r1pro.yaml",
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
