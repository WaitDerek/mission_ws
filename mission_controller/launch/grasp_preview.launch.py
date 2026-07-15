from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_file = LaunchConfiguration("config_file")
    start_rviz = LaunchConfiguration("start_rviz")
    start_robot_state_publisher = LaunchConfiguration("start_robot_state_publisher")
    joint_states_topic = LaunchConfiguration("joint_states_topic")
    robot_xacro = PathJoinSubstitution(
        [
            FindPackageShare("r1_pro_moveit_config"),
            "config",
            "r1_pro_with_gripper.urdf.xacro",
        ]
    )
    robot_srdf = PathJoinSubstitution(
        [
            FindPackageShare("r1_pro_moveit_config"),
            "config",
            "r1_pro_with_gripper.srdf",
        ]
    )
    robot_description = ParameterValue(Command(["xacro ", robot_xacro]), value_type=str)
    robot_description_semantic = ParameterValue(
        Command(["xacro ", robot_srdf]), value_type=str
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("mission_controller"), "config", "mission.yaml"]
                ),
            ),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument("start_robot_state_publisher", default_value="true"),
            DeclareLaunchArgument(
                "joint_states_topic",
                default_value="/mission/preview_joint_states",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="grasp_preview_robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
                remappings=[("/joint_states", joint_states_topic)],
                condition=IfCondition(start_robot_state_publisher),
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="grasp_preview_color_optical_tf",
                output="screen",
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--roll",
                    "-1.5707963267948966",
                    "--pitch",
                    "0",
                    "--yaw",
                    "-1.5707963267948966",
                    "--frame-id",
                    "hdas/camera_wrist_right_link",
                    "--child-frame-id",
                    "hdas/camera_wrist_right_color_optical_frame",
                ],
            ),
            Node(
                package="mission_controller",
                executable="mission_controller",
                name="mission_controller",
                output="screen",
                parameters=[config_file],
            ),
            Node(
                package="mission_controller",
                executable="grasp_preview_publisher",
                name="grasp_preview_publisher",
                output="screen",
                parameters=[{"joint_states_topic": joint_states_topic}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="grasp_preview_rviz",
                output="screen",
                arguments=[
                    "-d",
                    PathJoinSubstitution(
                        [
                            FindPackageShare("mission_controller"),
                            "config",
                            "grasp_preview.rviz",
                        ]
                    ),
                ],
                parameters=[
                    {
                        "robot_description": robot_description,
                        "robot_description_semantic": robot_description_semantic,
                    }
                ],
                condition=IfCondition(start_rviz),
            ),
        ]
    )
