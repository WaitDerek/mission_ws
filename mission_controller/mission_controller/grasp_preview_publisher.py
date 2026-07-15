import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


LEFT_ARM_JOINTS = [
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
]
RIGHT_ARM_JOINTS = [
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
]
TORSO_JOINTS = [
    "torso_joint1",
    "torso_joint2",
    "torso_joint3",
    "torso_joint4",
]
AUXILIARY_JOINTS = [
    "steer_motor_joint1",
    "wheel_motor_joint1",
    "steer_motor_joint2",
    "wheel_motor_joint2",
    "steer_motor_joint3",
    "wheel_motor_joint3",
    "left_gripper_finger_joint1",
    "left_gripper_finger_joint2",
    "right_gripper_finger_joint1",
    "right_gripper_finger_joint2",
]


class GraspPreviewPublisher(Node):
    def __init__(self) -> None:
        super().__init__("grasp_preview_publisher")
        self.declare_parameter("pose_topic", "/mission/preview_grasp_pose")
        self.declare_parameter("joint_states_topic", "/mission/preview_joint_states")
        self.declare_parameter(
            "camera_frame", "hdas/camera_wrist_right_color_optical_frame"
        )
        self.declare_parameter(
            "grasp_position", [0.0977175608, -0.0004082107, 0.4510000341]
        )
        self.declare_parameter(
            "grasp_orientation_xyzw",
            [-0.2109386398, -0.4507636392, -0.3676279393, 0.7855995990],
        )
        self.declare_parameter(
            "left_arm_positions",
            [-0.88, 0.84, -1.13, -1.80, 1.25, 0.29, 0.13],
        )
        self.declare_parameter(
            "right_arm_positions",
            [-0.98, -0.64, 1.13, -1.60, -1.25, 0.6, -0.13],
        )
        self.declare_parameter("torso_positions", [0.61, -0.81, -0.21, 0.0])
        self.declare_parameter(
            "gripper_finger_positions", [0.04, -0.04, 0.04, -0.04]
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("pose_topic").value), qos
        )
        self.joint_publisher = self.create_publisher(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            qos,
        )
        self.joint_timer = self.create_timer(0.1, self._publish_joint_state)
        self.pose_timer = self.create_timer(1.0, self._publish_pose_once)
        self._publish_joint_state()

    def _publish_pose_once(self) -> None:
        stamp = self.get_clock().now().to_msg()
        position = [
            float(value) for value in self.get_parameter("grasp_position").value
        ]
        orientation = [
            float(value) for value in self.get_parameter("grasp_orientation_xyzw").value
        ]
        if len(position) != 3 or len(orientation) != 4:
            self.get_logger().error("invalid fixed grasp preview pose")
            return

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = str(self.get_parameter("camera_frame").value)
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = orientation

        self.pose_publisher.publish(pose)
        self.pose_timer.cancel()
        self.get_logger().info(
            "published fixed Graspness sample once; continuing initial joint states"
        )

    def _publish_joint_state(self) -> None:
        gripper_finger_positions = [
            float(value)
            for value in self.get_parameter("gripper_finger_positions").value
        ]
        if len(gripper_finger_positions) != 4:
            self.get_logger().error(
                "gripper_finger_positions must contain four joint values"
            )
            return
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = (
            TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + AUXILIARY_JOINTS
        )
        joint_state.position = (
            [float(value) for value in self.get_parameter("torso_positions").value]
            + [float(value) for value in self.get_parameter("left_arm_positions").value]
            + [
                float(value)
                for value in self.get_parameter("right_arm_positions").value
            ]
            + [0.0] * (len(AUXILIARY_JOINTS) - 4)
            + gripper_finger_positions
        )
        if len(joint_state.position) != len(joint_state.name):
            self.get_logger().error("invalid fixed grasp preview joint positions")
            return

        self.joint_publisher.publish(joint_state)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspPreviewPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
